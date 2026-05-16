"""
Simulation campaigns for FDT analysis.

Campaign 1: spontaneous fluctuations -> Welch PSD G(omega).
Campaign 2: forced response at each driving frequency -> chi(omega) via lock-in.
"""
import math
import torch
from tqdm import tqdm

from core.config import FDTConfig
from core.Simulator.nadrowski_simulator import NadrowskiSimulator
from core.Simulator.hopf_simulator import HopfSimulator
from core.Simulator.bp_simulator import BPSimulator
from core.FDT.spectral import psd_welch, lock_in_chi

# Model -> simulator class. Matches core/SBI/pipeline.py:VALID_SIMS so FDT and SBI
# stay consistent.
VALID_SIMS = {
    "nadrowski": NadrowskiSimulator,
    "hopf":      HopfSimulator,
    "bp":        BPSimulator,
}

# Per-segment element budget for the solver's xs buffer. Sized to keep the per-segment
# allocation under ~800 MB at float32, matching the SBI side's CHUNK_LEN x reference
# batch (~100k steps x 2048 batch). The simulator's full `sol` tensor lives outside
# this budget; segs > 1 only shrinks the per-segment xs buffer.
FDT_MAX_ELEMENTS_PER_SEG = 200_000_000

# Fraction of currently-free CUDA memory to allocate as the per-batch budget. The
# remaining headroom absorbs PyTorch internals, fragmentation, and intermediate
# tensors during integration. 0.6 leaves ~40% buffer.
FDT_CUDA_MEM_FRACTION = 0.6


def _pick_n_segs(n_steps: int, batch_size: int) -> int:
    """Pick segs to keep the solver's per-segment xs buffer under the element budget."""
    max_steps_per_seg = max(1, FDT_MAX_ELEMENTS_PER_SEG // batch_size)
    return max(1, math.ceil(n_steps / max_steps_per_seg))


def _memory_budget_elements(device: torch.device, dtype: torch.dtype) -> int:
    """
    Per-batch element budget for the simulator's major device-resident tensors
    (sol + force). On CUDA, uses a fraction of currently-free GPU memory so we
    adapt to whatever's actually available. On CPU/MPS, defaults conservatively.
    """
    bytes_per_elem = 4 if dtype == torch.float32 else 8
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        budget_bytes = int(free_bytes * FDT_CUDA_MEM_FRACTION)
    elif device.type == "cpu":
        budget_bytes = 4 * 1024 ** 3   # 4 GB conservative cap for CPU
    else:
        budget_bytes = 1 * 1024 ** 3   # 1 GB for MPS / other
    return max(1, budget_bytes // bytes_per_elem)


def _plan_adaptive_batches(omegas_list: list, fpb_max: int, M: int, n_vars: int,
                            n_force: int, dt: float, burn_idx: int, n_periods: int,
                            budget_elements: int) -> list:
    """
    Walk omegas left-to-right, greedily packing as many consecutive frequencies into
    each batch as the memory budget allows (up to fpb_max).

    The per-group n_steps is determined by the LOWEST omega in the group (since that
    one needs the longest t-grid). So memory cost = (n_vars + n_force) * fpb * M * n_steps.

    :return: list of (start, count) tuples.
    """
    batches = []
    start = 0
    n = len(omegas_list)
    while start < n:
        omega_low = omegas_list[start]
        T_target = n_periods * 2.0 * math.pi / omega_low
        n_obs = int(round(T_target / dt))
        n_steps = burn_idx + n_obs

        elements_per_fpb_unit = (n_vars + n_force) * M * n_steps
        max_fpb_by_memory = max(1, budget_elements // elements_per_fpb_unit)
        actual_fpb = min(fpb_max, n - start, max_fpb_by_memory)
        batches.append((start, actual_fpb))
        start += actual_fpb
    return batches


def _get_simulator_cls(model: str):
    """Look up the simulator class for the model name. Case-insensitive."""
    cls = VALID_SIMS.get(model.lower())
    if cls is None:
        raise ValueError(f"Invalid model for FDT: {model}. Valid: {list(VALID_SIMS.keys())}")
    return cls


def _n_force_channels(cfg: FDTConfig) -> int:
    """Number of forcing channels the model expects: 2 if 'amp_y' is in force params
    (Hopf dual-channel), else 1. Matches the convention in core/SBI/pipeline.py:
    build_nondim_sin_force_tensor."""
    return 2 if "amp_y" in cfg.force_params_dict else 1


def run_campaign1_psd(cfg: FDTConfig, M: int = None, T_obs_nd: float = None,
                      nperseg: int = None, return_trajectory: bool = False):
    """
    Spontaneous-fluctuation PSD G(omega) of bundle deflection x.

    :param cfg: FDTConfig (defaults below pulled from cfg unless overridden).
    :param M: ensemble size; default cfg.ensemble_M.
    :param T_obs_nd: PSD observation duration in ND time; default cfg.psd_T_obs_nd.
    :param nperseg: Welch segment length; default min(2**14, steady-state length)
                    rounded down to nearest power of 2.
    :param return_trajectory: if True, also returns the full time axis and the
                              ensemble-mean trajectory (including burn-in) for
                              diagnostic plotting.
    :return: (omegas, G) if return_trajectory=False; otherwise
             (omegas, G, t_full, x_mean_full).
    """
    M = M if M is not None else cfg.ensemble_M
    T_obs_nd = T_obs_nd if T_obs_nd is not None else cfg.psd_T_obs_nd
    dt, burn = cfg.dt_nd, cfg.burn_in_nd
    device, dtype = cfg.hw.device, cfg.hw.dtype

    burn_idx = int(round(burn / dt))
    n_obs = int(round(T_obs_nd / dt))
    n_steps = burn_idx + n_obs
    t = torch.arange(n_steps, dtype=dtype, device=device) * dt

    n_force = _n_force_channels(cfg)
    force = torch.zeros((M, n_force, n_steps), dtype=dtype, device=device)
    inits = cfg.inits_for_M(M)
    params = cfg.params_for_M(M)

    n_segs = _pick_n_segs(n_steps, M)
    sim_cls = _get_simulator_cls(cfg.model)
    sim = sim_cls(params, force, inits, t,
                   freqs_per_batch=1, segs=n_segs, batch_size=M,
                   device=device)
    sol = sim.simulate(state_dep_drift=cfg.state_dep_drift)  # (n_vars, 1, M, n_steps)
    x_full = sol[0, 0, :, :]            # (M, n_steps) -- full trajectory including burn-in
    x_steady = x_full[:, burn_idx:]      # (M, n_obs)   -- post-burn-in for PSD

    if nperseg is None:
        nperseg = min(2 ** 14, x_steady.shape[-1])
        nperseg = 1 << int(math.log2(nperseg))  # nearest power of 2 <= nperseg

    omegas, G = psd_welch(x_steady, dt=dt, nperseg=nperseg)

    if return_trajectory:
        x_mean_full = x_full.mean(dim=0)  # (n_steps,) -- ensemble-mean trajectory
        return omegas, G, t, x_mean_full
    return omegas, G


def run_campaign2_chi(cfg: FDTConfig, omegas: torch.Tensor, M: int = None,
                      F0: float = None, freqs_per_batch: int = None,
                      show_progress: bool = True) -> torch.Tensor:
    """
    Forced-response chi(omega) by lock-in detection.

    Packs `freqs_per_batch` consecutive frequencies into each simulator call by
    using the simulator's native (n_vars, freqs_per_batch, ensemble_size, T) layout.
    Within a group, all frequencies share a single t-grid of length matching the
    longest required T_obs in the group; per-frequency lock-in then slices its own
    integer-period window from the result so leakage stays minimized.

    Trade-off: log-spaced freqs grouped by fpb=8 waste ~30% compute on the high-omega
    end of each group (their T_obs is smaller than the group max). This is dominated
    by the GPU-parallelism win when fpb > 1.

    Per group:
      - T_obs_target_k = T_obs_periods * 2*pi / omega_k for each k in group
      - n_obs_k        = round(T_obs_target_k / dt_nd)
      - n_steps_group  = burn_idx + max(n_obs_k)
      - force[k*M:(k+1)*M, 0, :] = F0 * cos(omega_k * t)
      - simulate, then per-k slice sol[0, k, :, burn_idx:burn_idx+n_obs_k].mean(dim=0)

    :param cfg: FDTConfig.
    :param omegas: (n_freqs,) driving angular frequencies.
    :param M: ensemble size; default cfg.ensemble_M.
    :param F0: forcing amplitude; default cfg.F0.
    :param freqs_per_batch: # of frequencies per simulator call; default cfg.freqs_per_batch.
    :param show_progress: tqdm bar across batch groups.
    :return: (n_freqs,) complex tensor of chi(omega_k).
    """
    M = M if M is not None else cfg.ensemble_M
    F0 = F0 if F0 is not None else cfg.F0
    fpb_max = freqs_per_batch if freqs_per_batch is not None else cfg.freqs_per_batch
    dt, burn = cfg.dt_nd, cfg.burn_in_nd
    n_periods = cfg.T_obs_periods
    device, dtype = cfg.hw.device, cfg.hw.dtype

    burn_idx = int(round(burn / dt))
    n_freqs = len(omegas)
    chis = torch.zeros(n_freqs, dtype=torch.complex128, device=device)

    # Plan batches adaptively so each fits in the memory budget. Low-omega groups
    # (long t-grid) may shrink to fpb=1; high-omega groups will use the requested fpb_max.
    omegas_list = omegas.tolist()
    n_force = _n_force_channels(cfg)
    n_vars = cfg.inits_tensor.shape[1]
    budget_elements = _memory_budget_elements(device, dtype)
    batches = _plan_adaptive_batches(omegas_list, fpb_max, M, n_vars, n_force,
                                       dt, burn_idx, n_periods, budget_elements)

    reduced = [(s, c) for s, c in batches if c < fpb_max]
    if reduced:
        print(f"Adaptive batching: {len(reduced)}/{len(batches)} groups capped below fpb_max={fpb_max} "
              f"due to memory budget. Smallest group: fpb={min(c for _, c in batches)}.")

    iterator = batches
    if show_progress:
        iterator = tqdm(batches, desc=f"Campaign 2 (chi sweep, fpb<={fpb_max})")

    for start, actual_fpb in iterator:
        omegas_batch = omegas_list[start:start + actual_fpb]

        # Per-frequency integer-period window length
        n_obs_per_freq = [int(round(n_periods * 2.0 * math.pi / w / dt)) for w in omegas_batch]
        n_obs_max = max(n_obs_per_freq)
        n_steps = burn_idx + n_obs_max

        t = torch.arange(n_steps, dtype=dtype, device=device) * dt
        batch_size = actual_fpb * M
        force = torch.zeros((batch_size, n_force, n_steps), dtype=dtype, device=device)
        for k, omega in enumerate(omegas_batch):
            # Drive only channel 0 (bundle position); other channels stay zero
            force[k * M:(k + 1) * M, 0, :] = F0 * torch.cos(omega * t)

        inits = cfg.inits_tensor.expand(batch_size, -1).contiguous()
        params = cfg.params_tensor.expand(batch_size, -1).contiguous()

        n_segs = _pick_n_segs(n_steps, batch_size)
        sim_cls = _get_simulator_cls(cfg.model)
        sim = sim_cls(params, force, inits, t,
                      freqs_per_batch=actual_fpb, segs=n_segs,
                      batch_size=batch_size, device=device)
        sol = sim.simulate(state_dep_drift=cfg.state_dep_drift)  # (n_vars, actual_fpb, M, n_steps)

        for k, omega in enumerate(omegas_batch):
            n_obs_k = n_obs_per_freq[k]
            x_mean = sol[0, k, :, burn_idx : burn_idx + n_obs_k].mean(dim=0)   # (n_obs_k,)
            t_slice = t[burn_idx : burn_idx + n_obs_k]
            T_obs_used = n_obs_k * dt
            chis[start + k] = lock_in_chi(t_slice, x_mean, omega, F0, T_obs_used)

        # Free fragmented allocations between batches; sol/force/t go out of scope here.
        del sol, force, t, sim, inits, params
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return chis
