"""
Sanity checks for the FDT analysis pipeline.

Each check runs a mini measurement and asserts a numerical property that would
catch a common bug class. Run them BEFORE the production sweep.
"""
import math
import numpy as np
import torch

from core.config import FDTConfig
from core.FDT.campaigns import (
    run_campaign1_psd, run_campaign2_chi,
    _get_simulator_cls, _n_force_channels, _pick_n_segs,
)
from core.FDT.spectral import psd_welch, lock_in_chi, eff_temp_ratio, gen_freqs_log


def _interp_log(x_new: torch.Tensor, x_old: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation in log-x. Aligns PSD onto the chi frequency grid."""
    x_new = x_new.to(torch.float64)
    x_old = x_old.to(torch.float64).clamp(min=1e-30)
    y_old = y_old.to(torch.float64)
    log_x_old = torch.log(x_old)
    log_x_new = torch.log(x_new.clamp(min=1e-30))
    idx = torch.searchsorted(log_x_old, log_x_new).clamp(1, len(log_x_old) - 1)
    x0, x1 = log_x_old[idx - 1], log_x_old[idx]
    y0, y1 = y_old[idx - 1], y_old[idx]
    frac = (log_x_new - x0) / (x1 - x0)
    return y0 + frac * (y1 - y0)


def check_passive_baseline(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    Pipeline with temp=1.0, tau_c=0.0, s=0.0: true thermal-equilibrium baseline.
    Expect T_eff/T ~ 1 across the full grid.

    Setting s=0 removes the calcium -> motor feedback (f_c = f_max*(1 - s*c)).
    Without this feedback, the (x, y) drift has a Boltzmann potential and FDT holds.
    Temp=1 makes the motor noise thermal; tau_c=0 zeros the channel-gating noise.
    This is the convention-correctness gate -- PSD/lock-in/noise prefactor bugs
    all show up as a uniform deviation from 1.0.

    Pass criterion: median(|ratio - 1|) < 0.2 AND max(|ratio - 1|) < 0.4.
    """
    M = min(128, cfg.ensemble_M)
    passive = cfg.with_overrides(temp=1.0, tau_c=0.0, s=0.0, ensemble_M=M,
                                  psd_T_obs_nd=min(4000.0, cfg.psd_T_obs_nd))

    omegas = gen_freqs_log(cfg.omega_0, 7, cfg.freq_bounds, cfg.hw.device, cfg.hw.dtype)

    freqs_psd, G = run_campaign1_psd(passive)
    chis = run_campaign2_chi(passive, omegas, show_progress=False)

    G_at_om = _interp_log(omegas, freqs_psd, G)
    n, beta = passive.params_dict["n"][0], passive.params_dict["beta"][0]
    ratio = eff_temp_ratio(G_at_om, chis.imag, omegas.to(torch.float64), n, beta).cpu().numpy()

    devs = np.abs(ratio - 1.0)
    med_dev, max_dev = float(np.median(devs)), float(np.max(devs))
    passed = (med_dev < 0.2) and (max_dev < 0.4)
    return passed, {"median_dev": med_dev, "max_dev": max_dev,
                    "ratios": ratio.tolist(), "omegas": omegas.cpu().tolist()}


def check_high_freq_fdt(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    Passive-noise baseline (temp=1, tau_c=0) but s != 0, evaluated only at the
    top three frequencies of the production grid.

    At omega >> 1/tau_c, calcium can't follow the bundle, so the active feedback
    loop is dynamically broken and FDT recovers. This validates that the calcium
    adaptation timescale is the locus of FDT violation in the Nadrowski model
    (as opposed to some other non-equilibrium source).

    Pass criterion: median(|ratio - 1|) < 0.15 AND max(|ratio - 1|) < 0.3.
    """
    M = min(128, cfg.ensemble_M)
    high_freq_cfg = cfg.with_overrides(temp=1.0, tau_c=0.0, ensemble_M=M,
                                        psd_T_obs_nd=min(4000.0, cfg.psd_T_obs_nd))

    all_omegas = gen_freqs_log(cfg.omega_0, 7, cfg.freq_bounds,
                                cfg.hw.device, cfg.hw.dtype)
    omegas = all_omegas[-3:]   # top 3 freqs of the 7-point check grid

    freqs_psd, G = run_campaign1_psd(high_freq_cfg)
    chis = run_campaign2_chi(high_freq_cfg, omegas, show_progress=False)

    G_at_om = _interp_log(omegas, freqs_psd, G)
    n, beta = high_freq_cfg.params_dict["n"][0], high_freq_cfg.params_dict["beta"][0]
    ratio = eff_temp_ratio(G_at_om, chis.imag, omegas.to(torch.float64), n, beta).cpu().numpy()

    devs = np.abs(ratio - 1.0)
    med_dev, max_dev = float(np.median(devs)), float(np.max(devs))
    passed = (med_dev < 0.15) and (max_dev < 0.3)
    return passed, {"median_dev": med_dev, "max_dev": max_dev,
                    "ratios": ratio.tolist(), "omegas": omegas.cpu().tolist()}


def check_linearity(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    Campaign-2 at omega_0 with F0, F0/2, F0/4: |chi| must be amplitude-invariant within 5%.
    Failure means the drive is outside the linear regime; lower F0.
    """
    omega = torch.tensor([cfg.omega_0], dtype=cfg.hw.dtype, device=cfg.hw.device)
    F0s = [cfg.F0, cfg.F0 / 2.0, cfg.F0 / 4.0]
    mags = []
    for F0 in F0s:
        chi = run_campaign2_chi(cfg, omega, F0=F0, show_progress=False)[0]
        mags.append(float(chi.abs()))
    mags_t = torch.tensor(mags)
    rel_spread = float(mags_t.std(unbiased=False) / mags_t.mean())
    passed = rel_spread < 0.05
    return passed, {"F0_values": F0s, "|chi|_values": mags, "rel_spread": rel_spread}


def check_ensemble_convergence(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    At omega_0, recompute chi'' from the first {32, 64, 128, 256} trajectories
    of a single M=256 simulation. Pass if |chi''_256 - chi''_128|/|chi''_256| < 0.1.

    Runs the simulator once; slices the (M, n_steady) trajectory tensor by prefix.
    """
    omega = cfg.omega_0
    M_max = 256
    dt, burn = cfg.dt_nd, cfg.burn_in_nd
    device, dtype = cfg.hw.device, cfg.hw.dtype

    burn_idx = int(round(burn / dt))
    n_obs = int(round(cfg.T_obs_periods * 2.0 * math.pi / omega / dt))
    n_steps = burn_idx + n_obs
    t = torch.arange(n_steps, dtype=dtype, device=device) * dt
    n_force = _n_force_channels(cfg)
    force = torch.zeros((M_max, n_force, n_steps), dtype=dtype, device=device)
    force[:, 0, :] = cfg.F0 * torch.cos(omega * t)

    inits = cfg.inits_for_M(M_max)
    params = cfg.params_for_M(M_max)
    n_segs = _pick_n_segs(n_steps, M_max)
    sim_cls = _get_simulator_cls(cfg.model)
    sim = sim_cls(params, force, inits, t,
                   freqs_per_batch=1, segs=n_segs, batch_size=M_max,
                   device=device)
    sol = sim.simulate(state_dep_drift=cfg.state_dep_drift)
    x_steady = sol[0, 0, :, burn_idx:]    # (M_max, n_obs)
    t_steady = t[burn_idx:]
    T_obs_used = n_obs * dt

    chi_imags = {}
    for M in (32, 64, 128, 256):
        x_mean = x_steady[:M].mean(dim=0)
        chi = lock_in_chi(t_steady, x_mean, omega, cfg.F0, T_obs_used)
        chi_imags[M] = float(chi.imag)
    rel_change = abs(chi_imags[256] - chi_imags[128]) / max(abs(chi_imags[256]), 1e-30)
    passed = rel_change < 0.1
    return passed, {"chi_imag_by_M": chi_imags, "rel_change_128_256": rel_change}


def check_psd_window(cfg: FDTConfig) -> tuple[bool, dict]:
    """
    Compute Campaign-1; split steady-state into halves; PSD each; compare at 5 freqs.
    Pass if max relative diff < 0.25.
    """
    M = min(128, cfg.ensemble_M)
    half_cfg = cfg.with_overrides(ensemble_M=M, psd_T_obs_nd=min(4000.0, cfg.psd_T_obs_nd))
    dt, burn = half_cfg.dt_nd, half_cfg.burn_in_nd
    device, dtype = half_cfg.hw.device, half_cfg.hw.dtype

    burn_idx = int(round(burn / dt))
    n_obs = int(round(half_cfg.psd_T_obs_nd / dt))
    n_steps = burn_idx + n_obs
    t = torch.arange(n_steps, dtype=dtype, device=device) * dt
    n_force = _n_force_channels(half_cfg)
    force = torch.zeros((M, n_force, n_steps), dtype=dtype, device=device)
    inits = half_cfg.inits_for_M(M)
    params = half_cfg.params_for_M(M)

    n_segs = _pick_n_segs(n_steps, M)
    sim_cls = _get_simulator_cls(half_cfg.model)
    sim = sim_cls(params, force, inits, t,
                   freqs_per_batch=1, segs=n_segs, batch_size=M, device=device)
    sol = sim.simulate(state_dep_drift=half_cfg.state_dep_drift)
    x_steady = sol[0, 0, :, burn_idx:]
    n_half = n_obs // 2
    x_first, x_second = x_steady[:, :n_half], x_steady[:, n_half:2 * n_half]

    nperseg = min(2 ** 13, n_half)
    nperseg = 1 << int(math.log2(nperseg))

    omegas1, G1 = psd_welch(x_first, dt=dt, nperseg=nperseg)
    _, G2 = psd_welch(x_second, dt=dt, nperseg=nperseg)

    n_freqs = G1.shape[-1]
    sample_idx = torch.linspace(1, n_freqs - 1, 5).long()
    rel_diff = ((G1[sample_idx] - G2[sample_idx]).abs() / (G1[sample_idx].abs() + 1e-30)).cpu().numpy()
    max_rel = float(rel_diff.max())
    passed = max_rel < 0.25
    return passed, {"max_rel_diff": max_rel, "rel_diffs": rel_diff.tolist(),
                    "omegas_sampled": omegas1[sample_idx].cpu().tolist()}


def run_all_sanity(cfg: FDTConfig) -> dict:
    """Run all four checks; print summary; return dict[name -> (passed, metrics)]."""
    print("\n" + "=" * 60)
    print("FDT Sanity Checks")
    print("=" * 60)

    checks = [
        ("passive_baseline",      check_passive_baseline,      "true equilibrium (s=0): T_eff/T ~ 1"),
        ("high_freq_fdt",         check_high_freq_fdt,         "high omega (s != 0): T_eff/T ~ 1"),
        ("linearity",             check_linearity,             "|chi(omega_0)| F0-invariant"),
        ("ensemble_convergence",  check_ensemble_convergence,  "chi'' stable by M=256"),
        ("psd_window",            check_psd_window,            "PSD halves agree"),
    ]
    results = {}
    for name, fn, desc in checks:
        print(f"\n[{name}] {desc}")
        passed, metrics = fn(cfg)
        results[name] = (passed, metrics)
        print(f"  {'PASS' if passed else 'FAIL'}  metrics: {metrics}")

    print("\n" + "=" * 60 + "\nSummary:")
    for name, (passed, _) in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("=" * 60 + "\n")
    return results
