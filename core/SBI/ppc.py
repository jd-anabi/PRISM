"""The posterior-predictive check's bin-by-bin simulation loop, split out of orchestrator.

Sorts the posterior draws by t_scale so each bin shares a similar fine-grid geometry, simulates
each bin (forced + spontaneous, or spontaneous + the chi block) into preallocated buffers by
index, and rides ``pipeline.retry_on_oom`` per bin. The conditioning assembly stays with the
caller: this module produces trajectories and the chi block, not conditioning rows.
"""
import math

import torch
from tqdm import tqdm

from core import forcing
from core.config import CHUNK_LEN, PPC_BIN_SIZE
from core.SBI import chi, pipeline
from core.SBI import clock as _clock


def simulate_ppc_bins(cfg, t, inits, samples_nd, samples_rescale, forcing_gt, N_points_obs,
                      expected_forcing_dim, dtype, device):
    """Simulate every posterior draw at the observation's geometry -> sorted trajectory buffers.

    :param samples_rescale: the SIM-side rescale block (tier-1 substitution already applied).
    :param expected_forcing_dim: the chi block's width for this config (passed in, not recomputed,
        so this module needs no orchestrator import).
    :return: ``(x_dim_sorted, x_spont_sorted, chi_block_sorted, inv_sort_idx)`` -- apply
        ``[inv_sort_idx]`` to recover the caller's draw order; ``chi_block_sorted`` is None
        outside chi mode, and ``x_dim_sorted`` is only meaningful under a forced config.
    """
    n_samples = samples_nd.shape[0]
    n_vars = inits.shape[-1]
    n_force_ch = forcing.n_force_channels(cfg.model, cfg.forcing_idx, n_vars)

    # Sort by t_scale (ascending) so each bin contains similar-scale samples
    t_scales_all = samples_rescale[:, cfg.rescale_idx["t_scale"]]
    sort_idx = torch.argsort(t_scales_all)
    inv_sort_idx = torch.argsort(sort_idx)
    samples_nd_sorted = samples_nd[sort_idx]
    samples_rescale_sorted = samples_rescale[sort_idx]

    x_dim_sorted = torch.empty((n_samples, N_points_obs), dtype=dtype, device=device)
    x_spont_sorted = torch.empty((n_samples, N_points_obs), dtype=dtype, device=device)
    n_bins = math.ceil(n_samples / PPC_BIN_SIZE)
    # chi-mode: per-sample chi(omega) block, filled bin-by-bin alongside the spontaneous run.
    chi_block_sorted = (torch.empty((n_samples, expected_forcing_dim),
                                    dtype=dtype, device=device) if cfg.chi_mode else None)

    with torch.no_grad():
        for b in tqdm(range(n_bins), desc="PPC simulations", leave=False):
            start = b * PPC_BIN_SIZE
            end = min(start + PPC_BIN_SIZE, n_samples)

            def _simulate_bin(start=start, end=end):
                """One bin's simulations, writing into the preallocated destination
                buffers by index -- so a re-run after an OOM is idempotent, which is what
                lets the whole body ride pipeline.retry_on_oom below. The bins had NO
                ladder before: the same failure the training loop survives was fatal here.
                """
                bs = end - start

                bin_nd = samples_nd_sorted[start:end]
                bin_rescale = samples_rescale_sorted[start:end]
                bin_t_scales = bin_rescale[:, cfg.rescale_idx["t_scale"]]

                # The bin's record clock: ONE stride, from its smallest t_scale, and each draw's own step,
                # so every draw is the cell at its own t_scale on the observation's frames (dt_exp apart)
                # with its drive's clock starting at the first frame (core/SBI/clock.py).
                clock_bin = _clock.rows_clock(cfg.dt_exp, bin_t_scales.double().cpu(), N_points_obs,
                                              cfg.dt_nd_min, cfg.steady_idx, CHUNK_LEN)
                t_fine_bin = clock_bin.grid(dtype, device)
                n_segs_bin = clock_bin.n_segments

                x_scale_col = bin_rescale[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
                x_offset_col = bin_rescale[:, cfg.rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in cfg.rescale_idx else 0.0

                # Forced run (Group G) then spontaneous run (Groups A-F); no-forcing / chi-mode = spontaneous
                # only (chi-mode's driven info is the separate K-freq chi block computed below).
                if cfg.has_forcing and not cfg.chi_mode:
                    force_bin = pipeline.build_nondim_sin_force_tensor(
                        forcing_gt.expand(bs, -1), None, bin_rescale, cfg.forcing_idx,
                        cfg.sim_rescale_idx, t_dim=clock_bin.drive_times(dtype, device))
                    run_specs = ((force_bin, x_dim_sorted), (torch.zeros_like(force_bin), x_spont_sorted))
                else:
                    force_bin = torch.zeros((bs, n_force_ch, clock_bin.n_fine), dtype=dtype, device=device)
                    run_specs = ((force_bin, x_spont_sorted),)
                for force_run, dest in run_specs:
                    x_nd_bin = pipeline.gen_obs(
                        model=cfg.model, params=bin_nd, t=t_fine_bin,
                        inits=inits.expand(bs, -1),
                        force=force_run, n_segs=n_segs_bin, steady_idx=clock_bin.n_transient,
                        state_dep_drift=cfg.state_dep_drift,
                        batch_size=bs, var_idx=0, dtype=dtype, device=device, clock=clock_bin,
                    )[0, :, :]  # (bs, n_fine_bin - steady_idx)
                    x_nd_ds = clock_bin.frames(x_nd_bin)                  # (bs, N_points_obs)
                    dest[start:end] = x_scale_col * x_nd_ds + x_offset_col
                    del x_nd_bin, x_nd_ds

                del force_bin
                if cfg.chi_mode:
                    # Single-tone probes for this bin, on the bin's clock (each draw at its own t_scale).
                    #
                    # Driven at the OBSERVATION'S ABSOLUTE FREQUENCIES, not at each sample's own
                    # mult_k*f_peak. The experiment fixed those frequencies; a PPC that re-derives them
                    # per posterior sample simulates a DIFFERENT experiment for every sample, and its chi
                    # z-scores then come out small for the wrong reason.
                    obs_freqs = getattr(cfg, "chi_obs_freqs", None)
                    if obs_freqs is None:
                        probe, absolute = chi.chi_multipliers_for(cfg, dtype=dtype, device=device), False
                    else:
                        probe, absolute = obs_freqs.to(device=device, dtype=dtype), True
                    chi_block_sorted[start:end] = pipeline.gen_chi_block(
                        cfg.model, bin_nd, bin_rescale, x_spont_sorted[start:end],
                        inits=inits.expand(bs, -1), rescale_idx=cfg.sim_rescale_idx,
                        multipliers=probe, f0_nd=cfg.chi_f0, k_pad=cfg.chi_k_pad, bounds=cfg.chi_freq_bounds,
                        absolute_freqs=absolute, max_cycles=cfg.chi_max_cycles,
                        state_dep_drift=cfg.state_dep_drift, dtype=dtype, device=device, clock=clock_bin)[0]

            pipeline.retry_on_oom(_simulate_bin, what=f"PPC bin {b + 1}/{n_bins}",
                                  device=device)
            # Guarded, and hygiene-only: a PPC bin that succeeded is not short of memory, so the
            # cuFFT plans and the captured graphs are worth keeping. An empty_cache() that raises
            # must not take the run down -- see pipeline._release_device_memory (2026-08-27).
            pipeline._release_device_memory(device, plans=False, graphs=False)

    return x_dim_sorted, x_spont_sorted, chi_block_sorted, inv_sort_idx
