"""SBC repeated K times on one posterior: the run-to-run DISTRIBUTION of the per-parameter KS
p-values, optionally stratified by chi probe count.

Folded from ``scripts/sbc_characterize.py`` (piece 2). What the script could not do and this does:
the calibration set is drawn through ``validate_calibration``'s own helpers, so the draw carries
``check_basis``, the region restriction (guardrail 8), the t_scale-override mirror in check_sbc's
reference sample, and the kept-fraction report. A single flat SBC can be a lucky draw; K repeats at a
raised n_cal separate sampling noise (high, variable KS p) from real miscalibration (KS p stays low).

``chi_k_fixed`` holds the probe COUNT instead of pooling over the training mixture. A pooled SBC over
a mixture of counts can be flat while each count is miscalibrated in compensating directions, so the
stratified protocol is a four-run comparison (2 / 6 / CHI_K_PAD, then pooled) -- each run is its own
artifact now, which is what the script's ``_pooled`` / ``_k<N>`` filename suffixes were standing in for.
"""
from __future__ import annotations

import numpy as np
import torch

from core import orchestrator as orch
from core.artifacts import resolve_store
from core.Helpers import file_manager
from core.runs import public_entry

from .rng import seeded


def _col(values) -> tuple:
    """(median, min, frac below 0.05) over a column's FINITE entries; NaNs when it has none.

    np.nanmedian over an all-NaN column warns and returns NaN, and a repeat whose calibration set
    came back empty legitimately produces one -- so the finite subset is taken explicitly rather than
    suppressing the warning.
    """
    col = np.asarray(values, dtype=float)
    col = col[np.isfinite(col)]
    if col.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(col)), float(col.min()), float((col < 0.05).mean())


def _rank_hist_bins(n_pooled: int, nps: int) -> int:
    """The bin count for the pooled rank histogram over ``n_pooled`` ranks in 0..nps.

    ``cap = max(1, min(n_pooled // 20, (nps + 1) // 10))``. sbc_rank_plot's own default is
    num_sbc_runs // 20, which is 0 for a small pooled set (matplotlib then refuses outright), hence the
    floor of 1. The second term keeps every bin at least ~10 of the nps + 1 integer ranks wide: the
    POOLED N is repeats x n_cal, so N // 20 alone gives ~950 bins over 1001 ranks at the defaults, some
    bins hold two ranks and spike above the band on a calibrated posterior, and past N = 20 (nps + 1)
    every other bin is empty.

    The count is then the LARGEST DIVISOR d of nps + 1 with ``cap // 2 <= d <= cap`` when one exists,
    and ``cap`` itself otherwise. sbi draws the panel with ``plt.hist(ranks, bins=<int>)``, whose edges
    are linspace(min, max, bins + 1) with the last bin closed, so over ranks 0..nps every bin holds the
    same number of integer ranks only when the count divides nps + 1. At the defaults the cap alone is
    100 bins, edges on multiples of 10, and a closed last bin [990, 1000] holding 11 ranks against 10
    elsewhere -- a spike above the band in ~14 % of panels on a calibrated posterior; 91 bins hold 11
    ranks each. Only divisors down to cap // 2 are taken: a pure largest-divisor rule collapses to 1 bin
    when nps + 1 is prime (nps = 40, 100) and to 3 bins at nps = 500, which is no histogram at all.
    Without such a divisor the cap stands, and the one last bin is up to one rank wider than the rest.
    The equal count also assumes a panel's ranks reach both 0 and nps, which a pooled calibrated set
    does with near certainty; a panel that misses an end is miscalibrated enough to show it regardless.
    """
    span = int(nps) + 1
    cap = max(1, min(int(n_pooled) // 20, span // 10))
    divisors = [d for d in range(max(1, cap // 2), cap + 1) if span % d == 0]
    return max(divisors) if divisors else cap


def _cell(v, spec: str) -> str:
    """One KS-table cell: the NUMBER formatted, or "-" (a column with no finite entry) right-justified
    to the same width. Formatting the number and not str(v) matters: a string cut to 8 characters
    prints 3.212345646893978e-20 as 3.212345, on exactly the rows the table sorts to the top."""
    width = int(spec.split(".")[0])
    return f"{'-':>{width}s}" if v is None else f"{v:{spec}}"


@public_entry
def sbc_repeats(cfg, posterior, prior, *, repeats: int = 10, n_cal: int = 2000,
                num_posterior_samples: int = 1000, cal_n_scales: int | None = None,
                chi_k_fixed: int | None = None, seed: int = 0,
                name: str = "", note: str = "", fig_sink=None, store=None):
    """SBC run ``repeats`` times on one posterior; writes a ``diagnostic`` artifact.

    :param posterior / prior: the LoadedPosterior and the LoadedPrior it was trained from. The
                     proposal is the posterior's own region when it has one (guardrail 8), exactly as
                     in validate_calibration -- these helpers are shared with it.
    :param repeats: independent SBC runs, each seeded ``seed + r``.
    :param n_cal: calibration datasets PER REPEAT. The KS test's power grows with it; below ~2000 a
                     mild marginal miscalibration does not surface reliably.
    :param chi_k_fixed: hold the chi probe count at this value instead of pooling over the training
                     mixture. Refused outside chi mode; its range is gen_training_data's own check.
    :param seed: base seed. Repeat r runs inside ``seeded(seed + r, cfg.hw.device)``, which restores
                     the caller's RNG afterwards.
    :param fig_sink: (title, fig) -> None; the writer's sink saves the PNG and forwards to this one.

    ⚠ FOR A TSNPE POSTERIOR, reproducibility is WHOLE-RUN, not per-repeat. ``val_latent_prior`` (the
    region-restricted proposal built once below, before the loop) is ONE ``TruncatedLatentPrior``
    shared by every repeat, and its rejection sampler sizes each draw's over-draw from the acceptance
    rate ACCUMULATED over every repeat run on it so far (core/SBI/truncate.py:434-436), not from that
    repeat alone. A full run at one seed reproduces bit-for-bit -- each repeat still reseeds at
    ``seed + r`` -- but repeat r's own draw is sized differently when it is the FIRST repeat of a
    fresh run (``--repeats 1 --seed r``) than when it is repeat r of a longer ``--seed 0`` run: the
    earlier repeats there have already fed the shared prior's running acceptance estimate. So
    ``--repeats 1 --seed r`` does NOT reproduce repeat r pulled out of a ``--repeats K --seed 0`` run.
    """
    store = resolve_store(store)
    store.assert_name_free("diagnostic", name)            # before K x n_cal simulations
    repeats, n_cal = int(repeats), int(n_cal)
    nps = int(num_posterior_samples)
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if n_cal < 1:
        raise ValueError(f"n_cal must be at least 1, got {n_cal}")
    if nps < 1:
        raise ValueError(f"num_posterior_samples must be at least 1, got {nps}")
    if chi_k_fixed is not None and not cfg.chi_mode:
        raise ValueError(
            f"chi_k_fixed only means something in chi(omega) mode; this config is "
            f"{cfg.observation_mode.upper()}. Drop --chi-k-fixed, or point the run at a chi posterior.")
    orch._assert_prior_used_matches_posterior(posterior.posterior, prior.prior, "SBC")

    labels = list(cfg.params_dict) + list(cfg.rescale_params)
    inferred_prior, force_prior = prior.prior, prior.force_prior
    # ONCE, and before create(): it can refuse (check_basis on a region measured in another basis),
    # and a refusal must not leave a directory behind. Its "PRIOR RESTRICTED" line is guardrail 8's
    # announcement and belongs on screen either way.
    val_latent_prior, T, truncation = orch._calibration_prior(cfg, posterior, prior)
    stratum = "pooled" if chi_k_fixed is None else f"k{int(chi_k_fixed)}"
    print(f"[sbc] probe-count stratum: "
          f"{'POOLED over the training mixture' if chi_k_fixed is None else f'FIXED K = {int(chi_k_fixed)}'}",
          flush=True)

    settings = {"repeats": repeats, "n_cal": n_cal, "num_posterior_samples": nps,
                "cal_n_scales": None if cal_n_scales is None else int(cal_n_scales),
                "chi_k_fixed": None if chi_k_fixed is None else int(chi_k_fixed), "seed": int(seed)}
    ks = np.full((repeats, len(labels)), np.nan)
    c2st_ranks = np.full((repeats, len(labels)), np.nan)
    c2st_dap = np.full((repeats, len(labels)), np.nan)
    ranks_all, repeat_col, n_valid = [], [], []

    with store.create("diagnostic", cfg, name=name, note=note) as w:
        for r in range(repeats):
            with seeded(int(seed) + r, cfg.hw.device):
                x_cal, theta_star = orch._draw_calibration_set(
                    cfg, val_latent_prior, T, force_prior,
                    n_cal=n_cal, cal_n_scales=cal_n_scales, chi_k_fixed=chi_k_fixed)
                rk, dap = orch.run_sbc(
                    thetas=theta_star.to(cfg.hw.device), xs=x_cal.to(cfg.hw.device),
                    posterior=posterior.posterior, num_posterior_samples=nps,
                    reduce_fns="marginals", use_batched_sampling=True, show_progress_bar=False)
                reference = orch._sbc_reference_sample(cfg, val_latent_prior, T, truncation,
                                                       inferred_prior, theta_star)
                stats = orch.check_sbc(ranks=rk.cpu(), prior_samples=reference,
                                       dap_samples=dap.cpu(), num_posterior_samples=nps)
            ks[r] = np.asarray(stats["ks_pvals"], dtype=float)
            c2st_ranks[r] = np.asarray(stats["c2st_ranks"], dtype=float)
            c2st_dap[r] = np.asarray(stats["c2st_dap"], dtype=float)
            rk_np = rk.cpu().numpy()
            ranks_all.append(rk_np)
            repeat_col.append(np.full(rk_np.shape[0], r, dtype=np.int64))
            n_valid.append(int(theta_star.shape[0]))
            # Guarded exactly as _col guards its own nanmedian/min: an all-NaN row (a repeat whose
            # calibration set came back empty) must print "nan", not warn "All-NaN slice" from a bare
            # np.nanmin.
            worst = _col(ks[r])[1]
            print(f"[sbc] repeat {r + 1}/{repeats}: n_valid={n_valid[-1]}  "
                  f"worst KS p={worst:.4f}", flush=True)

        pooled = np.concatenate(ranks_all, axis=0)
        print("\n=== KS p-value distribution over repeats (sorted by median; low = miscalibrated) ===")
        print(f"{'param':16s} {'median':>8s} {'min':>8s} {'frac<.05':>9s}")
        per_param = []
        for j, key in enumerate(labels):
            med, lo, frac = _col(ks[:, j])
            per_param.append({"name": key, "ks_p_median": orch._num(med), "ks_p_min": orch._num(lo),
                              "frac_ks_below_05": orch._num(frac),
                              "c2st_ranks_median": orch._num(_col(c2st_ranks[:, j])[0])})
        for rec in sorted(per_param, key=lambda p: (p["ks_p_median"] is None, p["ks_p_median"])):
            print(f"{rec['name']:16s} {_cell(rec['ks_p_median'], '8.2e')} {_cell(rec['ks_p_min'], '8.2e')} "
                  f"{_cell(rec['frac_ks_below_05'], '9.3f')}")

        n_rows = int(np.ceil(len(labels) / 4))
        num_bins = _rank_hist_bins(pooled.shape[0], nps)
        fig, _ = orch.sbc_rank_plot(ranks=torch.as_tensor(pooled), num_posterior_samples=nps,
                                    plot_type="hist", num_bins=num_bins,
                                    parameter_labels=labels, figsize=(16, 3.4 * n_rows))
        fig.subplots_adjust(hspace=0.75, wspace=0.3)
        w.fig_sink(fig_sink)("SBC ranks pooled over repeats (histogram)", fig)

        kept = None
        if truncation is not None:
            # The SHAPE matches the results["kept_fraction"] key validate_calibration already writes
            # on a calibration artifact: both numbers under the one key, so `kept_fraction` does not
            # mean two different things depending on which artifact kind you read it from.
            kept = {"acceptance": orch._num(val_latent_prior.acceptance_rate),
                    "containment": orch._num(val_latent_prior.recorded_containment)}
            print(f"[tsnpe] kept fraction: the region accepted {val_latent_prior.acceptance_rate:.3%} of "
                  f"prior draws at the rejection sampler.", flush=True)
        file_manager.atomic_savez(w.payload("sbc_repeats.npz"), {
            "ks": ks, "c2st_ranks": c2st_ranks, "c2st_dap": c2st_dap, "ranks": pooled,
            "repeat": np.concatenate(repeat_col, axis=0), "n_valid": np.asarray(n_valid, dtype=np.int64),
            "labels": np.array([str(s) for s in labels]), "nps": np.asarray(nps)})
        results = {"per_param": per_param, "n_valid": n_valid, "stratum": stratum,
                   "kept_fraction": kept,
                   "accepted": list(getattr(posterior, "accepted", []))}
        w.parents = {"posterior": posterior.id, "prior": prior.id}
        w.fingerprints["gmm"] = prior.fingerprint
        w.config.update(settings)
        w.body = {"diagnostic": "sbc", "variant": stratum, "settings": settings, "results": results}
    return store.load_diagnostic(w.id)
