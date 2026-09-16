"""Identifiability: what a trained posterior KEPT (``rotation``), and whether the information is
there at all (``laplace``, ``jacobian``).

Folded from ``scripts/{posterior_identifiability,identifiability_offgt,degeneracy_map}.py`` (piece 2).

The three answer different questions and must not be confused:
  rotation  reads the artifact's own Fisher eigenbasis. No simulation, runs in a second, and it is
            the only one that describes the posterior you actually have.
  laplace   SIMULATES a Laplace marginal-SD at K points -- the ground truth plus draws from the
            posterior's own training prior -- to ask whether SBC failures are an information deficit
            or a flow-calibration problem.
  jacobian  SIMULATES the standardized feature-Jacobian at the ground truth and reports degenerate
            pairs, the sloppy spectrum and dead channels. Chi-aware: under chi it is built over the
            FISHER channel set, not the conditioning block.

Both simulating variants build their Jacobian over the feature set the posterior actually conditions
on (``feature_sets``). Left on the 41-feature assumption they were literally independent of the chi
toggle -- they would report the kappa~x_scale alias as strong as ever and falsely refute the very
hypothesis chi mode exists to test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from core import orchestrator as orch
from core.artifacts import resolve_store
from core.Helpers import file_manager
from core.runs import public_entry

from . import feature_sets
from .rng import seeded

_SF, _SS, _SC = 1, 2, 3          # CRN seeds: forced / spontaneous / chi block


@public_entry
def identifiability_rotation(cfg, posterior, *, n_worst: int = 3, top_n: int = 4,
                             name: str = "", note: str = "", fig_sink=None, store=None):
    """Decompose a trained posterior's Fisher eigenbasis. Reads the artifact; simulates nothing.

    Answers what a corner plot cannot: a 13-D posterior with four well-constrained directions and
    nine prior-like ones looks like severe degeneracy on the physical axes and is a perfectly good
    result -- as long as you can say WHICH four. The columns of V are that basis, and the manifest
    carries them, so the decomposition is already on disk and free.

    :param n_worst: how many of the worst directions to total in the per-parameter table.
    :param top_n: loadings shown per direction.
    """
    store = resolve_store(store)
    store.assert_name_free("diagnostic", name)
    body = posterior.manifest.body
    tr, cond = body["transform"], body["conditioning"]
    label = posterior.name or posterior.id
    if tr.get("V") is None:
        raise ValueError(
            f"Posterior '{label}' records no Fisher rotation (V is None: the rotation was off for this "
            f"run), so there is no eigenbasis to decompose. Train with cfg.reparam_rotate on -- note "
            f"that a model without forcing disables it -- or point this at a posterior that has one.")
    # Refused, not clamped: a clamp turned --n-worst 0 into 1 without a word (the D6 trap).
    n_worst, top_n = int(n_worst), int(top_n)
    if n_worst < 1:
        raise ValueError(f"n_worst must be at least 1, got {n_worst}")
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")
    V = np.asarray(tr["V"], dtype=float)
    P = V.shape[0]
    if n_worst > P:
        # M8: W[i, P-n_worst:] with P-n_worst < 0 is a NEGATIVE slice -- Python reads it "from the
        # end", so bottom_share silently sums fewer than n_worst directions rather than crashing.
        raise ValueError(f"n_worst ({n_worst}) cannot exceed the number of directions P={P}.")
    names = list(tr["param_keys"])
    if len(names) != P:
        names = [f"p{i}" for i in range(P)]
    orth = float(np.abs(V.T @ V - np.eye(P)).max())
    print(f"[artifact] {label}   mode={body['mode']}   model={posterior.manifest.config.get('model')}")
    print(f"[V] {P}x{P}, orthogonal to {orth:.1e}")
    if cond.get("chi_k_pad") and cond.get("chi_elem_w"):
        k_pad, elem_w = int(cond["chi_k_pad"]), int(cond["chi_elem_w"])
        supplied, base = int(cond.get("chi_n_freqs") or 0), int(cond["input_dim"])
        print(f"[conditioning] width {base + k_pad * elem_w} = {base} base + {k_pad} probe slots x {elem_w}")
        print(f"[conditioning] {supplied} probe(s) supplied into {k_pad} slots -> "
              f"{(k_pad - supplied) * elem_w} elements are pure padding "
              f"({100.0 * (k_pad - supplied) / k_pad:.0f}% of the probe block never filled)")

    ev = tr.get("fisher_eigenvalues")
    if ev is None:
        # REPORTED, not refused: every TSNPE round carries None (it reuses the parent's V and never
        # runs a Fisher), and the ordering plus the loadings are still the answer to "which direction
        # is worst, and what it is made of" -- just not to "how much worse". Run this diagnostic on
        # the amortized PARENT, which carries them.
        print("\n[eigenvalues] NOT STORED for this artifact.")
        print("  Everything below is an ORDERING and a set of LOADINGS -- which direction is worst,")
        print("  and what it is made of -- but NOT how much worse it is. That scale is the question:")
        print("  a 3x spread means the experiment measures everything tolerably; 1e6 means it")
        print("  measures a handful of directions and returns the prior for the rest.")
        print("  Run this diagnostic on the amortized PARENT, which carries them.")
        ev_a = None
    else:
        ev_a = np.asarray(ev, dtype=float)
        pos = ev_a[ev_a > 0]
        spread = ev_a[0] / ev_a[-1] if ev_a[-1] > 0 else float("inf")
        pr = float((pos.sum() ** 2) / (pos ** 2).sum()) if pos.size else 0.0
        print(f"\n[eigenvalues] max {ev_a[0]:.4g}  min {ev_a[-1]:.4g}  spread {spread:.4g}")
        print(f"[eigenvalues] participation ratio {pr:.2f} of {P} "
              f"-> ~{pr:.1f} effectively constrained direction(s)")

    print("\n=== Fisher eigen-directions (columns of V), BEST-constrained first ===")
    print("loadings are in box-normalised coordinates, so they are comparable across parameters")
    directions = []
    for j in range(P):
        col = V[:, j]
        top = np.argsort(-np.abs(col))[:top_n]
        directions.append({"index": j, "eigenvalue": None if ev_a is None else float(ev_a[j]),
                           "loadings": [{"name": names[i], "loading": float(col[i])} for i in top]})
        part = "  ".join(f"{col[i]:+.2f}*{names[i]}" for i in top)
        ev_s = "" if ev_a is None else f"  [lambda={float(ev_a[j]):.3g}]"
        tag = "   <-- BEST" if j == 0 else ("   <-- WORST" if j == P - 1 else "")
        print(f"  dir {j:2d}: {part}{ev_s}{tag}")

    W = V ** 2                       # rows sum to 1: how parameter i is spread over the directions
    print(f"\n=== per parameter: share of its identifiability in the WORST directions ===")
    print(f"{'param':>10s} {'bottom-' + str(n_worst):>10s} {'top-4':>8s} {'peak dir':>9s}   verdict")
    per_param = []
    for i in range(P):
        bot, top4, peak = float(W[i, P - n_worst:].sum()), float(W[i, :4].sum()), int(np.argmax(W[i]))
        if bot > 0.6:
            verdict = "UNMEASURED" if bot > 0.95 else "very poor"
        elif bot > 0.25:
            verdict = "poor"
        elif top4 > 0.5:
            verdict = "good"
        else:
            verdict = "moderate"
        per_param.append({"name": names[i], "bottom_share": bot, "top4_share": top4,
                          "peak_dir": peak, "verdict": verdict})
    for rec in sorted(per_param, key=lambda p: -p["bottom_share"]):
        print(f"{rec['name']:>10s} {rec['bottom_share']:10.3f} {rec['top4_share']:8.3f} "
              f"{rec['peak_dir']:9d}   {rec['verdict']}")

    print("\n=== parameters that are their OWN near-null direction ===")
    print("(a parameter with ~all its weight on one bottom direction is FLAT, not aliased --")
    print(" no reparameterisation or prior rotation reaches it; only a new observable does)")
    flat_axes = []
    for i in range(P):
        j = int(np.argmax(W[i]))
        if j >= P - n_worst and W[i, j] > 0.9 and abs(V[i, j]) > 0.9:
            partners = [names[q] for q in range(P) if q != i and abs(V[q, j]) > 0.25]
            flat_axes.append({"name": names[i], "share": float(W[i, j]),
                              "loading": float(V[i, j]), "partners": partners})
            print(f"  {names[i]:>10s}: {W[i, j]:.3f} of its weight on dir {j}, loading {V[i, j]:+.3f}"
                  + (f", shared with {', '.join(partners)}" if partners else ", ALONE"))
    if not flat_axes:
        print("  none -- every poorly-constrained parameter is mixed with others, i.e. aliased "
              "rather than flat")

    settings = {"n_worst": n_worst, "top_n": top_n}
    # S1 (spec 4.1): every float reaching the manifest goes through orch._num here, in one place, so a
    # non-finite value becomes None instead of reaching the writer (which refuses NaN/inf outright).
    # The compute/print/sort logic above is untouched -- it keeps reading the RAW (unconverted) floats.
    num_directions = [{**d, "eigenvalue": orch._num(d["eigenvalue"]),
                       "loadings": [{**ld, "loading": orch._num(ld["loading"])} for ld in d["loadings"]]}
                      for d in directions]
    num_per_param = [{**p, "bottom_share": orch._num(p["bottom_share"]),
                      "top4_share": orch._num(p["top4_share"])} for p in per_param]
    num_flat_axes = [{**a, "share": orch._num(a["share"]), "loading": orch._num(a["loading"])}
                     for a in flat_axes]
    results = {"P": P, "orthogonality": orch._num(orth),
               "eigenvalues": None if ev_a is None else [orch._num(v) for v in ev_a],
               "directions": num_directions, "per_param": num_per_param, "flat_axes": num_flat_axes,
               "accepted": list(getattr(posterior, "accepted", []))}
    with store.create("diagnostic", cfg, name=name, note=note) as w:
        w.parents = {"posterior": posterior.id}
        w.config.update(settings)
        w.body = {"diagnostic": "identifiability", "variant": "rotation",
                  "settings": settings, "results": results}
    return store.load_diagnostic(w.id)


@dataclass
class _LapCtx:
    cfg: object
    n_obs: int
    names: list
    bounds: list
    is_log: list
    nd_dim: int
    m: int
    m_noise: int
    rel: float
    min_valid: float


def _training_latent_prior(latent):
    """The LATENT training prior pickled inside a posterior: walk ``.prior`` / ``.gen_dist`` until the
    generating distribution appears. For a TSNPE posterior that is the TRUNCATED prior, which is the
    proposal its flow was actually trained on -- drawing the off-GT points from the full prior would
    measure identifiability where this posterior has never seen a row."""
    obj = getattr(latent, "prior", None)
    for _ in range(6):
        if obj is None:
            break
        gen = getattr(obj, "gen_dist", None)
        if gen is not None:
            return gen
        obj = getattr(obj, "prior", None)
    raise ValueError(
        "this posterior carries no latent training prior (nothing with a `gen_dist` under its "
        "`.prior`), so the off-ground-truth points cannot be drawn from the distribution the flow "
        "was trained on.")


def _refuse_bad_arm_settings(m, rel, min_valid) -> None:
    """The finite-difference arms' own knobs, refused before the noise ensemble runs (laplace and
    jacobian share them): ``m`` rows per arm, the relative step ``rel`` and the validity floor
    ``min_valid``. Unchecked, ``m=0`` ran the whole noise ensemble and then every arm at batch 0, and
    ``rel=0`` on a zero-valued truth divided by zero on its way to lstsq, after the spend."""
    if int(m) < 1:
        raise ValueError(f"m must be at least 1 (rows per finite-difference arm), got {int(m)}")
    if not (math.isfinite(float(rel)) and float(rel) > 0):
        raise ValueError(f"rel must be a finite positive relative step, got {rel}")
    if not (0 < float(min_valid) <= 1):
        raise ValueError(f"min_valid must be a fraction in (0, 1], got {min_valid}")


def _laplace_raw(cfg, nd, res, force, m, crn, n_obs):
    """``(features (m, n_feat) float64, x_forced, x_spont)`` at one point. The fine grid is re-derived
    from ``res`` because perturbing t_scale changes the subsample factor and the grid length."""
    from core.config import CHUNK_LEN
    from core.SBI import pipeline
    dtype, device = cfg.hw.dtype, cfg.hw.device
    t_scale = float(res[cfg.rescale_idx["t_scale"]])
    subs = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    n_fine = min(cfg.steady_idx + n_obs * subs, len(cfg.t))
    t_fine = cfg.t[:n_fine]
    n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))
    p = nd.unsqueeze(0).expand(m, -1).contiguous()
    rv = res.unsqueeze(0).expand(m, -1).contiguous()
    fv = force.unsqueeze(0)
    forcef = pipeline.build_nondim_sin_force_tensor(fv.expand(m, -1), t_fine, rv,
                                                    cfg.forcing_idx, cfg.rescale_idx)

    def sim(f):
        return pipeline.gen_obs(model=cfg.model, params=p, t=t_fine,
                                inits=cfg.inits_tensor.expand(m, -1).contiguous(), force=f,
                                n_segs=n_segs, steady_idx=cfg.steady_idx,
                                state_dep_drift=cfg.state_dep_drift, batch_size=m, dtype=dtype,
                                device=device)[0][:, ::subs][:, :n_obs]

    if crn:
        # fork_rng CONFINED to the CRN-seeded arms only (R1). manual_seed(_SF)/(_SS) exist so the
        # +-d perturbation arms of ONE point's finite difference see the SAME simulated noise -- the
        # whole point of common random numbers -- and fork_rng keeps those fixed seeds from leaking
        # into the caller's stream once this call returns (the M5 defect this mirrors from
        # _jacobian_features). The crn=False noise-floor ensemble below must NOT be wrapped here: an
        # earlier version wrapped the whole function unconditionally, so fork_rng ALSO captured and
        # discarded the crn=False draw -- every Laplace POINT's m_noise ensemble then replayed the
        # same frozen state, and "independent" noise floors across GT/prior points were not
        # independent at all. The crn=False branch instead draws from the RUNNING seeded(...) stream,
        # so different points get different noise while the whole measurement stays reproducible
        # under --seed.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(_SF)
            xf = sim(forcef)
            torch.manual_seed(_SS)
            xs = sim(torch.zeros_like(forcef))
    else:
        xf = sim(forcef)
        xs = sim(torch.zeros_like(forcef))
    xsc = res[cfg.rescale_idx["x_scale"]].double()
    xof = res[cfg.rescale_idx["x_offset"]].double() if "x_offset" in cfg.rescale_idx else 0.0
    xf_d, xs_d = xsc * xf.double() + xof, xsc * xs.double() + xof
    amp_i, freq_i, phase_i = cfg.forcing_idx["amp"], cfg.forcing_idx["freq"], cfg.forcing_idx["phase"]
    feats = pipeline.gen_stats_features(xs_d, xf_d, cfg.dt_exp, fv[:, amp_i].expand(m).double(),
                                        fv[:, freq_i].expand(m).double(),
                                        fv[:, phase_i].expand(m).double(), device=device).numpy()
    return feats, xf_d, xs_d


def _analyze_point(ctx, nd, res, force):
    """``(marginal SD per parameter in prior-range units, measurable count)`` at one point.

    CRN central differences with a per-member validity filter and a one-sided fallback, standardized
    by the single-trajectory feature noise; SD = sqrt(diag((J^T J + I)^-1)).
    """
    cfg, P = ctx.cfg, len(ctx.names)
    device = cfg.hw.device
    n_feat = feature_sets.n_features(cfg)

    def _valid(xf_d, xs_d, cap):
        fin = torch.isfinite(xf_d).all(1) & torch.isfinite(xs_d).all(1)
        mag = (xf_d.abs().amax(1) < cap) & (xs_d.abs().amax(1) < cap)
        return (fin & mag).cpu().numpy()

    def measure(nd_, res_, force_, m, cap):
        feats, xf_d, xs_d = _laplace_raw(cfg, nd_, res_, force_, m, True, ctx.n_obs)
        v = _valid(xf_d, xs_d, cap)
        if not v.any():
            return np.full(n_feat, np.nan), 0.0
        return feats[v].mean(0), float(v.mean())

    def grad(perturb, base, d, cap):
        fp, vp = measure(*perturb(+d), cap)
        fm, vm = measure(*perturb(-d), cap)
        if vp >= ctx.min_valid and vm >= ctx.min_valid:
            return (fp - fm) / (2 * d)
        f0, _ = measure(*base, cap)
        if vp >= ctx.min_valid:
            return (fp - f0) / d
        if vm >= ctx.min_valid:
            return (f0 - fm) / d
        return np.full(n_feat, np.nan)

    f0, xf0, xs0 = _laplace_raw(cfg, nd, res, force, ctx.m_noise, False, ctx.n_obs)
    fin0 = (torch.isfinite(xf0).all(1) & torch.isfinite(xs0).all(1)).cpu().numpy()
    if fin0.sum() < 10:
        return np.full(P, np.nan), 0
    amax0 = torch.maximum(xf0.abs().amax(1), xs0.abs().amax(1)).cpu().numpy()
    cap = 100.0 * float(np.median(amax0[fin0]))
    keep = fin0 & (amax0 < cap)
    fnoise = np.maximum(f0[keep].std(0), 1e-9)
    cols, meas = [], 0
    for p in range(P):
        lo, hi = ctx.bounds[p]
        if p < ctx.nd_dim:
            base_val = float(nd[p])
            d = max(ctx.rel * (hi - lo), 1e-5 * abs(base_val))
            perturb = lambda dd, _i=p: (
                nd.clone().index_put_((torch.tensor([_i], device=device),), (nd[_i] + dd).reshape(1)),
                res, force, ctx.m)
        else:
            r = p - ctx.nd_dim
            base_val = float(res[r])
            d = max(ctx.rel * (hi - lo), 1e-5 * abs(base_val))
            perturb = lambda dd, _r=r: (
                nd, res.clone().index_put_((torch.tensor([_r], device=device),), (res[_r] + dd).reshape(1)),
                force, ctx.m)
        g = grad(perturb, (nd, res, force, ctx.m), d, cap)
        if np.isfinite(g).all():
            meas += 1
        g = np.nan_to_num(g) / fnoise
        fac = base_val * math.log(hi / lo) if (ctx.is_log[p] and base_val > 0 and lo > 0) else (hi - lo)
        cols.append(g * fac)
    J = np.stack(cols, axis=1)
    cov = np.linalg.inv(J.T @ J + np.eye(P))
    return np.sqrt(np.clip(np.diag(cov), 0, None)), meas


@public_entry
def identifiability_laplace(cfg, posterior, *, n_points: int = 6, m: int = 32, m_noise: int = 128,
                            rel: float = 0.02, min_valid: float = 0.5, sd_identified: float = 0.3,
                            t_obs_s: float | None = None, seed: int = 0,
                            name: str = "", note: str = "", fig_sink=None, store=None):
    """Does local identifiability hold ACROSS THE PRIOR, or only at the ground truth?

    SBC averages over the whole prior, so a metric evaluated only at the cell's GT cannot decide
    whether an SBC failure is an information deficit or a flow-calibration problem. This re-runs the
    Laplace marginal-SD analysis at ``n_points``: the GT plus draws from the posterior's OWN training
    prior. Stay pinned (SD << 1) prior-wide and the information is sufficient; degrade off-GT and it
    is not.

    :param sd_identified: SD below this counts as "identified" in the per-parameter fraction.
    :param t_obs_s: recording length in SECONDS the metric is measured at; None = config.T_MIN_EXP_S.
                     Used to size this diagnostic's own grid -- ``cfg.T_obs`` is never written.
    """
    from core import config
    store = resolve_store(store)
    store.assert_name_free("diagnostic", name)
    feature_sets.assert_not_chi(cfg, "identifiability laplace")
    feature_sets.assert_forced(cfg, "identifiability laplace")
    n_points = int(n_points)
    if n_points < 1:
        raise ValueError(f"n_points must be at least 1 (the ground truth is point 1), got {n_points}")
    if int(m_noise) < 10:
        # Below this the noise-floor ensemble is too small to estimate a per-feature std at all; left
        # unguarded it produces "Mean of empty slice" / zero-division RuntimeWarnings deep inside
        # _analyze_point rather than a refusal that names the knob to raise.
        raise ValueError(f"m_noise must be at least 10 (the feature-noise floor needs an ensemble), "
                         f"got {int(m_noise)}")
    _refuse_bad_arm_settings(m, rel, min_valid)
    _ = cfg.ground_truth                      # refuses a cell-free config, before anything is spent
    feature_sets.describe_features(cfg)

    dtype, device = cfg.hw.dtype, cfg.hw.device
    t_obs_s = float(config.T_MIN_EXP_S if t_obs_s is None else t_obs_s)
    n_obs_f = t_obs_s * cfg.get_unit_conversion_factor("s") / cfg.dt_exp
    if not math.isfinite(n_obs_f) or n_obs_f < 1:
        # R3: a bare `t_obs_s <= 0` check lets through a tiny positive value that still floors to
        # n_obs == 0 (a zero-length recording, not a refusal) and lets a NaN through to crash later
        # at int(nan). Guard on the computed n_obs instead -- and on non-finite t_obs_s directly,
        # since NaN * anything is NaN and n_obs_f < 1 alone would not catch +inf.
        raise ValueError(
            f"--t-obs must be a positive, finite recording length long enough for at least one "
            f"sample (n_obs = t_obs_s * unit_conversion_factor / dt_exp); t_obs_s={t_obs_s!r} gives "
            f"n_obs={n_obs_f!r}.")
    n_obs = int(n_obs_f)
    nd_dim = len(cfg.params_dict)
    res_names = list(cfg.rescale_params)
    names = list(cfg.params_dict) + res_names
    P = len(names)
    bounds = ([b for _, b in cfg.params_dict.values()]
              + [cfg.rescale_params[n][1] for n in res_names])
    # The UNIT is chosen by the parameter's NAME, not by the posterior's box: a "*_scale" rescale is
    # log-uniform in the prior, so its range is a ratio and its SD is in log-range units. Recorded per
    # record, because two numbers in one table would otherwise mean different things.
    is_log = [False] * nd_dim + ["scale" in n for n in res_names]
    ctx = _LapCtx(cfg=cfg, n_obs=n_obs, names=names, bounds=bounds, is_log=is_log, nd_dim=nd_dim,
                  m=int(m), m_noise=int(m_noise), rel=float(rel), min_valid=float(min_valid))
    settings = {"n_points": n_points, "m": int(m), "m_noise": int(m_noise), "rel": float(rel),
                "min_valid": float(min_valid), "sd_identified": float(sd_identified),
                "t_obs_s": t_obs_s, "seed": int(seed),
                "log_range_params": [n for n, lg in zip(names, is_log) if lg]}
    # M6: hoisted above store.create -- this can refuse (no `gen_dist` under the posterior's
    # `.latent.prior`), and a refusal must not leave a half-written diagnostic directory behind.
    latent_prior = _training_latent_prior(posterior.latent) if n_points > 1 else None

    with store.create("diagnostic", cfg, name=name, note=note) as w:
        with seeded(int(seed), device):
            gt_nd = cfg.params_tensor[0].clone()
            gt_res = torch.tensor([v for v, _ in cfg.rescale_params.values()], dtype=dtype, device=device)
            gt_force = torch.tensor([v for v, _ in cfg.force_params_dict.values()], dtype=dtype, device=device)
            points = [("GT", gt_nd, gt_res, gt_force)]
            if n_points > 1:
                T = posterior.posterior.T
                z = latent_prior.sample((n_points - 1,))
                theta = T(z.to(device))
                force_s = orch.build_forcing_prior(cfg).sample((n_points - 1,)).to(device)
                for k in range(n_points - 1):
                    points.append((f"prior{k + 1}", theta[k, :nd_dim].clone(),
                                   theta[k, nd_dim:].clone(), force_s[k].clone()))
            SD = np.full((len(points), P), np.nan)
            measurable = []
            for j, (tag, nd, res, force) in enumerate(points):
                sd, meas = _analyze_point(ctx, nd, res, force)
                SD[j] = sd
                measurable.append(int(meas))
                print(f"[{tag:8s}] measurable params={meas}/{P}", flush=True)

        print("\n=== marginal posterior SD per param across points (prior-range units) ===")
        print(f"{'param':9s} " + " ".join(f"{p[0][:7]:>7s}" for p in points)
              + f" {'median':>8s} {'frac<' + str(sd_identified):>8s}")
        per_param = []
        for p in range(P):
            row = SD[:, p]
            fin = row[np.isfinite(row)]
            med = float(np.median(fin)) if fin.size else float("nan")
            frac = float((fin < sd_identified).mean()) if fin.size else float("nan")
            per_param.append({"name": names[p], "unit": "log-range" if is_log[p] else "range",
                              "sd": [orch._num(v) for v in row], "median_sd": orch._num(med),
                              "frac_identified": orch._num(frac)})
            print(f"{names[p]:9s} "
                  + " ".join(f"{v:7.3f}" if np.isfinite(v) else f"{'nan':>7s}" for v in row)
                  + f" {med:8.3f} {frac:8.2f}")

        file_manager.atomic_savez(w.payload("laplace_sd.npz"), {
            "SD": SD, "points": np.array([p[0] for p in points]),
            "force": np.stack([p[3].detach().cpu().numpy() for p in points], axis=0),
            "names": np.array([str(s) for s in names])})
        results = {"points": [{"tag": p[0], "measurable": measurable[j]} for j, p in enumerate(points)],
                   "per_param": per_param,
                   "accepted": list(getattr(posterior, "accepted", []))}
        w.parents = {"posterior": posterior.id}
        w.config.update(settings)
        w.body = {"diagnostic": "identifiability", "variant": "laplace",
                  "settings": settings, "results": results}
    return store.load_diagnostic(w.id)


@dataclass
class _JacCtx:
    cfg: object
    n_obs: int
    mults: object                # chi probe multipliers (chi mode), else None
    forcing_gt: object           # (1, n_force) drive at the cell's own values (forced mode), else None
    keep_idx: list               # summary features that survive in chi mode
    feat_labels: list
    n_force_ch: int


def _jacobian_features(ctx, pvec, rescale_vec, m, crn):
    """``(features (m, n_feat) float64, x_for_validity, x_spont)`` at one point, in the mode's own
    feature set. The grid is re-derived from ``rescale_vec`` (perturbing t_scale changes it)."""
    from core.config import CHUNK_LEN
    from core.SBI import chi as chi_mod, pipeline
    cfg = ctx.cfg
    dtype, device = cfg.hw.dtype, cfg.hw.device
    t_scale = float(rescale_vec[cfg.rescale_idx["t_scale"]])
    subs = max(1, round((cfg.dt_exp / t_scale) / cfg.dt_nd_min))
    n_fine = min(cfg.steady_idx + ctx.n_obs * subs, len(cfg.t))
    t_fine = cfg.t[:n_fine]
    n_segs = max(1, math.ceil(n_fine / CHUNK_LEN))
    p = pvec.unsqueeze(0).expand(m, -1).contiguous()
    rv = rescale_vec.unsqueeze(0).expand(m, -1).contiguous()
    inits_m = cfg.inits_tensor.expand(m, -1).contiguous()

    def sim(f):
        return pipeline.gen_obs(model=cfg.model, params=p, t=t_fine, inits=inits_m, force=f,
                                n_segs=n_segs, steady_idx=cfg.steady_idx,
                                state_dep_drift=cfg.state_dep_drift, batch_size=m, dtype=dtype,
                                device=device)[0][:, ::subs][:, :ctx.n_obs]

    xsc = rescale_vec[cfg.rescale_idx["x_scale"]].double()
    xof = rescale_vec[cfg.rescale_idx["x_offset"]].double() if "x_offset" in cfg.rescale_idx else 0.0
    # fork_rng so the fixed CRN seeds do not leak out and pin the caller's global RNG (the same defect
    # that was fixed in SBI/decorrelate.feats).
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        if cfg.chi_mode:
            zero = torch.zeros((m, ctx.n_force_ch, t_fine.shape[0]), dtype=dtype, device=device)
            if crn:
                torch.manual_seed(_SS)
            xs_d = xsc * sim(zero).double() + xof
            # SEED AGAIN, right here. gen_chi_raw runs K MORE simulations whose noise is otherwise
            # completely unseeded, so the +delta and -delta arms of the central difference would see
            # different chi noise and the derivative would be swamped -- a plausible-looking,
            # meaningless map.
            if crn:
                torch.manual_seed(_SC)
            # The FISHER channel set, not the conditioning block: this builds a Jacobian, and
            # fnoise = max(std, 1e-9) is a DENOMINATOR, so a barely-varying channel is an amplifier.
            # ALL FOUR NAMED, NOTHING RE-SLICED: gen_chi_raw returns (chi, u, logcyc, valid), and a
            # `[:2]` here once bound `logcyc_v = u`.
            chi_v, _u_v, _logcyc_v, valid_v = pipeline.gen_chi_raw(
                model=cfg.model, params_nd=p, rescale=rv, x_spont_dim=xs_d.to(dtype),
                t_fine=t_fine, inits=inits_m, rescale_idx=cfg.rescale_idx, n_segs=n_segs,
                steady_idx=cfg.steady_idx, subsample=subs, N_points=ctx.n_obs, dt_exp=cfg.dt_exp,
                multipliers=ctx.mults, f0_nd=cfg.chi_f0, state_dep_drift=cfg.state_dep_drift,
                # Ceiling ON, filter OFF -- see the note at the same call in SBI/decorrelate.feats.
                max_cycles=cfg.chi_max_cycles, resolution_filter=False, dtype=dtype, device=device)
            if not bool(valid_v.all()):
                raise ValueError(
                    f"chi: {int((~valid_v).sum())} of {valid_v.numel()} probe measurements failed "
                    f"gen_chi_raw's finite / positive / 0.9x-Nyquist screen. fisher_features does not "
                    f"consult `valid`, so those entries would enter the Jacobian as if measured.")
            chi_block = chi_mod.fisher_features(chi_v)
            spont = pipeline.gen_stats_features(xs_d, None, cfg.dt_exp, None, None, None,
                                                device=device, spontaneous_only=True).numpy()
            feats = np.concatenate([spont[:, ctx.keep_idx], chi_block.double().cpu().numpy()], axis=1)
            return feats, xs_d, xs_d

        force = pipeline.build_nondim_sin_force_tensor(ctx.forcing_gt.expand(m, -1), t_fine, rv,
                                                        cfg.forcing_idx, cfg.rescale_idx)
        if crn:
            torch.manual_seed(_SF)
        xf = sim(force)
        if crn:
            torch.manual_seed(_SS)
        xs = sim(torch.zeros_like(force))
        xf_d, xs_d = xsc * xf.double() + xof, xsc * xs.double() + xof     # float64 redim
        amp_v = ctx.forcing_gt[:, cfg.forcing_idx["amp"]]
        freq_v = ctx.forcing_gt[:, cfg.forcing_idx["freq"]]
        phase_v = ctx.forcing_gt[:, cfg.forcing_idx["phase"]]
        feats = pipeline.gen_stats_features(xs_d, xf_d, cfg.dt_exp, amp_v.expand(m).double(),
                                            freq_v.expand(m).double(), phase_v.expand(m).double(),
                                            device=device).numpy()
        return feats, xf_d, xs_d


def _probe_budget(ctx, feats0, keep0, xs0, t_obs_s) -> None:
    """Does each chi probe MEASURE anything at this T_obs? Arithmetic, printed; plus the standing
    channel-identity check. Nothing here masks a probe that saw less than CHI_MIN_CYCLES drive cycles
    -- a sub-cycle lock-in returns the demeaned trace's residual drift: finite, in range, reproducible
    and not a susceptibility -- so the table is the only thing that says so."""
    from core.config import CHI_MIN_CYCLES
    from core.SBI import chi as chi_mod
    cfg = ctx.cfg
    if not cfg.chi_mode:
        return
    hz = cfg.get_unit_conversion_factor("s")
    t_obs_cell = t_obs_s * hz
    f_pk = chi_mod.peak_freq(xs0.to(cfg.hw.dtype), cfg.dt_exp).cpu().numpy()[keep0]
    f0_gt = float(np.median(f_pk))
    t_full = ctx.n_obs * cfg.dt_exp
    n_sp, n_ch = len(ctx.keep_idx), len(chi_mod.CHI_FISHER_CHANNELS)
    print(f"\n=== probe budget: T_obs={t_obs_cell:g} cell-time = {t_obs_s:g} s, "
          f"Omega_0={f0_gt * hz:.4g} Hz (ensemble median; p5..p95 "
          f"{np.percentile(f_pk, 5) * hz:.3g}..{np.percentile(f_pk, 95) * hz:.3g}) ===")
    print(f"  {'xOmega_0':>9} {'f (Hz)':>9} {'cycles':>8} {'floor':>7} {'ceiling':>8} "
          f"{'mean log|chi|':>14} {'cos^2+sin^2':>12}")
    bad = 0
    for j, mv in enumerate(ctx.mults.tolist()):
        cyc = mv * f0_gt * t_full
        lm = float(feats0[keep0][:, n_sp + n_ch * j + 0].mean())
        c = feats0[keep0][:, n_sp + n_ch * j + 1]
        s = feats0[keep0][:, n_sp + n_ch * j + 2]
        unit = float(np.mean(c ** 2 + s ** 2))
        bad += abs(unit - 1.0) > 1e-3
        print(f"  {mv:9.4f} {mv * f0_gt * hz:9.4g} {cyc:8.2f} "
              f"{'ok' if cyc >= CHI_MIN_CYCLES else 'DRIFT':>7} "
              f"{'PINNED' if cyc > cfg.chi_max_cycles else 'ok':>8} {lm:14.4g} {unit:12.6f}")
    if bad:
        raise ValueError(
            f"chi: {bad} probe(s) violate cos^2 + sin^2 == 1, so channels 1 and 2 of the Fisher block "
            f"are not the cosine and sine of one phase. Check what is being passed to "
            f"chi.fisher_features and the gen_chi_raw unpack (trap CHI10).")
    lo_b, hi_b = cfg.chi_freq_bounds
    print(f"  low edge {lo_b:g}x clears {CHI_MIN_CYCLES:g} cycles at T_obs >= "
          f"{CHI_MIN_CYCLES / (lo_b * f0_gt) / hz:.3g} s; high edge {hi_b:g}x stays under the "
          f"{cfg.chi_max_cycles:g}-cycle ceiling below T_obs = "
          f"{cfg.chi_max_cycles / (hi_b * f0_gt) / hz:.3g} s.")
    print(f"  adapt_placement (OFF here, see the gen_chi_raw call) would be a NO-OP iff the first of "
          f"those two numbers is <= this T_obs of {t_obs_s:g} s.", flush=True)


def _dead_channels(ctx, feats0, keep0, fnoise, noise_eps):
    """Feature rows whose ensemble spread is representation noise. fnoise is a DENOMINATOR: a row with
    no spread does not become a zero row of J, it becomes float32 rounding over float32 rounding --
    order 1 to 50, the magnitude of a real standardized feature, leading every table with nothing in
    the numbers to mark it. The test is RELATIVE: `u`'s std was 2.5e-8, twenty-five times above the
    1e-9 clamp, so an absolute test at the clamp would have missed the bug that motivated this."""
    fscale = np.maximum(np.abs(feats0[keep0]).max(0), 1e-30)
    dead = fnoise <= noise_eps * fscale
    if dead.any():
        print(f"\n!! DEAD FEATURE CHANNELS: {int(dead.sum())}/{len(fnoise)} rows ZEROED in J -- their "
              f"standardized entries would be amplified representation or quantization noise, not signal.")
        for i in np.flatnonzero(dead):
            print(f"     {ctx.feat_labels[i]:18s} std={fnoise[i]:9.3g}  |feat|={fscale[i]:9.3g}  "
                  f"ratio={fnoise[i] / fscale[i]:.2g}")
    else:
        print(f"[noise] no dead channels (min std/|feat| = {float((fnoise / fscale).min()):.2g} vs "
              f"noise_eps={noise_eps:g})", flush=True)
    return dead


def _jacobian(ctx, gt_nd, gt_rescale, fnoise, cap, m, rel, min_valid):
    """``(J, names, kinds, valid_frac)``: the standardized feature-Jacobian at the ground truth.
    One-sided where a perturbation side destabilizes (kept, flagged); a column is UNMEAS only if BOTH
    sides do."""
    cfg = ctx.cfg
    device = cfg.hw.device

    def feats_valid(pvec, rescale_vec, m_):
        feats, xf_d, xs_d = _jacobian_features(ctx, pvec, rescale_vec, m_, True)
        fin = torch.isfinite(xf_d).all(1) & torch.isfinite(xs_d).all(1)
        mag = (xf_d.abs().amax(1) < cap) & (xs_d.abs().amax(1) < cap)
        v = (fin & mag).cpu().numpy()
        return (feats[v].mean(0) if v.any() else np.full(feats.shape[1], np.nan)), float(v.mean())

    def grad(perturb, base, d):
        fp, vp = feats_valid(*perturb(+d))
        fm, vm = feats_valid(*perturb(-d))
        if vp >= min_valid and vm >= min_valid:
            return (fp - fm) / (2 * d) / fnoise, min(vp, vm), "central"
        f0, _ = feats_valid(*base)
        if vp >= min_valid:
            return (fp - f0) / d / fnoise, vp, "1-sided+"
        if vm >= min_valid:
            return (f0 - fm) / d / fnoise, vm, "1-sided-"
        return np.full_like(fnoise, np.nan), max(vp, vm), "UNMEAS"

    cols, names, vfr, kinds = [], [], [], []
    nd_bounds = [b for _, b in cfg.params_dict.values()]
    for i, nm in enumerate(cfg.params_dict):
        lo, hi = nd_bounds[i]
        d = max(rel * (hi - lo), 1e-5 * abs(float(gt_nd[i])))
        g, vf, kind = grad(
            lambda dd, _i=i: (gt_nd.clone().index_put_((torch.tensor([_i], device=device),),
                                                        (gt_nd[_i] + dd).reshape(1)), gt_rescale, m),
            (gt_nd, gt_rescale, m), d)
        cols.append(g); names.append(nm); vfr.append(vf); kinds.append(kind)
    for nm in cfg.rescale_params:
        r = cfg.rescale_idx[nm]
        lo, hi = cfg.rescale_params[nm][1]
        d = max(rel * (hi - lo), 1e-5 * abs(float(gt_rescale[r])))
        g, vf, kind = grad(
            lambda dd, _r=r: (gt_nd, gt_rescale.clone().index_put_(
                (torch.tensor([_r], device=device),), (gt_rescale[_r] + dd).reshape(1)), m),
            (gt_nd, gt_rescale, m), d)
        cols.append(g); names.append(nm); vfr.append(vf); kinds.append(kind)
    return np.stack(cols, axis=1), names, kinds, vfr


def _summaries(ctx, J, fnoise, dead, names, kinds, vfr, zero_tol, sink):
    """The tables, the two figures, and the extra arrays the caller folds into the npz payload.

    Returns a TUPLE, not "the results block" -- ``identifiability_jacobian``'s own ``results`` stays
    limited to spec Sec 4.5's keys, and everything restored here (the unique-handle fractions, the
    sloppiest direction, the top features, the rows dominating J) goes to the screen and to the npz's
    ``extra`` dict, the last element of the tuple, not into ``results``.

    ``dead`` is owned HERE (not by the caller) because the "zeroed N dead rows" amplification line
    needs J's PRE-zeroing values to report what was lost, and zeroing has to happen before every
    downstream statistic (norms, cosines, SVD) so none of them sees a channel that isn't really there.
    """
    from matplotlib import pyplot as plt
    if dead.any():
        # The amplification is printed because it is the EVIDENCE that the guard did something: a
        # dead row whose largest entry rivals the largest live one is a row that would have led the
        # payload table. ZEROED, not deleted, so every row index still matches feat_labels.
        fin = np.isfinite(J)
        print(f"[noise] zeroed {int(dead.sum())} dead rows of J; their largest standardized entry was "
              f"{np.abs(J[dead][fin[dead]]).max(initial=0.0):.3g}, against "
              f"{np.abs(J[~dead][fin[~dead]]).max(initial=0.0):.3g} over the live rows.", flush=True)
        J[dead, :] = 0.0
    P = J.shape[1]
    norms_std = np.array([np.linalg.norm(J[:, p]) if np.isfinite(J[:, p]).all() else np.nan
                          for p in range(P)])
    norms_raw = np.array([np.linalg.norm(J[:, p] * fnoise) if np.isfinite(J[:, p]).all() else np.nan
                          for p in range(P)])
    print("\n=== per-param gradient ===")
    print(f"{'param':11s} {'kind':9s} {'||g||_std':>10s} {'||g||_raw':>10s} {'valid':>6s}")
    for p in range(P):
        print(f"{names[p]:11s} {kinds[p]:9s} {norms_std[p]:10.3f} {norms_raw[p]:10.4g} {vfr[p]:6.2f}")

    # ---- which rows are driving J ---- advisory, no threshold: a row leading this table on a std
    # 1000x under the median is quantization (the probe budget above says which probe), not signal.
    abs_j = np.abs(np.nan_to_num(J, nan=0.0))
    rowmax, fmed = abs_j.max(1), float(np.median(fnoise))
    print(f"\n=== feature rows dominating J (median fnoise {fmed:.3g}; check it before believing one) ===")
    print(f"  {'row':18s} {'fnoise':>10s} {'/median':>9s} {'max|J|':>9s}  at param")
    row_top = np.argsort(-rowmax)[:8]
    for i in row_top:
        print(f"  {ctx.feat_labels[i]:18s} {fnoise[i]:10.3g} {fnoise[i] / max(fmed, 1e-30):9.3g} "
              f"{rowmax[i]:9.3g}  {names[int(np.argmax(abs_j[i]))]}")

    measurable = np.array([kinds[p] != "UNMEAS" for p in range(P)])
    stiff = measurable & (np.nan_to_num(norms_std) > zero_tol)
    mi = [p for p in range(P) if measurable[p]]
    si = [p for p in range(P) if stiff[p]]
    unmeasurable = [names[p] for p in range(P) if not measurable[p]]
    no_local_info = [names[p] for p in range(P) if measurable[p] and not stiff[p]]
    print(f"\nunmeasurable (both sides destabilize): {unmeasurable or 'none'}")
    print(f"no local info (||g||_std<{zero_tol}): {no_local_info or 'none'}")

    ns = [names[p] for p in mi]
    Jm = J[:, mi]
    Jn = Jm / np.maximum(np.linalg.norm(Jm, axis=0), 1e-12)
    C = np.abs(Jn.T @ Jn)
    print("\n=== |cos(grad_p, grad_q)| over measurable params (|cos|->1 = degenerate) ===")
    print("            " + " ".join(f"{n[:7]:>7s}" for n in ns))
    for i in range(len(mi)):
        print(f"{ns[i]:11s} " + " ".join(f"{C[i, j]:7.2f}" for j in range(len(mi))))
    pairs = [{"a": ns[i], "b": ns[j], "cos": float(C[i, j])}
             for i in range(len(mi)) for j in range(i + 1, len(mi)) if C[i, j] > 0.9]
    print("\ndegenerate pairs (|cos|>0.90): "
          + (", ".join(f"{p['a']}~{p['b']} ({p['cos']:.2f})" for p in pairs) or "none"))

    Js = J[:, si]
    if Js.size:
        _u, S, Vt = np.linalg.svd(Js, full_matrices=False)
    else:
        S, Vt = np.zeros(0), np.zeros((0, 0))
    nss = [names[p] for p in si]
    cond = float(S[0] / max(S[-1], 1e-12)) if S.size else float("nan")
    print(f"\n=== SVD over stiff columns {nss} ===")
    for k in range(S.size):
        print(f"  sigma[{k}] = {S[k]:9.3f}  (norm {S[k] / S[0]:.4f})")
    print(f"  condition number = {cond:.1f}")
    print("\n=== sloppiest stiff direction (smallest singular value) loadings ===")
    sloppiest = Vt[-1] if Vt.shape[0] else np.zeros(0)
    for j in np.argsort(-np.abs(sloppiest)):
        print(f"  {nss[j]:11s} {sloppiest[j]:+.3f}")

    # ---- unique-handle over measurable columns: ||g_p projected off span(others)|| / ||g_p|| ----
    print("\n=== unique-handle ||g_p _|_ span(others)|| / ||g_p|| (low = degenerate) ===")
    unique_frac = np.zeros(len(mi))
    for p in range(len(mi)):
        others = np.delete(Jm, p, axis=1)
        coef, *_ = np.linalg.lstsq(others, Jm[:, p], rcond=None)
        unique_frac[p] = np.linalg.norm(Jm[:, p] - others @ coef) / max(np.linalg.norm(Jm[:, p]), 1e-12)
    for p in sorted(range(len(mi)), key=lambda q: unique_frac[q]):
        print(f"  {ns[p]:11s} unique={unique_frac[p]:.3f}   ||g||_std={np.linalg.norm(Jm[:, p]):.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(C, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns, rotation=45, ha="right")
    ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns)
    for i in range(len(ns)):
        for j in range(len(ns)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] < 0.6 else "black", fontsize=6)
    ax.set_title(f"|cos| between standardized feature-gradients ({len(ctx.feat_labels)} features)")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    sink("Jacobian cosine matrix", fig)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    if S.size:
        ax2.bar(range(S.size), S / S[0], color="steelblue"); ax2.set_yscale("log")
    ax2.set_xlabel("singular index"); ax2.set_ylabel("sigma / sigma_max (log)")
    ax2.set_title("Jacobian singular spectrum over stiff cols (small = sloppy)")
    fig2.tight_layout()
    sink("Jacobian singular spectrum", fig2)

    # ---- top features per parameter -- the script calls this "the scientific payload": not merely
    # whether an alias weakened, but WHICH features drove it.
    print("\n=== top features per parameter ===")
    top_k = min(5, J.shape[0])
    top_feat_idx = np.full((P, top_k), -1, dtype=int)
    top_feat_val = np.full((P, top_k), np.nan)
    for p in range(P):
        col = J[:, p]
        if not np.isfinite(col).all():
            print(f"  {names[p]:11s} (unmeasurable)")
            continue
        top = np.argsort(-np.abs(col))[:top_k]
        top_feat_idx[p, :len(top)] = top
        top_feat_val[p, :len(top)] = col[top]
        print(f"  {names[p]:11s} " + ", ".join(f"{ctx.feat_labels[i]}={col[i]:+.2f}" for i in top))

    extra = {
        "measurable_mask": measurable, "stiff_mask": stiff,
        "C": C, "S": S, "sloppiest_loadings": sloppiest, "unique_frac": unique_frac,
        "pair_a": np.array([p["a"] for p in pairs], dtype=str),
        "pair_b": np.array([p["b"] for p in pairs], dtype=str),
        "pair_cos": np.array([p["cos"] for p in pairs], dtype=float),
        "top_feat_idx": top_feat_idx, "top_feat_val": top_feat_val,
        "row_dominant_idx": row_top.astype(int), "row_dominant_fnoise": fnoise[row_top],
        "row_dominant_ratio": fnoise[row_top] / max(fmed, 1e-30), "row_dominant_maxJ": rowmax[row_top],
    }
    return norms_std, norms_raw, unmeasurable, pairs, cond, extra


@public_entry
def identifiability_jacobian(cfg, *, m: int = 32, m_noise: int = 128, rel: float = 0.02,
                             min_valid: float = 0.5, zero_tol: float = 0.05, noise_eps: float = 1e-6,
                             t_obs_s: float | None = None, seed: int = 0,
                             name: str = "", note: str = "", fig_sink=None, store=None):
    """Degeneracy / sloppiness map over every inferred parameter, at the cell's ground truth.

    ``J[j, p] = d<feature_j>/d(param_p) / noise_std_j`` -- signal-to-noise units, i.e. an
    identifiability map -- then pairwise |cos| (degenerate pairs) and the SVD spectrum (sloppy
    directions). Takes no posterior: it is about the EXPERIMENT, not about a trained artifact.

    :param zero_tol: ||g||_std below this is "no local info"; such columns are left out of the SVD so
                     they cannot dominate it.
    :param noise_eps: fnoise/|feature| below this is a dead channel, ZEROED in J.
    :param t_obs_s: recording length in seconds the map is measured at; None = config.T_MIN_EXP_S. A
                     map at another T_obs is another measurement, not a redo -- compare equal-T runs.
    """
    from core import config, forcing
    from core.SBI import chi as chi_mod
    store = resolve_store(store)
    store.assert_name_free("diagnostic", name)
    feature_sets.assert_nadrowski(cfg, "the printed ND parameter names are Nadrowski's")
    if not cfg.chi_mode:
        # chi probes at its own frequencies and ignores the cell's drive, so only this branch needs one.
        feature_sets.assert_forced(cfg, "identifiability jacobian in forced mode")
    if int(m_noise) < 10:
        raise ValueError(f"m_noise must be at least 10 (the feature-noise floor needs an ensemble), "
                         f"got {int(m_noise)}")
    _refuse_bad_arm_settings(m, rel, min_valid)
    _ = cfg.ground_truth
    feature_sets.describe_features(cfg)

    dtype, device = cfg.hw.dtype, cfg.hw.device
    t_obs_s = float(config.T_MIN_EXP_S if t_obs_s is None else t_obs_s)
    n_obs_f = t_obs_s * cfg.get_unit_conversion_factor("s") / cfg.dt_exp
    if not math.isfinite(n_obs_f) or n_obs_f < 1:
        # R3: see the identical guard in identifiability_laplace -- a bare t_obs_s <= 0 check lets a
        # tiny positive value through to a zero-length recording (n_obs == 0) and lets NaN through to
        # crash at int(nan) instead of refusing here, before any simulation.
        raise ValueError(
            f"--t-obs must be a positive, finite recording length long enough for at least one "
            f"sample (n_obs = t_obs_s * unit_conversion_factor / dt_exp); t_obs_s={t_obs_s!r} gives "
            f"n_obs={n_obs_f!r}.")
    n_obs = int(n_obs_f)
    gt_nd = cfg.params_tensor[0].clone()
    gt_rescale = torch.tensor([v for v, _ in cfg.rescale_params.values()], dtype=dtype, device=device)
    ctx = _JacCtx(cfg=cfg, n_obs=n_obs,
                  mults=chi_mod.chi_multipliers_for(cfg) if cfg.chi_mode else None,
                  forcing_gt=(None if cfg.chi_mode else torch.tensor(
                      [[v for v, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)),
                  keep_idx=feature_sets.summary_keep_idx(),
                  feat_labels=feature_sets.feature_labels(cfg),
                  n_force_ch=forcing.n_force_channels(cfg.model, cfg.forcing_idx,
                                                      cfg.inits_tensor.shape[-1]))
    if cfg.chi_mode:
        print(f"[mode] probe multipliers of Omega_0: "
              f"{[round(v, 4) for v in ctx.mults.tolist()]}", flush=True)
    settings = {"m": int(m), "m_noise": int(m_noise), "rel": float(rel), "min_valid": float(min_valid),
                "zero_tol": float(zero_tol), "noise_eps": float(noise_eps), "t_obs_s": t_obs_s,
                "seed": int(seed)}

    with store.create("diagnostic", cfg, name=name, note=note) as w:
        with seeded(int(seed), device):
            feats0, xf0, xs0 = _jacobian_features(ctx, gt_nd, gt_rescale, int(m_noise), False)
            fin0 = (torch.isfinite(xf0).all(1) & torch.isfinite(xs0).all(1)).cpu().numpy()
            if fin0.sum() < 10:
                # M7: mirrors _analyze_point's own guard (laplace). Below this the noise floor is not
                # estimable at all -- unguarded, np.median(amax0[fin0]) on an empty selection warns
                # "Mean of empty slice", the derived CAP is NaN, keep0 ends up all-False, and
                # feats0[keep0].std(0) then warns "Degrees of freedom <= 0" on a zero-size reduction.
                raise ValueError(
                    f"identifiability jacobian: only {int(fin0.sum())} of {fin0.size} baseline "
                    f"trajectories at the ground truth were finite, so no feature-noise floor could "
                    f"be measured. The simulation is destabilizing at this T_obs/ground truth; check "
                    f"the cell and --t-obs.")
            amax0 = torch.maximum(xf0.abs().amax(1), xs0.abs().amax(1)).cpu().numpy()
            cap = 100.0 * float(np.median(amax0[fin0]))
            keep0 = fin0 & (amax0 < cap)
            fnoise = np.maximum(feats0[keep0].std(0), 1e-9)
            print(f"[noise] CAP={cap:.4g}  GT valid frac={keep0.mean():.2f}  "
                  f"median feature noise={np.median(fnoise):.4g}", flush=True)
            _probe_budget(ctx, feats0, keep0, xs0, t_obs_s)
            dead = _dead_channels(ctx, feats0, keep0, fnoise, float(noise_eps))
            J, names, kinds, vfr = _jacobian(ctx, gt_nd, gt_rescale, fnoise, cap, int(m),
                                             float(rel), float(min_valid))
        sink = w.fig_sink(fig_sink)
        norms_std, norms_raw, unmeasurable, pairs, cond, extra = _summaries(
            ctx, J, fnoise, dead, names, kinds, vfr, float(zero_tol), sink)
        file_manager.atomic_savez(w.payload("degeneracy_map.npz"), {
            "J": J, "fnoise": fnoise, "dead": dead, "norms_std": norms_std, "norms_raw": norms_raw,
            "feat_labels": np.array([str(s) for s in ctx.feat_labels]),
            "param_names": np.array([str(s) for s in names]),
            "kinds": np.array(kinds), "valid_frac": np.asarray(vfr, dtype=float),
            "mults": (ctx.mults.cpu().numpy() if cfg.chi_mode else np.zeros(0)),
            **extra})
        results = {"observation_mode": cfg.observation_mode, "T_obs_s": t_obs_s,
                   "n_features": len(ctx.feat_labels), "condition_number": orch._num(cond),
                   "unmeasurable": unmeasurable, "degenerate_pairs": pairs,
                   "dead_channels": [ctx.feat_labels[i] for i in np.flatnonzero(dead)]}
        w.parents = {}
        w.config.update(settings)
        w.body = {"diagnostic": "identifiability", "variant": "jacobian",
                  "settings": settings, "results": results}
    return store.load_diagnostic(w.id)
