"""Which conditioning channels can the flow actually SEE? Reads the artifacts; simulates nothing.

Folded from ``scripts/channel_ablation.py`` (piece 2).

THE TEST HAD TO BE CORRECTED BEFORE IT WOULD ANSWER. "Replace channel i with its fitted mean and
expect exactly zero change from a dead channel" gives 5.2e-5 for A1_mean and 4.6e-4 for
D3_bimodality -- not zero -- so both look alive, which discards the finding. The reason is that the
fitted mean is itself contaminated: A1_mean was fitted at std 4.19e11 against a physical range of
~1e3, so "substitute the fitted mean" is a large perturbation, not a null one.

WHAT THIS DOES INSTEAD: sweep each channel across its own real p1-p99 range, every other channel held
at a REAL row's values, and record the largest embedding displacement that produces. Two dead channels
reporting the same number to three figures is itself the tell -- that value is the network's own float
noise, identical because neither channel moves the output at all.
"""
from __future__ import annotations

import math

import torch

from core import orchestrator as orch
from core.artifacts import resolve_store

FLOAT32_EPS = 1.1920929e-07


def _find_net(est):
    """The EmbeddedNet inside a trained estimator, wherever sbi wrapped it."""
    from core.SBI import embedded_network
    for module in est.modules():
        if isinstance(module, embedded_network.EmbeddedNet):
            return module
    raise ValueError("no EmbeddedNet inside this posterior's estimator, so there is no conditioning "
                     "pathway to sweep.")


def _sweep_channels(emb, data, n_sum, n_sweep, labels):
    """``[(max displacement, label, p1, p99, note)]`` for each SUMMARY column.

    ``emb`` is the WHOLE conditioning path, standardizer included -- not the bare EmbeddedNet. For a
    spontaneous or forced posterior sbi puts a per-column affine in front (``z_score_x="independent"``),
    so sweeping the bare net feeds raw values to a net trained on z-scores; the two coincide only under
    chi, where every historical measurement of this happened to be taken.

    THE BASE POINT IS A REAL ROW, not the column-wise median vector: under chi most probe-slot columns
    are zero in most rows, so a median "row" has an all-pad probe block -- an observation with no live
    probes, which no recording can produce.
    """
    p = next(emb.parameters())
    dev, dt = p.device, p.dtype
    data = data.to(device=dev, dtype=dt)
    med = data.median(dim=0).values
    scale = data.std(dim=0).clamp(min=1e-12)
    row = int((((data - med) / scale) ** 2).sum(dim=1).argmin())
    base = data[row].unsqueeze(0)
    q = torch.tensor([0.01, 0.99], device=dev)
    out = []
    with torch.no_grad():
        e0 = emb(base)
        for j in range(n_sum):
            col = data[:, j].float().sort().values
            idx = (q * (col.numel() - 1)).long()
            lo, hi = float(col[idx[0]]), float(col[idx[1]])
            if hi <= lo:
                out.append((0.0, labels[j], lo, hi, "CONSTANT"))
                continue
            v = base.repeat(n_sweep, 1)
            v[:, j] = torch.linspace(lo, hi, n_sweep, dtype=v.dtype, device=dev)
            out.append((float((emb(v) - e0).norm(dim=-1).max()), labels[j], lo, hi, ""))
    # The base-row marker FIRST, one record per summary column after it: that is what
    # channel_ablation unpacks (`base_row, swept = int(swept[0][0]), swept[1:]`) and what the sweep
    # test indexes past. Its r[0] is the row INDEX, not a displacement.
    return [(row, "__base__", 0.0, 0.0, "__base__")] + out


def channel_ablation(cfg, posterior, *, rows: int = 200_000, n_sweep: int = 33,
                     name: str = "", note: str = "", fig_sink=None, store=None):
    """Sweep each summary channel across its own p1-p99 range and record the embedding displacement.

    :param rows: the LEADING rows, in batch order, read from the posterior's own simulation cache for
                     the quantiles -- not a random sample of it (``training_checkpoint.load_rows``
                     walks the recorded shard ranges in order and stops at the first one that reaches
                     this count).
    :param n_sweep: points per channel sweep. The sweep is deterministic and takes no seed.
    """
    from core.SBI import training_checkpoint
    from core.SBI.statistics import FEATURE_LABELS, SUMMARY_WIDTH, VALID_FLAG_LABELS
    store = resolve_store(store)
    store.assert_name_free("diagnostic", name)
    if int(rows) < 1:
        raise ValueError(f"--rows must be at least 1, got {int(rows)}.")
    if int(n_sweep) < 2:
        raise ValueError(
            f"--n-sweep must be at least 2 -- a sweep with fewer points cannot describe a range, only "
            f"the channel's p1 value -- got {int(n_sweep)}.")
    label = posterior.name or posterior.id
    digest = posterior.manifest.parents.get("simulation")
    if digest is None:
        raise ValueError(
            f"Posterior '{label}' names no simulation cache, so checkpointing was off for its training "
            f"run and its conditioning rows no longer exist. The ranges a channel is swept over have "
            f"to be the ranges THIS network was trained on; another cache's quantiles describe a "
            f"different experiment. Retrain with checkpointing on, or pick a posterior that has one.")
    sm = store.get("simulation", digest)
    est = posterior.latent.posterior_estimator
    net = _find_net(est)
    n_sum = int(net.input_dim)
    if n_sum != SUMMARY_WIDTH + 1:
        raise ValueError(
            f"Posterior '{label}' has a {n_sum}-wide summary block; this build's layout is "
            f"{SUMMARY_WIDTH + 1} ({len(FEATURE_LABELS)} features + {len(VALID_FLAG_LABELS)} valid "
            f"flags + logT). Labelling its columns from the current list would silently misname them, "
            f"which is exactly the table this diagnostic exists to read.")
    labels = list(FEATURE_LABELS) + list(VALID_FLAG_LABELS) + ["logT"]
    width = int(posterior.manifest.body["conditioning"]["width"])
    print(f"[artifact] {label}   summary {n_sum} + forcing {int(net.forcing_dim)} = {width}")
    print(f"[standardizer] {'rank-Gaussian' if 'rg_knots' in net._buffers else 'sbi affine'}")

    data, _ = training_checkpoint.load_rows(store.path("simulation", digest),
                                           int(sm.body["batches_done"]),
                                           int(sm.body["identity"]["run_size"]),
                                           x_only=True, max_rows=int(rows))
    if data is None:
        raise ValueError(f"the simulation cache {digest} holds no committed rows.")
    if int(data.shape[1]) != width:
        raise ValueError(
            f"the cache {digest} holds {int(data.shape[1])}-wide rows but posterior '{label}' "
            f"conditions on {width}; they do not describe the same measurement.")
    print(f"[data] {data.shape[0]:,} rows x {data.shape[1]} from {digest}")

    live_probes = None
    cond = posterior.manifest.body["conditioning"]
    settings = {"rows": int(rows), "n_sweep": int(n_sweep)}
    with store.create("diagnostic", cfg, name=name, note=note) as w:
        emb = est.embedding_net
        emb.eval()
        swept = _sweep_channels(emb, data, n_sum, int(n_sweep), labels)
        base_row, swept = int(swept[0][0]), swept[1:]
        if posterior.manifest.body["mode"] == "chi":
            elem_w = int(cond["chi_elem_w"])
            blk = data[base_row, n_sum:].reshape(-1, elem_w)
            live_probes = int((blk[:, -1] > 0.5).sum())
        print(f"[base] real row {base_row}"
              + ("" if live_probes is None
                 else f", {live_probes} live probe(s) of {int(cond['chi_k_pad'])} slots"))

        # NaN/Inf is excluded from the median baseline too: one channel whose sweep drove the network
        # to a non-finite output must not corrupt the scale every OTHER channel's verdict is judged
        # against.
        live = [d for d, _, _, _, n in swept if n != "CONSTANT" and math.isfinite(d)]
        med = float(torch.tensor(live).median()) if live else 0.0
        print(f"\n=== max ||delta embedding|| over each channel's real p1-p99 range ===")
        print(f"median over non-constant channels: {med:.4g};  float32 eps = {FLOAT32_EPS:.3g}\n")
        print(f"{'channel':<24} {'max|d emb|':>12} {'vs median':>10}   {'p1':>12} {'p99':>12}  verdict")
        print("-" * 92)
        channels, n_const, n_invis, n_nonfinite = [], 0, 0, 0
        for d, lab, lo, hi, tag in sorted(swept):
            if tag == "CONSTANT":
                verdict = "constant in training (structurally dead)"
                n_const += 1
            elif not math.isfinite(d):
                # A NaN/Inf displacement compares False against every numeric test below (< and >
                # with NaN are always False), so it used to fall all the way through to "healthy" --
                # the opposite of what happened: the perturbation broke the network numerically, it did
                # not confirm the channel is fine. §4.1 also refuses to write a non-finite float into a
                # manifest (allow_nan=False), so this verdict has to exist before results is built,
                # not merely be caught there.
                verdict = "NON-FINITE -- this perturbation drove the network to a NaN/Inf output"
                n_nonfinite += 1
            elif d < FLOAT32_EPS or (med and d < med * 1e-4):
                # Either test alone under-reports. The absolute one misses a channel whose whole range
                # moves the embedding a millionth as far as a typical channel's but still clears an
                # ulp; the relative one would flag a channel on a network whose outputs are all tiny.
                verdict = "*** INVISIBLE -- the whole physical range does nothing ***"
                n_invis += 1
            elif med and d < med / 100:
                verdict = "severely compressed"
            elif med and d < med / 10:
                verdict = "compressed"
            else:
                verdict = "healthy"
            rel = (d / med) if med else None
            # S1 (spec 4.1): every float here goes through orch._num, so a non-finite value becomes
            # None instead of reaching the manifest writer (which refuses NaN/inf outright, after the
            # whole sweep has already run).
            channels.append({"label": lab, "max_disp": orch._num(d), "rel_median": orch._num(rel),
                             "p1": orch._num(lo), "p99": orch._num(hi), "verdict": verdict})
            print(f"{lab:<24} {d:12.4g} {(f'{rel:.3g}x' if rel is not None else '-'):>10}   "
                  f"{lo:12.4g} {hi:12.4g}  {verdict}")
        counts = {"constant": n_const, "invisible": n_invis, "nonfinite": n_nonfinite,
                  "usable": n_sum - n_const - n_invis - n_nonfinite, "total": n_sum}
        print(f"\n{n_const} structurally dead, {n_invis} numerically invisible, {n_nonfinite} "
              f"non-finite, {counts['usable']} usable of {n_sum} summary channels")

        results = {"n_rows": int(data.shape[0]), "base_row": base_row, "live_probes": live_probes,
                   "median_disp": orch._num(med), "channels": channels, "counts": counts,
                   "accepted": list(getattr(posterior, "accepted", []))}
        w.parents = {"posterior": posterior.id, "simulation": digest}
        w.config.update(settings)
        w.body = {"diagnostic": "ablation", "variant": None, "settings": settings, "results": results}
    return store.load_diagnostic(w.id)
