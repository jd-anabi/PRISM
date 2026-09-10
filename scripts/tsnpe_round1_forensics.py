"""Forensics for the 2026-09-02 TSNPE round (F1-F4 of the round-1 post-mortem). READ-ONLY.

Reproduces, from the artifacts on this machine, every number in the 2026-09-09 Appendix A entry of
PRISM_HANDOFF.md -- so the verdict there is a measurement anyone can re-run, not a recollection.

    F1  which checkpoint slot each run resolved to, and which identity fields differ between slots
    F2  every posterior sidecar's V against every checkpoint header's V and V^T (the GUI wrote V^T),
        the rotation pickled INSIDE each posterior's own training prior, and the direction
        correspondence |V_round1^T V_round0|
    F3  what the sidecars record (amortized / truncation / x_obs_digest / fisher_eigenvalues)
    F4  the truncation region rebuilt from the round-0 posterior at the persisted observation, the
        ground truth's containment under the round-0 rotation, the round-1 rotation and the
        transposed sidecar rotation, and the fraction of the parent posterior's own draws the region
        keeps once re-expressed in round 1's basis -- the direct measurement of defect D1

It writes NOTHING. It never imports the GUI. The round-1 directory is found by its digest suffix, so
it is recognised under its quarantined name (QUARANTINED_..._train_0b471d560271) as well as the
original. Every identifier below can be overridden through the environment for a later round.

Run:  python scripts/tsnpe_round1_forensics.py   (biophys-env; ~1-2 min, CPU only)
"""
import hashlib
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# ── the run under investigation (override through the environment for another round) ────────────
ROUND0_CKPT = os.environ.get("ROUND0_CKPT", "train_3780fd37a16a")      # the parent's checkpoint
ROUND1_CKPT = os.environ.get("ROUND1_CKPT", "train_0b471d560271")      # the TSNPE round's checkpoint
KEEPER_CKPT = os.environ.get("KEEPER_CKPT", "train_230ae7cb5fc2")      # section 4.6's posterior_08232026
PARENT_POST = os.environ.get("PARENT_POST", "posterior_09022026.pt")   # the round-0 posterior
OBSERVATION = os.environ.get("OBSERVATION", "obs_20260902T124213_11a301215f0ba841.pt")
CELL = os.environ.get("CELL", "master_entrained.txt")                   # the simulated ground truth
N_DIRECTIONS = int(os.environ.get("N_DIRECTIONS", "5"))
HPD = float(os.environ.get("HPD", "0.999"))


def _ts(p) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))


def _canonical(obj):
    """A copy of training_checkpoint._canonical, so this script needs no core import for F1."""
    if isinstance(obj, torch.Tensor):
        return _canonical(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return repr(obj)
    return str(obj)


def _digest(identity: dict) -> str:
    blob = json.dumps(_canonical(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _cmp(A, B) -> str:
    A, B = A.double(), B.double()
    if A.shape != B.shape:
        return f"shape mismatch {tuple(A.shape)} vs {tuple(B.shape)}"
    return (f"max|A-B| = {float((A - B).abs().max()):.3e}   "
            f"max|A-B^T| = {float((A - B.T).abs().max()):.3e}")


def _find_ckpt(root: Path, digest_name: str) -> Path | None:
    """The directory for a checkpoint digest, under its original OR a quarantined name."""
    exact = root / digest_name
    if exact.is_dir():
        return exact
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.endswith(digest_name):
            return d
    return None


def _load_state(d: Path):
    for name in ("state.pt", "state.prev.pt"):
        f = d / name
        if f.exists():
            try:
                st = torch.load(str(f), map_location="cpu", weights_only=False)
            except Exception:                    # noqa: BLE001 -- torn state; try the previous one
                continue
            if isinstance(st, dict) and "batches_done" in st:
                return st, name
    return None, None


def _loadings(V: torch.Tensor, keys, tag: str, i_t: int) -> None:
    print(f"\nLOADINGS of {tag}  (column j = direction j; entries = parameters):")
    for j in range(V.shape[1]):
        col = V[:, j]
        idx = col.abs().argsort(descending=True)[:3]
        print(f"   dir {j:2d}: " + "  ".join(f"{float(col[i]):+.2f}*{keys[i]}" for i in idx))
    print("   |loading of t_scale on each direction| (row of t_scale):",
          [round(abs(float(x)), 2) for x in V[i_t, :]])


def main() -> None:
    from core.config import CHECKPOINT_PATH, OBSERVATION_PATH, POSTERIOR_PATH

    print("=" * 100)
    print("TSNPE ROUND-1 FORENSICS  (read-only)   run at", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 100)

    # ── the persisted GUI settings: which budget each tab last used ─────────────────────────────
    ini = Path(os.environ.get("APPDATA", "")) / "PRISM" / "PRISM.ini"
    if ini.exists():
        import configparser
        cp = configparser.ConfigParser(interpolation=None, strict=False)
        try:
            cp.read(ini, encoding="utf-8")
            print(f"\nPRISM.ini ({_ts(ini)}):")
            for sec, keys in (("inference_posterior", ("num_runs", "run_size_cap", "posterior")),
                              ("inference_tsnpe", ("num_runs", "run_size_cap", "observation", "hpd", "n_dirs")),
                              ("inference_prior", ("prior",)),
                              ("inference_config", ("chi_mode", "reparam_rotate", "chi_k", "chi_f0", "chi_lo", "chi_hi", "chi_k_pad"))):
                if cp.has_section(sec):
                    print(f"   [{sec}] " + "  ".join(f"{k}={cp.get(sec, k, fallback='?')}" for k in keys))
        except Exception as e:                   # noqa: BLE001 -- the ini is evidence, not a dependency
            print(f"   (could not parse PRISM.ini: {e})")

    # ── F1: the checkpoint slots ────────────────────────────────────────────────────────────────
    print("\n" + "#" * 100 + "\n# F1  CHECKPOINT SLOTS\n" + "#" * 100)
    ck = {}
    for d in sorted(p for p in CHECKPOINT_PATH.iterdir() if p.is_dir() and (p / "header.pt").exists()):
        h = torch.load(str(d / "header.pt"), map_location="cpu", weights_only=False)
        st, st_file = _load_state(d)
        ident = h.get("identity", {})
        dig = _digest(ident)
        quarantined = d.name != f"train_{dig}"
        ck[f"train_{dig}"] = (d, h, st)
        V, probe = h.get("V"), h.get("probe")
        print(f"\n== {d.name}" + (f"   [QUARANTINED; its identity digests to train_{dig}]" if quarantined else ""))
        print(f"   header {_ts(d / 'header.pt')} | state {_ts(d / 'state.pt') if (d / 'state.pt').exists() else '-'}"
              f" | batches_done {None if st is None else st.get('batches_done')}"
              f" | complete {None if st is None else st.get('complete')} (from {st_file})")
        print(f"   run_size {h.get('run_size')} | n_runs {h.get('n_runs')} | prior {ident.get('prior_fingerprint')}"
              f" | V {None if V is None else tuple(V.shape)} | probe {None if probe is None else tuple(probe.shape)}"
              f" | 'truncation' in identity: {'truncation' in ident}")
    print("\nidentity fields that DIFFER between slots:")
    for a, b in itertools.combinations(sorted(ck), 2):
        ia, ib = ck[a][1].get("identity", {}), ck[b][1].get("identity", {})
        diff = [k for k in sorted(set(ia) | set(ib)) if _canonical(ia.get(k)) != _canonical(ib.get(k))]
        print(f"   {a} vs {b}: {diff}")

    # ── F2: the rotations ───────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 100 + "\n# F2  ROTATIONS: sidecar V against every header's V and V^T\n" + "#" * 100)
    sides = {}
    for p in sorted(POSTERIOR_PATH.glob("*.rot.pt")):
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(d, dict):
            d = {"V": d}
        sides[p.name] = d
        Vs = d.get("V")
        if Vs is None:
            print(f"\n{p.name}: V=None")
            continue
        Vs = Vs.to("cpu", torch.float64)
        print(f"\n{p.name} ({_ts(p)}), orthogonality err {float((Vs.T @ Vs - torch.eye(Vs.shape[0], dtype=torch.float64)).abs().max()):.1e}")
        for name, (_, h, _) in ck.items():
            Vc = h.get("V")
            if Vc is None or Vc.shape != Vs.shape:
                continue
            Vc = Vc.to("cpu", torch.float64)
            tag = ""
            if torch.equal(Vc, Vs):
                tag = "   <== sidecar == header V (the CLI orientation)"
            elif torch.equal(Vc.T, Vs):
                tag = "   <== sidecar == header V^T EXACTLY (a GUI-saved sidecar: defect D6)"
            print(f"   vs {name}: {_cmp(Vc, Vs)}{tag}")
        # the rotation pickled inside the posterior's own training prior
        pt = POSTERIOR_PATH / p.name.replace(".rot.pt", ".pt")
        if pt.exists():
            try:
                post = torch.load(str(pt), map_location="cpu", weights_only=False)
                obj, chain = getattr(post, "prior", None), []
                for _ in range(6):
                    if obj is None:
                        break
                    chain.append(type(obj).__name__)
                    if hasattr(obj, "V") and torch.is_tensor(obj.V):
                        break
                    obj = getattr(obj, "gen_dist", None) or getattr(obj, "base", None)
                Vn = getattr(obj, "V", None)
                if torch.is_tensor(Vn):
                    Vn = Vn.detach().to("cpu", torch.float64)
                    print(f"   rotation pickled inside the posterior's prior ({' -> '.join(chain)}): "
                          f"== sidecar V: {torch.equal(Vn, Vs)}   == sidecar V^T: {torch.equal(Vn, Vs.T)}")
                else:
                    print(f"   no rotation found inside the posterior's prior ({' -> '.join(chain) or 'no prior'})")
            except Exception as e:               # noqa: BLE001
                print(f"   (could not inspect the pickled prior: {e})")

    r0, r1 = ck.get(ROUND0_CKPT), ck.get(ROUND1_CKPT)
    if r0 is not None and r1 is not None and r0[1].get("V") is not None and r1[1].get("V") is not None:
        V0 = r0[1]["V"].to("cpu", torch.float64)
        V1 = r1[1]["V"].to("cpu", torch.float64)
        print(f"\nround-1 V ({ROUND1_CKPT}) against round-0 V ({ROUND0_CKPT}): {_cmp(V1, V0)}")
        M = V1.T @ V0
        print("direction correspondence |V1^T V0| (round-1 direction -> nearest round-0 direction):")
        for j in range(M.shape[0]):
            i = int(M[j].abs().argmax())
            second = float(M[j].abs().sort(descending=True).values[1])
            print(f"   V1 dir {j:2d} ~ V0 dir {i:2d}  cos = {float(M[j, i]):+.3f}  (runner-up |cos| = {second:.3f})")
        print("per-column cosines cos(V0[:,j], V1[:,j]) for j = 0..4:",
              [round(float((V0[:, j] * V1[:, j]).sum()), 3) for j in range(min(5, V0.shape[1]))])

    # ── F3: what the sidecars record ────────────────────────────────────────────────────────────
    print("\n" + "#" * 100 + "\n# F3  SIDECAR RECORDS\n" + "#" * 100)
    for name, d in sides.items():
        tr = d.get("truncation")
        print(f"   {name}: amortized={d.get('amortized')} truncation={None if tr is None else tr.get('dims')} "
              f"x_obs_digest={d.get('x_obs_digest')} fisher_eigenvalues={'None' if d.get('fisher_eigenvalues') is None else 'present'}")
    obs = sorted(OBSERVATION_PATH.glob("obs_*.pt"))
    print("\npersisted observations:")
    for p in obs:
        rec = torch.load(str(p), map_location="cpu", weights_only=False)
        print(f"   {p.name} ({_ts(p)}): mode={rec.get('mode')} K={rec.get('chi_n_freqs')} "
              f"x_obs {tuple(rec['x_obs'].shape)} first4 {[round(float(x), 3) for x in rec['x_obs'].reshape(-1)[:4]]}")

    # ── F4: ground-truth containment ────────────────────────────────────────────────────────────
    print("\n" + "#" * 100 + "\n# F4  REGION AT THE PERSISTED OBSERVATION; GROUND-TRUTH CONTAINMENT\n" + "#" * 100)
    if r0 is None or r1 is None:
        print(f"   need both {ROUND0_CKPT} and {ROUND1_CKPT} under {CHECKPOINT_PATH}; skipping F4")
        return
    from torch.distributions.transforms import ComposeTransform
    from core import cli, config, orchestrator, registry
    from core.config import BOUNDS_PATH, CELL_PATH, VALID_LABELS, VALID_MODELS
    from core.SBI import reparam, truncate

    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / "nadrowski" / "master.txt"),
                              chi_mode=True, chi_n_freqs=6, chi_f0=0.15, chi_freq_bounds=(0.03, 0.3),
                              chi_k_pad=12, chi_max_cycles=20.0, reparam_rotate=True)
    cfg.hw = config.cpu_device()
    cli.load_and_validate_gt(cfg, str(CELL_PATH / "nadrowski" / CELL))
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    i_t = len(cfg.params_dict) + cfg.rescale_idx["t_scale"]
    theta_true = cfg.ground_truth_tensor.detach().cpu().to(torch.float32).reshape(1, -1)
    print("theta_true (" + CELL + "):", {k: round(float(v), 5) for k, v in zip(keys, theta_true[0])})

    post = torch.load(str(POSTERIOR_PATH / PARENT_POST), map_location="cpu", weights_only=False)
    post.device = post._device = torch.device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T_side = reparam.load_eval_bijection(cfg, PARENT_POST, POSTERIOR_PATH)    # sidecar V as written
    box = ComposeTransform(list(T_side.parts[1:]))
    V0 = r0[1]["V"].to("cpu", torch.float32)
    V1 = r1[1]["V"].to("cpu", torch.float32)
    T_right = reparam.build_rotated_bijection(box, V0)
    T_r1 = reparam.build_rotated_bijection(box, V1)

    rec = torch.load(str(OBSERVATION_PATH / OBSERVATION), map_location="cpu", weights_only=False)
    x_obs = rec["x_obs"]
    print(f"observation {OBSERVATION}: digest re-verifies = {orchestrator.observation_digest(x_obs) == rec['digest']}")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # DELIBERATELY the pre-fix call: no t_scale_idx, so the box covers the leading directions
        # exactly as the 2026-09-02 round drew it (direction 0 included, which for this rotation is
        # t_scale alone and which the fixed code now skips). F4 reproduces what the round did.
        region = truncate.region_from_posterior(post, x_obs, n_directions=N_DIRECTIONS, level=HPD, n_samples=20000)
        with torch.no_grad():
            w = post.sample((20000,), x=x_obs, show_progress_bars=False).detach().to(torch.float64).cpu()
    print("REGION in the flow's own latent (basis-free numbers):", region)
    print("posterior latent per-dim std:", [round(float(s), 3) for s in w.std(0)])
    for name, T in ((f"box o V   ({ROUND0_CKPT}, the basis the parent trained in)", T_right),
                    (f"box o V'  ({ROUND1_CKPT}, round 1's fresh rotation)", T_r1),
                    ("box o V^T (the sidecar as written; what load_eval_bijection builds)", T_side)):
        with torch.no_grad():
            wt = T.inv(theta_true).to(torch.float64)
        print(f"\n{name}: contains theta_true = {bool(region.contains(wt)[0])}")
        for d, lo, hi in zip(region.dims, region.lo, region.hi):
            v = float(wt[0, d])
            print(f"      dim {d}: w_true = {v:8.3f}   region [{float(lo):8.3f}, {float(hi):8.3f}]   "
                  f"{'inside' if lo <= v <= hi else 'OUTSIDE'}")
    z = w.to(torch.float32) @ V0.T                       # box-latent z = w @ V0^T (w = z @ V0)
    in_v1 = float(region.contains((z @ V1).to(torch.float64)).float().mean())
    in_vt = float(region.contains((z @ V0.T).to(torch.float64)).float().mean())
    print(f"\nD1 MEASURED: fraction of the parent posterior's draws at x_obs that the region contains")
    print(f"   in its own coordinates ............................ {float(region.contains(w).float().mean()):.4f}")
    print(f"   re-expressed in round 1's rotation V' ............. {in_v1:.4f}")
    print(f"   re-expressed through the transposed sidecar (V^T) . {in_vt:.4f}")

    _loadings(V0, keys, f"{ROUND0_CKPT} ({PARENT_POST}, CORRECT orientation)", i_t)
    _loadings(V1, keys, f"{ROUND1_CKPT} (round 1's fresh rotation)", i_t)
    rk = ck.get(KEEPER_CKPT)
    if rk is not None and rk[1].get("V") is not None:
        Vk = rk[1]["V"].to("cpu", torch.float32)
        _loadings(Vk, keys, f"{KEEPER_CKPT} (posterior_08232026, the section-4.6 keeper, CORRECT orientation)", i_t)
        _loadings(Vk.T, keys, f"TRANSPOSE of {KEEPER_CKPT} -- what a reader of its GUI-saved sidecar saw (section 4.6's table)", i_t)


if __name__ == "__main__":
    main()
