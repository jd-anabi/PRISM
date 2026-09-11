"""The artifact store: directories under one root, a manifest in each, and (from Task 4 on) the
loaders that refuse a mismatch before anything is spent.

Every generated artifact is ``<root>/<kind dir>/<name>__<id>/`` -- ``_unnamed__<id>`` until it is
named -- with ``manifest.json`` written LAST, atomically, by ``ArtifactWriter``. The store indexes by
READING manifests, so a hand-renamed folder still resolves and parent references (by id) survive.
The simulation cache is the exception: its directory is the identity digest and its manifest is
written by ``training_checkpoint`` (``write_simulation_manifest``), because a resumable cache must
survive an interrupted run, which a writer's remove-on-exception would delete.
"""
from __future__ import annotations

import contextlib
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import config
from core.Helpers import file_manager

from . import manifest as mf
from . import provenance as prov

KIND_DIRS = {"prior": "priors", "simulation": "simulations", "posterior": "posteriors",
             "observation": "observations", "calibration": "calibrations", "inference": "inferences"}
MANIFEST = "manifest.json"
# Which ``parents`` keys can name an artifact of a given kind.
_PARENT_KEYS = {"prior": ("prior",), "simulation": ("simulation",),
                "posterior": ("posterior", "parent_posterior"), "observation": ("observation",),
                "calibration": (), "inference": ()}


class StoreError(ValueError):
    """The store refused: a missing, incomplete, duplicate or depended-upon artifact."""


@dataclass(frozen=True)
class Accept:
    """The ONLY escape hatches on the load path. Every flag used is written into the downstream
    manifest's ``results.accepted``, so a number produced under one is marked."""
    truncated: bool = False            # load a NON-AMORTIZED (TSNPE) posterior
    other_observation: bool = False    # infer with it on an observation other than its region's

    def used(self) -> list:
        return [n for n in ("truncated", "other_observation") if getattr(self, n)]


@dataclass
class Summary:
    kind: str
    id: str
    name: str
    created: str
    note: str
    path: Path
    complete: bool
    reason: "str | None"
    mode: "str | None" = None
    width: "int | None" = None
    amortized: "bool | None" = None
    parents: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or f"(unnamed {self.id})"


@dataclass
class Loaded:
    kind: str
    id: str
    name: str
    manifest: mf.Manifest
    path: Path


@dataclass
class LoadedPrior(Loaded):
    prior: object = None            # the physical ProductPrior (ND x rescale)
    force_prior: object = None      # None for a no-forcing model
    nd_prior: object = None         # the TransformedDistribution over the latent GMM
    fingerprint: "str | None" = None


@dataclass
class LoadedPosterior(Loaded):
    posterior: object = None        # reparam.TransformedPosterior (carries .T, .truncation, .x_obs_digest)
    latent: object = None           # the sbi DirectPosterior
    fingerprint: "str | None" = None
    diagnostics: "dict | None" = None
    accepted: list = field(default_factory=list)


@dataclass
class LoadedObservation(Loaded):
    x_obs: object = None            # (1, W) conditioning row
    obs_data: object = None         # (1, N) the observed trace
    t_dim: object = None            # (1, N) seconds
    digest: str = ""
    mode: str = ""
    width: int = 0

    def install(self, cfg) -> None:
        """Put the observation's context back on ``cfg`` -- what infer_and_visualize reads from it:
        T_obs, n_obs, the chi probe frequencies, the forcing values, and for a simulated observation
        the ground truth and initial conditions (so show_truth works in a fresh session)."""
        import torch
        body = self.manifest.body
        cfg.T_obs = float(body["T_obs_cell"])
        cfg.n_obs = int(body["n_obs"]) if body["n_obs"] is not None else None
        cfg.chi_obs_freqs = (None if body["chi_obs_freqs"] is None else
                             torch.tensor(body["chi_obs_freqs"], dtype=cfg.hw.dtype, device=cfg.hw.device))
        src = body["source"]
        if src["kind"] == "simulated":
            cfg.inject_ground_truth(dict(src["inits"]), dict(src["params"]), dict(src["rescale"]),
                                    dict(src["forcing"]))
        cfg.set_observation_context(cfg.T_obs, dict(body["forcing_vals"] or {}))


@dataclass
class LoadedCalibration(Loaded):
    results: dict = field(default_factory=dict)


@dataclass
class LoadedInference(Loaded):
    results: dict = field(default_factory=dict)
    samples_path: "Path | None" = None


def slug(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
    return s or "figure"


def _rmtree_retry(path, retries: int = 3, backoff_s: float = 0.1) -> None:
    """shutil.rmtree with the same PermissionError patience as file_manager._atomic_write: on
    Windows a virus scanner or an Explorer preview holds files open for a moment."""
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff_s * (attempt + 1))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_manifest(path: Path, m: mf.Manifest) -> None:
    text = mf.to_json_text(m).encode("utf-8")
    file_manager._atomic_write(path / MANIFEST, lambda fh: fh.write(text))


class ArtifactWriter:
    """Context manager handed out by ``ArtifactStore.create``: creates the directory, hands out
    payload and figure paths, and on a clean exit hashes the payloads and writes the manifest LAST.
    On ANY exception (a cancel included) the directory is removed and the exception re-raised, so a
    half-artifact never looks real."""

    def __init__(self, store, kind, cfg, *, name, note, id, created):
        self.store, self.kind, self.cfg = store, kind, cfg
        self.name, self.note, self.id, self.created = name, note, id, created
        self.dir = store.kind_dir(kind) / f"{name or mf.UNNAMED_DIR}__{id}"
        self.config = mf.config_from_cfg(cfg) if cfg is not None else {}
        self.parents, self.fingerprints, self.body = {}, {}, {}
        self._payloads, self._figures = [], []
        self.manifest = None

    def payload(self, filename: str) -> Path:
        if "/" in filename or "\\" in filename:
            raise StoreError(f"a payload is a file directly inside the artifact directory: {filename!r}")
        if filename not in self._payloads:
            self._payloads.append(filename)
        return self.dir / filename

    def figure_path(self, title: str) -> Path:
        figs = self.dir / "figures"
        figs.mkdir(exist_ok=True)
        base = slug(title)
        p, n = figs / f"{base}.png", 2
        while p.exists():
            p = figs / f"{base}-{n}.png"
            n += 1
        self._figures.append(f"figures/{p.name}")
        return p

    def fig_sink(self, forward=None):
        """A ``(title, fig) -> None`` sink that saves the PNG here and THEN forwards to the GUI's sink
        (which closes the figure after rendering, base_panel._png_fig_sink) or closes it itself."""
        def _sink(title, fig):
            from matplotlib import pyplot as plt
            from core.Helpers.visualizers import save_figure
            save_figure(fig, self.figure_path(title), dpi=110, bbox_inches="tight")
            if forward is not None:
                forward(title, fig)
            else:
                plt.close(fig)
        return _sink

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            _rmtree_retry(self.dir)
            return False
        self._commit()
        return False

    def _commit(self) -> None:
        hw = getattr(self.cfg, "hw", None)
        d = dict(
            schema=mf.SCHEMA, kind=self.kind, id=self.id, name=self.name, created=self.created.isoformat(),
            note=self.note, prism=prov.git_info(config.REPO_ROOT),
            env=prov.env_info(hw if hw is not None else config.cpu_device()),
            inputs=(prov.inputs_from_cfg(self.cfg) if self.cfg is not None
                    else {"bounds": None, "cell": None, "units": None, "model": None}),
            config=self.config, parents=dict(self.parents), fingerprints=dict(self.fingerprints),
            payloads={f: prov.sha256_file(self.dir / f) for f in self._payloads},
            figures=list(self._figures), body=self.body,
        )
        self.manifest = mf.validate(d)
        _write_manifest(self.dir, self.manifest)


class ArtifactStore:
    def __init__(self, root, *, clock=None):
        self.root = Path(root)
        self._clock = clock or _utc_now

    def kind_dir(self, kind: str) -> Path:
        if kind not in KIND_DIRS:
            raise StoreError(f"unknown artifact kind {kind!r}")
        return self.root / KIND_DIRS[kind]

    # ── index ────────────────────────────────────────────────────────────────────────────────────
    def _entries(self, kind: str) -> list:
        """``(dir, Manifest | None, reason)`` for every directory under the kind."""
        d = self.kind_dir(kind)
        if not d.is_dir():
            return []
        out = []
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            mpath = sub / MANIFEST
            if not mpath.exists():
                out.append((sub, None, "no manifest.json (incomplete or interrupted)"))
                continue
            try:
                m = mf.from_json_text(mpath.read_text(encoding="utf-8"))
            except (mf.ManifestError, OSError) as e:
                out.append((sub, None, f"unreadable manifest: {e}"))
                continue
            if m.kind != kind:
                out.append((sub, None, f"manifest declares kind {m.kind!r}"))
                continue
            out.append((sub, m, None))
        return out

    def list(self, kind: str) -> list:
        rows = []
        for sub, m, reason in self._entries(kind):
            if m is None:
                rows.append(Summary(kind, sub.name, "", "", "", sub, False, reason))
                continue
            body = m.body
            rows.append(Summary(kind, m.id, m.name, m.created, m.note, sub, True, None,
                                mode=body.get("mode"), width=(body.get("conditioning") or {}).get("width"),
                                amortized=body.get("amortized"), parents=dict(m.parents)))
        rows.sort(key=lambda s: s.created, reverse=True)      # newest first ...
        rows.sort(key=lambda s: not s.complete)               # ... complete first (stable)
        return rows

    def _find(self, kind: str, ref: str):
        for sub, m, _ in self._entries(kind):
            if m is not None and (m.id == ref or (m.name and m.name == ref)):
                return sub, m
        return None, None

    def get(self, kind: str, ref: str) -> mf.Manifest:
        _, m = self._find(kind, ref)
        if m is None:
            incomplete = [s.name for s, mm, _ in self._entries(kind) if mm is None]
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r} under {self.kind_dir(kind)}"
                             + (f" (incomplete directories present: {incomplete})" if incomplete else ""))
        return m

    def path(self, kind: str, ref: str) -> Path:
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}")
        return sub

    # ── create / rename / delete ─────────────────────────────────────────────────────────────────
    def _new_id(self, kind: str) -> str:
        stamp = self._clock().strftime("%Y%m%dT%H%M%S")
        taken = {m.id for _, m, _ in self._entries(kind) if m is not None}
        d = self.kind_dir(kind)
        if d.is_dir():
            taken |= {p.name.rsplit("__", 1)[-1] for p in d.iterdir() if p.is_dir()}
        cand, n = stamp, 2
        while cand in taken:
            cand, n = f"{stamp}-{n}", n + 1
        return cand

    def create(self, kind: str, cfg=None, *, name: str = "", note: str = "") -> ArtifactWriter:
        if kind == "simulation":
            raise StoreError("simulation manifests are written by training_checkpoint (write_simulation_manifest)")
        if name and not mf.NAME_RE.match(name):
            raise StoreError(f"bad artifact name {name!r}: letters, digits, _ . - and at most 64 characters")
        if name and self._find(kind, name)[1] is not None:
            raise StoreError(f"a {kind} named {name!r} already exists; rename or delete it first")
        return ArtifactWriter(self, kind, cfg, name=name, note=note, id=self._new_id(kind), created=self._clock())

    def rename(self, kind: str, ref: str, new_name: str) -> mf.Manifest:
        if not mf.NAME_RE.match(new_name or ""):
            raise StoreError(f"bad artifact name {new_name!r}: letters, digits, _ . - and at most 64 characters")
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}")
        other = self._find(kind, new_name)[1]
        if other is not None and other.id != m.id:
            raise StoreError(f"a {kind} named {new_name!r} already exists; rename or delete it first")
        m.name = new_name
        _write_manifest(sub, m)                       # the index first; the directory name is convenience
        target = sub.with_name(m.dir_name)
        if target != sub:
            for attempt in range(3):
                try:
                    sub.rename(target)
                    break
                except PermissionError:
                    if attempt == 2:
                        break                         # tolerated: the manifest is what resolves
                    time.sleep(0.1 * (attempt + 1))
        return m

    def set_note(self, kind: str, ref: str, note: str) -> mf.Manifest:
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}")
        m.note = str(note)
        _write_manifest(sub, m)
        return m

    def dependents(self, kind: str, id_: str) -> list:
        """``[(kind, id, name)]`` of every artifact whose parents name this one."""
        out = []
        for k in KIND_DIRS:
            for _, m, _ in self._entries(k):
                if m is not None and any(m.parents.get(pk) == id_ for pk in _PARENT_KEYS.get(kind, ())):
                    out.append((k, m.id, m.name))
        return out

    def delete(self, kind: str, ref: str, *, force: bool = False) -> None:
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}")
        deps = self.dependents(kind, m.id)
        if deps and not force:
            listed = "; ".join(f"{k} {n or '(unnamed)'} [{i}]" for k, i, n in deps)
            raise StoreError(
                f"refusing to delete {kind} {m.name or m.id}: {len(deps)} artifact(s) name it as a "
                f"parent -- {listed}. Delete those first, or pass force=True to orphan them.")
        _rmtree_retry(sub)

    def unnamed(self, kind: str, *, older_than: "timedelta | None" = None) -> list:
        now = self._clock()
        out = []
        for s in self.list(kind):
            if s.name or not s.complete:
                continue
            if older_than is not None and datetime.fromisoformat(s.created) > now - older_than:
                continue
            out.append(s)
        return out

    def find_prior_by_fingerprint(self, fingerprint: "str | None"):
        if not fingerprint:
            return None
        for s in self.list("prior"):
            if s.complete and self.get("prior", s.id).fingerprints.get("gmm") == fingerprint:
                return s
        return None

    def simulation_for(self, identity: dict):
        from core.SBI.training_checkpoint import identity_digest, peek
        d = self.kind_dir("simulation") / identity_digest(identity)
        return d if peek(d) else None

    # ── loaders ──────────────────────────────────────────────────────────────────────────────────
    def load_prior(self, cfg, ref: str) -> LoadedPrior:
        """Refuses model, ND parameter set/order, box and log-mask mismatches BEFORE reading the
        payload -- the GMM is fit in its box's own coordinate, so a prior is meaningful only against
        the exact (model, parameter set + ORDER, box) it was built for -- then asserts the payload's
        GMM is the one the manifest fingerprinted."""
        import torch
        from core import orchestrator as _orch
        from core.SBI.reparam import nd_log_mask
        from core.SBI.run_guards import _gmm_fingerprint, _log_params_for
        sub, m = self._find("prior", ref)
        if m is None:
            raise StoreError(f"no complete prior named or id'd {ref!r} under {self.kind_dir('prior')}")
        label = m.name or m.id
        gmm = m.body["gmm"]

        def _bad(what, got, want):
            raise ValueError(
                f"Prior '{label}' does not match this configuration: {what} differs.\n"
                f"  prior:  {got}\n  config: {want}\n"
                f"A prior's GMM is fit in its own box coordinate, so loading it here would train the "
                f"flow against a different distribution than the one the samples came from. Build a new "
                f"prior for this bounds file, or pick the prior that belongs to it.")

        if m.config.get("model") != cfg.model:
            _bad("the model", m.config.get("model"), cfg.model)
        keys = list(cfg.params_dict)
        if list(gmm["param_keys"]) != keys:
            _bad("the ND parameter set or ORDER", list(gmm["param_keys"]), keys)
        want_lo = torch.tensor([b[0] for _, b in cfg.params_dict.values()], dtype=torch.float64)
        want_hi = torch.tensor([b[1] for _, b in cfg.params_dict.values()], dtype=torch.float64)
        got_lo = torch.tensor(gmm["box"]["nd_lows"], dtype=torch.float64)
        got_hi = torch.tensor(gmm["box"]["nd_highs"], dtype=torch.float64)
        if not (torch.allclose(got_lo, want_lo) and torch.allclose(got_hi, want_hi)):
            diff = [f"{n}: prior ({lo:g}, {hi:g}) vs config ({wl:g}, {wh:g})"
                    for n, lo, hi, wl, wh in zip(keys, got_lo.tolist(), got_hi.tolist(),
                                                 want_lo.tolist(), want_hi.tolist()) if lo != wl or hi != wh]
            _bad("the ND box", "; ".join(diff), "the bounds file in use")
        want_mask = [bool(v) for v in nd_log_mask(cfg, log_params=_log_params_for(cfg)).tolist()]
        if [bool(v) for v in gmm["box"]["log_mask"]] != want_mask:
            _bad("the log-box mask (which ND parameters use a geometric box)", gmm["box"]["log_mask"], want_mask)
        nd_prior = file_manager.load_mix_dist(str(sub / "prior.pt"), device=cfg.hw.device)
        fp = _gmm_fingerprint(nd_prior)
        if fp != m.fingerprints.get("gmm"):
            raise StoreError(f"prior '{label}': prior.pt holds GMM {fp}, not the one the manifest records "
                             f"({m.fingerprints.get('gmm')}); the artifact is inconsistent")
        prior = _orch.ProductPrior(distributions=[nd_prior, _orch.build_rescale_prior(cfg)],
                                   dims=[len(cfg.params_dict), len(cfg.rescale_params)])
        return LoadedPrior("prior", m.id, m.name, m, sub, prior=prior,
                           force_prior=_orch.build_forcing_prior(cfg), nd_prior=nd_prior, fingerprint=fp)

    def load_posterior(self, cfg, ref: str, *, accept: "Accept | None" = None) -> LoadedPosterior:
        """Every check _assert_mode_matches and build_posterior's load branch made, read from the
        manifest instead of a sidecar, BEFORE the payload is unpickled; then the payload's own view of
        its mode, the bijection rebuilt from the manifest, its rotation checked against the prior
        pickled inside the posterior, and the amortization gate."""
        import torch
        from sbi.inference import DirectPosterior
        from core import config as _config
        from core.SBI import reparam, truncate
        from core.SBI.run_guards import _assert_chi_config_is_deliberate, _gmm_fingerprint
        from core.SBI.statistics import SUMMARY_WIDTH
        from core.orchestrator import expected_forcing_dim
        accept = accept or Accept()
        sub, m = self._find("posterior", ref)
        if m is None:
            raise StoreError(f"no complete posterior named or id'd {ref!r} under {self.kind_dir('posterior')}")
        label = m.name or m.id
        # config.py is the one party that cannot go stale: a stale cfg agrees with the posterior
        # trained under that same stale cfg (the 2026-08-19 band).
        _assert_chi_config_is_deliberate(cfg)
        body, cond, tr = m.body, m.body["conditioning"], m.body["transform"]

        def _bad(msg):
            raise ValueError(f"Posterior '{label}' {msg}")

        if m.config.get("model") != cfg.model:
            _bad(f"was trained for model {m.config.get('model')}, but this config is for {cfg.model}.")
        want_keys = list(cfg.params_dict) + list(cfg.rescale_params)
        if list(tr["param_keys"]) != want_keys:
            _bad(f"was trained over a different inferred parameter set or ORDER.\n  posterior: {list(tr['param_keys'])}"
                 f"\n  config:    {want_keys}\nColumns bind positionally, so every reported value would refer "
                 f"to the wrong parameter. Pick the bounds file this posterior was trained with.")
        want_lo = [float(b[0]) for _, b in cfg.params_dict.values()] + [float(b[0]) for _, b in cfg.rescale_params.values()]
        want_hi = [float(b[1]) for _, b in cfg.params_dict.values()] + [float(b[1]) for _, b in cfg.rescale_params.values()]
        got_lo, got_hi = list(tr["nd_lows"]) + list(tr["rescale_lows"]), list(tr["nd_highs"]) + list(tr["rescale_highs"])
        if not (torch.allclose(torch.tensor(got_lo), torch.tensor(want_lo)) and
                torch.allclose(torch.tensor(got_hi), torch.tensor(want_hi))):
            diff = [f"{n}: posterior ({lo:g}, {hi:g}) vs config ({wl:g}, {wh:g})"
                    for n, lo, hi, wl, wh in zip(want_keys, got_lo, got_hi, want_lo, want_hi) if lo != wl or hi != wh]
            _bad(f"was trained in a different box than this config declares: {'; '.join(diff)}. The flow "
                 f"decodes every latent sample through its training box, so pick the bounds file it was trained with.")
        want_mode, want_dim = cfg.observation_mode, int(expected_forcing_dim(cfg))
        if body["mode"] != want_mode:
            _bad(f"was trained in {str(body['mode']).upper()} mode, but this config is {want_mode.upper()} mode "
                 f"(the chi toggle and the bounds file's Forcing section are what select the mode).")
        if body["mode"] == "chi":
            if cond["chi_layout"] != _config.CHI_LAYOUT:
                _bad(f"was trained under chi layout {cond['chi_layout']}; this build writes layout "
                     f"{_config.CHI_LAYOUT} (a padded probe set, {_config.CHI_ELEM_W} channels per slot). Retrain.")
            for key, want, what in (("chi_k_pad", int(cfg.chi_k_pad), "probe-slot capacity"),
                                    ("chi_elem_w", int(_config.CHI_ELEM_W), "channels per slot")):
                if int(cond[key]) != want:
                    _bad(f"has {what} {cond[key]}, but this config declares {want}. It is frozen into the "
                         f"trained network's input shape, so retrain or set {key} back to {cond[key]}.")
            if [float(v) for v in cond["chi_freq_bounds"]] != [float(v) for v in cfg.chi_freq_bounds]:
                _bad(f"was trained over chi band {tuple(cond['chi_freq_bounds'])}, but this config declares "
                     f"{tuple(cfg.chi_freq_bounds)}. The band fixes the encoder's frequency normalization.")
            if abs(float(cond["chi_max_cycles"]) - float(cfg.chi_max_cycles)) > 1e-9:
                _bad(f"was trained with a {float(cond['chi_max_cycles']):g}-cycle lock-in ceiling, but this "
                     f"config declares {float(cfg.chi_max_cycles):g}; logcyc is how the encoder weighs a probe.")
        if int(cond["forcing_dim"]) != want_dim or int(cond["width"]) != SUMMARY_WIDTH + 1 + want_dim:
            _bad(f"conditions on {cond['width']} features (forcing/chi block {cond['forcing_dim']}), but this "
                 f"config expects {SUMMARY_WIDTH + 1 + want_dim} (block {want_dim}). Conditioning widths are "
                 f"incompatible.")

        latent = torch.load(str(sub / "posterior.pt"), map_location=cfg.hw.device, weights_only=False)
        if not isinstance(latent, DirectPosterior):
            raise StoreError(f"posterior '{label}': posterior.pt is a {type(latent).__name__}, not a DirectPosterior")
        latent.device = latent._device = cfg.hw.device     # sbi caches the training device in both
        try:                                                # the trained net's own view must agree
            net_mode, net_dim, _ = reparam.posterior_mode(latent, None)
        except ValueError:
            net_mode = net_dim = None                       # a payload with no estimator (a stub): the manifest rules
        if net_mode is not None and (net_mode != body["mode"] or int(net_dim) != int(cond["forcing_dim"])):
            raise StoreError(f"posterior '{label}': the trained network is {net_mode} with a {net_dim}-wide block "
                             f"but the manifest says {body['mode']} / {cond['forcing_dim']}; the artifact is inconsistent")
        T = reparam.build_eval_bijection(cfg, tr)
        reparam.assert_rotation_consistent(T, getattr(latent, "prior", None), name=label)
        region = digest = None
        if not body["amortized"]:
            trd = body["truncation"] or {}
            if not accept.truncated:
                raise ValueError(
                    f"Posterior '{label}' is NOT AMORTIZED: it was trained by TSNPE on a prior truncated to a "
                    f"{trd.get('level', '?')}-HPD region along Fisher direction(s) {trd.get('dims')}, drawn around "
                    f"the observation with digest {trd.get('x_obs_digest')}. It is only valid for observations in "
                    f"that region -- outside it the flow has never seen a training row and will extrapolate "
                    f"confidently rather than return the prior. Pass Accept(truncated=True) to load it anyway "
                    f"(the Posterior tab does): its region then restricts calibration, and inference refuses any "
                    f"other observation unless told to accept it.")
            region = truncate.TruncationRegion.from_dict(mf.region_from_json(trd)) if trd else None
            if region is None or region.probe is None:
                _bad("declares itself NON-AMORTIZED but its manifest carries "
                     + ("no truncation region" if region is None else "a region without its basis")
                     + ", so the coordinate its box refers to cannot be verified. Run the round again.")
            region.check_basis(T, dim=len(want_keys), device=cfg.hw.device)
            digest = trd.get("x_obs_digest")
        post = reparam.TransformedPosterior(latent, T, truncation=region, x_obs_digest=digest)
        return LoadedPosterior("posterior", m.id, m.name, m, sub, posterior=post, latent=latent,
                               fingerprint=_gmm_fingerprint(getattr(latent, "prior", None)),
                               diagnostics=None, accepted=accept.used() if region is not None else [])


def write_simulation_manifest(path, identity: dict, *, parents=None, inputs=None, hw=None,
                              batches_done: int = 0, complete: bool = False, rows=None, V=None) -> mf.Manifest:
    """The simulation cache's manifest, written by training_checkpoint at create and refreshed on every
    save and at completion. Keeps id, created, note, prism, env, inputs and parents from an existing
    manifest (a resume must not rewrite who started the run); ``wall_seconds`` is the span from
    ``created`` to completion, across resumes."""
    from core.SBI.training_checkpoint import identity_digest
    path = Path(path)
    digest = identity_digest(identity)
    old = None
    mpath = path / MANIFEST
    if mpath.exists():
        with contextlib.suppress(mf.ManifestError, OSError):
            old = mf.from_json_text(mpath.read_text(encoding="utf-8"))
    created = old.created if old else _utc_now().isoformat()
    if complete:
        wall = (_utc_now() - datetime.fromisoformat(created)).total_seconds()
    else:
        wall = old.body.get("wall_seconds") if old else None
    v_digest = mf.tensor_digest(V) if V is not None else (old.fingerprints.get("V") if old else None)
    d = dict(
        schema=mf.SCHEMA, kind="simulation", id=digest, name="", created=created,
        note=old.note if old else "",
        prism=old.prism if old else prov.git_info(config.REPO_ROOT),
        env=old.env if old else prov.env_info(hw if hw is not None else config.cpu_device()),
        inputs=old.inputs if old else (inputs or {"bounds": None, "cell": None, "units": None,
                                                    "model": identity.get("model")}),
        config=dict(identity), parents=old.parents if old else dict(parents or {}),
        fingerprints={"gmm": identity.get("prior_fingerprint"), "V": v_digest},
        payloads={}, figures=[],
        body={"identity": dict(identity), "digest": digest, "batches_done": int(batches_done),
              "complete": bool(complete), "rows": None if rows is None else [int(r) for r in rows],
              "V_digest": v_digest, "wall_seconds": wall},
    )
    m = mf.validate(d)
    path.mkdir(parents=True, exist_ok=True)
    _write_manifest(path, m)
    return m


# ── the process default ──────────────────────────────────────────────────────────────────────────
_DEFAULT: "ArtifactStore | None" = None


def default_store() -> ArtifactStore:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ArtifactStore(config.artifacts_root())
    return _DEFAULT


def set_default_store(store: "ArtifactStore | None") -> None:
    global _DEFAULT
    _DEFAULT = store


@contextlib.contextmanager
def use_store(store: ArtifactStore):
    global _DEFAULT
    prev = _DEFAULT
    _DEFAULT = store
    try:
        yield store
    finally:
        _DEFAULT = prev


def resolve_store(store) -> ArtifactStore:
    """``store`` when given, else the process default -- the one line every stage function starts with."""
    return store if store is not None else default_store()
