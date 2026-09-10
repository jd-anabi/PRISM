# Artifact Store, Run Contract and Provenance — Implementation Plan (piece 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every generated artifact (prior, simulation cache, posterior, observation, calibration, inference) becomes a self-describing directory under `Artifacts/` with a `manifest.json`; every stage reads its parents by reference through one store, writes its own result at completion, and refuses on any verifiable mismatch.

**Architecture:** A new package `core/artifacts/` (manifest schema, provenance capture, the store and its writer, the simulation identity) sits under `core/orchestrator.py`. Stage functions take and return `Loaded*` wrappers that carry the manifest and id. The `.rot.pt` sidecar, the `Resources/{Priors,Posteriors,Checkpoints,Observations,Plots}` path constants and the deferred-save machinery are removed. The simulation cache keeps its digest-named directory but moves under the store and gains a manifest written by `training_checkpoint`.

**Tech Stack:** Python 3.12, torch 2.9, sbi 0.25, PySide6 6.9.3, pytest 9.1 (conda env `biophys-env`).

**Spec:** `docs/superpowers/specs/2026-09-10-artifact-store-design.md` (parent: `2026-09-10-pre-retrain-hardening-decomposition.md`).

## Global Constraints

- Interpreter: `C:\Users\J\anaconda3\envs\biophys-env\python.exe`. Run tests with `pytest` from the repo root; the root `conftest.py` sets `QT_QPA_PLATFORM=offscreen` and `KMP_DUPLICATE_LIB_OK=TRUE`.
- Fast gate after every commit: `pytest -m "not slow"` must be green. Full gate (`pytest`, ~1 h) once, at Task 13.
- **Do not edit any source file while a suite is running** (several tests assert on `inspect.getsource`).
- `core/orchestrator.py` does `from .config import X` at import (a snapshot). Remove each generated-path name from that import in the task its last user disappears, never earlier and never later.
- All writes go through `core/Helpers/file_manager.py`: `_atomic_write`, `atomic_torch_save`, `atomic_savez`. `torch.load(..., weights_only=False)` for every pickle that holds a `DirectPosterior` or a dict of tensors.
- `pipeline.py`'s bottom-of-file re-imports stay; new code calls `pipeline.train_nn` / `pipeline.gen_prior` through the module attribute so the suites' monkeypatches still land.
- Clean break: no backward compatibility for anything under `Resources/{Priors,Posteriors,Checkpoints,Observations,Plots}`. Inputs (`Resources/{Bounds,Cells,Units,Models}`) are unchanged.
- Timestamp ids: UTC `YYYYMMDDTHHMMSS`, unique within a kind, `-2`, `-3` on collision; the simulation kind's id is its identity digest (12 hex).
- Names: `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`, unique within a kind; `""` = unnamed (directory `_unnamed__<id>`).
- Commit messages: one commit per task, message as given in the task, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Deviation from the spec text, deliberate: manifest bodies are validated **dicts** (key sets in `manifest.BODY_KEYS`), not per-kind dataclasses. Same contract, less code.

## File map

| file | responsibility |
|---|---|
| `core/artifacts/__init__.py` | re-exports |
| `core/artifacts/manifest.py` | schema constants, `Manifest`, JSON encoding of tensors, `tensor_digest`, `validate`, `config_from_cfg`, `conditioning_block` |
| `core/artifacts/provenance.py` | `git_info`, `env_info`, `file_ref`, `inputs_from_cfg`, `_nvidia_smi_free_gib` (moved from the GUI) |
| `core/artifacts/store.py` | `ArtifactStore`, `ArtifactWriter`, `Summary`, `Accept`, `Loaded*`, the loaders, `write_simulation_manifest`, `default_store` / `set_default_store` / `use_store` |
| `core/artifacts/identity.py` | `SimulationIdentity` (`training-rows/2`) |
| `core/SBI/reparam.py` | `build_eval_bijection`, `assert_rotation_consistent` (added); `read_sidecar`, `sidecar_path`, `load_eval_bijection`, `reconcile_loaded_rotation` (removed in Task 7) |
| `core/SBI/training_checkpoint.py` | root under the store, digest-only directory names, simulation manifest at create/save/complete |
| `core/orchestrator.py` | every stage function takes/returns wrappers and writes through the store |
| `core/SBI/observations.py` | defects 3–4 fixed; `RecordingSet` |
| `core/config.py`, `core/sim_config.py`, `core/cli.py` | `RESOURCES_ROOT`, `artifacts_root()`, `SimConfig.sources` |
| `core/gui/widgets/artifact_picker.py` | `StorePicker` |
| `core/gui/session.py`, `core/gui/panels/inference/{prior_tab,posterior_tab,validate_tab,infer_tab,tsnpe_tab,runners,base}.py`, `core/gui/app.py` | wrappers in the session, pickers, rename-on-save |
| `scripts/_common.py`, `scripts/smoke_train.py` | interim re-point |
| `tests/test_artifact_store.py` (new), `tests/_tiny.py` (new), `tests/conftest.py` | the store suite, the shared tiny fixtures, the session sandbox |

---

### Task 1: Manifest schema and provenance (additive)

**Files:**
- Create: `core/artifacts/__init__.py`, `core/artifacts/manifest.py`, `core/artifacts/provenance.py`
- Modify: `core/config.py:174-197`, `core/sim_config.py:25-92`, `core/cli.py:479-522` and `:230-240`, `core/SBI/statistics.py:521`, `core/gui/panels/inference/base.py:51-69`
- Test: `tests/test_artifact_store.py` (new)

**Interfaces:**
- Produces: `manifest.SCHEMA`, `KINDS`, `NAME_RE`, `ID_RE`, `UNNAMED_DIR`, `HEADER_KEYS`, `BODY_KEYS`, `ManifestError`, `Manifest` (dataclass, `to_dict()`, `dir_name`), `tensor_to_json(t)`, `json_to_tensor(x, dtype, device)`, `tensor_digest(t) -> str`, `validate(d) -> Manifest`, `to_json_text(m) -> str`, `from_json_text(s) -> Manifest`, `config_from_cfg(cfg) -> dict`, `conditioning_block(cfg) -> dict`; `provenance.git_info(root) -> dict`, `env_info(hw) -> dict`, `file_ref(path, relative_to=None) -> dict`, `inputs_from_cfg(cfg) -> dict`, `device_name(hw) -> str`, `_nvidia_smi_free_gib()`; `config.RESOURCES_ROOT`, `config.artifacts_root()`; `SimConfig.sources: dict`; `statistics.FEATURE_SET_VERSION = 1`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_artifact_store.py`:

```python
"""The artifact store (piece 1 of the 2026-09-10 hardening programme): manifest schema, provenance,
the store and its writer, the loaders' refusals, the simulation identity, and the stage contract."""
import json
import math
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from core import config
from core.artifacts import manifest as mf
from core.artifacts import provenance as prov


def test_manifest_json_round_trips_tensors_exactly():
    V = torch.linalg.qr(torch.randn(13, 13, dtype=torch.float64))[0]
    as_list = mf.tensor_to_json(V)
    assert isinstance(as_list, list) and isinstance(as_list[0][0], float)
    back = mf.json_to_tensor(json.loads(json.dumps(as_list)))
    assert torch.equal(back, V), "float64 must survive repr -> json -> float64 bit for bit"
    assert mf.tensor_digest(back) == mf.tensor_digest(V)
    assert len(mf.tensor_digest(V)) == 16


def _header(**over):
    d = dict(schema=mf.SCHEMA, kind="calibration", id="20260910T120000", name="cal", created="2026-09-10T12:00:00+00:00",
             note="", prism={"git_rev": "unknown", "git_dirty": None, "branch": "unknown"},
             env={}, inputs={"bounds": None, "cell": None, "units": None, "model": "X"},
             config={}, parents={"posterior": "20260910T110000", "prior": "20260910T100000"},
             fingerprints={}, payloads={}, figures=[], body={"results": {"n_cal": 10}})
    d.update(over)
    return d


def test_manifest_validation_refuses_missing_header_wrong_schema_bad_name_bad_id_and_nan():
    mf.validate(_header())                                                # the baseline is valid
    for bad, why in ((dict(schema=99), "schema"), (dict(kind="posteriors"), "kind"),
                     (dict(id="yesterday"), "id"), (dict(name="bad name!"), "name"),
                     (dict(body={"results": {"x": float("nan")}}), "finite"),
                     (dict(body={"wrong": 1}), "body"), (dict(extra=1), "unknown")):
        with pytest.raises(mf.ManifestError, match=why):
            mf.validate(_header(**bad))
    d = _header(); del d["parents"]
    with pytest.raises(mf.ManifestError, match="parents"):
        mf.validate(d)
    assert mf.validate(_header(id="20260910T120000-2")).id == "20260910T120000-2"
    assert mf.validate(_header(kind="simulation", id="0123456789ab",
                               body={k: None for k in mf.BODY_KEYS["simulation"]})).id == "0123456789ab"


def test_provenance_records_git_rev_and_dirty_and_unknown_outside_a_repo():
    here = prov.git_info(Path(__file__).resolve().parents[1])
    assert len(here["git_rev"]) == 40 and here["git_dirty"] in (True, False) and here["branch"]
    with tempfile.TemporaryDirectory() as tmp:
        out = prov.git_info(tmp)
    assert out == {"git_rev": "unknown", "git_dirty": None, "branch": "unknown"}


def test_env_records_versions_and_device():
    env = prov.env_info(config.cpu_device())
    for key in ("python", "torch", "sbi", "PySide6", "numpy", "os", "hostname", "device_type", "device_name", "dtype"):
        assert key in env and env[key], key
    assert env["device_type"] == "cpu"


def test_config_root_does_not_depend_on_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert config.RESOURCES_ROOT.name == "Resources" and (config.RESOURCES_ROOT / "Bounds").is_dir()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    assert config.artifacts_root() == tmp_path / "A"
    monkeypatch.delenv("PRISM_ARTIFACTS")
    assert config.artifacts_root() == config.RESOURCES_ROOT.parent / "Artifacts"


def test_sim_config_records_its_sources_and_the_feature_set_is_versioned():
    from core import cli, registry
    from core.config import BOUNDS_PATH, CELL_PATH, VALID_LABELS, VALID_MODELS
    from core.SBI.statistics import FEATURE_SET_VERSION
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / "nadrowski" / "master.txt"))
    assert cfg.sources["bounds"].endswith("master.txt") and cfg.sources["units"].endswith("units.txt")
    cli.load_and_validate_gt(cfg, str(CELL_PATH / "nadrowski" / "master_spont.txt"))
    assert cfg.sources["cell"].endswith("master_spont.txt")
    inputs = prov.inputs_from_cfg(cfg)
    assert inputs["model"] == "NADROWSKI" and len(inputs["bounds"]["sha256"]) == 64
    assert inputs["bounds"]["path"] == "Bounds/nadrowski/master.txt"       # relative to Resources/
    assert isinstance(FEATURE_SET_VERSION, int) and FEATURE_SET_VERSION >= 1
    block = mf.conditioning_block(cfg)
    assert block["feature_set_version"] == FEATURE_SET_VERSION and block["width"] == block["input_dim"] + block["forcing_dim"]
    c = mf.config_from_cfg(cfg)
    assert c["model"] == "NADROWSKI" and c["param_keys"][0] == list(cfg.params_dict)[0]
    assert math.isfinite(c["dt_exp"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_artifact_store.py -v`
Expected: `ModuleNotFoundError: No module named 'core.artifacts'`

- [ ] **Step 3: Add the roots and `SimConfig.sources`**

In `core/config.py` replace lines 174-197 (`# === PATHS ===` through `MODELS_PATH`) with:

```python
# === PATHS ===
# Inputs (the hand-edited Bounds / Cells / Units / Models) live at <repo-root>/Resources, resolved
# from THIS FILE'S location -- never from the working directory, which used to decide it and moved
# the store with whatever directory the app was launched from. PRISM_RESOURCES overrides.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_ROOT = Path(os.environ.get("PRISM_RESOURCES") or (REPO_ROOT / "Resources"))
CELL_PATH    = RESOURCES_ROOT / "Cells"
BOUNDS_PATH  = RESOURCES_ROOT / "Bounds"
UNITS_PATH   = RESOURCES_ROOT / "Units"
MODELS_PATH  = RESOURCES_ROOT / "Models"      # user-defined model definitions (see core/registry.py)


def artifacts_root() -> Path:
    """Where generated artifacts live: PRISM_ARTIFACTS, else <repo-root>/Artifacts. A FUNCTION, read
    at every call, so a test or a script can point one process at a sandbox without rebinding
    module names (the `from .config import X` snapshot trap)."""
    return Path(os.environ.get("PRISM_ARTIFACTS") or (REPO_ROOT / "Artifacts"))


# Generated-kind constants. REMOVED in piece 1's Task 12; until then every remaining reader is a
# stage that has not yet moved to the store.
_ROOT = RESOURCES_ROOT
PRIOR_PATH   = _ROOT / "Priors"
POSTERIOR_PATH = _ROOT / "Posteriors"
PLOT_PATH    = _ROOT / "Plots"
CHECKPOINT_PATH = _ROOT / "Checkpoints"
OBSERVATION_PATH = _ROOT / "Observations"
```

In `core/sim_config.py`, after the `hw: DeviceConfig = field(default_factory=detect_device)` line (L92) add:

```python
    # The files this config was built from, as paths: {"bounds": ..., "units": ..., "cell": ...}.
    # Filled by cli.make_sim_config (bounds, units) and cli.load_and_validate_gt (cell); a hand-entered
    # bounds grid leaves "bounds" absent. Read by core.artifacts.provenance.inputs_from_cfg so every
    # manifest names its inputs by path AND content hash.
    sources: dict = field(default_factory=dict)
```

In `core/cli.py` `make_sim_config`: capture the units file and pass sources. Change

```python
    if units_override is None:
        units_dict = file_manager.parse_units_file(resolve_units_file(model))
```
to
```python
    units_path = None
    if units_override is None:
        units_path = resolve_units_file(model)
        units_dict = file_manager.parse_units_file(units_path)
```
and in the branch `elif isinstance(units_override, (str, Path)):` add `units_path = str(units_override)` before the parse. Then in the `return SimConfig(` call add the keyword
```python
        sources={k: str(v) for k, v in (("bounds", bounds_file), ("units", units_path)) if v},
```
In `load_and_validate_gt` (L230-240) add, before the `return`:
```python
    cfg.sources["cell"] = str(cell_path)
```

In `core/SBI/statistics.py`, directly above `VALID_FLAG_LABELS = [` (L521) add:

```python
# Bumped whenever the DEFINITION of any summary feature changes (a fix like section 7.6's FWHM, a
# new feature, a changed winsor rule). It rides in the simulation identity, so a cache of
# conditioning rows computed under the old definition is never resumed onto: only the flag LABELS
# were digested before, and a changed definition behind an unchanged label re-keyed nothing.
FEATURE_SET_VERSION = 1
```

- [ ] **Step 4: Write `core/artifacts/manifest.py`**

```python
"""Manifest schema for the artifact store (piece 1 of the 2026-09-10 hardening programme).

One JSON file per artifact: a common header (who wrote it, from what, with which knobs, from which
parents) and a kind-specific body. Small tensors are float64 lists -- Python's repr is
shortest-round-trip, so float64 survives exactly; anything large stays binary beside the manifest
with its sha256 in ``payloads``. The manifest is written LAST and atomically by the writer, so a
directory without one is incomplete by definition.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

SCHEMA = 1
KINDS = ("prior", "simulation", "posterior", "observation", "calibration", "inference")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# A UTC second, with -2/-3 on a same-second collision; or the simulation kind's 12-hex identity digest.
ID_RE = re.compile(r"^(\d{8}T\d{6}(-\d+)?|[0-9a-f]{12})$")
UNNAMED_DIR = "_unnamed"

HEADER_KEYS = ("schema", "kind", "id", "name", "created", "note", "prism", "env", "inputs", "config",
               "parents", "fingerprints", "payloads", "figures", "body")
BODY_KEYS = {
    "prior": ("gmm", "sweep", "stability"),
    "simulation": ("identity", "digest", "batches_done", "complete", "rows", "V_digest", "wall_seconds"),
    "posterior": ("mode", "conditioning", "transform", "amortized", "truncation", "training"),
    "observation": ("mode", "conditioning", "x_obs_digest", "T_obs_cell", "n_obs", "forcing_vals",
                    "chi_obs_freqs", "source"),
    "calibration": ("results",),
    "inference": ("results",),
}


class ManifestError(ValueError):
    """A manifest that does not describe a current PRISM artifact."""


@dataclass
class Manifest:
    schema: int
    kind: str
    id: str
    name: str
    created: str
    note: str
    prism: dict
    env: dict
    inputs: dict
    config: dict
    parents: dict
    fingerprints: dict
    payloads: dict
    figures: list
    body: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def dir_name(self) -> str:
        """``<name>__<id>`` (``_unnamed__<id>`` when unnamed); the simulation kind is just its digest."""
        if self.kind == "simulation":
            return self.id
        return f"{self.name or UNNAMED_DIR}__{self.id}"


def tensor_to_json(t):
    """A tensor (or None) as nested float64 lists."""
    if t is None:
        return None
    import torch
    return t.detach().cpu().to(torch.float64).tolist()


def json_to_tensor(x, dtype=None, device=None):
    """The inverse of tensor_to_json; None stays None. Default dtype float64 (exact)."""
    if x is None:
        return None
    import torch
    return torch.tensor(x, dtype=dtype or torch.float64, device=device or "cpu")


def tensor_digest(t) -> str | None:
    """16-hex sha256 over a tensor's float64 bytes, or None. THE one hashing rule -- it is what
    run_guards._gmm_fingerprint, orchestrator.observation_digest and truncate.rotation_digest all
    spelled out separately."""
    if t is None:
        return None
    import hashlib
    import torch
    b = t.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def _check_finite(obj, where: str) -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ManifestError(f"{where}: a non-finite number cannot be recorded (got {obj!r})")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_finite(v, f"{where}[{i}]")


def validate(d: dict) -> Manifest:
    """A dict -> Manifest, refusing anything that is not a current, complete, finite manifest."""
    if not isinstance(d, dict):
        raise ManifestError(f"manifest must be a JSON object, got {type(d).__name__}")
    unknown = set(d) - set(HEADER_KEYS)
    if unknown:
        raise ManifestError(f"unknown manifest keys {sorted(unknown)}")
    missing = [k for k in HEADER_KEYS if k not in d]
    if missing:
        raise ManifestError(f"manifest is missing {missing}")
    if d["schema"] != SCHEMA:
        raise ManifestError(f"manifest schema {d['schema']!r} is not the current schema {SCHEMA}; "
                            f"not a current PRISM artifact")
    if d["kind"] not in KINDS:
        raise ManifestError(f"unknown artifact kind {d['kind']!r}")
    if not isinstance(d["id"], str) or not ID_RE.match(d["id"]):
        raise ManifestError(f"bad artifact id {d['id']!r}")
    if d["name"] != "" and not NAME_RE.match(str(d["name"])):
        raise ManifestError(f"bad artifact name {d['name']!r} (letters, digits, _ . - ; max 64)")
    for key in ("prism", "env", "inputs", "config", "parents", "fingerprints", "payloads", "body"):
        if not isinstance(d[key], dict):
            raise ManifestError(f"manifest field {key!r} must be an object")
    if not isinstance(d["figures"], list):
        raise ManifestError("manifest field 'figures' must be a list")
    want = set(BODY_KEYS[d["kind"]])
    if set(d["body"]) != want:
        raise ManifestError(f"{d['kind']} body keys are {sorted(d['body'])}, expected {sorted(want)}")
    _check_finite(d["body"], "body")
    _check_finite(d["config"], "config")
    return Manifest(**d)


def to_json_text(m: Manifest) -> str:
    return json.dumps(m.to_dict(), sort_keys=True, indent=2, allow_nan=False)


def from_json_text(text: str) -> Manifest:
    try:
        return validate(json.loads(text))
    except json.JSONDecodeError as e:
        raise ManifestError(f"manifest is not valid JSON: {e}") from None


def conditioning_block(cfg) -> dict:
    """The conditioning geometry a posterior/observation is frozen into; one derivation."""
    from core import config as _config
    from core.SBI.statistics import FEATURE_SET_VERSION, SUMMARY_WIDTH, VALID_FLAG_LABELS
    from core.orchestrator import expected_forcing_dim
    fdim = int(expected_forcing_dim(cfg))
    chi = bool(cfg.chi_mode)
    return {
        "input_dim": SUMMARY_WIDTH + 1, "forcing_dim": fdim, "width": SUMMARY_WIDTH + 1 + fdim,
        "chi_layout": _config.CHI_LAYOUT if chi else None,
        "chi_k_pad": int(cfg.chi_k_pad) if chi else None,
        "chi_elem_w": _config.CHI_ELEM_W if chi else None,
        "chi_n_freqs": int(cfg.chi_n_freqs) if chi else None,
        "chi_f0": float(cfg.chi_f0) if chi else None,
        "chi_freq_bounds": [float(v) for v in cfg.chi_freq_bounds] if chi else None,
        "chi_max_cycles": float(cfg.chi_max_cycles) if chi else None,
        "summary_flags": list(VALID_FLAG_LABELS),
        "feature_set_version": int(FEATURE_SET_VERSION),
    }


def config_from_cfg(cfg) -> dict:
    """The SimConfig-derived identity vocabulary every manifest's ``config`` starts from. Stages
    ``.update()`` their own resolved knobs on top."""
    from core.SBI.reparam import resolved_log_params
    from core.SBI.run_guards import _log_params_for
    return {
        "model": cfg.model,
        "mode": cfg.observation_mode,
        "param_keys": list(cfg.params_dict) + list(cfg.rescale_params),
        "nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
        "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
        "rescale_lows": [float(b[0]) for _, b in cfg.rescale_params.values()],
        "rescale_highs": [float(b[1]) for _, b in cfg.rescale_params.values()],
        "log_params": list(resolved_log_params(cfg, log_params=_log_params_for(cfg))),
        "reparam_rotate": bool(cfg.reparam_rotate),
        "state_dep_drift": bool(cfg.state_dep_drift),
        "steady_idx": int(cfg.steady_idx), "dt_nd_min": float(cfg.dt_nd_min),
        "dt_exp": float(cfg.dt_exp), "t_min_exp": float(cfg.t_min_exp), "t_max_exp": float(cfg.t_max_exp),
        "t_scale_bounds": [float(v) for v in cfg.t_scale_bounds], "n_grid": int(cfg.t.shape[0]),
        "spontaneous_only": not cfg.has_forcing,
        "chi_mode": bool(cfg.chi_mode),
        "chi_k_pad": int(cfg.chi_k_pad) if cfg.chi_mode else None,
        "chi_f0": float(cfg.chi_f0) if cfg.chi_mode else None,
        "chi_freq_bounds": [float(v) for v in cfg.chi_freq_bounds] if cfg.chi_mode else None,
        "chi_max_cycles": float(cfg.chi_max_cycles) if cfg.chi_mode else None,
        "units": list(cfg.units_dict) if isinstance(cfg.units_dict, (list, tuple)) else None,
        "device": cfg.hw.device.type, "dtype": str(cfg.hw.dtype),
    }
```

- [ ] **Step 5: Write `core/artifacts/provenance.py`**

```python
"""Who produced an artifact, from what, on which machine: git, package versions, input file hashes."""
from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
import sys
from pathlib import Path

UNKNOWN_GIT = {"git_rev": "unknown", "git_dirty": None, "branch": "unknown"}


def _git(args, root) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, timeout=5)
    except Exception:                        # noqa: BLE001 -- no git binary, a hung index, a timeout
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(root) -> dict:
    """``{git_rev, git_dirty, branch}`` for the repo containing ``root``; UNKNOWN_GIT outside one.
    Never raises: provenance must not be the thing that loses a multi-day run."""
    rev = _git(["rev-parse", "HEAD"], root)
    if not rev:
        return dict(UNKNOWN_GIT)
    status = _git(["status", "--porcelain"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root) or "unknown"
    return {"git_rev": rev, "git_dirty": None if status is None else bool(status), "branch": branch}


def device_name(hw) -> str:
    dev = getattr(hw, "device", None)
    if getattr(dev, "type", None) == "cuda":
        try:
            import torch
            return torch.cuda.get_device_name(dev)
        except Exception:                    # noqa: BLE001
            return "cuda"
    return str(getattr(dev, "type", dev) or "cpu")


def _version(mod: str) -> str:
    try:
        return str(__import__(mod).__version__)
    except Exception:                        # noqa: BLE001 -- an optional package that is absent
        return "absent"


def env_info(hw) -> dict:
    return {
        "python": sys.version.split()[0], "torch": _version("torch"), "sbi": _version("sbi"),
        "PySide6": _version("PySide6"), "numpy": _version("numpy"),
        "os": f"{platform.system()} {platform.release()}", "hostname": socket.gethostname(),
        "device_type": str(getattr(getattr(hw, "device", None), "type", "cpu")),
        "device_name": device_name(hw), "dtype": str(getattr(hw, "dtype", "unknown")),
    }


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_ref(path, relative_to=None) -> dict:
    """``{"path": <relative when under relative_to, else absolute>, "sha256": ...}``. Refuses a missing
    file: an artifact must never claim an input it could not read."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"input file not found: {p}")
    shown = str(p.resolve())
    if relative_to is not None:
        try:
            shown = p.resolve().relative_to(Path(relative_to).resolve()).as_posix()
        except ValueError:
            pass
    return {"path": shown, "sha256": sha256_file(p)}


def inputs_from_cfg(cfg) -> dict:
    """The bounds / cell / units files a config was built from (``SimConfig.sources``), hashed."""
    from core import config
    src = getattr(cfg, "sources", None) or {}
    out = {"model": cfg.model}
    for key in ("bounds", "cell", "units"):
        out[key] = file_ref(src[key], relative_to=config.RESOURCES_ROOT) if src.get(key) else None
    return out


def _nvidia_smi_free_gib() -> "float | None":
    """Free VRAM in GiB according to ``nvidia-smi``, or None if it cannot be read.

    DELIBERATELY NOT ``torch.cuda.mem_get_info``: that overstates free VRAM on Windows by roughly the
    size of the desktop (measured 15037 MiB against nvidia-smi's 5814 at the same instant), and it is
    the number that green-lit the batch which killed the first chi retrain.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2.0)
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:                        # noqa: BLE001 -- no driver, no binary, a timeout: all "unknown"
        return None
```

Create `core/artifacts/__init__.py`:

```python
"""The artifact store: one directory per generated artifact, a manifest inside, one root."""
from .manifest import Manifest, ManifestError, SCHEMA, KINDS, tensor_digest  # noqa: F401
from .provenance import git_info, env_info, file_ref, inputs_from_cfg  # noqa: F401
```
(the store and identity exports are appended in Tasks 2 and 3.)

In `core/gui/panels/inference/base.py` delete the `_nvidia_smi_free_gib` function (L51-69) and add at the imports `from core.artifacts.provenance import _nvidia_smi_free_gib  # noqa: F401` (the name stays importable: `tests/test_settings_persistence.py` reaches it as `inference_tabs._nvidia_smi_free_gib`). Remove the now-unused `import subprocess` if nothing else in the file uses it.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_artifact_store.py -v`
Expected: 6 passed.

- [ ] **Step 7: Run the fast gate and commit**

Run: `pytest -m "not slow" -q`
Expected: green (`test_settings_persistence` still finds `_nvidia_smi_free_gib`; the cwd test in `test_artifact_consistency` still passes because `RESOURCES_ROOT` resolves to the same tree).

```bash
git add core/artifacts core/config.py core/sim_config.py core/cli.py core/SBI/statistics.py core/gui/panels/inference/base.py tests/test_artifact_store.py
git commit -F - <<'MSG'
artifacts: manifest schema, provenance, artifacts_root (piece 1, Task 1). core/artifacts/{manifest,provenance}.py: the manifest header/body key sets with validation (schema, kind, id, name, finite floats, no unknown keys), float64-exact tensor encoding, the one tensor_digest, config_from_cfg and conditioning_block; git/env/file-hash capture that never raises. config.RESOURCES_ROOT resolves from __file__ (PRISM_RESOURCES override) and artifacts_root() from PRISM_ARTIFACTS, never the working directory; the five generated-kind constants stay until Task 12. SimConfig.sources records the bounds/units/cell files a config was built from. statistics.FEATURE_SET_VERSION = 1. _nvidia_smi_free_gib moves to provenance (the GUI re-imports it).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 2: The store and its writer (no loaders yet)

**Files:**
- Create: `core/artifacts/store.py`
- Modify: `core/artifacts/__init__.py`, `core/gui/widgets/artifact_picker.py` (append `StorePicker`), `tests/conftest.py` (the `store` fixture and the session sandbox)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Consumes: everything Task 1 produced; `config.REPO_ROOT`, `config.artifacts_root()`, `config.cpu_device()`; `file_manager._atomic_write`; `visualizers.save_figure`; `training_checkpoint.identity_digest`, `.peek`.
- Produces: `store.KIND_DIRS`, `MANIFEST`, `StoreError`, `Accept(truncated, other_observation).used()`, `Summary(kind, id, name, created, note, path, complete, reason, mode, width, amortized, parents).label`, `slug(title)`, `ArtifactWriter` (`payload(filename)->Path`, `figure_path(title)->Path`, `fig_sink(forward=None)`, attributes `config`, `parents`, `fingerprints`, `body`, `id`, `dir`, `manifest`), `ArtifactStore(root, *, clock=None)` with `kind_dir`, `list`, `get`, `path`, `create`, `rename`, `set_note`, `dependents`, `delete`, `unnamed`, `find_prior_by_fingerprint`, `simulation_for`; `write_simulation_manifest(path, identity, *, parents, inputs, hw, batches_done, complete, rows, V)`; the wrappers `Loaded`, `LoadedPrior`, `LoadedPosterior`, `LoadedObservation`, `LoadedCalibration`, `LoadedInference`; `default_store()`, `set_default_store(store)`, `use_store(store)`, `resolve_store(store)`; `StorePicker(kind, allow_new=False, store=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/conftest.py`:

```python
@pytest.fixture(scope="session", autouse=True)
def _sandbox_default_store(tmp_path_factory):
    """Every stage now WRITES its artifact, so the whole session runs against a temp root; the real
    Artifacts/ must gain nothing. Replaces the per-suite PERSIST_OBSERVATIONS / _CKPT_DIRS_AT_IMPORT
    sandboxing the runners used to carry."""
    from core import config
    from core.artifacts import ArtifactStore, set_default_store
    real = config.artifacts_root()
    before = {p for p in real.rglob("*")} if real.exists() else set()
    set_default_store(ArtifactStore(tmp_path_factory.mktemp("artifacts")))
    yield
    after = {p for p in real.rglob("*")} if real.exists() else set()
    assert after == before, f"the suite wrote into {real}: {sorted(map(str, after - before))[:5]}"


@pytest.fixture
def store(tmp_path):
    """A fresh store that is ALSO the process default for the duration of the test."""
    from core.artifacts import ArtifactStore, use_store
    with use_store(ArtifactStore(tmp_path / "Artifacts")) as s:
        yield s
```

Append to `tests/test_artifact_store.py`:

```python
from datetime import datetime, timedelta, timezone

from core.artifacts import store as st


def _cal_body(n=10):
    return {"results": {"n_cal": n}}


def _make(store, kind="calibration", name="", body=None, parents=None, note=""):
    with store.create(kind, None, name=name, note=note) as w:
        w.body = body if body is not None else _cal_body()
        w.parents = dict(parents or {})
        p = w.payload("results.json")
        p.write_text("{}", encoding="utf-8")
    return w


def test_create_list_get_round_trip_per_kind(store):
    bodies = {
        "prior": {"gmm": {"n_components": 2, "param_keys": ["a"], "box": {"nd_lows": [0.0], "nd_highs": [1.0], "log_mask": [False]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}},
        "posterior": {"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                      "truncation": None, "training": {}},
        "observation": {"mode": "chi", "conditioning": {"width": 50}, "x_obs_digest": "0" * 16, "T_obs_cell": 1.0,
                        "n_obs": 10, "forcing_vals": {}, "chi_obs_freqs": None, "source": {"kind": "simulated"}},
        "calibration": _cal_body(), "inference": {"results": {"n_samples": 5}},
    }
    for kind, body in bodies.items():
        w = _make(store, kind, name=f"n_{kind}", body=body)
        rows = store.list(kind)
        assert [r.id for r in rows] == [w.id] and rows[0].complete and rows[0].name == f"n_{kind}"
        m = store.get(kind, w.id)
        assert m.to_dict() == store.get(kind, f"n_{kind}").to_dict() and m.body == body
        assert store.path(kind, w.id) == w.dir and m.payloads["results.json"] == prov.sha256_file(w.dir / "results.json")
        assert m.prism["git_rev"] != "" and m.env["python"]
    assert store.list("posterior")[0].width == 50 and store.list("posterior")[0].amortized is True


def test_writer_removes_the_directory_on_exception(store):
    class _Cancel(BaseException):           # the shape of gui.streams.WorkerCancelled
        pass
    with pytest.raises(_Cancel):
        with store.create("calibration", None) as w:
            w.payload("results.json").write_text("{}")
            raise _Cancel()
    assert not w.dir.exists() and store.list("calibration") == []


def test_a_truncated_manifest_is_listed_incomplete_and_never_loadable(store):
    w = _make(store, name="ok")
    (w.dir / st.MANIFEST).write_bytes((w.dir / st.MANIFEST).read_bytes()[:40])
    rows = store.list("calibration")
    assert len(rows) == 1 and not rows[0].complete and "manifest" in rows[0].reason
    with pytest.raises(st.StoreError, match="no complete"):
        store.get("calibration", "ok")
    (w.dir / st.MANIFEST).unlink()
    assert not store.list("calibration")[0].complete


def test_same_second_ids_get_a_suffix(tmp_path):
    fixed = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    s = st.ArtifactStore(tmp_path, clock=lambda: fixed)
    ids = [_make(s).id for _ in range(3)]
    assert ids == ["20260910T120000", "20260910T120000-2", "20260910T120000-3"]
    assert {r.id for r in s.list("calibration")} == set(ids)


def test_rename_keeps_the_id_and_dependents_resolve(store):
    p = _make(store, "posterior", name="post", body={"mode": "chi", "conditioning": {}, "transform": {},
                                                     "amortized": True, "truncation": None, "training": {}})
    c = _make(store, "calibration", name="cal", parents={"posterior": p.id})
    store.rename("posterior", "post", "post_v2")
    m = store.get("posterior", "post_v2")
    assert m.id == p.id and store.path("posterior", p.id).name == f"post_v2__{p.id}"
    assert store.get("calibration", c.id).parents["posterior"] == m.id
    assert store.dependents("posterior", p.id) == [("calibration", c.id, "cal")]
    with pytest.raises(st.StoreError, match="already exists"):
        _make(store, "posterior", name="post_v2", body=m.body)
    with pytest.raises(st.StoreError):
        store.rename("posterior", p.id, "bad name")
    store.set_note("posterior", p.id, "kept")
    assert store.get("posterior", p.id).note == "kept"


def test_delete_refuses_naming_dependents_and_force_deletes(store):
    p = _make(store, "posterior", name="post", body={"mode": "chi", "conditioning": {}, "transform": {},
                                                     "amortized": True, "truncation": None, "training": {}})
    _make(store, "inference", name="inf", body={"results": {}}, parents={"posterior": p.id})
    with pytest.raises(st.StoreError, match="inference inf"):
        store.delete("posterior", p.id)
    assert store.get("posterior", p.id)
    store.delete("posterior", p.id, force=True)
    with pytest.raises(st.StoreError):
        store.get("posterior", p.id)


def test_unnamed_artifacts_group_and_age_out(tmp_path):
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    clock = {"now": t0}
    s = st.ArtifactStore(tmp_path, clock=lambda: clock["now"])
    old = _make(s)
    clock["now"] = t0 + timedelta(days=3)
    new = _make(s)
    named = _make(s, name="keep")
    assert {r.id for r in s.unnamed("calibration")} == {old.id, new.id}
    assert [r.id for r in s.unnamed("calibration", older_than=timedelta(days=1))] == [old.id]
    assert s.list("calibration")[0].label.startswith("(unnamed") or named.name == "keep"


def test_store_fig_sink_saves_png_and_forwards(store):
    from matplotlib import pyplot as plt
    seen = []
    with store.create("calibration", None) as w:
        w.body = _cal_body()
        sink = w.fig_sink(forward=lambda title, fig: seen.append((title, fig.number)))
        f1 = plt.figure(); sink("SBC ranks (CDF)", f1)
        f2 = plt.figure(); sink("SBC ranks (CDF)", f2)
        assert plt.fignum_exists(f1.number), "a forwarded figure is the GUI's to close"
        w.fig_sink()("TARP coverage", plt.figure())
    m = store.get("calibration", w.id)
    assert m.figures == ["figures/sbc_ranks_cdf.png", "figures/sbc_ranks_cdf-2.png", "figures/tarp_coverage.png"]
    assert all((w.dir / f).stat().st_size > 0 for f in m.figures) and len(seen) == 2
    plt.close("all")


def test_default_store_is_swappable_and_resolves_from_the_root(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "R"))
    st.set_default_store(None)
    assert st.default_store().root == tmp_path / "R"
    other = st.ArtifactStore(tmp_path / "O")
    with st.use_store(other):
        assert st.default_store() is other and st.resolve_store(None) is other
    assert st.default_store().root == tmp_path / "R"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_artifact_store.py -v`
Expected: the nine new tests fail with `ImportError: cannot import name 'store'` / `AttributeError`.

- [ ] **Step 3: Write `core/artifacts/store.py`**

```python
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
```

Replace `core/artifacts/__init__.py` with:

```python
"""The artifact store: one directory per generated artifact, a manifest inside, one root."""
from .manifest import Manifest, ManifestError, SCHEMA, KINDS, tensor_digest  # noqa: F401
from .provenance import git_info, env_info, file_ref, inputs_from_cfg  # noqa: F401
from .store import (ArtifactStore, ArtifactWriter, Accept, Summary, StoreError,  # noqa: F401
                    Loaded, LoadedPrior, LoadedPosterior, LoadedObservation, LoadedCalibration,
                    LoadedInference, write_simulation_manifest,
                    default_store, set_default_store, use_store, resolve_store)
```

- [ ] **Step 4: Add `StorePicker` to `core/gui/widgets/artifact_picker.py`**

Append after `ArtifactPicker`:

```python
class StorePicker(QWidget):
    """A combo over one KIND of the artifact store -- the generated-kind twin of ArtifactPicker,
    which stays for the input pickers (cells, bounds). Items are complete artifacts only, labelled by
    name (or ``(unnamed <id>)``), with the id, creation time, mode, width and amortization in the
    tooltip; ``userData`` is the id, which is what ``key()`` persists and ``selected()`` returns."""
    NEW_LABEL = ArtifactPicker.NEW_LABEL

    def __init__(self, kind: str, allow_new: bool = False, store=None, parent=None):
        super().__init__(parent)
        self.kind, self._allow_new, self._store = kind, allow_new, store
        self.combo = QComboBox()
        refresh = QPushButton()
        refresh.setObjectName("iconButton")
        icons.apply_icon(refresh, "refresh")
        refresh.setFixedWidth(32)
        refresh.setToolTip("Rescan the artifact store")
        refresh.clicked.connect(self.refresh)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)
        layout.addWidget(refresh)
        self.refresh()

    def _resolved_store(self):
        from core.artifacts import resolve_store
        return resolve_store(self._store)

    def refresh(self):
        current = self.key()
        self.combo.clear()
        if self._allow_new:
            self.combo.addItem(self.NEW_LABEL, userData=None)
        try:
            rows = self._resolved_store().list(self.kind)
        except Exception:                        # noqa: BLE001 -- an unreadable root lists nothing
            rows = []
        for s in rows:
            if not s.complete:
                continue
            self.combo.addItem(s.label, userData=s.id)
            tip = f"{s.id} · {s.created}"
            if s.mode:
                tip += f" · {s.mode}"
            if s.width:
                tip += f" · width {s.width}"
            if s.amortized is not None:
                tip += " · amortized" if s.amortized else " · NON-AMORTIZED (TSNPE)"
            self.combo.setItemData(self.combo.count() - 1, tip, _TOOLTIP_ROLE)
        self.restore_key(current)

    def selected(self):
        """``(id_or_None, is_new)``."""
        data = self.combo.currentData()
        return data, (data is None)

    def has_entries(self) -> bool:
        return any(self.combo.itemData(i) is not None for i in range(self.combo.count()))

    def key(self) -> str:
        data = self.combo.currentData()
        return "" if data is None else str(data)

    def restore_key(self, key: str) -> None:
        if not key:
            return
        i = self.combo.findData(key)
        if i >= 0:
            self.combo.setCurrentIndex(i)
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_artifact_store.py -v`
Expected: 15 passed.

- [ ] **Step 6: Run the fast gate and commit**

Run: `pytest -m "not slow" -q` — expected green (the session sandbox fixture is inert until a stage writes).

```bash
git add core/artifacts tests/conftest.py tests/test_artifact_store.py core/gui/widgets/artifact_picker.py
git commit -F - <<'MSG'
artifacts: store, writer, ids, rename/delete (piece 1, Task 2). ArtifactStore lists/gets/renames/deletes by reading manifests (a renamed directory still resolves; delete refuses naming dependents), ArtifactWriter hands out payload and figure paths and writes the manifest LAST atomically, removing the directory on any exception; timestamp ids with -2/-3 on a same-second collision; write_simulation_manifest for the cache training_checkpoint owns; Accept, the Loaded* wrappers, default_store/use_store. StorePicker is the generated-kind twin of ArtifactPicker (unused until Task 4). tests/conftest.py sandboxes the process default store for the whole session and asserts the real Artifacts/ gained nothing.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 3: The simulation identity and the cache under the store

**Files:**
- Create: `core/artifacts/identity.py`
- Modify: `core/SBI/training_checkpoint.py` (module docstring L28-32, `resolve_dir` L93-96, `create` L146-168, `_sibling_diffs` L215-238, `checkpoints_using_prior` L254-277, `save` L310-327, `mark_complete` L330-334), `core/orchestrator.py:363-439` (`training_identity` becomes a delegate), `core/SBI/pipeline.py:1414-1417` and `:1819`, `core/orchestrator.py:1072-1079` (the checkpoint dict gains `parents`, `inputs`, `hw`), `core/artifacts/__init__.py`, `tests/test_user_sbi.py` (L61-63, ~L1838-1870, L2247-2282)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Consumes: `store.write_simulation_manifest`, `store.default_store`, `manifest.tensor_digest`, `statistics.FEATURE_SET_VERSION`, `run_guards._gmm_fingerprint`, `_log_params_for`, `reparam.resolved_log_params`, `TruncationRegion.identity_fields()`.
- Produces: `identity.FORMAT = "training-rows/2"`, `identity.SimulationIdentity(values)` with `from_cfg(cfg, prior, run_size, n_runs, *, truncation=None)`, `to_dict()`, `digest`, `truncated`; `training_checkpoint.resolve_dir(identity, root=None)` → `<store>/simulations/<digest12>`; `create(..., parents=None, inputs=None, hw=None)`; `mark_complete(path, batch_k, rows=None)`; `orchestrator.training_identity` unchanged in signature, now a delegate.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_store.py`:

```python
def _nad_cfg(**over):
    from core import cli, registry
    from core.config import BOUNDS_PATH, VALID_LABELS, VALID_MODELS
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / "nadrowski" / "master.txt"), **over)
    cfg.hw = config.cpu_device()
    return cfg


_IDENTITY_KEYS = {
    "format", "model", "prior_fingerprint", "mode", "param_keys", "nd_lows", "nd_highs", "rescale_lows",
    "rescale_highs", "log_params", "reparam_rotate", "run_size", "n_runs", "steady_idx", "dt_nd_min",
    "dt_exp", "t_min_exp", "t_max_exp", "t_scale_bounds", "n_grid", "spontaneous_only", "summary_flags",
    "feature_set_version", "chi_mode", "chi_layout", "chi_k_pad", "chi_elem_w", "chi_f0", "chi_freq_bounds",
    "chi_max_cycles", "device", "dtype", "truncation"}


def test_identity_carries_truncation_always_and_feature_set_version_rekeys(monkeypatch):
    from core import orchestrator
    from core.artifacts.identity import FORMAT, SimulationIdentity
    from core.SBI import statistics, truncate
    from core.SBI.training_checkpoint import identity_digest
    cfg = _nad_cfg(chi_mode=True)
    ident = SimulationIdentity.from_cfg(cfg, object(), 2048, 5000)
    d = ident.to_dict()
    assert d["format"] == FORMAT and d["truncation"] is None and d["prior_fingerprint"] is None
    assert d["feature_set_version"] == statistics.FEATURE_SET_VERSION and set(d) == _IDENTITY_KEYS
    assert identity_digest(orchestrator.training_identity(cfg, object(), 2048, 5000)) == ident.digest
    monkeypatch.setattr(statistics, "FEATURE_SET_VERSION", statistics.FEATURE_SET_VERSION + 1)
    assert SimulationIdentity.from_cfg(cfg, object(), 2048, 5000).digest != ident.digest, \
        "a changed feature definition must re-key the cache"
    monkeypatch.undo()
    Q = torch.linalg.qr(torch.randn(13, 13, dtype=torch.float64))[0]
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], level=0.999, n_latent=13, V=Q)
    t = SimulationIdentity.from_cfg(cfg, object(), 2048, 5000, truncation=region)
    assert t.truncated and t.to_dict()["truncation"] == region.identity_fields() and t.digest != ident.digest
    assert {k for k in _IDENTITY_KEYS if t.to_dict()[k] != d[k]} == {"truncation"}

    class _LoadedLike:                       # a LoadedPrior supplies its fingerprint outright
        fingerprint = "f" * 16
    assert SimulationIdentity.from_cfg(cfg, _LoadedLike(), 2048, 5000).to_dict()["prior_fingerprint"] == "f" * 16


def test_simulation_directories_live_under_the_store_without_a_prefix(store):
    from core.SBI import training_checkpoint as tc
    ident = {"format": "training-rows/2", "n_runs": 3, "prior_fingerprint": "a" * 16, "truncation": None}
    d = tc.resolve_dir(ident)
    assert d.parent == store.kind_dir("simulation") and d.name == tc.identity_digest(ident)
    assert not d.name.startswith("train_")
    other = dict(ident, n_runs=4)
    od = tc.resolve_dir(other)
    (od / "shards").mkdir(parents=True)
    torch.save({"format": tc.CHECKPOINT_FORMAT, "identity": other, "V": None, "probe": None}, od / "header.pt")
    torch.save({"batches_done": 2, "complete": False, "rng": None}, od / "state.pt")
    assert tc.near_miss_siblings(ident)[0]["field"] == "n_runs"
    assert "n_runs" in tc.describe_siblings(ident)
    assert tc.checkpoints_using_prior("a" * 16) == [(od.name, 2)]


def test_simulation_manifest_written_at_create_and_refreshed_on_save_and_complete(store):
    from core.SBI import training_checkpoint as tc
    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "run_size": 4,
             "prior_fingerprint": "b" * 16, "truncation": None}
    d = tc.resolve_dir(ident)
    tc.create(d, ident, schedule_t_scales=torch.ones(3), schedule_Ts=torch.ones(3), inits=torch.zeros(1, 2),
              V=None, probe=torch.zeros(7, 2, dtype=torch.float64), run_size=4, n_runs=3,
              parents={"prior": "20260910T100000"}, inputs=None, hw=config.cpu_device())
    m = store.get("simulation", d.name)
    assert m.id == d.name and m.body["batches_done"] == 0 and m.body["complete"] is False
    assert m.parents == {"prior": "20260910T100000"} and m.fingerprints["gmm"] == "b" * 16
    tc.save(d, from_batch=0, batch_k=2, rng=None, x_buf=torch.zeros(12, 5), th_buf=torch.zeros(12, 2), run_size=4)
    m2 = store.get("simulation", d.name)
    assert m2.body["batches_done"] == 2 and m2.created == m.created
    tc.mark_complete(d, 3, rows=(12, 5))
    m3 = store.get("simulation", d.name)
    assert m3.body["complete"] is True and m3.body["rows"] == [12, 5] and m3.body["wall_seconds"] >= 0.0
    assert store.simulation_for(ident) == d and store.list("simulation")[0].complete
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_artifact_store.py -k "identity or simulation" -v`
Expected: 3 failures (`No module named 'core.artifacts.identity'`; `resolve_dir` returns `Resources/Checkpoints/train_…`; `create()` rejects `parents=`).

- [ ] **Step 3: Write `core/artifacts/identity.py`**

```python
"""The simulation cache's identity: every config field a checkpoint of training rows must agree
with before it can be resumed. Successor of ``orchestrator.training_identity`` (training-rows/1),
which the clean break retires along with every directory it named.

``truncation`` is ALWAYS present (None for an amortized run) -- the old dict omitted the key so that
existing digests would not move; there are no existing digests any more. ``feature_set_version``
is new: the old identity digested only the valid-flag LABELS, so a changed feature definition behind
an unchanged label re-keyed nothing and a cache could be resumed onto rows that meant something else.
"""
from __future__ import annotations

from dataclasses import dataclass

FORMAT = "training-rows/2"


@dataclass(frozen=True)
class SimulationIdentity:
    values: dict

    @classmethod
    def from_cfg(cls, cfg, prior, run_size: int, n_runs: int, *, truncation=None) -> "SimulationIdentity":
        """``prior`` is a LoadedPrior (its ``.fingerprint`` is used), a bare prior distribution (its GMM
        is fingerprinted), or anything else (None -- the GUI's pre-Train status line passes stubs and
        must fail open)."""
        from core import config as _config
        from core.SBI.reparam import resolved_log_params
        from core.SBI.run_guards import _gmm_fingerprint, _log_params_for
        from core.SBI.statistics import FEATURE_SET_VERSION, VALID_FLAG_LABELS
        fp = getattr(prior, "fingerprint", None)
        if fp is None:
            fp = _gmm_fingerprint(prior)
        chi = bool(cfg.chi_mode)
        vals = {
            "format": FORMAT,
            "model": cfg.model,
            "prior_fingerprint": fp,
            "mode": cfg.observation_mode,
            "param_keys": list(cfg.params_dict) + list(cfg.rescale_params),
            "nd_lows": [b[0] for _, b in cfg.params_dict.values()],
            "nd_highs": [b[1] for _, b in cfg.params_dict.values()],
            "rescale_lows": [b[0] for _, b in cfg.rescale_params.values()],
            "rescale_highs": [b[1] for _, b in cfg.rescale_params.values()],
            "log_params": resolved_log_params(cfg, log_params=_log_params_for(cfg)),
            "reparam_rotate": bool(cfg.reparam_rotate),
            "run_size": int(run_size),
            "n_runs": int(n_runs),
            "steady_idx": int(cfg.steady_idx),
            "dt_nd_min": float(cfg.dt_nd_min),
            "dt_exp": float(cfg.dt_exp),
            "t_min_exp": float(cfg.t_min_exp),
            "t_max_exp": float(cfg.t_max_exp),
            "t_scale_bounds": list(cfg.t_scale_bounds),
            "n_grid": int(cfg.t.shape[0]),
            "spontaneous_only": not cfg.has_forcing,
            "summary_flags": list(VALID_FLAG_LABELS),
            "feature_set_version": int(FEATURE_SET_VERSION),
            "chi_mode": chi,
            "chi_layout": _config.CHI_LAYOUT if chi else None,
            "chi_k_pad": cfg.chi_k_pad if chi else None,
            "chi_elem_w": _config.CHI_ELEM_W if chi else None,
            "chi_f0": cfg.chi_f0 if chi else None,
            "chi_freq_bounds": list(cfg.chi_freq_bounds) if chi else None,
            "chi_max_cycles": float(cfg.chi_max_cycles) if chi else None,
            "device": cfg.hw.device.type,
            "dtype": str(cfg.hw.dtype),
            "truncation": None if truncation is None else truncation.identity_fields(),
        }
        return cls(vals)

    def to_dict(self) -> dict:
        return dict(self.values)

    @property
    def digest(self) -> str:
        from core.SBI.training_checkpoint import identity_digest
        return identity_digest(self.values)

    @property
    def truncated(self) -> bool:
        return self.values.get("truncation") is not None
```

Add to `core/artifacts/__init__.py`: `from .identity import SimulationIdentity, FORMAT as IDENTITY_FORMAT  # noqa: F401`.

- [ ] **Step 4: Move the cache under the store**

In `core/SBI/training_checkpoint.py`:

Module docstring, replace the `LAYOUT.` block (L28-32) with:
```
LAYOUT.  ``<Artifacts>/simulations/<digest12>/``   (core.artifacts.store.KIND_DIRS["simulation"])
    manifest.json  the store's record: identity, parents, batches_done, complete, rows, wall time
    header.pt      write-once: identity + the (t_scale, T) schedule + inits + V + the bijection probe
    state.pt       rewritten atomically; its ``batches_done`` is THE COMMIT POINT
    state.prev.pt  one generation back, a few KB, for the case where state.pt is lost mid-write
    shards/x_<from>_<to>.pt, th_<from>_<to>.pt    write-once row blocks, never mutated after commit
```

Replace `resolve_dir`:
```python
def _default_root() -> Path:
    from core.artifacts.store import default_store   # lazy: the store imports this module lazily too
    return default_store().kind_dir("simulation")


def resolve_dir(identity: dict, root=None) -> Path:
    """The directory this identity's checkpoint lives in: ``<root>/<digest12>``. Pure; creates nothing.
    ``root`` defaults to the process default store's simulations directory."""
    root = Path(root) if root is not None else _default_root()
    return root / identity_digest(identity)


def _committed_dirs(root: Path):
    """Every directory under ``root`` that holds a checkpoint header, sorted."""
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / _HEADER).exists())
```

Change `create`'s signature and body:
```python
def create(path, identity: dict, *, schedule_t_scales, schedule_Ts, inits, V, probe,
           run_size: int, n_runs: int, parents=None, inputs=None, hw=None) -> None:
```
and after the `atomic_torch_save({"batches_done": 0, ...}, path / _STATE)` line add:
```python
    from core.artifacts.store import write_simulation_manifest
    write_simulation_manifest(path, identity, parents=parents, inputs=inputs, hw=hw,
                              batches_done=0, complete=False, V=V)
```

In `_sibling_diffs` replace `root = Path(root) if root is not None else config.CHECKPOINT_PATH` with `root = Path(root) if root is not None else _default_root()` and `for d in sorted(root.glob("train_*")):` with `for d in _committed_dirs(root):`. Same two replacements in `checkpoints_using_prior`. Delete the `if not root.is_dir(): return` lines that `_committed_dirs` now covers (keep the `if not fingerprint` check).

In `save`, after the final `atomic_torch_save({... "batches_done": int(batch_k) ...}, prev)` add:
```python
    _refresh_manifest(path, batches_done=batch_k)
```
Replace `mark_complete` with:
```python
def mark_complete(path, batch_k: int, rows=None) -> None:
    path = Path(path)
    st = peek(path) or {}
    atomic_torch_save({"batches_done": int(batch_k), "complete": True,
                       "rng": st.get("rng")}, path / _STATE)
    _refresh_manifest(path, batches_done=batch_k, complete=True, rows=rows)


def _refresh_manifest(path: Path, *, batches_done: int, complete: bool = False, rows=None) -> None:
    """Best-effort: the manifest is the store's view of the cache, never its commit point (that is
    state.pt). Read the identity back from the header rather than threading it through save()."""
    try:
        from core.artifacts.store import write_simulation_manifest
        header = read_header(path)
        write_simulation_manifest(path, header.get("identity", {}), batches_done=batches_done,
                                  complete=complete, rows=rows, V=header.get("V"))
    except Exception:                        # noqa: BLE001 -- never lose a committed batch over the index
        pass
```
Remove `from core import config` if nothing else in the module reads it (only `resolve_dir`/`_sibling_diffs`/`checkpoints_using_prior` did).

In `core/orchestrator.py` replace `training_identity` (L363-439, docstring and body) with:
```python
def training_identity(cfg: SimConfig, prior, run_size: int, n_runs: int, truncation=None) -> dict:
    """The simulation identity, as the dict training_checkpoint digests. A delegate to
    core.artifacts.identity.SimulationIdentity, kept under this name for the GUI's Posterior tab,
    which computes it before Train is pressed (the checkpoint line and the near-miss modal) and passes
    whatever it holds for a prior -- a LoadedPrior, or a stub that must fail open."""
    from .artifacts.identity import SimulationIdentity
    return SimulationIdentity.from_cfg(cfg, prior, run_size, n_runs, truncation=truncation).to_dict()
```
In `build_posterior`'s checkpoint dict (L1072-1079) add the three keys:
```python
            "V": V, "every": TRAINING_CHECKPOINT_EVERY, "resume": "auto",
            "parents": {"prior": getattr(prior, "id", None)} if getattr(prior, "id", None) else {},
            "inputs": _inputs_from_cfg(cfg), "hw": cfg.hw,
```
with `from .artifacts.provenance import inputs_from_cfg as _inputs_from_cfg` in the import block (a sibling package import; no cycle, `provenance` imports nothing from `core` at module level). Until Task 4 `prior` is a bare distribution, so `parents` is `{}`.

In `core/SBI/pipeline.py` L1414-1417 pass the new keys through:
```python
            _tc.create(_ck_dir, checkpoint["identity"],
                       schedule_t_scales=batch_t_scales, schedule_Ts=batch_Ts, inits=inits,
                       V=checkpoint.get("V"), probe=checkpoint.get("probe"),
                       run_size=run_size, n_runs=n_runs,
                       parents=checkpoint.get("parents"), inputs=checkpoint.get("inputs"),
                       hw=checkpoint.get("hw"))
```
and L1819: `_tc.mark_complete(_ck_dir, n_runs, rows=tuple(int(v) for v in x_buf.shape))`.

- [ ] **Step 5: Update `tests/test_user_sbi.py`**

1. Delete the `_CKPT_DIRS_AT_IMPORT = frozenset(...)` block (L57-63, including its comment) and the whole `test_the_suite_never_writes_checkpoints` function (starts ~L1838; it is the one that reads `_CKPT_DIRS_AT_IMPORT`). The successor is `conftest._sandbox_default_store`.
2. In the identity test (`test_checkpoint_identity_digest_is_stable_and_field_sensitive`, ~L2247-2282) change:
   - `assert "truncation" not in ia, "..."` → `assert ia["truncation"] is None, "an amortized identity records truncation=None"`;
   - the golden digest: run `pytest tests/test_user_sbi.py -k identity_digest -x` once, read the digest the assertion prints, and pin the new value in place of `"463e81d156cd"`, keeping the comment (the old digest belonged to `training-rows/1` directories the clean break deletes);
   - the key-set assertion: add `"format"` is already there; add `"feature_set_version"` and `"truncation"` to the set literal;
   - `assert {k: v for k, v in it.items() if k != "truncation"} == ia` → `assert {k: v for k, v in it.items() if k != "truncation"} == {k: v for k, v in ia.items() if k != "truncation"}`.
3. `grep -n '"train_\|train_\*\|startswith("train_' tests/*.py` — any remaining pin on the `train_` prefix follows the digest-only name.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_artifact_store.py tests/test_user_sbi.py -k "identity or simulation or checkpoint" -v`
Expected: green.

- [ ] **Step 7: Run the fast gate and commit**

Run: `pytest -m "not slow" -q` — expected green. `test_settings_persistence`'s checkpoint-line tests resolve through the sandboxed default store.

```bash
git add core/artifacts core/SBI/training_checkpoint.py core/SBI/pipeline.py core/orchestrator.py tests/test_artifact_store.py tests/test_user_sbi.py
git commit -F - <<'MSG'
identity: SimulationIdentity (training-rows/2), simulations under the store (piece 1, Task 3). core/artifacts/identity.py carries every field training_identity did, with truncation ALWAYS present and a new feature_set_version, so a changed feature definition re-keys the cache; orchestrator.training_identity is a delegate. training_checkpoint resolves <store>/simulations/<digest12> (no train_ prefix, root from the default store), writes the simulation manifest at create and refreshes it on every save and at completion (best-effort; state.pt stays the commit point); the sibling and prior scans iterate directories holding a header. The checkpoint dict gains parents/inputs/hw. test_user_sbi: the golden digest moves with the format, the amortized identity records truncation=None, and the suite's checkpoint-litter guard is replaced by conftest's session sandbox.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

Interim note: between this task and Task 13, `scripts/smoke_train.py`'s `CHECKPOINT=1` path no longer honours `CKPT_DIR` (it rebinds `config.CHECKPOINT_PATH`, which `resolve_dir` no longer reads); Task 13 re-points it.

### Task 4: The prior — `build_prior` auto-persists and returns a `LoadedPrior`; `store.load_prior`

**Files:**
- Modify: `core/artifacts/store.py` (add `load_prior`), `core/orchestrator.py` (`build_prior` L448-576; delete L292-354 and L1282-1343; imports L23-41; the two call sites L871-872 and L925-927; `run` L76-77), `core/SBI/run_guards.py` (delete `_assert_prior_matches` L116-161 and the now-unused imports), `core/gui/panels/inference/prior_tab.py`, `core/gui/panels/inference/runners.py:8-24`, `core/gui/panels/inference/infer_tab.py:330-332`, `tests/test_artifact_consistency.py` (L117-160), `tests/test_user_sbi.py` (every `build_prior(` call; the `_assert_prior_is_saved` tests ~L3552-3585; the orphan test ~L3860-3912)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Consumes: `ArtifactStore.create/rename/find_prior_by_fingerprint`, `LoadedPrior`, `file_manager.save_mix_dist/load_mix_dist`, `reparam.nd_log_mask`, `run_guards._gmm_fingerprint/_find_nd_gmm/_log_params_for`, `orchestrator.build_rescale_prior/build_forcing_prior/ProductPrior`.
- Produces: `ArtifactStore.load_prior(cfg, ref) -> LoadedPrior`; `orchestrator.build_prior(cfg, ref, build_new, *, name="", note="", fig_sink=None, store=None, num_iterations=None, sweep_batch=None, max_sets=None, walk_step=None, stability_units=None, min_cluster_size=None, min_samples=None) -> LoadedPrior` (the `save`/`save_name` parameters are gone); `runners._run_simulated_inference(cfg, posterior, cell_path, T_obs_s, *, gt_dicts=None, prior=None, fig_sink=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_store.py`:

```python
def _gmm_in_box(lows, highs, mask, seed=0):
    from core.SBI import reparam
    torch.manual_seed(seed)
    d = len(lows)
    base = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=torch.tensor([0.5, 0.5])),
        torch.distributions.MultivariateNormal(torch.randn(2, d), covariance_matrix=torch.eye(d).expand(2, d, d)))
    T = reparam.build_box_bijection(torch.tensor(lows, dtype=torch.float32), torch.tensor(highs, dtype=torch.float32), mask)
    return torch.distributions.TransformedDistribution(base, T)


def _prior_artifact(store, cfg, *, name="p", lows=None, highs=None, keys=None, model=None, seed=0):
    """A prior artifact with a real 2-component GMM payload, laid out exactly as build_prior writes it."""
    from core.Helpers import file_manager
    from core.SBI.reparam import nd_log_mask
    from core.SBI.run_guards import _gmm_fingerprint, _log_params_for
    keys = keys or list(cfg.params_dict)
    lows = lows or [b[0] for _, b in cfg.params_dict.values()]
    highs = highs or [b[1] for _, b in cfg.params_dict.values()]
    mask = nd_log_mask(cfg, log_params=_log_params_for(cfg))
    dist = _gmm_in_box(lows, highs, mask, seed)
    with store.create("prior", cfg, name=name) as w:
        file_manager.save_mix_dist(dist, str(w.payload("prior.pt")), model=model or cfg.model, param_keys=keys)
        if model:
            w.config["model"] = model
        w.fingerprints["gmm"] = _gmm_fingerprint(dist)
        w.body = {"gmm": {"n_components": 2, "param_keys": list(keys),
                          "box": {"nd_lows": [float(v) for v in lows], "nd_highs": [float(v) for v in highs],
                                  "log_mask": [bool(v) for v in mask.tolist()]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}}
    return w


def test_load_prior_refuses_model_param_order_and_box_and_returns_a_wrapper(store):
    cfg = _nad_cfg()
    ok = _prior_artifact(store, cfg, name="ok")
    lp = store.load_prior(cfg, "ok")
    assert lp.id == ok.id and lp.name == "ok" and lp.fingerprint == store.get("prior", ok.id).fingerprints["gmm"]
    assert lp.prior.distributions[0] is lp.nd_prior and lp.force_prior is not None   # master.txt has a drive
    keys = list(cfg.params_dict)
    hi = [b[1] for _, b in cfg.params_dict.values()]
    for tag, kw, why in (("model", dict(model="HOPF"), "the model"),
                         ("order", dict(keys=keys[1:] + keys[:1]), "ORDER"),
                         ("box", dict(highs=[hi[0] * 2] + hi[1:]), "the ND box")):
        _prior_artifact(store, cfg, name=tag, **kw)
        with pytest.raises(ValueError, match=why):
            store.load_prior(cfg, tag)
    other = _prior_artifact(store, cfg, name="other", seed=1)
    (ok.dir / "prior.pt").write_bytes((other.dir / "prior.pt").read_bytes())
    with pytest.raises(st.StoreError, match="not the one the manifest records"):
        store.load_prior(cfg, "ok")


def test_build_prior_auto_persists_and_loads_back(store, monkeypatch):
    from core import orchestrator
    from core.SBI.reparam import nd_log_mask
    from core.SBI.run_guards import _log_params_for
    cfg = _nad_cfg()

    def stub_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds, **kw):
        lows = [float(b[0]) for b in prior_bounds]
        highs = [float(b[1]) for b in prior_bounds]
        return _gmm_in_box(lows, highs, kw.get("log_mask"))

    monkeypatch.setattr(orchestrator.pipeline, "gen_prior", stub_gen_prior)
    seen = []
    lp = orchestrator.build_prior(cfg, None, True, fig_sink=lambda title, fig: seen.append(title), num_iterations=1)
    assert lp.name == "" and [r.id for r in store.list("prior")] == [lp.id]
    assert (lp.path / "figures" / "prior.png").stat().st_size > 0 and seen == ["Prior"]
    m = lp.manifest
    assert m.body["gmm"]["param_keys"] == list(cfg.params_dict) and m.config["num_iterations"] == 1
    assert m.body["gmm"]["box"]["log_mask"] == nd_log_mask(cfg, log_params=_log_params_for(cfg)).tolist()
    assert m.inputs["bounds"]["path"] == "Bounds/nadrowski/master.txt" and m.fingerprints["gmm"] == lp.fingerprint
    again = orchestrator.build_prior(cfg, lp.id, False, fig_sink=lambda title, fig: None)
    assert again.id == lp.id and again.fingerprint == lp.fingerprint
    store.rename("prior", lp.id, "master_prior")
    assert orchestrator.build_prior(cfg, "master_prior", False, fig_sink=lambda title, fig: None).name == "master_prior"


def test_delete_refuses_a_prior_a_simulation_was_generated_against(store):
    from core.SBI import training_checkpoint as tc
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="p")
    fp = store.get("prior", p.id).fingerprints["gmm"]
    ident = {"format": "training-rows/2", "prior_fingerprint": fp, "n_runs": 3, "truncation": None}
    d = tc.resolve_dir(ident)
    tc.create(d, ident, schedule_t_scales=torch.ones(3), schedule_Ts=torch.ones(3), inits=torch.zeros(1, 2),
              V=None, probe=torch.zeros(7, 2, dtype=torch.float64), run_size=4, n_runs=3,
              parents={"prior": p.id}, hw=config.cpu_device())
    with pytest.raises(st.StoreError, match="simulation"):
        store.delete("prior", p.id)
    assert store.find_prior_by_fingerprint(fp).id == p.id
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_artifact_store.py -k prior -v`
Expected: 3 failures (`AttributeError: 'ArtifactStore' object has no attribute 'load_prior'`; `build_prior` rejects `store=`... no, `save=` — `TypeError` on the tuple unpack).

- [ ] **Step 3: Add `ArtifactStore.load_prior`**

In `core/artifacts/store.py`, inside `ArtifactStore` after `simulation_for`:

```python
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
```

- [ ] **Step 4: Rewrite `build_prior` and remove the save-side machinery**

In `core/orchestrator.py`:

1. Imports: in the `from .config import (` tuple drop `PRIOR_PATH`; in the `from .SBI.run_guards import (` list drop `_assert_prior_matches`; add
   ```python
   from .artifacts import LoadedPrior, resolve_store
   ```
   (`core.artifacts` imports `core.config` and `file_manager` only at module level, and `core.orchestrator` lazily, so there is no cycle.)
2. Delete `_UNSAVED_PRIOR_MIN_RUNS`, `_saved_prior_fingerprints`, `_assert_prior_is_saved` (L292-354) and `_refuse_to_orphan_a_checkpoint`, `save_prior_artifacts` (L1282-1343).
3. Replace `build_prior`'s signature and the two branches:

```python
def build_prior(cfg: SimConfig, ref: str | None, build_new: bool,
                *, name: str = "", note: str = "", fig_sink=None, store=None,
                num_iterations: int | None = None, sweep_batch: int | None = None,
                max_sets: int | None = None, walk_step: float | None = None,
                stability_units: float | None = None,
                min_cluster_size: int | None = None,
                min_samples: int | None = None) -> LoadedPrior:
    """
    Load a prior artifact by name or id, or construct a new product prior and WRITE it at once:
        ProductPrior = ND parameter prior x rescaling prior x forcing prior

    Returns a LoadedPrior -- the artifact's id and manifest travel with the prior, so a posterior
    trained from it can name its parent. A built prior is unnamed until store.rename gives it a
    name (the GUI's Save button); nothing is ever in memory only, which is what makes the
    checkpoint identity's prior fingerprint reproducible after this process ends.

    :param ref: a prior artifact's name or id; None when building.
    :param build_new: True to construct from scratch.
    :param name / note: the artifact's name ("" = unnamed) and free-text note when building.
    :param store: the ArtifactStore to read/write; None = the process default.
    (knob docs unchanged from the previous version -- keep them.)
    """
    store = resolve_store(store)
    _assert_chi_config_is_deliberate(cfg)
    # (the user-model order guard stays exactly as it was)
    ...
    if not build_new and ref is not None:
        loaded = store.load_prior(cfg, ref)
        visualizers.visualize_dist(loaded.nd_prior, labels=cfg.labels, title="Prior (loaded)", sink=fig_sink)
        return loaded

    # --- Build from scratch ---
    print("Constructing the prior from scratch.")
    # (the sweep block from `stab_units = ...` through `nd_prior = pipeline.gen_prior(...)` is UNCHANGED;
    #  the `time.sleep(5)` / `helpers.clear_screen()` pair before it is deleted)
    ...
    nd_log = nd_log_mask(cfg, log_params=_log_params_for(cfg))
    gmm = _find_nd_gmm(nd_prior)
    with store.create("prior", cfg, name=name, note=note) as w:
        file_manager.save_mix_dist(nd_prior, str(w.payload("prior.pt")),
                                   model=cfg.model, param_keys=list(cfg.params_dict.keys()))
        visualizers.visualize_dist(nd_prior, labels=cfg.labels, title="Prior", sink=w.fig_sink(fig_sink))
        w.fingerprints["gmm"] = _gmm_fingerprint(nd_prior)
        knobs = {"num_iterations": n_iter, "sweep_batch": sweep_batch, "max_sets": max_sets,
                 "walk_step": walk_step, "stability_units": stab_units,
                 "min_cluster_size": min_cluster_size, "min_samples": min_samples}
        w.config.update(knobs)
        w.body = {
            "gmm": {"n_components": int(gmm.mixture_distribution.probs.numel()) if gmm is not None else None,
                    "param_keys": list(cfg.params_dict),
                    "box": {"nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
                            "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
                            "log_mask": [bool(v) for v in nd_log.tolist()]}},
            "sweep": dict(knobs),
            # gen_prior does not return the accepted count; recorded honestly as unknown.
            "stability": {"accepted_sets": None, "iterations": n_iter},
        }
    return store.load_prior(cfg, w.id)
```
   (`build_rescale_prior` / `build_forcing_prior` are no longer called here; the loader composes the product prior.) Note `walk_step`/`max_sets` may be None: JSON null.
4. In `build_posterior`: delete L870-872 (the `_assert_prior_is_saved` call and its comment) and replace L924-927 with
   ```python
            _hit = resolve_store(None).find_prior_by_fingerprint(_want)
            raise ValueError(f"{_e} The region's fingerprint is that of prior '{_hit.label}' [{_hit.id}]."
                             if _hit else str(_e)) from None
   ```
5. In `run` (L76-77): `lp = build_prior(cfg, prior_choice, build_new)` and `inf_prior, force_prior = lp.prior, lp.force_prior` on the next line, so the rest of `run` is untouched until Task 5.

In `core/SBI/run_guards.py` delete `_assert_prior_matches` (L116-161) and the `from core.Helpers import file_manager` import if nothing else uses it; update the module docstring's list of guards that "stay in orchestrator" (all three named there are now gone: say so in one line).

- [ ] **Step 5: The GUI and the runner**

`core/gui/panels/inference/prior_tab.py`:
- `from core.config import BOUNDS_PATH` (drop `PRIOR_PATH`); `from ...widgets.artifact_picker import ArtifactPicker, StorePicker`; `from core.artifacts import default_store`.
- L41: `self.prior_picker = StorePicker("prior", allow_new=True)`.
- L179: `self.dispatch(orchestrator.build_prior, cfg, entry, is_new,` (drop `save=False,`).
- `_on_prior`:
  ```python
    def _on_prior(self, payload):
        self.session.inf_prior = payload                   # a LoadedPrior
        self.session.force_prior = payload.force_prior     # removed with the field in Task 7
        self.log_pane.append_line(f"Prior ready: {payload.name or '(unnamed, id ' + payload.id + ')'}. "
                                  f"Name it below to keep it.")
        self._screen.refresh_gates()
  ```
- `_save_prior` (a rename, on the GUI thread; not a run):
  ```python
    def _save_prior(self):
        name = self.prior_name.text().strip()
        lp = self.session.inf_prior
        if not name or lp is None:
            self.log_pane.append_line("Build a prior and enter a name first.", "warning")
            return
        try:
            lp.manifest = default_store().rename("prior", lp.id, name)
        except Exception as e:                       # noqa: BLE001 -- a bad or duplicate name is user input
            self._config_error(e)
            return
        lp.name = name
        self.prior_picker.refresh()
        self.prior_picker.restore_key(lp.id)
        self.log_pane.append_line(f"Prior named '{name}'.")
  ```
`core/gui/panels/inference/runners.py` `_run_simulated_inference`: signature `(cfg, posterior, cell_path, T_obs_s, *, gt_dicts=None, prior=None, fig_sink=None)`; body L22-24 becomes
```python
    if prior is not None:
        for msg in orchestrator.check_observation_in_distribution(cfg, prior.prior, prior.force_prior):
            print(f"WARNING: {msg}")
```
`core/gui/panels/inference/infer_tab.py` L330-332: `gt_dicts=gt_dicts, prior=self.session.inf_prior, provide_fig_sink=True)`.

`core/gui/panels/inference/tsnpe_tab.py` L106: `self.dispatch(_run_tsnpe_round, s.cfg, s.posterior, s.inf_prior.prior, s.inf_prior.force_prior, ...)` (the runner still takes bare distributions until Task 5).
`core/gui/panels/inference/validate_tab.py` L46-47: `s.inf_prior.prior, s.inf_prior.force_prior,` (until Task 9).
`core/gui/panels/inference/posterior_tab.py` L128-129: `orchestrator.build_posterior, cfg, self.session.inf_prior.prior, self.session.inf_prior.force_prior, entry, is_new, save=False,` (until Task 5). `_confirm_fresh_run` and `base._budget_checkpoint` pass `self.session.inf_prior` (the wrapper) to `training_identity`, which `SimulationIdentity.from_cfg` accepts.

- [ ] **Step 6: Update the suites**

- `tests/test_artifact_consistency.py`: delete `_write_prior` (L117-127) and `test_a_prior_from_another_config_is_refused` (L131-160) — the store test is their successor. Drop `PRIOR_PATH` from the `from core.config import` line if it is now unused.
- `tests/test_user_sbi.py`: for every `inferred_prior, force_prior = orchestrator.build_prior(cfg, None, True, save=False, fig_sink=sink)` write
  ```python
        lp = orchestrator.build_prior(cfg, None, True, fig_sink=sink)
        inferred_prior, force_prior = lp.prior, lp.force_prior
  ```
  (`grep -n "build_prior(" tests/test_user_sbi.py` lists them; `save=` is no longer accepted). Delete the `_assert_prior_is_saved` tests (~L3552-3585) and the orphan test (~L3860-3912, the one that rebinds `orchestrator.PRIOR_PATH`); their successors are in the store suite. Any test that stubs `orchestrator._saved_prior_fingerprints` goes with them.
- `tests/test_nav_and_gating.py`: `grep -n "_on_prior" tests/test_nav_and_gating.py` — a test that calls `panel._on_prior((prior, force))` passes a stub with `.prior`/`.force_prior`/`.name`/`.id` attributes instead.
- `tests/test_settings_persistence.py`: no change expected (`inf_prior=object()` stubs still fail open through `from_cfg`).

- [ ] **Step 7: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py tests/test_artifact_consistency.py tests/test_nav_and_gating.py -q` then `pytest -m "not slow" -q` — green.

```bash
git add core/artifacts/store.py core/orchestrator.py core/SBI/run_guards.py core/gui/panels/inference tests
git commit -F - <<'MSG'
prior: build_prior returns a LoadedPrior and auto-persists; store.load_prior (piece 1, Task 4). A built prior is written at once (prior.pt + corner figure + manifest with the GMM fingerprint, the box, the log mask and every sweep knob) and read back through the loader, so nothing is ever in memory only; the Save button is a rename. load_prior refuses model / parameter order / box / log-mask mismatches before reading the payload and an inconsistent payload after. Deleted: _assert_prior_is_saved, _saved_prior_fingerprints, _refuse_to_orphan_a_checkpoint (store.delete's dependents check is the successor), save_prior_artifacts, run_guards._assert_prior_matches (defect 1 with it). PRIOR_PATH leaves orchestrator's import.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 5: `build_posterior` takes the `LoadedPrior`; the identity and the parents go through the store

**Files:**
- Modify: `core/orchestrator.py` (`build_posterior` signature L617-630, body L694-1151, `run` L83-89), `core/gui/panels/inference/posterior_tab.py:128-130`, `core/gui/panels/inference/runners.py:66-89`, `core/gui/panels/inference/tsnpe_tab.py:106`, `tests/test_user_sbi.py`, `tests/test_artifact_consistency.py` (the `object()` stand-ins), `tests/test_conditioning_repair.py` (any `build_posterior(` call)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Produces: `build_posterior(cfg, prior: LoadedPrior, ref, train_new, *, save=True, save_name=None, fig_sink=None, store=None, num_runs=None, run_size_cap=None, truncation=None, x_obs_digest=None, accept_truncated=False, hidden_features=None, num_transforms=None, learning_rate=None, stop_after_epochs=None, fisher_m=None, fisher_dz=None, fisher_points=None) -> tuple[TransformedPosterior, dict | None]` (still the old return for one task; `force_prior` is gone — it is `prior.force_prior`); `runners._run_tsnpe_round(cfg, posterior, prior: LoadedPrior, obs_path, n_directions, level, num_runs, run_size_cap, *, fig_sink=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_artifact_store.py`:

```python
def test_a_training_run_records_the_prior_as_the_simulation_cache_parent(store, monkeypatch):
    """The checkpoint dict build_posterior hands gen_training_data names the LoadedPrior's id, and the
    identity's prior fingerprint is the wrapper's -- checked at the seam, with train_nn stubbed."""
    from core import orchestrator
    from core.artifacts.identity import SimulationIdentity
    cfg = _nad_cfg()
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    cfg.reparam_rotate = False
    captured = {}

    def fake_train_nn(plan, **kw):
        captured["plan"] = plan
        raise RuntimeError("stop before training")

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", fake_train_nn)
    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 1)
    with pytest.raises(RuntimeError, match="stop before training"):
        orchestrator.build_posterior(cfg, lp, None, True, save=False, num_runs=2, run_size_cap=4)
    ck = captured["plan"].checkpoint
    assert ck["parents"] == {"prior": lp.id} and ck["inputs"]["model"] == "NADROWSKI" and ck["hw"] is cfg.hw
    assert ck["identity"] == SimulationIdentity.from_cfg(cfg, lp, 4, 2).to_dict()
    assert ck["identity"]["prior_fingerprint"] == lp.fingerprint and ck["dir"].parent == store.kind_dir("simulation")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_artifact_store.py -k cache_parent -v` — `TypeError` (the old signature wants `force_prior` positionally).

- [ ] **Step 3: Change the signature and thread the wrapper**

In `build_posterior`: replace the two parameters `prior: Distribution, force_prior: Distribution,` with `prior: LoadedPrior,`; add `store=None` after `fig_sink=None`. First lines of the body:

```python
    store = resolve_store(store)
    inferred, force_prior = prior.prior, prior.force_prior
```
Then every use of the bare distribution reads `inferred`: L700 (`inferred.sample((4096,))`), L790 (`inferred.distributions[0]`), L796 (`inferred.distributions[1]`), L780 and L922 (`_assert_prior_matches_region(region, inferred, ...)`), L928 (`_gmm_fingerprint(inferred)`). The identity block (L873-876) becomes:

```python
        from .artifacts.identity import SimulationIdentity
        ident = SimulationIdentity.from_cfg(cfg, prior, run_size, n_runs, truncation=truncation).to_dict()
        ckpt_dir = training_checkpoint.resolve_dir(ident, store.kind_dir("simulation"))
```
and the checkpoint dict's `"parents"` line (Task 3) becomes `"parents": {"prior": prior.id},`. The `_hit = resolve_store(None)...` line from Task 4 becomes `_hit = store.find_prior_by_fingerprint(_want)`.

Callers:
- `run` L83: `posterior, pos_diagnostics = build_posterior(cfg, lp, pos_choice, train_new, accept_truncated=True)`.
- `posterior_tab.py` L128-130: `self.dispatch(orchestrator.build_posterior, cfg, self.session.inf_prior, entry, is_new, save=False,`.
- `runners._run_tsnpe_round(cfg, posterior, prior, obs_path, n_directions, level, num_runs, run_size_cap, *, fig_sink=None)` and inside `orchestrator.build_posterior(cfg, prior, None, True, save=False, ...)`; `tsnpe_tab.py` L106: `self.dispatch(_run_tsnpe_round, s.cfg, s.posterior, s.inf_prior, OBSERVATION_PATH / self.obs_picker.key(), ...)`.
- Tests: every `build_posterior(cfg, inferred_prior, force_prior, ...)` → `build_posterior(cfg, lp, ...)` (`grep -n "build_posterior(" tests/*.py`). Where a test passes `object()` for the prior (`test_artifact_consistency.py:357, 360, 366` and `test_conditioning_repair.py`), use this stand-in, defined once per file:
  ```python
  class _LoadedStub:
      """The LoadedPrior shape with no GMM: fails open through every fingerprint check."""
      id, name, fingerprint, force_prior = None, "", None, None
      def __init__(self, prior=None): self.prior = prior if prior is not None else object()
  ```
  and `test_user_sbi`'s guardrail-7 test, which passes hand-built priors, wraps them: `_LoadedStub(product_prior)`.

- [ ] **Step 4: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py tests/test_artifact_consistency.py tests/test_conditioning_repair.py -q`, then `pytest tests/test_user_sbi.py -k "tsnpe or identity or checkpoint" -q`, then `pytest -m "not slow" -q` — green.

```bash
git add core/orchestrator.py core/gui/panels/inference tests
git commit -F - <<'MSG'
posterior: build_posterior takes the LoadedPrior; the simulation identity and its parents go through the store (piece 1, Task 5). force_prior is the wrapper's; the checkpoint dict names the prior's id as the cache's parent; the identity is SimulationIdentity.from_cfg over the wrapper's fingerprint; the cache directory resolves under the store the stage was given. Return value unchanged for one more task.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 6: The posterior loader with every refusal (additive)

**Files:**
- Modify: `core/SBI/reparam.py` (add `build_eval_bijection`, `assert_rotation_consistent` beside `load_eval_bijection`), `core/artifacts/manifest.py` (add `region_to_json`, `region_from_json`), `core/artifacts/store.py` (add `load_posterior`)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Consumes: `reparam.build_box_bijection`, `_log_mask`, `build_rotated_bijection`, `rotation_of`, `rotation_of_prior`, `posterior_mode`, `TransformedPosterior`; `truncate.TruncationRegion.from_dict/check_basis`; `run_guards._assert_chi_config_is_deliberate`, `_gmm_fingerprint`; `orchestrator.expected_forcing_dim`; `statistics.SUMMARY_WIDTH`.
- Produces: `reparam.build_eval_bijection(cfg, transform: dict) -> ComposeTransform`; `reparam.assert_rotation_consistent(T, prior, *, name="posterior") -> None`; `manifest.region_to_json(region) -> dict`, `manifest.region_from_json(d) -> dict` (the dict `TruncationRegion.from_dict` takes); `ArtifactStore.load_posterior(cfg, ref, *, accept=None) -> LoadedPosterior`. The posterior body's `transform` block carries `param_keys` in addition to the spec's fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_store.py`:

```python
from sbi.inference import DirectPosterior


class _FakeDP(DirectPosterior):
    """A DirectPosterior by type only (the loader's isinstance accepts it); module-level so it pickles."""
    def __init__(self):
        pass


def _set_path(w, path, value):
    target = w.config if path[0] == "config" else w.body
    for key in path[1 if path[0] == "config" else 0:-1]:
        target = target[key]
    target[path[-1]] = value


def _posterior_artifact(store, cfg, *, name="post", amortized=True, region=None, V=None, prior=None, over=None):
    """A posterior artifact with a _FakeDP payload, laid out exactly as build_posterior writes it."""
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    with store.create("posterior", cfg, name=name) as w:
        dp = _FakeDP()
        dp.prior = prior
        torch.save(dp, str(w.payload("posterior.pt")))
        w.parents = {"prior": "20260910T100000"}
        w.body = {
            "mode": cfg.observation_mode, "conditioning": mf.conditioning_block(cfg),
            "transform": {"param_keys": keys, "log_params": [],
                          "nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
                          "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
                          "rescale_lows": [float(b[0]) for _, b in cfg.rescale_params.values()],
                          "rescale_highs": [float(b[1]) for _, b in cfg.rescale_params.values()],
                          "V": mf.tensor_to_json(V), "V_orientation": "columns",
                          "fisher_eigenvalues": None, "V_digest": mf.tensor_digest(V)},
            "amortized": amortized, "truncation": None if region is None else mf.region_to_json(region),
            "training": {}}
        for path, value in (over or {}).items():
            _set_path(w, path, value)
    return w


def test_load_posterior_refuses_each_mismatch_class(store):
    cfg = _nad_cfg(chi_mode=True)
    ok = _posterior_artifact(store, cfg, name="ok")
    lp = store.load_posterior(cfg, "ok")
    assert lp.id == ok.id and lp.posterior.truncation is None and lp.accepted == [] and lp.latent is not None
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    hi = [float(b[1]) for _, b in cfg.params_dict.values()]
    cases = {
        "model": ({("config", "model"): "HOPF"}, "trained for model"),
        "order": ({("transform", "param_keys"): keys[1:] + keys[:1]}, "ORDER"),
        "box": ({("transform", "nd_highs"): [hi[0] * 2] + hi[1:]}, "different box"),
        "mode": ({("mode",): "forced"}, "mode"),
        "layout": ({("conditioning", "chi_layout"): 1}, "layout"),
        "k_pad": ({("conditioning", "chi_k_pad"): int(cfg.chi_k_pad) + 1}, "probe-slot"),
        "elem_w": ({("conditioning", "chi_elem_w"): 5}, "channels per slot"),
        "band": ({("conditioning", "chi_freq_bounds"): [0.1, 10.0]}, "band"),
        "cycles": ({("conditioning", "chi_max_cycles"): 5.0}, "cycle"),
        "width": ({("conditioning", "forcing_dim"): 3, ("conditioning", "width"): 53}, "incompatible"),
    }
    for tag, (over, why) in cases.items():
        _posterior_artifact(store, cfg, name=tag, over=over)
        with pytest.raises(ValueError, match=why):
            store.load_posterior(cfg, tag)


def test_manifest_V_must_equal_the_rotation_in_the_pickled_prior(store):
    """The D6 spec test, moved: the V a manifest records is the one the posterior's own training prior
    carries. With no legacy sidecars left, the transpose is REFUSED, not repaired."""
    from core.SBI import reparam
    from core.SBI.training_checkpoint import bijection_probe
    cfg = _nad_cfg(chi_mode=True)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    torch.manual_seed(3)
    Q = torch.linalg.qr(torch.randn(P, P))[0]
    assert not torch.allclose(Q, Q.T)
    T_train = reparam.build_rotated_bijection(reparam.build_inferred_bijection(cfg, log_params=[]), Q)
    V = reparam.rotation_of(T_train)                      # what build_posterior records

    class _Wrap:                                          # SBIPriorWrapper's shape: .gen_dist
        def __init__(self, inner):
            self.gen_dist = inner
    base = torch.distributions.MultivariateNormal(torch.zeros(P), torch.eye(P))
    trained = _Wrap(reparam.RotatedLatentPrior(base, Q))

    _posterior_artifact(store, cfg, name="good", V=V, prior=trained)
    lp = store.load_posterior(cfg, "good")
    assert torch.allclose(bijection_probe(lp.posterior.T, P), bijection_probe(T_train, P))
    _posterior_artifact(store, cfg, name="transposed", V=V.T, prior=trained)
    with pytest.raises(ValueError, match="TRANSPOSE"):
        store.load_posterior(cfg, "transposed")
    _posterior_artifact(store, cfg, name="unrotated", V=None, prior=trained)
    with pytest.raises(ValueError, match="NO rotation"):
        store.load_posterior(cfg, "unrotated")
    _posterior_artifact(store, cfg, name="plain", V=V, prior=None)   # a prior without a rotation cannot arbitrate
    assert reparam.rotation_of(store.load_posterior(cfg, "plain").posterior.T) is not None


def test_a_non_amortized_posterior_needs_accept_and_the_flag_is_recorded(store):
    from core.artifacts import Accept
    from core.SBI import reparam, truncate
    from core.SBI.training_checkpoint import bijection_probe
    cfg = _nad_cfg(chi_mode=True)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    T = reparam.build_inferred_bijection(cfg, log_params=[])
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], n_latent=P, V=None,
                                       probe=bijection_probe(T, P), x_obs_digest="d" * 16)
    _posterior_artifact(store, cfg, name="trunc", amortized=False, region=region)
    with pytest.raises(ValueError, match="NOT AMORTIZED"):
        store.load_posterior(cfg, "trunc")
    lp = store.load_posterior(cfg, "trunc", accept=Accept(truncated=True))
    assert lp.posterior.truncation.dims == [0, 1] and lp.posterior.x_obs_digest == "d" * 16
    assert lp.accepted == ["truncated"] and torch.equal(lp.posterior.truncation.probe, region.probe)
    no_basis = truncate.TruncationRegion([0], [-1.0], [1.0], n_latent=P, V=None, probe=None, x_obs_digest="e" * 16)
    _posterior_artifact(store, cfg, name="noprobe", amortized=False, region=no_basis)
    with pytest.raises(ValueError, match="basis"):
        store.load_posterior(cfg, "noprobe", accept=Accept(truncated=True))
    assert store.list("posterior")[0].amortized is not None


def test_a_legacy_or_schemaless_directory_is_refused_as_not_a_current_artifact(store):
    d = store.kind_dir("posterior") / "legacy__20200101T000000"
    d.mkdir(parents=True)
    torch.save(_FakeDP(), d / "posterior.pt")
    (d / "posterior.rot.pt").write_bytes(b"")
    with pytest.raises(st.StoreError, match="no complete posterior"):
        store.load_posterior(_nad_cfg(), "legacy__20200101T000000")
    assert store.list("posterior")[0].complete is False
    (d / "manifest.json").write_text(json.dumps({"schema": 0}), encoding="utf-8")
    assert "schema" in store.list("posterior")[0].reason
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_artifact_store.py -k "posterior or legacy or rotation" -v` — `AttributeError: region_to_json` / `load_posterior`.

- [ ] **Step 3: The region encoders and the two reparam helpers**

In `core/artifacts/manifest.py` add:

```python
def region_to_json(region) -> dict:
    """``TruncationRegion.to_dict()`` with its tensors (lo, hi, V, probe) as float64 lists."""
    d = dict(region.to_dict())
    for key in ("lo", "hi", "V", "probe"):
        d[key] = tensor_to_json(d.get(key))
    d["level"] = float(d["level"])
    return d


def region_from_json(d: dict) -> dict:
    """The inverse: the dict ``TruncationRegion.from_dict`` takes (lists back to float64 tensors)."""
    out = dict(d)
    for key in ("lo", "hi", "V", "probe"):
        out[key] = json_to_tensor(d.get(key))
    return out
```

In `core/SBI/reparam.py`, directly after `load_eval_bijection`:

```python
def build_eval_bijection(cfg, transform: dict) -> ComposeTransform:
    """The EXACT physical-space bijection a saved posterior trained under, rebuilt from its
    manifest's ``transform`` block: the box and log mask recorded THERE (never the live config's),
    then the rotation V -- eigenvectors in COLUMNS, the orientation build_rotated_bijection expects
    and rotation_of returns. Successor of load_eval_bijection, which read the .rot.pt sidecar."""
    dev, dt = cfg.hw.device, cfg.hw.dtype
    lows = torch.tensor(list(transform["nd_lows"]) + list(transform["rescale_lows"]), dtype=dt, device=dev)
    highs = torch.tensor(list(transform["nd_highs"]) + list(transform["rescale_highs"]), dtype=dt, device=dev)
    names = list(transform["param_keys"])
    T = build_box_bijection(lows, highs, _log_mask(names, lows, list(transform.get("log_params") or [])))
    V = transform.get("V")
    if V is None:
        return T
    return build_rotated_bijection(T, torch.tensor(V, dtype=dt, device=dev))


def assert_rotation_consistent(T: ComposeTransform, prior, *, name: str = "posterior") -> None:
    """Refuse a posterior whose evaluation bijection does not rotate by the rotation pickled inside
    its own training prior. Successor of reconcile_loaded_rotation WITHOUT its repair branch: every
    artifact is now written by a writer that records V through rotation_of, so a disagreement is an
    inconsistent artifact, not a legacy sidecar. A prior with no rotation cannot arbitrate and passes."""
    V_side, V_net = rotation_of(T), rotation_of_prior(prior)
    if V_net is None:
        return
    if V_side is None:
        raise ValueError(
            f"Posterior '{name}': its manifest records NO rotation, but the training prior pickled "
            f"inside it holds a {tuple(V_net.shape)} one -- the flow was trained on w = z @ V and cannot "
            f"be decoded through the bare box. The artifact is inconsistent.")
    V_net = V_net.detach().to(device=V_side.device, dtype=V_side.dtype)
    if V_side.shape != V_net.shape:
        raise ValueError(f"Posterior '{name}': its manifest's rotation is {tuple(V_side.shape)} but the "
                         f"rotation inside its training prior is {tuple(V_net.shape)}; the artifact is inconsistent.")
    if torch.allclose(V_side, V_net, atol=1e-6):
        return
    hint = " -- it is the TRANSPOSE" if torch.allclose(V_side.transpose(-1, -2), V_net, atol=1e-6) else ""
    raise ValueError(
        f"Posterior '{name}': the manifest's rotation is not the rotation inside its training prior"
        f"{hint} (max|diff| = {float((V_side - V_net).abs().max()):.3g}); the artifact is inconsistent "
        f"and cannot be decoded.")
```

- [ ] **Step 4: Add `ArtifactStore.load_posterior`**

In `core/artifacts/store.py`, after `load_prior`:

```python
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
```

- [ ] **Step 5: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py -q` then `pytest -m "not slow" -q` — green (nothing calls the new loader yet).

```bash
git add core/artifacts core/SBI/reparam.py tests/test_artifact_store.py
git commit -F - <<'MSG'
artifacts: store.load_posterior with every refusal; reparam.build_eval_bijection and assert_rotation_consistent (piece 1, Task 6). The loader reads model, parameter order, box, mode, chi layout/pad/width/band/cycles and the conditioning width from the manifest before unpickling; then checks the trained net's own view, rebuilds the bijection from the manifest's box and V (eigenvectors in columns), refuses a V that is not the one inside the pickled training prior (the transpose included -- the D6 repair branch is retired), and gates a non-amortized artifact on Accept(truncated=True), recording the flag. region_to_json / region_from_json carry a TruncationRegion through JSON. Additive; nothing calls it yet.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 7: The posterior switch — `build_posterior` returns a `LoadedPosterior`, auto-persists, and the sidecar is retired

**Files:**
- Create: `tests/_fixtures.py` (the helpers `_FakeDP`, `_nad_cfg`, `_gmm_in_box`, `_prior_artifact`, `_posterior_artifact`, `_set_path`, `_LoadedStub` move here from `tests/test_artifact_store.py`, which imports them; no import-time side effects)
- Modify: `core/orchestrator.py` (`build_posterior` L617-1151; delete `_assert_mode_matches` L1384-1470 and `save_posterior_artifacts` L1473-1567; imports), `core/SBI/reparam.py` (delete `reconcile_loaded_rotation` L381-428, `sidecar_path` L431-434, `read_sidecar` L437-449, `load_eval_bijection` L518-570), `core/SBI/run_guards.py` (delete `truncation_from_sidecar` L221-244, `_assert_amortization_understood` L247-271, the `POSTERIOR_PATH` and `read_sidecar` imports), `core/gui/session.py`, `core/gui/panels/inference/{posterior_tab,tsnpe_tab,validate_tab,infer_tab,runners}.py`, `tests/test_artifact_consistency.py` (L201-570), `tests/test_nav_and_gating.py` (L289-365), `tests/test_conditioning_repair.py` (L129; the AST pin near L951), `tests/test_user_sbi.py`
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Produces: `build_posterior(cfg, prior: LoadedPrior, ref, train_new, *, name="", note="", fig_sink=None, store=None, num_runs=None, run_size_cap=None, truncation=None, x_obs_digest=None, observation=None, parent_posterior=None, accept=None, hidden_features=None, num_transforms=None, learning_rate=None, stop_after_epochs=None, fisher_m=None, fisher_dz=None, fisher_points=None) -> LoadedPosterior` (`save`, `save_name`, `accept_truncated` are gone; `x_obs_digest` stays until Task 11); `runners._run_tsnpe_round(...) -> LoadedPosterior`; `SbiSession(draft, cfg, inf_prior: LoadedPrior, posterior: LoadedPosterior, observation: LoadedObservation)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_artifact_store.py`:

```python
def test_build_posterior_auto_persists_and_returns_the_loaded_wrapper(store, monkeypatch):
    """train_nn is stubbed to return a _FakeDP carrying the training prior; everything around it --
    the identity, the parents, the transform block, the loss curve, the figure, the load-back -- is real."""
    from core import orchestrator
    from core.artifacts import Accept
    from core.SBI import reparam
    cfg = _nad_cfg()
    cfg.reparam_rotate = False
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)

    def fake_train_nn(plan, **kw):
        dp = _FakeDP()
        dp.prior = kw["prior"]                    # the SBIPriorWrapper build_posterior passed in
        return dp, {"training_loss": [1.0, 0.5], "validation_loss": [1.1, 0.6],
                    "best_validation_loss": 0.6, "epochs_trained": 2, "stop_after_epochs": 1}

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", fake_train_nn)
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data",
                        lambda plan, **kw: (torch.zeros(8, 50), torch.zeros(8, 13)))
    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 0)
    seen = []
    out = orchestrator.build_posterior(cfg, lp, None, True, fig_sink=lambda t, f: seen.append(t),
                                       num_runs=2, run_size_cap=4, hidden_features=8, num_transforms=1,
                                       stop_after_epochs=1, note="tiny")
    assert isinstance(out, st.LoadedPosterior) and out.name == "" and out.diagnostics["epochs_trained"] == 2
    m = store.get("posterior", out.id)
    assert m.parents == {"prior": lp.id} and m.note == "tiny" and m.body["amortized"] is True
    assert m.body["transform"]["V"] is None and m.body["transform"]["param_keys"][-1] in cfg.rescale_params
    assert m.body["training"]["hidden_features"] == 8 and m.body["training"]["best_validation_loss"] == 0.6
    assert m.config["num_runs"] == 2 and m.fingerprints["gmm"] == lp.fingerprint
    assert set(m.payloads) == {"posterior.pt", "loss.npz"} and m.figures == ["figures/training_loss.png"]
    assert seen == ["Training loss"]
    back = orchestrator.build_posterior(cfg, lp, out.id, False)
    assert back.id == out.id and reparam.rotation_of(back.posterior.T) is None
    with pytest.raises(ValueError, match="already carries"):
        orchestrator.build_posterior(cfg, lp, out.id, False, truncation=object())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_artifact_store.py -k auto_persists_and_returns -v` — the old tuple return fails the `isinstance`.

- [ ] **Step 3: Rewrite `build_posterior`'s edges**

Signature: replace `*, save: bool = True, save_name: str | None = None, fig_sink=None, store=None,` with `*, name: str = "", note: str = "", fig_sink=None, store=None,`; replace `accept_truncated: bool = False,` with `observation=None, parent_posterior=None, accept=None,`. Docstring: drop the `save`/`save_name`/`accept_truncated` paragraphs; add:

```
    :param name / note: the artifact's name ("" = unnamed) and note. Every training run is WRITTEN at
                     completion -- the Save button is a rename -- so a multi-day run can no longer be
                     lost to a forgotten click, and every posterior has an id its children can name.
    :param observation / parent_posterior: for a TSNPE round, the LoadedObservation the region was
                     drawn around and the LoadedPosterior it was drawn from; recorded as parents.
    :param accept: LOAD path only. An ``artifacts.Accept``; a NON-AMORTIZED artifact is refused unless
                     ``accept.truncated`` (the Posterior tab passes it), and the flag is recorded.
    :return: a LoadedPosterior; ``.posterior`` is the TransformedPosterior every downstream stage samples.
```

Add at the top of the body (after `store = resolve_store(store)` from Task 5):
```python
    from .artifacts import Accept
    from .artifacts.manifest import conditioning_block, region_to_json, tensor_digest, tensor_to_json
    accept = accept or Accept()
    _t0 = time.time()
    _spread = None
```

Load branch (L726-785) becomes:
```python
    if not train_new and ref is not None:
        if truncation is not None:
            raise ValueError(
                "build_posterior(truncation=...) restricts the prior for a NEW training run; a loaded "
                "posterior already carries whatever region it was trained under. Load it without a "
                "region, or train a new round from it.")
        loaded = store.load_posterior(cfg, ref, accept=accept)
        region = loaded.posterior.truncation
        if region is not None:
            # The region names the base prior its parent restricted; the prior loaded beside this
            # artifact must be that one (silent when either side is unverifiable, as in training).
            _assert_prior_matches_region(region, inferred, f"Posterior '{loaded.name or loaded.id}'")
            print(f"[tsnpe] loaded a NON-AMORTIZED posterior: valid only near the observation with "
                  f"digest {loaded.posterior.x_obs_digest or '(not recorded)'} ({region!r}). "
                  f"Calibration restricts its prior to that region; inference on any other observation "
                  f"refuses unless told to accept it.", flush=True)
        return loaded
```
(`ref` is the old `choice` parameter, renamed.) In the rotation block keep `print(f"[fisher] eigenvalue spread ...")` but assign `_spread = float(...)` first and print `_spread`.

Resolve the flow knobs BEFORE `train_nn` so the manifest records what was used:
```python
    hf = NSF_HIDDEN_FEATURES if hidden_features is None else int(hidden_features)
    nt = NSF_NUM_TRANSFORMS if num_transforms is None else int(num_transforms)
    lr = TRAINING_LEARNING_RATE if learning_rate is None else float(learning_rate)
    patience = TRAINING_STOP_AFTER_EPOCHS if stop_after_epochs is None else int(stop_after_epochs)
```
and pass `hidden_features=hf, num_transforms=nt, learning_rate=lr, stop_after_epochs=patience` to `pipeline.train_nn`.

Replace everything from `if save:` (L1115) to the end of the function (L1151) with:
```python
    # The tsnpe kept-fraction lines stay exactly as they were (guardrail 5); they run before the write.
    _acc = _in = _tot = None
    if truncation is not None and hasattr(train_prior, "acceptance_rate"):
        _acc = float(train_prior.acceptance_rate)
        _in, _tot = getattr(train_prior, "recorded_counts", (0, 0))
        print(...)   # the two existing [tsnpe] prints, unchanged
    assert isinstance(posterior_latent, DirectPosterior)

    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    V_rec = rotation_of(T_train)             # THE one decoder of the convention (D6); None when unrotated
    probe = training_checkpoint.bijection_probe(T_train, len(keys), device=cfg.hw.device)
    diag = pos_diagnostics or {}
    _best = diag.get("best_validation_loss")
    with store.create("posterior", cfg, name=name, note=note) as w:
        file_manager.atomic_torch_save(posterior_latent, w.payload("posterior.pt"))
        if diag.get("validation_loss"):
            file_manager.atomic_savez(w.payload("loss.npz"), dict(
                training_loss=np.asarray(diag.get("training_loss", []), dtype=float),
                validation_loss=np.asarray(diag.get("validation_loss", []), dtype=float),
                best_validation_loss=float(_best if _best is not None else float("nan")),
                epochs_trained=int(diag.get("epochs_trained") or -1),
                stop_after_epochs=int(diag.get("stop_after_epochs") or -1)))
            fig_loss = visualizers.plot_training_loss(diag)
            if fig_loss is not None:
                w.fig_sink(fig_sink)("Training loss", fig_loss)
        w.parents = {"prior": prior.id}
        if ckpt_dir is not None:
            w.parents["simulation"] = ckpt_dir.name
        if parent_posterior is not None:
            w.parents["parent_posterior"] = parent_posterior.id
        if observation is not None:
            w.parents["observation"] = observation.id
        w.fingerprints = {"gmm": prior.fingerprint, "V": tensor_digest(V_rec), "probe": tensor_digest(probe)}
        w.config.update({"num_runs": n_runs, "run_size": run_size, "hidden_features": hf, "num_transforms": nt,
                         "learning_rate": lr, "stop_after_epochs": patience, "fisher_m": fisher_m,
                         "fisher_dz": fisher_dz, "fisher_points": fisher_points})
        w.body = {
            "mode": cfg.observation_mode,
            "conditioning": conditioning_block(cfg),
            "transform": {"param_keys": keys,
                          "log_params": list(resolved_log_params(cfg, log_params=_log_params_for(cfg))),
                          "nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
                          "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
                          "rescale_lows": [float(b[0]) for _, b in cfg.rescale_params.values()],
                          "rescale_highs": [float(b[1]) for _, b in cfg.rescale_params.values()],
                          "V": tensor_to_json(V_rec), "V_orientation": "columns",
                          "fisher_eigenvalues": tensor_to_json(fisher_evals), "V_digest": tensor_digest(V_rec)},
            "amortized": truncation is None,
            "truncation": None if truncation is None else region_to_json(truncation),
            "training": {"n_runs": n_runs, "run_size": run_size,
                         "resumed_from_batch": int(_st["batches_done"]) if ckpt_resumed is not None else None,
                         "hidden_features": hf, "num_transforms": nt, "learning_rate": lr,
                         "stop_after_epochs": patience,
                         "epochs_trained": diag.get("epochs_trained"),
                         "best_validation_loss": (float(_best) if _best is not None and math.isfinite(float(_best)) else None),
                         "fisher_spread": (_spread if _spread is not None and math.isfinite(_spread) else None),
                         "tsnpe_acceptance": _acc,
                         "tsnpe_containment": None if _tot is None else {"inside": int(_in), "total": int(_tot)},
                         "wall_seconds": time.time() - _t0},
        }
    # Read back through the loader: the freshly written artifact is verified exactly as a later load
    # would verify it (box, width, the rotation against the pickled prior, the region's basis).
    loaded = store.load_posterior(cfg, w.id, accept=Accept(truncated=True))
    loaded.diagnostics = pos_diagnostics
    loaded.accepted = []                     # a round just trained is not an "accepted" load
    return loaded
```
Remove `x_obs_digest`'s use in the `[tsnpe] this artifact will be marked NON-AMORTIZED` print only if it is gone; it stays for now. `visualizers.plot_training_loss(diag)` is the no-`save_path` call (the figure goes through the writer's sink; the sink saves it).

Delete `_assert_mode_matches` and `save_posterior_artifacts`. Imports: drop `POSTERIOR_PATH, PLOT_PATH` from the config tuple; from the run_guards list drop `truncation_from_sidecar, _assert_amortization_understood`; from the reparam list drop `load_eval_bijection, read_sidecar, reconcile_loaded_rotation` (keep `rotation_of`, `rotation_of_prior`, `resolved_log_params`, `transform_device`, `UnitToBoxTransform`, `build_rotated_bijection`, `RotatedLatentPrior`, `build_inferred_bijection`, `build_rescale_bijection`, `nd_log_mask`, `TransformedPosterior`, `posterior_mode as reparam_posterior_mode` only if still used — `grep -n reparam_posterior_mode core/orchestrator.py`; drop it if not).

In `core/SBI/reparam.py` delete `reconcile_loaded_rotation`, `sidecar_path`, `read_sidecar`, `load_eval_bijection`. In `core/SBI/run_guards.py` delete `truncation_from_sidecar`, `_assert_amortization_understood`, `from core.config import POSTERIOR_PATH, SimConfig` → `from core.config import SimConfig`, and `from core.SBI.reparam import read_sidecar`. `grep -rn "read_sidecar\|load_eval_bijection\|reconcile_loaded_rotation\|sidecar_path" core scripts` must list only `scripts/` (re-pointed in Task 13).

- [ ] **Step 4: The session and the tabs**

`core/gui/session.py` — replace `SbiSession`:
```python
@dataclass
class SbiSession:
    draft: Any = None               # ConfigDraft from the Config tab (model + units + knobs)
    cfg: Any = None                 # SimConfig (built at the Prior stage, once bounds are chosen)
    inf_prior: Any = None           # artifacts.LoadedPrior -- .prior (ProductPrior), .force_prior, .id, .manifest
    posterior: Any = None           # artifacts.LoadedPosterior -- .posterior (TransformedPosterior, carrying
                                    # .T, .truncation, .x_obs_digest), .latent, .id, .manifest, .diagnostics
    observation: Any = None         # artifacts.LoadedObservation -- the Infer tab's last product; the TSNPE tab's input

    def reset_downstream(self, from_stage: str) -> None:
        """Invalidate artifacts that depend on an earlier stage when it is re-run. Everything here is
        already on disk (every stage writes at completion), so this only drops the session's handles."""
        order = ["config", "prior", "posterior", "validate"]
        i = order.index(from_stage)
        if i <= order.index("prior"):
            self.inf_prior = self.observation = None     # an observation is tied to a config's conditioning
        if i <= order.index("posterior"):
            self.posterior = None
```
Then `grep -rn "session\.\(force_prior\|truncation\|x_obs_digest\|diagnostics\|posterior_latent\|V\)\b\|\bs\.\(force_prior\|truncation\|x_obs_digest\|diagnostics\|posterior_latent\|V\)\b" core/gui tests` and fix every hit:

- `prior_tab._on_prior`: delete the `self.session.force_prior = ...` line (Task 4 added it).
- `posterior_tab.py`: `from core.config import POSTERIOR_PATH` → removed; `from ...widgets.artifact_picker import StorePicker`; `from core.artifacts import Accept, default_store`; L37-38 `self.post_picker = StorePicker("posterior", allow_new=True)`; `_build_posterior`'s dispatch: `orchestrator.build_posterior, cfg, self.session.inf_prior, entry, is_new, num_runs=n_runs, run_size_cap=cap, accept=Accept(truncated=True), hidden_features=..., ... provide_fig_sink=True, on_result=self._on_posterior` (drop `save=False`); `_on_posterior`:
  ```python
    def _on_posterior(self, payload):
        self.session.posterior = payload                 # a LoadedPosterior
        region = payload.posterior.truncation
        if region is not None:
            self.log_pane.append_line(
                f"This posterior is NON-AMORTIZED (observation digest {payload.posterior.x_obs_digest}): "
                f"Validate restricts its prior to the region it was trained on; Infer refuses any other "
                f"observation unless told to accept it.", "warning")
        self.log_pane.append_line(f"Posterior ready: {payload.name or '(unnamed, id ' + payload.id + ')'}. "
                                  f"Name it below to keep it.")
        self._screen.refresh_gates()
  ```
  `_save_posterior` mirrors the prior tab's rename (`default_store().rename("posterior", lp.id, name)`, `lp.name = name`, `self.post_picker.refresh(); self.post_picker.restore_key(lp.id)`); delete `_extract_rotation`; `refresh_local_gates`: `self.btn_save_post.setEnabled(self.session.posterior is not None)`.
- `tsnpe_tab.py`: `_on_round(self, payload)`: `s.posterior = payload` (a LoadedPosterior) and the existing warning line reading the digest from `payload.posterior.x_obs_digest`; L106 stays `s.inf_prior` (Task 5).
- `validate_tab.py` L46-50 (interim until Task 9):
  ```python
        self.dispatch(orchestrator.validate_calibration, s.cfg, s.posterior.posterior,
                      s.inf_prior.prior, s.inf_prior.force_prior, provide_fig_sink=True,
                      n_cal=max(1, self.cal_n.value()), cal_n_scales=max(1, self.cal_scales.value()),
                      truncation=s.posterior.posterior.truncation)
  ```
- `infer_tab.py`: where it reads `post = self.session.posterior` for the runners, pass `post.posterior` (interim until Task 10).
- `runners._run_tsnpe_round`: `return orchestrator.build_posterior(cfg, prior, None, True, fig_sink=fig_sink, num_runs=num_runs, run_size_cap=run_size_cap, truncation=region, x_obs_digest=rec.get("digest"))` — the tokens `build_truncation_region` and `truncation=region` stay (`test_nav_and_gating` pins them).
- `orchestrator.run` L83-89: `lp_post = build_posterior(cfg, lp, pos_choice, train_new, accept=Accept(truncated=True))`, `posterior = lp_post.posterior`, and the `validate_calibration` call reads `truncation=posterior.truncation`. `helpers.clear_screen()` stays.

- [ ] **Step 5: The suites**

- `tests/_fixtures.py`: move the helpers (Step 1 of this task's header lists them) and add `_LoadedStub` from Task 5; `tests/test_artifact_store.py` and the suites below import from it.
- `tests/test_artifact_consistency.py` L201-570: the `_sidecar` helper (L201-230) goes; `test_eval_box_comes_from_the_sidecar…` (~L230), `test_a_posterior_over_different_parameters…` (~L258), `…_cycle_ceiling…` (~L282) and `test_a_non_amortized_artifact_loads_only_when_accepted…` (L342-372) are replaced by Task 6's store tests — delete them; `test_a_gui_saved_rotation…` (~L430) and `test_a_transposed_sidecar_is_reconciled…` (~L463) are replaced by `test_manifest_V_must_equal_the_rotation_in_the_pickled_prior` — delete them; `test_the_sidecar_records_the_fisher_eigenvalues…` (~L543) becomes an assertion in the store suite that a rotated `build_posterior` run records `transform.fisher_eigenvalues` (add to `test_build_posterior_auto_persists_and_returns_the_loaded_wrapper` with `cfg.reparam_rotate = True` and `decorrelate.build_latent_fisher_rotation` monkeypatched to return `(Q, torch.arange(13, 0, -1).float())`: `assert m.body["transform"]["fisher_eigenvalues"][0] == 13.0`). The torn-write test (L571-600) becomes: write a posterior through `store.create`, inject `_failing` into `file_manager.atomic_torch_save`'s `torch.save`, assert the directory is gone and the first artifact intact:
  ```python
  def test_a_torn_posterior_write_leaves_no_half_artifact(store):
      cfg = _nad_cfg(chi_mode=True)
      first = _posterior_artifact(store, cfg, name="first")
      real_save = torch.save
      torch.save = _failing(real_save)
      try:
          with pytest.raises(_WriteFailed):
              with store.create("posterior", cfg, name="second") as w:
                  file_manager.atomic_torch_save({"generation": 2}, w.payload("posterior.pt"))
      finally:
          torch.save = real_save
      assert [r.name for r in store.list("posterior")] == ["first"] and not (w.dir).exists()
      assert not list(first.dir.glob("*.tmp"))
  ```
- `tests/test_nav_and_gating.py` L289-365 (`test_a_tsnpe_posterior_cannot_be_saved_as_amortized`): rewrite over the store — build a non-amortized posterior artifact with `_posterior_artifact(store, cfg, amortized=False, region=...)`, load it with `Accept(truncated=True)`, call `inf.tsnpe_panel._on_round(loaded)`, assert `inf.session.posterior.manifest.body["amortized"] is False`; type a name and call `_save_posterior`, assert `store.get("posterior", loaded.id).name` is that name and the body is still non-amortized; then `_on_posterior(amortized_loaded)` and assert the session's posterior is amortized. The V-forwarding test (the one that poisons `session.V`) is deleted; its successor is Task 6's rotation test. Gate tests that set `inf.session.posterior = object()` are unchanged.
- `tests/test_conditioning_repair.py`: L129 (a legacy posterior expected to load) → expect `StoreError`; the AST pin on `run` (~L951) asserts `accept=Accept(truncated=True)` is passed and drops the `truncation=` assertion (the region rides on the wrapper).
- `tests/test_user_sbi.py`: `post, _ = orchestrator.build_posterior(...)` → `lp_post = orchestrator.build_posterior(...)` and `post = lp_post.posterior`; drop `save=False`; `accept_truncated=True` → `accept=Accept(truncated=True)`; the guardrail-7 test's `_LoadedStub(prior)` from Task 5.

- [ ] **Step 6: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py tests/test_artifact_consistency.py tests/test_nav_and_gating.py tests/test_conditioning_repair.py tests/test_settings_persistence.py -q`; `pytest tests/test_user_sbi.py -k "tsnpe or identity or checkpoint or no_forcing" -q`; `pytest -m "not slow" -q` — green.

```bash
git add core tests
git commit -F - <<'MSG'
posterior: build_posterior returns a LoadedPosterior, auto-persists, the sidecar is retired (piece 1, Task 7). A training run writes posterior.pt, loss.npz, the loss figure and a manifest carrying the transform block (box, log params, V in columns, Fisher eigenvalues), the conditioning geometry, amortized/truncation, every resolved knob and the training numbers, with the prior, the simulation cache, the parent posterior and the observation as parents -- then reads itself back through store.load_posterior. The load branch is the store's. Deleted: save_posterior_artifacts, _assert_mode_matches, reparam.read_sidecar/sidecar_path/load_eval_bijection/reconcile_loaded_rotation, run_guards.truncation_from_sidecar/_assert_amortization_understood. SbiSession holds the wrappers (inf_prior, posterior, observation); the Save button renames. The sidecar suite becomes store tests; the torn-write test asserts no half-artifact.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 8: The observation stage writes its artifact; the experimental path is fixed and hashed

**Files:**
- Modify: `core/SBI/observations.py` (L68, L82; add `RecordingSet`), `core/artifacts/store.py` (add `load_observation`), `core/orchestrator.py` (`generate_observations` L144-286, delete `save_observation`/`load_observation`/`PERSIST_OBSERVATIONS` L442 and L1162-1195, `observation_digest` becomes a delegate, `infer_and_visualize` L1874-1884, `run` L92-139), `core/gui/panels/inference/runners.py`, `core/gui/panels/inference/infer_tab.py` (`_infer` L305-370), `core/gui/panels/inference_tabs.py` (re-exports), `tests/test_nav_and_gating.py:818-832`, `tests/test_conditioning_repair.py:925-948`, `tests/test_user_sbi.py` (every `generate_observations(` and `build_experiment_obs*` call)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Produces: `observations.RecordingSet(spont, forced=(), T_obs_s=0.0, forcing_params_si=None, F0_si=None)` (frozen dataclass; `forced` is `((path, freq_Hz | None), ...)`); `orchestrator.generate_observations(cfg, *, name="", note="", fig_sink=None, store=None) -> LoadedObservation`; `orchestrator.build_experiment_observation(cfg, rec, *, name="", note="", fig_sink=None, store=None) -> LoadedObservation`; `ArtifactStore.load_observation(cfg, ref) -> LoadedObservation`; `runners._run_simulated_inference(...) -> LoadedObservation`, `runners._run_experimental_inference(cfg, posterior, rec, *, fig_sink=None) -> LoadedObservation` (the `_chi` and `_spontaneous` runners are folded into it).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_store.py`:

```python
def _forced_cfg():
    from core import cli
    from core.config import CELL_PATH
    cfg = _nad_cfg()                                         # master.txt declares a drive -> forced mode
    cli.load_and_validate_gt(cfg, str(CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.T_obs = 200.0                                        # cell units: 200 frames at dt_exp = 1
    return cfg


def test_generate_observations_writes_an_artifact_that_reinstalls_its_context(store):
    from core import orchestrator
    cfg = _forced_cfg()
    seen = []
    obs = orchestrator.generate_observations(cfg, fig_sink=lambda t, f: seen.append(t), name="cell")
    assert obs.name == "cell" and obs.mode == "forced" and obs.width == obs.x_obs.shape[-1]
    assert seen == ["Ground-truth trace"] and obs.manifest.figures == ["figures/ground_truth_trace.png"]
    src = obs.manifest.body["source"]
    assert src["kind"] == "simulated" and src["cell"]["path"] == "Cells/nadrowski/master_weak.txt"
    assert src["params"] == {k: float(v) for k, (v, _) in cfg.params_dict.items()}
    assert obs.manifest.fingerprints["x_obs"] == obs.digest == mf.tensor_digest(obs.x_obs)
    fresh = _nad_cfg()
    assert not fresh.has_ground_truth
    store.load_observation(fresh, obs.id).install(fresh)
    assert fresh.has_ground_truth and fresh.T_obs == cfg.T_obs and fresh.n_obs == cfg.n_obs
    assert fresh.ground_truth == cfg.ground_truth
    with pytest.raises(ValueError, match="mode"):
        store.load_observation(_nad_cfg(chi_mode=True), obs.id)


def test_build_experiment_observation_hashes_recordings_and_refuses_a_missing_file(store, tmp_path):
    """Defects 3-4: the forced-recording path raised NameError before this task. It now runs, and the
    artifact names every recording by path and hash."""
    import numpy as np
    from core import orchestrator
    from core.SBI.observations import RecordingSet
    cfg = _forced_cfg()
    sim = orchestrator.generate_observations(cfg, fig_sink=lambda t, f: None)
    trace = sim.obs_data[0].numpy()
    spont, forced = tmp_path / "spont.npy", tmp_path / "forced.npy"
    np.save(spont, trace)
    np.save(forced, trace)
    T_obs_s = cfg.T_obs / cfg.get_unit_conversion_factor("s")
    si = {n: (5.0 if n == "freq" else 1e-12 if n == "amp" else 0.0) for n in cfg.force_params_dict}
    rec = RecordingSet(spont=str(spont), forced=((str(forced), None),), T_obs_s=T_obs_s, forcing_params_si=si)
    obs = orchestrator.build_experiment_observation(cfg, rec, fig_sink=lambda t, f: None)
    recs = obs.manifest.body["source"]["recordings"]
    assert [r["role"] for r in recs] == ["spont", "forced"] and all(len(r["sha256"]) == 64 for r in recs)
    assert obs.width == sim.width and obs.manifest.body["source"]["kind"] == "experimental"
    assert obs.manifest.body["forcing_vals"]["freq"] > 0
    with pytest.raises(FileNotFoundError):
        orchestrator.build_experiment_observation(
            cfg, RecordingSet(spont=str(tmp_path / "nope.npy"), forced=((str(forced), None),), T_obs_s=T_obs_s,
                              forcing_params_si=si), fig_sink=lambda t, f: None)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_artifact_store.py -k observation -v` — `TypeError` (`generate_observations` takes no `fig_sink`), `ImportError` (`RecordingSet`).

- [ ] **Step 3: Fix the observation builder and add `RecordingSet`**

In `core/SBI/observations.py`: L68 `budget = N_ND_MAX - cfg.steady_idx` → `budget = config.N_ND_MAX - cfg.steady_idx`; L82 `_FORCING_SI_UNITS = FORCING_SI_UNITS` → `_FORCING_SI_UNITS = config.FORCING_SI_UNITS`. Add after the imports:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RecordingSet:
    """What the bench produced, by FILE: the passive recording, the forced recordings with the
    frequency each was actually driven at (None in single-drive forced mode, whose drive is
    ``forcing_params_si``), the duration, and the drive. The observation stage hashes every file
    into the artifact's manifest, so a posterior's inputs are on record."""
    spont: str
    forced: tuple = ()                    # ((path, drive frequency in Hz or None), ...)
    T_obs_s: float = 0.0
    forcing_params_si: "dict | None" = None   # forced mode: {"amp", "freq", "phase", ...} in SI
    F0_si: "float | None" = None              # chi mode: physical drive amplitude (N)
```

- [ ] **Step 4: Add `ArtifactStore.load_observation`**

In `core/artifacts/store.py`, after `load_posterior`:

```python
    def load_observation(self, cfg, ref: str) -> LoadedObservation:
        """Refuses an observation whose model, parameter order, mode, conditioning width or chi
        layout/pad is not this config's; asserts the payload's digest is the manifest's."""
        import torch
        from core import config as _config
        from core.SBI.statistics import SUMMARY_WIDTH
        from core.orchestrator import expected_forcing_dim
        sub, m = self._find("observation", ref)
        if m is None:
            raise StoreError(f"no complete observation named or id'd {ref!r} under {self.kind_dir('observation')}")
        label = m.name or m.id
        body, cond = m.body, m.body["conditioning"]
        want_keys = list(cfg.params_dict) + list(cfg.rescale_params)
        if m.config.get("model") != cfg.model:
            raise ValueError(f"Observation '{label}' was recorded for model {m.config.get('model')}, not {cfg.model}.")
        if list(m.config.get("param_keys") or []) != want_keys:
            raise ValueError(f"Observation '{label}' was recorded over parameters {m.config.get('param_keys')}, "
                             f"not this config's {want_keys}.")
        if body["mode"] != cfg.observation_mode:
            raise ValueError(f"Observation '{label}' is a {str(body['mode']).upper()}-mode observation, but this "
                             f"config is {cfg.observation_mode.upper()} mode.")
        want_dim = int(expected_forcing_dim(cfg))
        if int(cond["forcing_dim"]) != want_dim or int(cond["width"]) != SUMMARY_WIDTH + 1 + want_dim:
            raise ValueError(f"Observation '{label}' is {cond['width']} wide (block {cond['forcing_dim']}); this "
                             f"config conditions on {SUMMARY_WIDTH + 1 + want_dim} (block {want_dim}).")
        if body["mode"] == "chi" and (cond["chi_layout"] != _config.CHI_LAYOUT
                                      or int(cond["chi_k_pad"]) != int(cfg.chi_k_pad)):
            raise ValueError(f"Observation '{label}' was packed under chi layout {cond['chi_layout']} with "
                             f"{cond['chi_k_pad']} slots; this config is layout {_config.CHI_LAYOUT} / {cfg.chi_k_pad}.")
        payload = torch.load(str(sub / "observation.pt"), map_location="cpu", weights_only=False)
        x_obs = payload["x_obs"]
        if mf.tensor_digest(x_obs) != body["x_obs_digest"]:
            raise StoreError(f"observation '{label}': observation.pt does not hash to the manifest's digest; "
                             f"the artifact is inconsistent")
        return LoadedObservation("observation", m.id, m.name, m, sub, x_obs=x_obs, obs_data=payload["obs_data"],
                                 t_dim=payload["t_dim"], digest=body["x_obs_digest"], mode=body["mode"],
                                 width=int(cond["width"]))
```

- [ ] **Step 5: The two observation stages in `orchestrator.py`**

Imports: drop `OBSERVATION_PATH` from the config tuple; add `from .SBI.observations import RecordingSet` to the re-export line at the bottom; `from .artifacts import LoadedObservation` alongside `LoadedPrior`; `from .artifacts.provenance import file_ref as _file_ref`.

`generate_observations(cfg: SimConfig, *, name: str = "", note: str = "", fig_sink=None, store=None) -> LoadedObservation` — the body is unchanged up to and including the chi/forced/spontaneous branches; replace the final `return x_dim, obs_stats, t_dim` with:

```python
    src = getattr(cfg, "sources", {}).get("cell")
    source = {
        "kind": "simulated",
        "cell": _file_ref(src, relative_to=config.RESOURCES_ROOT) if src else None,
        "params": {k: float(v) for k, (v, _) in cfg.params_dict.items()},
        "rescale": {k: float(v) for k, (v, _) in cfg.rescale_params.items()},
        "forcing": {k: float(v) for k, (v, _) in cfg.force_params_dict.items()},
        "inits": {k: float(v) for k, v in cfg.inits_dict.items()},
        "T_obs_s": float(cfg.T_obs / cfg.get_unit_conversion_factor("s")),
    }
    forcing_vals = {k: float(v) for k, (v, _) in cfg.force_params_dict.items()} if (cfg.has_forcing and not cfg.chi_mode) else {}
    return _write_observation(resolve_store(store), cfg, name, note, fig_sink, obs_stats, x_dim, t_dim,
                              title="Ground-truth trace", source=source, forcing_vals=forcing_vals)


def build_experiment_observation(cfg: SimConfig, rec: RecordingSet, *, name: str = "", note: str = "",
                                 fig_sink=None, store=None) -> LoadedObservation:
    """A bench recording set -> the observation artifact. Every file is checked and hashed BEFORE any
    compute (a missing recording is a FileNotFoundError here, not a traceback inside a worker), then
    the mode's builder runs and the artifact records the recordings, the drive and the context."""
    store = resolve_store(store)
    named = [(rec.spont, "spont", None)] + [(p, "forced", f) for p, f in rec.forced]
    refs = []
    for p, role, f in named:
        if not p or not os.path.isfile(str(p)):
            raise FileNotFoundError(f"the {role} recording was not found: {p!r}")
        r = _file_ref(p)
        r.update({"role": role, "freq_Hz": None if f is None else float(f)})
        refs.append(r)
    X_spont = file_manager.load_experimental_data(rec.spont, dtype=cfg.hw.dtype)
    if cfg.observation_mode == "chi":
        forced = [(file_manager.load_experimental_data(p, dtype=cfg.hw.dtype), float(f)) for p, f in rec.forced]
        obs_stats, obs_data, t_dim = build_experiment_obs_chi(cfg, X_spont, forced, rec.T_obs_s, rec.F0_si)
    elif cfg.has_forcing:
        if len(rec.forced) != 1:
            raise ValueError(f"forced mode takes exactly one forced recording, got {len(rec.forced)}")
        X_forced = file_manager.load_experimental_data(rec.forced[0][0], dtype=cfg.hw.dtype)
        obs_stats, obs_data, t_dim = build_experiment_obs(cfg, X_spont, X_forced, rec.T_obs_s,
                                                          rec.forcing_params_si or {})
    else:
        obs_stats, obs_data, t_dim = build_experiment_obs_spontaneous(cfg, X_spont, rec.T_obs_s)
    forcing_vals = {k: float(v) for k, (v, _) in cfg.force_params_dict.items()} if (cfg.has_forcing and not cfg.chi_mode) else {}
    source = {"kind": "experimental", "recordings": refs, "T_obs_s": float(rec.T_obs_s),
              "F0_si": None if rec.F0_si is None else float(rec.F0_si),
              "forcing_params_si": None if rec.forcing_params_si is None else
              {k: float(v) for k, v in rec.forcing_params_si.items()}}
    return _write_observation(store, cfg, name, note, fig_sink, obs_stats, obs_data, t_dim,
                              title="Observed trace", source=source, forcing_vals=forcing_vals)


def _write_observation(store, cfg, name, note, fig_sink, x_obs, obs_data, t_dim, *, title, source, forcing_vals):
    from .artifacts.manifest import conditioning_block, tensor_digest, tensor_to_json
    digest = tensor_digest(x_obs)
    with store.create("observation", cfg, name=name, note=note) as w:
        file_manager.atomic_torch_save({"x_obs": x_obs.detach().cpu(), "obs_data": obs_data.detach().cpu(),
                                        "t_dim": t_dim.detach().cpu()}, w.payload("observation.pt"))
        visualizers.plot(t_dim.squeeze(0).cpu().detach().numpy(), obs_data[0, :].cpu().detach().numpy(),
                         title=title, labels=(labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)),
                         sink=w.fig_sink(fig_sink))
        w.fingerprints["x_obs"] = digest
        w.body = {"mode": cfg.observation_mode, "conditioning": conditioning_block(cfg), "x_obs_digest": digest,
                  "T_obs_cell": float(cfg.T_obs),
                  "n_obs": int(cfg.n_obs) if cfg.n_obs is not None else int(obs_data.shape[-1]),
                  "forcing_vals": forcing_vals,
                  "chi_obs_freqs": tensor_to_json(getattr(cfg, "chi_obs_freqs", None)),
                  "source": source}
    return store.load_observation(cfg, w.id)


def observation_digest(x_obs: torch.Tensor) -> str:
    """16-hex digest of a conditioning vector -- the store's tensor_digest, kept under this name."""
    from .artifacts.manifest import tensor_digest
    return tensor_digest(x_obs)
```

Delete `PERSIST_OBSERVATIONS` (L442), `save_observation` and `load_observation` (L1162-1195), and the `if PERSIST_OBSERVATIONS:` block inside `infer_and_visualize` (L1874-1884). (`visualizers.plot` accepts `sink=` — the runner already called it that way.)

`run` (L92-139): the simulated branch drops its `visualizers.plot` call and reads
```python
        obs = generate_observations(cfg)
        infer_and_visualize(cfg, posterior, obs.x_obs, obs.obs_data, obs.t_dim, show_truth=True)
```
and the three experimental branches become one:
```python
    elif mode == "experimental":
        if cfg.chi_mode:
            spont_path, forced_paths, T_obs_s, F0_si = cli.get_inference_inputs_chi()
            rec = RecordingSet(spont=spont_path, forced=tuple((p, None) for p in forced_paths), T_obs_s=T_obs_s, F0_si=F0_si)
        elif not cfg.has_forcing:
            path, T_obs_s = cli.get_inference_inputs_spontaneous()
            rec = RecordingSet(spont=path, T_obs_s=T_obs_s)
        else:
            spont_path, forced_path, T_obs_s, forcing_params_si = cli.get_inference_inputs(list(cfg.force_params_dict.keys()))
            rec = RecordingSet(spont=spont_path, forced=((forced_path, None),), T_obs_s=T_obs_s, forcing_params_si=forcing_params_si)
        obs = build_experiment_observation(cfg, rec)
        infer_and_visualize(cfg, posterior, obs.x_obs, obs.obs_data, obs.t_dim, show_truth=False)
```
(`build_experiment_obs_chi` treats a bare recording list as driven at the config's multipliers; passing `(path, None)` pairs keeps the CLI's legacy behaviour: inside `build_experiment_observation` the chi branch passes `float(f)` — so for the CLI use `forced=tuple((p, chi.chi_multipliers_for(cfg)[i] * ...)`… no: keep it simple — the CLI is retired in piece 2; pass the multipliers as frequencies: `freqs = chi.chi_multipliers_for(cfg).tolist()` is in cell units, not Hz. Simplest honest interim: the chi branch of `build_experiment_observation` passes `(rec, f)` when `f is not None` and the bare recording when `f is None`, matching `build_experiment_obs_chi`'s two accepted forms.) Implement that: `forced = [(x, float(f)) if f is not None else x for x, f in ...]`.

- [ ] **Step 6: The runners and the Infer tab**

`core/gui/panels/inference/runners.py` becomes:
```python
def _run_simulated_inference(cfg, posterior, cell_path, T_obs_s, *, gt_dicts=None, prior=None, fig_sink=None):
    ignored = (cfg.inject_ground_truth(*gt_dicts) if gt_dicts is not None
               else cli.load_and_validate_gt(cfg, cell_path))
    if ignored:
        print(f"Note: the bounds file does not declare {', '.join(ignored)} — those cell values were "
              f"ignored (the bounds file defines the inferred set).")
    cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
    if prior is not None:
        for msg in orchestrator.check_observation_in_distribution(cfg, prior.prior, prior.force_prior):
            print(f"WARNING: {msg}")
    obs = orchestrator.generate_observations(cfg, fig_sink=fig_sink)     # writes the artifact + the trace
    orchestrator.infer_and_visualize(cfg, posterior, obs.x_obs, obs.obs_data, obs.t_dim, show_truth=True, fig_sink=fig_sink)
    return obs


def _run_experimental_inference(cfg, posterior, rec, *, fig_sink=None):
    """Any bench recording set (passive, driven, chi): the stage checks and hashes the files first."""
    obs = orchestrator.build_experiment_observation(cfg, rec, fig_sink=fig_sink)
    orchestrator.infer_and_visualize(cfg, posterior, obs.x_obs, obs.obs_data, obs.t_dim, show_truth=False, fig_sink=fig_sink)
    return obs
```
(`_run_experimental_inference_chi` and `_run_experimental_inference_spontaneous` are deleted; `_run_tsnpe_round` unchanged.) `grep -n "_run_experimental_inference" core/gui/panels/inference_tabs.py tests/*.py` and update the re-export line and any test import.

`infer_tab.py` `_infer`: every `self.dispatch(...)` gains `on_result=self._on_observation`; the chi branch builds `RecordingSet(spont=self.chi_spont.value(), forced=tuple(pairs), T_obs_s=self.chi_tobs.value(), F0_si=self.chi_f0_si.value())`, the passive branch `RecordingSet(spont=self.exp_spont.value(), T_obs_s=self.exp_tobs.value())`, the driven branch `RecordingSet(spont=self.exp_spont.value(), forced=((self.exp_forced.value(), None),), T_obs_s=self.exp_tobs.value(), forcing_params_si=forcing_si)`, each dispatched as `self.dispatch(_run_experimental_inference, cfg, post, rec, provide_fig_sink=True, on_result=self._on_observation)`; add
```python
    def _on_observation(self, payload):
        self.session.observation = payload
        self.log_pane.append_line(f"Observation recorded as {payload.name or '(unnamed, id ' + payload.id + ')'}; "
                                  f"the TSNPE tab can build a region around it.")
        self._screen.refresh_gates()
```
and `from core.SBI.observations import RecordingSet`.

- [ ] **Step 7: The suites**

- `tests/test_nav_and_gating.py:818-832`: the stubbed `orchestrator.generate_observations` returns an object with `.x_obs`, `.obs_data`, `.t_dim`, `.id`, `.name` (a `types.SimpleNamespace`), and the assertion on the "Ground-truth trace" figure now expects it from the observation stage's sink — keep asserting the title was emitted.
- `tests/test_conditioning_repair.py:925-948`: delete the `PERSIST_OBSERVATIONS` save/restore lines; the stub posterior test calls `infer_and_visualize(cfg, post, x, None, None, show_truth=False)` unchanged (Task 10 changes it again).
- `tests/test_user_sbi.py`: `x_dim, obs_stats, t_dim = orchestrator.generate_observations(cfg)` → `obs = orchestrator.generate_observations(cfg, fig_sink=sink); x_dim, obs_stats, t_dim = obs.obs_data, obs.x_obs, obs.t_dim`; `build_experiment_obs_spontaneous(cfg, x_dim[0].clone(), 1.0)` and friends stay (they are still exported and pure). Delete every `orchestrator.PERSIST_OBSERVATIONS = ...` line (L50-55 and any in-test save/restore).

- [ ] **Step 8: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py tests/test_nav_and_gating.py tests/test_conditioning_repair.py -q`; `pytest tests/test_user_sbi.py -k "no_forcing or generate_observations" -q`; `pytest -m "not slow" -q` — green.

```bash
git add core tests
git commit -F - <<'MSG'
observation: the stage writes the artifact; the experimental path is fixed and hashed (piece 1, Task 8). generate_observations and the new build_experiment_observation (one entry for passive, driven and chi recording sets) write observation.pt, the trace figure and a manifest carrying the mode, the conditioning geometry, the digest, the context (T_obs, n_obs, forcing values, chi probe frequencies) and the source -- the cell with its ground truth and inits, or every recording by path and sha256 -- so LoadedObservation.install(cfg) can put a fresh session back where the inference ran. observations.py used N_ND_MAX and FORCING_SI_UNITS unimported (defects 3-4): the forced-recording path runs for the first time since the refactor. save_observation, load_observation and PERSIST_OBSERVATIONS are gone; the three experimental runners are one.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 9: Calibration writes results, ranks and figures

**Files:**
- Modify: `core/orchestrator.py` (`validate_calibration` L1647-1836), `core/artifacts/store.py` (add `load_calibration`, `load_inference`), `core/gui/panels/inference/validate_tab.py`, `tests/_fixtures.py` (add `_tiny_gen_prior` — moved from `tests/test_user_sbi.py:66-84`, which imports it — and `build_tiny_run`), `tests/conftest.py` (the `tiny_run` fixture), `tests/test_user_sbi.py` (every `validate_calibration(` call)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Produces: `validate_calibration(cfg, posterior: LoadedPosterior, prior: LoadedPrior, *, name="", note="", fig_sink=None, store=None, n_cal=None, cal_n_scales=None, num_posterior_samples=1000) -> LoadedCalibration`; `ArtifactStore.load_calibration(ref) -> LoadedCalibration`, `ArtifactStore.load_inference(ref) -> LoadedInference`; `orchestrator._num(x) -> float | None` (finite or None); the `tiny_run` fixture: `SimpleNamespace(cfg, prior, posterior, store, sink, other_prior, teardown)`.

- [ ] **Step 1: The shared tiny run**

Move `_tiny_gen_prior` (with its `**_kw`, verbatim) from `tests/test_user_sbi.py` into `tests/_fixtures.py`; `tests/test_user_sbi.py` does `from tests._fixtures import _tiny_gen_prior`. Add to `tests/_fixtures.py`:

```python
def build_tiny_run(store):
    """A REAL SBITEST prior and posterior at tiny size, inside ``store`` (make it the default first).
    Mirrors test_user_sbi.test_no_forcing_user_model_full_sbi_pipeline's setup -- and its teardown:
    read that test's finally block and do the same in ``teardown``."""
    from types import SimpleNamespace
    from core import cli, config, orchestrator, registry
    from core.Helpers import model_store
    name = "SBITEST"
    doc = {"schema_version": 1, "name": name,
           "variables": [{"name": "x", "drift": "-k*x", "D": "d0", "init": 0.5, "forcing": None}],
           "params": {"k": 1.0, "d0": 0.05}, "rescale": {"x_scale": 10.0, "t_scale": 0.01}}
    model_store.save_user_model(doc)
    registry.load_user_models()
    cfg = cli.make_sim_config(name, registry.get(name).labels, registry.state_dep_drift(name),
                              str(config.BOUNDS_PATH / name.lower() / "default.txt"))
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / name.lower() / "default.txt"))
    cfg.hw = config.cpu_device()
    cfg.hw.batch_size = 8
    cfg.T_obs = 1.0
    saved = (orchestrator.pipeline.gen_prior, orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL,
             orchestrator.TRAINING_CHECKPOINT_EVERY)
    orchestrator.pipeline.gen_prior = _tiny_gen_prior
    orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL, orchestrator.TRAINING_CHECKPOINT_EVERY = 2, 60, 0
    sink = lambda title, fig: None                                   # noqa: E731
    prior = orchestrator.build_prior(cfg, None, True, fig_sink=sink, name="tiny_prior")
    posterior = orchestrator.build_posterior(cfg, prior, None, True, fig_sink=sink, name="tiny_post",
                                             hidden_features=8, num_transforms=1, stop_after_epochs=1)

    def other_prior():
        return orchestrator.build_prior(cfg, None, True, fig_sink=sink)   # another fit, another GMM

    def teardown():
        (orchestrator.pipeline.gen_prior, orchestrator.TRAINING_NUM_RUNS, orchestrator.SBC_N_CAL,
         orchestrator.TRAINING_CHECKPOINT_EVERY) = saved
        # + whatever test_no_forcing_user_model_full_sbi_pipeline's finally does to unregister SBITEST
    return SimpleNamespace(cfg=cfg, prior=prior, posterior=posterior, store=store, sink=sink,
                           other_prior=other_prior, teardown=teardown)
```

Append to `tests/conftest.py`:

```python
@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A real prior + posterior for the SBITEST user model at tiny size, in a store of its own that is
    the process default for the module. Minutes on CPU, built once per module."""
    from core.artifacts import ArtifactStore, use_store
    from tests._fixtures import build_tiny_run
    with use_store(ArtifactStore(tmp_path_factory.mktemp("tiny"))) as s:
        run = build_tiny_run(s)
        try:
            yield run
        finally:
            run.teardown()
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_artifact_store.py`:

```python
def test_calibration_writes_results_ranks_figures_and_refuses_a_foreign_prior(tiny_run):
    import numpy as np
    from core import orchestrator
    r = tiny_run
    cal = orchestrator.validate_calibration(r.cfg, r.posterior, r.prior, fig_sink=r.sink, n_cal=8, cal_n_scales=2,
                                            num_posterior_samples=40, name="cal")
    m, res = cal.manifest, cal.results
    keys = list(r.cfg.params_dict) + list(r.cfg.rescale_params)
    assert m.parents == {"posterior": r.posterior.id, "prior": r.prior.id} and m.config["n_cal"] == 8
    assert list(res["sbc"]["per_param"]) == keys
    assert set(res["sbc"]["per_param"][keys[0]]) == {"ks_p", "c2st_ranks", "c2st_dap"}
    assert set(res["tarp"]) == {"atc", "ks_p"} and res["num_posterior_samples"] == 40 and res["kept_fraction"] is None
    assert res["informativeness"] is None or "total_nats" in res["informativeness"]
    assert set(m.payloads) == {"ranks.npz", "results.json"}
    assert sorted(m.figures) == ["figures/sbc_ranks_cdf.png", "figures/sbc_ranks_histogram.png", "figures/tarp_coverage.png"]
    assert np.load(cal.path / "ranks.npz")["ranks"].shape[0] == 8
    assert json.loads((cal.path / "results.json").read_text(encoding="utf-8"))["tarp"] == res["tarp"]
    assert r.store.load_calibration(cal.id).results == res
    with pytest.raises(ValueError, match="not the one this posterior was trained with"):
        orchestrator.validate_calibration(r.cfg, r.posterior, r.other_prior(), fig_sink=r.sink, n_cal=4, cal_n_scales=1)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pytest tests/test_artifact_store.py -k calibration -v` — `TypeError` (the old signature wants `force_prior`).

- [ ] **Step 4: Rewrite `validate_calibration`'s edges**

Signature: `def validate_calibration(cfg: SimConfig, posterior: LoadedPosterior, prior: LoadedPrior, *, name: str = "", note: str = "", fig_sink=None, store=None, n_cal: int | None = None, cal_n_scales: int | None = None, num_posterior_samples: int = 1000) -> LoadedCalibration:`. Docstring: the `:param inferred_prior:`/`:param truncation:` paragraphs become `:param posterior / prior: the LoadedPosterior and the LoadedPrior it was trained from; the region comes off the posterior.` and `:param num_posterior_samples: draws per calibration point for SBC and TARP (1000 = the historical constant).`

Body: first lines
```python
    store = resolve_store(store)
    post, inferred_prior, force_prior = posterior.posterior, prior.prior, prior.force_prior
    truncation = post.truncation
    nps = int(num_posterior_samples)
    _assert_prior_used_matches_posterior(post, inferred_prior, "SBC/TARP calibration")
```
then `with store.create("calibration", cfg, name=name, note=note) as w:` wrapping everything from `t = cfg.t` to the end, indented one level, with: every `posterior` read as `post`; `1000` → `nps` (run_sbc, check_sbc, sbc_rank_plot x2, run_tarp); `n_cal=SBC_N_CAL if n_cal is None else int(n_cal)` resolved once into `n_cal_used` before `gen_cal_data`; `sink = w.fig_sink(fig_sink)` and `sink("SBC ranks (CDF)", f_cdf)`, `sink("SBC ranks (histogram)", f_hist)` (the `else: plt.show()` branch is deleted), `sink("TARP coverage", plt.gcf())` in place of `_emit(...)`; the informativeness `try` assigns `info = analysis.informativeness(...)` and the `except` sets `info = None`. After it:

```python
        file_manager.atomic_savez(w.payload("ranks.npz"), {
            "ranks": ranks.detach().cpu().numpy(), "theta_star": theta_star.detach().cpu().numpy(),
            "ecp": ecp.detach().cpu().numpy(), "alpha_grid": alpha_grid.detach().cpu().numpy()})
        keys = list(cfg.params_dict) + list(cfg.rescale_params)
        results = {
            "sbc": {"per_param": {k: {"ks_p": _num(sbc_stats["ks_pvals"][j]), "c2st_ranks": _num(sbc_stats["c2st_ranks"][j]),
                                      "c2st_dap": _num(sbc_stats["c2st_dap"][j])} for j, k in enumerate(keys)}},
            "tarp": {"atc": _num(atc), "ks_p": _num(tarp_kspval)},
            "informativeness": None if info is None else {
                "total_nats": _num(info["total_nats"]), "sem_nats": _num(info["sem_nats"]),
                "per_param": None if info["per_param"] is None else [_num(v) for v in info["per_param"]],
                "per_direction": None if info["per_direction"] is None else [_num(v) for v in info["per_direction"]],
                "n_used": int(info["n_used"]), "n_dropped": int(info["n_dropped"]),
                "description": analysis.describe_informativeness(info)},
            "kept_fraction": None if truncation is None else {
                "acceptance": _num(val_latent_prior.acceptance_rate),
                "containment": _num(val_latent_prior.recorded_containment)},
            "n_cal": int(n_cal_used), "cal_n_scales": None if cal_n_scales is None else int(cal_n_scales),
            "num_posterior_samples": nps,
        }
        w.payload("results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        w.parents = {"posterior": posterior.id, "prior": prior.id}
        w.fingerprints["gmm"] = prior.fingerprint
        w.config.update({"n_cal": int(n_cal_used), "cal_n_scales": results["cal_n_scales"], "num_posterior_samples": nps})
        w.body = {"results": results}
    return store.load_calibration(w.id)
```
Add module-level `import json` and:
```python
def _num(x):
    """A finite float, or None -- manifests refuse NaN/inf and a diagnostic may legitimately produce one."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None
```
`store.py` gains:
```python
    def load_calibration(self, ref: str) -> LoadedCalibration:
        sub, m = self._find("calibration", ref)
        if m is None:
            raise StoreError(f"no complete calibration named or id'd {ref!r}")
        return LoadedCalibration("calibration", m.id, m.name, m, sub, results=dict(m.body["results"]))

    def load_inference(self, ref: str) -> LoadedInference:
        sub, m = self._find("inference", ref)
        if m is None:
            raise StoreError(f"no complete inference named or id'd {ref!r}")
        return LoadedInference("inference", m.id, m.name, m, sub, results=dict(m.body["results"]),
                               samples_path=sub / "samples.pt")
```

`validate_tab._validate`:
```python
        self.dispatch(orchestrator.validate_calibration, s.cfg, s.posterior, s.inf_prior, provide_fig_sink=True,
                      n_cal=max(1, self.cal_n.value()), cal_n_scales=max(1, self.cal_scales.value()),
                      on_result=self._on_calibration)

    def _on_calibration(self, payload):
        res = payload.results
        info = res.get("informativeness") or {}
        self.log_pane.append_line(
            f"Calibration recorded as {payload.name or '(unnamed, id ' + payload.id + ')'}: TARP ATC="
            f"{res['tarp']['atc']}, KS p={res['tarp']['ks_p']}"
            + (f"; informativeness {info['total_nats']:.2f} nats" if info.get("total_nats") is not None else ""))
```
`run` (CLI): `validate_calibration(cfg, lp_post, lp)`. `tests/test_user_sbi.py`: `validate_calibration(cfg, posterior, inferred_prior, force_prior, fig_sink=sink)` → `validate_calibration(cfg, lp_post, lp, fig_sink=sink)`; a call passing `truncation=` drops it.

- [ ] **Step 5: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py -k "calibration" -q` (the `tiny_run` build takes minutes on its first use); `pytest tests/test_user_sbi.py -k "no_forcing or calibration" -q`; `pytest -m "not slow" -q` — green.

```bash
git add core tests
git commit -F - <<'MSG'
calibration: validate_calibration writes results, ranks and figures (piece 1, Task 9). The stage takes the LoadedPosterior and its LoadedPrior, calibrates on the posterior's own region when it has one, and writes a calibration artifact: results.json (SBC per parameter, TARP, the informativeness block that was computed and discarded, the kept fraction for a truncated posterior), ranks.npz, the three figures, and a manifest naming the posterior and prior as parents. num_posterior_samples is a parameter (was the literal 1000 in four places). tests/_fixtures.build_tiny_run and the module-scoped tiny_run fixture give the suites one real tiny prior + posterior.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 10: Inference takes the wrappers and writes results

**Files:**
- Modify: `core/orchestrator.py` (`infer_and_visualize` L1839-2031, `run`), `core/gui/panels/inference/runners.py`, `core/gui/panels/inference/infer_tab.py` (`_on_observation`, the `post` argument), `tests/test_conditioning_repair.py` (delete L900-948), `tests/test_user_sbi.py` (every `infer_and_visualize(` call)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Produces: `infer_and_visualize(cfg, posterior: LoadedPosterior, observation: LoadedObservation, *, name="", note="", fig_sink=None, store=None, accept=None, n_samples=1000) -> LoadedInference` (`show_truth` is derived from the observation's source; `obs_stats/obs_data/t_dim` are gone); `runners._run_simulated_inference(...) -> tuple[LoadedObservation, LoadedInference]`, `_run_experimental_inference(...) -> tuple[LoadedObservation, LoadedInference]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_store.py`:

```python
def test_inference_records_ppc_summary_and_ground_truth(tiny_run):
    from core import orchestrator
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="obs")
    inf = orchestrator.infer_and_visualize(r.cfg, r.posterior, obs, fig_sink=r.sink, n_samples=50, name="inf")
    m, res = inf.manifest, inf.results
    keys = list(r.cfg.params_dict) + list(r.cfg.rescale_params)
    assert m.parents == {"posterior": r.posterior.id, "observation": obs.id} and res["n_samples"] == 50
    assert res["accepted"] == [] and list(res["posterior_summary"]) == keys
    assert set(res["posterior_summary"][keys[0]]) == {"q05", "median", "q95"}
    assert res["ground_truth"] == {k: float(v) for k, v in zip(keys, r.cfg.ground_truth)}
    assert {"mean_abs_z", "max_abs_z", "coverage_90", "num_outside", "num_invalid"} <= set(res["ppc"])
    assert {"figures/posterior_corner.png", "figures/posterior_predictive_check.png", "figures/eye_test.png"} <= set(m.figures)
    assert set(m.payloads) == {"samples.pt", "results.json"}
    assert tuple(torch.load(inf.samples_path, weights_only=False).shape) == (50, len(keys))
    assert r.store.load_inference(inf.id).results == res and m.fingerprints["x_obs"] == obs.digest


def test_inference_refuses_a_foreign_observation_for_a_truncated_posterior_unless_accepted(tiny_run):
    """Successor of test_conditioning_repair's warning test: a NON-AMORTIZED posterior on an observation
    other than its region's is a REFUSAL now, and accepting it is written into the inference."""
    from copy import copy
    from core import orchestrator
    from core.artifacts import Accept
    from core.SBI import reparam
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink)
    claims_another = copy(r.posterior)
    claims_another.posterior = reparam.TransformedPosterior(r.posterior.latent, r.posterior.posterior.T,
                                                             truncation=None, x_obs_digest="f" * 16)
    with pytest.raises(ValueError, match="NOT AMORTIZED"):
        orchestrator.infer_and_visualize(r.cfg, claims_another, obs, fig_sink=r.sink, n_samples=20)
    inf = orchestrator.infer_and_visualize(r.cfg, claims_another, obs, fig_sink=r.sink, n_samples=20,
                                           accept=Accept(other_observation=True))
    assert inf.results["accepted"] == ["other_observation"]
    same = copy(r.posterior)
    same.posterior = reparam.TransformedPosterior(r.posterior.latent, r.posterior.posterior.T,
                                                  truncation=None, x_obs_digest=obs.digest)
    assert orchestrator.infer_and_visualize(r.cfg, same, obs, fig_sink=r.sink, n_samples=20).results["accepted"] == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_artifact_store.py -k inference -v` — `TypeError` (old positional signature).

- [ ] **Step 3: Rewrite `infer_and_visualize`'s edges**

Signature: `def infer_and_visualize(cfg: SimConfig, posterior: LoadedPosterior, observation: LoadedObservation, *, name: str = "", note: str = "", fig_sink=None, store=None, accept=None, n_samples: int = 1000) -> LoadedInference:`. Docstring: replace the parameter text with `:param posterior / observation: the wrappers; the observation's context (T_obs, n_obs, the drive, a simulated cell's truth) is put back on cfg by observation.install, so this runs in a fresh session too. show_truth = the observation is a simulated cell. :param accept: Accept(other_observation=True) lets a NON-AMORTIZED posterior run on an observation other than its region's; the flag is recorded in the artifact. :param n_samples: posterior draws for the corner, the PPC and the summary (1000 = the historical constant).`

Body start (replacing L1850-1884):
```python
    from .artifacts import Accept
    store = resolve_store(store)
    accept = accept or Accept()
    post = posterior.posterior
    observation.install(cfg)
    show_truth = observation.manifest.body["source"]["kind"] == "simulated"
    p_body = posterior.manifest.body
    if observation.mode != p_body["mode"] or observation.width != int(p_body["conditioning"]["width"]):
        raise ValueError(
            f"Observation '{observation.name or observation.id}' is {observation.mode} / {observation.width} wide, "
            f"but posterior '{posterior.name or posterior.id}' conditions on {p_body['mode']} / "
            f"{p_body['conditioning']['width']}. They do not describe the same measurement.")
    t, device, dtype, T_obs = cfg.t, cfg.hw.device, cfg.hw.dtype, cfg.T_obs
    inits = _observation_inits(cfg)
    obs_stats, obs_data, t_dim = observation.x_obs.to(device), observation.obs_data, observation.t_dim
    # GUARDRAIL 2 at the one place every inference passes through -- a REFUSAL now, not a warning:
    # outside its region a truncated flow extrapolates confidently. Accept(other_observation=True)
    # is the recorded exception (a simulated cell re-drawn with new noise is the legitimate case).
    _want, accepted = post.x_obs_digest, []
    if _want is not None and observation.digest != _want:
        _msg = (f"[tsnpe] this posterior is NOT AMORTIZED. TSNPE trained it on a prior restricted to a region drawn "
                f"around the observation with digest {_want}; the observation supplied has digest "
                f"{observation.digest}. Near that observation (the same cell re-simulated with new noise, say) it is "
                f"valid; anywhere else the flow has never seen a training row and extrapolates confidently rather "
                f"than returning the prior.")
        if not accept.other_observation:
            raise ValueError(_msg + " Use the recorded observation, an amortized posterior, or pass "
                             "Accept(other_observation=True) to run anyway -- the inference will record it.")
        print(_msg + " Running anyway (accepted).", flush=True)
        warnings.warn(_msg, stacklevel=2)
        accepted = accept.used()
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    with store.create("inference", cfg, name=name, note=note) as w:
        sink = w.fig_sink(fig_sink)
        samples = post.sample((int(n_samples),), x=obs_stats)
```
Then the existing body follows, indented under the `with`, with `posterior.sample((1000,), x=obs_stats.to(device))` replaced by the line above, every `_emit(fig_sink, title, fig)` → `sink(title, fig)`, `_emit_overlay_figures(cfg, obs_data, x_dim, sim_stats, obs_stats, samples, show_truth, sink)`, and after the eye test:

```python
        file_manager.atomic_torch_save(samples.detach().cpu(), w.payload("samples.pt"))
        q = torch.quantile(samples.detach().cpu().to(torch.float64),
                           torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64), dim=0)
        out = {
            "ppc": {"mean_abs_z": _num(results["mean_abs_z"]), "max_abs_z": _num(results["max_abs_z"]),
                    "coverage_90": _num(results["coverage_90"]), "num_outside": int(results["num_outside"]),
                    "num_invalid": int(results["num_invalid"]),
                    "invalid_breakdown": results.get("invalid_breakdown"), "note": _note or ""},
            "posterior_summary": {k: {"q05": _num(q[0, i]), "median": _num(q[1, i]), "q95": _num(q[2, i])}
                                  for i, k in enumerate(keys)},
            "ground_truth": {k: float(v) for k, v in zip(keys, cfg.ground_truth)} if show_truth else None,
            "n_samples": int(n_samples), "accepted": accepted,
        }
        w.payload("results.json").write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
        w.parents = {"posterior": posterior.id, "observation": observation.id}
        w.fingerprints["x_obs"] = observation.digest
        w.config.update({"n_samples": int(n_samples)})
        w.body = {"results": out}
    return store.load_inference(w.id)
```
(`results` is the PPC dict from `analysis.posterior_predictive_check`; `invalid_breakdown` is ints or None.) `run`: `infer_and_visualize(cfg, lp_post, obs)` in both branches. `runners`: `inf = orchestrator.infer_and_visualize(cfg, posterior, obs, fig_sink=fig_sink); return obs, inf` in both runners, where `posterior` is now the `LoadedPosterior` the Infer tab passes (`post = self.session.posterior`, no `.posterior`). `infer_tab._on_observation(payload)`: `obs, inf = payload; self.session.observation = obs;` and log the inference id plus `inf.results["ppc"]["coverage_90"]`.

- [ ] **Step 4: The suites**

- `tests/test_conditioning_repair.py:900-948` (`test_a_truncated_posterior_warns_on_a_foreign_observation_at_inference`): delete; its successor is above.
- `tests/test_user_sbi.py`: `orchestrator.infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True, fig_sink=sink)` → `orchestrator.infer_and_visualize(cfg, lp_post, obs, fig_sink=sink)`; the passive experimental block becomes
  ```python
        rec_path = tmp_path / "passive.npy"   # (add tmp_path to the test's signature; pytest injects it)
        np.save(rec_path, x_dim[0].numpy())
        obs_e = orchestrator.build_experiment_observation(cfg, RecordingSet(spont=str(rec_path), T_obs_s=1.0), fig_sink=sink)
        assert obs_e.width == SUMMARY_WIDTH + 1
        orchestrator.infer_and_visualize(cfg, lp_post, obs_e, fig_sink=sink)
  ```
  (`test_chi_mode_full_sbi_pipeline`'s experimental block is the chi twin: `RecordingSet(spont=..., forced=tuple((path, freq) ...), T_obs_s=..., F0_si=...)`.)
- `tests/test_nav_and_gating.py:818-832`: the stubbed `infer_and_visualize` returns a `SimpleNamespace(id="i", name="", results={"ppc": {"coverage_90": 0.9}})`.

- [ ] **Step 5: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py -k inference -q`; `pytest tests/test_user_sbi.py -k "no_forcing" -q`; `pytest -m "not slow" -q` — green.

```bash
git add core tests
git commit -F - <<'MSG'
inference: infer_and_visualize takes the Loaded wrappers and writes results (piece 1, Task 10). The observation reinstalls its context (so an inference reruns in a fresh session), show_truth is the observation's kind, a width/mode mismatch between posterior and observation is refused, and a NON-AMORTIZED posterior on another observation is refused unless Accept(other_observation=True), which is recorded. The inference artifact holds samples.pt, results.json (the PPC numbers that were computed and discarded, per-parameter quantiles in physical units, the ground truth for a simulated cell) and every figure, with the posterior and observation as parents.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 11: TSNPE over the wrappers; the observation picker reads the store

**Files:**
- Modify: `core/orchestrator.py` (`build_truncation_region` L1198-1279; `build_posterior`'s `x_obs_digest` parameter), `core/gui/panels/inference/runners.py` (`_run_tsnpe_round`), `core/gui/panels/inference/tsnpe_tab.py`, `tests/test_nav_and_gating.py:266-271`, `tests/test_conditioning_repair.py` (the `build_truncation_region` tests ~L610-775), `tests/test_user_sbi.py` (the guardrail-7 test ~L2318-2524)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Produces: `build_truncation_region(posterior: LoadedPosterior, observation: LoadedObservation, *, n_directions=None, level=None, t_scale_idx=None) -> TruncationRegion`; `build_posterior(...)` loses `x_obs_digest` (the region and `observation` carry it); `runners._run_tsnpe_round(cfg, posterior: LoadedPosterior, prior: LoadedPrior, obs_ref: str, n_directions, level, num_runs, run_size_cap, *, fig_sink=None) -> LoadedPosterior`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_artifact_store.py`:

```python
def test_a_round_records_parent_posterior_and_observation_as_parents(tiny_run):
    from core import orchestrator
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="round_obs")
    region = orchestrator.build_truncation_region(r.posterior, obs, n_directions=1, level=0.99)
    assert region.x_obs_digest == obs.digest and region.prior_fingerprint == r.posterior.fingerprint
    child = orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=r.sink, num_runs=2, hidden_features=8,
                                         num_transforms=1, stop_after_epochs=1, truncation=region, observation=obs,
                                         parent_posterior=r.posterior, name="round1")
    m = child.manifest
    assert m.body["amortized"] is False and child.posterior.truncation is not None
    assert m.parents["parent_posterior"] == r.posterior.id and m.parents["observation"] == obs.id
    assert m.parents["prior"] == r.prior.id and m.body["truncation"]["x_obs_digest"] == obs.digest
    with pytest.raises(ValueError, match="NOT AMORTIZED"):
        r.store.load_posterior(r.cfg, child.id)
    assert r.store.get("posterior", child.id).body["training"]["tsnpe_acceptance"] is not None
    with pytest.raises(ValueError, match="deleted prior support|does not match the observation"):
        other = orchestrator.generate_observations(r.cfg, fig_sink=r.sink)   # new noise, new digest
        orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=r.sink, num_runs=2, hidden_features=8,
                                     num_transforms=1, stop_after_epochs=1, truncation=region, observation=other,
                                     parent_posterior=r.posterior)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_artifact_store.py -k round_records -v` — `AttributeError` (the old `build_truncation_region` wants an `obs_record` dict).

- [ ] **Step 3: Rewrite `build_truncation_region` and the TSNPE plumbing**

```python
def build_truncation_region(posterior: LoadedPosterior, observation: LoadedObservation, *,
                            n_directions: int = None, level: float = None, t_scale_idx: int | None = None):
    """The TSNPE region for the NEXT round, drawn from ``posterior`` around ``observation``.

    GUARDRAIL 1 is now structural: the observation is an artifact whose payload the loader hashed
    against its manifest, so the region can only ever be drawn around a recorded observation. The
    region records the observation's digest, the parent's basis (its V and a probe of its whole
    training transform, guardrail 7) and the parent's training-prior fingerprint, so the retrain
    reuses that V and refuses any other prior. ``t_scale_idx`` defaults to t_scale's position in the
    observation's recorded parameter order (a t_scale-loaded direction is never truncated, D4).
    """
    x_obs = observation.x_obs
    if observation_digest(x_obs) != observation.digest:
        raise ValueError(f"Observation '{observation.name or observation.id}' does not hash to its own digest; "
                         f"a region drawn from it would delete prior support on the strength of data nobody recorded.")
    if t_scale_idx is None:
        keys = list(observation.manifest.config.get("param_keys") or [])
        if "t_scale" not in keys:
            raise ValueError("build_truncation_region needs t_scale's latent index and the observation's recorded "
                             f"parameter order {keys} contains no t_scale.")
        t_scale_idx = keys.index("t_scale")
    T_parent = posterior.posterior.T
    V = rotation_of(T_parent)
    latent = posterior.latent
    # (the rotation_of_prior consistency check and the box/probe lines stay exactly as they were)
    ...
    return truncate.region_from_posterior(
        latent, x_obs.to(transform_device(T_parent)),
        n_directions=truncate.DEFAULT_N_DIRECTIONS if n_directions is None else int(n_directions),
        level=truncate.DEFAULT_HPD if level is None else float(level),
        V=V, probe=probe, x_obs_digest=observation.digest, t_scale_idx=int(t_scale_idx),
        prior_fingerprint=posterior.fingerprint)
```
In `build_posterior`: remove the `x_obs_digest` parameter; at the top of the body `x_obs_digest = observation.digest if observation is not None else None`; the existing check "`x_obs_digest != truncation.x_obs_digest` → refuse" keeps its text (it is what the test's second `pytest.raises` matches — make the message say `does not match the observation the truncation region was drawn around`).

`runners._run_tsnpe_round`:
```python
def _run_tsnpe_round(cfg, posterior, prior, obs_ref, n_directions, level, num_runs, run_size_cap, *, fig_sink=None):
    """One TSNPE round: region from the posterior around a RECORDED observation -> prior RESTRICTED to
    it -> simulate -> retrain. The proposal is the truncated prior and never the posterior (the rule
    lives in core/SBI/truncate.py and is pinned by tests/test_conditioning_repair.py)."""
    from core.artifacts import resolve_store
    obs = resolve_store(None).load_observation(cfg, obs_ref)
    region = orchestrator.build_truncation_region(posterior, obs, n_directions=n_directions, level=level,
                                                  t_scale_idx=len(cfg.params_dict) + cfg.rescale_idx["t_scale"])
    print(f"[tsnpe] region from observation {obs.name or obs.id}: {region!r}", flush=True)
    return orchestrator.build_posterior(cfg, prior, None, True, fig_sink=fig_sink, num_runs=num_runs,
                                        run_size_cap=run_size_cap, truncation=region, observation=obs,
                                        parent_posterior=posterior)
```
`tsnpe_tab.py`: drop `from core.config import OBSERVATION_PATH`; `from ...widgets.artifact_picker import StorePicker`; L55 `self.obs_picker = StorePicker("observation")`; L106-109 `self.dispatch(_run_tsnpe_round, s.cfg, s.posterior, s.inf_prior, self.obs_picker.key(), n_dirs, level, max(1, n_runs), max(0, cap), provide_fig_sink=True, on_result=self._on_round)`; the warning label's "marked as such in its sidecar" → "recorded as such in its manifest".

- [ ] **Step 4: The suites**

- `tests/test_nav_and_gating.py:266-271`: the key stubs become ids (`"20260910T120000"`); the AST pin (L283) keeps matching (`build_truncation_region`, `truncation=region`).
- `tests/test_conditioning_repair.py` (`build_truncation_region` tests, ~L610-775) and `tests/test_user_sbi.py`'s guardrail-7 test: an observation is a `SimpleNamespace(x_obs=x, digest=orchestrator.observation_digest(x), id="o", name="", manifest=SimpleNamespace(config={"param_keys": keys}))`; a parent posterior is `SimpleNamespace(posterior=TransformedPosterior(...), latent=latent, fingerprint=None, id="p", name="")`; `x_obs_digest=` kwargs on `build_posterior` become `observation=<the stub>`.

- [ ] **Step 5: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py -k round -q`; `pytest tests/test_conditioning_repair.py tests/test_nav_and_gating.py -q`; `pytest tests/test_user_sbi.py -k "tsnpe" -q`; `pytest -m "not slow" -q` — green.

```bash
git add core tests
git commit -F - <<'MSG'
tsnpe: the region and the round run over the Loaded wrappers; the observation picker reads the store (piece 1, Task 11). build_truncation_region takes the parent LoadedPosterior and a LoadedObservation -- guardrail 1 is structural now, the observation is an artifact whose payload hashed against its manifest -- and records the digest, the parent's basis and its prior fingerprint; the round's posterior names the prior, the simulation cache, the parent posterior and the observation as parents and is non-amortized in its manifest. build_posterior's x_obs_digest parameter is gone (the observation carries it).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 12: Remove the generated-path constants; re-point FDT, CrossVal and Reduction; the source scan

**Files:**
- Modify: `core/config.py` (delete the block Task 1 left: `_ROOT`, `PRIOR_PATH`, `POSTERIOR_PATH`, `PLOT_PATH`, `CHECKPOINT_PATH`, `OBSERVATION_PATH`), `core/cli.py` (the two artifact pickers, ~L130-170, and the `from .config import` line L17), `core/FDT/fdt_pipeline.py:17,64-131`, `core/FDT/cross_validation.py:38`, `core/FDT/cross_validation_plots.py:19`, `core/Reduction/plots.py:20`, `core/Reduction/sweep.py:184`, `core/gui/panels/fdt_panel.py:13,133`, `core/gui/panels/crossval_panel.py:14,157`, `core/gui/panels/reduction_panel.py:13,75`, `core/gui/app.py:36-50`, `.gitignore`
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Consumes: `config.artifacts_root()`, `default_store`, `set_default_store`, `ArtifactStore`.
- Produces: nothing new; `config.PRIOR_PATH` etc. no longer exist (`grep -rn "PRIOR_PATH\|POSTERIOR_PATH\|PLOT_PATH\|CHECKPOINT_PATH\|OBSERVATION_PATH" core scripts tests` must list only `scripts/`, which Task 13 handles).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_artifact_store.py`:

```python
def test_no_literal_resource_paths_outside_config():
    """Every path under Resources/ or Artifacts/ is built from core.config. A literal anywhere else is a
    store that moves with the working directory -- what Resources/ used to do. Docstrings and
    comments are not Path() calls or `/` operands, so they do not trip this."""
    import ast
    root = Path(__file__).resolve().parents[1]
    roots = ("Resources/", "Resources\\", "Artifacts/", "Artifacts\\", "Resources", "Artifacts")

    def _lit(node):
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(roots)

    offenders = []
    for py in sorted((root / "core").rglob("*.py")):
        if py == root / "core" / "config.py":
            continue
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            hit = False
            if isinstance(node, ast.Call):
                fn = node.func
                fname = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                hit = fname in ("Path", "join") and node.args and _lit(node.args[0])
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                hit = _lit(node.left) or _lit(node.right)
            if hit:
                offenders.append(f"{py.relative_to(root)}:{node.lineno}")
    assert not offenders, "literal store paths outside config.py:\n" + "\n".join(offenders)
    for name in ("PRIOR_PATH", "POSTERIOR_PATH", "PLOT_PATH", "CHECKPOINT_PATH", "OBSERVATION_PATH"):
        assert not hasattr(config, name), f"config.{name} still exists"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_artifact_store.py -k literal -v` — lists the four `core/FDT` and `core/Reduction` sites and the five constants.

- [ ] **Step 3: Delete the constants and re-point the consumers**

`core/config.py`: delete from `# Generated-kind constants.` through `OBSERVATION_PATH = ...`.

| file | today | becomes |
|---|---|---|
| `core/FDT/fdt_pipeline.py` | `from core.config import ..., PLOT_PATH` (L17); `PLOT_PATH.mkdir(...)` (L64); `PLOT_PATH / ...` (L74, L95, L129-131) | `from core import config`; a module helper `def _out_dir(): d = config.artifacts_root() / "fdt"; d.mkdir(parents=True, exist_ok=True); return d` and `_out_dir() / ...` at every site |
| `core/FDT/cross_validation.py:38` | `_OUT_DIR = Path("Resources/CrossValidation")` | `def _out_dir(): return config.artifacts_root() / "crossval"` (+ `from core import config`), callers use `_out_dir()` |
| `core/FDT/cross_validation_plots.py:19` | `_PLOT_DIR = Path("Resources/Plots")` | `def _plot_dir(): return config.artifacts_root() / "crossval"` |
| `core/Reduction/plots.py:20` | `_PLOT_DIR = Path("Resources/Plots")` | `def _plot_dir(): return config.artifacts_root() / "reduction"` |
| `core/Reduction/sweep.py:184` | `out_dir = Path("Resources/ReductionMap")` | `out_dir = config.artifacts_root() / "reduction"` |
| `core/gui/panels/fdt_panel.py:13,133` | `from core.config import PLOT_PATH`; `watch_dir=PLOT_PATH` | `from core import config`; `watch_dir=config.artifacts_root() / "fdt"` |
| `core/gui/panels/crossval_panel.py:14,157` | same | `watch_dir=config.artifacts_root() / "crossval"` |
| `core/gui/panels/reduction_panel.py:13,75` | same | `watch_dir=config.artifacts_root() / "reduction"` |

(Reduction is out of scope; only its output directory moves so nothing writes into the inputs tree. Piece 5 wraps FDT/CrossVal in the store.) `plot_watcher.NewPngWatcher(directory)` must tolerate a directory that does not exist yet — check its constructor; if it does not, `mkdir(parents=True, exist_ok=True)` in each panel before dispatch.

`core/cli.py`: `select_or_build_prior()` and `select_or_train_posterior()` enumerate the store instead of a directory. Replace each function's listing with a call to one helper:

```python
def _pick_artifact(kind: str, what: str, allow_new: bool) -> tuple:
    """Interim until piece 2 retires this CLI: the store's complete artifacts of one kind, numbered;
    returns (id_or_None, build_new)."""
    from core.artifacts import default_store
    rows = [s for s in default_store().list(kind) if s.complete]
    if allow_new:
        print(f"  0) build/train a new {what}")
    for i, s in enumerate(rows, 1):
        print(f"  {i}) {s.label}   [{s.id}, {s.created}]")
    if not rows and not allow_new:
        raise SystemExit(f"No {what} artifacts exist yet.")
    idx = _prompt_index(len(rows), allow_zero=allow_new)     # the existing prompt helper (section 7.5)
    return (None, True) if idx == 0 else (rows[idx - 1].id, False)
```
keeping each function's docstring and prompt text; drop `PRIOR_PATH`/`POSTERIOR_PATH` from the import at L17. (If `_prompt_index`'s signature differs, call it the way the two functions call it today.)

`core/gui/app.py` `build_app`, after `registry.load_user_models()`:
```python
    # The artifact store, resolved once: a root that cannot be created fails HERE, at launch, not at
    # the first write hours into a run.
    from core.artifacts import ArtifactStore, set_default_store
    root = config.artifacts_root()
    root.mkdir(parents=True, exist_ok=True)
    set_default_store(ArtifactStore(root))
```

`.gitignore`: add
```
# Generated artifacts (piece 1 of the 2026-09-10 hardening programme). The /Resources/* lines above
# stay until the clean-break runbook deletes those trees; then they go.
/Artifacts/
```

- [ ] **Step 4: Run the tests, the fast gate, and commit**

Run: `pytest tests/test_artifact_store.py -k literal -q`; `pytest -m "not slow" -q` — green (`test_fdt_user` pins no plot path; the panels are constructed by `test_nav_and_gating` and only read the root).

```bash
git add core .gitignore tests/test_artifact_store.py
git commit -F - <<'MSG'
paths: the generated-kind constants are gone; FDT, CrossVal and Reduction outputs live under the artifacts root (piece 1, Task 12). config keeps RESOURCES_ROOT (inputs) and artifacts_root(); PRIOR/POSTERIOR/PLOT/CHECKPOINT/OBSERVATION_PATH no longer exist. The four literal Path("Resources/...") sites and the three panels' watch directories re-point to <Artifacts>/fdt, crossval, reduction as plain directories (piece 5 wraps FDT/CrossVal in the store). The CLI's pickers list the store. build_app creates the root and installs the default store at launch. A source scan pins that no module under core/ builds a literal Resources or Artifacts path outside config.py.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 13: The interim script re-point and the full gate

**Files:**
- Modify: `scripts/_common.py` (`load_posterior` L209-231, `require_mode` L233-260, the `from core.config import` line), `scripts/smoke_train.py` (L195-203 and every stage call)
- Test: the full suite

**Interfaces:**
- Produces: `_common.load_posterior(name, cfg, *, check_mode=True) -> (latent, T_eval, TransformedPosterior, Manifest)`; `smoke_train.py` honours `CKPT_DIR` as the ROOT of the store the run uses and `PRIOR` as a prior artifact name or id in it.

- [ ] **Step 1: Re-point `scripts/_common.py`**

```python
def load_posterior(name: str, cfg: SimConfig, *, check_mode: bool = True):
    """A posterior artifact (name or id) from the default store: ``(latent, T_eval, posterior, manifest)``.
    Interim until piece 2 folds the scripts into the command-line tool: the store's loader performs
    every check ``check_mode`` used to (and refuses rather than warns), so the flag is kept only for
    the call sites' sake."""
    from core.artifacts import Accept, default_store
    lp = default_store().load_posterior(cfg, name, accept=Accept(truncated=True))
    if lp.posterior.truncation is not None:
        print(f"[mode] {name}: NON-AMORTIZED (TSNPE), valid near observation {lp.posterior.x_obs_digest}", flush=True)
    return lp.latent, lp.posterior.T, lp.posterior, lp.manifest
```
`require_mode(cfg, posterior_latent, sidecar=None, name=...)`: accept a `Manifest` where it accepted a sidecar dict —
```python
    if hasattr(sidecar, "body"):                       # a manifest, since piece 1
        c = sidecar.body["conditioning"]
        sidecar = {"mode": sidecar.body["mode"], "forcing_dim": c["forcing_dim"], "chi_k_pad": c["chi_k_pad"]}
```
before the `posterior_mode(...)` call. Drop `POSTERIOR_PATH`, `read_sidecar`, `load_eval_bijection`, `reconcile_loaded_rotation` from `_common.py`'s imports.

- [ ] **Step 2: Re-point `scripts/smoke_train.py`**

Replace L195-203 (the `config.CHECKPOINT_PATH` block) with:
```python
    from core.artifacts import ArtifactStore, set_default_store
    _root = pathlib.Path(CKPT_DIR) if CKPT_DIR else pathlib.Path(tempfile.mkdtemp(prefix="prism_smoke_"))
    _root.mkdir(parents=True, exist_ok=True)
    set_default_store(ArtifactStore(_root))
    print(f"[smoke] artifacts (prior, cache, posterior, observation, calibration, inference) go to {_root}"
          f"{' (CKPT_DIR; REUSED across runs, so a resume is possible)' if CKPT_DIR else ' (a temp dir, never reused)'}",
          flush=True)
    if CHECKPOINT:
        orchestrator.TRAINING_CHECKPOINT_EVERY = max(1, NUM_RUNS // 2)
        if CKPT_DIR and not PRIOR:
            print("[smoke] ⚠ CKPT_DIR is set but PRIOR is not ...", flush=True)   # the existing warning text
    else:
        orchestrator.TRAINING_CHECKPOINT_EVERY = 0
```
(`import tempfile` at the top.) Stage calls:
```python
    prior = _stage("prior", lambda: orchestrator.build_prior(
        cfg, PRIOR, PRIOR is None, name="_smoke_prior" if (SAVE and PRIOR is None) else "", fig_sink=_sink))
    ...
    def _posterior():
        cfg.hw.batch_size = RUN_SIZE
        return orchestrator.build_posterior(cfg, prior, None, True, name="_smoke_posterior" if SAVE else "", fig_sink=_sink)
    post = _stage("posterior", _posterior)
    ...
    _stage("validate", lambda: orchestrator.validate_calibration(cfg, post, prior, fig_sink=_sink))   # + the existing n_cal/cal_n_scales knobs
    def _infer():
        obs = orchestrator.generate_observations(cfg, fig_sink=_sink)
        return orchestrator.infer_and_visualize(cfg, post, obs, fig_sink=_sink)
    _stage("infer", _infer)
```
(read the rest of the script for the exact existing `validate`/`infer` stage bodies and keep their knobs; `inferred_prior, force_prior = prior` and `posterior, _diag = post` unpacks go.) The docstring's `PRIOR` line becomes "name or id of a prior artifact in the store this run uses (CKPT_DIR)"; the resume drill's second command needs `PRIOR=_smoke_prior` with `SAVE=1` on the first run.

Scripts that still import a removed constant break until piece 2 — `grep -ln "PRIOR_PATH\|POSTERIOR_PATH\|PLOT_PATH\|CHECKPOINT_PATH\|OBSERVATION_PATH\|read_sidecar\|load_eval_bijection" scripts/*.py` lists them; add one line at the top of each one's docstring: `⚠ BROKEN since piece 1 (2026-09-10): reads Resources/Posteriors; folded into the command-line tool in piece 2.`

- [ ] **Step 3: The full gate**

Run: `pytest -q` (the slow chi pipeline test included; ~1 h). Expected: green. If `test_chi_mode_full_sbi_pipeline` fails on its experimental block, the fix is in Task 10's chi twin of the passive block (`RecordingSet(spont, forced=((path, freq_Hz), ...), T_obs_s, F0_si)`), not in the stage.

- [ ] **Step 4: Commit**

```bash
git add scripts tests
git commit -F - <<'MSG'
scripts: _common.load_posterior over the store; smoke_train on the store (piece 1, Task 13). load_posterior returns (latent, T_eval, posterior, manifest) through the store's loader; require_mode reads a manifest. smoke_train uses CKPT_DIR as the root of the store the whole run writes into (temp otherwise), so the resume drill and SAVE=1 exercise the real writers. The scripts that still read Resources/Posteriors are marked broken until piece 2 folds them into the command-line tool. Full gate: pytest green including the chi pipeline test.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
MSG
```

### Task 14: The clean-break runbook (user-executed, destructive — listed, not automated)

No code. Run in this order, from the repo root, with nothing else running:

1. Untrack the generated files git still holds (20 today, verified) and the run log:
   ```bash
   git rm --cached -q $(git ls-files -i -c --exclude-standard) sbc_run.log && git commit -m "Clean break: untrack generated files and sbc_run.log"
   ```
2. Delete the generated trees (`Resources/Checkpoints` is 1291 files, including the quarantined directory):
   ```bash
   rm -rf Resources/Priors Resources/Posteriors Resources/Checkpoints Resources/Observations Resources/Plots Resources/CrossValidation Resources/ReductionMap sbc_run.log
   ```
   `archive/` is untouched. `Resources/{Bounds,Cells,Units,Models}` stay.
3. Retire the two scripts whose inputs no longer exist:
   ```bash
   git mv scripts/tsnpe_round1_forensics.py scripts/migrate_checkpoint_flags.py archive/scripts/ && git commit -m "Clean break: archive the forensics and checkpoint-migration scripts"
   ```
4. `.gitignore`: delete the seven `/Resources/*` lines and their comment block (Task 12 added `/Artifacts/`); commit.
5. Record it in `docs/STATE.md` (the artifacts-on-disk section becomes "none; Artifacts/ starts empty") and note the date.

## Verification

- After every task: `pytest -m "not slow" -q` green; `tests/test_artifact_store.py` green from Task 1 on.
- Task 13: `pytest -q` green (the slow test included).
- **Stage contract, end to end** (the `tiny_run` fixture plus Tasks 8-11's tests): after Task 11, `pytest tests/test_artifact_store.py -q` leaves, in the fixture's store, exactly one prior (`tiny_prior`), one posterior (`tiny_post`), the observations, calibrations, inferences and the round's posterior written by the tests, every manifest valid, every `parents` id resolving, every payload hashing to its manifest — assert it once at the end of the suite:
  ```python
  def test_every_artifact_the_suite_wrote_is_complete_and_its_parents_resolve(tiny_run):
      s = tiny_run.store
      for kind in st.KIND_DIRS:
          for row in s.list(kind):
              assert row.complete, (kind, row.id, row.reason)
              m = s.get(kind, row.id)
              for pkind, pid in m.parents.items():
                  k = "posterior" if pkind == "parent_posterior" else pkind
                  assert s.get(k, pid).id == pid, (kind, row.id, pkind, pid)
              for f, sha in m.payloads.items():
                  assert prov.sha256_file(row.path / f) == sha, (kind, row.id, f)
  ```
- **GPU gate** (after Task 13): `CHI=1 TOBS_S=4.5 BOUNDS=Resources/Bounds/nadrowski/master.txt CELL=Resources/Cells/nadrowski/master_spont.txt CHECKPOINT=1 SAVE=1 CKPT_DIR=<scratch>/smoke python scripts/smoke_train.py`, then the same command with `PRIOR=_smoke_prior NUM_RUNS=2 STAGES=prior,posterior`: run 2 prints `Reusing the Fisher rotation stored with the training checkpoint` and `<scratch>/smoke/simulations/<digest>/manifest.json` reads `"complete": true`. Also `CHI=0 ... CELL=Resources/Cells/nadrowski/master_weak.txt`. Compare stage times with piece 0's baseline in `docs/STATE.md`.
- **Manual GUI check on a display** (`run.bat`): (1) `Artifacts/` is created at launch; (2) Config → Prior, "(from scratch)": `priors/_unnamed__<id>/` appears with `manifest.json`, `prior.pt`, `figures/prior.png`; Save with a name renames the directory and the picker shows the name with the id in the tooltip; (3) Posterior "(from scratch)" at 2 batches: the checkpoint line names `simulations/<digest>`; after training `posteriors/_unnamed__<id>/` has `posterior.pt`, `loss.npz`, `figures/training_loss.png`, manifest `amortized: true` with parents `prior` and `simulation`; (4) load that posterior from the picker: no dialog; a posterior built under another bounds file: a refusal dialog naming the field; (5) Validate: `calibrations/` with three PNGs and `results.json`; (6) Infer on a simulated cell: `observations/` then `inferences/`, parents `posterior` and `observation`, and the log line naming both; (7) TSNPE tab lists the observation; a round produces a posterior with `amortized: false` and four parents; loading it logs the NON-AMORTIZED line; Infer on another cell with it is refused by a dialog naming `Accept(other_observation=True)`; (8) Cancel during training: no `posteriors/` directory appears and `simulations/<digest>/manifest.json` reads `"complete": false`; (9) hand-truncate a `manifest.json`: the picker omits it. Record the result in `docs/STATE.md`.

## Notes for the executor

- `tests/_fixtures.py` is a plain module (no `conftest` magic): import from it explicitly.
- The `tiny_run` fixture builds a real posterior once per module; keep every test that needs a sampling posterior in `tests/test_artifact_store.py` so it is built once.
- The Reduction package is out of scope; only its output directory moves (Task 12).
- Interim breakages, all closed by piece 2: `scripts/{sbc_characterize,retrain_convergence,identifiability_offgt,posterior_identifiability,channel_ablation,degeneracy_map,chi_f0_sweep,feature_candidate_test,chi_mask_audit}.py` (marked in Task 13); the prompt CLI's `run()` shows no figures (the writer's sink saves and closes them).
- Deviation log (each deliberate): manifest bodies are validated dicts; the posterior `transform` block carries `param_keys`; the simulation kind's `id` is its digest and `ID_RE` allows both forms; `.gitignore` keeps the `/Resources/*` lines until the runbook; `env_info` records `"absent"` for a missing optional package.
