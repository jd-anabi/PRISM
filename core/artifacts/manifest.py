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
    # Schema FIRST: an older or foreign manifest is "not a current artifact", whatever else it lacks.
    if d.get("schema") != SCHEMA:
        raise ManifestError(f"manifest schema {d.get('schema')!r} is not the current schema {SCHEMA}; "
                            f"not a current PRISM artifact")
    unknown = set(d) - set(HEADER_KEYS)
    if unknown:
        raise ManifestError(f"unknown manifest keys {sorted(unknown)}")
    missing = [k for k in HEADER_KEYS if k not in d]
    if missing:
        raise ManifestError(f"manifest is missing {missing}")
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
    if d["kind"] == "posterior" and bool(d["body"]["amortized"]) != (d["body"]["truncation"] is None):
        # DEFECT D3's signature, refused at the schema: the amortization flag and the region are two
        # views of one fact, and the load path gates on the flag while calibration and inference gate
        # on the region. A manifest where they disagree is a posterior that is truncated in one
        # reader's eyes and amortized in another's -- which is how a "0.000%" round passed every check.
        raise ManifestError(
            f"posterior body says amortized={d['body']['amortized']!r} beside "
            f"{'a truncation region' if d['body']['truncation'] is not None else 'no truncation region'}; "
            f"a truncated posterior carries its region and an amortized one carries none (defect D3)")
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
