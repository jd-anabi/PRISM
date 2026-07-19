"""Persistence for user-defined models.

The source of truth is ``Resources/Models/<NAME>.json`` (schema below). Saving ALSO emits the
decoupled ``Bounds/<name_lower>/default.txt`` + ``Cells/<name_lower>/default.txt`` +
``Units/<name_lower>/units.txt`` triple so the whole existing config path (``cli.make_sim_config``,
``cli._parse_cell``, ``simulate_runner.build_stream_config``) consumes a user model exactly like a
built-in -- they resolve those files purely by folder name.

JSON schema (schema_version 2):
    {
      "schema_version": 2,
      "name": "MYMODEL",                       # uppercase; also the JSON filename stem
      "variables": [                            # declared order; variable 0 is the observable
        {"name": "x", "drift": "mu*x - x^3", "D": "d0", "init": 0.1,
         # D: params/numbers = additive noise; referencing state vars = multiplicative (state-dependent).
         "forcing": null | {"kind": "sin|step|triangular|exponential",
                            "params": {"amp": 1.0, ...},   # names WITHOUT the _<var> suffix
                            "sign": 1 | -1}},              # exponential grow/decay only
        ...
      ],
      "params": {"mu": {"value": 1.0, "lo": 0.0, "hi": 2.0},   # value + its SBI inference box (lo < value < hi)
                 "d0": {"value": 0.05, "lo": 0.01, "hi": 0.1}},
      "rescale": {"x_scale": 10.0, "t_scale": 0.01}
    }

schema_version 1 stored each param as a bare scalar value and let ``_nd_bounds`` auto-generate a
placeholder box; v2 lets the builder set per-parameter (lo, hi) for a tighter SBI prior. ``_normalize_to_v2``
migrates a v1 doc IN MEMORY (bare scalars -> the same placeholder box) on every load and save, so existing
``Resources/Models/*.json`` keep working after the bump. The emitted ND Bounds now carry the user's (lo, hi);
the x_scale/t_scale bounds stay the auto multiplicative range and must be strictly positive --
``SimConfig.t_nd_max`` divides by the t_scale LOWER bound.
"""
import json
import math
import re
import shutil
from pathlib import Path

from core import config
from core.forcing import FORCE_KINDS, FORCING_PARAM_NAMES
from core.Models.user_model import parse_user_model

SCHEMA_VERSION = 2
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,23}$")
_BUILTIN_NAMES = frozenset({"BP", "NADROWSKI", "HOPF"})
# Windows reserved device names: creating Bounds/<name>/... would half-fail (mkdir 'nul' silently
# does nothing on Windows), leaving a registered-but-broken model.
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL"}
                              | {f"COM{i}" for i in range(1, 10)}
                              | {f"LPT{i}" for i in range(1, 10)})

# Streaming feasibility ceiling for t_scale (seconds per ND unit): the emitted bounds DOUBLE t_scale,
# and SimConfig.steady_idx = TRANSIENT_ND_UNITS / (DT_EXP_S / (2*t_scale)) must stay < N_ND_MAX or
# every Simulate run dies on the config.py steady_idx assert after a clean save.
T_SCALE_MAX_S = config.N_ND_MAX * config.DT_EXP_S / (2.0 * config.TRANSIENT_ND_UNITS)


def validate_name(name: str) -> str:
    """Canonicalize + validate a model name; returns the UPPERCASE canonical form or raises ValueError."""
    name = str(name or "").strip()
    if not NAME_RE.match(name):
        raise ValueError(
            "Model name must start with a letter and use only letters/digits/_ (max 24 chars).")
    canonical = name.upper()
    if canonical in _BUILTIN_NAMES:
        raise ValueError(f"'{canonical}' is a built-in model name.")
    if canonical in _WINDOWS_RESERVED:
        raise ValueError(f"'{canonical}' is a reserved Windows device name.")
    if "nadrowski" in canonical.lower():
        raise ValueError("Model names containing 'nadrowski' are reserved.")
    return canonical


def _finite(value, label: str) -> float:
    """float() with a finiteness gate: inf/nan values would be emitted as tokens the bounds/cell
    parsers cannot read, persisting a registered-but-unusable model."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{label} must be a finite number.")
    return v


def list_user_models() -> list:
    """Saved user-model JSON paths, sorted by name. Empty if the folder doesn't exist yet."""
    if not config.MODELS_PATH.is_dir():
        return []
    return sorted(config.MODELS_PATH.glob("*.json"))


def _fmt(v: float) -> str:
    return repr(float(v))


def _nd_bounds(v: float) -> tuple:
    pad = max(abs(v), 1.0)
    return (v - pad, v + pad)


def _param_entry(value, lo=None, hi=None) -> dict:
    """One ND parameter's persisted form: its ground-truth value + the SBI inference box (lo, hi).
    lo/hi None -> the historical placeholder box ``_nd_bounds(value)``, so an auto-bounds parameter is
    byte-identical to the old scalar-schema emit."""
    value = float(value)
    if lo is None or hi is None:
        lo, hi = _nd_bounds(value)
    return {"value": value, "lo": float(lo), "hi": float(hi)}


def _migrate_params_v1(params: dict) -> dict:
    """schema_version 1 stored params as name -> scalar; v2 stores name -> {value, lo, hi}. Old scalars
    get the placeholder box ``_nd_bounds`` used to auto-generate, so a migrated model keeps its exact
    previous behaviour."""
    return {name: _param_entry(v) for name, v in params.items()}


def _normalize_to_v2(doc: dict) -> dict:
    """Return ``doc`` as schema_version 2 WITHOUT mutating the input. A v1 doc's scalar params are
    migrated to the {value, lo, hi} form; anything else is returned unchanged (a malformed params block
    is left for ``_check_schema`` to reject). Called before ``_check_schema`` on every load and save so
    old Resources/Models/*.json survive the schema bump."""
    if doc.get("schema_version") != 1:
        return doc
    params = doc.get("params")
    migrated = _migrate_params_v1(params) if isinstance(params, dict) else params
    return {**doc, "schema_version": 2, "params": migrated}


def _check_schema(doc: dict) -> None:
    """Strict structural validation; raises ValueError with a user-facing message on any deviation."""
    if not isinstance(doc, dict):
        raise ValueError("Model file is not a JSON object.")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version {doc.get('schema_version')!r} "
                         f"(this app reads version {SCHEMA_VERSION}).")
    validate_name(doc.get("name"))
    variables = doc.get("variables")
    if not isinstance(variables, list) or not variables:
        raise ValueError("'variables' must be a non-empty list.")
    for v in variables:
        if not isinstance(v, dict) or not v.get("name") or "drift" not in v or "D" not in v or "init" not in v:
            raise ValueError("Each variable needs 'name', 'drift', 'D', 'init' and 'forcing' entries.")
        _finite(v["init"], f"Variable '{v.get('name')}': init")
        forcing = v.get("forcing")
        if forcing is not None:
            if not isinstance(forcing, dict) or forcing.get("kind") not in FORCE_KINDS:
                raise ValueError(f"Variable '{v['name']}': forcing kind must be one of {FORCE_KINDS}.")
            params = forcing.get("params")
            needed = set(FORCING_PARAM_NAMES[forcing["kind"]])
            if not isinstance(params, dict) or set(params) != needed:
                raise ValueError(
                    f"Variable '{v['name']}': {forcing['kind']} forcing needs exactly {sorted(needed)}.")
            for pname, pv in params.items():
                _finite(pv, f"Variable '{v['name']}': forcing {pname}")
            if forcing["kind"] == "exponential" and forcing.get("sign") not in (1, -1):
                raise ValueError(f"Variable '{v['name']}': exponential forcing needs sign 1 or -1.")
    params = doc.get("params")
    if not isinstance(params, dict):
        raise ValueError("'params' must be an object of name -> {value, lo, hi}.")
    for pname, entry in params.items():
        if not isinstance(entry, dict) or set(entry) != {"value", "lo", "hi"}:
            raise ValueError(f"Parameter '{pname}' must be an object with value, lo and hi.")
        value = _finite(entry["value"], f"Parameter '{pname}'")     # _finite first: keeps the NaN message
        lo = _finite(entry["lo"], f"Parameter '{pname}' lo")
        hi = _finite(entry["hi"], f"Parameter '{pname}' hi")
        if not lo < hi:
            raise ValueError(f"Parameter '{pname}': lo must be < hi (got [{lo}, {hi}]).")
        if not lo <= value <= hi:
            raise ValueError(f"Parameter '{pname}': value {value} is outside its bounds [{lo}, {hi}].")
    rescale = doc.get("rescale")
    if not isinstance(rescale, dict) or set(rescale) != {"x_scale", "t_scale"}:
        raise ValueError("'rescale' must contain exactly x_scale and t_scale.")
    for key in ("x_scale", "t_scale"):
        if _finite(rescale[key], key) <= 0:
            raise ValueError(f"{key} must be > 0 (it scales the display axes).")
    if float(rescale["t_scale"]) >= T_SCALE_MAX_S:
        raise ValueError(
            f"t_scale must be below {T_SCALE_MAX_S:g} s: slower models exceed the transient budget "
            f"(N_ND_MAX={config.N_ND_MAX} fine steps at the {config.DT_EXP_S * 1e3:g} ms sample "
            "rate) and could never stream.")


def load_user_model(path) -> dict:
    """Load + validate one saved model file. The JSON filename stem must equal the model name."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc = _normalize_to_v2(doc)                                 # migrate v1 scalar params -> v2 objects
    _check_schema(doc)
    if path.stem.upper() != doc["name"].upper():
        raise ValueError(f"File name '{path.stem}' does not match model name '{doc['name']}'.")
    doc["name"] = doc["name"].upper()
    return doc


def save_user_model(doc: dict) -> Path:
    """Validate, compile-check, and persist a model: the JSON + the Bounds/Cells/Units triple.

    The parameter ORDER in the Bounds ND section is ``parse_user_model``'s discovery order -- the same
    order ``UserModel`` consumes constructor columns in (the load-bearing invariant). Every discovered
    parameter must have a value in ``doc['params']``.
    """
    doc = _normalize_to_v2(doc)                                 # accept v1 scalar-param docs unchanged
    _check_schema(doc)
    name = validate_name(doc["name"])
    doc = {**doc, "name": name}
    compiled = parse_user_model(doc["variables"])              # raises ModelParseError on bad exprs
    missing = [p for p in compiled.param_names if p not in doc["params"]]
    if missing:
        raise ValueError(f"Missing value(s) for parameter(s): {missing}.")

    name_lower = name.lower()
    x_scale = float(doc["rescale"]["x_scale"])
    t_scale = float(doc["rescale"]["t_scale"])

    # ── Bounds/<name_lower>/default.txt ──
    lines = ["# Non-dimensional Parameters"]
    for p in compiled.param_names:
        entry = doc["params"][p]
        lines.append(f"{p} in ({_fmt(entry['lo'])}, {_fmt(entry['hi'])})")   # user-set (or auto) SBI box
    lines.append("# Dimensional Parameters")
    for key, v in (("x_scale", x_scale), ("t_scale", t_scale)):
        lines.append(f"{key} in ({_fmt(v / 2)}, {_fmt(v * 2)})")   # strictly positive: t_nd_max divides by lo
    lines.append("# Forcing Parameters")
    for v in doc["variables"]:
        forcing = v.get("forcing")
        if forcing:
            for pname in FORCING_PARAM_NAMES[forcing["kind"]]:
                lo, hi = _nd_bounds(float(forcing["params"][pname]))
                lines.append(f"{pname}_{v['name']} in ({_fmt(lo)}, {_fmt(hi)})")
    bounds_text = "\n".join(lines) + "\n"

    # ── Cells/<name_lower>/default.txt ──
    lines = ["# Non-dimensional Initial Conditions"]
    for v in doc["variables"]:
        lines.append(f"{v['name']}_init = {_fmt(v['init'])}")
    lines.append("# Non-dimensional Parameters")
    for p in compiled.param_names:
        lines.append(f"{p} = {_fmt(doc['params'][p]['value'])}")
    lines.append("# Dimensional Parameters")
    lines.append(f"x_scale = {_fmt(x_scale)}")
    lines.append(f"t_scale = {_fmt(t_scale)}")
    lines.append("# Forcing Parameters")
    for v in doc["variables"]:
        forcing = v.get("forcing")
        if forcing:
            for pname in FORCING_PARAM_NAMES[forcing["kind"]]:
                lines.append(f"{pname}_{v['name']} = {_fmt(forcing['params'][pname])}")
    cells_text = "\n".join(lines) + "\n"

    # x_scale is interpreted as nm, t_scale as s (factor 1 from the experimental constants' seconds).
    units_text = "# Units\nnm s\n"

    # Write the triple FIRST and the JSON LAST: the registry loads models from the JSON alone, so the
    # JSON write is the commit point -- a failure mid-way leaves at worst orphan text files (fixed by
    # re-saving), never a registered model whose triple is missing or stale.
    for base, text in ((config.BOUNDS_PATH, bounds_text), (config.CELL_PATH, cells_text)):
        folder = base / name_lower
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "default.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
    units_folder = config.UNITS_PATH / name_lower
    units_folder.mkdir(parents=True, exist_ok=True)
    with open(units_folder / "units.txt", "w", encoding="utf-8") as fh:
        fh.write(units_text)
    config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    json_path = config.MODELS_PATH / f"{name}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return json_path


def delete_user_model(name: str) -> None:
    """Remove a user model's emitted Bounds/Cells/Units folders, then its JSON. Built-ins are refused.

    The JSON goes LAST (mirror of save_user_model's commit order): if a folder rmtree fails (e.g. a
    file held open on Windows), the JSON survives, the model stays registered, and a retry converges
    -- deleting the JSON first would orphan the folders with no UI path to remove them."""
    name = validate_name(name)                                  # also blocks the built-in names/folders
    for base in (config.BOUNDS_PATH, config.CELL_PATH, config.UNITS_PATH):
        folder = base / name.lower()
        if folder.is_dir():
            shutil.rmtree(folder)
    json_path = config.MODELS_PATH / f"{name}.json"
    if json_path.exists():
        json_path.unlink()
