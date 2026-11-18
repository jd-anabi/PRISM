"""
Prompt-free configuration builders, shared by PRISM's two front ends.

The interactive prompts this module was named for went with the retired prompt CLI. What is
left is pure: a cell file checked and its values put in, the units file found, the unit conversion factors,
and the ``make_*_config`` builders that the GUI (``core/gui``) and the command-line tool (``core/tool``) each
call with values they obtained their own way. The module keeps the name ``core.cli``.
"""
from collections import OrderedDict
from pathlib import Path

import pint

from core import config          # live-read config.CHI_MODE / CHI_N_FREQS (avoid import-time snapshot)
from .config import (
    SimConfig, detect_device,
    DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S,
    UNITS_PATH,
)
from .Helpers import file_manager
from .refusals import Refusal, missing_values_phrase, require_file


class UnitParseError(Refusal):
    """Raised when the units file, or a unit typed in its place, names a unit pint can't resolve.

    Previously the parsers printed an error and called exit(), which killed the whole process --
    fatal for a GUI. They now raise this instead; each front end catches it -- the command-line tool
    turns it into a refusal (exit 1), and the GUI surfaces it as an error dialog.
    """


# ── Per-model input files ────────────────────────────────────────────────────
def resolve_units_file(model: str) -> str:
    """
    Auto-resolve the per-model units file (no prompt): Resources/Units/<model>/units.txt.

    :raises Refusal: (field "units") if the units file for this model is missing.
    """
    path = UNITS_PATH / model.lower() / "units.txt"
    if not path.exists():
        raise Refusal(f"Missing units file for model '{model}': expected {path}", field="units")
    return str(path)


# ── Post-training inference ─────────────────────────────────────────────────
def validate_gt_file(cfg: SimConfig, cell_path: str) -> list:
    """Non-mutating dry run of ``load_and_validate_gt``: the problems that would make it raise.

    Lets a GUI validate a cell the moment it is PICKED, on the GUI thread, instead of discovering the
    failure inside a worker halfway through an inference run. Mirrors SimConfig._fill_checked's rules --
    MISSING parameters are fatal, extras are ignored, and only ND/rescale are range-checked (forcing is
    the known drive, whose range is deliberately not enforced).

    :return: human-readable problem strings; empty means the cell can be injected.
    """
    try:
        inits, param_vals, rescale_vals, forcing_vals = file_manager.parse_values_file(cell_path)
    except Exception as e:                            # noqa: BLE001 -- surfaced to the user as-is
        return [f"could not parse the cell file: {e}"]
    problems = []
    for label, vals, cfg_dict, check_bounds in (
            ("ND parameter", param_vals, cfg.params_dict, True),
            ("rescale parameter", rescale_vals, cfg.rescale_params, True),
            ("forcing parameter", forcing_vals, cfg.force_params_dict, False)):
        missing = sorted(set(cfg_dict) - set(vals))
        if missing:
            # the phrase takes its label as given; this loop's labels are singular because the
            # out-of-bounds line below reads "ND parameter k = ...", so the plural is spelled here
            problems.append(missing_values_phrase(f"{label}s", missing))
            continue
        if check_bounds:
            problems += [f"{label} {n} = {vals[n]:g} is outside its bounds ({lo:g}, {hi:g})"
                         for n, (_, (lo, hi)) in cfg_dict.items() if not (lo <= vals[n] <= hi)]
    if not inits:
        problems.append("the cell file declares no initial conditions")
    return problems


def load_and_validate_gt(cfg: SimConfig, cell_path: str) -> list:
    """
    Parse a cell file's VALUES + initial conditions and inject them into cfg, validating (via
    SimConfig.inject_ground_truth) that the ND/rescale values lie within the bounds file's bounds.

    :return: names of cell values the bounds file does not declare, which were therefore IGNORED
             (e.g. f_scale + the drive when a forced cell is paired with spontaneous bounds). Empty in
             the usual matched case; callers may surface it, and existing callers can ignore it.
    :raises Refusal: (field "cell") for a blank path or one that names no file, before the parser
             (a file that exists but does not parse keeps the parser's behaviour).
    """
    require_file("cell", cell_path, "cell")
    inits, param_vals, rescale_vals, forcing_vals = file_manager.parse_values_file(cell_path)
    cfg.sources["cell"] = str(cell_path)
    return cfg.inject_ground_truth(inits, param_vals, rescale_vals, forcing_vals)


# Display-only SI unit hints, indexed by forcing param name (the GUI's drive fields).
# DERIVED from config.FORCING_SI_UNITS, the authoritative conversion table, so the two cannot drift.
INFERENCE_PROMPT_UNITS = config.FORCING_DISPLAY_UNITS

# ── Units → conversion factors (standalone helper for the decoupled bounds/units path) ──────
def units_to_factors(units: tuple) -> tuple[list[float], float]:
    """
    From a set of unit strings compute (si_factors, s_to_cell). Called by ``make_sim_config``, and by the
    Config tab to refuse a typed unit token it cannot resolve before a run starts.
    """
    ureg = config.unit_registry()      # shared singleton: building one parses pint's full unit file
    try:
        si_factors = [ureg(unit).to_base_units().magnitude for unit in units]
    except pint.UndefinedUnitError as e:
        raise UnitParseError(f"{e}. Unrecognized unit in the units file.", field="units")
    time_unit = None
    for unit_str in units:
        try:
            if ureg.Quantity(1, unit_str).check("[time]"):
                time_unit = unit_str
                break
        except pint.UndefinedUnitError:
            continue
    if time_unit is None:
        raise Refusal("Could not detect a time unit from the units file. Include a time unit (e.g. ms).",
                      field="units")
    return si_factors, ureg.Quantity(1, "s").to(time_unit).magnitude


# ── Pure config cores (no prompts) — shared by `python -m core` and the GUI ────────────
def make_sim_config(model: str, labels: list[str], state_dep_drift: bool, bounds_file: str = None, *,
                    bounds_dicts=None, units_override=None,
                    chi_mode: bool | None = None, chi_n_freqs: int | None = None,
                    chi_f0: float | None = None, chi_freq_bounds: tuple | None = None,
                    chi_k_pad: int | None = None, chi_max_cycles: float | None = None,
                    reparam_rotate: bool | None = None,
                    hw: "DeviceConfig | None" = None) -> SimConfig:
    """
    Build a bounds-only SimConfig (no prompts) from a chosen model + bounds file. Ground-truth values,
    initial conditions, and T_obs are filled later (only for simulated inference). Shared by
    the command-line tool (core/tool) and the GUI's SBI config form.

    The chi(omega) knobs are explicit keyword args so a GUI can set them PER CONFIG; each falls back to
    the live ``config.CHI_*`` module value when None (the command-line tool passes only chi_mode and
    chi_n_freqs, and keeps this behaviour for the rest).
    They are stored ON the config, so a posterior trained from it is self-describing.

    ``units_override`` DECLARES what the numbers in the bounds/cell files mean; it never converts them.
    Pass a path to a units file, or an iterable of unit tokens (e.g. ``("nm", "ms", "pN", "kHz")``).
    None keeps the per-model default ``Resources/Units/<model>/units.txt``.

    ``bounds_dicts`` is the hand-entered alternative to ``bounds_file``: a
    ``(params, rescale, forcing)`` triple in ``parse_bounds_file``'s shape. Exactly one of the two must
    be given. Parameter ORDER within each dict is load-bearing (simulators bind columns positionally),
    so it is preserved verbatim.

    ``hw`` is the DeviceConfig to run on; None detects one. The GUI passes nothing and keeps
    detect_device(); the command-line tool builds it from --device. It is not a cosmetic setting --
    the device and dtype are part of the simulation identity (``SimulationIdentity.from_cfg`` records
    both).
    """
    if (bounds_file is None) == (bounds_dicts is None):
        raise Refusal("make_sim_config needs exactly one of bounds_file or bounds_dicts.")
    if bounds_dicts is not None:
        params_dict, rescale_params, force_params_dict = (OrderedDict(d) for d in bounds_dicts)
    else:
        # The file is checked by its input kind first, so both front ends refuse a missing one
        # with the field key; the parser's own FileNotFoundError is then reached only by a race.
        require_file("bounds", bounds_file, "bounds")
        params_dict, rescale_params, force_params_dict, _ = file_manager.parse_bounds_file(bounds_file)
    units_path = None
    if units_override is None:
        units_path = resolve_units_file(model)
        units_dict = file_manager.parse_units_file(units_path)
    elif isinstance(units_override, (str, Path)):
        units_path = str(units_override)
        units_dict = file_manager.parse_units_file(str(units_override))
    else:
        units_dict = tuple(str(u) for u in units_override)
    _, s_to_cell = units_to_factors(units_dict)

    # convert experimental constants from seconds to cell-file time units (training T-range)
    return SimConfig(
        model=model,
        labels=labels,
        state_dep_drift=state_dep_drift,
        inits_dict=OrderedDict(),               # filled from a cell file only if inferring on a simulation
        params_dict=params_dict,                # {name: (None, (lo,hi))} until a cell injects values
        rescale_params=rescale_params,
        force_params_dict=force_params_dict,
        units_dict=units_dict,
        # multi-frequency chi(omega) conditioning; explicit arg wins, else the live module value
        chi_mode=config.CHI_MODE if chi_mode is None else bool(chi_mode),
        chi_n_freqs=config.CHI_N_FREQS if chi_n_freqs is None else int(chi_n_freqs),
        chi_f0=config.CHI_F0 if chi_f0 is None else float(chi_f0),
        chi_freq_bounds=(config.CHI_FREQ_BOUNDS if chi_freq_bounds is None
                         else tuple(chi_freq_bounds)),
        chi_k_pad=config.CHI_K_PAD if chi_k_pad is None else int(chi_k_pad),
        chi_max_cycles=(config.CHI_MAX_CYCLES if chi_max_cycles is None else float(chi_max_cycles)),
        reparam_rotate=(config.REPARAM_ROTATE if reparam_rotate is None else bool(reparam_rotate)),
        dt_exp=DT_EXP_S * s_to_cell,
        t_min_exp=T_MIN_EXP_S * s_to_cell,
        t_max_exp=T_MAX_EXP_S * s_to_cell,
        T_obs=None,                             # observation duration is supplied later, at the
                                                 # inference step (the GUI's field, or --t-obs)
        hw=detect_device() if hw is None else hw,
        sources={k: str(v) for k, v in (("bounds", bounds_file), ("units", units_path)) if v},
    )

