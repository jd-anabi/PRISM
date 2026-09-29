"""How every flag of ``python -m core`` states its default: one table, keyed by subcommand and flag.

Each flag that takes a value is in exactly one of three classes.

- VALUE: the help ends with "(default X)", written by ``core.refusals.default_clause`` or
  ``default_text`` -- the same helpers that end a refusal line -- so a help page and a refusal never
  state one default two ways. X is, in this order: the parser's own default, when it is neither None
  nor ''; else the literal this table pins for that subcommand and flag; else the refusal registry's
  default for the flag's field key, found by reading ``core/tool/fields.py``'s FLAG in reverse.
- BEHAVIOUR: what happens without the flag is not a value but a rule ("the bounds file's parent
  folder, upper-cased"), and the help ends with "(default <that phrase>)".
- NONE: no default clause at all. A required flag, a flag that only names an input, and --name and
  --note, whose help says instead what an empty value means ("unnamed", "no note").

WHY A PINNED LITERAL OUTRANKS THE REGISTRY. The registry's default is the one a refusal line prints,
and a few of its keys are worded for another subcommand: the seed's reads "none: one is drawn and
recorded", which is what the FDT analyses and validate do, while the diagnostics seed with 0; and
the FDT resolution knobs' carry the sweep preset's clause, which reads wrongly on a single run. A literal
pinned here says what THIS subcommand does, and tests/test_tool.py checks each one against the
signature or constant that owns it.

Every value flag of every subcommand has an entry: a flag added without one fails that test, naming
the subcommand and the flag, rather than printing a help page with no default.

TORCH-FREE, like the parsers it describes: ``--help`` must never import ``core.config``, which
imports torch, so this module imports only ``core.refusals`` and ``core.tool.fields``, and both of
those import nothing but the standard library.
"""
from core.refusals import default_clause, default_text

from .fields import FLAG

VALUE, BEHAVIOUR, NONE = "value", "behaviour", "none"


def _value(*flags) -> dict:
    """Value flags whose default the parser or the refusal registry states truly."""
    return {f: (VALUE, None) for f in flags}


def _none(*flags) -> dict:
    """Flags that state no default: required ones, and those that only name an input."""
    return {f: (NONE, None) for f in flags}


# the groups the shared flag helpers add (core/tool/config_args.py and the two analyses' helpers)
_CONFIG = {"--bounds": (NONE, None),                       # add_config_flags
           "--model": (BEHAVIOUR, "the bounds file's parent folder, upper-cased"),
           "--chi": (VALUE, "--no-chi"), "--chi-k": (VALUE, None), "--device": (VALUE, None)}
_NAMES = {"--name": (NONE, None), "--note": (NONE, None)}
_TRAINING = _value("--num-runs", "--run-size", "--hidden-features", "--num-transforms",
                   "--learning-rate", "--stop-after-epochs", "--max-epochs", "--checkpoint-every")
_FISHER = _value("--fisher-m", "--fisher-dz", "--fisher-points")
_RESUME = {"--resume": (VALUE, None)}
_PROBE = {"--m": (VALUE, None), "--m-noise": (VALUE, None), "--rel": (VALUE, None),
          "--min-valid": (VALUE, "0.5"), "--seed": (VALUE, "0")}
_FDT_KNOBS = {"--store-root": (BEHAVIOUR, "the artifacts root"), "--freqs-per-batch": (VALUE, None),
              "--f0": (VALUE, None), "--seed": (VALUE, None)}
_COMPARE = {**_none("--record"), **_NAMES}

_PER_LEAF = {
    "prior": {**_CONFIG, **_NAMES,
              **_value("--num-iterations", "--sweep-batch", "--max-sets", "--walk-step",
                       "--stability-units", "--min-cluster-size", "--min-samples")},
    "train": {**_CONFIG, **_NAMES, **_none("--prior"), **_TRAINING, **_FISHER, **_RESUME},
    "validate": {**_CONFIG, **_NAMES, **_none("--posterior"),
                 **_value("--n-cal", "--cal-n-scales", "--posterior-samples"), "--seed": (VALUE, None)},
    "infer": {**_CONFIG, **_NAMES,
              **_none("--posterior", "--cell", "--spont", "--t-obs", "--forced", "--drive",
                      "--f0-si"),
              **_value("--n-samples")},
    "tsnpe": {**_CONFIG, **_NAMES, **_none("--posterior", "--observation"),
              **_value("--directions", "--level"), **_TRAINING, **_RESUME},
    "sbc": {**_CONFIG, **_NAMES, **_none("--posterior"), **_value("--repeats", "--cal-n-scales"),
            "--n-cal": (VALUE, "2000"), "--posterior-samples": (VALUE, "1000"),
            "--chi-k-fixed": (BEHAVIOUR, "pooled over the training mixture"),
            "--seed": (VALUE, "0")},
    "identifiability rotation": {**_CONFIG, **_NAMES, **_none("--posterior"),
                                 **_value("--n-worst", "--top-n")},
    "identifiability laplace": {**_CONFIG, **_NAMES, **_none("--posterior", "--cell", "--t-obs"),
                                **_value("--n-points"), **_PROBE,
                                "--sd-identified": (VALUE, "0.3")},
    "identifiability jacobian": {**_CONFIG, **_NAMES, **_none("--cell", "--t-obs"), **_PROBE,
                                 "--zero-tol": (VALUE, "0.05"), "--noise-eps": (VALUE, "1e-06")},
    "ablation": {**_CONFIG, **_NAMES, **_none("--posterior"), **_value("--rows", "--n-sweep")},
    "smoke": {**_CONFIG, **_none("--cell", "--prior"), "--t-obs": (VALUE, "1.0"),
              **_value("--seed", "--stages", "--num-runs", "--run-size", "--n-cal", "--max-epochs"),
              "--store-root": (BEHAVIOUR, "a fresh temporary directory, left on disk"), **_RESUME},
    "fdt": {**_none("--cell"), "--model": (BEHAVIOUR, "the cell file's parent folder, upper-cased"),
            **_NAMES, **_FDT_KNOBS, "--n-freqs": (VALUE, "60"), "--ensemble-m": (VALUE, "256")},
    "crossval": {**_none("--cell", "--s-grid", "--t-grid"), **_value("--preset"), **_NAMES,
                 **_FDT_KNOBS, "--n-freqs": (VALUE, "the preset's: exploratory 30, production 60"),
                 "--ensemble-m": (VALUE, "256")},
    "compare cells": {**_COMPARE},
    "compare repeats": {**_COMPARE},
    "compare renormalise": {**_COMPARE, **_none("--prefactor")},
    "compare sweeps": {**_COMPARE, **_none("--at")},
    # the artifacts family names its kind and artifact positionally; only two modes take a value flag
    "artifacts list": {},
    "artifacts show": {},
    "artifacts note": _none("--note"),
    "artifacts rm": {},
    "artifacts sweep": {},
    "artifacts summary": _none("--out"),
}

LEAVES = tuple(_PER_LEAF)
DEFAULT_CLASS: dict[tuple[str, str], tuple[str, str | None]] = {
    (leaf, flag): entry for leaf, flags in _PER_LEAF.items() for flag, entry in flags.items()}


def default_for(leaf_path: str, action) -> str | None:
    """The whole default clause for one flag, leading space included (" (default X)"), or None for
    a flag of class NONE. ``leaf_path`` is the subcommand as typed after ``python -m core``
    ("sbc", "identifiability laplace"); ``action`` is the argparse action the flag built.

    A flag with no entry raises KeyError, and a value default with no single field key to read it
    from raises LookupError, each naming the subcommand and the flag."""
    flag = action.option_strings[0]
    try:
        cls, text = DEFAULT_CLASS[(leaf_path, flag)]
    except KeyError:
        raise KeyError(f"{leaf_path} {flag}: no entry in DEFAULT_CLASS") from None
    if cls == NONE:
        return None
    if cls == BEHAVIOUR:
        return default_text(text)
    if cls != VALUE:
        raise ValueError(f"{leaf_path} {flag}: unknown default class {cls!r}")
    if action.default not in (None, ""):
        return default_text(str(action.default))
    if text is not None:
        return default_text(text)
    keys = [key for key, f in FLAG.items() if f == flag]
    if len(keys) > 1:
        keys = [key for key in keys if key == action.dest.lower()]
    if len(keys) != 1 or not default_clause(keys[0]):
        raise LookupError(f"{leaf_path} {flag}: no single field key with a default states this "
                          f"flag's default (found {keys}); pin its value in DEFAULT_CLASS")
    return default_clause(keys[0])
