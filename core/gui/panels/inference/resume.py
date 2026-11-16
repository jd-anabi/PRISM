"""A kept training after a reopening: what the window reads to offer it, what stops its Resume before anything is read,
and its configuration, its draft and its call, made from its own record and never from a form.

The record holds the training's inputs (its bounds file, its units file and their hashes), its configuration's own
block (the model, the kind of observation, the slots and the lock-in limit, the rotation, the units typed, the device
and the precision) and every setting of its fit; the stage takes the fit's settings from the record itself, so the call
names the kept training and nothing it would compare (``resume_call``).
"""
from pathlib import Path, PurePath

from core import config

from ... import continuing
from ...run import history
from ...session import ConfigDraft

#: What stops a Resume before anything is read, each a sentence.
BOUNDS_GONE_TEXT = "The bounds file {} is no longer there."
BOUNDS_CHANGED_TEXT = "The bounds file {} has changed since this training began."
TYPED_FORCING_TEXT = ("This training was started from limits typed by hand with a forcing section, which its record "
                      "cannot rebuild.")
UNITS_GONE_TEXT = "The units file {} is no longer there."
UNITS_CHANGED_TEXT = "The units file {} has changed since this training began."
PRIOR_GONE_TEXT = "The prior it was trained with is no longer in Records."
NO_CARD_TEXT = "It was trained on the graphics card; this computer has none now."
NO_MODEL_TEXT = "Its model, {}, is no longer one PRISM can train."
#: What stops a kept narrowing round's Resume besides: the posterior it narrows, or its observation, gone.
PARENT_GONE_TEXT = "The posterior it narrows, {}, is no longer in Records."
OBSERVATION_GONE_TEXT = "The observation it was trained around is no longer in Records."
#: The same, for a record that names neither.
NO_PARENT_TEXT = "Its record names no posterior it narrows."
NO_OBSERVATION_TEXT = "Its record names no observation it was trained around."


def read_offers(store, folder) -> tuple:
    """What a reopened window can offer, read now: the kept trainings of ``store`` (none read when the store lists no
    kept training), the run history in ``folder``, each simulation set's own count, read from the set, each kept
    training's prior as a sentence names it, and a kept narrowing round's observation and parent posterior the same
    way, each record read once; a kept training whose records its Resume loads are no longer in Records
    (``records_gone``, from those same readings) offers no Resume and says why. The one reading of these; its caller
    never asks while a run is live. What cannot be read offers nothing, and nothing raises."""
    from types import SimpleNamespace
    from core.SBI import fit_state, training_checkpoint
    from ... import record_words
    try:
        kept = fit_state.kept_fits(store) if store.list("fit") else []
    except Exception:                        # noqa: BLE001 -- a store that cannot be read keeps nothing to offer
        kept = []
    try:
        runs = history.runs_of(Path(folder), records_root=str(store.root))
    except OSError:
        runs = []

    def set_batches(folder: str):
        # The set's folder as the history holds it, its whole path, which the join leaves as it is.
        try:
            state = training_checkpoint.peek(store.kind_dir("simulation") / folder)
        except Exception:                    # noqa: BLE001 -- a set that cannot be read holds nothing to continue
            return None
        done = (state or {}).get("batches_done")
        return done if isinstance(done, int) and not isinstance(done, bool) else None

    held: dict = {}

    def present(kind: str, ident: str):
        # Each record read once, with ``store.get``; raises, as it does, for one no longer in Records.
        if (kind, ident) not in held:
            try:
                held[kind, ident] = store.get(kind, ident)
            except Exception:                # noqa: BLE001 -- a record no longer in Records is said so
                held[kind, ident] = None
        if held[kind, ident] is None:
            raise LookupError(kind)
        return held[kind, ident]

    def label(kind: str, ident: str):
        try:
            manifest = present(kind, ident)
            return record_words.phrase_of(kind, SimpleNamespace(kind=kind, name=manifest.name, manifest=manifest))
        except Exception:                    # noqa: BLE001 -- a record no longer in Records is said so
            return None

    return continuing.offers(runs, kept, records_root=str(store.root), set_batches=set_batches,
                             prior_label=lambda ident: label("prior", ident), label=label,
                             gone=lambda row: records_gone(row, present))


def _input(kept, key: str) -> "tuple[Path, str | None] | None":
    """The input file ``key`` the kept training was made from, where it is now and the hash it had; None for none."""
    held = (getattr(kept, "inputs", None) or {}).get(key)
    if not isinstance(held, dict) or not isinstance(held.get("path"), str) or not held["path"]:
        return None
    path = Path(held["path"])
    if not path.is_absolute():
        path = Path(config.RESOURCES_ROOT) / path
    sha = held.get("sha256")
    return path, sha if isinstance(sha, str) and sha else None


def _sha(path: Path) -> "str | None":
    from core.artifacts.provenance import sha256_file
    try:
        return sha256_file(path)
    except OSError:
        return None


def _in_records(present, kind: str, ident) -> bool:
    """Whether the record ``ident`` of ``kind`` is in Records, as ``present`` finds it (``store.get``, or a reading of
    it), which raises for one that is not; an id that is no name is in no Records."""
    if not isinstance(ident, str) or not ident:
        return False
    try:
        present(kind, ident)
    except Exception:                        # noqa: BLE001 -- deleted, torn, or never named
        return False
    return True


def prior_gone(kept, present) -> tuple[str, ...]:
    """The prior the kept training ``kept`` names gone from Records, as its sentence; none while ``present`` finds
    it."""
    prior = (getattr(kept, "parents", None) or {}).get("prior")
    return () if _in_records(present, "prior", prior) else (PRIOR_GONE_TEXT,)


def round_records_gone(kept, present) -> tuple[str, ...]:
    """The posterior the kept narrowing round ``kept`` narrows and the observation it was trained around, each gone from
    Records by the id its record names, as its sentence: the posterior named by that id, which is all that is known of
    a record that is gone; a record that names no such id says so."""
    found = []
    parents = getattr(kept, "parents", None) or {}
    for kind, key in (("posterior", "parent_posterior"), ("observation", "observation")):
        ref = parents.get(key)
        if not isinstance(ref, str) or not ref:
            found.append(NO_PARENT_TEXT if kind == "posterior" else NO_OBSERVATION_TEXT)
        elif not _in_records(present, kind, ref):
            found.append(PARENT_GONE_TEXT.format(ref) if kind == "posterior" else OBSERVATION_GONE_TEXT)
    return tuple(found)


def records_gone(kept, present) -> tuple[str, ...]:
    """What of the records the Resume of ``kept`` loads is no longer in Records, each a sentence, as ``present``
    (``store.get``, or a reading of it) finds them: its prior; for a kept narrowing round, then the posterior it narrows
    and its observation. The one rule its press refuses by and the reading of what is offered says before any press;
    no file is read and no device is asked."""
    found = prior_gone(kept, present)
    if getattr(kept, "region", None) is not None:
        found += round_records_gone(kept, present)
    return found


def resume_problems(kept, *, device, store) -> tuple[str, ...]:
    """What stops the Resume of ``kept`` before anything is read, each a sentence: its bounds file gone or changed since
    it began, or limits typed by hand with a forcing section, which its record cannot rebuild; its units file gone or
    changed; its prior gone from Records (``prior_gone``); its model no longer one PRISM can train; a training kept on
    the graphics card on a computer that has none now (``device``, the device check's answer). A library or the
    training code changed is the kept training's own ``why_not``, said before any Resume is offered."""
    found = []
    record = getattr(kept, "config", None) or {}
    model = record.get("model")
    if model not in config.VALID_MODELS:
        found.append(NO_MODEL_TEXT.format(model))
    bounds = _input(kept, "bounds")
    if bounds is None:
        if not record.get("spontaneous_only", True):
            found.append(TYPED_FORCING_TEXT)
    elif not bounds[0].is_file():
        found.append(BOUNDS_GONE_TEXT.format(bounds[0].name))
    elif bounds[1] is not None and _sha(bounds[0]) != bounds[1]:
        found.append(BOUNDS_CHANGED_TEXT.format(bounds[0].name))
    units = _input(kept, "units")
    if units is not None:
        if not units[0].is_file():
            found.append(UNITS_GONE_TEXT.format(units[0].name))
        elif units[1] is not None and _sha(units[0]) != units[1]:
            found.append(UNITS_CHANGED_TEXT.format(units[0].name))
    found += prior_gone(kept, store.get)
    if record.get("device") == "cuda" and getattr(getattr(device, "device", None), "type", None) != "cuda":
        found.append(NO_CARD_TEXT)
    return tuple(found)


def round_problems(kept, *, device, store) -> tuple[str, ...]:
    """What stops the Resume of the kept narrowing round ``kept`` before anything is read, each a sentence: what stops a
    training's (``resume_problems``), then the posterior it narrows and the observation it was trained around gone
    from Records (``round_records_gone``)."""
    return resume_problems(kept, device=device, store=store) + round_records_gone(kept, store.get)


def resume_draft(kept) -> ConfigDraft:
    """The draft of step 1 the kept training was made with, from its record alone: its model, its units (the model's
    file, another file by its path, or the units typed when its record names no file), its kind of observation, its
    slots and its lock-in limit, its rotation, and the number of frequencies at its built-in, which enters no
    identity."""
    from core import registry
    from core.cli import resolve_units_file
    record = getattr(kept, "config", None) or {}
    model = record["model"]
    chi = bool(record.get("chi_mode"))
    # Units read from a file are named by the record's inputs, and the record's block lists their tokens too; only
    # units typed name no file. A file other than the model's own is handed on by its path.
    units_file, units = _input(kept, "units"), record.get("units")
    if units_file is not None:
        try:
            own = Path(resolve_units_file(model)).resolve() == units_file[0].resolve()
        except Exception:                    # noqa: BLE001 -- a model whose own file cannot be named
            own = False
        override = None if own else str(units_file[0])
    else:
        override = tuple(str(unit) for unit in units) if isinstance(units, list) else None
    return ConfigDraft(model=model, labels=config.VALID_LABELS[config.VALID_MODELS.index(model)],
                       state_dep_drift=registry.state_dep_drift(model), units_override=override,
                       chi_mode=chi, chi_n_freqs=None, chi_f0=None, chi_freq_bounds=None,
                       chi_k_pad=record.get("chi_k_pad") if chi else None,
                       chi_max_cycles=record.get("chi_max_cycles") if chi else None,
                       reparam_rotate=record.get("reparam_rotate"))


def resume_config(kept):
    """The configuration the kept training was made with, from its record alone: its draft on its bounds file, or on
    the limits its record holds when they were typed by hand, on the device and at the precision it was made on, its
    batch as wide as the simulations per batch its set was made with, so that "Automatic" resolves as it did."""
    import torch
    from core import cli
    record = getattr(kept, "config", None) or {}
    draft = resume_draft(kept)
    device = torch.device(record.get("device") or "cpu")
    dtype = getattr(torch, str(record.get("dtype") or "torch.float32").replace("torch.", ""))
    width = record.get("run_size")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        width = config.detect_device().batch_size if device.type != "cpu" else config.cpu_device().batch_size
    hw = config.DeviceConfig(device=device, dtype=dtype, batch_size=width)
    bounds = _input(kept, "bounds")
    common = dict(units_override=draft.units_override, chi_mode=draft.chi_mode, chi_n_freqs=draft.chi_n_freqs,
                  chi_f0=None, chi_freq_bounds=None, chi_k_pad=draft.chi_k_pad, chi_max_cycles=draft.chi_max_cycles,
                  reparam_rotate=draft.reparam_rotate, hw=hw)
    if bounds is not None:
        return cli.make_sim_config(draft.model, draft.labels, draft.state_dep_drift, str(bounds[0]), **common)
    keys = list(record.get("param_keys") or [])
    lows = list(record.get("nd_lows") or []) + list(record.get("rescale_lows") or [])
    highs = list(record.get("nd_highs") or []) + list(record.get("rescale_highs") or [])
    count = len(record.get("nd_lows") or [])
    params = {name: (None, (lo, hi)) for name, lo, hi in zip(keys[:count], lows[:count], highs[:count])}
    rescale = {name: (None, (lo, hi)) for name, lo, hi in zip(keys[count:], lows[count:], highs[count:])}
    return cli.make_sim_config(draft.model, draft.labels, draft.state_dep_drift, bounds_dicts=(params, rescale, {}),
                               **common)


def resume_call(kept, name: str = "") -> dict:
    """The keywords the window's Resume hands the training stage beside its four arguments: the kept training named,
    and the name asked for when its record holds none. Nothing the stage would compare with the record: no budget, no
    network, no patience, no ceiling, no cadence, no note, no saving switch and no policy."""
    call = {"resume_fit": getattr(kept, "id")}
    if name:
        call["name"] = name
    return call


def round_call(kept, name: "str | None" = None) -> dict:
    """The keywords the window's Resume of a kept narrowing round hands the round beside its four arguments: the kept
    training named, and a name only when one was asked for. Nothing the round compares with the record: no number of
    directions, no level, no budget, no network, no cadence and no note; the round reads its region from the record."""
    call = {"resume_fit": getattr(kept, "id")}
    if name:
        call["name"] = name
    return call


def bounds_key(kept) -> "str | None":
    """The bounds file of ``kept`` as step 2's list names it in its model's folder; None for limits typed by hand or a
    file outside that folder."""
    bounds = _input(kept, "bounds")
    if bounds is None:
        return None
    folder = Path(config.BOUNDS_PATH) / str((getattr(kept, "config", None) or {}).get("model", "")).lower()
    try:
        return PurePath(bounds[0].resolve().relative_to(folder.resolve())).as_posix()
    except (ValueError, OSError):
        return None
