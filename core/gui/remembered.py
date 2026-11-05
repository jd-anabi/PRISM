"""What the window remembers is written here, and nowhere else: a setting of the Settings page when it is changed, and
later each pick when it comes into use. Every write reaches the settings file at once, so that a crash loses nothing
that was applied, and a session that only looks writes nothing.

A write that fails is said in the window's log, and its caller is told: what is remembered stops nothing that is being
done. Keys are passed by position: a settings key is no refusal's key.

What was in use last time is read here too, for Home: the picks the settings file remembers, each left out when it can
no longer be found as it was used, so that Home names exactly what one press can apply and load. Reading them writes
nothing.
"""
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import SimpleNamespace

from PySide6.QtCore import QSettings

from core import config
from core.config import BOUNDS_PATH

from . import settings

log = logging.getLogger(__name__)

#: The model and the kind of observation the first step applied last, put back as choices at the next launch.
MODEL_KEY = "inference_config/model"
KIND_KEY = "inference_config/chi_mode"
#: The bounds file and the prior of the prior that came into use last, put back as choices at the next launch, and the
#: model the bounds file came into use with, whose folder holds it.
BOUNDS_KEY = "inference_prior/bounds"
BOUNDS_MODEL_KEY = "inference_prior/bounds_model"
PRIOR_KEY = "inference_prior/prior"
#: The posterior that came into use last, written as it comes into use and put back as the choice of the Training
#: page's list of saved posteriors at the next launch; and the batches and the simulations per batch of the training
#: started last, written as the text of their boxes once its run has been started.
POSTERIOR_KEY = "inference_posterior/posterior"
NUM_RUNS_KEY = "inference_posterior/num_runs"
RUN_SIZE_KEY = "inference_posterior/run_size_cap"
#: The plan last used with a model, by its name, and when, each written with what comes into use or starts with it,
#: blank when that had no plan: the hint beside the list of plans on step 1. The model's name in lower case completes
#: each.
LAST_USED_KEY = "plans/last_used/{}"
LAST_USED_AT_KEY = "plans/last_used_at/{}"
#: The window's log, when the settings file could not be written: the file, and why.
NOT_WRITTEN_TEXT = "The settings file %s could not be written (%s): what was set holds until PRISM is closed."


def put(key: str, value: object) -> bool:
    """Remember ``value`` under ``key``, now. True when it reached the settings file."""
    return put_many(((key, value),))


def put_many(pairs: Iterable[tuple[str, object]]) -> bool:
    """Remember each ``(key, value)`` of ``pairs``, now, in one write of the file. True when they reached it."""
    store = None
    try:
        store = settings.settings()
        for key, value in pairs:
            store.setValue(key, value)
        store.sync()
    except Exception as error:               # noqa: BLE001 -- what is remembered stops nothing being done
        log.warning(NOT_WRITTEN_TEXT, "" if store is None else store.fileName(), f"{type(error).__name__}: {error}")
        return False
    if store.status() != QSettings.NoError:
        log.warning(NOT_WRITTEN_TEXT, store.fileName(), store.status().name)
        return False
    return True


# ── what was in use last time ──
#: The kind of observation as the settings file holds it: the words a truth value is written as.
_KINDS = {"1": True, "true": True, "0": False, "false": False}


@dataclass(frozen=True)
class Picks:
    """What was in use last time, each None when it can no longer be found as it was used: the model, by its engine
    name; the kind of observation, True for the frequency response and False for the kind the bounds file sets, None
    when the file holds no truth value; the bounds file, as step 2's list names it in the model's own folder; the saved
    prior and the saved posterior, by their ids, each with the words a sentence names it by ("the prior retrain_prior",
    "the unnamed posterior, 1 Oct 2026, 22:29"); and the plan last used with the model, by its name as the folder of
    plans lists it."""
    model: "str | None" = None
    chi_mode: "bool | None" = None
    bounds: "str | None" = None
    prior: "str | None" = None
    posterior: "str | None" = None
    plan: "str | None" = None
    prior_words: str = ""
    posterior_words: str = ""


def _manifest(store, kind: str, ident: str):
    """The manifest of the record ``ident`` of ``kind``, or None when it cannot be read: deleted, torn, or in a store
    that cannot be read."""
    try:
        return store.get(kind, ident)
    except Exception:                        # noqa: BLE001 -- a record that cannot be read is one that is gone
        return None


def _words_of(word: str, manifest) -> str:
    """How a sentence names a record by its manifest: "the prior retrain_prior", or "the unnamed prior, 1 Oct 2026,
    22:29"."""
    from . import record_words
    return record_words.phrase_of(word, SimpleNamespace(kind=manifest.kind, name=manifest.name, manifest=manifest))


def last_plan(model: str, name: str, *, plans_folder=None):
    """The plan ``name`` last used with ``model``, read and checked as step 1 reads a plan, in the model's folder of plans
    under ``plans_folder`` (the window's folder of plans unless one is handed), found without regard to case and named
    as the folder lists it: None when there is no name, or the plan is gone or has problems."""
    from core import plans as engine_plans
    from . import plans
    folder = plans.model_folder(plans.PLAN_FOLDER if plans_folder is None else plans_folder, model)
    path = plans.taken(folder, name) if name else None
    if path is None:
        return None
    try:
        return plans.read_checked(path, bounds_folder=BOUNDS_PATH / model.lower())
    except engine_plans.PlanRefusal:
        return None


def remembered_picks(*, store, plans_folder=None) -> Picks:
    """What was in use last time, read from the settings file, for Home to name and one press to apply and load: the
    model step 1 applied last, while the list of models holds it and parameter inference can take it (else nothing at
    all); the kind of observation applied with it; the bounds file of the prior that came into use last, while it came
    into use with that model and that model's folder lists it by exactly that name; that prior, while the bounds file
    stands and its record, made for that model on that very file, can be read; the posterior that came into use last,
    while the prior stands and its record, which can be read, names that prior as the one it was trained with and is
    of the kind of observation applied; and the plan last used with the model, while its file is there and has no
    problem. A bounds file remembered with no model, as an older PRISM wrote it, is the model's. Nothing is written,
    and nothing is read but the settings file, the model's folder of bounds files, two records' manifests and one plan
    file."""
    from core import registry
    from core.Helpers import file_manager
    from .panels.inference.prior_tab import recorded_file
    qs = settings.settings()
    model = settings.get_str(qs, MODEL_KEY, "")
    if model not in config.VALID_MODELS or (registry.is_user_model(model) and not registry.is_sbi_user_model(model)):
        return Picks()
    kind = _KINDS.get(str(qs.value(KIND_KEY, "")).strip().lower())
    bounds = prior = posterior = None
    prior_words = posterior_words = ""
    key, folder = settings.get_str(qs, BOUNDS_KEY, ""), BOUNDS_PATH / model.lower()
    if (key and (settings.get_str(qs, BOUNDS_MODEL_KEY, "") or model) == model and Path(folder).is_dir()
            and key in file_manager.list_dir(str(folder))):
        bounds = key
    ident = settings.get_str(qs, PRIOR_KEY, "")
    manifest = _manifest(store, "prior", ident) if bounds and ident else None
    if (manifest is not None and (manifest.config or {}).get("model") == model
            and recorded_file((manifest.inputs or {}).get("bounds")) == PurePath(bounds).as_posix()):
        prior, prior_words = ident, _words_of("prior", manifest)
    ident = settings.get_str(qs, POSTERIOR_KEY, "")
    manifest = _manifest(store, "posterior", ident) if prior and ident else None
    if (manifest is not None and (manifest.parents or {}).get("prior") == prior
            and ((manifest.body or {}).get("mode") == "chi") == (kind is True)):
        posterior, posterior_words = ident, _words_of("posterior", manifest)
    used = last_plan(model, settings.get_str(qs, LAST_USED_KEY.format(model.lower()), "").strip(),
                     plans_folder=plans_folder)
    return Picks(model, kind, bounds, prior, posterior, None if used is None else used.name, prior_words,
                 posterior_words)
