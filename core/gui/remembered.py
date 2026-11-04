"""What the window remembers is written here, and nowhere else: a setting of the Settings page when it is changed, and
later each pick when it comes into use. Every write reaches the settings file at once, so that a crash loses nothing
that was applied, and a session that only looks writes nothing.

A write that fails is said in the window's log, and its caller is told: what is remembered stops nothing that is being
done. Keys are passed by position: a settings key is no refusal's key.
"""
import logging
from collections.abc import Iterable

from PySide6.QtCore import QSettings

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
#: The posterior that came into use last, put back as a choice at the next launch; and the batches and the simulations
#: per batch of the training started last, written as the text of their boxes once its run has been started.
POSTERIOR_KEY = "inference_posterior/posterior"
NUM_RUNS_KEY = "inference_posterior/num_runs"
RUN_SIZE_KEY = "inference_posterior/run_size_cap"
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
