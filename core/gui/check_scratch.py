"""The scratch store of the two checks that keep nothing but their result, the timing of a short fit and the card
test: a store of their own, in a folder of the computer's temporary files made for the one run, handed to the stage
as its store and removed however the run ends. The process's store is never swapped for it, since every page reads
that one, and nothing either check makes reaches the records in use.

What a check's run reads of its own records it reads as they are said, by a handler on the engine's logger that only
notes; what it hands back is its result as names and numbers, which its run's history keeps. Both calls run on the
worker's thread.
"""
import gc
import logging
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from core.artifacts import ArtifactStore

log = logging.getLogger(__name__)

#: The name a scratch folder begins with, among the computer's temporary files.
SCRATCH_PREFIX = "prism_check_"
#: How many times in all, and how many seconds apart, a scratch folder's removal is tried before it is given up and
#: said: Windows may hold a file of it open for a moment after the run has let it go (a scanner, an indexer).
RMTREE_TRIES = 5
RMTREE_PAUSE_S = 0.2
#: What the run's text says, once, of a scratch folder that could not be removed.
LEFT_TEXT = "The scratch folder %s of a check that keeps nothing could not be removed (%s); it can be deleted by hand."


@contextmanager
def scratch_store(*, prefix: str = SCRATCH_PREFIX, pause=time.sleep):
    """A store of its own in a new temporary folder whose name begins ``prefix``, for the block; the folder removed
    when the block ends, however it ends (finished, failed, stopped). The caller drops its references to what the
    store loaded before the block ends. A removal Windows refuses is tried again, ``RMTREE_TRIES`` times in all,
    ``pause`` (``RMTREE_PAUSE_S`` seconds) between two, what is no longer referenced collected first; a folder that
    still cannot go is said once, as a warning that names it, and the block's own end passes on unchanged."""
    folder = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield ArtifactStore(folder)
    finally:
        _remove(folder, pause)


def _remove(folder: Path, pause) -> None:
    """Remove ``folder`` and all it holds, as ``scratch_store`` says; never raises."""
    error = None
    for tried in range(1, RMTREE_TRIES + 1):
        try:
            shutil.rmtree(folder)
            return
        except FileNotFoundError:
            return
        except OSError as refused:
            error = refused
            if tried < RMTREE_TRIES:
                gc.collect()                 # a record let go but still held in a cycle may hold a file of it open
                pause(RMTREE_PAUSE_S)
    if folder.exists():
        log.warning(LEFT_TEXT, folder, f"{type(error).__name__}: {error}")
