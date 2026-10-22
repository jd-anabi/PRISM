"""A training's kept fit: its files, its lock and the digests that make up its identity.

A fit is sbi's training of the neural posterior. Kept, it is a folder of its own:

    header.pt       written once, when the fit is first kept: its identity
    state.pt        the state after the last finished pass, rewritten after every pass
    state.prev.pt   the state one pass before it, so that a save torn mid-way still leaves a whole pass
    running.lock    whose operating-system lock is held by the process training the fit now

THE TWO GENERATIONS. ``save_state`` writes the new state to a temporary file, renames ``state.pt`` to
``state.prev.pt``, then renames the temporary file to ``state.pt`` (``atomic_torch_save`` with
``previous``). Torn at any step, one whole pass is left: ``load_state`` reads ``state.pt`` and, when that
is absent or cannot be read, ``state.prev.pt``. The whole save runs inside the deferred-cancel section
(``core.runs.cancel_deferred``) and nothing is printed or logged inside it: under the window every
record passes a cancel checkpoint, and a Cancel taken between the two renames would stop a save that
has to finish.

THE LOCK. A byte of ``running.lock`` locked by the operating system, so a process that dies lets it go
without anyone cleaning up: a lock file nobody holds says only that a process once trained here. The
lock belongs to an open handle, not to a process: a second handle in the same process cannot take it.

THE IDENTITY. A kept fit is resumed only while nothing it was trained with has changed. Its identity
has the parts named in ``IDENTITY_PARTS``, compared in that order by ``first_difference``: the
simulated rows, the number of batches read, the training settings, a digest of the source of each
function and class that builds or trains the network (``FIT_CODE``), the configuration's constants
that code reads (``FIT_CONSTANTS``), the libraries with a digest of the files of sbi's trainer, and the
shape of the network. Only the digests are made here; what fills each part is the trainer's. The code's
digests are the ones kept when that code was loaded (``loaded_code``): a process runs the code it
imported, and a window left open across an edit must not record the edited file as what it trains with.

THE RECORD. A kept fit is a record of the store's kind ``fit``, with no writer: ``FitSession`` makes
its folder at the first saved pass, refreshes its manifest and its ``log.txt`` at every pass, records
how the training stopped, and removes the record once the training's posterior is written.
``kept_fits`` lists the kept fits with whether each can be resumed and, when it cannot, why; it reads
their manifests and never a fit's state files, since a reader holding one would make the next save's
rename fail on Windows.

THE RESUME. ``kept_fit`` finds a kept fit by its id, as the listing reads it; ``FitSession.reopen``
takes its lock and reads its header and its newest whole pass, and the session then saves the resumed
fit into the same record. A check made once the rows are read back that finds the fit cannot continue
marks the record so (``FitSession.mark_unresumable``) and changes nothing else, and so does a header
saved in another format; a header or a pass that cannot be read at the moment marks nothing, since it
may read at the next attempt.
"""
import errno
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from core import runs
from core.Helpers.file_manager import atomic_torch_save
from core.refusals import Refusal

if os.name == "nt":
    import msvcrt
else:
    import fcntl

log = logging.getLogger(__name__)

FIT_FORMAT: int = 1
HEADER = "header.pt"
STATE = "state.pt"
STATE_PREV = "state.prev.pt"
LOCK = "running.lock"
# Where a resumed fit moves a newest state that could not be read, before its first save: a name
# ``load_state`` never reads, so the save's rotation cannot put it over the last good pass.
_STATE_UNREADABLE = "state.unreadable.pt"

# The functions and classes whose source makes up the code digest: what builds the embedding network
# and the features it reads, and what trains the network.
FIT_CODE: tuple[str, ...] = (
    "core.SBI.train.train_nn",
    "core.SBI.train.fit_settings",
    "core.SBI.train.build_trainer",
    "core.SBI.train.prepare_fit",
    "core.SBI.train.run_fit",
    "core.SBI.train.take_snapshot",
    "core.SBI.train.restore_trainer",
    "core.SBI.train._capped_zscore_check",
    "core.SBI.embedded_network._probit",
    "core.SBI.embedded_network._column_quantiles",
    "core.SBI.embedded_network.EmbeddedNet",
    "core.SBI.chi_encoder.ChiSetEncoder",
    "core.SBI.summaries.winsorize_summary_block",
    "core.SBI.chi.band_norm",
    "core.SBI.chi.n_chi_features",
    "core.orchestrator.build_embedding_net",
    "core.orchestrator.expected_forcing_dim",
)

# The constants of ``core.config`` the code above reads to shape the network or its features.
FIT_CONSTANTS = ("RANK_GAUSS_KNOTS", "WINSOR_PCT", "CHI_PHI_DIM", "CHI_BIN_DIM", "CHI_SET_OUT", "CHI_RHO_HIDDEN",
                 "CHI_KNOTS", "CHI_KNOT_SIGMA", "CHI_KNOT_SHRINK", "CHI_ELEM_W", "CHI_LAYOUT")

# The files of sbi's trainer, relative to the installed ``sbi`` package: the training loop, the
# posterior estimator's loss and the builders of the flow.
SBI_TRAINER_FILES = ("inference/trainers/base.py", "inference/trainers/npe/npe_base.py",
                     "inference/trainers/npe/npe_c.py", "neural_nets/factory.py",
                     "neural_nets/net_builders/flow.py")

IDENTITY_PARTS = ("simulation", "batches", "settings", "code", "constants", "libraries", "network")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


# ── the identity's digests ───────────────────────────────────────────────────────────────────────
def code_digests(names: tuple[str, ...] = FIT_CODE) -> dict[str, str]:
    """``{dotted name: digest of its source}``, the source read as the file stands now.

    Each name is a module path and an attribute (``"core.SBI.chi.band_norm"``). ``core.orchestrator`` is
    imported first: it loads the pipeline before the trainer, and importing ``core.SBI.train`` on its
    own is a circular import. An object whose source cannot be read digests to ``"unreadable"``.
    """
    importlib.import_module("core.orchestrator")
    out = {}
    for dotted in names:
        module, _, attr = dotted.rpartition(".")
        obj = getattr(importlib.import_module(module), attr)
        try:
            out[dotted] = _digest(inspect.getsource(obj).encode("utf-8"))
        except (OSError, TypeError, SyntaxError):   # no file, a built-in, a class's file mid-edit
            out[dotted] = "unreadable"
    return out


# The digests of the named code as this process loaded it, by dotted name (``remember_loaded_code``).
_LOADED: dict[str, str] = {}


def remember_loaded_code(names: tuple[str, ...] = FIT_CODE) -> None:
    """Keep, for the rest of the process, the digest of each named function or class as its file
    stands now; a name already kept keeps its first digest.

    Called as the last statement of the module that builds and trains the network, when the named code
    has just been loaded. A process runs the code it imported, while ``inspect.getsource`` reads the
    file as it stands when asked: a window left open across an edit would otherwise record, for a fit
    it starts, the edited text while it trains the code it loaded, and a resume in a new window,
    running the edited code, would match it. A name that does not resolve raises, as in
    ``code_digests``, so a list out of step with the code fails the import at once.
    """
    missing = tuple(name for name in names if name not in _LOADED)
    if missing:
        _LOADED.update(code_digests(missing))


def loaded_code(names: tuple[str, ...] = FIT_CODE) -> dict[str, str]:
    """``{dotted name: digest}`` of the named code as this process loaded it: what a fit records as
    its code. A name not kept yet is read now and kept (``remember_loaded_code``)."""
    remember_loaded_code(names)
    return {name: _LOADED[name] for name in names}


def _as_lists(value):
    if isinstance(value, (tuple, list)):
        return [_as_lists(v) for v in value]
    return value


def constants(names: tuple[str, ...] = FIT_CONSTANTS) -> dict:
    """``{name: value}`` of each constant of ``core.config`` as the configuration stands now, tuples as
    lists so the record's text can carry them."""
    from core import config        # here, so a value is read when the identity is made
    return {name: _as_lists(getattr(config, name)) for name in names}


def _sbi_root() -> Path:
    return Path(importlib.util.find_spec("sbi").origin).parent


def trainer_files_digest(root: Path | None = None) -> str:
    """One digest over the files of sbi's trainer: each file's path relative to ``root``, then its
    bytes with Windows line ends read as Unix ones. ``root`` defaults to the installed sbi package."""
    root = Path(root) if root is not None else _sbi_root()
    h = hashlib.sha256()
    for rel in SBI_TRAINER_FILES:
        h.update(rel.encode("utf-8"))
        h.update((root / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def libraries() -> dict[str, str]:
    """The versions the fit is trained under, and the digest of the files of sbi's trainer."""
    return {"torch": str(torch.__version__),
            "sbi": importlib.metadata.version("sbi"),
            "nflows": importlib.metadata.version("nflows"),
            "sbi_trainer_files": trainer_files_digest()}


def network_digest(net: torch.nn.Module) -> str:
    """A digest of the network's shape: the name and shape of every entry of its state, never a value."""
    entries = sorted(net.state_dict().items(), key=lambda item: item[0])
    return _digest("\n".join(f"{name}:{tuple(t.shape)}" for name, t in entries).encode("utf-8"))


def indices_digest(indices: torch.Tensor) -> str:
    """A digest of a split's row indices, in their order, read as 64-bit integers."""
    flat = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return _digest(flat.numpy().tobytes())


# ── comparing two identities ─────────────────────────────────────────────────────────────────────
class FitMismatch(Exception):
    """A check made once the rows are read found that the kept fit cannot continue. ``str(e)`` is a
    sentence fragment naming what differs."""


@dataclass(frozen=True)
class Difference:
    """The first thing two identities disagree on: the part, the name within it (the part's own name
    for a part that is one value), and the two values. A name one side lacks reads as None there."""
    part: str
    name: str
    kept: object
    now: object


def first_difference(kept: dict, now: dict, parts: tuple[str, ...] = IDENTITY_PARTS) -> Difference | None:
    """The first difference between a kept identity and the present one, or None.

    Parts are read in the order of ``parts``; a part ``now`` does not have is skipped (the present
    cannot say it yet). A part that is a dict on both sides is compared name by name, the kept dict's
    order first and then the names only ``now`` has; any other part is compared whole. Values are
    compared as the simulation cache's identity compares them: floats by their exact text, tuples as
    lists.
    """
    from core.SBI.training_checkpoint import _canonical

    for part in parts:
        if part not in now:
            continue
        old, new = kept.get(part), now[part]
        if isinstance(old, dict) and isinstance(new, dict):
            for name in [*old, *(n for n in new if n not in old)]:
                if (name in old) != (name in new) or _canonical(old.get(name)) != _canonical(new.get(name)):
                    return Difference(part, name, old.get(name), new.get(name))
        elif part not in kept or _canonical(old) != _canonical(new):
            return Difference(part, part, old, new)
    return None


# ── the files ────────────────────────────────────────────────────────────────────────────────────
def write_header(path: Path, header: dict) -> None:
    """Write the fit's header into its folder ``path``, atomically."""
    atomic_torch_save(header, Path(path) / HEADER)


def read_header(path: Path) -> dict:
    return torch.load(Path(path) / HEADER, map_location="cpu", weights_only=False)


def save_state(path: Path, state: dict) -> None:
    """Save the state after a pass into the fit's folder ``path``, keeping the one before it.

    The whole save is one deferred-cancel section and says nothing: a Cancel waits until it is done.
    """
    path = Path(path)
    with runs.cancel_deferred():
        atomic_torch_save(state, path / STATE, previous=path / STATE_PREV)


def load_state(path: Path) -> dict | None:
    """The newest whole state in the fit's folder ``path``: ``state.pt``, else ``state.prev.pt``; None
    when neither can be read. The dict gains ``"_state_file"``, the name of the file it came from."""
    path = Path(path)
    for name in (STATE, STATE_PREV):
        f = path / name
        if not f.exists():
            continue
        try:
            state = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:                    # noqa: BLE001 -- a torn or truncated file, or another version's
            continue
        if isinstance(state, dict):
            return {**state, "_state_file": name}
    return None


# ── the lock ─────────────────────────────────────────────────────────────────────────────────────
if os.name == "nt":
    _HELD = {errno.EACCES, errno.EDEADLOCK}          # what a refused non-blocking lock raises here

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)                 # the lock covers one byte from the current position
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as e:
            if e.errno in _HELD:
                return False
            raise
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    _HELD = {errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES}

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in _HELD:
                return False
            raise
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class FitLock:
    """The lock of the fit kept in the folder ``path``: held while a process trains it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Take the lock, opening or creating the lock file, and keep its handle open; False when
        another handle holds it. True at once when this one already does."""
        if self._fd is not None:
            return True
        fd = os.open(self.path / LOCK, os.O_RDWR | os.O_CREAT)
        try:
            taken = _try_lock(fd)
        except BaseException:
            os.close(fd)
            raise
        if not taken:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        """Let the lock go and close its handle. Releasing a lock not held does nothing."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            pass                                     # closing the handle lets it go all the same
        finally:
            os.close(fd)

    @property
    def held(self) -> bool:
        return self._fd is not None


def running(path: Path) -> bool:
    """True when an open handle holds the lock of the fit kept in ``path``, this process's included;
    False when there is no lock file. Never raises.

    A fresh handle tries the lock and lets it go at once, so for that instant it holds it: a resume
    whose ``FitLock.acquire`` lands in that instant is refused as being trained by a running process,
    once, and succeeds when asked again. Rare and harmless, so it is left so.
    """
    try:
        fd = os.open(Path(path) / LOCK, os.O_RDWR)
    except OSError:
        return False
    try:
        if not _try_lock(fd):
            return True
        _unlock(fd)
        return False
    except OSError:
        return False
    finally:
        os.close(fd)


# ── the record's state, in words ─────────────────────────────────────────────────────────────────
# The words a kept training's state is listed by, in the store's listing and on both front ends.
STATE_WORDS = ("running", "stopped unexpectedly", "kept when stopped", "cannot be resumed", "finished")


def state_word(path: Path, body: dict) -> str:
    """The word for the state of the kept training in the folder ``path`` whose manifest body is
    ``body``: the first that holds of

    - "running": a process holds its lock now (``running``);
    - "cannot be resumed": the record says why it cannot be (``unresumable``);
    - "finished": its fit ended, and its posterior was not written before its process ended or the
      posterior could not be written (``complete``);
    - "kept when stopped": it was stopped on purpose;
    - "stopped unexpectedly": anything else -- a crash, a restart, a power cut, or an error the process
      recorded with its reason.

    Never raises: a body a hand edit left in another shape reads as stopped unexpectedly.
    """
    if running(path):
        return "running"
    body = body if isinstance(body, dict) else {}
    if body.get("unresumable"):
        return "cannot be resumed"
    if body.get("complete"):
        return "finished"
    stopped = body.get("stopped")
    if isinstance(stopped, dict) and stopped.get("on_purpose"):
        return "kept when stopped"
    return "stopped unexpectedly"


# ── the record of a kept fit ─────────────────────────────────────────────────────────────────────
def _plain(value):
    """``value`` as a record's text carries it: JSON's numbers, strings, lists and mappings, so that a
    tuple reads as a list and a numpy float as a float, and an identity compares equal to the one read
    back from a record."""
    return json.loads(json.dumps(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value) -> "float | None":
    """A loss as a record holds it: a float, or None when it is not a finite number (the record cannot
    carry one)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _progress(snapshot: dict, wall: float, processes: int) -> dict:
    """How far the fit has gone, from the pass's own histories: the validation losses so far include
    the pass just saved, where the trainer's own best loss and its count of passes since the best are
    brought up to date only as the next pass begins."""
    losses = [_finite(v) for v in snapshot["summary"].get("validation_loss", [])]
    finite = [v for v in losses if v is not None]
    best = min(finite) if finite else None
    return {"passes": int(snapshot["passes"]), "best_validation_loss": best,
            "passes_since_best": None if best is None else len(losses) - 1 - losses.index(best),
            "last_validation_loss": losses[-1] if losses else None, "saved_at": _utc_now(),
            "wall_seconds": float(wall), "processes": int(processes)}


def _read_log(path: Path) -> str:
    """The kept record's ``log.txt``, the earlier processes' records; "" when there is none or it cannot
    be read."""
    from core.artifacts.store import LOG_FILE
    try:
        return (Path(path) / LOG_FILE).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _reason(exc: BaseException) -> str:
    """What stopped a training, in one line: the exception's name, then its words when it has any."""
    said = " ".join(str(exc).split())
    return type(exc).__name__ + (f": {said}" if said else "")


class FitSession:
    """The kept record of one training: handed the fit at the end of every pass (it is the trainer's
    saver), it keeps the fit as a ``fit`` record of the store, records how the training stopped, and
    removes the record once the training's posterior is written.

    The record's folder is made at the first pass saved, not before, and ``id`` is the record's id once
    a pass is kept (None until then): a training that ends or stops with no pass kept leaves nothing
    behind. Its manifest and its ``log.txt`` are refreshed at every pass. A save that fails (a
    full disk, a file another program holds) is warned and never stops the training; a Ctrl-C or the
    window's Cancel is never caught here. The lock taken at the first save is let go on every way out: a
    stop, a finish whose posterior cannot be written, and the end.

    ``identity`` holds the parts known before the fit (the rows, the batches, the code, the constants
    and the libraries); the settings and the network's shape come from the fit's start, at the first
    save. Nothing is written when the session is made.

    A kept fit resumed in a later process has a session of its own, made by ``reopen``: it holds the
    record's lock from the start, saves into the same record, and, until it has saved a pass, leaves
    the record as it found it when the training stops (but for a failed save's note).
    """

    def __init__(self, store, *, identity: dict, region: "dict | None", posterior_name: str, note: str, cfg,
                 config: dict, parents: dict, fingerprints: dict):
        self.store = store
        self.identity = _plain(identity)
        self.region = region
        self.posterior_name = posterior_name
        self.note = note
        self.cfg = cfg
        self.config = dict(config)
        self.parents = dict(parents)
        self.fingerprints = dict(fingerprints)
        self.id: "str | None" = None
        self.path: "Path | None" = None
        self.passes = 0                             # the last pass whose save went through whole
        self.processes = 1
        self.resumed = False
        self.log_before = ""                        # the earlier processes' log, for a resumed record
        self._wall_before = 0.0                     # the earlier processes' time in the fit
        self._started: "float | None" = None        # when this process's fit began
        self._lock: "FitLock | None" = None
        self._full: "dict | None" = None            # the whole identity, once the fit has begun
        self._kept = False                          # a manifest is on disk
        self._progress: "dict | None" = None
        self._stopped: "dict | None" = None
        self._unresumable: "str | None" = None
        self._save_error: "str | None" = None
        self._complete = False
        # A resumed record's: the start its fit was kept with and the pass it resumes from, handed to
        # the trainer; whether this process has saved a pass yet; and whether the record's newest state
        # could not be read, so that its first save moves that file out of the way first.
        self.resume_pair: "tuple[dict, dict] | None" = None
        self._saved_here = False
        self._newest_unreadable = False

    @classmethod
    def reopen(cls, store, kept: "KeptFit", *, cfg) -> "FitSession":
        """The session of the kept fit ``kept``, to be resumed in this process, with its lock taken.

        Refused when another handle holds the lock: a running process is training it. Then its record,
        its header and its newest whole pass are read (``state.pt``, else ``state.prev.pt``, which is
        said with the pass it resumes from). A header saved in another format marks the record as one
        that cannot be resumed, and is refused. A header, or every pass, that cannot be read now is
        refused and leaves the record as it was, to be tried again or deleted: a file another program
        holds for a moment reads as unreadable too. Nothing else is written here. The session carries
        the record's own identity, the earlier processes' time and log, and the start the fit was kept
        with (``resume_pair``); its first save in this process clears the stop an earlier process
        recorded. Every way out of here but the session handed back lets the lock go.
        """
        lock = FitLock(kept.path)
        if not lock.acquire():
            raise Refusal(f"The kept training {kept.id} is being trained by a running process.",
                          field="resume_fit")
        try:
            m = store.get("fit", kept.id)
            body = dict(m.body)
            session = cls(store, identity={}, region=body.get("region"),
                          posterior_name=body.get("posterior_name") or "", note=m.note, cfg=cfg,
                          config=dict(m.config), parents=dict(m.parents), fingerprints=dict(m.fingerprints))
            session.id, session.path, session._lock = m.id, Path(kept.path), lock
            session._kept = session.resumed = True
            session._progress, session._stopped = body.get("progress"), body.get("stopped")
            session._save_error, session._complete = body.get("save_error"), bool(body.get("complete"))
            try:
                header = read_header(session.path)
            except Exception:                       # noqa: BLE001 -- torn, truncated, held, or another version's
                header = None
            if isinstance(header, dict) and header.get("format") != FIT_FORMAT:
                # Another format is the record's own: no later attempt reads it, so it is marked.
                why = f"it was saved in format {header.get('format')}; this PRISM reads format {FIT_FORMAT}"
                session.mark_unresumable(why)
                raise Refusal(f"The kept training {kept.id} cannot be resumed: {why}.", field="resume_fit")
            # A header or a state that cannot be read now may read at the next attempt (on Windows a file
            # another program holds without sharing, a virus scanner say, reads as unreadable), so the
            # record is left exactly as it was: marking it would end a days-long training for good.
            state = None
            if not isinstance(header, dict) or not isinstance(header.get("identity"), dict) \
                    or not isinstance(header.get("start"), dict):
                unread = "its header cannot be read"
            else:
                state = load_state(session.path)
                unread = "no saved pass of it can be read" if state is None else None
            if unread is not None:
                raise Refusal(f"The kept training {kept.id} cannot be resumed now: {unread}; nothing of it was "
                              f"changed, and it can be tried again or deleted.", field="resume_fit")
            fell_back = state.pop("_state_file") == STATE_PREV
            session.identity = dict(header["identity"])
            session._full = {part: session.identity.get(part) for part in IDENTITY_PARTS}
            session.passes = int(state["passes"])
            session._wall_before = float(state.get("wall_seconds") or 0.0)
            session.processes = int(state.get("processes") or 1) + 1
            session.log_before = _read_log(session.path)
            session.resume_pair = (dict(header["start"]), state)
            # A newest state that is there and cannot be read is moved aside at the first save, not
            # here: a resume refused once the rows are read must leave every file as it was.
            session._newest_unreadable = fell_back and (session.path / STATE).exists()
            if fell_back:
                log.warning(f"[fit] the newest saved pass of the kept training {m.id} could not be read; it "
                            f"resumes from the pass before it, pass {session.passes}.")
            progress = _progress(state, 0.0, 0)
            best, since = progress["best_validation_loss"], progress["passes_since_best"]
            log.info(f"[fit] resuming the kept training {m.id} after pass {session.passes} ("
                     + (f"best validation loss {best:.4g}, {since} passes since the best" if best is not None
                        else "no finite validation loss yet")
                     + "); its simulation set is read back first.")
            return session
        except BaseException:
            lock.release()
            raise

    def mark_unresumable(self, reason: str) -> None:
        """Mark the kept fit as one that cannot be resumed, saying why, and let its lock go. Only the
        record's ``unresumable`` changes: its header, its two states and its log stay as they were. A
        record that cannot be written is warned about; never raises an ``Exception``."""
        self._unresumable = reason
        try:
            from core.artifacts.store import write_fit_manifest
            m = self.store.get("fit", self.id)
            write_fit_manifest(self.path, body={**m.body, "unresumable": reason}, cfg=self.cfg, note=m.note,
                               config=dict(m.config), parents=dict(m.parents), fingerprints=dict(m.fingerprints),
                               hw=getattr(self.cfg, "hw", None))
        except Exception as e:                      # noqa: BLE001 -- the refusal that follows comes first
            log.warning(f"[fit] could not mark the kept training {self.id} as one that cannot be resumed "
                        f"({type(e).__name__}: {e}).")
        finally:
            self._release()

    # ── the files ──
    def _begin(self, start: dict) -> None:
        """Everything a new record needs before its first state: its folder, its lock and its header.

        The folder, once made, is kept with its lock until a save goes through whole (it is then the
        record) or the session ends without one (it is then removed): a first save that fails is tried
        again in the same folder at the next pass, so a disk full from the first pass on leaves one folder
        at most, never one per pass. Only a folder that could not be made -- another process's record of
        the same second -- makes the next pass take another name."""
        if self.path is None:
            path = self.store.new_fit_dir()
            path.mkdir(parents=True, exist_ok=False)
            self.path = path
        if self._lock is None:
            lock = FitLock(self.path)
            if not lock.acquire():
                raise OSError(f"the lock of the new kept training {self.path} is held by another handle")
            self._lock = lock
        full = {**self.identity, "settings": _plain(start["settings"]), "network": start["network"]}
        self._full = {part: full[part] for part in IDENTITY_PARTS}
        write_header(self.path, {"format": FIT_FORMAT, "identity": self._full,
                                 "start": {k: v for k, v in start.items() if k != "started"}})

    def _write_log(self) -> None:
        from core.artifacts.store import LOG_FILE
        run_log = runs.current_run_log()
        if run_log is not None:
            (self.path / LOG_FILE).write_text(self.log_before + run_log.text(), encoding="utf-8", newline="\n")

    def _write_manifest(self, **body) -> None:
        from core.artifacts.store import write_fit_manifest
        record = {"posterior_name": self.posterior_name, "identity": self._full, "region": self.region,
                  "progress": self._progress, "stopped": self._stopped, "unresumable": self._unresumable,
                  "save_error": self._save_error, "complete": self._complete, **body}
        write_fit_manifest(self.path, body=record, cfg=self.cfg, note=self.note, config=self.config,
                           parents=self.parents, fingerprints=self.fingerprints,
                           hw=getattr(self.cfg, "hw", None))

    def _set_aside_newest(self) -> None:
        """Move the newest state that could not be read out of the save's way, under a name no reader
        reads. Left where it was, the save's rotation would rename it over the last good pass, and a save
        torn between its two renames would leave only the unreadable one."""
        newest = self.path / STATE
        if newest.exists():
            os.replace(newest, self.path / _STATE_UNREADABLE)
        self._newest_unreadable = False

    def _release(self) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def _discard(self) -> None:
        """The session ends with no pass kept: the lock is let go and the folder a first save made is
        removed, so that nothing is left behind."""
        self._release()
        path, self.path, self._full = self.path, None, None
        if path is not None:
            try:
                shutil.rmtree(path)
            except OSError as e:
                log.warning(f"[fit] could not remove the folder {path}, which holds no kept training "
                            f"({type(e).__name__}: {e}); the store lists it as a leftover to clear.")

    # ── the lifecycle ──
    def save_pass(self, take) -> None:
        """Keep the pass that just ended. One deferred-cancel section, inside which nothing is said; a
        failure inside it is warned once it closes, and the fit goes on."""
        failure = None
        with runs.cancel_deferred():
            try:
                snapshot, start = take()
                self._started = start["started"]
                wall = self._wall_before + (time.time() - start["started"])
                if not self._kept:
                    self._begin(start)
                if self._newest_unreadable:
                    self._set_aside_newest()
                save_state(self.path, {**snapshot, "wall_seconds": wall, "processes": self.processes})
                progress = _progress(snapshot, wall, self.processes)
                self._write_log()
                # The stop an earlier process recorded no longer describes the record once this one has
                # saved: a resumed training killed afterwards reads as stopped unexpectedly.
                self._write_manifest(progress=progress, stopped=None, save_error=None)
                self._progress, self._stopped, self._save_error = progress, None, None
                self.passes, self._kept, self._saved_here = progress["passes"], True, True
                self.id = self.path.name.rpartition("__")[2]
            except Exception as e:                  # noqa: BLE001 -- a failed save never stops the fit
                failure = e
        if failure is None:
            log.info(f"[fit] saved after pass {self.passes}.")
            return
        what = f"{type(failure).__name__}: {failure}"
        if self._kept:
            # Noted before the warning is said: under the window a Cancel pressed meanwhile is taken at
            # that warning, and the record the stop then writes carries the failure too.
            self._save_error = f"the last save failed ({what}); the kept training stands at pass {self.passes}"
        log.warning(f"[fit] the last save failed: {what}; "
                    + (f"the kept training stands at pass {self.passes}." if self._kept else "nothing is kept yet."))
        if self._kept:
            try:
                self._write_manifest()
            except Exception:                       # noqa: BLE001 -- the warning above already said it
                pass

    def stopped(self, exc: BaseException) -> None:
        """Record that the training stopped with ``exc`` and let its lock go. A Ctrl-C or the window's
        Cancel (an exception that is not an ``Exception``) is a stop on purpose; anything else stopped it
        unexpectedly, and is recorded as its reason. Never raises."""
        try:
            if not self._kept:
                self._discard()
                log.info("[fit] the training stopped before its first pass was saved; nothing is kept.")
                return
            if not self._saved_here:
                # A resumed record this process has saved no pass into is left as it was found (a failed
                # save's note aside): the stop an earlier process recorded still describes it.
                log.info(f"[fit] the training stopped before it saved a pass in this process; it is kept as "
                         f"fit {self.id}, as it was after pass {self.passes}.")
                return
            on_purpose = not isinstance(exc, Exception)
            reason = _reason(exc)
            self._stopped = {"on_purpose": on_purpose, "at": _utc_now(), "reason": reason}
            # Said first, so that the record's own log says how its training stopped.
            log.info(f"[fit] the training stopped after pass {self.passes} and is kept as fit {self.id} "
                     f"({'stopped on purpose' if on_purpose else 'stopped by ' + reason}).")
            for what, write in (("log", self._write_log), ("record", self._write_manifest)):
                try:
                    write()
                except Exception as e:              # noqa: BLE001 -- the stop that is propagating comes first
                    log.warning(f"[fit] could not write the kept training's {what} as it stopped "
                                f"({type(e).__name__}: {e}).")
        except Exception:                           # noqa: BLE001 -- the stop that is propagating comes first
            pass
        finally:
            self._release()

    def finished(self) -> None:
        """The fit has ended: the record says so until the posterior is written. Never raises."""
        self._complete = True
        if not self._kept:
            return
        try:
            self._write_manifest()
        except Exception as e:                      # noqa: BLE001 -- the posterior is still to be written
            log.warning(f"[fit] could not mark the kept training {self.id} finished ({type(e).__name__}: {e}).")

    def done(self, posterior_label: str) -> None:
        """The training's posterior is written: the record is marked so, its lock let go and the record
        removed. A record that cannot be removed is warned about and left, marked; never raises."""
        if not self._kept:
            self._discard()
            return
        self._unresumable = f"its posterior {posterior_label} is written"
        try:
            self._write_manifest()
        except Exception:                           # noqa: BLE001 -- the removal below is what matters
            pass
        finally:
            self._release()
        try:
            self.store.delete("fit", self.id)
        except Exception as e:                      # noqa: BLE001 -- the posterior is written
            log.warning(f"[fit] the kept training {self.id} could not be removed after its posterior was "
                        f"written ({e}); it can be deleted.")

    def wall_seconds(self) -> float:
        """The time the fit has run, in every process that trained it, up to now."""
        if self._started is None:
            return self._wall_before
        return self._wall_before + (time.time() - self._started)


# ── the listing ──────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KeptFit:
    """A kept fit as the listing reads it from its record: what it is for, how far it went, what it was
    trained with, and whether it can be resumed now (and, when it cannot, why)."""
    id: str
    path: Path
    created: str
    posterior_name: str
    note: str
    state: str
    passes: "int | None"
    best_validation_loss: "float | None"
    passes_since_best: "int | None"
    saved_at: "str | None"
    parents: dict
    inputs: dict
    identity: dict
    region: "dict | None"
    config: dict
    can_resume: bool
    why_not: "str | None"


def _why_not(store, state: str, body: dict, parents: dict, identity: dict, now: dict) -> "str | None":
    """Why the kept fit cannot be resumed now, or None when it can: the first that holds of a running
    training, a record marked so, its simulation set gone or not whole, and the code, the constants or
    the libraries changed since it was kept."""
    from core.SBI import training_checkpoint
    if state == "running":
        return "a running process is training it"
    if body.get("unresumable"):
        return str(body["unresumable"])
    name = parents.get("simulation")
    committed = (training_checkpoint.peek(store.kind_dir("simulation") / name)
                 if isinstance(name, str) and name else None)
    if committed is None:
        return f"its simulation set {name} is not in the store any more"
    batches = identity.get("batches")
    if not committed.get("complete") or committed.get("batches_done") != batches:
        return (f"its simulation set {name} holds {committed.get('batches_done')} committed batches, not the "
                f"{batches} it was trained on")
    difference = first_difference(identity, now, parts=("code", "constants", "libraries"))
    if difference is None:
        return None
    if difference.part == "code":
        return f"the code that builds or trains the network has changed since this training was kept ({difference.name})"
    if difference.part == "constants":
        return (f"the configuration constant {difference.name} was {difference.kept} when this training was "
                f"kept, and is {difference.now} now")
    if difference.name == "sbi_trainer_files":
        return "the installed sbi trainer files have changed since this training was kept"
    return f"{difference.name} was {difference.kept} when this training was kept, and is {difference.now} now"


def kept_fits(store) -> "list[KeptFit]":
    """Every kept fit with a record, newest first, with whether it can be resumed now. The code, the
    constants and the libraries are read once per call."""
    now = {"code": loaded_code(), "constants": constants(), "libraries": libraries()}
    out = []
    for row in store.list("fit"):
        if not row.complete:
            continue
        try:
            m = store.get("fit", row.id)
        except Exception:                           # noqa: BLE001 -- removed since it was listed
            continue
        body = m.body
        # A body below its key set is not checked by the store, so a hand-edited part of another shape
        # reads as empty here rather than taking the whole listing down: such a record is listed, as one
        # that cannot be resumed (an empty identity names no rows and no batches).
        identity = body.get("identity") if isinstance(body.get("identity"), dict) else {}
        progress = body.get("progress") if isinstance(body.get("progress"), dict) else {}
        state = row.fit_state or state_word(row.path, body)
        why = _why_not(store, state, body, dict(m.parents), identity, now)
        out.append(KeptFit(id=m.id, path=row.path, created=m.created, posterior_name=body.get("posterior_name") or "",
                           note=m.note, state=state, passes=progress.get("passes"),
                           best_validation_loss=progress.get("best_validation_loss"),
                           passes_since_best=progress.get("passes_since_best"), saved_at=progress.get("saved_at"),
                           parents=dict(m.parents), inputs=dict(m.inputs), identity=identity,
                           region=body.get("region"), config=dict(m.config), can_resume=why is None, why_not=why))
    return out


# What a new training compares with a kept fit to name it as one it could resume instead: everything
# but the network's shape, which is known only once the fit has begun.
_MATCHED_PARTS = ("simulation", "batches", "settings", "code", "constants", "libraries")


def matching_kept_fits(store, identity: dict) -> "list[KeptFit]":
    """The kept fits that can be resumed and were trained with ``identity``'s rows, batches, settings,
    code, constants and libraries, newest first."""
    now = _plain({part: identity.get(part) for part in _MATCHED_PARTS})
    return [kept for kept in kept_fits(store)
            if kept.can_resume and first_difference(kept.identity, now, parts=_MATCHED_PARTS) is None]


def kept_fit(store, ref: str) -> KeptFit:
    """The kept fit whose id is ``ref``, read as the listing reads it, with whether it can be resumed
    now and, when it cannot, why. Refused, naming the kept training to resume, when the store keeps
    none with that id."""
    ref = str(ref).strip()
    for kept in kept_fits(store):
        if kept.id == ref:
            return kept
    raise Refusal(f"There is no kept training with the id {ref!r} in the store.", field="resume_fit")
