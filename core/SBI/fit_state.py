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
"""
import errno
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from core import runs
from core.Helpers.file_manager import atomic_torch_save

if os.name == "nt":
    import msvcrt
else:
    import fcntl

FIT_FORMAT: int = 1
HEADER = "header.pt"
STATE = "state.pt"
STATE_PREV = "state.prev.pt"
LOCK = "running.lock"

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
