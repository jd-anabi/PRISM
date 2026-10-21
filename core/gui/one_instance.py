"""One PRISM at a time: the lock beside the settings file, the local server a second start knocks on, and the
choice among the cases a start can fall in.

The first PRISM takes the lock, ``PRISM.lock`` in the folder of its settings file, and listens on a local server
whose name is made from that file's path: one for each settings file, so one for each person, and one for each
temporary settings file in the suites. A second start finds the lock held and knocks: it connects to that server
and ends, having built no window. The open PRISM brings its window to the front when the connection arrives. What
is written on the connection is not read: a word written while the open PRISM is busy arrives late, or empty, while
the connection itself is taken at once, and heard as soon as the open PRISM's event loop runs again.

The lock has no stale time and is tried once, without waiting. A lock left by a PRISM that died is taken at once all
the same: the toolkit sees that the process the lock names is gone. When the lock is held and nothing takes the
connection, it is removed and taken; when it cannot be removed, its holder lives and does not answer, and this start
gives up. A lock that cannot be made at all holds no start back: PRISM then starts without it, and says so in a
warning.

The command-line tool takes no lock and does not see this one.
"""
import hashlib
import logging
import os
import sys
import time

from PySide6.QtCore import QCoreApplication, QLockFile, QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger(__name__)

LOCK_FILE = "PRISM.lock"          # beside the settings file
KNOCK = b"front"                  # the one word a second start writes; the open PRISM acts on the connection alone
CONNECT_MS = 1000                 # how long one knock waits for its connection, at most

_current = None                   # the claim this process holds, while it holds one


def server_name(settings_path: str) -> str:
    """The name of the local server that listens for a second start of the PRISM whose settings file is at
    ``settings_path``: one name for each file, the same for the same path written another way (with its letters in
    another case, with the other separator, with a "." or ".." among its folders), and one every system takes for a
    local server."""
    key = os.path.normcase(os.path.abspath(settings_path))
    return "PRISM-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def start_case(*, locked: bool, connected: bool, removed: bool | None) -> str:
    """The case a start of PRISM falls in, from what it found: ``"start"`` when it took the lock; ``"hand_over"``
    when the lock was held and the open PRISM took its connection; ``"start_after_removal"`` when nothing took the
    connection and the lock was removed and taken; ``"give_up"`` when it could not be removed. ``removed`` is None
    while no removal was tried."""
    if locked:
        return "start"
    if connected:
        return "hand_over"
    if removed:
        return "start_after_removal"
    return "give_up"


def allow_foreground(pid: int) -> None:
    """Let the process ``pid`` bring a window of its own to the front, which the system refuses a process the person
    did not just use: a second start, which the person did just use, hands that right to the open PRISM before it
    knocks. On Windows only; elsewhere nothing is done."""
    if sys.platform == "win32":
        import ctypes
        ctypes.WinDLL("user32").AllowSetForegroundWindow(int(pid))


class Claim(QObject):
    """What a start claimed: its case (``start_case``) and, when it took the lock, the lock and the server that
    listens for a second start, a child of the application. ``knocked`` is said each time the event loop takes the
    connections a second start made; nothing written on them is read."""

    knocked = Signal()

    def __init__(self, case: str, lock: QLockFile | None = None, name: str | None = None) -> None:
        super().__init__(QCoreApplication.instance() if lock is not None else None)
        self.case = case
        self._lock = lock
        self._server = None
        if lock is not None and name is not None:
            server = QLocalServer(self)
            server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)      # for this person alone
            # Where the system keeps a file for a local server, a PRISM that died left it behind; this start holds
            # the lock, so no other PRISM listens under the name. On Windows nothing is kept and nothing is done.
            QLocalServer.removeServer(name)
            if server.listen(name):
                server.newConnection.connect(self._knocked_on)
                self._server = server
            else:
                log.warning("PRISM cannot hear a second start of itself, so a second start of PRISM is turned away "
                            "while this one is open: the local server %s did not listen (%s).", name,
                            server.errorString())
                server.setParent(None)

    def _knocked_on(self) -> None:
        """A second start connected: its connections are let go unread, and ``knocked`` is said."""
        while self._server is not None and self._server.hasPendingConnections():
            connection = self._server.nextPendingConnection()
            connection.abort()
            connection.deleteLater()
        self.knocked.emit()

    def release(self) -> None:
        """Let go of the lock and stop listening; nothing when this claim holds neither."""
        global _current
        if self._server is not None:
            self._server.close()
            self._server.setParent(None)
            self._server = None
        if self._lock is not None:
            self._lock.unlock()
            self._lock = None
        self.setParent(None)
        if _current is self:
            _current = None


def _knock(name: str, tries: int, pause_ms: int) -> bool:
    """Connect to the server ``name`` up to ``tries`` times, ``pause_ms`` milliseconds apart, and on the first
    connection write the knock's word and let go. Whether a connection was made."""
    for attempt in range(max(1, tries)):
        if attempt:
            time.sleep(pause_ms / 1000)
        socket = QLocalSocket()
        socket.connectToServer(name)
        if socket.waitForConnected(CONNECT_MS):
            socket.write(KNOCK)
            socket.flush()
            socket.disconnectFromServer()
            return True
    return False


def _without_the_lock(lock_path: str, why: str) -> Claim:
    """A start that cannot make the lock: it starts all the same, holding no lock and hearing no second start."""
    log.warning("The lock %s could not be made (%s); PRISM starts without it, so a second start of PRISM opens a "
                "second PRISM.", lock_path, why)
    return Claim("start")


def claim(settings_path: str, *, tries: int = 3, pause_ms: int = 1000) -> Claim:
    """Claim the one PRISM of the settings file at ``settings_path``, as a start of PRISM does before it builds any
    window, and hand back the claim with the case the start falls in (``start_case``).

    The folder of the settings file is made first when it does not exist yet, as on a computer where PRISM has never
    run. A free lock is taken at once, and the server listens. A held lock names its holder's process: this start
    first lets that process bring its window to the front (``allow_foreground``), then knocks, up to ``tries``
    times, ``pause_ms`` milliseconds apart. When nothing takes the connection, the lock is removed and taken, and the
    server listens; a lock that cannot be removed has a living holder that does not answer, and this start gives up.
    A lock that cannot be made at all -- its folder cannot be made or written, or it names no holder and cannot be
    removed -- holds no start back: the start goes on without it, and a warning says so."""
    global _current
    path = os.path.abspath(settings_path)
    lock_path = os.path.join(os.path.dirname(path), LOCK_FILE)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        pass                                         # the lock's own attempt says what is wrong
    lock = QLockFile(lock_path)
    lock.setStaleLockTime(0)
    locked = lock.tryLock(0)
    if not locked and lock.error() != QLockFile.LockError.LockFailedError:
        return _without_the_lock(lock_path, lock.error().name)
    name = server_name(path)
    connected, removed = False, None
    if not locked:
        holder = int(lock.getLockInfo()[0] or 0)
        if holder > 0:
            allow_foreground(holder)                 # before the knock: the open PRISM acts on the connection
        connected = _knock(name, tries, pause_ms)
        if not connected:
            removed = lock.removeStaleLockFile() and lock.tryLock(0)
            if not removed and holder <= 0:
                return _without_the_lock(lock_path, "it names no process and cannot be removed")
    case = start_case(locked=locked, connected=connected, removed=removed)
    if case in ("start", "start_after_removal"):
        _current = Claim(case, lock, name)
        return _current
    return Claim(case)


def current() -> Claim | None:
    """The claim this process holds -- the lock it took and the server that listens -- or None when it holds none."""
    return _current


def release() -> None:
    """Let go of the claim this process holds, when it holds one: the one call a test that builds the whole program
    makes as it ends."""
    if _current is not None:
        _current.release()
