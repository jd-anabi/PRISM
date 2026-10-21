"""Build the QApplication + MainWindow, one PRISM at a time, and install a last-resort excepthook that
surfaces unhandled errors in a dialog instead of silently killing the app."""
import logging
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from core import config, logging_root, registry

from . import app_icon, design, fonts, mpl_theme, one_instance, settings, style, theming
from .main_window import MainWindow
from .screens.shell import NO_MESSAGE, _as_sentence
from .widgets.ask import Ask, ask

# What a start of PRISM that builds no window says as it ends, on the error stream when the process has one:
# started from a console, it is the only sign the person gets.
SAID = {"hand_over": "PRISM is already open: its window was brought to the front.",
        "give_up": "PRISM is already running but did not answer; this start ends."}

# The most a window's frame -- its title bar and its borders -- takes of a screen's height, which a question's own
# height leaves out.
FRAME = 40


def _room(parent) -> int:
    """The height a question may take, its frame left out, on the screen it is asked on: the window's, or the main
    screen when there is no window; the taskbar's room is not the screen's to give."""
    screen = parent.screen() if parent is not None else QApplication.primaryScreen()
    return screen.availableGeometry().height() - FRAME


def _fitted(sentence: str, room: int) -> str:
    """``sentence`` as the last-resort question can show it in ``room``, the height it may take: whole when the
    question then stands whole; else cut after its last word that leaves the question whole, an ellipsis put in
    place of the rest, which the full error holds. Measured on the question itself, built and never shown."""
    question = Ask(None, "Error", sentence, "OK", marker="err", detail=sentence)
    try:
        over = question.sizeHint().height() - room
        if over <= 0:
            return sentence
        lines = question.sentence
        width = lines.minimumWidth()
        allowed = lines.heightForWidth(width) - over
        lines.setMinimumHeight(0)           # the question holds it at the whole message's height
        words = sentence.split(" ")
        kept, most = 0, len(words) - 1
        while kept < most:                  # the most words that stand in what is allowed, the ellipsis after them
            middle = (kept + most + 1) // 2
            lines.set_text(" ".join(words[:middle]) + "\u2026")
            if lines.heightForWidth(width) <= allowed:
                kept = middle
            else:
                most = middle - 1
        return " ".join(words[:kept]) + "\u2026"
    finally:
        question.deleteLater()


def _install_excepthook(parent_getter):
    """Route any otherwise-unhandled exception to the shared question, telling: an error, whose one sentence is
    the error's message, written as a sentence as a part's error page writes it, with one answer; the error's kind
    and its trace are behind "Show the full error", and nowhere else. A message too long for the question to stand
    whole on its screen is cut, with an ellipsis (``_fitted``); the full error holds it whole. Over the window once
    there is one (``parent_getter``), and with no window under it before. Each error is still written to the console
    as before.

    Installed BEFORE MainWindow() is built, so that it stands for everything after. A part that cannot
    be built (e.g. a bad cell file breaking a panel's prefill) does not reach it: the window builds each
    part inside a guard and shows that part's error page in its place. What reaches it is an error no
    other code catches -- a slot's, an event handler's -- and it shows a dialog instead of a bare
    console traceback. One question at a time: an error raised while one is open, by a timer or a slot that
    runs in the question's own event loop, is written to the console alone, so that questions cannot pile up.
    """
    asking = []                              # holds a mark while a question of this hook is open

    def _excepthook(exc_type, exc, tb):
        try:
            if not asking:
                asking.append(True)
                try:
                    parent = parent_getter()
                    ask(parent, "Error", _fitted(_as_sentence(str(exc)) or NO_MESSAGE, _room(parent)), "OK",
                        marker="err", detail="".join(traceback.format_exception(exc_type, exc, tb)))
                finally:
                    asking.clear()
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook


def _library_record_sink(record) -> None:
    """One record no handler below the root has emitted, on the console, as ``logging_root.render``
    writes it: a library's named by its logger, a ``core`` record -- which reaches here only
    when no ``core`` handler is attached, i.e. between runs -- in the tool's own console shape.

    The stream is resolved AT EMIT TIME, never at construction, and that is the fix rather than the
    bug: what broke was ``logging.basicConfig``'s handler binding ``sys.stderr`` at construction and
    then writing into a stopped pump for the rest of the process. core/tool/logging_console.py states
    the same rule for the tool's two handlers.

    Resolving late also solves three problems at once and adds no plumbing. DURING A RUN these two
    names ARE the run's ``_SignalStream``s (core/gui/streams.py), which are thread-safe by
    construction and feed the pump -- so a library record arriving on the WORKER thread never touches
    a widget, which a ``log_pane.append_line`` from the handler would have done -- and outside a run
    they are the real console.

    The level split exists because ``_SignalStream`` on err is hard-wired to the pane's ``warning``
    level: routing an information record through stderr would put a warning triangle on it.
    """
    stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
    stream.write(logging_root.render(record) + "\n")
    stream.flush()


def _say(case: str) -> None:
    """Say what a start that builds no window did, in one line on the error stream; nothing when the process has
    no such stream, as one started with no console, or when the stream cannot be written."""
    stream = sys.stderr
    if stream is None:
        return
    try:
        stream.write(SAID[case] + "\n")
        stream.flush()
    except (OSError, ValueError):            # a stream that was closed, or whose reader is gone
        pass


def build_app(argv=None):
    # THE root-logger handler, before anything else in this process can log. A handler on the root
    # FROM START-UP is what stops a library's logging.warning from calling basicConfig and
    # installing a second one -- after which every ``core`` record would be emitted twice, the
    # second copy into whatever stream that handler captured at construction. Never removed: the
    # window owns the process for as long as it lives.
    logging_root.install(_library_record_sink)

    # Quiet the per-time-segment bar: it wraps segs in {1,2,3} and nests a whole level under the
    # training-data bar for nothing. The solver's per-step bar stays ON -- it feeds the Solver
    # Performance meter. The GUI is the only caller that flips this; the tool keeps both bars.
    config.QUIET_SEGMENT_BAR = True

    # Register saved user-defined models BEFORE MainWindow: the model combos read VALID_MODELS at
    # construction. Per-file failures land in registry.load_errors (listed under Your own models).
    registry.load_user_models()

    # The artifact store, resolved once: a root that cannot be created fails HERE, at launch, not at
    # the first write hours into a run.
    from core.artifacts import ArtifactStore, set_default_store
    root = config.artifacts_root()
    root.mkdir(parents=True, exist_ok=True)
    set_default_store(ArtifactStore(root))

    # Before the QApplication, and so before any window: this is the Windows shell's grouping key, and
    # setting it late leaves the taskbar button showing the interpreter's icon instead of ours.
    app_icon.set_windows_app_user_model_id()

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setOrganizationName(settings.ORG)   # both are needed for a stable QSettings store on Windows
    app.setApplicationName(settings.APP)
    # One PRISM at a time, settled as soon as the settings file is known and before the style, the fonts or any
    # window: a start that finds PRISM open hands over to it, or gives up when it does not answer, and ends here
    # having built nothing (one_instance.py).
    claimed = one_instance.claim(settings.settings().fileName())
    if claimed.case in SAID:
        _say(claimed.case)
        if claimed.case == "give_up":
            raise SystemExit(1)
        return app, None
    # On the APPLICATION, not on the window: every dialog, message box and popped-out figure window
    # inherits it, so there is no per-window wiring to forget. PRISM had no icon at all before this.
    app.setWindowIcon(app_icon.app_icon())
    design.set_text_scale(design.system_text_scale())    # first: every size read after this line is a scaled one
    # The small style over Fusion paints the tick boxes and the radio dots from the look's table, leaves
    # the toolkit's dotted focus mark out and places the focus ring. On the APPLICATION, and before the
    # font and the first sheet: a change of style throws every per-class font away. The follower that
    # shows the ring is built with it and starts to listen at the first change of focus, so the
    # window's construction does not pass through it. tests/_fixtures.whole_program takes both off.
    style.install(app)
    qs = settings.settings()
    fonts.load_app_font(app)                             # the lettering face, before any widget
    appearance = theming.Appearance(app)
    # The first apply: the palette, the first style sheet and, after both, the class fonts, which the style and
    # the sheet each empty (theming.Appearance sets them after every sheet it sets).
    appearance.set_mode(settings.get_appearance(qs))
    mpl_theme.install(appearance)   # the light figure settings, written once, whatever the appearance

    window_ref = {}
    _install_excepthook(lambda: window_ref.get("w"))
    try:
        window = MainWindow(appearance=appearance)
    except Exception as e:                   # noqa: BLE001 -- a construction failure must still show
        box = QMessageBox(None)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("PRISM could not start")
        box.setText(f"The application failed to start:\n{e}")
        box.setDetailedText(traceback.format_exc())
        box.exec()
        # We already showed the dialog and logged the traceback. Exit via SystemExit so the installed
        # sys.excepthook does NOT fire and pop a SECOND "Unhandled error" dialog for the same failure.
        traceback.print_exc()
        raise SystemExit(1) from e
    window_ref["w"] = window
    # A second start's knock brings this window to the front. One that came while the window was being built is
    # heard when the event loop first runs, once the window is shown.
    claimed.knocked.connect(window.come_to_front)
    # The taskbar draws the window CLASS icon whenever its icon query goes unanswered, and Qt's class
    # carries python.exe's generic glyph; PRISM's first show is busy long enough for that to happen
    # every time (app_icon.set_windows_class_icon has the measurements). Before the first show --
    # __main__ shows the window after this returns -- and winId() creates the HWND hidden.
    app_icon.set_windows_class_icon(window)
    return app, window
