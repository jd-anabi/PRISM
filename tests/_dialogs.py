"""What the window asked or told during a test, and how a test answers it: the dialog record, the
file-chooser record and their helpers.

At module level this file imports the standard library and nothing else -- PySide6 inside its
functions only, torch and the program nowhere -- so a suite that may not load torch can read a box
or answer a chooser too, and importing it changes nothing about the order in which a suite first
loads torch and the toolkit. tests/_fixtures.py re-exports every public name, and the lists are the
SAME objects whichever of the two modules a test names. ONE import spelling, ``tests._dialogs``
(pytest.ini puts the repo root on the path; tests/ has no __init__.py): ``from _dialogs import SHOWN``
would import this module a second time under another name, with a second, empty SHOWN that the guard
never touches.

``SHOWN`` holds the boxes and questions themselves, in the order they were opened (a box is a
``QMessageBox``; a question is the window's shared dialog, ``core.gui.widgets.ask.Ask``). Beside it stands
one ``Asked`` per box or question: its words, read by ``record`` at the moment it is OPENED and never
afterwards. A widget whose parent a test has since dropped can no longer be read, and the words a test
wants are the ones the operator would have seen at that moment. A record is words only -- strings,
tuples of strings and a bool; never the box, a button, an exception or a traceback -- so keeping one
keeps nothing of a failed run alive. ``CHOOSER_CALLS`` holds one ``ChooserCall`` per file chooser the
code under test opened. ``NATIVE_BOXES`` holds the words of each plain Windows message the entry point
asked for, which ``native_box`` keeps in its place.

No name here is ever bound a second time: ``forget_asked()`` empties the two lists in place. A
test reads a box or a question through ``asked()`` and ``all_asked()``, starts over with
``forget_asked()``, answers with ``choose()`` and leaves one open a while with ``hold_open()``; it
reads the plain Windows messages it caused with ``native_boxes()``.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

# The words of every plain Windows message the entry point asked for, in order, shown to nobody:
# tests/conftest.py::_no_native_boxes puts native_box in the entry point's own for the session, and
# _every_native_box_is_read empties the list before each test and fails a test that left one unread.
NATIVE_BOXES: list[str] = []


def native_box(text: str) -> None:
    """The entry point's plain Windows message, as the suites have it: its words kept, nothing shown."""
    NATIVE_BOXES.append(text)


def native_boxes() -> list[str]:
    """The words of every plain Windows message asked for since the last reading, in order; read, they are
    forgotten."""
    said = list(NATIVE_BOXES)
    NATIVE_BOXES.clear()
    return said


def check_native_boxes_read() -> None:
    """Fail when a plain Windows message was asked for and its test never read it."""
    if NATIVE_BOXES:
        raise AssertionError(f"the test asked for a plain Windows message it never read: {NATIVE_BOXES}")

# Every QMessageBox or shared question the code under test opened, in order: the dialogs themselves,
# shown to nobody. tests/conftest.py::_no_modal_dialogs replaces the exec of each of the two classes
# for the session with a call of record() that returns 0, and _clear_shown empties both lists before
# each test. Nothing but record() appends to either.
SHOWN: list = []
_ASKED: list = []            # one Asked per entry of SHOWN, read when that box was opened


@dataclass(frozen=True)
class Asked:
    """One box as its words. ``kind`` is "question" (two answers or more), "error", "refusal" or
    "notice"; ``marker`` is "warn", "err", "info" or "" for the icon; ``sentence`` is the line the
    box states and ``beneath`` what stands under it; ``detail`` is the text behind "Show Details".
    ``answers`` are the words on the buttons without their mnemonic marks, the safe one LAST;
    ``default`` is the answer Enter gives, None when the code set none on a box with several;
    ``safe`` is the answer Escape and the frame's close give, None when no single button rejects;
    ``loses_work`` says that one of the answers discards something."""
    kind: str
    title: str
    marker: str
    sentence: str
    beneath: str
    items: tuple[str, ...]
    detail: str
    answers: tuple[str, ...]
    default: str | None
    safe: str | None
    loses_work: bool

    @property
    def text(self) -> str:
        """Everything that stands under the sentence, one part to a line: the items, then
        ``beneath``; an empty part is skipped. NOT the sentence itself."""
        return "\n".join(part for part in (*self.items, self.beneath) if part)


def _words(button) -> str:
    """A button's words as the operator reads them: a doubled mnemonic mark is one "&", a single
    one is not there."""
    return button.text().replace("&&", "\0").replace("&", "").replace("\0", "&")


def _answer_buttons(box) -> list:
    """The buttons that answer the box, in the box's own order. "Show Details..." is an action, not
    an answer."""
    from PySide6.QtWidgets import QMessageBox
    return [b for b in box.buttons() if box.buttonRole(b) != QMessageBox.ActionRole]


def _safe_button(box, buttons):
    """The button that answers as Escape does: the only one; else the single one that rejects; else
    the single one that says no. None when there is no such button, and when there is no button."""
    from PySide6.QtWidgets import QMessageBox
    if len(buttons) == 1:
        return buttons[0]
    for role in (QMessageBox.RejectRole, QMessageBox.NoRole):
        of_role = [b for b in buttons if box.buttonRole(b) == role]
        if len(of_role) == 1:
            return of_role[0]
    return None


def _read(box) -> Asked:
    """The words of a box that was never shown. Such a box has no button yet when its code set none
    (the toolkit adds the OK at showing), so no answer button reads as the one answer "OK"."""
    from PySide6.QtWidgets import QMessageBox
    if not isinstance(box, QMessageBox):
        return _read_question(box)
    buttons = _answer_buttons(box)
    safe = _safe_button(box, buttons)
    if buttons:
        answers = tuple(_words(b) for b in buttons if b is not safe)
        answers += () if safe is None else (_words(safe),)
        safe_words = None if safe is None else _words(safe)
    else:
        answers, safe_words = ("OK",), "OK"
    chosen = box.defaultButton()
    if chosen is not None and any(chosen is b for b in buttons):
        default = _words(chosen)
    else:
        default = answers[0] if len(answers) == 1 else None      # never a guess among several
    icon, title = box.icon(), box.windowTitle()
    if len(answers) >= 2:
        kind = "question"
    elif icon == QMessageBox.Critical:
        kind = "error"
    elif title == "Check your inputs":
        kind = "refusal"
    else:
        kind = "notice"
    marker = {QMessageBox.Warning: "warn", QMessageBox.Critical: "err",
              QMessageBox.Information: "info"}.get(icon, "")
    return Asked(kind=kind, title=title, marker=marker, sentence=box.text(),
                 beneath=box.informativeText(), items=(), detail=box.detailedText(), answers=answers,
                 default=default, safe=safe_words,
                 loses_work=any(box.buttonRole(b) == QMessageBox.DestructiveRole for b in buttons))


def _read_question(box) -> Asked:
    """The words of a shared question, read from the plain words it keeps and from nothing else: no widget
    is asked, so the record is made when the question is opened and outlives it."""
    from core.gui.widgets.ask import Ask
    assert isinstance(box, Ask), f"what was recorded is not a message box or a question: {box!r}"
    said = box.wording
    if said.other is not None:
        kind = "question"
    elif said.marker == "err":
        kind = "error"
    elif said.title == "Check your inputs":
        kind = "refusal"
    else:
        kind = "notice"
    return Asked(kind=kind, title=said.title, marker=said.marker, sentence=said.sentence, beneath=said.beneath,
                 items=said.items, detail=said.detail,
                 answers=(said.safe,) if said.other is None else (said.other, said.safe),
                 default=said.other if said.label is not None else said.safe, safe=said.safe,
                 loses_work=said.loses_work)


def record(box) -> None:
    """Keep a box and its words, read now. The session guard and ``choose`` call this for every box
    that is opened; nothing else appends to ``SHOWN``. The words are read FIRST, so a thing that
    cannot be read is not kept."""
    told = _read(box)
    SHOWN.append(box)
    _ASKED.append(told)


def all_asked() -> list[Asked]:
    """Every box since the test began or since ``forget_asked()``, in order, as a new list. No widget
    is touched here, so it is safe inside a wait and after the box's parent is gone."""
    assert len(_ASKED) == len(SHOWN), (
        "SHOWN was changed by hand, so it no longer matches what was read: add a box with "
        "tests/_dialogs.record and empty the list with forget_asked()")
    return list(_ASKED)


def asked() -> Asked:
    """The last box."""
    shown = all_asked()
    assert shown, "no box was shown: the code under test neither asked nor told anything"
    return shown[-1]


def forget_asked() -> None:
    """Empty the record IN PLACE: the list is shared by name with tests/_fixtures.py."""
    SHOWN.clear()
    _ASKED.clear()


def choose(monkeypatch, *, words: str | None = None, other: bool = False, default: bool = False,
           during: Callable[[], None] | None = None, text: str | None = None) -> None:
    """Answer every box or question from now until the test ends, or until the next ``choose``, by
    CLICKING a button of the dialog itself -- the window reads an answer both from what ``exec`` returns
    and from the clicked button, and a click sets both.

    ``words``: the answer with these words, as ``Asked.answers`` spells them. ``other=True``: on a
    box with exactly two answers, the one that is not the safe one; on a question, its other answer.
    ``default=True``: the default button, as Enter presses it. None of the three: nothing is clicked,
    which is the session guard again. More than one of them is refused here, when ``choose`` is called:
    an exception raised inside the fake would be raised under a clicked slot and never reach the test.
    ``text``: for a question that asks for a line of text, typed into its field before the click (a
    question without one ignores it). It is no answer, so it does not count among the three.

    Every dialog is recorded first, then ``during`` runs (what happens while the question stands
    open), then the line is typed and the click made. A dialog that does not offer the chosen answer, or
    whose answer is greyed, is left unanswered exactly as the guard leaves it -- ``exec`` returns 0 and
    no button is clicked -- which every reader in the window takes as its safe branch. Undone with the
    test's ``monkeypatch``, whose undo puts the session guard back."""
    if (words is not None) + bool(other) + bool(default) > 1:
        raise TypeError("choose() takes at most one of words=, other=True and default=True")

    def answer(box) -> int:
        record(box)
        if during is not None:
            during()
        buttons = _answer_buttons(box)
        target = None
        if words is not None:
            target = next((b for b in buttons if _words(b) == words), None)
        elif other:
            safe = _safe_button(box, buttons)
            if safe is not None and len(buttons) == 2:
                target = next(b for b in buttons if b is not safe)
        elif default:
            chosen = box.defaultButton()
            if chosen is not None and any(chosen is b for b in buttons):
                target = chosen
        if target is None:
            return 0
        target.click()
        return box.result()

    def answer_question(dialog) -> int:
        record(dialog)
        if during is not None:
            during()
        if text is not None and dialog.field is not None:
            dialog.field.setText(text)
        said, target = dialog.wording, None
        if words is not None:
            if said.other is not None and words == said.other:
                target = dialog.other_button
            elif words == said.safe:
                target = dialog.safe_button
        elif other:
            target = dialog.other_button
        elif default:
            target = dialog.safe_button if said.label is None else dialog.other_button
        if target is None or not target.isEnabled():
            return 0
        target.click()
        return dialog.result()

    from PySide6.QtWidgets import QMessageBox
    from core.gui.widgets.ask import Ask
    monkeypatch.setattr(QMessageBox, "exec", answer)
    monkeypatch.setattr(Ask, "exec", answer_question)


def hold_open(monkeypatch, seen: list, *, during: Callable[[], None] | None = None) -> None:
    """Leave every box or question open a while, from now until the test ends or until the next ``choose`` or
    ``hold_open``, as a person reading it leaves it: it is recorded first, shown as ``exec`` shows it, modal to the
    whole program, then ``during`` runs (when one is handed), the event loop turns until what was due by then has been
    done, and whether the box and the window it stands over are still in sight is noted in ``seen`` as a pair. Then
    it is hidden, answered with nothing, as the guard leaves it. Undone with the test's ``monkeypatch``."""
    def held(box) -> int:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QApplication
        from tests._runs import turn_until
        record(box)
        box.setWindowModality(Qt.ApplicationModal)
        box.show()
        if during is not None:
            during()
        for _ in range(2):                        # two rounds of the loop's zero-time timers
            done = []
            QTimer.singleShot(0, lambda: done.append(True))
            turn_until(QApplication.instance(), lambda: bool(done))
        over = box.parentWidget()
        seen.append((box.isVisible(), over is not None and over.window().isVisible()))
        box.hide()
        return 0

    from PySide6.QtWidgets import QMessageBox
    from core.gui.widgets.ask import Ask
    monkeypatch.setattr(QMessageBox, "exec", held)
    monkeypatch.setattr(Ask, "exec", held)


class ChooserCall(NamedTuple):
    """One file chooser the code under test opened: "save" or "open", the title, the start value and
    the filter it was handed, and the path the test answered with. ``answer`` is "" when the test
    answered by cancelling and None when no test answered at all."""
    kind: str
    title: str
    start: str
    filter: str
    answer: str | None


# Every file chooser opened since the test began, in order. tests/conftest.py::_no_file_choosers puts
# chooser_guard in place of QFileDialog.getSaveFileName and getOpenFileName for the session, and
# _every_file_chooser_is_answered empties this list around each test and fails the test at teardown
# when it holds a call nobody answered.
CHOOSER_CALLS: list = []
_CHOOSER_ANSWER: dict = {}       # {"path": str} while a test's answer_chooser is in force


def chooser_guard(kind: str):
    """The stand-in for one of QFileDialog's two file-name statics; ``kind`` is "save" or "open".

    It RECORDS the call in CHOOSER_CALLS and returns ``(path, "")``: the path a test gave with
    answer_chooser, or "" -- what Cancel returns -- when no test answered. It never raises: a chooser
    is opened from a clicked slot, and an exception raised inside a slot does not reach the test that
    clicked. An unanswered chooser is reported by check_choosers_answered at the test's teardown.

    A ``staticmethod``, as the entry it replaces is. The parameter names are Qt's own keywords, and
    whatever follows the filter is accepted and ignored: matplotlib's toolbar in a popped-out figure
    passes a selected filter as well."""
    def stand_in(parent=None, caption="", dir="", filter="", *rest, **more):
        answer = _CHOOSER_ANSWER.get("path")
        CHOOSER_CALLS.append(ChooserCall(kind, str(caption), str(dir), str(filter), answer))
        return (answer or "", "")
    return staticmethod(stand_in)


def answer_chooser(monkeypatch, path: str) -> None:
    """Answer the next file chooser this test opens, and every later one, with ``path``. "" is the
    operator cancelling, and counts as an answer. Undone with the test's ``monkeypatch``; a second
    call replaces the answer."""
    monkeypatch.setitem(_CHOOSER_ANSWER, "path", str(path))


def check_choosers_answered() -> None:
    """Fail when CHOOSER_CALLS holds a chooser no test answered, naming each one's kind and title.
    Raised rather than asserted, so the check survives a run with assertions switched off."""
    unanswered = [c for c in CHOOSER_CALLS if c.answer is None]
    if unanswered:
        raise AssertionError(
            "a file chooser was opened that this test did not answer (it was recorded and returned "
            "as cancelled): " + "; ".join(f"{c.kind} '{c.title}'" for c in unanswered)
            + ". Answer it before the click with tests._fixtures.answer_chooser(monkeypatch, path); "
              "pass \"\" for a cancel.")
