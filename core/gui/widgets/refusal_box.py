"""The one yellow "Check your inputs" box, shared by every surface that shows a refusal.

Three callers, and the last two are why this is not a method. ``BasePanel._refusal`` shows it for
every run-a-thing panel; the Artifacts screen shows it for a refused note, a refused delete and a
refused lineage report, and the model builder for a refused box. Those two screens are plain
``QWidget``s on purpose -- a ``BasePanel`` enrols in ``BasePanel._instances``, so a run in any panel
would grey out their controls, and you could not read a log in the browser while training. So the
helper takes a PARENT WIDGET and assumes nothing about it.

What the box says is fixed here and nowhere else: the text is the core's own neutral sentence (it
names the setting, the rule, what was given and the default, never a box, tab or flag) and the
informative line is this front end's "where to fix it", looked up by the refusal's field key in
core/gui/fields.py -- the one place a control is named, so a renamed control is renamed once. No
Details: nothing here is a crash, and a traceback would only say so louder. A bug keeps the red box
(``BasePanel._on_error``).

It does NOT record the sentence anywhere. A panel has a log pane and the two screens have a status
line, so each caller keeps its own record; this helper only ever shows the box.
"""
from PySide6.QtWidgets import QMessageBox

from .. import fields as gui_fields


def show_refusal(parent, exc) -> None:
    """Show ``exc`` (a ``core.refusals.Refusal``, or any subclass such as ``StoreError``) as the
    yellow warning box, modally, parented to ``parent``.

    An INSTANCE dialog built and shown with ``.exec()``, never ``QMessageBox.warning(...)``:
    ``tests/conftest.py::_no_modal_dialogs`` patches the instance method only, and the C++ statics
    escape it -- a static call STALLS the offscreen suite instead of failing.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)                  # user input, not a crash
    box.setWindowTitle("Check your inputs")
    box.setText(exc.message)
    fix = gui_fields.fix_sentence(exc.field)
    if fix:
        box.setInformativeText(fix)
    box.setStandardButtons(QMessageBox.Ok)
    box.setDefaultButton(QMessageBox.Ok)              # one button, so the safe default is it
    box.exec()
