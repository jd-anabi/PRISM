"""Your own models: the list of the models you defined, and the builder that makes and edits one.

Two pages in one part. The list names each model with an Edit and a Delete button, and each file that failed to
load with the reason; its button "Open model builder" opens the builder on a blank form. Edit opens the builder
on that model, and the builder's "Back to your models" returns to the list and keeps what the form holds. The
part opens on the list in every launch and shows the page last shown after that; the page is never stored.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
                               QStackedWidget, QVBoxLayout, QWidget)

from core import registry
from core.Helpers import model_store

from ..panels.base_panel import BasePanel
from .model_builder_screen import ModelBuilderScreen

# the keys of the two pages, the list's first: the window refuses any other page by them, before the part is built
PAGES = ("list", "builder")


class ModelsScreen(QWidget):
    """The list of your own models over the builder, one page at a time."""

    def __init__(self, on_saved, parent=None):
        """``on_saved(name)`` is called after a model was saved (its name) or deleted (""), once the list names
        what is now there."""
        super().__init__(parent)
        self._on_saved = on_saved
        self.builder = ModelBuilderScreen(on_saved=self._saved, on_back=lambda: self._turn_to("list"))

        models_box = QGroupBox("User-defined models")
        mv = QVBoxLayout(models_box)
        self._models_list = QVBoxLayout()          # rebuilt by refresh_models()
        mv.addLayout(self._models_list)
        self._open = QPushButton("Open model builder")
        self._open.setProperty("accent", True)
        self._open.clicked.connect(self._open_builder)
        mv.addWidget(self._open)
        self.refresh_models()

        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.addWidget(models_box)
        iv.addStretch(1)
        # A list of many models scrolls, so that the button under it stays in reach in a low window.
        listed = QScrollArea()
        listed.setFrameShape(QFrame.NoFrame)
        listed.setWidgetResizable(True)
        listed.setWidget(inner)

        self._pages = QStackedWidget()
        self._pages.addWidget(listed)
        self._pages.addWidget(self.builder)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._pages)

    def show_page(self, name: str) -> None:
        """Show the list ("list") or the builder ("builder"); KeyError for any other name, before anything
        moves."""
        if name not in PAGES:
            raise KeyError(f"not a page of Your own models: {name!r}")
        self._pages.setCurrentIndex(PAGES.index(name))

    def current_page(self) -> str:
        """The key of the page shown: the list's until another is shown."""
        return PAGES[self._pages.currentIndex()]

    def first_focus(self) -> QWidget:
        """Where the keyboard starts: on the builder its first setting, the model's name; on the list, which has
        no setting, the button that opens the builder, which starts no run."""
        return self.builder.name_edit if self.current_page() == "builder" else self._open

    def _turn_to(self, name: str) -> None:
        """Show a page from a button of this part, and give its first stop the keyboard, as a move does."""
        self.show_page(name)
        self.first_focus().setFocus(Qt.OtherFocusReason)

    def _open_builder(self) -> None:
        """The list's button: the builder on a blank form."""
        self.builder.reset()
        self._turn_to("builder")

    def _saved(self, name: str) -> None:
        """The builder saved ``name``: the list names it, then the window hears it."""
        self.refresh_models()
        self._on_saved(name)

    def _tell(self, title: str, text: str, icon) -> None:
        """One plain box with an Ok button: an INSTANCE dialog shown with .exec().

        Never QMessageBox.warning/information. Those are C++ statics, and the suite's session guard
        patches the instance method only (tests/conftest.py::_no_modal_dialogs): offscreen a static
        spins a nested event loop nothing closes, so a test that reaches one stalls instead of
        failing. Every box of this part goes through here or is built inline like the confirmation
        below.
        """
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _edit_user_model(self, name: str):
        try:
            self.builder.load_existing(name)
        except Exception as e:                       # noqa: BLE001 -- a corrupt file must not crash
            self._tell("Cannot edit model", f"Could not load '{name}':\n{e}", QMessageBox.Warning)
            return
        self._turn_to("builder")

    def _delete_user_model(self, name: str):
        if BasePanel._running:
            self._tell("A task is running",
                       "Wait for the running task to finish before deleting a model.",
                       QMessageBox.Information)
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Delete model")
        box.setText(f"Delete the user-defined model '{name}'?")
        box.setInformativeText("This removes its definition and its generated Bounds/Cells/Units files.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            return
        try:
            model_store.delete_user_model(name)
        except Exception as e:                       # noqa: BLE001
            self._tell("Delete failed", str(e), QMessageBox.Warning)
            return
        registry.unregister(name)
        self.refresh_models()
        # The row whose Delete was pressed is gone with the keyboard it held: the list's first stop takes it, as
        # after a move of this part, rather than whatever the toolkit finds next.
        self.first_focus().setFocus(Qt.OtherFocusReason)
        self._on_saved("")

    def refresh_models(self) -> None:
        """Rebuild the user-model list (one row per registered model + any startup load failures)."""
        while self._models_list.count():
            item = self._models_list.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        names = registry.user_model_names()
        if not names and not registry.load_errors:
            empty = QLabel("No user-defined models yet. Build one to add it to every model list.")
            empty.setProperty("type", "caption")
            self._models_list.addWidget(empty)
        for name in names:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(name))
            h.addStretch(1)
            btn_edit = QPushButton("Edit")
            btn_edit.clicked.connect(lambda _=False, n=name: self._edit_user_model(n))
            btn_del = QPushButton("Delete")
            btn_del.clicked.connect(lambda _=False, n=name: self._delete_user_model(n))
            h.addWidget(btn_edit)
            h.addWidget(btn_del)
            self._models_list.addWidget(row)
        for path, msg in registry.load_errors:
            err = QLabel(f"Failed to load {path.name}: {msg}")
            err.setProperty("type", "caption")
            err.setWordWrap(True)
            self._models_list.addWidget(err)
