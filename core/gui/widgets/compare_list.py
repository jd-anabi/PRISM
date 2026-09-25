"""The comparison list: what the comparison controls hold instead of a multi-select picker.

``StorePicker`` is single-selection (one combo, one id), and spec §7.1 settles what to do about it:
the comparison controls hold a LIST the picker APPENDS to, and no multi-select widget is built. So a
record is chosen the way every other record in the application is chosen -- in the picker, with its
tooltip and its summary line -- and Add puts that choice on the list.

The list holds ids in ``Qt.UserRole`` and shows whatever the picker showed, which is the same
``<name>`` or ``(unnamed <id>)`` label the rest of the window uses. It is deliberately NOT persisted:
a remembered list would name records a later session may have deleted, and the comparison would
refuse on a choice nobody made this time.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

_ID_ROLE = Qt.UserRole


class CompareList(QWidget):
    """A list of artifact ids, filled from a single-selection picker."""

    def __init__(self, picker, parent=None):
        super().__init__(parent)
        self._picker = picker
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        self.btn_add = QPushButton("Add selected")
        self.btn_remove = QPushButton("Remove")
        self.btn_clear = QPushButton("Clear")
        self.btn_add.clicked.connect(self.add_selected)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.list.clear)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        for b in (self.btn_add, self.btn_remove, self.btn_clear):
            buttons.addWidget(b)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.list)
        column.addLayout(buttons)

    def add_selected(self) -> bool:
        """Append the picker's current selection. False when there is nothing selected, or when the
        list already holds it: the same record twice is one curve drawn twice, not two records."""
        id_, _is_new = self._picker.selected()
        if id_ is None or str(id_) in self.ids():
            return False
        item = QListWidgetItem(self._picker.selection_text())
        item.setData(_ID_ROLE, str(id_))
        self.list.addItem(item)
        return True

    def remove_selected(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def ids(self) -> list:
        """The ids, in the order the legend will read them."""
        return [str(self.list.item(i).data(_ID_ROLE)) for i in range(self.list.count())]
