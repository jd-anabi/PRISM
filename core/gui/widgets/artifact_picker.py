"""A combo box populated from ``file_manager.list_dir(base, keep=...)``: the GUI's INPUT-file picker
(cells, bounds). Optionally offers a '(from scratch)' sentinel. StorePicker below is its
generated-kind twin, which reads the artifact store instead of a directory."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

_TOOLTIP_ROLE = Qt.ToolTipRole

from core.Helpers import file_manager
from core.gui import icons


class ArtifactPicker(QWidget):
    # Plain ASCII, deliberately. This is a combo ITEM label, not a button, so icons.apply_icon cannot
    # reach it -- that helper sets a widget's font family, and a QComboBox item is not a widget. The
    # alternative, a QIcon rendered from the icon font, would be the one place in the app carrying a
    # baked-in colour that does NOT follow a theme flip (icons.py's opening note is that rendering
    # icons as TEXT is what makes them recolour for free). It used to be "➕", a heavy-plus emoji whose
    # coverage and colour vary per font -- exactly the quirk the icon font was introduced to sidestep.
    NEW_LABEL = "+  (from scratch)"

    def __init__(self, base_path, keep=None, allow_new: bool = False, parent=None):
        super().__init__(parent)
        self.base_path = Path(base_path)
        self._keep = keep
        self._allow_new = allow_new

        self.combo = QComboBox()
        refresh = QPushButton()                 # keep QPushButton -> QSS selector QPushButton#iconButton
        refresh.setObjectName("iconButton")     # compact square button -> small QSS padding (glyph fits)
        icons.apply_icon(refresh, "refresh")    # bundled icon font (falls back to "⟳")
        refresh.setFixedWidth(32)
        refresh.setToolTip("Rescan the folder")
        refresh.clicked.connect(self.refresh)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)
        layout.addWidget(refresh)
        self.refresh()

    def refresh(self):
        self.combo.clear()
        if self._allow_new:
            self.combo.addItem(self.NEW_LABEL, userData=None)
        entries = []
        if self.base_path.exists():
            entries = file_manager.list_dir(str(self.base_path), keep=self._keep)
        for entry in entries:
            # display forward-slashed for subfoldered layouts; keep the raw relpath as data
            self.combo.addItem(entry.replace("\\", "/"), userData=entry)
            # Per-item tooltip: AdjustToContentsOnFirstShow sizes the combo to whatever was present
            # at first show, and refresh() adds entries LATER (after a save, or a model change), so
            # those get silently elided with no way to read them.
            self.combo.setItemData(self.combo.count() - 1, entry.replace("\\", "/"), _TOOLTIP_ROLE)

    def selected(self):
        """Return (entry_or_None, is_new). ``entry`` is the path relative to base_path."""
        data = self.combo.currentData()
        return data, (data is None)

    def selected_path(self):
        """Full filesystem path of the selected entry, or None for the '(from scratch)' sentinel."""
        data = self.combo.currentData()
        return None if data is None else str(self.base_path / data)

    def has_entries(self) -> bool:
        return any(self.combo.itemData(i) is not None for i in range(self.combo.count()))

    # ── persistence ──────────────────────────────────────────────────────────
    def key(self) -> str:
        """A stable string identifying the current selection, for QSettings. The relpath (userData),
        not the index (order shifts as files are added), and "" for the '(from scratch)' sentinel."""
        data = self.combo.currentData()
        return "" if data is None else str(data)

    def restore_key(self, key: str) -> None:
        """Reselect a previously saved key. If the file is gone (findData == -1), leave the current
        selection alone rather than blanking the combo with setCurrentIndex(-1)."""
        if not key:
            return
        i = self.combo.findData(key)
        if i >= 0:
            self.combo.setCurrentIndex(i)

    def repoint(self, base_path, restore_key: str | None = None) -> None:
        """Point the picker at a different folder, rescan, and optionally re-apply a saved selection.

        THE ONE PLACE THE ORDER LIVES (four panels used to restate it): the folder must be set and
        refreshed BEFORE any saved key is re-applied, because refresh() rebuilds the combo and would
        wipe a selection restored first. Callers restoring from QSettings must also restore their
        MODEL combo first and call their _on_model_changed explicitly (currentTextChanged does not
        fire when the saved value equals the current text), since that is what routes here.
        """
        self.base_path = Path(base_path)
        self.refresh()
        if restore_key is not None:
            self.restore_key(restore_key)


# The visible item text marks the EXCEPTION, never the norm (piece 4, B13). Nearly every posterior is
# amortized, so an "amortized" suffix on almost every row would carry no information at all and would
# push the name itself out of a combo that AdjustToContentsOnFirstShow sized to its first listing. A
# TSNPE round's result is the rare row -- valid near ONE observation, and the one you must not pick by
# accident -- so it is the one that gets a suffix. Plain text, no icon: see ArtifactPicker.NEW_LABEL
# for why a combo ITEM cannot carry the bundled icon font.
NARROWED_SUFFIX = "  —  narrowed (TSNPE)"


def _summary_line(s) -> str:
    """One line describing a store row -- mode, conditioning width, amortization, creation time, in
    that order -- with the parts the kind does not carry left out. ``""`` for a row with none of them.

    THE ONE FORMATTER BEHIND THE LINE UNDER EVERY PICKER, so the three tabs cannot word the same
    facts three ways -- and the tooltip now words amortization the same way this does, "narrowed
    (TSNPE)", which is also what the item text's suffix says: one fact, one wording, wherever it is
    shown. The per-item TOOLTIP is still not BUILT from this function: it leads with the id and keeps
    every fact it carried (B13), while the line needs no id, because the picker's own text already
    says which item it is describing.

    ``getattr`` throughout, because this is handed ``Summary`` in production and a row stub in the
    suites, and a formatter feeding a read-only label must not raise over a missing attribute.
    """
    parts = []
    if getattr(s, "mode", None):
        parts.append(str(s.mode))
    if getattr(s, "width", None):
        parts.append(f"width {s.width}")
    amortized = getattr(s, "amortized", None)
    if amortized is not None:
        parts.append("amortized" if amortized else "narrowed (TSNPE)")
    if getattr(s, "created", None):
        parts.append(str(s.created))
    return " · ".join(parts)


class StorePicker(QWidget):
    """A combo over one KIND of the artifact store -- the generated-kind twin of ArtifactPicker,
    which stays for the input pickers (cells, bounds). Items are FINISHED artifacts only, labelled by
    name (or ``(unnamed <id>)``), with the id, creation time, mode, width and amortization in the
    tooltip; ``userData`` is the id, which is what ``key()`` persists and ``selected()`` returns.

    ``row_filter`` narrows the listing further -- one kind can hold several things a screen offers
    separately (spec §5.4).
    """
    NEW_LABEL = ArtifactPicker.NEW_LABEL

    def __init__(self, kind: str, allow_new: bool = False, store=None, row_filter=None,
                 parent=None):
        super().__init__(parent)
        self.kind, self._allow_new, self._store = kind, allow_new, store
        # ``Summary -> bool``, or None for "every row of the kind". One kind can hold several things
        # a screen wants to offer separately: the ``fdt`` kind holds single-cell measurements, sweeps
        # and comparisons, told apart by ``body["study"]`` (spec §2.1), and the FDT and CrossVal
        # screens each offer one of them. Applied AFTER the finished rule. Before ``parent`` (spec
        # §5.4, P9); no caller passes ``parent`` positionally.
        self._row_filter = row_filter
        # id -> the line selection_summary() returns, recorded by refresh(). BEFORE refresh() below,
        # which fills it.
        self._summaries: dict = {}
        self.combo = QComboBox()
        refresh = QPushButton()
        refresh.setObjectName("iconButton")
        icons.apply_icon(refresh, "refresh")
        refresh.setFixedWidth(32)
        refresh.setToolTip("Rescan the artifact store")
        refresh.clicked.connect(self.refresh)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)
        layout.addWidget(refresh)
        self.refresh()

    def _resolved_store(self):
        from core.artifacts import resolve_store
        return resolve_store(self._store)

    def refresh(self):
        current = self.key()
        self.combo.clear()
        self._summaries = {}                     # rebuilt here, so a deleted artifact's line goes too
        if self._allow_new:
            self.combo.addItem(self.NEW_LABEL, userData=None)
        try:
            rows = self._resolved_store().list(self.kind)
        except Exception:                        # noqa: BLE001 -- an unreadable root lists nothing
            rows = []
        for s in rows:
            # FINISHED, not ``complete``. ``complete`` means "has a valid manifest"; for the two
            # kinds written progressively (the training cache, and piece 5's ``fdt``) a record
            # carries one from its first moment, so ``complete`` would offer a run still going, or
            # one a cancel left half written, as a result (spec §5.4). For every other kind the two
            # are the same fact, because _commit writes the manifest last.
            if not s.finished:
                continue
            if self._row_filter is not None and not self._row_filter(s):
                continue
            # The suffix is appended to the DISPLAY only. key(), restore_key(), selected() and
            # has_entries() all go through userData, which is still the bare id, so nothing that
            # persists or resolves a selection can be broken by a change of wording here.
            self.combo.addItem(s.label + (NARROWED_SUFFIX if s.amortized is False else ""),
                               userData=s.id)
            self._summaries[str(s.id)] = _summary_line(s)
            tip = f"{s.id} · {s.created}"
            if s.mode:
                tip += f" · {s.mode}"
            if s.width:
                tip += f" · width {s.width}"
            if s.amortized is not None:
                # ONE WORDING FOR ONE FACT (B13): the same "narrowed (TSNPE)" the item
                # suffix and the line under the picker use, so nobody has to learn that a
                # shouted "NON-AMORTIZED" here and a quiet "narrowed" there are the same
                # thing. The tooltip keeps everything else it carried.
                tip += " · amortized" if s.amortized else " · narrowed (TSNPE)"
            self.combo.setItemData(self.combo.count() - 1, tip, _TOOLTIP_ROLE)
        self.restore_key(current)

    def selected(self):
        """``(id_or_None, is_new)``."""
        data = self.combo.currentData()
        return data, (data is None)

    def has_entries(self) -> bool:
        return any(self.combo.itemData(i) is not None for i in range(self.combo.count()))

    def key(self) -> str:
        data = self.combo.currentData()
        return "" if data is None else str(data)

    def selection_summary(self) -> str:
        """The current item spelled out -- "chi · width 18 · amortized · 2026-09-14T10:22:31" -- and
        "" when nothing is selected or the '(from scratch)' sentinel is (a sentinel is not an
        artifact and has nothing to describe).

        Read off what the last ``refresh()`` recorded, never off the store: this is called from a
        ``currentIndexChanged`` slot, and re-listing a kind's directory on every index change is how
        a combo becomes a disk scan.
        """
        data = self.combo.currentData()
        return "" if data is None else self._summaries.get(str(data), "")

    def restore_key(self, key: str) -> None:
        if not key:
            return
        i = self.combo.findData(key)
        if i >= 0:
            self.combo.setCurrentIndex(i)
