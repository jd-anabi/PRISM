"""The artifact browser (piece 4, §3): the Artifacts screen, its listing, its detail pane and its
actions. The eighteenth suite. Run directly: pytest tests/test_artifact_browser.py"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest                                                          # noqa: E402,F401

from core.artifacts import ArtifactStore                               # noqa: E402
from core.artifacts.store import KIND_DIRS                             # noqa: E402
from tests._fixtures import artifact_screen, build_browse_store, qt_app  # noqa: E402

KINDS = list(KIND_DIRS)


def _show(screen, kind: str) -> None:
    """Switch the kind selector, which fires the screen's refresh()."""
    screen.kind_combo.setCurrentIndex(KINDS.index(kind))
    assert screen.kind() == kind


def _row_text(table, i: int) -> str:
    """Every cell of one row, joined -- so a test reads what a user reads and does not depend on
    which column a kind happens to put a fact in (core/gui/widgets/artifact_table.py owns that)."""
    item = table.topLevelItem(i)
    return " | ".join(item.text(c) for c in range(table.columnCount()))


def _select(screen, i: int):
    """Select row ``i`` and return its Summary."""
    screen.table.setCurrentItem(screen.table.topLevelItem(i))
    return screen.table.current_summary()


def test_the_browser_lists_the_seven_kinds_with_the_incomplete_directories_last(tmp_path):
    """B1/B2 and §3.2. Seven kinds in KIND_DIRS order, one kind at a time, each artifact's own row
    named by Summary.label -- and for the prior kind, the three leftover directories after the real
    one, each showing why it was not read instead of the kind's own columns.

    The order is asserted off current_summary().complete rather than off the painted rows: what has
    to hold is that ArtifactStore.list's complete-first ordering survives into the table, and no
    arrangement of names can make that accidental.
    """
    ids = build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    screen = artifact_screen(store)

    assert [screen.kind_combo.itemData(i) for i in range(screen.kind_combo.count())] == KINDS

    for kind in KINDS:
        _show(screen, kind)
        assert screen.table.topLevelItemCount() >= 1, f"{kind} listed nothing"
        first = _row_text(screen.table, 0)
        if kind == "simulation":
            # a cache is always unnamed, so its label is "(unnamed <digest>)"
            assert ids["simulation"] in first and "unnamed" in first, first
        else:
            assert f"browse_{kind}" in first, first
            assert f"the {kind} row" in first, f"the note column is empty: {first}"

    _show(screen, "prior")
    assert screen.table.topLevelItemCount() == 4
    assert [_select(screen, i).complete for i in range(4)] == [True, False, False, False]
    tail = "\n".join(_row_text(screen.table, i) for i in (1, 2, 3))
    for dir_name in ids["bad"]:
        assert dir_name in tail, tail
    for reason in ("no manifest.json", "unreadable manifest", "declares kind"):
        assert reason in tail, tail
    assert "1 complete" in screen.status.text() and "3 incomplete" in screen.status.text(), \
        screen.status.text()


def test_a_kind_with_nothing_in_it_is_not_a_root_that_cannot_be_read(tmp_path):
    """§3.2. StorePicker.refresh swallows the exception and lists nothing (artifact_picker.py:135-137),
    so today an empty store and an unreadable one are the same picture. The browser is the one place
    that difference has to be visible: an absent kind directory reads "nothing here yet", and a root
    that raises puts the error, with its class, on the status line."""
    import types

    store = ArtifactStore(tmp_path)                    # nothing written at all
    screen = artifact_screen(store)
    assert screen.table.topLevelItemCount() == 0
    assert screen.status.text() == "Nothing here yet.", screen.status.text()
    assert not (tmp_path / KIND_DIRS["prior"]).exists(), "listing must not create the directory"

    def _boom(kind):
        raise PermissionError("Access is denied")

    screen._store = types.SimpleNamespace(list=_boom)
    screen.refresh()
    assert screen.table.topLevelItemCount() == 0
    assert screen.status.text().startswith("⚠ "), screen.status.text()
    assert "PermissionError" in screen.status.text() and "Access is denied" in screen.status.text()


def test_the_artifacts_tile_opens_the_browser_and_the_window_saves_its_state():
    """B1's five registration points, in one real window.

    The tile is on Home and live; the screen sits AFTER the four sections and BEFORE Settings, so the
    back-arrow slide direction stays monotone with the tile order; the name maps to its stack index;
    and _save_state calls the screen's save_settings BY NAME. It is NOT a BasePanel and NOT in
    _all_panels(), which is what keeps it out of both the app-wide control lock and
    _refresh_model_combos.
    """
    from PySide6.QtWidgets import QPushButton
    from core.gui import settings as st
    from core.gui.main_window import MainWindow
    from core.gui.panels.base_panel import BasePanel
    from core.gui.screens.artifact_screen import ArtifactScreen
    from core.gui.screens.home_screen import SECTIONS

    qt_app()
    assert SECTIONS[-1] == "Artifacts", SECTIONS

    w = MainWindow()
    assert isinstance(w.artifact_screen, ArtifactScreen)
    idx = w._section_index["Artifacts"]
    assert w._section_index["Simulate"] < idx < w._section_index["Settings"]
    assert w.nav.stack.widget(idx) is w.artifact_screen

    assert not isinstance(w.artifact_screen, BasePanel), \
        "a BasePanel enrols in _instances, so every run anywhere would grey the browser out"
    assert w.artifact_screen not in w._all_panels()

    home = w.nav.stack.widget(0)
    btn = next(b for b in home.findChildren(QPushButton) if b.text() == "Artifacts")
    assert btn.property("accent") is True, "a live tile is accent-styled; a dead one is not"
    btn.click()
    assert w.nav.stack.currentIndex() == idx

    w.artifact_screen.kind_combo.setCurrentIndex(KINDS.index("posterior"))
    w._save_state()
    qs = st.settings()
    qs.beginGroup("artifacts")
    got = {k: qs.value(k) for k in qs.childKeys()}
    qs.endGroup()
    assert got.get("kind") == "posterior", got


def test_the_browser_remembers_the_kind_and_the_sort_but_never_the_selection(tmp_path):
    """§3.5 / V5. The kind last viewed and the sort survive a relaunch; the SELECTED artifact
    deliberately does not -- a remembered id that has since been deleted is exactly the dangling
    state §3.4 removes from the pickers. A stale kind an older build left behind is ignored, not
    restored.

    The sort that is saved is deliberately NOT the default one: (1, 1) is DEFAULT_SORT, so a test
    that saved it could not tell a restored sort from a screen that had simply never been sorted at
    all, and the assertion below would hold even if restore_settings did nothing."""
    from core.gui import settings as st

    build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    first = artifact_screen(store)
    _show(first, "posterior")
    assert first.table.sort_state() == (1, 1), \
        "DEFAULT_SORT moved; pick a saved sort below that differs from it"
    first.table.apply_sort_state(0, 0)          # Name, ascending -- not the default
    assert _select(first, 0) is not None
    # Asserted HERE, on a screen that HAS a selection, because a re-list is the one place a
    # remembered id could come back. The same assertion on the fresh screen below could not fail
    # whatever the code did -- that screen's table is a different object that never had a selection.
    first.refresh()
    assert first.table.current_summary() is None, "a re-list must not restore the selected artifact"

    qs = st.settings()
    first.save_settings(qs)
    qs.sync()
    qs.beginGroup("artifacts")
    assert set(qs.childKeys()) == {"kind", "sort_col", "sort_order"}, sorted(qs.childKeys())
    qs.endGroup()

    again = artifact_screen(store)
    assert again.kind() == "posterior"
    assert again.table.sort_state() == (0, 0), "the saved sort was overwritten before it was applied"

    qs.setValue("artifacts/kind", "nosuchkind")
    qs.sync()
    third = artifact_screen(store)
    assert third.kind() == KINDS[0], "a stale kind key must be ignored, not restored"


def test_a_sort_clicked_on_a_kind_with_no_artifacts_is_still_saved(tmp_path):
    """§3.5. A kind with nothing in it still has a HEADER, so a sort clicked there is the user's
    choice like any other -- and it is exactly the case a "does the table have rows" test throws
    away. save_settings asks the table whether it has COLUMNS (i.e. whether set_rows has run), which
    is the question it means."""
    from PySide6.QtCore import Qt
    from core.gui import settings as st

    store = ArtifactStore(tmp_path)                    # empty: not one artifact of any kind
    screen = artifact_screen(store)
    assert store.list(screen.kind()) == []
    assert screen.table.topLevelItemCount() == 0 and screen.table.columnCount() >= 3
    stale = screen._sort

    # The user clicks the last header -- what a click does is move the sort indicator.
    screen.table.header().setSortIndicator(2, Qt.DescendingOrder)
    assert screen.table.sort_state() == (2, 1) != stale

    qs = st.settings()
    screen.save_settings(qs)
    qs.sync()
    assert (st.get_int(qs, "artifacts/sort_col", -1), st.get_int(qs, "artifacts/sort_order", -1)) \
        == (2, 1), "the sort clicked on an empty kind was discarded in favour of the stale one"


def test_a_store_change_in_the_browser_re_lists_every_artifact_picker():
    """B8. MainWindow connects the screen's store_changed to _refresh_store_pickers, the twin of
    _refresh_model_combos that a saved user model already drives. The pickers are found by TYPE, not
    by naming prior_picker / post_picker / obs_picker, so a fourth one added to a tab is covered by
    construction -- which is why what is asserted is EVERY picker the window holds, not a count. A
    count would make a later task that adds a picker break this test instead of being covered by it.
    The three kinds today are asserted as a SUBSET for the same reason."""
    from core.gui.main_window import MainWindow
    from core.gui.widgets.artifact_picker import StorePicker

    qt_app()
    w = MainWindow()
    pickers = [p for panel in w._all_panels() for p in panel.findChildren(StorePicker)]
    assert pickers, "the window holds no artifact picker at all, so this proves nothing"
    assert {"prior", "posterior", "observation"} <= {p.kind for p in pickers}

    seen = []
    for p in pickers:
        p.refresh = lambda _p=p: seen.append(_p)
    w.artifact_screen.store_changed.emit()
    assert [id(p) for p in seen] == [id(p) for p in pickers], \
        f"refreshed {[p.kind for p in seen]}, but the window holds {[p.kind for p in pickers]}"
