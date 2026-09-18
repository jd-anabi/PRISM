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

    # And it must survive switching kinds, not just quitting: refresh()'s own capture guard reads
    # columnCount() rather than topLevelItemCount(), so leaving an empty kind still hands the sort
    # forward instead of letting _apply_sort clobber it with a stale self._sort. "calibration" is
    # also empty in this store and shares the 3-column width, so column 2 stays valid throughout.
    _show(screen, "calibration")
    assert screen.table.sort_state() == (2, 1), "the sort was lost switching off an empty kind"
    _show(screen, "prior")
    assert screen.table.sort_state() == (2, 1), "the sort was lost switching back to an empty kind"


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


def test_the_detail_pane_shows_the_manifest_and_the_run_records(tmp_path):
    """B4 / §3.3. The manifest as Task 6 renders it, then the run's records verbatim with their
    HH:MM:SS level stamps -- one string, which is also exactly what Save writes (B10)."""
    from core.artifacts import render_manifest
    from core.artifacts.store import LOG_FILE

    ids = build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    # A run's records, where ArtifactWriter._commit would have written them.
    (store.path("calibration", ids["calibration"]) / LOG_FILE).write_text(
        "12:00:00 info building the calibration\n12:00:04 warning 1 dataset diverged\n",
        encoding="utf-8")

    screen = artifact_screen(store)
    _show(screen, "calibration")
    s = _select(screen, 0)
    text = screen.detail.toPlainText()
    assert render_manifest(store.get("calibration", s.id)) in text, text[:400]
    assert "12:00:04 warning 1 dataset diverged" in text, "the records are verbatim, stamps and all"
    assert screen.detail.isReadOnly()
    assert screen.btn_save.isEnabled()


def test_the_detail_pane_states_the_gaps_a_blank_pane_would_hide(tmp_path, monkeypatch):
    """§3.3's four sentences about the records, plus the stale-folder note. A training cache keeps no
    log BY DESIGN, an artifact written outside any run has none either, a tail announces itself, a run
    that said nothing says so -- read_log answers "" there and None for no file at all (§2.2), and
    piece 3's invariant is that silence is a record too -- and a folder whose name disagrees with its
    own manifest says so, the state ArtifactStore.rename leaves when the directory move is refused
    (store.py:430-441), which nothing has ever shown."""
    from core.artifacts.store import LOG_FILE
    from core.gui.screens import artifact_screen as mod

    ids = build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    screen = artifact_screen(store)

    # (a) the cache: no log, for its own reason
    _show(screen, "simulation")
    _select(screen, 0)
    assert "a training cache keeps no log" in screen.detail.toPlainText()

    # (b) an artifact whose run never existed
    _show(screen, "inference")
    _select(screen, 0)
    assert "written outside a run" in screen.detail.toPlainText()

    # (c) a tail says it is a tail. 1 MiB is the real ceiling; the test moves it rather than write one.
    assert mod.LOG_MAX_BYTES == 1 << 20
    monkeypatch.setattr(mod, "LOG_MAX_BYTES", 64)
    (store.path("inference", ids["inference"]) / LOG_FILE).write_text("x" * 500, encoding="utf-8")
    screen.refresh()
    _select(screen, 0)
    assert "only the last 64 bytes" in screen.detail.toPlainText()

    # (d) a run that said nothing: read_log answers "" here, not None, and the pane must not be blank
    #     under the Records heading. _show switches the kind, which re-lists.
    (store.path("diagnostic", ids["diagnostic"]) / LOG_FILE).write_text("", encoding="utf-8")
    _show(screen, "diagnostic")
    _select(screen, 0)
    assert store.read_log("diagnostic", ids["diagnostic"]) == ("", False)
    assert "the run recorded nothing" in screen.detail.toPlainText()

    # (e) the stale folder: move the directory and leave the manifest naming the old one
    sub = store.path("prior", ids["prior"])
    sub.rename(sub.with_name("was_moved__" + ids["prior"]))
    _show(screen, "prior")
    s = _select(screen, 0)
    assert s.complete and s.dir_name == "was_moved__" + ids["prior"]
    text = screen.detail.toPlainText()
    assert "was_moved__" in text and "browse_prior__" in text, text[:400]
    assert "manifest is what resolves" in text, text[:400]


def test_a_caches_folder_name_is_never_called_stale(tmp_path):
    """§3.3, the one exception to the stale-folder note: it is SUPPRESSED for the simulation kind.

    Manifest.dir_name is the bare digest for a cache (manifest.py:73-77) and
    write_simulation_manifest writes wherever it is handed (store.py:778-815), so a cache directory
    whose name is not its id is legitimate -- the note would fire on any hand-placed cache and tell
    the operator that a rename half-failed when nothing of the sort happened. The cache's own
    sentence about its missing log is still there, so this is a suppression and not a blank pane.
    """
    from core.artifacts import write_simulation_manifest
    from core.SBI.training_checkpoint import identity_digest

    store = ArtifactStore(tmp_path)
    identity = {"format": "training-rows/2", "prior_fingerprint": "a" * 16, "n_runs": 2,
                "truncation": None}
    m = write_simulation_manifest(store.kind_dir("simulation") / "hand_placed_cache", identity,
                                  batches_done=1, complete=False)
    assert m.dir_name == identity_digest(identity) != "hand_placed_cache"

    screen = artifact_screen(store)
    _show(screen, "simulation")
    s = _select(screen, 0)
    assert s.dir_name == "hand_placed_cache" and s.id == m.id
    text = screen.detail.toPlainText()
    assert "manifest is what resolves" not in text, text[:400]
    assert "only the folder name is out of date" not in text, text[:400]
    assert "a training cache keeps no log" in text, text[:400]


def test_an_incomplete_directorys_pane_shows_its_reason_its_path_and_its_folder(tmp_path):
    """§3.3's last line. There is no manifest to render, so what there is to say is why it was not
    read, where it is and what it is called."""
    ids = build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    screen = artifact_screen(store)          # opens on priors, where the three leftovers are

    seen = {}
    for i in range(screen.table.topLevelItemCount()):
        s = _select(screen, i)
        if not s.complete:
            seen[s.dir_name] = screen.detail.toPlainText()
    assert set(seen) == set(ids["bad"]), sorted(seen)
    for dir_name, text in seen.items():
        assert dir_name in text
        assert str(store.kind_dir("prior") / dir_name) in text
        assert any(r in text for r in ("no manifest.json", "unreadable manifest", "declares kind")), \
            text


def test_save_writes_exactly_what_the_pane_shows(tmp_path, monkeypatch):
    """B10: a FILE, chosen through QFileDialog the way simulate_panel._save_video chooses one, holding
    byte-for-byte what is on screen. Cancelling writes nothing and says nothing."""
    from PySide6.QtWidgets import QFileDialog

    build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    screen = artifact_screen(store)
    _select(screen, 0)
    shown = screen.detail.toPlainText()
    assert shown

    out = tmp_path / "report"                      # no suffix: Save adds .txt
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    screen.btn_save.click()
    assert out.with_suffix(".txt").read_text(encoding="utf-8") == shown
    assert "report.txt" in screen.status.text(), screen.status.text()

    calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: calls.append(1) or ("", ""))
    screen._set_status("")
    screen.btn_save.click()
    assert calls == [1] and screen.status.text() == ""
    assert sorted(p.name for p in tmp_path.glob("report*")) == ["report.txt"]
