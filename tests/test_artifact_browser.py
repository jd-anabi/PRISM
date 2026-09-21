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
from tests._fixtures import (_nad_cfg, _prior_artifact, artifact_screen,  # noqa: E402
                             build_browse_store, qt_app)

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


def _show_kind(scr, kind):
    """Point the screen at one kind and re-list it. The selector carries the kind key as its item
    data (Task 9), so a test names a kind rather than an index."""
    i = scr.kind_combo.findData(kind)
    assert i >= 0, f"the kind selector does not offer {kind!r}"
    scr.kind_combo.setCurrentIndex(i)
    scr.refresh()


def _select_ref(table, ident):
    """Make the row for ``ident`` (an artifact id or an incomplete directory's name) current, and
    return its Summary. Plain QTreeWidget API plus the table's own ``current_summary`` -- the rows
    are sorted (complete first, newest first), so a test must never assume an index."""
    for i in range(table.topLevelItemCount()):
        table.setCurrentItem(table.topLevelItem(i))
        s = table.current_summary()
        if s is not None and ident in (s.id, s.dir_name):
            return s
    raise AssertionError(f"no row for {ident!r} among {table.topLevelItemCount()} row(s)")


def _answer(monkeypatch, button):
    """Layer a chosen answer over tests/conftest.py::_no_modal_dialogs, the pattern
    tests/test_nav_and_gating.py::test_the_d7_and_d8_dialogs_default_to_cancel uses (a NAME, because
    the line it sits on moves). The session guard records every box and returns 0, which
    every confirmation here reads as No; these dialogs use STANDARD buttons, so the answer is the
    returned enum rather than a click on an added button."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN
    monkeypatch.setattr(QMessageBox, "exec", lambda self: SHOWN.append(self) or button)


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
    """§3.2. StorePicker.refresh swallows the exception and lists nothing (artifact_picker.py:175-178),
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
    (store.py:520-540), which nothing has ever shown."""
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
    write_simulation_manifest writes wherever it is handed (store.py:1010-1049), so a cache directory
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


def test_setting_a_note_trims_it_and_asks_nothing(store):
    """B5: the rule is core's (require_note) and the box is the front end's. One trimmed line,
    written through set_note; a blank clears it; nothing is asked."""
    from PySide6.QtWidgets import QLabel
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="annotated")
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    # The control core/gui/fields.py names ("Edit it in the Note box on the Artifacts screen.") has
    # to exist, spelled that way: the table is the ONE place a control is named, and a sentence
    # pointing at a box nobody can find is the failure mode V3 exists to stop.
    assert any(lbl.text() == "Note" for lbl in scr.findChildren(QLabel)), "no 'Note' label"
    scr.note_edit.setText("   spontaneous, 4.5 s   ")
    scr._set_note()
    assert store.get("prior", p.id).note == "spontaneous, 4.5 s"
    assert SHOWN == [], "setting a note must not ask anything"
    assert "Set the note" in scr.status.text(), scr.status.text()
    # Re-select: _after_change re-lists the kind, and the screen deliberately remembers no selection
    # (spec §3.5 -- a remembered id that has since been deleted is the dangling state B8 prevents).
    _select_ref(scr.table, p.id)
    scr.note_edit.setText("")
    scr._set_note()
    assert store.get("prior", p.id).note == ""
    assert "Cleared the note" in scr.status.text(), scr.status.text()


def test_an_over_long_note_is_refused_with_its_fix_sentence(store):
    """V2: refused, never clamped. The box carries core's neutral sentence (both numbers) and the
    front end's own "where to fix it" from core/gui/fields.py, and the manifest is untouched."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="annotated")
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    scr.note_edit.setText("x" * 201)
    # No setMaxLength on the box, deliberately: it would truncate at 200 and the refusal could never
    # fire, which is exactly the silent clamp B5 forbids.
    assert len(scr.note_edit.text()) == 201, "the Note box must not clamp what was typed"
    scr._set_note()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert "200" in box.text() and "201" in box.text(), box.text()
    assert box.informativeText() == "Edit it in the Note box on the Artifacts screen."
    assert store.get("prior", p.id).note == ""


def test_delete_refuses_an_artifact_with_dependents_and_offers_no_yes(store):
    """B6. No front end offers force=True, so ArtifactStore.delete refuses anything with dependents
    -- a confirmation could only ever be followed by a failure. dependents() is therefore read
    FIRST, and the yellow box names every dependent, including the training cache that holds the
    prior only by fingerprint and names it nowhere (ArtifactStore.dependents' second pass)."""
    from core.artifacts import store as st
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="ancestor")
    fp = store.get("prior", p.id).fingerprints["gmm"]
    with store.create("inference", cfg, name="child") as w:        # names the prior as its parent
        w.parents = {"prior": p.id}
        w.body = {"results": {}}
    ident = {"format": "training-rows/2", "prior_fingerprint": fp, "n_runs": 3, "truncation": None}
    cache = st.write_simulation_manifest(store.kind_dir("simulation") / "abcdef012345", ident,
                                         parents=None)             # keyed on the GMM, names nothing
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    scr._delete()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert len(box.buttons()) == 1, "a refusal offers no Yes"
    assert "prior ancestor" in box.text() and p.id in box.text(), box.text()
    assert f"inference child [{w.id}]" in box.text(), box.text()
    assert f"simulation (unnamed) [{cache.id}]" in box.text(), box.text()
    assert ("a training cache was generated against this prior and its rows are meaningless "
            "without it") in box.text(), box.text()
    assert box.informativeText() == "Select an artifact in the list on the Artifacts screen."
    assert store.get("prior", p.id).name == "ancestor", "nothing may be deleted"
    assert "2 dependent(s) and was not deleted" in scr.status.text(), scr.status.text()


def test_deleting_an_unfinished_cache_names_its_batches_and_defaults_to_no(store, monkeypatch):
    """B3 + B6: a manifested cache is COMPLETE (it has a valid manifest) and not FINISHED, and the
    batches already committed are the one thing a delete here destroys that a rerun cannot remake --
    so the confirmation names them, and No is the default button (which is also what the session
    dialog guard's exec()==0 reads as).

    The cache is written with NO rows, because that is the only mid-run state there is: rows are
    written by mark_complete alone and ``save`` passes none (P2), so the prompt names batches and
    says where the row counts come from rather than printing a confident, false "0 rows"."""
    import pytest
    from core.artifacts import StoreError, store as st
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    ident = {"format": "training-rows/2", "prior_fingerprint": None, "n_runs": 4, "truncation": None}
    cache = st.write_simulation_manifest(store.kind_dir("simulation") / "beef00112233", ident,
                                         batches_done=2, complete=False)
    scr = artifact_screen(store)
    _show_kind(scr, "simulation")
    s = _select_ref(scr.table, cache.id)
    assert s.complete and not s.finished, "B3: complete is 'has a manifest', finished is 'it ended'"
    assert s.rows is None, "``save`` records no rows: a mid-run cache has none to name"
    scr._delete()                                     # exec() returns 0 -> not Yes
    box = SHOWN[-1]
    assert box.button(QMessageBox.No) is box.defaultButton(), "No must be the default"
    assert "2 committed batch(es)" in box.informativeText(), box.informativeText()
    assert "rows are recorded when the cache finishes" in box.informativeText(), \
        box.informativeText()
    assert "0 rows" not in box.informativeText(), box.informativeText()
    assert store.get("simulation", cache.id).id == cache.id, "No must leave it on disk"
    assert "was not deleted" in scr.status.text(), scr.status.text()
    # Yes deletes it, and the change is announced (B8)
    _answer(monkeypatch, QMessageBox.Yes)
    _select_ref(scr.table, cache.id)
    changed = []
    scr.store_changed.connect(lambda: changed.append(True))
    scr._delete()
    with pytest.raises(StoreError):
        store.get("simulation", cache.id)
    assert changed, "a delete must emit store_changed"


def test_the_stores_own_refusal_is_the_last_word_on_a_delete(store, monkeypatch):
    """dependents() is read twice -- once here to avoid asking a question that could only fail, once
    inside delete() -- and the store's is the answer that counts. With the SCREEN's read stubbed
    empty the confirmation appears, and the store's own sentence is what the operator is shown, with
    its own fix sentence under it: that refusal carries field="artifact" (Task 5), so this race path
    shows it as it stands and invents nothing.

    The stub is a proxy over the real store, patched onto the SCREEN's _resolved_store, and not
    monkeypatch.setattr(store, "dependents", ...): ArtifactStore.delete calls self.dependents
    itself, so patching the store would blind the store too -- the delete would
    SUCCEED, the prior would be destroyed and every assertion below would be asserting the opposite
    of what it says (Q6). The proxy lies to the screen only."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="ancestor")
    with store.create("inference", cfg, name="child") as w:
        w.parents = {"prior": p.id}
        w.body = {"results": {}}
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)

    class _BlindToDependents:
        """Everything the screen asks of a store, delegated to the real one -- except dependents,
        which answers "none" the way a store would have a moment before the child was written."""

        def __init__(self, real):
            self._real = real

        def dependents(self, kind, id_):
            return []

        def get(self, *a, **k):
            return self._real.get(*a, **k)

        def list(self, *a, **k):
            return self._real.list(*a, **k)

        def delete(self, *a, **k):
            return self._real.delete(*a, **k)

    monkeypatch.setattr(scr, "_resolved_store", lambda: _BlindToDependents(store))
    _answer(monkeypatch, QMessageBox.Yes)
    scr._delete()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs", "the store's refusal takes the yellow box"
    assert "refusing to delete" in box.text() and w.id in box.text(), box.text()
    # Task 5 gave delete()'s dependents refusal field="artifact", so show_refusal adds the same fix
    # sentence it adds to every other fielded refusal -- the race path needs no refusal of its own.
    # The normal path still builds one, for the wording only: the store's sentence ends "pass
    # force=True to orphan them", and no front end offers that (B6).
    assert box.informativeText() == "Select an artifact in the list on the Artifacts screen."
    assert store.get("prior", p.id).name == "ancestor", "the artifact survives its own refusal"


def test_sweep_removes_only_the_leftovers_and_reports_what_it_could_not(store, monkeypatch):
    """B7: one action per kind and one for all seven, behind a confirmation that lists exactly the
    rows the table calls incomplete. A directory whose manifest is USABLE AND DECLARES THIS KIND is
    never touched -- fix round 1, RULED IN 4: the box used to promise this for "a directory that
    holds a manifest", full stop, which is false -- a directory whose manifest declares a DIFFERENT
    kind holds one too, and is a leftover the sweep removes (task-14-report.md's own probe removed
    exactly one). A directory that will not delete is reported and the rest still go, and nothing to
    do says so without asking."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="keeper")
    leftover = store.kind_dir("prior") / "_unnamed__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    (leftover / "prior.pt").write_bytes(b"half a run, no manifest")
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    scr._sweep(all_kinds=False)                          # No -> nothing goes
    box = SHOWN[-1]
    assert box.button(QMessageBox.No) is box.defaultButton()
    assert leftover.name in box.informativeText(), box.informativeText()
    assert "no manifest.json" in box.informativeText(), box.informativeText()
    assert ("A directory whose manifest is usable and declares this kind is never touched by this."
            in box.informativeText()), box.informativeText()
    assert leftover.is_dir() and "Nothing was removed." in scr.status.text()
    _answer(monkeypatch, QMessageBox.Yes)                # Yes -> the leftover goes, the artifact stays
    scr._sweep(all_kinds=False)
    assert not leftover.exists()
    assert store.get("prior", p.id).name == "keeper"
    assert "Removed 1 of 1" in scr.status.text(), scr.status.text()
    SHOWN.clear()                                        # nothing to do: no dialog at all
    scr._sweep(all_kinds=False)
    assert SHOWN == [] and "Nothing to remove" in scr.status.text(), scr.status.text()
    # a directory that will not delete (a handle held open on Windows) is reported, never fatal
    (store.kind_dir("prior") / "stuck__20260917T091000").mkdir()
    monkeypatch.setattr(store, "sweep_incomplete",
                        lambda kind=None: ([], [("prior", "stuck__20260917T091000",
                                                 "PermissionError: held open")]))
    scr._sweep(all_kinds=True)
    assert "could not be removed" in scr.status.text() and "held open" in scr.status.text()


def test_note_delete_and_sweep_are_refused_while_a_run_is_live_and_reading_is_not(store):
    """B6's second half, in the window's own wording (model_builder_screen.py:371 and :447). Every
    WRITE is refused while a run is live; READING never is -- which is the whole reason the browser
    is a plain screen and not a BasePanel (B1), so nothing here is greyed out either."""
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="busy")
    leftover = store.kind_dir("prior") / "_unnamed__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    scr.note_edit.setText("written during a train")
    BasePanel._running = True
    try:
        for action in (scr._set_note, scr._delete, lambda: scr._sweep(all_kinds=False)):
            scr._set_status("")
            action()
            assert "A task is running" in scr.status.text(), scr.status.text()
        assert SHOWN == [], "a refused action must not ask or explain in a box"
        assert store.get("prior", p.id).note == "", "the note must not be written"
        assert leftover.is_dir(), "the sweep must not run"
        scr.refresh()                                   # reading is never blocked
        assert scr.table.topLevelItemCount() >= 1
        _select_ref(scr.table, p.id)
        assert scr.note_edit.isEnabled() and scr.btn_delete.isEnabled(), \
            "the browser is not a BasePanel: a live run greys nothing here"
    finally:
        BasePanel._running = False
    # _select above re-read the row's (empty) note into the box, as a selection change must; type it
    # again and the same click works now that nothing is running.
    scr.note_edit.setText("written after the train")
    scr._set_note()
    assert store.get("prior", p.id).note == "written after the train"


def test_a_delete_in_the_browser_reaches_the_three_store_pickers(store, monkeypatch):
    """B8. Without this a picker keeps pointing at a deleted artifact: StorePicker.restore_key
    (artifact_picker.py:226-231) silently does nothing when the saved id has vanished, leaving
    whatever item happens to be current selected -- deliberate for a picker, and a defect the moment
    a browser can delete. Mirrors _refresh_model_combos: the window walks its own panels.

    The wiring (MainWindow._refresh_store_pickers and the store_changed connect) is the screen task's;
    the delete that emits store_changed is this task's. This is the test of the two together."""
    from PySide6.QtWidgets import QMessageBox
    from core.gui.main_window import MainWindow
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import qt_app
    qt_app()
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)
    cfg = _nad_cfg()
    keep = _prior_artifact(store, cfg, name="keep")
    doomed = _prior_artifact(store, cfg, name="doomed", seed=3)
    w = MainWindow()
    try:
        picker = w.inference_screen.prior_panel.prior_picker
        picker.refresh()
        picker.restore_key(doomed.id)
        assert picker.key() == doomed.id
        scr = w.artifact_screen
        _show_kind(scr, "prior")
        _select_ref(scr.table, doomed.id)
        _answer(monkeypatch, QMessageBox.Yes)
        scr._delete()
        ids = [picker.combo.itemData(i) for i in range(picker.combo.count())]
        assert doomed.id not in ids, "the picker still offers the deleted prior"
        assert keep.id in ids and picker.key() != doomed.id
        # and it is all THREE pickers, not just the one that happened to be looked at
        seen = []
        real = StorePicker.refresh
        monkeypatch.setattr(StorePicker, "refresh",
                            lambda self: seen.append(self.kind) or real(self))
        w._refresh_store_pickers()
        assert sorted(seen) == ["observation", "posterior", "prior"], seen
    finally:
        w.close()


def test_the_windows_three_model_dialogs_go_through_the_session_guard(monkeypatch):
    """The window's last three boxes are INSTANCE dialogs, so every box the GUI shows lands in
    SHOWN. main_window.py showed these three with the C++ statics (QMessageBox.warning in
    _edit_user_model and after a failed delete_user_model, QMessageBox.information in
    _delete_user_model's run guard), which escape tests/conftest.py::_no_modal_dialogs -- it
    patches QMessageBox.exec, the instance method. Offscreen a static spins a nested event loop
    nothing ever closes, so a test that reached one STALLED instead of failing, which is why these
    three sites had no test at all. The statics are patched here as a tripwire rather than left
    live: if the conversion is ever undone, this fails naming the site instead of hanging the run.
    """
    from PySide6.QtWidgets import QMessageBox
    from core.Helpers import model_store
    from core.gui.main_window import MainWindow
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import SHOWN, qt_app
    qt_app()
    statics = []
    for name in ("warning", "information"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: statics.append(_n) or QMessageBox.Ok)

    def boom(*a, **k):
        raise RuntimeError("corrupt definition")

    w = MainWindow()
    try:
        SHOWN.clear()                       # this test is about the three sites, not construction
        monkeypatch.setattr(w.model_builder_screen, "load_existing", boom)
        w._edit_user_model("BROKEN")        # site 1 -- a definition that will not load
        monkeypatch.setattr(model_store, "delete_user_model", boom)
        BasePanel._running = True
        try:
            w._delete_user_model("BROKEN")  # site 2 -- refused while a run is live, nothing asked
        finally:
            BasePanel._running = False
        _answer(monkeypatch, QMessageBox.Yes)
        w._delete_user_model("BROKEN")      # the confirmation, then site 3 -- the delete fails
    finally:
        w.close()
    assert statics == [], f"the window still shows a box with a C++ static: {statics}"
    assert [b.windowTitle() for b in SHOWN] == ["Cannot edit model", "A task is running",
                                                "Delete model", "Delete failed"], \
        [b.windowTitle() for b in SHOWN]
    assert SHOWN[0].icon() == QMessageBox.Warning and "corrupt definition" in SHOWN[0].text()
    assert SHOWN[1].icon() == QMessageBox.Information
    assert "before deleting a model" in SHOWN[1].text(), SHOWN[1].text()
    assert SHOWN[3].icon() == QMessageBox.Warning and "corrupt definition" in SHOWN[3].text()


def test_a_leftover_row_is_nothing_to_act_on_and_no_row_at_all_is_a_refusal(store):
    """``_selected``'s two answers, which no other test here reaches.

    An INCOMPLETE directory is not an artifact: the Note box and Delete are DISABLED for it and, if
    a stale click gets through anyway, the status line says there is nothing there and points at the
    sweep -- no rule was broken, so no yellow box. Nothing selected IS a refusal, and it carries the
    field key whose fix sentence names the list (core/gui/fields.py's "artifact").
    """
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    _prior_artifact(store, cfg, name="real")
    leftover = store.kind_dir("prior") / "half_a_run__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    scr = artifact_screen(store)
    _show_kind(scr, "prior")

    # (a) nothing selected -- a refresh leaves no selection (§3.5), so this is the launch state
    assert scr.table.current_summary() is None
    assert not scr.btn_delete.isEnabled() and not scr.note_edit.isEnabled()
    scr._delete()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert box.text() == "No artifact is selected."
    assert box.informativeText() == "Select an artifact in the list on the Artifacts screen."

    # (b) a leftover selected: disabled, and a forced call explains rather than refuses
    s = _select_ref(scr.table, leftover.name)
    assert not s.complete and s.dir_name == leftover.name
    assert not (scr.note_edit.isEnabled() or scr.btn_note.isEnabled()
                or scr.btn_delete.isEnabled()), "an incomplete row is not an artifact"
    assert scr.note_edit.text() == "", "there is no manifest, so there is no note to show"
    for doing, action in (("delete", scr._delete), ("annotate", scr._set_note)):
        SHOWN.clear()
        scr._set_status("")
        action()
        assert SHOWN == [], "nothing was refused, so nothing is explained in a box"
        said = scr.status.text()
        assert leftover.name in said and f"nothing to {doing}" in said, said
        assert "no manifest.json" in said and "a sweep removes" in said, said
    assert leftover.is_dir(), "neither action may touch it"


class _Proxy:
    """Everything the screen asks of a store, delegated to a real one. Subclasses override the one
    method they want to lie about, which keeps each test's lie to a single visible line."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_a_read_failure_after_a_change_beats_the_actions_own_sentence(store, monkeypatch):
    """The re-list has the last word on the status line.

    A delete that SUCCEEDS followed by a re-list that CANNOT READ the store must not report
    "Deleted ...": the table empties, and "there is nothing here" against "I could not look" is the
    one distinction this screen exists to keep apart (§3.2) -- the same distinction Task 9 built
    refresh()'s error branch for. A success sentence painted over it would erase it.

    _after_change owns that order: the action hands it a sentence, and it sets it only when
    refresh() could read the store. Which also makes the wrong order unrepresentable -- no caller
    can write the status line before the re-list any more, because none of them writes it at all.
    """
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="lonely")
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)

    class _ReadsFailOnceTheDeleteLands(_Proxy):
        """The real store until the delete goes through, and unreadable from then on -- a root that
        loses its permissions in the instant between the removal and the re-list."""

        deleted = False

        def delete(self, *a, **k):
            out = self._real.delete(*a, **k)
            self.deleted = True
            return out

        def list(self, kind):
            if self.deleted:
                raise PermissionError("Access is denied")
            return self._real.list(kind)

    proxy = _ReadsFailOnceTheDeleteLands(store)
    monkeypatch.setattr(scr, "_resolved_store", lambda: proxy)
    _answer(monkeypatch, QMessageBox.Yes)
    scr._delete()
    assert proxy.deleted, "the delete itself must have happened; this is about what is SAID"
    said = scr.status.text()
    assert "Could not read the prior artifacts" in said and "PermissionError" in said, said
    assert "Deleted" not in said, f"the success sentence hid the read failure: {said}"
    assert said.startswith("⚠ ") and scr.table.topLevelItemCount() == 0, said

    # (b) the same sentence DOES land when the re-list can read the store: the success message is
    #     withheld only by a failed read, never by the reordering itself.
    q = _prior_artifact(store, cfg, name="second", seed=5)
    monkeypatch.setattr(scr, "_resolved_store", lambda: store)
    scr.refresh()
    _select_ref(scr.table, q.id)
    changed = []
    scr.store_changed.connect(lambda: changed.append(True))
    scr._delete()
    assert scr.status.text() == "Deleted prior second.", scr.status.text()
    assert changed, "the pickers are told either way"


def test_an_unreadable_root_reports_on_the_status_line_instead_of_crashing_a_click(store, monkeypatch):
    """Every store read behind a button follows refresh()'s convention: reported, with its class, on
    the status line. Unguarded, an unreadable root turned a Delete click into an unhandled slot
    exception and the application's last-resort red box -- a crash report for a disk problem.

    Three reads are covered: dependents(), the list() _dependents_refusal does to say WHY each
    dependent depends, and the one _incomplete does for the sweep -- whose sentence now carries a
    next step, the one operator-facing line on this screen that had none.
    """
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="ancestor")
    with store.create("inference", cfg, name="child") as w:
        w.parents = {"prior": p.id}
        w.body = {"results": {}}
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)

    # (a) dependents() itself cannot be read
    class _NoDependentScan(_Proxy):
        def dependents(self, kind, id_):
            raise PermissionError("Access is denied")

    monkeypatch.setattr(scr, "_resolved_store", lambda: _NoDependentScan(store))
    SHOWN.clear()
    scr._delete()
    said = scr.status.text()
    assert "Could not read what depends on prior ancestor" in said, said
    assert "PermissionError" in said and "nothing was deleted" in said, said
    assert SHOWN == [], "a disk problem is not a refusal and asks nothing"
    assert store.get("prior", p.id).name == "ancestor"

    # (b) the dependents are known but the list() that explains WHY cannot be read. Unguarded this
    #     raised from inside _dependents_refusal, i.e. after the refusal was already unavoidable.
    class _NoListForTheWhy(_Proxy):
        def list(self, kind):
            if kind == "inference":
                raise PermissionError("Access is denied")
            return self._real.list(kind)

    monkeypatch.setattr(scr, "_resolved_store", lambda: _NoListForTheWhy(store))
    SHOWN.clear()
    scr._delete()
    assert "Could not read what depends on prior ancestor" in scr.status.text(), scr.status.text()
    assert SHOWN == [] and store.get("prior", p.id).name == "ancestor"

    # (c) the sweep's own read, and its next step
    class _NoListAtAll(_Proxy):
        def list(self, kind):
            raise PermissionError("Access is denied")

    monkeypatch.setattr(scr, "_resolved_store", lambda: _NoListAtAll(store))
    SHOWN.clear()
    scr._sweep(all_kinds=False)
    said = scr.status.text()
    assert SHOWN == [], "there is nothing to confirm when the candidates could not be read"
    assert "The prior directory could not be read" in said and "PermissionError" in said, said
    assert "sweep again" in said, f"the one sentence with no next step: {said}"
    assert said.startswith("⚠ "), said
    assert "posterior" not in said, "a one-kind sweep must not have gone looking at every kind"


def test_sweeping_one_kind_leaves_another_kinds_leftover_alone(store, monkeypatch):
    """Minor 2, behaviourally: "sweep this kind" reads self.kind(), the accessor everything else on
    this screen reads, and only the all-kinds button passes None. While the kind travelled as a
    falsy value, "all seven" was what a falsy kind meant downstream, so a one-kind sweep could
    widen to every directory in the store."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import artifact_screen, qt_app
    qt_app()
    mine = store.kind_dir("prior") / "junk_prior"
    theirs = store.kind_dir("diagnostic") / "junk_diagnostic"
    for d in (mine, theirs):
        d.mkdir(parents=True, exist_ok=True)
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _answer(monkeypatch, QMessageBox.Yes)
    scr._sweep(all_kinds=False)
    assert not mine.exists(), "this kind's leftover should have gone"
    assert theirs.is_dir(), "a one-kind sweep reached into another kind"
    assert "Removed 1 of 1" in scr.status.text(), scr.status.text()
    # ... and the all-kinds button does reach it.
    scr._sweep(all_kinds=True)
    assert not theirs.exists() and "Removed 1 of 1" in scr.status.text(), scr.status.text()


def test_a_typed_note_survives_a_re_list_that_lands_on_the_same_artifact(store):
    """Minor 3. A re-list clears the table's selection before a row is chosen again, so rewriting the
    box on every selection change threw away a note typed and not yet applied even when the same
    artifact came back. Only a change of ARTIFACT repopulates it; a leftover, which has no note at
    all, clears it.

    ``a`` is deliberately the ONLY complete prior while the draft is being typed. ``_select_ref``
    walks the rows in order, and passing over a neighbour on the way is a real change of artifact --
    with a second one present, the helper rather than the re-list would be what cleared the box.
    Rows are newest-first, so every selection below is the single change one click makes.
    """
    from tests._fixtures import artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    a = _prior_artifact(store, cfg, name="alpha")
    leftover = store.kind_dir("prior") / "junk__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    scr = artifact_screen(store)
    _show_kind(scr, "prior")

    assert _select_ref(scr.table, a.id).id == a.id
    scr.note_edit.setText("half-typed, not applied")
    scr.refresh()                                   # selection cleared; the draft must survive it
    assert scr.table.current_summary() is None and not scr.note_edit.isEnabled()
    assert scr.note_edit.text() == "half-typed, not applied", scr.note_edit.text()
    _select_ref(scr.table, a.id)                    # back to the SAME artifact
    assert scr.note_edit.text() == "half-typed, not applied", \
        "the draft was discarded by a re-list that came back to the same artifact"
    assert scr.note_edit.isEnabled()

    # A different artifact DOES repopulate, from its own manifest. Written now, so it is the newest
    # and therefore the first row -- one selection change from here.
    b = _prior_artifact(store, cfg, name="beta", seed=7)
    store.set_note("prior", b.id, "beta's own note")
    scr.refresh()
    _select_ref(scr.table, b.id)
    assert scr.note_edit.text() == "beta's own note", scr.note_edit.text()
    # And a leftover, which has no note to show, clears it.
    _select_ref(scr.table, leftover.name)
    assert scr.note_edit.text() == "" and not scr.note_edit.isEnabled()

    # A note that was WRITTEN shows what was stored, trimmed -- not the untrimmed draft.
    _select_ref(scr.table, a.id)
    scr.note_edit.setText("   trimmed on the way in   ")
    scr._set_note()
    assert store.get("prior", a.id).note == "trimmed on the way in"
    assert scr.note_edit.text() == "trimmed on the way in", scr.note_edit.text()


def test_the_lineage_report_writes_exactly_what_render_lineage_returns(store, monkeypatch, tmp_path):
    """B10 + §5: ONE renderer, and the GUI adds nothing to it -- no header, no banner, no trailing
    newline. Asserted against render_lineage itself rather than against the tool's `artifacts
    summary`, so the two front ends cannot drift: whichever wrote the file, a reviewer gets the same
    bytes. UTF-8 with LF endings, so 'the same bytes' is true on this platform too."""
    from PySide6.QtWidgets import QFileDialog
    from core.artifacts import render_lineage
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    ancestor = _prior_artifact(store, cfg, name="ancestor")
    with store.create("inference", cfg, name="descendant") as w:
        w.parents = {"prior": ancestor.id}
        w.body = {"results": {"n_samples": 8}}
    scr = artifact_screen(store)
    _show_kind(scr, "inference")
    _select_ref(scr.table, w.id)
    out = tmp_path / "lineage.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    scr._lineage_report()
    assert out.read_bytes() == render_lineage(store, "inference", w.id).encode("utf-8")
    text = out.read_text(encoding="utf-8")
    assert "ancestor" in text and ancestor.id in text, "the chain must reach the parent"
    assert SHOWN == [], "writing a report asks nothing"
    assert str(out) in scr.status.text(), scr.status.text()
    # It is a FILE and never an eighth kind: the store's seven directories gained nothing.
    assert [s.id for s in store.list("inference")] == [w.id]
    assert not (store.root / "reports").exists()


def test_the_lineage_report_is_a_read_and_needs_a_complete_row(store, monkeypatch, tmp_path):
    """Three legs of one rule. A report is a READ, so a live run does not refuse it (B6) -- unlike
    Task 11's note, delete and sweep. A leftover directory has no manifest and so no lineage. And a
    name the operator typed without an extension gets .txt rather than a file nothing opens."""
    from PySide6.QtWidgets import QFileDialog
    from core.artifacts import render_lineage
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="ancestor")
    leftover = store.kind_dir("prior") / "_unnamed__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    # A directory of its own: tmp_path also holds this test's prism.ini (_isolated_settings) and the
    # store fixture's Artifacts/, so it is not a place to count files in.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    bare = out_dir / "chain"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(bare), ""))
    BasePanel._running = True
    try:
        scr._lineage_report()
    finally:
        BasePanel._running = False
    written = bare.with_suffix(".txt")
    assert written.read_bytes() == render_lineage(store, "prior", p.id).encode("utf-8")
    assert not bare.exists(), "an extension-less name must gain .txt, not be written as given"
    # a leftover directory: no manifest, no lineage, and no file
    scr._set_status("")
    _select_ref(scr.table, leftover.name)
    scr._lineage_report()
    assert "no usable manifest" in scr.status.text(), scr.status.text()
    assert "lineage" in scr.status.text(), scr.status.text()
    # nothing selected at all: the fielded refusal, and still no file
    scr.table.clearSelection()
    scr.table.setCurrentItem(None)
    scr._lineage_report()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs"
    assert box.informativeText() == "Select an artifact in the list on the Artifacts screen."
    assert sorted(q.name for q in out_dir.iterdir()) == ["chain.txt"]


def test_the_lineage_button_is_disabled_without_a_complete_row(store):
    """Fix round 1: btn_lineage must track the same condition its own handler guards on, so the
    button and the guard cannot disagree -- unlike btn_save, which is enabled off the pane's TEXT
    (something an incomplete row also has, its reason and its path), a lineage walk needs a
    resolvable artifact to start from. So this follows btn_delete/note_edit's rule (`live`) rather
    than btn_save's: disabled with nothing selected, disabled on a leftover, enabled on a complete
    row."""
    from tests._fixtures import artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="ancestor")
    leftover = store.kind_dir("prior") / "_unnamed__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    scr = artifact_screen(store)
    _show_kind(scr, "prior")

    # (a) nothing selected -- a refresh leaves no selection (§3.5), so this is the launch state
    assert scr.table.current_summary() is None
    assert not scr.btn_lineage.isEnabled()

    # (b) a complete row -- the button can act
    _select_ref(scr.table, p.id)
    assert scr.btn_lineage.isEnabled()

    # (c) a leftover -- the pane still has text (its reason, its path), so Save stays enabled, but
    # there is no artifact to walk, so lineage is disabled -- matching _lineage_report's own "has no
    # usable manifest" branch rather than Save's rule.
    _select_ref(scr.table, leftover.name)
    assert scr.detail.toPlainText(), "an incomplete row still has pane text"
    assert scr.btn_save.isEnabled(), "Save's own rule is unaffected by this fix"
    assert not scr.btn_lineage.isEnabled()

    # back to the complete row re-enables it
    _select_ref(scr.table, p.id)
    assert scr.btn_lineage.isEnabled()
