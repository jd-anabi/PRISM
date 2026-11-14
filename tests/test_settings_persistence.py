"""QSettings round-trips, layout geometry, and the config-tab hardware fields. Split from test_gui_progress.py; run directly: pytest tests/test_settings_persistence.py"""
"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  pytest tests/test_settings_persistence.py
"""
import os
import tempfile
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

import torch                                                      # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from tests._fixtures import (build_inference, build_part, build_run_host, code_only, pump, qt_app,  # noqa: E402
                             write_simulation_set)
import contextlib                                                  # noqa: E402
import pytest                                                      # noqa: E402

def test_missing_picker_key_restores_to_default_not_blank():
    """A saved selection whose file is gone must leave the picker at its default, never blank it via
    setCurrentIndex(-1)."""
    from core.config import CELL_PATH
    from core.gui.widgets.artifact_picker import ArtifactPicker

    qt_app()
    picker = ArtifactPicker(CELL_PATH / "nadrowski")
    picker.restore_key("nadrowski/does_not_exist.txt")
    assert picker.combo.currentIndex() >= 0, "a stale key blanked the combo"
    picker.deleteLater()

def test_panel_splitter_is_sized_and_not_collapsible():
    """Every panel opens with a usable controls column that cannot be dragged to nothing.

    BasePanel builds the app's only QSplitter and there are nine live instances. It used to be a
    LOCAL with no setSizes and no setChildrenCollapsible, so every launch started at the minimum and
    one slip past the left edge collapsed the controls to zero width, recoverable only by finding a
    5px handle at x=0.
    """
    app = qt_app()

    panel = build_run_host()
    panel.resize(1300, 820)
    pump(app, 0.15)
    assert not panel.splitter.childrenCollapsible(), \
        "the controls column can still be collapsed to zero width"
    assert all(s > 0 for s in panel.splitter.sizes()), \
        f"splitter opened with a zero-width pane: {panel.splitter.sizes()}"
    # The old hard 460px cap could not be escaped by widening the window, so any wider form got a
    # permanent horizontal scrollbar in the left column.
    assert panel.controls_scroll.maximumWidth() > 1000, \
        f"controls column is still hard-capped at {panel.controls_scroll.maximumWidth()}px"

def test_panel_layout_round_trips_through_settings():
    """The splitter position must survive a restart -- 'I have to re-drag it every launch' is the
    complaint. Uses saveState/restoreState, which work before the widget is shown or polished."""
    app = qt_app()
    from core.gui import settings as st

    a = build_run_host()
    a.resize(1300, 820)
    pump(app, 0.15)
    a.splitter.setSizes([500, 700])
    pump(app, 0.05)
    want = a.splitter.sizes()
    qs = st.settings()
    a.save_layout(qs)
    qs.sync()

    b = build_run_host()                      # restore_layout runs in __init__
    b.resize(1300, 820)
    pump(app, 0.15)
    assert b.splitter.sizes() == want, \
        f"layout did not round-trip: saved {want}, restored {b.splitter.sizes()}"

def test_forms_grow_their_fields_and_numeric_boxes_have_a_floor():
    """Pins both halves of the 'input boxes are cut off' fix, across every form in every panel.

    Qt's Windows default is FieldsStayAtSizeHint: the field takes its size hint and stops, so numeric
    boxes rendered 3-6 characters wide and typing "0.033333" scrolled inside the box. Repo-wide there
    were ZERO setFieldGrowthPolicy calls over 18 QFormLayout sites. Asserting over discovered forms
    (not a fixed list) means a newly added form is covered too.
    """
    from PySide6.QtWidgets import QFormLayout, QLineEdit
    from core.gui.widgets.labeled_inputs import FloatField, IntField

    qt_app()
    inference = build_inference()
    panels = [inference.infer_panel, inference.tsnpe_panel]
    forms = [f for p in panels for f in p.findChildren(QFormLayout)]
    assert forms, "no QFormLayouts discovered — the test is not exercising anything"
    bad = [f for f in forms
           if f.fieldGrowthPolicy() != QFormLayout.AllNonFixedFieldsGrow]
    assert not bad, f"{len(bad)}/{len(forms)} forms do not grow their fields"

    narrow = [w for p in panels for w in p.findChildren(QLineEdit)
              if isinstance(w, (FloatField, IntField)) and w.minimumWidth() < 80]
    assert not narrow, \
        f"{len(narrow)} numeric field(s) have no usable minimum width (e.g. " \
        f"{narrow[0].minimumWidth()}px)"

def test_long_diagnostics_are_readable_without_horizontal_scrolling():
    """Panels emit long single-line diagnostics; the log pane must wrap them."""
    from PySide6.QtWidgets import QPlainTextEdit
    from core.gui.widgets.log_pane import LogPane

    qt_app()
    assert LogPane().lineWrapMode() == QPlainTextEdit.WidgetWidth, \
        "log pane still truncates long lines instead of wrapping them"

def test_the_memory_limit_is_remembered_per_computer_and_still_read_live(monkeypatch):
    """The memory limit is set in Settings and remembered for this computer, as text, when a change of it is accepted,
    and it comes back at the next launch: it belongs to the card, and a limit forgotten at every launch let a
    full-size generation die at batch 237.

    A PLAIN ASSIGNMENT IS ENOUGH, unlike every other knob the window offers. The sweep and flow fields had to become
    ARGUMENTS threaded into build_prior/build_posterior, because orchestrator does `from .config import ...` and binds
    them at import, so writing to the constant is a silent no-op (an imported name is a snapshot).
    pipeline.vram_ceiling_gib() reads the module each time the planner asks, so this one takes effect, with no plumbing,
    and this test would catch it if that ever changed."""
    import socket
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from core import config as _cfg
    from core.SBI import pipeline as _pipe
    from core.gui import memory_limit, settings as st
    from core.gui.screens.settings_screen import SettingsScreen

    monkeypatch.delenv(_pipe.VRAM_CEILING_ENV, raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "LAB-PC")
    assert _cfg.SIM_VRAM_CEILING_GIB == 0.0, "it must default to no limit"
    screen = build_part(SettingsScreen, on_mode=lambda _mode: None)
    assert (screen.no_limit.isChecked(), screen.limit_field.text()) == (True, "")

    screen.limited.click()
    QTest.keyClicks(screen.limit_field, "6")
    assert _cfg.SIM_VRAM_CEILING_GIB == 0.0, "judged when editing ends, never on a keystroke"
    QTest.keyClick(screen.limit_field, Qt.Key_Return)
    assert _cfg.SIM_VRAM_CEILING_GIB == 6.0, "the field must assign the config constant"
    assert st.settings().value("computer/LAB-PC/memory_limit_gib") == "6"
    assert _pipe.vram_ceiling_gib() == 6.0, "and the planner must read it live"
    # 2 GiB, not 6: the budget is a min() and the CPU branch of memory_budget_elements caps at 4 GiB, so a ceiling
    # above that would not bind and the assertion would prove nothing.
    memory_limit.set_memory_limit(2.0)
    saved_cap = _pipe._BUDGET_CAP_ELEMENTS
    monkeypatch.setattr(_pipe, "_BUDGET_CAP_ELEMENTS", None)        # the other term in the same min()
    assert _pipe.sim_memory_budget_elements(torch.device("cpu"), torch.float32) == (2 * 2 ** 30) // 4, (
        "the ceiling must bind the budget the planner actually uses")
    monkeypatch.setattr(_pipe, "_BUDGET_CAP_ELEMENTS", saved_cap)

    memory_limit.set_memory_limit(6.0)
    _cfg.SIM_VRAM_CEILING_GIB = 0.0                                 # as the next launch finds it
    memory_limit.apply_remembered()
    assert _cfg.SIM_VRAM_CEILING_GIB == 6.0, "the remembered limit did not come back"
    screen.no_limit.click()
    assert (_cfg.SIM_VRAM_CEILING_GIB, st.settings().value("computer/LAB-PC/memory_limit_gib")) == (0.0, "none")
    memory_limit.apply_remembered()
    assert _cfg.SIM_VRAM_CEILING_GIB == 0.0, "no limit is a choice remembered too"


@pytest.mark.parametrize("constant", (0.0, 4.0))
@pytest.mark.parametrize("raw, a_number", ((None, False), ("", False), (" ", False), ("6", True), ("6.5", True),
                                           ("6,5", False), ("abc", False), ("0", True), ("-2", True), ("nan", True),
                                           ("inf", True), ("1e999", True)))
def test_the_limit_in_force_is_the_planners_own_answer_over_a_table_of_texts(constant, raw, a_number, monkeypatch):
    """What the window says is in force is what the planner will plan with, whatever the variable of the environment
    holds, and it says where that limit was set by the planner's own test of the variable: a number there counts, one
    the planner reads as no limit among them, and anything else is passed over. Reading it logs nothing, though it is
    read at every draw of a page and the planner logs a warning at each call for a variable that is no number."""
    from core import config
    from core.SBI import pipeline
    from core.gui import memory_limit
    if raw is None:
        monkeypatch.delenv(pipeline.VRAM_CEILING_ENV, raising=False)
    else:
        monkeypatch.setenv(pipeline.VRAM_CEILING_ENV, raw)
    monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", constant)
    said = []
    monkeypatch.setattr(pipeline.log, "warning", lambda *args, **kwargs: said.append(args))
    force = memory_limit.memory_limit_in_force()
    assert said == [], "reading the limit logged"
    planner = pipeline.vram_ceiling_gib()
    assert len(said) == (raw in ("6,5", "abc")), "the planner's own warning is what the reading must not repeat"
    assert force.gib == (planner if planner > 0 else None), (raw, constant)
    assert force.source == ("environment" if a_number else "setting" if constant > 0 else "none"), (raw, constant)
    assert force.raw == (raw or "")


def test_one_module_of_the_window_assigns_the_memory_limit():
    """The memory limit is the one setting a run reads from the configuration while it runs, and the window writes it
    in one place: no other file under the window's folder assigns the constant, imports it by name or names it in a
    string."""
    import ast
    gui = Path(__file__).resolve().parents[1] / "core" / "gui"
    found = []
    for path in sorted(gui.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            assigned = (isinstance(node, ast.Attribute) and node.attr == "SIM_VRAM_CEILING_GIB"
                        and isinstance(node.ctx, ast.Store))
            named = isinstance(node, ast.Constant) and node.value == "SIM_VRAM_CEILING_GIB"
            imported = isinstance(node, ast.alias) and node.name == "SIM_VRAM_CEILING_GIB"
            if assigned or named or imported:
                found.append(path.relative_to(gui).as_posix())
    assert sorted(set(found)) == ["memory_limit.py"], found


def test_a_limit_remembered_for_another_computer_is_not_applied(monkeypatch):
    """A settings file carried to another computer applies no limit there: the key holds the name of the computer it
    was set on. A computer with no name to give keeps its limit under a key of its own, and that key is no other
    computer's."""
    import socket
    from core import config
    from core.gui import memory_limit, settings as st

    def no_name():
        raise OSError("this computer has no name")

    written = st.settings()
    written.setValue("computer/OTHER-PC/memory_limit_gib", "6")
    written.setValue("computer/unnamed/memory_limit_gib", "3")
    written.sync()
    monkeypatch.setattr(socket, "gethostname", lambda: "LAB-PC")
    monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", 0.0)
    assert memory_limit.stored_limit() == (None, True)
    memory_limit.apply_remembered()
    assert config.SIM_VRAM_CEILING_GIB == 0.0, "the limit of another computer was applied"
    memory_limit.set_memory_limit(2.5)
    assert memory_limit.stored_limit() == (2.5, True)
    kept = st.settings()
    assert (kept.value("computer/LAB-PC/memory_limit_gib"), kept.value("computer/OTHER-PC/memory_limit_gib"),
            kept.value("computer/unnamed/memory_limit_gib")) == ("2.5", "6", "3"), "the limit went under another key"
    monkeypatch.setattr(socket, "gethostname", no_name)
    assert memory_limit.computer() == ""
    assert memory_limit.stored_limit() == (3.0, True), "a computer with no name reads the key of no name"
    memory_limit.set_memory_limit(1.5)
    assert st.settings().value("computer/unnamed/memory_limit_gib") == "1.5"


@pytest.mark.parametrize("written", ("6,5", "abc", "", "0", "-2", "inf", "nan", "1e999", "1e300", "2e299"))
def test_a_remembered_limit_that_cannot_be_read_is_no_limit_and_says_so(written, monkeypatch):
    """A remembered limit edited by hand into what is no limit one can set is no limit: nothing is assigned, and
    Settings says so where it says what is in force. The file is written as a person writes it, so that the settings
    library reads the value as it reads such a line. Nothing is rewritten by reading it."""
    import socket
    from core import config
    from core.SBI import pipeline
    from core.gui import memory_limit, settings as st
    from core.gui.screens.settings_screen import SettingsScreen
    monkeypatch.delenv(pipeline.VRAM_CEILING_ENV, raising=False)
    monkeypatch.setattr(socket, "gethostname", lambda: "LAB-PC")
    path = Path(st.settings().fileName())
    held = f"[computer]\nLAB-PC\\memory_limit_gib={written}\n".encode("utf-8")
    path.write_bytes(held)
    assert memory_limit.stored_limit() == (None, False)
    monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", 2.5)
    memory_limit.apply_remembered()
    assert config.SIM_VRAM_CEILING_GIB == 2.5, "nothing is assigned"
    screen = build_part(SettingsScreen, on_mode=lambda _mode: None)
    assert screen.in_force.plain_text() == "In force: 2.5 GiB.", "the line says what is in force, whatever is held"
    monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", 0.0)
    screen = build_part(SettingsScreen, on_mode=lambda _mode: None)
    assert screen.in_force.plain_text() == "The remembered limit could not be read; no limit is in force."
    assert screen.no_limit.isChecked(), "no limit is what is in force"
    monkeypatch.setenv(pipeline.VRAM_CEILING_ENV, "6.5")
    assert build_part(SettingsScreen, on_mode=lambda _mode: None).in_force.plain_text() == (
        "Set outside PRISM by PRISM_VRAM_CEILING_GIB = 6.5. It takes precedence; the remembered limit could not be read.")
    monkeypatch.delenv(pipeline.VRAM_CEILING_ENV)
    assert path.read_bytes() == held, "reading the limit wrote the file"
    memory_limit.set_memory_limit(6.0)
    assert memory_limit.stored_limit() == (6.0, True), "a new limit puts the file right"


def test_a_remembered_limit_of_none_and_no_key_at_all_are_no_limit_read(monkeypatch):
    """No key, and the word for no limit, are both no limit and neither is a limit that could not be read. Neither
    assigns anything as the program starts."""
    from core import config
    from core.gui import memory_limit, settings as st
    assert memory_limit.stored_limit() == (None, True)
    monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", 2.5)
    memory_limit.set_memory_limit(None)
    assert config.SIM_VRAM_CEILING_GIB == 0.0
    assert st.settings().value(f"computer/{memory_limit.computer()}/memory_limit_gib") == "none"
    assert memory_limit.stored_limit() == (None, True)
    monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", 2.5)
    memory_limit.apply_remembered()
    assert config.SIM_VRAM_CEILING_GIB == 2.5, "no limit assigns nothing"


def test_a_limit_as_large_as_the_planner_can_read_is_accepted(monkeypatch):
    """A value is too large when the planner cannot turn it into a count of bytes, and no sooner: what it can is a limit,
    in the field and in the file, and the planner takes it."""
    from core.SBI import pipeline
    from core.gui import memory_limit
    monkeypatch.delenv(pipeline.VRAM_CEILING_ENV, raising=False)
    assert memory_limit.judged("1e299", memory_limit.memory_limit_in_force()) == (1e299, "")
    memory_limit.set_memory_limit(1e299)
    assert memory_limit.stored_limit() == (1e299, True)
    assert pipeline.sim_memory_budget_elements(torch.device("cpu"), torch.float32) > 0


def test_put_many_writes_every_pair_and_answers_for_the_file(monkeypatch):
    """Every pair goes in one write. Pairs that cannot be gone through, or a file that refuses them, answer False and
    are said in the window's log; nothing is raised into what was being done."""
    import stat
    from core.gui import remembered, settings as st
    said = []
    monkeypatch.setattr(remembered.log, "warning", lambda *args: said.append(args))
    assert remembered.put_many([("here/one", "1"), ("here/two", "2")]) is True
    kept = st.settings()
    assert (kept.value("here/one"), kept.value("here/two"), said) == ("1", "2", [])

    def pairs():
        yield "here/three", "3"
        raise OSError("the pairs ran out")

    assert remembered.put_many(pairs()) is False
    assert len(said) == 1 and "the pairs ran out" in said[0][-1], said
    path = Path(st.settings().fileName())
    os.chmod(path, stat.S_IREAD)
    try:
        assert remembered.put_many([("here/four", "4")]) is False
        assert remembered.put("here/five", "5") is False
    finally:
        os.chmod(path, stat.S_IWRITE)
    assert len(said) == 3, said


@pytest.mark.parametrize("constant", ("not-a-number", None, -1.0))
def test_a_constant_that_is_no_limit_is_read_as_the_planner_reads_it(constant, monkeypatch):
    """The constant is read here as the planner reads it, and quietly: what is no number, nothing and a negative are no
    limit, and nothing is raised at a draw."""
    from core import config
    from core.SBI import pipeline
    from core.gui import memory_limit
    monkeypatch.delenv(pipeline.VRAM_CEILING_ENV, raising=False)
    monkeypatch.setattr(config, "SIM_VRAM_CEILING_GIB", constant)
    force = memory_limit.memory_limit_in_force()
    assert (force.gib, force.source, memory_limit.chosen()) == (None, "none", None), constant
    assert pipeline.vram_ceiling_gib() == 0.0


def test_the_free_vram_readout_does_not_use_mem_get_info():
    """⚠ The readout beside the ceiling must come from nvidia-smi, never torch.cuda.mem_get_info.

    That reading overstates free VRAM on Windows by roughly the size of the desktop -- measured
    15037 MiB against nvidia-smi's 5814 at the same instant -- and it is the number that
    green-lit the batch which killed the first chi retrain. Printing it next to a field whose entire
    purpose is to bound VRAM would hand the user the exact lie the field defends against."""
    from core.gui.panels import inference_tabs as it
    src = code_only(it._nvidia_smi_free_gib)
    assert "mem_get_info" not in src, "the readout must not use the optimistic driver reading"
    assert "nvidia-smi" in src, "the readout must come from nvidia-smi"
    got = it._nvidia_smi_free_gib()
    assert got is None or got >= 0.0, f"unexpected reading {got!r}"

def test_a_one_field_near_miss_blocks_a_fresh_run_until_confirmed():
    """A passive status line was not enough, three times over.

    `_budget_checkpoint` already said "these settings match no checkpoint, so this starts a NEW run"
    and named the differing field -- but it is a label on a tab the user has scrolled past by the
    time they press Train. It failed to prevent 884 batches being lost outright on 2026-08-27, and a
    3989-batch checkpoint being abandoned twice on 2026-08-28. So a run that would start from zero
    while a committed checkpoint sits ONE identity field away now has to be confirmed.

    Narrow on purpose: exactly one differing field is the signature of an accident. Two or more is
    usually a genuinely different experiment, and warning there would make this noise."""
    from core.SBI import training_checkpoint as tc

    base = {"format": "training-rows", "model": "NADROWSKI", "n_runs": 10000,
            "run_size": 2048, "prior_fingerprint": "aaaa", "chi_mode": True}
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        # A committed sibling that differs in exactly ONE field.
        sib = dict(base, prior_fingerprint="bbbb")
        d = write_simulation_set(root, sib, 3989)

        near = tc.near_miss_siblings(base, root)
        assert len(near) == 1, f"expected the one-field sibling, got {near}"
        assert near[0]["field"] == "prior_fingerprint" and near[0]["batches"] == 3989, near[0]

        # TWO differing fields is a different experiment -- it must NOT warn.
        far = dict(base, prior_fingerprint="cccc", n_runs=5000)
        d2 = write_simulation_set(root, far, 500)
        names = {r["name"] for r in tc.near_miss_siblings(base, root)}
        assert d2.name not in names, "a two-field difference must not be reported as a near miss"

        # An identity that MATCHES a checkpoint is a resume, so there is nothing to confirm.
        assert tc.near_miss_siblings(sib, root) == [] or             all(r["name"] != d.name for r in tc.near_miss_siblings(sib, root)),             "a checkpoint must never be a near miss of itself"

def test_what_a_page_this_window_no_longer_has_wrote_is_read_by_nothing_and_left_as_it_was(monkeypatch):
    """A settings file an older PRISM wrote holds the groups of pages this window no longer has -- the effective
    temperature's two pages, the reduction map, the live simulation -- and their panels' layouts. The window opens over
    it, and every part is opened, as over a file without them: no box of any page shows a value of theirs; and closing
    the window, which writes what the parts opened remember, writes none of their keys and deletes none: each is in
    the file as it was, value for value."""
    from PySide6.QtCore import QByteArray, QSettings
    from PySide6.QtWidgets import QLineEdit
    from core.gui import keys
    from core.gui import settings as st
    from tests._fixtures import build_window
    qt_app()
    old = {"fdt/cell": "nadrowski/aaa_left_by_an_older_prism.txt", "fdt/n_freqs": "7731", "fdt/seed": "4242",
           "crossval/preset": "production", "crossval/s_lo": "0.5125",
           "reduction/cell": "nadrowski/aaa_left_by_an_older_prism.txt", "reduction/f0": "0.0731",
           "simulate/cell": "nadrowski/aaa_left_by_an_older_prism.txt", "simulate/t_obs": "12.75",
           "simulate/fps": "24", "layout/ReductionPanel/splitter": QByteArray(b"\x00\x00\x00\xff kept"),
           "layout/SimulatePanel/results": QByteArray(b"\x00\x00\x00\xff kept too")}
    held = st.settings()
    for key, value in old.items():
        held.setValue(key, value)
    held.sync()
    path = held.fileName()
    written = []
    kept = QSettings.setValue
    monkeypatch.setattr(QSettings, "setValue", lambda qs, key, value: (
        written.append(f"{qs.group()}/{key}" if qs.group() else key), kept(qs, key, value))[1])
    window = build_window()
    for place in keys.PARTS:
        window.go_to(place.key)
    shown = {box.text() for box in window.findChildren(QLineEdit)}
    assert not shown & {"7731", "4242", "0.5125", "0.0731", "12.75"}, "a page shows what a removed page wrote"
    window.close()
    assert not set(written) & set(old), f"the close wrote a removed page's key: {sorted(set(written) & set(old))}"
    after = QSettings(path, QSettings.IniFormat)
    assert {key: after.value(key) for key in old} == old, "a removed page's key was deleted or changed"


def test_the_written_keys_are_exactly_the_remembered_table(tmp_path, monkeypatch):
    """What the window writes of steps 1 to 3 into the settings file, each key at its moment and at no other: the model
    and the kind of observation as step 1 is applied; the bounds file, the model it came into use with and the prior as a
    prior's run returns it; the batches and the simulations per batch as a training starts, as the text of their boxes;
    the posterior as a training's run returns it; the mode, the cell file with its model or the recordings, and the
    driven recordings as an inference starts; the observation, the batches and the simulations per batch as a narrowing
    round starts, and its posterior as its run returns it; and at each of these moments the plan last used with the
    model and when, blank with no plan in use. Nothing is written as a choice is made or a button is pressed before a run
    starts, nor as a plan is filled; and closing writes the window's size and place and what Records keeps, never a key
    of the steps nor of the plans."""
    from types import SimpleNamespace
    from PySide6.QtCore import QSettings
    from core import orchestrator
    from core.artifacts import ArtifactStore, default_store, use_store
    from core.gui import settings as st
    from core.gui.run.controller import controller
    from tests._fixtures import _nad_cfg, _prior_artifact, build_run_host, build_window, write_observation
    from tests._runs import Clock, StandIn, at, controller_on, running, turn_until
    app = qt_app()
    written = []
    kept = QSettings.setValue
    monkeypatch.setattr(QSettings, "setValue", lambda qs, key, value: (
        written.append(f"{qs.group()}/{key}" if qs.group() else key), kept(qs, key, value))[1])

    def at_the_moment():
        moment = sorted(set(written))
        written.clear()
        return moment

    plans = ["plans/last_used/nadrowski", "plans/last_used_at/nadrowski"]
    with use_store(ArtifactStore(tmp_path / "Artifacts")):
        controller_on(tmp_path, Clock(at(8, 12, 4)))
        window = build_window()
        for part in ("records", "inference"):
            window.part(part)
        screen = window.part("inference")
        first, prior_page, page = screen.config_panel, screen.prior_panel, screen.posterior_panel
        first.kind_choice.setCurrentIndex(first.kind_choice.findData(False))
        assert at_the_moment() == [], "a choice of step 1 was written"
        first.fill_button.click()
        assert at_the_moment() == [], "a fill was written"
        first.foot.main_button.click()
        assert at_the_moment() == ["inference_config/chi_mode", "inference_config/model", *plans]
        assert [st.settings().value(key) for key in plans] == ["", ""], "a plan was remembered with none in use"
        prior_page.show_view("build")
        prior_page.bounds_picker.restore_key("master.txt")
        prior_page.names.name.setText("retrain_prior")
        prior_page.read_facts()
        built = StandIn(word="Prior build")
        monkeypatch.setattr(orchestrator, "build_prior", built)
        prior_page.main_pressed()
        assert built.started.wait(5) and at_the_moment() == [], "a prior's build wrote as it started"
        made = _prior_artifact(default_store(), _nad_cfg(), name="retrain_prior").id
        built.release(default_store().load_prior(_nad_cfg(), made))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert at_the_moment() == ["inference_prior/bounds", "inference_prior/bounds_model", "inference_prior/prior",
                                   *plans]
        with running(build_run_host()):
            page.names.name.setText("refused_while_live")
            page.main_pressed()                      # another run is live: nothing starts, and nothing is written
            assert at_the_moment() == [], "a training that did not start wrote its boxes"
        page.names.name.setText("retrain")
        page.form.rows["num_runs"].widget.setText("1234")
        page.read_facts()
        assert at_the_moment() == [], "the training's boxes were written as they were typed"
        trained = StandIn(word="Training")
        monkeypatch.setattr(orchestrator, "build_posterior", trained)
        page._main_clicked()
        assert trained.started.wait(5)
        assert at_the_moment() == ["inference_posterior/num_runs", "inference_posterior/run_size_cap", *plans]
        held = st.settings()
        assert (held.value("inference_posterior/num_runs"), held.value("inference_posterior/run_size_cap")) == (
            "1234", "0"), "the training's boxes were not written as their text"
        trained.release(SimpleNamespace(kind="posterior", id="20261008T130000", name="retrain", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert at_the_moment() == ["inference_posterior/posterior", *plans]
        inferred = screen.infer_panel
        window.go_to("inference", "infer")
        inferred.cell_picker.restore_key("master_spont.txt")
        inferred.form.rows["t_obs"].widget.setText("4.5")
        inferred.names.other_name.setText("cell_a")
        inferred.read_facts()
        assert at_the_moment() == [], "Inference's choices were written as they were made"
        inference = StandIn(word="Inference")
        monkeypatch.setattr(orchestrator, "simulated_inference", inference)
        inferred._main_clicked()
        assert inference.started.wait(5), inferred.foot.reason.text()
        assert at_the_moment() == ["inference_infer/cell", "inference_infer/cell_model", "inference_infer/infer_mode",
                                   *plans]
        assert st.settings().value("inference_infer/cell_model") == "NADROWSKI"
        inference.release((SimpleNamespace(kind="observation", id="20261008T140000", name="cell_a", manifest=None,
                                           path=None),
                           SimpleNamespace(kind="inference", id="20261008T140001", name="", manifest=None, path=None,
                                           results={})))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert at_the_moment() == [], "an inference's end wrote what it started with again"
        narrowed = screen.tsnpe_panel
        observed = write_observation(default_store(), screen.session.cfg, name="cell_b")
        monkeypatch.setattr(ArtifactStore, "load_observation", lambda store, cfg, ref: SimpleNamespace(
            kind="observation", id=ref, name="cell_b"))
        window.go_to("inference", "narrowing")
        narrowed.read_facts()
        narrowed.obs_picker.restore_key(observed)
        narrowed.names.name.setText("retrain_round1")
        narrowed.form.rows["num_runs"].widget.setText("321")
        narrowed.read_facts()
        assert at_the_moment() == [], "the round's choices were written as they were made"
        rounds = StandIn(word="Narrowing round")
        monkeypatch.setattr(orchestrator, "tsnpe_round", rounds)
        narrowed._main_clicked()
        assert rounds.started.wait(5), narrowed.foot.reason.text()
        assert at_the_moment() == ["inference_tsnpe/num_runs", "inference_tsnpe/observation",
                                   "inference_tsnpe/run_size_cap", *plans]
        held = st.settings()
        assert [held.value(f"inference_tsnpe/{key}") for key in ("observation", "num_runs", "run_size_cap")] == [
            observed, "321", "0"], "the round's choices were not written as they started"
        rounds.release(SimpleNamespace(kind="posterior", id="20261008T150000", name="retrain_round1", manifest=None))
        turn_until(app, lambda: not controller().is_live(), 10.0)
        assert at_the_moment() == ["inference_posterior/posterior", *plans]
        window._save_state()
        closing = at_the_moment()
        groups = {key.split("/")[0] for key in closing}
        assert "window/geometry" in closing
        assert groups <= {"window", "artifacts"}, sorted(groups)



# ── the shared test helpers ──────────────────────────────────────────────────────────────────────
def test_code_only_strips_docstrings_and_comments_at_every_depth_and_accepts_a_module():
    """The AST pins forbid words that the docstrings EXPLAINING the rule necessarily contain, so the
    helper must drop every docstring -- nested functions and classes too, not only the outermost one
    (test_nav_and_gating's local _stripped dropped only that) -- and comments, which ast.unparse never
    carries. One helper for the four copies the suites grew (two _strip_docstrings, _code_only,
    _unparsed); and it takes a module, so the cadence pin in test_user_sbi no longer re-reads
    pipeline.py by hand."""
    from tests._fixtures import code_only

    def outer():
        """DOCSTRING_OUTER says forbidden_word"""
        # COMMENT says forbidden_word
        def inner():
            """DOCSTRING_INNER says forbidden_word"""
            return "kept_string"

        class K:
            """DOCSTRING_CLASS says forbidden_word"""

        return inner, K

    src = code_only(outer)
    for gone in ("DOCSTRING_OUTER", "DOCSTRING_INNER", "DOCSTRING_CLASS", "COMMENT", "forbidden_word"):
        assert gone not in src, src
    assert "kept_string" in src and "def inner" in src and "class K" in src, src

    from core.gui import settings as st
    mod = code_only(st)
    assert "def use_ini_file(" in mod and "One QSettings store" not in mod, mod[:200]


@pytest.fixture(scope="module")
def _panel_built_at_module_scope(tmp_path_factory):
    """A panel built at MODULE scope, the way tiny_run builds its run and screen_run builds its
    screen. pytest sets module fixtures up before the function-scoped ones, so this runs with
    whatever settings path the SESSION left -- the case a function-only isolation misses. It hands the
    tests the panel, the path it saw, and the session's temp root for the test to check against, and
    lets go of the panel when the module ends, by its own deferred delete, so that it outlives the
    module in no set a run locks."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from core.gui import settings as st
    qt_app()
    seen = st._override_path
    # A bare page that every run locks and frees, as every page of the window is; it reads the settings as a page's
    # own __init__ ends by doing. A step of the inference screen would not do: one outside its screen has no session,
    # and the end of every run in this module, which frees each page still enrolled, would fail on it.
    panel = build_run_host()
    panel.restore_settings(st.settings())
    yield panel, seen, tmp_path_factory.getbasetemp()
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(panel, QEvent.DeferredDelete)


def test_a_module_scoped_fixture_reads_the_session_ini_and_never_the_real_one(_panel_built_at_module_scope):
    """The settings location is never the real PRISM.ini during a test process, AT ANY FIXTURE
    SCOPE. The old per-test redirect (_temp_settings) covered a test body only; a panel built
    by a module fixture was set up before it and read the developer's last session -- the same
    restore-wins-over-config mechanism that cost a ~5-day run on 2026-08-19, now with the answer
    depending on which test ran first. Two paths are asserted: the one the module fixture saw (the
    session file, tests/conftest.py::_settings_home) and the one this body sees (its own fresh file,
    _isolated_settings); both under the session temp, neither the user-scope file, and the fresh one
    not yet on disk -- QSettings creates it, and an empty file is not the same thing as no file."""
    from PySide6.QtCore import QSettings
    from core.gui import settings as st

    _panel, seen, basetemp = _panel_built_at_module_scope
    real = Path(QSettings(QSettings.IniFormat, QSettings.UserScope, st.ORG, st.APP).fileName()).resolve()
    assert seen is not None, "a module-scoped fixture built its panel against the real PRISM.ini"
    assert Path(seen).name == "prism.ini" and Path(seen).resolve() != real, seen
    assert basetemp in Path(seen).parents, f"{seen} is not under the session temp {basetemp}"
    now = st._override_path
    assert now is not None and now != seen and basetemp in Path(now).parents, now
    assert not Path(now).exists(), "the per-test file must not exist until Qt writes it"


def test_no_suite_points_the_settings_back_at_the_real_ini():
    """The teardown of every per-test redirect used to be a bare `use_ini_file` call with no path --
    and no path IS the real file. Under the session fixture that call would hand the next
    module-scoped fixture the developer's PRISM.ini, so the suites may not contain that call at all:
    tests/conftest.py alone resets the path, once, at session end, after asserting the real file's
    bytes are unchanged. The needle is assembled so this file does not match itself."""
    needle = "use_ini_file(" + "None)"
    here = Path(__file__).resolve().parent
    offenders = sorted(p.name for p in here.glob("test_*.py") if needle in p.read_text(encoding="utf-8"))
    assert offenders == [], f"these suites redirect the settings to the real PRISM.ini: {offenders}"


def test_the_check_of_the_real_settings_folder_sees_a_path_come_or_go_at_any_depth(tmp_path):
    """The session fixture of tests/conftest.py (_settings_home) lists the folder that holds the real settings
    file as the run begins and checks it as the run ends, through the two functions of tests/_settings_folder.py:
    whatever the window writes beside its settings file goes beside the test's own during a test, and one left
    beside the real file would be read by the next launch. Tried on a folder of this test's own: unchanged, it
    passes; a file left beside the settings file, one left in a folder beside it, the file itself taken away,
    and a folder made where there was none each fail it. And the fixture lists the real file's own folder at
    both ends with those two functions, which only the reading of its source can show."""
    import ast
    from tests import _settings_folder

    folder = tmp_path / "PRISM"
    (folder / "runs").mkdir(parents=True)
    (folder / "PRISM.ini").write_text("", encoding="utf-8")
    held = _settings_folder.listing(folder)
    _settings_folder.check_kept(folder, held)
    for left in (folder / "PRISM.lock", folder / "runs" / "one.log"):
        left.write_text("", encoding="utf-8")
        with pytest.raises(AssertionError, match="the suite changed what the real settings folder") as failed:
            _settings_folder.check_kept(folder, held)
        assert repr(str(left)) in str(failed.value), f"the check did not name {left}: {failed.value}"
        left.unlink()
    (folder / "PRISM.ini").unlink()
    with pytest.raises(AssertionError, match="the suite changed what the real settings folder") as failed:
        _settings_folder.check_kept(folder, held)
    assert repr(str(folder / "PRISM.ini")) in str(failed.value), f"the check did not name the file: {failed.value}"
    none = tmp_path / "none"
    _settings_folder.check_kept(none, _settings_folder.listing(none))
    none.mkdir()
    with pytest.raises(AssertionError, match="the suite changed what the real settings folder"):
        _settings_folder.check_kept(none, None)

    tree = ast.parse((Path(__file__).parent / "conftest.py").read_text(encoding="utf-8"))
    home, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_settings_home"]
    said = {ast.unparse(node) for node in ast.walk(home) if isinstance(node, (ast.Assign, ast.Call))}
    assert {"folder = real.parent", "held = _settings_folder.listing(folder)",
            "_settings_folder.check_kept(folder, held)"} <= said, (
        "the session no longer lists the real settings file's folder as it begins and checks it as it ends")


def test_each_builder_returns_a_new_widget_at_every_call_and_keeps_none(monkeypatch):
    """The suites reach a window, a part, the inference screen, a bare run host and a step of the look through
    five plain functions of tests/_fixtures.py, so a constructor's arguments are written once. Three
    things make them safe under tests that build twice to stand for a relaunch, or build after a
    patch, after a settings write or inside ``use_store``: each call builds anew, at the moment of
    the call; the helpers keep nothing they built, and the record of tests/_built.py holds each
    only until its test ends, when a fixture lets it go (the test below); and the
    inference builder does what it is handed and no more -- it assigns the session, installs the
    configuration, and refreshes no gate."""
    from core.gui.main_window import MainWindow
    from core.gui.panels.base_panel import BasePanel
    from core.gui.panels.inference.config_tab import ConfigPanel
    from core.gui.panels.inference.step_page import StepPage
    from core.gui.panels.inference_tabs import InferPanel
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from tests import _fixtures
    from tests._fixtures import build_inference, build_part, build_run_host, build_step_page, build_window

    def kept(widget):
        held = []
        for name, value in vars(_fixtures).items():
            inside = (value.values() if isinstance(value, dict)
                      else value if isinstance(value, (list, tuple, set, frozenset)) else ())
            if value is widget or any(item is widget for item in inside):
                held.append(name)
        return held

    first, second = build_part(InferPanel, screen=None), build_part(InferPanel, screen=None)
    assert type(first) is InferPanel and type(second) is InferPanel and first is not second
    step = build_part(ConfigPanel, screen=None)
    assert type(step) is ConfigPanel and step._screen is None

    # The constructor's own refreshes are counted where the constructor returns, apart from the
    # builder: a count read through the builder would take a refresh the builder made on every
    # call for the constructor's, and could not refuse it.
    refreshed, installed, as_constructed = [], [], []
    real_refresh, real_init = InferenceScreen.refresh_gates, InferenceScreen.__init__
    monkeypatch.setattr(InferenceScreen, "refresh_gates",
                        lambda self: (refreshed.append(self), real_refresh(self))[1])

    def counting_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        as_constructed.append(len(refreshed))

    monkeypatch.setattr(InferenceScreen, "__init__", counting_init)
    plain = build_inference()
    assert type(plain) is InferenceScreen and len(as_constructed) == 1 and as_constructed[-1] >= 1
    assert len(refreshed) == as_constructed[-1], "the builder refreshed the gates itself"
    session = SbiSession(cfg=object(), inf_prior=object())
    held = build_inference(session=session)
    assert held.session is session and held is not plain and len(as_constructed) == 2
    assert len(refreshed) == as_constructed[-1], "the builder refreshed the gates itself"
    monkeypatch.setattr(InferenceScreen, "install_config",
                        lambda self, cfg: installed.append((self, cfg)))
    cfg = object()
    built = build_inference(cfg=cfg)
    assert installed == [(built, cfg)] and len(as_constructed) == 3
    assert len(refreshed) == as_constructed[-1], "the builder refreshed the gates itself"
    assert "refresh_gates" not in code_only(build_inference), \
        "the builder leaves the gates to the test that builds with it"

    page, another = build_step_page(), build_step_page()
    assert isinstance(page, StepPage) and type(page).__name__ == "_Page" and page is not another
    assert page._screen is not another._screen and page._screen.panels()[2] is page
    assert type(build_step_page(type(page))) is type(page)
    host, other = build_run_host(), build_run_host()
    assert isinstance(host, BasePanel) and type(host) is not BasePanel and host is not other
    assert host.layout_key() == other.layout_key() == "P", "two hosts must share one layout name"
    assert type(build_run_host(BasePanel)) is BasePanel

    window, again = build_window(), build_window()
    assert type(window) is MainWindow and type(again) is MainWindow and window is not again

    for widget in (first, second, step, plain, held, built, page, another, host, other, window, again):
        assert kept(widget) == [], f"tests/_fixtures.py keeps what it built: {kept(widget)}"


# What the first turn of the test below made, kept for its second turn as a test that wanted its window to
# outlive it would keep one; and what one part of it saw as it went: the settings file in force, and whether
# that turn's drawing record was still unread; and a weak reference to a part it made and dropped.
_MADE_IN_THE_FIRST_TURN: list = []
_SEEN_AS_IT_WENT: list = []
_DROPPED_IN_THE_FIRST_TURN: list = []


@pytest.mark.parametrize("turn", ["builds", "after"])
def test_a_window_a_builder_made_is_gone_after_its_test(turn, _panel_built_at_module_scope):
    """A window, a part, the inference screen, a bare run host of either kind and a step of the look that a test made
    through the builders are deleted when that test ends, though something still holds them
    (tests/conftest.py::_built_widgets_let_go): left alive, each would be dressed again by every later change of
    sheet, style or font on the one application. The first turn builds one of each and keeps them in this
    module's list, beside a part it deletes itself, which the fixture must pass over; the second turn, the next
    test of the run, finds them deleted, and none of the panels among them left in the set of panels a run
    locks, though this module's list still holds them, as a run's own signals hold a panel that ran something.
    They go by their own deferred delete and never by a close, which writes settings, so the first turn's
    settings file was never written. They go while that file, and the room the turn left for the window's
    layout, are still in force and before the turn's drawing record is read, which one part notes as it goes:
    the fixture's place in the teardown order comes from the three fixtures it names, and nothing else holds
    it. The record holds what was built until the test ends, so a part the test itself dropped is still there
    until then, and goes with the rest on the window's thread: let go of by the collector of cycles, which runs on
    whichever thread is running Python when its turn comes, a window dropped by a test was deleted on a run's worker,
    and the process ended without a word. The part dropped here is held in a cycle of its own, as a window is, so it
    is gone at the second turn only because the record's end collects after its deletes. The panel a
    fixture of this module built is no test's own, and it is still there, still in the set. The two turns run
    together, in this order, in one process: the second alone always fails."""
    import shiboken6
    import weakref
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    from core.gui import design, reported
    from core.gui import settings as st
    from core.gui.panels.base_panel import BasePanel
    from core.gui.panels.inference_tabs import InferPanel
    from tests._fixtures import build_step_page, build_window

    module_panel = _panel_built_at_module_scope[0]
    if turn == "builds":
        gone = build_part(InferPanel, screen=None)
        gone.deleteLater()
        QCoreApplication.sendPostedEvents(gone, QEvent.DeferredDelete)
        dropped = build_part(QWidget)
        dropped.me = dropped                 # a cycle: only a collection lets it go
        alone = weakref.ref(dropped)
        del dropped
        assert alone() is not None, "the record let go of a part its test dropped before the test ended"
        _DROPPED_IN_THE_FIRST_TURN[:] = [alone]
        watched = build_part(QWidget)
        reported.COUNTS["unread"] = 0        # a mark the drawing check takes away once it has read the record
        watched.destroyed.connect(lambda *_: _SEEN_AS_IT_WENT.extend(
            [st._override_path, "unread" in reported.COUNTS, design._room]))
        _MADE_IN_THE_FIRST_TURN[:] = [build_window(), build_part(InferPanel, screen=None), build_inference(),
                                      build_run_host(), build_run_host(BasePanel), build_step_page(), watched, gone,
                                      st._override_path]
        assert [shiboken6.isValid(widget) for widget in _MADE_IN_THE_FIRST_TURN[:8]] == [True] * 7 + [False]
        design.set_room((1366, 728))         # the room this turn leaves in force, as a window it built would
        return
    assert len(_MADE_IN_THE_FIRST_TURN) == 9, (
        "the first turn did not run before this one: the two turns run together, in this order, in one process")
    *made, _deleted_by_the_test, written = _MADE_IN_THE_FIRST_TURN
    dropped, = _DROPPED_IN_THE_FIRST_TURN
    _DROPPED_IN_THE_FIRST_TURN.clear()
    assert dropped() is None, "a part its test dropped outlived the test"
    seen = list(_SEEN_AS_IT_WENT)
    _MADE_IN_THE_FIRST_TURN.clear()
    _SEEN_AS_IT_WENT.clear()
    assert [shiboken6.isValid(widget) for widget in made] == [False] * 7, (
        "a widget a builder made outlived the test that built it")
    assert [panel for panel in BasePanel._instances if not shiboken6.isValid(panel)] == [], (
        "a deleted panel is still among the panels a run locks")
    assert seen == [written, True, (1366, 728)], (
        "what the builders made went after the test's own settings file or room was put back, or after its "
        "drawing record was read")
    assert not Path(written).exists(), "a window was closed, or a widget's deletion wrote settings, when its test ended"
    assert shiboken6.isValid(module_panel), "the end of a test deleted what a fixture of wider scope built"
    assert module_panel in BasePanel._instances, "the end of a test took a living panel out of the set a run locks"


def test_no_suite_builds_a_window_a_part_or_a_step_by_hand():
    """Every window, part, step and bare run host a suite needs comes from the builders of
    tests/_fixtures.py. Parsed, not searched: a class named in prose, or handed to a builder, is not
    a construction. Refused in every file of this folder but that one: a call of a window class
    (also through an expression that names one, as a lookup in a dict of classes), a local subclass
    of a window class, BasePanel among them, and a call of build_app, which whole_program alone
    makes. A class called through a variable is not seen here; the builders take the class as an
    argument, so no suite has a reason to hold one and call it."""
    import ast
    built = {"MainWindow", "InferenceScreen", "ModelBuilderScreen", "ArtifactScreen", "ConfigPanel", "PriorPanel",
             "PosteriorPanel", "ValidatePanel", "InferPanel", "TSNPEPanel", "BasePanel",
             "HomeScreen", "SettingsScreen", "Shell", "HelpScreen",
             "ModelsScreen", "PartError", "RunView", "StepPage", "CheckPage"}
    # The list is kept closed over the program. Every class with a base that a file of the window's own
    # folder, of its screens, of its panels or of its runs defines under a public name is named above, or here among
    # those a suite still builds itself, or among the classes of the window's folder that are no window,
    # part or step. So a class a later change adds, or a folder it adds beside the window's, fails here
    # until it is placed.
    by_hand_still = set()                    # none today
    no_window = {"Place", "Claim", "NewPngWatcher", "WorkerCancelled", "PrismStyle", "FocusRing",
                 "FocusFollower", "Appearance", "WorkerSignals", "Worker", "RunController", "LossCurve",
                 "FigureNotShown", "SessionWatch", "LoadChain"}
    placed = built | by_hand_still | no_window
    gui = Path(__file__).resolve().parents[1] / "core" / "gui"
    folders = sorted(p.name for p in gui.iterdir() if p.is_dir() and p.name != "__pycache__")
    defined = {node.name for path in (*sorted(gui.glob("*.py")), *sorted((gui / "screens").rglob("*.py")),
                                      *sorted((gui / "panels").rglob("*.py")), *sorted((gui / "run").rglob("*.py")))
               for node in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(node, ast.ClassDef) and node.bases and not node.name.startswith("_")}
    assert (folders, sorted(defined - placed), sorted(placed - defined)) == (
        ["assets", "panels", "run", "screens", "widgets"], [], []), (
        "the window's folders; then the classes it defines that this scan names nowhere; then the names "
        "this scan holds that are defined no longer")

    def named(node):
        return ({n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)})

    def last_name(node):
        return (node.id if isinstance(node, ast.Name)
                else node.attr if isinstance(node, ast.Attribute) else None)

    found = []
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        if path.name == "_fixtures.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                callee = last_name(node.func)
                by_hand = (callee in built or callee == "build_app"
                           or (callee is None and bool(named(node.func) & built)))
            else:
                by_hand = (isinstance(node, ast.ClassDef)
                           and any(last_name(base) in built for base in node.bases))
            if by_hand:
                found.append(f"{path.name}:{node.lineno}")
    assert found == [], f"{len(found)} constructions written by hand: {found}"


def test_no_widget_the_builders_make_is_shown_as_a_window_of_its_own(monkeypatch):
    """STANDING. A widget made with no parent and made visible is a window of its own, shown on the screen at once, a
    flash in the real program before its page takes it in: a page gives each widget its parent first, then its
    visibility. Every widget the window, each of its parts, each step and a bare step page are built with, each check's
    page and form among them, and a configuration installed on Parameter inference, is watched as it is made visible,
    shown or unhidden: none is a window then but the window itself and the questions and message boxes the dialogs'
    guard records."""
    from PySide6.QtWidgets import QMessageBox, QWidget
    from core.gui import keys
    from core.gui.main_window import MainWindow
    from core.gui.widgets.ask import Ask
    from tests._fixtures import build_step_page, build_window, forced_cfg, qt_app
    qt_app()
    shown = []
    real = {name: getattr(QWidget, name) for name in ("setVisible", "show", "setHidden")}

    def heard(widget, on: bool) -> None:
        if on and widget.isWindow() and not isinstance(widget, (MainWindow, Ask, QMessageBox)):
            words = getattr(widget, "text", None)
            shown.append((type(widget).__name__, words() if callable(words) else ""))

    monkeypatch.setattr(QWidget, "setVisible", lambda widget, on: (heard(widget, on), real["setVisible"](widget, on)))
    monkeypatch.setattr(QWidget, "show", lambda widget: (heard(widget, True), real["show"](widget)))
    monkeypatch.setattr(QWidget, "setHidden", lambda widget, on: (heard(widget, not on), real["setHidden"](widget, on)))
    window = build_window()
    for part in keys.PARTS:
        window.part(part.key)
    window.part("inference").install_config(forced_cfg())
    build_step_page()
    for name, method in real.items():
        monkeypatch.setattr(QWidget, name, method)
    assert shown == [], f"{len(shown)} widgets were shown as windows of their own: {shown}"


def test_int_field_value_or_none_tells_a_blank_from_a_zero():
    """``IntField.value()`` returns 0 for "" and for "-" mid-typing, and 0 is a legal value for the
    two "0 = automatic" boxes (the rows-per-batch cap, the candidates per sweep round), so a tab that
    refuses a blank rather than reading it as zero needs the FloatField twin: None for anything that
    does not parse, the int otherwise, whitespace tolerated. ``value()`` keeps its old contract for
    the callers that still use it."""
    from core.gui.widgets.labeled_inputs import IntField
    from tests._fixtures import qt_app

    qt_app()
    f = IntField(5)
    assert f.value_or_none() == 5
    for text, want in (("0", 0), (" 12 ", 12), ("-3", -3), ("", None), ("-", None), ("1.5", None)):
        f.setText(text)                      # text set from code passes no key rule, as a restore's does not
        assert f.value_or_none() == want, (text, f.value_or_none())
    f.setText("")
    assert f.value() == 0, "value() still reads a blank as 0 for its remaining callers"


def test_the_config_tab_science_knobs_open_at_config_and_are_not_written():
    """On Model and options the driven recordings' count, slots and lock-in limit open at config.py's values on EVERY
    launch, the drive amplitude and band are text, never a field, and the rotation and the units choice open at their
    usual values. Only the model and the kind of observation are remembered, and only when they are applied: nothing is
    written at closing, and a key an older build left -- the units choice, the typed units, the rotation, every science
    key -- is neither read nor written again.

    The seed-then-restore pattern is what trained the 2026-08-19 retrain on the retired band: the tab seeded the boxes
    from config.CHI_* and then restored whatever the last session had saved, so a value written before config.py changed
    won silently on every launch afterwards. A stale key in an old PRISM.ini is IGNORED, never migrated.
    """
    from core import config
    from core.gui import settings as st
    from core.gui.panels import inference_tabs as it
    from tests._fixtures import qt_app

    qt_app()
    want = {"chi_n_freqs": config.CHI_N_FREQS, "chi_k_pad": config.CHI_K_PAD, "chi_max_cycles": config.CHI_MAX_CYCLES,
            "reparam_rotate": config.REPARAM_ROTATE, "units": "file"}

    def shown(panel):
        return {key: panel.form.rows[key].value() for key in want}

    # (a) an empty file: the autouse settings fixture points at a fresh path for this test
    fresh = build_part(it.ConfigPanel, screen=None)
    assert shown(fresh) == want, shown(fresh)
    for key in ("chi_f0", "chi_freq_bounds"):
        assert fresh.form.rows[key].widget is None, f"{key} is a field"
    assert (fresh.model_combo.currentText(), fresh.kind_choice.currentData()) == ("NADROWSKI", False)

    # (b) a file holding every old key, each one step from the default, beside the two selections
    stale = {"chi_k": str(config.CHI_N_FREQS + 1), "chi_k_pad": str(config.CHI_K_PAD - 1),
             "chi_f0": str(config.CHI_F0 / 2), "chi_lo": str(config.CHI_FREQ_BOUNDS[0] / 3),
             "chi_hi": str(config.CHI_FREQ_BOUNDS[1] * 3), "chi_max_cycles": str(config.CHI_MAX_CYCLES + 5),
             "units_mode": "direct", "units_text": "nm s pN Hz", "reparam_rotate": str(not config.REPARAM_ROTATE)}
    qs = st.settings()
    qs.beginGroup("inference_config")
    for key, value in stale.items():
        qs.setValue(key, value)
    qs.setValue("chi_mode", "1")
    qs.setValue("model", "HOPF")
    qs.endGroup()
    qs.setValue("layout/ConfigPanel/splitter", "stale")
    qs.sync()
    relaunched = build_part(it.ConfigPanel, screen=None)
    assert shown(relaunched) == want, shown(relaunched)
    assert relaunched.kind_choice.currentData() is True, "the kind of observation is a selection: remembered"
    assert relaunched.model_combo.currentText() == "HOPF", "the model is a selection: remembered"

    # (c) nothing is written at closing, and the old keys stay as they were, unread and unwritten
    relaunched.form.rows["chi_n_freqs"].widget.setText("3")
    relaunched.save_settings(qs)
    qs.sync()
    again = st.settings()
    again.beginGroup("inference_config")
    held = {key: again.value(key) for key in again.childKeys()}
    again.endGroup()
    assert held == {**stale, "chi_mode": "1", "model": "HOPF"}, held
    assert again.value("layout/ConfigPanel/splitter") == "stale"


def test_the_prior_picks_are_written_when_they_come_into_use(tmp_path, monkeypatch):
    """The bounds file and the prior are remembered as they come into use, when a run returns its prior, and at no other
    moment: not as they are chosen, not as a run starts, not at closing. Limits typed by hand leave the bounds file
    remembered as it was."""
    from types import SimpleNamespace
    from core import orchestrator
    from core.config import BOUNDS_PATH
    from core.gui import settings as st
    from core.gui.remembered import BOUNDS_KEY, PRIOR_KEY
    from core.gui.run.controller import controller
    from tests._fixtures import qt_app
    from tests._runs import Clock, StandIn, at, controller_on, turn_until
    app = qt_app()
    controller_on(tmp_path, Clock(at(8, 12, 4)))
    screen = build_inference()
    screen.config_panel.foot.main_button.click()
    page = screen.prior_panel
    stand_in = StandIn(word="Prior build")
    monkeypatch.setattr(orchestrator, "build_prior", stand_in)

    def remembered():
        qs = st.settings()
        qs.sync()
        return qs.value(BOUNDS_KEY), qs.value(PRIOR_KEY)

    page.bounds_picker.restore_key("master_tier1.txt")
    page.names.name.setText("retrain_prior")
    page.read_facts()
    assert remembered() == (None, None), "a choice was remembered as it was made"
    page.main_pressed()
    assert stand_in.started.wait(5) and remembered() == (None, None), "a pick was remembered as its run started"
    stand_in.release(SimpleNamespace(kind="prior", id="20261008T120400", name="retrain_prior", manifest=None))
    turn_until(app, lambda: not controller().is_live(), 10.0)
    assert remembered() == ("master_tier1.txt", "20261008T120400")
    assert screen.session.cfg.sources["bounds"] == str(BOUNDS_PATH / "nadrowski" / "master_tier1.txt")
    page.typed_tick.setChecked(True)
    page.names.name.setText("retrain_prior_2")
    page.read_facts()
    page.main_pressed()
    assert stand_in.started.wait(5)
    stand_in.release(SimpleNamespace(kind="prior", id="20261008T130000", name="", manifest=None))
    turn_until(app, lambda: not controller().is_live(), 10.0)
    assert remembered() == ("master_tier1.txt", "20261008T130000")
    typed = page.form.rows["bounds_typed"].line
    assert not typed.says() and screen.pending() == frozenset(), (typed.words, screen.pending())
    out = st.settings()
    page.save_settings(out)
    out.sync()
    assert remembered() == ("master_tier1.txt", "20261008T130000")


def test_the_saved_prior_of_last_time_is_put_back_at_launch(store):
    """The prior that came into use last is selected in the saved list at the next launch, marked kept from last time
    until it is touched, and never loaded; one no longer saved selects nothing and is marked nothing."""
    from core.gui import settings as st
    from core.gui.remembered import PRIOR_KEY
    from tests._fixtures import _nad_cfg, _prior_artifact, qt_app
    qt_app()
    made = _prior_artifact(store, _nad_cfg(), name="last_time").id
    _prior_artifact(store, _nad_cfg(), name="another")
    for remembered, selected, origin in ((made, made, "kept from last time"), ("gone_since", "", "")):
        qs = st.settings()
        qs.setValue(PRIOR_KEY, remembered)
        qs.sync()
        screen = build_inference()
        page = screen.prior_panel
        screen.config_panel.foot.main_button.click()
        page.show_view("saved")
        page.redraw()
        assert page.prior_picker.key() == selected and screen.session.inf_prior is None, remembered
        assert page.form.rows["prior"].origin.text() == origin, remembered


# ── what was in use last time, as Home reads it ──
#: The plan a launch before this one used last with Nadrowski, its bounds file another than the prior's.
_LAST_PLAN = "# Prior\nbounds = master_tier1.txt\n\n# Training\nnum_runs = 7000\n"


def _last_time(store, tmp_path, monkeypatch, **changed) -> dict:
    """What a launch before this one left: an unnamed prior made for Nadrowski on master.txt, a posterior named
    zz_retrain trained with it, a plan zz_last in the test's own folder of plans, each remembered as in use last time
    with the kind the bounds file sets. ``changed`` names keys of the settings file to write otherwise (None leaves one
    out). The records, the plans folder and what was written, by key."""
    from core.gui import remembered
    from tests._fixtures import _nad_cfg, _posterior_artifact, _prior_artifact
    cfg = _nad_cfg()
    prior = _prior_artifact(store, cfg, name="").id
    posterior = _posterior_artifact(store, cfg, name="zz_retrain", parents={"prior": prior}).id
    plans = tmp_path / "plans"
    (plans / "nadrowski").mkdir(parents=True)
    (plans / "nadrowski" / "zz_last.txt").write_bytes(_LAST_PLAN.encode("utf-8"))
    written = {remembered.MODEL_KEY: "NADROWSKI", remembered.KIND_KEY: "false", remembered.BOUNDS_KEY: "master.txt",
               remembered.BOUNDS_MODEL_KEY: "NADROWSKI", remembered.PRIOR_KEY: prior,
               remembered.POSTERIOR_KEY: posterior, remembered.LAST_USED_KEY.format("nadrowski"): "zz_last"}
    written.update(changed)
    remembered.put_many([(key, value) for key, value in written.items() if value is not None])
    return {"prior": prior, "posterior": posterior, "plans": plans, "written": written}


def _phrase(store, kind, ident) -> str:
    """The words a sentence names a record by, worked out here from its manifest: by its name, or, unnamed, by its kind
    and when it was made in local time."""
    from datetime import datetime
    from core.gui import design
    manifest = store.get(kind, ident)
    if manifest.name:
        return f"the {kind} {manifest.name}"
    return f"the unnamed {kind}, {design.date_text(datetime.fromisoformat(manifest.created).astimezone())}"


_PICKS_CASES = ["everything", "a model of your own whose file was deleted", "a model inference cannot take",
                "no model", "the bounds file gone", "the bounds file renamed in case alone",
                "the bounds file of another model", "the bounds file remembered with no model", "the prior deleted",
                "a prior made on another bounds file", "a prior made for another model", "a posterior of another prior",
                "a posterior that names no prior, with no prior",
                "a posterior of another kind", "the frequency response with a posterior of it",
                "a posterior of other slots", "a posterior of another lock-in limit",
                "the plan's slots with a posterior of them", "a kind not read",
                "the plan gone", "the plan with a problem", "the plan the window's checks refuse",
                "the plan under another spelling", "the plan of another model", "Hopf, from its own places"]


@pytest.mark.parametrize("case", _PICKS_CASES)
def test_the_picks_leave_out_what_can_no_longer_be_found(case, store, tmp_path, monkeypatch):
    """What was in use last time, each left out when it can no longer be found as it was used: the model, while the
    list of models holds it and parameter inference can take it, else nothing at all; the bounds file, while its model
    is the model's and the model's folder lists it by exactly that name; the prior, while the bounds file stands and
    its record, made for the model on that very file, can be read; the posterior, while the prior stands and its record
    names that prior and is of the kind of observation, and, of the frequency response, was trained with the slots and
    the lock-in limit the press applies, the plan's where it holds them; the plan, while its file is there, found
    without regard to case and named as the folder lists it, and has no problem. Each is read from the model's own
    places: its folder of bounds files, its records and its folder of plans. Records and the plan differ from what a
    session would hold."""
    from core import config, registry
    from core.gui import remembered
    from tests._fixtures import _nad_cfg, _posterior_artifact, _prior_artifact
    changed, plans_written = {}, {}
    if case == "a model of your own whose file was deleted":
        changed[remembered.MODEL_KEY] = "OWN_GONE"
    elif case == "a model inference cannot take":
        monkeypatch.setattr(config, "VALID_MODELS", [*config.VALID_MODELS, "OWN_FORCED"])
        monkeypatch.setattr(registry, "is_user_model", lambda name: name == "OWN_FORCED")
        monkeypatch.setattr(registry, "is_sbi_user_model", lambda name: False)
        changed[remembered.MODEL_KEY] = "OWN_FORCED"
    elif case == "no model":
        changed[remembered.MODEL_KEY] = None
    elif case == "the bounds file gone":
        changed[remembered.BOUNDS_KEY] = "zz_gone.txt"
    elif case == "the bounds file renamed in case alone":
        changed[remembered.BOUNDS_KEY] = "MASTER.TXT"
    elif case == "the bounds file of another model":
        changed[remembered.BOUNDS_MODEL_KEY] = "HOPF"
    elif case == "the bounds file remembered with no model":
        changed[remembered.BOUNDS_MODEL_KEY] = None
    elif case == "the prior deleted":
        changed[remembered.PRIOR_KEY] = "20200101T000000"
    elif case == "a posterior of another kind":
        changed[remembered.KIND_KEY] = "true"
    elif case == "a kind not read":
        changed[remembered.KIND_KEY] = "maybe"
    elif case == "the plan gone":
        changed[remembered.LAST_USED_KEY.format("nadrowski")] = "zz_gone"
    elif case == "the plan of another model":
        changed[remembered.LAST_USED_KEY.format("nadrowski")] = "zz_hopf"
    held = _last_time(store, tmp_path, monkeypatch, **changed)
    folder = held["plans"] / "nadrowski"
    if case == "a prior made on another bounds file":
        tier1 = _nad_cfg()
        tier1.sources = {**tier1.sources, "bounds": str(config.BOUNDS_PATH / "nadrowski" / "master_tier1.txt")}
        other = _prior_artifact(store, tier1, name="zz_tier1").id
        remembered.put(remembered.PRIOR_KEY, other)
    elif case == "a prior made for another model":
        other = _prior_artifact(store, _nad_cfg(), name="zz_hopf", model="HOPF").id
        remembered.put(remembered.PRIOR_KEY, other)
    elif case == "a posterior of another prior":
        other = _posterior_artifact(store, _nad_cfg(), name="zz_other", parents={"prior": "20200101T000000"}).id
        remembered.put(remembered.POSTERIOR_KEY, other)
    elif case == "a posterior that names no prior, with no prior":
        import json
        written = store.path("posterior", held["posterior"]) / "manifest.json"
        manifest = json.loads(written.read_text(encoding="utf-8"))
        manifest["parents"] = {}
        written.write_text(json.dumps(manifest), encoding="utf-8")
        remembered.put(remembered.PRIOR_KEY, "")
    elif case in ("the frequency response with a posterior of it", "a posterior of other slots",
                  "a posterior of another lock-in limit", "the plan's slots with a posterior of them"):
        slots = {"a posterior of other slots": 16, "the plan's slots with a posterior of them": 14}.get(case,
                                                                                                       config.CHI_K_PAD)
        ceiling = 25.0 if case == "a posterior of another lock-in limit" else config.CHI_MAX_CYCLES
        chi = _posterior_artifact(store, _nad_cfg(), name="zz_chi", parents={"prior": held["prior"]},
                                  over={("mode",): "chi", ("conditioning", "chi_k_pad"): slots,
                                        ("conditioning", "chi_max_cycles"): ceiling}).id
        remembered.put_many([(remembered.KIND_KEY, "true"), (remembered.POSTERIOR_KEY, chi)])
        held["posterior"] = chi
        if case == "the plan's slots with a posterior of them":
            (folder / "zz_last.txt").write_bytes(f"# Model and options\nchi_mode = on\nchi_k_pad = 14\n\n"
                                                 f"{_LAST_PLAN}".encode("utf-8"))
    elif case == "Hopf, from its own places":
        hopf = _nad_cfg()
        hopf.sources = {**hopf.sources, "bounds": str(config.BOUNDS_PATH / "hopf" / "cell.txt")}
        held["prior"] = _prior_artifact(store, hopf, name="", model="HOPF").id
        held["posterior"] = _posterior_artifact(store, hopf, name="zz_hopf", parents={"prior": held["prior"]}).id
        (held["plans"] / "hopf").mkdir()
        (held["plans"] / "hopf" / "zz_hopf.txt").write_bytes(b"# Training\nnum_runs = 7000\n")
        remembered.put_many([(remembered.MODEL_KEY, "HOPF"), (remembered.BOUNDS_KEY, "cell.txt"),
                             (remembered.BOUNDS_MODEL_KEY, "HOPF"), (remembered.PRIOR_KEY, held["prior"]),
                             (remembered.POSTERIOR_KEY, held["posterior"]),
                             (remembered.LAST_USED_KEY.format("hopf"), "zz_hopf")])
    elif case == "the plan with a problem":
        (folder / "zz_last.txt").write_bytes(b"# Training\nnum_runs = many\n")
    elif case == "the plan the window's checks refuse":
        (folder / "zz_last.txt").write_bytes(b"# Prior\nbounds = zz_gone.txt\n")
    elif case == "the plan under another spelling":
        (folder / "zz_last.txt").rename(folder / "ZZ_Last.txt")
    elif case == "the plan of another model":
        (held["plans"] / "hopf").mkdir()
        (held["plans"] / "hopf" / "zz_hopf.txt").write_bytes(b"# Training\nnum_runs = 7000\n")
    found = remembered.remembered_picks(store=store, plans_folder=held["plans"])
    prior, posterior = held["prior"], held["posterior"]
    whole = remembered.Picks("NADROWSKI", False, "master.txt", prior, posterior, "zz_last",
                             _phrase(store, "prior", prior), _phrase(store, "posterior", posterior))
    expected = {
        "everything": whole,
        "a model of your own whose file was deleted": remembered.Picks(),
        "a model inference cannot take": remembered.Picks(),
        "no model": remembered.Picks(),
        "the bounds file gone": remembered.Picks("NADROWSKI", False, plan="zz_last"),
        "the bounds file renamed in case alone": remembered.Picks("NADROWSKI", False, plan="zz_last"),
        "the bounds file of another model": remembered.Picks("NADROWSKI", False, plan="zz_last"),
        "the bounds file remembered with no model": whole,
        "the prior deleted": remembered.Picks("NADROWSKI", False, "master.txt", plan="zz_last"),
        "a prior made on another bounds file": remembered.Picks("NADROWSKI", False, "master.txt", plan="zz_last"),
        "a prior made for another model": remembered.Picks("NADROWSKI", False, "master.txt", plan="zz_last"),
        "a posterior of another prior": remembered.Picks("NADROWSKI", False, "master.txt", prior, plan="zz_last",
                                                         prior_words=_phrase(store, "prior", prior)),
        "a posterior that names no prior, with no prior": remembered.Picks("NADROWSKI", False, "master.txt",
                                                                           plan="zz_last"),
        "a posterior of another kind": remembered.Picks("NADROWSKI", True, "master.txt", prior, plan="zz_last",
                                                        prior_words=_phrase(store, "prior", prior)),
        "the frequency response with a posterior of it": remembered.Picks(
            "NADROWSKI", True, "master.txt", prior, posterior, "zz_last", _phrase(store, "prior", prior),
            "the posterior zz_chi"),
        "a posterior of other slots": remembered.Picks("NADROWSKI", True, "master.txt", prior, plan="zz_last",
                                                       prior_words=_phrase(store, "prior", prior)),
        "a posterior of another lock-in limit": remembered.Picks("NADROWSKI", True, "master.txt", prior, plan="zz_last",
                                                                 prior_words=_phrase(store, "prior", prior)),
        "the plan's slots with a posterior of them": remembered.Picks(
            "NADROWSKI", True, "master.txt", prior, posterior, "zz_last", _phrase(store, "prior", prior),
            "the posterior zz_chi"),
        "a kind not read": remembered.Picks("NADROWSKI", None, "master.txt", prior, posterior, "zz_last",
                                            _phrase(store, "prior", prior), "the posterior zz_retrain"),
        "the plan gone": remembered.Picks("NADROWSKI", False, "master.txt", prior, posterior, None,
                                          _phrase(store, "prior", prior), "the posterior zz_retrain"),
        "the plan with a problem": remembered.Picks("NADROWSKI", False, "master.txt", prior, posterior, None,
                                                    _phrase(store, "prior", prior), "the posterior zz_retrain"),
        "the plan the window's checks refuse": remembered.Picks("NADROWSKI", False, "master.txt", prior, posterior,
                                                                None, _phrase(store, "prior", prior),
                                                                "the posterior zz_retrain"),
        "the plan under another spelling": remembered.Picks("NADROWSKI", False, "master.txt", prior, posterior,
                                                            "ZZ_Last", _phrase(store, "prior", prior),
                                                            "the posterior zz_retrain"),
        "the plan of another model": remembered.Picks("NADROWSKI", False, "master.txt", prior, posterior, None,
                                                      _phrase(store, "prior", prior), "the posterior zz_retrain"),
        "Hopf, from its own places": remembered.Picks("HOPF", False, "cell.txt", prior, posterior, "zz_hopf",
                                                      _phrase(store, "prior", prior), "the posterior zz_hopf"),
    }[case]
    assert found == expected, (case, found)
    assert whole.prior_words.startswith("the unnamed prior, "), "the prior of last time has a name"


def test_a_picks_reading_writes_nothing(store, tmp_path, monkeypatch):
    """Reading what was in use last time, however often and whatever it finds, writes nothing: the settings file's
    bytes and the moment it was last written are as they were, and no key is set."""
    import os
    from PySide6.QtCore import QSettings
    from core.gui import remembered, settings
    held = _last_time(store, tmp_path, monkeypatch, **{remembered.KIND_KEY: "maybe"})
    path = Path(settings.settings().fileName())
    before = (path.read_bytes(), os.stat(path).st_mtime_ns)
    written = []
    kept = QSettings.setValue
    monkeypatch.setattr(QSettings, "setValue", lambda qs, key, value: (written.append(key), kept(qs, key, value))[1])
    for _ in range(3):
        assert remembered.remembered_picks(store=store, plans_folder=held["plans"]).model == "NADROWSKI"
    (held["plans"] / "nadrowski" / "zz_last.txt").unlink()
    assert remembered.remembered_picks(store=store, plans_folder=held["plans"]).plan is None
    assert (path.read_bytes(), os.stat(path).st_mtime_ns) == before and written == []


def test_the_picks_read_step_ones_heading_where_the_plans_name_it(monkeypatch):
    """A posterior of the frequency response is held to the slots and the lock-in limit of the plan last used under the
    heading the window's table of plans gives step 1, read from that table and written nowhere else: renamed there, the
    plan's values under the new heading are the ones compared."""
    from types import SimpleNamespace
    from core import config
    from core.gui import plans, remembered
    monkeypatch.setitem(plans.HEADING_OF, "model", "Step one")
    slots, ceiling = config.CHI_K_PAD + 3, config.CHI_MAX_CYCLES + 2.0
    manifest = SimpleNamespace(body={"mode": "chi", "conditioning": {"chi_k_pad": slots, "chi_max_cycles": ceiling}})
    renamed = SimpleNamespace(values={("Step one", "chi_k_pad"): slots, ("Step one", "chi_max_cycles"): ceiling})
    old = SimpleNamespace(values={("Model and options", "chi_k_pad"): slots,
                                  ("Model and options", "chi_max_cycles"): ceiling})
    assert remembered._conditioned_as_applied(manifest, renamed) is True
    assert remembered._conditioned_as_applied(manifest, old) is False


def test_science_knobs_open_at_config_and_are_not_written(tmp_path):
    """On the Prior, Training, Checks and Narrowing round pages (the first step is pinned the same way by
    test_the_config_tab_science_knobs_open_at_config_and_are_not_written), every SCIENCE KNOB opens at
    config.py's value -- the truncate module's, for the HPD level and the direction count -- on EVERY
    launch, and its key is neither written nor read. A stale key an older build left in PRISM.ini is
    IGNORED, never migrated.

    Why: each of these boxes was seeded from config.py and then overwritten from QSettings, so a value
    typed once won silently over config.py on every later launch. The sweep and clustering boxes
    decide which prior gets built, the network and Fisher boxes which posterior gets trained, the
    calibration boxes what "calibrated" was measured with, and nothing on screen compared any of them
    with config.py. The same seed-then-restore pattern trained the 2026-08-19 retrain on the retired
    χ band. The selections and the budget stay remembered, and the old file below holds one of those
    per tab that has one, so "opens at config.py" cannot pass merely because restore_settings never
    ran.

    Three legs: an empty file; a file holding every old key one step from config.py; a save after
    every knob box was edited, which must write nothing. Every page holds its settings as rows of its form,
    and writes nothing at a save: what it remembers is written as it comes into use, so the old keys of its
    boxes and the keys of its rows are both in the file, and neither is read.
    """
    from PySide6.QtCore import QSettings
    from core import config
    from core.gui import settings as st
    from core.SBI import truncate as _tr
    from tests._fixtures import qt_app

    qt_app()
    # (group, the screen's panel attribute, {key, which is also the box attribute or the form row's key: config.py's
    # value})
    knobs = (
        ("inference_prior", "prior_panel", {
            "num_iterations": config.PRIOR_SWEEP_ITERATIONS, "sweep_batch": config.PRIOR_SWEEP_BATCH,
            "max_sets": config.PRIOR_SWEEP_MAX_SETS, "walk_step": config.PRIOR_SWEEP_STEP,
            "stability_units": config.STABILITY_SWEEP_ND_UNITS,
            "min_cluster_size": config.PRIOR_CLUSTER_MIN_SIZE,
            "min_samples": config.PRIOR_CLUSTER_MIN_SAMPLES}),
        ("inference_posterior", "posterior_panel", {
            "hidden_features": config.NSF_HIDDEN_FEATURES, "num_transforms": config.NSF_NUM_TRANSFORMS,
            "learning_rate": config.TRAINING_LEARNING_RATE, "stop_after_epochs": config.TRAINING_STOP_AFTER_EPOCHS,
            "checkpoint_every": config.TRAINING_CHECKPOINT_EVERY, "fisher_m": config.REPARAM_FISHER_M,
            "fisher_dz": config.REPARAM_FISHER_DZ, "fisher_points": config.REPARAM_FISHER_POINTS}),
        ("inference_validate", "validate_panel", {
            "cal_n": config.SBC_N_CAL, "cal_scales": config.CAL_N_SCALES}),
        ("inference_tsnpe", "tsnpe_panel", {
            "hpd_level": _tr.DEFAULT_HPD, "n_directions": _tr.DEFAULT_N_DIRECTIONS}),
    )

    def stale(default):
        """One step from config.py: an int box plus one, a float box halved."""
        return str(default + 1) if isinstance(default, int) else str(default / 2)

    def box(inf, attr, key):
        """The knob's box: the panel's attribute, or the field of a page's form row, a ticked field's own box; the
        calibration's sizes on its check's page's form, by the stage's names for the keys an older build wrote."""
        panel = getattr(inf, attr)
        if attr == "validate_panel":
            widget = panel.check_pages["calibration"].form.rows[{"cal_n": "n_cal", "cal_scales": "cal_n_scales"}[key]]
            return getattr(widget.widget, "field", widget.widget)
        rows = getattr(getattr(panel, "form", None), "rows", None)
        if rows is None:
            return getattr(panel, key)
        widget = rows[key].widget
        return getattr(widget, "field", widget)

    def off_config(inf):
        """Every knob box whose number is not config.py's, as {(group, key): text}. Compared as
        numbers: the old restore wrote str(float(...)), so config.py's 1000 came back as "1000.0"."""
        return {(group, key): box(inf, attr, key).text()
                for group, attr, keys in knobs for key, default in keys.items()
                if float(box(inf, attr, key).text().replace(",", "")) != float(default)}

    # (a) an empty file (the autouse settings fixture points at a fresh path for this test)
    first = build_inference()
    assert off_config(first) == {}, f"an empty file opened off config.py: {off_config(first)}"

    # (b) a file holding every old knob key one step from config.py, the dead bounds_source, and one
    # remembered selection or budget per tab that has one -- the proof that restore_settings RAN
    qs = st.settings()
    for group, _attr, keys in knobs:
        for key, default in keys.items():
            qs.setValue(f"{group}/{key}", stale(default))
    for old in ("sweep_iters", "sweep_max_sets", "sweep_step", "sweep_units", "cluster_size", "cluster_samples"):
        qs.setValue(f"inference_prior/{old}", "7")
    for old in ("flow_hidden", "flow_transforms", "flow_lr", "flow_patience"):
        qs.setValue(f"inference_posterior/{old}", "7")
    for old in ("hpd", "n_dirs"):
        qs.setValue(f"inference_tsnpe/{old}", "0.5")
    qs.setValue("inference_prior/bounds_source", "direct")
    qs.setValue("inference_prior/bounds", "stale_bounds.txt")
    qs.setValue("inference_posterior/num_runs", "1234")
    qs.setValue("inference_tsnpe/num_runs", "4321")
    # the recording lengths of an older Inference page, one per form, and its drive amplitude: blank at every launch
    for old in ("sim_tobs", "exp_tobs", "chi_tobs", "chi_f0_si"):
        qs.setValue(f"inference_infer/{old}", "7.5")
    qs.sync()
    relaunched = build_inference()
    assert [relaunched.infer_panel.form.rows[key].widget.text() for key in ("t_obs", "chi_f0_si")] == ["", ""], (
        "an older page's length or drive amplitude was put back")
    assert off_config(relaunched) == {}, f"science knobs restored from an old file: {off_config(relaunched)}"
    assert relaunched.prior_panel._kept_pick == ("", "stale_bounds.txt"), "the bounds file is a selection"
    assert relaunched.prior_panel.typed_tick.isChecked() is False
    assert relaunched.posterior_panel.form.rows["num_runs"].value() == 1234, "the Training budget is remembered"
    assert relaunched.tsnpe_panel.form.rows["num_runs"].value() == 4321, "the round's budget is remembered"

    # (c) nothing writes them: every knob box edited, the four panels saved into a file of their own
    for _group, attr, keys in knobs:
        for key, default in keys.items():
            box(relaunched, attr, key).setText(stale(default))
    out = QSettings(str(tmp_path / "saved.ini"), QSettings.IniFormat)
    for panel in (relaunched.prior_panel, relaunched.posterior_panel, relaunched.validate_panel,
                  relaunched.tsnpe_panel):
        panel.save_settings(out)
    out.sync()
    expected = {"inference_prior": set(), "inference_posterior": set(), "inference_validate": set(),
                "inference_tsnpe": set()}
    for group, keys in expected.items():
        out.beginGroup(group)
        written = set(out.childKeys())
        out.endGroup()
        assert written == keys, f"[{group}] writes {sorted(written)} at a save"


def test_the_round_restores_its_observation_and_budget_only(monkeypatch):
    """The narrowing round puts back what the remembered table keeps of it, as a round started with them wrote it: the
    observation, chosen in its list, and the batches and the simulations per batch, as their rows' values, each marked
    kept from last time; nothing else. The region's level and its directions, and the network, are science settings and
    open at their usual values, whatever an older build wrote under their keys. A budget that is no whole number is
    passed over, and costs the observation nothing; with no observation remembered, none is chosen.

    The last leg pins the defect class, not the instance: every page with a restore_settings of its own calls it from
    its own __init__."""
    import types
    from core import config
    from core.gui import settings as st
    from core.gui.panels.inference import tsnpe_tab
    from core.gui.panels.inference.config_tab import ConfigPanel
    from core.gui.panels.inference.infer_tab import InferPanel
    from core.gui.panels.inference.posterior_tab import PosteriorPanel
    from core.gui.panels.inference.prior_tab import PriorPanel
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel
    from core.gui.panels.inference.validate_tab import ValidatePanel
    from core.gui.widgets.artifact_picker import StorePicker
    from core.SBI import truncate as _tr
    from tests._fixtures import code_only, qt_app

    qt_app()
    # Two complete observations, so a restored pick is told from none. The store is stubbed at the list's one seam;
    # each row carries exactly what its refresh reads.
    listed = [types.SimpleNamespace(complete=True, finished=True, label=label, id=id_, created="2026-09-16T12:00:00",
                                    mode="spontaneous", width=50, amortized=None)
              for label, id_ in (("first", "20260916T120000"), ("second", "20260916T130000"))]
    store = types.SimpleNamespace(list=lambda kind: list(listed) if kind == "observation" else [])
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)
    usual = {"hpd_level": _tr.DEFAULT_HPD, "n_directions": _tr.DEFAULT_N_DIRECTIONS,
             "num_runs": config.TRAINING_NUM_RUNS, "run_size_cap": config.TRAINING_RUN_SIZE,
             "hidden_features": config.NSF_HIDDEN_FEATURES, "num_transforms": config.NSF_NUM_TRANSFORMS}

    def values(page):
        return {key: page.form.rows[key].value() for key in usual}

    # (a) an empty file: no observation chosen, every usual value. Each screen is HELD in a local: it owns its pages'
    # objects, and a dropped screen takes its pages with it.
    first = build_inference()
    page = first.tsnpe_panel
    assert page.obs_picker.key() == "" and values(page) == usual, (page.obs_picker.key(), values(page))

    # (b) what a round wrote, beside what an older build wrote under the science settings' keys: the observation and the
    # budget come back, kept from last time; the rest opens at its usual values
    qs = st.settings()
    for key, value in ((tsnpe_tab.OBSERVATION_KEY, "20260916T130000"), (tsnpe_tab.NUM_RUNS_KEY, "1234"),
                       (tsnpe_tab.RUN_SIZE_KEY, "512"), ("inference_tsnpe/hpd", "0.5"),
                       ("inference_tsnpe/n_dirs", "3"), ("inference_tsnpe/hpd_level", "0.5")):
        qs.setValue(key, value)
    qs.sync()
    second = build_inference()
    again = second.tsnpe_panel
    assert again.obs_picker.key() == "20260916T130000", "the observation is a selection: put back as the page is built"
    assert values(again) == {**usual, "num_runs": 1234, "run_size_cap": 512}, values(again)
    assert [again.form.rows[key].kept for key in ("observation", "num_runs", "run_size_cap", "hpd_level")] == [
        True, True, True, False]

    # (c) a budget that is no whole number: passed over, and the observation still comes back
    for key in (tsnpe_tab.NUM_RUNS_KEY, tsnpe_tab.RUN_SIZE_KEY):
        qs.setValue(key, "")
    qs.sync()
    third = build_inference()
    assert values(third.tsnpe_panel) == usual, values(third.tsnpe_panel)
    assert third.tsnpe_panel.obs_picker.key() == "20260916T130000", "a blank budget must not cost the observation"

    # (d) the defect class: a restore_settings its own __init__ never calls is dead code
    call = "self.restore_settings(settings.settings())"
    for cls in (ConfigPanel, PriorPanel, PosteriorPanel, ValidatePanel, InferPanel, TSNPEPanel):
        if "restore_settings" in vars(cls):
            assert call in code_only(cls.__init__), \
                f"{cls.__name__} defines restore_settings and its __init__ never calls it"
    assert "restore_settings" in vars(TSNPEPanel)
    assert "restore_settings" not in vars(ValidatePanel), \
        "the Checks page restores nothing, so it carries no restore_settings to forget to call"


def test_consents_are_never_persisted(tmp_path):
    """A CONSENT is answered per run and never remembered. The narrowing round's consent to a new simulation set beside a
    saved one a setting away is a question asked at the stage's refusal, and the Inference page holds no consent at all,
    so an answer an older page's tick wrote is read by nothing.

    Three legs: no restore_settings names a consent; neither page holds a tick of one, and a save writes nothing for
    either; an old PRISM.ini that holds both answers opens the round, whose start hands no consent."""
    import types
    from PySide6.QtCore import QSettings
    from core import orchestrator
    from core.gui import settings as st
    from core.gui.panels.inference.infer_tab import InferPanel
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel
    from tests._fixtures import code_only, qt_app

    qt_app()
    # (a) the source: executable code only, so a comment explaining the rule cannot trip it
    for cls, names in ((TSNPEPanel, ("new_run",)), (InferPanel, ("other_obs", "accept"))):
        src = code_only(cls.restore_settings)
        for name in names:
            assert name not in src, f"{cls.__name__}.restore_settings names the consent {name}"

    # (b) no tick, and nothing written at a save
    inf = build_inference()
    assert not hasattr(inf.tsnpe_panel, "new_run") and not hasattr(inf.infer_panel, "other_obs")
    out = QSettings(str(tmp_path / "saved.ini"), QSettings.IniFormat)
    for panel in (inf.tsnpe_panel, inf.infer_panel):
        panel.save_settings(out)
    out.sync()
    assert out.allKeys() == [], f"a page wrote at a save: {out.allKeys()}"

    # (c) an old PRISM.ini holding both answers: the round's start hands no consent
    qs = st.settings()
    for key, value in (("inference_tsnpe/new_run", "1"), ("inference_infer/other_obs", "1")):
        qs.setValue(key, value)
    qs.sync()
    again = build_inference().tsnpe_panel
    sent = []
    again.dispatch = lambda fn, *args, **kwargs: (sent.append((fn, kwargs)), False)[1]
    knobs = again._read_knobs()
    again._start(object(), object(), object(), types.SimpleNamespace(id="o1", name="o1"), knobs, new_run=False)
    assert sent and sent[-1][0] is orchestrator.tsnpe_round and sent[-1][1]["new_run"] is False, sent


def test_get_bool_falls_back_on_an_unparseable_value():
    """settings.get_bool read "1"/"true"/"True" as True and EVERYTHING else as False, so a blank or
    hand-edited value ("maybe", "2") restored as False whatever the caller's default said. For the
    remembered boolean with a True default that is the Fisher rotation silently OFF (the Config tab's
    reparam_rotate, config.REPARAM_ROTATE). get_int already fell back to its default on text it
    cannot parse; get_bool now does the same."""
    from core import config
    from core.gui import settings as st
    from core.gui.panels import inference_tabs as it
    from tests._fixtures import qt_app

    qt_app()
    qs = st.settings()
    for key, value in (("one", "1"), ("true", "true"), ("True", "True"), ("zero", "0"),
                       ("false", "false"), ("False", "False"), ("blank", ""), ("maybe", "maybe"),
                       ("two", "2")):
        qs.setValue(f"flags/{key}", value)
    qs.sync()
    fresh = st.settings()
    for default in (True, False):
        for key in ("one", "true", "True"):
            assert st.get_bool(fresh, f"flags/{key}", default) is True, (key, default)
        for key in ("zero", "false", "False"):
            assert st.get_bool(fresh, f"flags/{key}", default) is False, (key, default)
        for key in ("blank", "maybe", "two", "missing"):
            assert st.get_bool(fresh, f"flags/{key}", default) is default, (key, default)

    # On screen: a corrupt kind of observation opens at the usual one, and is not said to be kept from last time.
    qs.setValue("inference_config/chi_mode", "maybe")
    qs.sync()
    panel = build_part(it.ConfigPanel, screen=None)
    assert panel.kind_choice.currentData() is False and panel.form.rows["chi_mode"].kept is False, (
        "a corrupt kind of observation did not open at the usual one")
    assert panel.rot_choice.currentData() == config.REPARAM_ROTATE
