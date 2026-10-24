"""Saving a training as it goes and resuming it.

A training's fit (sbi's training of the neural posterior) is kept in a folder of its own: a header
written once, the state of the last finished pass and the one before it, and a lock file whose
operating-system lock says a process is training it now. What the fit was trained with is kept as its
identity: the rows and their split, the settings, a digest of the code that builds or trains the
network, the configuration's constants that code reads, the libraries, and the network's shape. A kept
fit is resumed only while every one of those is what it was.

Here: the two generations of the state and what a torn save leaves readable, the save's deferred-cancel
section, the lock seen from this process and from another, and each digest of the identity with the
first difference between two of them. Then the training call in its two halves: a fit stopped after a
pass and put back in a fresh trainer ends as the fit never stopped, saving after every pass changes
nothing the fit computes, the restore refuses a kept fit that differs, and the settings a kept fit
records are the ones sbi uses. Last, the training stage on a real tiny training: a training whose rows
are in a kept simulation set is saved after every pass, kept as a record of its own when it stops (by
a Ctrl-C, the window's Cancel or an error), removed once its posterior is written, and listed with
whether it can be resumed and why not. And resumed: on its own settings, into the very posterior the
training writes uninterrupted, refused before anything is read when anything it was kept with has
changed, marked as one that cannot be resumed when its rows read back do not give the kept fit, and
never resumed by two processes at once.
"""
import ast
import copy
import importlib
import inspect
import io
import json
import linecache
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import sbi
from sbi.inference import SNPE
from sbi.neural_nets.net_builders import flow
from core import config, runs
from core.Helpers import file_manager
from core.refusals import Refusal
from core.SBI import fit_state, pipeline, train
from core.SBI.fit_state import (FIT_CODE, FIT_CONSTANTS, SBI_TRAINER_FILES, STATE, STATE_PREV, Difference,
                                FitLock, code_digests, constants, first_difference, indices_digest,
                                libraries, load_state, network_digest, running, save_state,
                                trainer_files_digest)
from tests._fixtures import calls_that_speak, code_only


# ── the two generations of the state ─────────────────────────────────────────────────────────────
def test_a_pass_is_saved_with_the_one_before_it_kept_beside_it(tmp_path):
    """Each save moves the last pass to ``state.prev.pt`` and puts the new one at ``state.pt``, so the
    pass before the last is always one file away; ``load_state`` takes the newest."""
    save_state(tmp_path, {"passes": 1})
    save_state(tmp_path, {"passes": 2})
    assert torch.load(tmp_path / STATE)["passes"] == 2
    assert torch.load(tmp_path / STATE_PREV)["passes"] == 1
    state = load_state(tmp_path)
    assert (state["passes"], state["_state_file"]) == (2, "state.pt"), state
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("step", ["write", "first_rename", "second_rename", "unreadable"])
def test_a_save_torn_at_each_of_its_steps_leaves_the_last_good_pass_readable(tmp_path, monkeypatch, step):
    """A save is a write to a temporary file, a rename of the last pass to the previous one, then a
    rename of the temporary file to the last pass. Torn at the write or at the first rename, both
    files are as they were; torn between the renames, the last pass is read from the previous file; a
    last-pass file that cannot be read gives the pass before it. No temporary file is left either way."""
    save_state(tmp_path, {"passes": 1})
    save_state(tmp_path, {"passes": 2})
    kept = {name: (tmp_path / name).read_bytes() for name in (STATE, STATE_PREV)}
    renames = []
    if step == "unreadable":
        (tmp_path / STATE).write_bytes(kept[STATE][:10])
    else:
        real_replace = os.replace
        fail_at = {"first_rename": 1, "second_rename": 2}.get(step)

        def _torn_write(obj, fh, *args, **kwargs):
            fh.write(b"half a pass")
            raise OSError("the disk is full")

        def _replace(src, dst):
            renames.append(Path(dst).name)
            if len(renames) == fail_at:
                raise OSError("the volume went away")
            real_replace(src, dst)

        with monkeypatch.context() as patch:
            if step == "write":
                patch.setattr(file_manager.torch, "save", _torn_write)
            else:
                patch.setattr(file_manager.os, "replace", _replace)
            with pytest.raises(OSError):
                save_state(tmp_path, {"passes": 3})

    state = load_state(tmp_path)
    if step in ("write", "first_rename"):
        assert {name: (tmp_path / name).read_bytes() for name in (STATE, STATE_PREV)} == kept
        assert (state["passes"], state["_state_file"]) == (2, "state.pt"), state
    elif step == "second_rename":
        assert not (tmp_path / STATE).exists()
        assert (state["passes"], state["_state_file"]) == (2, "state.prev.pt"), state
    else:
        assert (state["passes"], state["_state_file"]) == (1, "state.prev.pt"), state
    if step in ("first_rename", "second_rename"):
        # the last pass moves aside first, the new one lands second
        assert renames == [STATE_PREV, STATE][:fail_at], renames
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("refused_at", [1, 2])
def test_a_rename_another_program_refuses_for_a_moment_is_tried_again(tmp_path, monkeypatch, refused_at):
    """On Windows a file another program holds for a moment (a virus scanner, say) refuses a rename;
    each of the save's two renames is tried again, so the save lands, with the pass before it kept
    beside it and no temporary file left."""
    save_state(tmp_path, {"passes": 1})
    save_state(tmp_path, {"passes": 2})
    renames, real_replace = [], os.replace

    def _replace(src, dst):
        renames.append(Path(dst).name)
        if len(renames) == refused_at:
            raise PermissionError(13, "another program holds the file")
        real_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(file_manager.os, "replace", _replace)
        save_state(tmp_path, {"passes": 3})
    assert load_state(tmp_path)["passes"] == 3
    assert torch.load(tmp_path / STATE_PREV, weights_only=False)["passes"] == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_save_runs_inside_the_deferred_cancel_section(tmp_path, monkeypatch):
    """A Cancel pressed in the window is taken at the next record the run writes; inside the section it
    waits, so no record can stop a save between its two renames."""
    seen = []
    real_save = fit_state.atomic_torch_save

    def _spy(*args, **kwargs):
        seen.append(runs.cancel_is_deferred())
        return real_save(*args, **kwargs)

    monkeypatch.setattr(fit_state, "atomic_torch_save", _spy)
    save_state(tmp_path, {"passes": 1})
    assert seen == [True]
    assert not runs.cancel_is_deferred()


def _run_by(fn) -> list:
    """``fn`` and every function of the program's own that its parsed code calls by name, and the ones
    those call in turn: what runs when ``fn`` does, as far as a name can tell."""
    found, todo = [], [fn]
    while todo:
        f = todo.pop()
        if f in found:
            continue
        found.append(f)
        for node in ast.walk(ast.parse(code_only(f))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                callee = f.__globals__.get(node.func.id)
                if inspect.isfunction(callee) and callee.__module__.startswith("core."):
                    todo.append(callee)
    return found


def test_nothing_prints_or_logs_inside_a_saved_pass():
    """Under the window every print and every record passes a cancel checkpoint, so one said inside the
    save would let a Cancel stop it between its two renames. Read on the parsed code of the save and of
    every function of the program's own it reaches by name, the atomic writer's helpers among them, so a
    helper the save gains later is read too, and a docstring that quotes the rule cannot trip it."""
    ran = _run_by(fit_state.save_state)
    reached = {f"{f.__module__}.{f.__name__}" for f in ran}
    assert {"core.SBI.fit_state.save_state", "core.Helpers.file_manager.atomic_torch_save",
            "core.Helpers.file_manager._atomic_write", "core.Helpers.file_manager._replace_retrying",
            "core.Helpers.file_manager._discard"} <= reached, reached
    for fn in ran:
        said = calls_that_speak(code_only(fn))
        assert said == [], f"{fn.__module__}.{fn.__name__} speaks inside a saved pass: {said}"
    # the scan sees each kind of call it should flag
    assert len(calls_that_speak("def f():\n    log.info('x')\n    print('y')\n    sys.stderr.write('z')\n")) == 3


# ── the lock ─────────────────────────────────────────────────────────────────────────────────────
_HOLDER = ("import sys, time\n"
           "from core.SBI.fit_state import FitLock\n"
           "lock = FitLock(sys.argv[1])\n"
           "print('held' if lock.acquire() else 'refused', flush=True)\n"
           "time.sleep(60)\n")


def _first_line(proc, timeout_s: float) -> str:
    """The child's first line of output, or a failure once ``timeout_s`` has passed without one."""
    lines = queue.Queue()
    threading.Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True).start()
    try:
        return lines.get(timeout=timeout_s).strip()
    except queue.Empty:
        pytest.fail(f"the child printed nothing in {timeout_s} s")


def test_a_held_lock_reads_as_running_here_and_in_another_process_until_its_holder_dies(tmp_path):
    """The lock is the operating system's, on one byte of the lock file, and it belongs to an open
    handle: another handle in the same process cannot take it, a process that dies lets it go, and a
    lock file nobody holds says only that a process once trained here."""
    assert not running(tmp_path)
    holder = FitLock(tmp_path)
    assert holder.acquire() and holder.held
    assert running(tmp_path)
    other = FitLock(tmp_path)
    assert not other.acquire() and not other.held
    holder.release()
    assert not holder.held and not running(tmp_path)
    holder.release()                                    # a second release does nothing

    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    with subprocess.Popen([sys.executable, "-c", _HOLDER, str(tmp_path)], cwd=str(config.REPO_ROOT),
                          env=env, stdout=subprocess.PIPE, text=True) as proc:
        try:
            assert _first_line(proc, 120) == "held"
            assert running(tmp_path)
        finally:
            proc.kill()
            proc.wait()
    # The system lets a dead process's lock go as it closes the process's handles; give it a moment.
    deadline = time.monotonic() + 10
    while running(tmp_path) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not running(tmp_path)


# ── the identity's digests ───────────────────────────────────────────────────────────────────────
_PROBE = ("def f(x):\n"
          "    # the first wording\n"
          "    return x + 1\n"
          "\n"
          "\n"
          "def g(x):\n"
          "    return x * 2\n")


def test_the_code_digest_moves_with_an_edit_inside_a_named_function_and_not_outside_it(tmp_path, monkeypatch):
    """A named function's digest is read from its source as the file stands, so an edit of its own
    lines moves it, a comment included, and an edit of another function in the same file, or of the
    lines above it, does not."""
    probe = tmp_path / "fit_code_probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        before = code_digests(("fit_code_probe.f",))
        assert set(before) == {"fit_code_probe.f"} and before["fit_code_probe.f"] != "unreadable", before

        probe.write_text(_PROBE.replace("return x * 2", "return 2 * x + 0"), encoding="utf-8")
        linecache.checkcache()
        assert code_digests(("fit_code_probe.f",)) == before, "an edit outside the function moved its digest"

        # Lines added above the function, read as a new process reads them, the module loaded afresh:
        # the function's text is the same wherever it now starts, and so is its digest.
        above = "import os\n\n\n"
        sys.modules.pop("fit_code_probe", None)
        probe.write_text(above + _PROBE, encoding="utf-8")
        linecache.checkcache()
        importlib.invalidate_caches()
        assert code_digests(("fit_code_probe.f",)) == before, "an edit above the function moved its digest"

        probe.write_text(above + _PROBE.replace("the first wording", "a second, longer wording"), encoding="utf-8")
        linecache.checkcache()
        assert code_digests(("fit_code_probe.f",)) != before, "an edit of the function's comment was not seen"
    finally:
        # Taken out of the module table here rather than through the monkeypatch fixture: the fixture
        # puts back at its teardown whatever it took out, which would leave the probe imported.
        sys.modules.pop("fit_code_probe", None)


def test_the_code_digest_kept_for_a_process_is_the_code_it_loaded(tmp_path, monkeypatch):
    """A process runs the code it imported, while its file can be edited afterwards and is then read
    as it stands. The digest a fit records is the one kept when the code was loaded, so a window left
    open across an edit does not record the edited file as the code it trains with."""
    importlib.import_module("core.orchestrator")       # its own remembering lands in the process's store
    monkeypatch.setattr(fit_state, "_LOADED", {})      # this test's store, put back at its end
    probe = tmp_path / "fit_code_probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    names = ("fit_code_probe.f",)
    try:
        fit_state.remember_loaded_code(names)
        kept = fit_state.loaded_code(names)
        assert kept == code_digests(names) and kept["fit_code_probe.f"] != "unreadable", kept

        probe.write_text(_PROBE.replace("the first wording", "a second, longer wording"), encoding="utf-8")
        linecache.checkcache()
        assert code_digests(names) != kept, "the edit on disk was not seen"
        assert fit_state.loaded_code(names) == kept, "the kept digest followed the file, not the loaded code"
        fit_state.remember_loaded_code(names)
        assert fit_state.loaded_code(names) == kept, "remembering again replaced the first digest"
    finally:
        sys.modules.pop("fit_code_probe", None)         # for the reason the test above gives


def test_the_training_code_keeps_its_digests_as_it_finishes_loading():
    """The module that builds and trains the network takes the digests of the named code as its last
    statement, once every named module has loaded, so the code a process keeps is the code it runs."""
    orchestrator = importlib.import_module("core.orchestrator")
    last = ast.parse(inspect.getsource(orchestrator)).body[-1]
    assert ast.unparse(last) == "fit_state.remember_loaded_code()", ast.unparse(last)
    kept = fit_state.loaded_code()
    assert set(kept) == set(FIT_CODE)
    assert [name for name, digest in kept.items() if digest == "unreadable"] == []


def test_the_named_code_is_the_code_that_builds_or_trains_the_network():
    """The three lists name what a kept fit's identity reads; one changed without this test is a list
    nobody checked against the code that builds or trains the network."""
    assert FIT_CODE == (
        "core.SBI.train.train_nn", "core.SBI.train.fit_settings", "core.SBI.train.build_trainer",
        "core.SBI.train.prepare_fit", "core.SBI.train.run_fit", "core.SBI.train.take_snapshot",
        "core.SBI.train.restore_trainer", "core.SBI.train._capped_zscore_check", "core.SBI.embedded_network._probit",
        "core.SBI.embedded_network._column_quantiles", "core.SBI.embedded_network.EmbeddedNet",
        "core.SBI.chi_encoder.ChiSetEncoder", "core.SBI.summaries.winsorize_summary_block",
        "core.SBI.chi.band_norm", "core.SBI.chi.n_chi_features", "core.orchestrator.build_embedding_net",
        "core.orchestrator.expected_forcing_dim")
    assert FIT_CONSTANTS == (
        "RANK_GAUSS_KNOTS", "WINSOR_PCT", "CHI_PHI_DIM", "CHI_BIN_DIM", "CHI_SET_OUT", "CHI_RHO_HIDDEN",
        "CHI_KNOTS", "CHI_KNOT_SIGMA", "CHI_KNOT_SHRINK", "CHI_ELEM_W", "CHI_LAYOUT")
    assert SBI_TRAINER_FILES == (
        "inference/trainers/base.py", "inference/trainers/npe/npe_base.py", "inference/trainers/npe/npe_c.py",
        "neural_nets/factory.py", "neural_nets/net_builders/flow.py")
    digests = code_digests()
    assert set(digests) == set(FIT_CODE)
    assert [name for name, digest in digests.items() if digest == "unreadable"] == []
    installed = Path(sbi.__file__).parent
    assert [rel for rel in SBI_TRAINER_FILES if not (installed / rel).is_file()] == []


def test_the_named_training_code_is_as_recorded():
    """The code that builds or trains the network is held to the digests it had once each epoch's record
    was added to the fit. A kept training is resumed only by the code it was trained with, so an edit
    inside any of the seventeen named functions and classes -- a comment or a docstring included -- makes
    every training kept before it unresumable. Such an edit, when it is meant, changes the digests below in
    the same commit and says so there; one that is not meant fails here first. An edit outside them, or
    lines added above them, moves nothing."""
    assert code_digests() == {
        "core.SBI.chi.band_norm": "4c840c4007eba80b",
        "core.SBI.chi.n_chi_features": "972304c5ef59acee",
        "core.SBI.chi_encoder.ChiSetEncoder": "9d6313ca9c5bca10",
        "core.SBI.embedded_network.EmbeddedNet": "828f05f8093a2d41",
        "core.SBI.embedded_network._column_quantiles": "68cfa160cbd35d71",
        "core.SBI.embedded_network._probit": "0702084a0af1e7cf",
        "core.SBI.summaries.winsorize_summary_block": "85ef9d1071d23560",
        "core.SBI.train._capped_zscore_check": "8ecc6471b985829f",
        "core.SBI.train.build_trainer": "a2dd52444b51f20a",
        "core.SBI.train.fit_settings": "cd193d63abdeb68b",
        "core.SBI.train.prepare_fit": "dc6cde65a46d9e69",
        "core.SBI.train.restore_trainer": "5a17377fb8a9ede6",
        "core.SBI.train.run_fit": "2b7a0bce64570300",
        "core.SBI.train.take_snapshot": "dbb919bbe095bb54",
        "core.SBI.train.train_nn": "2cf84a17feb813e0",
        "core.orchestrator.build_embedding_net": "45bb3140bcebcfc7",
        "core.orchestrator.expected_forcing_dim": "d27f2a643829e111",
    }


def _copy_trainer_files(root: Path, *, crlf: bool = False, change_base: bool = False) -> Path:
    installed = Path(sbi.__file__).parent
    for rel in SBI_TRAINER_FILES:
        data = bytearray((installed / rel).read_bytes())
        if crlf:
            data = bytearray(bytes(data).replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        if change_base and rel.endswith("/base.py"):
            at = next(i for i, byte in enumerate(data) if chr(byte).isalpha())
            data[at] ^= 0x20                            # one letter's case, so no line end is made
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(bytes(data))
    return root


def test_the_libraries_name_their_versions_and_a_digest_of_sbis_trainer_files(tmp_path):
    """The trainer is sbi's, so a kept fit names the versions it was trained under and a digest of the
    files of sbi's trainer, read with Windows line ends as Unix ones, so a reinstall that only rewrites
    the line ends is not a change."""
    libs = libraries()
    assert set(libs) == {"torch", "sbi", "nflows", "sbi_trainer_files"}
    assert libs["sbi"] == "0.26.1" and libs["nflows"] == "0.14" and libs["torch"] == torch.__version__, libs
    assert libs["sbi_trainer_files"] == trainer_files_digest()
    installed = trainer_files_digest()
    assert trainer_files_digest(_copy_trainer_files(tmp_path / "same")) == installed
    assert trainer_files_digest(_copy_trainer_files(tmp_path / "changed", change_base=True)) != installed
    assert trainer_files_digest(_copy_trainer_files(tmp_path / "windows", crlf=True)) == installed


def test_the_constants_are_read_as_the_configuration_stands(monkeypatch):
    """The constants are read when the identity is made, not when the module was imported, and kept
    in a form the record's text can carry."""
    now = constants()
    assert list(now) == list(FIT_CONSTANTS)
    assert now["WINSOR_PCT"] == [0.001, 0.999] and now["CHI_KNOTS"] == [-1.0, 0.0, 1.0], now
    monkeypatch.setattr(config, "WINSOR_PCT", (0.01, 0.99))
    assert constants()["WINSOR_PCT"] == [0.01, 0.99]
    json.dumps(constants())


def test_the_first_difference_is_named_by_its_part_and_its_name():
    """Two identities are compared part by part, in order, and a part's names one by one, so what a
    refusal names is the first thing that changed; a part the present cannot say is not compared."""
    kept = {"simulation": {"n_runs": 4, "prior_fingerprint": "a"}, "batches": 4,
            "settings": {"hidden_features": 8, "learning_rate": 0.001}, "network": "x"}

    def now(**parts):
        return {**copy.deepcopy(kept), **parts}

    assert first_difference(kept, now()) is None
    assert first_difference(kept, now(simulation={"n_runs": 4, "prior_fingerprint": "b"})) == \
        Difference("simulation", "prior_fingerprint", "a", "b")
    assert first_difference(kept, now(settings={"hidden_features": 8, "learning_rate": 0.0010000000000000002})) \
        == Difference("settings", "learning_rate", 0.001, 0.0010000000000000002)
    without_network = now()
    del without_network["network"]
    assert first_difference(kept, without_network) is None
    assert first_difference(kept, now(network="y")) == Difference("network", "network", "x", "y")
    assert first_difference(kept, now(batches=5)) == Difference("batches", "batches", 4, 5)
    assert first_difference(kept, now(batches=5, simulation={"n_runs": 5, "prior_fingerprint": "a"})) == \
        Difference("simulation", "n_runs", 4, 5)
    added = first_difference(kept, now(settings={"hidden_features": 8, "learning_rate": 0.001, "dropout": 0.1}))
    assert (added.part, added.name, added.now) == ("settings", "dropout", 0.1), added
    assert first_difference({"settings": {"band": (0.5, 2.0)}}, {"settings": {"band": [0.5, 2.0]}}) is None
    # Values are compared as the record's text carries them, so two NaNs are alike; a name only the kept
    # identity has is a difference too (a name the code list dropped), the kept identity's names are read
    # first and in their order, and a part only the present has is a difference.
    assert first_difference({"s": {"x": float("nan")}}, {"s": {"x": float("nan")}}, ("s",)) is None
    assert first_difference({"s": {"a": 1, "b": 2}}, {"s": {"b": 2}}, ("s",)) == Difference("s", "a", 1, None)
    assert first_difference({"s": {"a": 1, "b": 2}}, {"s": {"b": 3, "a": 4}}, ("s",)).name == "a"
    assert first_difference({}, {"network": "x"}, ("network",)) == Difference("network", "network", None, "x")


def test_the_network_and_the_split_digests_read_shapes_and_indices():
    """The network's digest reads the names and shapes of its weights, never their values; the split's
    reads the indices in their order, whatever integer type carries them."""
    a, b = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    with torch.no_grad():
        b.weight.add_(1.0)
    assert network_digest(a) == network_digest(b) and len(network_digest(a)) == 16
    assert network_digest(torch.nn.Linear(3, 4)) != network_digest(a)
    split = torch.tensor([3, 1, 2])
    assert indices_digest(split) != indices_digest(torch.tensor([1, 2, 3]))
    assert indices_digest(split.to(torch.int32)) == indices_digest(split)
    assert len(indices_digest(split)) == 16


# ── the training call in two halves: a save after every pass, and the restore ───────────────────────
class _Stopped(Exception):
    """What the test saver raises after the pass it was told to stop at: the fit ends there, as it does
    when the process training it dies."""


class _Saves:
    """A saver that keeps each pass's ``(snapshot, start)`` as the bytes ``torch.save`` writes, and stops
    the fit after the pass numbered ``stop_after``."""

    def __init__(self, stop_after: int | None = None):
        self.stop_after = stop_after
        self.saved: list[bytes] = []

    def save_pass(self, take) -> None:
        snapshot, start = take()
        written = io.BytesIO()
        torch.save((snapshot, start), written)
        self.saved.append(written.getvalue())
        if snapshot["passes"] == self.stop_after:
            raise _Stopped()

    def read(self, i: int = -1) -> tuple[dict, dict]:
        """Save ``i`` read back, as a new process reads a kept fit."""
        return torch.load(io.BytesIO(self.saved[i]), weights_only=False)

    def passes(self) -> list[int]:
        return [self.read(i)[0]["passes"] for i in range(len(self.saved))]


_KNOBS = dict(hidden_features=8, num_transforms=1, num_bins=4, learning_rate=5e-3, stop_after_epochs=3,
              max_num_epochs=40, batch_size=16)


def _toy(monkeypatch, tmp_path, rows: int = 96):
    """The cheapest real fit, in its two halves. The rows are drawn once, from a generator of their own,
    and the generation is replaced by copies of them, so every preparation appends the same rows in the
    same order, as a simulation set read back does. The stand-in draws one number from the global stream
    before it hands its copies back, as a real generation moves that stream, so a split whose generator
    state were read before the generation, and not just before sbi's training method, would differ on a
    resume. ``prepare(hooks)`` builds the trainer and appends the rows; ``fit(prepared, saver, resume)``
    runs sbi's passes; ``train_nn(**kwargs)`` is the whole training call; ``generated`` records each call
    of the generation."""
    from sbi.utils import BoxUniform

    monkeypatch.chdir(tmp_path)
    prior = BoxUniform(torch.zeros(2), torch.ones(2))
    draw = torch.Generator().manual_seed(1)
    theta = torch.rand(rows, 2, generator=draw)
    x = torch.cat([theta, theta.sum(1, keepdim=True)], 1) + 0.05 * torch.randn(rows, 3, generator=draw)
    generated = []

    def fake_gen(*args, **kwargs):
        generated.append(len(theta))
        torch.rand(1)                                   # the stream moves, as a real generation moves it
        return x.clone(), theta.clone()

    monkeypatch.setattr(pipeline, "gen_training_data", fake_gen)
    plan = pipeline.TrainingPlan(model="toy", prior=prior, t=torch.zeros(1), run_size=rows, num_runs=1,
                                 steady_idx=0, dt_nd_min=1.0, dt_exp=1.0, t_min_exp=1.0, t_max_exp=1.0,
                                 t_scale_bounds=(1.0, 1.0), dtype=torch.float32, device=torch.device("cpu"))
    flow_knobs = {k: _KNOBS[k] for k in ("hidden_features", "num_transforms", "num_bins")}
    fit_knobs = {k: _KNOBS[k] for k in ("batch_size", "learning_rate", "stop_after_epochs", "max_num_epochs")}

    def prepare(hooks: bool = True):
        net = torch.nn.Identity()
        infer = train.build_trainer("nsf", prior, net, **flow_knobs, device="cpu", hooks=hooks)
        return train.prepare_fit(infer, plan, "nsf", net, None, 2, {}, {}, **_KNOBS, device="cpu")

    def fit(prepared, saver=None, resume=None):
        return train.run_fit(prepared, **fit_knobs, show_train_summary=False, saver=saver, resume=resume)

    def train_nn(**kwargs):
        return pipeline.train_nn(plan, "nsf", prior, torch.nn.Identity(), None, 2, {}, {}, **_KNOBS, **kwargs)

    return types.SimpleNamespace(prepare=prepare, fit=fit, train_nn=train_nn, generated=generated)


def _history(infer) -> dict:
    """sbi's histories of the fit: the losses pass by pass, as Python floats, and its two entries per
    training call."""
    summary = infer._summary
    return {"training_loss": [float(v) for v in summary["training_loss"]],
            "validation_loss": [float(v) for v in summary["validation_loss"]],
            "epochs_trained": list(summary["epochs_trained"]),
            "best_validation_loss": [float(v) for v in summary["best_validation_loss"]]}


def _unequal(a: dict, b: dict) -> list[str]:
    """The names of two state dicts whose tensors are not equal, or that only one of them has."""
    return sorted(set(a) ^ set(b)) + [name for name in a if name in b and not torch.equal(a[name], b[name])]


@pytest.mark.parametrize("where", ["first_pass", "no_improvement", "last_but_one"])
def test_a_fit_stopped_after_a_pass_and_resumed_in_a_fresh_trainer_ends_as_the_uninterrupted_one(
        monkeypatch, tmp_path, where):
    """A fit stopped after a pass, then put back in a fresh trainer (another random stream, the rows
    appended again, the network built again) from what its saver kept, ends exactly as the fit never
    stopped: the same losses pass by pass, the same counters, the same final and best weights. Stopped
    after the first pass; after the pass that follows the best one, the first of the passes without
    improvement that end the fit by its patience (where sbi's own resume within one process counts that
    pass twice and stops a pass early); and one pass before the end. The resumed fit goes on saving from
    the next pass under the start it was kept with. This is also the alarm for an sbi whose trainer no
    longer resumes this way. The stopped trainer and the resumed one each get their own per-pass summary
    back once the fit ends."""
    toy = _toy(monkeypatch, tmp_path)
    torch.manual_seed(0)
    reference = toy.prepare(hooks=False)
    toy.fit(reference)
    ref = reference.infer
    passes = ref._summary["epochs_trained"][-1]
    assert passes < 41, f"the uninterrupted fit ran to its ceiling ({passes} passes), not to its patience"
    losses = ref._summary["validation_loss"]
    best = losses.index(min(losses)) + 1                # the pass sbi keeps as the best
    stop = {"first_pass": 1, "no_improvement": best + 1, "last_but_one": passes - 1}[where]
    assert 1 <= stop < passes, (where, stop, passes)

    torch.manual_seed(0)
    before = _Saves(stop_after=stop)
    stopped = toy.prepare()
    with pytest.raises(_Stopped):
        toy.fit(stopped, saver=before)
    assert "_summarize_epoch" not in vars(stopped.infer), "the stopped trainer kept the saving wrapper"
    assert before.passes() == list(range(1, stop + 1))
    snapshot, start = before.read()

    torch.manual_seed(999)
    resumed = toy.prepare()
    after = _Saves()
    toy.fit(resumed, saver=after, resume=(start, snapshot))
    got = resumed.infer
    assert "_summarize_epoch" not in vars(got), "the resumed trainer kept the saving wrapper"
    assert _history(got) == _history(ref)
    assert len(got._summary["epochs_trained"]) == 1
    assert got._epochs_since_last_improvement == ref._epochs_since_last_improvement
    assert _unequal(got._neural_net.state_dict(), ref._neural_net.state_dict()) == []
    assert _unequal(got._best_model_state_dict, ref._best_model_state_dict) == []
    assert after.passes() == list(range(stop + 1, passes + 1))
    later = after.read()[1]
    assert torch.equal(later["split_rng"], start["split_rng"])
    assert {k: later[k] for k in ("validation_digest", "rows", "network", "settings")} == \
        {k: start[k] for k in ("validation_digest", "rows", "network", "settings")}


def test_saving_every_pass_changes_nothing_the_fit_computes(monkeypatch, tmp_path):
    """The save only reads: a fit saved after every pass computes what the same fit computes unsaved in
    sbi's own loop, its per-pass summary not wrapped at all, the same losses and weights, and leaves the
    random stream where that one leaves it. The saver is handed every pass, numbered from one, in order,
    and the trainer gets its own per-pass summary back once the fit ends. sbi's training method is handed
    exactly the settings the kept fit records for it, and nothing else, so a kept fit's settings are the
    ones it trained with."""
    toy = _toy(monkeypatch, tmp_path)
    torch.manual_seed(0)
    plain = toy.prepare(hooks=False)
    with monkeypatch.context() as patch:
        patch.setattr(train, "_wrap_summary", lambda infer, saver: False)      # sbi's own loop
        toy.fit(plain)
    stream_plain = torch.get_rng_state()

    asked, real_train = [], SNPE.train

    def recorded(self, *args, **kwargs):
        asked.append((args, kwargs))
        return real_train(self, *args, **kwargs)

    torch.manual_seed(0)
    saved = toy.prepare()
    saves = _Saves()
    with monkeypatch.context() as patch:
        patch.setattr(SNPE, "train", recorded)
        toy.fit(saved, saver=saves)
    stream_saved = torch.get_rng_state()
    assert "_summarize_epoch" not in vars(saved.infer), "the trainer kept the saving wrapper"
    kept = saved.settings
    assert asked == [((), {"training_batch_size": kept["training_batch_size"], "learning_rate": kept["learning_rate"],
                           "stop_after_epochs": kept["stop_after_epochs"], "max_num_epochs": kept["max_num_epochs"],
                           "show_train_summary": False})], asked

    assert _history(saved.infer) == _history(plain.infer)
    assert _unequal(saved.infer._neural_net.state_dict(), plain.infer._neural_net.state_dict()) == []
    assert torch.equal(stream_saved, stream_plain), "a save drew a random number"
    assert saves.passes() == list(range(1, plain.infer._summary["epochs_trained"][-1] + 1))


@pytest.mark.parametrize("what, said", [
    ("settings", ("z_score_x", "summaries' z-scoring")),
    ("rows", ("its rows number 96, not the 97",)),
    ("split", ("validation rows",)),
    ("network", ("differs in shape",)),
    ("standardisers", ("standardisers",)),
])
def test_the_restore_refuses_what_differs_from_the_kept_fit(monkeypatch, tmp_path, what, said):
    """Each check the restore makes once the rows are read refuses a kept fit that differs, naming what
    differs, and refuses before the optimizer exists, so nothing of the kept fit is put back."""
    toy = _toy(monkeypatch, tmp_path)
    torch.manual_seed(0)
    stopped = toy.prepare()
    saves = _Saves(stop_after=2)
    with pytest.raises(_Stopped):
        toy.fit(stopped, saver=saves)
    snapshot, start = saves.read()
    if what == "settings":
        start["settings"]["z_score_x"] = "none"
    elif what == "rows":
        start["rows"] += 1
    elif what == "split":
        start["validation_digest"] = "0" * 16
    elif what == "network":
        start["network"] = "0" * 16
    else:
        name = next(name for name, buffer in stopped.infer._neural_net.named_buffers()
                    if buffer.is_floating_point() and name in snapshot["net_state"])
        snapshot["net_state"][name] = snapshot["net_state"][name] + 1

    fresh = toy.prepare()
    with pytest.raises(fit_state.FitMismatch) as refused:
        train.restore_trainer(fresh.infer, start, snapshot, settings=fresh.settings)
    assert any(words in str(refused.value) for words in said), refused.value
    assert not hasattr(fresh.infer, "optimizer"), "the refusal came after the optimizer was made"


def test_saving_a_fit_needs_the_trainers_hook_points(monkeypatch, tmp_path):
    """Saving a fit wraps and calls private parts of sbi's trainer, so a trainer without one of them is
    refused as it is built, before any row exists, naming the part and the sbi installed: an sbi other
    than the one pinned stops the run at once, not at the first save or at a resume."""
    with pytest.raises(RuntimeError) as missing:
        train._require_hook_points(types.SimpleNamespace())
    assert "_summarize_epoch" in str(missing.value) and sbi.__version__ in str(missing.value), missing.value
    # Each of the parts alone: the per-pass summary, the network builder, the stored rows, the loaders,
    # the patience check, and the histories, which must be a dict of lists.
    called = ("_summarize_epoch", "_build_neural_net", "get_simulations", "get_dataloaders", "_converged")

    def whole():
        return types.SimpleNamespace(_summary={}, **{name: (lambda *a, **k: None) for name in called})

    train._require_hook_points(whole())
    for gone in (*called, "_summary"):
        trainer = whole()
        if gone == "_summary":
            trainer._summary = []
        else:
            delattr(trainer, gone)
        with pytest.raises(RuntimeError, match=f"has no {gone},"):
            train._require_hook_points(trainer)

    class _WithoutItsBuilder(SNPE):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            del self._build_neural_net

    toy = _toy(monkeypatch, tmp_path)
    monkeypatch.setattr(train, "SNPE", _WithoutItsBuilder)
    with pytest.raises(RuntimeError, match="_build_neural_net") as refused:
        toy.train_nn(fit_saver=_Saves())
    assert sbi.__version__ in str(refused.value)
    assert any(entry.name == "build_trainer" for entry in refused.traceback), "not refused as the trainer is built"
    assert toy.generated == []


def test_the_fit_settings_are_the_values_sbi_will_use_its_own_defaults_included(monkeypatch):
    """A kept fit records every setting it is trained with, the ones PRISM leaves to sbi included, as sbi
    resolves them: its training method's as installed, its posterior network factory's and its flow
    builder's. The training method's are read when the module is imported, so wrapping the method on the
    class, as some tests do, changes nothing; a builder without a setting records None for it."""
    asked = dict(model="nsf", hidden_features=8, num_transforms=1, num_bins=10, learning_rate=0.001,
                 stop_after_epochs=20, max_num_epochs=2_147_483_647, batch_size=512, z_score_x="none",
                 device="cpu", dtype=torch.float32)
    expected = {"hidden_features": 8, "num_transforms": 1, "learning_rate": 0.001, "stop_after_epochs": 20,
                "max_num_epochs": 2_147_483_647, "density_estimator": "nsf", "num_bins": 10,
                "training_batch_size": 512, "num_rounds": 1, "validation_fraction": 0.1, "clip_max_norm": 5.0,
                "z_score_theta": "independent", "z_score_x": "none", "tail_bound": 3.0,
                "hidden_layers_spline_context": 1, "num_blocks": 2, "dropout_probability": 0.0,
                "use_batch_norm": False, "data_device": "cpu", "device": "cpu", "dtype": "torch.float32"}
    assert list(train.fit_settings(**asked).items()) == list(expected.items())

    def wider_tails(batch_x, batch_y, tail_bound: float = 4.0, hidden_layers_spline_context: int = 1,
                    num_blocks: int = 2, dropout_probability: float = 0.0, use_batch_norm: bool = False,
                    **kwargs):
        raise AssertionError("the settings read the builder's signature; they never build")

    with monkeypatch.context() as patch:
        patch.setattr(flow, "build_nsf", wider_tails)
        assert train.fit_settings(**asked) == {**expected, "tail_bound": 4.0}
    with monkeypatch.context() as patch:
        patch.setattr(SNPE, "train", lambda self, *a, **k: None)
        assert train.fit_settings(**asked) == expected
    maf = train.fit_settings(**{**asked, "model": "maf"})
    assert maf == {**expected, "density_estimator": "maf", "tail_bound": None, "hidden_layers_spline_context": None}


def test_saving_a_fit_is_refused_for_more_than_one_round(monkeypatch, tmp_path):
    """A later round's rows are drawn from the round before's posterior, which a kept fit's identity does
    not hold, so saving or resuming a fit of more than one round is refused before any row is made."""
    toy = _toy(monkeypatch, tmp_path)
    for given in ({"fit_saver": _Saves()}, {"fit_resume": ({}, {})}):
        with pytest.raises(ValueError, match=r"num_rounds=2") as refused:
            toy.train_nn(num_rounds=2, x_obs=torch.zeros(1, 3), **given)
        assert "as it goes" in str(refused.value), refused.value
    assert toy.generated == []


# ── the training stage: a training saved after every pass, kept when it stops ────────────────────────
# Each test below trains a tiny network on the module's prior, on a simulation set of its own (its own
# number of batches and rows per batch, so no two tests share one by accident), with its rows kept:
# checkpointing every batch. A batch costs about a second of simulation whatever its width, so the sets
# are two or three batches wide, and the rotation is measured at one point with a small ensemble.
_TRAINING = dict(hidden_features=8, num_transforms=1, stop_after_epochs=3, max_num_epochs=8,
                 checkpoint_every=1, new_run=True)
_ROTATION = dict(fisher_points=1, fisher_m=4)


@pytest.fixture(autouse=True, scope="module")
def _no_training_of_the_configured_size():
    """Every training in this module simulates a handful of batches. One that reaches the simulation with
    more has fallen back on the configured budget of thousands of batches -- a resume that lost the kept
    training it names, or a narrowing round that no longer hands it on -- and fails here at once, instead
    of simulating for hours before an assertion can turn it red."""
    real = pipeline.gen_training_data

    def guarded(*args, **kwargs):
        batches = args[5] if len(args) > 5 else kwargs.get("n_runs")
        assert batches is not None and batches <= 10, (
            f"a training of {batches} batches reached the simulation: the configured budget, not this test's")
        return real(*args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(pipeline, "gen_training_data", guarded)
    yield
    patch.undo()


def _train(r, batches: int, width: int, **given):
    """A tiny training on ``tiny_run``'s prior, on ``batches`` batches of ``width`` rows, written into
    its store."""
    from core import orchestrator
    return orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=r.sink, store=r.store,
                                        num_runs=batches, run_size_cap=width,
                                        **{**_TRAINING, **_ROTATION, **given})


@pytest.fixture
def fits_left(tiny_run):
    """The folders of kept trainings in the module's store when the test starts. When it ends, whatever
    its outcome, every kept training it added is deleted (and a folder it left without a record is
    removed), so one test's kept training never reaches another's listing, ordering or matching line.
    Nothing is released here: a kept training a test left locked is the test's fault, and the teardown
    fails naming it once the others are gone."""
    root = tiny_run.store.kind_dir("fit")
    before = {p.name for p in root.iterdir()} if root.is_dir() else set()
    yield before
    locked = []
    for sub in sorted(root.iterdir()) if root.is_dir() else []:
        if sub.name in before:
            continue
        if running(sub):
            locked.append(sub.name)
        elif (sub / "manifest.json").is_file():
            tiny_run.store.delete("fit", sub.name.rsplit("__", 1)[-1])
        else:
            shutil.rmtree(sub)
    if locked:
        pytest.fail(f"the test left kept training(s) locked: {locked}")


def _new_fits(r, before) -> list:
    """The store's rows of the kept trainings added since ``before`` (what ``fits_left`` holds), newest
    first."""
    return [s for s in r.store.list("fit") if s.dir_name not in before]


def _added_folders(r, before) -> list:
    """Every folder under the kept trainings' directory added since ``before``, a record or not."""
    root = r.store.kind_dir("fit")
    return sorted(p.name for p in root.iterdir() if p.name not in before) if root.is_dir() else []


def _stop_after(patch, at: int, exc: BaseException) -> None:
    """Stop the training with ``exc`` right after pass ``at`` is saved, as a Ctrl-C, the window's Cancel
    or an error arriving between two passes would: the pass's save runs whole (its state, its record and
    its line), then ``exc`` is raised where the trainer hands the pass on."""
    real = fit_state.FitSession.save_pass

    def save_pass(self, take):
        real(self, take)
        if self.passes == at:
            raise exc

    patch.setattr(fit_state.FitSession, "save_pass", save_pass)


class _SaidInsideASave(logging.Handler):
    """Collects, while it is entered, every record the program's own loggers emit while a deferred-cancel
    section is open on this thread: under the window each would pass a cancel checkpoint inside a save
    that has to finish. Read at run time, so a record from a helper called in any form is seen."""

    def __init__(self):
        super().__init__()
        self.said = []

    def emit(self, record):
        if getattr(runs._deferred, "depth", 0) > 0:
            self.said.append(f"{record.name}: {record.getMessage()}")

    def __enter__(self):
        logging.getLogger("core").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("core").removeHandler(self)


def _log(post) -> str:
    return (post.path / "log.txt").read_text(encoding="utf-8")


def _said(text: str, level: str, prefix: str) -> list:
    """The messages of the records in ``text`` at ``level`` that start with ``prefix``, in order."""
    return [line.split(f" {level} ", 1)[1] for line in text.splitlines() if f" {level} {prefix}" in line]


def _said_but_epochs(text: str, level: str, prefix: str) -> list:
    """``_said`` without the records of each epoch's numbers (they begin ``[fit] epoch ``): what the
    training says about saving it, which the lists that use this hold whole."""
    return [said for said in _said(text, level, prefix) if not said.startswith("[fit] epoch ")]


def _posterior_names(r) -> set:
    return {s.name for s in r.store.list("posterior")}


def test_a_checkpointed_training_saves_every_pass_and_removes_its_kept_fit_after_the_posterior(
        tiny_run, fits_left, monkeypatch):
    """A training whose rows are in a kept simulation set is saved after every pass, into one folder, and
    says so once a pass; once its posterior is written the kept training is removed, and the posterior
    records which kept training it was, in how many processes and over how long. Nothing is said while
    a pass is being saved."""
    r = tiny_run
    saves, real = [], fit_state.save_state

    def spy(path, state):
        saves.append((Path(path), state["passes"]))
        real(path, state)

    monkeypatch.setattr(fit_state, "save_state", spy)
    with _SaidInsideASave() as inside:
        post = _train(r, 2, 8, name="fit_every_pass")
    assert inside.said == [], f"said while a pass was being saved: {inside.said}"
    training = post.manifest.body["training"]
    passes = training["epochs_trained"]
    assert [n for _, n in saves] == list(range(1, passes + 1)), saves
    folder = saves[0][0]
    assert {p for p, _ in saves} == {folder}
    assert _said(_log(post), "info", "[fit] saved") == [f"[fit] saved after pass {n}." for n in range(1, passes + 1)]
    assert not folder.exists() and _new_fits(r, fits_left) == []
    assert training["resumed_fit"] is False and training["fit_processes"] == 1
    assert training["fit_id"] == folder.name.rsplit("__", 1)[-1]
    assert training["fit_wall_seconds"] > 0


def test_a_training_stopped_on_purpose_is_kept_and_says_how_it_stopped(tiny_run, fits_left, monkeypatch):
    """A Ctrl-C after the second pass's save keeps the training as a record of its own: stopped on
    purpose, after pass 2, with the pass before it one file away, its identity whole and the same in its
    header and its record, its code the code as the process loaded it (made here to differ from the files
    as they stand, as after an edit made while a window is open), its lock let go, its log up to the
    stop, and the simulation set it trains on protected from deletion. No posterior is written."""
    r = tiny_run
    real_code = fit_state.loaded_code

    def as_loaded(names=FIT_CODE):
        return {**real_code(names), "core.orchestrator.build_embedding_net": "1" * 16}

    monkeypatch.setattr(fit_state, "loaded_code", as_loaded)
    _stop_after(monkeypatch, 2, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _train(r, 2, 7, name="fit_stopped")
    row, = _new_fits(r, fits_left)
    assert (row.fit_state, row.fit_passes) == ("kept when stopped", 2)
    m = r.store.get("fit", row.id)
    assert m.body["stopped"]["on_purpose"] is True and m.body["complete"] is False
    sim = m.parents["simulation"]
    assert m.parents == {"prior": r.prior.id, "simulation": sim}
    assert r.store.get("simulation", sim).body["complete"] is True
    assert torch.load(row.path / STATE, weights_only=False)["passes"] == 2
    assert torch.load(row.path / STATE_PREV, weights_only=False)["passes"] == 1
    # How far it went, as the record says it, read from the pass's own losses: after two passes the best
    # pass's place and the passes since it are never the same number.
    losses = [float(v) for v in torch.load(row.path / STATE, weights_only=False)["summary"]["validation_loss"]]
    best = min(losses)
    assert {k: m.body["progress"][k] for k in ("passes", "best_validation_loss", "passes_since_best",
                                                "last_validation_loss")} == \
        {"passes": 2, "best_validation_loss": best, "passes_since_best": len(losses) - 1 - losses.index(best),
         "last_validation_loss": losses[-1]}, (m.body["progress"], losses)
    header = fit_state.read_header(row.path)
    assert list(header["identity"]) == list(fit_state.IDENTITY_PARTS)
    loaded = as_loaded()
    assert loaded != code_digests()
    assert header["identity"]["code"] == loaded, "the training must record the code as this process loaded it"
    assert m.body["identity"]["code"] == loaded
    assert first_difference(header["identity"], m.body["identity"]) is None
    assert "started" not in header["start"]
    assert not running(row.path)
    assert "[fit] saved after pass 2." in (row.path / "log.txt").read_text(encoding="utf-8")
    assert "fit_stopped" not in _posterior_names(r)
    with pytest.raises(Refusal) as refused:
        r.store.delete("simulation", sim)
    assert refused.value.field == "artifact" and row.id in str(refused.value)


def test_a_cancel_from_the_window_lands_after_the_pass_is_saved_and_keeps_the_training(
        tiny_run, fits_left, monkeypatch):
    """The window's Cancel, pressed while a pass is being saved, waits for the save and is taken at the
    first record after it: the training is kept, stopped on purpose, after that pass, its state whole in
    ``state.pt``, and the window hears that it is kept. Driven through the window's own channel: its
    handler on the ``core`` logger is then the only place in the process where a cancel is taken."""
    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler
    r = tiny_run
    token = CancelToken()
    token.arm()
    heard = []
    handler = _PumpLogHandler(types.SimpleNamespace(sink=lambda kind, payload: heard.append(payload)), token)
    real = fit_state.save_state

    def save_then_cancel(path, state):
        real(path, state)
        if state["passes"] == 2:
            token.requested.set()                       # Cancel pressed; nothing has taken it yet

    monkeypatch.setattr(fit_state, "save_state", save_then_cancel)
    core_logger = logging.getLogger("core")
    core_logger.addHandler(handler)
    try:
        with pytest.raises(WorkerCancelled):
            _train(r, 2, 6, name="fit_cancelled")
    finally:
        core_logger.removeHandler(handler)
    assert token.fired is True
    row, = _new_fits(r, fits_left)
    assert (row.fit_state, row.fit_passes) == ("kept when stopped", 2)
    state = load_state(row.path)
    assert (state["passes"], state["_state_file"]) == (2, STATE)
    assert any(text.startswith(f"[fit] the training stopped after pass 2 and is kept as fit {row.id}")
               for text, _level in heard), heard


def test_a_training_that_fails_is_kept_as_stopped_unexpectedly_with_its_reason(tiny_run, fits_left, monkeypatch):
    """An error after the second pass's save keeps the training too: stopped unexpectedly, with the error
    as its reason, its lock let go."""
    r = tiny_run
    _stop_after(monkeypatch, 2, RuntimeError("the flow diverged"))
    with pytest.raises(RuntimeError, match="the flow diverged"):
        _train(r, 2, 5, name="fit_failed")
    row, = _new_fits(r, fits_left)
    assert (row.fit_state, row.fit_passes) == ("stopped unexpectedly", 2)
    stopped = r.store.get("fit", row.id).body["stopped"]
    assert stopped["on_purpose"] is False and "the flow diverged" in stopped["reason"], stopped
    assert not running(row.path)


def test_a_stop_before_the_first_pass_is_saved_keeps_nothing(tiny_run, fits_left, monkeypatch, caplog):
    """A training stopped before its first pass was saved leaves no kept training and no posterior, and
    the run says that no kept training was made."""
    r = tiny_run

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "run_fit", interrupted)
    caplog.set_level(logging.INFO, logger="core")
    with pytest.raises(KeyboardInterrupt):
        _train(r, 2, 4, name="fit_never_saved")
    assert _added_folders(r, fits_left) == []
    assert "fit_never_saved" not in _posterior_names(r)
    said = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("[fit] ")]
    assert said[-1:] == ["[fit] the training stopped before its fit saved a pass, so no kept training was made; "
                         "the simulation set keeps the batches it committed."], said


def test_a_posterior_that_cannot_be_written_leaves_its_finished_training_kept_and_unlocked(
        tiny_run, fits_left, monkeypatch):
    """A training that ended but whose posterior could not be written (a full disk after days of
    training) is kept, finished, with the failure as its reason and its lock let go: it can be resumed,
    which writes the posterior at once, or deleted, without restarting the program."""
    r = tiny_run
    real_create = r.store.create

    def create(kind, *args, **kwargs):
        if kind == "posterior":
            raise OSError("No space left on device")
        return real_create(kind, *args, **kwargs)

    monkeypatch.setattr(r.store, "create", create)
    with pytest.raises(OSError, match="No space left on device"):
        _train(r, 3, 8, name="fit_unwritten")
    row, = _new_fits(r, fits_left)
    assert row.fit_state == "finished"
    body = r.store.get("fit", row.id).body
    assert body["complete"] is True and "No space left on device" in body["stopped"]["reason"], body["stopped"]
    assert not running(row.path)
    assert "fit_unwritten" not in _posterior_names(r)


def test_a_save_that_fails_warns_and_the_training_goes_on(tiny_run, fits_left, monkeypatch):
    """A save that fails (a full disk, a file another program holds) never stops the training: each
    failed pass warns once with the pass the kept training stands at, the posterior is written, and the
    kept training is removed as usual. A save that fails from the first pass on, or a pass that cannot
    even be taken (an sbi that renamed what the snapshot reads), warns that nothing is kept yet, and
    leaves no folder behind, also when the training is stopped right after such a save."""
    r = tiny_run
    real = fit_state.save_state
    first_failing = {"pass": 2}

    def full_disk(path, state):
        if state["passes"] >= first_failing["pass"]:
            raise OSError("No space left on device")
        real(path, state)

    monkeypatch.setattr(fit_state, "save_state", full_disk)
    post = _train(r, 3, 7, name="fit_disk_full")
    passes = post.manifest.body["training"]["epochs_trained"]
    assert _said(_log(post), "warning", "[fit] ") == [
        "[fit] the last save failed: OSError: No space left on device; the kept training stands at pass 1."
    ] * (passes - 1)
    assert _said(_log(post), "info", "[fit] saved") == ["[fit] saved after pass 1."]
    assert post.manifest.body["training"]["fit_id"] is not None
    assert _new_fits(r, fits_left) == []

    first_failing["pass"] = 1
    never = _train(r, 2, 3, name="fit_disk_full_from_the_start")
    passes = never.manifest.body["training"]["epochs_trained"]
    assert _said(_log(never), "warning", "[fit] ") == [
        "[fit] the last save failed: OSError: No space left on device; nothing is kept yet."] * passes
    assert (never.manifest.body["training"]["fit_id"], never.manifest.body["training"]["fit_wall_seconds"]) == \
        (None, None)
    assert _added_folders(r, fits_left) == []
    # Stopped (a Ctrl-C) right after a first save that failed: the folder that save made goes with it.
    with monkeypatch.context() as patch:
        _stop_after(patch, 0, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            _train(r, 3, 2, name="fit_disk_full_then_stopped")
    assert _added_folders(r, fits_left) == [], "the stop left the folder its failed first save made"

    def renamed(infer):
        raise AttributeError("'NPE_C' object has no attribute '_best_val_loss'")

    monkeypatch.setattr(train, "take_snapshot", renamed)
    untaken = _train(r, 2, 2, name="fit_never_taken")
    passes = untaken.manifest.body["training"]["epochs_trained"]
    assert _said(_log(untaken), "warning", "[fit] ") == [
        "[fit] the last save failed: AttributeError: 'NPE_C' object has no attribute '_best_val_loss'; "
        "nothing is kept yet."] * passes
    assert _added_folders(r, fits_left) == []


def test_without_a_kept_simulation_set_nothing_is_saved_and_the_run_says_so(tiny_run, fits_left, monkeypatch):
    """With checkpointing off the rows are not kept, so the training is not saved and says so before its
    fit; asked to save all the same, the stage refuses before any row is made, naming the cadence."""
    from core import orchestrator
    r = tiny_run
    post = _train(r, 2, 8, checkpoint_every=0, name="fit_not_kept")
    assert _added_folders(r, fits_left) == []
    training = post.manifest.body["training"]
    assert training["fit_id"] is None and training["fit_wall_seconds"] is None
    assert _said_but_epochs(_log(post), "info", "[fit] ") == [
        "[fit] the simulation set is not kept (checkpointing off), so this training is not saved as it goes "
        "and cannot be resumed."]
    generated = []
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data", lambda *a, **k: generated.append(1))
    with pytest.raises(Refusal) as refused:
        _train(r, 2, 8, checkpoint_every=0, save_fit=True)
    assert refused.value.field == "checkpoint_every" and generated == []


def test_save_fit_false_saves_nothing(tiny_run, fits_left, monkeypatch):
    """Saving switched off saves nothing and says so. And the same training saved after every pass, from
    the same seed on the same rows, writes the very posterior the unsaved one writes: saving changes
    nothing a fit computes."""
    r = tiny_run
    saved, real = [], fit_state.save_state

    def spy(path, state):
        saved.append(state["passes"])
        real(path, state)

    monkeypatch.setattr(fit_state, "save_state", spy)
    _train(r, 3, 6, save_fit=False, name="fit_rows_made")               # makes the simulation set
    torch.manual_seed(5)
    off = _train(r, 3, 6, save_fit=False, name="fit_unsaved")
    assert saved == []
    assert _said_but_epochs(_log(off), "info", "[fit] ") == [
        "[fit] saving is switched off: this training is not saved as it goes and cannot be resumed."]
    assert off.manifest.body["training"]["fit_id"] is None
    torch.manual_seed(5)
    on = _train(r, 3, 6, name="fit_saved")
    assert saved == list(range(1, on.manifest.body["training"]["epochs_trained"] + 1)), saved
    assert on.diagnostics["validation_loss"] == off.diagnostics["validation_loss"]
    assert _unequal(on.latent.posterior_estimator.state_dict(), off.latent.posterior_estimator.state_dict()) == []


def test_a_new_training_names_the_kept_trainings_that_match_its_settings(tiny_run, fits_left, monkeypatch):
    """A training started with the settings of a kept one, on the same rows, names it before its fit as
    one that can be resumed, and starts a new one all the same; a training with other settings names
    none. A kept training's record that cannot be read (its manifest saved in another encoding) is listed
    as unreadable and is no kept training to resume: beside it a training names the kept ones as before,
    saves every pass and removes its own record as usual. And the line only informs: when the kept
    trainings cannot be listed at all, a training warns once and trains and writes its posterior all the
    same."""
    r = tiny_run
    with monkeypatch.context() as patch:
        _stop_after(patch, 1, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            _train(r, 3, 5, name="fit_first")
    first, = _new_fits(r, fits_left)
    second = _train(r, 3, 5, name="fit_second")
    assert _said(_log(second), "info", "[fit] 1 kept") == [
        f"[fit] 1 kept training(s) match these settings and can be resumed: {first.id} (kept when stopped, "
        f"after pass 1); this run starts a new one."]
    third = _train(r, 3, 5, hidden_features=16, name="fit_third")
    assert "match these settings" not in _log(third)
    assert [s.id for s in _new_fits(r, fits_left)] == [first.id]

    damaged = r.store.kind_dir("fit") / "_unnamed__20000101T000000"
    damaged.mkdir()
    (damaged / "manifest.json").write_bytes(b'{"kind": "fit", "note": "\xff\xfe"}')
    saves, real_save = [], fit_state.save_state

    def spy(path, state):
        saves.append(state["passes"])
        real_save(path, state)

    try:
        row, = [s for s in r.store.list("fit") if s.dir_name == damaged.name]
        assert row.complete is False and row.reason.startswith("unreadable manifest"), row
        assert [k.id for k in fit_state.kept_fits(r.store) if k.path == damaged] == []
        with monkeypatch.context() as patch:
            patch.setattr(fit_state, "save_state", spy)
            fourth = _train(r, 3, 5, name="fit_fourth")
    finally:
        shutil.rmtree(damaged)
    training = fourth.manifest.body["training"]
    assert saves == list(range(1, training["epochs_trained"] + 1)), saves
    assert training["fit_id"] is not None
    assert _said(_log(fourth), "warning", "[fit] ") == []
    assert _said(_log(fourth), "info", "[fit] 1 kept") == [
        f"[fit] 1 kept training(s) match these settings and can be resumed: {first.id} (kept when stopped, "
        f"after pass 1); this run starts a new one."]
    assert [s.id for s in _new_fits(r, fits_left)] == [first.id]

    def unlisted(store, identity):
        raise OSError("the kept trainings' folder could not be read")

    monkeypatch.setattr(fit_state, "matching_kept_fits", unlisted)
    fifth = _train(r, 3, 5, name="fit_fifth")
    assert _said(_log(fifth), "warning", "[fit] ") == [
        "[fit] the kept trainings could not be listed to name the ones that match these settings (OSError: "
        "the kept trainings' folder could not be read); this run starts a new one."]
    assert fifth.manifest.body["training"]["fit_id"] is not None
    assert [s.id for s in _new_fits(r, fits_left)] == [first.id]


def test_the_listing_of_kept_trainings_says_which_can_be_resumed_and_why_not(
        tiny_run, fits_left, monkeypatch, tmp_path):
    """The listing reads each kept training's record, newest first, with whether it can be resumed and,
    when it cannot, why: the code that builds the network has changed since it was kept, or the
    simulation set it was trained on is gone. A kept training matches the identity it was kept with, and
    one that cannot be resumed matches nothing. The listing never opens a kept training's saved passes:
    on Windows a reader holding one would make the running training's next save fail to rename it."""
    r = tiny_run
    with monkeypatch.context() as patch:
        _stop_after(patch, 1, KeyboardInterrupt())
        for width in (4, 3):
            with pytest.raises(KeyboardInterrupt):
                _train(r, 3, width, name="fit_listed")
    made = [s.id for s in _new_fits(r, fits_left)]

    def listed():
        return [k for k in fit_state.kept_fits(r.store) if k.id in made]

    kept = listed()
    assert [k.id for k in kept] == made and len(made) == 2
    assert [k.identity["simulation"]["run_size"] for k in kept] == [3, 4], "not newest first"
    assert [(k.can_resume, k.why_not) for k in kept] == [(True, None)] * 2
    assert [(k.state, k.passes, k.passes_since_best) for k in kept] == [("kept when stopped", 1, 0)] * 2
    assert all(k.best_validation_loss is not None and k.parents["prior"] == r.prior.id for k in kept)
    identities = [fit_state.read_header(k.path)["identity"] for k in kept]
    for k, identity in zip(kept, identities):
        matched = fit_state.matching_kept_fits(r.store, identity)
        assert [m.id for m in matched] == [k.id]

    opened, real_load = [], torch.load

    def load(f, *args, **kwargs):
        if isinstance(f, (str, os.PathLike)):
            opened.append(Path(f))
        return real_load(f, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", load)
        r.store.list("fit")
        fit_state.kept_fits(r.store)
        fit_state.matching_kept_fits(r.store, identities[0])
    fits = r.store.kind_dir("fit")
    states = [p for p in opened if fits in p.parents and p.name in (STATE, STATE_PREV)]
    assert states == [], f"the listing opened a kept training's saved pass: {states}"

    real_code = fit_state.loaded_code

    def edited(names=FIT_CODE):
        return {**real_code(names), "core.orchestrator.build_embedding_net": "0" * 16}

    with monkeypatch.context() as patch:
        patch.setattr(fit_state, "loaded_code", edited)
        assert [(k.can_resume, k.why_not) for k in listed()] == [
            (False, "the code that builds or trains the network has changed since this training was kept "
                    "(core.orchestrator.build_embedding_net)")] * 2
        assert [m.id for m in fit_state.matching_kept_fits(r.store, identities[0]) if m.id in made] == [], \
            "a kept training that cannot be resumed was named as one that matches"

    sim = r.store.kind_dir("simulation") / kept[1].parents["simulation"]
    aside = tmp_path / "aside"
    aside.mkdir()
    moved = [name for name in (STATE, STATE_PREV) if (sim / name).exists()]
    for name in moved:
        shutil.move(str(sim / name), str(aside / name))
    try:
        assert [(k.can_resume, k.why_not) for k in listed()] == [
            (True, None), (False, f"its simulation set {sim.name} is not in the store any more")]
    finally:
        for name in moved:
            shutil.move(str(aside / name), str(sim / name))


def test_a_narrowing_rounds_kept_fit_names_its_observation_and_parent_posterior(tiny_run, fits_left, monkeypatch):
    """A narrowing round kept when it stops records its region and names the observation it was drawn
    around and the posterior it was drawn from, so neither can be deleted under it. A round asked not to
    save, through the round's own entry, saves nothing: the choice reaches the training stage."""
    from core import orchestrator
    from core.artifacts.manifest import region_to_json
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="fit_round_obs")
    region = orchestrator.build_truncation_region(r.posterior, obs, n_directions=1, level=0.99)
    with monkeypatch.context() as patch:
        _stop_after(patch, 1, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            _train(r, 2, 8, truncation=region, observation=obs, parent_posterior=r.posterior, name="fit_round")
    row, = _new_fits(r, fits_left)
    m = r.store.get("fit", row.id)
    assert m.parents == {"prior": r.prior.id, "simulation": m.parents["simulation"],
                         "parent_posterior": r.posterior.id, "observation": obs.id}
    assert m.body["region"] == json.loads(json.dumps(region_to_json(region)))
    with pytest.raises(Refusal) as refused:
        r.store.delete("observation", obs.id)
    assert refused.value.field == "artifact" and row.id in str(refused.value)

    unsaved = orchestrator.tsnpe_round(r.cfg, r.posterior, r.prior, obs, n_directions=1, num_runs=2,
                                       save_fit=False, name="fit_round_unsaved", fig_sink=r.sink,
                                       store=r.store, **_TRAINING)
    assert _added_folders(r, fits_left) == [row.dir_name]
    assert _said_but_epochs(_log(unsaved), "info", "[fit] ") == [
        "[fit] saving is switched off: this training is not saved as it goes and cannot be resumed."]


# ── resuming a kept training ─────────────────────────────────────────────────────────────────────────
# A training kept when it stopped is resumed by naming it: the training stage takes every setting it is
# not given from the kept training's own record, checks before anything is read that nothing the
# training was kept with has changed, reads its simulation set back, puts the kept pass back onto a fresh
# trainer and goes on saving into the same record. Each test keeps its trainings on a simulation set of
# its own, four or five batches wide (the saving tests above use two and three).


def _keep(r, batches: int, width: int, *, at: int = 1, **given):
    """A tiny training stopped by a Ctrl-C right after pass ``at`` is saved: the store's row of the kept
    training it leaves."""
    before = {s.id for s in r.store.list("fit")}
    with pytest.MonkeyPatch.context() as patch:
        _stop_after(patch, at, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            _train(r, batches, width, **given)
    row, = [s for s in r.store.list("fit") if s.id not in before]
    return row


def _resume(r, ref: str, prior=None, **given):
    """The training stage resuming the kept training ``ref``, given nothing else but ``given``."""
    from core import orchestrator
    return orchestrator.build_posterior(r.cfg, r.prior if prior is None else prior, None, True, resume_fit=ref,
                                        fig_sink=r.sink, store=r.store, **given)


def _bytes(path) -> dict:
    """Every file of the folder ``path``, by name, with its bytes."""
    return {p.name: p.read_bytes() for p in sorted(Path(path).iterdir()) if p.is_file()}


def _row(r, fit_id: str):
    return next(s for s in r.store.list("fit") if s.id == fit_id)


def test_a_kept_training_resumed_writes_the_same_posterior_as_an_uninterrupted_one(tiny_run, fits_left):
    """A training stopped after its second pass and resumed in a new call -- another random stream, the
    rows read back again, the network built again -- writes the very posterior the same training writes
    uninterrupted: the same losses pass by pass, the same best loss and number of passes, the same
    weights. The resumed posterior has the stopped training's name and note, says it was resumed over two
    processes, holds the stopped process's records before its own, and the kept training is gone. The
    two trainings start from one seed: the embedding network is drawn before the simulation set read
    back puts the random stream where its last batch left it."""
    r = tiny_run
    _train(r, 4, 8, save_fit=False, name="fit_same_rows")              # makes the simulation set, whole
    torch.manual_seed(2026)
    whole = _train(r, 4, 8, name="fit_same_whole")
    torch.manual_seed(2026)
    row = _keep(r, 4, 8, at=2, name="fit_same_split", note="split in two")
    torch.manual_seed(12345)                    # the resumed network is built, then the kept one put back
    resumed = _resume(r, row.id)
    assert (resumed.name, resumed.manifest.note) == ("fit_same_split", "split in two")
    a, b = (np.load(p.path / "loss.npz") for p in (whole, resumed))
    assert np.array_equal(a["training_loss"], b["training_loss"])
    assert np.array_equal(a["validation_loss"], b["validation_loss"])
    tw, tr = whole.manifest.body["training"], resumed.manifest.body["training"]
    assert (tr["epochs_trained"], tr["best_validation_loss"]) == (tw["epochs_trained"], tw["best_validation_loss"])
    assert tr["epochs_trained"] > 2
    assert _unequal(whole.latent.posterior_estimator.state_dict(),
                    resumed.latent.posterior_estimator.state_dict()) == []
    assert (tr["resumed_fit"], tr["fit_processes"], tr["fit_id"]) == (True, 2, row.id)
    assert tr["fit_wall_seconds"] > 0
    text = _log(resumed)
    assert text.index("[fit] saved after pass 1.") < text.index(f"[fit] resuming the kept training {row.id}")
    assert _new_fits(r, fits_left) == []


def test_a_resume_runs_on_the_kept_trainings_own_settings_name_and_note(tiny_run, fits_left, monkeypatch):
    """A resume given nothing but the kept training trains with the kept training's own budget, flow,
    patience and ceiling, and writes the posterior under the name and note it was started with; a name
    given with the resume is the posterior's instead. A kept name another posterior has taken meanwhile
    is refused before a single row is read, not after the whole fit."""
    from core import orchestrator
    r = tiny_run
    first = _keep(r, 4, 7, name="fit_own", note="kept note")
    second = _keep(r, 4, 7, name="fit_own", note="kept note")
    seen, real = [], orchestrator.pipeline.train_nn

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", spy)
    own = _resume(r, first.id)
    assert [(k["hidden_features"], k["num_transforms"], k["stop_after_epochs"], k["max_num_epochs"])
            for k in seen] == [(8, 1, 3, 8)]
    assert (own.manifest.config["num_runs"], own.manifest.config["run_size"]) == (4, 7)
    assert (own.name, own.manifest.note) == ("fit_own", "kept note")
    generated = []
    with monkeypatch.context() as patch:
        patch.setattr(orchestrator.pipeline, "gen_training_data", lambda *a, **k: generated.append(1))
        with pytest.raises(Refusal) as refused:
            _resume(r, second.id)                       # its kept name, fit_own, is the posterior's now
    assert refused.value.field == "name" and generated == [], refused.value
    assert _row(r, second.id).fit_state == "kept when stopped" and not running(second.path)
    renamed = _resume(r, second.id, name="fit_renamed")
    assert (renamed.name, renamed.manifest.note) == ("fit_renamed", "kept note")
    assert _new_fits(r, fits_left) == []


@pytest.fixture(scope="module")
def kept_once(tiny_run):
    """One kept training the identity cases below share, stopped after its first pass; deleted when the
    module ends."""
    row = _keep(tiny_run, 4, 6, name="fit_identity")
    yield row
    tiny_run.store.delete("fit", row.id)


def _clip(r, patch):
    real = train._sbi_defaults
    patch.setattr(train, "_sbi_defaults", lambda model: {**real(model), "clip_max_norm": 1.0})
    return {}


def _edited_code(r, patch):
    real = fit_state.loaded_code
    patch.setattr(fit_state, "loaded_code",
                  lambda names=FIT_CODE: {**real(names), "core.SBI.train.restore_trainer": "0" * 16})
    return {}


def _constant(r, patch):
    patch.setattr(config, "WINSOR_PCT", (0.002, 0.998))
    return {}


def _library(r, patch):
    real = fit_state.libraries
    patch.setattr(fit_state, "libraries", lambda: {**real(), "sbi": "0.27.0"})
    return {}


# Each part of the identity changed alone: what changes it, the setting that answers the refusal, and
# words the refusal says.
_IDENTITY_CASES = {
    "batches": (lambda r, patch: {"num_runs": 5}, "num_runs", "the number of training batches was 4 when it was kept"),
    "prior": (lambda r, patch: {"prior": r.other_prior()}, "prior", "the prior was"),
    "hidden_features": (lambda r, patch: {"hidden_features": 16}, "hidden_features",
                        "the number of hidden features was 8 when it was kept, and is 16 now"),
    "num_transforms": (lambda r, patch: {"num_transforms": 2}, "num_transforms",
                       "the number of flow transforms was 1 when it was kept, and is 2 now"),
    "learning_rate": (lambda r, patch: {"learning_rate": 0.01}, "learning_rate",
                      "the learning rate was 0.001 when it was kept, and is 0.01 now"),
    "stop_after_epochs": (lambda r, patch: {"stop_after_epochs": 4}, "stop_after_epochs",
                          "the early-stop patience, in epochs was 3 when it was kept, and is 4 now"),
    "max_num_epochs": (lambda r, patch: {"max_num_epochs": 9}, "max_num_epochs",
                       "the maximum number of epochs was 8 when it was kept, and is 9 now"),
    "gradient_clip": (_clip, "resume_fit", "the gradient clip was 5.0 when it was kept, and is 1.0 now"),
    "code": (_edited_code, "resume_fit", "the code that builds or trains the network has changed since this "
                                         "training was kept (core.SBI.train.restore_trainer)"),
    "constant": (_constant, "resume_fit", "the configuration constant WINSOR_PCT was"),
    "library": (_library, "resume_fit", "sbi was"),
}


@pytest.mark.parametrize("case", list(_IDENTITY_CASES))
def test_each_part_of_a_kept_trainings_identity_changed_alone_refuses_the_resume_and_names_itself(
        tiny_run, kept_once, fits_left, monkeypatch, case):
    """A kept training is resumed only while everything it was trained with is what it was: its rows
    (the batches, the prior), its settings (the ones a call gives and the ones sbi fills in), the code
    that builds or trains the network, the constants that code reads and the libraries. Each changed
    alone refuses the resume, naming what differs and the setting that answers it, before a single row
    is read, and leaves every file of the kept training as it was, its lock free."""
    from core import orchestrator
    r = tiny_run
    change, field, words = _IDENTITY_CASES[case]
    before = _bytes(kept_once.path)
    generated = []
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data", lambda *a, **k: generated.append(1))
    given = change(r, monkeypatch)
    with pytest.raises(Refusal) as refused:
        _resume(r, kept_once.id, **given)
    assert refused.value.field == field, refused.value
    assert words in str(refused.value), str(refused.value)
    assert generated == []
    assert _bytes(kept_once.path) == before
    assert not running(kept_once.path)


def test_a_resume_is_refused_for_a_load_with_saving_off_without_a_simulation_set_or_never_resuming(
        tiny_run, fits_left, monkeypatch):
    """A kept training is resumed only by a training call that saves it as it goes, on its kept
    simulation set: a load, saving switched off, checkpointing off and a policy that never resumes a set
    are each refused, naming what answers it, and so are an id no kept training has and a kept training
    whose simulation set is gone. None of them touches the kept training. A kept training whose header,
    or every saved pass of it, cannot be read when it is resumed -- a file damaged, or one another
    program holds for a moment -- is refused before a single row is read and left exactly as it was, its
    lock free, to be tried again; only a header saved in another format, which no later attempt can
    read, marks it as one that cannot be resumed."""
    from core import orchestrator
    r = tiny_run
    row = _keep(r, 4, 5, name="fit_refused")
    before = _bytes(row.path)
    for ref, train_new, given, field, words in (
            (r.posterior.id, False, {}, "resume_fit", "this call loads a posterior"),
            (None, True, {"save_fit": False}, "resume_fit", "saving cannot be switched off"),
            (None, True, {"checkpoint_every": 0}, "checkpoint_every", "checkpointing is off"),
            (None, True, {"resume": "never"}, "resume", "resume='never'")):
        with pytest.raises(Refusal) as refused:
            orchestrator.build_posterior(r.cfg, r.prior, ref, train_new, resume_fit=row.id, fig_sink=r.sink,
                                         store=r.store, **given)
        assert (refused.value.field, words in str(refused.value)) == (field, True), refused.value
    with pytest.raises(Refusal) as refused:
        _resume(r, "20000101T000000")
    assert refused.value.field == "resume_fit" and "no kept training with the id" in str(refused.value)
    assert _bytes(row.path) == before and not running(row.path)

    # A resume that passes the checks and fails before its fit begins (here the embedding network cannot
    # be built) leaves the kept training unlocked and as it was, to be resumed again in the same process.
    def unbuildable(*args, **kwargs):
        raise RuntimeError("the embedding network could not be built")

    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, "build_embedding_net", unbuildable)
        with pytest.raises(RuntimeError, match="could not be built"):
            _resume(r, row.id)
    assert not running(row.path), "a resume that failed before its fit kept the kept training locked"
    assert _bytes(row.path) == before

    # A kept training of its own, on the same simulation set, stopped after its second pass so that it
    # has both saved passes.
    unread = _keep(r, 4, 5, at=2, name="fit_unread")
    whole = {name: data for name, data in _bytes(unread.path).items()
             if name in (fit_state.HEADER, STATE, STATE_PREV)}
    assert set(whole) == {fit_state.HEADER, STATE, STATE_PREV}, whole.keys()
    generated = []
    with monkeypatch.context() as patch:
        patch.setattr(orchestrator.pipeline, "gen_training_data", lambda *a, **k: generated.append(1))
        for damaged, words in (((fit_state.HEADER,), "its header cannot be read"),
                               ((STATE, STATE_PREV), "no saved pass of it can be read")):
            for name, data in whole.items():
                (unread.path / name).write_bytes(data[:10] if name in damaged else data)
            files = _bytes(unread.path)
            with pytest.raises(Refusal) as refused:
                _resume(r, unread.id)
            said = str(refused.value)
            assert refused.value.field == "resume_fit" and words in said and "can be tried again" in said, said
            assert _bytes(unread.path) == files, f"the kept training was changed by a read that failed ({words})"
            assert _row(r, unread.id).fit_state == unread.fit_state == "kept when stopped"
            listed, = [k for k in fit_state.kept_fits(r.store) if k.id == unread.id]
            assert (listed.can_resume, listed.why_not) == (True, None)
            assert not running(unread.path)
        for name, data in whole.items():
            (unread.path / name).write_bytes(data)
        fit_state.write_header(unread.path, {**fit_state.read_header(unread.path), "format": 2})
        states = {name: (unread.path / name).read_bytes() for name in (STATE, STATE_PREV)}
        with pytest.raises(Refusal) as refused:
            _resume(r, unread.id)
        assert refused.value.field == "resume_fit" and "it was saved in format 2" in str(refused.value), refused.value
        assert _row(r, unread.id).fit_state == "cannot be resumed"
        assert "it was saved in format 2" in r.store.get("fit", unread.id).body["unresumable"]
        assert {name: (unread.path / name).read_bytes() for name in (STATE, STATE_PREV)} == states
        assert not running(unread.path)
    assert generated == []

    sim = r.store.get("fit", row.id).parents["simulation"]
    r.store.delete("simulation", sim, force=True)
    with pytest.raises(Refusal) as refused:
        _resume(r, row.id)
    assert refused.value.field == "resume_fit", refused.value
    assert f"its simulation set {sim} is not in the store any more" in str(refused.value)


def test_a_resume_is_refused_unless_its_settings_name_the_kept_simulation_set_whole_and_its_cadence(
        tiny_run, fits_left, monkeypatch, tmp_path):
    """A resume reads back the very simulation set the kept training was trained on, and records the
    cadence its rows were committed at, or it is refused before a row is read or simulated. Settings whose
    rows compare equal to the kept ones but name another set's folder -- a bound given as a numpy float of
    the same value, which the folder's name reads and the comparison does not -- are refused naming both
    sets: given consent to a new set, as the command that was stopped had it, the resume would otherwise
    compute a new rotation and simulate a whole new set. A set removed between the listing and the resume
    is refused as gone, and a cadence given that differs from the kept one is refused naming it. Each
    leaves every file of the kept training as it was, its lock free, and the store's sets as they were."""
    from core import orchestrator
    r = tiny_run
    row = _keep(r, 6, 4, name="fit_own_set")
    sim = r.store.get("fit", row.id).parents["simulation"]
    before = _bytes(row.path)
    sets = sorted(p.name for p in r.store.kind_dir("simulation").iterdir())
    read = []

    def gen_training_data(*args, **kwargs):
        read.append(Path(kwargs["checkpoint"]["dir"]).name)
        raise RuntimeError(f"the resume went on to read or simulate the simulation set {read[-1]}")

    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data", gen_training_data)

    def refused(cfg=r.cfg, **given) -> Refusal:
        with pytest.raises(Refusal) as caught:
            orchestrator.build_posterior(cfg, r.prior, None, True, resume_fit=row.id, fig_sink=r.sink,
                                         store=r.store, new_run=True, **_ROTATION, **given)
        assert _bytes(row.path) == before and not running(row.path)
        return caught.value

    numpy_bound = r.cfg.copy_for_run()
    key, (value, (lo, hi)) = next(iter(numpy_bound.params_dict.items()))
    numpy_bound.params_dict[key] = (value, (np.float64(lo), hi))
    said = refused(numpy_bound)
    assert said.field == "resume_fit", said
    assert "they name the simulation set " in str(said) and f", and it was trained on {sim}." in str(said), said
    assert f"they name the simulation set {sim}," not in str(said), said

    folder, real_kept_fit = r.store.kind_dir("simulation") / sim, fit_state.kept_fit

    def listed_then_removed(store, ref):
        kept = real_kept_fit(store, ref)
        os.replace(folder, tmp_path / sim)          # removed by another process just after the listing
        return kept

    with monkeypatch.context() as patch:
        patch.setattr(fit_state, "kept_fit", listed_then_removed)
        try:
            said = refused()
        finally:
            if (tmp_path / sim).exists():
                os.replace(tmp_path / sim, folder)
    assert said.field == "resume_fit" and f"its simulation set {sim} is not in the store any more" in str(said), said

    said = refused(checkpoint_every=2)
    assert said.field == "checkpoint_every", said
    assert "the checkpoint cadence, in batches (0 = off) was 1 when it was kept, and is 2 now" in str(said), said
    assert read == []
    assert sorted(p.name for p in r.store.kind_dir("simulation").iterdir()) == sets


_LATE = {"network": ((4, 4), "the network as built now differs in shape from the kept one"),
         "standardisers": ((4, 3), "the network's fitted standardisers differ from the kept ones"),
         "probe": ((5, 8), "was generated under a DIFFERENT parameter transform")}


@pytest.mark.parametrize("check", list(_LATE))
def test_a_late_check_that_fails_marks_the_kept_training_and_leaves_its_files_as_they_were(
        tiny_run, fits_left, monkeypatch, check):
    """A kept training whose rows, once read back, no longer give the fit it was kept with -- a network
    of another shape, other fitted standardisers, a simulation set generated under another parameter
    transform -- is refused then: it is marked as one that cannot be resumed, saying why, its lock is let
    go, its header, its two states and its log are as they were, its record changes in that one entry,
    and no posterior is written."""
    from core.SBI import training_checkpoint
    r = tiny_run
    (batches, width), words = _LATE[check]
    row = _keep(r, batches, width, name=f"fit_late_{check}")
    files = {name: data for name, data in _bytes(row.path).items() if name != "manifest.json"}
    record = json.loads((row.path / "manifest.json").read_text(encoding="utf-8"))
    if check == "network":
        monkeypatch.setattr(fit_state, "network_digest", lambda net: "0" * 16)
    elif check == "standardisers":
        real_winsorize = pipeline.winsorize_summary_block

        def shifted(data, n_sum):
            out = real_winsorize(data, n_sum).clone()
            out[:, 0] += 1.0
            return out

        monkeypatch.setattr(pipeline, "winsorize_summary_block", shifted)
    else:
        real_verify = training_checkpoint.verify
        monkeypatch.setattr(training_checkpoint, "verify", lambda path, identity, probe=None, **kw: real_verify(
            path, identity, None if probe is None else probe + 1.0, **kw))
    with pytest.raises(Refusal) as refused:
        _resume(r, row.id)
    assert refused.value.field == "resume_fit" and words in str(refused.value), refused.value
    assert {name: data for name, data in _bytes(row.path).items() if name != "manifest.json"} == files
    now = json.loads((row.path / "manifest.json").read_text(encoding="utf-8"))
    reason = now["body"]["unresumable"]
    assert words in reason, reason
    now["body"]["unresumable"] = None
    assert now == record, "the record changed in more than whether it can be resumed"
    assert not running(row.path)
    assert _row(r, row.id).fit_state == "cannot be resumed"
    listed, = [k for k in fit_state.kept_fits(r.store) if k.id == row.id]
    assert (listed.can_resume, listed.why_not) == (False, reason)
    assert f"fit_late_{check}" not in _posterior_names(r)


def test_every_refusal_of_a_simulation_set_that_no_longer_matches_is_of_the_one_class_a_resume_marks():
    """A resumed training marks its kept record as one that cannot be resumed only for a simulation set
    that no longer matches what it was trained on, which it tells from any other error by the error's
    class; so every refusal the check of a set makes is of that class, and a refusal of another class
    would leave the kept training to be tried again and again."""
    from core.SBI import training_checkpoint
    raised = [n for n in ast.walk(ast.parse(code_only(training_checkpoint.verify))) if isinstance(n, ast.Raise)]
    assert raised and all(isinstance(n.exc, ast.Call) and ast.unparse(n.exc.func) == "CheckpointMismatch"
                          for n in raised), [ast.unparse(n) for n in raised]


def test_a_kept_training_another_process_holds_is_refused_and_listed_as_running(tiny_run, fits_left, monkeypatch):
    """A kept training whose lock another handle holds -- the same training resumed in another window or
    shell -- is listed as running and as one that cannot be resumed, and a resume is refused before a
    single row is read; a resume that slips past the listing (it raced the lock) is refused when it
    takes the lock."""
    from core import orchestrator
    r = tiny_run
    row = _keep(r, 5, 7, name="fit_held")
    generated = []
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data", lambda *a, **k: generated.append(1))
    other = FitLock(row.path)
    assert other.acquire()
    try:
        assert _row(r, row.id).fit_state == "running"
        listed, = [k for k in fit_state.kept_fits(r.store) if k.id == row.id]
        assert (listed.can_resume, listed.why_not) == (False, "a running process is training it")
        with pytest.raises(Refusal) as refused:
            _resume(r, row.id)
        assert refused.value.field == "resume_fit" and "a running process is training it" in str(refused.value)
        monkeypatch.setattr(fit_state, "running", lambda path: False)       # the listing raced the lock
        with pytest.raises(Refusal) as refused:
            _resume(r, row.id)
        assert refused.value.field == "resume_fit", refused.value
        assert "is being trained by a running process" in str(refused.value)
        assert generated == []
    finally:
        other.release()
    assert not running(row.path)


def test_a_kept_training_whose_newest_state_is_unreadable_resumes_from_the_pass_before_it(
        tiny_run, fits_left, monkeypatch):
    """A kept training whose newest saved pass cannot be read (a save torn by a crash, a file damaged)
    resumes from the pass before it, and says so. A resume stopped while its simulation set is read back
    leaves every file as it was, the unreadable one too, and lets the lock go. The resume that goes on
    moves the unreadable file aside at its first save, so that the save's rotation keeps the pass before
    it, and writes the posterior."""
    from core.SBI import training_checkpoint
    r = tiny_run
    row = _keep(r, 5, 6, at=3, name="fit_torn")
    (row.path / STATE).write_bytes((row.path / STATE).read_bytes()[:10])
    before = _bytes(row.path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(training_checkpoint, "load_rows", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _resume(r, row.id)
    assert _bytes(row.path) == before and not running(row.path)
    assert _row(r, row.id).fit_state == "kept when stopped"

    seen, real = {}, fit_state.FitSession.save_pass

    def save_pass(self, take):
        real(self, take)
        if not seen:
            seen.update(newest=torch.load(self.path / STATE, weights_only=False)["passes"],
                        before_it=torch.load(self.path / STATE_PREV, weights_only=False)["passes"],
                        aside=(self.path / fit_state._STATE_UNREADABLE).read_bytes())

    monkeypatch.setattr(fit_state.FitSession, "save_pass", save_pass)
    with _SaidInsideASave() as inside:
        post = _resume(r, row.id)
    assert inside.said == [], f"said while a pass was being saved: {inside.said}"
    text = _log(post)
    assert _said(text, "warning", "[fit] the newest") == [
        f"[fit] the newest saved pass of the kept training {row.id} could not be read; it resumes from the "
        f"pass before it, pass 2."]
    assert seen == {"newest": 3, "before_it": 2, "aside": before[STATE]}
    resumed = text[text.index(f"[fit] resuming the kept training {row.id} after pass 2"):]
    assert _said(resumed, "info", "[fit] saved")[0] == "[fit] saved after pass 3."
    assert post.manifest.body["training"]["resumed_fit"] is True and _new_fits(r, fits_left) == []


def test_a_finished_training_resumed_writes_its_posterior_with_no_pass_trained(tiny_run, fits_left, monkeypatch):
    """A training whose fit ended but whose posterior could not be written is kept finished; resumed, it
    trains no pass -- the patience is spent, or the ceiling passed, where it stopped -- and writes the
    posterior at once, under its name, with the passes it was kept with and the time the earlier process
    spent fitting, and the kept training is gone. A resume whose posterior is refused in its turn (a
    posterior of that name written meanwhile) leaves it finished and resumable: only a check that finds
    the rows no longer give the kept fit marks a kept training as one that cannot be resumed."""
    r = tiny_run
    real_create = r.store.create

    def create(kind, *args, **kwargs):
        if kind == "posterior":
            raise OSError("No space left on device")
        return real_create(kind, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(r.store, "create", create)
        with pytest.raises(OSError, match="No space left on device"):
            _train(r, 5, 5, name="fit_finished")
    row, = _new_fits(r, fits_left)
    assert row.fit_state == "finished"
    kept_wall = r.store.get("fit", row.id).body["progress"]["wall_seconds"]

    def refused(kind, *args, **kwargs):
        if kind == "posterior":
            raise Refusal("a posterior named fit_finished was written meanwhile", field="name")
        return real_create(kind, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(r.store, "create", refused)
        with pytest.raises(Refusal) as caught:
            _resume(r, row.id)
    assert caught.value.field == "name", caught.value
    listed, = [k for k in fit_state.kept_fits(r.store) if k.id == row.id]
    assert (listed.state, listed.can_resume) == ("finished", True), (listed.state, listed.why_not)
    saved, real = [], fit_state.save_state

    def spy(path, state):
        saved.append(state["passes"])
        real(path, state)

    monkeypatch.setattr(fit_state, "save_state", spy)
    post = _resume(r, row.id)
    training = post.manifest.body["training"]
    assert saved == []
    assert post.name == "fit_finished"
    assert training["epochs_trained"] == row.fit_passes
    assert (training["resumed_fit"], training["fit_processes"], training["fit_id"]) == (True, 2, row.id)
    assert training["fit_wall_seconds"] == kept_wall, "the posterior lost the time the earlier process spent"
    assert _new_fits(r, fits_left) == []


def test_a_resumed_training_clears_its_old_stop_at_its_first_save_so_a_crash_reads_stopped_unexpectedly(
        tiny_run, fits_left, monkeypatch, tmp_path):
    """A training kept when it was stopped on purpose, once resumed and saved again, no longer says it
    was stopped on purpose: a crash of the resumed process -- which lets the lock go and writes nothing --
    leaves it reading "stopped unexpectedly". Stopped on purpose again, it says so, after its new pass.
    Its first save in the new process rotates the kept pass to the previous file, moves nothing aside
    (the kept pass was read whole), and keeps the earlier process's records in the record's log before
    this one's. Resumed again, it ends over three processes, with both earlier processes' records."""
    r = tiny_run
    row = _keep(r, 5, 4, name="fit_restopped")
    assert row.fit_state == "kept when stopped"
    seen, real = [], fit_state.FitSession.save_pass

    def save_pass(self, take):
        real(self, take)
        if self.resumed and not seen:
            on_disk = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
            copy = tmp_path / "after_a_crash"
            copy.mkdir()
            (copy / "manifest.json").write_text(json.dumps(on_disk), encoding="utf-8")
            previous = self.path / STATE_PREV
            earlier, resumed_here, _ = (self.path / "log.txt").read_text(encoding="utf-8").partition(
                f"[fit] resuming the kept training {self.id}")
            seen.append((self.passes, on_disk["body"]["stopped"], fit_state.state_word(copy, on_disk["body"]),
                         (self.path / fit_state._STATE_UNREADABLE).exists(),
                         torch.load(previous, weights_only=False)["passes"] if previous.exists() else None,
                         "[fit] saved after pass 1." in earlier and bool(resumed_here)))
            raise KeyboardInterrupt

    monkeypatch.setattr(fit_state.FitSession, "save_pass", save_pass)
    with pytest.raises(KeyboardInterrupt):
        _resume(r, row.id)
    assert seen == [(2, None, "stopped unexpectedly", False, 1, True)], seen
    after = _row(r, row.id)
    assert (after.fit_state, after.fit_passes) == ("kept when stopped", 2)
    assert not running(row.path)
    # Resumed once more, to its end: three processes trained it, and its posterior holds the records of
    # both earlier ones, in order, before its own.
    post = _resume(r, row.id)
    assert post.manifest.body["training"]["fit_processes"] == 3
    text = _log(post)
    resuming = f"[fit] resuming the kept training {row.id} after pass"
    marks = [text.find("[fit] saved after pass 1."), text.find(f"{resuming} 1"),
             text.find("[fit] saved after pass 2."), text.find(f"{resuming} 2")]
    assert -1 not in marks and marks == sorted(marks), marks
    assert _new_fits(r, fits_left) == []


def test_a_training_whose_later_saves_failed_resumes_from_its_last_good_pass(tiny_run, fits_left, monkeypatch):
    """A training whose saves failed from its second pass on (a full disk), then stopped, is kept at its
    last good pass with the failure on its record; resumed, it goes on from that pass and writes the
    posterior."""
    r = tiny_run
    real = fit_state.save_state

    def full_disk(path, state):
        if state["passes"] == 4:
            raise KeyboardInterrupt
        if state["passes"] >= 2:
            raise OSError("No space left on device")
        real(path, state)

    with monkeypatch.context() as patch:
        patch.setattr(fit_state, "save_state", full_disk)
        with pytest.raises(KeyboardInterrupt):
            _train(r, 5, 3, name="fit_full_disk")
    row, = _new_fits(r, fits_left)
    assert row.fit_passes == 1
    assert "OSError: No space left on device" in r.store.get("fit", row.id).body["save_error"]
    post = _resume(r, row.id)
    text = _log(post)
    resumed = text[text.index(f"[fit] resuming the kept training {row.id} after pass 1 ("):]
    assert _said(resumed, "info", "[fit] saved")[0] == "[fit] saved after pass 2."
    assert post.manifest.body["training"]["resumed_fit"] is True and _new_fits(r, fits_left) == []


def test_a_kept_narrowing_round_resumes_in_its_own_region(tiny_run, kept_once, fits_left, monkeypatch):
    """A narrowing round kept when it stopped resumes in the region it was trained in, read from its
    record and never drawn again, so a number of directions given with the resume is refused; the
    resumed round trains on the same simulation set, around the same observation and from the same
    parent, and its posterior records the kept region. A round asked to resume a kept training that is
    no round's, or around another observation, is refused before a single row is read."""
    from core import orchestrator
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="fit_resumed_round_obs")
    with monkeypatch.context() as patch:
        _stop_after(patch, 1, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            orchestrator.tsnpe_round(r.cfg, r.posterior, r.prior, obs, n_directions=1, num_runs=4, run_size_cap=8,
                                     name="fit_resumed_round", fig_sink=r.sink, store=r.store, **_TRAINING)
    row, = _new_fits(r, fits_left)
    kept = r.store.get("fit", row.id)
    with pytest.raises(Refusal) as refused:
        orchestrator.tsnpe_round(r.cfg, r.posterior, r.prior, obs, resume_fit=row.id, n_directions=1,
                                 fig_sink=r.sink, store=r.store)
    assert refused.value.field == "resume_fit" and "the region it was trained in" in str(refused.value)
    with pytest.raises(Refusal) as refused:
        orchestrator.tsnpe_round(r.cfg, r.posterior, r.prior, obs, resume_fit=kept_once.id, fig_sink=r.sink,
                                 store=r.store)
    assert refused.value.field == "resume_fit" and "is not a narrowing round's" in str(refused.value)
    other = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="fit_resumed_round_other_obs")
    generated = []
    with monkeypatch.context() as patch:
        patch.setattr(orchestrator.pipeline, "gen_training_data", lambda *a, **k: generated.append(1))
        with pytest.raises(Refusal) as refused:
            orchestrator.tsnpe_round(r.cfg, r.posterior, r.prior, other, resume_fit=row.id, fig_sink=r.sink,
                                     store=r.store)
    assert refused.value.field == "observation" and generated == [], refused.value
    assert not running(row.path)
    drawn, real_draw = [], orchestrator.build_truncation_region

    def draw(*args, **kwargs):
        drawn.append(1)
        return real_draw(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "build_truncation_region", draw)
    post = orchestrator.tsnpe_round(r.cfg, r.posterior, r.prior, obs, resume_fit=row.id, fig_sink=r.sink,
                                    store=r.store)
    assert drawn == []
    assert post.name == "fit_resumed_round"
    assert post.manifest.body["truncation"] == kept.body["region"]
    assert dict(post.manifest.parents) == dict(kept.parents)
    assert post.manifest.body["training"]["resumed_fit"] is True and _new_fits(r, fits_left) == []


# ── the record of each epoch ─────────────────────────────────────────────────────────────────────────
# Each epoch of a fit (a pass, in the lines about saving it) is said in one record of the run, once sbi has
# summarised it and, when the fit is saved as it goes, once it is saved: its number, its two losses, the
# seconds it took and the epochs since the best one. Under the window every record is where a requested
# stop is taken, which is why the record comes after the save and never before it.
_EPOCH_LINE = re.compile(r"\[fit\] epoch (\d+): training loss (\S+), validation loss (\S+), took (\S+) s, "
                         r"epochs since the best (\d+)$")


def _epoch_records(caplog) -> list:
    """The epoch records the training call said since the capture was last read, in order, the seconds
    each took left out (they differ from run to run); the capture is then cleared."""
    said = [re.sub(r", took \S+ s,", ", took _ s,", rec.getMessage()) for rec in caplog.records
            if rec.name == "core.SBI.train" and rec.getMessage().startswith("[fit] epoch ")]
    caplog.clear()
    return said


def test_a_window_stop_during_the_fit_keeps_every_pass_saved_before_it(tiny_run, fits_left, monkeypatch):
    """A Stop pressed in the window while a pass of the fit is being trained lands at the first record
    after that pass is saved: the training is kept with every pass finished before the stop landed, the
    one that was in progress included. Requested here from sbi's own summary of the second pass, before
    that pass is saved, as a click lands while the pass runs. Under the window every record is where a
    requested stop is taken, so any record said between the summary and the save -- the epoch's own above
    all -- would stop the training with its second pass unsaved. The record that carries the stop is never
    delivered: the window hears the first pass's saved line and its epoch line, never the second pass's
    saved line. The second epoch's own record, which would have followed that line, is said once the stop
    is taken, from the numbers the pass was saved with, just before the line that says the training is
    kept, so the run's text and the kept training's log hold every epoch it kept. Driven through the
    window's own channel, as the test of a Cancel pressed during a save is."""
    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler
    r = tiny_run
    token = CancelToken()
    token.arm()
    heard = []
    handler = _PumpLogHandler(types.SimpleNamespace(sink=lambda kind, payload: heard.append(payload)), token)
    summarize = SNPE._summarize_epoch

    def summarize_then_stop(self, *args, **kwargs):
        summarize(self, *args, **kwargs)
        if self.epoch + 1 == 2:
            token.requested.set()                       # Stop pressed during pass 2; nothing has taken it yet

    monkeypatch.setattr(SNPE, "_summarize_epoch", summarize_then_stop)
    core_logger = logging.getLogger("core")
    core_logger.addHandler(handler)
    try:
        with pytest.raises(WorkerCancelled) as stopped:
            _train(r, 6, 3, name="fit_stopped_in_its_fit")
    finally:
        core_logger.removeHandler(handler)
    assert token.fired is True
    row, = _new_fits(r, fits_left)
    assert (row.fit_state, row.fit_passes) == ("kept when stopped", 2)
    assert torch.load(row.path / STATE, weights_only=False)["passes"] == 2
    assert torch.load(row.path / STATE_PREV, weights_only=False)["passes"] == 1
    assert (stopped.value.kept_fit.id, stopped.value.kept_fit.passes) == (row.id, 2)
    assert stopped.value.kept_fit.saved_here is True
    said = [text for text, _level in heard]
    assert "[fit] saved after pass 1." in said and any(text.startswith("[fit] epoch 1: ") for text in said), said
    assert any(text.startswith(f"[fit] the training stopped after pass 2 and is kept as fit {row.id}")
               for text in said), said
    assert "[fit] saved after pass 2." not in said, said
    summary = torch.load(row.path / STATE, weights_only=False)["summary"]
    losses = summary["validation_loss"]
    second = train.fit_epoch_line(2, summary["training_loss"][-1], losses[-1], summary["epoch_durations_sec"][-1],
                                  train.epochs_since_best(losses))
    kept = next(i for i, text in enumerate(said) if text.startswith("[fit] the training stopped after pass 2"))
    assert [text for text in said if text.startswith("[fit] epoch 2: ")] == [second] and said[kept - 1] == second, said
    assert _epochs_in(_log(row)) == [1, 2]


def _epochs_in(text: str) -> list:
    """The numbers of the epoch records in the log ``text``, in order."""
    return [int(_EPOCH_LINE.match(record).group(1)) for record in _said(text, "info", "[fit] epoch ")]


def test_a_training_kept_at_a_stop_says_every_epoch_it_kept_once_and_so_does_its_resume(tiny_run, fits_left,
                                                                                       monkeypatch):
    """A stop taken right after a pass is saved -- at the save's own line, as the window's Stop is, or by a
    Ctrl-C arriving there -- comes before the record of that pass's epoch. The training stage, handling the
    stop, says that record once, from the numbers the pass was saved with, before the line that says the
    training is kept; so the kept training's log holds every epoch it kept, once, and the posterior a
    resume writes holds every epoch of the fit, once each, in order. A stop that comes once the epoch is
    said, while the next pass trains, adds nothing: the record is not said twice."""
    r = tiny_run
    with monkeypatch.context() as patch:
        _stop_after(patch, 2, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            _train(r, 7, 3, name="fit_epochs_kept")
    row, = _new_fits(r, fits_left)
    text = _log(row)
    assert _epochs_in(text) == [1, 2], text
    lines = text.splitlines()
    kept = next(i for i, line in enumerate(lines) if " info [fit] the training stopped after pass 2 " in line)
    assert " info [fit] epoch 2: " in lines[kept - 1], lines[kept - 2:kept + 1]
    post = _resume(r, row.id)
    epochs = post.manifest.body["training"]["epochs_trained"]
    assert epochs > 2 and _epochs_in(_log(post)) == list(range(1, epochs + 1)), _log(post)

    summarize = SNPE._summarize_epoch

    def stopped_while_training(self, *args, **kwargs):
        if self.epoch + 1 == 3:
            raise KeyboardInterrupt                     # the third pass trains; the second is saved and said
        summarize(self, *args, **kwargs)

    monkeypatch.setattr(SNPE, "_summarize_epoch", stopped_while_training)
    with pytest.raises(KeyboardInterrupt):
        _train(r, 7, 5, name="fit_epochs_kept_once")
    row, = _new_fits(r, fits_left)                      # the first was removed once its posterior was written
    assert row.fit_passes == 2 and _epochs_in(_log(row)) == [1, 2], _log(row)


def test_the_kept_epochs_record_is_said_whole_from_the_numbers_it_was_saved_with_and_once():
    """The record of the epoch a stop came before is said from the histories its pass was saved with, whole:
    the pass's number, the last training loss, the last validation loss and the last seconds, and the epochs
    since the best one counted over the whole validation history. A second stop handled in the same run
    finds it said and adds nothing. A pass kept with histories that cannot be read says nothing and raises
    nothing, since the stop being handled comes first. Read with no training, on a stand-in holding a pass
    number and the histories it was saved with, every number in them a different one."""
    histories = {"training_loss": [3.25, 2.5, 1.75], "validation_loss": [1.125, 0.5, 0.625],
                 "epoch_durations_sec": [9.0, 8.0, 7.3]}
    kept = types.SimpleNamespace(passes=3, _kept_epoch=(3, histories))
    with runs.capture_run() as run_log:
        fit_state.FitSession._say_kept_epoch(kept)
        said = [line.split(" ", 1)[1] for line in run_log.lines]
        fit_state.FitSession._say_kept_epoch(kept)
        again = [line.split(" ", 1)[1] for line in run_log.lines]
    assert said == ["info [fit] epoch 3: training loss 1.75, validation loss 0.625, took 7.3 s, "
                    "epochs since the best 1"], said
    assert again == said, again
    with runs.capture_run() as run_log:
        fit_state.FitSession._say_kept_epoch(types.SimpleNamespace(passes=3, _kept_epoch=(3, {})))
    assert run_log.lines == [], run_log.lines


def test_the_epoch_record_says_each_epochs_numbers_once(tiny_run, fits_left):
    """A training saved after every pass says each epoch in one record of its log, numbered from one, on
    the line right after the one that says its pass is saved: the training and the validation loss as the
    posterior's loss curves hold them, the seconds the epoch took, to one decimal, and the epochs since the
    best one, read from the validation losses so far. The fit has room to end by its patience, so that
    epochs come after the best one."""
    r = tiny_run
    post = _train(r, 6, 2, max_num_epochs=30, name="fit_epochs_said")
    lines = _log(post).splitlines()
    epochs = post.manifest.body["training"]["epochs_trained"]
    said = _said(_log(post), "info", "[fit] epoch ")
    assert len(said) == epochs, said
    curves = np.load(post.path / "loss.npz")
    counts = []
    for n, record in enumerate(said, start=1):
        got = _EPOCH_LINE.match(record)
        assert got is not None, record
        number, training, validation, took, since = got.groups()
        assert int(number) == n, record
        assert (training, validation) == (f"{curves['training_loss'][n - 1]:.6g}",
                                          f"{curves['validation_loss'][n - 1]:.6g}"), record
        assert re.fullmatch(r"\d+\.\d", took), record
        assert int(since) == train.epochs_since_best(curves["validation_loss"][:n]), record
        counts.append(int(since))
        saved = next(i for i, line in enumerate(lines) if line.endswith(f" info [fit] saved after pass {n}."))
        assert lines[saved + 1].endswith(f" info {record}"), (n, lines[saved:saved + 2])
    assert max(counts) > 0, f"every epoch improved on the ones before ({counts}): no count was read back"


def test_the_epoch_record_comes_without_a_saver_too(tiny_run, fits_left):
    """A training that is not saved as it goes (its simulation set not kept) says each epoch all the
    same, one record per epoch, numbered from one: the record waits on a save only when there is one."""
    r = tiny_run
    post = _train(r, 2, 8, checkpoint_every=0, name="fit_epochs_unsaved")
    said = _said(_log(post), "info", "[fit] epoch ")
    epochs = post.manifest.body["training"]["epochs_trained"]
    assert [int(_EPOCH_LINE.match(record).group(1)) for record in said] == list(range(1, epochs + 1)), said
    assert _added_folders(r, fits_left) == []


def test_the_epoch_record_changes_nothing_the_fit_computes(monkeypatch, tmp_path, caplog):
    """The record of each epoch only reads, and so does the wrapper that says it: from the same seed, a fit
    that says its epochs, saved as it goes or not, ends with the same weights, the same best weights and the
    same loss histories as the same fit run by sbi's own loop, its per-pass summary not wrapped at all, and
    leaves the random stream in the same place. So a fit never interrupted computes exactly what it computed
    before the record was said. Each record's seconds are the trainer's own for its epoch, to one decimal:
    every epoch of so small a fit takes well under a tenth of a second, so here each is given seconds of its
    own once sbi has timed it, half a second more for each epoch."""
    toy = _toy(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="core")
    summarize = SNPE._summarize_epoch

    def timed(self, *args, **kwargs):
        summarize(self, *args, **kwargs)
        self._summary["epoch_durations_sec"][-1] += 0.5 * (self.epoch + 1)

    def fit(saving: bool):
        torch.manual_seed(0)
        prepared = toy.prepare(hooks=saving)
        toy.fit(prepared, saver=_Saves() if saving else None)
        return prepared.infer, torch.get_rng_state()

    with monkeypatch.context() as patch:
        patch.setattr(train, "_wrap_summary", lambda infer, saver: False)      # sbi's own loop
        plain, plain_stream = fit(False)
    assert _epoch_records(caplog) == []
    for saving in (True, False):
        with monkeypatch.context() as patch:
            patch.setattr(SNPE, "_summarize_epoch", timed)
            said, said_stream = fit(saving)
        took = [_EPOCH_LINE.match(rec.getMessage()).group(4) for rec in caplog.records
                if rec.name == "core.SBI.train" and rec.getMessage().startswith("[fit] epoch ")]
        assert took == [f"{d:.1f}" for d in said._summary["epoch_durations_sec"]], saving
        assert len(_epoch_records(caplog)) == said._summary["epochs_trained"][-1], saving
        assert _history(said) == _history(plain), saving
        assert _unequal(said._neural_net.state_dict(), plain._neural_net.state_dict()) == [], saving
        assert _unequal(said._best_model_state_dict, plain._best_model_state_dict) == [], saving
        assert torch.equal(said_stream, plain_stream), f"the fit drew a random number (saving={saving})"


def test_epochs_since_the_best_is_read_from_the_history_and_holds_across_a_resume(monkeypatch, tmp_path, caplog):
    """The epochs since the best one are read from the validation losses so far, never counted by the
    record itself: back from the last entry to the first least loss that is a finite number. A tie is no
    improvement, as sbi counts its patience, and a loss that is not a finite number, minus infinity
    included, is never the best, as the kept training's record counts it; with no finite loss, every epoch
    so far. So a fit stopped at its best pass and resumed in a fresh trainer from what its saver kept says
    each epoch after the stop as the same fit says it uninterrupted (the seconds aside): each a count that a
    record keeping its own would have started again from nothing. The pass the stop came with is said by
    neither part: the stop is raised by its save, and its record would come after it."""
    nan, minus_infinity = float("nan"), float("-inf")
    assert train.epochs_since_best([3.0, 2.0, 2.5, 2.0, nan]) == 3
    assert train.epochs_since_best([5.0, minus_infinity, 6.0]) == 2
    assert train.epochs_since_best([]) == 0
    assert train.epochs_since_best([nan, nan]) == 2
    assert train.epochs_since_best([1.0]) == 0
    for history in ([3.0, 2.0, 2.5, 2.0, nan], [5.0, minus_infinity, 6.0]):
        kept = fit_state._progress({"summary": {"validation_loss": history}, "passes": len(history)}, 0.0, 1)
        assert kept["passes_since_best"] == train.epochs_since_best(history), history

    toy = _toy(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="core")
    torch.manual_seed(0)
    whole = toy.prepare()
    toy.fit(whole, saver=_Saves())
    uninterrupted = _epoch_records(caplog)
    losses = whole.infer._summary["validation_loss"]
    stop = losses.index(min(losses)) + 1                # the best pass: every pass after it counts on from it
    assert len(uninterrupted) == len(losses) > stop, (stop, uninterrupted)

    torch.manual_seed(0)
    before = _Saves(stop_after=stop)
    with pytest.raises(_Stopped):
        toy.fit(toy.prepare(), saver=before)
    assert _epoch_records(caplog) == uninterrupted[:stop - 1]
    snapshot, start = before.read()
    torch.manual_seed(999)
    toy.fit(toy.prepare(), saver=_Saves(), resume=(start, snapshot))
    resumed = _epoch_records(caplog)
    assert resumed == uninterrupted[stop:], (stop, resumed, uninterrupted)
    assert all(int(_EPOCH_LINE.match(record).group(5)) > 0 for record in resumed), resumed


def test_the_epoch_line_is_written_as_the_window_reads_it():
    """The record of an epoch, word for word: the window reads its numbers back from this text."""
    assert train.fit_epoch_line(12, -3.41274, -3.39812, 3541.2, 0) == (
        "[fit] epoch 12: training loss -3.41274, validation loss -3.39812, took 3541.2 s, epochs since the best 0")


def test_a_trainer_with_no_per_pass_summary_trains_with_one_warning(caplog):
    """A trainer with no per-pass summary to wrap (an sbi other than the one pinned) is not wrapped: a fit
    not saved as it goes trains all the same, with one warning that no record is said per epoch. Saving one
    as it goes needs that summary and is refused, as the trainer is built for it (the test of the trainer's
    hook points) and here again."""
    caplog.set_level(logging.INFO, logger="core")
    assert train._wrap_summary(types.SimpleNamespace(), None) is False
    said = [(rec.levelname, rec.getMessage()) for rec in caplog.records if rec.name == "core.SBI.train"]
    assert said == [("WARNING", "[fit] this sbi's trainer has no per-pass summary to read, so no line is "
                                "written per epoch.")], said
    with pytest.raises(RuntimeError, match="has no _summarize_epoch,"):
        train._wrap_summary(types.SimpleNamespace(), _Saves())
    caplog.clear()
    assert train._wrap_summary(types.SimpleNamespace(_summarize_epoch=lambda *a, **k: None), None) is True
    assert [rec for rec in caplog.records if rec.name == "core.SBI.train"] == []


def test_an_epoch_record_that_cannot_be_written_is_skipped_and_a_stop_still_passes(monkeypatch, tmp_path, caplog):
    """An epoch's record that cannot be made is skipped and the fit trains to its end, saved after every
    pass as before: the record is the fit's account, never a reason to stop it. The first record a training
    skips is said in one warning naming the error, and no later one is, however many epochs it skips; the
    next training says its own. The warning is said where the record would have been, once the error is
    handled, so that under the window a stop is taken there as it is at the record. A stop raised while the
    record is made (a Ctrl-C, the window's Stop: no ordinary error) is not caught, and stops the fit."""
    toy = _toy(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="core")
    warned = []

    class _Warned(logging.Handler):
        """Each warning of the training call, with whether a stop would be taken where it is said."""

        def emit(self, record):
            if record.name == "core.SBI.train" and record.levelno == logging.WARNING:
                warned.append((record.getMessage(), not runs.cancel_is_deferred()))

    def unwritable(*args, **kwargs):
        raise ValueError("the record cannot be made")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "fit_epoch_line", unwritable)
    said = ("[fit] an epoch's record could not be written (ValueError: the record cannot be made), so the "
            "epochs of this training are not all said.", True)
    handler = _Warned()
    logging.getLogger("core").addHandler(handler)
    try:
        for training in (1, 2):
            torch.manual_seed(0)
            prepared, saves = toy.prepare(), _Saves()
            toy.fit(prepared, saver=saves)
            epochs = prepared.infer._summary["epochs_trained"][-1]
            assert epochs > 1 and saves.passes() == list(range(1, epochs + 1)), (epochs, saves.passes())
            assert warned == [said] * training, warned
    finally:
        logging.getLogger("core").removeHandler(handler)
    assert _epoch_records(caplog) == []
    monkeypatch.setattr(train, "fit_epoch_line", interrupted)
    with pytest.raises(KeyboardInterrupt):
        toy.fit(toy.prepare(), saver=_Saves())


def test_the_stop_the_training_stage_raises_names_the_training_it_kept(tiny_run, fits_left, monkeypatch):
    """The stop or the failure the training stage raises carries the kept training it leaves, as
    ``kept_fit``: its id, the passes kept, whether it was resumed, whether this process saved a pass, and
    the processes that have trained it and saved a pass, as its record counts them -- what the window needs
    to ask whether to keep the training and to offer to resume it, which otherwise only the run's lines
    say. A stop and an error after the second pass's save carry it; a stop before the first save carries
    none, nothing being kept; a kept training resumed and stopped before it saved a pass in its new process
    carries the record as it was, with nothing saved into it here and its one process; the same training
    resumed again and stopped once a pass is saved in that process counts two; and a narrowing round's stop
    carries one, through the round's own entry."""
    from core import orchestrator
    from core.SBI import training_checkpoint
    r = tiny_run
    for exc, width in ((KeyboardInterrupt(), 6), (RuntimeError("the flow diverged"), 7)):
        with monkeypatch.context() as patch:
            _stop_after(patch, 2, exc)
            with pytest.raises(type(exc)) as stopped:
                _train(r, 6, width, name="fit_noted")
        note = stopped.value.kept_fit
        assert note == fit_state.KeptNote(note.id, 2, False, True, 1), note
        assert note.id in {row.id for row in _new_fits(r, fits_left)} and _row(r, note.id).fit_passes == 2

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(train, "run_fit", interrupted)
        with pytest.raises(KeyboardInterrupt) as stopped:
            _train(r, 6, 8, name="fit_noted_never_saved")
    assert not hasattr(stopped.value, "kept_fit"), stopped.value.kept_fit

    kept = _keep(r, 7, 4, at=2, name="fit_noted_resumed")
    with monkeypatch.context() as patch:
        patch.setattr(training_checkpoint, "load_rows", interrupted)
        with pytest.raises(KeyboardInterrupt) as stopped:
            _resume(r, kept.id)
    assert stopped.value.kept_fit == fit_state.KeptNote(kept.id, 2, True, False, 1), stopped.value.kept_fit
    assert r.store.get("fit", kept.id).body["progress"]["processes"] == 1
    with monkeypatch.context() as patch:
        _stop_after(patch, 3, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt) as stopped:
            _resume(r, kept.id)
    assert stopped.value.kept_fit == fit_state.KeptNote(kept.id, 3, True, True, 2), stopped.value.kept_fit
    assert r.store.get("fit", kept.id).body["progress"]["processes"] == 2

    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="fit_noted_round_obs")
    with monkeypatch.context() as patch:
        _stop_after(patch, 1, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt) as stopped:
            orchestrator.tsnpe_round(r.cfg, r.posterior, r.prior, obs, n_directions=1, num_runs=3, run_size_cap=8,
                                     name="fit_noted_round", fig_sink=r.sink, store=r.store, **_TRAINING)
    note = stopped.value.kept_fit
    assert (note.passes, note.resumed, note.saved_here, note.processes) == (1, False, True, 1), note
    assert r.store.get("fit", note.id).body["region"] is not None


def test_a_posterior_whose_name_was_taken_during_its_fit_is_saved_unnamed_when_asked_and_its_kept_training_goes(
        tiny_run, fits_left, monkeypatch):
    """Asked to, the training stage writes a posterior whose name another posterior took while it trained
    without that name, under its id, and its log says so; the kept training it was saved as goes, as it goes
    for every posterior written. Not asked, the write refuses the name, as before, and the finished training
    stays kept, to be resumed under another name (the test of a finished training resumed)."""
    from tests._fixtures import _posterior_artifact
    r = tiny_run
    real = fit_state.FitSession.save_pass

    def save_pass(self, take):
        real(self, take)
        if self.passes == 1:
            _posterior_artifact(r.store, r.cfg, name="fit_taken_meanwhile")      # another posterior takes the name

    monkeypatch.setattr(fit_state.FitSession, "save_pass", save_pass)
    post = _train(r, 6, 5, name="fit_taken_meanwhile", if_name_taken="unnamed")
    assert post.name == "" and r.store.get("posterior", "fit_taken_meanwhile").id != post.id
    assert (f" warning PreflightWarning: the name 'fit_taken_meanwhile' was taken by another posterior while this "
            f"run was going, so this posterior is saved without a name, under its id {post.id}; it can be named "
            f"afterwards.") in _log(post)
    assert post.manifest.body["training"]["fit_id"] is not None and _new_fits(r, fits_left) == []
