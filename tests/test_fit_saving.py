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
records are the ones sbi uses.
"""
import ast
import copy
import importlib
import inspect
import io
import json
import linecache
import os
import queue
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import torch

import sbi
from sbi.inference import SNPE
from sbi.neural_nets.net_builders import flow
from core import config, runs
from core.Helpers import file_manager
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
    lines moves it, a comment included, and an edit of another function in the same file does not."""
    probe = tmp_path / "fit_code_probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        before = code_digests(("fit_code_probe.f",))
        assert set(before) == {"fit_code_probe.f"} and before["fit_code_probe.f"] != "unreadable", before

        probe.write_text(_PROBE.replace("return x * 2", "return 2 * x + 0"), encoding="utf-8")
        linecache.checkcache()
        assert code_digests(("fit_code_probe.f",)) == before, "an edit outside the function moved its digest"

        probe.write_text(_PROBE.replace("the first wording", "a second, longer wording"), encoding="utf-8")
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
    same order, as a simulation set read back does. ``prepare(hooks)`` builds the trainer and appends the
    rows; ``fit(prepared, saver, resume)`` runs sbi's passes; ``train_nn(**kwargs)`` is the whole
    training call; ``generated`` records each call of the generation."""
    from sbi.utils import BoxUniform

    monkeypatch.chdir(tmp_path)
    prior = BoxUniform(torch.zeros(2), torch.ones(2))
    draw = torch.Generator().manual_seed(1)
    theta = torch.rand(rows, 2, generator=draw)
    x = torch.cat([theta, theta.sum(1, keepdim=True)], 1) + 0.05 * torch.randn(rows, 3, generator=draw)
    generated = []

    def fake_gen(*args, **kwargs):
        generated.append(len(theta))
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
    longer resumes this way."""
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
    with pytest.raises(_Stopped):
        toy.fit(toy.prepare(), saver=before)
    assert before.passes() == list(range(1, stop + 1))
    snapshot, start = before.read()

    torch.manual_seed(999)
    resumed = toy.prepare()
    after = _Saves()
    toy.fit(resumed, saver=after, resume=(start, snapshot))
    got = resumed.infer
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
    """The save only reads: a fit saved after every pass computes what the same fit computes unsaved, the
    same losses and weights, and leaves the random stream where the unsaved one leaves it. The saver is
    handed every pass, numbered from one, in order."""
    toy = _toy(monkeypatch, tmp_path)
    torch.manual_seed(0)
    plain = toy.prepare(hooks=False)
    toy.fit(plain)
    stream_plain = torch.get_rng_state()

    torch.manual_seed(0)
    saved = toy.prepare()
    saves = _Saves()
    toy.fit(saved, saver=saves)
    stream_saved = torch.get_rng_state()

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
