"""Saving a training as it goes and resuming it.

A training's fit (sbi's training of the neural posterior) is kept in a folder of its own: a header
written once, the state of the last finished pass and the one before it, and a lock file whose
operating-system lock says a process is training it now. What the fit was trained with is kept as its
identity: the rows and their split, the settings, a digest of the code that builds or trains the
network, the configuration's constants that code reads, the libraries, and the network's shape. A kept
fit is resumed only while every one of those is what it was.

Here: the two generations of the state and what a torn save leaves readable, the save's deferred-cancel
section, the lock seen from this process and from another, and each digest of the identity with the
first difference between two of them.
"""
import copy
import json
import linecache
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

import sbi
from core import config, runs
from core.Helpers import file_manager
from core.SBI import fit_state
from core.SBI.fit_state import (FIT_CODE, FIT_CONSTANTS, SBI_TRAINER_FILES, STATE, STATE_PREV, Difference,
                                FitLock, code_digests, constants, first_difference, indices_digest,
                                libraries, load_state, network_digest, running, save_state,
                                trainer_files_digest)


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


def test_the_named_code_is_the_code_that_builds_or_trains_the_network():
    """The three lists name what a kept fit's identity reads; one changed without this test is a list
    nobody checked against the code that builds or trains the network."""
    assert FIT_CODE == (
        "core.SBI.train.train_nn", "core.SBI.train._capped_zscore_check", "core.SBI.embedded_network._probit",
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
    assert indices_digest(split) == indices_digest(split.to(torch.int32).to(torch.int64))
    assert len(indices_digest(split)) == 16
