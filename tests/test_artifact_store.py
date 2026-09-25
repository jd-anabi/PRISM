"""The artifact store (piece 1 of the 2026-09-10 hardening programme): manifest schema, provenance,
the store and its writer, the loaders' refusals, the simulation identity, and the stage contract."""
import json
import math
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch

from core import config
from core.artifacts import manifest as mf
from core.artifacts import provenance as prov
from core.refusals import Refusal

from tests._fixtures import (CODE_FILES, CODE_ROOTS, _FakeDP, _gmm_in_box, _nad_cfg,
                             _posterior_artifact, _prior_artifact, _set_path, assert_cfg_unchanged,
                             backdate_tree, snapshot_cfg)


def _close(title, fig):
    """A closing figure sink. The writer's own sink saves the PNG and forwards here, and closes the
    figure itself only when nothing is forwarded, so a no-op sink leaks every figure it is handed."""
    from matplotlib import pyplot as plt
    plt.close(fig)


def _recording(seen):
    """A closing sink that also records each title."""
    return lambda title, fig: (seen.append(title), _close(title, fig))


def test_manifest_json_round_trips_tensors_exactly():
    V = torch.linalg.qr(torch.randn(13, 13, dtype=torch.float64))[0]
    as_list = mf.tensor_to_json(V)
    assert isinstance(as_list, list) and isinstance(as_list[0][0], float)
    back = mf.json_to_tensor(json.loads(json.dumps(as_list)))
    assert torch.equal(back, V), "float64 must survive repr -> json -> float64 bit for bit"
    assert mf.tensor_digest(back) == mf.tensor_digest(V)
    assert len(mf.tensor_digest(V)) == 16


def _header(**over):
    d = dict(schema=mf.SCHEMA, kind="calibration", id="20260910T120000", name="cal", created="2026-09-10T12:00:00+00:00",
             note="", prism={"git_rev": "unknown", "git_dirty": None, "branch": "unknown"},
             env={}, inputs={"bounds": None, "cell": None, "units": None, "model": "X"},
             config={}, parents={"posterior": "20260910T110000", "prior": "20260910T100000"},
             fingerprints={}, payloads={}, figures=[], body={"results": {"n_cal": 10}})
    d.update(over)
    return d


def _post_body(**over):
    d = {"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
         "truncation": None, "training": {}}
    d.update(over)
    return d


def test_a_posterior_manifests_amortized_flag_must_agree_with_its_region():
    """Defect D3 at the schema: the flag and the region are two views of one fact, and different
    readers gate on different ones -- the load path on the flag, calibration and inference on the
    region. A manifest where they disagree cannot be written or read."""
    ok = _header(kind="posterior", body=_post_body())
    assert mf.validate(ok).body["amortized"] is True
    region = {"dims": [0], "lo": [-1.0], "hi": [1.0], "level": 0.99}
    for bad in (_post_body(truncation=region),                          # amortized beside a region
                _post_body(amortized=False)):                           # truncated with none
        with pytest.raises(mf.ManifestError, match="D3"):
            mf.validate(_header(kind="posterior", body=bad))
    assert mf.validate(_header(kind="posterior", body=_post_body(amortized=False, truncation=region))).id


def test_manifest_validation_refuses_missing_header_wrong_schema_bad_name_bad_id_and_nan():
    mf.validate(_header())                                                # the baseline is valid
    for bad, why in ((dict(schema=99), "schema"), (dict(kind="posteriors"), "kind"),
                     (dict(id="yesterday"), "id"), (dict(name="bad name!"), "name"),
                     (dict(body={"results": {"x": float("nan")}}), "finite"),
                     (dict(body={"wrong": 1}), "body"), (dict(extra=1), "unknown")):
        with pytest.raises(mf.ManifestError, match=why):
            mf.validate(_header(**bad))
    d = _header(); del d["parents"]
    with pytest.raises(mf.ManifestError, match="parents"):
        mf.validate(d)
    assert mf.validate(_header(id="20260910T120000-2")).id == "20260910T120000-2"
    assert mf.validate(_header(kind="simulation", id="0123456789ab",
                               body={k: None for k in mf.BODY_KEYS["simulation"]})).id == "0123456789ab"
    ok = _header(kind="diagnostic", body={"diagnostic": "sbc", "variant": "k4", "settings": {"repeats": 2},
                                          "results": {"n_valid": 8}})
    assert mf.validate(ok).body["diagnostic"] == "sbc"
    short = dict(ok["body"]); del short["variant"]
    with pytest.raises(mf.ManifestError, match="diagnostic body keys"):
        mf.validate(_header(kind="diagnostic", body=short))


def test_provenance_records_git_rev_and_dirty_and_unknown_outside_a_repo():
    here = prov.git_info(Path(__file__).resolve().parents[1])
    assert len(here["git_rev"]) == 40 and here["git_dirty"] in (True, False) and here["branch"]
    with tempfile.TemporaryDirectory() as tmp:
        out = prov.git_info(tmp)
    assert out == {"git_rev": "unknown", "git_dirty": None, "branch": "unknown"}


def test_env_records_versions_and_device():
    env = prov.env_info(config.cpu_device())
    for key in ("python", "torch", "sbi", "PySide6", "numpy", "os", "hostname", "device_type", "device_name", "dtype"):
        assert key in env and env[key], key
    assert env["device_type"] == "cpu"


def test_config_root_does_not_depend_on_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert config.RESOURCES_ROOT.name == "Resources" and (config.RESOURCES_ROOT / "Bounds").is_dir()
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "A"))
    assert config.artifacts_root() == tmp_path / "A"
    monkeypatch.delenv("PRISM_ARTIFACTS")
    assert config.artifacts_root() == config.RESOURCES_ROOT.parent / "Artifacts"


def test_sim_config_records_its_sources_and_the_feature_set_is_versioned():
    from core import cli, registry
    from core.config import BOUNDS_PATH, CELL_PATH, VALID_LABELS, VALID_MODELS
    from core.SBI.statistics import FEATURE_SET_VERSION
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / "nadrowski" / "master.txt"))
    assert cfg.sources["bounds"].endswith("master.txt") and cfg.sources["units"].endswith("units.txt")
    cli.load_and_validate_gt(cfg, str(CELL_PATH / "nadrowski" / "master_spont.txt"))
    assert cfg.sources["cell"].endswith("master_spont.txt")
    inputs = prov.inputs_from_cfg(cfg)
    assert inputs["model"] == "NADROWSKI" and len(inputs["bounds"]["sha256"]) == 64
    assert inputs["bounds"]["path"] == "Bounds/nadrowski/master.txt"       # relative to Resources/
    assert isinstance(FEATURE_SET_VERSION, int) and FEATURE_SET_VERSION >= 1
    block = mf.conditioning_block(cfg)
    assert block["feature_set_version"] == FEATURE_SET_VERSION and block["width"] == block["input_dim"] + block["forcing_dim"]
    c = mf.config_from_cfg(cfg)
    assert c["model"] == "NADROWSKI" and c["param_keys"][0] == list(cfg.params_dict)[0]
    assert math.isfinite(c["dt_exp"])


from datetime import datetime, timedelta, timezone

from core.artifacts import store as st


def _cal_body(n=10):
    return {"results": {"n_cal": n}}


def _bodies():
    """One valid body per WRITER kind -- the seven ``store.create`` accepts. A FRESH dict per call, so
    a test that hands one to the writer cannot leave a mutation behind for the next.

    The simulation kind is deliberately absent: its manifest has no writer at all (``store.create``
    refuses it outright) and ``write_simulation_manifest`` builds its body itself. That absence is
    why ``test_bodies_covers_every_writer_kind`` closes this dict against ``BODY_KEYS`` MINUS that
    one kind rather than against ``BODY_KEYS`` itself (piece 5, checklist 22).
    """
    return {
        "prior": {"gmm": {"n_components": 2, "param_keys": ["a"],
                          "box": {"nd_lows": [0.0], "nd_highs": [1.0], "log_mask": [False]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}},
        "posterior": {"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                      "truncation": None, "training": {}},
        "observation": {"mode": "chi", "conditioning": {"width": 50}, "x_obs_digest": "0" * 16,
                        "T_obs_cell": 1.0, "n_obs": 10, "forcing_vals": {}, "chi_obs_freqs": None,
                        "source": {"kind": "simulated"}},
        "calibration": _cal_body(), "inference": {"results": {"n_samples": 5}},
        "diagnostic": {"diagnostic": "sbc", "variant": None, "settings": {"repeats": 2},
                       "results": {"n_valid": 8}},
        # A finished single-cell measurement at its smallest: the five keys a run knows before it
        # starts, the three a finished single run fills, and the two that are null for a single run
        # (``points``) and for anything but a comparison (``compared``). ``complete`` is True
        # because _make exits its writer cleanly, and the writer is what sets that flag (Task 3).
        "fdt": {"study": "single", "settings": {"n_freqs": 2, "ensemble_M": 8}, "seed": 11,
                "grid": {"omega_0": 1.0, "n_freqs": 2}, "points": None, "offgrid": {"blanks": 0,
                "of": 2}, "notices": [], "compared": None, "complete": True,
                "results": {"ratio_at_resonance": 1.5}},
    }


def test_bodies_covers_every_writer_kind():
    """Checklist 22, a SILENT pin. ``_bodies`` feeds the per-kind round trip and the finished-per-kind
    test, and neither is closed against the schema -- so a kind added to ``BODY_KEYS`` and forgotten
    here is simply never round-tripped, and nothing says so. The simulation kind is subtracted rather
    than listed: it has no writer at all (``store.create`` refuses it), which is why ``_bodies``
    omits it on purpose.
    """
    assert set(_bodies()) == set(mf.BODY_KEYS) - {"simulation"}, \
        "a kind in BODY_KEYS with no body here is silently never written by any test"
    for kind, body in _bodies().items():
        assert set(body) == set(mf.BODY_KEYS[kind]), kind


def test_load_fdt_verifies_the_payload_hash_and_nothing_else(store):
    """Checklist 6. An fdt record is a MEASUREMENT of a cell, not a constraint on a later run, so
    there is no configuration it has to match and nothing is ever trained from it -- ``load_diagnostic``
    is the precedent (D5). What it DOES check is the one thing the manifest can be checked against:
    the payload's own sha256, so a data.h5 edited, truncated or deleted since the commit is refused
    rather than read as the numbers the record claims.

    A NULL recorded hash is not a mismatch. A progressive record hashes its payloads at the final
    commit only (spec §2.2), so an unfinished record lists data.h5 with a null hash and must still
    load -- reading what an interrupted run managed to write is the whole point of E2.
    """
    from core.artifacts import LoadedFdt
    # "measured" never wrote numbers: _make gives it results.json and nothing else.
    _make(store, "fdt", name="measured", body=_bodies()["fdt"])
    # "measured2" did. A payload is registered on the WRITER, so it is written inside the with.
    with store.create("fdt", None, name="measured2") as w2:
        w2.body = _bodies()["fdt"]
        w2.payload("data.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"0" * 64)

    loaded = store.load_fdt("measured2")
    assert isinstance(loaded, LoadedFdt) and loaded.kind == "fdt" and loaded.id == w2.id
    assert loaded.body["study"] == "single" and loaded.body["seed"] == 11
    assert loaded.data_path == w2.dir / "data.h5" and loaded.data_path.is_file()
    assert loaded.manifest.payloads["data.h5"] == prov.sha256_file(w2.dir / "data.h5")

    # a record that never wrote numbers: no data.h5, and the loader says so rather than pointing at
    # a path that is not there
    assert store.load_fdt("measured").data_path is None

    # a null recorded hash (an unfinished record) loads
    mpath = w2.dir / st.MANIFEST
    d = json.loads(mpath.read_text(encoding="utf-8"))
    d["payloads"]["data.h5"] = None
    mpath.write_text(json.dumps(d), encoding="utf-8")
    assert store.load_fdt("measured2").data_path is not None, "a null hash is 'not yet', not 'wrong'"

    # a payload edited since the commit IS refused
    d["payloads"]["data.h5"] = "0" * 64
    mpath.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(st.StoreError, match="data.h5") as edited:
        store.load_fdt("measured2")
    assert edited.value.field == "artifact"

    # ... and so is one DELETED since the commit (ruling F29). A recorded hash is a claim the loader
    # can check, and a file that is gone fails it; skipping it and answering data_path=None would
    # pass a finished record off as one that never wrote numbers. Only a NULL hash means "not yet".
    with store.create("fdt", None, name="measured3") as w3:
        w3.body = _bodies()["fdt"]
        w3.payload("data.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"1" * 64)
    (w3.dir / "data.h5").unlink()
    with pytest.raises(st.StoreError, match="data.h5") as gone:
        store.load_fdt("measured3")
    assert gone.value.field == "artifact"

    with pytest.raises(st.StoreError, match="no complete fdt") as nosuch:
        store.load_fdt("nosuch")
    assert nosuch.value.field == "artifact"


def _make(store, kind="calibration", name="", body=None, parents=None, note=""):
    with store.create(kind, None, name=name, note=note) as w:
        w.body = body if body is not None else _cal_body()
        w.parents = dict(parents or {})
        p = w.payload("results.json")
        p.write_text("{}", encoding="utf-8")
    return w


def test_create_list_get_round_trip_per_kind(store):
    bodies = _bodies()
    for kind, body in bodies.items():
        w = _make(store, kind, name=f"n_{kind}", body=body)
        rows = store.list(kind)
        assert [r.id for r in rows] == [w.id] and rows[0].complete and rows[0].name == f"n_{kind}"
        m = store.get(kind, w.id)
        assert m.to_dict() == store.get(kind, f"n_{kind}").to_dict() and m.body == body
        assert store.path(kind, w.id) == w.dir and m.payloads["results.json"] == prov.sha256_file(w.dir / "results.json")
        assert m.prism["git_rev"] != "" and m.env["python"]
    assert store.list("posterior")[0].width == 50 and store.list("posterior")[0].amortized is True
    assert store.list("diagnostic")[0].mode is None, \
        "the diagnostic body key is 'variant', not 'mode': Summary.mode is the OBSERVATION mode"


def test_writer_removes_the_directory_on_exception(store):
    class _Cancel(BaseException):           # the shape of gui.streams.WorkerCancelled
        pass
    with pytest.raises(_Cancel):
        with store.create("calibration", None) as w:
            w.payload("results.json").write_text("{}")
            raise _Cancel()
    assert not w.dir.exists() and store.list("calibration") == []


# The kinds ``store.create`` accepts that are NOT written progressively. Spelled out, and checked
# against _bodies below, so the pin that follows cannot quietly shrink as kinds are added.
ORDINARY_KINDS = ("prior", "posterior", "observation", "calibration", "inference", "diagnostic")


def test_every_ordinary_kind_still_removes_its_directory_on_a_failure(store):
    """Spec §11, risk row 2, and §8.2. The progressive mode touches ``ArtifactWriter``, which every
    kind uses, and piece 4's review caught a delete that destroyed a finished artifact -- so the six
    ordinary kinds' remove-on-exception is pinned HERE, before the mode exists, and this test is what
    says the mode changed nothing for them.

    Three exception shapes, because they arrive through the same ``__exit__``: a plain Exception, a
    BaseException (the shape of gui.streams.WorkerCancelled, which is what a Cancel click raises), and
    a ``Refusal`` -- the one class the progressive branch singles out. Each is raised both after a
    payload and with NOTHING written, which is exactly the condition that branch keys on (fix round 1,
    finding 4): an ordinary kind must remove its directory whichever way that test would come out, so
    neither "nothing was written" nor "it was a refusal" can be what decides.
    """
    class _Cancel(BaseException):
        pass

    assert set(ORDINARY_KINDS) == set(_bodies()) - {"fdt"}, \
        "a kind added to _bodies must be classified here: progressive, or removed on failure"

    for kind in ORDINARY_KINDS:
        for boom in (RuntimeError, _Cancel, Refusal):
            for wrote in (True, False):
                tag = f"{boom.__name__}_{'wrote' if wrote else 'nothing'}"
                with pytest.raises(boom):
                    with store.create(kind, None, name=f"{kind}_{tag}") as w:
                        w.body = _bodies()[kind]
                        if wrote:
                            w.payload("results.json").write_text("{}", encoding="utf-8")
                        raise boom("the run failed")
                assert not w.dir.exists(), f"{kind} kept a half-written directory after {tag}"
        assert store.list(kind) == [], f"{kind} left a row behind"


def test_loose_files_sees_what_no_listing_can_and_removes_one_by_name(store):
    """E10 and spec §6.3. ``_entries`` iterates DIRECTORIES only, so a file sitting inside a kind
    directory is invisible to every listing in both front ends -- and the owner has two of them, the
    PNGs an older FDT run dropped into ``Artifacts/fdt``. This is the only route by which either
    front end can see or clear one; nothing is removed on the owner's behalf.

    It can never reach a directory: the candidate is matched against ``loose_files``'s own entries,
    which are files. ``remove_incomplete`` owns the directories and refuses a non-directory, so the
    two calls cannot do each other's job -- the same safety property piece 4 gave ``delete`` and
    ``remove_incomplete`` (B7).
    """
    d = store.kind_dir("fdt")
    d.mkdir(parents=True, exist_ok=True)
    (d / "fdt_ratio_20260915_153042.png").write_bytes(b"\x89PNG stray")
    (d / "fdt_spont_20260915_153042.png").write_bytes(b"\x89PNG stray too")
    w = _make(store, "fdt", name="real", body=_bodies()["fdt"])   # writes results.json INSIDE its dir

    loose = store.loose_files("fdt")
    assert [f.name for f in loose] == ["fdt_ratio_20260915_153042.png",
                                       "fdt_spont_20260915_153042.png"], \
        "a valid record's payloads are never offered: they are inside its own directory"
    assert loose[0].size == len(b"\x89PNG stray") and loose[0].mtime > 0

    with pytest.raises(st.StoreError, match="was written") as recent:
        store.remove_loose("fdt", "fdt_ratio_20260915_153042.png")
    assert recent.value.field == "artifact", "the recency guard, shared with remove_incomplete"

    backdate_tree(d)
    store.remove_loose("fdt", "fdt_ratio_20260915_153042.png")
    assert [f.name for f in store.loose_files("fdt")] == ["fdt_spont_20260915_153042.png"]
    assert store.get("fdt", w.id).id == w.id, "a real record is untouched"

    # what it refuses
    for bad in ("real__" + w.id, "sub/x.png", "..", "", "nosuch.png"):
        with pytest.raises(st.StoreError):
            store.remove_loose("fdt", bad)
    assert (store.path("fdt", w.id)).is_dir(), "the record's directory is remove_incomplete's, not this call's"
    with pytest.raises(st.StoreError, match="unknown artifact kind"):
        store.loose_files("plot")
    assert store.loose_files("prior") == [], "a kind directory that does not exist has no loose files"


def test_a_legacy_directory_beside_the_kind_directories_is_seen_and_cleared(store):
    """E10's second half. The store never walks its own ROOT, so a ``crossval/`` written by an older
    build sits beside the kind directories and no command in either front end can see it. The list of
    such names is a CLOSED literal: a store root is not a place to guess at, and offering to remove
    whatever happens to be there is how a sweep destroys something nobody meant it to.

    It can never reach a kind directory: a name that is one is refused even if it were listed, which
    is the mirror of ``remove_loose`` never reaching a directory.
    """
    assert st.LEGACY_DIRS == ("crossval",), \
        "the one directory an older build wrote beside the kind directories (spec §2.1)"
    assert not set(st.LEGACY_DIRS) & set(st.KIND_DIRS.values()), \
        "a legacy name that is also a kind directory would make this call reach a real artifact"

    assert store.legacy_dirs() == [], "nothing on disk, nothing offered"
    old = store.root / "crossval"
    old.mkdir(parents=True)
    (old / "sweep_S.h5").write_bytes(b"an older build's numbers")
    assert store.legacy_dirs() == ["crossval"]

    with pytest.raises(st.StoreError, match="was written"):
        store.remove_legacy("crossval")
    backdate_tree(old)
    store.remove_legacy("crossval")
    assert not old.exists() and store.legacy_dirs() == []

    kind_dir = store.root / st.KIND_DIRS["prior"]
    kind_dir.mkdir(parents=True, exist_ok=True)
    backdate_tree(kind_dir)                  # old enough that only the name rule can protect it
    for bad in ("priors", "nosuch", "../etc", "", "."):
        with pytest.raises(st.StoreError) as exc:
            store.remove_legacy(bad)
        assert exc.value.field == "artifact", bad
    assert kind_dir.is_dir(), "a kind directory is never this call's to remove"


@pytest.mark.skipif(os.name != "nt", reason="a directory junction is a Windows construct")
def test_a_legacy_name_that_is_a_junction_is_neither_offered_nor_followed(store):
    """``remove_legacy`` promises that a junction called ``crossval`` is never followed -- but on
    Windows ``Path.is_symlink()`` is False for a junction, so that check alone would offer one in
    ``legacy_dirs`` and hand it to the removal. The junction here points at a KIND directory inside
    the root, the case the resolved-parent comparison cannot catch (its parent IS the root): without
    the explicit junction check, only ``shutil.rmtree``'s own internals stood between the tidy-up
    and a real kind directory.
    """
    import _winapi
    kind_dir = store.root / st.KIND_DIRS["prior"]
    kind_dir.mkdir(parents=True)
    (kind_dir / "keep.txt").write_bytes(b"a real kind directory's contents")
    link = store.root / "crossval"
    _winapi.CreateJunction(str(kind_dir), str(link))
    try:
        backdate_tree(kind_dir)              # old enough that only the link rule can protect it
        assert link.is_dir() and not link.is_symlink(), "the premise: is_symlink() misses a junction"
        assert store.legacy_dirs() == [], "a junction is never offered"
        with pytest.raises(st.StoreError, match="no legacy directory") as exc:
            store.remove_legacy("crossval")
        assert exc.value.field == "artifact"
        assert (kind_dir / "keep.txt").is_file()
    finally:
        os.rmdir(link)                       # the junction only; its target is untouched


def test_a_legacy_name_that_resolves_to_another_directory_is_neither_offered_nor_removed(
        store, monkeypatch):
    """The whole-piece review's N27 (L395). Windows addresses one directory by several names -- an
    8.3 short name among them -- so a ``crossval`` that is really an alias of a KIND directory would
    pass every string check, and its resolved parent IS the root, so the realpath-parent check passes
    too. ``realpath`` expands the alias, so the resolved BASENAME must be the legacy name itself.
    The alias cannot be made without an administrator, so ``os.path.realpath`` is made to resolve
    ``crossval`` to ``simulations``, which is what an alias would look like to this code."""
    old = store.root / "crossval"
    old.mkdir(parents=True)
    (old / "sweep_S.h5").write_bytes(b"what an alias of a kind directory would hold")
    backdate_tree(old)                       # old enough that only the name rule can protect it
    real = os.path.realpath

    def _alias(p, *a, **k):
        r = real(p, *a, **k)
        if os.path.basename(r).lower() == "crossval":
            return os.path.join(os.path.dirname(r), st.KIND_DIRS["simulation"])
        return r

    monkeypatch.setattr(os.path, "realpath", _alias)
    assert store.legacy_dirs() == [], "an alias of another directory was offered"
    with pytest.raises(st.StoreError, match="resolves to") as exc:
        store.remove_legacy("crossval")
    assert exc.value.field == "artifact"
    assert (old / "sweep_S.h5").is_file(), "the aliased directory was removed"


def test_a_progressive_record_has_a_valid_manifest_from_its_first_moment(store):
    """Spec §2.2 step 1 and E2. ``validate`` refuses a PARTIAL body key set, so the first manifest
    cannot carry only what is known -- it carries every key of BODY_KEYS["fdt"] with the unknown ones
    null. Five are known before the run starts (study, settings, seed, notices, complete) and five
    are not (grid, points, offgrid, compared, results), which is what makes an in-flight record
    readable in the browser while it runs.

    ``complete`` is the WRITER's field, not the stage's: it is False here because the writer wrote it
    so, and it is the writer that sets it True at the commit. A stage that forgot to touch it cannot
    therefore produce a record that claims to have finished.
    """
    w = store.create("fdt", None, name="inflight")
    assert w.progressive is True and w._wrote_anything is False
    assert not w.dir.exists(), "create() mints the id and the path and creates NOTHING (spec §1.2)"
    w.body = {"study": "single", "settings": {"n_freqs": 2}, "seed": 5, "notices": []}
    with w:
        assert w.dir.is_dir()
        m = mf.from_json_text((w.dir / st.MANIFEST).read_text(encoding="utf-8"))
        assert set(m.body) == set(mf.BODY_KEYS["fdt"])
        assert m.body["complete"] is False and m.body["study"] == "single"
        assert [k for k in mf.BODY_KEYS["fdt"] if m.body[k] is None] == \
            ["grid", "points", "offgrid", "compared", "results"]
        row = store.list("fdt")[0]
        assert row.complete and not row.finished, "listed while it runs, and honestly unfinished"
        w.body["grid"] = {"omega_0": 1.0}
        w.refresh()
        assert mf.from_json_text((w.dir / st.MANIFEST).read_text(encoding="utf-8")).body["grid"] \
            == {"omega_0": 1.0}, "refresh re-writes the manifest as the run proceeds"
        w.payload("data.h5").write_bytes(b"numbers")
        assert w._wrote_anything is True
        assert mf.from_json_text((w.dir / st.MANIFEST).read_text(encoding="utf-8")) is not None
    assert store.list("fdt")[0].finished, "the commit is what makes it finished"
    assert store.get("fdt", w.id).body["complete"] is True
    assert store.get("fdt", w.id).payloads["data.h5"] == prov.sha256_file(w.dir / "data.h5"), \
        "payloads are hashed at the COMMIT, and the commit is where the hash lands"


def test_a_cancel_between_the_first_manifest_and_the_first_payload_keeps_the_record(store):
    """Review Focus 1, and E2. The record exists, nothing has been written into it, and the cancel is
    a BaseException -- the shape ``core/gui/streams.py``'s WorkerCancelled has. The folder must
    survive: a run that took hours and was stopped is exactly what E2 keeps a folder for, and the
    spontaneous spectrum inside it is what diagnoses why it was stopped.

    A REFUSAL at the same point must do the opposite (spec §2.2 step 3) and the two arrive through
    the same ``__exit__``, which is why both are asserted here and not in two places.

    And the store half of spec §8.2's "the leftover sweep never offers it" (controller ruling F7): the
    kept record carries a manifest, so ``remove_incomplete`` -- the one call the sweep removes through
    -- refuses it even once it is old enough to be past the recency guard. The tool half is Task 30's.
    """
    class _Cancel(BaseException):
        pass

    w = store.create("fdt", None, name="cancelled")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(_Cancel):
        with w:
            raise _Cancel()
    assert w.dir.is_dir(), "E2: an interrupted record keeps its folder"
    row = [r for r in store.list("fdt") if r.id == w.id][0]
    assert row.complete and not row.finished, "a manifest, and an honest 'did not finish'"
    assert store.get("fdt", w.id).body["complete"] is False

    # the same point, a REFUSAL, nothing written: the directory goes
    w2 = store.create("fdt", None, name="refused")
    w2.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(Refusal):
        with w2:
            raise Refusal("the band reaches below what the spectrum resolves", field="freq_bounds")
    assert not w2.dir.exists(), \
        "a pre-spend refusal must not leave a permanent empty record: the sweep can never clear one"

    # a refusal AFTER something was written is an interrupted run like any other
    w3 = store.create("fdt", None, name="refused_late")
    w3.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(Refusal):
        with w3:
            w3.payload("data.h5").write_bytes(b"the spontaneous spectrum")
            raise Refusal("every grid frequency came back blank", field=None)
    assert w3.dir.is_dir() and (w3.dir / "data.h5").is_file(), \
        "what the run did measure is what the message tells the reader to look at"

    # F7: past the recency guard, and still never a leftover -- it carries a manifest
    backdate_tree(w.dir)
    with pytest.raises(st.StoreError, match="holds a valid fdt manifest"):
        store.remove_incomplete("fdt", w.dir.name)
    assert w.dir.is_dir()
    assert not [r for r in store.list("fdt") if not r.complete], \
        "an unfinished fdt record is never listed as a manifest-less leftover the sweep could offer"


def test_a_second_run_is_refused_by_name_while_an_unfinished_record_holds_it(store):
    """Review Focus 2. ``assert_name_free`` runs at ``create()``, before anything is spent, and a
    progressive record occupies its name from its first moment -- so a second run under the same name
    is refused at the click rather than colliding with a directory halfway through an hours-long
    measurement. The refusal carries ``field="name"``, which is what lets each front end name its own
    control.
    """
    w = store.create("fdt", None, name="repeat")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    w.__enter__()             # entered and NEVER exited: the record is live on disk, mid-run
    assert (w.dir / st.MANIFEST).is_file() and not store.list("fdt")[0].finished

    with pytest.raises(st.StoreError, match="already exists") as exc:
        store.create("fdt", None, name="repeat")
    assert exc.value.field == "name"


def test_a_failure_inside_a_progressive_record_writes_the_log_up_to_it(store):
    """Spec §2.2 step 3: the final refresh on the failure path is what puts the run's records on
    disk. Without it the one document that says WHY the run stopped would exist only for runs that
    did not stop -- which is the opposite of when it is needed. The log is written from
    ``runs.current_run_log()``, which is thread-local, so it is written here from inside a real run.
    """
    import logging
    from core import runs

    def _boom():
        w = store.create("fdt", None, name="logged")
        w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
        with w:
            logging.getLogger("core.test").info("the spontaneous campaign finished")
            raise RuntimeError("the driven campaign failed")

    entry = runs.public_entry(lambda: _boom())
    with pytest.raises(RuntimeError):
        entry()
    text, truncated = store.read_log("fdt", "logged")
    assert text is not None and "the spontaneous campaign finished" in text, text
    assert not truncated


def test_a_refused_record_that_will_not_delete_is_described_by_its_mode(store, monkeypatch):
    """Controller ruling F30. When the removal on the failure path itself fails (a held handle on
    Windows), the writer warns rather than let a PermissionError replace the exception that ended
    the run. For an ordinary kind the directory left behind has no manifest, and the warning says the
    store ignores it. A progressive record refused before it wrote anything DOES carry its first
    manifest, so the same sentence would be false: it names what the operator can do instead.
    """
    def _held(path, **kw):
        raise PermissionError("held open by a preview pane")
    monkeypatch.setattr(st, "_rmtree_retry", _held)

    w = store.create("fdt", None, name="held")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.warns(UserWarning, match="listed as unfinished") as rec:
        with pytest.raises(Refusal, match="below what the spectrum resolves"):
            with w:
                raise Refusal("the band reaches below what the spectrum resolves", field="freq_bounds")
    assert not [r for r in rec if "has no manifest" in str(r.message)], \
        "a progressive record keeps its first manifest; the warning must not say it has none"

    with pytest.warns(UserWarning, match="it has no manifest, so the store ignores it") as rec2:
        with pytest.raises(RuntimeError, match="the run failed"):
            with store.create("calibration", None, name="held_cal") as w2:
                w2.payload("results.json").write_text("{}", encoding="utf-8")
                raise RuntimeError("the run failed")
    # Both warnings name the caller's ``with`` line, not the store's own frames -- one removal helper
    # now serves __enter__ and __exit__ alike, and its stacklevel is counted for that extra frame.
    assert {Path(r.filename).name for r in [*rec, *rec2] if "could not remove" in str(r.message)} \
        == {Path(__file__).name}


def test_a_final_refresh_that_fails_never_replaces_the_exception_that_ended_the_run(store, monkeypatch):
    """Spec §2.2 step 3's final refresh runs INSIDE ``__exit__``, while the run's own exception is in
    flight. If that write fails too, the operator must still see why the RUN stopped -- a cancel, a
    refusal, a simulator error -- and not an OSError about a manifest; the record keeps the manifest
    it had, and a warning names the failed refresh. The log and the manifest are written in two
    separate attempts (fix round 1, finding 3), so each failing is its own warning and neither is the
    exception the caller sees.
    """
    w = store.create("fdt", None, name="stuck")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}

    def _disk_full(**kw):
        raise OSError("no space left on device")

    with pytest.warns(UserWarning) as rec:
        with pytest.raises(RuntimeError, match="the driven campaign failed"):
            with w:
                w.payload("data.h5").write_bytes(b"the spontaneous spectrum")
                monkeypatch.setattr(w, "_write", _disk_full)
                monkeypatch.setattr(w, "_write_log", _disk_full)
                raise RuntimeError("the driven campaign failed")
    said = [str(r.message) for r in rec]
    assert [s for s in said if s.startswith("could not write the log of the unfinished fdt record")], said
    assert [s for s in said if s.startswith("could not refresh the unfinished fdt record")], said
    assert w.dir.is_dir() and store.get("fdt", w.id).body["complete"] is False


def test_a_first_manifest_that_cannot_be_written_leaves_no_folder_behind(store):
    """Fix round 1, finding 1. ``__exit__`` never runs when ``__enter__`` raises, so a first manifest
    that fails would otherwise leave a manifest-less folder for a run that never started -- listed as
    "incomplete or interrupted", and refused by the sweep for RECENT_WRITE_SECONDS -- and every
    refused click that put a bad value in the first body would add one.

    Two shapes. A NaN in the settings, which ``validate`` refuses (a ManifestError, itself a
    Refusal). And a numpy float32, which ``_check_finite`` does not see as a float, so it passes
    validation and fails in ``json.dumps`` AFTER the log has been written: without the cleanup that
    folder would hold log.txt alone. Each is followed by a retry under the SAME name, which enters
    cleanly and commits.
    """
    import numpy as np
    from core import runs

    def _first(settings):
        w = store.create("fdt", None, name="first")
        w.body = {"study": "single", "settings": settings, "seed": 1, "notices": []}
        return w

    w_nan = _first({"f0": float("nan")})
    with pytest.raises(mf.ManifestError, match="non-finite"):
        with w_nan:
            pass
    assert not w_nan.dir.exists() and store.list("fdt") == []

    def _float32():
        # inside a real run, so a run log exists and log.txt IS written before the manifest fails
        w = _first({"f0": np.float32(0.05)})
        with pytest.raises(TypeError, match="float32"):
            with w:
                pass
        return w

    w32 = runs.public_entry(_float32)()
    assert not w32.dir.exists() and store.list("fdt") == []

    with _first({"f0": 0.05}) as ok:
        pass
    assert store.get("fdt", "first").id == ok.id and store.list("fdt")[0].finished


def test_a_committed_record_refuses_a_refresh(store):
    """Fix round 1, finding 2. A refresh writes ``complete: false`` and null payload hashes, so one
    arriving after the commit would turn a finished record back into an unfinished one -- listed as
    not finished, and never again hash-verified by ``load_fdt``, which skips a null hash as "not yet".
    Later tasks call ``refresh()`` from their stages; a late call must be refused, and must change
    nothing on disk.
    """
    w = store.create("fdt", None, name="done")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with w:
        w.payload("data.h5").write_bytes(b"numbers")
    before = (w.dir / st.MANIFEST).read_bytes()

    with pytest.raises(st.StoreError, match="already committed"):
        w.refresh()
    assert (w.dir / st.MANIFEST).read_bytes() == before
    assert store.list("fdt")[0].finished
    assert store.get("fdt", w.id).payloads["data.h5"] == prov.sha256_file(w.dir / "data.h5")


def test_an_unfinished_records_log_ends_with_what_stopped_it_and_nothing_says_it_twice(store, caplog):
    """The whole-piece review's N1 (L280, L569, L808). The keep branch wrote the run's log as it
    stood, and nothing put the exception in it: a record stopped by a refusal, a crash or an
    interrupt had a log.txt ending at the last progress line, so the one document that says why the
    run stopped did not say it. The branch now ends log.txt with one stamped line, ``error stopped:
    <Type>: <message>``, written STRAIGHT to the file -- never as a logging record, which the tool's
    console and the window's pane would show a second time beside the front end's own refusal line or
    box. Outside a run (no run log) it writes nothing, as ``_write_log`` always has."""
    import logging
    import re
    from core import runs

    def _run(name, exc):
        w = store.create("fdt", None, name=name)
        w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
        with w:
            logging.getLogger("core.test").info("the spontaneous campaign finished")
            w.figure_path("Spontaneous PSD").write_bytes(b"\x89PNG")
            raise exc

    for name, exc, said in (
            ("refused_late", Refusal("the band reaches below what the spectrum resolves",
                                     field="freq_bounds"),
             "Refusal: the band reaches below what the spectrum resolves"),
            ("interrupted", KeyboardInterrupt(), "KeyboardInterrupt")):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="core"):
            with pytest.raises(type(exc)):
                runs.public_entry(lambda: _run(name, exc))()
        text, _ = store.read_log("fdt", name)
        lines = text.splitlines()
        assert lines[-2].endswith("info the spontaneous campaign finished"), lines
        assert re.fullmatch(r"\d\d:\d\d:\d\d error stopped: " + re.escape(said), lines[-1]), lines
        assert not [r for r in caplog.records if "stopped" in r.getMessage()], \
            "the store logged the stop as a record: every front end would show it twice"

    w = store.create("fdt", None, name="no_run_log")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(RuntimeError):
        with w:
            w.payload("data.h5").write_bytes(b"numbers")
            raise RuntimeError("the driven campaign failed")
    assert w.dir.is_dir() and not (w.dir / st.LOG_FILE).exists(), "no run log, so no log.txt"


def test_an_unfinished_record_lists_only_the_figures_on_disk_and_a_commit_lists_every_one(store):
    """The whole-piece review's N2 (L573, L576, L589). ``figure_path`` records a figure when the
    PATH is handed out, and every manifest listed it -- so a run stopped between the hand-out and
    the save (Ctrl-C inside the Nadrowski passive check, a figure that failed to draw) left an
    unfinished record listing a picture that was never written. An unfinished manifest lists only
    the figures on disk. The COMMIT lists every one, as it always has, so a finished record's
    manifest -- any kind's -- is byte-for-byte what it was (the GPU gate's condition)."""
    w = store.create("fdt", None, name="phantom")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(KeyboardInterrupt):
        with w:
            w.figure_path("Spontaneous trajectory").write_bytes(b"\x89PNG")
            w.figure_path("Passive baseline ratio")          # handed out, never drawn
            raise KeyboardInterrupt
    assert store.get("fdt", w.id).figures == ["figures/spontaneous_trajectory.png"]

    done = store.create("fdt", None, name="committed")
    done.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with done:
        done.figure_path("Drawn").write_bytes(b"\x89PNG")
        done.figure_path("Handed out only")
    assert store.get("fdt", done.id).figures == ["figures/drawn.png", "figures/handed_out_only.png"], \
        "the commit's listing is unchanged: every figure handed out"


def test_a_record_removed_while_its_run_was_writing_it_is_not_rebuilt(store):
    """The whole-piece review's M3 (H1 + L278). A progressive record looks exactly the same whether
    its run was interrupted or is still being written from another window or a terminal, and nothing
    stops it being deleted meanwhile (spec §1.3 leaves the cross-process lock out). The live run then
    fails at its next write, and its ``__exit__`` took the keep branch -- whose manifest write goes
    through ``_atomic_write``, which creates the parent folder -- so the deleted record came BACK as a
    manifest-only husk, listing a figure that was gone, under the id just deleted.

    The keep branch now checks the folder first: gone, it keeps nothing and says so, once."""
    import shutil

    w = store.create("fdt", None, name="removed")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    w.__enter__()
    w.figure_path("Spontaneous trajectory").write_bytes(b"\x89PNG")
    w.refresh()
    shutil.rmtree(w.dir)                        # `artifacts rm fdt <id>` from another process

    with pytest.warns(UserWarning) as rec:
        assert w.__exit__(RuntimeError, RuntimeError("the run's next write failed"), None) is False
    said = [str(r.message) for r in rec]
    assert len(said) == 1 and "was removed while its run was writing it" in said[0], said
    assert not w.dir.exists(), "the keep branch rebuilt the folder it was told was deleted"
    assert store.list("fdt") == []


def test_a_refresh_outside_the_writers_with_block_is_refused_and_creates_nothing(store):
    """The whole-piece review's N3 (L279). ``refresh`` had no lifecycle check, and its manifest write
    creates the parent folder -- so a refresh BEFORE ``__enter__`` made the folder the later ``with``
    then refused as existing, leaving a permanent unfinished record, and one after a pre-spend
    refusal brought back the folder the refusal had just removed. Both are refused now, with nothing
    written. (After a COMMIT the refusal is the existing "already committed" one.)"""
    w = store.create("fdt", None, name="early")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(st.StoreError, match="not being written"):
        w.refresh()
    assert not w.dir.exists() and store.list("fdt") == []

    w2 = store.create("fdt", None, name="refused")
    w2.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(Refusal, match="below what the spectrum resolves"):
        with w2:
            raise Refusal("the band reaches below what the spectrum resolves", field="freq_bounds")
    assert not w2.dir.exists()
    with pytest.raises(st.StoreError, match="not being written"):
        w2.refresh()
    assert not w2.dir.exists() and store.list("fdt") == []


def test_the_log_up_to_a_failure_reaches_disk_even_when_the_last_manifest_cannot(store):
    """Fix round 1, finding 3; spec §2.2 step 3 ("the log up to the failure is on disk"). A stage that
    stores a value the manifest cannot hold (here a NaN in ``results``) and then fails leaves a final
    refresh that cannot validate. The log must not depend on it: the records since the last GOOD
    refresh are exactly the ones that say why the run stopped. The run's own exception still wins.
    """
    import logging
    from core import runs
    log = logging.getLogger("core.test")

    def _run():
        w = store.create("fdt", None, name="unvalidatable")
        w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
        with w:
            log.info("X: the spontaneous campaign finished")
            w.refresh()
            log.info("Y: the driven campaign reached its third probe")
            w.body["results"] = {"ratio_at_resonance": float("nan")}
            raise RuntimeError("the driven campaign failed")

    with pytest.warns(UserWarning, match="could not refresh the unfinished fdt record"):
        with pytest.raises(RuntimeError, match="the driven campaign failed"):
            runs.public_entry(_run)()
    text, _ = store.read_log("fdt", "unvalidatable")
    assert "X: the spontaneous campaign finished" in text, text
    assert "Y: the driven campaign reached its third probe" in text, \
        "a record emitted after the last good refresh must still reach log.txt"
    m = store.get("fdt", "unvalidatable")
    assert m.body["complete"] is False and m.body["results"] is None, "it keeps the manifest it had"


def test_an_input_file_that_vanished_during_the_run_is_recorded_unhashed_not_lost(store, tmp_path):
    """The commit runs at the END of a run that may have taken days. An input file moved or edited
    meanwhile must be recorded as unhashed (and warned about), never raise out of the writer -- losing
    a finished run to a renamed cell file would be the worst possible failure mode of provenance."""
    cfg = _nad_cfg()
    gone = tmp_path / "vanished_cell.txt"
    gone.write_text("not a real cell", encoding="utf-8")
    cfg.sources["cell"] = str(gone)
    gone.unlink()
    with pytest.warns(UserWarning, match="vanished_cell") as rec:
        with store.create("calibration", cfg) as w:
            w.body = _cal_body()
    # Attributed to the caller's ``with``, as it was before the commit was split into _write and its
    # helpers (piece 5, Task 3's amendment 3): a warning that names store.py points nowhere useful.
    assert [Path(r.filename).name for r in rec if "vanished_cell" in str(r.message)] == \
        [Path(__file__).name]
    m = store.get("calibration", w.id)
    assert m.inputs["cell"]["sha256"] is None and "vanished_cell" in m.inputs["cell"]["path"]
    assert len(m.inputs["bounds"]["sha256"]) == 64, "the files that ARE there are still hashed"
    with pytest.raises(FileNotFoundError):
        prov.inputs_from_cfg(cfg)                      # the default is still to refuse
    # provenance.file_ref's missing branch must show the path the SAME way the found branch does
    # (resolve, then relative_to) -- a raw str(p) would leave a relative/".."-laden path exactly as
    # given, which a PRESENT file at the same location would never record.
    rel = os.path.relpath(gone, Path.cwd())
    missing_path = prov.file_ref(rel, missing_ok=True)["path"]
    gone.write_text("back", encoding="utf-8")
    try:
        present_path = prov.file_ref(rel)["path"]
    finally:
        gone.unlink()
    assert missing_path == present_path and not missing_path.startswith(".")


def test_a_name_shaped_like_an_id_is_refused(store):
    """get()/path() resolve a ref as an id OR a name, so a name shaped like an id would shadow the
    artifact whose id it is -- which could then never be addressed at all."""
    cfg = _nad_cfg()
    for bad in ("20260910T120000", "20260910T120000-2", "abcdef012345"):
        assert mf.ID_RE.match(bad)
        with pytest.raises(st.StoreError, match="shaped like an artifact id"):
            store.create("prior", cfg, name=bad)
    w = _make(store, "prior", name="p", body={
        "gmm": {"n_components": 2, "param_keys": ["a"],
                "box": {"nd_lows": [0.0], "nd_highs": [1.0], "log_mask": [False]}},
        "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}})
    with pytest.raises(st.StoreError, match="shaped like an artifact id"):
        store.rename("prior", w.id, "20260910T120000")
    assert store.rename("prior", w.id, "p").name == "p", "a rename to its OWN name is not a collision"


def test_a_truncated_manifest_is_listed_incomplete_and_never_loadable(store):
    w = _make(store, name="ok")
    (w.dir / st.MANIFEST).write_bytes((w.dir / st.MANIFEST).read_bytes()[:40])
    rows = store.list("calibration")
    assert len(rows) == 1 and not rows[0].complete and "manifest" in rows[0].reason
    with pytest.raises(st.StoreError, match="no complete"):
        store.get("calibration", "ok")
    (w.dir / st.MANIFEST).unlink()
    assert not store.list("calibration")[0].complete


def test_same_second_ids_get_a_suffix(tmp_path):
    fixed = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    s = st.ArtifactStore(tmp_path, clock=lambda: fixed)
    ids = [_make(s).id for _ in range(3)]
    assert ids == ["20260910T120000", "20260910T120000-2", "20260910T120000-3"]
    assert {r.id for r in s.list("calibration")} == set(ids)


def test_two_writers_created_before_either_is_entered_get_different_ids(tmp_path):
    """Piece 5, spec §4.1: the sweep study's front end creates BOTH records before it dispatches, and
    create() puts nothing on disk -- so an id checked only against the disk would be minted twice in
    one second, and the second record would load as the first."""
    fixed = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)
    s = st.ArtifactStore(tmp_path, clock=lambda: fixed)
    a, b = s.create("fdt", None), s.create("fdt", None)
    assert a.id != b.id and a.dir != b.dir, (a.id, b.id)


def test_rename_keeps_the_id_and_dependents_resolve(store):
    p = _make(store, "posterior", name="post", body={"mode": "chi", "conditioning": {}, "transform": {},
                                                     "amortized": True, "truncation": None, "training": {}})
    c = _make(store, "calibration", name="cal", parents={"posterior": p.id})
    store.rename("posterior", "post", "post_v2")
    m = store.get("posterior", "post_v2")
    assert m.id == p.id and store.path("posterior", p.id).name == f"post_v2__{p.id}"
    assert store.get("calibration", c.id).parents["posterior"] == m.id
    assert store.dependents("posterior", p.id) == [("calibration", c.id, "cal")]
    with pytest.raises(st.StoreError, match="already exists"):
        _make(store, "posterior", name="post_v2", body=m.body)
    with pytest.raises(st.StoreError):
        store.rename("posterior", p.id, "bad name")
    store.set_note("posterior", p.id, "kept")
    assert store.get("posterior", p.id).note == "kept"


def test_set_note_refuses_an_unknown_ref_with_the_artifact_field(store):
    """Design §2.4. ``set_note`` was the store mutation whose refusal carried no field key, so the
    front ends had nothing to look up and the yellow box came up with no "where to fix it" line under
    it. It carries a key now, like every other pre-spend refusal (piece 3, V3).

    The key is ``"artifact"``, NOT ``"note"``: this sentence is about a MISSING ARTIFACT, so the note
    key would send the operator to the Note box when the fix is to select an artifact that exists.
    ``"note"`` stays for ``core.refusals.require_note``'s own refusals, where the note text is what is
    wrong. The window's entry for ``artifact`` is "Select an artifact in the list on the Artifacts
    screen." and the tool has no flag for it (``FLAG["artifact"] is None``), so the tool's line ends at
    the message with no trailing parenthetical -- both pinned in ``tests/test_refusals.py`` and
    ``tests/test_nav_and_gating.py`` by the steps above. The sentence itself is unchanged and still
    names no box and no flag."""
    c = _make(store, name="cal")
    assert store.set_note("calibration", c.id, "kept for the paper").note == "kept for the paper"
    assert store.get("calibration", c.id).note == "kept for the paper"
    with pytest.raises(st.StoreError) as e:
        store.set_note("calibration", "nope", "x")
    assert str(e.value) == "no complete calibration artifact named or id'd 'nope'"
    assert e.value.field == "artifact", "a store refusal without a field key cannot name a control"
    assert isinstance(e.value, Refusal)


def test_delete_refuses_naming_dependents_and_force_deletes(store):
    p = _make(store, "posterior", name="post", body={"mode": "chi", "conditioning": {}, "transform": {},
                                                     "amortized": True, "truncation": None, "training": {}})
    _make(store, "inference", name="inf", body={"results": {}}, parents={"posterior": p.id})
    with pytest.raises(st.StoreError, match="inference inf"):
        store.delete("posterior", p.id)
    assert store.get("posterior", p.id)
    store.delete("posterior", p.id, force=True)
    with pytest.raises(st.StoreError):
        store.get("posterior", p.id)


def test_unnamed_artifacts_group_and_age_out(tmp_path):
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    clock = {"now": t0}
    s = st.ArtifactStore(tmp_path, clock=lambda: clock["now"])
    old = _make(s)
    clock["now"] = t0 + timedelta(days=3)
    new = _make(s)
    named = _make(s, name="keep")
    assert {r.id for r in s.unnamed("calibration")} == {old.id, new.id}
    assert [r.id for r in s.unnamed("calibration", older_than=timedelta(days=1))] == [old.id]
    assert s.list("calibration")[0].label.startswith("(unnamed") or named.name == "keep"


def test_store_fig_sink_saves_png_and_forwards(store):
    from matplotlib import pyplot as plt
    seen = []
    with store.create("calibration", None) as w:
        w.body = _cal_body()
        sink = w.fig_sink(forward=lambda title, fig: seen.append((title, fig.number)))
        f1 = plt.figure(); sink("SBC ranks (CDF)", f1)
        f2 = plt.figure(); sink("SBC ranks (CDF)", f2)
        assert plt.fignum_exists(f1.number), "a forwarded figure is the GUI's to close"
        w.fig_sink()("TARP coverage", plt.figure())
    m = store.get("calibration", w.id)
    assert m.figures == ["figures/sbc_ranks_cdf.png", "figures/sbc_ranks_cdf-2.png", "figures/tarp_coverage.png"]
    assert all((w.dir / f).stat().st_size > 0 for f in m.figures) and len(seen) == 2
    plt.close("all")


def test_default_store_is_swappable_and_resolves_from_the_root(monkeypatch, tmp_path):
    session_default = st.default_store()                 # the conftest sandbox; put it back at the end
    try:
        monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "R"))
        st.set_default_store(None)
        assert st.default_store().root == tmp_path / "R"
        other = st.ArtifactStore(tmp_path / "O")
        with st.use_store(other):
            assert st.default_store() is other and st.resolve_store(None) is other
        assert st.default_store().root == tmp_path / "R"
    finally:
        st.set_default_store(session_default)


_IDENTITY_KEYS = {
    "format", "model", "prior_fingerprint", "mode", "param_keys", "nd_lows", "nd_highs", "rescale_lows",
    "rescale_highs", "log_params", "reparam_rotate", "run_size", "n_runs", "steady_idx", "dt_nd_min",
    "dt_exp", "t_min_exp", "t_max_exp", "t_scale_bounds", "n_grid", "spontaneous_only", "summary_flags",
    "feature_set_version", "chi_mode", "chi_layout", "chi_k_pad", "chi_elem_w", "chi_f0", "chi_freq_bounds",
    "chi_max_cycles", "device", "dtype", "truncation"}


def test_identity_carries_truncation_always_and_feature_set_version_rekeys(monkeypatch):
    from core import orchestrator
    from core.artifacts.identity import FORMAT, SimulationIdentity
    from core.SBI import statistics, truncate
    from core.SBI.training_checkpoint import identity_digest
    cfg = _nad_cfg(chi_mode=True)
    ident = SimulationIdentity.from_cfg(cfg, object(), 2048, 5000)
    d = ident.to_dict()
    assert d["format"] == FORMAT and d["truncation"] is None and d["prior_fingerprint"] is None
    assert d["feature_set_version"] == statistics.FEATURE_SET_VERSION and set(d) == _IDENTITY_KEYS
    assert identity_digest(orchestrator.training_identity(cfg, object(), 2048, 5000)) == ident.digest
    monkeypatch.setattr(statistics, "FEATURE_SET_VERSION", statistics.FEATURE_SET_VERSION + 1)
    assert SimulationIdentity.from_cfg(cfg, object(), 2048, 5000).digest != ident.digest, \
        "a changed feature definition must re-key the cache"
    monkeypatch.undo()
    Q = torch.linalg.qr(torch.randn(13, 13, dtype=torch.float64))[0]
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], level=0.999, n_latent=13, V=Q)
    t = SimulationIdentity.from_cfg(cfg, object(), 2048, 5000, truncation=region)
    assert t.truncated and t.to_dict()["truncation"] == region.identity_fields() and t.digest != ident.digest
    assert {k for k in _IDENTITY_KEYS if t.to_dict()[k] != d[k]} == {"truncation"}

    class _LoadedLike:                       # a LoadedPrior supplies its fingerprint outright
        fingerprint = "f" * 16
    assert SimulationIdentity.from_cfg(cfg, _LoadedLike(), 2048, 5000).to_dict()["prior_fingerprint"] == "f" * 16


def test_simulation_directories_live_under_the_store_without_a_prefix(store):
    from core.SBI import training_checkpoint as tc
    ident = {"format": "training-rows/2", "n_runs": 3, "prior_fingerprint": "a" * 16, "truncation": None}
    d = tc.resolve_dir(ident)
    assert d.parent == store.kind_dir("simulation") and d.name == tc.identity_digest(ident)
    assert not d.name.startswith("train_")
    other = dict(ident, n_runs=4)
    od = tc.resolve_dir(other)
    (od / "shards").mkdir(parents=True)
    torch.save({"format": tc.CHECKPOINT_FORMAT, "identity": other, "V": None, "probe": None}, od / "header.pt")
    torch.save({"batches_done": 2, "complete": False, "rng": None}, od / "state.pt")
    assert tc.near_miss_siblings(ident)[0]["field"] == "n_runs"
    assert "n_runs" in tc.describe_siblings(ident)
    assert tc.checkpoints_using_prior("a" * 16) == [(od.name, 2)]


def test_simulation_manifest_written_at_create_and_refreshed_on_save_and_complete(store):
    from core.SBI import training_checkpoint as tc
    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "run_size": 4,
             "prior_fingerprint": "b" * 16, "truncation": None}
    d = tc.resolve_dir(ident)
    tc.create(d, ident, schedule_t_scales=torch.ones(3), schedule_Ts=torch.ones(3), inits=torch.zeros(1, 2),
              V=None, probe=torch.zeros(7, 2, dtype=torch.float64), run_size=4, n_runs=3,
              parents={"prior": "20260910T100000"}, inputs=None, hw=config.cpu_device())
    m = store.get("simulation", d.name)
    assert m.id == d.name and m.body["batches_done"] == 0 and m.body["complete"] is False
    assert m.parents == {"prior": "20260910T100000"} and m.fingerprints["gmm"] == "b" * 16
    tc.save(d, from_batch=0, batch_k=2, rng=None, x_buf=torch.zeros(12, 5), th_buf=torch.zeros(12, 2), run_size=4)
    m2 = store.get("simulation", d.name)
    assert m2.body["batches_done"] == 2 and m2.created == m.created
    tc.mark_complete(d, 3, rows=(12, 5))
    m3 = store.get("simulation", d.name)
    assert m3.body["complete"] is True and m3.body["rows"] == [12, 5] and m3.body["wall_seconds"] >= 0.0
    assert store.simulation_for(ident) == d and store.list("simulation")[0].complete


def test_load_prior_refuses_model_param_order_and_box_and_returns_a_wrapper(store):
    cfg = _nad_cfg()
    ok = _prior_artifact(store, cfg, name="ok")
    lp = store.load_prior(cfg, "ok")
    assert lp.id == ok.id and lp.name == "ok" and lp.fingerprint == store.get("prior", ok.id).fingerprints["gmm"]
    assert lp.prior.distributions[0] is lp.nd_prior and lp.force_prior is not None   # master.txt has a drive
    keys = list(cfg.params_dict)
    hi = [b[1] for _, b in cfg.params_dict.values()]
    from core.SBI.reparam import nd_log_mask
    from core.SBI.run_guards import _log_params_for
    # The log mask says WHICH coordinate the GMM was fit in, so a prior built when
    # REPARAM_LOG_PARAMS said something else is a distribution over different numbers entirely --
    # and unlike the box, nothing downstream would ever notice.
    flipped = [not bool(v) for v in nd_log_mask(cfg, log_params=_log_params_for(cfg)).tolist()]
    for tag, kw, why in (("model", dict(model="HOPF"), "the model"),
                         ("order", dict(keys=keys[1:] + keys[:1]), "ORDER"),
                         ("box", dict(highs=[hi[0] * 2] + hi[1:]), "the ND box"),
                         ("mask", dict(mask=flipped), "log-box")):
        _prior_artifact(store, cfg, name=tag, **kw)
        with pytest.raises(ValueError, match=why):
            store.load_prior(cfg, tag)
    other = _prior_artifact(store, cfg, name="other", seed=1)
    (ok.dir / "prior.pt").write_bytes((other.dir / "prior.pt").read_bytes())
    with pytest.raises(st.StoreError, match="not the one the manifest records"):
        store.load_prior(cfg, "ok")


def test_build_prior_auto_persists_and_loads_back(store, monkeypatch):
    from core import orchestrator
    from core.SBI.reparam import nd_log_mask
    from core.SBI.run_guards import _log_params_for
    cfg = _nad_cfg()

    def stub_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds, **kw):
        lows = [float(b[0]) for b in prior_bounds]
        highs = [float(b[1]) for b in prior_bounds]
        return _gmm_in_box(lows, highs, kw.get("log_mask"))

    monkeypatch.setattr(orchestrator.pipeline, "gen_prior", stub_gen_prior)
    seen = []
    lp = orchestrator.build_prior(cfg, None, True, fig_sink=_recording(seen), num_iterations=1)
    assert lp.name == "" and [r.id for r in store.list("prior")] == [lp.id]
    assert (lp.path / "figures" / "prior.png").stat().st_size > 0 and seen == ["Prior"]
    m = lp.manifest
    assert m.body["gmm"]["param_keys"] == list(cfg.params_dict) and m.config["num_iterations"] == 1
    assert m.body["gmm"]["box"]["log_mask"] == nd_log_mask(cfg, log_params=_log_params_for(cfg)).tolist()
    assert m.inputs["bounds"]["path"] == "Bounds/nadrowski/master.txt" and m.fingerprints["gmm"] == lp.fingerprint
    again = orchestrator.build_prior(cfg, lp.id, False, fig_sink=_close)
    assert again.id == lp.id and again.fingerprint == lp.fingerprint
    store.rename("prior", lp.id, "master_prior")
    assert orchestrator.build_prior(cfg, "master_prior", False, fig_sink=_close).name == "master_prior"


def test_loading_a_prior_with_no_sink_closes_its_figure(store):
    """V8 (spec §6.3). build_prior's LOAD branch handed fig_sink straight to visualize_dist, whose None
    fallback is a bare plt.show(). Under the tool's Agg backend (and the suites') that does nothing but
    warn "FigureCanvasAgg is non-interactive, and thus cannot be shown", and it never closed the corner
    figure, so every library load leaked one live figure for the life of the process. The BUILD branch
    never did: its writer's sink (ArtifactWriter.fig_sink) saves the PNG and closes the figure when
    nothing is forwarded. Now both branches close what they draw when given no sink, while
    visualize_dist and emit_figure keep their own fallback for a bare-library caller. Pinned as the FDT
    precedent is (test_tool.py::test_fdt_plot_functions_close_a_saved_figure_instead_of_show): no new
    open figure, no non-interactive warning.

    The three fig_sink docstrings (build_prior, build_posterior, infer_and_visualize) now say the same
    one thing, and it is true of every stage. Before, one said "None => plt.show()", one said "None is
    the bare-library fallback to plt.show()", and only the load branch behaved like either."""
    import inspect
    import warnings

    from matplotlib import pyplot as plt

    from core import orchestrator

    cfg = _nad_cfg()
    _prior_artifact(store, cfg, name="p")
    before = plt.get_fignums()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        loaded = orchestrator.build_prior(cfg, "p", False)
    assert loaded.name == "p"
    assert not any("non-interactive" in str(w.message) for w in rec), [str(w.message) for w in rec]
    assert plt.get_fignums() == before, \
        f"the load leaked figures {sorted(set(plt.get_fignums()) - set(before))}"

    for fn in (orchestrator.build_prior, orchestrator.build_posterior, orchestrator.infer_and_visualize):
        doc = " ".join(inspect.getdoc(fn).split())
        assert "Every front end passes one; a stage given None closes what it draws" in doc, fn.__name__
        assert "plt.show()" not in doc, fn.__name__


def _weights_that_renormalise_inexactly(n=64):
    """Float32 mixture weights whose SAVED form is not a fixed point of torch's Categorical
    re-normalisation on this CPU. ``Categorical(probs=w)`` stores ``w / w.sum()``; the sum of an
    already-normalised float32 vector is 1 +/- 1 ulp, so re-normalising what was saved moves bits
    for some vectors and not others, depending on the values and the device's reduction order.
    Searched rather than hard-coded, so a change in torch's summation cannot silently make the tests
    below vacuous; the raise is the precondition they rely on."""
    for seed in range(500):
        q = torch.rand(n, generator=torch.Generator().manual_seed(seed))
        saved = torch.distributions.Categorical(probs=q).probs        # what _gmm_in_box holds and save_mix_dist writes
        reloaded = torch.distributions.Categorical(probs=saved).probs  # what a naive loader rebuilds
        if not torch.equal(reloaded, saved):
            return q
    raise AssertionError("no seed produced weights that re-normalise inexactly on this CPU; the round-trip "
                         "tests would be vacuous")


def test_load_mix_dist_is_the_exact_inverse_of_save_mix_dist(tmp_path):
    """A prior's fingerprint hashes its GMM's weight and mean BYTES, so a save/load round trip has to
    return exactly what was saved -- on every device. The 2026-09-11 GPU gate found a prior that
    reloaded bit-exactly on CUDA but not on the CPU, and another that reloaded on neither and was
    refused as 'inconsistent' by its own store the moment build_prior read it back."""
    from core.Helpers import file_manager
    from core.SBI.reparam import nd_log_mask
    from core.SBI.run_guards import _find_nd_gmm, _gmm_fingerprint, _log_params_for
    cfg = _nad_cfg()
    lows = [float(b[0]) for _, b in cfg.params_dict.values()]
    highs = [float(b[1]) for _, b in cfg.params_dict.values()]
    dist = _gmm_in_box(lows, highs, nd_log_mask(cfg, log_params=_log_params_for(cfg)),
                       weights=_weights_that_renormalise_inexactly())
    path = tmp_path / "prior.pt"
    file_manager.save_mix_dist(dist, str(path), model=cfg.model, param_keys=list(cfg.params_dict))
    back = file_manager.load_mix_dist(str(path), device=torch.device("cpu"))
    g0, g1 = _find_nd_gmm(dist), _find_nd_gmm(back)
    assert torch.equal(g1.mixture_distribution.probs, g0.mixture_distribution.probs), "weights changed on reload"
    assert torch.equal(g1.component_distribution.loc, g0.component_distribution.loc), "means changed on reload"
    assert _gmm_fingerprint(back) == _gmm_fingerprint(dist)
    # Still a working distribution: it samples, and an expanded copy carries the exact weights too
    # (Categorical.expand copies _param, which is why the loader restores both attributes).
    assert back.sample((3,)).shape == (3, len(lows))
    assert torch.equal(g1.mixture_distribution.expand(torch.Size([2])).probs[0], g0.mixture_distribution.probs)


def test_build_prior_reloads_a_gmm_whose_weights_do_not_renormalise_exactly(store, monkeypatch):
    """The gate failure of 2026-09-11 reproduced on the CPU: build_prior writes the prior, records its
    fingerprint and reads it straight back through store.load_prior, which refused the artifact it
    had just written ("prior.pt holds GMM ..., not the one the manifest records")."""
    from core import orchestrator
    w = _weights_that_renormalise_inexactly()

    def stub_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds, **kw):
        return _gmm_in_box([float(b[0]) for b in prior_bounds], [float(b[1]) for b in prior_bounds],
                           kw.get("log_mask"), weights=w)

    monkeypatch.setattr(orchestrator.pipeline, "gen_prior", stub_gen_prior)
    cfg = _nad_cfg()
    lp = orchestrator.build_prior(cfg, None, True, fig_sink=_close, num_iterations=1)
    assert lp.fingerprint == lp.manifest.fingerprints["gmm"]
    assert store.load_prior(cfg, lp.id).fingerprint == lp.fingerprint


def test_a_taken_name_is_refused_before_the_spend(store, tiny_run, monkeypatch):
    """A duplicate name is refused at the stage's ENTRY, not by the write at the end. store.create is
    where the collision used to surface, and it runs after the ~9-minute sweep / the days of
    training -- so the stage would do all of its work and then throw it away. monkeypatch restores
    both stubs on its own, including on failure."""
    from core import orchestrator
    cfg = _nad_cfg()
    _prior_artifact(store, cfg, name="p")
    swept = []
    monkeypatch.setattr(orchestrator.pipeline, "gen_prior", lambda *a, **k: swept.append(1))
    with pytest.raises(st.StoreError, match="already exists"):
        orchestrator.build_prior(cfg, None, True, name="p")
    assert swept == [], "the stability sweep ran before the name was checked"

    r = tiny_run
    assert r.posterior.name, "the fixture's posterior must be named for this to test anything"
    trained = []
    monkeypatch.setattr(orchestrator.pipeline, "train_nn", lambda *a, **k: trained.append(1))
    with pytest.raises(st.StoreError, match="already exists"):
        orchestrator.build_posterior(r.cfg, r.prior, None, True, name=r.posterior.name, store=r.store,
                                     num_runs=2, run_size_cap=8, fig_sink=r.sink)
    assert trained == [], "the training run started before the name was checked"


def test_delete_refuses_a_prior_a_simulation_was_generated_against(store):
    from core.SBI import training_checkpoint as tc
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="p")
    fp = store.get("prior", p.id).fingerprints["gmm"]
    ident = {"format": "training-rows/2", "prior_fingerprint": fp, "n_runs": 3, "truncation": None}
    d = tc.resolve_dir(ident)
    tc.create(d, ident, schedule_t_scales=torch.ones(3), schedule_Ts=torch.ones(3), inits=torch.zeros(1, 2),
              V=None, probe=torch.zeros(7, 2, dtype=torch.float64), run_size=4, n_runs=3,
              parents={"prior": p.id}, hw=config.cpu_device())
    with pytest.raises(st.StoreError, match="simulation"):
        store.delete("prior", p.id)
    assert store.find_prior_by_fingerprint(fp).id == p.id


def test_delete_refuses_a_prior_a_simulation_was_generated_against_by_fingerprint(store):
    """The spec's prior-fingerprint clause. The cache directory is KEYED on the prior's GMM and its
    rows are meaningless without the prior they were drawn from, so a simulation generated against
    this prior blocks the delete whether or not it names it as a PARENT -- a cache written before the
    parent was recorded, or by a script holding a stand-in prior, names none."""
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="p")
    fp = store.get("prior", p.id).fingerprints["gmm"]
    ident = {"format": "training-rows/2", "prior_fingerprint": fp, "n_runs": 3, "truncation": None}
    m = st.write_simulation_manifest(store.kind_dir("simulation") / "abcdef012345", ident, parents=None)
    assert m.parents == {} and m.fingerprints["gmm"] == fp
    assert store.dependents("prior", p.id) == [("simulation", m.id, "")]
    with pytest.raises(st.StoreError, match=m.id):
        store.delete("prior", p.id)
    other = _prior_artifact(store, cfg, name="other", seed=7)      # another GMM is not a dependency
    assert store.dependents("prior", other.id) == []
    store.delete("prior", p.id, force=True)
    with pytest.raises(st.StoreError):
        store.get("prior", p.id)


def test_a_training_run_records_the_prior_as_the_simulation_cache_parent(store, monkeypatch):
    """The checkpoint dict build_posterior hands gen_training_data names the LoadedPrior's id, and the
    identity's prior fingerprint is the wrapper's -- checked at the seam, with train_nn stubbed.

    The cadence and the epoch cap ride in as ARGUMENTS (piece 2, §2.6): checkpoint_every reaches the
    plan's checkpoint dict and max_num_epochs reaches train_nn, neither of them through a module
    constant that orchestrator snapshotted at import."""
    from core import orchestrator
    from core.artifacts.identity import SimulationIdentity
    cfg = _nad_cfg()
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    cfg.reparam_rotate = False
    captured = {}

    def fake_train_nn(plan, **kw):
        captured["plan"], captured["kw"] = plan, kw
        raise RuntimeError("stop before training")

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", fake_train_nn)
    with pytest.raises(RuntimeError, match="stop before training"):
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4,
                                     checkpoint_every=1, max_num_epochs=7)
    ck = captured["plan"].checkpoint
    assert ck["parents"] == {"prior": lp.id} and ck["inputs"]["model"] == "NADROWSKI"
    # V1: the plan shares the hardware of the stage's PRIVATE copy -- equal to the caller's by value
    # (DeviceConfig is an eq dataclass), never the caller's object
    assert ck["hw"] == cfg.hw and ck["hw"] is not cfg.hw
    assert ck["identity"] == SimulationIdentity.from_cfg(cfg, lp, 4, 2).to_dict()
    assert ck["identity"]["prior_fingerprint"] == lp.fingerprint and ck["dir"].parent == store.kind_dir("simulation")
    assert ck["every"] == 1 and ck["resume"] == "auto", "the cadence and the policy must be the arguments"
    assert captured["kw"]["max_num_epochs"] == 7, "max_num_epochs was accepted and dropped"


class _PriorWrap:
    """SBIPriorWrapper's shape (.gen_dist), module-level so a payload carrying it pickles."""
    def __init__(self, inner):
        self.gen_dist = inner


def test_load_posterior_refuses_each_mismatch_class(store):
    cfg = _nad_cfg(chi_mode=True)
    ok = _posterior_artifact(store, cfg, name="ok")
    lp = store.load_posterior(cfg, "ok")
    assert lp.id == ok.id and lp.posterior.truncation is None and lp.accepted == [] and lp.latent is not None
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    hi = [float(b[1]) for _, b in cfg.params_dict.values()]
    cases = {
        "model": ({("config", "model"): "HOPF"}, "trained for model"),
        "order": ({("transform", "param_keys"): keys[1:] + keys[:1]}, "ORDER"),
        "box": ({("transform", "nd_highs"): [hi[0] * 2] + hi[1:]}, "different box"),
        "mode": ({("mode",): "forced"}, "mode"),
        "layout": ({("conditioning", "chi_layout"): 1}, "layout"),
        "k_pad": ({("conditioning", "chi_k_pad"): int(cfg.chi_k_pad) + 1}, "probe-slot"),
        "elem_w": ({("conditioning", "chi_elem_w"): 5}, "channels per slot"),
        "band": ({("conditioning", "chi_freq_bounds"): [0.1, 10.0]}, "band"),
        "cycles": ({("conditioning", "chi_max_cycles"): 5.0}, "cycle"),
        "width": ({("conditioning", "forcing_dim"): 3, ("conditioning", "width"): 53}, "incompatible"),
    }
    for tag, (over, why) in cases.items():
        _posterior_artifact(store, cfg, name=tag, over=over)
        with pytest.raises(ValueError, match=why):
            store.load_posterior(cfg, tag)


def test_manifest_V_must_equal_the_rotation_in_the_pickled_prior(store):
    """The D6 spec test, moved: the V a manifest records is the one the posterior's own training prior
    carries. With no legacy sidecars left, the transpose is REFUSED, not repaired."""
    from core.SBI import reparam
    from core.SBI.training_checkpoint import bijection_probe
    cfg = _nad_cfg(chi_mode=True)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    torch.manual_seed(3)
    Q = torch.linalg.qr(torch.randn(P, P))[0]
    assert not torch.allclose(Q, Q.T)
    T_train = reparam.build_rotated_bijection(reparam.build_inferred_bijection(cfg, log_params=[]), Q)
    V = reparam.rotation_of(T_train)                      # what build_posterior records
    base = torch.distributions.MultivariateNormal(torch.zeros(P), torch.eye(P))
    trained = _PriorWrap(reparam.RotatedLatentPrior(base, Q))   # module-level: it is pickled inside the payload

    _posterior_artifact(store, cfg, name="good", V=V, prior=trained)
    lp = store.load_posterior(cfg, "good")
    assert torch.allclose(bijection_probe(lp.posterior.T, P), bijection_probe(T_train, P))
    _posterior_artifact(store, cfg, name="transposed", V=V.T, prior=trained)
    with pytest.raises(ValueError, match="TRANSPOSE"):
        store.load_posterior(cfg, "transposed")
    _posterior_artifact(store, cfg, name="unrotated", V=None, prior=trained)
    with pytest.raises(ValueError, match="NO rotation"):
        store.load_posterior(cfg, "unrotated")
    _posterior_artifact(store, cfg, name="plain", V=V, prior=None)   # a prior without a rotation cannot arbitrate
    assert reparam.rotation_of(store.load_posterior(cfg, "plain").posterior.T) is not None


def test_a_non_amortized_posterior_needs_accept_and_the_flag_is_recorded(store, caplog):
    from core.artifacts import Accept
    from core.refusals import Refusal
    from core.SBI import reparam, truncate
    from core.SBI.training_checkpoint import bijection_probe
    cfg = _nad_cfg(chi_mode=True)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    T = reparam.build_inferred_bijection(cfg, log_params=[])
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], n_latent=P, V=None,
                                       probe=bijection_probe(T, P), x_obs_digest="d" * 16)
    _posterior_artifact(store, cfg, name="trunc", amortized=False, region=region)
    with pytest.raises(Refusal, match="NOT AMORTIZED") as excinfo:
        store.load_posterior(cfg, "trunc")
    # V3: the message is NEUTRAL. It names the Python hatch -- the core API's own name -- and no
    # control a front end owns; the FIELD is what each front end maps to its dialog or its flag
    # (core/gui/fields.py, core/tool/fields.py), so a renamed button can never go stale here.
    assert excinfo.value.field == "accept_truncated"
    assert "Accept(truncated=True)" in str(excinfo.value)
    for banned in ("Posterior tab", "--accept-truncated", "command line"):
        assert banned not in str(excinfo.value), banned
    lp = store.load_posterior(cfg, "trunc", accept=Accept(truncated=True))
    assert lp.posterior.truncation.dims == [0, 1] and lp.posterior.x_obs_digest == "d" * 16
    assert lp.accepted == ["truncated"] and torch.equal(lp.posterior.truncation.probe, region.probe)
    # The same load through build_posterior's LOAD branch says so at WARNING, once, from
    # core.orchestrator (spec §4.1, walkthrough row C9): this line reaching the pane plain, at info,
    # is the defect piece 3 was motivated by. A stand-in prior: the load branch reads none of it.
    import logging
    from types import SimpleNamespace
    from core import orchestrator
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="core"):
        got = orchestrator.build_posterior(cfg, SimpleNamespace(prior=None, force_prior=None), "trunc",
                                           False, accept=Accept(truncated=True), store=store)
    assert got.posterior.x_obs_digest == "d" * 16
    said = [(r.name, r.levelname) for r in caplog.records
            if r.getMessage().startswith("[tsnpe] loaded a NON-AMORTIZED posterior")]
    assert said == [("core.orchestrator", "WARNING")], \
        [(r.name, r.levelname, r.getMessage()[:60]) for r in caplog.records]
    no_basis = truncate.TruncationRegion([0], [-1.0], [1.0], n_latent=P, V=None, probe=None, x_obs_digest="e" * 16)
    _posterior_artifact(store, cfg, name="noprobe", amortized=False, region=no_basis)
    with pytest.raises(ValueError, match="basis"):
        store.load_posterior(cfg, "noprobe", accept=Accept(truncated=True))
    # ... and a region that does not say WHICH observation it was drawn around is refused the same
    # way: guardrail 2's refusal in infer_and_visualize compares against that digest, so without it
    # the artifact would serve any observation as if it were the one it is valid near.
    no_digest = truncate.TruncationRegion([0], [-1.0], [1.0], n_latent=P, V=None,
                                          probe=bijection_probe(T, P), x_obs_digest=None)
    _posterior_artifact(store, cfg, name="nodigest", amortized=False, region=no_digest)
    with pytest.raises(ValueError, match="does not name the observation"):
        store.load_posterior(cfg, "nodigest", accept=Accept(truncated=True))
    assert store.list("posterior")[0].amortized is not None


def test_a_legacy_or_schemaless_directory_is_refused_as_not_a_current_artifact(store):
    d = store.kind_dir("posterior") / "legacy__20200101T000000"
    d.mkdir(parents=True)
    torch.save(_FakeDP(), d / "posterior.pt")
    (d / "posterior.rot.pt").write_bytes(b"")
    with pytest.raises(st.StoreError, match="no complete posterior"):
        store.load_posterior(_nad_cfg(), "legacy__20200101T000000")
    assert store.list("posterior")[0].complete is False
    (d / "manifest.json").write_text(json.dumps({"schema": 0}), encoding="utf-8")
    assert "schema" in store.list("posterior")[0].reason


def test_build_posterior_auto_persists_and_returns_the_loaded_wrapper(store, monkeypatch):
    """train_nn is stubbed to return a _FakeDP carrying the training prior; everything around it --
    the identity, the parents, the transform block, the loss curve, the figure, the load-back -- is real."""
    from core import orchestrator
    from core.artifacts import Accept
    from core.SBI import reparam
    cfg = _nad_cfg()
    cfg.reparam_rotate = False
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)

    def fake_train_nn(plan, **kw):
        dp = _FakeDP()
        dp.prior = kw["prior"]                    # the SBIPriorWrapper build_posterior passed in
        return dp, {"training_loss": [1.0, 0.5], "validation_loss": [1.1, 0.6],
                    "best_validation_loss": 0.6, "epochs_trained": 2, "stop_after_epochs": 1}

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", fake_train_nn)
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data",
                        lambda plan, **kw: (torch.zeros(8, 50), torch.zeros(8, 13)))
    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 0)
    seen = []
    out = orchestrator.build_posterior(cfg, lp, None, True, fig_sink=_recording(seen),
                                       num_runs=2, run_size_cap=4, hidden_features=8, num_transforms=1,
                                       stop_after_epochs=1, note="tiny")
    assert isinstance(out, st.LoadedPosterior) and out.name == "" and out.diagnostics["epochs_trained"] == 2
    m = store.get("posterior", out.id)
    assert m.parents == {"prior": lp.id} and m.note == "tiny" and m.body["amortized"] is True
    assert m.body["transform"]["V"] is None and m.body["transform"]["param_keys"][-1] in cfg.rescale_params
    assert m.body["training"]["hidden_features"] == 8 and m.body["training"]["best_validation_loss"] == 0.6
    assert m.config["num_runs"] == 2 and m.fingerprints["gmm"] == lp.fingerprint
    # V7: an unrotated run ran no Fisher, so it records no Fisher settings -- not defaults it never used
    assert (m.config["fisher_m"], m.config["fisher_dz"], m.config["fisher_points"]) == (None, None, None), \
        m.config
    assert set(m.payloads) == {"posterior.pt", "loss.npz"} and m.figures == ["figures/training_loss.png"]
    assert seen == ["Training loss"]
    back = orchestrator.build_posterior(cfg, lp, out.id, False)
    assert back.id == out.id and reparam.rotation_of(back.posterior.T) is None
    with pytest.raises(ValueError, match="already carries"):
        orchestrator.build_posterior(cfg, lp, out.id, False, truncation=object())

    # The store suite's successor to the retired sidecar test: a ROTATED run's manifest carries the
    # Fisher eigenvalues descending, the same field the retired '.rot.pt' sidecar used to hold.
    cfg.reparam_rotate = True
    Q, _ = torch.linalg.qr(torch.randn(13, 13))
    monkeypatch.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation",
                        lambda *a, **k: (Q, torch.arange(13, 0, -1).float()))
    out2 = orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4,
                                        hidden_features=8, num_transforms=1, stop_after_epochs=1)
    m2 = store.get("posterior", out2.id)
    assert m2.body["transform"]["fisher_eigenvalues"][0] == 13.0
    assert m2.config["fisher_m"] == config.REPARAM_FISHER_M, "a rotated run records what its Fisher ran with"


def test_a_torn_posterior_write_at_the_stage_leaves_no_half_artifact(store, monkeypatch):
    """The stage's own posterior.pt write must stay atomic: a failing torch.save inside build_posterior
    propagates and leaves NO posterior directory behind (the writer removes it) -- the property the
    retired save_posterior_artifacts test pinned at its call site, pinned again at the new one."""
    from core import orchestrator
    from tests._fixtures import _FakeDP, _WriteFailed, _failing, _nad_cfg, _prior_artifact
    cfg = _nad_cfg()
    cfg.reparam_rotate = False
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)

    def fake_train_nn(plan, **kw):
        dp = _FakeDP()
        dp.prior = kw["prior"]
        return dp, {"training_loss": [1.0], "validation_loss": [1.0], "best_validation_loss": 1.0,
                    "epochs_trained": 1, "stop_after_epochs": 1}

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", fake_train_nn)
    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 0)
    before = [r.id for r in store.list("posterior")]
    real_save = torch.save
    monkeypatch.setattr(torch, "save", _failing(real_save))
    with pytest.raises(_WriteFailed):
        orchestrator.build_posterior(cfg, lp, None, True, fig_sink=_close, num_runs=2, run_size_cap=4,
                                     hidden_features=8, num_transforms=1, stop_after_epochs=1)
    monkeypatch.setattr(torch, "save", real_save)
    assert [r.id for r in store.list("posterior")] == before, "a torn write left a posterior directory behind"
    assert not list(store.kind_dir("posterior").rglob("*.tmp"))


def test_a_torn_posterior_write_leaves_no_half_artifact(store):
    """The end-of-run write is one ``store.create`` block: a failure mid-write must leave no half
    artifact directory and must not disturb an already-complete sibling."""
    from core.Helpers import file_manager
    from tests._fixtures import _WriteFailed, _failing
    cfg = _nad_cfg(chi_mode=True)
    first = _posterior_artifact(store, cfg, name="first")
    real_save = torch.save
    torch.save = _failing(real_save)
    try:
        with pytest.raises(_WriteFailed):
            with store.create("posterior", cfg, name="second") as w:
                file_manager.atomic_torch_save({"generation": 2}, w.payload("posterior.pt"))
    finally:
        torch.save = real_save
    assert [r.name for r in store.list("posterior")] == ["first"] and not (w.dir).exists()
    assert not list(first.dir.glob("*.tmp"))


def _forced_cfg():
    from core import cli
    from core.config import CELL_PATH
    cfg = _nad_cfg()                                         # master.txt declares a drive -> forced mode
    cli.load_and_validate_gt(cfg, str(CELL_PATH / "nadrowski" / "master_weak.txt"))
    cfg.T_obs = 200.0                                        # cell units: 200 frames at dt_exp = 1
    return cfg


def test_generate_observations_writes_an_artifact_that_reinstalls_its_context(store):
    from core import orchestrator
    cfg = _forced_cfg()
    seen = []
    obs = orchestrator.generate_observations(cfg, fig_sink=_recording(seen), name="cell")
    assert obs.name == "cell" and obs.mode == "forced" and obs.width == obs.x_obs.shape[-1]
    assert seen == ["Ground-truth trace"] and obs.manifest.figures == ["figures/ground_truth_trace.png"]
    src = obs.manifest.body["source"]
    assert src["kind"] == "simulated" and src["cell"]["path"] == "Cells/nadrowski/master_weak.txt"
    assert src["params"] == {k: float(v) for k, (v, _) in cfg.params_dict.items()}
    assert obs.manifest.fingerprints["x_obs"] == obs.digest == mf.tensor_digest(obs.x_obs)
    fresh = _nad_cfg()
    assert not fresh.has_ground_truth
    store.load_observation(fresh, obs.id).install(fresh)
    # V1: the resolved length is read off the ARTIFACT. The stage wrote it on its private copy, so the
    # caller's config still has none.
    assert fresh.has_ground_truth and fresh.T_obs == cfg.T_obs
    assert fresh.n_obs == obs.manifest.body["n_obs"] == 200 and cfg.n_obs is None
    assert fresh.ground_truth == cfg.ground_truth
    with pytest.raises(ValueError, match="mode"):
        store.load_observation(_nad_cfg(chi_mode=True), obs.id)


def test_build_experiment_observation_hashes_recordings_and_refuses_a_missing_file(store, tmp_path):
    """Defects 3-4: the forced-recording path raised NameError before this task. It now runs, and the
    artifact names every recording by path and hash."""
    import numpy as np
    from core import orchestrator
    from core.SBI.observations import RecordingSet
    cfg = _forced_cfg()
    sim = orchestrator.generate_observations(cfg, fig_sink=_close)
    trace = sim.obs_data[0].numpy()
    spont, forced = tmp_path / "spont.npy", tmp_path / "forced.npy"
    np.save(spont, trace)
    np.save(forced, trace)
    T_obs_s = cfg.T_obs / cfg.get_unit_conversion_factor("s")
    si = {n: (5.0 if n == "freq" else 1e-12 if n == "amp" else 0.0) for n in cfg.force_params_dict}
    rec = RecordingSet(spont=str(spont), forced=((str(forced), None),), T_obs_s=T_obs_s, forcing_params_si=si)
    obs = orchestrator.build_experiment_observation(cfg, rec, fig_sink=_close)
    recs = obs.manifest.body["source"]["recordings"]
    assert [r["role"] for r in recs] == ["spont", "forced"] and all(len(r["sha256"]) == 64 for r in recs)
    assert obs.width == sim.width and obs.manifest.body["source"]["kind"] == "experimental"
    assert obs.manifest.body["forcing_vals"]["freq"] > 0
    # A Refusal naming the passive recording's field (spec §3.3), not a FileNotFoundError: the front
    # ends show it as "check your inputs" and name the control, not as a crash.
    with pytest.raises(Refusal) as e:
        orchestrator.build_experiment_observation(
            cfg, RecordingSet(spont=str(tmp_path / "nope.npy"), forced=((str(forced), None),), T_obs_s=T_obs_s,
                              forcing_params_si=si), fig_sink=_close)
    assert e.value.field == "recording_spont" and "nope.npy" in str(e.value), str(e.value)
    assert not isinstance(e.value, FileNotFoundError)


def test_chi_mode_refuses_a_forced_recording_without_its_drive_frequency(store, tmp_path, monkeypatch):
    """D9: every driven chi recording states the frequency (Hz) it was driven at.

    The refusal has to be scoped to the FORCED role. The stage's file-check loop is
    ``[(rec.spont, "spont", None)] + [(p, "forced", f) for p, f in rec.forced]``, so its first element
    is the passive recording with no frequency BY CONSTRUCTION -- an unscoped check would reject every
    chi observation. The lock-in is a sinc in the frequency error: a probe aimed at a guessed
    ``mult_k * Omega_0`` rather than at the drive the bench actually applied decays to noise, silently.
    Both legs are pinned: the frequency-less set is refused before anything is read or built, and the
    complete set reaches the chi builder as ``(recording, Hz)`` pairs.
    """
    import numpy as np
    from core import orchestrator
    from core.Helpers import file_manager
    from core.SBI.observations import RecordingSet

    cfg = _nad_cfg(chi_mode=True)
    cfg.T_obs = 200.0                                        # cell units: 200 frames at dt_exp = 1
    spont = tmp_path / "passive.npy"
    np.save(spont, np.zeros(200, dtype=np.float32))
    driven = []
    for i in range(2):
        p = tmp_path / f"driven_{i}.npy"
        np.save(p, np.zeros(200, dtype=np.float32))
        driven.append(str(p))

    class _Reached(RuntimeError):
        """Raised by the chi-builder spy: the stage got past every refusal."""

    read, seen = [], []

    def _record_load(path, dtype=None):
        read.append(str(path))
        return torch.zeros(200)

    def _spy(cfg_, X_spont, X_forced_list, T_obs_s, F0_si):
        seen.append(X_forced_list)
        raise _Reached

    monkeypatch.setattr(file_manager, "load_experimental_data", _record_load)
    monkeypatch.setattr(orchestrator, "build_experiment_obs_chi", _spy)

    # (a) one driven recording without its frequency: refused, by name, before anything is read
    gap = RecordingSet(spont=str(spont), forced=((driven[0], 12.5), (driven[1], None)),
                       T_obs_s=1.0, F0_si=1.0)
    with pytest.raises(ValueError, match="no drive frequency") as e:
        orchestrator.build_experiment_observation(cfg, gap, fig_sink=_close)
    assert "driven_1.npy" in str(e.value), str(e.value)
    assert read == [], f"a recording was read before the refusal: {read}"
    assert seen == [], "the frequency-less set reached the chi builder"
    assert not list(store.kind_dir("observation").glob("*")), "a refused set wrote an observation"

    # (b) every driven recording names its frequency: not refused, and the builder gets (x, Hz) pairs
    ok = RecordingSet(spont=str(spont), forced=((driven[0], 12.5), (driven[1], 25.0)),
                      T_obs_s=1.0, F0_si=1.0)
    with pytest.raises(_Reached):
        orchestrator.build_experiment_observation(cfg, ok, fig_sink=_close)
    assert [f for _x, f in seen[0]] == [12.5, 25.0]
    assert all(torch.is_tensor(x) and isinstance(f, float) for x, f in seen[0]), seen[0]


def test_calibration_writes_results_ranks_figures_and_refuses_a_foreign_prior(tiny_run):
    import numpy as np
    from core import orchestrator
    r = tiny_run
    cal = orchestrator.validate_calibration(r.cfg, r.posterior, r.prior, fig_sink=r.sink, n_cal=8, cal_n_scales=2,
                                            num_posterior_samples=40, name="cal")
    m, res = cal.manifest, cal.results
    keys = list(r.cfg.params_dict) + list(r.cfg.rescale_params)
    assert m.parents == {"posterior": r.posterior.id, "prior": r.prior.id} and m.config["n_cal"] == 8
    assert [r["name"] for r in res["sbc"]["per_param"]] == keys
    assert set(res["sbc"]["per_param"][0]) == {"name", "ks_p", "c2st_ranks", "c2st_dap"}
    assert set(res["tarp"]) == {"atc", "ks_p"} and res["num_posterior_samples"] == 40 and res["kept_fraction"] is None
    assert res["informativeness"] is None or "total_nats" in res["informativeness"]
    assert set(m.payloads) == {"ranks.npz", "results.json"}
    assert sorted(m.figures) == ["figures/sbc_ranks_cdf.png", "figures/sbc_ranks_histogram.png", "figures/tarp_coverage.png"]
    assert np.load(cal.path / "ranks.npz")["ranks"].shape[0] == 8
    assert json.loads((cal.path / "results.json").read_text(encoding="utf-8"))["tarp"] == res["tarp"]
    assert r.store.load_calibration(cal.id).results == res
    with pytest.raises(ValueError, match="not the one this posterior was trained with"):
        orchestrator.validate_calibration(r.cfg, r.posterior, r.other_prior(), fig_sink=r.sink, n_cal=4, cal_n_scales=1)


def test_inference_records_ppc_summary_and_ground_truth(tiny_run):
    from core import orchestrator
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="obs")
    inf = orchestrator.infer_and_visualize(r.cfg, r.posterior, obs, fig_sink=r.sink, n_samples=50, name="inf")
    m, res = inf.manifest, inf.results
    keys = list(r.cfg.params_dict) + list(r.cfg.rescale_params)
    assert m.parents == {"posterior": r.posterior.id, "observation": obs.id} and res["n_samples"] == 50
    assert res["accepted"] == [] and [r["name"] for r in res["posterior_summary"]] == keys
    assert set(res["posterior_summary"][0]) == {"name", "q05", "median", "q95"}
    assert res["ground_truth"] == {k: float(v) for k, v in zip(keys, r.cfg.ground_truth)}
    assert {"mean_abs_z", "max_abs_z", "coverage_90", "num_outside", "num_invalid"} <= set(res["ppc"])
    assert {"figures/posterior_corner.png", "figures/posterior_predictive_check.png", "figures/eye_test.png"} <= set(m.figures)
    assert set(m.payloads) == {"samples.pt", "results.json"}
    assert tuple(torch.load(inf.samples_path, weights_only=False).shape) == (50, len(keys))
    assert r.store.load_inference(inf.id).results == res and m.fingerprints["x_obs"] == obs.digest


def test_inference_refuses_a_foreign_observation_for_a_truncated_posterior_unless_accepted(tiny_run):
    """Successor of test_conditioning_repair's warning test: a NON-AMORTIZED posterior on an observation
    other than its region's is a REFUSAL now, and accepting it is written into the inference."""
    from copy import copy
    from core import orchestrator
    from core.artifacts import Accept
    from core.SBI import reparam
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink)
    claims_another = copy(r.posterior)
    claims_another.posterior = reparam.TransformedPosterior(r.posterior.latent, r.posterior.posterior.T,
                                                             truncation=None, x_obs_digest="f" * 16)
    # A REFUSED inference must not install the rejected observation's context: T_obs, the probe
    # frequencies and the truth would stay on cfg for whatever the session does next. Perturbed first
    # so the assertion has teeth (install would overwrite it with the observation's own value).
    was = r.cfg.T_obs
    r.cfg.T_obs = was + 7.0
    try:
        with pytest.raises(Refusal, match="NOT AMORTIZED") as excinfo:
            orchestrator.infer_and_visualize(r.cfg, claims_another, obs, fig_sink=r.sink, n_samples=20)
        # V3: the message names the Python hatch only; each front end appends its own control from its
        # table (core/gui/fields.py, core/tool/fields.py), keyed by the field.
        assert excinfo.value.field == "accept_other_observation"
        assert "Accept(other_observation=True)" in str(excinfo.value)
        for banned in ("Run on a different observation", "Infer tab", "--accept-other-observation"):
            assert banned not in str(excinfo.value), banned
        assert r.cfg.T_obs == was + 7.0, "a refused inference installed the observation's context anyway"
    finally:
        r.cfg.T_obs = was
    inf = orchestrator.infer_and_visualize(r.cfg, claims_another, obs, fig_sink=r.sink, n_samples=20,
                                           accept=Accept(other_observation=True))
    assert inf.results["accepted"] == ["other_observation"]
    same = copy(r.posterior)
    same.posterior = reparam.TransformedPosterior(r.posterior.latent, r.posterior.posterior.T,
                                                  truncation=None, x_obs_digest=obs.digest)
    assert orchestrator.infer_and_visualize(r.cfg, same, obs, fig_sink=r.sink, n_samples=20).results["accepted"] == []


def test_a_round_records_parent_posterior_and_observation_as_parents(tiny_run):
    from core import orchestrator
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="round_obs")
    region = orchestrator.build_truncation_region(r.posterior, obs, n_directions=1, level=0.99)
    assert region.x_obs_digest == obs.digest and region.prior_fingerprint == r.posterior.fingerprint
    # CHECKPOINTED every batch, so this round writes a REAL simulation cache: without it the kind is
    # empty for the whole module and the integrity test at the end of this file has no simulation row
    # to walk (its manifest, its prior parent, its completion).
    child = orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=r.sink, num_runs=2,
                                         hidden_features=8, num_transforms=1, stop_after_epochs=1,
                                         truncation=region, observation=obs,
                                         parent_posterior=r.posterior, name="round1",
                                         checkpoint_every=1)
    m = child.manifest
    assert m.body["amortized"] is False and child.posterior.truncation is not None
    assert m.parents["parent_posterior"] == r.posterior.id and m.parents["observation"] == obs.id
    assert m.parents["prior"] == r.prior.id and m.body["truncation"]["x_obs_digest"] == obs.digest
    sim = m.parents["simulation"]
    assert len(sim) == 12 and set(sim) <= set("0123456789abcdef"), sim
    assert r.store.get("simulation", sim).body["complete"] is True
    with pytest.raises(ValueError, match="NOT AMORTIZED"):
        r.store.load_posterior(r.cfg, child.id)
    assert r.store.get("posterior", child.id).body["training"]["tsnpe_acceptance"] is not None
    other = orchestrator.generate_observations(r.cfg, fig_sink=r.sink)   # new noise, new digest
    with pytest.raises(ValueError, match="does not match the observation"):
        orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=r.sink, num_runs=2, hidden_features=8,
                                     num_transforms=1, stop_after_epochs=1, truncation=region, observation=other,
                                     parent_posterior=r.posterior)


def test_no_literal_resource_paths_outside_config():
    """Every path under Resources/ or Artifacts/ is built from core.config. A literal anywhere else is a
    store that moves with the working directory -- what Resources/ used to do. Docstrings and
    comments are not Path() calls or `/` operands, so they do not trip this.

    It walks CODE_ROOTS rather than a hard-coded "core", so the command-line tool (core/tool/) is
    covered for free and any future top-level package is covered the moment
    test_the_source_scans_cover_every_code_directory forces it into CODE_ROOTS. It also walks
    CODE_FILES, the loose top-level *.py files (conftest.py today) that live outside every
    CODE_ROOTS directory and so would otherwise escape both this scan and that guard. The matcher
    itself is unchanged."""
    import ast
    root = Path(__file__).resolve().parents[1]
    roots = ("Resources/", "Resources\\", "Artifacts/", "Artifacts\\", "Resources", "Artifacts")

    def _lit(node):
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(roots)

    def _offenses(py):
        hits = []
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            hit = False
            if isinstance(node, ast.Call):
                fn = node.func
                fname = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                hit = fname in ("Path", "join") and node.args and _lit(node.args[0])
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                hit = _lit(node.left) or _lit(node.right)
            if hit:
                hits.append(node.lineno)
        return hits

    missing = [name for name in CODE_FILES if not (root / name).is_file()]
    assert not missing, (
        f"CODE_FILES names a file that no longer exists: {missing}. Update tests/_fixtures.CODE_FILES.")

    offenders, scanned = [], 0
    for sub in CODE_ROOTS:
        for py in sorted((root / sub).rglob("*.py")):
            if py == root / "core" / "config.py":
                continue
            scanned += 1
            offenders += [f"{py.relative_to(root)}:{ln}" for ln in _offenses(py)]
    for name in CODE_FILES:
        py = root / name
        scanned += 1
        offenders += [f"{py.relative_to(root)}:{ln}" for ln in _offenses(py)]
    assert scanned >= 20, f"the scan walked only {scanned} files -- CODE_ROOTS is {CODE_ROOTS}"
    assert not offenders, "literal store paths outside config.py:\n" + "\n".join(offenders)
    for name in ("PRIOR_PATH", "POSTERIOR_PATH", "PLOT_PATH", "CHECKPOINT_PATH", "OBSERVATION_PATH"):
        assert not hasattr(config, name), f"config.{name} still exists"


def test_the_source_scans_cover_every_code_directory():
    """CODE_ROOTS and CODE_FILES are what the two source scans walk, so a top-level package or loose
    *.py file missing from one of them is scanned by nothing. That is not hypothetical: scripts/
    carried thirteen files past the literal-path scan from piece 1 until piece 2 dissolved it, and
    the tool could just as easily have been written as a top-level package of its own. The guard is
    the closure: any new directory or file holding Python either joins CODE_ROOTS/CODE_FILES or
    fails here.

    Directories excluded by name, each for its own reason: tests/ (the suites themselves, not
    shipped code); the gitignored archive/ (not shipped code); the gitignored .claude/ (Claude
    Code's local state, which can hold other checkouts -- machine-local, not code that lives in
    .claude on purpose); and .git/, .pytest_cache/, __pycache__/, Artifacts/, Resources/ (data,
    caches, VCS or tool state, not product code). sbi-logs/ is not excluded any more: nothing writes
    it since piece 3 (V9), and tests/conftest.py::_no_sbi_logs fails the session if it reappears.
    .idea/, .superpowers/ and
    docs/ hold no *.py today so they need no explicit exclusion; they fall out of `with_py` on
    their own, and would have to be added here (or to CODE_ROOTS) the day one of them gained a
    Python file. Loose top-level files need no exclusion set: every *.py at the repo root must be
    in CODE_FILES."""
    root = Path(__file__).resolve().parents[1]
    skip = {"tests", "archive", ".claude", ".git", ".pytest_cache", "Artifacts", "Resources",
            "__pycache__"}
    with_py = {d.name for d in root.iterdir()
               if d.is_dir() and d.name not in skip and next(d.rglob("*.py"), None) is not None}
    assert with_py == set(CODE_ROOTS), (
        f"top-level directories holding Python: {sorted(with_py)}; CODE_ROOTS: {sorted(CODE_ROOTS)}. "
        f"Add the directory to tests/_fixtures.CODE_ROOTS, or exclude it here if it is not product code.")
    with_py_files = {f.name for f in root.iterdir() if f.is_file() and f.suffix == ".py"}
    assert with_py_files == set(CODE_FILES), (
        f"top-level Python files: {sorted(with_py_files)}; CODE_FILES: {sorted(CODE_FILES)}. "
        f"Add the file to tests/_fixtures.CODE_FILES, or exclude it here if it is not product code.")


def test_a_round_whose_region_names_no_observation_is_refused_before_the_spend(tiny_run):
    """A hand-built region without x_obs_digest used to train the whole round and fail only at the
    store's read-back; the refusal now comes before any simulation. The region is built through
    build_truncation_region -- matching the parent's rotation, basis, probe and prior fingerprint --
    and then stripped of its x_obs_digest, so it is the NEW refusal that fires here rather than the
    earlier rotate mismatch or check_basis, which a bare hand-built TruncationRegion(V=None,
    probe=None) would trip first against this SBITEST posterior (trained with REPARAM_ROTATE on)."""
    from core import orchestrator
    from core.SBI import pipeline as pipeline_mod
    r = tiny_run
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="no_digest_obs")
    region = orchestrator.build_truncation_region(r.posterior, obs, n_directions=1, level=0.99)
    assert region.x_obs_digest == obs.digest              # sanity: build_truncation_region set it
    region.x_obs_digest = None
    calls = []
    saved = pipeline_mod.train_nn
    pipeline_mod.train_nn = lambda *a, **k: calls.append(1)
    try:
        with pytest.raises(ValueError, match="must name the observation"):
            orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=r.sink, num_runs=2,
                                         hidden_features=8, num_transforms=1, stop_after_epochs=1,
                                         truncation=region, store=r.store)
    finally:
        pipeline_mod.train_nn = saved
    assert calls == []


def test_every_artifact_the_suite_wrote_is_complete_and_its_parents_resolve(tiny_run):
    """Runs last: every artifact the module's tests wrote into the shared tiny store is complete, every
    parent id resolves to an artifact of the right kind, and every payload still hashes to what its
    manifest recorded -- the store's contract, checked over a whole session's worth of real writes."""
    s = tiny_run.store
    for kind in st.KIND_DIRS:
        for row in s.list(kind):
            assert row.complete, (kind, row.id, row.reason)
            m = s.get(kind, row.id)
            for pkind, pid in m.parents.items():
                k = "posterior" if pkind == "parent_posterior" else pkind
                assert s.get(k, pid).id == pid, (kind, row.id, pkind, pid)
            for f, sha in m.payloads.items():
                assert prov.sha256_file(row.path / f) == sha, (kind, row.id, f)


def test_checkpoint_every_and_max_epochs_are_recorded_in_the_manifest(store, monkeypatch):
    """Both knobs are recorded, so an artifact says what cadence it was written under and what epoch
    cap it trained to. checkpoint_every=0 is checkpointing OFF: no plan.checkpoint, no simulation
    parent, and nothing under simulations/ -- which is what the whole suite now runs with."""
    from core import orchestrator
    cfg = _nad_cfg()
    cfg.reparam_rotate = False
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    captured = {}

    def fake_train_nn(plan, **kw):
        captured["plan"], captured["kw"] = plan, kw
        dp = _FakeDP()
        dp.prior = kw["prior"]
        return dp, {"training_loss": [1.0], "validation_loss": [1.0], "best_validation_loss": 1.0,
                    "epochs_trained": 1, "stop_after_epochs": 1}

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", fake_train_nn)
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data",
                        lambda plan, **kw: (torch.zeros(8, 50), torch.zeros(8, 13)))
    out = orchestrator.build_posterior(cfg, lp, None, True, fig_sink=_close,
                                       num_runs=2, run_size_cap=4, checkpoint_every=0,
                                       max_num_epochs=11, hidden_features=8, num_transforms=1,
                                       stop_after_epochs=1)
    m = store.get("posterior", out.id)
    assert captured["plan"].checkpoint is None, "checkpoint_every=0 must leave the plan uncheckpointed"
    assert captured["kw"]["max_num_epochs"] == 11
    assert m.config["checkpoint_every"] == 0 and m.config["max_num_epochs"] == 11
    assert "simulation" not in m.parents
    sims = store.kind_dir("simulation")
    assert not sims.exists() or not list(sims.iterdir()), "checkpointing off still wrote a cache"


def test_resume_is_validated_and_refused_before_any_spend(store, monkeypatch):
    """`resume` is an ARGUMENT with a LITERAL default -- 'auto', 'require' or 'never', refused
    otherwise -- rather than a None-resolved knob, because there is no config constant behind it.

    And a policy with checkpointing OFF is refused up front: with checkpoint_every=0 no cache is read
    or written, so the policy has nothing to act on, and a `--resume require` drill would otherwise
    exit 0 having tested nothing. That is exactly how the 2026-09-11 GPU run 2 passed while resuming
    nothing at all."""
    import inspect as _inspect
    from core import orchestrator
    cfg = _nad_cfg()
    cfg.reparam_rotate = False
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    spent = []
    monkeypatch.setattr(orchestrator.pipeline, "train_nn", lambda *a, **k: spent.append("train"))
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data", lambda *a, **k: spent.append("sim"))
    with pytest.raises(Refusal, match="resume policy") as e:
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4,
                                     checkpoint_every=1, resume="yes please")
    assert e.value.field == "resume" and "'yes please'" in str(e.value), str(e.value)
    with pytest.raises(Refusal, match="checkpointing on") as e:
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4,
                                     checkpoint_every=0, resume="require")
    assert e.value.field is None and "--checkpoint-every" not in str(e.value), str(e.value)
    assert spent == [], "a refused resume policy started the run anyway"
    assert _inspect.signature(orchestrator.build_posterior).parameters["resume"].default == "auto"
    # The flow and training knobs are refused with the cadence, not after every simulation: a
    # --max-epochs -1 used to simulate the whole budget and then die inside sbi, and a zero patience or
    # learning rate silently wrote an untrained posterior.
    bad = [("max_num_epochs", {"max_num_epochs": 0}), ("max_num_epochs", {"max_num_epochs": -1}),
           ("hidden_features", {"hidden_features": 0}), ("num_transforms", {"num_transforms": 0}),
           ("stop_after_epochs", {"stop_after_epochs": 0}),
           ("learning_rate", {"learning_rate": 0}), ("learning_rate", {"learning_rate": float("nan")}),
           ("checkpoint_every", {"checkpoint_every": -1})]
    for knob, kw in bad:
        kw = {"checkpoint_every": 1, **kw}
        with pytest.raises(Refusal) as e:
            orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4, **kw)
        # The sentence names the setting in neutral words ("The maximum number of epochs ..."), never by
        # its keyword; the field key is what names the knob.
        assert e.value.field == knob, (kw, e.value.field, str(e.value))
        assert spent == [], f"{kw} was refused only after the spend: {spent}"


def test_flow_knobs_are_range_checked_only_when_the_call_trains(store, monkeypatch):
    """A LOAD never uses the flow knobs, so it must not be refused over them. The Posterior tab sends one
    build_posterior call for train and load alike, and its learning-rate field accepts a negative value:
    loading an existing posterior with -0.001 in that field was refused. The same values on a call that
    TRAINS are still refused before any spend (the train side is test_resume_is_validated_...)."""
    from core import orchestrator
    cfg = _nad_cfg()
    cfg.reparam_rotate = False
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    post = _posterior_artifact(store, cfg, name="stored")
    spent = []
    monkeypatch.setattr(orchestrator.pipeline, "train_nn", lambda *a, **k: spent.append("train"))
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data", lambda *a, **k: spent.append("sim"))
    bad = {"learning_rate": -0.001, "max_num_epochs": 0, "hidden_features": 0, "num_transforms": 0,
           "stop_after_epochs": 0, "fisher_m": 0, "fisher_dz": -0.1, "fisher_points": 0}
    # Piece 3: the budget and the cadence join the exemption (spec §3.4, "the load path"). A load reads
    # neither, and the Posterior tab forwards its budget boxes on a load as well.
    budget = {"num_runs": 0, "run_size_cap": -1, "checkpoint_every": -1}
    loaded = orchestrator.build_posterior(cfg, lp, post.id, False, **bad, **budget)
    assert loaded.id == post.id and spent == []
    for knob, value in {**bad, **budget}.items():
        kw = {"num_runs": 2, "run_size_cap": 4, "checkpoint_every": 1, knob: value}
        with pytest.raises(Refusal) as e:
            orchestrator.build_posterior(cfg, lp, None, True, **kw)
        assert e.value.field == knob, (knob, e.value.field, str(e.value))
    assert spent == []


def test_the_budget_line_prints_with_default_arguments(store, monkeypatch, caplog):
    """Guardrail 6 on the command line: what this run will actually simulate, said once, whether or
    not a cap or a batch count was overridden. The two conditional announcements stay -- with the
    defaults neither of them fires, which is precisely the run whose size used to be invisible.

    Since piece 3 the line is an INFO record on core.orchestrator (V4): the tool's stdout handler
    prints it with exactly this text, where the GPU recipe reads it (spec §4.5), and the window puts
    it in the pane. So it is read off caplog, with its logger and its level."""
    from core import orchestrator
    cfg = _nad_cfg()
    cfg.reparam_rotate = False
    cfg.hw.batch_size = 4
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)

    def stop(*a, **k):
        raise RuntimeError("stop before training")

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", stop)
    caplog.clear()
    with pytest.raises(RuntimeError, match="stop before training"):
        orchestrator.build_posterior(cfg, lp, None, True)
    n = config.TRAINING_NUM_RUNS
    said = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
    assert ("core.orchestrator", "INFO", f"[budget] {n:,} batches x 4 rows = {n * 4:,} training rows") in said, said
    assert not any("capped at" in m or "COUNT overridden" in m for _, _, m in said), (
        "the default run fires neither conditional announcement -- that is why [budget] is unconditional")


def test_make_sim_config_takes_the_device_as_an_argument():
    """The command-line tool builds `hw` from --device; the GUI passes nothing and keeps
    detect_device(). A hard-coded detect_device() inside the builder made the CPU unreachable without
    mutating the config afterwards, and the device is part of the simulation identity."""
    from core import cli, registry
    from core.config import BOUNDS_PATH, VALID_LABELS, VALID_MODELS
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    bounds = str(BOUNDS_PATH / "nadrowski" / "master.txt")
    hw = config.cpu_device()
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"), bounds, hw=hw)
    assert cfg.hw is hw and cfg.hw.device.type == "cpu"
    default = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"), bounds)
    assert default.hw.device.type == config.detect_device().device.type


def test_a_near_miss_cache_is_refused_before_the_fisher(store, monkeypatch):
    """D7. A run that would start a NEW simulation cache while a committed one sits ONE identity field
    away is refused BEFORE the Fisher, before any simulation, and before the cache's own create().

    This is the 2026-09-11 GPU incident made loud: run 2 was given NUM_RUNS=2 against run 1's 4, keyed
    a new directory, recomputed the rotation and re-simulated -- silently, exit 0. The pipeline's own
    `resume='require'` refusal (pipeline.py:1367-1368) fires only AFTER a freshly computed Fisher,
    which is the most expensive thing a run does before it simulates, so it is hoisted here and the
    pipeline keeps its copy as a second line.
    """
    from core import orchestrator
    from core.artifacts.identity import SimulationIdentity
    from core.SBI import training_checkpoint as tc
    cfg = _nad_cfg()
    cfg.reparam_rotate = True
    cfg.hw.batch_size = 4
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)

    # A committed sibling that differs in EXACTLY one field: 3 batches where this run asks for 2.
    sib = SimulationIdentity.from_cfg(cfg, lp, 4, 3).to_dict()
    d = tc.resolve_dir(sib, store.kind_dir("simulation"))
    (d / "shards").mkdir(parents=True)
    torch.save({"format": tc.CHECKPOINT_FORMAT, "identity": sib, "V": None, "probe": None}, d / "header.pt")
    torch.save({"batches_done": 2, "complete": False, "rng": None}, d / "state.pt")

    def _fisher(*a, **k):
        raise AssertionError("Fisher reached")

    monkeypatch.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation", _fisher)
    monkeypatch.setattr(orchestrator.pipeline, "train_nn", lambda *a, **k: None)

    # (a) refused, naming the field and both values
    with pytest.raises(Refusal) as e:
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4, checkpoint_every=1)
    msg = str(e.value)
    assert "n_runs" in msg and "3" in msg and "2" in msg and "new_run" in msg, msg
    assert d.name in msg, "the refusal must name the cache it would abandon"
    # `new_run` stays: it is the keyword. The button and the flag come from the front-end tables.
    assert e.value.field == "new_run"
    for banned in ("Posterior tab", "TSNPE tab", "--new-run"):
        assert banned not in msg, banned

    # (b) consent overrides it, and the run proceeds as far as the Fisher
    with pytest.raises(AssertionError, match="Fisher reached"):
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4,
                                     checkpoint_every=1, new_run=True)

    # (c) resume='require' refuses before the Fisher too, and says why
    with pytest.raises(Refusal, match="no resumable cache") as e:
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4,
                                     checkpoint_every=1, resume="require")
    assert e.value.field is None, "two controls answer a require with no cache; no single field"

    # (d) with checkpointing off there is nothing to be near: no read, no write, no question
    with pytest.raises(AssertionError, match="Fisher reached"):
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4, checkpoint_every=0)

    # (e) THE DETECTOR RESOLVES run_size EXACTLY AS THE STAGE DOES: min(hw, cap). The Posterior tab
    # used `cap or _hw_batch(cfg)`, so with a cap ABOVE the hardware batch it asked about a directory
    # the run never touches -- two fields would differ and the near miss would vanish.
    rows = orchestrator.fresh_run_near_misses(cfg, lp, num_runs=2, run_size_cap=4096,
                                              checkpoint_every=1)
    assert [r["field"] for r in rows] == ["n_runs"] and rows[0]["batches"] == 2
    assert orchestrator.fresh_run_near_misses(cfg, lp, num_runs=2, run_size_cap=4,
                                              checkpoint_every=0) == [], "off means no question"


def test_the_training_preview_agrees_with_the_stage_on_width_directory_and_cadence(store, monkeypatch,
                                                                                   tmp_path):
    """V6. The budget group's lines are only as good as their agreement with the run, and they used to
    derive their answers themselves: the cadence from config.TRAINING_CHECKPOINT_EVERY (the LIVE module
    copy -- the stage reads orchestrator's import-time binding, and tests/conftest.py rebinds only that
    one), the directory from resolve_dir's process-default root (not the store the run writes to), and
    the width through a helper of the tab's own. Each is a way for "Resumes an existing checkpoint" to
    sit above a Train button that then simulates from zero.

    So the preview is checked against the STAGE ITSELF, not against a restatement of it: build_posterior
    runs as far as its cache decision and is stopped there (read_header on a resume, near_miss_siblings
    on a fresh run -- each records the directory and the identity's run_size, then raises), or, with
    checkpointing off, as far as the Fisher. A cache is laid by hand at the directory the STAGE resolved
    (as test_a_near_miss_cache_is_refused_before_the_fisher lays one), and the preview must find it."""
    from core import orchestrator
    from core.artifacts import ArtifactStore
    from core.artifacts.identity import SimulationIdentity
    from core.SBI import training_checkpoint as tc

    cfg = _nad_cfg()
    cfg.reparam_rotate = True
    cfg.hw.batch_size = 8
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)

    class _AtTheCache(Exception):
        pass

    seen = []

    def _at_resume(path):
        seen.append(("resume", Path(path), None))
        raise _AtTheCache()

    def _at_fresh(ident, root):
        seen.append(("new", tc.resolve_dir(ident, root), ident["run_size"]))
        raise _AtTheCache()

    def _fisher(*a, **k):
        raise AssertionError("Fisher reached")

    def stage(**kw):
        """build_posterior as far as its cache decision: (branch, directory, run_size), or None when it
        never looked at a cache and ran on to the Fisher (checkpointing off)."""
        seen.clear()
        with pytest.MonkeyPatch.context() as m:
            m.setattr(orchestrator.training_checkpoint, "read_header", _at_resume)
            m.setattr(orchestrator.training_checkpoint, "near_miss_siblings", _at_fresh)
            m.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation", _fisher)
            try:
                orchestrator.build_posterior(cfg, lp, None, True, **kw)
            except _AtTheCache:
                return seen[-1]
            except AssertionError as e:
                assert "Fisher reached" in str(e), e
                return None
        raise AssertionError("build_posterior returned without reaching its cache decision or the Fisher")

    # (1) CADENCE: the stage's own binding, never config's live copy, and an explicit argument wins
    monkeypatch.setattr(config, "TRAINING_CHECKPOINT_EVERY", 50)       # what the old tab line read
    assert orchestrator.TRAINING_CHECKPOINT_EVERY == 0, "the session binding (tests/conftest.py)"
    p = orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=4)
    assert (p.checkpoint, p.cadence) == ("off", 0), p
    assert stage(num_runs=2, run_size_cap=4) is None, "the preview said off, and the stage looked anyway"
    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 7)
    monkeypatch.setattr(config, "TRAINING_CHECKPOINT_EVERY", 0)
    p = orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=4)
    assert (p.checkpoint, p.cadence) == ("new", 7), p
    assert stage(num_runs=2, run_size_cap=4)[0] == "new", "the preview said on, and the stage never looked"
    p = orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=4, checkpoint_every=0)
    assert (p.checkpoint, p.cadence) == ("off", 0), p
    assert stage(num_runs=2, run_size_cap=4, checkpoint_every=0) is None

    # (2) WIDTH: the identity's run_size, cap by cap; a cap above the hardware batch is a ceiling
    dirs = {}
    for cap, want in ((0, 8), (4, 4), (16, 8)):
        p = orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=cap)
        branch, d, run_size = stage(num_runs=2, run_size_cap=cap)
        assert (p.width, p.hw_batch) == (want, 8) and run_size == p.width, (cap, p, run_size)
        assert branch == "new" and p.checkpoint == "new", (cap, branch, p)
        dirs[cap] = d
    assert dirs[0] == dirs[16] != dirs[4], dirs

    # (3) DIRECTORY: a partial cache laid where the STAGE looked is the cache the preview finds
    d = dirs[4]
    ident = SimulationIdentity.from_cfg(cfg, lp, 4, 2).to_dict()
    assert d == tc.resolve_dir(ident, store.kind_dir("simulation"))
    (d / "shards").mkdir(parents=True)
    torch.save({"format": tc.CHECKPOINT_FORMAT, "identity": ident, "V": None, "probe": None}, d / "header.pt")
    torch.save({"batches_done": 1, "complete": False, "rng": None}, d / "state.pt")
    p = orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=4)
    assert (p.checkpoint, p.batches_done, p.siblings) == ("resume_partial", 1, ""), p
    assert stage(num_runs=2, run_size_cap=4) == ("resume", d, None), "the stage resumes where the preview said"
    torch.save({"batches_done": 2, "complete": True, "rng": None}, d / "state.pt")
    p = orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=4)
    assert (p.checkpoint, p.batches_done) == ("resume_complete", 2), p

    # ...under the GIVEN store's simulations directory (the default here is `store`), not another's
    other = ArtifactStore(tmp_path / "other")
    assert orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=4, store=other).checkpoint == "new"
    assert orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=4,
                                         store=store).checkpoint == "resume_complete"

    # (4) one field away: the stage starts a new directory, and the preview names the sibling it skips
    p = orchestrator.training_preview(cfg, lp, num_runs=3, run_size_cap=4)
    branch, d3, _ = stage(num_runs=3, run_size_cap=4)
    assert branch == "new" and d3 != d, (branch, d3)
    assert p.checkpoint == "new_with_siblings" and p.batches_done == 0, p
    assert d.name in p.siblings and "differs in n_runs" in p.siblings, p.siblings


def test_the_training_preview_resolves_without_a_config_and_never_raises(store, monkeypatch):
    """The budget group is on screen at LAUNCH, before any config exists, and _sync_budget runs from
    refresh_gates(), so the preview's no-config branch is explicit and it never raises (spec §6.1).

    No config: the hardware is the config's, else hw= (the tab's memoised detect_device()), else
    detect_device(), so the width and the total line always resolve; the memory geometry is the
    hardware defaults (N_ND_MAX, 3 state variables, no steady-state cut); the checkpoint needs a config
    AND a prior, the prior being part of the identity. A real config's geometry is its own, and n_vars
    is the initial-condition width the training batch is simulated from -- the old line's
    len(cfg.inits_dict) is 0 on a config built from a bounds file with no cell, so it quoted a batch
    holding no state at all. Off CUDA the planner never splits (_max_sim_batch returns the batch
    unchanged), so there is nothing to estimate and nothing failed.

    Fail-soft: a config it cannot read, a planner budget that raises and a cache state that cannot be
    read each land in a field -- estimate_error or checkpoint_error -- and never in an exception."""
    import types
    from core import orchestrator
    from core.SBI import pipeline

    cfg = _nad_cfg()                          # built BEFORE detect_device is replaced below
    cfg.hw.batch_size = 8
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 7)
    cpu = config.cpu_device()

    # (a) launch: no config, no prior
    p = orchestrator.training_preview(None, None, num_runs=3, run_size_cap=16, hw=cpu)
    assert (p.n_runs, p.width, p.hw_batch, p.device_type, p.itemsize) == (3, 16, 64, "cpu", 4), p
    assert (p.n_fine, p.n_vars) == (config.N_ND_MAX, 3), p
    assert (p.need_elements, p.have_elements, p.estimate_error) == (None, None, None), p
    assert (p.checkpoint, p.cadence, p.checkpoint_error) == ("needs_config", 7, None), p

    # neither a config's hardware nor hw=: detect_device(); on CUDA the planner's own two numbers
    fake_cuda = types.SimpleNamespace(device=torch.device("cuda"), dtype=torch.float32, batch_size=2048)
    monkeypatch.setattr(config, "detect_device", lambda: fake_cuda)
    monkeypatch.setattr(pipeline, "sim_memory_budget_elements", lambda device, dtype: 123_456_789)
    p = orchestrator.training_preview(None, None, num_runs=2, run_size_cap=0)
    assert (p.width, p.hw_batch, p.device_type, p.itemsize) == (2048, 2048, "cuda", 4), p
    assert p.need_elements == pipeline.peak_sim_elements(2048, config.N_ND_MAX, 0, 3, 1, 1), p
    assert p.have_elements == 123_456_789 and p.estimate_error is None, p

    # (b) a config's own hardware wins over hw=, and its geometry is its own
    p = orchestrator.training_preview(cfg, None, num_runs=2, run_size_cap=0, hw=fake_cuda)
    assert (p.width, p.hw_batch, p.device_type) == (8, 8, "cpu"), p
    assert p.n_fine == min(config.N_ND_MAX, cfg.t.shape[0]), p
    assert not cfg.inits_dict and p.n_vars == orchestrator._observation_inits(cfg).shape[-1] == 3, p
    assert p.checkpoint == "needs_config", "without a prior the cache cannot be named"

    # (c) fail-soft: a config it cannot read
    p = orchestrator.training_preview(object(), object(), num_runs=2, run_size_cap=0, hw=cpu)
    assert p.width == 64 and p.estimate_error.startswith("AttributeError"), p
    assert (p.n_fine, p.n_vars, p.need_elements, p.have_elements) == (None, None, None, None), p
    assert p.checkpoint_error.startswith("AttributeError"), p
    assert (p.checkpoint, p.batches_done, p.siblings) == ("new", 0, ""), p

    # ...a planner budget that raises
    def _no_card(device, dtype):
        raise RuntimeError("no card")

    monkeypatch.setattr(pipeline, "sim_memory_budget_elements", _no_card)
    p = orchestrator.training_preview(None, None, num_runs=2, run_size_cap=512, hw=fake_cuda)
    assert (p.estimate_error, p.need_elements, p.width) == ("RuntimeError: no card", None, 512), p

    # ...a cache state that cannot be read
    def _unreadable(path):
        raise OSError("disk gone")

    monkeypatch.setattr(orchestrator.training_checkpoint, "peek", _unreadable)
    p = orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=0, store=store)
    assert (p.checkpoint, p.checkpoint_error, p.batches_done) == ("new", "OSError: disk gone", 0), p
    # checkpointing off is decided before anything is read, so there is nothing to fail
    p = orchestrator.training_preview(cfg, lp, num_runs=2, run_size_cap=0, store=store, checkpoint_every=0)
    assert (p.checkpoint, p.checkpoint_error, p.cadence) == ("off", None, 0), p


# ── the compositions: one flow under the GUI and the command line (piece 2, §2.3-§2.4) ───────────
def _sim_post(*, x_obs_digest=None, truncation=None, T=None):
    """A LoadedPosterior-shaped stand-in. simulated_inference reads only .posterior.x_obs_digest,
    .posterior.truncation and .posterior.T before it simulates, which is the whole point: every
    refusal and every judgement happens before the first SDE step."""
    from types import SimpleNamespace
    return SimpleNamespace(posterior=SimpleNamespace(x_obs_digest=x_obs_digest, truncation=truncation, T=T),
                           id="post", name="")


def _spont_cfg():
    """NADROWSKI off master_spont.txt: 12 inferred parameters, no f_scale and no Forcing section, so a
    forced cell loaded against it has values the bounds deliberately ignore."""
    from core import cli, config, registry
    from core.config import BOUNDS_PATH, VALID_LABELS, VALID_MODELS
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    cfg.hw = config.cpu_device()
    return cfg


def test_simulated_inference_warns_about_ignored_values_t_obs_and_an_out_of_distribution_truth(store, monkeypatch):
    """The three judgements the front ends used to make in three different places -- orchestrator.run,
    the GUI runner and scripts/_common, each with its own wording and two of them as bare prints --
    now come from ONE channel. That matters twice over: the GUI routes warnings.showwarning to the log
    pane at WARNING severity while a print lands at info, and the T_obs range check reached the GUI for
    the first time here."""
    from core import orchestrator
    from core.config import CELL_PATH
    cfg = _spont_cfg()
    cell = str(CELL_PATH / "nadrowski" / "master_weak.txt")      # a FORCED cell: f_scale + a drive
    made = []
    monkeypatch.setattr(orchestrator, "generate_observations", lambda c, **k: made.append(k) or "OBS")
    monkeypatch.setattr(orchestrator, "infer_and_visualize", lambda *a, **k: "INF")
    monkeypatch.setattr(orchestrator, "check_observation_in_distribution",
                        lambda c, p, f, **k: ["Ground-truth parameter 'k' = 1 lies outside the training "
                                              "prior's 1-99% range"])

    class _P:                                                    # the LoadedPrior shape
        prior = force_prior = None

    with pytest.warns(orchestrator.PreflightWarning) as rec:
        got = orchestrator.simulated_inference(cfg, _sim_post(), 0.5, cell=cell, prior=_P(), store=store)
    said = [str(w.message) for w in rec]
    assert got == ("OBS", "INF") and made == [{"fig_sink": None, "store": store}]
    assert any("below the training range minimum" in m for m in said), said
    assert any("the bounds file does not declare" in m and "f_scale" in m for m in said), said
    assert any("lies outside the training prior" in m for m in said), said
    # the range check is two-sided
    with pytest.warns(orchestrator.PreflightWarning) as rec2:
        orchestrator.simulated_inference(cfg, _sim_post(), 999.0, cell=cell, store=store)
    said2 = [str(w.message) for w in rec2]
    assert any("exceeds the training range maximum" in m for m in said2), said2
    assert any("the bounds file does not declare" in m and "f_scale" in m for m in said2), said2


def test_a_judgement_points_at_the_caller_not_at_the_run_boundary(store, monkeypatch):
    """public_entry's wrapper adds a frame between a stage and whoever called it, so a warning whose
    stacklevel was counted for the undecorated stage named core/runs.py and the wrapper's
    ``return fn(*args, **kwargs)`` as its source -- on the tool's stderr and in the window. Every such
    warning skips runs.py's frames when counting (``skip_file_prefixes``, Python 3.12), so the
    judgement points at the caller again: here, this test.

    The call is made from this file, so each PreflightWarning's filename must be this file. The AST
    half covers the stage-level sites a CPU test cannot reach cheaply: every ``warnings.warn`` with a
    ``stacklevel`` made directly in a @public_entry body, and _preflight_warn's own, passes
    ``skip_file_prefixes``."""
    import ast
    import inspect
    from core import orchestrator, runs
    import core.diagnostics.ablation as ablation
    import core.diagnostics.identifiability as identifiability
    import core.diagnostics.sbc as sbc
    from core.config import CELL_PATH
    cfg = _spont_cfg()
    cell = str(CELL_PATH / "nadrowski" / "master_weak.txt")
    monkeypatch.setattr(orchestrator, "generate_observations", lambda c, **k: "OBS")
    monkeypatch.setattr(orchestrator, "infer_and_visualize", lambda *a, **k: "INF")
    with pytest.warns(orchestrator.PreflightWarning) as rec:
        orchestrator.simulated_inference(cfg, _sim_post(), 0.5, cell=cell, store=store)
    judged = [w for w in rec if issubclass(w.category, orchestrator.PreflightWarning)]
    assert len(judged) >= 2, [str(w.message) for w in rec]           # T_obs + the ignored cell values
    for w in judged:
        assert Path(w.filename).resolve() != Path(runs.__file__).resolve(), (w.filename, w.lineno)
        assert Path(w.filename).resolve() == Path(__file__).resolve(), (str(w.message), w.filename)

    def _warns_with_stacklevel(body_owner):
        out = []
        stack = list(body_owner.body)
        while stack:                                             # the body, not nested defs
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "warn" and ast.unparse(node.func.value) == "warnings"
                    and any(k.arg == "stacklevel" for k in node.keywords)):
                out.append(node)
            stack.extend(ast.iter_child_nodes(node))
        return out

    checked = 0
    for mod in (orchestrator, sbc, identifiability, ablation):
        tree = ast.parse(inspect.getsource(mod))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            decorated = any(ast.unparse(d).endswith("public_entry") for d in fn.decorator_list)
            if not (decorated or fn.name == "_preflight_warn"):
                continue
            for call in _warns_with_stacklevel(fn):
                checked += 1
                assert any(k.arg == "skip_file_prefixes" for k in call.keywords), \
                    f"{mod.__name__}.{fn.name}:{call.lineno} counts a stacklevel through public_entry's wrapper"
    assert checked >= 6, checked            # _preflight_warn + the five stage-level sites today


def test_simulated_inference_refuses_a_non_amortized_posterior_before_it_simulates(store, monkeypatch):
    """The refusal used to fire inside infer_and_visualize, i.e. AFTER generate_observations had
    simulated and WRITTEN an observation artifact -- one orphan per refusal. And for SIMULATED
    inference it is unconditional on the digest: a cell simulated now draws new noise, so it can never
    be the observation a region was drawn around."""
    from core import orchestrator
    from core.artifacts import Accept
    from core.config import CELL_PATH
    from core.SBI import reparam, truncate
    cfg = _nad_cfg()
    cell = str(CELL_PATH / "nadrowski" / "master_weak.txt")
    made = []
    monkeypatch.setattr(orchestrator, "generate_observations", lambda c, **k: made.append(1) or "OBS")
    monkeypatch.setattr(orchestrator, "infer_and_visualize", lambda *a, **k: "INF")
    with pytest.raises(Refusal, match="NOT AMORTIZED") as excinfo:
        orchestrator.simulated_inference(cfg, _sim_post(x_obs_digest="d" * 16), 1.0, cell=cell, store=store)
    assert excinfo.value.field == "accept_other_observation"
    assert "Accept(other_observation=True)" in str(excinfo.value)
    for banned in ("Run on a different observation", "Infer tab", "--accept-other-observation"):
        assert banned not in str(excinfo.value), banned
    assert made == [], "the refused run simulated anyway"
    assert not cfg.has_ground_truth, "a refused run injected the cell's truth into the session's cfg"
    assert store.list("observation") == [], "a refused run left an orphan observation behind"

    # accepted: the run proceeds, and the REGION check (new; reachable only past this refusal) reports
    # a truth outside the box the non-amortized flow was trained on
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    T = reparam.build_inferred_bijection(cfg, log_params=[])
    far = truncate.TruncationRegion([0], [9.0], [10.0], level=0.999, n_latent=P)
    with pytest.warns(orchestrator.PreflightWarning, match="NON-AMORTIZED posterior was trained on"):
        orchestrator.simulated_inference(cfg, _sim_post(x_obs_digest="d" * 16, truncation=far, T=T), 1.0,
                                         cell=cell, store=store, accept=Accept(other_observation=True))
    assert made == [1], "the region check is a judgement, not a refusal: the run must still happen"


def test_hand_entered_values_replace_the_recorded_cell(store, monkeypatch):
    """inject_ground_truth never touches cfg.sources, so before this the provenance of a hand-entered
    inference named -- and content-hashed -- whichever cell file the session had loaded earlier.

    Since piece 3 (V1) the composition works on a private copy of the config, so the cell is dropped
    from THAT copy: the pin reads the config the stubbed generate_observations received, and the
    caller's config still names the cell it loaded."""
    from core import cli, orchestrator
    from core.config import CELL_PATH
    cfg = _nad_cfg()
    cell = str(CELL_PATH / "nadrowski" / "master_weak.txt")
    cli.load_and_validate_gt(cfg, cell)                          # the session's earlier inference
    assert cfg.sources["cell"].endswith("master_weak.txt")
    gt = (dict(cfg.inits_dict),
          {k: v for k, (v, _) in cfg.params_dict.items()},
          {k: v for k, (v, _) in cfg.rescale_params.items()},
          {k: v for k, (v, _) in cfg.force_params_dict.items()})
    handed = []                                                  # the config the composition hands on
    monkeypatch.setattr(orchestrator, "generate_observations", lambda c, **k: handed.append(c) or "OBS")
    monkeypatch.setattr(orchestrator, "infer_and_visualize", lambda *a, **k: "INF")
    orchestrator.simulated_inference(cfg, _sim_post(), 1.0, gt_values=gt, store=store)
    got = handed[-1]
    assert got is not cfg and "cell" not in got.sources and got.has_ground_truth
    assert cfg.sources["cell"].endswith("master_weak.txt"), \
        "the composition dropped the cell from the CALLER's config instead of its own copy"
    for kw in (dict(cell=cell, gt_values=gt), {}):
        with pytest.raises(Refusal, match="exactly one of") as e:
            orchestrator.simulated_inference(cfg, _sim_post(), 1.0, store=store, **kw)
        assert e.value.field is None


def test_a_taken_inference_name_is_refused_before_the_observation_is_simulated(store, monkeypatch):
    """assert_name_free used to run inside infer_and_visualize, which is after generate_observations
    has simulated and written an observation nothing will ever name."""
    from core import orchestrator
    from core.config import CELL_PATH
    cfg = _forced_cfg()
    with store.create("inference", cfg, name="taken") as w:
        w.body = {"results": {}}
    made = []
    monkeypatch.setattr(orchestrator, "generate_observations", lambda c, **k: made.append(1) or "OBS")
    monkeypatch.setattr(orchestrator, "infer_and_visualize", lambda *a, **k: "INF")
    with pytest.raises(st.StoreError, match="already exists"):
        orchestrator.simulated_inference(cfg, _sim_post(), 1.0, name="taken", store=store,
                                         cell=str(CELL_PATH / "nadrowski" / "master_weak.txt"))
    assert made == [] and store.list("observation") == []


def test_installing_an_experimental_observation_clears_a_stale_truth(store, tmp_path):
    """install means "put back THIS observation's context", and a bench recording has no truth.
    Nothing ever cleared one, so after a simulated inference the next round read the stale cell as this
    observation's -- "the loaded cell's GROUND TRUTH lies OUTSIDE the truncation region" -- or was
    silently satisfied by it, and the experimental PPC started from the stale cell's inits."""
    import numpy as np
    from matplotlib import pyplot as plt
    from core import orchestrator
    from core.SBI.observations import RecordingSet
    cfg = _forced_cfg()
    sim = orchestrator.generate_observations(cfg, fig_sink=lambda t, f: plt.close(f))
    trace = sim.obs_data[0].numpy()
    spont, forced = tmp_path / "spont.npy", tmp_path / "forced.npy"
    np.save(spont, trace)
    np.save(forced, trace)
    T_obs_s = cfg.T_obs / cfg.get_unit_conversion_factor("s")
    si = {n: (5.0 if n == "freq" else 1e-12 if n == "amp" else 0.0) for n in cfg.force_params_dict}
    rec = RecordingSet(spont=str(spont), forced=((str(forced), None),), T_obs_s=T_obs_s,
                       forcing_params_si=si)
    exp = orchestrator.build_experiment_observation(cfg, rec, fig_sink=lambda t, f: plt.close(f))
    assert cfg.has_ground_truth and "cell" in cfg.sources          # the stale truth, still there
    exp.install(cfg)
    assert not cfg.has_ground_truth, "an experimental observation installed a truth that is not its own"
    assert "cell" not in cfg.sources and not cfg.inits_dict
    assert cfg.T_obs == float(exp.manifest.body["T_obs_cell"])
    sim.install(cfg)                                               # a simulated one puts its truth back
    assert cfg.has_ground_truth


def test_installing_a_simulated_observation_puts_back_its_own_verified_cell(store):
    """install puts back THIS observation's context, and the cell it was simulated from is part of it.
    Leaving cfg.sources["cell"] alone meant a TSNPE round around observation A, run after an inference
    on cell B, recorded cell B (and B's hash) as its input while naming A as its parent. A recorded cell
    that no longer hashes to its record is dropped rather than restored, because build_posterior hashes
    sources after the Fisher and a moved cell would become a refusal after the spend."""
    from matplotlib import pyplot as plt
    from core import orchestrator
    from core.config import CELL_PATH
    cfg = _forced_cfg()
    obs = orchestrator.generate_observations(cfg, fig_sink=lambda title, fig: plt.close(fig))
    cfg.sources["cell"] = str(CELL_PATH / "nadrowski" / "master_spont.txt")   # a later inference's cell
    obs.install(cfg)
    assert cfg.sources["cell"].endswith("master_weak.txt"), cfg.sources["cell"]
    obs.manifest.body["source"]["cell"]["sha256"] = "0" * 64                 # the file no longer matches
    cfg.sources["cell"] = str(CELL_PATH / "nadrowski" / "master_spont.txt")
    obs.install(cfg)
    assert "cell" not in cfg.sources, cfg.sources


def test_an_experimental_observation_records_its_own_length_and_drive_frequencies(store, tmp_path,
                                                                                 monkeypatch):
    """The observation records the context of THE RECORDING, not whatever the session's cfg last held.
    n_obs was written from cfg whenever an earlier simulated observation had set it, so a shorter bench
    recording recorded the old length and every PPC on it failed at the band plot. A chi recording set
    never wrote the frequencies it was driven at, so the manifest recorded None (the PPC then re-derived
    probes per sample: a different experiment) or the previous cell's frequencies."""
    import numpy as np
    from matplotlib import pyplot as plt
    from core import orchestrator
    from core.SBI.observations import RecordingSet
    from core.SBI.statistics import SUMMARY_WIDTH
    closing = lambda title, fig: plt.close(fig)                   # noqa: E731

    # M1: a real forced build on a recording SHORTER than the simulated observation before it
    cfg = _forced_cfg()
    sim = orchestrator.generate_observations(cfg, fig_sink=closing)
    # V1: the length is the ARTIFACT's -- the stage resolved it on its private copy of cfg
    assert sim.manifest.body["n_obs"] == 200 and cfg.n_obs is None
    trace = sim.obs_data[0].numpy()[:150]
    spont, forced = tmp_path / "spont.npy", tmp_path / "forced.npy"
    np.save(spont, trace)
    np.save(forced, trace)
    T_obs_s = 150 * cfg.dt_exp / cfg.get_unit_conversion_factor("s")
    si = {n: (5.0 if n == "freq" else 1e-12 if n == "amp" else 0.0) for n in cfg.force_params_dict}
    rec = RecordingSet(spont=str(spont), forced=((str(forced), None),), T_obs_s=T_obs_s,
                       forcing_params_si=si)
    # a stale probe set on the session's cfg (say, from a chi session's install): a forced recording
    # has no probes, so the manifest must record None, not these
    cfg.chi_obs_freqs = torch.tensor([0.1, 0.2])
    exp = orchestrator.build_experiment_observation(cfg, rec, fig_sink=closing)
    assert exp.manifest.body["n_obs"] == 150, exp.manifest.body["n_obs"]
    assert exp.manifest.body["chi_obs_freqs"] is None, exp.manifest.body["chi_obs_freqs"]

    # M2: a chi recording set records its drive frequencies in cell units, in the recordings' order
    chi_cfg = _nad_cfg(chi_mode=True)
    width = SUMMARY_WIDTH + 1 + orchestrator.expected_forcing_dim(chi_cfg)

    def _builder(cfg_, X_spont, X_forced_list, T_obs_s_, F0_si):
        # what the real builder does to cfg, and nothing else: no lock-in runs
        cfg_.set_observation_context(T_obs_s_ * cfg_.get_unit_conversion_factor("s"), {})
        return (torch.zeros(1, width, dtype=torch.float64), torch.zeros(1, 50, dtype=cfg_.hw.dtype),
                torch.zeros(1, 50, dtype=cfg_.hw.dtype))

    monkeypatch.setattr(orchestrator, "build_experiment_obs_chi", _builder)
    passive = tmp_path / "passive.npy"
    np.save(passive, np.zeros(50, dtype=np.float32))
    driven = []
    for i in range(2):
        p = tmp_path / f"driven_{i}.npy"
        np.save(p, np.zeros(50, dtype=np.float32))
        driven.append(str(p))
    chi_rec = RecordingSet(spont=str(passive), forced=((driven[0], 5.0), (driven[1], 9.0)),
                           T_obs_s=1.0, F0_si=1.0)
    want = pytest.approx([5.0 * chi_cfg.freq_si_to_cell, 9.0 * chi_cfg.freq_si_to_cell])
    fresh = orchestrator.build_experiment_observation(chi_cfg, chi_rec, fig_sink=closing)       # (a)
    assert fresh.manifest.body["chi_obs_freqs"] == want, fresh.manifest.body["chi_obs_freqs"]
    assert fresh.manifest.body["n_obs"] == 50
    assert fresh.manifest.body["conditioning"]["chi_n_freqs"] == 2
    # V1 (spec §2.4): the stage wrote the probe count, the length and the frequencies on ITS copy. A
    # bench chi observation with two probes must leave the caller's K at config.py's, or the next
    # simulated chi inference on the same session simulates two.
    assert chi_cfg.chi_n_freqs == config.CHI_N_FREQS != 2 and chi_cfg.n_obs is None
    assert getattr(chi_cfg, "chi_obs_freqs", None) is None
    chi_cfg.chi_obs_freqs = torch.tensor([0.1, 0.2, 0.3])          # (b) a stale simulated context
    chi_cfg.n_obs = 7
    stale = orchestrator.build_experiment_observation(chi_cfg, chi_rec, fig_sink=closing)
    assert stale.manifest.body["chi_obs_freqs"] == want, stale.manifest.body["chi_obs_freqs"]
    assert stale.manifest.body["n_obs"] == 50
    assert stale.manifest.body["conditioning"]["chi_n_freqs"] == 2


def test_zero_posterior_samples_or_calibration_points_are_refused_before_any_spend(store, monkeypatch):
    """A zero sample count used to be refused by whatever broke first: `validate --posterior-samples 0`
    simulated the whole calibration set and then failed in run_sbc, and `infer --n-samples 0` simulated
    and WROTE an observation nothing would name, then failed at samples.median."""
    from types import SimpleNamespace
    from core import orchestrator
    from core.config import CELL_PATH
    from core.SBI.observations import RecordingSet
    made = []
    monkeypatch.setattr(orchestrator, "_calibration_prior", lambda c, p, q: (None, None, None))
    monkeypatch.setattr(orchestrator, "_draw_calibration_set",
                        lambda *a, **k: made.append("cal") or (torch.zeros(4, 3), torch.zeros(4, 2)))
    monkeypatch.setattr(orchestrator, "generate_observations", lambda c, **k: made.append("sim") or "OBS")
    monkeypatch.setattr(orchestrator, "build_experiment_observation",
                        lambda c, r, **k: made.append("exp") or "OBS")
    prior = SimpleNamespace(prior=None, force_prior=None, id="prior", fingerprint=None)
    cfg = _forced_cfg()
    legs = [
        ("num_posterior_samples", lambda: orchestrator.validate_calibration(
            cfg, _sim_post(), prior, store=store, num_posterior_samples=0)),
        ("n_cal", lambda: orchestrator.validate_calibration(cfg, _sim_post(), prior, store=store, n_cal=0)),
        ("n_samples", lambda: orchestrator.simulated_inference(
            cfg, _sim_post(), 1.0, store=store, n_samples=0,
            cell=str(CELL_PATH / "nadrowski" / "master_weak.txt"))),
        ("n_samples", lambda: orchestrator.experimental_inference(
            cfg, _sim_post(), RecordingSet(spont="passive.npy"), store=store, n_samples=0)),
        ("n_samples", lambda: orchestrator.infer_and_visualize(cfg, _sim_post(), None, store=store,
                                                               n_samples=0)),
    ]
    for knob, call in legs:
        with pytest.raises(Refusal) as e:
            call()
        assert e.value.field == knob and "at least 1" in str(e.value), (knob, e.value.field, str(e.value))
        assert made == [], f"the {knob}=0 refusal came after the spend: {made}"
    assert store.list("calibration") == [] and store.list("observation") == []


def test_the_compositions_forward_every_keyword_unchanged(store, monkeypatch):
    """The SECOND hop is where a composition silently drops a caller's choice. accept, the store, the
    figure sink, the name and the note all have to arrive at the stage that WRITES the artifact -- and
    n_samples must be ABSENT when the caller did not set it, so the stage's own literal default stays
    the single place that number is written down.

    The CONFIG is the one thing that does not arrive as the caller passed it (piece 3, V1): each
    composition works on a private copy and hands THAT copy to both of its stages, carrying its own
    writes (the cell's truth and path, T_obs in cell units), while the caller's config stays exactly as
    it was. Never compared with == against the caller's: the composition wrote on its copy."""
    from core import orchestrator
    from core.artifacts import Accept
    from core.config import CELL_PATH, SimConfig
    from core.SBI.observations import RecordingSet
    seen = {}
    monkeypatch.setattr(orchestrator, "generate_observations",
                        lambda c, **k: seen.update(gen=k, gen_cfg=c) or "OBS")
    monkeypatch.setattr(orchestrator, "build_experiment_observation",
                        lambda c, r, **k: seen.update(exp=k, rec=r, exp_cfg=c) or "OBS")
    monkeypatch.setattr(orchestrator, "infer_and_visualize",
                        lambda *a, **k: seen.update(inf=k, args=a) or "INF")
    cfg = _nad_cfg()
    snap = snapshot_cfg(cfg)
    acc = Accept(other_observation=True)
    sink = lambda title, fig: None                                 # noqa: E731
    orchestrator.simulated_inference(cfg, _sim_post(), 1.0, accept=acc, n_samples=7, name="n1",
                                     note="t1", fig_sink=sink, store=store,
                                     cell=str(CELL_PATH / "nadrowski" / "master_weak.txt"))
    assert seen["gen"] == {"fig_sink": sink, "store": store}
    fwd = seen["args"][0]
    assert isinstance(fwd, SimConfig) and fwd is not cfg and seen["args"][2] == "OBS"
    assert seen["gen_cfg"] is fwd, "the composition must hand its ONE copy to both of its stages"
    assert fwd.has_ground_truth and fwd.sources["cell"].endswith("master_weak.txt")
    assert fwd.T_obs == 1.0 * fwd.get_unit_conversion_factor("s")
    assert_cfg_unchanged(cfg, snap)
    assert seen["inf"] == {"name": "n1", "note": "t1", "fig_sink": sink, "store": store,
                           "accept": acc, "n_samples": 7}
    rec = RecordingSet(spont="x.npy", T_obs_s=1.0)
    orchestrator.experimental_inference(cfg, _sim_post(), rec, accept=acc, name="n2", note="t2",
                                        fig_sink=sink, store=store)
    assert seen["rec"] is rec and seen["exp"] == {"fig_sink": sink, "store": store}
    assert isinstance(seen["exp_cfg"], SimConfig) and seen["exp_cfg"] is not cfg and seen["exp_cfg"] is not fwd
    assert seen["args"][0] is seen["exp_cfg"], "the composition must hand its ONE copy to both of its stages"
    assert_cfg_unchanged(cfg, snap)
    assert seen["inf"] == {"name": "n2", "note": "t2", "fig_sink": sink, "store": store, "accept": acc}, \
        "n_samples must be ABSENT when the caller did not set it"


def test_the_config_snapshot_counts_nan_equal_to_nan_inside_a_tensor():
    """assert_cfg_unchanged's field comparison says NaN equals NaN, and a tensor field is where a NaN
    lives (a chi_obs_freqs slot, a truth vector). torch.equal says a tensor holding a NaN differs from
    its own clone, so an untouched config would read as changed. Shapes and dtypes must still match,
    and every other element must still be equal."""
    from tests._fixtures import _same_value
    nan = float("nan")
    t = torch.tensor([1.0, nan, 3.0])
    assert _same_value(t, t.clone()), "a tensor holding a NaN must equal its own clone"
    assert _same_value({"f": (t, nan)}, {"f": (t.clone(), nan)}), "and inside a container"
    assert not _same_value(t, torch.tensor([1.0, 2.0, 3.0])), "a NaN is not equal to a number"
    assert not _same_value(t, torch.tensor([1.0, nan, 4.0])), "the other elements still count"
    assert not _same_value(t, t.to(torch.float64)) and not _same_value(t, t[:2]), "dtype and shape"
    assert _same_value(torch.tensor([1, 2]), torch.tensor([1, 2]))
    assert not _same_value(torch.tensor([True]), torch.tensor([False]))


def test_copy_for_run_drops_the_caches_first_and_keeps_chi_obs_freqs():
    """V1 (spec §2.1). copy_for_run is what every public entry point does to the config it is
    handed, so it has to be cheap enough to do on every call -- and a plain deepcopy is not: with
    the 2.4M-point grid and the pint registry cached on the object it costs 19 ms and a transient
    9.6 MB, and it duplicates a registry whose quantities the original's cannot be combined with.
    So the cached properties are POPPED off a shallow copy before the deep copy, and the copy
    recomputes them lazily on the thread that uses it; `_ureg` recomputes to the same process-wide
    instance. Everything a stage may write on -- the four OrderedDicts, hw, sources, labels, the
    chi_obs_freqs tensor -- is an independent equal object on the copy, and the caller's caches
    stay where they were.

    Best of five for the timing, so a busy core cannot fail it; the budget is sixty times the
    measured cost, and a deep-copied grid overshoots it two hundredfold.

    The closure pin comes first: _CACHED must name EVERY cached_property on SimConfig, so a new
    one cannot be added and silently deep-copied on every entry."""
    import time
    from collections import OrderedDict
    from functools import cached_property

    from core import sim_config
    from core.config import SimConfig

    cached = {n for n, v in vars(SimConfig).items() if isinstance(v, cached_property)}
    assert set(sim_config._CACHED) == cached, f"_CACHED {sim_config._CACHED} vs the class's {cached}"

    cfg = _nad_cfg()
    cfg.T_obs = 1.0
    _ = cfg.t, cfg.length_unit, cfg.time_unit               # materialise the grid and the registry
    cfg.chi_obs_freqs = torch.tensor([1.0, 2.0, 3.0])
    assert all(k in cfg.__dict__ for k in ("t", "_ureg", "length_unit", "time_unit"))

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        c = cfg.copy_for_run()
        times.append(time.perf_counter() - t0)
    assert min(times) < 0.005, f"copy_for_run took {min(times) * 1e3:.2f} ms: the caches were deep-copied"

    assert isinstance(c, SimConfig) and c is not cfg
    assert not any(k in c.__dict__ for k in sim_config._CACHED), "the copy starts with no caches"
    assert all(k in cfg.__dict__ for k in ("t", "_ureg", "length_unit", "time_unit")), \
        "the caller keeps its caches: only the shallow copy was stripped"
    for name in ("inits_dict", "params_dict", "rescale_params", "force_params_dict"):
        a, b = getattr(cfg, name), getattr(c, name)
        assert isinstance(b, OrderedDict) and a == b and a is not b, name
    assert c.hw == cfg.hw and c.hw is not cfg.hw
    assert c.sources == cfg.sources and c.sources is not cfg.sources
    assert c.labels == cfg.labels and c.labels is not cfg.labels
    assert torch.equal(c.chi_obs_freqs, cfg.chi_obs_freqs) and c.chi_obs_freqs is not cfg.chi_obs_freqs
    assert c.T_obs == 1.0 and c.units_dict == cfg.units_dict and c.chi_mode == cfg.chi_mode
    assert torch.equal(c.t, cfg.t), "the grid recomputes equal on the copy"
    assert c._ureg is cfg._ureg, "the registry recomputes to the process-wide instance"
    assert c.length_unit == cfg.length_unit == "nm"

    # a write on the copy -- what a stage does -- never reaches the caller
    first = next(iter(c.params_dict))
    c.T_obs = 2.0
    c.sources["cell"] = "somewhere"
    c.params_dict[first] = (1.0, c.params_dict[first][1])
    assert cfg.T_obs == 1.0 and "cell" not in cfg.sources
    assert cfg.params_dict[first][0] is None and not cfg.has_ground_truth


def test_fdt_config_carries_sources_and_a_seed_and_copies_itself_for_a_run():
    """``public_entry`` copies the config it is handed ONLY when that config has ``copy_for_run``
    (core/runs.py:243-247, ``if hasattr(kwargs["cfg"], "copy_for_run")``). FDTConfig had none, so
    decorating run_fdt would have delivered the run log and silently NOT V1's private copy -- and
    ``cfg.omega_0 = ...`` at the top of run_fdt would have kept writing on the caller's object, which
    is the defect spec §1 names. The copy is a plain deep copy: FDTConfig carries no cached_property,
    so it needs none of SimConfig.copy_for_run's _CACHED popping, and this asserts that (a new cached
    property on FDTConfig must come with the popping, as SimConfig's did).

    ``sources``, ``seed`` and ``preset_name`` are DEFAULTED fields, because the reduction map shares
    this dataclass and is outside the programme (spec §1.3): a required field would break it at every
    construction site. Everything a run writes on -- omega_0, the four OrderedDicts, sources -- is an
    independent equal object on the copy."""
    from collections import OrderedDict
    from functools import cached_property

    from core.config import FDTConfig, cpu_device

    assert not [n for n, v in vars(FDTConfig).items() if isinstance(v, cached_property)], \
        "FDTConfig gained a cached_property: copy_for_run must pop it first, as SimConfig's does"

    cfg = FDTConfig(model="HOPF", state_dep_drift=False, inits_dict=OrderedDict(x=0.0),
                    params_dict=OrderedDict(sigma_x=(0.1, (0.0, 1.0))),
                    rescale_params=OrderedDict(), force_params_dict=OrderedDict(),
                    units_dict=("nm", "ms"), hw=cpu_device())
    assert cfg.sources == {} and cfg.seed is None, "both default, for the reduction map's sake"
    assert cfg.preset_name is None, "a single-cell run has no preset (P72); only the sweep sets it"

    cfg.sources["cell"] = "Cells/hopf/cell.txt"
    cfg.seed = 7
    cfg.preset_name = "exploratory"
    c = cfg.copy_for_run()
    assert isinstance(c, FDTConfig) and c is not cfg
    assert c.sources == cfg.sources and c.sources is not cfg.sources
    assert c.params_dict == cfg.params_dict and c.params_dict is not cfg.params_dict
    assert c.seed == 7 and c.hw is not None
    assert c.preset_name == "exploratory", "the preset's name travels with the run's copy"

    c.omega_0 = 3.5
    c.sources["cell"] = "elsewhere.txt"
    assert cfg.omega_0 is None, "the caller's resonance is untouched: this is what V1 buys run_fdt"
    assert cfg.sources["cell"] == "Cells/hopf/cell.txt"


def test_the_two_fdt_builders_fill_sources_and_the_manifest_has_an_fdt_branch():
    """Every artifact names its inputs by path and hash, and an fdt record is no exception (spec
    §3.2): the two FDT builders fill ``sources`` so provenance.inputs_from_cfg works with no new
    provenance code. make_reduction_config is UNTOUCHED (P19; spec §1.2, §1.3) -- it is the
    reduction map's builder, out of scope, and the only FDTConfig in the tree not pinned to the CPU
    (§1.3), so this also pins that it still builds a working settings object.

    config_from_cfg needed a branch or store.create("fdt", cfg, ...) would raise AttributeError on
    cfg.observation_mode, the SECOND key it reads (manifest.py:246-284). The branch is taken on the
    ABSENCE of observation_mode, and it records the settings AS GIVEN TO THE BUILDER: omega_0 is
    deliberately not in it, because the writer computes this block at create() time from the
    caller's object while the run refines the resonance on its private copy (§1.2). Every value is
    finite, which validate() requires of the whole config block (manifest.py:168)."""
    from core import cli, config, registry
    from core.artifacts import manifest as mfm
    from core.artifacts import provenance as provm

    cell = str(config.CELL_PATH / "nadrowski" / "master_weak.txt")
    cfg = cli.make_fdt_config("NADROWSKI", registry.state_dep_drift("NADROWSKI"), cell)
    assert cfg.sources == cli.cell_sources(cell, "NADROWSKI")
    assert cfg.hw.device.type == "cpu", "the FDT path stays pinned to the CPU"

    inputs = provm.inputs_from_cfg(cfg)
    assert inputs["model"] == "NADROWSKI"
    assert inputs["cell"]["path"] == "Cells/nadrowski/master_weak.txt"      # relative to Resources/
    assert inputs["bounds"]["path"] == "Bounds/nadrowski/master.txt"
    assert len(inputs["cell"]["sha256"]) == 64 and len(inputs["bounds"]["sha256"]) == 64

    block = mfm.config_from_cfg(cfg)
    assert block["model"] == "NADROWSKI" and block["n_freqs"] == 60 and block["ensemble_M"] == 256
    assert block["freq_bounds"] == [0.1, 30.0] and block["device"] == "cpu"
    assert block["seed"] is None and "omega_0" not in block, \
        "the resonance the RUN discovers belongs in body.grid, not in the config block"
    mfm._check_finite(block, "config")          # what validate() does to it on every write

    red = cli.make_reduction_config(str(config.CELL_PATH / "nadrowski" / "master_spont.txt"))
    assert red.model == "NADROWSKI" and red.params_dict, "the reduction builder still builds"
    assert red.sources == {} and red.seed is None, "make_reduction_config is left alone (P19)"


# ── V1: no public entry writes on the configuration it is handed (piece 3, spec §2.2-§2.4) ──────────
from types import SimpleNamespace


class _Injected(RuntimeError):
    """Raised by a seam AFTER a write on the stage's config: a stage failing mid-run."""


class _BodyDone(Exception):
    """Raised inside a stage's write block; _EntryWriter absorbs it, so the block ends at its first
    write and the stage returns its load_*() result as a completed run does."""


def _leak(c):
    """A stage's own writes -- the observation length, the resolved sample count, the probe frequencies
    and the cell's path -- done to whichever config the seam is handed. Done to the CALLER's object,
    any one of them is the V1 defect: the next run on that session inherits it."""
    c.T_obs = 4242.0
    c.n_obs = 4242
    c.chi_obs_freqs = torch.tensor([4.0, 2.0])
    c.sources["cell"] = "written by the stage"


def _raise_injected(*a, **k):
    raise _Injected("the stage failed after writing on its config")


def _body_done(*a, **k):
    raise _BodyDone("the write block ends here")


class _EntryWriter:
    """What _EntryStore.create hands back. Entering is free; the FIRST touch of the writer's surface
    (fig_sink, payload, parents, config, fingerprints, body) raises _BodyDone, and __exit__ absorbs
    _BodyDone and nothing else. So a stage's write block ends at its first write -- or at the seam a
    leg installs before it -- and the stage returns store.load_*(w.id): a success from the caller's
    side, with no solver, flow or figure run."""

    def __init__(self):
        object.__setattr__(self, "id", "20260916T000000")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, _BodyDone)

    def __getattr__(self, name):
        raise _BodyDone(name)

    def __setattr__(self, name, value):
        raise _BodyDone(name)


class _EntryStore:
    """The store surface the fifteen entries touch, over nothing. Every method a stage hands its
    WORKING config to (create, load_prior, load_posterior) writes on it first (_leak) and then, for
    the "boom" case, raises _Injected. A name of "taken" is refused as the real store refuses one."""

    def __init__(self, case):
        self.case = case

    def assert_name_free(self, kind, name):
        if name == "taken":                      # as the real store refuses one: field "name"
            raise st.StoreError(f"A {kind} named 'taken' already exists.", field="name")

    def _handed(self, cfg):
        _leak(cfg)
        if self.case == "boom":
            _raise_injected()

    def create(self, kind, cfg, *, name="", note=""):
        self._handed(cfg)
        return _EntryWriter()

    def load_prior(self, cfg, ref):
        self._handed(cfg)
        return SimpleNamespace(nd_prior=None, id=ref, name="")

    def load_posterior(self, cfg, ref, accept=None):
        self._handed(cfg)
        return SimpleNamespace(posterior=SimpleNamespace(truncation=None), id=ref, name="")

    def load_calibration(self, ref):
        return "LOADED"

    def load_inference(self, ref):
        return "LOADED"

    def load_diagnostic(self, ref):
        return "LOADED"

    def get(self, kind, ref):                    # channel_ablation reads the cache's manifest body
        return SimpleNamespace(body={"batches_done": 1, "identity": {"run_size": 4}})

    def path(self, kind, ref):
        return Path("unused")


def _prior_stub():
    """The LoadedPrior shape with no GMM: fails open through every fingerprint check."""
    return SimpleNamespace(prior=None, force_prior=None, id="prior", name="", fingerprint=None)


def _experimental_obs(cfg, width=7):
    """A LoadedObservation of a bench recording, as the loader returns one: install() is REAL, so it
    writes T_obs, n_obs and chi_obs_freqs, clears the truth and sets the context on the config it is
    given -- the writes infer_and_visualize and tsnpe_round make before their spend."""
    from core.artifacts import LoadedObservation
    body = {"T_obs_cell": 200.0, "n_obs": 200, "chi_obs_freqs": None, "forcing_vals": {},
            "source": {"kind": "experimental"}}
    return LoadedObservation(kind="observation", id="obs", name="", path=None,
                             manifest=SimpleNamespace(body=body), digest="a" * 16,
                             x_obs=torch.zeros(1, width), obs_data=torch.zeros(1, 200),
                             t_dim=torch.zeros(1, 200), mode=cfg.observation_mode, width=width)


# One leg per public entry: leg(case, monkeypatch, tmp_path) -> (the object the pin watches, the call).
# "success" returns; "refusal" raises a Refusal before any spend, with the field _REFUSAL_FIELDS names;
# "boom" raises _Injected after a write on the stage's working config.
def _leg_generate_observations(case, monkeypatch, tmp_path):
    from core import orchestrator
    cfg = _forced_cfg()                     # a real run: `cfg.n_obs = N_obs` is written before the solver
    if case == "boom":
        monkeypatch.setattr(orchestrator.pipeline, "gen_obs", _raise_injected)
    monkeypatch.setattr(orchestrator, "_write_observation", lambda store, c, *a, **k: (_leak(c), "OBS")[1])
    name = "taken" if case == "refusal" else ""
    return cfg, lambda: orchestrator.generate_observations(cfg, name=name, fig_sink=_close,
                                                           store=_EntryStore(case))


def _leg_build_experiment_observation(case, monkeypatch, tmp_path):
    import numpy as np
    from core import orchestrator
    from core.SBI.observations import RecordingSet
    cfg = _nad_cfg(chi_mode=True)           # K = config.CHI_N_FREQS; the bench drives THREE probes
    paths = []
    for stem in ("passive", "driven_0", "driven_1", "driven_2"):
        p = tmp_path / f"{stem}.npy"
        np.save(p, np.zeros(50, dtype=np.float32))
        paths.append(str(p))

    def _builder(c, X_spont, forced, T_obs_s, F0_si):
        c.set_observation_context(T_obs_s * c.get_unit_conversion_factor("s"), {})
        if case == "boom":
            _raise_injected()
        z = torch.zeros(1, 50, dtype=c.hw.dtype)
        return torch.zeros(1, 1, dtype=torch.float64), z, z

    monkeypatch.setattr(orchestrator, "build_experiment_obs_chi", _builder)
    monkeypatch.setattr(orchestrator, "_write_observation", lambda store, c, *a, **k: (_leak(c), "OBS")[1])
    freqs = (5.0, None, 9.0) if case == "refusal" else (5.0, 7.0, 9.0)      # D9: a probe with no frequency
    rec = RecordingSet(spont=paths[0], forced=tuple(zip(paths[1:], freqs)), T_obs_s=1.0, F0_si=1.0)
    return cfg, lambda: orchestrator.build_experiment_observation(cfg, rec, fig_sink=_close,
                                                                  store=_EntryStore(case))


def _leg_build_prior(case, monkeypatch, tmp_path):
    from core import orchestrator
    cfg = _nad_cfg()
    monkeypatch.setattr(orchestrator.visualizers, "visualize_dist", lambda *a, **k: None)
    if case == "refusal":
        return cfg, lambda: orchestrator.build_prior(cfg, None, True, num_iterations=0,
                                                     store=_EntryStore(case))
    return cfg, lambda: orchestrator.build_prior(cfg, "p1", False, fig_sink=_close, store=_EntryStore(case))


def _leg_build_posterior(case, monkeypatch, tmp_path):
    from core import orchestrator
    cfg = _nad_cfg()
    if case == "refusal":
        return cfg, lambda: orchestrator.build_posterior(cfg, _prior_stub(), None, True, num_runs=0,
                                                         store=_EntryStore(case))
    return cfg, lambda: orchestrator.build_posterior(cfg, _prior_stub(), "post1", False,
                                                     store=_EntryStore(case))


def _leg_build_truncation_region(case, monkeypatch, tmp_path):
    """No config at all (spec §2.2: it captures logs only), so this leg watches the OBSERVATION wrapper
    it is handed, and its seam checks that the decorator handed the posterior wrapper through as is."""
    from core import orchestrator
    from core.SBI import reparam
    T = reparam.build_inferred_bijection(_nad_cfg(), log_params=[])
    x_obs = torch.zeros(1, 5)
    obs = SimpleNamespace(x_obs=x_obs, digest="0" * 16 if case == "refusal" else mf.tensor_digest(x_obs),
                          id="obs", name="")
    post = SimpleNamespace(posterior=SimpleNamespace(T=T), latent=SimpleNamespace(prior=None),
                           fingerprint=None, id="post", name="")

    def _region(latent, x, **kw):
        assert latent is post.latent, "a non-SimConfig first argument must pass through the decorator"
        if case == "boom":
            _raise_injected()
        return "REGION"

    monkeypatch.setattr(orchestrator.truncate, "region_from_posterior", _region)
    return obs, lambda: orchestrator.build_truncation_region(post, obs, n_directions=1, level=0.999,
                                                             t_scale_idx=9)


def _leg_validate_calibration(case, monkeypatch, tmp_path):
    from core import orchestrator
    cfg = _nad_cfg()
    monkeypatch.setattr(orchestrator, "_calibration_prior", _body_done)
    kw = {"n_cal": 0} if case == "refusal" else {}
    return cfg, lambda: orchestrator.validate_calibration(cfg, _sim_post(), _prior_stub(), fig_sink=_close,
                                                          store=_EntryStore(case), **kw)


def _leg_infer_and_visualize(case, monkeypatch, tmp_path):
    from core import orchestrator
    cfg = _nad_cfg()
    obs = _experimental_obs(cfg)
    post = SimpleNamespace(posterior=SimpleNamespace(x_obs_digest=None, truncation=None, T=None),
                           manifest=SimpleNamespace(body={"mode": obs.mode,
                                                          "conditioning": {"width": obs.width}}),
                           id="post", name="")
    kw = {"n_samples": 0} if case == "refusal" else {}
    return cfg, lambda: orchestrator.infer_and_visualize(cfg, post, obs, fig_sink=_close,
                                                         store=_EntryStore(case), **kw)


def _leg_simulated_inference(case, monkeypatch, tmp_path):
    from core import orchestrator
    from core.config import CELL_PATH
    cfg = _nad_cfg()

    def _gen(c, **k):
        _leak(c)
        if case == "boom":
            _raise_injected()
        return "OBS"

    monkeypatch.setattr(orchestrator, "generate_observations", _gen)
    monkeypatch.setattr(orchestrator, "infer_and_visualize", lambda *a, **k: "INF")
    kw = {"n_samples": 0} if case == "refusal" else {}
    return cfg, lambda: orchestrator.simulated_inference(cfg, _sim_post(), 2.0,
                                                         cell=str(CELL_PATH / "nadrowski" / "master_weak.txt"),
                                                         store=_EntryStore(case), **kw)


def _leg_experimental_inference(case, monkeypatch, tmp_path):
    from core import orchestrator
    from core.SBI.observations import RecordingSet
    cfg = _nad_cfg()

    def _build(c, rec, **k):
        _leak(c)
        if case == "boom":
            _raise_injected()
        return "OBS"

    monkeypatch.setattr(orchestrator, "build_experiment_observation", _build)
    monkeypatch.setattr(orchestrator, "infer_and_visualize", lambda *a, **k: "INF")
    kw = {"n_samples": 0} if case == "refusal" else {}
    rec = RecordingSet(spont="x.npy", T_obs_s=1.0)
    return cfg, lambda: orchestrator.experimental_inference(cfg, _sim_post(), rec, store=_EntryStore(case),
                                                            **kw)


def _leg_tsnpe_round(case, monkeypatch, tmp_path):
    from core import orchestrator
    cfg = _nad_cfg()
    obs = _experimental_obs(cfg)

    def _child(c, *a, **k):
        _leak(c)
        if case == "boom":
            _raise_injected()
        return "CHILD"

    monkeypatch.setattr(orchestrator, "build_truncation_region", lambda *a, **k: "REGION")
    monkeypatch.setattr(orchestrator, "build_posterior", _child)
    kw = {"n_directions": 0} if case == "refusal" else {}
    return cfg, lambda: orchestrator.tsnpe_round(cfg, _sim_post(), _prior_stub(), obs,
                                                 store=_EntryStore(case), **kw)


def _leg_sbc_repeats(case, monkeypatch, tmp_path):
    from core import orchestrator
    from core.diagnostics import sbc
    cfg = _nad_cfg()
    monkeypatch.setattr(orchestrator, "_calibration_prior", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(orchestrator, "_draw_calibration_set", _body_done)
    kw = {"repeats": 0} if case == "refusal" else {}
    return cfg, lambda: sbc.sbc_repeats(cfg, _sim_post(), _prior_stub(), fig_sink=_close,
                                        store=_EntryStore(case), **kw)


def _leg_identifiability_rotation(case, monkeypatch, tmp_path):
    from core.diagnostics import identifiability
    cfg = _nad_cfg()
    names = list(cfg.params_dict) + list(cfg.rescale_params)
    V = None if case == "refusal" else torch.eye(len(names), dtype=torch.float64).tolist()
    body = {"mode": cfg.observation_mode, "conditioning": {},
            "transform": {"V": V, "param_keys": names, "fisher_eigenvalues": None}}
    post = SimpleNamespace(manifest=SimpleNamespace(body=body, config={"model": cfg.model}),
                           id="post", name="")
    return cfg, lambda: identifiability.identifiability_rotation(cfg, post, n_worst=1, top_n=2,
                                                                 store=_EntryStore(case))


def _leg_identifiability_laplace(case, monkeypatch, tmp_path):
    from core.diagnostics import identifiability
    cfg = _forced_cfg()                     # the ground truth is point 1
    monkeypatch.setattr(identifiability, "_analyze_point", _body_done)
    post = SimpleNamespace(posterior=SimpleNamespace(T=None), latent=None, id="post", name="")
    n_points = 0 if case == "refusal" else 1
    return cfg, lambda: identifiability.identifiability_laplace(cfg, post, n_points=n_points, m=4,
                                                                m_noise=16, t_obs_s=2.0,
                                                                store=_EntryStore(case))


def _leg_identifiability_jacobian(case, monkeypatch, tmp_path):
    from core.diagnostics import identifiability
    cfg = _forced_cfg()
    monkeypatch.setattr(identifiability, "_jacobian_features", _body_done)
    m_noise = 0 if case == "refusal" else 16
    return cfg, lambda: identifiability.identifiability_jacobian(cfg, m=4, m_noise=m_noise, t_obs_s=2.0,
                                                                 store=_EntryStore(case))


def _leg_channel_ablation(case, monkeypatch, tmp_path):
    from core.diagnostics import ablation
    from core.SBI import training_checkpoint
    from core.SBI.statistics import SUMMARY_WIDTH
    cfg = _nad_cfg()
    net = SimpleNamespace(input_dim=SUMMARY_WIDTH + 1, forcing_dim=0, _buffers={})
    monkeypatch.setattr(ablation, "_find_net", lambda est: net)
    monkeypatch.setattr(training_checkpoint, "load_rows", lambda *a, **k: (torch.zeros(4, 7), None))
    monkeypatch.setattr(ablation, "_sweep_channels", _body_done)
    est = SimpleNamespace(embedding_net=SimpleNamespace(eval=lambda: None))
    post = SimpleNamespace(manifest=SimpleNamespace(parents={"simulation": "d" * 16},
                                                    body={"mode": cfg.observation_mode,
                                                          "conditioning": {"width": 7}}),
                           latent=SimpleNamespace(posterior_estimator=est), id="post", name="")
    rows = 0 if case == "refusal" else 4
    return cfg, lambda: ablation.channel_ablation(cfg, post, rows=rows, n_sweep=3, store=_EntryStore(case))


class _FdtWriter:
    """The writer surface run_fdt touches, over a real temp directory but no store.

    PROGRESSIVE, like the real one (spec §2.2): __exit__ KEEPS the directory on an exception, so a leg
    that refuses or booms leaves its folder behind exactly as E2 requires. It absorbs _BodyDone and
    nothing else, so a leg can end the measurement at its first campaign and still take run_fdt's
    `return writer.store.load_fdt(writer.id)` line -- a success from the caller's side, with no
    solver, no figure and no h5py."""

    def __init__(self, tmp_path):
        self.id = "20260922T000000"
        self.dir = Path(tmp_path) / "fdt" / f"_unnamed__{self.id}"
        self.body, self.parents, self.fingerprints, self.config = {}, {}, {}, {}
        self.store = SimpleNamespace(load_fdt=lambda ref: SimpleNamespace(id=ref, body=self.body))
        self.refreshed = 0

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, _BodyDone)

    def refresh(self):
        self.refreshed += 1

    def payload(self, filename):
        return self.dir / filename

    def figure_path(self, title):
        (self.dir / "figures").mkdir(exist_ok=True)
        return self.dir / "figures" / f"{title}.png"


def _fdt_cfg_for_leg(case):
    """A real HOPF FDTConfig at the smallest size that is not "too thin to trust": n_freqs and
    ensemble_M sit exactly AT FDT_THIN_N_FREQS / FDT_THIN_ENSEMBLE_M, so the run raises no
    PreflightWarning that this pin would leak into the gate's count. The "refusal" case deletes the
    parameter observable_noise_prefactor needs, which is the run's own pre-spend refusal (spec
    §3.4)."""
    from core import cli, config as _cfgmod
    cfg = cli.make_fdt_config("HOPF", False, str(_cfgmod.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=_cfgmod.FDT_THIN_N_FREQS, ensemble_M=_cfgmod.FDT_THIN_ENSEMBLE_M)
    if case == "refusal":
        cfg.params_dict.pop("sigma_x")
    return cfg


def _leg_run_fdt(case, monkeypatch, tmp_path):
    """run_fdt's three endings. Its own write on its working config is `cfg.omega_0 = ...` (step 1 of
    its body) and `cfg.seed = ...`, both made before Campaign 1 -- so the seam that raises _Injected is
    Campaign 1 itself, and "boom" therefore lands AFTER a write, which is what the pin is for."""
    from core.FDT import fdt_pipeline
    cfg = _fdt_cfg_for_leg(case)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd",
                        _raise_injected if case == "boom" else _body_done)
    w = _FdtWriter(tmp_path)
    return cfg, lambda: fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                                             writer=w, seed=11)


def _leg_run_param_study_cli(case, monkeypatch, tmp_path):
    """The study's three endings. Its own write on its working config is ``cfg.seed = ...`` -- the one
    integer both records must agree on -- made before either sweep runs, so the seam that raises
    _Injected is the first sweep. Its pre-spend refusal is §3.4's normalisation check, applied to the
    sweep for the same reason it is applied to the single run: a cell missing n or beta would
    otherwise cost the whole first phase before anything noticed."""
    import numpy as np
    from core.FDT import cross_validation as cv
    cfg = _fdt_cfg_for_leg(case)
    cfg.model = "HOPF"

    def _sweep(c, sweep_param, sweep_grid, fixed_overrides=None, *, writer):
        if case == "boom":
            _raise_injected()
        return f"REC-{sweep_param}"

    monkeypatch.setattr(cv, "run_fdt_param_sweep", _sweep)
    writers = {"s": _FdtWriter(tmp_path / "s"), "temp": _FdtWriter(tmp_path / "t")}
    return cfg, lambda: cv.run_param_study_cli(cfg, s_grid=np.array([0.0, 0.1]),
                                               t_grid=np.array([1.0, 1.1]),
                                               writers=writers, seed=5)


def _leg_compare(case, monkeypatch, tmp_path):
    """The comparison facility's leg. A comparison takes NO configuration -- it draws saved records --
    so the watched config is one the entry never receives; the leg is here because the set is closed
    against the scan, and what it still pins is real: the three endings, and that the refusal is the
    entry's own (an arity it will not draw), carrying its field. A real store in tmp_path rather than
    _EntryStore, because the writer's body is set BEFORE the `with`, which _EntryWriter answers by
    raising _BodyDone outside any block that absorbs it."""
    from core.artifacts import ArtifactStore
    from core.FDT import compare as comparisons
    from tests._fixtures import build_fdt_record
    cfg = _nad_cfg()
    store = ArtifactStore(tmp_path / "compare_leg")
    ids = [build_fdt_record(store, name=f"leg_{i}") for i in range(2)]

    def _stub(w, records, *, sink):
        if case == "boom":
            _raise_injected()
        return {"n_records": len(records)}, []

    monkeypatch.setitem(comparisons._DRAWERS, "cells", _stub)
    refs = ids[:1] if case == "refusal" else ids
    return cfg, lambda: comparisons.compare("cells", refs, store=store)


_UNTOUCHED_LEGS = {
    "generate_observations": _leg_generate_observations,
    "build_experiment_observation": _leg_build_experiment_observation,
    "build_prior": _leg_build_prior,
    "build_posterior": _leg_build_posterior,
    "build_truncation_region": _leg_build_truncation_region,
    "validate_calibration": _leg_validate_calibration,
    "infer_and_visualize": _leg_infer_and_visualize,
    "simulated_inference": _leg_simulated_inference,
    "experimental_inference": _leg_experimental_inference,
    "tsnpe_round": _leg_tsnpe_round,
    "sbc_repeats": _leg_sbc_repeats,
    "identifiability_rotation": _leg_identifiability_rotation,
    "identifiability_laplace": _leg_identifiability_laplace,
    "identifiability_jacobian": _leg_identifiability_jacobian,
    "channel_ablation": _leg_channel_ablation,
    "run_fdt": _leg_run_fdt,
    "run_param_study_cli": _leg_run_param_study_cli,
    "compare": _leg_compare,
}

# The field each "refusal" leg's refusal carries, for EVERY entry. The leg used to accept any
# ValueError, so a leg's refusal could silently regress to a different one -- or to a bug that happens
# to raise ValueError -- and the pin would not notice. Each leg refuses ON PURPOSE, through the rule
# its own knob or input names: build_prior's and build_posterior's legs take their build branch
# deliberately, with a zero sweep-round count and a zero batch count that the knob rules refuse before
# any prior or simulation is touched.
_REFUSAL_FIELDS = {
    "generate_observations": "name",                 # _EntryStore refuses "taken" as the store does
    "build_experiment_observation": "recording_probe",   # D9: a chi probe with no frequency
    "build_prior": "num_iterations",
    "build_posterior": "num_runs",
    "build_truncation_region": "observation",        # an observation that does not hash to its digest
    "validate_calibration": "n_cal",
    "infer_and_visualize": "n_samples",
    "simulated_inference": "n_samples",
    "experimental_inference": "n_samples",
    "tsnpe_round": "n_directions",
    "sbc_repeats": "repeats",
    "identifiability_rotation": "posterior",         # a posterior with no Fisher rotation
    "identifiability_laplace": "n_points",
    "identifiability_jacobian": "m_noise",
    "channel_ablation": "rows",
    "run_fdt": "cell",                               # a cell with no FDT normalisation constant (§3.4)
    "run_param_study_cli": "cell",                   # the same check, before the first phase's spend
    "compare": "compare_records",                    # one record, for a mode that draws at least two
}


def test_the_public_entries_carry_public_entry_and_nothing_else_does():
    """V1 (spec §2.2). The private copy is kept by ONE decorator on exactly the functions named below:
    the ten stages and compositions of core/orchestrator.py, the five diagnostics, and -- since piece
    5 -- core/FDT's single-cell measurement, its two-record sweep study, and the comparison
    facility's one entry, core/FDT/compare.py. Read off the source
    (every `@public_entry` in CODE_ROOTS and CODE_FILES), not off `__wrapped__`, which any
    functools.wraps decorator sets: a public stage added without it hands its body the caller's config,
    and a helper given it (`_write_observation`, `_draw_calibration_set`, `training_identity`, ...)
    copies again inside a stage that already holds a copy. The parametrised pin below runs one leg per
    name in this set, so the two cannot drift apart."""
    import ast
    repo = Path(__file__).resolve().parents[1]
    found = set()
    paths = [p for root in CODE_ROOTS for p in (repo / root).rglob("*.py")] + [repo / f for f in CODE_FILES]
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    ast.unparse(d) == "public_entry" for d in node.decorator_list):
                found.add((path.relative_to(repo).as_posix(), node.name))
    want = {("core/orchestrator.py", n) for n in (
        "generate_observations", "build_experiment_observation", "build_prior", "build_posterior",
        "build_truncation_region", "validate_calibration", "infer_and_visualize",
        "simulated_inference", "experimental_inference", "tsnpe_round")}
    want |= {("core/diagnostics/sbc.py", "sbc_repeats"), ("core/diagnostics/ablation.py", "channel_ablation")}
    want |= {("core/diagnostics/identifiability.py", n) for n in (
        "identifiability_rotation", "identifiability_laplace", "identifiability_jacobian")}
    # Piece 5: the FDT measurement writes a record and must not write on the caller's FDTConfig --
    # `cfg.omega_0 = ...` twice in its own body is exactly the V1 defect, and copy_for_run (T7) is
    # what stops it reaching the panel's settings object. The sweep study writes `cfg.seed` -- the one
    # integer its two records must agree on (spec §4.1) -- and it too must land on a private copy.
    want |= {("core/FDT/fdt_pipeline.py", "run_fdt"),
             ("core/FDT/cross_validation.py", "run_param_study_cli")}
    # The comparison takes no configuration at all, but it writes a record, so its run's log belongs
    # in log.txt like every other entry's -- and ONE entry for all four modes (P11), so this set and
    # its two companions name it once.
    want |= {("core/FDT/compare.py", "compare")}
    assert found == want, f"missing {sorted(want - found)}; unexpected {sorted(found - want)}"
    assert set(_UNTOUCHED_LEGS) == {name for _, name in want}, sorted(set(_UNTOUCHED_LEGS) ^ {n for _, n in want})


@pytest.mark.parametrize("case", ("success", "refusal", "boom"))
@pytest.mark.parametrize("entry", sorted(_UNTOUCHED_LEGS))
def test_every_public_entry_leaves_the_callers_config_untouched(entry, case, monkeypatch, tmp_path):
    """V1 (spec §2.4, stage level). No public stage, composition or diagnostic writes on the config it
    is handed, however the call ends. The window builds ONE config at Build/Load prior and used to let
    every later run write onto it: a bench chi inference with three probes made the next simulated chi
    inference simulate three, a refused inference left a cell's truth that the next amortized training
    anchored its Fisher rotation on, and nothing ever cleared either.

    Three endings per entry, because the decorator's copy has to hold on each: "success" returns;
    "refusal" is the entry's own pre-spend refusal, a Refusal carrying the field _REFUSAL_FIELDS
    names; "boom" raises _Injected AFTER a write on the stage's working config -- the
    stage's own write where it has one before a cheap seam (generate_observations' resolved length,
    install's context, the chi builder's context), else _leak's at the first place the stage hands its
    config on (the store, a composed stage's stub). Every leg is stubbed below its first write, so the
    cases cost about two seconds, nearly all of it generate_observations' one real solve.

    The window-level pin (a dispatched simulated inference leaves session.cfg equal to its snapshot)
    needs the real-session fixture and lands with Task 15."""
    watched, call = _UNTOUCHED_LEGS[entry](case, monkeypatch, tmp_path)
    snap = snapshot_cfg(watched)
    if case == "success":
        assert call() is not None
    elif case == "refusal":
        # A Refusal carrying the field the leg's own rule names -- not just some ValueError or other
        with pytest.raises(Refusal) as e:
            call()
        assert e.value.field == _REFUSAL_FIELDS[entry], (entry, e.value.field, str(e.value))
    else:
        with pytest.raises(_Injected):
            call()
    assert_cfg_unchanged(watched, snap)


def test_a_bad_or_taken_name_is_refused_with_the_name_field(store):
    """Spec section 3.6: ``assert_name_free`` raises ``Refusal(field="name")`` -- still a StoreError,
    so every ``except StoreError`` in the tree holds, but now carrying the key both front ends map
    to their own control: the Save box (core/gui/fields.py) and --name (core/tool/fields.py). The
    message itself names NO control; the tables do, so a renamed box cannot go stale here.
    ``rename`` shares the rule, so its two refusals carry the same field.
    """
    from core.artifacts import store as st
    from core.refusals import Refusal
    cfg = _nad_cfg()
    _prior_artifact(store, cfg, name="taken")
    for bad, why in (("taken", "already exists"), ("no spaces", "bad artifact name"),
                     ("20260910T120000", "shaped like an artifact id")):
        with pytest.raises(st.StoreError, match=why) as e:
            store.assert_name_free("prior", bad)
        assert isinstance(e.value, Refusal) and e.value.field == "name", (bad, e.value.field)
        for control in ("Save box", "--name", "GUI", "command line"):
            assert control not in str(e.value), (bad, control)
    with pytest.raises(st.StoreError, match="already exists") as e:
        store.create("prior", cfg, name="taken")
    assert e.value.field == "name", "create checks the same rule and must carry the same field"
    other = _prior_artifact(store, cfg, name="other")
    for bad in ("taken", ""):
        with pytest.raises(st.StoreError) as e:
            store.rename("prior", other.id, bad)
        assert e.value.field == "name", (bad, e.value.field)
    assert store.assert_name_free("prior", "") is None, "unnamed is always free"


def test_the_loaders_mismatch_refusals_name_their_artifact_field(store):
    """Spec section 3.3: "the store's load mismatches ... become Refusals with a field key where one
    control answers them". A prior, posterior or observation that does not belong to this
    configuration is answered by picking another one, so the refusal names that picker --
    ``prior``, ``posterior``, ``observation`` -- and the front-end tables render the sentence. The
    messages are unchanged (the mismatch-class tests above pin their words); only the type and the
    key are new. The observation is hand-written: its mode check fires before the payload is read,
    so no simulation is needed to provoke it, and it is written in one mode and loaded in the other
    exactly as test_generate_observations_writes_an_artifact_that_reinstalls_its_context does.
    """
    from core.refusals import Refusal
    cfg = _nad_cfg(chi_mode=True)
    _prior_artifact(store, cfg, name="hopf_prior", model="HOPF")
    with pytest.raises(Refusal, match="the model") as e:
        store.load_prior(cfg, "hopf_prior")
    assert e.value.field == "prior"
    _posterior_artifact(store, cfg, name="hopf_post", over={("config", "model"): "HOPF"})
    with pytest.raises(Refusal, match="trained for model") as e:
        store.load_posterior(cfg, "hopf_post")
    assert e.value.field == "posterior"
    plain = _nad_cfg()                                       # master.txt declares a drive -> forced mode
    with store.create("observation", plain, name="forced_obs") as w:
        w.body = {"mode": plain.observation_mode, "conditioning": mf.conditioning_block(plain),
                  "x_obs_digest": "d" * 16, "T_obs_cell": 1.0, "n_obs": 1, "forcing_vals": {},
                  "chi_obs_freqs": None, "source": {"kind": "simulated"}}
    with pytest.raises(Refusal, match="mode") as e:
        store.load_observation(cfg, "forced_obs")
    assert e.value.field == "observation"


def test_the_experimental_builders_refuse_as_refusals_before_any_lock_in():
    """The builders' own refusals (spec §3.6): a Refusal each, with the key of the one control that
    answers it -- the drive box for a drive value that was not given (a KeyError until now, which the
    tool printed as a crash with a traceback), the probe count for a chi set outside 1..chi_k_pad --
    and None for a length mismatch, which no single control fixes. All three fire before any lock-in
    or summary statistic is computed, on tensors a few samples long."""
    from core.SBI import observations as obsm
    cfg = _nad_cfg()                                             # master.txt: a drive of amp/freq/phase
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs_s = 8 * cfg.dt_exp / s_to_cell                         # exactly 8 frames: no length warning
    si = {n: (5.0 if n == "freq" else 1e-12 if n == "amp" else 0.0) for n in cfg.force_params_dict}
    with pytest.raises(Refusal, match="same length") as e:
        obsm.build_experiment_obs(cfg, torch.zeros(8), torch.zeros(9), T_obs_s, si)
    assert e.value.field is None
    with pytest.raises(Refusal, match="'freq'") as e:
        obsm.build_experiment_obs(cfg, torch.zeros(8), torch.zeros(8), T_obs_s,
                                  {k: v for k, v in si.items() if k != "freq"})
    assert e.value.field == "drive_frequency"
    chi_cfg = _nad_cfg(chi_mode=True)
    with pytest.raises(Refusal, match="forced recordings") as e:
        obsm.build_experiment_obs_chi(chi_cfg, torch.randn(512), [], 512 * cfg.dt_exp / s_to_cell, 1e-12)
    assert e.value.field == "recording_probe"


def test_fisher_settings_are_recorded_only_when_the_rotation_ran(store, monkeypatch):
    """V7 (spec §6.2). A posterior's record names the Fisher ensemble, step and operating-point count
    only when the rotation RAN in this process; otherwise all three are None. Until piece 3 the
    manifest re-resolved them from the arguments at the write, so an unrotated run, a resumed run (which
    reuses the checkpoint's V and never calls the Fisher) and a truncated round (which reuses the
    region's V, guardrail 7) all recorded the settings of a Fisher nobody computed -- provenance that
    reads as a measurement and is not one.

    Also pins that the resolution moved UP: build_posterior resolves the three at entry and hands the
    Fisher the numbers, so decorrelate's `m or REPARAM_FISHER_M` can no longer turn a 0 into the
    default out of sight, and what is recorded is exactly what the Fisher was called with."""
    from core import orchestrator
    from core.artifacts.identity import SimulationIdentity
    from core.SBI import reparam, truncate
    from core.SBI import training_checkpoint as tc
    from core.SBI.run_guards import _log_params_for
    cfg = _nad_cfg()
    cfg.hw.batch_size = 4
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    P = len(cfg.params_dict) + len(cfg.rescale_params)

    def fake_train_nn(plan, **kw):
        dp = _FakeDP()
        dp.prior = kw["prior"]                    # the SBIPriorWrapper build_posterior passed in
        return dp, {"training_loss": [1.0], "validation_loss": [1.0], "best_validation_loss": 1.0,
                    "epochs_trained": 1, "stop_after_epochs": 1}

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", fake_train_nn)
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data",
                        lambda plan, **kw: (torch.zeros(8, 50), torch.zeros(8, 13)))
    budget = dict(num_runs=2, run_size_cap=4, hidden_features=8, num_transforms=1, stop_after_epochs=1,
                  fig_sink=_close)

    def recorded(out):
        m = store.get("posterior", out.id)
        return (m.config["fisher_m"], m.config["fisher_dz"], m.config["fisher_points"]), m

    # (a) the rotation RAN: the settings it ran with are recorded, passed or defaulted, and the Fisher
    # itself received the resolved numbers -- never a None for decorrelate to re-resolve
    calls = []
    Q, _ = torch.linalg.qr(torch.randn(P, P))

    def fisher(*a, **k):
        calls.append((k["m"], k["dz"], k["n_points"]))
        return Q, torch.arange(P, 0, -1).float()

    monkeypatch.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation", fisher)
    cfg.reparam_rotate = True
    got, _m = recorded(orchestrator.build_posterior(cfg, lp, None, True, checkpoint_every=0, fisher_m=7,
                                                    fisher_dz=0.3, fisher_points=2, **budget))
    assert calls == [(7, 0.3, 2)] and got == (7, 0.3, 2), (calls, got)
    defaults = (config.REPARAM_FISHER_M, config.REPARAM_FISHER_DZ, config.REPARAM_FISHER_POINTS)
    got, _m = recorded(orchestrator.build_posterior(cfg, lp, None, True, checkpoint_every=0, **budget))
    assert calls[-1] == defaults, f"the Fisher must receive the resolved settings, not None: {calls[-1]}"
    assert got == defaults, got

    # (b) UNROTATED: no Fisher, so nothing recorded -- even with a Fisher knob passed
    cfg.reparam_rotate = False
    got, m = recorded(orchestrator.build_posterior(cfg, lp, None, True, checkpoint_every=0, fisher_m=7,
                                                   **budget))
    assert got == (None, None, None) and m.body["transform"]["fisher_eigenvalues"] is None, (got, m.config)

    # (c) RESUMED: a committed cache of this run's own identity stores V, so the stage reuses it and
    # never calls the Fisher; the checkpoint header carries no m, dz or points to record instead
    cfg.reparam_rotate = True

    def fisher_must_not_run(*a, **k):
        raise AssertionError("a resumed run recomputed the Fisher")

    monkeypatch.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation", fisher_must_not_run)
    ident = SimulationIdentity.from_cfg(cfg, lp, 4, 2).to_dict()
    d = tc.resolve_dir(ident, store.kind_dir("simulation"))
    tc.create(d, ident, schedule_t_scales=torch.zeros(2), schedule_Ts=torch.zeros(2), inits=torch.zeros(4, 3),
              V=Q, probe=torch.zeros(0), run_size=4, n_runs=2)
    torch.save({"batches_done": 1, "complete": False, "rng": None}, d / "state.pt")
    got, m = recorded(orchestrator.build_posterior(cfg, lp, None, True, checkpoint_every=1, fisher_m=7,
                                                   **budget))
    assert m.body["training"]["resumed_from_batch"] == 1 and m.parents["simulation"] == d.name, m.body
    assert got == (None, None, None) and m.body["transform"]["fisher_eigenvalues"] is None, (got, m.config)

    # (d) TRUNCATED: the round trains in the region's basis and the Fisher is never reached
    cfg.reparam_rotate = False
    T = reparam.build_inferred_bijection(cfg, log_params=_log_params_for(cfg))
    region = truncate.TruncationRegion([0], [-50.0], [50.0], n_latent=P, V=None,
                                       probe=tc.bijection_probe(T, P, device=cfg.hw.device),
                                       x_obs_digest="d" * 16)
    got, m = recorded(orchestrator.build_posterior(cfg, lp, None, True, checkpoint_every=0, truncation=region,
                                                   fisher_m=7, **budget))
    assert m.body["amortized"] is False, m.body["amortized"]
    assert got == (None, None, None) and m.body["transform"]["fisher_eigenvalues"] is None, (got, m.config)
    assert len(calls) == 2, "only the two rotated fresh runs may have computed a Fisher"


def test_cal_n_scales_and_the_fisher_knobs_are_refused_not_clamped(store, monkeypatch):
    """Spec §3.3: the calibration operating-point count and the Fisher rotation's three knobs are
    REFUSED at stage entry, never clamped or defaulted below it. Each was a silent substitution:
    gen_cal_data clamped cal_n_scales to max(1, ...), and that count is t_scale's effective sample size
    (trap X5), so a 0 ran a different measurement than the one asked for; decorrelate's `m or
    REPARAM_FISHER_M` and `n_points or ...` turned a 0 into the default, and a negative dz went straight
    into the central difference; construct_prior's max(2, min_cluster_size) turned a 1 into a 2, and
    that count decides how many modes the prior has. Every refusal lands before the draw, the Fisher
    or the sweep it protects, with the field key the front ends map to their control."""
    from types import SimpleNamespace
    from core import orchestrator
    from core.SBI import analysis

    def reached(what):
        def _stub(*a, **k):
            raise AssertionError(f"{what} reached")
        return _stub

    # validate_calibration: before the calibration prior is even built, let alone the set drawn
    monkeypatch.setattr(orchestrator, "_calibration_prior", reached("the calibration draw"))
    monkeypatch.setattr(orchestrator, "_draw_calibration_set", reached("the calibration draw"))
    prior = SimpleNamespace(prior=None, force_prior=None, id="prior", fingerprint=None)
    for bad in (0, -3):
        with pytest.raises(Refusal) as e:
            orchestrator.validate_calibration(_forced_cfg(), _sim_post(), prior, store=store, n_cal=10,
                                              cal_n_scales=bad)
        assert e.value.field == "cal_n_scales", (bad, e.value.field, str(e.value))
        assert f"must be at least 1; got {bad} (default {config.CAL_N_SCALES})" in str(e.value), str(e.value)
    assert store.list("calibration") == []

    # ... and gen_cal_data itself, for the callers that do not go through validate_calibration
    # (core.diagnostics.sbc_repeats): refused before the first calibration batch is simulated
    monkeypatch.setattr(analysis.pipeline, "gen_training_data", reached("the calibration simulation"))
    with pytest.raises(Refusal) as e:
        analysis.gen_cal_data(model="NADROWSKI", prior=None, forcing_prior=None, t=torch.zeros(4),
                              steady_idx=0, dt_nd_min=1.0, n_cal=10, nd_dim=1, forcing_idx={},
                              rescale_idx={}, cal_n_scales=0)
    assert e.value.field == "cal_n_scales", str(e.value)

    # build_posterior: the Fisher knobs, refused before the Fisher (which would run: rotation ON)
    cfg = _nad_cfg()
    cfg.reparam_rotate = True
    cfg.hw.batch_size = 4
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    monkeypatch.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation", reached("the Fisher"))
    monkeypatch.setattr(orchestrator.pipeline, "gen_training_data", reached("the training simulation"))
    monkeypatch.setattr(orchestrator.pipeline, "train_nn", reached("training"))
    for knob, value, words in (("fisher_m", 0, "must be at least 1; got 0"),
                               ("fisher_m", -4, "must be at least 1; got -4"),
                               ("fisher_dz", 0.0, "must be greater than 0; got 0"),
                               ("fisher_dz", -0.1, "must be greater than 0; got -0.1"),
                               ("fisher_dz", float("nan"), "must be greater than 0; got nan"),
                               ("fisher_points", 0, "must be at least 1; got 0")):
        with pytest.raises(Refusal) as e:
            orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4, checkpoint_every=0,
                                         **{knob: value})
        assert e.value.field == knob and words in str(e.value), (knob, value, str(e.value))
    assert store.list("posterior") == []

    # build_prior: min_cluster_size=1 is refused before the sweep, not clamped up to 2
    monkeypatch.setattr(orchestrator.pipeline, "gen_prior", reached("the stability sweep"))
    with pytest.raises(Refusal) as e:
        orchestrator.build_prior(cfg, None, True, fig_sink=_close, min_cluster_size=1)
    assert e.value.field == "min_cluster_size", str(e.value)
    assert "must be at least 2; got 1 (default 50)" in str(e.value), str(e.value)


def test_build_prior_resolves_its_knobs_refuses_them_before_the_sweep_and_records_what_it_used(store, monkeypatch):
    """Spec §3.4, build_prior. The seven sweep and clustering knobs are resolved AT THE STAGE (None ->
    the config constant) and refused there, before cfg.t is sliced and the ~9-minute sweep starts. Only
    num_iterations was checked before; max_sets, walk_step, min_cluster_size and min_samples travelled
    as None into gen_prior, which resolved and clamped them out of sight while the prior's manifest
    recorded None. So: every out-of-rule value is refused with its field; gen_prior receives numbers,
    never None; the manifest's knobs are the numbers the sweep used; and a LOAD, which reads none of the
    seven, is never refused over them (spec §1.2, "a stage with a load branch")."""
    from core import orchestrator
    cfg = _nad_cfg()
    swept = []

    def stub_gen_prior(model, t, global_batch_size, local_batch_size, segs, prior_bounds, **kw):
        swept.append(dict(kw, global_batch_size=global_batch_size))
        return _gmm_in_box([float(b[0]) for b in prior_bounds], [float(b[1]) for b in prior_bounds],
                           kw.get("log_mask"))

    monkeypatch.setattr(orchestrator.pipeline, "gen_prior", stub_gen_prior)

    # (a) refused, each with its field and its rule, before the sweep and before any write
    for knob, value, words in (("num_iterations", 0, "must be at least 1; got 0"),
                               ("sweep_batch", -1, "must be at least 0; got -1"),
                               ("max_sets", 0, "must be at least 1; got 0"),
                               ("walk_step", 0.0, "must be greater than 0; got 0"),
                               ("walk_step", -0.01, "must be greater than 0; got -0.01"),
                               ("stability_units", float("nan"), "must be greater than 0; got nan"),
                               ("min_samples", 0, "must be at least 1; got 0")):
        with pytest.raises(Refusal) as e:
            orchestrator.build_prior(cfg, None, True, fig_sink=_close, **{knob: value})
        assert e.value.field == knob and words in str(e.value), (knob, value, str(e.value))
    assert swept == [] and store.list("prior") == []

    # (b) nothing passed but the round count: gen_prior gets the config constants as numbers, and the
    # manifest records those same numbers (sweep_batch 0 = follow the hardware batch, recorded resolved)
    lp = orchestrator.build_prior(cfg, None, True, fig_sink=_close, num_iterations=1)
    kw = swept[-1]
    assert (kw["n_max"], kw["step"], kw["min_cluster_size"], kw["min_samples"]) == (
        config.PRIOR_SWEEP_MAX_SETS, config.PRIOR_SWEEP_STEP, config.PRIOR_CLUSTER_MIN_SIZE,
        config.PRIOR_CLUSTER_MIN_SAMPLES), kw
    assert kw["num_iterations"] == 1 and kw["global_batch_size"] == cfg.hw.batch_size, kw
    knobs = lp.manifest.config
    assert (knobs["max_sets"], knobs["walk_step"], knobs["stability_units"], knobs["min_cluster_size"],
            knobs["min_samples"], knobs["sweep_batch"]) == (
        config.PRIOR_SWEEP_MAX_SETS, config.PRIOR_SWEEP_STEP, config.STABILITY_SWEEP_ND_UNITS,
        config.PRIOR_CLUSTER_MIN_SIZE, config.PRIOR_CLUSTER_MIN_SAMPLES, cfg.hw.batch_size), knobs
    assert lp.manifest.body["sweep"]["min_cluster_size"] == config.PRIOR_CLUSTER_MIN_SIZE

    # (c) passed values arrive as given, at the rules' boundaries: a 2-point cluster floor, one sample
    orchestrator.build_prior(cfg, None, True, fig_sink=_close, num_iterations=1, sweep_batch=6, max_sets=40,
                             walk_step=0.02, stability_units=250, min_cluster_size=2, min_samples=1)
    kw = swept[-1]
    assert (kw["global_batch_size"], kw["n_max"], kw["step"], kw["min_cluster_size"], kw["min_samples"]) == \
        (6, 40, 0.02, 2, 1), kw

    # (d) a LOAD reads none of the seven, so out-of-rule values do not stop it and nothing is swept
    n = len(swept)
    again = orchestrator.build_prior(cfg, lp.id, False, fig_sink=_close, min_cluster_size=1, max_sets=0,
                                     walk_step=-1.0, num_iterations=0)
    assert again.id == lp.id and len(swept) == n


def test_the_compositions_refuse_t_obs_at_or_below_zero_before_any_spend(store, monkeypatch):
    """V2 at the compositions (spec §3.3, §3.4). A blank observation length reached the simulated path
    as 0.0 from the window, simulated, WROTE the observation and then died in math.log(0.0) inside the
    conditioning row -- after the spend, with an orphan left behind; the experimental path had no
    check at all. Both compositions now refuse a length at or below 0, not a number, or blank, with
    the field key, in the order the pins rely on -- the sample count, then the length, then the name,
    then the cell file -- and before any cell is parsed, any recording is built or anything simulated.
    Outside the training range stays a judgement: the experimental path now gives the same
    PreflightWarning the simulated one always did."""
    from core import cli, orchestrator
    from core.config import CELL_PATH, T_MIN_EXP_S
    from core.SBI.observations import RecordingSet
    cell = str(CELL_PATH / "nadrowski" / "master_weak.txt")
    spent = []
    # cfg BEFORE the monkeypatches: _forced_cfg() calls the real cli.load_and_validate_gt to install its
    # own ground truth, and that setup call must not count as spend from the compositions under test.
    cfg = _forced_cfg()
    monkeypatch.setattr(cli, "load_and_validate_gt", lambda c, p: spent.append("parse") or [])
    monkeypatch.setattr(orchestrator, "generate_observations", lambda c, **k: spent.append("sim") or "OBS")
    monkeypatch.setattr(orchestrator, "build_experiment_observation",
                        lambda c, r, **k: spent.append("exp") or "OBS")
    monkeypatch.setattr(orchestrator, "infer_and_visualize", lambda *a, **k: spent.append("inf") or "INF")

    def sim(T_obs_s, **kw):
        return orchestrator.simulated_inference(cfg, _sim_post(), T_obs_s, store=store, **{"cell": cell, **kw})

    def exp(T_obs_s, **kw):
        return orchestrator.experimental_inference(
            cfg, _sim_post(), RecordingSet(spont="passive.npy", T_obs_s=T_obs_s), store=store, **kw)

    # (a) at or below zero, not a number, or blank: refused on both paths, naming the length
    for bad in (0.0, -2.5, float("nan"), None):
        for run in (sim, exp):
            with pytest.raises(Refusal) as e:
                run(bad)
            assert e.value.field == "t_obs", (run.__name__, bad, e.value.field, str(e.value))
            assert ("is blank" if bad is None else "must be greater than 0") in str(e.value), str(e.value)
    assert spent == [], f"a refused length reached the spend: {spent}"

    # (b) the order: the sample count before the length, the length before the name, the name before
    # the cell file
    with store.create("inference", cfg, name="taken") as w:
        w.body = {"results": {}}
    for run in (sim, exp):
        with pytest.raises(Refusal) as e:
            run(0.0, n_samples=0, name="taken")
        assert e.value.field == "n_samples", str(e.value)
        with pytest.raises(Refusal) as e:
            run(0.0, name="taken")
        assert e.value.field == "t_obs", str(e.value)
        with pytest.raises(st.StoreError) as e:
            run(1.0, name="taken")
        assert e.value.field == "name", str(e.value)
    missing = cell + ".nope"
    with pytest.raises(st.StoreError):
        sim(1.0, name="taken", cell=missing)
    for bad_cell, words in ((missing, "was not found"), ("", "is blank")):
        with pytest.raises(Refusal) as e:
            sim(1.0, cell=bad_cell)
        assert e.value.field == "cell" and words in str(e.value), str(e.value)
    assert spent == [], f"a refusal came after the spend: {spent}"

    # (c) positive but below the training range: a judgement on BOTH paths, and the run proceeds
    for run, spend in ((sim, ["parse", "sim", "inf"]), (exp, ["exp", "inf"])):
        spent.clear()
        with pytest.warns(orchestrator.PreflightWarning, match="below the training range minimum"):
            run(T_MIN_EXP_S / 2)
        assert spent == spend, (run.__name__, spent)


def test_a_missing_or_blank_recording_is_refused_with_its_roles_field_before_anything_is_read(store, tmp_path,
                                                                                               monkeypatch):
    """Spec §3.3: every recording is "given, and the file exists", refused as a Refusal whose field is the
    recording's ROLE -- the passive recording, forced mode's one driven recording, or a chi probe's
    recording -- so the window's yellow box can say which box to fix and the tool which flag. It was a
    FileNotFoundError ("the spont recording was not found: ''") that both front ends showed as a crash.
    Blank and missing are told apart, and both land before any recording is read or any observation is
    written."""
    import numpy as np
    from core import orchestrator
    from core.Helpers import file_manager
    from core.SBI.observations import RecordingSet
    read = []

    def _record_load(path, dtype=None):
        read.append(str(path))
        return torch.zeros(200)

    monkeypatch.setattr(file_manager, "load_experimental_data", _record_load)
    present = tmp_path / "present.npy"
    np.save(present, np.zeros(200, dtype=np.float32))
    missing = str(tmp_path / "missing.npy")
    forced_cfg, chi_cfg = _nad_cfg(), _nad_cfg(chi_mode=True)      # master.txt declares a drive: forced
    si = {n: (5.0 if n == "freq" else 1e-12 if n == "amp" else 0.0) for n in forced_cfg.force_params_dict}
    legs = (
        (forced_cfg, dict(spont=missing, forced=((str(present), None),), forcing_params_si=si),
         "recording_spont", "was not found"),
        (forced_cfg, dict(spont="", forced=((str(present), None),), forcing_params_si=si),
         "recording_spont", "is blank"),
        (forced_cfg, dict(spont=str(present), forced=((missing, None),), forcing_params_si=si),
         "recording_forced", "was not found"),
        (chi_cfg, dict(spont=str(present), forced=((str(present), 12.5), (missing, 25.0)), F0_si=1.0),
         "recording_probe", "was not found"),
    )
    for cfg, rec, field, words in legs:
        with pytest.raises(Refusal) as e:
            orchestrator.build_experiment_observation(cfg, RecordingSet(T_obs_s=1.0, **rec), fig_sink=_close)
        assert e.value.field == field and words in str(e.value), (field, str(e.value))
        assert words == "is blank" or "missing.npy" in str(e.value), str(e.value)
        assert read == [], f"a recording was read before the refusal: {read}"
    assert not list(store.kind_dir("observation").glob("*")), "a refused set wrote an observation"


def test_the_driven_builder_refuses_a_zero_drive_and_the_chi_builder_a_zero_amplitude(monkeypatch):
    """Spec §3.3, the driven bench branch: the drive amplitude and frequency are "finite, > 0" (a zero
    drive is the passive branch's job, and 0 Hz used to reach the lock-in), the phase is "finite", and
    the chi builder's physical amplitude is "finite, > 0" (a blank box arrived as 0.0 and divided every
    lock-in by zero, inside the worker). Refused on the SI values as given, before any unit conversion,
    peak search, lock-in or summary statistic, each with its field key. A drive value that is absent
    altogether is Task 6's refusal (test_the_experimental_builders_refuse_as_refusals_before_any_lock_in)."""
    from core.SBI import observations as obsm

    def reached(what):
        def _stub(*a, **k):
            raise AssertionError(f"{what} reached")
        return _stub

    monkeypatch.setattr(obsm.pipeline, "gen_stats", reached("the summary statistics"))
    monkeypatch.setattr(obsm.chi, "peak_freq", reached("the passive peak search"))
    cfg = _nad_cfg()
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs_s = 8 * cfg.dt_exp / s_to_cell                         # exactly 8 frames: no length warning
    good = {n: (5.0 if n == "freq" else 1e-12 if n == "amp" else 0.0) for n in cfg.force_params_dict}
    assert {"amp", "freq", "phase"} <= set(good), "master.txt's drive is amp/freq/phase"
    for name, value, field, words in (("amp", 0.0, "drive_amplitude", "must be greater than 0; got 0"),
                                      ("amp", -1e-12, "drive_amplitude", "must be greater than 0; got -1e-12"),
                                      ("amp", None, "drive_amplitude", "is blank"),
                                      ("freq", 0.0, "drive_frequency", "must be greater than 0; got 0"),
                                      ("freq", float("inf"), "drive_frequency", "must be greater than 0; got inf"),
                                      ("phase", float("nan"), "drive_phase", "must be a finite number; got nan"),
                                      ("phase", None, "drive_phase", "is blank")):
        with pytest.raises(Refusal) as e:
            obsm.build_experiment_obs(cfg, torch.zeros(8), torch.zeros(8), T_obs_s, dict(good, **{name: value}))
        assert e.value.field == field and words in str(e.value), (name, value, str(e.value))
    chi_cfg = _nad_cfg(chi_mode=True)
    for F0, words in ((0.0, "must be greater than 0; got 0"), (-1.0, "must be greater than 0; got -1"),
                      (None, "is blank")):
        with pytest.raises(Refusal) as e:
            obsm.build_experiment_obs_chi(chi_cfg, torch.zeros(8), [(torch.zeros(8), 5.0)], T_obs_s, F0)
        assert e.value.field == "chi_f0_si" and words in str(e.value), (F0, str(e.value))


def test_each_artifact_gets_the_log_of_the_entry_that_wrote_it(store, monkeypatch):
    """V4's file. Records are buffered from the OUTERMOST public entry (core/runs.py), and the writer
    commits the buffer so far into ``log.txt`` beside the manifest -- so a composition's two artifacts
    hold two different files: the observation's ends at its own commit, the inference's holds the
    whole composition, the pre-spend judgements included. A reviewer opening an inference folder a
    month later reads what the run said, in order, and the warning it said first.

    The stages are stubs that WRITE through the store and log one line each: this task lands the
    plumbing before any print is converted (T17-T19), so the records here are the test's own. The
    format is pinned too -- ``HH:MM:SS level message`` -- because piece 4's browser will show it. An
    artifact written outside any entry gets no file at all: the file is a run's record, not a
    directory decoration."""
    import logging
    import re

    from core import orchestrator
    from core.artifacts.store import LOG_FILE
    from core.config import CELL_PATH
    log = logging.getLogger("core.orchestrator")

    def _gen(c, **k):
        log.info("simulating the observation")
        with k["store"].create("observation", c) as w:
            w.body = {"mode": c.observation_mode, "conditioning": mf.conditioning_block(c),
                      "x_obs_digest": "0" * 16, "T_obs_cell": 1.0, "n_obs": 1, "forcing_vals": {},
                      "chi_obs_freqs": None, "source": {"kind": "simulated"}}
        return w

    def _inf(c, p, obs, **k):
        log.warning("inferring on a stub")
        with k["store"].create("inference", c, name=k["name"]) as w:
            w.body = {"results": {}}
        return w

    monkeypatch.setattr(orchestrator, "generate_observations", _gen)
    monkeypatch.setattr(orchestrator, "infer_and_visualize", _inf)
    cfg = _spont_cfg()
    cell = str(CELL_PATH / "nadrowski" / "master_weak.txt")
    with pytest.warns(orchestrator.PreflightWarning):
        obs, inf = orchestrator.simulated_inference(cfg, _sim_post(), 0.5, cell=cell, name="logged", store=store)
    obs_log = (obs.dir / LOG_FILE).read_text(encoding="utf-8")
    inf_log = (inf.dir / LOG_FILE).read_text(encoding="utf-8")
    stamp = r"\d\d:\d\d:\d\d"
    assert re.search(rf"^{stamp} warning PreflightWarning: T_obs=0\.50s is below the training range minimum",
                     obs_log, re.M), obs_log
    assert re.search(rf"^{stamp} info simulating the observation$", obs_log, re.M), obs_log
    assert "inferring on a stub" not in obs_log, "the observation's file must end at its own commit"
    assert inf_log.startswith(obs_log), "the inference's file holds everything the observation's does"
    assert re.search(rf"^{stamp} warning inferring on a stub$", inf_log, re.M), inf_log
    assert obs_log.endswith("\n") and inf_log.endswith("\n")
    assert LOG_FILE not in store.get("inference", inf.id).payloads, "the log is not a payload"
    plain = _make(store, "calibration", name="plain")                   # outside any entry
    assert not (plain.dir / LOG_FILE).exists(), "no run, no file"
    # A run that said NOTHING before its commit still gets the file, empty (spec §4.4, V4): a clean
    # simulated observation logs no record, and "every committed artifact but the cache has one" is
    # the invariant piece 4's browser reads. Silence is a record too.
    from core.runs import capture_run
    with capture_run() as run:
        quiet = _make(store, "calibration", name="quiet")
        assert run.lines == [], run.lines                                # non-vacuous: nothing was said
    assert (quiet.dir / LOG_FILE).exists(), "a run with no records must still write its log.txt"
    assert (quiet.dir / LOG_FILE).read_text(encoding="utf-8") == ""


def test_a_checkpointed_training_logs_into_the_posterior_and_never_into_the_cache(tiny_run, monkeypatch):
    """The simulation cache is the one artifact kind with no ``log.txt`` (spec §1.2): its manifest is
    training_checkpoint's, refreshed per batch across resumes, and one cache is shared by every
    posterior that names it -- no single commit holds one entry's records, and the store refuses the
    writer for the kind (store.create). The records land in the POSTERIOR's file when build_posterior
    commits. Pinned on a real checkpointed training at tiny size, with train_nn wrapped to emit one
    record from inside the entry (the pipeline's own prints become records in T18; until then the
    wrapper is the only voice in there)."""
    import logging

    from core import orchestrator
    from core.artifacts.store import LOG_FILE
    r = tiny_run
    real_train_nn = orchestrator.pipeline.train_nn

    def _logged(*a, **k):
        logging.getLogger("core.SBI.pipeline").info("[stub] training on the cache")
        return real_train_nn(*a, **k)

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", _logged)
    post = orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=r.sink, num_runs=2,
                                        hidden_features=8, num_transforms=1, stop_after_epochs=1,
                                        name="logged_post", checkpoint_every=1)
    text = (post.path / LOG_FILE).read_text(encoding="utf-8")
    assert "info [stub] training on the cache" in text, text
    cache = r.store.kind_dir("simulation") / post.manifest.parents["simulation"]
    assert (cache / "manifest.json").exists(), "the run was not checkpointed"
    assert not (cache / LOG_FILE).exists(), "the simulation cache must never carry a log file"
    assert LOG_FILE not in post.manifest.payloads, "the log is not a payload"


def test_python_warnings_reach_the_artifact_log(store):
    """The buffer tees ``warnings.showwarning`` while a run is active (spec §4.4): a PreflightWarning
    judgement or a library's RuntimeWarning is what a reviewer wants in the file, and neither is a
    logging record. The tee CALLS THE PREVIOUS HOOK -- here pytest.warns's recorder, in the window
    streams' own -- and puts it back afterwards, so the window and tool routes for warnings are
    unchanged and the two nest cleanly."""
    import re
    import warnings

    from core import runs
    from core.artifacts.store import LOG_FILE
    before = warnings.showwarning
    with pytest.warns(RuntimeWarning, match="shared with the desktop"):
        hook_under_pytest = warnings.showwarning
        with runs.capture_run():
            assert warnings.showwarning is not hook_under_pytest, "the buffer must tee the hook"
            warnings.warn("the card is shared with the desktop", RuntimeWarning)
            w = _make(store, "calibration", name="warned")
        assert warnings.showwarning is hook_under_pytest, "the tee must put pytest's hook back"
    assert warnings.showwarning is before
    text = (w.dir / LOG_FILE).read_text(encoding="utf-8")
    assert re.search(r"^\d\d:\d\d:\d\d warning RuntimeWarning: the card is shared with the desktop$",
                     text, re.M), text


def test_the_duplicated_judgements_are_said_once(store, caplog, capsys):
    """Two judgements used to be said TWICE, once per channel. build_posterior's "the loaded cell's
    GROUND TRUTH lies OUTSIDE the truncation region" and infer_and_visualize's accepted "this
    posterior is NOT AMORTIZED ... Running anyway (accepted)." were each a print, which the window
    showed at info, AND a warnings.warn, which it showed at warning -- so the pane and the tool's
    terminal carried each one twice, and a per-artifact log.txt would have too. A message carries
    its own level now, so each is said ONCE, as the Python warning (spec §4.1): the pane shows it with
    a triangle, the tool prints it on stderr, the run buffer tees it into log.txt. The accepted
    sentence keeps "Running anyway (accepted)." because walkthrough row B3 reads it in the pane.

    The warning is a PreflightWarning, not a bare UserWarning. A bare one is shown once per call site
    and text, so a second identical judgement in one session -- the same posterior on the same foreign
    observation, twice -- would be silent, where the print it replaces said it every time; the
    module's "always" filter on PreflightWarning keeps the single channel saying it.

    Behaviourally on the cheap site: the stand-in observation stops infer_and_visualize at install,
    the first step after the judgement, so nothing is sampled. The GROUND TRUTH block sits behind a
    prior build and a region; its behaviour is pinned end to end by test_user_sbi.py::
    test_a_tsnpe_round_reuses_the_parents_basis_and_refuses_every_mismatch, and both sites are pinned
    here at the AST: the message variable reaches warnings.warn, as a PreflightWarning, and nothing
    else but the Refusal it is also raised as."""
    import ast
    import warnings
    from types import SimpleNamespace

    from core import orchestrator
    from core.artifacts import Accept
    from tests._fixtures import code_only

    class _Stop(Exception):
        """Raised by the stand-in observation's install: the step right after the judgement."""

    def _install(cfg):
        raise _Stop("stopped at install")

    post = SimpleNamespace(posterior=SimpleNamespace(x_obs_digest="f" * 16), name="", id="post",
                           manifest=SimpleNamespace(body={"mode": "forced", "conditioning": {"width": 61}}))
    obs = SimpleNamespace(digest="a" * 16, mode="forced", width=61, name="bench", id="obs",
                          manifest=SimpleNamespace(body={"source": {"kind": "experimental"}}),
                          install=_install)
    caplog.clear()
    capsys.readouterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_Stop):
            orchestrator.infer_and_visualize(SimpleNamespace(), post, obs, store=store,
                                             accept=Accept(other_observation=True))
    said = [w for w in caught if "NOT AMORTIZED" in str(w.message)]
    assert len(said) == 1, [str(w.message) for w in caught]
    assert str(said[0].message).endswith(" Running anyway (accepted)."), str(said[0].message)
    assert "a" * 16 in str(said[0].message) and "f" * 16 in str(said[0].message), str(said[0].message)
    assert issubclass(said[0].category, orchestrator.PreflightWarning), said[0].category
    echoed = [(r.levelname, r.getMessage()) for r in caplog.records if "NOT AMORTIZED" in r.getMessage()]
    assert echoed == [], f"the judgement was said again as a record: {echoed}"
    assert "NOT AMORTIZED" not in capsys.readouterr().out, "the judgement was said again on stdout"

    def _handed(fn, var):
        """The callee of every call in fn's code that is handed ``var`` anywhere in its arguments."""
        return sorted(ast.unparse(c.func) for c in ast.walk(ast.parse(code_only(fn)))
                      if isinstance(c, ast.Call)
                      and any(isinstance(n, ast.Name) and n.id == var
                              for a in (*c.args, *(k.value for k in c.keywords)) for n in ast.walk(a)))

    assert _handed(orchestrator.build_posterior, "_msg") == ["warnings.warn"], \
        "build_posterior's GROUND TRUTH judgement must be said once, as the warning"
    assert _handed(orchestrator.infer_and_visualize, "_msg") == ["Refusal", "warnings.warn"], \
        "the NOT AMORTIZED judgement is raised, or warned once when accepted -- never also printed or logged"
    for fn in (orchestrator.build_posterior, orchestrator.infer_and_visualize):
        warns = [c for c in ast.walk(ast.parse(code_only(fn))) if isinstance(c, ast.Call)
                 and ast.unparse(c.func) == "warnings.warn" and "_msg" in ast.unparse(c)]
        assert len(warns) == 1 and len(warns[0].args) == 2 \
            and ast.unparse(warns[0].args[1]) == "PreflightWarning", \
            (fn.__name__, [ast.unparse(w) for w in warns])


def test_the_orchestrator_says_everything_through_its_logger():
    """V4 in core/orchestrator.py: every in-stage message is a record on ONE logger named for the
    module, and the module sets no level (core/runs.py set the family's, once, at import). A print
    left behind reaches the window at info whatever it says, is never in the artifact's log.txt (the
    run buffer hears records and Python warnings, not stdout), and on the tool lands on stdout even
    when it is a warning. Pinned at the AST over the whole module, because the regression is the next
    print someone adds."""
    import ast
    import logging

    from core import orchestrator
    from tests._fixtures import code_only

    src = code_only(orchestrator)
    printed = [ast.unparse(n)[:90] for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
    assert printed == [], f"core/orchestrator.py still prints ({len(printed)}): {printed}"
    assert orchestrator.log is logging.getLogger("core.orchestrator")
    assert orchestrator.log.level == logging.NOTSET, "the module must not set a level of its own"
    assert orchestrator.log.getEffectiveLevel() == logging.INFO
    assert "setLevel" not in src


def test_a_summary_says_whether_the_run_finished_for_every_kind(store):
    """B3 (spec §2.1, §2.6). ``complete`` means "has a valid manifest"; ``finished`` means "the run
    finished". For six of the eight kinds they are the same fact -- ``ArtifactWriter._commit`` writes
    the manifest LAST, so a manifest exists only for a run that reached the end. For the two kinds
    whose body carries a ``complete`` flag they differ: ``training_checkpoint.create`` manifests the
    simulation cache BEFORE the first batch is simulated, and an fdt record is written progressively
    and keeps its folder when the run is interrupted (E2). For both, ``body["complete"]`` is the field
    that says whether the run got to the end.

    ``complete`` is NOT redefined: StorePicker.refresh skips rows without it, ``list``'s secondary
    sort puts them last, and this suite asserts on it. The honest question goes beside it instead.
    """
    from core import artifacts
    from core.SBI.training_checkpoint import identity_digest
    assert artifacts.KIND_DIRS is st.KIND_DIRS, \
        "re-exported from the package, so nothing outside the store imports the submodule for it"

    for kind, body in _bodies().items():
        w = _make(store, kind, name=f"n_{kind}", body=body)
        row = store.list(kind)[0]
        assert (row.id, row.complete, row.finished) == (w.id, True, True), kind
        assert (row.batches_done, row.batches_planned, row.rows) == (None, None, None), kind

    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "prior_fingerprint": "c" * 16,
             "truncation": None}
    digest = identity_digest(ident)
    st.write_simulation_manifest(store.kind_dir("simulation") / digest, ident, batches_done=1)
    cache = store.list("simulation")[0]
    assert (cache.id, cache.complete) == (digest, True), "a cache is manifested from its first batch on"
    assert cache.finished is False, "... and it is not FINISHED until its rows are all there"
    assert (cache.batches_done, cache.batches_planned, cache.rows) == (1, 3, None), \
        "1 of the 3 batches the identity PLANS; the row counts land only at mark_complete"

    (store.kind_dir("inference") / "leftover__20260101T000000").mkdir(parents=True)
    leftover = [r for r in store.list("inference") if not r.complete]
    assert len(leftover) == 1 and leftover[0].finished is False, "no manifest, no finished run"


def test_an_fdt_row_carries_its_study_its_points_and_whether_it_finished(store):
    """Checklist 8 and 9. ``finished`` used to be "True unless this is the simulation kind" -- one
    kind named by hand. Two kinds now carry a ``complete`` flag in their body, and the branch is
    derived from BODY_KEYS rather than restated, so a third can never be forgotten.

    The point counts come off ``body["points"]`` in ONE manifest read, like every other Summary
    field (B2): a listing costs one directory scan and a browser row costs nothing. A single run and
    a comparison have no points at all -- ``body["points"]`` is null -- and their counts stay None
    rather than becoming a confident, false 0/0.
    """
    sweep = _bodies()["fdt"]
    sweep.update(study="sweep", points={"param": "S", "planned": 12, "done": 10, "failed": 2})
    w = _make(store, "fdt", name="a_sweep", body=sweep)
    row = [r for r in store.list("fdt") if r.id == w.id][0]
    assert (row.study, row.points_done, row.points_planned, row.points_failed) == ("sweep", 10, 12, 2)
    assert row.complete and row.finished, "a committed record is finished"

    single = _bodies()["fdt"]
    w2 = _make(store, "fdt", name="a_single", body=single)
    one = [r for r in store.list("fdt") if r.id == w2.id][0]
    assert one.study == "single"
    assert (one.points_done, one.points_planned, one.points_failed) == (None, None, None), \
        "a single run has no operating points; it must not report 0 of 0"

    # every other kind leaves all four alone
    for kind in ("prior", "calibration", "diagnostic"):
        other = _make(store, kind, name=f"o_{kind}", body=_bodies()[kind])
        s = [r for r in store.list(kind) if r.id == other.id][0]
        assert (s.study, s.points_done) == (None, None), kind
        assert s.finished is True, "a kind with no ``complete`` key is finished by construction"

    # the branch is DERIVED, not a hand-written pair of names
    assert {k for k, keys in mf.BODY_KEYS.items() if "complete" in keys} == {"simulation", "fdt"}, \
        "the two kinds whose manifest exists before the run has finished"


def test_a_cache_row_carries_its_progress_until_mark_complete_flips_it(store):
    """§2.6's second pin, over a real checkpoint's whole life: ``create`` manifests it at zero batches,
    ``save`` refreshes ``batches_done``, and ``mark_complete`` is the only thing that makes it
    finished. The rows-per-batch come back as a TUPLE, so a front-end formatter can sum them without
    caring that the manifest stores a JSON list.

    ``batches_planned`` comes off the identity and never moves, so the progress cell reads
    ``0/3``, ``2/3``, ``3/3`` over this run's life -- and the row count is None for the first two,
    because ``save`` passes no rows (P2)."""
    from core.SBI import training_checkpoint as tc
    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "run_size": 4,
             "prior_fingerprint": "d" * 16, "truncation": None}
    d = tc.resolve_dir(ident)                      # the ``store`` fixture is the process default
    tc.create(d, ident, schedule_t_scales=torch.ones(3), schedule_Ts=torch.ones(3),
              inits=torch.zeros(1, 2), V=None, probe=torch.zeros(7, 2, dtype=torch.float64),
              run_size=4, n_runs=3, hw=config.cpu_device())
    fresh = store.list("simulation")[0]
    assert (fresh.complete, fresh.finished, fresh.batches_done, fresh.batches_planned,
            fresh.rows) == (True, False, 0, 3, None)
    tc.save(d, from_batch=0, batch_k=2, rng=None, x_buf=torch.zeros(12, 5), th_buf=torch.zeros(12, 2),
            run_size=4)
    mid = store.list("simulation")[0]
    assert (mid.complete, mid.finished, mid.batches_done, mid.batches_planned,
            mid.rows) == (True, False, 2, 3, None), "2 of 3, and no row count from a save"
    tc.mark_complete(d, 3, rows=(12, 5))
    done = store.list("simulation")[0]
    assert (done.complete, done.finished, done.batches_done, done.batches_planned) == (True, True, 3, 3)
    assert done.rows == (12, 5) and isinstance(done.rows, tuple), done.rows
    assert done.dir_name == d.name == done.id, "the cache's folder IS its identity digest"


def test_every_row_carries_its_own_directory_name(store, monkeypatch):
    """``dir_name`` is the folder, ALWAYS -- the handle ``remove_incomplete`` takes (§2.3). ``id``
    keeps exactly today's meaning: the manifest id for a complete row, the directory name for an
    incomplete one, because ``get()``'s refusal already lists incomplete directory names and
    ``StorePicker`` keys on ``id``.

    The two disagree on a COMPLETE row too, and the listing is the first place it shows: ``rename``
    writes the new name into the manifest FIRST and moves the directory second, tolerating a
    PermissionError on the move because the manifest is what resolves. A held handle therefore leaves
    a row whose manifest says ``renamed__<id>`` in a folder still called ``p__<id>``.
    """
    w = _make(store, "calibration", name="p")
    row = store.list("calibration")[0]
    assert row.dir_name == w.dir.name == f"p__{w.id}" and row.id == w.id

    (store.kind_dir("calibration") / "half_written__20260101T000000").mkdir(parents=True)
    incomplete = [r for r in store.list("calibration") if not r.complete][0]
    assert incomplete.dir_name == "half_written__20260101T000000" == incomplete.id

    def _held(self, target):
        raise PermissionError("a preview window holds the folder open")

    monkeypatch.setattr(Path, "rename", _held)     # _atomic_write uses os.replace, so the manifest still lands
    store.rename("calibration", w.id, "renamed")
    monkeypatch.undo()
    assert store.get("calibration", w.id).dir_name == f"renamed__{w.id}", "the manifest took the name"
    moved = [r for r in store.list("calibration") if r.complete][0]
    assert (moved.name, moved.id) == ("renamed", w.id)
    assert moved.dir_name == f"p__{w.id}", "the folder did not move, and the listing says so"


def test_a_diagnostics_row_carries_its_variant_and_no_observation_mode(store):
    """A diagnostic records its mode under ``variant``, never under ``mode``. The reason is in
    ``core/artifacts/manifest.py``'s BODY_KEYS comment: ``Summary.mode`` is the OBSERVATION mode, so a
    listing that showed "jacobian" in that column would be reporting a conditioning geometry that does
    not exist. §2.1 surfaces ``variant`` beside it rather than changing that."""
    _make(store, "diagnostic", name="ident",
          body={"diagnostic": "identifiability", "variant": "jacobian", "settings": {}, "results": {}})
    row = store.list("diagnostic")[0]
    assert row.variant == "jacobian" and row.mode is None

    _make(store, "calibration", name="cal", body=_cal_body())
    assert store.list("calibration")[0].variant is None, "only a diagnostic has one"

    post = _make(store, "posterior", name="post", body=_bodies()["posterior"])
    prow = store.list("posterior")[0]
    assert prow.id == post.id and (prow.mode, prow.width, prow.amortized) == ("chi", 50, True)
    assert prow.variant is None
    assert (prow.batches_done, prow.batches_planned, prow.rows) == (None, None, None), \
        "progress is the simulation cache's alone"


def test_a_malformed_simulation_body_degrades_that_field_not_the_whole_row(store):
    """Review finding (task 3, fix round 1): ``manifest.validate`` checks the body's KEY-SET and
    top-level types only -- ``_check_finite`` rejects a non-finite FLOAT, nothing else -- so a
    hand-edited or partially-written manifest can hold a non-numeric ``identity["n_runs"]`` or a
    non-numeric element of ``rows`` and still pass validation. ``list()`` must not raise on that: one
    corrupt cache directory would otherwise empty StorePicker.refresh's whole listing (it wraps
    ``list()`` in a bare ``except Exception: rows = []``). The malformed field degrades to None; every
    other field on the row -- including ``complete``, which still means "has a valid manifest" -- stays
    intact."""
    from core.SBI.training_checkpoint import identity_digest
    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "prior_fingerprint": "e" * 16,
             "truncation": None}
    digest = identity_digest(ident)
    d = store.kind_dir("simulation") / digest
    st.write_simulation_manifest(d, ident, batches_done=2)

    mpath = d / "manifest.json"
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    doc["body"]["identity"]["n_runs"] = "not-a-number"     # corrupt the PLANNED total
    doc["body"]["rows"] = [10, "bad", 5]                   # one bad element spoils the whole tuple
    mpath.write_text(json.dumps(doc), encoding="utf-8")

    rows = store.list("simulation")                        # must not raise
    assert len(rows) == 1
    row = rows[0]
    assert (row.id, row.dir_name, row.complete, row.finished, row.batches_done) == \
        (digest, digest, True, False, 2), "everything the corrupt fields don't touch stays intact"
    assert row.batches_planned is None, "non-numeric n_runs degrades to None, not a raise"
    assert row.rows is None, "one non-numeric element spoils the tuple, not a partial conversion"


def test_read_log_tells_a_committed_run_from_a_silent_one_and_from_no_file_at_all(store):
    """B4's four answers (spec §2.2, §2.6). The browser has to say something honest in each case, and
    one place works it out so the two front ends cannot disagree:

      * a committed artifact's records, verbatim, stamped ``HH:MM:SS level`` -- the format
        test_each_artifact_gets_the_log_of_the_entry_that_wrote_it pins BECAUSE this reader shows it;
      * ``("", False)``: the run said nothing, and silence is a record too;
      * ``(None, False)``: there is NO file. The simulation cache never gets one (no writer, a
        manifest refreshed batch by batch across resumes, one cache shared by every posterior that
        names it), and neither does an artifact written outside any run.
    """
    import logging
    import re

    from core import runs
    from core.SBI.training_checkpoint import identity_digest

    with runs.capture_run():
        logging.getLogger("core.orchestrator").info("said something")
        spoke = _make(store, "calibration", name="spoke")
    text, truncated = store.read_log("calibration", spoke.id)
    assert truncated is False
    assert re.fullmatch(r"\d\d:\d\d:\d\d info said something\n", text), repr(text)
    assert store.read_log("calibration", "spoke")[0] == text, "a name resolves like an id"

    with runs.capture_run() as run:
        quiet = _make(store, "calibration", name="quiet")
        assert run.lines == [], run.lines            # non-vacuous: nothing was said before the commit
    assert store.read_log("calibration", quiet.id) == ("", False)

    outside = _make(store, "calibration", name="outside")
    assert store.read_log("calibration", outside.id) == (None, False), "no run, no file"

    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "truncation": None}
    digest = identity_digest(ident)
    st.write_simulation_manifest(store.kind_dir("simulation") / digest, ident, batches_done=1)
    assert store.read_log("simulation", digest) == (None, False), "the cache never carries one"

    # Both refusals are ITS OWN and carry field="artifact" (P1): routing through ``path()`` would
    # inherit a field-less refusal, and neither front end could then name the way out of it.
    with pytest.raises(st.StoreError, match="no complete calibration") as e:
        store.read_log("calibration", "nothing_by_that_name")
    assert e.value.field == "artifact"
    with pytest.raises(st.StoreError, match="unknown artifact kind") as e:
        store.read_log("priors", "spoke")
    assert e.value.field == "artifact"


def test_read_log_returns_the_tail_over_max_bytes(store):
    """The detail pane reads at most 1 MiB and a run of days writes more than that, so the TAIL is
    what matters -- the crash is at the end -- and ``truncated`` is how the pane knows to say so.
    At or above the file's own size nothing is cut and ``truncated`` stays False."""
    from core import runs
    with runs.capture_run():
        w = _make(store, "calibration", name="long")
    body = "head\n" + "x" * 40 + "\ntail\n"
    # newline="\n" so the byte offsets below are the ones this test means: write_text's default
    # newline=None would put CRLF on disk and every max_bytes leg would be cutting different bytes.
    (w.dir / st.LOG_FILE).write_text(body, encoding="utf-8", newline="\n")
    size = (w.dir / st.LOG_FILE).stat().st_size
    assert store.read_log("calibration", w.id) == (body, False)
    assert store.read_log("calibration", w.id, max_bytes=size) == (body, False)
    assert store.read_log("calibration", w.id, max_bytes=size + 10) == (body, False)
    assert store.read_log("calibration", w.id, max_bytes=5) == ("tail\n", True)
    # 0 asks for nothing and must GET nothing: ``data[-max_bytes:]`` would return the WHOLE file here
    assert store.read_log("calibration", w.id, max_bytes=0) == ("", True)
    # A tail cut through a multi-byte character is replaced, never raised: the pane must still render
    (w.dir / st.LOG_FILE).write_bytes("a\u00b5".encode("utf-8"))        # b'a\xc2\xb5'
    assert store.read_log("calibration", w.id, max_bytes=1) == ("\ufffd", True)
    # A file an older build wrote is CRLF on disk (write_text's default newline=None on Windows).
    # It reads back as LF, so one newline convention reaches the pane, the report and both front
    # ends whoever wrote the file.
    (w.dir / st.LOG_FILE).write_bytes(b"head\r\nmid\r\ntail\r\n")
    assert store.read_log("calibration", w.id) == ("head\nmid\ntail\n", False)


def _leftovers(store, kind="prior"):
    """The three shapes ``_entries`` classifies as incomplete, written into one kind: no manifest at
    all, a manifest that will not parse, and a manifest that parses but declares another kind. Also
    writes the real calibration whose manifest the third one copies. Returns the three names, sorted
    the way ``_entries`` walks them."""
    d = store.kind_dir(kind)
    d.mkdir(parents=True, exist_ok=True)
    (d / "no_manifest__20260101T000001").mkdir()
    torn = d / "unreadable__20260101T000002"
    torn.mkdir()
    (torn / st.MANIFEST).write_text("{not json", encoding="utf-8")
    wrong = d / "wrong_kind__20260101T000003"
    wrong.mkdir()
    other = _make(store, "calibration", name="elsewhere", body=_cal_body())
    (wrong / st.MANIFEST).write_bytes((other.dir / st.MANIFEST).read_bytes())
    return ["no_manifest__20260101T000001", "unreadable__20260101T000002",
            "wrong_kind__20260101T000003"]


_backdate = backdate_tree       # tests/_fixtures.py's; see R3 and store.RECENT_WRITE_SECONDS


def test_remove_incomplete_refuses_everything_that_is_not_a_leftover(store, tmp_path):
    """B7 (spec §2.3, §2.6). It is the COMPLEMENT of ``delete``: that one resolves through ``_find``,
    which returns only manifest-bearing entries, so it can only ever remove a real artifact and
    always runs the dependency check; this one asks ``_entries`` and removes only what came back
    WITHOUT a manifest. Neither can do the other's job, which is the safety property worth keeping.

    Four refusal classes, each a StoreError carrying ``field="artifact"`` so both front ends can name
    the way out of it from their own table."""
    real = _make(store, "calibration", name="keepme", body=_cal_body())
    (store.kind_dir("calibration") / "leftover__20260101T000000").mkdir(parents=True)
    (store.kind_dir("calibration") / "notadir.txt").write_text("x", encoding="utf-8")
    nested = store.kind_dir("calibration") / "sub" / "leftover__20260101T000000"
    nested.mkdir(parents=True)

    for kind, name, why in (
            ("priors", "leftover__20260101T000000", "unknown artifact kind"),   # not a kind at all
            ("calibration", "sub/leftover__20260101T000000", "not the name of a directory"),
            ("calibration", "sub\\leftover__20260101T000000", "not the name of a directory"),
            ("calibration", "..", "not the name of a directory"),
            ("calibration", "", "not the name of a directory"),
            ("calibration", str(tmp_path), "not the name of a directory"),      # an absolute path
            ("calibration", "/etc", "not the name of a directory"),             # ... posix-spelled
            ("calibration", "notadir.txt", "no directory named"),               # a plain file
            ("calibration", "never_existed", "no directory named"),
            ("calibration", real.dir.name, "holds a valid calibration manifest")):   # a REAL artifact
        with pytest.raises(st.StoreError, match=why) as e:
            store.remove_incomplete(kind, name)
        assert e.value.field == "artifact", (kind, name, e.value.field)

    assert real.dir.is_dir() and store.get("calibration", real.id).name == "keepme"
    assert (store.kind_dir("calibration") / "notadir.txt").is_file()
    assert nested.is_dir(), "a path was refused, not followed"
    assert (store.kind_dir("calibration") / "leftover__20260101T000000").is_dir(), \
        "none of the refusals removed anything"

    target = store.kind_dir("calibration") / "leftover__20260101T000000"
    _backdate(target)
    assert store.remove_incomplete("calibration", "leftover__20260101T000000") == target
    assert not target.exists()


def test_remove_incomplete_removes_only_a_directory_with_no_manifest_file_at_all(store):
    """R1 (whole-piece review). The three shapes ``_entries`` classifies as incomplete are NOT the
    same thing, and only the first of them is a leftover this may remove: a crash before the manifest
    was written.

    A manifest that EXISTS but this build cannot parse, and one that declares another kind, both mean
    "something is here that I do not understand". A reviewer probed the old rule deleting a REAL
    calibration with its payload, because a manifest that is valid under a DIFFERENT SCHEMA reads as
    "no artifact here" -- and ``manifest.SCHEMA`` is a versioned constant that is expected to move.
    Those two are REPORTED: they keep their row in the listing, with their reason, and that is where
    they stay.
    """
    names = _leftovers(store, "prior")
    _backdate(store.kind_dir("prior"))
    reasons = {r.dir_name: r.reason for r in store.list("prior")}
    assert set(reasons) == set(names), reasons
    assert reasons[names[0]] == st.NO_MANIFEST_REASON
    assert "unreadable manifest" in reasons[names[1]]
    assert "declares kind 'calibration'" in reasons[names[2]]

    removed = store.remove_incomplete("prior", names[0])
    assert removed.name == names[0] and not removed.exists()

    for name in names[1:]:
        with pytest.raises(st.StoreError, match="carries a manifest.json") as e:
            store.remove_incomplete("prior", name)
        assert e.value.field == "artifact"
        assert reasons[name] in str(e.value), "the refusal must carry the listing's own reason"
        assert (store.kind_dir("prior") / name).is_dir(), f"{name} was removed"
    assert {r.dir_name for r in store.list("prior")} == set(names[1:]), \
        "the two that were refused must still be listed, with their reason"
    assert store.get("calibration", "elsewhere").name == "elsewhere", \
        "the real artifact whose manifest the wrong-kind folder copied is untouched"


def test_remove_incomplete_skips_a_directory_something_may_still_be_writing(store):
    """R3. ``ArtifactWriter`` creates the directory FIRST and writes the manifest LAST, so for the
    whole of a run -- minutes for a prior, days for a training -- a live run's directory looks
    exactly like a leftover. A reviewer probed a sweep from a SECOND process removing one: the run
    died at its commit, and a running cache lost its committed shards. A cross-process lock is out of
    scope for this piece; a directory whose TREE was touched in the last few minutes is refused
    instead, and the refusal says how old it is.

    The directory's own mtime is not the question: a long write to a file already inside it leaves
    the folder's mtime alone, so the guard reads the newest mtime in the whole tree.
    """
    d = store.kind_dir("prior")
    d.mkdir(parents=True, exist_ok=True)
    live = d / "inflight__20260101T000001"
    live.mkdir()
    (live / "prior.pt").write_bytes(b"a run still writing this")

    with pytest.raises(st.StoreError, match="may still be writing") as e:
        store.remove_incomplete("prior", live.name)
    assert e.value.field == "artifact" and live.is_dir()

    old = time.time() - 3600
    os.utime(live, (old, old))                  # the FOLDER is old; the payload inside it is not
    with pytest.raises(st.StoreError, match="may still be writing"):
        store.remove_incomplete("prior", live.name)
    assert live.is_dir()

    _backdate(live)
    assert store.remove_incomplete("prior", live.name) == live and not live.exists()


def test_sweep_incomplete_removes_exactly_the_entries_it_was_handed(store):
    """R2: the sweep is BOUND to the list that was shown. It used to re-scan at removal time, so a
    directory that appeared while the confirmation sat open was deleted although the operator never
    saw it -- probed, and reported as "Removed 2 of 1 leftover directories". Now the caller computes
    the list, shows it, and hands exactly it here, and each entry goes by name through
    ``remove_incomplete`` -- so every refusal that call was hardened with applies, and the count in
    the caller's sentence is true by construction.
    """
    names = _leftovers(store, "prior")
    elsewhere = store.kind_dir("inference") / "orphan__20260101T000009"
    elsewhere.mkdir(parents=True)
    _backdate(store.root)

    assert store.sweep_incomplete([]) == ([], []), "an empty list removes nothing"

    appeared = store.kind_dir("prior") / "appeared_while_the_dialog_was_open"
    appeared.mkdir()
    _backdate(appeared)

    removed, failed = store.sweep_incomplete([("prior", names[0])])
    assert removed == [("prior", names[0])] and failed == []
    assert appeared.is_dir(), "a directory the operator never saw was removed"
    assert elsewhere.is_dir(), "a kind that was not handed in was swept anyway"

    # Anything the list names that is NOT removable comes back in `failed`, with remove_incomplete's
    # own refusal as the reason -- never as a raise that would cost the rest of the list.
    real = _make(store, "calibration", name="keepme", body=_cal_body())
    removed, failed = store.sweep_incomplete(
        [("calibration", real.dir.name), ("priors", "nosuchkind"), ("prior", names[1]),
         ("prior", "appeared_while_the_dialog_was_open")])
    assert removed == [("prior", "appeared_while_the_dialog_was_open")]
    assert [(k, d) for k, d, _ in failed] == [("calibration", real.dir.name), ("priors", "nosuchkind"),
                                              ("prior", names[1])]
    assert "holds a valid calibration manifest" in failed[0][2]
    assert "unknown artifact kind" in failed[1][2]
    assert "carries a manifest.json" in failed[2][2]
    assert real.dir.is_dir() and store.get("calibration", real.id).name == "keepme"


def test_sweep_incomplete_finishes_the_rest_when_one_directory_will_not_delete(store, monkeypatch):
    """B7's one action per kind, and per store. A directory that will not delete is REPORTED, never
    fatal: on Windows a held handle -- an Explorer preview, a virus scanner, a file this process still
    has open -- makes ``shutil.rmtree`` raise PermissionError, and ``_rmtree_retry`` waits 0.1 s and
    then 0.2 s before giving up. One such folder must not cost the operator the other six.

    The failure is INJECTED rather than provoked with a real handle, so the pin holds on any
    filesystem and costs no sleep."""
    names = _leftovers(store, "prior")
    orphan = store.kind_dir("inference") / "orphan__20260101T000009"
    orphan.mkdir(parents=True)
    real = _make(store, "calibration", name="keepme", body=_cal_body())
    _backdate(store.root)

    stuck = store.kind_dir("prior") / names[0]
    real_rmtree = st._rmtree_retry

    def _one_held(path, **kw):
        if Path(path) == stuck:
            raise PermissionError("another process holds this folder open")
        return real_rmtree(path, **kw)

    monkeypatch.setattr(st, "_rmtree_retry", _one_held)
    removed, failed = store.sweep_incomplete([("prior", names[0]), ("inference", orphan.name)])
    monkeypatch.undo()
    assert removed == [("inference", orphan.name)], "one held handle stopped the rest of the sweep"
    assert len(failed) == 1 and failed[0][:2] == ("prior", names[0])
    assert "another process holds this folder open" in failed[0][2]
    assert stuck.is_dir(), "the one it could not remove is still there"
    assert not orphan.exists()

    removed, failed = store.sweep_incomplete([("prior", names[0])])
    assert removed == [("prior", names[0])] and failed == []
    assert store.get("calibration", real.id).name == "keepme", "a real artifact is never swept"


def test_delete_refuses_with_the_artifact_field_and_says_force_true(store):
    """``delete``'s own two refusals carry ``field="artifact"`` too (P1, P20), because the fix for
    either is the same: pick a different artifact in the list. The WORDING is untouched --
    test_delete_refuses_naming_dependents_and_force_deletes (``:278-287``) already pins the message and
    the force path -- and this test pins only the key, plus the fact that the dependents message still
    names ``force=True``. Naming a Python keyword argument is not naming a control: no front end offers
    it (B6), and whoever reads that line in ``log.txt`` needs to know what the escape hatch is called.
    """
    parent = _make(store, "posterior", name="mother", body=_bodies()["posterior"])
    child = _make(store, "inference", name="child", body={"results": {"n_samples": 5}},
                  parents={"posterior": parent.id})
    with pytest.raises(st.StoreError, match="refusing to delete") as e:
        store.delete("posterior", parent.id)
    assert e.value.field == "artifact" and "force=True" in str(e.value)
    assert parent.dir.is_dir() and child.dir.is_dir(), "a refusal removed nothing"

    with pytest.raises(st.StoreError, match="no complete calibration") as e:
        store.delete("calibration", "never_existed")
    assert e.value.field == "artifact"

    store.delete("inference", child.id)
    store.delete("posterior", parent.id)           # the dependent is gone; no force needed
    assert store.list("posterior") == [] and store.list("inference") == []


def _short_name(path: Path) -> "str | None":
    """The Windows 8.3 short name for ``path``'s own component, or None when this platform is not
    Windows, or the volume's 8dot3name generation is disabled (a per-volume policy) so no distinct
    short alias exists -- the leg that uses this skips cleanly rather than assume one exists."""
    if os.name != "nt":
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(260)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
    if not n:
        return None
    short = Path(buf.value[:n]).name
    return short if short.lower() != path.name.lower() else None


def test_remove_incomplete_cannot_be_defeated_by_an_alias_for_a_real_directory(store):
    """Fix round 1 (CRITICAL, reproduced against the pre-fix code by the review): a plain string
    compare (``s.name == dir_name``) against ``_entries``'s listing can be defeated by any spelling
    Windows resolves to the SAME directory as a real artifact's -- a case variant, an 8.3 short
    name, or a trailing dot that Win32 strips -- deleting a complete, possibly dependency-bearing
    artifact while bypassing ``delete()``'s dependency check entirely. The fix drives the match off
    the filesystem's own identity (``os.path.samefile`` against each ``_entries`` row) rather than
    a string, so it cannot be fooled by a spelling; this test pins that every alias is refused and
    the artifact is untouched.
    """
    real = _make(store, "posterior", name="mother", body=_bodies()["posterior"])
    real_name = real.dir.name

    def _still_there():
        assert real.dir.is_dir() and store.get("posterior", real.id).name == "mother"

    aliases = [real_name.upper()]
    if real_name.lower() != real_name:
        aliases.append(real_name.lower())
    aliases.append(real_name + ".")
    for alias in aliases:
        with pytest.raises(st.StoreError, match="holds a valid posterior manifest") as e:
            store.remove_incomplete("posterior", alias)
        assert e.value.field == "artifact", (alias, e.value.field)
        _still_there()

    short = _short_name(real.dir)
    if short is not None:
        with pytest.raises(st.StoreError, match="holds a valid posterior manifest") as e:
            store.remove_incomplete("posterior", short)
        assert e.value.field == "artifact"
        _still_there()

    # The dependency-bearing artifact from the delete tests: remove_incomplete refuses it as a
    # COMPLETE artifact before dependents even enter into it -- it does not need the dependency
    # check to refuse it -- but it must never be reachable through this path either.
    child = _make(store, "inference", name="child", body={"results": {"n_samples": 5}},
                  parents={"posterior": real.id})
    with pytest.raises(st.StoreError, match="holds a valid posterior manifest") as e:
        store.remove_incomplete("posterior", real_name)
    assert e.value.field == "artifact"
    _still_there()
    assert child.dir.is_dir()


def test_remove_incomplete_wraps_an_rmtree_failure_as_a_fielded_refusal(store, monkeypatch):
    """Fix round 1 [Important]: a junction or symlink makes ``shutil.rmtree`` raise a plain OSError,
    which is not a ``Refusal`` and carries no field key -- a front end catching ``Refusal`` would
    see an unhandled crash instead. ``remove_incomplete`` must wrap it as a fielded ``StoreError``,
    the way ``sweep_incomplete`` already handles its own per-directory failures."""
    names = _leftovers(store, "prior")
    _backdate(store.kind_dir("prior"))
    target = store.kind_dir("prior") / names[0]

    def _boom(path, **kw):
        raise OSError("simulated junction failure")

    monkeypatch.setattr(st, "_rmtree_retry", _boom)
    with pytest.raises(st.StoreError, match="simulated junction failure") as e:
        store.remove_incomplete("prior", names[0])
    monkeypatch.undo()
    assert e.value.field == "artifact"
    assert target.is_dir(), "the injected failure must not have removed anything"


def test_remove_incomplete_refuses_to_report_success_when_the_directory_survives(store, monkeypatch):
    """Fix round 1 [Minor]: a path trick (a trailing space or ``...`` that Win32 quietly resolves to
    nothing) can make ``shutil.rmtree`` raise ``FileNotFoundError``, which ``_rmtree_retry`` treats
    as "already gone" -- so a naive caller reports success on a directory it never touched.
    ``remove_incomplete`` must check for itself that the directory is actually gone before
    returning it. The failure is injected (a no-op fake ``_rmtree_retry``) so the pin holds on any
    filesystem."""
    names = _leftovers(store, "prior")
    _backdate(store.kind_dir("prior"))
    target = store.kind_dir("prior") / names[0]
    monkeypatch.setattr(st, "_rmtree_retry", lambda path, **kw: None)   # "succeeds" but removes nothing
    with pytest.raises(st.StoreError, match="was not removed") as e:
        store.remove_incomplete("prior", names[0])
    monkeypatch.undo()
    assert e.value.field == "artifact"
    assert target.is_dir(), "the monkeypatched rmtree never actually removed it"


def test_remove_incomplete_is_not_defeated_by_a_degenerate_inode_volume(store, monkeypatch):
    """Fix round 2: ``os.path.samefile`` identifies a file by ``(st_dev, st_ino)``; on a volume where
    every entry's ``st_ino`` is 0 -- FAT/exFAT, some network shares, and the artifacts root is
    relocatable to exactly such a drive -- EVERY comparison would report a match, so the loop would
    settle on whichever ``_entries`` row happens to come first rather than the one actually named,
    possibly deleting a directory that was never named at all. ``os.path.realpath`` is the second,
    independent check: it compares the resolved path STRING, which is not built from the (possibly
    all-zero) inode, so it disagrees with every wrong candidate even when ``samefile`` is fooled.

    Simulated by monkeypatching ``samefile`` to always return True (a real exFAT mount cannot be
    made in this environment): a leftover that sorts before the real, complete artifact must not be
    the one removed when the real artifact is what was actually named.
    """
    d = store.kind_dir("calibration")
    d.mkdir(parents=True, exist_ok=True)
    decoy = d / "aaa_leftover__20260101T000000"
    decoy.mkdir()
    real = _make(store, "calibration", name="keepme", body=_cal_body())
    monkeypatch.setattr(os.path, "samefile", lambda a, b: True)
    with pytest.raises(st.StoreError, match="holds a valid calibration manifest") as e:
        store.remove_incomplete("calibration", real.dir.name)
    monkeypatch.undo()
    assert e.value.field == "artifact"
    assert real.dir.is_dir() and store.get("calibration", real.id).name == "keepme"
    assert decoy.is_dir(), "a stubbed samefile match must not delete a directory that was never named"


def test_sweep_incomplete_reports_a_directory_the_removal_never_actually_touched_as_failed(store, monkeypatch):
    """Fix round 2 [Important]: the same false-success class fixed in ``remove_incomplete`` lives in
    its sibling. A directory that vanishes between ``_entries``' classification and the removal
    attempt -- or a path trick that makes ``shutil.rmtree`` raise ``FileNotFoundError``, which
    ``_rmtree_retry`` treats as "already gone" and swallows -- must not be reported as removed.
    Injected with a no-op fake ``_rmtree_retry`` so the pin holds on any filesystem.

    Since R2 the sweep removes each entry BY NAME through ``remove_incomplete``, so the check is
    inherited rather than duplicated -- and the reason the sweep reports is that call's own refusal.
    """
    d = store.kind_dir("prior")
    d.mkdir(parents=True, exist_ok=True)
    for name in ("goes__20260101T000001", "survives__20260101T000002"):
        (d / name).mkdir()
    _backdate(d)
    real_rmtree = st._rmtree_retry
    survivor = d / "survives__20260101T000002"

    def _fake(path, **kw):
        if Path(path) == survivor:
            return None                 # "succeeds" but removes nothing
        return real_rmtree(path, **kw)

    monkeypatch.setattr(st, "_rmtree_retry", _fake)
    removed, failed = store.sweep_incomplete([("prior", "goes__20260101T000001"),
                                              ("prior", survivor.name)])
    monkeypatch.undo()
    assert removed == [("prior", "goes__20260101T000001")]
    assert len(failed) == 1 and failed[0][:2] == ("prior", survivor.name)
    assert "was not removed" in failed[0][2], failed
    assert survivor.is_dir(), "the one the fake rmtree never actually touched is still there"


def test_delete_verifies_the_directory_is_gone_and_wraps_a_removal_failure(store, monkeypatch):
    """R7. ``_rmtree_retry`` treats ANY ``FileNotFoundError`` as "it is already gone", including one
    raised from INSIDE the walk after part of the tree has been removed -- so ``delete`` returned
    normally and both front ends announced success over a half-removed artifact (probed: the
    directory survived, the manifest did not, and the artifact then listed as a leftover). And a
    ``PermissionError`` -- the ordinary Windows case, a handle held by a preview or a scanner --
    escaped ``artifacts rm`` past the ladder's ``Refusal`` rung as a raw traceback, for a condition
    that is just "try again in a moment".

    Both are what ``remove_incomplete`` and ``sweep_incomplete`` were given in fix round 2. ``delete``
    is the ONLY one of the three that removes a real artifact, and it was the least guarded.
    """
    real = _make(store, "calibration", name="keepme", body=_cal_body())

    monkeypatch.setattr(st, "_rmtree_retry", lambda path, **kw: None)   # "succeeds", removes nothing
    with pytest.raises(st.StoreError, match="was not removed") as e:
        store.delete("calibration", "keepme")
    monkeypatch.undo()
    assert e.value.field == "artifact"
    assert real.dir.is_dir() and store.get("calibration", "keepme").name == "keepme"

    def _held(path, **kw):
        raise PermissionError("another process holds this folder open")

    monkeypatch.setattr(st, "_rmtree_retry", _held)
    with pytest.raises(st.StoreError, match="another process holds this folder open") as e:
        store.delete("calibration", "keepme")
    monkeypatch.undo()
    assert isinstance(e.value, Refusal) and e.value.field == "artifact"
    assert real.dir.is_dir(), "a failed removal must leave the artifact where it was"

    store.delete("calibration", "keepme")           # and an ordinary removal still works
    assert not real.dir.exists()


# ── The report renderers (piece 4, §5 / B10) ─────────────────────────────────────────────────────


def _chain(store, cfg) -> dict:
    """prior -> simulation cache -> posterior -> observation -> inference, wired through ``parents``.

    The cheapest thing the REAL writers accept: one 2-component GMM for the prior (so one step has
    input files, hashes, a payload digest and a fingerprint), and `_make`'s minimal bodies for the
    rest. Nothing here simulates, trains or infers -- the renderers are pure functions of manifests,
    so a chain built by hand exercises every branch a five-hour pipeline would.
    """
    from core.SBI.training_checkpoint import identity_digest
    # ``store``'s real clock has one-second resolution and ``_new_id`` only dedupes WITHIN a kind
    # (test_same_second_ids_get_a_suffix), so four artifacts of four DIFFERENT kinds created back to
    # back -- as this helper does, on purpose, to stay fast -- can land on the identical id string
    # by pure timing. That collision is invisible to the store (each kind is its own directory) but
    # would make this test's own "the prior appears once" check count an unrelated sibling that
    # happens to share the prior's stamp. Pin the clock and step it a full second between artifacts
    # so the four ids this chain hands out are always distinct, the same way
    # ``test_unnamed_artifacts_group_and_age_out`` pins it above. 2030, not 2026-01-01, so it cannot
    # collide with the orphan test's hardcoded missing-parent id "20260101T000000" right below.
    real_clock = store._clock
    tick = {"now": datetime(2030, 1, 1, tzinfo=timezone.utc)}
    store._clock = lambda: tick["now"]
    try:
        p = _prior_artifact(store, cfg, name="chain_prior")
        fp = store.get("prior", p.id).fingerprints["gmm"]
        ident = {"format": "training-rows/2", "model": cfg.model, "prior_fingerprint": fp,
                 "n_runs": 2, "run_size": 4, "truncation": None}
        sim = st.write_simulation_manifest(store.kind_dir("simulation") / identity_digest(ident), ident,
                                           parents={"prior": p.id}, batches_done=1, rows=[4])
        tick["now"] += timedelta(seconds=1)
        post = _make(store, "posterior", name="chain_post",
                     body={"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                           "truncation": None, "training": {}},
                     parents={"prior": p.id, "simulation": sim.id})
        tick["now"] += timedelta(seconds=1)
        obs = _make(store, "observation", name="chain_obs",
                    body={"mode": "chi", "conditioning": {"width": 50}, "x_obs_digest": "0" * 16,
                          "T_obs_cell": 1.0, "n_obs": 10, "forcing_vals": {}, "chi_obs_freqs": None,
                          "source": {"kind": "simulated"}})
        tick["now"] += timedelta(seconds=1)
        inf = _make(store, "inference", name="chain_inf",
                    body={"results": {"n_samples": 5, "accepted": ["other_observation"]}},
                    parents={"posterior": post.id, "observation": obs.id})
    finally:
        store._clock = real_clock
    return {"prior": p.id, "simulation": sim.id, "posterior": post.id, "observation": obs.id,
            "inference": inf.id}


def test_render_manifest_names_every_fact_the_manifest_holds(store):
    """§5 / B10: the manifest as text -- schema, id, name, created, note, the git revision and the
    environment, the input files WITH their hashes, the parents, the payloads with their digests, the
    figures and the body's knobs -- laid out deterministically, so the window's Save and the tool's
    ``artifacts show`` print the same bytes.

    Sorted where order is not meaning: ``to_json_text`` writes the file with ``sort_keys=True``, so a
    manifest read back from disk is already in that order, and sorting here is what makes a manifest
    built in memory print identically to the same manifest read back.
    """
    from core.artifacts import render_manifest
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="rep_prior")
    m = store.get("prior", p.id)
    text = render_manifest(m)
    assert text.startswith(f"prior rep_prior  [{p.id}]\n"), text[:200]
    assert text.endswith("\n"), "a saved report ends with a newline"
    for line in (f"id:      {p.id}", "name:    rep_prior", f"created: {m.created}", "note:    (none)"):
        assert f"\n{line}\n" in text, line
    # the input files WITH their hashes, relative to Resources/ exactly as file_ref records them
    assert f"\n  bounds: Bounds/nadrowski/master.txt  sha256 {m.inputs['bounds']['sha256']}\n" in text
    assert "\n  cell: (none)\n" in text, "a bounds-built config has no cell, and the report says so"
    # the payload with its digest, and the GMM fingerprint
    assert f"\n  prior.pt  sha256 {m.payloads['prior.pt']}\n" in text
    assert f"\n  gmm: {m.fingerprints['gmm']}\n" in text
    assert "\nFigures\n  (none)\n" in text, "a prior written without a figure sink has none"
    assert "\nParents\n  (none)\n" in text
    # the knobs, and the body rendered as a sorted tree
    assert text.count("model: NADROWSKI") == 2, "once under Inputs, once under Knobs"
    assert "\n  gmm:\n    box:\n      log_mask: [" in text, text
    assert "\n    n_components: 2\n" in text and "\n  sweep: {}\n" in text
    # a long sequence is summarised, never inlined: a 13x13 rotation would bury every knob, and its
    # digest is printed two sections above
    from core.artifacts.report import _fmt_list
    assert _fmt_list([[0.0] * 13] * 13) == "[13 x 13 numbers]"
    assert _fmt_list(list(range(20))) == "[0, 1, 2, ... (20 entries)]"
    assert _fmt_list([]) == "[]" and _fmt_list([True, None]) == "[true, (none)]"
    assert render_manifest(m) == render_manifest(store.get("prior", p.id)), \
        "two calls, identical bytes -- and an in-memory manifest prints as a re-read one"


def test_render_lineage_walks_the_chain_and_prints_a_missing_parent(store):
    """§5: the artifact, then its parents transitively through ``_PARENT_KEYS``, each artifact before
    its own parents so the chain reads newest first and oldest last. Each step names the kind, the
    name, the id, the creation time, the input files with their hashes, the knobs that decided it and
    any Accept it recorded.

    A parent a manifest names but the store does not hold prints as MISSING with its id and with who
    named it: dropping it would make a broken chain read as a complete one, which is the one thing a
    provenance report must never do.
    """
    from core.artifacts import render_lineage
    cfg = _nad_cfg()
    ids = _chain(store, cfg)
    text = render_lineage(store, "inference", ids["inference"])
    assert text.startswith(f"Lineage of inference chain_inf [{ids['inference']}]\n")
    steps = [ln for ln in text.splitlines() if ln[:1].isdigit()]
    assert [ln.split(". ", 1)[1].split(" ", 1)[0] for ln in steps] == [
        "inference", "observation", "posterior", "prior", "simulation"], steps
    # the prior is named by BOTH the posterior and the cache and appears ONCE (the visited set)
    assert text.count(f"[{ids['prior']}]  created ") == 1
    assert f"parents: prior={ids['prior']}, simulation={ids['simulation']}" in text
    # the step's own facts: its inputs with their hashes, its knobs, its Accept
    assert "    bounds: Bounds/nadrowski/master.txt  sha256 " in text
    assert "model=NADROWSKI" in text, "the knobs the prior ran under"
    assert "\n  accepted: other_observation\n" in text, "the escape hatch the inference recorded"
    assert "\n  knobs:\n    (none)\n" in text, "a manifest written with no config says so"
    assert text.endswith("\n") and render_lineage(store, "inference", ids["inference"]) == text

    # a parent nothing holds: MISSING, with its id and who named it
    orphan = _make(store, "posterior", name="orphan",
                   body={"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                         "truncation": None, "training": {}},
                   parents={"prior": "20260101T000000"})
    text = render_lineage(store, "posterior", orphan.id)
    assert "2. MISSING prior [20260101T000000]" in text, text
    assert "named by posterior 'orphan' as 'prior'" in text
    assert str(store.kind_dir("prior")) in text

    # an unknown ref is the store's own refusal, unchanged
    with pytest.raises(st.StoreError, match="no complete posterior"):
        render_lineage(store, "posterior", "nope")


def test_render_lineage_resolves_what_a_comparison_compared(store):
    """§7.3 and checklist 12. A comparison names the runs it drew in its BODY, because ``parents`` is
    a flat {key: id} map that cannot carry an arbitrary number of ids without widening the store's
    contract for every kind -- so deleting a record a comparison used is NOT refused, and this branch
    is the only thing that keeps the comparison's own provenance readable afterwards.

    A record the store no longer holds prints MISSING with its kind and id, the same wording the
    parents walk already uses: skipping it would make a broken chain read as a complete one, which is
    the one thing a provenance report must never do.
    """
    from core.artifacts import render_lineage
    a = _make(store, "fdt", name="cell_a", body=_bodies()["fdt"])
    b = _make(store, "fdt", name="cell_b", body=_bodies()["fdt"])
    comp_body = _bodies()["fdt"]
    comp_body.update(study="comparison", seed=None, grid=None, offgrid=None, results=None,
                     compared={"mode": "cells",
                               "records": [{"kind": "fdt", "id": a.id, "name": "cell_a"},
                                           {"kind": "fdt", "id": b.id, "name": "cell_b"}]})
    c = _make(store, "fdt", name="two_cells", body=comp_body)

    text = render_lineage(store, "fdt", c.id)
    assert "  compared (cells):" in text, text
    assert f"    fdt cell_a [{a.id}]" in text, text
    assert f"    fdt cell_b [{b.id}]" in text, text
    assert "MISSING" not in text, "both are on disk"

    store.delete("fdt", a.id)
    text = render_lineage(store, "fdt", c.id)
    assert f"    MISSING fdt [{a.id}]" in text, text
    assert str(store.kind_dir("fdt")) in text, "where it was looked for"
    assert f"    fdt cell_b [{b.id}]" in text, "the one still there is unaffected"

    # a record with no comparison says nothing extra
    assert "compared" not in render_lineage(store, "fdt", b.id)


def test_render_lineage_reads_a_malformed_compared_block_without_dropping_or_faking_it(store):
    """Task 5's review fix. Only a hand-edited manifest or a writer bug reaches these shapes, but the
    lineage's two promises hold for them too: its one refusal is the head ref (the browser catches a
    Refusal and nothing else, so any other exception is a crash), and a broken record never reads as
    a complete one. So a malformed ``compared`` block yields a line saying WHAT is unreadable --
    never silence (a comparison that reads as having drawn nothing), never an exception, and never a
    ``MISSING fdt []`` made up from a value that was not an entry at all.
    """
    from core.artifacts import render_lineage
    real = _make(store, "fdt", name="real_cell", body=_bodies()["fdt"])
    made = iter(range(100))

    def lineage_with(compared):
        body = _bodies()["fdt"]
        body.update(study="comparison", compared=compared)
        w = _make(store, "fdt", name=f"malformed_{next(made)}", body=body)
        return render_lineage(store, "fdt", w.id)

    # a block that is there but is not a mapping: one line naming its type
    for bad, type_name in ((["x"], "list"), ("cells", "str"), (5, "int")):
        text = lineage_with(bad)
        assert f"\n  compared: UNREADABLE ({type_name}, not a mapping of mode and records)\n" in text, text

    # records that are not a list: the header and one line -- not a walk over characters or keys
    for bad, type_name in ((5, "int"), (True, "bool"), ("abc", "str"), ({"id": real.id}, "dict"),
                           (None, "null")):
        text = lineage_with({"mode": "cells", "records": bad})
        assert (f"\n  compared (cells):\n    UNREADABLE records ({type_name}, not a list of entries)\n"
                in text), text
        assert "MISSING" not in text, text
    assert "    UNREADABLE records (null, not a list of entries)\n" in lineage_with({"mode": "cells"}), \
        "no records key at all"

    # an entry that is not a record prints its repr, so a bare id is not thrown away; an entry that
    # names no id says so; the good entry beside them is unaffected
    text = lineage_with({"mode": "cells", "records": [
        {"kind": "fdt", "id": real.id, "name": "real_cell"}, "20260101T000000", None,
        {"kind": "fdt", "name": "no_id_here"}]})
    assert f"\n    fdt real_cell [{real.id}]\n" in text, text
    assert "\n    UNREADABLE entry '20260101T000000' (not a mapping of kind, id and name)\n" in text, text
    assert "\n    UNREADABLE entry None (not a mapping of kind, id and name)\n" in text, text
    assert "\n    UNREADABLE entry {'kind': 'fdt', 'name': 'no_id_here'} (names no id)\n" in text, text
    assert "MISSING" not in text, "nothing here was an id the store could have held"


# ── load_observation's two payload guards (piece 4, §8.1) ────────────────────────────────────────


def _obs_artifact(store, cfg, *, name, x_obs, digest=None):
    """An observation artifact whose payload is exactly the row handed in: the real writer, the real
    body, ``conditioning_block(cfg)`` for the geometry, and by default the row's own digest.

    Cheap ON PURPOSE. ``generate_observations`` simulates a trace and plots it, and neither guard
    below needs one: both fire after ``torch.load`` and before anything is used. ``digest`` records a
    digest OTHER than the row's, which is the only way to reach the digest guard without editing a
    committed manifest.
    """
    from core.Helpers import file_manager
    d = mf.tensor_digest(x_obs) if digest is None else digest
    with store.create("observation", cfg, name=name) as w:
        file_manager.atomic_torch_save({"x_obs": x_obs, "obs_data": torch.zeros(1, 4, dtype=torch.float64),
                                        "t_dim": torch.zeros(1, 4, dtype=torch.float64)},
                                       w.payload("observation.pt"))
        w.fingerprints["x_obs"] = d
        w.body = {"mode": cfg.observation_mode, "conditioning": mf.conditioning_block(cfg),
                  "x_obs_digest": d, "T_obs_cell": 1.0, "n_obs": 4, "forcing_vals": {},
                  "chi_obs_freqs": None, "source": {"kind": "simulated"}}
    return w


def test_load_observation_refuses_a_payload_that_disagrees_with_its_manifest(store):
    """Spec §8.1: the two guards AFTER ``torch.load``, each in isolation.

    Everything above them compares the MANIFEST -- the config's model, parameter order, mode, width and
    chi layout against what the manifest declares -- so a payload that is not the payload the manifest
    describes is invisible to all of them. The two are ordered: the digest first (the bytes are not the
    bytes), then the width (the shape is not the shape). Reaching the second therefore needs a row that
    hashes to what the manifest records, which is why the narrow artifact is written with its own
    digest rather than corrupted after the fact.

    The width guard carries ``field="observation"``, matching the Refusal above it; the digest guard
    carries none. That asymmetry is the spec's ruling, and pinning it is what makes changing it
    deliberate.
    """
    cfg = _nad_cfg()                                      # master.txt declares a drive -> forced mode
    W = int(mf.conditioning_block(cfg)["width"])
    _obs_artifact(store, cfg, name="intact", x_obs=torch.zeros(1, W, dtype=torch.float64))
    good = store.load_observation(cfg, "intact")
    assert good.width == W and int(good.x_obs.shape[-1]) == W, "the control leg: an intact artifact loads"

    # (1) the DIGEST guard. The manifest is untouched and the row on disk is the RIGHT width, so the
    # width guard below cannot be what fires -- only the bytes changed.
    payload = store.path("observation", "intact") / "observation.pt"
    torch.save({"x_obs": torch.ones(1, W, dtype=torch.float64),
                "obs_data": torch.zeros(1, 4, dtype=torch.float64),
                "t_dim": torch.zeros(1, 4, dtype=torch.float64)}, str(payload))
    with pytest.raises(st.StoreError, match="does not hash to the manifest's digest") as e:
        store.load_observation(cfg, "intact")
    assert "the artifact is inconsistent" in str(e.value) and "intact" in str(e.value)
    assert e.value.field is None, "no control answers 'the bytes on disk changed'"

    # (2) the WIDTH guard, reachable only once the digest agrees.
    _obs_artifact(store, cfg, name="narrow", x_obs=torch.zeros(1, W - 1, dtype=torch.float64))
    with pytest.raises(st.StoreError, match=f"holds a {W - 1}-wide conditioning row") as e:
        store.load_observation(cfg, "narrow")
    assert f"manifest declares {W}" in str(e.value)
    assert "compared the MANIFEST, not this row" in str(e.value)
    assert e.value.field == "observation", "the same key as the width Refusal above it"
    assert isinstance(e.value, Refusal), "a StoreError is a Refusal: the front ends route it as one"
