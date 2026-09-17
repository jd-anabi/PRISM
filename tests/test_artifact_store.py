"""The artifact store (piece 1 of the 2026-09-10 hardening programme): manifest schema, provenance,
the store and its writer, the loaders' refusals, the simulation identity, and the stage contract."""
import json
import math
import os
import tempfile
from pathlib import Path

import pytest
import torch

from core import config
from core.artifacts import manifest as mf
from core.artifacts import provenance as prov
from core.refusals import Refusal

from tests._fixtures import (CODE_FILES, CODE_ROOTS, _FakeDP, _gmm_in_box, _nad_cfg,
                             _posterior_artifact, _prior_artifact, _set_path, assert_cfg_unchanged,
                             snapshot_cfg)


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


def _make(store, kind="calibration", name="", body=None, parents=None, note=""):
    with store.create(kind, None, name=name, note=note) as w:
        w.body = body if body is not None else _cal_body()
        w.parents = dict(parents or {})
        p = w.payload("results.json")
        p.write_text("{}", encoding="utf-8")
    return w


def test_create_list_get_round_trip_per_kind(store):
    bodies = {
        "prior": {"gmm": {"n_components": 2, "param_keys": ["a"], "box": {"nd_lows": [0.0], "nd_highs": [1.0], "log_mask": [False]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}},
        "posterior": {"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                      "truncation": None, "training": {}},
        "observation": {"mode": "chi", "conditioning": {"width": 50}, "x_obs_digest": "0" * 16, "T_obs_cell": 1.0,
                        "n_obs": 10, "forcing_vals": {}, "chi_obs_freqs": None, "source": {"kind": "simulated"}},
        "calibration": _cal_body(), "inference": {"results": {"n_samples": 5}},
        "diagnostic": {"diagnostic": "sbc", "variant": None, "settings": {"repeats": 2},
                       "results": {"n_valid": 8}},
    }
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


def test_an_input_file_that_vanished_during_the_run_is_recorded_unhashed_not_lost(store, tmp_path):
    """The commit runs at the END of a run that may have taken days. An input file moved or edited
    meanwhile must be recorded as unhashed (and warned about), never raise out of the writer -- losing
    a finished run to a renamed cell file would be the worst possible failure mode of provenance."""
    cfg = _nad_cfg()
    gone = tmp_path / "vanished_cell.txt"
    gone.write_text("not a real cell", encoding="utf-8")
    cfg.sources["cell"] = str(gone)
    gone.unlink()
    with pytest.warns(UserWarning, match="vanished_cell"):
        with store.create("calibration", cfg) as w:
            w.body = _cal_body()
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


def test_a_non_amortized_posterior_needs_accept_and_the_flag_is_recorded(store):
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
    .claude on purpose); the gitignored sbi-logs/ (a log directory; holds no *.py, listed for the
    same reason as the rest below); and .git/, .pytest_cache/, __pycache__/, Artifacts/,
    Resources/ (data, caches, VCS or tool state, not product code). .idea/, .superpowers/ and
    docs/ hold no *.py today so they need no explicit exclusion; they fall out of `with_py` on
    their own, and would have to be added here (or to CODE_ROOTS) the day one of them gained a
    Python file. Loose top-level files need no exclusion set: every *.py at the repo root must be
    in CODE_FILES."""
    root = Path(__file__).resolve().parents[1]
    skip = {"tests", "archive", ".claude", ".git", ".pytest_cache", "sbi-logs", "Artifacts",
            "Resources", "__pycache__"}
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
        if name == "taken":
            raise st.StoreError(f"A {kind} named 'taken' already exists.")

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
# "success" returns; "refusal" raises a ValueError before any spend (a Refusal from Tasks 5-8 on is a
# ValueError too); "boom" raises _Injected after a write on the stage's working config.
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
}

# Task 7's own deferred finding (flagged at dispatch): the "refusal" leg below used to accept any
# ValueError, so a leg's refusal could silently regress to a different one and the pin would not
# notice. Named here only for the entries THIS task turned into a Refusal with a field -- the pre-spend
# numeric knobs at build_prior, build_posterior, validate_calibration and tsnpe_round, and the shared
# _refuse_no_samples every n_samples=0 leg (infer_and_visualize, simulated_inference,
# experimental_inference) now raises through. Every other leg's refusal is untouched by this task and
# keeps the bare ValueError check.
_REFUSAL_FIELDS = {
    "build_prior": "num_iterations",
    "build_posterior": "num_runs",
    "validate_calibration": "n_cal",
    "infer_and_visualize": "n_samples",
    "simulated_inference": "n_samples",
    "experimental_inference": "n_samples",
    "tsnpe_round": "n_directions",
}


def test_the_fifteen_public_entries_carry_public_entry_and_nothing_else_does():
    """V1 (spec §2.2). The private copy is kept by ONE decorator on exactly fifteen functions: the ten
    stages and compositions of core/orchestrator.py and the five diagnostics. Read off the source
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
    "refusal" is the entry's own pre-spend refusal (a ValueError today, a Refusal from Tasks 5-8 on,
    which is a ValueError); "boom" raises _Injected AFTER a write on the stage's working config -- the
    stage's own write where it has one before a cheap seam (generate_observations' resolved length,
    install's context, the chi builder's context), else _leak's at the first place the stage hands its
    config on (the store, a composed stage's stub). Every leg is stubbed below its first write, so the
    forty-five cases cost about two seconds, nearly all of it generate_observations' one real solve.

    The window-level pin (a dispatched simulated inference leaves session.cfg equal to its snapshot)
    needs the real-session fixture and lands with Task 15."""
    watched, call = _UNTOUCHED_LEGS[entry](case, monkeypatch, tmp_path)
    snap = snapshot_cfg(watched)
    if case == "success":
        assert call() is not None
    elif case == "refusal":
        with pytest.raises(ValueError) as e:
            call()
        if entry in _REFUSAL_FIELDS:
            # Tightened by Task 7 (a deferred Task 3 finding): this leg's refusal is now a Refusal
            # carrying the field its own rule names, not just some ValueError or other.
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
