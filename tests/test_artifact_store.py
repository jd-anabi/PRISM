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

from tests._fixtures import _FakeDP, _gmm_in_box, _nad_cfg, _posterior_artifact, _prior_artifact, _set_path


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
    lp = orchestrator.build_prior(cfg, None, True, fig_sink=lambda title, fig: seen.append(title), num_iterations=1)
    assert lp.name == "" and [r.id for r in store.list("prior")] == [lp.id]
    assert (lp.path / "figures" / "prior.png").stat().st_size > 0 and seen == ["Prior"]
    m = lp.manifest
    assert m.body["gmm"]["param_keys"] == list(cfg.params_dict) and m.config["num_iterations"] == 1
    assert m.body["gmm"]["box"]["log_mask"] == nd_log_mask(cfg, log_params=_log_params_for(cfg)).tolist()
    assert m.inputs["bounds"]["path"] == "Bounds/nadrowski/master.txt" and m.fingerprints["gmm"] == lp.fingerprint
    again = orchestrator.build_prior(cfg, lp.id, False, fig_sink=lambda title, fig: None)
    assert again.id == lp.id and again.fingerprint == lp.fingerprint
    store.rename("prior", lp.id, "master_prior")
    assert orchestrator.build_prior(cfg, "master_prior", False, fig_sink=lambda title, fig: None).name == "master_prior"


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
    lp = orchestrator.build_prior(cfg, None, True, fig_sink=lambda title, fig: None, num_iterations=1)
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
    assert ck["parents"] == {"prior": lp.id} and ck["inputs"]["model"] == "NADROWSKI" and ck["hw"] is cfg.hw
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
    from core.SBI import reparam, truncate
    from core.SBI.training_checkpoint import bijection_probe
    cfg = _nad_cfg(chi_mode=True)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    T = reparam.build_inferred_bijection(cfg, log_params=[])
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], n_latent=P, V=None,
                                       probe=bijection_probe(T, P), x_obs_digest="d" * 16)
    _posterior_artifact(store, cfg, name="trunc", amortized=False, region=region)
    with pytest.raises(ValueError, match="NOT AMORTIZED"):
        store.load_posterior(cfg, "trunc")
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
    out = orchestrator.build_posterior(cfg, lp, None, True, fig_sink=lambda t, f: seen.append(t),
                                       num_runs=2, run_size_cap=4, hidden_features=8, num_transforms=1,
                                       stop_after_epochs=1, note="tiny")
    assert isinstance(out, st.LoadedPosterior) and out.name == "" and out.diagnostics["epochs_trained"] == 2
    m = store.get("posterior", out.id)
    assert m.parents == {"prior": lp.id} and m.note == "tiny" and m.body["amortized"] is True
    assert m.body["transform"]["V"] is None and m.body["transform"]["param_keys"][-1] in cfg.rescale_params
    assert m.body["training"]["hidden_features"] == 8 and m.body["training"]["best_validation_loss"] == 0.6
    assert m.config["num_runs"] == 2 and m.fingerprints["gmm"] == lp.fingerprint
    assert m.config["fisher_m"] == config.REPARAM_FISHER_M, "an unpassed fisher_m must record the value used"
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
        orchestrator.build_posterior(cfg, lp, None, True, fig_sink=lambda t, f: None, num_runs=2, run_size_cap=4,
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
    obs = orchestrator.generate_observations(cfg, fig_sink=lambda t, f: seen.append(t), name="cell")
    assert obs.name == "cell" and obs.mode == "forced" and obs.width == obs.x_obs.shape[-1]
    assert seen == ["Ground-truth trace"] and obs.manifest.figures == ["figures/ground_truth_trace.png"]
    src = obs.manifest.body["source"]
    assert src["kind"] == "simulated" and src["cell"]["path"] == "Cells/nadrowski/master_weak.txt"
    assert src["params"] == {k: float(v) for k, (v, _) in cfg.params_dict.items()}
    assert obs.manifest.fingerprints["x_obs"] == obs.digest == mf.tensor_digest(obs.x_obs)
    fresh = _nad_cfg()
    assert not fresh.has_ground_truth
    store.load_observation(fresh, obs.id).install(fresh)
    assert fresh.has_ground_truth and fresh.T_obs == cfg.T_obs and fresh.n_obs == cfg.n_obs
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
    sim = orchestrator.generate_observations(cfg, fig_sink=lambda t, f: None)
    trace = sim.obs_data[0].numpy()
    spont, forced = tmp_path / "spont.npy", tmp_path / "forced.npy"
    np.save(spont, trace)
    np.save(forced, trace)
    T_obs_s = cfg.T_obs / cfg.get_unit_conversion_factor("s")
    si = {n: (5.0 if n == "freq" else 1e-12 if n == "amp" else 0.0) for n in cfg.force_params_dict}
    rec = RecordingSet(spont=str(spont), forced=((str(forced), None),), T_obs_s=T_obs_s, forcing_params_si=si)
    obs = orchestrator.build_experiment_observation(cfg, rec, fig_sink=lambda t, f: None)
    recs = obs.manifest.body["source"]["recordings"]
    assert [r["role"] for r in recs] == ["spont", "forced"] and all(len(r["sha256"]) == 64 for r in recs)
    assert obs.width == sim.width and obs.manifest.body["source"]["kind"] == "experimental"
    assert obs.manifest.body["forcing_vals"]["freq"] > 0
    with pytest.raises(FileNotFoundError):
        orchestrator.build_experiment_observation(
            cfg, RecordingSet(spont=str(tmp_path / "nope.npy"), forced=((str(forced), None),), T_obs_s=T_obs_s,
                              forcing_params_si=si), fig_sink=lambda t, f: None)


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
        with pytest.raises(ValueError, match="NOT AMORTIZED"):
            orchestrator.infer_and_visualize(r.cfg, claims_another, obs, fig_sink=r.sink, n_samples=20)
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
    comments are not Path() calls or `/` operands, so they do not trip this."""
    import ast
    root = Path(__file__).resolve().parents[1]
    roots = ("Resources/", "Resources\\", "Artifacts/", "Artifacts\\", "Resources", "Artifacts")

    def _lit(node):
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(roots)

    offenders = []
    for py in sorted((root / "core").rglob("*.py")):
        if py == root / "core" / "config.py":
            continue
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            hit = False
            if isinstance(node, ast.Call):
                fn = node.func
                fname = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                hit = fname in ("Path", "join") and node.args and _lit(node.args[0])
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                hit = _lit(node.left) or _lit(node.right)
            if hit:
                offenders.append(f"{py.relative_to(root)}:{node.lineno}")
    assert not offenders, "literal store paths outside config.py:\n" + "\n".join(offenders)
    for name in ("PRIOR_PATH", "POSTERIOR_PATH", "PLOT_PATH", "CHECKPOINT_PATH", "OBSERVATION_PATH"):
        assert not hasattr(config, name), f"config.{name} still exists"


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
    out = orchestrator.build_posterior(cfg, lp, None, True, fig_sink=lambda t, f: None,
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
    with pytest.raises(ValueError, match="resume"):
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4,
                                     checkpoint_every=1, resume="yes please")
    with pytest.raises(ValueError, match="checkpointing on"):
        orchestrator.build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4,
                                     checkpoint_every=0, resume="require")
    assert spent == [], "a refused resume policy started the run anyway"
    assert _inspect.signature(orchestrator.build_posterior).parameters["resume"].default == "auto"


def test_the_budget_line_prints_with_default_arguments(store, monkeypatch, capsys):
    """Guardrail 6 on the command line: what this run will actually simulate, said once, whether or
    not a cap or a batch count was overridden. The two conditional announcements stay -- with the
    defaults neither of them fires, which is precisely the run whose size used to be invisible."""
    from core import orchestrator
    cfg = _nad_cfg()
    cfg.reparam_rotate = False
    cfg.hw.batch_size = 4
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)

    def stop(*a, **k):
        raise RuntimeError("stop before training")

    monkeypatch.setattr(orchestrator.pipeline, "train_nn", stop)
    with pytest.raises(RuntimeError, match="stop before training"):
        orchestrator.build_posterior(cfg, lp, None, True)
    out = capsys.readouterr().out
    n = config.TRAINING_NUM_RUNS
    assert f"[budget] {n:,} batches x 4 rows = {n * 4:,} training rows" in out
    assert "capped at" not in out and "COUNT overridden" not in out, (
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
