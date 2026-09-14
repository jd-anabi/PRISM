"""Diagnostics: the ``diagnostic`` store kind, the shared feature-set and seeding helpers, and the
five diagnostic functions (piece 2 of the 2026-09-11 one-flow design, tasks T13-T18).

Everything here is CPU-sized and stubbed where a simulation would otherwise dominate: a diagnostic is
a measurement ABOUT a trained posterior, so the thing worth pinning is what it reads, what it refuses
and what it writes -- not the numbers a real training run would give it.
"""
import pytest

from core.artifacts import LoadedDiagnostic
from core.artifacts import store as st
from tests._fixtures import _nad_cfg, _posterior_artifact


def _diagnostic(store, cfg, *, name="diag", parents=None, variant=None):
    """A diagnostic artifact written the way every diagnostic function writes one (spec 4.1)."""
    with store.create("diagnostic", cfg, name=name) as w:
        w.parents = dict(parents or {})
        w.config.update({"repeats": 2})
        w.body = {"diagnostic": "sbc", "variant": variant, "settings": {"repeats": 2},
                  "results": {"n_valid": 8, "accepted": []}}
    return w


def test_a_diagnostic_loads_without_a_config_and_blocks_deleting_what_it_names(store):
    """D5, the whole contract of the kind: its own directory and manifest body; a loader that takes a
    ref and NOTHING else, because there is no config to verify a measurement against and nothing is
    ever trained from one; and, because ``dependents`` walks every kind, the artifacts it names cannot
    be deleted out from under it without ``force``.
    """
    import inspect
    cfg = _nad_cfg()
    post = _posterior_artifact(store, cfg, name="p")
    w = _diagnostic(store, cfg, name="sbc_rep", parents={"posterior": post.id}, variant="k4")

    assert store.kind_dir("diagnostic") == store.root / "diagnostics"
    assert w.dir.parent.name == "diagnostics" and w.dir.name == f"sbc_rep__{w.id}"

    assert list(inspect.signature(store.load_diagnostic).parameters) == ["ref"], \
        "load_diagnostic takes a ref only: no cfg to check against, no Accept to pass"
    d = store.load_diagnostic("sbc_rep")
    assert isinstance(d, LoadedDiagnostic) and d.kind == "diagnostic" and d.id == w.id
    assert (d.diagnostic, d.variant) == ("sbc", "k4")
    assert d.settings == {"repeats": 2} and d.results == {"n_valid": 8, "accepted": []}
    assert store.load_diagnostic(w.id).id == w.id                 # by id as well as by name
    with pytest.raises(st.StoreError, match="no complete diagnostic"):
        store.load_diagnostic("nope")

    assert store.dependents("posterior", post.id) == [("diagnostic", w.id, "sbc_rep")]
    with pytest.raises(st.StoreError, match="diagnostic sbc_rep"):
        store.delete("posterior", post.id)
    store.delete("posterior", post.id, force=True)
    assert store.load_diagnostic("sbc_rep").id == w.id            # the measurement itself survives
