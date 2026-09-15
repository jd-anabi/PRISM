"""GPU-only paths of the stage contract (piece 1): what the CPU suites cannot see.

Every other suite runs on the CPU, where ``.to(device)`` is a no-op and a tensor on the wrong device
is invisible. The 2026-09-11 GPU gate (``python -m core smoke``) failed in the infer stage with a
CUDA observation row meeting CPU simulated statistics inside the posterior predictive check -- after
prior, posterior and validate had passed. This module is the regression test for that class: the
tiny SBITEST run of ``tests/_fixtures.build_tiny_run`` on ``config.detect_device()``.

``gpu``-marked: the root conftest skips it when CUDA is absent. It does not replace the smoke run
(``docs/STATE.md``), which exercises the real bounds/cell files and the checkpoint resume.
"""
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def tiny_run_cuda(tmp_path_factory):
    """build_tiny_run on the detected device, in a store of its own that is the process default for
    the module. Skips (rather than building on the CPU) when the detected device is not CUDA."""
    from core import config
    from core.artifacts import ArtifactStore, use_store
    from tests._fixtures import build_tiny_run
    hw = config.detect_device()
    if hw.device.type != "cuda":
        pytest.skip("needs CUDA")
    with use_store(ArtifactStore(tmp_path_factory.mktemp("tiny_cuda"))) as s:
        run = build_tiny_run(s, hw=hw)
        try:
            yield run
        finally:
            run.teardown()


def test_inference_runs_on_cuda_with_a_loaded_observation(tiny_run_cuda):
    """generate_observations writes the row from the CPU and the loader hands it back; the PPC then
    compares it with conditioning rows that statistics.conditioning_rows assembles on the CPU by
    contract. On CUDA the two met on different devices (orchestrator.infer_and_visualize, 2026-09-11)."""
    from core import orchestrator
    r = tiny_run_cuda
    assert r.cfg.hw.device.type == "cuda"
    obs = orchestrator.generate_observations(r.cfg, fig_sink=r.sink, name="obs")
    inf = orchestrator.infer_and_visualize(r.cfg, r.posterior, obs, fig_sink=r.sink, n_samples=20, name="inf")
    assert inf.manifest.parents == {"posterior": r.posterior.id, "observation": obs.id}
    assert inf.results["n_samples"] == 20 and inf.results["accepted"] == []
