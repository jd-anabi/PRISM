"""GPU-only paths of the stage contract: what the CPU suites cannot see.

Every other suite runs on the CPU, where ``.to(device)`` is a no-op and a tensor on the wrong device
is invisible. The 2026-09-11 GPU gate (``python -m core smoke``) failed in the infer stage with a
CUDA observation row meeting CPU simulated statistics inside the posterior predictive check -- after
prior, posterior and validate had passed. This module is the regression test for that class: the
tiny SBITEST run of ``tests/_fixtures.build_tiny_run`` on ``config.detect_device()``.

``gpu``-marked: the root conftest skips it when CUDA is absent. It does not replace the smoke gate
on the card (``python -m core smoke`` run on the GPU), which exercises the real bounds/cell files
and the checkpoint resume.
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


def _stop_after(patch, at: int, exc: BaseException) -> None:
    """Stop the training with ``exc`` right after pass ``at`` is saved, as a Ctrl-C between two passes
    would: the pass's save runs whole (its state, its record and its line), then ``exc`` is raised where
    the trainer hands the pass on."""
    from core.SBI import fit_state
    real = fit_state.FitSession.save_pass

    def save_pass(self, take):
        real(self, take)
        if self.passes == at:
            raise exc

    patch.setattr(fit_state.FitSession, "save_pass", save_pass)


def test_a_training_stopped_on_the_card_resumes_with_its_split_counters_and_optimizer_on_the_device(
        tiny_run_cuda, monkeypatch):
    """A training on the card, its rows in a kept simulation set, stopped by a Ctrl-C after its second
    pass is saved, is resumed in a new call with nothing but the kept training named. When sbi's training
    method is called to continue it, every parameter has its moments back in the optimizer, on the card,
    and every weight of the network is on the card with its kept value (the kept state is read onto the
    processor and must be moved back), the pass count, the history of the two passes and the
    early-stopping counters are the kept ones, and the validation rows are the ones the training was split
    into. The posterior is written, trained past the kept passes, and says it was resumed over two
    processes. Its losses are not compared with an uninterrupted fit's: on the card the spline's gather is
    not deterministic."""
    import numpy as np
    import torch
    from sbi.inference import SNPE
    from core import orchestrator
    from core.SBI import fit_state
    r = tiny_run_cuda
    assert r.cfg.hw.device.type == "cuda"
    with monkeypatch.context() as patch:
        _stop_after(patch, 2, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=r.sink, store=r.store,
                                         name="card_resumed", num_runs=3, run_size_cap=8, hidden_features=8,
                                         num_transforms=1, stop_after_epochs=3, max_num_epochs=8,
                                         checkpoint_every=1, new_run=True, fisher_points=1, fisher_m=4)
    row, = r.store.list("fit")
    assert (row.fit_state, row.fit_passes) == ("kept when stopped", 2)
    kept_validation = fit_state.read_header(row.path)["start"]["validation_digest"]
    kept = fit_state.load_state(row.path)

    resumed, real = [], SNPE.train

    def train(self, *args, **kwargs):
        if kwargs.get("resume_training"):
            # Checked here, before sbi's own passes run: a fault these checks name would otherwise end the
            # fit first with an error of torch's own (an optimizer step across two devices).
            resumed.append(True)
            params = list(self._neural_net.parameters())
            moments = [self.optimizer.state[p]["exp_avg"].device.type for p in params if p in self.optimizer.state]
            assert len(moments) == len(params) and set(moments) == {"cuda"}, f"the optimizer's moments: {moments}"
            weights = [p.device.type for p in params]
            assert set(weights) == {"cuda"}, f"the network's weights are not all on the card: {weights}"
            changed = [name for name, t in self._neural_net.state_dict().items()
                       if not torch.equal(t.detach().cpu(), kept["net_state"][name].cpu())]
            assert changed == [], f"the kept weights were not put back: {changed}"
            # The kept state was taken at the second pass's summary, before sbi counted the pass, and holds
            # the number sbi gives the next pass; counted from nought, that is the third pass's number, 2,
            # with the two passes trained before the stop in sbi's own history.
            assert (self.epoch, len(self._summary["validation_loss"])) == (2, 2), (self.epoch, self._summary)
            assert (float(self._best_val_loss), self._epochs_since_last_improvement) == \
                (float(kept["best_val_loss"]), kept["passes_since_best"]), "the early-stopping counters"
            assert fit_state.indices_digest(self.val_indices) == kept_validation, "the split is not the kept one"
        return real(self, *args, **kwargs)

    monkeypatch.setattr(SNPE, "train", train)
    post = orchestrator.build_posterior(r.cfg, r.prior, None, True, resume_fit=row.id, fig_sink=r.sink,
                                        store=r.store)
    assert resumed == [True], "sbi's training method was never called to resume the kept training"
    assert post.name == "card_resumed"
    validation_loss = np.load(post.path / "loss.npz")["validation_loss"]
    # the two kept passes' losses, as the history showed at the resume, and at least one pass trained here
    assert len(validation_loss) > 2 and np.isfinite(validation_loss).all(), validation_loss
    training = post.manifest.body["training"]
    assert (training["resumed_fit"], training["fit_processes"], training["fit_id"]) == (True, 2, row.id)
    assert r.store.list("fit") == []
