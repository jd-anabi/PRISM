"""Shared base classes and budget machinery for the inference tabs."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from core import config, orchestrator
from core.artifacts.provenance import _nvidia_smi_free_gib  # noqa: F401 -- re-exported for inference_tabs
from core.config import CELL_PATH

from ..base_panel import BasePanel
from ...fields import label
from ...widgets.artifact_picker import ArtifactPicker


class _StagePanel(BasePanel):
    """Common base for the five inference tabs: holds a back-reference to the owning InferenceScreen and
    reads/writes the shared session through it (never caching the session object, which Config replaces
    wholesale on each build)."""

    def __init__(self, screen, parent=None):
        super().__init__(parent)
        self._screen = screen

    @property
    def session(self):
        return self._screen.session



class _CellPreviewMixin:
    """Cell-picker handling for the Infer tab (its only user): the cell folder follows the BUILT
    config's model (there is no live model combo in that tab), so the picker is repointed in
    on_config_built and the saved key re-applied there (it could not resolve at __init__, before
    any config exists)."""

    def _init_cell_picker(self):
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self._saved_cell_key = ""

    def on_config_built(self, cfg):
        self.cell_picker.repoint(CELL_PATH / cfg.model.lower(), self._saved_cell_key)



def _hw_batch(cfg) -> int:
    """The rows-per-batch the run will actually use when the cap field is 0 (= auto)."""
    return (getattr(getattr(cfg, "hw", None), "batch_size", None)
            or config.detect_device().batch_size)


class _TrainingBudgetMixin:
    """Batches x rows-per-batch, and the three derived lines that say what it will cost.

    EXTRACTED rather than duplicated when the TSNPE tab arrived (each round is a simulation
    campaign, not a click, so its cost belongs on screen). A second copy of _budget_memory would
    have been a second place for pipeline's cost model to be restated wrongly, and the whole
    point of that method is that it reads the planner's own numbers instead of restating them.

    A user of this mixin must create ``num_runs``, ``run_size_cap`` (IntFields with
    ``.value_or_none()``) and the three ``budget_*`` labels, then call ``_sync_budget()``. A blank
    or out-of-rule box is SAID on the total line (see _budget_problem), never clamped or defaulted;
    the click refuses it through the same two rules.
    """
    @staticmethod
    def _derived_label() -> QLabel:
        """A read-only derived line under the budget fields.

        Word-wrapped and PlainText on purpose: these carry generated strings that can be long (a
        checkpoint directory name and the field it differs in), and an unwrapped label widens the
        whole controls column -- the same unwrapped-label defect that once put a permanent
        horizontal scrollbar on the crossval panel.
        """
        lab = QLabel()
        lab.setWordWrap(True)
        lab.setTextFormat(Qt.PlainText)
        return lab

    def _hardware(self):
        """The DeviceConfig a config would be built with. Memoised: detect_device() probes the
        accelerator and _sync_budget runs on every keystroke in the two fields."""
        if getattr(self, "_hw_cache", None) is None:
            self._hw_cache = config.detect_device()
        return self._hw_cache

    def _budget_values(self) -> tuple:
        """(batches, rows-per-batch cap) as typed, or None where a box is blank or half-typed.

        value_or_none(), never value(): value() reads "" and a lone "-" as 0, and 0 is a legal cap
        (automatic) but a refusal for the batch count, so the two must be told apart. Neither a
        clamp nor a default -- the click refuses a None through the shared rules, and the live lines
        name the box (V2)."""
        return self.num_runs.value_or_none(), self.run_size_cap.value_or_none()

    def _budget_problem(self) -> "str | None":
        """The sentence the total line shows INSTEAD of a number while a budget box is blank or out
        of rule; None when both pass. The same two rules the click runs (batches at least 1, the cap
        at least 0), said in the box's own label so the line and the yellow box agree on the name."""
        n_runs, cap = self._budget_values()
        if n_runs is None:
            return f"{label('num_runs')} is blank."
        if n_runs < 1:
            return f"{label('num_runs')} must be at least 1."
        if cap is None:
            return f"{label('run_size_cap')} is blank."
        if cap < 0:
            return f"{label('run_size_cap')} must be at least 0."
        return None

    def _sync_budget(self) -> None:
        """Recompute the three derived lines. Pure and cheap -- safe on every keystroke.

        A BLANK OR OUT-OF-RULE BOX IS SAID, NOT REPAIRED. This used to read value() -- 0 for "" and
        for a lone "-" mid-typing -- and then max(1, ...) / max(0, ...) the result, so a blank Batches
        box showed the budget for ONE batch and a negative cap the hardware width, both as if someone
        had typed them. A live line cannot pop a dialog, so it names the box and shows no number until
        it is fixed; the memory and checkpoint lines go empty with it, because an estimate for a
        width nobody asked for is the same lie in GiB. No clamp, no default, no raise.

        ONLY FORMATS (V6). The width, the memory geometry, the cache directory and the cadence are
        orchestrator.training_preview's, which resolves them as build_posterior does -- this used to
        derive each of them itself, and the cadence from config's live copy, which a run never reads.

        ⚠ Passes `session.cfg` through as it is, never trusting it to be a SimConfig: the gate tests set
        it to a bare `object()` sentinel, and a derived STATUS LINE must never be able to raise into
        refresh_gates() and take the whole tab down with it. The preview never raises -- a config it
        cannot read comes back as estimate_error / checkpoint_error, shown as "unavailable" -- and the
        formatters below read the preview, plus the config's chi flag through getattr.
        """
        problem = self._budget_problem()
        if problem is not None:
            self.budget_total.setText(problem)
            self.budget_mem.setText("")
            self.budget_ckpt.setText("")
            return
        cfg = self.session.cfg
        n_runs, cap = self._budget_values()
        # ONE call, then formatting only. hw= is the memoised detect_device() for the launch-time case
        # (no config yet); a config's own hw wins inside the preview.
        preview = orchestrator.training_preview(cfg, self.session.inf_prior, num_runs=n_runs,
                                                run_size_cap=cap, hw=self._hardware())
        self.budget_total.setText(self._budget_total(preview, cfg))
        self.budget_mem.setText(self._budget_memory(preview, cfg))
        self.budget_ckpt.setText(self._budget_checkpoint(preview))

    def _budget_total(self, preview, cfg) -> str:
        """The total line: simulations = batches x rows, the hardware batch the cap binds against, and
        what the batch COUNT buys. The chi note is the one thing read off the config itself, through
        getattr, because no config (launch) and a stub have no chi flag."""
        p = preview
        capped = "" if p.width == p.hw_batch else f" (capped from {p.hw_batch:,})"
        chi = ""
        if getattr(cfg, "chi_mode", False):
            chi = (f"\nIn chi mode each row costs 1+K solver passes, K up to "
                   f"{getattr(cfg, 'chi_k_pad', '?')}.")
        return (f"{p.n_runs * p.width:,} simulations = {p.n_runs:,} batches x {p.width:,} rows{capped}."
                f"\nBatches is also the (t_scale, T) diversity count: every row in a batch shares one "
                f"operating point, so batch COUNT is the statistics and batch WIDTH is not.{chi}")

    def _budget_memory(self, preview, cfg) -> str:
        """Peak device memory for ONE batch at the worst geometry the Sobol pre-filter admits.

        The WORST case is the one worth showing: n_fine swings from a median ~40k to a p99 ~283k, so a
        width that fits the median still OOMs on a few percent of batches -- which is how the
        2026-08-10 and 2026-08-11 retrains both died. gen_training_data rejects any (t_scale, T) whose
        n_fine exceeds min(N_ND_MAX, len(t)), so that IS the ceiling, by construction.

        Formats orchestrator.training_preview, which reads pipeline's own cost model
        (peak_sim_elements / sim_memory_budget_elements) at the training stage's own width, so this
        cannot drift from what the planner does. ``cfg`` decides one thing only: whether the geometry
        was the hardware defaults because no config is built yet.
        """
        p = preview
        if p.device_type != "cuda":
            return f"Peak-memory estimate is CUDA-only; this config runs on {p.device_type}."
        if p.estimate_error is not None:
            return f"Peak-memory estimate unavailable: {p.estimate_error}"
        gib = p.itemsize / float(1 << 30)
        verdict = ("fits in one piece" if p.need_elements <= p.have_elements else
                   "does NOT fit -- the planner will split this batch, costing wall-clock")
        return (f"Worst-case peak ~{p.need_elements * gib:.2f} GiB per batch (n_fine <= {p.n_fine:,}, "
                f"{p.n_vars} state vars); planner budget right now ~{p.have_elements * gib:.2f} GiB, "
                f"so it {verdict}.\n"
                f"That budget is an UPPER bound: free-VRAM readings overstate what is really "
                f"available by roughly the size of the desktop, so closing browsers is a bigger lever "
                f"than lowering the cap." + ("" if cfg is not None else
                                             "\n(Estimated from hardware defaults until a config is built.)"))

    def _budget_checkpoint(self, preview) -> str:
        """THE GUARD THAT MAKES THESE TWO FIELDS SAFE TO EXPOSE AT ALL.

        Both are inside the training-checkpoint identity, which is DIGESTED into the checkpoint's
        directory name. Change either and a resumable multi-day run silently resolves to a different
        directory -- there is no error, just a run that starts from zero. A tooltip is not a strong
        enough guard for that, so the state is stated inline and re-evaluated as you type;
        describe_siblings even names the field that differs.

        Formats the preview's verdict, in its order of precedence. The directory is the one the run
        will use (the store's simulations directory) and "off" is the training stage's own cadence --
        so the line and what Train then does cannot disagree. The "off" sentence keeps the words
        config.TRAINING_CHECKPOINT_EVERY because config.py is where the operator changes it.
        """
        p = preview
        if p.checkpoint == "off":
            return "Checkpointing is off (config.TRAINING_CHECKPOINT_EVERY = 0): a crash loses the run."
        if p.checkpoint == "needs_config":
            return "Checkpoint status needs a config and a prior -- the prior is part of the identity."
        if p.checkpoint_error is not None:
            return f"Checkpoint status unavailable: {p.checkpoint_error}"
        if p.checkpoint == "resume_complete":
            return (f"Resumes a COMPLETE checkpoint ({p.batches_done:,} batches) -- simulation will be "
                    f"skipped entirely and only the flow retrained.")
        if p.checkpoint == "resume_partial":
            return f"Resumes an existing checkpoint: {p.batches_done:,}/{p.n_runs:,} batches already done."
        if p.checkpoint == "new_with_siblings":
            return ("WARNING: these settings match no checkpoint, so this starts a NEW run.\n"
                    + p.siblings)
        return "No checkpoint exists yet; this starts a new run."

