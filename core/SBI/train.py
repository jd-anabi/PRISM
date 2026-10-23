"""The sbi training call, split out of pipeline.py (which stays the public facade).

Every consumer -- orchestrator.build_posterior, the test suites --
reaches these as ``pipeline.train_nn`` / ``pipeline._capped_zscore_check``: pipeline.py re-imports
them at its bottom, which also keeps monkeypatching ``pipeline.<name>`` effective. Calls back into
the generation machinery go through the pipeline MODULE OBJECT at call time (``_pipeline.``), never
``from`` imports, so a test that rebinds ``pipeline.gen_training_data`` or
``pipeline.winsorize_summary_block`` is honoured here too.

The training call comes in two halves, so that a fit kept partway can be put back between them:
``prepare_fit`` appends one round's rows to the trainer, ``run_fit`` runs sbi's own passes. Every
pass ends in the trainer's per-pass summary, which is wrapped: with a saver the fit is handed to it
there, and then the pass is said in one record (``fit_epoch_line``), after its save.
``restore_trainer`` puts a kept pass back onto a fresh trainer in a new process. sbi's private
bookkeeping is read and written here and nowhere else in the program.
"""
import contextlib
import inspect
import logging
import math
import sys
import time
import typing
import warnings
from typing import Callable, Sequence

import torch
from tqdm import tqdm
from sbi.inference.posteriors import DirectPosterior
from sbi.inference import SNPE
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.net_builders import flow as _sbi_flow
from torch.distributions.transforms import Transform

from core.SBI import fit_state
from core.SBI import pipeline as _pipeline

log = logging.getLogger(__name__)


import dataclasses


@dataclasses.dataclass
class TrainingPlan:
    """Everything ``gen_training_data`` is driven with, typed -- replacing the ~27-key dict whose
    every consumer did its own ``.get(...)`` defaulting. The defaults now live HERE, once.

    ``chi_n_freqs`` is deliberately ABSENT: it is the count an OBSERVATION supplies, and training
    draws K per batch over [CHI_K_MIN_TRAIN, chi_k_pad] then subsets per row -- which is what lets
    one posterior serve any probe count. It used to be threaded in and silently ignored, an
    invitation to "fix" gen_training_data into honouring it and destroy exactly that property.

    ``chi_k_fixed`` belongs to the ``sbc`` diagnostic: ``orchestrator.build_posterior`` never sets
    it; the stratified SBC path does (one probe count per calibration stratum). ``checkpoint`` stays
    a plain dict -- its key set (dir/identity/probe/V/fisher_eigenvalues/every/resume, plus the
    parents/inputs/hw the cache's own manifest records) is the resumable-checkpoint contract, and
    ``gen_training_data``'s own signature is unchanged.
    """
    model: str
    prior: object
    t: torch.Tensor
    run_size: int
    num_runs: int
    steady_idx: int
    dt_nd_min: float
    dt_exp: float
    t_min_exp: float
    t_max_exp: float
    t_scale_bounds: tuple
    dtype: torch.dtype
    device: torch.device
    state_dep_drift: bool = False
    spontaneous_only: bool = False
    chi_mode: bool = False
    chi_f0: float = None
    chi_freq_bounds: tuple = None
    chi_k_pad: int = None
    chi_k_fixed: int = None
    chi_max_cycles: float = None
    n_vars: int = None
    nd_idx: dict = None
    k_b_cell: float = None
    checkpoint: dict = None


import sbi
import sbi.utils.sbiutils as _sbiutils

# sbi's check on the training data before it z-scores it, which append_simulations calls on the WHOLE
# training tensor. Resolved when this module is IMPORTED, so an sbi that renamed it again (0.26 replaced
# warn_if_zscoring_changes_data with this one) fails when the pipeline loads, before any simulation, not
# at the end of a generation run.
_ZSCORE_CHECK_NAME = "warn_if_invalid_for_zscoring"
_ZSCORE_CHECK = getattr(_sbiutils, _ZSCORE_CHECK_NAME, None)
if _ZSCORE_CHECK is None:
    raise ImportError(
        f"sbi {sbi.__version__} has no sbi.utils.sbiutils.{_ZSCORE_CHECK_NAME}, the check PRISM caps "
        f"before a training run; install the sbi requirements.txt pins.")

# A strided subsample of at most this many rows. Far under the 2**24 rows torch.quantile accepts along
# the reduced dimension, which the full 20.48M-row retrain table exceeds.
_ZSCORE_CHECK_MAX_ROWS = 200_000


@contextlib.contextmanager
def _capped_zscore_check(max_rows: int = _ZSCORE_CHECK_MAX_ROWS, *, skip: bool = False):
    """Run sbi's z-scoring check on a SUBSAMPLE for the duration of this block, or not at all.

    WHY IT HAS TO BE CAPPED. append_simulations calls the check UNCONDITIONALLY on the full training
    tensor. In sbi 0.26 it takes a per-column standard deviation, then ``torch.quantile`` over the
    whole table and a full (rows, columns) boolean outlier mask. At the retrain's size -- 10,000 x
    2,048 = 20.48M rows x 122 columns float32, 10 GB -- that is a sorted copy of the table plus several
    GB of temporaries on a host already holding the training data, and torch refuses the quantile
    outright beyond 2**24 rows: a failure at the END of a multi-week generation, when checkpointing
    keeps the rows but the fit never starts. A strided subsample answers the same question at a fraction
    of the cost, though a single outlier the stride skips goes unreported. STRIDED, not a prefix: rows
    are grouped by batch and each batch shares one (t_scale, T) stratum, so a prefix would sample a few
    strata while ``x[::n]`` spans all of them.

    ``skip`` makes it a no-op, for a run whose embedding standardizes its own input: sbi z-scores
    nothing then (``z_score_x="none"``), so the check's constant-column and outlier warnings would
    describe a scaling that never happens. The chi encoder is such an embedding.

    sbi's own arguments are passed on exactly as given: the check's second positional parameter is its
    outlier factor, and forwarding anything else there would flag almost every column.

    PATCHED BY REBINDING IN EVERY MODULE THAT HOLDS THE NAME, not just in sbiutils. npe_base imports the
    function by name, so its call resolves against npe_base's OWN globals -- patching sbiutils alone
    changes nothing. Rather than hard-code that module path (sbi has moved it before), this finds every
    module whose attribute IS the resolved function object. A few ms, once per round, and it cannot rot.

    SCOPED AND RESTORING, including on an exception: the patch is live only across append_simulations,
    so nothing else in the process -- another panel, a script that imported sbi, a later round -- ever
    sees a monkeypatched sbi.
    """
    original = _ZSCORE_CHECK
    rows = min(int(max_rows), 2 ** 24)

    def _replacement(x, *args, **kwargs):
        if skip:
            return None
        stride = max(1, math.ceil(x.shape[0] / rows))
        # .contiguous() explicitly: the check would materialise the strided view anyway, and doing it
        # here keeps the copy's size obvious (<= rows x n_features) rather than implicit.
        return original(x[::stride].contiguous(), *args, **kwargs)

    targets = []
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, _ZSCORE_CHECK_NAME, None) is original:
                targets.append(mod)
        except Exception:            # noqa: BLE001 -- a lazy module's __getattr__ must not break this
            continue
    try:
        for mod in targets:
            setattr(mod, _ZSCORE_CHECK_NAME, _replacement)
        yield
    finally:
        for mod in targets:
            setattr(mod, _ZSCORE_CHECK_NAME, original)


class _NoTracker:
    """The tracker train_nn hands sbi by default: it discards the training curves.

    Given no tracker, sbi's trainer builds a TensorBoard writer under
    ``<cwd>/sbi-logs/NPE_C/<timestamp>/`` when the trainer is CONSTRUCTED (``_default_tracker`` over
    ``sbi.utils.io.get_log_root``). Every training therefore left a directory behind in whatever
    directory the process was started from: 1359 of them had piled up at the repo root (1360 directories
    with NPE_C), none ever read. The curves PRISM keeps are the ones train_nn returns in its
    diagnostics, which build_posterior writes into the posterior artifact.

    This is sbi's ``Tracker`` protocol (``sbi.sbi_types.Tracker``). Its trainer calls ``log_metric`` and
    ``flush``; the rest are there because the protocol names them. sbi's older ``summary_writer``
    argument still works but warns on every construction. Only the trainer's unpickling path
    (``__setstate__``, also reached by a deep copy) still builds sbi's default tracker, and PRISM never
    pickles or copies a trainer. tests/conftest.py::_no_sbi_logs fails the test session if the
    directory comes back.
    """

    log_dir = None

    def log_metric(self, name, value, step=None):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_params(self, params):
        pass

    def add_figure(self, name, figure, step=None):
        pass

    def flush(self):
        pass


# ── the record of each epoch ────────────────────────────────────────────────────────────────────────
# Every pass of a fit (an epoch, on screen) is said in one record once sbi has summarised it and, when the
# fit is saved as it goes, once it is saved: the window and the command line follow a fit by these, and the
# run's log keeps them. sbi reports its curves to a tracker only after its loop has ended, so they are the
# fit's only account while it runs. None of these functions is in the code a kept fit's identity names:
# they only read, and a rewording of the record must not make a kept fit unresumable.
def fit_epoch_line(epoch: int, training_loss: float, validation_loss: float, seconds: float,
                   since_best: int) -> str:
    """The record of one epoch, as the window reads it back: its number, counted from one, its training
    and validation losses to six significant digits, the seconds it took to one decimal, and the epochs
    since the best one."""
    return (f"[fit] epoch {int(epoch)}: training loss {float(training_loss):.6g}, validation loss "
            f"{float(validation_loss):.6g}, took {float(seconds):.1f} s, epochs since the best {int(since_best)}")


def epochs_since_best(validation_losses: Sequence[float]) -> int:
    """The epochs since the best one, read from a fit's validation losses so far, the last entry being the
    epoch just ended: counted back from the last entry to the first least loss that is a finite number, so
    that a tie, or a loss that is not a finite number, is no improvement, as sbi counts its patience. With
    no finite loss, the number of entries, which is what sbi's own count then reads; 0 with none. Read from
    the history and kept nowhere, so it holds in a fit resumed in a new process, whose history comes back
    with it."""
    losses = [float(v) for v in validation_losses]
    finite = [i for i, v in enumerate(losses) if math.isfinite(v)]
    if not finite:
        return len(losses)
    return len(losses) - 1 - min(finite, key=lambda i: losses[i])


def _announce(infer) -> None:
    """Say the pass sbi has just summarised in one record (``fit_epoch_line``), from the trainer's own
    histories: its number (sbi counts the pass only after the summary), its two losses, its time and the
    epochs since the best. It only reads -- it draws no random number and touches no tensor -- so a fit
    computes the same with it or without it. A record that cannot be made is skipped: only an
    ``Exception`` is caught, so a stop raised while it is said passes."""
    try:
        summary = infer._summary
        log.info(fit_epoch_line(infer.epoch + 1, summary["training_loss"][-1], summary["validation_loss"][-1],
                                summary["epoch_durations_sec"][-1], epochs_since_best(summary["validation_loss"])))
    except Exception:                                   # noqa: BLE001 -- a record is never a reason to stop a fit
        pass


def _wrap_summary(infer, saver) -> bool:
    """Whether the fit wraps the trainer's per-pass summary, where each pass is handed to the saver and
    said in a record: True when the trainer has one. A trainer without one (an sbi other than the one
    pinned) cannot save a fit as it goes, so with a ``saver`` it is refused (``_require_hook_points``; a
    trainer built for saving was refused so already); without one the fit trains all the same, and one
    warning says that no record is said per epoch."""
    if callable(getattr(infer, "_summarize_epoch", None)):
        return True
    if saver is not None:
        _require_hook_points(infer)
    log.warning("[fit] this sbi's trainer has no per-pass summary to read, so no line is written per epoch.")
    return False


# ── the fit's settings, as sbi will use them ────────────────────────────────────────────────────────
# The signature of sbi's training method as installed, read once, when this module is imported: the
# defaults PRISM does not pass (the validation fraction, the gradient clip) are read from it. Read here
# and not when a fit is prepared, so that a test which wraps the training method on the class cannot
# change what a preparation records.
_SNPE_TRAIN_SIGNATURE = inspect.signature(SNPE.train)

# Where the rows stay while the network trains (see prepare_fit): one constant, handed to sbi and
# recorded in the fit's settings, so the two cannot drift apart.
_DATA_DEVICE = "cpu"

# The flow builder's own settings PRISM does not pass. A builder that lacks one of them records None
# there, so the settings always carry the same names.
_BUILDER_DEFAULTS = ("tail_bound", "hidden_layers_spline_context", "num_blocks", "dropout_probability",
                     "use_batch_norm")


def _default_of(signature: inspect.Signature | None, name: str):
    """The default of the parameter ``name`` in ``signature``; None when it has none or no such
    parameter."""
    parameter = None if signature is None else signature.parameters.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def _sbi_defaults(model: str) -> dict:
    """The values sbi uses for the settings PRISM does not pass: its training method's (as installed,
    read at import), its posterior network factory's and its flow builder's (both read now)."""
    builder = getattr(_sbi_flow, f"build_{model}", None)
    built = inspect.signature(builder) if callable(builder) else None
    return {
        "validation_fraction": _default_of(_SNPE_TRAIN_SIGNATURE, "validation_fraction"),
        "clip_max_norm": _default_of(_SNPE_TRAIN_SIGNATURE, "clip_max_norm"),
        "z_score_theta": _default_of(inspect.signature(posterior_nn), "z_score_theta"),
        **{name: _default_of(built, name) for name in _BUILDER_DEFAULTS},
    }


def fit_settings(*, model: str, hidden_features: int, num_transforms: int, num_bins: int, learning_rate: float,
                 stop_after_epochs: int, max_num_epochs: int, batch_size: int, z_score_x: str, device, dtype,
                 num_rounds: int = 1) -> dict:
    """Every setting a fit is trained with, by value, the ones sbi fills in itself included.

    A kept fit is resumed only under the same settings. The ones PRISM never passes are recorded as sbi
    resolves them, so a later sbi whose default changed is seen as a change instead of passing unseen.
    Plain Python values, so a record can carry them and two equal settings compare equal.
    """
    defaults = _sbi_defaults(model)
    return {
        "hidden_features": int(hidden_features),
        "num_transforms": int(num_transforms),
        "learning_rate": float(learning_rate),
        "stop_after_epochs": int(stop_after_epochs),
        "max_num_epochs": int(max_num_epochs),
        "density_estimator": str(model),
        "num_bins": int(num_bins),
        "training_batch_size": int(batch_size),
        "num_rounds": int(num_rounds),
        "validation_fraction": defaults["validation_fraction"],
        "clip_max_norm": defaults["clip_max_norm"],
        "z_score_theta": defaults["z_score_theta"],
        "z_score_x": str(z_score_x),
        "tail_bound": defaults["tail_bound"],
        "hidden_layers_spline_context": defaults["hidden_layers_spline_context"],
        "num_blocks": defaults["num_blocks"],
        "dropout_probability": defaults["dropout_probability"],
        "use_batch_norm": defaults["use_batch_norm"],
        "data_device": _DATA_DEVICE,
        "device": torch.device(device).type,
        "dtype": str(dtype),
    }


# ── the trainer and its rows ────────────────────────────────────────────────────────────────────────
# What saving a fit as it goes calls on sbi's trainer: the per-pass summary it wraps, the network
# builder and the stored rows a resume rebuilds from, the loaders and the patience check whose order the
# restore relies on. The summary lists are read as a dict.
_HOOK_POINTS = ("_summarize_epoch", "_build_neural_net", "get_simulations", "get_dataloaders", "_converged")


def _require_hook_points(infer) -> None:
    """Refuse, loudly and before any row exists, a trainer without what saving a fit as it goes relies
    on: with another sbi than the one pinned, the first save, or a resume, would otherwise fail late or
    continue a different fit."""
    missing = next((name for name in _HOOK_POINTS if not callable(getattr(infer, name, None))), None)
    if missing is None and not isinstance(getattr(infer, "_summary", None), dict):
        missing = "_summary"
    if missing is not None:
        raise RuntimeError(
            f"sbi {sbi.__version__}'s trainer has no {missing}, which saving a training as it goes relies "
            f"on; install the sbi requirements.txt pins, or train without saving.")


def build_trainer(model: str, prior, embedding_net: torch.nn.Module, *, hidden_features: int, num_transforms: int,
                  num_bins: int, device, tracker=None, hooks: bool = False):
    """sbi's trainer for the posterior, built once, before any round's rows exist. With ``hooks`` it is
    checked at once for what saving a fit as it goes relies on."""
    # sbi's default z_score_x="independent" fits a PER-COLUMN affine over the conditioning vector.
    # Under the chi SET layout that is permutation-BREAKING (two orderings of one probe set would be
    # scaled differently, destroying the encoder's whole guarantee), and the near-constant mask column
    # becomes a ~1e7 amplifier under sbi's 1e-7 min-std clamp. The chi net standardizes itself instead
    # -- per channel, over live probes only. Derived from the net, never a separate argument, so the
    # standardizer and the encoder cannot be configured apart.
    _owns = getattr(embedding_net, "owns_standardization", False)
    neural_posterior = posterior_nn(model=model, embedding_net=embedding_net,
                                    z_score_x=("none" if _owns else "independent"),
                                    hidden_features=hidden_features, num_transforms=num_transforms,
                                    num_bins=num_bins)
    # tracker: sbi's default builds a TensorBoard writer under <cwd>/sbi-logs at THIS line (see
    # _NoTracker). The loss curve PRISM keeps is the one train_nn returns in its diagnostics.
    infer = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device),
                 tracker=_NoTracker() if tracker is None else tracker)
    if hooks:
        _require_hook_points(infer)
    return infer


@dataclasses.dataclass
class PreparedFit:
    """A trainer with one round's rows appended, ready for its fit: the trainer, the embedding network it
    conditions on, the number of rows appended and the settings the fit is trained with."""
    infer: object
    embedding_net: torch.nn.Module
    n_rows: int
    settings: dict


def prepare_fit(infer, training_params: TrainingPlan, model: str, embedding_net: torch.nn.Module,
                forcing_prior: torch.distributions.Distribution, nd_dim: int, forcing_idx: dict, rescale_idx: dict,
                *, theta_transform: Transform | None = None, fixed_dict: dict = None, proposal=None,
                hidden_features: int, num_transforms: int, num_bins: int, learning_rate: float,
                stop_after_epochs: int, max_num_epochs: int, batch_size: int, device,
                num_rounds: int = 1) -> PreparedFit:
    """Generate one round's rows (or read them back from the simulation set) and append them to the
    trainer it is given, which keeps its own copy; nothing here holds the rows afterwards."""
    _owns = getattr(embedding_net, "owns_standardization", False)
    # train the density estimator
    data, thetas = _pipeline.gen_training_data(
        training_params.model, training_params.prior, forcing_prior, training_params.t,
        training_params.run_size, training_params.num_runs,
        training_params.steady_idx, training_params.dt_nd_min,
        nd_dim, forcing_idx, rescale_idx,
        dt_exp=training_params.dt_exp, t_min_exp=training_params.t_min_exp,
        t_max_exp=training_params.t_max_exp, t_scale_bounds=training_params.t_scale_bounds,
        proposal=proposal,
        theta_transform=theta_transform,
        fixed_dict=fixed_dict,
        state_dep_drift=training_params.state_dep_drift,
        spontaneous_only=training_params.spontaneous_only,
        chi_mode=training_params.chi_mode,
        chi_f0=training_params.chi_f0,
        chi_freq_bounds=training_params.chi_freq_bounds,
        chi_k_pad=training_params.chi_k_pad,
        chi_k_fixed=training_params.chi_k_fixed,
        chi_max_cycles=training_params.chi_max_cycles,
        n_vars=training_params.n_vars,
        checkpoint=training_params.checkpoint,
        nd_idx=training_params.nd_idx,
        k_b_cell=training_params.k_b_cell,
        dtype=training_params.dtype, device=training_params.device,
    )

    # Filter the data -- and the TARGETS. thetas is the LATENT target (a logit); its row can in
    # principle go non-finite while the corresponding data row stays perfectly finite, and
    # filtering thetas only BY data would then feed the flow a +-inf target and NaN the loss for
    # the whole round with no diagnostic. On torch 2.9 the box round-trip cannot actually produce
    # one (SigmoidTransform._inverse clamps internally -- see gen_training_data), so this should
    # never fire; it exists so that if the transform stack ever changes, the failure arrives as a
    # message naming the offending columns instead of as a silently poisoned multi-hour run.
    nan_mask = torch.isfinite(data).all(dim=1)
    theta_finite_mask = torch.isfinite(thetas).all(dim=1)
    # WINSORISATION REPLACES THE OLD `abs(data) < 1e15` ROW FILTER, and
    # the change of instrument is the point. A row filter answers one bad channel by discarding
    # every other value in that row, and at a 1e15 threshold it caught 10 rows in 10.24M while the
    # outliers under the threshold survived: they are what dragged A1_mean's fitted std to
    # 4.19e11 and made the channel invisible to the flow.
    # Clipping each COLUMN at its own 0.1/99.9 percentile removes the leverage without removing
    # any row, and it protects sbi's own z-scoring on the non-chi paths too, which is why it is
    # not conditioned on who owns the standardisation.
    n_sum = getattr(embedding_net, "input_dim", None)
    if n_sum is None:
        # An embedding net that does not declare its summary width (a stub, a hand-built net in a
        # test) gets the pre-2026-08-26 behaviour exactly: there is no safe column split to clip
        # on, and guessing one could clip the chi block. See the block comment on
        # winsorize_summary_block.
        valid_idx = nan_mask & (torch.abs(data) < 1e15).all(dim=1) & theta_finite_mask
    else:
        valid_idx = nan_mask & theta_finite_mask
    n_bad_theta = int((~theta_finite_mask).sum())
    if n_bad_theta:
        bad_cols = torch.nonzero(~torch.isfinite(thetas).all(dim=0)).flatten().tolist()
        # stacklevel 3: this function, then train_nn, then the caller the warning is meant to name.
        warnings.warn(
            f"train_nn: dropped {n_bad_theta}/{thetas.shape[0]} training rows with non-finite "
            f"LATENT targets (columns {bad_cols}). The box round-trip is supposed to make this "
            f"impossible -- treat it as a bug in the transform stack, not as expected attrition.",
            stacklevel=3,
        )
    # Only pay for the gather when it actually drops something. `data[valid_idx]` is a boolean
    # gather: it allocates a SECOND full-size tensor while the first is still live -- about 0.45 GiB
    # per million rows, another ~9.3 GiB at the retrain's 20.48M -- and reinstates exactly the
    # doubled host peak that gen_training_data's preallocated accumulators were introduced to
    # remove. The mask is
    # all-true in practice (the box round-trip cannot produce a non-finite latent on torch 2.9;
    # SigmoidTransform._inverse clamps internally), so this is behaviour-identical and is what
    # makes that preallocation pay.
    if not bool(valid_idx.all()):
        thetas = thetas[valid_idx]
        data = data[valid_idx]

    if n_sum is not None:
        data = _pipeline.winsorize_summary_block(data, n_sum)

    # Fit OUR standardizers, once, on the post-filter data. posterior_nn was already called (it is
    # constructed before any data exists), so this has to happen here -- before append_simulations,
    # because build_nsf/get_numel run the net inside infer.train(). Unfitted, 41 raw statistics
    # spanning log-variances, nm-scale means and PLVs would reach the flow unscaled and the only
    # symptom would be a worse loss curve, days later; the encoder raises rather than allow it.
    if _owns and not embedding_net.standardization_fitted:
        embedding_net.fit_standardization(data)
        assert embedding_net.standardization_fitted, "chi standardization silently did not fit"

    # data_device="cpu": sbi otherwise defaults it to the TRAINING device (npe_base.py:174-175)
    # and moves the whole conditioning tensor onto the GPU -- at the retrain's 20.48M x 122 float32,
    # ~9.3 GiB resident, a ~18.6 GiB transient for the `x = x[is_valid_x]` filter, two (20.48M,)
    # bool masks from handle_invalid_x, and later a ~8.4 GiB gather when it slices the training
    # split (about 0.45 GiB per million rows for each full copy). All of
    # that on the same card that has to hold the flow, and all of it AFTER a multi-day generation
    # run (whose rows the simulation cache keeps when checkpointing is on). The data is ALREADY on
    # the CPU here (gen_training_data appends .cpu() tensors), so this also silences
    # validate_theta_and_x's "Moving x to the data_device" warning, which describes a ~9.3 GiB
    # copy nobody asked for.
    #
    # Training does not care: sbi's loop moves each minibatch to the training device itself, so a
    # CPU-resident dataset costs one host-to-device copy of (training_batch_size, 122) float32 --
    # tens of KB -- against a flow forward+backward that dominates it by orders of magnitude. It
    # hands ~9.3 GiB of VRAM back to the flow, which is a straight win.
    with _capped_zscore_check(skip=_owns):
        infer.append_simulations(thetas, data, proposal=proposal, data_device=_DATA_DEVICE)
    n_rows = int(thetas.shape[0])
    # sbi copies what it is handed (its `x[is_valid_x]` filter always does) and keeps its copies
    # through the fit, so this round's own reference is dead weight from here on: ~9.3 GiB of the
    # host's 63 at the retrain's size. Nothing below reads either name.
    del data, thetas
    settings = fit_settings(model=model, hidden_features=hidden_features, num_transforms=num_transforms,
                            num_bins=num_bins, learning_rate=learning_rate, stop_after_epochs=stop_after_epochs,
                            max_num_epochs=max_num_epochs, batch_size=batch_size,
                            z_score_x="none" if _owns else "independent", device=device,
                            dtype=training_params.dtype, num_rounds=num_rounds)
    return PreparedFit(infer=infer, embedding_net=embedding_net, n_rows=n_rows, settings=settings)


# ── the fit, saved after every pass and put back in a new process ───────────────────────────────────
class FitSaver(typing.Protocol):
    """What a fit is handed to at the end of every pass."""

    def save_pass(self, take: Callable[[], tuple[dict, dict]]) -> None:
        """Called once per pass, after sbi has summarised it. ``take()`` returns ``(snapshot, start)``:
        the fit as it stands and what was fixed when it began. The saver calls it inside its own
        handling of a failed save, so a snapshot that cannot be taken fails like a write that cannot be
        made."""


def take_snapshot(infer) -> dict:
    """The fit as it stands at the end of a pass, taken where sbi summarises the pass and before it
    counts it: what a new process needs to continue it. It only reads, and draws no random number.

    The weights and the optimizer's state are the trainer's own tensors, not copies, so a snapshot is
    written before the next pass changes them; a saver writes it at once.
    """
    return {
        "passes": infer.epoch + 1,                  # the number sbi gives the next pass
        "val_loss": infer._val_loss,
        "best_val_loss": infer._best_val_loss,
        "passes_since_best": infer._epochs_since_last_improvement,
        "best_state": infer._best_model_state_dict,
        "net_state": infer._neural_net.state_dict(),
        "optimizer_state": infer.optimizer.state_dict(),
        "summary": {k: list(v) for k, v in infer._summary.items()},
        "cpu_rng": torch.get_rng_state(),
    }


def restore_trainer(infer, start: dict, state: dict, *, settings: dict) -> None:
    """Put a kept pass back onto a fresh trainer holding the same rows, so that sbi's
    ``train(resume_training=True)`` continues the fit as if it had never stopped.

    In this order: the settings; the number of rows sbi holds; the train/validation split, drawn again
    from the generator's state kept when the fit began and checked by the digest of its validation
    rows; the network, built by sbi's own builder from the training rows, checked by its shape and by
    its fitted standardisers (fitted to the same rows they come out the same, so equal standardisers say
    the rows are the same); then the weights, the move to the training device, the optimizer and its
    state (after the move: the state is cast to each parameter's device as it loads), the counters, the
    best weights and the histories; last, the generator's state, so the next pass shuffles as the
    uninterrupted fit's would. The next pass's number comes from a snapshot taken at the pass's
    summary, before sbi counts the pass towards its patience, so the pass is not counted twice where the
    fit resumes. Raises ``fit_state.FitMismatch``, before the optimizer exists, when anything differs.
    """
    difference = fit_state.first_difference({"settings": start["settings"]}, {"settings": settings},
                                            parts=("settings",))
    if difference is not None:
        raise fit_state.FitMismatch(
            f"the setting {difference.name} was {difference.kept!r} when this training was kept, and is "
            f"{difference.now!r} now")

    theta, x, _ = infer.get_simulations(0)          # sbi's own stored rows, which its split is drawn over
    n = int(theta.size(0))
    if n != start["rows"]:
        raise fit_state.FitMismatch(f"its rows number {n}, not the {start['rows']} it was trained on")

    torch.set_rng_state(start["split_rng"])
    perm = torch.randperm(n)
    n_train = int((1 - start["settings"]["validation_fraction"]) * n)
    infer.train_indices, infer.val_indices = perm[:n_train], perm[n_train:]
    if fit_state.indices_digest(infer.val_indices) != start["validation_digest"]:
        raise fit_state.FitMismatch("its validation rows are not the ones it was trained with")

    infer._round = 0
    net = infer._build_neural_net(theta[infer.train_indices].to("cpu"), x[infer.train_indices].to("cpu"))
    del theta, x
    if fit_state.network_digest(net) != start["network"]:
        raise fit_state.FitMismatch("the network as built now differs in shape from the kept one")
    kept = state["net_state"]
    # Only the buffers the network's state keeps: one it does not keep (the flow's base distribution
    # holds its normalising constant so) is neither saved nor loaded, and the build alone makes it.
    persistent = net.state_dict()
    for name, buffer in net.named_buffers():
        if name in persistent and not torch.equal(buffer.detach().cpu(), kept[name].detach().cpu()):
            raise fit_state.FitMismatch(
                "the network's fitted standardisers differ from the kept ones, so its rows are not the same")

    net.load_state_dict(kept)
    infer._neural_net = net
    net.to(infer._device)
    infer.optimizer = torch.optim.Adam(list(net.parameters()), lr=settings["learning_rate"])
    infer.optimizer.load_state_dict(state["optimizer_state"])
    infer.epoch = state["passes"]
    infer._val_loss = infer.val_loss = state["val_loss"]
    infer._best_val_loss = state["best_val_loss"]
    infer._epochs_since_last_improvement = state["passes_since_best"]
    infer._best_model_state_dict = state["best_state"]
    infer._summary = {k: list(v) for k, v in state["summary"].items()}
    torch.set_rng_state(state["cpu_rng"])


def run_fit(prepared: PreparedFit, *, batch_size: int, learning_rate: float, stop_after_epochs: int,
            max_num_epochs: int, show_train_summary: bool, saver: FitSaver | None = None,
            resume: tuple[dict, dict] | None = None):
    """sbi's own passes over the prepared rows; returns sbi's density estimator.

    ``resume`` is ``(start, state)`` of a kept fit: it is put back onto the trainer first and sbi
    continues it. With a ``saver`` the fit is handed to it at the end of every pass. Every pass then
    ends with one record of its numbers (``fit_epoch_line``), said after the pass's save.
    """
    infer = prepared.infer
    # The passes are sbi's own. The trainer's per-pass summary is wrapped whenever it has one, saver or
    # not: once sbi has summarised a pass, and before it counts it, the saver (when there is one) is
    # handed a way to take the fit as it stands with what was fixed when it began, and then the pass is
    # said in one record. Under the window every record is where a Cancel is taken. With a saver it is
    # taken at the first record after the pass's save, the "[fit] saved" line, so a stop loses no
    # finished pass; without one, at the pass's own record, which loses the pass in progress, since
    # nothing is kept. A stopped fit continues from its last saved pass in a new process: the pass is put
    # back onto a fresh trainer holding the same rows, and sbi resumes.
    # Plain Python numbers: sbi 0.26 type-checks these strictly, and a numpy or tensor scalar would be
    # refused here, after the generation. Any error from the fit is a FAILURE, not a refusal: sbi
    # reports a non-finite loss as a ValueError, which the front ends read as a check made before
    # anything was spent, but this one arrives after the generation and part of the fit.
    kept_start, state = resume if resume is not None else (None, None)
    hooked = False
    try:
        knobs = dict(training_batch_size=int(batch_size), learning_rate=float(learning_rate),
                     stop_after_epochs=int(stop_after_epochs), max_num_epochs=int(max_num_epochs),
                     show_train_summary=bool(show_train_summary))
        if resume is not None:
            restore_trainer(infer, kept_start, state, settings=prepared.settings)
            knobs["resume_training"] = True
        if _wrap_summary(infer, saver):
            summarize = infer._summarize_epoch
            begun = {}                                  # the start, made at the first save

            def take() -> tuple[dict, dict]:
                if not begun:
                    begun.update(kept_start if kept_start is not None else {
                        "split_rng": split_rng,
                        "validation_digest": fit_state.indices_digest(infer.val_indices),
                        "rows": len(infer.train_indices) + len(infer.val_indices),
                        "network": fit_state.network_digest(infer._neural_net),
                        "settings": prepared.settings,
                    })
                    begun["started"] = started
                return take_snapshot(infer), begun

            def _summarize_epoch(*args, **kwargs):
                summarize(*args, **kwargs)
                if saver is not None:
                    saver.save_pass(take)
                _announce(infer)

            infer._summarize_epoch = _summarize_epoch
            hooked = True
        if saver is not None:
            # Read last, just before sbi's training method, whose first draw is the train/validation
            # split: this state draws the same split again when the fit is resumed.
            split_rng = None if kept_start is not None else torch.get_rng_state()
            started = time.time()
        return infer.train(**knobs)
    except ValueError as e:
        raise RuntimeError(f"the flow's fit failed after the training data was generated: {e}") from e
    finally:
        if hooked:
            del infer._summarize_epoch                  # the trainer's own method again


def train_nn(training_params: TrainingPlan, model: str, prior: torch.distributions.Distribution, embedding_net: torch.nn.Module,
             forcing_prior: torch.distributions.Distribution, nd_dim: int, forcing_idx: dict, rescale_idx: dict,
             x_obs: torch.Tensor = None, theta_obs: torch.Tensor = None, num_rounds: int = 1, return_diagnostics: bool = False, theta_transform: Transform | None = None,
             fixed_dict: dict = None,
             hidden_features: int = 50, num_transforms: int = 5, num_bins: int = 10,
             learning_rate: float = 5e-4, stop_after_epochs: int = 20, max_num_epochs: int = 2_147_483_647,
             show_train_summary: bool = False,
             batch_size: int = 128, device: torch.device = torch.device('cpu'),
             tracker=None, fit_saver: FitSaver | None = None,
             fit_resume: tuple[dict, dict] | None = None) -> DirectPosterior | tuple[DirectPosterior, dict]:
    """
    Generate training data and fit the NPE flow (SNPE when ``num_rounds > 1``).

    :param training_params: a ``TrainingPlan`` -- the typed carrier of everything
        ``gen_training_data`` is driven with; its field defaults are the single source of the
        old per-call-site ``.get()`` defaulting.
    :param model: sbi density-estimator name (e.g. "nsf"); NOT the simulation model, which rides in
        ``training_params.model``.
    :param embedding_net: the conditioning net; ``owns_standardization`` decides z_score_x below.
    :param x_obs: observed data, required for SNPE (``num_rounds > 1``).
    :param theta_obs: ground-truth parameters, required only for ``return_diagnostics``.
    :param hidden_features: hidden units per flow transform; ``num_transforms`` the coupling-layer
        count; ``num_bins`` the spline bins (NSF only); ``learning_rate`` /
        ``stop_after_epochs`` / ``max_num_epochs`` / ``batch_size`` sbi's fit-loop knobs;
        ``show_train_summary`` prints sbi's per-epoch loss table.
    :param tracker: what sbi reports the fit's curves to; None (every PRISM caller) = ``_NoTracker``,
        which discards them instead of letting sbi write ``<cwd>/sbi-logs``. Any object following sbi's
        ``Tracker`` protocol (``log_metric(name, value, step)``, ``flush()`` and the rest) is handed to
        sbi as is. sbi hands a tracker its curves only after its loop has ended; the per-epoch record is
        the run's line for each epoch.
    :param fit_saver: handed the fit at the end of every pass (a ``FitSaver``); None saves nothing.
    :param fit_resume: ``(start, state)`` of a kept fit, which the fit continues instead of starting.
    :return: the trained ``DirectPosterior`` -- ``(posterior, diagnostics)`` when
        ``return_diagnostics``.
    """
    if num_rounds > 1 and x_obs is None:
        raise ValueError("x_obs must be specified for SNPE algorithm")
    if num_rounds > 1 and training_params.checkpoint is not None:
        # Refused loudly rather than half-supported. Rounds >= 2 sample from a PROPOSAL -- a trained
        # DirectPosterior -- whose identity a checkpoint would have to capture and re-validate, which
        # is a separate problem from the one the training checkpoint solves. TRAINING_NUM_ROUNDS is 1
        # (amortized NPE), so this costs nothing today and closes the hole rather than leaving it to
        # be discovered.
        raise ValueError(
            f"Training-data checkpointing is not supported for SNPE (num_rounds={num_rounds}); the "
            f"per-round proposal is not part of the checkpoint's identity. Use num_rounds=1, or "
            f"drop the plan's checkpoint.")
    saving = fit_saver is not None or fit_resume is not None
    if num_rounds > 1 and saving:
        # The same hole for a kept fit: a later round's rows come from the proposal, which its identity
        # does not hold.
        raise ValueError(
            f"Saving a training as it goes is not supported for SNPE (num_rounds={num_rounds}); the "
            f"per-round proposal is not part of a kept training's identity. Use num_rounds=1, or "
            f"train without saving.")

    infer = build_trainer(model, prior, embedding_net, hidden_features=hidden_features,
                          num_transforms=num_transforms, num_bins=num_bins, device=device, tracker=tracker,
                          hooks=saving)

    proposal = None # set up initial proposal distribution
    posterior = None

    # diagnostics storage
    diagnostics = {
        "log_prob_true": [],
        "posterior_means": [],
        "posterior_stds": [],
    }

    for _ in tqdm(range(num_rounds), desc=f"Training neural posterior", leave=False):
        prepared = prepare_fit(infer, training_params, model, embedding_net, forcing_prior, nd_dim, forcing_idx,
                               rescale_idx, theta_transform=theta_transform, fixed_dict=fixed_dict,
                               proposal=proposal, hidden_features=hidden_features,
                               num_transforms=num_transforms, num_bins=num_bins, learning_rate=learning_rate,
                               stop_after_epochs=stop_after_epochs, max_num_epochs=max_num_epochs,
                               batch_size=batch_size, device=device, num_rounds=num_rounds)
        density_estimator = run_fit(prepared, batch_size=batch_size, learning_rate=learning_rate,
                                    stop_after_epochs=stop_after_epochs, max_num_epochs=max_num_epochs,
                                    show_train_summary=show_train_summary, saver=fit_saver, resume=fit_resume)
        posterior = infer.build_posterior(density_estimator)
        assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"

        # compute diagnostics after each round
        if return_diagnostics and x_obs is not None:
            x_obs_device = x_obs.to(device)

            # log probability of ground truth
            if theta_obs is not None:
                theta_true_device = theta_obs.to(device)
                if theta_true_device.dim() == 1:
                    theta_true_device = theta_true_device.unsqueeze(0)
                log_prob = posterior.log_prob(theta_true_device, x=x_obs_device).item()
                diagnostics["log_prob_true"].append(log_prob)

            # posterior mean and std from samples
            samples = posterior.sample((10000,), x=x_obs_device)
            diagnostics["posterior_means"].append(samples.mean(dim=0).cpu())
            diagnostics["posterior_stds"].append(samples.std(dim=0).cpu())

        # need to now check if num_runs > 1: if so, then that is equivalent to SNPE, and if not, that is equivalent to NPE
        if num_rounds > 1:
            assert x_obs is not None, "x_obs must be specified for SNPE algorithm"
            proposal = posterior.set_default_x(x_obs.to(device)) # if SNPE, the user has to specify x_obs

    assert isinstance(posterior, DirectPosterior), f"Expected DirectPosterior, got {type(posterior)}"

    # Capture the per-epoch train/validation loss curve from the sbi trainer so the
    # convergence ("under-fit vs converged") check is reproducible. sbi keeps these in
    # infer._summary; they are otherwise discarded when this function returns.
    if return_diagnostics:
        summary = getattr(infer, "_summary", {}) or {}
        diagnostics["training_loss"] = list(summary.get("training_loss", []))
        diagnostics["validation_loss"] = list(summary.get("validation_loss", []))
        best_val = summary.get("best_validation_loss", [])
        diagnostics["best_validation_loss"] = best_val[-1] if len(best_val) else None
        epochs = summary.get("epochs_trained", [])
        diagnostics["epochs_trained"] = epochs[-1] if len(epochs) else None
        diagnostics["stop_after_epochs"] = stop_after_epochs
        return posterior, diagnostics
    return posterior
