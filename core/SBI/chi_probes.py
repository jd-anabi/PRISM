"""The chi(omega) probe loop, split out of pipeline.py (which stays the public facade).

Consumers -- gen_training_data's _rows closure, orchestrator's observation and predictive-check
paths, the test suites -- reach these as ``pipeline.gen_chi_raw`` / ``gen_chi_block`` /
``_subset_probe_rows`` via the facade's bottom re-import, which also keeps monkeypatching
``pipeline.<name>`` effective. Calls back into pipeline machinery (gen_obs, the force builder,
the hot-loop release, the batch tag) go through the module object at call time.
"""
import inspect
import math
import warnings
from dataclasses import dataclass

import torch

from core import config
from core import forcing as _forcing
from core.Helpers import helpers
from core.SBI import chi
from core.SBI import pipeline as _pipeline


@dataclass(frozen=True, eq=False)
class ProbeRecord:
    """What one ``gen_chi_block`` call inside the training generator measured, for a probe observer
    (``pipeline.gen_training_data``'s ``probe_observer``). Built only while an observer is attached,
    and handed to it only once the batch's rows are stored. Every tensor is detached, on the CPU and
    a copy of its own, so nothing the observer does can touch the batch. Records compare by identity:
    a field-by-field comparison of tensor fields has no single truth value.

    :param batch_tag: the batch the rows belong to, as the generator's messages name it.
    :param lo: first row of this call's range within its training batch.
    :param hi: one past the range's last row; ``hi - lo`` rows.
    :param f_peak: (rows,) each row's own passive peak frequency, in cell frequency units:
                   ``chi.peak_freq`` of the spontaneous trace the probes were placed against.
    :param duration_frac: (K,) the per-probe fractions of the recording ``gen_chi_raw`` was handed to
                          lock each probe in over, before the cycle ceiling shortens them; None when it
                          was handed none (every probe over the full length).
    :param k: the number of probes simulated for every row, K.
    :param u: (rows, K) ``gen_chi_raw``'s returned log of each probe's frequency over ``f_peak``.
    :param logcyc: (rows, K) ``gen_chi_raw``'s returned log of the drive cycles each probe was
                   actually locked in over.
    :param valid: (rows, K) ``gen_chi_raw``'s returned verdict per probe.
    :param packed_mask: (rows, k_pad) the packer's mask, in slot order, before the per-row subsetting.
    :param dt_exp: the sampling interval of the recording, in cell time units.
    :param n_points: the full recording length in samples, of which ``duration_frac`` takes its
                     fractions; with ``dt_exp`` it gives the longest lock-in any probe could have had.

    ``gen_chi_raw`` returns only ``valid``, so an audit that wants to know WHY a probe failed
    recomputes the non-finite and Nyquist predicates itself, from the driven frequency
    ``f_peak * exp(u)`` and ``dt_exp``.

    ``u``, ``logcyc`` and ``valid`` are in PROBE order; ``packed_mask`` is in SLOT order -- the packer
    moves each row's live probes to the front, ascending in frequency, and every column from ``k`` on
    is padding, always False. So ``packed_mask[:, j]`` is not probe ``j``, and the two compare only
    through per-row counts of ``packed_mask[:, :k]``.
    """
    batch_tag: str
    lo: int
    hi: int
    f_peak: torch.Tensor
    duration_frac: torch.Tensor | None
    k: int
    u: torch.Tensor
    logcyc: torch.Tensor
    valid: torch.Tensor
    packed_mask: torch.Tensor
    dt_exp: float
    n_points: int


def _subset_probe_rows(block: torch.Tensor, mask: torch.Tensor, k_pad: int, generator) -> torch.Tensor:
    """Randomly keep a PREFIX of each row's live probes, in place, and re-zero what is dropped.

    Half the rows keep their full set; the rest keep Uniform{1..n} of them. The drive set is shared
    across the batch (one simulation per probe serves every row), so this costs nothing and is the
    only way to decouple the per-row probe count from the batch's (t_scale, T) stratum -- otherwise
    the flow could read K off the batch's other conditioning and the encoder would never have to
    generalise. Dropping a PREFIX is safe because pack_probe_block already ordered the live probes
    contiguously; the survivors stay contiguous and frequency-ordered.
    """
    B = block.shape[0]
    e = block.reshape(B, k_pad, config.CHI_ELEM_W).clone()
    n = mask.sum(dim=1)                                                   # (B,) live probes per row
    full = torch.rand(B, generator=generator) < 0.5
    frac = torch.rand(B, generator=generator).to(n.device)
    keep = torch.where(full.to(n.device), n, (frac * n.to(frac.dtype)).floor().long() + 1)
    keep = torch.minimum(keep, n).clamp(min=0)
    slots = torch.arange(k_pad, device=block.device).unsqueeze(0)         # (1, k_pad)
    alive = slots < keep.unsqueeze(1).to(slots.device)                    # (B, k_pad)
    return (e * alive.unsqueeze(-1).to(e.dtype)).reshape(B, k_pad * config.CHI_ELEM_W)


def gen_chi_raw(model: str, params_nd: torch.Tensor, rescale: torch.Tensor, x_spont_dim: torch.Tensor,
                t_fine: torch.Tensor, inits: torch.Tensor, rescale_idx: dict,
                n_segs: int, steady_idx: int, subsample, N_points: int, dt_exp: float,
                multipliers: torch.Tensor, f0_nd: float, state_dep_drift: bool = False,
                fixed_dict: dict = None,
                absolute_freqs: bool = False, resolution_filter: bool = True,
                duration_frac=None, max_cycles: float | None = None,
                adapt_placement: bool = False, bounds: tuple | None = None,
                dtype: torch.dtype = torch.float32,
                device: torch.device = torch.device('cpu')) -> tuple:
    """
    K single-tone forced runs -> the RAW probe measurements
    ``(chi (B,K) complex, u (B,K), logcyc (B,K), valid (B,K) bool)``. Generalizes the single-frequency
    Group-G lock-in to a susceptibility CURVE (see config.CHI_MODE + core/SBI/chi.py). One forced
    simulation per probe = the "single-tone x K recordings" protocol.

    Exactly K simulations run, never k_pad -- padding is free.

    Drives at a FIXED ND amplitude f0_nd by passing dimensional amp = f0_nd * f_scale to
    build_nondim_sin_force_tensor (which divides it back to f0_nd), so lock-in SNR is uniform across
    the f_scale prior. chi = redimensionalized response / dimensional drive = (x_scale/f_scale)*chi_nd
    carries the physical scale magnitude (like Group-G's gain); its SHAPE over omega carries the ND
    resonance.

    :param params_nd: (B, n_nd) ND params (the inferred ND block).
    :param rescale: (B, n_rescale) PHYSICAL rescale params (x_scale/t_scale/f_scale...).
    :param rescale_idx: the SIMULATOR's rescale index (``core.SBI.derived.for_simulation``). One that
                        names T and no f_scale raises RuntimeError before anything is simulated.
    :param x_spont_dim: (B, N_points) physical spontaneous trace -> Omega_0 per sample.
    :param t_fine: (T_full,) fine ND time grid the drive/sim use.
    :param inits: (B, n_vars) initial conditions.
    :param subsample: fine->dt_exp downsample factor. A scalar int (uniform t_scale: training batches,
                      a single GT) OR a (B,) per-sample tensor (posterior samples in PPC, whose t_scale
                      differs per sample); applied via gather so both cases share one code path.
    :param multipliers: (K,) or (B, K). Relative multipliers of each row's own measured Omega_0, or --
                        with ``absolute_freqs`` -- frequencies in cell freq units. ABSOLUTE is what the
                        experimental path and the PPC use: the experiment fixed the drive frequencies,
                        and re-deriving them per posterior sample from that sample's own f_peak would
                        simulate a different experiment and make the PPC agree for the wrong reason.
    :param f0_nd: ND drive amplitude (config.CHI_F0).
    :param resolution_filter: mark probes below config.CHI_MIN_CYCLES drive cycles invalid. **Pass
                              False for the Fisher.** The filter depends on f_peak, which depends on
                              theta, so a probe can CROSS the threshold between the +dz and -dz arms
                              of a central difference -- a step of 1 divided by fnoise's 1e-9 floor
                              puts ~1e9 into the Jacobian, and V becomes that discontinuity.
    :param duration_frac: (K,) fractions of N_points to lock each probe in over. None = full length.
    :param max_cycles: CEILING on the drive cycles each probe is locked in over; None reads
                       ``config.CHI_MAX_CYCLES``, ``math.inf`` disables it. Applied AFTER
                       ``duration_frac``, so it is a ceiling on that draw rather than a replacement.
                       This is not a filter -- nothing is masked or dropped, the SEGMENT is shortened
                       -- which is why it lives here rather than in a caller: training, the Fisher
                       rotation, the PPC and the experimental path must all measure the same
                       observable, and a ceiling applied in only one of them is silent. See
                       config.CHI_MAX_CYCLES for the measurement behind it.
    :param adapt_placement: lift each ROW's multipliers into the sub-band its own Omega_0 can resolve
                            (:func:`core.SBI.chi.resolvable_multipliers`). **TRAINING ONLY.** The
                            experimental path drove at frequencies the experiment chose and the PPC
                            must reproduce the observation's, so both pass ``absolute_freqs`` and are
                            never adapted -- moving a probe there would answer a different experiment
                            than the one that was run. Ignored when ``absolute_freqs`` is set.
    :param bounds: the chi band, for ``adapt_placement``; None reads ``config.CHI_FREQ_BOUNDS``.
    :return: (chi (B,K) complex, u (B,K), logcyc (B,K), valid (B,K) bool). Use
             :func:`gen_chi_block` for the padded conditioning block.
    """
    _forcing.require_simulator_index(rescale_idx)
    max_cycles = config.CHI_MAX_CYCLES if max_cycles is None else float(max_cycles)
    B = params_nd.shape[0]
    f_peak = chi.peak_freq(x_spont_dim, dt_exp)                         # (B,) cell freq units
    x_scale = rescale[:, rescale_idx["x_scale"]].unsqueeze(1)
    x_offset = rescale[:, rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in rescale_idx else 0.0
    if "f_scale" in rescale_idx:
        f_scale_eff = rescale[:, rescale_idx["f_scale"]]                # (B,)
    else:  # Hopf-style (neither f_scale nor T; T alone raised above): the builder uses x_scale / t_scale
        f_scale_eff = rescale[:, rescale_idx["x_scale"]] / rescale[:, rescale_idx["t_scale"]]
    amp_dim = f0_nd * f_scale_eff                                       # (B,) dimensional; ND drive == f0_nd
    T_obs = N_points * dt_exp
    nyq = 0.5 / dt_exp                                                  # dt_exp-sampling Nyquist (cell freq units)
    # Fine -> dt_exp downsampling. gen_obs solves on t_fine and returns [..., steady_idx:], so x_nd's
    # width is this same value for every one of the K runs -- the choice below is loop-invariant and
    # is made ONCE, and the index tensor (when needed at all) is built ONCE.
    #   * uniform int subsample AND a fine grid long enough that the clamp cannot bind -> plain
    #     strided slicing, exactly what the non-chi branches of gen_training_data do. This builds NO
    #     (B, N_points) int64 index at all; the old code kept two of them live, ~2 GB at run_size=2048.
    #   * (B,) per-sample subsample (the PPC path, whose rows have different strides), or a fine grid
    #     that ran out -> keep the gather. `t_fine = t[:n_fine_total]` SILENTLY CLIPS, which happens
    #     for ~20% of accepted draws on model-builder bounds (t_scale in (v/2, v*2) makes len(t)
    #     shorter than the N_ND_MAX filter allows). There the clamp REPLICATES the last sample, where
    #     slicing would quietly return fewer than N_points columns -- desynchronising the trace from
    #     the T_obs that normalises chi below, a bias that would show up only in that corner.
    n_avail = t_fine.shape[0] - steady_idx
    s_int = None if torch.is_tensor(subsample) else max(1, int(subsample))
    idx_c = None
    if s_int is None or s_int * (N_points - 1) >= n_avail:
        subs = (subsample.to(device=device).long().clamp(min=1) if torch.is_tensor(subsample)
                else torch.full((B,), s_int, device=device, dtype=torch.long))
        idx_c = (subs.unsqueeze(1)
                 * torch.arange(N_points, device=device, dtype=torch.long).unsqueeze(0)
                 ).clamp_(max=n_avail - 1)                              # (B, N_points), clamped in place
    fidx = {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
    n_force_ch = _forcing.n_force_channels(model, fidx, inits.shape[-1])

    # Resolve the probe frequencies ONCE, before the loop. `multipliers` may be relative (the usual
    # case: mult_k * the passive trace's own peak) or ABSOLUTE cell-frequency values, which is what
    # the experimental path and the PPC need -- there the drive frequencies were fixed by the
    # experiment, and re-deriving them per posterior sample from that sample's own f_peak would
    # simulate a different experiment.
    mults = multipliers if torch.is_tensor(multipliers) else torch.as_tensor(multipliers)
    mults = mults.to(device=device, dtype=f_peak.dtype)
    if mults.dim() == 1:
        mults = mults.unsqueeze(0)                                      # (1, K) -> broadcast over B
    if adapt_placement and not absolute_freqs:
        # Per-ROW placement: one shared multiplier set cannot resolve across a prior spanning ~4
        # decades of Omega_0 -- live-probe fraction goes 0% below 3 Hz to 98% above 30 Hz when the
        # set is shared, and the driver is the row's own Omega_0. Uses the FULL duration as the budget --
        # duration_frac and the CHI_MAX_CYCLES ceiling below only ever SHORTEN the window, so a
        # multiplier chosen against the full length is the most permissive honest choice; the floor
        # check below still has the last word on the duration actually used.
        mults = chi.resolvable_multipliers(mults, f_peak, N_points * dt_exp, bounds=bounds)
    freqs = mults if absolute_freqs else mults * f_peak.unsqueeze(1)    # (B, K) or (1, K)
    freqs = freqs.expand(B, -1)
    K = freqs.shape[1]

    chis, u_list, logcyc_list = [], [], []
    # Per-probe validity. A probe is never MOVED and never silently dropped -- it is masked, and the
    # caller is told how many. Clamping to Nyquist (what this used to do) relabels a probe as a
    # different frequency than the one requested, which is invisible downstream.
    valid = torch.ones((B, K), dtype=torch.bool, device=device)
    for k in range(K):
        freq_k = freqs[:, k].contiguous()                              # (B,) absolute, cell freq units
        valid[:, k] &= torch.isfinite(freq_k) & (freq_k > 0) & (freq_k < 0.9 * nyq)
        # Per-probe duration: lock in over a PREFIX of the trace. Free (the samples already exist) and
        # it makes the (duration, frequency) trade-off -- what a real session actually varies -- an
        # axis of the training distribution rather than a constant.
        N_k = N_points if duration_frac is None else max(1, int(round(float(duration_frac[k]) * N_points)))
        # THE DURATION CEILING (config.CHI_MAX_CYCLES), applied PER ROW.
        #
        # It used to be one scalar keyed on the batch's FASTEST row, because lock_in_batched took a
        # scalar T_obs. That cost was real and measured: Omega_0 spans ~4 decades inside a training
        # batch, so keying on the fastest truncated the slow rows to a fraction of a cycle and masked
        # them -- ~48 % of rows carried no live probe at all. lock_in_batched
        # now takes an (B,) n_samples, so each row gets exactly the prefix its own frequency needs.
        # Rows whose full length is already under the ceiling are untouched.
        #
        # Computed over the ALREADY-VALIDATED rows: freq_k still holds the non-finite / out-of-Nyquist
        # entries of rows masked on the line above, and dividing by those would poison the length. An
        # invalid row keeps the full N_k -- it is masked anyway, so its length changes nothing.
        N_row = torch.full((B,), N_k, dtype=torch.long, device=device)
        if math.isfinite(max_cycles):
            ok_k = valid[:, k] & (freq_k > 0)
            if bool(ok_k.any()):
                cap_row = torch.floor(max_cycles / freq_k.clamp(min=1e-30).double() / dt_exp)
                cap_row = cap_row.clamp(min=1.0, max=float(N_k)).long()
                N_row = torch.where(ok_k, cap_row, N_row)
        # T_row is what each row was ACTUALLY integrated over: it normalises that row's chi, sets its
        # cycle count for the floor below, and is what logcyc reports. There is deliberately no
        # scalar counterpart any more -- keeping one around is how logcyc would come to describe a
        # duration the lock-in did not use.
        T_row = N_row.to(torch.float64) * dt_exp
        if resolution_filter:
            # A lock-in over a fraction of a cycle returns the demeaned trace's residual drift plus
            # spontaneous 1/f content: finite, in range, and REPRODUCIBLE -- which is exactly why it
            # survived a CV screen -- but it is not a susceptibility.
            # Against the row's OWN duration -- the whole point of the per-row lock-in duration is
            # that these differ.
            valid[:, k] &= (freq_k.double() * T_row) >= config.CHI_MIN_CYCLES
        forcing_params = torch.zeros((B, 4), dtype=dtype, device=device)
        forcing_params[:, 0] = amp_dim
        forcing_params[:, 1] = freq_k
        forcing_params[:, 2] = math.pi / 2.0                           # phase -> cos drive (FDT convention)
        force = _pipeline.build_nondim_sin_force_tensor(forcing_params, t_fine, rescale, fidx, rescale_idx)
        if force.shape[1] < n_force_ch:
            # The sin builder emits ONE channel (fidx above declares no "amp_y"), but the model's
            # drift may index more: HopfModel reads force_step[:, 1] unconditionally, and a user
            # model reads one channel per state variable -- so chi mode used to die with an
            # IndexError on anything but Nadrowski/BP. Probe channel 0 and leave the rest at zero,
            # which is the same convention the FDT campaigns drive (see forcing.n_force_channels).
            padded = torch.zeros((B, n_force_ch, force.shape[2]), dtype=force.dtype, device=force.device)
            padded[:, :force.shape[1], :] = force
            force = padded
        x_nd = _pipeline.gen_obs(model=model, params=params_nd, t=t_fine, inits=inits, force=force,
                       n_segs=n_segs, steady_idx=steady_idx, fixed_dict=fixed_dict,
                       state_dep_drift=state_dep_drift, batch_size=B, var_idx=0,
                       dtype=dtype, device=device)[0, :, :]
        x_sub = x_nd[:, ::s_int][:, :N_points] if idx_c is None else torch.gather(x_nd, 1, idx_c)
        x_dim = helpers.rescale(x_sub[:, :N_k], x_scale, x_offset)      # (B, N_k), a FRESH tensor
        # Release the simulation BEFORE the lock-in: x_nd is a view and so pins its whole base, and
        # force is a (B, n_force_ch, T_fine) tensor of its own. helpers.rescale has already
        # materialised x_dim, so nothing below reads any of them. (idx_c is loop-invariant --
        # do NOT drop it.)
        del force, x_nd, x_sub
        # n_samples/T_row, not the scalar: each row is integrated over its own prefix of x_dim.
        chis.append(chi.lock_in_batched(x_dim, 2.0 * math.pi * freq_k, amp_dim, T_row, dt_exp,
                                        n_samples=N_row))
        # The frequency ACTUALLY locked in at, and the cycles actually seen -- both carried as
        # features rather than implied by slot index, which is what makes placement free. logcyc uses
        # T_row for the same reason the filter above does: it is the encoder's record of how much
        # evidence this probe rests on, so it must describe the integration that really happened.
        u_list.append(torch.log(freq_k / f_peak.clamp(min=1e-30)))
        logcyc_list.append(torch.log(torch.clamp(freq_k.double() * T_row, min=1e-30)).to(freq_k.dtype))
        del x_dim
        # plans/graphs OFF -- see gen_stats. Every probe replays the same captured graph, so
        # dropping it here would recapture K times per batch.
        _pipeline._release_device_memory(device, plans=False, graphs=False)

    return (torch.stack(chis, dim=1), torch.stack(u_list, dim=1),
            torch.stack(logcyc_list, dim=1), valid)


def gen_chi_block(*args, k_pad: int = None, bounds: tuple = None,
                  row_range: tuple[int, int] | None = None, **kwargs) -> tuple:
    """``gen_chi_raw`` + :func:`core.SBI.chi.pack_probe_block` -> the padded CONDITIONING block.

    Split from the raw lock-in deliberately: the Fisher (``SBI/decorrelate.feats``) needs the
    susceptibilities WITHOUT the frequency and mask channels, because both are theta-independent
    there and poison the Jacobian -- see CHI_FISHER_CHANNELS. Sharing the simulation loop keeps the
    two feature sets provably from drifting apart.

    :param row_range: the rows' [lo, hi) within their training batch. Given only by the training
                      generator; with it, and while a batch probe tally is active, this call's masked
                      and simulated probe counts are reported to that tally under the range -- with a
                      :class:`ProbeRecord` when the tally has a probe observer, and nothing built
                      otherwise. Never forwarded to ``gen_chi_raw``.
    :return: ((B, CHI_ELEM_W*k_pad) block, (B, k_pad) bool mask).
    """
    # `bounds` goes to BOTH: the packer normalises u_hat by it, and adapt_placement compresses into
    # it. Forwarding to only one would let a probe be placed against one band and screened against
    # another -- the same class of mismatch the load-time band check (``ArtifactStore.load_posterior``)
    # exists to catch.
    chi_stack, u, logcyc, valid = _pipeline.gen_chi_raw(*args, bounds=bounds, **kwargs)
    block, mask = chi.pack_probe_block(chi_stack, u, logcyc, valid, k_pad=k_pad, bounds=bounds)
    B, K = chi_stack.shape
    dropped = int((~mask[:, :K]).sum())
    tally = _pipeline._ACTIVE_LEDGER
    if row_range is not None and tally is not None:
        lo, hi = int(row_range[0]), int(row_range[1])
        record = None
        if tally.observer is not None:
            record = _probe_record(args, kwargs, bounds, lo, hi, K, u, logcyc, valid, mask)
        tally.add(lo, hi, dropped, B * K, record=record)
    if dropped:
        # Silent attrition is what made the first chi posterior inexplicable. Make it a number.
        warnings.warn(
            f"{_pipeline._batch_tag()}: chi: {dropped}/{B * K} probes masked "
            f"(below {config.CHI_MIN_CYCLES} drive cycles, "
            f"at/above Nyquist, out of band, or a non-finite lock-in).", stacklevel=2)
    return block, mask


def _cpu_copy(t: torch.Tensor) -> torch.Tensor:
    """A detached CPU tensor with storage of its own, whatever device ``t`` is on."""
    return t.detach().to(device="cpu", copy=True)


def _probe_record(args: tuple, kwargs: dict, bounds, lo: int, hi: int, k: int, u: torch.Tensor,
                  logcyc: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor) -> ProbeRecord:
    """The :class:`ProbeRecord` of one ``gen_chi_block`` call: what it handed ``gen_chi_raw`` and what
    came back. The training generator passes the trace, the sampling interval and the recording
    length POSITIONALLY, so they are read by binding the call to ``gen_chi_raw``'s own signature,
    never by position. Draws no random numbers: the peak frequency is recomputed with the same
    deterministic estimator ``gen_chi_raw`` used on the same trace."""
    bound = inspect.signature(gen_chi_raw).bind(*args, bounds=bounds, **kwargs).arguments
    dt_exp = float(bound["dt_exp"])
    dfrac = bound.get("duration_frac")
    return ProbeRecord(
        batch_tag=_pipeline._batch_tag(), lo=lo, hi=hi,
        f_peak=_cpu_copy(chi.peak_freq(bound["x_spont_dim"], dt_exp)),
        duration_frac=None if dfrac is None else _cpu_copy(torch.as_tensor(dfrac)),
        k=int(k), u=_cpu_copy(u), logcyc=_cpu_copy(logcyc), valid=_cpu_copy(valid),
        packed_mask=_cpu_copy(mask), dt_exp=dt_exp, n_points=int(bound["N_points"]))

