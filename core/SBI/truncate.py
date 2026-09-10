"""TSNPE: truncated sequential NPE (Deistler, Goncalves & Macke 2022). Section 11.6.

    posterior -> HPD region A -> sample theta from the PRIOR RESTRICTED TO A -> simulate -> retrain

⚠⚠ THE ONE THING THAT MUST NOT BE GOT WRONG, and it is why this module exists rather than a few lines
inside build_posterior:

    THE PROPOSAL IS THE TRUNCATED PRIOR. IT IS NEVER THE POSTERIOR.

The posterior only ever says WHERE TO LOOK. Fitting a density to the posterior and proposing from it
gives ``p_L ∝ L^(L+1) q`` -- tempering. Credible intervals then contract as ``(L+1)^(-1/2)`` with NO
new information entering: at L = 4 the posterior is 2.2x narrower than the data supports. And SBC
comes out FLAT anyway, because SBC validates the flow against the proposal it was trained on. No
diagnostic in this project catches that, which is why ``tests/`` carries a dedicated pinning test
asserting the round-1 credible width does not contract by sqrt(2).

Truncation is a RESTRICTION, not a REWEIGHTING, which is the property that makes TSNPE need no
proposal correction where SNPE-A/B/C each need one: within A the proposal IS the prior, up to the
constant 1/P(A), so the NPE loss is unchanged there and zero outside.

WHY THE REGION LIVES IN THE FISHER EIGENBASIS (guardrail 3). ``k``, ``delta_E`` and ``temp`` sit at or
near prior on ``posterior_08232026``, and ``k`` in particular is FLAT rather than aliased -- 99.9% of
its weight on one direction loading -1.00*k. An HPD box drawn in 13-D physical space would cut those
axes on NOISE, and deleted support is permanent: truncation is a one-way ratchet, and a round-2 run
cannot recover a region round 1 threw away. So the region is expressed along the flow's own latent
axes, which under REPARAM_ROTATE *are* V's columns -- truncate directions 0..K-1, leave the flat ones
full width. This also dissolves the clustering concern outright: in that basis the
region is approximately axis-aligned, so there is no curved ridge to fragment and no clustering
needed at all.

THE REGION CARRIES ITS BASIS (guardrail 7, 2026-09-09). A box over "directions 0..K-1" means nothing
without the V those directions are columns of, and V is NOT reproducible across processes: the Fisher
draws its operating points from the unseeded global RNG, so a retrain that recomputes it gets a
different rotation with a different column order. The 2026-09-02 round did exactly that -- drew the
region in the parent's basis and enforced it in a fresh one -- and the truncated prior kept 0.01% of
the parent posterior's mass and excluded the ground truth (Appendix A 2026-09-09, defect D1). So a
``TruncationRegion`` records the parent's V and its bijection probe, and ``build_posterior`` REUSES that
V for a truncated round, refusing on any mismatch, rather than ever running the Fisher again.
"""
import hashlib
import math

import torch

from core.SBI import reparam as _reparam, training_checkpoint as _tc


def t_scale_loading_max(n_latent: int) -> float:
    """A direction whose |V[t_scale, j]| exceeds this is NOT truncated.

    gen_training_data overwrites t_scale per batch AFTER the proposal draw and recomputes the latent
    target, so along a direction that loads on t_scale a box is not a restriction at all: the rows
    that reach the simulator have been carried out of it, the proposal becomes the prior TILTED by
    P(A | theta_-t) (a no-op when the direction IS the t_scale axis), and NPE converges to
    p(theta|x) * P(A|theta_-t) rather than the truncated posterior (defect D4, Appendix A 2026-09-09).
    The round-0 rotation puts t_scale on direction 0 alone; round 1's spread it 0.66/0.75 over two.

    1/sqrt(d) is the RMS entry of a random d-dimensional rotation -- the post-mortem's 0.1 sat below
    it and would have flagged directions that barely touch t_scale (a judgement taken with the user
    on 2026-09-09). The governing scalar is really sum_j V[t_scale, j]^2 over the truncated
    directions, which region_from_posterior prints.
    """
    return 1.0 / math.sqrt(max(1, int(n_latent)))


# 99.9%, not 95%: guardrail 5. The cost of an over-wide region is simulations; the cost of a narrow
# one is deleted support that no later round can recover.
DEFAULT_HPD = 0.999
# How many of the best-constrained directions to truncate. The rest keep full prior width.
DEFAULT_N_DIRECTIONS = 5

# The rejection sampler's blind first guess at P(A), and the ceiling on any single proposal draw.
# Both exist to keep a tight region from turning into one enormous allocation before there is any
# measurement to size it from.
_FIRST_PASS_RATE = 0.25
_MAX_DRAW = 1_000_000


def rotation_digest(V) -> str | None:
    """16-hex sha256 of a rotation's float64 bytes, or None for an unrotated run.

    Same shape as ``orchestrator._gmm_fingerprint`` and ``observation_digest``. The caveat in
    ``training_checkpoint.bijection_probe`` -- that hashing V's bytes is a brittle way to VERIFY a
    rotation -- does not apply here: the V this names is carried verbatim with the region and copied,
    never recomputed, so its bytes are stable by construction. The digest is only ever a NAME (the
    region's ``identity_fields``, the region's repr); the actual check is
    ``TruncationRegion.check_basis``, which compares the matrices and the probe.
    """
    if V is None:
        return None
    b = V.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


class TruncationRegion:
    """An axis-aligned box in the flow's LATENT coordinate, over a subset of directions.

    Latent, not physical, and that is load-bearing -- see the module docstring. ``dims`` are indices
    into the latent vector; under REPARAM_ROTATE, dim j is column j of the Fisher rotation V, so it
    is the same "direction j" that scripts/posterior_identifiability.py reports.

    ``V`` and ``probe`` are the basis those indices refer to: the parent posterior's rotation
    (eigenvectors in COLUMNS, ``w = z @ V``; None for an unrotated run) and
    ``training_checkpoint.bijection_probe`` of the parent's whole training bijection (box + rotation).
    ``x_obs_digest`` is the observation the region was drawn around (``observation_digest``), and
    ``prior_fingerprint`` the GMM fingerprint (``run_guards._gmm_fingerprint``) of the parent's
    training prior -- the base prior the box restricts. A region built by
    ``orchestrator.build_truncation_region`` always carries the first three, and the fingerprint
    whenever the parent's prior has a GMM to fingerprint; ``build_posterior`` reuses the
    V, refuses through ``check_basis`` to apply the box in any other coordinate, and refuses a
    supplied prior whose fingerprint verifiably differs. They are optional here only so a
    hand-built region (tests, legacy sidecars) still constructs.
    """

    def __init__(self, dims, lo, hi, *, level: float = DEFAULT_HPD, n_latent: int | None = None,
                 V=None, probe=None, x_obs_digest: str | None = None,
                 excluded=None, t_scale_idx: int | None = None,
                 prior_fingerprint: str | None = None):
        self.dims = [int(d) for d in dims]
        # Directions region_from_posterior skipped for their t_scale loading (a record, not a
        # constraint), and the latent index of t_scale it judged them by.
        self.excluded = [int(d) for d in (excluded or [])]
        self.t_scale_idx = None if t_scale_idx is None else int(t_scale_idx)
        self.lo = torch.as_tensor(lo, dtype=torch.float64).reshape(-1)
        self.hi = torch.as_tensor(hi, dtype=torch.float64).reshape(-1)
        if not (len(self.dims) == self.lo.numel() == self.hi.numel()):
            raise ValueError(f"TruncationRegion: {len(self.dims)} dims but {self.lo.numel()} lo / "
                             f"{self.hi.numel()} hi bounds.")
        if bool((self.hi <= self.lo).any()):
            raise ValueError("TruncationRegion: every interval must have hi > lo.")
        self.level = float(level)
        self.n_latent = None if n_latent is None else int(n_latent)
        # V keeps its own dtype (the checkpoint header stores `V.detach().cpu()` the same way, so the
        # two compare bitwise); the probe is float64 on the CPU, as bijection_probe returns it. Both
        # are CLONED: rotation_of returns a transposed VIEW of the parent transform's own matrix, and
        # a region is a measurement, not an alias of the thing it measured.
        self.V = None if V is None else V.detach().cpu().clone()
        self.probe = None if probe is None else probe.detach().to(torch.float64).cpu().clone()
        self.x_obs_digest = None if x_obs_digest is None else str(x_obs_digest)
        # The GMM fingerprint of the parent's TRAINING prior -- the base prior this box restricts.
        # check_basis compares V and the box but not the GMM, so without this a round started with
        # another prior loaded would train that prior restricted to a box nobody measured on it,
        # with every basis check green. None for a hand-built region or a pre-2026-09-10 sidecar.
        self.prior_fingerprint = None if prior_fingerprint is None else str(prior_fingerprint)
        if self.V is not None:
            if self.V.dim() != 2 or self.V.shape[0] != self.V.shape[1]:
                raise ValueError(f"TruncationRegion: V must be a square (P, P) rotation, got "
                                 f"{tuple(self.V.shape)}.")
            if self.n_latent is not None and self.V.shape[0] != self.n_latent:
                raise ValueError(f"TruncationRegion: V is {tuple(self.V.shape)} but the region has "
                                 f"n_latent={self.n_latent}.")
        if (self.probe is not None and self.probe.dim() == 2 and self.n_latent is not None
                and self.probe.shape[1] != self.n_latent):
            raise ValueError(f"TruncationRegion: probe is {tuple(self.probe.shape)} but the region "
                             f"has n_latent={self.n_latent}.")

    def contains(self, z: torch.Tensor) -> torch.Tensor:
        """(N, P) latent -> (N,) bool. Untruncated directions are unconstrained by construction."""
        lo = self.lo.to(z.device, z.dtype)
        hi = self.hi.to(z.device, z.dtype)
        sel = z[:, self.dims]
        return ((sel >= lo) & (sel <= hi)).all(dim=1)

    def check_basis(self, T_train, *, dim: int, device=None, atol: float = 1e-6) -> None:
        """Refuse unless ``T_train`` is the bijection this region was measured in.

        Two comparisons, and both are needed. The rotation is compared DIRECTLY (``reparam.rotation_of``
        against the recorded V): a probe alone cannot see a permutation of V's columns, because the
        probe grid's rows have all coordinates equal and a column permutation leaves ``z @ V^T``
        unchanged -- and a permuted V is exactly a same-eigenvectors, different-order rotation, the
        kind of "almost the same basis" that would put the box's dims on other directions with no
        numeric warning. The PROBE is then compared too, because the box can change with V held fixed
        (bounds, a log-box setting), and the probe is what catches that.
        """
        if self.probe is None or self.probe.numel() == 0:
            raise ValueError(
                "This TruncationRegion carries no bijection probe (built by hand, or from a sidecar "
                "written before the basis travelled with the region), so the coordinate its box refers "
                "to cannot be verified. Rebuild it with orchestrator.build_truncation_region.")
        V_train = _reparam.rotation_of(T_train)
        if (V_train is None) != (self.V is None):
            raise ValueError(
                f"The truncation region was measured {'WITH' if self.V is not None else 'WITHOUT'} a "
                f"Fisher rotation, but the training bijection has "
                f"{'none' if V_train is None else 'one'}. Its box indexes the parent posterior's "
                f"latent directions; applying it along other axes deletes support the parent never "
                f"excluded (defect D1, Appendix A 2026-09-09).")
        if self.V is not None:
            a = V_train.detach().cpu().to(torch.float64)
            b = self.V.to(torch.float64)
            if a.shape != b.shape or not torch.allclose(a, b, rtol=0, atol=atol):
                diff = float((a - b).abs().max()) if a.shape == b.shape else float("inf")
                raise ValueError(
                    f"The training bijection rotates by a DIFFERENT V than the one the truncation region "
                    f"was measured in (max|diff| = {diff:.3g}). The box's dims index the PARENT "
                    f"posterior's Fisher directions; along any other rotation the same numbers select "
                    f"a slab the parent never occupied -- the 2026-09-02 round kept 0.01% of the parent "
                    f"posterior that way (defect D1). A truncated round must reuse the region's V and "
                    f"never recompute the Fisher.")
        got = _tc.bijection_probe(T_train, dim, device=device)
        if got.shape != self.probe.shape:
            raise ValueError(f"The training bijection's probe is {tuple(got.shape)} but the region's is "
                             f"{tuple(self.probe.shape)}: a different parameter count or grid.")
        if not torch.allclose(got, self.probe, rtol=atol, atol=atol):
            raise ValueError(
                f"The training bijection differs from the one the truncation region was measured in "
                f"(probe max|diff| = {float((got - self.probe).abs().max()):.3g}) although the rotation "
                f"matches: either the BOX changed -- bounds, or a log-box setting -- since the parent "
                f"was trained, or the region's recorded probe does not describe the bijection its own V "
                f"builds. Load the config the parent was trained with, or rebuild the region from a "
                f"posterior trained under this one.")

    def identity_fields(self) -> dict:
        """The region as the checkpoint identity records it: JSON-able, and enough to say whether a
        stored checkpoint's rows were drawn under THIS region -- dims, level, both bounds, and the
        rotation's digest. A checkpoint whose identity lacks this record, or records another region,
        holds rows this round must not resume onto.

        The bounds are QUANTISED to five significant digits. They come from posterior draws, and the
        checkpoint directory is named from this dict, so for a crashed round to find its own rows
        again the redrawn box must digest to the same name: ``region_from_posterior`` seeds the draw
        for that, and the quantisation absorbs the last-ULP differences a GPU can still introduce.
        ``contains`` uses the exact bounds; only the NAME is rounded.
        """
        return {"dims": list(self.dims), "level": float(self.level),
                "lo": [float(f"{float(v):.5g}") for v in self.lo.tolist()],
                "hi": [float(f"{float(v):.5g}") for v in self.hi.tolist()],
                "V_digest": rotation_digest(self.V)}

    def check_checkpoint_V(self, V_stored, *, where: str = "the training checkpoint",
                           atol: float = 1e-6) -> None:
        """Refuse to resume onto rows whose stored rotation is not this region's basis.

        A checkpoint's rows are LATENT targets in the V stored beside them. Reusing them under
        another V would train the flow on targets from two coordinates -- and reusing them under
        this region's V while the header says otherwise means the rows are not this round's at all.
        """
        if (V_stored is None) != (self.V is None):
            raise ValueError(
                f"{where} stores {'no rotation' if V_stored is None else 'a rotation'} but the "
                f"truncation region was measured {'with one' if self.V is not None else 'without one'}. "
                f"Its rows are latent targets in another coordinate and cannot be reused for this "
                f"round; rename that directory or change the budget so the round starts its own.")
        if self.V is None:
            return
        a = V_stored.detach().cpu().to(torch.float64)
        b = self.V.to(torch.float64)
        if a.shape != b.shape or not torch.allclose(a, b, rtol=0, atol=atol):
            diff = float((a - b).abs().max()) if a.shape == b.shape else float("inf")
            raise ValueError(
                f"{where} stores a rotation that is not the one the truncation region was measured in "
                f"(max|diff| = {diff:.3g}). Its rows are latent targets in ANOTHER basis and cannot be "
                f"reused for this round. Rename that directory or change the budget so the round starts "
                f"its own checkpoint.")

    def to_dict(self) -> dict:
        return {"basis": "fisher-latent", "dims": list(self.dims), "level": self.level,
                "lo": self.lo.clone(), "hi": self.hi.clone(), "n_latent": self.n_latent,
                # The basis itself, so a sidecar can say which coordinate its box is in and a later
                # round can reuse it. Absent from sidecars written before 2026-09-09 -> None.
                "V": None if self.V is None else self.V.clone(),
                "probe": None if self.probe is None else self.probe.clone(),
                "V_digest": rotation_digest(self.V),
                "x_obs_digest": self.x_obs_digest,
                "prior_fingerprint": self.prior_fingerprint,
                "excluded": list(self.excluded), "t_scale_idx": self.t_scale_idx}

    @staticmethod
    def from_dict(d: dict) -> "TruncationRegion":
        return TruncationRegion(d["dims"], d["lo"], d["hi"],
                                level=d.get("level", DEFAULT_HPD), n_latent=d.get("n_latent"),
                                V=d.get("V"), probe=d.get("probe"), x_obs_digest=d.get("x_obs_digest"),
                                excluded=d.get("excluded"), t_scale_idx=d.get("t_scale_idx"),
                                prior_fingerprint=d.get("prior_fingerprint"))

    def __repr__(self) -> str:
        parts = ", ".join(f"d{d}:[{float(a):.3g},{float(b):.3g}]"
                          for d, a, b in zip(self.dims, self.lo, self.hi))
        basis = "unrotated" if self.V is None else f"V:{rotation_digest(self.V)[:8]}"
        return f"TruncationRegion(level={self.level:.4g}, {parts}, basis={basis})"


def region_from_posterior(posterior_latent, x_obs: torch.Tensor, *,
                          n_directions: int = DEFAULT_N_DIRECTIONS,
                          level: float = DEFAULT_HPD, n_samples: int = 20000,
                          V=None, probe=None, x_obs_digest: str | None = None,
                          t_scale_idx: int | None = None,
                          prior_fingerprint: str | None = None,
                          max_loading: float | None = None) -> TruncationRegion:
    """Draw from the posterior at ``x_obs`` and take a per-direction HPD interval in LATENT space.

    ⚠ GUARDRAIL 4: UNWEIGHTED draws. Not "the M best fits". Selecting on goodness of fit applies a
    second, undeclared likelihood with the discrepancy metric as a hidden hyperparameter --
    ``overlay.posterior_overlay`` takes the best 50, which is right for a figure and wrong as the seed
    of a prior.

    The interval is a marginal quantile range per direction, which is what makes the region a box.
    Its mass is NOT "at least the joint HPD's": for k independent directions a box of per-direction
    level q holds q^k of the posterior (0.999^5 = 0.995), and the union bound is the honest statement
    -- the box misses at most k(1 - q) of the posterior's mass. Generous by construction, then, and
    the level is 99.9% for exactly that reason (guardrail 5).

    ``V`` and ``probe`` are the parent posterior's basis, recorded on the region (see
    ``TruncationRegion``); ``orchestrator.build_truncation_region`` always supplies them, and
    ``posterior_latent``'s samples ARE coordinates in that V -- the flow was trained on ``w = z @ V``.
    ``prior_fingerprint`` is the parent's training-prior GMM digest, recorded verbatim on the region so
    ``build_posterior`` can refuse another base prior; None when the caller cannot name one.

    ``t_scale_idx`` is t_scale's index in the latent ([ND | rescale] order); when it is given, any
    direction whose |V[t_scale_idx, j]| exceeds ``max_loading`` (default ``t_scale_loading_max``) is
    SKIPPED and the box takes the first ``n_directions`` eligible ones instead -- the per-batch
    t_scale override would carry rows out of a box along such a direction, turning the restriction
    into a reweighting (defect D4). With V None the latent is unrotated and the only loaded
    direction is t_scale's own axis. Skipped directions are recorded on the region as ``excluded``.

    THE DRAW IS SEEDED, from the observation and the settings, under ``fork_rng`` so the caller's
    stream is untouched. The region is part of the round's checkpoint identity, so the SAME
    posterior, observation, level, direction count and sample count must redraw the SAME box: that is
    what lets a round that died at batch 3000 of 5000 find its own rows again instead of drawing a
    new box, routing to a new directory and simulating from zero while the old shards rot.
    """
    if x_obs is None:
        raise ValueError(
            "region_from_posterior needs the observation the region is being drawn around. An "
            "amortized posterior has no default_x -- persist x_obs at INFERENCE time and pass it "
            "here.")
    seed_src = (x_obs.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
                + f"|{int(n_directions)}|{float(level)!r}|{int(n_samples)}".encode("utf-8"))
    seed = int(hashlib.sha256(seed_src).hexdigest()[:8], 16)
    dev = getattr(posterior_latent, "device", None)
    fork_devices = [torch.device(dev)] if dev is not None and torch.device(dev).type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        z = posterior_latent.sample((int(n_samples),), x=x_obs)
    z = z.detach().to(torch.float64).cpu()
    p = z.shape[-1]
    if V is not None and tuple(V.shape) != (p, p):
        raise ValueError(f"region_from_posterior: the posterior's latent is {p}-dimensional but V is "
                         f"{tuple(V.shape)}; that is not the basis these samples are in.")
    k = max(1, min(int(n_directions), p))
    tail = (1.0 - float(level)) / 2.0
    q = torch.tensor([tail, 1.0 - tail], dtype=torch.float64)
    if t_scale_idx is None:
        dims, excluded = list(range(k)), []     # latent axes are already sorted best-constrained first
    else:
        i_t = int(t_scale_idx)
        if not 0 <= i_t < p:
            raise ValueError(f"region_from_posterior: t_scale_idx={i_t} is outside the {p}-dimensional latent.")
        limit = t_scale_loading_max(p) if max_loading is None else float(max_loading)
        load = (V.detach().cpu().to(torch.float64).abs()[i_t] if V is not None
                else torch.eye(p, dtype=torch.float64)[i_t])
        eligible = [j for j in range(p) if float(load[j]) <= limit]
        dims = eligible[:k]
        # Everything passed over while filling the box: up to the last kept direction when the box
        # filled, the WHOLE latent when the eligible set ran out first (then every ineligible
        # direction really was skipped, and the record must say so).
        scanned = (p if len(dims) < k else dims[-1] + 1) if dims else p
        excluded = [j for j in range(scanned) if j not in dims]
        for j in excluded:
            src = f"|V[t_scale, {j}]|" if V is not None else f"the t_scale axis' weight on direction {j} (unrotated latent)"
            print(f"[tsnpe] direction {j} NOT truncated: {src} = {float(load[j]):.3f} > {limit:.3f}. The "
                  f"per-batch t_scale override would carry rows out of a box along it, turning the "
                  f"restriction into a reweighting (D4).", flush=True)
        if len(dims) < k:
            print(f"[tsnpe] only {len(dims)} of the requested {k} directions are eligible for truncation.",
                  flush=True)
        if dims:
            print(f"[tsnpe] fraction of the t_scale axis inside the truncated subspace: "
                  f"{float((load[dims] ** 2).sum()):.3f} (0 = the override cannot move a row out of the box)",
                  flush=True)
    if not dims:
        raise ValueError("region_from_posterior: every direction loads on t_scale above the limit; "
                         "nothing can be truncated.")
    bounds = torch.quantile(z[:, dims], q, dim=0)
    return TruncationRegion(dims, bounds[0], bounds[1], level=level, n_latent=p, V=V, probe=probe,
                            x_obs_digest=x_obs_digest, excluded=excluded, t_scale_idx=t_scale_idx,
                            prior_fingerprint=prior_fingerprint)


class TruncatedLatentPrior:
    """The latent prior RESTRICTED to a region. Sampling is rejection; nothing is reweighted.

    Deliberately a thin wrapper with the same duck-type ``gen_training_data`` already expects of a
    prior (``sample``, ``log_prob``), rather than a torch Distribution: the latent prior it wraps is
    itself a hand-built ProductPrior/RotatedLatentPrior, and the pipeline's standing rule for it is
    sample-only.

    ``log_prob`` returns the BASE log-density inside the region and -inf outside, i.e. it is off by
    the constant log P(A). NPE never needs that constant -- the loss is over q(theta|x) and, because
    truncation is a restriction rather than a reweighting, TSNPE applies no proposal correction at
    all. Anything that does need a normalised density must estimate P(A) itself; ``acceptance_rate``
    is the estimator.
    """

    def __init__(self, base, region: TruncationRegion, *, max_tries: int = 64):
        self.base = base
        self.region = region
        self.max_tries = int(max_tries)
        self._accepted = 0
        self._proposed = 0
        # What gen_training_data actually RECORDED as inside the region, after its per-batch t_scale
        # override -- the sampler's own count above describes draws BEFORE it. None until measured.
        self._recorded_inside = 0
        self._recorded_total = 0

    def note_recorded(self, inside: int, total: int) -> None:
        """gen_training_data's tally of post-override latent targets inside the region."""
        self._recorded_inside += int(inside)
        self._recorded_total += int(total)

    @property
    def recorded_counts(self) -> tuple:
        """``(inside, total)`` of the RECORDED training targets -- resumed rows included, since a
        checkpoint a truncated round resumes was drawn under this very region."""
        return self._recorded_inside, self._recorded_total

    @property
    def recorded_containment(self) -> float | None:
        """Fraction of the RECORDED training targets inside the region, or None when nothing was
        recorded in this process. This is the number that says how much of the training set actually
        lies in the region; ``acceptance_rate`` does not."""
        return (self._recorded_inside / self._recorded_total) if self._recorded_total else None

    def sample(self, sample_shape=torch.Size()):
        n = int(torch.Size(sample_shape).numel()) if len(torch.Size(sample_shape)) else 1
        out, got, tries = [], 0, 0
        while got < n and tries < self.max_tries:
            # Over-draw by the MEASURED acceptance rate, and only once there is one to measure.
            # Seeding the first pass with a 1e-3 floor asked for (n / 1e-3) * 1.3 draws before any
            # evidence -- 2.66 MILLION rows for a 2048-row batch, out of a 13-D GMM, on the very
            # first call. The first pass is a probe: draw a modest multiple, then let the observed
            # rate size the rest. _MAX_DRAW caps any single allocation so a pathologically tight
            # region fails through max_tries with a message rather than through an OOM.
            rate = self.acceptance_rate if self._proposed else _FIRST_PASS_RATE
            want = int((n - got) / max(rate, 1e-6) * 1.3) if rate else (n - got) * 4
            draw = self.base.sample((min(max(want, n - got, 64), _MAX_DRAW),))
            keep = draw[self.region.contains(draw)]
            self._proposed += draw.shape[0]
            self._accepted += keep.shape[0]
            if keep.shape[0]:
                out.append(keep[: n - got])
                got += out[-1].shape[0]
            tries += 1
        if got < n:
            raise RuntimeError(
                f"TruncatedLatentPrior: only {got} of {n} draws landed inside the truncation region "
                f"after {tries} attempts (acceptance ~{self.acceptance_rate:.2e}). The region is far "
                f"out in the prior's tail, which usually means the posterior it came from disagrees "
                f"with the prior rather than sharpening it -- check the observation before widening "
                f"the HPD level.")
        z = torch.cat(out, dim=0)
        return z if len(torch.Size(sample_shape)) else z[0]

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        lp = self.base.log_prob(theta)
        inside = self.region.contains(theta.reshape(-1, theta.shape[-1]))
        return torch.where(inside.reshape(lp.shape), lp, torch.full_like(lp, float("-inf")))

    @property
    def acceptance_rate(self) -> float:
        """Measured P(A) under the prior, at the rejection sampler -- i.e. BEFORE gen_training_data's
        per-batch t_scale override, which re-opens any t_scale-loaded direction. For what the training
        set actually contains see ``recorded_containment``. 1 - this is the fraction of prior mass the
        region excludes along its directions (guardrail 5's honest failure rate)."""
        return (self._accepted / self._proposed) if self._proposed else 0.0

    def __getattr__(self, name):
        # Everything else (device, dims, event_shape, ...) is the base prior's business.
        # KeyError would be WRONG here: __getattr__ is consulted before __init__ has run during
        # copy/pickle, and Python requires an AttributeError to treat the attribute as absent --
        # a KeyError escapes and breaks copying instead of falling through.
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(base, name)
