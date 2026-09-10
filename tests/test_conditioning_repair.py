"""The informativeness programme: conditioning repair, tier-1 consistency, TSNPE.

Pure torch, no simulation, no Qt -- seconds. Deliberately a separate suite in the spirit of
test_chi_set_encoder: the invariants here are the kind whose violation is INVISIBLE. A contaminated
standardiser trains perfectly happily and produces a posterior nobody can explain (that is exactly
what `posterior_08232026` is), and a TSNPE round that proposes from the posterior instead of the
truncated prior contracts its credible intervals with no new information and passes SBC while doing
it. Neither shows up as a crash.

Every test below was checked to FAIL against the pre-change code.

Run:  python tests/test_conditioning_repair.py
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core import config
from core.SBI import derived, pipeline, reparam, statistics, training_checkpoint, truncate
from core.SBI.embedded_network import EmbeddedNet, _probit

_SENT = float(torch.log(torch.tensor(1e-12, dtype=torch.float32)))


def _fitted_net(data: torch.Tensor, n_sum: int, k_pad: int = 4):
    net = EmbeddedNet(input_dim=n_sum, output_dim=8, layer_dims=(16, 12),
                      forcing_dim=config.CHI_ELEM_W * k_pad, forcing_layer_dims=(16, 8),
                      merge_layer_dim=16, chi_k_pad=k_pad, chi_band=config.CHI_FREQ_BOUNDS)
    net.fit_standardization(data)
    return net


# ── 1.1 rank-Gaussianisation ─────────────────────────────────────────────────────────────────────
def test_rank_gaussianize_is_monotone_and_bounded():
    """The transform must be order-preserving; if it is not, it is not a reparameterisation of the
    channel at all and the flow is being shown a different variable than the one measured."""
    torch.manual_seed(0)
    col = torch.randn(20000, 1) * 3.0 + 7.0
    q = 256
    p = (torch.arange(q, dtype=torch.float64) + 0.5) / q
    knots = torch.quantile(col[:, 0].double(), p).float().reshape(1, q)
    z = _probit(p).float()
    keep = torch.ones(1, dtype=torch.uint8)

    probe = torch.linspace(float(col.min()) - 5, float(col.max()) + 5, 997).reshape(-1, 1)
    out = EmbeddedNet.rank_gaussianize(probe, knots, z, keep).reshape(-1)
    assert bool((out[1:] >= out[:-1] - 1e-6).all()), "rank-Gaussianisation is not monotone"
    assert float(out.min()) >= float(z[0]) - 1e-6 and float(out.max()) <= float(z[-1]) + 1e-6, \
        "values outside the fitted range must clamp to the extreme knots, not extrapolate"


def test_a_large_point_mass_maps_to_its_MID_rank_not_an_edge():
    """69.7% of B1_log_Q sits on exactly log(1e-12). A run of identical knots must resolve to the
    MIDDLE of its own rank interval -- taking whichever end searchsorted returns would make the sign
    of the resulting jump depend on a float comparison rather than on the data."""
    q = 512
    p = (torch.arange(q, dtype=torch.float64) + 0.5) / q
    z = _probit(p).float()
    # 60% of the mass on one value, sitting between a lower and an upper tail.
    knots = torch.cat([torch.linspace(-3.0, -1.0, 100),
                       torch.full((312,), 0.0),
                       torch.linspace(1.0, 3.0, 100)]).reshape(1, q)
    keep = torch.ones(1, dtype=torch.uint8)
    got = float(EmbeddedNet.rank_gaussianize(torch.zeros(1, 1), knots, z, keep))
    want = 0.5 * (float(z[100]) + float(z[411]))
    assert abs(got - want) < 1e-5, f"tie mapped to {got:.4f}, expected the mid-rank {want:.4f}"
    # and it must not be either edge of the run, which is the bug this guards
    assert abs(got - float(z[100])) > 1e-3 and abs(got - float(z[411])) > 1e-3, \
        "the point mass landed on an EDGE of its rank interval"


def test_a_structurally_dead_channel_passes_through_as_zero():
    """Group G is identically zero under chi. Ranking a constant is undefined, and out of
    distribution it would clamp to +-z_max and inject a full-scale signal from a channel that carries
    nothing."""
    torch.manual_seed(1)
    n_sum = statistics.SUMMARY_WIDTH + 1
    data = torch.randn(4000, n_sum + config.CHI_ELEM_W * 4)
    data[:, 5] = 0.0                                            # a dead channel
    net = _fitted_net(data, n_sum)
    assert int(net.rg_keep[5]) == 0, "a constant channel was not marked pass-through"
    out = net.standardize_summary(data[:, :n_sum])
    assert torch.equal(out[:, 5], torch.zeros(data.shape[0])), "a dead channel produced non-zero output"
    # an out-of-distribution value on that channel must STILL be inert
    probe = data[:1, :n_sum].clone()
    probe[0, 5] = 1e6
    assert float(net.standardize_summary(probe)[0, 5]) == 0.0, \
        "a dead channel amplified an out-of-distribution value"


def test_the_contaminated_channel_becomes_visible_to_the_network():
    """THE PHASE-1 GATE, in miniature. A channel whose distribution carries an extreme outlier is
    annihilated by a mean/std affine (measured on posterior_08232026: A1_mean fitted at std 4.19e11,
    so its whole physical range moved the embedding by 3.2e-7). Under the rank transform its response
    must be the same order as a healthy channel's."""
    torch.manual_seed(2)
    n_sum = statistics.SUMMARY_WIDTH + 1
    n = 8000
    data = torch.randn(n, n_sum + config.CHI_ELEM_W * 4)
    bad = 0
    data[:, bad] = torch.randn(n) * 3.0
    data[0, bad] = -4.6e29                                       # the pathological trajectory
    net = _fitted_net(data, n_sum)

    def response(j):
        base = data.median(0).values.unsqueeze(0)
        lo = torch.quantile(data[:, j].float(), 0.01)
        hi = torch.quantile(data[:, j].float(), 0.99)
        v = base.repeat(2, 1)
        v[0, j], v[1, j] = lo, hi
        with torch.no_grad():
            return float((net(v) - net(base)).norm(dim=-1).max())

    healthy = sorted(response(j) for j in (1, 2, 3, 4))[1]       # a typical channel
    assert response(bad) > healthy / 100, (
        f"the outlier-bearing channel moves the embedding by {response(bad):.3g} against a healthy "
        f"channel's {healthy:.3g} -- it is still being annihilated by its own standardiser")


def test_a_legacy_posterior_still_loads_and_evaluates():
    """A pre-2026-08-26 posterior unpickles with sum_mean/sum_std and no rank buffers. Without a
    branch on which buffers are present, EVERY existing artifact becomes unloadable -- including
    posterior_08232026, which is the baseline every conditioning-repair gate is measured against."""
    torch.manual_seed(3)
    n_sum = 42
    net = EmbeddedNet(input_dim=n_sum, output_dim=8, layer_dims=(16, 12),
                      forcing_dim=config.CHI_ELEM_W * 4, forcing_layer_dims=(16, 8),
                      merge_layer_dim=16, chi_k_pad=4, chi_band=config.CHI_FREQ_BOUNDS)
    # Reproduce the on-disk shape of an old artifact: legacy buffers, no rank buffers.
    for k in ("rg_knots", "rg_z", "rg_keep"):
        del net._buffers[k]
    net.register_buffer("sum_mean", torch.zeros(n_sum))
    net.register_buffer("sum_std", torch.full((n_sum,), 2.0))
    net.forcing_net.fitted.fill_(1)
    x = torch.randn(5, n_sum + config.CHI_ELEM_W * 4)
    got = net.standardize_summary(x[:, :n_sum])
    assert torch.allclose(got, x[:, :n_sum] / 2.0), "the legacy affine was not applied"
    assert net(x).shape == (5, 8), "a legacy net could not run forward"


# ── 1.2 valid flags ──────────────────────────────────────────────────────────────────────────────
def test_valid_flags_fire_on_the_sentinel_and_only_on_it():
    n = len(statistics.FEATURE_LABELS)
    feats = torch.randn(3, n)
    feats[1, statistics.FEATURE_LABELS.index("B1_log_Q")] = _SENT
    feats[2, statistics.FEATURE_LABELS.index("C7_log_slowenv_relvar")] = _SENT
    fl = statistics.derive_valid_flags(feats, 1.0)
    assert fl.shape == (3, len(statistics.VALID_FLAG_LABELS))
    b1 = statistics.VALID_FLAG_LABELS.index("V_B1_Q")
    c7 = statistics.VALID_FLAG_LABELS.index("V_C7_slowenv")
    assert float(fl[1, b1]) == 0.0 and float(fl[0, b1]) == 1.0
    assert float(fl[2, c7]) == 0.0 and float(fl[0, c7]) == 1.0


def test_the_B7_pair_shares_one_flag():
    """freq==0 and height==0 are written together by `has_sec`, and over 10.24M cached rows the XOR
    fired ZERO times -- which is what licenses one flag for two columns."""
    n = len(statistics.FEATURE_LABELS)
    feats = torch.randn(2, n)
    feats[1, statistics.FEATURE_LABELS.index("B7_log_sec_freq_ratio")] = 0.0
    feats[1, statistics.FEATURE_LABELS.index("B7_sec_height_ratio")] = 0.0
    fl = statistics.derive_valid_flags(feats, 1.0)
    j = statistics.VALID_FLAG_LABELS.index("V_B7_secondary")
    assert float(fl[0, j]) == 1.0 and float(fl[1, j]) == 0.0


def test_the_tau_slow_flag_follows_dt():
    """E1_log_tau_slow's sentinel is log(1e6 * dt), so a wrong dt would flag every row valid. That is
    why derive_valid_flags takes dt as a REQUIRED argument rather than defaulting it."""
    n = len(statistics.FEATURE_LABELS)
    j = statistics.FEATURE_LABELS.index("E1_log_tau_slow")
    k = statistics.VALID_FLAG_LABELS.index("V_E1_tau_slow")
    feats = torch.zeros(1, n)
    feats[0, j] = math.log(1e6 * 4.0)
    assert float(statistics.derive_valid_flags(feats, 4.0)[0, k]) == 0.0, "sentinel not detected at dt=4"
    assert float(statistics.derive_valid_flags(feats, 1.0)[0, k]) == 1.0, "dt was ignored"


def test_the_summary_block_splits_where_it_claims():
    s = torch.randn(4, statistics.SUMMARY_WIDTH)
    feats, flags = statistics.split_summary_block(s)
    assert feats.shape[-1] == len(statistics.FEATURE_LABELS)
    assert flags.shape[-1] == len(statistics.VALID_FLAG_LABELS)
    assert torch.equal(torch.cat([feats, flags], dim=-1), s)


# ── 1.3 winsorisation ────────────────────────────────────────────────────────────────────────────
def test_winsorisation_leaves_the_chi_block_BITWISE_untouched():
    """A padded probe slot is exactly 0.0 in all six channels and must stay bitwise inert. Clipping a
    probe column whose 0.1th percentile is non-zero would push every pad off 0.0 and turn it into a
    phantom probe -- the exact defect the packer's nan_to_num removal fixed once already."""
    torch.manual_seed(4)
    n_sum, k_pad = statistics.SUMMARY_WIDTH + 1, 4
    chi_w = config.CHI_ELEM_W * k_pad
    data = torch.cat([torch.randn(2000, n_sum) * 50.0,
                      torch.rand(2000, chi_w) + 1.0], dim=1)     # chi block strictly positive
    data[0, n_sum:] = 0.0                                        # a fully padded row
    # SNAPSHOT FIRST. winsorize_summary_block clips IN PLACE and returns the same tensor (it is 5 GiB
    # at the production shape, so a copy-and-cat would triple the host peak) -- so comparing `out`
    # against `data` afterwards compares a tensor with itself and asserts nothing. This test was
    # vacuous for exactly that reason until the in-place change exposed it.
    before = data.clone()
    out = pipeline.winsorize_summary_block(data, n_sum)
    assert out is data, "winsorisation should clip in place and return the same tensor"
    assert torch.equal(out[:, n_sum:], before[:, n_sum:]), "winsorisation reached into the chi block"
    assert torch.equal(out[0, n_sum:], torch.zeros(chi_w)), "a pad slot stopped being exactly 0.0"


def test_winsorisation_clips_the_outlier_instead_of_dropping_its_row():
    torch.manual_seed(5)
    n_sum = 6
    data = torch.randn(5000, n_sum)
    data[0, 0] = 1e29
    before = data.clone()                                        # in place; see the test above
    out = pipeline.winsorize_summary_block(data, n_sum)
    assert out.shape == before.shape, "winsorisation changed the row count -- it must clip, not filter"
    assert float(out[:, 0].max()) < 100.0, "the 1e29 outlier survived clipping"
    # The row survives, and everything except the clipped extremes is untouched. Asserted as a
    # FRACTION rather than as equality, because a 0.1/99.9 clip is supposed to move ~0.2% of a
    # Gaussian column -- an `assert ... or True` here would be vacuous, which is a defect this
    # project has caught in its own fixtures before.
    moved = float((out != before).float().mean())
    assert 0.0 < moved < 0.01, f"{moved:.4%} of elements moved; expected a fraction of a percent"
    assert float(out[0, 1]) == float(before[0, 1]), "an untouched row's untouched column changed"


# ── 1.4 pathological-trajectory counter ──────────────────────────────────────────────────────────
def test_pathological_counter_separates_the_three_populations():
    acc = dict.fromkeys(("rows", "nonfinite", "constant", "overflow"), 0)
    x = torch.randn(5, 100)
    x[0] = float("nan")
    x[1] = 3.0                                                   # exactly constant -> D3 = 1/_EPS
    x[2] = 1e29                                                  # constant AND overflow
    pipeline.count_pathological(x, acc)
    assert acc["rows"] == 5
    assert acc["nonfinite"] == 1, acc
    assert acc["constant"] == 2, acc                             # the flatline and the overflow row
    assert acc["overflow"] == 1, acc


# ── tier 1: the derived force scale ──────────────────────────────────────────────────────────────
def test_derived_f_scale_round_trips_through_the_implied_temperature():
    nd_idx = {"n": 0, "beta": 1}
    r_idx = {"x_scale": 0, "t_scale": 1, "T": 2}
    k_b = 1.380649e-2
    nd = torch.tensor([[50.0, 14.1], [300.0, 100.0]])
    resc = torch.tensor([[62.14, 3.73, 300.0], [10.0, 2.0, 300.0]])
    sim = derived.to_sim_rescale(nd, resc, r_idx, nd_idx, k_b)
    assert not torch.equal(sim[:, 2], resc[:, 2]), "T was not replaced by the derived f_scale"
    back = derived.implied_temperature(nd, sim, derived.sim_rescale_idx(r_idx), nd_idx, k_b)
    assert torch.allclose(back, resc[:, 2], rtol=1e-4), f"round trip gave {back}, expected 300 K"


def test_a_box_declaring_f_scale_is_untouched_and_the_same_object():
    r_idx = {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    resc = torch.randn(4, 3)
    out = derived.to_sim_rescale(torch.randn(4, 2), resc, r_idx)
    assert out is resc, "the pre-tier-1 path copied the tensor instead of passing it through"
    assert derived.sim_rescale_idx(r_idx) == r_idx


def test_tier1_without_its_inputs_refuses_rather_than_guessing():
    r_idx = {"x_scale": 0, "t_scale": 1, "T": 2}
    try:
        derived.to_sim_rescale(torch.randn(2, 2), torch.randn(2, 3), r_idx)
    except ValueError as e:
        assert "nd_idx" in str(e) and "k_b_cell" in str(e)
        return
    raise AssertionError("a tier-1 box simulated with a TEMPERATURE in f_scale's column")


# ── Phase 2: the informativeness scalar ──────────────────────────────────────────────────────────
def test_prior_log_prob_falls_back_per_block_on_a_device_error():
    """The inferred prior is ProductPrior([nd_gmm, rescale]) and the two halves need not share a
    device -- which is why the rest of the pipeline's rule for this object is "sample-only, never
    .log_prob". One plain call raises on a GPU box, so the scalar would have been silently
    unavailable on exactly the machine that matters. This pins the retry."""
    from core.SBI import analysis

    class _Block:
        def __init__(self, dim, fail_first):
            self.dim, self.calls, self.fail_first = dim, 0, fail_first

        def sample(self, shape):
            n = int(torch.Size(shape).numel())
            return torch.zeros(n, self.dim)

        def log_prob(self, v):
            self.calls += 1
            if self.fail_first and self.calls == 1:
                raise RuntimeError("Expected all tensors to be on the same device")
            return v.sum(-1)

    class _Product:
        def __init__(self, dists, dims):
            self.distributions, self.dims = dists, dims

        def log_prob(self, theta):
            idx, out = 0, None
            for d, dim in zip(self.distributions, self.dims):
                lp = d.log_prob(theta[..., idx: idx + dim])
                out = lp if out is None else out + lp
                idx += dim
            return out

    nd, resc = _Block(3, fail_first=True), _Block(2, fail_first=False)
    prod = _Product([nd, resc], [3, 2])
    theta = torch.arange(10.0).reshape(2, 5)
    got = analysis.prior_log_prob(prod, theta)
    assert nd.calls == 2, "the per-block retry never ran"
    assert torch.allclose(got, theta.sum(-1)), f"the retry returned {got}, expected the row sums"


def test_informativeness_is_zero_when_the_posterior_IS_the_prior():
    """The scalar's zero point, which is the whole reason it exists: a posterior that returns the
    prior learned nothing, and every calibration diagnostic in the project passes it perfectly."""
    from core.SBI import analysis

    class _Same:
        def log_prob_batched(self, theta, x=None):
            return -0.5 * (theta ** 2).sum(-1)

        def log_prob(self, theta):
            return -0.5 * (theta ** 2).sum(-1)

        def sample(self, shape):
            return torch.randn(int(torch.Size(shape).numel()), 3)

    torch.manual_seed(11)
    same = _Same()
    theta = torch.randn(400, 3)
    info = analysis.informativeness(same, theta, torch.zeros(400, 4), same, n_decompose=0)
    assert abs(info["total_nats"]) < 1e-6, f"expected 0 nats, got {info['total_nats']}"
    assert info["n_used"] == 400 and info["n_dropped"] == 0


# ── Phase 4: TSNPE ───────────────────────────────────────────────────────────────────────────────
class _Gaussian:
    """A prior/posterior stand-in with an exact density, so the pinning test has a known answer."""

    def __init__(self, mu, sd, dim=1):
        self.mu, self.sd, self.dim = float(mu), float(sd), dim

    def sample(self, shape=torch.Size()):
        n = int(torch.Size(shape).numel()) if len(torch.Size(shape)) else 1
        z = torch.randn(n, self.dim) * self.sd + self.mu
        return z if len(torch.Size(shape)) else z[0]

    def log_prob(self, theta):
        t = theta.reshape(-1, self.dim)
        return (-0.5 * ((t - self.mu) / self.sd) ** 2 - math.log(self.sd * math.sqrt(2 * math.pi))
                ).sum(-1)


def test_the_proposal_is_the_TRUNCATED_PRIOR_and_not_the_posterior():
    """⚠ THE TEST SBC CANNOT DO, and the one thing TSNPE must not get wrong.

    Proposing from a fitted posterior instead of the prior-restricted-to-A gives p_L ∝ L^(L+1) q --
    tempering. For a Gaussian the round-1 width then contracts by exactly 1/sqrt(2) with NO new
    information entering, and SBC comes out flat anyway because it validates the flow against the
    proposal it was trained on.

    So: build a wide prior and a narrow posterior, take the region from the posterior, and assert the
    proposal's width is the PRIOR's width over that region -- NOT the posterior's, and NOT the
    posterior's over sqrt(2).
    """
    torch.manual_seed(7)
    prior = _Gaussian(0.0, 10.0)
    post = _Gaussian(0.0, 1.0)
    region = truncate.TruncationRegion([0], [-2.0], [2.0], level=0.95, n_latent=1)
    tp = truncate.TruncatedLatentPrior(prior, region)
    draws = tp.sample((40000,)).reshape(-1)

    assert float(draws.min()) >= -2.0 and float(draws.max()) <= 2.0, "a draw escaped the region"
    # A near-flat prior restricted to [-2, 2] is nearly UNIFORM there: sd -> 4/sqrt(12) = 1.155.
    got = float(draws.std())
    assert abs(got - 4.0 / math.sqrt(12.0)) < 0.05, \
        f"proposal sd {got:.4f}; the prior restricted to [-2,2] is ~{4/math.sqrt(12):.4f}"
    # The two failure modes this exists to catch, stated as numbers:
    tempered = 1.0 / math.sqrt(2.0)
    assert abs(got - 1.0) > 0.1, "the proposal has the POSTERIOR's width -- it is sampling the posterior"
    assert abs(got - tempered) > 0.1, "the proposal contracted by 1/sqrt(2) -- this is TEMPERING"


def test_the_region_is_built_over_the_leading_fisher_directions_only():
    """Guardrail 3: k, delta_E and temp sit at or near prior, so cutting every axis would delete
    support on noise -- and deleted support is a one-way ratchet."""
    torch.manual_seed(8)

    class _P:
        def sample(self, shape, x=None):
            n = int(torch.Size(shape).numel())
            return torch.randn(n, 13) * torch.tensor([0.1] * 5 + [5.0] * 8)

    region = truncate.region_from_posterior(_P(), torch.zeros(1, 4), n_directions=5, n_samples=8000)
    assert region.dims == [0, 1, 2, 3, 4], region.dims
    assert region.n_latent == 13
    # the untruncated directions must be unconstrained, whatever value they take
    z = torch.randn(50, 13) * 1000.0
    z[:, :5] = 0.0
    assert bool(region.contains(z).all()), "an untruncated direction was constrained anyway"


def test_the_rejection_sampler_does_not_over_draw_before_it_has_measured_anything():
    """The blind first pass used to seed its acceptance estimate at 1e-3, asking for (n / 1e-3) * 1.3
    draws before any evidence -- 2.66 MILLION rows for a 2048-row batch, out of a 13-D GMM, on the
    very first call. It must probe modestly and then size the rest from the MEASURED rate."""
    torch.manual_seed(12)
    prior = _Gaussian(0.0, 1.0)
    seen = []

    class _Counting:
        def sample(self, shape):
            seen.append(int(torch.Size(shape).numel()))
            return prior.sample(shape)

    tp = truncate.TruncatedLatentPrior(
        _Counting(), truncate.TruncationRegion([0], [-1.0], [1.0]))
    tp.sample((2048,))
    assert seen, "the sampler never drew"
    assert seen[0] <= 2048 * 8, f"first blind draw was {seen[0]:,} rows for 2048 wanted"
    assert max(seen) <= truncate._MAX_DRAW, "a single draw exceeded the allocation cap"


def test_a_region_that_the_prior_almost_never_reaches_says_so():
    torch.manual_seed(9)
    tp = truncate.TruncatedLatentPrior(_Gaussian(0.0, 1.0),
                                       truncate.TruncationRegion([0], [50.0], [51.0]), max_tries=3)
    try:
        tp.sample((100,))
    except RuntimeError as e:
        assert "acceptance" in str(e)
        return
    raise AssertionError("an unreachable truncation region sampled without complaint")


def test_the_truncated_prior_is_the_base_density_inside_and_minus_inf_outside():
    prior = _Gaussian(0.0, 1.0)
    tp = truncate.TruncatedLatentPrior(prior, truncate.TruncationRegion([0], [-1.0], [1.0]))
    inside, outside = torch.tensor([[0.5]]), torch.tensor([[3.0]])
    assert torch.allclose(tp.log_prob(inside), prior.log_prob(inside)), \
        "the truncated prior REWEIGHTED inside the region; truncation is a restriction"
    assert float(tp.log_prob(outside)) == float("-inf")


def test_the_region_survives_a_sidecar_round_trip():
    r = truncate.TruncationRegion([0, 3], [-1.5, 0.25], [2.5, 4.0], level=0.999, n_latent=13)
    back = truncate.TruncationRegion.from_dict(r.to_dict())
    assert back.dims == r.dims and back.level == r.level and back.n_latent == r.n_latent
    assert torch.equal(back.lo, r.lo) and torch.equal(back.hi, r.hi)


# ── Phase 4, guardrail 7: the region carries its basis ───────────────────────────────────────────
class _Wide:
    """A posterior stand-in whose latent has five tight and eight wide directions (13-D)."""

    def sample(self, shape, x=None):
        n = int(torch.Size(shape).numel())
        return torch.randn(n, 13) * torch.tensor([0.1] * 5 + [5.0] * 8)


def _orthogonal(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    return q


def _rotated_box(Q: torch.Tensor, width: float = 10.0):
    p = Q.shape[0]
    box = reparam.build_box_bijection(torch.zeros(p), width * torch.ones(p))
    return box, reparam.build_rotated_bijection(box, Q)


def test_the_region_carries_its_basis_through_a_sidecar_round_trip():
    """A box over 'directions 0..K-1' is meaningless without the V those directions are columns of,
    and V is not reproducible across processes. So the region records the parent's V and the probe
    of its whole training bijection, and both survive the sidecar; a sidecar written before they
    existed loads with None rather than failing."""
    P = 13
    Q = _orthogonal(P, 3)
    _, T_train = _rotated_box(Q)
    probe = training_checkpoint.bijection_probe(T_train, P)
    r = truncate.TruncationRegion([0, 3], [-1.5, 0.25], [2.5, 4.0], level=0.999, n_latent=P,
                                  V=Q, probe=probe, x_obs_digest="0123456789abcdef")
    d = r.to_dict()
    assert d["V_digest"] == truncate.rotation_digest(Q) and len(d["V_digest"]) == 16
    back = truncate.TruncationRegion.from_dict(d)
    assert torch.equal(back.V, Q) and torch.equal(back.probe, probe)
    assert back.dims == r.dims and torch.equal(back.lo, r.lo) and torch.equal(back.hi, r.hi)
    assert back.x_obs_digest == "0123456789abcdef"
    assert "V:" + d["V_digest"][:8] in repr(back)
    assert back.identity_fields() == r.identity_fields()
    assert r.identity_fields()["V_digest"] == d["V_digest"] and r.identity_fields()["lo"] == [-1.5, 0.25]
    # the region is a measurement, not an alias of the transform it was measured from
    assert r.V.data_ptr() != T_train.parts[0].M.data_ptr()

    legacy = truncate.TruncationRegion.from_dict({"dims": [0], "lo": [-1.0], "hi": [1.0]})
    assert legacy.V is None and legacy.probe is None and "unrotated" in repr(legacy)
    assert legacy.x_obs_digest is None and legacy.identity_fields()["V_digest"] is None
    assert truncate.rotation_digest(None) is None

    # region_from_posterior forwards both onto the region it measures
    torch.manual_seed(4)
    region = truncate.region_from_posterior(_Wide(), torch.zeros(1, 4), n_directions=5,
                                            n_samples=2000, V=Q, probe=probe)
    assert torch.equal(region.V, Q) and torch.equal(region.probe, probe) and region.n_latent == P

    for bad in ({"V": torch.eye(12)}, {"V": torch.ones(13)}, {"probe": torch.zeros(7, 12)}):
        try:
            truncate.TruncationRegion([0], [-1.0], [1.0], n_latent=P, **bad)
            raise AssertionError(f"a mis-sized basis was accepted: {bad}")
        except ValueError:
            pass
    try:
        truncate.region_from_posterior(_Wide(), torch.zeros(1, 4), n_samples=200, V=torch.eye(12))
        raise AssertionError("a V of the wrong size was accepted by region_from_posterior")
    except ValueError:
        pass


def test_a_region_measured_in_one_basis_is_refused_in_a_sign_flipped_one():
    """⚠ THE REGRESSION TEST FOR THE 2026-09-02 ROUND (defect D1).

    That round drew its box in the parent posterior's V and enforced it under a freshly computed
    V' -- different signs, different column ORDER, different mixing -- so the truncated prior kept
    0.01% of the parent posterior and excluded the ground truth. check_basis must refuse every such
    mismatch: a sign flip on a truncated column, a flip on an UNTRUNCATED one (the box's dims still
    index other directions), a column swap (which a probe alone cannot see -- the probe grid's rows
    have all coordinates equal, so a permutation of V's columns leaves it unchanged), an unrelated
    rotation, and a changed box under the same rotation. A region with no probe is refused outright.
    """
    P = 13
    Q = _orthogonal(P, 5)
    box, T_train = _rotated_box(Q)
    probe = training_checkpoint.bijection_probe(T_train, P)
    torch.manual_seed(6)
    region = truncate.region_from_posterior(_Wide(), torch.zeros(1, 4), n_directions=5,
                                            n_samples=2000, V=Q, probe=probe)
    region.check_basis(T_train, dim=P)                       # the parent's own basis passes

    flip2, flip9 = torch.ones(P), torch.ones(P)
    flip2[2], flip9[9] = -1.0, -1.0
    swapped = Q[:, [1, 0] + list(range(2, P))]
    for tag, V2 in (("a sign flip on truncated column 2", Q @ torch.diag(flip2)),
                    ("a sign flip on UNTRUNCATED column 9", Q @ torch.diag(flip9)),
                    ("a swap of columns 0 and 1", swapped),
                    ("an unrelated rotation", _orthogonal(P, 8))):
        try:
            region.check_basis(reparam.build_rotated_bijection(box, V2), dim=P)
            raise AssertionError(f"{tag} was accepted as the region's basis")
        except ValueError as e:
            assert "DIFFERENT V" in str(e), f"{tag}: {e}"
    # the same V over a different box: the rotation matches, the probe does not
    other_box, _ = _rotated_box(Q, width=20.0)
    try:
        region.check_basis(reparam.build_rotated_bijection(other_box, Q), dim=P)
        raise AssertionError("a changed box under the same rotation was accepted")
    except ValueError as e:
        assert "BOX" in str(e)
    # an unrotated bijection against a rotated region, and the mirror image
    try:
        region.check_basis(box, dim=P)
        raise AssertionError("an unrotated bijection was accepted for a rotated region")
    except ValueError as e:
        assert "measured WITH a Fisher rotation, but the training bijection has none" in str(e), e
    plain = truncate.TruncationRegion([0], [-1.0], [1.0], n_latent=P, V=None,
                                      probe=training_checkpoint.bijection_probe(box, P))
    plain.check_basis(box, dim=P)
    try:
        plain.check_basis(T_train, dim=P)
        raise AssertionError("a rotated bijection was accepted for an unrotated region")
    except ValueError as e:
        assert "measured WITHOUT a Fisher rotation, but the training bijection has one" in str(e), e
    # a probe-less region
    bare = truncate.TruncationRegion([0], [-1.0], [1.0], n_latent=P, V=Q)
    try:
        bare.check_basis(T_train, dim=P)
        raise AssertionError("a region without a probe was accepted")
    except ValueError as e:
        assert "probe" in str(e)


def test_build_truncation_region_records_the_parents_basis():
    """The RECORDING end of guardrail 7: orchestrator.build_truncation_region must read the parent
    posterior's V through reparam.rotation_of (V, not its transpose -- the GUI's mistake), probe the
    parent's whole bijection, and bind the observation digest; a posterior with no transform, or a
    transform with no box, is refused rather than guessed at."""
    from core import orchestrator

    P = 13
    Q = _orthogonal(P, 11)
    box, T_parent = _rotated_box(Q)
    x = torch.linspace(-1.0, 1.0, 7).reshape(1, 7)
    rec = {"digest": orchestrator.observation_digest(x)}
    post = reparam.TransformedPosterior(_Wide(), T_parent)
    torch.manual_seed(12)
    region = orchestrator.build_truncation_region(post, rec, x, n_directions=5, level=0.999)
    assert torch.equal(region.V, Q) and not torch.equal(region.V, Q.T)
    assert torch.equal(region.probe, training_checkpoint.bijection_probe(T_parent, P))
    assert region.x_obs_digest == rec["digest"] and region.dims == [0, 1, 2, 3, 4] and region.n_latent == P
    region.check_basis(T_parent, dim=P)
    # an unrotated parent records V=None and a probe of the box alone
    plain = orchestrator.build_truncation_region(reparam.TransformedPosterior(_Wide(), box), rec, x)
    assert plain.V is None and torch.equal(plain.probe, training_checkpoint.bijection_probe(box, P))
    # refusals: the wrong observation, a bare posterior with no transform, a transform with no box
    try:
        orchestrator.build_truncation_region(post, {"digest": "0" * 16}, x)
        raise AssertionError("a region was drawn around an observation that is not the recorded one")
    except ValueError:
        pass
    for bad, tag in ((_Wide(), "a posterior with no transform"),
                     (reparam.TransformedPosterior(_Wide(), torch.distributions.transforms.ComposeTransform(
                         [reparam.OrthogonalTransform(Q.T)])), "a transform with no box")):
        try:
            orchestrator.build_truncation_region(bad, rec, x)
            raise AssertionError(f"{tag} was accepted")
        except ValueError:
            pass


def test_the_same_posterior_and_observation_redraw_the_same_region():
    """The region is part of the round's checkpoint identity, so a round that died at batch 3000 must
    redraw the SAME box to find its own rows again: the draw is seeded from the observation and the
    settings, under fork_rng so the caller's stream is untouched, and the identity quantises the
    bounds so a last-ULP difference cannot rename the directory. A different observation, level or
    direction count draws a different box."""
    from core import orchestrator

    P = 13
    Q = _orthogonal(P, 13)
    _, T_parent = _rotated_box(Q)
    post = reparam.TransformedPosterior(_Wide(), T_parent)
    x = torch.linspace(-1.0, 1.0, 7).reshape(1, 7)
    rec = {"digest": orchestrator.observation_digest(x)}
    torch.manual_seed(1)
    before = torch.get_rng_state()
    a = orchestrator.build_truncation_region(post, rec, x, n_directions=5, level=0.999)
    assert torch.equal(torch.get_rng_state(), before), "the region draw disturbed the caller's RNG stream"
    torch.manual_seed(999)                                          # a different global state...
    b = orchestrator.build_truncation_region(post, rec, x, n_directions=5, level=0.999)
    assert torch.equal(a.lo, b.lo) and torch.equal(a.hi, b.hi), "the same inputs drew a different box"
    assert a.identity_fields() == b.identity_fields()
    x2 = x + 0.5
    c = orchestrator.build_truncation_region(post, {"digest": orchestrator.observation_digest(x2)}, x2)
    d = orchestrator.build_truncation_region(post, rec, x, n_directions=5, level=0.99)
    e = orchestrator.build_truncation_region(post, rec, x, n_directions=4, level=0.999)
    assert not torch.equal(a.lo, c.lo) and not torch.equal(a.lo, d.lo) and e.dims == [0, 1, 2, 3]
    # quantisation: a bound perturbed far below the fifth significant digit names the same directory.
    # Perturb the QUANTISED values (they sit on grid points, half a quantum from any rounding
    # boundary), so the check cannot straddle a boundary by accident.
    lo_q = torch.tensor(a.identity_fields()["lo"], dtype=torch.float64)
    hi_q = torch.tensor(a.identity_fields()["hi"], dtype=torch.float64)
    grid = truncate.TruncationRegion(a.dims, lo_q, hi_q, level=a.level, n_latent=P, V=Q, probe=a.probe)
    nudged = truncate.TruncationRegion(a.dims, lo_q * (1 + 1e-8), hi_q * (1 + 1e-8), level=a.level,
                                       n_latent=P, V=Q, probe=a.probe)
    assert nudged.identity_fields() == grid.identity_fields() == a.identity_fields()
    assert not torch.equal(nudged.lo, grid.lo), "the exact bounds must stay exact; only the NAME is rounded"
    moved = truncate.TruncationRegion(a.dims, a.lo * 1.01, a.hi, level=a.level, n_latent=P, V=Q, probe=a.probe)
    assert moved.identity_fields() != a.identity_fields()


def test_a_resumed_checkpoint_with_another_rotation_is_refused():
    """A checkpoint's rows are latent targets in the V stored beside them. A truncated round may
    resume onto them only if that V IS the region's; anything else mixes coordinates."""
    P = 13
    Q = _orthogonal(P, 7)
    _, T_train = _rotated_box(Q)
    region = truncate.TruncationRegion([0], [-1.0], [1.0], n_latent=P, V=Q,
                                       probe=training_checkpoint.bijection_probe(T_train, P))
    region.check_checkpoint_V(Q.clone())                     # the same rotation, another copy
    region.check_checkpoint_V(Q.to(torch.float64))           # dtype is not identity
    flip = torch.ones(P)
    flip[0] = -1.0
    for bad in (Q @ torch.diag(flip), _orthogonal(P, 9), None):
        try:
            region.check_checkpoint_V(bad)
            raise AssertionError(f"a checkpoint rotation that is not the region's was accepted: {bad}")
        except ValueError:
            pass
    plain = truncate.TruncationRegion([0], [-1.0], [1.0], n_latent=P)
    plain.check_checkpoint_V(None)
    try:
        plain.check_checkpoint_V(Q)
        raise AssertionError("an unrotated region accepted a rotated checkpoint")
    except ValueError:
        pass


# ── the prior sweep's device and its knobs ───────────────────────────────────────────────────────
def test_the_local_sweep_falls_back_to_the_cpu_without_an_accelerator():
    """The flood-fill now runs on the accelerator (measured 6.32 s per inner-loop iteration on the
    CPU against 0.357 s on CUDA, 17.7x, and it is the dominant cost of a prior build). It must
    DEGRADE on a machine with no CUDA rather than raise halfway through a sweep."""
    from core.SBI.Priors import prior as prior_mod

    assert prior_mod.resolve_sweep_device(torch.device("cpu")).type == "cpu"
    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        got = prior_mod.resolve_sweep_device(torch.device("cuda"))
    finally:
        torch.cuda.is_available = real
    assert got.type == "cpu", f"a cuda device with no CUDA resolved to {got}, not the CPU"
    if real():
        assert prior_mod.resolve_sweep_device(torch.device("cuda")).type == "cuda", \
            "the sweep refused the accelerator that IS present"


def test_the_local_sweep_is_no_longer_a_staticmethod_pinned_to_the_cpu():
    """The regression this guards is the original defect: every _local_map was a @staticmethod, so
    none of them could see self.device, so all four silently simulated on the CPU while the global
    sweep used the accelerator."""
    import ast
    import inspect
    import textwrap
    from core.SBI.Priors import bp_prior, hopf_prior, nadrowski_prior, user_prior

    for mod, cls in ((nadrowski_prior, "NadrowskiPrior"), (bp_prior, "BPPrior"),
                     (hopf_prior, "HopfPrior"), (user_prior, "UserPrior")):
        fn = getattr(getattr(mod, cls), "_local_map")
        params = list(inspect.signature(fn).parameters)
        assert params and params[0] == "self", f"{cls}._local_map is not an instance method"
        # ast.unparse, not the raw source: the comment that DOCUMENTS this fix necessarily contains
        # the string it forbids, so a naive text search flags the explanation instead of the code.
        # Same false positive the TSNPE runner check hit -- see test_gui_progress.
        code = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(fn))))
        assert "torch.device('cpu')" not in code and 'torch.device("cpu")' not in code, \
            f"{cls}._local_map still hardcodes the CPU"
        assert "self.sweep_device" in code, f"{cls}._local_map does not use self.sweep_device"
        # and the accept loop must not sync per row -- that would hand back most of the device move
        assert "for i in range(batch_size)" not in code, \
            f"{cls}._local_map still walks rows one at a time (a device-to-host sync per row)"

if __name__ == "__main__":
    failures = 0
    for test_name, fn in sorted(globals().items()):
        if test_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {test_name}")
            # Exception, NOT AssertionError. A test that raises anything else -- a ValueError
            # from a stale str.index, a CUDA error from a hostile card -- used to abort the
            # ENTIRE run at that point, silently losing every test after it. That cost 26
            # tests twice on 2026-08-28. A crash is a failure of THAT test, not of the suite.
            except Exception as e:
                failures += 1
                print(f"FAIL  {test_name}\n      {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
