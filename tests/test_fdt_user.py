"""FDT unit tests, Qt-free except for that one widget read: the user-model normalization gate
(FEATURE 1 v3 + B-d) and, since piece 5, the records the two measurements write.

Locks down the generalized effective-temperature normalization and its gate:
  * campaigns.observable_noise_prefactor computes the per-model coupling/D_x -- n*beta for NADROWSKI,
    2/sigma_x^2 for HOPF, 2*tau_hb/eta_hb^2 for BP (experimental), 1/D_0 for an additive-noise user
    model -- and raises FDTModelError for a user model whose observable noise is multiplicative, zero,
    or negative.
  * registry.fdt_support admits the built-ins and additive-noise user models; it rejects user models
    with multiplicative / zero observable noise or intrinsic forcing.
  * campaigns._n_force_channels returns one channel per state variable for a user model.

The normalisation and gate tests need no cell file and no QApplication: a tiny fake cfg supplies only
.model / .params_dict (and, for the force-channel test, .inits_tensor / .force_params_dict), which is
all those functions read. The FDT config-builder tests added by piece 5 DO read the real Nadrowski
cell, because what they pin is the builder refusing a knob before it parses one. One of them also
builds two numeric widgets (offscreen) to read the value a blank box really produces. The record
tests build a real HOPF config and write into the ``store`` fixture's temp store, with both campaigns
replaced by arithmetic; the run tests that are about the pipeline's messages hand run_fdt a
``_LogWriter`` instead, which names each figure after its title.

Run:  pytest tests/test_fdt_user.py
"""
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                       # noqa: E402

from core import registry                                         # noqa: E402
from core.FDT.campaigns import (observable_noise_prefactor, _n_force_channels,   # noqa: E402
                                FDTModelError)
from core.Models.user_model import parse_user_model               # noqa: E402


class _FakeCfg:
    """The subset of FDTConfig the FDT model helpers read."""
    def __init__(self, model, params=None, n_vars=None, force_params=None):
        self.model = model
        self.params_dict = {k: (v, None) for k, v in (params or {}).items()}   # name -> (value, bounds)
        if n_vars is not None:
            self.inits_tensor = torch.zeros(1, n_vars)
        self.force_params_dict = force_params or {}


class _LogWriter:
    """Enough writer for a run whose RECORDS are the subject, not its manifest: a real directory, a
    body, payload and figure paths inside it, and a store that hands back what it was given. A real
    ArtifactWriter would drag a manifest (and its validation) into tests about the pipeline's logs."""
    def __init__(self, d):
        self.id, self.dir, self.body, self.config = "x", Path(d), {}, {}
        self.store = SimpleNamespace(load_fdt=lambda ref: ref)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def refresh(self):
        pass

    def payload(self, filename):
        return self.dir / filename

    def figure_path(self, title):
        return self.dir / f"{title}.png"


def _register_user(name, variables):
    compiled = parse_user_model(variables)
    registry.register(registry.ModelSpec(
        name=name, labels=list(compiled.param_names), state_dep_drift=compiled.state_dep_noise,
        is_user_model=True, n_vars=len(compiled.var_names), variables=variables, compiled=compiled))
    return compiled


def test_observable_noise_prefactor_builtins():
    """The built-in prefactors reduce to the coded per-model formulas (Nadrowski = the historical value)."""
    assert observable_noise_prefactor(_FakeCfg("NADROWSKI", {"n": 50.0, "beta": 14.1})) == 50.0 * 14.1
    assert observable_noise_prefactor(_FakeCfg("HOPF", {"sigma_x": 0.1})) == 2.0 / 0.1 ** 2
    # BP (experimental): coupling 1/tau_hb, so prefactor = 2*tau_hb/eta_hb^2
    assert observable_noise_prefactor(_FakeCfg("BP", {"tau_hb": 1.0, "eta_hb": 0.5})) == 2.0 * 1.0 / 0.5 ** 2
    # a missing FDT parameter is a readable FDTModelError, not a bare KeyError
    try:
        observable_noise_prefactor(_FakeCfg("HOPF", {}))
    except FDTModelError as e:
        assert "sigma_x" in str(e), e
    else:
        raise AssertionError("missing HOPF param not reported")


def test_observable_noise_prefactor_user_additive():
    """An additive-noise user model's prefactor is 1/D_0 evaluated at the cell param values."""
    name = "FDTUADD"
    try:
        _register_user(name, [{"name": "x", "drift": "-k*x", "D": "Dx"}])
        pref = observable_noise_prefactor(_FakeCfg(name, {"k": 1.0, "Dx": 0.5}))
        assert pref == 1.0 / 0.5
    finally:
        registry.unregister(name)


def test_observable_noise_prefactor_user_rejects_multiplicative_zero_negative():
    name = "FDTUBAD"
    # multiplicative observable noise -> refused
    try:
        _register_user(name, [{"name": "x", "drift": "-k*x", "D": "0.5*x^2"}])
        try:
            observable_noise_prefactor(_FakeCfg(name, {"k": 1.0}))
        except FDTModelError as e:
            assert "multiplicative" in str(e), e
        else:
            raise AssertionError("multiplicative observable noise not refused")
    finally:
        registry.unregister(name)
    # zero and negative constant D_0 -> refused at runtime (D0 <= 0)
    try:
        _register_user(name, [{"name": "x", "drift": "-x", "D": "d0"}])
        for d0 in (0.0, -1.0):
            try:
                observable_noise_prefactor(_FakeCfg(name, {"d0": d0}))
            except FDTModelError as e:
                assert "non-positive" in str(e) or "zero" in str(e), e
                # A CELL value is what makes D0 <= 0 here: a model whose D is identically zero is
                # refused upstream (registry.fdt_support, as "model"), so the fix this refusal names
                # is the cell's parameter, not the model picker.
                assert e.field == "cell", f"D0={d0}: field={e.field!r}"
            else:
                raise AssertionError(f"D0={d0} not refused")
    finally:
        registry.unregister(name)


def test_the_prefactor_is_refused_before_anything_is_simulated(tmp_path, monkeypatch):
    """Spec §3.4, first bullet. The per-model normalisation prefactor was resolved at step 8 of
    run_fdt -- AFTER both campaigns -- so a cell missing `n` or `beta` cost the entire run, hours of
    it, before the pipeline said the one thing it could have said in a second. It is resolved first
    now, and carried down to step 8.

    The assertion that carries the point is that NO CAMPAIGN RAN. A refusal merely moved a few lines
    up in the source but still sitting behind a campaign would pass a message-only test and buy the
    operator nothing. Both entry shapes are checked: the sanity branch spends a campaign of its own
    before the production one, so a check placed after the sanity gate would still be too late.

    The field key is the second half. An FDTModelError is a Refusal (tests/test_refusals.py's
    "every domain error is a refusal and carries a field"), but until now every FDT one carried
    field=None, so neither front-end table could name the control or the flag that answers it.

    The third is the order against the thin-setting notice (ruling F10): the stub is thin on both
    knobs, so the notice WOULD fire if it ran first, and a refused run must not first warn the
    operator about how far to trust a result it will never produce. No PreflightWarning may precede
    the refusal.
    """
    import warnings

    import pytest

    from core import config
    from core.FDT import fdt_pipeline
    from core.FDT.campaigns import FDTModelError
    from core.refusals import PreflightWarning

    spent = []

    def _never(*a, **kw):
        spent.append(a)
        raise RuntimeError("a campaign ran: the prefactor must be refused before anything is spent")

    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _never)
    monkeypatch.setattr(fdt_pipeline, "run_all_sanity", _never)

    class _NoN:
        """A Nadrowski cell carrying k -- so the omega_0 estimate would have succeeded -- but no n,
        which is half of the Nadrowski prefactor n*beta."""
        model = "NADROWSKI"
        params_dict = {"k": (1.0, None), "beta": (14.1, None)}
        # below both thin thresholds on purpose: the notice would fire if it ran before the prefactor
        n_freqs, ensemble_M = 1, 2
        freqs_per_batch, F0 = 1, 0.05
        freq_bounds, burn_in_nd, T_obs_periods = (0.1, 30.0), 100.0, 30
        dt_nd, psd_T_obs_nd, seed = 0.01, 8000.0, None
        hw = config.cpu_device()

    for skip_sanity in (True, False):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(FDTModelError) as e:
                fdt_pipeline.run_fdt(_NoN(), skip_sanity=skip_sanity, confirm_production=True,
                                     writer=_LogWriter(tmp_path), seed=1)
        assert "'n'" in str(e.value), str(e.value)
        assert e.value.field == "cell", f"skip_sanity={skip_sanity}: field={e.value.field!r}"
        thin = [str(w.message) for w in rec if issubclass(w.category, PreflightWarning)]
        assert thin == [], f"skip_sanity={skip_sanity}: the refused run first warned {thin}"
    assert spent == [], "the prefactor refusal arrived only after a campaign had been spent"


def test_a_probe_below_the_first_real_bin_comes_back_blank_not_blended():
    """Spec §3.5, E9 -- and the demonstration of the defect, which is the whole point of the fix.

    _interp_log's docstring promises NaN outside the grid, and PRISM_HANDOFF names the consequence of
    not having it: "widen freq_bounds past the PSD resolution and T_eff/T acquires a smooth,
    plausible-looking, entirely fabricated tail". A Welch grid starts at exactly 0.0, the helper
    clamps that bin to 1e-30 before taking logarithms, and the in-range test compares against the
    CLAMPED bin -- so every positive frequency passed it and the low end returned a value blended off
    the zero-frequency bin. Here the zero bin holds 100 and the first real bin holds 1: a probe at
    half the first real bin used to come back at about 1.993, a number with no measurement behind it.
    It is a blank now, and the assertion message prints what came back instead so the defect is
    legible in the failure rather than only in this docstring."""
    import math

    from core.FDT.sanity import _interp_log

    x_old = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)   # a Welch grid: the DC bin is 0.0
    y_old = torch.tensor([100.0, 1.0, 2.0], dtype=torch.float64)
    got = float(_interp_log(torch.tensor([0.5], dtype=torch.float64), x_old, y_old)[0])
    assert math.isnan(got), (f"a probe at 0.5, below the spectrum's first real bin at 1.0, came back "
                             f"blended off the zero bin instead of blank: {got:.4f}")


def test_every_probe_outside_the_resolved_span_is_blank_and_the_covered_ones_are_exact():
    """Spec §3.5. The count of excluded points is what reaches the record as body.offgrid (spec
    §2.3) and what the sweep's per-point summary reports, so it has to be honest at BOTH ends: the
    upper end already blanked, the lower end never did. The covered points are asserted too, because
    a range test that blanked too much would also make the count "honest" and would silently throw
    away measured frequencies -- 2**0.5 sits at exactly half a log-decade between the 1.0 and 2.0
    bins, so its interpolated value is exactly halfway between their values."""
    from core.FDT.sanity import _interp_log

    x_old = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    y_old = torch.tensor([100.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    probes = torch.tensor([0.5, 1.0, 2.0 ** 0.5, 2.0, 4.0, 8.0], dtype=torch.float64)

    got = _interp_log(probes, x_old, y_old)
    blanks = torch.isnan(got)
    assert int(blanks.sum()) == 2, (f"expected 2 blanks -- 0.5 below the first real bin and 8.0 above "
                                    f"the last -- got {int(blanks.sum())}: {got.tolist()}")
    assert bool(blanks[0]) and bool(blanks[-1]), got.tolist()
    assert torch.allclose(got[1:5], torch.tensor([1.0, 1.5, 2.0, 4.0], dtype=torch.float64),
                          atol=1e-12), got.tolist()

    from core.FDT.sanity import _resolved_span
    assert _resolved_span(x_old) == (1.0, 4.0), "the zero bin is not the band's lower end"
    assert _resolved_span(torch.tensor([0.0], dtype=torch.float64)) == (float("inf"), float("-inf")), \
        "a grid with no positive bin resolves nothing: every probe is then out of range (P7)"


def test_the_sanity_checks_name_the_band_the_spectrum_actually_resolves(monkeypatch):
    """Spec §3.5: "the callers' count of excluded points becomes honest". The count is only half of
    it. Both sanity checks print the excluded band as freqs_psd.min()..freqs_psd.max(), and a Welch
    grid's min is exactly 0.0 -- so the sentence read "fall outside the Welch PSD grid (0..2)" while
    excluding a probe at 0.5, which is inside 0..2. The band the reader is asked to narrow towards
    has to be the band the spectrum resolves.

    check_high_freq_fdt is the one to pin: it reads the TOP three frequencies of the probe grid,
    precisely where the PSD runs out, so it was the check most exposed to the old extrapolation. Its
    campaign seams are stubbed, so nothing is simulated."""
    import pytest

    from core.FDT import sanity

    class _Cfg:
        ensemble_M, psd_T_obs_nd, omega_0, freq_bounds = 8, 100.0, 1.0, (0.1, 30.0)

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

        def with_overrides(self, **kw):
            return self

    # A spectrum resolving 0.5..2.0; the top three of the 7-point probe grid are 4.48, 11.59 and 30.
    freqs_psd = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 2.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(sanity, "run_campaign1_psd", lambda c: (freqs_psd, G))
    monkeypatch.setattr(sanity, "run_campaign2_chi",
                        lambda c, om, **kw: torch.full((len(om),), 1 + 1j, dtype=torch.complex128))
    monkeypatch.setattr(sanity, "observable_noise_prefactor", lambda c: 1.0)

    with pytest.warns(UserWarning, match="3/3") as rec:
        passed, metrics = sanity.check_high_freq_fdt(_Cfg())

    msg = str(rec[0].message)
    assert "(0.5..2)" in msg, msg
    assert "(0..2)" not in msg, f"the warning named the zero bin as the band's lower end: {msg}"
    assert passed is False and metrics["n_off_grid"] == 3, metrics


def test_a_band_below_the_spectrums_resolution_refuses_before_the_driven_campaign(tmp_path, monkeypatch):
    """Spec §3.4, second bullet, and E9. The probe grid is built around a resonance the spontaneous
    campaign found, so the grid's lowest frequency and the lowest frequency the spectrum resolves are
    only comparable once Campaign 1 is done. That is the first moment the condition is knowable, and
    it sits directly before the expensive half of the run -- which is the only reason the check is
    worth anything. Before T14 this band silently produced a fabricated low-frequency tail; after T14
    it produces blanks; refusing says so before the drive is spent.

    The assertion that carries the point is that Campaign 2 NEVER RAN. The message is checked too:
    it names the band that was asked for, the band that exists, and what sets it -- and it names no
    box, tab or flag. Its field is "freq_bounds" (P75): the key is registered, and both front-end
    tables map it to None because no control and no flag exposes the band, so neither table offers a
    fix sentence and neither pretends to.

    What sets the resolution is the spectrum's Welch SEGMENT, not the recording (ruled after Task 14's
    review, which measured it): Campaign 1's segment stops growing at WELCH_NPERSEG_CAP samples,
    163.84 ND at dt_nd = 0.01. This recording is 200 ND -- 20,000 samples, past the cap, as both
    shipped durations are -- so the refusal must NOT send the operator to lengthen it, and says why
    instead. The below-cap branch is the next test."""
    import pytest

    from core.FDT import fdt_pipeline
    from core.FDT.campaigns import WELCH_NPERSEG_CAP
    from core.refusals import Refusal

    driven = []

    class _Cfg:
        """The subset of FDTConfig run_fdt reads, at a size nothing simulates."""
        model = "HOPF"
        params_dict = {"sigma_x": (0.1, None)}
        n_freqs, freq_bounds = 5, (0.1, 30.0)
        ensemble_M = 8        # read first by the thin-setting check; at its threshold, so it says nothing
        burn_in_nd, dt_nd, psd_T_obs_nd, omega_0 = 0.0, 0.01, 200.0, 1.0
        freqs_per_batch, F0, T_obs_periods, seed = 1, 0.05, 30, None     # read into body.settings

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

    # A spectrum resolving 1.0..3.0 whose peak is at 2.0, so the grid runs 0.2 .. 60 -- its lowest
    # probe sits a factor of five below the lowest frequency the spectrum resolves.
    freqs_psd = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 5.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd",
                        lambda cfg, return_trajectory=False: (
                            freqs_psd, G, torch.arange(4, dtype=torch.float64),
                            torch.zeros(4, dtype=torch.float64)))
    monkeypatch.setattr(fdt_pipeline, "plot_spontaneous_trajectory", lambda *a, **kw: None)
    monkeypatch.setattr(fdt_pipeline, "plot_psd",
                        lambda *a, save_path=None, **kw: save_path.write_bytes(b"png"))

    def _no_drive(*a, **kw):
        driven.append(a)
        raise AssertionError("the driven campaign ran: the band refusal must come first")

    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", _no_drive)

    with pytest.raises(Refusal) as e:
        fdt_pipeline.run_fdt(_Cfg(), skip_sanity=True, confirm_production=True,
                             writer=_LogWriter(tmp_path), seed=1)
    msg = str(e.value)
    assert "0.2" in msg and "psd_T_obs_nd = 200" in msg, msg
    assert "Raise freq_bounds' lower multiplier above 0.5" in msg, msg
    # past the segment's cap: a longer recording is not offered, and the message says why
    assert round(_Cfg.psd_T_obs_nd / _Cfg.dt_nd) >= WELCH_NPERSEG_CAP, "the premise of this case"
    assert "lengthen the spontaneous recording" not in msg, msg
    assert "would not help" in msg and "Welch segment" in msg, msg
    assert f"cap of {WELCH_NPERSEG_CAP} samples" in msg, msg
    assert e.value.field == "freq_bounds", e.value.field
    from core.tool.fields import fix_sentence
    assert fix_sentence("freq_bounds") == "", "no flag exposes the band, so no fix is offered (P2, P75)"
    assert driven == [], "Campaign 2 was entered before the band was checked"
    written = [p.name for p in tmp_path.glob("*.png")]
    assert written == ["Spontaneous PSD.png"], ("the spontaneous spectrum's picture must be on disk "
                                                f"when the band refusal fires -- it is what diagnoses "
                                                f"it (P22): {written}")


def test_the_band_refusal_offers_a_longer_recording_only_where_it_lowers_the_resolution(
        tmp_path, monkeypatch):
    """The other branch of the ruling above. Below the Welch segment's cap a longer recording DOES
    lengthen the segment, so it lowers the spectrum's first real bin, and the refusal offers it beside
    the band -- with the cap, so the operator knows how far it goes. 100 ND at dt_nd = 0.01 is 10,000
    samples, cut into 8,192-sample segments.

    The boundary is pinned on the one helper the refusal and check_passive_baseline share: a
    recording of exactly the cap already fills it, one sample shorter does not. And the cap is ONE
    constant, read by name in Campaign 1 as well, so the advice cannot drift from what the campaign
    does."""
    import inspect
    from types import SimpleNamespace

    import pytest

    from core.FDT import campaigns, fdt_pipeline
    from core.FDT.campaigns import WELCH_NPERSEG_CAP
    from core.FDT.sanity import _low_end_advice
    from core.refusals import Refusal

    class _Cfg:
        """As in the test above, with a recording shorter than the segment's cap."""
        model = "HOPF"
        params_dict = {"sigma_x": (0.1, None)}
        n_freqs, freq_bounds, ensemble_M = 5, (0.1, 30.0), 8
        burn_in_nd, dt_nd, psd_T_obs_nd, omega_0 = 0.0, 0.01, 100.0, 1.0
        freqs_per_batch, F0, T_obs_periods, seed = 1, 0.05, 30, None     # read into body.settings

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

    freqs_psd = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 5.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd",
                        lambda cfg, return_trajectory=False: (
                            freqs_psd, G, torch.arange(4, dtype=torch.float64),
                            torch.zeros(4, dtype=torch.float64)))
    # every figure is stubbed, so nothing is drawn wherever it sits relative to the check
    for figure in ("plot_spontaneous_trajectory", "plot_psd"):
        monkeypatch.setattr(fdt_pipeline, figure, lambda *a, **kw: None)

    def _no_drive(*a, **kw):
        raise AssertionError("the driven campaign ran: the band refusal must come first")

    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", _no_drive)

    assert round(_Cfg.psd_T_obs_nd / _Cfg.dt_nd) < WELCH_NPERSEG_CAP, "the premise of this case"
    with pytest.raises(Refusal) as e:
        fdt_pipeline.run_fdt(_Cfg(), skip_sanity=True, confirm_production=True,
                             writer=_LogWriter(tmp_path), seed=1)
    msg = str(e.value)
    assert e.value.field == "freq_bounds", e.value.field
    assert ("Raise freq_bounds' lower multiplier above 0.5, or lengthen the spontaneous recording "
            "(psd_T_obs_nd = 100)") in msg, msg
    assert "would not help" not in msg, msg
    # the cap, and the floor it reaches (2*pi/163.84): a band below that needs the multiplier anyway
    assert f"cap of {WELCH_NPERSEG_CAP} samples (163.84 ND)" in msg and "down to 0.0383495" in msg, msg

    # the boundary, on the shared helper: exactly the cap fills it; one sample less does not
    at_cap = _low_end_advice(SimpleNamespace(psd_T_obs_nd=163.84, dt_nd=0.01), 1.0, "the run")
    short = _low_end_advice(SimpleNamespace(psd_T_obs_nd=163.83, dt_nd=0.01), 1.0, "the run")
    assert "would not help" in at_cap and "or lengthen the run" not in at_cap, at_cap
    assert "or lengthen the run (psd_T_obs_nd = 163.83)" in short and "would not help" not in short, short

    # one cap: Campaign 1 reads the same constant, not a literal of its own
    src = inspect.getsource(campaigns.run_campaign1_psd)
    assert "WELCH_NPERSEG_CAP" in src and "2 ** 14" not in src, "the campaign's cap is not the advice's"


def test_the_sanity_warnings_name_what_really_bounds_each_end_of_the_spectrum(monkeypatch):
    """The same ruling, for the two sanity warnings T14 made print the resolved band. Both gave advice
    that cannot work at the shipped settings. check_high_freq_fdt blanks at the TOP of the grid and
    said "raise psd_T_obs_nd" -- but the top of a Welch spectrum is its Nyquist frequency, pi/dt_nd,
    which no recording length moves. check_passive_baseline said "lengthen the passive run", which
    moves the BOTTOM only while the recording is shorter than the Welch segment's cap; its passive run
    is min(4000, psd_T_obs_nd) ND, past the cap at every shipped setting. Each now names what bounds
    its end, and the passive one offers a longer run only below the cap, in the band refusal's words
    (one helper). Nothing is simulated: the campaign seams are stubbed, and the passive check keeps
    four covered probes, so its bare ValueError (Task 33's) is not reached."""
    import copy

    import pytest

    from core.FDT import sanity

    class _Cfg:
        model, ensemble_M, omega_0, freq_bounds, dt_nd = "NADROWSKI", 8, 1.0, (0.1, 30.0), 0.01

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

        def __init__(self, psd_T_obs_nd):
            self.psd_T_obs_nd = psd_T_obs_nd

        def with_overrides(self, **kw):
            c = copy.copy(self)
            vars(c).update(kw)
            return c

    # A spectrum resolving 0.5..2.0 with its peak at 1.0: the passive check's 20-point grid then runs
    # 0.1..30 and keeps four probes; the high-frequency check's top three (4.48, 11.59, 30) all blank.
    freqs_psd = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 2.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(sanity, "run_campaign1_psd", lambda c: (freqs_psd, G))
    monkeypatch.setattr(sanity, "run_campaign2_chi",
                        lambda c, om, **kw: torch.full((len(om),), 1 + 1j, dtype=torch.complex128))
    monkeypatch.setattr(sanity, "observable_noise_prefactor", lambda c: 1.0)

    # the top end: the step bounds it, and a longer recording is not offered
    with pytest.warns(UserWarning, match="3/3") as rec:
        sanity.check_high_freq_fdt(_Cfg(4000.0))
    msg = str(rec[0].message)
    assert "psd_T_obs_nd" not in msg, f"a longer recording cannot raise the top: {msg}"
    assert "Lower cfg.freq_bounds' upper edge" in msg and "Nyquist" in msg and "pi/dt_nd" in msg, msg

    # the bottom end, past the cap (the passive run is min(4000, 8000) = 4000 ND): not offered
    with pytest.warns(UserWarning, match="16/20") as rec:
        sanity.check_passive_baseline(_Cfg(8000.0))
    msg = str(rec[0].message)
    assert "or lengthen the passive run" not in msg, msg
    assert ("Lengthening the passive run (psd_T_obs_nd = 4000) would not help" in msg
            and "Welch segment" in msg), msg

    # ...and below the cap (100 ND is 10,000 samples): offered, with the cap it stops at
    with pytest.warns(UserWarning, match="16/20") as rec:
        sanity.check_passive_baseline(_Cfg(100.0))
    msg = str(rec[0].message)
    assert "Narrow cfg.freq_bounds, or lengthen the passive run (psd_T_obs_nd = 100)" in msg, msg
    assert "would not help" not in msg and "cap of 16384 samples" in msg, msg


def test_a_run_that_can_measure_nothing_refuses_and_leaves_the_spectrum_behind(tmp_path, monkeypatch):
    """Spec §3.6, E4. A run whose every probe frequency is blank has measured nothing: today it
    divides blanks by blanks, saves a figure with no points on it, and reports success -- an answer
    indistinguishable from a real one until someone opens the picture. It refuses instead, naming the
    band and the two settings that set it. Its field is "freq_bounds", as the band refusal's is (F37):
    the same subject, and neither front end exposes it, so neither offers a fix sentence.

    E4's other half is asserted here too: the run leaves behind what diagnoses the failure. The
    spontaneous spectrum's picture is written BEFORE the driven campaign, so it is on disk when this
    refusal fires, while the ratio and susceptibility figures -- which would have been empty -- are
    not. That directory is the record's folder, which the writer keeps unfinished (E2; the store-backed
    case is test_a_band_refused_after_the_spectrum_keeps_its_record_and_a_refused_cell_leaves_none);
    the ordering is what makes the promise keepable.

    Nothing about it needs the drive: whether a probe is blank depends only on the grid and Campaign
    1's spectrum, so it is refused BEFORE Campaign 2 is spent (the review of Task 16), and Campaign 2
    here is a recorder that must never be entered.

    The low end is gated by T15, so the reachable case is the UPPER end (spec §3.5): here the probe
    grid runs 1.0..6.0 against a spectrum that resolves 0.1..0.3. The four plot helpers are replaced
    by recorders that write their save_path, so the test asserts WHICH figures a run leaves without
    paying for matplotlib (the real figures are the next tests')."""
    import pytest

    from core.FDT import fdt_pipeline
    from core.refusals import Refusal

    class _Cfg:
        model = "HOPF"
        params_dict = {"sigma_x": (0.1, None)}
        n_freqs, freq_bounds = 5, (5.0, 30.0)
        ensemble_M = 8        # read first by the thin-setting check; at its threshold, so it says nothing
        burn_in_nd, dt_nd, psd_T_obs_nd, omega_0 = 0.0, 0.01, 50.0, 1.0
        freqs_per_batch, F0, T_obs_periods, seed = 1, 0.05, 30, None     # read into body.settings

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

    # Resolves 0.1..0.3, peak at 0.2 -> the grid runs 1.0 .. 6.0: its LOWEST probe is above the
    # spectrum's highest bin, so T15's low-end gate passes and every probe is blank anyway.
    freqs_psd = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 5.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd",
                        lambda cfg, return_trajectory=False: (
                            freqs_psd, G, torch.arange(4, dtype=torch.float64),
                            torch.zeros(4, dtype=torch.float64)))
    driven = []

    def _no_drive(*a, **kw):
        driven.append(a)
        raise AssertionError("the driven campaign ran: nothing-measurable must be refused first")

    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", _no_drive)
    for name in ("plot_spontaneous_trajectory", "plot_psd", "plot_eff_temp_ratio",
                 "plot_chi_components"):
        monkeypatch.setattr(fdt_pipeline, name,
                            lambda *a, save_path=None, **kw: save_path.write_bytes(b"png"))

    with pytest.raises(Refusal) as e:
        fdt_pipeline.run_fdt(_Cfg(), skip_sanity=True, confirm_production=True,
                             writer=_LogWriter(tmp_path), seed=1)
    msg = str(e.value)
    assert "0.1..0.3" in msg and "psd_T_obs_nd = 50" in msg, msg
    assert e.value.field == "freq_bounds", e.value.field
    assert driven == [], "Campaign 2 was entered before nothing-measurable was refused"

    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == ["Spontaneous PSD.png", "Spontaneous trajectory.png"], written


class _ToneCfg:
    """The subset of FDTConfig run_fdt and the REAL Campaign 1 read, at a size that takes no time:
    2048 ND at dt_nd = 0.5 is 4096 samples, one Welch segment, well inside the cap. Paired with
    _use_tone_simulator, which replaces the simulator and nothing else."""
    model = "HOPF"
    params_dict = {"sigma_x": (0.1, None)}
    force_params_dict, state_dep_drift = {}, False
    n_freqs, ensemble_M, burn_in_nd, omega_0 = 5, 8, 0.0, 1.0
    freqs_per_batch, F0, T_obs_periods, seed = 1, 0.05, 30, None     # read into body.settings

    class hw:
        device = torch.device("cpu")
        dtype = torch.float64

    def __init__(self, freq_bounds, dt_nd=0.5, psd_T_obs_nd=2048.0):
        self.freq_bounds, self.dt_nd, self.psd_T_obs_nd = freq_bounds, dt_nd, psd_T_obs_nd

    def inits_for_M(self, M):
        return None

    def params_for_M(self, M):
        return None


def _use_tone_simulator(monkeypatch, *, diverge=False):
    """Replace Campaign 1's simulator -- only it -- by a noiseless sin(t) in every trajectory, so the
    Welch segment, the grid, the peak (near 1 ND) and the top are what the campaign's own code
    computes. ``diverge=True`` sends one trajectory to +inf halfway through: a spontaneous simulation
    that diverged."""
    from core.FDT import campaigns

    class _Tone:
        def __init__(self, t, M):
            self.t, self.M = t, M

        def simulate(self, state_dep_drift=False):
            x = torch.sin(self.t).expand(self.M, -1).clone()
            if diverge:
                x[self.M // 2, self.t.numel() // 2:] = float("inf")
            return x.reshape(1, 1, self.M, -1)

    monkeypatch.setattr(campaigns, "_make_simulator",
                        lambda cfg, params, force, inits, t, **kw: _Tone(t, kw["batch_size"]))


def _figures_in(directory) -> list:
    """The figures a run left in ``directory``, by file name: _LogWriter names each after its title."""
    return sorted(p.name for p in directory.glob("*.png"))


def test_the_nothing_measurable_refusal_says_only_what_is_true_of_the_spectrum(
        tmp_path, monkeypatch, caplog):
    """The ruling carried from Tasks 14 and 15: every sentence a refusal says about what the spectrum
    resolves must be TRUE of the spectrum the run measured -- advice that sends the operator to double
    a spontaneous campaign for an identical band is worse than none. The test above pins the refusal
    on a four-bin stub, where no sentence about a Nyquist frequency can be true; this one runs the
    REAL Campaign 1 -- only the simulator is replaced, by a noiseless tone at 1 ND, so the peak is
    known -- and holds each claim the refusal makes to the code that decides it:

    * the top it names is the spectrum's Nyquist frequency pi/dt_nd, which a step half as long
      doubles and a recording four times as long leaves where it is (it lowers only the bottom) --
      so the refusal says a longer recording would not help, and offers none;
    * the band it prints works when copied: a lower multiplier inside the range it names measures
      some probes -- the run completes and warns ONCE how many are blank -- and an upper multiplier
      below the bound it names measures every probe, with no warning at all.

    The grid lies wholly above the top here (lower multiplier 8, top 2*pi at dt_nd = 0.5), which is
    the one way every probe can blank once the band check has gated the low end: the resonance is a
    bin of the spectrum, so a lower multiplier of 1 or less keeps the lowest probe resolved. Lowering
    the UPPER multiplier alone therefore cannot help, and the refusal does not say it would. Campaign
    2 is a recorder: the refusal leaves it untouched, and only the two copied runs drive it."""
    import logging
    import math
    import re

    import pytest

    from core.FDT import campaigns, fdt_pipeline
    from core.FDT.sanity import _resolved_span
    from core.FDT.spectral import find_spectral_peak
    from core.refusals import Refusal

    _use_tone_simulator(monkeypatch)
    driven = []

    def _drive(cfg, om, **kw):
        driven.append(len(om))
        return torch.full((len(om),), 1 + 1j, dtype=torch.complex128)

    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", _drive)
    for name in ("plot_spontaneous_trajectory", "plot_psd", "plot_eff_temp_ratio",
                 "plot_chi_components"):
        monkeypatch.setattr(fdt_pipeline, name, lambda *a, **kw: None)

    def span(**kw):
        return _resolved_span(campaigns.run_campaign1_psd(_ToneCfg((8.0, 30.0), **kw))[0])

    def warned():
        return [r.getMessage() for r in caplog.records
                if r.name == "core.FDT.fdt_pipeline" and r.levelno == logging.WARNING]

    freqs, G = campaigns.run_campaign1_psd(_ToneCfg((8.0, 30.0)))
    lo, hi = _resolved_span(freqs)
    w0 = find_spectral_peak(freqs, G)
    assert hi == pytest.approx(math.pi / 0.5, rel=1e-12), "the spectrum's top is not pi/dt_nd"
    long_lo, long_hi = span(psd_T_obs_nd=4 * 2048.0)
    assert long_hi == hi and long_lo < lo, "a longer recording must lower the bottom and leave the top"
    assert span(dt_nd=0.25)[1] == pytest.approx(2 * hi, rel=1e-12), "a shorter step must raise the top"

    with pytest.raises(Refusal) as e:
        fdt_pipeline.run_fdt(_ToneCfg((8.0, 30.0)), skip_sanity=True, confirm_production=True,
                             writer=_LogWriter(tmp_path), seed=1)
    msg = str(e.value)
    assert e.value.field == "freq_bounds", e.value.field
    assert driven == [], "the driven campaign was spent before nothing-measurable was refused"
    assert f"the band it resolves is {lo:g}..{hi:g} (ND)" in msg, msg
    assert f"The lowest probe frequency, {8.0 * w0:g} (ND), is above that band's top" in msg, msg
    assert "Nyquist frequency pi/dt_nd (dt_nd = 0.5): only a shorter integration step raises it" in msg, msg
    assert "Lengthening the spontaneous recording (psd_T_obs_nd = 2048) would not help." in msg, msg
    assert "or lengthen" not in msg, msg

    got = re.search(r"lower multiplier into ([0-9.e+-]+?)\.\.([0-9.e+-]+) for any probe to be measured, "
                    r"and its upper multiplier below ([0-9.e+-]+) for all of them", msg)
    assert got, msg
    low, high, upper = (float(v) for v in got.groups())
    assert low == pytest.approx(lo / w0, rel=1e-5) and high == upper == pytest.approx(hi / w0, rel=1e-5)

    # copied: a lower multiplier inside the named range measures some probes, and says how many not
    caplog.clear()
    fdt_pipeline.run_fdt(_ToneCfg((0.9 * high, 30.0)), skip_sanity=True, confirm_production=True,
                         writer=_LogWriter(tmp_path), seed=1)
    assert warned() == [f"4/5 probe frequencies have no value in the spontaneous spectrum, whose "
                        f"resolved band is {lo:g}..{hi:g} (ND), and are blank in the ratio."], warned()
    # ...and an upper multiplier below the named bound measures all of them
    caplog.clear()
    fdt_pipeline.run_fdt(_ToneCfg((high / 4, 0.99 * upper)), skip_sanity=True, confirm_production=True,
                         writer=_LogWriter(tmp_path), seed=1)
    assert warned() == [], warned()
    assert driven == [5, 5], driven


def test_the_spectrum_figure_draws_what_the_spectrum_holds_whatever_the_band(
        tmp_path, monkeypatch, caplog):
    """The review of Task 16 (Critical), at the figure. The spontaneous-spectrum figure is drawn BEFORE
    the two band refusals, as the picture that diagnoses them -- and plot_psd clipped the spectrum to
    the probe band and log-scaled both axes, so a band holding no spectrum point (wholly above the
    Nyquist top or wholly below the first bin: exactly the cases those refusals exist for) raised "Data
    has no positive values" from tight_layout, and so did a spectrum with no finite value. Every run
    test stubbed plot_psd, which is why none saw it. The REAL function is driven here; the figure it
    saves is caught at savefig and read back:

    * a band that holds points: the ordinary figure, unchanged -- log/log, the in-band points only,
      the resonance line, nothing added;
    * a band above the top, and one below the first bin: the WHOLE finite spectrum, with the band
      shaded on the same axis -- a figure clipped to an empty band would show nothing, and the
      distance between the band and the spectrum is what diagnoses the refusal;
    * no finite value at all: an annotated figure on linear axes, beside the existing warning record."""
    import logging

    import numpy as np

    from core.FDT import plots

    figures = []
    real_savefig = plots.plt.savefig

    def _keep(*a, **kw):
        figures.append(plots.plt.gcf())
        return real_savefig(*a, **kw)

    monkeypatch.setattr(plots.plt, "savefig", _keep)

    omegas = np.linspace(0.0, 2 * np.pi, 9)             # a Welch-shaped grid: the DC bin, then 8 bins
    G = 1.0 / (1.0 + (omegas - np.pi) ** 2)             # positive everywhere, peaked at pi
    w0 = np.pi

    def draw(name, spectrum=G, band=None):
        plots.plot_psd(omegas, spectrum, save_path=tmp_path / f"{name}.png", omega_natural=w0,
                       plot_band=band)
        assert (tmp_path / f"{name}.png").is_file(), name
        return figures[-1].axes[0]

    def legend_of(ax):
        return [t.get_text() for t in ax.get_legend().get_texts()]

    caplog.clear()
    ax = draw("ordinary", band=(1.0, 5.0))
    inside = (omegas >= 1.0) & (omegas <= 5.0)
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")
    data, resonance = ax.get_lines()
    assert np.array_equal(data.get_xdata(), omegas[inside] / w0), data.get_xdata()
    assert np.array_equal(data.get_ydata(), G[inside]), data.get_ydata()
    assert list(resonance.get_xdata()) == [1.0, 1.0]
    assert len(ax.patches) == 0 and len(ax.texts) == 0, "the ordinary figure gained an artist"
    assert len(legend_of(ax)) == 1 and "Omega_0" in legend_of(ax)[0], legend_of(ax)

    for name, band in (("above_the_top", (8.0, 30.0)), ("below_the_first_bin", (1e-4, 1e-3))):
        ax = draw(name, band=band)
        assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log"), name
        data = ax.get_lines()[0]
        assert np.array_equal(data.get_xdata(), omegas[1:] / w0), (name, data.get_xdata())
        (shade,) = ax.patches
        x_lo, x_hi = ax.get_xlim()
        assert x_lo <= band[0] / w0 and band[1] / w0 <= x_hi, (name, ax.get_xlim(), "the band is off the axis")
        assert any("probe band" in t for t in legend_of(ax)), (name, legend_of(ax))
    assert [r for r in caplog.records if r.name == "core.FDT.plots"] == [], "a figure with points warned"

    ax = draw("nothing_finite", spectrum=np.full_like(G, np.nan), band=(1.0, 5.0))
    assert (ax.get_xscale(), ax.get_yscale()) == ("linear", "linear")
    assert len(ax.get_lines()) == 0 and len(ax.texts) == 1, list(ax.texts)
    assert "no finite" in ax.texts[0].get_text(), ax.texts[0].get_text()
    assert [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "core.FDT.plots"] == \
        [(logging.WARNING, "plot_psd: dropping 8/8 non-finite PSD points.")]


def test_both_band_refusals_leave_the_real_spectrum_figure_and_never_drive(tmp_path, monkeypatch):
    """The same finding end to end. With the real plot_psd the run died of a plotting ValueError
    before either band refusal was reached: a band wholly above the spectrum's top (nothing
    measurable), and a band wholly below its first bin -- which Task 15 had refused cleanly before
    the figure moved above its check, so that one was a regression.

    Nothing is stubbed here but the simulator and Campaign 2: the real Campaign 1 (a noiseless tone),
    the real plot_spontaneous_trajectory and plot_psd. Each band ends in its own refusal, keyed
    "freq_bounds", with both pictures on disk, and the driven campaign is never entered -- both
    refusals are knowable from the grid and Campaign 1 alone (the review's order finding)."""
    import pytest

    from core.FDT import fdt_pipeline
    from core.refusals import Refusal

    _use_tone_simulator(monkeypatch)
    driven = []

    def _no_drive(*a, **kw):
        driven.append(a)
        raise AssertionError("the driven campaign ran: a band refusal must come first")

    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", _no_drive)

    cases = (("above_the_top", (8.0, 30.0), "has no value at any of the 5 probe frequencies"),
             ("below_the_first_bin", (1e-4, 1e-3),
              "The frequency band reaches below what the spontaneous spectrum resolves"))
    for name, band, says in cases:
        out = tmp_path / name
        out.mkdir()
        with pytest.raises(Refusal) as e:
            fdt_pipeline.run_fdt(_ToneCfg(band), skip_sanity=True, confirm_production=True,
                                 writer=_LogWriter(out), seed=1)
        assert says in str(e.value), (name, str(e.value))
        assert e.value.field == "freq_bounds", (name, e.value.field)
        assert _figures_in(out) == ["Spontaneous PSD.png", "Spontaneous trajectory.png"], \
            (name, _figures_in(out))
    assert driven == [], "the driven campaign was entered before a band refusal"


def test_a_diverged_spontaneous_simulation_is_refused_as_the_cells_before_the_peak_search(
        tmp_path, monkeypatch, caplog):
    """The review of Task 16 (Important 2). A spontaneous simulation that diverged leaves a spectrum
    with no finite value, and find_spectral_peak then returns the FIRST bin as the resonance (argmax
    takes NaN for the largest value) -- so the band check told the operator to raise freq_bounds'
    lower multiplier, and, followed, the nothing-measurable refusal named a resolved band: both false,
    and the real diagnosis never said. It is refused now as what it is, right after Campaign 1 and
    BEFORE the peak search, keyed "cell": no band and no recording length makes a diverged
    simulation measurable.

    The folder keeps the picture that shows the divergence -- the spontaneous trajectory, drawn by
    the REAL plot_spontaneous_trajectory, which draws NaN and inf without raising -- and nothing the
    refusal precedes. Two ways in: the real Campaign 1 with one of eight trajectories sent to +inf,
    which the real psd_welch turns into NaN at EVERY bin (one non-finite sample in a Welch segment
    reaches every frequency), and a Campaign 1 stub whose spectrum and trajectory are all NaN.

    The boundary: a spectrum non-finite in only SOME bins is not this refusal. It takes the
    some-blank path -- the run completes and says how many probes are blank, in words true of blanks
    INSIDE the resolved band too -- or, when every probe is blank inside that band, the
    nothing-measurable refusal without the Nyquist sentence, which would be false there."""
    import logging

    import pytest

    from core.FDT import fdt_pipeline
    from core.refusals import Refusal

    driven = []

    def _drive(cfg, om, **kw):
        driven.append(len(om))
        return torch.full((len(om),), 1 + 1j, dtype=torch.complex128)

    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", _drive)

    def refused(cfg, name):
        out = tmp_path / name
        out.mkdir()
        with pytest.raises(Refusal) as e:
            fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                                 writer=_LogWriter(out), seed=1)
        return e.value, _figures_in(out)

    # the real Campaign 1, one trajectory sent to +inf; the real figures
    _use_tone_simulator(monkeypatch, diverge=True)
    e, written = refused(_ToneCfg((0.1, 30.0)), "one_member_inf")
    msg = str(e)
    assert e.field == "cell", e.field
    assert msg.startswith("The spontaneous simulation diverged: its spectrum holds no finite value"), msg
    assert "freq_bounds" in msg and "psd_T_obs_nd = 2048" in msg and "dt_nd = 0.5" in msg, msg
    assert "multiplier" not in msg and "resolve" not in msg, f"a band sentence about no spectrum: {msg}"
    assert written == ["Spontaneous trajectory.png"], written

    # a Campaign 1 whose spectrum and trajectory are NaN throughout
    freqs_psd = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    nan4 = torch.full((4,), float("nan"), dtype=torch.float64)

    def campaign1(G, x=nan4):
        monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd",
                            lambda cfg, return_trajectory=False: (
                                freqs_psd, G, torch.arange(4, dtype=torch.float64), x))

    campaign1(nan4)
    e, written = refused(_ToneCfg((0.1, 30.0)), "all_nan")
    assert e.field == "cell" and "diverged" in str(e), (e.field, str(e))
    assert written == ["Spontaneous trajectory.png"], written
    assert driven == [], "a diverged run reached the driven campaign"

    # the boundary, with the figures stubbed: non-finite in SOME bins is not the cell's refusal
    for name in ("plot_spontaneous_trajectory", "plot_psd", "plot_eff_temp_ratio",
                 "plot_chi_components"):
        monkeypatch.setattr(fdt_pipeline, name, lambda *a, **kw: None)
    zeros4 = torch.zeros(4, dtype=torch.float64)
    campaign1(torch.tensor([9.0, 1.0, 5.0, float("nan")], dtype=torch.float64), zeros4)
    caplog.clear()
    fdt_pipeline.run_fdt(_ToneCfg((0.5, 1.2)), skip_sanity=True, confirm_production=True,
                         writer=_LogWriter(tmp_path), seed=1)
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "3/5 probe frequencies have no value in the spontaneous spectrum, whose resolved band is "
        "0.1..0.3 (ND), and are blank in the ratio."], caplog.records
    assert driven == [5], driven

    campaign1(torch.tensor([9.0, 1.0, float("nan"), float("nan")], dtype=torch.float64), zeros4)
    e, _ = refused(_ToneCfg((1.0, 1.4)), "inside_the_band")
    msg = str(e)
    assert e.field == "freq_bounds" and "has no value at any of the 5 probe frequencies" in msg, msg
    assert "Nyquist" not in msg and "multiplier" not in msg, f"a false sentence about the top: {msg}"
    assert driven == [5], "every probe was blank, yet the driven campaign ran"


def test_fdt_support_gate():
    """Built-ins are supported; user models are gated on additive, non-zero, unforced observable noise."""
    for m in ("NADROWSKI", "HOPF", "BP"):
        assert registry.fdt_support(m) == (True, ""), m
    assert registry.fdt_support("NOPE")[0] is False

    cases = [
        ("FDTGOK",  [{"name": "x", "drift": "-k*x", "D": "d0"}],                       True,  ""),
        ("FDTGMUL", [{"name": "x", "drift": "-k*x", "D": "0.5*x^2"}],                  False, "multiplicative"),
        ("FDTGZ",   [{"name": "x", "drift": "-x", "D": "0"}],                          False, "deterministic"),
        ("FDTGF",   [{"name": "x", "drift": "-k*x", "D": "d0",
                      "forcing": {"kind": "sin", "params": {}}}],                      False, "forcing"),
    ]
    for name, variables, ok, needle in cases:
        try:
            _register_user(name, variables)
            got_ok, reason = registry.fdt_support(name)
            assert got_ok is ok, (name, got_ok, reason)
            assert needle in reason, (name, reason)
        finally:
            registry.unregister(name)


def test_n_force_channels_user_vs_builtin():
    """A user model needs one force channel per state variable; built-ins keep the 1-or-2 convention."""
    name = "FDTNCH"
    try:
        _register_user(name, [{"name": "x", "drift": "v", "D": "0"},
                              {"name": "v", "drift": "-k*x", "D": "d0"}])
        assert _n_force_channels(_FakeCfg(name, n_vars=2)) == 2
    finally:
        registry.unregister(name)
    assert _n_force_channels(_FakeCfg("BP", force_params={})) == 1
    assert _n_force_channels(_FakeCfg("HOPF", force_params={"amp_y": 0.0})) == 2


def test_the_fdt_messages_are_records_with_their_own_levels(tmp_path, monkeypatch, caplog, capsys):
    """V4 in the FDT pipeline and the sweep study (piece 3). Banners, per-point progress, the sanity
    table and the saved-plot paths are information; a failed sanity verdict and a plot that had to
    drop non-finite points are warnings; a Campaign-2 point that FAILED and was recorded rather than
    raised is an error. Until piece 3 all of it was print(): the only severity marker was a hand-typed
    "WARNING:" word, and the per-point failure of an overnight sweep scrolled past at the level of its
    progress lines.

    Stubbed at the campaign seams, so nothing is simulated: two operating points, the first failing in
    Campaign 2. The sweep still completes, logs one error, warns about the count (a Python warning,
    unchanged) and says so. The hand-typed "WARNING: " word is gone from the sanity verdict because
    the level carries it -- on the tool it would have read "warning: WARNING: ...". No record starts or
    ends with a blank line, and nothing reaches stdout: the front ends' handlers put records there
    (Task 16)."""
    import logging

    import numpy as np
    import pytest

    from core import config
    from core.FDT import cross_validation as cv
    from core.FDT import fdt_pipeline, plots, sanity

    assert logging.getLogger("core").level == logging.INFO, "core/runs.py sets it at import"

    def records():
        return [(r.name, r.levelname, r.getMessage()) for r in caplog.records if r.name.startswith("core")]

    # ── the sweep study: information progress, one ERROR per failed point ─────────────────────────
    class _SweepCfg:
        model, n_freqs, ensemble_M, F0, psd_T_obs_nd = "STUB", 4, 2, 0.1, 1.0
        freqs_per_batch, freq_bounds, burn_in_nd = 1, (0.1, 30.0), 100.0
        T_obs_periods, dt_nd, seed = 30, 0.01, 5
        preset_name = None
        hw = config.cpu_device()

        def with_overrides(self, **kw):
            return self

    calls = []

    def _campaign2(cfg_op, omegas, freqs_psd, G):
        calls.append(cfg_op)
        if len(calls) == 1:
            raise RuntimeError("stub out of memory")
        return torch.ones(4, dtype=torch.complex128), torch.full((4,), 2.0, dtype=torch.float64)

    monkeypatch.setattr(cv, "run_campaign1_psd",
                        lambda c: (torch.linspace(0.1, 3.0, 8, dtype=torch.float64),
                                   torch.ones(8, dtype=torch.float64)))
    monkeypatch.setattr(cv, "_detect_resonance", lambda omegas, G, w0: (1.0, True))
    monkeypatch.setattr(cv, "_build_common_grid",
                        lambda cfg, w0s, res: (torch.linspace(0.5, 2.0, 4, dtype=torch.float64), 1.0))
    monkeypatch.setattr(cv, "_campaign2_ratio", _campaign2)
    monkeypatch.setattr(cv, "observable_noise_prefactor", lambda c: 1.0)   # the "STUB" model has none
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param", lambda *a, save_path=None, **k: None)
    w = _LogWriter(tmp_path)
    caplog.clear()
    # The stub's two trajectories are below the thin-setting threshold (E5). A sweep called on its own
    # does NOT warn it -- the study warns once, at its top, for both of its records (Task 12's review,
    # pinned by test_a_thin_study_warns_once_and_both_records_keep_the_notice) -- but it still KEEPS
    # the sentence in its body. Only the failed-point count is raised here.
    with pytest.warns(UserWarning, match="1/2 operating points failed") as raised:
        cv.run_fdt_param_sweep(_SweepCfg(), "s", np.array([0.0, 0.1]), {"temp": 1.0}, writer=w)
    assert len(raised) == 1, [str(x.message) for x in raised]
    assert w.body["notices"] == fdt_pipeline.thin_notices(_SweepCfg()) and w.body["notices"]
    got = records()
    name = "core.FDT.cross_validation"
    assert (name, "INFO", "--- Phase A (s sweep): spontaneous PSD + omega_0 detection ---") in got, got
    assert (name, "INFO", "  [A 1/2] s=0 (temp=1.0)") in got, got
    assert (name, "INFO", "      omega_0 = 1.0000") in got, got
    assert (name, "INFO", "Common grid: 4 pts spanning [0.5000, 2.0000] (omega_0_ref=1.0000)") in got, got
    assert (name, "ERROR", "      Campaign 2 FAILED: stub out of memory") in got, got
    assert (name, "INFO", "      T_eff/T peak = 2") in got, got
    assert got[-1] == (name, "INFO", f"s sweep complete (1/2 points). Saved to: {w.dir}"), got
    assert [lvl for _, lvl, _ in got].count("ERROR") == 1, got

    # ── the FDT run: a failed sanity verdict is a WARNING, without the hand-typed word ─────────────
    class _Hopf:
        model = "HOPF"
        # run_fdt resolves the normalisation prefactor before its first record (spec §3.4), and the
        # HOPF prefactor is 2/sigma_x^2 -- so this stub has to carry the one parameter it reads.
        params_dict = {"sigma_x": (0.1, None)}
        # run_fdt judges the thin settings next (E5) and reads both knobs to do it. At these values
        # nothing is said, so the exact record list below is unchanged.
        n_freqs, ensemble_M = 60, 256
        # ...and it records every knob in body.settings and seeds on hw's device (piece 5).
        freqs_per_batch, F0 = 1, 0.05
        freq_bounds, burn_in_nd, T_obs_periods = (0.1, 30.0), 100.0, 30
        dt_nd, psd_T_obs_nd, seed = 0.01, 8000.0, None
        hw = config.cpu_device()

    monkeypatch.setattr(fdt_pipeline, "run_all_sanity",
                        lambda cfg, passive_plot_path=None: {"linearity": (False, {"ratio": 0.5})})
    caplog.clear()
    fdt_pipeline.run_fdt(_Hopf(), skip_sanity=False, confirm_production=False,
                         writer=_LogWriter(tmp_path), seed=1)
    name = "core.FDT.fdt_pipeline"
    assert records() == [
        (name, "INFO", "Cell file natural-frequency estimate: omega_0 ~= 1.0 (ND Hopf natural frequency)"),
        (name, "WARNING", "One or more sanity checks failed (see metrics above)."),
        (name, "INFO", "Aborted by user."),
    ], records()

    # ── the sanity table: information, a FAIL row included, no blank-line records ──────────────────
    monkeypatch.setattr(sanity, "check_linearity", lambda c: (True, {"x": 1}))
    monkeypatch.setattr(sanity, "check_ensemble_convergence", lambda c: (False, {"x": 2}))
    monkeypatch.setattr(sanity, "check_psd_window", lambda c: (True, {"x": 3}))
    caplog.clear()
    res = sanity.run_all_sanity(_Hopf())
    assert res == {"linearity": (True, {"x": 1}), "ensemble_convergence": (False, {"x": 2}),
                   "psd_window": (True, {"x": 3})}
    got = records()
    assert {(n, lvl) for n, lvl, _ in got} == {("core.FDT.sanity", "INFO")}, got
    msgs = [m for _, _, m in got]
    assert len(msgs) == 15 and msgs[1] == "FDT Sanity Checks" and msgs[-1] == "=" * 60, msgs
    assert "  FAIL  metrics: {'x': 2}" in msgs and "=" * 60 + "\nSummary:" in msgs, msgs
    assert not any(m.startswith("\n") or m.endswith("\n") for m in msgs), msgs

    # ── the plot helpers: dropped non-finite points are WARNINGs ───────────────────────────────────
    caplog.clear()
    plots.plot_psd(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 1.5]), save_path=tmp_path / "psd.png")
    plots.plot_eff_temp_ratio(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.inf, 1.2]),
                              save_path=tmp_path / "ratio.png")
    assert records() == [
        ("core.FDT.plots", "WARNING", "plot_psd: dropping 1/3 non-finite PSD points."),
        ("core.FDT.plots", "WARNING", "plot_eff_temp_ratio: dropping 1/3 non-finite (NaN/inf) points."),
    ], records()

    assert capsys.readouterr().out == ""


def test_make_fdt_config_refuses_every_zero_knob_a_blank_box_produces():
    """Review Focus 3, and spec §3.3's table. Nothing on this path was checked: freqs_per_batch=0
    makes campaigns._plan_adaptive_batches append (start, 0) for ever -- the application LOOKS HUNG
    rather than failed -- ensemble_M=0 raises ZeroDivisionError inside _pick_n_segs
    (FDT_MAX_ELEMENTS_PER_SEG // batch_size), F0=0 divides by zero in spectral.lock_in_chi
    (2.0 / (F0 * T_obs)), and n_freqs=0 produces an empty grid, an empty figure and exit 0.

    Each floor is asserted against the value a BLANK BOX produces and not only against a typed zero,
    because IntField.value() and FloatField.value() both return 0 for an empty field: a rule written
    as "reject below zero" would accept every blank field in the application. The refusals are
    raised BEFORE parse_cell, so a bad knob costs no file parsing, and each carries the field key
    its front-end table maps to a control or a flag."""
    import pytest

    from core import cli, config
    from core.refusals import Refusal

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    ok = dict(n_freqs=4, ensemble_M=8, freqs_per_batch=1, F0=0.05)

    assert cli.make_fdt_config("NADROWSKI", True, cell, **ok).n_freqs == 4

    for knob, field, sentence in (
            ("n_freqs", "n_freqs",
             "The number of drive frequencies must be at least 1; got 0 "
             "(default 60, or the preset's in a sweep)."),
            ("ensemble_M", "ensemble_m",
             "The number of trajectories per frequency must be at least 1; got 0 "
             "(default 256, or the preset's in a sweep)."),
            ("freqs_per_batch", "freqs_per_batch",
             "The number of frequencies per simulator call must be at least 1; got 0 (default 1)."),
            ("F0", "f0",
             "The non-dimensional drive amplitude must be greater than 0; got 0 (default 0.05).")):
        with pytest.raises(Refusal) as e:
            cli.make_fdt_config("NADROWSKI", True, cell, **{**ok, knob: 0})
        assert e.value.field == field and str(e.value) == sentence, str(e.value)
    # a seed is optional (None draws one), but a given one is a non-negative integer
    with pytest.raises(Refusal) as e:
        cli.make_fdt_config("NADROWSKI", True, cell, **ok, seed=-1)
    assert e.value.field == "seed" and "must be at least 0; got -1" in str(e.value), str(e.value)

    # ...and the value a blank box really produces, read off the widgets themselves
    from core.gui.widgets.labeled_inputs import FloatField, IntField
    from tests._fixtures import qt_app
    qt_app()
    blank_int, blank_float = IntField(60), FloatField(0.05)
    blank_int.setText("")
    blank_float.setText("")
    assert blank_int.value() == 0 and blank_float.value() == 0.0, "the premise of this test"
    with pytest.raises(Refusal, match="at least 1"):
        cli.make_fdt_config("NADROWSKI", True, cell, **{**ok, "n_freqs": blank_int.value()})
    with pytest.raises(Refusal, match="greater than 0"):
        cli.make_fdt_config("NADROWSKI", True, cell, **{**ok, "F0": blank_float.value()})


def test_make_fdt_config_refuses_a_missing_cell_an_unsupported_model_and_a_broken_band():
    """The rest of spec §3.3's table. A missing cell surfaced as FileNotFoundError from the parser;
    it is now refused by its input kind first, with field="cell", exactly as make_sim_config refuses
    a missing bounds file. The model row is NOT require_choice: registry.fdt_support is a predicate
    that returns a TAILORED diagnostic sentence per model (intrinsic forcing, multiplicative noise,
    a deterministic observable), not a list of choices, so the rule is refuse("model", reason) and
    fdt_support's own words are kept verbatim.

    The frequency band, the burn-in, the two durations and the step are parameters of neither
    builder and are exposed by neither front end (§1.2), so they are checked DEFENSIVELY under their
    own registered keys, which map to None in both front-end tables: the message names the setting
    and fix_sentence adds nothing, because there is no control and no flag (P2, P75). They are
    reached through with_overrides, which is how a hand-edited preset or a caller can produce one.

    The burn-in floor is 0 and 0 itself is legal (E5). A fractional negative and a NaN are refused
    too (F33): the count rule would have read -0.5 through int() as 0 and passed it, and raised a
    bare ValueError on a NaN."""
    import pytest

    from core import cli, config, registry
    from core.gui import fields as gui_fields
    from core.refusals import Refusal
    from core.tool import fields as tool_fields

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")

    with pytest.raises(Refusal) as e:
        cli.make_fdt_config("NADROWSKI", True, str(config.CELL_PATH / "nadrowski" / "nope.txt"))
    assert e.value.field == "cell" and "was not found" in str(e.value)
    with pytest.raises(Refusal) as e:
        cli.make_fdt_config("NADROWSKI", True, "")
    assert e.value.field == "cell" and "is blank" in str(e.value)

    try:
        _register_user("FDT_FORCED_CHECK", [
            {"name": "x", "drift": "-k*x", "D": "d0", "forcing": {"kind": "sin", "params": {}}}])
        ok_, reason = registry.fdt_support("FDT_FORCED_CHECK")
        assert not ok_ and "forcing" in reason, reason
        with pytest.raises(Refusal) as e:
            cli.make_fdt_config("FDT_FORCED_CHECK", False, cell)
        assert e.value.field == "model" and str(e.value) == reason, \
            "fdt_support's own per-model reason, kept verbatim and given a field key"
    finally:
        registry.unregister("FDT_FORCED_CHECK")

    good = cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=4, ensemble_M=8)
    for bad, key, needle in ((dict(dt_nd=0.0), "dt_nd", "must be greater than 0"),
                             (dict(psd_T_obs_nd=0.0), "psd_t_obs_nd", "must be greater than 0"),
                             (dict(T_obs_periods=0), "t_obs_periods", "must be greater than 0"),
                             (dict(burn_in_nd=-1.0), "burn_in_nd", "must be between 0 and inf; got -1 "),
                             (dict(burn_in_nd=-0.5), "burn_in_nd", "must be between 0 and inf; got -0.5 "),
                             (dict(burn_in_nd=math.nan), "burn_in_nd", "must be a finite number; got nan"),
                             (dict(burn_in_nd=math.inf), "burn_in_nd", "must be a finite number; got inf"),
                             (dict(freq_bounds=(0.0, 30.0)), "freq_bounds", "must be greater than 0"),
                             (dict(freq_bounds=(30.0, 0.1)), "freq_bounds",
                              "lower bound below its upper bound")):
        with pytest.raises(Refusal) as e:
            cli.check_fdt_settings(good.with_overrides(**bad))
        assert needle in str(e.value), str(e.value)
        assert e.value.field == key, (key, e.value.field)
        assert gui_fields.fix_sentence(key) == "" and tool_fields.fix_sentence(key) == "", \
            "no control and no flag: the message names the setting and offers no fix (P2)"
    assert cli.check_fdt_settings(good.with_overrides(burn_in_nd=0.0)) is None, "E5: a zero burn-in is legal"
    assert cli.check_fdt_settings(good) is None, "the built config passes its own check"


def test_make_param_sweep_config_refuses_a_blank_grid_and_records_the_preset_name():
    """Spec §4.4. The window checks NOTHING about its two grids today, and _GridRow.spec() is three
    value() calls -- so a grid whose 'max' was left empty arrives as (0.0, 0.0, 0) and np.linspace
    produces a sweep of zero points without a word. Each grid is therefore checked as a whole: both
    ends finite, at least 2 points, and the minimum below the maximum.

    ``preset_name`` exists because the builder takes ``preset`` as an already-RESOLVED dict and both
    call sites drop the name (core/tool/fdt.py's ``dict(cli.SWEEP_PRESETS[args.preset])``), while
    body.settings["preset"] has to hold the name a reader can act on. It is a closed choice in both
    front ends and is checked as one."""
    import numpy as np
    import pytest

    from core import cli, config
    from core.refusals import Refusal

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    preset = dict(cli.SWEEP_PRESETS["exploratory"])
    ok = dict(preset=preset, preset_name="exploratory", s_spec=(0.0, 0.5, 3), t_spec=(1.0, 1.5, 3))

    cfg, s_grid, t_grid = cli.make_param_sweep_config(cell, **ok)
    assert cfg.model == "NADROWSKI" and cfg.hw.device.type == "cpu"
    assert cfg.n_freqs == preset["n_freqs"] and cfg.ensemble_M == preset["ensemble_M"], \
        "an unset knob falls back to the preset, in the builder rather than at each call site"
    assert np.allclose(s_grid, np.linspace(0.0, 0.5, 3)) and len(t_grid) == 3
    assert cfg.sources["cell"] == cell
    assert cfg.preset_name == "exploratory", "body.settings['preset'] is read off the config (P72)"
    assert cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=4, ensemble_M=8).preset_name is None

    for spec, field, needle in (
            ((0.0, 0.0, 0), "s_grid", "at least 2 points"),
            ((0.0, 0.5, 1), "s_grid", "at least 2 points"),
            ((0.5, 0.1, 3), "s_grid", "lower bound below its upper bound"),
            ((float("nan"), 0.5, 3), "s_grid", "must be a finite number")):
        with pytest.raises(Refusal) as e:
            cli.make_param_sweep_config(cell, **{**ok, "s_spec": spec})
        assert e.value.field == field and needle in str(e.value), str(e.value)

    with pytest.raises(Refusal) as e:
        cli.make_param_sweep_config(cell, **{**ok, "t_spec": (1.5, 1.0, 3)})
    assert e.value.field == "t_grid", "the refusal names the grid the user actually broke"

    with pytest.raises(Refusal) as e:
        cli.make_param_sweep_config(cell, **{**ok, "preset_name": "overnight"})
    assert e.value.field == "preset" and str(e.value) == (
        "The resolution preset must be one of exploratory, production; got 'overnight' "
        "(default exploratory).")

    # §3.3's four shared knobs are checked here too, with the same wording the single-cell builder
    # uses: one rule set, two builders.
    with pytest.raises(Refusal) as e:
        cli.make_param_sweep_config(cell, **ok, ensemble_M=0)
    assert e.value.field == "ensemble_m" and "at least 1" in str(e.value)


def test_a_temperature_grid_reaching_below_zero_is_refused_and_zero_is_allowed():
    """The whole-piece review's M1, fix 3, as the owner ruled it (R-F1). A negative T_a/T is
    unphysical -- the active temperature below zero -- and it is the one known way a sweep point
    diverges: the model takes a square root of it, which torch answers with NaN rather than an error,
    so every such point's spectrum came back empty and the sweep booked the point as done. It is
    refused under ``t_grid`` before anything is spent. Zero stays LEGAL (no active noise is a
    meaningful operating point), and S has no floor: any other diverging point is caught by the
    sweep's own per-point checks."""
    import pytest

    from core import cli, config
    from core.refusals import Refusal

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    ok = dict(preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
              s_spec=(0.0, 0.5, 3), t_spec=(1.0, 1.5, 3))
    for spec in ((-1.0, -0.5, 2), (-0.1, 1.0, 3)):
        with pytest.raises(Refusal) as e:
            cli.make_param_sweep_config(cell, **{**ok, "t_spec": spec})
        msg = str(e.value)
        assert e.value.field == "t_grid", e.value.field
        assert msg.startswith("The temperature sweep grid must not reach below 0") and \
            f"got a minimum of {spec[0]:g}" in msg, msg
    _cfg, _s, t_grid = cli.make_param_sweep_config(cell, **{**ok, "t_spec": (0.0, 1.0, 3)})
    assert t_grid[0] == 0.0, "a zero T_a/T is a meaningful operating point and stays legal"
    _cfg, s_grid, _t = cli.make_param_sweep_config(cell, **{**ok, "s_spec": (-0.5, 0.5, 3)})
    assert s_grid[0] == -0.5, "the ruling sets no floor on S"


def test_a_thin_setting_warns_and_hands_back_the_sentence_for_the_record():
    """E5: checks refuse what BREAKS; a setting too thin to trust warns instead, and the warning is
    recorded. A one-frequency grid and a two-trajectory ensemble both produce a real number -- the
    computation is defined -- but the number is a quick look, and a record that does not say so
    reads later as a measurement. A floor here would forbid the quick look, which E5 explicitly
    does not.

    The channel is PreflightWarning, the same one every other judgement in the tree uses
    (core/refusals.py, re-exported by core.orchestrator), so the window shows it at warning
    severity, the tool sends it to stderr, and the run buffer copies it into the record's log.txt --
    and the SENTENCES come back so the stage can put them in body.notices, which is the half a
    warning alone cannot do.

    Eight trajectories is the threshold because the existing end-to-end test runs at eight and must
    keep passing without a notice; two frequencies because one frequency is not a spectrum."""
    import warnings

    import pytest

    from core import cli, config
    from core.FDT import fdt_pipeline
    from core.orchestrator import PreflightWarning

    assert (config.FDT_THIN_N_FREQS, config.FDT_THIN_ENSEMBLE_M) == (2, 8)

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    fat = cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=2, ensemble_M=8)
    assert fdt_pipeline.thin_notices(fat) == [], "at the thresholds exactly: nothing to say"

    thin = cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=1, ensemble_M=2)
    said = fdt_pipeline.thin_notices(thin)
    assert said == [
        "The frequency grid has 1 point, below the 2 this measurement is trusted at: read the "
        "result as a quick look, not as a measurement.",
        "The ensemble is 2 trajectories, below the 8 this measurement is trusted at: read the "
        "result as a quick look, not as a measurement.",
    ], said

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        returned = fdt_pipeline.warn_thin_settings(thin)
    assert returned == said, "the sentences come back for body.notices, not only to the warning hook"
    assert [str(w.message) for w in rec] == said
    assert all(issubclass(w.category, PreflightWarning) for w in rec), [w.category for w in rec]

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert fdt_pipeline.warn_thin_settings(fat) == []
    assert rec == [], "a run at the thresholds is not annotated"

    # the run entries raise it themselves: a notice the operator never sees is not a notice. The sweep
    # study warns at ITS top, once for both records, not in each sweep (Task 12's review: a thin study
    # warned twice, and the second warning arrived only after the whole first sweep had run).
    import inspect
    from core.FDT import cross_validation
    for fn in (fdt_pipeline.run_fdt, cross_validation.run_param_study_cli):
        assert "warn_thin_settings" in inspect.getsource(fn), fn.__name__
    assert "warn_thin_settings" not in inspect.getsource(cross_validation.run_fdt_param_sweep)


def test_the_thin_notice_is_the_one_judgement_class_and_names_the_stages_caller(tmp_path, monkeypatch):
    """Task 12, fix round 1. The class lives in the torch-free core.refusals so the FDT path can raise
    it without importing the SBI stack, and the orchestrator re-exports the SAME object, so every
    ``pytest.warns(orchestrator.PreflightWarning)`` in the tree still catches an FDT notice.

    The warning names whoever called the STAGE, as every other judgement does (orchestrator's
    _preflight_warn: stacklevel=3 through run boundaries). Pointing at the stage's own
    ``notices = warn_thin_settings(cfg)`` line told the operator nothing on the tool's stderr. The
    call goes through run_fdt here, because that is the frame layout the stacklevel is counted for:
    helper, stage, caller -- and since run_fdt became a public entry (Task 17), the @public_entry
    wrapper's frame between the stage and its caller is the one RUN_BOUNDARY_FILES skips."""
    import pytest

    from core import config, orchestrator, refusals
    from core.FDT import fdt_pipeline

    assert orchestrator.PreflightWarning is refusals.PreflightWarning

    class _ThinHopf:
        model = "HOPF"
        # the prefactor (2/sigma_x^2) is resolved before the thin check, so this stub carries sigma_x
        params_dict = {"sigma_x": (0.1, None)}
        n_freqs, ensemble_M = 1, 2
        freqs_per_batch, F0 = 1, 0.05
        freq_bounds, burn_in_nd, T_obs_periods = (0.1, 30.0), 100.0, 30
        dt_nd, psd_T_obs_nd, seed = 0.01, 8000.0, None
        hw = config.cpu_device()

    monkeypatch.setattr(fdt_pipeline, "run_all_sanity",
                        lambda cfg, passive_plot_path=None: {"linearity": (True, {"ratio": 1.0})})
    with pytest.warns(refusals.PreflightWarning) as rec:
        fdt_pipeline.run_fdt(_ThinHopf(), skip_sanity=False, confirm_production=False,
                             writer=_LogWriter(tmp_path), seed=1)
    said = [w for w in rec if issubclass(w.category, refusals.PreflightWarning)]
    assert [str(w.message) for w in said] == fdt_pipeline.thin_notices(_ThinHopf()), \
        [str(w.message) for w in said]
    for w in said:
        assert Path(w.filename).resolve() == Path(__file__).resolve(), (w.filename, w.lineno)


def test_the_thin_notice_loads_no_sbi_and_keeps_its_always_filter():
    """Task 12, fix round 1. An FDT or sweep run needs no inference machinery, and importing it cost
    every ``fdt``/``crossval`` tool run about two seconds plus two false pytensor "g++ not
    available" warnings at the head of its output, thin settings or not. Checked in a FRESH
    interpreter, because this process holds the orchestrator through the session fixtures, so a
    sys.modules check here would pass vacuously.

    The same probe pins the "always" filter: it is installed when core.refusals is imported, not as a
    side effect of a lazy orchestrator import, so a repeated notice from one call site is still shown
    the second time (without the filter Python's once-per-location registry would drop it)."""
    import subprocess

    probe = ("import sys, warnings\n"
             "from core.FDT import fdt_pipeline\n"
             "class C:\n"
             "    def __init__(self, n, m):\n"
             "        self.n_freqs, self.ensemble_M = n, m\n"
             "def stage(c):\n"
             "    return fdt_pipeline.warn_thin_settings(c)\n"
             "with warnings.catch_warnings(record=True) as rec:\n"
             "    for _ in range(2):\n"
             "        assert stage(C(60, 256)) == []\n"
             "        assert len(stage(C(1, 2))) == 2\n"
             "shown = [str(w.message)[:24] for w in rec]\n"
             "bad = sorted(m for m in ('core.orchestrator', 'sbi', 'pytensor') if m in sys.modules)\n"
             "sys.exit(0 if len(shown) == 4 and not bad else repr((shown, bad)))\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(Path(__file__).resolve().parents[1]),
                       capture_output=True, text=True, timeout=300,
                       env={**os.environ, "MPLBACKEND": "Agg", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    assert r.returncode == 0, r.stdout + r.stderr


def _stub_campaigns(monkeypatch, n_psd=1601):
    """Campaign 1 and Campaign 2 replaced by arithmetic: a single-peaked spectrum on a positive grid
    and a flat susceptibility. The record's SHAPE is what these tests are about; a real campaign is
    slow-marked and lives in tests/test_tool.py.

    The spectrum runs 0..40 with a bin every 0.025 and its peak at 1.0, so the default probe grid
    (0.1..30 around that peak) is resolved end to end: the band refusal (spec §3.4) and the blanks
    (spec §3.5) stay out of a test that is not about them."""
    import torch
    from core.FDT import fdt_pipeline

    omegas_psd = torch.linspace(0.0, 40.0, n_psd, dtype=torch.float64)
    G = torch.exp(-((omegas_psd - 1.0) ** 2) / 0.02) + 1e-6
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)

    def _c1(cfg, return_trajectory=False):
        if return_trajectory:
            return omegas_psd, G, t, torch.zeros_like(t)
        return omegas_psd, G

    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _c1)
    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi",
                        lambda cfg, omegas: torch.full(omegas.shape, 1 + 1j, dtype=torch.complex128))
    for name in ("plot_psd", "plot_eff_temp_ratio", "plot_chi_components",
                 "plot_spontaneous_trajectory"):
        monkeypatch.setattr(fdt_pipeline, name,
                            lambda *a, save_path=None, **k: Path(save_path).write_bytes(b"\x89PNG"))


def test_run_fdt_writes_a_record_and_leaves_the_callers_config_alone(store, monkeypatch):
    """E1 and V1 together, which is the whole point of making this a public entry. Before piece 5 the
    measurement returned None, wrote five timestamped PNGs into a flat <artifacts root>/fdt that no
    command could list or delete, and wrote ``cfg.omega_0`` on the panel's own settings object twice
    (core/FDT/fdt_pipeline.py's steps 1 and 4) -- so the frequency grid of the NEXT run started from
    the last run's resonance. The record answers the first; copy_for_run, which public_entry is
    duck-typed on, answers the second.

    The body must carry EVERY key of BODY_KEYS["fdt"] (manifest.validate compares the key set by
    equality), with the keys a single-cell run cannot fill left null. Two trajectories is below the
    trust threshold, so the run warns -- asserted, never leaked -- and the record KEEPS the sentence.

    The seed argument OVERRIDES the one the config was built with (P12), and the record's config block
    must say so from its FIRST manifest: store.create computes that block from the caller's object,
    before any seed is resolved, so it read the builder's seed (or none) while the body held the one
    the run used -- and `artifacts show` prints the config block (fix round 1, finding 3)."""
    import json

    import pytest

    from core import cli, config
    from core.artifacts import manifest as mf
    from core.FDT import fdt_pipeline
    from core.refusals import PreflightWarning
    from tests._fixtures import assert_cfg_unchanged, snapshot_cfg

    _stub_campaigns(monkeypatch)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=5, ensemble_M=2, seed=3)
    snap = snapshot_cfg(cfg)

    w = store.create("fdt", cfg, name="run1", note="a stubbed measurement")
    stub_c1, first_manifest_seed = fdt_pipeline.run_campaign1_psd, []

    def _c1(c, return_trajectory=False):
        # Campaign 1 runs after __enter__ and before the first refresh: this is the FIRST manifest.
        first_manifest_seed.append(
            json.loads((w.dir / "manifest.json").read_text(encoding="utf-8"))["config"]["seed"])
        return stub_c1(c, return_trajectory=return_trajectory)

    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _c1)
    with pytest.warns(PreflightWarning, match="The ensemble is 2 trajectories"):
        rec = fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True, writer=w, seed=11)

    assert first_manifest_seed == [11], f"the first manifest's config block said {first_manifest_seed}"
    assert rec.manifest.config["seed"] == 11, "the config block names the seed the run USED"
    assert rec.name == "run1" and rec.body["study"] == "single"
    assert set(rec.body) == set(mf.BODY_KEYS["fdt"]), "validate compares the key set by equality"
    assert rec.body["seed"] == 11 and rec.body["complete"] is True
    assert rec.body["points"] is None and rec.body["compared"] is None
    assert rec.body["notices"] == fdt_pipeline.thin_notices(cfg) and len(rec.body["notices"]) == 1, \
        "ensemble_M=2 is below FDT_THIN_ENSEMBLE_M: T12's sentence is KEPT in the record (P51)"
    assert rec.body["offgrid"]["blanks"] == 0
    assert rec.body["settings"]["confirm_production"] is True, "P70"
    assert rec.body["grid"]["n_freqs"] == 5 and rec.body["grid"]["omega_0"] > 0.0
    assert rec.body["offgrid"]["of"] == 5
    assert rec.body["settings"]["ensemble_M"] == 2 and rec.body["settings"]["skip_sanity"] is True
    assert sorted(rec.manifest.figures) == ["figures/chi_components.png",
                                            "figures/effective_temperature_ratio.png",
                                            "figures/spontaneous_psd.png",
                                            "figures/spontaneous_trajectory.png"]
    assert (rec.path / "log.txt").read_text(encoding="utf-8").strip(), \
        "log.txt is written from runs.current_run_log(), which only exists on the thread that " \
        "entered the writer -- an empty file means the front end entered it instead"
    assert_cfg_unchanged(cfg, snap)


def test_the_single_cell_record_holds_the_numbers_not_only_the_pictures(store, monkeypatch):
    """E1 / spec §3.8. Every number the four figures draw is recoverable from the record: before piece
    5 the grid, the spectrum, the susceptibility and the ratio existed only inside the run, so a
    re-plot -- and any comparison of two cells -- meant re-running hours of simulation. The ``study``
    attribute at the root is what lets a reader check the layout before reading it: three layouts
    share this filename (spec §2.3). The dataset names are the interface contract's, the vocabulary
    the sweep file already uses (P5, P71).

    Two trajectories is below the trust threshold, so the run warns; asserted, never leaked."""
    import h5py
    import numpy as np
    import pytest
    from core import cli, config
    from core.FDT import fdt_pipeline
    from core.refusals import PreflightWarning

    _stub_campaigns(monkeypatch)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=5, ensemble_M=2)
    with pytest.warns(PreflightWarning, match="The ensemble is 2 trajectories"):
        rec = fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                                   writer=store.create("fdt", cfg, name="numbers"), seed=2)

    assert rec.data_path == rec.path / "data.h5" and rec.data_path.exists()
    assert rec.manifest.payloads["data.h5"], "a committed payload is hashed (spec §2.2)"
    with h5py.File(rec.data_path, "r") as h5:
        assert h5.attrs["study"] == "single"
        assert h5.attrs["model"] == "HOPF"
        assert float(h5.attrs["prefactor"]) == 2.0 / cfg.params_dict["sigma_x"][0] ** 2
        assert h5["omega_grid"].shape == (5,) and h5["T_eff_over_T"].shape == (5,)
        assert h5["chi_prime"].dtype == np.float64 and h5["chi_prime"].shape == (5,)
        assert h5["chi_double_prime"].shape == (5,)
        assert np.allclose(h5["chi_double_prime"][...], 1.0), "the stub's chi is 1+1j"
        assert float(h5.attrs["omega_0"]) == rec.body["grid"]["omega_0"]
        assert h5["PSD_omegas"].shape == h5["PSD_G"].shape, \
            "the spontaneous spectrum carries its OWN frequency axis -- it is not on the chi grid"
        assert h5["PSD_omegas"].shape != h5["omega_grid"].shape


def test_a_failed_fdt_run_keeps_its_record_marked_unfinished(store, monkeypatch):
    """E2: an interrupted or crashed measurement keeps its folder, plainly marked unfinished, and the
    spontaneous spectrum it did collect is what diagnoses the failure. The six ordinary kinds still
    delete theirs -- tests/test_artifact_store.py pins that -- so this is the one place the
    progressive mode is visible from a stage.

    The numbers file is asked for only once both campaigns are in (P49), so a run that dies in the
    driven campaign leaves a record that neither lists ``data.h5`` nor holds one: a listed payload
    that was never written would be a phantom, as a listed figure that was never drawn is."""
    import json

    import pytest
    from core import cli, config
    from core.FDT import fdt_pipeline

    _stub_campaigns(monkeypatch)
    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi",
                        lambda cfg, omegas: (_ for _ in ()).throw(RuntimeError("stub out of memory")))
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=3, ensemble_M=8)
    w = store.create("fdt", cfg, name="halfway")
    with pytest.raises(RuntimeError, match="stub out of memory"):
        fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True, writer=w, seed=3)

    (summary,) = store.list("fdt")
    assert summary.name == "halfway" and summary.complete and not summary.finished
    assert (w.dir / "figures" / "spontaneous_trajectory.png").exists(), \
        "the figures drawn before the failure stay on disk"
    payloads = json.loads((w.dir / "manifest.json").read_text(encoding="utf-8"))["payloads"]
    assert payloads == {} and not (w.dir / "data.h5").exists(), \
        f"the numbers file was handed out before the driven campaign: {payloads}"


def test_a_figure_that_fails_after_the_driven_campaign_does_not_cost_the_numbers(store, monkeypatch):
    """E2's point is that a failed run keeps what it MEASURED. Once both campaigns are in, the numbers
    are hours of simulation and the two final figures are minutes of matplotlib, so ``data.h5`` is
    written first: a figure that fails to draw (a matplotlib error, a full disk) leaves an unfinished
    record whose numbers are on disk and readable, and a re-plot needs no re-run (Task 18's ruling).
    The file is listed with a null hash, as every payload of an unfinished record is (P47)."""
    import json

    import h5py
    import pytest
    from core import cli, config
    from core.FDT import fdt_pipeline

    _stub_campaigns(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("stub figure failure")

    monkeypatch.setattr(fdt_pipeline, "plot_eff_temp_ratio", _boom)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=3, ensemble_M=8)
    w = store.create("fdt", cfg, name="unplotted")
    with pytest.raises(RuntimeError, match="stub figure failure"):
        fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True, writer=w, seed=3)

    (summary,) = store.list("fdt")
    assert summary.name == "unplotted" and summary.complete and not summary.finished
    payloads = json.loads((w.dir / "manifest.json").read_text(encoding="utf-8"))["payloads"]
    assert payloads == {"data.h5": None}, payloads
    with h5py.File(w.dir / "data.h5", "r") as h5:
        assert h5.attrs["study"] == "single"
        for name in ("omega_grid", "T_eff_over_T", "chi_prime", "chi_double_prime"):
            assert h5[name][...].shape == (3,), name
        assert h5["PSD_omegas"][...].shape == h5["PSD_G"][...].shape


def test_a_band_refused_after_the_spectrum_keeps_its_record_and_a_refused_cell_leaves_none(
        store, monkeypatch):
    """The two refusals a single-cell run can meet, and why they end differently (spec §2.2 step 3,
    E2, E4). The band refusal is knowable only once the spontaneous campaign has run and its two
    figures are on disk, so it is not a pre-spend refusal: the record is KEPT, unfinished, with the
    spectrum that diagnoses the refusal inside it and the resonance already in its body. The
    prefactor refusal is raised before anything is spent -- before the writer is even entered -- so
    a cell FDT cannot normalise leaves no folder at all, and every refused click would otherwise add
    one."""
    import json

    import pytest
    from core import cli, config
    from core.FDT import fdt_pipeline
    from core.refusals import Refusal

    _stub_campaigns(monkeypatch)
    driven = []
    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", lambda cfg, omegas: driven.append(1))
    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    # The stub spectrum's first real bin is 0.025 and its peak 1.0: a lower multiplier of 0.01 puts
    # the lowest probe below everything the spectrum resolves.
    cfg = cli.make_fdt_config("HOPF", False, cell, n_freqs=5, ensemble_M=8).with_overrides(
        freq_bounds=(0.01, 30.0))
    w = store.create("fdt", cfg, name="band")
    with pytest.raises(Refusal) as e:
        fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True, writer=w, seed=4)
    assert e.value.field == "freq_bounds", e.value.field
    assert driven == [], "the band refusal came after the driven campaign"

    (summary,) = store.list("fdt")
    assert summary.name == "band" and summary.complete and not summary.finished
    assert sorted(p.name for p in (w.dir / "figures").glob("*.png")) == [
        "spontaneous_psd.png", "spontaneous_trajectory.png"]
    body = json.loads((w.dir / "manifest.json").read_text(encoding="utf-8"))["body"]
    assert body["complete"] is False and body["grid"]["omega_0"] == pytest.approx(1.0), body["grid"]
    assert (w.dir / "log.txt").is_file(), "the unfinished record keeps its log up to the refusal"

    bad = cli.make_fdt_config("HOPF", False, cell, n_freqs=5, ensemble_M=8)
    bad.params_dict.pop("sigma_x")
    w2 = store.create("fdt", bad, name="nocell")
    with pytest.raises(Refusal) as e:
        fdt_pipeline.run_fdt(bad, skip_sanity=True, confirm_production=True, writer=w2, seed=4)
    assert e.value.field == "cell", e.value.field
    assert not w2.dir.exists(), "a refusal before anything was spent left a folder behind"
    assert [s.name for s in store.list("fdt")] == ["band"]


def test_run_fdt_draws_and_records_a_seed_when_none_is_given(store, monkeypatch):
    """E7: every run records the seed it used, so repeats of one cell can be told apart and their
    spread read as the measurement error. A blank Seed box and an absent --seed both mean "draw one
    and record it" -- a run whose seed were simply unset could never be repeated.

    The draw comes from a seeded Random patched in as the module's ``random``, never from reseeding
    Python's global stream: that would leave it seeded for every later test in the one-process gate.
    The config block records the drawn seed too, not the builder's None (fix round 1, finding 3)."""
    import random
    from core import cli, config
    from core.FDT import fdt_pipeline

    _stub_campaigns(monkeypatch)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=3, ensemble_M=8)
    drawn = random.Random(4242).randrange(2 ** 31)
    monkeypatch.setattr(fdt_pipeline, "random", random.Random(4242))
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                               writer=store.create("fdt", cfg, name="drawn"))
    assert rec.body["seed"] == drawn, "the drawn seed is the recorded seed"
    assert rec.manifest.config["seed"] == drawn, "the config block names the drawn seed, not None"
    assert cfg.seed is None, "the draw lands on the private copy, never on the caller's config"


def test_the_seed_determines_the_numbers(store, monkeypatch):
    """P82 / E7. Recording a seed is worth nothing unless the seed DETERMINES what the run draws --
    and compare repeats (E8) reads the spread across repeats as the measurement error on exactly that
    premise. Campaign 1 is stubbed to draw from the ambient torch generator, which is what the solver
    draws its noise from: one seed twice draws the same, another seed draws differently."""
    import torch
    from core import cli, config
    from core.FDT import fdt_pipeline

    _stub_campaigns(monkeypatch)
    stub_c1 = fdt_pipeline.run_campaign1_psd
    drawn = []

    def _c1(cfg, return_trajectory=False):
        drawn.append(float(torch.rand(())))
        return stub_c1(cfg, return_trajectory=return_trajectory)

    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _c1)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=3, ensemble_M=8)
    for seed in (5, 5, 6):
        fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                             writer=store.create("fdt", cfg), seed=seed)
    assert drawn[0] == drawn[1], "one seed, one draw: the seed determines the numbers"
    assert drawn[0] != drawn[2], "a different seed draws differently"


def test_a_sanity_run_lists_only_the_figures_it_drew(store, monkeypatch):
    """Fix round 1, finding 4. The passive-baseline check is NADROWSKI-only -- run_all_sanity drops it
    for every other model -- but run_fdt asked the writer for its figure path before the sanity run
    regardless, so every HOPF, BP or user-model sanity run listed figures/passive_baseline_ratio.png
    in its manifest and the file never existed. One helper in core/FDT/sanity.py now owns the rule and
    both sides ask it.

    The REAL run_all_sanity runs here, so the two sides are held to the same rule: only the five
    checks are recorders (the passive one writes the figure it was handed a path for). For each
    model, the passive check runs exactly when its figure is listed, and every listed figure is on
    disk."""
    from core import cli, config, registry
    from core.FDT import fdt_pipeline, sanity

    _stub_campaigns(monkeypatch)
    ran = []

    def _recorder(name):
        def check(cfg, save_plot_path=None):
            ran.append(name)
            if save_plot_path is not None:
                Path(save_plot_path).write_bytes(b"\x89PNG")
            return True, {}
        return check

    for name in ("check_passive_baseline", "check_high_freq_fdt", "check_linearity",
                 "check_ensemble_convergence", "check_psd_window"):
        monkeypatch.setattr(sanity, name, _recorder(name))

    for model, cell, passive in (("HOPF", "hopf/cell.txt", False),
                                 ("NADROWSKI", "nadrowski/master_spont.txt", True)):
        ran.clear()
        cfg = cli.make_fdt_config(model, registry.state_dep_drift(model), str(config.CELL_PATH / cell),
                                  n_freqs=3, ensemble_M=8)
        rec = fdt_pipeline.run_fdt(cfg, skip_sanity=False, confirm_production=True,
                                   writer=store.create("fdt", cfg, name=model.lower()), seed=2)
        assert ("check_passive_baseline" in ran) is passive, (model, ran)
        assert ("figures/passive_baseline_ratio.png" in rec.manifest.figures) is passive, \
            (model, rec.manifest.figures)
        missing = [f for f in rec.manifest.figures if not (rec.path / f).is_file()]
        assert missing == [], f"{model}: the manifest lists figures that were never drawn: {missing}"


def test_an_fdt_run_loads_no_inference_machinery(tmp_path):
    """Fix round 1, finding 1. The seeding context used to live in core/diagnostics/rng.py, and
    importing ANY submodule runs its package's __init__ -- which imports the five diagnostics and
    through them core.orchestrator, sbi and pytensor: about two seconds and two false pytensor "g++"
    lines on stderr at the head of every `python -m core fdt`, refused runs included. It lives in
    core/rng.py now. The import-time probe above cannot see a cost paid when the run STARTS, so this
    one RUNS run_fdt in a fresh interpreter -- campaigns and figures stubbed, a temp store -- and then
    looks. A fresh interpreter because this process holds the orchestrator through the session
    fixtures, so a sys.modules check here would pass vacuously.

    The bare ``sbi`` package IS loaded, by the store rather than the run: provenance.env_info reads
    ``sbi.__version__`` into every manifest of every kind, and sbi's ``__init__`` imports nothing but
    its version string (``sbi`` and ``sbi.__version__``, measured at under 0.01 s). Every OTHER sbi
    module is the inference stack, and is forbidden here."""
    import subprocess

    probe = (
        "import sys\n"
        "import torch\n"
        "from core import cli, config\n"
        "from core.artifacts import ArtifactStore\n"
        "from core.FDT import fdt_pipeline as fp\n"
        "w = torch.linspace(0.0, 40.0, 1601, dtype=torch.float64)\n"
        "G = torch.exp(-((w - 1.0) ** 2) / 0.02) + 1e-6\n"
        "t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)\n"
        "fp.run_campaign1_psd = lambda cfg, return_trajectory=False: (w, G, t, torch.zeros_like(t))\n"
        "fp.run_campaign2_chi = lambda cfg, om: torch.full(om.shape, 1 + 1j, dtype=torch.complex128)\n"
        "for name in ('plot_psd', 'plot_eff_temp_ratio', 'plot_chi_components',\n"
        "             'plot_spontaneous_trajectory'):\n"
        "    setattr(fp, name, lambda *a, save_path=None, **k: save_path.write_bytes(b'png'))\n"
        "cfg = cli.make_fdt_config('HOPF', False, str(config.CELL_PATH / 'hopf' / 'cell.txt'),\n"
        "                          n_freqs=3, ensemble_M=8)\n"
        "store = ArtifactStore(sys.argv[1])\n"
        "rec = fp.run_fdt(cfg, skip_sanity=True, confirm_production=True,\n"
        "                 writer=store.create('fdt', cfg), seed=1)\n"
        "assert rec.body['complete'] is True, rec.body\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if m in ('core.diagnostics', 'core.orchestrator', 'pytensor')\n"
        "             or (m.startswith('sbi.') and m != 'sbi.__version__'))\n"
        "sys.exit(0 if not bad else repr(bad))\n")
    r = subprocess.run([sys.executable, "-c", probe, str(tmp_path / "Artifacts")],
                       cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True,
                       timeout=300, env={**os.environ, "MPLBACKEND": "Agg", "KMP_DUPLICATE_LIB_OK": "TRUE",
                                         "PRISM_ARTIFACTS": str(tmp_path / "Artifacts")})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pytensor" not in r.stderr, r.stderr


def _stub_the_study(monkeypatch):
    """Both of the sweep's campaigns replaced by arithmetic, and its 3-D figure by a file write: every
    operating point resonates at 1.0 and its ratio is 2.0 everywhere. The study's RECORDS are the
    subject; a real sweep is slow-marked and lives in tests/test_tool.py."""
    from core.FDT import cross_validation as cv

    monkeypatch.setattr(cv, "run_campaign1_psd",
                        lambda c: (torch.linspace(0.1, 3.0, 8, dtype=torch.float64),
                                   torch.ones(8, dtype=torch.float64)))
    monkeypatch.setattr(cv, "_detect_resonance", lambda omegas, G, w0: (1.0, True))
    monkeypatch.setattr(cv, "_campaign2_ratio",
                        lambda c, om, f, g: (torch.ones(om.shape, dtype=torch.complex128),
                                             torch.full(om.shape, 2.0, dtype=torch.float64)))
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param", lambda *a, save_path=None, **k:
                        Path(save_path).write_bytes(b"\x89PNG"))


def _thin_study_cfg():
    """The Nadrowski sweep config at its smallest: two points per grid, three frequencies, and two
    trajectories -- below FDT_THIN_ENSEMBLE_M, so the study says so once (E5)."""
    from core import cli, config
    return cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)


def test_a_study_writes_one_record_per_swept_parameter_under_one_seed(store, monkeypatch):
    """Spec §4.1 (E4, E7). Before piece 5 the study ran the S sweep, plotted it, then ran the T sweep,
    and an all-failed S sweep raised before the T sweep even started -- one listing entry for two
    measurements, and the second measurement hostage to the first. Two records make each sweep
    answerable on its own.

    ONE seed is drawn once and recorded on BOTH, because the two sweeps are one study: a reader who
    wants to repeat the study repeats it, not half of it. The config block says so too, from each
    record's first manifest (Task 17's fix round 1: `artifacts show` prints that block).

    The two records are created back to back, before either is entered, which is what the store's
    in-memory "minted" set is for: without it both got one id, and the T sweep loaded as the S one."""
    import h5py
    import pytest

    from core.FDT import cross_validation as cv
    from core.refusals import PreflightWarning

    _stub_the_study(monkeypatch)
    cfg, s_grid, t_grid = _thin_study_cfg()
    writers = {"s": store.create("fdt", cfg, name="sweep_s"),
               "temp": store.create("fdt", cfg, name="sweep_t")}
    with pytest.warns(PreflightWarning, match="The ensemble is 2 trajectories"):
        recs = cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=77)

    assert [r.name for r in recs] == ["sweep_s", "sweep_t"]
    assert len({r.id for r in recs}) == 2, [r.id for r in recs]
    assert [r.body["points"]["param"] for r in recs] == ["s", "temp"]
    assert {r.body["seed"] for r in recs} == {77}, "one study, one seed (spec §4.1)"
    assert {r.manifest.config["seed"] for r in recs} == {77}, "the config block names the seed USED"
    assert [r.body["settings"]["sweep_grid"] for r in recs] == [[0.0, 0.1, 2], [1.0, 1.1, 2]], \
        "F40: [min, max, N] as spec §2.3 says; the array itself is in data.h5"
    for r in recs:
        assert r.body["study"] == "sweep" and r.body["complete"] is True
        assert r.body["grid"] is None, "each point's grid is in data.h5, not in the body (§2.3)"
        assert r.body["settings"]["preset"] == "exploratory"
        assert r.data_path.exists() and r.manifest.figures, \
            "each sweep plots into its OWN record when it finishes -- no midpoint special case"
        with h5py.File(r.data_path, "r") as h5:
            assert h5.attrs["study"] == "sweep" and "prefactor" in h5.attrs, "the contract's root attributes (P6)"
            assert h5.attrs["omega_0"] == h5.attrs["omega_0_ref"] == 1.0
        assert r.body["settings"]["skip_sanity"] is None, "P70: a sweep has no sanity branch"
        assert (r.path / "log.txt").read_text(encoding="utf-8").strip(), \
            "each sweep enters its own writer on the thread whose run log becomes log.txt"


def test_a_thin_study_warns_once_and_both_records_keep_the_notice(store, monkeypatch):
    """E5, and Task 12's review. The notice used to be raised inside each sweep, so a thin study warned
    TWICE -- and the second warning, the T sweep's, arrived only after the whole S sweep had run, which
    is not a warning before the spend at all. It is raised ONCE, at the top of the study, and it names
    the front end's call (the study is the public entry, so stacklevel=3 past the run boundary lands
    on the caller -- here, this file). Each record still KEEPS the sentence in body.notices, because a
    warning scrolls away and the record is what a reader has later."""
    import pytest

    from core.FDT import cross_validation as cv
    from core.FDT import fdt_pipeline
    from core.refusals import PreflightWarning

    _stub_the_study(monkeypatch)
    cfg, s_grid, t_grid = _thin_study_cfg()
    writers = {"s": store.create("fdt", cfg), "temp": store.create("fdt", cfg)}
    with pytest.warns(PreflightWarning) as rec:
        recs = cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=1)

    said = [w for w in rec if issubclass(w.category, PreflightWarning)]
    assert [str(w.message) for w in said] == fdt_pipeline.thin_notices(cfg) and len(said) == 1, \
        [str(w.message) for w in said]
    assert Path(said[0].filename).resolve() == Path(__file__).resolve(), (said[0].filename, said[0].lineno)
    assert [r.body["notices"] for r in recs] == [fdt_pipeline.thin_notices(cfg)] * 2


def test_a_cell_the_sweep_cannot_normalise_opens_no_record(store, monkeypatch):
    """Spec §3.4, applied to the sweep, and Task 17's ordering carried to it. The normalisation
    constant is resolved BEFORE the writer is entered: resolved inside, after data.h5 had been handed
    out, a cell missing ``beta`` would have left an unfinished record around an empty file, for a run
    that never simulated anything. The study refuses the same cell at its top, before its thin notice
    (ruling F10): a study that is refused must not first warn about how far to trust its result."""
    import warnings

    import pytest

    from core.FDT import cross_validation as cv
    from core.FDT.campaigns import FDTModelError
    from core.refusals import PreflightWarning

    _stub_the_study(monkeypatch)
    cfg, s_grid, t_grid = _thin_study_cfg()
    cfg.params_dict.pop("beta")

    w = store.create("fdt", cfg)
    with pytest.raises(FDTModelError) as e:
        cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)
    assert e.value.field == "cell" and not w.dir.exists(), "a refused cell opens no record"

    writers = {"s": store.create("fdt", cfg), "temp": store.create("fdt", cfg)}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(FDTModelError):
            cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=1)
    assert not [w for w in rec if issubclass(w.category, PreflightWarning)], \
        "the refusal comes before the thin notice (F10)"
    assert not any(wr.dir.exists() for wr in writers.values()) and store.list("fdt") == []


def _sweep_stubs(monkeypatch, *, phase_a_fail=(), phase_b_fail=()):
    """Campaign 1 and Campaign 2 stubbed per operating point, with chosen indices made to fail.

    The counters are per CALL, so they run on across both sweeps of a study: Campaign-2 calls 0-1 are
    a two-point S grid's and 2-3 the T grid's. The figure stub writes the file it was handed, as
    _stub_the_study's does: a figure the manifest lists and the disk lacks is a phantom (Task 17).
    The spectrum spans 0.1..40, past the common grid's 0.2..30 (every stubbed omega_0 is 1.0), so no
    probe is off-grid -- consistent with a ratio that is finite everywhere -- unless a test says so."""
    import torch
    from core.FDT import cross_validation as cv

    seen_a = {"n": 0}

    def _c1(cfg_op):
        idx = seen_a["n"]
        seen_a["n"] += 1
        if idx in phase_a_fail:
            raise RuntimeError(f"stub phase-A failure at {idx}")
        return (torch.linspace(0.1, 40.0, 8, dtype=torch.float64),
                torch.ones(8, dtype=torch.float64))

    seen_b = {"n": 0}

    def _c2(cfg_op, omegas, freqs_psd, G):
        idx = seen_b["n"]
        seen_b["n"] += 1
        if idx in phase_b_fail:
            raise RuntimeError(f"stub phase-B failure at {idx}")
        return (torch.ones(omegas.shape, dtype=torch.complex128),
                torch.full(omegas.shape, 2.0, dtype=torch.float64))

    monkeypatch.setattr(cv, "run_campaign1_psd", _c1)
    monkeypatch.setattr(cv, "_detect_resonance", lambda omegas, G, w0: (1.0, True))
    monkeypatch.setattr(cv, "_campaign2_ratio", _c2)
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param",
                        lambda *a, save_path=None, **k: Path(save_path).write_bytes(b"\x89PNG"))


def test_a_sweep_counts_failures_in_both_phases(store, monkeypatch):
    """E4: some points failed is a COMPLETED record carrying the count, and both phases count.

    Phase A caught nothing before piece 5 (core/FDT/cross_validation.py's Phase A loop has no try at
    all), so one operating point whose spontaneous campaign raised -- a transient OOM, a solver blow-up
    at the grid's far end -- ended the whole study with a traceback and left no record of the points
    that had already worked. The counts are what let a reader judge the answer.

    The counts are kept CURRENT as the sweep runs -- a browser row watching an hours-long sweep reads
    them from the manifest each refresh writes -- and ``done`` is COUNTED, never derived as planned
    minus failed, which would call every not-yet-run point done from the first refresh on. A finished
    sweep fills ``results`` and ``offgrid`` too (P78: spec §2.3 leaves them null only until the run
    finishes)."""
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch, phase_a_fail=(0,), phase_b_fail=(1,))
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.3, 4), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    cfg.seed = 1
    w = store.create("fdt", cfg, name="partly")
    refreshed, refresh = [], w.refresh

    def _spy():
        refreshed.append((w.body["points"]["done"], w.body["points"]["failed"]))
        refresh()

    w.refresh = _spy
    with pytest.warns(UserWarning, match="operating points failed"):
        rec = cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)

    assert rec.body["complete"] is True, "some points failed is a completed record (E4)"
    assert rec.body["points"] == {"param": "s", "planned": 4, "done": 2, "failed": 2}
    assert rec.body["results"] is not None and rec.body["results"]["peak_ratio"] == 2.0, "P78"
    assert rec.body["offgrid"] == {"blanks": 0, "of": 2 * 3}, \
        "the 2 LANDED points x the 3-point common grid (every stubbed omega_0 is 1.0): the two " \
        "failed points' slots were never measured (the whole-piece review's M2, departing from P78)"
    assert refreshed == [(0, 1), (1, 1), (1, 2), (2, 2), (2, 2)], \
        "one refresh per point as it lands or fails (A: point 0 fails; B: 1 lands, 2 fails, 3 " \
        f"lands), then the final one: {refreshed}"
    (summary,) = store.list("fdt")
    assert (summary.points_done, summary.points_failed, summary.points_planned) == (2, 2, 4)


def test_a_sweep_with_every_point_failed_refuses_after_its_record_is_written(store, monkeypatch):
    """E4 and §4.3. A run that measured NOTHING refuses, naming the setting to change -- today it is a
    RuntimeError the command line reports as a crash -- and the refusal is raised AFTER the final
    refresh, so the spectra the message tells the reader to look at are already on disk. The folder
    stays (E2): that is the whole reason the first phase's PSDs are worth keeping. The message names
    the cell by its file name as well as the grid (F41: spec §4.3's "naming the grid and the cell").
    The sweep's own count warning fires first, as it does for any failed point; asserted, never
    leaked."""
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch, phase_b_fail=(0, 1))
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    cfg.seed = 1
    w = store.create("fdt", cfg, name="nothing")
    with pytest.warns(UserWarning, match="2/2 operating points failed"):
        with pytest.raises(Refusal) as e:
            cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)
    assert e.value.field == "s_grid"
    assert "all 2" in str(e.value)
    assert "master_spont.txt" in str(e.value), "F41: the refusal names the cell"
    assert "spontaneous spectra of the 2" in str(e.value), "every first campaign landed, and it says so"

    (summary,) = store.list("fdt")
    assert summary.complete and not summary.finished, "unfinished, listed, and deletable"
    assert summary.points_failed == 2
    body = store.get("fdt", summary.id).body
    assert body["points"]["failed"] == 2, "the final refresh ran BEFORE the refusal"
    assert body["results"] is None and body["offgrid"] is None, "null until the run finishes (§2.3)"
    assert (w.dir / "data.h5").exists(), "the message points at the PSDs; they must be there"


def test_a_sweep_whose_first_phase_lost_every_point_refuses_without_a_common_grid(store, monkeypatch):
    """Step 6's guard. When Phase A loses EVERY point there is no resonance to build the common grid
    around -- ``_build_common_grid`` would raise ``ValueError: min() arg is an empty sequence`` -- so
    Phase B is skipped and the sweep ends in the same calm refusal. The message must stay true of
    what the record holds: not one spontaneous campaign landed, so it holds no spectra and says so."""
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch, phase_a_fail=(0, 1))
    driven = []
    monkeypatch.setattr(cv, "_campaign2_ratio", lambda *a: driven.append(a))
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    cfg.seed = 1
    w = store.create("fdt", cfg, name="no_spectra")
    with pytest.warns(UserWarning, match="2/2 operating points failed"):
        with pytest.raises(Refusal) as e:
            cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)
    assert e.value.field == "s_grid" and "all 2" in str(e.value)
    assert "no spectra" in str(e.value) and "spectra of the" not in str(e.value), str(e.value)
    assert driven == [], "Phase B ran with no surviving point"
    (summary,) = store.list("fdt")
    assert (summary.points_done, summary.points_failed) == (0, 2) and not summary.finished


def _diverging_phase_a(monkeypatch, diverged):
    """Phase A's campaign stubbed so the points whose CALL index is in ``diverged`` come back the way a
    diverged simulation does -- a spectrum with no finite value at any positive frequency -- and every
    other point comes back as ``_sweep_stubs``' flat spectrum."""
    from core.FDT import cross_validation as cv

    calls = {"n": 0}

    def _c1(cfg_op):
        idx = calls["n"]
        calls["n"] += 1
        G = (torch.full((8,), math.nan, dtype=torch.float64) if idx in diverged
             else torch.ones(8, dtype=torch.float64))
        return torch.linspace(0.1, 40.0, 8, dtype=torch.float64), G

    monkeypatch.setattr(cv, "run_campaign1_psd", _c1)


def _point_errors(data_path) -> dict:
    """``{group key: its error attribute}`` for every operating point of a sweep's data.h5 that has one."""
    import h5py
    with h5py.File(data_path, "r") as h5:
        ops = h5["operating_points"]
        return {k: str(ops[k].attrs["error"]) for k in sorted(ops) if "error" in ops[k].attrs}


def test_a_sweep_point_whose_spontaneous_simulation_diverged_is_a_failed_point(store, monkeypatch):
    """The whole-piece review's M1 (C1). Phase A never looked at the spontaneous spectrum: a point
    whose simulation diverged returned a spectrum NaN in every bin, the resonance search quietly fell
    back to the linearised estimate, Phase B then booked every one of its probes as off-grid and
    counted the point DONE -- so a sweep where every point diverged committed a finished record,
    ``done N, failed 0``, which the CrossVal picker offered and ``compare sweeps`` accepted. A
    single-cell run refuses the same spectrum (field ``cell``). The sweep now applies that run's own
    test, with its own sentence, straight after each point's Campaign 1: the point is FAILED, its
    reason in its ``error`` attribute, and a sweep that measured nothing reaches the all-failed
    refusal it always should have.

    Two cases: every point diverged (the refusal, an unfinished record, Phase B never entered), and one
    point of three (a finished record counting it failed)."""
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch)
    _diverging_phase_a(monkeypatch, {0, 1})
    driven = []
    monkeypatch.setattr(cv, "_campaign2_ratio", lambda *a: driven.append(a))
    cfg, s_grid, _t = _thin_study_cfg()
    cfg.seed = 1
    w = store.create("fdt", cfg, name="all_diverged")
    with pytest.warns(UserWarning, match="2/2 operating points failed"):
        with pytest.raises(Refusal) as e:
            cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)
    assert e.value.field == "s_grid" and "measured nothing" in str(e.value), str(e.value)
    assert driven == [], "Phase B ran on points whose spectra diverged"
    (summary,) = store.list("fdt")
    assert not summary.finished and (summary.points_done, summary.points_failed) == (0, 2)
    errors = _point_errors(w.dir / "data.h5")
    assert sorted(errors) == ["000", "001"], errors
    assert all(m.startswith("The spontaneous simulation diverged") for m in errors.values()), errors

    # one point of three diverged: counted failed, and the sweep finishes on the other two
    _sweep_stubs(monkeypatch)
    _diverging_phase_a(monkeypatch, {1})
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.2, 3), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2, seed=1)
    with pytest.warns(UserWarning, match="1/3 operating points failed"):
        rec = cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                     writer=store.create("fdt", cfg, name="one_diverged"))
    assert rec.body["complete"] is True
    assert rec.body["points"] == {"param": "s", "planned": 3, "done": 2, "failed": 1}, rec.body["points"]
    assert list(_point_errors(rec.data_path)) == ["001"]


def test_a_sweep_point_whose_driven_simulation_diverged_is_a_failed_point(store, monkeypatch):
    """M1's Phase-B half. A driven campaign whose susceptibility holds no finite value at any probe
    measured nothing at that point, whatever the spontaneous spectrum supplied: it used to LAND, as a
    point with every probe blank. It is a failed point now, with its reason recorded, and the rest of
    the sweep is unaffected. A chi'' that is non-finite at SOME probes still lands (the off-grid test
    above pins that it adds nothing to ``offgrid``)."""
    import pytest
    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch)
    driven = []

    def _c2(cfg_op, omegas, freqs_psd, G):
        chis = torch.full(omegas.shape, 1 + 1j, dtype=torch.complex128)
        if not driven:                                       # point 0's driven campaign diverged
            chis = torch.full(omegas.shape, complex(math.nan, math.nan), dtype=torch.complex128)
        driven.append(1)
        return chis, torch.full(omegas.shape, 2.0 if len(driven) > 1 else math.nan,
                                dtype=torch.float64)

    monkeypatch.setattr(cv, "_campaign2_ratio", _c2)
    cfg, s_grid, _t = _thin_study_cfg()
    cfg.seed = 1
    with pytest.warns(UserWarning, match="1/2 operating points failed"):
        rec = cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                     writer=store.create("fdt", cfg, name="chi_diverged"))
    assert rec.body["points"] == {"param": "s", "planned": 2, "done": 1, "failed": 1}, rec.body["points"]
    errors = _point_errors(rec.data_path)
    assert list(errors) == ["000"] and "diverged" in errors["000"], errors


def _stub_resonances(monkeypatch, *, fail_s=()):
    """Both campaigns of a sweep stubbed by one law, with a resonance that MOVES with S -- ``1 + 2 S``,
    the spectrum's peak, which the detector takes -- and a ratio ``1 + S + 0.5 log(omega / (1 + 2 S))``,
    so every operating point has a ratio at its own resonance of its own. A driven campaign at an S in
    ``fail_s`` raises: a failed point."""
    from core.FDT import cross_validation as cv

    freqs = torch.linspace(0.05, 70.0, 1400, dtype=torch.float64)

    def res(cfg_op):
        return 1.0 + 2.0 * float(cfg_op.params_dict["s"][0])

    def _c1(cfg_op):
        return freqs, torch.exp(-((freqs - res(cfg_op)) ** 2) / 0.02) + 1e-6

    def _c2(cfg_op, omegas, freqs_psd, G):
        s = float(cfg_op.params_dict["s"][0])
        if any(abs(s - f) < 1e-9 for f in fail_s):
            raise RuntimeError(f"stub driven-campaign failure at S = {s:g}")
        om = omegas.to(torch.float64)
        return torch.ones(om.shape, dtype=torch.complex128), 1.0 + s + 0.5 * torch.log(om / res(cfg_op))

    monkeypatch.setattr(cv, "run_campaign1_psd", _c1)
    monkeypatch.setattr(cv, "_detect_resonance",
                        lambda omegas, G, w0: (float(omegas[int(torch.argmax(G))]), True))
    monkeypatch.setattr(cv, "_campaign2_ratio", _c2)
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param",
                        lambda *a, save_path=None, **k: Path(save_path).write_bytes(b"\x89PNG"))


def test_a_sweeps_results_are_per_point_and_its_offgrid_counts_only_what_was_measured(store,
                                                                                       monkeypatch):
    """The whole-piece review's M2 (C2 + S3), departing from P78/A6. A finished sweep's ``results`` was
    the single-cell summary run over every landed point's curve CONCATENATED: its
    ``ratio_at_resonance`` was then the FIRST landed point's ratio at the sweep's LARGEST resonance --
    no operating point's ratio at its own resonance -- and its peak did not say which point produced
    it. ``offgrid.of`` counted every PLANNED point's slots, the failed points' never-measured ones
    included, beside blanks and a usable fraction taken over the landed points only.

    Now: one entry per landed point, each with its OWN ratio at its own resonance (what
    ``load_param_sweep``'s row gives at the probe nearest that row's resonance); the top-level
    ``ratio_at_resonance`` null, because a sweep has no one resonance; the peak names its operating
    point; and ``of`` counts the landed points' probes. The nested list must survive the manifest's
    validation and print legibly, one line per field of each point."""
    import numpy as np
    import pytest
    from core import cli, config
    from core.artifacts.report import render_manifest
    from core.FDT import cross_validation as cv

    _stub_resonances(monkeypatch)
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")

    def sweep(name, s_spec):
        cfg, s_grid, _t = cli.make_param_sweep_config(
            cell, preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
            s_spec=s_spec, t_spec=(1.0, 1.1, 2), n_freqs=8, ensemble_M=2, seed=1)
        return cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                      writer=store.create("fdt", cfg, name=name))

    rec = sweep("resonances", (0.1, 0.5, 3))             # S = 0.1, 0.3, 0.5: resonances 1.2, 1.6, 2.0
    res = rec.body["results"]
    rows = cv.load_param_sweep(rec.data_path)
    assert res["ratio_at_resonance"] is None, "a sweep has no one resonance to read a ratio at"
    assert [p["param_value"] for p in res["points"]] == pytest.approx([0.1, 0.3, 0.5])
    own = []
    for p, row in zip(res["points"], rows):
        k = int(np.argmin(np.abs(row["omega_grid"] - row["omega_0_resonance"])))
        own.append(float(row["T_eff_over_T"][k]))
        assert p["omega_0_resonance"] == row["omega_0_resonance"] and p["is_resonant"] is True, p
    assert [p["ratio_at_resonance"] for p in res["points"]] == own, (res["points"], own)
    assert len(set(own)) == 3, f"three points, three resonances, three ratios at them: {own}"
    peaks = [float(np.nanmax(row["T_eff_over_T"])) for row in rows]
    best = int(np.argmax(peaks))
    assert res["peak_ratio"] == peaks[best] and res["peak_param_value"] == rows[best]["param_value"], res
    grid = rows[0]["omega_grid"].size
    assert rec.body["offgrid"] == {"blanks": 0, "of": 3 * grid}, "every point landed: all their probes"

    text = render_manifest(store.get("fdt", rec.id))
    assert "    points:\n      [0]\n" in text and "        ratio_at_resonance: " in text, text
    assert f"    peak_param_value: {rows[best]['param_value']!r}\n" in text, text

    # one point failed: `of` counts the two points that were measured, not the failed one's slots
    _stub_resonances(monkeypatch, fail_s=(0.3,))
    with pytest.warns(UserWarning, match="1/3 operating points failed"):
        part = sweep("one_failed", (0.1, 0.5, 3))
    assert part.body["offgrid"] == {"blanks": 0, "of": 2 * grid}, part.body["offgrid"]
    assert [p["param_value"] for p in part.body["results"]["points"]] == pytest.approx([0.1, 0.5])


def test_a_mid_run_refresh_refused_by_a_file_lock_is_warned_and_the_sweep_goes_on(store, monkeypatch):
    """The whole-piece review's N4 (H2). A refresh is bookkeeping -- each point's numbers are already
    flushed to data.h5 before it runs -- yet one refused by the OS ended the sweep: Windows fails a
    write to a file another program holds without write sharing (Explorer's preview pane on log.txt,
    a scanner), and the refresh after every operating point let that PermissionError out, so the
    rest of an overnight sweep never ran. A mid-run refresh now warns once and returns; the next
    refresh, or the commit (which stays strict), writes what it could not."""
    import pytest
    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch)
    cfg, s_grid, _t = _thin_study_cfg()
    cfg.seed = 1
    w = store.create("fdt", cfg, name="locked")
    real, calls = w._write_log, {"n": 0}

    def _write_log():
        calls["n"] += 1
        if calls["n"] == 2:                 # the refresh after the first operating point
            raise PermissionError(13, "The process cannot access the file", str(w.dir / "log.txt"))
        real()

    w._write_log = _write_log
    with pytest.warns(UserWarning) as said:
        rec = cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)
    msgs = [str(x.message) for x in said]
    assert len(msgs) == 1 and "could not refresh" in msgs[0] and "PermissionError" in msgs[0], msgs
    assert rec.body["complete"] is True and rec.body["points"]["done"] == 2, rec.body["points"]


def test_an_empty_ratio_is_a_counted_failure_not_a_numpy_crash(store, monkeypatch, caplog):
    """The nanmax defect (§4.3). ``log.info(f"... {np.nanmax(ratio.cpu().numpy()):.3g}")`` sits
    OUTSIDE the try that guards Campaign 2, so a point whose ratio comes back EMPTY raises
    ``ValueError: zero-size array to reduction operation fmax which has no identity`` from numpy --
    after the first phase's whole cost has been paid, and with a traceback rather than a count. Inside
    the try it is one logged, counted, recorded failure like any other."""
    import logging
    import pytest
    import torch
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch)
    monkeypatch.setattr(cv, "_campaign2_ratio",
                        lambda c, om, f, g: (torch.zeros(0, dtype=torch.complex128),
                                             torch.zeros(0, dtype=torch.float64)))
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    cfg.seed = 1
    with caplog.at_level(logging.INFO, logger="core"):
        with pytest.warns(UserWarning, match="2/2 operating points failed"):
            with pytest.raises(Refusal) as e:
                cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                       writer=store.create("fdt", cfg, name="empty"))
    assert e.value.field == "s_grid", "every point failed, so the ending is the all-failed refusal"
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 2 and all("Campaign 2 FAILED" in m for m in errors), errors


def test_the_dead_single_point_helper_is_gone():
    """``_fdt_measure`` has no caller anywhere in the tree, and its ``cfg.omega_0 = omega_0_emp`` is
    the only write on a caller's settings object left in this module -- exactly the V1 defect the rest
    of the piece removes, sitting in code nothing runs. Dead code that models the wrong thing is worse
    than dead code."""
    from core.FDT import cross_validation as cv
    assert not hasattr(cv, "_fdt_measure")


def test_an_all_failed_first_sweep_does_not_cost_the_second(store, monkeypatch, caplog):
    """P77 and spec §4.3/§8.2. Before piece 5 an all-failed S sweep raised out of the study before the
    T sweep had started. Now the S record stays on disk, unfinished (E2), the study says so at error,
    and the T sweep runs and finishes. _sweep_stubs' Campaign-2 counter is shared by both sweeps:
    calls 0-1 are the S grid's. Two trajectories is below the trust threshold, so the study warns
    once at its top; that and the S sweep's count warning are asserted, never leaked."""
    import logging
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import PreflightWarning

    _sweep_stubs(monkeypatch, phase_b_fail=(0, 1))
    cfg, s_grid, t_grid = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    writers = {"s": store.create("fdt", cfg, name="s_half"),
               "temp": store.create("fdt", cfg, name="t_half")}
    with caplog.at_level(logging.INFO, logger="core"):
        with pytest.warns(UserWarning) as said:
            recs = cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=3)

    assert sorted((w.category.__name__, " ".join(str(w.message).split()[:4])) for w in said) == [
        ("PreflightWarning", "The ensemble is 2"),
        ("UserWarning", "s sweep: 2/2 operating")], [str(w.message) for w in said]
    assert [r.name for r in recs] == ["t_half"], "only the sweep that measured something returns"
    rows = {s.name: s for s in store.list("fdt")}
    assert rows["t_half"].finished, "the temperature sweep ran and finished"
    assert rows["s_half"].complete and not rows["s_half"].finished, "the S record stays, unfinished"
    assert rows["s_half"].points_failed == 2
    errors = [r.getMessage() for r in caplog.records
              if r.levelname == "ERROR" and "measured nothing" in r.getMessage()]
    assert len(errors) == 1 and errors[0].startswith("The s sweep measured nothing"), errors


def test_a_study_whose_two_sweeps_both_measured_nothing_refuses(store, monkeypatch, caplog):
    """P77's other half: only when BOTH sweeps measured nothing does the study refuse -- with the
    activity sweep's refusal, after logging both at error (ruling F14/F45) -- and both unfinished
    records stay on disk."""
    import logging
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch, phase_b_fail=(0, 1, 2, 3))
    cfg, s_grid, t_grid = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    writers = {"s": store.create("fdt", cfg, name="s_none"),
               "temp": store.create("fdt", cfg, name="t_none")}
    with caplog.at_level(logging.INFO, logger="core"):
        with pytest.warns(UserWarning) as said:
            with pytest.raises(Refusal) as e:
                cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=3)
    assert e.value.field == "s_grid"
    assert sorted((w.category.__name__, " ".join(str(w.message).split()[:4])) for w in said) == [
        ("PreflightWarning", "The ensemble is 2"),
        ("UserWarning", "s sweep: 2/2 operating"),
        ("UserWarning", "temp sweep: 2/2 operating")], [str(w.message) for w in said]
    rows = store.list("fdt")
    assert len(rows) == 2 and all(r.complete and not r.finished for r in rows)
    errors = [r.getMessage() for r in caplog.records
              if r.levelname == "ERROR" and "measured nothing" in r.getMessage()]
    assert [m.split(";")[0] for m in errors] == ["The s sweep measured nothing",
                                                 "The temp sweep measured nothing"], errors


def test_a_sweep_point_is_reproducible_from_the_seed_and_its_index(store, monkeypatch):
    """P82 / spec §4.1: every operating point draws from a stream derived from (the study's seed, the
    sweep, the phase, the point's index) -- ``cross_validation._point_seed`` -- so a point is
    reproducible from the seed and its index, and depends on nothing else.

    What the derivation replaced (Task 19's review): ``seed + k`` for Phase A and ``seed + n + k`` for
    Phase B. Point k of the S sweep and point k of the T sweep drew the SAME Phase-A stream -- the two
    records' noise was correlated, which a comparison of sweeps would read as signal -- one sweep's
    Phase B could land on the other's Phase A when the grids differ in length, and Phase B depended on
    how many points the grid held. Each of the three is asserted here."""
    import pytest
    import torch
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import PreflightWarning
    from core.rng import seeded                  # never core.diagnostics.rng: that loads the SBI stack

    _sweep_stubs(monkeypatch)
    c1, c2 = cv.run_campaign1_psd, cv._campaign2_ratio
    drawn_a, drawn_b = [], []

    def _c1(cfg_op):
        drawn_a.append(float(torch.rand(())))
        return c1(cfg_op)

    def _c2(cfg_op, omegas, freqs_psd, G):
        drawn_b.append(float(torch.rand(())))
        return c2(cfg_op, omegas, freqs_psd, G)

    monkeypatch.setattr(cv, "run_campaign1_psd", _c1)
    monkeypatch.setattr(cv, "_campaign2_ratio", _c2)

    def config_of(n):
        return cli.make_param_sweep_config(
            str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
            preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
            s_spec=(0.0, 0.1 * (n - 1), n), t_spec=(1.0, 1.0 + 0.1 * (n - 1), n), n_freqs=3,
            ensemble_M=2)

    def sweep(n):
        drawn_a.clear(), drawn_b.clear()
        cfg, s_grid, _t = config_of(n)
        cfg.seed = 40
        cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=store.create("fdt", cfg))
        return list(drawn_a), list(drawn_b)

    def alone(seed):
        with seeded(seed, torch.device("cpu")):
            return float(torch.rand(()))

    a3, b3 = sweep(3)
    assert a3 == [alone(cv._point_seed(40, "s", 0, k)) for k in range(3)]
    assert b3 == [alone(cv._point_seed(40, "s", 1, k)) for k in range(3)]
    a2, b2 = sweep(2)
    assert (a2, b2) == (a3[:2], b3[:2]), "point k draws the same whatever the grid's length"

    # One study: its S and T sweeps share the seed, and no stream of one is a stream of the other.
    drawn_a.clear(), drawn_b.clear()
    cfg, s_grid, t_grid = config_of(3)
    writers = {"s": store.create("fdt", cfg), "temp": store.create("fdt", cfg)}
    with pytest.warns(PreflightWarning, match="The ensemble is 2 trajectories"):
        cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=40)
    assert (drawn_a[:3], drawn_b[:3]) == (a3, b3), "the study's S sweep is the lone sweep above"
    assert drawn_a[3:] == [alone(cv._point_seed(40, "temp", 0, k)) for k in range(3)]
    assert drawn_b[3:] == [alone(cv._point_seed(40, "temp", 1, k)) for k in range(3)]
    assert len(set(drawn_a + drawn_b)) == 12, "twelve streams in one study, no two alike"


def test_a_malformed_study_call_is_refused_before_the_first_sweep_spends(store, monkeypatch):
    """Task 19's review. A study called with the wrong writers -- one missing, one extra, one writer
    handed in for both sweeps, or one already entered -- used to find out only when the sweep that
    needed it reached it: the T sweep's KeyError, or its FileExistsError, arrived after the whole S
    sweep had been paid for. So did a sweep parameter the plot tables have no label for, at the END of
    that sweep. Each is a programming error, not an operator's (no front end can build one), so it is
    a plain ValueError -- not a Refusal, which would tell the operator to change a setting -- raised
    before anything is spent and before the thin-setting notice."""
    import warnings

    import pytest
    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch)
    ran = []
    monkeypatch.setattr(cv, "run_fdt_param_sweep", lambda *a, **k: ran.append(k.get("sweep_param")))
    cfg, s_grid, t_grid = _thin_study_cfg()

    def study(writers):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(ValueError) as e:
                cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=1)
        assert not rec, [str(w.message) for w in rec]
        return str(e.value)

    s, t = store.create("fdt", cfg), store.create("fdt", cfg)
    assert "missing ['temp']" in study({"s": s})
    assert "unexpected ['k']" in study({"s": s, "temp": t, "k": store.create("fdt", cfg)})
    assert "distinct" in study({"s": s, "temp": s})
    entered = store.create("fdt", cfg, name="entered")
    with entered:
        assert "entered" in study({"s": s, "temp": entered})
    monkeypatch.delitem(cv._PARAM_SYMBOL, "temp")
    assert "_PARAM_SYMBOL" in study({"s": s, "temp": t})

    assert ran == [], "a sweep started"
    assert not s.dir.exists() and not t.dir.exists()


def test_a_negative_seed_is_refused_before_the_sweep_opens_its_record(store, monkeypatch):
    """``_point_seed`` derives every stream through numpy's SeedSequence, whose domain is the
    non-negative integers. Both builders refuse a negative seed; a sweep handed one directly is
    refused by the same rule, before its writer is entered. Inside the per-point guard it would fail
    EVERY point and end in a false "measured nothing" refusal naming the grid -- the wrong setting.

    The study refuses the same seed at its top, before its thin-setting notice (fix round 1, F10): a
    refused study must not first warn about how far to trust a result it will never produce."""
    import warnings

    import pytest
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch)
    cfg, s_grid, t_grid = _thin_study_cfg()
    cfg.seed = -1
    w = store.create("fdt", cfg)
    with pytest.raises(Refusal) as e:
        cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)
    assert e.value.field == "seed", e.value.field
    assert not w.dir.exists() and store.list("fdt") == [], "refused before anything was spent"

    writers = {"s": store.create("fdt", cfg), "temp": store.create("fdt", cfg)}
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        with pytest.raises(Refusal) as e:
            cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=-1)
    assert e.value.field == "seed", e.value.field
    assert [str(x.message) for x in said] == [], "the quick-look notice came before the refusal"
    assert not any(x.dir.exists() for x in writers.values()) and store.list("fdt") == []


def test_a_seed_above_the_generators_ceiling_is_refused_by_both_builders_and_the_study(
        store, monkeypatch):
    """Task 28's ruling on ``--seed``, which takes any integer. A seed above 2**64 - 1 overflows the
    generator ``seeded`` hands it to -- "Overflow when unpacking long long", a bare ValueError raised
    INSIDE the run, after its record is open -- so the one seed rule both builders, the study and
    each sweep apply has a ceiling as well as its floor, refused under the same field key before
    anything is spent.

    The ceiling itself is legal and comes back EXACTLY: a rule that compared in float would round
    2**64 - 1 up to 2**64 and refuse the largest seed the generator takes. The sentence is pinned
    whole because the one existing rule with an upper end renders both ends of this range as
    "1.84467e+19"."""
    import warnings

    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal
    from core.rng import SEED_MAX

    assert SEED_MAX == 2 ** 64 - 1
    # refuse()'s shape: the caller's sentence with its own period, then the default clause
    sentence = ("The random seed must be at most 18446744073709551615, the largest the random number "
                "generator accepts; got 18446744073709551616. (default none: one is drawn and "
                "recorded)")
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    single = dict(n_freqs=4, ensemble_M=8)
    with pytest.raises(Refusal) as e:
        cli.make_fdt_config("NADROWSKI", True, cell, **single, seed=SEED_MAX + 1)
    assert e.value.field == "seed" and str(e.value) == sentence, str(e.value)
    top = cli.make_fdt_config("NADROWSKI", True, cell, **single, seed=SEED_MAX).seed
    assert top == SEED_MAX and isinstance(top, int), top

    sweep = dict(preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
                 s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2))
    with pytest.raises(Refusal) as e:
        cli.make_param_sweep_config(cell, **sweep, seed=SEED_MAX + 1)
    assert e.value.field == "seed" and str(e.value) == sentence, str(e.value)
    top = cli.make_param_sweep_config(cell, **sweep, seed=SEED_MAX)[0].seed
    assert top == SEED_MAX and isinstance(top, int), top

    # The study, handed the seed directly (it overrides cfg.seed, P12): refused before its
    # thin-setting notice and before either record is opened, as the negative seed is above.
    _sweep_stubs(monkeypatch)
    cfg, s_grid, t_grid = _thin_study_cfg()
    writers = {"s": store.create("fdt", cfg), "temp": store.create("fdt", cfg)}
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        with pytest.raises(Refusal) as e:
            cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers,
                                   seed=SEED_MAX + 1)
    assert e.value.field == "seed" and str(e.value) == sentence, str(e.value)
    assert [str(x.message) for x in said] == [], "the quick-look notice came before the refusal"
    assert not any(x.dir.exists() for x in writers.values()) and store.list("fdt") == []

    # ...and a sweep called on its own, which reads the seed off cfg.seed
    cfg.seed = SEED_MAX + 1
    w = store.create("fdt", cfg)
    with pytest.raises(Refusal) as e:
        cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)
    assert e.value.field == "seed" and str(e.value) == sentence, str(e.value)
    assert not w.dir.exists() and store.list("fdt") == [], "refused before anything was spent"


def test_a_sweeps_offgrid_counts_only_the_probes_its_spectra_could_not_supply(store, monkeypatch):
    """Fix round 1 (Important). ``offgrid.blanks`` counts the probe frequencies the SPONTANEOUS
    spectrum could not supply (spec §2.3, E9) -- exactly what a single-cell record counts
    (``torch.isnan(G_at_omegas)`` in fdt_pipeline), and the comparisons read the field from both study
    types. Counting every NaN in a landed ratio booked a NaN chi'' from the DRIVEN campaign -- a solver
    blow-up at one probe -- as off-grid, so it read as a band problem.

    Here the spectrum stops at 3.0, so the common grid's probes above it are off-grid at both points,
    and point 0's driven campaign also returns a NaN chi'' at one probe INSIDE the band: that probe
    must add nothing. Neither point fails, so nothing warns."""
    import warnings

    import h5py
    from core import cli, config
    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch)
    monkeypatch.setattr(cv, "run_campaign1_psd",
                        lambda c: (torch.linspace(0.1, 3.0, 8, dtype=torch.float64),
                                   torch.ones(8, dtype=torch.float64)))
    driven = []

    def _c2(cfg_op, omegas, freqs_psd, G):
        w = omegas.to(torch.float64)
        in_band = (w >= 0.1) & (w <= 3.0)             # blank off-grid, as the real ratio is
        ratio = torch.where(in_band, torch.full_like(w, 2.0), torch.full_like(w, math.nan))
        chis = torch.full(omegas.shape, 1 + 1j, dtype=torch.complex128)
        if not driven:                                # point 0: a NaN chi'' inside the band
            chis[1], ratio[1] = complex(1.0, math.nan), math.nan
        driven.append(1)
        return chis, ratio

    monkeypatch.setattr(cv, "_campaign2_ratio", _c2)
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=6, ensemble_M=2)
    cfg.seed = 1
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        rec = cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                     writer=store.create("fdt", cfg, name="offgrid"))
    assert [str(x.message) for x in said] == []

    with h5py.File(rec.data_path, "r") as h5:
        grid = h5["operating_points/000/omega_grid"][...]
    off = int(((grid < 0.1) | (grid > 3.0)).sum())
    assert off > 0 and 0.1 <= grid[1] <= 3.0, grid
    assert rec.body["points"]["done"] == 2
    assert rec.body["offgrid"] == {"blanks": 2 * off, "of": 2 * grid.size}, \
        f"{off} off-grid probe(s) at each point, none for the NaN chi'': {rec.body['offgrid']}"
    assert rec.body["results"]["offgrid_blanks"] == 2 * off


def test_an_all_blank_point_lands_without_a_numpy_warning(store, monkeypatch, caplog):
    """Fix round 1. The per-point peak line took ``np.nanmax`` over the whole ratio, and on a ratio
    with no finite value numpy warns ``RuntimeWarning: All-NaN slice encountered`` -- which reached
    the run log and stderr as if the operator had something to act on. The peak is taken over the
    FINITE values only: a point with none still LANDS, its line says there is no peak, and an EMPTY
    ratio is still a counted failure (test_an_empty_ratio_is_a_counted_failure_not_a_numpy_crash).

    The susceptibility is FINITE here: a ratio with no finite value beside a measured chi is what a
    spectrum that supplied no probe gives. A chi with no finite value is a diverged driven campaign,
    a failed point since the whole-piece review's M1
    (test_a_sweep_point_whose_driven_simulation_diverged_is_a_failed_point)."""
    import logging
    import warnings

    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch)
    monkeypatch.setattr(cv, "_campaign2_ratio",
                        lambda c, om, f, g: (torch.full(om.shape, 1 + 1j, dtype=torch.complex128),
                                             torch.full(om.shape, math.nan, dtype=torch.float64)))
    cfg, s_grid, _t = _thin_study_cfg()
    cfg.seed = 1
    with caplog.at_level(logging.INFO, logger="core"):
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter("always")
            rec = cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                         writer=store.create("fdt", cfg, name="blank"))
    assert [f"{x.category.__name__}: {x.message}" for x in said] == []
    assert rec.body["points"] == {"param": "s", "planned": 2, "done": 2, "failed": 0}
    assert rec.body["results"]["peak_ratio"] is None, "no finite value, so no peak (and no NaN)"
    peaks = [r.getMessage() for r in caplog.records if "T_eff/T peak" in r.getMessage()]
    assert peaks == ["      T_eff/T peak = none (no finite value on the grid)"] * 2, peaks


def test_a_refusal_that_is_not_measured_nothing_ends_the_study_at_once(store, monkeypatch, caplog):
    """Task 20's departure from A7, pinned (fix round 1). ``StoreError``, ``ManifestError`` and
    ``FDTModelError`` are all Refusals, so catching every Refusal from a sweep would log a store
    failure as "the s sweep measured nothing; its unfinished record is kept" -- false -- and then
    spend the whole T sweep. Only the all-failed refusal carries its grid's field; any other refusal
    ends the study at once and the T writer is never entered."""
    import logging

    import pytest
    from core.artifacts.store import StoreError
    from core.FDT import cross_validation as cv
    from core.refusals import PreflightWarning

    _sweep_stubs(monkeypatch)
    cfg, s_grid, t_grid = _thin_study_cfg()
    writers = {"s": store.create("fdt", cfg, name="s_store"),
               "temp": store.create("fdt", cfg, name="t_store")}

    def _refresh():
        raise StoreError("stub: the manifest could not be written", field="artifact")

    writers["s"].refresh = _refresh                  # the first per-point refresh of the S sweep
    with caplog.at_level(logging.INFO, logger="core"):
        with pytest.warns(PreflightWarning, match="The ensemble is 2 trajectories"):
            with pytest.raises(StoreError, match="stub: the manifest"):
                cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=1)
    assert not [r.getMessage() for r in caplog.records if "measured nothing" in r.getMessage()]
    assert not writers["temp"].dir.exists(), "the T sweep was entered after the S sweep's store failure"


def test_a_finished_sweep_draws_its_real_figure_into_its_record(store, monkeypatch):
    """Task 19's review. Every other sweep test stubs ``plot_fdt_3d_vs_param`` with a lambda that
    swallows any keyword, so a misspelled keyword at the sweep's one call -- or a figure path the
    writer never handed out -- would pass them all and fail only at the end of a real sweep, hours in.
    Here the REAL drawing function runs (Agg, the root conftest's backend) on the stubbed campaigns'
    numbers, and the file the manifest lists is on disk and is a PNG."""
    from core.FDT import cross_validation as cv

    real = cv.plot_fdt_3d_vs_param
    _sweep_stubs(monkeypatch)
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param", real)
    cfg, s_grid, _t = _thin_study_cfg()
    cfg.seed = 1
    rec = cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                 writer=store.create("fdt", cfg, name="drawn"))

    assert rec.body["complete"] is True
    (figure,) = rec.manifest.figures
    assert (rec.path / figure).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", figure
