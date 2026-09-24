"""FDT-for-user-models unit tests (FEATURE 1 v3 + B-d), Qt-free except for that one widget read.

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
builds two numeric widgets (offscreen) to read the value a blank box really produces.

Run:  pytest tests/test_fdt_user.py
"""
import math
import os
import sys
from pathlib import Path

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

    from core.FDT import fdt_pipeline
    from core.FDT.campaigns import FDTModelError
    from core.refusals import PreflightWarning

    spent = []

    def _never(*a, **kw):
        spent.append(a)
        raise RuntimeError("a campaign ran: the prefactor must be refused before anything is spent")

    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _never)
    monkeypatch.setattr(fdt_pipeline, "run_all_sanity", _never)

    class _NoN:
        """A Nadrowski cell carrying k -- so the omega_0 estimate would have succeeded -- but no n,
        which is half of the Nadrowski prefactor n*beta."""
        model = "NADROWSKI"
        params_dict = {"k": (1.0, None), "beta": (14.1, None)}
        # below both thin thresholds on purpose: the notice would fire if it ran before the prefactor
        n_freqs, ensemble_M = 1, 2

    for skip_sanity in (True, False):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(FDTModelError) as e:
                fdt_pipeline.run_fdt(_NoN(), skip_sanity=skip_sanity, confirm_production=True)
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

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

    # A spectrum resolving 1.0..3.0 whose peak is at 2.0, so the grid runs 0.2 .. 60 -- its lowest
    # probe sits a factor of five below the lowest frequency the spectrum resolves.
    freqs_psd = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 5.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
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
        fdt_pipeline.run_fdt(_Cfg(), skip_sanity=True, confirm_production=True)
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
    written = [p.name.rsplit("_", 2)[0] for p in tmp_path.glob("*.png")]
    assert written == ["psd"], ("the spontaneous spectrum's picture must be on disk when the band "
                                f"refusal fires -- it is what diagnoses it (P22): {written}")


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

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

    freqs_psd = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 5.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
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
        fdt_pipeline.run_fdt(_Cfg(), skip_sanity=True, confirm_production=True)
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
    not. T17 turns that directory into the record's unfinished folder; the ordering is what makes the
    promise keepable.

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

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

    # Resolves 0.1..0.3, peak at 0.2 -> the grid runs 1.0 .. 6.0: its LOWEST probe is above the
    # spectrum's highest bin, so T15's low-end gate passes and every probe is blank anyway.
    freqs_psd = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 5.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
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
        fdt_pipeline.run_fdt(_Cfg(), skip_sanity=True, confirm_production=True)
    msg = str(e.value)
    assert "0.1..0.3" in msg and "psd_T_obs_nd = 50" in msg, msg
    assert e.value.field == "freq_bounds", e.value.field
    assert driven == [], "Campaign 2 was entered before nothing-measurable was refused"

    written = sorted(p.name.rsplit("_", 2)[0] for p in tmp_path.glob("*.png"))
    assert written == ["psd", "spontaneous_trajectory"], written


class _ToneCfg:
    """The subset of FDTConfig run_fdt and the REAL Campaign 1 read, at a size that takes no time:
    2048 ND at dt_nd = 0.5 is 4096 samples, one Welch segment, well inside the cap. Paired with
    _use_tone_simulator, which replaces the simulator and nothing else."""
    model = "HOPF"
    params_dict = {"sigma_x": (0.1, None)}
    force_params_dict, state_dep_drift = {}, False
    n_freqs, ensemble_M, burn_in_nd, omega_0 = 5, 8, 0.0, 1.0

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
    """The kinds of figure a run left in ``directory``: each file name less its _<date>_<time>.png."""
    return sorted(p.name.rsplit("_", 2)[0] for p in directory.glob("*.png"))


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

    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
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
        fdt_pipeline.run_fdt(_ToneCfg((8.0, 30.0)), skip_sanity=True, confirm_production=True)
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
    fdt_pipeline.run_fdt(_ToneCfg((0.9 * high, 30.0)), skip_sanity=True, confirm_production=True)
    assert warned() == [f"4/5 probe frequencies have no value in the spontaneous spectrum, whose "
                        f"resolved band is {lo:g}..{hi:g} (ND), and are blank in the ratio."], warned()
    # ...and an upper multiplier below the named bound measures all of them
    caplog.clear()
    fdt_pipeline.run_fdt(_ToneCfg((high / 4, 0.99 * upper)), skip_sanity=True, confirm_production=True)
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
        monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda out=out: out)
        with pytest.raises(Refusal) as e:
            fdt_pipeline.run_fdt(_ToneCfg(band), skip_sanity=True, confirm_production=True)
        assert says in str(e.value), (name, str(e.value))
        assert e.value.field == "freq_bounds", (name, e.value.field)
        assert _figures_in(out) == ["psd", "spontaneous_trajectory"], (name, _figures_in(out))
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
        monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: out)
        with pytest.raises(Refusal) as e:
            fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True)
        return e.value, _figures_in(out)

    # the real Campaign 1, one trajectory sent to +inf; the real figures
    _use_tone_simulator(monkeypatch, diverge=True)
    e, written = refused(_ToneCfg((0.1, 30.0)), "one_member_inf")
    msg = str(e)
    assert e.field == "cell", e.field
    assert msg.startswith("The spontaneous simulation diverged: its spectrum holds no finite value"), msg
    assert "freq_bounds" in msg and "psd_T_obs_nd = 2048" in msg and "dt_nd = 0.5" in msg, msg
    assert "multiplier" not in msg and "resolve" not in msg, f"a band sentence about no spectrum: {msg}"
    assert written == ["spontaneous_trajectory"], written

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
    assert written == ["spontaneous_trajectory"], written
    assert driven == [], "a diverged run reached the driven campaign"

    # the boundary, with the figures stubbed: non-finite in SOME bins is not the cell's refusal
    for name in ("plot_spontaneous_trajectory", "plot_psd", "plot_eff_temp_ratio",
                 "plot_chi_components"):
        monkeypatch.setattr(fdt_pipeline, name, lambda *a, **kw: None)
    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    zeros4 = torch.zeros(4, dtype=torch.float64)
    campaign1(torch.tensor([9.0, 1.0, 5.0, float("nan")], dtype=torch.float64), zeros4)
    caplog.clear()
    fdt_pipeline.run_fdt(_ToneCfg((0.5, 1.2)), skip_sanity=True, confirm_production=True)
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

    from core.FDT import cross_validation as cv
    from core.FDT import fdt_pipeline, plots, sanity
    from core.orchestrator import PreflightWarning

    assert logging.getLogger("core").level == logging.INFO, "core/runs.py sets it at import"

    def records():
        return [(r.name, r.levelname, r.getMessage()) for r in caplog.records if r.name.startswith("core")]

    # ── the sweep study: information progress, one ERROR per failed point ─────────────────────────
    class _SweepCfg:
        model, n_freqs, ensemble_M, F0, psd_T_obs_nd = "STUB", 4, 2, 0.1, 1.0

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
    out_h5 = tmp_path / "s.h5"
    caplog.clear()
    # The stub's two trajectories are below the thin-setting threshold (E5), so the sweep raises that
    # notice first. It is asserted, not left to leak: pytest.warns re-emits every warning it did not
    # match when it closes, so the inner block alone would pass and still move the gate's count.
    with pytest.warns(PreflightWarning, match="The ensemble is 2 trajectories") as thin, \
            pytest.warns(UserWarning, match="1/2 operating points failed"):
        cv.run_fdt_param_sweep(_SweepCfg(), "s", np.array([0.0, 0.1]), {"temp": 1.0}, output_path=out_h5)
    assert {Path(w.filename).resolve() for w in thin if issubclass(w.category, PreflightWarning)} \
        == {Path(__file__).resolve()}, "the notice names the sweep's caller, not the sweep's own line"
    got = records()
    name = "core.FDT.cross_validation"
    assert (name, "INFO", "--- Phase A (s sweep): spontaneous PSD + omega_0 detection ---") in got, got
    assert (name, "INFO", "  [A 1/2] s=0 (temp=1.0)") in got, got
    assert (name, "INFO", "      omega_0 = 1.0000") in got, got
    assert (name, "INFO", "Common grid: 4 pts spanning [0.5000, 2.0000] (omega_0_ref=1.0000)") in got, got
    assert (name, "ERROR", "      Campaign 2 FAILED: stub out of memory") in got, got
    assert (name, "INFO", "      T_eff/T peak = 2") in got, got
    assert got[-1] == (name, "INFO", f"s sweep complete (1/2 points). Saved to: {out_h5}"), got
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

    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    monkeypatch.setattr(fdt_pipeline, "run_all_sanity",
                        lambda cfg, passive_plot_path=None: {"linearity": (False, {"ratio": 0.5})})
    caplog.clear()
    fdt_pipeline.run_fdt(_Hopf(), skip_sanity=False, confirm_production=False)
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

    # the run entries raise it themselves: a notice the operator never sees is not a notice
    import inspect
    for fn in (fdt_pipeline.run_fdt,):
        assert "warn_thin_settings" in inspect.getsource(fn), fn.__name__
    from core.FDT import cross_validation
    assert "warn_thin_settings" in inspect.getsource(cross_validation.run_fdt_param_sweep)


def test_the_thin_notice_is_the_one_judgement_class_and_names_the_stages_caller(tmp_path, monkeypatch):
    """Task 12, fix round 1. The class lives in the torch-free core.refusals so the FDT path can raise
    it without importing the SBI stack, and the orchestrator re-exports the SAME object, so every
    ``pytest.warns(orchestrator.PreflightWarning)`` in the tree still catches an FDT notice.

    The warning names whoever called the STAGE, as every other judgement does (orchestrator's
    _preflight_warn: stacklevel=3 through run boundaries). Pointing at the stage's own
    ``notices = warn_thin_settings(cfg)`` line told the operator nothing on the tool's stderr. The
    call goes through run_fdt here, because that is the frame layout the stacklevel is counted for:
    helper, stage, caller."""
    import pytest

    from core import orchestrator, refusals
    from core.FDT import fdt_pipeline

    assert orchestrator.PreflightWarning is refusals.PreflightWarning

    class _ThinHopf:
        model = "HOPF"
        # the prefactor (2/sigma_x^2) is resolved before the thin check, so this stub carries sigma_x
        params_dict = {"sigma_x": (0.1, None)}
        n_freqs, ensemble_M = 1, 2

    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    monkeypatch.setattr(fdt_pipeline, "run_all_sanity",
                        lambda cfg, passive_plot_path=None: {"linearity": (True, {"ratio": 1.0})})
    with pytest.warns(refusals.PreflightWarning) as rec:
        fdt_pipeline.run_fdt(_ThinHopf(), skip_sanity=False, confirm_production=False)
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
