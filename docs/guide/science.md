# The science behind the settings

Checked against commit 010011b.

## Nondimensionalisation

Every model's drift and noise are written in non-dimensional form. The parameters in a bounds
file's non-dimensional section (`k`, `lam`, `f_max`, `tau` and the rest for Nadrowski) carry no
units, and neither does the simulated trajectory. Physics enters only through the rescale
parameters of the dimensional section:

- `x_scale` turns the model's displacement into the cell's length unit (nanometres for Nadrowski);
- `t_scale` turns the model's time into the cell's time unit (milliseconds);
- `f_scale` turns a force in the cell's unit into the model's.

They are inferred with the other parameters, not derived from them. The one exception is the tier-1
box, where `f_scale` is derived from the temperature
([The tier-1 constraint and temperature](#the-tier-1-constraint-and-temperature)).

**Nothing nondimensionalises automatically.** A value in a bounds or cell file is used as written:
a non-dimensional one as non-dimensional, a dimensional one in the unit the model's units file
declares. The units file declares what the numbers mean and is never used to convert them
([Input files](recordings.md#input-files)). A new model or cell has to be written in this form by
hand.

**Frequency is inverse cell time, by construction.** The forcing evaluates sin(2π·freq·t) with t
in cell time units, and the summary statistics and the chi lock-in index their spectra with a time
step in cell time units. So a frequency is cycles per cell time unit everywhere in the simulator,
the statistics and chi. A drive given in hertz is converted through the time unit alone
(`core.sim_config.SimConfig.freq_si_to_cell`, 0.001 for a millisecond cell). It is never converted
by matching the units file's frequency token: against a millisecond cell that match resolves to 1,
and every drive would be a thousand times too fast. `SimConfig.check_unit_consistency` warns when
the declared frequency token is not the reciprocal of the declared time token. How to give a
recording's frequencies is in
[Recording files and drive frequencies](recordings.md#recording-files-and-drive-frequencies).

**The integration step comes from a prior bound.** The solver's finest non-dimensional step is
`SimConfig.dt_nd_min`, the camera frame interval divided by the upper bound of `t_scale`'s box:
1 ms / 40 = 0.025 on the master boxes. Each recorded sample then takes
(frame interval / t_scale) / dt_nd_min fine steps: 40 for a batch whose `t_scale` is 1, one at the
top of the box.

- This is a sampling constraint wearing an integration constraint's clothes. Widening `t_scale`'s
  box from 40 to 80 would silently double every simulation's cost and change its accuracy, with no
  change to any physics.
- Nothing has established what step the non-dimensional dynamics actually need.
- The obvious fix, a step chosen per batch from that batch's own `t_scale`, is wrong. It would make
  the integration error a deterministic function of `t_scale`, an inferred parameter, so the
  network could learn to read `t_scale` off a numerical artefact that is present in training and
  absent from real data.
- The defensible fix is a convergence study: halve the step until each of the 41 features, taken
  one at a time and measured in units of its own noise, stops moving. That gives a required step
  independent of `t_scale`. It has not been done
  ([Open questions](#open-questions)).

## Conditioning features

The flow never sees a trace. It conditions on one row per observation, built by
`core.SBI.statistics.conditioning_rows`. That function is the one constructor of the layout:
training, the simulated observation, the posterior predictive check and the experimental builders
all call it, so a change cannot reach one of them and miss another. The row is

`[ 41 summary features | 8 valid flags | log T_obs | forcing or chi block ]`

The features and the flags are `core.SBI.statistics.SUMMARY_WIDTH` = 49 columns. With log T_obs
they feed the conditioning network's summary pathway; the last block feeds its second pathway.

**The 41 features** are `core.SBI.statistics.FEATURE_LABELS`, computed by
`core.SBI.statistics.SummaryStatistics`. Groups A to F describe the passive trace. Group G is a
lock-in to the known drive on the driven trace. Positive quantities are logged, phases are
(cos, sin) pairs, and only `A1_mean` keeps the trace's mean; every other feature is computed on the
demeaned trace. The width never changes: a feature that is undefined for a trace is given a
sentinel value, which is what the valid flags are for.

| group | describes | features |
|---|---|---|
| A | dimensional anchors | `A1_mean`, `A2_log_var`, `A3_log_fpeak`, `A4_log_acf_decay` |
| B | spectral shape | `B1_log_Q`, `B2_log_peak_floor`, `B3_log_centroid_ratio`, `B4_spec_entropy`, `B5_pow_frac_peakband`, `B6_low_freq_frac`, `B6_log_rolloff_ratio`, `B7_log_sec_freq_ratio`, `B7_sec_height_ratio` |
| C | amplitude envelope | `C1_log_norm_env`, `C2_log_env_cv`, `C3_env_onset`, `C4_mode_ratio`, `C5_env_skew`, `C6_log_slowenv_corrtime`, `C7_log_slowenv_relvar` |
| D | marginal distribution | `D1_excess_kurt`, `D2_skew`, `D3_bimodality` |
| E | several timescales, waveform | `E1_log_tau_fast`, `E1_log_tau_slow`, `E1_w_fast`, `E2_log_h2`, `E2_log_h3` |
| F | phase-amplitude coupling | `F1_noniso_slope`, `F2_corr_omega_A2` |
| G | forced response | `G1_log_gain`, `G2_cos`, `G2_sin`, `G3_log_h2_ratio`, `G4_log_h3_ratio`, `G5_cos`, `G5_sin`, `G6_plv`, `G7_log_amp`, `G7_orient_cos`, `G7_orient_sin` |

Group G is all zeros in spontaneous mode, which has no drive, and in chi mode, where the chi block
replaces it.

**The three modes and their widths.** The width of the last block is
`core.orchestrator.expected_forcing_dim`:

| mode | last block | conditioning width | inferred parameters on the master boxes |
|---|---|---|---|
| spontaneous | none | 49 + 1 = 50 | 12, on `master_spont.txt` |
| forced | the drive, one column per forcing parameter (4 on `master.txt`) | 49 + 1 + 4 = 54 | 13, on `master.txt` |
| chi | the probe set: 6 channels in each of `CHI_K_PAD` = 12 slots, 72 columns | 49 + 1 + 72 = 122 | 13, on `master.txt` |

- A spontaneous bounds file leaves out `f_scale`. It only ever divides a force, this mode builds
  none, and so its marginal would just return the prior.
- The chi width depends on the number of slots, never on the number of probes. That is what lets
  one posterior read an observation with any probe count up to 12.
- Which mode to choose, and what each needs at the bench, is in
  [Observation modes](recordings.md#observation-modes).

**Why eight valid flags.** For several features the sentinel is far from small: log(1e-12) ≈ −27.6,
five decades below any real value. Without a flag the flow cannot tell "the peak has no measurable
width" from "the peak is extremely sharp", and the sentinel mass drags the channel's scale with it.
The flags (`core.SBI.statistics.VALID_FLAG_LABELS`, derived by
`core.SBI.statistics.derive_valid_flags`) are 1 where the feature is a measurement and 0 where it
was substituted, the same convention as the chi mask. There are eight because a census over the
10.24 million rows of an earlier training cache found substitution in six single channels and in
two pairs of columns, not in `B1_log_Q` alone, the one case first suspected:

| flag | covers | substituted in the census |
|---|---|---|
| `V_B1_Q` | `B1_log_Q` | 69.7 % |
| `V_C7_slowenv` | `C7_log_slowenv_relvar` | 30.0 % |
| `V_B7_secondary` | `B7_log_sec_freq_ratio` and `B7_sec_height_ratio` | 14.5 % |
| `V_E1_tau_slow` | `E1_log_tau_slow` | 8.0 % |
| `V_E1_tau_fast` | `E1_log_tau_fast` and `E1_w_fast` | 7.5 % |
| `V_E2_h3` | `E2_log_h3` | 5.5 % |
| `V_C6_slowenv_tau` | `C6_log_slowenv_corrtime` | 5.1 % |
| `V_E2_h2` | `E2_log_h2` | 2.8 % |

- **One flag for each pair**, because the pairing was checked rather than argued: on all
  10.24 million rows the two members of each pair were never one substituted and one real.
- **The fast pair's flag keys on `E1_w_fast`** equal to 1, meaning no fast component is left to fit.
  `E1_log_tau_fast`'s value also carries a lag index and is only incidentally constant; the two
  agreed on every row, and `E1_w_fast` stays the right test if the lag clamp ever moves.
  `E1_log_tau_slow`, whose sentinel is log(1e6 × dt), has its own flag, `V_E1_tau_slow`.
- **Not flagged, each by measurement:** `C2_log_env_cv` (predicted at 5.6 %, measured at 0 rows);
  `A4_log_acf_decay`, whose point masses are the discreteness of an integer lag, not a
  substitution; `B6_log_rolloff_ratio`, at 0.2 %, under the 1 % threshold.
- **A comparison bug that reads as good news.** The first census cast the stored float32 rows to
  float64 before comparing them with log(1e-12). The two values are not equal, so every count came
  back 0: a clean table saying the problem did not exist. `derive_valid_flags` compares in the
  data's own type.

**The flags never reach the Fisher.** The Fisher rotation
(`core.SBI.decorrelate.build_latent_fisher_rotation`) divides each feature's derivative by its
ensemble spread, floored at 1e-9. A flag is constant almost everywhere, but one that steps between
the two arms of a central difference divides a step of 1 by that floor and writes about 1e9 into the
matrix that defines the flow's coordinates. So the rotation, and the degeneracy map
(`identifiability jacobian`), build over the statistics without their flags
(`core.SBI.summaries.gen_stats_features`). A flag says which features are real: a statement about
the observation, with no gradient in the parameters.

**Rank-Gaussianisation, in chi mode.** Under chi the conditioning network standardises its own
inputs (next paragraph). The summary pathway's 50 columns are rank-Gaussianised
(`core.SBI.embedded_network.EmbeddedNet.rank_gaussianize`): each column's empirical quantile
function, sampled at `core.config.RANK_GAUSS_KNOTS` = 1024 levels, mapped through the probit.

- It replaced a mean-and-standard-deviation standardisation that a handful of pathological traces
  had contaminated. Fitted on an earlier training run, `A1_mean`'s standard deviation came out at
  4.19e11 against a physical range of about 1e3, and `D3_bimodality`'s at 4.42e8 against (0, 1].
  Sweeping either channel across its whole physical range moved the embedding by about 1e-7,
  against about 1.4 for a healthy channel: an attenuation of about 1e7, below one float32 step. The
  flow could not see two of its own channels.
- The transform is monotone and invertible, so nothing is lost. Being invariant to any monotone
  change, it settles the log-or-linear question for every column at once. A sentinel point mass
  becomes a point mass at a known quantile, which the flow can key on, instead of a scale factor.
- 1024 levels put the finest quantile step near 0.1 %, well under the smallest flagged point mass
  (`E2_log_h2`, 2.8 %).
- Spontaneous and forced modes keep sbi's own per-column standardisation, which the winsorisation
  below protects.

**Chi trains with no per-column z-score.** sbi's default fits one affine map per column of the
conditioning vector. Over probe slots that breaks permutation invariance: two orderings of one
probe set would be scaled differently. And the mask column, nearly constant, becomes an amplifier of
about 1e7 under sbi's 1e-7 floor on a standard deviation. So under chi `core.SBI.train.train_nn`
turns sbi's standardisation off, and the network standardises itself: the probe encoder
(`core.SBI.chi_encoder.ChiSetEncoder`) per channel over live probes only, and the summary block by
rank as above.

**Per-column winsorisation, in every mode.** Before training, each summary column (the flags and
log T_obs included) is clipped to its own 0.1 % and 99.9 % percentiles (`core.config.WINSOR_PCT`,
applied by `core.SBI.summaries.winsorize_summary_block`), and a `[winsor]` line says how many
elements moved. It replaced a row filter that dropped any row holding a value
above 1e15. That filter threw a whole row away for one bad value, and the outliers it let through,
below its threshold, were what dragged `A1_mean`'s fitted standard deviation to 4.19e11; clipping
each column at its own percentiles removes the leverage without removing a row. The chi block is
never clipped, for the reason given under the chi block below.

**Pathological trajectories are counted as they are simulated**
(`core.SBI.summaries.count_pathological`): non-finite, exactly constant, or over 1e15 in magnitude.
A `[patho]` warning names each batch in which new ones appear, since they cluster in particular
(`t_scale`, T_obs) strata, and a `[patho]` run total closes the generation. Read it before trusting
a flag histogram: the statistics map a non-finite value to 0, so a non-finite trace's flags read as
valid.

**The chi block.** Each slot is six channels: u = log(f/Ω₀), log|χ|, cos, sin, logcyc = log(f × T)
with T the duration actually locked in over, and a mask. The probe's frequency is carried in u, not
implied by its slot, and the encoder is permutation-invariant, so both the number of probes and
their placement are free. Two rules hold throughout (`core.SBI.chi.pack_probe_block`):

- **A dead slot is exactly 0.0** in all six channels, whether a pad or a masked probe. It is then
  bitwise inert, and no finite-value filter drops its row. This is also why winsorisation never
  touches the chi block: the magnitude and phase columns are dense over live probes, so their
  0.1 % percentile is not zero, and clipping would move every pad off 0.0 and turn it into a
  phantom probe.
- **A failed probe is masked, never a phantom.** An earlier packer turned a non-finite lock-in into
  a live-looking (0, 0, 0) triple, which cos² + sin² = 1 says no real probe can produce. The mask is
  the one verdict: simulated, finite, resolvable and in band.

Live probes are packed first, in ascending frequency, so the simulated and experimental paths
produce byte-comparable blocks.

**`CHI_K_PAD` is frozen into every artifact.** sbi bakes the conditioning shape into a saved
posterior, so the posterior's manifest records the slot count with the layout version
(`core.config.CHI_LAYOUT`), and the store refuses a mismatch when it loads one. Width alone cannot
identify a layout: 6 × 5 = 3 × 10 = 30, an exact collision with the retired layout of three
channels per probe.

## Chi probe design

A chi observation is one passive recording plus K single-tone driven recordings, the probes. Ω₀,
the peak frequency of the passive trace's own oscillation (`core.SBI.chi.peak_freq`), sets the
frame. Each probe sits at a multiple of Ω₀ and is locked in at its drive frequency, which gives one
complex χ.

- A simulated observation places `CHI_N_FREQS` = 6 probes on a log-spaced grid across the band
  (`core.SBI.chi.chi_multipliers`).
- Training draws its probe count per batch, from 2 to 12, and jitters the placement within strata
  of the band (`core.SBI.chi.sample_multipliers`). A fixed training grid would leave the network's
  frequency channel with only six values, and a bench recording at 0.07 × Ω₀ would be outside
  everything it was trained on.

Each value below was fixed by a measurement on the master cells, whose Ω₀ is about 23 Hz.
[The chi assumptions, and checking them](recordings.md#the-chi-assumptions-and-checking-them) says
what the values mean for a recording and how a lab checks them on its own cell.

**The band, (0.03, 0.3) × Ω₀** (`core.config.CHI_FREQ_BOUNDS`). Every probe sits between 0.03 and
0.3 times its own row's Ω₀, below resonance. The band was first fixed from a sweep of drive strength
against probe frequency at a single 5 s recording length. There |χ| was reproducible only below
about 0.25 × Ω₀, and at the higher multipliers it did not improve from 5 s to 25 s, which was read
as a frequency limit. A second sweep, with the recording length as a third axis and a re-lock of
the same traces over shorter prefixes, separated two different failures:

- **Inside the band, failure tracks drive cycles, not frequency.** Every in-band probe that failed
  at full length had been locked in over more than about 31 drive cycles, and every probe that
  survived over fewer. The boundary runs along constant multiplier × T_obs: 0.3 × fails at 5.2 s
  (35.5 cycles), 0.0646 × at 26.9 s (39.8 cycles). A fixed 5 s slice reaches that wall soonest at
  the high multipliers, which is why it looked like a frequency limit.
- **Re-locking recovers the in-band probes.** Locking the same traces in again over a prefix of at
  most 20 cycles restored every failing in-band point. At 0.139 × Ω₀ and 11.8 s, for example, the
  spread of |χ| between runs fell from 0.63 to 0.056 of its mean, and the driven lock-in rose from
  2.3 to 18 times the same lock-in on undriven runs. No new simulation was involved, only fewer
  samples in the sum, and a stationary, noise-limited lock-in cannot behave that way. So the
  bundle's response at fixed parameters wanders on the scale of tens of drive cycles, and a longer
  lock-in accumulates the wander instead of averaging it away. The mechanism is inferred, not
  established; phase diffusion of the free-running oscillation is the obvious candidate, and it has
  not been tested.
- **At and above resonance, no shortening helps.** Capped at 20 cycles, probes of the band's first
  setting still failed: at 0.5 × and 2 × Ω₀ the drive entrains the bundle, and at 1 × and 10 × the
  driven response is no larger than the bundle's own activity at that frequency (the driven lock-in
  was 0.12 times the undriven one at 1 ×). Entrainment belongs to the driven trace's spectrum, not
  to the lock-in window, so a cap cannot reach it.

So the band's interior was a duration problem, which the lock-in ceiling below fixes. Its high edge
is a different problem. With the ceiling in force, eleven multipliers from 0.03 to 0.6 × Ω₀ (32 runs
each, at five lengths) showed where the edge belongs:

- **Reproducibility and signal do not discriminate.** The |χ| spread stays at 0.06 to 0.10 of its
  mean throughout (0.16 at the low edge, a few-cycles effect), and the driven lock-in never falls
  below 9 times the undriven one. The ceiling solves reproducibility over twice the band's width.
- **Entrainment has a knee at 0.35 to 0.4 × Ω₀.** The share of the bundle's own peak power that
  survives the drive erodes gently up to 0.35 × (0.67 at 0.3 ×, 0.57 at 0.35 ×), then falls fast:
  0.44, 0.26 and 0.11 at 0.4, 0.5 and 0.6 ×, each step keeping only 77, 60 and 43 % of the last.
  That is a change of character, not a threshold crossing, so it belongs to the cell. It sets the
  band's edge at 0.3, below the knee with margin.
- **Phase scatter has no knee.** It grows smoothly, from 0.13 to 1.52 rad across the sweep, so
  wherever a threshold cuts it reports the threshold, not the cell. A 0.5 rad screen would have put
  the edge at 0.12 ×, a band 2.5 times narrower. Phase is therefore advisory:
  [probes band](command-line.md#probes-band) reports it and judges it only when given a threshold.

The judgement inside this is reasoning, not measurement. A probe whose phase scatters has lost its
cos and sin channels but keeps a reproducible magnitude (spread about 0.08, signal about 12), so it
is half useful; an entrained probe reports the drive back to itself, so it carries information about
the drive instead of the bundle. That asymmetry is why entrainment sets the edge and phase does not.
Settling it would take two posteriors, trained over (0.03, 0.12) and over (0.03, 0.3): two multi-day
runs, not yet made. The band's first setting, (0.1, 10), put eight of its ten probes above
0.25 × Ω₀, each locked in with no cycle ceiling; an earlier chi posterior trained over it calibrated
well and stayed at the prior, because the flow correctly learned that those probes carried nothing.

**The drive, `CHI_F0` = 0.15**, non-dimensional. Every probe is driven at this fixed
non-dimensional amplitude, a physical drive of `CHI_F0` × `f_scale`, so the lock-in's signal is the
same across the `f_scale` prior. χ, the response over the drive, cancels the amplitude in the
linear regime. But an oscillating bundle has no clean linear regime near its own frequency, so the
value was chosen by reproducibility, and it is bounded from both sides:

- too weak, and |χ| stops being reproducible: at 0.05 × Ω₀ the spread was 0.090 of the mean at a
  drive of 0.05, against 0.026 at 0.15;
- too strong, and the drive entrains the bundle, which abandons its own rhythm and follows the
  drive: onset at 0.2, the previous default, measured with the drive at 1.4 × Ω₀.

0.15 is the strongest drive that stays reproducible across the band and leaves the bundle running
free: its own peak kept at least 84 % of its undriven power at every probe from 0.05 to 0.2 × Ω₀.

**The cycle floor, `CHI_MIN_CYCLES` = 2.** A lock-in over less than two drive cycles returns the
demeaned trace's residual drift plus the bundle's own low-frequency content: finite, in range and
reproducible, which is how such probes once passed a reproducibility screen, but not a
susceptibility. Such a probe is masked, never moved and never dropped, and the count is reported.
(An earlier version clamped probes to the sampling limit, which silently relabelled a probe as a
frequency it was not driven at.) The floor is 2 and not higher because measurement refuses a higher
one: at 5 s the 0.03 × probe has 3.39 cycles and gave the best |χ| spread in the sweep, 0.024, so a
floor of 8 would delete the best probe in the experiment.

**The lock-in ceiling, `CHI_MAX_CYCLES` = 20.** Every instinct says a longer lock-in is a better
one, and past about 30 cycles on this model it is not. Re-locking the same traces over every prefix
length (48 runs, in-band probes only, so frequency cannot confound it) brackets the wall:

| ceiling (cycles) | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 |
|---|---|---|---|---|---|---|---|---|
| worst \|χ\| spread | 0.042 | 0.039 | 0.047 | 0.062 | 0.086 | 0.123 | 0.198 | 0.456 |
| worst driven / undriven | 18.8 | 22.0 | 21.9 | 18.3 | 15.9 | 12.8 | 7.9 | 3.6 |

- The wall is at 32 to 36 cycles, reached by a steady climb rather than a cliff, so there is no
  correct value, only a trade.
- 20 sits in the flat part: about three times inside the 0.2 spread screen, ten times the floor,
  and the ceiling the re-lock measurement had already validated on every failing in-band point.
- 12 to 16 reproduce slightly better and were not chosen. A shorter lock-in is also less
  frequency-selective, and no experiment here has priced that side of the trade.
- The ceiling is not a filter: nothing is masked or dropped by it; the segment is shortened. It
  applies per row, inside `core.SBI.chi_probes.gen_chi_raw`, where every caller goes through it:
  training, the Fisher rotation, the posterior predictive check and the experimental path. A
  ceiling applied in only one of them would have the network read an observable it was not trained
  on, silently.
- A posterior records its ceiling, because the ceiling sets the logcyc channel the encoder weighs a
  probe by. `SimConfig` refuses a ceiling at or under the floor, which would shorten every probe
  below two cycles and mask them all.
- **The experimental path truncates an over-long probe instead of refusing it**: the recording is
  sound, only its tail lies past the length training locked in over. What each probe verdict means
  at the bench is in
  [Recording files and drive frequencies](recordings.md#recording-files-and-drive-frequencies).

**The Ω₀ physics of masking.** An early end-to-end chi training run masked three quarters of its
probes. An audit that separated the masking predicates, over a real screened prior, found why:

- **Only the cycle floor masks.** The sampling limit, non-finite frequencies and the band filter
  masked nothing.
- **The driver is each row's own Ω₀.** The live fraction was flat across recording lengths and
  across the band's multipliers, and sharp in Ω₀:

  | Ω₀ (Hz) | under 1 | 1 to 3 | 3 to 10 | 10 to 30 | over 30 |
  |---|---|---|---|---|---|
  | live probes | 0 % | 0 % | 14 % | 69 % | 98 % |

- **The slow draws are real oscillators.** Their spectral peaks stand thousands of times above the
  median power, so they are not a peak finder's argmax landing on a featureless spectrum, and
  screening them out of the prior would be wrong. The mask is correct physics: the prior spans about
  four decades of Ω₀, and a band relative to Ω₀, with recordings of at most 60 s, reaches only its
  fast end.
- **About a third of rows keep no live probe** even after both fixes below. A row resolves no probe
  at any placement when Ω₀ × T_obs is under 2 / 0.3, about 6.7 cycles. Only a change to the prior
  can reach those rows: restricting its Ω₀ range is defensible if the slow draws lie outside the
  regime of interest, but it changes the question the posterior answers and would have to be
  declared with it.
- **Raising the longest recording was rejected.** Resolving the whole band on a 0.1 Hz bundle takes
  two cycles at 0.003 Hz, about 670 s: outside experimental reality, and it would dominate the
  simulation budget.

**Per-row placement** (`core.SBI.chi.resolvable_multipliers`, training only). Each row's multipliers
are lifted into the sub-band its own Ω₀ can resolve at the recording's length,
[max(0.03, 2 / (Ω₀ × T_obs)), 0.3], by an affine map in log frequency that keeps their order,
spacing and jitter. It costs nothing: the probe frequencies were always per row; only the
multipliers had been shared.

- **Placement is judged by frequency span, not by live count.** A rule tuned on the live count
  alone would park every probe on one frequency: perfectly resolved, and measuring no shape. On its
  own, placement raised the live fraction from 41 % to 46 % with a median live span of 5.0 × out
  of the band's 10 ×, and only 7.9 % of rows left with a single probe.
- **The rejected variant bounded placement by the ceiling as well as the floor.** It collapses the
  band: the band's two edges are a factor of 10 apart, and so are the ceiling and the floor
  (20 / 2), so satisfying both leaves a row one feasible multiplier for all but one value of
  Ω₀ × T_obs. The asymmetry decides it: under the floor a probe is masked outright, while over the
  ceiling it is only shortened.
- A slow row still ends up probed near the band's top, so it gains resolution but loses frequency
  spread. Placement turns inert rows into weakly informative ones, not into good ones.

**Per-row lock-in durations** (`core.SBI.chi.lock_in_batched` with `n_samples`). With one lock-in
duration per batch, keyed on its fastest row so that no row passed the ceiling, the slow rows that
placement had rescued were cut back under the floor. Each row is now integrated over its own prefix,
by masking the columns past it rather than slicing, so the batch stays rectangular.

| | shared placement and duration | per-row placement | and per-row durations |
|---|---|---|---|
| live probes | 41.1 % | 46.1 % | 64.2 % |
| rows with no live probe | 55.1 % | 47.9 % | 32.6 % |
| median live span | | 4.99 × | 5.38 × |
| rows with one live probe | | 7.9 % | 1.4 % |

Per-row durations took rows at 1 to 3 Hz from 4 % live (after placement) to 48 %, and rows at 0.3
to 1 Hz from none to 9 %. The span and the single-probe rows improved with the count, so this was
not a trade of shape for count. Two invariants had to survive the masking, and both fail silently:

- **The mean is taken over each row's own prefix.** A mean over the full width subtracts a level the
  row's samples never had, and the residual lands at zero frequency, where a sub-resonance lock-in
  is most sensitive.
- **The mask is applied after demeaning.** Zeroing first leaves minus the mean standing in every
  dead column: a step at the prefix boundary, again with its energy near zero frequency.

`test_lock_in_per_row_durations_match_locking_each_row_alone` pins both, against the one
unambiguous reference: each row locked in on its own.

**The masked fraction, and how precise it is.** Training's masked fraction settled at about 37 %:
36.8 % on the run that confirmed the two fixes, and a mean of about 36 % over seven runs on the
corrected band (31.7 % to 44.9 %). A smoke run's reading is judged against that, within ±12 points
([The card smoke gate](testing.md#the-card-smoke-gate)), and the band is that wide for a reason:

- every row in a batch shares one (`t_scale`, T_obs) draw and one probe set, so the effective
  sample size is the number of batches, not of probes;
- per-batch fractions ran from 13.5 % to 57.3 %, a standard deviation of 12.2 points over 12
  batches, not the 1.8 points a binomial count over one run's 704 probes would suggest;
- a smoke run has four batches, so its standard deviation is about 12.2 / √4 ≈ 6.1 points, and
  ±12 points is about two of them. Compare the mean of two or three runs against 37 %; one run
  inside the band says little.

On the GPU a single run's count can also move by one row's probes from process to process (260 and
254 of 704 have both been read with identical settings), which is one more reason the criterion is a
±12-point band and not an exact count. [probes mask](command-line.md#probes-mask) measures the same
fraction over any prior, with its batch count as the effective sample size.

The posterior predictive check at inference masks about 80 %, by design, because it reads the
posterior rather than the probe machinery. It drives every posterior sample at the observation's
own absolute frequencies, deliberately never re-placed, since moving the probes would simulate a
different experiment. A sample whose Ω₀ is far from the observation's cannot resolve those
frequencies, so the fraction reports how well the posterior constrains Ω₀: near the prior on a
smoke-sized run, and expected to fall on a real one.

**Re-measured on the GPU on 29 September 2026**, with the `probes` checks on the `master_spont`
cell:

- [probes band](command-line.md#probes-band): the configured band (0.03, 0.3) and the drive 0.15
  both hold, and the 0.6 × control is captured at every length (its own peak keeps 0.12 to 0.18 of
  its power).
- [probes drive](command-line.md#probes-drive), with the drive at 1.4 × Ω₀: Ω₀ 22.600 Hz. At the
  original own-peak window of 0.018 over the original sixteen strengths, the strongest free-running
  drive is 0.02 and the weakest captured is 0.2, as first measured. On the check's default grid and
  window it is free-running at 0.02 and captured from 0.15. That capture is read above resonance
  and moves with the window, which is why this check does not judge the chi drive and
  [probes band](command-line.md#probes-band) does.
- [probes mask](command-line.md#probes-mask), over a prior built at the smoke run's size: 753 of
  2,144 probes masked (35.1 %), the cycle floor the only cause: 31.2 % too slow even at the band's
  top, 4.0 % shortened below it by the draw of lock-in durations.

## What chi buys

Chi was designed to separate parameters that a passive trace and one forced trace see only in
combination: those two traces see two products, one setting the amplitude and one the timescale,
and the shape of χ(ω) over frequency was meant to separate `k`, `lam`, `x_scale` and `t_scale` one
by one. Measured with the degeneracy map (`identifiability jacobian`), part of that holds and part
does not.

**The measurement.** Made in August 2026 on the master cell at T_obs 4.5 s, with 32 runs per
perturbation arm, 128 for the noise floor and seed 0: forced mode on `master_weak` against chi mode
with six probes on `master_spont`. The two cells differ only in their drive, which chi ignores. A
unique-handle score is the fraction of a parameter's feature gradient that the other parameters'
gradients cannot explain: 1 alone, 0 fully aliased.

| parameter | forced, 4.5 s | chi, 4.5 s | forced, 1.0 s (control) |
|---|---|---|---|
| `k` | 0.040 | 0.091 | 0.076 |
| `x_scale` | 0.043 | 0.125 | 0.082 |
| `t_scale` | 0.219 | 0.447 | 0.206 |
| `lam` | 0.278 | 0.472 | 0.270 |
| `delta_E` | 0.091 | 0.102 | |
| `f_scale` | 0.871 | 0.695 | |

- **Every parameter's unique handle improves**, `f_scale`'s included, although its score falls. A
  score is a ratio, and a parameter with no gradient is trivially unique: forced mode barely
  measures `f_scale` at this weak drive (gradient norm 0.018), so its 0.871 is noise being unique.
  Under chi the norm is 3.789, 213 times larger, and `f_scale` becomes a real, partly correlated
  handle.
- **The phase channels carry `lam` and `f_scale`; the magnitude channel carries `t_scale`.** A sin
  channel is the top feature for `lam`, all five of `f_scale`'s top features are chi channels, and
  magnitude channels are three of `t_scale`'s top five, the top two above `A3_log_fpeak`.
- **`lam`~`t_scale` was never degenerate on this cell**: their gradients' |cos| is 0.59 in forced
  mode, before chi does anything.
- **`k`~`x_scale` survives chi**: |cos| 0.98 forced, 0.95 chi (0.97 when measured again), and the
  two keep the worst unique handles. Both are led by `A1_mean`: stiffness and displacement scale
  move the trace's mean together, and a sub-resonance susceptibility does not touch the mean. Three
  measurements agree. Chi will not break this pair, and no retraining will change that.
- **The control.** Forced mode's Group G lock-in has no cycle ceiling, so at 4.5 s it ran 142 drive
  cycles, far past the wall of about 31 ([Chi probe design](#chi-probe-design)). That inflates its
  noise and flatters chi. The 1.0 s column (31.6 cycles) is a control on that lock-in, not a chi
  comparison: there the forced `k` and `x_scale` scores roughly double, which shrinks chi's gain on
  them to +0.015 and +0.043. Chi helps `k` and `x_scale` a little, within the effect of recording
  length. The `t_scale`, `lam` and `f_scale` gains are untouched by the control, and they are chi's
  real product.
- **Compare only maps made at the same T_obs, and read a unique-handle score beside its gradient
  norm.** A scalar summary of the whole map does not measure chi: recording length alone moved the
  forced map's condition number further than chi did.

**The rotation is on under chi.** It was once switched off in chi mode, on the argument that chi
already attacks the degeneracy the rotation targets. The measurement above shows it does not, so
`core.orchestrator.build_posterior` rotates in every mode.

- Under chi the Fisher builds its Jacobian over 41 + 3K features: the statistics without their
  flags (Group G is then all zero, which costs nothing) and three channels per probe,
  `core.SBI.chi.CHI_FISHER_CHANNELS` = (logmag, cos, sin). K is the observation's probe count.
- The other three conditioning channels would each be an amplifier over the 1e-9 floor. u is fixed
  by the multiplier grid and varies only by rounding. The mask steps between the arms of a central
  difference. logcyc duplicates `A3_log_fpeak` exactly below the ceiling, and is a quantisation
  sawtooth where the ceiling binds.
- The chi simulations are seeded again immediately before the chi block, so the two arms of each
  central difference share their chi noise; without that the derivative is swamped and the rotation
  meaningless. `test_chi_fisher_rotation_builds_over_the_chi_feature_set` pins it.
- The cost is K + 1 simulations per Fisher evaluation instead of 2.

**Six probes into twelve slots is an untested lever.** An observation supplies `CHI_N_FREQS` = 6
probes into `CHI_K_PAD` = 12 slots, so half the chi block is padding before anything is masked,
while training draws its probe count from 2 to 12. Whether more probes, or a slot count matched to
what is supplied, would buy information has never been measured.

**Running the comparison.** `identifiability jacobian` measures the map at a cell's ground truth and
needs no posterior; its flags are in
[identifiability jacobian](command-line.md#identifiability-jacobian). Run it once per mode:

```bash
python -m core identifiability jacobian --no-chi --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_weak.txt --t-obs 4.5
python -m core identifiability jacobian --chi --chi-k 6 --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt --t-obs 4.5
```

- **Match T_obs across the pair.** The map is in signal-to-noise units, and the feature noise falls
  roughly as 1/√T, so a 1.0 s map against a 4.5 s one is not a chi-versus-forced comparison at all.
  Measure above T*, the length at which the band's low edge first clears the floor; 4.5 s is above
  it on the master cells ([Recording length](recordings.md#recording-length)).
- **The chi map is an unmasked upper bound.** It is built without the cycle-floor mask, which
  depends on Ω₀ and so on the parameters, and would put a step of 1 over a 1e-9 floor into the
  Jacobian. It therefore counts every probe, while training masks about a third of them.
- **The lock-in ceiling is applied**, so the probes are as long as training's; without it the map
  would measure a lock-in longer than the network ever sees.
- **Per-row placement is off.** It would make the placement depend on the parameters, the same
  defect as the mask; above T* it would change nothing anyway, and the run prints the threshold so
  this can be checked rather than trusted.

## The prior

To be written.

## The tier-1 constraint and temperature

To be written.

## Reading calibration honestly

To be written.

## Narrowing rounds

To be written.

## Identifiability limits

To be written.

## The August 2026 retrain

To be written.

## The solver's physics check

To be written.

## Open questions

To be written.
