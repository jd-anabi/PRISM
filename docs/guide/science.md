# The science behind the settings

Checked against commit 30ff8a3.

## Nondimensionalisation

Every model's drift and noise are written in non-dimensional form. The parameters in a bounds
file's non-dimensional section (`k`, `lam`, `f_max`, `tau` and the rest for Nadrowski) carry no
units, and neither does the simulated trajectory. Physics enters only through the rescale
parameters of the dimensional section:

- `x_scale` turns the model's displacement into the cell's length unit (nanometres for Nadrowski);
- `t_scale` turns the model's time into the cell's time unit (milliseconds);
- `f_scale` turns a force in the cell's unit into the model's.

They are inferred with the other parameters, not derived from them, with two exceptions:

- on the tier-1 box `f_scale` is derived from the temperature
  ([The tier-1 constraint and temperature](#the-tier-1-constraint-and-temperature));
- a bounds file that declares a drive but neither `f_scale` nor the temperature, as the Hopf
  model's does, gets `f_scale` = `x_scale` / `t_scale` (`core.forcing.build_nondim_force_tensor`),
  so its force unit is in effect a length over a time.

**Nothing nondimensionalises automatically.** A value in a bounds or cell file is used as written:
a non-dimensional one as non-dimensional, a dimensional one in the unit the model's units file
declares. The units file declares what the numbers mean and is never used to convert them
([Input files](recordings.md#input-files)). A new model or cell has to be written in this form by
hand.

**Frequency is inverse cell time, by construction.** The forcing evaluates sin(2π·freq·t) with t
in cell time units, and the summary statistics and the chi lock-in index their spectra with a time
step in cell time units. So a frequency is cycles per cell time unit everywhere in the simulator,
the statistics and chi. A drive given in hertz is converted through the time unit alone
(`core.sim_config.SimConfig.freq_si_to_cell`, 0.001 for a millisecond cell), which gives the right
factor whatever frequency token the units file declares, or none. It is never converted by matching
the frequency token: a millisecond cell whose units file declared Hz would resolve that match to 1,
and every drive would be a thousand times too fast. How the tokens must agree, and where a
disagreement is reported, is in [Input files](recordings.md#input-files); how to give a recording's
frequencies is in
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
inputs (next paragraph). Each of the summary pathway's 50 columns that varies in training is
rank-Gaussianised (`core.SBI.embedded_network.EmbeddedNet.rank_gaussianize`): the column's
empirical quantile function, sampled at `core.config.RANK_GAUSS_KNOTS` = 1024 levels, mapped through
the probit. A column constant in training, such as Group G's under chi, passes as 0.

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
applied by `core.SBI.summaries.winsorize_summary_block`), and, when any element moves, a `[winsor]`
line says how many. It replaced a row filter that dropped any row holding a value above 1e15. That
filter threw a whole row away for one bad value, and the outliers it let through, below its
threshold, were what dragged `A1_mean`'s fitted standard deviation to 4.19e11; clipping each column
at its own percentiles removes the leverage without removing a row. The chi block is never clipped,
for the reason given under the chi block below.

**Pathological trajectories are counted as they are simulated**
(`core.SBI.summaries.count_pathological`): non-finite, exactly constant, or over 1e15 in magnitude.
The passive trace is always counted, and the driven trace in forced mode; chi's probe traces are
not counted. A `[patho]` warning names each batch in which new ones appear, since they cluster in
particular (`t_scale`, T_obs) strata, and a `[patho]` run total closes the generation. Read it
before trusting a flag histogram: the statistics map a non-finite value to 0, so a non-finite
trace's flags read as valid.

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
- **From half of Ω₀ upward, no shortening helps.** Capped at 20 cycles, probes of the band's first
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
  drive. Measured with the drive at 1.4 × Ω₀, on a grid of strengths with no point between 0.1 and
  0.2, the onset fell between those two and was read as 0.2, the previous default.

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
  applies per row, inside `core.SBI.chi_probes.gen_chi_raw`, which training, the Fisher rotation,
  the posterior predictive check and a simulated observation all go through; the experimental path
  applies the same ceiling to a real recording through `core.SBI.chi.probe_verdict`, which truncates
  it. A ceiling applied in only one of them would have the network read an observable it was not
  trained on, silently.
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

  | Ω₀ (Hz) | under 0.3 | 0.3 to 1 | 1 to 3 | 3 to 10 | 10 to 30 | over 30 |
  |---|---|---|---|---|---|---|
  | live probes | 0 % | 0 % | 0 % | 14 % | 69 % | 98 % |
  | median peak power over median power | not measured | 6,202 | 2,378 | 273,350 | 143,863 | 7,516 |

- **Down to 0.3 Hz the slow draws are real oscillators.** Their spectral peaks stand thousands of
  times above the median power, so they are not a peak finder's argmax landing on a featureless
  spectrum, and screening them out of the prior would be wrong. The mask is correct physics: the
  prior spans about four decades of Ω₀, and a band relative to Ω₀, with recordings of at most 60 s,
  reaches only its fast end.
- **The slowest bucket is open.** Under 0.3 Hz, about 16 % of rows, the audit returned no
  measurable peak-to-median ratio, and those rows have not been examined, so whether a prior screen
  is right for them is an open question.
- **About a third of rows keep no live probe** even after both fixes below. A row resolves no probe
  at any placement when Ω₀ × T_obs is under 2 / 0.3, about 6.7 cycles. Only a change to the prior
  can reach those rows: restricting its Ω₀ range is defensible if the slow draws lie outside the
  regime of interest, but it changes the question the posterior answers and would have to be
  declared with it.
- **Raising the longest recording was rejected.** Resolving the whole band on a 0.1 Hz bundle takes
  two cycles at 0.003 Hz, about 670 s (a single probe at the band's top would need about 67 s, still
  past the longest recording, and would measure no frequency spread): outside experimental reality,
  and it
  would dominate the simulation budget.

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
  batches;
- a smoke run has four batches, so its standard deviation is about 12.2 / √4 ≈ 6.1 points, not the
  1.8 a binomial count over its 704 probes would suggest, and ±12 points is about two of them.
  Compare the mean of two or three runs against 37 %; one run inside the band says little.

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
  window it is free-running at 0.02 and captured from 0.15. The two readings agree at every
  strength both grids contain; the original grid has no point between 0.1 and 0.2, and at this
  detune 0.15 is already captured. Capture is read at 1.4 × Ω₀, above resonance, never at a probe's
  frequency, and its onset depends on that detune, which is why this check does not judge the chi
  drive and [probes band](command-line.md#probes-band) does.
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
- **`k`~`x_scale` survives chi**: |cos| 0.98 forced; under chi, 0.95 in an earlier map made at
  another band, probe count and recording length, and 0.97 in the one tabulated above. The two keep
  the worst unique handles. Both are led by `A1_mean`:
  stiffness and displacement scale move the trace's mean together, and a sub-resonance
  susceptibility does not touch the mean. Three measurements agree. Chi will not break this pair,
  and no retraining will change that.
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
  `core.SBI.chi.CHI_FISHER_CHANNELS` = (logmag, cos, sin). K is the run's configured probe
  count, `CHI_N_FREQS` = 6 by default.
- Each of the other three conditioning channels would corrupt the Jacobian. u is fixed by the
  multiplier grid and varies only by rounding, so dividing by its spread amplifies that rounding.
  The mask steps between the arms of a central difference, over the 1e-9 floor. logcyc duplicates
  `A3_log_fpeak` exactly below the ceiling, and is a quantisation sawtooth where the ceiling binds.
- The chi simulations are seeded again immediately before the chi block, so the two arms of each
  central difference share their chi noise; without that the derivative is swamped and the rotation
  meaningless. `test_chi_fisher_rotation_builds_over_the_chi_feature_set` pins it.
- The cost is K + 1 simulations per Fisher evaluation instead of 2.

**Six probes into twelve slots is an untested lever.** A simulated observation supplies
`CHI_N_FREQS` = 6 probes into `CHI_K_PAD` = 12 slots, so half the chi block is padding before
anything is masked, while training draws its probe count from 2 to 12. What would settle it is
under [Open questions](#open-questions), "Probes against slots".

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

The inferred prior is the product of two independent blocks, the non-dimensional parameters and the
rescale parameters (`core.orchestrator.ProductPrior`). A forced-mode drive has a third, separate
prior that training draws from and never infers (`core.orchestrator.build_forcing_prior`).

**The non-dimensional block is screened for stability** (`core.orchestrator.build_prior`, through
`core.SBI.Priors.prior.Prior.construct_prior`), in three stages:

1. **A census of the box.** Each round draws candidates from a scrambled Sobol sequence, uniformly
   over the linear box, and simulates them; a candidate is accepted when its trajectory stays
   finite. At the defaults on a CUDA card that is 50 rounds of 2,048 candidates.
2. **A flood-fill of the stable region.** Every accepted point seeds a random walk: a batch of normal
   steps with one fixed standard deviation (0.01 by default) in the parameters' own units, each
   simulated and accepted on the same test, every accepted step seeding more, until the accepted set
   passes its limit (175,000 by default) or nothing is left to walk from.
3. **A mixture fitted to the accepted set.** The points are mapped into the box's unbounded latent
   coordinate, HDBSCAN clusters them there, and a Gaussian mixture with one component per cluster is
   fitted. Pushed back through the box, it is a prior whose support is exactly the box.

What the screen does, read from the code:

- **Stable means finite.** The only test is that the simulated trajectory stays finite over the
  screen. A draw that diverges inside the screen is rejected; one that runs away more slowly is
  accepted. That is why the screen's duration defines what "stable" means.
- **The census screens half as long as the flood-fill.** The census integrates only the first half
  of the stability duration (`core.SBI.prior_screen.gen_prior` asks `construct_prior` for that), and
  the flood-fill integrates all of it. The census's accepted points go straight into the
  flood-fill's accepted set and are never screened again, so part of the fitted set passed only the
  half-length test. How large a part is not known: at the defaults on a CUDA card the census screens
  102,400 candidates against the flood-fill's limit of 175,000; the prior's record stores its
  accepted count as unknown, and the census's share is not recorded.
- **The walk is not confined to the box.** A step past a bound is simulated and accepted like any
  other if it stays finite. Before the fit such points are clamped just inside the box, at 10⁻⁶ of
  the range from the bound (`core.SBI.reparam.clamp_to_box`), which puts them near ±13.8 in the
  latent coordinate the mixture is fitted in. How many accepted points that affects has not been
  counted.
- **The clusters set only the component count.** HDBSCAN's labels are used for their number alone;
  the mixture is fitted to every accepted point, the ones HDBSCAN calls noise included.
- **For a built-in model, one accepted set gives one prior.** The mixture's fit starts from a fixed
  random state (`core.SBI.Priors.prior.GMM_RANDOM_STATE`), and the built-in models sort the accepted
  set first, so the fit is reproducible from its points; a user model's accepted set is not sorted.
  Nothing before the fit is seeded: the census's Sobol scramble, the walk's steps and a built-in
  model's random starting points draw from the global random streams, and `prior` seeds none of
  them, so two builds over one box differ. That is why a simulation cache names its prior by the
  mixture's fingerprint, not by its box.

The second and third points are an open question ([Open questions](#open-questions)).

**The rescale block is log-uniform for every name containing "scale"**
(`core.orchestrator.build_rescale_prior`). x_scale, t_scale and, on the master box, f_scale are
positive and span one to three decades, and a uniform prior would over-weight the high end.
Temperature, T, on the tier-1 box, is uniform.

**The box coordinate is linear.** `core.config.REPARAM_LOG_PARAMS` is empty, so for the built-in
models every parameter's box bijection, the coordinate the mixture is fitted in and the flow trains
in, is linear (a user model may declare its own log box). That coordinate is not how prior mass is
spread: the rescale block is sampled log-uniformly all the same.

- A geometric (log) box for f_scale was tried, to cure a mild tilt in its rank test, and the
  posterior trained under it was worse, in its coverage and in f_scale's own rank test.
- A geometric box on k, lam, x_scale and t_scale over-mixed those parameters in an earlier posterior.

So all stay linear. [Settings that are not speed dials](window.md#settings-that-are-not-speed-dials)
says which of the prior's settings change the prior rather than the time it takes, with the component
counts measured for two cluster sizes.

## The tier-1 constraint and temperature

**The master box samples one energy twice.** The gating-spring energy appears in the non-dimensional
block as n · beta (beta in units of k_B·T), and in the rescale block as f_scale · x_scale, a force
times a length. Nothing required the two to agree, and their ratio is an implied bath temperature.
Over the training rows of [the August 2026 retrain](#the-august-2026-retrain) it ran 14, 80, 286, 939
and 6,292 K at the 5th, 25th, 50th, 75th and 95th percentiles. The median is room temperature, so the
problem was spread, not bias: only about 2.27 % of the rows sat at a physical 280 to 310 K.

**Tier 1 derives the force scale from a temperature** (`core.SBI.derived`):

f_scale = n · beta · k_B · T / x_scale

- The bounds file declares `T in (280, 310)`, in kelvin, where the master box declares f_scale, and T
  is inferred. The prior stays a proper density over independent parameters; making f_scale a
  function of the other twelve would make it singular. The force scale is computed wherever
  parameters are split for simulation, into f_scale's own column, so everything downstream reads it
  unchanged (`core.SBI.derived.to_sim_rescale`).
- k_B is in the cell's own units, derived from the units file (`core.sim_config.SimConfig.k_b_cell`):
  k_B = 1.380649e-2 pN·nm/K for the nanometre and piconewton master cell, so k_B·T = 4.1419 pN·nm
  at 300 K, the figure to check a derived force scale against.
- **Derive f_scale, never beta.** beta is a drift parameter, and the stability screen runs on the
  non-dimensional block alone ([The prior](#the-prior)). Deriving beta would make that block depend
  on the rescale block and invalidate the screen. Deriving f_scale confines the change to the rescale
  block, and the model never sees T.

**The derived force scale is a new distribution, not a relabelling.** Over the box's corners it runs
from about 0.19 to about 12,840 pN (0.21 to 12,426 pN at 300 K), where `master.txt` declares f_scale
in (1, 1000). Every chi probe is driven at 0.15 times the force scale, so the physical drive a probe
needs moves with it. On the card run of 29 September 2026, over a prior built at the smoke run's
size, the derived scale's median was about 714 pN, against about 32 pN for the master box's
log-uniform f_scale.

- A training run on the box prints a `[tier1]` line before its first simulation: the derived force
  scale over the prior's draws at the 1st, 50th and 99th percentiles. In chi mode a second gives the
  drive amplitudes they imply ([train](command-line.md#train)).
- The lines report and never refuse. Whether a drive of that size is reasonable is a judgement about
  the preparation, not a threshold for the code.

**Temperature enters only through the force scale.**

- In spontaneous mode it has no effect: nothing is driven, and the force scale only ever divides a
  force.
- In forced mode it sets the non-dimensional amplitude of the cell's drive, the physical amplitude
  over the force scale, so the driven trace depends on it.
- In chi mode every probe is driven at a fixed non-dimensional amplitude, and χ is the
  redimensionalised response over the physical drive: x_scale / f_scale times the non-dimensional
  response (`core.SBI.chi_probes.gen_chi_raw`). So T divides every probe's |χ| by one common factor
  and does nothing else. Over its prior that factor moves log|χ| by at most log(310/280) ≈ 0.10.
- n and T enter the force scale only as their product n · T.

So in chi mode temperature may be only weakly informed, and it is nearly degenerate with n.

**Measured on the tier-1 box at smoke size**, on the card run of 29 September 2026 (four training
batches, a network of 256 hidden features × 10 transforms). These are indications at a tiny training
size, not the retrain's result:

- The degeneracy map at the tier-1 cell (`identifiability jacobian`, chi, 4.5 s, six probes) gives
  temperature a gradient norm of 0.214 per kelvin, in units of the feature noise, and a unique handle
  of 0.134. Its |cos| with n is 0.92, a near-degenerate pair, as the product n · T predicts; with
  t_scale it is 0.49, and with x_scale 0.44. Its strongest features are the chi log-magnitudes.
- The calibration verdict was FAIL, which is expected at that size. Temperature's own rank test
  passed there (KS p 0.479) and failed in the calibration of the narrowing round on the same card, a
  round capped at `--max-epochs 5`, which trains six epochs ([train](command-line.md#train));
  neither reading decides anything at this size.

The Jacobian column and the product n · T together say that temperature is weakly informed in chi
mode and nearly degenerate with n, so its estimate is not a measurement of the bath temperature.

**Reported as an assumed input.** On a box that declares T,
`core.sim_config.SimConfig.assumed_params` is `("T",)`, and every report marks temperature as an
assumed input ([What a run shows](window.md#what-a-run-shows) says where). It is still inferred, and
it stays in the calibration verdict and in the joint coverage test whether or not the data inform it;
[Reading calibration honestly](#reading-calibration-honestly) says why.

**Records carry the constraint.** Every record written from a tier-1 configuration has, in its
manifest's `config` block, a `"tier1"` entry (the relation, k_B in the cell's units and T's range)
and `"assumed_params"` (`core.artifacts.manifest.config_from_cfg`). A record therefore says what its
simulations were driven at without its bounds file.

**Tier 1 is opt-in by the box.** A run gets it only when its bounds file declares T in place of
f_scale (`core.SBI.derived.uses_derived_f_scale`); `master.txt`, the default master box, declares
f_scale.

- The tier-1 box has its own files, `Resources/Bounds/nadrowski/master_tier1.txt` and the cell
  `master_spont_tier1.txt` (T = 300), instead of an edited `master.txt`, so both boxes stay runnable
  side by side ([Input files](recordings.md#input-files)).
- A prior built on `master.txt` loads under `master_tier1.txt`. A prior is checked against the
  non-dimensional box alone: model, parameter set and order, box and log mask
  (`core.artifacts.store.ArtifactStore.load_prior`). The rescale block, T's uniform prior included,
  is rebuilt from the bounds file in use.
- The retrain uses the tier-1 box ([Decisions](retrain.md#decisions)).

**Beyond tier 1.** Tier 2 would constrain by the instrument: x_scale from bead or photodiode
calibration, and the sampling interval and recording length from the protocol. Tier 3 would
constrain by the population: the Ω₀ band a preparation actually produces, recorded as derived from
data. Both are open, and so is the alternative of fixing T at 300 K for a 12-dimensional box
([Open questions](#open-questions)).

## Reading calibration honestly

**The verdict** (`core.orchestrator.calibration_verdict`) is PASS when both of these hold:

- every inferred parameter's rank-uniformity test (SBC) reaches KS p ≥ 0.05 ÷ the number of inferred
  parameters, temperature included, which keeps one 5 % false-alarm rate across all of them;
- the joint coverage test (TARP) reaches KS p ≥ 0.05.

Three things follow:

- **A FAIL is a result, not an error.** The calibration is written and returned all the same.
- **Temperature is judged like every other parameter** (`calibration_verdict`). Its rank test is
  valid whether or not the data inform it, and a temperature the flow mis-models is a real flaw. It
  is marked, not excused.
- **Every verdict carries one caveat.** t_scale's rank test rests on the calibration's
  (t_scale, T_obs) operating points, 200 by default, not on its 2,000 datasets, because every
  dataset in a batch shares its batch's t_scale. It has less power than the others
  ([Settings that are not speed dials](window.md#settings-that-are-not-speed-dials)).

[validate](command-line.md#validate) lists what the verdict prints and records.

**Reading the tests.**

- **Read the KS p-values, not `c2st_ranks`.** The rank-test table prints both, and about 0.58 is
  c2st's finite-sample floor, so a c2st rank score near it says nothing.
- **Read the pooled rank histograms beside KS.** At 2,000 datasets KS flags a mild miscalibration
  reliably, where 1,000 did not; the histogram shows how severe it is, and a near-flat histogram
  under a low KS p means a mild one.
- **A flat SBC means "not overconfident", never "informative".** A posterior that returns the prior
  is flat by construction, because calibration is a property of the joint distribution, not of any
  one conditional. An earlier chi posterior was flat on 12 of its 13 parameters, with a TARP KS p of
  1.000, and had every non-dimensional marginal at the prior.
- **A TARP KS p of 1.000 is not "perfectly calibrated".** It sits at the ceiling partly because a
  wide, conservative posterior lands there, and the joint coverage test is less sensitive than the
  marginal rank tests. The honest claim is "no detectable overconfidence".

**The best-fit table is not a recovery measurement.** Inference draws a figure, "Best fit — summary
stats": the posterior draw whose simulated summary statistics lie closest to the observation's, with
its parameter values and an RMS z.

- `core.SBI.overlay.rank_by_stats` standardises each feature by its spread across the posterior's own
  draws, not by measurement noise. The score says the draw sits inside the predictive cloud, not that
  it is indistinguishable from the truth.
- It weights every live feature equally, so along a degenerate direction the best draw is close to a
  free draw. In one run n came back within 4.5 % of its truth while its marginal ramped to the top of
  its box.
- Trust the table only where a marginal is sharp; elsewhere, read where the truth falls in the
  marginal.

**Informativeness** (`core.SBI.analysis.informativeness`). Every test above measures calibration, and
a posterior that returns the prior passes them all. The number for whether a run learned anything is
the expected prior-to-posterior KL, estimated as the mean of log q(θ* | x) − log p(θ*) over the
calibration set just simulated, so it costs no simulation. `validate` prints it as the
`Informativeness` block and records it.

- **Read its sign first.** It is not bounded below by zero: a flow that gives the truth less density
  than the prior does scores negative. A smoke-sized train (`--max-epochs 5`, which trains six
  epochs) measured −23.1 nats, the right answer for it: worse than the prior.
- **Never compare it with a figure measured on training rows.** The flow has fitted those rows, so a
  figure there is optimistic by an unknown amount. Compare posteriors on fresh calibration sets only.
- **It is joint:** one log-density ratio over every parameter together, temperature included.
  Temperature's own line in the per-parameter breakdown is marked "(assumed input)", and the total is
  not adjusted.
- **The breakdown is not a set of KLs.** The per-parameter figures are entropy reductions estimated
  from draws, how much narrower each marginal got: a marginal that moves without narrowing scores 0,
  and a widened one scores below 0. The per-direction figures are in the Fisher eigenbasis, in the
  order `identifiability rotation` reports.
- For a narrowed posterior the total is still measured against the full prior, and is inflated by
  −log P(A) nats ([Narrowing rounds](#narrowing-rounds)).

**Pooled SBC over mixed probe counts.** In chi mode a calibration draws its probe count per batch,
over the mixture training saw, so its rank test pools across counts. A pooled test can be flat while
each count is miscalibrated in compensating directions.

- `sbc --chi-k-fixed` holds the count, one stratum at a time ([sbc](command-line.md#sbc)).
- `sbc`'s per-repeat verdict is the rank-uniformity half only; the joint coverage half runs in
  `validate`.
- Every repeat draws the same probe layout ([Open questions](#open-questions)).

**Repeating a calibration.** `validate --seed` repeats one. Every draw but the chi probe layout runs
on a stream derived from the seed and a fixed calibration tag (`core.rng.calibration_seed`), so the
stream never replays the one a training run seeded with the same number started on.
[validate](command-line.md#validate) says how exactly a repeat reproduces.

## Narrowing rounds

A narrowing round (TSNPE) draws a region around one observation along a posterior's Fisher
directions, trains a new posterior on the prior restricted to that region, and is valid only near
that observation ([tsnpe](command-line.md#tsnpe)).

**Draw from the truncated prior, never the posterior.** The posterior only says where to look.

- Proposing from a density fitted to the posterior multiplies the likelihood in again every round,
  which tempers. Credible intervals contract as (L+1)^−1/2 after L rounds with no new information: at
  L = 4 they are 2.2 times narrower than the data support.
- SBC comes out flat all the same, because it validates the flow against the proposal it trained on,
  so no diagnostic here would notice.
- A test pins the proposal instead: a wide prior, a narrow posterior and a region, and the proposal's
  width must be the prior's over the region, neither the posterior's nor the posterior's over √2
  (`test_the_proposal_is_the_TRUNCATED_PRIOR_and_not_the_posterior`).
- Restricted to a region, the proposal is the prior up to a constant, so the round needs no proposal
  correction.

**The rules, and why each holds.** [Narrowing-round safety
rules](rules-and-traps.md#narrowing-round-safety-rules) gives each rule, where the code enforces it
and the test that pins it. The reasoning:

- **The truncated-prior rule** (a round draws from the prior restricted to the region, never from the
  posterior): proposing from the posterior tempers, and only a direct test would catch it.
- **The observation-digest rule** (a round refuses unless the stored observation matches): a region
  deletes prior support for good, so it must be drawn around data that were recorded and are
  verifiably the same.
- **The narrowed-model rule** (a narrowed posterior is marked as such and never loads or infers as a
  broad one): its flow saw rows only inside its region, so anywhere else it extrapolates.
- **The eigenbasis rule** (the region is cut in the rotation's leading directions, flat ones left
  full width): a box in the physical parameters would cut the barely constrained ones on noise, and
  deleted support never comes back. In the Fisher eigenbasis the posterior is close to
  axis-aligned.
- **The unweighted-draws rule** (the region comes from unweighted posterior draws, not best fits):
  choosing the best fits applies a second, undeclared likelihood, with the discrepancy measure as a
  hidden setting.
- **The generous-region rule** (a 99.9 % region, with the truth-outside rate watched): a region too
  wide costs only simulations, one too tight deletes support for good, and how often the truth falls
  outside is the honest failure rate.
- **The cost-on-screen rule:** a round is a full simulation campaign, so its budget is on screen
  before it starts.
- **The region-carries-its-basis rule** (the child reuses the parent's rotation, never recomputes it,
  skips directions loaded on t_scale, and names its base prior): the rotation is not reproducible
  from one process to the next, so a box means nothing outside the rotation it was measured in.
- **The calibrate-on-the-region rule:** the flow converges to the posterior under the restricted
  prior, so calibration draws from that prior, and a flat result certifies calibration on the region
  only.

**The t_scale override turns a box into a reweighting.** Every training batch shares one
(t_scale, T_obs) pair from a stratified Sobol schedule, and overwrites each row's t_scale with it
after the draw (`core.SBI.pipeline.gen_training_data`).

- Along a direction that loads on t_scale, that carries rows out of the box. The proposal becomes the
  smooth reweighting p(θ) · P(A | θ₋ₜ) / P(A), θ₋ₜ being every parameter but t_scale, and NPE does
  not correct a reweighting.
- The box stays an exact restriction only when no truncated direction loads on t_scale, and its
  interval along a direction that is the t_scale axis itself is a no-op.
- So a direction whose |V[t_scale, j]| exceeds 1/√d, the root-mean-square entry of a random rotation
  (0.277 at d = 13), is left full width, and the next eligible direction is truncated instead
  (`core.SBI.truncate.t_scale_loading_max`). On the card run's tier-1 round the first direction
  loaded 0.960 on t_scale and was skipped.

**The region's mass has a bound, not an equality.** The region is a box of per-direction intervals at
level q over k directions, and its mass is not the joint HPD region's. By the union bound it misses
at most k(1 − q) of the posterior's mass, so it holds at least 1 − k(1 − q): 99.5 % for five
directions at 99.9 %.

**The region is generous.** The default level is 99.9 %, and a level below 0.99 raises a pre-flight
warning, never a refusal (`core.orchestrator.tsnpe_round`).

**Every round reports its truth, inside and outside.** A round drawn around an observation simulated
from a cell reports, for every truncated direction, where the truth lies against the interval
(`core.SBI.truncate.TruncationRegion.containment`), and records the answers in the new posterior's
`training.truth_containment` ([tsnpe](command-line.md#tsnpe) lists what the round records and warns).
Each round's answer is kept, and a rate across rounds is not. The round also prints how many of its
recorded training targets lie inside the region after the t_scale override, beside the fraction the
rejection sampler accepted before it.

**The −log P(A) inflation is printed, not corrected.** A narrowed posterior's informativeness is
still measured against the full prior, so its joint KL is inflated by −log P(A) nats. The
calibration prints the amount beside the kept fraction, and nothing subtracts it.

**Deliberately open:** pooling rows across rounds, the batch-by-scale t_scale override, and changes
to how the Fisher eigenbasis is built (`core.SBI.decorrelate.build_latent_fisher_rotation`,
`core.SBI.reparam.fisher_eigenbasis`). [Open questions](#open-questions) says what is unknown about
each.

## Identifiability limits

**Three parameters are seen only through noise.** In `core.Models.nadrowski_model.NadrowskiModel`,
n, temp (the adaptation motors' temperature relative to the bath, 1 being thermal) and tau_c appear
in no drift term.

- n enters all three noise amplitudes: the bundle row's √(2/(n·beta)), the adaptation row's
  √(2·temp/(n·beta·lam)) and the calcium row's √(2·tau_c·p·(1 − p)/n)/tau.
- temp enters only the adaptation row's, and tau_c only the calcium row's.
- The observable is state column 0, the bundle's displacement.

So the data see these three only as latent noise propagating into the displacement. No retraining,
feature repair or reparameterisation gives them a handle the observable lacks. In the August 2026
retrain they dominated the three least-constrained directions
([The August 2026 retrain](#the-august-2026-retrain)). What could reach them is a different
observation: replicate recordings for all three, and the thermal tail's level for n
([Open questions](#open-questions)). On the tier-1 box, in forced and chi mode, n gains a second
route, through the derived force scale, which it shares with the bath temperature T as the product
n · T ([The tier-1 constraint and temperature](#the-tier-1-constraint-and-temperature)).

**The Fisher analysis is measured on clean features.** The rotation, its eigenvalues and the
directions `identifiability rotation` decomposes come from a Jacobian standardised by feature noise
measured locally, at each operating point: each feature's spread over an ensemble of single
trajectories there, 192 runs at the default settings, floored at 1e-9
(`core.SBI.decorrelate.build_latent_fisher_rotation`). They do not use the training-set
standardisation that the conditioning repair replaced ([Conditioning features](#conditioning-features)).

- So the identifiability findings are not artifacts of that contamination: a gradient small against
  trajectory noise is small whatever the flow can see.
- The repair lets the flow see what the features carry. It cannot add information the observable
  never held.

**A clean loss plateau reads as a limit in the data.** Train and validation losses that track each
other and plateau well before the best epoch, the validation loss no longer descending as it nears
its best, say the wide marginals are an identifiability limit, not under-fitting
(`core.Helpers.visualizers.plot_training_loss`), as in the August 2026 retrain. The epochs between
the best one and the stop show no improvement by construction, so they are not the plateau to read.
Not for every axis either: for t_scale, T_obs and the probe design, whose effective sample size is
probably the batch count, a plateau may say nothing about convergence
([Open questions](#open-questions)).

## The August 2026 retrain

The last full chi retrain before the conditioning repair and the tier-1 box: 5,000 batches × 2,048
rows, 10.24 million simulations, finished on 23 August 2026. It ran on the master box with f_scale
inferred, at the chi band, drive and lock-in ceiling that [Chi probe design](#chi-probe-design)
settles, with six probes supplied into twelve slots. Its artifacts no longer exist, so none of this
can be re-run; the numbers stand as a record. A second run, 10,000 × 2,048 on the master box with the
repaired conditioning, completed on 29 August 2026 and was never characterised: no full-size
calibration or informativeness exists for it, and its artifacts are gone too.

- **Calibration was excellent.** The SBC rank histograms were flat on all 13 parameters, t_scale's
  on fewer independent values than the rest
  ([Reading calibration honestly](#reading-calibration-honestly)), and the rank CDFs and TARP sat on
  the diagonal.
- **Prediction was over-dispersed.** The posterior predictive check covered 99.1 % at a nominal 90 %
  (mean |z| 0.490), and its 95 % band was 2 to 3 times the observation's envelope. The intervals were
  honest and far too wide.
- **The loss ruled out under-fitting.** Train and validation loss tracked each other and plateaued
  from about epoch 90; the best validation loss came at epoch 110, and training stopped at 130 on its
  20-epoch patience. A clean plateau well before the best epoch points to an identifiability limit,
  not to too few epochs.
- **t_scale alone is the best-constrained direction**, with nothing else loading above 0.02.
  - k loads on the second and fifth directions (with s and f_max on the second, beta and s on the
    fifth), both at the well-constrained end of the ordering.
  - The three least-constrained directions are dominated by n, temp and tau_c
    ([Identifiability limits](#identifiability-limits)).
  - The eigenvalues were not stored, so this is an ordering and a set of loadings without a scale:
    whether the best direction is ten or a million times better constrained than the worst cannot be
    recovered. A rotated posterior now records them in its own record
    ([identifiability rotation](command-line.md#identifiability-rotation) says when they go
    unrecorded).
- **Eigenvectors are columns.** V holds one direction per column (w = z @ V), and
  `core.SBI.reparam.rotation_of` is the one place that convention is decoded. This run's rotation was
  first read from a copy saved transposed, which has the same shape and is just as orthogonal, so
  nothing flagged it, and each parameter's row was read as a direction. Only the correct orientation
  is given here. Loading now refuses a posterior whose recorded rotation is not the one inside its
  own training prior (`core.SBI.reparam.assert_rotation_consistent`).
- **Only 6 of the 12 chi slots were filled**, and after masking, the predictive check's zero-variance
  count came to about four live probes ([What chi buys](#what-chi-buys)).
- **Its "more capacity will not help" was measured on the broken conditioning.** The plateau was read
  to mean that more epochs, a larger flow or more simulations would not help. But that flow could not
  see two of its own channels ([Conditioning features](#conditioning-features)), and a larger network
  may use what the repair restored. The retrain therefore trains a larger network, 256 hidden features
  × 10 transforms ([Decisions](retrain.md#decisions)).

## The solver's physics check

The solver replays its Euler–Maruyama step loop from captured CUDA graphs by default
(`core.config.SOLVER_CUDA_GRAPHS`;
[The solver, CUDA graphs and reproducibility](architecture.md#the-solver-cuda-graphs-and-reproducibility)).
Measured in August 2026 on the card (an RTX 5070 Ti, a batch of 2,048, 100,000 steps), it ran
123,349 steps/s against 12,639 for the eager loop, about ten times faster. Two checks stand behind
trusting it with the physics:

- **A bitwise test, with the noise off.** `test_the_cuda_graph_step_matches_the_eager_step_bitwise`
  (gpu-marked) integrates a Nadrowski model with every noise channel zeroed and a time-varying drive,
  over two full graph chunks and a short eager tail, and requires the graphed and eager trajectories
  to be equal bit for bit. With noise live the two paths draw their random numbers in a different
  order and can only be compared statistically, so the bitwise test covers only a noise-zeroed model.
- **A statistical test, with the noise live.** In August 2026: graphs on against graphs off, 3,072
  rows per arm, both random streams seeded. Ω₀, the quantity that drives chi masking, was
  indistinguishable: a two-sample z of −0.01, and a KS D of 0.0020 against a 5 % critical value of
  0.0347. The traces' mean and standard deviation agreed to 0.05 % and 0.025 % of one standard
  deviation. It is the only evidence with noise live that the default solver did not change the
  simulated physics.

**Batch count against width.** The solver's time is set by its sequential steps, not by its rows, so
a batch's width is nearly free in simulation time
([The inference settings](window.md#the-inference-settings) gives the timings). The flow's fit is
not: its time grows with the rows. The two budget numbers are therefore not interchangeable.

- The batch count sets the diversity of (t_scale, T_obs): every row of a batch shares its batch's
  pair.
- The width is replication at that pair.
- So 5,000 × 2,048 and 10,000 × 1,024 hold the same 10.24 million rows and are different training
  sets: the second has twice the operating points, each simulated half as often. Both numbers are
  part of the simulation cache's identity.

## Open questions

Each is unknown today; each paragraph says what would settle it.

**M-replicate conditioning, first.** A posterior's width mixes a real degeneracy in the parameters
with the sampling noise of summary statistics computed from one realisation, and nothing here
separates the two. Simulating M independent passive traces for each parameter set, and conditioning
on the set with a permutation-invariant encoder like the probes' (`core.SBI.chi_encoder.ChiSetEncoder`),
would. Widths falling as M^−1/2 mean estimator noise, which longer or repeated recordings fix; widths
that saturate mean a real degeneracy, which only a new observable reaches. It comes before any
experiment design, or an experiment could be designed for a parameter whose width was never
degeneracy.

**The thermal tail.** Above the bundle's corner frequency its displacement spectrum tends to
2k_BT / (λω²), with λ the bundle's friction. In the model's parameters that is
2 · x_scale² / (n · beta · t_scale · ω²), and on the tier-1 box λ = f_scale · t_scale / x_scale. It is
an absolute reading of those scales, and no feature takes it: every Group B feature is a ratio or a
fraction, normalised by the spectrum's own frequencies or power, so the spectral block is blind to
the tail's level. The proposal is one new feature, the mean of log(ω² S_X(ω)) over about a decade
above the corner, with a valid flag for rows whose tail is unresolved. It pairs with tier 1, whose
constraint is what makes the prefactor absolute. What would settle it: add the feature and read its
handles in the degeneracy map; a change to the feature set forces a full re-simulation.

**The strata test.** Every row of a training batch shares one t_scale, T_obs and probe set, so along
those axes the effective sample size is probably the batch count (5,000 at the August 2026 retrain,
10,000 at the planned retrain), not the row count, which is 2,048 times larger. If so, a clean loss
plateau says nothing about whether those axes converged. The test: train 5,000 × 2,048 and
10,000 × 1,024, the same 10.24 million rows over 5,000 and over 10,000 operating points, and compare
t_scale's rank test and posterior width ([The solver's physics check](#the-solvers-physics-check)
has count against width). With today's tool each arm keys its own simulation cache, so the test
needs its own simulation, although both arms are subsets of a 10,000 × 2,048 cache's rows: a way to
train on part of a cache would run it without simulating.

**New observables: a step for k, intermodulation for the nonlinearity.** Only after the
M-replicate study.

- For k, a step or force-clamp transient. A transient never meets the wall at about 30 drive cycles
  ([Chi probe design](#chi-probe-design)), because that wall belongs to a steady-state lock-in.
- For the parameters that shape the model's nonlinearity (delta_E and beta in the channels' open
  probability, and f_max and s in the motor force and its calcium feedback, which set where on it the
  bundle sits), two-tone intermodulation at 2ω₁ − ω₂, with both drives inside the sub-resonance
  band. The measured frequency is not driven, which avoids entrainment, and the product's amplitude
  reads the nonlinearity directly.

Nothing in the tool measures either. Settling each means simulating the protocol and reading its
handles in the degeneracy map.

**Probes against slots.** A simulated observation supplies six probes into twelve slots
([What chi buys](#what-chi-buys)), and the retrain keeps both numbers. Whether more probes, or a slot
count matched to what is supplied, would buy information has never been measured. Two trainings that
differ only there, compared by informativeness, would settle it.

**Tiers 2 and 3, and fixing T.** Tier 2 would take x_scale from the instrument's calibration and the
sampling interval and recording length from the protocol; tier 3 would restrict the prior to the Ω₀
band a preparation actually produces, recorded as derived from data. Neither is built. Tier 2 needs
the lab's calibration of its displacement measurement (a bead or photodiode calibration) and its
recording protocol; tier 3 needs Ω₀ measured across a population of the preparation's cells. The
alternative to tier 1's inferred temperature is to fix T at 300 K and infer 12 parameters: a cleaner
reading, since T is weakly informed and nearly degenerate with n
([The tier-1 constraint and temperature](#the-tier-1-constraint-and-temperature)), and a larger
change. The retrain's own measurement would settle it: how much temperature's marginal narrows
against its prior (its line in the informativeness breakdown), and where the truth falls in it.

**The stability screen's two gaps.** The census accepts points on half the stability duration, and
they seed the flood-fill's accepted set without being screened again; the walk can also step outside
the box, and those points are fitted clamped at its edge ([The prior](#the-prior)). A science
question: should the census screen the full duration, or should its points be re-screened over the
full duration before they seed the fill? A count, on a prior built at full size, of how many fitted
points each affects would say whether it matters.

**Log-sampling the non-dimensional census.** The census draws uniformly in the linear box (the
`_global_map` of every prior, `core.SBI.Priors.user_prior.UserPrior._global_map` included). A user
model's per-parameter log box changes only the coordinate the mixture is fitted in, not where the
census puts its candidates, so it moves no prior mass. Making the census geometric would move mass
toward the low end of the wide parameters, and would have to move the built-in models too. It is a
decision about what the prior should mean, not a measurement.

**Pooling rows across rounds.** A narrowing round trains only on the rows it simulates: the region is
part of its cache's identity, so a round never resumes an amortized run's rows. Pooling rows across
rounds would cut a round's cost. It is left open because the rounds' rows come from different
proposals, and their mixture is proportional to the prior, with one constant, only inside every
round's region at once. Settling it takes that argument made for this code's regions and its t_scale
override, and a test with a known answer like the one that pins the truncated-prior rule.

**The batch-by-scale t_scale override.** Training draws one (t_scale, T_obs) pair per batch and
overwrites every row's t_scale with it ([Narrowing rounds](#narrowing-rounds)). It is why
t_scale-loaded directions cannot be truncated, and why t_scale sees only one value per batch. A
schedule that respected a round's region along those directions would let them be cut.
Settling it means changing how batches draw their t_scale, and showing, with the post-override
containment a round already prints, that the recorded training targets stay inside the box.

**Changes to the Fisher eigenbasis.** The rotation comes from a simulated Fisher averaged over eight
operating points drawn from unseeded random streams, so it is not reproducible from one process to
the next, and a narrowing round inherits its parent's ([Narrowing rounds](#narrowing-rounds)).
Whether to change how it is built (its operating points, its feature set, a seeded computation) is
left open, because any change moves the coordinates a trained flow, its eigenvalues and every region
drawn from it live in. The honest test of a change is to build the rotation both ways at several
operating points and compare the eigenbases.

**The step the dynamics need.** The solver's finest non-dimensional step comes from a prior bound,
not from the dynamics ([Nondimensionalisation](#nondimensionalisation)). The convergence study
described there would settle it: halve the step until each of the 41 features, in units of its own
noise, stops moving.

**Three chi measurements left open** ([Chi probe design](#chi-probe-design)):

- the band's top edge, set at 0.3 by reasoning about entrainment against phase scatter, which two
  posteriors trained over (0.03, 0.12) and over (0.03, 0.3) would settle;
- the rows under 0.3 Hz, about 16 % of the masking audit's, whose spectra have not been examined;
  looking at them decides whether a prior screen is right for them;
- the wander of the bundle's response over tens of drive cycles, inferred from re-locking the same
  traces, whose mechanism is untested; phase diffusion of the free-running oscillation is the
  candidate, and comparing how fast the undriven oscillation's phase scatter grows with how fast the
  lock-in wanders, over the same lengths, would test it.

**Clarity against the whole band.** [probes drive](command-line.md#probes-drive) counts the undriven
cell as an oscillation when the power of its spectrum's highest bin above zero frequency, over the
median power of the whole band, reaches the clarity threshold (`core.diagnostics.probes.probe_drive`).

- An overdamped, low-pass spectrum has its power at low frequency and its median far down the tail,
  so its highest bin can clear the threshold with no oscillation at all.
- The one guard that catches it is the window's reach: a drive less than one frequency bin from the
  own-peak window is not judged. That fires only when the maximum sits in the lowest bins (the lowest
  seven, at the default detune and window), so it catches a monotone low-pass spectrum, whose maximum
  is the lowest bin.
- A low-pass spectrum whose maximum sits above the lowest few bins may not be rejected, and its
  strengths are then judged as if it oscillated.

A clarity measured against a local baseline around the peak, or a test on a quiescent cell whose
spectrum has a mid-band hump, would settle it.

**A NaN in the degeneracy map.** In `identifiability jacobian`, a NaN in a measurable column of the
Jacobian makes the least-squares step raise after every simulation has been spent. The choice is
between excluding that column and zeroing that row.

**One probe layout for every repeat.** The probe generator is seeded with one fixed number on every
call that generates rows (`core.SBI.pipeline.gen_training_data`): training, a calibration and each
`sbc` repeat alike. So in chi mode every repeat draws the same probe design whatever `--seed` says,
and repeat-SBC cannot see probe-design variance. Seeding the probe draw from the run's seed would
let it.

**One-probe simulated observations.** The tool accepts `--chi-k 1`: the configuration's own check
allows one probe, while the window's Config tab refuses fewer than two, and training draws at least
two per batch (`core.config.CHI_K_MIN_TRAIN`). Training rows do end with a single live probe, since
half of each batch's rows keep only a random number of their live probes
(`core.SBI.chi_probes._subset_probe_rows`), and a one-probe simulated observation puts its probe at
the band's low edge. Whether such an observation is in distribution is not established.
`sbc --chi-k-fixed 1`, the rank test on one-probe calibration sets ([sbc](command-line.md#sbc)),
answers this only in part: a calibration set jitters its single probe across the band, while a
one-probe simulated observation always puts it at the band's low edge, one placement among many. If
such an
observation is not in distribution, a floor of 2 in the configuration's check would close it; the
bench path, which rightly takes a single recording, never re-runs that check.

**Truncating every direction.** A round may ask for as many directions as the posterior's latent
width, and is refused only above that. It then truncates every direction the t_scale rule allows,
the flat ones included, which is what the eigenbasis rule exists to prevent. Whether the largest
count should be lower is still to decide.

**Correcting the −log P(A) inflation.** A narrowed posterior's informativeness is inflated by
−log P(A), which is printed and never subtracted ([Narrowing rounds](#narrowing-rounds)); whether to
subtract it has not been decided.

**The FDT passive baseline's Ω₀.** The FDT analysis's passive-baseline check
(`core.FDT.sanity.check_passive_baseline`) sets s = 0 and temp = 1, which puts the bundle's
displacement and adaptation in thermal equilibrium. That process is reversible, so its spectrum has
no peak at a finite frequency, and the check's "resonance" is always the first bin of its search
band. The verdict stays meaningful, since 18 of its 20 probes still test the ratio, but the reported
Ω₀ misleads, and the other two, low probes that fall off the spectrum's grid, are simulated and then
dropped, at about 37 % of the check's 466 s. The options: anchor on the configured Ω₀, report "no
peak", clip the probes to the resolved span, or relabel the line.

**The FDT ratio at resonance, and the sign of χ″.** An FDT run's summary reads the
effective-temperature ratio at the probe nearest the natural frequency (`ratio_at_resonance`), which
on a two-frequency grid can be a decade away. And no test pins the sign of the lock-in's χ″: the
end-to-end sanity-check test accepts a failing passive baseline, so a flipped sign would pass the
suite. Whether to pin it is still to decide.

**A NaN bin as the FDT peak.** `core.FDT.spectral.find_spectral_peak` takes a NaN bin as the peak,
which moves ω₀ for a spectrum that is NaN in some bins; correcting it would shift two counts the
tests pin. It is left for a pass over the FDT science.

**The FDT band for the Hopf model.** At the default band the FDT analyses refuse the shipped Hopf
cell. The Welch segment's cap of 2^14 samples fixes the spectrum's first bin at 0.0383 in
non-dimensional units, and at the shipped settings a cell is refused exactly when its spontaneous
peak lies below 0.3835. One band for every model, or a preset per model, is still to decide.

**The tier-1 cell in the FDT pickers.** `master_spont_tier1.txt` has no bounds file of its own name,
so it resolves to `master.txt` and is refused there for lacking f_scale, yet the FDT and CrossVal
screens both offer it. Fix the cell, add a same-named bounds file, or leave it out of their pickers.
