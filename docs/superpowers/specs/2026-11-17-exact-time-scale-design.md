# Task A9: the cell at its exact time scale — THE DESIGN (final)

Written 2026-11-17, 12:22 PST by `date`, by the designer of A9 (opus): the draft (`briefs/a9-design-draft.md`, 11:45)
with every ruling of `briefs/a9-design-rulings.md` (D1–D18) folded in, after `reviews/a9-design-reader.md`. Read at
`ef86cba4` (`git show`); nothing run, nothing edited. Line numbers are `ef86cba4`'s (see the ground). Rests on
`reports/a0-stride.md` and `reviews/a0-check.md`.

**The owner's answers (11:53).** The design is approved, with its four smaller choices as recommended: keep the cost
limit (question 2), refuse a posterior made before the fix (4), leave the two probe-check records (5), and if M1
fails, stop and decide with the numbers (6). The chi settings are re-measured after the fix, and `core/config.py`
changes only on the owner's yes (1). The bench recordings CAN start with the drive at the known point: "Yes, they
can" (3). The camera's frame interval (7), answered between 12:22 and 12:53: "The camera's frame rate can differ, I'm pretty sure";
reading a bench recording at its own frame interval is a later task, owed before any inference on bench recordings.

## 1. The meaning, in plain words

After the fix, a simulated row is a recording of one cell made the way a bench camera makes one. The cell runs at
exactly the time scale written in its row, not at the nearest of 40, 20, 13.3, 10 … ms as now. Before the recording
it settles exactly as today, for a hundred model time units (0.1 to 4 s), at the step the prior's stability check
approved, with the drive already running. At the end of that settling the recording starts, and so does the drive's
clock, so a probe or a forced drive is at a known point of its cycle there, as a bench drive is when the camera
starts. From that moment the simulation takes small steps that divide each camera frame exactly, so every kept
point is exactly one frame after the last in that cell's own time. The step is exact from the first stretch of a
long simulation to the last, and no point is repeated where two stretches meet. A row then holds what a correct
clock and a correct lock-in read from a bench recording of that cell; its label, unchanged in value, is true; and
the simulated observation and the predictive check show the cell at its labelled time scale, with a phase that
means something. The finished set of 10,000 batches stays on disk as it is, and no new training will ever pick it up.

## 2. The stride

**The rule (one function, `core/SBI/clock.py`, new).** For a batch at time scale `t_scale_k`, the frame in model
time is `Δ_k = dt_exp / t_scale_k`, and `r_k = Δ_k / dt_nd_min = t_scale_hi / t_scale_k`, between 1 and 40.

- **The transient (D5): today's, at the certified step.** `steady_idx` steps (4000 on the real config;
  `sim_config.py:171-185`) at exactly `dt_nd_min`, as a segment of its own. It is as long in the cell's own time as
  today: `steady_idx·dt_nd_min` = 100 model units (`TRANSIENT_ND_UNITS`, `config.py:616`). `steady_idx` stays the
  ARGUMENT that sets it, so the suites' 500-step transients keep working and no knob becomes a config read. The
  first kept frame IS the transient's end, fine index `n_tr = steady_idx`. The assert at `sim_config.py:179-184` and
  `model_store.T_SCALE_MAX_S` (`model_store.py:75-78`) keep their meaning.
- **The recorded part.** Stride `s_k = ceil((1 − 1e-6)·r_k)` fine steps per frame, written so: the factor absorbs
  float32 time scales on a whole `r`. Step `h_k = Δ_k / s_k` from the first kept frame on. So `dt_nd_min/2 <
  h_k ≤ dt_nd_min` (to one part in a million), and `h_k = dt_nd_min` exactly where `r_k` is whole. Today's step is
  not `dt_nd_min`: it is the float32 grid's spacing, 0.0250000104 (`sim_config.py:164-166`, `sdeint.py:199-200`). So
  even at a whole `r` the new step differs from today's by about 4e-7.
- **The case `r < 1` (D9).** An arm of a central difference may pass the box's top, where `t_scale > t_scale_hi`,
  `r < 1`, `s = 1` and `h = Δ < dt_nd_min`: finer, not coarser. No clamp is needed.
- **Never coarser than `dt_nd_min`, and why.** The prior's stability screen accepted its parameter sets by
  integrating them at that step (`orchestrator.py:830-847`). An explicit Euler–Maruyama step coarser than the one it
  certified can blow up on a set the prior admitted. A finer one stays inside the region the screen tested, and its
  accuracy is never worse than today's. The transient is now exactly that certified step, and `ceil` is the smallest
  stride that keeps the promise for the recorded part. `round` (`pipeline.py:1655`) would make the step up to half
  again coarser (`r` just under 1.5). No floor is needed: the step never falls below half of `dt_nd_min`, so a frame
  never costs more than twice today's steps.
- **The grid.** `N_k = int(T_nd_k / dt_nd_k)` frames, as today (`pipeline.py:1656`, truncation kept);
  `n_fine_k = steady_idx + s_k·N_k`. Kept frame `j` is fine index `steady_idx + j·s_k`, at model time
  `100 + j·Δ_k`: `gen_obs(...)[:, ::s_k][:, :N_k]`, since gen_obs already drops the transient.

| t_scale (ms) | 1, 2, 5, 40 | 3.0 | 3.73 | 6.0 | 7.215 | 12 | 13.9 | 16 | 22 | 25 | 27 | 30 | 37.13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stride today / new | r / r | 13 / 14 | 11 / 11 | 7 / 7 | 6 / 6 | 3 / 4 | 3 / 3 | 2 / 3 | 2 / 2 | 2 / 2 | 1 / 2 | 1 / 2 | 1 / 2 |
| recorded step (today 0.0250000104) | 0.025 | 0.02381 | 0.02437 | 0.02381 | 0.0231 | 0.02083 | 0.02398 | 0.02083 | 0.02273 | 0.02 | 0.01852 | 0.01667 | 0.01347 |
| fine steps, 10-s record, today / new (thousands) | same | 134 / 144 | 114 / 114 | 74 / 74 | 64 / 64 | 34 / 44 | 34 / 34 | 24 / 34 | 24 / 24 | 24 / 24 | 14 / 24 | 14 / 24 | 14 / 24 |

**What each consumer is handed: a frozen `FrameClock`.** It holds the stride; the recorded step (one float, or one
value per row); `n_transient` (= `steady_idx`) and its step (`dt_nd_min`); `n_points`; `dt_exp`; and `chunk_len`
(`CHUNK_LEN`, passed by the caller as an argument). It derives:
- `n_fine`, and `segments()`: the transient as one segment, then the recorded part split into at most `chunk_len`
  steps each, so `n_segments = 1 + ceil(s·N / chunk_len)`;
- `nd_grid()`, the model time of each fine step, built in float64 and cast. It is defined only for a one-step clock
  (D10). A per-row clock, used only by the predictive check, hands `gen_obs` a length-only grid, because the solver
  reads `t`'s length alone;
- `drive_times()` (section 3).

How each consumer takes it:
- **`gen_obs`** REQUIRES `clock=` or `dt=` (D11). A call with neither raises. It asserts that `t`'s length,
  `steady_idx` and `n_segs` agree with the clock, so a forgotten call site cannot run silently on a legacy step on
  `cfg.t`'s prefix. A source-scan pin checks that every `gen_obs` call in `core/` passes one of the two.
- **The kept trace** goes to `gen_stats` with `dt_exp`, now its true spacing.
- **`gen_chi_raw`** takes the clock as a keyword-only `clock=` in place of `t_fine`, `steady_idx` and `subsample`.
  Its call sites are converted to keywords; the positional `init_rows` trap at `pipeline.py:1098-1102` is the
  reason. It reads `N_points` and `dt_exp` from the clock, and `_probe_record` (`chi_probes.py:347-362`) reads them
  there too (D16).
- **The label** stays `t_scale_k` (`pipeline.py:1682`).

The constructors:
- `frame_clock(dt_exp, t_scale, n_points, dt_nd_min, steady_idx, chunk_len, *, stride=None)`;
- `rows_clock(..., t_scales)`: one stride from the smallest time scale, one step per row. It serves the predictive
  check's bins, and the arms of a central difference, which then share one step count and one noise sequence;
- `training_n_fine(...)`, vectorised in float64 from the float32 schedule values for the schedule's filter (D12). A
  pin holds it equal to `frame_clock(...).n_fine` over a whole Sobol draw, so the filter and the loop share one
  arithmetic; today they do not (`probe_math.py:6-9`);
- `RECORD_CLOCK = "frame-exact/1"` (section 6).

**Every place that puts time on the clock today, and its change (with D4).**

| place (HEAD) | change |
|---|---|
| `pipeline.py:1651-1659` the batch geometry; `:1031-1059` `_BatchGeometry` | one `frame_clock` call; the geometry carries the clock |
| `pipeline.py:1062-1210` the three row makers | `clock=`, `::s_k`, the drive on `drive_times()` |
| `pipeline.py:1251-1305` `_batch_schedule` (`:1280-1282`) | `training_n_fine`; the cap `min(N_ND_MAX, len(t))` kept (`:1272`) |
| `pipeline.py:1783-1787` `_per_row`; `:686-978` gen_obs and its two helpers | the clock's `n_fine`; the clock passed down; a per-row step sliced with the rows in gen_obs' split (`:747-753`) and `_gen_obs_retry`'s halving (`:838-844`) (D10) |
| `chi_probes.py:183-191, 263, 273-277` | the stride is always one int (the `(B,)` gather and its clamp go); the drive on the clock |
| `orchestrator.py:165-263` the simulated observation | `frame_clock` at `t_scale_gt`; the OOD warning and the `len(t)` clip use its `n_fine` |
| `orchestrator.py:2725-2753` the central trajectories; `ppc.py:63-124` the bins | `frame_clock`; `rows_clock` for each bin, with plain slicing |
| `decorrelate.py:151-277`; `identifiability.py:254-283, 506-575` | `rows_clock` over each point's arms (all `2P+1` time scales), one step per arm, which keeps common random numbers |
| `probe_math.py:25-60`; `probes.py:95-133, 262-272, 1407-1418` | `Geometry` and the ceiling from the clock |
| `observations.py:93-107`, the bench's out-of-distribution warning (D5) | onto the clock's rule (section 3); its arithmetic only, so every bench row stays byte-identical |
| `gui/screens/model_builder_screen.py:488-490`, the builder's Validate smoke (D4) | passes the step (a `(1, 1)` tensor of its `dt_nd`) |
| `model_store.py:75-78, 161-169`, `T_SCALE_MAX_S` (D4, D5) | value and meaning unchanged (the transient is `steady_idx` steps again); its comment says a recording near the box's top needs `ceil(r)/r` more recorded steps |
| `orchestrator.py:657-668` the preview; `:826-847` the prior screen | unchanged: the cap is unchanged, and the screen works in model time only |

`sim_config.py:143-185` is unchanged: `dt_nd_min` is now the transient's step and the recorded step's ceiling, and
`cfg.t`'s values are read only by the prior screens; its length stays the ceiling it is today.

**The cost, from the box and the schedule's law (D6; not measured).**
- **How the schedule is drawn.** `t_scale` and `T` are drawn log-uniformly and filtered by `n_fine ≤ N_ND_MAX` =
  300,000 (`pipeline.py:1272-1283`). The transient is today's, so only the recorded part changes, by
  `ceil(r)/round(r)`.
- **Below 8.9 ms** (`r > 4.5`, about four fifths of the set's simulation), the cap binds both before and after, so a
  batch costs the same. The longest recording drawn shrinks by `round(r)/ceil(r)` wherever those differ, below 10 ms:
  at worst by a sixth, at 7.3–8 ms, and only to 59.2 s at 8.9–10 ms.
- **Above 10 ms** recordings reach 60 s both times, and the steps per frame rise: 2× at 26.7–40 ms, 1.5× at
  16–20 ms, 4/3 at 11.4–13.3 ms (and 5/4 at 8.9–10 ms).
- **Over the schedule.** The reader's careful count gives +7.6 % fine steps and +7.7 % with the planner's splits at
  the 6 GiB limit: 2,048 rows fit whole up to about 112,000 steps, take 2 chunks to 213,000 and 4 to 357,000, each
  chunk a full pass of a launch-bound solver. That is 28.9 h against the finished set's 96,666 s (26.9 h), and it
  is the upper end. Keeping today's transient takes off a few tenths of a percent, so the cost is about +7 to +8 %,
  28.7–28.9 h. M2 settles it.
- **The accepted region.** The accepted (t_scale, T) area of the built-in box shrinks by about 1 %.
- **User-model boxes** (`t_scale` in (v/2, 2v), so `r` in (1, 4), and a grid of 240,000 steps). Near the top of
  such a box, the longest recording drawn shrinks by up to `round(r)/ceil(r)`: half, just above `r = 1`, once their
  long transient leaves less than 120,000 steps for the recording, that is for v above about 0.6 s. The filter
  drops those draws without a word, as it drops every over-long draw today. The owner is told this; it is not a new
  question.
- **Memory.** The largest batch is still 300,000 steps with a 4,000-step transient, so the planner meets no new
  regime.
- **The graph capture.** A grid of another length per batch costs nothing, because the length is not in the graph's
  key (`sdeint.py:102-104`). The step is, so section 4 makes it a captured buffer: one capture per batch width
  serves the whole run, as today.
- **The alternative the owner declined.** Keeping today's recording range (filter `N·r ≤ N_ND_MAX − steady_idx`)
  costs about +12 %, and up to 21 % more memory in the longest batches.

## 3. The drive's clock

- **Today.** `t_dim = t_scale·t_nd + t_offset`, with `t_nd` counted from model time 0, the start of the discarded
  transient (`forcing.py:158-171`). Both lock-ins count from the first kept frame (`chi.py:463`,
  `statistics.py:414`). So every probe carries an extra `2π·f·t_scale·100`: tens of degrees, or many whole cycles
  (A0, lead 2).
- **After: `drive_times()`.**
  - Through the transient, fine index `j ≤ n_tr`: `t_j = (j − n_tr)·dt_nd_min·t_scale`. These are negative physical
    times, from that segment's own step, reaching 0 exactly at the first kept frame.
  - From there on: `t_j = (j − n_tr)·dt_exp/s`.
  - Both plus `t_offset`.
  - The recorded part does not depend on `t_scale`, because `s` fine steps are one frame. The transient part does,
    so a per-row clock gives one `(B, n_fine)` time grid. That is no new memory: the builder already makes a
    `(B, T)` `t_dim` (`forcing.py:171`).
  - `build_nondim_force_tensor` and its sine wrapper (`forcing.py:110-198, 233-267`) take the grid as a new keyword
    `t_dim`. `t_scale` is still read for the Hopf-style force scale.
  - The `t_nd` path stays for the user-model kinds, which only tests call (`build_user_force_tensor` has no
    production caller). Its golden test (`tests/test_user_models.py:220`) is untouched.
- **Chi mode.** The phase is π/2 (`chi_probes.py:262`), so at the first kept frame the drive stands at the top of a
  cosine: the reference `lock_in_batched` assumes, and an ideal response reads χ = 1 at phase 0. On a box that infers
  `t_offset`, the probe at the first frame stands at cos(ω·t_offset) instead (D8). The retrain's box has no
  `t_offset`.
- **Forced mode.** The drawn phase is the drive's phase at the first kept frame, in the sine convention. G2 = arg R₁
  − phase then carries the cell's own phase, plus `2π·f·t_offset` on a box that infers `t_offset`, which is what
  `statistics.py:24-26` says makes `t_offset` identifiable.
- **The transient's drive** is on from the start, at negative times, so the cell is in its driven steady state at
  the first frame. A drive switched on at the first frame would put its onset into every lock-in.
- **The bench path is right, and its rows do not move.**
  - In forced mode it takes the typed phase (`observations.py:68-70, 109-142`), and Group G locks in from sample 0.
  - In chi mode it takes each recording's frequency and one amplitude (`observations.py:192-303`), and locks in
    against a cosine from sample 0 (`:278`).
  - The window hands exactly those, under Inference, "Your own recordings": the drive's four SI rows, or a table of
    files with their Hz and one Drive amplitude.
  - So the bench's convention is t = 0 at the first frame, and after the fix the simulation uses the same one.
    Today's phase channels were scrambled in training (A0: cos and sin sd 0.707 and 0.704, the uniform circle's).
    After the fix they carry the cell's phase.
  - The owner answered that every driven recording CAN start with the drive at that known point: the top of a cosine
    for the frequency response, the typed phase for forced mode. The recordings guide says so in this task, and the
    "no" branch is not designed (D3).
- **The bench's out-of-distribution warning (D5).** `observations.py:93-107` assumes the continuous rule
  `steady_idx + N·r ≤ N_ND_MAX`, and so names a shortest time scale below the one training draws. It moves onto the
  clock's rule: training admits `ceil(r) ≤ q = floor((N_ND_MAX − steady_idx)/N)`, so the shortest time scale
  trained is `t_scale_hi / q`. A helper beside `training_n_fine` computes this and is pinned against it. Only lines
  93-107 change, and a test pins the three bench builders' rows equal before and after.
- **The bench's one remaining clock (D3).** With the simulation exact, the only clock left is the camera's own frame
  interval against the units file's `dt` (1 ms). A 1 % error there reads |sinc(π·20·0.01)| ≈ 0.94 at 20 cycles, and
  shifts the time-unit features by log 1.01. This is question 7, open.
- **The display axis.** The simulated observation's time axis (`orchestrator.py:246`) already counts from the first
  frame, plus `t_offset`, so the drive and the axis now agree.

## 4. The solver's step

- **Today.** Every segment re-derives its step from a float32 `linspace` of its own end points (`sdeint.py:199-200,
  267-268`, from `simulator.py:160-161`). A later segment starts near t = 2,500–5,000, where float32 spacing is
  2.4e-4, hence +0.098 % and −0.391 % (A0, lead 3). A float64 grid does not help: `linspace` without a dtype is
  float32.
- **After: the caller hands the step.**
  - The solver never derives a step from times. `euler` and `euler_compiled` take the step as a required argument in
    place of `ts` (D4: the builder's Validate and `tests/test_user_models.py:111, 145, 153` pass it).
  - `Simulator.simulate(state_dep_drift, plan=None)` integrates the clock's segments, each at its own constant step:
    the transient at `dt_nd_min`, then the recorded segments at `h`. The step changes once, at the first kept frame.
    The constructor does not change, so the four subclasses are untouched.
  - Only `Simulator.simulate` keeps a derivation, when it is handed no plan, for the prior screens
    (`core/SBI/Priors/*_prior.py`, `orchestrator.py:830-847`). It reads the step ONCE from the grid's first
    interval, exactly as segment 1 computes it today, and uses it in every segment.
- **The step's form.** The step is a `(B, 1)` tensor, and so is its square root: one value for every row of a
  training batch, each draw's own in a predictive-check bin.
  - The compiled steps take `dt: Tensor, sqrt_dt: Tensor` (`nadrowski_model.py:4-33`, `hopf_model.py:4-17`, the
    generated text at `user_model.py:210, 221`). The arity is unchanged.
  - In the graph they are static buffers copied in per call, like the parameters (`sdeint.py:149, 288-289`), and the
    step leaves the graph's key.
  - On the CPU the one-segment prior screen is expected bitwise as today. On the card it takes the compiled path, and
    a tensor operand may fuse differently. That is expected, not proven (D7); a move there, while the CPU screen is
    bitwise, is reported as the arithmetic's.
- **What float64 time costs on the card: nothing on the solver's path**, because no time tensor exists there.
  - The step is a Python double, rounded once to float32 (one part in 1.7e7). The cell's clock ends at most 0.02
    fine steps off after 300,000 steps, against 783 behind today.
  - The drive's time grid is one float64 `arange` per batch (2.4 MB), cast to float32.
  - The `(B, n_fine)` drive stays float32, as today: a float64 one would be 4.9 GB at 2,048 × 300,000. Its phase
    error at a 60-s forced drive is at most 0.004 rad, as today.
- **The seam.**
  - Segment k ≥ 1 starts from the state at the previous segment's last index.
  - It is handed the drive from that index (`simulator.py:79-80` slices from `a_k − 1`), integrates its own length,
    and writes only its new states.
  - So index g holds the state after g steps in every segment, and drive index m serves step m → m+1.
  - A segmented run equals one long run: bitwise on the CPU, with the same draws in the same order.
  - The warnings at `simulator.py:92-99` and `architecture.md:989-994` go.

## 5. What a row holds after the fix

| part of the row | the finished set | after |
|---|---|---|
| width, layout | chi 49 + 1 + 72 = 122, forced 54, spontaneous 50; 12 slots × 6 channels, `CHI_LAYOUT` 2 | unchanged |
| label `t_scale` | the drawn value; the trace's cell was at `40/s` ms | the same value, now the trace's |
| A–F, the time-unit features | the cell at `40/s` ms | `A3_log_fpeak` moves by −log c; `A4_log_acf_decay` and E1 τ fast/slow by +log c (\|log c\| ≤ 0.405 over the box, at most about 0.2 of a spread) |
| other A–F | — | ≤ 0.05 of a spread on average; `B7_*` and `E2_*` move with any change of the sampled path (A0); plus the step's own change, measured by M1 |
| log T | `log T_k` | unchanged, now the trace's true length |
| χ: u, logcyc, mask | about a peak read on the wrong clock | the same formulas, about the true peak; masked share expected near 37 % (read on the card) |
| χ: log\|χ\| | an ideal response read 0.23 (median, 20 cycles) | reads 1 |
| χ: cos, sin | the uniform circle | the response's phase against a cosine at the first frame |
| G (forced) | G1–G7 at the wrong frequency; G2 scrambled; G6 0.03–0.22 for entrained cells | at the drive's frequency; G2 the cell's phase; G6 0.45–0.67 there (A0) |

- **`FIT_CONSTANTS`** (`fit_state.py:109-110`): none changes.
- **`FIT_CODE`** (`fit_state.py:88-106`): no listed function changes.
  - The edits to `orchestrator.py` stay outside `build_embedding_net` and `expected_forcing_dim`.
  - `chi.py`, `train.py`, `embedded_network.py`, `chi_encoder.py` and `summaries.py` are not touched.
  - This holds only while `gen_training_data`'s signature and `TrainingPlan` hold, because `prepare_fit` calls it
    positionally (`train.py:412-416`; D15). Neither changes: the clock is computed inside the loop from the same `t`,
    `steady_idx` and `dt_nd_min`. A build that would change either STOPS.
  - `tests/test_fit_saving.py:353` pins the digests.
- **`FEATURE_SET_VERSION`** stays 1: no feature's definition changes.

## 6. How a set is recognised

- **The change.** `SimulationIdentity.from_cfg` (`identity.py:86-122`) gains `"record_clock": clock.RECORD_CLOCK`
  (`"frame-exact/1"`). `FORMAT` (`identity.py:45`) moves from `"training-rows/3"` to `"training-rows/4"`, which is
  the module's own convention when the identity gains an entry (`identity.py:17-18`). `identity_digest`
  (`training_checkpoint.py:111-114`) then names a new 12-character folder for every configuration. The value for the
  retrain's settings can be had only by running, and the builder records it.
- **Why both.** The old set then differs in two entries, so it is never "one setting away" (`near_miss_siblings`
  keeps exactly one, `training_checkpoint.py:362-366`). `build_posterior`'s refusal (`orchestrator.py:1395-1399`)
  and the command line's `--new-run` never fire on it. Either change alone would make it a near miss whose "setting"
  no one can set.
- **What is said of the old set.**
  - The run's log: `[checkpoint] 1 other checkpoint(s) exist and do NOT match this run: eda5c2a1e468 (10000 batches,
    differs in format)`.
  - The window's Training step asks its question before simulating from zero, listing the model's saved sets
    (`posterior_tab.py:793-815, 905-918`, Cancel preselected). The old set reads "Made by another version of PRISM",
    because `record_clock` joins that row (`fields.py:239`).
  - Records and `artifacts list simulation` show it as now: complete, 10,000 batches, unnamed. `show` prints its
    stored identity.
  - Nothing writes into its folder: a run writes only into its own digest's (`training_checkpoint.py:122-126`).
- **Posteriors.** `conditioning_block` (`manifest.py:186-204`) records the clock as well, so a posterior and an
  observation say which clock made their rows. The owner answered yes to question 4, so `load_posterior`
  (`store.py:1214-1334`) refuses a posterior whose record lacks the entry or holds another value. None exists on disk.

## 7. What else the old clock made, and how the owner learns of it

On disk (listed, read only): `priors/retrain_prior__20261001T222935`, `simulations/eda5c2a1e468`,
`diagnostics/retrain_band__20261001T222941`, `diagnostics/retrain_mask__20261001T223011`, and an empty `fdt/`. No
posterior, fit, observation, calibration or inference exists.

- **The prior: unchanged.** Its screen integrates model time only, at `dt_nd_min`, in one segment of 40,000 steps,
  with no time scale and no drive (`orchestrator.py:830-847`). So neither the stride, nor the drive, nor the
  later-segment step reached it, and the legacy step keeps it bitwise on the CPU. It loads as now, and the new set
  uses it; its fingerprint, in the new identity, does not move. The fix does not change what a prior accepts.
- **The finished set:** unmatched, kept, and never continued (section 6). It is unnamed and complete, so the store
  lists it among a sweep's candidates (`store.py:1140-1148`). The owner is told (D18) that naming it in Records keeps
  it out of every sweep's list; the audit touches nothing under `Artifacts/`.
- **The two probe-check records** were made by the old clock. The band check simulated the master cell at 3.73 ms,
  stride 11, c = 1.026, where an ideal response reads 0.625 at 20 cycles. The mask check ran the training generator
  over the prior. Both are stale but loadable: `load_diagnostic` checks nothing (`store.py:1416-1435`). They stay, as
  the owner chose; new ones come from the chi re-measurement (section 10, step 7).
- **The settings chosen by measurement under the old clock.**
  - Which: the lock-in ceiling `CHI_MAX_CYCLES` 20 (`config.py:530-562`), the band's top edge 0.3 (`:456-489`),
    `CHI_F0` 0.15 (`:573`), and, less so, `CHI_MIN_CYCLES` 2.
  - Where: all measured on the master cells, whose time scale is 3.73 ms (c = 1.0258). There the clock alone makes an
    ideal lock-in fall as |sinc(π·n·0.0258)|: 0.62 at 20 cycles, 0.23 at 31, and a null at 39.
  - Why they are suspect: the recorded "wall at about 31 cycles", whose spread climbs to its first failure at 36, is
    what that mismatch predicts. So is "re-locking the same trace over a shorter prefix reverses it"
    (`science.md:240-263`), as well as the phase diffusion the guide proposes.
  - The verdict: likely an artefact, not established. The owner chose to re-measure them after the fix (section 10,
    step 7).
- **The guides.**
  - `science.md:42-58` argues that a step chosen per batch is wrong, because the network could read `t_scale` off
    integration error. This design does choose one, for the recorded part (between half and all of `dt_nd_min`).
    M1 answers that argument, and the page is rewritten with its numbers.
  - `rules-and-traps.md:44` ("derives its step from the time grid; nothing passes it one") becomes false.
- The owner learns all of this from this design and the build's report; nothing under `Artifacts/` is written.

## 8. How the fix is shown right, by numbers

**Before, at HEAD.** A0's scripts and logs (`C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\audit\a0\`) are
the before. The builder adds (e) at HEAD before the first commit.

**Measured before building (the builder, scratch scripts; no repository edit).**
- **M1, the step (D2): a paired ensemble test.**
  - The cells: per cell and time scale, 64 to 256 rows (A0's batch-49 cells read from the shards, or the master
    cell), each on one Brownian path at `h` and at `h/4`: the fine increments, summed.
  - The time scales: 39, 30, 22, 13.9, 7.215 and 3.73 ms, plus the jump at each whole `r`, just below and just
    above 13.33, 20 and 40 ms. 3-s records.
  - The channels: every one the network reads. That is the live A–F features; the chi block's log|χ|, cos and sin
    per probe; and Group G in forced mode, with drives across the forcing prior's top decade.
  - The spreads: the finished set's for A–F, and the ensemble's own for the χ channels and G, because the finished
    set's are the scrambled ones.
  - The reading: the mean of feature(`h`) − feature(`h/4`), with its 95 % interval.
  - PASS: |mean| ≤ 0.05 of the feature's spread, with the interval inside that bound. Per-row differences are
    reported, never gated.
  - Reported as its own number: forced mode's G2, at slow cells and fast drives (ω·h up to about 1.4 rad).
  - A FAIL on any channel stops the build and comes to the owner with the numbers (the owner's answer to question 6).
- **M2, the cost (D6).** Σ `n_fine`·chunks over a fresh 10,000-batch schedule drawn with the new filter, against the
  same draw's today rule, at the 6 GiB limit. Arithmetic only, seconds.
- **M3, on the card when it is free.** The graphed step's time at width 2,048 with the tensor step, against today's
  6.65 µs. A slowdown over 2 % is reported before building on.

**After (seeded, on the CPU; each commit's tests, or scratch).**
- **(a)** A0's table 3(a) on the new clock: every row reads 1.000 within the finite-window term (≤ 0.006 at 2
  cycles, ≤ 0.001 at 20 over whole cycles), with |arg χ| ≤ 1e-3 rad. Over 220 log-uniform time scales the worst
  reading at 20 cycles is within 1e-3 of 1. In the suite.
- **(a2)** A0's 3(a2): `gen_chi_raw` with "response = drive" at its ten time scales. Mean |χ|/true lies in
  [0.99, 1.01] at each (today 0.09–0.85), and the median |phase| is ≤ 0.01 rad. In the suite.
- **(b) The order of convergence (D13).** A cell's kept trace over a short window (its first 200 frames), at `h`,
  `h/2` and `h/4`, each against `h/8` on the same path: the RMS difference falls with the step at Euler–Maruyama's
  order, ½ for state-dependent noise and 1 for additive.
- **(c)** The phase of an ideal response is 0: in (a). Also `drive_times()[n_transient] == 0`, and the step switches
  there.
- **(d) The seam and the segments (D13).** Through `Simulator.simulate` on the CPU, with a non-constant drive and the
  noise on: a run cut into many segments (a small `chunk_len`) equals the same clock in two segments bitwise. A
  clock model (drift 1, no noise), in float64, holds `g·dt_nd_min` through the transient and
  `n_tr·dt_nd_min + (g − n_tr)·h` after it, at every index `g`, to 1e-12, across every seam (A0 measured 783 steps
  behind).
- **(e) A whole stride (D1).** At 5 ms, `r` = 8 and the stride is today's; the step, 0.025, differs from today's
  0.0250000104 by 4e-7. 3-s rows (one recorded segment), 8 rows, chi and forced:
  - the spontaneous features lie within 1e-3 of their spread of HEAD's, and log T is equal;
  - with the ideal stub, |χ| is equal to 1e-6 and the phase moves by 2π·f·500.0002 ms within 1e-4 rad (the drive's
    old origin, `5·cfg.t[4000]`), with nothing else moving;
  - with the real cell, |χ| lies within its seed spread;
  - at 15 s (two recorded segments), samples on both sides of the seam may move by the same tolerance.

  A bitwise claim is made only for a scratch comparison against HEAD with `cfg.t` replaced by an exact
  `0.025·arange` grid.
- **(f) The card smoke gate and the diagnostic card** (`testing.md:240-456`).
  - The exits are as recorded.
  - The prior lines are expected unchanged (`kept 101779 of 102400 candidates over 40000 steps`, `175718 sets in
    all`). A move on the card while the CPU screen is bitwise is reported as the arithmetic's (D7).
  - Masked probes within 37 ± 12 %, recorded.
  - Stage timings are recorded against the ratio of Σ `n_fine`, new over old, on the smoke's own schedule, not failed
    (D14).
  - The `probes band` and `probes drive` readings are recorded, not failed: they are the chi re-measurement's.

## 9. Tests that pin today's numbers, and the new ones

**Brought to the new numbers, each for the reason given.**
- **The geometry pin.** `tests/test_diagnostics.py:1758-1847` lifts the batch loop's five geometry statements and
  runs them. It is replaced by a pin that the loop and `probe_math` both take `frame_clock`'s answer: one function
  replaces the restatement.
- **The values the pin checks.** The values at 3.73 ms do NOT move: the stride is 11 before and after, and the
  transient is 4,000 steps (D5). So `test_diagnostics.py:1858` `Geometry(5000, 59000, 11)`, the ceiling 26.909 s
  and the default lengths (… 26.9) stand. The grid-fit refusal (`:3005-3025`) uses the clock's ceiling.
- **The strided slice against the per-row gather.** `tests/test_user_sbi.py:588-626`: the gather goes, because a bin
  has one stride.
- **A batch never outruns the grid.** `test_user_sbi.py:720-880` is kept, because the filter keeps `len(t)`; its
  `n_fine` comes from the clock.
- **Callers of `gen_chi_raw`.** `test_user_sbi.py:891-950` and `tests/test_tier1.py:361` pass `clock=`.
- **The solver and the graph.** The compiled step's arity (`test_user_sbi.py:1682-1716`, `n_in − 5` unchanged),
  `:1718-1750`, `:3834-3893` and `:3896-` take the step and its root as `(B, 1)` tensors, and the graph key without
  the step. The direct `euler` calls in `tests/test_user_models.py:111, 145, 153` pass the step (D4).
- **The golden digest.** `test_user_sbi.py:2898-2908`: `1d799334fcf7` becomes the new value. The comment says the
  clock entry joined the identity (training-rows/4) so that no set simulated under the old clock is ever continued.
  The key set gains `record_clock`.
- **The identity's keys.** `tests/test_artifact_store.py:701-705` (`_IDENTITY_KEYS`) and `:744` (`FORMAT`) change
  likewise. `tests/test_nav_and_gating.py:2682-2687` (every identity entry has a row) holds with `fields.py:239`.
- **The stand-in.** `tests/_fixtures.py:602-615` (`stand_in_gen_obs`) gains `clock=None, dt=None`; the other
  stand-ins take `**kw`.
- **Multi-segment numbers.** Any CPU number from a multi-segment simulation that the fast gate finds moved is
  brought to the new value: the seam no longer repeats a sample.
- **The card's expected readings** (`testing.md`, and `docs/STATE.md`'s gate table, word for word) are updated only
  where (f) reads another number, with its reason.

**New.**
- **The clock's arithmetic:** section 2's table; `dt_nd_min/2 < h ≤ dt_nd_min` over 10,000 time scales; `h = Δ`
  for `r < 1`; whole strides at `dt_nd_min`; `training_n_fine == frame_clock(...).n_fine` over a whole Sobol draw.
- **The ideal readings:** (a) and (a2).
- **The drive's origin and the step's switch:** (c).
- **The seam and the segments:** (d).
- **A legacy plan-less `Simulator.simulate`** keeps its first interval in every segment.
- **Every `gen_obs` call in `core/` passes a clock or a step:** a source scan.
- **The per-row step:** a split, or a halved, batch with per-row steps equals the unsplit one with the noise off;
  and a predictive-check bin's row equals that row simulated alone.
- **A Fisher point's arms** share their stride.
- **The bench:** the three builders' rows are byte-identical; the bench warning follows the clock's rule.
- **The identity:** one without `record_clock` is never a near miss, and reads "Made by another version of PRISM".
- **The refusal:** `load_posterior` refuses a posterior whose record lacks the clock (question 4).
- **On the card (gpu-marked):** two batches at different steps share one capture.

## 10. The build in steps (one Opus builder, commits in order)

0. **Measurements first.** M1 and M2 (on the CPU, in scratch), and M3 (on the card, when it is free) are reported
   first. M1 failing stops the build.
1. **The solver's step, given and exact.** Files: `core/Solvers/sdeint.py`, `core/Simulator/simulator.py`,
   `core/Models/nadrowski_model.py`, `hopf_model.py`, `user_model.py`, `core/gui/screens/model_builder_screen.py`
   (Validate), and `core/SBI/pipeline.py` (gen_obs and its two helpers carry the step and slice it with the rows).
   Also the solver tests and the stand-ins. Rows do not move yet, except across seams.
2. **The record clock.** Files: `core/SBI/clock.py` (new), `core/forcing.py` (`t_dim`), and
   `core/Helpers/model_store.py` (`T_SCALE_MAX_S`'s comment). Its tests, and (a).
3. **Training rows on the clock.** Files: `pipeline.py` (the loop, `_BatchGeometry`, the row makers, the schedule,
   `_per_row`, gen_obs requiring a clock or a step, and the scan pin) and `core/SBI/chi_probes.py`. Tests: (a2), (e)
   and the geometry pins.
4. **Every other simulated path.** Files: `core/orchestrator.py` (the observation, the central trajectories),
   `core/SBI/ppc.py`, `core/SBI/decorrelate.py`, `core/diagnostics/identifiability.py`, `probe_math.py`,
   `probes.py`, and `core/SBI/observations.py:93-107` (the warning only). Their tests.
5. **The identity names the clock.** Files: `core/artifacts/identity.py`, `manifest.py`, `store.py` (the refusal)
   and `core/gui/fields.py`. The digest and key-set tests.
6. **The guides.** `science.md` (the step, with M1's numbers; the ceiling's history as measured under the old
   clock), `architecture.md` (the solver, the seam, the graph key), `rules-and-traps.md:44`, `recordings.md` (the
   drive's point at the first frame of every driven recording) and `testing.md`, each with its checked-against line.
   `docs/STATE.md` is the controller's.
7. **The chi re-measurement (D17).** After the gates below have passed: the probe checks on the master cell, on the
   card (`probes band`, `probes drive`, `probes mask` over the prior in use). Their numbers go to the owner. On a yes
   to a change of `core/config.py`, that change is its own commit, and the gates it moves run again (the card's
   expected readings and the identity's chi entries move with it). Only then is the new simulation started, from the
   window, by the owner.

**The gates.** After step 5: the fast gate, as one `pytest -m "not slow"` process in the background (about an hour);
then the card smoke gate with run 5's drill, and the diagnostic card.

**What must not move.**
- `core/config.py`, byte-identical until step 7's own commit.
- `core/SBI/observations.py` outside lines 93-107; every bench row byte-identical.
- `Artifacts/`, listed before and after.
- `FIT_CODE`'s sources (`code_digests` equal), `gen_training_data`'s signature, `TrainingPlan`, and
  `FIT_CONSTANTS`.
- The conditioning widths and layout.
- The prior screen.

## 11. The owner's questions, as answered

1. **Re-measure the χ settings before the new simulation?** Answered: re-measured after the fix (section 10, step
   7); `core/config.py` changes only on the owner's yes.
2. **Keep the cost limit, or the recording range?** Answered: keep the limit, as recommended. It shortens the
   longest recordings training draws below 10 ms, by at most a sixth (at 7.3–8 ms), for about +7 to +8 % card time.
   On user-model boxes, recordings near the top of the box can halve (section 2); this is told, not asked.
3. **Can the bench recordings start with the drive at a known point?** Answered: "Yes, they can". The recordings
   guide says so in this task.
4. **Refuse a posterior made before the fix?** Answered: yes, as recommended (section 6).
5. **The two probe-check records of 1 October?** Answered: leave them, as recommended. Naming the finished set in
   Records keeps it out of every sweep's list (D18).
6. **If M1 shows the step matters?** Answered: stop and decide with the numbers, as recommended.
7. **Is the camera's frame interval exactly the units file's `dt` (1 ms)?** Answered (17 November, between 12:22 and 12:53): "The camera's frame
   rate can differ, I'm pretty sure." This build is unchanged: training is simulated, on the exact clock. Reading a
   bench recording at its own measured frame interval is a later task, owed before any inference on bench
   recordings; the recordings guide says, in this task, that a recording is read as if its frames are exactly the
   units file's `dt` apart, and that a frame interval off by 1 % reads a 20-cycle response about 0.94 of its size.

## What reading could not settle (the builder measures)

- M1: is the recorded step converged enough to vary with `t_scale`, on every channel?
- M2 and M3: the true cost, and the tensor step's speed.
- Whether the χ settings were artefacts (step 7).
- The new digest.
- The masked-probe share on the card.
- Whether any CPU test pins a multi-segment number.
- Whether a tensor step keeps the compiled path bitwise equal to the eager one on the card (both read the same
  tensor, so it should).

## What changed from the draft

1. **The head** records the owner's answers of 11:53; section 11 now gives the answers, with question 3 answered
   "Yes, they can", question 6 as the owner chose, and a new open question 7 on the frame interval (D3).
2. **D5: the transient** is today's: `steady_idx` steps at exactly `dt_nd_min`, as a segment of its own. The step
   switches to `Δ/s` at the first kept frame, which is the transient's end. Consequences:
   - `n_tr = steady_idx` again; the whole-frame transient (`m_k`) is gone, and so is the table's transient row;
   - the drive's time through the transient comes from its own step;
   - the assert and `T_SCALE_MAX_S` keep their meaning;
   - the values at 3.73 ms that the draft moved (59103, 26.899 s, 26.89) now stand;
   - the bench's out-of-distribution warning (`observations.py:93-107`, its arithmetic only) moves onto the clock's
     rule;
   - user-model boxes are told what they lose near their top.
3. **D1: today's step** is the float32 grid's 0.0250000104, not `dt_nd_min`, so proof (e) carries tolerances. Its
   bitwise claim is kept only against an exact-grid scratch run of HEAD.
4. **D2: M1** is a paired ensemble test over every channel the network reads, the jumps at whole `r` included, with
   forced mode's G2 at fast drives reported on its own.
5. **D4: the model builder's Validate and the direct `euler` calls in the tests** pass the step; `euler` takes it in
   place of `ts`.
6. **D6: the cost** is about +7 to +8 % (28.7–28.9 h); the shortening applies "below 10 ms" (it was 8); M2 runs over
   a fresh schedule.
7. **D7: the card's prior lines** are expected bitwise, not proven.
8. **D8: on a `t_offset` box,** a probe at the first frame stands at cos(ω·t_offset).
9. **D9: the `r < 1` case** is stated: the step is finer there, and no clamp is needed.
10. **D10: a per-row step** is sliced with the rows; `nd_grid()` is defined only for one step.
11. **D11: `gen_obs` requires a clock or a step,** with a source-scan pin; the legacy derivation stays in
    `Simulator.simulate` only.
12. **D12: the stride formula** is written so it cannot be misread; `training_n_fine` is float64, pinned equal to
    `frame_clock` over a Sobol draw.
13. **D13: proof (b)** is an order of convergence over a short window; proof (d) runs a non-constant drive with the
    noise on.
14. **D14: the timings** are recorded against Σ `n_fine`'s ratio, not failed.
15. **D15: `FIT_CODE`'s dependence** on `gen_training_data`'s signature and on `TrainingPlan` is stated: a build that
    would change them stops.
16. **D16: `gen_chi_raw`** reads `N_points` and `dt_exp` from the clock.
17. **D17: step 7,** the chi re-measurement, its answer and its own commit, joins section 10.
18. **D18:** the owner is told that naming the finished set keeps it out of a sweep.
19. **The reader's one sentence:** the simulated observation and the predictive check show the cell at its labelled
    time scale, with a phase that means something (section 1).

## The ground

**The tree.** All code was read at `ef86cba4` through `git show`. By 12:28 PST the other builder had committed
`a2857c57` and `b809e497`, which remove dead code. Their only cited file is `core/SBI/pipeline.py`: 11 lines near its
top are gone, so its line numbers here are `ef86cba4`'s and sit 11 lower at the new HEAD. I edited no repository
file and wrote only this file. I ran no Python and no pytest in this pass.

**`Artifacts/`.** Nothing there was touched in this pass.

**The draft pass's slip.** It began one bash call with a stray `python3 -c`, which the harness refused; nothing ran.
