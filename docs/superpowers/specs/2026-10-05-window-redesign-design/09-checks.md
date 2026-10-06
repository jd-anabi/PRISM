## 9. Checks

Step 4 of Parameter inference: one page that lists every check there is, and one page per check
(U16). It replaces the Validate tab (`core/gui/panels/inference/validate_tab.py:18-94`), which
offers one of the twelve checks and shows its result only as lines in the log pane and three
figures (`:79-90`; W057). Nothing else under `core/gui` reaches a check today:
`core/gui/fields.py:178-186` maps the other checks' own settings to no control (W188).

Code read at commit `c45cc4a`. "In use" means what the session holds applied: `SbiSession.cfg`,
`.inf_prior`, `.posterior` (`core/gui/session.py:41-47`). Part 5 owns the session, the words "In
use: …" and the pieces a page is made of (5.2 to 5.9). A check never changes what a step holds (B3).

A check is a run, and part 6 owns everything between its start and its result (6.2 to 6.9). For
each check this part supplies the function and its arguments, `spec` (the run's word "Check", the
check's name, its page, its stage names, its record kind) and the result block. No check starts
while a run is live (U25, B15).

### 9.1 The twelve checks

Group headings on screen: "Before the run", "After simulating", "After the fit", "Tools": U16's four
groups, named from its own phrases (a choice the owner has not seen). The role on screen reads
"Decides", "Informs" or "Tool" (the three words: a choice the owner has not seen).

| # | on-screen name | the sentence under its name | group | role |
|---|---|---|---|---|
| 1 | "Frequency band" | "Can driven recordings across the frequency band, at the set drive amplitude, be measured on this cell without the drive taking over the cell's own oscillation?" | Before the run | Decides |
| 2 | "Unusable driven recordings: audit" | "What share of the driven recordings that training will simulate from this prior is unusable, and is that share in the expected range?" | Before the run | Decides |
| 3 | "Unusable driven recordings: simulation set against the audit" | "Does the share of unusable driven recordings in the finished simulation set agree with the audit?" | After simulating | Decides |
| 4 | "Calibration" | "Is the posterior honest about its uncertainty?" | After the fit | Decides |
| 5 | "Use of each part of the input" | "Does the network respond to every part of an observation?" | After the fit | Decides |
| 6 | "Identifiability of the posterior" | "Is it on record how strongly the posterior pins down each direction in parameter space?" | After the fit | Decides |
| 7 | "What the observations can pin down" | "At this cell and recording length, which parameters do the observations pin down, and which move an observation alike?" | Before the run | Informs |
| 8 | "Timing of a short fit" | "How long does one epoch of the fit take on this computer, and so how long should the full fit take?" | Before the run | Informs |
| 9 | "Calibration repeated at fixed numbers of driven recordings" | "Does the rank test still pass when every observation holds the same number of driven recordings?" | After the fit | Informs |
| 10 | "Card test" | "Does every stage run end to end on this graphics card?" | Tools | Tool |
| 11 | "Drive study" | "How hard can this cell be driven before the drive takes over its own oscillation?" | Tools | Tool |
| 12 | "Identifiability across the prior" | "Could recordings pin each parameter down at points across the prior, whatever the posterior learned?" | Tools | Tool |

Every check is a screen over an engine entry that exists today, except that check 3 calls nothing
(9.5), check 10 needs E3 (9.7), and checks 5 and 12 gain refusals under E6 (9.8).

The chosen mock-up (`structure-chosen-rail-4-checks.png`) names five checks: "Calibration",
"Frequency band", "Unusable driven recordings", "Identifiability", "Use of each part of the input".
The other names, the two longer forms of those, and the twelve sentences are (a choice the owner has
not seen). Check 12 is U16's "third identifiability method". The six deciding checks are the
runbook's two pre-run probe checks (`docs/guide/retrain.md:288-306`) and its four gates
(`:558-590`): gate 1 is check 4, gate 2 check 5, gate 3 check 3, gate 4 check 6.

#### What each calls, needs, costs, writes, and where its verdict is read

`F:` marks the arguments the form sends. Every call also sends `name`, `note`, the window's figure
sink and the default store, unless stated. "Record" gives the manifest's kind and, for a
diagnostic, `body.diagnostic` / `body.variant`.

| # | engine entry and arguments | needs | cost, where stated | record written | verdict or reading taken from |
|---|---|---|---|---|---|
| 1 | `core.diagnostics.probe_band(cfg, F: lengths, repeats, seed)` (`core/diagnostics/probes.py:512-516`) | bounds file, cell | 5 lengths × 5 frequencies × 24 simulations at the defaults (`:536-544`); no time stated anywhere | diagnostic `probes` / `band`, no parent (`:700-702`); `probe_band.npz`; one figure per drive (`:681-684`) | `body.results.band_holds` and `.drive_holds` (`:694`) |
| 2 | `probe_mask(cfg, prior, F: num_runs, run_size, seed)` (`probes.py:1090-1092`) | bounds file, the prior in use | 12 batches of 32 took 43 s on the card; 100 about six minutes, an estimate (`retrain.md:293-295`) | diagnostic `probes` / `mask`, parent the prior (`:1181-1184`); `probe_mask.npz`; no figure (`:1116`) | `body.results.masked_fraction` (`:978`, a fraction), with `.invariants.hold` (`:968`), `.attribution.holds` (`:971`), `.cross_check.holds` (`:949`), `.causes.non_finite_lock_in.count` (`:966`, `:979`) |
| 3 | none: it reads two figures (9.5) | a training's simulation set, and check 2's record for its prior | none | none | a computed difference, in points |
| 4 | `orchestrator.validate_calibration(cfg, posterior, prior, F: n_cal, cal_n_scales, num_posterior_samples, seed)` (`core/orchestrator.py:1872-1876`) | posterior, and the prior it was trained on (`:1949`) | simulates one batch per operating point (`core/config.py:276`), then draws for the rank test and the joint coverage test; 2,000 datasets at 200 points took 43 minutes on the card (`retrain.md:227-228`) | calibration, parents posterior and prior (`orchestrator.py:2091`); `results.json`, `ranks.npz`, three figures (`:2022-2023`, `:2042`) | `body.results.verdict.passed` (`:2066`, `:2083`; the rule is `calibration_verdict`, `:1814-1850`) |
| 5 | `channel_ablation(cfg, posterior, F: rows, n_sweep)` (`core/diagnostics/ablation.py:84-86`) | posterior and its own simulation set (`:103-110`, `:125-128`) | simulates nothing; reads the first 200,000 simulations (`:89-93`) | diagnostic `ablation`, parents posterior and simulation (`:204-206`); no figure | `body.results.channels[].verdict` for `A1_mean` and `D3_bimodality` (`retrain.md:567-571`), and `.counts.invisible`, `.counts.nonfinite` (`ablation.py:196-197`) |
| 6 | `identifiability_rotation(cfg, posterior, F: n_worst, top_n)` (`core/diagnostics/identifiability.py:46-48`) | a posterior trained with the rotation (`:64-69`) | simulates nothing; "runs in a second" (`:7`) | diagnostic `identifiability` / `rotation`, parent posterior (`:203-207`) | `body.results.eigenvalues` is not null (`:200`; `retrain.md:589-590`) |
| 7 | `identifiability_jacobian(cfg, F: t_obs_s, m, m_noise, rel, seed)` (`identifiability.py:844-848`) | bounds file of the Nadrowski model (`:865`), with a forcing section unless the frequency response is on (`:866-868`); cell; recording length | simulates; not stated | diagnostic `identifiability` / `jacobian`, no parent (`:941-944`); `degeneracy_map.npz`; figures (`:927-929`) | `results.unmeasurable`, `.degenerate_pairs`, `.condition_number`, `.dead_channels` (`:937-940`); the two recording lengths from the record's text (the line built at `:626-629`, written with the frequency response only, `:596-597`) |
| 8 | `orchestrator.build_posterior(cfg, prior, None, True, …)` twice, into a scratch store (9.6) | bounds file, the prior in use | two short trainings; minutes; no whole figure is recorded | none in the real store; the result goes to the run history | no verdict; `training.epochs_trained` (`orchestrator.py:1473`) and `created` of each returned posterior, and the window's clock (9.6) |
| 9 | `sbc_repeats(cfg, posterior, prior, F: repeats, n_cal, num_posterior_samples, cal_n_scales, chi_k_fixed, seed)` (`core/diagnostics/sbc.py:86-90`) | posterior and its prior; the frequency response for a fixed number (`:125-129`) | each repeat about one calibration (`retrain.md:414-416`); the runbook's four runs of ten repeats about 28 hours (`:229-230`) | diagnostic `sbc` / `pooled` or `k<N>`, parents posterior and prior (`sbc.py:243-246`); `sbc_repeats.npz`; one figure (`:223`) | `results.rank_verdict.fraction_passed`, `.per_repeat` (`:241-242`); `results.per_param[]` (`:197-200`) |
| 10 | `core.diagnostics.card_test(cfg, …)`, E3 as 11.4 defines it; today `core/tool/smoke.py:139-238` | bounds file, cell | 336 s, 189 s and 437 s for the three configurations of the last card run (`docs/guide/testing.md:206-210`) | none in the real store; five records in a scratch store; the result goes to the run history | no verdict: it completes or it fails |
| 11 | `probe_drive(cfg, F: t_obs_s, repeats, detune, strengths, seed)` (`probes.py:1338-1342`) | bounds file with a forcing section (`:1391-1397`), cell | 13.8 s on the card at sixteen strengths (`docs/STATE.md`, the piece-6 card run's row: `:1166` at `c45cc4a`) | diagnostic `probes` / `drive`, no parent (`:1627-1629`); `probe_drive.npz`; one figure (`:1621`) | `results.strongest_free_running`, `.weakest_captured`, `.strengths[]`, `.suggested_forcing`, `.oscillating` (`:1600-1616`) |
| 12 | `identifiability_laplace(cfg, posterior, F: n_points, t_obs_s, m, m_noise, rel, seed)` (`identifiability.py:381-385`) | a posterior trained without the frequency response, on a bounds file with a forcing section (`:401-402`); cell; recording length | simulates; not stated | diagnostic `identifiability` / `laplace`, parent posterior (`:489-492`); `laplace_sd.npz` | `results.per_param[].median_sd`, `.frac_identified`, `results.points` (`:486-488`) |

Checks 1, 7, 11 and 12 need a configuration with a cell loaded (check 2 reads no cell:
`core/tool/probes.py:168`, `core/diagnostics/probes.py:1147`), and 1, 2 and 11 one in a fixed mode
(the frequency response on for 1 and 2, off for 11, as the command line builds them:
`core/tool/probes.py:163`, `:193`, `:238`). The check builds its own with
`dataclasses.replace(draft, chi_mode=…).make_config(bounds_path)` (`core/gui/session.py:27-37`) and,
where one is needed, loads the cell on that copy (`core/cli.py:85-99`), so the session's
configuration is never written.

### 9.2 Reading a deciding check

Each rule is a pure function from a manifest to a state, kept in the window. The engine decides
checks 1 and 4 itself (their fields are verdicts). For checks 2, 3, 5 and 6 the pass rule exists
today only in the runbook: the numbers they are judged by are named constants in this part's code,
each with its reason in words, and are on the audit's list as science numbers living outside the
engine (B10).

| # | "Passed" | "Look closer" | "Failed" | a record counts only when |
|---|---|---|---|---|
| 1 | `band_holds` and `drive_holds` both true | either is null (not judged) and neither is false | either is false | `settings.repeats` ≥ 24; `cv_max` 0.2, `snr_min` 3.0, `sup_min` 0.5, `peak_window` 0.1, `phase_max` null; `settings.configured` equals the configuration's (`probes.py:158-164`, `:591-594`) |
| 2 | the three `hold` fields true, the non-finite count 0, and the share within 8 points of 37 % | the three hold and the share is 8 to 12 points away; or the non-finite count is above 0 | more than 12 points away, or any `hold` field false | `settings.num_runs` ≥ 100, `settings.chi_k_fixed` null, `settings.configured` equal but for its `probe_count`, which here holds the range the count is drawn from (`probes.py:1140-1142`) |
| 3 | difference at most 3 points; or a "Look closer" the owner has accepted (9.5) | above 3 and at most 5 | above 5 | the audit counts by row 2; a total short of the whole set reads "Not compared" (C16; 9.5) |
| 4 | `verdict.passed` true | never | false | `results.n_cal` ≥ 2000, `.cal_n_scales` ≥ 200, `.num_posterior_samples` ≥ 1000 (`config.py:276`, `:279`; `orchestrator.py:1876`) |
| 5 | `A1_mean` and `D3_bimodality` read `healthy`, and `counts.invisible` and `counts.nonfinite` are 0 | both read `healthy` but another part is invisible or non-finite | either does not read `healthy` | `settings.rows` ≥ 200000, `settings.n_sweep` ≥ 33 |
| 6 | `results.eigenvalues` is a list | never | null | always |

- Row 2's figures are the runbook's (`retrain.md:300-306`) and assume a 100-batch reading
  (`:309-312`); row 3's limits are U11's (`retrain.md:572-582`).
- "Look closer" is the word B1 gives check 3; using it for checks 1, 2 and 5 is (a choice the owner
  has not seen). It never counts toward the headline. A check that can be run again counts once a
  later run passes; check 3 cannot be, so it has the acceptance of 9.5.
- Row 5's first half is gate 2 as the runbook states it. The second half and its "Look closer" are
  (a choice the owner has not seen): a check named for every part of the input should not read
  "Passed" while its record calls a part invisible (`ablation.py:176-181`).
- "Counts only when": a record made at less than the usual size never decides, whichever way it
  came out. With no other record the row reads "Not run yet" and adds "Run only at a reduced size
  (40 test datasets; the usual is 2,000), which does not count." (A choice the owner has not seen.)
  The card test's calibration is such a record by design (`core/tool/smoke.py:41-43`).

### 9.3 Which records count, and the headline

For the posterior in use, P, let Q be its own prior (`P.manifest.parents["prior"]`,
`orchestrator.py:1432`) and S its simulation set (`parents["simulation"]`, `:1433-1434`). With no
posterior in use, Q is the prior in use. Records are matched by id, never by name: a record can be
renamed (`core/artifacts/store.py:852-872`).

| # | a record belongs when | several | none |
|---|---|---|---|
| 1 | its `config` block (`store.py:357`) equals the configuration's in `model`, `param_keys`, `nd_lows`, `nd_highs`, `rescale_lows`, `rescale_highs` (`core/artifacts/manifest.py:273-281`) | the newest counting record of each cell (`manifest.inputs["cell"]`, `core/artifacts/provenance.py:102-108`); the state is the worst over the cells | "Not run yet" |
| 2 | `parents["prior"]` is Q's id | the newest counting record | "Not run yet" |
| 3 | derived from S and from row 2's record | not applicable | no finished S: "Waiting for the simulations"; no audit that counts: "Waiting for the audit"; a total that covers part of S, or none of it: "Not compared" (C16) |
| 4, 5, 6 | `parents["posterior"]` is P's id | the newest counting record | "Not run yet" |

- Several: the newest decides, and the row says when an earlier one disagreed: "Passed, 12 Oct 2026,
  14:05. An earlier run failed (11 Oct 2026, 09:30)." The runbook fixes the calibration rule
  beforehand so that it cannot be bent to the result (`retrain.md:83-87`); naming the earlier
  failure keeps a third try visible. (A choice the owner has not seen, as is "worst over the
  cells" in row 1.)
- The date is the record's `created` (B22's form); for check 3, the time the simulation set was
  completed (`created` plus `wall_seconds`, `store.py:1575-1577`). Records are named by part 10's
  formatter (10.1).
- Informing checks and tools: 7 and 11 by row 1's rule; 9 and 12 by `parents["posterior"]` (check 9
  shows one line per `variant`); 8 and 10 from the newest `check` entry of that check in the run
  history (6.5).
- A check that cannot apply reads "Does not apply" with the reason. Checks 1, 2 and 3 apply only
  with the frequency response: `probe_band` and `probe_mask` refuse any other mode
  (`probes.py:550-553`, `:1122-1125`), and the generator counts unusable driven recordings in that
  mode only (`core/SBI/pipeline.py:1614-1618`, `:2004`).

**The headline** is the page's only statement at heading size: "n of 6 deciding checks passed",
where n counts the deciding rows whose state is "Passed". For a posterior trained without the
frequency response it reads "n of 3 deciding checks passed; 3 do not apply to this posterior"
(C15, which amends U16's "always n of 6" for this one case; a choice the owner has not seen). The
sentence under it is the first that applies:

| case | sentence |
|---|---|
| no prior in use | "Nothing can be checked yet. Build or load a prior in step 2." |
| any "Failed" | "Failed: Calibration." (the names, comma-separated) |
| any "Look closer" | "Look closer: Unusable driven recordings: audit." |
| no posterior in use, three "Does not apply" | "Three of the six do not apply to a posterior trained without the frequency response." |
| no posterior in use | "The other four are made after simulating and after the fit." |
| posterior in use, a check that applies not passed | "Inference can be run now; 2 deciding checks are still open." |
| every check that applies passed | "This posterior has passed every deciding check."; of 3, "… every deciding check that applies to it." |

The fourth row comes before the fifth because "The other four …" is false without the frequency
response. With no posterior in use the headline stays "of 6": C15 amends it for a posterior only.
"Does not apply" and the sentences are (a choice the owner has not seen; the sixth follows the
mock-up's).

**What other parts read.** One function with no widget, usable before the Checks page is built
(B21), gives the headline and the summary. Part 3 takes the summary as its chain's `checks` and
writes the rail's and Home's words (3.4, 3.6): how many passed, of the 6 or the 3 that apply, step
4 reading "Done" when all of them have (C15); whether any reads "Failed"; whether any reads "Look
closer"; whether a check that applies is still open; whether anything can be checked yet; and a
second line, the first of "1 failed", "1 to look closer at", "4 wait for step 3", "2 not run yet".

For "Before you start" on the Training and Narrowing round pages this part supplies one fact (5.4)
from the states and dates of checks 1 and 2 for the prior in use. Its sentence is part 8's (8.4);
its lead word is C12's (a choice the owner has not seen): "Fine." only when both read "Passed";
"Information." when they do not apply; "Attention." otherwise, a check not yet run included. It
never greys "Start training"; after a "Failed" the start asks one question first, "Cancel"
preselected (8.4).

### 9.4 The page

Heading "Checks of retrain" (the posterior in use), or "Checks before training on retrain_prior",
or "Checks". Under it: "A posterior passes when all six deciding checks have passed."; for a
posterior trained without the frequency response, "… when the three deciding checks that apply to
it have passed." (C15). Then "In use: posterior retrain, trained on prior retrain_prior", which
closes W022 for this step. Then the headline. Then one ruled table (U6), with the group headings
as rows:

| column | content |
|---|---|
| "Check" | the on-screen name |
| "Role" | "Decides", "Informs" or "Tool" |
| "State" | the marker and its word: "Not run yet", "Running", "Passed", "Look closer", "Failed"; also "Waiting for a posterior" (or "… a prior", "… the simulations", "… the audit"), "Not compared" (check 3 alone; C16) and "Does not apply"; for an informing check or a tool that has run, "Done" |
| "Last run" | the date, or blank |

The markers are as part 4 assigns them (4.5). "Done" is (a choice the owner has not seen): "Passed"
would claim a decision those checks do not make.

The page can be opened at any time (U16, B7); with nothing in use it shows the same table with the
reason in each row. It is read when it is shown, when a run ends and when Records changes the store,
and reads manifests only, but for check 3 (9.5). Selecting a row opens that check's page beside the
table, or in its place in the small layout, with "All checks" to return.

From Records (10.2): "Check this posterior" on a posterior asks its one question up front, then
loads it and opens this page (B4, C9); on a calibration or a check of a posterior it does so for
the posterior that record tested and opens that check's page. Records hosts this part's result
block on the record's own page and describes a check's record by the sentence of 9.1, looked up by
`body.diagnostic` and `body.variant`. A posterior from a narrowing round has the same page and six
checks; its calibration records that it was made on the narrowed region (`results.accepted`,
`orchestrator.py:2084-2086`), and the result says so.

#### A check's page

Top to bottom: the name; the sentence of 9.1; the state line ("Passed, 12 Oct 2026, 14:05"); the
button; the result from the record that counts; "Earlier runs". The button reads "Run the
calibration", "Run the audit", "Run the frequency-band check", "Run the input-use check", "Run the
identifiability check", or "Run this check"; once a record exists, "Run again". Check 3 has no such
button (9.5). It is greyed, with the reason beside it (5.6), when:

| reason shown | rule |
|---|---|
| the live-run sentence of 5.6, with its link | the one-run rule |
| "Needs a posterior: train one in step 3, or load a saved one." | checks 4, 5, 6, 9, 12 with none in use |
| "The prior in use is not the one this posterior was trained on (retrain_prior). Load it in step 2." | checks 4 and 9: the engine refuses the pair (`orchestrator.py:1949`, `core/diagnostics/sbc.py:139`) |
| "This posterior was trained without keeping its simulations, and this check reads them." | check 5: `parents` holds no `simulation` (`ablation.py:103-109`), or that set is gone or unfinished |
| "This posterior was trained without the rotation, so there are no directions to read." | check 6: `body.transform.V` is null (`identifiability.py:64-69`) |
| "Applies only with the frequency response." | checks 1, 2, 3, and a fixed number in check 9 |
| "Applies only to the Nadrowski model." | check 7 (`core/diagnostics/feature_sets.py:98-103`) |
| "Needs a bounds file with a forcing section." | checks 11, 12, and 7 without the frequency response (`feature_sets.py:115-121`, `probes.py:1391-1397`) |
| "Does not apply with the frequency response." | check 12 (`feature_sets.py:80-89`) |

Checks 5 and 6 in the two "trained without" cases read "Not run yet" and count as not passed. The
rules above answer, before the click and in the window's words, the refusals these checks raise
with no setting named (`probes.py:553`, `:1125`, `:1397`; `feature_sets.py:89`, `:103`, `:121`),
which stay in the engine as the second line. "Frequency response" and "forcing section" are part
8's words (8.2, 8.3).

#### The form

"Run …" opens the form in place, built from part 5's pieces. Check 4's three sizes are the plan's
`Calibration` values (7.1.3): its form exposes `plan_fields()` and shows each value's origin.
Nothing on a form is remembered between launches (U15); a "Cell file" picker opens on the cell picked
in step 5, or on nothing. A blank or bad number is said under its field and blocks (5.7), by the
rules the stage applies again (`core/refusals.py:296-305`).

| # | settings (label → argument, opens at) | "Rarely changed" | not offered in the window |
|---|---|---|---|
| 1 | "Cell file" | "Recording lengths (s)" → `lengths`, blank gives five from the shortest expected to this cell's longest in training; "Simulations per point" → `repeats`, 24; "Seed" → `seed`, 0 | `multipliers`, `drives`, `cycle_caps` and the five thresholds: they change what "holds" means |
| 2 | "Batches to audit" → `num_runs`, 100 | "Simulations per batch" → `run_size`, 32; "Seed" → `seed`, 0 | `chi_k_fixed`: a fixed-number audit is not the reference check 3 needs |
| 4 | "Test datasets" → `n_cal`, 2,000; "Operating points" → `cal_n_scales`, 200 | "Samples per dataset" → `num_posterior_samples`, 1,000; "Seed" → `seed`, blank | none |
| 5 | none | "Simulations read" → `rows`, 200,000; "Points per sweep" → `n_sweep`, 33 | none |
| 6 | none | "Weakest directions totalled" → `n_worst`, 3; "Largest contributions shown" → `top_n`, 4 | none |
| 7 | "Cell file"; "Recording length (s)" → `t_obs_s`, blank | "Simulations per comparison" → `m`, 32; "Simulations for the noise level" → `m_noise`, 128; "Step, as a share of each range" → `rel`, 0.02; "Seed" → `seed`, 0 | `min_valid`, `zero_tol`, `noise_eps` |
| 8 | "Batches" → `num_runs`, 20; "Epochs", 10 → `max_num_epochs` = 9 | the two networks, read-only | everything else |
| 9 | "Fixed number of driven recordings" (C14), tick "Mixed, as in training" → `chi_k_fixed`: opens ticked, which sends `None`; else a number from 1 to the slots (8.2); "Repeats" → `repeats`, 10; "Test datasets" (unit "per repeat") → `n_cal`, 2,000 | "Operating points" → `cal_n_scales`, 200; "Samples per dataset" → 1,000; "Seed" → `seed`, 0 | none |
| 10 | "Cell file"; "Recording length (s)" → `t_obs_s`, 1.0 as the command line (`core/tool/smoke.py:172`) | "Batches" → `num_runs`, 4; "Simulations per batch" → `run_size_cap`, 32; "Test datasets" → `n_cal`, 40; "Epochs", 6 → `max_num_epochs` = 5; network width and depth, as the Training page's form holds them; "Seed" → `seed`, 0 | `checkpoint`, `save`, `prior_ref`, `resume`, `new_run`, `stages` |
| 11 | "Cell file"; "Recording length (s)" → `t_obs_s`, 5.0 | "Simulations per strength" → `repeats`, 16; "Drive frequency, × the cell's own" → `detune`, 1.4; "Strengths" → `strengths`, blank gives sixteen from 0.01 to 20 plus the set amplitude; "Seed" → `seed`, 0 | the four thresholds |
| 12 | "Cell file"; "Recording length (s)" → `t_obs_s`, blank; "Points across the prior" → `n_points`, 6 | as check 7 | `min_valid`, `sd_identified` |

- An "Epochs" field holds epochs trained. The window sends one less, because the engine's limit N
  trains N+1 (`orchestrator.py:835-836`; W035), and a result divides by `training.epochs_trained`,
  never by the typed number. Its rule is at least 2: the limit sent must be at least 1
  (`orchestrator.py:953`). It is these two forms' own row, not `label("max_num_epochs")`: that key
  keeps Training's "Epoch limit", which holds the limit itself (8.4), and the rule here leaves the
  stage nothing to refuse under it.
- The table names arguments; a label belongs to the key the stage refuses the argument under, and
  `fields.py` holds one label for a key (5.7). So a setting reads the same wherever it is shown
  ("Batches" and "Simulations per batch" as in step 3; "Test datasets" on checks 4, 9 and 10), and
  a unit in brackets is the row's unit, not part of its label (U16): `t_obs` keeps step 5's
  "Recording length" and `cell` its "Cell file" (8.5; ruling D2). The three probe checks refuse under keys of their own: `probe_seed`,
  `band_repeats`, `probe_lengths` (`probes.py:563-565`); `mask_num_runs`, `mask_run_size`
  (`:1129-1130`); `drive_t_obs`, `drive_repeats`, `drive_detune`, `drive_strengths` (`:1406-1425`).
- The recording length of checks 7 and 12 opens blank and must be typed; the stage's own fallback
  of 1 s (`identifiability.py:413`, `:875`) is not used (a choice the owner has not seen).
- The audit opens at 100 batches, the runbook's size (`retrain.md:278`), not the engine's 12
  (`probes.py:1091`); the settings marked "not offered" stay on the command line. (A choice the
  owner has not seen.)
- Check 4's "Seed", blank meaning one is drawn and recorded as today (`validate_tab.py:76`), is
  B23's (a choice the owner has not seen): a calibration can be repeated only by its seed
  (`retrain.md:603-605`). It is never remembered.
- Row labels come from `core/gui/fields.py` through `label(key)`, as today (`validate_tab.py:39-40`).
  Each key that has no control today moves from `None` to step 4 and a label (`fields.py:71`,
  `:179-186`) in the task that gives it its field (5.7), which first rewrites the pinned set of
  keys with no control (`tests/test_nav_and_gating.py:1619-1630`) and the pin on the Validate tab's
  rows (`:3629-3649`).

"Before you start", as facts (5.4): what will be simulated, in counts ("200 batches, holding 2,000
test datasets, then 1,000 samples for each"); how long this check took at these sizes the last time
on this computer, from the run history, or "Not timed on this computer yet"; part 7's memory lines
(7.2.3) where the check simulates on the card, so not on checks 5 and 6 (U9, C13); "Nothing else
can run until it ends"; and what is written ("One record, under Records", or for checks 8 and 10
"Nothing is kept but the result"). Checks 5 and 6 say "Reads records and simulates nothing; takes
seconds." Stating a time only from this computer's own history is (a choice the owner has not
seen): the times in 9.1 are the owner's card.

Name and note are 5.5's, with the kind `calibration` for check 4 and `diagnostic` otherwise; a name
is optional. Checks 3, 8 and 10 take neither. A check opens its writer before it spends anything
(11.8), so a taken name is refused at the click and costs nothing; E7 is not needed here.

While a check runs, its page shows part 6's live view. The stop button names the run as the Run
button did (6.6): "Stop the calibration…", "Stop the audit…", "Stop this check…". The question (B2)
repeats it, with "Keep it running" preselected and "Stop and discard it" drawn as destructive
(B27); the two answers are (a choice the owner has not seen). Its text says a check writes its
record only when it finishes, so nothing is kept (`store.py:341-344`); a stopped or failed check
leaves the state as it was. Stage names: check 4, "Simulating the test datasets", "Rank test",
"Joint coverage test", "Saving"; checks 8 and 10, the stages of 9.6 and 9.7; every other check, one
stage, "Measuring". Check 4's stages begin, for part 6's `spec` (6.2): the first at the start,
counted by the progress row `Generating training data` (`pipeline.py:1630-1631`); "Rank test" at
the record `[patho] run total: …`, which ends generation in every mode (`pipeline.py:1998-2003`);
"Joint coverage test" at `SBC uniformity checks:` (`orchestrator.py:1987`); "Saving" at
`TARP: ATC=…` (`:2039`). Check 8's begin at a training's lines (6.3.1). (Read from the code, not
run.) Check 9's ranking reports no progress and cannot be stopped until it ends
(`sbc.py:172`, `:188`; W095, left for the audit), so its stage line says: "Ranking repeat 3 of 10:
no progress is reported during this part, and Stop takes effect when it ends."

#### The result

One block, from a manifest and its folder to a widget, shown on the check's page and hosted by
Records. It is drawn from the record, not from the run, so it reads the same after a restart
(W057). Three parts, in the result area of 5.9.

**A verdict sentence**: a message row (4.6), in the window's words, built from the fields of 9.1 and
never from the engine's own sentences, which use the engine's names (`probes.py:624-646`,
`orchestrator.py:1856-1859`):

| # | example |
|---|---|
| 1 | "Frequency band: passed for cell master_spont_tier1. Every frequency inside the band (0.03 to 0.3 times the cell's own) can be measured at every recording length tried, and the drive amplitude 0.15 does not take over the cell." |
| 2 | "Unusable driven recordings, audit: passed. 37.9 % of 28,800 driven recordings simulated from prior retrain_prior were unusable. The expected share is 37 %; within 8 points is fine. Every one was too short or too slow to hold two cycles of the drive." |
| 3 | "Unusable driven recordings, simulation set against the audit: passed. 40.1 % in the simulation set; the audit read 37.9 %. The difference is 2.2 points: within 3 is fine, up to 5 look closer, beyond 5 stop." |
| 4 | "Calibration: passed. All 13 parameters pass the rank test, and the joint coverage test passes. Made on 2,000 test datasets at 200 operating points (pairs of time scale and recording length), seed 1840273." |
| 5 | "Use of each part of the input: passed. The network responds to 47 of the 50 parts of an observation's summary; 3 never vary in the training simulations; none is invisible to it." |
| 6 | "Identifiability of the posterior: passed. The record holds how strongly each of the 13 directions is pinned down: the strongest 1.2e6 times the weakest, about 4.3 effectively pinned down. This check passes when those figures are on record; the table says what they are." |

**A table**, three rules, numbers right-aligned, every row shown (U18, long lists in full):

| # | rows | columns, from |
|---|---|---|
| 1 | one per frequency, then one per drive amplitude | multiple of the cell's own frequency, inside the band, result, reasons (`results.frequencies[]`, `results.drives[]`) |
| 2 | one per cause | count, share (`results.causes`); below, the per-batch mean and spread (`results.per_batch`) |
| 3 | simulation set; audit | share, driven recordings counted, batches, per-batch median and range |
| 4 | one per parameter, then "All together" | "Parameter", "Unit", "p-value", "Result", "Remark" (`verdict.parameters[]`, `verdict.tarp`; the record holds no unit, so "Unit" is the configuration's, as 8.5 reads it; "Remark" takes from the record the assumed-input mark, `verdict.caveat` for the time scale and `verdict.alpha` for the last row, in the window's words); footnote "Rank test: a parameter passes at p = 0.0038 or more (0.05 divided by 13)." from `verdict.threshold`; then `results.informativeness` as two rows, labelled a reading |
| 5 | one per part, weakest first | response, against the typical part, range in training, reading (`results.channels[]`) |
| 6 | one per parameter; then one per direction | share in the weakest directions, reading (`results.per_param[]`); strength, largest contributions (`results.directions[]`) |
| 7 | parameters with no local information; pairs that move alike | `results.unmeasurable`, `results.degenerate_pairs`; above, "Driven recordings at the low end of the band hold two cycles from 3.1 s." |
| 9 | one per parameter | median p-value, lowest, share of repeats below 0.05 (`results.per_param[]`); above, "7 of 10 repeats pass the rank test. This is the rank half only; the joint coverage test is Calibration's." (`results.rank_verdict.scope`) |
| 11 | one per strength | force in the cell's unit, share of its own oscillation kept, reading; below, the suggested forcing lines, to copy (`results.suggested_forcing`; the engine never writes them, `probes.py:1357-1359`) |
| 12 | one per parameter | spread at each point, median, share of points pinned down (`results.per_param[]`) |

The words of a "result", "reading" or "reasons" column are the window's, built from the record's
fields; the record's own strings there are the engine's and are not shown (`probes.py:347-357`;
`ablation.py:165-187`; `identifiability.py:158-165`). Checks 8 and 10 are in 9.6 and 9.7.

**The figures** the record holds (`manifest.figures`), in the figure viewer of 5.9, captioned by
the window; for check 4, "Rank test, cumulative", "Rank test, histogram", "Joint coverage test".
They are the engine's, so they are light pictures in both themes (U23, B18), and nothing in them
changes but their lines, as E12 has it (U4; 11.14). Checks 2, 5, 6 and 12 have none.

Then the next step (U16): the next deciding check not yet passed, by its button; when every one
that applies has passed (6 of 6, or C15's 3 of 3), "Go to Inference".

### 9.5 After simulating: the automatic comparison (check 3)

For: gate 3, which has no place in the window today (W034). The window makes it, with no engine
call and no form.

**The training's total.** When generation ends, also when a continued run finds its set complete
and simulates nothing, the engine logs one record,
`[chi] masked probes: M of T (P%) over n batches, <scope>; per batch …`
(`core/SBI/pipeline.py:224-236`, `:2004-2015`). In a training or a narrowing round part 6 hands
`M`, `T` and the scope of the first such record to this part's function (6.3.5), which computes the
share from the two counts, not from the rounded P. The scope
`every committed batch of the simulation cache` or `this process only (no simulation cache)` is the
whole set; `only the batches this process generated…` (`pipeline.py:2012-2013`) covers part of it. A
calibration logs a line of its own (`retrain.md:588`), which is why only a training's is taken.

**The audit's reading** is `results.masked_fraction` of the record that counts for check 2 and the
run's prior (9.3): one manifest, read from the store at that moment.

**The three bands** (U11, B1), on D, the absolute difference of the two shares in points, unrounded:

| D | check 3 reads | handed to the run's page |
|---|---|---|
| at most 3 | "Passed"; the run's row, "Fine" | the health row's sentence: "Unusable driven recordings 40.1 %. The audit read 37.9 %; within 3 points is fine." |
| above 3, at most 5 | "Look closer" | the sentence, and one "Attention" message: "The simulation set and the audit differ by 3.8 points. Look closer before leaving the fit to run: step 4, Checks, shows the per-batch figures of both." |
| above 5 | "Failed" | the sentence, then "The advice is to stop the fit and find out why. The simulation set is complete, so stopping loses only the hours of fitting so far."; buttons "Stop training…" (the usual question, U12) and "Keep training" (U7's one name for it). The run's page and the rail's foot say "needs you" until one is chosen. The window never stops the fit |

The run's health row reads as the check does in every case but the first (C16); the announcement
and the history are part 6's (6.3.5, 6.5). The answer beyond 5 and its time show on check 3's page
("You chose to keep training on 2 Oct 2026, 09:14"); an answer never changes the state.

**"Accept this difference"** (B1; a choice the owner has not seen). On check 3's page a "Look
closer" shows the table of 9.4 and this button. Pressing it, during the run or after it, writes the
acceptance with its date into the history of the training that made the set (`history.runs_for`,
6.5). From then the check reads "Passed", and its state line and verdict sentence say "A difference
of 3.8 points was accepted on 3 Oct 2026, 09:14." The acceptance is of that audit, by the record id
part 6's `health` entry holds (6.5), and lapses when another comes to count. A set made from the
command line has no history to hold one, and the button is greyed with that reason. (How the check
reads afterwards, the lapse and this last case: choices the owner has not seen.)

**No audit, or part of the set.** Neither is compared and neither raises "needs you" (B1; 6.3.5);
each gets one "Attention" message, and the check and the run's row read the same word (C16). With
no audit that counts, "Waiting for the audit": "There is no audit of prior retrain_prior to
compare with. The simulation set holds 40.1 % unusable driven recordings. Run the audit in step 4,
Checks, when this run has ended." A total that covers part of the set, or none of it (the record
then reads `none counted` and holds no share, `pipeline.py:231-232`), reads "Not compared" and
says which batches were counted. It never counts toward the headline and cannot be accepted: no
difference over the whole set exists to accept. (A choice the owner has not seen.)

**Afterwards.** With no run live, the Checks page computes check 3 afresh each time from the
engine's stored counts:
`core.SBI.training_checkpoint.peek(store.path("simulation", S))["chi_masked"]`, a list of
`[unusable, total]` per batch (`core/SBI/training_checkpoint.py:162-170`; written at `:391-394`,
carried to the finished set at `:402-403`), summed, and used only when its length equals
`batches_done`. The set's own record does not hold them (`store.py:1591-1593`). When S is gone or
holds no counts, the same line is read from the posterior's text (`store.read_log`,
`store.py:757-799`; `retrain.md:698`). While a run is live no state file is read (the engine
replaces it at each save, 6.8): check 3 shows what that training's history holds. Before the
posterior exists, S is the set the run history names for the live or last training.

For a posterior from a narrowing round the same rule is applied to the round's own set, and the
page adds "The audit measured the whole prior; this round simulated inside a narrowed region." (A
choice the owner has not seen.)

### 9.6 Timing of a short fit (check 8)

For: pre-flight step 2 of the runbook (`retrain.md:154-249`), which the window cannot do today
(W092). It uses only what `build_posterior` already takes (`orchestrator.py:786-802`).

| what | how |
|---|---|
| a store that is not the real one | `ArtifactStore(tempfile.mkdtemp(prefix="prism_check_"))`, passed as `store=`. The process default is never swapped: `use_store` (`store.py:1617-1625`) would hand the scratch store to every screen. The folder is removed when the check ends, however it ends |
| the prior | the prior in use, passed as the object. The posterior's writer records its id and looks nothing up (`orchestrator.py:1432`), nor does the read-back (`store.py:1302-1422`); the one store lookup of a prior is inside the narrowing branch (`orchestrator.py:1172`). This was read from the code, not run: the check's first task is a test of it |
| no saving of simulations | `checkpoint_every=0` (`orchestrator.py:837-840`), `resume="auto"`: no simulation set is read or written, in either store |
| an epoch limit | `max_num_epochs` = "Epochs" minus one |
| the size | `num_runs` from the form; `run_size_cap` as the Training page's form holds it |
| the networks | first `hidden_features` × `num_transforms` as the Training page's form holds them, then the built-in 128 × 8 (`config.py:363-364`); one fit when they are equal |

The stage line reads "Rotation", "Simulation", "Fit" for each network in turn. Each fit's span runs
from the record that ends generation (9.5's; without the frequency response, the stage change part
6 reports), on the window's clock, to the returned posterior's `created`, the moment its writer is
made as the fit ends (`orchestrator.py:1420`; `store.py:850`), not to the return of the call, which
also holds the writing and reading back of the record. Seconds per epoch is the span over
`training.epochs_trained`. That is the runbook's definition, whose span ends when the record's
folder is created (`retrain.md:169-199`), so its figures stay comparable. E1's per-epoch records, as part 6 reads them (6.3.1), give a second figure, the
median time of one epoch.

The result page: "Timing of a short fit: done. One epoch of the 256 × 10 network took 7.4 s on
40,960 simulations; of the 128 × 8 network, 5.2 s." Table: one row per network with epochs timed,
span, seconds per epoch, median epoch. Then the projection for the budget the Training page's form
holds, by the runbook's rule (`retrain.md:203-207`): seconds per epoch × (planned simulations ÷
timed simulations) × E ÷ 3,600, where E is Training's "Expected epochs" as B9 resolves it, the
value part 8 exposes (8.4; C28). It reads "About 134 hours of fitting, if the fit takes 130 epochs
(the last training of this kind, retrain_aug)." With no E the page gives the seconds per epoch
alone and says "There is no epoch count to project with. Type one in "Expected epochs" in step 3."
The check informs: it has no limit and no pass, and the runbook's stop points (`:234-241`), which
belong to the owner's budget, are not shown (a choice the owner has not seen).

The result goes into the run history's `check` entry (6.5), which part 6's `history.estimate`
reads as the estimate before a training's first epoch ends (6.3.3).

### 9.7 The card test (check 10; E3)

For: a screen for the test that runs every stage end to end at very small sizes. The engine step
(`core.diagnostics.card_test`, returning a `CardTestResult`), what moves out of
`core/tool/smoke.py`, and its tests are part 11's (11.4).

The screen makes one call (B25), with the form's arguments (9.4), `store=` a scratch store as in
9.6, and an `on_stage` that only emits a signal and never raises. The five-command gate, with its
continue and its refusal legs (`docs/guide/testing.md:104-139`), stays a command-line recipe (B25).

"Before you start" says: "This builds a small prior, trains a small posterior, calibrates it and
runs one inference on a simulated cell: four real stages, at very small sizes, on the graphics
card. It takes minutes (5 min 36 s here last time). Nothing else can run meanwhile, and nothing it
makes is kept." The stage line reads "Prior", "Training", "Calibration", "Inference", driven by
`on_stage`. The result, kept in the history's `check` entry: "Card test: done. Every stage ran on
the graphics card in 5 min 36 s." or "Card test: failed in Training."; a table of stage, seconds
(`CardTestResult.seconds`), and seconds the last time; the unusable-driven-recordings share of its
training, from part 6's reading of the run's text because the result object does not carry it,
with "At this size a share within about 12 points of 37 % says nothing either way."
(`docs/guide/command-line.md:915-921`); the number of batches that had to be split for lack of
memory (6.3.1; `pipeline.py:825`); and "This shows that the chain runs. It says nothing about
calibration: at these sizes the test has no power." Its calibration's verdict is not shown.

### 9.8 Refusals that say where to fix it (E6)

The edits and their tests are part 11's (11.7), to the extent B14 rules. What reaches this page:

| refusal | field | on this page |
|---|---|---|
| the input-use check's five plain errors (`core/diagnostics/ablation.py:41`, `:105-109`, `:115-119`, `:130`, `:132-134`; W093) | `posterior` | shown as a refusal (5.7: "Check your inputs", B2), not as a failed run |
| check 12's plain error when the posterior carries no training prior, raised before anything is spent (`core/diagnostics/identifiability.py:238-241`; B14, an extension the owner has not seen) | `posterior` | the same |
| the two training refusals a continue policy reaches (`core/orchestrator.py:958-965`, `:1110-1112`; W094, W099) | `resume` | not raised from this page: checks 8 and 10 send `resume="auto"` |

The fix sentence names where the posterior in use is chosen: `CONTROL["posterior"]`
(`core/gui/fields.py:127`, the Posterior tab today) is part 8's entry, step 3's "Saved posterior"
(8.1). E6 changes the type and the field only; the sentences are reworded by E11, in the last stage
(U22). Check 7's two plain errors raised after it has simulated (`identifiability.py:560`, `:621`)
are outside E6 and show as a failed run.

### 9.9 What must never be shown

- No heading-size, boxed or coloured "PASS" or "Passed" for one check. A check's verdict is a
  message row that begins with its name.
- No sentence that a posterior "passed" below 6 of 6 (C15's 3 of 3). Today's line "Calibration
  verdict PASS" (`validate_tab.py:87-90`) is dropped, and the engine's `[verdict] PASS`
  (`orchestrator.py:1856`) is seen only under "Show the full text".
- Under a passed calibration with others open: "Calibration alone does not show that the posterior
  is sound: 2 deciding checks are still open."
- No "Passed" or "Failed" as the state of an informing check, a tool, or a record made at reduced
  size. A card test that did not complete says so in its sentence ("Card test: failed in
  Training."), which reports a run that failed, not a verdict; its row reads "Done" only from an
  entry that completed.
- No pass for check 6 without its clause that the figures are on record, and none for check 3 after
  an acceptance without the words that the difference was accepted.

### 9.10 Kept, changed, and where it lives

Kept from `ValidatePanel`: the click-time rules (`validate_tab.py:56-59`); the call (`:75-77`),
which gains `name`, `note`, `num_posterior_samples` and an optional seed; the gate on a posterior
and a prior, never on the forcing prior (`:21-22`, `:92-94`); rows named from the control table
(`:39-40`). Dropped: the group box, the unwrapped caption (`:35`), the summary line (`:79-90`). The
tests that reach `cal_n`, `cal_scales`, `_validate` and `_on_calibration` by name are rewritten
first.

New, under `core/gui/checks/` (the names are a choice the owner has not seen): `catalogue.py` (the
twelve entries of 9.1), `readings.py` (9.2, with B10's constants), `lineage.py` (9.3, with the
headline and the summary), `scratch.py` (9.6), all without Qt and tested as plain functions; and
`page.py`, `form.py`, `result.py`. The suite is `tests/test_checks.py` (part 13), which asserts the
headline for every combination of states on the function alone.

### Choices in this part the owner has not seen

Most consequential first.

- For a posterior trained without the frequency response, three deciding checks read "Does not
  apply", the headline reads "n of 3 deciding checks passed; 3 do not apply to this posterior"
  where the owner decided "always n of 6", and the posterior has passed when those three have;
  with no posterior in use the headline keeps "of 6".
- A pre-run deciding check that failed does not grey "Start training", but the start then asks one
  question first with "Cancel" preselected; a check not yet run is led by "Attention" and one that
  does not apply by "Information", and neither blocks.
- After simulating, a difference of more than 3 and up to 5 points counts toward the headline only
  once the owner presses "Accept this difference", kept with its date in that training's history;
  the check then reads "Passed" and says the difference was accepted.
- Every deciding check's in-between state reads "Look closer", and a check that can be run again
  counts only once a later run passes.
- A check made at less than its usual size (a calibration on 40 test datasets, an audit of fewer
  than 100 batches) never decides: the row stays "Not run yet" and says so.
- The input-use check passes only when the two parts the runbook names read healthy and no other
  part is invisible or non-finite; with another part invisible it reads "Look closer".
- When simulation ends and that prior has no audit, the check reads "Waiting for the audit" and the
  run's page says so without asking for the owner.
- A total of unusable driven recordings that covers only part of the simulation set, or none of
  it, reads "Not compared": it never counts toward the headline and cannot be accepted.
- An accepted difference lapses when a newer audit comes to count, and cannot be given for a
  training that was not started from the window.
- When a check has been run several times the newest run decides, and its row says when an earlier
  run came out differently.
- When the frequency band has been checked on several cells, the worst result among them decides.
- For a posterior from a narrowing round the comparison with the audit is made the same way, with a
  sentence that the audit measured the whole prior.
- The timing check projects the full fit from Training's "Expected epochs" and shows no stop point
  and no pass.
- The calibration form has a "Seed" field under "Rarely changed": blank means one is drawn and
  recorded, and it is never remembered.
- The audit's form opens at 100 batches (the runbook's size, not the engine's 12), and the
  thresholds that define "holds" in the frequency-band check and the drive study are not offered.
- A check's form states how long that check took last time on this computer and shows no built-in
  times from the owner's card.
- The two checks that only read records show no memory lines, where the owner decided that every
  "Before you start" shows them.
- The timing check and the card test run in a scratch folder that is removed afterwards; only their
  result is kept, in the run history.
- The recording length of the two identifiability checks that simulate opens blank and must be
  typed.
- Informing checks and tools read "Done" once they have run, never "Passed" or "Failed".
- The third identifiability method's error for a posterior with no training prior becomes a refusal
  that points at the posterior.
- The names of seven checks, the longer forms of two names from the mock-up, all twelve one-sentence
  questions, the four group headings and the three role words are as written in 9.1.
- The wording of the sentence under the headline, of the states that wait or do not apply, of each
  check's verdict sentence, greyed reason and form labels is as written in 9.3 and 9.4.
- A check's stop button names the check as its start button did; the answers are "Keep it running"
  (preselected) and "Stop and discard it".
- The Checks code lives in small files of its own inside the window's package.

### What this part fixes

- W022, for this step: the page names the posterior and the prior in use (9.4).
- W034: gate 3 gets a place, a comparison with its limits, and a lasting source (9.5).
- W035, for the checks' forms: an "Epochs" field holds the epochs trained (9.4).
- W057: the calibration states what it will simulate before the button and keeps its verdict,
  table and figures on its page (9.4).
- W092: the timing of a short fit runs from the window (9.6).
- W093: the input-use check's plain errors become refusals (9.8, with part 11), and its need for
  kept simulations is said before the click (9.4).
- W094, W099: the two "must continue" refusals carry a field (9.8, with part 11); the control they
  name is part 8's.
- W188: every pre-run check and the card test has a screen (9.1, 9.7).
- W189: the calibration at a fixed number of driven recordings has a screen (check 9).
