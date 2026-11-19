# The code audit: the plan

> **For agentic workers:** the loop is the owner's leaner way (11 November 2026): one Opus builder per task from a
> dispatch, two readers of its diff (Sonnet, Opus), one fix round, a fresh fast gate on the final head with the
> real-screen run; the three-reader review once at the end, one fix wave; then the exit. Each task's dispatch is
> written from this plan into `.superpowers/sdd/2026-11-17-code-audit/briefs/`.

**Goal:** close what still holds on the code audit's list before the retrain from the window, without moving the
science, a simulated row, the simulation set's identity or what an uninterrupted fit computes.

**Architecture:** the list (`docs/superpowers/notes/2026-10-29-the-code-audits-list.md`) was sorted against the code at
`b245f20b` by two readers into one table, `.superpowers/sdd/2026-11-17-code-audit/reviews/sorted-table.md` (249 rows:
102 to the audit's tasks, 20 to the last stage, 72 left with their reasons, 8 near the science and told, 47 closed).
The tasks go by area: what removed code left, the words in comments, the store and the kept training, refusals and
the command line, the memory planner's edges, the window's run controller, the tests, and the retrain's window route.
A measurement of the time-stride lead comes first if the owner says yes.

**Written:** 2026-11-17, by the controller, from the sorted table. **The owner's answers (2026-11-17, 10:36 PST):**
Q1 "Measure it first" (A0 runs first); Q2 "All eight" (pyqtgraph, imageio, imageio-ffmpeg, h5py, pandas, pyarrow,
antropy and pybispectra leave `requirements.txt`); Q3 "Yes, all sixteen"; the plan "Yes, build it".

## Global Constraints

- THE OWNER'S LINES: the science (the models, the simulation, the features, the training and calibration methods,
  every setting in `core/config.py`) does not change; NO TASK EDITS `core/config.py` (not even its comments); what a
  simulated row holds and the simulation set's identity do not change (`SimulationIdentity`, `training_checkpoint`'s
  `_canonical`, `identity_digest`, `resolve_dir` untouched; `Artifacts/simulations/eda5c2a1e468` stays usable); a fit
  never interrupted computes what it computes today; `core/Simulator`, `core/Solvers` (but for one comment, A2) and
  `core/Models` are untouched; `CLAUDE.md` unedited (edits go to the owner at the exit, on their exact words); nothing
  under `Artifacts/` and not `C:\Users\J\prism-retrain-console.log` is written, renamed or deleted; no training of
  real size.
- NO TASK EDITS THE NAMED TRAINING CODE (`core.SBI.fit_state.FIT_CODE`, seventeen functions in `core/SBI/train.py`,
  `embedded_network.py`, `chi_encoder.py`, `summaries.py`, `chi.py` and `core/orchestrator.py`'s
  `build_embedding_net` and `expected_forcing_dim`), a comment or docstring included: the seventeen digests stay as
  `stage-5/gates/digests-at-exit.json` records them (the fence after every task), and
  `test_the_named_training_code_is_as_recorded` stays green.
- A CHANGE THAT COULD MOVE A NUMBER is shown not to, in its report, on a tiny seeded run before and after (the same
  seed, the same rows, the posterior's network state and the record's numbers equal). The card smoke gate runs after
  A3 and A5; the diagnostic card after A5.
- EVERY ENGINE CHANGE (any file under `core/` outside `core/gui/`) is registered in the ledger's "## Engine changes
  landed by the audit" in the task that makes it, and on the code audit's list where an item names it.
- The window's words and the engine's sentences are the last stage's: a task writes a new refusal's sentence plainly,
  in the house's form, and changes no sentence a person reads today unless its item says so.
- A refusal is a `core.refusals.Refusal` with a registered field; `core.refusals.FIELDS` holds 98 keys and a task that
  needs a new one says so in its report (none is expected).
- THE FIRM REQUIREMENT of the window: nothing of the run page hidden under the form at the least size and every width
  above it (`test_nothing_of_the_run_page_is_cut_on_a_step_page_at_any_width`,
  `test_nothing_of_the_run_page_is_hidden_at_150_percent_display_scaling` green).
- The reference rule (`tests/test_source_hygiene.py`): no shippable file cites the working record or its labels; the
  audit's item numbers, task names (A0 to A8) and question numbers are never written into code, tests or the guide.
- The carried-over list (`.superpowers/sdd/2026-10-06-window-redesign/carried-over.md`, 1525 rows, missing 0) stays
  current: a removed or renamed test's row is closed naming its successor; a new test is born.
- A page never holds a toolkit item across its list's rebuild; a test that reads state the whole process shares takes
  a reading first or works on a stand-in; a guard's or a test's message never prints what it read from the computer.
- Git: the local `main`, one commit per logical step, a short subject, never amended, the Co-Authored-By trailer; a
  builder has the tree to itself; every suite a commit edits, and every suite that imports a module it changes
  directly, runs WHOLE before the commit; the broad set (every suite that reads a file the task edits) runs ONCE per
  task, before its last commit that changes code (the owner, 2026-11-17 15:33 PST: "Yes, once per task").

## Review Focus

1. A name removed as dead that the window reaches only when a page opens, or by a string (a lazy import, `getattr`,
   a test's `monkeypatch.setattr` by text): the window must still open every step, Records and Settings.
2. A comment or line wrap that changes a value: for every file changed for words or wrapping alone, the parsed tree
   (`ast.dump` without positions) is equal before and after, and every string a person reads is unchanged.
3. A refusal moved earlier that refuses what works today: every file under `Resources/Bounds`, `Units` and `Cells`
   reads as before, every command of the card smoke gate and the diagnostic card still runs, and a plan saved from the
   window still fills its forms.
4. The store's change for a finished set that is reused: proven on a copy in a scratch store, never on `Artifacts/`;
   the set's folder name, its identity and its rows unchanged.
5. A run that ends with no outcome: the page unlocked, keep-awake and the reason against shutting down released, the
   history closed, and Stop disabled, on the run's every kind (a training, a round, an inference, a check).

---

## The owner's questions (asked with this plan, each in plain words, with a recommendation)

- **Q1, the time stride (near the science).** On 5 October a reading of the code found that the simulation keeps
  every n-th point of the model's fine steps, with n rounded to a whole number, while every later step treats the kept
  points as exactly the recording's spacing apart. Reproduced that day on the CPU: a simulated cell then behaves as a
  cell at another time scale than the one its row is labelled with, and the drive is out of tune with the lock-in that
  reads the response (a response equal to its drive, true answer 1, read 0.625 at the master cell's time scale, and 0
  at 25 and 30). Not yet shown: the size of the effect on rows like the finished set's, a whole training batch,
  whether forced mode shares it. If it stands, the finished set cannot be trained on as it is, and any fix changes the
  simulation (the set simulated again). Two smaller leads were never reproduced (the drive's clock starting before the
  discarded start of a run; the solver's step differing between segments), and the 5 October plan's check of the math
  against a PDF of the models from the owner is not in this chat's prompt. **Recommended: measure it first (A0), a
  measurement only, before the retrain; any fix stays the owner's.**
- **Q2, the packages.** Five packages only the removed pages used are pinned in `requirements.txt` (pyqtgraph,
  imageio, imageio-ffmpeg, h5py, pandas); nothing in PRISM imports them, and no package PRISM uses needs them. Three
  more pins are in the same state: pyarrow (pandas' helper), antropy and pybispectra (unused since the owner's early
  summary-statistics work). Dropping a line changes only what a fresh install fetches; nothing is uninstalled from
  `biophys-env`. **Recommended: drop the five and pyarrow; keep antropy and pybispectra unless the owner drops them.**
- **Q3, what a stage refuses and records.** Sixteen small engine changes. *Refusals*: a crash, a bare error or a
  refusal that comes after time is spent becomes a refusal before anything is spent, naming the setting at fault (a
  training told to load with no prior; a kept narrowing round resumed through plain training, which drops its region;
  a tier-1 box whose units file names no force, refused today only after a nine-minute sweep; a count given as
  10000.5, cut to 10000 today without a word; a bounds file's line the reader cannot read, taken today as a parameter
  with no limits; a repeated calibration's seed whose last repeat is out of range, refused today after the first
  repeat; an observation refused for another posterior after it was already saved; a training setting that is not a
  number, a bare error today; a second recording missing, found only after the first was read; an unsupported model,
  a bare error today; and three refusals that point at the wrong control: the observation mode on a resume, the
  round's advice after Ctrl-C, an observation's name taken during the run). *Records*: a finished simulation set
  reused by a training keeps its recorded generation time (every reuse rewrites it today as the time since the set was
  made, which the retrain on the finished set would do);
  a training from a resumed simulation set records its Fisher spread as an uninterrupted one does (null today); a
  model's units are listed in the file's order (today in an order that changes at each launch). None changes the
  science, a simulated row, the set's identity or what a fit computes. **Recommended: yes to all sixteen.**
- **Not asked, because no task does it:** an edit inside `FIT_CODE`; the removal of a command-line flag or subcommand
  (item 64's flag for a name taken during a run is left: no new flag either).

## Near the science, not in the audit (told; nothing drafted)

Item 9 (the set's key knows nothing of the code or a user model's equations), 11 (the load-check audit: settle it
before any change to `core/config.py` or the features after the retrain), 28 (the Laplace pass of 29 September,
untraced), 43 (`CHI_UHAT_MAX` and `CHI_MIN_CYCLES` in no identity), S4-4 (a stability length past the grid cut
silently), S5-8, S5-10 and S5-16 (the checks' reading numbers live in the window, as designed), and the open list's
science lines (in the sorted table's pointers section).

## Sent to the last stage (its words and its screen)

23, 58, 62, 63, 73, 90, 103, 133 (a pinned shortcut and the window as two taskbar buttons, on the real screen), X3-2,
X3-3, X3-4, X3-17 (the run page's heights at the least size, measured on the real screen), X3-6, S4-15, S4-20, S4-29,
S4-30, S4-31, S5-15, S5-27, and the open list's help and fix sentences (`--lengths`, `--chi-k`, `identifiability`,
the raw labels quoted, the `units` field with no flag, `describe_siblings`' first field). STATE's "Carried to the last
stage" names them.

---

## Task A0: The time stride, measured (only on Q1's yes)

**Items:** the pointer P2's 3A-8 and its two neighbours. **No file of the repository changes.**

- [ ] One Opus agent, CPU only (`CUDA_VISIBLE_DEVICES=-1`, asserted), scripts under
  `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\audit\a0\`, run with the program's own functions:
  (a) over the box the finished set was simulated on (its `manifest.json`, read only): for time scales across the box,
  the stride the code takes, the time scale the kept trace belongs to, and the lock-in's reading of a response equal to
  its drive, with the code's spacing and with the true spacing of the kept points; (b) a handful of rows simulated at
  the finished set's settings, tiny: the features as the code computes them against the same with the true spacing,
  in units of each feature's spread over the rows; (c) the same for forced mode; (d) the two leads tried; (e) the share
  of the prior's draws where the effect is large.
- [ ] Report `reports/a0-stride.md` in plain words with the numbers; no fix designed; it ends with what the owner must
  decide (what the model at a time scale is meant to be; a fix or not; a fix means the set simulated again).
- [ ] One Opus reader re-derives the headline numbers on its own script, after the agent ends (never beside it).
- [ ] No gate (nothing in the repository changes); the ground checked after.

## Task A1: What only removed code used

**Items:** S5-18 (every part but `freqs_per_batch`'s packing, `CUDA_MEM_FRACTION`, `UNIQUE_FREQS`, and the parts
found in use), S5-17, S5-20, S5-21, S4-24, 68, the pointer P1's two files with no importer; on Q2's yes, the packages.

- [ ] Remove, each after a search at HEAD that finds no reader in `core/`, `tests/` or the top-level files by name, by
  string or by `getattr`: `pipeline.sim_class`; `core/cli.py`'s `parse_cell`, `cell_sources`, `resolve_bounds_for_cell`,
  `_merge_vals_bounds` (and the tests that test only them; the comments that name them reworded); `refusals.require_below`
  (and its tests); `config_args.name_help`; `labels.GUI_HTML`'s entries no page reads; `core/Helpers/model_helpers.py`;
  `core/SBI/Priors/Forcing Priors/sin_prior.py`; `FigureStack.png_path` and `add_png`; `watch_dir` and the plot
  watcher's feed; `BasePanel` with `insert_result_widget`, `_fit_the_form`, `_run_page_beside`, `_OWN_INSET["inference"]`
  and its loop, `_hw_batch` and its re-export, `budget_cfg`, and the tests' bare run host moved to a stand-in of its own;
  `run/model.py`'s `LEAVES_OUT`, the reading's `i`/`n` stage count, the unfiled end, `provide_stream`/`on_chunk`,
  `own_view`, `set_count`; `theming.theme_changed`; `PageNames`; `GAP["rail_rest"]`; `keys.PAGES` and its comment; the
  run page's command slot and `SelectableLines` (`_Lines` stays); `offer_pressed`'s round branch, only if shown
  unreachable (else left, with the reason in the report); `InferenceScreen.install_config` (the tests moved to
  `take_prior`) and `ArtifactPicker(allow_new=)` with `NEW_LABEL`; `_GLYPH_PX["px_i_nav"]` and the icon names "back"
  and "settings" (the font's code points stay). `theming.active_appearance` is kept where a test reads the product's
  state through it.
- [ ] On Q2's answer ("All eight"): the lines of pyqtgraph, imageio, imageio-ffmpeg, h5py, pandas, pyarrow, antropy
  and pybispectra, and the comment over imageio's two, out of `requirements.txt`; nothing uninstalled.
- [ ] Tests: the suites that named a removed name pass whole; a removed test's carried-over row closed; the window
  built and every step, Records and Settings opened in one test that already exists (named in the report).
- [ ] Gate: fast + real-screen; no tensors move (no card gate); registered engine changes.

## Task A2: Words in comments, and names that mislead

**Items:** 31, 32, 33, 34, 67 (not its `core/config.py` line), 69, 81, 93, 94, X3-20, S4-25, S4-28, S5-19, S5-26, and
the pointer P1's `checkpoints_using_prior` docstring.

- [ ] Comments and docstrings made true as the code is (the progress pane, the Cancel button, `BasePanel._running`,
  "Run cancelled.", the stop points' count, the Settings screen, the tabs and the Artifacts screen, the window as it
  was, `request_stop`'s "safe from any thread" narrowed to the window's thread).
- [ ] `rule_table.py`'s two private imports from `page.py` made public names; the three `FINISHED_TEXT`s and the two
  `NO_MESSAGE`s given names of their own, each sentence unchanged.
- [ ] The lines over 120 characters under `core/gui/` wrapped, words unchanged.
- [ ] Proof in the report: for every file changed for words or wrapping alone, `ast.dump` equal before and after
  (docstrings listed apart); the strings a person reads unchanged; the suites that read source text run whole.
- [ ] Gate: fast + real-screen.

## Task A3: The store, the kept training, and what they record

**Items:** 4, 18, 47, 50, 51, 53, 54, 57, 60, 116, the pointer P1's `RunLog.detach`; on Q3's yes, 7, 25, S4-17.

- [ ] A kept training's removal takes its lock and refuses while it is taken (4); the atomic writer refuses
  `retries < 1` (18); the libraries' versions kept when the code is remembered at load (47); `_discard` through the
  store's retrying removal (50); `reopen` re-reads the record under the lock and refuses a record marked unresumable
  or gone (51); a pass's log written in its own `try`, so a log that cannot be written leaves the pass saved (53) and
  written through the atomic writer (54); the in-process part of `wall_seconds` from a monotonic clock (57);
  `RunLog.detach` without `attach` leaves `warnings.showwarning` as it was.
- [ ] On Q3's yes: a complete set reused keeps its recorded generation time (7); a training from a resumed simulation
  set records `fisher_spread` from the stored eigenvalues under the rule an uninterrupted one uses (25); the units
  file's reader keeps the file's order, each unit once (S4-17).
- [ ] Tests first, each red on its fault: one leg per cure above; a renamed kept training keeps its name (60); a
  failed save's `save_error` is recorded before its warning (116); the reuse leg on a copy in a scratch store.
- [ ] A tiny seeded training before and after (fresh and from a resumed set): the network state and every number
  but `fisher_spread` equal.
- [ ] Gate: fast + real-screen; the card smoke gate (the save path runs in every fit on the card).

## Task A4: Refusals in time, and the command line

**Items:** 21, 66, 113, the pointer P1's Ctrl-C advice in `artifacts`; on Q3's yes, 22, 26, 27, 65, S4-3, S4-18,
S5-3, S5-4, S5-14 and the pointer P1's tier-1 force, knob conversion, experiment hashing and unsupported model.

- [ ] `build_posterior`'s handlers name the kept note `kept`, not `note` (21); `_leftover_hint` names only rows with no
  manifest as leftovers (66); `_FIT_FIELD`'s fields registered, and its settings part equal to `fit_settings`' keys
  with `_SETTING_WORDS` (113); an `artifacts` command interrupted prints no training cache's advice.
- [ ] On Q3's yes, each a `Refusal` with its field, before anything is read, simulated or written: a load with no
  prior (22); the observation mode named by its own field (26); a kept round through `train` (27); the round's
  `--directions` or `--level` refusal under that flag's field (65); a count that is not whole (S4-3); a bounds file's
  line it cannot read, naming the line (S4-18); a recording refused for another posterior before its observation is
  written (S5-3); an observation's name taken during the run under `observation_name` (S5-4); the repeated
  calibration's last repeat's seed at the entry (S5-14); a tier-1 units file with no force before the sweep; a
  training's knobs checked by their rules before any conversion; a missing recording before any is hashed; an
  unsupported model.
- [ ] Tests first, each red on its fault, and each asserting nothing was written; every shipped bounds, units and cell
  file reads as before; the card gate's commands unchanged.
- [ ] Gate: fast + real-screen.

## Task A5: The memory planner's edges, the engine's figures, the calibration's stop

**Items:** 14, 20, 38, 39, S4-14.

- [ ] `_release_device_memory` drops the solver's graph cache before it empties the allocator's cache (38); a
  non-finite `PRISM_VRAM_CEILING_GIB` is no number, as a word is (S4-14); a `PRISM_MEM_LOG_EVERY` that is not a whole
  number of at least 1 is taken as unset (39); a figure that fails while it is drawn is closed (20); the repeated
  calibration can be stopped between ranks, and says which repeat it is in (14).
- [ ] Tests first, each red on its fault; a tiny seeded repeated calibration before and after, its numbers equal.
- [ ] Gate: fast + real-screen; the card smoke gate (tensors move); the diagnostic card.

## Task A6: The window's run controller and history

**Items:** 92, 95, X3-7, S4-12, S4-13, S4-22, S4-27, S4-33, S5-9, S5-11, S5-22.

- [ ] A run that ends with no result, no error and no stop ends as a failure that says so, through `_end` (S4-12), and
  a `SystemExit` raised inside a run ends that run, not PRISM (92); a history that cannot be rewritten after Discard
  shows the discarded words (95); `release_figures`' disconnect checked (X3-7); `health`'s counts summed across a
  run's tries (S4-13); two runs of one second ordered by their count (S4-22); `_gone`'s branches each driven, or shown
  unreachable and removed (S4-27); a Resume's bounds key matched through `_entry_named` (S4-33); the window's two
  private engine names and `_RESCALE_KIND` made public names in the engine, every reader moved (S5-9, S5-11); a
  resumed round's history holds the directions its kept record asked for (S5-22).
- [ ] Tests first, each red on its fault; every display-marked test read for a stale expectation.
- [ ] Gate: fast + real-screen.

## Task A7: The tests

**Items:** 86, 105, 109, 110, 111, 112, 114, 115, 117, 119, 121, 127, X3-9, X3-10, X3-11, X3-19, X3-21, S4-26, S5-23,
S5-24.

- [ ] The session guard watches `Resources/Bounds`, `Cells`, `Units` and `Models` as it watches `Resources/Plans`,
  and the test model's installer takes itself out however its test ends (105); the native message's guard walks every
  file under `core/` (109); the unpack names its reason (110); the hygiene scan gains a family for "stage N" (111,
  its hits read first); one stop helper and one batch guard in `tests/_fixtures.py` (112), one `_out_of_memory` (S4-26),
  one `_the_most_rows_in_error` (S5-23); the named code's completeness pinned by a walk (114, reading `FIT_CODE` only);
  the cache's commit scan found and made a walk, or closed if gone (115); the unseeded test seeded and timed (117);
  the folder test's order after `yield` pinned (119); the page names' end check where it can fail (121); the margins
  test on a patched list of models (127); every panel class in `_PANEL_PARTS` (86); the held-figure check watching the
  renderer too (X3-9), `_appearance_over`'s window loop in its own `try` (X3-10), the figure tests' windows let go (X3-11),
  the caption test's holder in a `finally` (X3-19), the window's entry switching to Agg before it imports the window
  (X3-21); the guards against a widget shown as a window watching Inference's and the round's run ends (S5-24).
- [ ] Each new or changed assertion seen red on its fault; the carried-over list current.
- [ ] Gate: fast + real-screen.

## Task A8: The retrain's window route

**Items:** 134. Documents only.

- [ ] `docs/guide/retrain.md` rewritten as the window route: the steps as on-screen controls (tab, field, button),
  the memory limit in Settings, what to note before starting (`git rev-parse HEAD`), what a stop keeps and how a kept
  training resumes, what is still the command line's and why; and a statement of readiness that says where Q1 stands.
- [ ] One Opus drafter, one Sonnet reader against the window at HEAD; the exit's `CLAUDE.md` edits drafted for the
  owner (the sentence that calls the runbook a command-line one).

## Task A9: The cell at its exact time scale (added 2026-11-17; runs after A1, before A2)

**Items:** the pointer P2's 3A-8 and its two neighbours, as A0 measured them (`reports/a0-stride.md`, checked by
`reviews/a0-check.md`). **Spec:** `docs/superpowers/specs/2026-11-17-exact-time-scale-design.md`. **The owner's
answers (2026-11-17):** 11:25 "Exact, all three fixed" and "In this audit"; 11:53 the design "Yes, as recommended",
the chi settings "Yes, re-measure", the bench's drive at the known point "Yes, they can"; between 12:22 and 12:53 the
frame rate "can differ" (reading a bench recording at its own measured frame interval is a later task, owed before any
inference on bench recordings). **The dispatch:** `.superpowers/sdd/2026-11-17-code-audit/briefs/a9-dispatch.md`.

On those answers this task alone opens what the Global Constraints fence, and only as the spec's section 10 says:
`core/Solvers`, `core/Simulator`, `core/Models`, the simulated paths it names, and `core/artifacts/identity.py` (a new
entry, `record_clock`; `FORMAT` `training-rows/3` to `training-rows/4`). A simulated row changes, and so does the
simulation set's identity: the finished set `Artifacts/simulations/eda5c2a1e468` stays on disk untouched, but a
training of the new code no longer matches it, and the set is simulated again, from the window, by the owner, after
the audit. `core/config.py` stays byte-identical until step 7's own commit, on the owner's yes; `FIT_CODE`,
`gen_training_data`'s signature, `TrainingPlan` and `FIT_CONSTANTS` do not change.

- [ ] Step 0: M1 (the varying step leaks into no channel the network reads), M2 (the cost) and M3 (the card's step
  time); a checkpoint; an M1 FAIL goes to the owner with its numbers.
- [ ] Steps 1 to 5 as the spec's section 10, each test first; a checkpoint after step 3 (the training rows).
- [ ] Gate after step 5: fast + real-screen, then the card smoke gate with run 5's drill and the diagnostic card.
- [ ] Step 6, the guides; step 7, the chi re-measurement on the card, its numbers to the owner.

## Task A10: A step-correcting rule, measured (added 2026-11-18; after A9's build, before the retrain)

**Items:** M1's FAIL, accepted by the owner at the step as designed: the simulator's rule takes an error that shrinks
only in proportion to the step (`reports/a9-builder.md`, "Checkpoint 1b": no finer step passes M1's bar, at 29 to 52
card hours). **The owner's answer (2026-11-17, 22:54):** "Yes, after A9's build": measure a rule that corrects each
step, before the retrain; adopting it is the owner's separate decision, taken before the retrain so the set is
simulated once.

- [ ] In scratch only (`audit\a10\`), no file under `core/` changed, about an hour on the processor: one
  step-correcting rule that the solver's noise form admits, on M1's paired ensemble at A9's step, its error left and
  its cost per step against today's rule at the step and at 2 and 4 times finer; the same twelve time scales, the same
  bar.
- [ ] Its numbers to the owner as one plain question with a recommendation. A yes opens `core/Solvers` in a task of its
  own, planned then; a no leaves A9's solver as built.

## Task A11: The corrected step (added 2026-11-18, on the owner's yes; after A8, before the review)

**Items:**
- A10's measurement (`reports/a10-builder.md`): stochastic Heun at A9's step removes nearly all the step error that M1
  measured.
  - A-F FAIL 78 to 14 of 360, and G 22 to 8 of 132.
  - The worst pooled bias falls from 0.53 to 0.04 of a spread; the stiff cell's from 2.9 to 0.003.
  - It does not strictly pass M1's bar.
  - It costs about 1.7 times the step on the card: about 46 card hours for a 10,000-batch set instead of 29, with the
    same memory, batch sizes and recording lengths.
- **The owner's answer (2026-11-18, 22:50 PST):** "Adopt it in the audit". It is its own task after A8 and before the
  review, so the retrain's set is simulated once, with it.

**What moves, by the owner's yes:**
- every simulated row;
- the simulation set's identity (a new token for the rule, so the retrain's set id leaves `d7002cdb4be8`);
- the golden amortized digest;
- `core/Solvers`, `core/Simulator` and the models' compiled steps under `core/Models`.

**What does not move:**
- `core/config.py`;
- the seventeen fit-code digests;
- what an uninterrupted fit computes;
- anything under `Artifacts/`.

**The steps:**
- [ ] **A design note first** (the builder's, ruled by the controller, any point that moves what a stage refuses or a
  record holds to the owner):
  - which models take the rule: those whose noise's Ito-to-Stratonovich correction is zero, which the Nadrowski model
    meets. The Hopf model and a user model are checked. A user model whose noise depends on its own variable keeps
    today's rule, or is refused, since averaging its noise integrates another equation. How that is decided (from the
    model's declared form, or checked) and what a person sees;
  - the prior's stability screen keeps today's rule (Heun's stable region holds Euler's, so the screen stays
    conservative, and the prior is unchanged);
  - the identity's token for the rule, and the refusal of a set or posterior made under the other rule;
  - the graph drive buffer one point longer, and the drive at each step's end across a seam.
- [ ] **The build:**
  - the rule in `core/Solvers/sdeint.py`'s loops (both), the simulator's drive slice and the models' compiled steps;
  - the identity token;
  - every pin the rule moves;
  - tests: the noise-off order 2 on the built solver; the predictor equal to today's step bit for bit; a segmented
    run equal to one long run; the drive at a step's end across a seam; the rule's choice per model; the refusal of the
    other rule's set and posterior.
- [ ] **Measured again on the card:**
  - A9's chi checks (`probes band`, `drive`, `mask` on the tier-1 cell and the retrain prior);
  - the lock-in ceiling's sweep (64 still free of a wall);
  - M1's ensemble at K = 1 on the BUILT solver against A10's scratch Heun (equal, or within the card's one ulp).
- [ ] **The guide:** `science.md`'s step error with A10's numbers, and `architecture.md`'s solver.
- [ ] **The new retrain set id** reported.
- [ ] **Gate:** fast + real-screen, the card smoke gate with its resume drill, the diagnostic card.

## The review and the exit

- [ ] The three-reader review of `b245f20b..HEAD` (seats as `stage-5/briefs/review-seats.md`, re-pointed to this plan),
  one fix wave, its re-read.
- [ ] The exit, with the owner: the gates of record once on the final head (the quiet fast gate, the slow set, the
  card smoke gate with run 5's drill, the diagnostic card, one card test through the Checks page's call, the
  real-screen run; the owner told first the commit, the folder and the last run's time); the guide's pages each change
  made false; the code audit's list marked as built; the owner's yes or no to any `CLAUDE.md` edit on its exact words;
  the list of everything decided without the owner (the sorted table's "left" reasons among it); STATE.
