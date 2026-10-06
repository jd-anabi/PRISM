## 11. The engine changes

The twelve changes E1 to E12 (U18, U21 to U23), each as it is to be built and audited; then the
three things done in the window instead, the list left for the audit, and the registry. Line
numbers are those of commit `c45cc4a` (no file under `core/` or `tests/` differs from `8c6ac2f`);
`sbi base.py` stands for
`C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/sbi/inference/trainers/base.py`
(sbi 0.26.1). Engine text is shown in code type; log lines keep the engine's words, refusal
sentences come under U7 with E11. Sections 11.1 to 11.11 keep the numbers other parts cite: E8 sits
with E5 (11.6), E10 with E7 (11.8), and E9, E11 and E12 follow as 11.12 to 11.14.

| id | for | files touched | card test owed |
|---|---|---|---|
| E1 | U11 | `core/SBI/train.py` | yes |
| E2 | U10 | `core/SBI/training_checkpoint.py`, `core/orchestrator.py` | no |
| E3 | U16 | new `core/diagnostics/card.py`; `core/diagnostics/__init__.py`; `core/tool/smoke.py` | yes |
| E4 | U16, U21 | `core/orchestrator.py`, `core/refusals.py`, the two front-end tables | no |
| E5 | U8, U21 | new `core/plans.py`, `core/refusals.py`, the two front-end tables | no |
| E6 | U18, B14 | `core/orchestrator.py`, `core/diagnostics/ablation.py`, `core/diagnostics/identifiability.py` | no |
| E7 | U14, C22 | `core/artifacts/store.py`, `core/orchestrator.py` | yes |
| E8 | U8, U21 | `core/config.py`, one new line | no |
| E9 | U17, U21, B26 | `core/artifacts/store.py`, one value; three docstrings (11.12) | no: a fast test (11.12; C21) |
| E10 | U21, C22 | `core/artifacts/store.py`, `core/orchestrator.py` | yes |
| E11 | U22, C23 | the refusal sentences of 20 files under `core/`; the support sentences of `core/registry.py` | no: text only |
| E12 | U23, B18, C20 | new `core/Helpers/plot_colours.py`; seven plotting files | no |

Which stage each change lands in, and when the card test runs, are part 13's (13.1.7, 13.2; C21):
one run at the exit of its stage 3, covering E1, E3, E7 and E10, and again at stage 9 only if code
that moves tensors changed after it. (a choice the owner has not seen: which changes count as
touching training; U19 names none.)

### 11.1 What no change touches

| invariant | where it lives | why E1 to E12 leave it alone |
|---|---|---|
| the simulation set's identity | the 34 keys built at `core/artifacts/identity.py:87-122`, digested at `core/SBI/training_checkpoint.py:110-113` | no change edits either file's identity code or anything the keys are computed from; E2 only reads identities; the name E9 keeps is in neither the keys nor the digest |
| what a simulated row holds | `gen_training_data` and below (`core/SBI/pipeline.py:1308` onward), `core/SBI/chi_probes.py`, the features | not edited by any change |
| every setting in `core/config.py` | `core/config.py` | E8 adds one path line; no existing line is edited; no change writes a config constant |
| no print or record between steps 1 and 3 of a checkpoint save | `training_checkpoint.save` (from `:359`) | E1's record is emitted inside the fit, after generation has returned; E7's and E10's lines are Python warnings raised outside any save; E9 changes one value in the manifest written after step 3 (`:395`) and adds no output |

Visible on the command line although no flag changes (U4): E1's line on the console (11.2) and the
tails of eight refusal lines (11.7), seven of which the owner was told of (U21); a renamed
simulation set's row in `artifacts list` (11.12); the reworded refusal sentences (11.13); the line
colours of saved figures (11.14); and E10's warning (11.8), the one change to what a command-line
run does: a training that fails today while its last figure is drawn or saved ends with its
posterior and exit code 0 (C22). E7 changes nothing there.

### 11.2 E1: one record per epoch from the fit

**For.** U11 (epoch, pace, the loss so far, epochs since the best).

**Where.** `core/SBI/train.py` only: a module logger (the file has none; `pipeline.py:32` is the
pattern); a private helper `_log_each_epoch(infer)` beside `_NoTracker` (after `train.py:181`); one
call to it between the trainer's construction (`train.py:241-242`) and the round loop
(`train.py:254`), so it runs before `gen_training_data` (`train.py:256`) and a trainer it cannot
wrap is found before anything is simulated. The `tracker` docstring (`train.py:208-211`) is
corrected to say sbi calls a tracker only after the loop (sbi base.py:1009, :1210-1250).

**Mechanism.** sbi's loop has no callback. Per epoch it trains (sbi base.py:984), validates (:991),
calls `self._summarize_epoch(train_loss, self._val_loss, epoch_start_time, summarization_kwargs)`
(:993-995), which appends to three lists in `_summary` (:1120-1123), then increments `self.epoch`
(:997) and prints its counter (:998, :1269-1274). The helper replaces that one method on the
trainer instance with a function that calls the original with the same arguments and then emits
the record. `train.py:404` already reads the same private `_summary`.

| value | read from | note |
|---|---|---|
| epoch number | `infer.epoch + 1` | equals the number sbi's counter prints next |
| training loss, validation loss | `_summary["training_loss"][-1]`, `_summary["validation_loss"][-1]` | the numbers that later go into `loss.npz` (`train.py:403-411`) |
| time taken | `_summary["epoch_durations_sec"][-1]` | training plus validation of that epoch |
| epochs since the best | counted by the helper | see below |

sbi's own count (`_epochs_since_last_improvement`) is updated by `_converged` at the top of the
next iteration (sbi base.py:978-979, :1143-1148), so at :993 it is one epoch stale. The helper
keeps its own with sbi's rule: best starts at infinity; a validation loss strictly below the best
resets the count to 0, anything else (a tie, a NaN) adds 1; both reset when `infer.epoch` is 0.
On a new trainer, which is all PRISM trains (one per training, `train.py:241`; one round,
`core/config.py:366`), its value is what sbi will hold after its next test; the fit stops when it
reaches the patience (sbi base.py:1151). In a second round of one trainer sbi's best would start
from the loss it last held, not from infinity (sbi base.py:221, :1143-1144); PRISM runs none.

**The record** (level info, logger `core.SBI.train`; losses `.6g`, seconds `.1f`), built by one
module-level function, `fit_epoch_line(epoch, training_loss, validation_loss, seconds, since_best)`,
which the tests' stand-in run also calls (part 13, 13.1.4):

```
[fit] epoch 12: training loss -3.41274, validation loss -3.39812, took 3541.2 s, epochs since the best 0
```

(a choice the owner has not seen: this wording.) Part 6's reading table parses it (6.3.1); the
engine's suite and the window's pin one literal copy each.

**Edges.**

| case | behaviour |
|---|---|
| the trainer has no `_summarize_epoch` (another sbi) | one warning record before generation; no per-epoch records; the training runs |
| building the text raises | caught as `Exception`; that epoch's record is skipped; never `BaseException`, so a stop still passes |
| stop requested during an epoch | every `core` record is a stop point in the window (`core/gui/streams.py:251-253`), so the stop now lands at this record, a few statements before sbi's print (sbi base.py:993-998). That print and its own stop point (`streams.py:204-205`) are untouched code. Same epoch end, fit lost as today (U12) |
| where it goes | the window's pump and so the full text; the posterior's `log.txt` at the commit (`core/runs.py:81-87`, `core/artifacts/store.py:552-574`); on the command line, stdout (`core/tool/logging_console.py:26-37`), printed straight after sbi's counter on the same console line because the counter ends without a newline (sbi base.py:1274) |

**Must not change.** The original is called first and the helper appends nothing, so `loss.npz`,
the loss figure and the manifest's `epochs_trained` and `best_validation_loss`
(`orchestrator.py:1422-1431`, `:1473-1474`) are byte-identical. The helper makes no torch call and
draws no random number: the fit's draws are the split (sbi base.py:444), the two samplers
(:457-463) and the network's initialisation. PRISM never pickles the trainer (`train.py:160-162`),
so the replaced method reaches no file.

**Tests.** Pinning today: `test_training_creates_no_sbi_logs_directory`
(`tests/test_user_sbi.py:5248`), `test_a_non_finite_loss_during_the_fit_is_a_failure_not_a_refusal`
(:5302), the counter's two tests (`tests/test_vt_progress.py:236`, `:534`),
`test_no_print_call_remains_in_the_converted_modules` (`tests/test_tool.py:3031`). New, first, on
`_toy_training` (`tests/test_user_sbi.py:5214`): one `[fit]` record per epoch whose numbers equal
the returned curves; the count against a hand computation; with the same seed, the trained weights
and `torch.get_rng_state()` identical with the helper and with it replaced by a no-op; a trainer
without the method trains with one warning; under `redirect_streams`, a requested stop raises at
the record.

### 11.3 E2: every differing setting, per saved simulation set

**For.** U10 (the window asks first, and names every difference).

**Today.** `_sibling_diffs` already computes every differing key per committed sibling
(`training_checkpoint.py:285-287`); `describe_siblings` prints only the first (`:294-295`) and
`near_miss_siblings` keeps only siblings with exactly one (`:347-351`). `training_preview` carries
the one-line string (`orchestrator.py:554`), which the window prints verbatim (W023; part 8, 8.4).

**Interface.**

- `training_checkpoint.sibling_differences(identity, root=None) -> list[dict]`, after
  `near_miss_siblings`: one entry per other committed simulation set holding at least one batch,
  richest first; `name` is the set's folder, its 12-character id; fields in sorted key order;
  values raw and uncut, as `near_miss_siblings` returns them (`:347-348`). `_sibling_diffs` yields
  `complete` as a fifth element so the state file is read once.

  ```python
  {"name": "0c1d2e3f4a5b", "batches": 3989, "complete": False,
   "fields": [{"field": "n_runs", "mine": 10000, "theirs": 5000}, ...]}
  ```
- `TrainingPreview` (`orchestrator.py:448-478`) gains a last field `sibling_diffs: tuple = ()`. The
  default keeps `tests/test_settings_persistence.py:305` constructing it. `training_preview` fills
  it in the branch at `orchestrator.py:553-555` from one walk and builds the old `siblings` string
  from the same rows, so that string stays byte-identical and the disk reads do not double (W100
  is not made worse). It is `()` when the cadence is off, a config or prior is missing, the run's
  own set holds batches, or `checkpoint_error` is set.

**Not changed.** What `describe_siblings`, `near_miss_siblings` and `fresh_run_near_misses` return
(the first two unpack the fifth element and ignore it; `describe_siblings` formats through the one
private function the preview also calls), and the stage's binding refusal, which still fires only for a set exactly one setting away
(`orchestrator.py:1110-1114`) and which `new_run=True` alone silences (`orchestrator.py:844-845`).
The pipeline's own warning (`pipeline.py:1540-1542`) keeps its text. Read-only: nothing is written,
no identity is built differently.

(a choice the owner has not seen: this return shape.)

**What the window does with it.** The on-screen name of each of the 34 keys, and which sets are
listed, are part 8's (8.7), in a table kept beside `CONTROL` in `core/gui/fields.py`. A narrowing
round's set cannot be found beforehand: its region is drawn inside the stage (`orchestrator.py:2643`), and drawing it earlier
to preview it is not designed in (C11). The page shows the stage's refusal of a set one setting
away as a question at the click (B11, a choice the owner has not seen; part 8, 8.6), with the
engine's sentence as it stands (C11; `orchestrator.py:567-587`, reworded by E11). Structured facts
on the refusal would be an engine change beyond E1 to E12 and are not designed in.

**Tests.** Pinning today: `tests/test_user_sbi.py:3474-3509`, `tests/test_artifact_store.py:1129-1130`,
`:2460-2641`, `tests/test_settings_persistence.py:305`, `:831-868`. New, first: two and three
differing keys all returned with both values uncut; `truncation` listed; a set with no batches or
an unreadable header absent; the old string unchanged; the preview still never raises.

### 11.4 E3: the card test as an engine step

**For.** U16 (every check has a screen).

**Today.** `run_smoke(args, store)` (`core/tool/smoke.py:139-238`) takes the argparse namespace,
prints, and wraps the four stage calls (`:212-231`) in one `seeded(seed, device)` (`:211`);
`_infer` adds two shape checks (`:241-257`). Tests pin its four keyword sets
(`tests/test_tool.py:1189-1274`) and that its framing lines are `print` calls in
`core.tool.smoke` (`tests/test_tool.py:2961-2964`); no engine module may print (`:3031`).

**Interface** (a choice the owner has not seen: the place, the name and the callback).

```python
# core/diagnostics/card.py, re-exported from core/diagnostics/__init__.py
STAGES = ("prior", "posterior", "validate", "infer")

@public_entry
def card_test(cfg, *, cell, t_obs_s=None, seed=0, stages=STAGES,
              num_runs=4, run_size_cap=32, n_cal=40, max_num_epochs=5,
              hidden_features=None, num_transforms=None,
              checkpoint=False, save=False, prior_ref=None, resume=None, new_run=False,
              on_stage=None, fig_sink=None, store=None) -> CardTestResult
```

`CardTestResult` (frozen): `stages_run`, `seconds` (stage to wall seconds), the five wrappers
(`prior`, `posterior`, `calibration`, `observation`, `inference`, each None when not run),
`conditioning_width`. `on_stage(stage, event, seconds)` is called on the calling thread with event
`"skipped"`, `"started"`, `"finished"` or `"failed"`. `resume`, `hidden_features` and
`num_transforms` reach the training stage only when not None, as `knobs` forwards them today
(`core/tool/config_args.py:242-248`; `--resume` defaults to None, `:130`), so each default stays
with the stage and the keyword sets are today's.

| moves into `card_test` | stays in `core/tool/smoke.py` |
|---|---|
| the cell check on a throwaway config (`:167-171`), now `require_file("cell", ...)` then `cli.load_and_validate_gt(cfg.copy_for_run(), cell)` | argparse and `STAGES` for the parse-time check (`:54-74`, torch-free) |
| the default length `config.T_MIN_EXP_S` (`:172`); the cadence rule `max(1, num_runs // 2)` (`:191`) | the three usage errors (`:149-162`), `build_cfg` and the `[smoke]` banner (`:166`, `:176-189`), and its own cell check before the banner, so the console's order is unchanged |
| the one `seeded` context, read from `core.diagnostics.rng` at call time (`:144`, `:211`) | every print: the stage framing (`:194-209`) through `on_stage`, the closing lines (`:215-217`, `:225-227`, `:233-237`) from the result |
| the four stage calls with today's keywords and the names `smoke_prior`, `smoke_posterior` (`:212-231`); the two shape checks (`:253-256`) | `main`'s own root, its clean-up and its Ctrl-C advice (`core/tool/__init__.py:119-134`, `:213-214`, `:56-74`) |

`card_test` emits no record and no print of its own. For a caller that did not pre-check it raises
two neutral refusals before the prior, when the training is among the stages (`:153`):
`resume` of `"require"` or `"never"` without `checkpoint` (`field="resume"`), and
`resume="require"` without `prior_ref` (`field="prior"`).

**Who calls it.** The window's screen makes one call into a scratch store with its own config; its
callback and what it reads from the run's text are part 9's (9.7). The gate's five
command lines stay a command-line recipe (B25) and reach the step through `core/tool/smoke.py`,
which is why the step keeps `checkpoint`, `prior_ref`, `resume` and `new_run`. The stage calls stay
looked up on `orchestrator` at call time.

**One difference.** As a public entry it shares one run buffer with its stages
(`core/runs.py:145-166`), so each record in the card test's own store carries the lines since the
card test began, as an inference's does. The console's bytes, the exit codes, which records are
written and their manifests are unchanged; only each `log.txt` grows. The pinned set of public
entries (`tests/test_artifact_store.py:3751-3792`)
gains this one and one leg. (a choice the owner has not seen: a public entry, with this effect.)

**Tests.** Pinning today: the smoke tests at `tests/test_tool.py:1115-1478`, `:2961-2964`, and
`tests/test_tier1.py:617`. They must pass unchanged; the seeding patch
(`tests/test_tool.py:1227-1230`) stays valid because the step reads `seeded` from the same module
at call time. New, first: `card_test` called directly with recorder stages gives the same four
keyword sets; a missing cell refuses with `field="cell"` before the prior; the caller's config is
untouched; the callback sees each event once in order; the two `STAGES` tuples are equal.

### 11.5 E4: an observation's name

**For.** U16 (the observation an inference records can be named).

**Today.** Both observation stages take `name` and `note` and refuse a taken name at entry
(`orchestrator.py:127-139`, `:293-300`); the two compositions pass neither (`:2528`, `:2558`).

**Change.** `simulated_inference` (`:2449-2452`) and `experimental_inference` (`:2536-2538`) gain
`observation_name: str = ""`. Each checks it in its pre-spend block directly after the inference's
own name (`:2483`, `:2554`), re-raising the store's refusal with a new key so the window can point
at the right field, then forwards it as `name=` to the observation stage. The return stays
`(observation, inference)`. (a choice the owner has not seen: the name only; the observation's
note stays empty and is edited from Records.)

| registry | entry |
|---|---|
| `core/refusals.py` `FIELDS` | `Field("observation_name", "the observation's name", None)` (U21) |
| `core/tool/fields.py` `FLAG` | `None`: the command line gains no flag (U4) and `core/tool/stages.py:202-209` is not edited |
| `core/gui/fields.py` `CONTROL` | "Name of the observation" on the Inference page (part 8, 8.5) |

**Edges.** An empty name is free (`store.py:831-832`). The same text may name the observation and
the inference: a name is unique per kind. The stage's own entry check (`:139`, `:300`) stays and
carries `field="name"`; nothing is simulated between the two checks. A name taken during the
simulation is E7's case, for a caller that opts in (11.8).

**Tests.** Pinning today: `tests/test_artifact_store.py:2828-2841`, `:4287-4301` (the order of
pre-spend refusals), `tests/test_tool.py:1266-1269`. New, first: the observation is written under
the name; a taken observation name refuses with `field="observation_name"` before the cell is
parsed; the registry count and both tables (11.11).

### 11.6 E5 and E8: the plan reader and its folder

**For.** U8, U21. The file's form, what a plan may and may not hold, the reader's interface and
its problem sentences are part 7's (7.1.2 to 7.1.5). This part fixes the place and the contract
they keep.

**Place.** A new module `core/plans.py`, standard library only like `core/refusals.py`
(`refusals.py:22-25`), so the window can read a plan on its own thread and the command line could
later read one before importing torch. It imports `core.refusals` and nothing else under `core/`.

**Contract.**

- `read_plan(path)` reads one file and returns a whole `Plan` or raises. It writes nothing, reads
  no store and no `core/config.py`, knows no folder and looks at no second file: whether the bounds
  file exists and the units resolve is checked by the window (part 7). No stage reads a plan.
- Every failure is a `Refusal` with `field="plan"`, each problem carrying the key of the setting it
  concerns.
- Settings are named by registry keys (B13, a choice the owner has not seen), and their range
  rules are the registry's own (part 7, 7.1.5).
- `plan_text` is the inverse as pure text, so reader and writer cannot drift; the window does the
  writing and the asking. (a choice the owner has not seen: the engine holds a plan's text form as
  well as its reader.)

**E8** (approved in U21, as the rule on that file asks). One line in the PATHS block after
`core/config.py:184`: `PLANS_PATH = RESOURCES_ROOT / "Plans"`. No existing line changes; it is a
path, not a setting; no stage and not `core/plans.py` reads it (the window lists a model's
sub-folder of it, part 7). The scan for literal input paths
outside that file (`tests/test_artifact_store.py:2116`) keeps passing. The tool's help still lists
four input folders (`core/tool/__init__.py:28`): the command line reads no plan (U4).

**Tests.** None today. New, first: the module imports with neither torch nor Qt in a fresh
interpreter (the pattern of `tests/test_refusals.py:469`); each refusal carries `field="plan"`;
`read_plan(plan_text(v, about))` returns `v` and `about`. For E8: `config.PLANS_PATH` is
`RESOURCES_ROOT / "Plans"`, and the commit's change to `core/config.py` is that one added line.

### 11.7 E6: refusals that say where to fix it

**For.** U18, with the extent B14 rules.

| site | condition | today | after |
|---|---|---|---|
| `orchestrator.py:958-965` | a policy other than auto with the cadence at 0 | `Refusal`, no field, by the comment at `:961` (two settings answer it); pinned `tests/test_artifact_store.py:2297-2300` | `field="resume"`: the policy is the setting that was moved off its default |
| `orchestrator.py:1111-1112` | `resume="require"` and no saved batches | `Refusal`, no field; pinned `tests/test_artifact_store.py:2441-2444` | `field="resume"`, as the pipeline's copy already has (`pipeline.py:1510-1512`) |
| `ablation.py:41` | no conditioning network inside the posterior | `ValueError` | `Refusal(field="posterior")` |
| `ablation.py:105` | the posterior names no simulation set | `ValueError` | `Refusal(field="posterior")` |
| `ablation.py:115` | its summary width is not this build's | `ValueError` | `Refusal(field="posterior")` |
| `ablation.py:130` | its simulation set holds no rows | `ValueError` | `Refusal(field="posterior")` |
| `ablation.py:132` | the set's row width is not the posterior's | `ValueError` | `Refusal(field="posterior")` |
| `identifiability.py:238-241` | the third identifiability method: the posterior carries no training prior (reached from `:442`, above the writer at `:444`) | `ValueError` | `Refusal(field="posterior")` (B14; an extension the owner has not seen) |

Other refusals without a field key stay as they are (B14). (a choice the owner has not seen: the
key each of the eight gets.) Every sentence stays word for word here; E11 rewords it later. All
eight are raised before anything is simulated or written (`ablation.py:140` opens the writer after
its five). No new registry key: the scan (`tests/test_refusals.py:1067-1078`) and the two
table-closure tests stay green as they are.

**Consequence on the command line.** The tool appends the flag for a refusal's field
(`core/tool/__init__.py:185-194`): the two training refusals gain ` (--resume)`, and the six
checks' lines lose `ValueError:` and the `[raised at ...]` tail (`:195-206`) and gain
` (--posterior)`.

**In the window.** The two keys' rows in `CONTROL` are part 8's (8.1, 8.4).

**Tests.** Rewritten first: the two pins above to `field == "resume"`. New: each of the six is a
`Refusal` with `field="posterior"` (`tests/test_diagnostics.py:1325`, `:1513`, `:1540` keep
passing, a refusal being a `ValueError`); `ablation.py` holds no `raise ValueError`, read off the
source as `tests/test_diagnostics.py:153-155` does.

### 11.8 E7 and E10: nothing at the end of a run costs the result

#### E7: a name taken during the run

**For.** U14 (a record is named before it is made), U18.

**Today.** Every stage refuses a taken name as its first act; `create` checks again
(`store.py:846-850`) before the writer exists. For three kinds that second check comes after the
spend: the prior after its sweep (`orchestrator.py:726`), the posterior after the fit (`:1420`,
which also serves the narrowing round), the observation after its simulation (`:371`). Nothing
locks a name between processes (`store.py:52-58`).

**Interface.** `ArtifactStore.create(kind, cfg=None, *, name="", note="", if_name_taken="refuse")`.

| value | behaviour |
|---|---|
| `"refuse"` (default) | today's, byte for byte; the direct calls for the progressive kind keep it (the front ends' at `fdt_panel.py:301`, `crossval_panel.py:306`, `core/tool/fdt.py:235`, `:260`; the comparison's own at `core/FDT/compare.py:385`) |
| `"unnamed"` | a malformed name is still refused (`store.py:833-841`); a name another record of the kind holds is dropped: the writer is made with `name=""`, so the folder is `_unnamed__<id>`, and one Python warning says so |

**Opt-in, and only the window opts in (C22).** The stages that reach the three sites (`build_prior`,
`build_posterior`, `tsnpe_round`, the two observation stages, and the two inference compositions
for their observation only) take the same keyword, default `"refuse"`, check it at entry and hand
it on. The window's Prior, Training, Narrowing round and Inference pages pass `"unnamed"` (part 8,
8.1); `core/tool` passes nothing, so a command-line run is unchanged by E7.

No other call changes: the calibration, the inference and
all eight checks open their writer before anything is simulated (`orchestrator.py:1950`, `:2227`;
`sbc.py:163`, `identifiability.py:203`, `:444`, `:903`, `probes.py:598`, `:1149`, `:1459`,
`ablation.py:140`; two of them first do arithmetic on a stored matrix or read saved rows,
`identifiability.py:139-203`, `ablation.py:125-140`), so there the check is still a refusal that
loses no simulation (part 9 relies on it, 9.4). `"unnamed"` for a progressive kind is refused as a
caller's error. (choices the owner has not seen: these three kinds, a warning as the line, and
C22's opt-in.)

**The line** (a warning, not a record: a warning is no stop point, `streams.py:296-297`, and the
store's own notice that a record was written despite a moved input file is one,
`store.py:585-593`; it reaches the full text and the record's own `log.txt` through
`core/runs.py:122-131`):

```
the name 'retrain' was taken by another posterior while this run was going, so this posterior is saved without a name, under its id 20261008T141502; it can be named afterwards.
```

**Edges.** The stage's entry refusal is unchanged, so a name taken before the run still costs
nothing. The wrapper returned has `name == ""`, which is how the window knows to say so and offer
"Rename…" (parts 6, 8 and 10). Between `create` and the commit a second process
can still commit the same name; nothing is lost and two records then share it. Inside an inference
composition the inference's name is checked again after the observation is written (`:2174`): a
name taken meanwhile still refuses the inference, the observation is kept, and the repeated click
simulates one observation again (part 8, 8.1). Not
changed (C22, a choice the owner has not seen: E7 is not carried to an inference made together
with its observation; the cost is one observation, not a training).

**Must not change.** The manifest's schema, ids, `assert_name_free`, `rename`.

**Tests.** Pinning today: `tests/test_artifact_store.py:557-567`, `:953`, `:1307-1328`,
`:3830-3856`; the stub store's `create` signature (`:3290`) is widened first. New, first: at the
store, the four behaviours of the table; at each of the three stages, with the keyword passed, a
same-named record made while the stubbed spend runs gives a finished unnamed record, the line in
its `log.txt`, and the other record still holding the name; without it the stage still refuses
there, and nothing under `core/tool` passes it (C22).

#### E10: the engine's own loss figure

**For.** U21. The engine's side of the hazard 11.9's first row closes in the window.

**Today.** Inside the posterior's writer the engine draws the loss figure and saves it through the
writer's sink (`orchestrator.py:1429-1431`; `store.py:401-412`), which records the figure's path
before saving to it (`store.py:389-399`, `:407`). Any exception there leaves the `with` block, and
the writer removes the folder with the trained network in it (`store.py:436`, `:479-481`).

**Change.** `ArtifactWriter.fig_sink(forward=None, *, optional=False)`. With `optional=True` the
sink catches `Exception` from the save and from the forward. A save that did not finish takes the
path back out of the writer's list and removes any part-written file, so the manifest lists only
what is on disk; the figure is closed; one Python warning names the posterior's id and the error,
says whether the figure was saved, and that `loss.npz` holds the curves (the id keeps Python's
once-per-location rule from dropping a second such warning in one session, the trap
`core/refusals.py:58-63` describes). `orchestrator.py:1431` passes `optional=True`,
and the drawing at `:1429` is guarded the same way. The default is today's, byte for byte: for
every other figure a drawing that fails is a failed stage. A stop still passes (`WorkerCancelled`
is a `BaseException`, `core/gui/streams.py:38`).

**Both front ends (C22).** The guard is the stage's own, with no opt-in, so it changes a
command-line run too (11.1). On the run's page the warning is a message (part 6, 6.3.6; C6). (a
choice the owner has not seen: the posterior is then saved without the figure, with a warning.)

**Tests.** Pinning today: `tests/test_artifact_store.py:1602-1603` (the figure listed and forwarded
once). New, first: with the drawing, the save and the forward made to raise in turn, the posterior
commits and loads, its manifest lists no missing figure and the warning is in its `log.txt`; with
`optional=False` a failing save or forward still removes the record.

### 11.9 Done in the window instead

| item | closes | why the window can do it alone | built in |
|---|---|---|---|
| a display failure cannot remove a finished record | W192, W185, W162, W173 | The writer's rule, to remove its folder on any exception (`store.py:436`, `:479-481`), is right for a failed run. The only forward that can fail for a display reason is the window's own sink, whose last line is an unguarded emit (`core/gui/panels/base_panel.py:44-62`) | part 6 (6.13) |
| long lists shown in full | W081, W183, W184 | The cut is in the shared text renderer (`core/artifacts/report.py:30`, `:57-60`), which the command line prints and which must stay one document for both. Every value is in the manifest the store already returns (`store.py:743-749`) | part 10 |
| the memory limit a run used | W098 | The limit in force is one public call, `pipeline.vram_ceiling_gib()` (`core/SBI/pipeline.py:584-603`), which already folds in the environment override; settings are locked while a run is live (U11), so what the window reads at the start is what every batch plan reads (`pipeline.py:636`). A record made from the command line has no history file, and its page says the limit was not recorded | part 6, the run's history |

U24 to U27 need nothing from the engine: "Continue with these" uses today's load calls, and the
save on "shut down anyway" is today's stop (part 6, 6.10).

### 11.10 Left for the audit

| item | findings | code it concerns | what this design does meanwhile |
|---|---|---|---|
| the memory planner's arithmetic for whether a batch fits | W025 | `pipeline.py:562-578`, `:643-683` (one drive counted); `:67-75`, `:1775-1787` (the fourfold build peak, charged only after a failure); `chi_probes.py:263`, `:273-276`; `orchestrator.py:520-532` | U10 drops the sentence; the preview's two figures stay computed and unshown, and the window never calls the preview while a run is live (part 5) |
| failing fast after a memory failure that cannot be recovered | the first note's lead 3A.2 | `pipeline.py:313` (`_is_oom`), `:360`, the ladders at `:767`, `:848`, `:1817-1877`; `config.py:320`, `:326`; the learned limit, `pipeline.py:100` | the window's three tries follow only an out-of-memory failure (U12, C5; part 6, 6.8), sit above the ladders and do not shorten them |
| dates in a record's own log lines | W096 | `core/runs.py:59-60`, `:85`, `:129`; `store.py:570` | the dated full-text file carries the date; `log.txt` is unchanged |
| the prior not checked against the whole bounds file | W138 (W091 and W159 nearby) | `store.py:1275-1291`, `:1297-1300`; `orchestrator.py:1006-1024` | the Prior page shows the bounds file a saved prior was built on and warns (U16) |
| simulation sets know nothing of the code or of a user model's equations | W037, W191 | `identity.py:87-122` (`:89`: the name only); `provenance.py:22-30`; `store.py:1584` | nothing |

These five are U18's own. Part 12 carries the audit's whole list (12.7), B10's numbers and U21's
additions (E8, E9, E10 and the registry's new names) among them.

### 11.11 Registry and tables, in one place

`FIELDS` goes from 109 keys to 114 (U21): `observation_name` (E4), and for the plan reader (E5)
`plan`, `chi_mode`, `reparam_rotate` and `expected_epochs`, the last a window-only setting like the
model builder's (B9, a choice the owner has not seen); their phrases are part 7's (7.1.5). Rewritten first: `BASE_KEYS` and the
count (`tests/test_refusals.py:33-54`, `:116`). Both front-end tables must equal the registry
(`tests/test_refusals.py:995`; `tests/test_nav_and_gating.py:1520`): `core/tool/fields.py` maps all
five to `None`; `core/gui/fields.py` maps them as parts 7 and 8 say.

### 11.12 E9: a simulation set keeps its name

**For.** U17 ("Rename…" for every kind), U21, B26.

**Today.** `write_simulation_manifest` writes `name=""` (`store.py:1582`) each time it runs: when a
set is created (`training_checkpoint.py:217-219`), at every save and at completion (`:395`, `:411`,
through `_refresh_manifest`, `:465-474`). `ArtifactStore.rename` already serves the kind: it writes
the name into the manifest and leaves the folder, which for a set is its digest (`store.py:852-872`;
`core/artifacts/manifest.py:85-89`). So a rename lasts only until the set is next saved or reused.

**Change.** That one value becomes `old.name if old else ""`, as the same call already keeps
`created`, `note`, `prism`, `env`, `inputs` and `parents` (`store.py:1575-1588`).

**Must not change.** The identity, the digest, the folder's name, `config` and `body`: the name is
in none of them (`store.py:1588-1593`). Nothing is printed or logged.

**Edges.** A manifest that cannot be read is still rewritten as new (`store.py:1572-1574`; W160,
for the audit) and then loses its name with its note. A name shaped like a set's id is refused by
today's rule (`store.py:836-841`). On the command line a renamed set's row in `artifacts list`
shows the name where it showed `(unnamed <id>)` (`store.py:160-162`; `core/tool/browse.py:149-155`).
The three docstrings that say a set is always unnamed are corrected (`browse.py:151-152`,
`tests/test_worker_dispatch.py:829`, `tests/_fixtures.py:539`).

**Tests.** New, first, in a fast test with the saving of simulations on (C21; the suites turn it
off, `tests/conftest.py:105-128`): a set renamed through the store keeps its name across a
save, a completion and a reuse; its digest, folder and identity are unchanged; a new set is still
unnamed.

### 11.13 E11: the refusal sentences in U7's words

**For.** U22, with the scope C23 rules: what parts 8 and 10 leave of W113, W131 and W154.

**What.** Every sentence a `Refusal` carries (its five subclasses there included: `StoreError`,
`ManifestError`, and `UnitParseError`, `FDTModelError` and `ModelParseError`, `core/cli.py:29`,
`core/FDT/campaigns.py:31`, `core/Models/user_model.py:31`) and every phrase of the registry
(`core/refusals.py:79-214`), under `core/` outside `core/gui` and `core/tool`: 218 raise lines in
20 files. By a count of `raise Refusal(`, `raise StoreError(`, `raise ManifestError(` and
`refuse(`, 193 lines in 18 files, most of them in `core/artifacts/store.py` (63),
`core/orchestrator.py` (26) and `core/FDT/compare.py` (21); the three other subclasses add 25
(`user_model.py` 16, `campaigns.py` 7, `cli.py` 2), of which six, all in `campaigns.py`, hold a
retired word (`FDT`). Included although outside any raise (C23): the five support sentences of
`registry.fdt_support` (`core/registry.py:96-111`; three say `FDT`), raised at
`core/cli.py:401-405` and shown beside a greyed start button (part 10, 10.3). Of the
109 phrases, twenty hold such a word: the seven `chi_*`, `recording_probe`, `probe_seed`,
`probe_multipliers`, the three `fisher_*`, `checkpoint_every`, `new_run`, `name`, `artifact`,
`repeats`, `accept_truncated` and `n_directions`. The plan reader's problem lines are built from
those phrases, so they follow.

**Rule.** A word in U7's third column, or an engine word for a thing U7 names, gives way to U7's
name: `artifact` to "record"; `cache`, `simulation cache` and `checkpoint` to "simulation set";
`probe` to "driven recording" and `masked` to "unusable"; `chi` to "frequency response";
`Fisher` to "rotation"; `TSNPE`, `truncated` and `non-amortized` to "narrowing round"; `SBC` to
"calibration"; `FDT` to "effective temperature" (`campaigns.py:111-171`, `registry.py:102-110`);
and the store's words for kinds (`fdt`, `diagnostic`, `simulation`) to the word
part 10 gives each kind of record (10.1). Text only: no rule, no order of refusals, no field key,
no number changes (U22).

**Stays.** The names of settings as the engine has them (`n_runs`, `chi_k_pad`) and Python
keywords (`new_run=True`), which a core sentence may name (`core/refusals.py:10-14`): plans are
written under those names (B13) and the card gate reads `n_runs` off one refusal. Engine words U7
does not name ("manifest", "HPD level", "ND units"). Log records, warnings and plain errors, which
are not refusal sentences: the card gate's criteria and part 6's reading table read log lines
(`[chi] masked probes`, `[checkpoint] resuming at batch`). The tool's own usage errors and help
text. (a choice the owner has not seen: these four limits.) So the gate's criterion that a refusal
names `n_runs` holds as written (part 13, 13.1.7), and the fast gate holds it without the card
(`tests/test_tool.py:1160`). Reason strings that are not refusals stay too (C23), such as a driven
recording's verdict the engine only warns of (`core/SBI/observations.py:259-271`).

**Consequences.** The command line prints the new sentences; its flags and the tails it appends
are unchanged. Guide pages that quote a refusal are corrected in the same stage (part 13, 13.3.1);
until then the window shows the engine's present words (part 5, 5.7).

**Order.** The last stage (U22). First the tests: every `match=` and compared literal under
`tests/` that holds a retired word is listed when that stage opens (part 13, 13.1.2), rewritten to
the new sentence and seen to fail; then the strings. One new scan in the pattern of
`tests/test_refusals.py:1067-1078`: no retired word in a registry phrase, in a string inside a
`refuse` call or a raise of `Refusal` or of any of its five subclasses under the scoped folders,
or in a string `fdt_support` returns.

### 11.14 E12: the plotting code's line colours

**For.** U23, B18, C20, ruling D1: E12 is only the colours the plotting files name themselves.

**Today.** Seven files name the colour of a line, a fill or, once, a note in the figure
(`visualizers.py:378`) themselves, at 50 call sites:

| file | sites |
|---|---|
| `core/Helpers/visualizers.py` | `:64`, `:157`, `:161-162`, `:165-172`, `:254`, `:261`, `:266`, `:268`, `:329`, `:344`, `:346`, `:369`, `:371`, `:378`, `:403`, `:405`, `:441-442`, `:445`, `:449` |
| `core/FDT/plots.py` | `:63`, `:77`, `:116`, `:119`, `:192`, `:197`, `:200`, `:251`, `:254`, `:262`, `:265` |
| `core/FDT/compare.py` | `:601`, `:604`, `:859` |
| `core/FDT/cross_validation_plots.py` | `:116` |
| `core/Reduction/plots.py` | `:60`, `:65`, `:69`, `:75`, `:91`, `:94`, `:179` |
| `core/diagnostics/identifiability.py` | `:809` |
| `core/diagnostics/probes.py` | `:1325-1327` |

**Change.** A new module `core/Helpers/plot_colours.py`, standard library only (the FDT and
reduction plotting files do not import `visualizers.py`, which imports torch), holds four colours
by role: the light values of the four hues of the look's `SERIES` (part 4, 4.2). The ink is not
held: it comes from the figure settings (C20). Each site takes its colour from the module:

| today's name | role |
|---|---|
| `steelblue`, `tab:blue` | blue, #0072B2 |
| `red`, `r`, `crimson`, `tab:red` | vermilion, #C24E00 |
| `green`, `tab:green` | green, #007A5A |
| `darkorange`, `orange`, `tab:orange` | orange, #E69F00 |
| `gray` (`visualizers.py:172` only, the crosses for values with no variance) | the axis colour of matplotlib's settings, as the line at `:160` already reads it |

Where two names differ on one axes today, the two roles differ. Because the engine holds the
values, the lines the code colours by name are the same from both front ends. A line that names no
colour follows matplotlib's cycle, which the command line leaves alone and the window sets (part 4,
4.9): the look's colours and nothing else (ruling D1 withdraws C20's dashed repeat). E12 edits no
line style: those lines (`FDT/compare.py:545`, `:607`, `:880`) keep theirs, and no plotting call is
edited for a dash. (C20, a choice the owner has not seen: the engine holds the
four values and this mapping.) A test in the look's suite holds the module
equal to those four of `SERIES`. The two comments that describe a change of theme during a run
(`visualizers.py:20-22`; `Reduction/plots.py:41-43`) are corrected with it.

**Stays.** Every number, array, label, width, marker and transparency, and every dash a call names
(U23). The colour maps (`viridis`, `magma`). White or black lettering and the white line drawn over
a colour map with its outline (`identifiability.py:803`, `probes.py:497`, `Reduction/plots.py:168`,
`:177`). Every colour already read from matplotlib's settings (`visualizers.py:83`, `:160`, `:264`;
`FDT/plots.py:66`), which B18's light settings govern. `save_figure`. The exported video is window
code, not E12's: it follows the light figure settings (C20; part 4, 4.9).

**Tests.** Pinning today: none on a series colour (`tests/test_figures.py:390-393`, `:451-452` pin
table cells and boxes, untouched). New, first: the module imports with neither torch nor
matplotlib; a scan finds none of the names in the table as a colour argument in the seven files;
the numbers each plotting function draws are recorded before the edit and equal after it (stage
1: part 13, 13.2.1).

### Choices in this part the owner has not seen

- If the loss figure cannot be drawn or saved at the end of a training, the training no longer
  fails but saves the posterior without the figure, with a warning, the loss curves staying in the
  record's data file; this holds from the command line as well as from the window, so it changes
  what a command-line run does at its last write, beside the two changes to its printed text the
  owner was told of (ruled by the controller).
- When a name is taken while a prior, a posterior or an observation is being made from the window,
  the record is saved without a name under its id, while a run from the command line still fails
  there as today (ruled by the controller) and every other kind of record still refuses before
  anything is simulated, because it opens its record first.
- One case is left as today (ruled by the controller): when an inference is made together with its
  observation and the inference's name is taken while the observation is being made, the inference
  is refused and the observation is kept.
- The changes that count as touching training, and so owe the test on the graphics card, are the
  per-epoch line, the card test's own move and the two changes to a training's last write (ruled
  by the controller; carried by the part on tests and stages).
- The rewording of refusal sentences replaces only words the approved vocabulary names, so the
  engine's names for settings, engine words the vocabulary does not list (such as "manifest" and
  "HPD level"), log lines, warnings, plain errors and the command line's own usage messages keep
  their text; by the controller's ruling, reasons that are not refusals keep theirs too, and the
  reasons shown beside a greyed start button for an effective-temperature measurement are reworded
  with the refusals.
- The engine's plotting code holds the four line colours itself (blue, vermilion, green, orange),
  kept equal to the look's light colours by a test, with the ink taken from the figure's settings,
  so figures saved from the command line change colour too: today's blues become blue, reds
  vermilion, greens green, oranges orange, and the one grey marker takes the axis colour (ruled by
  the controller; no line style is edited, and the exported video is in the look part's list).
- The third identifiability method's error for a posterior that carries no training prior becomes
  a refusal that points at the posterior (ruled by the controller as an extension).
- Both refusals about continuing a simulation set point at the continue setting, also in the case
  where the setting for how often simulations are saved would answer it as well, and the six check
  errors point at the posterior.
- A stop requested during an epoch takes effect at the new per-epoch line, a few statements
  earlier than today, at the same epoch end.
- If a later version of the fitting library has no per-epoch step to attach to, the training runs
  with one warning and no per-epoch lines, instead of failing.
- The pre-run preview lists every other saved simulation set with its batch count, whether it is
  complete, and every differing setting with both values uncut, while the engine's own refusal
  still fires only for a set exactly one setting away (how a narrowing round shows that refusal is
  in the steps part's list).
- On the command line's list of records, a simulation set renamed in the window shows its name
  where it showed "unnamed".
- The card test becomes one engine function that reports each stage through a callback while the
  command line keeps printing its own lines, and, being one run, each record in the test's own
  scratch store carries the text since the test began.
- The card test's engine function repeats the cell check and the two refusals about continuing, in
  neutral words, for callers other than the command line.
- Only the observation's name is added to an inference (its note stays empty and is edited from
  Records), the name has its own entry in the engine's list of settings, and the command line gets
  no flag for it.
- The engine holds a plan's text form as well as its reader.
- The engine's list of settings gains an entry for the expected number of epochs, which only the
  window uses (ruled by the controller; how a plan names its settings is in the plans part's list).
- The wording of the new lines the engine writes (one per epoch, the warning when a name was taken,
  the warning when the loss figure failed), shown on the run's page as they stand wherever the
  window has no sentence of its own (ruled by the controller; carried by the run part), and that
  on the command line the per-epoch line follows the fitting library's own counter on the same
  console line.

### What this part fixes

- W010: a name taken during a run started from the window no longer costs the result (E7); a command-line run still fails there (C22).
- W023, engine half: every differing setting with both values, uncut (E2); the on-screen names are part 8's table.
- W028, in part: the fit speaks at the end of every epoch; the first epoch is still silent (E1).
- W080, the line colours: the plotting code takes the look's set (E12); the light page is part 4's.
- W093, first half: the input-use check refuses instead of crashing (E6); the second half is said on the Training page (part 8, 8.4).
- W094, W099: the two refusals a continue-only run can meet carry a field key (E6).
- W113, W131, the engine's sentences: refusals use U7's names (E11). W154, the engine's sentence: it says "simulation set" and still names the engine's setting and keyword.
- W130, the engine's half: a simulation set keeps the name "Rename…" gives it (E9).
- W135: the tracker's docstring no longer reads as a live hook (E1).
- W169: all differing settings per saved simulation set, not the first alphabetically (E2).
- W170, engine half: loss and time per epoch reach the full text and the record's log (E1).
- W188, for the card test: it is an engine step a screen can call (E3).
- W190, the observation: it can be named when it is made (E4).
- W192, the engine's side: a failure while the engine draws or saves the loss figure no longer removes the posterior (E10); the window's side is part 6's.
- Without a finding of its own: the plans folder is named in one place (E8).
