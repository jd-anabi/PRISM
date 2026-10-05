# The stopped retrain and the window route — findings, 2026-10-05

A working document, written the day the owner stopped the console retrain and decided that the
retrain will be run from the PRISM window, after a complete UI/UX redesign and then a code audit.
It records what the stopped run measured, what the window cannot yet do (or does unsafely) for a
days-long run, and the leads the audit should start from. `docs/STATE.md` holds the moving state;
the two hand-off prompts are in `docs/superpowers/prompts/`.

**How this was established.** Part 1 and leads 1–5 of part 3A are from the run itself: the console
capture, the records on disk, `nvidia-smi` and the process readings taken while it ran (lead 8 was
added later the same day and has its own note). Part 2,
leads 6 and 7 of part 3A and parts 3B–3C are from a read-only check of the code and the reader pages on
2026-10-05: four readers (cache reuse, a long run in the window, the post-fit steps in the window,
what the project's pages say), three of them followed by a second reader told to refute the first.
All three verdicts were "partly": the main conclusions held, and the corrections are folded in.
The whole document was then checked by six more readers (the run's numbers against the capture,
every code citation against the code), and their corrections are folded in too. **Nothing in the
code-reading parts was executed** — no window was opened and no code was run. Every claim there is
a reading of the code as it stood at commit `d0b1601` (no file under `core/` or `tests/` had
changed since `3ec5bf6`); a line marked *unverified* is one the readers could not settle from the
code. None of it is a decision of the owner's: it is material to bring to the owner.

---

## 1. The stopped retrain (1–5 October 2026)

The owner ran `docs/guide/retrain.md` from PowerShell, with the console captured to
`C:\Users\J\prism-retrain-console.log` (UTF-16, 263 MB). **Keep that file**: it is the only place
that holds the first process's `[fisher]` lines for the rotation stored in the simulation cache
(a resumed process prints none, and a posterior's `log.txt` holds only its last process).

| when | what |
|---|---|
| 1 Oct ~15:52 | Started. Opening lines as the runbook expects; Fisher 8/8 operating points, eigenvalue spread 2.21e+16; the cache created at 15:55:39. |
| 1 Oct 16:23 | **The first process died at batch 237** (`n_fine` 198,874, 2 segments): `torch.AcceleratorError: CUDA error: out of memory` (the solver failed after 99,436 steps of the batch's second simulation, its first probe run by the look of the bars); every split (1,024 / 512 / 256 rows) and every recovery call (`empty_cache`, `mem_get_info`, the RNG restore, a 256-row model construction) failed the same way through all four whole-batch attempts (waits of 15, 60 and 180 s); the rescue save then committed 236 batches. No VRAM ceiling was set. |
| 1 Oct ~16:39 | Resumed by hand with `PRISM_VRAM_CEILING_GIB=8`, `PRISM_MEM_LOG_EVERY=1` and a restart loop around the command (re-run on an out-of-memory failure, at most 5 times). |
| 1 Oct 17:12 | Stopped by hand at the batch-500 commit and restarted with the ceiling at **6** and the loop at 20 tries, because the longest batches were at the card's edge under 8 (below). |
| 2 Oct 18:46 | **Generation ended**: 10,000 batches. After the first death no process logged an out-of-memory line or a failure, and **the restart loop never had to act**: the run was three processes, the second and third both started by hand. |
| 2 Oct ~19:00 → 5 Oct ~10:25 | The fit: 62 epochs finished, at 61–63 minutes each (the capture's last count is 61: a count reaches the file only when the next one is printed). **Stopped by the owner** (Ctrl-C) about 63 hours in; the fit is lost, as any stopped fit is. |

**What is on disk** (the owner has not asked for any of it to be deleted):

- `Artifacts/priors/retrain_prior__20261001T222935` — the tier-1 chi prior (sweep kept 101,777 of
  102,400; fill kept 73,960; 2 components), fingerprint `7855ed3cd42ae05b`.
- `Artifacts/simulations/eda5c2a1e468` — the simulation cache, **complete**: 10,000 × 2,048 =
  20,480,000 rows × 122 columns, ~10.3 GiB.
- `Artifacts/diagnostics/retrain_band__20261001T222941` and `retrain_mask__20261001T223011` — the
  pre-flight's probe checks (the mask check read P = 37.9 % over 100 batches).
- No posterior: the fit never finished.

**What the run measured**

- *Gate 3 passed on that cache.* `[chi] masked probes: 57,703,562 of 143,831,040 (40.1%) over
  10000 batches, every committed batch of the simulation cache; per batch 16.5-92.9%, median
  37.8%`, against P = 37.9 %: 2.2 points, inside the 3-point limit. That reading belongs to
  `retrain_prior` and to the code of 1 October.
- *Failed trajectories.* 30,847 of 20,480,000 (0.151 %), 30,825 of them non-finite, 22 over 1e+15.
  This had to be summed by hand over the three processes (727 + 875 + 29,245): the
  `[patho] run total` line counts only the process that prints it.
- *Clipping.* `[winsor] clipped 852,885 of 1,024,000,000 summary elements (0.083%)`; the runbook
  expects "a few tenths of a percent".
- *Card memory during generation* (peak allocated / peak reserved, GiB, per batch, from the
  `[mem]` lines):

  | ceiling | the longest batches | all batches of the phase |
  |---|---|---|
  | none | died at batch 237 (`n_fine` 198,874) | one `[mem]` line only, batch 1 |
  | 8 GiB | `n_fine` 268–297k (ten batches): 9.1–10.1 / **13.4–14.8** | 18 of 269 batches reserved ≥ 12 |
  | 6 GiB | `n_fine` 299,998 (the cap is 300,000): 10.43 / **11.81**; the highest allocated was 10.50 | none ≥ 12 in 9,500 batches |

  The desktop and the CUDA context held about 1.3 GiB of the card's 15.92 (the floor of a
  one-minute `nvidia-smi` sample), so a batch that reserves 14.8 GiB went through only on Windows'
  overcommit, the mode the first process died in. Under 8 GiB a split batch reserved about 45 %
  more than it allocated; under 6 GiB the two nearly match.
- *Pace of generation.* 6.0 s per batch with no ceiling (236 batches in 23 min 30 s; 5.2 was the
  running average at batch 90); 7.2 s at 8 GiB with a memory line per batch (269 batches in 32 min
  20 s; 8.5 over batches 309–386 alone); 9.7 s at 6 GiB (9,500 batches in 25 h 32 min). The cache's
  manifest records 96,666 s (26.9 h) of wall clock, gaps included.
- *Pace of the fit.* 63.0 minutes for one timed epoch (the capture's count 60 → 61, which is sbi
  printing 61 → 62: the 62nd epoch), and at most 61.3 minutes on average over the 61 before it.
  The pre-flight timing read 6.2 s per epoch at 20 batches, and the runbook's rule ("about 500
  times") projected 52 minutes: the factor measured is about 610.
- *Host memory during the fit.* A flat 24.7 GB working set (29 GB private) for the 20.48M-row
  table, with 17 GB of 63 free; the runbook expects about 50 of 63.

---

## 2. What the window cannot do, or does unsafely, on a days-long run

The window and the command-line tool call the same stage functions, and a training run started in
the window with the retrain's settings should resolve to the same simulation cache: the cache's
key holds 34 science settings and nothing about the front end or about the code that made the rows
(`core/artifacts/identity.py:87-122`). The differences are in what surrounds the run.

### 2A. Before the button

- **The network size reopens at 128 × 8 on every launch** and is not part of the cache's key, so a
  relaunch that forgets to retype 256 and 10 fits the smaller network for days with no refusal and
  no warning (`posterior_tab.py:85-109`, `fields.py:82-89`; the runbook gives this as its reason
  for the command line, `docs/guide/retrain.md:93-98`).
- **Protection against an accidental fresh run is thin.** The window always resumes on "auto";
  there is no "must resume" choice. The "This starts a NEW run" dialog (its default is Cancel)
  appears only when a committed cache is exactly one setting away (`posterior_tab.py:183-238`).
  Two or more differing settings give only a passive line under the budget boxes, and the run
  simulates from zero.
- **That passive line can name the wrong thing.** It reports the first differing key in
  alphabetical order (`describe_siblings`, `training_checkpoint.py:285-301`), so the tier-1 box
  swapped for `master.txt` reads as "differs in param_keys" (the two files' non-dimensional limits
  are identical), and the prior, when it differs too, is never the key named. The one-setting
  dialog is not affected.
- **The remembered selections can be wrong for the run.** On 5 October the owner's `PRISM.ini` held
  Bounds = `master.txt` (not the tier-1 box) and an empty Prior pick, which restores to "(from
  scratch)". Pressing **Build / Load prior** in that state builds a new prior on the wrong box.
- **A launch that is closed before "Apply model & options" is pressed overwrites the remembered
  Bounds.** The picker opens on the Nadrowski folder's first file, `master.txt`
  (`prior_tab.py:70`); the remembered key is only stashed at launch (`:373`) and applied when a
  model is applied (`:141`); and every accepted close writes whatever the picker shows (`:363`,
  `main_window.py:447-452`). One look-and-close session therefore replaces a remembered
  `master_tier1.txt` with `master.txt`, the value found on 5 October (whether that is how it got
  there was not established). The Infer tab's Cell is stashed and written the same way
  (`base.py:36-40`, `infer_tab.py:530`, `:545`).
- **Choosing a prior in the picker loads nothing** until **Build / Load prior** is pressed, and the
  line under the picker describes the entry shown, not the prior the session holds
  (`docs/guide/window.md:55-59`).
- **Every confirmed press of "Apply model & options" re-selects the Bounds remembered at launch and
  releases the loaded prior** (`prior_tab.py:141`, `:373`; `screens/inference_screen.py:163-164`).
  The release is asked first: when the session holds a configuration, a prior, a posterior or an
  observation, Apply opens "Start a new session?", lists them and defaults to **Keep this session**
  (`config_tab.py:263-303`, `:339-340`), and a line under the button says what the session holds.
  The Bounds jump is silent and the dialog does not mention it: a pick made during the session
  never updates the key it restores, so after a correction on the Config tab the Prior tab shows
  the launch-time bounds file again.
- **The VRAM ceiling resets to 0 on every launch** unless `PRISM_VRAM_CEILING_GIB` is set in the
  environment the window starts in (the field then opens at that value and the variable wins), and
  it is locked while a run is live (`config_tab.py:108-165`). Part 1 shows a full-size run on this
  card with no ceiling dies. The per-batch memory line's cadence is an environment variable read at
  import, with no control.
- **No name and no note before the run.** The posterior is written unnamed and named afterwards with
  **Save**, so a taken name is discovered after days, not before (`posterior_tab.py:148-150`,
  `:296-317`). The note is set later on the Artifacts screen, which refuses while a run is live.
- **No control at all** for an epoch ceiling, the checkpoint cadence (always 50), the resume policy
  (always auto) or the device (always auto) (`fields.py:60-61`, `:86`, `:125`). The training stage
  already takes a name, a note, an epoch ceiling, a checkpoint cadence and a resume policy as
  arguments, and refuses a taken name before anything is spent (`orchestrator.py:792-799`, `:892`);
  the command line passes them. Offering them is a change to the window only.
- **Launching the window touches the card**: building the Posterior tab asks the planner for free
  memory, which creates a CUDA context before any button is pressed (`posterior_tab.py:112` →
  `base.py:138-139` → `orchestrator.py:529-532` → `config.py:122-130`). The same launch also calls
  `config.detect_device()` from the Prior tab (`prior_tab.py:121` → `:297`) and runs `nvidia-smi`
  from the Config tab (`config_tab.py:119` → `:151`), again on every keystroke in the ceiling box.
  A window must not be opened beside another process's run.

### 2B. During the run

- **No lasting record of the run's lines.** Output goes to the tab's log pane only: an in-memory
  text box capped at 5,000 lines, without timestamps or a save control (`streams.py:283-299`,
  `widgets/log_pane.py:15`). A full-size generation wrote a `[patho]` warning for nearly every
  batch (9,551 in the stopped run's capture; `pipeline.py:1925-1930`), so the opening lines scroll
  out. The record's `log.txt` is written only when the posterior commits at the very end
  (`runs.py:17-21`, `store.py:340-344`, `:552-558`).
- **The fit's progress is one caption.** sbi's epoch counter shows as the progress pane's caption
  under an indeterminate bar; it carries no clock time and reaches no file, and no loss is shown
  until the fit ends (`widgets/progress_pane.py:16-22`, `:376-389`). Pacing a fit means noting the
  count and the clock by hand.
- **"no output for N s" stands through every epoch** (read from code): the stall notice fires after
  45 s of silence, and a full-size epoch is silent for about an hour (`progress_pane.py:69`,
  `:329-340`).
- **The whole window is locked for the days the run takes**: every panel's controls are disabled,
  only **Cancel** stays live, and the Artifacts screen's Note, Delete and Sweep are refused
  (`base_panel.py:221-225`, `:302-333`). Nothing guards against a run in *another* process: the
  store has no cross-process lock (`artifact_screen.py:122-124`).
- **An out-of-memory failure ends in a red box and waits.** The in-process recovery is the
  console's (two halving ladders, three whole-batch retries); when it is exhausted the generator
  rescue-saves, a modal "Error" opens, and nothing restarts the run
  (`pipeline.py:1817-1877`, `:1951-1990`; `worker.py:146-162`). A relaunch then resets the network
  boxes and the ceiling (2A). The selections and the training budget are saved only when a close is
  accepted (`main_window.py:444-456`), so after a crash or a killed process the relaunch shows what
  the last clean close saved; Batches is part of the cache's key.

### 2C. Stopping and closing

- **The fit is not saved as it goes**, in either front end: it is one training call with no
  checkpoint (`core/SBI/train.py:365-370`). A cancel, a close, a crash or a restart of the machine
  during the fit loses all of it; the simulation cache is unaffected.
- **Cancel during the fit lands at the next epoch boundary** — about an hour at full size — while
  the line Cancel writes to the log pane once pressed, and the close dialog, say "up to ~1 min
  during training" (`base_panel.py:299-300`, `main_window.py:429`;
  `docs/guide/architecture.md:667-668` says one epoch). The tab's **Cancel** acts on a single
  click, with no confirmation (`base_panel.py:150-152`, `:291-296`), it is the only live control on
  the tab for the whole fit, it cannot be withdrawn, and one pressed in the final epoch discards a
  fit that has just converged.
- **The close dialog's default button is "Cancel task & quit"** (`main_window.py:435`): an
  accidental close followed by Enter cancels a multi-day fit.
- **"Quit anyway" probably loses the record at the end** (*unverified, read from code*). The guide
  says it "leaves the run going in the background until it finishes", and the close dialog itself
  says "the task keeps running in the background until it finishes on its own"
  (`main_window.py:430-431`). The code's own comments say that after the window closes the Qt
  signal objects are gone and emits raise (`worker.py:207-210`, `streams.py:152-155`); the loss
  figure is forwarded by an unguarded emit (`base_panel.py:62`) from inside the posterior writer
  (`store.py:404-409`, `orchestrator.py:1420-1431`), and the writer removes its directory on any
  exception (`store.py:342-344`), after which the worker swallows the error. What Qt does to the
  worker thread at process exit could not be settled from the repository.
- **During generation the tab's Cancel is the stronger stop**: it rescue-saves the batches since
  the last commit (`pipeline.py:1951-1990`), where Ctrl-C under the console's capture pipe loses
  them. "Cancel task & quit" requests the same cancel and then closes the window at once, without
  waiting for the run to unwind (`main_window.py:438-445`); whether the rescue save then completes
  rests on the same unsettled question as "Quit anyway" (*unverified*; the walkthrough's row 20
  passed once, on 2026-09-11, at a small run). During a fit that button leaves the process fitting
  with no window until the epoch ends.

### 2D. Around the fit: what the runbook's other steps need

| runbook step | in the window | missing |
|---|---|---|
| pre-flight: `smoke`, `identifiability jacobian`, `probes band / mask` (and the diagnostic card's `probes drive`) | **nothing** | all of them (`fields.py:178-186`) |
| `validate` (gate 1) | Validate tab; the two boxes open at the runbook's values | a seed box (one is drawn and recorded); a name; a note at the run |
| `identifiability rotation` (gate 4) | **nothing** | the whole diagnostic; and the Artifacts pane abbreviates any list of more than 12 numbers to its first three, so a 13-parameter posterior's eigenvalues show as three of thirteen (`core/artifacts/report.py:30`, `:57-60`). A calibration's per-parameter informativeness list is cut the same way, but its `description` text in the same block, and the record's log, give all thirteen |
| `ablation` (gate 2) | **nothing** | the whole diagnostic |
| the four stratified `sbc` runs | **nothing** | a fixed probe count is not an argument of the calibration stage |
| `infer` | Infer tab | a name; the draw count (the default equals the runbook's) |
| `tsnpe` (the certification round) | TSNPE tab | **the network size**: a round trains at 128 × 8 whatever the Posterior tab says (`tsnpe_tab.py:157-161`, `orchestrator.py:936-937`); also resume policy and epoch ceiling; the tab cannot say in advance whether a round resumes |
| `validate` on the round | Validate tab | run straight after the round, its record's `accepted` is empty where the runbook expects `truncated`; reloading the round through the "Load a NON-AMORTIZED posterior?" dialog first records it (`orchestrator.py:1488`). The same holds for the optional inference on the round |
| note and lineage report | Artifacts screen: the Note box and **Lineage report…** | figures (the pane is text only); the pane shows the last 1 MiB of a `log.txt` |

Calibrations, inferences and observations made in the window stay `_unnamed__<id>` for good: on the
inference tabs only priors and posteriors can be renamed (`prior_tab.py:328`, `posterior_tab.py:307`;
the FDT and sweep-study tabs name their own records), and the TSNPE tab's Observation picker
restores its remembered entry rather than the observation just written.

### 2E. What the window already does well (keep it, or say why not)

- Before the button: the row total, the worst-case batch memory against the planner's budget, and
  whether the run resumes a cache or starts a new one.
- A dialog, defaulting to Cancel, on a one-setting near miss, and a dialog before a session that
  holds work is replaced.
- The VRAM ceiling as a visible field with a note showing free memory and whether the environment
  overrides it.
- Live progress (the header's elapsed clock, the overall bar, the solver rate), the loss figure at
  the end, and the finished posterior held by the session for Validate and Infer at once.
- The rescue save on the tab's Cancel during generation (2C).

### 2F. Never exercised

On today's code the window's training path has been seen on a real screen only at 2 to 4 batches
(`docs/checklists/display-walkthrough.md`). Not in the checklist at all: the VRAM ceiling field,
"Quit anyway", a training at 2,048 rows or thousands of batches, epochs longer than the stall
threshold, the red box after a training crash, a resume of a large cache after a relaunch, and
256 × 10 typed for a real run. Every card gate in the working record is a command-line run.

---

## 3. Leads for the code audit

### 3A. From the run

1. **A full-size run with every default in place runs out of memory on this card.** With no
   ceiling the first process died at batch 237; the window's ceiling opens at 0. The ceiling caps
   only the budget `gen_obs` plans its chunks against (`pipeline.py:606-640`, `:643-683`). That
   plan counts a drive for the rows of one chunk (`pipeline.py:577`) and, when it splits, reserves
   the whole result (`:670`). What it leaves out is outside `gen_obs`: the probe stage builds the
   drive for the *whole* row block before the chunking (`core/SBI/chi_probes.py:263`, handed to
   `gen_obs` at `:273`), and building it peaks at about four tensors of that size
   (`pipeline.py:67-75`, charged only to the learned cap after a failure, `:1785-1787`). The
   capture's `[mem]` lines fit that: on the longest batches the peak allocated is 4.4–4.6 times one
   2,048 × `n_fine` float32 tensor under either ceiling. So the ceiling is not a bound on what a
   batch holds (9.1–10.5 GiB allocated on the longest batches under a 6–8 GiB ceiling). What a
   default run should do, and whether the plan should count the drive's build, is the owner's to
   decide.
2. **An out-of-memory failure that leaves `empty_cache` and `mem_get_info` failing cannot be
   recovered in the process**, yet the ladders run their full course (minutes of waits) before the
   rescue save. Owed since 1 October: fail fast and distinctly in that state; consider a
   per-process memory fraction as an argument (it might turn the overcommit into a clean
   out-of-memory error the ladders handle; untested); swap `empty_cache` and the graph drop in
   `_release_device_memory`.
3. **The fit has no checkpoint.** About sixty-three hours were lost by choice this time; a power
   cut or a forced Windows restart would lose them just the same.
4. **`[patho] run total` counts one process.** A resumed run under-reports unless someone sums the
   per-process lines.
5. **The epoch projection is 20 % short** (factor ~610, not 500), and **the clipping expectation is
   off** (0.083 % against "a few tenths"). Either the runbook's numbers or the code's behaviour
   needs explaining.
6. **What a source edit or a commit during a run does to it** (*read from code, not run*). A
   posterior stamps the checkout's git revision and dirty flag when its manifest is built at the
   end, not when the run starts (`store.py:598-599`), so a commit of any file during the run,
   documents included, moves that stamp, and an uncommitted edit marks the record dirty: twelve
   documents commits were made during the stopped run. Imports also sit inside functions on the
   save path (the figure sink's, `store.py:405-406`, among them); those found name modules
   `core/orchestrator.py` has already loaded at import, which a later edit cannot reach, and
   whether any import on a run's path names a module not yet loaded was not established. Until it
   is, no file under `core/` is edited while a long run is live.
7. **The cache's key knows nothing about the code.** A fix that changes what a simulated row holds
   and leaves the settings alone leaves the key alone, and a training started with the same
   settings then resumes the finished cache and fits rows the old code made, without a word.
8. **Model time is put on the recording's clock by a whole-number stride** — found the same day
   while mapping where the models' math lives, and reproduced with the program's own functions:
   each simulated trace is that of a cell with a different time scale from its label, and the
   drive is detuned from the lock-in that measures the response. It is lead 1 of
   `docs/superpowers/notes/2026-10-05-where-the-models-math-lives.md`, the note the audit's math
   check starts from; if it stands, the finished cache cannot be trained on.

### 3B. From the window's code (read, not run)

8. "Quit anyway" during a fit, and "Cancel task & quit" during generation (2C) — the
   highest-stakes items here: establish what really happens and make each safe or remove it.
9. The waiting time shown for Cancel is wrong by a factor of sixty at full size, and Cancel takes
   one click with no confirmation (2C).
10. Reusing a complete cache rewrites its manifest: `wall_seconds` is recomputed as now minus
    created (`store.py:1576-1577`, via `pipeline.py:2038`), so the recorded generation time is
    replaced by a figure of days.
11. The passive "these settings match no checkpoint" line names only the first differing key
    alphabetically (`describe_siblings`, `training_checkpoint.py:290-295`; 2A).
12. The Artifacts pane's list abbreviation hides ten of thirteen eigenvalues (2D).
13. **Edit values** on the Bounds grid: the cache's key compares floats by `repr`
    (`training_checkpoint.py:105-106`); whether a round trip through the grid reproduces the file's
    numbers was not traced.
14. A window's "Delete" on the Artifacts screen is not blocked by a run in another process (2B).
15. A round just trained carries an empty `accepted` list into the records made from it (2D).
16. The remembered Bounds is overwritten by a look-and-close session, and a crash loses the
    session's selections and budget (2A, 2B).

### 3C. Page errors found on the way (fix when the pages are next rewritten)

- `docs/guide/window.md`: Cancel "can take about a minute during training" (it is up to an epoch);
  "Quit anyway … leaves the run going in the background until it finishes" (see 2C); "0, the
  default, is right on an idle card" for the VRAM ceiling (false at full size on this card: with
  no ceiling the 1 October run died at batch 237, and 6 GiB was the first setting that held); no
  mention of the 5,000-line pane or the stall notice.
- `docs/guide/retrain.md`: the ×500 epoch projection; the `[winsor]` expectation; "None of this was
  timed on the card" (now timed: 6.2 s at the timing's size, 63 minutes at full size); pre-flight
  step 5's card-memory margin ("about 11.44 GiB against a budget of about 12.48 GiB", "A run
  survives a busy card by splitting its batches": the run reserved up to 14.80 GiB under an 8 GiB
  ceiling and died with none, and the page names no ceiling); its host-memory expectation of about
  50 of 63 GiB (measured: 24.7 GB working set); and the whole page is console-only, where the
  owner wants the window route.

The open list in `docs/STATE.md` ("Open, with no piece owning them yet", and the parked blocks
that follow it) is the audit's other starting point; none of it was re-examined here.

---

## 4. Questions the readers left for the owner

None is answered. They belong to the redesign's brainstorming or to the audit.

- Should the certification round match the retrain's network (256 × 10)? The window cannot do that
  today.
- Should the diagnostics behind gates 2 and 4, and the stratified calibration runs, be reachable
  from the window?
- Is a calibration seed drawn by the window acceptable for gate 1, or must it be chosen beforehand?
- Are unnamed calibration and inference records acceptable, or must they be nameable?
- What should happen after a failure in a days-long window run, with no restart loop?
- What replaces the console capture: a timestamped file written as the run goes?
- Should the fit be saved as it goes, so that a stop does not lose it?
- What should a run started with every default do about card memory on this card?
- When the retrain is run again: train on the complete cache (the window does so by itself when the
  settings match, and gate 3 has already passed on it) or simulate afresh (a new prior, or deleting
  the cache)? On 5 October the owner said reuse is not required. If a fix changes what a simulated
  row holds, reuse is off the table (3A, leads 7 and 8).
- Does the four-week budget of 30 September stand as it was? The stopped run spent about 90 hours
  of the card (27 of generation, 63 of fit).
