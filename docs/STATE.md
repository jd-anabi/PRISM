# PRISM — state

**Last updated:** 2026-09-11 (piece 1 MERGED as `461d780`; the CLEAN-BREAK RUNBOOK is EXECUTED —
`8dbd93e`, `e37df41`, `19363d2`, `d5c9fd3`; the POST-PIECE-1 GPU GATE is GREEN at `bb38f1a` after
its first attempt found two piece-1 regressions, fixed in `896e8ff` and `bb38f1a` with a new
gpu-marked suite; the NINE GUI CHECKS PASS on the user's screen, and the one defect they turned up —
the taskbar showing the generic glyph — is fixed in `df7491e`; `Artifacts/` exists now with the
walkthrough's artifacts; next: piece 2; all work happens directly on the local `main` branch)

## Where things stand

- The pre-retrain hardening programme is approved: the seven-piece decomposition
  (`docs/superpowers/specs/2026-09-10-pre-retrain-hardening-decomposition.md`) and the piece-1
  artifact-store design (`docs/superpowers/specs/2026-09-10-artifact-store-design.md`, whose §11
  lists the ten deviations ruled during execution).
- On `main`: the 2026-09-09/10 TSNPE fixes X1–X5 and the parent-prior refusal (`ed16397`), the
  specs (`aff8f7e`), piece 0 (`4b1834a`..`d05d3cd`, `3913a1b`), the piece-1 plan (`afa484f`).
- **Piece 0 is DONE** (2026-09-10): pytest, `CLAUDE.md`, this file, the launchers, the display
  checklist, the GPU smoke baseline (table below).
- **Piece 1 is MERGED** (2026-09-11, merge commit `461d780`; the branch and its worktree are
  deleted). It was built on branch `piece-1-artifact-store` (forked from `main` at `8ec18aa`,
  28 commits, tip `7bce752`; `main`'s own `70cd11a` sandboxed the old artifact-consistency tests
  and is superseded by the branch's rewrite of that file): the 14 tasks of
  `docs/superpowers/plans/2026-09-10-artifact-store.md`, one commit each with a per-task review
  and fix loop, then a whole-branch review ("ready with fixes"), one fix wave (`830cee1`) and one
  residual fix (`3f1a1b3`). The execution ledger (rulings R1–R7, every parked finding, every
  review verdict) was git-ignored scratch and is deleted with the workspace once the branch is
  finished; its rulings live on in the decisions log below and in the design spec's §11, and
  the items it carried forward are in "Owed" item 5. What landed: `core/artifacts/`
  (store, writer, manifests, provenance, identity); every stage takes and returns `Loaded*`
  wrappers and writes its artifact at completion under `Artifacts/<kind>/<name>__<id>/`; the
  `.rot.pt` sidecar, the save/load helpers and the generated-path constants are gone; loading
  refuses any verifiable mismatch with `Accept(truncated, other_observation)` as the only
  hatches; the simulation identity is `training-rows/2`; pickers list the store; scripts are
  re-pointed (nine marked broken until piece 2). Defects 3–4 of the decomposition are fixed
  (the forced-recording path runs); defect 1 disappeared with `load_prior`; defect 2 is piece 3's.
- **Decision 2026-09-10: CLEAN BREAK.** Every generated artifact is deleted once piece 1 is
  merged (runbook: design spec §9 = plan Task 14). The old runbook's Run A, Run B and the TSNPE
  round (`PRISM_HANDOFF.md` §11.9) are ABANDONED; the retrain restarts from scratch after piece 6.
  **Executed 2026-09-11** (section "Clean break" below).

## Owed, in order

1. ~~Merge piece 1~~ — done 2026-09-11 (`461d780`); the one-process fast gate on merged `main`
   is recorded in the table below.
2. ~~Clean-break runbook~~ — done 2026-09-11 (design spec §9 / plan Task 14, run by Claude at
   the user's request; the "Clean break" section below records every step, the one deviation
   and the fast gate afterwards).
3. ~~GPU gate after piece 1~~ — done 2026-09-11, GREEN at `bb38f1a` (table below). **The first
   attempt at `19363d2` FAILED twice** and both failures were piece-1 regressions on paths no CPU
   suite can reach: (a) infer's PPC met a CUDA observation row and CPU simulated rows —
   `store.load_observation` handed the row back on `cfg.hw.device` and `infer_and_visualize` bound
   it, while `statistics.conditioning_rows` assembles every row on the CPU by contract; (b) the
   forced-mode prior was refused by its own store the moment `build_prior` read it back
   ("prior.pt holds GMM …, not the one the manifest records") — `file_manager.load_mix_dist`
   rebuilt `Categorical(probs=w)`, whose re-normalisation `w / w.sum()` is not a bitwise fixed
   point in float32 and depends on the device's reduction order (the chi prior round-tripped
   exactly on CUDA and not on the CPU; the forced one on neither). Fixes, each with a failing test
   first: `896e8ff` the loader restores the saved weights into `.probs`/`._param` (exact round
   trip; two CPU tests reproduce the refusal); `bb38f1a` the row stays on the CPU and `sample()`
   moves its own copy, pinned by the new gpu-marked `tests/test_gpu_paths.py` (the tiny SBITEST
   chain on `config.detect_device()`, ~12 s; `build_tiny_run(store, hw=None)`).
   **Recipe correction:** run 2 must use the SAME `NUM_RUNS` as run 1 — the simulation identity
   includes `n_runs`, so `NUM_RUNS=2` after a `NUM_RUNS=4` run 1 keys a NEW cache directory,
   rebuilds the Fisher rotation and never resumes (it did exactly that here, silently, exit 0).
   The drill that certifies the resume is run 1 `CHI=1 TOBS_S=4.5 BOUNDS=…/master.txt
   CELL=…/master_spont.txt CHECKPOINT=1 SAVE=1 CKPT_DIR=<scratch>/smoke`, then run 2 with
   `PRIOR=smoke_prior STAGES=prior,posterior`, the same `NUM_RUNS`, `SAVE` unset.
4. ~~Manual GUI check on a display~~ — done 2026-09-11 by the user on the real screen: rows A1–A9
   of `docs/checklists/display-walkthrough.md` all pass (reported "everything passes"). The one
   failure was the walkthrough's row 1, the APP ICON: the window's title bar showed the mark, the
   taskbar button the generic Windows "application" glyph. Root-caused the same day by screenshot
   experiments (the "Taskbar icon" entry in the decisions log) and fixed by installing the mark as
   Qt's Win32 window-CLASS icon from a new `assets/app/prism.ico`
   (`app_icon.set_windows_class_icon`); verified by a screenshot of the real launcher's button.
   Re-check row 1 on the next launch.
5. **Pieces 2 → 3 → (4 ∥ 5) → 6**, each brainstormed → spec → plan → implementation. Carried
   into them from piece 1 (spec §11 and the ledger): **piece 2** extends the source scan to
   `scripts/`, retires `_common.require_mode` and the unconditional `Accept(truncated=True)` in
   `_common.load_posterior`, threads the store into the TSNPE runner, fixes the stale
   comments the clean break left (listed below), and makes the resume drill loud: with
   `CKPT_DIR` and `PRIOR` set, a run whose identity resolves to a NEW simulation directory beside
   a complete one should say which field differs (`n_runs` here) instead of quietly regenerating;
   **piece 3** copy-on-run session config (a
   refused stage must leave nothing on the session), `tsnpe_tab.restore_settings`; **piece 4**
   annotate (`set_note` has no GUI caller), cleanup of incomplete directories,
   `Summary.complete` for simulations means "has a manifest", the observation width guard in
   `load_observation` has no isolated test.
6. **The retrain.**

## Clean break — EXECUTED 2026-09-11 (design spec §9 / plan Task 14)

- **Pre-flight.** A read-only four-lens audit (runtime code, tests, scripts and imports,
  Reduction and misc) found NO live dependency on the seven trees, on `sbc_run.log` or on the
  two scripts: every generated kind resolves through the store or `artifacts_root()`, the only
  on-disk assertion in the suites is that `Resources/Bounds` is a directory, the one source scan
  that walks `scripts/` (`tests/test_conditioning_repair.py:923`) has no needle in either script,
  and `core/Reduction` writes only under `artifacts_root()/reduction`. Only stale comments (below).
- **`8dbd93e`** — `git rm --cached` of the 20 tracked-but-ignored files: 7 `CrossValidation/*.h5`,
  5 `Plots/*`, `Priors/shm.pt`, 6 `ReductionMap/*.parquet`, `sbc_run.log` (6.9 MB, still in
  history). `git ls-files -i -c --exclude-standard` is empty since.
- **Deleted from disk** (no commit; all gitignored): `Resources/{Priors,Posteriors,Checkpoints,
  Observations,Plots,CrossValidation,ReductionMap}` — 1351 files, ~32 GB (Checkpoints alone:
  `QUARANTINED_tsnpe_round1_truncated_rows_train_0b471d560271` 5.2 G, `train_3780fd37a16a` 11 G,
  `train_230ae7cb5fc2` 5.2 G, `train_6c80f7d8037d` 4.9 G, `train_98aebd93ed17` 4.9 G,
  `train_85226b80f5d1` 933 M, `train_005ff030d387` 371 M, `train_24922afa61da` 98 M) — and
  `sbc_run.log`. `Resources/` now holds only `Bounds/ Cells/ Units/ Models/` (9 / 10 / 5 / 2
  files). `archive/` untouched. `Artifacts/` does not exist yet; the GUI creates it at launch.
- **`e37df41`** — `scripts/tsnpe_round1_forensics.py` and `scripts/migrate_checkpoint_flags.py`
  moved on disk into the gitignored `archive/scripts/` and `git rm --cached`. **The one deviation
  from the plan's letter:** the plan said `git mv`, but `archive/` is ignored and holds nothing
  tracked, so a `git mv` would have re-created exactly the tracked-but-ignored class step 1
  removed; `0a84fc8` archived the five diagnostics already there the same way. Nothing imported
  either script; `scripts/` keeps 13 files.
- **`19363d2`** — `.gitignore`: the seven `/Resources/*` lines and their comment block are gone;
  `/sbc_run.log` stays ignored (its comment now says untracked and deleted); `/Artifacts/` stays.
- **Docs** — this file; `CLAUDE.md`'s sentence about the old trees is past tense now.
- **Left alone, for piece 2** (stale comments, none a code path): `core/gui/screens/
  inference_screen.py:122` and `tests/test_nav_and_gating.py:273` (say the observation gate reads
  `Resources/Observations`), `core/SBI/pipeline.py:1411` and `core/SBI/training_checkpoint.py:163`
  ("a read-only `Resources/`" for a write that goes under `Artifacts/`), `core/SBI/statistics.py:506`
  (the substitution-rate table's provenance was `Resources/Checkpoints/train_98aebd93ed17`, now
  deleted — the numbers stand as history), `tests/test_user_sbi.py:47` (checkpoints "into
  `Resources/Checkpoints`") and `:2209` (cites the archived `migrate_checkpoint_flags.py` as the
  digest-migration precedent), `scripts/smoke_train.py:6`, and `run.sh:4-5` plus
  `scripts/generate_bundle_videos.py:82-84` (both claim `config.py` resolves `Resources/` from the
  CWD; it resolves from its own location). Also seen and left alone: stale
  `.claude/worktrees/{jolly-jang,trusting-einstein,upbeat-rhodes-c8d30f}` directories (git lists
  no worktrees) and six old `claude/*` branches.

## Last gate

| gate | result |
|---|---|
| `pytest --collect-only -q` | 2026-09-11 at `3f1a1b3`: 356 (355 run by the fast suite + the slow one; Reduction's 5 are collected with it) |
| fast suite, ONE process, `pytest -m "not slow" -q` | 2026-09-11 at `3f1a1b3` (branch tip): 355 passed, 1 deselected, 10 min 36 s, exit 0. **On merged `main` `461d780`: 355 passed, 1 deselected**, 115 warnings, 10 min 56 s, exit 0; the real `Artifacts/` gained nothing and the user-model suite's temporary `Resources/*/sbitest` inputs were cleaned up (the conftest teardown assertion and `git status` both clean afterwards) |
| fast suite AFTER THE CLEAN BREAK, ONE process, `pytest -m "not slow" -q` | 2026-09-11 at `19363d2` (trees deleted, scripts archived, `.gitignore` trimmed; `CLAUDE.md` and this file edited in the working tree): **355 passed, 1 deselected**, 115 warnings, 10 min 54 s, exit 0; the real `Artifacts/` still absent afterwards and the user-model suite's temporary `Resources/*/sbitest` inputs cleaned up (`git status` showed only the two doc edits) |
| fast suite AFTER THE GPU-GATE FIXES, ONE process, `pytest -m "not slow" -q` | 2026-09-11 on the working tree that became `896e8ff`+`bb38f1a` (identical content): **358 passed, 1 deselected** (355 + the three new tests, one of them the gpu-marked CUDA inference test), 125 warnings, 11 min 13 s, exit 0; the real `Artifacts/` still absent |
| fast suite AFTER THE TASKBAR-ICON FIX, ONE process, `pytest -m "not slow" -q` | 2026-09-11 at `df7491e` (the tree was committed before the run finished; identical content): **359 passed, 1 skipped** (the display-marked class-icon test, offscreen), 1 deselected, 125 warnings, 11 min 08 s, exit 0; the real `Artifacts/` — which exists now, created by the user's walkthrough launch — gained nothing |
| display-marked tests, `QT_QPA_PLATFORM=windows pytest -m display` (real screen, hidden native windows only) | 2026-09-11: `test_the_window_class_icon_becomes_ours_on_a_real_windows_display` 1 passed (RED before `df7491e` at the missing function, with the premise assertion — Qt's class icon is the stock IDI_APPLICATION handle — already passing) |
| `tests/test_gpu_paths.py` (gpu-marked; skipped without CUDA) | 2026-09-11: 1 passed on the RTX 5070 Ti, ~10 s fixture + 2 s test; RED before `bb38f1a` with the gate's exact RuntimeError (two devices in `analysis.posterior_predictive_check`) |
| slow test `pytest tests/test_user_sbi.py -m slow -q` | 2026-09-11 at `e4e60eb` (Task 13): 1 passed, 31 min 16 s, exit 0; re-run at `3f1a1b3` after the fix wave: **1 passed**, 93 deselected, 30 min 11 s, exit 0 — the full suite is green on the branch as it stands |
| five-part fast gate (per task; last at `830cee1`) | store suite 42–43 passed; `--ignore=test_user_sbi` 261; `test_user_sbi` non-slow 1 / 16 / 12 / 64 — all green |
| `core/Reduction/tests` under pytest | 2026-09-10: 5 passed (collected with the fast suite since) |
| GPU `scripts/smoke_train.py` baseline (pre-piece-1 code, `BOUNDS=Resources/Bounds/nadrowski/master.txt`, `TOBS_S=4.5`, defaults NUM_RUNS=4 RUN_SIZE=32) | 2026-09-10: CHI=1 (cell `master_spont`): prior 102 s, posterior 747 s, validate 23 s, infer 68 s, exit 0, no OOM lines. CHI=0 (cell `master_weak`): prior 101 s, posterior 226 s, validate 12 s, infer 23 s, exit 0. |
| **GPU gate after piece 1** (`scripts/smoke_train.py` at `bb38f1a`, same BOUNDS/TOBS_S/defaults, `CHECKPOINT=1 SAVE=1 CKPT_DIR=<scratch>/smoke`, `PRISM_ARTIFACTS` pointed at scratch) | 2026-09-11: **run 1** CHI=1 `master_spont`: prior 99 s, posterior 750 s, validate 22 s, infer 59 s, exit 0 — within a few seconds of the baseline on every stage; wrote `smoke_prior`, `smoke_posterior`, `simulations/ff956b932534` (`"complete": true`, 4 batches, rows [128, 122]), an observation, a calibration and an inference. **run 2 as first recipe'd** (`PRIOR=smoke_prior NUM_RUNS=2 STAGES=prior,posterior`): exit 0 but NO resume — prior loaded in 1.4 s, then a new `simulations/294e1c3b81ec` (n_runs 2) and a recomputed Fisher, posterior 544 s (the recipe correction in item 3). **run 2 corrected** (`NUM_RUNS=4`): "Reusing the Fisher rotation stored with the training checkpoint (4/4 batches — COMPLETE, so generation will be skipped)", `[checkpoint] resuming at batch 4/4`, all stages in 3.3 s, exit 0 — the CPU→CUDA rehoming of the stored V is certified. **run 3** CHI=0 `master_weak` (own store `<scratch>/smoke_chi0`): prior 99 s, posterior 219 s, validate 12 s, infer 21 s, exit 0. No OOM lines, no OOD warnings in any leg; run 1's training masked-probe counts 121–159 of 300 (40–53 %), inside the documented ±12 pp band around 37 %. **First attempt at `19363d2`:** run 1 failed in infer (two devices in the PPC) after prior/posterior/validate passed; run 3 failed at the prior's own read-back (fingerprint) — both fixed, see item 3. Scratch stores deleted afterwards. |
| display walkthrough (`docs/checklists/display-walkthrough.md`) | never done |

## Decisions log

- **2026-09-10** — brainstorming: clean break; all four audiences; GUI + one tested CLI tool with
  the prompt CLI retired; reduction map out of scope; everything lands before the retrain;
  pytest adopted; artifact store shape B (directory per artifact + manifest); timestamp ids;
  auto-persist.
- **2026-09-10/11** — piece 1 execution rulings (full text in the ledger): R1 keep the old
  `build_posterior` return for one task; R2 three pre-flight plan fixes; R3 worktree; R4 piece 0's
  gate as baseline; R5 `.gitignore` keeps the `/Resources/*` lines until the runbook; R6
  parameter-keyed results are ordered record lists (manifest JSON sorts keys); R7 the region's
  prior fingerprint is the walk over the parent's pickled prior. The final review's "refuse
  before the spend" principle: a taken name, and a region without an observation digest, are
  refused at stage entry. The recorded fast gate is one process (a split gate hid a
  default-store leak once). The smoke-train artifact names are `smoke_prior`/`smoke_posterior`
  (a leading underscore is not a legal name).
- **2026-09-11** — after the merge: **no more feature branches or worktrees.** Every piece from
  here on is built directly on the local `main` branch in the one checkout; the user pushes.
  (The branch cost a merge, a conflict on a file both sides touched, and a stale copy of this file
  on `main` while it was open.)
- **2026-09-11** — clean break executed (section above). Ruling: archiving into the gitignored
  `archive/` means a move on disk plus `git rm --cached`, never `git mv` — the repository holds no
  tracked-but-ignored file again. `/sbc_run.log` stays in `.gitignore`. A file changed with an
  editor is NOT staged: `git commit` without `git add` commits nothing and says so only in its
  output (the runbook's step 4 needed a second attempt because of it; always check the log).
- **2026-09-11** — GPU gate rulings. (1) A prior's save/load round trip is the IDENTITY, bit for
  bit: `load_mix_dist` restores the saved weights rather than letting `Categorical` re-normalise
  them, because every fingerprint that names a prior hashes those bytes and torch's
  re-normalisation is device-dependent at the last ulp. Every other fingerprint comparison (the
  posterior's pickled training prior, the region, the run guards) is unchanged and now agrees
  across devices for free. (2) A loaded observation row lives on the CPU like every conditioning
  row; the flow's `sample()` is the one consumer that moves a copy. (3) The resume drill's two
  runs share `NUM_RUNS` (the identity includes `n_runs`). (4) GPU-only device errors are the class
  the smoke gate exists for: run it after every piece, before any record run; the gpu-marked
  suite covers the inference path cheaply but is not a substitute (no checkpoint resume, no real
  bounds/cell files).
- **2026-09-11** — **Taskbar icon** (walkthrough row 1, found by the user; fixed in `df7491e`).
  What was seen: the title bar showed the mark, the taskbar button Windows' generic "application"
  glyph — not python's icon either. Ruled out by measurement, each with a launch and a screenshot
  of the button: the PNG set (loads, seven sizes), the AppUserModelID (set and read back; removing
  it or using a fresh id changed nothing), the shell's icon cache, and the multi-size QIcon (a
  minimal window using the very same QIcon got the mark). What decided it: a minimal window idle
  right after showing got the mark; the same window blocked 4 s after showing got the glyph; PRISM's
  `show()` keeps the thread busy ~150 ms AFTER the native show (Qt lays out the whole tree —
  profiled: all inside `QWidget.show`), and the shell's icon query at button creation times out
  and falls back to the window CLASS icon, which Qt registers as the stock IDI_APPLICATION glyph
  because python.exe has no icon resource (the class handle read back equals `LoadIcon(NULL,
  IDI_APPLICATION)`). With the class icon set to ours, the blocked window got the mark too.
  Rulings: (1) the mark is installed as Qt's window-CLASS icon before the first show
  (`app_icon.set_windows_class_icon`, from the new `assets/app/prism.ico` that
  `build_app_icon.py` assembles from the PNG set — LoadImage cannot read a PNG); re-asserting
  `setWindowIcon` on the first loop turn also worked but depends on timing piece 4's start-up work
  could break; (2) the AppUserModelID stays for grouping/pinning and its docstring no longer
  claims it decides the icon; (3) a display-marked test pins the class-icon change on the real
  platform (`QT_QPA_PLATFORM=windows pytest -m display`), an offscreen test pins that prism.ico's
  256 frame IS prism-256.png; (4) verified by screenshot of the fixed launcher's button; the user
  re-checks row 1 on the next launch.
