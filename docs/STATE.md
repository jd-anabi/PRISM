# PRISM — state

**Last updated:** 2026-09-16. **Piece 2, "one flow underneath", is DONE**: commits
`0016dae`..`d34997c` (55) on the local `main` branch, pushed by the user on 2026-09-15. The prompt
CLI is retired. The GUI and the new `python -m core <subcommand>` tool are two front ends over the
same orchestrator stages and three compositions. Six scripts became four subcommands (`smoke` and
three diagnostics, the diagnostics writing a new `diagnostic` store kind), six were archived, and
`scripts/` is gone. A whole-branch review and its fixes closed the piece. The GPU gate of record ran
at `0d96d2f`; the final one-process fast gate (457 passed at `d34997c`) and the slow set are in the
table below, and the user ran rows B1–B8 of `docs/checklists/display-walkthrough.md` on the real
screen on 2026-09-15: all eight pass. **Piece 3** (validation and misuse-proofing, logging with
severity, the private copy of the session config) is brainstormed, specified and planned: the design
`docs/superpowers/specs/2026-09-15-validation-and-logging-design.md` (approved; `ca583f8`, decisions
V1–V9) and the 24-task plan `docs/superpowers/plans/2026-09-16-validation-and-logging.md` (`a8bcec2`).
**Next: execute the plan** task by task on the local `main` branch; no task has started.

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
- **Piece 2 is DONE** (2026-09-15, commits `0016dae`..`d34997c`, on the local `main` branch, pushed 2026-09-15). It
  follows the design `docs/superpowers/specs/2026-09-11-one-flow-design.md` (§1.1 holds decisions
  D1–D13; §11 lists the 58 deviations ruled during execution) and the 25 tasks of
  `docs/superpowers/plans/2026-09-12-one-flow.md`. Each task had a review and fix loop and a
  one-process fast gate. A whole-branch review then found six must-fix defects and 21 cheap fixes;
  they landed in `2010250`..`d34997c` with a scoped re-review. The execution ledger is gitignored
  scratch under `.superpowers/sdd/2026-09-12-one-flow/`. Its rulings live on in the decisions log
  below and in spec §11; its open items that a later piece or the owner must take are in "Owed"
  item 7. What landed: the prompt CLI is gone (D1); `python -m core <subcommand>` (`core/tool/`)
  passes every flag to its stage as a keyword argument (D2, D6); `simulated_inference`,
  `experimental_inference` and `tsnpe_round` hold the checks each front end used to carry alone
  (D3); the diagnostics write a `diagnostic` store kind (D4, D5); training refuses a "near miss",
  meaning a saved simulation cache that differs from the new run in exactly one setting, before
  the Fisher step (the costly rotation computed before training simulates) and before any
  simulation (D7); the tool refuses by default and `--accept-*` maps onto `Accept` (D8); every
  driven chi recording states its drive frequency in Hz (D9).
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
   The recipe is four command lines since T19 (`CLAUDE.md`, spec §3.8): `python -m core smoke --chi
   --t-obs 4.5 --bounds …/master.txt --cell …/master_spont.txt --checkpoint --save --store-root
   <scratch>/smoke` (run 1); run 2 is that line WITHOUT `--save`, with `--prior smoke_prior --stages
   prior,posterior --resume require` added (must resume); run 2b is run 2's line WITHOUT `--resume
   require`, with `--num-runs 2` added instead (must EXIT 1 naming `n_runs` — the incident, now
   loud); then the forced-mode run against its own store. Read literally, keeping `--save` on run 2
   would refuse on the taken name `smoke_posterior` instead of resuming — fixed in the fix-round-1
   pass over this recipe (K4), kept in sync with `CLAUDE.md`. The environment-variable recipe is
   gone with `scripts/smoke_train.py`.
4. ~~Manual GUI check on a display~~ — done 2026-09-11 by the user on the real screen: rows A1–A9
   of `docs/checklists/display-walkthrough.md` all pass (reported "everything passes"). The one
   failure was the walkthrough's row 1, the APP ICON: the window's title bar showed the mark, the
   taskbar button the generic Windows "application" glyph. Root-caused the same day by screenshot
   experiments (the "Taskbar icon" entry in the decisions log) and fixed by installing the mark as
   Qt's Win32 window-CLASS icon from a new `assets/app/prism.ico`
   (`app_icon.set_windows_class_icon`); verified by a screenshot of the real launcher's button
   and then by the user on the next launch (row 1 PASS). **The user also ran rows 2–20 of the
   piece-4 walkthrough table the same day: all pass** — piece 4's display walkthrough is therefore
   already done once, on the code as of `df7491e`; piece 4 re-checks only what it changes.
5. ~~**Piece 2**~~ — DONE 2026-09-15, commits `0016dae`..`d34997c`, on the local `main` branch, pushed
   2026-09-15. Every item piece 1 carried into it is closed: ~~extend the source scan to
   `scripts/`~~ (`scripts/` is gone; the scans walk `CODE_ROOTS` plus `CODE_FILES` in
   `tests/_fixtures.py`, and `test_the_source_scans_cover_every_code_directory` keeps the set
   closed — T21, T22 and the final review); ~~retire `_common.require_mode` and the unconditional
   `Accept(truncated=True)`~~ (`_common.py` is gone; the tool refuses by default, and
   `--accept-truncated` / `--accept-other-observation` map 1:1 onto `Accept` — T12, T21);
   ~~thread the store into the TSNPE runner~~ (`tsnpe_round` takes a `LoadedObservation` and
   `store=` — T5, T7, T12); ~~the stale comments the clean break left~~ (T23, `7c5ff48`); ~~make
   the resume drill loud~~ (a cache one setting away is refused before the Fisher step and before
   any simulation, with `--resume require` and `--new-run`; GPU run 2b exited 1 naming `n_runs` —
   T2, T19).
6. ~~**The display walkthrough's piece-2 rows B1–B8**~~ — done 2026-09-15 by the USER on the real
   screen: **all eight pass**, recorded in that file's last two columns and in the gate table below.
   B8 re-checked the FDT panel's full run, because `bb22ac4` changed the plots it draws.
7. **Pieces 3 → (4 ∥ 5) → 6**, each brainstormed → spec → plan → implementation. Carried into them
   from piece 2 (design spec §1.3, the final review's "left open" list, and the ledger):
   - **Piece 3.** Spec `ca583f8` and plan `a8bcec2` cover every bullet below (spec §1.3 lists what it
     leaves to pieces 4 and 5). Execution has not started. The plan's drafting was not independently
     verified task by task (the verifier run was cut off by a usage limit); its cross-task names were
     checked by hand, so each task's own review at execution is the first check of its quoted code.
     - Copy-on-run session config: a refused stage must leave nothing on the session. It also
       covers an amortized training in a session whose config carries a ground truth, which then
       anchors the Fisher step on that truth. And it covers a GUI session in which an experimental
       chi inference leaves `cfg.chi_n_freqs` at the recording's probe count, so a later simulated
       chi inference in the same session simulates that count (`build_experiment_observation`
       sets it since `2010250`, which makes an experimental chi observation record its own drive
       frequencies). Installing any stored chi observation already did this before piece 2.
     - Logging with severity, in place of the in-stage `print`s piece 2 left.
     - Boundary validation of every field, and error dialogs that name the fix. One known case:
       the TSNPE refusal "5 directions requested but the latent has 4" names neither the default
       nor the `--directions` flag or its GUI control.
     - `tsnpe_tab.restore_settings` is never called.
     - The Config tab's chi drive and band fields are editable and restored from QSettings, while
       every non-default value is now refused (D11; spec §2.9).
     - The budget status line derives the simulation identity separately (`base.py`).
     - The TSNPE manifest records Fisher defaults that did not run.
     - `build_prior`'s load branch calls `plt.show()` when `fig_sink` is None.
     - sbi writes `<cwd>/sbi-logs`.
   - **Piece 4.** The artifact browser, including a listing of diagnostics; annotate (`set_note`
     has no GUI caller); cleanup of incomplete directories; `Summary.complete` for simulations means
     only "has a manifest"; the observation width guard in `load_observation` has no isolated test.
   - **Piece 5.** FDT/CrossVal hardening and wrapping them in the store (their outputs still go to
     `artifacts_root()/fdt` and `/crossval`). Two test gaps: the cell-folder branch of the `fdt`
     unsupported-model hint is untested (`core/tool/fdt.py`), and only `plot_psd` of the four
     `core/FDT/plots.py` functions is unit-tested for closing its figure.
   - **Piece 6.** The `docs/` split of the handoff, including the `PRISM_HANDOFF.md` lines D1 makes
     false (`:49,53-54,76,190-191`), and a README reference section for the tool.
   - **Open, with no piece owning them yet** (the owner decides each, or where it goes):
     - *Science.* In `identifiability jacobian`, a NaN in a measurable Jacobian column makes the
       least-squares step raise `LinAlgError` after all the simulations are spent (the retired
       script crashed the same way). The choice is between excluding that column and zeroing that
       row.
     - *Science.* Repeat-SBC in chi mode cannot see probe-design variance: `pipeline.py` reseeds the
       chi probe generator to a fixed seed (20260805) on every `gen_training_data` call, so every
       repeat uses the same probe design whatever `--seed` says.
     - *Provenance.* A calibration and a TSNPE child do not record `--accept-truncated` in
       `results.accepted` (the spec requires that record only for inferences and diagnostics), and
       an inference on a non-amortized posterior's own anchor observation records `accepted: []`
       even when `--accept-truncated` loaded it.
     - *Spec.* `tsnpe_round` refuses more directions than the latent width but allows exactly as
       many, which truncates every direction. That matches spec §2.5 word for word, so tightening
       it needs a spec decision.
     - *Dormant.* `core.diagnostics.rng.seeded()` restores only the CUDA device it is given, while
       `torch.manual_seed` reseeds every CUDA device; on a machine with several GPUs the others'
       streams are not restored. Revisit if a gpu-marked test starts to depend on test order.
     - *Tests.* The code-directory guard skips an enumerated list of directories, so a local
       virtual environment at the repository root would fail it for a non-code reason.
   - The rest of the final review's 46 "left open" items are test-coverage gaps and cosmetics, each
     with its reason, in the gitignored `.superpowers/sdd/2026-09-12-one-flow/final-review.md`.
8. **The retrain.**

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
- ~~**Left alone, for piece 2** (stale comments, none a code path)~~ — ALL FIXED by piece 2:
  T23 (`7c5ff48`) made `inference_screen.py` and `test_nav_and_gating.py` name the store's
  `observations/` listing; `pipeline.py` and `training_checkpoint.py` say "a read-only artifact
  root (`Artifacts/`, or `PRISM_ARTIFACTS`)"; `statistics.py` dates its substitution table to the
  deleted pre-piece-1 checkpoint; `test_user_sbi.py` cites `migrate_checkpoint_flags.py` at
  `e37df41^`; `run.sh` no longer claims the working directory decides the roots.
  `smoke_train.py` and `generate_bundle_videos.py` went with their files (`187a096` removed the
  first, `7433ced` archived the second). The original list, for the record: `core/gui/screens/
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
| **GPU gate of record, piece 2** (`python -m core smoke`, the four command lines of `CLAUDE.md`, at `0d96d2f`) | 2026-09-14: **run 1** chi `master_spont`: prior 92 s, posterior 198 s, validate 22 s, infer 85 s, exit 0; **run 2** `--resume require`: "Reusing the Fisher rotation stored with the training checkpoint (4/4 batches — COMPLETE, so generation will be skipped)", `[checkpoint] resuming at batch 4/4`, exit 0 in 8 s; **run 2b** `--num-runs 2`: **exit 1** in 6 s, "differs only in n_runs: this run 2, that cache 4", no `[fisher]` line, `simulations/` unchanged (still only `4d8022b100db`); **run 3** forced `master_weak` (own store): prior 92 s, posterior 61 s, validate 14 s, infer 20 s, exit 0. No OOM lines and no Traceback in any run; run 1's training masked probes 79/224, 15/96, 108/192, 58/192 = 260/704 (36.9 %), inside the ±12 pp band around 37 %. Diagnostic card runs, `PRISM_ARTIFACTS` at the same stores, all on the card: `sbc --chi --repeats 1 --n-cal 20` (19 s), `identifiability jacobian --chi` (84 s), `ablation --chi` (6 s) against run 1's store, `identifiability laplace` (128 s) against run 3's — each exit 0, one `diagnostics/` directory each. Scratch stores deleted. **Timings: prior and validate match `bb38f1a` within seconds; posterior and infer do NOT and are not comparable** — `smoke` builds a truth-free config, so the Fisher anchors on the prior median plus seven prior draws (under a numpy-seeded run, different draws) instead of `master_spont`'s truth, and the Fisher's simulation cost is set by those operating points (1341 simulation segments in the posterior stage at `50a9bb6`'s `smoke_train` run vs 608 here). For the same reason V and `fisher_spread` (now `inf`: a numerically zero smallest eigenvalue, where `smoke_train` printed 5.26e16) are not comparable. The interim check at `50a9bb6` (still `smoke_train`, truth-anchored) matched `bb38f1a` on every stage: run 1 93/724/21/59 s, run 3 94/212/12/24 s, masked 37.8 %. **Not repeated after `0d96d2f`:** the only later line that creates a tensor is the final review's fix in `build_experiment_observation` (`2010250`), which builds `cfg.chi_obs_freqs` on `cfg.hw.device` as `generate_observations` does; no `smoke` command reaches that line, because `smoke` simulates its observation; and the gpu-marked `tests/test_gpu_paths.py` ran on the RTX 5070 Ti inside the final fast gate. |
| display walkthrough (`docs/checklists/display-walkthrough.md`) | 2026-09-11, the user on the real screen at `df7491e`: rows 1–20 all pass (row 1 after the taskbar-icon fix) and rows A1–A9 all pass. Piece 4 re-checks only the rows it changes |
| `pytest --collect-only -q` after piece 2 | 2026-09-15 at `d34997c`: **460** collected (the fast gate's 457 passed and 1 skipped, plus the 2 slow tests; Reduction's 5 are collected with them). That is 99 more than the 361 at `df7491e`, where spec §8.5 had budgeted about 42 ± 8 added (spec §11 row 58). Two new suites, `tests/test_tool.py` (the tool, in-process through `main(argv)`) and `tests/test_diagnostics.py`; `tests/test_*.py` holds sixteen suites |
| **final fast gate, piece 2**, ONE process, `pytest -m "not slow" -q --durations=15` | 2026-09-15 at `d34997c`, after the whole-branch review's fixes: **457 passed, 1 skipped** (the display-marked class-icon test, offscreen), 2 deselected (the two slow tests), 174 warnings, **13 min 38 s** (inside spec §8.5's 14-minute target), exit 0. No "More than 20 figures" line. The real `Artifacts/` gained nothing, and `Resources/Models` held only `SHM.json` and `SHM2.json` afterwards. The gpu-marked `tests/test_gpu_paths.py` ran on the RTX 5070 Ti inside it. Slowest test: `test_user_sbi.py::test_train_and_validate_without_a_loaded_cell`, 180 s. Per-suite durations were not measured, so §8.5's ~90 s targets for `test_tool.py` and `test_diagnostics.py` are unchecked. Every task had its own one-process gate; those results are in the execution ledger (spec §11 row 42) |
| slow set, `pytest -m slow -q --durations=5` (piece 2) | 2026-09-15 at `d34997c`: **2 passed**, 458 deselected, 96 warnings, **30 min 53 s**, exit 0. `test_user_sbi.py::test_chi_mode_full_sbi_pipeline` 1606 s (26 min 46 s); `test_tool.py::test_fdt_and_crossval_run_at_tiny_size` 243 s. No "More than 20 figures" line (the slow chi test's sink now closes its figures). The warnings are the known library classes (sklearn KMeans, sbi's SBC-count and `__array__` warnings, nflows `triangular_solve`, the reparam batch cap) plus the expected chi probe-masking notices; `Resources/` was left clean. The set is the chi full-pipeline test in `tests/test_user_sbi.py` (it now passes the drive frequencies it simulated as `(path, Hz)` pairs, D9) and the FDT/crossval tiny-size run in `tests/test_tool.py`. The chi test feeds back exactly the frequencies it simulated, so it cannot tell a stale recorded frequency from a correct one; `tests/test_artifact_store.py::test_an_experimental_observation_records_its_own_length_and_drive_frequencies` pins that |
| display walkthrough, piece-2 rows B1–B8 (`docs/checklists/display-walkthrough.md`) | 2026-09-15, the user on the real screen, piece 2 as pushed: **rows B1–B8 all pass** — the D8 load dialog and its Cancel, the round's own install with no dialog, the other-observation box, the D7 near-miss dialog, the TSNPE stage checks (0 directions, HPD 0.95, the near-miss refusal after reloading the amortized parent), the short-T_obs warning, the D12 refusal, and the FDT panel's full run after `bb22ac4` (B8, added in `d6b1f03`). Rows 1–20 and A1–A9 stand from 2026-09-11 |

**The GPU gate, as command lines.** This is `CLAUDE.md`'s recipe of record (its Tests section),
copied verbatim; keep the two copies identical. Run it from the repository root, with `$S` set to
an empty scratch directory. `PRISM_ARTIFACTS` is not needed for the four `smoke` runs, which touch
only `--store-root`; the diagnostic card runs need it pointed at the same store.

````markdown
  ```powershell
  $py = "C:\Users\J\anaconda3\envs\biophys-env\python.exe"
  $B  = "--bounds","Resources/Bounds/nadrowski/master.txt"
  $C  = "--cell","Resources/Cells/nadrowski/master_spont.txt"
  $S  = "<scratch>"
  # run 1: chi; builds and names smoke_prior/smoke_posterior; writes the simulation cache
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --save --store-root "$S/smoke"
  # run 2: same store, same --num-runs; must resume
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --store-root "$S/smoke" --prior smoke_prior --stages prior,posterior --resume require
  # run 2b: one setting away; must exit 1 naming n_runs
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --store-root "$S/smoke" --prior smoke_prior --stages prior,posterior --num-runs 2
  # run 3: forced mode, its own store
  & $py -m core smoke --no-chi --t-obs 4.5 @B --cell Resources/Cells/nadrowski/master_weak.txt --checkpoint --save --store-root "$S/smoke_chi0"
  ```

  Pass criteria, read off `$LASTEXITCODE` and the printed lines after each command:
  - run 1 and run 3: `$LASTEXITCODE` 0, no OOM lines, stage timings near the last recorded gate in
    `docs/STATE.md`, masked-probe counts within ±12 pp of 37 %;
  - run 2: prints `Reusing the Fisher rotation stored with the training checkpoint` and
    `[checkpoint] resuming at batch 4/4`, `$LASTEXITCODE` 0;
  - run 2b: `$LASTEXITCODE` 1, the refusal names `n_runs`, no `[fisher]` line, no new directory
    under `$S/smoke/simulations/`.

  `--bounds` is required: the same-named sibling rule would otherwise resolve the 12-dim
  spontaneous box. After a change under `core/diagnostics` that moves tensors, also run the
  diagnostic card (`sbc`, `identifiability jacobian`, `ablation` against `$S/smoke` with
  `$env:PRISM_ARTIFACTS` set; `identifiability laplace` against `$S/smoke_chi0`), as recorded in
  `docs/STATE.md`'s piece-2 GPU gate row. Delete `$S` afterwards; the last result is in
  `docs/STATE.md`.
````

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
- **2026-09-11/15** — **piece 2 (one flow underneath): decisions D1–D13** (spec
  `docs/superpowers/specs/2026-09-11-one-flow-design.md` §1.1; plan
  `docs/superpowers/plans/2026-09-12-one-flow.md`). D1 the prompt CLI is retired WHOLE
  (`orchestrator.run`, `core/app.py`, the `__main__` menu, the prompt half of `core/cli.py`,
  `helpers.clear_screen`'s callers, `run_fdt`'s `input()` calls, whose two booleans are now
  required). D2/D6 the flags-only tool `python -m core <subcommand>`: every flag reaches its stage
  as a keyword argument, `--bounds` is required wherever a prior is built or training runs, and the
  tool reads only `PRISM_RESOURCES` and `PRISM_ARTIFACTS`. D3 three compositions beside the stages
  (`simulated_inference`, `experimental_inference`, `tsnpe_round`), each carrying the checks that
  used to live in one front end only. D4 six scripts became four subcommands (`smoke`, `sbc`,
  `identifiability` with three modes, `ablation`), six were archived, `_common` dissolved. D5 the
  `diagnostic` store kind, whose loader has no refusals. D7 a committed simulation cache ONE
  setting away (a "near miss") is refused before the Fisher step and before any simulation;
  `new_run=True` / `--new-run` overrides it; the resume policy `auto`/`require`/`never` is a stage
  argument and the `--resume` flag; the GUI dialog asks `fresh_run_near_misses`, and the stage
  restates its checks field for field (spec §11 row 2). D8 the tool
  refuses by default and `--accept-*` maps 1:1 onto `Accept`; the Posterior tab asks before it
  loads a stored non-amortized posterior. D9 every driven chi recording states its frequency in Hz;
  the legacy branch is deleted. D10 the entry sets `KMP_DUPLICATE_LIB_OK` and the Agg backend
  before any torch or core import, so `--help` stays torch-free. **D11 there is NO chi override
  anywhere: a non-default band or drive amplitude means editing `config.py` deliberately.** **D12 a
  TSNPE round whose parent is itself non-amortized and whose observation is not that parent's is
  refused, with no escape hatch.** D11 and D12 are STANDING REFUSALS: a later piece must not
  re-open them. D13 `PRISM_VRAM_CEILING_GIB` and `PRISM_MEM_LOG_EVERY` stay core-level environment
  settings, named in the tool's `--help` epilog and in `CLAUDE.md`, not arguments;
  `PRISM_VRAM_CEILING_GIB` is read live on each batch plan, `PRISM_MEM_LOG_EVERY` once, when
  `core.SBI.pipeline` is imported. Spec §1.2's readings stand: `runners.py` is deleted rather than
  wrapped; the one-setting rule still exempts a cache that differs only in its truncation region (a
  TSNPE round at its parent's budget); only `smoke` and the diagnostics take `--seed`; a
  hand-entered truth drops the cell from the recorded sources.
- **2026-09-14/15** — piece 2's working rules. **Folding versus archiving:** a FOLDED script is
  `git rm`'d (git history is its archive, and an archived copy would drift from its replacement);
  an ARCHIVED script is moved on disk into the gitignored `archive/scripts/` and then `git rm
  --cached`, never `git mv` — the same ruling as the clean break's. **Test-side knob defaults:**
  `tests/conftest.py`'s session-scoped autouse `_checkpointing_off_unless_asked` is the only
  session-wide knob default the suites install, and every test that wants a simulation cache
  passes `checkpoint_every` explicitly. **Scan roots:** `CODE_ROOTS` (the top-level directories,
  today `core`) plus `CODE_FILES` (top-level Python files, today `conftest.py`) in
  `tests/_fixtures.py` are what the source scans walk (the literal-path, `input()` and rotation-reader scans); a guard fails if a
  directory holding Python is missing from `CODE_ROOTS`, and a missing `CODE_FILES` entry fails
  with a message naming it. **The Fisher is truth-free on the command line:** `smoke` builds a
  config with no ground truth, so its rotation anchors at the prior median plus seven prior draws
  rather than at the cell's truth as `smoke_train` did. So only the prior and validate timings (and
  the masked-probe share) compare with `bb38f1a`; the posterior and infer timings, `V` and
  `fisher_spread` do not.
- **2026-09-14/15** — piece 2's execution rulings, condensed to the load-bearing ones. The ledger
  (`.superpowers/sdd/2026-09-12-one-flow/progress.md`, compiled in `rulings-and-open-items.md`) is
  gitignored scratch; the durable record is this log plus spec §11.
  - Each task's gate result goes into the ledger, not this file; this table gets only gates of
    record and the final rows (R-C).
  - A plan's predicted counts, error texts and file lists are predictions; what must hold is the
    behaviour: the test fails before the change and passes after (R-A).
  - A leftover-mention search accepts history notes ("Folded from scripts/x.py"); a hit that
    imports, builds a path to, or tells someone to run a retired thing fails it (R-B).
  - Piece 2's commit range starts after `b578f08`; a task whose check changes nothing needs no
    commit (R-H).
  - One GPU recipe text: `CLAUDE.md`'s block, copied verbatim under the gate table (R-I, R-V).
  - Refuse before the spend: `infer` checks its recording flags before loading anything, and
    `infer --cell` with any recording flag exits 2 (R-L); the near-miss refusal sits before the
    truncation refusals and the Fisher step (T2); bad knobs and zero sample counts are refused
    before any simulation (T16–T18, final review).
  - The near-miss refusal message names both GUI controls and `--new-run` (R-O).
  - The diagnostics declare and build their accept flags through the shared `config_args` helpers
    (R-T).
  - Warnings in test output are findings: a leaked warning is captured and asserted; a third-party
    warning from a real run the spec requires is accepted unfiltered and the baseline moves (T4, T6,
    T11, T12).
  - Only inferences and diagnostics record accept flags in `results.accepted`; whether a
    calibration or TSNPE child should is parked for the owner (R-S; "Owed" item 7).
  - `identifiability laplace` isolates only its fixed-seed calls, so its noise-floor draws stay
    independent per point and `--seed` governs every point (T17).
  - The GPU gate of record ran alone on the card at `0d96d2f`, after T19's fix round, and its row
    was committed separately (R-U). It passed although posterior and infer missed "within a few
    seconds of `bb38f1a`": the truth-free Fisher explains the difference, and prior and validate
    match (T19).
  - Walkthrough row B8 was added because `bb22ac4` changed the FDT plots after row 15 last passed
    (T24).
  - End order: whole-branch review, one fix dispatch and a scoped re-review, then the slow set and
    the final fast gate, then these documents from the measured numbers (R-W); the documents
    follow the ledger where the T25 brief was stale (R-X).
  - The final review's six must-fix and 21 fix-now items landed in one dispatch, each behaviour fix
    test-first; 46 items stay open with reasons. Follow-ups: the SBC rank histogram's bin count is
    the largest divisor of `nps + 1` between half the cap and the cap, else the cap; the flow-knob
    range checks run only on a call that trains; the chi probe count left on a GUI session is
    piece 3's.
