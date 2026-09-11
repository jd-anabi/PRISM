# PRISM — state

**Last updated:** 2026-09-11 (piece 1 MERGED as `461d780`; the CLEAN-BREAK RUNBOOK is EXECUTED —
`8dbd93e`, `e37df41`, `19363d2` and the docs commit that carries this file; `Artifacts/` starts
empty; next: the post-piece-1 GPU gate and the GUI checks, then piece 2; all work happens
directly on the local `main` branch)

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
3. **GPU gate after piece 1** (plan §Verification): run 1 `CHI=1 TOBS_S=4.5
   BOUNDS=Resources/Bounds/nadrowski/master.txt CELL=Resources/Cells/nadrowski/master_spont.txt
   CHECKPOINT=1 SAVE=1 CKPT_DIR=<scratch>/smoke python scripts/smoke_train.py`; run 2 the same
   with `PRIOR=smoke_prior NUM_RUNS=2 STAGES=prior,posterior` and **`SAVE` unset** (a second
   `SAVE=1` is refused by name). Expect "Reusing the Fisher rotation stored with the training
   checkpoint" and `<scratch>/smoke/simulations/<digest>/manifest.json` with `"complete": true`;
   then the `CHI=0 … master_weak` leg. Compare stage times with the baseline row below.
4. **Manual GUI check on a display** (plan §Verification, nine checks; `run.bat`): `Artifacts/`
   created at launch; prior from scratch → `priors/_unnamed__<id>/`, Save renames it; posterior
   at 2 batches → `posteriors/` with parents prior + simulation; load it (no dialog) and a
   posterior under another bounds file (refusal naming the field); Validate → `calibrations/`;
   Infer → `observations/` then `inferences/`; TSNPE round → `amortized: false`, four parents,
   loading logs NON-AMORTIZED, Infer on another cell refused naming `Accept(other_observation=True)`;
   cancel during training → no `posteriors/` directory, `simulations/<digest>/manifest.json`
   `"complete": false`; a hand-truncated `manifest.json` is omitted by the picker. Record here.
5. **Pieces 2 → 3 → (4 ∥ 5) → 6**, each brainstormed → spec → plan → implementation. Carried
   into them from piece 1 (spec §11 and the ledger): **piece 2** extends the source scan to
   `scripts/`, retires `_common.require_mode` and the unconditional `Accept(truncated=True)` in
   `_common.load_posterior`, threads the store into the TSNPE runner, and fixes the stale
   comments the clean break left (listed below); **piece 3** copy-on-run session config (a
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
| slow test `pytest tests/test_user_sbi.py -m slow -q` | 2026-09-11 at `e4e60eb` (Task 13): 1 passed, 31 min 16 s, exit 0; re-run at `3f1a1b3` after the fix wave: **1 passed**, 93 deselected, 30 min 11 s, exit 0 — the full suite is green on the branch as it stands |
| five-part fast gate (per task; last at `830cee1`) | store suite 42–43 passed; `--ignore=test_user_sbi` 261; `test_user_sbi` non-slow 1 / 16 / 12 / 64 — all green |
| `core/Reduction/tests` under pytest | 2026-09-10: 5 passed (collected with the fast suite since) |
| GPU `scripts/smoke_train.py` baseline (pre-piece-1 code, `BOUNDS=Resources/Bounds/nadrowski/master.txt`, `TOBS_S=4.5`, defaults NUM_RUNS=4 RUN_SIZE=32) | 2026-09-10: CHI=1 (cell `master_spont`): prior 102 s, posterior 747 s, validate 23 s, infer 68 s, exit 0, no OOM lines. CHI=0 (cell `master_weak`): prior 101 s, posterior 226 s, validate 12 s, infer 23 s, exit 0. **Post-piece-1 run owed (item 3 above).** |
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
