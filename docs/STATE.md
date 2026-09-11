# PRISM — state

**Last updated:** 2026-09-11 (piece 1 executed on branch `piece-1-artifact-store`; closing gates
recorded below; the merge to `main` and the clean-break runbook are the user's next two steps)

## Where things stand

- The pre-retrain hardening programme is approved: the seven-piece decomposition
  (`docs/superpowers/specs/2026-09-10-pre-retrain-hardening-decomposition.md`) and the piece-1
  artifact-store design (`docs/superpowers/specs/2026-09-10-artifact-store-design.md`, whose §11
  lists the ten deviations ruled during execution).
- On `main`: the 2026-09-09/10 TSNPE fixes X1–X5 and the parent-prior refusal (`ed16397`), the
  specs (`aff8f7e`), piece 0 (`4b1834a`..`d05d3cd`, `3913a1b`), the piece-1 plan (`afa484f`).
- **Piece 0 is DONE** (2026-09-10): pytest, `CLAUDE.md`, this file, the launchers, the display
  checklist, the GPU smoke baseline (table below).
- **Piece 1 is EXECUTED** (2026-09-10/11) on branch `piece-1-artifact-store` in the worktree
  `.worktrees/piece-1-artifact-store/` (forked from `main` at `8ec18aa`): the 14 tasks of
  `docs/superpowers/plans/2026-09-10-artifact-store.md`, one commit each with a per-task review
  and fix loop, then a whole-branch review ("ready with fixes"), one fix wave (`830cee1`) and one
  residual fix (`3f1a1b3`). The ledger with rulings R1–R7 and every parked finding is
  `.worktrees/piece-1-artifact-store/.superpowers/sdd/2026-09-10-artifact-store/progress.md`
  (git-ignored scratch; keep it until the branch is merged). What landed: `core/artifacts/`
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

## Owed, in order

1. **Merge piece 1** (user; the worktree cannot move `main`): from the main checkout,
   `git merge --no-ff piece-1-artifact-store`, then `git worktree remove .worktrees/piece-1-artifact-store`
   and `git branch -d piece-1-artifact-store`. The branch's own gates are in the table below.
2. **Clean-break runbook** (user-executed; destructive): design spec §9 / plan Task 14 — untrack
   the generated files and `sbc_run.log`, delete `Resources/{Priors,Posteriors,Checkpoints,
   Observations,Plots,CrossValidation,ReductionMap}`, archive `scripts/tsnpe_round1_forensics.py`
   and `scripts/migrate_checkpoint_flags.py`, drop the seven `/Resources/*` lines from
   `.gitignore`, record it here.
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
   `_common.load_posterior`, threads the store into the TSNPE runner; **piece 3** copy-on-run
   session config (a refused stage must leave nothing on the session), `tsnpe_tab.restore_settings`;
   **piece 4** annotate (`set_note` has no GUI caller), cleanup of incomplete directories,
   `Summary.complete` for simulations means "has a manifest", the observation width guard in
   `load_observation` has no isolated test.
6. **The retrain.**

## Artifacts on disk (`Resources/`, gitignored) — ALL to be deleted by the clean break

- Priors: `3d_master_08102026.pt`, `prior_08272026.pt`, `prior_08282026.pt`,
  `prior_08282026_1.pt`, `shm.pt`.
- Posteriors: `posterior_09022026` (+ `.rot.pt`; the TSNPE parent, 50-wide),
  `posterior_08232026` (42-wide), `08192026_posterior_RETIRED_band_0p1_to_10`.
- Checkpoints: 8 directories incl. `QUARANTINED_tsnpe_round1_truncated_rows_train_0b471d560271`
  (5.2 GiB), `train_230ae7cb5fc2` (Run A's cache), `train_3780fd37a16a` (posterior_09022026's).
- Observations: two `obs_*.pt` plus smoke-train litter.
- None of it is readable by the piece-1 code: the sidecars, the `train_*` caches and the bare
  `.pt` priors/posteriors have no manifest, and the store refuses a directory without one. The
  `.rot.pt` sidecars hold V transposed (the load-time repair that reconciled them is deleted).
- Nothing above must be kept. `Artifacts/` starts empty after the merge.

## Last gate

| gate | result |
|---|---|
| `pytest --collect-only -q` | 2026-09-11 at `3f1a1b3`: see the fast-suite line (Reduction's 5 are collected with it) |
| fast suite, ONE process, `pytest -m "not slow" -q` | 2026-09-11 at `3f1a1b3`: running when this file was written — result appended below when it exits |
| slow test `pytest tests/test_user_sbi.py -m slow -q` | 2026-09-11 at `e4e60eb` (Task 13): **1 passed**, 93 deselected, 31 min 16 s, exit 0; re-run at `3f1a1b3` after the fix wave: running when this file was written — result appended below |
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
