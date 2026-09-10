# PRISM — state

**Last updated:** 2026-09-10 (piece 0 done; piece 1 in progress on branch `piece-1-artifact-store`)

## Where things stand

- The pre-retrain hardening programme is approved: the seven-piece decomposition
  (`docs/superpowers/specs/2026-09-10-pre-retrain-hardening-decomposition.md`) and the piece-1
  artifact-store design (`docs/superpowers/specs/2026-09-10-artifact-store-design.md`).
- On `main`: the 2026-09-09/10 TSNPE fixes X1–X5 and the parent-prior refusal (`ed16397`), the
  specs (`aff8f7e`), piece 0 (`4b1834a`..`d05d3cd`, `3913a1b`), the piece-1 plan (`afa484f`).
- **Piece 0 is DONE** (2026-09-10): pytest adopted (322 tests in `tests/` + 5 in
  `core/Reduction/tests`, which run for the first time), `CLAUDE.md`, this file, the launchers set
  `KMP_DUPLICATE_LIB_OK`, the display checklist, the GPU smoke baseline (table below). One test was
  order-dependent under pytest and is now seeded (`d05d3cd`).
- **Piece 1 is IN PROGRESS** on branch `piece-1-artifact-store` in the worktree
  `.worktrees/piece-1-artifact-store/` (forked from `main` at `8ec18aa`), executed
  task-by-task from `docs/superpowers/plans/2026-09-10-artifact-store.md` with a ledger at
  `.worktrees/piece-1-artifact-store/.superpowers/sdd/2026-09-10-artifact-store/progress.md`.
  Merge to `main` when its final review is clean.
- **Decision 2026-09-10: CLEAN BREAK.** Every generated artifact is deleted once piece 1 lands
  (runbook: design spec §9). The old runbook's Run A, Run B and the TSNPE round
  (`PRISM_HANDOFF.md` §11.9) are ABANDONED; the retrain restarts from scratch after piece 6.

## Owed, in order

1. **Piece 0 — verification baseline** (this session): pytest, `CLAUDE.md`, this file, the
   launchers, the Reduction tests under pytest, the display checklist, the GPU smoke baseline.
2. **Piece 1 — artifact store**: plan via `superpowers:writing-plans` into
   `docs/superpowers/plans/`, then TDD, one commit per step, fast suite green at every commit.
3. **Clean-break runbook** (user-executed; destructive).
4. **Pieces 2 → 3 → (4 ∥ 5) → 6**, each brainstormed → spec → plan → implementation.
5. **The retrain.**

## Artifacts on disk (`Resources/`, gitignored) — ALL to be deleted by the clean break

- Priors: `3d_master_08102026.pt`, `prior_08272026.pt`, `prior_08282026.pt`,
  `prior_08282026_1.pt`, `shm.pt`.
- Posteriors: `posterior_09022026` (+ `.rot.pt`; the TSNPE parent, 50-wide),
  `posterior_08232026` (42-wide), `08192026_posterior_RETIRED_band_0p1_to_10`.
- Checkpoints: 8 directories incl. `QUARANTINED_tsnpe_round1_truncated_rows_train_0b471d560271`
  (5.2 GiB), `train_230ae7cb5fc2` (Run A's cache), `train_3780fd37a16a` (posterior_09022026's).
- Observations: two `obs_*.pt` plus smoke-train litter.
- The three `.rot.pt` sidecars hold V transposed (reconciled at load; moot after the clean break).
- Nothing above must be kept.

## Last gate

| gate | result |
|---|---|
| `pytest --collect-only -q` | 2026-09-10: 327 (322 in `tests/`, 5 in `core/Reduction/tests`) |
| fast suite `pytest -m "not slow"` | 2026-09-10 15:03 on `d05d3cd`: **326 passed, 1 deselected**, 10 min 28 s |
| full suite `pytest` | not yet run under pytest; the slow test is unchanged since the 2026-09-09 23:49 gate (321/321 under the old runners) — piece 1's Task 13 runs it |
| `core/Reduction/tests` under pytest | 2026-09-10: 5 passed |
| GPU `scripts/smoke_train.py` baseline (pre-piece-1 code, `BOUNDS=Resources/Bounds/nadrowski/master.txt`, `TOBS_S=4.5`, defaults NUM_RUNS=4 RUN_SIZE=32) | 2026-09-10: CHI=1 (cell `master_spont`): prior 102 s, posterior 747 s, validate 23 s, infer 68 s, exit 0, no OOM lines. CHI=0 (cell `master_weak`): prior 101 s, posterior 226 s, validate 12 s, infer 23 s, exit 0. Re-run after pieces 1, 2, 3 and compare. |
| display walkthrough (`docs/checklists/display-walkthrough.md`) | never done |

## Decisions log

- **2026-09-10** — brainstorming: clean break; all four audiences; GUI + one tested CLI tool with
  the prompt CLI retired; reduction map out of scope; everything lands before the retrain;
  pytest adopted; artifact store shape B (directory per artifact + manifest); timestamp ids;
  auto-persist.
