# PRISM — state

**Last updated:** 2026-09-10 (piece 0 of the hardening programme in progress)

## Where things stand

- The pre-retrain hardening programme is approved: the seven-piece decomposition
  (`docs/superpowers/specs/2026-09-10-pre-retrain-hardening-decomposition.md`) and the piece-1
  artifact-store design (`docs/superpowers/specs/2026-09-10-artifact-store-design.md`).
- On `main`: the 2026-09-09/10 TSNPE fixes X1–X5 and the parent-prior refusal (`ed16397`), the
  specs (`aff8f7e`), then piece 0's commits.
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
| `pytest --collect-only -q` | pending (piece 0) |
| fast suite `pytest -m "not slow"` | pending (piece 0) |
| full suite `pytest` | 2026-09-09 23:49: 321/321 under the old per-file runners (322 after `6a62e13`) |
| `core/Reduction/tests` under pytest | pending (piece 0) |
| GPU `scripts/smoke_train.py` (CHI=1, CHI=0, `BOUNDS=…/master.txt`) | pending — baseline running this session |
| display walkthrough (`docs/checklists/display-walkthrough.md`) | never done |

## Decisions log

- **2026-09-10** — brainstorming: clean break; all four audiences; GUI + one tested CLI tool with
  the prompt CLI retired; reduction map out of scope; everything lands before the retrain;
  pytest adopted; artifact store shape B (directory per artifact + manifest); timestamp ids;
  auto-persist.
