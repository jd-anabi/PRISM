# PRISM artifact store, run contract and provenance — design (piece 1)

**Date:** 2026-09-10 · **Status:** approved · **Parent:** `2026-09-10-pre-retrain-hardening-decomposition.md`

## 1. Purpose

Every generated artifact (prior, simulation cache, posterior, observation, calibration, inference)
becomes a self-describing directory with a manifest that records where it came from, what
produced it, and what it is valid for. Every stage reads its parents by reference, writes its
result through one store, and refuses on any verifiable mismatch. A user of any audience can
answer "what is this, where did it come from, can I use it for what I am about to do" from the
artifact alone.

Existing artifacts are deleted (clean break), so nothing here is backward compatible with the
`.pt` + `.rot.pt` sidecar layout. Inputs (`Resources/Bounds`, `Cells`, `Units`, `Models`) are
unchanged.

## 2. Layout, ids, names

Inputs stay in `Resources/` (tracked). Generated artifacts live in `Artifacts/` (gitignored). Root:
`PRISM_ARTIFACTS` env var, else `<repo>/Artifacts` resolved from `__file__` — never the working
directory (today `config._ROOT` is cwd-first, `core/config.py:178`). Inputs resolve the same way
(`PRISM_RESOURCES` override).

```
Artifacts/
  priors/<name>__<id>/         manifest.json  log.txt  prior.pt  figures/corner.png
  simulations/<digest12>/      manifest.json  header.pt  state.pt  state.prev.pt  shards/
  posteriors/<name>__<id>/     manifest.json  log.txt  posterior.pt  loss.npz  figures/loss.png
  observations/<name>__<id>/   manifest.json  log.txt  observation.pt  figures/trace.png
  calibrations/<name>__<id>/   manifest.json  log.txt  results.json  ranks.npz  figures/*.png
  inferences/<name>__<id>/     manifest.json  log.txt  results.json  samples.pt  figures/*.png
  diagnostics/<name>__<id>/    manifest.json  log.txt  <diagnostic>.npz  figures/*.png
```

The seventh kind, `diagnostics/`, arrived with piece 2 (D4/D5): `sbc`, `identifiability` and
`ablation` each write one directory, whose payload is that diagnostic's own `.npz`
(`sbc_repeats.npz`, `laplace_sd.npz`, `degeneracy_map.npz`) and whose numbers live in the manifest.
`core/artifacts/store.py`'s `KIND_DIRS` is the list of record.

`log.txt` is the run's records, added to every kind but the simulation cache by piece 3 (V4); the
cache has no writer and so no file. `Plots/` is removed; figures live with the run that made them.
FDT, cross-validation and reduction outputs move to plain directories `Artifacts/fdt`,
`Artifacts/crossval`, `Artifacts/reduction` in this piece (today they hard-code
`Path("Resources/...")`); piece 5 wraps FDT/CrossVal in the store.

- **id:** UTC `YYYYMMDDTHHMMSS`, unique within a kind; a same-second collision appends `-2`, `-3`.
  Stored in the manifest. The directory name `<name>__<id>` sorts by creation and is convenience
  only: the store indexes by reading manifests, so a renamed folder still resolves and parent
  references (`{kind: id}`) survive.
- **simulations** are keyed by the identity digest (deduplication is their purpose) and carry a
  manifest all the same.
- **names:** user-typed, `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`, unique within a kind; a save with an
  existing name is refused. An unnamed artifact has `name=""` and lives at `<kind>/_unnamed__<id>/`.
- **fingerprints** (GMM means+weights, observation digest, V digest, bijection probe) are manifest
  fields used for verification, never for addressing.

## 3. The manifest

One `manifest.json` per artifact, written last and atomically. A directory without one is
incomplete: listed as such, never loadable, offered for cleanup.

**Common header (every kind):**

```
schema: 1
kind: prior | simulation | posterior | observation | calibration | inference
id, name, created (ISO 8601 UTC), note
prism:        {git_rev, git_dirty, branch}          "unknown"/null outside a repo; never omitted
env:          {python, torch, sbi, PySide6, numpy, os, hostname, device_type, device_name, dtype}
inputs:       {bounds: {path, sha256}, cell: {path, sha256}, units: {path, sha256}, model}
config:       every identity field + every knob that reached the stage, resolved to the value used
parents:      {prior: id, simulation: digest, posterior: id, parent_posterior: id, observation: id}
fingerprints: {gmm, x_obs, V, probe}   whichever apply
payloads:     {"posterior.pt": sha256, ...}
figures:      ["figures/corner.png", ...]
body:         kind-specific (below)
```

`config` has one exception since piece 3 (V7): a posterior's `fisher_m`, `fisher_dz` and
`fisher_points` are recorded null unless the Fisher rotation ran in that process
(`orchestrator.py`'s `_fisher_ran`). A resumed run reuses the `V` stored with the checkpoint, so it
records "not run" rather than the setting it would have used.

**Kind-specific body:**

| kind | body |
|---|---|
| prior | `gmm {n_components, param_keys, box {nd_lows, nd_highs, log_mask}}`; `sweep {num_iterations, sweep_batch, max_sets, walk_step, stability_units, min_cluster_size, min_samples}`; `stability {accepted_sets, iterations}` |
| simulation | `identity` (the full `SimulationIdentity`), `digest`, `batches_done`, `complete`, `rows [n, width]`, `V_digest`, `wall_seconds` (span from `created`, across resumes) |
| posterior | `mode`; `conditioning {input_dim, forcing_dim, width, chi_layout, chi_k_pad, chi_elem_w, chi_n_freqs, chi_f0, chi_freq_bounds, chi_max_cycles, summary_flags, feature_set_version}`; `transform {log_params, nd_lows, nd_highs, rescale_lows, rescale_highs, V, V_orientation: "columns", fisher_eigenvalues, V_digest}`; `amortized`; `truncation` (region dict incl. `prior_fingerprint`, `x_obs_digest`, `probe`, `excluded`, `t_scale_idx`; or null); `training {n_runs, run_size, resumed_from_batch, hidden_features, num_transforms, learning_rate, stop_after_epochs, epochs_trained, best_validation_loss, fisher_spread, tsnpe_acceptance, tsnpe_containment {inside, total}, wall_seconds}` |
| observation | `mode`, `conditioning` (same block), `x_obs_digest`, `T_obs_cell`, `n_obs`, `forcing_vals`, `chi_obs_freqs`, `source`: `{kind: "simulated", cell, ground_truth, T_obs_s}` or `{kind: "experimental", recordings: [{path, sha256, role, freq_Hz}], T_obs_s, F0_si, forcing_params_si}` |
| calibration | `results {sbc {per_param {label: {ks_p, c2st_ranks, c2st_dap}}}, tarp {atc, ks_p}, informativeness {total_nats, sem_nats, per_param, per_direction, n_used, n_dropped, description}, kept_fraction {acceptance, containment} or null, n_cal, cal_n_scales, num_posterior_samples}` |
| inference | `results {ppc {mean_abs_z, max_abs_z, coverage_90, num_outside, num_invalid, invalid_breakdown, note}, posterior_summary {name: {median, q05, q95}} (physical units), ground_truth or null, n_samples, accepted: [str]}` |

Small tensors (V, boxes, the 7×P probe, eigenvalues) are float64 lists in the JSON (Python's
`repr` round-trips float64 exactly; `allow_nan=False`, non-finite values refused). Large payloads
(the pickled `DirectPosterior`, GMM covariances, samples, ranks) stay binary beside the manifest
with their sha256 in `payloads`. `validate` refuses a wrong schema, an unknown kind, a bad id or
name, missing header or body keys, unknown keys, and non-finite floats. The `.rot.pt` sidecar is
retired: `transform`, `conditioning`, `amortized` and `truncation` are the manifest.

## 4. The store

Package `core/artifacts/`: `manifest.py` (schema, dataclasses, validation, JSON encoding,
`tensor_digest`), `store.py` (`ArtifactStore`, `ArtifactWriter`, loaders, wrappers),
`identity.py` (`SimulationIdentity`), `provenance.py` (git, env, file hashes, device name).
`core.artifacts` imports only `core.config` and `core.Helpers.file_manager` at module level;
`core.SBI.*` lazily, so `training_checkpoint` can import it without a cycle.

**`ArtifactStore(root)`:**
- `list(kind) -> [Summary]` (name, id, created, note, path, complete, mode, width, amortized,
  parents; complete first, newest first; incomplete flagged and never loadable)
- `get(kind, ref) -> Manifest` (ref is an id or a name); `path(kind, id)`
- `create(kind, cfg, *, name="", note="") -> ArtifactWriter` (refuses a duplicate non-empty name)
- `rename(kind, id, new_name)`, `set_note(kind, id, note)`
- `delete(kind, id, *, force=False)` — refuses, naming them, when any manifest lists the id as a
  parent or any simulation was generated against that prior's fingerprint
- `unnamed(kind, *, older_than=None)`, `find_prior_by_fingerprint(fp)`, `simulation_for(identity)`
- `load_prior(cfg, ref) -> LoadedPrior`; `load_posterior(cfg, ref, *, accept=Accept()) ->
  LoadedPosterior`; `load_observation(cfg, ref) -> LoadedObservation`; `load_calibration`,
  `load_inference`
- `default_store()` / `set_default_store(store)` / `use_store(store)` (context manager). Every
  stage function takes `store=None` (None → default). The GUI sets the default at startup; tests
  sandbox it.

**`ArtifactWriter`** (context manager): `payload(filename) -> Path` (hashed at commit),
`figure_path(title)`, `fig_sink(forward=None)` (saves the PNG, then forwards to the GUI sink or
closes the figure). `__enter__` creates the directory (`exist_ok=False`). `__exit__` on any
`BaseException` (a cancel included) removes the directory and re-raises; otherwise it hashes the
payloads, builds and validates the manifest, writes the run's `log.txt` (piece 3, V4 — every kind
but the simulation cache, which has no writer), and writes the manifest last via
`file_manager._atomic_write`.
The simulation cache is the exception: its manifest is written by `training_checkpoint` at
create, refreshed on every save and at completion, because a resumable cache must survive an
interrupted run.

**Loaded wrappers** carry `kind, id, name, manifest` plus the live object: `LoadedPrior(prior,
force_prior, nd_prior, fingerprint)`, `LoadedPosterior(posterior: TransformedPosterior, latent,
fingerprint, diagnostics, accepted)`, `LoadedObservation(x_obs, obs_data, t_dim, digest, mode,
width)` with `install(cfg)` (sets `T_obs`, `n_obs`, `chi_obs_freqs` and the observation context),
`LoadedCalibration(results)`, `LoadedInference(results, samples_path)`. Stage functions take
wrappers for anything that is a parent, and record the parent ids themselves.

**`Accept`** (`truncated=False`, `other_observation=False`) is the only escape hatch besides
`PRISM_CHI_OVERRIDE`; any flag used is written into the downstream manifest's `results.accepted`.

## 5. The run contract

| stage | reads | writes | refuses when |
|---|---|---|---|
| `build_prior` | bounds/cell/units files | `prior` (+ `figures/corner.png`) | chi config not deliberate (unless `PRISM_CHI_OVERRIDE`) |
| `build_posterior` (train) | `LoadedPrior`; simulation cache by identity | `simulation` (incremental), `posterior` | prior fingerprint ≠ cache's; TSNPE: prior ≠ parent's training prior; region basis/probe ≠ parent's; resumed cache not recording this region under this V |
| `build_posterior` (load) | `posterior` | — | model / param order / box / mode / width / layout / chi settings ≠ cfg; manifest V ≠ the rotation pickled inside the posterior's own prior (self-consistency; the transposed-sidecar repair is retired); non-amortized without `accept.truncated` |
| `validate_calibration` | `LoadedPosterior`, `LoadedPrior` | `calibration` (results, `ranks.npz`, three figures) | prior fingerprint ≠ posterior's training prior |
| `generate_observations` / `build_experiment_observation` | cell file, or recordings (`RecordingSet`) | `observation` (+ `figures/trace.png`) | width/layout ≠ cfg; a recording file missing (checked before any compute) |
| `infer_and_visualize` | `LoadedPosterior`, `LoadedObservation` | `inference` (results, samples, figures) | observation width/layout ≠ posterior's; non-amortized posterior on an observation whose digest ≠ the region's, without `accept.other_observation` |
| TSNPE round (`build_truncation_region` + train) | parent `LoadedPosterior`, `LoadedObservation`, `LoadedPrior` | `posterior` (`amortized: false`; parents prior, simulation, parent_posterior, observation) | as train; observation digest ≠ what the region was drawn around |

**Policy.** Refuse on any verifiable mismatch. Every artifact has a manifest, so "unverifiable"
no longer exists inside the store; a directory with no manifest or an older schema is refused as
"not a current PRISM artifact".

**Loader order.** `load_prior`: model → ND param set/order → box → log mask → payload → fingerprint
asserted equal to the manifest's. `load_posterior`: chi config deliberate → model, param order,
box, chi layout/k_pad/elem_w/band/cycles from the manifest → mode and width from the trained net
→ payload → bijection rebuilt from the manifest → rotation consistency against the pickled prior →
amortization. `load_observation`: mode, width, chi layout/k_pad, param_keys.

**Persistence.** Every stage writes its artifact itself, at completion. There is no in-memory-only
prior or posterior; the guard against training from an unsaved prior is retired. The GUI's Save
buttons become "name and annotate" (`rename` + `set_note`). Calibration and inference, which write
nothing today, write their artifacts like any other stage. The session holds the wrappers.

**Session.** `SbiSession` keeps `inf_prior` and `posterior` (now wrappers) and adds
`observation`; `force_prior`, `diagnostics`, `posterior_latent`, `V`, `truncation` and
`x_obs_digest` are dropped (all derivable from the wrappers). `orchestrator.training_identity`
stays as a delegate for the Posterior tab's checkpoint line and near-miss check.

## 6. Simulation identity

`SimulationIdentity` (frozen dataclass) replaces the `training_identity` dict: `format:
"training-rows/2"`; every field it has today (model, prior_fingerprint, mode, param_keys, the four
box lists, log_params, reparam_rotate, run_size, n_runs, steady_idx, dt_nd_min, dt_exp, t_min_exp,
t_max_exp, t_scale_bounds, n_grid, spontaneous_only, summary_flags, chi_mode / layout / k_pad /
elem_w / f0 / freq_bounds / max_cycles, device, dtype); `truncation` always present (null or the
region's identity fields); new `feature_set_version` (an int constant in `core/SBI/statistics.py`,
bumped whenever a summary feature's definition changes, so a cache never resumes onto rows that
mean something else). Digest: sha256 of canonical JSON, 12 hex. The "frozen docstring" rule is
lifted; the identity is versioned by `format`. `verify`, `near_miss_siblings`,
`describe_siblings` and `checkpoints_using_prior` keep their semantics over
`Artifacts/simulations/`; the directory is `<digest12>` with no `train_` prefix.

## 7. Paths

`config.py` keeps `RESOURCES_ROOT` with `CELL_PATH`, `BOUNDS_PATH`, `UNITS_PATH`, `MODELS_PATH`,
and gains `artifacts_root()`. `PRIOR_PATH`, `POSTERIOR_PATH`, `PLOT_PATH`, `CHECKPOINT_PATH` and
`OBSERVATION_PATH` are removed. `SimConfig` gains `sources` (the bounds, cell and units paths the
config was built from) so manifests can hash inputs. A source-scan test asserts that no module
under `core/` or `scripts/` builds a literal `Resources` or `Artifacts` path outside `config.py`.

The GUI's generated-kind pickers become a `StorePicker(kind)` listing `store.list(kind)`: label =
name (or the id for unnamed), tooltip = id · created · mode · width · amortized. The artifact
browser proper is piece 4.

## 8. Tests

New `tests/test_artifact_store.py` (pytest; `tmp_path` and a `store` fixture): round-trip every
kind; manifest validation refusals; writer removes the directory on exception; a truncated
manifest is incomplete and never loadable; same-second ids; rename keeps id and lineage; delete
refuses naming dependents; one test per load-refusal class; the `Accept` flags land in the
downstream manifest; provenance inside and outside a git repo; identity re-keys on
`feature_set_version`; a stage-contract test that runs every stage at tiny size (including the
forced experimental path) and asserts exactly the declared artifact, parents and results; the
source scan. The five existing suites that touch sidecars, checkpoints or the observation picker
are updated to the store; `tests/conftest.py` sandboxes the default store for the whole session.

## 9. Clean-break runbook (user-executed, destructive)

1. `git rm --cached $(git ls-files -i -c --exclude-standard)` (the 20 tracked-but-ignored files)
   and `git rm --cached sbc_run.log`; commit.
2. Delete `Resources/{Priors,Posteriors,Checkpoints,Observations,Plots,CrossValidation,ReductionMap}`,
   including `QUARANTINED_tsnpe_round1_truncated_rows_train_0b471d560271`. `archive/` untouched.
3. Move `scripts/tsnpe_round1_forensics.py` and `scripts/migrate_checkpoint_flags.py` to
   `archive/scripts/` (their inputs no longer exist).
4. `.gitignore`: the seven `/Resources/*` lines become `/Artifacts/`.
5. Record it in `docs/STATE.md`.

## 10. Out of scope for this piece

GUI widgets beyond the picker source (piece 4); field-level validation such as `T_obs > 0` and
path fields (piece 3), except that the observation stage now checks recordings exist; the script
consolidation (piece 2) — interim, `scripts/_common.load_posterior` is re-pointed at the store and
the scripts that import a removed path constant stay broken until piece 2; FDT/CrossVal outputs
as store kinds (piece 5); `logging` (piece 3). The prompt CLI's `run()` loses interactive figure
display (figures are on disk); it is retired in piece 2.

## 11. Amendments made during execution (2026-09-10/11, piece 1)

Each is a deliberate deviation from the text above, ruled during the task-by-task execution and
confirmed by the final whole-branch review; the ledger of the run holds the reasoning in full.

1. **Results keyed by parameter are ordered LISTS of records, not dicts** (ruling R6): the
   calibration body's `sbc.per_param` is `[{"name", "ks_p", "c2st_ranks", "c2st_dap"}, ...]` and
   the inference body's `posterior_summary` is `[{"name", "q05", "median", "q95"}, ...]`, in the
   bounds file's parameter order. The manifest is written with sorted keys, so a dict keyed by name
   cannot carry the order, and in this code the order is the binding. The loaders read the manifest
   body; `results.json` is the human-readable copy and is never read back.
2. **The region's `prior_fingerprint` is the fingerprint of the prior pickled inside the parent
   posterior** (ruling R7), the one copy no writer can get wrong, and the parent wrapper's own
   fingerprint is cross-checked against it (a disagreement is refused).
3. **Manifest bodies are validated dicts** (key sets per kind in `manifest.BODY_KEYS`), not per-kind
   dataclasses. Same contract, less code.
4. **The source scan covers `core/` only** for now (§8 says `core/` and `scripts/`): nine scripts
   legitimately import the removed path constants until piece 2 folds them into the command-line
   tool, and each carries a banner saying so. Piece 2 extends the scan to `scripts/`.
5. **`LoadedPosterior.accepted` is informational.** Only the flags the inference stage itself is
   given reach a body (`results.accepted` of the inference); the calibration body carries no
   `accepted` field (§3 and §5 disagreed; this is the resolution — a calibration on a
   non-amortized posterior is recognisable from its parent's `amortized: false`).
6. **Save = rename.** `set_note` exists and is tested; the GUI's "annotate" field and the cleanup
   action for incomplete directories ("offered for cleanup", §3) are piece 4's browser. The store
   lists incomplete directories with a `reason` so the browser has what it needs.
7. **`.gitignore` keeps the `/Resources/*` lines until the clean-break runbook** (ruling R5), so
   the still-existing trees do not flood `git status`; the runbook's step 4 removes them.
8. **The simulation kind's id is its identity digest** (12 hex) and the id pattern admits both
   forms; `Summary.complete` for a simulation means "has a valid manifest" — `body["complete"]`
   says whether the cache is finished.
9. **`env_info` records `"absent"`** for an optional package that is not installed.
10. **The TSNPE runner resolves the process default store** (`runners._run_tsnpe_round` takes no
    `store` argument); piece 2's stage API threads it like the other stages.
