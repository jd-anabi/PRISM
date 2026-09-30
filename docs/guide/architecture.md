# Architecture

Checked against commit 3c4e393.

This page is for whoever maintains the code: how `core/` is laid out, how a run flows through it,
and why the load-bearing parts are built the way they are. Code is cited by module and function name.
The facts a user needs live on their own pages and are linked from here, not repeated:
[Getting started](getting-started.md) for the records, their kinds, names and ids and where each run
saves; [The command-line tool](command-line.md) for every flag, output, refusal and exit code;
[The window](window.md) for every control; [The science behind the settings](science.md) for why each
setting has its value.

## Module map

Every module under `core/`, one line each, grouped by package. A package's `__init__.py` is left out
when it holds nothing but a docstring, and so is the reduction map's own test folder,
`core/Reduction/tests`.

### Entry points

- `core/__main__.py`: `python -m core`. Sets `KMP_DUPLICATE_LIB_OK` and matplotlib's Agg backend
  before anything imports torch, then exits with `core.tool.main`'s return code.
- `core/gui/__main__.py`: `python -m core.gui`. Forces the Agg backend before any `core` import, then
  builds and shows the window.

### The top of `core`

- `config.py`: the constants every default comes from, device detection (`detect_device`,
  `cpu_device`), the inputs root and the records root, the memory-budget reading
  (`memory_budget_elements`) and the allocator policy; it re-exports `SimConfig` and `FDTConfig`.
- `sim_config.py`: the `SimConfig` and `FDTConfig` data carriers: the box, the observation mode, the
  simulator's rescale index, the tier-1 helpers and `copy_for_run`.
- `cli.py`: the configuration builders both front ends share (`make_sim_config`, `make_fdt_config`,
  `make_param_sweep_config`, `make_reduction_config`), cell parsing (`parse_cell`), the bounds file a
  cell resolves to (`resolve_bounds_for_cell`) and the sweep presets (`SWEEP_PRESETS`).
- `orchestrator.py`: the stages and the compositions that chain them
  ([Stages and compositions](#stages-and-compositions)).
- `runs.py`: the run boundary: `public_entry`, the per-run buffer that becomes `log.txt`, and the
  deferred-cancel sections; sets the `core` logger's level once, at import.
- `refusals.py`: `Refusal`, `PreflightWarning`, the field registry (`FIELDS`) and the rules that
  refuse a bad input before anything is spent; it imports only the standard library.
- `rng.py`: the one seeding context (`seeded`), the seed rule (`require_seed`) and a calibration's
  derived seed (`calibration_seed`).
- `logging_root.py`: the one root-logger handler both front ends install, so a library's record is
  shown once and a `core` record is never shown twice.
- `registry.py`: the model registry: the three built-in models, the user models loaded from the
  inputs root, and which screens accept a user model (`is_sbi_user_model`, `fdt_support`).
- `forcing.py`: the drive builders (`build_nondim_force_tensor`, `build_nondim_sin_force_tensor`,
  `build_user_force_tensor`), the zero drive, the drive-channel rule (`n_force_channels`) and the
  check that raises `RuntimeError` on an inferred rescale index (`require_simulator_index`).
- `progress.py`: `SOLVER`, the step counter the solver publishes and the window's solver meter reads.

### `core/SBI`

- `pipeline.py`: the façade: memory planning, the out-of-memory ladders, `gen_obs`,
  `gen_training_data` and the per-batch probe tally.
- `train.py`: the sbi training call (`train_nn`) and its typed plan (`TrainingPlan`).
- `summaries.py`: `gen_stats`, per-column winsorisation and the count of pathological trajectories.
- `statistics.py`: the hand-made summary features, their valid flags, `FEATURE_SET_VERSION` and
  `conditioning_rows`, the one constructor of a conditioning row.
- `chi.py`: the χ(ω) arithmetic: probe placement, the peak-frequency estimator, the lock-in and the
  probe-set packer (`pack_probe_block`); no simulation.
- `chi_probes.py`: the probe loop (`gen_chi_raw`, `gen_chi_block`) and `ProbeRecord`.
- `chi_encoder.py`: the permutation-invariant encoder of the probe set.
- `embedded_network.py`: `EmbeddedNet`, the flow's embedding network, which in chi mode standardises
  the whole conditioning row itself.
- `reparam.py`: the bijection between the latent coordinates and the box, the rotated latent prior,
  and `TransformedPosterior`.
- `decorrelate.py`: the simulated Fisher matrix and its eigenbasis, the rotation the flow trains in.
- `derived.py`: the force scale derived from temperature on a tier-1 box, and `for_simulation`.
- `truncate.py`: TSNPE: the truncation region and the truncated latent prior.
- `training_checkpoint.py`: the simulation cache on disk: its header, state and shards, the resume
  check (`verify`), the set-aside of uncommitted shards (`set_aside_uncommitted`) and the search for
  caches one setting away.
- `run_guards.py`: the checks that stop an expensive run before the spend, and the prior's
  fingerprint.
- `prior_screen.py`: `gen_prior`, the entry to the stability screen.
- `observations.py`: conditioning observations built from bench recordings, and `RecordingSet`.
- `ppc.py`: the posterior predictive check's bin-by-bin simulation loop.
- `overlay.py`: the figures that set posterior-predictive traces against the observation.
- `analysis.py`: the calibration data generator (`gen_cal_data`), the predictive-check statistic and
  the informativeness measure.
- `Priors/prior.py`: `Prior`, the stability-screened prior: a global census of the box, a random-walk
  flood-fill of the stable region, HDBSCAN, then a Gaussian mixture (`construct_prior`).
- `Priors/nadrowski_prior.py`: the Nadrowski model's census and flood-fill.
- `Priors/hopf_prior.py`: the Hopf model's census and flood-fill.
- `Priors/bp_prior.py`: the BP model's census and flood-fill.
- `Priors/user_prior.py`: the same screen for a user model.
- `Priors/log_uniform.py`: a log-uniform distribution that sbi's `MultipleIndependent` can move
  between devices.
- `Priors/sbi_prior_wrapper.py`: gives sbi a mean and a standard deviation for a prior it can only
  sample.
- `Priors/Scaling Priors/scaling_prior.py`: the rescale block's prior, uniform or log-uniform per
  parameter.
- `Priors/Forcing Priors/forcing_prior.py`: the drive parameters' prior.
- `Priors/Forcing Priors/sin_prior.py`: a four-parameter sine-drive prior; nothing in `core` imports
  it.
- `Priors/Product Prior/product_prior.py`: `ProductPrior`, the ND prior and the rescale prior as one
  distribution.

### `core/Simulator`, `core/Solvers` and `core/Models`

- `Simulator/simulator.py`: `Simulator`, which integrates a batch segment by segment and picks
  `euler` or `euler_compiled`, and `SimulationError`.
- `Simulator/nadrowski_simulator.py`: binds the Nadrowski parameter columns, by position, to
  `NadrowskiModel`.
- `Simulator/hopf_simulator.py`: the same for `HopfModel`.
- `Simulator/bp_simulator.py`: the same for the BP model, choosing the full or the steady-state form
  by the number of parameters.
- `Simulator/user_simulator.py`: the same for a user model.
- `Solvers/sdeint.py`: the Itô Euler–Maruyama solver: a plain Python loop (`euler`), and a
  TorchScript step (`euler_compiled`) replayed from CUDA graphs where it can be.
- `Models/nadrowski_model.py`: the Nadrowski model's drift and noise, and its TorchScript step
  (`compiled_step`).
- `Models/hopf_model.py`: the same for the Hopf normal form.
- `Models/bp_model.py`: the BP model in full; it has no TorchScript step, so BP runs the plain Python
  loop on every device.
- `Models/bp_model_steady.py`: the BP model with its fourth time constant at zero.
- `Models/user_model.py`: a user model: its equations parsed with sympy into a torch model that meets
  the solver's contract.

### `core/artifacts`

- `__init__.py`: the package's public names, re-exported from the modules below.
- `store.py`: `ArtifactStore`, `ArtifactWriter`, the loaders, `Accept`, `KIND_DIRS` and the default
  store (`default_store`, `use_store`, `resolve_store`).
- `manifest.py`: the manifest's schema: the header keys, each kind's body keys, `validate`, and the
  config and conditioning blocks.
- `identity.py`: `SimulationIdentity`, the simulation cache's identity, and its `FORMAT`.
- `provenance.py`: who made a record and from what: git, package versions, input files and their
  hashes.
- `report.py`: the two text renderers, a record's manifest (`render_manifest`) and its lineage
  (`render_lineage`).

### `core/diagnostics`

- `__init__.py`: re-exports the five diagnostics and the three probe checks.
- `sbc.py`: SBC repeated on one posterior: the spread of each parameter's KS p-value, optionally one
  probe count at a time (`sbc_repeats`).
- `identifiability.py`: what a posterior kept (`identifiability_rotation`) and whether the data
  carry the information at all (`identifiability_laplace`, `identifiability_jacobian`).
- `ablation.py`: which conditioning channels the flow can see (`channel_ablation`).
- `feature_sets.py`: the features a configuration's diagnostics are built over, and the guards that
  refuse a configuration they do not cover.
- `probes.py`: the probe checks (`probe_band`, `probe_mask`, `probe_drive`): they measure the chi
  probe settings on a lab's cell and change none of them.
- `probe_math.py`: the pure helpers the probe checks share: recording geometry, default grids and
  spectrum measures.
- `rng.py`: the old address of `core.rng.seeded`, kept as a re-export of the same object.

### `core/tool`

- `__init__.py`: `build_parser` and `main`: parse, choose the store root, run one handler, turn the
  outcome into an exit code.
- `stages.py`: `prior`, `train`, `validate`, `infer` and `tsnpe`, one thin handler per stage.
- `diagnostics.py`: `sbc`, `identifiability` and `ablation`.
- `probes.py`: the `probes` family.
- `smoke.py`: `smoke`, every stage end to end at tiny sizes.
- `fdt.py`: `fdt`, `crossval` and `compare`.
- `browse.py`: the `artifacts` family: read, annotate and tidy the store.
- `config_args.py`: the shared flag groups, the configuration builder (`build_cfg`), `knobs`, the
  figure sink and the success line.
- `fields.py`: which flag answers each refusal field (`FLAG`).
- `help_defaults.py`: the one table of how each flag's help states its default.
- `logging_console.py`: the two console handlers: information to stdout, warnings and errors to
  stderr.

### `core/gui`

- `app.py`: builds the `QApplication` and `MainWindow`, installs the root-logger handler, and a
  last-resort exception hook that shows an unhandled error in a dialog.
- `main_window.py`: `MainWindow`: the navigation shell over Home, the four section screens, the
  Artifacts screen, Settings and the model builder; saves every panel's state on close.
- `screens/nav_shell.py`: the navigation shell: the PRISM title, the back arrow and the screen stack.
- `screens/home_screen.py`: Home, one button per section (`SECTIONS`).
- `screens/section_screen.py`: a section: a heading over a tab widget of its panels.
- `screens/inference_screen.py`: Parameter Inference: six tabs over one shared `SbiSession`, and the
  tab gates (`refresh_gates`).
- `screens/artifact_screen.py`: the Artifacts screen, over `core.artifacts`.
- `screens/settings_screen.py`: Settings, and the help reference assembled from each section.
- `screens/model_builder_screen.py`: the user-model builder.
- `panels/base_panel.py`: `BasePanel`: a controls column, a results area (figures, progress, log),
  `dispatch()` onto a worker thread, the one-run-at-a-time rule and the refusal box.
- `panels/inference_tabs.py`: the import surface of the six inference tabs; Settings reads it by
  string path.
- `panels/inference/base.py`: the tabs' shared base classes and the training-budget lines.
- `panels/inference/config_tab.py`: the Config tab: the model and its options.
- `panels/inference/prior_tab.py`: the Prior tab, which builds the configuration and dispatches
  `build_prior`.
- `panels/inference/posterior_tab.py`: the Posterior tab, which dispatches `build_posterior` and asks
  the near-miss question first.
- `panels/inference/validate_tab.py`: the Validate tab, which dispatches `validate_calibration`.
- `panels/inference/infer_tab.py`: the Infer tab, which dispatches `simulated_inference` or
  `experimental_inference`.
- `panels/inference/tsnpe_tab.py`: the TSNPE tab, which dispatches `tsnpe_round`.
- `panels/inference/help_text.py`: the tabs' help text, keyed by field.
- `panels/inference/rows.py`: the chi band and probe rows.
- `panels/fdt_panel.py`: the FDT analysis tab.
- `panels/crossval_panel.py`: the sweep-study tab.
- `panels/record_view.py`: reads an `fdt` record back onto its panel.
- `panels/reduction_panel.py`: the reduction-map tab.
- `panels/simulate_panel.py`: the live simulation's tab.
- `panels/simulate_runner.py`: the live simulation's worker side, which streams frames to the tab.
- `panels/simulate_export.py`: the live simulation's video export, on a worker.
- `session.py`: `SbiSession` and `ConfigDraft`, what the six inference tabs share.
- `worker.py`: `Worker`, which runs a callable on the global thread pool with its output routed to Qt
  signals.
- `streams.py`: the routing of a run's output to the panes, `CancelToken` and `WorkerCancelled`.
- `vt.py`: turns what tqdm writes into progress rows and log lines.
- `plot_watcher.py`: shows each PNG a run saves into its record's `figures/` folder.
- `fields.py`: which control answers each refusal field.
- `settings.py`: the one `QSettings` store and its typed helpers.
- `design.py`: the design tokens and the palette and stylesheet builders.
- `theming.py`: the appearance modes (follow the system, light, dark, by time of day).
- `fonts.py`: the bundled Inter font.
- `icons.py`: the bundled icon font.
- `mpl_theme.py`: matplotlib's settings, following the theme.
- `app_icon.py`: the application icon, loaded from its PNG set.
- `assets/app/build_app_icon.py`: renders the icon's editable SVG source to the PNGs that ship.
- `assets/icons/build_prism_icons.py`: generates the five-glyph icon font.
- `widgets/artifact_picker.py`: the input-file picker (`ArtifactPicker`) and its store twin
  (`StorePicker`).
- `widgets/artifact_table.py`: the Artifacts screen's table and its one row formatter.
- `widgets/progress_pane.py`: one overall bar, one caption and the solver rate.
- `widgets/log_pane.py`: the read-only log.
- `widgets/figure_stack.py`: a run's figures, as tabs.
- `widgets/figure_window.py`: a figure popped out into its own window.
- `widgets/refusal_box.py`: the one "Check your inputs" box.
- `widgets/param_grid.py`: grids for typing bounds and cell values.
- `widgets/labeled_inputs.py`: the typed numeric fields and file pickers.
- `widgets/source_toggle.py`: a switch between a file and typed values.
- `widgets/help_badge.py`: the "?" badge beside a setting.
- `widgets/live_hair_bundle.py`: the live simulation's trace and heat map.
- `widgets/compare_list.py`: the list the comparison controls hold.
- `widgets/field_row.py`: labelled fields in one row.
- `widgets/forms.py`: the one place a form layout is built.
- `widgets/adaptive_stack.py`: a stacked widget sized to the visible page, not the tallest.
- `widgets/anim.py`: screen and tab transitions.

### `core/FDT`

- `fdt_pipeline.py`: `run_fdt`, the effective-temperature measurement for one cell.
- `campaigns.py`: the two simulation campaigns: spontaneous fluctuations to a power spectrum, and
  driven responses to χ(ω).
- `spectral.py`: `psd_welch`, `lock_in_chi`, `eff_temp_ratio`, the frequency grid and the peak finder.
- `sanity.py`: the five sanity checks (`run_all_sanity`).
- `cross_validation.py`: the parameter-sweep study (`run_param_study_cli`, `run_fdt_param_sweep`).
- `cross_validation_plots.py`: the sweep study's 3D figures.
- `compare.py`: `compare`, which compares saved `fdt` records.
- `plots.py`: the single-cell figures.

### `core/Helpers`

- `file_manager.py`: parsing the bounds, cell, units and model files; atomic saves; listing an input
  folder; the prior's mixture on disk.
- `helpers.py`: `rescale`, and `get_even_ids`, the segment boundaries.
- `labels.py`: the symbols and unit annotations of plots and the window.
- `model_store.py`: saving and loading user models.
- `visualizers.py`: shared plotting: `save_figure`, the predictive-check, overlay and loss figures,
  and `emit_figure`.
- `fdt.py`: two old torch helpers (`gen_freqs`, `force`) above a disabled block of numpy spectrum
  and lock-in helpers; nothing in `core` imports it.
- `model_helpers.py`: rescaling helpers; nothing in `core` imports it.

### `core/Reduction`

- `__init__.py`: the public names (`reduce_nwk_to_hopf`, `sweep_f_max`, `run_reduction_map`).
- `reduce.py`: one Nadrowski point reduced to its Hopf normal form (`ReductionRecord`).
- `fixed_point.py`: the fixed point.
- `linear.py`: the linear part.
- `cubic.py`: the cubic Lyapunov coefficient.
- `projection.py`: the drive and the noise projected onto the normal form.
- `rescaling.py`: from the Nadrowski model's units to the Hopf form's, and the dimensional factors.
- `sweep.py`: the sweep, and the Reduction panel's entry (`run_reduction_map`).
- `plots.py`: the reduction map's figures.

### Two notes

- **`core.SBI.pipeline` is a façade.** Its extracted siblings (`train`, `prior_screen`,
  `chi_probes`, `summaries`, and the sine drive builder from `core.forcing`) are re-imported at the
  end of the module, because tests rebind names on it: a test that rebinds `pipeline.<name>` changes
  what code reads through the module at call time, and nothing else. So every consumer must read
  these names as `pipeline.<name>`, and the siblings that call back into the façade (`train`,
  `chi_probes` and `summaries`) do so through the module object (`_pipeline.<name>`), never through a
  `from` import. A consumer that bypasses the seam fails silently: the patch stops landing, and tests
  go slow or quiet, not red. Deleting a re-import fails when the name is next read:
  - a `NameError` from the façade's own code, which uses several as module globals (`gen_stats`,
    `gen_chi_block`, `count_pathological`, and `_PATHO_MAG` only in its `[patho]` lines, the last of
    them after generation ends);
  - an `AttributeError` for a consumer that reads `pipeline.<name>`;
  - an `ImportError` for one that imports it.

  Two re-imports, `VALID_PRIORS` and `_ZSCORE_CHECK_MAX_ROWS`, are read by nothing through the
  façade. The same re-export pattern holds in `core/config.py` (its re-export of `SimConfig`), in
  `inference_tabs.py`, at the end of `core/orchestrator.py` (the bench-recording builders, which the
  suites patch there) and in `core/diagnostics/rng.py`. Deleting one of those fails loudly too, with
  one exception: Settings reads `core/gui/panels/inference_tabs.py`'s `HELP` by string path and skips
  it when it is absent.
- **Reduction is not used by inference.** Only the Reduction Map panel imports `core.Reduction`.

## Stages and compositions

**The flow.** A bounds file and a units file become one configuration, and four stages in
`core.orchestrator` run on it in order:

```
bounds + units -> cli.make_sim_config (a bounds-only configuration) -> build_prior
               -> build_posterior -> validate_calibration -> infer_and_visualize
```

A cell's truth enters only where it is needed, through `cli.load_and_validate_gt`: inside
`simulated_inference` (which takes hand-entered values through `SimConfig.inject_ground_truth`
instead), in the tool's `build_cfg(load_gt=True)` (the Laplace and Jacobian identifiability checks
and the probe checks' `band` and `drive`), and in the live simulation's runner. A narrowing round
puts back its observation's recorded truth (`LoadedObservation.install`), which it uses only to
report whether the truth lies inside the region. The training rows never depend on the cell.

- `build_prior` builds the stability-screened prior over the ND box (`pipeline.gen_prior`) and
  writes a prior record.
- `build_posterior` trains a neural posterior in latent coordinates and writes a posterior record.
  With the rotation on, a fresh run first computes the Fisher rotation (`decorrelate`). A resume, or
  a refit of a complete cache, reuses the rotation stored in its simulation cache's header, and a
  narrowing round reuses the parent's rotation carried in its region, so neither runs the Fisher.
  Then `train.train_nn`, handed a `TrainingPlan`, runs `pipeline.gen_training_data` and trains the
  flow on its rows. Each batch takes one (t_scale, T_obs) pair from a Sobol schedule, shared by every
  row of the batch, simulates at a fine ND step and downsamples to the experiment's sampling; in chi
  mode each batch draws its probe count, placement and lock-in lengths, and
  `chi_probes.gen_chi_block` builds the probe block. `statistics.conditioning_rows` assembles every
  row.
- `validate_calibration` runs SBC and the joint coverage test (TARP) on data simulated from the
  posterior's own prior (`analysis.gen_cal_data`, which calls `gen_training_data` with no cache) and
  writes a calibration record.
- `infer_and_visualize` samples the posterior for one observation, draws the corner plot, runs the
  posterior predictive check (`ppc`) and the overlay figures, and writes an inference record.

**The observation builders** each write an observation record, which an inference and a narrowing
round name as a parent. `generate_observations` simulates a cell's ground truth at a fine step and
downsamples it as training does a row; `build_experiment_observation` builds one from bench
recordings (`RecordingSet`, through the builders in `core.SBI.observations`).

**The compositions** chain stages for the front ends and refuse every bad input before the first of
them spends anything.

- `simulated_inference`: the sample count, the observation length, the name and the cell are checked
  first, then `generate_observations` and `infer_and_visualize`; it returns both records.
- `experimental_inference`: the same for recordings, through `build_experiment_observation`.
- `build_truncation_region`: the narrowing round's region, drawn from a posterior around a recorded
  observation.
- `tsnpe_round`: `build_truncation_region`, then `build_posterior(truncation=region)` on the same
  prior; the proposal is the truncated prior, never the posterior.
- Two readers serve the window: `training_preview`, behind the Posterior and TSNPE tabs' budget
  lines, which reports what `build_posterior` would do at a budget (the width, the peak memory
  against the planner's budget, whether it resumes) and never raises; and `fresh_run_near_misses`,
  through which the Posterior tab asks the near-miss question of
  [The artifact store](#the-artifact-store) before Train is pressed. Neither is a public entry: they
  copy nothing and write nothing.

A composition looks its stages up in the module's globals at call time, so a patched stage takes
effect inside it.

**`core.runs.public_entry`.** Every public stage, composition and diagnostic, and the three FDT
entries, carries this decorator, and nothing else does
(`test_the_public_entries_carry_public_entry_and_nothing_else_does` in
`tests/test_artifact_store.py` pins the set).

- It hands the function a private copy of its configuration, and no stage mutates what it is handed
  ([Rules the code keeps](rules-and-traps.md#rules-the-code-keeps) gives the rule). The window builds
  one configuration and runs every stage from it; before the copy, one run's values (a recording's
  probe count, a cell's truth) leaked into the next.
- It runs the call inside `capture_run()`, which buffers every `core` record and Python warning from
  the moment the outermost entry began; a composition and the stages it calls share one buffer.
  `ArtifactWriter` writes that buffer as the record's `log.txt`.

**Every knob travels as an argument, never as a configuration edit.** `core.orchestrator` imports
its defaults by name (`from .config import TRAINING_NUM_RUNS, ...`), which binds each one in its
own namespace when the module is first imported. Assigning `core.config.TRAINING_NUM_RUNS` later
changes nothing the stage reads, and the run uses the default without a word. So the front ends pass
every setting to the stage, some by keyword and some by position (the tool's `prior` handler calls
`orchestrator.build_prior(cfg, None, True, ...)`). The window assigns two module constants, and both
are read through the module at the moment of use: `config.QUIET_SEGMENT_BAR`, which silences the
per-segment progress bar, and `config.SIM_VRAM_CEILING_GIB`, the memory ceiling of
[Memory planning](#memory-planning-and-out-of-memory-recovery). Neither is a science setting.

**Refusals.** A bad input is refused before anything is spent, as a `core.refusals.Refusal` that
names its setting by a field key each front end maps to its own control or flag
([Rules the code keeps](rules-and-traps.md#rules-the-code-keeps) gives the rule and its known
gaps). A judgement that does not stop the run is a `PreflightWarning`.

**Messages** are `logging` records at info, warning or error, on children of the `core` logger, whose
level `core.runs` sets once. The handlers belong to the front ends: the window's
`streams._PumpLogHandler` for one run, the tool's `logging_console.console_handlers` for one handler
call, `runs._RunLogHandler` for the record's `log.txt`, and `core.logging_root`'s one root handler for
library records. A judgement or a count that must not stop the run is a Python warning (a
`PreflightWarning` among them, or the masked-probe notice), and the run log captures those too. The
only prints left under `core` are the command-line tool's own output (its banner, record and success
lines, and the `artifacts` family's listings), the two asset build scripts under `core/gui/assets`,
and the reduction map's report (`Reduction.sweep`).

**The force scale on a tier-1 box.** A bounds file that declares temperature in place of the force
scale puts T in the force scale's column, and the force scale is derived from it
([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)).
Everything that simulates must hand the simulator the derived block and the simulator's index, which
names that column `f_scale`.

- `derived.for_simulation(cfg, params_nd, rescale)` is the one call that the simulating
  diagnostics (`identifiability`), the probe checks and the live simulation's runner
  (`core.gui.panels.simulate_runner`) use.
- Training, the observation builder, the posterior predictive check and the Fisher rotation make the
  same substitution inline, with `derived.to_sim_rescale` and the simulator's index,
  `cfg.sim_rescale_idx`. `gen_training_data`, which is handed index dictionaries rather than a
  configuration, calls `derived.sim_rescale_idx`, the function behind that property.
- The drive builder (`forcing.build_nondim_force_tensor`, which `build_nondim_sin_force_tensor`
  delegates to) and the probe loop (`chi_probes.gen_chi_raw`) both raise `RuntimeError`, through
  `forcing.require_simulator_index`, on an index that names T and no `f_scale`, before they build a
  drive or simulate anything.

**The probe observer and the per-batch probe tally.**

- `gen_training_data(probe_observer=...)` takes an optional callable that receives one
  `chi_probes.ProbeRecord` per committed chi row range, in batch order: the rows' own peak
  frequencies, each probe's log frequency over that peak (`u`), the log of the cycles it was locked
  in over (`logcyc`), each probe's verdict and the packer's mask, all detached copies on the CPU.
  The probe checks' `mask` mode is its user. Attached, it adds one peak estimate per range on the
  device, so on a card short of memory the rows match an unobserved run bit for bit only when no
  halving happens.
- The tally is `pipeline._BatchProbeLedger`, installed as `pipeline._ACTIVE_LEDGER` for the life of
  one chi-mode `gen_training_data` call. It is a module global for the reason the batch tag is one:
  its producer, `gen_chi_block`, sits several calls below the batch loop.
- `gen_chi_block` reports each row range's masked and simulated probe counts with `add`, which
  drops any pending report for an overlapping range, so a batch re-run in halves is counted once.
  The batch-level retry calls `discard` before it re-runs an abandoned attempt. The batch loop calls
  `commit` only after it has stored the batch's rows, which is also when the observer is handed the
  batch's records.
- A cache commit stores the per-batch counts in `state.pt` as `"chi_masked"`
  (`training_checkpoint.save`) only when the tally covers every committed batch
  (`pipeline._chi_counts`). A resume restores them when they describe exactly the committed batches;
  otherwise, after a resume of a cache committed without counts or with a list of another length,
  the tally starts at the resume point, and that cache stores no counts from then on.
- At the end of the call the tally logs the masked-probe run total, the `[chi] masked probes` line,
  and names its scope: every committed batch of the simulation cache, only the batches this process
  generated, or this process alone when there is no cache. The calibration logs one of its own,
  scoped to this process. The line scoped to every committed batch is the one the card smoke gate
  reads ([The card smoke gate](testing.md#the-card-smoke-gate)).

## The artifact store

What a record is, the eight kinds, names and ids, and where each run saves are in
[The artifact store](getting-started.md#the-artifact-store) on the getting-started page. This section
is how the store is built, in `core/artifacts/`.

- **One directory per record, found by its manifest.** A record lives in
  `<records root>/<KIND_DIRS[kind]>/<name>__<id>/`. The store indexes by reading manifests, never by
  parsing directory names, so a hand-renamed folder still resolves and a parent is named by its id.
- **The manifest** (`manifest.json`) has a fixed set of header keys (`HEADER_KEYS`) and a body whose
  keys are a closed set for each kind (`BODY_KEYS`, in `core.artifacts.manifest`). `validate` refuses
  a key that is missing or unknown, a schema other than the current one, a non-finite number, and a
  posterior whose amortized flag disagrees with its truncation region. Small tensors are recorded as
  float64 lists; large payloads stay binary beside the manifest, with their sha256 in `payloads`. A
  stage's settings start from `config_from_cfg` and it adds its resolved knobs; a posterior's and an
  observation's conditioning geometry is `conditioning_block`.
- **Writing.** `ArtifactStore.create` refuses a taken name and returns an `ArtifactWriter`. The writer
  creates the directory on entry and hands out payload and figure paths (`payload`, `figure_path`);
  at the commit it hashes the payloads, writes `log.txt`, then the manifest, last and atomically. For
  every kind but `fdt`, any exception removes the directory. A directory with no manifest is
  therefore incomplete by definition, and it is the only record directory a sweep removes. Every
  sweep also clears the loose files inside the kind directories it covers (`remove_loose`); the
  all-kinds sweep alone also clears the legacy `crossval/` directory, which sits beside the kind
  directories at the records root (`legacy_dirs`, `remove_legacy`). Anything written in the last
  five minutes is refused as possibly live (`RECENT_WRITE_SECONDS`).
- **Saving is a rename.** `ArtifactStore.rename` rewrites the manifest's name, then renames the
  directory. A directory rename that Windows refuses is tolerated, because the manifest is what
  resolves.
- **`fdt` records are progressive** (`PROGRESSIVE_KINDS`). The writer writes a first manifest on
  entry, with every body key present, the unknown ones null, and `complete` false; `refresh()`
  rewrites the manifest and the log as the run goes; an exception keeps the directory, marked
  unfinished. The exception is a `Refusal` raised before any payload or figure path was handed out,
  which removes the directory, so a pre-spend refusal leaves no empty record behind.
- **The simulation cache** has no writer. `gen_training_data` commits it through
  `training_checkpoint.save` at its cadence (every 50 batches by default), at the end of generation,
  and on the way out of a cancel or a crash, and `store.write_simulation_manifest` keeps its
  manifest. `header.pt` is written once: the identity,
  the (t_scale, T_obs) schedule, the initial conditions, the rotation and its eigenvalues, and a probe
  of the bijection. `state.pt` holds `batches_done` (the commit point), `complete`, the restore point
  and, when the tally covered every batch, `"chi_masked"`. The shards are written once. A commit
  writes and flushes the shards, copies `state.pt` to `state.prev.pt`, then replaces `state.pt`, all
  inside `runs.cancel_deferred()`, so a cancel cannot land between the shards and the state that
  points at them. A complete cache is kept: it lets the flow be retrained without simulating again.
  A shard that ends past `batches_done` was written by a save that stopped before its commit. Every
  load skips it (`training_checkpoint.load_rows`). Before its first commit, a training run moves
  every such shard, `x_` and `th_` alike, into a new numbered folder `uncommitted/<n>/` beside
  `shards/` (`training_checkpoint.set_aside_uncommitted`), which no load reads; nothing is deleted.
  The exception is a resume that read its count from `state.prev.pt` (`training_checkpoint.peek`
  reports which file it read): `state.pt` may commit those shards, so the run moves nothing and
  warns, if any lie past the count. `load_rows` refuses two shards whose ranges overlap, naming both
  and the folder to move the stray one into.
- **The cache's identity** (`core.artifacts.identity.SimulationIdentity`, `FORMAT` =
  `"training-rows/3"`) names its directory by a 12-character digest of these fields: the format; the
  model; the prior's fingerprint; the observation mode; the parameter keys and both boxes; the log-box
  parameters; the rotation switch; the rows per batch and the batch count; the time grid's settings;
  whether the box declares a drive; the summary flags and `feature_set_version`; in chi mode the
  layout, slots, channel width, drive, band and lock-in ceiling; the device type and number type; the
  truncation region, None when the run is amortized; and `units_sha256`, the units declaration's
  fingerprint.
  - Bumping `statistics.FEATURE_SET_VERSION` re-keys every cache: a changed feature definition behind
    an unchanged label would otherwise resume onto rows that meant something else.
  - The prior's fit (`prior_fingerprint`) is part of the identity because a resume under another
    prior would splice rows drawn from two priors into one training set, silently changing the
    proposal the flow trains against.
  - Everything in it is known before the Fisher runs, because the digest names the directory the
    rotation is stored in.
  - The flow's network settings are not in it, nor the Fisher settings, the checkpoint cadence or the
    memory ceiling, so a complete cache can be refit at another capacity, and a resumed run reuses the
    stored rotation
    ([The Fisher settings on a resumed run](window.md#the-fisher-settings-on-a-resumed-run)).
  - On a resume, `training_checkpoint.verify` re-checks the header field by field and names the field
    that differs, and the bijection probe catches a changed box.
- **Loading** refuses these mismatches: the manifest's checks run before the payload is read, and
  the payload's own checks after it.
  - `load_prior`: the model, the ND parameter set and its order, the ND box and the log-box mask; then
    the payload's mixture must be the one the manifest fingerprinted. The mode is not checked, so one
    prior serves every mode over its box.
  - `load_posterior`: first `run_guards._assert_chi_config_is_deliberate` (a chi configuration's band
    and drive must be `core/config.py`'s); then the model, the parameter set and its order, both
    boxes and the mode; in chi mode the layout, the slots, the channel width, the band, the lock-in
    ceiling and the drive (`chi_f0`); the conditioning width; then the trained network's own view of
    its mode and block width, its rotation against the prior pickled inside it, and the amortization
    gate.
  - `load_observation`: the model, the parameter order, the mode, the width, and in chi mode the
    layout and slots; then the payload must hash to the manifest's digest and have the declared width.
  - Recorded but never compared on load:
    - a posterior's and an observation's `feature_set_version` and `summary_flags`; only a change of
      width is caught;
    - an observation's chi drive, band and lock-in ceiling (a posterior's are compared), so a
      narrowing round can be drawn around an observation made at another drive;
    - a posterior's and an observation's time grid (the configuration block) and units file hash
      (`inputs`); only the simulation cache's identity carries them.
  - Not recorded at all:
    - the cycle floor below which a probe is masked (`CHI_MIN_CYCLES`, read by
      `chi_probes.gen_chi_raw` when it runs);
    - the smallest probe count training draws (`CHI_K_MIN_TRAIN`, read by `gen_training_data`).

    Neither is part of `SimulationIdentity` or recorded in a posterior's manifest, so a change to
    either is not caught at load, and a retrain could reuse a cache simulated under the old value.
- **The two acceptances.** `Accept(truncated, other_observation)` holds the only escape hatches on
  the load path, and a loaded posterior carries the ones it was loaded under (`accepted`). Each use is
  recorded where the number it produced lives: an inference's `results.accepted` (with its own
  `other_observation`), a calibration's `results.accepted`, a narrowing round's child posterior's
  `training.accepted` (the ones its parent was loaded under), and the `results.accepted` of each
  diagnostic that loads a posterior.
- **The near-miss refusal is exactly one field.** Before the Fisher and before any simulation,
  `build_posterior` looks for committed caches whose identity differs from this run's in exactly one
  field (`training_checkpoint.near_miss_siblings`; `truncation` alone never counts). It refuses
  (field `new_run`) unless the caller consents with `new_run=True`. A cache one setting away is
  almost always the same experiment with something nudged by accident, and the digest would turn the
  nudge into a silent restart from zero; two or more fields usually is a different experiment, so the
  run proceeds with a warning naming the first field each other cache differs in
  (`describe_siblings`). The Posterior tab asks the same question in advance through
  `orchestrator.fresh_run_near_misses`. The flags are in
  [Training-cache flags](command-line.md#training-cache-flags).
- **Reports.** `core.artifacts.report` renders a record as text (`render_manifest`) and its lineage
  back through its parents (`render_lineage`), for both front ends. Both are pure functions of the
  manifests, deterministic, and written to files, never stored as a kind.
- **The default store.** Every stage that writes a record takes `store=` and starts with
  `resolve_store(store)`; the two FDT measurements (`run_fdt`, `run_param_study_cli`) are handed
  writers the front end created instead. The tool
  passes the store at every call inside `use_store`, which puts the previous default back on exit;
  nothing calls `set_default_store` in a run.

## The command-line tool

[The command-line tool](command-line.md) owns every subcommand, flag, output, refusal and exit code.
This section is how `core/tool/` is put together.

- **One module per family**, each with a `register` function that `core.tool.build_parser` calls:
  `stages`, `diagnostics`, `probes`, `smoke`, `fdt` (for `fdt`, `crossval` and `compare`) and
  `browse` (for `artifacts`). `config_args`, `fields`, `help_defaults` and `logging_console` are
  shared.
- **Parsers are built torch-free.** Each handler imports its heavy modules inside its own body;
  `fields` imports nothing, and `help_defaults` only `core.refusals` and `fields`; `main` loads the
  user models only after the parse. So `--help` never imports torch.
- **Knobs are forwarded only when given** (`config_args.knobs`), so each default lives in the stage
  that owns it, read from `core/config.py` or the stage's signature. The exception is
  `validate --seed`: the handler always passes a seed, drawing one with `random.randrange(2**31)` when
  none is given, so the calibration record names the seed it ran with. `smoke` passes its own small
  sizes.
- **`core.tool.main(argv)` returns an exit code** and never calls `sys.exit`, because the suite runs
  it in process. It parses (argparse's own exit becomes 0 or 2), chooses the store root, installs the
  root-logger sink, judges `--note` once, and runs the handler inside `use_store(ArtifactStore(root))`
  and the console handlers. The ladder that follows: Ctrl-C gives 130, with the family's
  `interrupt_note` or the resume advice; a `UsageError` gives 2, caught before `Refusal`, which it
  subclasses; a `Refusal` gives 1, with `fields.fix_sentence(field)` appended; a bare `ValueError` or
  `FileNotFoundError` gives 1, with its class and innermost frame; anything else prints its traceback
  and gives 1. `core.__main__` exits with the code. The printed shapes are in
  [Exit codes](command-line.md#exit-codes).
- **`core/tool/fields.py`** is the only place a refusal key is paired with a flag (`FLAG`);
  `tests/test_refusals.py` pins every value against the parser's real option strings, and the key
  set against the refusal registry, both ways.
- **The console handlers** (`logging_console.console_handlers`) live for one handler call:
  information to stdout as the bare message, warnings and errors to stderr with the level as a prefix.
  Each resolves its stream when it emits, because the suite swaps the streams per test. Library
  records reach stderr through `main`'s root-logger sink. The tool's own output (its banner, record
  and success lines, and the `artifacts` family's listings) is printed, never logged
  ([Conventions](command-line.md#conventions)).
- **`help_defaults`** is the one table of the default clauses: every value flag of every subcommand
  has an entry, and a flag added without one fails the help-walking test, which names the subcommand
  and the flag. `tests/test_docs.py` checks that the command-line page states each default as the
  help does.
- **`smoke` has its own store root**, keyed on `args.temp_store_root`, a property only `smoke`'s
  parser sets. Without `--store-root` it makes a fresh temporary root per run
  (`tempfile.mkdtemp`), and a failed run's root that is still empty is removed with `rmdir`, which
  cannot remove anything that was written. `fdt` and `crossval` take `--store-root` too, but default
  to the `PRISM_ARTIFACTS` root, and `main` must never make a temporary root for them or `rmdir`
  theirs, which is why the key is `smoke`'s own property and neither the flag's presence nor the
  subcommand's name.

## The window

The screens and their controls are on [The window](window.md). This section is how `core/gui/` runs a
stage without freezing, shows its output and stops it.

- **Screens and panels.** `MainWindow` is a navigation shell over Home, the four section screens, the
  Artifacts screen, Settings and the model builder ([Screens](window.md#screens)). Every tab is a
  `BasePanel`: a controls column, a results area (a figure stack over a log pane, with the progress
  pane and its Cancel button beneath them), and `dispatch()`, which runs a callable on a worker
  thread with its output wired to those panes. The six inference tabs share one `SbiSession`, owned
  by `InferenceScreen`, which greys the tabs through `refresh_gates`. `BasePanel._running` is
  class-level, so one run is live in the whole window
  ([One run at a time](window.md#one-run-at-a-time)); the `_REDIRECT` lock in `streams` is the
  backstop.
- **Worker threads.** `Worker` is a `QRunnable` on the global thread pool.
  - `BasePanel` keeps each worker in `_workers`, with auto-delete off, until the worker reports
    `finished`; otherwise Qt deletes the signal sender as soon as `run()` returns and drops the
    result still queued. On `finished` it clears the worker's callable and arguments, which would
    otherwise pin the configuration, the prior and the posterior for the life of the process.
  - A figure is rendered to PNG on the worker (`base_panel._png_fig_sink`), and pickled for the
    pop-out window; the GUI thread shows the image and never paints a figure built on the worker,
    which deadlocks on matplotlib's global lock. A run that saves its figures to disk instead is shown
    through `plot_watcher`, which watches the folder the run writes its figures into: an `fdt`
    record's `figures/`, or the Reduction map's own folder.
  - `core.gui.__main__` forces matplotlib's Agg backend before any `core` import, so a stray
    `plt.show()` in a stage is a no-op on a worker thread.
- **Stream routing.** `streams.redirect_streams` swaps `sys.stdout` and `sys.stderr` for two
  `_SignalStream`s, puts `_PumpLogHandler` on the `core` logger, and routes Python warnings to the log
  pane. Each stream feeds a `vt.StreamRouter`, which classifies each atomic chunk tqdm writes rather
  than scanning for line ends: a paint sits at row n exactly when the next chunk is an up-move of n.
  A daemon `_Pump` publishes to Qt at 15 Hz; that pump is load-bearing, because tqdm redraws without
  any time gate on every `set_description` and `reset`. `_SignalStream` must never gain a `fileno()`
  or an `encoding` attribute: tqdm probes for both, and finding them changes the frames `vt` parses.
- **Progress.** The progress pane shows one overall bar, which follows the live row with the largest
  total above 1, and one caption, which falls back to the deepest row whose total is not 1 when nothing
  reports a percentage, as during sbi's training, whose only output is a printed epoch counter.
  - The solver meter reads `core.progress.SOLVER`, the step count the solver publishes, differenced
    on the pane's 100 ms tick, and never the solver bar's text: a solver call shorter than the bar's
    refresh interval never paints a rate, which is what CUDA graphs made happen.
  - The solver bar is found by its description (`config.SOLVER_BAR_DESC`) only to exclude it from
    the overall bar. It stays enabled, unlike the segment bar `config.QUIET_SEGMENT_BAR` silences,
    because the cooperative cancel and the stall detector both depend on its writes.
- **Cancellation** is cooperative, through a `streams.CancelToken`.
  - Its checkpoints are `_SignalStream.write` and `_PumpLogHandler.emit`: every print, tqdm redraw
    and `core` record of the run passes through one of them, sbi's per-epoch print included.
  - The raise is `WorkerCancelled`, which derives from `BaseException`, so it passes every
    `except Exception` in the pipeline and in sbi; `Worker.run` catches it by name.
  - Latency is about a second almost everywhere, and longer only where a run writes nothing for a
    while: up to one training epoch while the flow trains, and the calibration's C2ST block.
    `pipeline._cancellable_wait` sleeps in one-second slices and logs every five, so a waiting retry
    is cancellable too.
  - tqdm's `refresh()` releases its write lock by hand, so a cancel raised inside a redraw leaks the
    lock and the next bar would deadlock; `redirect_streams` calls `reset_tqdm_lock` when the token
    fired.
  - The token raises only on its owner thread: tqdm's monitor thread also writes to the stream, and
    consuming the token there would lose the cancel.
  - A cancel is deferred, never discarded, inside a `runs.cancel_deferred()` section and while the
    thread is handling an exception (`runs.cancel_is_deferred`), so it can neither split a checkpoint
    commit nor replace a crash that is still unwinding.
  - A hard cancel, running each job in its own process so it can be killed at once, was considered
    and declined in favour of the cooperative one.
- **Settings persistence.** `core/gui/settings.py` is the one `QSettings` store and its typed helpers,
  which survive the store's string round-trip; `use_ini_file` points it at a test's file. The restore
  order for a picker lives in one place, `ArtifactPicker.repoint`: point at the folder, rescan, then
  re-apply the saved key, because a rescan wipes a selection restored first. What is remembered, and
  why, is in [What the window remembers](window.md#what-the-window-remembers).
- **`core/gui/fields.py`** is the window's half of the refusal-naming rule: which control answers each
  refusal field, shown in the yellow "Check your inputs" box.
- **Icons ship as PNGs.** The Qt here has no SVG image plugin, so an icon loaded from an SVG file is
  silently empty; `assets/app/build_app_icon.py` renders the SVG source to the PNGs `app_icon` loads,
  and the small glyphs are a bundled icon font.
- **No third-party widget library.** The Fluent look is the app's own (`design`, `theming`):
  qfluentwidgets was rejected because it is GPLv3.

## The solver, CUDA graphs and reproducibility

- **The solver.** `core.Solvers.sdeint` integrates the stochastic differential equations by Itô
  Euler–Maruyama with diagonal noise. `Simulator` integrates a batch one time segment at a time and
  chooses `euler_compiled` when the device is CUDA and the model exposes a TorchScript
  `compiled_step` (Nadrowski, Hopf, and a user model whose step could be generated; never BP), and
  the plain Python loop, `euler`, otherwise.
- **CUDA graphs.** `euler_compiled` captures `SOLVER_GRAPH_CHUNK` (50) Euler steps into a CUDA graph
  and replays it; the last steps that do not fill a chunk run the TorchScript step without a graph,
  and so does the whole of a call shorter than one chunk. `config.SOLVER_CUDA_GRAPHS` turns the
  graphs off, and is read at each call. The reason: TorchScript removes Python overhead but not
  kernel-launch overhead, and launch overhead was 88 % of solver time. Per step, at a batch of 2,048,
  54.87 µs became 6.65 µs (8.25×); end to end, a simulator call at a batch of 2,048 and 100,000
  steps went from 5,520 ms to 698 ms, a 7.9× gain. The physics checks on the graphed path are in
  [The solver's physics check](science.md#the-solvers-physics-check).
- **Rank a speed-up against a production run, not a smoke run.** Of a production retrain's
  simulation:
  - training generation is about 97 %;
  - the calibration about 3 %;
  - the Fisher rotation 1 to 2 %;
  - `gen_stats`' 41 features about 0.3 %, about 7.5 minutes over the whole run.

  So the Fisher's call structure and the duplicate FFTs in `core.SBI.statistics` are together worth
  about 20 minutes of roughly 38 hours. On a four-batch smoke run the Fisher dominates instead, which
  is why it looks like the target.
- **The graph path's invariants.**
  - The parameters are static buffers copied in on every call, never captured by reference, which is
    what lets one capture serve every batch; a captured reference would replay one model's physics
    for the next batch, silently.
  - The cache is a module-level dictionary, `_GRAPH_CACHE`, keyed on the step, the batch shape, the
    drive channels, the chunk, the number type, the device and the time step, and bounded at
    `SOLVER_GRAPH_CACHE_MAX` (8) because graph memory lives in private pools `empty_cache` cannot
    reclaim. It is not held on `Solver`, which is built once per segment and must stay patchable
    (`test_the_graph_cache_is_not_hung_off_the_solver_class`).
  - The graphed path advances the progress bar and the step counter together, by the chunk, through
    `sdeint._advance`; never call `bar.update` alone. The ungraphed loops count through
    `sdeint._step_iter`.
  - The state is carried forward inside the graph: its state buffer is both the capture's input and
    its output, so consecutive chunks need no copy between replays.
- **A capture failure falls back to the ungraphed TorchScript loop**, with a warning, for the rest
  of the process (`_acquire_graph`): a solver that refuses to run is worse than a slow one. On an
  out-of-memory path `drop_graph_cache` releases the captured graphs, because the halving retry is
  about to capture another at the smaller width.
- **`Solver()` is built on every call, on purpose.** `Simulator.__sols` constructs it once per
  segment, so resolving `sdeint.Solver` at call time is the seam a test uses to swap the solver out
  (`test_solver_failure_raises_instead_of_killing_the_process`). A solver failure is raised as a
  `SimulationError` chained to its cause.
- **The segment seam duplicates one sample.** Each segment starts from the previous segment's last
  state and writes it again as its first sample, so every boundary repeats a sample: a run of k
  segments advances k − 1 fewer steps than its time grid implies, and the grid and the solution are
  not exactly co-indexed. This is negligible for the spectral and autocorrelation features at three
  segments or fewer; a feature that read an instantaneous phase or a finite difference across a seam
  would see a zero-length step.
- **Seeding.** `core.rng.seeded(seed, device)` seeds torch and numpy for a block and restores both
  afterwards; `smoke`, every diagnostic that draws (SBC, the Laplace and Jacobian checks, the probe
  checks), the FDT measurement and a seeded calibration run inside it. A run is seeded once and its
  streams run on: seeding each stage would start draws that must be independent from the same state.
  A seeded calibration derives its own seed (`calibration_seed`), so it never replays the strata of a
  training run seeded with the same number.
- **Runs are not bitwise-reproducible on CUDA or across devices.** A kernel's reduction order is not
  fixed; the graphed path draws its noise in a different order from the ungraphed loop; and on a card
  the planner and the halving ladders split a batch by the memory free at that moment, which redraws
  its noise in other blocks. A seed buys a repeatable experiment on one device, and a bitwise repeat
  on the CPU only.
- **TorchScript is not bitwise-reproducible on its first runs.** Its profiling executor runs the first
  calls of a scripted step unoptimised, then fuses them, and the fused kernel differs by about one
  unit in the last place: three ungraphed runs of one deterministic model gave a first run that
  differed from the second, and a second equal to the third. What that asks of a bitwise comparison
  is in [Randomness and reproducibility](rules-and-traps.md#randomness-and-reproducibility). The
  TorchScript path runs only on CUDA, so the CPU suite's reproducibility checks are untouched.

## Memory planning and out-of-memory recovery

Everything here is in `core.SBI.pipeline` unless named otherwise. The settings
(`SIM_VRAM_CEILING_GIB`, the batch-retry attempts and delays, the allocator policy) are in
`core/config.py`; the planner's own constants (`_SIM_MEM_FRACTION`, `_MIN_SIM_CHUNK`,
`_FORCE_BUILD_PEAK_MULTIPLE`, the learned budget's back-off and recovery, `_MEM_LOG_EVERY`) are
module constants of `core.SBI.pipeline`.

- **The planner.** `gen_obs` asks `_max_sim_batch` for the largest batch whose peak fits the budget.
  - The peak is `peak_sim_elements`: the solution buffer and the drive, which live throughout, plus
    the larger of the solver's segment buffer and the kept copy. It is public so the Posterior tab's
    memory line reads the same formula.
  - The budget is `sim_memory_budget_elements`: `config.memory_budget_elements` (the free memory plus
    the allocator's reusable cache, times `_SIM_MEM_FRACTION`, 0.85; a failed reading falls back to a
    small, safe 1 GiB), capped by the learned cap below, then by `vram_ceiling_gib()` when that is
    above 0. The ceiling is `PRISM_VRAM_CEILING_GIB` when set, else `config.SIM_VRAM_CEILING_GIB`, and
    is read on every plan; how to size it is in
    [The inference settings](window.md#the-inference-settings).
  - A batch that does not fit is split into power-of-two chunks, because the solver specialises on the
    batch width, no smaller than `_MIN_SIM_CHUNK` (256), each writing into one preallocated result.
    When not even a 256-row chunk fits beside the result, the batch runs as asked, and an
    out-of-memory error is left to the ladders below.
    Off CUDA the planner never splits.
- **The learned budget.** On Windows, `torch.cuda.mem_get_info` counts other processes' evictable
  surfaces as free (measured: 15,037 MiB reported against the 5,814 MiB `nvidia-smi` showed), so the
  reading is only an upper bound. `_BUDGET_CAP_ELEMENTS` learns the real ceiling from outcomes:
  - an out-of-memory error at N elements caps the budget at 0.8 N (`_budget_note_oom`); the row
    ladder charges it at `_per_row`, the element cost of one row at that batch's geometry;
  - after 32 clean units the cap rises by 10 % (`_budget_note_ok`), still clamped by the reading;
  - the credit is per batch: inside `gen_training_data` only the batch tail's
    `_budget_note_ok(batch_level=True)` counts, once per completed batch, because a chi batch makes
    1 + K `gen_obs` calls and counting those unwound the back-off in about three batches;
  - it adapts the budget in elements, not the batch width, because whether a batch fits depends on its
    geometry, which the planner already handles; and it is never persisted, because the right cap
    depends on what else is on the card now.
- **The drive's build peak.** `_FORCE_BUILD_PEAK_MULTIPLE` is 4: building a drive holds four
  tensors the size of the one it returns (measured 4.10× on the card). `_per_row` charges it, so an
  out-of-memory error teaches the cap what really failed.
- **The ladders.** Each catches `RuntimeError` alone, so a `WorkerCancelled` passes straight through,
  and re-raises anything that is not an out-of-memory error at once, traceback intact. `_is_oom`
  accepts `torch.OutOfMemoryError`, a raw driver `AcceleratorError`, and either wrapped in a
  `SimulationError`.
  - `_gen_obs_retry`, inside `gen_obs`, halves the simulator batch and re-runs it, down to 256 rows.
  - `_rows_with_oom_retry` halves a training batch's rows and re-runs everything for them (the
    spontaneous run, the statistics, every probe). It catches what the simulator-level split cannot
    see: the drive tensors, the gather indices, the stitched result. The halves share the batch's
    (t_scale, T_obs) and its probe set, so together they are the batch that would have been produced,
    apart from how its noise was drawn.
  - `retry_on_oom`, for newer call sites, waits and retries: the Fisher rotation wraps each operating
    point in it (`decorrelate`; there `cudaErrorUnknown` is retryable too, and an exhausted point is
    skipped) and the posterior predictive check each bin (`ppc`).
- **The batch-level retry**, inline in `gen_training_data`, is the outermost: when both halving
  ladders are exhausted it waits and re-runs the whole batch, up to `TRAINING_BATCH_RETRY_ATTEMPTS`
  (3) times, waiting `TRAINING_BATCH_RETRY_DELAYS_S` (15, 60 and 180 s) before each. Tests assert the
  order of its statements on the parsed source, so it stays inline.
- **Restore, then release, then wait.** The retry restores the random streams first, from a snapshot
  taken just before the rows were made: a few KB, served from the allocator's cache while the process
  still holds it. It then hands the cache back (`_release_device_memory`), logs one `[mem]` line, and
  only then waits. Released first, the few KB became a fresh driver request on a contended card, and
  that request killed a run. The restore is best-effort (`_try_rng_restore`): skipping it makes the
  re-run a different, equally valid draw.
- **Notice first, release outside the handler.** Every ladder logs its warning before it releases,
  and releases after the `except` clause has closed, when the failed attempt's tensors are gone.
  `_release_device_memory` never raises: the allocator cache, the cuFFT plan cache and the captured
  graphs are released under separate guards. A hot loop passes `plans=False, graphs=False`, and the
  per-batch tail clears the cuFFT plan cache but keeps the captured graphs.
- **The rows matter more than the restore point.** Resuming without a restore point draws fresh
  noise from that batch on, a reproducibility loss; a skipped write throws away hours of simulation.
  So a failed per-batch snapshot leaves that batch with no restore point rather than a wrong one
  (`_pending_rng` is paired with its batch index), a failed snapshot at a cadence boundary defers the
  write to the next one, and the rescue write on a cancel or crash commits the completed batches
  without a restore point rather than skipping them.
- **`_we_are_the_holder`.** Waiting helps only when someone else holds the memory. When this process's
  reserved pool is larger than every other process's usage combined, the retry goes straight back to
  the halving ladders instead of sleeping; a run was once found waiting for memory it held itself,
  about 15 GiB. The captured graphs could not have been that memory: the cache holds at most eight,
  whose static buffers come to about 1.7 MB each at 2,048 rows, and even at a generous 20 MB per
  private pool the total is about 174 MB, so the graph cache cannot be the memory a stalled batch
  waits for.
- **WDDM spills into shared memory.** On Windows a batch larger than the card's memory pages into
  shared system memory instead of failing: a 21.67 GiB allocation completed on a 15.92 GiB card,
  9× slower than unpressured. So a slow run is the warning sign, long before any out-of-memory error,
  and the `[mem]` line's peak reserved is what shows it. What does kill a run is a transient, a moment
  when the desktop takes the memory, which is why waiting beats shrinking; the ceiling's value is
  keeping a batch inside the card's own memory.
- **The `[mem]` line** (`_log_memory`) reports peak allocated and peak reserved, then resets both
  peaks: on batch 0, every `PRISM_MEM_LOG_EVERY` batches (its default is under
  [Environment variables](command-line.md#environment-variables)), and after every out-of-memory
  error that reaches the batch-level retry or `retry_on_oom`. Peak reserved against peak allocated
  separates a batch too big for the card from an allocator that cannot hand memory back. It is
  best-effort: a diagnostic must never kill the run.
- **The allocator policy.** `core.config` gives `PYTORCH_CUDA_ALLOC_CONF` an allocator policy when
  it is imported, unless one is exported (the value is under
  [Environment variables](command-line.md#environment-variables)). It recovers about 6 GiB: a
  batch's fine-step count runs from a median of about 40,000 to a 99th percentile of about 283,000,
  and without the policy one large batch carved segments the smaller ones could not reuse. The trap
  in the variable's name is in [Memory and devices](rules-and-traps.md#memory-and-devices).
- **Accepted gaps.**
  - The prior's stability sweep simulates through `Simulator.simulate` directly
    (`core/SBI/Priors/*_prior.py`), so no out-of-memory ladder protects it.
  - The Fisher rotation wraps each feature evaluation in `torch.random.fork_rng`, whose exit restores
    the card's generator state, a device call, while an out-of-memory error unwinds; on a starved card
    that restore can fail, and its error then replaces the out-of-memory traceback.
  - On a resume, copying the stored initial states to the device and restoring the generators run
    before the generation loop's `try`, so a failure there gets no rescue write; nothing has been
    generated yet, so it costs only a restart (`core.SBI.pipeline.gen_training_data`).

## The FDT pipeline

`core/FDT/` measures the effective temperature of a simulated bundle; the commands are
[fdt](command-line.md#fdt), [crossval](command-line.md#crossval) and [compare](command-line.md#compare).

- **`fdt_pipeline.run_fdt`** is the measurement for one cell. The front end mints the record's writer
  (`store.create("fdt", ...)`) and hands it over unentered, so it can point the figure watcher at the
  record's `figures/` first; `run_fdt` enters it on the worker thread, where the run's log buffer
  lives. Before the record opens it resolves the noise prefactor
  (`campaigns.observable_noise_prefactor`) and the seed, so a cell the analysis cannot normalise opens
  no record. Then, inside `seeded(seed, device)`:
  - `sanity.run_all_sanity`: five checks (the passive baseline, the high-frequency limit, linearity,
    ensemble convergence and the spectrum window), the first two for the Nadrowski model only;
  - Campaign 1 (`campaigns.run_campaign1_psd`): an ensemble of spontaneous runs, and its one-sided
    Welch power spectrum, `spectral.psd_welch`;
  - Campaign 2 (`campaigns.run_campaign2_chi`): a driven ensemble at each frequency, and χ(ω) by
    `spectral.lock_in_chi`;
  - `spectral.eff_temp_ratio`: T_eff/T = p ω G(ω) / (4 χ″(ω)), where p is the model's prefactor,
    the drive's coupling over the observable's diffusion coefficient (n · beta for the Nadrowski
    model); `plots` draws the figures.
- **`cross_validation.run_param_study_cli`** is the sweep study: `run_fdt_param_sweep` runs twice,
  the S sweep at T_a/T = 1 and the T_a/T sweep at S = 0, each into its own `fdt` record, with each
  point's seed derived from the study's (`_point_seed`). `cross_validation_plots` draws the surfaces.
- **`compare.compare(mode, refs, ...)`** reads saved records (cells, repeats, renormalise, sweeps),
  puts them on a common grid and writes an `fdt` record whose body names what it compared
  (`compared`), not its `parents`.
- **The CPU only.** `cli.make_fdt_config` and `cli.make_param_sweep_config` set the device to
  `config.cpu_device()`, because the solver steps a small ensemble (M ≈ 256, three to five state
  variables) one time step at a time, so each step is a handful of tiny tensor operations. The figure
  the code gives, about 3.4× faster on the CPU at M ≈ 256 with a crossover near M ≈ 4096, dates from
  before CUDA graphs existed, so it measured the ungraphed loop, and it has not been re-measured
  since.
- **Its records are progressive.** Each is refreshed as the run goes, a cancel or a crash keeps the
  folder marked unfinished, and nothing resumes ([The artifact store](#the-artifact-store)).
