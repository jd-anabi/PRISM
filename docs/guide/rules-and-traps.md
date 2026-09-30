# Rules and traps

Checked against commit 30ff8a3.

This page is for whoever changes the code. It holds the rules the code keeps, each with where it is
enforced; the safety rules of a narrowing round, as a table; and the traps: mistakes that once
failed without a word, grouped by subsystem, each with the guard that now stands in the way. Code is
cited by module and function name, a test as `tests/<file>.py::<name>`. The design is in
[Architecture](architecture.md), the reasons behind the science settings in
[The science behind the settings](science.md), every flag and refusal in
[The command-line tool](command-line.md) and every control in [The window](window.md); this page
links to them rather than repeating them.

## Rules the code keeps

**The model and solver contract.** There is no model base class (`core/Models/__init__.py` is
empty): a model is whatever the simulator and the solver can call, and a break in the contract fails
silently.

- `f(x, t)` is the drift, `(batch, d)` in and out. `t` is the solver's integer step index within the
  current time segment, not a time, and the drive is folded into the drift as an additive
  `self.force[:, channel, t]`: there is no separate force method. `Simulator.simulate`
  (`core.Simulator.simulator`) slices the drive to each segment, so that index reads the right
  values.
- `g()`, or `g(x)` when the noise depends on the state (`state_dep_drift=True`), returns a
  `(batch, d)` vector of diagonal noise amplitudes, which the solver (`core.Solvers.sdeint`)
  multiplies elementwise; a noiseless channel is 0. `core.Models.bp_model_steady.BPModelSteady.g`
  still returns full `(batch, d, d)` matrices, which the solver does not consume: do not copy it. No
  shipped bounds file reaches it, because every BP box declares 17 parameters, which selects the full
  model (`core.Simulator.bp_simulator.BPSimulator`).
- `force` is a plain attribute that the simulator overwrites once per segment
  (`self.sde.force = ...`); assigning the simulator's own `force` property instead rebuilds the
  whole model. The model also carries `device`, to which the solver moves the initial state.
- Construction is positional, so the parameter order in the bounds file must equal the model's
  argument order. Each built-in simulator's `_set_up_model` calls
  `Model(*torch.unbind(params, dim=1), force, ...)`, and a failed construction raises
  `core.Simulator.simulator.SimulationError`, chained to its cause.
  `core.Simulator.user_simulator.UserSimulator` hands the same columns, in the same order, to
  `UserModel` as one tuple after the compiled definition, and raises a plain `RuntimeError`.
- The observable is state column 0 on every path: most `core.SBI.pipeline.gen_obs` callers ask for
  `var_idx=0`, so that only that variable is copied; the identifiability Jacobians take row 0 of the
  full result; and the FDT measurements and the prior screens index row 0 of what
  `Simulator.simulate` returns. A user model's first variable is its observable.
- The solver is Itô Euler–Maruyama and derives its step from the time grid; nothing passes it one
  ([The solver, CUDA graphs and reproducibility](architecture.md#the-solver-cuda-graphs-and-reproducibility)).
- On CUDA a model that has a TorchScript `compiled_step` takes the compiled path
  (`sdeint.Solver.euler_compiled`), which never calls `f`: it passes `compiled_params()` into the
  step by position, so that tuple's order is load-bearing and a new parameter is only ever appended
  (`NadrowskiModel.compiled_params`). A reordering binds parameters to the wrong names with no error,
  and the CPU suites, which run the plain loop, never see it.

**The force-channel rule.** `core.forcing.n_force_channels` is the one answer to how wide a drive
tensor must be, and it is a property of the model's drift, not of the drive a cell declares: one
channel for Nadrowski and BP; two for Hopf, whose drift reads channel 1 whether or not the cell
declares a second amplitude; one per state variable for a user model, whose drive parameters are
named `<parameter>_<variable>`. A tensor sized from the cell's drive is too narrow for Hopf, and one
sized by the number of state variables made the largest tensor of a training batch three times too
big for Nadrowski and five times for BP.

**The bounds, cell and units files.** The bounds file declares which parameters are inferred, in
what order and over what box, and so the observation mode; a cell file holds values and initial
conditions only; the units file declares what the numbers mean and never converts them. A cell's
model is its folder's name (`core.cli.parse_cell`). `SimConfig.inject_ground_truth`
(`core.sim_config`) refuses a cell that lacks an inferred parameter the bounds file declares, or puts
one outside its box (field `cell`), and ignores and reports the values the bounds file does not
declare, so one cell serves several bounds files. Every inference command is given its bounds file;
only the runs handed a cell alone look one up (`core.cli.resolve_bounds_for_cell`). The formats and
the lookup are in [Input files](recordings.md#input-files) and
[How a cell finds its bounds file](recordings.md#how-a-cell-finds-its-bounds-file). Pinned by
`tests/test_artifact_consistency.py::test_bounds_resolution_prefers_a_sibling_then_falls_back_to_master`
and `tests/test_artifact_consistency.py::test_every_master_cell_injects_under_every_mode`.

**Everything is non-dimensional.** Drift and noise are written without units. Physics enters only
through the rescale parameters `x_scale`, `t_scale` and `f_scale`, which are inferred with the rest,
not derived, apart from the two exceptions [Nondimensionalisation](science.md#nondimensionalisation)
lists. Nothing non-dimensionalises a value automatically, and a frequency is cycles per cell time
unit throughout, converted from hertz by `SimConfig.freq_si_to_cell` alone.

**The conditioning layout.** Every conditioning row is
`[41 summary features | 8 valid flags | log T_obs | forcing or chi block]`, and one constructor builds
it, `core.SBI.statistics.conditioning_rows`, which training, the simulated and experimental
observations and the posterior predictive check all call. In the chi block a dead slot, a pad or a
masked probe, is exactly 0.0 in all six channels (`core.SBI.chi.pack_probe_block`), so it is bitwise
inert and no finite-value filter drops its row; a probe that failed is masked, never packed as a
live-looking value. The widths and the reasons are in
[Conditioning features](science.md#conditioning-features). Pinned by
`tests/test_chi_set_encoder.py::test_packer_round_trips_and_masks_failures` and
`tests/test_chi_set_encoder.py::test_a_failed_probe_is_masked_not_a_phantom`.

**Every setting is an argument, never a configuration edit.** The front ends pass every setting to
the stage, because assigning a `core.config` constant after the stages are imported changes nothing
they read ([Stages and compositions](architecture.md#stages-and-compositions) says why). The one
run setting the window writes into `core.config` is the memory ceiling, which the batch planner
reads through the module every time it plans
([The inference settings](window.md#the-inference-settings)); the simulation pipeline also reads
two environment variables of its own
([Environment variables](command-line.md#environment-variables)). Pinned by
`tests/test_user_sbi.py::test_build_posterior_takes_the_budget_as_arguments_because_the_constants_are_snapshotted`
and
`tests/test_settings_persistence.py::test_the_training_budget_reaches_build_posterior_as_arguments_not_via_config`.

**Every public entry works on a private copy of its configuration.** `core.runs.public_entry` hands
every public stage, composition and diagnostic, and the FDT measurement and sweep study,
`cfg.copy_for_run()`, so the caller's configuration is unchanged whether the run succeeds, is
refused or crashes; the FDT comparison (`core.FDT.compare.compare`) takes no configuration and
carries the decorator for its run log alone. No stage mutates what it is handed: a caller that
wants what a stage produced reads the record, or the `Loaded*` wrapper the stage returns. Pinned by
`tests/test_artifact_store.py::test_every_public_entry_leaves_the_callers_config_untouched` and
`tests/test_artifact_store.py::test_the_public_entries_carry_public_entry_and_nothing_else_does`.

**Every pre-spend refusal names its setting by a field key.** A bad input is refused before anything
is spent, as a `core.refusals.Refusal` whose `field`, when one setting is at fault, is a key
registered in `core.refusals.FIELDS`. The core's message names the setting in words and never a box,
tab, flag or button; each front end appends the control or flag that answers the key from its own
table (`core/gui/fields.py`, and `FLAG` in `core/tool/fields.py`). A programming error is a plain
exception, never a refusal. Known gaps, which are not the rule, include:

- plain `ValueError`s: the `ablation` diagnostic's refusals about its simulation cache and its
  network (`core.diagnostics.ablation.channel_ablation`), the command-line tool's refusal of an
  unsupported model (`core.tool.config_args.make_cfg`), the Laplace identifiability check's refusal
  of a posterior with no latent training prior (`core.diagnostics.identifiability`), and a narrowing
  round's refusal when every direction loads on t_scale (`core.SBI.truncate.region_from_posterior`,
  reached before any simulation);
- a store lookup's error for a reference that resolves to no record carries no field in the loaders
  (`core.artifacts.store.ArtifactStore.load_prior`, `load_posterior`, `load_observation`,
  `load_calibration`, `load_inference`, `load_diagnostic`) and in `get`, `path` and `rename`;
  `set_note`, `delete`, `read_log` and `load_fdt` name the field `artifact`;
- a units-file refusal carries the field `units`, which no flag answers.

How the tool prints each is in [Exit codes](command-line.md#exit-codes). Pinned by
`tests/test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered`,
`tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it`
and `tests/test_refusals.py::test_every_domain_error_is_a_refusal_and_carries_a_field`.

**No chi override for anything that trains or infers.** The chi band, the chi drive amplitude and
the cycle floor below which a probe is masked come only from `core/config.py` (`CHI_FREQ_BOUNDS`,
`CHI_F0`, `CHI_MIN_CYCLES`); no front end sets them for a run.

- `core.SBI.run_guards._assert_chi_config_is_deliberate` refuses a chi configuration whose band or
  drive differs from `core/config.py`'s, with no override: `build_prior` and `build_posterior` run it
  before the first simulation, `probe_mask` before its audit, and `ArtifactStore.load_posterior`
  before any comparison and before the payload is read. A different band or drive means editing
  `core/config.py` deliberately. The probe count is not checked, on purpose: it is what one
  observation supplies, and training draws its own for every batch.
- The window's Config tab also has boxes for the lock-in ceiling and the number of probe slots
  ([The inference settings](window.md#the-inference-settings)).
- `ArtifactStore.load_posterior` refuses a chi posterior whose band, drive, lock-in ceiling or slot
  count differs from the loading configuration's.
- The cycle floor is outside the cache's identity and every record
  ([The artifact store](architecture.md#the-artifact-store)); see
  [The chi probe set and its Fisher](#the-chi-probe-set-and-its-fisher).
- The `probes` checks may take their own frequency, drive and ceiling grids because they only
  measure: none of their grids reaches a configuration or the training generator
  ([probes](command-line.md#probes)).

Pinned by
`tests/test_artifact_consistency.py::test_a_chi_run_at_a_non_default_band_is_refused_before_the_simulation_spend`,
`tests/test_artifact_store.py::test_load_posterior_refuses_a_chi_posterior_trained_at_another_drive`,
`tests/test_nav_and_gating.py::test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config`
and
`tests/test_diagnostics.py::test_a_probes_run_leaves_a_training_configuration_and_config_py_untouched`.

**A narrowed parent is refused around another observation, with no escape hatch.**
`core.orchestrator.tsnpe_round` refuses a narrowed parent posterior with any observation but its
own, before the region is drawn, and no acceptance answers it: a narrowed posterior is valid only
near its own observation, so a region drawn from it anywhere else is drawn where its flow
extrapolates. Draw the round around the parent's own observation, or start from an amortized
posterior. Pinned by
`tests/test_conditioning_repair.py::test_a_non_amortized_parent_is_refused_on_another_observation`.

**A persisted value that defines what is measured is compared with `core/config.py` before the
spend.** A remembered preference is harmless; a remembered measurement definition outlives the change
to `core/config.py` that retired it, which once cost a multi-day run
([What the window remembers](window.md#what-the-window-remembers)). So the inference tabs remember
no science setting, the band and drive are read-only displays, and
`_assert_chi_config_is_deliberate` compares them with `core/config.py`, not with another record,
before the first simulation. A setting persisted in future that defines the measurement needs the
same comparison. Pinned by
`tests/test_settings_persistence.py::test_science_knobs_open_at_config_and_are_not_written` and
`tests/test_settings_persistence.py::test_the_config_tab_science_knobs_open_at_config_and_are_not_written`.

**A pre-flight reads its configuration from the same place as the run it clears.** Smoke runs that
built their configuration from `core/config.py` were once green while the window, restoring its own
saved band, was about to train on another: no measurement of a pre-flight can reveal that. So the
band and drive check lives on the path both front ends share (`core.SBI.run_guards`, run by the
stages in `core.orchestrator`) and compares against `core/config.py`; both front ends build their
configuration through `core.cli.make_sim_config` (the window through `core.gui.session.ConfigDraft`,
the tool through `core.tool.config_args.make_cfg`); and `smoke` runs the same stages.

**The comment policy.** A comment states its reason in words, and cites functions, not line
numbers; nothing scans for line numbers, and some comments still carry them. It cites no planning
or working document, nor any label one of them coined, and that half is enforced:
`tests/test_source_hygiene.py::test_no_scanned_file_cites_a_working_document` reads the comments
and strings under `core/` and `tests/`, the root `conftest.py`, `README.md`, `requirements.txt`,
`pytest.ini`, `run.bat`, `run.sh` and every page of this guide, and
`tests/test_source_hygiene.py::test_no_test_name_carries_a_process_label` the test names. A
legitimate token shaped like a label goes on the scan's allowlist in the commit that introduces it.
Compress a comment, but never delete a measured number, a note that something was tried and
regressed, an ordering requirement, or a note that something looks removable and is not: each exists
because something died without it. Several tests read the parsed source and the order of the calls
inside a function, so a rename or a reordering can fail them by design, and a test that pins an
order asserts the order, never that two calls are neighbours
([Writing tests](testing.md#writing-tests)).

## Narrowing-round safety rules

A narrowing round (TSNPE, `core.orchestrator.tsnpe_round`) draws a region around one observation
from a posterior and trains a new posterior on the prior restricted to it
([tsnpe](command-line.md#tsnpe)). The reasoning behind each rule is in
[Narrowing rounds](science.md#narrowing-rounds); this table says what each prevents, where the code
enforces it and the test that pins it.

| rule | what it prevents | where it is enforced | the test that pins it |
|---|---|---|---|
| **The truncated-prior rule:** a narrowing round draws from the prior restricted to the region, never from the posterior | tempering: intervals that narrow round after round with no new information, while SBC stays flat | `core.orchestrator.build_posterior` wraps the training prior in `core.SBI.truncate.TruncatedLatentPrior`, which rejection-samples the prior and whose `log_prob` is the prior's inside the region and −∞ outside; `tsnpe_round` hands the stage only the region | `tests/test_conditioning_repair.py::test_the_proposal_is_the_TRUNCATED_PRIOR_and_not_the_posterior`; `tests/test_conditioning_repair.py::test_the_truncated_prior_is_the_base_density_inside_and_minus_inf_outside` |
| **The observation-digest rule:** a round refuses unless the stored observation matches | a region, which deletes support for good, drawn around data nobody recorded, or a narrowed posterior that can never be checked against its own observation | `orchestrator.build_truncation_region` takes a stored observation record and re-checks its digest; `build_posterior` refuses a region that names no observation before any simulation; `tsnpe_round` refuses a narrowed parent with another observation; `ArtifactStore.load_posterior` refuses a narrowed record whose region names none | `tests/test_artifact_store.py::test_a_round_whose_region_names_no_observation_is_refused_before_the_spend`; `tests/test_conditioning_repair.py::test_a_non_amortized_parent_is_refused_on_another_observation` |
| **The narrowed-model rule:** a narrowed posterior is marked as such and never loads or infers as a broad one | a flow that saw rows only inside its region, extrapolating with confidence everywhere else | `core.artifacts.manifest.validate` (the amortized flag must agree with the region); `ArtifactStore.load_posterior` (refused without `Accept(truncated=True)`); `orchestrator.infer_and_visualize` and `simulated_inference` (another observation refused without `Accept(other_observation=True)`); each acceptance is recorded | `tests/test_artifact_store.py::test_a_posterior_manifests_amortized_flag_must_agree_with_its_region`; `tests/test_artifact_store.py::test_a_non_amortized_posterior_needs_accept_and_the_flag_is_recorded`; `tests/test_artifact_store.py::test_simulated_inference_refuses_a_non_amortized_posterior_before_it_simulates`; `tests/test_tool.py::test_validate_refuses_a_non_amortized_posterior_without_accept_truncated` |
| **The eigenbasis rule:** the region is cut in the rotation's leading directions, flat ones left full width | barely constrained directions cut on noise | `core.SBI.truncate.region_from_posterior` takes intervals on the leading directions of the parent's latent only; `build_posterior` wraps the truncated prior around the rotated latent prior; `tsnpe_round` refuses more directions than the latent has | `tests/test_conditioning_repair.py::test_the_region_is_built_over_the_leading_fisher_directions_only`; `tests/test_conditioning_repair.py::test_tsnpe_round_refuses_bad_direction_counts_and_hpd_levels_before_any_spend` |
| **The unweighted-draws rule:** the region comes from unweighted posterior draws, not best fits | a second, undeclared likelihood, with the discrepancy measure as a hidden setting | `truncate.region_from_posterior` takes per-direction quantiles of the flow's own draws (20,000, seeded from the observation), with no ranking step | structural, no dedicated test |
| **The generous-region rule:** a 99.9 % region, with the truth-outside rate watched | a tight region deleting support no later round can recover | `truncate.DEFAULT_HPD`; `tsnpe_round` refuses a level outside (0, 1) and warns below 0.99; `build_posterior` reports, for a round around a simulated cell, whether the truth lies inside each truncated direction (`TruncationRegion.containment`), records it in `training.truth_containment` and warns when it is outside, and logs the fraction of the prior the region kept | `tests/test_conditioning_repair.py::test_tsnpe_round_refuses_bad_direction_counts_and_hpd_levels_before_any_spend`; `tests/test_conditioning_repair.py::test_a_tight_hpd_warns_through_the_preflight_channel_and_still_runs`; `tests/test_conditioning_repair.py::test_a_narrowing_round_reports_and_records_whether_the_truth_lies_inside_each_direction`; `tests/test_conditioning_repair.py::test_a_region_reports_containment_per_truncated_direction` |
| **The cost-on-screen rule** | a round, a full simulation campaign, started without its cost in view | the TSNPE tab shows the Posterior tab's budget group (`core.gui.panels.inference.base._TrainingBudgetMixin`); on the command line `build_posterior` logs its `[budget]` line on every training run | `tests/test_artifact_store.py::test_the_budget_line_prints_with_default_arguments`; `tests/test_settings_persistence.py::test_the_training_budget_shows_the_simulation_count_and_the_cap_trade`; `tests/test_settings_persistence.py::test_the_budget_lines_only_format_the_preview` (the TSNPE tab's own lines too) |
| **The region-carries-its-basis rule:** the child reuses the parent's rotation, never recomputes it, skips directions loaded on t_scale, and names its base prior | a region drawn in one rotation and enforced in another: one such round kept 0.01 % of its parent posterior and lost the truth | `build_truncation_region` records the parent's rotation, a probe of its training bijection, the observation's digest and the parent's prior fingerprint; `build_posterior` trains in that rotation without running the Fisher and refuses a basis mismatch (`TruncationRegion.check_basis`), a rotation setting that disagrees, a resumed cache from another region or rotation (`TruncationRegion.check_checkpoint_V`) and another prior (`run_guards._assert_prior_matches_region`); `region_from_posterior` skips a direction loaded on t_scale above `truncate.t_scale_loading_max` | `tests/test_conditioning_repair.py::test_a_region_measured_in_one_basis_is_refused_in_a_sign_flipped_one`; `tests/test_conditioning_repair.py::test_build_truncation_region_records_the_parents_basis`; `tests/test_conditioning_repair.py::test_a_truncated_round_refuses_a_prior_other_than_the_parents`; `tests/test_conditioning_repair.py::test_a_t_scale_loaded_direction_is_excluded_from_the_region`; `tests/test_user_sbi.py::test_a_tsnpe_round_reuses_the_parents_basis_and_refuses_every_mismatch`; `tests/test_user_sbi.py::test_a_truncated_round_routes_to_its_own_checkpoint_and_the_amortized_digest_is_untouched` |
| **The calibrate-on-the-region rule** | a calibration drawn from the full prior, which reports a miscalibration that is not one | `orchestrator._calibration_prior`, shared by `validate_calibration` and `core.diagnostics.sbc_repeats`, restricts the rotated calibration prior to the region after checking its basis; `orchestrator._sbc_reference_sample` draws the rank test's reference from that same proposal, t_scale override included | `tests/test_user_sbi.py::test_calibration_theta_star_lies_inside_the_region_when_one_is_given`; `tests/test_diagnostics.py::test_the_calibration_draw_is_three_helpers_with_the_stratification_seam` |

Three rules are still only prose; nothing in the code enforces them.

- **The rate of truths outside the region.** Each round reports and records its own answer, per
  direction, but nothing keeps a rate across rounds, and a round around a bench recording has no
  truth to check.
- **A region cut in the box when the parent has no rotation.** A parent trained with the rotation
  off yields a region on its latent box axes, taken in the bounds file's order (t_scale's own axis
  skipped), not ranked by how well each is constrained; `build_posterior` checks only that the
  round's rotation setting agrees with the region's, so nothing refuses or warns.
- **The −log P(A) inflation, printed, not corrected.** A narrowed posterior's informativeness is
  measured against the full prior; `validate_calibration` prints the inflation beside the kept
  fraction, from the rejection sampler's acceptance before the t_scale override, and records the
  total unadjusted.

## Traps

Each trap is one rule, the silent failure it prevents, and the guard that now stands in the way,
cited by module and function. Where nothing guards it, the trap says "No guard". The design around
each group is on the page linked at its head.

### The window: threads, figures and paths

How `core/gui/` runs a stage off the GUI thread is in [The window](architecture.md#the-window).

- **Keep a worker's reference until it finishes.** Otherwise Qt drops the worker's queued result and
  `finished` signals and the panel stays busy for ever. Guard: `BasePanel.dispatch`
  (`core.gui.panels.base_panel`), which keeps each worker in `_workers` until `finished`; never start
  a raw `Worker` outside it.
- **Never paint a figure built on a worker.** Painting one in a live canvas deadlocks on
  matplotlib's global lock, with no traceback. Guard: `base_panel._png_fig_sink`, which renders each
  figure to PNG on the worker.
- **Force the Agg backend first.** Under Agg a stray `plt.show()` in a stage is a no-op, which is
  what makes a stage safe to run on a worker thread. Guard: `core.gui.__main__` and `core.__main__`
  call `matplotlib.use("Agg")` before any `core` import; do not move either line.
- **One run in the whole window.** A run redirects `sys.stdout` and `sys.stderr` for the whole
  process, so two at once would corrupt both. Guard: `BasePanel._running` is a class attribute, and
  the `_REDIRECT` lock in `core.gui.streams` declines a second redirect with a warning
  ([One run at a time](window.md#one-run-at-a-time)).
- **Save a figure with `visualizers.save_figure`, never `savefig`.** matplotlib bakes artist colours
  when a figure is built but reads the saved background when it is saved, and the window rewrites
  matplotlib's settings on every appearance change, so a long run that straddles one saves unreadable
  figures. Guard: `core.Helpers.visualizers.save_figure` pins the save to the figure's own colours.
  Not yet followed: the FDT figures (`core.FDT.plots`: `plot_eff_temp_ratio`,
  `plot_spontaneous_trajectory`, `plot_psd`, `plot_chi_components`; and
  `core.FDT.cross_validation_plots.plot_fdt_3d_vs_param`) and the reduction map's
  (`core.Reduction.plots.plot_sweep_summary`, `plot_cross_validation_3d`) still call `savefig`.
  Pinned by `tests/test_figures.py::test_a_theme_flip_between_build_and_save_cannot_split_a_figure`.
- **Give a table cell an explicit background, never `"none"`.** A transparent cell shows the figure's
  background behind text coloured for the axes, and once left the best-fit table blank. Guard:
  `visualizers._param_table` pins every cell to the axes surface. Pinned by
  `tests/test_figures.py::test_the_parameter_table_is_readable_against_its_own_cells`.
- **Drop and report non-finite draws before a quantile.** `torch.quantile` spreads one non-finite
  entry across the whole reduction and refuses an input over 2**24 elements; one divergent draw in a
  thousand once erased a predictive band with no message. Guard: `core.SBI.overlay.psd_band` drops
  non-finite rows and returns how many, `overlay.cycle_average` masks them before binning, and
  `overlay._column_quantile` works in chunks under the size limit. Report the count: it is a finding
  about the posterior. Pinned by
  `tests/test_nav_and_gating.py::test_a_divergent_draw_does_not_erase_the_whole_psd_band` and
  `tests/test_nav_and_gating.py::test_the_cycle_average_masks_non_finite_samples_instead_of_binning_them`.
- **Detach a pop-out figure from pyplot and import the Qt backend lazily.** Unpickling a figure
  registers it with pyplot again, where it leaks and `Worker.run`'s `plt.close("all")` tears it down
  under the viewer. Guard: `core.gui.widgets.figure_window.build_interactive_window` destroys what
  the unpickle registered; `InteractiveFigureWindow.__init__` imports the Qt canvas only when a
  figure pops out, so headless tests never pull it in (never call `plt.show()` or
  `matplotlib.use()` there); and `FigureStack._windows` holds each pop-out, or it is collected as
  soon as it is shown. Pinned by
  `tests/test_figures.py::test_pop_out_of_a_pickle_builds_a_qtagg_canvas_and_keeps_pyplot_clean`.
- **Inputs live in per-model folders.** A picker left on another model's folder offers the wrong
  files. Guard: `ArtifactPicker.repoint`, called whenever a picker's model changes; records are
  listed by the store, not by folder.

### Progress, the solver meter and cancellation

The design is in [The window](architecture.md#the-window); these are the rules it depends on.

- **Classify each chunk tqdm writes; never scan for line ends.** A line-based reader turned every
  redraw of a nested bar into a log line. Guard: `core.gui.vt.StreamRouter.feed`, which places a
  paint by the up-move that follows it, reads a bare `"\r"` as row 0, and leaves a bare `"\n"` to
  later evidence; `StreamRouter._settle_closing` resolves a `close(leave=True)`, which is otherwise
  byte-identical to a move down and leaves a finished bar pegged at 100 %.
- **Never give the redirected stream a `fileno()` or an `encoding`.** tqdm probes for both, and
  finding them changes the width and the glyphs of the frames `vt` parses. No guard: the invariant
  is stated on `core.gui.streams._SignalStream`.
- **The 15 Hz pump is load-bearing.** tqdm's `set_description` and `reset` redraw with no time gate,
  and the prior sweeps call both every iteration; unthrottled, the redraws flood the GUI's event
  queue. Guard: `streams._Pump` publishes at `PUMP_HZ`, and its final publish runs on its own thread
  (`_Pump.stop`), since from the GUI thread it would jump ahead of queued ticks and scramble the log.
- **Key rows by tqdm position, never by description.** The prior sweeps rewrite their description
  with a live counter, so a description-keyed map would mint a row per iteration. Guard:
  `StreamRouter._key`.
- **The overall bar is the live row with the largest total above 1.** The outermost bar wraps a
  single step, and the deepest, the segment bar, would sweep the overall bar from 0 to 100 % every
  few seconds. When nothing reports a percentage, the caption falls back to the deepest row whose
  total is not 1, sbi's epoch counter during training; without it the longest phase shows an
  indeterminate bar with no caption. Guard: `core.gui.widgets.progress_pane.ProgressPane._retarget`
  and `_paint_caption`. Pinned by
  `tests/test_vt_progress.py::test_the_overall_bar_follows_the_largest_non_degenerate_total` and
  `tests/test_vt_progress.py::test_the_caption_falls_back_to_the_sbi_epoch_counter`.
- **The solver meter reads the step counter, never the bar's text.** A solver call shorter than its
  bar's refresh interval never paints a rate, which CUDA graphs made the norm. Guard: the meter
  differences `core.progress.SOLVER`. The solver bar is found by its description
  (`config.SOLVER_BAR_DESC`) only to exclude it from the overall bar, whose election it would
  otherwise win; it is never rendered as a row, since every time segment of every simulation makes
  one; and it stays enabled, because the cancel checkpoint and the stall detector depend on its
  writes. `vt.parse_rate` still inverts tqdm's `s/it`, though nothing acts on the rate now. Pinned by
  `tests/test_vt_progress.py::test_the_solver_meter_reads_the_step_counter_not_a_rendered_bar` and
  `tests/test_vt_progress.py::test_the_solver_bar_never_drives_the_overall_bar_and_is_not_the_caption`.
- **Reset tqdm's lock after a cancel.** tqdm's `refresh()` releases its write lock by hand, so a
  cancel raised inside a redraw leaks it and the next bar deadlocks. Guard: `redirect_streams` calls
  `streams.reset_tqdm_lock` when the token fired. Pinned by
  `tests/test_worker_dispatch.py::test_cancel_token_latches_and_leaves_tqdm_usable`.
- **Raise the cancel only on the owner thread, and never swallow it.** tqdm's monitor thread also
  writes to the stream; consuming the token there would raise where nothing catches it and lose the
  cancel. Guard: `CancelToken.check` raises only on the thread `arm()` recorded. `WorkerCancelled`
  derives from `BaseException` so that it passes every `except Exception`; never catch it and carry
  on: a handler that must see a cancel catches `BaseException` and re-raises, as the cache commit in
  `pipeline.gen_training_data` does. Pinned by
  `tests/test_worker_dispatch.py::test_cancel_is_not_consumed_by_a_non_worker_thread` and
  `tests/test_worker_dispatch.py::test_worker_cancelled_passes_through_except_exception`.

### Settings persistence and the inference screen

What the window remembers, and why, is in
[What the window remembers](window.md#what-the-window-remembers); the tabs and their gates are in
[The Parameter Inference tabs](window.md#the-parameter-inference-tabs).

- **The restore order.** A picker's saved key re-applied before its folder is rescanned is wiped by
  the rescan. Guard: `ArtifactPicker.repoint` points, rescans, then re-applies. A panel restoring a
  model combo sets it first and calls its `_on_model_changed` explicitly, because
  `currentTextChanged` does not fire when the value is already the default
  (`FdtPanel.restore_settings`).
- **The sweep study's grid ends are never persisted.** A value saved against another cell would be a
  wrong bound. Guard: `CrossValPanel.save_settings` and `restore_settings`, which re-derive the ends
  from the cell. Pinned by
  `tests/test_settings_persistence.py::test_crossval_does_not_persist_cell_derived_bounds`.
- **A picker's key is its relative path, restored with a missing-item guard.** An index shifts as
  files are added, and restoring a vanished entry would blank the combo. Guard: `ArtifactPicker.key`
  and `ArtifactPicker.restore_key`. Pinned by
  `tests/test_settings_persistence.py::test_missing_picker_key_restores_to_default_not_blank`.
- **The gating table, TSNPE included.** Validate and TSNPE gate on the prior loaded on the Prior tab,
  never on the drive prior ([The Parameter Inference tabs](window.md#the-parameter-inference-tabs)
  says why). Guard: `InferenceScreen.refresh_gates`, re-run
  after every stage, and `TSNPEPanel.refresh_local_gates` for the TSNPE button's observation. Pinned
  by
  `tests/test_nav_and_gating.py::test_inference_tab_gates_follow_the_session` and
  `tests/test_nav_and_gating.py::test_tsnpe_tab_is_gated_and_never_proposes_from_the_posterior`.
- **A new draft replaces the session; installing a configuration edits it.** Mixing them up discards
  the session's prior, posterior and observation, or the draft in the middle of a stage.
  `InferenceScreen.new_draft` replaces the session, because another model or unit system invalidates
  every record it holds; `install_config` sets the configuration in place, because the Prior tab
  installs it as the first step of building the prior. Guard: the Config tab asks before a new draft
  releases anything ([The dialogs](window.md#the-dialogs)). Pinned by
  `tests/test_nav_and_gating.py::test_apply_confirms_before_it_discards_the_session`.
- **Pickers follow the built configuration's model.** A picker on the wrong model's folder offers
  cells or bounds for another parameter set. Guard: `InferPanel.on_config_built` repoints the cell
  picker, and `PriorPanel.on_draft_set` the bounds picker.
- **Hand entry edits numbers, never the schema.** Parameter names and order belong to the model and
  bind by position. Guard: the grids always load from the selected file
  (`core.gui.widgets.param_grid`) and are never persisted.
- **The control lock covers all six tabs.** A model changed on a sibling tab mid-run would repoint
  pickers and re-seed inputs under the running worker. Guard: `BasePanel._set_busy`, over
  `BasePanel._instances`.
- **`core/gui/panels/inference_tabs.py` is a re-export shim.** The six tabs live in
  `core/gui/panels/inference/`, one module per tab. Deleting the shim fails the suites, which import
  through it, but Settings, which reads its help text by string path, skips it without a word. No
  guard beyond those imports.
- **Read a numeric box through `value_or_none()`.** `FloatField.value()` and `IntField.value()`
  (`core.gui.widgets.labeled_inputs`) read a blank or half-typed box as 0, a legal value for several
  boxes; `value_or_none()` returns None, which the tab refuses as blank. No guard: each tab must
  choose the call.
- **Help text lives in the panel module's `HELP` dict.** Each badge takes it as `HELP[key]` through
  `core.gui.widgets.help_badge.add_help_row`, and Settings assembles its help reference from each
  section module's `HELP`, so text written only into a call shows as a badge and never reaches that
  reference. No guard: the panels follow the convention.

### Simulate and video export

The live simulation drives the solver itself (`core.gui.panels.simulate_runner`), outside the
simulator's own guards.

- **A frame of m steps needs m + 1 grid points, and emits all but the first.** The solver derives
  its step from the point count, so m points would change the step without a word; the first point
  repeats the previous frame's last. Guard: `simulate_runner.frame_time_grid`.
- **Wrap the loop in `torch.no_grad()`.** The solver has no autograd guard of its own; it lived in
  the simulator, which this path bypasses. Guard: `simulate_runner.run_simulation_stream`.
- **Cancel between frames.** A cancel raised inside a tqdm redraw would leak its write lock. Guard:
  `run_simulation_stream` polls `should_stop` once per frame and raises between frames.
- **Set the model's drive directly.** Assigning the simulator's `force` rebuilds the whole model.
  Guard: each frame sets `sim.sde.force` (`run_simulation_stream`).
- **Pin the CPU.** The batch-of-one loop is fastest there, and the direct drive assignment moves
  nothing to another device. Guard: `simulate_runner.build_stream_config`.
- **Export frames contiguous, with even dimensions.** A frame's RGB slice is a non-contiguous view,
  and H.264 needs even sides. Guard: `core.gui.panels.simulate_export.export_animation`, which also
  removes a cancelled or failed export's partial file in a `finally`, closing the writer under its
  own guard.
- **GIF durations are in milliseconds, and MP4 needs ffmpeg.** Guard: `simulate_export._open_writer`
  passes milliseconds, and `SimulatePanel._save_video` checks for ffmpeg on the GUI thread before it
  dispatches anything.
- **Import imageio after torch.** Imported first, it loads a second OpenMP runtime and the process
  aborts with OMP Error #15 and no traceback. Guard: `simulate_export._open_writer` imports it only
  when a writer opens.

### Labels and units

These are conventions, and no guard enforces them.

- **Time axes are in seconds**, converted where the data are made
  (`orchestrator.generate_observations` builds its time axis in seconds). Never relabel the t_scale
  parameter's axis in seconds: its value is in the cell's time unit per model unit
  (`core.Helpers.labels.rescale_axis_label`).
- **`SimConfig.inferred_labels` returns LaTeX**, which matplotlib renders; Qt cannot, so the window's
  own labels use `labels.pretty_gui`'s rich text, and a log line that prints a parameter's label
  shows the raw `$...$`.
- **In pyqtgraph, a displacement unit goes in the label text**, never in `units=`, which pyqtgraph
  prefixes as an SI unit and would mangle "nm"; time keeps `units="s"`
  (`core.gui.widgets.live_hair_bundle.LiveHairBundleView`).
- **Most figures are non-dimensional** and say "(ND)"; only the dimensional trace axes carry units.

### User-defined models

The rules themselves are in
[Defining a model of your own](recordings.md#defining-a-model-of-your-own); these are their guards.

- **Numbers are stripped before parameter names are found.** Otherwise the `e` of a number's
  exponent (`1e-3`) or the `x1F` of a hex literal becomes a dead parameter that shifts every
  positional binding after it. Guard:
  `core.Models.user_model._identifiers`.
- **`E` is an ordinary parameter; `pi` is the only constant.** Guard: `user_model._CONSTANTS`.
- **The model's parameter order must equal its bounds file's.** Otherwise every value binds to the
  wrong parameter with no error. Guard: `core.SBI.run_guards._assert_user_model_in_sync` and
  `simulate_runner.build_stream_config` refuse a drifted file.
- **A save commits with its JSON.** The registry knows a model by its JSON alone, so a save that
  failed part-way must never leave a JSON behind. Guard: `core.Helpers.model_store.save_user_model`
  writes the bounds, cell and units files first and the JSON last.
- **Re-fire a panel's model hook only on a real change.** The hooks reset the pickers, so firing
  after every save or delete would discard picker selections across the window. Guard:
  `MainWindow._refresh_model_combos`.
- **The builder refuses while a run is live.** Its short integration runs on the GUI thread and would
  write into the running task's redirected streams, and the control lock does not reach a screen that
  is not a panel. Guard: `ModelBuilderScreen._validate` and `_save` check `BasePanel._running`.

### The chi probe set and its Fisher

The chi block's design and measurements are in [Chi probe design](science.md#chi-probe-design) and
[Conditioning features](science.md#conditioning-features). `tests/test_chi_set_encoder.py` runs in
seconds with no simulation: run it first after touching anything chi.

- **Three feature sets, and conflating them is invisible.** The conditioning block, six channels in
  each padded slot (`core.SBI.chi.CHI_COND_CHANNELS`), is what the network reads. The Fisher set,
  three channels per probe and no pads (`chi.CHI_FISHER_CHANNELS`, built by `chi.fisher_features`),
  is what the Fisher rotation and the identifiability diagnostics differentiate. The diagnostics'
  feature rows in chi mode (`core.diagnostics.feature_sets.feature_labels`) are the summary features
  without Group G, plus the Fisher set. The frequency, cycle-count and mask channels stay out of the
  Fisher set. The frequency channel barely varies with the parameters and the mask steps, and in a
  standardised Jacobian either is an amplifier; the cycle count, below the ceiling, duplicates a
  summary feature exactly, so K probes weight that one direction K-fold, and at the ceiling it barely
  varies. Guard: `chi.fisher_features` takes one argument, so a mis-wired channel is a `TypeError`.
  Pinned by
  `tests/test_chi_set_encoder.py::test_fisher_features_takes_one_argument_and_its_channels_are_what_they_claim`
  and `tests/test_diagnostics.py::test_the_chi_feature_set_drops_group_g_and_adds_the_fisher_block`.
- **The Fisher needs `resolution_filter=False` and a seeded chi block.** The cycle-floor filter
  depends on each row's peak frequency, hence on the parameters, so a probe can cross it between the
  two arms of a central difference and put about 1e9 into the Jacobian; and the chi block runs its
  own simulations, so an unseeded one gives the two arms different noise. Guard:
  `core.SBI.decorrelate.build_latent_fisher_rotation` and the Jacobian in
  `core.diagnostics.identifiability` pass `resolution_filter=False` and seed before the chi block;
  the lock-in ceiling stays on.
- **Never a per-column z-score under chi.** It breaks the probe set's permutation invariance and
  amplifies the mask column ([Conditioning features](science.md#conditioning-features)). Guard:
  `core.SBI.train.train_nn` derives sbi's `z_score_x` from `EmbeddedNet.owns_standardization`,
  never from a separate argument.
- **The mask gate on both sides of the element network.** A network with a bias maps a zero slot to
  something non-zero, so a gate before it alone lets dead slots in, and the embedding drifts with the
  pad width. Guard: `core.SBI.chi_encoder.ChiSetEncoder.pool` multiplies by the mask before and after
  `phi`. Pinned by `tests/test_chi_set_encoder.py::test_post_gate_is_present`.
- **No max-pool and no BatchNorm.** The expected maximum of n draws grows with n, so a masked max
  shifts every channel with the probe count; BatchNorm breaks because sbi runs the network on one CPU
  row when it builds the flow, and a single observation must match its row of a batch. Guard:
  `ChiSetEncoder`. Pinned by
  `tests/test_chi_set_encoder.py::test_no_batchnorm_and_single_row_matches_batch`.
- **Width cannot identify the layout.** Six channels in five slots and three in ten are both 30
  columns, an exact collision with the retired layout, so a width check alone would pass a posterior
  of the wrong layout. Guard: `ArtifactStore.load_posterior` checks the manifest's `chi_layout`, the
  slot count and the channel width before it compares the conditioning width.
- **The slot capacity is frozen into every record.** sbi bakes the conditioning shape into a saved
  posterior, so raising `CHI_K_PAD` means retraining, and an unchecked change would surface as a
  shape error hours into a run; it costs only input columns, so choose it generously once. Guard: it
  is recorded with every chi posterior and observation, the loaders refuse another, and the
  simulation cache's identity carries it.
- **The lock-in ceiling is per row, inside the probe loop.** Past about 30 drive cycles a longer
  lock-in measures worse, and the cap defines the measurement, so training, the Fisher, the
  predictive check and the experimental path must all apply it the same way. A single duration for a
  whole batch is a regression: keyed on the fastest row, it once left about 48 % of rows with no live
  probe. Faced with a chi failure, ask for the cycle count before the frequency. Guard:
  `core.SBI.chi_probes.gen_chi_raw` caps each row from its own probe frequency, the experimental path
  truncates a longer recording to the same cap (`chi.probe_verdict`), and `SimConfig.__post_init__`
  refuses a ceiling at or below the cycle floor. Pinned by
  `tests/test_user_sbi.py::test_lock_in_per_row_durations_match_locking_each_row_alone`,
  `tests/test_user_sbi.py::test_lock_in_duration_is_capped_at_chi_max_cycles` and
  `tests/test_user_sbi.py::test_chi_max_cycles_must_clear_the_min_cycles_floor`.
- **Hertz go through `SimConfig.freq_si_to_cell`.** Matching a declared "Hz" token instead is a
  thousand-fold error on a millisecond cell that lands as a valid χ far off resonance
  ([Nondimensionalisation](science.md#nondimensionalisation) says why). Guard: `chi.probe_verdict` and the bench-recording builders
  (`core.SBI.observations.build_experiment_obs`, `build_experiment_obs_chi`) convert through it, and
  `SimConfig.check_unit_consistency` warns when the units file's frequency token is not the
  reciprocal of its time token.
- **Never clip the chi block.** A clip would move pads off 0.0 into phantom probes
  ([Conditioning features](science.md#conditioning-features)). Guard:
  `core.SBI.summaries.winsorize_summary_block` takes the summary width and touches nothing past it.
  Pinned by
  `tests/test_conditioning_repair.py::test_winsorisation_leaves_the_chi_block_BITWISE_untouched`.
- **A new statistics channel reaches the Fisher too.** A channel added to `summaries.gen_stats`
  reaches every caller; decide for each whether it builds a conditioning row, which is joined to
  log T_obs and wants the channel, or a Jacobian, which does not. Guard:
  `summaries.gen_stats_features`, the one name for the features without the valid flags, which every
  Jacobian calls (`decorrelate.build_latent_fisher_rotation`, `core.diagnostics.identifiability`).
- **A near-constant channel inflates a standardised Jacobian.** The Fisher divides each feature's
  difference by its ensemble spread, floored at 1e-9 (`decorrelate.build_latent_fisher_rotation`):
  an exactly constant channel is harmless, 0 over 1e-9 being 0, and a nearly constant one is an
  amplifier. A capped probe's cycle channel once reached a Jacobian entry of 2.0e4 against 289 for
  the largest real feature. No guard: apply that test to any channel added to any Fisher here.
- **Binary flags stay out of the Fisher.** A valid flag is constant almost everywhere and then steps
  by 1 between the two arms at some operating point, which the 1e-9 floor turns into about 1e9.
  Guard: the Fisher's three simulation paths call `gen_stats_features`, which drops the flags; the
  chi mask is left out of the Fisher set for the same reason.
- **After changing a constant or a feature definition that no loader compares, retrain against a
  new simulation cache, and reuse neither the old cache nor the posteriors and observations made
  before the change.** The cycle floor comes first: it is outside the cache's identity and every
  record ([The artifact store](architecture.md#the-artifact-store)), so a plain retrain finds the
  old cache under the same identity and reuses rows simulated at the old floor, while an old
  posterior, or a chi observation whose probes were masked at the old floor, loads beside the new
  configuration without a word. The prior is unaffected: its stability screen does not read the
  floor. Three routes:
  - Build a new prior. A new fit changes `prior_fingerprint`, so the cache is keyed afresh. When the
    prior is the only difference, the first training run is refused as one setting away from the old
    cache and goes ahead on consent (`--new-run`, or the Posterior tab's question).
    [The retrain runbook](retrain.md#decisions) takes this route.
  - Remove the old cache. `--resume never` refuses rather than start over it
    ([Training-cache flags](command-line.md#training-cache-flags)), and
    [artifacts rm](command-line.md#artifacts-rm) refuses while any record names it as a parent, so
    delete from the leaves up: the calibrations, inferences, diagnostics and narrowing rounds built
    on each posterior trained from the cache, then those posteriors, then
    `artifacts rm simulation <ref>`. Each refusal lists the records that still name it as a parent.
  - Keep the old records, and train with `train --checkpoint-every 0`, which reads and writes no
    simulation cache. The run then cannot be resumed, `--resume require` and `--resume never` are
    refused beside it, and the posterior names no cache as a parent. The window has no control for
    the checkpoint cadence. The old cache stays on disk: a later run at the default cadence with the
    same identity finds it and reuses its rows, so remove it, or build a new prior, before that run.

  The window offers the first two routes, through the Prior tab and a delete on the Artifacts screen.

  A feature change must also bump `core.SBI.statistics.FEATURE_SET_VERSION`, which re-keys every
  cache but is not compared when a posterior or an observation loads. No guard stands here yet:
  [The artifact store](architecture.md#the-artifact-store) lists every key a record carries that no
  loader compares.

### Memory and devices

The planner, the ladders and the recovery order are in
[Memory planning and out-of-memory recovery](architecture.md#memory-planning-and-out-of-memory-recovery).

- **`torch.cuda.mem_get_info()` overstates free memory on Windows.** Under WDDM it counts other
  processes' evictable surfaces as free, so a batch sized from it plans against memory it does not
  have and fails as a raw driver `AcceleratorError`. Read `nvidia-smi` when measuring by hand.
  Guard: `config.memory_budget_elements` is only a hint, `pipeline._BUDGET_CAP_ELEMENTS` learns the
  real ceiling from failures, and `pipeline._is_oom` accepts the raw error.
- **An unwrapped out-of-memory error rules out the solver.** Guard: `Simulator.__sols` wraps what
  the solver raises in a `SimulationError` that names the model, the method and the step count, so a
  bare `CUDA error: out of memory` came from the batch's own tensors: read the prefix before
  theorising. sbi's `append_simulations` is outside every retry, which is why `train.train_nn` keeps
  the training data on the CPU and caps sbi's standardisation check
  (`train._capped_zscore_check`).
- **A recovery path logs before it releases and never raises; the batch-level retry restores
  first.** A release that raised inside a recovery once hid the out-of-memory error it was
  recovering from, and a release before a wait once gave away the memory the restore then needed.
  Guard: every ladder logs its notice before it releases, and releases through
  `pipeline._release_device_memory`, which never raises. The batch-level retry in
  `pipeline.gen_training_data` also restores the random streams first (`pipeline._try_rng_restore`),
  then releases, then waits. A diagnostic inside a recovery is best-effort (`pipeline._log_memory`).
  Pinned by `tests/test_user_sbi.py::test_the_oom_notice_is_printed_before_the_release`,
  `tests/test_user_sbi.py::test_the_release_path_survives_a_failing_empty_cache` and
  `tests/test_user_sbi.py::test_the_rng_restore_happens_before_the_release_and_the_wait`.
- **Never wait on memory you hold yourself.** A run once waited indefinitely for memory its own
  process held. Guard: `pipeline._we_are_the_holder`, on which the batch-level retry goes straight
  back to the halving ladders and `pipeline.retry_on_oom` retries at once, instead of waiting. Pinned
  by `tests/test_user_sbi.py::test_the_retry_does_not_wait_when_THIS_process_holds_the_card` and
  `tests/test_user_sbi.py::test_retry_on_oom_notices_releases_and_honours_the_holder_gate`.
- **The allocator variable is `PYTORCH_CUDA_ALLOC_CONF`.** On this build (torch 2.9.0) the name
  torch's own deprecation warning recommends is ignored without a word. Guard: `core.config` sets
  this one with `setdefault` at import, so an operator's export wins. Test any allocator experiment
  with a deliberately invalid value first.

### Randomness and reproducibility

What a seed buys, and on which devices, is in
[The solver, CUDA graphs and reproducibility](architecture.md#the-solver-cuda-graphs-and-reproducibility).

- **Resolve the solver at call time.** A module-level instance would be built at import, and a test
  that patches the class would silently stop applying. Guard: `Simulator.__sols` builds
  `sdeint.Solver()` on every call. Pinned by
  `tests/test_user_sbi.py::test_solver_failure_raises_instead_of_killing_the_process`.
- **`SimConfig.chi_mode` is a plain `False`.** The other chi fields read `core/config.py` when a
  configuration is built; `chi_mode` does not, so a `SimConfig` built by hand is never in chi mode.
  No guard: build configurations through `core.cli.make_sim_config`, the one bridge from
  `config.CHI_MODE`.
- **Seed immediately before the chi block.** A common-random-number scheme that seeds only before
  the passive run gives its two arms different probe noise, and the result looks plausible and means
  nothing. Guard: the Fisher rotation and the identifiability Jacobian seed before the chi block, and
  training's probe draw uses a dedicated generator (`chi_gen` in `pipeline.gen_training_data`,
  seeded with a fixed value on every call), never the global stream those seeds would disturb.
- **numpy draws training's initial states.** `pipeline._training_inits` draws the built-in models'
  initial states with `np.random.randint`, which `torch.manual_seed` does not touch, so a claim that
  a seeded run reproduces needs both seeded. Guard: `core.rng.seeded` seeds both, and a resume reads
  the initial states from the cache's header rather than drawing them again. Pinned by
  `tests/test_user_sbi.py::test_gen_training_data_is_reproducible_from_a_seed_in_every_mode`.
- **Warm TorchScript up, three runs, before a bitwise comparison.** On CUDA a scripted step's first
  runs are not bitwise-reproducible against later ones, so an unwarmed comparison measures the
  compiler, not the change: ask whether the baseline reproduces against itself before concluding a
  change broke it.
  No guard in the code; the warm-up is shown in
  `tests/test_user_sbi.py::test_the_cuda_graph_step_matches_the_eager_step_bitwise`.
- **The rotation is not reproducible across processes, so a resume reuses the stored one.** The
  Fisher draws its operating points from the caller's unseeded global stream, so a restarted process
  computes another rotation, and resumed rows would sit in a different coordinate from their stored
  targets. Guard: `orchestrator.build_posterior` resolves the cache before the rotation and reuses
  the rotation in its header; a narrowing round reuses its parent's
  ([The Fisher settings on a resumed run](window.md#the-fisher-settings-on-a-resumed-run)). The card
  drill is in [The card smoke gate](testing.md#the-card-smoke-gate).
- **`torch.save` of a view writes the whole storage.** A row slice of a preallocated buffer is
  already contiguous, so `.contiguous()` does not help; without `.clone()` every cache commit would
  write the whole multi-GiB buffer, which shows only as slow checkpointing. Guard:
  `core.SBI.training_checkpoint.save` clones each slice. Pinned by
  `tests/test_user_sbi.py::test_checkpoint_shards_do_not_serialize_the_whole_accumulator`.

### Rotations and narrowing rounds

The narrowing-round rules themselves are in the table above.

- **The rotation holds its eigenvectors in columns, and only `reparam.rotation_of` decodes it.** The
  rotated latent prior samples `w = z @ V` and the training transform stores the rotation
  transposed. A second reader once saved the rotation transposed, and every rotated posterior the
  window saved reloaded through the wrong bijection, with nothing to say so. Guard:
  `core.SBI.reparam.rotation_of` is the one place the convention is read, and
  `ArtifactStore.load_posterior` checks a posterior's rotation against the one pickled inside it
  (`reparam.assert_rotation_consistent`). An identity matrix is blind to a transpose, so an
  orientation test needs a non-symmetric rotation. Pinned by
  `tests/test_conditioning_repair.py::test_reparam_is_the_only_reader_of_the_rotation_matrix` and
  `tests/test_artifact_store.py::test_manifest_V_must_equal_the_rotation_in_the_pickled_prior`.
- **Calibrate on the truncated prior.** Calibrated against the full prior, a narrowed posterior
  reports a miscalibration that is not one. Guard: `validate_calibration` takes no region and reads
  it off the loaded posterior (`orchestrator._calibration_prior`), so calibrate the narrowed
  posterior itself, loaded with its acceptance.
- **The region belongs to the simulation cache's identity.** Otherwise a round at its parent's
  budget would resume, or reuse whole, the amortized run's rows while saying it was restricted.
  Guard: `SimulationIdentity` carries the region (None for an amortized run), so a round has a cache
  of its own; `training_checkpoint.near_miss_siblings` ignores a cache that differs in the region
  alone. The amortized identity's digest is pinned to a fixed value, and any change to the
  identity's fields re-keys every cache on disk. The TSNPE tab cannot say in advance whether a round
  will resume, since the region is drawn when the round starts. Pinned by
  `tests/test_user_sbi.py::test_a_truncated_round_routes_to_its_own_checkpoint_and_the_amortized_digest_is_untouched`
  and
  `tests/test_settings_persistence.py::test_the_tsnpe_tab_never_claims_it_will_resume_the_amortized_checkpoint`.
- **A direction loaded on t_scale turns the restriction into a reweighting.** Training overwrites
  every row's t_scale with its batch's after the draw, which carries rows out of a box along such a
  direction, and NPE does not correct a reweighting. Guard: `truncate.region_from_posterior` skips a
  direction whose t_scale loading exceeds `truncate.t_scale_loading_max` (1/√d), and the kept
  fraction is reported after the override as well as before it
  ([Narrowing rounds](science.md#narrowing-rounds)). Pinned by
  `tests/test_conditioning_repair.py::test_a_t_scale_loaded_direction_is_excluded_from_the_region`
  and
  `tests/test_user_sbi.py::test_the_reported_kept_fraction_is_measured_after_the_t_scale_override`.
- **torch's sigmoid inverse clamps at the box edge.** `SigmoidTransform`'s inverse clamps to just
  inside (0, 1) on torch 2.9, so the box round trip never gives an infinite latent target; do not
  write code or a bug report that assumes it can. Guard: that clamp is torch's, pinned by
  `tests/test_user_sbi.py::test_box_roundtrip_never_yields_a_nonfinite_latent_target`, so a torch
  upgrade that drops it fails a test instead of a training run.
- **A fitted posterior used as the next prior is tempering.** Fit a density to one round's posterior
  and use it as the next round's prior at the same observation, and every round multiplies the
  likelihood in again: intervals shrink with no new information, and SBC, which checks the flow
  against its own proposal, stays flat. Guard: `truncate.TruncatedLatentPrior`, which restricts the
  prior and never reshapes it (the truncated-prior rule above).

### Module layout

- **The façade re-imports are load-bearing.** A consumer that bypasses `core.SBI.pipeline`'s
  re-imports fails silently, as the patch stops landing. No guard:
  [Two notes](architecture.md#two-notes) gives the rule and the other modules that follow it.
- **`core/config.py`'s constants are snapshotted at import**, so assigning `config.NAME` later is a
  silent no-op for a module that imported it by name, and moving a constant can change which value
  a caller sees ([Stages and compositions](architecture.md#stages-and-compositions) has the
  mechanism). No guard; the three fixes in use are not interchangeable: read through the module at
  the moment of use, for a process-wide
  switch (`config.QUIET_SEGMENT_BAR`, `config.SIM_VRAM_CEILING_GIB`); carry it on the configuration,
  for anything a trained record must describe (the chi fields, `reparam_rotate`); or take it as an
  argument that defaults to the constant, for a per-run setting (the stages' arguments).

### Reading diagnostics

How to read each test is in
[Reading calibration honestly](science.md#reading-calibration-honestly). No guard enforces these
readings; the calibration verdict's caveat line states the third on every run.

- **A flat SBC is not informativeness**: a posterior that returns the prior is flat by
  construction.
- **The best-fit table is not recovery**: its score says a draw sits inside the predictive cloud,
  not that it matches the truth.
- **The calibration's operating points are t_scale's effective sample size**, so lowering their
  count buys a different, weaker calibration, not the same one faster
  ([Settings that are not speed dials](window.md#settings-that-are-not-speed-dials)).
- **A pooled SBC over mixed probe counts can hide a miscalibration** that differs by count
  ([sbc](command-line.md#sbc) holds the count).
