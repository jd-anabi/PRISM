# Rules and traps

Checked against commit 26ac121.

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
- Construction is positional: each simulator's `_set_up_model` calls
  `Model(*torch.unbind(params, dim=1), force, ...)`, so the parameter order in the bounds file must
  equal the constructor's argument order. A failed construction raises
  `core.Simulator.simulator.SimulationError`, chained to its cause.
- The observable is state column 0: every simulation that feeds a statistic asks
  `core.SBI.pipeline.gen_obs` for `var_idx=0`, and a user model's first variable is its observable.
- The solver is Itô Euler–Maruyama and derives its step from the time grid; nothing passes it one
  ([The solver, CUDA graphs and reproducibility](architecture.md#the-solver-cuda-graphs-and-reproducibility)).

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

**Every setting is an argument, never a configuration edit.** `core.orchestrator` imports its
defaults by name from `core/config.py`, binding each one when it is first imported, so assigning a
`core.config` constant afterwards changes nothing the stage reads and the run uses the default
without a word
([Stages and compositions](architecture.md#stages-and-compositions)). The front ends pass every
setting to the stage, some by keyword and some by position; the one exception is the window's memory
ceiling, which the batch planner reads through the module every time it plans
([The inference settings](window.md#the-inference-settings)). Pinned by
`tests/test_user_sbi.py::test_build_posterior_takes_the_budget_as_arguments_because_the_constants_are_snapshotted`
and
`tests/test_settings_persistence.py::test_the_training_budget_reaches_build_posterior_as_arguments_not_via_config`.

**Every public entry works on a private copy of its configuration.** `core.runs.public_entry` hands
every public stage, composition and diagnostic, and the three FDT entries, `cfg.copy_for_run()`, so
the caller's configuration is unchanged whether the run succeeds, is refused or crashes. No stage
mutates what it is handed: a caller that wants what a stage produced reads the record, or the
`Loaded*` wrapper the stage returns. Pinned by
`tests/test_artifact_store.py::test_every_public_entry_leaves_the_callers_config_untouched` and
`tests/test_artifact_store.py::test_the_public_entries_carry_public_entry_and_nothing_else_does`.

**Every pre-spend refusal names its setting by a field key.** A bad input is refused before anything
is spent, as a `core.refusals.Refusal` whose `field`, when one setting is at fault, is a key
registered in `core.refusals.FIELDS`. The core's message names the setting in words and never a box,
tab, flag or button; each front end appends the control or flag that answers the key from its own
table (`core/gui/fields.py`, and `FLAG` in `core/tool/fields.py`). A programming error is a plain
exception, never a refusal. The known gaps, which are not the rule: the `ablation` diagnostic's
refusals about its simulation cache (`core.diagnostics.ablation.channel_ablation`) and the
command-line tool's refusal of an unsupported model (`core.tool.config_args.make_cfg`) are plain
`ValueError`s; the loaders' error for a reference that resolves to no record
(`core.artifacts.store.ArtifactStore.load_prior`, `load_posterior`, `load_observation`) carries no
field; and a
units-file refusal carries the field `units`, which no flag answers. How the tool prints each is in
[Exit codes](command-line.md#exit-codes). Pinned by
`tests/test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered`,
`tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it`
and `tests/test_refusals.py::test_every_domain_error_is_a_refusal_and_carries_a_field`.

**No chi override for anything that trains or infers.** The chi band, the chi drive amplitude and
the cycle floor below which a probe is masked come only from `core/config.py` (`CHI_FREQ_BOUNDS`,
`CHI_F0`, `CHI_MIN_CYCLES`); no front end sets them for a run.

- `core.SBI.run_guards._assert_chi_config_is_deliberate` refuses a chi configuration whose band or
  drive differs from `core/config.py`'s, with no override: `build_prior` and `build_posterior` run it
  before the first simulation, `probe_mask` before its audit, and `ArtifactStore.load_posterior`
  before anything else. A different band or drive means editing `core/config.py` deliberately. The
  probe count is not checked, on purpose: it is what one observation supplies, and training draws its
  own for every batch.
- The window's Config tab also has boxes for the lock-in ceiling and the number of probe slots
  ([The inference settings](window.md#the-inference-settings)).
- `ArtifactStore.load_posterior` refuses a chi posterior whose band, drive, lock-in ceiling or slot
  count differs from the loading configuration's.
- The cycle floor is neither recorded with a posterior nor part of the simulation cache's identity
  (`core.artifacts.identity.SimulationIdentity`), so a change to it is not caught at load, and a
  plain retrain can reuse a cache simulated at the old floor
  ([The chi probe set and its Fisher](#the-chi-probe-set-and-its-fisher)).
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

**The comment policy.** A comment states its reason in words. It cites functions, not line numbers,
and no planning or working document, nor any label one of them coined:
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
| **The cost-on-screen rule** | a round, a full simulation campaign, started without its cost in view | the TSNPE tab shows the Posterior tab's budget group (`core.gui.panels.inference.base._TrainingBudgetMixin`); on the command line `build_posterior` logs its `[budget]` line on every training run | `tests/test_artifact_store.py::test_the_budget_line_prints_with_default_arguments`; `tests/test_settings_persistence.py::test_the_training_budget_shows_the_simulation_count_and_the_cap_trade` |
| **The region-carries-its-basis rule:** the child reuses the parent's rotation, never recomputes it, skips directions loaded on t_scale, and names its base prior | a region drawn in one rotation and enforced in another: one such round kept 0.01 % of its parent posterior and lost the truth | `build_truncation_region` records the parent's rotation, a probe of its training bijection, the observation's digest and the parent's prior fingerprint; `build_posterior` trains in that rotation without running the Fisher and refuses a basis mismatch (`TruncationRegion.check_basis`), a rotation setting that disagrees, a resumed cache from another region or rotation (`TruncationRegion.check_checkpoint_V`) and another prior (`run_guards._assert_prior_matches_region`); `region_from_posterior` skips a direction loaded on t_scale above `truncate.t_scale_loading_max` | `tests/test_conditioning_repair.py::test_a_region_measured_in_one_basis_is_refused_in_a_sign_flipped_one`; `tests/test_conditioning_repair.py::test_build_truncation_region_records_the_parents_basis`; `tests/test_conditioning_repair.py::test_a_truncated_round_refuses_a_prior_other_than_the_parents`; `tests/test_conditioning_repair.py::test_a_t_scale_loaded_direction_is_excluded_from_the_region`; `tests/test_user_sbi.py::test_a_tsnpe_round_reuses_the_parents_basis_and_refuses_every_mismatch`; `tests/test_user_sbi.py::test_a_truncated_round_routes_to_its_own_checkpoint_and_the_amortized_digest_is_untouched` |
| **The calibrate-on-the-region rule** | a calibration drawn from the full prior, which reports a miscalibration that is not one | `orchestrator._calibration_prior`, shared by `validate_calibration` and `core.diagnostics.sbc_repeats`, restricts the rotated calibration prior to the region after checking its basis; `orchestrator._sbc_reference_sample` draws the rank test's reference from that same proposal, t_scale override included | `tests/test_user_sbi.py::test_calibration_theta_star_lies_inside_the_region_when_one_is_given`; `tests/test_user_sbi.py::test_the_reported_kept_fraction_is_measured_after_the_t_scale_override` |

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

Each trap is one rule, the silent failure it prevents, and where the guard lives. A trap with no
guard says so.

### The window: threads, figures and paths

How `core/gui/` runs a stage off the GUI thread is in [The window](architecture.md#the-window).

- **Keep a worker's reference until it finishes.** `BasePanel.dispatch`
  (`core.gui.panels.base_panel`) turns auto-delete off and keeps each `Worker` in `_workers` until
  its `finished` signal. A worker held only as a local is collected when `run()` returns, Qt drops
  its queued result and `finished` signals, and the panel stays busy for ever. Never start a raw
  `Worker` outside `dispatch`.
- **Never paint a figure built on a worker.** Painting a worker-built pyplot figure in a live canvas
  deadlocks on matplotlib's global lock, with no traceback. `base_panel._png_fig_sink` renders each
  figure to PNG on the worker; the GUI thread shows only images.
- **Force the Agg backend first.** `core.gui.__main__` and `core.__main__` call
  `matplotlib.use("Agg")` before any `core` import, which makes a stray `plt.show()` in a stage a
  no-op on a worker thread. Do not move either line.
- **One run in the whole window.** A run redirects `sys.stdout` and `sys.stderr` for the whole
  process, so `BasePanel._running` is a class attribute and every panel's controls lock while a run
  is live ([One run at a time](window.md#one-run-at-a-time)). The `_REDIRECT` lock in
  `core.gui.streams` is the backstop: a second redirect declines and warns rather than corrupt the
  streams.
- **Save a figure with `visualizers.save_figure`, never `savefig`.** matplotlib bakes artist colours
  when a figure is built but reads the saved background when it is saved, and the window rewrites
  matplotlib's settings on every appearance change, so a multi-day run that straddles one saves
  unreadable figures. `core.Helpers.visualizers.save_figure` pins the save to the figure's own
  colours. Not yet followed: the FDT figures (`core.FDT.plots`: `plot_eff_temp_ratio`,
  `plot_spontaneous_trajectory`, `plot_psd`, `plot_chi_components`; and
  `core.FDT.cross_validation_plots.plot_fdt_3d_vs_param`) and the reduction map's
  (`core.Reduction.plots.plot_sweep_summary`, `plot_cross_validation_3d`) still call `savefig`.
  Pinned by `tests/test_figures.py::test_a_theme_flip_between_build_and_save_cannot_split_a_figure`.
- **Drop and report non-finite draws before a quantile.** `torch.quantile` spreads one non-finite
  entry across the whole reduction and refuses an input over 2**24 elements; one divergent draw in a
  thousand once erased a predictive band with no message. `core.SBI.overlay.psd_band` drops
  non-finite rows and returns how many, `overlay.cycle_average` masks non-finite samples before
  binning, and `overlay._column_quantile` works in chunks under the size limit. Report the count: it
  is a finding about the posterior. Pinned by
  `tests/test_nav_and_gating.py::test_a_divergent_draw_does_not_erase_the_whole_psd_band` and
  `tests/test_nav_and_gating.py::test_the_cycle_average_masks_non_finite_samples_instead_of_binning_them`.
- **Detach a pop-out figure from pyplot and import the Qt backend lazily.** Unpickling a figure
  registers it with pyplot again; left there it leaks, and `Worker.run`'s `plt.close("all")` tears it
  down under the viewer. `core.gui.widgets.figure_window.build_interactive_window` destroys whatever
  the unpickle registered. `InteractiveFigureWindow.__init__` imports the Qt canvas only when a
  figure pops out, so app start and headless tests never pull it in; never call `plt.show()` or
  `matplotlib.use()` there. `FigureStack` holds each pop-out window (`_windows`), or it is collected
  as soon as it is shown. Pinned by
  `tests/test_figures.py::test_pop_out_of_a_pickle_builds_a_qtagg_canvas_and_keeps_pyplot_clean`.
- **Inputs live in per-model folders.** `Resources/Bounds`, `Cells` and `Units` each hold one folder
  per model, so an input picker is repointed whenever its model changes (`ArtifactPicker.repoint`);
  records are listed by the store, not by folder.

### Progress, the solver meter and cancellation

The design is in [The window](architecture.md#the-window); these are the rules it depends on.

- **Classify each chunk tqdm writes; never scan for line ends.** tqdm redraws a nested bar as three
  writes, the last with no terminator, and a line-based reader turned every redraw into a log line.
  `core.gui.vt.StreamRouter.feed` classifies each chunk: a paint sits at row n exactly when the chunk
  after it is an up-move of n. A bare `"\n"` can end a status line, end a `print()` or move the
  cursor; a bare `"\r"` is always row 0; and `close(leave=True)` at row 0 is byte-identical to a move
  down, which `StreamRouter._settle_closing` resolves later, or the finished bar stays pegged at
  100 %.
- **Never give the redirected stream a `fileno()` or an `encoding`.** tqdm probes for both
  (`core.gui.streams._SignalStream`); found, they change the width and the glyphs of the frames `vt`
  parses.
- **The 15 Hz pump is load-bearing.** tqdm's `set_description` and `reset` redraw with no time gate,
  and the prior sweeps call both every iteration. `streams._Pump` publishes at `PUMP_HZ` from its own
  thread, and its final publish runs on that thread (`_Pump.stop`): published from the GUI thread, it
  would run ahead of ticks still queued and scramble the log.
- **Key rows by tqdm position, never by description.** The prior sweeps rewrite their description
  with a live counter, and a description-keyed map minted a row per iteration
  (`StreamRouter._key`).
- **The overall bar is the live row with the largest total above 1.** Not the outermost, which wraps
  a single step, and not the deepest: the segment bar, deeper and far coarser, would sweep it from 0
  to 100 % every few seconds (`core.gui.widgets.progress_pane.ProgressPane._retarget`). When nothing reports a
  percentage the caption falls back to the deepest row whose total is not 1, which during sbi's
  training is its epoch counter: hours of a long build (`ProgressPane._paint_caption`). Pinned by
  `tests/test_vt_progress.py::test_the_overall_bar_follows_the_largest_non_degenerate_total` and
  `tests/test_vt_progress.py::test_the_caption_falls_back_to_the_sbi_epoch_counter`.
- **The solver meter reads the step counter, never the bar's text.** A solver call shorter than its
  bar's refresh interval never paints a rate, which CUDA graphs made the norm. The meter differences
  `core.progress.SOLVER`. The solver bar is found by its description (`config.SOLVER_BAR_DESC`) only
  to exclude it from the overall bar, whose election it would otherwise win; it is never rendered as
  a row, since every time segment of every simulation makes one; and it stays enabled, because the
  cancel checkpoint and the stall detector depend on its writes. `vt.parse_rate` still
  inverts tqdm's `s/it`, though nothing acts on the rate now. Pinned by
  `tests/test_vt_progress.py::test_the_solver_meter_reads_the_step_counter_not_a_rendered_bar` and
  `tests/test_vt_progress.py::test_the_solver_bar_never_drives_the_overall_bar_and_is_not_the_caption`.
- **Reset tqdm's lock after a cancel.** tqdm's `refresh()` releases its write lock by hand, so a
  cancel raised inside a redraw leaks it and the next bar deadlocks; `redirect_streams` calls
  `streams.reset_tqdm_lock` when the token fired. Pinned by
  `tests/test_worker_dispatch.py::test_cancel_token_latches_and_leaves_tqdm_usable`.
- **Raise the cancel only on the owner thread.** tqdm's monitor thread also writes to the stream;
  consuming the token there would raise where nothing catches it and lose the cancel for the run.
  `CancelToken.check` raises only on the thread `arm()` recorded. `WorkerCancelled` derives from
  `BaseException` so it passes every `except Exception`; never widen one of those to
  `BaseException`. Pinned by
  `tests/test_worker_dispatch.py::test_cancel_is_not_consumed_by_a_non_worker_thread` and
  `tests/test_worker_dispatch.py::test_worker_cancelled_passes_through_except_exception`.

### Settings persistence and the inference screen

What the window remembers, and why, is in
[What the window remembers](window.md#what-the-window-remembers).

- **The restore order.** `ArtifactPicker.repoint` points a picker at its folder, rescans, and only
  then re-applies the saved key, because the rescan wipes a selection restored first. A panel that
  restores a model combo sets it first and calls its `_on_model_changed` explicitly, because
  `currentTextChanged` does not fire when the value is already the default (`FdtPanel.restore_settings`).
- **The sweep study's grid ends are never persisted.** They are re-derived from the cell; a value
  saved against another cell would be a wrong bound (`CrossValPanel.save_settings`,
  `restore_settings`). Pinned by
  `tests/test_settings_persistence.py::test_crossval_does_not_persist_cell_derived_bounds`.
- **A picker's key is its relative path, restored with a missing-item guard.** `ArtifactPicker.key`
  saves the entry's path, not its index, and `ArtifactPicker.restore_key` leaves the selection alone
  when the saved entry is gone, never blanking the combo. Pinned by
  `tests/test_settings_persistence.py::test_missing_picker_key_restores_to_default_not_blank`.
- **The gating table, TSNPE included.** `InferenceScreen.refresh_gates` re-runs after every stage.
  Validate and TSNPE need a posterior and the prior loaded on the Prior tab, never the drive prior,
  which is None for every bounds file without a Forcing section and once made Validate unreachable
  for all of them; the TSNPE button also needs an observation (`TSNPEPanel.refresh_local_gates`).
  When a gate greys the visible tab, the screen falls back to Config
  ([The Parameter Inference tabs](window.md#the-parameter-inference-tabs)). Pinned by
  `tests/test_nav_and_gating.py::test_inference_tab_gates_follow_the_session` and
  `tests/test_nav_and_gating.py::test_tsnpe_tab_is_gated_and_never_proposes_from_the_posterior`.
- **A new draft replaces the session; installing a configuration edits it.**
  `InferenceScreen.new_draft` replaces the whole session, because another model or unit system
  invalidates every record the session holds; `install_config` sets the configuration in place, because the Prior tab
  installs it as the first step of building the prior. Mixing them up discards the session's prior,
  posterior and observation, or the draft mid-stage. The Config tab asks before a new draft releases
  anything ([The dialogs](window.md#the-dialogs)). Pinned by
  `tests/test_nav_and_gating.py::test_apply_confirms_before_it_discards_the_session`.
- **Pickers follow the built configuration's model.** The Infer tab's cell picker follows the model
  the configuration was built for (`InferPanel.on_config_built`), and the Prior tab's bounds picker
  re-applies its saved key when a model is applied (`PriorPanel.on_draft_set`).
- **Hand entry edits numbers, never the schema.** The bounds and cell grids always load from the
  selected file (`core.gui.widgets.param_grid`), because parameter names and order belong to the
  model and bind by position; the grids are never persisted.
- **The control lock covers all six tabs.** A run in any panel locks every panel's controls
  (`BasePanel._set_busy`, over `BasePanel._instances`), so a model changed on a sibling tab mid-run
  cannot repoint pickers and re-seed inputs under the running worker.
- **`core/gui/panels/inference_tabs.py` is a re-export shim.** The six tabs live in
  `core/gui/panels/inference/`, one module per tab. The shim stays: Settings reads its help text by
  string path and skips it silently when it is missing, and the suites import through it.
- **Read a numeric box through `value_or_none()`.** `FloatField.value()` and `IntField.value()`
  (`core.gui.widgets.labeled_inputs`) read a blank or half-typed box as 0, and 0 is a legal value
  for several boxes; `value_or_none()` returns None, which the tab refuses as blank.
- **Help rows.** A setting's help badge is `add_help_row` with the panel's `HELP` text, which
  Settings also assembles into its help reference.

### Simulate and video export

The live simulation drives the solver itself (`core.gui.panels.simulate_runner`), outside the
simulator's own guards.

- **A frame of m steps needs m + 1 grid points, and emits all but the first.** The solver derives
  its step as the span over one less than the point count, so m points would change the step without
  a word (`simulate_runner.frame_time_grid`); the first point repeats the previous frame's last.
- **Wrap the loop in `torch.no_grad()`.** The solver has no autograd guard of its own; it lived in
  the simulator, which this path bypasses (`simulate_runner.run_simulation_stream`).
- **Cancel between frames.** `run_simulation_stream` polls `should_stop` once per frame and raises
  between frames, never inside a tqdm redraw, so no write lock leaks.
- **Set the model's drive directly.** Each frame sets `sim.sde.force`; assigning the simulator's
  `force` rebuilds the whole model.
- **Pin the CPU.** `simulate_runner.build_stream_config` sets the configuration to the CPU: the
  batch-of-one loop is fastest there, and the direct drive assignment moves nothing to another
  device.
- **Export frames contiguous, with even dimensions.** A frame's RGB slice is a non-contiguous view,
  and H.264 needs even sides: `core.gui.panels.simulate_export.export_animation` copies each frame
  contiguous and crops to even. A cancel rides the per-frame tqdm redraw, and a cancelled or failed
  export removes its partial file in a `finally`, closing the writer under its own guard.
- **GIF durations are in milliseconds**, and MP4 needs ffmpeg, which `SimulatePanel._save_video`
  checks on the GUI thread before it dispatches anything.
- **Import imageio after torch.** Imported first, it loads a second OpenMP runtime and the process
  aborts with OMP Error #15 and no traceback; `simulate_export` imports it only inside the writer.

### Labels and units

- **Time axes are in seconds**, converted where the data are made (`orchestrator.generate_observations`
  builds its time axis in seconds). Never relabel the t_scale parameter's axis in seconds: its value
  is in the cell's time unit per model unit (`core.Helpers.labels.rescale_axis_label`).
- **`SimConfig.inferred_labels` returns LaTeX**, which matplotlib renders; Qt cannot, so the window's
  own labels use `labels.pretty_gui`'s rich text, and a log line that prints a parameter's label
  shows the raw `$...$`.
- **In pyqtgraph, a displacement unit goes in the label text**, never in `units=`, which pyqtgraph
  prefixes as an SI unit and would mangle "nm"; time keeps `units="s"`
  (`core.gui.widgets.live_hair_bundle.LiveHairBundleView`).
- **Most figures are non-dimensional** and say "(ND)"; only the dimensional trace axes carry units.

### User-defined models

The rules a lab member needs are in
[Defining a model of your own](recordings.md#defining-a-model-of-your-own); these are their guards.

- **Numbers are stripped before parameter names are found** (`core.Models.user_model._identifiers`);
  otherwise `1e-3` would add a parameter named `e`, a dead column that shifts every positional
  binding after it.
- **`E` is an ordinary parameter; `pi` is the only constant** (`user_model._CONSTANTS`).
- **The model's parameter order must equal its bounds file's**, or `torch.unbind` binds every value
  to the wrong parameter with no error: `core.SBI.run_guards._assert_user_model_in_sync` (the prior
  stage and `probes mask`) and `simulate_runner.build_stream_config` refuse a drifted file.
- **A save commits with its JSON.** `core.Helpers.model_store.save_user_model` refuses bad values and
  names, writes the bounds, cell and units files first and the JSON last, because the registry loads
  a model from its JSON alone.
- **Re-fire a panel's model hook only on a real change.** `MainWindow._refresh_model_combos`
  rebuilds every model list after a save or delete and calls `_on_model_changed` only when the
  selection changed; the hooks reset the pickers, so firing on every save would discard picker
  selections across the window.
- **The builder refuses while a run is live.** `ModelBuilderScreen._validate` and `_save` run a short
  integration on the GUI thread, whose progress bar would write into the running task's redirected
  streams; the screen is not a panel, so the control lock does not reach it, and `BasePanel._running`
  is checked instead.

### The chi probe set and its Fisher

The chi block's design and measurements are in [Chi probe design](science.md#chi-probe-design) and
[Conditioning features](science.md#conditioning-features). `tests/test_chi_set_encoder.py` runs in
seconds with no simulation: run it first after touching anything chi.

- **Three feature sets, and conflating them is invisible.** The conditioning block, six channels in
  each padded slot (`core.SBI.chi.CHI_COND_CHANNELS`), is what the network reads. The Fisher set,
  three channels per probe and no pads (`chi.CHI_FISHER_CHANNELS`, built by `chi.fisher_features`),
  is what the Fisher rotation and the identifiability diagnostics differentiate. The diagnostics'
  feature rows in chi mode (`core.diagnostics.feature_sets.feature_labels`) are the summary features
  without Group G, plus the Fisher set. The frequency, cycle-count and mask channels are left out of
  the Fisher set: each either barely varies with the parameters or steps, and in a standardised
  Jacobian either is an amplifier (see the near-constant channel below). `chi.fisher_features` takes
  one argument, so a mis-wired channel is a `TypeError`. Pinned by
  `tests/test_chi_set_encoder.py::test_fisher_features_takes_one_argument_and_its_channels_are_what_they_claim`
  and `tests/test_diagnostics.py::test_the_chi_feature_set_drops_group_g_and_adds_the_fisher_block`.
- **The Fisher needs `resolution_filter=False` and a seeded chi block.** The cycle-floor filter
  depends on each row's peak frequency, hence on the parameters, so a probe can cross it between the
  two arms of a central difference; a mask step over a 1e-9 floor puts about 1e9 into the Jacobian,
  and the rotation becomes that discontinuity. And the chi block runs its own simulations, so a
  common-random-number scheme must seed immediately before it too, or the two arms see different
  noise and the derivative means nothing. `core.SBI.decorrelate.build_latent_fisher_rotation` and the
  Jacobian in `core.diagnostics.identifiability` do both; the lock-in ceiling stays on.
- **Never a per-column z-score under chi.** sbi's default standardisation fits one affine map per
  column, which breaks the probe set's permutation invariance and turns the nearly constant mask
  column into an amplifier of about 1e7. `core.SBI.train.train_nn` derives sbi's `z_score_x` from
  `EmbeddedNet.owns_standardization`, never from a separate argument, so the two cannot disagree.
- **The mask gate on both sides of the element network.** A network with a bias maps a zero slot to
  something non-zero, so a gate before it alone lets dead slots contribute, and the symptom is an
  embedding that drifts with the pad width. `core.SBI.chi_encoder.ChiSetEncoder.pool` multiplies by
  the mask before and after `phi`. Pinned by `tests/test_chi_set_encoder.py::test_post_gate_is_present`.
- **No max-pool and no BatchNorm.** The expected maximum of n draws grows with n, so a masked max
  writes a shift that depends on the probe count into every channel; BatchNorm breaks because sbi
  runs the network on one CPU row when it builds the flow, and a single observation must match its
  row of a batch (`ChiSetEncoder`). Pinned by
  `tests/test_chi_set_encoder.py::test_no_batchnorm_and_single_row_matches_batch`.
- **Width cannot identify the layout.** Six channels in five slots and three in ten are both 30
  columns, an exact collision with the retired layout. `ArtifactStore.load_posterior` checks the
  manifest's `chi_layout`, the slot count and the channel width before it compares the conditioning
  width.
- **The slot capacity is frozen into every record.** sbi bakes the conditioning shape into a saved
  posterior, so `CHI_K_PAD` is recorded with every chi posterior and observation, the loaders refuse
  another, and the simulation cache's identity carries it; raising it means retraining. It costs
  only input columns, since the encoder's size does not depend on it, so choose it generously once.
- **The lock-in ceiling is per row, inside the probe loop.** Past about 30 drive cycles a longer
  lock-in measures worse, so `CHI_MAX_CYCLES` caps it. `core.SBI.chi_probes.gen_chi_raw` applies the
  cap row by row, from each row's own probe frequency, because it defines the measurement: training,
  the Fisher and the predictive check all go through it, and the experimental path truncates a
  longer recording to the same cap (`chi.probe_verdict`). A single duration for a whole batch is a
  regression: keyed on the fastest row, it once left about 48 % of rows with no live probe. Faced
  with a chi failure, ask for the cycle count before the frequency. `SimConfig.__post_init__`
  refuses a ceiling at or below the cycle floor. Pinned by
  `tests/test_user_sbi.py::test_lock_in_per_row_durations_match_locking_each_row_alone`.
- **Hertz go through `SimConfig.freq_si_to_cell`.** Matching a declared "Hz" token gives 1.0
  against a millisecond cell, a thousand-fold error that lands as a valid χ far off resonance.
  `chi.probe_verdict` and the bench-recording builders (`core.SBI.observations.build_experiment_obs`,
  `build_experiment_obs_chi`) convert through it, and `SimConfig.check_unit_consistency` warns when
  the units file's frequency token is not the reciprocal of its time token.
- **Never clip the chi block.** A pad must stay exactly 0.0, and the magnitude and phase channels
  are dense over live probes, so their low percentile is not 0 and a clip would move every pad off
  0.0 into a phantom probe. `core.SBI.summaries.winsorize_summary_block` takes the summary width and
  touches nothing past it. Pinned by
  `tests/test_conditioning_repair.py::test_winsorisation_leaves_the_chi_block_BITWISE_untouched`.
- **A new statistics channel reaches the Fisher too.** A channel added to `summaries.gen_stats`
  reaches every caller, so decide for each whether it builds a conditioning row, which is joined to
  log T_obs and wants the channel, or a Jacobian, which does not. `summaries.gen_stats_features` is
  the one name for the features without the valid flags, and every Jacobian calls it
  (`decorrelate.build_latent_fisher_rotation`, `core.diagnostics.identifiability`).
- **A near-constant channel inflates a standardised Jacobian.** The Fisher divides each feature's
  difference by its ensemble spread, floored at 1e-9 (`decorrelate.build_latent_fisher_rotation`).
  An exactly constant channel is harmless, 0 over 1e-9 being 0; a nearly constant one is an
  amplifier: a capped probe's cycle channel once reached a Jacobian entry of 2.0e4 against 289 for
  the largest real feature. Apply that test to any channel added to any Fisher here.
- **Binary flags stay out of the Fisher.** A valid flag is constant almost everywhere and then steps
  by 1 between the two arms at some operating point, which the 1e-9 floor turns into about 1e9 in the
  Jacobian. The Fisher's three simulation paths call `gen_stats_features`, which drops the flags; the
  chi mask is left out of the Fisher set for the same reason.
- **After changing a constant or a feature definition that no loader compares, retrain against a
  new simulation cache and reuse no old record.** The cycle floor comes first: it is neither part of
  the cache's identity nor recorded with a posterior, so a plain retrain finds the old cache under
  the same identity and reuses rows simulated at the old floor, and an old posterior loads beside
  the new configuration without a word. `--resume never` refuses rather than start over a cache
  ([Training-cache flags](command-line.md#training-cache-flags)), so remove the old one first with
  [artifacts rm](command-line.md#artifacts-rm), which refuses while any record still names it as a
  parent. A feature change must also bump `core.SBI.statistics.FEATURE_SET_VERSION`, which re-keys
  every cache but is not compared when a posterior or an observation loads. No guard stands here yet:
  [The artifact store](architecture.md#the-artifact-store) lists every key a record carries that no
  loader compares.

### Memory and devices

The planner, the ladders and the recovery rules are in
[Memory planning and out-of-memory recovery](architecture.md#memory-planning-and-out-of-memory-recovery).

- **`torch.cuda.mem_get_info()` overstates free memory on Windows.** Under WDDM it counts other
  processes' evictable surfaces as free (measured 15,037 MiB against the 5,814 MiB `nvidia-smi`
  showed), so a batch sized from it plans against memory it does not have, and the failure arrives
  as a raw driver `AcceleratorError`, not `torch.OutOfMemoryError`. Read `nvidia-smi` when measuring
  by hand. `config.memory_budget_elements` is only a hint: `pipeline._BUDGET_CAP_ELEMENTS` learns the
  real ceiling from failures, and `pipeline._is_oom` accepts both error types.
- **An unwrapped out-of-memory error rules out the solver.** `Simulator.__sols` wraps everything the
  solver raises in a `SimulationError` that begins with the model, the method and the step count, so
  a bare `CUDA error: out of memory` came from the batch's own tensors, not the integration: read the
  prefix before theorising. sbi's `append_simulations` sits outside every retry, which is why
  `train.train_nn` keeps the training data on the CPU and caps sbi's standardisation check
  (`train._capped_zscore_check`).
- **A recovery path restores before it releases, and never raises.** A release that raised inside a
  recovery once killed a run with a traceback that said nothing about the out-of-memory error it was
  recovering from, and a release followed by a wait once gave away the memory the restore then
  needed. Every ladder logs its notice first, restores the random streams before releasing
  (`pipeline._try_rng_restore`), and releases through `pipeline._release_device_memory`, which never
  raises; a diagnostic inside a recovery is best-effort (`pipeline._log_memory`).
- **Never wait on memory you hold yourself.** A run once waited indefinitely at batch 93 for memory
  its own process held. `pipeline._we_are_the_holder` sends the batch-level retry straight back to
  the halving ladders when this process's reserved pool exceeds everyone else's usage.
- **The allocator variable is `PYTORCH_CUDA_ALLOC_CONF`.** On this build (torch 2.9.0) the name
  torch's own deprecation warning recommends is ignored without a word, while this one is parsed.
  `core.config` sets it with `setdefault` at import, so an operator's export wins. Test any allocator
  experiment with a deliberately invalid value first.

### Randomness and reproducibility

What a seed buys, and on which devices, is in
[The solver, CUDA graphs and reproducibility](architecture.md#the-solver-cuda-graphs-and-reproducibility).

- **Resolve the solver at call time.** `Simulator.__sols` builds `sdeint.Solver()` on every call; a
  module-level instance would be built at import, and a test that patches the class would silently
  stop applying. Pinned by
  `tests/test_user_sbi.py::test_solver_failure_raises_instead_of_killing_the_process`.
- **`SimConfig.chi_mode` is a plain `False`.** The other chi fields read `core/config.py` when a
  configuration is built; `chi_mode` does not, and only `core.cli.make_sim_config` bridges
  `config.CHI_MODE` onto a configuration, so a `SimConfig` built by hand is never in chi mode.
- **Seed immediately before the chi block.** The chi block runs simulations of its own, so a
  common-random-number scheme that seeds only before the passive run gives its two arms different
  probe noise, and the result looks plausible and means nothing. The Fisher rotation and the
  identifiability Jacobian seed before it; training's probe draw uses a dedicated generator
  (`chi_gen` in `pipeline.gen_training_data`, seeded with a fixed value on every call), never the
  global stream those seeds would disturb.
- **numpy draws training's initial states.** `pipeline.gen_training_data` draws the built-in models'
  initial states with `np.random.randint`, which `torch.manual_seed` does not touch, so a claim that
  a seeded run reproduces needs both seeded; `core.rng.seeded` seeds both. Pinned by
  `tests/test_user_sbi.py::test_gen_training_data_is_reproducible_from_a_seed_in_every_mode`.
- **Warm TorchScript up before a bitwise comparison.** On CUDA a scripted step's first runs are not
  bitwise-reproducible against later ones, so an unwarmed comparison measures the compiler, not the
  change. Ask whether the baseline reproduces against itself before concluding a change broke it.
- **The rotation is not reproducible across processes, so a resume reuses the stored one.** The
  Fisher draws its operating points from the caller's unseeded global stream, so a restarted process
  computes another rotation, and resumed rows would sit in a different coordinate from their stored
  targets. `orchestrator.build_posterior` resolves the cache before the rotation and reuses the
  rotation in its header; a narrowing round reuses its parent's
  ([The Fisher settings on a resumed run](window.md#the-fisher-settings-on-a-resumed-run)). The card
  drill is in [The card smoke gate](testing.md#the-card-smoke-gate).
- **`torch.save` of a view writes the whole storage.** A row slice of a preallocated buffer is
  already contiguous, so `.contiguous()` does not help; only `.clone()` detaches it. Without it every
  cache commit would write the whole multi-GiB buffer, which shows only as slow checkpointing
  (`core.SBI.training_checkpoint.save`). Pinned by
  `tests/test_user_sbi.py::test_checkpoint_shards_do_not_serialize_the_whole_accumulator`.

### Rotations and narrowing rounds

The narrowing-round rules themselves are in the table above.

- **The rotation holds its eigenvectors in columns, and only `reparam.rotation_of` decodes it.** The
  rotated latent prior samples `w = z @ V`, the training transform stores the rotation transposed,
  and `core.SBI.reparam.rotation_of` is the one place that convention is read. A second reader once
  saved the rotation transposed, and every rotated posterior the window saved reloaded through the
  wrong bijection, with nothing to say so. `ArtifactStore.load_posterior` checks a posterior's
  rotation against the one pickled inside it (`reparam.assert_rotation_consistent`). An identity
  matrix is blind to a transpose, so an orientation test needs a non-symmetric rotation. Pinned by
  `tests/test_conditioning_repair.py::test_reparam_is_the_only_reader_of_the_rotation_matrix` and
  `tests/test_artifact_store.py::test_manifest_V_must_equal_the_rotation_in_the_pickled_prior`.
- **Calibrate on the truncated prior.** `validate_calibration` takes no region: it reads it off the
  loaded posterior, so calibrate the narrowed posterior itself, loaded with its acceptance.
- **The region belongs to the simulation cache's identity.** `SimulationIdentity` carries the region
  (None for an amortized run), so a round has a cache of its own and can neither resume nor be
  resumed by the amortized run's rows; `training_checkpoint.near_miss_siblings` ignores a cache that
  differs in the region alone. The amortized identity's digest is pinned to a fixed value, and any
  change to the identity's fields re-keys every cache on disk. The TSNPE tab cannot say in advance
  whether a round will resume, since the region is drawn when the round starts. Pinned by
  `tests/test_user_sbi.py::test_a_truncated_round_routes_to_its_own_checkpoint_and_the_amortized_digest_is_untouched`
  and
  `tests/test_settings_persistence.py::test_the_tsnpe_tab_never_claims_it_will_resume_the_amortized_checkpoint`.
- **A direction loaded on t_scale turns the restriction into a reweighting.** Training overwrites
  every row's t_scale with its batch's after the draw, which carries rows out of a box along such a
  direction; NPE does not correct a reweighting. `truncate.region_from_posterior` skips a direction
  whose t_scale loading exceeds `truncate.t_scale_loading_max` (1/√d), and the kept fraction is
  measured after the override ([Narrowing rounds](science.md#narrowing-rounds)).
- **torch's sigmoid inverse clamps at the box edge.** `SigmoidTransform`'s inverse clamps to just
  inside (0, 1) on torch 2.9, so the box round trip never gives an infinite latent target; do not
  write code or a bug report that assumes it can. Pinned by
  `tests/test_user_sbi.py::test_box_roundtrip_never_yields_a_nonfinite_latent_target`, so a torch
  upgrade that drops the clamp fails a test instead of a training run.
- **A fitted posterior used as the next prior is tempering.** Fit a density to one round's posterior
  and use it as the next round's prior at the same observation, and every round multiplies the
  likelihood in again: intervals shrink with no new information, and SBC, which checks the flow
  against its own proposal, stays flat. Restrict the prior, never reshape it
  (`truncate.TruncatedLatentPrior`; the truncated-prior rule above).

### Module layout

- **The façade re-imports are load-bearing.** `core.SBI.pipeline` re-imports its extracted siblings
  at the end of the module because tests rebind names on it, and the same pattern holds in
  `core/config.py`, `core/gui/panels/inference_tabs.py`, the end of `core/orchestrator.py` and
  `core/diagnostics/rng.py`. A consumer that bypasses the seam fails silently: the patch stops
  landing, and tests go slow or quiet, not red. Deleting a re-import fails loudly, except the
  inference shim's, which Settings skips without a word; the façade note under
  [Module map](architecture.md#module-map) has the rule in full.
- **`core/config.py`'s constants are snapshotted at import.** `from .config import NAME` binds the
  value when the importing module loads, so assigning `config.NAME` later is a silent no-op for that
  module, and moving a constant can change which value a caller sees. The three fixes in use are not
  interchangeable: read through the module at the moment of use, for a process-wide switch
  (`config.QUIET_SEGMENT_BAR`, `config.SIM_VRAM_CEILING_GIB`); carry it on the configuration, for
  anything a trained record must describe (the chi fields, `reparam_rotate`); or take it as an
  argument that defaults to the constant, for a per-run setting (the stages' keyword arguments).

### Reading diagnostics

- **A flat SBC is not informativeness.** A posterior that returns the prior is flat by
  construction; read the informativeness `validate` reports
  ([Reading calibration honestly](science.md#reading-calibration-honestly)).
- **The best-fit table is not recovery.** `core.SBI.overlay.rank_by_stats` scores a draw against the
  spread of the posterior's own draws, not against measurement noise, so along a degenerate direction
  the best fit is close to a free draw.
- **The calibration's operating points are t_scale's effective sample size.** Every dataset in a
  calibration batch shares its batch's t_scale, so lowering the operating-point count buys a weaker
  calibration, not a faster one; the verdict's caveat line says so every time
  ([Settings that are not speed dials](window.md#settings-that-are-not-speed-dials)).
- **A pooled SBC over mixed probe counts can hide a miscalibration** that differs by count; hold the
  count with `sbc --chi-k-fixed` ([sbc](command-line.md#sbc)).
