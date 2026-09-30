# The window

Checked against commit 3c4e393.

This page describes the PySide6 window screen by screen: what each screen is for, every setting on
the Parameter Inference tabs and what it really changes, what the window remembers between sessions,
and which models each screen accepts. [Launching PRISM](getting-started.md#launching-prism) says how
to start the window, and [A first run in the window](getting-started.md#a-first-run-in-the-window)
walks through the inference tabs once, in order.

## Screens

The window always opens on Home. Home offers five sections (`core.gui.screens.home_screen.SECTIONS`);
a back arrow under the PRISM title returns to Home from any of them. A gear at the bottom left of
every screen opens the appearance menu and, through **Full settings…**, the Settings screen.

| section | its tabs (`core.gui.main_window.MainWindow`) | what it is for |
|---|---|---|
| Reduction Map | NWK → Hopf reduction map | the Nadrowski model reduced to a Hopf normal form; analytical, with no stochastic simulation |
| FDT Analysis | FDT analysis; Sweep study cross-validation | the effective-temperature measurement for one cell, and the parameter-sweep study over S and T_a/T; their runs write `fdt` records, and each tab's comparison box compares saved ones |
| Parameter Inference | Config, Prior, Posterior, Validate, Infer, TSNPE | the inference chain, described below |
| Simulate | Live simulation | a live view of a cell's hair-bundle motion, streamed as it is simulated, which **Save video…** writes to an animation |
| Artifacts | none: one screen | browses the record store, as [Browsing the store](getting-started.md#browsing-the-store) describes |

Two more screens are not on Home: the gear's **Full settings…** opens Settings, and Settings opens
the model builder.

- **Settings**: the appearance (Follow system, Light, Dark or Auto (time of day), the Windows accent
  colour, the Inter font), the list of user-defined models with **Open model builder**, **Edit** and
  **Delete**, and a help reference assembled from each section's help text.
- **The model builder** defines a user model and writes its files into the inputs root;
  [Defining a model of your own](recordings.md#defining-a-model-of-your-own) describes it.

### The Parameter Inference tabs

The six tabs share one session: the configuration, the prior, the posterior and the observation
currently loaded (`core.gui.screens.inference_screen.InferenceScreen`). A tab is greyed until the
session holds what it needs, and hovering over a greyed tab says what is missing, by this rule
(`InferenceScreen.refresh_gates`):

| tab | what it does | enabled when |
|---|---|---|
| Config | records the model-level choices (model, units, chi mode and its probe settings, the Fisher rotation) and applies them with **Apply model & options** | always |
| Prior | picks the bounds file, which builds the configuration and selects the observation mode, then builds or loads the prior with **Build / Load prior** | a model has been applied |
| Posterior | trains or loads a posterior with **Train / Load posterior** | **Build / Load prior** has built the configuration; training also needs a loaded prior, loading a saved posterior does not |
| Validate | calibrates the posterior with **Run calibration** | a posterior and the prior loaded on the Prior tab |
| Infer | runs the posterior on a simulated cell or on recordings with **Run inference**, and records the observation | a posterior |
| TSNPE | a narrowing round around one observation with **Run TSNPE round** | as for Validate; the button also needs an observation in the tab's Observation picker, which lists the store's observation records (the Infer tab writes one each time it runs) |

Validate needs the prior loaded on the Prior tab, not the drive prior: a configuration whose bounds
file declares no Forcing section has no drive prior (`core.orchestrator.build_forcing_prior`), and
requiring it once made Validate unreachable for every model with no forcing. When a stage
greys the tab you are on, the screen falls back to Config.

**The prior picker does not load a prior.** Choosing an entry in the Prior tab's Prior picker changes
nothing until you press **Build / Load prior**: the session's prior is set only when that stage
completes (`PriorPanel._on_prior`), and **Train / Load posterior** trains on that prior, whatever
the picker shows (`PosteriorPanel._build_posterior`). The line under the picker describes the entry
it shows, not the prior the session holds.

**The Posterior tab's three lines** under the training budget update as you type: the number of
simulations the budget buys, the worst-case peak memory of one batch against what the planner has
free, and whether training resumes a simulation cache or starts a new one. The last line is the
check to read before a long run: it must begin "Resumes" when you mean to continue one.
[The artifact store](getting-started.md#the-artifact-store) describes the cache.

### One run at a time

Only one run can be live in the whole window, because a run's output is captured process-wide
(`core.gui.panels.base_panel.BasePanel`).

- While a run is live, every panel's controls are locked; the running panel's **Cancel** button
  stays live, and you can still move between screens and tabs.
- The header shows what is running and for how long ("Running: Posterior — 4:07"); clicking it goes
  to that tab. The running section's Home tile and its tab carry a marker.
- **Cancel** is a request: the run stops at its next checkpoint, which can take about a minute
  during training.
- Closing the window during a run asks first. **Cancel task & quit**, the default, cancels the run;
  **Quit anyway** closes the window and leaves the run going in the background until it finishes;
  **Don't quit** keeps the window open.
- The model builder refuses to validate or save, and Settings refuses to delete a user model, while
  a run is live. The Artifacts screen's own rule is in
  [Browsing the store](getting-started.md#browsing-the-store).

### The dialogs

- **Applying a model to a session that holds work.** When the session holds a configuration, a
  prior, a posterior or an observation, **Apply model & options** first asks "Start a new
  session?", listing them. **Keep this session** is the default; **Apply and start a new session**
  releases them. Nothing is deleted either way: each stays on disk and can be picked again. The
  line under the button says the same before you press it.
- **A training run one setting away from a cache.** When **Train / Load posterior** would start a
  new simulation cache while a committed one differs from this run in exactly one setting, the
  Posterior tab asks "This starts a NEW run", naming that cache, its batch count, the setting and
  both values. **Cancel** is the default. **Start a new run anyway** gives the consent the stage
  needs (`build_posterior(new_run=True)`; on the command line, `--new-run`); without it the stage
  refuses the same near miss itself. A narrowing round draws its region only after it starts, so the
  TSNPE tab cannot ask in advance: the round is refused, and you tick **Start a new simulation even
  if a cache one setting away exists** and run it again. The tick clears itself as soon as a round
  starts.
- **Loading a narrowed posterior.** Picking a posterior that a narrowing round trained and pressing
  **Train / Load posterior** asks "Load a NON-AMORTIZED posterior?". The dialog names the region's
  HPD level, its directions and the observation it was drawn around, and says what follows: Validate
  restricts its prior to the region, Infer refuses any other observation unless **Run on a
  different observation** is ticked, and a further round can be drawn only around this posterior's
  own observation. **Cancel** is the default; **Load it** is the command line's
  `--accept-truncated`.
- **Run on a different observation**, a tick on the Infer tab, runs a narrowed posterior on an
  observation other than its region's (`--accept-other-observation`). It is enabled only while the
  session's posterior is a narrowed one, and it clears whenever the posterior changes. A simulated
  inference needs it for any narrowed posterior: re-simulating the cell draws new noise, so the
  observation can never be the region's own. How each acceptance is recorded is in
  [The artifact store](getting-started.md#the-artifact-store).

### What a run shows

Each tab has a log pane, a progress pane and the run's figures, which appear as the stage draws them.

- **Validate** draws a fresh seed for each run (`ValidatePanel._validate`) and the calibration record
  keeps it; there is no Seed box, and the window remembers no seed. To repeat a calibration set, pass
  its recorded seed to the command line's `validate --seed N`
  ([validate](command-line.md#validate)). The log ends on the stage's `[verdict]` record: a head line
  with PASS or FAIL, the rule and the seed; one row per inferred parameter with its rank test's KS p
  and whether it passes, temperature marked "(assumed input)" on a box that declares temperature
  in place of the force scale; the joint coverage row; and a caveat about t_scale's test. The tab
  then adds its own line naming the verdict and the seed it drew. A FAIL is a result, not an error.
- **Infer** logs a summary of the posterior (the 5 %, 50 % and 95 % quantiles of each parameter) and
  draws a corner plot; on a box that declares temperature in place of the force scale, both mark it
  "T (assumed input, K)" (`core.sim_config.SimConfig.report_labels`). [The tier-1 constraint and
  temperature](science.md#the-tier-1-constraint-and-temperature) explains why.
- **TSNPE**, for a round drawn around an observation simulated from a known cell, logs for every
  truncated direction whether the cell's truth lies inside or outside the region, with its interval,
  and records the same answers on the new posterior; a truth outside also raises a warning.

## The inference settings

Every setting on these tabs reaches its stage as an argument, never as an assignment to a
`core.config` constant ([Stages and compositions](architecture.md#stages-and-compositions) says
why). The VRAM ceiling is the one exception, described after the table.

In the table, **flag** is the command-line tool's flag; **argument** is the parameter that receives
the value: a stage's in `core.orchestrator`, `core.cli.make_sim_config`'s for a setting that shapes
the configuration, or a field of `core.SBI.observations.RecordingSet` for a recording. **Default**
is the value a box opens at and a flag falls back to: `core/config.py`'s constant, or, where there
is none, the stage's own (the HPD level and the direction count come from `core.SBI.truncate`).
"Part of the cache's identity" means that changing the setting makes a training run start a new
simulation cache instead of resuming the old one.

| tab | field | flag | argument | default | what it really changes |
|---|---|---|---|---|---|
| Config | Model | `--model` | `make_sim_config(model)` | window: NADROWSKI; tool: the bounds file's folder name, upper-cased | Which model's equations are simulated and inferred. Each press of **Apply model & options** starts a new session, even with the same model; when the session holds work it asks first ([the dialogs](#the-dialogs)). |
| Config | Units | none: the tool always reads the model's units file | `make_sim_config(units_override=)` | the model's units file | Declares what the numbers in the bounds and cell files mean. It never converts them, so a change reinterprets the files rather than rescaling them. Part of the cache's identity. |
| Config | Multi-frequency χ(ω) conditioning | `--chi` / `--no-chi` | `make_sim_config(chi_mode=)` | off | Chi mode: an observation is a passive recording plus single-tone driven recordings, and the network conditions on the χ(ω) curve they give. Training simulates 1 + K recordings per row where forced mode simulates 2, with K drawn for each batch from 2 to the probe slots, and a posterior loads only in the mode it was trained in. |
| Config | χ probes per observation | `--chi-k` | `make_sim_config(chi_n_freqs=)` | 6 | The probes one observation is measured at: the number a simulated observation gets, the number the Fisher rotation is computed with, and the number of probe rows the Infer tab starts with. Training draws its own count for every batch, so a posterior accepts any count up to its slots. The Config tab accepts 2 up to the slots. |
| Config | χ probe slots (capacity) | none: the tool takes `CHI_K_PAD` | `make_sim_config(chi_k_pad=)` | 12 | The network's probe capacity, frozen into every posterior trained with it; training draws from 2 up to this many probes per batch. Raising it later means retraining. It costs input columns only. Part of the cache's identity. |
| Config | χ lock-in ceiling (cycles) | none: the tool takes `CHI_MAX_CYCLES` | `make_sim_config(chi_max_cycles=)` | 20.0 | The longest lock-in, in drive cycles, used for any one probe; a longer recording is truncated to it, not refused. Past about 30 cycles χ stops being reproducible at fixed parameters, so a longer lock-in adds noise, not signal. It must exceed the 2-cycle floor below which a probe is masked. Part of the cache's identity. |
| Config | χ drive F₀ (ND), read-only | none | none: `CHI_F0` | 0.15 | The non-dimensional drive amplitude of every probe, fixed by measurement and baked into every trained posterior. A display of `core/config.py`: no run can change it; the prior and training stages, `probes mask` and every posterior load refuse a configuration that disagrees with it (`core.SBI.run_guards._assert_chi_config_is_deliberate`), and loading refuses a chi posterior trained at another drive. |
| Config | χ frequency range, read-only | none | none: `CHI_FREQ_BOUNDS` | 0.03 to 0.3 | The band the probes are log-spaced across, as multiples of each observation's own measured spontaneous peak Ω₀, so the probes follow the resonance wherever t_scale puts it. Fixed by measurement and guarded like the amplitude. |
| Config | Decorrelating Fisher rotation | none: the tool takes `REPARAM_ROTATE` | `make_sim_config(reparam_rotate=)` | on | Rotates the flow's latent coordinates into the eigenbasis of a simulated Fisher matrix, so that strongly correlated parameters become axis-aligned. The rotation is orthogonal, so it adds and removes no information; computing it costs extra simulations before training. Part of the cache's identity. |
| Config | VRAM ceiling per batch (GiB, 0 = off) | none: the `PRISM_VRAM_CEILING_GIB` environment variable | none: the field sets `core.config.SIM_VRAM_CEILING_GIB` | 0 (off) | A hard ceiling on the card memory one simulation batch may plan to hold. See below the table. |
| Prior | Bounds | `--bounds` | `make_sim_config(bounds_file)`; with **Edit values**, `bounds_dicts=` | none | Which parameters are inferred, in what order and over what box, and so the observation mode. **Edit values** starts from the selected file and lets you change only the numbers. |
| Prior | Prior | `--prior` on `train`; `prior` always builds | `build_prior(ref, build_new)` | (from scratch) | A saved prior to load, or "(from scratch)" to build one. Choosing an entry does nothing until **Build / Load prior**. |
| Prior | Global rounds | `--num-iterations` | `build_prior(num_iterations=)` | 50 | The rounds of the global census; the candidates screened are the rounds times the candidates per round, and each round integrates its candidates over the first half of the Stability duration (the flood-fill integrates the whole of it), so rounds cost time. |
| Prior | Candidates per round (0 = auto) | `--sweep-batch` | `build_prior(sweep_batch=)` | 0: the hardware batch | Not a speed dial: see [below](#settings-that-are-not-speed-dials). |
| Prior | Max accepted sets | `--max-sets` | `build_prior(max_sets=)` | 175000 | The accepted sets that stop the flood-fill: the point cloud the prior is fitted to. Not a speed dial. |
| Prior | Random-walk step | `--walk-step` | `build_prior(walk_step=)` | 0.01 | The size of the flood-fill's random step, in the parameters' own units: the same absolute size for every non-dimensional parameter, not a fraction of each one's range. Too small and the walk never leaves its seed points; too large and it steps across the stable region instead of tracing it. |
| Prior | Stability duration (ND units) | `--stability-units` | `build_prior(stability_units=)` | 1000 | Not a speed dial. |
| Prior | Min cluster size | `--min-cluster-size` | `build_prior(min_cluster_size=)` | 50 | Not a speed dial. |
| Prior | Min samples | `--min-samples` | `build_prior(min_samples=)` | 10 | Not a speed dial. |
| Posterior | Posterior | `--posterior` on the commands that load one; `train` always trains | `build_posterior(ref, train_new)` | (from scratch) | A saved posterior to load, or "(from scratch)" to train one on the loaded prior. |
| Posterior | Batches | `--num-runs` | `build_posterior(num_runs=)` | 5000 | The training batches to simulate. Every row of a batch shares one (t_scale, T_obs) operating point, so the batch count is the training set's operating-point diversity as well as its size, and it is what wall-clock scales with. Part of the cache's identity. |
| Posterior | Max rows per batch (0 = auto) | `--run-size` | `build_posterior(run_size_cap=)` | 0: the hardware batch | A memory escape hatch, not a speed control. The hardware batch is 2,048 rows on a CUDA card and 64 elsewhere; the solver is bound by kernel launches, so a narrower batch is not faster (measured: 7.37 s at 2,048 rows against 7.74 s at 1,024), and a lower cap trades training rows for peak memory about one for one. Part of the cache's identity. |
| Posterior | Hidden features | `--hidden-features` | `build_posterior(hidden_features=)` | 128 | The flow's width: hidden units per spline transform. Not part of the cache's identity, so a complete cache can retrain the flow at another width without simulating again. |
| Posterior | Transforms | `--num-transforms` | `build_posterior(num_transforms=)` | 8 | The flow's depth: the number of spline transforms. Retrainable against a complete cache, as the width is. |
| Posterior | Learning rate | `--learning-rate` | `build_posterior(learning_rate=)` | 0.001 | Adam's learning rate for the flow. |
| Posterior | Early-stop patience | `--stop-after-epochs` | `build_posterior(stop_after_epochs=)` | 20 | The epochs training waits for a better validation loss before it stops. |
| Posterior | Ensemble per perturbation | `--fisher-m` | `build_posterior(fisher_m=)` | 48 | The ensemble per latent perturbation in the Fisher estimate; its cost is linear in this. Ignored on a resumed run ([below](#the-fisher-settings-on-a-resumed-run)). |
| Posterior | Central-difference step | `--fisher-dz` | `build_posterior(fisher_dz=)` | 0.1 | The latent central-difference step of the Fisher's Jacobian. Ignored on a resumed run. |
| Posterior | Operating points | `--fisher-points` | `build_posterior(fisher_points=)` | 8 | The operating points the Fisher is averaged over. Averaging is what makes one linear rotation valid across the whole prior. Ignored on a resumed run. |
| Validate | Calibration datasets | `--n-cal` | `validate_calibration(n_cal=)` | 2000 | The datasets drawn from the prior and simulated for the rank test (SBC) and the joint coverage test (TARP). |
| Validate | (t_scale, T_obs) operating points | `--cal-n-scales` | `validate_calibration(cal_n_scales=)` | 200 | t_scale's effective sample size in the calibration. Not a speed dial. |
| Validate | none: drawn for each run | `--seed` | `validate_calibration(seed=)` | one is drawn and recorded | The calibration set's seed. On one device the same seed repeats the calibration: bit for bit on the CPU, not bitwise on a CUDA card. |
| Infer | Mode | `--cell` or `--spont` | `simulated_inference` or `experimental_inference` | Simulated (cell ground truth) | Simulated re-simulates a cell's ground truth; Experimental data takes your recordings, in the shape the configuration's observation mode asks for ([Observation modes](recordings.md#observation-modes)). |
| Infer | Cell | `--cell` | `simulated_inference(cell=)`; with **Edit values**, `gt_values=` | none | The cell whose ground truth is simulated. The picker lists the configured model's cells and checks a picked cell against the bounds file at once; **Run inference** stays greyed while it does not fit. |
| Infer | T_obs (s), one box on each page | `--t-obs` | `simulated_inference(T_obs_s)`; `RecordingSet(T_obs_s=)` | window: 1; tool: none, it must be given | The observation length in seconds: the length the cell is re-simulated at, or your recordings' length ([Recording length](recordings.md#recording-length)). |
| Infer | Spontaneous; on the χ page, Passive | `--spont` | `RecordingSet(spont=)` | none | The undriven recording. |
| Infer | Forced | `--forced PATH` | `RecordingSet(forced=)` | none | Forced mode's one driven recording; the row is hidden when the bounds file declares no Forcing section. |
| Infer | A (N), f (Hz), φ (rad) | `--drive NAME=VALUE` | `RecordingSet(forcing_params_si=)` | window: 0, which is refused for A and f; tool: none | The drive the forced recording was made at, in SI units: one box per forcing parameter the bounds file declares. |
| Infer | Drive F₀ (N) | `--f0-si` | `RecordingSet(F0_si=)` | window: 1; tool: none, required in chi mode | The physical drive amplitude every chi recording was made at, in newtons, one value for every probe. The lock-in divides each response by it, so it must be the amplitude actually applied; nothing checks the value you give. An active bundle does not respond linearly, so how hard you drive also matters ([Observation modes](recordings.md#observation-modes)). |
| Infer | Forced probes, the χ probe table | `--forced PATH@HZ`, repeated | `RecordingSet(forced=)` | as many rows as χ probes per observation | One row per single-tone recording: the file, and the frequency you actually drove at, in Hz. Any count from 1 to the posterior's slots works. **Plan probes…** measures Ω₀ from the passive recording and says what is in band for this cell and how long each probe must be. |
| Infer | Run on a different observation | `--accept-other-observation` | `accept=Accept(other_observation=True)` | off | See [the dialogs](#the-dialogs). |
| TSNPE | Observation | `--observation` | `tsnpe_round(observation)` | none | The observation the region is drawn around. Pressing **Run TSNPE round** loads it and checks its mode and conditioning width against the configuration before anything is spent. |
| TSNPE | HPD level | `--level` | `tsnpe_round(level=)` | 0.999 | The credible level of the region's interval along each truncated direction. Generous on purpose: truncation deletes prior support that no later round can recover, while a region too wide costs only simulations. A level below 0.99 warns. |
| TSNPE | Directions truncated | `--directions` | `tsnpe_round(n_directions=)` | 5 | How many directions are truncated: the leading Fisher directions of a rotated posterior, best-constrained first, or the leading parameters, in box order, of an unrotated one. The rest keep the full prior width, so the least-constrained directions are not cut on noise, and a direction whose t_scale loading is above a fixed threshold is skipped and the next one taken. At most the posterior's latent width. |
| TSNPE | Batches | `--num-runs` | `tsnpe_round(num_runs=)` | 5000 | As on the Posterior tab, for this round. |
| TSNPE | Max rows per batch (0 = auto) | `--run-size` | `tsnpe_round(run_size_cap=)` | 0: the hardware batch | As on the Posterior tab, for this round. |
| TSNPE | Start a new simulation even if a cache one setting away exists | `--new-run` | `tsnpe_round(new_run=)` | off | See [the dialogs](#the-dialogs). |
| command line only | none | `--checkpoint-every` | `build_posterior(checkpoint_every=)`, `tsnpe_round(checkpoint_every=)` | 50 | Batches between commits of the simulation cache; 0 keeps no cache, so nothing can resume. The window always uses the default. |
| command line only | none | `--resume` | `build_posterior(resume=)`, `tsnpe_round(resume=)` | auto | What a training run does with its own cache: see [Training-cache flags](command-line.md#training-cache-flags). The window always uses auto. |
| command line only | none | `--max-epochs` | `build_posterior(max_num_epochs=)`, `tsnpe_round(max_num_epochs=)` | no ceiling | A ceiling on training epochs; see [train](command-line.md#train) for how many run and which network is kept. |
| command line only | none | `--posterior-samples` | `validate_calibration(num_posterior_samples=)` | 1000 | Posterior draws per calibration dataset. |
| command line only | none | `--n-samples` | `simulated_inference(n_samples=)`, `experimental_inference(n_samples=)` | 1000 | Posterior draws for the corner plot, the posterior predictive check and the summary. |
| command line only | none | `--device` | the configuration's device (`core.tool.config_args.make_cfg`) | auto | Where the run computes: see [Configuration flags](command-line.md#configuration-flags). The window always takes auto. |

A narrowing round started from the TSNPE tab trains its flow at `core/config.py`'s network settings:
the Posterior tab's network boxes do not reach it. On the command line, `tsnpe` takes
`--hidden-features`, `--num-transforms`, `--learning-rate`, `--stop-after-epochs`, `--max-epochs`,
`--checkpoint-every` and `--resume`, and no Fisher flag.

**The VRAM ceiling** is the one field on these tabs that assigns a `core.config` attribute
(`SIM_VRAM_CEILING_GIB`, in `ConfigPanel._apply_vram_ceiling`). The assignment works because the
batch planner reads the ceiling afresh every time it plans a batch
(`core.SBI.pipeline.vram_ceiling_gib`), not through a binding made at import. It caps what one
simulation batch may plan to hold on the card; a batch that would need more is split, which costs
wall-clock. It cannot free memory: with nothing free, not even the smallest chunk fits and the batch
runs as asked. What it buys is keeping a run that has headroom inside the card's own memory, where
otherwise Windows pages the batch into shared system memory and it runs up to nine times slower with
nothing in the log to say why. Set it to about the free memory `nvidia-smi` reports, minus about
1 GiB for the CUDA context; 0, the default, is right on an idle card. On the command line the same
ceiling is the `PRISM_VRAM_CEILING_GIB` environment variable, which the planner reads in the same
way; when it is set it wins over the field, and the note under the field says so. The field opens
at the environment variable's value when that is set, else at 0.

## Settings that are not speed dials

Five of the settings look like a trade of time for precision, and are not: each changes the prior
or the measurement. [The prior](science.md#the-prior) and
[Reading calibration honestly](science.md#reading-calibration-honestly) give the science behind them.

- **Candidates per round** (Prior tab, `--sweep-batch`). The global census runs a fixed number of
  rounds (Global rounds), and each round integrates its candidates over the first half of the
  Stability duration whatever the round's width, so a narrower round screens fewer candidates in
  about the same time. Half this number is also the batch of the local flood-fill, which runs until
  Max accepted sets are accepted. Shrinking it makes the prior worse without making it faster:
  measured, a prior build took 527 s at 2,048 candidates per round, and at 32 it was still
  unfinished after more than 70 minutes.
- **Stability duration** (Prior tab, `--stability-units`). How long, in non-dimensional time units,
  the flood-fill integrates each candidate; the census integrates the first half of it. It defines
  what "stable" means: a longer screen rejects slow instabilities that a shorter one accepts, so it
  moves the prior's support, not only the time the sweep takes.
- **Min cluster size and Min samples** (Prior tab, `--min-cluster-size`, `--min-samples`). HDBSCAN
  clusters the accepted point cloud, and the number of clusters it finds is passed straight to the
  Gaussian mixture as its component count (`core.SBI.Priors.prior.Prior.construct_prior`), so these
  two decide how many modes the prior has. Measured on a small prior build, changing nothing else:
  a minimum cluster size of 5 gave 15 components, and 60 gave 1. Min samples sets how conservative
  HDBSCAN's density estimate is, and so how many points it calls noise and how many clusters it
  finds; the mixture itself is then fitted to every accepted point, noise included. A different
  component count is a different prior, not a faster one.
- **The (t_scale, T_obs) operating points** (Validate tab, `--cal-n-scales`). Every calibration
  dataset in one batch shares that batch's t_scale, so the rank test for t_scale rests on as many
  independent values as there are operating points (200 by default), not on the 2,000 datasets. This
  number is t_scale's effective sample size: lowering it does not buy the same calibration faster,
  it buys a different and weaker one. The verdict's caveat line says so on every calibration.
- **Max accepted sets** (Prior tab, `--max-sets`). The accepted sets are the point cloud HDBSCAN
  clusters and the mixture is fitted to. More of them buy coverage of the stable region, not
  statistical precision: a mixture of a few components needs nothing like 175,000 points.

## The Fisher settings on a resumed run

A training run whose settings match a committed simulation cache resumes it; the Posterior tab's
checkpoint line says so before you press **Train / Load posterior**. With the Fisher rotation on, a
resumed run reuses the rotation stored in the cache's header and does not run the Fisher step
(`core.orchestrator.build_posterior`). That is required, not an optimisation
([Randomness and reproducibility](rules-and-traps.md#randomness-and-reproducibility) says why). The
log says "Reusing the Fisher rotation stored with the training checkpoint".

- **The three Fisher settings do nothing on a resume.** Ensemble per perturbation, Central-difference
  step and Operating points (`--fisher-m`, `--fisher-dz`, `--fisher-points`) are not part of the
  cache's identity, so changing them does not stop a resume, and the resumed run ignores them. The
  posterior's record stores all three as empty (null): they are written only when the Fisher ran in
  that process, and a freshly computed set of eigenvalues is the one witness that it did.
- **The rotation's eigenvalues carry over.** They say how strongly each rotated direction is
  constrained, and the cache header stores them beside the rotation, under `"fisher_eigenvalues"`
  (`core.SBI.training_checkpoint.create`), so a resumed posterior records them. A cache written
  before the header kept them has no such key: the run still resumes, its log says the checkpoint
  stores no rotation eigenvalues, and the posterior records them as unknown.
- **A narrowing round takes no Fisher settings at all.** It trains in its parent posterior's
  rotation, the one its region was measured in, and records the parent's eigenvalues. The TSNPE tab
  has no Fisher boxes, and `tsnpe` has no Fisher flags.
- With the rotation off, no Fisher runs and the three settings are recorded as empty too.

## What the window remembers

The window keeps what it remembers in `PRISM.ini`, a Qt settings file in your user settings folder
(`%APPDATA%\PRISM\` on Windows, `~/.config/PRISM/` on macOS and Linux; `core.gui.settings.settings`).
Each panel writes its own group when the window closes (`MainWindow._save_state`); a splitter
position is written a moment after you drag it, and the appearance as soon as you change it. A key
an older build wrote and this one does not read is ignored, never restored.

| screen or tab | kept from the last session | opens fresh on every launch |
|---|---|---|
| Config | the model; the units source and the typed units; the chi tick; the rotation tick | χ probes per observation, χ probe slots and χ lock-in ceiling, at `core/config.py`'s values; the VRAM ceiling; the drive amplitude and band, which are displays and are never stored |
| Prior | the Bounds and Prior pickers | the seven sweep and clustering boxes; the bounds grid, which always opens on its file |
| Posterior | the Posterior picker; Batches; Max rows per batch | the four network boxes and the three Fisher boxes |
| Validate | nothing | both boxes; the seed is drawn for each run |
| Infer | the mode; the Cell picker; the three T_obs boxes; the Spontaneous, Forced and Passive paths; Drive F₀ (N) | the drive boxes and the probe table, which are built from the configuration; the cell's value grid; **Run on a different observation** |
| TSNPE | the Observation picker; Batches; Max rows per batch | HPD level and Directions truncated; the new-simulation tick |
| Live simulation | the model, the cell, T_obs, Steps / frame, Max FPS | nothing |
| FDT analysis | the model, the cell, the Saved run picker, n_freqs, M_ensemble, freqs / batch, F0 | Seed, Record name, Note; the two ticks, Skip sanity checks and Proceed to the production sweep after sanity; the Compare saved runs box |
| Sweep study cross-validation | the preset, the cell, the Saved sweep picker, F0, freqs / batch, and each grid's number of points | n_freqs and M_ensemble, which come from the preset; each grid's ends, which come from the cell; Seed, Record name, Note; the Compare saved sweeps box |
| NWK → Hopf reduction map | the cell and F0 | nothing |
| Artifacts | the kind and the table's sort | nothing |
| the window | its size and position, each panel's splitter positions, the appearance | the screen: it always opens on Home |

On the Parameter Inference tabs, then, the window remembers selections (the pickers, the model, the
units, the chi and rotation ticks), the boxes that describe a recording, and the training budget;
every other setting opens at its default on every launch. The reasons, one per rule:

- **A science setting on the inference tabs is never remembered**, because a remembered one
  outlives a change to `core/config.py`: a chi band retired in `core/config.py` was once restored
  from saved settings on every launch and trained a run of about five days on the old band. The
  window now neither writes nor reads the band or the drive amplitude, and the stages refuse a band
  or amplitude that differs from `core/config.py`'s.
- **A consent is never remembered**: the new-simulation tick, **Run on a different observation**
  and the FDT analysis tab's two ticks are answered by the session that runs, so a yes given once
  cannot silence a later refusal.
- **A seed is never remembered**: a remembered seed would silently make every run a repeat of the
  last one.
- **A record name and a note are never remembered**: a remembered name would be refused as taken at
  the next launch's first run.
- **The comparison boxes are never remembered**: a remembered list would name records that a later
  session may have deleted.
- **The VRAM ceiling is never remembered**: a forgotten ceiling does not fail, it makes every later
  run split its batches and take several times longer, with nothing in the log to explain it.
- **Grid ends and hand-edited grids are re-derived**: the cross-validation grids' ends come from the
  cell, and the bounds and cell value grids start from the selected file, because a value saved
  against another cell or model would be a wrong bound or a mis-bound parameter.

## Supported models

The three built-in models, BP, NADROWSKI and HOPF, are the first entries of
`core.config.VALID_MODELS`. A user model saved in the model builder is added to that list, so it
appears in every model list, but not every screen accepts it.

| screen | built-in models | user models | on the command line |
|---|---|---|---|
| Live simulation | all three | all | none: the tool has no live simulation |
| Parameter Inference | all three | only one with no forcing and at least one non-dimensional parameter (`core.registry.is_sbi_user_model`); for any other, **Apply model & options** is greyed and the log says why | `prior`, `train`, `validate`, `infer`, `tsnpe`, `smoke`, `sbc`, `identifiability`, `ablation` and `probes`, under the same rule, refused in `core.tool.config_args.make_cfg`; `probes drive` also needs a bounds file with a Forcing section ([probes drive](command-line.md#probes-drive)) |
| FDT analysis | all three | only one whose observable, its first variable, has additive, non-zero noise, and which has no forcing of its own (`core.registry.fdt_support`); for any other, **Run FDT analysis** is greyed and the log says why | `fdt`, under the same check; `compare cells`, `compare repeats` and `compare renormalise` compare saved records |
| Sweep study cross-validation | Nadrowski only (`_MODEL` in `core.gui.panels.crossval_panel`) | none | `crossval`, Nadrowski only; `compare sweeps` compares saved sweeps |
| Reduction Map | Nadrowski only (`_MODEL` in `core.gui.panels.reduction_panel`): the map is a property of that model's equations | none | none |
| Artifacts | all | all | `artifacts` |

FDT refuses a user model with forcing of its own because it drives the observable itself to measure
the response, and the model's own drive would be silently replaced.

The inference pipeline is tuned on the Nadrowski model. It runs the whole chain for an eligible user
model too, but calibration is not pre-tuned for user models: validate such a posterior before you
trust it. The log says so when you apply a user model on the Config tab.
