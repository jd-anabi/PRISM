## 8. The steps of Parameter inference

Five pages: 1 Model and options, 2 Prior, 3 Training (setting up only), 5 Inference, 6 Narrowing
round. This part gives each page's settings, "Before you start" facts, buttons, own questions and
result. What the pages share is other parts' and is used by reference (8.1). Engine: E2, E4, E6 and
E7 as part 11 gives them, and E11's wording; nothing beyond E1 to E12.

Columns of the control tables. **Place**: M main setting, R "Rarely changed", N the name-and-note
block. **Plan**: whether a plan may hold it (7.1.3, 7.1.4). Where U7 gives no name or wording, the
on-screen labels, the sentences, the questions' titles, texts and answers (U10 gives "Cancel"), the
runs' words, and which settings sit under "Rarely changed", are this part's (a choice the owner has
not seen).

### 8.1 What the five pages share

#### Used as other parts define it

| Thing | Defined in | These pages add |
|---|---|---|
| What a step holds changes only when a run returns its record (B3) | 5.3 | `install_config(cfg)` and the release of the posterior and the observation move from the click (`prior_tab.py:209-210`) into `_on_prior` (`prior_tab.py:315-319`); the release at `posterior_tab.py:142` moves into `_on_posterior` (`posterior_tab.py:287-298`). No configuration exists without its prior, so loading a posterior cannot crash at `core/orchestrator.py:902` (W090, W147). Checked by: after a prior or posterior run that is refused, fails or is stopped, the session's `cfg`, `inf_prior`, `posterior` and `observation` are the objects they were before the click |
| "In use", "Changed, not applied" | 5.3 | The selection rows: step 1, every row; step 2, "Bounds file" (or the typed limits) and "Saved prior"; step 3, "Saved posterior" |
| A question | B2, 4.6; the questions before something held is let go, 5.3 | Those of 8.4 to 8.6. No second answer of theirs loses work, so none is drawn as destructive (B27) |
| Name and note | 5.5: required for a prior, a posterior and an observation, optional for an inference | The rows in the tables. The two "Save" boxes go (U14) |
| The main button, its greyed reason, refusals | 5.6, 5.7 | Five problems that are only text-pane lines today (`prior_tab.py:152-153`, `:188-191`, `:195-197`, `:207-208`, `:324-326`) become facts or lines under their field (W047) |
| A step opens whatever the earlier steps hold (B7; a choice the owner has not seen) | 3.4 | The greyed button's reason is the step's `StepState` detail |
| File lists keep their selection | 7.3.1 rule 3, 5.10 | Signals are held during a re-list, so no entry in passing is judged (W148). For a remembered file that is gone: "The bounds file used last time, master_tier1.txt, is no longer in the folder." |

Used with nothing added: the frame of "Before you start" (5.4), and what is remembered and when it
is written (7.3.2; B6, B12). Each page's facts come from a function with no widget, tested as
values (13.1.5).

#### What these pages give the chain (3.4)

The conditions, markers, state words and a model's name on screen are 3.4's. This part supplies the
words for a step in hand: "choose a model" (step 1), "choose a prior" (step 2), "setting up" (steps
3 and 5). For step 3's "Unfinished" it supplies the detail "simulations saved to batch N" and, to
fit the rail, the short form "batch N", N from the run history's `simulation_set` (6.5).
`observation_on_offer` is true when step 6's "Observation" list holds an entry that can be chosen
(8.6).

#### Opened or filled from elsewhere

| Caller | What the pages do |
|---|---|
| A plan or a record filling the forms (7.1.7; "Set up a run like this one", 10.2) | Each page exposes `plan_fields()` (7.1.3) for the rows whose Plan column says yes and shows the plan line of 7.1.6 under its heading. Step 1 places the "Plan" row under "Model" |
| A next-step button on a record's page (B4, C9; 10.2) and "Continue with these" (U24, B16, C1; 3.6) | One entry on the inference screen, `load_chain(model, options, bounds_file, prior, posterior=None, observation=None)`. In order: step 1 is applied with the model and options given; steps 2 and 3 run their own "Load this …" through `load_record(id)`, rules included; an observation is loaded into the session by `load_observation(cfg, id)`, as a round does today (`tsnpe_tab.py:147-148`). The caller asks its one question up front, what is loaded and what let go, so 5.3's are not asked again; a consent is never implied, so the narrowed-posterior question (8.4) is still asked inside the chain when that load arises (C9). A link that is refused or fails ends the chain and the earlier ones stay (B3). Nothing is built or trained. Records passes what the record was made with (10.2); for an observation alone, whose button is greyed until a posterior is in use (C9), only the last link runs. Home passes last time's picks, with the remembered kind of observation (B12) and every other option at its built-in value or the plan's (C1, a choice the owner has not seen; 3.6) |
| "Continue from batch N" (C2, C4; 6.8) | The forms are filled from the run history's `settings` through `fill_forms` (7.1.7), the earlier steps applied with its picks through `load_chain`, and Training or Narrowing round passes `resume="require"`; Training shows "Only continue". "Name" and "Note" open blank, with "The earlier run was named retrain_3." under "Name" (C4; 6.8). The main button reads "Continue from batch N", or after a failure in the fit "Fit again on the saved simulations" (C2): for a training, N from the preview; for a narrowing round, whose set the preview cannot name before the region is drawn, N from the run history's `simulation_set` (6.5, 6.8) |
| Records deleting or renaming a record in use (10.2) | `SbiSession` gains `holds(kind, id)` and `release(kind)` (the slot and what `reset_downstream` lets go with it, `session.py:49-57`); "In use" reads the record's name again when the store changes |
| A record's page in Records showing its result (10.2) | Each page's "Afterwards" is one block built from a record's manifest and folder, as 9.4's is, so Records hosts it; the "In use" line (5.3) is no part of it |

#### What each page hands the run

Each start is `dispatch(fn, *args, spec, on_result=None, **kwargs)` (6.2).

| Page | The run's word | Record written | Stage names | Also in `spec` |
|---|---|---|---|---|
| Prior, build | "Prior build" | prior | none | |
| Prior or Training, load | "Prior load", "Posterior load" | none | none | that it only loads: "Loading…" at the rail's foot, then idle with nothing to acknowledge; no history or text file (C3, a choice the owner has not seen; 6.5) |
| Training | "Training" | posterior | "Rotation", "Simulation", "Fit", "Saving" (U7) | whether a saved set is continued; the patience and epoch limit passed; "Expected epochs" when typed (B9) |
| Inference | "Inference" | observation, inference result | none | |
| Narrowing round | "Narrowing round" | posterior | as Training; "Rotation" never runs, the posterior's own being reused (8.6) | as Training, without "Expected epochs" |

When a record comes back with no name although one was typed (E7, 11.8), the result repeats the
run's sentence (6.9), "It was saved without its name: 'retrain' was taken while the run was
going.", and offers "Rename…" (10.2). Only the window opts in to E7 (C22), for the prior build,
Training, the Narrowing round and, for its observation only, Inference, as 11.8 gives it. It is not
carried to the inference made together with that observation (C22; a choice the owner has not
seen): there the inference's name taken meanwhile is a refusal (5.7) that costs one observation,
not a training (11.8).

#### `core/gui/fields.py`

`CONTROL` (`:55-187`) takes the labels of the tables below and its places become the step names;
each entry and its pinned tests are edited in the task that builds the page (5.7). Keys that gain a
control: `checkpoint_every` (`:60`), `resume` (`:61`), `n_samples` (`:72`), `max_num_epochs`
(`:86`), and the new `observation_name` (E4), `chi_mode`, `reparam_rotate` and `expected_epochs`
(7.1.5).
`CONTROL["posterior"]` names step 3's "Saved posterior". The sentences for `accept_truncated`
(`:91`) and `accept_other_observation` (`:92`, which quotes the tick that goes) name the questions
of 8.4 and 8.5.

### 8.2 Step 1: Model and options

Sentence under the heading: "Choose the model and the kind of observation. Everything after this
step is made for these choices."

| On screen | Today | Key | Place | Built-in value, from | Reaches | Plan |
|---|---|---|---|---|---|---|
| "Model" | "Model" (`config_tab.py:44-51`) | `model` | M | NADROWSKI (`config_tab.py:46`), then the remembered one | `ConfigDraft.model` (`config_tab.py:329-330`), then `cli.make_sim_config(model, …)` (`core/cli.py:273`) | no (B13) |
| "Kind of observation": "Frequency response" or "Set by the bounds file" | tick "Multi-frequency χ(ω) conditioning" (`config_tab.py:64`) | `chi_mode` | M | "Set by the bounds file" (the tick's fallback, `config_tab.py:421`), then the remembered one (B12) | `chi_mode` (`cli.py:275`, `:335`) | yes |
| "Driven recordings per observation" | "χ probes per observation" (`fields.py:138`) | `chi_n_freqs` | R | 6 (`config.py:463`) | `chi_n_freqs` (`cli.py:336`) | yes |
| "Slots for driven recordings" | "χ probe slots (capacity)" (`fields.py:139`) | `chi_k_pad` | R | 12 (`config.py:528`) | `chi_k_pad` (`cli.py:340`) | yes |
| "Lock-in limit", unit "drive cycles" | "χ lock-in ceiling (cycles)" (`fields.py:140`) | `chi_max_cycles` | R | 20 (`config.py:538`) | `chi_max_cycles` (`cli.py:341`) | yes |
| "Rotation", on or off | tick "Decorrelating Fisher rotation" (`config_tab.py:89-90`) | `reparam_rotate` | R | on (`config.py:404`); no longer remembered (U15) | `reparam_rotate` (`config_tab.py:334`; `cli.py:278`, `:342`) | yes |
| "Units": "From the model's units file" or "Typed here" | "Units": "Model's units file" or "Type units" (`config_tab.py:54-60`) | `units` | R | "From the model's units file"; no longer remembered (U15) | `units_override`: nothing, or the typed words (`config_tab.py:192-197`; `cli.py:274`, `:313-321`) | yes |

The three settings of the frequency response are shown only when that kind is chosen; otherwise the
draft carries nothing for them and the builder takes `core/config.py`'s (`config_tab.py:225-226`).
Rules kept from `config_tab.py:230-260`: 2 ≤ per observation ≤ slots ≤ 24 (`config.py:509`), lock-in
limit above 2 (`config.py:530`), typed units must resolve.

Explanations given once, under their settings: "Frequency response (χ(ω)): one spontaneous
recording and several driven recordings, each at one frequency."; "Set by the bounds file: a
spontaneous recording, and one driven recording when the file has a forcing section."; "Slots: the
most driven recordings one observation can hold. Fixed in the posterior once it is trained.";
"Lock-in limit: the longest stretch of a driven recording, in drive cycles, used to measure the
response."

**The two measured definitions** are text with the origin "fixed by measurement" (5.2; W110), and
no plan holds them (U8):

- "Drive amplitude in training: 0.15 (no unit)." From `config.CHI_F0` (`config.py:581`).
- "Frequency band in training: 0.03 to 0.3 times the cell's own frequency." From
  `config.CHI_FREQ_BOUNDS` (`config.py:464`).

Neither is read from the page: the draft carries nothing for both (`config_tab.py:332`), so every
configuration takes `core/config.py`'s (`cli.py:337-339`).

**Units.** The line reads, for example, "kHz ms nm pN, from Units/nadrowski/units.txt": the words
in alphabetical order (W111), the path from `cli.resolve_units_file` (`cli.py:39-48`) relative to
`config.RESOURCES_ROOT`, not the literal built at `config_tab.py:188`. Under "Typed here": "Typed
units cannot continue a simulation set made with the model's units file, even when they match it."
(`core/artifacts/identity.py:28-32`, `:56-67`: a file is fingerprinted by its bytes, typed words by
their sorted text, so only a set made with the same typed words is continued; W141). The typed box
is refilled from the selected model's file whenever the model changes (a choice the owner has not
seen; today it is filled only while empty, `config_tab.py:189-190`, and remembered, so it can
declare another model's units, W146).

**Main button:** "Apply model and options". As `ConfigPanel._build_config` today
(`config_tab.py:305-360`): read the fields, build a `ConfigDraft`, call `InferenceScreen.new_draft`
(`core/gui/screens/inference_screen.py:148-165`), after 5.3's question. For one of your own models that
inference cannot take, the button is greyed with the reason ("This model has external forcing, so
parameter inference cannot use it."); today that is a text-pane line (`config_tab.py:204-213`).

**Afterwards**, in place of the text-pane line (`config_tab.py:342-355`): "In use: Nadrowski.
Frequency response, 6 driven recordings in 12 slots, lock-in limit 20 drive cycles. Rotation on.
Units kHz ms nm pN, from the model's file." Rotation is always stated, on or off (W020, W142). Next
step: "Go to Prior".

**Gone from this page:** the "Hardware" group (`config_tab.py:108-119`, `:128-165`; Settings',
U9) and the session line (`config_tab.py:362-387`; the chain says it).

### 8.3 Step 2: Prior

Sentence: "The prior is the range of parameter values training draws from: the bounds file's
limits, cut down to the values for which the model is stable."

Two views, each with one main button: "Use a saved prior" and "Build a new prior". The heading's
plain button (5.1) switches between them ("Build a new prior instead…", "Use a saved prior
instead…"). The page opens on the saved view when a saved prior exists (a choice the owner has not
seen; it joins U14's split buttons to U16's one main button).

| On screen | Today | Key | Place | Built-in value, from | Reaches | Plan |
|---|---|---|---|---|---|---|
| "Bounds file" | "Bounds", "Use file" (`prior_tab.py:70-75`) | `bounds` | M | the remembered file, else nothing selected | `ConfigDraft.make_config(bounds_path)` (`session.py:27-37`), then `make_sim_config(bounds_file=)` (`cli.py:273`, `:311-312`) | yes |
| "Edit the limits by hand" | "Edit values" (`prior_tab.py:72-73`, `:145-166`) | `bounds` | R | off at every launch (`prior_tab.py:376`) | `make_config(bounds_dicts=)` (`prior_tab.py:192`; `cli.py:306-307`) | no |
| "Saved prior" | the entries of "Prior" (`prior_tab.py:76-77`) | `prior` | M | the remembered prior | `build_prior(cfg, ref, False)` (`orchestrator.py:597-598`, `:663-671`) | no |
| "Name" (build; required) | none; "Save" renames afterwards | `name` | N | blank | `build_prior(name=)` (`orchestrator.py:599`, `:652`) | no |
| "Note" (build) | none | `note` | N | blank | `note=` (`orchestrator.py:599`, `:726`) | no |
| "Sweep rounds" | "Global rounds" | `num_iterations` | R | 50 (`config.py:632`) | `num_iterations` | yes |
| "Candidates per round", tick "Automatic" | "Candidates per round (0 = auto)" | `sweep_batch` | R | Automatic, passed as 0 (`config.py:643`): the computer's batch size (`orchestrator.py:704`) | `sweep_batch` | yes |
| "Stable sets to collect" | "Max accepted sets" | `max_sets` | R | 175,000 (`config.py:650`) | `max_sets` | yes |
| "Random-walk step" | "Random-walk step" | `walk_step` | R | 0.01 (`config.py:654`) | `walk_step` | yes |
| "Stability test length", unit "model time units" | "Stability duration (ND units)" | `stability_units` | R | 1000 (`config.py:628`) | `stability_units` | yes |
| "Smallest cluster" | "Min cluster size" | `min_cluster_size` | R | 50 (`config.py:671`) | `min_cluster_size` | yes |
| "Cluster density" | "Min samples" | `min_samples` | R | 10 (`config.py:672`) | `min_samples` | yes |

The seven build settings keep their rule table (`prior_tab.py:29-37`), fields
(`prior_tab.py:105-114`, `:129-132`; today's labels at `fields.py:74-80`) and arguments
(`orchestrator.py:600-604`, resolved at `:685-700`). They are shown only in "Build a new prior": a
load reads none of them (`prior_tab.py:259-260`), and today they stay live beside a load (W046).

**What the bounds file is** (W014). Under the list, read with `file_manager.parse_bounds_file`
(`core/Helpers/file_manager.py:208`) when a file is selected:

- "13 parameters: 10 of the model and 3 scales (x_scale, t_scale, T)", then the names in the file's
  order.
- "It has a forcing section." or "It has no forcing section."
- "With the kind of observation applied in step 1, a posterior trained on it reads: a frequency
  response." The other two readings are "a spontaneous recording and one driven recording" and "a
  spontaneous recording only". The rule is `SimConfig.observation_mode`
  (`core/sim_config.py:367-393`).

After the prior is applied, two text-pane lines of today are shown here as "Attention": the units
messages of `cfg.check_unit_consistency()` (`prior_tab.py:207-208`) and the force-scale sentence
for a file that lists f_scale but is read as a spontaneous recording only (`prior_tab.py:220-224`).

**Limits typed by hand.** The grid keeps what was typed while the page moves between the file and
hand entry; it is reloaded only when another bounds file is selected or step 1 is applied (a choice
the owner has not seen; today any switch reloads it, `prior_tab.py:141-166`). Under it: "Typed
limits are not saved as a file. Records made from them name no bounds file." A file with a
malformed line is refused beside the list; today `bounds_grid.load` runs outside the handler's
guard (`prior_tab.py:166`; W049).

**The saved prior's facts**, under the list, from `default_store().get("prior", id)`: name, when it
was made, its model, the number of parameters (`config["param_keys"]`, the model's and the scales,
`core/artifacts/manifest.py:276`; `body["gmm"]["param_keys"]` holds the model's alone,
`orchestrator.py:737`), and the bounds file it was built on (`inputs["bounds"]`, a path and a
content hash recorded by `store.py:576-594` from `SimConfig.sources`, `cli.py:349`). A prior made for another model greys the button with that
reason; the engine would refuse it (`store.py:1275-1276`).

**"Before you start"**

| Fact | Example | Read from |
|---|---|---|
| The bounds file a saved prior was built on, against the one selected | "Fine. Built on master_tier1.txt, the bounds file selected." / "Attention. Built on master.txt; the bounds file selected is master_tier1.txt. PRISM's own check compares only the model's parameters and their limits, which these two files share, so the load goes through." / "Attention. Built on master_tier1.txt as it was then; the file has been edited since." / "Information. Built on limits typed by hand." | `inputs["bounds"]` against `provenance.file_ref(selected, relative_to=config.RESOURCES_ROOT)` (`core/artifacts/provenance.py:69-99`); the engine's four comparisons are `store.py:1275-1291` |
| The simulation sets made with a saved prior | "2 saved simulation sets were made with this prior; the largest holds 10,000 batches." | `training_checkpoint.checkpoints_using_prior` (`core/SBI/training_checkpoint.py:303-326`) |
| What a build costs | "Tests 102,400 candidates (50 rounds of 2,048), then walks outward until 175,000 stable sets are collected." | the arithmetic of `_sync_sweep` (`prior_tab.py:281-313`), recomputed when the page opens and when a setting changes, on the memoised device reading of `base.py:77-82` (W107) |
| How long a build takes | "About a minute and a half: the last build on this computer took 1 min 32 s." | `history.estimate(…)` (6.3.3) |
| What a new prior means for saved simulations | "A new prior cannot continue a saved simulation set: the 2 saved sets for this model were made with other priors." | the prior is part of a set's identity (`identity.py:90`) |

Three choices the owner has not seen:

- The bounds-file line never blocks. The three shipped Nadrowski files share the part PRISM checks
  (W138), so one prior is rightly used with all three; the engine's own check is left to the audit
  (U18). "…which these two files share, so the load goes through." is said only when the record's
  parameters and limits (`body["gmm"]`) equal the selected file's; otherwise the line ends "…which
  differ between these two files, so PRISM will refuse the load." and the refusal follows the click
  (5.7).
- The cost sentence also gives the steps each candidate is integrated over (`prior_tab.py:303-304`)
  when the configuration in use was built on the selected bounds file; otherwise it leaves them out
  and never prints "?" (W108).
- The time in the build button is the last finished build's; with none, the bracket is left out and
  the line reads "No prior has been built on this computer yet, so the time is not known."

**Main buttons.** "Load this prior": `dispatch(orchestrator.build_prior, cfg, id, False)`.
"Build a new prior (about a minute and a half)": `dispatch(orchestrator.build_prior, cfg, None,
True, …)` with the name, the note and the seven settings. In both, `cfg` is the candidate of 5.3
(`prior_tab.py:199-206`). When step 3 holds a posterior, 5.3's question comes first (U14; today
nothing is asked, `prior_tab.py:168-245`).

**Afterwards:** "In use: retrain_prior. Built on master_tier1.txt, 13 parameters. Saved as a
record." with the figure the stage hands over ("Prior" or "Prior (loaded)", `orchestrator.py:670`,
`:729`). "Name it below to keep it" (`prior_tab.py:317-318`) goes (W109, W155). The saved list is
re-read and the new prior selected (W015). Next step: "Go to Training".

### 8.4 Step 3: Training (setting up)

Sentence: "Makes the simulation set, then fits the network to it. The result is saved as a
posterior: the trained network that turns an observation into parameter estimates with their
uncertainty."

The page is the training form. The heading's plain button "Use a saved posterior instead…" opens
the load view (as drawn in `structure-chosen-rail-2-setup.png`), where it reads "Set up a training
instead…".

| On screen | Today | Key | Place | Built-in value, from | Reaches (`build_posterior`, `orchestrator.py:786-802`) | Plan |
|---|---|---|---|---|---|---|
| "Batches" | "Batches" (`posterior_tab.py:69`) | `num_runs` | M | 5000 (`config.py:282`); remembered (U15) | `num_runs` | yes |
| "Simulations per batch", tick "Automatic" | "Max rows per batch (0 = auto)" (`posterior_tab.py:70`) | `run_size_cap` | M | Automatic, passed as 0 (`config.py:328`): the computer's batch size, capped as `orchestrator.py:415-419`; remembered | `run_size_cap` | yes |
| "Network width" | "Hidden features" (`posterior_tab.py:88`) | `hidden_features` | M | 128 (`config.py:363`), at every launch | `hidden_features` | yes |
| "Network depth" | "Transforms" (`posterior_tab.py:89`) | `num_transforms` | M | 8 (`config.py:364`), at every launch | `num_transforms` | yes |
| "Continue or start new" | none (`fields.py:61`) | `resume` | M | "Continue the saved simulation set if there is one" | `resume`: "auto", "require" or "never" (`orchestrator.py:799`, `:957`) | no (7.1.4) |
| "Name" (required) | none; "Save" renames afterwards | `name` | N | blank | `name` (checked at `:892`) | no |
| "Note" | none | `note` | N | blank | `note` | no |
| "Learning rate" | "Learning rate" (`posterior_tab.py:90`) | `learning_rate` | R | 0.001 (`config.py:368`) | `learning_rate` | yes |
| "End the fit after … epochs without improvement" | "Early-stop patience" (`posterior_tab.py:91`) | `stop_after_epochs` | R | 20 (`config.py:369`) | `stop_after_epochs` | yes |
| "Epoch limit", tick "No limit" | none (`fields.py:86`) | `max_num_epochs` | R | "No limit": nothing is passed and the stage takes `config.py:370` (`orchestrator.py:935`) | `max_num_epochs` | yes |
| "Expected epochs (for the time estimate only)" (B9) | none | `expected_epochs` (7.1.3) | R | blank | no stage: the run's `spec` (8.1) | yes |
| "Save the simulations every … batches", tick "Never" | none (`fields.py:60`) | `checkpoint_every` | R | 50 (`config.py:303`); "Never" is 0 | `checkpoint_every`; also given to `training_preview` and `fresh_run_near_misses` (`orchestrator.py:422-423`, `:481-482`) | yes |
| "Rotation: simulations per perturbation" | "Ensemble per perturbation" (`posterior_tab.py:102`) | `fisher_m` | R | 48 (`config.py:405`) | `fisher_m` | yes |
| "Rotation: step" | "Central-difference step" (`posterior_tab.py:103`) | `fisher_dz` | R | 0.1 (`config.py:406`) | `fisher_dz` | yes |
| "Rotation: parameter sets averaged" | "Operating points" (`posterior_tab.py:104`) | `fisher_points` | R | 8 (`config.py:410`) | `fisher_points` | yes |
| "Saved posterior" (load view) | the entries of "Posterior" (`posterior_tab.py:40-42`) | `posterior` | M | the remembered posterior | `build_posterior(cfg, prior, ref, False, accept=…)` (`:976-992`) | no |

**Sentences on the page, beside their settings**

- Under width and depth: "Width and depth open at 128 and 8 each time PRISM starts. A plan can set
  them." (U8).
- "Continue or start new" has three choices: "Continue the saved simulation set if there is one"
  (auto); "Only continue: do not start if there is nothing to continue" (require); "Only start new:
  do not start if a saved simulation set would be continued" (never).
- Epoch limit, its two traps (W035, W036; `orchestrator.py:835-836`): "A limit of N lets the fit
  run N + 1 epochs." and "A fit that reaches the limit keeps the network of its last epoch, which
  may not be its best."
- "Expected epochs" (B9; a choice the owner has not seen): "Used only to say when the fit should
  end. It never limits the fit. Left blank, PRISM takes the epochs of the last finished training of
  this kind in Records." With neither: "No earlier training of this kind: the run's page will say
  only when the fit cannot end before." The page exposes the value as B9 resolves it (typed; else
  the last training's, by 6.3.3's rule; else none) with its source; the "How long" fact and the
  timing check read it (9.6; C28).
- "Never": "No simulation set is kept or continued. A stop or a failure during simulation loses
  every simulation; the fit cannot be tried again without simulating again; and the check 'Use of
  each part of the input' cannot be run on the posterior." (`orchestrator.py:837-840`, `:1083`,
  `:1433-1434`; `core/diagnostics/ablation.py:103-109`). With "Never", "Continue or start new" is
  fixed at its first choice and greyed with the reason, so the refusal at
  `orchestrator.py:958-965` is not reachable.
- The three rotation settings are used only when the rotation is computed afresh
  (`orchestrator.py:1239-1245`). When the run continues a saved set, or rotation is off in step 1,
  they are read-only with "Not used: this training reuses the saved simulation set's rotation." or
  "Not used: rotation is off.", are not judged and are not passed (W060; today a blank one stops a
  fit that would never read it, `posterior_tab.py:178-180`).

**"Before you start"**

| Fact | Example | Read from |
|---|---|---|
| What the run uses | "Model Nadrowski; bounds file master_tier1.txt; prior retrain_prior; frequency response with 12 slots; rotation on." | `session.cfg` (`model`, `sources["bounds"]`, `observation_mode`, `chi_k_pad`, `reparam_rotate`) and `session.inf_prior` |
| How many simulations | "10,000 batches of 2,048 make 20,480,000 simulations. Each simulates one spontaneous recording and 2 to 12 driven recordings." | `training_preview` (`n_runs`, `width`; `orchestrator.py:481-564`); the second sentence replaces `base.py:150-153` |
| Continue or new | "Fine. Continues the saved simulation set: 3,989 of 10,000 batches are done." / "Fine. No simulations this time: the saved 20,480,000 are used." / "Information. Starts a new simulation set; none is saved for this model yet." / "Attention. Starts a new simulation set. 1 saved set for this model does not match these settings:" then every difference (8.7) / with "Never": "Attention. The simulations of this training are not saved, so no saved simulation set is continued." | `TrainingPreview.checkpoint`, `batches_done` (`orchestrator.py:448-478`) and `sibling_diffs` (E2, 11.3) |
| The fit cannot be resumed | "Attention. A fit cannot be continued. Stopping it loses the whole fit; the simulations are kept." With "Never" the second sentence is "Stopping it loses the whole fit and the simulations, which are not saved." | fixed, but for the cadence |
| How long | "Information. Simulation: about 28 hours at 9.7 s per batch, the pace of the training retrain (5 Oct 2026, 09:10). Fit: about 130 epochs of about 63 minutes." / "Information. No time estimate: no training of this size has run on this computer, and the timing of a short fit has not been run." | `history.estimate(…)` (6.3.3); "Expected epochs" as resolved (above) |
| The memory limit, and what the last run measured | 7.2.3's sentences (the last run's is 7.2.2's), led by "Information." | `this_computer.memory_limit_in_force()`; the run history |
| The plan and each departure from it | 7.1.9's sentences | `plan_facts(headings)` |
| The pre-run checks | "Fine. Frequency band: Passed, 3 Oct 2026, 14:02. Unusable driven recordings: audit: Passed, 3 Oct 2026, 14:40." / "Attention. Unusable driven recordings: audit: Not run yet." / "Attention. Frequency band: Failed, 3 Oct 2026, 14:02. Starting will ask first." / "Information. The two pre-run checks do not apply: they need the frequency response." | Checks' reading of its checks 1 and 2 for the prior in use (9.3) |

Today's memory line (`base.py:158-185`) is dropped (U10; W017, W024, W025). Of these pages, part
7's memory lines stand only on Training and the Narrowing round, which simulate training batches on
the card (C13);
a prior build, which the limit does not govern (`core/SBI/prior_screen.py:16`), and an inference
carry none (a choice the owner has not seen). The checks line is led by "Fine." only when both
read "Passed", by "Information." when both read "Does not apply", otherwise by "Attention.", a
check not yet run included. It never greys "Start training"; after a "Failed" the click asks first
(C12; a choice the owner has not seen).

When "Only continue" is chosen and there is nothing to continue, or "Only start new" and there is,
the start button is greyed with that reason. E6 gives the stage's two refusals for a continue
choice (`orchestrator.py:958-965`, `:1111-1112`) the key `resume` (11.7; W094, W099).

**Main button:** "Start training". `dispatch(orchestrator.build_posterior, cfg, prior, None, True,
…)` with `new_run` and every argument of the table, the rotation three only when used. Today
`name`, `note`, `max_num_epochs`, `checkpoint_every` and `resume` are never passed
(`posterior_tab.py:148-150`; W187).

**Question, after a pre-run deciding check failed** (C12), asked before the one below:

> Title "Start training although a check failed?" Text: each failed check as its fact gives it,
> "Frequency band: Failed, 3 Oct 2026, 14:02.", then "Step 4, Checks, says why." Buttons: "Start
> training anyway"; "Cancel" (the safe answer, preselected).

**Question, before simulating from zero** (U10; replaces "This starts a NEW run",
`posterior_tab.py:183-238`, which is asked only when exactly one setting differs, W023):

> Title "Start a new simulation set?" Text: "This training will simulate 10,000 batches from zero.
> Saved simulation sets for this model exist, and none matches these settings." Then, for each one,
> under its label from Records (10.1): "Simulation set for retrain_prior (2 Oct 2026, 09:10; 10,000
> batches, complete) differs in:" and one line per differing setting by its on-screen name with
> both values (8.7). Then: "To continue one of them, cancel and set these back." Buttons: "Start
> training on a new simulation set" (passes `new_run=True`); "Cancel" (the safe answer).

It is asked whenever the preview says the run would not continue its own set and `sibling_diffs`
holds a set of this model, one whose differing entries do not include `model` (U10 to the letter,
B11). A set made by a narrowing round is listed like the others; its difference reads "Made by a
narrowing round: a training never continues it." With "Never", the preview reads nothing
(`orchestrator.py:541-542`), so the page asks it once more at the built-in cadence to learn which
sets exist; a set that reading would continue also counts (U10) and heads the list with "No setting
differs. It is not continued because the simulations of this training are not saved."; the text
then ends "…exist, and none will be continued." and the first button reads "Start training without
saving the simulations". If the stage still refuses with the key `new_run` (a set appeared in between), the
page reads the preview again and asks this question; the engine's sentence, which today ends "pass
new_run=True" (`orchestrator.py:578-587`; reworded by E11), is not shown (W114, W154). If the
preview cannot read the saved sets (`checkpoint_error`, `orchestrator.py:459-460`), "Before you
start" says so as "Attention" and the run is not blocked, as now (`posterior_tab.py:201-213`). The
labels and differences are the shared dialog's list (C10; 4.6): one to a line under the sentence,
scrolling beyond eight, never behind a link.

**Load view.** Facts under "Saved posterior", from its record: kind of observation, network size,
simulations, the prior it was trained with (`parents["prior"]`, `orchestrator.py:1432`), the bounds
file (`inputs["bounds"]`), and "from a narrowing round" where `body["amortized"]` is false.
"Before you start" compares the prior: "Fine. Trained with retrain_prior, the prior in use." or
"Attention. Trained with retrain_prior; the prior in use is other_prior. Calibration and a
narrowing round will refuse this pair. Load retrain_prior in step 2 first." (a choice the owner has
not seen; today nothing compares them, W091). Button "Load this posterior".

**Question, loading a posterior from a narrowing round** (kept from `posterior_tab.py:240-285`,
reworded):

> Title "Load a posterior from a narrowing round?" Text: "retrain_round1 was narrowed around the
> observation cell_a. It is valid only near that observation; elsewhere it has seen no training
> simulations and gives confident answers that are wrong." Then: "If you load it: calibration runs
> on the narrowed prior; inference on any other observation asks first; a further narrowing round
> can only be drawn around the same observation." Buttons: "Load this posterior" (passes
> `Accept(truncated=True)`); "Cancel" (the safe answer).

The observation's name is read from the record the posterior's `parents["observation"]` names
(`orchestrator.py:1437-1438`).

**Afterwards** (through this page's `on_result`): "In use: retrain_posterior. 131 epochs, network
256 by 10, on 20,480,000 simulations, prior retrain_prior. Saved as a record." from
`manifest.body["training"]` (`orchestrator.py:1469-1482`), with the "Training loss" figure
(`orchestrator.py:1429-1431`). When the epochs trained exceed a set limit
(`config["max_num_epochs"]`, `orchestrator.py:1455`): "The fit reached the epoch limit; the network
kept is its last epoch's." The form's required name is now empty (5.5; W021). Next step: "Go to
Checks".

### 8.5 Step 5: Inference

Sentence: "Runs the posterior on one observation and reports each parameter's estimate with its
uncertainty."

Two modes, as a switch at the top (U16): "A simulated cell" and "Your own recordings" (today a
list reading "Simulated (cell ground truth)" and "Experimental data", `infer_tab.py:83-89`). In the
second, the form shown is the one the posterior reads, named on the page ("This posterior reads a
frequency response."); the rule is unchanged (`infer_tab.py:220-229`, `:375-384`). With no
posterior in use (B7) the second mode shows, in place of a form, "The form depends on the
posterior. Load or train one in step 3." (a choice the owner has not seen).

| On screen | Today | Key | Place | Built-in value, from | Reaches | Plan |
|---|---|---|---|---|---|---|
| "Cell file" (simulated; the key's one label on every page, ruling D2) | "Cell", "Use file" (`infer_tab.py:98-101`) | `cell` | M | the remembered file | `simulated_inference(cell=)` (`orchestrator.py:2448-2452`) | no |
| "Edit the values by hand" | "Edit values" (`infer_tab.py:98-99`) | `cell` | R | off | `gt_values=` (`infer_tab.py:404`, `:477`) | no |
| "Recording length", unit "s" | "T_obs (s)", three boxes seeded 1.0 (`infer_tab.py:96`, `:108`, `:120`) | `t_obs` | M | blank (U15, B12) | `T_obs_s` (simulated), `RecordingSet(T_obs_s=)` (`core/SBI/observations.py:21-30`) | no |
| "Spontaneous recording", one name on every form (a choice the owner has not seen) | "Spontaneous" and "Passive" (`infer_tab.py:111`, `:123`; `fields.py:129-130`) | `recording_spont` | M | the remembered path | `RecordingSet(spont=)` | no |
| "Driven recordings" (table: file, "at", frequency, "Hz") | "Forced probes" (`infer_tab.py:128-131`; `rows.py:22-75`) | `recording_probe` | M | the remembered rows (U15) | `RecordingSet(forced=((file, Hz), …))` (`infer_tab.py:429-431`, `:481-482`) | no |
| "Drive amplitude", unit "N" | "Drive F₀ (N)", seeded 1.0 (`infer_tab.py:121`) | `chi_f0_si` | M | blank (U15, B12) | `RecordingSet(F0_si=)` | no |
| "Driven recording" (one driven recording) | "Forced" (`infer_tab.py:112`) | `recording_forced` | M | the remembered path | `RecordingSet(forced=((file, None),))` (`infer_tab.py:486`) | no |
| the drive's own rows, labelled as today (`infer_tab.py:34-38`) | seeded 0.0 (`infer_tab.py:371`) | `drive_amplitude`, `drive_frequency`, `drive_phase` | M | blank | `RecordingSet(forcing_params_si=)` | no |
| "Posterior draws" (new on screen: a choice the owner has not seen) | none (`fields.py:72`) | `n_samples` | R | 1000 (`orchestrator.py:2152`) | `n_samples` of both compositions (`:2451`, `:2537`) | yes |
| "Name of the inference" | none | `name` | N | blank | `name` (`:2452`, `:2538`; checked at `:2483`, `:2554`) | no |
| "Name of the observation" (required: 5.5's rule, a choice the owner has not seen; U16 says it can be named) | none | `observation_name` | N | blank | `observation_name=` of both compositions (E4, 11.5) | no |
| "Note" | none | `note` | N | blank | `note`, the inference's; the observation's stays empty and is edited in Records (11.5) | no |

**The driven-recordings table.** One row stays one widget holding a file and its frequency
(`rows.py:22-35`). Buttons: "Add a driven recording" (today "+ Add probe") and "Plan the driven
recordings…" (today "Plan probes…", `infer_tab.py:289-358`). Planning still measures the cell's own
frequency from the spontaneous recording, fills only blank frequencies and never a typed zero
(`infer_tab.py:48-57`, `:331-338`); its answer is shown on the page instead of the text pane: one
sentence with the measured frequency and the band in hertz, and beside each row one of the four
readings of `chi.probe_verdict` (`infer_tab.py:351`; `core/SBI/chi.py:135-200`): used, "Fine, 23
drive cycles"; cut at the lock-in limit, "Fine, 20 drive cycles: only the first 3.2 s are used";
too short, "Unusable: 1.4 drive cycles, fewer than the 2 needed. Record at least 0.4 s at this
frequency."; refused, "Error." with the verdict's reason, a refusal sentence E11 rewords
(`core/SBI/observations.py:258`). The window words the first three from the verdict's numbers, not
its reason, which says "probe" and "MASKED" and is only a warning in the engine, so E11 leaves it
(`observations.py:259-271`). If more rows are remembered than the
posterior has slots, all are shown and the page says "This posterior takes at most 12 driven
recordings; remove 2."; none is dropped silently.

**Blank at every launch** (U15, B12): "Recording length" in both modes and "Drive amplitude". The
drive's own rows open blank too, not at 0 (a choice the owner has not seen). Under the length: "The
posterior was trained on lengths from 1 to 60 s." (`config.py:223-224`).

**"Before you start"**

| Fact | Example | Read from |
|---|---|---|
| What it runs on | "Posterior retrain_posterior: frequency response, up to 12 driven recordings." | `session.posterior.manifest` |
| A narrowed posterior | "Attention. This posterior was narrowed around the observation cell_a and is valid only near it." | `posterior.posterior.truncation` is set (today's test, `infer_tab.py:521-522`); the observation's name from the record `parents["observation"]` names (`orchestrator.py:1437-1438`) |
| The cell fits the bounds file | "Error. The cell gives no value for T." | `cli.validate_gt_file` (`cli.py:52-82`), run when the cell is picked, as now (`infer_tab.py:189-207`) |
| The length is inside the trained range | "Attention. 75 s is above the 60 s the posterior was trained on; it may extrapolate." | the rule of `orchestrator.py:2433-2445`, said before the run |
| The check against what training saw | "Information. The cell is not checked against what training saw: the prior in use is not the one this posterior was trained with." | see below |

The simulated path checks the cell against the prior it is handed (`orchestrator.py:2515-2519`).
Today the page hands over whatever prior the session holds (`infer_tab.py:478`). New: it hands it
over only when its id is the posterior's `parents["prior"]`, and otherwise passes none and says so
(W091).

**Main button:** "Run inference". `dispatch(orchestrator.simulated_inference, cfg, posterior,
length, …)` or `dispatch(orchestrator.experimental_inference, cfg, posterior, RecordingSet(…), …)`
(`infer_tab.py:460-488`), each with `prior=`, `accept=` and the table's arguments. The button is
greyed with its reason while a field is blank or the cell does not fit; today that reason is a
tooltip (`infer_tab.py:516-518`).

**Question, with a narrowed posterior** (replaces the tick "Run on a different observation",
`infer_tab.py:149-154`, `:490-499`):

> Title "Run on another observation?" Text: "This posterior is valid only near the observation it
> was narrowed around, cell_a." For a simulated cell: "A cell simulated now has new noise, so it is
> never that observation." For recordings: "If these are the same recordings with the same
> description, PRISM recognises the observation and nothing is recorded." Then: "Elsewhere the
> result is an extrapolation. The inference records that you accepted this." Buttons: "Run
> inference anyway" (passes `Accept(other_observation=True)`); "Cancel" (the safe answer).

It is asked at every such run, in both modes, and never remembered (a choice the owner has not
seen). The engine records the acceptance only when the observation really differs
(`orchestrator.py:2192-2213`).

**Afterwards** (U16), from the inference record (`orchestrator.py:2380-2399`):

- a ruled table "Parameter", "5 %", "50 %", "95 %", "Unit" from `results["posterior_summary"]`
  (`orchestrator.py:2119-2132`), in the record's order, an assumed input marked "assumed input";
  the record holds no unit, so "Unit" is the configuration's (`core/sim_config.py:543`, `:548`,
  `:554`; kelvin for an assumed temperature) and blank where a parameter has none, or where
  Records hosts the block with no configuration in use (8.1); for a simulated cell a column "True
  value" from `results["ground_truth"]` (a choice the owner has not seen);
- the corner plot ("Posterior corner", `orchestrator.py:2241`); beneath it the record's other
  figures ("Posterior predictive check", `:2317`; the overlay figures, up to five and written only
  when they can be drawn, `:2364`, `core/SBI/overlay.py:281-283`; "Eye test", `:2377`) and the
  observation's trace;
- "Of the observation's features, 93 % fall inside what the posterior predicts (90 % interval)."
  from `results["ppc"]["coverage_90"]`, left out when the record holds none;
- "Saved as the inference … and the observation …".

Next steps: "Narrow around this observation" (opens step 6 with it selected) and "Run another
inference".

### 8.6 Step 6: Narrowing round

Sentence: "Trains a second posterior for one observation only: the prior is cut down to the region
the posterior in use finds believable for that observation, and training is repeated there
(truncated sequential neural posterior estimation, TSNPE). The result is valid only near that
observation."

| On screen | Today | Key | Place | Built-in value, from | Reaches (`tsnpe_round`, `orchestrator.py:2565-2574`) | Plan |
|---|---|---|---|---|---|---|
| "Observation" | "Observation" (`tsnpe_tab.py:58-59`) | `observation` | M | the observation the session holds (made in step 5, or loaded from Records), else the remembered one | `default_store().load_observation(cfg, id)` (`tsnpe_tab.py:147-148`), then `observation` | no |
| "Region level" | "HPD level" (`tsnpe_tab.py:68`) | `hpd_level` | M | 0.999 (`core/SBI/truncate.py:71`) | `level` | yes |
| "Directions narrowed" | "Directions truncated" (`tsnpe_tab.py:69`) | `n_directions` | M | 5 (`truncate.py:73`) | `n_directions` | yes |
| "Batches" | "Batches" (`tsnpe_tab.py:84`) | `num_runs` | M | 5000; remembered, apart from Training's (B12) | `num_runs` | yes |
| "Simulations per batch", tick "Automatic" | "Max rows per batch (0 = auto)" (`tsnpe_tab.py:85`) | `run_size_cap` | M | Automatic; remembered (B12) | `run_size_cap` | yes |
| "Network width", "Network depth" | none (W026, W194) | `hidden_features`, `num_transforms` | M | 128 and 8 (`config.py:363-364`), or the plan's | `hidden_features`, `num_transforms` | yes |
| "Learning rate"; "End the fit after … epochs without improvement" | none | `learning_rate`, `stop_after_epochs` | R | 0.001; 20 (`config.py:368-369`) | `learning_rate`, `stop_after_epochs` | yes |
| "Epoch limit"; "Save the simulations every … batches" | none | `max_num_epochs`, `checkpoint_every` | R | as in 8.4 | the same two arguments | yes |
| "Name" (required), "Note" | none (W053) | `name`, `note` | N | blank | `name`, `note` (checked at `:2602`) | no |

The page always passes the network settings it shows, so what is on screen is what trains; today
the tab sends none and the round falls back to 128 by 8 (`tsnpe_tab.py:157-161`;
`orchestrator.py:2648-2655`, `:936-939`).
Under them: "The posterior in use was trained at 256 by 10." (from its `body["training"]`,
`orchestrator.py:1469-1471`), and as "Attention" when the two differ. There are no rotation
settings: "Rotation: the posterior's own is reused." (the round never computes one,
`orchestrator.py:1125-1206`). There is no "Continue or start new" and no "Expected epochs": "Before
you start" says what a round continues (C11) and B9 gives the field to Training. `resume` is
passed as "auto", as today (`orchestrator.py:2573`), and the fit's estimate is 6.3.3's rule for a
run with no typed value. The region's draw is seeded (`core/SBI/truncate.py:347-360`), so a round
started again with the same posterior, observation, region level, directions and budget draws the
same region and, under "auto", continues its own earlier set (`orchestrator.py:1095-1096`): the
case C11's sentence names. The page cannot tell before the click which it will be: drawing the
region beforehand to preview it is not designed in (C11). The run's page shows the batches already
there (6.3).

**The observation.** When an inference finishes, the list moves to its observation (U16). Today it
keeps the earlier one while the text says the new one is ready (`infer_tab.py:504-505`;
`tsnpe_tab.py:205-208`; W009). If the page then shows a pick the user did not make, a line says
so: "Changed to the observation just made, cell_b. Before: cell_a." With a narrowed posterior in
use, only observations whose recorded digest (`body["x_obs_digest"]`, `orchestrator.py:380`) equals
the posterior's can be chosen; the others are listed greyed with "Cannot be used: this posterior
was narrowed around another observation", because the engine refuses them with no override
(`orchestrator.py:2622-2635`; a choice the owner has not seen: today the refusal comes after the
click).
"Region level" below 0.99 shows the engine's warning before the run (`orchestrator.py:2617-2621`):
"Attention. Tighter than 0.99. Narrowing removes part of the prior for good; no later round gets it
back." "Directions narrowed" is checked while typed against the number of parameters
(`orchestrator.py:2610-2614`).

**"Before you start"**

| Fact | Example | Read from |
|---|---|---|
| What it uses | "Posterior retrain_posterior and its prior retrain_prior; observation cell_b, a simulated cell, 4.5 s." | the session; the observation's record |
| The prior is the posterior's own | "Fine." / "Error. This posterior was trained with retrain_prior; the prior in use is other_prior. Load retrain_prior in step 2." (greys the button) | `parents["prior"]` against `session.inf_prior.id`; the stage's refusal is `orchestrator.py:1168-1174` |
| How many simulations, and what is continued (C11) | "Information. Simulates 10,240,000 simulations in the round's own simulation set. A round never continues a training's simulations; if this same round was run before with the same settings, its saved simulations are continued." | `training_preview` (`n_runs`, `width`) |
| What the result is | "Attention. The posterior made here replaces the one in use in steps 3 to 5 and is valid only near this observation." | fixed (`tsnpe_tab.py:164-177`) |

The fit-cannot-be-resumed, time, memory, plan and checks lines are as in 8.4. When the page was
filled by "Continue from batch N" (8.1), the simulations line reads "Continues this round's
simulation set from batch N." and `resume="require"` is passed: a region that came out different
is refused before anything is simulated (`orchestrator.py:1111-1112`).

**Main button:** "Start the narrowing round". `dispatch(orchestrator.tsnpe_round, cfg, posterior,
prior, observation, …)` with the table's arguments, `resume` and `new_run=False`. Today the handler
returns silently when something is missing and the button is grey with nothing beside it
(`tsnpe_tab.py:134-135`, `:209-210`; W058).

**Nothing is asked about starting from zero** (B11). One case is shown as a question, in place of
the tick "Start a new simulation even if a cache one setting away exists" (`tsnpe_tab.py:96-103`)
and of the "Check your inputs" dialog for this one key (a choice the owner has not seen; B11, C11). The
region is drawn inside the stage, so the window cannot look beforehand: when the stage refuses with
the key `new_run`, seconds after the click and before anything is simulated
(`orchestrator.py:1110-1114`), the page asks:

> Title "Start the narrowing round beside a saved simulation set?" Text: the engine's sentence,
> which names the saved simulation set, its batches and the one setting that differs with both
> values (`orchestrator.py:567-587`), then "Starting now simulates this round from zero and leaves
> that set as it is." Buttons: "Start the narrowing round" (dispatches again with `new_run=True`);
> "Cancel" (the safe answer).

The engine's sentence is shown as it stands (C11). After E11 it is in U7's words but still names
the set by its folder, the setting by the engine's name (for example n_runs) and ends "pass
new_run=True" (11.13); structured facts on the refusal are beyond E1 to E12 (11.3). Checked by: a
stage refusal with the key `new_run` opens this question and no "Check your inputs" dialog;
"Cancel" dispatches nothing.

**Afterwards:** the handler installs the round's posterior, as now (`tsnpe_tab.py:164-177`). The
page shows "In use from step 3 on: retrain_round1, narrowed around cell_b. The region kept 12 % of
the prior's draws." (`body["training"]["tsnpe_acceptance"]`, `orchestrator.py:1476`) and "The
posterior it came from, retrain_posterior, stays in Records; load it in step 3 to go back." Step 3
then shows that name with "narrowed" (3.4; W022). Next steps: "Check this posterior", which opens
Checks, the same page and deciding checks as for any posterior (9.4), and "Run inference on the
same observation".

### 8.7 A saved simulation set's differences, by on-screen name

Used by the question and the "Before you start" line of 8.4; it is the table 11.3 places in
`core/gui/fields.py`. The settings are the 34 entries of a set's identity (`identity.py:87-122`), read from
`sibling_diffs`. Lists are compared entry by entry and only the entries that differ are shown, by
parameter name, never cut short. Checked by: a test takes the keys of an identity built by
`SimulationIdentity.from_cfg` and fails when one is neither `model` nor in a row here.

| Identity entries | Shown as | Where it is changed |
|---|---|---|
| `n_runs` | "Batches" | step 3 |
| `run_size` | "Simulations per batch" | step 3 |
| `prior_fingerprint` | "Prior", each side by the prior's label (10.1; `store.find_prior_by_fingerprint`, `store.py:1238-1244`) or "a prior no longer in Records" | step 2 |
| `param_keys`, `nd_lows`, `nd_highs`, `rescale_lows`, `rescale_highs` | "Bounds file: parameters", "Bounds file: limits of …" | step 2 |
| `spontaneous_only` | "Bounds file: forcing section" | step 2 |
| `chi_mode`, `mode` | "Kind of observation" | step 1 |
| `chi_k_pad` | "Slots for driven recordings" | step 1 |
| `chi_max_cycles` | "Lock-in limit" | step 1 |
| `reparam_rotate` | "Rotation" | step 1 |
| `units_sha256`, `dt_exp`, `t_min_exp`, `t_max_exp` | "Units"; the three times follow the units (`cli.py:343-345`) and fold into that line | step 1, or the model's units file |
| `t_scale_bounds`, `dt_nd_min`, `n_grid`, `steady_idx` | "Time grid", with the entry's value on each side | it follows the bounds file's limits on t_scale and the units (`core/sim_config.py:138-186`): step 2, step 1 |
| `log_params` | "Parameters on a logarithmic range" | not on these pages: `core/config.py`, or one of your own models' own setting |
| `chi_f0`, `chi_freq_bounds` | "Drive amplitude in training", "Frequency band in training" | fixed by measurement |
| `truncation` | "Made by a narrowing round: a training never continues it." | not a setting |
| `device`, `dtype` | "This computer: made on the graphics card (or the processor)" | not on these pages |
| `format`, `summary_flags`, `feature_set_version`, `chi_layout`, `chi_elem_w` | "Made by another version of PRISM" | not a setting |

A set whose `model` differs is another model's and is not listed (U10).

### Choices in this part the owner has not seen

Most consequential first.

1. A pre-run check that has not been run shows as "Attention" in Training's "Before you start" and
   never greys "Start training"; one that failed does not grey it either, but the click then asks
   one question first, with "Cancel" preselected; one that does not apply shows as "Information".
2. A narrowing round has no "Continue or start new" setting and asks nothing about starting from
   zero: it continues its own saved simulations when the same round is started again with the same
   settings, and never a training's; only when PRISM finds a saved simulation set one setting away
   from the round's does it ask, in a question that takes the place of today's tick box and carries
   the engine's own sentence, the engine's name for the setting included.
3. Loading a saved posterior warns when the prior in use is not the one it was trained with, but
   still loads it; an inference on a simulated cell then skips, and says it skips, the check
   against what training saw.
4. A saved prior built on a different bounds file gets an "Attention" line but can still be loaded;
   the line says when PRISM will refuse the load, and the button is not greyed for it.
5. Home's "Continue with these" applies the kind of observation that was in use last time, and
   every other option at its built-in value or the plan's.
6. The rule that a name taken by another record during a run costs nothing covers a prior build, a
   training, a narrowing round and the observation an inference makes, not the inference itself:
   there the name is refused, and what is lost is at most the simulation of one observation.
7. Training has "Expected epochs (for the time estimate only)" under "Rarely changed"; it never
   limits the fit, and the narrowing round has no such field.
8. With a posterior from a narrowing round in use, "Run inference" asks the other-observation
   question at every run, in both modes, in place of today's tick box.
9. With a narrowed posterior in use, the Narrowing round page shows observations other than its
   own as unusable before the click, not refused after it.
10. With "Save the simulations every … batches" at "Never", a saved simulation set with these very
    settings is listed in the from-zero question as one the training will not continue, and the
    question's first button reads "Start training without saving the simulations".
11. An inference cannot start until a name is typed for its observation; the inference's own name
    may be left blank.
12. Of these pages only Training and the Narrowing round state the memory limit in "Before you
    start"; a prior build and an inference, which also compute on the graphics card, do not.
13. Prior and Training each have two views (use a saved one, or make a new one), switched by the
    plain button at the right of the heading; Prior opens on the saved view when one exists.
14. The time in "Build a new prior (about …)" is the last finished build's on this computer; with
    none, the bracket is left out and the page says the time is not known.
15. The prior build's cost sentence leaves out the number of steps when it is not known.
16. The typed-units box is refilled from the selected model's units file when the model changes.
17. Limits typed by hand survive switching to the file and back, and are dropped only when another
    bounds file is selected or step 1 is applied.
18. The boxes for the drive's own amplitude, frequency and phase open blank, not at 0.
19. Inference gains "Posterior draws" under "Rarely changed"; its result table shows a "True value"
    column for a simulated cell, and its "Unit" column is blank where PRISM has no unit to show.
20. With no posterior in use, "Your own recordings" shows a sentence in place of a form.
21. The undriven recording is called "Spontaneous recording" on every form.
22. Loading a saved prior or posterior shows "Loading…" at the foot of the rail, leaves nothing to
    acknowledge when it is done, and keeps no file in the run history.
23. The on-screen labels where U7 gives none (for example "Kind of observation", "Slots for driven
    recordings", "Lock-in limit", "Region level", "Directions narrowed", "Continue or start new"
    and its three choices), the pages' sentences, the questions' titles, texts and answers other
    than "Cancel", the four readings beside a planned driven recording, the runs' and the chain's
    words, and which settings are main: on step 1 only the model and the kind of observation; on
    Training the budget, network width and depth and "Continue or start new".

### What this part fixes

- W002: the rescan button keeps the selected bounds file.
- W003: each page shows what is in use; a changed choice reads "Changed, not applied".
- W008: the rotation choice is not remembered.
- W009: the narrowing round opens on the observation just made; an observation can be named.
- W013: a prior load or build asks first while a posterior is held, and lets go only on success.
- W014: the bounds file says its parameters and the kind of observation it implies.
- W015: a new prior is listed, selected and remembered.
- W017, W024, W025: the "fits in one piece" memory line is gone.
- W020, W142: rotation is always stated, on or off.
- W021: after a training the required name is empty, so a second click starts nothing.
- W022: steps 5 and 6 name the posterior and prior they act on; step 3 marks a narrowed posterior.
- W023, W169: the from-zero question names every differing setting on screen (with E2).
- W026, W194: the narrowing round shows and passes its network settings.
- W035, W036: the epoch limit's two traps are stated, and the result says when it was reached.
- W044, W052, W109, W155: the "Save" boxes and "Name it below to keep it" go.
- W045, W053: every run here is named and annotated before it starts.
- W046: loading and building a prior are separate, each with its cost.
- W047: five text-pane-only problems become reasons beside the field or button.
- W048: typed limits survive a switch to the file and back.
- W049: a malformed bounds line is refused beside the list.
- W050, W054, W148: the cell list keeps its selection and judges no entry in passing.
- W055: every row of the driven-recordings table is remembered.
- W056: recording length, drive amplitude and the drive's rows open blank.
- W058: the greyed narrowing button says what it needs.
- W059, W149: a refused, failed or stopped training leaves the posterior in use.
- W060: unused rotation settings are read-only, not judged, not passed.
- W090, W147: no configuration exists without its prior.
- W091: the load view shows a posterior's own prior; the simulated check is skipped, and said so, for a foreign one.
- W094, W099: "Continue or start new" is the control for `resume` (with E6).
- W107, W108: the build's cost line uses one device reading and never prints a question mark.
- W110: the two measured definitions are text, not fields.
- W111: units are listed in a fixed order.
- W113: one name, "simulation set", on all five pages.
- W114, W154: a stage's refusal of a new simulation set becomes the page's own question; the narrowing round's still carries the engine's sentence (8.6).
- W138: a saved prior shows its bounds file, with a warning when it is not the one selected.
- W141, W146: typed units are not remembered, follow the model, and are said not to continue a simulation set made with the units file.
- W187: name, note, epoch limit, save cadence and continue-or-new are offered.
