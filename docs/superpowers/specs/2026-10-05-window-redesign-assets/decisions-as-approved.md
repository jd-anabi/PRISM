# The window redesign: what the owner has decided (locked; do not re-decide)

Everything here was decided by PRISM's owner on 2026-10-05, in chat, from pictures or from a
question with its options laid out, or by approving one of six design sections. A drafter must not
change, soften or add to these. Where the design needs a choice that is not here, make the smallest
sensible one and LIST it as "a choice the owner has not seen".

## U1 Who it is for
The owner runs PRISM from the window. **It may be released**, so strangers will open it with no one
to ask. "I require extreme clarity for this science application."

## U2 What the owner dislikes today
Home ("5 buttons just slapped in the center"); the screens under each tab ("very cluttered,
unintuitive, and hard to use/read"); getting around the whole program; the progress bars
("cluttered and not interpretable"); the console text shown by default (people "need to only see
what they need to by default"); the design language.

## U3 Toolkit
Stay on PySide6 unless an approved design needs something it cannot do; then stop and bring the
blockage and the cost. A sample of the chosen look was built in PySide6 and nothing blocked
(assets: `toolkit-*.png`, `toolkit-sample.py.txt`).

## U4 Scope
The look, the layout and the flow of every screen; PLUS everything the engine can already do that
the window does not offer or say; PLUS small engine changes (U18). Saving the fit as it goes is a
SEPARATE design, to accept or leave. No unattended restart in a fresh program. No installer. No
change to the command line. No change to the figures' content beyond colours, axes and lettering.

## U5 Getting around
- A **rail** down the left edge that never goes away: PRISM; Home; Parameter inference;
  Effective temperature; Live simulation; Reduction map; Records; and at the foot Your own models,
  Settings, Help.
- When Parameter inference is open its **six steps sit inside the rail as a chain**: 1 Model and
  options, 2 Prior, 3 Training, 4 Checks, 5 Inference, 6 Narrowing round. Each has a marker, a
  state word, the record it holds (for example "retrain_prior") or what it is waiting for.
- **The run's state sits at the foot of the rail on every screen** and is clickable to go to the
  run. (No track across the top of the page; no strip along the bottom of the window.)
- **Home is a status page**: the one next step with its button; where each inference step stands;
  what each part is for in one line; the latest records; how the last runs ended. With nothing on
  disk it opens on **"What do you want to do?"** in plain words.
- **Records** shows for each record what it was made from and what uses it, with the next action
  offered on the record's own page.

## U6 The look: "from the scientific page"
Dark ink on a clear page, no boxes round groups; a cool grey column for the rail; every list a
table with three rules (top, under the header, bottom; no inner lines; numbers right-aligned with
units); one marker SHAPE per state, always with its word; one accent blue (#0072B2 light, #56B4E9
dark) for the current place, the selection and the main action only; state colours from the
colour-blind-safe set used in scientific figures; progress on a labelled time axis whose hatched
end is the range in which the run should finish; charts drawn like a journal figure. Lettering in
Inter (bundled, Regular and SemiBold, digits of equal width); whole sentences the program says in
a book face (Cambria). Grafted on: **real system dialogs**, **stock control shapes**, and a
**"Recent / All epochs" switch** on the loss curve. Exact colours, sizes and devices: the assets
`look-chosen-scientific-page.html` and `toolkit-sample.py.txt`.

## U7 Vocabulary: the field's terms explained once, plain words for the machinery, one name per thing
| thing | the one name on screen | today's name |
|---|---|---|
| the chain of inference | Parameter inference; steps Model and options, Prior, Training, Checks, Inference, Narrowing round | Config, Prior, Posterior, Validate, Infer, TSNPE |
| what a training makes | posterior (explained once: the trained network that turns an observation into parameter estimates with their uncertainty) | posterior |
| the run that makes it | training; its stages are rotation, simulation, fit, saving | "Train / Load posterior" |
| any background job | run | task |
| the saved training simulations | simulation set | checkpoint, cache |
| anything PRISM writes | record; the screen is Records | artifact; Artifacts |
| the graphics-card setting | memory limit | VRAM ceiling |
| a driven recording at one frequency | driven recording | probe |
| a driven recording too short or too slow to use | unusable driven recording | masked probe |
| the multi-frequency observation | frequency response (χ(ω) given once) | chi mode |
| the decorrelating step | rotation | Fisher rotation |
| the narrowing retrain | narrowing round (TSNPE given once) | TSNPE |
| the honesty test | calibration (check) | Validate, SBC |
| the other measurement | Effective temperature: One cell, Sweep study, Compare | FDT analysis, Sweep study cross-validation |
| a saved set of settings | plan | (none) |
| the pre-run facts | Before you start | (budget lines) |
| the console text | the full text (button "Show the full text") | log pane |
Buttons name their outcome and an action keeps one name: "Start training", "Stop training…",
question "Stop training?", answers "Keep training" (the default) and "Stop and lose the fit".
The command line keeps its own names.

## U8 A long run's settings: named run plans
- A plan is a small text file kept beside the bounds and cell files, one per kind of run (for
  example "retrain").
- It MAY hold the choices of every inference step: the kind of observation and its probe settings,
  rotation on or off, which bounds file, the prior's build settings, the training budget and
  network, the calibration sizes, the narrowing round's settings including its network.
- It may NOT hold: the two measured definitions (the frequency band and the drive amplitude, which
  stay read-only displays of `core/config.py`), anything about the computer (memory limit,
  device), names, notes, seeds, consents.
- Picking a plan fills every step's form in one action; each value then shows where it came from
  ("from plan retrain", "changed from the plan", "usual").
- **Nothing is applied at launch**: the plan picker opens on "No plan" every time and shows which
  plan was used last as a hint. Every launch still opens at the built-in values. The window never
  restores a science setting on its own.
- "Before you start" names the plan and every departure from it.
- The window can save the current settings as a new plan, never over an existing one without
  asking.
- The plan's reader lives in the engine so the command line can later take the same file (E5).

## U9 The memory limit
Lives in **Settings, "This computer"**, per computer, and **is remembered**. Shown in every
"Before you start"; written into the run's history. A plan does not carry it. This REVERSES the
standing rule for this one setting (`CLAUDE.md`: "the VRAM ceiling is never remembered").
"This computer" also holds the keep-awake switch.

## U10 Before a run starts
- "Before you start" in plain sentences: how many simulations; whether it continues the saved
  simulation set or starts a new one, and if new, every setting that differs from the saved one, by
  its on-screen name; how long it should take; that a fit cannot be resumed; the state of the
  pre-run checks.
- About memory it says only what is known: the limit in force and what the last run on this card
  measured. Today's "fits in one piece" line is DROPPED (it would have reassured about the run that
  died).
- The run's name and note are typed before it starts; the name is checked as it is typed.
- The start button is greyed, with the reason beside it, until everything holds.
- **Whenever a run would simulate from zero while a saved simulation set for the same model exists,
  the window asks first, names every difference, and preselects Cancel** (today: only when exactly
  one setting differs).

## U11 While a run is live
- One sentence says where things stand; a stage line (rotation, simulation, fit, saving); the time
  axis (start, now, expected end; a range during the fit); health figures each with a state word.
- During simulation: batches done, pace, a firm end time. During the fit: epoch, pace, the loss so
  far and the epochs since the best one (E1).
- The console text is behind "Show the full text" and is written to a dated file as the run goes;
  the file survives a crash.
- When simulation ends, the unusable-driven-recordings total is compared with the pre-run audit and
  shown with its limits (within 3 points fine; up to 5 look closer; beyond 5 stop). Beyond 5 the
  window does NOT stop the fit by itself: it raises "needs you" and the owner decides.
- Only one run at a time in the whole program (unchanged). Settings are locked while it runs;
  moving around and reading stay possible, and saved effective-temperature runs stay browsable and
  comparable while a training runs (today they are locked).
- While a run is live PRISM asks Windows not to sleep.

## U12 Stopping, closing, failing, ending
- "Stop training…" opens a system dialog saying what is kept and what is lost; "Keep training" is
  preselected. After a yes the page says "Stopping: takes effect when this epoch ends, up to about
  an hour"; until then a "Don't stop after all" button withdraws it.
- **Closing the window during a run asks and keeps running by default.** The question names the
  run, how long it has been going and exactly what stopping would lose; the preselected answer is
  "Keep running" and the window stays open; the other answer, "Stop the run and close", stops the
  run, shows that it is stopping, waits until it really has, then closes. "Quit anyway" goes.
  Nothing ever runs unseen.
- **After a failure** the run's page says in plain words what failed, what was saved and what was
  lost, keeps the full error in the run's file, and offers "Continue from batch N". Before giving
  up the window tries again by itself: three tries with a pause, in place. It says plainly when a
  failure is of the kind only reopening the program can clear; after reopening, Home offers the
  same "Continue".
- When a run ends the rail's foot and Home say how and when until it is acknowledged, and the
  taskbar button flashes.
- The launcher starts PRISM without the black console window.

## U13 Appearance, text, keyboard, size
Appearance settings become three: follow Windows, light, dark. Gone: "Auto (time of day)", "Use the
Windows accent colour", "Use the Inter font everywhere". Text size follows Windows' text-size
setting. Everything is reachable from the keyboard: a key to jump between the rail and the page,
keys for the parts, Escape always gives the safe answer in a question, help opens from the
keyboard. The smallest window, 900 by 600, keeps working (form scrolling above a pinned start
button).

## U14 Applied against pending; buttons; naming
- Each step states what it holds ("In use: retrain_prior"). A choice changed afterwards reads
  "Changed, not applied" until applied. A run always uses what is applied.
- The double-duty buttons split: "Load this prior" / "Build a new prior (about a minute and a
  half)"; "Load this posterior" / "Start training".
- Building or loading a prior while a posterior is held asks first.
- The "Save" boxes that really rename go: a record is named before it is made, and Records offers
  "Rename…" for every kind.

## U15 What is remembered between launches
- Still remembered: which model, bounds file and saved records were picked; the description of the
  recordings; **the training budget (batches and rows per batch)**.
- Now also remembered: every row of the driven-recordings table (file and frequency).
- No longer remembered: the rotation tick and the typed-units choice (they come from the plan or
  the built-in value).
- Open blank and must be typed: the recording length and the drive amplitude (today 1 s and 1 N).
- Fixed: a remembered pick is saved when it is applied, not whatever a list showed at closing; the
  rescan button keeps the selection.
- Never remembered, as now: consents, seeds, names, notes, comparison lists. The plan picker's
  choice is not applied at launch (U8). The memory limit IS remembered (U9).

## U16 The steps' pages and the Checks page
- Every page: a heading and one plain sentence; the few main settings as label, field, unit;
  "Rarely changed" folded with its values still readable; "Before you start"; name and note; one
  main button; the result on the same page afterwards with the next step offered.
- Prior: the bounds file says what it is (how many parameters, what kind of observation it
  implies); loading a saved prior shows which bounds file it was built on and warns when that is
  not the one selected; building says what it costs first.
- Inference: a simulated cell or your own recordings as two clearly separate modes; the result is
  a table of each parameter's 5 %, 50 % and 95 % values and the corner plot; the observation it
  records can be named (E4).
- Narrowing round: the observation just made is preselected; its network settings come from the
  plan and are shown on the page.
- **Checks: every check there is has a screen**, on one list reachable at any time, in four groups.
  Deciding checks (six): before the run, the frequency band holds, and the unusable-recordings
  audit is in range; after simulating, the unusable-recordings total against the audit (automatic);
  after the fit, calibration, use of each part of the input, identifiability of the posterior.
  Informing: which parameters the observations can pin down (before the run), timing of a short
  fit (before the run), repeated calibrations at fixed numbers of driven recordings (after the
  fit). Tools: the card test, the drive study, the third identifiability method. "Run" on a check
  opens its small form with the cost stated, then its result (a verdict sentence, a table, the
  figures). The headline is always "n of 6 deciding checks passed", never a large PASS for one
  check alone. The Training page's "Before you start" shows the pre-run checks' state.

## U17 Records, the other parts, help, run history
- Records: one ruled table, a filter by kind and a search box; local time with the date; a record
  being written by a run in this window reads "being written". A record's page says in plain words
  what it is, when, its note, its settings IN FULL, what it was made from and what uses it, its
  figures (every kind), its complete log (searchable); actions Rename… (every kind), Edit note,
  and the next step that makes sense ("Check this posterior", "Run it on recordings", "Set up a run
  like this one"). Deleting says what would be lost in real terms, and for an unfinished record
  that another program may still be writing it. Records others depend on still cannot be deleted;
  nothing can be changed while a run is live here.
- Effective temperature, Live simulation, Reduction map: the same page anatomy and look; cost
  stated before a long measurement; stop asks first; an unfinished run shown on its own page as
  unfinished; a name usable again once its record is renamed or deleted; no "finished" line after
  a cancel or a crash.
- Your own models (the builder): keeps its steps; asks before unsaved edits are thrown away; its
  result line sits next to the buttons that produce it.
- Help: a page in the window's own words (a short section per part, a glossary of the science
  terms); a "?" beside each setting that opens its explanation from the keyboard as well as the
  mouse; the first-time opening.
- **Run history**: the window keeps one small file per run (what was started, with which plan and
  settings, the memory limit, when it ended and how, where its full text is). It lets Home say how
  the last run ended after a restart of the program, and is what "Continue from batch N" reads.

## U18 Engine changes (all small; none may change what a simulated row holds or the simulation set's identity)
- E1 Loss per epoch: the fit writes one log line per epoch (training loss, validation loss, time
  taken, epochs since the best). Nothing about the training itself changes.
- E2 Every differing setting named: the pre-run preview returns all the settings that differ from
  each saved simulation set. Read-only.
- E3 The card test becomes an engine step, so it can have a screen.
- E4 An observation can be named.
- E5 A reader for run plans, in the engine. New code; no stage changes.
- E6 Refusals that say where to fix it: the two refusals that today name no setting, and the
  input-use check's plain errors, become proper refusals.
- E7 A name taken during the run cannot cost the result: at the end it is saved under its id with a
  line saying so, instead of failing.
Done in the WINDOW instead of the engine: the guard that stops a display failure from removing a
finished record; showing long lists in full; recording the memory limit a run used.
Left for the AUDIT, deliberately: the memory planner's wrong "fits" arithmetic; failing fast after
an unrecoverable memory failure; dates in a record's own log lines; the prior not being checked
against the full bounds file inside the engine; saved simulations knowing nothing about the code or
a user model's equations.

## U19 How it is built
Test-first; every task reviewed and gated (one-process fast gate, started with the review). Nine
stages, the window usable after each: (1) the look's foundation, the old layout wearing the new
look; (2) the shell: rail, step chain, Home; (3) the run: its page, the full text in a file, stop,
close, failure, the end, run history; (4) Training, plans, Settings, applied against pending;
(5) the other steps and all the checks; (6) Records; (7) Effective temperature, Live simulation,
Reduction map, Your own models; (8) Help and the first-time opening; (9) the guide pages, the
owner's walkthrough rows, the final gates. The card test on the graphics card runs after the engine
changes that touch training, and the owner is told before it runs. Size: 70 to 100 tasks.

## U20 The design plugins
Their methods are applied by hand; the design is recorded in the project's own documents (this
design, and once built a page in the guide stating the look); nothing of a plugin's is written
into the repository and impeccable's program is never run. No plugin's name or file name appears
in code, tests or the guide.

## What must not change without the owner's say-so (from the hand-off prompt and CLAUDE.md)
- The science: the models, the simulation, the features, the training and calibration methods,
  every setting in `core/config.py`. The simulation set's identity must not change.
- The load-check audit on the open list comes before any change to `core/config.py` or to the
  features; if the design needs one, stop.
- Every knob reaches its stage as an argument, never as a config write (the memory limit is the
  one existing exception, read live by the planner).
- A refusal carries a field key; the window names the control from its own table
  (`core/gui/fields.py`); no core message names a box, tab, flag or button.
- A consent, a seed, a record name and a note are never remembered.
- Nothing under `core/`, `tests/` or `docs/guide/` cites the working documents or their labels
  (U-, E-, W-ids, section numbers, plugin names).
- One run at a time app-wide (the stream swap and the run buffer are process-wide).
- Every dialog is an instance dialog (a `QMessageBox` or `QDialog` instance, never a static call):
  the test guard depends on it.
- Never print or log between steps 1 and 3 of a checkpoint save; cancel checkpoints are deferred
  while an exception unwinds.
