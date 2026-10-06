## 3. The shell: the rail, the chain of steps, Home, the keyboard

What is on screen on every page, how a person moves, and what the window builds when it starts (U5,
U13, U24, U26). Sizes, colours, markers and their words are the look part's (4.2, 4.4, 4.5) and are
quoted here, not defined.

### 3.1 What goes, what stays, what changes

| Today | Fate | Detail |
|---|---|---|
| `MainWindow` (`core/gui/main_window.py:75-456`) | changes | Stays the `QMainWindow`, title "PRISM" (:88); its central widget is `Shell` (3.2) instead of `NavShell` (:97-98). **Kept:** the geometry guards (:36-72), the default size (:92), the geometry restore and its two checks (:187-196), `panel(cls)` (:412-414). **Goes:** the eager build of every screen (:100-146); `_section_index` (:148-152); `_tab_screens`, `_panel_home` (:162-169); the run slot's clock and marks (:173-180, :376-379, :394-410); the gear menu (:198-214); `_set_system_accent`, `_set_force_inter` (:234-251); the panels' sweep at closing (:451-455; B6). **Moves:** `closeEvent` (:416-445) to part 6 (6.7); `_tell` and the user-model handlers (:254-303) to part 10 (10.6), their questions put through the shared question dialog, so no `QMessageBox` is left here (B2). **Changes:** the minimum size (:95; 3.8); `set_appearance_mode` (3.9); `_on_run_state`, `_go_to_running` (3.5); `_all_panels`, `_on_user_models_changed`, `_refresh_model_combos`, `_refresh_store_pickers`, `_save_state` (3.10). |
| `build_app` (`core/gui/app.py:61-127`) | changes | 3.10 |
| `NavShell`, `running_banner`, `mark_tabs`, `RUNNING_MARK` (`screens/nav_shell.py`, whole file) | goes | Replaced by `screens/shell.py`: `Shell`, `Rail`, `StepRow`, `RunFoot`, and the tables `PARTS` and `STEPS`. No title row, back arrow, run slot or gear (:60-110). |
| `HomeScreen`, `greeting`, `SECTIONS` (`screens/home_screen.py`, whole file) | rewritten | Class name and file kept. `greeting` (:19-25), `SECTIONS` (:16), `tiles` (:52-63), `set_running_section` (:71-80) go. `navigate` (:33) stays a signal and carries `(part, page)`. |
| `SectionScreen` (`screens/section_screen.py`, whole file) | goes | A part's page is mounted in the shell's stack directly. |
| `InferenceScreen` (`screens/inference_screen.py:39-223`) | changes | **Kept:** `session` (:49), the six `*_panel` attributes (:60-67), `panels()` (:104-106), `session_contents()` (:108-140), `new_draft` (:148-165), `install_config` (:167-181), the name `refresh_gates`. **Goes:** the heading label and `title` argument (:47, :51-52); the `QTabWidget` and all that serves it: `tabs`, `_tab_labels` (:54, :68-73), `_on_tab_changed`, `_size_to_current_tab` (:86-102), `set_running_tab` (:142-146), every `setTabEnabled` and `setTabToolTip` (:195-215), the fall-back to Config (:220-223). **New:** `pages` (a `QStackedWidget` of the six step pages in `STEPS` order, no tab bar), `show_step(key)`, `current_step()`, signal `states_changed(object)`. |
| `SbiSession`, `ConfigDraft` (`core/gui/session.py`) | stays | Unchanged by this part (part 8 adds `holds` and `release`, 8.1); when its slots are written is part 5's (5.3; B3). A new pure module `core/gui/chain.py` (no Qt import) reads it. |
| `theming.py`; the appearance helpers of `settings.py` (:69-93) | change | 3.9 |

New files: `core/gui/screens/shell.py`, `core/gui/chain.py`, `core/gui/keys.py`, `core/gui/one_instance.py`.
New suites, as part 13 names them (13.1.5): `tests/test_shell.py`, `tests/test_step_chain.py`. The
tests that pin what goes are in part 13's Table B, stage 2 (13.1.2).

`PARTS` and `STEPS` (3.2) are the list of places: `core/gui/fields.py` names a place from them (5.7),
and the rewritten pin reads them off the built window (13.1.3).

### 3.2 The window and its two regions

`Shell` is a `QWidget` with a horizontal layout, no margins, no spacing: `Rail` (width 220) and a
`QStackedWidget` of pages.

| Table | Rows (key, on-screen name, shortcut) |
|---|---|
| `PARTS` | `home` "Home" Ctrl+1; `inference` "Parameter inference" Ctrl+2; `temperature` "Effective temperature" Ctrl+3; `live` "Live simulation" Ctrl+4; `reduction` "Reduction map" Ctrl+5; `records` "Records" Ctrl+6; `models` "Your own models" Ctrl+7; `settings` "Settings" Ctrl+8; `help` "Help" Ctrl+9 and F1 |
| `STEPS` | `model` "Model and options" Alt+1; `prior` "Prior" Alt+2; `training` "Training" Alt+3; `checks` "Checks" Alt+4; `infer` "Inference" Alt+5; `narrowing` "Narrowing round" Alt+6 |

`MainWindow.go_to(part, page=None)` is the one way to move and, with `go_to_run()` (3.5), the call
behind every next-step button and link on a page. It builds the part if needed (3.10), shows it, calls
its `show_page(page)` when a page is named, marks the rail, places the keyboard (3.7) and emits
`place_changed(part, page)`, all before it returns, as `NavShell._navigate` does today
(`nav_shell.py:124-128`). The page changes at once: today's 280 ms slide (`nav_shell.py:126-130`,
`widgets/anim.py:106-117`) goes, because its direction encoded the order of Home's tiles
(`main_window.py:119-121`) (a choice the owner has not seen). The window always opens on Home; neither
the part nor the step is restored (`main_window.py:78`, :183).

Every part's top widget offers `show_page(name)`, `panels()` (its `BasePanel`s, possibly none),
`first_focus()` (3.7) and `set_small(bool)` (3.8). Effective temperature's three pages are chosen on
that part's own page (10.3), not in the rail.

### 3.3 The rail

**Replaces** Home's five tiles as the only way between sections, the back arrow and the gear
(`home_screen.py:53-63`, `nav_shell.py:63-69`, :95-100). Closes W104: there is no header whose height
changes.

Top to bottom: the name "PRISM", a label and not a control; "Home"; "Parameter inference" and, while
that part is open, the chain of six steps beneath it (3.4); after a gap "Effective temperature", "Live
simulation", "Reduction map"; after a wider gap "Records"; a stretch; at the foot "Your own models",
"Settings", "Help"; last the run's state (3.5). Each row is `go_to(key)`. "Parameter inference" opens
the step in hand (3.4) at its first opening in a launch and the step last shown afterwards (a choice
the owner has not seen).

Rows are text only (the sample's `nav` class, `toolkit-sample.py.txt:287-290`). The open part's name
is SemiBold and its group stands on the page's ground between two hairlines (sample :1599-1607). The
accent tint and bar of 4.5 mark exactly one row: the open step while Parameter inference is open,
otherwise the open part's own row (a choice the owner has not seen: the sample shows only the first
case). A part's row carries no mark for a live run; the foot says it (a choice the owner has not seen;
today a "●" is put on a tile and a tab, `nav_shell.py:22`, :51, `home_screen.py:78`). When the rows
do not fit, 3.8 applies.

### 3.4 The chain of steps

**Replaces** the six tabs, their enabled states and the reasons kept in tooltips of disabled tabs
(`inference_screen.py:195-215`). Closes W012; with the "In use" lines of U14 it answers W022 and W058.

A step is one `StepRow` (sample :758-818; drawn as 4.6 gives): marker, number and name, and on the
second line the words, `"<state word>: <detail>"`; the detail is the record the step holds or what it
is doing. A step that cannot start shows only what it needs, as the look's key says of "To come"
(`look-chosen-scientific-page.html:818`). Words too long for the row are cut at their end with "…";
the tooltip, the accessible name ("Step 5, Inference. To come: needs a posterior") and Home's table
carry them whole. The line leaving a step is solid when that step is done, dotted otherwise.

No step is ever disabled: one that cannot start still opens and can be read, its main button greyed
with the reason beside it (B7; a choice the owner has not seen; `structure-chosen-rail.html:1163-1164`).

**The rule.** `chain.step_states(session, *, live=None, stopping=False, ended=None,
pending=frozenset(), checks=None, observation_on_offer=False)` returns six `StepState(key, marker,
word, detail, rail_words, can_start, done)`. It is pure, reads every session field through `getattr`,
and never raises: the gate tests put a bare `object()` on session fields
(`tests/test_nav_and_gating.py:313-321`), the rule `inference_screen.py:119-122` states today.

Inputs. `s` is `InferenceScreen.session` (`session.py:41-47`). `narrowed` is
`getattr(getattr(s.posterior, "posterior", None), "truncation", None) is not None`, the test
`infer_tab.py:522` uses. From the `RunStatus` (6.2): `live`, the step whose run is live, and
`stopping`. From the run history (6.5): `ended`, the batch N kept by the latest training of this
launch when it was stopped or failed. From part 5 (5.3): `pending`, and `<name>`, the name on the
step's "In use" line, from its one function. From part 9 (9.3): `checks`, its summary. From part 8
(8.1): `<doing>`, "choose a model", "choose a prior" or "setting up", and `observation_on_offer`.
This part's: `<model>` is `chain.model_label(name)`, "Nadrowski", "Hopf" and "BP" for the built-in
models (`core/config.py:203`) and a model of the user's own as it is stored (a choice the owner has
not seen).

For each step the first row that holds decides. `checks=None` counts as none of six passed. Today's
gates, which the conditions of steps 1, 2, 3, 5 and 6 were checked against: `inference_screen.py:185-204`,
`prior_tab.py:352`, `posterior_tab.py:327-330`, `infer_tab.py:503`, :516, `tsnpe_tab.py:172-173`,
:209-210; step 4, "Changed, not applied" (W003) and "Unfinished" have none.

| Step | Condition | Marker | Rail words |
|---|---|---|---|
| any | `live` is this step | Now | "Now: stopping" when `stopping`, else "Now: running" |
| any | this step is in `pending` (U14) | Attention | "Changed, not applied" |
| 1 | `s.draft` is not None | Done | "Done: <model>" |
| 2 | `s.inf_prior` is not None | Done | "Done: <name>" |
| 2 | `s.draft` is None | To come | "Needs a model" |
| 3 | `s.posterior` is not None | Done | "Done: <name>", then ", narrowed" when `narrowed` |
| 3 | `s.inf_prior` is None | To come | "Needs a prior" |
| 3 | `ended` is given | Unfinished | "Unfinished: batch N"; the detail is "simulations saved to batch N" (8.1) |
| 4 | `checks` says nothing can be checked yet | To come | "Needs a prior" |
| 4 | a deciding check reads "Failed" | Error | "Error: <p> of <a> passed" |
| 4 | a deciding check reads "Look closer" (9.2; B1: until it is accepted) | Attention | "Attention: <p> of <a> passed" |
| 4 | every deciding check that applies has passed (C15) | Done | "Done: <a> of <a> passed" |
| 4 | otherwise | Now if in hand, else To come | "Now: <p> of <a> passed" or "To come: <p> of <a> passed" |
| 5 | `s.observation` is not None | Done | "Done: <name>" |
| 5 | `s.posterior` is None | To come | "Needs a posterior" |
| 6 | `narrowed` | Done | "Done: <name>" |
| 6 | `s.posterior` is None | To come | "Needs a posterior" |
| 6 | `s.inf_prior` is None | To come | "Needs a prior" |
| 6 | not `observation_on_offer` | To come | "Needs an observation" |
| 1, 2, 3, 5, 6 | none of the above | Now if in hand, else To come | "Now: <doing>" or "To come" |

**The step in hand** is the first of steps 1 to 5 that is neither done nor unable to start. Step 4
counts only while `checks` reports a check that applies still open; step 6 is optional and never in
hand. (Counting Checks among the steps in hand, and step 6 as optional, are choices the owner has not
seen.) Step 3 reads `s.inf_prior`, not `s.cfg` as today (`inference_screen.py:186`, :196): under B3
both are written together, when the prior's run returns (5.3). `<a>` is the number of deciding
checks that apply: 6, or 3 for a posterior trained without the frequency response (9.3), whose step
4 therefore reads "Done: 3 of 3 passed" once those three have passed (C15; a choice the owner has
not seen: it amends U16's "always n of 6" for this one case).

The second line has 160 px, about 25 characters. So the rail writes the Checks count without the
words "deciding checks", and part 9's second line ("1 failed", "4 wait for step 3") goes to the
tooltip, the accessible name and Home's table, which writes the headline of 9.3 in full (a choice the
owner has not seen).

`InferenceScreen.refresh_gates()` keeps its name and its callers (`posterior_tab.py:298`,
`infer_tab.py:509`, `tsnpe_tab.py:177` and the like). Its new body: compute `step_states`, call every
panel's `refresh_local_gates()` as now (:217-218), emit `states_changed`, which `MainWindow` connects
to `Rail.set_step_states` and `HomeScreen.refresh`. `MainWindow` also recomputes the states (not the
panels' gates) on `status_changed` (3.5), so "Now: running" appears without a click. Each step page's
greyed main button takes its reason from the same `StepState`, so the rail and the page cannot
disagree.

### 3.5 The run's state at the foot of the rail

**Replaces** the header's run slot, "Running: <tab label> — m:ss" (`nav_shell.py:25-38`, :75-78;
`main_window.py:351-379`), which names no stage and vanishes when the run ends (:364-367). Closes W039
and, for what is shown, W038.

`RunFoot` (sample :1555-1570): a hairline, a marker, a SemiBold title that may wrap to two lines, and
up to two lines of words. Like the chain and Home it reads only the run part's `RunStatus` (6.2;
today's `RUN_STATE`, `base_panel.py:66-84`, widened), never a panel, and is redrawn on its
`status_changed`, so the shell runs no timer. `<Run>` is the run's word ("Training"). The eight
rows cover the states 6.2 publishes; the markers are 4.5's. A time within seven days carries its
weekday, "Sun 4 Oct, 12:10"; an older one is written as B22 gives it (C18).

| State | Marker | Title | Words |
|---|---|---|---|
| Nothing running, nothing unacknowledged | the dash | "Nothing is running" | "Only one run at a time." |
| Running; also while starting or trying again, with the run part's words | Now | "<Run> is running" | the stage in a few words, then "Settings are locked." (sample: "Fitting, epoch 62 of about 130. Settings are locked.") |
| A run that only loads a record (C3; a choice the owner has not seen) | Now | "Loading…" | none; like a Live simulation and a video export it ends into the first state, with nothing to acknowledge (6.5) |
| Stopping | Now | "<Run> is stopping" | when it takes effect (U12) |
| Needs you | Attention | "<Run> needs you" | the one reason (U11; for the comparison after simulating, only beyond 5 points, B1) |
| Ended: finished | Done | "<Run> finished" | "Sun 4 Oct, 12:10." |
| Ended: stopped | Unfinished | "<Run> stopped" | the date and time, then what is kept |
| Ended: failed; also a run that died, which 6.5 records at the next launch as not seen to end | Error | "<Run> failed" | the date and time (for a run that died, when last heard), then what is kept |

In the first state the foot is not a control and not a Tab stop (a choice the owner has not seen); in
every other it is one button that goes to the run, which for an ended run acknowledges it. The idle
sentence is C31's, in U7's words; the sample reads "Only one thing runs at a time." (:1566; a choice
the owner has not seen).
`MainWindow.go_to_run()` replaces `_go_to_running` (`main_window.py:381-392`): it calls `go_to` with
the run part's `where(run)`. A live run takes the foot; an ended run shows only while nothing is
live. After a restart the foot shows the newest unacknowledged run of the history for the records folder
in use (6.5), and a click opens
it in the run view's past mode (6.3, 6.9).

**Acknowledged** means: the run's page was the page showing while PRISM's window was the active
window (a choice the owner has not seen). It is tested when the run ends, at every `place_changed`
and when the window is activated; the shell then calls the run part's `acknowledge(run)`, which
writes it into the run's history (6.5), so an end nobody has seen is still said after PRISM is
reopened (U12, U17). The taskbar flash is the run part's (6.9).

### 3.6 Home

**For:** U5's status page and U24's "Continue with these". **Replaces** the greeting and five tiles
(`home_screen.py:39-65`).

Home states only what it reads at that moment: the session, the `RunStatus`, the run history, the
records (`ArtifactStore.list`, `core/artifacts/store.py:681-735`) and the remembered picks. Of the
run history it counts only the files that name the records folder in use (ruling D5; 6.5). It never
reads the card, never starts a program, never calls `orchestrator.training_preview` (which reads card
memory, `core/orchestrator.py:529-532`). `HomeScreen.refresh()` runs once while the window is built,
each time `go_to` opens Home afterwards (never at the window's first show, 3.10) and, while Home is
the page showing, on `states_changed`, on a change of `RunStatus` and on the Records part's
`store_changed`; while another page shows, these only mark Home to be read at its next opening.

#### Mid-project

Head: "Home", and beneath it "What to do next, where your work stands, and what each part is for."

| Block | What the user sees | Read from |
|---|---|---|
| Next | a heading "Next: …", one or two sentences, one main button (table below) | `RunStatus`, run history, session, remembered picks, `checks` |
| "Parameter inference" | a ruled table of six rows: "Step", "State" (marker and word), "Holds or waits for" (the detail in full; for Checks the headline of 9.3, which C15 gives for a posterior trained without the frequency response, and part 9's second line). A row opens its step. | `chain.step_states`, the rail's own call; before Parameter inference is built, on an empty `SbiSession`, which is the truth |
| "The parts of PRISM" | a ruled table, one row per rail entry but Home: name (opens it) and one line | a further column of `PARTS` |
| "Latest records" | the five newest records of all kinds: label (opens the record's page), kind word, date and time, as 10.1 writes them; "All records" opens Records | `ArtifactStore.list` for each kind in `KIND_DIRS` (`store.py:33`) |
| "Last runs" | the three latest runs made against the records folder in use: "Run", "Started", "Ended" (B22), "How it ended" (marker and word); a row opens the run in its past view | `history.latest(3)` (6.5) |

The steps table and the two right-hand tables sit in two columns that stack on a narrow page (3.8).
(Five records and three runs: a choice the owner has not seen.)

The parts' lines, with which Help's section for each part opens (10.8): "Infer a cell's model
parameters from recordings, in six steps."; "Measure it on one cell, or in a sweep study over two
parameters; compare saved runs."; "Watch a cell's bundle move as it is simulated; save a video.";
"The analytical map from the Nadrowski model to the Hopf normal form."; "Everything PRISM has
written, what each record was made from and what uses it."; "Define a model of your own.";
"Appearance, and this computer."; "Each part explained, and the science terms."

**The next step.** The first row that holds (the order is a choice the owner has not seen):

| Condition | Heading and sentence | Main button and where it goes |
|---|---|---|
| a run is live and needs someone | "Next: look at the <run>" and its reason | "See why", the run |
| a run is live | "Next: nothing until the <run> ends" and "<Run> <name> is running, and no other run can start until it ends. You can still move around and read results. Keep PRISM open; you can minimise it." While it is stopping, "is stopping" stands for "is running". | "Go to the run" |
| the latest run has ended and is unacknowledged | finished: "Next: see the <run>"; otherwise "Next: see what happened to the <run>"; then how and when it ended and what is kept | finished: "See the result"; stopped: "Go to the run"; failed or not seen to end: "Continue from batch N" where C2's rule (below) offers it, with "See what happened" as a link, else "See what happened" |
| the newest training or narrowing round failed or was not seen to end, that end has been acknowledged, and C2's rule (below) still offers it | "Next: continue the <run>", when it failed and what is kept | "Continue from batch N" |
| `s.draft` is None and something was in use last time | "Next: continue with what was in use last time" and "Last time: the model Nadrowski, the frequency response as the kind of observation, the bounds file master_tier1.txt, the prior retrain_prior, the posterior retrain, the plan retrain. Nothing is loaded until you press the button." | "Continue with these" (U24), the block's one button |
| `s.draft` is None | "Next: choose a model" and "Nothing is loaded: PRISM starts every launch with nothing loaded." | "Open Model and options" |
| `s.inf_prior` is None | "Next: build or load a prior" | "Open Prior" |
| `s.posterior` is None | "Next: start a training" and "The Training page says what it will cost before it starts." | "Set up training" |
| step 4 is in hand (3.4) | "Next: check the posterior" and the headline of 9.3 | "Open Checks" |
| `s.observation` is None | "Next: run the posterior on a cell or on your recordings" | "Open Inference" |
| otherwise | "Next: nothing is waiting" and one sentence that a narrowing round is optional, or that one was made | "Open Narrowing round", or "Open Records" once narrowed |

**"Continue from batch N" (C2).** Home offers it for the newest training or narrowing round that
ended by a failure, or was not seen to end, whether or not its end was acknowledged (rows three and
four), for as long as its simulation set is unfinished and no later run of that kind has started;
`history.continuable(run)` says so and gives the batch, among the runs of the records folder in use
(6.5; ruling D5). When the failure was in the fit the button reads "Fit again on the saved
simulations" (6.8) and stays on offer until a later run of that kind starts or the saved simulation
set is removed (ruling D6). For a run the person stopped, Home's next
step is the ordinary one: the Training page, whose "Before you start" says it continues the saved
simulation set. The press does what 6.8 defines and opens the run's own step with the name and the
note blank and the earlier name shown as a hint (C4). (U12 asks for the offer after a reopening;
making it without one too is a choice the owner has not seen.)

**"Continue with these" (U24, B16, C1).** "Last time" is what `remembered.remembered_picks()` returns
(7.3.1): the model, the kind of observation, the bounds file, the prior, the posterior and the plan,
each left out when it can no longer be found. The sentence names, by their on-screen names, exactly
what the press applies and loads, the kind of observation among them. One
press fills the plan in, if one is named, with part 7's `fill_forms(…, keep_remembered=True)` (7.1.7;
the explicit act U8 requires), then calls part 8's `InferenceScreen.load_chain(model, options,
bounds_file, prior, posterior)` (8.1), which applies step 1 and runs the own "Load this …" of steps 2
and 3. Home passes the remembered kind of observation (B12) and every other option at its built-in
value or the plan's (B16), as C1 rules (a choice the owner has not seen): under the built-in kind a
posterior trained with the frequency response is refused at its load
(`core/artifacts/store.py:1343-1346`). It passes the posterior only when its record names that prior
as the one it was trained on (`parents`, `core/orchestrator.py:1432`), which is how "one was in use"
is told from an older pick (a choice the owner has not seen). Nothing is built, trained or checked
(B16). The person stays on Home; each load shows at the foot as "Loading…" and leaves nothing to
acknowledge (C3); when the last returns, Home shows the real next step. A load that is refused or
fails ends the chain and leaves what was held (B3), and that step is then Home's next.

Staging (U19; 13.2.2): the first-time page is built with Home in stage 2 and its help completed in
stage 8 (C24; a choice the owner has not seen: U19 names the first-time opening in stage 8). The
foot's ended states and the rows that read the run history come in stage 3, "Continue with these" and
the reopened "Continue from batch N" in stage 4; until then they are left out, not shown empty.

#### The first time

Shown when all three hold: no record of any kind in the store, no file in the run history that
names the records folder in use (ruling D5), and the session holds nothing (`s.draft` is None).
(That test is a choice the owner has not seen.)

Head: "What do you want to do?", and beneath it "PRISM simulates models of the hair bundle, the motion
sensor of the inner ear's hair cells, and infers a cell's model parameters from recordings. Nothing has
been made yet."

| Button | Sentence beside it | Goes to |
|---|---|---|
| "Watch a live simulation" (the main button) | "Needs nothing first. See a cell's bundle move as it is simulated." | Live simulation |
| "Begin parameter inference" | "Six steps, each needing the one before. It starts with choosing a model." | step 1 |
| "Measure an effective temperature" | "On one simulated cell, or in a sweep study over two parameters." | Effective temperature |
| "Read what each part is for" | "A short section per part, and the science terms." | Help, at its first section (10.8) |

Below: "The parts of PRISM" as above, and "Latest records" reading "No records yet."; the steps table
and "Last runs" are left out. These choices, their order and their sentences are final between the
parts (B5); anything longer is Help's (10.8). That there are four and which is the main button is a
choice the owner has not seen; the mock-up had the first two.

### 3.7 The keyboard

**Replaces** nothing: a search of `core/gui` for `QShortcut`, `QKeySequence`, `setShortcut`,
`keyPressEvent`, `eventFilter`, `setTabOrder`, `setBuddy` finds none, and the one focus-policy call
opts a help badge out (`widgets/help_badge.py:29`). Closes W040.

The map is one table, `keys.KEYS` in `core/gui/keys.py`, made of the rows of `PARTS` and `STEPS` and
the others below. `MainWindow` makes its `QShortcut`s from it (window context, so none fires while a
question is open) and Help prints it (10.8).

| Key | What it does |
|---|---|
| F6, Shift+F6 | moves the keyboard between the rail and the page |
| Ctrl+1 to Ctrl+9 | opens the nine parts in rail order (3.2) |
| Alt+1 to Alt+6 | opens steps 1 to 6 from anywhere |
| Ctrl+R | goes to the run, whenever the foot is a button |
| F1 | opens Help at the section for the page showing; on a setting with a "?", that setting's explanation (5.8) |
| Up, Down, Home, End (in the rail) | moves through the parts and, when shown, the steps, in the order seen |
| Enter, Space (in the rail) | opens the row |
| Escape | in a question, the safe answer (U13; the shared dialog's, B2); closes an open explanation or list; in the rail, returns the keyboard to the page; otherwise nothing. It never moves between pages and never stops a run. |

Ctrl+1 to Ctrl+6, Alt+1 to Alt+6, Ctrl+R and F6 are the chosen mock-up's
(`structure-chosen-rail.html:1144-1148`); Shift+F6, Ctrl+7 to Ctrl+9, F1 and what Escape does outside
a question are added (choices the owner has not seen). "F1" and "F6" have the shape the reference scan
hunts; 13.1.8 says what the commit that first writes either must add.

**Focus order.** Tab reaches, in order: the rail's list as ONE stop (the arrows move within it), the
run's state when it is a button, then the page from its head down to its main button. After any move
to a page the keyboard is on the page's `first_focus()`, its first setting, never a button that starts
or stops a run; Enter in a field starts nothing. The focus ring is 4.5's; `first_focus()` and the
labels as their fields' buddies are part 5's.

**Access keys** (Alt and a letter). A page's main button and the buttons in its head have one each,
unique among what that page shows; digits are kept for the steps. Rail rows have none (their keys are
above and in each row's tooltip), nor have field labels. (A choice the owner has not seen.)

### 3.8 Sizes and the small layout

The window's opening size and restored position are kept as they are (3.1). Its smallest size is 900
by 600 at Windows' ordinary text size and grows with the text size, held to what the screen offers
(C17; a choice the owner has not seen); part 4 scales the opening size and the two layout thresholds
with it (4.8). A window lower than 700 px gets the small layout (4.4; sample :105, :2039-2046): in the
rail a step keeps its second line only when it is the step in hand or the first step that cannot
start (sample :1579-1590). `Shell` tells every built page `set_small`.

Heights, from the look's table and the sample, with all of U5's rows and the chain open:

| Layout | Name | Rows and gaps | Chain | Run's state | Needed |
|---|---|---|---|---|---|
| usual | 58 | 9 rows of 36, gaps 17, hairlines 2 | 6 of 46 | 79 | 756 of 820 |
| small | 48 | 9 rows of 32, gaps 17, hairlines 2 | 2 of 46, 4 of 32 | 79 | 654 of 600 |

At the smallest window, with the six steps showing, the rail's list is 54 px too tall, so it
scrolls inside the rail when it must, the open row is always brought into view, and the name and the
run's state stay put: the look's note on the small layout has the list of steps scroll above a pinned
run line (`look-chosen-scientific-page.html:74-77`; 4.8). The sample fitted because it drew three rows
fewer than U5 lists: no Home, Your own models or Help (sample :1537-1552). (That the whole list
scrolls, the parts with the steps, is a choice the owner has not seen; the alternative is tighter
rows.) A wrapped title adds 20 px to the run's state. At a larger Windows text size the same scroll
keeps the list whole; how text size is followed waits for B20's test.

At 900 wide the page has 680 px, below the 940 at which a page sets two columns side by side (4.4):
Home's columns stack and the page scrolls.

### 3.9 Appearance: three choices

**For:** U13. `theming.MODE_LABELS` (`theming.py:24-25`) becomes `("system", "Follow Windows")`,
`("light", "Light")`, `("dark", "Dark")`. Out of `theming.py`: `_DAY_START`, `_NIGHT_START` (:29-30),
the `auto` branch and `hour` argument of `_scheme_for` (:41-48), the 60 s timer (:65-67, :81-84),
`_system_accent`, `system_accent_enabled`, `set_system_accent` and the `accent` passed on (:63, :86-94,
:110-112), and the `datetime` import. `_on_os_scheme` (:96-98) stays: "Follow Windows" still follows a
change made while PRISM is open. Closes W101. What an apply does, `design.build_palette` and
`build_qss` losing their `accent` argument, and the engine's figures staying light in both themes
(B18) are the look part's (4.1, 4.8, 4.9).

The three are offered only on the Settings page (10.7); the gear's pop-up list goes with the gear
(`main_window.py:198-214`; a choice the owner has not seen). `MainWindow.set_appearance_mode(mode)`
(:216-231) keeps its name, still writes `appearance/mode` at once and applies it, loses the menu
mirroring (:228-230), and tells the Settings page only if it is built. The appearance stays
changeable while a run is live (C29; 6.12).

Stored values of what is removed follow the standing rule that a stale key is ignored, not restored
(a choice the owner has not seen). `settings.get_appearance` (`settings.py:69-71`) returns "system"
for any stored word but the three, as `Appearance.set_mode` already does (`theming.py:79`): a stored
`auto` shows "Follow Windows", and the file is rewritten only at the next pick. The getters and
setters of `appearance/system_accent` and `appearance/force_inter` go (`settings.py:78-93`); the keys
stay in the file, unread.

### 3.10 Start-up

#### What is built

`build_app` keeps its order (`app.py:67-126`; the progress-display flag at :72 stays, 6.1), and
`__main__.py:9-13` still forces the Agg backend before any `core` import. Changes:

- One PRISM at a time (below) comes right after the application has its names (:89-91), before
  style, fonts or any window.
- `this_computer.apply_remembered()` (7.2.3) is called once the settings store exists and before the
  window is built.
- :100 loses `prefer_inter=settings.get_force_inter(qs)`; :102 (`set_system_accent`) goes; :104
  (`mpl_theme.install`) is the look part's (B18).
- The error hook (:106-107) stays; its box (:25-30) becomes the shared question dialog in its telling
  form (B2; 4.6): the error's message, and that the full error is in the window's log file. The
  error's type name is seen only behind "Show the full error", with the traceback (C7). That
  file, the "PRISM could not start" message (:111-116) and the three writes that expect a console
  (:32, :56-58, :119) are the launcher's (6.11).

`MainWindow.__init__` builds only: `Shell` with the rail (chain folded, the foot from `RunStatus`),
Home with one `refresh()`, the shortcuts, and the geometry. Every other part is built the first time
it is opened; each, Home included, by `MainWindow._build(part)`, inside a guard (B21; a choice the
owner has not seen). If
a part's construction raises, its place in the stack shows, in the page's look: the Error marker,
"<Part> could not be opened.", the error's message, "The rest of PRISM still works.", and the
buttons "Try again" and "Show the full error" (the error's type name and the traceback, seen only
there, C7, in the fixed-width face of B19; both also go to the log). The six step pages are built together, at the first opening of Parameter inference,
because they call one another (`inference_screen.py:164`, :180, :217-218). `panel(cls)` builds the
owning part when asked. A part built while a run is live asks the run controller and comes up locked
(6.12; today only panels that already exist are locked, `base_panel.py:316-317`). Closes W041: today
one failing constructor anywhere ends in "PRISM could not start" (`main_window.py:100-146`).

Consequences. `_save_state` (`main_window.py:447-456`) writes the window's size and place and nothing
else (B6); the panels' `save_settings` go, and when each remembered value is written is part 7's
(7.3). `_all_panels`, `_on_user_models_changed`, `_refresh_model_combos` and
`_refresh_store_pickers` (:305-348) reach built parts only; an unbuilt part reads the lists when it
is built (10.6).

#### One PRISM at a time (U26, B17)

`core/gui/one_instance.py` holds a `QLockFile` on `PRISM.lock` beside the settings file
(`settings.py:20-23`), with no stale time, tried once without waiting.

| Case | What happens |
|---|---|
| the lock is free | PRISM takes it, starts, and listens on a `QLocalServer` whose name is made from the settings file's path: one per user, and one per temporary settings file in the suites |
| the lock is held and the open PRISM takes the connection | the new process gives it the right to the foreground (`AllowSetForegroundWindow`, with the process id from `QLockFile.getLockInfo()`; left out off Windows), sends one word and exits with status 0, having built no window and shown nothing. The open window leaves the minimised state, rises and takes the keyboard (`MainWindow.come_to_front()`); if Windows still withholds the foreground, its taskbar button flashes |
| the lock is held and nothing takes the connection in three tries a second apart | the lock was left by a PRISM that died: it is removed and taken, and PRISM starts (a choice the owner has not seen). If the removal fails, its holder is alive: the new process exits, building no window |

The command line is untouched and the lock does not see it. A run-history file that still says
"running" at launch is therefore always a run that died (B17; 6.5). The suites build `MainWindow`
directly and take no lock; the one test that calls `build_app` (`tests/test_user_models.py:1226`)
does, and must release it and close the listener before it ends, as 13.1.4 has it (C26), through one
call `one_instance` offers. The choice among the three cases is one pure function, pinned in
`tests/test_shell.py`; a real second start is a walkthrough row (13.4). Three things are to be
established by a scratch test before this is built, as B20 does for text size: that the lock is
released when its process dies, that the connection is taken while the open window is busy, and that
removing the lock fails while its holder lives. Without the last, a silent connection alone must not
start a second PRISM: B17's reading of the run history rests on it. Closes W143.

#### What start-up must not touch

Before the first click the window reads only files: the settings file, the user-model files
(`app.py:76`), the records' manifests and the run history. It asks the card nothing and starts no
other program. Today, before any click, it does both, on the window's own thread: `nvidia-smi` with a
2 s limit (`config_tab.py:119` to :151, `core/artifacts/provenance.py:122-123`);
`config.detect_device()` (`prior_tab.py:121` to :297, `inference/base.py:77-82`); and card memory
through the training preview (`posterior_tab.py:112`, `tsnpe_tab.py:105`,
`core/orchestrator.py:529-532`), repeated by the first `refresh_gates` (`inference_screen.py:80`).
With no page but Home built at start, none of that can run at launch. Closes W180 and the launch half
of W043.

Pinned by one test in `tests/test_shell.py`: with `subprocess.run`, `torch.cuda.is_available`,
`torch.cuda.mem_get_info` and `config.detect_device` replaced by functions that record each call,
building `MainWindow` and showing Home records none (functions that raise would prove nothing: today's
callers swallow the exception, `core/artifacts/provenance.py:127-128`, `core/orchestrator.py:533-535`,
`prior_tab.py:312-313`). Extended to opening each part, the same test is the rule for
every page's constructor: reading the card or running `nvidia-smi` belongs to an explicit moment on
the page (the press in Settings, 7.2.3), never to construction. Nothing new runs between the first
show and the first idle turn: Home's read happens during construction, and no timer is started (the
rule at `nav_shell.py:71-74`, `main_window.py:171-172`).

### Choices in this part the owner has not seen

- For a posterior trained without the frequency response, the Checks step and Home count only the three deciding checks that apply: Home's headline reads "n of 3 deciding checks passed; 3 do not apply to this posterior", and the step reads "Done" once those three have passed. This amends "always n of 6" for this one case.
- A step that cannot start yet can still be opened and read, its main button greyed with the reason; today such a tab cannot be opened.
- Each part of the program is built when it is first opened, not at start-up; one that fails to build shows an error page with "Try again", and the rest works.
- A run's end counts as seen once its page has been showing while PRISM's window was the active one.
- Home picks its one next step in a fixed order: a live run, an end nobody has seen, a failed run that can be continued, "Continue with these", then the steps in order.
- Home offers "Continue from batch N" for the newest training or narrowing round that failed or was not seen to end, with or without a reopening, until its simulation set is finished or a later run of that kind starts (after a failure in the fit, until a later run of that kind starts or the saved simulation set is removed); after a stop Home shows the ordinary next step.
- "Continue with these" applies the kind of observation that was in use last time, and every other option at its built-in value or the plan's; Home's sentence names the kind.
- "Continue with these" loads the saved posterior only if it was trained on the prior remembered with it.
- A run that only loads a saved record shows "Loading…" at the rail's foot and then nothing: no end to acknowledge and no file in the run history, which was to keep one per run.
- The smallest window is 900 by 600 at Windows' ordinary text size and grows with the text size, held to the screen; the decision gave 900 by 600 alone.
- The idle sentence at the rail's foot reads "Only one run at a time."; the built sample read "Only one thing runs at a time.".
- Checks counts as a step in hand: after a training the chain shows Checks as "Now" and Inference as "To come" while a deciding check is still open, though Inference can be run.
- The Narrowing round is optional: never the step in hand, and "To come" once it can start.
- The first-time Home shows only with no record, no run history for the records folder in use and nothing loaded, and offers four choices with "Watch a live simulation" as the main button (the mock-up had two).
- The first-time page is built with Home in the second of the nine stages and its help completed in the eighth, where the list of stages names the first-time opening.
- After a crash the next start takes over the lock the dead PRISM left, without asking; a second start while PRISM is open shows nothing of its own.
- The accent bar marks the open part's own row when no step is open (the sample showed it only on a step).
- At the smallest window with the six steps showing, the rail's whole list scrolls (by about 54 px) under a fixed name and above a fixed run's state; the alternative is tighter rows.
- In the rail the Checks step reads "3 of 6 passed" without "deciding checks", and words too long for a row are cut with "…"; the tooltip and Home give them whole.
- A part's row in the rail carries no mark for a live run; only the foot says it.
- With nothing running and nothing unseen, the run's state at the foot is plain text, not a button.
- "Parameter inference" opens the step in hand the first time in a launch and the step last shown afterwards.
- Pages change at once, with no slide.
- Home lists the five newest records and the three latest runs.
- Five keys are added to the mock-up's: Shift+F6, Ctrl+7, Ctrl+8, Ctrl+9 and F1.
- Escape outside a question only closes what is open or returns from the rail to the page.
- Access keys (Alt and a letter) go only on a page's main button and the buttons in its heading.
- The built-in models are written "Nadrowski", "Hopf" and "BP"; a model of the user's own as it is stored.
- A stored "Auto (time of day)" is read as "Follow Windows"; the two removed appearance settings stay in the settings file, ignored.
- The appearance choices are offered only on the Settings page; the gear's pop-up list goes with the gear.

### What this part fixes

- W012: the chain and Home's steps table say what the session holds, on every Parameter inference page and on Home.
- W038: the rail's foot and Home say how and when a run ended until it is acknowledged, also after a restart (with the run part's history and flash).
- W039: the run's state at the foot names the run, its stage and whether it needs someone, in place of a grey label with a clock.
- W040: a key for every part and step, a key between rail and page, a key to the run, Escape defined, one table of keys.
- W041: a part is built at its first opening inside a guard; one failing part no longer stops PRISM from starting.
- W101: "Auto (time of day)" and its once-a-minute restyle are gone.
- W104: no header; the rail and the page keep their sizes on every page.
- W143: starting PRISM while it is open brings the open window to the front instead of opening a second (U26).
- W180: opening PRISM no longer touches the graphics card.
- W043, in part: `nvidia-smi` no longer runs at launch (the per-keystroke half is the Settings part's).
- W022, in part: on the Checks, Inference and Narrowing round pages the chain names the prior and the posterior in use, and says when the posterior is the narrowed one.
- W058, in part: step 6 reads "Needs an observation" in the rail (the reason beside its button is that page's).
