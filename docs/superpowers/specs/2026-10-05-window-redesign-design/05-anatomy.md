## 5. The anatomy of a page, and the pieces every page shares

Every working page (the six inference steps, each check's form, Effective temperature, Live
simulation, Reduction map) is built from the pieces below, defined once (U14, U16 first bullet, U10).
The page parts say what their pages hold; they do not redefine a piece. A run's progress, stopping,
closing, failure and end are part 6's: 5.11 says where a page hands over. Paths are under
`core/gui/` unless they start with `core/` or `tests/`. A number such as 4.6 or 8.3 is a section of
another part.

### 5.1 The page, top to bottom

**For.** One arrangement to learn (U1, U2). **Replaces** `BasePanel`'s frame: a controls column in a
scroll area beside a figure stack over a log pane, two splitters, and a progress row with "Cancel"
(`panels/base_panel.py:128-191`), with every group in a `QGroupBox` (`panels/inference/posterior_tab.py:37`,
`:66`, `:85`, `:99`). The shape below is pictured in `toolkit-setup-light.png` and
`toolkit-smallest-window-setup.png`.

| # | Region | What the user sees | Scrolls | Locked while a run is live |
|---|---|---|---|---|
| 1 | Heading | The page's name; one plain sentence under it; at the right, plain buttons: "Help", which opens Help at this page's section, and, where the page has a second view, the way to it (U14; drawn as "Use a saved posterior instead…" in `structure-chosen-rail-2-setup.png`) | no | no |
| 2 | In use | What the step holds and the result of the run that made it (5.3, 5.9). A page with no held record shows the last run made on it; an Effective temperature page, its saved runs and the record opened (10.3) | yes | no |
| 3 | Form | "Main settings"; "Rarely changed (n settings)"; the block for this run: "Name", "Note", and read-only rows naming what the run reads from other steps | yes | yes |
| 4 | Foot | "Before you start" (5.4); then one line: the reason, "Check again", the main button (5.6) | no | the two buttons; the facts stay readable |

- Every piece is drawn as part 4 gives it (4.4, 4.6), and `Page`, `PageHead`, `Scroller` and
  `Columns` live where it puts them (`widgets/page.py`, 4.1); no box surrounds a group (U6). This
  part says what they do, and adds `widgets/form.py` (`Form` and its rows, 5.2; the plan may rename
  it).
- Regions 2 and 3 share one scrolling body, region 2 above region 3 (a choice the owner has not
  seen). Region 3 is two columns from a page width of 940 px and one below it. A window lower than
  700 px gets the small layout through `Page.set_small` (3.8): the heading's sentence is hidden and
  the facts list shows only what blocks or needs attention, with a link "Show all 5" / "Show fewer".
  The foot never scrolls, so at the smallest window, 900 by 600 at Windows' ordinary text size (C17,
  a choice the owner has not seen; 3.8, 4.8), the main button stays on screen (U13).
- `Page.first_focus()` is the page's first setting, never a button that starts or stops a run; each
  label is its field's buddy; Enter in a field starts nothing (3.7).
- On the page that started a run, `RunView` takes the place of regions 2 to 4 until the run is over
  (6.3); every other page keeps its regions, locked as the last column says (6.12).
- A page prints no free text outside a run. What a tab writes to its own log pane today outside a run
  becomes a line of region 2, a fact, or a line under a field: for example "Prior ready: … Name it
  below to keep it." (`panels/inference/prior_tab.py:317-318`) and "Posterior ready: …"
  (`posterior_tab.py:296-297`).
- Gone with the frame: both splitters and their saved state (`base_panel.py:355-396`, the stale
  comments at `:359` and `:368` included: W105, W119, W128; the `layout/<Class>/…` keys stay in
  `PRISM.ini` unread, 7.3.1 rule 4), the per-page log pane and progress row, and
  `insert_result_widget` (`:343-353`). No page keeps a divider the user can drag (a choice the owner
  has not seen), so `tests/test_settings_persistence.py:636` and `:662` are retired; `:688`, which
  pins the old form builder, is rewritten for `Form`.

### 5.2 A setting row

**For.** Label, field, unit, read left to right, built from the setting's field key so that one table
names it (`fields.py:1-10`). **Replaces** `make_form` with `add_help_row` (`widgets/forms.py:16-22`,
`widgets/help_badge.py:55-63`). A page adds a row with `form.add(key, widget, unit="")`; the form
keeps `form.rows[key]`, which is how a refusal finds its row (5.7). A row offers `value()`,
`set_value(v)`, `usual()` and `set_origin(text)`, which is what a page's `plan_fields()` hands the
plans part (7.1.3), and `label_text()` for the tests that read a row's label back.

| Part | Content | Rule |
|---|---|---|
| Label | `fields.label(key)` through `labels.pretty_gui` (as `help_badge.py:42`) | The label column of 4.4; a long label wraps |
| "?" | Opens the setting's explanation (5.8) | |
| Field | A number, text, a path or a picker, each as 4.6 draws it | Today's line edits (`widgets/labeled_inputs.py:15-85`), changed as 5.10 says, read only through `value_or_none()`; no spin boxes |
| Unit | Plain text after the field: "s", "Hz", "epochs" | |
| Origin | Where the value came from (below) | Small type; words, never colour alone |
| Line under the field | An error sentence (5.7), or "Changed, not applied. In use: …" (5.3), or nothing | Each with its marker (4.5) |

A value the window may not change is drawn as text, not as a field (4.5), with the origin "fixed by
measurement" (the opening words of today's caption, which goes on to name `config.py`,
`config_tab.py:84`). Today the two read-only boxes are ordinary line edits and look editable
(`config_tab.py:71-75`; W110).

**Origin (U8).** For a field, the five texts and the condition for each are 7.1.8's: "usual", "from
plan retrain", "changed from the plan", "changed from the usual", "kept from last time". This part
draws them and works each out in one function, from `plan_value(heading, name)` (7.1.8), the row's
`usual()` (the built-in value the field opens at: `core/config.py`'s, or on the Narrowing round
`core/SBI/truncate.py`'s for two rows, `tsnpe_tab.py:68-69`) and the field.

**"Rarely changed".** The fold of 4.6, folded at every launch, with its settings' values in one
sentence under it while folded. When a folded setting is not at its usual value the button reads
"Rarely changed (7 settings, 2 not usual)" and the sentence names those first; when a folded field is
in error the fold opens and stays open until the error is gone (both: a choice the owner has not
seen).

### 5.3 Applied against pending (U14)

#### What "the session holds" means

Today one `SbiSession` (`session.py:40-47`) is owned by `InferenceScreen`
(`screens/inference_screen.py:49`); tabs read it through the screen and never cache it
(`panels/inference/base.py:14-25`).

| Slot (step) | Is | Written today |
|---|---|---|
| `draft` (1) | model, units, kind of observation and its settings, rotation (`session.py:7-25`) | `new_draft`, which replaces the whole session (`inference_screen.py:148-165`) |
| `cfg` (2) | the configuration built from the draft and a bounds file; `cfg.sources["bounds"]` names the file (`core/cli.py:349`) | `install_config`, at the click, before the prior exists (`prior_tab.py:209-210`) |
| `inf_prior` (2) | a `LoadedPrior` | `prior_tab.py:316` |
| `posterior` (3, 6) | a `LoadedPosterior` | `posterior_tab.py:288`; replaced by a narrowing round (`panels/inference/tsnpe_tab.py:173`) |
| `observation` (5) | a `LoadedObservation` | `panels/inference/infer_tab.py:503` |

"Held" and "in use" both mean: in one of these slots. Every record is on disk whether or not it is
held (`session.py:50-51`). `InferenceScreen` stays the owner of the six step pages (3.1) and keeps
the two dataclasses, `session`, `new_draft`, `install_config`, `session_contents`
(`inference_screen.py:108-181`) and `refresh_gates` by name. What changes is when the slots are
written and that every page shows them. Pages outside Parameter inference hold nothing between runs.

#### What "In use: …" shows

Region 2 of each step opens with one line, the record's label in semibold as 10.1 writes it: "In
use: retrain_prior"; for an unnamed record "In use: Unnamed prior, 1 Oct 2026, 22:29". One sentence
follows with the facts that tell it from its neighbours (the page part writes it). Both are read from
the held object, never from a control: step 1 reads `draft`; step 2 reads `inf_prior` and `cfg`,
including the bounds file from `cfg.sources`, which no widget reads today (W003). With nothing held:
"In use: nothing yet." and what the step waits for. One function gives the label (it replaces
`_held_ref`, `inference_screen.py:23-36`); the rail and Home show the same name from it (3.4). It
never raises (the rule at `inference_screen.py:119-122`). `session_contents()` names each record by
that function and no longer lists the built configuration, never held without its prior (W090):
today "the built config for NADROWSKI, chi observations" (`inference_screen.py:129-133`; pinned at
`tests/test_nav_and_gating.py:4462-4465`, `:4477-4479`, rewritten first) (a choice the owner has not
seen).

The run's block in region 3 names, as read-only rows, every held record the run will read: "Prior in
use: retrain_prior", "Posterior in use: retrain" (W022). Today only the Config tab's line and the
Apply question say what is held (`config_tab.py:362-387`; W012).

#### When a choice reads "Changed, not applied"

A *selection row* is a row whose value picks what the step holds: step 1, every row; step 2, the
bounds file or the typed limits and the saved prior; step 3, the saved posterior. It declares how to
read its in-use value from the session. Once the step holds something (U14 speaks of a choice changed
afterwards), whenever the row's control differs from that value the line under it reads, with the
Attention marker (4.5), "Changed, not applied. In use: master.txt." and the "In use" line gains "2
choices below are changed and not applied." The line goes when the choice is applied or put back.
The steps with such a row are the `pending` set the chain takes, so the rail says the same (3.4).
Rows that only set up the next run (budget, network, sizes, a narrowing round's observation, name,
note) are not selection rows and never carry this line. (Naming what is in use on the line, and the
count: a choice the owner has not seen.)

#### What applying does

Applying is the step's main button (a page with two views has one in each: 8.3, 8.4). In order:

1. The page's fields are read through its rule table. A refusal changes nothing (5.7).
2. The page's questions are asked, each with its safe answer preselected (below; U10's and C12's
   are part 8's). A safe answer changes nothing.
3. Step 1 applies at once (`new_draft`). Every other step hands over to the run (5.11) with a
   *candidate*: step 2 builds the configuration with `draft.make_config(...)` (`session.py:27-37`) and
   passes it to `build_prior` without installing it.
4. Only when the run returns its record are the slots written (B3), in one step on the window's
   thread: the new object in; the slots that depended on the old one released (`reset_downstream`,
   `session.py:49-57`); the page's pickers read again and moved to the new record; the page's
   remembered picks written (U15, B6; the keys are 7.3.2's); every "In use" line and step state
   derived again.
5. A run that is refused, fails or is stopped writes no slot.

Today the release and `install_config` happen at the click (`prior_tab.py:209-210`,
`posterior_tab.py:142`), so a refused, failed or cancelled run has already let go of what was held
(W013, W059, W149) and can leave a configuration with no prior (W090, W147). Under B3 the previously
held posterior stays in memory during a new training; what that costs on the card is not measured.

A record's next-step button asks one question up front (B4, C9; 10.2): what it will load and what it
will let go, the second in the phrases of `session_contents()`; Home's "Continue with these" (U24,
B16) has nothing held to release. Each then applies through these steps, one load at a time
(`load_chain`, 8.1), with none of the questions below asked again, and neither leaves a choice
"Changed, not applied". A consent is never implied by that question (C9): a narrowed posterior asks
its own inside the chain when that load arises (8.4), and running it on a different observation asks
at "Run inference" (8.5). "Narrow a posterior around it" on an observation is greyed until a
posterior is in use (10.2).

#### What a run reads

A run reads the session's slots, and its own page's fields as read at the click; never another page's
controls, as the stages are called today. When another step has a choice "Changed, not applied" that
feeds a slot this run reads, "Before you start" adds "Attention. Step 2, Prior, has changes that are
not applied. This run uses what is in use: retrain_prior." It does not block (a choice the owner has
not seen).

#### The questions before something held is released

Each is the one question dialog of B2, through `ask` (4.6), which holds the click until it is
answered: the safe answer is the default and the answer Escape gives; it is asked after the fields
are read and before anything changes (the rules at `config_tab.py:263-303`, `:335-340`). No question
when nothing would be released. Every record stays in Records, so no answer here loses work and none
is drawn as destructive (B27).

| Button pressed | Asked when | Title | Sentence | Safe answer (default) | Other answer |
|---|---|---|---|---|---|
| "Apply model and options" | `session_contents()` is not empty (today's rule, `config_tab.py:285-287`) | "Start a new session?" (kept) | "Applying releases what this session holds: the prior retrain_prior; the posterior retrain. A different model, units or kind of observation makes them invalid. Each stays in Records and can be loaded again." | "Keep this session" (kept) | "Apply and start a new session" (kept) |
| "Load this prior"; "Build a new prior (about a minute and a half)" | a posterior is held (new, U14) | "Load this prior?"; "Build a new prior?" | "This session holds the posterior retrain and the observation cell_a. They belong to the prior in use, so another prior releases them once it has loaded. Each stays in Records and can be loaded again." For a build, "once it is built" | "Keep the posterior" | "Load and release the posterior"; "Build and release the posterior" |

The new titles, sentences and answers are a choice the owner has not seen. Without an observation the
sentence names the posterior alone. An observation held with no posterior is released by another
prior without a question, and "Load this posterior" and "Start training" ask nothing when a
posterior is held: U14 names only a prior loaded or built while a posterior is held (a choice the
owner has not seen).

### 5.4 "Before you start" (U10)

**For.** Everything a person must know before the main button, as sentences, in one place, with the
button's state following from it. **Replaces** the three plain labels under the budget boxes
(`budget_total`, `budget_mem`, `budget_ckpt`; `posterior_tab.py:74-78`, `panels/inference/base.py:144-216`),
button tooltips (`posterior_tab.py:330`, `infer_tab.py:517-518`) and warning lines in the log pane.

#### A fact

| Part of a fact | Values |
|---|---|
| Marker and lead word | "Error." (`err`) / "Blocked." (`lock`) / "Attention." (`warn`) / "Information." (`info`) / "Fine." (`fine`): the markers of 4.5 |
| Sentence | One or two plain sentences, in a message row (4.6). A setting is named by its on-screen label, a record by its label (10.1) |
| Blocks | Either nothing, or the few words that go beside the button ("the name cannot be used") |

Every "Error." and every "Blocked." blocks, and nothing else does, so a marker never disagrees with
the button. "Blocked." is for an earlier step that is not done ("Blocked. Needs a prior: build or load
one in step 2, Prior."); its words come from this step's `StepState` (3.4), and the page can still be
opened and read (B7, a choice the owner has not seen). "Error." is for something on this page. (Using
"Blocked." here is a choice the owner has not seen.) Blocking facts come first, then the page's own
order.

The form's facts come from the page's rule table (5.7), the session's from the slots (5.3), the
records' from `assert_name_free` (5.5), the preview's from one engine call (below). The others are
supplied, each with its lead word: the pre-run checks' by part 9 (9.3; never "Error.", since no
reading of a check blocks: after a "Failed" the main button asks one question first, 8.4; C12, a
choice the owner has not seen); the plan and each departure from it by `plan_facts(headings)`
(7.1.9); and, only where the run simulates on the card (C13), the memory limit in force by
`memory_limit_in_force()` and what the last run on this card measured by the run history (7.2.2,
7.2.3). Facts from the form and the session read nothing from the disk; the preview alone can read
the card. Every page builds its facts in a function that takes values and no widget (13.1.5).

#### The single preview

Kept: one engine call, then formatting only, with no second copy of the engine's arithmetic in the
window (`base.py:118-126`). For Training it is `orchestrator.training_preview(cfg, prior, num_runs=,
run_size_cap=, checkpoint_every=, hw=)` (`core/orchestrator.py:481-564`), called today, without
`checkpoint_every`, from `_TrainingBudgetMixin._sync_budget` (`base.py:108-142`). It never raises
(`orchestrator.py:508-511`), and the page passes `session.cfg` as it is.

Each state of its `checkpoint` field (`orchestrator.py:448-478`) becomes one fact, in part 8's
sentence (8.4): a saved set that is continued, "Fine."; a new set, "Information."; a new set while a
saved one exists, "Attention.", with every differing setting from E2's `sibling_diffs` (11.3) by its
on-screen name (8.7), after which the main button asks first (U10; today a string of folder codes and
setting keys, `base.py:213-215`; W023); simulations not saved, or saved sets that cannot be read,
"Attention."; no configuration or no prior yet (`:543-544`), none, since the session's "Blocked." fact
says it. The counts make an "Information." fact. The memory figures are not shown: the line "fits in
one piece" is dropped (`base.py:158-185`; U10; W024, W025). A narrowing round uses the same call for
its counts only; what its "Before you start" says of the round's own simulation set is C11's
sentence (8.6). No other run has an engine preview.

#### When the facts are read again

Facts from the form and the session follow every edit. Facts that read the disk or the card are read
when the page is shown, 300 ms after the last edit to something they depend on, when any run ends,
when Records changes the store (`main_window.py:124`), on "Check again", and once more at the click,
which is the reading that counts. Today the preview runs on every keystroke in two boxes
(`posterior_tab.py:80-81`; W061, W100). They are never read while a run is live, as today
(`base_panel.py:316-317`; `posterior_tab.py:143`, `:298`, `:338`): a run has the card to itself
(U25); a test counts no call to the preview or to `assert_name_free` from a page between a run's
start and its end. At the right of the "Before you start" heading a small line says "The saved
records were read at 12:04." and, during a run, "A run is live. Read at 12:04; read again when it
ends.", or, on a page first built during the run, "A run is live. Not read yet; read when it ends."
(The 300 ms and these three sentences: a choice the owner has not seen.)

### 5.5 Name and note (U10, U14)

**For.** A record is named and described before it is made. **Replaces** the two "Save" boxes, which
rename (`prior_tab.py:91-98`, `:321-338`; `posterior_tab.py:55-62`, `:300-317`; W044, W052). It also
fills a gap: every inference stage takes `name` and `note` and the window sends neither (for example
`core/orchestrator.py:599` against `prior_tab.py:244-245`, `:2574` against `tsnpe_tab.py:157-161`;
W045, W053).

| | "Name" | "Note" |
|---|---|---|
| Checked by | `default_store().assert_name_free(kind, text)` (`core/artifacts/store.py:815-844`), the call every stage makes at its entry (for example `core/orchestrator.py:652`, `:892`); `kind` is the store's kind of the record the run writes (`store.py:33-41`) | `core.refusals.require_note("note", text)` (`core/refusals.py:370-387`): one line, at most 200 characters. No stage and no store runs this rule; each front end does |
| When | 300 ms after the last keystroke (it reads that kind's manifests), and at the click | Every keystroke (no disk) |
| Passes | "Fine. This name is free." beside the field | Nothing |
| Refused | The field outlined in the error colour; the refusal's own sentence under it; the same sentence as an "Error." fact that blocks | The same |
| Blank | Where the page requires a name, an "Error." fact "Type a name."; otherwise nothing | Allowed |
| Sent as | `name=` the stripped text | `note=` what `require_note` returned |

A name is required for a prior, a posterior and an observation, the records a later step picks from a
list, and optional elsewhere. Both fields are emptied after a run that made its record and kept, as
typed, after a refusal, a failure or a stop (both rules: a choice the owner has not seen). Neither is
ever refilled, also not after a restart (C4): "Continue from batch N" then shows the earlier name
only as a hint under "Name" (6.8). With a required name, pressing the button a second time cannot
start a second training (W021). A page that makes two records has a row for each: Inference names
its inference (`name`) and, with E4, its observation (`observation_name`), each checked against its
own kind (8.5).

The store refuses a name for three causes (not a legal name, shaped like an id, taken) and tells them
apart only in its sentence: all three carry the key `name` and nothing more (`store.py:833-844`). A
cause the window could read is beyond E1 to E12, so the words beside the button are "the name cannot
be used" and the line under the field is the engine's sentence, which says which it is.

### 5.6 The one main button

One button in the accent colour, at the right of the foot, named for its outcome (U7). It is enabled
exactly when no fact blocks, no run is live, and, on a step with nothing to run (step 1), a choice is
"Changed, not applied" or nothing is in use. When it is greyed the reason is plain text to its left,
and is also the button's accessible description.

| State | Text beside the button |
|---|---|
| One blocking fact | "Cannot start yet: the name cannot be used." |
| Several | "Cannot start yet: the name cannot be used, and 2 more above." |
| A run is live | "A run is live. Only one runs at a time." (6.12's words) with a link "Go to the run" (`go_to_run`, 3.5) |
| Nothing to apply | "Nothing to apply: these choices are in use." |

(The "and 2 more above", the link and the last sentence: a choice the owner has not seen.) No click
handler returns silently, and no input problem is only a line in the full text: a click on an enabled
main button always ends in a run, an applied step, a question or a dialog. Today the narrowing
round's button is greyed with no reason and its handler returns silently (`tsnpe_tab.py:134-135`,
`:209-210`; W058), as do `validate_tab.py:63-64` and `infer_tab.py:462-463`; the Prior tab's five
warnings (8.1; W047) and `posterior_tab.py:119-121` go to the log pane alone.

### 5.7 How a refusal is shown

| Where it arises | What the user sees |
|---|---|
| In a field the page can check before the click | The field outlined; under it the window's short sentence built from the row's label and rule, "Batches is blank." or "Batches must be at least 1." (today's live form, `base.py:93-106`, `prior_tab.py:264-279`); the same sentence as a blocking fact. No dialog |
| At the click, or from the stage on the worker | "Check your inputs": the one dialog of B2, telling, with one button (`ask` with no other answer, 4.6): the engine's sentence, and the fix sentence as `beneath`. `show_refusal` (`widgets/refusal_box.py:25-42`) builds it (5.10). When it closes, the row for `exc.field` on this page is outlined, takes the focus, and keeps the engine's sentence under it until it is edited. A refusal with no row on this page becomes an "Error." row at the head of "Before you start" until the facts are next read |
| From the stage, with the key `new_run` | Not that dialog: the page asks its own question with the engine's facts. Training reads the preview again and asks U10's question (8.4); the Narrowing round, whose region is drawn inside the stage (`tsnpe_tab.py:96-101`), asks when the stage's refusal comes back, seconds after the click, with the engine's sentence as it stands (C11), and a yes dispatches again with `new_run=True`, in place of today's tick box (B11, a choice the owner has not seen; 8.6) |
| A failure that is not a refusal | In a run: a state of the run's page (6.8); the red box goes (5.10). At a click on the window's thread (ten handlers catch one today, for example `prior_tab.py:163`): the same dialog, telling, with the error marker, the title "Error" and the error's message as today (`base_panel.py:446-451`); its type, a code name, and the traceback are the `detail`, seen only behind "Show the full error" (C7; 4.6) (a choice the owner has not seen) |

Each page has one rule table (field key, widget, rule from `core.refusals`) read by both the live
check and the click, as the Prior tab's is today (`prior_tab.py:24-37`, `:247-262`); the stage runs
the same rules again.

The engine's sentences use the engine's words today ("artifact", "simulation cache", "probe",
"Fisher"; W131, W154). E11 rewords them to U7's vocabulary (U22; 11.13), text only, in the last stage
of the build; until then the dialog and the line under "Name" carry the engine's words.

#### The fix sentence

`fields.py` stays the one table, keyed by `core.refusals.FIELDS`; `label(key)` (`:190-201`) is
unchanged.

| Today | Redesign |
|---|---|
| A place is a tab title or one of `SCREENS` (`fields.py:14-18`, `:45`), rendered by `_where` as "the X tab" or "the X screen" (`:204-218`) | A place is a step ("step 3, Training") or a part of the rail, from the shell's `STEPS` and `PARTS` (3.2), with its page where it has several ("Effective temperature, Sweep study"), rendered by its name alone; `SCREENS` goes |
| "Set it in the 'Batches' box on the Posterior or TSNPE tab." (`:58`, `:231`) | `fix_sentence(key, here=None)`, one pure function; `here` is the place showing. On a page that shows the control: "Set it in 'Batches' on this page." Elsewhere: "Set it in 'Batches' in step 3, Training, or step 6, Narrowing round." (W121) |
| `new_run`: "Answer 'Start a new run anyway' in the dialog on the Posterior tab, or tick '…' on the TSNPE tab." (`:62-63`) | "Answer the question that the start button asks on this page." (W114) |
| `name`: "Choose another name in the Save box, or in the 'Record name' or 'Comparison name' box on …" (`:96-97`) | "Choose another name in 'Name' on this page.", with the row's own label where a page words it otherwise ('Name of the inference', 8.5) |
| `note` (`:102-103`) | "Change it in 'Note' on this page." |
| `artifact`: "Select an artifact in the list on the Artifacts screen." (`:101`) | "Select a record in the table in Records." |
| `_FIXED`: "Fixed by measurement: change it in config.py, deliberately." (`:47`) | "Fixed by measurement: it cannot be changed in the window." No on-screen sentence names a code file |
| `None` for a key with no control (`:60-61`, `:71-72`, `:86`, `:125`, `:161`, `:179-186`) | A key loses `None` in the task that gives it a control; `device` keeps it |
| (none) | The keys E4 and E5 add (11.11): each names its row on the page that shows it (8.1); `plan` is the plans part's sentence (7.1.5) |

(The new sentences' wording is a choice the owner has not seen.)
`show_refusal(parent, exc, here=…)` passes `here` on. A refused delete in Records no longer comes
through it (10.2; W132).

The pin `test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it`
(`tests/test_nav_and_gating.py:1492-1638`) is rewritten first, as 13.1.3 gives: the key set against
`FIELDS` (`:1520`) stays, and the places, the verbatim sentences (`:1567-1617`), the `None` set
(`:1620-1630`) and the drive labels (`:1633-1638`) change. The read-back test (`:3989`) reads
`form.rows` and `label_text()`, not `QFormLayout` rows; the test that counts `add_help_row(` calls in
the Prior tab's source (`:3373`) is retired, the read-back covering it (13.1.2). Every page part
edits `CONTROL` and these pins in the task that adds, renames or moves a control.

### 5.8 The "?" beside a setting (U13, U17)

Today `HelpBadge` is a 16 px button with `Qt.NoFocus`; its text is a tooltip that a click pins at the
cursor (`widgets/help_badge.py:16-33`). The keyboard cannot reach it and each text is a paragraph
(W051). The texts are in six `HELP` dicts under keys of their own (`panels/inference/help_text.py:7`
and the five that 10.8 lists).

- It opens by a click on "?" or by F1 while the focus is anywhere in the row (3.7). The "?" is not a
  Tab stop, so a form's Tab order is its fields (a choice the owner has not seen).
- What opens is a small panel under the row, not a tooltip: the setting's label; its explanation, set
  as a message row's sentence with no marker (4.3), no wider than a question; its usual value, and
  the plan's value when a plan sets one; a link "More in Help". The focus moves into it. Escape, F1
  or a click elsewhere closes it and returns the focus to the field.
- The texts are the entries of Help's one table (10.8), found by field key, so a label always comes
  from `fields.label(key)` and Help lists the texts under the names on screen. Part 10 rewrites them
  (W103, W115, W125) and says where "Help" in a page's heading and "More in Help" land.
- `HelpBadge` stays as the "?", drawn as 4.6 gives; its tooltip goes, and `help_label`,
  `add_help_row` and `with_badge` (`help_badge.py:36-76`) go with the form builder they served (5.2).

### 5.9 The result area and the figure viewer (U16)

After a run, region 2 holds, in order: a sentence with its marker saying what was made, when and in
how long; a three-rule table when the record has one; the figures; a plain button for the next step,
named for where it goes (`go_to`, 3.2). The page part writes the sentence and the table. When E7 saved
the record without the name asked for (11.8), the result says so and offers "Rename…", as 8.1 words
it.

**Figures today.** `FigureStack` is a `QTabWidget` of closable, movable tabs
(`widgets/figure_stack.py:49-54`). Every figure of every run becomes a new front tab (`:109-110`) and
nothing clears them (`clear_all` has no caller but a test, `:148-150`; W067, W073). "Pop out" opens
parentless windows held only by the stack (`figure_stack.py:112-131`, `widgets/figure_window.py:18-20`), which
outlive the main window (W145, W197).

| Piece | Stays | Changes |
|---|---|---|
| `FigureStack` | The class; `add_figure`, `add_png`, `png_path`; fit to width (`:19-46`); a PNG made on the worker, never a live canvas | It shows one record's figures. `begin(record_name)` empties it when a run starts on the page or another record is opened. Tabs are neither closable nor movable. Under the figure, a caption in the book face: the figure's title, the record once the run has returned it, when it was made (the date as 4.3 gives it: B22's; on the run's page C18's) |
| "Pop out" | Both window classes and the detach on unpickling (`figure_window.py:34-51`) | The button reads "Open larger". The window is owned by the main window, so it closes with it. "Save As…" (`:142`) reads "Save as…" and opens the file chooser through the one function of 13.1.4 (today a static call, `:169`) |

(One record at a time, and tabs that cannot be closed: a choice the owner has not seen.)

Every figure the viewer shows is one the engine built, so it is a light picture in both themes (U23,
B18; 4.9). While a run is live `RunView` places this same viewer under its messages, fed by the
sink and `NewPngWatcher` as today, and the result keeps it (6.3).

### 5.10 Today's shared code: what stays, goes or changes

| File | Stays | Goes | Changes |
|---|---|---|---|
| `panels/base_panel.py` | `_refusal` (`:406-426`) | The frame and its layout persistence (`:128-203`, `:343-396`); `log_pane`, `progress_pane` and `btn_cancel` as widgets of a page; the red box of `_on_error` (`:428-457`; 5.7); `save_settings` (`:398-399`) with the sweep at closing (B6; 7.3.1 rule 1) | `_refusal` also marks the row. `refresh_local_gates` (`:337-340`) becomes the reading of the facts and of "In use". `restore_settings` (`:401-404`) puts remembered values back as choices, never applied (7.3.1 rule 2). `dispatch`, `RUN_STATE`, `_set_busy` and `set_controls_enabled` (`:66-84`, `:206-335`) pass to part 6's run controller (6.1), with the lock on region 3 and the foot's two buttons only (6.12) |
| `panels/inference/base.py` | `_StagePanel.session` (`:23-25`); the one preview call | `_budget_memory` (`:158-185`); `_derived_label` (`:63-75`) | `_sync_budget` yields facts |
| `widgets/forms.py` | | `make_form`, `apply_form_policy`, with their last caller | |
| `widgets/field_row.py` | `LabeledFieldRow` (`:15-41`), as the field cell of a row with several fields | | Captions in the small type |
| `widgets/labeled_inputs.py` | `FloatField`, `IntField`, `value_or_none`, `number_or_blank`, `PathField` | `value()` (`:25-29`, `:67-71`), once no caller is left | Drawn and grouped as 4.6 gives (the comma stripped on reading, `IntField`'s validator gone); an error state; "Browse…" opens the file chooser through the one function of 13.1.4 (today a static call, `:105`) |
| `widgets/param_grid.py` | `BoundsGrid`, `ValuesGrid`, `problems()`, `to_dicts()` | The nested scroll area, its 320 px cap and its wheel pass-through (`:21`, `:33-65`): the page scrolls | Drawn as a three-rule table. `problems()` become blocking facts and marks on the rows, not a log line (`prior_tab.py:188-191`) |
| `widgets/artifact_picker.py` | Both pickers; the id as the stored key (`:201-205`); finished records only (`:192-198`) | "+  (from scratch)" and `allow_new` (`:22`, `:46-47`, `:185-186`): loading and making are separate (U14; W015, W021) | A list read again keeps its selection and never falls to the first file, as 7.3.1 rule 3 gives (`:44-57` against `:182`, `:219`; W002, W029). The line under a picker (`:113-137`) and the suffix (`:110`) use the record's label and the kind words of 10.1 |
| `widgets/source_toggle.py` | All of it | | The page supplies the two labels |
| `widgets/refusal_box.py` | `show_refusal`, by name, with its callers | The `QMessageBox` (B2) | Builds the one dialog; takes `here` (5.7) |

Under U19 an old widget stays until the page that uses it is rebuilt.

### 5.11 Where a page hands over to the run

A page's part ends when its main button has read the fields and had its questions answered. It then
calls the run controller's `dispatch(fn, *args, spec, on_result=None, **kwargs)` (6.2) with the
stage, the arguments read at the click and the `spec` the page fills. The page gets back one of
three things: its `on_result` with the record (5.3 step 4, then 5.9); a refusal (5.7); or nothing,
when the run failed or was stopped, and then it shows what it showed before once "Back to the form"
is pressed (6.3).

### Choices in this part the owner has not seen

- A pre-run check never greys the start button: one not yet run, or one that failed, is an "Attention" line, one that does not apply an "Information" line, and after a deciding check that failed the start asks one question first, with "Cancel" preselected.
- A name must be typed for a prior, a posterior and an observation and may be left blank for other records; the name and the note are emptied after a run that made its record and kept after a refusal, a failure or a stop, though never brought back once PRISM has been reopened.
- A choice left "Changed, not applied" on another step is an "Attention" line on the page whose run reads it, and does not stop the start.
- Facts that read the disk or the graphics card are read when the page is shown, 300 ms after typing stops, when a run ends, when Records changes, on "Check again" and at the click, and never while a run is live; a small line says when they were read, or, on a page first opened during a run, that they have not been read yet.
- A missing earlier step is shown in "Before you start" with the "Blocked" marker and word rather than "Error", on a page that can still be opened and read; only "Error" and "Blocked" lines grey the button.
- A refusal about starting a new simulation set is never shown as "Check your inputs": the page asks its own question, on the Narrowing round page in place of today's tick box and with the engine's sentence as it stands.
- The question before a prior is loaded or built while a posterior is held: titles "Load this prior?" and "Build a new prior?", answers "Keep the posterior" (the default) and "Load and release the posterior" or "Build and release the posterior", with the sentences of 5.3.
- Loading a saved posterior or starting a training while a posterior is held asks nothing, and an observation held with no posterior is released by another prior without a question.
- The sentences that say what the session holds name the records only; the built configuration is no longer listed, since it is never held without its prior.
- A failure at a click that is neither a refusal nor part of a run is shown in the shared dialog with the error marker and the error's message; its type and full trace are seen only behind "Show the full error".
- The "In use" block and the run's result sit at the top of the scrolling body, above the form.
- The figure viewer shows one record's figures at a time, is emptied when a run starts on the page, and its tabs cannot be closed or moved; each figure has a caption with its title, its record and when it was made; "Pop out" becomes "Open larger" in a window that closes with the main window.
- The line under a changed choice also names what is in use ("Changed, not applied. In use: …"), and the "In use" line counts such choices.
- The texts beside a greyed main button: "Cannot start yet: …, and 2 more above.", "A run is live. Only one runs at a time." with a link "Go to the run", and "Nothing to apply: these choices are in use."
- The "?" is not a Tab stop: F1 anywhere in a row opens that setting's explanation as a small panel under the row, with its usual value, the plan's value and a link "More in Help".
- The new sentences that say where to fix a refused input, such as "Set it in 'Batches' on this page." and "Set it in 'Batches' in step 3, Training, or step 6, Narrowing round." (all seven are in 5.7); the one for a value fixed by measurement no longer names a code file.
- The "Rarely changed" button counts the settings that are not at their usual value ("7 settings, 2 not usual"), and the fold opens itself and stays open while a folded field is in error.
- No page keeps a divider the user can drag, and divider positions saved by the old layout are ignored.
- The smallest window, at which the main button must still be on screen, is 900 by 600 at Windows' ordinary text size and grows with the text size.

### What this part fixes

- W002: the bounds picker's rescan keeps the selection.
- W003: a changed choice reads "Changed, not applied", and "In use" names the bounds file the configuration was built from.
- W012: every step's page says what the session holds.
- W013: loading or building a prior asks before a posterior is released, and releases only on success.
- W015: "+  (from scratch)" goes and the picker moves to the prior just built.
- W021: the picker moves to the posterior just trained, and a required name stops a second click from starting a second training.
- W022: a page names the prior and the posterior its run will read.
- W024: the memory line that went stale is gone.
- W025: the "fits in one piece" sentence is gone.
- W029: the cell picker's rescan keeps the selection.
- W044: the prior's "Save" box, which renamed, is gone.
- W045: a prior has "Name" and "Note" before it is built.
- W047: an input problem is a fact or a refusal, never a line in the full text alone.
- W051: "?" opens from the keyboard, as a panel, not a tooltip.
- W052: the posterior's "Save" box, which renamed, is gone.
- W053: a narrowing round has "Name" and "Note" before it starts.
- W058: a greyed main button always has its reason beside it.
- W059: a refused, failed or stopped training or load leaves the held posterior in place.
- W061: the preview is read after typing stops, not on each keystroke.
- W073: the figure viewer shows one record's figures and is emptied when a run starts.
- W090: a configuration is never installed without its prior, and a missing prior blocks the button.
- W100: the saved simulation sets are read after typing stops, not on each keystroke.
- W105: no page has a divider to drag, so none moved just before quitting is forgotten; the wrong comments beside the layout code go with it.
- W109: "Name it below to keep it" is gone for a prior.
- W110: a value fixed by measurement is text, not a field.
- W112: no page carries a figure area or a progress row it does not use.
- W114: the fix sentence for a refused new run points at the question the page in use asks.
- W119, in part: the stale comment on layout saving goes with the dividers (the library line number is part 13's).
- W121: a fix sentence names the page in use, not every page that has the control.
- W128, in part: the stale comment in the layout code goes with the dividers (the other is part 10's).
- W145: a larger figure window is owned by the main window and closes with it.
- W147: as W090.
- W149: as W059.
- W155: "Name it below to keep it" is gone for a posterior.
- W197: with no parentless figure window left, closing the main window ends the program.
