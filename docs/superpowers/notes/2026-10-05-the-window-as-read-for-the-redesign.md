# The window as read for the redesign — findings, 2026-10-05

A working document, written in the redesign chat on its first day. It records what the owner has
decided so far, what a second and a third reading of the window's code established, one experiment
that was run, and what the test suites pin. `docs/STATE.md` holds the moving state; the earlier note,
`2026-10-05-the-stopped-retrain-and-the-window-route.md` ("the first note" below), is what this one
checks and extends. Two appendices hold the detail:

- `2026-10-05-the-window-as-read-findings.md` — every finding, 198 entries with ids W001–W198, each
  with what a person would see, the correction a second reader made, and the code it rests on;
- `2026-10-05-the-window-as-read-inventory.md` — the readers' maps of the window, screen by screen
  and control by control, what each stage accepts against what the window offers, where a long run
  could attach, and what the tests pin.

**How this was established.** The code is as at commit `71a53b5` (no file under `core/` or `tests/`
has changed since `3ec5bf6`). Ten readers went through `core/gui/`, the engine side, the reader pages,
the walkthrough and the earlier designs, each re-checking the first note's claims that fall in its
files. Seven second readers were then told to refute every fault the first ten had written down, and
two more sized the tests. All of that is reading: **nothing was executed by a reader.** One thing was
run, by the controller, and is described in part 2. The off-screen pictures of the window were made
on a scratch settings file and an empty scratch store. Counts: of the first note's claims re-checked,
127 held as written, 30 held with a correction and 5 could not be settled by reading; of 174 entries
the second readers went through, 97 hold, 70 hold with a correction, 1 is wrong and 6 could not be
settled by reading (five of the six are the "Quit anyway" question of part 2). The second readers
added 24 faults of their own, which only one reader has seen.

---

## 1. What the owner has decided (2026-10-05, in the redesign chat)

- **What is disliked today**, in the owner's words: the main screen is "5 buttons just slapped in the
  center"; the screens under each tab are "very cluttered, unintuitive, and hard to use/read";
  "navigating the whole app is unintuitive"; "the progress bars are confusing, like they seem
  cluttered and not interpretable"; the console output should not be in the window by default,
  people "need to only see what they need to by default"; and the design language is disliked, with
  no direction in mind ("maybe you have some ideas?"). Above all: "I require extreme clarity for
  this science application."
- **Who it is for:** it may be released, so strangers will open it with no one to ask.
- **The toolkit:** stay on PySide6 unless an approved design turns out to need something it cannot
  do; then stop and bring the blockage and the cost before proposing a change.
- **The scope:** beyond look, layout and flow, the redesign takes on (1) everything the engine can
  already do and the window does not offer or say, and (2) the small engine changes (the loss per
  epoch while fitting, a guard so that closing the window cannot destroy a finished result, naming
  every setting that differs from an earlier simulation set). Saving the fit as it goes (3) is to be
  brought separately as its own design, to accept or leave.

Decided later the same day, each from pictures or from a question with its options laid out:

- **Getting around:** the parts of the program on a rail down the left edge that never goes away;
  Home as a status page (what to do next, where the inference steps stand, what each part is for);
  the six inference steps as a chain with a state and a reason each; a strip that follows any live
  run from every screen. Grafted on: for a first-time visitor with nothing on disk, the opening that
  asks "what do you want to do?" in plain words; and in Records the "made from / used by" view of a
  record, with the next action offered on the record's own page. (Three structures were drawn as
  wireframes, each critiqued and revised: this one, a task-first guided path, and a records-first
  workbench.)
- **The look:** "from the scientific page" — dark ink on a clear page with no boxes round groups,
  every list a table with three rules, one marker shape per state always with its word, one accent
  blue for the current place, the selection and the main action, the colour-blind-safe colours of
  scientific figures for states, progress on a labelled time axis whose hatched end is the range in
  which the run should finish, charts drawn like a journal figure; lettering in the bundled Inter,
  whole sentences the program says in a book face (Cambria). Grafted on: real system dialogs and
  stock control shapes, and a "recent / all epochs" switch on the loss curve. (Three looks were
  drawn on one sample screen, each critiqued with measured contrast and revised: this one, one drawn
  from the hair bundle's own imagery, and the Windows standard played straight.) The look is to be
  proven as a small sample in PySide6 before the design is written.
- **A long run's settings: named run plans.** A plan is a small file per kind of run, kept beside
  the bounds and cell files (for example "retrain": 10,000 batches, network 256 by 10). Every launch
  still opens at the built-in values; picking a plan fills the form, and "before you start" names
  the plan and everything that departs from it or from the usual. The two measured definitions behind
  the old incident (the chi band and drive) cannot be in a plan. Science settings are still never
  restored by the window on its own.
- **The card memory limit lives in Settings, per computer, and is remembered.** This reverses the
  standing rule for that one setting ("the VRAM ceiling is never remembered"); the old fear is
  answered by showing the limit in every "before you start" and storing it in each run's record. A
  plan does not carry it.
- **Closing the window during a run asks, and keeps running by default.** The question names the
  run, how long it has been going and exactly what stopping would lose; the preselected answer is
  "Keep running" and the window stays open; the other answer stops the run, shows that it is
  stopping, waits until it really has, then closes. "Quit anyway" goes.
- **When a long run fails partway:** the run's page says what failed, what was saved and what was
  lost, keeps the full error in the run's file and offers "Continue from batch N"; before giving up
  the window tries again by itself a few times with a pause, and says plainly when a failure is of
  the kind only reopening the program can clear. An unattended restart in a fresh program is not in
  this redesign.
- **Vocabulary:** the field's own terms, each explained once where it first appears and in Help
  (prior, posterior, calibration, effective temperature); plain words for the program's machinery
  (Records, simulation set, memory limit, training, narrowing round with TSNPE given once, Model and
  options); one name per thing everywhere.
- **Checks: every check there is gets a screen** — the pre-run checks, the comparison when
  simulation ends, the checks after the fit, and also the card test and the two investigation tools
  — on one Checks pattern, each saying whether it decides pass or fail. The card test is tool code
  today, so its screen needs it moved into the engine as a step: a small engine change, to be listed
  with the others.
- **The design plugins:** their methods are applied by hand and the design is recorded in the
  project's own documents (the written design, and once built a page in the guide stating the look);
  nothing of a plugin's is written into the repository and impeccable's program is never run.

Still open when this was last updated: whether the inference steps sit inside the rail or as a track
across the top of the page (both were drawn); what Home shows a first-time visitor in detail; what
tells the owner a run has ended; keeping the computer awake during a run and a launcher with no
console window (W004, W139); the details of a plan (what it may hold, whether the window can save
one); and the separate design for saving the fit as it goes.

---

## 2. One experiment: closing the window under a run

The first note could not say what "Quit anyway" does, and no reader could settle it from the code:
it turns on what the compiled windowing library does when the program's main function returns while
a run is still on a pool thread. So it was run, once, on 2026-10-05, with no card, no store and no
training.

**Method.** A scratch script (about 120 lines, kept only in the chat's scratch folder) builds a bare
main window and wires PRISM's own `core.gui.worker.Worker`, `core.gui.streams.CancelToken`, the
stream redirect and `core.gui.panels.base_panel._png_fig_sink` exactly as `BasePanel.dispatch` does
(the worker held in a set, auto-delete off, started on the global thread pool). The stage it runs
ticks once a second for 20 s, printing a line at each tick (a cancel checkpoint, and a line for the
pump), then hands a figure to the sink — where `build_posterior` hands over its loss figure — and
returns. A timer closes the window at 3 s. The script ends the way `core/gui/__main__.py` does
(`raise SystemExit(main())`), keeps the window referenced from the exception hook as `core/gui/app.py`
does, and writes a timestamped line to a file at every event. Run off-screen, and case A again on the
real Windows platform with the window hidden: the same result.

| case | what was done | what happened |
|---|---|---|
| A, "Quit anyway" | the window closed at 3 s, no cancel | The event loop returned and `main()` returned at once; **the process stayed alive** and the stage ran all twenty ticks with no window. Then the figure hand-over **raised `RuntimeError: Signal source has been deleted`**. The interpreter's exit handlers ran only after the worker had left. |
| B, "Cancel task & quit" | the cancel requested, then the window closed at once | The run stopped at its next printed line (0.2 s), and the process waited for it to unwind before exiting. |
| C, a second top-level window left open | as A, with a parentless window open (a popped-out figure) | The event loop did **not** return when the main window closed. The run finished normally and its signals were delivered; the process was still alive, with no main window, when the experiment's watchdog ended it at 45 s. |

**What follows, joined to the reading.** In PRISM that figure hand-over sits inside the posterior's
writer (`core/orchestrator.py:1420-1431`, `core/artifacts/store.py:404-411`), the writer removes its
folder on any exception (`store.py:436`, `:479-480`), and the worker swallows the `RuntimeError` it
then meets when it tries to report (`core/gui/worker.py:207-210`). So a fit left to finish behind
"Quit anyway" would run to its end unseen, for days, and then delete the posterior it had just
written, with nothing shown or logged — while the dialog says "the task keeps running in the
background until it finishes on its own" and `docs/guide/window.md:78-80` repeats it. "Cancel task &
quit" during generation is safe as far as the process goes: it stays alive until the run has unwound,
so the rescue save can finish. During a fit that button leaves the process fitting with no window
until the epoch ends, and the fit is then lost as with any cancel.

**Limits.** The experiment used PRISM's own worker and figure sink but not the full window, a real
store or a real stage: the store's removal of the folder was read, not run. It did not load the
card, so whether a worker that outlives the window can still use it was not tested. A real check is
one row for the walkthrough, on a scratch store: a two-batch training, "Quit anyway" during the fit,
then look for a record under `posteriors/`.

---

## 3. Corrections to the first note that change the picture

The full list is section C of the findings appendix (W161–W195). The ones that matter to the design:

- **"The whole window is locked for the days the run takes" is overstated.** Only each tab's left
  column of settings is disabled (`base_panel.py:316-317`, `:333`). Moving between screens and tabs,
  the figure tabs and their pop-outs, the run's text, the splitters, the appearance menu, Settings and
  reading on the Artifacts screen all stay live. What the lock does cost: while a days-long training
  runs, no saved FDT run or sweep can be browsed or compared on its own tab (their pickers sit in the
  locked column), and no CPU-only run can start.
- **Most of what the first note lists as missing is a change to the window only.** Every stage
  already takes a name and a note and refuses a taken name first; the training stage and the
  narrowing round take an epoch ceiling, a save cadence, a resume policy, and (the round) its own
  network size, learning rate and patience; the calibration takes a seed; `ArtifactStore.rename` works
  for every kind; and each of the eight diagnostics is a public entry that takes arguments and writes
  one record. Details and the traps each control carries are in the inventory appendix ("what the
  stages accept against what the window offers").
- **What needs the engine to change:** saving and resuming the fit (large: it changes how the fit is
  driven, and by one reader's reading of sbi's loop a fit driven in chunks advances the patience
  count once per chunk); the loss per epoch as
  a log record (small, `core/SBI/train.py` only; sbi's `tracker` is called only after the loop ends,
  so it is not a live hook); all differing settings on the training preview, not the first
  alphabetically (small, read-only); a guard around the figure hand-over in the store, or a sink that
  cannot raise (small); refusal hygiene for new screens (`channel_ablation` raises plain `ValueError`,
  the hoisted "must resume" refusal carries no field key); and the memory limit and the per-batch
  memory line as arguments, which would reverse a recorded choice.
- **A run's lines can be written to a lasting file from the window alone**, at the pump's sink
  (`core/gui/streams.py:121-130`), where every finished line of all three channels passes on the
  worker thread. The FDT and sweep records already have their log on disk while they run.
- **During a fit the screen says less than the first note thought.** For the whole first epoch
  (about an hour at full size) the caption is empty: sbi prints its counter only after an epoch ends,
  so the pane shows a moving bar with no text, a solver reading of idle, then "no output for N s".
  After that only the header clock moves.
- **The passive "matches no checkpoint" line** lists every committed sibling, each with its first
  differing key alphabetically; the prior is named whenever nothing sorts before it. The substance
  stands: a bounds file swapped for another reads as "differs in param_keys", with no values.
- **The Artifacts pane's cut of long lists** also hides ten of thirteen parameter names, not only
  numbers; the rotation diagnostic's own record does show all thirteen eigenvalues, one block per
  direction.
- **The simulation cache's key** has 34 entries, seven of which are code markers or hardware, not
  science settings; no code revision is among them. The rotation switch is in the key; the network
  size, the memory limit, the save cadence, a name and a note are not.

---

## 4. The faults that can cost days of work

Each is in the findings appendix under its id, with the sequence of clicks that leads to the loss.
All of them fall inside the scope the owner chose, except where marked.

- **Closing the window during a run** (W001): the preselected button cancels the run, the question is
  the same whatever is running and never says a fit would be lost, and no test pins the dialog.
- **"Quit anyway"** (part 2 above; first note 2C, 3B item 8).
- **Cancel during a fit** (first note 2C; W005 for the FDT and sweep runs): one click, no question,
  cannot be withdrawn; the same button on every tab.
- **The screen can show one thing while the run uses another** (W003): boxes and lists stay
  editable after they have been applied and nothing marks the difference. Three ways to end up on
  the wrong bounds file without a word: a look-and-close session overwrites the remembered one
  (first note 2A); the small rescan button beside a file list puts it back on its first entry
  (W002); and loading a saved prior does not check the bounds file showing, because the three
  shipped Nadrowski files are identical in the part the load compares (W138, one reader).
- **The rotation tick is remembered** (W008, W142): left off in an earlier session it opens off, the
  run simulates from zero without the rotation, and no line of the run says so.
- **The narrowing round** (W009; first note 2D): after an inference its picker still shows the
  earlier observation while the message just printed says it can work on the new one; and it always
  trains at the default network size.
- **Nothing keeps Windows awake or reacts to a sign-out or restart** (W004), and **closing the
  console window the launcher leaves open kills the program with no question** (W139, one reader).
- **Deleting from a second window** (W006, W007, W137, W140): a cache that a run elsewhere is
  writing can be deleted after one generic question, and the run then silently rebuilds a broken
  partial cache. The owner's rule already forbids a second window during a long run; nothing in the
  program enforces it.
- **Reachable once a run can be named before it starts** (W010, W011): the name is checked again at
  the very end, before the writer exists, so a name taken meanwhile by another process loses the
  trained network; and a failure of the final write leaves the network in a folder the leftover
  sweep would remove.

Beside these, 26 + 2 entries can mislead a long run (wrong or stale reassurance about card memory,
the help that advises leaving the limit off, a blank limit silently meaning off, nothing naming the
prior and posterior a button will act on, the first hour of a fit looking hung, a crash's traceback
existing only inside a dialog, times in two clocks with no date), 63 + 10 are everyday faults, and
36 + 8 are cosmetic or wrong comments and pages. The design takes them up screen by screen.

---

## 5. What the tests pin

From two readers, not run. Nine suites hold the window: `tests/test_nav_and_gating.py` (101 test
functions) and eight others (221). They hold it by **attribute name** (about 230 names across 26
classes, almost all on widgets built off-screen with no arguments), by **wording** (labels, dialog
texts and the fix sentences that name a tab or a screen), by **dialog mechanism** (a refusal is a
`QMessageBox` read off a list the session guard fills; any other modal would stall the suite, not
fail it) and, in about thirty places, by **source text**. If classes, attribute names, labels, tab
titles and the dialogs were kept and only the arrangement changed, about 279 of the 322 would pass
unchanged. A new way of getting around, renamed steps, a different log and progress view and new
controls for settings that are pinned today as having none (`checkpoint_every`, `resume`,
`max_num_epochs`, `device` among 43 keys) break many more. The inventory appendix lists the names
the tests reach for and every pin a redesign collides with; the plan must rewrite those tests first.

---

## 6. The design plugins, as found on this machine

Read from their files; nothing of theirs has been run except ui-ux-pro-max's local lookup, which
reads its own tables and writes nothing. impeccable's start-up step runs a program that is already
in the owner's user folder (`~\.impeccable\bin\0.1.11\impeccable.exe`, dated 2026-10-05 10:48; its
launcher fetches it from the project's GitHub releases when it is missing); what that step itself
sends or writes was not established, so it has not been run. Used in full, impeccable writes
`PRODUCT.md` and later `DESIGN.md` at the repository root and a `.impeccable/` folder, fetches its
catalogue of directions from its website and sends an anonymous note when a direction is chosen. The
repository root held none of these when this was written.
