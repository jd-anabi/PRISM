# PRISM — the redesign of the window: design

**Status: APPROVED by the owner on 2026-10-06, in chat ("The design markdown file looks good").
Nothing is built.** No line of part 2 or of appendix A was struck. One point of part 2 was put as a
recommendation to decide at the review and was not answered: item 16, whether a figure's line
colours number five or eight. It is the first thing to ask before the look's foundation is planned. **Answered by the owner on 2026-10-06: seven** (section 8, row 1).
The code is as at commit `39c37c1`; no file under `core/` or `tests/` has changed since `3ec5bf6`.
**Since then:** stage 1 is built (`8b2ca92`..`f9c6899`, its exit passed on 2026-10-14); item 16 was answered on
2026-10-06 (seven line colours); what was planned or built differently from this design is under "Deviations recorded
while planning and building" (section 8).

This is the design of a complete redesign of PRISM's window: how it looks, how it is laid out, how
work moves through it, and what it must do so that a training of days can be started, watched,
stopped and judged from it. The order of work it sits in is unchanged: this redesign, then the
audit of the whole code, then the retrain from the window. No training of real size is started by
anything in it.

## How this document is laid out

- **This file** is the design at the level at which the owner approved it, and, in part 2, what
  the owner has not yet seen. It is the file to review.
- **The eleven parts** beside it, in `2026-10-05-window-redesign-design/` (`03-shell.md` to
  `13-tests-stages-docs.md`), are the exact design: every page, control and sentence, every stored
  key, each engine change, each test and each stage. About 105,000 words. Each was drafted against
  the locked record of decisions, revised against the later rulings, and corrected by an
  independent reader who checked its claims about today's code against the code (1,351 claims
  checked; 103 corrected). The implementation plan is written from the parts.
- **The assets** are in `2026-10-05-window-redesign-assets/`: the window as it is today
  (`today-*.png`); the chosen way of getting around and the two turned down (`structure-*`); the
  chosen look and the two turned down (`look-*`); the chosen look built in the real toolkit
  (`toolkit-*.png`, and the sample itself, `toolkit-sample.py.txt`); and the two locked records,
  `decisions-as-approved.md` (U1 to U20, E1 to E7) and `rulings-after-the-drafts.md` (the owner's
  U21 to U27 with E8 to E12, then the controller's rulings B, C and D on the seams between parts).
  Where the records differ, the later wins.
- **Deviations recorded while planning and building** (section 8, after appendix A) records, stage by stage, what
  differs from this file and the parts, with why and with what it costs if it was wrong.
- **The findings** the design answers are in `docs/superpowers/notes/`:
  `2026-10-05-the-window-as-read-for-the-redesign.md` and its two appendices (ids W001 to W198).

How it was made: the three design plugins the owner installed were used by their methods, applied
by hand (U20): one's way of asking what a product is for, its guidance for working tools, its
scored critique and its way of drawing a look from the subject's own world; another's lookups of
usability rules, palettes, typefaces and chart choices; the third's check against the looks that
generated designs fall into and its rules for wording. None of their programs was run, nothing of
theirs is in the repository, and no label of theirs may appear in code, tests or the guide.

---

## 1. What was decided

All of it by the owner, on 5 and 6 October 2026, from pictures or from a question with its options
laid out. The records give each in full.

**Who and what for**
- U1. The owner runs PRISM from the window. It may be released, so strangers will open it with no
  one to ask. "I require extreme clarity for this science application."
- U2. Disliked today: Home as five buttons; cluttered, hard-to-read screens; getting around; the
  progress bars; the console text shown by default; the design language.
- U3. The toolkit stays PySide6 unless an approved design is blocked by it. A sample of the chosen
  look was built in it and nothing blocked.
- U4. Scope: the look, the layout and the flow of every screen; everything the engine can already
  do that the window does not offer or say; and small engine changes, each listed for the audit.
  Saving the fit as it goes is a separate design. No unattended restart, no installer.

**The window**
- U5, U24, U26. Every part of the program on a rail that never goes away; the six steps of
  Parameter inference inside the rail as a chain, each with a marker, a state word and what it
  holds or waits for; the run's state at the rail's foot on every screen; Home as a status page
  with the one next step; on reopening mid-project one button, "Continue with these", and nothing
  loaded until it is pressed; with nothing on disk, "What do you want to do?". One PRISM at a time.
- U6, U13, U23. The look "from the scientific page": ink on a clear page, three-rule tables, one
  marker shape per state always with its word, one accent blue, the colour-blind-safe colours of
  scientific figures, progress on a labelled time axis with a hatched range, journal-style charts,
  Inter for lettering and a book face for the program's sentences; real system dialogs and stock
  control shapes. Three appearance choices; text size follows Windows; everything reachable from
  the keyboard; the smallest window keeps working. Figures take the look's colours, and a record's
  figures are always drawn light.
- U7, U22. Vocabulary: the field's own terms explained once, plain words for the machinery, one
  name per thing; the engine's refusal sentences are reworded to match.

**A long run**
- U8, U9, U15. Named run plans; the memory limit in Settings per computer, remembered; what is and
  is not remembered between launches (the training budget stays remembered; the rotation tick and
  the typed-units choice no longer are; the driven-recordings table now is).
- U10, U11, U12, U25, U27. "Before you start"; a question before any start from zero; one sentence,
  a stage line, the time axis and health figures while it runs; the console text behind a button
  and in a dated file; a stop that says what is kept and lost and can be withdrawn; closing that
  asks and keeps running by default; after a failure what was saved, "Continue from batch N" and a
  few tries by itself; the end said until acknowledged; Windows kept awake and given a reason
  against a shutdown; a launcher with no console window. During a run saved effective-temperature
  runs can be browsed, not compared.

**The pages**
- U14, U16, U17. Applied against pending; the double-duty buttons split; a record named before it
  is made. Every check there is on one Checks page, six of them deciding pass or fail. Records with
  settings in full, figures and complete logs for every kind, Rename for every kind, and the next
  step offered on a record's page. Help in the window's own words. A run history.

**The engine and the build**
- U18, U21, U22, U23. Twelve small engine changes, E1 to E12 (part 4 of this file); a stated list
  left to the audit.
- U19. Test-first; every task reviewed and gated; nine stages, the window usable after each.
- U20. The design plugins' methods by hand; the design recorded in the project's own documents.

---

## 2. What the owner has not seen: read this first

Writing the design exactly meant choosing many things no question had settled. Each part ends with
its own list of such choices; all of them are gathered in appendix A (272 lines). Most are detail
of wording, order and layout. The seventeen below are the ones that touch something already approved
or that the owner would want a say in. **Approving the design approves these and the appendix
unless a line is struck.**

1. **The checks headline for a posterior trained without the frequency response.** Three of the six
   deciding checks cannot be run on it. The headline then reads "n of 3 deciding checks passed; 3
   do not apply to this posterior", and the step reads "Done" when those three have passed. This
   amends "always n of 6" for that one case. *Saying no:* the headline stays "of 6" and such a
   posterior can never read done.
2. **The window's own three tries follow only an out-of-memory failure of a training or a narrowing
   round**, after pauses of 1, 3 and 10 minutes; a try that saves at least one more batch starts
   the count again; a try that fails without saving one ends it and says PRISM must be reopened.
   Any other failure is reported at once with "Continue from batch N". *Saying no:* every failure
   is retried, including ones that will fail the same way each time.
3. **Not every run leaves a file in the run history.** A run that only loads a record, a Live
   simulation and a video export leave none; they also leave nothing to acknowledge at the rail's
   foot. *Saying no:* each load of "Continue with these" would leave an end to acknowledge.
4. **The command line is touched in four places**, none a flag: E1 prints a line per epoch; E6
   changes the tails of eight refusal lines; E11 rewords the engine's refusal sentences, which is
   218 sentences in 20 files; and **E10 changes what a command-line run does at its last write**: a
   failure while the final figure is drawn no longer fails the run, which ends with its record and
   a warning. E7 (a name taken during the run) is opt-in and only the window opts in, so from the
   command line a name taken during a run still costs the trained network.
5. **Plans.** A plan fills the forms when the button "Fill the forms from this plan" beside the
   list is pressed; choosing in the list alone changes nothing. Before a fill replaces values that
   were typed, the window asks and lists them. Plans live in one folder per model, are written
   under the engine's setting names with the on-screen label as a remark on each line, hold only
   what differs from the built-in values plus the bounds file, are tracked in git like bounds and
   cell files (and committed before a long run, so its records are not stamped as made from a
   changed tree), and none is shipped by this redesign.
6. **"Accept this difference."** When the unusable-recordings total is over 3 and up to 5 points
   from the audit, the check reads "Look closer" and counts toward the headline only once the owner
   presses "Accept this difference" on its page; the acceptance is recorded with its date.
7. **A pre-run deciding check that failed does not grey "Start training"**; the start then asks one
   question first, with "Cancel" preselected. A check not yet run never blocks.
8. **A run's files (its dated full text and its history) live in a `runs` folder beside the
   settings file**, per user and per records folder in use, not inside the records folder.
9. **"Expected epochs (for the time estimate only)"** is a new field under "Rarely changed" on
   Training, which a plan may hold. It never limits the fit. Blank means the epoch count of the
   last finished training of the same kind; with neither, the page says only when the fit cannot
   end before. The first retrain from the window has no earlier training to read, so without this
   field its time axis would show no expected end.
10. **Each part of the program is built when it is first opened**, not at start-up, and a part that
    fails to build shows an error page while the rest works. **A step that cannot start yet can
    still be opened and read.**
11. **Names.** A name is required for a prior, a posterior and an observation, and optional for
    other records. A name and a note are never refilled, also not by "Continue from batch N" after
    a restart: both open blank with the earlier name as a hint.
12. **The numbers the deciding checks are judged by** (the expected share of unusable driven
    recordings and its widths, the 3- and 5-point limits, the two parts of the input the input-use
    check must find healthy) are in no engine code today. They become named constants in the
    window's Checks code and go on the audit's list as science numbers living outside the engine.
13. **Type and dates.** Console text (a run's full text, a record's log) is set in a fixed-width
    face, a third typeface. Dates read "3 Oct 2026, 14:02"; on a run's page a time within seven
    days carries its weekday.
14. **The order of the build.** Stage 4 delivers the pages of steps 1 to 3 with plans, so the plan
    picker has its page; stage 5 delivers Checks, Inference and the Narrowing round. The first-time
    opening's page is built in stage 2 and its help completed in stage 8.
15. **Two edges of the engine changes.** E6 also turns the one plain error of the third
    identifiability method, raised before anything is spent, into a proper refusal. E7 is not
    carried to an inference made together with its observation.
16. **A figure with more curves than colours.** The look's figures take five line colours (ink and
    the four colour-blind-safe hues). A figure with more unnamed curves than that repeats colours,
    where today the plotting library's ten are used. It shows first in the calibration's rank
    figure, which has one curve per parameter: thirteen curves in five colours. Telling curves apart
    by line style would change the plotting code beyond colours, which U4 rules out. *The
    alternative, which I recommend deciding at this review:* lengthen the cycle to eight
    colour-blind-safe colours, which halves the repeats and stays within "colours only". **Answered 2026-10-06, from a picture of that figure drawn each way: seven colours** (ink, blue, vermilion, green, orange, sky blue, purple); the eighth of the standard set, yellow, is 1.3 to 1 on a white page. Section 8, row 1.
17. **Seven things are settled by a small experiment before they are built**, because reading could
    not settle them: whether the toolkit already follows Windows' text-size setting; how the
    one-PRISM lock behaves when its holder dies; what Windows does with a registered shutdown
    reason and how long it then allows; whether a try in the same program can succeed after the
    card runs out of memory; whether PRISM runs correctly with no console; whether a figure drawn
    by the window can collide with one the run is building; and how to stage a failed save for the
    walkthrough. Each is a task with a stated fallback.

---

## 3. The design in brief

Each paragraph names the part that carries the detail.

**3.1 The shell** (part 3). A rail 220 px wide holds PRISM's name, Home, the five working parts,
Records, and at its foot Your own models, Settings and Help; under Parameter inference the six
steps form a chain whose state is computed by one rule from what the session holds. The run's
state at the foot has eight states (idle, loading, running by stage, stopping, needs you, finished,
stopped, failed) and takes the user to the run. Home shows one next step chosen in a fixed order
(a live run, an end nobody has seen, a failed run that can be continued, "Continue with these",
then the steps in order), the chain, the parts with a line each, the five newest records and the
three latest runs. Keys: Ctrl+1 to Ctrl+9 for the parts, F6 between the rail and the page, F1 for
help, Escape for the safe answer. Nothing at start-up touches the card or runs another program.
One PRISM at a time is a lock beside the settings file.

**3.2 The look** (part 4). Thirty named colours with light and dark values, each with one job; a
type table (Inter at 12, 14, 16, 18 and 24 px, with digits of equal width only where numbers stand under one another:
section 8, row 37; the book face at 16 px in
four places only; a fixed-width face for console text); a size table; ten marker shapes; the shared
pieces and how each is made (a rule of the style sheet, a role of the palette, or a small painted
widget); the rules for what stays out of the style sheet, learned from the sample; the one
shared question dialog in the system's frame; figures by figure settings alone, always light when
the engine draws them; a table of measured contrast that a test holds.

**3.3 A page** (part 5). Four regions: the heading and its sentence; "In use"; the form (main
settings, then "Rarely changed" folded with its values readable); and a foot that stays in view
with "Before you start", name and note, and the one main button, greyed with its reason. What a
step holds changes only when a run returns its record. A refusal is the shared question with the
engine's sentence and the window's line saying where to fix it, and the field is marked. A "?"
beside a setting opens its explanation from the keyboard.

**3.4 A run** (part 6). A run is shown in place of the page that started it. Today's tested
plumbing (the worker, the stream redirect, the cancel token) moves into one controller almost
unchanged; what is new is small: every line is written to a dated file as it arrives, the token can
withdraw a stop, the outcome and any error go to that file, the figure hand-over cannot raise, and
a small history file per run records what was started and how it ended. The time axis is drawn for
the stage that is running. Closing waits for the run to unwind; a failure page says what was saved.

**3.5 Plans, "This computer", and what is remembered** (part 7). The plan file's form, every
setting a plan may hold with the stage argument each reaches, what it may not hold,
the reader in the engine (E5) and its refusals; "This computer" (the memory limit, keep-awake); and
a table of every stored key marked kept, added, removed or changed, with the rule that a pick is
written when it is applied.

**3.6 The steps** (part 8). For Model and options, Prior, Training, Inference and the Narrowing
round: a table of every control (its new label, today's label and key, main or rarely changed, its
built-in value, the stage argument it reaches, whether a plan may hold it), the facts of "Before
you start" and where each is read, the buttons, the questions, the result and the next step. A
table turns the thirty-four entries of a simulation set's identity into on-screen names, for the
question before a start from zero.

**3.7 Checks** (part 9). Twelve checks in four groups, each with the engine entry it calls, what it
needs, what it costs, the record it writes and the field its verdict is read from; the rule for
which records count toward the headline for the posterior and prior in use; each check's small
form and result page; the comparison after simulating; the timing of a short fit in a store that
is not the real one; the card test as one engine step.

**3.8 Records and the other parts** (part 10). Records as one table and a page per record;
Effective temperature as three pages (One cell, Sweep study, Compare); Live simulation; the
Reduction map; Your own models; Settings; Help with a glossary of the science terms; the
first-time opening.

---

## 4. The engine changes

Twelve, each small, none changing what a simulated row holds, a simulation set's identity, or a
setting in `core/config.py` (E8 adds one path line there, which the owner was told of). Part 11
gives each exactly, with the tests that pin today's behaviour and the tests it needs first.

| id | what | on the command line |
|---|---|---|
| E1 | The fit writes one line per epoch: training loss, validation loss, time taken, epochs since the best. | prints that line |
| E2 | The pre-run preview returns every setting that differs from each saved simulation set. Read-only. | none |
| E3 | The card test becomes an engine step, so it can have a screen. | the tool calls the step; its output is unchanged |
| E4 | An observation can be named. | none (no new flag) |
| E5 | A reader for run plans: new code, no stage changes. | none |
| E6 | Refusals that say where to fix it: the two training refusals a continue policy reaches, the input-use check's five plain errors, one of the third identifiability method. | the tails of those eight lines change |
| E7 | A name taken during the run cannot cost the result (saved under its id, with a line saying so). Opt-in; only the window opts in. | none |
| E8 | One line in `core/config.py` names the plans folder. | none |
| E9 | A simulation set keeps its name when saved again or reused. | none |
| E10 | A failure while the engine draws the last figure of a training cannot remove the finished posterior. | a run that failed there now ends with its record and a warning |
| E11 | The engine's refusal sentences are reworded to the window's vocabulary. Text only; last stage. | 218 sentences read differently; flags unchanged |
| E12 | The plotting code takes the look's four line colours; no line style and no number changes. | figures' line colours change |

The card test on the graphics card is run once, at the end of stage 3, after E1, E3, E7 and E10
have landed, and again at the end only if code that moves tensors changed after it. The owner is
told before it runs.

---

## 5. What becomes of the findings

Part 12 gives a row for every finding W001 to W198. Of the 198:

- **100 are fixed** by a decision.
- **74 are fixed in passing**: the rebuilt screen no longer has the thing at fault. Twenty-five of
  these rest on a choice the owner has not seen, and fall back to "left as it is" if that choice
  is struck; part 12 marks each.
- **12 are left to the audit.**
- **10 are left as they are**, each with its reason (by design, out of scope, not a fault, or never
  measured).
- **2 are dropped**: the second readers found one wrong and one unsettled.

Twenty-four of the 174 closed rows leave a remainder, eleven of them on the audit's list.

**The findings that can lose days of work, and what becomes of each**

- Above every item: the fit is still not saved as it goes (U4 makes that a separate design); any end
  of the program during a fit loses the whole fit, and the simulation set survives.
- W001 FIXED (U12; parts 6, 13): the close question names the run and what stopping loses, and "Keep
  running" is preselected.
- W002 FIXED (U15; parts 5, 7, 8): the rescan keeps the selected bounds file.
- W003 FIXED (U14, U5; parts 5, 8): "In use" against "Changed, not applied"; a run uses what is
  applied.
- W004 FIXED (U11, U9, U27; parts 6, 7, 13): keep-awake, and a reason that holds a sign-out or
  shutdown. Still exposed: shutting down anyway during the fit loses the fit; a shutdown Windows
  forces without asking still ends the run; whether the registered reason alone makes Windows wait
  is settled only by a scratch test in stage 3.
- W005 FIXED (U12, U17; parts 6, 10): every stop that loses work asks first. Remainder, left as it
  is: an Effective temperature run or sweep still cannot be resumed.
- W006 FIXED (U17; part 10): deleting a finished simulation set says what it holds.
- W007 and W137 FIXED for the wording (U17; part 10). Remainder to the audit (item 7): the store
  still deletes with no check that anything is writing.
- W008 FIXED (U15; parts 7, 8) and W142 FIXED (U15, U11, U10; parts 8, 6): rotation is no longer
  remembered and is always stated. W142's remainder to the audit (item 16): the run's own log has no
  line when rotation is off.
- W009 FIXED (U16; part 8), W026 and W194 FIXED (U16, U8; part 8): the narrowing round opens on the
  observation just made and shows and passes its network settings.
- W010 FIXED (E7; part 11) for a run started from the window. Remainder, left as it is by ruling
  C22: from the command line a name taken during the run still costs the trained network.
- W011 LEFT TO THE AUDIT (item 6): a failed last write still leaves the trained network in a folder
  nothing can load, and clearing leftovers would remove it. Part 10 names such a leftover before it
  is cleared; part 6's failure page says the file may still be on disk.
- W138 FIXED (U16; part 8): a saved prior shows the bounds file it was built on and warns. Remainder
  to the audit (item 4): the engine still loads a prior without checking the whole bounds file.
- W139 FIXED (U12; part 6): the launcher leaves no console window that can end PRISM.
- W140 LEFT TO THE AUDIT (item 8): a simulation set deleted under its run is still rebuilt broken
  without a word; since U26 that takes the command line.
- W143 FIXED (U26; parts 3, 13): a second start brings the open window to the front.
- W182 LEFT TO THE AUDIT (item 7): no lock between programs; since U26 the other program is the
  command line.
- W162, W173, W178 FIXED (U12, U18; part 6, W178 also 13) and W164 FIXED (U12; parts 6, 13): "Quit
  anyway" goes and the window cannot close under a live run.
- W185 FIXED (U18, U12; part 6) and W192 FIXED (U12, U18, E10; parts 6, 11): a figure that cannot be
  shown or drawn cannot remove the finished record, on either front end for E10 (C22).
- W172 FIXED (U12; part 6): a stop asks first and can be withdrawn. Remainder, left as it is: a
  confirmed stop still loses the fit.
- W174 and W193 FIXED (U12; part 6): the window waits for the run to stop, and for the rescue save
  during simulation, before it closes.

**Left to the audit** (the audit chat starts from this list)

- 1. The memory planner's arithmetic behind today's 'fits' line: one drive counted, four built
  (W025's remainder). U18's own.
- 2. Failing fast after a memory failure that cannot be recovered (no finding; the first note's lead
  3A.2). U18's own.
- 3. Dates in a record's own log lines (W096; W032's remainder). U18's own.
- 4. The prior is not checked against the whole bounds file inside the engine (W138's remainder);
  beside it, the engine's posterior load takes no prior (W091's remainder,
  core/artifacts/store.py:1302) and the command line's comment claims a pairing check not made
  (W159, core/tool/config_args.py:351-353). U18's own, with two additions.
- 5. A simulation set knows nothing of the code, or of a user model's equations (W191, W037). U18's
  own.
- 6. A failed final write leaves a trained network in a folder with no record (W011;
  core/artifacts/store.py:482).
- 7. No lock or in-use check between programs: delete, set a note and clear leftovers under a live
  run; since U26 the other program is the command line (W182; remainders of W007, W137, W031;
  core/artifacts/store.py:52-58, :918-947, :1024-1034).
- 8. A simulation set deleted under its run is rebuilt as a broken partial set and the failure is
  swallowed (W140; core/SBI/training_checkpoint.py:372, :465-474).
- 9. An unreadable set record is rewritten as if new (W160; core/artifacts/store.py:1570-1588).
- 10. Reusing a complete set replaces its recorded generation time (W186;
  core/artifacts/store.py:1576-1577).
- 11. A posterior's id and date are those of its end; durations cover the last process only (W136;
  core/artifacts/store.py:846-850).
- 12. The repeated calibration reports nothing and cannot be stopped during its rank step (W095;
  core/diagnostics/sbc.py:172).
- 13. The memory-line setting at zero divides by zero; a non-number stops the program while it loads
  (W097; core/SBI/pipeline.py:269).
- 14. The training stage reads the prior before its load branch, so a load with none is a crash, not
  a refusal (remainders of W090, W147; core/orchestrator.py:902).
- 15. An Effective temperature run stopped after its first checks is marked complete (W070's
  remainder; core/FDT/fdt_pipeline.py:316, :348-349).
- 16. A training with the rotation off logs no line saying so (W142's remainder;
  core/orchestrator.py:1255-1256).
- 17. A calibration on a round just trained records an empty list of consents (no finding; the first
  note's 2D and lead 15; core/orchestrator.py:1488).
- 18. The numbers the deciding checks are judged by are science numbers kept in the window's Checks
  code, not in the engine (no finding; B10 lists them, part 9 names the constants).
- 19. The four engine additions the owner listed for the audit (U21): E8, E9, E10 and the setting
  names E4 and E5 add (W130 for E9, W192 for E10; today core/config.py:181-184,
  core/artifacts/store.py:1582, core/orchestrator.py:1429-1431).
- Kept off the list on purpose: the remainders of W036 (the fitting library's rule), W010 (ruling
  C22) and W154 (wording the engine keeps).

---

## 6. How it is built

Test-first. Every task has a review and one one-process fast gate, started together; never two
pytest processes; no source edit while a suite runs. Nine stages, the window usable and its suites
green after each:

| stage | delivers | engine changes |
|---|---|---|
| 1 | The look's foundation: the tables of colours, type and sizes, the style sheet, the painted pieces, the figure settings; the old layout wearing the new look. | E12 |
| 2 | The shell: the rail, the chain of steps, Home, the first-time opening's page, one PRISM at a time. | none |
| 3 | The run: its page, the full text in a file, stop and withdraw, close, failure and the tries, the end, the run history, keep-awake, the shutdown reason, the launcher. | E1, E3 (engine half), E6 (training), E7, E9, E10 |
| 4 | Steps 1 to 3 (Model and options, Prior, Training), plans, "This computer", applied against pending, what is remembered. | E2, E5, E8 |
| 5 | Checks, Inference, the Narrowing round. | E4, E6 (checks) |
| 6 | Records. | none (E9 landed in stage 3, so the card test passes through it) |
| 7 | Effective temperature, Live simulation, the Reduction map, Your own models. | none |
| 8 | Help and the first-time opening's help. | none |
| 9 | The engine's sentences reworded; the guide pages; the owner's walkthrough rows; the final gates. | E11 |

**Tests** (part 13.1). Today 322 tests hold the window, by attribute name, by wording, by the
dialog mechanism and in about thirty places by source text. Each gets a stated fate: a pin on
behaviour (what a click does, what is refused, what is written, what is never remembered, stop and
close, figures) is carried over whatever the layout; a pin on wording, structure or source text is
rewritten to this design before the code it covers. Fourteen suites are new, each named for the
behaviour it pins: the look (its contrast, its painted pieces drawn into images at four scalings),
the shell, the chain of steps, the run's states, the run's text and history, stop and close, the
tries after a failure, "Before you start", plans, applied against pending, Checks, a record's
page, Help, and a scan that no retired word is left on screen. The slow set (five tests, about 75
minutes) runs at the ends of stages 1, 3, 5 and 9. Tests that need the real screen stay marked as
such and create hidden windows only.

**The owner's rows** (part 13.4). Thirty-three new rows for the display walkthrough, G1 to G33, in
the checklist's own format: what only a real screen shows (the system's frame on a question,
pointer-over states, the focus ring under a real keyboard, 125 % and 150 % scaling, the smallest
window), every stop, close and failure path on a small training of about a quarter of an hour, a
plan applied and departed from, the memory limit across a launch, one PRISM at a time, a shutdown
during a run, and each rebuilt page once. No row asks for a training of real size or touches the
records kept from the stopped retrain.

**The pages** (part 13.3). `docs/guide/window.md` and `getting-started.md` are rewritten; the
window parts of `architecture.md` and `rules-and-traps.md` and the model-builder part of
`recordings.md` are brought up to date; a new page states the look (U20). Replacement wording for
`CLAUDE.md`'s rule on what is remembered, and for its launch line, is given in parts 7.3.3 and
13.3.4 for the owner to approve; `CLAUDE.md` is not edited before that. The retrain runbook's
window route is the audit chat's.

**The stages in tasks**, as part 13 estimates them: 11, 10, 17, 15, 12, 7, 10, 5 and 8, about 95.

Size: 70 to 100 tasks. The plan should be written and approved one stage at a time, each stage's
plan from the parts as they stand when the stage before it has landed.

---

## 7. What is not in this design

- **Saving the fit as it goes.** Its own design, to follow this one. Until it exists, any end of
  the program during a fit loses the whole fit; the simulations are kept.
- An unattended restart in a fresh program; an installer; a change of toolkit; other languages.
- Any change to a figure's content beyond its colours, axes and lettering; the Reduction map's
  science and outputs; a record kind for the Reduction map.
- Any change to a flag of the command line; the retrain runbook's window route (the audit chat
  writes it once the code is settled).
- Anything on the audit's list in part 5.

---

## Appendix A. Every choice made in drafting that the owner has not seen, part by part

### Part 3, the shell (30)

- For a posterior trained without the frequency response, the Checks step and Home count only the
  three deciding checks that apply: Home's headline reads "n of 3 deciding checks passed; 3 do not
  apply to this posterior", and the step reads "Done" once those three have passed. This amends
  "always n of 6" for this one case.
- A step that cannot start yet can still be opened and read, its main button greyed with the reason;
  today such a tab cannot be opened.
- Each part of the program is built when it is first opened, not at start-up; one that fails to
  build shows an error page with "Try again", and the rest works.
- A run's end counts as seen once its page has been showing while PRISM's window was the active one.
- Home picks its one next step in a fixed order: a live run, an end nobody has seen, a failed run
  that can be continued, "Continue with these", then the steps in order.
- Home offers "Continue from batch N" for the newest training or narrowing round that failed or was
  not seen to end, with or without a reopening, until its simulation set is finished or a later run
  of that kind starts (after a failure in the fit, until a later run of that kind starts or the
  saved simulation set is removed); after a stop Home shows the ordinary next step.
- "Continue with these" applies the kind of observation that was in use last time, and every other
  option at its built-in value or the plan's; Home's sentence names the kind.
- "Continue with these" loads the saved posterior only if it was trained on the prior remembered
  with it.
- A run that only loads a saved record shows "Loading…" at the rail's foot and then nothing: no end
  to acknowledge and no file in the run history, which was to keep one per run.
- The smallest window is 900 by 600 at Windows' ordinary text size and grows with the text size,
  held to the screen; the decision gave 900 by 600 alone.
- The idle sentence at the rail's foot reads "Only one run at a time."; the built sample read "Only
  one thing runs at a time.".
- Checks counts as a step in hand: after a training the chain shows Checks as "Now" and Inference as
  "To come" while a deciding check is still open, though Inference can be run.
- The Narrowing round is optional: never the step in hand, and "To come" once it can start.
- The first-time Home shows only with no record, no run history for the records folder in use and
  nothing loaded, and offers four choices with "Watch a live simulation" as the main button (the
  mock-up had two).
- The first-time page is built with Home in the second of the nine stages and its help completed in
  the eighth, where the list of stages names the first-time opening.
- After a crash the next start takes over the lock the dead PRISM left, without asking; a second
  start while PRISM is open shows nothing of its own.
- The accent bar marks the open part's own row when no step is open (the sample showed it only on a
  step).
- At the smallest window with the six steps showing, the rail's whole list scrolls (by about 54 px)
  under a fixed name and above a fixed run's state; the alternative is tighter rows.
- In the rail the Checks step reads "3 of 6 passed" without "deciding checks", and words too long
  for a row are cut with "…"; the tooltip and Home give them whole.
- A part's row in the rail carries no mark for a live run; only the foot says it.
- With nothing running and nothing unseen, the run's state at the foot is plain text, not a button.
- "Parameter inference" opens the step in hand the first time in a launch and the step last shown
  afterwards.
- Pages change at once, with no slide.
- Home lists the five newest records and the three latest runs.
- Five keys are added to the mock-up's: Shift+F6, Ctrl+7, Ctrl+8, Ctrl+9 and F1.
- Escape outside a question only closes what is open or returns from the rail to the page.
- Access keys (Alt and a letter) go only on a page's main button and the buttons in its heading.
- The built-in models are written "Nadrowski", "Hopf" and "BP"; a model of the user's own as it is
  stored.
- A stored "Auto (time of day)" is read as "Follow Windows"; the two removed appearance settings
  stay in the settings file, ignored.
- The appearance choices are offered only on the Settings page; the gear's pop-up list goes with the
  gear.

### Part 4, the look (22)

- When Windows' text size is raised, spacing, controls and the smallest window grow with the
  lettering; line weights and corner radii do not. The smallest window is therefore 900 by 600 only
  at Windows' ordinary text size, and larger above it, though never larger than the screen.
- A figure's lines are coloured from a set of five, in the order ink, blue, vermilion, green,
  orange; a figure with more lines than that repeats the colours, and nothing but colour tells such
  lines apart (today such lines take the plotting library's ten colours). It shows first in the
  calibration's cumulative rank figure, which has one curve per parameter.
- The exported video keeps its white page and black lettering; its axes and typeface take the same
  style as every other stored figure, and its trace turns from blue to ink.
- In a dialog that asks for one line of text ("Save as a plan…", "Rename…"), Enter means the main
  answer once the text is acceptable and does nothing before; Escape cancels. In every other
  question Enter means the safe answer.
- States that no picture drew reuse the ten shapes: "Needs you", "Look closer", "Not compared" and
  "Changed, not applied" get the attention triangle, "Being written" and a running check the blue
  dot, a check that does not apply the dash, a leftover or checks-only record the information ring,
  an unreadable record the error square.
- The padlock carries two words: "Blocked" in general, "Locked" when the cause is a live run.
- Text quoted from the console (a run's full text, a record's log, a full error) is set in a
  fixed-width typeface, a third beside the look's two.
- The question dialog can also carry a list under its sentence (one item to a line, scrolling beyond
  eight), a closing line and, for a failure, the full error behind a link "Show the full error";
  none of the three was pictured.
- When no end can be estimated, the time axis has no hatched range and runs to the earliest moment
  the run could end, or only to now; a firm end has no hatched range either. Neither was pictured.
- On the run's page a time within seven days is written "Sun 4 Oct, 12:10", where the pictures had
  "Sun 12:10"; on the time axis's ticks a day is written "2 Oct", where the pictures had the
  weekday's name.
- Figures the engine builds keep their own text sizes; only their typeface, colours and axis style
  change.
- A figure's legend stands on a patch of the page's colour with no outline (the sample's figures had
  no legend).
- The area inside a figure's axes is filled with the page's colour instead of being left
  transparent; the two look the same.
- Quantities of four or more digits carry a comma; identifiers such as a seed or a record's id do
  not.
- Numbers and dates are written the same on every computer (a full stop, a comma, English letters
  for the month and the weekday), whatever Windows' regional setting.
- A setting's explanation and Help's running text are set in the book face in the same way as a
  message's sentence, with or without a marker, so that the book face keeps its four uses.
- Links are in the accent blue, a fourth use beside the current place, the selection and the main
  action; the chosen mock-up drew them so.
- A value that cannot be edited is shown as text, never as a greyed field; the sample showed this
  for one row only, and the two measured definitions are shown so.
- A fold, when opened, shows its settings rows in place of the one-line summary, its chevron
  pointing down.
- A table that can be sorted shows a small chevron after the name of the column it is sorted by.
- If Cambria is missing on a computer, its four places fall back to Georgia and then to Inter.
- The small icon font and its builder script leave the repository (git history keeps them); every
  glyph is painted.

### Part 5, the anatomy of a page (19)

- A pre-run check never greys the start button: one not yet run, or one that failed, is an
  "Attention" line, one that does not apply an "Information" line, and after a deciding check that
  failed the start asks one question first, with "Cancel" preselected.
- A name must be typed for a prior, a posterior and an observation and may be left blank for other
  records; the name and the note are emptied after a run that made its record and kept after a
  refusal, a failure or a stop, though never brought back once PRISM has been reopened.
- A choice left "Changed, not applied" on another step is an "Attention" line on the page whose run
  reads it, and does not stop the start.
- Facts that read the disk or the graphics card are read when the page is shown, 300 ms after typing
  stops, when a run ends, when Records changes, on "Check again" and at the click, and never while a
  run is live; a small line says when they were read, or, on a page first opened during a run, that
  they have not been read yet.
- A missing earlier step is shown in "Before you start" with the "Blocked" marker and word rather
  than "Error", on a page that can still be opened and read; only "Error" and "Blocked" lines grey
  the button.
- A refusal about starting a new simulation set is never shown as "Check your inputs": the page asks
  its own question, on the Narrowing round page in place of today's tick box and with the engine's
  sentence as it stands.
- The question before a prior is loaded or built while a posterior is held: titles "Load this
  prior?" and "Build a new prior?", answers "Keep the posterior" (the default) and "Load and release
  the posterior" or "Build and release the posterior", with the sentences of 5.3.
- Loading a saved posterior or starting a training while a posterior is held asks nothing, and an
  observation held with no posterior is released by another prior without a question.
- The sentences that say what the session holds name the records only; the built configuration is no
  longer listed, since it is never held without its prior.
- A failure at a click that is neither a refusal nor part of a run is shown in the shared dialog
  with the error marker and the error's message; its type and full trace are seen only behind "Show
  the full error".
- The "In use" block and the run's result sit at the top of the scrolling body, above the form.
- The figure viewer shows one record's figures at a time, is emptied when a run starts on the page,
  and its tabs cannot be closed or moved; each figure has a caption with its title, its record and
  when it was made; "Pop out" becomes "Open larger" in a window that closes with the main window.
- The line under a changed choice also names what is in use ("Changed, not applied. In use: …"), and
  the "In use" line counts such choices.
- The texts beside a greyed main button: "Cannot start yet: …, and 2 more above.", "A run is live.
  Only one runs at a time." with a link "Go to the run", and "Nothing to apply: these choices are in
  use."
- The "?" is not a Tab stop: F1 anywhere in a row opens that setting's explanation as a small panel
  under the row, with its usual value, the plan's value and a link "More in Help".
- The new sentences that say where to fix a refused input, such as "Set it in 'Batches' on this
  page." and "Set it in 'Batches' in step 3, Training, or step 6, Narrowing round." (all seven are
  in 5.7); the one for a value fixed by measurement no longer names a code file.
- The "Rarely changed" button counts the settings that are not at their usual value ("7 settings, 2
  not usual"), and the fold opens itself and stays open while a folded field is in error.
- No page keeps a divider the user can drag, and divider positions saved by the old layout are
  ignored.
- The smallest window, at which the main button must still be on screen, is 900 by 600 at Windows'
  ordinary text size and grows with the text size.

### Part 6, a run from start to end (28)

- The window tries again by itself only after the graphics card runs out of memory in a training or
  a narrowing round, and reports any other failure at once; this narrows U12's "tries again by
  itself" to the one kind of failure a pause can cure.
- Its three tries follow pauses of 1, 3 and 10 minutes, and a try that saves at least one more batch
  starts the count again.
- It stops trying and says PRISM must be reopened when a try runs out of memory without saving a new
  batch, or when the card reports a failure that is not about memory.
- A run that only loads a record shows "Loading…" at the rail's foot, and it, a Live simulation and
  a video export leave no history file, no text file and no end to acknowledge; this narrows U17's
  one small file per run.
- A warning of the engine's that the window has no sentence for is shown on the run's page as an
  "Attention" message that begins "From the engine:" and gives the engine's sentence as it stands;
  every other message there is the window's own sentence.
- A live or just-ended run is shown in the place of the form that started it, not on a page of its
  own.
- After reopening, "Continue from batch N" fills and applies the earlier steps from the run's
  history and opens the run's own step, where one press starts it once a name is typed; the name and
  the note are not brought back, and the earlier name is shown as a hint under "Name".
- A stopped run whose simulation set holds a batch offers "Continue from batch N" on its own page,
  as a failed one does.
- After a failure during the fit the button reads "Fit again on the saved simulations", and stays on
  offer until a later run of that kind starts or the saved simulation set is removed.
- When the engine cannot draw or save its last loss figure, the run ends with its posterior and a
  warning instead of failing, from the window and from the command line alike (the change is the
  engine part's).
- A run's full text and its history are kept in a folder beside the settings file, per user, not in
  the records folder; PRISM deletes nothing there except the two files of a start that was refused.
- A refusal counts as a start that never was, and its two files go, only when no simulation step was
  taken, no later stage had begun and no record was written; any other refusal ends the run as
  failed.
- Closing the window asks about a model's unsaved edits after the question about a live run, and no
  run is stopped until both are answered.
- PRISM starts through the windowless Python; what it prints outside a run goes to a log file beside
  the settings file; the macOS and Linux launcher sets a new variable; a failure to start at all is
  shown in a plain Windows message.
- The fit's expected length is the "Expected epochs" typed on the Training page or, when blank, that
  of the last finished training of the same kind in Records, shown as a range one patience (20
  epochs) either side; with neither, only "Cannot end before" is said.
- The time axis is drawn for the stage that is running, with a line saying what follows, not for the
  whole run.
- The simulation's end time comes from the pace of the last 200 batches, and from the last run of
  the same size until 20 batches are timed.
- In the comparison with the audit the run's row reads "Waiting for the audit" with no audit and
  "Not compared" with a partial total or nothing counted, and neither asks for the owner.
- A difference of 3 to 5 points counts as passed only after "Accept this difference" on the check's
  page; the run's history keeps it with its date.
- "Failed simulations" reads "Fine" up to 1 % and "Look closer" above; the
  unusable-driven-recordings row says "Counted when simulation ends" until then; "Graphics-card
  memory" reads "Look closer" once a batch had to be split or failed whole; a continued run's failed
  count says since which batch, or over how many runs.
- A silent run becomes "Attention" and "needs you" after twice the epoch time in the fit, and
  elsewhere "Attention" after 45 seconds and "needs you" after 10 minutes; the next output clears it
  without an answer, and the taskbar button flashes for "needs you" as it does for an end.
- Messages that differ only in their numbers are one message with a count, and the page keeps the
  newest 50.
- "Recent" on the loss figure is greyed until epoch 12.
- Outside the fit the stop question's second answer is "Stop training" (rotation) or "Stop and keep
  the simulations" (simulation); the stop button is greyed while the posterior is being saved.
- Closing the window while a stop is pending offers "Wait" (preselected) and "Close when it has
  stopped".
- The sentences Windows shows as PRISM's reason during a shutdown are this part's wording.
- Appearance and the keep-awake switch stay changeable during a run; every other setting is locked.
- The run's page's sentences while live, after a failure and at the end are this part's wording,
  written to U7, U11 and U12; only the fit's "Now: …" line, the stage words and the loss caption
  come from the approved pictures.

### Part 7, plans, "This computer" and what is remembered (33)

- Filling from a plan sets every setting a plan can hold, to the plan's value or else the usual one,
  so values typed before do not survive a fill.
- Saving a plan writes only the settings that differ from their usual values, plus the bounds file,
  so a setting left out keeps following the built-in value.
- "Continue with these" applies the kind of observation that was in use last time, and every other
  option at its built-in value or the plan's.
- A plan is filled into the forms by the button "Fill the forms from this plan" beside the list;
  choosing in the list alone changes nothing.
- Plan files are tracked in git like bounds and cell files and are committed before a long run, so
  that its records are not stamped as made from a changed tree.
- The memory lines stand only in the "Before you start" of runs that simulate on the card, where the
  decision on the memory limit said every "Before you start".
- A plan may hold every setting on a step's form that is not excluded, including the learning rate,
  the patience, the epoch limit, how often simulations are saved, the rotation's three measurement
  settings, the draws per calibration dataset and an inference's posterior draws.
- A plan may hold typed units.
- Plans are kept in one folder per model, and a plan never names or changes the model.
- A plan file names each setting by the engine's name; a plan the window saves writes the on-screen
  label beside it as a remark.
- No plan file is shipped; the first one is saved from the window.
- A plan may not hold "Continue or start new"; it is treated like a consent.
- Of the Inference step a plan holds only the number of posterior draws, never the cell, the
  recordings, their lengths or the drive amplitude.
- A plan may hold the expected number of epochs, which feeds the time estimate only and never limits
  the fit.
- Before a fill replaces a value that was typed or kept from last time, the window asks, lists each
  one, and preselects "Keep what is there".
- "Continue with these" fills last time's plan without a further question and leaves in place the
  values kept from last time: the kind of observation, the bounds file and the two budgets.
- The memory limit and the keep-awake switch are stored under the computer's name, so a settings
  file carried to another machine applies no limit there.
- Free graphics-card memory is read only when its button is pressed in Settings: never at launch,
  when the page opens, or while a run is live.
- In "Before you start" the plan line and the memory line are marked "Information"; the only
  "Attention" is for a plan file changed since it was filled in.
- The keep-awake switch stays usable during a run, while the memory limit is locked.
- Two origin phrases are added to the decision's three, "changed from the usual" and "kept from last
  time", and the plan's two phrases are also said of a record or a run that filled the forms.
- A plan counts as used, for the "Last used" hint and for "Continue with these", when a step is
  applied or a run starts with it, not when it is filled in; a step applied or a run started with no
  plan clears the hint.
- The cell file picked for a simulated inference stays remembered.
- When a remembered or selected file is gone, or nothing is remembered, a file list shows no
  selection instead of falling to the folder's first file.
- A plan file may carry a free-text section, "About this plan", shown under the picker and never
  used as a record's note.
- The file's heading for the calibration sizes is "Calibration", not the step's name, "Checks".
- A plan's name is letters, digits, "_" and "-", at most 40 characters.
- A plan file is held to the window's floor of 2 driven recordings per observation, not the engine's
  1.
- The engine's plan module holds a plan's text form, which the window's save uses, as well as its
  reader.
- The memory-limit lines in Settings and "Before you start" name the environment variable that
  overrides the limit, when it is set: the one code name on screen in this part.
- The observation length of a simulated cell opens blank at every launch, with the two recordings'
  lengths.
- When the plan in use last time is gone or has problems, "Continue with these" leaves it out and
  goes on.
- "Save as a plan…" asks for the plan's name only; a description is written by hand in the file,
  under "About this plan".

### Part 8, the steps of Parameter inference (23)

- A pre-run check that has not been run shows as "Attention" in Training's "Before you start" and
  never greys "Start training"; one that failed does not grey it either, but the click then asks one
  question first, with "Cancel" preselected; one that does not apply shows as "Information".
- A narrowing round has no "Continue or start new" setting and asks nothing about starting from
  zero: it continues its own saved simulations when the same round is started again with the same
  settings, and never a training's; only when PRISM finds a saved simulation set one setting away
  from the round's does it ask, in a question that takes the place of today's tick box and carries
  the engine's own sentence, the engine's name for the setting included.
- Loading a saved posterior warns when the prior in use is not the one it was trained with, but
  still loads it; an inference on a simulated cell then skips, and says it skips, the check against
  what training saw.
- A saved prior built on a different bounds file gets an "Attention" line but can still be loaded;
  the line says when PRISM will refuse the load, and the button is not greyed for it.
- Home's "Continue with these" applies the kind of observation that was in use last time, and every
  other option at its built-in value or the plan's.
- The rule that a name taken by another record during a run costs nothing covers a prior build, a
  training, a narrowing round and the observation an inference makes, not the inference itself:
  there the name is refused, and what is lost is at most the simulation of one observation.
- Training has "Expected epochs (for the time estimate only)" under "Rarely changed"; it never
  limits the fit, and the narrowing round has no such field.
- With a posterior from a narrowing round in use, "Run inference" asks the other-observation
  question at every run, in both modes, in place of today's tick box.
- With a narrowed posterior in use, the Narrowing round page shows observations other than its own
  as unusable before the click, not refused after it.
- With "Save the simulations every … batches" at "Never", a saved simulation set with these very
  settings is listed in the from-zero question as one the training will not continue, and the
  question's first button reads "Start training without saving the simulations".
- An inference cannot start until a name is typed for its observation; the inference's own name may
  be left blank.
- Of these pages only Training and the Narrowing round state the memory limit in "Before you start";
  a prior build and an inference, which also compute on the graphics card, do not.
- Prior and Training each have two views (use a saved one, or make a new one), switched by the plain
  button at the right of the heading; Prior opens on the saved view when one exists.
- The time in "Build a new prior (about …)" is the last finished build's on this computer; with
  none, the bracket is left out and the page says the time is not known.
- The prior build's cost sentence leaves out the number of steps when it is not known.
- The typed-units box is refilled from the selected model's units file when the model changes.
- Limits typed by hand survive switching to the file and back, and are dropped only when another
  bounds file is selected or step 1 is applied.
- The boxes for the drive's own amplitude, frequency and phase open blank, not at 0.
- Inference gains "Posterior draws" under "Rarely changed"; its result table shows a "True value"
  column for a simulated cell, and its "Unit" column is blank where PRISM has no unit to show.
- With no posterior in use, "Your own recordings" shows a sentence in place of a form.
- The undriven recording is called "Spontaneous recording" on every form.
- Loading a saved prior or posterior shows "Loading…" at the foot of the rail, leaves nothing to
  acknowledge when it is done, and keeps no file in the run history.
- The on-screen labels where U7 gives none (for example "Kind of observation", "Slots for driven
  recordings", "Lock-in limit", "Region level", "Directions narrowed", "Continue or start new" and
  its three choices), the pages' sentences, the questions' titles, texts and answers other than
  "Cancel", the four readings beside a planned driven recording, the runs' and the chain's words,
  and which settings are main: on step 1 only the model and the kind of observation; on Training the
  budget, network width and depth and "Continue or start new".

### Part 9, Checks (25)

- For a posterior trained without the frequency response, three deciding checks read "Does not
  apply", the headline reads "n of 3 deciding checks passed; 3 do not apply to this posterior" where
  the owner decided "always n of 6", and the posterior has passed when those three have; with no
  posterior in use the headline keeps "of 6".
- A pre-run deciding check that failed does not grey "Start training", but the start then asks one
  question first with "Cancel" preselected; a check not yet run is led by "Attention" and one that
  does not apply by "Information", and neither blocks.
- After simulating, a difference of more than 3 and up to 5 points counts toward the headline only
  once the owner presses "Accept this difference", kept with its date in that training's history;
  the check then reads "Passed" and says the difference was accepted.
- Every deciding check's in-between state reads "Look closer", and a check that can be run again
  counts only once a later run passes.
- A check made at less than its usual size (a calibration on 40 test datasets, an audit of fewer
  than 100 batches) never decides: the row stays "Not run yet" and says so.
- The input-use check passes only when the two parts the runbook names read healthy and no other
  part is invisible or non-finite; with another part invisible it reads "Look closer".
- When simulation ends and that prior has no audit, the check reads "Waiting for the audit" and the
  run's page says so without asking for the owner.
- A total of unusable driven recordings that covers only part of the simulation set, or none of it,
  reads "Not compared": it never counts toward the headline and cannot be accepted.
- An accepted difference lapses when a newer audit comes to count, and cannot be given for a
  training that was not started from the window.
- When a check has been run several times the newest run decides, and its row says when an earlier
  run came out differently.
- When the frequency band has been checked on several cells, the worst result among them decides.
- For a posterior from a narrowing round the comparison with the audit is made the same way, with a
  sentence that the audit measured the whole prior.
- The timing check projects the full fit from Training's "Expected epochs" and shows no stop point
  and no pass.
- The calibration form has a "Seed" field under "Rarely changed": blank means one is drawn and
  recorded, and it is never remembered.
- The audit's form opens at 100 batches (the runbook's size, not the engine's 12), and the
  thresholds that define "holds" in the frequency-band check and the drive study are not offered.
- A check's form states how long that check took last time on this computer and shows no built-in
  times from the owner's card.
- The two checks that only read records show no memory lines, where the owner decided that every
  "Before you start" shows them.
- The timing check and the card test run in a scratch folder that is removed afterwards; only their
  result is kept, in the run history.
- The recording length of the two identifiability checks that simulate opens blank and must be
  typed.
- Informing checks and tools read "Done" once they have run, never "Passed" or "Failed".
- The third identifiability method's error for a posterior with no training prior becomes a refusal
  that points at the posterior.
- The names of seven checks, the longer forms of two names from the mock-up, all twelve one-sentence
  questions, the four group headings and the three role words are as written in 9.1.
- The wording of the sentence under the headline, of the states that wait or do not apply, of each
  check's verdict sentence, greyed reason and form labels is as written in 9.3 and 9.4.
- A check's stop button names the check as its start button did; the answers are "Keep it running"
  (preselected) and "Stop and discard it".
- The Checks code lives in small files of its own inside the window's package.

### Part 10, Records, the other parts, Settings and Help (36)

- A next-step button on a record's page also loads whatever its step needs and is not loaded yet:
  the record's model and options (its own, not the built-in ones), its bounds file and the records
  it was made from.
- The words for the kinds of record that U7 does not give ("inference result", "check",
  "measurement", "sweep", "comparison") also go into what the command line prints, when the engine's
  refusal sentences are reworded.
- Deleting a record that Parameter inference holds asks with one more sentence, then makes it let
  go; today the session goes on holding the deleted record.
- One "Clear leftovers…" button for all kinds replaces the two sweep buttons.
- A Live simulation and the saving of a video leave no run-history file and no full-text file, and
  neither does a run that only loads a record, as each load behind a next-step button is; this
  narrows U17's "one small file per run".
- A simulation set made by a narrowing round offers neither "Train from these simulations" nor "Set
  up a run like this one", because a training never continues it; its delete question speaks of the
  round, not of a training.
- A measurement's likely length is taken from the last finished one with the same settings, as the
  time from its making to the last writing of its log.
- The measurement page's two tick boxes become one choice of three, "What to run", which opens on
  "The checks, then the measurement" at every launch.
- Comparing is one page: one choice of four kinds and a tick on each saved measurement, in place of
  an add-and-remove list on two tabs.
- In Live simulation time is counted from the start of the recording; the settling-in stretch is
  drawn before zero and labelled, on screen and in the saved video.
- Live simulation gains a "Seed" field; blank means a new stream each time, and it is never
  remembered.
- The saved video is drawn with the light figure settings like any stored figure, in both themes: a
  white page, and its trace in the page's ink where it is blue today.
- When an effective-temperature run leaves a record, finished or not, Name, Note and Seed are
  emptied and the result states the seed used; after a refusal that leaves no record they stay.
- On Sweep study a typed number survives a change of resolution or cell file; only numbers still
  following them are replaced, and a line says which.
- A measurement that ran only its checks has its own state, "Checks only", and cannot be compared.
- Saving over an existing model asks first and says that records made with the earlier equations
  cannot be told from new ones.
- Closing the window with an unsaved model asks "Throw away the unsaved changes?" after any question
  about a live run, and "Keep editing" keeps the window open.
- The list of your own models moves from Settings to "Your own models".
- The three Effective temperature pages are chosen from a row of names above the page, which opens
  on "One cell" every time.
- A record with no name is called by its kind and date, and a simulation set with no name by its
  prior.
- In a narrow window a record's page takes the place of the list, with "All records" to return.
- A record's page shows its folder with "Copy path" and has no button that opens the folder.
- A record's page says whether each input file is unchanged, changed or gone, by reading the file
  again when the page opens.
- "Save report…" writes the command line's text, so lists over 12 entries are still shortened in the
  saved file.
- The next-step buttons beyond the three the owner named: "Use this prior", "Train from these
  simulations", "Run it on its observation", "Narrow a posterior around it", "Narrow around its
  observation", "Compare it with others", "Measure again with these settings", "Compare these
  again".
- "Narrow a posterior around it" is greyed until a posterior is in use, because an observation's
  record names none.
- Live simulation's "Recording length" stays remembered: B12's "recording lengths" is read as the
  Inference step's alone.
- Live simulation's stop asks nothing, because nothing is lost; stopping the saving of a video does
  ask.
- The plain-word labels of the settings on these pages, for example "Drive frequencies",
  "Trajectories per frequency", "Right-hand side".
- The sentences of this part's questions (deleting each kind of record, clearing leftovers, loading
  from a record, stopping a measurement or the saving of a video, a model's three questions) and of
  the glossary.
- The Reduction map lists earlier results as its own figures in its folder and leaves the folder's
  other files unlisted.
- A folder of the kind the live run writes, changed since that run began, reads "Being written" even
  when a command-line run is writing it.
- An unreadable folder gets a sentence of the window's, with the store's reason beneath as console
  text; an unfinished measurement's last log line, which names the error's type, is console text
  behind "Show the full error".
- A setting with no on-screen name is listed on a record's page under its stored name.
- The words of the Reduction map's stop question.
- Appearance, like the keep-awake switch, can be changed while a run is live; every other setting is
  locked then.

### Part 11, the engine changes (18)

- If the loss figure cannot be drawn or saved at the end of a training, the training no longer fails
  but saves the posterior without the figure, with a warning, the loss curves staying in the
  record's data file; this holds from the command line as well as from the window, so it changes
  what a command-line run does at its last write, beside the two changes to its printed text the
  owner was told of (ruled by the controller).
- When a name is taken while a prior, a posterior or an observation is being made from the window,
  the record is saved without a name under its id, while a run from the command line still fails
  there as today (ruled by the controller) and every other kind of record still refuses before
  anything is simulated, because it opens its record first.
- One case is left as today (ruled by the controller): when an inference is made together with its
  observation and the inference's name is taken while the observation is being made, the inference
  is refused and the observation is kept.
- The changes that count as touching training, and so owe the test on the graphics card, are the
  per-epoch line, the card test's own move and the two changes to a training's last write (ruled by
  the controller; carried by the part on tests and stages).
- The rewording of refusal sentences replaces only words the approved vocabulary names, so the
  engine's names for settings, engine words the vocabulary does not list (such as "manifest" and
  "HPD level"), log lines, warnings, plain errors and the command line's own usage messages keep
  their text; by the controller's ruling, reasons that are not refusals keep theirs too, and the
  reasons shown beside a greyed start button for an effective-temperature measurement are reworded
  with the refusals.
- The engine's plotting code holds the four line colours itself (blue, vermilion, green, orange),
  kept equal to the look's light colours by a test, with the ink taken from the figure's settings,
  so figures saved from the command line change colour too: today's blues become blue, reds
  vermilion, greens green, oranges orange, and the one grey marker takes the axis colour (ruled by
  the controller; no line style is edited, and the exported video is in the look part's list).
- The third identifiability method's error for a posterior that carries no training prior becomes a
  refusal that points at the posterior (ruled by the controller as an extension).
- Both refusals about continuing a simulation set point at the continue setting, also in the case
  where the setting for how often simulations are saved would answer it as well, and the six check
  errors point at the posterior.
- A stop requested during an epoch takes effect at the new per-epoch line, a few statements earlier
  than today, at the same epoch end.
- If a later version of the fitting library has no per-epoch step to attach to, the training runs
  with one warning and no per-epoch lines, instead of failing.
- The pre-run preview lists every other saved simulation set with its batch count, whether it is
  complete, and every differing setting with both values uncut, while the engine's own refusal still
  fires only for a set exactly one setting away (how a narrowing round shows that refusal is in the
  steps part's list).
- On the command line's list of records, a simulation set renamed in the window shows its name where
  it showed "unnamed".
- The card test becomes one engine function that reports each stage through a callback while the
  command line keeps printing its own lines, and, being one run, each record in the test's own
  scratch store carries the text since the test began.
- The card test's engine function repeats the cell check and the two refusals about continuing, in
  neutral words, for callers other than the command line.
- Only the observation's name is added to an inference (its note stays empty and is edited from
  Records), the name has its own entry in the engine's list of settings, and the command line gets
  no flag for it.
- The engine holds a plan's text form as well as its reader.
- The engine's list of settings gains an entry for the expected number of epochs, which only the
  window uses (ruled by the controller; how a plan names its settings is in the plans part's list).
- The wording of the new lines the engine writes (one per epoch, the warning when a name was taken,
  the warning when the loss figure failed), shown on the run's page as they stand wherever the
  window has no sentence of its own (ruled by the controller; carried by the run part), and that on
  the command line the per-epoch line follows the fitting library's own counter on the same console
  line.

### Part 12, what becomes of each finding (8)

- An engine fault outside E1 to E12 that U18's audit list does not name is handed to the audit, not
  designed into this redesign and not left as it is: items 6 to 16 of 12.7, and W159 with W091's
  remainder beside item 4. The one that can lose days is W011 (12.5).
- A row that closes only through a rule the owner has not seen counts as fixed in passing, and
  becomes left as it is if the owner refuses the rule (25 rows). A fault its part simply mends,
  listing no rule for it, is fixed in passing with no dagger.
- A finding with two halves is counted under the half a decision covers; the other half is listed as
  a remainder (24 rows).
- Five findings the second readers could not settle by reading count as settled by the experiment
  and as fixed, not dropped (W164, W173, W178, W192, W197).
- The audit's list carries one lead of the first note that has no finding of its own and is not on
  U18's list: the empty list of consents after a narrowing round (item 17).
- That a fit ended by the epoch limit keeps its last network is left as the fitting library's rule
  and kept off the audit's list (W036), as are the remainders of W010 and W154.
- The pause when a model is saved is left as it is: never measured, and no part takes it up (W122).
- A finding that is only a wrong comment counts as fixed in passing when the build removes or
  rewrites the lines that hold it (W119, W128, W153, half of W105). The rules behind the daggers,
  and those 12.2 names as leaned on, are the carrying parts' or the controller's, not this part's.

### Part 13, tests, the stages of the build, the pages and the walkthrough (30)

- A posterior trained without the frequency response is judged on three deciding checks: the other
  three read "Does not apply", its headline reads "n of 3 deciding checks passed; 3 do not apply to
  this posterior", and the tests and one walkthrough row hold it so.
- A training started from the command line whose last figure fails to draw now ends with its record
  and a warning instead of failing, which the guide's command-line page will say beside the new line
  per epoch and the reworded refusals; the fallback for a name taken during a run is used by the
  window alone, so the command line is unchanged by it.
- Which stage each engine change is built in: the figure colours in stage 1; the loss line per
  epoch, the fallback for a name taken during a run, the guard round the loss figure, a simulation
  set keeping its name, the engine half of the card test and the two training refusals in stage 3;
  the list of every differing setting, the plan reader and the plans-folder line in stage 4; the
  observation's name and the checks' refusals in stage 5.
- Two engine changes that owe no test on the graphics card are still built before the one such test,
  so that it runs through them: the figure colours, and a simulation set keeping its name.
- The window tries again by itself only after the graphics card runs out of memory in a training or
  a narrowing round; the tests and the walkthrough's failure rows expect every other failure to be
  reported at once.
- "Continue with these" applies the kind of observation that was in use last time, and Home's
  sentence names it; a test and a walkthrough row expect it.
- A plan fills the forms only when "Fill the forms from this plan" is pressed, never by choosing it
  in the list; and plan files are kept in git like bounds and cell files and committed before a long
  run, which the proposed wording for the project's instructions says.
- A pre-run deciding check that has not been run never blocks "Start training"; one that failed does
  not grey the button either, but the start then asks first, with "Cancel" preselected.
- A run that only loads a record, a Live simulation and a video export leave no history file, no
  text file and nothing to acknowledge.
- An engine warning the window has no sentence for is shown on the run's page after the words "From
  the engine:", in the engine's own words, and the scan of the window's vocabulary allows for it.
- The smallest window is 900 by 600 at Windows' ordinary text size and grows with the text size,
  held to the screen; the tests and two walkthrough rows expect it.
- The engine keeps its four line colours in one place, held equal to the look's by a test, and
  changes no line style; and the exported video is on a light page like any stored figure.
- The first-time opening's page is built with "Home" in stage 2, though the list of stages names it
  with Help in stage 8; stage 8 adds its longer explanations in Help.
- Where the later decisions are built: one PRISM at a time in stage 2; the reason against shutting
  down in stage 3; "Continue with these" in stage 4; reading saved effective-temperature runs during
  a run in stage 7.
- Keeping Windows awake and the launcher with no console are built in stage 3; until the keep-awake
  switch arrives in stage 4 the request is always made.
- A list is kept of all 322 of today's window tests, each with what becomes of it; a stage fails if
  a test has gone without an entry.
- The slow tests run only at the ends of stages 1, 3, 5 and 9.
- A test scans the window's own words for the retired names of the vocabulary table.
- Tests that call the real taskbar alert, keep-awake request, shut-down reason and bring-to-front
  call run only in the set that needs a real screen, so the ordinary gate never changes the
  machine's power or session state.
- One walkthrough row signs the owner out of Windows for real during a scratch run, to see whether
  the simulations done so far are saved.
- The two failure rows are staged by denying writes to the scratch store's folder of simulation
  sets.
- The walkthrough's "small training" is 100 batches at the usual simulations per batch with a 256 by
  10 network, about 1 % of a full training, run with the memory limit at 6 GiB.
- The eight walkthrough rows for the pages rebuilt after the run use tiny sizes and come last, the
  narrowing round's among them, and one of them trains a tiny posterior without the frequency
  response only so that its headline is seen on a real screen.
- The owner may run a walkthrough row early, once everything it names is built.
- The owner is shown off-screen pictures of every screen at the ends of stages 1 and 2.
- From stage 2 until its rewrite, one line under the stamp of the guide's window page says it
  describes the earlier window.
- Two headings of the guide's window page are renamed and the other eight kept, so only eight links
  need editing.
- The new guide page is named "The look".
- The replacement wording for the project's read-me and for its instructions is a proposal; nothing
  of it is written before the owner approves it.
- The fourth stage builds the pages of the first three steps (Model and options, Prior, Training)
  with plans, "This computer" and applied against pending, and the fifth builds Checks, Inference
  and the Narrowing round, whose forms a plan fills from then on; the list of stages put every step
  but Training in the fifth.

---

## 8. Deviations recorded while planning and building

Where what is planned or built departs from this design, the departure is recorded here with what it costs if it is wrong. A row wins over the part it departs from. Rows 1 to 32 were recorded on 2026-10-06, when stage 1 was planned (`docs/superpowers/plans/2026-10-06-window-redesign-stage-1.md`, whose head holds the same rows under the same numbers); rows marked "owner" are the owner's own answers, and the others were put to the owner with that plan. Rows added while a stage is built follow, each with its stage and date. Rows 54 to 78 were recorded on 2026-10-14, when stage 2 was planned (`docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md`).

**Stage 1, the look's foundation (`8b2ca92`..`f9c6899`, 2026-10-06 to 2026-10-14).**

| # | what | why | what it costs if it is wrong | who |
|---|---|---|---|---|
| 1 | A figure's line colours number seven, not five: sky blue #56B4E9 and purple #CC79A7 after orange (on a dark page the two blues change places). The look's table has 32 names. | The owner, 2026-10-06, from a picture of the thirteen-curve calibration figure. | Two more named colours and two rows of the contrast table; sky blue on white is 2.31 to 1, a named exception like orange's 2.25. One edit to the table and one test to go back. | owner; decided by the owner on 2026-10-06 |
| 2 | The order inside the stage: the two appearance ticks and "Auto (time of day)" go before lettering and the sheet; the text-size experiment may run at any moment before the look's table. | Lettering and the sheet remove the arguments those ticks pass; at 13.2.1's order the window is broken between commits. | None on screen: everything of 13.2.1 is delivered. If the order hides a dependency the gate of the affected task shows it. | controller; approved with the plan on 2026-10-07 |
| 3 | `widgets/plot.py` (a figure the window draws itself) is not built in stage 1. | Ruling D7 lets a stage-3 test decide whether a figure library may run on the window's thread at all. | Stage 3 builds it, or a painted widget in its place; nothing in stage 1 uses it. | controller; approved with the plan on 2026-10-07 |
| 4 | Whole numbers are grouped with commas page by page: `IntField(grouped=False)` by default in stage 1. | The owner, 2026-10-06: today's forms keep plain digits until each is rebuilt (part 5.10). | Each rebuilt page must ask for grouping; a page that forgets shows plain digits, which its own tests catch. | owner; decided by the owner on 2026-10-06 |
| 5 | The figure settings' face is a list, Inter first, then "DejaVu Sans". | With Inter alone a character Inter lacks is a missing glyph with a warning. | A label with such a character is drawn in a second face; remove the second name to go back. | controller; approved with the plan on 2026-10-07 |
| 6 | The wheel over a closed drop-down no longer changes its choice. Not in the design. | A science setting can otherwise change silently while a form is scrolled. | A user who chose by the wheel must open the list; one answer in the small style to go back. | controller, for the owner to strike; not struck: approved with the plan on 2026-10-07, and told again at the exit, 2026-10-14 |
| 7 | `PaintedIcon`'s clone keeps its new engine referenced. | The sample's form ends the process when a copy of such an icon is changed (measured). | None seen: the list of kept engines grows by one small object per copied icon. | controller; approved with the plan on 2026-10-07 |
| 8 | The step row, the settings row, the facts list, the fold and the redrawn "?" are not built in stage 1. | Each belongs to the stage that builds its page (parts 3 and 5; 13.2.1's own table). | None: no stage-1 test or screen needs them. | controller; approved with the plan on 2026-10-07 |
| 9 | The style-sheet test admits the interim rules for today's widgets as named exceptions. | Part 4.10 forbids what part 4.1 requires for the old layout. | A stray rule could hide among the exceptions; the test names each, and stage 9's test fails a rule whose widget is gone. | controller; approved with the plan on 2026-10-07 |
| 10 | The record a test reads of a box also holds its title, its marker and its safe answer. | Today's tests pin all three; 13.1.4's list has none of them. | None: more is recorded, nothing less. | controller; approved with the plan on 2026-10-07 |
| 11 | The slow set runs at this exit as the design has it, though its stated reason is wrong (the fast gate already draws through 42 of E12's 50 sites and the new record through all 50). | Cheap insurance for the first engine change. | 75 minutes of the card. | controller; approved with the plan on 2026-10-07 |
| 12 | A marker drawn disabled keeps its inner mark in the page's colour; only its shape greys. | Greyed whole, as part 4.5's letter has it, four of the ten markers lose their inner mark and read as discs. | A disabled marker is a little more present than the design's; one argument to go back. | controller; approved with the plan on 2026-10-07 |
| 13 | The focus ring: the follower listens from the application's first change of focus; a Tab arrival shows the ring only when a key was pressed last; every widget that scrolls takes the ring just inside its edge. | An application-wide event filter written in Python is paid for by every event (33,000 while the window is built), so it starts at the first change of focus; the key rule stops a ring appearing after a MOUSE click on a control that greys or hides itself (the toolkit then moves the focus on with the Tab key's reason); a ring outside a scrolling area lands on its neighbours. | The ring's behaviour under a real keyboard is the owner's to judge on a real screen; each is a few lines to change. | controller; approved with the plan on 2026-10-07 |
| 14 | The pictures for the owner are one set, made on the real platform with the window hidden, with one more at a forced text factor of 1.5 and one at 150 % scaling; and they are taken at three moments (the plan's commit, after Task 10, the exit). | Off-screen there are no system fonts, so a second, off-screen set would show the wrong faces; early pictures let the owner correct the look before six tasks build on it. | Two more runs that build the real window and read the card's free memory. | controller; approved with the plan on 2026-10-07 |
| 15 | Console text is Cascadia Mono, then Consolas, then Courier New; in stage 1 it reaches a record's text only, with no line height set. | The owner, 2026-10-06, from a picture; the log pane is rebuilt in stage 3. | On a computer without Cascadia Mono the text is Consolas; the order is one tuple. | owner; decided by the owner on 2026-10-06 |
| 16 | Both toolkit validators leave the number boxes on all of today's forms. In a decimal box a typed comma stays, is outlined and refused in words (in the box's tool tip there, and by the click's own refusal); a whole-number box does not take the comma key or a "+"; a box built with no value is empty, not the word "None"; a whole number stored with commas by an older build is read as that number and saved back without them. "Right-aligned, 120 wide" is set by each rebuilt page. | Measured: today a typed 6,5 silently becomes 65, in every regional setting; the validators cannot be kept without that. | Today's forms behave differently for a mistyped number: a refusal where there was a silent wrong value. On today's forms the sentence is in a tool tip until the page is rebuilt, which part 4.5 does not allow for a rebuilt page. | controller, for the owner to strike; not struck: approved with the plan on 2026-10-07, and told again at the exit, 2026-10-14 |
| 17 | The size tables hold the numbers of the ordinary text size and each reader calls `scaled()`; the "now" mark's height is scaled; a figure's tick length, the ring's inset on a row and the scroll bar's gap and handle are not. | Scaling "before the first widget" would make every test depend on the order of imports; the four keys are not named by part 4.8 either way. | A reader that forgets `scaled()` shows at the forced-factor pictures; no test enforces it. | controller; approved with the plan on 2026-10-07 |
| 18 | The text factor is Windows' own documented value, read once at start, also where the toolkit's default font follows the setting. | It is right whichever way the toolkit behaves, because PRISM sets every font itself; the experiment still decides whether a change is heard while PRISM is open. | A change of Windows' text size shows at PRISM's next start, not while it is open. | controller; the owner if the two numbers differ; approved with the plan on 2026-10-07 |
| 19 | With no book face on a computer a sentence is Inter at the sentence size (16 on 22), not at the body size. | Off-screen no book face exists, so the suites would otherwise measure every sentence device at the fall-back's size and never at the designed one. | On such a computer sentences are larger than body text, as designed for the book face. | controller; approved with the plan on 2026-10-07 |
| 20 | The time axis: a late axis grows by as many whole days as it needs; the 30 s floor holds in every form; an end with no earliest end reads as firm. | Part 4.6 leaves these cases open or gives one day only. | Seen first in stage 3, on the run's page; each is a line of the layout function. | controller; approved with the plan on 2026-10-07; the whole days of its late growth changed by the owner's answer at the exit, 2026-10-14 (built in task 19q, with its own row) |
| 21 | No row for the owner to run on a real screen is written in stage 1. | The first rows' words name pages later stages build; a row that cannot be run as written misleads. | Two rows (the appearance choices; the text size) are written with their group in stage 3 instead. | controller, for the owner to strike; not struck: approved with the plan on 2026-10-07, and told again at the exit, 2026-10-14 |
| 22 | The four places of the book face are held as modules, one shared sentence class serving three of them. | One class sets the line height once; four copies would drift. | The scan names modules, so a fifth caller in an allowed module would pass; the review reads for it. | controller; approved with the plan on 2026-10-07 |
| 23 | Nineteen implementer tasks and four controller tasks, where the design estimated eleven; what each plotting function draws is recorded as a task of its own before the figure settings change. | The tests' move, the painted devices and two cuts for the sake of a readable gate. | About six hours of fast gates in all. | controller; approved with the plan on 2026-10-07 |
| 24 | "No accent pixel on a page with no selection, current place or main action" is held for a table and for a page of the shared pieces; "current place" waits for the rail. | No device for the current place exists before stage 2. | Stage 2's shell suite carries the rest. | controller; approved with the plan on 2026-10-07 |
| 25 | No rule of the sheet gives an ordinary button inside a question the raised ground. | The question is mounted on no screen in stage 1; the sheet's counts are pinned by then. | In dark such a button would stand one shade darker than its dialog; the first stage that mounts a question adds the rule. | controller; approved with the plan on 2026-10-07 |
| 26 | The six modules that import the icon module and call nothing in it keep the import until the icon font goes. | Stage 1 edits no screen for it. | Six dead import lines for a few stages. | controller; approved with the plan on 2026-10-07 |
| 27 | E12 renames one keyword (`c=` to `color=`) at one site. | A colour read from the figure settings and passed as `c=` with three points is mapped through a colour map (measured). | None: the marker is drawn as before, in the axis colour. | controller; approved with the plan on 2026-10-07 |
| 28 | The session guard on file choosers stays on the two static calls for good. | The one chooser function is built from those two calls, and the stock toolbar of a popped-out figure opens a chooser of its own that no scan of PRISM's code sees (13.1.4 has the guard on the statics only "until then"). | None: the guard records a chooser however it is reached. | controller; approved with the plan on 2026-10-07 |
| 29 | Beyond 13.1.4's three builders: a fourth for a bare run host, and two standing scans over the tests (no test reads a raw message box; no suite builds a window, a part or a step by hand). | Eighteen tests build a bare run host; without the scans the move cannot be seen to fail first and nothing stops a later test going back. | Each stage that adds a window, part or step class adds its name to the second scan's list. | controller; approved with the plan on 2026-10-07 |
| 30 | The one test that builds the whole program puts back everything that building sets on the process. | Part 4.1 says none of the application-level look runs under the suites; one test does build it, and today it leaves the style, the font, the figure settings and the error hook for the 200 tests after it. | From the task that does it those tests run without the leftovers; one that had leaned on them shows in that task's gate. | controller; approved with the plan on 2026-10-07 |
| 31 | The two layout thresholds (the small layout; two columns stacked) are scaled by their readers and held to the screen by the shell in stage 2. | The first has no reader before the shell exists. | At a large text size on a small screen a page of the shared pieces could keep two columns too long; nothing mounts one in stage 1. | controller; approved with the plan on 2026-10-07 |
| 32 | In the shared question an answer's and a label's words are handed in plain; the class doubles an ampersand for the toolkit. | The toolkit reads a single ampersand in a button's words as a keyboard mark and swallows it. | None seen: a caller who wants a keyboard mark on an answer cannot have one. | controller; approved with the plan on 2026-10-07 |
| 33 | CLOSED 2026-10-07. The look's table and its text factor were built before the text-size test was run; the test was then run with the owner the same day: Windows reports the number set (1.5 at 150 %) and enlarges its own text by the same amount, and the toolkit tells a running program, so what was built is what the test asks for (a change still shows at PRISM's next start: row 18). | The owner had not answered at three gaps between tasks, and every later task of the stage stands behind this one. | Nothing: no line is redone. | controller, in the owner's absence; closed by the owner's test |
| 34 | The rules of the kept sheet that give a field its text colour (three: the two plain rules and the disabled one) gain a fifth change against the sample: the hint text's colour is named in the sheet. | Whenever a sheet rule gives a field a colour, the toolkit paints its hint as that colour at half strength and never reads the palette's hint colour, so the hint was 3.36 to 1 in the light theme where the design asks the hint colour of the table (6.01) and a floor of 4.5. | None seen: one declaration in three rules. | controller (Task 9's review, reader two; the skeptic could not refute it); told to the owner at the exit, 2026-10-14 |
| 35 | In one of the old message boxes the focus ring is shown on the button that has the focus whatever brought the focus there (ruled for Task 10; it leaves with the last old box). | The task that builds the ring removes the 1 px focus border, which is all that marks the answer Enter gives in an old box; the ring otherwise shows only after Tab, a shortcut or a click in a typing field, so a freshly opened box would mark no answer; the design forbids a sheet rule on the default state, which follows the focus. | A ring on a message box's button also after a mouse click outside it; one branch of the follower and its test go when the last old box is replaced by the shared question. | controller (Task 9's review, reader three, and its skeptic); told to the owner at the exit, 2026-10-14 |
| 36 | The main window's class handles its events through a method written in Python that only calls the base class, so that a repaint keeps the interpreter for its whole length; and the window's own drawing code reports its errors itself (a guard on every method the toolkit calls while the window is drawn or laid out, a standing scan that lets no new one in without it, a session fixture that fails a test in which one raised). The small style and the focus follower stay on the application as the design has them. (First ruled otherwise, on 2026-10-07: the style and the follower taken off the application; a feasibility probe found three ways that arrangement could end the process, and it was not built. A lighter reduction was probed and not built either: it halved the waiting and cost five guarantees.) | Measured in the real window: with the style and the follower in Python on the application a repaint of a page entered Python 28 to 211 times where it entered it never, and while the run's thread executed plain Python (it draws its figures there) each group of entries waited for the interpreter: 0.2 to 1.5 s a repaint, where 1 to 4 ms stood. With the method: 16 to 20 ms on every page; beside the real figure work the share of repaints over a tenth of a second falls from 2 % to 0.1 %; the run loses nothing measurable. The design chose the two application-wide mechanisms (parts 4.1 and 4.6) and listed their cost in the real window as not measured (part 4.11). | With the method the binding drops an error raised by Python code the toolkit calls from inside the window's handling of an event when another such method is entered after it; the guard keeps the rule that an error is shown for the code this program writes, but an error inside a drawing library's own code in the main window (the live simulation's plot) shows no box, and the figure pop-out is left without the method for that reason. Not cured, only about halved: a change of page or tab, a resize, the pointer and the keyboard while a figure is drawn. | controller (Task 10's review; two measurements in the real window, three probes; the reviews of Task 10f); told to the owner at the exit, 2026-10-14 |
| 37 | Digits of equal width are switched on only where numbers stand under one another or change in place (lists, tables and their headers; the progress read-outs; the run slot; a device that asks), not on the application's font. | In the bundled Inter the feature replaces 61 glyphs, not only the ten digits: the hyphen and the asterisk become as wide as a digit, so every compound word of the window read "cross - validation"; the feature cannot be had for the digits alone. | Numbers in running text and in fields are of unequal width (a 1 narrower than a 0); a number that changes in place outside those devices may shift by a pixel until its device asks for the feature (the later tasks that paint number columns, tick labels and number fields ask for it). In a list a hyphen stays a digit wide. | controller (the first pictures of the new look, read by four readers); told to the owner at the exit, 2026-10-14 |
| 38 | Four things the first pictures showed are named in the sheet or the code beyond what the design's table of rules holds: a list widget and a text viewer take the field's outline and ground, with a rule for their greyed state; a scrolling area with the default frame takes the faint rule in both themes; the corner between two scroll bars takes the track's ground; the help badge's ring takes the faint rule again and its corner follows its scaled size; and the small style answers the focus frame's two margins for the ring alone. | The toolkit frames what no rule names by darkening the page's colour: a pale line in light and none at all in dark, so the two themes differed in more than colour; the badge's mark lay unreadable on a ring of nearly its own tone; lists and tables take their cell margin from the metric the small style answered for every widget, which moved every cell by 2 px. | None seen: interim rules for the old layout, which leave with it; the badge's mark still stands left of centre (the icon font's own fault, left for the stage that owns the icons). | controller (the same pictures; the review of Task 10g); told to the owner at the exit, 2026-10-14 |
| 39 | The colours of the old layout's tabs were taken over from the old sheet as they were; they now differ from it in two places. The word of a tab that can be opened is in the full text colour, where it was in the secondary text colour; a locked tab keeps the readable grey. And a tab that can be opened and is not in front takes, under the pointer, the ground an ordinary button takes there, where its word used to take the full text colour. A locked tab takes no ground under the pointer (its tool tip still shows). The tab one is on takes none either and keeps its blue line: its rule names its own ground again, one more declaration in a rule that stands, which changes nothing to see against the old sheet. | The owner, 2026-10-10, from the pictures: a locked tab could hardly be told from an open one (their words stood 1.41 to 1 apart in light and 1.38 in dark, where 2.37 and 3.02 stood before the stage; now 2.90 and 2.35). With the open word already in the full text colour the old pointer rule changed nothing, so an open tab would no longer have answered the pointer at all: the controller ruled the ground beside the owner's change. Measured on a stock tab bar: the toolkit gives the sheet's rule for the pointer to no greyed tab (the bar still knows the pointer is over it, and shows its tool tip), so the plain rule is enough to keep a locked tab's look. The controller, 2026-10-11, spared the tab one is on: clicked, it leads nowhere, so it should not answer the pointer; the lone tab of Reduction Map and of Simulate would otherwise light up under the pointer; and the owner was told that the page as it first opens does not change, which without the sparing held only while the pointer was elsewhere. | The tab one is on is told from another open tab by its line alone (the owner was told). On Parameter Inference as it first opens, with one tab in front and the five others locked, nothing changes: the page is the same picture as before, to the pixel, with the pointer elsewhere and with the pointer on any of its tabs (the owner was told, with the remedy drawn in the real window; measured on a stock bar of one tab in front and five locked, and on a lone tab), and the open words show only once a second tab can be opened. Every tab bar takes the changes until it leaves: the bars of Parameter Inference and of the section pages at the exit of stage 2, the figure tabs by stage 7. The tabs of the "Figure options" box that a popped-out figure's toolbar opens take them too (read in the plotting library's source, not drawn); that box is the library's own and is in no stage's list of bars that leave. The sparing holds only while the rule of the tab one is on stands after the pointer's rule (measured: with the two the other way round the tab takes the ground), and no test reads that rule's text: one painted line of the pointer's test holds it, and fails if the declaration goes or the rule is moved. To go back if the words are wrong: one colour name, its sentences in the comment above the sheet's template, the tab test and one line of the pointer's test (the three helpers born with the tab tests stay, because the greyed main button's test uses them, so the commit of the tabs cannot simply be reverted by itself); if the ground is: two declarations (the pointer's ground, and the sparing, which would then spare nothing) and the pointer's test; if the sparing alone is: one declaration, one expected value of the pointer's test, and the two clauses that say so (the last sentence of the tabs' passage in the comment above the sheet's template; "its ground and" in the pointer test's docstring). | owner (the open words in the full text colour); controller (the pointer's ground on an open tab, and the tab one is on spared it; the owner was told on 2026-10-11); the owner's answer of 2026-10-11, told built at the exit, 2026-10-14 |
| 40 | One word on a screen changes in a stage defined as changing no word: the model builder's empty variables box shows "e.g. x, y" where it showed "x, y". The colour of hints is not changed anywhere. | The owner, 2026-10-11, from the pictures. The design makes the hint in an empty box dark enough to read (6.01 to 1 in light, 6.71 in dark), and this one hint is itself a valid entry, so it read as two variables already typed. It is the only one of the window's fourteen hints that can be an entry; the other examples on the forms already begin with "e.g.". | One string and its one test to go back. It makes no hint paler: an owner who finds every hint too dark is not answered by it (a hint colour of its own was offered beside it, and not chosen). | owner; the owner's answer of 2026-10-11, told built at the exit, 2026-10-14 |
| 41 | One rule of the old layout names a weight in a state, against the sheet's rule that no state names a weight or a size: greyed, the main button of today's forms has its words in the ordinary weight, like every other greyed button, and is bold again when it can be pressed. The rules that stay are not touched: on a rebuilt page the greyed main button is as approved, bold, with its reason beside it. The test of the sheet's rule lets this one rule name this one weight and nothing else, and fails when the old button is gone. | The owner, 2026-10-11, from the pictures: greyed and alone on its page, in bold, the main button looked as if it could be pressed, and today's forms do not have the sentence the design puts beside it. The sheet's rule is there because a control is measured once, at the weight of its class rule, and wider words would be cut; here the class rule is the bold one, so the lighter words are narrower than what was measured and cannot be cut (measured: a button first measured usable and one first measured greyed ask for the same size, 154 by 34 for "Build / Load prior", and the greyed main button's words are, to the pixel, those of an ordinary greyed button). | It takes the confident look away and does not make "off" unmistakable when nothing live stands near (the owner was told); what does that is the reason in words, which comes with each rebuilt page. Greyed, a main button is then told from an ordinary one by its place and its width alone. It is the first exception of its kind, so a later rule could point to it: the test names the one rule and the one weight for that reason. To go back: one declaration and its sentences in the comment above the sheet's template, two lines of the sheet rule's test, and the two tests, the helper and the constant born with the exception. It leaves with the old forms (stages 2, 4, 5 and 7); the bold greyed button of the rebuilt pages is the owner's to judge on the first rebuilt page. | owner; the owner's answer of 2026-10-11, told built at the exit, 2026-10-14 |
| 42 | A length of the approved look moves. A table's last column keeps room after its words where the plan and the approved drawing give it none: the first column's room before its words less the bar, 6 px at the ordinary text size (9, 11, 16 and 19 px at text sizes of 125, 150, 200 and 225 %). So a fitted last column is that much wider and its table's least width with it, and a value or a right-set title in a last column stands that far inside the table's edge on every table, also one that takes no selection and never wears a ring. The ring, its place and its test do not move. | The owner, 2026-10-11, from a picture: the ring round the current row lies 2 to 4 px inside the row's right edge and the words of a last column ended on that edge, so the ring was drawn across the last letter (measured in dark at 100 %: 13 lettering pixels under it for "Done" in a column fitted to it, 17 for "Unfinished", 12 for a value; the approved picture does not show it, because the row it rings holds a word shorter than its column). The owner chose the room. The controller chose its form, so that the ring stands as clear of the words at the right end as at the left at every text size (measured after: no letter crossed in 2,160 of 2,160 cases drawn, where 1,088 were before; 2, 7 and 15 px clear at both ends at 100, 150 and 225 %). | In a FIXED last column the room is taken from the words, and it grows faster than the text because the bar never grows: in the approved health table, whose last column is a fixed 56, "Fine" is cut to "Fi…" at 109 of the 126 text sizes from 100 to 225 % (the first at 104 %), where it was whole at all of them before; the other words the design gives that column ("Failed" and three longer) did not fit it before either, so the stage that mounts that table makes the column fitted or at least 62 wide whatever stands here. A fitted last column loses no word, and its table is 6 to 19 px wider. Two other forms were weighed and not built: 6 px at every text size (the right end then stands 2 px clear where the left stands 7 at 150 %; "Fine" cut at 4 of the 126 sizes), and no room in a table that selects nothing (two tables on one page would then end their last columns differently). To change the amount or the form: one line of one function and one test line. To go back: revert the commit of the room and the three test lines that hold it; the ring then crosses the last letters again, and nothing else moves. | owner (the room); controller (its form; the owner is told its cost in a fixed column at the stage's exit); the owner's answer of 2026-10-11; its form and its cost told to the owner at the exit, 2026-10-14 |
| 43 | A behaviour the design does not have: an ellipsis in a number. A value or a state's word too long for its column is drawn cut at its end with an ellipsis: a value's unit first; then, when not even an ellipsis fits after the number, number and unit cut as one, so that the start of a number is never lost; a value's second line and a state's word likewise. Where the words fit, nothing changes: columns and rows are measured as before, and no pixel of any table the tests draw moved. The plan's sentence that a state's word and a value are not cut is replaced. | The owner, 2026-10-11, from a picture: "101.3 of 128 GiB", 111 px wide, in the approved health table's column of values (a place of 102 px), was drawn "01.3 of 128 GiB": the toolkit cuts lettering at the edge of the rectangle it is drawn in and a value stands at the right of its place, so the start of the number went, with no sign that anything was cut. As built it reads "101.3 of 128 …". | A cut number says that something is cut, not how much: in a place of 34 px "10…" stands for 10.4 and for 101.3 alike. Across a band as wide as an ellipsis (12 px at the ordinary text size, 28 at 225 %) a number that would fit whole loses one to three characters to the ellipsis, because the number shown whole with its unit silently gone was ruled the worse (the controller's ruling: one branch of one function and one test leg to change). Where the unit is cut to the ellipsis alone it stands with no space ("101.3…"). In a place narrower than the ellipsis itself (13 px for a value) nothing is drawn, a state then shows its marker alone, and words of one letter are never cut but clipped as before. The whole words are shown nowhere on screen. No approved column is narrow enough for any of this but the unit's cut. For the stage that first mounts a table: a fixed column of values is sized for its longest number, and a cell that is cut shows its whole words under the pointer. To go back: revert the commit of the cut and the commit of its tests' holds; a too-long value then loses its start again with no sign. | owner (that a cut is shown); controller (number and unit cut as one); the owner's answer of 2026-10-11; its form and its cost told to the owner at the exit, 2026-10-14 |
| 44 | The time axis: an earliest end that lies AFTER the end reads as one firm end, in the line under the axis as on the axis itself ("Expected to end by" the end, with no hatch key; late once the clock has passed the end). The design and the plan say what an end with no earliest end means, and nothing of two ends handed over crossed. | Task 15's review, 2026-10-12: built from the plan's table of forms, the line read a range that runs backwards, with the hatch key ("Expected to end between Thu 8 Oct, 16:56 and Thu 8 Oct, 08:32"), while the axis drew a firm end. By the design's own rule for a fit's range (part 6.3.3: from the later of "cannot end before" and the planned epoch less the patience, to the planned epoch plus the patience) such a pair is handed over whenever a fit's best epoch lies past the planned one and the fit is not yet late, for up to as many epochs as the patience. | The device no longer contradicts itself, but it says "by" the end of a run that by its own measure cannot end before a LATER moment. What the page should say then is for the stage that mounts the axis to design; it may hand the axis a range the right way round instead. To go back: one condition, one docstring sentence and three test rows. | controller (Task 15's review, seat two; its skeptic could not refute it); told to the owner at the exit, 2026-10-14 |
| 45 | A grouped whole number is grouped as its digits stand: a zero a person typed or did not touch is never dropped (`007` is shown `007` and reads 7; `0000` shows `0,000`), where the approved sample's own field and two expectations of the plan's dictated tests read the digits as a number first (`007` shown as `7`). A grouped text longer than the toolkit's limit for one line (32,767 characters) is left ungrouped. No form of today groups: nothing changes on a screen in this stage. | Task 17's review, 2026-10-12 (two readers independently; a skeptic could not refute it): read as a number first, the digits lose every zero behind a first digit that is deleted: in `5,000`, Home, Delete, `8` gave `80`, and one Delete turned `1,000,000` into `0`; 6 of 17 ordinary edits of a first digit came out as meant (17 of 17 as built), and after a text that is no number was set from code every key raised. The design asks that the comma appears as the digits are typed and that the caret keeps its place, and says nothing of zeros. | A grouped box shows a leading zero a person typed until they delete it, and what the window remembers of it keeps the zero where the text has no comma (`007` is saved as `007`; a text with commas is saved as the number it reads). Measured: the ungrouped field, which every form of today uses, differs in nothing (3,000 runs, 42,443 steps). To go back: one line of one method, the two expectations and the test lines that hold the zeros. | controller (Task 17's review, seats two and three; its skeptic); told to the owner at the exit, 2026-10-14 |
| 46 | A whole-number box of today's forms no longer stops at ten digits. The stop was the toolkit's own number check, which the plan takes off both fields (its departures 4 and 16, so that no typed text is rewritten into another number when editing ends); the plan's text did not name the stop as something that goes with it. A text no number can be read from, one of more than 4,300 digits among them, reads as "no number", as letters do, and never makes the field's own code fail. | Found by the implementer and all three readers of Task 17, 2026-10-12. Measured on today's forms at start: 155 number fields at all 61 places they are built, and no fact, no saved key and no pixel differs; what differs is what can be typed. | A number above what a run can use reaches the click and the stage's own refusal, as 2,000,000,000 always could; the Seed box gains (its rule allows twenty digits). From the same review, as the plan intends and for the owner to hear: a decimal comma typed into a box (`6,5`) is now kept, outlined and refused at the click, where eight of nine boxes tried went on silently with a number ten or a hundred times the one meant; but on six of those nine the refusal's sentence says that the box "is blank" while it shows `6,5`, and the right sentence ("Use a point as the decimal sign.") is in the tool tip alone until the form is rebuilt. To put a stop back: one condition of the whole-number key rule and its test. | controller (recorded for the owner; nothing chosen); told to the owner at the exit, 2026-10-14 |
| 47 | A question is wider than the designed 448 px whenever its two answers need more room than its row has; it then grows to the right, and its marker and its sentences stay at the left, where the designed width has them. The design gives a question one width and says nothing of answers that do not fit it. | Task 18's review, 2026-10-13 (two readers; a skeptic settled their measurements): under the window's full dress three of the design's own pairs of answers pass 448 at every text size ("Keep the posterior" with "Load and release the posterior", 8 px over at the ordinary text size, and with "Build and release the posterior", 7 over; "Keep measuring" with "Stop and keep what is measured", 42 over), and "Keep this session" with "Apply and start a new session" passes it at two of four text sizes. As first built the marker and the text then drifted to the right, a third of the excess each (the marker at 31 for 20, the text at 80 for 58); the fix round put a stretch after the text column. An answer is never cut and never wrapped. | Those questions are a little wider than every other; nothing is drawn differently at the designed width (0 of 171 pictures differ). In the form that asks for a line of text, when its ANSWERS widen it, the field ends where the sentence ends and not at the question's right edge (no page of the design has such answers). To hold every question to 448: the answers are worded shorter in the stage that first mounts each, or the design names what a too-long answer does. To go back: one line of the module and its test. | controller (recorded for the owner; the owner judges the wider box on the first page that asks one); told to the owner at the exit, 2026-10-14 |
| 48 | Five things the shared question does that the design and the plan's dictated code do not state, each built by ruling after the task's trial run showed what the dictated build did: (a) a full error opened by keyboard takes the keyboard's focus (as dictated the focus moved on to an answer, and a second press of Space gave it); (b) the Tab key goes through a question in reading order (the list while it scrolls, the field, the link, the other answer, the safe answer, the full error once opened), the focus starting on the safe answer, or in the field of the form that asks for a line; (c) each item of a list is held to the column's width, so that one item that cannot break (a path, a long name) no longer makes every item as wide as itself; (d) the sentence and the closing lines keep their full height when a question is wider than designed; (e) the main answer of the form that asks for a line is greyed before the caller's check of the line is asked, so that a check which fails with an error cannot leave a line takeable that it never passed. The full error's box also stands in a plain holder, so that it is twelve lines high under the window's style sheet as designed (as dictated it came out 192 px high at every text size). | Task 18's preparation and review, 2026-10-13: each was measured on a build of the dictated text against the real devices, and again by the review's drawing seat beside a variant with the five taken out. Beside the approved sample's own question the built one differs in 0 of 57,344 pixels in light and in dark. | Nothing is on a screen in this stage. A person who opens a full error by keyboard has the focus in the error's text and reaches the answers with the Tab key; Enter and Escape still give the safe answer. One item that cannot break is itself still cut at its edge with no sign, as a sentence is today (the owner's pending question on a long path). To go back: one to nine lines each and their test lines. | controller (Task 18's skeptic and its review; told to the owner at the stage's exit); told to the owner at the exit, 2026-10-14 |
| 49 | The time axis: where no tick of the spacing its length chooses falls on it, the next finer spacing is taken, as often as it takes, down to 30 s; an axis that has a tick keeps its spacing, however few its ticks. | The owner, 2026-10-14, from a picture at the stage's exit: about four runs in ten of 5 to 15 minutes and of 2 to 6 hours had no tick and no time under the axis (a run of 9 minutes begun at 16:02; of 3 hours begun at 13:00). | Such an axis now has finer ticks, with seconds in its labels when they are 30 s apart (the 9-minute run: 17 ticks, nine labels); an axis of 30 s or less still has none. No other axis changes (measured over 60,000 moments). To go back: the loop and its comment in one function, and one test. | owner, 2026-10-14 (asked with a picture at the stage's exit); built in task 19q, `80249b6`..`066b55c` |
| 50 | The time axis: a late axis grows by a quarter of the run's planned length (from its start to its end) at a time, 30 s at the least and a day at the most, where row 20 had whole days; a run planned for four days or more grows as before; an end handed over before the start grows by 30 s. Built with row 49, without which a short late axis can have no tick. The function's promise now says it may raise within two days of the calendar's last moment, not one: measured, it already did at the head. | The owner, 2026-10-14, from a picture at the stage's exit: a run of 40 minutes one second late kept 28 px of its bar out of 1,038 (831 now), one of 4 minutes 20 s late 3 px (900 now). | A short late run's axis grows in steps a quarter of its plan long, so its scale changes more often while one watches (a 40-minute run every 10 minutes late; a 4-minute run every minute), and the bar jumps back at each step. To go back: one line of the function, its comment and one test. | owner, 2026-10-14 (asked with a picture at the stage's exit); built in task 19q, `80249b6`..`066b55c` |
| 51 | In the sentence label every message row, page heading and question is set in, a word of 40 characters or more is given a place to break, of no width, after each backslash, slash, underscore and full stop that has more of the word after it, in the bold opening as in the rest, counted in the words as handed in; the label's accessible name is now its words as handed in. The question's line under its field, a plain label, is not changed. | The owner, 2026-10-14, from a picture at the stage's exit: a message naming a long path was cut at the right edge with no sign (the store's folder, a cell file, the saved simulations' folder: 431 to 637 px asked where a page gives a message 354). | A long word that already fitted may break its lines elsewhere (14 of 36 fitting cases tried draw differently; none taller); a word with nothing to break at (a digest) is still cut with no sign; the label's own text carries the invisible characters, so a later device that copies a sentence must take its words, not its text. A sentence with no such word is written as before, to the byte. To go back: revert the commit. | owner, 2026-10-14 (asked with a picture at the stage's exit); built in task 19q, `80249b6`..`066b55c` |
| 52 | A message row with no marker keeps the marker's room and the gap after it: its words start, and end, where a marked row's do at every text size (36 px in at 100 %, where they started 10 px in). | The owner, 2026-10-14, "Line them up", from a picture at the stage's exit. | Such a row's words have 26 px less width at 100 % (39 at 150 %), so a long one may take a line more. To go back: revert the commit. | owner, 2026-10-14 (asked with a picture at the stage's exit); built in task 19q, `80249b6`..`066b55c` |
| 53 | In the figure that compares an FDT ratio across repeats, the repeats take the line colours in force less the mean's blue (#0072B2, found whatever the letter case or form), so that every curve has a colour of its own; the band and the mean stay blue. Under the window's figure settings the repeats are ink, vermilion, green, orange, sky blue and purple, in turn; under matplotlib's own (the command line) nothing changes. An engine file the stage already gave its colours: seven lines added to one function, nothing else. | The owner, 2026-10-14, "Repeats skip the blue", from a picture at the stage's exit: under the window's settings the second repeat was drawn in the mean's blue and could not be told from it. | The repeats' colours under the window's settings shift by one from the third repeat on (the third is green, where it was vermilion), so a figure saved before reads otherwise than one saved after; with seven repeats or more two share a colour, where before it took eight. The first repeat stays the ink of the equilibrium line, and on the command line matplotlib's first blue stays near the mean's. To go back: revert the commit. | owner, 2026-10-14 (asked with a picture at the stage's exit); built in task 19q, `80249b6`..`066b55c` |

**Stage 2, the shell (planned 2026-10-14; rows 69 and 70 are the owner's answers of that day).**

| # | what | why | what it costs if it is wrong | who |
|---|---|---|---|---|
| 54 | The tables of places and keys (`PARTS`, `STEPS`, `PAGES`, `KEYS`) live in `core/gui/keys.py`, a module with no toolkit, where 3.1 puts `PARTS` and `STEPS` in `screens/shell.py`; the rail's devices (`PartRow`, `StepRow`, `RunFoot`, `Rail`) live in `core/gui/widgets/rail.py` and the row of page names in `widgets/page_names.py`; `screens/shell.py` holds `Shell` and the error page. | `core/gui/fields.py` and the chain read the tables and must not import a widget module (the look's modules may not import `fields`, and the refusal tables are read without a window); the devices are tested in the look's suite, which builds devices directly, while the builders' scan closes `screens/` to any class a suite builds by hand. | None on screen. Files are where a reader looks for a device, not where 3.1 says. | controller |
| 55 | Until the checks are counted (stage 5 builds the summary `checks`), step 4 reads "Needs a prior" with no prior in use, "Needs a posterior" with no posterior, and "To come" otherwise (it can start only with both, as today's calibration does), with no count, and it is never the step in hand (amended 2026-10-15 from the review of the build: the first wording read "To come" over today's greyed calibration button). | Read to the letter, `checks=None` counts as none of six passed: after any training step 4 would be "Now: 0 of 6 passed" and Home would say "Next: check the posterior" for ever, though nothing in this stage can mark a check passed, and it would print a headline that does not exist yet. | Until stage 5, after a training Home sends the owner to Inference, not to Checks, and the chain does not say how many checks passed. One branch of `step_states` to go back. | controller |
| 56 | The run's state at the rail's foot has two states in stage 2: "Nothing is running" (the dash, "Only one run at a time.", not a control) and, while a run is live, the Now marker, "Running: <place>" (the step's or the part's name) and "<m:ss> so far. Settings are locked.", ticking once a second while the run is live, as today's run slot does. | `RUN_STATE` carries the running panel and nothing more; the stage line, "stopping", "needs you" and how a run ended come with the run part in stage 3; 13.2.2 asks the stage-2 foot for "which place, for how long", which needs a clock, while 3.5 (the finished foot) runs none. | One timer runs while a run is live (none at start: the launch check holds). Stage 3 replaces the words and the clock. | controller |
| 57 | Closing PRISM still writes what each BUILT part remembers (each panel's own save, as today); a part never opened in the launch writes nothing and keeps its stored values. 3.10 says closing writes only the window's size and place (B6). | The moment each remembered value is written (when applied) is part 7's, built in stages 4 to 7; removing the closing writes now would stop every selection being remembered until then. | None for the owner: what is remembered is what is remembered today. The closing writes go page by page with the design's write-on-apply. | controller |
| 58 | A step whose needs are unmet opens with today's panel, its main button greyed by the panel's own rule; the reason is in the step's row in the rail and in Home's steps table, not beside the button. | Today's panels have no sentence beside their buttons; adding one is the rebuild of each page (5.6). | Until stages 4, 5 and 7 a person reads the reason in the rail, beside the page, not under the button. | controller |
| 59 | The FIRST OPENING of Parameter inference still reads the card (one `nvidia-smi`, three device checks, four reads of the card's memory, as today's constructors do); the launch reads nothing. A standing check holds the list exact; it may only shrink, as the pages are rebuilt. | The six step pages are today's panels, built together because they call one another; the rule that a page's constructor reads no card is kept for every rebuilt page (3.10). | Opening Parameter inference costs what opening PRISM cost before, until stages 4 and 5. | controller |
| 60 | One PRISM at a time: the open PRISM acts on a CONNECTION to its local server, not on a word read from it; the second process allows the open one the foreground, connects and exits 0. A lock left by a PRISM that died is taken at once by the toolkit (it sees the holder's process gone); the three tries a second apart and the removal of the lock are reached only when the holder's process lives but does not answer, or its number was reused. | Measured on this computer (`stage-2/plan-drafts/lockprobe-2026-10-14.out`): a connection is accepted at once while the holder's main thread is busy, but what is written is read only when its loop runs again, and one write of three arrived empty; a fresh lock attempt after a kill succeeds at once; removing the file fails while the holder lives. | None seen. A second start during the open window's own start (before it listens) waits up to three seconds and then hands over or gives up, as 3.10's third case says. | controller (the three scratch tests 3.10 asks, run 2026-10-14) |
| 61 | No row for the owner to run on a real screen is written in stage 2 (as row 21 for stage 1). | The shell's rows (G1 one PRISM at a time from the launcher, G3 the rail and the first opening) start from the launcher with no console (stage 3) or need Help's explanations (stage 8). | The rows are written with their group in stage 3. | controller |
| 62 | Home's next step while a run is live names the place, not the run: "Next: nothing until the run ends" and "A run is going in <place>, and no other run can start until it ends. You can still move around and read results. Keep PRISM open; you can minimise it." | The run's word ("training") is the run part's (stage 3). | Stage 3 rewords one row. | controller |
| 63 | Today's parts offer none of `show_page`, `panels()`, `first_focus()`, `set_small()`; the shell asks for each with `getattr` and does without. A page with no `first_focus()` gives the keyboard to its first SETTING: the first control a person sets (a field, a box, a list of choices), not a button, a log pane or a view (amended 2026-10-15 from Task 5's trial run: read literally, "its first control that takes the keyboard and is not a button" lands on each panel's log pane, the live view and Settings' scrolling area). | 3.2 has every part's top widget offer the four; today's panels are unchanged in this stage. | A move to a page of today puts the keyboard on its first setting as the order of today's controls has it, which may not be the setting 3.2 would choose. One function (`_is_setting`) to change. | controller |
| 64 | The last-resort error, now the shared question in its telling form, does not yet say that the full error is in the window's log file. | The log file is the launcher's (stage 3). | Stage 3 adds the sentence. | controller |
| 65 | A part built while a run is live is locked by the shell from the run state (`BasePanel._running`), as the parts already open are. | 3.10 has the run controller do it (6.12, stage 3); lazy building lands in stage 2 and a part built during a run would otherwise come up editable. | Stage 3 moves the call into the run controller. | controller |
| 66 | A record's label and kind word (10.1's one function) are built in stage 2 (`core/gui/record_words.py`), for Home's latest records and the chain's "Done: <name>". | Home needs them now; 10.1 is Records' (stage 6). | Stage 6 reuses the module. | controller |
| 67 | A row of Home's tables that opens something opens on one click, on Enter and on Space, and shows the pointing hand under the pointer. | The look gives a table no pointer state (4.11) and the three-rule table has no signal for a row opened. | A pointing hand where the look has none. | controller |
| 68 | A latest record on Home opens Records at that record's kind. | A record's own page is stage 6's. | Until stage 6 the owner finds the record in Records' list. | controller |
| 69 | The row of page names over Effective temperature is built in stage 2, with two names ("One cell", "Sweep study"); "Compare" joins it in stage 7; until then each panel keeps its own box for comparing saved runs. | The owner, 2026-10-14, from sketches: "Row of names now". 10.3 builds the row with the part's pages in stage 7; the tab bars go in stage 2. | One more device built a stage early; stage 7 adds a name. | owner, 2026-10-14 (asked with sketches while the plan was written) |
| 70 | The list of your own models and the help text leave Settings in stage 2: "Your own models" shows today's list with its builder (the builder's "Back to settings" reads "Back to your models", and its refusal of a taken name says "edit it from Your own models" where it said "edit it from Settings", the second added 2026-10-15 from Task 6's preparation: Settings no longer holds the list); "Help" shows today's help text and, from Task 9, a table of the keys; Settings keeps the appearance choices. | The owner, 2026-10-14, from sketches: "Move them now". 10.6, 10.7 and 10.8 move them with their stages (7 and 8). | Today's help text is shown under Help until stage 8 rewrites it; the builder's back button and its refusal of a taken name each have one changed place. | owner, 2026-10-14 (asked with sketches while the plan was written) |
| 71 | Two of the parts' lines are worded for what stage 2 holds: Settings "The appearance." (3.6: "Appearance, and this computer."), Help "What each part's settings mean, and the keys." (3.6: "Each part explained, and the science terms."). | "This computer" arrives in stage 4, the glossary in stage 8; a line on Home must not promise what the part does not hold. | Stage 4 and stage 8 each change one line of `keys.PARTS`. | controller |
| 72 | Step 6's `observation_on_offer` is read, in stage 2, from today's Narrowing round panel's own list of observations (its picker holds an entry), after the panels' own gates have refreshed it. | Part 8's list is stage 5's. | None seen. | controller |
| 73 | `InferenceScreen.refresh_gates()` calls every panel's own gate FIRST and then computes the chain's states (3.4 has the states first). | Step 6's state reads the observation list the narrowing round's gate refreshes (row 72). | None seen. | controller |
| 74 | The two layout thresholds are held to the screen through one room the main window hands the look's table when it sizes itself (`design.set_room`), read by `design.small_below()` and `design.two_columns_from()`, each `max(the ordinary number, min(the scaled number, what the screen offers))`, what the screen offers being its height for the small layout and its width for two columns (an amendment of 2026-10-15 to the widest middle a page offers was undone the same day by Task 5's review: at 150 % text on 1366 by 768 it set Home in two columns too narrow for its words, where held to the screen's width Home stacks them): at the ordinary text size they stay 700 and 940; a larger text size raises them only as far as the screen offers. | Row 31 leaves "held to the screen by the shell" open; a page's `Columns` does not know the screen; a plain `min` would also lower the ordinary 700 on a 768-high screen, which 3.8 does not ask. | One more process-wide value (its key in `process_state`, put back after every test). | controller |
| 75 | Ten building tasks and two controller tasks (the design's estimate: ten). | The shell's devices are built and tested before they are mounted, the switch is cut in three so that a failing gate points at one cause, and one PRISM at a time is a task of its own. | About ten full runs of the fast tests. | controller |
| 76 | Table B's retirement of the test that pins the panel docstrings' counts (`tests/test_nav_and_gating.py:4683`) is not done in stage 2. | It reads two files no stage-2 task edits; it breaks only when their words are changed. | None: the test is retired by the task that rewords those docstrings. | controller |
| 77 | While a part is built at its first opening the pointer shows the wait cursor, and the rail marks the part BEFORE it is built, so the first opening of Parameter inference (the card's reads, about two seconds) is seen to be under way; a part that fails to build keeps its row marked over its error page. | 3.2 has `go_to` mark the rail after it builds; the first opening of Parameter inference would otherwise look like a click that did nothing. | None seen. | controller |
| 78 | A second start of PRISM writes one line on its error stream when there is one ("PRISM is already open: its window was brought to the front." or "PRISM is already running but did not answer; this start ends.") before it exits. | 3.10 has it show nothing; from a console (`run.bat`) the line is the only sign the person gets, and under the launcher with no console (stage 3) there is no stream and nothing is written. | None seen. | controller |
| 79 | Three more lines a person reads are worded for what stage 2 holds, as row 71 words two parts' lines: in Help's table of the keys the F1 row reads "Opens Help." (3.7: Help at the section for the page showing, or a setting's explanation) and the Escape row leaves out "an open explanation"; Records' line on Home reads "Everything PRISM has written, and what each record was made from." (3.6 adds "and what uses it"). | In stage 2 F1 only opens Help and there are no explanations (stage 8); today's Records shows what uses a record only when it refuses a delete (stage 6); a line must not promise what the part does not hold (row 71's rule). | Stage 8 changes two rows of `keys.KEYS` and stage 6 one line of `keys.PARTS`. | controller (from Task 2's trial run) |
| 80 | In the small layout (a window lower than the threshold) the OPEN step keeps its second line too, beside the step in hand and the first step that cannot start (3.8 keeps those two only; the approved sample kept the open step's). | Until each step's page says its own reason (stages 4, 5 and 7), the rail is the only place a step's reason is shown (row 58): an open step that lost its second line would show its reason nowhere. | 14 px more in a low window when the open step is neither of the two; one condition to drop when the pages say their reasons. | controller (from Task 3's trial run) |
| 81 | The rail's list shows the look's scroll bar while it scrolls (a window too low for the whole list): the rows narrow by 8 px, the open part's band stops at the bar and the rail's edge line moves 8 px in beside it. Neither the approved sample nor the mock-up draws one. | A list that scrolls with no bar hides that rows are below (Your own models, Settings, Help sit at its end); the look gives every scrolling list its bar. | The band does not reach the page while the list scrolls; one line to hide the bar if the owner prefers. | controller (from Task 3's trial run; for the owner to see at the exit) |
| 82 | The first-time opening's fourth sentence, beside "Read what each part is for", reads "A short section per part." (3.6: "A short section per part, and the science terms."). | Help holds today's help text (a section per panel, with what its settings mean) and no science terms until the glossary (stage 8); a line on Home must not promise what the part does not hold (row 71's rule). 3.6 calls the four sentences final between the parts; no other part prints this one. | Stage 8 changes one string of Home and one line of its test. | controller (from Task 4's trial run) |
