# Prompt for the next chat — the window redesign: stage 2 (the shell), from Task 10

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the session of 19 and 20 October 2026 that built stage 2's Tasks 7, 8 and 9. What in it
is not the owner's own words:

- The owner said nothing in that session beyond the pasted prompt of 16 October
  (`2026-10-16-window-redesign-stage-2-from-task-7.md`), whose words stand: "How much to spend" and the
  usage rule ("Try to also finish things up at a clean state when usage limits are near their max, I
  won't buy anymore credits") are repeated from it.
- Everything else is that session's record: what it learned is added to "How a task is run", "Facts"
  and "Mine to hear". Every count, time, cost and measurement is that session's, not something the owner
  measured.

---

PRISM — the redesign of the window: go on building stage 2, the shell, from Task 10.

**Where things stand.** The plan of stage 2 (`docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md`) is
being built. Tasks 0 to 9 are complete, each building task after one fix round, its review clean and its gate
green. Task 7 (`896725e`..`f07308c`) builds each part when it is first opened: Home alone at start, an error page
with "Try again" for a part that cannot be built, a part built during a run locked, closing writing built parts
only, and opening PRISM reading no card; its review found two Critical faults (a click on Parameter inference's
row while its error page stood; half-built panels breaking every later run's end), cured in its fix round. Task 8
(`b436b4b`..`3b424ab`) makes the refusals name the rail's places and the steps ("in step 5, Inference, or in Live
simulation"). Task 9 (`85463b6`..`c914688`) gives every part, step and the run a key, F6 between the rail and the
page, F1 for Help, and Help a table of the keys. Code head `c914688`; the last gate on it: 1558 passed, 4 skipped, 5 deselected (19 min 28 s, not a quiet-machine time); the real-screen set 4 passed. Departure 59
was amended (a first opening of Parameter inference during a run reads the card then), and the registry gained
`Shell.replace_part`, `models_screen.PAGES`, `places_words`' joining and Task 9's names. I pushed up to `1ad02f5`;
everything after it is not pushed, and I push when I choose. Nothing was running when this was written; check
that the graphics card is free and no python is alive before any work. My real `PRISM.ini` was written at 16:14:13
on 19 October with the same bytes as before (not established by what; every window script since asserts a scratch
settings file): 3,296 bytes, sha256 79BF94DF…C83E is the baseline the gates compare against.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no workflow unless I ask
for one in my own words; every agent given its model by name, Sonnet where reading is enough; one preparation pass a
task (for Task 10, one skeptic on Opus who builds the stand-in and writes its own rulings, marked, for you); the
implementer on Opus; the review by three readers for every task that builds, one skeptic on the Important findings,
a fix re-read on Sonnet. On 16 October I said: "Try to also finish things up at a clean state when usage limits are
near their max, I won't buy anymore credits". So: read the usage card when you start, tell me where the weekly bar
stands, and read it again before every dispatch; do not count on extra usage; when the weekly bar nears its end
(about 90 %), start nothing that cannot be finished below it (at this session's pace a task with its fix round costs
6 to 12 % of the week), and stop at a clean state: no half-built task in the repository, every agent stopped and no
process left, and STATE, the ledger and a prompt like this one written and committed. The week resets on 25 October
at 04:00 UTC (the evening of 24 October here); the bar stood at 24 % when this was written.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND BUILD", only. The
   paragraph "THE BUILD OF STAGE 2, TASKS 7, 8 AND 9" is the last record.
2. The plan: its head whole (for the owner, Global Constraints, Review Focus, File Map, the registry, the standing
   checks, How to run this plan, the departures 54 to 82), then Task 10; Task 11 when its turn comes. The head and
   each task's section are cut in the ledger's `stage-2/briefs/` (`plan-head.md` cut at the hand-over's commit,
   current; cut it again if the plan changes). Stage 1's plan (its head as cut for implementers is
   `stage-1/briefs/plan-head.md` in the ledger) holds the registry of every name stage 1 built and its fourteen
   standing checks. Use `superpowers:subagent-driven-development`; the prompt of 10 October says which skills are
   used and which are not (its own scripts for a workspace, a brief and a review package are not run: the ledger is
   the workspace).
3. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md`: the first 47 lines; then the section "Stage
   2 — the shell (plan)" from the line that begins "- NEW CHAT 2026-10-19 (morning)" to its end (the entries before
   it are summed up in STATE); then "Deviations from the design" rows 54 to 82 and the last ten lines of "Gates".
   Search the stage-2 section for "CARRIED", "carried to Task", "Noticed for later tasks", "FOR THE OWNER AT THE EXIT"
   and "FOR THE STAGE'S LAST REVIEW": Task 9's close gathered Task 10's items in one line, the one that begins "-
   CARRIED to Task 10", and Task 11's after "STILL CARRIED". Never add a line above the three lines that begin "-
   HAND-OFF". Its lines are very long: search it, do not read it through.
4. What the plan was written from, in the ledger's `stage-2/plan-drafts/` (for Task 10 chiefly the lock's scratch
   tests, `lockprobe-2026-10-14.out`): a task's preparation reads the parts of them it needs.
5. The design the plan cites (`docs/superpowers/specs/2026-10-05-window-redesign-design.md`, its section 8, and its
   parts, chiefly `03-shell.md` and, for Task 10, its 3.10 "What is built" and "One PRISM at a time" and
   `06-run.md`'s 6.11), the locked records, the sketches I chose from (`questions/q20a-*`, `q20b-*`) and the approved
   sample, as the prompt of 14 October that starts the build lists them; have readers read what a task needs, and read
   yourself what you rule on.
6. The prompts of 6, 10, 11, 12, 13, 14, 15 and 16 October beside this file, for everything this one does not repeat
   (the skills and plug-ins and their limits, what must not change without my say-so, the standing rules, what "done"
   means for a stage). Where they differ, the later wins, and this one wins over all of them.

**Tools.** In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` writes into `stage-2/gates/` (labels
`s2task<NN>-<sha>`, after a fix `s2task<NN>-fix<k>-<sha>`); `ground.ps1` (from the PowerShell tool) writes there too and
also lists the real settings folder; `ledger_append.py` (mode `stage1` appends at the end of the stage-2 section,
`deviations` after the table's last row, `gates` at the file's end); `insert_rows_after.py` adds a departure row after a
given one in a document; `pictures.py` and `compare_page.py` draw and compare pages for the exit. Text copies: `gate.sh`
and `ground.ps1` in the ledger's `stage-2/tools/`; `task01_test_names.py`, `ledger_append.py`, `insert_rows_after.py`,
`pictures.py` and `compare_page.py` in `stage-1/tools/`; the scratch-settings harness every window script uses,
`stage-2/tools/task-07-fix1-harness.py.txt`. The carried-over list (the ledger's `carried-over.md`) is checked with
`task01_test_names.py --check` on it (at Task 9's close: `rows 367; twice []; missing 0: []`); the scripts that brought it up to date at Tasks 4 to 9's
closes are `s2\co_rows_s2task04.py` to `s2\co_rows_s2task09.py` (models for the next). Stage 2's scratch is `s2\`.

**What to do.** Build Task 10 (one PRISM at a time, the last-resort error through the shared question, the fonts once),
as "How to run this plan" says; then the stage's last review, as stage 1 had (three readers over the stage's whole range,
from the baseline of stage 2's Task 0, then one fix wave), which reads the ledger's lines kept "for the stage's last
review"; then Task 11, the exit, which is yours. After the exit the build pauses for the separate design for saving the
fit, as I decided on 6 October. Hand over to a new chat, with STATE, the ledger and a prompt like this one, at two thirds
of the room or when the usage rule above says stop, whichever comes first.

**Ask me** what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under "What
must not change" in the earlier prompts; one question at a time, in plain words, with a picture and your
recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list of them at
the exit (as `stage-1/exit/rulings-for-the-owner.md` was): I may undo any of it.

**At the exit:** pictures of every place in light and dark, as the plan's Task 11 lists them; the fast gate of record
taken when I have said the computer is quiet (ask me first), its time recorded; the tests that need a real screen run
with hidden windows only.

**How a task is run.** Everything the prompts of 14, 15 and 16 October list, and what the last session learned:

* A subagent sometimes cannot write its report file (the tool refused report files from three of them): every dispatch
  says "if you cannot write it, put the whole report in your final message", and the controller saves it.
* A reader must run NO python that builds a PRISM window or part while the gate runs, not only no pytest: at Task 8 a
  reader's window script ran beside the gate. The review dispatches now say so in capitals.
* Every script that builds PRISM's window or a part uses a scratch settings file and asserts it (the harness above).
* Off-screen, a window-context shortcut fires only on a window brought to the front with `QApplication.setActiveWindow`
  (the tests' helper `_in_front`); a window shown with `WA_DontShowOnScreen` fires nothing otherwise, and
  `activateWindow()` does nothing to it.
* A brief's put-back text must stand once in its file as a STRING, not only as a whole line (the Edit tool matches
  strings): one implementer had to anchor a put-back with its next line.
* A ruling that moves pixels is stated as a condition and measured; the skeptics measured every one this session (Task
  7's 741 of 768, Task 8's 528 boxes, Task 9's 2,432 key presses) and each held.
* The time of a dispatch in the ledger is read from `date`, not guessed (one was guessed and corrected).

**Facts the plan stands on**, as the prompts of 14, 15 and 16 October list them, and, measured since: the window builds
Home alone at start and the other parts at their first opening; opening PRISM reads no card and starts no timer (a
standing check); the first opening of Parameter inference reads the card as listed (a standing check, `nvidia-smi` 1,
the device check 3, the card's memory three ways 4 each); during a run the run's own log keeps its own thread's records,
another thread's going to the console; the window makes nineteen shortcuts as it is built; a test of the gate writes
`Resources/{Bounds,Cells,Units}/sbitest/` and `Resources/Models/SBITEST.json` into the real `Resources/` and takes them
out (older, for the audit's list).

**Mine to hear, still open:** everything the prompts of 14, 15 and 16 October list, and, new from Tasks 7 to 9, for the exit
(each with its pictures in the ledger's `stage-2/pictures/task-07-*`, `task-08-*`, `task-09-*` and `stage-2/reviews/`;
the ledger's lines "FOR THE OWNER AT THE EXIT, from Task 7/8/9" have each): the error page's form and its sentence (a
message that starts with a path or a name is capitalised too); during a run, a page whose settings are all locked giving
the keyboard to its rail row, and F6 from there landing on the page's scrolling column; during a run, a record of the
window's own thread going to the console, not the running page's log; a first opening of Parameter inference during a
run reading the card then; Enter not pressing "Try again"; the joining of places in a refusal and five refusal boxes one
line taller; the Save box's and the passive recording's sentences; the 29 places of the window's own words that still
name an old place (Task 8's preparation report, section 4, and the engine's "Settings model builder"); Help's keys table
under the text and the text's room (today 13 lines of help text at the usual laptop window against 30 with the table
below the text: both layouts drawn in `stage-2/pictures/task-09-fix1/owner-help-room/`; my choice); two scroll bars in a
low Help; step 1's "Apply model & options" taking Alt+Space (carried to the stage that rebuilds step 1); and every other
ruling, in the list of rulings.

Do not start the retrain, or any training of real size, in this chat. Everything else in the prompts of 6, 10, 11, 12,
13, 14, 15 and 16 October stands.
