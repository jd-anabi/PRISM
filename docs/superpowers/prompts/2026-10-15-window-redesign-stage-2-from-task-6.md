# Prompt for the next chat — the window redesign: stage 2 (the shell), from Task 6

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the session of 15 October 2026 that built stage 2's Tasks 4 and 5. What in it is not
the owner's own words:

- The owner's own words in that session are two, both on spending: "I can buy more credits if the usage
  limit is hit, do not worry about that", and, asked whether to stop at 95 % of the weekly bar after
  Task 5: "You give me the prompt and update all the documentation for task 6 and I can begin it in
  another chat if I decide to work over the weekend". "How much to spend" below follows them: the stop
  at 95 % that the prompt of 15 October (from Task 4) added is gone.
- Everything else repeats the prompt of 15 October that started Task 4
  (`2026-10-15-window-redesign-stage-2-from-task-4.md`, whose own header says which of its words are the
  owner's), with what the session of 15 October learned added to "How a task is run", "Facts" and
  "Mine to hear". The stage's last review before the exit under "What to do" is that session's reading,
  not the owner's words or the plan's: strike it if it is not wanted. Every count, time, cost and measurement is that session's record, not something the
  owner measured.

---

PRISM — the redesign of the window: go on building stage 2, the shell, from Task 6.

**Where things stand.** The plan of stage 2 (`docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md`) is
being built. Tasks 0 to 5 are complete, each building task after one fix round, its review clean and its
gate green: Task 1 the front, Task 2 the rules with no widget, Task 3 the rail's devices, Task 4 Home (the
status page and its first opening, the layout thresholds held to the screen, the ruled table painting in
its measured fonts and breaking a long word), Task 5 the rail mounted (the old header, back arrow, gear,
menu and run slot gone; `go_to`, the foot with the run's state, Help). Code head `827ff3d`; the last
gate on it: 1522 passed, 4 skipped, 5 deselected (25 min 24 s, not a quiet-machine time). Departure 82 was added and 63 amended while building (the plan's head, the
design's section 8 and the ledger have them); 74 was amended and put back the same day (holding the two-column threshold to a page's middle set Home in two columns too narrow for its words at a large text size on a small screen), so it reads as before, with a sentence that records the amendment undone. I pushed up to `3a1944c` on the morning
of 15 October; the sixteen commits after it (`65e94d3`..`827ff3d`) and the one that adds this prompt are not
pushed, and I push when I choose. Nothing was running when this was written; check that the graphics card is free and no python is
alive before any work.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no workflow
unless I ask for one in my own words; every agent given its model by name, Sonnet where reading is
enough; one preparation pass a task (the plan has a first reader on Sonnet for Task 6, then a skeptic on
Opus who builds the stand-in); the implementer on Opus; the review by three readers for every task that
builds, one skeptic on the Important findings, a fix re-read on Sonnet. Read the usage card when you
start and tell me where the weekly bar stands (it read 82 % when the last chat began and 96 % when it
handed over; the week resets on the evening of 17 October). On 15 October I said "I can buy more credits if
the usage limit is hit, do not worry about that" (extra usage is now switched on), so a task is not stopped or
cut short for the limit. At the last chat's pace a task costs about 7 % of the week (82 % to 89 % to 96 % over
Tasks 4 and 5) and 4.5 to 6.5 hours.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND BUILD",
   only. The paragraph "THE BUILD OF STAGE 2, TASKS 4 AND 5" is the last record.
2. The plan: its head whole (for the owner, Global Constraints, Review Focus, File Map, the registry, the
   standing checks, How to run this plan, the departures 54 to 82), then Task 6; each later task when its
   turn comes. The head and each task's section are cut in the ledger's `stage-2/briefs/` (`plan-head.md`
   cut at `47d57f2`: cut it again if the plan has changed since). Stage 1's plan (its head as cut for
   implementers is `stage-1/briefs/plan-head.md` in the ledger) holds the registry of every name stage 1
   built and its fourteen standing checks. Use `superpowers:subagent-driven-development`; the prompt of
   10 October says which skills are used and which are not (its own scripts for a workspace, a brief and
   a review package are not run: the ledger is the workspace).
3. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md`: the first 47 lines; then the
   section "Stage 2 — the shell (plan)" from the line that begins "- NEW CHAT 2026-10-15 (morning)" to its
   end (the entries before it are summed up in STATE); then "Deviations from the design" rows 54 to 82 and
   the last ten lines of "Gates". Search the stage-2 section for "CARRIED", "carried to Task", "Noticed for
   later tasks" and "FOR THE OWNER AT THE EXIT": what the reviews carried to the tasks ahead, and what is
   kept for my eyes at the exit, is there; Task 4's close gathered Task 5's in one line, and Task 5's close
   gathered Task 6's the same way, in the line that begins "- CARRIED to Task 6": today's headings and tab bars
   flush against the rail (the old header's frame gone); Your own models opening the builder still editing the
   last model; until the tab bars go, a change of Parameter inference's tab marking the rail (replaced by
   `show_step`); the unknown page's test leg kept in the rewrite of `_move`; the Help test's line that compares
   Help with Settings' viewer; the minors each fix skeptic left for Task 6. Never add a line above the three lines that begin "- HAND-OFF". Its
   lines are very long: search it, do not read it through.
4. What the plan was written from, in the ledger's `stage-2/plan-drafts/`: a task's preparation reads the
   parts of them it needs.
5. The design the plan cites (`docs/superpowers/specs/2026-10-05-window-redesign-design.md`, its section
   8, and its parts, chiefly `03-shell.md` and, for Task 6, `10-records-and-parts.md`), the locked records,
   the sketches I chose from (`questions/q20a-*`, `q20b-*`) and the approved sample, as the prompt of 14
   October that starts the build lists them; have readers read what a task needs, and read yourself what
   you rule on.
6. The prompts of 6, 10, 11, 12, 13, 14 and 15 October beside this file, for everything this one does
   not repeat (the skills and plug-ins and their limits, what must not change without my say-so, the
   standing rules, what "done" means for a stage). Where they differ, the later wins, and this one wins
   over all of them.

**Tools.** In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` writes into
`stage-2/gates/` (labels `s2task<NN>-<sha>`, after a fix `s2task<NN>-fix<k>-<sha>`); `ground.ps1` (from the
PowerShell tool) writes there too and also lists the real settings folder; `ledger_append.py` (mode
`stage1` appends at the end of the stage-2 section, `deviations` after the table's last row, `gates` at
the file's end); `insert_rows_after.py` adds a departure row after a given one in a document; `pictures.py` and
`compare_page.py` draw and compare pages for the exit. Text copies: `gate.sh` and `ground.ps1` in the ledger's
`stage-2/tools/`; `task01_test_names.py`, `ledger_append.py`, `insert_rows_after.py`, `pictures.py` and
`compare_page.py` in `stage-1/tools/`. The carried-over list (the ledger's `carried-over.md`) is checked with
`task01_test_names.py --check` on it (at Task 5's close: `rows 365; twice []; missing 0: []`); the scripts that
brought it up to date at Tasks 4 and 5's closes are `s2\co_rows_s2task04.py` and `s2\co_rows_s2task05.py`
(models for the next). Stage 2's scratch is `s2\`.

**What to do.** Build stage 2 from Task 6, in the plan's order (6, 7, 8, 9, 10, 11), one task at a time,
each with its review and its gate, as "How to run this plan" says; Task 11, the exit, is yours. Before it, the
stage's last review, as stage 1 had (three readers over the stage's whole range, then one fix wave), which reads
the ledger's lines kept "for the stage's last review" (the plan does not name it; the last chat's reading of
its own rulings, which deferred minors to it). After the
exit the build pauses for the separate design for saving the fit, as I decided on 6 October. Hand over to
a new chat, with STATE, the ledger and a prompt like this one, after about three tasks or at two thirds
of the room, whichever comes first.

**Ask me** what is mine: what I decided or what the design leaves to me, what cannot be undone, and
anything under "What must not change" in the earlier prompts; one question at a time, in plain words,
with a picture and your recommendation. Decide the small things yourself, write each as a ruling with
its cost, and give me the list of them at the exit (as `stage-1/exit/rulings-for-the-owner.md` was): I
may undo any of it.

**At the exit:** pictures of every place in light and dark, as the plan's Task 11 lists them; the fast
gate of record taken when I have said the computer is quiet (ask me first), its time recorded; the tests
that need a real screen run with hidden windows only.

**How a task is run.** Everything the prompts of 14 and 15 October list, and what the last chat learned:

* For a task with a first reader (Task 6), the controller rules on the reader's open points in a file of
  its own (`stage-2/briefs/task-NN-rulings.md`, each ruling binding unless the trial run shows it wrong)
  and hands it to the skeptic with the notes: Task 5's skeptic applied all sixteen and brought none back.
  A ruling can still be wrong where the trial run cannot see it: Task 5's ruling on the two-column
  threshold looked right on the stand-in and set two columns too narrow for Home's words at a large text
  size; seat two found it by drawing Home at 150 % on a small screen. Keep seat two drawing every place
  at the large text size on a small screen.
* Every building task so far needed one fix round; in Task 5 three seats found the same fault (a click
  on a rail row taken for a key press, because the window read where the keyboard was instead of what
  pressed the row). A dispatch to seat three names the faults the NEXT tasks are most likely to make.
* A check that reads only tracked files (`git grep`) prints nothing for a file not yet added: the
  implementer's dispatch says so, and to confirm such a count read-only and go on (Task 4's implementer
  stopped there once).
* At a task's close, the items carried to the next task are gathered in ONE ledger line, "CARRIED to Task
  N", which the next dispatch reads.
* The time of a dispatch in the ledger is read from `date`, not guessed.

**Facts the plan stands on**, as the prompt of 14 October lists them, and, measured since: the offscreen
platform's screen is 800 by 800, so a window the suites show is 740 high, in the usual layout; hidden
windows get no focus events, so a test of the focus ring feeds the follower by hand; the window already
hands the look its room when it sizes itself (Task 4); the session guard's two lines against the older
spelling of a dialog's blocking call are applied (Task 1, `6f65567`); the real `PRISM.ini` was written at
09:41:57 on 15 October, thirteen seconds after my push of `3a1944c`, with the same bytes and hash as before,
while nothing of the build ran (not explained; the last chat did not ask me).

**Mine to hear, still open:** everything the prompts of 14 and 15 October list, and, new from the last
chat, for the exit (each with its pictures in the ledger's `stage-2/pictures/` and `stage-2/reviews/`):
Home's latest records below the window's foot on a laptop or at a large text size, against the layout
with the records under the steps (the ledger's "Task 4, Ruling (P4)"); the first opening's sentence beside "Read what each part
is for" reading "A short section per part." until the glossary (row 82); a measurement's kind word on two
lines on Home instead of cut; "Simulation set for unnamed prior, …" in small letters; "All records" opening
the last kind chosen; Enter not pressing Home's buttons (a choice for the whole window); a mouse press on
Home's buttons or the foot giving a typed field the ring; at the smallest window Live simulation and
Reduction map scrolling sideways as today, the live view's title cut ("Hair-bundle displacemen") and
Records' Created column cut beside the rail; during a run, a page whose settings are all locked giving the keyboard to nothing (unless Task 7 cures it); Home staying in one column at a large text size on a small screen (a rule led by the blocks' own widths
was described in Task 5's fix report, not built); and every other ruling, in the list of rulings.

Do not start the retrain, or any training of real size, in this chat. Everything else in the prompts of
6, 10, 11, 12, 13, 14 and 15 October stands.
