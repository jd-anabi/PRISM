# Prompt for the next chat — the window redesign: stage 2 (the shell), from Task 4

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the session of 14 and 15 October 2026 that built stage 2's Tasks 0 to 3. What in it is
not the owner's own words:

- The owner gave no new words in that session: it ran from the prompt of 14 October that starts the
  build (`2026-10-14-window-redesign-stage-2-build.md`, whose own header says which of its words are the
  owner's), with the owner's choice on spending of 12 October and the owner's approval of the plan.
  Everything below is that session's record and its way of handing over, to be changed if it is not
  wanted.
- "How much to spend", "Ask me", "At the exit" and "Mine to hear" repeat that prompt; "How a task is run"
  adds what this session learned to its list; the hand-over after about three tasks, or at two thirds of
  the room, is the last line of that prompt's "How a task is run" (the plan has no such rule).
- One instruction is that session's own and not the owner's: under "How much to spend", to finish the
  task at hand and stop when the weekly bar reaches 95 %, so that no task is cut off half-built when
  the week's allowance runs out (extra usage is switched off). Strike it if it is not wanted.
- Every count, time, cost and measurement is that session's record, not something the owner measured.

---

PRISM — the redesign of the window: go on building stage 2, the shell, from Task 4.

**Where things stand.** The plan of stage 2 (`docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md`)
is being built. Task 0 (the workspace and the baseline; no commit) and Tasks 1, 2 and 3 are complete,
each building task after one fix round, its review clean and its gate green: Task 1 the front (the
tests' guards; every window a test builds is let go when its test ends), Task 2 the rules with no widget
(`core/gui/keys.py`, `chain.py`, `record_words.py`), Task 3 the rail's devices and the row of page names
(`core/gui/widgets/rail.py`, `page_names.py`), mounted on no screen yet. Code head `b9efe89`; the last
gate on it: 1485 passed, 4 skipped, 5 deselected (17 min 42 s, not a quiet-machine time). Departures 79,
80 and 81 were added and 55 amended while building (the plan's head, the design's section 8 and the
ledger have them). Nothing pushed after `0747a4c`; I push when I choose. Nothing was running when this
was written; check that the graphics card is free and no python is alive before any work.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no
workflow unless I ask for one in my own words; every agent given its model by name, Sonnet where
reading is enough; one preparation pass a task (the plan has a first reader on Sonnet for Tasks 5
and 6, then a skeptic on Opus who builds the stand-in); the implementer on Opus; the review by three
readers for every task that builds, one skeptic on the Important findings, a fix re-read on Sonnet.
Read the usage card when you start and tell me where the weekly bar stands (it read 69 % when the last
chat began and 82 % when it handed over; the week resets on the evening of 17 October). At the last
chat's pace a task costs about 3 to 5 % of the week, so the bar reaches 95 % about three tasks after
this chat begins: when it reaches 95 %, finish the task at hand (its review and gate), write the state
down and stop until the week resets.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND BUILD",
   only. The paragraph "THE BUILD OF STAGE 2" is the last record.
2. The plan: its head whole (for the owner, Global Constraints, Review Focus, File Map, the registry, the
   standing checks, How to run this plan, the departures 54 to 81), then Task 4; each later task when its
   turn comes. The head and each task's section are cut in the ledger's `stage-2/briefs/`
   (`plan-head.md` was cut at `0db3980`: cut it again from the plan, which now holds rows 79 to 81 and
   the amended 55, before you hand it to an agent). Stage 1's plan (its head as cut for implementers is
   `stage-1/briefs/plan-head.md` in the ledger) holds the registry of every name stage 1 built and its
   fourteen standing checks. The plan's banner offers `superpowers:subagent-driven-development` or
   `executing-plans`: use the first; the prompt of 10 October says which skills are used and which are
   not (its own scripts for a workspace, a brief and a review package are not run: the ledger is the
   workspace).
3. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md`: the first 47 lines; then the
   section "Stage 2 — the shell (plan)" from the line that begins "- NEW CHAT 2026-10-14 (late
   afternoon)" to its end; then "Deviations from the design" rows 54 to 81 and the last ten lines of
   "Gates". In the stage-2 section, search for "CARRIED", "carried to Task" and "Noticed for later tasks":
   what the reviews carried to the tasks ahead is there, and each goes into that task's preparation
   dispatch. Never add a line above the three lines that begin "- HAND-OFF". Its lines are very long:
   search it, do not read it through.
4. What the plan was written from, in the ledger's `stage-2/plan-drafts/` (the three readers' reports, the
   lock's scratch tests, the cross-check): a task's preparation reads the parts of them it needs.
5. The design the plan cites (`docs/superpowers/specs/2026-10-05-window-redesign-design.md`, its section
   8, and its parts, chiefly `03-shell.md`), the locked records, the sketches I chose from and the
   approved sample, as the prompt of 14 October that starts the build lists them; have readers read what
   a task needs, and read yourself what you rule on.
6. The prompts of 6, 10, 11, 12, 13 and 14 October beside this file, for everything this one does not
   repeat (the skills and plug-ins and their limits, what must not change without my say-so, the standing
   rules, what "done" means for a stage). Where they differ, the later wins, and this one wins over all
   of them.

**Tools.** In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` writes into
`stage-2/gates/` (labels `s2task<NN>-<sha>`, after a fix `s2task<NN>-fix<k>-<sha>`); `ground.ps1` (from
the PowerShell tool) writes there too and also lists the real settings folder; their text copies are in
the ledger's `stage-2/tools/`. The other tools (`task01_test_names.py`, `ledger_append.py`,
`insert_rows_after.py`, `pictures.py`, `compare_page.py`) have their text copies in `stage-1/tools/`.
Stage 2's scratch is `s2\`.

**What to do.** Build stage 2 from Task 4, in the plan's order (4, 5, 6, 7, 8, 9, 10, 11), one task at a
time, each with its review and its gate, as "How to run this plan" says; Task 11, the exit, is yours.
After the exit the build pauses for the separate design for saving the fit, as I decided on 6 October.
Hand over to a new chat, with STATE, the ledger and a prompt like this one, after about three tasks or
at two thirds of the room, whichever comes first.

**Ask me** what is mine: what I decided or what the design leaves to me, what cannot be undone, and
anything under "What must not change" in the earlier prompts; one question at a time, in plain words,
with a picture and your recommendation. Decide the small things yourself, write each as a ruling with
its cost, and give me the list of them at the exit (as `stage-1/exit/rulings-for-the-owner.md` was):
I may undo any of it.

**At the exit:** pictures of every place in light and dark, as the plan's Task 11 lists them; the fast
gate of record taken when I have said the computer is quiet (ask me first), its time recorded; the
tests that need a real screen run with hidden windows only.

**How a task is run.** Everything the prompt of 14 October that starts the build lists, and what the
last chat learned:

- The implementer's brief accepts "a later head that changed documents only, named by the dispatch".
  Documents (STATE, a departure row) are committed between a task's close and the next dispatch, or
  between the preparation and the implementer, never while an implementer or a gate is at work; the
  implementer's dispatch then names that head.
- The skeptic writes the brief on its proposed remedies, marked; where the controller accepts them as
  written, the brief stands as walked; where the controller rules otherwise, the skeptic is resumed to
  fold the ruling into the one brief and walk it again end to end, as the plan says.
- Every building task so far needed one fix round: the three readers each found something the
  preparation's one-run faults had not (Task 1's preparation ran 16, Task 2's 117, Task 3's 123): a
  step's state that disagreed with today's panel's button; a guard that would turn one failure into a
  cascade; digits that make a ticking clock jump. Seat two drives TODAY'S panels (or draws against the
  approved sample) and seat three types the faults a later task is most likely to make: keep both
  dispatches pointed that way.
- The fix round's skeptic is given the controller's conditional ruling when a finding conflicts with
  the plan's text, so that it can confirm and cure in one pass.
- The time of a dispatch in the ledger is read from `date`, not guessed.

**Facts the plan stands on**, as the prompt of 14 October lists them, with one change: the session
guard's two lines against the older spelling of a dialog's blocking call are now applied (Task 1,
`6f65567`).

**Mine to hear, still open:** everything the prompt of 14 October that starts the build lists, and,
new from the last chat, for the exit: the rail's scroll bar in a low window (row 81) and the open step
keeping its words in a low window (row 80), with their pictures in the ledger's
`stage-2/pictures/task-03-prep/`; the row of page names spaced by the look's table (16 px between
names) where the sketch I chose drew about 24 (its picture in the same folder); Help's F1 and Escape rows
and Records' line on Home worded for what stage 2 holds (row 79, words only); and every other ruling, in
the list of rulings.

Do not start the retrain, or any training of real size, in this chat. Everything else in the prompts of
6, 10, 11, 12, 13 and 14 October stands.
