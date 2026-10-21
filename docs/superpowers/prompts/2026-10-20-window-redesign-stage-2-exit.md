# Prompt for the next chat — the window redesign: stage 2 (the shell), the exit from its Step 4

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the session of 20 October 2026 that built stage 2's Task 10, ran the stage's last review
and began the exit. What in it is not the owner's own words:

- The owner's words in that session were the pasted prompt of 20 October
  (`2026-10-20-window-redesign-stage-2-from-task-10.md`), whose words stand, and the answers to the
  questions that session asked: fourteen answers (Q1 to Q14), each the recommendation, recorded in the
  ledger's lines "THE OWNER'S ANSWERS, ASKED EARLY" and "THE OWNER'S ANSWERS ON TASK 10'S REVIEW"; "yes"
  to drafting the list of rulings early; and "Quiet now: run it" for the gate of record.
- Everything else is that session's record. Every count, time, cost and measurement is that session's, not
  something the owner measured.

---

PRISM — the redesign of the window: finish stage 2's exit, Task 11, from its Step 4.

Where things stand. The plan of stage 2 (docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md) is
built: Tasks 0 to 10 are complete, each building task after one fix round, its review clean and its gate
green; the stage's last review (three readers over `0db3980..d8b68a1`) is complete with its one fix wave
(`9ababa1`..`c7106c3`), its re-read clean. Task 10 (`d5a7a0b`..`ffc0a93`) made one PRISM at a time (a lock
beside my settings file; a second start hands over to the open window, or gives up with status 1), the
last-resort error through the shared question, an error's message written as a sentence by my rule, the
class fonts set once; its review found a Critical fault (a PRISM whose window closed during a run turned
later starts away), cured as I chose: closing the window lets go of the lock. The last review found the same
kind of fault with a popped-out figure window open, cured the same way, and a gap in the card checks.
Code head `c7106c3`; documents above it. The exit's Steps 1 to 3 are done: the fence holds (nothing under
`core/` outside `core/gui/`; `core/config.py`, `README.md`, `CLAUDE.md`, `docs/checklists/`, the launchers
untouched; 25 suites; 109 refusal keys); THE GATE OF RECORD on a quiet machine at the owner's word, alone, on `ec6147d` (code `c7106c3`): 1583 passed, 5 skipped, 5 deselected, 205 warnings, 19 min 21 s, exit 0 (3 min 21 s over the 16-minute target, with 176 more tests than stage 1's record of 1407 in 16 min 9 s; the busy gates of the day took 19:22 to 19:27, so the time is the tests', not the machine's); the real-screen set 5 passed, 1588 deselected, nothing on the screen and no window activated; 1,593 collected, checked against the carried-over list (`rows 369; twice []; missing 0: []`; the 22 gone each with its closing fate). I pushed up
to `1ad02f5`; everything after it is not pushed, and I push when I choose. Nothing was running when this was
written; check that the graphics card is free and no python is alive before any work (a `python -m core.gui`
started from `run.bat` is my own PRISM: leave it alone and wait). My real `PRISM.ini` is 3,296 bytes, sha256
79BF94DF…C83E, written 09:54:52 on 20 October by my own PRISM; since Task 10, while my PRISM is open a
`PRISM.lock` stands beside it, so nothing is gated while my PRISM is open.

How much to spend is decided: do not ask again. "In between", as I chose on 12 October: no workflow unless I
ask for one in my own words; every agent given its model by name, Sonnet where reading is enough. On 16
October I said: "Try to also finish things up at a clean state when usage limits are near their max, I won't
buy anymore credits". So: read the usage card when you start, tell me where the weekly bar stands, and read
it again before every dispatch; do not count on extra usage; when the weekly bar nears its end (about 90 %),
start nothing that cannot be finished below it, and stop at a clean state. The week resets on 25 October at
04:00 UTC; the bar stood at 39 % when this was written.

Read first, in this order.

1. docs/STATE.md — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND BUILD", only.
   The paragraph "THE BUILD OF STAGE 2, TASK 10, ITS LAST REVIEW AND THE EXIT'S FIRST STEPS" is the last
   record.
2. The plan: its head as cut for agents (the ledger's `stage-2/briefs/plan-head.md`, current: departures 54
   to 82, the registry, the standing checks) and Task 11 (`stage-2/briefs/task-11-section.md`). Stage 1's exit
   is the model of this one: STATE's paragraph "STAGE 1 OF THE WINDOW REDESIGN IS BUILT" and the ledger's
   `stage-1/exit/`.
3. The ledger, .superpowers/sdd/2026-10-06-window-redesign/progress.md: the first 47 lines; then the section
   "Stage 2 — the shell (plan)" from the line that begins "- NEW CHAT 2026-10-20 (morning)" to its end; then
   "Deviations from the design" rows 54 to 82 and the last ten lines of "Gates". Search the stage-2 section for
   "FOR THE OWNER AT THE EXIT", "CARRIED", "THE OWNER'S ANSWERS" and "TASK 11". Never add a line above the three
   lines that begin "- HAND-OFF". Its lines are very long: search it, do not read it through.
4. The last review's reports in the ledger's `stage-2/reviews/`: `stage-2-final-review-sorting.md` (its table:
   12 things to tell me, 48 to carry by stage, 24 for the audit's list, 41 to drop), the product seat's section
   "For stage 3's planner" and the tests seat's "For the next stage's planner".
5. The draft list of rulings, `stage-2/exit/rulings-for-the-owner.md` (drafted from the ledger up to Task 10's
   review; entry A-21 corrected by the last review): it is completed at the exit with Task 10's fix round, the
   last review, its wave and the exit itself.
6. The prompts of 6, 10, 11, 12, 13, 14, 15, 16 and 20 October beside this file, for everything this one does
   not repeat. Where they differ, the later wins, and this one wins over all of them.

Tools. In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` (kinds fast, quiet, display,
collect) and `ground.ps1` write into `stage-2/gates/`; `ledger_append.py` (mode `stage1` appends at the end of
the stage-2 section, `deviations` after the table's last row, `gates` at the file's end); `pictures.py` and
`compare_page.py` drew stage 1's exit sets and are to be taught the rail (Task 11 Step 4); the scratch-settings
harness every window script uses is the ledger's `stage-2/tools/task-07-fix1-harness.py.txt`; text copies of
the tools are in the ledger's `stage-1/tools/` and `stage-2/tools/`. The carried-over list (`carried-over.md`)
is current to the last review's wave (`task01_test_names.py --check` on it: `rows 369; twice []; missing 0:
[]`).

What to do. Task 11 from Step 4, as its section says, with what was learned:

* Step 4, the pictures: `pictures.py` taught the rail (every part, each of the six steps, both Effective
  temperature pages, Home first-time on an empty scratch store and mid-project on a store filled by the suites'
  record builders, the foot while a run is shown, a part's error page, every part at the smallest window, and a
  page after a click on a rail row and after the same move by its key, to show where the ring stands), and the
  last-resort question rendered; three sets (plain, a forced text factor of 1.5, 150 % scaling), light and dark;
  a comparison page against stage 1's exit set; the nine checks of a picture run on each. Every picture made on
  the real platform with the window hidden, on a scratch settings file it asserts and a scratch store, every
  dialog rendered and never shown, and NO WINDOW EVER ACTIVATED (measured on 20 October: bringing even a
  never-shown window forward takes my keyboard; it happened to me about six times during Task 10's
  preparation); a picture agent (Opus) builds and runs it and writes an index; you look only at what you put to
  me.
* Step 5, my questions: fourteen are answered (the ledger has each); still to ask, each alone in plain words
  with its pictures and your recommendation: the last-resort question's look (its pictures are in the ledger's
  `stage-2/pictures/task-10-fix1/`: the long message cut to its screen, "Can't …", the general sentence before
  a quoted name); the corrected counts in `CLAUDE.md` (25 suites; the session's guards and per-test checks as
  Tasks 1, 4 and 10 and the last review left them: put the sentences to me, edit only on my yes, in a commit
  of its own); and anything the pictures show that is mine. Then tell me the rest, grouped, shortest first:
  the sorting's twelve, the ledger's "FOR THE OWNER AT THE EXIT" lines not yet asked, and that the cut
  question grows past a 768-high screen again when its full error is opened.
* Step 6, the documents: the departures brought into line in the design's section 8, the plan and the ledger
  (63's first sentence, 74 saying the room is the primary screen's read once, 76's reason, the plan's file map
  crediting Task 10 with the lettering file's warning, which was stage 1's; 78 is done); `docs/guide/window.md`
  gains its one line, `tests/test_docs.py` green; `docs/STATE.md`'s stage paragraph at the top; the list of
  rulings completed and shown to me.
* Step 7: what is carried on (the ledger's "CARRIED" lines for stage 3, 4, 6, 7, the sorting's table, the
  audit's list) written into STATE and the ledger.
* Step 8: the build pauses for the separate design for saving the fit, as I decided on 6 October. Bring
  STATE, the ledger and the memory up to date, commit, and tell me; write no prompt for stage 3 until that
  design is made.

Ask me what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under
"What must not change" in the earlier prompts; one question at a time, in plain words, with a picture and your
recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list of
them at the exit. Hand over to a new chat, with STATE, the ledger and a prompt like this one, at two thirds of
the room or when the usage rule says stop, whichever comes first.

Do not start the retrain, or any training of real size, in this chat. Everything else in the prompts of 6, 10,
11, 12, 13, 14, 15, 16 and 20 October stands.
