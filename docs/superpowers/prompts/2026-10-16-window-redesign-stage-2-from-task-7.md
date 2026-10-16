# Prompt for the next chat — the window redesign: stage 2 (the shell), from Task 7

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the session of 15 and 16 October 2026 that built stage 2's Task 6. What in it is not
the owner's own words:

- The owner's own words in that session were "Try again" (when the usage limit cut off Task 7's
  preparation) and "Try to also finish things up at a clean state when usage limits are near their
  max, I won't buy anymore credits", which stopped it. Everything else is that session's record and its
  way of handing over, to be changed if it is not wanted.
- Under "How much to spend", how the owner's words of 16 October are applied (the bar read before each
  dispatch; nothing started that cannot be finished below the limit; the stop at a clean state) is that
  session's reading, not the owner's.
- "Ask me", "At the exit" and "Mine to hear" repeat the earlier prompts; "How a task is run" adds what
  that session learned. Every count, time, cost and measurement is that session's record.
- This prompt was not checked by a reader (the check cost about 0.24 million tokens at the last two
  hand-overs; left out to spare the owner's usage): read it against the ledger before acting on it.

---

PRISM — the redesign of the window: go on building stage 2, the shell, from Task 7.

**Where things stand.** The plan of stage 2 (`docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md`) is being
built. Tasks 0 to 6 are complete, each building task after one fix round, its review clean and its gate green. Task 6
(`22b8827`..`c1628d0`) took every tab bar and section heading away: the six steps are pages and every step opens;
Effective temperature chooses "One cell" or "Sweep study" by the row of page names; Live simulation and the Reduction
map are their panels; Your own models opens on the list of my models over the builder; Help holds the help text;
Settings keeps the appearance; every place but Home stands 12 px from the rail; and its fix round mended the Training
step's cut budget lines. Code head `c1628d0`; the last gate on it: 1532 passed, 4 skipped, 5 deselected (15 min 19 s,
not a quiet-machine time). Departure 70 was amended while building (the builder's "edit it from Your own models"). Task
7's preparation was begun and stopped when I said to stop at a clean state: nothing of it is in the repository, and its
dispatch is kept for reuse (the ledger's `stage-2/briefs/task-07-prep-dispatch.md`, which also says what its scratch
holds; do not trust that scratch, start the preparation afresh). I pushed up to `3a1944c` on 15 October; everything
after it is not pushed, and I push when I choose. Nothing was running when this was written; check that the graphics
card is free and no python is alive before any work. My real `PRISM.ini` was rewritten at 20:54:18 on 15 October, before
that session's first command (most likely my own PRISM opened and closed): 3,301 bytes, sha256 FF1B0FDB…39D5 is the
baseline the gates compare against.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no workflow unless I ask
for one in my own words; every agent given its model by name, Sonnet where reading is enough; one preparation pass a
task (for Task 7, one skeptic on Opus who builds the stand-in and writes its own rulings, marked, for you); the
implementer on Opus; the review by three readers for every task that builds, one skeptic on the Important findings, a
fix re-read on Sonnet. On 16 October I said: "Try to also finish things up at a clean state when usage limits are near
their max, I won't buy anymore credits". So: read the usage card when you start, tell me where the weekly bar stands,
and read it again before every dispatch; do not count on extra usage; when the weekly bar nears its end (about 90 %),
start nothing that cannot be finished below it (a task's preparation, build, review, fix round and gate together cost
about 7 to 8 % of the week at the last pace), and stop at a clean state: no half-built task in the repository, every
agent stopped and no process left, and STATE, the ledger and a prompt like this one written and committed. The week
resets on the evening of 17 October (9 pm), and the bar then starts from nothing.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND BUILD", only. The
   paragraph "THE BUILD OF STAGE 2, TASK 6" is the last record.
2. The plan: its head whole (for the owner, Global Constraints, Review Focus, File Map, the registry, the standing
   checks, How to run this plan, the departures 54 to 82), then Task 7; each later task when its turn comes. The head
   and each task's section are cut in the ledger's `stage-2/briefs/` (`plan-head.md` cut at `22b8827`, current; cut it
   again if the plan changes). Stage 1's plan (its head as cut for implementers is `stage-1/briefs/plan-head.md` in the
   ledger) holds the registry of every name stage 1 built and its fourteen standing checks. Use
   `superpowers:subagent-driven-development`; the prompt of 10 October says which skills are used and which are not (its
   own scripts for a workspace, a brief and a review package are not run: the ledger is the workspace).
3. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md`: the first 47 lines; then the section "Stage
   2 — the shell (plan)" from the line that begins "- NEW CHAT 2026-10-15 (evening)" to its end (the entries before it
   are summed up in STATE); then "Deviations from the design" rows 54 to 82 and the last ten lines of "Gates". Search the
   stage-2 section for "CARRIED", "carried to Task", "Noticed for later tasks" and "FOR THE OWNER AT THE EXIT": Task 6's
   close gathered Task 7's items in one line, the one that begins "- CARRIED to Task 7", and the later tasks' after
   "STILL CARRIED". Never add a line above the three lines that begin "- HAND-OFF". Its lines are very long: search it,
   do not read it through.
4. What the plan was written from, in the ledger's `stage-2/plan-drafts/`: a task's preparation reads the parts of them
   it needs.
5. The design the plan cites (`docs/superpowers/specs/2026-10-05-window-redesign-design.md`, its section 8, and its
   parts, chiefly `03-shell.md` and, for Task 7, its 3.10 and `06-run.md`'s 6.12), the locked records, the sketches I
   chose from (`questions/q20a-*`, `q20b-*`) and the approved sample, as the prompt of 14 October that starts the build
   lists them; have readers read what a task needs, and read yourself what you rule on.
6. The prompts of 6, 10, 11, 12, 13, 14 and 15 October beside this file, for everything this one does not repeat (the
   skills and plug-ins and their limits, what must not change without my say-so, the standing rules, what "done" means
   for a stage). Where they differ, the later wins, and this one wins over all of them.

**Tools.** In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` writes into `stage-2/gates/` (labels
`s2task<NN>-<sha>`, after a fix `s2task<NN>-fix<k>-<sha>`); `ground.ps1` (from the PowerShell tool) writes there too and
also lists the real settings folder; `ledger_append.py` (mode `stage1` appends at the end of the stage-2 section,
`deviations` after the table's last row, `gates` at the file's end); `insert_rows_after.py` adds a departure row after a
given one in a document; `pictures.py` and `compare_page.py` draw and compare pages for the exit. Text copies: `gate.sh`
and `ground.ps1` in the ledger's `stage-2/tools/`; `task01_test_names.py`, `ledger_append.py`, `insert_rows_after.py`,
`pictures.py` and `compare_page.py` in `stage-1/tools/`. The carried-over list (the ledger's `carried-over.md`) is
checked with `task01_test_names.py --check` on it (at Task 6's close: `rows 366; twice []; missing 0: []`); the scripts
that brought it up to date at Tasks 4 to 6's closes are `s2\co_rows_s2task04.py` to `s2\co_rows_s2task06.py` (models for
the next). Stage 2's scratch is `s2\`.

**What to do.** Build stage 2 from Task 7, in the plan's order (7, 8, 9, 10, 11), one task at a time, each with its
review and its gate, as "How to run this plan" says; Task 11, the exit, is yours. Before it, the stage's last review, as
stage 1 had (three readers over the stage's whole range, then one fix wave), which reads the ledger's lines kept "for the
stage's last review" (the plan does not name it; it is the reading of the chats of 15 October, whose reviews deferred
minors to it). After the exit the build pauses for the separate design for saving the fit, as I decided on 6 October.
Hand over to a new chat, with STATE, the ledger and a prompt like this one, after about three tasks, at two thirds of the
room, or when the usage rule above says stop, whichever comes first.

**Ask me** what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under "What
must not change" in the earlier prompts; one question at a time, in plain words, with a picture and your recommendation.
Decide the small things yourself, write each as a ruling with its cost, and give me the list of them at the exit (as
`stage-1/exit/rulings-for-the-owner.md` was): I may undo any of it.

**At the exit:** pictures of every place in light and dark, as the plan's Task 11 lists them; the fast gate of record
taken when I have said the computer is quiet (ask me first), its time recorded; the tests that need a real screen run
with hidden windows only.

**How a task is run.** Everything the prompts of 14 and 15 October list, and what the last chat learned:

* A ruling the trial run cannot see can still be wrong: state a ruling that moves pixels as a condition to be measured
  ("cure it, on condition that every other panel draws pixel-identical; if anything else moves, bring it back"). In
  Task 6's fix round the cure the controller named first moved 90 pictures that had nothing wrong; the skeptic measured
  it and found a narrower one. Keep seat two drawing every place at the large text size on a small screen, and driving
  with REAL inputs (a real bounds and cell file, a real configuration) as well as stand-ins: Task 6's cut budget lines
  hid behind the preparation's stand-in configuration.
* Seat three's dispatch names the faults the NEXT tasks are most likely to make; in Task 6 every Important finding was a
  test gap of that kind, each with its cure line already run.
* Pictures drawn in two processes differ unless `PYTHONHASHSEED=0` is set (the Config tab's units line lists its
  units in the order of a set); a copy of the tree made with `tar` holds `run.sh` as 100644 where the repository has
  100755, so its tree ids differ while every commit's patch is the same.
* If the usage limit cuts an agent off, check that it left no process and that its scratch copies hold no fault typed
  in, then resume it (`SendMessage`) only if the usage rule allows; otherwise stop it and keep its dispatch for reuse.
* A check that reads only tracked files (`git grep`) prints nothing for a file not yet added: the implementer's dispatch
  says so, and to confirm such a count read-only and go on.
* At a task's close, the items carried to the next task are gathered in ONE ledger line, "CARRIED to Task N", which the
  next dispatch reads. The time of a dispatch in the ledger is read from `date`, not guessed.

**Facts the plan stands on**, as the prompts of 14 and 15 October list them, and, measured since: the offscreen
platform's screen is 800 by 800, so a window the suites show is 740 high, in the usual layout; hidden windows get no
focus events, so a test of the focus ring feeds the follower by hand; the toolkit's own child margin on this computer is
9 px (Records and Help stand on it; the margins test holds it).

**Mine to hear, still open:** everything the prompts of 14 and 15 October list, and, new from Task 6, for the exit (each
with its pictures in the ledger's `stage-2/pictures/task-06-prep/`, `task-06-fix1/` and `stage-2/reviews/`): Effective
temperature opening on "One cell" in every launch and on the page last shown afterwards (the design's "every time" read
as every launch); Your own models opening on its list in every launch; every place but Home 12 px from the rail, as the
sketch I chose drew it, while Home keeps its own wider margin, and Settings' group 22 px in because its frame is kept; Help still printing "split into six
tabs" until it is rewritten; the panel column that mends the cut budget lines costing a step change about 5 ms at the
median and up to 82 ms at the longest; and every other ruling, in the list of rulings.

Do not start the retrain, or any training of real size, in this chat. Everything else in the prompts of 6, 10, 11, 12,
13, 14 and 15 October stands.
