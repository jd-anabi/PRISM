# Prompt for the next chat — the window redesign: go on with stage 1, from Task 15

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice. These things in it are not the owner's own words, and are there to be changed if they
are not wanted:

- The section "How much to spend" is the last chat's proposal, made when the owner asked how to ease
  the weekly token usage. The owner has not chosen yet: the first thing the next chat does is ask.
- Everything under "Where things stand", every count and every measurement is the last chat's record.
- The prompt of 10 October (`2026-10-10-window-redesign-continue-stage-1.md`, beside this file) still
  holds for everything this one does not say: the skills and plug-ins and their limits, how to work,
  what must not change without the owner's say-so, the standing rules, what "done" means. Where the
  two differ, this one wins.

---

PRISM — the redesign of the window: go on with stage 1, from Task 15.

**Where things stand.** Stage 1 of the redesign, the look's foundation, is being built from the plan
I approved on 7 October 2026. Since the chat of 10 October these are done, each with its tests
first, a review and a one-process fast gate: my three answers on the pictures (built as Task 12q);
Task 13, the painted devices, with a small tests-only front; Task 14, the three-rule table, with a
small front of mends. The last commit that touched code is `ac8416a`; the last fast gate ran on it:
1328 passed, 3 skipped, 5 deselected. Tasks 15 to 19 are left. Nothing has been pushed since
`cb2b0de`: I push when I choose. Nothing was running when this was written; check that the graphics
card is free and no python is alive before any work.

**The first thing: ask me how much to spend.** The last chat used about a third of my week's
allowance in a day, nearly all of it on background agents that read and check. Ask me, once and in
plain words, which way to go on (the section "How much to spend" below has the last chat's proposal
and its price), and work that way until I say otherwise. Also tell me whether Ultracode is on in
this chat: if I chose the leaner way and it is on, say so, because it asks for the opposite.

**The second thing: my two answers on the table, as one small task before Task 15.** The review of
Task 14 found two things the approved sample itself does, and I chose the remedy for each on
11 October:

- the last column of a table keeps a little room after its words (6 px at ordinary size), so that
  the ring round the current row crosses no letter;
- a number or a state's word that is too long for its column is shown cut with an ellipsis: the
  unit shortened first, then the number, and the start of the number is never lost.

Both are changes to the look as built, so they are a task of their own: tests first, each seen to
fail, a review, a gate, and deviation rows 42 and 43 in the design's section 8, the plan's head and
the ledger. The text for the first was written and measured by the review's skeptics; the text for
the second was drafted and NOT run, so it is run on the product before anyone builds from it. Both
are kept in the ledger, `stage-1/reviews/task-14-owner-answers-cures.md`. When it is built, show me
the same two pictures again, as built (`questions/q9-…` and `q10-…` in the ledger are the ones I
saw).

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND
   BUILD", only. The paragraph "THE NEW CHAT" is the last chat's record.
2. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/` (gitignored): of `progress.md` the
   first 47 lines, then everything from the line that begins "- NEW CHAT 2026-10-10" to the end of
   that section (the last chat's entries: every ruling it made, what each task's review found, what
   is carried to which later task), then the last three sections. It is the record and the recovery
   map: trust it and `git log` over anything remembered. Never add a line to it above the three
   lines that begin "HAND-OFF": another file cites it by line number.
3. `stage-1/carry-to-tasks-13-19.md` in the ledger: its sections for Tasks 15 to 19 and "Every later
   task of the stage".
4. The plan's head, `stage-1/briefs/plan-head.md` in the ledger (cut again on 11 October: it holds
   deviation rows 1 to 41), down to the first task.
5. The prompt of 10 October, beside this file, for everything this one does not repeat.

**Task 15, the time axis, is prepared.** Its brief was checked against the code and nine open
choices are ruled in the ledger (search `progress.md` for "Task 15: Ruling"). Its second notes, the
file an implementer is given beside the brief, were being written when the last chat ended:
`stage-1/briefs/task-15-dispatch-notes-v2.md`. Check that the file is there and finished (it ends
with a list headed "STOP and report"). What its last writer had to say to the controller (what to
rule, what to say in the dispatch, which predictions nobody ran) is the last result line of
`C:\Users\J\.claude\projects\C--Users-J-PycharmProjects-PRISM\8ead0bad-2e86-4bcc-9caa-27ad95a52ccd\subagents\workflows\wf_85c44fad-7db\journal.jsonl`
(the agent labelled "finish-notes-v2"); read that before the dispatch. If the file is missing or
unfinished, the first notes (`task-15-dispatch-notes.md`) and the rulings are what there is, and the
second notes are made again in whatever way I have chosen to spend. The small task for the table
lands first, so the second notes' line numbers into `tests/test_look.py` and
`core/gui/widgets/rule_table.py` will have moved: the notes tell the implementer to find every place
by its words.

**Tasks 16 to 18** have their briefs in the ledger and nothing else: the last chat stopped the first
reading of Task 16's brief when I asked about the token usage. The carry list and the ledger say
what each must be told. **Task 19**, the stage's exit, is as the prompt of 10 October has it; tell
me before the slow set starts.

**How a task has been run, and what to keep whatever I choose to spend.**

- One implementer at a time, on Opus, from a brief that somebody has checked against the code as it
  is now. The plan's briefs are stale: every one so far needed between eight and eleven corrections
  before it could be built from, and the guard on drawing code (every method the toolkit calls while
  a window is drawn carries it) is in none of them.
- Tests first, each seen to fail for its stated reason; for a tests-only change, each new assertion
  seen to fail once by a one-line fault typed in and typed back.
- When the implementer reports: the listing of the real `Artifacts/`, then the gate in the
  background (`gate.sh fast <label>` in the scratch folder), and the review at the same moment. The
  run of the three tests marked for the real screen (`gate.sh display <label>`) follows every gate on
  a head that changes the product or the helper that reads the process's state. A fix is a commit of
  its own, re-reviewed, with a fresh gate.
- A review finding that the plan's text or the approved sample itself dictates is not the
  implementer's slip: rule it, or bring it to me with a picture if it changes how something looks.
- Every ruling goes into the ledger with what it costs if it is wrong; I am given the whole list at
  the stage's exit, with the five places where the drawing code now differs in form from the
  approved sample's (three `try`/`finally`, the whole-pixel fill made with the smoothing off, the
  hatching kept inside its caller's clip) and the measure that no picture the sample draws moved.

**How much to spend (the last chat's proposal; ask me).**

What it cost so far: about 4 to 6 million tokens for each task's preparation (the brief read by
seven agents while the task before was built, then second notes by four with a stand-in of the new
module, then a brief for a small front by four), 1 to 2 million for a review by three readers with
skeptics, and 0.2 to 0.4 million for the implementer. All on Opus, with Ultracode on.

- **The leaner way (proposed).** Ultracode off. One preparation pass a task, made after the task
  before has landed: one reader on Sonnet holds the brief against the code and writes the notes; one
  skeptic on Opus builds a stand-in and runs the added tests only where the dispatch adds tests the
  plan did not have. The implementer on Opus. One reviewer on Opus a task, as the plan's own loop
  has it; three readers only for a task that draws on the screen and for the stage's last review. A
  short re-review of a fix on Sonnet. A minor finding goes to the stage's last review unless a later
  task builds on it. This should cost about a quarter of what the last chat spent on a task. Its
  price: fewer independent eyes. Of what the three-reader reviews found, two faults in tests the plan
  dictated and the two questions on the table might have been found later, or not at all.
- **As before.** A workflow at every substantive step, every agent on Opus.
- **In between.** As the leaner way, but with the three-reader review kept for every task.

A new chat also saves by itself: each step of a long chat reads the whole chat again.

**Still mine to hear at the stage's exit, new since 10 October** (the ledger has each): the pale
ground a tab you can open takes under the pointer also shows on the figure tabs and on the plotting
library's own "Figure options" box; a column of times in the new table is ragged where the approved
picture has it even, until a column can ask for digits of equal width (for the plan of the stage
that first mounts a table, with two smaller things about tables); the time axis comes out with no
tick and no label for some runs (5 to 15 minutes that cross no quarter hour; 2 to 6 hours that cross
none of 00:00, 06:00, 12:00, 18:00), as the approved rule gives it, to be shown to me when the axis
is first mounted; the greyed main button of the first rebuilt page is shown to me (stage 4).

Do not start the retrain, or any training of real size, in this chat. Everything else in the prompt
of 10 October stands.
