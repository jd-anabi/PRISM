# Prompt for the next chat — the window redesign: the exit of stage 1 (Task 19), then the plan of stage 2

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the chat of 13 October 2026. What in it is not the owner's own words:

- The owner's own words of 13 October are three: "we will keep the pace" (stages 2 to 9 are not
  squeezed into a week), "we won't start then" (no network is trained while the build goes on), and
  "let's continue on as is". Everything else is that chat's record and its way of handing over, to
  be changed if it is not wanted.
- The pasted part does not name the multi-agent mode of the app by its name, on purpose: the app
  switches that mode on for a message that holds the word.
- Every count, time and measurement is that chat's record, not something the owner measured.

---

PRISM — the redesign of the window: the exit of stage 1, then the plan of stage 2.

**Where things stand.** Stage 1 of the redesign, the look's foundation, is BUILT: every task of the
plan I approved on 7 October 2026 that builds something is done (Tasks 0 to 18 and the tasks the
plan did not have), each with its tests first, a review and a one-process fast gate. On 13 October
the last three landed: the focus ring on a control scrolled into sight (task 10r), the shared
question (Task 18), and the stage's last review by three readers with its one fix wave. The last
commit that touched code is `f9c6899`; the last fast gate ran on it: 1407 passed, 4 skipped, 5 deselected. What is left
of the stage is Task 19, its exit, which needs me. Nothing has been pushed since `cb2b0de`: I push
when I choose. Nothing was running when this was written; check that the graphics card is free and
no python is alive before any work.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no
workflow unless I ask for one in my own words; every agent given its model by name; the review by
three readers for every task that builds. The exit is mostly your own work (gates, pictures,
documents, my questions); use readers for sets of pictures, on Sonnet where reading is enough.
What 13 October cost in background agents: about 1.9 million tokens for task 10r, 4.0 for Task 18
with its preparation, 2.8 for the stage's last review and its fix wave; the weekly bar went from
48 % to about 61 %. Read the usage card when you start and tell me where the weekly bar stands.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND
   BUILD", only. The paragraph "THE CHAT OF 13 OCTOBER" is the last chat's record.
2. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/` (gitignored): of `progress.md` the
   first 47 lines; then everything from the line that begins "- NEW CHAT 2026-10-13" to the end of
   that section; then the section "Deviations from the design" (rows 47 and 48 are new) and the
   last ten lines of "Gates". It is the record and the recovery map: trust it and `git log` over
   anything remembered. Never add a line to it above the three lines that begin "- HAND-OFF"
   (another file cites it by line number): append with `ledger_append.py`. Its lines are very long: search
   it, do not read it through.
3. Task 19's brief, `stage-1/briefs/task-19-brief.md` in the ledger (sixteen steps, yours to run:
   no implementer), and `stage-1/carry-to-tasks-13-19.md`: its section "Task 19, the stage's exit"
   (what the brief gets wrong or leaves out), "Still open with the owner", "For the audit's list"
   and "The controller's tools".
4. The stage's last review, in the ledger's `stage-1/reviews/`: `stage-1-final-review-sorting.md`
   (its sections "TELL THE OWNER", "CARRY" by stage, "AUDIT"; the whole table only when you need a
   row), and the verdict and the minors of `stage-1-final-review-product.md` and
   `stage-1-final-review-tests.md`.
5. The plan's head, `stage-1/briefs/plan-head.md` in the ledger (cut again on 13 October: rows 1 to
   48): "Global Constraints", "How to run this plan", and the last two rows of its table.
6. The prompts of 10, 11 and 12 October, beside this file, for everything this one does not repeat
   (the skills and plug-ins and their limits; what must not change without my say-so; the standing
   rules; what "done" means for a stage).

**Task 19, the exit, in the order its brief has it, with what the last chats add.**

- The fast gate of record on a QUIET machine (its Step 3). On 13 October two gates took 23 min 48 s
  and 23 min 17 s, near twice the 14 to 16 minutes of a quiet gate, with only a reader's small
  scripts beside them: the computer was busier than the build's agents explain. Ask me
  whether I am using it before you start the gate of record, and do not record a time taken beside
  anything.
- The tests marked for the real screen (four now), the pictures of every screen in light and dark
  for me (three sets), and the pictures no set has yet (the carry list names them; they need a
  scratch store that holds records).
- **Tell me before the slow set starts**: it has the card to itself for well over an hour.
- The deviations brought into line in the design, the plan's head and the ledger (48 rows; the
  brief's count is out of date).
- `docs/STATE.md`, and the two sentences of `CLAUDE.md` put to me: `CLAUDE.md` is not edited before
  I say yes. The brief's proposed sentences are one guard short (the drawing code's per-test
  fixture, `_drawing_code_raises_nothing`); count the standing guards and the suites as they are
  now before you ask.
- What the sorting found is out of date in the working documents and that the brief has no step
  for (its rows carried to "the exit").

**Mine to hear at the exit.** Ask each question alone, in plain words, with its pictures and your
recommendation; then tell me the rest, grouped, shortest first. Everything is in the ledger with
its numbers; the sorting's section "TELL THE OWNER" and the lists in the prompts of 11 and 12
October are the base. In short:

- THREE QUESTIONS, each alone: (1) the time axis with no tick and no label for about four runs in
  ten of 5 to 15 minutes and of 2 to 6 hours; (2) a short run that turns late keeps almost none of
  its bar; (3) a message that names a long path cannot wrap and is cut at the right with no sign
  (pictures in the ledger's `questions/`: `q13a` to `q13c`, and more of a question in
  `questions/q18-long-path-in-a-question/`, whose README says which two show a list as first
  dictated and not as built).
- THE FOCUS RING, which I judge by opening PRISM: what it does as measured (the carry list has the
  list); that since 13 October it is on every control the Tab key brings into sight (pairs before
  and after in `stage-1/reviews/task-10r-review-two-pics/`); that in the grid of typed bounds on
  the Prior tab its bottom side is cut on most rows until that page is rebuilt; that a tick box's
  ring runs through the edge of the help badge beside it; and what the last fix wave changed: no
  ring round a control of no size, and a press of the mouse ends the keyboard's turn (so after a
  click that gave the focus to nothing, a window that comes back to the front shows no ring on a
  button, and a list opened by keyboard and answered with the mouse gives the focus back without
  one).
- THE QUESTION (sixteen sheets made on the real platform, each form in light and dark at 100 % and
  150 % text: `stage-1/reviews/task-18-review-two-pics/sheets-for-the-owner/`): the form that asks
  for a line of text and its choices; three of the design's own pairs of answers make a question
  wider than designed (row 47): reworded shorter, or I judge the wider box; Enter on "Show the
  full error" gives the safe answer and does not open the error, as the design has it; above the
  ordinary text size a very long question is taller than a low screen; the five things the
  question does that the design does not state (row 48).
- NUMBERS IN BOXES (rows 45 and 46, and the box that says "is blank" while it shows `6,5`), TABLES
  (rows 42 and 43 and their worst cases), THE MESSAGE ROW with no marker, FIGURES (the mean and the
  second repeat share one blue: my call), and the smaller things of the earlier lists.
- THE RULINGS MADE WITHOUT ME: every entry of `progress.md` that holds the word "Ruling", in plain
  words, each with what it costs if it is wrong. I can undo any of them.

**After the exit: the plan of stage 2** (the shell: the rail, the chain of steps, Home, the
first-time opening's page, one PRISM at a time; no engine change). Written from the design's parts
as they stand, with the `writing-plans` skill, and given to me to approve; built only when I have.
What stage 1 carries to that plan is gathered in three places: the sorting's rows carried to stage
2, the product review's minors that it names for stage 2 (a page tells only what it was handed; a
handed-in length is scaled by some devices and not by others; the scan of drawing methods does not
know a hide event), and the tests review's (the window suites leave every window alive; the two
lines of the session guard against a dialog that would stall the gate, to be proved by a gate in
stage 2's first task; which colour each plotting call takes, for the first stage that edits a
plotting file). The design sizes stage 2 at about ten tasks; stage 1 was sized at eleven and came
to about thirty. After stage 2 the build pauses for the separate design for saving the fit.

**How a task was run on 13 October, and what to keep.** Everything the prompts of 12 October list,
and:

- The preparation skeptic builds the task as a stand-in against the REAL devices before the
  implementer starts. For Task 18 that found five things a person would meet that no dictated test
  saw; all five were ruled into the task with a test line each, and the build then came out as
  briefed (its review still found two more, closed in one fix round).
- When rulings change what a brief or notes say, the skeptic is resumed to fold them into ONE
  document and walk it again; an implementer is never given "the notes, and these overrides".
- When two readers measure the same thing differently, the fix round's skeptic settles it before
  anything is built (both were right, on different inputs).
- The run on the real screen after a gate is started as a loop that tries every few seconds until
  no reader's script is alive; on 13 October it got through on the first try four times of five,
  and the fifth waited about 25 minutes for the readers' scripts.
- A reader that builds PRISM's whole window does so only while no pytest is alive, and checks the
  process list before every such run.
- A `cd` inside a Bash call moves the session's working directory: end such a call back at the
  repository's root.

**What I decided on 13 October.** Stages 2 to 9 are not to be finished in a week: the design sizes
them at about 84 tasks, a task takes three to four hours and two to four million tokens at the
care I chose, and I want nothing rushed or half-built. No network of real size is trained while
the build goes on; the order I set on 5 October stands (the redesign, then the code audit, then
the retrain from the window).
The last chat told me that a small training of a few minutes, or opening PRISM, is fine at a moment
I name first, so that it does not share the card with a gate; I did not object.

Do not start the retrain, or any training of real size, in this chat. Everything else in the
prompts of 10, 11 and 12 October stands.
