# Prompt for the next chat — the window redesign: go on with stage 1, from Task 16

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the chat of 12 October 2026. What in it is not the owner's own words:

- The owner's own words of 12 October are two: the choice "In between" for how much to spend, and
  "show me the two pictures when the gate is green". Everything else is that chat's record and its
  way of handing over, to be changed if it is not wanted.
- The pasted part does not name the multi-agent mode of the app by its name, on purpose: the app
  switches that mode on for a message that holds the word, as it did for the prompt of 11 October.
- Every count, time and measurement is that chat's record, not something the owner measured.

---

PRISM — the redesign of the window: go on with stage 1, from Task 16.

**Where things stand.** Stage 1 of the redesign, the look's foundation, is being built from the plan
I approved on 7 October 2026. Since the chat of 11 October these are done, each with its tests
first, a review by three readers and a one-process fast gate: my two answers on the table (built as
Task 14q, with one fix round) and Task 15, the time axis (with one fix round). The last commit that
touched code is `6da8215`; the last fast gate ran on it: 1360 passed, 3 skipped, 5 deselected.
Tasks 16 to 19 are left. Nothing has been pushed since `cb2b0de`: I push when I choose. Nothing was
running when this was written; check that the graphics card is free and no python is alive before
any work.

**How much to spend is decided: do not ask again.** On 12 October I chose "in between": no
workflow at every step, and do not run a workflow unless I ask for one in my own words; one
preparation pass a task, made after the task before has landed (one reader on Sonnet holds the
brief against the code and writes the notes; then one skeptic on Opus builds a stand-in in a
scratch copy and runs everything, because every dispatch so far has added tests the plan did not
have); the implementer on Opus; the review by THREE readers for every task (the last chat gave the
first seat, what was asked against what was built, to Sonnet and the two seats that draw and that
try faults to Opus), with one skeptic on Opus set on the Important findings together; a fix re-read
on Sonnet. Give every agent its model by name. What it cost in the last chat: about 3.5 million
tokens for the table task from its brief to its close, about 2.5 million for Task 15's build,
review and fix round, about 0.9 million to prepare Task 16. Read the usage card when you start and
tell me where the weekly bar stands.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND
   BUILD", only. The paragraph "THE CHAT OF 12 OCTOBER" is the last chat's record.
2. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/` (gitignored): of `progress.md` the
   first 47 lines; then everything from the line that begins "- NEW CHAT 2026-10-12" to the end of
   that section (every ruling the last chat made, what each review found, what is carried to which
   later task or stage); then the section "Deviations from the design" (rows 42 to 44 are new) and
   the last ten lines of "Gates". It is the record and the recovery map: trust it and `git log`
   over anything remembered. Never add a line to it above the three lines that begin "- HAND-OFF"
   (another file cites it by line number): append with the tool named below.
3. `stage-1/carry-to-tasks-13-19.md` in the ledger: its sections for Tasks 16 to 19 and "Every
   later task of the stage".
4. The plan's head, `stage-1/briefs/plan-head.md` in the ledger (cut again on 12 October: rows 1
   to 44): "Global Constraints", "How to run this plan", and the last three rows of its table.
5. The prompts of 10 and 11 October, beside this file, for everything this one does not repeat.

**Task 16, the page pieces, is prepared: dispatch it.** Its brief is the plan's section
(`stage-1/briefs/task-16-brief.md`, not edited); its notes
(`stage-1/briefs/task-16-dispatch-notes.md`, 760 lines) were written against the code and then RUN:
a skeptic built both modules as a stand-in, ran the brief's fourteen tests (all pass as dictated),
wrote one more test and eight one-run faults, and walked the finished notes on a fresh copy. The
rulings are in the ledger (search `progress.md` for "Task 16: Ruling"). What the dispatch must
say: the head it starts from, the first count of the look's test file on it (`145 passed, 1
skipped` at `6da8215`) and d = 1 (the notes were written at `d444fc4`, and that file has gained one
test item since; the notes' section 3 has the rule); that the three imports the file's head lacks
and the four names the block adds at the margin were searched again on that head; that the brief's
Step 7, the one run on the real platform with hidden windows, is the implementer's own, and that no
long run of mine is live; that a typed backslash-escape is checked in the file's bytes; that
scratch files go under the scratch folder alone. The gate to expect is 1375 passed, 3 skipped,
5 deselected.

**Tasks 17 and 18 have their briefs and nothing else.** One preparation pass each, after the task
before has landed. The carry list and the ledger say what each must be told; for Task 17 one thing
is not ruled and must be before its notes are written: where a number field asks for digits of
equal width (by the field's class at build, which changes every number field of today's forms, or
by each rebuilt page). Task 19, the stage's exit, is as the prompt of 10 October has it, with the
stage's last review by three readers BEFORE its Step 3, over everything from `24368c7` to the head,
sorting the deferred minors (the ledger's lines that hold "minor (deferred"). Tell me before the
slow set starts.

**How a task was run on 12 October, and what to keep.**

- One implementer at a time, on Opus, from the brief and the run notes. Tests first, each seen to
  fail for its stated reason; every assertion a dispatch adds beyond its brief seen red once by a
  one-line fault typed in and typed back.
- When the implementer reports: the listing of the real `Artifacts/`, the review package, then the
  gate in the background, then the three readers, in that order (the gate refuses to start while a
  python is alive). The three tests marked for the real screen follow the gate at the first moment
  no python is alive. Readers who draw beside a gate slow it from 18 to 24 minutes: such a time is
  never a quiet-machine time.
- While an implementer is at work, touch no tracked file and start no python: its own checks of a
  clean tree and of no python alive would stop it. Never commit while a gate runs. The state file
  and the documents are committed between a task's close and the next dispatch.
- A preparer or a skeptic runs pytest in a scratch COPY of the tracked tree (the two conftest files
  work there as they are), never in the repository, and never two pytest processes at once on the
  machine.
- A fix round: ONE skeptic on Opus tries to refute all the Important findings, then writes the
  fix's brief and walks it on a fresh copy; the first implementer is resumed with it; a scoped
  re-read on Sonnet and a fresh gate start together. A finding that the plan's text, the approved
  sample or one of your own rulings dictates is ruled, or brought to me with a picture if it
  changes how something looks. Minors whose cure is a line or two already written and run go into
  a fix round that is owed anyway; the others go to the stage's last review.
- Every ruling goes into the ledger with what it costs if it is wrong; I am given the whole list at
  the stage's exit.
- The tools are in `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`, with text copies in the
  ledger's `stage-1/tools/`: `gate.sh <kind> <label>`; `ground.ps1 <before|after> <label>` (the
  listing of the real `Artifacts/`, compared after the gate, with the settings file's hash and the
  python processes alive); `ledger_append.py <stage1|deviations|gates> <text file>` (appends at the
  END of a section of the ledger); `insert_rows_after.py` (a deviation row into the design and the
  plan); `task01_test_names.py --check` for the carried-over list.

**Mine to hear at the stage's exit, new since 11 October** (the ledger has each, with its numbers;
the earlier list in the prompt of 11 October still stands):

- Two questions on the time axis, each asked alone, with pictures and a recommendation, so that a
  remedy I choose is built before stage 3 mounts the axis. (1) An axis with no tick and no label:
  about four runs in ten of 5 to 15 minutes and of 2 to 6 hours, as the approved rule gives it; the
  remedy on offer is the next finer unit when no tick falls on the axis. (2) A short run that turns
  late keeps almost none of its bar, because the axis grows by a whole day (a 40-minute run one
  second late shows 28 px of 1,038); the remedy on offer is to grow by a share of the planned
  length, a day at most. The reader's pictures are kept in the ledger,
  `stage-1/reviews/task-15-review-two-pics/`; make plain ones for me from them.
- For stage 3's design, with a picture: the line under the axis cannot be narrower than its two
  sentences, which for a six-day fit are wider than the page in the smallest window on 151 of 365
  start dates, and more often at larger text sizes. And: two ends handed over crossed now read as
  one firm end (row 44); what the page should say of a fit that cannot end before its expected end
  is that stage's.
- The room after a table's last column grows with the text size (the last chat's ruling, row 42).
  Its cost: in a FIXED last column the room is taken from the words; the approved health table's
  last column of 56 cuts "Fine" to "Fi…" at every text size above about 104 %, and never fitted the
  design's longer words for it. That column is widened or fitted when the table is mounted. The two
  other forms the room could take (6 px at every text size; no room in a table that never shows a
  ring) are one line each.
- What my second answer on the table looks like at its worst (row 43): in a very narrow column
  "10…" stands for 10.4 and for 101.3 alike, and the whole value is shown nowhere. No approved
  column is that narrow; the readers suggest the whole words under the pointer when a table is
  first mounted.
- From Task 16, as its brief dictates: a message row with no marker starts its sentence 10 px in,
  not under the other rows' sentences. It will be in that task's pictures.

Do not start the retrain, or any training of real size, in this chat. Everything else in the
prompts of 10 and 11 October stands.
