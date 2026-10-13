# Prompt for the next chat — the window redesign: go on with stage 1, from task 10r

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice. These things in it are not the owner's own words, and are there to be changed if they
are not wanted:

- Everything under "Where things stand", every count and every measurement is the last chat's record.
- "How much to spend" is the owner's choice of 12 October ("in between"), carried over as it was
  said then.
- The points under "How a task was run" that are new here (an agent that needs pytest waits for the
  word; no process is left behind; second faults are not run) are the last chat's way of working.
- The prompts of 10, 11 and 12 October (beside this file; the one of 12 October is
  `2026-10-12-window-redesign-continue-stage-1.md`) still hold for everything this one does not say:
  the skills and plug-ins and their limits, how to work, what must not change without the owner's
  say-so, the standing rules, what "done" means. Where they differ from this one, this one wins.

---

PRISM — the redesign of the window: go on with stage 1, from task 10r.

**Where things stand.** Stage 1 of the redesign, the look's foundation, is being built from the plan
I approved on 7 October 2026. Since the morning of 12 October these are done, each with its tests
first, a review by three readers, one fix round and a one-process fast gate: Task 16, the page
pieces, and Task 17, the number fields and the drop-down. The last commit that touched code is
`0bdc77a`; the last fast gate ran on it: 1386 passed, 3 skipped, 5 deselected. Left: a small task the
plan did not have, 10r (the focus ring on a control that is scrolled into sight), Task 18, the
stage's last review and Task 19. Nothing has been pushed since `cb2b0de`: I push when I choose.
Nothing was running when this was written; check that the graphics card is free and no python is
alive before any work.

**How much to spend is decided: do not ask again.** On 12 October I chose "in between": no workflow
at every step, and do not run a workflow unless I ask for one in my own words; one preparation pass a
task, made after the task before has landed (one reader on Sonnet holds the brief against the code
and writes the notes; then one skeptic on Opus builds a stand-in in a scratch copy and runs
everything); the implementer on Opus; the review by THREE readers for every task (the first seat,
what was asked against what was built, on Sonnet; the two seats that draw or use and that try faults
on Opus), with one skeptic on Opus set on the Important findings together; a fix re-read on Sonnet.
Give every agent its model by name. What it cost in the last chat: about 2.2 million tokens for
Task 16 from its dispatch to its close, about 3.4 million for Task 17 with its preparation, about
0.5 million for 10r's brief; the weekly bar went from 40 to about 48 %. Read the usage card when you
start and tell me where the weekly bar stands.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND
   BUILD", only. The paragraph "THE SECOND CHAT OF 12 OCTOBER" is the last chat's record.
2. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/` (gitignored): of `progress.md` the
   first 47 lines; then everything from the line that begins "- NEW CHAT 2026-10-12, THE SECOND OF
   THE DAY" to the end of that section (every ruling the last chat made, what each review found, what
   is carried to which later task or stage); then the section "Deviations from the design" (rows 45
   and 46 are new) and the last ten lines of "Gates". It is the record and the recovery map: trust it
   and `git log` over anything remembered. Never add a line to it above the three lines that begin
   "- HAND-OFF" (another file cites it by line number): append with `ledger_append.py`. Its lines are
   very long: search it, do not read it through.
3. `stage-1/carry-to-tasks-13-19.md` in the ledger: its sections for Task 18, "Every later task of
   the stage" and "Task 19, the stage's exit".
4. The plan's head, `stage-1/briefs/plan-head.md` in the ledger (cut again on the evening of
   12 October: rows 1 to 46): "Global Constraints", "How to run this plan", and the last two rows of
   its table.
5. The prompts of 10, 11 and 12 October, beside this file, for everything this one does not repeat.

**Task 10r is prepared: dispatch it.** Task 16's review found that a control the Tab key reaches
while it lies under the fold of a scrolling area is scrolled into sight with no focus ring. A skeptic
confirmed it in PRISM's real window (no ring on 4 of 11 stops of the Config tab, 6 of 45 of the FDT
page, 4 of 24 of the Prior tab, each on a press that scrolled the form), chose a cure of three lines
in the ring's own code and wrote the task's brief, `stage-1/briefs/task-10r-brief.md`, against the
code at `d781ed8`, and walked it on a fresh copy. The rulings are in the ledger (search `progress.md`
for "Task 10r"). What the dispatch must say: the head it starts from; that Task 17's fix round landed
AFTER the brief was walked and bore one test at the foot of the look's test file, so d = 1 (the
brief has the rule): the first count is `171 passed, 1 skipped`, every count of deselected and every
whole-file count of passed is one higher than written, the file has 5,542 lines where the brief
counts 5,440, and the gate to expect is 1391 passed, 3 skipped, 5 deselected; that the names the
block adds (four tests, three helpers) and the words its runs select by were searched again on that
head, and that `core/gui/style.py` has not changed since `d781ed8`; that a typed backslash-escape is
checked in the file's bytes; that scratch files go under the scratch folder alone and no process is
left behind. For its review, the seat that draws walks the real window with the Tab key, before
against after, off-screen on a scratch settings file and an empty scratch store. 10r lands before
Task 18, whose question holds a scrolling list.

**Task 18** has its brief and nothing else. One preparation pass, after 10r has landed. The carry
list and the ledger say what it must be told; from the last chat besides: the scan of the book face
follows neither the function imported under another name nor a class built on the sentence in
another module (Task 18 adds the next place to that scan's list); the test toolkit's typed keys end
the interpreter with no word on a character outside its own table (send ASCII only); a sentence that
holds a long path cannot wrap, and stays so until I have answered the question below. **Task 19**,
the stage's exit, is as the prompt of 10 October has it, with the stage's last review by three
readers BEFORE its Step 3, over everything from `24368c7` to the head, sorting the deferred minors
(the ledger's lines that hold "minor (deferred"). Tell me before the slow set starts.

**How a task was run on 12 October, and what to keep.** Everything the prompt of 12 October lists,
and:

- One pytest at a time on the machine, also in scratch copies. A skeptic or a reader that will need
  pytest while another agent holds it is started with the word that it runs none until told: it does
  everything else first (tests called as plain functions, real keys, the draft of its brief), stops
  with "WAITING FOR PYTEST", and is resumed when pytest is free. So nobody waits idle and two never
  hold it.
- Every agent that runs scripts is told to leave no process behind and to look at the process list
  last. One hung probe, exited and still listed by Windows, stopped the gate script until it was
  ended through PowerShell's `Invoke-CimMethod ... Terminate`.
- The real-screen run after a gate is tried again every few seconds until no reader's script is
  alive; the gate script refuses cleanly meanwhile.
- A fix round's one-run faults: the least set that shows every added assertion red once, and the
  faults for added lines that are calls; a second fault that stops at a line already seen red is not
  run (ruled twice, with its cost in the ledger).
- For a task whose device is live on today's forms, the seat that draws works before against after
  in two copies and counts what differs on the real panels.
- A ledger entry is written as a text file under the scratch folder (`t16\ledger-NN-*.txt` holds the
  last chat's, in order) and appended with the tool; `ground.ps1` is run from the PowerShell tool,
  not from Bash (the execution policy refuses it there).

**Mine to hear at the stage's exit, new since the morning of 12 October** (the ledger has each, with
its numbers; the earlier lists in the prompts of 11 and 12 October still stand):

- A third question, asked alone, with pictures and a recommendation, beside the two on the time
  axis: a message that names a long path cannot wrap, because a Windows path has no place to break,
  and its row is cut at the right with no sign (with this computer's folders the program's own
  messages need 431 to 637 px, against 354 px on the run page). The remedy on offer is a place to
  break after each backslash, slash, underscore and full stop inside a long word: every real message
  is then whole, and nothing moves where no word is 40 characters long; a run with nothing to break
  at, such as a long hash, is not cured. The three pictures are in the ledger, `questions/q13a…`,
  `q13b…`, `q13c…` (true size in `questions/q13-true-size/`).
- A message row with no marker and all its later lines begin 10 px in, 26 px left of the other rows'
  sentences, as Task 16's brief dictates; the sheets that show it are in the ledger,
  `stage-1/reviews/task-16-review-two-pics/`. One number and one test line if I want it under them.
- A whole-number box on today's forms no longer stops at ten digits (deviation row 46: the stop was
  the toolkit's check, which the plan takes off). A grouped whole number keeps a zero I typed (row
  45); no form groups yet.
- With `6,5` typed into a decimal box, the click now refuses where it silently ran with 65 before;
  but on six of nine boxes the refusal says the box "is blank" while it shows `6,5` outlined, and the
  right sentence is in the tool tip alone until that form is rebuilt.
- Older than this stage and seen on the way: the Config tab's main button is written with a single
  ampersand, which the toolkit takes for a shortcut mark; the Tab key, coming round into a form that
  stands scrolled down, lands on a control out of sight and the form does not move.
- What a control that stands first in a scrolling middle looks like (no top side to its ring), and
  the carried rule that a middle never begins with a control.

Do not start the retrain, or any training of real size, in this chat. Everything else in the prompts
of 10, 11 and 12 October stands.
