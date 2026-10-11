# Prompt for the next chat — the window redesign: go on with stage 1, from Task 13

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice. These things in it are not the owner's own words, and are there to be changed if they
are not wanted:

- Under "Skills and plug-ins", only the first sentence is the owner's ("I want to use all relevant
  skills and plug-ins"). Which skills count as relevant, what each is for, which three do not fit
  this repository's rules, that the execution skill's own scripts are not run, and the point on
  workflows are the last chat's reading of that sentence.
- In the same section, the limits on the Qt plugin, on claude-mem and on the three design plugins
  are carried over from the prompt of 6 October, where they stood in the owner's voice. That the
  design plugins' methods are "applied by hand", that nothing of theirs is written into the
  repository and that impeccable's program is never run is also the owner's decision U20 in the
  locked record, and this prompt KEEPS it in those words. One sentence is the last chat's, the one
  that begins "So for these three": it says how U20 sits beside "all relevant skills and plug-ins"
  (what a plugin says about design may be read, from its files or by loading its skill, and is
  applied by hand; nothing it says to run or to write is done). U20 and the design say only "by
  hand" and that no program of theirs was run; whether a skill of theirs was ever loaded is not on
  record. If the owner wants it stricter (their skills never loaded) or looser, that sentence is
  the one to change. The Qt sentence no longer says the plugin is "installed but switched off",
  because the list of installed plugins did not settle whether it still is.
- Under "How to work", the sentence that agrees to multi-agent workflows is carried over from the
  prompt of 6 October; the last chat ran with Ultracode switched on by the owner. The third point
  there (small things are decided and recorded as rulings, not asked) and, under "What to do", the
  list of rulings handed over at the stage's exit are the execution skill's rule, by which the last
  chat worked; the owner has not said them.
- Under "The first thing", the remedy offered for the locked tabs and the way an answer is ruled
  are those of the plan the owner approved (its Task 10p, Steps 8 to 10), shortened by the last
  chat. The three things the questions are about are worded as the last chat's readers found
  them, and the order of asking (the three questions, then anything else) is the last chat's.
- Under "What to do", the order (finish stage 1, then plan stage 2 and wait for approval) is the
  order of the prompt of 6 October.
- The list under "Read first", what it leaves out of the list of 6 October, the carry list
  (`stage-1/carry-to-tasks-13-19.md`) and the instruction that every dispatch carries what it names
  are the last chat's way of handing over.
- Every count, time and measurement is the last chat's record, not something the owner measured.

---

PRISM — the redesign of the window: go on with stage 1, from Task 13.

**Where things stand.** Stage 1 of the redesign, the look's foundation, is being built from the plan
I approved on 7 October 2026. Its Tasks 0 to 12 are done, and so are two tasks the plan did not
have: one fixed the faults the first pictures of the new look showed; the other took the long waits
out of the window's repaints while a run draws figures (a change of page or tab, a resize, the
pointer and the keyboard are still slowed while a figure is drawn, about half as much as before).
Tasks 13 to 19 are left. The window now takes its colours, its lettering, its style sheet, its tick
boxes and its focus ring from one table, so it already looks different from the window of a week
ago. One of the design's twelve engine changes has landed: E12, the plotting code's line colours.
The last commit that touched code is `d78c923`; the last fast gate ran on it: 1297 passed,
3 skipped, 5 deselected, in 16 minutes. I have pushed everything up to `cb2b0de`; the commit that
adds this prompt is on top of it. The order of work is unchanged: this redesign; then, in another
chat, the audit of the whole code; then the retrain, from the window. Nothing was running when this
was written; check that the graphics card is free before any work on it.

**The first thing: I have not answered about the pictures.** On 7 October I was shown pictures of
every screen in the new look, before and now, and asked for my remarks: anything I want changed, or
"fine as it is", and, of the same screens at a larger text size, what I can and cannot live with
until each page is rebuilt. Three questions were put with them. I have not answered, and Task 13
waits for my answer. Ask me again: first the three questions, one at a time, in plain words, each
with the pictures that show it; then whether there is anything else I want changed. The pictures
are kept in the ledger under `stage-1/pictures/`: twelve before-and-now pairs in `for-the-owner/`, a
page of all 47 pairs, `compare-before-now.html`, and the same screens at a text size of 150 % in
`now-text150/`. The pair of the Posterior tab, `for-the-owner/07-inference-posterior-light.png` and
`-dark.png`, shows all three. They are what the last chat's readers found and asked me about; I
have not said whether I see them so:

- whether I can still tell a locked tab of Parameter Inference from an open one at a glance; if
  not, the remedy on offer is to set the open tabs' words in the full text colour, at the cost that
  an open tab's word no longer changes under the pointer; the tab bar itself leaves with stage 2;
- whether hint text in an empty box reads to me as if it were a value;
- whether a greyed main button looks to me as if it could be pressed when it stands alone.

How an answer is ruled and written down is in the plan, Task 10p, Steps 9 and 10: each answer goes
into the ledger in my own words with its ruling; a change to the look as built is a small task of
its own, the test it moves rewritten first, with its review and its gate, before Task 13 unless the
ruling says why it can wait, and the pictures of the screens it changes are made again and shown to
me; what only a later stage can give is written down for that stage and does not hold Task 13.
Task 13 starts when I have answered, every answer is ruled and written down, and every change ruled
"before Task 13" is complete, its review clean and its gate green.

None of the pictures of the window I was shown on 7 October has a focus ring, a message box, a
filled list or a figure in it. No picture of the ring has been shown to me, and the trial pictures
of it are not to be sent: four of them show a trial variant, not the ring as built. The ring is
mine to judge, by opening PRISM myself, which waited for the task on the repaints, now done: say so
when you ask, and that PRISM now opens dark on this computer.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND
   BUILD", only. The first paragraph says how the ones after it are ordered: the hand-off, then the
   build in four paragraphs, each as it was written on its date. The file is about 1,900 lines, some
   of them very long; search the rest when you need it, do not read it through.
2. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/` (gitignored, so no git command finds
   it): of `progress.md` the first 47 lines (how a task is run, my answers so far, what is open with
   me) and the last three sections ("Engine changes landed", "Deviations from the design", "Gates");
   then `stage-1/carry-to-tasks-13-19.md`, all of it. That file says what each remaining task must
   be told that its brief does not know, and where the way a task is run is written out in full.
   `progress.md` is the record and the recovery map: trust it and `git log` over anything
   remembered, and read a task's entries there when you reach that task.
3. The plan, `docs/superpowers/plans/2026-10-06-window-redesign-stage-1.md`: its head, down to the
   first task. It is about 10,000 lines; each task's section is cut out for its implementer as
   `stage-1/briefs/task-NN-brief.md` in the ledger, and you read a section when you dispatch it.
4. The design, `docs/superpowers/specs/2026-10-05-window-redesign-design.md`: section 8, the
   deviations recorded while planning and building (38 rows, the same rows as in the plan's head).
   Of its eleven parts, part 4 (the look) and part 13 (the tests, the stages, the walkthrough) as a
   task needs them; have agents read what a task needs.
5. `docs/superpowers/specs/2026-10-05-window-redesign-assets/decisions-as-approved.md`, then
   `rulings-after-the-drafts.md` — the locked records of what I decided. Where they differ, the
   later wins: the rulings over the decisions, and within the rulings part D over C over B.
6. `docs/superpowers/prompts/2026-10-06-window-redesign-plan-and-build.md`, the prompt the last chat
   started from. Its two opening questions are answered (seven line colours; the design for saving
   the fit is made between stages 2 and 3). Its "Where things stand" and its reading list are
   replaced by the ones here; the experiment it says stage 1 owes, the text-size test, was run with
   me on 7 October; and stage 1 is built in the plan's order. Where that prompt and this one
   differ, this one wins. The rest of it still holds, and what matters most is repeated below.

The plan's briefs were written before Tasks 6 to 12 were built. Check a brief's counts, line numbers
and expectations against the code as it is now before you dispatch it.

**What to do.**

- Ask me about the pictures. Then build Tasks 13 to 18 in the plan's order, one at a time, each
  dispatch carrying what the carry list names for it. Task 13 begins with a small tests-only commit
  that the last review ruled.
- Then Task 19, the stage's exit, as its brief lists it: the fast gate of record on a quiet machine,
  the tests marked for the real screen, pictures of every screen in light and dark for me, the slow
  set. Tell me before the slow set starts: it has the card to itself for well over an hour. At the
  exit give me, in plain words, the list of everything you and the last chat ruled without me, with
  what each costs if it is wrong.
- Then write the plan for stage 2 from the design's parts as they stand, and give it to me to
  approve. Build it only when I have. After stage 2 the build pauses for the separate design for
  saving the fit as it goes, which I decided is made between stages 2 and 3.
- Every task: its tests written or rewritten first and seen to fail for the stated reason; then the
  code; then a review and ONE one-process fast gate, started together. A task is complete when its
  review is clean and its gate is green on the same commit. A fix is a commit of its own, reviewed
  again, with a fresh gate.
- The engine changes are the twelve of the design, each in the stage part 13 gives it, and nothing
  else; E12 has landed, none other. If a stage turns out to need more of the engine, stop and tell
  me. List every engine change in `docs/STATE.md` as it lands: the audit chat will read that list
  first.
- When what is built departs from the design, say so, ask me if it touches something I decided, and
  record it in the design's section 8, the plan's head and the ledger's table as a deviation with
  what it costs if it is wrong. The next row is 39.
- The card smoke gate in `CLAUDE.md`, with the diagnostic card, is owed once, at the end of stage 3.
  Tell me before you run it.
- How it looks and feels on my screen is mine to judge. The rows I must run on a real screen are in
  the design's part 13.4: add them to `docs/checklists/display-walkthrough.md` as a new group when
  their stage lands, and never edit an earlier row. Stage 1 writes none (row 21 of the design's
  section 8): its two rows come with stage 3's group. No row may ask for a training of real size or
  touch the records kept from the stopped retrain.
- To look at the window yourself, render a screen to an image with the window hidden and read the
  image, before and after a change, on a scratch settings file and a scratch record store. A window
  built on the real ones reads my remembered selections and overwrites them when it closes. The
  last chat's script does it on the real display with the window never shown, because with no
  display the system's typefaces are missing; it builds the real window, which takes a little
  memory on the graphics card, so a picture run never stands beside a test run or a long run of
  mine. That script, the gate script and the other tools are in
  `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`, with text copies in the ledger's
  `stage-1/tools/` in case Windows has cleared that folder.
- Keep the ledger as it is kept, and leave `docs/STATE.md` at the end of every session so that a new
  chat can start from it and from this prompt.

**Skills and plug-ins.** I want to use all relevant skills and plug-ins. Load a skill before the
work it covers.

- superpowers. `subagent-driven-development` runs a plan task by task: one implementer at a time, a
  review after each task, the ledger. Stage 1 has been built with it; go on with it.
  `test-driven-development` for every task's tests. `requesting-code-review` and
  `receiving-code-review` for each task's review and what it finds.
  `verification-before-completion` before anything is called done. `systematic-debugging` when a
  test, a gate or the window fails for a reason not yet known. `dispatching-parallel-agents` for
  reading that can be shared out. `writing-plans` for stage 2's plan. `brainstorming` before
  anything new is designed: the design for saving the fit, and any remedy my answers call for.
  Three of its skills do not fit this repository's rules and are not used: `using-git-worktrees`
  and `finishing-a-development-branch` (the work is on local `main`, with no branch and no
  worktree, and I push), and `executing-plans` (the plan is run by agents, not inline). Where a
  skill and a rule of mine differ, my rule wins. Two cases: the ledger is not deleted when a
  stage's last review is clean, because it is the whole redesign's; and the ledger named above IS
  the workspace, so the skill's own scripts for finding a workspace, cutting a brief and packaging
  a review are not run as they stand (they would start a second, empty workspace for this plan and
  count stage 1 as not begun). The carry list says what is used in their place.
- Workflows (the `workflow-authoring` skill). Run a workflow for the heavy steps, as the last chat
  did: the review of a task that changes how the whole window is drawn (Tasks 9, 10 and the task on
  the repaints had several readers, with skeptics who tried to refute each finding; the other
  reviews had one reviewer, as the plan's loop says), a probe before a risky change, several
  readers for a set of pictures, the checks of a plan before I see it. When Ultracode is on, follow
  what it asks.
- The three design plugins (impeccable, ui-ux-pro-max, frontend-design): their methods are applied
  by hand where they help, as the design chat did. So for these three, using them means reading
  what they say about design and applying it yourself; whatever one of them says to run or to
  write is not done (impeccable's skill opens by telling you to run its program, which would also
  write a file into the repository: leave that step out, and say once that you did). Nothing of
  theirs is written into the repository, impeccable's program is never run, and their labels and
  file names stay out of the code, the tests and the guide. None of them knows Qt.
- The Qt Development Skills plugin: do not enable it and do not use any part of it. Using it means
  accepting Qt's AI Services terms, and I do not accept them. To check what a widget, a layout or a
  style sheet can actually do, read Qt's public documentation or the PySide6 installed here, and try
  it in a small scratch script before you build on it.
- claude-mem and claude-mem-cowork are installed but cannot work on this machine until I set them
  up. Do not rely on them and do not set them up yourself. If a hook error from them appears on
  each prompt, ignore it.

**How to work.**

- I agree to multi-agent workflows in this chat. Parallel agents read, plan and review, and a second
  reader checks findings and plans; only one of them at a time edits the code, nobody edits while a
  test suite is running, there are never two pytest processes at once, and you, not an agent, start
  and wait for the long gates.
- Ask me before you assume. One question at a time, in plain language, with a short background first
  and no code names, and your recommendation with it. Show me a picture when a picture is clearer
  than a description.
- A plan in progress does not wait for me on small things: decide them, write each decision in the
  ledger as a ruling with what it costs if it is wrong, and go on. Stop and ask for what I decided,
  for what cannot be undone, and for anything under "What must not change".

**What must not change without my say-so.**

- The science: the models, the simulation, the features, the training and calibration methods and
  every setting in `core/config.py`. The design adds one line to that file, which names the folder
  the plans live in; I was told and approved it, and nothing else there changes. The load-check
  audit on the open list comes before any other change to `core/config.py` or to the features; if
  the build needs one, stop and tell me.
- What a setting means. The simulation cache's identity must not change by accident: a change to it
  makes the finished simulations unusable. Tell me before any change that does this.
- The rules in `CLAUDE.md`: every knob reaches its stage as an argument; a refusal carries a field
  key and the window names the control from its own table; a consent, a seed, a record name and a
  note are never remembered; nothing under `core/`, `tests/` or `docs/guide/` cites the working
  documents or their labels — no decision, engine-change or finding id, no part or stage number,
  no ruling, no plugin's name. Two of its rules change WITH this redesign, and only when the code
  that changes them lands: what the window remembers (the memory limit for the graphics card
  becomes remembered, per computer, in Settings) and the launch line. The replacement wording is in
  the design's parts 7.3.3 and 13.3.4 for me to approve. Stage 1 also leaves two counts in
  `CLAUDE.md` out of date (the test suites; the standing guards of the tests); the corrected
  wording is put to me at the stage's exit. Do not edit `CLAUDE.md` before I have approved.
- On disk, and I have not asked for any of it to be deleted: the prior `retrain_prior`, the complete
  simulation cache `Artifacts/simulations/eda5c2a1e468`, the two probe records under
  `Artifacts/diagnostics/` (`retrain_band` and `retrain_mask`), anything else already in
  `Artifacts/`, and the console capture `C:\Users\J\prism-retrain-console.log`. Read them freely;
  ask me before anything deletes, renames, rewrites or trains on them, and write no row for me to
  run on screen that would.

Do not start the retrain, or any training of real size, in this chat. The tiny trainings inside the
test suites, the card smoke gate once I have been told, and the small ones in the rows I run on
screen are fine.

**Standing rules.** Work on the local `main` branch; one commit for each logical step, a short
subject, never amended, ending with the Co-Authored-By line the harness provides; I push. Never two
pytest processes at once, and never edit a source file while a suite is running. A run on the
graphics card has the card to itself. Undo an edit with `git stash`, never with
`git checkout -- <file>`. Scratch files go under `C:\Users\J\AppData\Local\Temp\`, and no `.py` file
goes under `.superpowers/` or under `docs/` (the source scan fails on any folder that holds one and
is not on its list). Update `docs/STATE.md` at the end of every session.

**Done means, for a stage.** I approved its plan; it is built; every task had its review and its
gate; the stage's exit as part 13 lists it has passed, with the gate's time recorded on a quiet
machine; the window is usable and its suites are green; and `docs/STATE.md` says what landed, which
engine changes, which deviations from the design, which gates were not run, and which rows are now
mine to run.

**Done means, for the redesign.** As the prompt of 6 October lists it; nothing in that list has
changed.
