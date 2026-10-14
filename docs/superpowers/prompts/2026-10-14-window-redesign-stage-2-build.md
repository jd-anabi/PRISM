# Prompt for the next chat — the window redesign: the build of stage 2 (the shell)

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the session of the evening of 14 October 2026 that wrote stage 2's plan. What in it
is not the owner's own words:

- The owner's own words of that evening are the two answers to the questions put while the plan was
  written ("Row of names now (Recommended)" and "Move them now (Recommended)") and the approval:
  "The plan is approved. Write the hand-off prompt so the build starts with Task 0 in a new chat."
  Everything else is that session's record and its way of handing over, to be changed if it is not
  wanted.
- "How much to spend", "Read first" and "Ask me" repeat the owner's prompts of 6, 10, 12 and 14
  October; "At the exit" repeats the prompt of 6 October (pictures of every screen in light and dark at
  the ends of stages 1 and 2; the gate's time recorded on a quiet machine) and the practice of the
  chat of 13 October (asking the owner whether the computer is quiet); the list "How a task is run"
  gathers what the chats of 12, 13 and 14 October learned. The standing rules and what "done" means
  are in the prompts that "Read first" item 7 names.
- Every count, time, cost and measurement is that session's record, not something the owner measured.

---

PRISM — the redesign of the window: build stage 2, the shell, from Task 0.

**Where things stand.** Stage 1, the look's foundation, is built and its exit passed (14 October 2026;
code head `066b55c` after task 19q; no file under `core/` or `tests/` changed since). On the evening of
14 October the plan of stage 2 was written, committed (`e90832e`, with the design's departures 54 to 78)
and approved by me: `docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md`. Nothing of it is
built. It has ten building tasks and two you run yourself (0, the workspace and baseline; 11, the exit);
its own guess is sixteen to twenty by the end, at 35 to 50 million tokens. While it was written I
answered two questions from sketches: Effective temperature chooses between its two pages by a row of
names ("One cell", "Sweep study") built now, and the list of my own models and the help text move out of
Settings into the rail's "Your own models" and "Help" now. Both are in the plan (departures 69 and 70).
I pushed up to `0747a4c` on 14 October; `e90832e` and the commit that adds this prompt are not pushed,
and I push when I choose. Nothing was running when this was written; check that the graphics card is
free and no python is alive before any work.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no workflow
unless I ask for one in my own words (this replaces the agreement to workflows in the prompts of 6 and
10 October); every agent given its model by name, Sonnet where reading is enough; one preparation pass
a task (the plan has a first reader on Sonnet only for Tasks 5 and 6, then a skeptic on Opus who builds
the stand-in); the implementer on Opus; the review by three readers for every task that builds, one
skeptic on the Important findings, a fix re-read on Sonnet. Read the usage card when you start and tell
me where the weekly bar stands (it read 66 % when the planning session began and 68 % when the plan was
handed over; the week resets on the evening of 17 October).

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND BUILD",
   only. The paragraph "THE PLAN OF STAGE 2" is the last record.
2. The plan, `docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md`: its head whole (for the
   owner, Global Constraints, Review Focus, File Map, the registry, the standing checks, How to run this
   plan, the departures), then Task 0 and Task 1; each later task when its turn comes. Stage 1's plan
   (`docs/superpowers/plans/2026-10-06-window-redesign-stage-1.md`; its head as cut for implementers is
   `.superpowers/sdd/2026-10-06-window-redesign/stage-1/briefs/plan-head.md`) holds the registry of every
   name stage 1 built and its fourteen standing checks: the new plan names only what it adds. The plan's
   banner offers `superpowers:subagent-driven-development` or `executing-plans`: use the first; the
   prompt of 10 October says which skills are used and which are not.
3. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/` (gitignored): of `progress.md` the first 47
   lines, then the section "Stage 2 — the shell (plan)" whole, then "Deviations from the design" (rows 1
   to 78; 54 to 78 are stage 2's) and the last ten lines of "Gates". It is the record and the recovery
   map: trust it and `git log` over anything remembered. Never add a line above the three lines that
   begin "- HAND-OFF". Its lines are very long: search it, do not read it through.
4. What the plan was written from, in `stage-2/plan-drafts/`: the three readers' reports (the shell's
   code at the head, the tests that hold it, the look and the parts), the lock's scratch tests
   (`lockprobe-2026-10-14.out`; more runs than the design's three), the cross-check of the plan and the
   check of this prompt. A task's preparation reads the parts of them it needs; nobody reads them all.
5. What I chose from: the sketches of 14 October, `.superpowers/sdd/2026-10-06-window-redesign/questions/`
   `q20a-1-row-of-page-names-light.png` (the row of names I chose), `q20b-1-your-own-models-moved-light.png`,
   `q20b-1-help-moved-light.png`, `q20b-1-settings-after-light.png` (the move I chose), with
   `stage-2/plan-drafts/sketches-report.md` on where a sketch differs from the real look; and the built
   sample of the chosen look, `docs/superpowers/specs/2026-10-05-window-redesign-assets/toolkit-sample.py.txt`,
   which Task 3 ports from.
6. The design the plan cites: `docs/superpowers/specs/2026-10-05-window-redesign-design.md` (its section
   8, the departures, wins over the parts where they differ) and its parts, chiefly `03-shell.md` and
   `13-tests-stages-docs.md`, with what `04-look.md`, `05-anatomy.md` and `10-records-and-parts.md` give the
   shell; the locked records `…-assets/decisions-as-approved.md` and `…-assets/rulings-after-the-drafts.md`
   (the later wins). Have readers read what a task needs; read yourself what you rule on.
7. The prompts of 6, 10, 11, 12, 13 and 14 October beside this file, for everything this one does not
   repeat: the skills and plug-ins and their limits, what must not change without my say-so, the
   standing rules, what "done" means for a stage. Where they differ, the later wins, and this one wins
   over all of them.

**Tools.** All in `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`, text copies in the ledger's
`stage-1/tools/` (Windows may empty the scratch folder; each copy's first lines say what it does):
`gate.sh` (the gates), `ground.ps1` (the record folder listed before and after a gate; run it from the
PowerShell tool, not from Bash), `task01_test_names.py` (the carried-over list's check),
`ledger_append.py` (its mode `stage1` appends at the end of the last section above "## Engine changes
landed", which is now stage 2's; `deviations` after the last row of the table), `insert_rows_after.py`,
`pictures.py`, `compare_page.py`. `gate.sh` and `ground.ps1` both write into `stage-1/gates/` until Task 0
teaches them stage 2's folder; Task 0 also makes the scratch folder `s2\` and the ledger's folders
`stage-2/{briefs,reports,reviews,gates,pictures,exit}` (only `plan-drafts/` and `tools/` exist).

**What to do.** Build stage 2 from Task 0, in the plan's order (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
one task at a time, each with its review and its gate, as "How to run this plan" says. Task 0 and Task 11
are yours. After Task 11, the exit, the build pauses for the separate design for saving the fit, as I
decided on 6 October: stage 3 is planned on my answer to it.

**Ask me** what is mine: what I decided or what the design leaves to me, what cannot be undone, and
anything under "What must not change" in the earlier prompts; one question at a time, in plain words,
with a picture and your recommendation. Decide the small things yourself, write each as a ruling with
its cost, and give me the list of them at the exit (as `stage-1/exit/rulings-for-the-owner.md` was):
I may undo any of it.

**At the exit:** pictures of every place in light and dark, as the plan's Task 11 lists them; the fast
gate of record taken when I have said the computer is quiet (ask me first), its time recorded; the
tests that need a real screen run with hidden windows only.

**How a task is run** (what the last chats learned; the plan's "How to run this plan" has the loop):

- Before a task is dispatched, one skeptic (Opus) builds it as a stand-in in a scratch copy of the tree,
  against the REAL devices and today's real panels, runs the section's tests (each seen red for its
  reason) and the one-run faults, and writes ONE brief. What its trial run finds that a person would meet
  is ruled, written into the brief with a test line, and the brief walked again; an implementer is never
  given "the brief, and these overrides".
- On the implementer's DONE: no python alive, the tree clean, the real `Artifacts/` listed; then the one-
  process fast gate in the background AND the three readers at the same moment (seat one Sonnet, the
  brief against what was built; seat two Opus, the product before against after on the real devices;
  seat three Opus, the tests and the faults they see). A fix round is written and walked by a skeptic,
  built as its own commit, re-read by one reader, with a fresh gate.
- One pytest at a time on the machine, scratch copies included. An agent that will need pytest while
  another holds it does the rest first and waits until it is told. Every agent that runs scripts leaves
  no process behind and reads the process list before any run that builds PRISM's whole window or
  imports torch. A reader that builds the whole window does so only while no pytest is alive.
- The run on the real screen (`QT_QPA_PLATFORM=windows pytest -m display`) comes after the gate of
  Task 10 and of any task that adds or changes a display-marked test, and at the exit; start it as a
  loop that tries every few seconds until no reader's script is alive.
- A picture script never shows a dialog, even hidden: it renders it. Pictures for a question can be
  sketches composed from the exit's pictures (the two sets of 14 October cost a helper on Sonnet about
  0.15 million tokens); a remedy to choose between is better drawn in the real window, hidden, on a
  scratch settings file and an empty scratch store.
- A `cd` inside a Bash call moves the session's working directory: end such a call back at the
  repository's root.
- A ledger entry is written as a text file in the scratch folder and appended with the tool.
- Earlier controlling chats stood at 50 to 62 % of their room after two or three tasks closed, and the
  planning session at about 65 %: hand over to a new chat, with STATE, the ledger and a prompt like
  this one, after about three tasks or at two thirds of the room, whichever comes first.

**Facts the plan stands on, measured on 14 October** (the ledger has each): a lock left by a PRISM that
died is taken at once by the toolkit's own check; a lock whose holder lives cannot be taken or removed;
a holder whose main thread is busy accepts a connection at once but may lose a word written to it, so
the open PRISM acts on the connection. The session guard's two lines against the older spelling of a
dialog's blocking call were never applied in stage 1 (Task 1 applies them, and its gate is their proof).
Two tests, not one, build the whole program, both through `whole_program()`. The builders' scan is
closed over `main_window.py`, `screens/` and `panels/`: every public class with a base defined there is
named in it, or taken out of it, in the commit that adds or removes the class.

**Mine to hear, still open:** at the exit, the words that still name the old places (log lines, a few
dialogs, the help text; the plan's Task 11 Step 5), each with where it goes; at the exit too, the
corrected counts in `CLAUDE.md` (the suites, two more; the session-wide guards and per-test checks, which
Tasks 1, 4 and 10 add to), put to me and not edited before I say yes; the replacement wording of
`CLAUDE.md`'s rule on what the window remembers and of its launch line (the design's parts 7.3.3 and
13.3.4), for me to approve before `CLAUDE.md` is edited, which happens only when the code that changes
them lands (stages 3 and 4); bundling Cascadia Mono so every computer shows it (not asked). Carried for
later stages, not stage 2: a selectable path in a sentence copies its words as handed in (the records
part); which colour each plotting call takes, held by no test, and the latent loss of a colour cycle's
other keys in the repeats' figure (the first stage that edits a plotting file, and the audit); on the
command line the FDT repeats' first colour stays the plotting library's blue, near the mean's (the audit).

Do not start the retrain, or any training of real size, in this chat. Everything else in the prompts of
6, 10, 11, 12, 13 and 14 October stands.
