# Prompt for the next chat — the window redesign: the plan of stage 2 (the shell)

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice by the chat of 14 October 2026. What in it is not the owner's own words:

- The owner's own words of 14 October are the answers to the questions put at stage 1's exit (each
  the option offered as the recommendation; the ledger quotes them) and "Start now" for the gate of
  record. Everything else is that chat's record and its way of handing over, to be changed if it is
  not wanted. That the plan of stage 2 goes to a new chat is that chat's proposal: its context stood
  past half its room, and a plan of about ten tasks is written better from a clean start.
- Every count, time and measurement is that chat's record, not something the owner measured.

---

PRISM — the redesign of the window: the plan of stage 2, the shell.

**Where things stand.** Stage 1 of the redesign, the look's foundation, is BUILT and its exit is
PASSED (14 October 2026): the plan I approved on 7 October, `8b2ca92`..`f9c6899`, 100 commits; the
fast gate of record on a quiet machine 1407 passed, 4 skipped, 5 deselected in 16 min 9 s; the
display-marked run 4 passed; the slow set 5 passed in 1 h 6 min; the pictures of every screen in
three sets and 51 extra ones. At the exit I answered five questions with pictures, and the answers
were built at once as a small task of their own, 19q (`80249b6`..`066b55c`: a time axis with no tick
takes finer ticks; a late axis grows by a quarter of the run; a long word may break after a
backslash, slash, underscore or full stop; a message row with no marker lines up with the others;
the FDT repeats skip the mean's blue), reviewed by three readers and gated: 1413 passed, 4 skipped,
5 deselected. The last commit that touched code is `066b55c`. Nothing has been pushed since
`d8828f9`: I push when I choose. Nothing was running when this was written; check that the graphics
card is free and no python is alive before any work.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no
workflow unless I ask for one in my own words; every agent given its model by name, Sonnet where
reading is enough; the review by three readers for every task that builds. What 14 October cost in
background agents: about 2.6 million tokens for the exit and 2.5 for 19q; read the usage card when
you start and tell me where the weekly bar stands.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE WINDOW REDESIGN, PLAN AND
   BUILD", only. The paragraphs "TASK 19q" and "STAGE 1 OF THE WINDOW REDESIGN IS BUILT" are the
   last chat's record.
2. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/` (gitignored): of `progress.md` the
   first 47 lines; then everything from the line that begins "- NEW CHAT 2026-10-14" to the end of
   the stage's section; then "Deviations from the design" (rows 49 to 53 are new) and the last ten
   lines of "Gates". It is the record and the recovery map: trust it and `git log` over anything
   remembered. Never add a line to it above the three lines that begin "- HAND-OFF": append with
   `ledger_append.py`. Its lines are very long: search it, do not read it through.
3. What stage 1 carries to stage 2's plan (below, "What the plan must carry"), each in the place
   named there.
4. The design, `docs/superpowers/specs/2026-10-05-window-redesign-design.md` (section 8, the 53 rows
   of departures, wins over the parts where they differ) and its parts: `03-shell.md` whole (the
   shell is stage 2), `13-tests-stages-docs.md` (13.1, the tests; 13.2.2, stage 2's list; 13.3 and
   13.4 for what stage 2 owes the pages and the walkthrough), and what `04-look.md`, `05-anatomy.md`
   and `10-records-and-parts.md` say that the shell uses; the locked records
   `docs/superpowers/specs/2026-10-05-window-redesign-assets/decisions-as-approved.md` and
   `rulings-after-the-drafts.md` (the later wins). Have readers read the parts against today's code;
   read yourself what the plan's head needs.
5. Stage 1's plan as the model of a plan: its head as cut for implementers,
   `.superpowers/sdd/2026-10-06-window-redesign/stage-1/briefs/plan-head.md` ("Global Constraints",
   "How to run this plan", the registry, the standing checks), and how its tasks were briefed.
6. The prompts of 10, 11, 12 and 13 October beside this file, for everything this one does not
   repeat (the skills and plug-ins and their limits; what must not change without my say-so; the
   standing rules; what "done" means for a stage).

**What to do.** Write the plan of stage 2 from the design's parts as they stand, with the
`writing-plans` skill, and give it to me to approve. Build nothing before I have. The design sizes
stage 2 at about ten tasks; stage 1 was sized at eleven and came to about thirty, mostly because the
briefs were stale and the trial runs found what reading had not: say honestly what you expect, and
put the preparation the last chats learned into the plan (a skeptic builds each task as a stand-in
against the REAL devices before its implementer starts; a brief changed by rulings is folded into
ONE document and walked again). Stage 2 delivers, by part 13.2.2: the rail; the six steps as a
chain; Home as a status page with its first-time opening; each part built when first opened; one
PRISM at a time, after the scratch tests part 3.10 asks for; the run's state at the rail's foot as
`RUN_STATE` knows it today; the keys. Today's panels sit inside each place unchanged, in the new look;
the tab bars and the back arrow go. No engine change. Its exit: the display-marked run (the start-up
path changes) and pictures. Ask me, one question at a time with a picture and your recommendation,
what the parts leave to me; decide the small things yourself and write each as a ruling with its
cost. After stage 2 is built, the build pauses for the separate design for saving the fit, as I
decided on 6 October.

**What the plan must carry** (gathered by the last chats; each is in the ledger with its numbers):

- From stage 1's last review, the sorting's rows for stage 2
  (`stage-1/reviews/stage-1-final-review-sorting.md`, section 4): every new window, part or step
  class named in the builders' scan (the scan is now closed over the window's files: a new class
  fails until it is named); the resize in the unusable-geometry branch of the window-size code
  pinned; the class fonts set twice at start (`core/gui/app.py`); Home's tiles lower than the code
  asks; the shell tells a page it is small after it is built, so a gap, a part or a body added to a
  small page keeps its usual form (the three-line cure: the page's adders use the form in force);
  the stray ring after a mouse click on a tab leaves with the old tab bars (pictures and a test). The
  sorting's row on the message row with no marker is DONE (19q).
- From the product seat (`stage-1-final-review-product.md`, minors 1 to 7): a page tells only what it
  was handed (a page head handed in inside a layout is never told the window is small; two columns
  stack only when they ARE the scroller's body); a handed-in length is scaled inside by some devices
  and used as given by others: one registry line saying which; the scan of drawing methods knows no
  hide event, change event or layout geometry call: the hide event joins it before the shell writes
  one; a font handed over late is dropped by any new dressing, `set_class` among them (the rail's
  rows are given a class at every change of place: fonts at build); anything the shell opens as a
  window of its own paints slowly beside a run.
- From the tests seat (`stage-1-final-review-tests.md`): the window suites leave every window alive
  (368 top-level widgets after the settings suite; the helper that reads the process's state has no
  key for it): the builders hand out what a fixture lets go; the two lines of the session guard
  against a dialog that would stall the gate (its M-2), proved by a gate in stage 2's FIRST task.
- From 19q's review: four small test gaps with their lines written and run (a short word beside a
  long one; the characters other than the four that break; the accessible name after a second
  `set_text`; the repeats' order under the window's settings and the blue written as numbers), a
  word of a docstring ("its designed width", `tests/test_look.py:5133`) and a comment for the time
  axis's loop: a tests-and-words front of stage 2's first task.
- From the exit: a picture script never SHOWS a dialog, even hidden (one message box reached the real
  screen for 0.3 s in the extras' first run): it renders it. The shell's pictures at stage 2's exit
  use `pictures.py` (`...\prism-redesign-scratch\`, text copies in `stage-1/tools/`), which must be
  taught the rail in place of the tab bars and the section pages.
- For later stages, not stage 2 (carry them on): a selectable path in a sentence copies its words as
  handed in (the records part plans one); which colour each plotting call takes, held by no test (the
  first stage that edits a plotting file, and the audit); on the command line the FDT repeats' first
  colour stays matplotlib's blue, near the mean's (the audit).

**Mine to hear, still open:** the replacement wording of `CLAUDE.md`'s rule on what the window
remembers and of its launch line (the design's parts 7.3.3 and 13.3.4), approved only when the code
that changes them lands (stages 3 and 4); bundling Cascadia Mono so every computer shows it (not
asked). The list of everything decided without me in stage 1 is in the ledger,
`stage-1/exit/rulings-for-the-owner.md`: I may undo any of it; keep giving me such a list at each
stage's exit.

Do not start the retrain, or any training of real size, in this chat. Everything else in the
prompts of 10, 11, 12 and 13 October stands.
