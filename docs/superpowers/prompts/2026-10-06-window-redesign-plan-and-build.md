# Prompt for the next chat — the window redesign: the plan and the build

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice. Four things in it are not the owner's own words, and are there to be changed if they
are not wanted: under "How to work", the sentence that agrees to multi-agent workflows in that chat
(the owner agreed to them for the design chat); under "What to do", the order "plan one stage,
build it, then plan the next" (the design recommends it, and the owner approved the design); the
recommendation attached to each of the two opening questions (the design chat's); and the name of
the ledger's folder.

---

PRISM — the redesign of the window: plan it and build it, one stage at a time.

**Where things stand.** The design of the redesign is written, and I approved it on 6 October 2026
without striking anything. Nothing is built: no file under `core/` or `tests/` has changed since
`3ec5bf6`, and the fast gate has not run since `771a460`. The order of work is unchanged: this
redesign; then, in another chat, the audit of the whole code; then the retrain, from the window.
Nothing was running when this was written; check that the graphics card is free before any work on
it.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**Earlier, 2026-10-01", only; the
   part that matters most starts at "THE REDESIGN CHAT HAS STARTED". The file is about 1,600 very
   long lines; search the rest when you need it, do not read it through.
2. `docs/superpowers/specs/2026-10-05-window-redesign-design.md` — all of it. It is the design I
   approved: my decisions, seventeen choices made in drafting, the design in brief, the twelve
   engine changes, what becomes of each finding, the nine stages, and an appendix of every other
   choice made in drafting. The seventeen and the appendix stand as written.
3. `docs/superpowers/specs/2026-10-05-window-redesign-assets/decisions-as-approved.md`, then
   `rulings-after-the-drafts.md` — the locked records of what I decided. Where they differ, the
   later wins: the rulings over the decisions, and within the rulings part D over C over B.
4. In `docs/superpowers/specs/2026-10-05-window-redesign-design/`: part 13 (the tests, the stages,
   the pages, the walkthrough) in full, and the parts the stage you are about to plan is built
   from, in full. For stage 1 that is part 4, the look. Part 11 is the engine changes. The eleven
   parts are about 108,000 words: do not read them all at once, and have agents read the ones a
   stage needs.
5. In the assets folder, `toolkit-sample.py.txt` and the `toolkit-*.png` pictures: the chosen look,
   built and proven in PySide6, with the pictures I chose from beside them (`structure-chosen-*`,
   `look-chosen-*`). The sample is something to learn from, not code to copy in.
6. `docs/superpowers/notes/2026-10-05-the-window-as-read-for-the-redesign.md`, and its two
   appendices when a part cites a finding (ids W001 to W198) or you need the inventory of today's
   window.

The design's parts were checked against the code as it stood on 6 October by readers who ran
nothing. Check a line number or a claim against the code before you build on it.

**Two things to ask me before anything else, one at a time, in plain words.**

- Item 16 of the design's part 2: whether a figure's line colours number five or eight. It was put
  to me as a recommendation (eight) and I did not answer it. Stage 1 needs the answer.
- When the separate design for saving the fit as it goes is made. I asked for it as a design of
  its own, to accept or leave, and have not said when. Until it exists, any end of the program
  during a fit loses the whole fit. The design chat's recommendation: before stage 3 is planned at
  the latest, because stage 3 builds everything the window says about what a stop, a close or a
  failure costs; stages 1 and 2 do not depend on it.

**What to do.**

- Write the plan for ONE stage, from the parts as they stand, and give it to me to approve; then
  build that stage, task by task; then plan the next. Start with stage 1, the look's foundation,
  in the order part 13 gives for it.
- Seven things are settled by a small experiment before what depends on them is built (the design's
  part 2, item 17). Each is a task with its fallback stated; stage 1 owes the one on whether the
  toolkit already follows Windows' text-size setting.
- The tests pin the window as it is now, and the redesign rewrites many of them. Part 13 gives each
  of today's 322 window tests its fate and names the fourteen new suites. Every task: its tests
  written or rewritten first and seen to fail for the stated reason; then the code; then a review
  and ONE one-process fast gate, started together.
- The engine changes are the twelve of the design, E1 to E12, each in the stage part 13 gives it,
  and nothing else. If a stage turns out to need more of the engine, stop and tell me. List every
  engine change in `docs/STATE.md` as it lands: the audit chat will read that list first.
- The card smoke gate in `CLAUDE.md`, with the diagnostic card, is owed once, at the end of
  stage 3. Tell me before you run it.
- At the end of stages 1 and 2 show me pictures of every screen, in light and dark. You can look at
  the window yourself: render a screen off-screen to an image and read the image, before and after
  a change. When you do, give the window a scratch settings file and a scratch record store, the
  way the test suites do (`docs/STATE.md` says how): a window built on the real ones reads my
  remembered selections and overwrites them when it closes. Building the window also takes a
  little memory on the graphics card.
- How it looks and feels on my screen is mine to judge. The rows I must run on a real screen are
  in part 13.4 (G1 to G33): add them to `docs/checklists/display-walkthrough.md` as a new group
  when their stage lands, and never edit an earlier row. No row may ask for a training of real
  size or touch the records kept from the stopped retrain.
- When what is built departs from the design, say so, ask me if it touches something I decided,
  and record it in the design as a deviation with what it costs if it is wrong.
- This will take many sessions. Keep a ledger for the piece in a gitignored workspace,
  `.superpowers/sdd/2026-10-06-window-redesign/` (the progress file, each task's brief and report,
  the review reports, the gate logs), as the earlier pieces did, and leave `docs/STATE.md` at the
  end of every session so that a new chat can start from it and from this prompt.

**How to work.**

- I agree to multi-agent workflows in this chat. Parallel agents read, plan and review, and a
  second reader checks findings and plans; only one of them at a time edits the code, nobody
  edits while a test suite is running, there are never two pytest processes at once, and you, not
  an agent, start and wait for the long gates.
- Ask me before you assume. One question at a time, in plain language, with a short background
  first and no code names, and your recommendation with it. Show me a picture when a picture is
  clearer than a description.
- The three design plugins (impeccable, ui-ux-pro-max, frontend-design): their methods are applied
  by hand where they help, as the design chat did. Nothing of theirs is written into the
  repository, impeccable's program is never run, and their labels and file names stay out of the
  code, the tests and the guide.
- The Qt Development Skills plugin is installed but switched off, because using it means accepting
  Qt's AI Services terms, and I do not accept them. Do not enable it and do not use any part of
  it. To check what a widget, a layout or a style sheet can actually do, read Qt's public
  documentation or the PySide6 installed here, and try it in a small scratch script before you
  build on it.
- claude-mem and claude-mem-cowork are installed but cannot work on this machine until I set them
  up. Do not rely on them and do not set them up yourself. If a hook error from them appears on
  each prompt, ignore it.

**What must not change without my say-so.**

- The science: the models, the simulation, the features, the training and calibration methods and
  every setting in `core/config.py`. The design adds one line to that file, which names the folder
  the plans live in; I was told and approved it, and nothing else there changes. The load-check
  audit on the open list comes before any other change to `core/config.py` or to the features; if
  the build needs one, stop and tell me.
- What a setting means. The simulation cache's identity must not change by accident: a change to
  it makes the finished simulations unusable. Tell me before any change that does this.
- The rules in `CLAUDE.md`: every knob reaches its stage as an argument; a refusal carries a field
  key and the window names the control from its own table; a consent, a seed, a record name and a
  note are never remembered; nothing under `core/`, `tests/` or `docs/guide/` cites the working
  documents or their labels — no decision, engine-change or finding id, no part or stage number,
  no ruling, no plugin's name. Two of its rules change WITH this redesign, and only when the code
  that changes them lands: what the window remembers (the memory limit for the graphics card
  becomes remembered, per computer, in Settings) and the launch line. The replacement wording is
  in the design's parts 7.3.3 and 13.3.4 for me to approve; do not edit `CLAUDE.md` before I have.
- On disk, and I have not asked for any of it to be deleted: the prior `retrain_prior`, the
  complete simulation cache `Artifacts/simulations/eda5c2a1e468`, the two probe records under
  `Artifacts/diagnostics/` (`retrain_band` and `retrain_mask`), anything else already in
  `Artifacts/`, and the console capture `C:\Users\J\prism-retrain-console.log`. Read them freely;
  ask me before anything deletes, renames, rewrites or trains on them, and write no row for me to
  run on screen that would.

Do not start the retrain, or any training of real size, in this chat. The tiny trainings inside the
test suites, the card smoke gate once I have been told, and the small ones in the rows I run on
screen are fine.

**Standing rules.** Work on the local `main` branch; one commit for each logical step, a short
subject, never amended; I push. Never two pytest processes at once, and never edit a source file
while a suite is running. A run on the graphics card has the card to itself. Scratch files go under
`C:\Users\J\AppData\Local\Temp\`, and no `.py` file goes under `.superpowers/` or under `docs/`
(the source scan fails on any folder that holds one and is not on its list). Update
`docs/STATE.md` at the end of every session.

**Done means, for a stage.** I approved its plan; it is built; every task had its review and its
gate; the stage's exit as part 13 lists it has passed, with the gate's time recorded on a quiet
machine; the window is usable and its suites are green; and `docs/STATE.md` says what landed, which
engine changes, which deviations from the design, which gates were not run, and which rows are now
mine to run.

**Done means, for the redesign.** All nine stages; the one-process fast gate green on the final
commit; the tests marked for the real screen run on it; the card smoke gate passed after the
changes that touch training; the rows G1 to G33 written, and run by me on the real screen;
`docs/guide/window.md` and `docs/guide/getting-started.md` describe the new window, every other
guide page that describes it is brought up to date as part 13.3 says, and the new page states the
look (the retrain runbook's window route is written by the audit chat, once the code is settled);
`docs/STATE.md` says what changed, lists every engine change and every deviation, says which of
the findings were fixed and which were left, says which gates were not run, and says what the
audit should look at; and neither the retrain nor any training of real size has been started.
