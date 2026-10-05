# Prompt for the next chat — the UI/UX redesign

Paste everything below the line into a new chat opened on this repository. It is written in the
owner's voice. One sentence in it agrees to multi-agent workflows in that chat (under "How to
work"); delete that sentence if that is not wanted.

---

PRISM — a complete UI/UX redesign of the window.

**What I want.** A complete redesign of PRISM's window: how it looks, how it is laid out, and how
the work flows through it. I run PRISM from the window, not from the command line, and the next
full-size training (the retrain) will be started from the window too, so the window has to be fit
for that run, not only look better. I have not told you what I want it to look like or how it
should be organised. Find that out from me first. Whether this redesign also closes the gaps listed
in the findings note below, or deals only with the look, the layout and the flow and leaves some of
them to the audit or to the command line, is one of the first things to ask me.

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**Earlier, 2026-10-01" (about
   thirty lines), and the short paragraph that begins "**Next, in order (since 2026-10-05):**",
   only. The file is about 1,500 very long lines with no blank line between those blocks; search
   the rest when you need it, do not read it through.
2. `docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md` — all of it.
   Part 2 is what the window cannot do, or does unsafely, on a days-long run, and what it already
   does well; part 4 is the questions nobody has answered yet. These are findings and open
   questions to bring to me during this redesign; none of them is a decision I have made. Part 3 is
   written for the audit chat, but several of its leads are faults in the window itself (what
   "Quit anyway" does, the wrong waiting time shown for Cancel, the line that names the wrong
   setting, the list cut to three values, Delete during another program's run), and part 3C lists
   what `docs/guide/window.md` says wrongly. Where the redesign rebuilds that part of the window or
   rewrites that page, fix them, and say in `docs/STATE.md` which you fixed and which you left for
   the audit. The note was read from the code and not run: check a claim against the code before
   you build on it.
3. `docs/guide/window.md` and `docs/guide/getting-started.md` — what the window is today, screen
   by screen, and what it remembers.
4. `docs/guide/rules-and-traps.md`, the window parts of its traps, and the section on the window
   in `docs/guide/architecture.md` — the mistakes this window has already made once, and how its
   runs, threads and Cancel work.
5. `docs/checklists/display-walkthrough.md` — what has been checked on a real screen, and at what
   size.
6. The window's code, `core/gui/` (about 12,300 lines): start-up in `app.py` and `main_window.py`;
   `screens/` (Home, the navigation shell, the plain section screen, the inference screen, the
   Artifacts screen, Settings, the model builder); `session.py`, the one session the six inference
   tabs share; `panels/` (the six inference tabs under `panels/inference/`, FDT, the sweep study,
   Simulate, the reduction map, and `base_panel.py`, which every panel shares); `widgets/`; the
   look in `design.py`, `theming.py`, `fonts.py`, `icons.py` and `mpl_theme.py`; a run's plumbing
   in `worker.py`, `streams.py` and `vt.py`; what is remembered in `settings.py`; and `fields.py`,
   the table that turns a refusal's field key into the name of the control on screen.
7. The earlier design work on the window, for what was decided then and why:
   `docs/superpowers/specs/2026-09-17-gui-usability-and-artifact-browser-design.md`, the last
   design given wholly to the window, and the window parts of the two designs after it
   (`2026-09-22-secondary-analyses-design.md` for the FDT, sweep-study, Simulate and model-builder
   screens; `2026-09-25-documentation-and-retrain-readiness-design.md`).

**Where things stand.** The retrain was started from the command line on 1 October and I stopped
it on 5 October, 62 epochs into the fit, because I never wanted to run it from the console. The
order of work is now: this redesign; then, in another chat, an audit of the whole code for bugs
and mistakes; then the retrain, from the window. Nothing was running when this was written; check
that the graphics card is free before any work on it. On disk, and I have not asked for any of it
to be deleted: the prior `retrain_prior`, the complete simulation cache
`Artifacts/simulations/eda5c2a1e468` (20,480,000 rows), the two pre-flight probe records under
`Artifacts/diagnostics/` (`retrain_band` and `retrain_mask`), anything else already in
`Artifacts/`, and the console capture `C:\Users\J\prism-retrain-console.log`. Read them freely;
ask me before anything deletes, renames, rewrites or trains on them, and write no row for me to
run on screen that would.

Do not start the retrain, or any training of real size, in this chat. The tiny trainings inside
the test suites, and tiny ones in the rows I run on screen, are fine.

**How to work.**

- This is a large design task. Start with brainstorming, and ask me before you assume: what I
  dislike in the window today, what I want it to feel like, how I want to move through a piece of
  work, what a long run should look like while it is going, and which of the things that exist
  only on the command line I want in the window. One question at a time, in plain language, with a
  short background first and no code names. Show me options as pictures when a picture is clearer
  than a description.
- Ask early whether the window stays on PySide6. Everything here is built on it, and changing the
  toolkit would be a far larger project than a redesign; I have not asked for that, so say what it
  would cost before proposing it. The design skills installed here are written mostly for web
  pages: use them for direction, hierarchy, type, colour, wording and critique, and for mock-ups I
  can react to, not as a source of components.
- You can look at the window yourself: render a screen or a panel off-screen to an image and read
  the image. Do that before and after a change. When you do, give the window a scratch settings
  file and a scratch record store, the way the test suites do: a window built on the real ones
  reads my remembered selections and overwrites them when it closes, and what is remembered there
  now is evidence in the findings note. Building the window also takes a little memory on the
  graphics card. How it looks and feels on my screen is still mine to judge: write the rows I
  should check into the display walkthrough, and I will run them.
- Then a written design for me to approve, then a plan, then the build, task by task. Every task
  gets a review and a one-process fast gate, started together.
- I agree to multi-agent workflows in this chat. Parallel agents read, design and review, and a
  second reader checks findings and designs; only one of them at a time edits the code, nobody
  edits while a test suite is running, there are never two pytest processes at once, and you, not
  an agent, start and wait for the long gates.
- The tests pin the window as it is now: `tests/test_nav_and_gating.py` (about 5,100 lines, the
  table that names each control among them), `test_worker_dispatch.py`,
  `test_artifact_browser.py`, `test_settings_persistence.py`, `test_vt_progress.py`,
  `test_simulate.py`, `test_figures.py`, `test_user_models.py` (the model builder), and
  `tests/test_docs.py`, which checks the guide. Several more touch the window in a few places:
  search `tests/` for the window's package before you size the work. A redesign rewrites many of
  them. Plan that from the start and write each test before the code it covers.

**What the redesign should take up with me.** Take the detail from the findings note; these are
the headings.

- *Setting up a long run without traps.* Today the network size silently reopens at its default
  on every launch; the memory ceiling for the graphics card also reopens switched off on every
  launch, and the stopped run showed that a full-size run on this card with no ceiling dies; the
  remembered bounds file and prior can be the wrong ones; choosing a prior in the list loads
  nothing until the build-or-load button is pressed; re-applying the model puts the bounds file
  back to the one remembered at launch without saying so; and the guard against simulating from
  zero by accident is thin: a question is asked only when the run is exactly one setting away from
  an existing set of simulations, any other mismatch gets one passive line, and that line can name
  the wrong setting. Two of these collide with standing rules: science settings are deliberately
  never remembered between launches, because a remembered one once trained a five-day run on a
  retired value, and the memory ceiling is never remembered by a rule of its own (a forgotten
  ceiling slows every later run with nothing to say why). If a run's settings should be easier to
  get right,
  ask me what would help, lay out the options with what each costs and risks, and let me decide.
  Do not quietly start remembering them.
- *Watching a long run.* No lasting file of the run's lines, a 5,000-line pane, an epoch counter
  with no clock and no loss curve until the end, a "no output" notice that stands for an hour, and
  the whole window locked for days.
- *Stopping, closing and failing.* What Cancel, closing the window and a failure do during a
  simulation and during a fit, and what the window says about each. Some of what it says today is
  wrong, one click on Cancel discards a fit of days with no question asked, and one way of closing
  may lose a finished fit.
- *Around the run.* Names and notes, the calibration's seed, the checks that exist only on the
  command line (two of them decide whether a retrained model passes, and the checks made before a
  run are among them), the narrowing round's network size, and reading a record's numbers and
  figures.
- *What already works and should survive.* Before the button: the count of simulations, the
  worst-case memory of one batch against what is free, and whether the run resumes or starts anew;
  the question asked on a one-setting near miss, with Cancel as its default, and the question
  before a session that holds work is replaced; the memory ceiling as a visible field; the live
  progress and the loss figure; the finished model held ready for the next step; and Cancel during
  a simulation saving the work done so far. If the design drops one of these, say so and why.
- *Everything else in the window* — Home, Simulate, the FDT and sweep-study tabs, the reduction
  map, the Artifacts screen, Settings, the model builder — as part of one design, not left behind.

**What must not change without my say-so.**

- The science: the models, the simulation, the features, the training and calibration methods and
  every setting in `core/config.py`. If the design needs the engine to do something new — save a
  fit as it goes, write a run's lines to a file — say so in the design as an engine change, keep
  it small, and list every one of them in `docs/STATE.md` at the end: the audit chat will read
  that list first. Every stage already takes a name and a note and checks the name before it
  spends anything; the training stage also takes a resume policy, an epoch ceiling and a
  checkpoint cadence. Offering those in the window is not an engine change.
- The rules in `CLAUDE.md`: every knob reaches its stage as an argument; a refusal carries a field
  key and the window names the control from its own table; a consent, a seed, a record name and a
  note are never remembered; nothing under `core/`, `tests/` or `docs/guide/` cites the working
  documents.
- What a setting means. The simulation cache's identity must not change by accident: a change to
  it makes the finished simulations unusable. I have said reusing them is not required, but tell
  me before any change that does this. The load-check audit on the open list comes before any
  change to `core/config.py` or to the features; if the design needs one, stop and tell me.

**Standing rules.** Work on the local `main` branch; one commit for each logical step, a short
subject, never amended; I push. Never two pytest processes at once, and never edit a source file
while a suite is running. A run on the graphics card has the card to itself. Scratch files go under
`C:\Users\J\AppData\Local\Temp\`, and no `.py` file goes under `.superpowers/`. Update
`docs/STATE.md` at the end of every session.

**Done means.** I have approved the design; it is built; the one-process fast gate is green, the
tests marked for the real screen have been run on it, and the card smoke gate in `CLAUDE.md` has
passed if an engine change moved data on the graphics card (tell me before you run it); the new
walkthrough rows are written and I have run them on the real screen; `docs/guide/window.md` and
`docs/guide/getting-started.md` describe the new window, and every other guide page that describes
it has been brought up to date (at least the window parts of `architecture.md` and
`rules-and-traps.md`, and the model-builder part of `recordings.md`; the retrain runbook's window
route is written by the audit chat, once the code is settled); `docs/STATE.md` says what changed,
lists every engine change, says which of the note's window faults were fixed and which were left,
says which gates were not run, and says what the audit should look at; and neither the retrain nor
any training of real size has been started.
