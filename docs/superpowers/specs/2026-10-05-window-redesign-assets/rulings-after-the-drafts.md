# Rulings after the first drafts (2026-10-06)

Read with DECISIONS.md. Part A is the owner's (decided in chat after the drafts, each from a
question with its options laid out): it has the same force as DECISIONS.md and where the two
differ, this file wins. Part B is the controller's: it settles the seams and contradictions between
the drafted parts. Every part must be brought into line with both. A ruling in part B that is a
choice the owner has not seen stays marked "(a choice the owner has not seen)" in the text.

## A. The owner's further decisions

- **U21 Four engine additions are approved**, each small and listed for the audit:
  - **E8** one line in `core/config.py` naming the plans folder, beside the lines that name the
    bounds, cells, units and models folders. (The owner was told, as the rule on that file asks.)
  - **E9** a simulation set keeps its name when it is saved again or reused (today
    `core/artifacts/store.py` rewrites it empty), so "Rename…" holds for every kind.
  - **E10** a guard where the engine itself draws and saves the loss figure at the end of a
    training, so a failure there cannot remove the finished posterior.
  - The engine's list of settings gains the names the plan reader and the observation's name need
    to say what is wrong (part of E4 and E5).
  The owner was also told that E1 and E6 change text the command line prints (a line per epoch; the
  tails of seven refusal lines) with no change to a flag or to behaviour.
- **U22 The engine's refusal sentences are reworded to the vocabulary of U7 (E11).** Text only: no
  rule changes. The command line prints the same sentences, so its wording changes too; its flags
  keep their names. Done in the LAST stage of the build, the tests that pin those sentences
  rewritten first.
- **U23 Figures.** The plotting code stops naming its own line colours and takes the look's
  colour-blind-safe set (**E12**; plotting code only; no number in any figure changes). **Figures
  saved into a record are always drawn on a light page**, whatever the window's theme. In the dark
  theme a stored figure is shown as a light picture. Figures the window itself draws live (the loss
  curve on a run's page, the Live simulation view) follow the theme.
- **U24 Home, on reopening mid-project:** Home says what was in use last time (for example
  Nadrowski, the bounds file master_tier1.txt, the prior retrain_prior) and offers one button,
  **"Continue with these"**. One press loads them; Home then shows the real next step. Nothing is
  loaded until it is pressed.
- **U25 During a run, saved effective-temperature runs can be browsed but not compared.** Opening,
  reading and viewing figures stay possible at any time. A new comparison waits until the run ends;
  its button says so. (This amends U11's "browsable and comparable".) No second job of any kind
  runs beside a live run.
- **U26 One PRISM at a time.** Starting PRISM while it is already open brings the open window to
  the front instead of opening another. The command line is unaffected.
- **U27 Windows shutting down or signing out during a run.** While a run is live PRISM registers a
  reason with Windows, so a sign-out or shutdown shows it and waits for "shut down anyway" or
  "cancel". If the owner shuts down anyway during simulation, PRISM uses the moment to save the
  simulations done so far. Nothing changes when no run is live.

The engine changes are therefore E1 to E12:
E1 loss per epoch; E2 every differing setting named; E3 the card test as an engine step; E4 an
observation's name; E5 the plan reader; E6 refusals that say where to fix it; E7 a name taken
during the run cannot cost the result; E8 the plans folder named in core/config.py; E9 a simulation
set keeps its name; E10 the guard round the engine's own loss figure; E11 the refusal sentences
reworded; E12 the plotting code's line colours.

## B. The controller's rulings on seams

1. **The comparison after simulating (U11) has three bands, and "needs you" only beyond 5.**
   Up to 3 points from the audit: "Fine"; the check reads "Passed". Over 3 and up to 5: "Look
   closer": one Attention message on the run's page, the run is not interrupted and the rail's foot
   does not say "needs you"; on the Checks list the check reads "Look closer" and counts toward the
   headline only once the owner presses "Accept this difference" on the check's page, which is
   recorded with its date in the run history (a choice the owner has not seen). Beyond 5: the check
   reads "Failed", the run's page and the rail's foot say "needs you", and the window never stops
   the fit by itself.
2. **One question dialog.** Every question, and "Check your inputs", is one shared dialog class in
   the system's own frame (a dialog instance shown modally; never a static call), laid out as the
   toolkit sample's: the safe answer the default and the answer to Escape, the two answers at
   opposite ends, the sentence in the book face. `QMessageBox` is no longer used by the window. The
   test guard that today patches `QMessageBox.exec` is extended to this class in the task that
   introduces it, and tests read every question through one helper.
3. **What a step holds changes only when a run returns its record.** A refused, failed or stopped
   build, load or training leaves in use what was held before.
4. **A next-step button on a record's page ("Check this posterior", "Run it on recordings", "Set up
   a run like this one") acts:** it asks to load the record (with the release question when
   something is held), then opens the step. It does not leave the record "Changed, not applied".
5. **The first-time opening** is the shell part's: its choices, order and short sentences are
   final. Longer explanations live in Help.
6. **Closing writes the window's size and place only.** Remembered picks are written when they are
   applied; the training budget and the description of the recordings when a run starts with them.
7. **A step whose needs are unmet can still be opened and read**; its main button is greyed with
   the reason (a choice the owner has not seen).
8. **A run's files (its dated full text and its history) live in a `runs` folder beside the
   settings file**, per user, not under the records root (a choice the owner has not seen).
9. **The fit's expected length.** Training has, under "Rarely changed", "Expected epochs (for the
   time estimate only)", which a plan may hold. Blank means: the epoch count of the last finished
   training of the same kind in Records; with neither, the page says only when the fit cannot end
   before. It never limits the fit (a choice the owner has not seen).
10. **The numbers the deciding checks are judged by** (the expected share of unusable driven
    recordings and its widths; the 3- and 5-point limits; the two parts of the input the input-use
    check must find healthy) exist in no engine code. They are named constants in the window's
    Checks code, each with the reason in words, and are on the audit's list as science numbers
    living outside the engine.
11. **The narrowing round** always simulates a new set (its region is new). Its "Before you start"
    says so with the cost. It asks nothing about starting from zero, except that the engine's own
    refusal of a set one setting away is shown as a question at the click, with the engine's facts,
    in place of today's tick box (a choice the owner has not seen). Training keeps U10's rule to the
    letter.
12. **Still remembered, as today:** the kind of observation, and the narrowing round's own budget.
    **Never remembered:** the recording lengths and the drive amplitude, which open blank at every
    launch.
13. **Plans**: one sub-folder per model; a plan never names the model; settings are written under
    the engine's setting names, and a plan the window saves carries each on-screen label as a
    remark on its line; the window writes only settings that differ from the built-in values, plus
    the bounds file; no plan file is shipped by this redesign (choices the owner has not seen).
14. **E6's extent**: the two training refusals a continue policy reaches, the input-use check's
    five plain errors, and the one plain error of the third identifiability method raised before
    anything is spent when the posterior carries no training prior. Other refusals without a field
    key stay as they are. (The last item is an extension the owner has not seen.)
15. **U25 removes the "side job"**: nothing computes beside a live run. Parts 6 and 10 drop it.
16. **"Continue with these" (U24)** applies last time's model and options at the built-in values,
    the bounds file, and loads the saved prior and, if one was in use, the saved posterior. If a
    plan was in use last time the button's sentence names it and the press fills it in as well; the
    press is the explicit act that U8 requires. Nothing is run.
17. **One PRISM at a time (U26)**: a lock beside the settings file and a message to the open
    window to come to the front. A run-history file that still says "live" at launch therefore
    always means a run that died, never one live elsewhere.
18. **Engine-built figures are always drawn with the light settings** (U23): the window installs
    the light figure settings for everything the engine draws, in both themes. Only figures the
    window draws itself follow the theme. E12 is the plotting files' colours only.
19. **Console text** (a run's full text, a record's log) is set in a fixed-width face, a third
    typeface beside the two of U6 (a choice the owner has not seen).
20. **Text size**: whether the toolkit already follows Windows' text-size setting is to be
    established by a scratch test in the first stage, before any mechanism is built.
21. **Each part of the program is built when it is first opened**, not at start-up; a part that
    fails to build shows an error page in its place and the rest works (a choice the owner has not
    seen).
22. **Dates on screen** are written one way, as "3 Oct 2026, 14:02".
23. **The calibration check's form has "Seed" under "Rarely changed"**: blank means one is drawn and
    recorded; it is never remembered (a choice the owner has not seen).
24. **Live simulation's stop does not ask**, because nothing is lost there; every stop that loses
    work asks (a choice the owner has not seen).
25. **The card test's screen runs the chain once**; the five-command gate stays a command-line
    recipe.
26. **Rename is offered for a simulation set too** (E9).
27. **A second answer in a question is drawn as destructive only when it loses work.**
28. **Shutdown (U27)**: registered when a run enters "running" and withdrawn when it ends; on
    "shut down anyway" PRISM requests the stop and holds the shutdown for as long as Windows allows
    while the simulations done so far are saved; during a fit nothing can be saved and the reason
    says so.
