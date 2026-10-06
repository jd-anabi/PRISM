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

## C. The controller's second rulings (after the independent readers)

These settle what the readers handed up. They are later than part B and win where they differ.
Every one is the controller's; those marked "(unseen)" are choices the owner has not seen and stay
marked so in the text and in the part's closing list.

1. **"Continue with these" passes the remembered kind of observation** (B12) and every other option
   at its built-in value or the plan's (B16); Home's sentence names the kind. Parts 3, 7 and 8 say
   the same. (unseen)
2. **Home's "Continue from batch N"** is offered for the newest training or narrowing round that
   ended by a failure, or was not seen to end, whether or not its end was acknowledged, for as long
   as its simulation set is unfinished and no later run of that kind has started. When the failure
   was in the fit the button reads "Fit again on the saved simulations". For a run the user
   stopped, Home's next step is the ordinary one (the Training page, whose "Before you start" says
   it continues the saved set). Parts 3, 6 and 12 say the same.
3. **A run that only loads a record** shows "Loading…" at the rail's foot while it runs and returns
   to idle with nothing to acknowledge; it leaves no history file and no text file. Neither do a
   Live simulation or a video export. (unseen: it narrows U17's "one small file per run")
4. **A name and a note are never refilled**, also not after a restart: "Continue from batch N"
   opens the page with both blank and the earlier name shown as a hint. Parts 3, 6, 7 and 13 say
   the same.
5. **The window's own three tries (U12) follow an out-of-memory failure of a training or a
   narrowing round.** Any other failure is reported at once, with "Continue from batch N" where it
   applies. (unseen: it narrows U12's "tries again by itself")
6. **An engine warning the window has no sentence for** is shown on the run's page as an Attention
   message that begins "From the engine:" and gives the engine's sentence as it stands; it is never
   hidden behind a count alone. (unseen)
7. **A failure's kind (the error's type name) appears only behind "Show the full error"**, in every
   failure display: the run's page, a part that failed to build, the last-resort dialog.
8. **Closing the window** asks about a live run first (U12); then, if a model of the user's own has
   unsaved edits, asks "Throw away the unsaved changes?" (U17). The close is part 6's; the words
   are part 10's.
9. **A next-step button on a record's page** asks one question up front (what it will load and what
   it will let go), then loads. A consent (a narrowed posterior; a different observation) is never
   implied: it is asked inside the chain when it arises. "Narrow a posterior around it" on an
   observation is greyed until a posterior is in use. Parts 5, 8 and 10 say the same.
10. **The shared question dialog can carry a list** under its one sentence, one item to a line,
    scrolling beyond eight. The from-zero question, the plan-fill question and Records' questions
    use it; a list a decision requires (U10's "names every difference") is never behind a link.
11. **B11 is amended.** The narrowing round's "Before you start" reads: "Simulates N simulations in
    the round's own simulation set. A round never continues a training's simulations; if this same
    round was run before with the same settings, its saved simulations are continued." The engine's
    refusal of a set one setting away is shown as a question with the engine's sentence as it
    stands. Drawing the region beforehand to preview it is not designed in.
12. **Pre-run deciding checks and "Start training"**: a check not yet run never blocks and is led by
    "Attention"; a check that does not apply is led by "Information"; a pre-run deciding check that
    FAILED does not grey the button, but the start then asks one question first, with "Cancel"
    preselected. (unseen)
13. **The memory lines** appear only in the "Before you start" of runs that simulate on the card.
14. **The repeated calibration's count field** is labelled "Fixed number of driven recordings".
15. **A posterior trained without the frequency response**: the three checks that do not apply read
    "Does not apply"; the headline reads "n of 3 deciding checks passed; 3 do not apply to this
    posterior"; step 4 in the chain reads "Done" when every deciding check that applies has passed.
    (unseen: it amends U16's "always n of 6" for this one case; put it first in the list)
16. **The same case has the same words on the run's page and on the Checks list**: no audit yet,
    "Waiting for the audit"; a total that covers part of the set, or none counted, "Not compared";
    up to 3 points "Fine" and "Passed"; over 3 up to 5 "Look closer"; beyond 5 "Failed" with "needs
    you". The run history's health entry holds the audit record's id beside its reading.
17. **The smallest window is 900 by 600 at Windows' ordinary text size and grows with the text
    size**, held to the screen. Part 3 follows part 4. (unseen)
18. **Dates.** In tables, records and sentences: "3 Oct 2026, 14:02" (B22). On the run's page and
    at the rail's foot a time within seven days carries its weekday ("Sun 4 Oct, 12:10"), as the
    approved pictures did. Axis ticks stay short ("2 Oct", "2 Oct 00:00").
19. **The line that says where a run stands** is set in the lettering face under the stage line, as
    pictured. **A check's verdict sentence** is a message row: a marker and a sentence in the book
    face.
20. **E12.** One engine module holds the four line colours (blue, vermilion, green, orange); a test
    holds them equal to the look's set; the ink comes from the figure settings. After the colours
    have been used once a figure's further lines repeat them dashed, so curves stay distinguishable.
    The exported video follows the light figure settings like any stored figure. (unseen)
21. **The card test on the graphics card**: part 13 owns when. One run at the exit of stage 3,
    covering E1, E3, E7 and E10; again at stage 9 only if code that moves tensors changed after it.
    E9, E11 and E12 owe none by themselves; E9 needs a fast test with the saving of simulations on.
22. **E7 is opt-in and only the window opts in**, so a command-line run is unchanged by it. **E10
    applies to both front ends**: a failure while the last figure is drawn no longer fails the run;
    the run ends with its record and a warning. That IS a change to what a command-line run does at
    its last write, and it goes first in the list for the owner beside E1's and E6's text changes.
    E7 is not carried to an inference made together with its observation. (unseen)
23. **E11's scope** is every refusal sentence the engine raises, the three further refusal kinds
    (units, an effective-temperature model, a model's equations) and the support sentences shown
    beside a greyed start button included; "FDT" becomes "effective temperature". Reason strings
    that are not refusals stay as they are.
24. **The first-time opening's page is built in stage 2** (Home must open on something) and its
    help is completed in stage 8. (unseen)
25. **Walkthrough rows are owed for every rebuilt surface**, not only the shell, the look and the
    run: also the Checks list, a record's page and its actions, the two modes of Inference, a small
    narrowing round, Effective temperature, Live simulation, Your own models. The new group's
    heading says which earlier rows it supersedes; earlier rows are never edited.
26. **A test that builds the whole program must release the one-PRISM lock**; part 3 follows part
    13.
27. **Plans.** U8's "one action" is the button "Fill the forms from this plan" beside the list;
    choosing in the list alone changes nothing (unseen: it differs from the letter of "picking").
    Plan files are tracked in git like bounds and cell files and are committed before a long run,
    so its records are not stamped as made from a changed tree. (unseen)
28. **The timing check** reads Training's expected epochs as B9 resolves them; part 8 exposes the
    resolved value.
29. **The appearance and the keep-awake switch stay changeable during a run**; every other setting
    is locked.
30. **Part 13 owns the order in which tests are rewritten and the interim style rules** of stage 1;
    part 4 adopts its order.
31. **Housekeeping for every part**: delete "Seam:" lines that are now answered; check every
    reference to another part's section number against that part's file as it now stands; the
    idle sentence at the rail's foot is "Only one run at a time."; five keys are added to the
    mock-up's (Ctrl+7, Ctrl+8, Ctrl+9, F1, Shift+F6).

## D. The controller's last rulings (after the cross-check of all eleven parts)

Later than part C; they win where they differ.

1. **C20's dashed repeat is withdrawn.** The figure settings' line cycle is the look's colours and
   nothing else. E12 changes the plotting files' line colours only and edits no line style (B18,
   U4). A figure with more unnamed lines than colours repeats colours, as any cycle does; telling
   many curves apart by style is not designed in. The rest of C20 stands.
2. **One label for one key, everywhere.** The cell's file is "Cell file" on every page that has it.
   The length of an observation is "Recording length" on every page that has it, Live simulation
   included; that page's sentence says the settling-in time is simulated before it and is shown
   marked.
3. **The order of the build, stages 4 and 5.** Stage 4 delivers the pages of steps 1 to 3 (Model
   and options, Prior, Training) with plans, "This computer" and applied against pending, so the
   plan picker has its page and a fill reaches steps 1 to 3. Stage 5 delivers Checks, Inference and
   the Narrowing round, whose forms join the fill then; the test that one press fills every step
   is completed in stage 5. (unseen: U19 put all steps but Training in stage 5)
4. **Every modal dialog the window opens is the shared dialog class or the one file-chooser
   function.** "Save as a plan…" and "Rename…" are the shared dialog in its form that asks for one
   line of text; part 4 defines that form, parts 7 and 10 use it, and part 13's guard covers the
   class, so no dialog can stall a run of the suites.
5. **The run history is read for the records folder in use.** Each history file names the records
   folder its run wrote to; Home, the first-time opening and "Continue from batch N" count only
   the files of the folder in use.
6. **"Fit again on the saved simulations"** stays on offer until a later run of that kind starts or
   the saved simulation set is removed.
7. **A figure the window draws itself.** Part 13 lists, as a task of stage 3 before the loss curve
   is built, a test that renders such a figure on the window's thread many times while another
   thread builds and saves a figure. If they can collide, the loss curve is drawn from its numbers
   by a painted widget and no figure library runs on the window's thread.
8. **A run ended by "shut down anyway"** counts as stopped by the user: its simulations are saved
   as far as Windows allowed, and Home shows the ordinary next step.
9. **A comma typed as the decimal sign** is refused in words when editing ends ("Use a point as the
   decimal sign."); the key is not silently swallowed.
