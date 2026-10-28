PRISM — the redesign of the window: stage 3, the run. Its exit, Task 15, with me.

(Written on 28 October by the session that finished stage 3's last review, for me to paste. My own words are quoted:
"I approve it", "Three tries", "Edit provenance.py (Recommended)", "Let's just keep the same rigor, it doesn't seem
worth it to rush things", "Try to also finish things up at a clean state when usage limits are near their max, I won't
buy anymore credits". The rest is that session's record, written in my voice.)

Where things stand

Stages 1 (the look) and 2 (the shell) of the window's redesign are built, and so is the fit-saving stage. Stage 3's
plan, docs/superpowers/plans/2026-10-22-window-redesign-stage-3.md (f5faaaa, corrected as built through Task 14), is
approved (23 October, "I approve it"). Every departure is in its table: 83 to 114, and 115 to 121 added by the build;
118 ("Three tries") and 121 (the provenance edit) are mine.

Tasks 0 to 14 are complete. On 28 October:
* Task 14's third fix round, `bb560ba`: the launcher tests run a copy of `run.bat` only when its bytes have the digest
  written in the test, a standing test holds that digest to today's file, and every home folder the copy can name is
  the test's own. Its gate: 1987 passed, 9 skipped, 5 deselected (25 min 12 s); the real screen's set 9; its re-read
  clean.
* The stage's last review, three readers over `6dd4f12..bb560ba`: no Critical, one Important (a Discard pressed after
  another run had started left the page, the foot and Home saying the training was kept). Its ONE fix wave, three
  commits: `2fb637c`, `ba4b281`, `4abfd8e`. Its gate: 1989 passed, 9 skipped, 5 deselected (27 min 26 s); the real
  screen's set 9; its re-read clean.

The code's last commit is `4abfd8e`; the tree is clean at the documents commit after it (STATE and this prompt). What is
left is Task 15, the exit. It needs me.

My answers on the plan stand: Resume is on the run's own page; "Keep this training so you can resume it later?" is
asked on the run's page once the stop has landed; the launcher finds PRISM's Python itself, says so in a Windows
message when it cannot, and makes a desktop shortcut once; one extra real-screen row at the exit, about fifteen
minutes, with a Windows sign-out I start and cancel, none completed. That row has two more steps, after "Cancel" with
the run still live: the window's close asks its question; `window-log.txt` holds the cancelled line. If either fails,
Windows sent PRISM no "not ending" message, and a cure follows as a small task of its own.

Pushing. I pushed up to 20eb415 (23 October); the commits after it are not pushed (read `git status -sb`), and I push
when I choose.

Before any work. Nothing is running as this is handed over. Check that:
* the tree is clean;
* the graphics card is free;
* no `python.exe` or `pythonw.exe` is alive (`gate.sh` looks for `python.exe` alone; `ground.ps1` lists both);
* no PRISM.lock stands beside my settings file;
* my desktop has its ten names and no `PRISM.lnk`.

A PRISM started from `run.bat` or its shortcut (a `pythonw.exe -m core.gui`) is my own: leave it alone and wait; while
it is open no gate runs. My real PRISM.ini is 3,296 bytes, sha256 79BF94DF…C83E.

How much to spend (decided; do not ask again)

"In between", as I chose on 12 October: no workflow unless I ask for one in my own words; every agent given its model
by name, Sonnet where reading is enough; the review by three readers for every task that builds. On 26 October: "Let's
just keep the same rigor, it doesn't seem worth it to rush things", so do not propose cutting the loop for speed. My
words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't
buy anymore credits". So: read the usage card when you start, tell me where the weekly bar stands, and read it again
before every dispatch; do not count on extra usage; near about 90 %, start nothing that cannot be finished below it,
and stop at a clean state. The week resets on 1 November at 04:00 UTC; the bar read 51 % at 13:03 on 28 October.
Hand over to a new chat at two thirds of the room, or when the usage rule says stop. Read the clock before you write a
time.

Read first, in this order

1. docs/STATE.md: its head and the paragraph "THE LAST REVIEW OF STAGE 3 AND TASK 14's THIRD FIX ROUND". The code
   audit's list is NOT one paragraph: it is every "The code audit's list gains" sentence of STATE's paragraphs (search
   "audit's list"), the ledger's "## Engine changes landed" ("for the code audit" notes), and the last review's sorting,
   section 5 (`stage-3/reviews/stage-3-final-review-sorting.md`). Step 9 gathers them into one list.
2. `CLAUDE.md`.
3. The plan's head, down to "## Task 0", and Task 15's section: its ten steps are the exit, and Steps 5, 8, 9 and 10
   hold more than the lists below (`powercfg /requests`; the walkthrough's two carried rows, a knock while a question is
   open and a real second start minimised and maximised, and G1's amendments; the command-line page's items; a change
   asked for is a small task of its own).
4. The ledger, .superpowers/sdd/2026-10-06-window-redesign/progress.md (gitignored): its first 47 lines; the stage 3
   section from the FIRST line that begins "- NEW CHAT 2026-10-28" to its end; "## Engine changes landed". Never add a
   line above the lines that begin "- HAND-OFF" (the last is "- HAND-OFF 2026-10-28"). Its lines are very long: search
   it, do not read it through.
5. The exit's files, in the ledger's `stage-3/exit/`:
   * `rulings-for-the-owner.md`: what was decided without me, through the last review; its last section, "The exit's
     questions already carried", is the list of questions to put to me with the pictures;
   * `claude-md-launch-line.md`: the draft of `CLAUDE.md`'s launch line and of the Tests section's counts. THE COUNTS ARE
     TO BE RECOUNTED from `tests/conftest.py` and the draft corrected before I am asked: the draft's "six and six" misses
     the per-test check `windows_asked`; the module check `_no_panel_outlives_its_module` (the fix wave) is a third kind,
     neither session-wide nor per test, and needs words of its own; the suites are thirty, not twenty-six, and the four
     new ones (`test_run_text.py`, `test_run_states.py`, `test_stop_and_close.py`, `test_retry.py`) need their lines in
     `CLAUDE.md`'s `tests/` bullet.
6. The last review's rulings, `stage-3/briefs/stage-3-final-fix-rulings.md`: its section "Not in the wave" is what the
   exit carries. The facts behind departures 122 and 123, `rng.py`, and departures 106's and 87's words are seat one's
   M1, M2, M3 and M7 in `stage-3/reviews/stage-3-final-review-product.md`.
7. For the pictures: stage 2's picture dispatch `stage-2/briefs/task-11-pictures-dispatch.md`, its script
   `s2\t11-pictures\pictures2.py` in the scratch folder, and its ledger lines (the nine checks): the model for this
   stage's.
8. The design's parts as Task 15 cites them (`docs/superpowers/specs/2026-10-05-window-redesign-design/`, 13.1.7,
   13.3, 13.4), and the prompts of 14 to 27 October beside this file. Where they differ the later wins, and this one
   wins over all.

Tools

The scratch folder is `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`; `gate.sh` and `ground.ps1` are in it.

The gate. `gate.sh` (kinds fast, quiet, slow, display, collect; labels `s3exit-<sha>` for the exit). Run it from Bash
in the background.

The ground. Run `ground.ps1`'s commands INLINE in the PowerShell tool: `$when` and `$label` set first; its param line
and exit-2 guard left out; `Test-Path` on the listing instead. Add the desktop's count and `PRISM.lnk`'s absence.

The ledger. Append it with the Edit tool: read its last lines first; the build's lines go just above "## Engine
changes landed", and a departure's row after the table's last row; LF only.

The carried-over list is `.superpowers/sdd/2026-10-06-window-redesign/carried-over.md`, up to date through the last
review's fix wave (`all rows 931; twice []; missing 0: []; gone 40`). Its check is `task01_test_names.py --check
carried-over.md` (as `stage-3/briefs/stage-3-final-fix-carried-over-dispatch.md`, step 5, runs it), and
`s3\co_rows_s3final.py` in the scratch folder is the newest model for an update. Step 6 holds the collection against
it.

Copies for a trial. Make them with `git -C <repo> -c core.autocrlf=false archive <sha> | tar -x -C <folder>`: the
plain archive writes CRLF here. Restore `run.sh`'s 100755 in the copy's commits; archive loses it.

Rules every agent was given, each learned the hard way:
* pytest is started by ONE tool call at a time, never two in one message;
* no `python -c` beside another python (a reader typed `python3 -c` on 28 October; it is not on the path, so nothing
  ran);
* a test that reads state the whole application or process shares takes a reading first, or works on a stand-in;
* a reader runs NO python while a gate runs;
* a `cd` inside a Bash call moves the session's working directory: end it back at the repository root;
* never write a file into the repository while an implementer works there (its brief checks `git status`);
* GNU sed reads `\c` in a replacement as a control character: write a path with a backslash through the editor, and
  check a written document with `file` before committing it.

What to do: Task 15, the exit

First, one question to me: the order. The plan runs the gates first (Steps 2 to 5) and the pictures after (Step 7). A
change I ask for at the pictures is a small task of its own, and would make the quiet gate, the slow set and the card
test stale. Recommended: take the pictures and my questions on them first (on hidden windows; nothing of mine is
touched), build any change I ask for, then run the gates of record once, on the final head.

The steps, as the plan has them:
1. The fence (Step 1): `git diff --stat f5faaaa..HEAD -- core ':!core/gui'` (the plan's commit). Beside the files
   Tasks 3 and 4 list and `core/SBI/pipeline.py`'s one comment, it shows two the plan's file map does not name:
   `core/diagnostics/rng.py` (a docstring, `36ee51a`, Task 4's fix) and `core/artifacts/provenance.py` (the 'no
   window' flag on its two calls, `bc4f43e`, departure 121, my yes of 27 October). Both are now in the ledger's
   "Engine changes landed".
2. The fast gate of record on a quiet machine (Step 2): ask me when the machine is quiet; then the slow set (Step 3,
   about an hour and a half).
3. The card test with the diagnostic card and run 5 (Step 4): tell me first the commit, the scratch folder, how long
   the last run took, and that nothing else may use the card.
4. The real-screen run (Step 5).
5. The collection against the carried-over list (Step 6).
6. The pictures (Step 7): a picture script taught the run page in every state, the foot's eight states, Home during and
   after a run, each question rendered (never shown), the full text open; three sets, light and dark; the nine checks.
   One agent (opus) builds it from stage 2's model, on hidden windows only. Among them: the run page at the window's
   least size (900 x 600), a fit whose time axis's caption names its end, and a page with figures and no message. Then
   my questions on them, each alone with its picture and your recommendation.
7. The walkthrough rows (Step 8), my extra row among them with its two new steps.
8. The documents (Step 9), and my yes or no to `CLAUDE.md`'s new launch line and its corrected counts. At the same
   yes, ask me whether `CLAUDE.md`'s long-run bullet ("the window's Resume comes with its redesign") is to be reworded
   now that Resume is on the run's page, and name anything else in `CLAUDE.md` the stage made stale.
9. The named training code's digests into STATE (Step 10): `stage-3/gates/digests-after-task03.json`, unchanged.
10. The list of everything decided without me (`rulings-for-the-owner.md`).

Carried into the exit's documents (Step 9)

From the earlier tasks: docs/guide/retrain.md:477 and docs/guide/command-line.md:112-118; `architecture.md`'s list of
the `core/diagnostics` modules (`card.py`); the lines naming the removed run flags, the progress pane or the Cancel
button (`architecture.md:226`, `:648`, `:650`, `:673`; `rules-and-traps.md:260`, `:319`, `:385`, `:464`;
`window.md:74`, `:78`, `:119`); the guide's lines on the entry module ("forces Agg before any core import", "do not move
either line": `architecture.md:23-24`, `:664`, `rules-and-traps.md:256-258`, and `testing.md`'s variables bullet);
`testing.md` names the new guards and checks (`_no_native_boxes`, `_every_native_box_is_read`, `_streams_put_back`,
`windows_asked`, and the module check `_no_panel_outlives_its_module`); a block "Windows fatal exception" in
`window-log.txt` followed by more launch lines is not a crash; the named code's digests, unchanged; the getting-started
page's first line ("Checked against commit …") stamped.

From the last review: departure 122 (figure windows stay parentless, as before; the plan's File Map line corrected) and
departure 123 (the history's `plan`, `settings` and `check` empty and `health` short until stage 4), each in the plan's
table, the design's section 8 and the ledger, with its cost; `core/diagnostics/rng.py` and `core/artifacts/provenance.py`
in Task 4's and Task 14's "As built" lines; departures 106's and 87's words made true; the plan's registry ("Names tasks
share"), whose `discard` line is corrected to `discard(run) -> tuple[bool, str, str] | None` (gone, words, the end's new
sentence); in STATE's audit list, the history writes that wait for the disk on the window's thread (up to 0.8 s when a
scanner holds the file) added, and `streams.py:58` and `worker.py:197-199` taken off (settled by Task 10).

For me at the exit: `stage-3/exit/rulings-for-the-owner.md`, which holds, from the last review: the time axis's caption
can widen the run page; the run page's head at the window's least size; the messages' head keeps a 6 px gap when no
message shows; the fast gate's time (below); the fix wave's choices.

Carried to later stages (in the last review's rulings): to stage 4 the history's empty fields, the reopen command by
mouse only, the inert opt-in for a taken name, the helpers defined two to five times, `history.continuable` with no
caller; to stage 7 Live simulation's sentence and its failed compact page; to stage 9 the reason during a retry's
pause, the refusal leg of the waiting close, ` (--resume)` at the end of the last listed line.

The code audit's list gains, from the last review: `build_posterior` rebinding its parameter `note`; two constants of
one name with different words (`FINISHED_TEXT`, `NO_MESSAGE`); `request_stop`'s "safe from any thread"; the native
message's guard reading one module; a history file that cannot be written leaving another run's page with its old words
under "The training was discarded."; `tests/test_look.py`'s unpack of the axis name's box that names no reason; the
sorting's sixteen. And from Tasks 12 to 14 (now in STATE's last paragraph): a modal box opened during Windows'
end-message wait; a process ended between the posterior's write and the kept training's removal; `_end` letting go of
keep-awake and the reason partway; the history's readings unbounded; the past view's replay; the claim's two warnings
printed bare; `platform.release()`'s shell fallback.

Facts the exit stands on

* The fast gate's time: this stage's gates, none on a quiet machine, ran from about 21 minutes to 27 min 26 s (the last
  with a reader beside it); the last review puts the time the stage's own tests add at about 3.7 minutes at today's
  speed (a third of it five engine tests of Tasks 3 and 4). `CLAUDE.md`'s recorded one-process target is 16 minutes;
  the last quiet gate of record was stage 2's, 19 min 21 s.
* The slow set took 1 h 33 min on 22 October.
* The desktop's share of the card is about 1.2 to 3.0 GiB.
* Times in the ledger are read from `date`.
* In this checkout `run.bat` and `run.sh` read `i/lf w/crlf`; the launcher pin reads both forms as one. An edit to
  `run.bat` is an edit to the pin in `tests/test_shell.py`, in the same commit.

Ask me what is mine

Ask me about what I decided, what the design leaves to me, what cannot be undone, and anything under "What must not
change"; one question at a time, in plain words, with a picture where it helps and your recommendation. Decide the
small things yourself, write each as a ruling with its cost, and give me the list at the exit. AskUserQuestion waits
for my answer: ask while long runs go. Still mine to hear: `CLAUDE.md`'s launch line and counts, at this stage's exit,
only on my yes; the rule on what the window remembers (7.3.3), in stage 4; bundling Cascadia Mono (never asked).

What must not change without my say-so

* The science: the models, the simulation, the features, the training and calibration methods, every setting in
  core/config.py.
* What a simulated row holds, and the simulation set's identity: the finished set
  `Artifacts/simulations/eda5c2a1e468` stays usable.
* What a fit that is never interrupted computes.
* The rules in `CLAUDE.md`, edited only on my yes.
* Anything on disk under `Artifacts/`, and the console capture C:\Users\J\prism-retrain-console.log: read freely; ask
  before anything deletes, renames, rewrites or trains on them.
* Nothing signs me out, locks or shuts down the computer, and nothing writes into my real desktop folder: the
  launcher's shortcut is tested in a scratch folder until I run it myself.
* No further edit of the named training code (`core.SBI.fit_state.FIT_CODE`; Task 3's one edit, `c08acbf`, stands).
* No further engine file changes (the fence lists those made).

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card
gates are fine.

Standing rules

* The local main branch; one commit per logical step, with a short subject, never amended, ending with the
  Co-Authored-By line the harness provides; I push.
* Never two pytest processes at once; never edit a source file while a suite runs; a run on the card has the card to
  itself.
* Scratch files under `C:\Users\J\AppData\Local\Temp\`; no `.py` file under `.superpowers/` or `docs/`.
* Update docs/STATE.md at the end of every session.
