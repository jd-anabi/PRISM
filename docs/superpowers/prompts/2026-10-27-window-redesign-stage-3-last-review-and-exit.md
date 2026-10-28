# Prompt for the next chat — the window redesign, stage 3 (the run): Task 14's last fix, the last review, the exit

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the
session of 27 October 2026 that built stage 3's Tasks 12, 13 and 14.

What in it is not the owner's own words:
- **The owner's own words in that session,** beyond the pasted prompt
  `2026-10-26-window-redesign-stage-3-from-task-12.md` (whose "How much to spend", "Ask me what is mine" and "What must
  not change" are repeated here), are one answer. Asked whether the console windows that `git` and `nvidia-smi` would
  flash under the new launcher should be cured by editing `core/artifacts/provenance.py`, the owner chose "Edit
  provenance.py (Recommended)". It is departure 121, applied as `bc4f43e`.
- **Everything else is that session's record:** the commits, counts, gate times, costs and the bar's figures are that
  session's or its agents', and how a task is run is that session's way.

---

PRISM — the redesign of the window: stage 3, the run. Finish Task 14 with its third fix round, then the stage's last
review and Task 15, the exit.

## Where things stand

Stages 1 (the look) and 2 (the shell) of the window's redesign are built, and so is the fit-saving stage.

**Stage 3's plan,** `docs/superpowers/plans/2026-10-22-window-redesign-stage-3.md` (f5faaaa, corrected as built through
Task 14's second fix round), is approved (23 October, "I approve it"). Every departure is in its table: 83 to 114, and
115 to 121 added by the build. 118 ("Three tries") and 121 (the provenance edit) are mine; 120 is the build's:
shutting down while a training saves its posterior stops nothing, and the window waits for the save.

**Tasks 0 to 13 are complete:**
- Task 12 (keep-awake, the reason against shutting down, the flash, Windows' end of a session): `93ca7f7`, `4ca7d70`,
  fix `53f9995`.
- Task 13 (the end's words in `core/gui/run/ends.py`, the foot from the status alone, "seen", Ctrl+R, Home's next
  step and "Last runs", the past view on a page of its own, the launch's closing and recall): `ff5e563`, `6307cd4`,
  fix `9be258e`, fix 2 `988137c`.

**Task 14 (the launcher) is built, but not complete:** `62b37e5`, `81a4282`, `c57b24f`, fix `bc4f43e` (my provenance
edit) and `f5f9cc9`, fix 2 `c8caede`. Its last gate, on `c8caede`: 1986 passed, 9 skipped, 5 deselected (24 min 16 s).
The real screen's set: 9 passed.

**Task 14's third fix round is owed.** Its fix rounds made the suites' guards fail closed. The last re-read showed that
the launcher tests' guard, reading `run.bat` line by line, can always be slipped past by a new edit, for example a line
that empties `PRISM_DESKTOP`; no test at `c8caede` crosses my desktop line. Its rulings, written in the ledger's folder
as `stage-3/briefs/task-14-fix3-rulings.md`, end that:
- the launcher tests run a copy only when its LF-normalised bytes have the SHA-256 pinned in the test;
- a standing test holds the pin to today's `run.bat`;
- every home the copy can name (`HOME`, `HOMEDRIVE`, `HOMEPATH`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`,
  `ProgramData`, `PRISM_DESKTOP`) points into the test's folder.

**My answers on the plan stand:**
- Resume is on the run's own page.
- "Keep this training so you can resume it later?" is asked on the run's page once the stop has landed.
- The launcher finds PRISM's Python itself, says so in a Windows message when it cannot, and makes a desktop shortcut
  once.
- One extra real-screen row at the exit, about fifteen minutes, with a Windows sign-out I start and cancel, none
  completed. That row now has two more steps, after "Cancel" with the run still live:
  - the window's close asks its question;
  - `window-log.txt` holds the cancelled line.

**Pushing.** I pushed up to 20eb415 (23 October); the 60-odd commits after it are not pushed (read `git status -sb`),
and I push when I choose.

**Before any work.** Nothing is running as this is handed over. Check that:
- the graphics card is free;
- no python is alive;
- no PRISM.lock stands beside my settings file;
- my desktop has its ten names and no `PRISM.lnk`.

A `python -m core.gui` started from run.bat is my own PRISM: leave it alone and wait; while it is open no gate runs. My
real PRISM.ini is 3,296 bytes, sha256 79BF94DF…C83E.

## How much to spend (decided; do not ask again)

"In between", as I chose on 12 October:
- no workflow unless I ask for one in my own words;
- every agent given its model by name, Sonnet where reading is enough;
- the review by three readers for every task that builds.

On 26 October: "Let's just keep the same rigor, it doesn't seem worth it to rush things", so do not propose cutting the
loop for speed. My words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near
their max, I won't buy anymore credits".

So:
- read the usage card when you start, tell me where the weekly bar stands, and read it again before every dispatch;
- do not count on extra usage;
- near about 90 %, start nothing that cannot be finished below it, and stop at a clean state.

The week resets on 1 November at 04:00 UTC; the bar read 46 % at 23:19 on 27 October. Tasks 12, 13 and 14 cost about 5,
6 and 4 points.

## Read first, in this order

1. `docs/STATE.md`: its head, and the paragraphs "THE BUILD OF STAGE 3, TASKS 12 TO 14", "… TASKS 9 TO 11", "… TASKS 6
   TO 8, AND TASK 9 PREPARED", "… TASKS 4 AND 5 …" and "… TASKS 0 TO 3"; and "THE FIT-SAVING STAGE'S LAST REVIEW AND
   EXIT" for the code audit's list.
2. `CLAUDE.md`.
3. The plan's head, down to "## Task 0", and Task 15's section.
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored):
   - its first 47 lines;
   - the stage 3 section from the line "NEW CHAT 2026-10-27" to its end;
   - "## Engine changes landed".

   Never add a line above the lines that begin "- HAND-OFF". Its lines are very long: search it, do not read it
   through.
5. The way of working, in the ledger's `stage-3/briefs/`. The trailer line in `review-seats.md` is out of date: each
   implementer uses its own harness's line. The newest models:
   - `review-seats.md`;
   - the dispatches and rulings of Task 14's fix rounds: `task-14-fix1-rulings.md`, `task-14-fix1-skeptic-dispatch.md`,
     `task-14-fix2-skeptic-dispatch.md`, `task-14-fix1-reread-dispatch.md`, `task-14-fix2-reread-dispatch.md`;
   - `task-13-carried-over-dispatch.md`;
   - `task-14-review-notes.md`.

   The implementer's dispatch is in this session's ledger lines.
6. The exit's files, in the ledger's `stage-3/exit/`:
   - `rulings-for-the-owner.md`, the list of what was decided without me, started at Task 12 and holding Tasks 12 to
     14 and the carried items;
   - `claude-md-launch-line.md`, the draft of `CLAUDE.md`'s launch line and the corrected counts of the guards (six and
     six), for my yes.
7. The design's parts as each task cites them (`docs/superpowers/specs/2026-10-05-window-redesign-design/`), and the
   prompts of 14 to 26 October beside this file. Where they differ the later wins, and this one wins over all.

## Tools

The scratch folder is `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`.

**The gate.** `gate.sh` (kinds fast, quiet, slow, display, collect; labels `s3taskNN-<sha>`, after a fix
`s3taskNN-fix1-<sha>`, `-fix2-`, `-fix3-`, and for the last review `s3final-…`). Run it from Bash in the background.

**The ground.** Run `ground.ps1`'s commands INLINE in the PowerShell tool:
- `$when` and `$label` set first;
- its param line and exit-2 guard left out;
- `Test-Path` on the listing instead.

Add the desktop's count and `PRISM.lnk`'s absence, as this session did.

**The ledger.** Append it with the Edit tool:
- Read its last lines first;
- the build's lines go just above "## Engine changes landed", and a departure's row after the table's last row;
- LF only.

**The carried-over list.** It is up to date through Task 13 (`all rows 913; twice []; missing 0: []; gone 40`).
`s3\co_rows_s3task13.py` is the newest model for an update. Task 14's update, after its fix round 3, is owed.

**Copies for a trial.** Make them with `git -C <repo> -c core.autocrlf=false archive <sha> | tar -x -C <folder>`: the
plain archive writes CRLF here. Restore `run.sh`'s 100755 in the copy's commits; archive loses it.

**Rules every agent was given this session**, each learned the hard way:
- pytest is started by ONE tool call at a time, never two in one message;
- no `python -c` beside another python;
- a test that reads state the whole application or process shares takes a reading first, or works on a stand-in:
  Task 12's first gate failed on that;
- a reader runs NO python while a gate runs.

## What to do

1. **Task 14's fix round 3**, through the loop:
   - one skeptic (opus) with the rulings file builds and proves the cure in a scratch copy of `c8caede`, the owner's
     lines first;
   - your rulings go into its brief;
   - one Sonnet implementer;
   - `cmp`;
   - the fresh gate and a scoped re-read at the same moment;
   - complete when the re-read is clean and the gate green on the same head;
   - the real-screen run after it;
   - then Task 14's carried-over list, and the plan's "As built" lines finished for the fix.
2. **The stage's last review** over the whole stage, from Task 1's first commit `ad4b7e2` to Task 14's head: three
   readers, one fix wave.
3. **Task 15, the exit, which needs me:**
   - a quiet machine for the gate of record, then the slow set;
   - the card test with the diagnostic card and run 5 (tell me first: the commit, the scratch folder, how long the last
     run took, and that nothing else may use the card);
   - the real-screen run;
   - the pictures, and my questions on them;
   - the walkthrough rows, my extra row among them, with its two new steps;
   - my yes or no to `CLAUDE.md`'s new launch line and its corrected counts;
   - the list of everything decided without me.

What the build departs from goes into the plan's table, the design's section 8 and the ledger, with its cost. Ask me
when it touches something I decided. Hand over to a new chat after the last review, at two thirds of the room, or when
the usage rule says stop. Read the clock before you write a time.

## Carried (each is in the ledger with its reason)

**Into the last review**, from the 26 October prompt's list (still owed):
- no standing check that no panel stays enrolled in `BasePanel._instances`;
- a test that fails while its run is live costs ten seconds;
- nothing pins that a start taken back lets go of its watchers;
- the loss tick's gap from "Loss"; the ground behind the best epoch's words;
- the store's refusal on Discard has lost its pin;
- the failure test's 0.8 s of real time;
- the narrowing round's reopen command compared as a string;
- the reopen command selectable by mouse only;
- the time axis's caption can widen the run page;
- the health table's gap when hidden;
- the refusal leg of a run's outcome not tested apart;
- the run page's head cut at the window's least size.

And from this session:
- `core/gui/run/history.py:15-20` does not say the open try is written "stopped" while a Windows stop waits;
- a ragged docstring wrap at `tests/test_run_states.py:2396-2399`;
- `_to_the_figures`'s early return when the figures are in sight is unpinned;
- the two breaches of the one-python rule (recorded; check the rule held since).

**Into the exit's documents step:**
- `docs/guide/retrain.md:477` and `docs/guide/command-line.md:112-118`;
- `architecture.md`'s list of the `core/diagnostics` modules (`card.py`);
- the lines naming the removed run flags, the progress pane or the Cancel button: `architecture.md:226`, `:648`,
  `:650`, `:673`; `rules-and-traps.md:260`, `:319`, `:385`, `:464`; `window.md:74`, `:78`, `:119`;
- the guide's lines on the entry module ("forces Agg before any core import", "do not move either line"):
  `architecture.md:23-24`, `:664`, `rules-and-traps.md:256-258`, and `testing.md`'s variables bullet;
- `testing.md` names the new guards (`_no_native_boxes`, `_every_native_box_is_read`, `_streams_put_back`);
- a block "Windows fatal exception" in `window-log.txt` followed by more launch lines is not a crash;
- the named code's digests of `digests-after-task03.json`, unchanged;
- the getting-started page's first line ("Checked against commit …") stamped.

**For me at the exit:** the 26 October prompt's list, and `stage-3/exit/rulings-for-the-owner.md`, which now holds,
among others:
- departure 120;
- the reason cut to 256 characters;
- the end's sentences I have not seen;
- the past view drawing stages and health with no figures;
- a new run setting aside an end nobody saw;
- the 2 px hyphen;
- the shortcut's Explorer environment (`PRISM_VRAM_CEILING_GIB` from a shell does not reach a window the shortcut
  starts);
- a moved repository;
- the shortcut's mark and how to remake it;
- `platform.release()`'s fallback to `ver` through a shell;
- a pinned shortcut and the window as two taskbar buttons.

**The code audit's list gains:**
- a modal box opened during Windows' end-message wait;
- a process ended between the posterior's write and the kept training's removal;
- `_end` letting go of keep-awake and the reason partway;
- the history's readings unbounded on a folder of many runs (about 60 ms per 1,000 warm, 3 s cold);
- the past view's replay (about 12 µs a line, half a second near 40,000 lines);
- the claim's two warnings printed bare;
- `platform.release()`'s shell fallback.

## Facts the build stands on

- The fast gate took 23 min 38 s to 24 min 39 s on this session's ten gates.
- The slow set took 1 h 33 min on 22 October.
- The desktop's share of the card is about 1.2 to 3.0 GiB.
- Times in the ledger are read from `date`.
- A `cd` inside a Bash call moves the session's working directory.
- In this checkout `run.bat` and `run.sh` read `i/lf w/crlf` since Task 14's build.

## Ask me what is mine

Ask me about what I decided, what the design leaves to me, what cannot be undone, and anything under "What must not
change"; one question at a time, in plain words, with a picture where it helps and your recommendation. Decide the
small things yourself, write each as a ruling with its cost, and give me the list at the exit. AskUserQuestion waits
for my answer: ask while long runs go.

Still mine to hear:
- `CLAUDE.md`'s launch line, at this stage's exit, only on my yes;
- the rule on what the window remembers (7.3.3), in stage 4;
- bundling Cascadia Mono (never asked).

## What must not change without my say-so

- The science: the models, the simulation, the features, the training and calibration methods, every setting in
  `core/config.py`.
- What a simulated row holds, and the simulation set's identity: the finished set `Artifacts/simulations/eda5c2a1e468`
  stays usable.
- What a fit that is never interrupted computes.
- The rules in `CLAUDE.md`, edited only on my yes.
- Anything on disk under `Artifacts/`, and the console capture `C:\Users\J\prism-retrain-console.log`: read freely; ask
  before anything deletes, renames, rewrites or trains on them.
- Nothing signs me out, locks or shuts down the computer, and nothing writes into my real desktop folder: the
  launcher's shortcut is tested in a scratch folder until I run it myself.
- No task edits the named training code (`core.SBI.fit_state.FIT_CODE`).
- No engine file changes but the one comment of Task 7 and my provenance edit of 27 October (done).

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card
gates are fine.

## Standing rules

- The local main branch; one commit per logical step, with a short subject, never amended, ending with the
  Co-Authored-By line the harness provides; I push.
- Never two pytest processes at once; never edit a source file while a suite runs; a run on the card has the card to
  itself.
- Scratch files under `C:\Users\J\AppData\Local\Temp\`; no `.py` file under `.superpowers/` or `docs/`.
- Update `docs/STATE.md` at the end of every session.
