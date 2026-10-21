# Prompt for the design chat — saving the fit as it goes

Paste everything below the line into a chat opened on this repository, if the design is not finished in the
chat of 21 October that wrote this. It is written in the owner's voice by that chat, after stage 2's exit. What
in it is not the owner's own words:

- The owner's words behind it: the approved design's U4 ("Saving the fit as it goes is a SEPARATE design, to
  accept or leave"); the answer of 6 October that it is made between stages 2 and 3, and that stage 3 is planned
  on the answer to it (the redesign ledger's table "The owner's answers in this chat"); the order of 5 October
  (the redesign, then the code audit, then the retrain from the window; no training of real size before them);
  the spending rules of 12 and 16 October; and on 21 October "Write the prompt for the fit-saving design chat".
- Everything else is the chat's record and reading: the list of questions the design must settle is the chat's
  framing, offered as questions, not as anything the owner has decided.

---

PRISM — the separate design for saving the fit as it goes.

Where things stand. The window's redesign is built through stage 2 (the shell): STATE's paragraph "STAGE 2 OF THE
WINDOW REDESIGN IS BUILT" says what landed. The build pauses here, as I decided on 6 October, for this design; stage
3 (the run: its page, stop, close, failure, the end, the run history) is planned on my answer to it. The approved
redesign (U4) left saving the fit as it goes to a separate design, "to accept or leave", and says until it exists
"any end of the program during a fit loses the whole fit; the simulations are kept". That is what happened on 5
October: I stopped the console retrain about 63 hours into its fit (62 epochs of about an hour each) and the whole
fit was lost; the complete simulation cache `Artifacts/simulations/eda5c2a1e468` survived. Nothing is running; check
that the graphics card is free and no python is alive before any work (a `python -m core.gui` started from `run.bat`
is my own PRISM: leave it alone and wait).

How much to spend is decided: do not ask again. "In between", as I chose on 12 October: no workflow unless I ask for
one in my own words; every agent given its model by name, Sonnet where reading is enough. On 16 October I said: "Try
to also finish things up at a clean state when usage limits are near their max, I won't buy anymore credits". So:
read the usage card when you start, tell me where the weekly bar stands, and read it again before every dispatch; do
not count on extra usage; when the weekly bar nears its end (about 90 %), start nothing that cannot be finished below
it, and stop at a clean state.

Read first, in this order.

1. `docs/STATE.md` — from the top to the line that begins "**THE BUILD OF STAGE 2, TASK 10", only.
2. `CLAUDE.md` — its rules hold for anything this design asks to build.
3. What the stopped retrain measured and what the window cannot do on a days-long run:
   `docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md`, section 1 (the run, the fit's
   pace, the host memory) and section 2C (stopping and closing), and its section 4's questions.
4. The approved redesign where the fit is named: `docs/superpowers/specs/2026-10-05-window-redesign-design.md`
   (U4, U10 to U12 and U18 in its decisions; section 7, "What is not in this design"), its part
   `2026-10-05-window-redesign-design/06-run.md` (the run page this design changes), and finding W172 in
   `12-findings.md`; the locked decisions in `2026-10-05-window-redesign-assets/decisions-as-approved.md` are not
   re-decided.
5. The code, by reader agents, not read through here: where the fit runs (`core/SBI/train.py`, the one training
   call; the training stage in `core/orchestrator.py`), how the simulation cache already checkpoints and resumes
   (`core/SBI/training_checkpoint.py` and its tests), how a posterior is written and loaded (`core/artifacts/`), and
   what the installed sbi 0.26.1 offers for continuing a training (in the `biophys-env` interpreter).

What to do. Use the `brainstorming` skill: read, then ask me one question at a time, in plain words, with a short
background and your recommendation, and a picture where a picture is clearer. Among what the design must settle
(your framing, for me to answer, not decided): whether to save the fit at all, or leave it; what a saved fit holds
(the network's weights, the optimizer's state, the epoch, the best validation loss and the early-stopping count,
the random states, which rows were held out for validation) so that a resumed fit continues the same fit; how often
it is saved, and what it costs in time and disk; where it lives in the record store and what makes a saved fit not
resumable (any setting that changes what is trained); whether a fit stopped by hand can leave a usable posterior
(the best so far) and what that posterior would be called; what the window says and offers (U10's "a fit cannot be
resumed", U12's "Stop training…", the close question, the run history) and what the command line does; and how a
resumed fit is tested at tiny size. Measure what you can with tiny trainings in a scratch copy, on a scratch store.

What must not change without my say-so: the science (the models, the simulation, the features, the training and
calibration methods, every setting in `core/config.py`); what a simulated row holds and the simulation cache's
identity (the finished cache must not become unusable by accident; tell me before any change that would do it);
what a trained posterior is, for a fit that is never stopped; the rules in `CLAUDE.md`; anything on disk under
`Artifacts/` and the console capture `C:\Users\J\prism-retrain-console.log` (read freely; ask before anything
deletes, renames, rewrites or trains on them).

Done means: a written design under `docs/superpowers/specs/`, with every choice I have not seen listed for me, which
I approve, change or leave; the engine changes it asks for named for the code audit; `docs/STATE.md` and the
redesign ledger (`.superpowers/sdd/2026-10-06-window-redesign/progress.md`, a section of its own after stage 2's;
never a line above the three lines that begin "- HAND-OFF") say where it stands; then the plan of stage 3, which
takes the design into account, for my approval. Hand over to a new chat, with STATE, the ledger and a prompt like
this one, at two thirds of the room or when the usage rule says stop, whichever comes first.

Do not start the retrain, or any training of real size, in this chat. Standing rules: the local `main` branch; one
commit per logical step, a short subject, never amended, ending with the Co-Authored-By line the harness provides; I
push. Never two pytest processes at once; never edit a source file while a suite runs; a run on the card has the card
to itself. Scratch files under `C:\Users\J\AppData\Local\Temp\`; no `.py` file under `.superpowers/` or `docs/`.
