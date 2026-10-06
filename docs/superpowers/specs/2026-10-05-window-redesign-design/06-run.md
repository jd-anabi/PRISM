## 6. A run, from start to end

What the window does and shows from the click that starts a run until its end has been seen (U11, U12, U25, U27; U10's question at the click is 8.4's). A run is any background job (U7). Code read at commit `c45cc4a`. Paths are under `core/gui/` unless they start with `core/`, `tests/` or `docs/`, or name a file at the repository root; a path given in full once is shortened afterwards; "sbi base.py" is `site-packages/sbi/inference/trainers/base.py` of the pinned 0.26.1. Engine lines are in code type; only words in double quotes appear on screen. Dates are C18's: on the run's page a time within seven days carries its weekday ("Sun 11 Oct, 12:10"); any other, and any date in a question, is B22's ("11 Oct 2026, 12:10").

A page's part ends at `dispatch` (5.11). From there to the acknowledged end everything is this part's, held by one object, the run controller.

### 6.1 What stays of today's plumbing

| Today | Where | In the redesign |
|---|---|---|
| One run per process | `panels/base_panel.py:102-106`, `:221-225`; `streams.py:31-35`, `:272-276`; `core/runs.py:100-103` | Unchanged rule. The flag moves from `BasePanel._running` to the controller; its four outside readers (`screens/artifact_screen.py:587`, `screens/model_builder_screen.py:394`, `:510`, `main_window.py:283`) ask the controller; the fifth, the close question (`main_window.py:423`), is replaced (6.7) |
| `dispatch`: a fresh token, the worker kept in a set with auto-delete off, the payload released at the end | `base_panel.py:238-289` | Moved as it is into `RunController.dispatch`. `provide_fig_sink`, `provide_stream`, `on_chunk`, `watch_dir`, `on_result` keep their meaning and stay named parameters, never passed on to `fn`. `on_finished` (`:284-285`; its callers are `fdt_panel.py:324` and `crossval_panel.py:333`) becomes the controller's `run_ended` (6.2) |
| `Worker.run`: exactly one of result, error, cancelled, then finished; tracebacks dropped; a stop noted or too late | `worker.py:104-215` | Kept. One addition: it writes its outcome to the run's file (6.4) |
| The stream swap, the three channels, the 15 Hz pump; `vt.StreamRouter` | `streams.py:104-173`, `:176-321`; `vt.py:152-335` | Kept. The sink gains a file and stderr prints lose the warning level (6.4) |
| `CancelToken`: request, latch, owner thread; a stop deferred inside a commit and while an exception unwinds | `streams.py:48-80`; `core/runs.py:171-228` | Kept; the token gains `withdraw()` (6.6) |
| The progress pane | `widgets/progress_pane.py` | The widget goes: bar, spinner, plus-meter, sparkline (U2). Three of its rules stay as functions of the run's model: the row with the largest total drives, the solver's row left out (`:345-374`); the fit is found through sbi's counter row (`:376-389`); a moving solver count is a sign of life (`:296-314`) |
| The log pane | `widgets/log_pane.py:11-34` | Kept as the full-text view, hidden by default (6.4) |
| `RUN_STATE` | `base_panel.py:66-84`, `:318-321` | Becomes `RunStatus` (6.2), still emitted last |
| The yellow refusal box | `base_panel.py:406-426`; `widgets/refusal_box.py` | Its sentences stay; it is shown in the one question dialog (B2; 5.7) |
| The red "Error" box | `base_panel.py:428-457` | Goes for a run: a failure is a state of the run's page (6.8). A failure at a click outside a run is 5.7's |
| The close question | `main_window.py:416-445` | Replaced (6.7) |
| `config.QUIET_SEGMENT_BAR = True` at start-up | `app.py:72` | Stays |

New code is a package `core/gui/run/`: `controller.py`, `model.py` (Qt-free: readings, pace, the axis's four times), `readings.py`, `textfile.py`, `history.py`, `view.py`; the plan may rename the files. It draws with part 4's `TimeAxis`, `Marker`, `MessageRow`, `Ruled` and `Plot`, and every question it asks is part 4's `ask` (B2): this part keeps no `QMessageBox`.

### 6.2 The states of a run

| State | Entered when | Leaves for |
|---|---|---|
| Idle | nothing is live | Starting |
| Starting | `dispatch` is accepted: history file written, text file open (where kept, 6.5), lock on, Windows asked to stay awake, worker queued | Running, at the worker's first line or row; Ended, failed |
| Running, in one stage | first output; the shutdown reason is registered (6.10) | Stopping; Retrying; Ended |
| Needs you (shown over Running) | a difference beyond 5 points (6.3.5); a long silence (6.3.6) | cleared by the person's answer (6.3.5); a silence's by the next output (6.3.6) |
| Stopping | a stop is confirmed (6.6, 6.7, 6.10) | Running, when withdrawn; Ended |
| Retrying | a failure the window tries again (6.8); nothing computes | Running, next try; Ended, failed |
| Ended: finished, stopped or failed | the worker's `finished`, after `result`, `cancelled` or `error` (`worker.py:179-215`) | Acknowledged |
| Not seen to end | found at launch: a history file that still says "running" (6.5) | Acknowledged |
| Acknowledged | the shell calls `acknowledge(run)` (3.5) | Idle |

A run that only loads a record (its `spec` says so; 8.1) is published as loading: the foot shows "Loading…" (3.5), and it ends into Idle with nothing to acknowledge (C3; 6.5).

`dispatch(fn, *args, spec, on_result=None, **kwargs)`. `spec` holds what the page knows and the stage does not: the run's word ("Training") and name; its place `(part, page)`; its stage names in order, each with the engine line that begins it and, where one exists, the line that counts its progress (a training's: 6.3.1; a check's: 9.4, 9.6; Effective temperature's: 10.3; the card test, 9.7, and Live simulation, 10.4, report theirs by a call, 6.3.3); the kinds of record it writes and any record id known at the click; whether it has a simulation set to continue; the patience, the epoch limit and the "Expected epochs" (B9) it was started with. Each page part fills it; the plans part adds the history's `plan`, `settings` and `memory_limit` (7.1.11, 7.2.3).

Of the three things a page can get back (5.11), only `on_result` with the record changes what a step holds (B3); a refusal, a stop and a failure leave it. `run_ended` then makes every page read its facts again (5.4).

`RunStatus` is all that the foot, the chain and Home read (3.4 to 3.6): the run's word and name; its state, one of the above, and the word for it ("running", "loading", "stopping", "trying again"); the stage words; the reason when it needs someone; when it ended and what is kept; whether it is acknowledged; whether it can be continued, and from which batch (6.8); its place; and, for Records' "Being written" (10.1), its start time, the kinds of record it writes and the record ids it knows. The foot's titles and markers are 3.5's. `status_changed` fires at every change and at each save of the simulation set that is passed, so the shell runs no timer. A save is taken as passed when the count of batches done reaches a multiple of the cadence: the `[checkpoint] writing …` line names it, and a continued run, which prints none, takes the preview's `cadence` (`core/orchestrator.py:475`, `:539`). The engine may put a save off (`core/SBI/pipeline.py:1942-1944`), so this count is for display only; the batch a run can be continued from is read from the set itself, after the run has ended (6.8). `acknowledge(run)` writes into the run's history file; `where(run)` gives its page.

A `Refusal` from the stage crosses as the `error` signal's object (`worker.py:146-150`), and the page shows it (5.7; a narrowing round's refusal of a set one setting away as part 8's question, with the engine's sentence as it stands: C11, 8.6). One that comes before anything is spent is not a run: the state returns to Idle and the files of 6.4 and 6.5 are removed. That is tested, never assumed. Three things must hold: the solver's step count (`core/progress.py:45`, the count behind 6.1's sign of life) has not moved since `dispatch`; no stage after the run's first has begun; and the store lists no record of the kinds the run writes made since the start (`ArtifactStore.list`, `core/artifacts/store.py:681-735`). A training's own refusals pass, though they may follow the run's first line (`core/orchestrator.py:1071`, `:1110-1114`). Not every refusal does: the store's are refusals too (`store.py:92`), so the read-back of a posterior just trained can raise one after the whole fit (`orchestrator.py:1486`; `store.py:1374`, `:1381`), and an inference on recordings writes its observation before it can be refused (`orchestrator.py:2558-2559`, `:2200`). A refusal that fails the test ends the run as failed (6.8), its files kept. (The test: a choice the owner has not seen.)

### 6.3 The run page

`RunView` takes the place of regions 2 to 4 of the page that started the run (5.1); the heading stays. Modes: live; ended; past, drawn from the files of 6.4 and 6.5 alone (after a restart, and from Home's "Last runs"). A finished run hands its record to the page, which shows the result (5.9) and a link "How the run went". After a stop or a failure the view stays until "Back to the form". (Shown in the place of its own page, not on a page of its own: a choice the owner has not seen; it follows `toolkit-run-light.png`.)

Top right: "Show the full text" and the stop button. Smaller runs use the same view with the parts they have. Figures that arrive while a run is live (`provide_fig_sink`, `watch_dir`) show in the page's own viewer (5.9), which the view places under its messages and the result keeps.

#### 6.3.1 What the window reads

Every number comes from a line the engine already writes, read on the window's thread from the pump's batches. One table in `readings.py`; a line no row matches is text, never an error. Each row is pinned by a test: against the engine's own formatter where one exists (`_BatchProbeLedger.summary_line`), against a literal otherwise; E1's record has one literal copy in the window's suite and one in the engine's (11.2).

| Line | Built at | Feeds |
|---|---|---|
| `[budget] N batches x R rows = T training rows` | `core/orchestrator.py:1071` | totals |
| `Computing decorrelating Fisher rotation …`; `[fisher] eigenvalue spread …` | `orchestrator.py:1240`, `:1251` | rotation begins; done |
| `Reusing the Fisher rotation stored with the training checkpoint …` | `orchestrator.py:1218-1223` | rotation reused |
| `[tsnpe] basis: reusing the PARENT posterior's rotation …` | `orchestrator.py:1201-1204` | rotation is the parent's |
| `[checkpoint] writing to <dir> every N batches …`; `[checkpoint] resuming at batch k/n from <dir> …` | `core/SBI/pipeline.py:1568-1569`, `:1537-1538` | simulation begins; the simulation set's folder; `k` already saved |
| the progress row `Generating training data`, its `n/total […]` | `pipeline.py:1630-1631`; `vt.py:48-49`, `:135-149` | batches done |
| `[checkpoint] complete: n batches in <dir> …` | `pipeline.py:2039-2040` | simulation done |
| `[winsor] clipped …` | `core/SBI/summaries.py:191` | the fit is being prepared; logged only when something was clipped (`:190`), so no stage begins on it |
| `[fit] epoch n: training loss a, validation loss b, took s s, epochs since the best k` | E1 (11.2) | epoch, pace, loss figure |
| sbi's printed row `Training neural network. Epochs trained: N` | sbi base.py:1269-1274; `vt.py:289-300` | when the trainer gives no `[fit]` record (11.2's first edge): the epoch count and, stamped by the window at each change, the pace; then no loss figure |
| `[patho] batch k: …` (warning); `[patho] run total: …` | `pipeline.py:1926-1929`, `:2000-2003` | failed simulations |
| `<batch>: chi: d/n probes masked (…)` (a Python warning) | `core/SBI/chi_probes.py:335-338` | one grouped message; never summed |
| `[chi] masked probes: M of T (P%) over n batches, <scope>; …`, or `[chi] masked probes: none counted, <scope>` | `pipeline.py:224-236`, `:2004-2015` | 6.3.5 |
| `[mem] <batch>: peak allocated A GiB, peak reserved B GiB, f/t GiB reported free …, learned cap C` | `pipeline.py:294-298` | card memory |
| `… OOM at simulation batch N; retrying in chunks of …`; `… OOM with N rows OUTSIDE the simulator retry …` (warnings; `N` is a number of rows in both, the batch is named by the line's prefix) | `pipeline.py:825-826`, `:911-912` | batches that had to be split |
| `… batch FAILED after both halving retries (a/b) …` (error) | `pipeline.py:1839-1841` | a message; card memory "Look closer" |
| `[checkpoint] stopping: saving N completed batches …` | `pipeline.py:1981-1982` | the stopping sentence |

#### 6.3.2 The sentence

One sentence under the stage line, as `toolkit-run-light.png` pictures it, in the lettering face: part 4's `head` at 400 (4.3; C19).

| State | Sentence |
|---|---|
| Starting | "Training retrain_3 is starting." |
| Rotation | "The rotation is being computed before any simulation." |
| Simulation | "Simulating: batch 6,150 of 10,000, 9.7 s per batch; the simulations should be done by Fri 9 Oct, 22:00." |
| Fit, before its first epoch ends | "The simulations are done. The fit has started; its first epoch has not finished." and, when 6.3.3 has a pace, "At this size an epoch last took about 63 minutes." |
| Fit | "Now: Sun 11 Oct, 12:10, running 65 hours, epoch 62 of about 130." |
| Fit, past the estimate | "Epoch 152. The estimate was about 130. The fit ends by itself 20 epochs after its last improvement: not before Thu 15 Oct, 14:10." |
| Saving | "The fit ended after 131 epochs. The posterior is being saved." |
| Stopping, Retrying, Ended | 6.6, 6.8, 6.9 |

Closes W028, W170 and W175.

#### 6.3.3 The stage line and the time axis

**The stage line** shows the run's stages with the chain's markers and words ("Done", "Now", "To come"). A training's (U7):

| Stage | Begins | Its words |
|---|---|---|
| Rotation | `orchestrator.py:1240` | "Done"; "Reused from the saved simulation set" (`:1218`); "The parent posterior's" (`:1201`); "Off", from `spec`, because an unrotated run logs nothing (`:1255-1256`; W142) |
| Simulation | the two `[checkpoint]` lines, else the row's first frame | "Done, 20,480,000 simulations"; "Not needed: the saved simulations are used" |
| Fit | simulation done | "Now, about 63 minutes per epoch" |
| Saving | a `[fit]` record whose count equals the patience or whose epoch is the limit plus one; or sbi's row `Neural network successfully converged after N epochs.` (sbi base.py:1166-1174) | "Done" at `result` |

**The time axis** is part 4's `TimeAxis(start, now, end_from, end_by)` with its line of words (4.6), drawn for the stage that is running; a line under it names what follows ("Then the fit: about 5 to 6 days."). (Per stage, not per run: a choice the owner has not seen; it is what `toolkit-time-axis-light.png` shows.) Where no end is known it takes part 4's form for that (`end_by=None`), and the right-hand words are this part's, given below. All times are the window's clock, local.

Simulation. `n`, `M`: the row's two numbers. `p`: seconds per batch over the last 200 batches of this try. The end is firm (U11): `end_from` and `end_by` are both `now + (M − n) × p`, part 4's firm form, "Expected to end by Fri 9 Oct, 22:00". It is recomputed at every batch, so a simulation is never "later than expected". Until 20 batches are timed `p` is the history's estimate (below); with none, no end is known: "No end time yet: the first batches are being timed." (The 200 and the 20: a choice the owner has not seen.)

Fit. `n`: epochs finished. `p`: the median "took" of the last ten `[fit]` records. `s`: epochs since the best. `P`: the patience passed. `t_e`: when the newest record arrived. Start: the first record's time less its "took"; before that, the time simulation ended.

| What | Rule |
|---|---|
| Cannot end before | `t_e + (P − s) × p`: the fit stops by itself only `P` epochs after its last improvement (sbi base.py:1143-1153). With an epoch limit `L`, never later than epoch `L + 1` (sbi base.py:978; W035) |
| The expected epochs `E` (B9; a choice the owner has not seen) | "Expected epochs (for the time estimate only)" of the Training page, from `spec`; it never limits the fit. When blank: the epochs of the newest finished posterior in Records with the same model, kind of observation and narrowing state (`body.training.epochs_trained`, `orchestrator.py:1473`), read once at the start; part 8 exposes the resolved value (C28). The page says which: "about 130 epochs: the number given on the Training page" or "about 130 epochs: the last training of this kind, retrain_aug, ran 130" |
| The range | from the later of "cannot end before" and epoch `E − P`, to epoch `E + P`, each at `p` per epoch. (One patience each way: a choice the owner has not seen.) |
| No `E` | no end is known, and `end_from` is "cannot end before": "Cannot end before Tue 13 Oct, 09:30. No expected number of epochs was given, and there is no earlier training of this kind." |
| Before the first epoch ends | `p` is the history's estimate (below). With none, no end is known: "No end time yet: the first epoch has not finished." |
| Past the estimate | part 4's late form: the axis grows a day at a time, "Later than expected: …" |

Before a stage has timed itself, `p` is `history.estimate(…)`, which "Before you start" quotes too (8.3, 8.4): the median pace of the newest run of the same word and size (simulations per batch for a batch; network and number of simulations for an epoch); failing that, for an epoch, the timing check's newest result for the same network (9.6), scaled by the number of simulations as the runbook's projection is (`docs/guide/retrain.md:203`, `:539`). Its source is said in words.

Other runs. A run with a counted row gets the simulation rule on the driving row. One that prints none may call `RunView.set_count(done, total, unit)`, which the sentence shows (Live simulation's samples, 10.4), and `RunView.set_stage(name)` where it reports its stages itself (9.7, 10.4).

#### 6.3.4 Health figures

A three-rule table, "Training health": "Measure", "Now", "State". Shown for runs that simulate training rows.

| Measure | Number | When the engine prints it | Until then | State word |
|---|---|---|---|---|
| "Failed simulations" | the counts of the newest `[patho] batch` line (non-finite, constant, overflow, of rows simulated); the `run total` line replaces it | a `[patho] batch` line only in a batch where a new one appears; the total only when simulation ends, and only if this process simulated a row (`pipeline.py:1998`). Both count this process alone (`:1621`) | "None so far" | "Fine" up to 1 %, "Look closer" above (a choice the owner has not seen) |
| "Unusable driven recordings" | `M` and `T` of the total line | once, when simulation ends, also when the saved set was already complete (`pipeline.py:2004-2006`). Frequency response only (`:1614-1618`). The per-batch warnings are one sample each and are not added up: a split or repeated batch reports twice | "Counted when simulation ends" (a choice the owner has not seen) | 6.3.5 |
| "Graphics-card memory" | the largest "peak allocated" so far, "of" the card's total, then "limit 6 GiB" | at batch index 0 and every 250th batch (`pipeline.py:269`, `:1919-1920`), so a run continued at batch 237 prints first at 250; after each whole-batch failure (`:1864`); on a graphics card only (`:288`); each line covers the batches since the last (`:305-310`); nothing during the fit | "Not measured yet"; in the fit "during simulation" follows the number | "Fine"; "Look closer" once a batch had to be split or failed whole (a choice the owner has not seen) |

A continued run's failed count is "since batch 237". When every earlier part of the same simulation set has a history file, the page adds them and says "over 3 runs". (A choice the owner has not seen.)

#### 6.3.5 The comparison with the pre-run audit

When the first `[chi] masked probes` line of a training or a narrowing round arrives, the controller hands `M`, `T` and the scope to Checks' function (9.5), which returns the two shares, their difference `D`, the band and the sentences. The limits are Checks' named constants (B10). The row's word is the Checks list's for the same case (C16), except "Fine", where the check reads "Passed".

| Case; `D` in points | The row's state word | On the run's page | "Needs you" |
|---|---|---|---|
| `D` at most 3 | "Fine" | the row | no |
| above 3, at most 5 | "Look closer" | the row and one "Attention" message; the run is not interrupted | no. On the Checks list the check counts only after "Accept this difference" on its page (B1; a choice the owner has not seen) |
| above 5 | "Failed" | the row, an "Attention" message and 9.5's two buttons: "Stop training…" opens 6.6's question, the other keeps the fit | yes, until one is chosen. The window never stops the fit (U11) |
| no audit yet | "Waiting for the audit" | the row and one "Attention" message | no |
| a total that covers part of the set, or the engine's `none counted` form (`pipeline.py:231-232`) | "Not compared" | the row and one "Attention" message | no |

"Needs you" is announced by the rail's foot (3.5), Home, a taskbar flash (6.9) and the message. The answer beyond 5 points and the acceptance of the middle band go to the history with their time, the second also after the run has ended. (That the last two cases raise no "needs you": a choice the owner has not seen; 9.5 reads B1 so too.) With Checks, closes W034.

#### 6.3.6 Messages

A list of part 4's message rows, newest first, each "Attention, Sun 11 Oct, 12:10. …" with its marker.

- A message is raised by: a `core` record at warning or error; a Python warning; a notice of the window's (silence, 6.3.5, a try, a stop, the file). Prints are never messages.
- A message is the window's own sentence, in U7's words, wherever the window has one. For each warning or error row of 6.3.1, `readings.py` holds the sentence beside the pattern: "Failed simulations: 3 more in batch 6,150."; "Batch 6,150 did not fit in the graphics card's memory and was run in smaller pieces."; "Batch 237 failed for lack of graphics-card memory and is being run again after a pause."; and the one below. A warning or error that no row matches is an "Attention" message that begins "From the engine:" and gives the engine's sentence as it stands, in the engine's words; it is never hidden behind a count alone (C6). The engine's other lines show only in the full text. (These sentences, and quoting what is not recognised: a choice the owner has not seen.)
- Messages that differ only in digits are one message with a count: "Unusable driven recordings were reported in 6,150 batches (latest: batch 6,150, 9,120 of 22,528)." The page keeps the newest 50; all are in the full text. (A choice the owner has not seen.)
- Silence, in minutes or hours, never raw seconds (W118). In the fit: "Information. No output for 12 minutes. This is normal during an epoch."; it becomes "Attention", and the run needs you, once the silence is twice the epoch time. Elsewhere: "Attention" after 45 s with neither output nor solver steps (today's `STALL_S`, `progress_pane.py:69`); needs you after 10 minutes. The next line or solver step clears it, with no answer asked. (Both limits, and that it clears by itself: a choice the owner has not seen.)

#### 6.3.7 The loss figure

Shown during "Fit" and "Saving", from the `[fit]` records held in the model, drawn as part 4 gives it (4.9) in a `Plot`: a `matplotlib.figure.Figure` on an Agg canvas on the window's thread, never pyplot, so the worker's `plt.close("all")` (`worker.py:166-168`) cannot reach it. Before it is built, a test of stage 3 renders such a figure on the window's thread many times while another thread builds and saves a figure (13.2.2); if they can collide, the loss curve is drawn from its numbers by a painted widget and no figure library runs on the window's thread (ruling D7). It follows the theme (U23): it is redrawn once per record and on a change of appearance. Caption: "Loss by epoch. Loss is the negative log-probability; lower is better. Fitting ends when validation has not improved for 20 epochs.", with the patience passed. The best epoch is marked and named, "best so far: epoch 58" (`toolkit-sample.py.txt:1190-1197`): beside the sentence's epoch, that is where U11's epochs since the best are read. "Recent" starts at epoch 10 on a scale fitted to what it shows, with the note "epochs 1 to 9 are left out" (`toolkit-sample.py.txt:1168-1202`); it is greyed until epoch 12 (a choice the owner has not seen). The engine's own "Training loss" figure still arrives at the end, light in both themes, and belongs to the result (8.4).

### 6.4 The full text and its file

**The file.** Every finished line of a run that keeps files (6.5) is written, as it is produced, to `<run id>.txt`.

- Attachment: `redirect_streams(signals, cancel, line_sink=None)` hands `line_sink` to `_Pump`. In `_Pump.sink` (`streams.py:121-130`), branch "log" (`:128-129`), after the append and still under the pump's lock, it calls `line_sink(text, level)`. Checked: all three channels end there. Prints: `vt.py:181-186`. Records: `streams.py:251-258`. Warnings: `:296-297`. The degraded path: `:207`, `:216`. The only lines that bypass it are the worker's own emits (`worker.py:186-190`, `:202`; `streams.py:273`). It runs on the thread that wrote the line, so it needs neither the pump's thread nor a live window.
- The sink never raises, prints or logs, and is no stop point: it is reached while the engine logs inside `cancel_deferred()` (`pipeline.py:1971-1989`). On its first error it detaches itself and queues one warning line.
- `streams.py:289` passes level "err" in place of "warning", so a library's stderr print or a bar's last frame carries no warning mark (W066).
- `_Pump.sink` stops appending to `_logs` once the pump has stopped on a dead signal (`streams.py:152-155`); the file still gets the line (W116).
- `Worker(fn, *args, cancel=None, text=None, **kwargs)` writes the outcome to the file before it emits anything (`worker.py:179-205`): for a failure, "TypeName: message" (always the type, W065; on the page, 6.8) and the traceback it already formats (`worker.py:150`); for a stop and for a finish, one line; and the lines that bypass the sink. If the capture cannot start, the `RuntimeError` is caught where `redirect_streams` is entered and reported as a failure: "The run could not start: PRISM could not capture its output (…)". Today it is swallowed (`worker.py:207-210`; W064, W144).
- Lines: `2026-10-08 19:02:11 warning text`, local time with the date; levels `info`, `warning`, `error`, `err`, and `window` for the window's own lines. A traceback's lines are indented under their stamp. The first lines name the run and the memory limit in force with its source (7.2.3). Each try opens with a separator line. Flushed after every line; `os.fsync` at most every 30 s and at the end.
- Where (B8; a choice the owner has not seen): the folder `runs` beside the settings file, `Path(settings.settings().fileName()).parent / "runs"` (`settings.py:20-23`), with the history files. The tests' temporary settings file (`tests/conftest.py:132-167`) takes the folder with it. PRISM deletes nothing here except the two files of a start that was refused (6.2); the folder is the person's to clear (a choice the owner has not seen). The view offers "Open the file" and "Open its folder"; the path is not written on the page, being no file the person chose.

With 6.8 this closes W027 and W179.

**The view.** Hidden until "Show the full text"; then a `LogPane` under the page, in the fixed-width face (B19; 4.3), with "Hide the full text" and "Find…". It is fed by `log_batch` as today and holds the last 5,000 lines (`log_pane.py:15`), headed "Earlier lines are in the file." when cut. It follows the end only while it is at the end; today it always jumps (`log_pane.py:32-34`; W042, W063). It holds one run's lines (W067).

### 6.5 The run history

One JSON file per run, `<run id>.json`, beside its text file. The id is the local start time and the run's word, `20261008T190211-training`. Written by the window alone, through a temporary file and a rename: at the start, at each change of state or stage, each time a save of the simulation set is passed, at each epoch, at the end, at the acknowledgement, and when Checks records an answer or an acceptance (6.3.5). A run that only loads, a Live simulation and a video export write neither file; such a run ends straight into Idle, with no end to acknowledge and no flash. (C3; a choice the owner has not seen: it narrows U17's one file per run.)

| Field | Content |
|---|---|
| `id`, `word`, `name`, `place`, `records_root` | which run, where it shows, the records folder its run wrote to (ruling D5) |
| `started`, `ended`, `outcome`, `stage`, `acknowledged` | `outcome` is "running" until the end, or until a shutdown's stop is set (6.10) |
| `tries` | per try: start, end, outcome, kind of failure, batches saved before and after |
| `plan`, `settings`, `memory_limit` | as 7.1.11 and 7.2.3 supply them; `settings` holds every argument, enough to start the same run again |
| `simulation_set` | folder name; batches in all, at the start, at the end; simulations per batch; median seconds per batch |
| `health` | failed simulations; the unusable total, its scope, the audit's record id and reading, `D`, the band, the person's answer or acceptance and when; the largest memory peak; batches split; whether the run ended for lack of memory |
| `fit` | epochs, best epoch, best validation loss, median seconds per epoch, what ended it |
| `check` | for a check: which, at what sizes; for the two that write no record (timing of a short fit, the card test), the result as Checks gives it (9.6, 9.7) |
| `result` | the kind, id and name of each record written (a sweep study writes two, 10.3); whether one was saved without its name (E7) |
| `failure` | type, message, kind, whether PRISM must be reopened |
| `text` | the text file's name |

One PRISM runs at a time (U26, B17; the lock is shell 3.10's), so a file that still says "running" at launch is always a run that died, never one live elsewhere: the window then writes its `outcome` as "not seen to end", with its last write as the time last heard. The foot shows such a run as it shows a failed one (3.5); its sentence is 6.9's.

`history.py` offers `latest(n)`, `continuable(run)` (whether, by C2's rule, and from which batch: `peek` on the set's folder as 6.8 reads it, the run having ended), `estimate(…)` (6.3.3) and `runs_for(simulation_set)`. Readers: Home and the foot (3.5, 3.6); Records, to say "Being written" and to link a record or a simulation set to its runs (10.1, 10.2); Checks (9.3, 9.5); Settings and "Before you start" (the last run's memory, durations and pace; 7.2.2, 8.3, 8.4); `RunView` in past mode. The history is read for the records folder in use (ruling D5): Home, the first-time opening (3.6) and "Continue from batch N" (6.8) count only the files whose `records_root` is the one in use. The engine and the command line never read it. Closes W098 with 7.2.3.

### 6.6 Stopping

The button names the run: "Stop training…". It opens the question (`ask`) titled "Stop training?".

| Stage | Text | Answers; the first is the safe one, preselected, and Enter's and Escape's |
|---|---|---|
| Rotation | "Nothing has been simulated yet. The rotation will be computed again next time." | "Keep training"; "Stop training" |
| Simulation | "The 6,150 batches simulated so far are kept; the batch in progress is not. You can continue from batch 6,150." | "Keep training"; "Stop and keep the simulations" |
| Fit | "The simulations are kept. The fit so far, 62 epochs and 65 hours, will be lost and cannot be resumed." If the count since the best is one short of the patience: "This epoch may be the last: if validation does not improve, the fit ends by itself at about 11 Oct 2026, 14:10." | "Keep training"; "Stop and lose the fit" |
| Saving | the button is greyed: "The posterior is being saved; this cannot be stopped." | |

Only "Stop and lose the fit" is drawn as losing work (B27). (The wording outside the fit: a choice the owner has not seen.) Other runs take their words from their own parts (a check, 9.4; One cell and Sweep study, 10.3; the video export, 10.4; the Reduction map, 10.5), each saying what its stop loses (U17; W005, W123). Live simulation's stop does not ask, because nothing is lost (B24). The button is greyed as soon as the window sees the fit end, because a stop that lands between sbi's closing print (sbi base.py:1166-1174), the round bar's closing write (`core/SBI/train.py:254`) and the record's write (`orchestrator.py:1420`) still discards the fit (W172); the gap left is the pump's tick.

After a yes: `token.requested.set()`, as today (`base_panel.py:291-294`). The state is Stopping on the page, the foot and Home (W068): "Stopping: takes effect when this epoch ends, up to about an hour" in the fit, the bound being the measured epoch time; "Stopping: the finished batches are being saved." in simulation. A button "Don't stop after all" withdraws it.

**Withdrawing.** Today nothing clears the flag. A bare `requested.clear()` has one race: `check()` (`streams.py:74-80`) may have read the flag as set and not yet set `fired`; the window would say "withdrawn" while the run unwinds. The token gains a lock and

```python
def withdraw(self) -> bool:      # the window's thread
    with self._lock:
        if self.fired:
            return False
        self.requested.clear()
        return True
```

`check()` keeps its first, lock-free test and its owner-thread test (`streams.py:77-78`), then takes the lock, tests `requested` again, sets `fired`, releases, raises. `True`: "The stop was withdrawn. The training goes on." `False`: "Too late: the run had already begun to stop." A deferred stop only waits for the next check (`core/runs.py:171-199`), so clearing it loses nothing; the two other readers then see the flag clear and say nothing (`worker.py:157`, `:201`). A run that polls the flag itself (Live simulation, `panels/simulate_runner.py:219-220`) stops within a frame and gets no withdraw button.

Closes W005, W068, W123 and W172.

### 6.7 Closing the window

`closeEvent` while a run is live always ignores the event, then asks with `ask`; a shutdown is 6.10's. `BasePanel.request_cancel_all` (`base_panel.py:114-118`) goes.

> "Training retrain_3 is running." Then: "It has been running for 2 days 17 hours and is in its fit, at epoch 62. Stopping it now loses the fit, which cannot be continued; the 20,480,000 simulations are kept." Answers: "Keep running" (preselected; Escape) and "Stop the run and close".

The second sentence is 6.6's for the stage, and the second answer is drawn as losing work where 6.6's is. "Keep running" changes nothing. "Stop the run and close" sets the token and a flag; the window stays open on the run's page in the Stopping state, with "PRISM closes when the run has stopped." and "Don't stop after all", which withdraws both. When the worker reports finished, the history is written and the window closes itself. If a stop is already pending the question is "Training retrain_3 is stopping." with "Wait" (preselected) and "Close when it has stopped". (A choice the owner has not seen.) "Quit anyway" and "Cancel task & quit" go.

**Unsaved edits to a model (C8).** When the builder holds unsaved edits, a close also asks part 10's "Throw away the unsaved changes?" (10.6): after the question about a live run, if that answer would close the window, or alone when none is live. "Keep editing" (preselected; Escape) abandons the close and stops nothing. (That no run is stopped until both are answered: a choice the owner has not seen.)

An accepted close writes the window's size and place only (B6) and closes every other top-level widget of the application, so the event loop ends (W145, W197; 5.9 makes the figure windows the main window's own).

**Why nothing may run unseen.** In the experiment of the notes' part 2, a close with no stop left the process alive with the stage running on, and the figure hand-over raised `RuntimeError: Signal source has been deleted`. In PRISM that hand-over is inside the posterior's writer (`orchestrator.py:1420-1431`; `core/artifacts/store.py:404-411`), which removes its folder on any exception (`store.py:434-436`, `:479-481`), and the worker swallows the report (`worker.py:207-210`). Now the window cannot close before `finished`, so a stop during simulation has its rescue save behind it before the window goes, and 6.13 removes the raise. Closes W001, W162, W164, W173, W174, W178 and W193.

### 6.8 Failure, the retry and "Continue from batch N"

**The page.** Marker "Error", then in the book face what failed, what was saved, what was lost:

"Training failed on Sat 3 Oct, 03:12, during simulation: the graphics card ran out of memory at batch 237. Saved: the simulations up to batch 236. Lost: nothing else."

Then the error's message, quoted in the fixed-width face (B19); "The full error is in the run's file." with "Show the full error", which opens the error's type and traceback, the type appearing nowhere else in the display (C7); and the main button "Continue from batch 236". In the fit: "Lost: the fit, 41 epochs and 43 hours.", and the button reads "Fit again on the saved simulations" (C2; a choice the owner has not seen: with every batch saved, "from batch 10,000" would mislead). At "Saving": "The fit finished but its record could not be written. The network file may still be in the records folder; look there before clearing anything in Records." (W011 stays the engine's.) No dialog is shown.

**The retry** (U12). Below it sit the engine's ladders, untouched: two halvings and three whole-batch waits (`pipeline.py:767-845`, `:848-922`, `:1817-1877`; `core/config.py:320`, `:326`), then the rescue save (`:1951-1990`).

| Question | Answer |
|---|---|
| Which failures | those the engine calls out of memory, `pipeline._is_oom(exc)` (`pipeline.py:313-339`), on the exception the worker hands over (its chain is intact, `worker.py:18-29`), in a training or a narrowing round. Any other failure is reported at once, with "Continue from batch N" where it applies: a programming error or a full disk fails the same way again. (C5; a choice the owner has not seen.) |
| How many, how spaced | three tries, after pauses of 1, 3 and 10 minutes; a try that saves at least one further batch starts the count again. (A choice the owner has not seen.) |
| What a try is | the same function and arguments in a new worker with a new token, with `resume="require"` once the set holds a batch (`orchestrator.py:799`, `:1095-1096`); with none, the arguments as they were. It continues at the saved batch and reuses the stored rotation (`:1207-1238`; `pipeline.py:1513-1538`). A narrowing round redraws the same region, its draw being seeded (`core/SBI/truncate.py:348-366`) |
| What `N` is | `training_checkpoint.peek(…)["batches_done"]` (`core/SBI/training_checkpoint.py:162-184`) on the set's folder, named by the `[checkpoint]` line and reached through the store's `kind_dir("simulation")` (as `orchestrator.py:1089-1090` does). Read after the worker has ended, never while a run is live: the engine replaces that file at each save |
| On screen | state Retrying: "The graphics card ran out of memory at batch 237. Trying again in 3 minutes: the second of three tries. The simulations up to batch 236 are saved." Button "Stop trying". Same page, same history file, same text file |
| What detects "only reopening can clear this" | (a) a try ends out of memory and the set has gained no batch since the last failure; or (b) a device failure that is not out of memory: a `torch.AcceleratorError` in the chain with `_is_oom` false. Then no further try. (Both rules: a choice the owner has not seen.) |
| What it says then | the failure page, with "Trying again here did not help: this copy of PRISM still holds the graphics card's memory. Close PRISM and open it again; Home will offer 'Continue from batch 236'." |

Rule (a) bounds a useless retry to one pause: on 1 October nothing recovered inside the process and a new process with a limit did (`docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md:35-38`), and the learned limit is never cleared in a process (`pipeline.py:100`, `:119`; it only loosens by a tenth after 32 clean batches, `:145-147`).

**"Continue from batch N".** On the page of a failed run, and of a stopped one whose set holds a batch, it starts at once: the same call, with the name and note the form still holds (5.5) and `resume="require"`. It cannot simulate from zero, so U10's question does not arise; a set that no longer matches is refused before anything is spent (`orchestrator.py:1110-1112`; E6 names the setting). Home offers it (3.6), and after a reopening the run's page in its past mode too, by C2's rule, held in `continuable(run)`: for the newest training or narrowing round, among the runs of the records folder in use (ruling D5), that ended by a failure, or was not seen to end, whether or not its end was acknowledged, for as long as its simulation set is unfinished and no later run of that kind has started. When the failure was in the fit the button reads "Fit again on the saved simulations"; that set is complete, so the offer stays until a later run of that kind starts or the saved simulation set is removed (ruling D6). For a run the person stopped, Home's next step is the ordinary one: the Training page, whose "Before you start" says it continues the saved set (8.4). The press fills the earlier steps from the file's `settings` through part 7's `fill_forms` (7.1.7), applies them (8.1), and opens the run's own step: Training, with "Continue or start new" at "Only continue" (8.4), or Narrowing round. The main button reads "Continue from batch 236", from the preview's own count (5.4; a narrowing round's from the history, 8.1), and starts it with "Before you start" in view. A name and a note are never refilled, also not after a restart (C4): the page opens with both blank and the earlier name as a hint under "Name", "The earlier run was named retrain_3."; the same name may be typed, no record having taken it. No consent is replayed, because `resume="require"` needs none. (The same button on a stopped run's own page; after reopening, a name to type and one press; the hint's words: a choice the owner has not seen.)

### 6.9 The end

| Ended | Sentence |
|---|---|
| finished | "Training finished on Wed 14 Oct, 14:10, after 5 days 19 hours. The posterior retrain_3 is saved." When the record came back without the name asked for (E7, 11.8): "It was saved without its name: 'retrain_3' was taken while the run was going." (the result page offers "Rename…") |
| fit ended early | "The fit ended at epoch 84, earlier than the estimate of about 130: validation had not improved for 20 epochs (best at epoch 64)." |
| fit ended by its limit (`n = L + 1`, count since the best below `P`) | "The fit stopped at its epoch limit after 51 epochs, before it had stopped improving. The saved network is the last epoch's, not the best (epoch 49)." (W036) |
| stopped | "Training was stopped on Sun 11 Oct, 12:31, during the fit. The 20,480,000 simulations are kept; the fit, 62 epochs and 65 hours, is lost." |
| failed | 6.8 |
| the stop came too late (`worker.py:14-15`) | the finished sentence, then "A stop was asked for, but the run had already finished." |

No run prints "finished" after a stop or a failure (W120). At the end the controller writes the history, releases the lock, lets Windows sleep, withdraws the shutdown reason (6.10), publishes `RunStatus` (the foot and Home then say how and when until `acknowledge`), and calls `QApplication.alert(window)`, which flashes the taskbar button until the window is active. The same flash marks "needs you" (a choice the owner has not seen). After a restart the newest unacknowledged file of the records folder in use (6.5) fills `RunStatus`; a run not seen to end (6.5) reads "PRISM closed while this training was running; last heard Sat 3 Oct, 03:12, simulation, batch 6,150 of 10,000." Closes W062 and, with the shell, W038.

### 6.10 Keeping Windows awake, and a shutdown during a run

**Awake.** The controller calls part 7's `this_computer.keep_awake` (7.2.6) on the window's thread: on at Starting, off at Ended, and again when the switch changes during a run. The request stays through Retrying and Stopping.

**A shutdown or a sign-out (U27, B28).** When a run enters Running the controller registers a reason with Windows, on the window's thread (`ShutdownBlockReasonCreate`, through `ctypes`, on the main window's handle as `app_icon.py:91` takes it), and withdraws it at Ended (`ShutdownBlockReasonDestroy`). It stays through Stopping and Retrying; nothing is registered when no run is live. The reason is set again when the stage changes:

| Stage | Reason |
|---|---|
| Simulation | "Training retrain_3 is simulating. Shutting down stops it; the simulations done so far are saved first." |
| Fit, Saving | "Training retrain_3 is fitting. Shutting down loses the fit: nothing of it can be saved. The simulations are kept." |
| Rotation, and any other run | "<Run> <name> is running. Shutting down stops it.", then what its stop keeps (6.6; in rotation, "Nothing has been simulated yet.") |

On "shut down anyway" PRISM sets the token, as a confirmed stop does, writes the history as stopped, with "Windows shut down", and holds the shutdown for as long as Windows allows while the worker unwinds; during simulation that is the engine's own save of the finished batches (`pipeline.py:1971-1990`). A run ended this way counts as stopped by the person (ruling D8): its simulations are saved as far as Windows allowed (if Windows ends PRISM first, the simulation set holds what its last regular save wrote), and Home shows the ordinary next step, not "Continue from batch N" (6.8). On "cancel" nothing changes. PRISM asks no question of its own during a shutdown: 6.7's two are for a close by the person. The rule to build to, and to test with the session's end simulated: from the moment Windows asks, a `closeEvent` raises no dialog (the tests' dialog helper, B2, records none); the token is set only on "shut down anyway", once, and never on "cancel". (The reasons' wording: a choice the owner has not seen.) Whether the registered reason alone makes Windows wait, how the toolkit hands the session's end to the window, and how long Windows then allows are not established: a scratch test settles them in stage 3, before this is built.

### 6.11 The launcher

| File | Today | Redesign |
|---|---|---|
| `run.bat` | runs `python -m core.gui` in the batch file's console and ends in `pause` (`run.bat:9`, `:16`); closing that console ends PRISM and any run with no question (W139) | `start "PRISM" pythonw -m core.gui`, no `pause`; the line that sets `KMP_DUPLICATE_LIB_OK` (`run.bat:5`) stays, as does `run.sh:14`. The process has no console and outlives the one that launched it |
| `run.sh` | `exec python3 -m core.gui` (`run.sh:16`) | sets `PRISM_DETACHED=1` and starts `nohup python3 -m core.gui >/dev/null 2>&1 &` |
| `__main__.py` | forces the Agg backend, builds, runs (`:9-23`) | first of all, when `sys.stdout` or `sys.stderr` is `None` (as under `pythonw`) or `PRISM_DETACHED` is set, binds both to `window-log.txt` beside the settings file (in the system's temporary folder if that path cannot be had), appending, with a dated line per launch |

That file takes the three writes that expect a console (shell 3.10): `_library_record_sink` (`app.py:56-58`), `sys.__excepthook__` (`app.py:32`), `traceback.print_exc()` (`app.py:119`). A failure before the application object exists is written there and shown in a native message box through `ctypes`: "PRISM could not start. The details are in <path>." It is not the question dialog, because the toolkit may be what failed; no test imports `core.gui.__main__`, so the tests' dialog guard is not concerned. `python -m core.gui` from a console still works as now. That box is the one place this part writes a path on screen: nothing there can offer a button that opens the file. (The file, the variable and the box: a choice the owner has not seen.)

### 6.12 The lock

Today the whole controls column of every panel is disabled, the saved-run pickers of the two effective-temperature tabs with it (`base_panel.py:312-317`, `:333`; W077), because a model or picker changed mid-run acts under the running worker (`:323-333`). That does not cover reading.

From Starting to Ended, Retrying included:

| Disabled, with "A run is live. Only one runs at a time." | Usable |
|---|---|
| region 3 and the two buttons of region 4 on every working page (5.1): every setting, picker, plan choice, name, note, start, load and apply | the rail, every page, Home, Help, the keyboard |
| Settings: the memory limit (7.2.2); the planner reads it at every batch (`pipeline.py:584-603`, `:636`) | Settings: appearance, the keep-awake switch |
| Records: "Rename…", "Edit note", delete, clearing leftovers (today refused only after the click, `artifact_screen.py:580-590`; W087) | Records: every list, page, figure, log and the made-from view |
| Your own models: validate, save, delete (`model_builder_screen.py:394`, `:510`; `main_window.py:283`) | the builder's form |
| every other run's start, a new comparison of saved effective-temperature runs included: nothing computes beside a live run, and its button says it waits until the run ends (U25, B15; the words are 10.3's) | the run's own page; "Open larger" on a figure |
| | saved effective-temperature runs: their tables, record views and figures (region 2; 10.3) |

A page built while a run is live comes up locked (shell 3.10): it asks the controller when built. (Appearance and the keep-awake switch staying changeable, every other setting locked: C29; a choice the owner has not seen; appearance is so today, `main_window.py:216-231`.)

### 6.13 A display failure cannot remove a finished record

`_png_fig_sink` (`base_panel.py:44-63`) is called by the store after it has saved its own copy of the figure (`core/artifacts/store.py:407-409`), inside the writer. Its pickling is guarded (`:48-51`); its render and its emit are not (`:53-62`). New: the whole body is inside `except Exception`. A failure becomes one Python warning, itself guarded, "The figure 'Training loss' could not be shown (…). The record is not affected."; a warning is not a stop point (`streams.py:296-297`) and reaches the full text, the record's log and, the sentence being the window's own, the run's messages (6.3.6). `plt.close(fig)` stays in `finally`. `WorkerCancelled` is a `BaseException` and passes. With 6.7 this closes W185 and W192 for the display. A failure while the engine itself draws or saves the loss figure (`orchestrator.py:1429`; `store.py:407`) is guarded by E10 (11.8): on either front end the run then ends with its record and a warning (C22, a choice the owner has not seen; 6.3.6).

### Choices in this part the owner has not seen

Most consequential first.

- The window tries again by itself only after the graphics card runs out of memory in a training or a narrowing round, and reports any other failure at once; this narrows U12's "tries again by itself" to the one kind of failure a pause can cure.
- Its three tries follow pauses of 1, 3 and 10 minutes, and a try that saves at least one more batch starts the count again.
- It stops trying and says PRISM must be reopened when a try runs out of memory without saving a new batch, or when the card reports a failure that is not about memory.
- A run that only loads a record shows "Loading…" at the rail's foot, and it, a Live simulation and a video export leave no history file, no text file and no end to acknowledge; this narrows U17's one small file per run.
- A warning of the engine's that the window has no sentence for is shown on the run's page as an "Attention" message that begins "From the engine:" and gives the engine's sentence as it stands; every other message there is the window's own sentence.
- A live or just-ended run is shown in the place of the form that started it, not on a page of its own.
- After reopening, "Continue from batch N" fills and applies the earlier steps from the run's history and opens the run's own step, where one press starts it once a name is typed; the name and the note are not brought back, and the earlier name is shown as a hint under "Name".
- A stopped run whose simulation set holds a batch offers "Continue from batch N" on its own page, as a failed one does.
- After a failure during the fit the button reads "Fit again on the saved simulations", and stays on offer until a later run of that kind starts or the saved simulation set is removed.
- When the engine cannot draw or save its last loss figure, the run ends with its posterior and a warning instead of failing, from the window and from the command line alike (the change is the engine part's).
- A run's full text and its history are kept in a folder beside the settings file, per user, not in the records folder; PRISM deletes nothing there except the two files of a start that was refused.
- A refusal counts as a start that never was, and its two files go, only when no simulation step was taken, no later stage had begun and no record was written; any other refusal ends the run as failed.
- Closing the window asks about a model's unsaved edits after the question about a live run, and no run is stopped until both are answered.
- PRISM starts through the windowless Python; what it prints outside a run goes to a log file beside the settings file; the macOS and Linux launcher sets a new variable; a failure to start at all is shown in a plain Windows message.
- The fit's expected length is the "Expected epochs" typed on the Training page or, when blank, that of the last finished training of the same kind in Records, shown as a range one patience (20 epochs) either side; with neither, only "Cannot end before" is said.
- The time axis is drawn for the stage that is running, with a line saying what follows, not for the whole run.
- The simulation's end time comes from the pace of the last 200 batches, and from the last run of the same size until 20 batches are timed.
- In the comparison with the audit the run's row reads "Waiting for the audit" with no audit and "Not compared" with a partial total or nothing counted, and neither asks for the owner.
- A difference of 3 to 5 points counts as passed only after "Accept this difference" on the check's page; the run's history keeps it with its date.
- "Failed simulations" reads "Fine" up to 1 % and "Look closer" above; the unusable-driven-recordings row says "Counted when simulation ends" until then; "Graphics-card memory" reads "Look closer" once a batch had to be split or failed whole; a continued run's failed count says since which batch, or over how many runs.
- A silent run becomes "Attention" and "needs you" after twice the epoch time in the fit, and elsewhere "Attention" after 45 seconds and "needs you" after 10 minutes; the next output clears it without an answer, and the taskbar button flashes for "needs you" as it does for an end.
- Messages that differ only in their numbers are one message with a count, and the page keeps the newest 50.
- "Recent" on the loss figure is greyed until epoch 12.
- Outside the fit the stop question's second answer is "Stop training" (rotation) or "Stop and keep the simulations" (simulation); the stop button is greyed while the posterior is being saved.
- Closing the window while a stop is pending offers "Wait" (preselected) and "Close when it has stopped".
- The sentences Windows shows as PRISM's reason during a shutdown are this part's wording.
- Appearance and the keep-awake switch stay changeable during a run; every other setting is locked.
- The run's page's sentences while live, after a failure and at the end are this part's wording, written to U7, U11 and U12; only the fit's "Now: …" line, the stage words and the loss caption come from the approved pictures.

### What this part fixes

- W001: the close question names the run, how long it has run and what a stop loses; "Keep running" is preselected.
- W004: Windows is asked to stay awake during a run (with 7.2.6), and a sign-out or shutdown shows PRISM's reason and waits (U27); a shutdown Windows forces without asking still ends the run.
- W005: a stop asks first and says what is lost, for every kind of run that can lose work.
- W027: the traceback is in the run's file.
- W028: the first epoch of a fit says it has not finished, and why nothing is printed.
- W034: the unusable-recordings total is compared with the audit when simulation ends (with Checks).
- W036: the end says when the epoch limit, not convergence, ended the fit.
- W038: the end is said at the foot and on Home until acknowledged (with the shell).
- W042, W063: the full text follows the end only while it is at the end.
- W062: the end carries its time and duration, and the taskbar flashes.
- W064, W144: a run whose capture cannot start says so.
- W065: a failure's type is in the run's file and behind "Show the full error".
- W066: stderr prints carry no warning mark.
- W067: each run has its own page, text and, where kept, file.
- W068: a pending stop shows on the page, the foot and Home, and closing knows of it.
- W076, in part: closing asks before a model's unsaved edits are thrown away (with 10.6).
- W077, in part: saved effective-temperature runs can be browsed during a run; a new comparison waits until it ends (U25) and no second run can start.
- W098: the memory limit a run used is in its history and its text (with 7.2.3).
- W116: a stopped pump no longer fills an undrained list.
- W118: silence is given in minutes or hours.
- W120: no run says "finished" after a stop or a failure.
- W123: the stop and close wording follow the run that is live.
- W139: the launcher leaves no console window that can end PRISM.
- W142, in part: the stage line says when rotation is off.
- W145, W197: closing the main window closes every other PRISM window.
- W162, W173, W178: "Quit anyway" goes; the window cannot close under a live run.
- W164: both quit buttons go; closing waits for the run to unwind.
- W170: the fit shows its epoch, pace, loss and expected end (with E1).
- W172: a stop can be withdrawn, and the button is greyed once the fit has ended.
- W174: "Cancel task & quit" goes; the window stays until the run has stopped.
- W175: something moves during the fit besides a clock.
- W179: every run that makes a record leaves a dated file of its lines.
- W185: a figure that cannot be shown cannot remove the record.
- W192: as W185, for the display; a failure while the engine itself draws is E10's.
- W193: a stop during simulation waits for the rescue save before the window may close.
