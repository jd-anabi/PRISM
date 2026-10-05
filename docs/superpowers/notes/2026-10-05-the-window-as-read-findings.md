# The window as read for the redesign: every finding, as two readers left it

Appendix to `2026-10-05-the-window-as-read-for-the-redesign.md`. Generated on 2026-10-05 from the
readers' own structured returns; the wording is theirs and nothing was added. The code is as at
commit `71a53b5` (no file under `core/` or `tests/` has changed since `3ec5bf6`). Nothing here was
executed: every entry is a reading of the code, first by one reader and then by a second who was told
to refute it. "On screen" is the second reader's plain statement of what a person would experience.
The ids (W001 ...) are stable handles for the design, the plan and `docs/STATE.md`; they belong to the
working record and must not appear under `core/`, `tests/` or `docs/guide/`.

## A. Faults the first note does not list

### Can lose days of work (11)

- **W001.** The close dialog does not say what is running, for how long, or what would be lost, and offers no
  'cancel, wait until stopped, then quit'; no test pins it.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: Closing the window during a run brings up the same generic three-button question whatever is
  running, with no mention that a fit in progress would be thrown away, and the preselected button is the one
  that cancels it.
  Correction: Sequence that loses days: a training is in its fit; the window's X (or Alt+F4) is hit by accident;
  the dialog opens with 'Cancel task & quit' preselected (main_window.py:432-435); Enter or a click on it
  requests the cancel (:438-439), saves the selections and closes the window at once (:444-445); the fit is one
  training call with no checkpoint (core/SBI/train.py:365-370), so everything fitted so far is gone. The
  simulated rows stay in the cache.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/main_window.py:424-431 fixed title and text, the
  same for every run; the window holds the run's tab label and start time (:174-175, :369-370) and closeEvent
  (:416-445) reads neither. Three buttons only (:432-434). My grep of tests/ for 'Quit anyway', 'Cancel task',
  'still running', "Don't quit" found nothing; closeEvent appears in tests only in a docstring
  (tests/test_worker_dispatch.py:340).

- **W002.** The Bounds picker's own refresh button resets the selection to the folder's first file.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: Clicking the small rescan button beside the bounds list silently puts the list back on its first
  entry; if that goes unnoticed, the saved prior still loads and a training then runs for days on the wrong
  parameter set with no dialog.
  Correction: Nothing to correct in the finding (it is true of every file picker of this class, the Infer tab's
  Cell included). Sequence that reaches a spoiled multi-day run: Apply; the Prior tab's Bounds list shows the
  temperature box; click the small rescan button beside it; the list is rebuilt and falls to its first entry
  (the ordinary box, if the folder lists alphabetically); a saved prior is selected; press Build / Load prior.
  The prior loads with no refusal, because the prior load compares only the non-dimensional section and the
  three shipped Nadrowski bounds files are identical there, and the 'Config built' log line reads the same as it
  would for the temperature box (10 ND + 3 rescale, same mode). Posterior tab, '(from scratch)', Train: the
  cache key differs from the tier-1 cache in three fields, the one-setting dialog needs exactly one, so
  simulation starts from zero on the box without temperature. Passive signs only: the picker's text, the
  Posterior tab's 'starts a NEW run … differs in param_keys' line, and the absence of the tier-1 lines in the
  log.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:36 (button wired to refresh),
  :44-57 (clear and repopulate, nothing re-selected), against :181-219 (the store picker keeps its key).
  C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:310-315 says the same in a comment.
  C:/Users/J/PycharmProjects/PRISM/core/Helpers/file_manager.py:316-321 (order is os.walk's, unsorted).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:1275-1291 (prior load checks model, ND names, ND box,
  log mask only). C:/Users/J/PycharmProjects/PRISM/Resources/Bounds/nadrowski/master.txt:1-11 and
  master_tier1.txt:1-11 identical.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:216-219.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/identity.py:92-96.
  C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:347-350.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:214-216.

- **W003.** The boxes and pickers show pending choices, not what the session holds, and nothing marks the
  difference; the built config's bounds file, probe count, slots, units and rotation are on no permanent line.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: The boxes and lists can be changed after they have been applied and nothing shows that they no
  longer match the session, so a days-long run can use one bounds file while the Prior tab shows another.
  Correction: Nothing to correct. Sequence that reaches a spoiled multi-day run: Apply; on the Prior tab press
  Build / Load prior while the Bounds list shows the ordinary box (the config is built from it at that click);
  notice, change the Bounds list to the temperature box, and do not press the button again; Posterior tab,
  Train. The run uses the config built at the click for its whole length while the Prior tab shows the other
  file; no widget anywhere in the window reads the file the config was built from, and the 'Config built' line
  does not name it. The same holds for a Config box changed after Apply (for example the probe slots, which are
  frozen into the posterior): the run uses the values read at Apply. The posterior's record names the bounds
  file only afterwards.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:194-210 (picker read at the
  click only), :216-219. C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:115
  (training takes session.cfg). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:329-341
  (draft built at the click). C:/Users/J/PycharmProjects/PRISM/core/gui/session.py:27-37.
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/inference_screen.py:129-133 (session line: model and mode
  only). A search of C:/Users/J/PycharmProjects/PRISM/core/gui for 'sources' finds no reader of cfg.sources.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/provenance.py:102-111.

- **W004.** Nothing keeps the machine awake or handles a Windows sign-out or restart during a run.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: Nothing visible; if Windows sleeps, signs out or restarts during a run, the program does nothing to
  prevent it or to save anything first.
  Correction: The absence is confirmed for the whole program, command line included; what Windows then does
  cannot be read here. Sequence that would lose a fit: training left unattended -> Windows restarts for an
  update or the user is signed out (or the machine sleeps and the card does not resume cleanly; not established)
  -> the process ends during the fit, which is one call with no checkpoint (core/SBI/train.py:365-370) -> the
  whole fit is gone; simulated rows up to the last cache commit survive. Mitigating fact: the 63-hour console
  fit ran under the same absence, so the present power settings evidently let a long run go on.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] grep over core for SetThreadExecutionState,
  commitDataRequest, saveStateRequest, aboutToQuit, atexit: no match. core/gui/__main__.py:16-23 and
  core/gui/app.py:61-127 install only an excepthook and a logging sink.

- **W005.** One click on Cancel ends an hours-long sweep study or FDT run for good; the record is kept
  unfinished and nothing resumes it.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds, with a
  correction.
  On screen: A single stray click on Cancel stops the run at once with no 'are you sure', and it cannot be
  picked up again; on the training tab the same click throws away a fit that may have taken days.
  Correction: True as written, with two refinements. (1) For these two analyses the loss is hours, not days (the
  code itself calls the production sweep an overnight run), and what was measured stays on disk, but the window
  cannot use it: an unfinished record is not offered on its own tab and a comparison refuses an unfinished
  record. (2) The days-long case is the SAME button, because all ten panels share it. Exact sequence: Parameter
  Inference, Posterior tab, a training that has reached its fitting phase, one click on Cancel (no confirmation,
  the button then reads 'Cancelling...' and cannot be withdrawn). The cancel is raised at the next printed epoch
  line; the fit is one library call whose result exists only when it returns, and the posterior is written to
  the store only after that, so the whole fit is gone. The simulated rows in the cache are kept.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:150-152 (button wired straight to
  _request_cancel), 291-296 (sets the token, disables the button, no way back).
  C:/Users/J/PycharmProjects/PRISM/core/gui/streams.py:191-205 (every print is the cancel checkpoint, 'reaching
  even inside sbi's fit loop'). C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:434-472 (progressive
  kind keeps the folder, complete false).
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:147-149 ('There is no resume for these
  runs'). Grep for 'resume' under core/FDT finds only compare.py:28 ('nothing resumes it').
  C:/Users/J/PycharmProjects/PRISM/core/cli.py:447-453 ('a full overnight run').
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:197-198 (unfinished rows skipped).
  C:/Users/J/PycharmProjects/PRISM/core/FDT/compare.py:243-246 (unfinished record refused). Training:
  C:/Users/J/PycharmProjects/PRISM/core/SBI/train.py:366-370 (the fit is one infer.train call),
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1420-1421 (posterior written only after).

- **W006.** Deleting a finished simulation cache asks one generic Yes/No question that says nothing about what
  the cache holds.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: Asked to delete a finished training cache, the window shows the same short question as for any
  small record, with its id, its folder and 'cannot be undone', and no mention that it holds millions of
  simulated rows or how long they took.
  Correction: Narrower, and sharper for the owner's present disk. A finished cache that a posterior names cannot
  be deleted at all (the click is answered with the list of records that depend on it), so the generic question
  appears only for a finished cache with no posterior, which is the state the stopped fit left the 10,000-batch
  cache in. The table row does show its batches and row count; the question does not, calls it 'simulation
  (unnamed <digest>)', and defaults to No. The same generic question is all that stands before a trained
  posterior that nothing has been made from yet. Sequence that loses the cache: no run live in the window;
  Artifacts; Kind 'Simulation caches (training rows)'; select the row; Delete...; Yes.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:154-171 (kind-specific sentence
  only when complete and not finished; text is 'Delete {kind} {label}?'), :543-546 (Delete enabled on any row
  with a manifest), :682-716 (running check, dependents check, default No, then delete).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:81-83 and :891-907 (a cache's dependents are records
  whose parents name it), :918-947 (dependents, then the whole folder is removed), :148-150 (the listing carries
  batches and rows, not the wall time). C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1433-1434 (a
  posterior names its cache as a parent, and is written only after the fit, :1420).
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_table.py:100-109 (the row's Progress cell).

- **W007.** The delete question for an unfinished cache does not say the cache may be in use by a run in another
  process, though the FDT question does and Delete has no recency guard.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: The delete question for a half-built cache says its batches will be thrown away but not that a run
  elsewhere may still be writing it, while the question for a half-finished FDT record does give that warning.
  Correction: The wording gap is exactly as described. Followed further: in the window that owns the run Delete
  is refused, so the loss needs a second process, a second PRISM window or a terminal (the owner's rule forbids
  both during a long run). Sequence: a training is generating in window A; in window B: Artifacts; Simulation
  caches; select the unfinished cache; Delete...; Yes. Window A is not told and does not stop: its next save
  re-creates the folder holding only the new rows and a state file that claims every batch so far, its manifest
  update fails without a word, and at the end it still logs that the cache is complete and worth keeping. The
  earlier rows then exist only in memory, so the training can still finish, but any stop or failure of the fit
  loses the whole simulation and nothing can resume it. Deleting the finished cache from window B during the fit
  has the same end. Read, not run.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:128-132 (cache sentence, no
  word about a live run) against :133-136 (FDT sentence), :580-590 and :682 (the check reads this process's flag
  only). C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:918-947 (delete checks dependents, nothing
  about age or liveness). C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:371-372 (save
  re-creates the folder), :384-395 (writes only the rows since the last save and a state with the full count),
  :465-474 (manifest refresh reads the header and swallows any failure), :398-411 (completion does the same).
  C:/Users/J/PycharmProjects/PRISM/core/Helpers/file_manager.py:34 (every atomic write re-creates a missing
  parent). C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:2034-2041 (final save, 'complete ... keeping it
  lets you retrain' line, rows returned from memory).

- **W008.** The rotation on/off switch is a window-only setting that is remembered between launches; a
  remembered 'off' would train an unrotated posterior, which gate 4's diagnostic refuses.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: If the rotation box was unticked in an earlier session it opens unticked again, and a training
  started without noticing runs for days without the rotation, with no line in the log saying so.
  Correction: Holds, with four sharpenings. (1) The first reader's open point is settled: the rotation setting
  IS part of the cache key. An 'off' run therefore never reuses a rotated run's rows; it simulates from zero. If
  a committed cache exists that differs in this setting alone, the new-run question (default Cancel) fires and
  names it; with any second difference only the passive line shows. (2) Remembering the tick is deliberate: the
  tab's own description lists it among the selections it keeps. (3) A training with the rotation off says
  nothing about it: the only rotation lines are on the 'on' paths, and the line printed on Apply mentions the
  rotation only when it is on and chi mode is off. The finished record does keep the flag. (4) The default is
  on; 'off' is remembered only after someone unticked it and then closed the window normally. Sequence that
  costs days: untick 'Decorrelating Fisher rotation' once, for any test, and close the window; at a later launch
  press Apply without looking at the box, build or load the prior, and press Train when no rotated cache sits
  exactly one setting away. The run simulates and fits for days without the rotation, and the eigenvalue reading
  then refuses the result.
  Evidence: core/gui/panels/inference/config_tab.py:36-38 (docstring: the rotation tick is persisted), :89-90,
  :334, :343-351, :407, :422; core/gui/main_window.py:444-456 (settings written on an accepted close);
  core/config.py:404 (default True); core/cli.py:342 and core/tool/config_args.py:200-201 (the tool passes none;
  grep of core/tool for reparam_rotate finds nothing). core/artifacts/identity.py:98 (in the key);
  core/SBI/training_checkpoint.py:347-350 (one-field near miss). core/orchestrator.py:1116, :1218, :1240,
  :1255-1256 (the unrotated branch logs nothing). core/diagnostics/identifiability.py:64-69 (refusal).
  core/artifacts/manifest.py:282 (the record keeps the flag).

- **W009.** After an inference the TSNPE tab still points at the previously selected observation, while the
  Infer tab's line says the TSNPE tab can build a region around the new one; observations cannot be named, so
  they are told apart only by id and time.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: After an inference the narrowing tab still has the earlier observation selected, although the
  message just printed says it can work on the new one; pressing Run there starts a multi-day round around the
  old observation.
  Correction: Holds as written; one exception and the limits on the damage. Exception: when the store held no
  observation at all, the first one written does become the selection. Sequence that costs days: the narrowing
  tab's picker holds observation A (it is remembered across launches); run an inference on B; the inference tab
  prints that the narrowing tab can build a region around the new observation; open the narrowing tab and press
  Run without changing the picker. The round simulates and fits around A. What limits it: the picker and the
  line under it show A's label and time; the round's first lines name the observation; and the finished round
  records A, so the first inference on B is refused unless the override box is ticked. Nothing flags the
  mismatch before the days are spent.
  Evidence: core/gui/panels/inference/infer_tab.py:501-505, :509 (the session's observation is set, the line is
  printed, the gates refresh); core/gui/panels/inference/tsnpe_tab.py:134, :148 (the round reads the picker,
  never the session's observation), :205-208, :219, :226; core/gui/widgets/artifact_picker.py:181-183, :219,
  :249-254 (a refresh re-selects the previous entry; with none, the first row stays current).
  core/orchestrator.py:2528, :2558 (no name passed for the observation); :2645 (the round logs which observation
  it used); :2192-2202 (a later inference on another observation is refused).

- **W010.** The record's name is checked a second time at the very end, before the writer exists; if another
  process has taken the name during the run, the trained network is lost.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: Nothing from today's window; with a name given before the run, a second program taking that name
  while the run is going would make the finished training fail to save.
  Correction: Holds, but it cannot happen from today's window: the window sends no name, and an empty name is
  always free, so the end-of-run check does nothing there. It is real for a named run, which today means the
  command line, and it becomes reachable from the window the day a name box is added before the run. Sequence:
  start a training with a name; during the days it runs, a second process (another window's Save, or another
  command) gives a different posterior that name; when the fit ends the store refuses to open the record, and
  the trained network, held only in memory, is gone. Within one window it cannot happen, because every name
  control is locked while a run is live.
  Evidence: core/orchestrator.py:892 (first check, before anything is spent), :1420 (store.create at the end);
  core/artifacts/store.py:831-832 (an empty name returns at once), :846-850 (create re-checks before the writer
  exists); core/gui/panels/inference/posterior_tab.py:148-150 (no name sent); core/tool/stages.py:163 (the tool
  passes its --name); core/gui/panels/base_panel.py:316-317, :333 (controls locked during a run);
  core/artifacts/store.py:52-58 (no lock between processes).

- **W011.** If the final manifest write fails after the fit, the trained network is left in a folder with no
  manifest, listed as an incomplete leftover that the sweep may delete after five minutes.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: Only if the very last write fails for an outside reason: the run ends with an error, and the
  trained network sits in a folder the Artifacts screen lists as an incomplete leftover, which its sweep would
  delete.
  Correction: The mechanism holds; the one trigger the first reader suggested does not. A non-finite eigenvalue
  cannot reach the record: operating points with non-finite features or derivatives are dropped before the
  Fisher is formed and the rotation aborts if none is left; on a resume or a narrowing round the values come
  from a header or a parent record that were already finite. The other numbers in the record are each guarded or
  finite by construction. I found no value the training path can produce that fails the final check. What
  remains is a failure from outside at the very last step, such as a full disk or a file that cannot be written.
  Sequence: the fit ends, the network file is saved, the final record write fails; the folder stays with the
  network in it but no record, so nothing in PRISM can load it; the Artifacts screen lists it as an incomplete
  leftover, and its sweep, which is a deliberate click and never automatic, deletes it once it is five minutes
  old. Until that click the network file is still on disk.
  Evidence: core/artifacts/store.py:482 (the commit on a clean exit is not guarded), :544-550, :627-629;
  core/artifacts/manifest.py:120-129, :169-170 (non-finite numbers refused); core/orchestrator.py:1421 (network
  saved first), :1466 (eigenvalues recorded as given), :1474-1475 (two values guarded), :1227-1228;
  core/SBI/decorrelate.py:282-283, :293, :328-334 (non-finite points dropped; all-failed aborts);
  core/SBI/truncate.py:186-190, :496; core/artifacts/store.py:51, :58, :1013-1034 (the sweep's rule and its
  five-minute guard); core/gui/screens/artifact_screen.py:889 and core/tool/browse.py:478 (the only callers of
  the sweep).

### Can mislead a long run (26)

- **W012.** What the inference session holds (config, prior, posterior, observation) is shown nowhere in the
  shell or on Home; only as one line on the Config tab and in the Apply dialog.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: To see which prior, posterior and observation the session really holds you have to go to the Config
  tab; Home, the header and the other tabs do not say, so a training can be started against a prior other than
  the one the Prior tab is showing.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/screens/inference_screen.py:108-140 is the one
  describer; its only callers in core are core/gui/panels/inference/config_tab.py:285 (the Apply dialog) and
  :376 (the Config tab's line, :362-387), by grep. Home is a greeting and five tiles
  (core/gui/screens/home_screen.py:39-65); the header is title, back arrow and run slot
  (core/gui/screens/nav_shell.py:60-89). Elsewhere the held prior and posterior are named only in log lines at
  load time (prior_tab.py:317, posterior_tab.py:296); the other tabs' lines describe the picker entry
  (posterior_tab.py:319-322, tsnpe_tab.py:203). The tab enabling (inference_screen.py:195-204) shows only that
  something is held, not which.

- **W013.** Build / Load prior releases the session's posterior, observation and prior with no question and no
  line, although Apply asks first.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: After a fit has finished, one press of the Prior tab's main button greys the Validate, Infer and
  TSNPE tabs without asking; the trained result is still on disk but is not in the posterior list until that
  list is rescanned, and pressing Train there would fit again from the start.
  Correction: True as written; two additions. (a) The release and the new config are installed BEFORE the worker
  starts, so a load that is then refused, or a build that fails or is cancelled, has already released all three;
  the code comment that 'a refused click leaves the session exactly as it found it' covers only the click-time
  checks. (b) Nothing leaves the disk, but the Posterior tab's list is not refreshed after a fit, so the
  just-trained posterior is not offered there until the list's rescan button or Save is used; the tab shows
  '(from scratch)', and pressing Train in that state refits on the complete cache with no dialog. The visible
  signs of the release are the Validate, Infer and TSNPE tabs greying and the Config tab's session line
  shortening; no log line.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:209
  reset_downstream('prior'), :210 install_config, both before the dispatch at :244; :177-181 the comment; no
  dialog anywhere in :168-245. C:/Users/J/PycharmProjects/PRISM/core/gui/session.py:52-57 sets inf_prior,
  observation and posterior to None.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:263-303, 339-340 is Apply's question.
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1420 and :1486 write then reload the posterior; :443-444
  reports no near miss when the run's own cache already holds batches.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:287-298 (no list refresh after a
  fit; only Save refreshes, :315-316), :204-216.
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/inference_screen.py:192-204.

- **W014.** The Bounds picker says nothing about the file it shows, and the mode is learned only from a log line
  after the button is pressed.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: The bounds list shows file names only; the log line that appears after the button is pressed gives
  parameter counts and the mode, and reads the same for the ordinary box and the temperature box.
  Correction: Three points. (1) Before the press, 'Edit values' does lay the file out under Non-dimensional /
  Dimensional / Forcing headings, so its content can be inspected, though nothing names the mode. (2) With the
  multi-frequency option ticked the mode is 'chi' whatever the bounds file, so what the file decides is the
  parameter set, and the log line gives that only as counts: the ordinary box and the temperature box both read
  '10 ND + 3 rescale params' with the same mode, and the line never names the file. So the line is weaker than
  the finding assumes. (3) After the press the Config tab's session line also carries the model and the mode, so
  the log line is not the only place.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:75 (no summary row for
  Bounds; the Prior row has one at :82-85), :216-219 (the line: model, counts, mode; no file name), written
  after :209-210 and before the dispatch at :244.
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:51-57.
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/param_grid.py:101-113.
  C:/Users/J/PycharmProjects/PRISM/core/sim_config.py:391-393.
  C:/Users/J/PycharmProjects/PRISM/Resources/Bounds/nadrowski/master.txt:13-16 and master_tier1.txt:13-16 (three
  dimensional parameters each, both with a Forcing section).
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/inference_screen.py:129-133.

- **W015.** After a from-scratch build the Prior picker still shows '+ (from scratch)', so a second press builds
  another prior and a close remembers an empty pick.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: After building a new prior the list still says from scratch, the new prior is not in the list, a
  second press builds yet another one, and the next launch opens on from scratch again.
  Correction: True, and the new prior is not even in the list until the rescan button or Save is used. The route
  from the remembered empty pick to a lost run is partly guarded: a prior rebuilt at the next launch changes
  only the prior's fingerprint in the cache key, which is the one-setting case the Posterior tab's 'This starts
  a NEW run' dialog catches (default Cancel) when the earlier cache exists, and Validate refuses a prior that is
  not the posterior's. The guard fails when anything else differs as well, for example the bounds file having
  jumped.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:315-319 (no picker refresh
  or re-select), :336-337 (only Save does both), :362 (writes '' for the from-scratch entry).
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:229-231, 249-251.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:183-216.
  C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:347-350.
  C:/Users/J/PycharmProjects/PRISM/core/SBI/run_guards.py:64-82.

- **W016.** A blank, half-typed or negative VRAM ceiling is silently 0 (off); it is the one box on these tabs
  where blank is not refused.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: Clearing the ceiling box, or leaving a half-typed or negative number in it, turns the ceiling off
  without a question; only the small note underneath changes to say Off.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:148 max(0.0,
  self.vram_ceiling.value()), applied on every keystroke (:117); :158-161 the note then reads 'Off: …'.
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/labeled_inputs.py:25-29 value() returns 0.0 for text that
  does not parse. Every other numeric box on the two tabs goes through value_or_none and a rule:
  config_tab.py:230-250, C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:261-262.

- **W017.** Typing a VRAM ceiling does not update the Posterior tab's 'planner budget right now' line; it
  refreshes only on a budget keystroke or after a stage.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: After typing a ceiling in reaction to the Posterior tab's memory line, that line still shows the
  old budget until a budget box is edited or a button that rebuilds the session is pressed.
  Correction: The line is also recomputed by every Apply, by every press of Build / Load prior and whenever any
  run ends, because each of those re-runs the gates. It is stale only when the ceiling is typed after the last
  of those, which is the natural case of going back to Config in reaction to that very line. The run itself uses
  the typed ceiling at once.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:117 (the box is wired to
  _apply_vram_ceiling only). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:81,
  :112, :338. C:/Users/J/PycharmProjects/PRISM/core/gui/screens/inference_screen.py:165, :181, :217-218.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:333-335.
  C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:635-639 (the budget includes the ceiling).

- **W018.** The window's own help and note advise leaving the ceiling off ('off is right on an idle card', 'only
  if you need the desktop').
  Area: the Config and Prior tabs; second reader: holds.
  On screen: The help bubble and the note under the ceiling box both say that leaving it off is right on an idle
  card, which is the setting under which the full-size run of 1 October ran out of memory.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/help_text.py:47-48 ('0 = off, and off is
  right on an idle card'), while :53-54 of the same entry says to set it to about the free figure minus 1 GiB.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:159-161. That the advice is wrong at
  full size rests on the run record, not on code:
  C:/Users/J/PycharmProjects/PRISM/docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md:35
  (died at batch 237, no ceiling set) and :71 (desktop and context held about 1.3 GiB). The program's own
  out-of-memory message recommends a ceiling: C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:1870-1875.

- **W019.** With an unparseable PRISM_VRAM_CEILING_GIB the note says the variable 'is set and OVERRIDES this
  field' while the planner ignores it and uses the field.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: If the environment setting for the ceiling is mistyped, the note says it overrides the box and is
  planning to the box's own figure, 0.00 when the box is at 0, so the run goes ahead with no ceiling while the
  note claims one is in force.
  Correction: True; the consequence is worth more than a docs label. The figure the note prints after 'planning
  to' is then the box's own value, so with the box at its opening 0 the note reads 'OVERRIDES this field —
  planning to 0.00 GiB' and the ceiling is in fact off. Setting this variable is the documented way to carry a
  ceiling across launches, so a mistyped value (for example a comma for the decimal point) leaves a full-size
  run uncapped under a note that claims an override. The planner also logs a warning on every call, i.e. at
  every batch plan. Narrow: it needs a mistyped variable.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:149-157 (the test is only
  that the variable is non-blank; 'effective' comes from the planner).
  C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:589-600 (ValueError: warn, then fall through to the
  config constant the box has just written).

- **W020.** With χ on, the 'Model applied' line never says whether the Fisher rotation is on.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: With the multi-frequency option ticked, the line written after Apply lists the probe settings and
  says nothing about whether the rotation is on.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:342-351: the rotation text
  is in an elif after the χ branch. The rotation tick is restored from the settings file at launch (:422) and is
  part of the cache key (C:/Users/J/PycharmProjects/PRISM/core/artifacts/identity.py:98), and this line is the
  only echo of what Apply read.

- **W021.** A days-long training starts on one click with no confirmation; after it finishes the picker still
  shows '+ (from scratch)', so a second click starts another full fit and first drops the session's posterior.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: When a training ends the selection box still reads 'from scratch', and pressing the same button
  again starts a second multi-day fit at once without asking; the first result stays on disk, but the window
  lets go of it and the later tabs go grey until it is picked and loaded again.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/. core/gui/panels/inference/posterior_tab.py:51 one
  fixed-label button; :114-150 the click asks nothing unless _confirm_fresh_run finds a near miss;
  core/orchestrator.py:443-444 returns no near miss when this budget's own cache holds batches, so
  posterior_tab.py:214-215 returns (True, False). posterior_tab.py:142 drops the held posterior before the
  dispatch at :148. After a run neither _on_posterior (:287-298) nor refresh_local_gates (:332-338) re-lists the
  picker; the only re-lists are Save (:315), the picker's own refresh button
  (core/gui/widgets/artifact_picker.py:170) and a change made on the Artifacts screen
  (core/gui/main_window.py:124, :331-344). The second run takes the resume branch on the complete cache and
  refits (orchestrator.py:1095-1096, :1217-1223). Not a loss: the first posterior was written at completion
  (orchestrator.py:1420) and stays on disk, and the line under the budget boxes does say 'Resumes a COMPLETE
  checkpoint ... only the flow retrained' (core/gui/panels/inference/base.py:208-210). I found no route here to
  losing the trained result.

- **W022.** Validate, Infer and TSNPE never show which posterior and prior they will act on; after a TSNPE round
  the session silently holds the narrowed posterior while the Posterior picker still shows the parent.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds, with a correction.
  On screen: On the Validate, Infer and TSNPE tabs nothing says which posterior and prior the button will use,
  and after a narrowing round the window is working with the narrowed result while the Posterior tab's selection
  box still shows something else.
  Correction: 'Silently' is too strong and 'still shows the parent' is narrower. When a round ends the TSNPE
  tab's own log prints one warning line saying the posterior is now non-amortized, the Infer tab's 'Run on a
  different observation' box becomes available, and the Config tab's session line names the posterior held (by
  name or id, not as narrowed). The Posterior picker shows the parent only if the parent was loaded from it or
  named with Save; after a from-scratch training it still reads '+ (from scratch)'. What holds in full: no tab
  states, where the button is, which posterior and prior the run will use.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/gui/. panels/inference/validate_tab.py:33-46,
  infer_tab.py:80-160, tsnpe_tab.py:47-104: no label of the held posterior or prior. posterior_tab.py:319-322:
  the line under the picker is the picker's own selection_summary. tsnpe_tab.py:172-177 replaces
  session.posterior and logs one warning line. infer_tab.py:521-526 enables the other-observation box for a
  narrowed posterior. screens/inference_screen.py:127-140 with panels/inference/config_tab.py:99-100, :362-387:
  the session line. posterior_tab.py:300-307: Save then renames whatever the session holds, i.e. the round.

- **W023.** The near-miss dialog and the passive line speak in cache directory codes and identity keys, never
  the on-screen control; the dialog cuts both values at 60 characters so two long lists can print as identical
  lines.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds, with a correction.
  On screen: The warning meant to stop an accidental restart from zero names the other run by a 12-character
  folder code and the setting by an internal word such as 'n_runs' instead of 'Batches', so the owner has to
  translate it before knowing which box to change.
  Correction: The internal-names part holds as written. The identical-lines part is true of the code but I could
  not reach it from the window: the long lists are the non-dimensional limits and the parameter order, and a
  prior loaded against a changed non-dimensional box or order is refused, while a rebuilt prior changes the
  prior key as well, which makes two differences and therefore no dialog. The lists that can differ on their own
  (the three rescale limits, the time-scale limits, the chi band) are short, and the usual one-setting keys
  (batch count, rows per batch, prior fingerprint) are short too.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/. core/gui/panels/inference/posterior_tab.py:220-222
  prints r['name'], r['field'] and str(value)[:60]; core/SBI/training_checkpoint.py:347-350 name = directory
  name (12-hex digest, :110-113, :121-125), field = identity key; :294-300 the passive line;
  core/gui/panels/inference/base.py:213-215. core/artifacts/identity.py:92-96 the list-valued keys.
  core/artifacts/store.py:1275-1291 load_prior refuses a different model, non-dimensional parameter set/order,
  box or log mask, and checks nothing about the rescale limits.

- **W024.** The memory line is not recomputed when the VRAM ceiling is changed on the Config tab, and a 'does
  NOT fit' verdict neither blocks the click nor asks.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds, with a correction.
  On screen: After setting a ceiling on the Config tab, the memory sentence on the Posterior tab still shows the
  old budget and may still say the batch 'fits in one piece' although the run will now split batches and take
  longer.
  Correction: First half holds: typing a ceiling updates only the Config tab's own note; the Posterior and TSNPE
  lines keep the budget from their last refresh (a keystroke in a budget box, or the end of any stage). The line
  is recomputed at the Train click itself, but the run starts in the same instant. Second half is by design
  rather than a fault: 'does NOT fit' means the planner will split the batch (slower), which is what the line
  says, not that the run will fail.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/panels/inference/config_tab.py:117 and
  :128-165: the handler assigns config.SIM_VRAM_CEILING_GIB (:148) and sets vram_note only; no call to
  refresh_gates. SBI/pipeline.py:584-603, :635-639: the budget reads the ceiling live, so a recompute would
  reflect it. gui/panels/inference/base.py:108-142 and posterior_tab.py:80-81, :338, tsnpe_tab.py:94-95, :211:
  the only triggers. posterior_tab.py:143 refresh at the click, :148 dispatch. base.py:177-178 verdict wording;
  pipeline.py:667-683 the split.

- **W025.** (suspected) The 'Worst-case peak ... fits in one piece' line is the planner's cost model for one
  simulation chunk and would have reassured about the uncapped run that died.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds, with a correction.
  On screen: Before a full-size run with no ceiling the window would say the worst batch fits in one piece,
  although a run with exactly those settings ran out of card memory within half an hour on 1 October.
  Correction: Followed to the end in code, the mechanism holds. The figure on the line is the cost model of ONE
  solver call: the solution buffer, one drive, and the larger of the segment buffer and the kept copy. In chi
  mode the drive for the whole row block is built before that call, outside the planner's chunking, and building
  it peaks at four tensors of its size; the pipeline's own after-a-failure accounting charges that fourfold
  cost, the line's formula does not. The runbook's 11.44 GiB is exactly this formula at 2,048 rows x 300,000
  steps (5 x 2,048 x 300,000 x 4 bytes), so on an idle card with no ceiling the line reads 'fits in one piece'.
  Limits: that the window would have shown this on 1 October is inferred from that match, not observed; the
  batch that died was shorter than the worst case, so the death is not explained by the line's own thin margin;
  and the line does add that its budget is an upper bound. No days are lost through this: generation commits
  every 50 batches and rescue-saves.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/. core/orchestrator.py:520-532 need =
  pipeline.peak_sim_elements(width, n_fine, steady, n_vars, n_ch, 1), have = sim_memory_budget_elements.
  core/SBI/pipeline.py:562-578 the formula (drive counted once at :577); :67-75 the build peaks at 4x the
  returned tensor; :1775-1787 the per-row accounting that charges 4x and says the chi builder runs again per
  probe. core/SBI/chi_probes.py:259-263 the whole-block drive is built, :273-276 only then handed to gen_obs.
  core/gui/panels/inference/base.py:176-185 the sentence and its 'UPPER bound' caveat. docs/guide/retrain.md:323
  '11.44 GiB against ... 12.48 GiB'.
  docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md:35, :62-74 the death at batch
  237 and the measured peaks (a record, not code).

- **W026.** The TSNPE tab's budget is its own remembered pair opening at 5000 batches, and nothing on the tab
  states the network size, learning rate or patience the round will use (always config.py's 128 x 8); the
  network cannot be typed there at all.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: A narrowing round started from the window trains a 128 x 8 network at whatever batch count the
  TSNPE tab last held, whatever the retrain used, and nothing on the tab says so before the button.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/panels/inference/tsnpe_tab.py:84-85 the
  boxes seed from config; :218-221, :225-230 they are saved and restored under the tab's own group, so they open
  at the last TSNPE budget, or at config.py's 5000 (config.py:282) when none was saved, never at the Posterior
  tab's; :157-161 the dispatch passes only directions, level, batches, rows cap and the consent.
  orchestrator.py:2648-2655 a knob is forwarded only when given; :936-939 the stage then resolves hidden
  features, transforms, learning rate and patience to config.py's (config.py:363-364, :368-369);
  orchestrator.py:2653 train_new=True, so it is a fresh flow of that size. The size is recorded in the manifest
  at the end (orchestrator.py:1450, :1471).

- **W027.** A crash's traceback exists only inside the modal Error box and is gone once the box is dismissed.
  Area: how a run is started, shown, cancelled and closed; second reader: holds.
  On screen: When a run fails, a box shows a one-line message; the full error trail is behind its Details button
  and cannot be got back once the box is closed.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/base_panel.py:446-457: the pane gets only
  str(exception) (:447), the traceback goes only to setDetailedText (:452-453), then box.exec().
  core/gui/worker.py:150 formats it and :205 hands it to that slot; nothing prints or logs it (grep over core
  for traceback.print_exc and .exception(: only core/tool/__init__.py:208 and core/gui/app.py:119, neither on
  the worker path). core/artifacts/store.py:479-480 removes a failed posterior's folder, log included. Narrower
  for FDT/sweep records: the folder is kept and its log ends with one stamped line naming the exception's type
  and message (store.py:460-461, :567-570), still no traceback.

- **W028.** For the whole first epoch of a fit, and the data preparation before it, the progress pane says
  nothing: empty caption, indeterminate bar, solver idle, then the stall notice.
  Area: how a run is started, shown, cancelled and closed; second reader: holds.
  On screen: For roughly the first hour of training the progress area shows a moving bar with no text, a solver
  reading of idle and then 'no output for N s', which looks exactly like a hung run.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/SBI/train.py:254 opens the only bar live during the
  fit, total 1. core/gui/widgets/progress_pane.py:366-369 leaves the bar indeterminate when no row has a total
  above 1; :388-389 drops a total-1 row from the caption. sbi prints its counter only after an epoch:
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/sbi/inference/trainers/base.py:997-998 and :1269-1274;
  the loop body :981-995 prints nothing. Nothing heartbeats meanwhile, so progress_pane.py:333-339 shows the
  stall text after 45 s (:69) and :313-314 sets the solver to idle after 8 s (:60). tqdm's monitor does not
  repaint this bar (site-packages/tqdm/_monitor.py:76-84 requires miniters > 1). With a complete cache the
  generation loop (core/SBI/pipeline.py:1630-1631) has nothing to iterate, so the run goes almost straight to
  this state. The 'about an hour' is the first reader's measurement; it cannot be read from code.

- **W029.** The cell picker's rescan button resets the selection to the first file; on the sweep tab that also
  refills both grids from that first cell.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Press the small rescan button beside a cell list and the choice silently jumps back to the first
  file in the folder; on the sweep tab the two grids are rewritten for that other cell, so the next hours-long
  run can be on a cell nobody chose.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:36 (button wired to refresh),
  44-57 (clear and refill, the selection is never put back) against 181-183 and 219 (the store picker does put
  it back). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:114 and 234-258 (an index change
  rewrites the 'Cell values' label and both grids' min and max; the point counts are left alone).
  C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:311-315 states the same in a comment ('resets them to
  their first entry'). The same picker class is used at simulate_panel.py:75, reduction_panel.py:52,
  fdt_panel.py:127, inference/base.py:36 and for the bounds file at inference/prior_tab.py:70; what a reset
  bounds file does downstream I did not follow.

- **W030.** A calibration or an inference in flight in this window is listed as a broken folder that nothing can
  load.
  Area: the Artifacts screen; second reader: holds.
  On screen: While a calibration or an inference is running, its row on the Artifacts screen shows a folder
  name, no date and a reason saying it is incomplete or interrupted, and the pane calls it an incomplete
  directory that nothing can load.
  Evidence: Read, not run. C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1950 (the calibration's writer
  opens before the set is simulated, :1969-1980) and :2227 (the inference's opens before sampling and the
  predictive simulations). C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:414-415 (the folder is
  created on entry), :627-628 (manifest only at commit), :667-668 and :689-690 (a folder without one is listed
  as incomplete with the fixed reason).
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_table.py:152-154 (row = folder name, blank date,
  the reason). C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:373-379 (pane text),
  :337-338 (counted as incomplete). By contrast a prior's and a posterior's writers open only for the final
  write (orchestrator.py:726, :1420).

- **W031.** The sweep dialog promises nothing still being written is removed, but the guard is only 'written in
  the last 300 seconds', which a calibration's idle folder outlives.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: The sweep's confirmation says nothing still being written is removed, yet a calibration that
  another window started more than five minutes ago is listed in it and, on Yes, its folder is removed and that
  calibration later fails.
  Correction: Confirmed by reading, and narrower than it sounds. In the window that owns the run every sweep is
  refused, so this needs a second process (a second window or a terminal). The first file to land in a
  calibration's folder is its first figure, written after the calibration set is simulated and the rank test has
  run (200 simulation batches by default), so for all that time the folder's newest date is its creation and
  after five minutes it passes the age test and is removed. The run does not survive: its first figure fails
  because the folder is gone, the writer gives up, and the calibration is lost after the simulations were spent.
  An inference is exposed the same way between its first figure and its later ones. A training is not exposed:
  its cache has a manifest from the first batch and its posterior folder exists only for the final write. What
  can be lost is the calibration or inference in flight, hours rather than the training.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:829 (refused in the running
  window), :871-881 (each leftover listed, then the promise), :887-889 (the listed folders are removed).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:58, :293-309 (newest date in the tree, the folder's
  own included), :1024-1034 (refused only if younger than 300 s), :414-415 (folder created on entry), :389-391
  (the figures subfolder is made without creating parents, so it raises once the record folder is gone),
  :434-481 (the writer then removes what is left and re-raises).
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1950, :1969-1980 (simulate, then rank test, nothing
  written), :2021-2023 (first write: the first figure), :2062 (first payload), :2227-2241 (inference).
  C:/Users/J/PycharmProjects/PRISM/core/config.py:276-279 (200 batches, 2,000 sets by default).

- **W032.** Times are in two clocks: Created and the pane header are raw UTC ISO strings, while log lines carry
  local time of day with no date.
  Area: the Artifacts screen; second reader: holds.
  On screen: The Created column and the pane's header show a long raw timestamp in universal time, while each
  log line carries only a local hour, minute and second, so the lines of a run lasting several days cannot be
  placed on a date and do not line up with the creation time.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:331-332 (UTC clock), :614 (created written
  with isoformat), :850 (taken when the writer is created).
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_table.py:166 (shown as stored).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:163 (pane header).
  C:/Users/J/PycharmProjects/PRISM/core/runs.py:59-60 (local now, hour:minute:second only), :85 and :129 (every
  record and warning line stamped that way).

- **W033.** Deleting a prior, posterior or observation the inference session holds loaded: whether the session
  lets go or goes on to write records naming a deleted parent was not traced.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: After deleting a prior or posterior that is currently loaded on the inference tabs, those tabs go
  on working with it as if nothing had happened, and what they produce next points back to a record that no
  longer exists.
  Correction: Traced: the session does not let go. After a delete only the pick-lists are re-read; the loaded
  prior, posterior and observation stay in memory and the Validate, Infer, TSNPE and Train buttons stay enabled,
  so a run started afterwards completes and writes a record whose parent no longer exists (a lineage report then
  prints MISSING for it). The delete is possible only when no run is live and nothing on disk depends on the
  record yet. Narrow worst case: build or load a prior, delete that prior on the Artifacts screen, then press
  Train. The cache and the days-long posterior are then tied to a prior that is gone, and in a later session
  that posterior cannot be calibrated or narrowed through the window, because both refuse any prior but the one
  it was trained with.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:124 and :331-344 (the change signal
  re-lists pickers and nothing else). C:/Users/J/PycharmProjects/PRISM/core/gui/session.py:40-57 (the session's
  handles are dropped only by its own reset).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/validate_tab.py:75 and :94, infer_tab.py:461-488
  and :516, tsnpe_tab.py:157 and :209, posterior_tab.py:148 and :328-329 (each run is handed the in-memory
  objects; gates test only 'is not None'). C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1432 and :2091
  (the new record names those ids as parents). C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:891-916
  (dependents are read from manifests on disk). C:/Users/J/PycharmProjects/PRISM/core/SBI/run_guards.py:64-82 (a
  calibration refuses a prior other than the posterior's own).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:283-288 (MISSING line).

- **W034.** Gate 3 has no place in the window and is missing from the note's 2D table.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: The masked-probe total the third gate needs appears once, as a plain line in the log pane when
  simulation ends, with nothing to compare it against and no lasting place to read it until the whole run has
  finished.
  Correction: Half of it is in the window, unmarked. The run total of masked probes is logged as an ordinary
  information line when simulation ends and so does appear in the training tab's log pane; it is also logged
  again when a later run reuses the complete cache. What is missing is everything around it: the figure it must
  be compared with comes from a probe audit that exists only on the command line; nothing marks the line or
  holds it; the run's log file is written only when the whole run finishes, which is after the moment the gate
  is meant to be read; and the per-batch counts are kept inside the cache's state file but not in the cache's
  record, so the Artifacts screen cannot show them. The note's table has no row for this gate.
  Evidence: core/SBI/pipeline.py:224-236 (the line's text), :2004-2015 (logged at the end of generation, also
  when a complete cache simulated nothing); core/gui/streams.py:251-258 (records reach the pane);
  core/gui/widgets/log_pane.py:15 (the pane keeps 5,000 lines); core/artifacts/store.py:544-550, :627-629 (log
  file written at the commit only); core/SBI/training_checkpoint.py:216, :391-394 (counts stored in the state
  file); core/artifacts/store.py:1591-1593 (not in the cache's record); core/gui/fields.py:181-184 (the mask
  audit's knobs have no window control); docs/guide/retrain.md:572-576; the note's table at :213-223.

- **W035.** An epoch ceiling control carries two traps: N means at most N+1 epochs, and a run the ceiling stops
  keeps the last epoch's network, not the best.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: Nothing visible today; an epoch-limit box would train one more epoch than typed and, when the limit
  ends the fit, keep the last network rather than the best one.
  Evidence: Read in the installed library,
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/sbi/inference/trainers/base.py:976-980 (the count
  starts at 0 and the loop runs while it is at most the ceiling, so ceiling+1 epochs), :1125-1155 (the best
  weights are restored only when the patience runs out), :1166-1180 (the end-of-run check).
  core/SBI/train.py:365-370 (the ceiling is passed straight through). The command line's help already says both
  things: core/tool/config_args.py:101-104; core/orchestrator.py:835-836. core/config.py:370 (the default
  ceiling is 2,147,483,647, so it never binds unless set).

- **W036.** A fit stopped by the epoch ceiling returns the last epoch's weights, not the best ones, while the
  record's best validation loss describes an earlier epoch; sbi's 'maximum epochs reached' warning also never
  fires.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: Nothing today because the window has no epoch limit; with one, a fit ended by the limit would save
  its last network while reporting the best loss of an earlier epoch, and no line would say the limit ended it.
  Evidence: Installed library,
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/sbi/inference/trainers/base.py:976-980 (once the count
  passes the ceiling the loop ends without its convergence test), :1000-1002 and :1166-1180 (the end check runs
  the test once more; the 'maximum reached' branch compares the ceiling with ceiling+1 and so is never true),
  :1143-1153 (best weights restored only when the patience has run out; if the last epoch was itself the best,
  the two coincide), :1005-1006 (the recorded best loss is the best epoch's). core/SBI/train.py:365-370,
  :407-410; core/orchestrator.py:1455, :1473-1474 (the record keeps the ceiling, the epoch count and the best
  loss, and no line or key says which of the two ended the fit); core/config.py:370 (default ceiling never
  binds).

- **W037.** The cache's key holds a model's name only; a user-defined model edited and saved again under the
  same name would resume rows simulated with the old equations (suspected).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: After editing a user-defined model's equations and saving it under the same name, a training with
  the same prior and the same parameter ranges quietly reuses the rows simulated with the old equations.
  Correction: Confirmed for user-defined models, in a narrower case. The model builder does allow saving an
  edited model over itself: the 'already exists' refusal applies only when the name was changed. The equations
  live only in the model's own file; the key and every record hold the name alone. Saving also rewrites the
  model's bounds file, so if a parameter's name, order or range changed, the key changes and nothing is reused.
  The silent case is an edit to the equations that keeps the same parameters and ranges, followed by a training
  with the same saved prior: the old rows are reused, and a complete cache skips simulation altogether. Built-in
  models are not edited this way; for them the equivalent is a code change, covered by the claim about the key
  holding no code revision.
  Evidence: core/artifacts/identity.py:89 (name only); core/artifacts/provenance.py:107 (records hold the name
  only); core/gui/screens/model_builder_screen.py:520-526 (refused only when the name differs from the one being
  edited), :537-540 (editing keeps the name); core/Helpers/model_store.py:387-399 (bounds, cell and units files
  regenerated, the model file written last); core/orchestrator.py:1095-1096 (any committed batch means a
  resume).

### Everyday use (63)

- **W038.** Nothing app-wide says that a run has ended, how, or when; the run slot and the marks are just
  cleared, with no taskbar alert, tray icon or sound.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: After a run ends the window simply looks idle; unless an error box is still up you must open the
  right tab and read its last log lines to learn whether it finished, was cancelled or failed, and nothing says
  when.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/main_window.py:362-367 on run end stops the timer
  and hides the run slot; :405-410 clears the Home-tile and tab marks. core/gui/widgets/progress_pane.py:254-265
  hides the progress strip at the end. The outcome is said only in the running tab: a log line on success
  (core/gui/panels/inference/posterior_tab.py:296), 'Run cancelled.' (core/gui/panels/base_panel.py:263), or a
  modal Error box plus a log line (base_panel.py:447-457). Log lines carry no time
  (core/gui/widgets/log_pane.py:21-30). My grep of core/gui for alert(, QSystemTrayIcon, beep, QSound, winsound,
  FlashWindow, activateWindow found only raise_() on pop-out and animation widgets (widgets/figure_stack.py:131,
  widgets/anim.py:91).

- **W039.** The app-wide run indicator is a small grey bordered button with only the tab label and an elapsed
  clock: no stage, no start time, no days field; marks are a plain dot.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: While a run is live the only app-wide sign is a small grey label beside the title reading for
  example 'Running: Posterior — 63:02:11', plus a dot on one Home tile and one tab.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/screens/nav_shell.py:34-38 builds 'Running: <tab
  label> — h:mm:ss' with hours unbounded; core/gui/main_window.py:368-379 feeds it the tab label only and a
  monotonic elapsed time; core/gui/design.py:295-299 styles it at caption size (12 px, design.py:40) in the
  secondary text colour with a 1 px border. Marks: nav_shell.py:22 and :51 (tab prefix),
  core/gui/screens/home_screen.py:78 (tile suffix). It is clickable and jumps to the running tab
  (main_window.py:381-392), with a tooltip naming section and tab (:372-373).

- **W040.** The shell has no keyboard route: no shortcut for Home, back, a section or the running task, and
  Escape does nothing.
  Area: the shell, start-up, Settings and the look; second reader: holds, with a correction.
  On screen: There are no shortcut keys anywhere in the shell and Escape does nothing on a screen; without a
  mouse the only way round is tabbing from button to button.
  Correction: There are no shortcuts, accelerators or key handlers, and Escape is unhandled on the screens. 'No
  keyboard route' is overstated: nothing in the code removes the standard focus traversal (Tab between buttons,
  Space to press, arrow keys on tabs); the only focus-policy call in core/gui is the help badge opting out
  (widgets/help_badge.py:29). The Home tiles are ordinary push buttons (screens/home_screen.py:54-63) and the
  back arrow, run slot and gear are ordinary tool buttons (screens/nav_shell.py:63-78, :95-100). Not checked on
  a screen.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: my grep of core/gui for QShortcut, QKeySequence,
  setShortcut, keyPressEvent, Key_, eventFilter, setTabOrder, setBuddy matched only setFocusPolicy at
  core/gui/widgets/help_badge.py:29. A grep for a single '&' mnemonic inside a string found none (the one '&&'
  at main_window.py:432 is an escaped ampersand). The only menu actions are the gear's
  (core/gui/main_window.py:204-213).

- **W041.** One panel's constructor failing anywhere stops the whole window from opening, because every screen
  is built eagerly at launch.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: If any one section hits an unexpected problem while being built at start-up (a bad input file, an
  unreadable record), PRISM shows 'PRISM could not start' and closes, even when a different section was wanted.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/main_window.py:101-146 builds every panel and
  screen inside the window's constructor; core/gui/app.py:108-120 turns any exception from it into a 'PRISM
  could not start' box and exit code 1. The cases already met are guarded one by one, which shows the exposure
  is real: core/gui/panels/crossval_panel.py:238-248 (an unreadable cell file),
  tests/test_nav_and_gating.py:172-183 and :2655-2661 (an unreadable saved record). A new unguarded failure in
  any section has the same effect.

- **W042.** The log pane jumps to the bottom on every appended line or batch, so earlier lines cannot be read
  back while a run is printing; it also uses the proportional UI font.
  Area: the shell, start-up, Settings and the look; second reader: holds, with a correction.
  On screen: Whenever a new line arrives the log snaps back to the bottom, so scrolling up to read earlier lines
  only works in quiet stretches, and numbers in the log do not line up in columns.
  Correction: The jump happens only when a line actually arrives, not on every tick: a batch is sent only when
  there are lines (streams.py:149-150), and progress bars never reach this pane (log_pane.py:3-4). So reading
  back fails during chatty phases (the generation writes a line for most batches) and works in quiet ones (a fit
  prints almost nothing to the pane between epochs). Whether it is disruptive on screen: not observed.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/widgets/log_pane.py:21-34 scrolls to the end
  unconditionally after every append; core/gui/streams.py:28, :141-151 (15 Hz ceiling, lines only). No font is
  set on the pane (log_pane.py:11-19) and the style sheet's text-box rule names no font family
  (core/gui/design.py:220-227), so it uses the application font, Segoe UI or Inter (core/gui/fonts.py:17,
  :33-39); the Artifacts detail pane, by contrast, sets a fixed-pitch font
  (core/gui/screens/artifact_screen.py:219-225).

- **W043.** nvidia-smi is run synchronously on the GUI thread at launch and on every keystroke in the ceiling
  box; the free figure is never refreshed otherwise and carries no time.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: The window pauses briefly at start-up and on each keystroke in the ceiling box while it asks the
  card for free memory, and the free figure shown is whatever it was at the last keystroke, with no time beside
  it.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:117, :119 are the only two
  callers of _apply_vram_ceiling (search of core/gui), which calls _nvidia_smi_free_gib at :151.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/provenance.py:122-123 is a blocking subprocess.run with a 2 s
  timeout. How long each call takes was not measured.

- **W044.** 'Save' is an in-place rename of whatever prior the session holds, including a loaded, already named
  one.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: Typing a name and pressing Save on a prior that was loaded with a name already renames that prior
  rather than saving a copy.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:92 (placeholder 'name to
  save prior as…'), :321-338, :353 (the button is live whenever the session holds a prior).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:852-872 (the manifest's name is rewritten and the
  folder renamed). What survives a rename: the remembered pick and parent links go by id
  (C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:204-205; store.py:4-6) and the cache key
  by fingerprint (C:/Users/J/PycharmProjects/PRISM/core/artifacts/identity.py:83-90); anything that refers to
  the prior by its name does not.

- **W045.** A prior cannot be named or annotated before it is built, so a taken name is found only after the
  build.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: A name for a new prior can only be typed after the build; if it is taken the yellow box says so and
  another name can be tried at once, with nothing rebuilt.
  Correction: True, but the cost is small: the prior is already written unnamed when the build ends, Save
  refuses a taken name with the yellow box, and another name can be tried at once with nothing rebuilt. The
  missing note is the larger half: the Prior tab has no note box at all.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:244-245 (neither name nor
  note passed), :327-331. C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:598-599, :617-622, :652 (the
  stage takes both and checks the name at entry), :726-745 (written at completion).

- **W046.** One button label, 'Build / Load prior', for two very different costs, and nothing is asked before
  either.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: The same button either loads a saved prior in moments or starts a stability sweep that takes
  minutes or longer, depending on a list above it, and it starts either without asking.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:87 (fixed label), :176-183
  and :244-245 (branch taken from the picker at the click, no question), :259-260 (on a load none of the seven
  sweep and clustering boxes is read, though they stay live and the note under them still describes a census).

- **W047.** Three input problems on the Prior tab are log-pane warning lines only, with no box: bad grid values,
  no bounds file, Save without a name.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: For several mistakes on the Prior tab the only response is a line in the log pane, so the click
  looks as if it did nothing.
  Correction: There are five, not three. Also log-only: 'Select a bounds file first — direct entry starts from
  it' when Edit values is chosen with no file, and the warning that the declared units contradict the pipeline,
  after which the build simply goes ahead.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:188-191, :195-197, :324-326,
  plus :152-153 and :207-208. C:/Users/J/PycharmProjects/PRISM/core/sim_config.py:495-519 (what the units
  warning is). The yellow box is C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:406-426.

- **W048.** Hand-edited bounds are discarded without warning by toggling 'Use file' and back or by any Apply,
  cannot be saved as a file, and leave no bounds source in the record.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: Numbers typed into the bounds grid vanish when switching to the file and back or on any Apply,
  cannot be saved as a bounds file, and are gone at the next launch; the records made from them still store the
  ranges but name no file.
  Correction: (1) The reload on re-entering 'Edit values' is stated in the Bounds help bubble; there is no
  warning at the moment it happens. (2) What the first reader left open is settled: the numbers do stay in the
  record. Every artifact's config block stores the parameter names and the non-dimensional and dimensional lows
  and highs; only the file reference is null, and the Forcing ranges are not in that block. (3) An Apply made
  while the session holds only an applied model shows no dialog, and the Config tab's line then reads 'applying
  a model discards nothing', yet that Apply reloads the grid and drops the typed numbers. (4) The grid is not
  kept between launches either.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:141-143, :145-166, :365-366.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/help_text.py:187-190.
  C:/Users/J/PycharmProjects/PRISM/core/cli.py:306-307, :349.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/provenance.py:102-111.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/manifest.py:273-280.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:285-287, :380-383. No code under
  core/gui writes a bounds file from the grid
  (C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/param_grid.py:76-137 has no writer).

- **W049.** A bounds line whose bracket holds other than two numbers makes 'Edit values' raise outside the
  handler's try, leaving a cleared, half-built grid.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: Choosing Edit values on a bounds file with a malformed line most likely produces an error dialog
  and a half-filled grid instead of the usual yellow box; this was read, not run.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/Helpers/file_manager.py:89, :95, :235-238 (any count of
  numbers is kept). C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/param_grid.py:98-100 (rows cleared first),
  :109-113 (names recorded, then the two-value unpack).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:156-166 (load() is after the try).
  What appears was not run; the app installs a last-resort handler that shows an 'Unhandled error' dialog
  (C:/Users/J/PycharmProjects/PRISM/core/gui/app.py:15-34, :107). Traced further: a later Build / Load prior
  cannot build a config from the half grid: with no rows it reports no bounds loaded (param_grid.py:118-120),
  otherwise to_dicts fails on the first name without a row (:133-136), again outside any try (prior_tab.py:192).

- **W050.** Every Build / Load prior press re-selects the Infer tab's Cell to the launch-time file.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: Each press of the Prior tab's main button puts the Infer tab's cell list back on the cell
  remembered at launch.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/inference_screen.py:179-181.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/infer_tab.py:209-211, :545 (the only later writer
  of the stashed key). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/base.py:35-40.
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:79-100 (rescan, then the stashed key;
  with no key or a vanished file the list stays on its first entry).

- **W051.** Help is reachable by mouse only, and each entry is a long paragraph in a tooltip.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: Beside each control the help is a long paragraph in a hover bubble that the keyboard cannot reach;
  the same text can also be read as a plain list on the Settings screen, under internal names.
  Correction: Beside the control, yes. But every help entry is also shown as selectable, scrollable plain text
  on the Settings screen ('Help — what each section does'), so it is not reachable by mouse only. That list is
  on a different screen and is keyed by internal names (for example vram_ceiling, chi_k), not by the labels on
  the tabs.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/help_badge.py:24, :29, :30-33.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/help_text.py:47-56, :101-114.
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/settings_screen.py:23-29, :114-126, :184-200.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference_tabs.py:20.

- **W052.** 'Save' is a rename of whatever posterior the session holds, including a loaded one that already has
  a name; the wording reads as save-a-copy, and 'Name it below to keep it' is printed even after loading a named
  posterior.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: Typing a name and pressing Save does not make a copy: it renames the posterior the window currently
  holds, even one that was loaded under another name, and the log line wrongly suggests an unnamed result would
  not be kept.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/. core/gui/panels/inference/posterior_tab.py:56-57
  placeholder and button text; :296-297 the 'ready ... Name it below to keep it' line is unconditional (train
  and load alike); :300-317 Save calls store.rename on the session posterior's id; :334 Save is enabled for any
  held posterior. core/artifacts/store.py:852-872: rename accepts an artifact that already has a name
  (assert_name_free with allow=its own id, then m.name = new_name and the directory is renamed). The record
  already exists before Save (core/orchestrator.py:1420; docstring :809-811). No sweep deletes unnamed records:
  store.unnamed (store.py:1227) has no caller under core/.

- **W053.** The TSNPE tab has no name or Save control and its completion line does not say how to name the
  round; the only route is the Posterior tab's Save box.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: A finished narrowing round stays unnamed unless the owner knows to go back to the Posterior tab and
  use its Save box, which at that moment renames the round and not whatever that tab's selection box shows.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/. tsnpe_tab.py:47-104 no name
  box or Save button; :157-161 the round is dispatched with no name; :164-177 the completion line says only that
  the posterior is non-amortized. posterior_tab.py:300-307 Save renames the session's posterior, which after a
  round is the round (tsnpe_tab.py:173); :334 it is enabled whenever a posterior is held.

- **W054.** The Cell picker jumps back to the launch-time remembered file every time a config is built,
  discarding a cell picked this session; with nothing remembered it falls to the folder's first file.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: Each time a prior is built or loaded, the Cell choice on the Infer tab quietly returns to the one
  remembered from the last time the window was closed.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/gui/. panels/inference/base.py:35-40
  repoint(folder, self._saved_cell_key); the key is assigned only at base.py:37 and infer_tab.py:545 (launch).
  widgets/artifact_picker.py:88-100 repoint re-lists and re-applies that key; :79-86 an empty key leaves index
  0. Called from screens/inference_screen.py:179-180 on every install_config, which
  panels/inference/prior_tab.py:209-210 runs on every 'Build / Load prior' click. The first file is in
  file-system order (core/Helpers/file_manager.py:316-322).

- **W055.** The chi probe table (K recording files and K typed drive frequencies) is not remembered between
  launches, although the passive path, T_obs and Drive F0 are.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: Every time the window is opened, each probe recording has to be browsed to again and each drive
  frequency typed again before an inference on real chi recordings can run.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/. infer_tab.py:528-541
  save_settings writes the cell, mode, three T_obs boxes, three single paths and Drive F0, and the comment at
  :540-541 says the probe rows are not persisted; :543-557 restore reads the same keys; :284-287 rows are seeded
  blank; the class docstring :68-71 says the same. Within one session the rows survive a config rebuild
  (:256-287).

- **W056.** Boxes describing a recording open at seeded values nobody typed and pass the click-time rules: T_obs
  = 1.0 on all three pages and Drive F0 = 1.0 newton; the help says nothing checks F0.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds, with a correction.
  On screen: On a first inference from real recordings the length box already reads 1.0 second and the
  drive-amplitude box 1.0 newton, and pressing Run uses them; a wrong length earns a warning line in the log, a
  wrong amplitude earns nothing.
  Correction: True on first use only: all four boxes are remembered, so after the first accepted close they open
  at the last session's values (also values nobody typed for the recording at hand). A T_obs that disagrees with
  a recording's length is not refused but does put a warning line in the log; only Drive F0 has no check of any
  kind beyond being positive.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/panels/inference/infer_tab.py:96, :108, :120
  FloatField(T_MIN_EXP_S) with config.py:223 = 1.0; :121 FloatField(1.0); :399, :413-414, :432 require_positive
  only; :532-538, :550-556 the four are saved and restored (gui/settings.py:96-102 leaves the seed when no key
  exists); :231-235 the probe frequency boxes were made blank for this reason. SBI/observations.py:85-91,
  :175-178, :233-236 a length mismatch only warns; :208-212, :224-228 F0 is taken as given.
  gui/panels/inference/help_text.py:181-186.

- **W057.** Validate shows no cost or description of what it will simulate before the button, keeps no result on
  screen beyond one log line and three figures, and its caption is an unwrapped label that can widen the column.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds, with a correction.
  On screen: The Validate tab gives no idea of how much it will simulate before the button, and afterwards the
  verdict lives only in the scrolling log and three pictures.
  Correction: The middle part is overstated. Besides the tab's own one-line summary, the log pane receives the
  calibration stage's own records: the per-parameter rank-test table, the coverage line, the informativeness
  description and the verdict sentence. What holds: nothing outside the log pane and the three figures shows the
  result, there is no cost or description line before the button, and the caption label is not word-wrapped
  (whether it actually widens the column on screen I cannot tell without running it).
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/panels/inference/validate_tab.py:33-46 one
  QLabel (:35, no setWordWrap), two boxes, one button; :79-90 the tab's single summary line.
  orchestrator.py:1987-1992, :2039, :2053, :2089 the stage logs the table, TARP, informativeness and verdict;
  gui/streams.py:251-256, :293-294 every record of the core logger goes to the pane. orchestrator.py:2022-2023,
  :2042 three figures. config.py:276 the operating-point count is the number of batches, i.e. the simulation
  cost. gui/design.py:28-29 the column's minimum and default width (360 / 420).

- **W058.** 'Run TSNPE round' can be greyed with no reason shown: the tab is enabled on posterior + prior, the
  button also needs a selected observation, has no tooltip, and _round returns silently.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: Right after a training, the TSNPE tab opens but its button is grey and nothing next to it says that
  an inference must be run first to record an observation.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/gui/. panels/inference/tsnpe_tab.py:209-210 the
  button needs a non-empty observation selection; no setToolTip on btn_round anywhere in the file (:74-77);
  :134-135 silent return. screens/inference_screen.py:204 the tab is enabled on posterior + prior alone;
  :213-215 the sentence about needing an observation is the TAB's tooltip and is set only while the tab is
  disabled. The only hint on the tab is the '?' beside Observation (panels/inference/help_text.py:22-25). The
  state is the normal one right after a training, before any Infer run has written an observation.

- **W059.** reset_downstream('posterior') runs before the worker starts, so a training or load that is then
  refused, fails or is cancelled leaves the session without the posterior it held and Validate, Infer and TSNPE
  go grey.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: If a load or a training is refused, fails or is cancelled, the posterior the window was holding is
  gone from the session and the three later tabs go grey until it is picked and loaded again.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/gui/. panels/inference/posterior_tab.py:139-143
  the drop and re-gate happen after the two dialogs but before dispatch (:148); nothing restores the handle on
  error or cancel (panels/base_panel.py:262-263, :267-285). session.py:49-57 only the handle is dropped.
  screens/inference_screen.py:192-204 the three tabs are gated on it. Nothing on disk is affected
  (session.py:50-51).

- **W060.** The three Fisher boxes stay editable and are validated at the click even when the run resumes a
  cache, where the stage ignores all three; a blank Fisher box blocks a fit that would never read it.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: When retrying a fit on a finished cache, the three rotation boxes still look live and an empty one
  stops the run, although their values will not be used.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/panels/inference/posterior_tab.py:169-181
  every knob is read whenever the picker is on from-scratch, the three Fisher boxes at :178-180.
  orchestrator.py:945-956 the stage applies the same rules on any training call; :1207-1238 a resumed run reuses
  the stored rotation and never calls the Fisher; :1239-1245 the three are used only on a fresh rotated run;
  :1255-1256 not at all when the rotation is off. gui/panels/inference/help_text.py:84-86 says so only in the
  '?' text.

- **W061.** (suspected) Every keystroke in Batches or Max rows runs, on the GUI thread, a store scan reading
  every committed cache's header plus a CUDA memory query; the TSNPE tab pays for the same scan and discards the
  result.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds, with a correction.
  On screen: Possibly a slight stutter while typing in the two budget boxes once many simulation caches exist;
  nothing visible otherwise.
  Correction: The structure holds; the size of the cost cannot be told from reading and is narrower than 'every
  keystroke scans every cache'. The scan is skipped while no config or no prior is held, and when this budget's
  own cache already holds batches only that one state file is read. Otherwise each other committed cache costs
  two small loads (its state file and its header: two schedule vectors of one number per batch, one block of
  initial values, the rotation and a probe), never the row shards. The same work also runs on both tabs after
  every stage and after every run anywhere in the window.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/panels/inference/posterior_tab.py:80-81,
  tsnpe_tab.py:94-95 textChanged -> _sync_budget; base.py:138-139 one training_preview call.
  orchestrator.py:529-532 the planner query on CUDA (config.py:130, :135 mem_get_info / memory_reserved /
  memory_allocated); :541-544 skipped when off or with no config/prior; :547-555 identity, own state, then
  describe_siblings only when the own cache holds nothing. SBI/training_checkpoint.py:275-287 per sibling: peek
  (torch.load of the state, :162-184) and read_header (:222-223); header contents :201-215; initial values are
  one row block (SBI/pipeline.py:1237-1242). artifacts/identity.py:56-62 the units file is read each time.
  gui/panels/inference/tsnpe_tab.py:179-198 the fixed paragraph ignores the scan's result. Re-run on every
  refresh: posterior_tab.py:338, tsnpe_tab.py:211, panels/base_panel.py:316-317, :333-335.

- **W062.** Nothing marks the end of a run: no finish line, no duration, no clock time, no sound or taskbar
  alert; the clock and the progress pane disappear.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: When a run ends the tab's log gains one closing line with no time on it, the header clock and
  progress area vanish, and nothing flashes or sounds, so on coming back to the machine one cannot tell when it
  ended or how long it took.
  Correction: There IS a closing line for every ending, written by each tab rather than by the shared frame: on
  success e.g. 'Posterior ready: ... Name it below to keep it.'
  (core/gui/panels/inference/posterior_tab.py:296-297), 'Prior ready' (prior_tab.py:317), 'TSNPE round complete'
  (tsnpe_tab.py:174-176), 'Calibration verdict ...' (validate_tab.py:87-90), 'FDT run finished.'
  (core/gui/panels/fdt_panel.py:253); a cancel writes 'Run cancelled.' (base_panel.py:263); a failure writes the
  error line (base_panel.py:447). What is missing is the time of day, the duration, and any alert.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/base_panel.py:267-285 (_finished writes no
  line of its own); core/gui/main_window.py:364-367 stops the clock and hides the header slot;
  core/gui/widgets/progress_pane.py:254-265 hides the pane; core/gui/widgets/log_pane.py:21-30 adds no
  timestamp; grep over core for .alert(, beep, FlashWindow, QSystemTrayIcon: no match.

- **W063.** The log pane jumps to the bottom on every new line, so earlier lines cannot be read while a run is
  printing.
  Area: how a run is started, shown, cancelled and closed; second reader: holds.
  On screen: Scrolling up to read an earlier line is undone each time a new line arrives.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/widgets/log_pane.py:21-34: append_line and
  append_lines both end in _scroll_to_end(), which sets the scrollbar to its maximum unconditionally. During a
  fit few lines arrive (sbi's counter is a progress row, not a log line: core/gui/vt.py:289-300), so this bites
  mainly during generation and sweeps.

- **W064.** The worker's last-resort handler also swallows any RuntimeError from setting up or tearing down the
  stream redirect: no result, no error, no line, the tab just unlocks.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: In the rare case the machine cannot start one more thread, clicking Run greys the tab for an
  instant and returns with nothing in the log and no result.
  Correction: Only the set-up half has a concrete source: failing to start the pump thread (Python raises
  RuntimeError 'can't start new thread'). In that case the function never started, so nothing is lost; the click
  simply does nothing and says nothing. I found no RuntimeError source in the teardown (router.close and the
  tqdm reset are guarded, the join and the lock release cannot raise as used). A RuntimeError raised by the
  stage itself does NOT take this path; it is caught inside and reported normally. Even in the worst theoretical
  teardown case a posterior would already be on disk.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/worker.py:107-108 and :207-210: the outer try
  encloses the with-statement, so an exception from entering it lands in 'except RuntimeError: return'; :146
  catches the stage's own exceptions first. core/gui/streams.py:286-287 (_Pump() inside the try), :118-119
  (thread start), :278-281 (the comment naming thread exhaustion), :301-321 (teardown: :309-312 guarded,
  :315-316 join with timeout, :321 release). The same handler covers the decline message at streams.py:273 and
  the report emits at worker.py:186-205.

- **W065.** The red box shows the exception's message without its type, so a bare assertion shows an empty box
  and a KeyError shows only the key.
  Area: how a run is started, shown, cancelled and closed; second reader: holds.
  On screen: Some failures show an Error box that is empty or holds only a quoted name, with the kind of error
  visible only under Details.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/base_panel.py:446 message = str(exc), :447
  the same text to the pane, :451 box.setText(message). The app-level hook does include the type
  (core/gui/app.py:28). A message-less assert exists on the training path: core/orchestrator.py:1413. The type
  is visible only as the last line of Details (base_panel.py:452-453).

- **W066.** Harmless progress frames appear in the log pane as warnings because every stderr line gets the
  warning prefix.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: Ordinary progress lines and library chatter appear in the log with a warning triangle although
  nothing is wrong, which makes real warnings harder to spot.
  Correction: Certain by reading: anything the parser turns into a log line from stderr carries the warning
  mark, and the bars that keep their last frame do so on every normal completion ('Constructing latent
  prior...', 'Campaign 2 ...', 'Exporting animation', plus any library bar). Bars created to vanish are retired
  without a log line. Whether a cancel logs 'every live bar's frame' depends on the order the bars close during
  the unwind; not established.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/streams.py:289 (stderr stream at level 'warning');
  core/gui/vt.py:181-186 (_log uses the stream's level), :218-221 (a finished kept bar's frame is logged),
  :191-194 (a replaced bar's frame), :301-302 (a vanishing bar is retired, not logged), :333-335 (rows still
  live at teardown are logged). No tqdm call under core passes file= (grep: core/SBI/Priors/prior.py:130,
  core/FDT/campaigns.py:300, core/gui/panels/simulate_export.py:129 keep their frame; the others pass
  leave=False, e.g. core/Solvers/sdeint.py:33-38, core/SBI/pipeline.py:1630).

- **W067.** Runs pile up in one tab with nothing between them: no header between runs' log lines, figure tabs
  never cleared, each new figure takes the front tab.
  Area: how a run is started, shown, cancelled and closed; second reader: holds.
  On screen: After several runs in one tab the log is one unbroken stream and the figure area still holds every
  earlier run's figures, with no mark of where a run began.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/base_panel.py:206-289 (dispatch writes no
  start line and clears nothing); core/gui/widgets/figure_stack.py:109-110 (new tab made current), :148-150
  clear_all has no caller (grep: only a docstring mention at core/gui/panels/record_view.py:146). Small
  exception: the FDT and CrossVal tabs close the tabs their saved-record viewer opened (fdt_panel.py:254,
  crossval_panel.py:333); run-made tabs stay.

- **W068.** While a cancel is pending the only signs are the greyed 'Cancelling...' button and one log line; the
  header still reads 'Running' and the close dialog offers 'Cancel task & quit' again.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: After clicking Cancel during training, the header keeps saying 'Running' with the clock counting
  for up to an epoch, and closing the window offers to cancel again as if nothing had been asked.
  Correction: Holds, except 'one log line that scrolls away': during a fit no further log lines arrive until the
  epoch ends, so the line stays in view there. Also, a cancel requested through the close dialog changes neither
  the button nor the log (base_panel.py:114-118).
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/base_panel.py:291-300 (button disabled,
  text, one line); core/gui/main_window.py:376-379 with core/gui/screens/nav_shell.py:25-38 (the banner is title
  plus elapsed time, whatever the token says); main_window.py:423-435 (the dialog tests only whether a run is
  live).

- **W069.** A cancelled or crashed FDT/sweep record keeps its name but is invisible on its own tab; re-running
  under the same name is refused, also after a successful named run, and the fix line points at a 'Save box' the
  tab lacks.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds, with a
  correction.
  On screen: Type a name, run, and the next run under that name is refused whether the first one finished,
  failed or was cancelled; a cancelled one is nowhere to be seen on the tab that made it, and the advice shown
  mentions a box that tab does not have and a rename the window does not offer.
  Correction: The fix line is not only wrong: it reads 'Choose another name in the Save box, or in the 'Record
  name' or 'Comparison name' box on the FDT analysis or Sweep study cross-validation tab', so the right box is
  named, after one these tabs do not have. The sentence above it says 'rename or delete it first', and the
  window has no rename for these records at all (only priors and posteriors can be renamed); deleting is done on
  the Artifacts screen, which does list the unfinished record. A run refused before it wrote anything leaves no
  record and frees its name. A sweep study cancelled in its first half leaves only the '-s' name taken; the
  '-temp' record was never put on disk.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:192-198.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:737-741 and 842-844 (any manifest claims the name),
  434-481 (kept on any exception except a refusal before anything was handed out).
  C:/Users/J/PycharmProjects/PRISM/core/gui/fields.py:96-97.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:300-302 (name claimed at the click), 326-336
  (nothing cleared). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:305-307, 335-349. Grep
  for '.rename(' under core/gui: only inference/prior_tab.py:328 and inference/posterior_tab.py:307.
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:133-149, 543-546 (unfinished fdt rows are
  listed and deletable). C:/Users/J/PycharmProjects/PRISM/core/FDT/cross_validation.py:643-649 (the second
  writer is entered only when the second sweep starts).

- **W070.** With 'Proceed to the production sweep after sanity' unticked, the run logs 'Aborted by user.' and
  writes a record marked complete that holds no result, listed under 'Saved run' like a finished measurement.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Untick the 'proceed' box and the run stops after the sanity checks, announces that a record was
  written, and that record then sits in the 'Saved run' list looking like any finished measurement although it
  holds none.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/FDT/fdt_pipeline.py:341-350 (early return after the sanity
  checks), 308-317 (complete set True after _measure returns, whatever it did).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:482 and 627-629 (clean exit commits complete true),
  700 (finished is read from that flag), 1545-1558 (loading passes with no payload).
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:197-200 and
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:142 (listed), 326-336 (pane says 'FDT record
  written' and the picker moves onto it).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/record_view.py:100-101 (no 'did not finish' line; the only
  trace is 'confirm_production=False' in the raw settings line).
  C:/Users/J/PycharmProjects/PRISM/core/FDT/compare.py:247-250 (a comparison does refuse it, for holding no
  numbers).

- **W071.** Sweep tab: changing Preset silently overwrites n_freqs, M_ensemble and both point counts; changing
  Cell overwrites both grids' ends; the preset also fixes three values shown nowhere before the run.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Change the preset and four numbers you may have typed are replaced without a word; change the cell
  and the grid ends are replaced; and three settings the preset decides appear nowhere until the finished record
  is read back.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:118 and 260-265 (four boxes
  rewritten on a preset change), 114 and 256-258 (grid ends rewritten on a cell change), 37 (the only help text:
  'quick/coarse' against 'finer grids and more frequencies'), 359-360 and 375-376 (point counts are remembered,
  then overwritten by the next preset change). C:/Users/J/PycharmProjects/PRISM/core/cli.py:455-460 (the preset
  also carries the frequency band, periods per probe and spontaneous length), 538-543 (used unshown).

- **W072.** Nothing before the button says what an FDT run or a sweep study will cost, while the Posterior tab
  has three such lines.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: On the two FDT tabs you press Run without any indication of how many simulations it means or how
  long it may take.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:118-178 and
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:109-173 build no derived line of any kind.
  Contrast C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/base.py:108-216 (total, memory and
  checkpoint lines recomputed as the boxes change).

- **W073.** Figure tabs pile up across runs under identical titles; nothing clears the stack at a new run, the
  FDT docstring's 'this run's figures alone' is false after a second run, and every reduction tab is titled
  'reduction sweep'.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: After a second run the picture area holds two sets of tabs with the same names and nothing saying
  which run each belongs to.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:206-289 (dispatch never touches
  existing tabs). C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/figure_stack.py:67-70 and 109-110 (every
  figure is a new tab), 148-150 (clear_all exists; grep finds no caller in core/gui).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:249-254 (the claim) against
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/record_view.py:183-194 (only the viewer's own tabs are
  closed, never the earlier run's). C:/Users/J/PycharmProjects/PRISM/core/gui/plot_watcher.py:33-39 (title is
  the file name without its stamp) and C:/Users/J/PycharmProjects/PRISM/core/Reduction/plots.py:102-103.

- **W074.** Simulate streams the settling-in transient before the T_obs samples, unmarked and on the same time
  axis, while the pane says 'for <T_obs> s of observation'; the bar is indeterminate with an empty caption;
  there is no seed.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: The live trace runs longer than the time asked for, starting with a settling-in stretch that is not
  marked; the progress bar just pulses; and the same stream can never be played again.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/simulate_runner.py:89 ('shown, not discarded'),
  128-131 (stream length is transient plus T_obs), 212-218, 254-264 (the seconds axis counts from the first
  emitted sample, transient included), 187-188 (no seed argument).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/simulate_panel.py:124-125.
  C:/Users/J/PycharmProjects/PRISM/core/sim_config.py:172-186 and
  C:/Users/J/PycharmProjects/PRISM/core/config.py:624 (100 non-dimensional units; its length in seconds depends
  on the cell). C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/progress_pane.py:274-283 (the solver's bar is
  dropped) and 366-389 (no other bar: indeterminate, caption empty).
  C:/Users/J/PycharmProjects/PRISM/core/Solvers/sdeint.py:219, 226 (noise from the unseeded global generator);
  grep for manual_seed under core/gui finds nothing.

- **W075.** Model builder: the line reporting Validate/Save outcomes sits inside the scrolled area under
  'Display scales' while the buttons are in the fixed bar; nothing scrolls it into view; some failures use only
  that line, others also open the yellow box.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: With a long model on screen, pressing Validate or Save can appear to do nothing at all, because the
  answer, including 'Saved', is written below the visible part of the form.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/model_builder_screen.py:286-287 and 302-314 (the
  status label is the last widget in the scrolled column), 316-330 (buttons outside it), 584-585 (setting the
  status only sets text; the file has no scroll-into-view call), 387-393 with 394-406, 435-442, 490-496 (status
  line only) against 410-478 and 571-582 (status line plus box). Success is reported the same way only: 506-507
  ('is valid... Ready to save') and 532 ('Saved ...').

- **W076.** Model builder has no unsaved-changes guard: 'Open model builder' blanks the form and 'Edit' replaces
  it without asking; saving the model being edited overwrites its generated files without asking.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Leave the builder for a moment and come back through its only button, and a half-typed model is
  gone without a question; saving an edited model replaces its files without a question either.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:254-256 (reset, then navigate), 274-280
  (load, then navigate). These are the only two ways onto the builder screen (main_window.py:141-146, 148-152),
  so a half-typed new model left with 'Back to settings' cannot be returned to: the only way back in blanks it.
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/model_builder_screen.py:520-530 (no question when the name
  is the one being edited), 556-569. C:/Users/J/PycharmProjects/PRISM/core/Helpers/model_store.py:387-399 (the
  default bounds and cell files, the units file and the definition are rewritten in place).

- **W077.** While a days-long training runs, saved FDT runs and sweeps cannot be browsed or compared on their
  tabs and no CPU-only FDT, sweep or Simulate run can start; the lock exists for console redirection, not for
  the card.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds, with a
  correction.
  On screen: For as long as a training runs, the FDT, sweep and Simulate tabs are greyed out: no earlier result
  can be reopened or compared there and no small run can be started, although earlier records can still be read
  on the Artifacts screen.
  Correction: What the user meets is as described, but two parts are too narrow. (1) Saved records can still be
  read during a run on the Artifacts screen, which is deliberately outside the lock; it is only the two FDT
  tabs' pickers and Compare buttons that are greyed. (2) The lock guards more than the console. The code gives a
  second reason itself (a model list changed mid-run repoints pickers under the running job), and a second run
  in the same process would also share things the first depends on: every FDT run seeds and restores the
  process-wide random generators, and every finishing job closes all open matplotlib figures in the process. So
  'these runs are CPU-only, let them through' is not a safe change as the code stands.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:102-106, 221-225, 312-317, 333.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:168, 177-178 and crossval_panel.py:165, 172-173
  (pickers and compare box inside the locked column). C:/Users/J/PycharmProjects/PRISM/core/cli.py:420, 544 and
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/simulate_runner.py:75 (CPU).
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:439 and 580-588 (reads are never
  refused). C:/Users/J/PycharmProjects/PRISM/core/rng.py:96-107 (process-wide torch and numpy state seeded, then
  restored). C:/Users/J/PycharmProjects/PRISM/core/gui/worker.py:163-170 (plt.close('all') when any job ends).
  C:/Users/J/PycharmProjects/PRISM/core/gui/streams.py:31-35.

- **W078.** An unsupported model's reason goes to the log pane only (also at every launch when it is the
  remembered model), the disabled 'Run FDT analysis' button has no tooltip, and 'Select a cell file first.' is a
  pane line, not a box.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: The Run button is simply grey; the reason is a line in the text pane on the other side of the
  window, easy to miss and scrolled away by later output.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:256-263 (button disabled, reason
  appended to the pane), 355-356 (the same handler is called on every launch), 150-152 (no tooltip is ever set
  on the button; grep for setToolTip in the file finds none), 266-269. Contrast
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:330 (the disabled Train button
  carries its reason as a tooltip). Same pane-only line at crossval_panel.py:268-271, reduction_panel.py:66-69,
  simulate_panel.py:102-105.

- **W079.** The reduction map leaves two loose stamped files with no manifest, name or note, invisible on the
  Artifacts screen; its Part-A report exists only in the pane, and the returned record is stored and never
  shown.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: A reduction run leaves a table and a picture in a folder the window never lists, with no name, note
  or settings beside them, and its printed report is gone once the pane scrolls or the window closes.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/Reduction/sweep.py:126-159 (the report is print() only),
  185-190 (a stamped table file), 196-201. C:/Users/J/PycharmProjects/PRISM/core/Reduction/plots.py:99-104 (a
  stamped picture in the same folder). C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:33-41 and 79 (no
  such kind, not a legacy folder either); grep for 'reduction' in
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py finds nothing.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/reduction_panel.py:89-94 (the result is kept on the panel and
  used by nothing). C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/log_pane.py:15 (5,000-line cap).

- **W080.** Run figures saved into a record carry the window theme of the moment for good, and re-opened tabs
  mix themes.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: A run made while the window is dark stores dark pictures in its record permanently; reopened later
  in a light window, or beside a run made from the command line, the pictures do not match.
  Evidence: The part the first reader could not read holds: the FDT plots are built under the global colours and
  saved with a plain savefig, C:/Users/J/PycharmProjects/PRISM/core/FDT/plots.py:62-88, 129, 216, 272 and
  C:/Users/J/PycharmProjects/PRISM/core/FDT/cross_validation_plots.py:127.
  C:/Users/J/PycharmProjects/PRISM/core/gui/mpl_theme.py:16-33 (the window rewrites those global colours to its
  dark or light palette, including the saved page colour), 36-42.
  C:/Users/J/PycharmProjects/PRISM/core/gui/app.py:104.
  C:/Users/J/PycharmProjects/PRISM/core/Reduction/plots.py:104. The command-line tool never installs that theme,
  so the same run made there has white figures.

- **W081.** The pane cuts the parameter-name list to three of thirteen, so a record's parameters and their order
  cannot be read.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: On a prior or posterior made with the usual thirteen-parameter boxes, the parameter list reads as
  the first three names followed by '(13 entries)', in the pane, in the saved text and in the lineage file.
  Correction: True of any list of more than 12 names. The 13-name lists of the master and tier-1 boxes are cut
  in the Knobs section of every record, in a posterior's Body, in the saved text and in the lineage file; the
  12-name spontaneous box prints in full. Not true of every pane: a calibration's per-parameter results (and a
  rotation check's per-parameter block) are lists of named records and print all thirteen names in order, and
  every pane still shows the bounds file's path under Inputs. On the two 13-name boxes the three names shown are
  identical; the one that differs (T against f_scale) is the thirteenth, so the cut line cannot tell the boxes
  apart, though the tier-1 block and the three-entry range lines still can.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:36-37 (a string counts as a scalar),
  :57-60 (more than 12 scalars -> first three and a count), :87-88 (every list in the tree goes through it),
  :124-125 (the lineage's knob lines too). C:/Users/J/PycharmProjects/PRISM/core/artifacts/manifest.py:276
  (config.param_keys = ND names + rescale names), :296-300 (tier1 block only on that box).
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1415 and :1459 (the same 13 names in a posterior's
  body.transform), :2068-2070 (calibration per_param is a list of {name,...} records, printed whole by
  report.py:82-86). C:/Users/J/PycharmProjects/PRISM/Resources/Bounds/nadrowski/master.txt:2-16 and
  master_tier1.txt:2-16 (10 + 3 names each, differing only at line 16); master_spont.txt:2-15 (10 + 2 = 12,
  printed in full).

- **W082.** During a training the Artifacts screen has nothing to read about the run except the cache's batch
  count, and that only after Refresh.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: While a training runs, the Artifacts screen lists the cache with a batch count only as fresh as the
  last visit or Refresh, shows no log for the run, and shows no posterior at all until the very end.
  Correction: The count is also re-read on every return to the screen and on a change of Kind, not only on
  Refresh; nothing updates it while the screen stays open. During the fit itself (the days-long part) even the
  count stops changing: the cache is already finished and no posterior row exists until the fit ends. The run's
  live lines are on the running tab's own log pane (capped at 5,000 lines), not on this screen.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:3-7 (stated reason for staying
  unlocked), :205, :272, :279-284 (the only three re-read triggers; the file imports no timer and no run-state
  signal, :22-36), :62-63 and :392 (a cache has no log).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:544-574 and :627-628 (log.txt written with the
  manifest at commit). C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1420 (the posterior's writer opens
  only after the fit). C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:395 and :465-472 (cache
  manifest rewritten at each save); C:/Users/J/PycharmProjects/PRISM/core/config.py:303 (every 50 batches).
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/log_pane.py:15 (5,000-line cap).

- **W083.** A log longer than 1 MiB cannot be read in full from the window: the pane keeps the end, Save writes
  the same cut text, and the notice gives the size in bytes.
  Area: the Artifacts screen; second reader: holds.
  On screen: For a record whose log is longer than about a megabyte the pane shows only the end, under a line
  stating the limit in bytes; the saved text is the same cut, and the window offers no way to reach the opening
  lines.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:55 (limit), :389-395 (tail read
  and the '(only the last 1048576 bytes are shown; the file is longer)' line), :411-423 (Save writes the pane's
  text), :429-475 (the lineage file carries no log).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:786-799 (whole file read, last bytes kept).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:152-157 (the folder is not shown, so the file cannot
  be located from the pane either). Whether a real training log passes the limit cannot be told by reading;
  C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:1925-1929 writes one warning line of roughly 150
  characters for each batch that has a failed trajectory, so a 10,000-batch generation in one process plausibly
  does.

- **W084.** A real record's folder is never shown and nothing opens or copies it, so the figures and data files
  the pane lists cannot be reached from the screen.
  Area: the Artifacts screen; second reader: holds.
  On screen: The pane lists a record's figures and data files by name but never says where the record's folder
  is, and there is no button to open or copy it; the folder's path appears only inside the delete question.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:147-174 (header, sections, payload and
  figure names; no directory). C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:16-18
  (open-the-folder declined), :376-379 (path printed only for a folder without a manifest), :170 (the delete
  question is the one place a real record's path appears). A search of C:/Users/J/PycharmProjects/PRISM/core/gui
  for openUrl, QDesktopServices, clipboard, startfile and artifacts_root finds only app.py:81 and
  panels/reduction_panel.py:89, neither a display or an open action.

- **W085.** Stored figures can be reopened only for finished single-cell FDT runs and finished sweeps, from
  those two tabs; every other kind of record has no viewer.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: Figures of a saved record can be looked at again only for finished single-cell FDT runs and
  finished sweeps, on those two tabs; a training's loss curve, a calibration's plots, an inference's plots and
  comparison figures cannot be reopened anywhere in the window.
  Correction: One addition: loading a prior on the Prior tab draws its corner figure afresh from the stored
  distribution (a new drawing, not the stored picture). The rest stands; a posterior's loss curve is stored as a
  picture and as numbers and nothing in the window reads either back.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/record_view.py:105-117 and :211-227 (the only code
  in core/gui that reads a record's figures folder; my search of core/gui for 'figures' and add_png found no
  other reader). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:142 and
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:141 (pickers filtered to 'single' and
  'sweep'). C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_picker.py:197-200 (unfinished rows
  dropped). C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:663-671 (a loaded prior is re-plotted),
  :1423-1431 (loss.npz and the loss picture are written; a search of core finds no reader of loss.npz).

- **W086.** Every refresh drops the selection and empties the pane, including a return to the screen, Refresh
  and Set.
  Area: the Artifacts screen; second reader: holds.
  On screen: Each time the screen is revisited, Refresh is pressed or a note is set, the selected row is
  deselected and the right-hand pane goes blank, so the record has to be found and selected again and the place
  in its log is lost.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:279-284 (re-list on every
  show), :301-306 and :329-333 (the table is rebuilt with nothing selected and the pane is re-rendered for no
  selection), :350-361 (no selection -> empty text), :574 (every change re-lists), :640-641 (Set goes through
  it). C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_table.py:220-223 and :276 (clear on every
  fill, selection deliberately not kept). C:/Users/J/PycharmProjects/PRISM/core/gui/screens/nav_shell.py:91 and
  :127 (screens sit in a stacked widget, so coming back is a show).

- **W087.** Set, Delete... and both sweep buttons look available during a run; the refusal appears only after
  the click, as a line at the bottom.
  Area: the Artifacts screen; second reader: holds.
  On screen: While a run is live the Set, Delete and sweep buttons look clickable; clicking one only puts a
  warning sentence on the line at the bottom of the screen.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:542-547 (enabled states depend
  on the selected row only), :580-590 (the running check writes a status sentence and returns), :617, :682, :829
  (the three callers), :932-934 (status line with a warning prefix). The file connects nothing to the run-state
  signal (:22-36 imports, :252-272 connections).

- **W088.** There is no filter, search or multi-selection, and Enter in the Note box does nothing.
  Area: the Artifacts screen; second reader: holds.
  On screen: Unnamed calibrations and inferences can be told apart only by their time and note, records are
  removed one confirmed click at a time, and pressing Enter after typing a note does nothing.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:200-210 (Kind list and Refresh
  are the only controls above the table), :489-498 (the Note box is connected to nothing; only the Set button's
  click is), :672-716 (one record per delete, one confirmation each).
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_table.py:26-39 (calibrations and inferences have no
  column of their own, only Name, Created, Note), :248 (single selection).

- **W089.** On the FDT and sweep tabs the saved-run line shows raw key=value settings and a raw UTC timestamp,
  and a restored record's figures open only by picking it again; there is no button.
  Area: the Artifacts screen; second reader: holds.
  On screen: Under the saved-run list the settings appear as a comma-separated string of internal names and
  values with a long universal-time stamp, and to see a remembered run's figures after a launch the run has to
  be chosen again from the list.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/record_view.py:64-65 (the created string as
  stored), :70-74 (settings as 'key=value' under their stored keys), :164-166 (at launch the line is filled and
  no figure opens; figures hang on the list's user-pick signal alone), :202-204.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:245-254 and
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:229-231, :333 (the panels' own show-record
  method is wired to no button; only the run-finished hook calls the viewer).

- **W090.** Loading a saved posterior when the session has a config but no prior crashes with an AttributeError
  (red box, traceback) instead of a refusal.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: After a prior build or load that was cancelled, refused or failed, picking a saved posterior and
  pressing the train-or-load button opens a red error box with a programmer's message instead of a sentence
  saying a prior is needed first.
  Evidence: core/gui/panels/inference/posterior_tab.py:114-121 (the missing-prior guard covers only a new
  training), :148-150 (the session's prior, possibly None, is passed on a load), :324-329 (the button is enabled
  for a load without a prior). core/gui/screens/inference_screen.py:186, :196 (the Posterior tab opens on the
  config alone). core/gui/panels/inference/prior_tab.py:209-210 (the prior is cleared and the new config
  installed before the stage runs at :244-245; the prior is set again only on success, :315-316);
  core/gui/session.py:54-55. core/orchestrator.py:892-902: line 902 reads prior.prior, far ahead of the load
  branch at :976. core/runs.py:241-249 (the entry wrapper does not touch the prior). core/gui/worker.py:146-150,
  :205 and core/gui/panels/base_panel.py:444-457 (anything that is not a refusal gets the red box with a
  traceback).

- **W091.** The window pairs prior and posterior by hand; loading an amortized posterior never checks the
  pairing, and the Infer tab's in-distribution check then runs against whatever prior the session holds.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: A saved posterior can be loaded beside a prior it was not trained with and nothing says so until
  calibration refuses; meanwhile a simulated inference's warning about being outside the training range is
  judged against the wrong prior.
  Correction: Holds, and the damage is bounded. A load compares the posterior with the config (model, parameter
  order, box, mode, widths) and with the prior stored inside the posterior, never with the session's prior.
  Calibration refuses a mismatched pair, and a narrowing round refuses it too. What stays unchecked is only the
  advisory warning a simulated inference gives about a truth lying outside what training saw: it is judged
  against the session's prior, so it can be wrong in either direction. The inference result itself does not use
  the session's prior.
  Evidence: core/orchestrator.py:976-992 (the load branch checks the prior only when the posterior carries a
  region); core/artifacts/store.py:1302 (load_posterior takes no prior), :1327-1384 (what it compares).
  core/gui/panels/inference/infer_tab.py:476-478 (the session's prior is passed); core/orchestrator.py:2515-2519
  (used for the warning, no fingerprint comparison). core/orchestrator.py:1949 with core/SBI/run_guards.py:64-82
  (calibration refuses a mismatch); core/orchestrator.py:1168-1174 (a narrowing round refuses).
  core/tool/config_args.py:348-361 (the tool always loads the posterior's own prior).

- **W092.** The pre-flight's epoch timing cannot be run from the window.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: The short timing run that predicts how long a full fit will take cannot be made from the window; it
  has to be typed at the command line.
  Correction: Holds for the two things that matter: the window has no epoch ceiling and cannot turn the cache
  off, and the timing needs both. 'No store root' is too strong: the window takes its store from the same
  environment setting the command line uses, read at launch, so it can be started on a scratch store; there is
  no control for it. The note's table of runbook steps has no row for this step.
  Evidence: docs/guide/retrain.md:154-163 (two trainings with --max-epochs 9 --checkpoint-every 0 under a
  scratch store); core/gui/fields.py:60, :86 (no control for cadence or epoch ceiling);
  core/gui/panels/inference/posterior_tab.py:171-181 (the nine arguments the tab sends); core/gui/app.py:80-83
  with core/config.py:187-191 (store root from PRISM_ARTIFACTS);
  docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md:213-223 (no row for the timing).

- **W093.** The diagnostic behind gate 2 (ablation) fails with plain ValueError rather than a refusal, and
  depends on the checkpoint cadence, a setting the window may soon expose.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: Nothing today because there is no screen for it; if one is added as the code stands, the ordinary
  'this posterior cannot be checked' cases would appear as red crash boxes, and a posterior trained with the
  cache switched off could never pass this gate.
  Evidence: core/diagnostics/ablation.py:41, :104-109, :114-119, :129-134 (plain ValueError for 'no cache',
  'wrong width', 'no rows'); core/refusals.py:34 (a refusal is a subclass of ValueError, so a plain ValueError
  is not one); core/gui/panels/base_panel.py:444-457 (only a refusal gets the yellow box);
  core/orchestrator.py:1433-1434 (a posterior trained with the cache off records no cache), :948 (cadence 0 is
  legal). No screen calls this diagnostic today (grep of core/gui).

- **W094.** The first refusal a 'must resume' choice would hit carries no field key, so the window could not say
  where to fix it.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: Nothing visible today; if a 'must resume' choice is added, its refusal would appear without the
  line that says which control to change.
  Evidence: core/orchestrator.py:1110-1112 (the hoisted refusal has no field); core/SBI/pipeline.py:1505-1512
  (the later copy and the 'never' refusal carry field='resume'); core/gui/fields.py:61 (the window maps 'resume'
  to no control); core/orchestrator.py:957-965 (the cadence/policy conflict has no field by design);
  core/gui/panels/inference/posterior_tab.py:148-150 (the window sends no resume policy, so neither can be
  reached from it today).

- **W095.** A screen for the repeated calibration test would sit silent for long stretches (suspected).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: Such a screen would show a moving bar while its test data is simulated and then nothing at all,
  with Cancel not responding, while each repeat's ranks are computed.
  Correction: Partly. While each repeat's calibration set is simulated, the ordinary generation bar runs,
  because that draw goes through the same loop as training data. The silent part is what follows in each repeat:
  the rank computation is asked for no progress bar, and the checks after it print nothing; one line is logged
  per repeat. During that stretch Cancel cannot take effect either, because a cancel only lands on a print or a
  log line. How long the stretch is at full size cannot be told without running it. The Validate tab's own stage
  asks for progress bars at the same step.
  Evidence: core/diagnostics/sbc.py:163-189 (show_progress_bar=False at :172; one record per repeat at :188);
  core/SBI/analysis.py:169 with core/SBI/pipeline.py:1630 (the calibration draw runs the generation loop and its
  bar); core/orchestrator.py:1979, :2036 (the calibration stage passes show_progress_bar=True);
  core/gui/streams.py:191-205, :251-253 (where a cancel can be raised); core/gui/panels/base_panel.py:297-298.

- **W096.** A run's log lines are stamped with the time of day only; a log covering several days cannot say
  which day a line belongs to.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: In the log of a run that lasted several days every line shows only a clock time, so two lines
  cannot be placed on the right day without counting midnights.
  Evidence: core/runs.py:59-60 (hours, minutes, seconds only), used at :85 and :129; core/artifacts/store.py:570
  (the 'stopped' line uses the same stamp). docs/guide/retrain.md:197 already tells the reader to add 24 hours
  by hand to a span that crosses midnight.

- **W097.** PRISM_MEM_LOG_EVERY=0 stops a generation at its second batch with a division by zero, and a
  non-number stops the window or the tool at import; a control for this cadence must refuse anything below 1.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: A training started with that setting at zero stops with an error box after its second batch; with a
  non-number in it the window does not open at all.
  Correction: Holds for 0 and for a non-number. Two refinements. A negative value does not fail: the remainder
  test still fires on multiples of its size, so the rule a control needs is 'a whole number other than zero'.
  And on a resumed run the failure comes on the first batch generated, not the second, because only batch number
  0 is exempt. Batches finished before the failure are rescue-saved, so what is lost is minutes, not committed
  rows. With a non-number the window does not open and no dialog is shown, because the failure happens while the
  program is still being loaded.
  Evidence: core/SBI/pipeline.py:269 (read once at import; '0' becomes 0, a non-number raises), :1919 (the
  remainder by that value, skipped only for batch 0), :1951-1990 (the rescue save on the way out);
  core/gui/panels/inference/config_tab.py:9 (the window imports that module); core/gui/app.py:12 (the main
  window is imported before any error dialog is installed).

- **W098.** The VRAM ceiling a run used is recorded nowhere: no line at the start of the run, no key in the
  posterior's manifest, none in the cache's.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: Looking at a finished or a failed run, there is no way to tell from its log or its record what
  memory ceiling it ran under.
  Evidence: core/SBI/pipeline.py:584-603 (the only lines that name it are two bad-value warnings), :1870-1875
  (and one warning after an out-of-memory), :292-298 (the memory line prints the learned cap, not the ceiling);
  core/orchestrator.py:1450-1455 and core/artifacts/manifest.py:273-299 (the posterior's settings block has no
  ceiling); core/artifacts/store.py:1581-1593 (nor the cache's record); core/artifacts/identity.py:87-122 (nor
  the key). Grep of core/ for the ceiling's names: core/config.py, core/SBI/pipeline.py,
  core/gui/panels/inference/config_tab.py, help_text.py:47 and core/tool/__init__.py:34 only.

- **W099.** The 'must resume' refusal raised by the training stage carries no field key, and the generator's
  copy carries a key the window maps to nothing.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: Nothing visible today; if a resume choice is added, a refused 'must resume' would not say which
  control to change.
  Evidence: Same fault as the one listed in the other report. core/orchestrator.py:1110-1112 (no field);
  core/SBI/pipeline.py:1506-1512 (field='resume'); core/gui/fields.py:61 ('resume' maps to no control). The
  stage's refusal fires first whenever there is no cache to resume, so the generator's copy is reached only if
  the cache disappears in between.

- **W100.** The status line under the budget boxes loads every other cache's state and header files from disk,
  on the window's own thread, on each keystroke in the two boxes.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: Typing in the two budget boxes may stutter when many training caches exist; otherwise nothing
  visible.
  Correction: Holds in a narrower case. The disk reads happen only once the session holds both a config and a
  prior, the cache is on, and the typed budget matches no cache of its own with committed batches; then each
  keystroke loads two small files for every other committed cache, on the window's thread. Before a config and a
  prior exist, nothing is read from the store. The same lines are recomputed on the narrowing tab and on both
  tabs after every stage. With a handful of caches this is unlikely to be felt.
  Evidence: core/gui/panels/inference/posterior_tab.py:80-81, :338;
  core/gui/panels/inference/tsnpe_tab.py:94-95, :211; core/gui/panels/inference/base.py:134-139;
  core/orchestrator.py:541-555 (off, or no config or prior: nothing read; own cache with batches: one file read;
  otherwise the siblings are described); core/SBI/training_checkpoint.py:162-184, :222-223, :275-287 (two file
  loads per committed sibling).

### Cosmetic, or a page or comment that is wrong (36)

- **W101.** In 'Auto (time of day)' mode the palette and style sheet are re-applied every 60 s whether or not
  anything changed, theme_changed fires each time, and the theme flips at 07:00 and 19:00 mid-run.
  Area: the shell, start-up, Settings and the look; second reader: holds, with a correction.
  On screen: Only with the appearance set to 'Auto (time of day)': the whole window is restyled once a minute,
  and at 07:00 and 19:00 it switches between light and dark even during a run, leaving the figures already shown
  in the old colours.
  Correction: True, but only when 'Auto (time of day)' has been chosen; the default is 'Follow system'
  (settings.py:69-71, theming.py:61). The per-minute tick with an unchanged theme writes the same matplotlib
  values again, so it cannot by itself mis-colour a figure; only a real flip can. For the inference figures the
  save is already pinned to the colours the figure was built with (core/Helpers/visualizers.py:13-35, used at
  base_panel.py:58-59 and core/artifacts/store.py:406-407). The FDT, sweep and reduction plot modules still call
  savefig directly (core/FDT/plots.py:87,129,216,272; core/FDT/cross_validation_plots.py:127;
  core/Reduction/plots.py:104,193), but in the one I read the figure is built and saved inside one call
  (core/FDT/plots.py:62-88), so the exposed window is that call. Cost and visible effect of the per-minute
  restyle: not measured, cannot tell.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/theming.py:65-67 (60 s timer to _reapply), :81-84
  (started only in auto), :100-115 (_reapply always calls setColorScheme, setPalette, setStyleSheet and emits;
  no 'unchanged' check). Listeners: core/gui/mpl_theme.py:31-33, :42 (global matplotlib settings) and
  core/gui/widgets/live_hair_bundle.py:109-122 (the live view's pens and titles). theming.py:96-98: 'Follow
  system' re-applies on an OS flip. Figures already shown are not recoloured (mpl_theme.py:8-9).

- **W102.** The text-size scaling built into the design tokens never takes effect: the app font is pinned to 14
  px, so the scale is always 1.0 and the Windows text-size setting is ignored.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: Turning up the Windows 'make text bigger' setting makes no difference to PRISM's text or control
  sizes.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/fonts.py:27, :38-39 set the application font to a
  pixel size of 14; called at core/gui/app.py:100 before the style sheet is first built (:101-103) and again,
  still 14, when the Inter toggle changes (core/gui/main_window.py:251). core/gui/design.py:346-355 returns
  pixelSize()/14 clamped to at least 1.0, so 1.0 in the launched app; design.py:332-335 says the scale exists to
  follow the OS text setting. Its users: design.py:371-380, core/gui/widgets/help_badge.py:27. No other
  application-wide setFont exists (grep: only icons.py:70 per widget and artifact_screen.py:225). Display
  scaling (the monitor's percentage) is a separate mechanism and is not touched by this.

- **W103.** Settings → Help shows internal dictionary keys as the item names, and for Parameter Inference its
  summary is a developer docstring naming code classes.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: The help list in Settings labels its entries with internal names such as vram_ceiling or tsnpe_hpd
  instead of the labels printed on the screen, and the Parameter Inference summary reads like a programmer's
  note.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/screens/settings_screen.py:192-194 prints the
  first paragraph of each module's docstring and :195-200 prints each help entry as bold key then text.
  core/gui/panels/inference_tabs.py:1 is that first paragraph for Parameter Inference ('...ONE shared SbiSession
  owned by the InferenceScreen.'), and :20 re-exports the help table whose keys are e.g. 'tobs', 'tsnpe_hpd',
  'tsnpe_new_run', 'vram_ceiling' (core/gui/panels/inference/help_text.py:16, :26, :38, :47). Same for
  'freqs_per_batch', 'skip_sanity', 'record_name' (core/gui/panels/fdt_panel.py:39-46).

- **W104.** The header changes height between Home and every other screen (the back arrow is stacked under the
  title and takes no space when hidden), so the content and the run slot shift.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: Going between Home and any other screen, the strip at the top gets taller or shorter by the height
  of the back arrow, so the running label hops up or down and the page area changes size.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/screens/nav_shell.py:80-84 stacks title and back
  arrow in one vertical layout; :69 and :132-133 hide the arrow on Home and show it elsewhere; nothing asks Qt
  to keep the space of a hidden widget (grep for retainSizeWhenHidden: none); :86-89 centres the run slot
  vertically against that column; :106-110 puts the page stack directly below. Read from the layout code, not
  seen on a screen; the change always coincides with a change of screen.

- **W105.** Comments say the close saves splitter layouts; it does not, and a splitter drag made less than 1.5 s
  before quitting is lost.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: Nothing visible, except that a divider dragged in the last second and a half before quitting comes
  back at its old position.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/panels/base_panel.py:358-359 ('driven from
  BasePanel itself (restore) and MainWindow._save_state (save)') and :368 ('Called by MainWindow._save_state and
  the debounce timer'); core/gui/main_window.py:447-456 calls only each panel's save_settings and the Artifacts
  screen's. By grep the only callers of save_layout are base_panel.py:395 (the debounced write) and one test
  (tests/test_settings_persistence.py:679). The write is a single-shot 1500 ms timer restarted on every drag
  (base_panel.py:196-201), and nothing flushes it at close. The window's own docstring states the truth
  (main_window.py:80-83).

- **W106.** The design system has no colour for error, warning, success or a destructive action; severity is
  carried only by glyphs and message-box icons, and destructive buttons look neutral.
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: Warnings, errors and risky buttons are not colour-coded: a destructive choice looks like any other
  button, and trouble is marked only by a small symbol in the text or the standard box icon.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/design.py:50-69 and :70-89 are 18 colour tokens
  per theme, none for a state; :199-214 styles a neutral and an accent button only. My grep of core/gui outside
  design.py for setStyleSheet(, QColor(, hex colours, named colours, setForeground/setBackground found only the
  application style sheet (theming.py:112), the live view's background (widgets/live_hair_bundle.py:115) and the
  icon assets. Severity markers: core/gui/widgets/log_pane.py:8, core/gui/screens/artifact_screen.py:932-934 (a
  warning-sign prefix), and box icons (widgets/refusal_box.py:34, panels/base_panel.py:449). 'Quit anyway' and
  'Start a new run anyway' are added with a destructive role but no styling (main_window.py:433;
  panels/inference/posterior_tab.py:232).

- **W107.** The Prior tab's detect_device() is not memoised: it re-probes the card on every keystroke in four
  sweep boxes until a config is built.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: Nothing visible.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:118-119 (four boxes wired to
  _sync_sweep), :296-297 (detect_device() whenever the session has no config, which is again the case after
  every Apply). C:/Users/J/PycharmProjects/PRISM/core/config.py:64-85. Contrast
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/base.py:77-82. What each call costs was not
  measured.

- **W108.** The sweep note keeps saying 'each integrated over ? steps' after the config is built, until someone
  types in a sweep box.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: The note under the sweep boxes shows a question mark for the step count even after the
  configuration exists, until one of those boxes is edited.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:303-304 ('?' without a
  config's step), :119 and :121 are the only callers of _sync_sweep (search of core/gui), :351-354
  refresh_local_gates does not call it.

- **W109.** 'Prior ready: <name>. Name it below to keep it.' is written for a loaded, named prior too, and
  implies an unnamed prior is not kept.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: After loading a prior that already has a name the log still says to name it below to keep it,
  although every prior is kept on disk whether named or not.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:315-318 (one unconditional
  message for both the load and the build result). C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:726-745
  (every build is written before it is returned).

- **W110.** The read-only χ amplitude and band boxes look like editable fields when χ is on; the caption
  explaining them sits under the range row only.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: With the multi-frequency option ticked, the fixed amplitude and band boxes look exactly like boxes
  that can be typed in.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:71-75 (read-only), :82-86
  (amplitude row, range row, then the caption), :172-174 (enabled with the tick).
  C:/Users/J/PycharmProjects/PRISM/core/gui/design.py:219-227 styles a line edit for normal, focus and disabled
  only; no read-only rule. Read from the stylesheet, not seen on screen.

- **W111.** The units caption and the seeded typed-units text can list the tokens in a different order from one
  launch to the next.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: The line showing the model's units can list them in a different order from one launch to the next.
  Correction: True of the caption. The typed-units box is narrower: it is filled only while empty and its text
  is saved at close and restored before that check, so after the first clean close it keeps whatever order it
  was first given.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/Helpers/file_manager.py:250-262 (a tuple made from a set of
  strings). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:183-190, :405, :418-419.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/identity.py:24-27. A search of the repository's py, bat, sh,
  ini and md files finds no PYTHONHASHSEED, so string-set order is free to differ between processes.

- **W112.** The Config tab gives two thirds of its width to a figure stack and a progress row it never uses, and
  puts Hardware below the primary button.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: On the Config tab most of the width is an empty figure area above the log, and the Hardware box
  sits below the Apply button.
  Correction: The right-hand column also holds the log pane, which the Config tab does use for its 'Model
  applied' and warning lines; what is never used there is the figure area and the progress row. Two thirds is
  the starting split, which can be dragged and is remembered.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:142-183.
  C:/Users/J/PycharmProjects/PRISM/core/gui/design.py:29 (420 against 880).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:92 (button row) before :108-118
  (Hardware); log lines at :211, :308, :352, :357; the file contains no dispatch call.

- **W113.** One thing has three names in one flow: 'checkpoint' on the budget line and dialog, 'cache' on the
  TSNPE checkbox, 'simulation cache' in the stage's refusal.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: The saved simulation rows are called a checkpoint in one place, a cache in another and a simulation
  cache in a third, so it is not obvious they are the same thing.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/panels/inference/base.py:203-216
  'Checkpointing', 'checkpoint'; posterior_tab.py:228 'A checkpoint that is ONE setting away'; tsnpe_tab.py:102
  'a cache one setting away', :192-198 both words in one paragraph; orchestrator.py:584-587 'simulation cache',
  'committed cache'.

- **W114.** The fix sentence for a refused new run tells the user to answer a dialog on the Posterior tab, but
  on that tab the stage's refusal can only arrive when the dialog was not shown.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: In a rare case the yellow box would tell the owner to answer a dialog that never appeared.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/fields.py:62-63 the sentence.
  gui/panels/inference/posterior_tab.py:204-216: a shown dialog either cancels the click or sends new_run=True;
  new_run=False is sent only with no prior, a detector that raised, or no near miss found.
  orchestrator.py:1110-1114 the stage refuses only when it finds a near miss and new_run is False, so on this
  tab that needs the detector to have failed open or the store to have changed in between. The TSNPE half of the
  sentence is reachable and apt (tsnpe_tab.py:96-103, :160).

- **W115.** Tooltip text out of date: the Observation help claims a bitwise comparison with 'the one currently
  loaded'; Hidden features quotes '~46 h against ~57 h'; the VRAM ceiling help says off is right on an idle
  card.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: Three of the '?' help texts say things the last full-size run or the code contradicts: a check that
  does not exist, hour figures from a smaller run, and advice to leave the memory ceiling off.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/. core/gui/panels/inference/help_text.py:22-25 against
  core/gui/panels/inference/tsnpe_tab.py:148 (the picked record is loaded from the store; nothing compares it
  with the session's observation), core/orchestrator.py:1525-1531 (the observation is checked against its own
  digest) and :2627-2635 (against a narrowed parent's digest). help_text.py:87-89 the hour figures;
  help_text.py:47-48 'off is right on an idle card'. The contradicting measurements are records, not code:
  docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md:35, :38-39, :75-80.

- **W116.** After the pump stops on a dead signal, every later log line of the run is appended to a list nobody
  drains.
  Area: how a run is started, shown, cancelled and closed; second reader: holds.
  On screen: Nothing visible.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/streams.py:148-155 sets the stop flag on the emit's
  RuntimeError; :157-162 the loop then publishes once more and returns; :121-130 sink keeps appending to _logs;
  :164-173 stop() at teardown only joins the already-dead thread. Only log lines grow; rows are a keyed dict. It
  arises only if the run outlives the window, which is the unread library question under 'Quit anyway'.

- **W117.** When the one-run guard is bypassed, the second run's output is not captured and it cannot be
  cancelled.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: Nothing visible today.
  Correction: True of that branch, but nothing in the window can reach it: workers are created in exactly one
  place, behind the guard, and the guard flag changes only there and when a run reports finished. And the output
  would not be 'not captured': the first run's replaced console is process-wide, so the second run's prints and
  records would land in the FIRST run's tab, where they cannot trigger that run's cancel either. A live-stream
  run would still stop through its own polled flag. This is a constraint on any future queue or second runner,
  not a present fault.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/streams.py:272-276 (decline before arm at :284-285
  and before the streams at :288-294); :77-78 (a write from another thread never raises); grep over core:
  Worker( only at core/gui/panels/base_panel.py:240, redirect_streams( only at core/gui/worker.py:108, _running
  assigned only at base_panel.py:304 via _set_busy (:271, :288); the guard is base_panel.py:221-225; the
  live-stream stop flag is base_panel.py:254.

- **W118.** The stall notice counts in raw seconds in a 150 px label that can clip it.
  Area: how a run is started, shown, cancelled and closed; second reader: holds.
  On screen: During a long quiet spell the notice reads something like 'no output for 3712s', possibly cut off
  at the right edge.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/widgets/progress_pane.py:337 (the text, whole
  seconds), :211 (fixed width 150), :339 (same text on the tooltip, with the code's own remark that long text
  can clip). Whether a given count actually clips depends on the font; not measured.

- **W119.** Stale comments: the layout docstrings say the main window saves the splitters at quit, and vt.py
  cites sbi's epoch print at an old line number.
  Area: how a run is started, shown, cancelled and closed; second reader: holds.
  On screen: Nothing visible, apart from a splitter dragged in the last second and a half before quitting not
  being remembered.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/base_panel.py:359 and :368 name
  MainWindow._save_state as a writer; core/gui/main_window.py:447-456 calls only save_settings; save_layout( is
  called only at base_panel.py:395 (grep). MainWindow's own docstring is right (main_window.py:81-83).
  core/gui/vt.py:139 and :234 cite base.py:1024; the installed print is at
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/sbi/inference/trainers/base.py:1274.

- **W120.** After a cancel or a crash the FDT tab's last pane line is 'FDT run finished.'
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: After cancelling, or after an error box, the last thing written in the FDT tab's text pane is 'FDT
  run finished.', directly under 'Run cancelled.' or the error.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:249-254 (line appended on every
  outcome), 324 (passed as on_finished). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:263
  ('Run cancelled.' on the cancelled signal), 267-285 (_finished calls on_finished), 447-457 (error line
  appended, then a modal box whose own event loop delivers the queued finished signal).
  C:/Users/J/PycharmProjects/PRISM/core/gui/worker.py:179-215 (cancelled or error is emitted first, finished
  last). Only the FDT tab: the sweep tab passes close_figures alone
  (C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:333). The same line also follows a refusal
  raised by the worker.

- **W121.** The yellow box's fix line names every tab that has the control, not the tab in use.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: A blank box on the Simulate tab is answered with 'on the Infer or Live simulation tab', and a bad
  cell file with a list of five tabs.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/fields.py:57 (T_obs: Infer and Live simulation), 121-122
  (cell: five tabs), 150-154, 204-218 (names joined with ' or '), 221-232.
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/refusal_box.py:36-39 (that sentence is the box's second
  line).

- **W122.** 'Save model' re-registers every saved model on the GUI thread, including a script-compile step per
  model, although the screen's docstring says everything there takes milliseconds.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: The window may freeze briefly on every Save, longer the more models have been saved; the size of
  the pause is not known.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/model_builder_screen.py:10-11 (the claim), 530
  (called from the click handler, no worker). C:/Users/J/PycharmProjects/PRISM/core/registry.py:156-182 (loops
  over every saved model file; 167-169 builds the compiled step for each, under a comment saying the GUI thread
  never pays that cost). C:/Users/J/PycharmProjects/PRISM/core/Models/user_model.py:188-233 (torch.jit.script
  called unconditionally at 231, failures swallowed). The same loop runs once at launch
  (C:/Users/J/PycharmProjects/PRISM/core/gui/app.py:76). How long it takes was not measured: nothing was run.

- **W123.** The Cancel line and the close dialog speak of 'training' whatever is running.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Cancel a live simulation or a video export and the message still talks about waiting up to a minute
  'during training'.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:299-300 (one sentence for all ten
  panels: '(up to ~1 min during training)'). C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:423-431
  (the dialog is shown for any running panel, same wording). A Simulate stream checks the flag once per frame
  (C:/Users/J/PycharmProjects/PRISM/core/gui/panels/simulate_runner.py:219-220) and a video export once per
  frame (simulate_export.py:80-81, 129), so the wait named does not apply to them.

- **W124.** The saved-run details block prints raw setting keys and a UTC ISO timestamp where the form uses
  other labels.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Under the 'Saved run' picker the run is described by a line of programmer names and a timestamp in
  world time rather than local time.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/record_view.py:63-74 (created printed as stored;
  settings printed as key=value, sorted by key). C:/Users/J/PycharmProjects/PRISM/core/FDT/fdt_pipeline.py:60-66
  (the keys: n_freqs, ensemble_M, freqs_per_batch, F0, freq_bounds, burn_in_nd, T_obs_periods, dt_nd,
  psd_T_obs_nd, skip_sanity, confirm_production).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:331-332 and 614 (UTC, full ISO form with fractional
  seconds and +00:00). C:/Users/J/PycharmProjects/PRISM/core/gui/fields.py:150-154 (the form's labels differ).

- **W125.** Settings -> Help lists raw HELP-dict keys as item headings and opens each section with the module's
  '<MODE> mode:' docstring line.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: The Help page reads like a developer's notes: 'FDT mode: ...' followed by entries headed
  'ensemble_M', 'tobs', 'skip_sanity'.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/settings_screen.py:32-39 (first paragraph of the
  module docstring), 184-205 (each key printed in bold as the heading).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:1 and 32-63 (keys such as ensemble_M,
  skip_sanity, compare_list); C:/Users/J/PycharmProjects/PRISM/core/gui/panels/simulate_panel.py:1 and 33-43
  (tobs, frame, fps); crossval_panel.py:1; reduction_panel.py:1.

- **W126.** 'Add selected' does nothing, silently, for an empty or duplicate pick; the Compare buttons are
  always enabled with an empty list; a list can keep the id of a record deleted on the Artifacts screen.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Pressing 'Add selected' twice on the same run does nothing with no explanation; Compare with an
  empty or stale list starts a job that immediately ends in a yellow box.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/compare_list.py:30, 42-51 (returns False, says
  nothing). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:200-201, 234-236 and
  crossval_panel.py:189-190, 218-220 (button never gated on the list).
  C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:331-344 (only the pickers are re-listed after a
  delete). The stage then refuses with a yellow box:
  C:/Users/J/PycharmProjects/PRISM/core/FDT/compare.py:214-217 (too few) and 224-227 (names no record).

- **W127.** 'Start streaming' is enabled at launch even with no cell file because the gate runs only after a
  run; 'Start streaming' and 'Run reduction map' query the card on the GUI thread although neither uses it.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds, with a
  correction.
  On screen: Nothing visible, except that with an empty cell folder the Start button looks usable and answers
  with a line in the pane.
  Correction: Both halves are in the code, both are small. The gate half matters only when the chosen model's
  cell folder is empty, and the click then just writes 'Select a cell file first.'; changing the model does not
  re-run the gate either. The card half adds nothing new in practice: on a machine with a card the window has
  already initialised it at launch (the Posterior tab's budget lines read free memory during construction), so
  the click only repeats a cheap properties query. The reduction never reads the device it asked for.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/simulate_panel.py:56-66 (gate not called in the
  constructor), 98-105, 173-175; C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:333-335 (only
  caller, when a run frees the controls).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/simulate_runner.py:63, 75 with
  C:/Users/J/PycharmProjects/PRISM/core/cli.py:348;
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/reduction_panel.py:79 with cli.py:443;
  C:/Users/J/PycharmProjects/PRISM/core/config.py:64-70. Grep for hw, device, cuda and torch under
  core/Reduction finds nothing. Launch already does it: inference/posterior_tab.py:112, inference/base.py:77-82
  and 138-139.

- **W128.** Stale statements: crossval_panel says the excepthook is installed only after MainWindow is built;
  base_panel says MainWindow._save_state saves the layout.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Nothing visible, beyond a pane divider dragged just before closing the window not being remembered.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:241-243 against
  C:/Users/J/PycharmProjects/PRISM/core/gui/app.py:18-20 and 106-109 (installed before the window is built).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:359 and 368 against
  C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:447-456 (no layout call); grep finds save_layout
  called only from base_panel.py:395, the 1.5-second timer set at 196-201. Consequence read off the code: a
  divider dragged in the last second and a half before closing is not remembered.

- **W129.** The status line counts an unfinished cache or an interrupted FDT record as 'complete', beside a cell
  that says Finished: no.
  Area: the Artifacts screen; second reader: holds.
  On screen: The line under the table can say '1 complete, 0 incomplete' for a cache whose own row says
  Finished: no.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:337-338 (counts rows by 'has a
  manifest'). C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:132-138 and :147 (that flag is not 'the
  run finished'), :700 (finished is read from the body for caches and FDT records).
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_table.py:157 and :165 (the Finished cell).

- **W130.** Every simulation cache is called '(unnamed <digest>)' in the Name column and in the delete question,
  although a cache can never be named.
  Area: the Artifacts screen; second reader: holds.
  On screen: Every training cache appears as '(unnamed' followed by twelve characters, and its delete question
  reads 'Delete simulation (unnamed ...)?', though there is no way to give a cache a name.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:1582 (the cache's manifest is always
  written with an empty name, on every refresh), :160-162 (label falls back to '(unnamed <id>)').
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/artifact_table.py:166 (Name cell).
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:171 (delete question). The screen has no
  rename action (whole file read).

- **W131.** Messages use the store's internal kind words ('simulation', 'fdt') rather than the labels of the
  Kind list.
  Area: the Artifacts screen; second reader: holds.
  On screen: The list says 'Simulation caches (training rows)' but the messages say 'Delete simulation ...?' or
  'Could not read the fdt artifacts'.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:40-51 (the labels, used only to
  fill the list at :200-202), :171, :326, :475, :634, :640-641, :696, :701, :713, :726, :729 (sentences built
  from the bare kind key).

- **W132.** A delete refused for dependents is shown in the 'Check your inputs' box with the advice to select an
  artifact in the list, answering neither what was wrong nor what to do.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: When a delete is refused because other records depend on it, a yellow box titled 'Check your
  inputs' lists those records and says to delete them first, followed by an unrelated line about selecting an
  artifact in the list.
  Correction: Overstated. The box's main text does say what was wrong and what to do: it names the record, lists
  each record that depends on it with the reason, and ends 'Delete those first. Nothing here can orphan them.'
  What is off is only the frame: the window title 'Check your inputs' and the extra line about selecting an
  artifact in the list.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:643-670 (the message: the
  record, one line per dependent with why, 'Delete those first'), :699-703 (shown through the shared box, plus a
  status sentence). C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/refusal_box.py:33-41 (fixed title, message
  as text, the field's sentence beneath). C:/Users/J/PycharmProjects/PRISM/core/gui/fields.py:101 (the sentence
  for 'artifact').

- **W133.** A multi-line text inside a manifest is printed raw, so its continuation lines fall out of the pane's
  indented layout.
  Area: the Artifacts screen; second reader: holds.
  On screen: In a calibration's pane the informativeness description starts on an indented line and its
  following lines jump back toward the left margin.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:50 (a string is printed as is) and :90
  (one 'key: value' line, no re-indent). C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:2077 (a
  calibration stores the informativeness description in its body).
  C:/Users/J/PycharmProjects/PRISM/core/SBI/analysis.py:377-392 (that text is several lines joined by newlines).

- **W134.** Selecting a row re-parses every manifest of the kind twice and reads the whole log before keeping
  its tail, on the GUI thread; deleting a large cache blocks that thread with no progress.
  Area: the Artifacts screen; second reader: holds.
  On screen: Probably nothing noticeable at today's store size; at most a short pause when selecting a record
  with a very long log or deleting a multi-gigabyte cache.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:381 and :389 (two store calls
  per selection, no worker anywhere in the file), :716 (delete called directly).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:659-679 and :737-741 (each call scans and parses
  every manifest of the kind), :786-795 (whole file read, then sliced), :937-947 (blocking removal). Cost not
  measured and probably small: a manifest is a few kilobytes, and a cache is two files per save at one save per
  50 batches (C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:386-387;
  C:/Users/J/PycharmProjects/PRISM/core/config.py:303).

- **W135.** The tracker argument of train_nn is not a live per-epoch hook, although its docstring reads as one;
  sbi calls the tracker only once, after the training loop has ended.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: Nothing visible; it means a live loss curve during the fit cannot be had by passing a tracker and
  needs another route.
  Evidence: core/SBI/train.py:208-211 (docstring: 'what sbi reports its per-epoch curves to'), :241-242.
  Installed library,
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/sbi/inference/trainers/base.py:978-998 (inside the
  loop only the in-memory per-epoch summary at :993 and the counter print at :998), :1009 (the one call that
  reaches the tracker, after the loop), :1182-1250 (every tracker call is inside that function); a grep of the
  trainers folder for the tracker finds no other call on this path.

- **W136.** A posterior's id and creation time are those of its END, its recorded wall time covers only the last
  process, and a cache built over several processes shows only the first process's git revision.
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds.
  On screen: A posterior's date in the lists is when its run ended, not when it began, and its recorded duration
  leaves out earlier sessions of a resumed run.
  Evidence: core/artifacts/store.py:802-813, :846-850 (id and creation time are taken when the writer is
  created), which is core/orchestrator.py:1420, after the fit; core/orchestrator.py:900, :1482 (the wall time is
  measured from the start of this call only); core/artifacts/store.py:1575, :1584 (the cache's record keeps the
  first process's creation time and revision across resumes).

## B. Faults the second readers added

- **W137.** The delete confirmation for an unfinished training cache never says the cache may still be being
  written by a run in another window or at a terminal, the store's delete has no in-use check, and the 'a task
  is running' refusal sees only this window's own run.
  Area: the shell, start-up, Settings and the look; at stake: can lose days of work. Read by one reader only.
  On screen: From a second PRISM window, the Artifacts screen will delete the training cache a live run is still
  writing after one confirmation that never mentions it might be in use.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/screens/artifact_screen.py:128-132 (the cache's
  sentence: batches are thrown away, nothing about a live run) against :133-136 (the measurement records'
  sentence, which does warn) and :122-126 (the docstring: there is no cross-process lock, and a live measurement
  was once deleted on the strength of the old wording). The guard is this process only (:580-590).
  core/artifacts/store.py:918-947: delete checks dependents and then removes the tree; an unfinished cache has
  no dependents yet. The leftover sweep has a recent-write guard (store.py:960-965); delete has none. Sequence
  that loses days: a generation is live in window A or at a terminal; in a second window open Artifacts, choose
  'Simulation caches', select the unfinished cache, press Delete…, answer Yes (default is No). Whether Windows
  lets the removal finish while A holds a file open was not established; a half-removed cache is the other
  outcome (artifact_screen.py:723-728). The cross-process half is already lead 14 of the owner's note; the
  missing warning in this prompt is the new part.

- **W138.** Loading a saved prior is no check on the bounds file, contrary to the first reader's addition under
  [2A bullet 4] ('with an existing prior picked on the wrong bounds the store refuses the load, so only the
  from-scratch state builds on the wrong box'). The load compares only the model, the non-dimensional names,
  their ranges and the log mask, and the three shipped Nadrowski bounds files are identical in all of those;
  they differ only in the dimensional and forcing sections, which the load rebuilds from whatever file is
  showing. Training adds only a log-mask check. Sequence to a spoiled multi-day run: launch; Apply; the Bounds
  list shows the ordinary box (the launch default, a look-and-close overwrite, an Apply re-select, or the rescan
  button); select the saved tier-1 prior; Build / Load prior: it loads, and the 'Config built' line reads as it
  would for the temperature box; Posterior tab, '(from scratch)', Train: three key fields differ from the tier-1
  cache, the dialog needs exactly one, and simulation starts from zero on the box without temperature.
  Area: the Config and Prior tabs; at stake: can lose days of work. Read by one reader only.
  On screen: With the wrong bounds file showing, loading the saved prior goes through without a word and the log
  reads exactly as it would for the right file; a training started from there simulates from zero on the wrong
  parameter set, with no dialog.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:1275-1291 (the four comparisons),
  :1297-1300 (the dimensional and forcing priors are rebuilt from the current config).
  C:/Users/J/PycharmProjects/PRISM/Resources/Bounds/nadrowski/master.txt:1-11, master_spont.txt:1-11 and
  master_tier1.txt:1-11 are identical; they differ from line 13.
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:663-671 (the load branch adds no other check),
  :1006-1024 (training checks only the log mask).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:216-219.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/identity.py:92-96.
  C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:347-350.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:214-216. A search of
  C:/Users/J/PycharmProjects/PRISM/core finds no code that compares a prior's recorded bounds file or
  dimensional ranges with the current config.

- **W139.** The window is started from a console window that stays open for the whole session, and the program
  dies with it: closing that console ends any run at once, with none of the window's 'A task is still running'
  questions. Sequence: start with the launcher -> start a training -> close the black console window -> process
  gone; during the fit the whole fit is lost, during generation everything since the last cache commit.
  Area: how a run is started, shown, cancelled and closed; at stake: can lose days of work. Read by one reader
  only.
  On screen: Closing the black console window that sits behind the program ends the program and any run
  immediately, with no question asked.
  Evidence: C:/Users/J/PycharmProjects/PRISM/run.bat:9 runs `python -m core.gui` in the foreground of the batch
  file's console and :16 `pause` keeps that console open; no `start`, no windowless interpreter. grep over
  C:/Users/J/PycharmProjects/PRISM/core for signal.signal, SetConsoleCtrlHandler, FreeConsole, pythonw: no
  match. The close question lives only in MainWindow.closeEvent (core/gui/main_window.py:416-445), which that
  route never reaches. That Windows ends a console's attached processes when the console window is closed is the
  operating system's behaviour, not something read from the repository.

- **W140.** A training cache deleted while its run is still generating is silently rebuilt as a broken partial
  folder, and the run reports nothing. The save re-creates the folder, writes only the rows since the last save
  together with a state that claims every batch so far, and the manifest update, which needs the deleted header,
  fails inside a catch-all that discards the error. At the end the run still logs that the cache is complete and
  worth keeping for a retrain. The first reader listed the missing warning in the delete question but not what
  the run does afterwards. Read, not run; reachable only from a second process (a second window or a terminal),
  because the owning window refuses Delete.
  Area: the Artifacts screen; at stake: can lose days of work. Read by one reader only.
  On screen: The training carries on and even announces a complete cache, but the folder on disk holds only the
  last part of the rows and shows on the Artifacts screen as an interrupted leftover; if the fit is then stopped
  or fails, the whole simulation is gone and cannot be resumed.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:371-372 (folder re-created with
  parents), :384-395 (rows of this save only; state with the full batch count), :465-474 (manifest refresh reads
  the header; any exception is passed over), :398-411 (completion likewise).
  C:/Users/J/PycharmProjects/PRISM/core/Helpers/file_manager.py:34 (atomic writes re-create a missing parent
  folder). C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:2034-2041 (final save, the 'complete ...
  keeping it lets you retrain the flow without re-simulating' line, rows returned from memory).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:667-668 (the rebuilt folder, having no manifest, is
  then listed as an interrupted leftover that a sweep may remove).

- **W141.** 'Type units' always gives a different cache key from 'Model's units file', even when the typed
  tokens are the same as the file's, and which of the two is selected is remembered between launches. A training
  started with 'Type units' selected therefore never finds a cache simulated under the units file. If that is
  the only difference the one-setting dialog appears, naming the field by its internal key with two hash
  strings; if anything else differs as well there is no dialog.
  Area: the Config and Prior tabs; at stake: can mislead a long run. Read by one reader only.
  On screen: With the typed-units choice selected, a training does not pick up simulations made under the
  model's units file even though the units are the same; at best a dialog names an internal setting with two
  long codes, otherwise the run starts from zero.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:192-197 and :251-260 (typed
  mode sends a token tuple), :404 and :420 (the toggle is saved and restored).
  C:/Users/J/PycharmProjects/PRISM/core/cli.py:313-321, :349 (a token tuple leaves no units path in the config's
  sources). C:/Users/J/PycharmProjects/PRISM/core/artifacts/identity.py:28-32, :56-67 (a file is fingerprinted
  by its bytes, tokens by a prefixed sorted list that is by design never a file's fingerprint).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:220-222 (the dialog prints the key
  name and both values cut to 60 characters).

- **W142.** A training that runs with the rotation switched off never says so, and in chi mode the line printed
  when a model is applied never states the rotation either way.
  Area: the engine side (stages, the store, the fit, the cache); at stake: can mislead a long run. Read by one
  reader only.
  On screen: During a multi-day run there is no line anywhere on screen that says whether the rotation is on;
  the only sign is the tick box on the first tab.
  Evidence: core/orchestrator.py:1218, :1240, :1251 are the only rotation lines of an ordinary training and all
  sit on the 'on' paths; the unrotated branch at :1255-1256 logs nothing.
  core/gui/panels/inference/config_tab.py:343-351: the Apply line adds 'decorrelating Fisher rotation on' only
  in the branch taken when chi mode is off and the rotation is on.
  core/gui/panels/inference/prior_tab.py:216-237: the 'Config built' lines do not mention it. The finished
  record does keep the flag (core/artifacts/manifest.py:282).

- **W143.** Nothing stops or warns a second PRISM window from being opened while one is already open or running;
  the project's own rule against a second window beside a long run is not enforced anywhere.
  Area: the shell, start-up, Settings and the look; at stake: everyday use. Read by one reader only.
  On screen: Starting PRISM a second time simply opens a second full window, with no hint that one is already
  open or that a run is live in it.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/app.py:61-127 builds the application with no check
  for another instance; my grep of core for QLockFile, QSharedMemory, QLocalServer, lock-file libraries and
  'single instance' found nothing (the only 'already running' is the in-process message at
  core/gui/panels/base_panel.py:221-225). Both windows would use the same settings file
  (core/gui/settings.py:20-23), written at each accepted close (core/gui/main_window.py:447-456), so the last
  one closed decides the remembered selections. The run guards are per process
  (core/gui/screens/artifact_screen.py:580-590). That a launch reads the card is the first reader's own
  confirmed claim; I did not re-trace it.

- **W144.** A run whose output capture fails to start ends with no message at all: the worker treats any
  RuntimeError raised outside the stage call as 'the window was closed' and returns silently.
  Area: the shell, start-up, Settings and the look; at stake: everyday use. Read by one reader only.
  On screen: In a rare low-resource state, pressing a run button would grey the controls and release them again
  at once, with no line in the log and no error box.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/worker.py:107-108 enters the capture; :207-210
  catches every RuntimeError raised outside the inner try and returns, and :211-215 still emits 'finished',
  which unlocks the controls (core/gui/panels/base_panel.py:267-285). core/gui/streams.py:287 starts the pump
  thread inside the capture's set-up, and :278-281 names the failure this can raise (thread exhaustion in a
  torch process, or interpreter shutdown); Python raises that as a RuntimeError. Nothing is written to the log
  pane and no box opens on that path. Narrow: I found no other RuntimeError source outside the stage call
  besides the dead-window emits the clause was written for.

- **W145.** Popped-out figure windows are separate parentless windows that the main window's close does not
  close, and the application never asks to quit on its own.
  Area: the shell, start-up, Settings and the look; at stake: everyday use. Read by one reader only.
  On screen: If a figure has been popped out into its own window, closing the main window probably leaves that
  figure window on screen and PRISM still running without its main window.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/widgets/figure_window.py:18-20 ('Both are
  parentless top-level windows', held only by the figure stack); core/gui/widgets/figure_stack.py:112-131 shows
  them; core/gui/main_window.py:416-445 closes nothing else and my grep of core/gui for .quit(, closeAllWindows,
  setQuitOnLastWindowClosed, aboutToQuit found none; core/gui/__main__.py:16-19 just runs the event loop. What
  Qt then does when the main window closes with a pop-out still open was not run: under Qt's documented
  last-window rule the application would keep running until the pop-outs are closed. This also bears on the
  'Quit anyway' question above (with a pop-out open the application may not exit at all).

- **W146.** The typed-units box is filled from a model's units file only while it is empty, and its text is
  saved at close and restored before that check, so after the first launch it never follows a change of model or
  an edited units file. The code's own comment says switching to 'Type units' 'starts from the truth'. Applying
  in typed mode after choosing another model declares the first model's units for it.
  Area: the Config and Prior tabs; at stake: everyday use. Read by one reader only.
  On screen: Switching to typed units after choosing another model shows the first model's units rather than the
  chosen model's, and applying them declares those.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:180-190 (seeded only when
  blank), :405 (saved), :418-419 (restored before the seeding call), :251-260 (typed tokens are what Apply
  uses). C:/Users/J/PycharmProjects/PRISM/Resources/Units/bp/units.txt:2 ('nm ms') against
  C:/Users/J/PycharmProjects/PRISM/Resources/Units/nadrowski/units.txt:2 ('nm ms pN kHz').
  C:/Users/J/PycharmProjects/PRISM/core/cli.py:322, :343-345 (the declared time unit sets the config's time
  grid).

- **W147.** Loading a stored posterior while the session holds a config but no prior appears to crash with an
  internal error instead of loading, although the help and the gating say a load needs no prior. Read, not run.
  Area: the Posterior, Validate, Infer and TSNPE tabs; at stake: everyday use. Read by one reader only.
  On screen: After a prior build that was refused, failed or cancelled, choosing a saved posterior and pressing
  the button brings up a red error box about an internal fault instead of loading it.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. gui/panels/inference/posterior_tab.py:119 only
  the from-scratch branch is stopped without a prior; :148-150 a load is dispatched with session.inf_prior,
  which is None; :324-330 the button stays enabled for a load ('loading an existing posterior is always
  allowed'). orchestrator.py:902 'inferred, force_prior = prior.prior, prior.force_prior' runs unconditionally,
  before the load branch at :976, so a None prior raises AttributeError, which is not a refusal and goes to the
  red box (gui/panels/base_panel.py:444-457). gui/panels/inference/help_text.py:20-21 says loading does not need
  a prior; gui/screens/inference_screen.py:193 says inference needs no prior. The state is reachable:
  gui/panels/inference/prior_tab.py:209-210 installs the config and clears the prior before the prior stage
  runs, so a prior build or load that the stage refuses, that fails or that is cancelled leaves config without
  prior, and the Posterior tab is enabled on the config alone (inference_screen.py:196). posterior_tab.py:142
  has by then already dropped any held posterior. No code or test calls the stage with a None prior (grep over
  the repository for build_posterior(<cfg>, None found nothing).

- **W148.** Each config build re-lists the Cell picker, and every intermediate selection during the re-list is
  validated against the new config, so a warning about a cell nobody picked can be written to the Infer tab's
  log on every prior build or load. Read, not run; depends on the folder's listing order.
  Area: the Posterior, Validate, Infer and TSNPE tabs; at stake: everyday use. Read by one reader only.
  On screen: On the Infer tab's log, a warning that a cell 'does not fit the bounds file' may appear after every
  prior build or load, naming a cell the owner never selected.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/. core/gui/panels/inference/infer_tab.py:77 the
  picker's index-change signal is wired to _on_cell_changed; :189-207 that validates the selected file against
  session.cfg and logs 'This cell does not fit the bounds file ...'. core/gui/widgets/artifact_picker.py:44-57 a
  re-list clears and refills the combo, :88-100 then re-applies the remembered key;
  core/gui/panels/inference/prior_tab.py:343-345 states in the repository's own words that a refresh fires a
  batch of index-change signals. core/gui/screens/inference_screen.py:179-180 the new config is already
  installed when the re-list runs; infer_tab.py:210-211 a further explicit validation follows. With the tier-1
  box the folder's alphabetically first file has no temperature entry
  (Resources/Cells/nadrowski/master_entrained.txt:18-21 against Resources/Bounds/nadrowski/master_tier1.txt:16),
  so it would be reported if it lists first; the order is the file system's
  (core/Helpers/file_manager.py:316-322), which I could not establish.

- **W149.** Pressing Train unloads the posterior the session was holding before the new run starts, and nothing
  puts it back if that run is cancelled or fails. The stored record is untouched.
  Area: how a run is started, shown, cancelled and closed; at stake: everyday use. Read by one reader only.
  On screen: After a training that is cancelled or fails, the window no longer has the previously loaded
  posterior and it must be loaded again before validating or inferring.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/inference/posterior_tab.py:139-143
  (reset_downstream('posterior') just before dispatch); core/gui/session.py:49-57 (sets the session's posterior
  to None); only the success handler sets it again (posterior_tab.py:287-288); the cancel line and the error
  path do not touch the session (core/gui/panels/base_panel.py:263, :428-457).

- **W150.** A sweep study cancelled or crashed during its second sweep never announces or lists the first sweep,
  although that record is finished and on disk.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; at stake: everyday use. Read by
  one reader only.
  On screen: Cancel during the second half of a sweep study and the first half, which finished, is missing from
  the 'Saved sweep' list and from the pane until the rescan button is pressed or the window is restarted.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/FDT/cross_validation.py:643-649 (S sweep first) and 587-592
  (each sweep commits its own record when it ends).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/crossval_panel.py:330-333 and 335-349 (the 'Sweep record
  written' lines and the picker re-list happen only in the result handler).
  C:/Users/J/PycharmProjects/PRISM/core/gui/worker.py:179-205 (no result is emitted after a cancel or a
  failure). Nothing else re-lists the picker except its own rescan button and a change made on the Artifacts
  screen (C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:124, 331-344).

- **W151.** The FDT tab's Run button is not re-judged when the selected user model is edited and saved, so it
  can stay grey for a model that now qualifies.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; at stake: everyday use. Read by
  one reader only.
  On screen: Fix a model so that FDT can run on it, go back to the FDT tab with that model still selected, and
  the Run button is still grey until another model is picked and then this one again.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:256-263 (the button is gated only
  inside the model-changed handler). C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:310-329 (after a
  model save the handler is re-fired only when the selected name changed). FdtPanel has no gate of its own to
  re-apply (grep for refresh_local_gates: not defined in fdt_panel.py).

- **W152.** Deleting a user model removes its whole Bounds, Cells and Units folders, including any files added
  by hand, while the confirmation speaks only of its 'generated' files.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; at stake: everyday use. Read by
  one reader only.
  On screen: Extra cell or bounds files written by hand for a model disappear with it, although the question
  asked before deleting mentions only the files the builder made.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/Helpers/model_store.py:403-416 (rmtree of each folder).
  C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:288-296 ('This removes its definition and its
  generated Bounds/Cells/Units files.').

- **W153.** Comments and test docstrings still say the last-resort error hook is installed only after the main
  window is built; it is installed before, and construction failures are caught by a separate try in any case.
  Area: the shell, start-up, Settings and the look; at stake: cosmetic, or a page or comment that is wrong. Read
  by one reader only.
  On screen: nothing visible
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/panels/crossval_panel.py:241-244 and
  tests/test_nav_and_gating.py:174-177, :2658-2661 against core/gui/app.py:15-21 (docstring: installed BEFORE
  the window) and :106-120 (hook installed at :107, window built at :109 inside a try that shows 'PRISM could
  not start').

- **W154.** The stage's own near-miss refusal, which is the only form of that warning on the TSNPE tab, is
  worded for a programmer: it prints the cache's full folder path, the identity key and both values in code
  form, and ends 'To start a new cache anyway, pass new_run=True.'
  Area: the Posterior, Validate, Infer and TSNPE tabs; at stake: cosmetic, or a page or comment that is wrong.
  Read by one reader only.
  On screen: When a narrowing round is refused for being one setting away from an existing run, the yellow box
  shows a folder path, internal setting names and the instruction to 'pass new_run=True', with the usable advice
  (tick the box) only in the smaller line beneath.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/core/. orchestrator.py:567-575 and :578-587 build the
  text; :1113-1114 raise it with field 'new_run'. gui/widgets/refusal_box.py:33-39 shows the message verbatim
  with the fix sentence beneath it; gui/panels/base_panel.py:424-426 copies it to the log.
  gui/panels/inference/tsnpe_tab.py:96-103 states that on this tab the refusal arrives from the stage and the
  tick box is the answer.

- **W155.** The success line 'Posterior ready: ... Name it below to keep it.' (and the same for a prior) implies
  an unnamed result may not be kept, but the record is already committed to the store before the line is written
  and nothing removes unnamed records on its own; the Save button only renames it.
  Area: how a run is started, shown, cancelled and closed; at stake: cosmetic, or a page or comment that is
  wrong. Read by one reader only.
  On screen: After days of training the closing line suggests the result will be lost unless it is named, when
  it is in fact already saved.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/inference/posterior_tab.py:296-297 and
  prior_tab.py:317-318 (the wording); core/orchestrator.py:1420-1483 (the record is written before the stage
  returns); core/artifacts/store.py:4-5 (an unnamed record's folder is `_unnamed__<id>`); posterior_tab.py:307
  (Save is a rename); store.py:1227 `unnamed()` has no caller anywhere under core (grep).

- **W156.** 'Open model builder' blanks the name, variables and parameters but leaves the two scale boxes at
  whatever the last edited model had.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; at stake: cosmetic, or a page or
  comment that is wrong. Read by one reader only.
  On screen: A new model started after editing an old one opens with the old one's length and time scales
  already filled in.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/model_builder_screen.py:556-569 (reset does not
  touch the scale boxes), 546-547 (editing a model fills them), 281-282 (their defaults, 10 and 0.01).

- **W157.** With 'Skip sanity checks' ticked, the 'Proceed to the production sweep' box is greyed in whatever
  state it was left; left unticked, the form shows 'do not proceed' while the production sweep runs.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; at stake: cosmetic, or a page or
  comment that is wrong. Read by one reader only.
  On screen: The form can show a greyed, unticked 'proceed' box beside a run that does proceed.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:144-148 (the box is only disabled, its
  tick is kept). C:/Users/J/PycharmProjects/PRISM/core/FDT/fdt_pipeline.py:335-336 (with the skip, the
  production run goes ahead regardless) and 292-297 (a comment acknowledging the window can pass False there).

- **W158.** A training cache's pane prints its full list of identity settings twice, once under Knobs and once
  under Body, because the cache's manifest stores the same dictionary in both places and the renderer prints
  both sections whole.
  Area: the Artifacts screen; at stake: cosmetic, or a page or comment that is wrong. Read by one reader only.
  On screen: Selecting a training cache shows the same long block of settings two times in the right-hand pane,
  one under each heading.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:1588 (config = the identity) and :1591
  (body.identity = the identity again). C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:172-173 (Knobs
  and Body both rendered in full).

- **W159.** The command line's own description of how it pairs a posterior with its prior states a check that
  the training stage does not make for an ordinary (amortized) posterior.
  Area: the engine side (stages, the store, the fit, the cache); at stake: cosmetic, or a page or comment that
  is wrong. Read by one reader only.
  On screen: Nothing visible; a reader of the code is told the engine guards the pairing when, for an ordinary
  posterior, only the command line's habit does.
  Evidence: core/tool/config_args.py:348-354 says build_posterior refuses a prior that is not the one the
  posterior's record names. core/orchestrator.py:976-992: the load branch compares the prior only when the
  posterior carries a region; core/artifacts/store.py:1302 takes no prior at all. The tool is safe because it
  always loads the recorded prior (config_args.py:356-358); the window relies on the check that does not exist.

- **W160.** If a cache's own record cannot be read at the moment it is refreshed, it is silently rewritten as if
  new: a new creation time, the current code revision, an empty note, and no parent or input files named. The
  refresh also swallows every error of its own without a line.
  Area: the engine side (stages, the store, the fit, the cache); at stake: cosmetic, or a page or comment that
  is wrong. Read by one reader only.
  On screen: Rarely, a training cache's entry in the Artifacts list would lose its note, its start date and its
  link to the prior, or show a stale batch count, with nothing in any log to say why.
  Evidence: core/artifacts/store.py:1570-1588: an unreadable or invalid existing record is suppressed
  (:1573-1574) and every kept field then falls back to a fresh value (:1575, :1583-1588).
  core/SBI/training_checkpoint.py:465-474: the refresh passes no parents, inputs or hardware, and catches every
  exception with no message. The rows and the commit point are not touched (the state file is the commit point),
  and a prior's delete guard still finds the cache by fingerprint (core/artifacts/store.py:908-915).

## C. The first note's claims that did not simply hold

The first note is `2026-10-05-the-stopped-retrain-and-the-window-route.md`. Of its claims re-checked
by the first readers, 127 held as written and are not repeated here. These are the rest, as the second
readers left them.

- **W161.** [2B bullet 4] The whole window is locked for the run: every panel's controls are disabled and only
  Cancel stays live (first reader: partly).
  Area: the shell, start-up, Settings and the look; second reader: holds, with a correction.
  On screen: During a run the left-hand form of every tab is greyed out, but you can still move between screens
  and tabs, change the appearance, browse artifacts, pop out figures and scroll logs; only changes to stored
  records and models are refused.
  Correction: Only the left-hand form of each run panel is disabled. Still live during a run: Home, the back
  arrow, every tab, the gear and the appearance choices (which restyle the window and change the colours of
  figures built afterwards), the Settings screen, opening and editing in the model builder, browsing, saving
  text and writing a lineage report on the Artifacts screen, figure tabs and pop-outs, the log pane and the
  dividers. Refused during a run: starting a second run, a note, delete or sweep on the Artifacts screen,
  validating or saving a model, deleting a model. Because the form is disabled, the Save-name boxes for a prior
  or posterior are also unusable for the whole run.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/panels/base_panel.py:316-317 and :333 disable only
  each panel's controls widget; the Cancel button sits outside it (:150-157). core/gui/main_window.py:401-403
  ('Nothing is DISABLED here ... navigation stays free'); :216-251 appearance and font changes are unguarded;
  :283-287 model delete refused. core/gui/screens/artifact_screen.py:580-590 with :617, :682, :829 (writes
  refused), :439 (lineage is a read, never refused). core/gui/screens/model_builder_screen.py:394-398, :510-512.
  Second run refused: base_panel.py:221-225. Name boxes inside the form:
  core/gui/panels/inference/posterior_tab.py:55-63.

- **W162.** [2C bullet 4] 'Quit anyway' probably loses the record at the end, although the dialog says the task
  keeps running in the background until it finishes (first reader: not established).
  Area: the shell, start-up, Settings and the look; second reader: holds, with a correction.
  On screen: Choosing 'Quit anyway' during a fit closes the window with a promise that the task will finish in
  the background; on every reading of the code but one there is no trained posterior afterwards and nothing
  anywhere says why.
  Correction: Every link inside the repository checks out; the one link that decides the outcome is outside it
  and was not run. Sequence: start a training on the Posterior tab; while it is fitting, close the window;
  choose 'Quit anyway'. Certain: nothing is done to the run and the window closes (main_window.py:440-445).
  After that there are three possible outcomes, and only what Qt and PySide do to a pool thread at process exit
  (compiled code, no source installed) picks one. (a) The process ends with the window: the fit, which has no
  checkpoint (core/SBI/train.py:365-370), is lost at once. (b) The process lives on but the window's signal
  objects are gone, which is what the code's own comments say happens (worker.py:208-209, streams.py:153-154):
  the fit runs to its end, the posterior file is written (orchestrator.py:1420-1421), the loss figure is then
  forwarded to the window from inside the same write (orchestrator.py:1429-1431; store.py:404-409;
  base_panel.py:62, unguarded), that raises, the writer deletes the whole posterior folder (store.py:434-436,
  :479-481; posteriors are not a keep-on-failure kind, store.py:73), and the worker swallows the error with
  nothing shown or printed anywhere (worker.py:146-150, :204-210; streams.py:286-290). (c) The process lives on
  and the signal objects still work: the posterior is kept. Only (c) keeps the result; the simulated rows
  survive in all three. A two-batch trial on the real screen (train, close, 'Quit anyway', then look for the
  posterior) would settle which it is.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: dialog text core/gui/main_window.py:428-431; buttons and
  branches :432-445. The training is dispatched with the window's figure sink
  (core/gui/panels/inference/posterior_tab.py:148-150; core/gui/panels/base_panel.py:246-247; passed on at
  core/orchestrator.py:2655). core/orchestrator.py:1420-1431; core/artifacts/store.py:401-412, :434-481, :73;
  core/gui/panels/base_panel.py:44-63; core/gui/worker.py:105-215; core/gui/streams.py:141-155, :282-300.
  Nothing in core/gui waits for the worker or handles application quit (grep for waitForDone, aboutToQuit,
  setQuitOnLastWindowClosed, .quit(: none); core/gui/__main__.py:16-23. The guide repeats the dialog's promise
  (docs/guide/window.md:78-80).

- **W163.** [2F] 'Quit anyway' is not in the display checklist (first reader: partly, the rest of the list not
  read).
  Area: the shell, start-up, Settings and the look; second reader: holds.
  On screen: The record of things tried on a real screen has never included pressing 'Quit anyway', nor a
  training of real size.
  Evidence: C:/Users/J/PycharmProjects/PRISM/docs/checklists/display-walkthrough.md: a whole-file search for
  'Quit anyway' finds nothing; row 20 (line 29) covers the three-button dialog and names only 'Cancel task &
  quit', passed 2026-09-11. For the other items the note lists as absent, the same file has no 'VRAM', '2,048',
  '256' or 'resum…'; its only 'ceiling' is the lock-in ceiling (line 99), its only 'red box' is an FDT row
  saying none appears (line 190), and its 'stall' hits are inside 'installs' (lines 60, 70). Training rows are
  at 2 batches (lines 15-16, 20). A word search cannot rule out a paraphrase.

- **W164.** [3B item 8] Lead: establish what 'Quit anyway' during a fit and 'Cancel task & quit' during
  generation really do, and make each safe or remove it (first reader: not established).
  Area: the shell, start-up, Settings and the look; second reader: could not be settled by reading.
  On screen: What the two quit buttons really do to a live run cannot be known without trying them: one may
  silently discard days of fitting, the other at worst loses the simulated batches since the last automatic
  save.
  Correction: What the two buttons do inside the window is certain; what happens to the run afterwards depends
  on Qt's behaviour at process exit, which is not in the repository and was not run. 'Quit anyway' during a fit:
  see the 2C bullet 4 entry; close the window while fitting, choose 'Quit anyway', and in two of the three
  possible outcomes the days of fitting are lost, in one of them silently at the very end. 'Cancel task & quit'
  during generation: the cancel is requested and the window closes at once without waiting
  (main_window.py:438-445); the worker notices at its next line of output (streams.py:204-205) and saves the
  batches finished since the last automatic save on its way out (core/SBI/pipeline.py:1951-1990); that save
  touches no window object (records go to an in-memory queue, streams.py:251-256, :141-155), so it completes if
  the process stays alive long enough. If it does not, the loss is bounded: at most the batches since the last
  save, which the window makes every 50 batches (core/config.py:303; core/orchestrator.py:434; the tab passes no
  cadence, posterior_tab.py:171-181). That is hours at most, not days. During a fit the same button discards the
  fit by design.
  Evidence: Under C:/Users/J/PycharmProjects/PRISM/: core/gui/main_window.py:416-445;
  core/gui/panels/base_panel.py:114-118; core/gui/streams.py:141-155, :191-205, :251-256;
  core/SBI/pipeline.py:1951-1990; core/config.py:303; core/orchestrator.py:434, :1420-1431;
  core/artifacts/store.py:434-481. The one real-screen record: docs/checklists/display-walkthrough.md:29
  ('Cancel task & quit' stops cleanly, 2026-09-11, at a small run). No PySide or Qt source is installed to read
  the exit behaviour from (no .cpp or pyside*.h under
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/PySide6).

- **W165.** [2A bullet 8, partly] The VRAM ceiling is locked while a run is live; the lock is the app-wide
  control lock, not anything in the Config tab.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: While anything is running the ceiling box is greyed out with every other control, so changing it
  means cancelling the run first.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:108-118 (the Hardware box
  sits in the controls column). C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:302-317 and
  :323-335 (every panel's controls column is disabled for any run in any panel).
  C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:584-603, :635-639 (the planner reads the ceiling afresh
  at each batch plan, so only the window stops a change mid-run).

- **W166.** [2E bullets 2-3, partly] Keep the ceiling field with its free-memory note and the dialog before a
  session is replaced; but the free figure can be hours stale and the 'Off' note advises against a ceiling.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: The free-memory figure under the ceiling box is only as fresh as the last keystroke in that box,
  and the question before a session is replaced appears on Apply only, not on the Prior tab's button.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:111-119 (the note is
  rebuilt at construction and on a keystroke in that box, nowhere else), :150-165, :263-303.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/provenance.py:114-128. One limit on the second half of the
  claim, read in the same code: the dialog guards Apply alone; the Prior tab's button releases the same things
  unasked (C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/prior_tab.py:209), and Apply shows no
  dialog when only an applied model is held (config_tab.py:285-287).

- **W167.** [3B item 13, partly] The cache key compares floats by repr; an untouched bounds grid reproduces the
  file's numbers exactly.
  Area: the Config and Prior tabs; second reader: holds, with a correction.
  On screen: Nothing visible: a grid left as loaded behaves exactly like the file it came from.
  Correction: The trace is right, and the part left open is now settled: building the config does nothing to the
  bounds (its only start-up checks are the three multi-frequency ones), the hand-entered dictionaries are copied
  verbatim, and the key reads the lows and highs straight off the config. So 'Edit values' with the numbers left
  as loaded resolves to the same cache as 'Use file', and a typed 0.10 equals a file's 0.1. Still not settled:
  what a comma-decimal Windows number format does to typing. No locale is set anywhere in the window's code, the
  validators are default-constructed and parsing is Python's own, so it could not be told without running.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:94-106.
  C:/Users/J/PycharmProjects/PRISM/core/Helpers/file_manager.py:237.
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/param_grid.py:72, :110-113, :130-137.
  C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/labeled_inputs.py:21-22, :39-42.
  C:/Users/J/PycharmProjects/PRISM/core/gui/session.py:27-37.
  C:/Users/J/PycharmProjects/PRISM/core/cli.py:304-307, :325-349.
  C:/Users/J/PycharmProjects/PRISM/core/sim_config.py:107-135.
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/identity.py:92-96. A search of
  C:/Users/J/PycharmProjects/PRISM/core/gui for QLocale finds nothing.

- **W168.** [3C bullet 1, partly] window.md says '0, the default, is right on an idle card'; the same advice is
  in the window's help badge and its 'Off' note.
  Area: the Config and Prior tabs; second reader: holds.
  On screen: The reader page, the help bubble and the note in the window all say that no ceiling is right on an
  idle card.
  Evidence: C:/Users/J/PycharmProjects/PRISM/docs/guide/window.md:219.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/help_text.py:47-48.
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/config_tab.py:159-161. That it is false at full
  size on this card is a fact of the run record, not of the code:
  C:/Users/J/PycharmProjects/PRISM/docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md:35,
  :67-71.

- **W169.** [note 2A bullet 3, marked partly] The passive line names the first differing key alphabetically, so
  the tier-1 box swapped for master.txt reads 'differs in param_keys' and the prior is 'never' the key named;
  the first reader corrected 'never' as too strong.
  Area: the Posterior, Validate, Infer and TSNPE tabs; second reader: holds.
  On screen: After the wrong bounds file is picked, the only signal is a line under the budget boxes saying the
  earlier run 'differs in param_keys'; it shows no values and does not mention the prior even when that changed
  too, and the run then simulates from zero.
  Evidence: Paths under C:/Users/J/PycharmProjects/PRISM/. core/SBI/training_checkpoint.py:285-286 the differing
  keys are collected in sorted order; :290-300 the line prints the first one per sibling (:295), every committed
  sibling listed. core/artifacts/identity.py:87-122 the key names: prior_fingerprint sorts after param_keys,
  n_runs, nd_highs, nd_lows and others, and before run_size, rescale_*, so the first reader's correction is
  right (a different prior plus a different rows-per-batch reads 'differs in prior_fingerprint').
  Resources/Bounds/nadrowski/master.txt:2-16 and master_tier1.txt:2-16: identical non-dimensional limits, last
  rescale entry f_scale against T. core/config.py:420 the log list is empty. Addition: for this swap three keys
  differ even with the same prior loaded (param_keys, rescale_highs, rescale_lows), and the same prior does load
  under both boxes because the prior check compares only the non-dimensional part
  (core/artifacts/store.py:1275-1291), so the one-setting dialog never appears for this swap
  (training_checkpoint.py:350) and the passive line is the only signal.

- **W170.** [2B bullet 2] The fit's progress is one caption under an indeterminate bar: sbi's epoch counter, no
  clock time, reaching no file, no loss until the fit ends.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: During training the only progress text is 'Epochs trained: N', which first appears after the first
  epoch and changes once per epoch; there is no time per epoch and no loss until the end.
  Correction: As the first reader corrected: for the whole first epoch the caption is empty, because the counter
  is first printed after an epoch. From then on it changes once per epoch. The count reaches the log pane only
  when the fit ends. All else in the claim holds.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/widgets/progress_pane.py:366-369, :385-389;
  core/SBI/train.py:254;
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/sbi/inference/trainers/base.py:997-998, :1269-1274;
  core/gui/vt.py:289-300 (a bare carriage return re-publishes the row, no log line), :228-241 (logged only when
  a newline ends it); core/runs.py:81-87 (the record's log holds program records only, not prints);
  core/SBI/train.py:241-242 (sbi's per-epoch curves are discarded); sbi base.py:1012-1013 (summary only after
  the loop).

- **W171.** [2B bullet 4, first part] The whole window is locked for the run: every panel's controls are
  disabled and only Cancel stays live.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: During a run the forms on every tab are greyed, but one can still move around, open and close
  figures, read the log and the Artifacts screen, and change the appearance.
  Correction: Overstated, as the first reader said. What is locked is each tab's left-hand form column. Still
  live: moving between screens and tabs, figure tabs with their close buttons and Pop out, the log, the
  splitters, the header slot, the appearance menu and settings toggles, and reading on the Artifacts screen. The
  writes outside the tabs refuse on their own: Artifacts note/delete/sweep, model validate/save, model delete.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/base_panel.py:312-317, :323-333 (the lock is
  controls.setEnabled); core/gui/main_window.py:399-403 (navigation deliberately free), :216-231 (appearance
  change has no run check), :283-287 (model delete refuses); core/gui/widgets/figure_stack.py:52-54, :87-98;
  core/gui/screens/artifact_screen.py:580-590, :617, :682, :829;
  core/gui/screens/model_builder_screen.py:394-398, :510-512.

- **W172.** [2C bullet 2, second sentence] Cancel acts on one click with no confirmation, is the only live
  control on the tab, cannot be withdrawn, and one pressed in the final epoch discards a fit that has just
  converged.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: One click on Cancel during training throws away the whole fit at the end of the current epoch, with
  no 'are you sure' and no way to take it back.
  Correction: All holds except 'the only live control' (figure tabs, Pop out and the log are live; it is the
  only RUN control). Sequence that loses a multi-day fit: Posterior tab, during the fit -> one left click on
  Cancel, or close the window and press Enter (the dialog's default button sets the same flag) -> nothing can
  take it back -> at the end of the current epoch the fit is thrown away, even if that epoch was the one at
  which training would have stopped by itself. It is also true just AFTER the last epoch: sbi's closing message
  and the round bar's closing write both come before the record is written, so a Cancel clicked while a finished
  fit is being wrapped up still discards it.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/panels/base_panel.py:150-152, :291-300; the flag is
  only ever set (base_panel.py:118, :294) and read (core/gui/worker.py:157, :201; core/gui/streams.py:75), never
  cleared (grep). C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/sbi/inference/trainers/base.py:978-998
  (the print at :998 precedes the next convergence test at :978), :1166-1174 (closing message).
  core/SBI/train.py:254 with :366-373 (the bar closes after the fit returns), core/orchestrator.py:1420 (the
  record is written later). core/gui/main_window.py:432-439 (default button, same flag).

- **W173.** [2C bullet 4; 3B item 8] 'Quit anyway' probably loses the record at the end of a fit.
  Area: how a run is started, shown, cancelled and closed; second reader: could not be settled by reading.
  On screen: Not established; if the chain holds, choosing 'Quit anyway' during training keeps the graphics card
  busy with no window until the fit ends and then leaves no trained result and no message.
  Correction: Not settled by reading. The repository half is certain: IF the process lives on after the window
  and the worker's signal object is dead, the fit runs to its end, the loss-figure hand-off raises inside the
  posterior's writer, the writer deletes the record's folder, and the error report is itself swallowed. Both IFs
  are inside compiled PySide6/Qt. Three outcomes are possible and two lose the fit: (a) the process ends with
  the window: fit lost at once, contrary to the dialog's promise; (b) process lives, signal dead: fit lost at
  its end, silently, after holding the card unseen; (c) process lives, signal still usable: record written.
  Sequence: Posterior tab -> Train -> during generation or the fit click the window's X -> 'Quit anyway', with
  no popped-out figure window open. The program's own comments and dialog text assert (b)'s two preconditions.
  The first reader's one-minute script settles it.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/main_window.py:416-445 (no wait; docstring :419-421
  and dialog text :429-431 say the task keeps running); core/gui/__main__.py:16-23 (nothing waits; grep over
  core for waitForDone, aboutToQuit, atexit: none); core/gui/worker.py:207-210 and core/gui/streams.py:152-155
  (comments quoting 'Signal source has been deleted'); core/gui/panels/base_panel.py:62 (unguarded emit);
  core/artifacts/store.py:404-409 (forward called after the save), :434-481 and :485-495 (folder removed on any
  exception); core/orchestrator.py:1420-1431 (posterior.pt and loss.npz written, then the sink); only that one
  sink call exists in the training stage (grep fig_sink in orchestrator.py).
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/PySide6/QtCore.pyi:10808 shows __moduleShutdown
  exists; what it does is not readable. docs/guide/window.md:78-79 repeats the promise.

- **W174.** [2C bullet 5, rest; 3B item 8] 'Cancel task & quit' requests the same cancel and closes the window
  at once without waiting; whether the rescue save completes is unsettled; in a fit the process keeps fitting
  unseen until the epoch ends.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: Choosing 'Cancel task & quit' makes the window disappear at once while the program may go on
  working unseen until the run reaches its next stopping point.
  Correction: The code half holds. The unsettled half stays unsettled (same unread library question as 'Quit
  anyway'), but its worst case is bounded by reading: if the process ends before the rescue save, the cache
  stays at its last commit and at most the batches since then are simulated again. In a fit the cancel discards
  the fit either way; the only extra is up to an epoch of unseen work on the card. The one recorded real-screen
  check says only 'stops cleanly'.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/main_window.py:438-445 (flag set, state saved,
  close accepted, no wait); core/SBI/pipeline.py:1951-1990 (rescue save, no Qt object touched);
  core/SBI/training_checkpoint.py:384-395 (shards written, then the state file); core/SBI/pipeline.py:1605
  (uncommitted shards set aside on resume); docs/checklists/display-walkthrough.md:29 (row 20, pass on
  2026-09-11).

- **W175.** [2E bullet 4] The window already gives live progress (elapsed clock, overall bar, solver rate), the
  loss figure at the end, and the finished posterior held by the session.
  Area: how a run is started, shown, cancelled and closed; second reader: holds, with a correction.
  On screen: During training only the header clock moves; the bar shows no percentage and the solver reads idle.
  Correction: All five exist, but in the fit only the header clock moves: the bar has no percentage, the solver
  reads idle, and the clock is elapsed time only and disappears when the run ends. The loss figure and the
  session's posterior arrive only on a successful end.
  Evidence: [root C:/Users/J/PycharmProjects/PRISM] core/gui/main_window.py:351-379 with
  core/gui/screens/nav_shell.py:25-38 (clock), :364-367 (cleared at the end);
  core/gui/widgets/progress_pane.py:345-374 (bar; :368-369 indeterminate), :296-326 (rate; :313-314 idle after 8
  s); core/orchestrator.py:1429-1431 through core/gui/panels/base_panel.py:44-62 to
  core/gui/widgets/figure_stack.py:59-65 (loss figure); core/gui/panels/inference/posterior_tab.py:287-298
  (session posterior, on result only).

- **W176.** Note claim 'while a run is live every panel's controls are disabled and only Cancel stays live' is
  only partly right: just each panel's left column is disabled.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: During a run the form on the left of every tab is grey, but you can still move between screens,
  close or pop out picture tabs, read the Artifacts screen and change the appearance.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:316-317 and 333 (only the left-hand
  controls widget is disabled), 144-182 (figure tabs, log pane and both dividers are outside it).
  C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:401-403 (navigation deliberately left free), 283-287
  (model delete refused). C:/Users/J/PycharmProjects/PRISM/core/gui/screens/model_builder_screen.py:394-398,
  510-512 (Validate and Save refuse; the form itself stays editable).
  C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:580-588, 617, 682, 829 (the three writes
  refuse; reads do not, 439). The appearance switches on the Settings screen stay live too
  (C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:216-231).

- **W177.** Note claim 'Cancel is the only live control on the tab for the whole run' is only partly right: the
  right-hand column stays enabled.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: While a run is going, the picture tabs can still be closed, reordered and popped out and the
  dividers dragged; only the form on the left is locked.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:315 (the comment speaks of the
  controls column), 333 (the only widget disabled), 150-157 (Cancel sits on the progress row), 166-182 (figure
  stack, log pane, dividers). C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/figure_stack.py:52-54, 87-98
  (tab close, reorder and 'Pop out' are never gated).

- **W178.** Note claim, left not established: 'Quit anyway' says the task keeps running, but after the window
  closes signal emits raise and a figure is forwarded by an unguarded emit, so the record is probably lost.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: could not be
  settled by reading.
  On screen: Choosing 'Quit anyway' during a training is a gamble the dialog does not describe: the run may die
  with the window, or finish unseen and then have its result thrown away, or finish properly, and there would be
  no message in any of the three cases.
  Correction: Still not settled, but narrowed to one link I cannot read (what the compiled Qt binding and the
  interpreter do to the worker thread at exit; nothing was run). Sequence: Parameter Inference, Posterior tab, a
  training running; close the window; choose 'Quit anyway'. The code then allows three outcomes: (a) the process
  ends and takes the run with it: the fit is lost, the cache is kept; (b) the run goes on with the window's
  objects gone: forwarding the training-loss picture then raises inside the block that writes the posterior, and
  the store removes the posterior it had just written, with no window left to report it; (c) the run goes on and
  the forward does not raise: the posterior lands. The dialog promises (c). Two comments in the code say the
  forwarding does raise once the window is gone, and a third treats a closed window as the death of the process.
  For FDT and sweep runs no picture is forwarded, so only (a) against (c) is open for them.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:419-421 and 428-431 (the promise), 440-445
  ('Quit anyway' does nothing to the run). C:/Users/J/PycharmProjects/PRISM/core/gui/__main__.py:16-23 (the
  event loop returns and the process exits; grep finds no wait on the thread pool anywhere under core).
  C:/Users/J/PycharmProjects/PRISM/core/gui/worker.py:207-210 and
  C:/Users/J/PycharmProjects/PRISM/core/gui/streams.py:148-155 (comments: 'the window was closed while this run
  was still going' and the signal objects are gone).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:62 (emit with no guard).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:401-412 (forward called inside the sink), 434-481
  (ordinary kinds: folder removed on any exception).
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1420-1431 (the loss figure is forwarded inside the
  posterior's write block, after the posterior file is saved).
  C:/Users/J/PycharmProjects/PRISM/core/FDT/fdt_pipeline.py:456-460 ('a closed window, a kill -- runs no
  __exit__'). No figure sink on FDT or sweep runs: fdt_panel.py:319-324, crossval_panel.py:330-333.

- **W179.** Note claim 'no lasting record of a run's lines' is only partly right: true for a training, not for
  FDT and sweep runs, whose log is rewritten on disk as they go.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: A long FDT or sweep run leaves a time-stamped log on disk even if it is stopped; a training that
  fails or is cancelled leaves none, only what is still in the text pane until the window closes.
  Evidence: Pane: C:/Users/J/PycharmProjects/PRISM/core/gui/widgets/log_pane.py:8, 15, 21-30 (5,000 lines, no
  time stamps, no save). Training: C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:340-344 and 627-629
  (log written at the commit only) with C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1420 (the
  posterior's writer is opened only after the fit), so a cancelled or crashed training leaves no log at all. FDT
  and sweep: store.py:501-550 and 552-574 (each refresh rewrites the log, time-stamped; a stop appends one
  'error stopped' line, 460-471 and 567-570), called at
  C:/Users/J/PycharmProjects/PRISM/core/FDT/fdt_pipeline.py:387, 461, 479, 495 and
  C:/Users/J/PycharmProjects/PRISM/core/FDT/cross_validation.py:164-174, 412, 449, 540, 546, 591. That file is
  only as fresh as the last refresh and holds the program's own records and warnings, not everything printed.

- **W180.** Note claim 'launching the window touches the card: building the Posterior tab asks the planner for
  free memory' was only half followed by the first reader.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Simply opening the window starts using the graphics card, before any button is pressed, which is
  why a second window must not be opened beside a long run.
  Evidence: The whole chain is there, on the GUI thread, during window construction:
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:111-112,
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/base.py:77-82 (device detection) and 138-139,
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:513 and 529-532,
  C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:635,
  C:/Users/J/PycharmProjects/PRISM/core/config.py:64-70 and 122-136 (torch.cuda.mem_get_info, memory_reserved,
  memory_allocated). Installed library:
  C:/Users/J/anaconda3/envs/biophys-env/Lib/site-packages/torch/cuda/memory.py:822-838 and
  torch/cuda/__init__.py:602-618 (both initialise CUDA in the process first). The TSNPE tab does the same
  (inference/tsnpe_tab.py:105). How much card memory that takes was not measured. The four secondary panels'
  constructors make no such call (fdt_panel.py:108-116, crossval_panel.py:98-107, simulate_panel.py:56-66,
  reduction_panel.py:42-46).

- **W181.** Note claim 'the window and the command-line tool call the same stage functions' is only partly
  right: reduction is window-only and the Simulate stream is the panel's own runner.
  Area: Simulate, FDT, the sweep study, the reduction map, the model builder; second reader: holds.
  On screen: Nothing visible; the reduction map and the live simulation exist only in the window, so there is no
  command-line way to repeat them.
  Evidence: Shared: C:/Users/J/PycharmProjects/PRISM/core/tool/fdt.py:239, 268, 349 against
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/fdt_panel.py:18-19, 75, 234 and crossval_panel.py:18-19, 218,
  330. Window-only: C:/Users/J/PycharmProjects/PRISM/core/cli.py:428-431 and
  C:/Users/J/PycharmProjects/PRISM/core/Reduction/sweep.py:162-167 ('no command-line subcommand'); grep for
  run_reduction_map and run_simulation_stream under core/tool finds nothing;
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/simulate_runner.py:187.

- **W182.** Note claim (2B, second half): nothing guards against a run in another process; the store has no
  cross-process lock.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: A second window or a terminal can delete or annotate a record that a run in the first window is
  still writing, with no warning in either.
  Correction: There is no lock, and Delete and Set note check nothing about another process. The sweeps do carry
  one guard, an age test of 300 seconds on the newest file in the folder, which stops protecting a folder that
  has sat idle for five minutes (see the sweep finding). Of the delete questions only the one for an unfinished
  FDT record warns that a run elsewhere may still be writing it. Sequence that loses work: a training generating
  in one window (or at a terminal); in a second window: Artifacts; Simulation caches; select the cache;
  Delete...; Yes. The first run is not stopped or told (see the unfinished-cache finding).
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/gui/screens/artifact_screen.py:122-126 (the statement itself),
  :128-136 (the two delete sentences), :580-590 (process-local flag).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:52-58 (a lock is declared out of scope; the 300 s
  constant), :874-889 (set note: find, rewrite), :918-947 (delete: dependents only), :1024-1034, :1117-1125,
  :1213-1218 (the three age tests, sweeps only).

- **W183.** Note claim (2D, rotation row): the pane abbreviates any list of more than 12 numbers to its first
  three, so a 13-parameter posterior's eigenvalues show as three of thirteen.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: On a posterior's record the thirteen eigenvalues read as three numbers and '(13 entries)'; on the
  rotation check's record the full set appears further down, one block per direction.
  Correction: The first reader's two corrections stand. The cut applies to any list of more than 12 plain
  values, names included, not only numbers. A posterior record's eigenvalue line is cut as stated. The rotation
  check's own record is different: its one-line eigenvalue list is cut, but the same record lists every
  direction as a block of its own with its eigenvalue, so all thirteen can be read in that pane.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:30, :36-37, :57-60 (the cut), :82-86 (a
  list of records prints one block per entry). C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1228,
  :1252, :1466 (a posterior stores the eigenvalues as one flat list).
  C:/Users/J/PycharmProjects/PRISM/core/diagnostics/identifiability.py:145 and :192-201 (the check stores both a
  flat list and one record per direction with its eigenvalue).

- **W184.** Note claim (3B item 12): the Artifacts pane's list abbreviation hides ten of thirteen eigenvalues.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: The eigenvalues are cut to three on a posterior's record but can be read in full, direction by
  direction, on a rotation check's record.
  Correction: True of a posterior record's eigenvalue line and of the rotation check's one-line list; not true
  of the rotation check's record as a whole, which also prints each direction's eigenvalue in a block of its
  own. The same cut hides ten of thirteen parameter names (first finding above).
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/artifacts/report.py:57-60 and :82-86.
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1466.
  C:/Users/J/PycharmProjects/PRISM/core/diagnostics/identifiability.py:192-201.

- **W185.** Note claim (2C, Quit anyway): the loss figure is forwarded from inside the posterior writer, and the
  writer removes its directory on any exception.
  Area: the Artifacts screen; second reader: holds, with a correction.
  On screen: If the window is closed with 'Quit anyway' during a fit, the fit may run to its end and then
  discard the finished posterior without any message; whether it does cannot be settled without running it.
  Correction: Every code link holds, and two more can be added: the window's figure hook sends the picture to
  the window with no guard around the send, and if the run then fails the worker's attempt to report it is
  itself swallowed, so nothing would be shown or logged. The one link that cannot be told by reading is whether
  that send really raises after the window has been closed with 'Quit anyway' (and whether the process keeps
  running at all); that is Qt's behaviour at shutdown. The code's own comments at the two other send sites say
  it does raise, and both of those sites catch it; this one does not. If it raises: the folder holding the
  just-saved posterior is deleted and the whole fit is lost, the simulated rows surviving in the cache.
  Sequence: Posterior tab; train a new posterior; while it is fitting, close the window; choose 'Quit anyway';
  the fit runs on unseen to its end.
  Evidence: C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1420-1431 (posterior saved, then the loss
  figure sent through the writer's sink, inside the writer's block).
  C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:404-411 (saves the picture, then calls the forwarded
  hook unguarded), :434-436 and :479-481 (any exception removes the folder of an ordinary kind).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/base_panel.py:44-62 (the hook ends in an unguarded signal
  emit), :246-247 (installed for the run).
  C:/Users/J/PycharmProjects/PRISM/core/gui/panels/inference/posterior_tab.py:148-150 (training is dispatched
  with that hook). C:/Users/J/PycharmProjects/PRISM/core/gui/worker.py:146-150 and :204-210 (a failure is
  reported by another emit, and a RuntimeError there returns silently; the comment says the signals are gone
  once the window is closed). C:/Users/J/PycharmProjects/PRISM/core/gui/streams.py:148-155 (the log pump catches
  the same error). C:/Users/J/PycharmProjects/PRISM/core/gui/main_window.py:423-445 (the dialog and its 'keeps
  running in the background' text).

- **W186.** Note claim (3B item 10): reusing a complete cache rewrites its manifest, and wall_seconds is
  recomputed as now minus created.
  Area: the Artifacts screen; second reader: holds.
  On screen: After the next training that reuses the finished cache, the cache's pane will show a generation
  time counted from the day the cache was first created to that day, in place of the roughly 27 hours recorded
  now.
  Evidence: The step the first reader left untraced, followed end to end.
  C:/Users/J/PycharmProjects/PRISM/core/orchestrator.py:1083-1096 (a complete cache counts as a resume and the
  training goes on to generation). C:/Users/J/PycharmProjects/PRISM/core/SBI/train.py:256-279 (generation is
  always called, with the checkpoint). C:/Users/J/PycharmProjects/PRISM/core/SBI/pipeline.py:1503-1527 (the
  stored count becomes the start and the rows are loaded into the buffer), :1630 (the batch loop is then empty),
  :2023-2038 (buffer present, no final save needed, and completion is marked unconditionally).
  C:/Users/J/PycharmProjects/PRISM/core/SBI/training_checkpoint.py:398-411 and :465-472 (completion rewrites the
  state and the manifest). C:/Users/J/PycharmProjects/PRISM/core/artifacts/store.py:1575-1577 (with complete
  set, the wall time is now minus the original creation time; id, created, note, parents are kept, :1582-1588).

- **W187.** Offering a name, a note, an epoch ceiling, the checkpoint cadence and the resume policy is a change
  to the window only (first reader: partly).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: Nothing visible today; these five could be added to the training screen without changing what a run
  computes, except that one refusal would not point at its control.
  Correction: True for all five: the training stage already takes each as an argument and the tab simply sends
  none. The first reader's caveats all check out. (1) The cadence must also be handed to the two read-only
  helpers behind the budget lines and the new-run question; both already accept it. (2) The one-line rule for a
  note is run by each front end, not by the stage or the store. (3) The 'must resume but there is nothing to
  resume' refusal carries no field key, so pointing it at a new control needs a one-word engine change. (4) The
  device is not one of the five and is not a stage argument: it is fixed when the config is built. (5) One test
  pins the set of settings that have no control and must change with them.
  Evidence: All repo paths are under C:/Users/J/PycharmProjects/PRISM. core/orchestrator.py:792-799 (name, note,
  max_num_epochs, checkpoint_every, resume are arguments); core/gui/panels/inference/posterior_tab.py:148-150
  with :171-181 (the tab sends none); core/orchestrator.py:422-423 and :481-482 (fresh_run_near_misses and
  training_preview take checkpoint_every) against posterior_tab.py:207-208 and
  core/gui/panels/inference/base.py:138-139 (the tab passes none); core/artifacts/store.py:846-850 and
  core/artifacts/manifest.py:132-171 (no note rule in the store), core/tool/__init__.py:156 (the tool runs the
  rule itself); core/orchestrator.py:1111-1112 (no field) against core/SBI/pipeline.py:1510-1512
  (field='resume'); core/cli.py:279, :348 and core/gui/session.py:32-37 (device chosen at config build; the
  window passes none); tests/test_nav_and_gating.py:1621.

- **W188.** The pre-flight checks (smoke, identifiability jacobian, the probe checks) have nothing in the
  window; the first reader adds that the epoch timing is missing too (partly).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: None of the checks that come before a long run can be started from the window; each still has to be
  typed at the command line.
  Correction: Confirmed: no part of the window reaches the diagnostics or the smoke command, and the epoch
  timing cannot be done either (no epoch ceiling, no cadence control). One point is too strong: 'no store root'.
  The window resolves its store at launch from the same environment setting the command line uses, so it can be
  started against a scratch store; what is missing is a control, not the ability.
  Evidence: Grep of core/gui for core.diagnostics, core.tool, smoke, sbc_repeats, channel_ablation,
  identifiability: the only hits are three unrelated mentions of a 'smoke integration' in
  core/gui/screens/model_builder_screen.py:10, :236, :388. core/gui/fields.py:178-186 (every diagnostic and
  probe key maps to None), :60, :86 (cadence and epoch ceiling None). docs/guide/retrain.md:154-163 (the timing
  uses --max-epochs 9 --checkpoint-every 0 in a scratch store). core/gui/app.py:80-83 with
  core/config.py:187-191 (the window's store is PRISM_ARTIFACTS when set).

- **W189.** The four stratified sbc runs have nothing in the window; a fixed probe count is not an argument of
  the calibration stage (partly).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: The calibration test at a fixed number of probes cannot be run from the window at all.
  Correction: The calibration stage indeed has no such argument and always pools. But the runs the runbook means
  go through a different entry, the repeated calibration test, which does take a fixed probe count and writes
  its own record, so a screen for it is a window-only change. It gives only the rank half of the verdict, and
  the runbook calls these runs a characterisation, not a gate.
  Evidence: core/orchestrator.py:1873-1876 (no probe-count argument), :1965-1971 (chi_k_fixed=None passed on
  purpose); core/diagnostics/sbc.py:87-90 (sbc_repeats takes chi_k_fixed), :125-138 (its refusals), :163 (writes
  a diagnostic record), :207-215 (rank half only); docs/guide/retrain.md:88 ('characterise the posterior and
  gate nothing').

- **W190.** Calibrations, inferences and observations made in the window stay unnamed for good; only priors and
  posteriors can be renamed; the TSNPE picker restores its remembered entry rather than the observation just
  written (partly).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: Calibrations, inferences and observations appear in every list only by id and time, and there is
  nowhere in the window to give them a name.
  Correction: All three facts hold for the window as it is, and the command line has no rename either. 'For
  good' is too strong about the engine: the store's rename works for any kind, so a rename control for these
  three is a window-only change. They are unnamed in the first place because the window sends no name to the
  calibration and inference stages and the inference compositions pass none on to the observation.
  Evidence: Grep for '.rename(' under core/gui: only core/gui/panels/inference/prior_tab.py:328 and
  posterior_tab.py:307; under core/tool: none. core/artifacts/store.py:852-872 (rename takes any kind).
  core/gui/panels/inference/validate_tab.py:75-77 and infer_tab.py:470-488 (no name sent);
  core/orchestrator.py:2528, :2558 (no name passed to the observation stage).
  core/gui/widgets/artifact_picker.py:181-183, :219 (a refresh re-selects what was selected);
  core/gui/panels/inference/tsnpe_tab.py:134, :148, :205-207, :226.

- **W191.** The cache's key holds 34 science settings and nothing about the front end or about the code that
  made the rows (first reader: partly).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: A cache made by an older version of the code is resumed as if it were current whenever the settings
  match; nothing warns that rows from two code versions are being mixed.
  Correction: There are exactly 34 keys and none comes from a front end. Not all are science settings: five are
  hand-maintained code markers (the format label, the list of validity flags, a feature-set version, the chi
  layout and its width) and two are the device and number type. So 'nothing about the code' is slightly too
  strong, but the substance stands: no code revision or source hash is in the key, and nothing ever compares the
  revision recorded in a cache's record with the running code. A code change that does not bump one of those
  markers is not noticed on a resume.
  Evidence: core/artifacts/identity.py:87-122 (34 keys; :88, :109, :110, :112, :114 are code constants, :118-119
  hardware); core/artifacts/store.py:1584 (the cache's record keeps the first process's revision); grep of core/
  for git_rev: produced in core/artifacts/provenance.py:22-30 and compared nowhere;
  core/SBI/run_guards.py:130-132 (the probe cycle floor is a module constant outside the key).

- **W192.** 'Quit anyway' probably loses the record at the end: the loss figure is forwarded by an unguarded
  emit from inside the posterior writer, the writer removes its directory on any exception, and the worker
  swallows the error (first reader: not established).
  Area: the engine side (stages, the store, the fit, the cache); second reader: could not be settled by reading.
  On screen: Cannot tell: after closing the window with 'Quit anyway' during a training, the run may finish and
  leave its record as the dialog promises, or the trained network may be lost at the very end with no message
  anywhere.
  Correction: Not settled by reading, and it cannot be without running. Every step inside PRISM is as described.
  What cannot be read here is what the windowing library does when the program's main function returns while a
  run is still on a pool thread. Three outcomes are possible and only the third keeps the result: (a) the
  process ends with the window, and the fit, which lives only in memory, is gone; (b) the process lives on and
  forwarding the loss picture raises because the window's signal objects are gone, the writer then deletes the
  folder that already holds the trained network, and the error is swallowed; (c) the process lives on, the
  forward succeeds and the record is written. PRISM's own comments at two guarded emit sites state the premise
  of (b) as what happens after the window closes; the forward of the picture is the one emit on that path left
  unguarded. The same removal follows any error while the picture is drawn or saved, since those steps also sit
  inside the writer after the network file is written. Sequence, if (a) or (b) is what happens: start a
  training, close the window during simulation or the fit, choose 'Quit anyway'. The simulated rows are safe
  either way, because they are committed batch by batch; what is at risk is the fit. Calibrations and inferences
  forward their pictures the same way.
  Evidence: core/gui/main_window.py:423-445 ('Quit anyway' just lets the window close);
  core/gui/__main__.py:16-23; grep of the repository for waitForDone, aboutToQuit, setQuitOnLastWindowClosed,
  os._exit: no hit in core/. core/orchestrator.py:1420-1431 (the picture is drawn and forwarded inside the
  writer, after the network file is saved at :1421); core/Helpers/visualizers.py:433-463 (the drawing has no
  guard); core/artifacts/store.py:401-412 (save the picture, then forward, unguarded), :434-436 and :479-481
  (any exception removes the folder); core/gui/panels/base_panel.py:44-62 (bare emit at :62);
  core/gui/worker.py:146-150, :205-210 (the error emit's own failure is swallowed); core/gui/streams.py:148-155
  and worker.py:207-210 (comments: 'Signal source has been deleted' after the window is closed);
  streams.py:191-217 (ordinary output never emits from the worker thread, so nothing stops the run earlier).

- **W193.** During generation the tab's Cancel rescue-saves the batches since the last commit, where Ctrl-C
  under the console's capture pipe loses them; 'Cancel task & quit' closes at once without waiting (first
  reader: partly).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: Pressing Cancel during simulation keeps every finished batch; whether the same holds when the
  window is closed with 'Cancel task & quit' is not settled.
  Correction: The rescue save is real: any stop that unwinds through the generation loop, the window's Cancel
  included, commits every finished batch not yet on disk before re-raising, provided at least one such batch
  exists; the batch in progress is not kept. The code makes no difference between the window's Cancel and
  Ctrl-C: both land in the same handler, so the loss reported under the console's capture pipe is an observation
  of that run, not something this code shows. 'Cancel task & quit' does request the cancel and then closes the
  window without waiting; whether the rescue save then completes rests on the same unsettled question as 'Quit
  anyway'. At worst the batches since the last commit are simulated again, at most 49 at the default interval.
  Evidence: core/SBI/pipeline.py:1951-1990 (catches every exception type, saves batches from the last commit up
  to the current one when the cache is on and at least one is uncommitted at :1972, re-raises at :1990; the
  comment at :1952-1954 names both the cancel and KeyboardInterrupt); core/gui/streams.py:204-205, :252-253
  (where the window's cancel is raised); core/gui/main_window.py:438-445 (cancel requested, state saved, window
  closed, no wait); core/config.py:303 (interval 50).

- **W194.** A narrowing round trains at 128 x 8 whatever the Posterior tab says (first reader: partly).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: A narrowing round started from the window always trains the default-size network, whatever size was
  typed on the training tab, and nothing on that tab says so.
  Correction: Holds for the window; the limit is the tab, not the engine. The narrowing tab sends only the
  direction count, the level, the two budget numbers and the new-run consent, so the round's network size,
  learning rate and patience fall back to the defaults, whatever the training tab shows and whatever size the
  parent posterior was trained at. The engine and the command line accept all of them. Sequence that costs days:
  train or load a larger posterior (the runbook's is 256 x 10), run an inference, open the narrowing tab and
  press Run. The round simulates its budget and fits a 128 x 8 network; nothing on the tab or in the run's lines
  states the size, and it is found in the finished record. The round's simulated rows are kept, so what must be
  repeated is the fit, and only the command line can ask for another size on those rows.
  Evidence: core/gui/panels/inference/tsnpe_tab.py:157-161 (what the tab sends); core/orchestrator.py:2566-2574
  (the round takes hidden_features, num_transforms, learning_rate, stop_after_epochs, max_num_epochs,
  checkpoint_every, resume, name, note), :2648-2655 (forwards only those set), :936-939 (unset ones resolve to
  the defaults); core/config.py:363-364 (128 and 8); core/orchestrator.py:1450-1451 (the record keeps the sizes
  used); docs/guide/retrain.md:154-155.

- **W195.** The cache's key compares floats by repr; whether a round trip through the Bounds grid reproduces the
  file's numbers was not traced (first reader: partly).
  Area: the engine side (stages, the store, the fit, the cache); second reader: holds, with a correction.
  On screen: Nothing visible: bounds passed through the grid unchanged resume the same cache as the file they
  were loaded from.
  Correction: The untraced part is now traced, and it is safe. A bounds file's numbers are parsed to decimals,
  shown in the grid as the shortest text that reproduces each one, and read back by the same conversion, which
  returns the identical number and always a decimal, never a whole number. Unedited values that pass through the
  grid therefore give the same cache key as the file. The whole-number-versus-decimal difference the first
  reader describes is real in the key itself but can only be reached by code that hands the config builder
  hand-made whole numbers directly; neither front end does.
  Evidence: core/SBI/training_checkpoint.py:103-106 (whole numbers kept as they are, decimals by repr);
  core/artifacts/identity.py:93-96 (limits taken as given); core/Helpers/file_manager.py:237 (file values
  through float()); core/gui/widgets/param_grid.py:71-73, :110-111 (grid rows seeded with those values),
  :130-137 (read back through _num); core/gui/widgets/labeled_inputs.py:21 (shown as str of the float), :39-40
  (read back with float()); core/cli.py:306-307 (hand-entered dicts used as given).

## D. Dropped, or not settled by reading

- **W196.** The last-resort error hook builds a modal dialog with no check of its thread; an exception escaping
  the worker's run() would reach it off the GUI thread.
  Area: the shell, start-up, Settings and the look; second reader: could not be settled by reading.
  Why: Under C:/Users/J/PycharmProjects/PRISM/: certain: core/gui/app.py:22-34 builds and shows a message box
  with no thread check. Certain: what can leave the worker's run() is narrow. Inside the stage call every
  ordinary exception and every cancel is caught (core/gui/worker.py:109-162); outside it only RuntimeError is
  caught (:207-210). So only a non-RuntimeError raised in the bookkeeping outside the stage call, or a
  SystemExit/KeyboardInterrupt-class exception raised by the stage, can escape; my grep of core for sys.exit(,
  exit(), os._exit, raise SystemExit found none on a stage path (only entry points: core/__main__.py:8,
  core/gui/__main__.py:23, core/gui/app.py:120, an icon build script). Cannot tell: whether PySide hands such an
  exception to the hook on the pool thread; that is PySide's compiled code, and no source for it is installed
  (no .cpp or pyside*.h under site-packages/PySide6). No test touches the hook (grep of tests/ for 'Unhandled
  error', '_install_excepthook', 'sys.excepthook': none).

- **W197.** Closing the main window while a pop-out figure window is open probably does not end the program; the
  main window is hidden for good and a live run carries on.
  Area: how a run is started, shown, cancelled and closed; second reader: could not be settled by reading.
  Why: [root C:/Users/J/PycharmProjects/PRISM] Repository half confirmed: pop-outs are built with no parent
  (core/gui/widgets/figure_stack.py:119, :128; figure_window.py:57-58, :116-117) and held in a set
  (figure_stack.py:57, :129); MainWindow.closeEvent neither closes them nor calls quit
  (core/gui/main_window.py:416-445); grep over core for setQuitOnLastWindowClosed, WA_QuitOnClose,
  closeAllWindows: no match; __main__.py:19 just returns app.exec(). The last link, that Qt keeps the event loop
  running while any such top-level window is visible, is Qt's own rule inside the compiled library; I could not
  read it here. If it holds, note the side effect: the run's signals stay connected to the hidden window, so the
  run would finish and write its record normally, and the program ends only when the last pop-out is closed. Ten
  seconds by hand settles it: pop out a figure, close the main window, see whether the figure and the console
  stay.

- **W198.** A cancel that fires after the posterior is committed but before the stage returns would say 'Run
  cancelled.' while the record is on disk and the session never receives it.
  Area: how a run is started, shown, cancelled and closed; second reader: WRONG.
  Why: Followed to the end, there is no place for a cancel to land there. After the writer commits, the stage
  only reads the record back and returns; that read-back and everything it calls contain no print and no program
  log record, and the tab runs the stage directly. A cancel asked for in that window stays unfired, the worker
  writes 'The run finished before the cancel could take effect; nothing was stopped, and its result is
  complete.' and the result is delivered. Only a third-party library writing to the console in that instant
  could fire it; I found none (torch.load itself not read). The writer's own block is likewise free of
  repository prints and records on this path, though it is not a protected section.

