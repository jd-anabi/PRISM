# The window as read for the redesign: the inventory

Appendix to `2026-10-05-the-window-as-read-for-the-redesign.md`. The first readers' own maps of the
window, as returned on 2026-10-05 against commit `71a53b5`; their wording, unedited. Read from the
code, not run. Use it as a map to check against the code, not as a specification.

## The shell, start-up, settings and the look

### What the user sees

Paths: bare file names are under `C:/Users/J/PycharmProjects/PRISM/core/gui/` (`screens/`, `panels/`, `widgets/` as named).

**Window.** Title "PRISM" (main_window.py:88). Opens at min(1300, avail-40) x min(820, avail-60) of the primary screen (:44-49, :92); minimum 900x600 (:95). Saved geometry is used only if `restoreGeometry` succeeds and the frame intersects an attached screen, else default size, centred (:187-196). Fusion style (app.py:98). No menu bar, status bar or toolbar; no keyboard shortcut anywhere in core/gui (grep: no QShortcut, QKeySequence, keyPressEvent).

**Shell, on every screen** (nav_shell.py:56-110; margins 10/8/10/10):
- top-left label "PRISM" (20 px semibold); directly under it a back-arrow tool button (icon-font glyph, fallback "←"), tooltip "Back to the home screen", hidden on Home (:60-69);
- right of that column the run slot: a bordered 12 px tool button, hidden unless a run is live, text "Running: <tab label> — m:ss" (from one hour "h:mm:ss"; unknown panel reads "a task") (:25-38, :75-78), tooltip "<Section> → <Tab>: go to it" (main_window.py:372-373);
- a QStackedWidget of 8 screens (stretch);
- bottom-left gear tool button, tooltip "Appearance & settings", instant popup: "Follow system", "Light", "Dark", "Auto (time of day)" (exclusive, checkable), separator, "Full settings…" (nav_shell.py:95-104; main_window.py:198-214; theming.py:24-25).

**Home** (stack 0; home_screen.py:35-66). One centred column (stretch 1 above, 2 below, spacing 14): a greeting label, 28 px semibold — "Good morning" (05-11 h), "Good afternoon" (12-16), "Good evening" (otherwise), refreshed on every show; 10 px gap; five accent push buttons, minimum 260x44: "Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate", "Artifacts". The running section's tile gains the suffix "  ●" (:71-80). Nothing else: no status, no last run, no session summary.

**Section screens** (section_screen.py:19-38; inference_screen.py:47-84): a heading label (16 px semibold) over a QTabWidget.
- "Reduction Map": tab "NWK → Hopf reduction map".
- "FDT Analysis": tabs "FDT analysis", "Sweep study cross-validation".
- "Parameter Inference": tabs "Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE"; at launch only Config is enabled. Disabled-tab tooltips (inference_screen.py:206-215): "Apply a model in Config first."; "Choose a bounds file and build/load a prior first — that is what builds the config."; "Needs a posterior AND its prior — build/load a prior, then a posterior."; "Train or load a posterior first."; "Needs a posterior AND its prior. It also needs an observation, which the Infer tab records when it runs."
- "Simulate": tab "Live simulation".

The running tab gains the prefix "● " (nav_shell.py:41-53). Every tab page is a BasePanel: horizontal splitter [controls in a scroll area, minimum width 360, default 420 | results, default 880]; results = vertical splitter [figure stack | log pane] (default 520/200, non-collapsible) above a progress row ending in "Cancel" (hidden until a run; becomes a disabled "Cancelling…") (panels/base_panel.py:128-191, 291-296). Panel contents belong to other readers.

**Artifacts** (stack 5) and **Model builder** (stack 7; reached from Settings; its own Back returns to Settings, the shell arrow goes Home; main_window.py:143-146): not my files.

**Settings** (stack 6; only via gear → "Full settings…"; settings_screen.py:68-142), one scroll area:
- heading "Settings";
- group "Appearance": radios "Follow system" (default, settings.py:71), "Light", "Dark", "Auto (time of day)"; checkbox "Use the Windows accent colour" (default off; hidden off Windows); checkbox "Use the Inter font everywhere" (default off); caption "Font changes apply immediately; some views refresh fully after a restart.";
- group "User-defined models": caption "No user-defined models yet. Build one to add it to every model list." or one row per model (name, "Edit", "Delete"); captions "Failed to load <file>: <msg>"; accent button "Open model builder";
- group "Help — what each section does": a QTextBrowser (minimum height 260) of generated HTML: per section ("Reduction Map", "FDT Analysis", "Cross-validation (sweep study)", "Parameter Inference", "Simulate") the panel module docstring's first paragraph, then one bullet per entry of that module's HELP dict as "**<dict key>**: text" (:184-205).

No setting for the store location, the VRAM ceiling, the device, logging, notifications or anything about runs.

**Dialogs, verbatim.**
- Close with a run live (main_window.py:424-435): Warning icon; title "A task is still running"; text "A task is still running."; informative text "Cancelling stops it at its next checkpoint (up to ~1 min during training). Quitting anyway closes the window now, but the task keeps running in the background until it finishes on its own."; buttons "Cancel task & quit" (AcceptRole, the DEFAULT), "Quit anyway" (DestructiveRole), "Don't quit" (RejectRole). No escape button is set in code; by Qt's documented rule Esc and the dialog's X resolve to the single RejectRole button.
- "Delete model": "Delete the user-defined model '<name>'?" / "This removes its definition and its generated Bounds/Cells/Units files."; Yes/No, default No (:288-294). "A task is running": "Wait for the running task to finish before deleting a model." (:284-286). "Cannot edit model", "Delete failed" (:278, :300).
- "Unhandled error": "<Type>: <message>", traceback under Details (app.py:25-30). "PRISM could not start": "The application failed to start:\n<e>" (app.py:111-116).

### How the work flows

**Start-up, in order.**
1. `__main__.py:9-13`: Agg backend, then `core.gui.app` is imported, which imports `core.config` (`import torch`, core/config.py:10) and every panel and screen module (main_window.py:13-30). `KMP_DUPLICATE_LIB_OK` is set only by run.bat:5.
2. `build_app` (app.py:61-127): root log handler (:67) → `config.QUIET_SEGMENT_BAR = True` (:72) → `registry.load_user_models()` (:76) → artifact root created, default store set (:80-83) → AppUserModelID (:87) → QApplication, org/app names (:89-91) → icon (:94) → Fusion (:98) → font (:100) → `Appearance`; `set_system_accent` then `set_mode` each run a full palette + QSS apply (:101-103) → matplotlib rcParams (:104) → excepthook (:107) → `MainWindow` (:109; any exception → "PRISM could not start", exit 1) → window-class icon through `winId()` (:126).
3. `MainWindow.__init__` builds everything eagerly (:97-196): NavShell; ReductionPanel; FdtPanel, CrossValPanel; InferenceScreen (six panels, then `refresh_gates`); SimulatePanel; Home; ArtifactScreen; Settings; model builder; index map; `_panel_home`; an unstarted 1 s timer; gear menu; `go_home()`; geometry.
4. Card and external program before any click: ConfigPanel runs `nvidia-smi` synchronously, 2 s timeout (panels/inference/config_tab.py:119→151; core/artifacts/provenance.py:122-123); PriorPanel calls `config.detect_device()` (prior_tab.py:121→297; core/config.py:66-68); PosteriorPanel and TSNPEPanel call `_sync_budget` → `training_preview` → `torch.cuda.mem_get_info` (posterior_tab.py:112; tsnpe_tab.py:105; inference/base.py:138-139; core/orchestrator.py:529-532; core/SBI/pipeline.py:635; core/config.py:130); `refresh_gates` repeats both (inference_screen.py:80, 217-218; posterior_tab.py:338; tsnpe_tab.py:211), as does every later stage and every keystroke in a budget box.
5. `window.show()`; `app.exec()` (`__main__.py:18-19`).

**Navigation.** Tile → `navigate(name)` → `nav.go_to(index)` (main_window.py:152). `_navigate`: snapshot, `setCurrentIndex`, back arrow visible iff index ≠ 0, slide (nav_shell.py:122-133). Back arrow → Home from every screen. No shortcut and no Escape handling. Screen and tab are never restored.

**Inference gates** (inference_screen.py:183-223): Prior needs a draft; Posterior a config; Validate and TSNPE a posterior and a prior; Infer a posterior; then every panel's `refresh_local_gates()`; a current tab that became disabled falls back to Config. `new_draft` (Config's Apply) replaces the session, calls `prior_panel.on_draft_set`, re-gates (:163-165); `install_config` (Prior stage) sets `cfg` in place, calls `infer_panel.on_config_built`, re-gates (:179-181).

**Run state.** `BasePanel._set_busy` disables every panel's controls column and emits `RUN_STATE.changed(panel | None)` (base_panel.py:302-321) → `_on_run_state` (main_window.py:351-374): marks tile and tab, shows the slot, starts the 1 s `time.monotonic` clock; on None it stops, hides and clears — nothing records how or when the run ended. Slot click → `_go_to_running` (screen and tab, :381-392). Navigation, gear, Settings and Artifacts browsing stay live during a run.

**Other cross-screen signals.** `artifact_screen.store_changed` → `_refresh_store_pickers` (every `StorePicker.refresh()`; :124, :331-344). Model builder save or model delete → `_refresh_model_combos` + `settings_screen.refresh_models()` (:305-329).

**Close** (:416-445). Run live → dialog. "Cancel task & quit": `BasePanel.request_cancel_all()` — sets the token's event and nothing more (base_panel.py:114-118) — then `_save_state()` and the window closes at once. "Quit anyway": nothing is done to the run; `_save_state()`; close. Any other outcome: `event.ignore()`, nothing saved. Afterwards `app.exec()` returns and `SystemExit` is raised (`__main__.py:19-23`) with the pool thread still inside the stage; nothing in core/gui waits for it (no `waitForDone`, no `setQuitOnLastWindowClosed`). What the process does next: not established.

**Appearance.** Gear action or radio → `set_appearance_mode` (:216-231): write + sync, `Appearance._reapply` (`setColorScheme`, `setPalette`, `setStyleSheet`, `theme_changed`; theming.py:100-115), mirror the other control. Auto mode re-applies every 60 s (theming.py:65-67).

**Blocks the GUI thread:** all construction before the first show; `nvidia-smi` at launch and on every keystroke in the ceiling box (config_tab.py:117); every modal `QMessageBox.exec()`.

### What is remembered and what reopens fresh

**Store.** `QSettings(IniFormat, UserScope, "PRISM", "PRISM")` (settings.py:8-9, 20-23); tests redirect it with `use_ini_file` (:14-17).

**Keys handled in my files.**
- `window/geometry` — read once at construction (main_window.py:187-196); written only by `_save_state` on an accepted close (:450).
- `appearance/mode` (default "system"), `appearance/system_accent` (default off), `appearance/force_inter` (default off) — read in `build_app` (app.py:99-103) and `MainWindow.__init__` (:128-138); written and synced the moment they change (:223-225, :236-238, :244-246).
- `layout/<PanelClass>/splitter` and `/results` — restored in `BasePanel.__init__` (base_panel.py:203, 372-389); written 1.5 s after a splitter drag (:196-201, 391-396). NOT written at close: `_save_state` never calls `save_layout`, although base_panel.py:359 and :368 say it does.
- Every panel's selections: `_save_state` calls each panel's `save_settings(qs)` and `artifact_screen.save_settings(qs)`, then `sync()` (main_window.py:451-456). This runs on all three accepted closes (no run; "Cancel task & quit"; "Quit anyway") and never on "Don't quit", a crash, a kill, or a timer. Each panel restores itself inside its own `__init__`.

**Reopens at its default on purpose.**
- The screen: always Home (main_window.py:78, 183-186). The tab inside a section: neither screen class persists anything (section_screen.py:16; inference_screen.py:44).
- The whole inference session: a fresh `SbiSession()` (inference_screen.py:49), so Prior through TSNPE open disabled and every launch needs Apply, then Build / Load prior, before the Posterior tab can be reached.
- The run slot, log panes and figures.

**Surprising.**
- The Prior tab's remembered Bounds is only stashed at launch and shown after Apply — and re-applied on every later Apply (prior_tab.py:66, 141, 373) — while the close-time write stores whatever the picker shows (:363). The Infer tab's Cell behaves the same way (inference/base.py:35-40; infer_tab.py:530, 545).
- An empty remembered pick leaves a picker on its first row: `restore_key("")` returns at once (widgets/artifact_picker.py:79-86, 249-254).
- "Auto" appearance follows the wall clock: light 07:00-18:59, dark otherwise (theming.py:29-30, 47).
- `get_int` and `get_bool` fall back to the caller's default on a blank or unparseable value (settings.py:40-66); `restore_field` leaves a field alone when its key is absent (:96-102).

### How the look is produced

**One source, one sheet.** `design.py` holds the tokens; `theming.Appearance._reapply` calls `app.setPalette(build_palette(...))` and ONE application-level `app.setStyleSheet(build_qss(...))` (theming.py:111-112) over Fusion (app.py:98). No per-widget `setStyleSheet` exists in core/gui (grep: theming.py:112 only). Widgets opt in by dynamic property (`type` = title/subtitle/heading/caption; `accent` = true) or objectName (#iconButton, #navBack, #navSettings, #navRunning, #helpBadge) (design.py:193-304).

**Themes: 2 (light, dark); 4 modes** — system, light, dark, auto (theming.py:24-30) — plus an optional Windows-accent override (design.py:96-141). Tokens, light / dark (design.py:50-89): window #F3F3F3 / #202020; base #FFFFFF / #2B2B2B; alt_base #F7F7F7 / #333333; mid #D1D1D1 / #3D3D3D; mid_strong #C4C4C4 / #4A4A4A; text #1B1B1B / #F5F5F5; text_2nd #616161 / #C7C7C7; text_disabled #A0A0A0 / #6E6E6E; button #FBFBFB / #313131; button_hover #F0F0F0 / #3A3A3A; button_press #E5E5E5 / #454545; accent #0F6CBD / #4CA0E0; accent_hover #115EA3 / #62B0EE; accent_press #0C3B5E / #3A88CC; on_accent #FFFFFF / #1B1B1B; tooltip_bg #FFFFFF / #2B2B2B; scrollbar #B8B8B8 / #4A4A4A; scrollbar_hover #9A9A9A / #5E5E5E. No status colours (error, warning, success), no elevation or shadow tokens; a destructive button has no style of its own.

**Type.** Family: Segoe UI Variable Text → Segoe UI → bundled Inter (Inter first when forced); base 14 px (fonts.py:17, 27-39). Ramp px/weight: caption 12/400 (in text_2nd), body 14/400, body_strong 14/600 (not in the QSS), heading 16/600, subtitle 20/600, title 28/600 (design.py:39-46). QSS extras: icon button 18, nav glyphs 20, badge 13 (:378-380).

**Sizes.** Radius 4/8; control height 32/28; SPACE (4, 8, 12, 16, 20, 24), used at three call sites only (artifact_screen.py:511, 513; model_builder_screen.py:188). Layout: CONTROLS_MIN_W 360, DEFAULT_SPLIT (420, 880), DEFAULT_RESULTS_SPLIT (520, 200), FIELD_MIN_W 96, PATH_FIELD_MIN_W 220, LABEL_MAX_W 190, HELP_VIEWER_MIN_H 260 (design.py:18-35). Scattered literals: QSS paddings and radii (:200-324); shell margins 10/8/10/10, spacing 2 (nav_shell.py:82, 107); Home 14, 10, 260x44 (home_screen.py:45-56); panel margin 6 (base_panel.py:190); 32 px refresh buttons; progress pane 88x18, 150, 18.

**Icons.** A five-glyph icon FONT (`assets/icons/prism-icons.ttf`, U+E000-E004: back, settings, refresh, help, close; generator `build_prism_icons.py`) drawn as button TEXT, so the QSS `color` tints it; unicode fallbacks (icons.py:19-25, 62-73). App icon: `assets/app/prism-{16,24,32,48,64,128,256}.png`, `prism.ico`, `prism.svg`, `build_app_icon.py`. Fonts: `assets/fonts/Inter-Regular.ttf`, `Inter-SemiBold.ttf`.

**Figures.** matplotlib rcParams from the neutral tokens: figure and savefig = window; axes and legend face = base; axes edge and ticks = text_2nd; text and labels = text; grid and legend edge = mid (mpl_theme.py:16-28). Series colours and fonts stay matplotlib's defaults. Figures are PNGs at dpi 110 shown as pixmaps; a theme flip does not recolour ones already shown (:8-9). The pyqtgraph view re-themes on `theme_changed`.

**Custom paint and motion.** `_Sparkline` reads QPalette.Mid and Highlight (design.py:8-11). `_TransitionOverlay` animates page snapshots: screen slide 280 ms OutCubic, tab cross-fade 180 ms InOutQuad, off when offscreen or not yet visible (anim.py:19-27, 106-133).

**Resists change.** Unstyled by the QSS (Fusion + palette): QCheckBox, QRadioButton, QSplitter handles, QListWidget (compare_list.py:25), QTextBrowser, QMessageBox. Every screen is built inline in a long `__init__` (MainWindow :85-196). Size-policy workarounds: `_size_to_current_tab` (inference_screen.py:90-102), AdaptiveStack. Test-pinned: tab labels, `running_banner` text, #navRunning, icons.NAMES against the font's cmap.

### What a redesign must keep, or ask before changing

- Paths below are relative to C:/Users/J/PycharmProjects/PRISM/ ; 'gui/' means core/gui/.
- The Agg backend must be forced before any core import (gui/__main__.py:5-13): it is what makes plotting on the
  worker thread safe. A direct launch still needs KMP_DUPLICATE_LIB_OK set by hand (run.bat:5; __main__.py sets
  only the backend).
- Theming stays out of MainWindow: only build_app constructs an Appearance and tests build
  MainWindow(appearance=None) (gui/app.py:95-97; gui/theming.py:9-10; gui/main_window.py:87, 128-129, 226, 239).
- The style sheet uses named-class selectors only, never '*' or bare QWidget: those would paint behind the
  sparkline, the pyqtgraph view, figure labels and the transition overlays (gui/design.py:188-190). Do not null
  QComboBox::down-arrow: there is no arrow asset (:229).
- QPalette.Mid and QPalette.Highlight are read by custom-painted widgets, and the Disabled group must be filled
  because the whole controls column is disabled during a run (gui/design.py:8-11, 145-147, 166, 171, 177-183;
  gui/widgets/progress_pane.py:132, 152).
- Every dialog is an instance QMessageBox shown with .exec(), never a static call: the suite's guard patches the
  instance method only and a static would stall the run (gui/main_window.py:258-266; tests/conftest.py:171-190).
  Safe defaults are set by name, not by button index (gui/panels/inference/config_tab.py:276-279); tests pin
  'Keep this session' and 'Cancel' as defaults (tests/test_nav_and_gating.py:1323-1324, 4530-4531).
- ASK THE OWNER: the close dialog's default button, its wording and whether 'Quit anyway' stays. No test pins it
  (gui/main_window.py:424-435), and what the two quitting buttons do to a live run at process exit is not
  established.
- Nothing may run on the launch path between the native show and the first idle turn: the run slot is built
  empty and its timer unstarted, and the window-class icon is set before the first show — the taskbar showed
  python's icon otherwise (gui/screens/nav_shell.py:71-74; gui/main_window.py:171-172; gui/app.py:122-126;
  gui/app_icon.py:54-72).
- Navigation state changes synchronously before any animation, and animations are off offscreen or before the
  window is visible: tests read the index and the back arrow with no event-loop pump (gui/widgets/anim.py:10-13,
  23-27; gui/screens/nav_shell.py:124-130).
- Both tab-hosting screen kinds must expose .tabs (a QTabWidget), panels() in tab order and set_running_tab:
  MainWindow builds its panel-to-place map, its settings sweep, the model-combo refresh (needs panel.model_combo
  and _on_model_changed) and the picker refresh (finds StorePicker children) from them
  (gui/main_window.py:154-169, 316-329, 342-348, 409-410).
- Tab labels are read back verbatim by tests and running marks must be rewritten from the stored labels, not
  edited in place (gui/screens/nav_shell.py:41-53; tests/test_nav_and_gating.py:1535, 4375, 4386-4387,
  4411-4412). refresh_gates owns each tab's enabled state and tooltip, so a mark may only be tab text
  (gui/screens/inference_screen.py:142-146).
- running_banner's output and the navRunning object name with its selector are pinned
  (tests/test_nav_and_gating.py:4224-4235, 4250-4252). The running mark must stay a plain character, not an
  icon-font glyph (gui/screens/nav_shell.py:17-22).
- Icons are text in a bundled icon font so they recolour from the style sheet; icons.NAMES must match the font's
  character map, so a new icon means regenerating prism-icons.ttf with gui/assets/icons/build_prism_icons.py
  (gui/icons.py:1-25; tests/test_user_models.py:1130-1150). A QIcon would not follow a theme flip
  (gui/widgets/artifact_picker.py:16-22). Qt's SVG image plugin is absent here: the app icon must load from PNG
  and ICO (gui/app_icon.py:3-7).
- Pinned token behaviour: design.tokens accent override and the on_accent luminance rule, build_palette and
  build_qss signatures, and fonts.load_app_font(prefer_inter=True) returning 'Inter'
  (tests/test_user_models.py:1044-1056, 1070).
- The Artifacts screen must not become a BasePanel: a run anywhere would grey out the log being read. Its place
  in the stack keeps the slide direction in step with the Home tile order (gui/main_window.py:119-121;
  gui/panels/base_panel.py:92-93).
- One run at a time app-wide, held at class level because output redirection is process-wide; the run-state
  handler must touch only its own window's widgets (gui/panels/base_panel.py:102-112;
  gui/main_window.py:355-359).
- The two ways a config enters the session must stay distinct: new_draft replaces the session and is the one
  that is confirmed; install_config writes in place mid-stage and is never confirmed
  (gui/screens/inference_screen.py:148-181). session_contents() and every derived status line must never raise,
  and name no control (:119-125). The Validate gate must not require a force prior (:187-192).
- ASK THE OWNER before changing what is remembered: always opening on Home with a fresh session is deliberate
  (gui/main_window.py:78, 183-186); science knobs and consents reopen at config.py on every launch and the VRAM
  ceiling is deliberately not persisted (project rule in CLAUDE.md; gui/panels/inference/config_tab.py:137-143).
  A Settings screen that stores any of these breaks that rule. Selections are written only on an accepted close
  (gui/main_window.py:444-456) — changing that is the owner's call too.
- Source-text pins: the MainWindow class source must contain '_persist_layout' and '1500' and must not contain
  'the only QSettings WRITE site' (tests/test_nav_and_gating.py:4760-4764). No shippable file may cite the
  working record or its labels (tests/test_source_hygiene.py, project rule).
- Settings' Help imports the five panel modules by string path and reads their docstring and HELP dict;
  gui/panels/inference_tabs.py is kept as that import surface and tests reach _ChiProbeRow and
  _nvidia_smi_free_gib through it (gui/screens/settings_screen.py:23-29, 184-205;
  gui/panels/inference_tabs.py:15-18).
- Keep at launch: the root log handler installed first, the artifact root created and resolved before any
  window, and the error hook installed before MainWindow is built (gui/app.py:62-67, 78-83, 106-107).
- Geometry restore must check restoreGeometry's return value and that the frame is on an attached screen; the
  900x600 minimum and the fit-to-screen default exist because controls fell below the fold on small or scaled
  screens (gui/main_window.py:89-95, 190-196).
- Figures are built on the worker and shown as PNG pixmaps; the GUI never paints a live canvas created
  off-thread (gui/panels/base_panel.py:31-43). Figure theming goes through global rcParams so the core plot
  modules stay usable from the command line (gui/mpl_theme.py:5-9); the Simulate video export stays white
  (gui/theming.py:15).
- The help badge takes no keyboard focus and is sized through scaled(); form labels wrap at LABEL_MAX_W so one
  long name cannot narrow every field (gui/widgets/help_badge.py:25-29, 43-48).
- ASK THE OWNER: opening the window reads the card and runs nvidia-smi (see the 2A claim). The project rule 'no
  second window beside a long run' rests on that; a redesign either keeps the rule or moves those reads off the
  launch path.

## The config and prior tabs

### What the user sees

**Shell around both tabs.** Heading "Parameter Inference" over a QTabWidget: Config, Prior, Posterior, Validate, Infer, TSNPE (inference_screen.py:51-73). Each tab is a BasePanel: horizontal splitter, left = scroll area with the controls column (min 360 px, opens 420), right = vertical splitter (figure stack over log pane, 520/200) plus a progress row with a hidden "Cancel" (base_panel.py:128-191). Disabled-tab tooltips: Prior "Apply a model in Config first."; Posterior "Choose a bounds file and build/load a prior first — that is what builds the config." (inference_screen.py:206-209).

**Config tab** (config_tab.py:42-119). Group box "Config", one form; each row = label + "?" badge whose tooltip is the HELP paragraph:
1. "Model" — combo of VALID_MODELS (BP, NADROWSKI, HOPF, plus user models); NADROWSKI, then the remembered model (:44-46, :415).
2. "Units" — radios "Model's units file" | "Type units" (:58-59). File page: caption "<tokens>   —  Resources/Units/<model>/units.txt" (:188). Typed page: line edit, placeholder "e.g. nm ms pN kHz", seeded from the file's tokens when empty (:189-190).
3. Checkbox "Multi-frequency χ(ω) conditioning"; off unless remembered (:64, :421).
4. "χ probes per observation" — IntField, 6 (config.CHI_N_FREQS).
5. "χ probe slots (capacity)" — IntField, 12 (CHI_K_PAD).
6. "χ drive F₀ (ND)" — read-only FloatField, 0.15 (CHI_F0).
7. "χ frequency range" — "×Ω₀ from" [0.03] "to" [0.3], both read-only (rows.py:14-16).
8. Caption "fixed by measurement; change it in config.py" (:84).
9. "χ lock-in ceiling (cycles)" — FloatField, 20.0 (CHI_MAX_CYCLES).
   Fields 4-9 are disabled while the tick is off (:172-174).
10. Checkbox "Decorrelating Fisher rotation"; remembered, else config.REPARAM_ROTATE = True (:89, :422).
11. Accent button "Apply model & options" (:48-50, :92).
12. Permanent session line (:99-100, :380-387): "This session holds nothing yet, so applying a model discards nothing." or "This session holds <phrases joined by '; '>.\nApplying a model starts a new session and releases them. Each one stays on disk and can be selected again in the pickers." Phrases (inference_screen.py:129-139): "the built config for NADROWSKI, chi observations", "the prior 'name'" or "the prior (unnamed, id <id>)", likewise posterior and observation. The applied draft is never listed.

Group box "Hardware", below the Apply button (:108-118): "VRAM ceiling per batch (GiB, 0 = off)" — FloatField opening at `pipeline.vram_ceiling_gib()`: "0.0", or the environment variable's value (:111). Note under it, one of (:152-165): "Off: batches are planned from the free-memory reading and the learned cap alone. nvidia-smi reports X.XX GiB free — set a ceiling near that, minus ~1 GiB, only if you need the desktop." / "Batches will be planned to fit X.XX GiB. nvidia-smi reports X.XX GiB free. Above the real free figure this does nothing; below it, expect more splitting and more wall-clock." / "(warning sign) PRISM_VRAM_CEILING_GIB=<v> is set and OVERRIDES this field — planning to X.XX GiB. …". Without the binary: "free VRAM unreadable (no nvidia-smi)".

**Dialog on Apply when the session holds anything** (:288-303): warning icon; title "Start a new session?"; text "Applying a model replaces the whole session."; informative "This session holds:\n\n  • <phrase> …\n\nEvery one of them STAYS ON DISK and can be selected again in the pickers — what is released is this session's hold on them. They cannot be carried over: a different model or unit system invalidates them, which is why applying one starts a new session rather than keeping what it can." Buttons "Apply and start a new session" (destructive role) and "Keep this session" (reject role, the DEFAULT, :301).

**Prior tab** (prior_tab.py:67-134). Group "Prior":
- "Bounds" — radios "Use file" | "Edit values" (:72-73). File page: combo of every file under Resources/Bounds/<model> plus a refresh icon button (tooltip "Rescan the folder"); opens on the nadrowski folder's first file (:70). Edit page: scroll area capped at 320 px with captions "Non-dimensional", "Dimensional", "Forcing" and one row per parameter, "<name>  min [ ]  max [ ]"; placeholder "(load a bounds file to edit its values)" (param_grid.py:21, 92, 101-113). No line describes the chosen file.
- "Prior" — combo: "+  (from scratch)" first, then each finished prior by name or "(unnamed <id>)"; refresh button "Rescan the artifact store"; item tooltip "<id> · <created> …" (artifact_picker.py:22, 185-218; store.py:161-162). Under it one line describing the entry shown (mode · width · amortization · created, whichever the kind carries; empty for from-scratch) (:82-85, :349).
- Accent button "Build / Load prior" (:87).
- Line edit, placeholder "name to save prior as…", and button "Save" (:91-98).

Group "Stability sweep": "Global rounds" 50; "Candidates per round (0 = auto)" 0; "Max accepted sets" 175000; "Random-walk step" 0.01; "Stability duration (ND units)" 1000 (config.py:628-654). Note (:305-311), at launch on CUDA: "Global census screens 102,400 candidates (50 rounds x 2,048), each integrated over ? steps.\nThe LOCAL flood-fill then runs until 175,000 sets are accepted — that is the dominant cost of a prior build, and it now runs on the same device as the global sweep (falling back to the CPU when there is no accelerator)."

Group "Clustering / GMM": "Min cluster size" 50; "Min samples" 10 (config.py:671-672).

**Refusal box** (refusal_box.py:33-42): modal, warning icon, title "Check your inputs", the core's sentence, then "Set it in the '<label>' box on the <tab> tab.", one Ok. Bugs get a red "Error" box with Details (base_panel.py:448-457).

### How the work flows

**Gates.** Prior tab enabled once a draft exists; Posterior once `session.cfg` exists, which is set at the Build / Load prior CLICK, before the prior is ready; Validate and TSNPE need posterior + prior; Infer a posterior (inference_screen.py:183-204). "Build / Load prior" enabled iff a draft; "Save" iff a prior is held (prior_tab.py:351-354). Apply is disabled only for an SBI-ineligible user model (config_tab.py:206-207). During any run every panel's whole controls column is disabled (base_panel.py:316-317, 333).

**Apply model & options** (config_tab.py:305-360; GUI thread, no worker). `_read_inputs`: with χ on, 2 <= K <= slots <= 24 and ceiling > 2 cycles; typed units non-blank and resolvable (`cli.units_to_factors`); a failure is a Refusal → yellow box, session untouched. Builds `ConfigDraft(model, labels, state_dep_drift, units_override, chi_mode, chi_n_freqs, chi_f0=None, chi_freq_bounds=None, chi_k_pad, chi_max_cycles, reparam_rotate)`. If `session_contents()` is non-empty → "Start a new session?". Then `InferenceScreen.new_draft`: a fresh `SbiSession(draft=…)`, `PriorPanel.on_draft_set` (bounds picker repointed to Bounds/<model>, the LAUNCH-time key re-applied; grid reseeded when in Edit values), `refresh_gates`. Log line, in the Config tab's own pane: "Model applied: … Now pick a bounds file on the Prior tab — that builds the config and selects the observation mode." No tab switch.

**Build / Load prior** (prior_tab.py:168-245). Nothing is asked. Picker = from-scratch → the seven knobs through `_check_knob` (Refusal → yellow box); picker = a stored prior → no knob is read. Bounds: the file path, or the grid (`problems()` → a log-pane warning line only) as `bounds_dicts`. `draft.make_config(...)` → `cli.make_sim_config` on the GUI thread. Then `session.reset_downstream("prior")`, which drops the held prior, observation AND posterior with no question (session.py:49-57); `install_config(cfg)` (sets `session.cfg`; `InferPanel.on_config_built` repoints the Cell picker with its own launch-time key; re-gates); log lines "Config built: … Observation mode: CHI (…)"; then `dispatch(orchestrator.build_prior, cfg, entry, is_new, fig_sink=…, num_iterations, sweep_batch, max_sets, walk_step, stability_units, min_cluster_size, min_samples)` on a QThreadPool worker — no name, no note. `_on_prior`: `session.inf_prior = payload`, log "Prior ready: <name>. Name it below to keep it.", `refresh_gates`. The picker is not refreshed, so after a from-scratch build it still shows "+  (from scratch)".

**Save** (prior_tab.py:321-338): `default_store().rename("prior", lp.id, name)` — an in-place rename of the held record, loaded ones included (store.py:852-861); then picker refresh and re-select.

**Edit values** (prior_tab.py:145-166): parses the selected file and rebuilds the grid. Toggling away and back, or any Apply, reloads from the file and discards edits.

**VRAM field** (config_tab.py:117, 128-165). Every keystroke assigns `config.SIM_VRAM_CEILING_GIB = max(0, value())` (blank, "-" or a negative = 0 = off) and runs `nvidia-smi` synchronously on the GUI thread (2 s timeout, provenance.py:122-123); once more at construction. No CUDA call. Nothing notifies the Posterior tab: its memory line re-reads the planner budget only on a budget keystroke or after a stage (posterior_tab.py:81, 338).

**Other GUI-thread work.** Prior tab: `config.detect_device()` at construction and again on every keystroke in four sweep boxes until a config exists (prior_tab.py:118-121, 297; not memoised). `StorePicker.refresh` lists the prior store at construction, on its refresh button, and whenever the Artifacts screen emits `store_changed` (main_window.py:124, 331-344). After every stage `refresh_gates` calls each panel's `refresh_local_gates` (Config: session line; Prior: two buttons + prior line; Posterior: budget lines).

**Logs are per tab**: Apply's lines sit in Config's pane, the config and mode lines in Prior's; neither is saved.

### What is remembered and what reopens fresh

Written only by `MainWindow._save_state` on an accepted close (main_window.py:444-456; no other caller of `save_settings` exists); read once, when each panel is constructed.

- `inference_config/model` — combo text (config_tab.py:403, 415).
- `inference_config/units_mode` — "file"/"direct" (:404, 420).
- `inference_config/units_text` — the typed-units box, written in either mode (:405, 418).
- `inference_config/chi_mode` — "1"/"0", default False (:406, 421).
- `inference_config/reparam_rotate` — "1"/"0", default config.REPARAM_ROTATE (:407, 422).
- `inference_prior/prior` — prior id, or "" for from-scratch; restored at construction (prior_tab.py:362, 370).
- `inference_prior/bounds` — path relative to the model's bounds folder; read into `_saved_bounds_key` at construction (:373), applied only at each Apply (:141).
- `layout/ConfigPanel|PriorPanel/splitter|results` — written 1.5 s after a drag (base_panel.py:196-201, 367-396).

Open at default on every launch, by design: χ probes, slots, lock-in ceiling; F₀ and band (displays); the VRAM ceiling (0.0, or the environment value); the seven sweep and clustering boxes; the Bounds source (always "Use file", prior_tab.py:376); the grid; the name box.

Surprising:
- Bounds is the only selection restored late. Before the first Apply the picker shows the nadrowski folder's first file, and that is what a close writes.
- `_saved_bounds_key` is never updated during a session, so every Apply returns the picker to the launch-time file, or, with an empty or vanished key, to the folder's first file.
- A remembered prior id that no longer exists restores silently to "+  (from scratch)". A from-scratch build never moves the picker off that entry, so closing after a build without Save writes "".
- The remembered χ tick, rotation tick, units and bounds file are all inside the simulation cache's identity (identity.py:92-98, 111, 121); the χ probe count is not.
- The Bounds picker's own refresh button resets the selection to the first file (artifact_picker.py:44-57; StorePicker re-selects, :182, :219).
- `units_text` seeded from one model is kept after the model changes (config_tab.py:189).
- A crash or a killed process writes nothing.

### How the look is produced

- One global style sheet and palette in `design.py` (`_QSS` template :191-314; LIGHT/DARK tokens :50-89; control heights 32/28, radii 4/8, type ramp caption 12 / body 14 / heading 16, :18-46). My files carry no per-widget style sheet. Styling hooks are dynamic properties and object names: `accent=True` on the two primary buttons (config_tab.py:49, prior_tab.py:88), `type="caption"` labels (config_tab.py:55, 85; param_grid.py:93, 107, 157), `iconButton` (artifact_picker.py:32, 166; rows.py:43), `helpBadge` (help_badge.py:22). QGroupBox is styled as a card (design.py:264-271).
- Hard-coded sizes: refresh button `setFixedWidth(32)` (artifact_picker.py:34, 168); probe frequency max 96 and remove button max 32 (rows.py:41, 45); grid ceiling `_MAX_VISIBLE_HEIGHT = 320`, its minimum height computed once at construction from the placeholder (param_grid.py:21, 62-64); row spacing 6 (field_row.py:24); toggle spacing 4 (source_toggle.py:49); FIELD_MIN_W 96, PATH_FIELD_MIN_W 220, LABEL_MAX_W 190, CONTROLS_MIN_W 360 (design.py:28-35); badge `scaled(16)`.
- Forms: every QFormLayout comes from `make_form` (fields grow, long rows wrap, labels left; forms.py:16-22); label + badge from `add_help_row`, which passes the text through `labels.pretty_gui` (help_badge.py:55-63).
- Derived lines (session line, VRAM note, prior line, sweep note) are plain word-wrapped PlainText QLabels in body type (base.py:63-75): no caption property, no colour; a warning sign and capitals are the only emphasis. Read-only χ boxes have no read-only style (design.py has `QLineEdit:disabled` only, :227), so with χ on they look editable.
- No custom painting and no matplotlib in these controls. The Prior tab's corner plot arrives as PNG bytes in the shared figure stack (base_panel.py:31-63). The Config tab runs no stage, so its figure stack and progress row are always empty.
- Resists a relayout:
  - both panels build every widget inline in `__init__` (config_tab.py:40-126, prior_tab.py:64-136) with order-dependent wiring (labels before signal connections; `restore_settings` last; session line last);
  - widgets are public attributes that tests read by name (`vram_ceiling`, `session_line`, `_saved_bounds_key`; tests/test_settings_persistence.py:754-784, 1123; tests/test_nav_and_gating.py:4454-4557);
  - row labels come from `fields.label(key)` and a read-back test compares them with the QLabel on the tab the table names (fields.py:6-10, 33-34);
  - `ConfigPanel` and `PriorPanel` borrow `_TrainingBudgetMixin._derived_label` as a static helper;
  - `inference_tabs.py` is an import surface that Settings' help reads by string path and tests reach private names through (:15-23);
  - `ConfigPanel` must construct with `screen=None` (config_tab.py:369-372);
  - the bounds grid is the app's only nested scroll area and needs the wheel pass-through (param_grid.py:33-53); `AdaptiveStack` makes the toggle's height follow the visible page (adaptive_stack.py:31-44);
  - `layout_key()` is the class name (base_panel.py:362-365), so renaming a panel class orphans its saved splitter state.
- A literal display path, "Resources/Units/<model>/units.txt", is built in a caption (config_tab.py:188).
- Unused imports in both tabs (`icons`, `PathField`, `ValuesGrid`, `with_badge`, `IntField` in places; `forcing` in prior_tab.py:5 is shadowed by a local at :157).

### What a redesign must keep, or ask before changing

- Science knobs are never remembered: the χ probe count, slots and lock-in ceiling, and the seven sweep and
  clustering boxes open at config.py every launch, and stale keys are ignored (config_tab.py:395-401, 410-423;
  prior_tab.py:356-364). Why: a restored value trained the 2026-08-19 run on a retired band. Ask the owner
  before remembering any of them.
- The VRAM ceiling is never persisted, opens off, and the environment variable wins; the planner reads it LIVE
  on every batch plan, which is the only reason assigning config.SIM_VRAM_CEILING_GIB works
  (config_tab.py:128-148; pipeline.py:584-603; config.py:161-168). Pinned by
  tests/test_settings_persistence.py:729-784. It is deliberately outside the cache identity. Remembering it, or
  turning it into an argument, is the owner's decision (the second is an engine change).
- The χ drive amplitude and band are read-only displays; the draft carries None for both so every config takes
  config.py's values (config_tab.py:66-75, 325-333). Do not make them editable.
- Two session entry points with different rules: new_draft replaces the whole session and is confirmed;
  install_config sets the config in place and is never confirmed, because it runs mid-stage
  (inference_screen.py:148-181). The dialog is silent when nothing is held, comes after the boxes are read and
  before anything is installed, and its safe default is set by NAME, not index (config_tab.py:263-303, 335-340)
  - Enter once started the very run a dialog existed to stop.
- Every modal box is an instance QMessageBox shown with .exec(), never a static: the test guard patches the
  instance method only and a static stalls the offscreen suite (config_tab.py:279-281; refusal_box.py:29-31).
- A refusal names a setting by field key; the window names the control from fields.CONTROL, and rows are built
  from fields.label(key). A renamed or moved control must be changed there (the place is the tab title exactly
  as shown), and a read-back test compares label and QLabel (fields.py:1-34, 190-232).
- A derived status line may never raise into refresh_gates: tests put object() on session fields and build
  ConfigPanel with screen=None (base.py:122-126; config_tab.py:368-379; prior_tab.py:283-284, 312-313;
  inference_screen.py:119-122).
- A blank box is a refusal at the click, never a zero; live lines say '<label> is blank.' and never clamp or
  default (base.py:108-116; prior_tab.py:286-289; labeled_inputs.py:31-42). The note and the click read ONE rule
  table (prior_tab.py:24-37).
- Every knob reaches build_prior as an argument, never a config write, and a load reads none of the seven boxes
  (prior_tab.py:239-262). The Config boxes reach the stage through ConfigDraft → cli.make_sim_config
  (session.py:27-37).
- Hand entry edits numbers only: parameter names and order come from the selected file because simulators bind
  columns by position; the grid is never persisted and the Bounds source always opens on 'Use file'
  (param_grid.py:1-11; prior_tab.py:61-62, 365-366, 374-376).
- Tabs never cache the session object; they read it through the screen, because Apply replaces it wholesale
  (base.py:14-25).
- Pickers: StorePicker lists FINISHED records only (artifact_picker.py:192-198); the id is the userData that
  key/selected/restore use, so display wording is free to change (:201-203); the summary line must not re-list
  the store on each index change (:238-241); repoint's order is folder, refresh, then saved key (:88-100); a
  combo item cannot carry the icon font (:16-22).
- The whole controls column of every panel locks during a run, because a model combo changed mid-run repoints
  pickers under the worker (base_panel.py:312-317, 323-333). Anything the redesign wants live during a run (the
  ceiling, a note) must sit outside `controls` and be shown safe; ask the owner.
- The free-memory figure shown to the user comes from nvidia-smi, never torch.cuda.mem_get_info, which
  overstates it by the desktop's share (provenance.py:114-128; base.py:182-184). Whether launch, or a keystroke,
  may probe the card at all beside a live run is a question for the owner.
- The bounds file is a remembered selection and `_saved_bounds_key` is pinned by name
  (tests/test_settings_persistence.py:1123). Fixing the look-and-close overwrite or the Apply jump changes when
  it is written and restored; keep 'selections are remembered' and rewrite that test first.
- A record name, a note, a seed and a consent are never remembered: the prior name box is not saved
  (prior_tab.py:356-364), and the session dialog is answered each time.
- An SBI-ineligible user model must stay unable to Apply, with the backstop in the click handler
  (config_tab.py:204-213, 307-311).
- Units declare what the numbers mean and never convert; typed tokens are checked at Apply, not mid-run
  (config_tab.py:53, 251-260), and they fingerprint the simulation cache (identity.py:28-32). The χ tick,
  rotation tick, slots, lock-in ceiling and bounds are in that identity too (identity.py:92-98, 111-117): any
  change to how these controls default or restore can re-key the finished cache. Tell the owner first.
- Layout fixes already paid for: make_form's grow-and-wrap policy (forms.py:1-22), FIELD_MIN_W and LABEL_MAX_W
  with wrapped labels (labeled_inputs.py:9-12; help_badge.py:43-48), no maximum on the controls column
  (base_panel.py:137-140), the nested scroll area's wheel pass-through (param_grid.py:33-53), AdaptiveStack and
  size-to-current-tab (adaptive_stack.py:1-15; inference_screen.py:55-59, 90-102). Each answers a reported
  'input boxes are cut off' or stuck-scroll defect.
- Worker figures cross to the GUI as PNG bytes, never a live canvas, which deadlocks on matplotlib's lock
  (base_panel.py:31-63); the Prior tab's corner plot relies on it.
- `core/gui/panels/inference_tabs.py` is an import surface: Settings reads HELP and the docstring by string
  path, and tests reach `_ChiProbeRow` and `_nvidia_smi_free_gib` through it (inference_tabs.py:15-23). HELP is
  one table shared by several tabs (help_text.py:1-3).
- Splitter state is saved apart from selections and keyed by the panel's class name (base_panel.py:355-370);
  renaming or merging panel classes orphans it.
- One probe is one widget pairing a recording with its frequency, so a deletion cannot mis-pair them
  (rows.py:22-35). In my files, used by the Infer tab.
- No code outside core/config.py builds a literal Resources/ or Artifacts/ path, and nothing shippable cites the
  working record (project rules). The caption at config_tab.py:188 prints a literal path string; a redesign that
  shows paths should take them from config's roots.

## The posterior, validate, infer and tsnpe tabs

### What the user sees

Paths are under `C:\Users\J\PycharmProjects\PRISM\core\gui\`. Every tab is a `BasePanel` (panels/base_panel.py:127-191): horizontal splitter, left = scroll area holding the controls column (top-aligned QVBoxLayout, min width 360, default split 420/880), right = vertical splitter (figures over log, 520/200) with a progress row and **Cancel** (hidden until a run) beneath. Groups are QGroupBox + QFormLayout (`make_form`). Row labels carry a 16 px "?" badge (tooltip = `HELP[...]`, click pins it). All numeric boxes are validated QLineEdits, no spin boxes.

**Posterior** (posterior_tab.py:37-109)
1. Group "Posterior": **Posterior** = combo + 32 px refresh icon ("Rescan the artifact store"); first item `+  (from scratch)`, then finished posteriors by name or unnamed-with-id, narrowed ones suffixed `  —  narrowed (TSNPE)`; opens on the remembered id, else from scratch. Unlabelled line: `mode · width N · amortized|narrowed (TSNPE) · created`. Accent button **Train / Load posterior**. Row: line edit (placeholder `name to save posterior as…`) + **Save**.
2. Group "Training budget": **Batches** (5000, `config.TRAINING_NUM_RUNS`), **Max rows per batch (0 = auto)** (0, `TRAINING_RUN_SIZE`); three wrapped plain-text lines (inference/base.py:144-216):
 - `{N×W:,} simulations = {N:,} batches x {W:,} rows[ (capped from {hw:,})].` / `Batches is also the (t_scale, T_obs) diversity count: every row in a batch shares one operating point, so batch COUNT is the statistics and batch WIDTH is not.` [/ `In chi mode each row costs 1+K solver passes, K up to {chi_k_pad}.`]. A bad box instead gives `Batches is blank.` / `Batches must be at least 1.` and empties the other two.
 - `Worst-case peak ~X GiB per batch (n_fine <= N, V state vars); planner budget right now ~Y GiB, so it fits in one piece|does NOT fit -- the planner will split this batch, costing wall-clock.` / `That budget is an UPPER bound: free-VRAM readings overstate what is really available by roughly the size of the desktop, so closing browsers is a bigger lever than lowering the cap.` [/ `(Estimated from hardware defaults until a config is built.)`]
 - one of: `Checkpoint status needs a config and a prior -- the prior is part of the identity.` | `Resumes a COMPLETE checkpoint ({n:,} batches) -- simulation will be skipped entirely and only the flow retrained.` | `Resumes an existing checkpoint: {done:,}/{N:,} batches already done.` | `WARNING: these settings match no checkpoint, so this starts a NEW run.` + `[checkpoint] {k} other checkpoint(s) exist and do NOT match this run: {dir} ({n} batches, differs in {key}); …` | `No checkpoint exists yet; this starts a new run.` | `Checkpointing is off (config.TRAINING_CHECKPOINT_EVERY = 0): a crash loses the run.`
3. Group "Density estimator": **Hidden features** 128, **Transforms** 8, **Learning rate** 0.001, **Early-stop patience** 20 (config.py:363-369).
4. Group "Fisher rotation": **Ensemble per perturbation** 48, **Central-difference step** 0.1, **Operating points** 8 (config.py:405-410).

Dialog **"This starts a NEW run"** (Warning icon; posterior_tab.py:218-238). Text: `This will simulate {n:,} batches from zero.` Informative: `A checkpoint that is ONE setting away already exists:` then up to three blocks `  • {dir}: {n:,} batches — differs only in {identity key}` / `      this run: {value, first 60 chars}` / `      that one: {…}`, then `If you meant to continue that run, cancel and change the setting named above — for a prior, remember that choosing it in the picker does nothing until you press "Build / Load prior".` Buttons **Start a new run anyway** (destructive role), **Cancel** (default, set by name).

Dialog **"Load a NON-AMORTIZED posterior?"** (Warning; :262-285): `'{name}' was trained by a TSNPE round.`; region/level/digest sentence; `It is valid only near that observation; …`; `If you load it:` three bullets (Validate restricts its prior; Infer refuses any other observation unless "Run on a different observation" is ticked; a further round only around its own observation, "there is no override for that one"). Buttons **Load it**, **Cancel** (default). Full text in note_claims.

**Validate** (validate_tab.py:33-46): group "Validate (SBC + TARP)": caption `Data-free calibration. Needs a posterior and the prior it was trained against.` (not word-wrapped); **Calibration datasets** 2000 (`SBC_N_CAL`); **(t_scale, T_obs) operating points** 200 (`CAL_N_SCALES`); accent **Run calibration**. Nothing else.

**Infer** (infer_tab.py:80-160): group "Infer": **Mode** combo `Simulated (cell ground truth)` | `Experimental data`; then a stack sized to its current page:
 - simulated: **Cell** = radios `Use file` / `Edit values` over a file combo (Resources/Cells/<model>) or a values grid; **T_obs (s)** 1.0 (`T_MIN_EXP_S`).
 - experimental: **Spontaneous**, **Forced** (line edit + `Browse…`, filter `Data (*.csv *.npy);;All files (*)`; Forced row hidden without forcing), **T_obs (s)** 1.0, placeholder `(build config to list drive params)` replaced by one box per drive parameter (0.0; label `<symbol> (N|Hz|rad)`, symbols for amp/freq not established).
 - χ: **Passive**, **T_obs (s)** 1.0, **Drive F₀ (N)** 1.0, **Forced probes** = rows `[path][Browse…] at [blank Hz box, max 96 px] Hz [✕, "Remove this probe"]`, buttons **+ Add probe**, **Plan probes…** (tooltip `Measure Ω₀ from the passive recording and report what is in band, and how long each probe must be recorded.`); without a χ config `(build a χ config to enable the probe table)`.
 Then checkbox **Run on a different observation** and accent **Run inference**.

**TSNPE** (tsnpe_tab.py:47-104): group "TSNPE round": caption `Restricts the PRIOR to the posterior's credible region and retrains there. It never proposes from the posterior itself. The result is NON-AMORTIZED and is marked as such in its manifest: loading it later asks first, and Infer refuses any other observation unless 'Run on a different observation' is ticked.`; **Observation** combo + refresh, summary line beneath; **HPD level** 0.999; **Directions truncated** 5 (SBI/truncate.py:71-73); accent **Run TSNPE round**. Group "Simulation budget for this round": **Batches** 5000, **Max rows per batch (0 = auto)** 0, the total and memory lines as above, then a fixed paragraph (:192-198) beginning `A TSNPE round is checkpointed under its OWN identity: the truncation region is part of it and is drawn when the round starts, so it never resumes the amortized checkpoint at these settings.` and ending `anything else simulates the full budget from zero.`; checkbox **Start a new simulation even if a cache one setting away exists**.

### How the work flows

**Gating** (screens/inference_screen.py:183-223): Posterior tab needs a built config; Validate and TSNPE need posterior + prior; Infer needs a posterior. `refresh_gates()` runs after every stage and calls every panel's `refresh_local_gates`. Train button: enabled iff config and not (from scratch with no prior), tooltip `Build or load a prior first to train a new posterior.` (posterior_tab.py:324-330). Save: iff the session holds a posterior (:334). Run TSNPE round: also needs a selected observation (tsnpe_tab.py:209-210). Run inference: blocked while a picked cell file does not fit the bounds (infer_tab.py:511-518).

**Train / Load posterior** → `_build_posterior` (posterior_tab.py:114-150):
1. `_read_inputs(is_new)`: a load returns `{}`; a train reads nine boxes through `require_at_least`/`require_positive`; first `Refusal` → yellow "Check your inputs" box plus a log line (base_panel.py:406-426).
2. Train: `_confirm_fresh_run` → `orchestrator.fresh_run_near_misses(cfg, prior, num_runs=, run_size_cap=)`; a non-empty answer opens the NEW-run dialog; Cancel aborts, "anyway" sets `new_run=True`. Load: `_accept_for_load` → `default_store().get`; non-amortized → dialog → `Accept(truncated=True)` or abort.
3. `session.reset_downstream("posterior")`, `refresh_gates()`.
4. `dispatch(orchestrator.build_posterior, cfg, session.inf_prior, entry, is_new, accept=, new_run=, provide_fig_sink=True, on_result=_on_posterior, num_runs, run_size_cap, hidden_features, num_transforms, learning_rate, stop_after_epochs, fisher_m, fisher_dz, fisher_points)`. Never passed: `name`, `note`, `max_num_epochs`, `checkpoint_every`, `resume`. No question is asked in the ordinary case.
After: `_on_posterior` puts the `LoadedPosterior` in the session, logs `Posterior ready: … Name it below to keep it.` (plus a NON-AMORTIZED warning), `refresh_gates()`. The picker is not refreshed.

**Save** → `_save_posterior` (:300-317), GUI thread: `default_store().rename("posterior", session.posterior.id, name)`; Refusal → yellow box; then picker refresh + reselect, log `Posterior named '<name>'.`

**Run calibration** → `_validate` (validate_tab.py:61-77): `dispatch(orchestrator.validate_calibration, cfg, posterior, inf_prior, provide_fig_sink=True, n_cal=, cal_n_scales=, seed=random.randrange(2**31), on_result=_on_calibration)`. After: one log line `Calibration verdict PASS|FAIL (seed N), recorded as …: TARP ATC=…, KS p=…[; informativeness … nats]`; no gate refresh; the session keeps nothing.

**Run inference** → `_infer` (infer_tab.py:460-488): `_branch()` = simulated / chi / passive / driven. Simulated → `orchestrator.simulated_inference(cfg, post, t_obs, cell=path|None, gt_values=dicts|None, prior=session.inf_prior, accept=, provide_fig_sink=True, on_result=_on_observation)`. Experimental → `RecordingSet(spont=, forced=, T_obs_s=, F0_si= | forcing_params_si=)` → `orchestrator.experimental_inference(cfg, post, rec, accept=, …)`. `accept` is `Accept(other_observation=True)` only while the box is enabled and ticked (:490-499). No `name`, `note`, `n_samples`. After: `session.observation = obs`, two log lines, `refresh_gates()`.

**Run TSNPE round** → `_round` (tsnpe_tab.py:132-162): `_read_inputs`; `default_store().load_observation(cfg, key)` on the GUI thread; `dispatch(orchestrator.tsnpe_round, cfg, posterior, inf_prior, obs, n_directions=, level=, num_runs=, run_size_cap=, new_run=<checkbox>, provide_fig_sink=True, on_result=_on_round)`; the checkbox is then unticked. After: the round's posterior REPLACES `session.posterior` (:173); warning log; `refresh_gates()`.

**Budget lines**: `_sync_budget` (inference/base.py:108-142) on every keystroke in the two boxes and on every `refresh_local_gates`; one call `orchestrator.training_preview(cfg, prior, num_runs=, run_size_cap=, hw=detect_device())` (orchestrator.py:481-564). Width = `min(cap, hw.batch_size)` (2048 on CUDA, config.py:78-79). Need = `pipeline.peak_sim_elements(width, min(N_ND_MAX=300000, len(cfg.t)), steady, n_vars, n_ch, 1)`. Have = `pipeline.sim_memory_budget_elements` = (`torch.cuda.mem_get_info` free + reusable) × 0.85, min the learned cap, min the VRAM ceiling (pipeline.py:606-640). Resume state = `training_checkpoint.peek` of this identity's directory, else `describe_siblings`.

**What blocks**: one run app-wide (base_panel.py:221-225); a run disables every panel's controls column, only Cancel stays live (:312-317, :333). All dialogs are modal `exec()`. GUI-thread work: the budget preview (store scan + CUDA query) per keystroke; the near-miss scan at the click; `cli.validate_gt_file` on each cell pick (infer_tab.py:189-207); **Plan probes…** loads and FFTs the passive recording (:312-318); the TSNPE observation re-hash; the TSNPE picker re-lists every observation manifest on every gate refresh (tsnpe_tab.py:207); `rename` retries with sleeps up to 0.3 s (store.py:864-871).

### What is remembered and what reopens fresh

Written only by `save_settings`, which `MainWindow._save_state` calls at an ACCEPTED close (main_window.py:444-456); read at the end of each `__init__`.

**Posterior**, group `inference_posterior` (posterior_tab.py:340-359): `posterior` (picker id; "" = from scratch), `num_runs`, `run_size_cap` (box TEXT; restored through `get_int`, so blank/garbage/missing → 5000 / 0). Reopen at config.py every launch: Hidden features, Transforms, Learning rate, Early-stop patience, the three Fisher boxes. Never kept: the Save name box.

**Validate**: nothing (validate_tab.py:24-29); both boxes reopen at 2000 / 200; the seed is drawn per run.

**Infer**, group `inference_infer` (infer_tab.py:528-557): `cell` (picker key), `infer_mode` (index), `sim_tobs`, `exp_tobs`, `chi_tobs`, `exp_spont`, `exp_forced`, `chi_spont`, `chi_f0_si` (text). Not kept: the `Use file`/`Edit values` choice and the grid's numbers, the driven page's drive boxes, every χ probe row (file and frequency), the other-observation tick.

**TSNPE**, group `inference_tsnpe` (tsnpe_tab.py:213-231): `observation` (id), `num_runs`, `run_size_cap` — a pair separate from the Posterior tab's. Reopen at defaults: HPD level 0.999, Directions truncated 5. Never kept: the new-simulation tick (also cleared after every dispatch, :162).

**Layout**: `layout/<PanelClassName>/splitter` and `/results`, written 1.5 s after a drag and at close (base_panel.py:196-201, :362-396).

**Surprising**
- Cell: the remembered key is only stashed at launch (infer_tab.py:545) and applied when a config is built (inference/base.py:39-40). A close without building a config writes the Nadrowski folder's first file over it (:530). Every later config build re-applies the LAUNCH-time key, discarding a cell picked this session.
- The TSNPE Observation picker keeps whatever it showed across refreshes (`refresh()` re-selects the current key, widgets/artifact_picker.py:182, :219), so after an Infer run it stays on the remembered observation; the new one becomes selected only when the picker was empty.
- After a crash or kill nothing of the session is saved: the relaunch shows the last clean close's Batches and selections.
- A remembered posterior id that no longer exists leaves the picker on `+  (from scratch)` silently (`restore_key`, artifact_picker.py:249-254).
- The χ probe table is rebuilt empty-seeded (blank Hz boxes) each launch, although rows survive config rebuilds within a session (infer_tab.py:256-287).

### How the look is produced

**Where the look comes from.** None of the four tab files or inference/base.py sets a colour, font or stylesheet. Everything is the global QSS in design.py (`build_qss`, :191-326) keyed on class names and three hooks: dynamic property `accent=true` on the primary button (posterior_tab.py:52, validate_tab.py:43, infer_tab.py:157, tsnpe_tab.py:75), objectName `iconButton` (refresh and remove buttons: widgets/artifact_picker.py:32, :166; inference/rows.py:43) and `helpBadge` (widgets/help_badge.py:22). Tokens: `LIGHT`/`DARK` dicts (design.py:50-89), `TYPE` ramp, `CTL_H` 32 / `CTL_H_SM` 28, radii 4/8, all scaled by `ui_scale()`. QGroupBox is styled as a card with a 600-weight title (design.py:264-271). Derived lines are plain word-wrapped `QLabel`s in PlainText (`_derived_label`, inference/base.py:63-75) with no caption/warning style: the `WARNING: … starts a NEW run` line looks like every other line. The Validate caption is an unwrapped QLabel (validate_tab.py:35).

**Hard-coded geometry**: refresh button fixed 32 px (artifact_picker.py:34, :168); probe Hz box max 96 px, remove button max 32 px (rows.py:41, :45); zero contents margins on composite rows; `FIELD_MIN_W` 96, `PATH_FIELD_MIN_W` 220, `LABEL_MAX_W` 190, `CONTROLS_MIN_W` 360, `DEFAULT_SPLIT` (420, 880), `DEFAULT_RESULTS_SPLIT` (520, 200) (design.py:28-36); panel margin 6 (base_panel.py:190).

**Figures**: no matplotlib figure is built in these files. A stage hands figures to an injected `fig_sink`; `_png_fig_sink` renders PNG at dpi 110 on the worker thread via `visualizers.save_figure` and pickles the figure for pop-out (base_panel.py:31-63). Figure sizes are fixed in the stages (e.g. SBC grids 16 × 3.4·rows in, corner up to 24 in; orchestrator.py:2013-2018, :2238).

**What resists a restyle or relayout**
- Each tab's widgets are built inline in one `__init__` (posterior 35-112, infer 73-161, tsnpe 43-111, validate 31-46), with signal wiring and `restore_settings` order interleaved (comments at posterior_tab.py:79, tsnpe_tab.py:106-111).
- The Infer tab depends on QFormLayout row calls: `setRowVisible`, `removeRow`, anchor labels (infer_tab.py:216, :270-282, :364-373).
- "Locked during a run" = the single `controls` widget (base_panel.py:333); Cancel sits outside it.
- Tests reach widgets by attribute name (`post_name`, `budget_ckpt`, `num_runs`, `new_run`, `other_obs`, `cal_n`, `cal_scales`, `cell_picker`, `obs_picker`, `obs_line`, `_ask_new_run`, `_ask_load_non_amortized`, `_save_posterior`, `_validate`): about 196 hits in tests/test_nav_and_gating.py and tests/test_settings_persistence.py. Settings group and key names are pinned too.
- Splitter state is keyed by panel class name (base_panel.py:362-365).

**fields.py** (232 lines): one dict `CONTROL` (:55-187) with 109 keys, pinned equal to `core.refusals.FIELDS` (109) by tests/test_nav_and_gating.py:1520. Three entry shapes: 54 `(place, label)` tuples (place = a tab title or one of `SCREENS = {"Artifacts", "Model Builder"}`, or a tuple of places); 12 whole sentences (consents, name, note, lists, `_FIXED`); 43 `None`. `label(key)` (:190-201) returns a tuple entry's label and raises KeyError otherwise; the tabs build their rows from it. `fix_sentence(key)` (:221-232) gives `Set it in the '<label>' box on the <place> tab.`, the sentence itself, or "". The `None` entries: six stage arguments the window never exposes — `checkpoint_every` :60, `resume` :61, `num_posterior_samples` :71, `n_samples` :72, `max_num_epochs` :86, `device` :125 — plus five FDT settings (:161), eleven diagnostics knobs (:179-180) and twenty-one probe-check knobs (:182-186). Unused imports remain in the tab files (`icons`, `PathField`, `with_badge`, `BoundsGrid`).

### What a redesign must keep, or ask before changing

- Every knob travels to the stage as an ARGUMENT, never a config write: orchestrator binds config constants at
  import, so assigning config.X is a silent no-op (posterior_tab.py:144-150; orchestrator.py:873-878; project
  rule). The VRAM ceiling is the one deliberate exception, read live (config_tab.py:128-148).
- Only selections and the training budget are remembered; every science knob (network, learning rate, patience,
  Fisher, HPD, directions, calibration sizes) reopens at config.py each launch, and stale keys are ignored
  (posterior_tab.py:340-345; tsnpe_tab.py:213-217; validate_tab.py:24-29). Reason on record: a restored value
  silently trained a different network and a stale setting cost a ~5-day run (config_tab.py:137-143).
  Remembering the network size, or any other change here, is the owner's decision.
- A consent is never persisted and is per run: the TSNPE new-simulation tick is cleared after each dispatch
  (tsnpe_tab.py:162) and never saved (:213-217); 'Run on a different observation' is cleared whenever the
  session's posterior changes and is read only while enabled (infer_tab.py:490-499, :519-526).
- A seed, a record name and a note are never remembered (project rule). If a name or note box is added before
  the run it must not be saved, and the name sentences in fields.py:96-97 and :102-103 list every box a name or
  note is typed in, so they need the new box added.
- Both questions come BEFORE the session is reset, so Cancel leaves it untouched (posterior_tab.py:139-143).
- Cancel is the default BY NAME on both dialogs; ordering by role once made Enter start the run
  (posterior_tab.py:233-236, :282-283).
- The near-miss question and the stage's refusal share one detector (orchestrator.fresh_run_near_misses /
  training_checkpoint.near_miss_siblings) and stay deliberately narrow: exactly one differing key, never
  'truncation' alone (posterior_tab.py:183-203; SBI/training_checkpoint.py:329-351). Widening it is the owner's
  call; the code records three past accidental restarts as the reason for the modal.
- The budget lines only FORMAT orchestrator.training_preview; no second copy of the planner's cost model or of
  the cache resolution may live in the GUI (inference/base.py:118-126; pipeline.py:566-569;
  orchestrator.py:486-496).
- A derived status line must never raise: tests put a bare object() on session fields and a raise would take the
  tab down through refresh_gates (inference/base.py:122-126; infer_tab.py:519-522;
  screens/inference_screen.py:119-122).
- A blank or out-of-rule box is SAID, never clamped or defaulted, on the live line and at the click, through
  core.refusals' rules; the stage re-runs the same rules (inference/base.py:111-116; posterior_tab.py:163-167;
  validate_tab.py:51-54; tsnpe_tab.py:116-121). Boxes are read with value_or_none(), not value().
- A control is named in ONE place: rows are built from fields.label(key); CONTROL's key set must equal
  core.refusals.FIELDS; places must be real tab titles; sentences and the None set are pinned verbatim
  (fields.py:1-34; tests/test_nav_and_gating.py:1520-1634, :1955-1990). Renaming a tab or row, or adding a
  window control for resume, checkpoint_every, max_num_epochs, device, n_samples or num_posterior_samples, means
  editing CONTROL and those pins.
- Dialogs must be instance QMessageBox objects shown with exec(), never the static helpers: the test guard
  patches the instance method and a static call stalls the suite (widgets/refusal_box.py:29-31). The two
  question dialogs are separate methods so tests can answer them (posterior_tab.py:218-219, :262-263).
- A refusal is the yellow 'Check your inputs' box with no traceback; a bug is the red 'Error' box with Details;
  a Refusal raised in the worker must cross the thread as the exception object (base_panel.py:406-457;
  worker.py:146-150).
- One run at a time app-wide, and the whole controls column of EVERY panel is locked during it; only Cancel,
  which sits outside `controls`, stays live (base_panel.py:102-106, :221-225, :312-335). Anything a redesign
  wants usable during a run must live outside `controls`.
- Figures are rendered to PNG on the worker thread; the UI never paints a live canvas created off-thread, which
  deadlocks (base_panel.py:31-43).
- TSNPE: the tab must never propose from the posterior (tsnpe_tab.py:24-29); the observation is loaded and
  re-hashed on the GUI thread before dispatch so a mismatch is a dialog now (:144-156); the near-miss consent is
  a checkbox because the region is drawn inside the worker (:96-101); the checkpoint line must not promise a
  resume it cannot resolve (:179-198); on_result must install the round's posterior or hours of work are
  discarded (:164-171).
- A non-amortized posterior is never loaded without asking (posterior_tab.py:240-260; store.py:1386-1400); a
  further round around another observation has no override (orchestrator.py:2622-2635).
- Infer: experimental mode with a χ config MUST show the χ page, or a wrong-width conditioning vector is built
  silently (infer_tab.py:220-229). A probe row is ONE widget holding file and frequency so a deletion cannot
  mispair them; the frequency box starts blank; rows survive config rebuilds; 'Plan probes…' fills only blank
  boxes and never a typed zero (inference/rows.py:22-35; infer_tab.py:48-57, :231-235, :256-268, :331-338).
- A picked cell is validated against the bounds on the GUI thread at pick time and Run stays disabled until it
  fits (infer_tab.py:189-207, :511-518).
- Validate is gated on posterior AND inf_prior, never on force_prior, which is None for every no-forcing model
  (validate_tab.py:21-22; screens/inference_screen.py:187-192).
- The budget boxes are saved as TEXT so a box left blank reopens at config.py's default rather than 0
  (posterior_tab.py:340-345; settings.py:40-50).
- Derived lines must stay word-wrapped plain text: generated strings (directory names, key names) otherwise
  widen the whole controls column (inference/base.py:63-75).
- The HELP table is shared by several tabs and also surfaced in Settings -> Help through the inference_tabs
  module path (help_text.py:1-3); the panels are imported through panels/inference_tabs.py
  (screens/inference_screen.py:16-17).
- Long runs get the card to themselves and opening a window touches the card (project rule;
  inference/base.py:77-82, :138-139). If the redesign keeps a launch-time memory line, that is a decision for
  the owner.

## How a run is started, shown, cancelled and closed

### What the user sees

**Frame shared by all ten panels** (`core/gui/panels/base_panel.py:120-203`). Outer margins 6 (:190). Horizontal QSplitter, non-collapsible, default (420, 880) (`design.py:29`):
- left: QScrollArea, min width 360, no max (:133-136), holding `controls` (QVBoxLayout, top-aligned) that each subclass fills;
- right: vertical QSplitter `results_split` (non-collapsible, stretch 3:1, default (520, 200)): figure stack over log pane (:166-172); BELOW it, outside the splitter, one row: progress pane (stretch) + Cancel (:153-157, :178).

**Figure stack** (`widgets/figure_stack.py`). QTabWidget, tabs closable and movable; empty at launch, no placeholder; one tab per figure, the newest takes the front (:109-110); never cleared between runs (no caller of `clear_all`, :148). Each tab: QPushButton "Pop out" top right (tooltip "Open an interactive, zoomable copy in a new window", or "Open a zoomable copy in a new window" for a disk PNG, :87-89) over a scroll area showing the PNG scaled down to width, never up (:19-46). Failure text "(could not render figure: {title})" (:80).
Pop-outs (`widgets/figure_window.py`) are separate parentless windows: interactive, 900x650, matplotlib's own toolbar over a live canvas (:57-79); or image viewer, 900x700, buttons "Fit", "100%", "Zoom in", "Zoom out", "Save As…" (dialog "Save figure as", filter "PNG image (*.png)"), wheel zoom, drag pan (:112-171); "(could not load figure: {title})" (:129).

**Log pane** (`widgets/log_pane.py`, 35 lines). Read-only QPlainTextEdit, wraps at widget width, 5,000-block cap (:15). A line is its text with "⚠ " (warning) or "✖ " (error) in front (:8). No timestamp, colour, filter, find, clear, copy or save control; only Qt's built-in selection, Ctrl+C and context menu. App font, not fixed pitch. Jumps to the end on every append (:21-34).

**Progress pane** (`widgets/progress_pane.py:167-245`), hidden unless a run is live (:245); three lines; there are NO per-stage bars:
1. Label "Solver Performance: {meter}  ({rate})": one "+" per decade of steps/s, "·" under 10, "—" idle; rate "142.4k it/s" / "59 it/s" / "2.5 it/s" / "idle" (:72-91, :321). Tooltip "The SDE solver is not running right now." or "SDE solver: N integration steps/sec.\nOne '+' per order of magnitude." (:322-326). Beside it an 88x18 sparkline, tooltip "Recent SDE solver throughput (log it/s)" (:108-110).
2. One QProgressBar, text "%p%", min height 18, indeterminate until a bar with total > 1 reports (:196-200, :366-372); tooltip = the whole live nest, indented, or "Nothing is reporting progress right now." (:391-397). Right of it a 150 px label: a braille spinner stepping every 100 ms, replaced after 45 s without output by "⏳ no output for {N}s", frozen (:69, :211, :329-342).
3. Caption: "{bar description}  —  {tqdm stats}", e.g. "Generating training data  —  1902/5000 [05:12<13:41, …]" (tqdm's own elapsed<remaining). When nothing reports a percentage: the deepest row whose total is not 1; in a fit that is "Training neural network. Epochs trained: N" (:376-389).

**Cancel**: QPushButton "Cancel", neutral style, hidden unless a run is live; after one click, disabled and "Cancelling…" (`base_panel.py:150-152, :295-296`).

**Header run slot** (shell): button "Running: {tab title} — m:ss" (h:mm:ss from an hour), tooltip "{section} → {tab}: go to it"; a click goes to the running tab; hidden when the run ends (`main_window.py:351-392`; `screens/nav_shell.py:25-38, :75-88`). The running tab and its Home tile carry a mark.

**Boxes (stock QMessageBox, modal)**
- Yellow: Warning icon, title "Check your inputs", text = the refusal's sentence, second line "Set it in the '<label>' box on <place>." when the field is known, one OK (`widgets/refusal_box.py:33-42`; `fields.py:221-232`).
- Red: Critical icon, title "Error", text = `str(exception)`, traceback behind Qt's Details button, plus "A cancel was requested before the run failed." when it applies (`base_panel.py:28, :446-457`).
- Close while running (`main_window.py:424-435`): title "A task is still running", text "A task is still running.", then "Cancelling stops it at its next checkpoint (up to ~1 min during training). Quitting anyway closes the window now, but the task keeps running in the background until it finishes on its own."; buttons "Cancel task & quit" (default), "Quit anyway", "Don't quit".

**Fixed log lines**
- "A task is already running (in this tab|in another tab); please wait for it to finish." (`base_panel.py:223-224`)
- "Cancelling — will stop at the next checkpoint (up to ~1 min during training)." (:299-300)
- "Run cancelled." (:263)
- "The run finished before the cancel could take effect; nothing was stopped, and its result is complete." (`worker.py:14-15`)
- "Cancelled. There may be a failure that was still in flight -- it cannot be told apart here from one the run had already recovered from. Its traceback follows, in case it helps." (`worker.py:186-189`)
- "Another task already owns the console; this run's output is not captured." (`streams.py:273`)
- "[progress parser disabled after an internal error]" (`streams.py:216`)

### How the work flows

**Start.** A panel's click handler checks inputs on the GUI thread (a `Refusal` goes to `_refusal`), asks its own consent dialogs, then calls `dispatch(fn, *args, provide_fig_sink=, provide_stream=, on_chunk=, watch_dir=, on_result=, on_finished=, **kwargs)` (`base_panel.py:206-289`). `dispatch` asks nothing. It refuses with a log line if `BasePanel._running` (:221-225); builds one `NewPngWatcher` per watched folder (:231-236); a fresh `CancelToken` (:238-239); `Worker(fn, *args, cancel=token, **kwargs)` (:240), `setAutoDelete(False)`, kept in `_workers` (:244-245); injects `fig_sink` (:247) or `emit_chunk`/`should_stop` (:253-254); `_set_busy(True)`; `QThreadPool.globalInstance().start(worker)` (:289). Arguments reach the stage exactly as given, e.g. `orchestrator.build_posterior(cfg, prior, entry, is_new, accept=, new_run=, fig_sink=, **knobs)` (`posterior_tab.py:148-150`).

**Worker** (`worker.py:104-215`): runs `fn` inside `redirect_streams`; emits exactly one of `result(payload)`, `error(exception, traceback, cancel_noted)`, `cancelled()`, then always `finished()`. Other signals: `log`, `log_batch`, `rows`, `figure`, `chunk` (:68-80).

**Output** (`streams.py:261-321`): sys.stdout becomes a stream at "info", sys.stderr one at "warning"; `warnings.showwarning` logs at warning; a handler on logger `core` passes the message text at the record's level (:251-258). Each stream feeds `vt.StreamRouter`, which classifies every tqdm write: a bar frame becomes a row (description, percent, total, rate with s/it inverted, stats); a blank paint retires it; a bare "\r" then text (sbi's epoch counter) is a row with no percentage, logged only when a newline ends it; a finished, replaced or leftover bar's last frame becomes a log line; the rest are log lines (`vt.py:135-335`). A daemon pump sends `log_batch` then a `rows` snapshot at 15 Hz, only when something changed (`streams.py:104-173`).

**Cancel checkpoints**: every non-empty stdout/stderr write and every `core` record on the worker thread, unless `runs.cancel_is_deferred()` (inside a commit section or while an exception is being handled) (`streams.py:204-205, :252-253`; `runs.py:171-199`). Not: `warnings.warn`, other threads.

**While live** (`_set_busy`, :302-321): Cancel shown, pane begun, `controls.setEnabled(False)` on EVERY panel, `RUN_STATE.changed(panel)` to the shell. Still live: navigation and tabs, figure tabs, Pop out, log scrolling and selection, splitters, appearance menu, Artifacts reading. Artifacts writes and model-builder saves refuse on `BasePanel._running` (`artifact_screen.py:580-590`).

**Figures**: `fig_sink(title, fig)` on the worker pickles and renders PNG (dpi 110), emits `figure` and a tab appears as each stage draws (`base_panel.py:44-62`). Watched folders are polled every 1.2 s with a 1 s settle and one last scan at the end (`plot_watcher.py:27-28, :61-84`).

**End**: `result` runs the panel's `on_result` (e.g. `_on_posterior`: session posterior, "Posterior ready…", `refresh_gates`; `posterior_tab.py:287-298`); `finished` runs `_finished` (:267-285): watchers stopped, `_set_busy(False)` (Cancel hidden, pane cleared and hidden, all controls back plus `refresh_local_gates`, shell slot cleared), payload released, `on_finished`. No finish line, duration, sound or taskbar alert.

**Failure**: `error` reaches `_on_error` (:428-457): a Refusal gets the yellow box; anything else "✖ message" in the pane and the modal red box.

**Cancel click** (:291-300): sets the token, disables the button, writes the line. Honoured at the next checkpoint: about a second in sweeps and generation (solver bar repaints every 0.1 s, `sdeint.py:36`), then the rescue save; in the fit only at sbi's end-of-epoch print (`sbi/inference/trainers/base.py:997-998, :1274`), first epoch included: 61-63 minutes at full size. Preparation between generation and fit, the SBC block and the out-of-memory waits: not established.

**Blocks the UI**: only the modal boxes. A run never blocks the GUI thread.

### What is remembered and what reopens fresh

**Remembered (these files): layout only.** Two keys per panel, `layout/<PanelClassName>/splitter` and `layout/<PanelClassName>/results` (QSplitter.saveState blobs; `base_panel.py:362-370`).
- Read once, in `BasePanel.__init__` through `restore_layout()` (:203, :372-389). A blob `restoreState` rejects falls back to (420, 880) / (520, 200).
- Written 1.5 s after a drag of either splitter, with `sync()` (:196-201, :391-396). Nothing else writes them: the docstrings say `MainWindow._save_state` does too (:359, :368), but it calls only `save_settings` (`main_window.py:447-456`), so a drag in the last 1.5 s before quitting is lost.
- The key is the class name: renaming a panel class drops its saved split without a word.

`save_settings` / `restore_settings` are no-ops in the base (:398-404). Subclasses own the selections, written only when a close is accepted (`main_window.py:444-456`).

**Deliberately or by construction not kept**
- The cancel token: new per run (:238), never persisted, never cleared once set.
- The progress pane: `end()` clears and hides it at every finish, failure included (`progress_pane.py:254-265`); the solver rate and sparkline are re-baselined per run (:286-294).
- The elapsed clock: cleared when the run ends (`main_window.py:364-367`); the duration is shown nowhere afterwards.
- The log pane's text: memory only, 5,000 lines, gone at exit.
- Figure tabs, the current tab, pop-out windows, their size and zoom.

**Surprising**
- Within a session the log pane and the figure tabs are NOT reset between runs: a second run's lines follow the first's with no separator, and each new figure takes the front tab (`figure_stack.py:109-110`).
- The pane and Cancel reset while the red box is still open, because `finished` is handled inside that box's event loop (`worker.py:205, :213`; `base_panel.py:457`).
- `core.progress.SOLVER.steps` is process-wide and never reset (`core/progress.py:24-27`).
- `BasePanel._running`, `_active_cancel` and `_instances` are class attributes: one run for the whole process (:106-112).

### How the look is produced

**No styling of their own.** None of the ten files sets a style sheet, an object name or a dynamic property (grep over `core/gui`). Their look is the app-wide QSS and palette from `design.py` (`_QSS` :191-326, applied at `theming.py:112`): QPushButton (Cancel, Pop out, viewer buttons: neutral, min height 32, radius 4), QPlainTextEdit (log: 1 px border, radius 4, padding 2px 8px), QProgressBar (borderless, radius 3, accent chunk, centred text), QTabBar (pivot with accent underline), QScrollBar, QToolTip, `QToolButton#navRunning` (header slot). Cancel has no accent or destructive variant. Boxes are stock QMessageBox.

**Hard-coded here**
- `base_panel.py`: margins 6 (:190); PNG at dpi 110, tight bbox (:59); 1500 ms layout debounce (:198).
- `progress_pane.py`: margins (0,2,0,2), spacing 2 and 8 (:190, :217, :232-233); bar min height 18 (:200); spinner width 150 (:211); sparkline 88x18, pad 2, pens 1.0 and 1.5 (:108, :129-153); 100 ms tick (:51); glyphs "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", "+", "·", "—", "⏳" (:50, :79-81, :337).
- `log_pane.py`: cap 5000 (:15); "⚠ ", "✖ " (:8); no font set, so the UI font (the Artifacts pane uses a fixed font, `artifact_screen.py:225`).
- `figure_window.py`: 900x650, 900x700 (:70, :124); zoom 0.05-40, step 1.15 (:29-31).
- Tokens from `design.py`: `CONTROLS_MIN_W` 360, `DEFAULT_SPLIT`, `DEFAULT_RESULTS_SPLIT` (:28-33).

**Custom painting**: only `_Sparkline.paintEvent` (`progress_pane.py:121-164`), reading `QPalette.Mid` and `QPalette.Highlight`; the QSS does not reach it (`design.py:8-11`).

**matplotlib**: no live canvas in a panel. Figures are PNG pixmaps rendered on the worker by `visualizers.save_figure`, in the theme current when the figure was built (`base_panel.py:54-59`); tabs already shown do not re-theme. The interactive pop-out embeds `FigureCanvasQTAgg` and matplotlib's own `NavigationToolbar2QT` (`figure_window.py:64-79`), unstyled by the app.

**Resists a relayout**
- `BasePanel.__init__` builds the whole frame inline (:120-203); ten subclasses append to `controls_layout`.
- The progress row sits outside the splitter on purpose (:161-165); `insert_result_widget` is the one seam into the results column (:343-353); the results split's child count varies (:385-389).
- A panel has no title of its own; MainWindow maps panels to tab labels (`main_window.py:154-169`).
- The caption's `QSizePolicy.Ignored`, its never-hidden state and the spinner's fixed width exist to stop relayout jitter on every tick (`progress_pane.py:204-212, :224-229`).

**Tests-facing names and strings**: `btn_cancel`, `log_pane`, `figure_stack`, `progress_pane`, `controls_scroll`; `caption`, `overall`, `spinner`, `solver_label`, `_rate_at`, `_beat_at`, `STALL_S`, `SOLVER_IDLE_S`, `plus_meter`; `png_ready`; "Pop out", "Cancelling…", "Run cancelled.", "Check your inputs", "no output for" (`tests/test_worker_dispatch.py:228-242, :469`; `tests/test_vt_progress.py:211-212, :350-360, :376-381, :401-419, :481-490, :547-549`; `tests/test_figures.py:153`; `tests/test_settings_persistence.py:659`; `tests/test_nav_and_gating.py:228`).

### What a redesign must keep, or ask before changing

- ONE RUN AT A TIME FOR THE WHOLE PROCESS. `redirect_streams` swaps sys.stdout/stderr process-wide, so
  `BasePanel._running` is class-level and `_REDIRECT` backs it (base_panel.py:102-106; streams.py:31-35); the
  run log assumes a single attached buffer (core/runs.py:100-103). A queue or a second concurrent run needs that
  reworked, not a new button.
- LOCK EVERY PANEL'S CONTROLS WHILE A RUN IS LIVE (base_panel.py:312-317, :323-333): a model combo changed
  mid-run repoints pickers and re-seeds inputs under the running worker. Navigation stays free on purpose
  (main_window.py:399-403). Ask the owner before unlocking anything.
- THE ARTIFACTS SCREEN AND THE MODEL BUILDER ARE DELIBERATELY NOT BasePanels, so a run does not grey out the log
  being read (base_panel.py:91-93; refusal_box.py:5-8). Their writes check `BasePanel._running` themselves
  (artifact_screen.py:580-590).
- NEVER PAINT A WORKER-BUILT FIGURE ON A LIVE CANVAS: it deadlocks on matplotlib's lock with no traceback.
  Figures cross as PNG bytes plus a pickle (base_panel.py:31-43; figure_stack.py:1-5); a pop-out unpickles on
  the GUI thread and must keep the Gcf detach, because `Worker.run` closes all pyplot figures after every run
  (figure_window.py:37-45; worker.py:166-168). The app stays on the Agg backend (core/gui/__main__.py:9-11).
- THE REDIRECTED STREAM MUST NOT GAIN `fileno()` OR `encoding`: tqdm would change its frame format and the
  parser would break (streams.py:179-182; vt.py:35-38).
- KEEP THE PUMP: tqdm redraws are not rate-limited by tqdm, so events are coalesced at 15 Hz, emitted outside
  the lock, and the final publish runs on the pump thread to keep log order (streams.py:12-15, :145-147,
  :164-173). A new live view (a log file, a loss curve) should hang off the pump or the signals, not write to
  widgets from the worker.
- CANCEL IS COOPERATIVE AND ITS RULES ARE SAFETY PROPERTIES: `WorkerCancelled` is a BaseException
  (streams.py:38-45); one raise, then quiet, on the worker thread only (:48-80); deferred inside commit sections
  and while an exception is unwinding (core/runs.py:171-228); never print or log between a checkpoint save's
  steps (streams.py:233-238). Any new per-run handler or file writer must not raise and must not become a
  checkpoint inside those sections. The Cancel wording, a confirmation, a way to withdraw it, the close dialog's
  default and the fate of 'Quit anyway' are the owner's decisions.
- THE WORKER MUST OUTLIVE ITS QUEUED SIGNALS AND THEN LET GO OF ITS PAYLOAD: `setAutoDelete(False)`, the
  `_workers` set, and the release in `_finished` (base_panel.py:241-245, :273-281); pinned by
  tests/test_worker_dispatch.py:44 and :75.
- A REFUSAL IS THE YELLOW BOX, A BUG THE RED ONE (base_panel.py:406-457; refusal_box.py). Title 'Check your
  inputs' with the Warning icon is asserted in dozens of tests (e.g. tests/test_worker_dispatch.py:469, :531;
  tests/test_nav_and_gating.py:1681). The traceback must stay out of the body
  (tests/test_worker_dispatch.py:302).
- EVERY BOX IS AN INSTANCE SHOWN WITH `.exec()`, NEVER A STATIC `QMessageBox.warning(...)`: the suite's guard
  patches the instance method only and a static call stalls the offscreen run (refusal_box.py:29-32;
  main_window.py:258-266; tests/test_worker_dispatch.py:335-358).
- THE CORE NEVER NAMES A CONTROL. The 'where to fix it' line comes from core/gui/fields.py by field key
  (base_panel.py:410-414; refusal_box.py:10-15; fields.py:221-232). A control that is renamed or moved is
  renamed there, once; tab titles are read back verbatim by tests (screens/nav_shell.py:45-47).
- `RUN_STATE` IS THE ONLY BROADCAST OF 'SOMETHING IS RUNNING', emitted last in `_set_busy` (base_panel.py:82-84,
  :318-321). Its listener must touch only its own window, because discarded test windows still receive it
  (main_window.py:355-359).
- PROGRESS PANE INVARIANTS: the solver bar is excluded from the overall bar by `config.SOLVER_BAR_DESC`
  (vt.py:58-69; progress_pane.py:277-283); the rate comes from `core.progress.SOLVER`, never from bar text
  (progress_pane.py:28-34; core/progress.py:17-27, one writer, never reset); the caption fallback is what shows
  a fit at all (progress_pane.py:16-22; tests/test_vt_progress.py:534-549); `end()` clears everything
  (tests/test_vt_progress.py:211-212). The solver bar must stay enabled in the window: its redraws are the most
  frequent cancel checkpoint (core/Solvers/sdeint.py:53-56).
- PALETTE ROLES `Mid` AND `Highlight` ARE READ BY THE CUSTOM-PAINTED SPARKLINE (progress_pane.py:132, :152;
  design.py:8-11), and the QSS must use named-class selectors only, never `*` or bare `QWidget`, or it paints
  behind custom-painted widgets (design.py:188-190).
- LABELS THAT CHANGE EVERY TICK MUST NOT RE-LAY-OUT THE PANE: the caption is `QSizePolicy.Ignored` and never
  hidden, the spinner has a fixed width (progress_pane.py:204-212, :224-229). The same holds for any new clock,
  loss or epoch readout.
- LAYOUT PERSISTENCE IS SEPARATE FROM SELECTIONS ON PURPOSE, because nine of ten panels override `save_settings`
  without calling super (base_panel.py:355-361); the key is the panel's class name (:362-365). The controls
  column has no maximum width (tests/test_settings_persistence.py:659-660). `insert_result_widget` is the
  supported way into the results column (base_panel.py:343-353).
- THE LOG PANE ONLY APPENDS COMPLETED LINES; progress never goes into it (log_pane.py:1-5;
  base_panel.py:142-143). The `core` logger's level is set once in core/runs.py:46-47 and the window's handler
  is installed after the streams and removed first (streams.py:291-294, :302-303).
- PROJECT RULES THAT BIND NEW UI TEXT AND STATE (CLAUDE.md): a consent is never remembered; science knobs reopen
  at config.py; no shippable file may cite the working record or its labels; a long run has the card to itself,
  so nothing new may open a second window or touch the card beside one.
- TESTS READ THESE NAMES DIRECTLY: `btn_cancel`, `log_pane`, `figure_stack`, `progress_pane`, `controls_scroll`,
  the pane's `caption`, `overall`, `spinner`, `solver_label`, `STALL_S`, `SOLVER_IDLE_S`, `plus_meter`,
  `png_ready`, the strings 'Pop out', 'Cancelling…', 'Run cancelled.' (tests/test_worker_dispatch.py:228-242;
  tests/test_vt_progress.py:350-419, :481-490; tests/test_figures.py:153; tests/test_nav_and_gating.py:228). A
  rename moves the tests with it; do not edit sources while a suite runs.

## Simulate, fdt, the sweep study, the reduction map, the model builder

### What the user sees

Paths are relative to C:\Users\J\PycharmProjects\PRISM\.

**Shell.** Window "PRISM", 1300×820 clipped to the screen, min 900×600 (core/gui/main_window.py:88-95). Home = greeting + five accent tiles "Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate", "Artifacts" (screens/home_screen.py:16,53-63). A section = heading label over a QTabWidget (screens/section_screen.py:22-38). The model builder has no tile: gear → "Full settings…" → group "User-defined models" → "Open model builder", or per-model "Edit"/"Delete" (screens/settings_screen.py:104-111,165-177).

**Every panel** (panels/base_panel.py:128-191): horizontal splitter; left = scroll area with the controls (min 360 px, opens 420); right = vertical splitter [figure tabs | log pane] (520/200) over a fixed progress row ending in "Cancel" (hidden until a run). Every number box is a QLineEdit with a validator (min 96 px); there is no spin box (widgets/labeled_inputs.py:15-65). Row labels carry a "?" badge (tooltip; click pins it).

**Reduction Map › "NWK → Hopf reduction map"**, group of the same title (reduction_panel.py:49-63): label "Model is fixed to NADROWSKI (the reduction is Nadrowski-specific)."; "Cell" file combo over Resources/Cells/nadrowski + rescan icon (tooltip "Rescan the folder"), default first file; "F0 (ND forcing amplitude)" 0.05 (panel literal); accent "Run reduction map".

**FDT Analysis › "FDT analysis"**, group "FDT analysis" (fdt_panel.py:119-177), top to bottom: "Model" combo (BP, NADROWSKI, HOPF + user models; NADROWSKI); "Cell"; "n_freqs" 60; "M_ensemble" 256; "freqs / batch" 1; "F0 (ND forcing amplitude)" 0.05; "Seed" blank (= draw one); "Record name" (placeholder "name for this run's record (optional)…"); "Note" ("a note to keep with it (optional)…"); "Saved run" store combo of finished single-cell records (unnamed shown "(unnamed <id>)"; rescan tooltip "Rescan the artifact store"); an unlabelled wrapped details block ("Written <UTC ISO time> · seed N", "Cell: <path>", raw key=value settings, "Notice: …"; record_view.py:63-102); checkbox "Skip sanity checks" (off); checkbox "Proceed to the production sweep after sanity" (on; greyed while the first is ticked); accent "Run FDT analysis". Defaults are panel literals equal to cli.make_fdt_config's (core/cli.py:380-382). Group "Compare saved runs" (180-213): "Runs to compare" list (max height 110) over "Add selected" / "Remove" / "Clear"; "Comparison" combo cells | repeats | renormalise; "Normalisation constant" blank, enabled only for renormalise; "Comparison name" ("name for the comparison's record (optional)…"); "Comparison note"; plain button "Compare saved runs".

**› "Sweep study cross-validation"**, group "FDT parameter-sweep study" (crossval_panel.py:110-173): label "Model is fixed to NADROWSKI."; "Cell"; "Cell values" read-only "S = 0.1234,  T_a/T = 1.5000" or "(could not read cell: …)"; "Preset" exploratory | production (exploratory); "S grid  (T_a/T = 1)" and "T_a/T grid  (S = 0)", each "min" "max" "n" — ends from the cell (0…S, 1…T_a/T, ascending), n from the preset (8 / 12); "n_freqs" (30 / 60) and "M_ensemble" (256) from the preset (cli.py:455-460); "freqs / batch" 1; "F0 (ND forcing amplitude)" 0.05; "Seed" blank; "Record name" ("base name for this study's two records (optional)…"); "Note" ("a note to keep with both (optional)…"); "Saved sweep"; details block; accent "Run sweep study". Group "Compare saved sweeps" (179-199): "Sweeps to compare" list; "Slice at" blank (= middle of the shared range); "Comparison name"; "Comparison note"; "Compare saved sweeps".

**Simulate › "Live simulation"**, group "Live simulation" (simulate_panel.py:69-96): "Model" (NADROWSKI); "Cell"; "T_obs (s)" 1.0 (config.T_MIN_EXP_S); "Steps / frame" 2000; "Max FPS" 30; label "Streaming holds the app single-task until it finishes or you Cancel."; accent "Start streaming"; "Save video…" (tooltip "Save the last run as an animation (.mp4 or .gif)", disabled until something is recorded). Right: a pyqtgraph view replaces the figure tabs — trace "Hair-bundle displacement" (last 2000 samples, y "x (<cell unit>)") over heatmap "Top-down hair bundle" (widgets/live_hair_bundle.py:59-98).

**Model builder** (scrolling page, no log pane; screens/model_builder_screen.py:251-330): heading "Model builder"; group "Definition": "name"; "variables" (placeholder "x, y") + "Set variables". One group "d<var>/dt" per variable: "drift" ("e.g. mu*x - x^3"), "D" ("0"), "init" (0.0), "forcing" combo None | Sinusoidal | Step | Triangular | Exponential, then that kind's fields (amp 0, freq 1, phase 0, offset 0, t0 0, tau 1; radios "Grow (+t/tau)" / "Decay (-t/tau)", Decay on). Group "Parameter values": "Detect parameters"; per parameter "value" 1.0, checkbox "auto" (on), "min"/"max" (greyed: value ∓ max(|value|,1)), "box" linear | log (tooltip, no badge). Group "Display scales": "x_scale (nm)" 10.0, "t_scale (s)" 0.01. Status line. Sticky bar outside the scroll: "Validate", accent "Save model", "Back to settings".

**Dialogs.** Yellow: title "Check your inputs", the refusal sentence, then "Set it in the '<label>' box on the <tab or tabs> tab.", OK (widgets/refusal_box.py:33-42). Red: title "Error", message, Details = traceback, optionally "A cancel was requested before the run failed." (base_panel.py:448-457). "Delete model": "Delete the user-defined model '<name>'?" / "This removes its definition and its generated Bounds/Cells/Units files.", Yes/No, No default (main_window.py:288-294). Video: "Save animation", filter "MP4 video (*.mp4);;GIF (*.gif)" (simulate_panel.py:139-140).

**A cancelled FDT/sweep record.** Its own tab never lists it: both pickers show finished records only (widgets/artifact_picker.py:197-198), so record_view's "This run did not finish." line (record_view.py:100-101) cannot appear in the window. The pane shows "Run cancelled." (then "FDT run finished." on the FDT tab). On Artifacts, kind fdt: columns Name, Created, Study, Points, Finished = "no", Note (widgets/artifact_table.py:38,164-166); the delete prompt adds "This … record is UNFINISHED: … There is no resume for these runs -- deleting it means measuring again from the start." (screens/artifact_screen.py:133-149).

### How the work flows

**Common rail.** `BasePanel.dispatch` (base_panel.py:206-289) runs one callable on QThreadPool; stdout/stderr, warnings and `core` log records reach that panel's log pane and tqdm the progress pane through a 15 Hz pump (streams.py:28,261-300). ONE task app-wide (`_running`, 102-106, because stream redirection is process-wide): `_set_busy` disables the `controls` column of all ten BasePanels, shows Cancel on the running panel only, and emits RUN_STATE → header button "Running: <tab title> — m:ss" (click jumps to the tab) plus "●" on the Home tile and tab (base_panel.py:302-321; main_window.py:351-410). Right column, navigation, Settings and the builder's form stay live. Cancel: one click sets the token, button reads "Cancelling…", pane gets "Cancelling — will stop at the next checkpoint (up to ~1 min during training)." (291-300); the worker raises at its next print/bar/log record; then "Run cancelled.".

**FDT.** "Run FDT analysis" → `_run` (fdt_panel.py:265-324). Before: no cell → pane line "Select a cell file first."; then `cli.make_fdt_config(model, state_dep_drift, cell, n_freqs, ensemble_M, freqs_per_batch, F0, seed)`, `require_note`, `default_store().create("fdt", cfg, name, note)` — the name is claimed at the click; any Refusal → yellow box. No question is asked; the two checkboxes are the answers. Then `dispatch(run_fdt(cfg, skip_sanity, confirm_production, writer, seed), watch_dir=<record>/figures)`; PNGs appear as tabs as files land (1.2 s poll, plot_watcher.py:27). On result: "FDT record written: <name|(unnamed)> [<id>]."; picker re-listed and moved onto it. On finish, any outcome: "FDT run finished."; the viewer's tabs close. Model change → cell folder repointed, Run enabled per `registry.fdt_support`, reason to the pane (256-263). Runs on CPU (cli.py:420).
"Compare saved runs" → `compare(mode, ids, name, note[, prefactor])` with a figure sink; "Comparison record written: …". "Add selected" copies the Saved-run picker's item into the list.
A user pick in "Saved run" fills the details block and opens that record's PNGs; programmatic changes refresh the text only (record_view.py:132-139).

**Sweep study.** "Run sweep study" → `_run` (crossval_panel.py:267-333): four knobs `require_given`, `cli.make_param_sweep_config(cell, preset, preset_name, s_spec, t_spec, seed, …)`, note, two `create` calls named `<base>-s` / `<base>-temp`, then `run_param_study_cli(cfg, s_grid, t_grid, writers, seed)` with two figure watchers (CPU, cli.py:544). Result: one "Sweep record written: …" per finished sweep; picker moves to the last. Cell change refills both grids' ends; preset change refills n_freqs, M_ensemble and both n. "Compare saved sweeps" → `compare("sweeps", ids, name, note[, at])`.

**Reduction.** "Run reduction map" → `require_positive(f0)`, `require_file(cell)`, `cli.make_reduction_config(cell, F0)`, `dispatch(run_reduction_map, cfg, watch_dir=Artifacts/reduction)`; "Reduction map complete." (reduction_panel.py:65-94). Writes `sweep_<stamp>.parquet` and `reduction_sweep_<stamp>.png` loose (Reduction/sweep.py:186-190; plots.py:100-104).

**Simulate.** "Start streaming" → t_obs > 0, frame_steps ≥ 1, fps > 0, `build_stream_config(model, cell)` (bounds resolved from the cell; forced to CPU), view and recording reset, `dispatch(run_simulation_stream, cfg, t_obs, frame_steps, fps, provide_stream=True)`; each chunk → `live_view.push` and appended to `_record`; "Simulation complete." (simulate_panel.py:101-133). "Save video…" → file dialog, ffmpeg check for .mp4, warning above 600 frames, `dispatch(export_animation, series, path, <live-view geometry>, video_fps)`; a cancelled export deletes its partial file (simulate_export.py:151-160).

**Model builder** (GUI thread, no dispatch). "Set variables" rebuilds the per-variable groups keeping typed content; "Detect parameters" parses and rebuilds parameter rows; "Validate" = name rule, numeric refusals, parse, 100-step batch-1 integration with forcing off; "Save model" = validate, name-exists check, `model_store.save_user_model` (Bounds/Cells `default.txt`, Units `units.txt`, then `Resources/Models/<NAME>.json`), `registry.load_user_models()`; MainWindow then refills every `model_combo` and the Settings list (model_builder_screen.py:333-534; main_window.py:305-329). The model then appears in the Model combos of FDT, Simulate and the Config tab; FDT gates it by `fdt_support`, inference by `is_sbi_user_model` (registry.py:70-112). Validate/Save refuse while any task runs.

**Cross-screen.** Artifacts `store_changed` → every StorePicker re-listed (main_window.py:331-344). Selections are written only on an accepted close (444-456).

### What is remembered and what reopens fresh

**Written** only by `MainWindow._save_state` on an accepted close (main_window.py:444,447-456) → each panel's `save_settings`; **read** once, in each panel's `__init__`, by `restore_settings`. Number boxes are stored as their TEXT (settings.py:96-107).

- `fdt/` model, cell, record (the Saved-run selection), n_freqs, ensemble_m, freqs_per_batch, f0 (fdt_panel.py:338-364).
- `crossval/` preset, cell, record, f0, freqs_per_batch, s_points, t_points (crossval_panel.py:351-377).
- `simulate/` model, cell, tobs, frame_steps, fps (simulate_panel.py:177-196).
- `reduction/` cell, f0 (reduction_panel.py:96-106).
- `layout/<PanelClass>/splitter` and `/results`: written 1.5 s after a splitter drag, read in `BasePanel.__init__` (base_panel.py:196-203,367-396).
- Model builder: nothing in QSettings; the form keeps its content in memory until "Open model builder" (`reset()`) or "Edit" (`load_existing`) replaces it.

**Deliberately reopen at default:** Seed (blank), Record name and Note on both FDT tabs; "Skip sanity checks" (off) and "Proceed…" (on) as consents; both compare groups whole (list, mode, constant, slice point, comparison name and note) (fdt_panel.py:95-106; crossval_panel.py:88-96; compare_list.py:9-11). CrossVal n_freqs and M_ensemble (re-derived from the preset) and both grids' min/max (re-derived from the cell).

**Surprising:**
- Seed, name and note are not cleared after a run within a session (fdt_panel.py:326-336): a typed seed makes every later run that session a repeat; a typed name is refused on the next click.
- CrossVal n_freqs/M_ensemble are editable yet forgotten; the remembered point counts are overwritten whenever the Preset combo changes (crossval_panel.py:260-265).
- A box left blank at close is restored blank, not at its default (settings.py:96-102); the Posterior tab falls back to config instead (posterior_tab.py:357-358) — likely, not run.
- A remembered cell/record that no longer exists leaves the first entry selected and the next close overwrites the key (artifact_picker.py:79-86,249-254).
- A crash or kill loses every change since the last clean close.
- The Simulate recording, all figure tabs and the log text live in memory only.
- `_save_state` does not save layout although base_panel.py:359,368 say it does; a drag within 1.5 s of closing is lost.

### How the look is produced

**Source of the look.** One app-wide QSS + QPalette built from `core/gui/design.py` tokens (LIGHT/DARK 50-89; radius 4/8; control heights 32/28; type ramp via `QLabel[type=…]`), applied by `theming.Appearance` (theming.py:112). None of my files calls `setStyleSheet`. Hooks used: `setProperty("accent", True)` on the CTAs (fdt_panel.py:151; crossval_panel.py:144; reduction_panel.py:55; simulate_panel.py:81; model_builder_screen.py:292), `type="heading"` (model_builder_screen.py:252), object names `iconButton`, `helpBadge`; `actionBar` (model_builder_screen.py:322) has no rule in design.py. Group boxes are the "cards" (design.py:264-271).

**Hard-coded sizes.** Compare list max height 110 (compare_list.py:26); rescan button width 32 (artifact_picker.py:34,168); panel margin 6, splits (420,880)/(520,200), controls min 360, field min 96, label max 190 (base_panel.py:190; design.py:28-35); action bar margins 8/6 (model_builder_screen.py:324); spinner 150 px, sparkline 88×18 (progress_pane.py:108,211).

**Painting.** Live view = pyqtgraph `GraphicsLayoutWidget`: background, axes, curve and titles from `design.tokens(dark)["base"/"text"]`, re-applied on `theme_changed`; heatmap fixed "inferno", levels 0–1, 96×~250 grid, pen width 1 (live_hair_bundle.py:59-131). Sparkline is custom-painted from palette Mid/Highlight (progress_pane.py:121-164). Matplotlib figures are never live canvases: PNG pixmaps in closable, movable tabs with "Pop out" (figure_stack.py:49-110), coloured by global rcParams at build time (mpl_theme.py:16-33) — figures saved into a record keep the theme they were built in and open tabs never recolour. Video export is pinned white, 640×480 (simulate_export.py:91-116).

**Construction that resists change.** Each panel builds one QGroupBox + QFormLayout inline in `_build_controls` (+ `_build_compare`); the builder is a 90-line `__init__` with rows created and destroyed dynamically. BasePanel fixes the skeleton; Simulate inserts its view at results-splitter index 0 after `restore_layout` already ran (simulate_panel.py:63-64; base_panel.py:203) — whether its split is ever restored: not established. Row labels and tab titles are pinned by `fields.CONTROL`, `labels.GUI_HTML` (exact-string keys) and tests/test_nav_and_gating.py; widget attribute names are read by six test files (209 hits); the exporter and tests read the live view's private geometry (simulate_panel.py:168-170).

**Consistency.** Shared: BasePanel rails, make_form, FloatField/IntField, badges, yellow/red boxes, pickers. Different: name-then-run with Seed and Note on the FDT tabs vs run-then-"Save" rename, no note, no seed on Prior/Posterior; no record at all for Reduction (loose files) and Simulate; a saved-record viewer with figures only on the FDT tabs; pre-run cost lines only on Posterior/TSNPE; labels from `fields.label()` on FDT tabs, literals on Reduction, mixed on Simulate; lower-case code labels in the builder ("drift", "D", "init"); one screen named four ways ("Sweep study cross-validation" tab, "FDT parameter-sweep study" group, "Cross-validation (sweep study)" in Help, "Run sweep study"); completion lines differ ("FDT run finished." on any outcome, "Reduction map complete." / "Simulation complete." on success, none for a sweep); heading + single tab + group repeat one title on Reduction and Simulate; run figures arrive as disk PNGs (zoom-only pop-out), comparison figures as pickled figures (interactive pop-out) on the same tab.

### What a redesign must keep, or ask before changing

- One task at a time app-wide, because redirect_streams swaps sys.stdout/stderr for the whole process
  (base_panel.py:102-106, 221-225; streams.py:31-35, 272-276). Anything meant to run or be browsed beside a long
  training needs the owner's decision and a change to this rule, not just a layout change.
- Anything that must stay usable during a run has to live outside a BasePanel's `controls` widget or outside a
  BasePanel: every instance's controls column is disabled (base_panel.py:312-317, 333); the Artifacts screen and
  the model builder are plain QWidgets on purpose (base_panel.py:91-93; model_builder_screen.py:571-580;
  main_window.py:119-122).
- A matplotlib figure built on the worker must never be painted by a live canvas (deadlock on matplotlib's
  global lock): figures cross as PNG bytes plus an optional pickle, or as disk PNGs (base_panel.py:31-43;
  figure_stack.py:1-12). pyqtgraph on the GUI thread is the allowed live surface (live_hair_bundle.py:4-8).
- The front end CREATES an fdt writer (mints the id, claims the name) and the STAGE enters it on the worker
  thread, because log.txt comes from a thread-local run log; the body dict is updated in place, never replaced
  (fdt_panel.py:290-302, 310-315; crossval_panel.py:299-322; fdt_pipeline.py:257-263). Name and note are judged
  at the click, before anything is spent.
- Never persisted, by the project's rule: Seed, record name, note, the two consent checkboxes, and both compare
  groups (fdt_panel.py:95-106, 346-347, 362-363; crossval_panel.py:88-96, 356; compare_list.py:9-11). CrossVal
  n_freqs/M_ensemble come from the preset and the grid ends from the cell (crossval_panel.py:89-91, 370-372).
  Ask the owner before remembering any of them.
- Restore order is load-bearing: model combo first, an explicit _on_model_changed, then picker keys
  (fdt_panel.py:352-358; simulate_panel.py:188-192; artifact_picker.py:88-100); on the sweep tab preset, then
  cell, then the free knobs (crossval_panel.py:365-376). The saved-run viewer is built after the restore so a
  launch opens no figures (fdt_panel.py:112-116).
- Every number box is read with value_or_none()/number_or_blank at the click, never value() (a blank reads as
  0); no clamps or silent defaults; every refusal is a Refusal with a field key shown in the yellow box, and a
  control is named only through core/gui/fields.py (labeled_inputs.py:31-58; fdt_panel.py:278-289;
  crossval_panel.py:276-293; simulate_panel.py:108-113; reduction_panel.py:71-73; base_panel.py:406-426).
- Tab titles and row labels are pinned: fields.CONTROL holds (tab title exactly as shown, row label),
  labels.GUI_HTML is keyed by the exact label strings, and tests/test_nav_and_gating.py reads both back from the
  built window (fields.py:14-18, 33-34, 45; core/Helpers/labels.py:52-61; nav_shell.py:41-53). Renaming a tab or
  a row means changing all three.
- Widget attribute names are read directly by tests (209 hits in tests/test_nav_and_gating.py,
  test_settings_persistence.py, test_user_models.py, test_simulate.py, test_artifact_browser.py,
  test_refusals.py), and the QSettings groups and keys (fdt, crossval, simulate, reduction) are the stored
  format (fdt_panel.py:338-364 etc.).
- Message boxes must be instance dialogs shown with exec(), never QMessageBox statics: the test guard patches
  the instance method only and a static stalls the suite (main_window.py:258-266; refusal_box.py:29-31).
  simulate_panel.py:139-140 already uses the static QFileDialog.getSaveFileName.
- Code that runs during panel construction or inside a combo slot must not raise: CrossValPanel._on_cell_changed
  is called from __init__ and a raise there stops the launch (crossval_panel.py:240-248); record_view feeds
  read-only labels and must never raise (record_view.py:11-14).
- Labels fed by unbounded generated text must be word-wrapped plain text, or they widen the controls column and
  bring back the permanent horizontal scrollbar (crossval_panel.py:120-124; record_view.py:31-42). Forms go
  through make_form (fields grow, long rows wrap) and the controls column has a 360 px floor and no cap
  (widgets/forms.py:16-22; design.py:28; base_panel.py:136-140).
- StorePicker lists finished records only and the fdt kind is split by study per tab; saved-run figures open
  only on a user's pick (QComboBox.activated), never on a programmatic index change, and only the viewer's own
  tabs are ever closed (artifact_picker.py:192-200; fdt_panel.py:142; crossval_panel.py:141;
  record_view.py:132-151).
- Simulate runs batch 1 on CPU with every tensor on one device; the frame grid's m+1 points, the bare
  `sim.sde.force =` write and the stop check BETWEEN frames are load-bearing (simulate_runner.py:75, 155-163,
  194-196, 240). The exporter and tests read the live view's private geometry, and hb_center/gaussian_field are
  shared so view and video cannot drift (simulate_panel.py:168-170; simulate_export.py:26, 72-81;
  live_hair_bundle.py:37-43). The exported video is pinned white whatever the theme (simulate_export.py:91-116).
- Model builder: save writes the Bounds/Cells/Units files first and the JSON last (the commit point);
  Validate/Save refuse while any task runs; built-in models must keep positions 0-2 in VALID_MODELS; model
  combos re-fire their hook only when the selection really changed; _ParamRow.spec() and its constructor are
  positionally matched; the action bar is outside the scroll area because Save was once reported missing
  (model_store.py:384-399; model_builder_screen.py:316-330, 367-369, 394-398, 510-512; registry.py:126-136;
  main_window.py:310-329).
- Settings → Help is assembled from each panel module's first docstring paragraph and its HELP dict
  (settings_screen.py:23-29, 184-205): those texts are user-facing, and as shippable files they may cite no
  working-record label (tests/test_source_hygiene.py, per the project rules).
- Palette roles Mid and Highlight feed custom painting and must stay correct; the QSS uses named-class selectors
  only, never `*` or bare QWidget, because of the pyqtgraph view, the sparkline and the animation overlays
  (design.py:8-11, 188-190).
- The reduction map writes to config.artifacts_root()/"reduction" in three places that must agree
  (reduction_panel.py:89; Reduction/sweep.py:186; Reduction/plots.py:23); 'reduction' is not a store kind and
  core/Reduction is out of scope per the project instructions — ask the owner before making it a record.
- The header run slot and the tile/tab markers are driven only by RUN_STATE from _set_busy, and MainWindow maps
  panel → (section, tab title) once from the built tab widgets (base_panel.py:82-84, 318-321;
  main_window.py:162-169, 351-410); screens are built once in MainWindow.__init__ and the window always opens on
  Home (main_window.py:100-117, 183).

## The artifacts screen

### What the user sees

**Artifacts screen** (Home's fifth section; plain QWidget, one QVBoxLayout, `core/gui/screens/artifact_screen.py:257-264`). Top to bottom:

1. Label **"Artifacts"** (heading, :197).
2. Row: label **"Kind"**; combo (stretches) with, in order, "Priors", "Simulation caches (training rows)", "Posteriors", "Observations", "Calibrations", "Inferences", "Diagnostics", "FDT measurements, sweeps and comparisons" (:40-51, :200-210); opens on Priors unless remembered; button **"Refresh"** (no tooltip).
3. Horizontal QSplitter, children not collapsible, right side takes the stretch (:215-217, :251).
   - Left: table (flat QTreeWidget, alternating rows, one row selectable, not editable, header click sorts; `widgets/artifact_table.py:239-250`). Columns "Name", "Created", the kind's own, "Note". Kind's own: simulation "Progress", "Finished"; posterior "Mode", "Width", "Amortized"; observation "Mode", "Width"; diagnostic "Variant"; fdt "Study", "Points", "Finished"; prior, calibration, inference none (:20-39). Cells: Name = name or "(unnamed <id>)"; Created = the manifest's raw ISO string (UTC, `store.py:614`); Progress "3/4 batches" / "4/4 batches · 96 rows"; Finished "yes"/"no"; Amortized "amortized"/"narrowed (TSNPE)"; Points "10/12 · 2 failed" (:87-166). A folder with no usable manifest shows folder name, blank, the reason in column 3, and always sorts last (:152-154, :169-210). Default sort Created, newest first (:46); Progress/Points/Width sort by leading number (:53). Columns resized to contents on every fill (:287-288). No filter, search, context menu, multi-select or tooltips.
   - Right: read-only QPlainTextEdit, no wrap, system fixed-pitch font (:222-225). Under it, right-aligned: **"Save…"** (tooltip "Write what is shown here to a text file"; disabled when the pane is empty) and **"Lineage report…"** (tooltip "Write this artifact and its parents, back through the chain, to a text file") (:227-243).
4. Group box **"Actions"**, one row (:484-516): label "Note"; line edit, placeholder "one line, at most 200 characters; empty clears it"; **"Set"**; **"Delete…"**; **"Sweep this kind…"** (tooltip "Remove this kind's directories that have no manifest at all, and any loose file inside its folder"); **"Sweep all kinds…"** (tooltip "Remove every kind's directories that have no manifest at all, any loose file inside a kind's folder, and a legacy directory an older build wrote beside them"). Note box, Set, Delete…, Lineage report… are enabled only on a selected row that has a manifest (:543-547); the sweeps always.
5. Status label, word-wrapped, "⚠ " prefix on trouble (:254, :934): "Nothing here yet." or "N row(s): X complete, Y incomplete." (:335-338).

**Detail pane text** (`core/artifacts/report.py:158-174`, then `artifact_screen.py:389-401`): `<kind> <name|(unnamed)>  [<id>]`; a rule of 72 "="; `schema:`, `kind:`, `id:`, `name:`, `created:`, `note:`; sections "PRISM", "Environment", "Inputs" (path, "sha256 <hash>" or "NOT READ AT COMMIT"), "Parents", "Payloads" (file name + sha256), "Figures" (file names only), "Fingerprints", "Knobs", "Body"; an empty section reads "(none)". Mappings in sorted key order, two-space indent; a list of records as "[0]", "[1]" blocks; floats as `repr`; "true"/"false"; None "(none)"; up to 12 scalars inline, more as "[a, b, c, ... (N entries)]"; a list of lists as "[13 x 13 numbers]" (:40-91). Body keys per kind: `manifest.py:28-56`. Then "── the run's records (log.txt) ──" and the log verbatim (`HH:MM:SS level` stamps), or "written outside a run", "the run recorded nothing", or for a cache "a training cache keeps no log: it is written batch by batch across resumes and shared by every posterior that names it"; a long log is preceded by "(only the last 1048576 bytes are shown; the file is longer)". A manifest-less folder: "<kind> — an incomplete directory, which nothing can load." with `folder:`, `path:`, `reason:` (:376-379). A real record's folder path is not shown.

**Dialogs.**
- Delete (Warning; title "Delete artifact"; Yes/No, default **No**; :704-711): "Delete <kind> <label>?"; informative "id <id>"; for an unfinished cache "This training cache is UNFINISHED: N committed batch(es). Deleting it throws those batches away and a later run starts from zero. (Only the batch count is known while a cache is running: the rows are recorded when the cache finishes.)", or the fdt sentence (:128-151); then "This removes <path> and everything in it, and cannot be undone."
- Delete refused for dependents: yellow "Check your inputs" box, OK only: "Refusing to delete <kind> <label> [<id>]: N artifact(s) depend on it.", one line per dependent ("…: it names this <kind> as a parent." or "…: a training cache was generated against this prior and its rows are meaningless without it."), "Delete those first. Nothing here can orphan them."; below it "Select an artifact in the list on the Artifacts screen." (:667-670; `fields.py:101`).
- Sweep (Warning; title "Remove what no artifact accounts for"; Yes/No, default **No**; :867-883): "Remove N items that no artifact accounts for?", one line per item, then a paragraph beginning "A directory that carries a manifest.json of any kind is never touched by this."
- File dialogs "Save what is shown" and "Save lineage report", filter "Text file (*.txt)".

**`panels/record_view.py`** (FDT tab row "Saved run", sweep tab row "Saved sweep"; `fdt_panel.py:168-172`, `crossval_panel.py:165-169`): under the picker a wrapped plain-text label: "Written <created> · seed N"; "Cell: <path>"; "key=value, …"; "B of N probe frequencies came back blank."; "D of P operating points of the <sweep> sweep measured (F failed)."; "Notice: …"; "This run did not finish." (:64-101). The record's PNGs open as tabs in that panel's figure area.

### How the work flows

All store calls run synchronously on the GUI thread; the screen has no worker, no timer and no connection to the run-state signal (`artifact_screen.py`, whole file).

**Listing.** `showEvent` → `refresh()` on every visit (:279-284); "Refresh" → `refresh()` (:205); kind change → `refresh()` (:272). `refresh` → `_relist`: `store.list(kind)` (one directory scan, one manifest read per record, `store.py:681-735`), `table.set_rows`, `_apply_sort`, `_on_selection_changed`, status line (:308-339). A read failure empties the table and says "Could not read the <kind> artifacts: …" (:320-328). Every refresh clears the selection and the pane (:331-333; `artifact_table.py:220-223`).

**Selection.** `table.selection_changed` → `_on_selection_changed` → `_detail_text(s)`: `store.get(kind, id)` + `render_manifest(m)` + `store.read_log(kind, id, max_bytes=1<<20)` (:380-389). Each of the two store calls re-parses every manifest of the kind (`_find` → `_entries`, `store.py:737-741`), and `read_log` reads the whole file before slicing the tail (`store.py:788-795`). Then `_sync_actions` sets enabled states and fills the note box, only when the selected artifact changed (:542-553).

**Set** → `_refuse_while_running("setting a note")` → `_selected("annotate")` → `require_note("note", text)` (`core/refusals.py:370-387`) → `store.set_note(kind, id, note)` → box rewritten with the trimmed text → `_after_change("Set the note on <kind> <label>." | "Cleared the note on …")` (:610-641).

**Delete…** → `_refuse_while_running("deleting an artifact")` → `_selected("delete")` → `store.dependents(kind, id)`; with dependents the refusal box and stop; otherwise the Yes/No dialog → `store.delete(kind, id)` (never `force`) → `_after_change("Deleted <kind> <label>.")`; a No leaves "<kind> <label> was not deleted." (:672-729). The removal is a blocking `rmtree` (`store.py:937-947`).

**Sweep this kind… / Sweep all kinds…** → `_refuse_while_running("removing leftovers")` → `_incomplete(store, kind|None)` + `_loose_and_legacy` (:731-800); nothing found → status "Nothing to remove: …"; otherwise the dialog → `store.sweep_incomplete([(kind, dir_name)])`, `store.remove_loose(kind, name)` per file, `store.remove_legacy(name)` per directory → `_after_change("Removed N of M …")` (:887-930). The list confirmed is the list removed.

**Save…** → `QFileDialog.getSaveFileName` → writes the pane's text, UTF-8, LF (:404-427). **Lineage report…** → `render_lineage(store, kind, id)` → file dialog → write (:429-475). Neither checks for a live run.

**After a change.** `_after_change` (:555-578): `refresh()`, `_sync_actions()`, `store_changed.emit()`, then the action's sentence, only if the re-list could read the store. `MainWindow` connects `store_changed` to `_refresh_store_pickers` (`main_window.py:124`), which calls `refresh()` on every `StorePicker` found by type in every panel (:331-344); a picker whose selection was deleted falls back to whatever is current (`artifact_picker.py:181-219`, :249-254). A note change triggers the same re-listing.

**While a run is live in this window.** `BasePanel._running` is a class-level bool of this process (`base_panel.py:106`, :304). Set, Delete… and the sweeps stay enabled; a click puts "⚠ A task is running -- wait for it to finish before setting a note." / "…before deleting an artifact." / "…before removing leftovers." on the status line, with no dialog (:580-590, :617, :682, :829). Kind, Refresh, selection, Save… and Lineage report… keep working. The screen is not a `BasePanel`, so the app-wide lock never greys it (:3-7; `main_window.py:119-122`).

**Other direction.** Nothing tells the screen that a run wrote a record or advanced a cache; it re-reads only on a visit, a Refresh, a kind change or its own change.

**record_view.** `picker.combo.currentIndexChanged` → rewrite the line (`store.get("fdt", ref)`) and queue one `QTimer.singleShot(0)` settle that closes the viewer's own tabs if the selection moved; `picker.combo.activated` (a user's pick only) → `_open(ref)`: close own tabs, `store.path("fdt", ref)`, `stack.add_png(title, png)` per PNG not already shown; nothing opens while the panel's run is busy; the panel calls `close_figures` when a run finishes (`record_view.py:157-227`).

### What is remembered and what reopens fresh

**Remembered** (QSettings group `artifacts`, `artifact_screen.py:942-976`):

- `artifacts/kind`: the kind key ("prior", "simulation", …). Read in `__init__` before the combo's signal is connected (:271-272); written on an accepted close by `MainWindow._save_state`, which calls the screen by name because it is not a panel (`main_window.py:447-456`). An unknown kind is ignored and Priors stands (:973-975).
- `artifacts/sort_col`, `artifacts/sort_order`: plain ints stored as strings (0 ascending, 1 descending). Same read and write points. Missing or unreadable → `DEFAULT_SORT = (1, 1)`, Created newest first (`artifact_table.py:46`).

Like every panel's selections, these are written only on an accepted close (`main_window.py:444-456`), so a crash or a killed process keeps the previous values.

**Deliberately not remembered:** the selected artifact (a deleted id would dangle; :12-14), which is also dropped on every refresh; the note draft beyond the session.

**Not remembered, by omission:** splitter position, column widths (recomputed every fill), the pane's scroll position, the file dialogs' folder (the suggested name is a bare file name, :415-416, :463-465).

**Surprising:**

- The sort is remembered as a column INDEX, and the columns differ per kind (index 2 is Mode for posteriors, Progress for caches, Note for priors). `_relist` captures the table's sort before each rebuild (:311-317) and `_apply_sort` skips an out-of-range column (:345-347), so viewing a narrower kind replaces a remembered sort on a wide kind's later column with the default (traced by reading, not run).
- A window closed without ever opening the screen saves the sort it read, not the default (:957).
- The note box keeps its text, disabled, through the no-selection state a re-list passes through, and is rewritten only when a different artifact is selected (:548-553).

**record_view.py** remembers nothing itself. It is built after the panel's `restore_settings`, fills the line for the restored record and opens no figures at launch (`record_view.py:164`; `fdt_panel.py:111-116`).

### How the look is produced

**Style sources.** No style sheet, object name or colour is set in `artifact_screen.py`, `artifact_table.py` or `record_view.py`. The look is the global QSS in `core/gui/design.py`: `QLabel[type="heading"]` (16 px, weight 600; :195, the one dynamic property the screen sets, `artifact_screen.py:198`); the `QTreeView` / `QTreeView::item` / `QHeaderView::section` block written for this table, the app's only item view (:243-261); `QGroupBox` as a card (:264-271); `QPushButton` (:199-207); and `QLineEdit, QPlainTextEdit` sharing one rule (:220-226), so the detail pane is styled as a text input.

**Hard-coded.** Pane font `QFontDatabase.systemFont(QFontDatabase.FixedFont)` (:225), no token. Two `addSpacing(SPACE[2])` = 12 px in the actions row (:511, :513); `setContentsMargins(0, 0, 0, 0)` on the pane's column (:247); every other margin is Qt's default. Glyphs in strings: "…" on four buttons, "⚠ " (:934), "── the run's records (log.txt) ──" (:66), "—" (:376, :872), " · " in cells (`artifact_table.py:109`, :138). The manifest text is laid out for a fixed-pitch, unwrapped pane: 72-character rule (`report.py:26`), two-space indent (:24), 100-character knob lines in the lineage report (:33).

**No custom painting and no matplotlib here.** record_view shows PNG files already on disk through the panel's figure stack (`record_view.py:220-226`); they were saved at dpi 110 when the run made them (`store.py:407`), so a theme change cannot recolour them.

**What resists a restyle or relayout.**

- The whole screen is built inline in `__init__` (:185-272) plus `_build_actions` (:478-516); the actions box is placed by position, `insertWidget(count() - 1, …)` (:264); the pane's button row is a kept layout that the lineage button is appended to (:235-243).
- The detail pane is one string (`_detail_text`, :363-402), and Save… writes exactly that string. The same renderer produces `python -m core artifacts show` and the lineage file (`report.py:9-10`), so a structured or graphical pane needs either a second renderer or a change both front ends share.
- Table behaviour hangs on column TITLES: `NUMERIC_COLUMNS = {"Progress", "Points", "Width"}` is read off the header text in `_Row.__lt__` (`artifact_table.py:53`, :203-210). Per-kind columns are a closed table pinned by tests against `KIND_DIRS` (:22-39) and mirrored in lower case by the command line (`core/tool/browse.py:71-78`).
- A QTreeWidget used flat with a custom item class; sort state as plain ints because `int(Qt.SortOrder)` raises on the pinned PySide6 (:299-306); the sort indicator must be set before `sortItems` (:346-352).
- Tests-facing names: `kind_combo`, `btn_refresh`, `table`, `split`, `detail`, `btn_save`, `btn_lineage`, `detail_actions`, `note_edit`, `btn_note`, `btn_delete`, `btn_sweep`, `btn_sweep_all`, `status`, `_noted`, `_sort`, `_relisting` (136 uses in `tests/test_artifact_browser.py`), and the module constants and helpers (`KIND_LABELS`, `LOG_MAX_BYTES`, `_delete_prompt`, `_unfinished_note`, `MAX_INLINE`, `columns_for`, `cells_for`, `record_summary`; 99 uses across four suites).
- Fix sentences name this screen's controls: "the list on the Artifacts screen", "the Note box on the Artifacts screen" (`core/gui/fields.py:101-103`).
- Minimum window 900 × 600 (`main_window.py:95`): the actions row carries a label, a line edit and four buttons on one line.

### What a redesign must keep, or ask before changing

- Keep the screen a plain QWidget, never a BasePanel: a BasePanel is disabled app-wide while any run is live,
  and reading the store must never be blocked (core/gui/screens/artifact_screen.py:3-7;
  core/gui/main_window.py:119-122).
- Every action that changes the store must pass the live-run check, and nothing that only reads may
  (artifact_screen.py:580-590; calls at :617, :682, :829; :439). The check sees this process only; a
  cross-process lock was ruled out of scope in the store (core/artifacts/store.py:52-58), so adding one is an
  engine change to raise with the owner.
- No store read while the window is being built: the screen lists on show, not in __init__, because a directory
  scan at launch once lost the taskbar icon (artifact_screen.py:268-272, :279-284).
- The selected artifact is never remembered, across refreshes or launches: a remembered id that was since
  deleted is a dangling selection (artifact_screen.py:12-14; core/gui/widgets/artifact_table.py:220-223).
  Remembered: the kind and the sort only (:942-976). Ask the owner before remembering more.
- Text only, no figure rendering, no open-the-folder button and no jump into a stage tab were considered and
  DECLINED on 2026-09-17 (artifact_screen.py:16-18). The redesign prompt puts 'reading a record's numbers and
  figures' on the table, so ask the owner before adding any of the three.
- Delete: dependents are read before anything is asked; a record with dependents gets a refusal naming each one,
  not a confirmation; no front end offers force; the confirmation defaults to No (artifact_screen.py:672-716;
  store.py:918-933).
- Sweep: the list shown in the confirmation is the list removed (no re-scan); only folders with no manifest at
  all are offered; folders with an unreadable or foreign manifest are named and left alone; 'all kinds' is an
  explicit None, never a falsy kind; the legacy directory is offered by the all-kinds sweep only
  (artifact_screen.py:731-762, :802-835, :887-909; store.py:46-51, :1134-1166).
- Dialogs are QMessageBox INSTANCES shown with exec(), never the static calls: the suites' no-modal guard
  patches the instance method only and a static call stalls the offscreen run (artifact_screen.py:705-706;
  core/gui/widgets/refusal_box.py:29-31).
- The note is refused, never clamped: no setMaxLength on the box; the rule is core.refusals.require_note, shared
  with the command line; after a Set the box shows the trimmed text that was stored (artifact_screen.py:492-494,
  :622-624, :637-639; core/refusals.py:367-387). A note is never remembered between launches (CLAUDE.md).
- A draft note must survive a re-list: the box is rewritten only when a different artifact is selected, and
  selection changes during a table rebuild are ignored (artifact_screen.py:286-306, :531-553). This is a trap
  already hit once.
- After any change: re-list, tell the window, then say what happened, and only if the re-list could read the
  store, so 'nothing here' and 'could not look' stay distinct (artifact_screen.py:555-578). MainWindow refreshes
  every StorePicker found by TYPE (main_window.py:331-344).
- One renderer for both front ends, deterministic: the pane's text is what `python -m core artifacts show`
  prints, and the lineage file is byte for byte what `artifacts summary --out` writes
  (core/artifacts/report.py:9-15; artifact_screen.py:433-437). Changing the 12-item cut (report.py:30) changes
  the command line's output and the tests that pin MAX_INLINE; it is not a window-only change.
- A report is a file the operator names, never a kind in the store; files are written UTF-8 with LF
  (artifact_screen.py:405-409, :423, :433-435, :471; report.py:3-10).
- Refusals carry a field key and the window names the control from core/gui/fields.py (:101-103 for 'artifact'
  and 'note'); core messages name no control. The screen rewrites the store's dependents sentence because the
  store's ends 'pass force=True', which no front end offers (artifact_screen.py:643-670).
- Table rules: rows without a manifest sort last under every column in both directions
  (artifact_table.py:169-210); sort state is plain ints, read with `.value`, because int(Qt.SortOrder) raises on
  the pinned PySide6 6.9.3 (:299-306); the indicator is set before sortItems (:346-352); sorting is off while
  rows are inserted (:259-262); a cell is built from the listing's Summary alone, with no second read, and the
  pickers use the same facts so two surfaces cannot disagree about whether a cache is finished (:4-8).
- Per-kind columns are closed against the store's kinds by tests and mirrored by the command line's listing
  (artifact_table.py:22-39; core/tool/browse.py:71-78); numeric sorting is keyed on the column titles
  'Progress', 'Points', 'Width' (artifact_table.py:53).
- Pickers elsewhere list FINISHED records only, not merely manifest-bearing ones, so a run still going or half
  written is never offered as a result (core/gui/widgets/artifact_picker.py:192-198).
- The 'folder name disagrees with the manifest' notice must not fire for a simulation cache, whose folder is
  legitimately the bare digest (artifact_screen.py:383-388).
- record_view: nothing in it may raise, since a formatter that throws inside a combo slot takes the panel and
  the launch down (core/gui/panels/record_view.py:11-14); a null is never printed (:55-59); figures open only on
  a user's pick (`activated`), never on a programmatic index change, and never at launch (:132-139); the viewer
  closes only the tabs it opened (:141-147); it does not use the store's load path, which re-hashes a sweep's
  data on every keystroke (:6-9).
- From the redesign prompt (docs/superpowers/prompts/2026-10-05-ui-ux-redesign.md:58-68): nothing may delete,
  rename or rewrite what is in the owner's Artifacts/ without asking, and no walkthrough row may either. Set
  note rewrites a record's manifest (store.py:884-888), so rows that exercise Set, Delete or the sweeps must run
  against a scratch store.

## What the stages accept against what the window offers

### Findings

All citations are repo-relative to `C:\Users\J\PycharmProjects\PRISM\` (read-only; nothing run). Key files: `C:\Users\J\PycharmProjects\PRISM\core\orchestrator.py`, `...\core\tool\stages.py`, `...\core\tool\config_args.py`, `...\core\gui\fields.py`, `...\core\tool\fields.py`, `...\core\refusals.py`, `...\core\diagnostics\{sbc,identifiability,ablation,probes}.py`.

#### Window-only additions
The stage already takes the argument; the window never passes it.
- **Name and note at the start of a run.** Every stage takes both and refuses a taken name as its first act: build_prior (orchestrator.py:599, :652), build_posterior (:792, :892), validate_calibration (:1874, :1938), both inference compositions (:2452/:2483, :2538/:2554), tsnpe_round (:2574, :2602). Window calls pass neither (prior_tab.py:244-245, posterior_tab.py:148-150, validate_tab.py:75-77, infer_tab.py:470-488, tsnpe_tab.py:157-161). The window must judge the note itself with `require_note`; the store does not (store.py:881-882; pattern at fdt_panel.py:300-302).
- **Epoch ceiling, checkpoint cadence, resume policy**: build_posterior (:798-799), tsnpe_round (:2572-2573). A cadence control must also be passed to `training_preview` (:481-482; called base.py:138-139) and `fresh_run_near_misses` (:422-423; posterior_tab.py:207-208), or the status line and the dialog judge another cadence.
- **A round's network, learning rate, patience**: tsnpe_round takes them (:2570-2571) and forwards only what is given (:2648-2652).
- **Draw counts**: `num_posterior_samples` (:1876), `n_samples` (:2152).
- **Calibration seed box**: `seed` (:1876); the window draws one (validate_tab.py:76).
- **Device**: not a stage argument. It is `cli.make_sim_config(hw=)` (cli.py:279); `ConfigDraft.make_config` passes none (session.py:32-37). The tool's rule is config_args.py:190-199. The budget lines read `detect_device()` (base.py:77-82; prior_tab.py:297) and would have to follow the choice.
- **Rename for every kind**: `ArtifactStore.rename` is kind-generic (store.py:852-872); only two call sites exist (prior_tab.py:328, posterior_tab.py:307).
- **Loading a posterior's own prior with it**, as the tool does (config_args.py:348-361).
- **All eight diagnostics**: each is a `@public_entry` taking the config, loaded wrappers, keywords, `name`, `note`, `fig_sink`, `store`, and writing one `diagnostic` record (sbc.py:86-90; identifiability.py:46-48, :381-385, :844-848; ablation.py:84-86; probes.py:512-516, :1090-1092, :1338-1342).

#### Would need an engine change
- A checkpoint for the fit (note 2C; not re-read here).
- A seed for a prior build, a training or a round: none takes one (orchestrator.py:598-604, :787-801, :2566-2574). `smoke` seeds by wrapping the chain (smoke.py:211); nothing records it.
- A fixed probe count in the calibration stage itself: hard-coded `None` (:1969-1971). Only `sbc_repeats` has it, and that gives the rank half of the verdict only (sbc.py:207-215).
- A name for an observation: the compositions pass none (:2528, :2558), in both front ends.
- `smoke` as one action: the composition is tool code taking the argparse namespace and printing (smoke.py:139-238).
- Flow settings that are constants, not arguments: spline bins, minibatch 512, estimator family (orchestrator.py:1381-1390).
- FDT's five unexposed settings (cli.py:391-392; sim_config.py:611-618).
- The memory-line cadence, read once at import (core/tool/__init__.py:36-37).
- Refusal hygiene under a new screen: `channel_ablation` raises plain `ValueError` (ablation.py:41, :105, :115, :130, :132), which the window shows as the red crash box; the hoisted `resume='require'` refusal has no field (:1111-1112; the pipeline's copy has, pipeline.py:1510-1512); build_posterior dereferences `prior` before its load branch (:902).
- Progress inside `sbc_repeats` (`show_progress_bar=False`, sbc.py:172).

#### Diagnostics and checks with no screen
Nothing under core/gui imports core.diagnostics or smoke (gui/fields.py:178-186 all `None`).
- **Gate 1**, the verdict: has a screen, Validate (orchestrator.py:2066, :2089; validate_tab.py:87-90).
- **Gate 2**: `ablation`. No screen. It needs the posterior's own cache (ablation.py:103-109), so a cadence of 0 forfeits the gate.
- **Gate 3**: the training run's `[chi] masked probes` total (pipeline.py:232-234) against `probes mask`'s reading. Neither half has a screen, and it is read when generation ends (retrain.md:572-575), when no record yet holds the line. The tally is in the cache state (`chi_masked`, pipeline.py:1616); whether the cache's manifest shows it: not established. The note's 2D table omits this gate.
- **Gate 4**: `identifiability rotation`. No screen; the Artifacts pane shows 3 of 13 eigenvalues (report.py:30, :57-60).
- **Stratified SBC**: four `sbc` runs; the runbook calls them a characterisation, not a gate (retrain.md:88-91, :404-405). No screen.
- **Pre-flight**: smoke; the epoch timing, two trainings with `--max-epochs 9 --checkpoint-every 0` in a scratch store (retrain.md:154-163), impossible in the window; `identifiability jacobian` twice (:251-266); `probes band`, which must hold (:288-292); `probes mask` (:293-305).
- **Certification round**: truth-containment lines plus a calibration whose `accepted` shows `truncated` (retrain.md:662-668); empty when run straight after the round in the window (:1488).

#### Refusals
- The core raises `Refusal(message, field=key)` (refusals.py:34-45); the key must be in `FIELDS` (:79-214), enforced by an AST scan (tests/test_refusals.py:1067-1078).
- Window, at the click: each tab's `_read_inputs` runs the same `require_*` rules on `value_or_none()` (e.g. posterior_tab.py:171-181), then `BasePanel._refusal` (base_panel.py:406-426) logs `message + fix` and opens the yellow box (refusal_box.py:25-42): text = the core sentence, informative line = `gui_fields.fix_sentence(exc.field)`. A Refusal raised in the worker reaches the same box through `_on_error` (base_panel.py:428-435).
- `fix_sentence` (gui/fields.py:221-232): a `(place, label)` entry renders "Set it in the '<label>' box on the <place> tab."; a string entry is shown verbatim; `None` shows nothing.
- A new control must: (1) change `CONTROL[key]` from `None` to a `(place, label)` pair or a sentence, where `place` is a real tab title or a member of `SCREENS` (gui/fields.py:45; pinned tests/test_nav_and_gating.py:1534-1544); (2) build its row from `label(key)` so the read-back pin finds it (tests/test_nav_and_gating.py:3989 onward); (3) update the pinned no-control set (:1620-1630) and the docstring's "six settings" (gui/fields.py:23-25); (4) for `name`, `note` and `seed`, rewrite the entries and their verbatim pins, since they name only the Save box and the FDT tabs today (gui/fields.py:96-97, :102-103, :158; tests :1582-1584, :1615-1617); (5) for a setting with no key yet, add a `Field` to `FIELDS` and an entry to both tables, which must equal the registry (tests/test_refusals.py:995; tests/test_nav_and_gating.py:1520); (6) keep every core message free of box, tab and flag names.
- All keys the missing controls need already exist: `checkpoint_every`, `resume`, `max_num_epochs`, `device`, `n_samples`, `num_posterior_samples`, the eleven diagnostics knobs and the twenty-one probe keys (refusals.py:87-88, :102, :106-107, :127, :175-213).

### Tables

Paths are repo-relative to `C:\Users\J\PycharmProjects\PRISM\`. W = window, CL = command line. "no" in the window column means no control and nothing passed.

#### (1) Arguments, stage by stage

#### 1.0 The configuration every inference stage is handed
`cli.make_sim_config` (core/cli.py:273-279). W: `ConfigDraft.make_config` (core/gui/session.py:27-37). CL: `make_cfg` (core/tool/config_args.py:173-201).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| model | which model | the bounds folder's | Config: Model | `--model` |
| bounds_file | which parameters are inferred, their order and limits; sets the observation mode | none | Prior: Bounds (file) | `--bounds` (required) |
| bounds_dicts | the same limits typed by hand | None | Prior: Bounds, "Edit values" (prior_tab.py:192) | no |
| units_override | declared units of the files' numbers | the model's units file | Config: Units, "Type units" | no (tool/fields.py:94) |
| chi_mode | chi observation mode on or off | config.py | Config: "Multi-frequency χ(ω) conditioning" (remembered, config_tab.py:406, :421) | `--chi` / `--no-chi` |
| chi_n_freqs | probes an observation supplies | 6 | Config: χ probes per observation | `--chi-k` |
| chi_k_pad | probe slots | 12 | Config: χ probe slots (capacity) | no (tool/fields.py:109) |
| chi_max_cycles | lock-in ceiling | 20 | Config: χ lock-in ceiling (cycles) | no (tool/fields.py:110) |
| chi_f0, chi_freq_bounds | drive strength and band | config.py | read-only displays (config_tab.py:71-75) | no |
| reparam_rotate | Fisher rotation on or off | True (config.py:404) | Config: "Decorrelating Fisher rotation", remembered in PRISM.ini (config_tab.py:407, :422) | no flag; always config.py's |
| hw | compute device | auto-detect | **no** (gui/fields.py:125) | `--device auto / cpu / cuda` (config_args.py:59-63, :190-199) |

#### 1.1 Prior build
`build_prior` (orchestrator.py:597-604). W: prior_tab.py:244-245. CL: `prior` (stages.py:153-154).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| ref / build_new | load a saved prior, or build one | none | Prior picker; "(from scratch)" builds | `prior` always builds; `train`, `smoke`, `probes mask` load with `--prior REF` |
| name | the record's name; a taken one is refused at entry (:652) | "" | **no** at the build; the Save box renames afterwards (prior_tab.py:321-338) | `--name` |
| note | one-line note | "" | **no** (Artifacts Note box later) | `--note` |
| num_iterations | global sweep rounds | 50 | Prior: Global rounds | `--num-iterations` |
| sweep_batch | candidates per round, 0 = hardware batch | 0 | Prior: Candidates per round (0 = auto) | `--sweep-batch` |
| max_sets | accepted sets that stop the local fill | 175,000 | Prior: Max accepted sets | `--max-sets` |
| walk_step | random-walk stride | 0.01 | Prior: Random-walk step | `--walk-step` |
| stability_units | how long a candidate must stay bounded | 1000 | Prior: Stability duration (ND units) | `--stability-units` |
| min_cluster_size | smallest island | 50 | Prior: Min cluster size | `--min-cluster-size` |
| min_samples | cluster density rule | 10 | Prior: Min samples | `--min-samples` |
| store | which artifact store | process default | no (always the default store) | `PRISM_ARTIFACTS`; `--store-root` on smoke only |
| a seed | not an argument | none | no | no (`smoke --seed` wraps the chain) |

#### 1.2 Training
`build_posterior` (orchestrator.py:786-802). W: posterior_tab.py:148-150, values from :171-181. CL: `train` (stages.py:159-167, knobs :19-21).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| prior | the prior to train on | none | the session's prior (Prior tab) | `--prior REF` (required) |
| ref / train_new | load a saved posterior, or train | none | Posterior picker; "(from scratch)" trains | `train` always trains; other subcommands load with `--posterior` |
| name | record name; taken name refused at entry (:892) | "" | **no**; Save box renames after the run (posterior_tab.py:300-317) | `--name` |
| note | one-line note | "" | **no** | `--note` |
| num_runs | training batches | 5000 | Posterior: Batches (remembered) | `--num-runs` |
| run_size_cap | rows-per-batch ceiling, 0 = hardware batch | 0 | Posterior: Max rows per batch (0 = auto) (remembered) | `--run-size` |
| hidden_features | network width | 128 | Posterior: Hidden features (opens at 128 every launch) | `--hidden-features` |
| num_transforms | network depth | 8 | Posterior: Transforms (opens at 8) | `--num-transforms` |
| learning_rate | optimiser step | 0.001 | Posterior: Learning rate | `--learning-rate` |
| stop_after_epochs | early-stop patience | 20 | Posterior: Early-stop patience | `--stop-after-epochs` |
| max_num_epochs | epoch ceiling (trains at most N+1) | 2,147,483,647 = none | **no** (gui/fields.py:86) | `--max-epochs` |
| checkpoint_every | batches between cache commits, 0 = no cache | 50 | **no** (gui/fields.py:60) | `--checkpoint-every` |
| resume | auto / require / never | auto | **no** (gui/fields.py:61) | `--resume` |
| new_run | consent to start a cache beside one a single setting away | False | dialog "This starts a NEW run" (posterior_tab.py:183-238) | `--new-run` |
| fisher_m, fisher_dz, fisher_points | rotation's ensemble, step, operating points | 48, 0.1, 8 | Posterior: Ensemble per perturbation, Central-difference step, Operating points | `--fisher-m`, `--fisher-dz`, `--fisher-points` |
| accept | consent to load a narrowed posterior (load only) | refuse | dialog "Load a NON-AMORTIZED posterior?" (posterior_tab.py:240-285) | `--accept-truncated` |
| truncation, observation, parent_posterior | a narrowing round's region and parents | None | set by the TSNPE tab's stage | set by `tsnpe` |
| spline bins, minibatch 512, estimator family | not arguments (orchestrator.py:1381-1390) | config.py | no | no |
| a seed | not an argument | none | no | no |

#### 1.3 Calibration
`validate_calibration` (orchestrator.py:1872-1876). W: validate_tab.py:75-77. CL: `validate` (stages.py:170-181).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| posterior, prior | what is calibrated, and its training prior | none | the session's two | `--posterior`; the prior is read from its manifest |
| name | record name (refused at entry, :1938) | "" | **no** | `--name` |
| note | note | "" | **no** | `--note` |
| n_cal | calibration datasets | 2000 | Validate: Calibration datasets | `--n-cal` |
| cal_n_scales | operating points | 200 | Validate: (t_scale, T_obs) operating points | `--cal-n-scales` |
| num_posterior_samples | draws per dataset | 1000 | **no** (gui/fields.py:71) | `--posterior-samples` |
| seed | makes the calibration repeatable | None | **no box**; drawn per run, passed and recorded (validate_tab.py:76; orchestrator.py:2083) | `--seed`, else drawn (stages.py:176) |
| a fixed probe count | not an argument; hard `None` (:1969-1971) | pooled | no | no (`sbc --chi-k-fixed` instead) |

#### 1.4 Inference
`simulated_inference` (orchestrator.py:2448-2452), `experimental_inference` (:2535-2538), `infer_and_visualize` (:2149-2152). W: infer_tab.py:470-488. CL: `infer` (stages.py:184-211).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| posterior | the model to run | none | the session's | `--posterior` |
| T_obs_s | recording length, seconds | none | Infer: T_obs (s) | `--t-obs` |
| cell | cell file whose truth is re-simulated | none | Infer: Cell | `--cell` |
| gt_values | the truth typed by hand | None | Infer: direct entry grid (infer_tab.py:404, :477) | no |
| prior | for the in-distribution check | None | the session's prior, whatever it is (infer_tab.py:478) | the posterior's own |
| rec | the bench recordings, drive and amplitude | none | Infer: recording boxes, χ probe table, Drive F₀ (N), drive rows | `--spont`, `--forced PATH[@HZ]`, `--f0-si`, `--drive NAME=VALUE` |
| accept | consent to run a narrowed posterior on another observation | refuse | Infer: "Run on a different observation" | `--accept-other-observation` |
| n_samples | posterior draws | 1000 | **no** (gui/fields.py:72) | `--n-samples` |
| name | names the INFERENCE record (refused at :2483 / :2554) | "" | **no** | `--name` |
| note | note | "" | **no** | `--note` |
| the observation's own name | not forwarded (:2528, :2558) | unnamed | no | no |

#### 1.5 The narrowing round
`tsnpe_round` (orchestrator.py:2565-2574). W: tsnpe_tab.py:157-161. CL: `tsnpe` (stages.py:214-223, knobs :25-27).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| posterior, prior | the parent and its prior | none | the session's two | `--posterior` |
| observation | what the region is drawn around | none | TSNPE: Observation picker | `--observation` |
| n_directions | directions cut | 5 | TSNPE: Directions truncated | `--directions` |
| level | region's level | 0.999 | TSNPE: HPD level | `--level` |
| num_runs | batches | 5000 | TSNPE: Batches (its own box, remembered) | `--num-runs` |
| run_size_cap | rows ceiling | 0 | TSNPE: Max rows per batch (0 = auto) | `--run-size` |
| hidden_features, num_transforms | network size | 128, 8 | **no** | `--hidden-features`, `--num-transforms` |
| learning_rate, stop_after_epochs | optimiser step, patience | 0.001, 20 | **no** | `--learning-rate`, `--stop-after-epochs` |
| max_num_epochs | epoch ceiling | none | **no** | `--max-epochs` |
| checkpoint_every | cache cadence | 50 | **no** | `--checkpoint-every` |
| resume | resume policy | auto | **no** | `--resume` |
| new_run | consent for a new cache | False | tick "Start a new simulation even if a cache one setting away exists" (tsnpe_tab.py:102, :160) | `--new-run` |
| name, note | record name and note (name refused at :2602) | "" | **no** | `--name`, `--note` |

#### 1.6 Live simulation (window only)
`run_simulation_stream(cfg, t_obs_s, frame_steps=2000, fps=30)` (core/gui/panels/simulate_runner.py:187-188). W: Model, Cell, T_obs (s), Steps / frame, Max FPS, "Save video…" (simulate_panel.py:72-95). CL: none (tool/fields.py:77). No store call in simulate_panel.py.

#### 1.7 Effective-temperature analysis
`cli.make_fdt_config` (cli.py:380-382) then `run_fdt` (core/FDT/fdt_pipeline.py:247-248). W: fdt_panel.py:284-324. CL: `fdt` (core/tool/fdt.py:172-186, :207-242).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| model | model | cell folder's | Model | `--model` |
| cell_file | the cell | none | Cell | `--cell` |
| n_freqs | drive frequencies | 60 | n_freqs | `--n-freqs` |
| ensemble_M | trajectories per frequency | 256 | M_ensemble | `--ensemble-m` |
| freqs_per_batch | frequencies per call | 1 | freqs / batch | `--freqs-per-batch` |
| F0 | drive strength | 0.05 | F0 (ND forcing amplitude) | `--f0` |
| seed | random seed; blank draws and records one | None | Seed | `--seed` |
| skip_sanity | skip the sanity checks | required bool | "Skip sanity checks" | `--skip-sanity` |
| confirm_production | go on to the full sweep | required bool | "Proceed to the production sweep after sanity" | `--no-production` (inverse) |
| name, note | record name and note, claimed before anything is spent | "" | Record name, Note | `--name`, `--note` |
| store root | where the record goes | default store | no | `--store-root` |
| freq_bounds, burn_in_nd, T_obs_periods, dt_nd, psd_T_obs_nd | band, burn-in, durations, step | sim_config.py:611-618 | no | no (not builder arguments, cli.py:391-392) |

#### 1.8 The sweep study
`cli.make_param_sweep_config` (cli.py:487-491) then `run_param_study_cli` (core/FDT/cross_validation.py:596-597). W: crossval_panel.py:267-333. CL: `crossval` (core/tool/fdt.py:188-203, :245-271).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| cell_file | a Nadrowski cell | none | Cell | `--cell` |
| preset | resolution preset | exploratory | Preset | `--preset` |
| s_spec, t_spec | each sweep's min, max, point count | none | the two grid rows | `--s-grid`, `--t-grid` |
| n_freqs, ensemble_M, freqs_per_batch, F0 | resolution knobs | preset's / 1 / 0.05 | four boxes | `--n-freqs`, `--ensemble-m`, `--freqs-per-batch`, `--f0` |
| seed | one seed for both records | None | Seed | `--seed` |
| name stem, note | two records named `<name>-s`, `<name>-temp` | "" | Record name, Note | `--name`, `--note` |
| store root | where the records go | default | no | `--store-root` |

#### 1.9 Comparisons
`compare(mode, refs, *, name, note, fig_sink, store, **options)` (core/FDT/compare.py:439-440). W: fdt_panel.py:215-236, crossval_panel.py:201-220. CL: `compare <mode>` (core/tool/fdt.py:310-350).

| argument | plain meaning | default | window | command line |
|---|---|---|---|---|
| mode | cells / repeats / renormalise / sweeps | none | "Comparison" combo (first three); the sweep tab's button (sweeps) | the mode word |
| refs | the saved records | none | "Runs to compare" / "Sweeps to compare" lists | `--record` (repeat) |
| prefactor | renormalise's constant | none | Normalisation constant | `--prefactor` |
| at | sweeps' slice point | middle of shared range | Slice at | `--at` |
| name, note | the comparison record's | "" | Comparison name, Comparison note | `--name`, `--note` |

#### (2) What exists only on the command line

| item | what it is for | needs | weight (where stated) | writes | window only, or engine change |
|---|---|---|---|---|---|
| `--device` (every SBI subcommand) | pin the device; `cuda` refuses when absent | nothing | none | nothing | Window only: `make_sim_config(hw=)` exists (cli.py:279). Budget lines and the sweep note must follow it (base.py:77-82; prior_tab.py:297). |
| `--name`, `--note` on prior, train, validate, infer, tsnpe | name and annotate the record at the start; taken name refused before the spend | nothing | none | manifest fields | Window only; the window must run `require_note`. |
| `--max-epochs`, `--checkpoint-every`, `--resume` on train, tsnpe | epoch ceiling, cache cadence, resume policy | a cache for the policy to act on | none | cadence 0 writes no cache | Window only. Cadence also to `training_preview` and `fresh_run_near_misses`. Engine tidy-up: the hoisted `require` refusal has no field (orchestrator.py:1111-1112). |
| `--hidden-features`, `--num-transforms`, `--learning-rate`, `--stop-after-epochs` on tsnpe | the round's network and fit | parent posterior, observation | a second full training | the round's posterior | Window only (orchestrator.py:2570-2571). |
| `--posterior-samples` (validate), `--n-samples` (infer) | draw counts | a posterior | scales the sampling | recorded | Window only. |
| `--seed` on validate | choose the calibration seed beforehand | a posterior | none | `results.seed` | Window only (argument exists). |
| `--store-root` (smoke, fdt, crossval), `PRISM_ARTIFACTS` | another artifact root | nothing | none | everything, elsewhere | Every stage takes `store=`; the window calls `default_store()` throughout. Window only, but touches every panel. |
| `smoke` | every stage end to end at tiny sizes; the card gate | bounds, cell | "minutes" (smoke.py:7) | prior, cache (with `--checkpoint`), posterior, observation, calibration, inference, in its own root | **Engine change**: the composition is tool code (smoke.py:139-238) taking the argparse namespace, printing, seeding by wrapper (:211). |
| `sbc` = `sbc_repeats` (sbc.py:86-90) | calibration rank test repeated, optionally at one fixed probe count | posterior (its prior from the manifest), bounds | default 10 repeats × 2000 datasets; each repeat about one `validate` (retrain.md:416) | one diagnostic record (`sbc_repeats.npz`, pooled histogram) | Window only. No progress bar inside a repeat (sbc.py:172). |
| `identifiability rotation` (identifiability.py:46-48) | what the posterior's rotation measured; gate 4 | a posterior with a rotation | simulates nothing; "runs in a second" (identifiability.py:7) | one diagnostic record | Window only. |
| `identifiability laplace` (:381-385) | is the information there across the prior | posterior, cell, length; refuses chi mode and unforced boxes (:401-402) | simulates; cost not stated | one diagnostic record (`laplace_sd.npz`) | Window only; not usable on the chi retrain. |
| `identifiability jacobian` (:844-848) | degeneracy map at a cell; pre-flight recording-length check | bounds, cell, length; Nadrowski only (:865); no posterior | simulates; cost not stated | one diagnostic record (`degeneracy_map.npz`, figures) | Window only. |
| `ablation` = `channel_ablation` (ablation.py:84-86) | which input channels the network can see; gate 2 | posterior and its own simulation cache | simulates nothing; reads the first 200,000 rows | one diagnostic record | Window only to call; **engine change** to make its failures refusals (plain `ValueError` at ablation.py:41, :105, :115, :130, :132). |
| `probes band` (probes.py:512-516) | do the configured band and drive hold for this cell; pre-flight | bounds, cell; builds its config in chi mode | default 5 lengths × 5 frequencies × 1 drive × 24 runs; cost not stated | one diagnostic record, figure | Window only. |
| `probes mask` (probes.py:1090-1092) | why training throws probes out, over a prior; gate 3's reference | bounds, a prior | 12 batches of 32 rows took 43 s on the card (retrain.md:294); the runbook's 100 batches about six minutes, an estimate | one diagnostic record (`probe_mask.npz`); no cache | Window only. |
| `probes drive` (probes.py:1338-1342) | how hard a lab can drive a cell | bounds with a Forcing section, cell; chi off | 1 undriven + 17 driven ensembles of 16 runs at 5 s by default; cost not stated | one diagnostic record, figure | Window only. |
| `PRISM_MEM_LOG_EVERY` | batches between memory log lines | set before launch | none | log lines | **Engine change** for a live control: read once at import (core/tool/__init__.py:36-37). |
| the `[cfg]` banner | prints the parameter order on every run (config_args.py:204-232) | nothing | none | console | Window only; the Prior tab's "Config built" line gives counts, not the order (prior_tab.py:216-219). |
| the posterior's own prior, loaded automatically | no hand pairing (config_args.py:348-361) | a posterior | one prior load | nothing | Window only. |

Not command-line only: `artifacts list/show/note/rm/sweep/summary` (the Artifacts screen), `compare`, `fdt`, `crossval`.

#### (3) What exists only in the window

| item | evidence |
|---|---|
| Rotation on/off, probe slots, lock-in ceiling, typed units | config_tab.py:76-77, :89-90, :329-334; tool/fields.py:94, :109-110 |
| Bounds typed by hand; truth typed by hand | prior_tab.py:192; infer_tab.py:404, :477 |
| VRAM ceiling as a field (the tool has the environment variable only) | config_tab.py:111-119 |
| Rename of a prior or posterior after the run ("Save") | prior_tab.py:328; posterior_tab.py:307; no rename mode in core/tool/browse.py:553-554 |
| Live simulation, model builder, NWK to Hopf reduction map | tool/fields.py:73-77; cli.py:428-431 |

## Where a long run could attach

### Findings

**Read:** `docs/superpowers/prompts/2026-10-05-ui-ux-redesign.md` in full, the note, the assigned engine files, and sbi 0.26.1's trainer as installed (`C:\Users\J\anaconda3\envs\biophys-env\Lib\site-packages\sbi\inference\trainers\base.py`, "sbi base.py" below). Nothing was run. None of (a)-(h) needs a change to the cache key or to what a row holds.

#### (a) A lasting, timestamped file written as the run goes
Today a run's lines sit in memory (`RunLog.lines`, core/runs.py:107), stamped `HH:MM:SS` with no date (:59-60), and reach disk only in `ArtifactWriter._write_log` at the commit (store.py:552-574, 627-629). The posterior's writer opens after the fit (orchestrator.py:1420), `refresh()` is refused for it (store.py:525-527), and the buffer holds only `core` records and warnings, never sbi's prints. Smallest point, window only: `_Pump.sink`, "log" branch (core/gui/streams.py:121-130). Every finished line of all three channels passes it on the worker thread (prints vt.py:181-186, records streams.py:251-258, warnings :296-297); written there rather than in `_publish` it outlives the pump's stop (:152-155). The file cannot sit in the posterior's folder: none exists yet. Engine alternative: tee at runs.py:81-87 and 122-131 (runs.py is torch-free and cannot import config, so the path must be handed in). Making the posterior a progressive kind (store.py:73) is not small: manifest.py:157-168 rejects a partial body.

#### (b) Clock time and loss per epoch
Per epoch sbi appends train loss, validation loss and duration to the trainer's `_summary` (sbi base.py:993-995, 1120-1123), then prints one `\r` line with the count (:998, 1274). No callback exists. The `tracker` that `train_nn` accepts (train.py:192, 241-242) is NOT live: sbi calls `log_metric` only after the loop (base.py:1009, 1210-1250), and `build_posterior` passes none (orchestrator.py:1380-1391). The trainer is a local (train.py:241); durations are dropped (train.py:403-411). Clock: window only. The count arrives as a row event with `pct=None` (vt.py:289-300, 311-319, 188-198, then streams.py:124-125); stamp each change there. Loss: engine change, train.py only. Wrap the trainer's `_summarize_epoch` before train.py:365 and emit one `core` info record per epoch; it then reaches the pane, `log.txt` and any file from (a) with no new argument. It uses an sbi private name, as train.py:404 already does, and draws no random numbers.

#### (c) Saving the fit as it goes, and resuming
One call (train.py:365-370). All fit state is on the in-memory trainer: net, `optimizer`, `epoch`, `_best_val_loss`, `_epochs_since_last_improvement`, `_best_model_state_dict`, `_summary`, the train/validation split (sbi base.py:971-976, 1143-1148, 444-446). `resume_training=True` only continues that same object (npe_base.py:274-277). Save point: the same per-epoch wrapper. Resume: rebuild from the complete cache, load the state, `train(resume_training=True)`. Driving the fit in chunks is not neutral: each `train()` call ends with an extra `_converged` (base.py:1000-1002, 1169, 1147-1148), which advances patience once per chunk. Engine change, the largest here, and it touches how the fit is driven. A fit checkpoint needs its own key (cache digest plus network size, bins, learning rate, patience, minibatch). Kept under the cache folder it changes neither the digest (training_checkpoint.py:110-125) nor what `load_rows` reads (:502).

#### (d) Restart after an out-of-memory failure
Three ladders (pipeline.py:767-845, 848-922, 1817-1877; three whole-batch retries at 15/60/180 s, config.py:320, 326), then the rescue save (:1951-1990), and the error reaches `Worker.run` (worker.py:146-162). Nothing calls again. A restart is the same `build_posterior` call: it resumes at `batches_done` and reuses the stored rotation (orchestrator.py:1090-1096, 1207-1238; pipeline.py:1513-1538). Window only: re-dispatch with `resume="require"` (orchestrator.py:799, 1111-1112) when `pipeline._is_oom` (:313-339) is true. Whether a retry in the SAME process can work is not established, and part 1 of the note says it did not; the learned cap (pipeline.py:100, never reset) and the CUDA context persist. A fresh process means the window running the stage in a child process instead of a pool thread (base_panel.py:289): window architecture, no engine change.

#### (e) A name and a note checked before anything is spent
Name: already the first statement of `build_posterior` and `tsnpe_round` (orchestrator.py:892, 2602); `assert_name_free` is public (store.py:815-844). The window sends neither (posterior_tab.py:148-150, 171-181). Window only. Note: no stage checks it; `require_note` (refusals.py:370-387) is each front end's job (store.py:881-882). Hazard: the name is checked again at the end (store.py:849, reached from orchestrator.py:1420), before anything is written.

#### (f) The ceiling and the memory-line cadence as arguments
Ceiling: one consumer, `sim_memory_budget_elements` (pipeline.py:636), called from `_max_sim_batch` (:666) inside `gen_obs` (:724), plus the preview (orchestrator.py:532). It reads the environment first, then `config.SIM_VRAM_CEILING_GIB` (:584-603), which the window's field writes (config_tab.py:148). As an argument it must travel through every stage that simulates, or be installed by the stage for the call's duration. Cadence: bound at import (pipeline.py:269), read at :1919; one keyword down to `gen_training_data`. Both are engine changes and both reverse a recorded choice (core/tool/__init__.py:38-40). Neither is in the key (config.py:164-168); the ceiling re-blocks the noise exactly as it does today (pipeline.py:720-723). The ceiling a run used is recorded nowhere.

#### (g) Knowing before the button whether the run resumes, and why not
`training_preview` (orchestrator.py:481-564) gives the state and `batches_done`; the window prints its `siblings` string verbatim (core/gui/panels/inference/base.py:201-216). The full list of differing keys is already computed (`_sibling_diffs`, training_checkpoint.py:285-287); `describe_siblings` prints `diff[0]` (:294-295). Smallest: carry the lists on `TrainingPreview` (orchestrator.py:554). Engine change, read-only. The key's names are not the window's field keys (`n_runs` against `num_runs`, `prior_fingerprint` against `prior`, five keys for one bounds file), so the window needs its own table. The preview only peeks; it does not run `verify` (:226-263).

#### (h) Closing the window under a run
`closeEvent` sets the token or not, saves the settings and accepts at once (main_window.py:423-445); `main()` returns and the interpreter exits (core/gui/__main__.py:16-23). Nothing waits for the pool thread, and what happens to it is NOT established. On disk: the cache keeps every committed batch (state.pt, training_checkpoint.py:384-395); the rescue save needs the thread to reach a checkpoint and finish writing. A fit leaves nothing: the posterior folder is created only at orchestrator.py:1420 (store.py:415). If the thread does live to the end, the loss figure's emit (base_panel.py:62) runs inside the writer (store.py:408-409), any exception there removes the folder (:436, 479-480), and the worker swallows the report (worker.py:146, 205-210). Smallest guard: catch around `forward` at store.py:408-409 (engine), or a sink that cannot raise (window).

### Tables

**The simulation cache's key: 34 settings (core/artifacts/identity.py:87-122). The digest is 12 hex characters of sha256 over the canonical JSON (training_checkpoint.py:110-113).**

| # | key | what it is | window control |
|---|---|---|---|
| 1 | `format` | version tag of the key itself, "training-rows/3" (identity.py:45) | none: code constant |
| 2 | `model` | the model's NAME only | Config, "Model" (fields.py:124) |
| 3 | `prior_fingerprint` | fingerprint of the prior's fitted mixture | Prior, "Prior" picker plus Build / Load prior (fields.py:126) |
| 4 | `mode` | observation mode | none of its own: follows the bounds file and the chi checkbox |
| 5 | `param_keys` | names and order of the inferred parameters | Prior, "Bounds" (fields.py:120) |
| 6 | `nd_lows` | lower limits, non-dimensional parameters | Prior, "Bounds" |
| 7 | `nd_highs` | upper limits, non-dimensional parameters | Prior, "Bounds" |
| 8 | `rescale_lows` | lower limits, dimensional parameters | Prior, "Bounds" |
| 9 | `rescale_highs` | upper limits, dimensional parameters | Prior, "Bounds" |
| 10 | `log_params` | parameters placed in a log box | none: `config.REPARAM_LOG_PARAMS` (config.py:420, now `[]`), or a user model's own setting (run_guards.py:192-212) |
| 11 | `reparam_rotate` | Fisher rotation on or off | Config, checkbox "Decorrelating Fisher rotation" (config_tab.py:89-90; not in fields.py) |
| 12 | `run_size` | rows per batch = min(cap, hardware batch) (orchestrator.py:415-419) | Posterior/TSNPE, "Max rows per batch (0 = auto)" (fields.py:59); hardware batch is 2048 on CUDA (config.py:78-79) |
| 13 | `n_runs` | number of batches | Posterior/TSNPE, "Batches" (fields.py:58) |
| 14 | `steady_idx` | where the transient ends on the fine grid | none in fields.py; derivation not traced |
| 15 | `dt_nd_min` | finest non-dimensional time step | none in fields.py; derivation not traced |
| 16 | `dt_exp` | sampling interval in the cell's time unit | none: `DT_EXP_S` times the units' time factor (cli.py:343), so the Units choice moves it |
| 17 | `t_min_exp` | shortest training recording | none: `T_MIN_EXP_S` times the units factor (cli.py:344) |
| 18 | `t_max_exp` | longest training recording | none: `T_MAX_EXP_S` times the units factor (cli.py:345) |
| 19 | `t_scale_bounds` | limits of the time scale | none in fields.py; source not traced |
| 20 | `n_grid` | length of the fine time grid | none: derived |
| 21 | `spontaneous_only` | no drive | none of its own: follows the mode |
| 22 | `summary_flags` | labels of the feature-validity flags | none: code constant |
| 23 | `feature_set_version` | version number of the summary features | none: code constant |
| 24 | `chi_mode` | chi conditioning on or off | Config, checkbox "Multi-frequency chi conditioning" (config_tab.py:64, 79) |
| 25 | `chi_layout` | layout version of the chi block (None when chi is off) | none: `config.CHI_LAYOUT` (config.py:526) |
| 26 | `chi_k_pad` | probe slots | Config, "chi probe slots (capacity)" (fields.py:139) |
| 27 | `chi_elem_w` | channels per probe slot | none: `config.CHI_ELEM_W` (config.py:527) |
| 28 | `chi_f0` | probe drive amplitude | read-only display; "Fixed by measurement" (fields.py:141) |
| 29 | `chi_freq_bounds` | probe frequency band | read-only display (fields.py:142) |
| 30 | `chi_max_cycles` | lock-in ceiling in cycles | Config, "chi lock-in ceiling (cycles)" (fields.py:140) |
| 31 | `device` | "cuda" or "cpu" | none (fields.py:125 is None); `detect_device()` |
| 32 | `dtype` | float32 | none |
| 33 | `truncation` | the narrowing region's fields; None for an ordinary run | TSNPE tab only, through the region (fields.py:65-67) |
| 34 | `units_sha256` | fingerprint of the units file, or of typed unit tokens (identity.py:51-67) | Config, "Units" (fields.py:123) |

**Asked about, and NOT in the key**

| setting | in the key? | evidence |
|---|---|---|
| network size (hidden features, transforms) | no | identity.py:87-122; resolved at orchestrator.py:936-937 |
| VRAM ceiling | no, deliberately | config.py:164-168 |
| epoch ceiling | no | identity.py:87-122 |
| checkpoint cadence | no; the header stores none either | orchestrator.py:837-840 |
| name | no; a cache's name is always "" | store.py:1582 |
| note | no | store.py:1583 |

Also absent: learning rate, patience, the three Fisher settings and the rotation itself (identity.py:10-13), the flow's minibatch size, the resume policy, the memory-line cadence, chi probes per observation, `state_dep_drift`, the git revision, and a user model's definition (only its name is keyed).

## Guide and walkthrough

## 1. The window today, by the pages

- **Shell.** Always opens on Home; five sections; back arrow under the PRISM title; gear bottom-left opens the appearance menu, then **Full settings…** (`docs/guide/window.md:13-15`, `:306`). The records root is resolved once at start (`getting-started.md:61-62`).
- **Sections** (`window.md:17-23`):
  - Reduction Map: one analytical tab, Nadrowski only.
  - FDT Analysis: "FDT analysis" and "Sweep study cross-validation"; both write `fdt` records and each has a comparison box.
  - Parameter Inference: six tabs.
  - Simulate: Live simulation, **Save video…**.
  - Artifacts: one screen, no tabs.
- **Not on Home** (`window.md:25-32`): Settings (appearance, user-model list with Open model builder / Edit / Delete, help reference built from each section's `HELP`) and the model builder. Builder flow, `recordings.md:416-434`: name and variables, **Set variables**, equations / noise / initial condition / drive, **Detect parameters**, display scales, **Validate**, **Save model**.
- **Inference order and gates** (`getting-started.md:88-103`, `window.md:41-53`):
  - Config, **Apply model & options**: always enabled.
  - Prior, **Build / Load prior**: needs an applied model; the Bounds picker builds the configuration and fixes the mode.
  - Posterior, **Train / Load posterior**: needs the built configuration; training also needs a loaded prior.
  - Validate, **Run calibration**: needs a posterior and the Prior-tab prior.
  - Infer, **Run inference**: needs a posterior; writes an observation.
  - TSNPE, **Run TSNPE round**: as Validate, plus an observation in its picker.
  - A greyed tab's hover says what is missing; a stage that greys the current tab falls back to Config.
  - Naming is after the fact: Save on the Prior and Posterior tabs only, and it is a rename (`getting-started.md:102-103`).
- **Each tab**: controls column, figure stack over log pane, progress pane with Cancel (`architecture.md:626-629`). The Posterior tab has three live lines: simulation count, worst-case batch memory against free, resumes or starts new (`window.md:61-65`).
- **One run in the whole window**: all panels locked, navigation free, clickable header "Running: Posterior — 4:07", tile and tab markers (`window.md:69-75`).
- **Dialogs** (`window.md:87-113`): "Start a new session?" (default Keep); "This starts a NEW run" (default Cancel, one-setting near miss only); "Load a NON-AMORTIZED posterior?" (default Cancel); the ticks **Run on a different observation** and TSNPE's new-simulation.
- **Artifacts** (`getting-started.md:231-249`): kind, table, detail pane (manifest and log); **Save…**, **Lineage report…**, note **Set**, **Delete…**, two Sweeps. Writes are refused during a window run; another process is guarded only by the sweeps' five-minute recency check.
- **Remembered** (`PRISM.ini`, written on an accepted close; splitters and appearance at once; `window.md:287-306`):
  - Config: model, units, chi tick, rotation tick.
  - Prior: Bounds and Prior pickers.
  - Posterior: picker, Batches, Max rows.
  - Validate: nothing.
  - Infer: mode, Cell, three T_obs, recording paths, Drive F₀ (N).
  - TSNPE: Observation picker, Batches, Max rows.
  - Live simulation: all five fields.
  - FDT: model, cell, Saved run, n_freqs, M_ensemble, freqs / batch, F0.
  - Sweep study: preset, cell, Saved sweep, F0, freqs / batch, grid point counts.
  - Reduction: cell, F0. Artifacts: kind, sort. Window: geometry, splitters, appearance.
  - Fresh every launch: every science knob, the VRAM ceiling, seeds, names, notes, consents, compare lists, grids, the screen.

### 2. Rules and traps that concern the window

- **Selections versus science knobs.** Mistake: a retired chi band restored from saved settings trained about five days on it (`window.md:312-316`). Rule: the inference tabs remember selections, recording boxes and the budget only; band and drive are read-only; any persisted measurement definition is compared with `core/config.py` before the spend (`rules-and-traps.md:171-180`). Enforced by `run_guards._assert_chi_config_is_deliberate`. Tests: `test_settings_persistence.py::test_science_knobs_open_at_config_and_are_not_written`, `::test_the_config_tab_science_knobs_open_at_config_and_are_not_written`, `test_nav_and_gating.py::test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config`.
- **A pre-flight reads its configuration where the run does.** Mistake: smoke was green while the window was about to train on its own saved band. Both front ends build through `cli.make_sim_config` (window: `session.ConfigDraft`) (`rules-and-traps.md:182-188`).
- **Never remembered** (`window.md:317-330`). No test is named on the page except the last.
  - Consents (new-simulation tick, Run on a different observation, FDT's two ticks): a remembered yes silences a later refusal.
  - Seed: it would make every run a repeat.
  - Record name and note: refused as taken at the next launch.
  - Compare boxes: the records may have been deleted.
  - Grid ends and hand-edited grids: `test_crossval_does_not_persist_cell_derived_bounds`.
- **VRAM ceiling never remembered**: a forgotten ceiling slows every later run with nothing in the log (`window.md:326-327`). It is the one field that assigns `core.config` (`ConfigPanel._apply_vram_ceiling`, read live; `window.md:210-222`, `rules-and-traps.md:713-720`); the environment variable wins. No test named.
- **Every setting is an argument.** Mistake: assigning a config constant after import is a silent no-op. Tests: `test_user_sbi.py::test_build_posterior_takes_the_budget_as_arguments_because_the_constants_are_snapshotted`, `test_settings_persistence.py::test_the_training_budget_reaches_build_posterior_as_arguments_not_via_config` (`rules-and-traps.md:90-100`).
- **Refusal and field key** (`rules-and-traps.md:111-133`). `Refusal.field` is registered in `core.refusals.FIELDS`; the core message names no box, tab or flag; the window appends the control from `core/gui/fields.py` in the yellow "Check your inputs" box. Tests: `test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered`, `test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it`, `test_refusals.py::test_every_domain_error_is_a_refusal_and_carries_a_field`. Known gaps are at `:118-128`.
- **Threads** (`rules-and-traps.md:249-262`, `:384-386`). Keep the worker reference (`BasePanel.dispatch`, `_workers`) or the panel stays busy for ever. One run: `BasePanel._running` is a class attribute and `streams._REDIRECT` is the backstop. The control lock covers all six tabs (`_set_busy` over `_instances`).
- **Figures** (`:253-291`).
  - Never paint a worker-built figure: deadlock, no traceback; guard `_png_fig_sink`.
  - Agg first (`core.gui.__main__`).
  - `visualizers.save_figure`, never `savefig` (theme flip; `test_figures.py::test_a_theme_flip_between_build_and_save_cannot_split_a_figure`). The FDT and reduction figures still call `savefig`.
  - Explicit table-cell background (`::test_the_parameter_table_is_readable_against_its_own_cells`).
  - Pop-out detached from pyplot, Qt backend imported lazily (`::test_pop_out_of_a_pickle_builds_a_qtagg_canvas_and_keeps_pyplot_clean`).
- **Progress and cancel** (`:300-343`).
  - Classify tqdm chunks (`vt.StreamRouter.feed`); rows keyed by position; the 15 Hz pump is load-bearing.
  - No `fileno()` or `encoding` on `_SignalStream`: no guard.
  - Overall bar is the largest total above 1; the caption falls back to sbi's epoch counter (two `test_vt_progress.py` tests).
  - Solver meter reads `core.progress.SOLVER`.
  - Reset tqdm's lock after a cancel (`test_worker_dispatch.py::test_cancel_token_latches_and_leaves_tqdm_usable`).
  - Cancel raises on the owner thread only; `WorkerCancelled` is a `BaseException` (`::test_cancel_is_not_consumed_by_a_non_worker_thread`, `::test_worker_cancelled_passes_through_except_exception`).
- **Modal dialogs**: an unexpected modal stalls the offscreen suite rather than failing it; the `_no_modal_dialogs` fixture (`testing.md:90-91`).
- **Session and persistence traps** (`:351-398`).
  - Restore order lives in `ArtifactPicker.repoint`; a picker's key is its relative path with a missing-item guard.
  - Validate and TSNPE gate on the Prior-tab prior (`test_inference_tab_gates_follow_the_session`).
  - A new draft replaces the session, `install_config` edits it (`test_apply_confirms_before_it_discards_the_session`).
  - `value_or_none()`, the `HELP` dict convention and the `inference_tabs.py` shim (Settings reads it by string path): no guard.
- **Non-panels**: the model builder and Settings' delete check `BasePanel._running` themselves, because the lock does not reach them (`:462-464`).
- **Cost on screen**: the TSNPE tab shows the budget group and cannot say in advance whether a round resumes (`:221`, `:683-687`; `test_settings_persistence.py::test_the_tsnpe_tab_never_claims_it_will_resume_the_amortized_checkpoint`).

### 3. Runs, threads and Cancel, by architecture.md

All from `architecture.md:620-692`.

- **Dispatch.** `BasePanel.dispatch()` runs a callable as a `Worker` (`QRunnable`, global pool). The worker is held in `_workers` with auto-delete off until `finished`, then its callable and arguments are cleared. Figures are rendered to PNG and pickled on the worker; `plot_watcher` shows figures a run saves into a record's `figures/`.
- **Output.** `streams.redirect_streams` swaps `sys.stdout` and `sys.stderr` process-wide, puts `_PumpLogHandler` on the `core` logger and routes warnings to the log pane. A daemon `_Pump` publishes at 15 Hz.
- **Cancel** is cooperative, through `CancelToken`.
  - Checkpoints are `_SignalStream.write` and `_PumpLogHandler.emit`, sbi's per-epoch print included.
  - Latency is about a second, but "up to one training epoch while the flow trains" and the calibration's C2ST block (`:667-668`).
  - It is deferred, never discarded, inside `runs.cancel_deferred()` and while an exception unwinds.
  - A hard cancel (a process per job) was considered and declined (`:679-680`).
- **Close.** This section says only that `MainWindow` saves every panel's state on close (`:174-175`). The three-button dialog is described in `window.md:78-80` alone; what the worker does after the window is gone is not on any page.

### 4. The walkthrough

`docs/checklists/display-walkthrough.md`: 87 rows in seven groups. No test reads it.

| ids | rows | covers | run | result |
|---|---|---|---|---|
| 1–20 | 20 | icon, navigation, each tab once, progress pane, Simulate, video, FDT, sweep study, builder, Settings, layout, close while running | 2026-09-11 | 19 pass; row 1 failed (taskbar icon), fixed in `df7491e`, re-checked pass |
| A1–A9 | 9 | what each stage leaves under `Artifacts/`, which refusals reach a dialog | 2026-09-11 | pass |
| B1–B8 | 8 | narrowed-posterior load dialog, near-miss dialog, TSNPE checks | 2026-09-15 | pass |
| C1–C11 | 11 | read-only chi values, yellow box, what a relaunch restores, budget lines, `log.txt` | 2026-09-17 | pass |
| D1–D17 | 17 | Artifacts screen, header run line, session dialog, table in both themes | 2026-09-21 | pass |
| E1–E17 | 17 | FDT and sweep records, comparisons, tidy-up | 2026-09-25 | pass |
| F1–F5 | 5 | verdict lines, temperature label, tooltips, truth containment | 2026-09-30 | pass |

- **Screen size**: only row 19, "resize to 1366×768; drag every splitter" (`:28`). No display scaling, DPI or monitor is recorded anywhere.
- **Row format**: `| # | surface | do | expect | date | result |`. A typical row (`:94`):

  `| C6 | Posterior tab: science knobs open at config.py | Set "Hidden features" to 7 and "Learning rate" to 0.01; quit; relaunch. | Both boxes show `config.py`'s values; the Batches box and the posterior picker are restored as before. | 2026-09-17 | pass (the user's run on the real screen) |`

- **Group format**: a heading, a paragraph stating that earlier rows are not edited and which ones the new rows supersede, and optional "How to read the rows below" bullets (setup, order, names, what the screen cannot reach). On-screen strings are quoted verbatim.
- **What the checklist itself says is not exercised**: only the cases it calls unreachable from the screen and pins by tests instead (E5 normalisation refusal, E7 partial point failures, E8 one-sided sweep failure, the single-cell refusal at `:144-148`, and E12's refusal reached at a terminal). Every row is dated and passed.
- **Note 2F, checked**: the checklist contains no row for the VRAM ceiling field, "Quit anyway", 256 × 10, 2,048 rows, the stall notice or a training-crash red box. Its only "ceiling" is the chi lock-in ceiling (C11).
- **Correction to the note**: training sizes are stated as 2, 3 or 4 batches wherever a row states one, but rows 7, 20, D3, D10, D11 and D15 ("a full training") state none. The note's "row 20 … at a small run" is not established by the row.

### 5. Page errors

Part 3C, all four confirmed against `window.md`:

1. `window.md:76-77`: "**Cancel** is a request: the run stops at its next checkpoint, which can take about a minute during training." Wrong: the bound is one epoch (`architecture.md:667-668`; the code comment at `base_panel.py:297-298` says the same). The page copies the on-screen strings, which carry the same error (`base_panel.py:299-300`, `main_window.py:429`). Walkthrough rows 7 and A8 record "within about a minute".
2. `window.md:78-80`: "**Quit anyway** closes the window and leaves the run going in the background until it finishes". This mirrors the dialog (`main_window.py:430-431`) and is unproven.
   - Quit anyway requests no cancel and goes straight to `_save_state()` and close (`main_window.py:438-445`).
   - Afterwards the worker's emits raise `RuntimeError` and are swallowed (`worker.py:207-210`); the pump stops itself (`streams.py:152-155`); the figure sink's emit is unguarded (`base_panel.py:62`).
   - Whether the record is still written: not established (store path not traced here).
   - The same sentence says "**Cancel task & quit**, the default, cancels the run". The default is confirmed (`main_window.py:435`), but it only requests the cancel and closes at once; the page omits that.
3. `window.md:219`: "0, the default, is right on an idle card." Present, and repeated on screen (`help_text.py:47-48`; the field's note at `config_tab.py:159-161`, "only if you need the desktop"). The note's part 1 contradicts it.
4. No mention of the 5,000-line pane or the stall notice: confirmed. `window.md:117` is all the page says about the panes. The code has both (`log_pane.py:15`; `progress_pane.py:69`, `:334-337`). `architecture.md:661` and `rules-and-traps.md:328` name "the stall detector" with no threshold or wording.

Other statements the note's part 2 contradicts or leaves incomplete:

- `window.md:218-219`, "Set it to about the free memory `nvidia-smi` reports, minus about 1 GiB": the stopped run reserved up to 14.8 GiB under 8, and 6 was the first setting that held.
- `window.md:160`, "A hard ceiling on the card memory one simulation batch may plan to hold": it bounds the plan, not what the batch holds (9.1–10.5 GiB allocated under 6–8).
- `window.md:326-327`, "a forgotten ceiling does not fail": one-sided; the ceiling reopening at 0 is what killed the run.
- `window.md:63-65`, "it must begin "Resumes"": says nothing of the passive "WARNING: these settings match no checkpoint" line (walkthrough C7) or that it can name the wrong setting.
- `window.md:296`, the Bounds picker "kept from the last session": the note says it is applied only at Apply and overwritten by a look-and-close session. Not re-checked here (`prior_tab.py` not read).
- `window.md:289`, written "when the window closes": true (`main_window.py:444-456`), but a crash or kill saves nothing.
- `getting-started.md:234`, "the detail pane shows its manifest and its run's log": omits the three-of-thirteen list cut and the 1 MiB log tail. Not re-checked here.
- `docs/guide/README.md:39-40`, "a run started from the window and the same run started from the tool do the same thing": not so for a TSNPE round's network, the resume policy, the epoch ceiling or the device (`window.md:198-208`).

The note's citations into the pages hold at the stated lines: `window.md:55-59`, `architecture.md:667-668`, `retrain.md:93-98`.

### 6. Pages a redesign must rewrite

- **`window.md`**, all 352 lines:
  - Screens `:11-133`, with The Parameter Inference tabs `:34-65`, One run at a time `:67-83`, The dialogs `:85-113`, What a run shows `:115-133`.
  - The inference settings `:135-222`.
  - Settings that are not speed dials `:224-259`.
  - The Fisher settings on a resumed run `:261-283`.
  - What the window remembers `:285-330`.
  - Supported models `:332-352`.
- **`getting-started.md`**: Launching PRISM `:8-32`; A first run in the window `:81-110`; the "written by the window" column `:157-166`; outputs outside the kinds `:171-176`; the window names of the two acceptances `:219-223`; Browsing the store, "In the window" `:229-249`.
- **`docs/guide/README.md`**: Two front ends `:24-40`; the reading-path and page-list lines `:52-56`, `:75-77`, `:123-124`.
- **`architecture.md`**: the `core/gui` module map `:170-240` (one line per file); The window `:620-692`; window mentions in Stages and compositions `:354-359`, `:382-385`, `:393-394`. The rest of the page was not read.
- **`rules-and-traps.md`**:
  - The window: threads, figures and paths `:245-294`.
  - Progress, the solver meter and cancellation `:296-343`.
  - Settings persistence and the inference screen `:345-398`.
  - Simulate and video export `:400-425`.
  - Labels and units `:427-441`.
  - The builder guard `:462-464`.
  - Rule paragraphs at `:90-100`, `:111-133`, `:145-146`, `:171-188`, `:221`, `:571-575`, `:683-687`.
- **`recordings.md`**: Defining a model of your own `:410-464`; six links into `window.md` at `:133`, `:162`, `:202`, `:324`, `:414`, `:464` (rest of the page not read).
- **`docs/checklists/display-walkthrough.md`**: new rows only; earlier rows are never edited.

Tests the pages say read them:

- `tests/test_docs.py` (`testing.md:100-102`): every relative link and heading anchor in the guide and the root `README.md` resolves; every page opens `# Title` then `Checked against commit <hash>.`; the command-line page matches the parser. All ten page names must exist (`tests/test_docs.py:14-15`).
- `tests/test_source_hygiene.py::test_no_scanned_file_cites_a_working_document` scans every guide page (`rules-and-traps.md:193-196`).

Because anchors are checked, renaming a `window.md` heading breaks inbound links: 38 anchored links from eight pages use `#screens`, `#the-parameter-inference-tabs`, `#one-run-at-a-time`, `#the-dialogs`, `#what-a-run-shows`, `#the-inference-settings`, `#settings-that-are-not-speed-dials`, `#the-fisher-settings-on-a-resumed-run`, `#what-the-window-remembers` and `#supported-models`.

## Earlier designs

## 1. Decisions that shaped the window

Sources, all under `C:\Users\J\PycharmProjects\PRISM\docs\superpowers\specs\`: **P2** = `2026-09-11-one-flow-design.md` (only §1.1 and §5, for D7/D8), **P3** = `2026-09-15-validation-and-logging-design.md`, **P4** = `2026-09-17-gui-usability-and-artifact-browser-design.md`, **P5** = `2026-09-22-secondary-analyses-design.md`, **P6** = `2026-09-25-documentation-and-retrain-readiness-design.md`. `core/` was not read, so every verdict below is spec against note, not code.

| id | decided | reason given | protects against |
|---|---|---|---|
| D7 (P2:34, :49, :1266) | Near-miss dialog only when a sibling cache differs in exactly one field (a `truncation`-only difference is exempt); Cancel is the default | Enter took the destructive path | A fresh cache one setting from a committed one (final review M5) |
| D8 (P2:35, :819-825, :857-874) | "Load a NON-AMORTIZED posterior?" dialog, Cancel default; a round just run installs with no dialog; unpersisted "Run on a different observation" box | A narrowed posterior is valid only near its observation | Silent misuse of a TSNPE posterior |
| P2 §5.3/§5.6 (:832, :884) | The window passes nothing for `resume`, `checkpoint_every`, `max_num_epochs`, `name`, `note`, device | "today's behaviour" | Nothing named: an omission, not a guard |
| V1 (P3:48) | Every stage works on a private copy; the session config is never written | One shared `SimConfig` leaked between runs (P3:14-22) | Fisher anchored on a leaked truth; a 3-probe bench run changing the next simulated run |
| V2 (P3:49, :67) | A blank box is a refusal at the click; no clamps; live derived lines never open a dialog | Blanks read as 0 (P3:27-31) | `math.log(0)` after the simulation was spent |
| V3 (P3:50, :316-334) | Yellow "Check your inputs" box (no traceback, fix sentence from `core/gui/fields.py`); red box only for bugs; the default button is the safe one | Refusal and crash shared one red dialog | — |
| V4 (P3:51, :388-393, :411-419, :75) | Run-scoped handler on the `core` logger with a cancel checkpoint in `emit`; `log.txt` written only at artifact commit; epoch counter and bars never in the file; the cache gets no log | Severity by channel, not message | Consequence stated: an interrupted process's records are lost |
| V5 (P3:52, :445-451) | Remember selections and budget only; science knobs (including `flow_hidden`, `flow_transforms`) open at `config.py`; consents never; chi drive/band read-only | Saved settings overwrote the science file (P3:36-40) | Stale `PRISM.ini` chi keys (P3:350-352); the five-day incident is not named in this spec |
| VRAM ceiling (P3:68, :312) | Not persisted, applied on every keystroke, outside the refusal scheme | "Unchanged" | Reason for non-persistence: not established in these specs |
| V6 (P3:53, :504-511) | Three budget lines come from one preview using the run's own resolution; runs at launch on hardware defaults | The line and the run can never disagree | — |
| B1 (P4:47) | Artifacts is a fifth Home tile, a plain `QWidget`, not a `BasePanel` | A `BasePanel` is greyed by any run | Not being able to read a log while training |
| B3 (P4:49; P5:673-676) | `finished` is separate from `complete`; FDT pickers filter on `finished` | `complete` means "has a manifest" | Half-written records offered as results |
| B4 (P4:50, :321) | Detail pane is text only: manifest plus the last 1 MiB of `log.txt` | Owner's choice | — |
| B6/B7 (P4:52-53; row 16, :930) | Delete one at a time, no `force`; dependents give a refusal with no Yes; Note/Delete/Sweep refused while a run is live; sweep removes only manifest-less directories, bound to the list shown | A confirmation that could only fail | A review probe deleted a real calibration |
| B11 (P4:57, :501-502) | Run shown app-wide (header button, tile and tab markers); navigation stays free | One-task-at-a-time was invisible | — |
| B12 (P4:58, :520-525) | Apply shows what the session holds and confirms before discarding, default not applying | Two-entry-point trap (M1b) | Silent loss of session work |
| B13 (P4:59, :539-548) | Narrowed marker in the closed picker; read-only line under each store picker describing the selection | Facts were tooltip-only | — |
| B14 (P4:60; row 15, :929) | Root handler from start-up; library lines shown once, prefixed; streams resolved at emit | `basicConfig` fired by sbi's `logging.warning` (P4:562-575) | Doubled, then missing, pane lines |
| B15 (P4:61; rows 3, 19) | Cancel checkpoints deferred while an exception is unwinding and during the checkpoint commit | Cancel raised inside the rescue block (P4:631-645) | Up to 49 batches lost; "told you cancelled a run that crashed" |
| P4 §1.2 (:82-93; row 21) | Every dialog is an instance `QMessageBox`; header run slot built empty | Static calls escape the test guard | A hung offscreen suite; the 2026-09-11 taskbar-icon incident |
| E1/E2 (P5:97-98, :121-131) | FDT and sweep runs write progressive records; the front end creates the writer, the stage enters it | The run log is thread-local | A cancelled run leaving nothing |
| E6 (P5:102, :629-643) | Fix-hint table names screens as well as tabs; `cell`, `model`, `t_obs` list every place | — | A bad cell on FDT pointing at the Infer tab |
| E7 + row 5 (P5:164-169, :693, :1057) | Seed, Record name and Note boxes on FDT and sweep screens; none remembered | — | Every run silently repeating the last seed |
| H7/H8 (P6:95-96, :173) | `probes` is command line only; measuring never changes what trains | Owner's scope | — |
| P6 §5.4 (:905-906) | Validate tab draws and records a seed; no Seed box | — | — |

### 2. Rejected then, and why

- **Figures in the browser, an open-folder button, "use this" jumps**: declined by the owner 2026-09-17; "additive later" (P4:97-98).
- **A session summary at quit**: declined in favour of Save and the lineage report (P4:99).
- **Rename from the browser**: it stays on the Prior and Posterior tabs; rename on a cache is unsound (P4:100-104).
- **A "tidy unnamed" action**: `unnamed()` returns every cache, so it would offer committed training rows (P4:105-107).
- **`force` delete, and a confirmation for an artifact with dependents** (P4:52, :338-345).
- **Redefining `complete`** (P4:152-155); **a presentation-aware describer in the store** (P4:71-76).
- **A catch-time `__context__` walk for cancel**: would report a GPU crash as "Run cancelled" (P4 row 3, :906).
- **A name-based drop of `core` records at the root** (P4 row 15).
- **A multi-select picker**: comparison controls hold a list instead (P5:786-787).
- **A second writer for progressive records** (P5:115-120); **widening the `units` hint** (P5:636-639).
- **A window front end for probes** (P6:173); **a scripted runbook** (P6:89, :108-109).
- **A device control** (P3:81); **the resume policy in the window** (P2:832); **a truth-anchored Fisher control**, "later, if ever" (P3:90).
- **qfluentwidgets** (GPLv3) and **a hard process cancel**: named only (P6:1101); the reasoning is in the archived handoff, not established here.

### 3. Deviations and things handed on

**Deviations that changed window behaviour**
- An unfinished cache shows batches, never rows (P4 row 2).
- The stale-folder notice is suppressed for caches (P4 row 5).
- Sweep skips directories modified in the last 300 s, the only cross-process guard (P4 row 16; P5:406-411).
- A run that finishes before a pending cancel is reported as success with a line saying so (P4 row 19).
- A typed zero refuses; a blank probe box is "blank" (P4 row 22).
- Budget refusals name "Posterior or TSNPE" (P3 row 1).
- The worker strips tracebacks before emitting, because held frames kept GPU tensors and a retry could run out of memory (P3 row 10).
- A one-sided sweep failure is a pane error line, not a yellow box (P5 row 21).
- Figures re-open only on the user's own pick (P5 row 22).
- One figure watcher per record (P5 row 25).
- The delete confirmation for an unfinished `fdt` row warns it may be being written elsewhere (P5 row 36).
- The Infer tab logs the posterior summary; Validate's closing line names the verdict (P6 rows 5, 6).

**Handed on, unowned**
- A cross-process lock: "nobody yet" (P5:204).
- Cache rename unsoundness (P4:100-104).
- Qt's C++ diagnostics are uncaptured (P3:92).
- The Hopf cell is refused at the default band, with no front-end `freq_bounds` (P5 row 16).
- The runbook concedes a window retrain "would need them typed into the Posterior tab" (P6:1059-1060).

**Note claims against the specs**
- *By design, confirmed:*
  - Network size reopens at 128 × 8 (V5; walkthrough C6, P3:713).
  - Resume is always auto, with no epoch, cadence, name, note or device control (P2:832, :884; P3:81).
  - One-setting dialog with Cancel default (D7).
  - Ceiling not persisted, applied per keystroke (P3:68).
  - The line under a picker describes the selection, not the session (B13); "the prior picker loads nothing until Build/Load" was carried as a known fact (P6:1099-1100).
  - `log.txt` only at commit, epoch counter in no file (P3:411-419).
  - Pane is text only with a 1 MiB tail (B4).
  - Only priors and posteriors are renamable (P4:100).
  - No Validate seed box (P6:905-906).
  - Probes, ablation, rotation and `sbc` are tool-only (H7; P3:72).
  - TSNPE tab has no network boxes (P3:445-449; P6:1071-1072).
  - A round just trained carries no load acceptance (P2:825; P6:979-983, row 34).
- *Overstated:* "the whole window is locked". Navigation and all reading of the Artifacts screen stay free by design (P4:501-502, :355-358).
- *Wording to check in code:* the note's "Keep this session" button; the spec fixes only Yes/No with No as default (P4:522-525).
- *Not established in these specs:* the 5,000-line pane, the 45 s stall notice, the "~1 min" Cancel text, the close dialog and its default, "Quit anyway", the 12-to-3 list abbreviation, the Bounds stash/overwrite, and the ceiling lock during a run. If the close dialog's default is "Cancel task & quit", it contradicts the stated safe-default rule (P3:334; P2:1266).

### 4. Principles the window was built to

- Selections and budget remembered, nothing else; consents, seed, name and note never (P3:52; P5:693, :1057).
- Refusals carry the rules, and the front-end table names the control (P4:78-80; P3:227-235).
- The safe button is the default (P3:334; P2:1266).
- Every dialog is an instance dialog (P4:82-86).
- The Artifacts pane is text only; no force delete; one artifact at a time; reading is never blocked (P4:50, :52, :355-358).
- One run app-wide; the lock stands; navigation stays free (P4:501-502).
- Nothing at start-up that lays out or scans (P4:88-93).
- One formatter per fact, per front end (P4:71-76; rows 8, 13).
- Streams resolved at emit, never at construction (P4:600-609).
- A live line never opens a dialog (P3:67).
- "No new control is added" (P3:565); "Nothing new is persisted" (P2:769).
- Reports are files, never a store kind (P4:56).
- The owner's walkthrough rows are claimed by no task (P4:858-859).

### 5. What a redesign must ask before undoing

1. **V5 (science knobs reopen at `config.py`).** Remembering network size trades the forgotten-256×10 trap for a stale remembered value training for days unseen.
2. **B15 and V4 cancel checkpoints.** Any new log route (a run file, a new handler) that checks cancel between checkpoint steps 1 and 3, or during unwind, loses batches or reports a crash as a cancel.
3. **The app-wide one-run lock.** The stream swap and the run buffer are process-wide and rely on it (P3 row 14); unlocking controls mid-run mixes two runs' lines and records.
4. **B6/B7.** A force delete, or deleting during a run, can remove a prior a 10 GiB cache needs; a wider sweep already deleted a real calibration once.
5. **D7/D8 Cancel defaults and unpersisted consents.** Enter starts a fresh multi-day cache or loads a narrowed posterior.
6. **B1.** Making the browser a panel greys it for the whole run.
7. **Worker traceback stripping.** An error view that keeps the exception re-creates the retry out-of-memory failure.
8. **V2.** Clamps or defaults for blanks return the crash after the spend.
9. **Pickers filtering on `finished`.** Otherwise unfinished records are offered as results.
10. **Seed, name and note not remembered.** Otherwise silent repeat runs, and a taken name refused at the next launch.
11. **Instance dialogs and the empty start-up slot.** Otherwise a hung test gate and the lost taskbar icon.
12. **V1.** A screen that reads stage writes off the session config, not the artifact, re-opens the leak.

## What the tests pin about the window

### Tests nav and gating

**File:** `C:\Users\J\PycharmProjects\PRISM\tests\test_nav_and_gating.py` (5,097 lines, 101 `def test_`; 8 are parametrised, so 119 collected items by arithmetic — pytest was not run, so the collected number is not confirmed).

**What it pins and how.** Almost nothing is tested through layout geometry; the tests build widgets off-screen and reach in by attribute name. 37 functions construct `InferenceScreen()` standalone and 2 more get one from the `screen_run` fixture (conftest.py:81-102); 24 construct `FdtPanel()`/`CrossValPanel()` with no arguments; 9 build `MainWindow()`; 2 build `NavShell()`. The usual shape is: set box text via an attribute (`pp.flow_hidden.setText`), rebind `panel.dispatch` and `panel._refusal` on the instance, call the click's private slot (`_build_posterior`, `_infer`, `_run`, `_round`, `_validate`, `_build_config`), then assert on what was dispatched or refused. That is behaviour, and it protects runs: refusal at the click, knobs as arguments, consents (`new_run`, `other_obs`, the non-amortized load) never persisted, the session config left pristine, Apply's confirmation.

Three other kinds of pin sit on top:
- **Dialogs.** 20 functions read `QMessageBox` instances off `SHOWN` (conftest.py:188); 13 assert the title "Check your inputs" with `text`/`informativeText`/`detailedText`, 3 assert "Error". Inline validation or a custom dialog class breaks all of them; a non-QMessageBox modal would stall the suite rather than fail it (conftest.py:172-175).
- **Navigation and tabs.** 12 functions pin the shell: tab titles in order, tab indices, `nav.stack` index 0 = Home, home tiles by section name, the run-slot destination map, objectName/QSS, QFormLayout label rows.
- **Source text.** 15 functions assert on `inspect.getsource`/AST; 6 of them read the code that BUILDS a screen (`PriorPanel.__init__`, `ValidatePanel.__init__`, `TSNPEPanel.__init__`, `InferPanel.__init__`, `FdtPanel._build_controls`, `NavShell.__init__`) and 2 read docstring prose that counts panels and tabs.

**What a rebuild does.**
- If class names, attribute names, labels, tab titles and QMessageBox refusals are kept and only the arrangement changes: my estimate is 86 of 101 functions pass unchanged (104 of 119 items). 7 of those 86 are not window tests at all (overlay/visualizer/orchestrator, lines 916-1081 and 1088).
- 15 functions collide with any new navigation or rebuilt constructors: lines 275, 290, 300, 324, 1492, 1944, 3373, 3629, 3989, 4238, 4267, 4303, 4355, 4720, 4751.
- The 86 are conditional on the de facto interface listed below (about 190 attribute names across 10 panel classes plus the shell). Renaming `InferenceScreen` or its six `*_panel` attributes alone breaks 39 functions at setup.
- Renamed tab titles break the two control-table tests and 8 more that quote a tab name inside a fix sentence.
- Lines 34-39 import six `core.gui` modules at module level; five of those lines import names the file never uses, so moving `core.gui.streams`, `vt`, `widgets.log_pane`, `widgets.progress_pane` or `worker` fails all 101 at collection.

**Not in this file:** no assertion on the number of Home tiles, no splitter sizes, and no table of remembered settings keys (only three "not persisted" source scans and two QSettings keys).

| file | lines | tests | touches the window | survive a relayout | pin wording | pin structure | pin source text |
|---|---|---|---|---|---|---|---|
| `tests/test_nav_and_gating.py` | 5097 | 101 | mostly | 86 | 56 | 28 | 15 |
| `tests/conftest.py` | 198 | 0 | in a few places | 0 | 0 | 0 | 0 |
| `tests/_fixtures.py` | 818 | 0 | in a few places | 0 | 0 | 0 | 0 |

Notes per file:

- `tests/test_nav_and_gating.py`: COUNTS ARE TEST FUNCTIONS (Grep count of ^def test_ = 101). Parametrised:
  2099-2101 (4x2=8), 2136 (2), 2325-2327 (2x3=6), 2410 (2), 2454 (2), 2741 (2), 2835 (2), 2920 (2) -> 119
  collected items by arithmetic; not confirmed, pytest not run. None of the 15 non-survivors is parametrised, so
  104 of 119 items survive. NOT AT ALL window (7): 916, 961, 996, 1006, 1027, 1050, 1088 (core.SBI.overlay,
  Helpers.visualizers, orchestrator). Pure helpers in core/gui with no widget built (9): 133, 213, 266, 2541,
  2581, 2629, 3019, 3652, 4217. survive_a_relayout = 86 means 'would pass unchanged if class names, attribute
  names, labels, tab titles, QMessageBox refusals and the figure stack's tab API are all kept'. NOT surviving
  (15), by def line: 275 (NavShell back arrow), 290 (nav.stack index 0), 300 (tab enabled vector by index 0-4),
  324 (tabs.count()==6, tabText(5)), 1492 (ten tab titles in order + SCREENS + None set), 1944 (QFormLayout
  read-back + source of FdtPanel._build_controls), 3373 (source of PriorPanel.__init__, nine rows in order),
  3629 (source of ValidatePanel/TSNPEPanel.__init__), 3989 (QFormLayout read-back over every place + source of
  InferPanel.__init__), 4238 (objectName/QSS/source of NavShell.__init__), 4267 (run slot text/tooltip/timer),
  4303 (_panel_home tuples, tab index), 4355 (tab titles, tile texts, marker), 4720 (docstring counts), 4751
  (MainWindow source). Of the 86 survivors, 7 still carry a source-text pin on a NON-layout method or file: 769,
  1439, 1450, 1701, 1783, 3063, 4859. pin_wording = 56 counts any assertion on text the window authors (labels,
  tab titles, dialog titles/buttons, fix sentences, tooltips, status lines, panel log lines, help text): 38
  assert a whole string, 18 only a fragment. Fragments of CORE refusal sentences ('is blank', 'at least 1',
  '(default N)') are not counted; where a sentence is authored (core or core/gui) I cannot tell from this file
  alone for every case. 13 functions assert the dialog title 'Check your inputs' (1681, 1741, 1773, 1850, 1881,
  1898, 1938, 2093, 2131, 2176, 2318, 2362, 2446, 4977, 4986, 5018, 5093), 3 assert 'Error' (1694, 1752, 1864).
  pin_structure = 28 in three groups: (a) 13 pin the shell, tabs, form rows, child lookup or object names: 275,
  290, 300, 324, 1492, 1944, 2836 (findChild(QPushButton) at 2873), 3373, 3989, 4238, 4267, 4303, 4355; (b) 9
  pin which tab a control is on through a literal in a sentence: 1641 (1684-1685), 1783 (1884), 1911 (1941),
  2137 (2179), 2411 (2450), 3478 (3523), 3552 (3614), 4992 ('above' in help text, 5031), 5035 (5056-5063); (c) 6
  read the figure stack as a tab widget (count/tabText/widget(i)/tabCloseRequested/currentIndex/png_path): 2650,
  2742, 2881, 2921, 2981, 4859. Not counted as structure but order-dependent: combo index pins -- infer_mode
  0=simulated/1=experimental (726, 1172, 1194, 3706, 3715, 3772, 3841, 3908, 4078) and index 0 = '(from
  scratch)' (543, 601, 4636, 4642). pin_source_text = 15: 324 (365-380), 769 (789-797), 1439, 1450 (1486-1489),
  1701 (1777-1780), 1783 (1902-1908), 1944 (1976-1978), 3063 (3118-3119), 3373, 3629, 3989 (4067-4071), 4238
  (4257), 4720, 4751, 4859 (4915-4920). Stubbing seams the tests rely on: instance rebinding of panel.dispatch
  (about 35 functions), panel._refusal (13 functions: 513, 677, 837, 3121, 3208, 3396, 3478, 3552, 3664, 3745,
  3815, 3889, 3939), panel._on_error, picker.selected / .key / .selected_path, panel.figure_stack.add_png, and
  log_pane.append_line/append_lines via PaneCapture. Module-level imports at 34-39 are a collection-time pin
  (five of the six lines import names used nowhere else in the file).
- `tests/conftest.py`: No test functions; ten fixtures. Three touch the window. screen_run (81-102, module
  scope) builds InferenceScreen(), calls install_config(cfg), sets session.inf_prior / session.posterior and
  calls refresh_gates(): it needs exactly those four names. _no_modal_dialogs (170-190, session autouse)
  replaces QMessageBox.exec with a recorder that appends to tests/_fixtures.SHOWN and returns 0; a dialog that
  is not a QMessageBox instance escapes it and stalls (172-175). _settings_home (131-153) and _isolated_settings
  (156-167) use core.gui.settings.use_ini_file, settings._override_path, settings.ORG and settings.APP. qapp
  (12-16) is not requested by test_nav_and_gating.py, which calls qt_app() instead. The other fixtures (store
  59-64, tiny_run 67-78, _sandbox_default_store 19-31, _no_sbi_logs 34-56, _checkpointing_off_unless_asked
  105-128, _clear_shown 193-198) do not touch the window.
- `tests/_fixtures.py`: No test functions. What test_nav_and_gating.py imports from it: qt_app (63-67), SHOWN
  (60), code_only (101-112; the helper behind every source-text pin: inspect.getsource -> ast -> docstrings
  stripped -> ast.unparse), PaneCapture (115-154; rebinds panel.log_pane.append_line and
  panel.log_pane.append_lines, so a panel must expose log_pane with those two methods and BasePanel.dispatch
  must resolve them at connect time, 135-136), _nad_cfg (157-166), _posterior_artifact (269-290),
  build_fdt_record (613-680), compare_preflight_refusals (683-746), snapshot_cfg (788-802), assert_cfg_unchanged
  (805-818); build_tiny_run (384-429) and install_sbitest (351-381) reach it through conftest's tiny_run. Only
  qt_app, SHOWN and PaneCapture touch the window; artifact_screen (749-761) builds ArtifactScreen(store=...) but
  is not used by test_nav_and_gating.py. pump (70-75) is not imported by that file.

#### The names the tests reach for

Names actually used in `tests/test_nav_and_gating.py` (Grep over `inf.`, `screen.`, `w.`, `window.`, `pp.`, `cfgp.`, `vp.`, `panel.`, `xv.`, `fdt.`, `reduction.`, `nav.`, `mb.`, `combo.`, `row.`, `stack.`, `lst.`, `box.`). One representative line each. Private names (leading `_`) are part of the interface the tests use.

### MainWindow — 13 names
`panel(cls)` (207, 2859, 4286), `nav` (298), `reduction_screen`, `fdt_screen`, `inference_screen`, `simulate_screen` (1531-1532), `artifact_screen` (2871, 4038), `model_builder_screen` (4039), `home_screen` (4377), `_run_timer` (4282, 4295), `_running_panel` (4283), `_panel_home` (4319), `_section_index` (4320, 4341).

### NavShell — 7 names (+2 module-level)
`add_screen` (282), `go_home` (283), `go_to` (285, 4343), `btn_back` (284), `stack` (298, 4329), `btn_running` (4250), `set_running` (4259). Module: `running_banner` (4222), `RUNNING_MARK` (4369).

### HomeScreen — 1 name (+1 module-level)
`tiles` (dict keyed by section name; 4377, 4421). Module: `greeting` (267).

### Section screens (fdt_screen, reduction_screen, simulate_screen) — 1 name
`tabs` (a QTabWidget: `tabText`, `count`, `widget`, `currentIndex`, `setCurrentIndex`, `isTabEnabled`; 1533, 4037, 4325, 4389).

### ArtifactScreen — 1 name
`store_changed` signal (2871, 2913).

### Model builder screen — 4 names
`vars_edit`, `_set_variables`, `_var_rows` (each row has `.drift`), `_detect_params` (4046-4049).

### InferenceScreen — 12 names
`tabs` (308, 339), `session` (310), `refresh_gates` (310), `new_draft` (619, 3143), `install_config` (633), `session_contents` (4450), `config_panel` (845), `prior_panel` (526), `posterior_panel` (413), `validate_panel` (460), `infer_panel` (634), `tsnpe_panel` (351).
Session object: `SbiSession(draft=, cfg=, inf_prior=, posterior=)` with `.observation` (523, 4460). `ConfigDraft(model=, labels=, state_dep_drift=)` (619) with `units_override`, `chi_mode`, `chi_f0`, `chi_freq_bounds`, `chi_n_freqs`, `chi_k_pad`, `chi_max_cycles`, `make_config` (857-861, 3102-3103, 3195-3205).

### ConfigPanel — 16 names
`model_combo` (846), `_on_model_changed` (847), `units_toggle` (849), `units_text` (853), `_units_override` (850), `_build_config` (855), `_refusal` (869), `chi_check` (3096), `chi_f0` (3089), `chi_range` (`.lo`, `.hi`, `.value()`; 3089, 3093), `chi_fixed_note` (3091), `chi_k`, `chi_pad`, `chi_cycles` (3148-3150), `session_line` (4454), `log_pane` (via PaneCapture, 3095).

### PriorPanel — 19 names
`prior_picker` (568), `bounds_source` (569), `bounds_picker` (570, 620, 1728), `cluster_size`, `cluster_samples`, `sweep_iters` (572-574), `sweep_batch`, `sweep_max_sets`, `sweep_step`, `sweep_units` (3279-3282), `sweep_note` (3310), `prior_name` (1670), `prior_line` (4644), `_build_prior` (576), `_save_prior` (1671), `_on_prior` (3305), `_refusal` (582), `dispatch` (527), `log_pane` (3369). Module `prior_tab`: `_KNOB_RULES`, `_NOTE_KEYS`, `_rule_words` (3330), `default_store` (1677).

### PosteriorPanel — 25 names
`post_picker` (483), `post_name` (421), `post_line` (4631), `btn_post` (604), `num_runs`, `run_size_cap` (1242-1243), `flow_hidden`, `flow_transforms`, `flow_lr`, `flow_patience`, `fisher_dz` (538-542), `fisher_m` (3445), `fisher_points` (3448), `_build_posterior` (484), `_save_posterior` (422), `_on_posterior` (430), `_ask_new_run` (1258), `_ask_load_non_amortized` (413), `_confirm_fresh_run` (3419), `_accept_for_load` (3464), `refresh_local_gates` (603), `_refusal` (537), `dispatch` (478), `_set_busy` (4401), `log_pane` (1285). Module `posterior_tab`: `default_store` (1677).

### ValidatePanel — 8 names
`cal_n`, `cal_scales` (529-530), `_validate` (473), `_on_calibration` (3546), `_refusal` (3499), `_on_error` (3500), `dispatch` (472), `log_pane` (3545).

### InferPanel — 27 names
`infer_mode` (726), `sim_tobs` (1176), `exp_tobs` (3720), `chi_tobs` (727), `chi_f0_si` (728), `chi_spont` (729), `exp_spont`, `exp_forced` (3716-3717), `cell_source` (1173), `cell_picker` (634, 1174), `values_grid` (3935), `other_obs` (1213), `_forcing_fields` (dict by forcing name, 3719), `_chi_forced_fields` (648), `_cell_problems` (1175), `_add_chi_probe` (650), `_remove_chi_probe` (656), `_plan_chi_probes` (3965), `_infer` (732), `_accept` (1469), `_on_observation` (1182), `_refusal` (722), `dispatch` (723), `_busy` (4091), `log_pane` (3957), `save_settings`, `restore_settings` (source read, 1486-1488). Module `infer_tab`: `_blank_frequency` (815). Probe row `rows._ChiProbeRow`: `path` (`.edit`), `freq`, `pair()`, `problems(i)` (670-674, 707).

### TSNPEPanel — 16 names
`obs_picker` (352), `obs_line` (4647), `btn_round` (354), `n_dirs`, `hpd`, `num_runs`, `run_size_cap` (1373-1376), `new_run` (1387), `_round` (1377), `_on_round` (415), `refresh_local_gates` (353), `_refusal` (3585), `_on_error` (1370), `dispatch` (1368), `save_settings`, `restore_settings` (source read, 1444-1446). Module `tsnpe_tab`: `default_store` (1371).

### FdtPanel — 27 names
`cell_picker` (1929), `record_picker` (1971), `record_line` (2686), `record_name`, `record_note`, `seed` (2024-2026), `n_freqs`, `ensemble_m`, `freqs_per_batch`, `f0` (by `getattr(panel, key)` at 2125, keys at 2099 — the attribute name must equal the refusal field key), `figure_stack` (2688), `compare_list` (4817), `compare_mode` (4822), `renorm_prefactor` (4830), `compare_name`, `compare_note` (4964-4965), `btn_compare` (4823), `_run` (1935), `_compare` (5026), `_on_record` (2209), `_show_record` (2685), `_busy` (2693), `dispatch` (2023), `log_pane` (2200), `save_settings`, `restore_settings` (4916), `_build_controls` (source read, 1976). Module `fdt_panel`: `_run_fdt_guarded` (158), `run_fdt` (156), `HELP["compare_list"]` (5031).

### CrossValPanel — 30 names
`cell_picker` (2396), `cell_values` (208), `preset_combo` (2260), `record_picker` (2534), `record_line` (2686), `record_name`, `record_note`, `seed` (2245-2247), `n_freqs`, `ensemble_m`, `freqs_per_batch`, `f0` (2125), `s_grid`, `t_grid` (by `getattr(panel, grid)` at 2354; each with `.lo`, `.hi`, `.points`, `.spec_or_none()`), `figure_stack` (2499), `compare_list` (4842), `slice_at` (4846), `compare_name`, `compare_note` (4847-4848), `btn_compare` (4849), `_run` (2248), `_compare` (5023), `_on_cell_changed` (2397), `_show_record` (2685), `_busy` (2693), `_set_busy` (4287), `dispatch` (2244), `log_pane` (2497), `save_settings`, `restore_settings` (4916). Module `crossval_panel`: `HELP["compare_list"]` (5032).

### ReductionPanel — 5 names
`cell_picker`, `dispatch` (1829-1830), `_run` (1831), `f0` (1877), `_on_error` (1893).

### SimulatePanel — 3 names
`cell_picker`, `dispatch` (1829-1830), `_start` (1832). Module `simulate_panel`: `build_stream_config` (1839).

### BasePanel (bare instance or throwaway subclass) — 5 names, plus one that must be absent
`figure_stack` (254), `dispatch(fn, *a, watch_dir=[...])` (255), `_set_busy` (4206), `_running` (class flag, 4208), `_busy` (4091). `hasattr(BasePanel, "_config_error")` must be False (1902). Module: `RUN_STATE.changed` (4204).

### Key widgets
- **StorePicker** (`prior_picker`, `post_picker`, `obs_picker`, `record_picker`) — 9 names: `combo`, `selected()`, `key()`, `restore_key()`, `refresh()`, `has_entries()`, `selection_summary()`, `_resolved_store` (class-level monkeypatch seam, 1969, 4609), `NEW_LABEL` (4617); constructor `StorePicker(kind, row_filter=, allow_new=)` (4695-4715); its first `QPushButton` child is clicked as Rescan (2873). Module: `NARROWED_SUFFIX` (4582).
- **File pickers** (`bounds_picker`, `cell_picker`) — 3 names: `combo`, `base_path`, `selected_path()`.
- **Source toggles** (`units_toggle`, `bounds_source`, `cell_source`) — 2 names: `set_direct()`, `is_direct()`.
- **Path rows** (`chi_spont`, `exp_spont`, `exp_forced`, probe `row.path`) — 1 name: `edit`.
- **Number boxes** — `setText`, `text`, `value`, `value_or_none`, `isReadOnly`, `insert` (750-754, 2168, 3090).
- **Figure stack** — 10 names: `add_png`, `add_figure`, `count`, `tabText`, `widget`, `tabCloseRequested`, `setCurrentIndex`, `currentIndex`, `clear_all`, `png_path` (254, 2688-2692, 2803-2811, 2865-2866, 2972).
- **CompareList** — 5 names: `list` (QListWidget; items carry the record id under `Qt.UserRole`, 4818), `btn_add`, `ids()`, `add_selected()`, `remove_selected()` (4788-4798, 4896).
- **Log pane** — 3 names: `append_line`, `append_lines`, `toPlainText` (3369, 3545; `_fixtures.py`:142-143).
- **BoundsGrid / ValuesGrid** — `problems()`, `load()`, `to_dicts()`, `_rows[(kind, name)].lo/.hi` (891-911).
- **Dialogs as read off SHOWN** — `windowTitle()`, `icon()`, `text()`, `informativeText()`, `detailedText()`, `defaultButton()`, `buttons()` (1681-1686, 4529-4549).

### Other core.gui module names imported
`fields.CONTROL`, `SCREENS`, `label`, `fix_sentence`, `_where` (1515-1612); `design.build_qss` (4251); `settings.settings()` (293); `record_view.record_summary`, `record_figures` (2551, 2636); `plot_watcher.NewPngWatcher` (`png_ready`, `start`, `stop`; 219-232); `help_text.HELP` keys `chi_f0`, `chi_range`, `cal_scales`, `num_runs` (3116, 3659-3660); `labeled_inputs.FloatField`, `help_badge.HelpBadge`, `field_row.LabeledFieldRow`, `param_grid.BoundsGrid`/`ValuesGrid`; the package path `core.gui.panels.inference` and module `core.gui.panels.inference.base` (1715, 4736). At module level (34-39): `panels.base_panel.BasePanel`, `streams.redirect_streams`, `vt.StreamRouter`, `vt.parse_bar`, `widgets.log_pane.LogPane`, `widgets.progress_pane.ProgressPane`, `worker.WorkerSignals`.

#### Tables and fixtures

## Tables

**1. The table that names each control for a refusal's field key.** It lives in `C:\Users\J\PycharmProjects\PRISM\core\gui\fields.py:55-187` (`CONTROL`): 109 keys by my count — 54 `(place, label)` tuples, 12 sentences, 43 `None`. The test file does not hold a copy; it pins it in pieces:
- key set equals `core.refusals.FIELDS` in both directions — test_nav_and_gating.py:1520-1522
- every tuple's place is a tab title or a screen, its label is non-empty, and `fix_sentence(key) == f"Set it in the '{text}' box on {_where(tab)}."`; sentence entries end with a period and have no `label()`; `None` entries render "" — 1540-1560
- sentences verbatim: `t_obs`, `n_directions`, `accept_other_observation`, `new_run`, `accept_truncated`, `recording_spont`, `recording_probe`, `name`, `chi_f0`/`chi_freq_bounds`, `run_size_cap`, `cell`, `model`, `n_freqs`, `ensemble_m`, `artifact`, `note` — 1567-1617; `compare_records`, `prefactor`, `slice_at` — 5053-5063
- single entries: `num_runs == (("Posterior", "TSNPE"), "Batches")` — 1588; the three drive entries and their labels "A (N)", "f (Hz)", "φ (rad)" — 1633-1638; the nine Prior keys each `== ("Prior", label(key))` — 3393
- `_where` phrasing for screens and mixed places — 1609-1612
- read-back of every tuple entry against the QLabel text of a QFormLayout row on the named place — 4053-4065

**2. The pinned set of settings with no control** — test_nav_and_gating.py:1620-1630, 43 keys asserted as exactly the `None` entries:
- six the window never exposes (1621): `checkpoint_every`, `resume`, `device`, `n_samples`, `num_posterior_samples`, `max_num_epochs`
- eleven tool-only diagnostics knobs (1622-1623): `repeats`, `n_points`, `n_worst`, `top_n`, `m`, `m_noise`, `rel`, `min_valid`, `rows`, `n_sweep`, `chi_k_fixed`
- twenty-one probe-check keys (1625-1628)
- five FDT settings (1630): `freq_bounds`, `burn_in_nd`, `t_obs_periods`, `dt_nd`, `psd_t_obs_nd`
Giving any of these a control in the window fails this assertion until the set and `fields.py` are edited together.

**3. Tab titles and screens.**
- all ten tab titles, in section order reduction → FDT → inference → simulate — 1531-1536: "NWK → Hopf reduction map", "FDT analysis", "Sweep study cross-validation", "Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE", "Live simulation"
- `SCREENS == frozenset({"Artifacts", "Model Builder"})` — 1537 (source: fields.py:45)
- the six inference tab titles again — 4386-4387, 4411-4412; the FDT section's two — 4375-4376
- tab title → panel attribute map for the six inference tabs — 4015-4016
- section names: "FDT Analysis", "Parameter Inference" (4319-4322), "Simulate" (4341), also as Home tile keys (4377, 4385, 4404)
- inference tab indices in comments and assertions: 0-4 = Config, Prior, Posterior, Validate, Infer (308); 5 = TSNPE (339); Posterior = 2 (4322, 4403); cross-validation = FDT tab 1 (4320, 4330)

**4. Remembered keys.** This file holds no list of remembered settings keys. What it has:
- three source scans that a control is NOT persisted: TSNPE `new_run` (1444-1447), Infer `other_obs` (1486-1489), the six comparison controls on both measurement panels (4915-4920)
- two QSettings keys written by tests: `window/tab` as a stale key that must be ignored (294), and `<fdt|crossval>/record` (2779-2783, 2901-2905)
Where a full list lives I cannot tell from this file; line 4575 points at `tests/test_settings_persistence.py`.

**5. Attribute ↔ field-key tables inside tests** (each row ties a widget attribute name to the refusal key it must raise):
- Config: `chi_k`→`chi_n_freqs`, `chi_pad`→`chi_k_pad`, `chi_cycles`→`chi_max_cycles` — 3157-3168
- Prior: `cluster_size`→`min_cluster_size`, `cluster_samples`→`min_samples`, `sweep_iters`→`num_iterations`, `sweep_batch`→`sweep_batch`, `sweep_max_sets`→`max_sets`, `sweep_step`→`walk_step`, `sweep_units`→`stability_units` — 3259-3267
- Posterior: nine boxes → nine keys — 3434-3449
- Validate: `cal_n`→`n_cal`, `cal_scales`→`cal_n_scales` — 3509-3513
- TSNPE: `num_runs`, `run_size_cap`, `hpd`→`hpd_level`, `n_dirs`→`n_directions` — 3599-3606
- FDT/CrossVal: attribute name equals key for `n_freqs`, `ensemble_m`, `freqs_per_batch`, `f0` (2099) and `s_grid`, `t_grid` × `lo`, `hi`, `points` (2325-2326)

**6. Row-source tables** (keys each builder must pass through `label(key)`): Prior's nine in order (3390-3391); Validate's two and TSNPE's five (3640-3642); Infer's four (4068); FDT's seven (1977).

## Fixtures and helpers

**In `tests/conftest.py`**
- `store` (59-64, function): a fresh `ArtifactStore` under `tmp_path` installed as process default. Used at 382, 434.
- `tiny_run` (67-78, module): a real SBITEST prior + posterior via `build_tiny_run`; minutes on CPU.
- `screen_run` (81-102, module): `InferenceScreen()` + `install_config(truth-free copy)` + `session.inf_prior`/`session.posterior` + `refresh_gates()`; returns `SimpleNamespace(screen, run, cell)`. Used at 4100, 4151.
- `_no_modal_dialogs` (170-190, session, autouse): stubs `QMessageBox.exec` to append the box to `SHOWN` and return 0.
- `_clear_shown` (193-198, function, autouse): empties `SHOWN` before each test.
- `_settings_home` (131-153, session, autouse) and `_isolated_settings` (156-167, function, autouse): a temp `.ini` through `core.gui.settings.use_ini_file`; teardown asserts the real settings file is byte-identical.
- `_sandbox_default_store` (19-31), `_no_sbi_logs` (34-56), `_checkpointing_off_unless_asked` (105-128): session autouse, no window contact.
- `qapp` (12-16, session): not requested by this file.

**In `tests/_fixtures.py`**
- `qt_app()` (63-67): the one `QApplication`; called in nearly every window test.
- `SHOWN` (60): the list of dialogs shown.
- `code_only(obj)` (101-112): source with comments and docstrings stripped; behind the source pins.
- `PaneCapture(panel)` (115-154): replaces `panel.log_pane.append_line` and `.append_lines`, recording `(level, text)`.
- `_nad_cfg` (157-166): a real NADROWSKI `SimConfig` off the master bounds file.
- `_posterior_artifact` (269-290): a posterior artifact with a fake payload.
- `build_fdt_record` (613-680): one `fdt` record with manifest, `data.h5` and one figure.
- `compare_preflight_refusals` (683-746): eleven comparison refusal cases over records it writes.
- `snapshot_cfg` / `assert_cfg_unchanged` (788-818): config-unchanged check.

**Local to `tests/test_nav_and_gating.py`**
- stand-ins: `_prior_stub` (47), `_posterior_stub` (53)
- stub configs that put the Infer tab on a page: `_chi_cfg` (59), `_forced_cfg` (78), `_spont_cfg` (100), `_chi_built_cfg` (123)
- `_saved_fdt_record` (2708): a finished `fdt` record with one PNG per title
- `_choose(combo, row)` (2721): `setCurrentIndex` then `activated.emit` — a user's pick
- `_tabs(panel)` (2730): the figure stack's tab titles
- `_settle(app)` (2734): three `processEvents`
- `_pick_simulated(panel, cell, t_obs)` (4074): sets `infer_mode` index 0, `cell_source`, `cell_picker.selected_path`, `_cell_problems`, `sim_tobs`
- `_wait_for_run(app, panel, limit)` (4085): spins until `panel._busy` drops

**What the tests stub on the window**
- per instance: `panel.dispatch`, `panel._refusal`, `panel._on_error`, `picker.selected`, `picker.key`, `cell_picker.selected_path`, `bounds_picker.selected_path`, `figure_stack.add_png`, `values_grid.problems`, `cell_source.is_direct`, `_ask_new_run`, `_ask_load_non_amortized`, `_confirm_fresh_run`, `_accept_for_load`, `inf.install_config`, `screen.new_draft`
- per class or module: `StorePicker._resolved_store` (1969, 4609, 4692), `default_store` in `prior_tab`, `posterior_tab`, `tsnpe_tab` (1677, 1371), `fdt_panel.run_fdt` (156, 2954), `simulate_panel.build_stream_config` (1839), `QMessageBox.exec` (1306, 4540)

#### Pins a redesign collides with

- CERTAIN — tests/test_nav_and_gating.py:1531-1536: reads tab titles off window.reduction_screen / fdt_screen /
  inference_screen / simulate_screen `.tabs` and asserts the ten titles in this exact order: 'NWK → Hopf
  reduction map', 'FDT analysis', 'Sweep study cross-validation', 'Config', 'Prior', 'Posterior', 'Validate',
  'Infer', 'TSNPE', 'Live simulation'. Each of the four sections must be a MainWindow attribute owning a
  QTabWidget called `tabs`.
- CERTAIN — tests/test_nav_and_gating.py:1537-1560: SCREENS must equal frozenset({'Artifacts', 'Model
  Builder'}); every (place, label) entry of core/gui/fields.py CONTROL must name one of those tab titles or
  screens, and its sentence must read "Set it in the '<label>' box on the <place> tab/screen." Any new place (a
  sidebar page, a step of a wizard) or a control that is not a 'box on a tab' has no shape in this grammar.
- CERTAIN — tests/test_nav_and_gating.py:1567-1617: fix sentences verbatim, each naming tabs — t_obs 'on the
  Infer or Live simulation tab', cell on five tabs (1593-1595), model on three (1596-1597), run_size_cap 'on the
  Posterior or TSNPE tab' (1589-1590), name and note naming the 'Save box', 'Record name', 'Comparison name',
  'Note box on the Artifacts screen' (1582-1584, 1615-1617), new_run naming 'the dialog on the Posterior tab'
  and the TSNPE tick box (1573-1575). Moving or renaming any of these controls changes a pinned sentence.
- CONDITIONAL — tests/test_nav_and_gating.py:1620-1630: the 43 keys with no window control are pinned as an
  exact set, including checkpoint_every, resume, device and max_num_epochs. If the redesign gives the window a
  control for any of them, this assertion fails until the set is edited. Whether the redesign will add one I
  cannot tell.
- CERTAIN — tests/test_nav_and_gating.py:308-322: the session gates are asserted as `[inf.tabs.isTabEnabled(i)
  for i in range(5)]` in the order Config, Prior, Posterior, Validate, Infer, through five session states. The
  gating behaviour is worth keeping; its expression as tab-enabled flags by index is not layout-neutral.
- CERTAIN — tests/test_nav_and_gating.py:339-344: `inf.tabs.count() == 6 and inf.tabs.tabText(5) == 'TSNPE'`,
  and TSNPE gating read as `isTabEnabled(5)`.
- CERTAIN — tests/test_nav_and_gating.py:298: `w.nav.stack.currentIndex() == 0` — the window opens on Home and
  Home is index 0 of a stacked nav.
- CERTAIN — tests/test_nav_and_gating.py:280-288: NavShell's model — add_screen, go_home, go_to(2), and btn_back
  hidden on Home and shown on a section.
- CERTAIN — tests/test_nav_and_gating.py:4250-4257: `nav.btn_running.objectName() == 'navRunning'`,
  'QToolButton#navRunning' present in design.build_qss(False) and build_qss(True), `autoRaise()` true, hidden
  with empty text at build, and `code_only(NavShell.__init__).count('apply_icon') == 2` (exactly two icon loads
  in the shell's constructor source).
- CERTAIN — tests/test_nav_and_gating.py:4281-4300: the header run slot — hidden and empty at launch,
  `w._run_timer` not active; while a panel runs its text is running_banner('Sweep study cross-validation', 0 or
  1), its tooltip is exactly 'FDT Analysis → Sweep study cross-validation: go to it', timer interval 1000;
  cleared after.
- CERTAIN — tests/test_nav_and_gating.py:4319-4322: `w._panel_home[panel] == ('FDT Analysis', 'Sweep study
  cross-validation', w._section_index['FDT Analysis'], 1)` and the Posterior panel `== ('Parameter Inference',
  'Posterior', index, 2)` — section name, tab title, stack index and tab index for each panel.
- CERTAIN — tests/test_nav_and_gating.py:4324-4350: clicking btn_running must set `nav.stack.currentIndex()` to
  the section and `fdt_screen.tabs.currentIndex() == 1`; `_section_index['Simulate'] != 0`; an unknown panel
  shows 'Running: a task — ' with an empty tooltip and navigates nowhere.
- CERTAIN — tests/test_nav_and_gating.py:4374-4424: run markers — `fdt.tabText(1) == f'{RUNNING_MARK} Sweep
  study cross-validation'`, `home_screen.tiles['FDT Analysis'].text() == f'FDT Analysis {RUNNING_MARK}'` (two
  spaces), `inf.tabText(2) == f'{RUNNING_MARK} Posterior'`, tiles 'Simulate' and 'Parameter Inference' by name,
  every tile's text equal to its key when idle (4421), the six inference titles restored byte-identical
  (4411-4412).
- CONDITIONAL — tests/test_nav_and_gating.py:4224-4235: running_banner's format, e.g. 'Running: Posterior —
  0:00', 'Running: Posterior — 1:02:05', 'Running: a task — 0:12'. Collides if the run indicator's wording or
  clock format changes.
- CERTAIN — tests/test_nav_and_gating.py:4015-4065: the control-table read-back walks
  `findChildren(QFormLayout)` on every place, takes each row's LabelRole widget, reads the QLabel (or the QLabel
  inside the help holder) and requires every (place, label) of CONTROL to be among them. It also needs
  window.artifact_screen, window.model_builder_screen with vars_edit / _set_variables / _var_rows[0].drift /
  _detect_params, and LabeledFieldRow captions (4045-4052). A screen whose rows are not QFormLayout label rows,
  or a box moved to another place, fails it.
- CERTAIN — tests/test_nav_and_gating.py:4067-4071: the source of InferPanel.__init__ must contain
  label('t_obs'), label('cell'), label('recording_forced'), label('chi_f0_si') and must not contain the literals
  'T_obs (s)' or 'Drive F₀ (N)'.
- CERTAIN — tests/test_nav_and_gating.py:3385-3393: the source of PriorPanel.__init__ must contain exactly nine
  `add_help_row(` calls, none with a literal label, with keys in this order: bounds, prior, num_iterations,
  sweep_batch, max_sets, walk_step, stability_units, min_cluster_size, min_samples; and each CONTROL entry must
  be ('Prior', label).
- CERTAIN — tests/test_nav_and_gating.py:3640-3649: the source of ValidatePanel.__init__ must contain
  label('n_cal') and label('cal_n_scales'), and TSNPEPanel.__init__ label('observation'), label('hpd_level'),
  label('n_directions'), label('num_runs'), label('run_size_cap'), with none of the literal labels present.
- CERTAIN — tests/test_nav_and_gating.py:1976-1991: the source of FdtPanel._build_controls must contain
  label(key) for model, cell, n_freqs, ensemble_m, freqs_per_batch, f0, seed; and a QFormLayout read-back of the
  panel must find rows labelled 'Record name' and 'Note'.
- CERTAIN — tests/test_nav_and_gating.py:4738-4748: core/gui/panels/base_panel.py must contain 'Ten of these
  exist', '9 of the 10', 'all ten differ' and core/gui/panels/inference/base.py 'the six inference tabs'
  (whitespace-normalised source). A different number of panels or inference tabs, or rewritten docstrings, fails
  it.
- CERTAIN — tests/test_nav_and_gating.py:4762-4764: the source of the MainWindow class must contain
  '_persist_layout' and '1500' and must not contain 'the only QSettings WRITE site'. A rewritten MainWindow has
  to keep those two tokens in its text.
- CERTAIN (collection-time) — tests/test_nav_and_gating.py:34-39: module-level imports of
  core.gui.panels.base_panel.BasePanel, core.gui.streams.redirect_streams, core.gui.vt.StreamRouter and
  parse_bar, core.gui.widgets.log_pane.LogPane, core.gui.widgets.progress_pane.ProgressPane,
  core.gui.worker.WorkerSignals. Grep finds the last five lines' names nowhere else in the file. Renaming or
  moving any of those modules fails all 101 tests before one runs.
- CONDITIONAL — the refusal dialog contract, e.g. tests/test_nav_and_gating.py:1681-1698, 1741-1754, 1850-1866,
  2093-2095: a refusal is a QMessageBox with windowTitle 'Check your inputs', icon Warning, text = the core's
  sentence, informativeText = fields.fix_sentence(key), detailedText empty; a bug is 'Error', icon Critical,
  traceback in detailedText. 20 functions read SHOWN; tests/conftest.py:188 only intercepts QMessageBox.exec.
  Inline validation or a custom dialog class breaks these, and a different modal class would stall the suite
  (conftest.py:172-175).
- CONDITIONAL — tests/test_nav_and_gating.py:1318-1336, 4531, 4549: consent dialogs found by button text —
  default 'Cancel' against 'Start a new run anyway' and 'Load it'; default 'Keep this session' against 'Apply
  and start a new session'. The default-is-safe behaviour protects a run; the exact button words and the
  QMessageBox buttons() API are what is pinned.
- CONDITIONAL — the figure area as a tab widget: tests/test_nav_and_gating.py:2688-2689, 2731, 2800-2831,
  2864-2866, 2972-2976, 4909-4913 read figure_stack.count / tabText / widget(i) / tabCloseRequested /
  currentIndex / png_path and assert tab ORDER (e.g. ['comparison', *titles]). A gallery or a single-figure
  viewer breaks seven functions.
- CONDITIONAL — tests/test_nav_and_gating.py:2873: `panel.record_picker.findChild(QPushButton).click()` is taken
  to be the picker's Rescan button. A second button inside the picker, or a Rescan that is not a QPushButton
  child, breaks it.
- CONDITIONAL — tests/test_nav_and_gating.py:5031-5032: fdt_panel.HELP['compare_list'] must contain "'Saved
  run'" and the word 'above', crossval_panel.HELP['compare_list'] "'Saved sweep'". The help text states a
  relative position; if the picker is no longer above the list the test still passes while the text is false.
- CONDITIONAL — the Infer tab's page model: `panel.infer_mode.setCurrentIndex(0)` = simulated and `(1)` =
  experimental (tests/test_nav_and_gating.py:726, 1172, 1194, 3706, 3715, 3772, 3841, 3908, 4078), and three
  separate observation-length boxes sim_tobs, exp_tobs, chi_tobs 'one box per page' (3670, 3710, 3720, 3739). A
  single T_obs box or a different mode switch breaks about ten functions.
- CONDITIONAL — tests/test_nav_and_gating.py:543, 601-602, 4636, 4642: index 0 of a store picker's combo is the
  '(from scratch)' sentinel; 4616-4625 pin the item text (only a narrowed posterior gets NARROWED_SUFFIX) and
  the tooltip format '<id> · <created> · chi · width 18 · amortized'; 4630-4647 pin the read-only lines
  post_line, prior_line, obs_line under each picker.
- CONDITIONAL — tests/test_nav_and_gating.py:4450-4480: session_contents() phrases verbatim ('the built config
  for NADROWSKI, spontaneous observations', "the prior 'p_master'", 'the posterior (unnamed, id
  20260915T090000)', "the observation 'obs1'") and the Config tab's permanent session_line containing 'nothing
  yet' / 'stays on disk'.
- CONDITIONAL — tests/test_nav_and_gating.py:1777-1780 and 1902-1908: AST scans over
  `core/gui/panels/inference/*.py` and over every .py under the directory two levels above simulate_panel.py for
  the attribute name `_config_error`. The paths are derived from module locations, so they follow a move, but
  1903 assumes simulate_panel.py sits exactly one package below core/gui.

### Tests other window suites

**Scope.** 20 files: 8 window-centred suites, 3 fixture/config files, 9 suites that touch `core/gui` in a few places or only through whole-tree source scans. `tests/test_nav_and_gating.py` is excluded (other reader). The 8 window suites and the 3 fixture files were read in full. The large suites with a few touches (test_user_sbi, test_tool, test_artifact_store, test_fdt_user, test_refusals, test_tier1, test_conditioning_repair, test_artifact_consistency) were read at every grep hit for `core.gui`, `core/gui`, `PySide6`, `qt_app`, `SHOWN`, `CODE_ROOTS`, `rglob`, not end to end.

**Counting rule.** `survive_a_relayout` = tests with no structure pin and no positive source-text pin on a panel's builder or click-handler code. Wording pins stay inside it (they pass while labels, tab names and dialog texts are unchanged). The three pin columns overlap. In the few-touch suites the pin columns count window-related pins only.

**The eight window suites: 221 tests, 124 build a widget, panel, screen or window.**
- 193 pass unchanged if classes, no-arg constructors, attribute names and labels are kept and only the arrangement changes.
- 28 do not: test_settings_persistence 15, test_worker_dispatch 3, test_artifact_browser 3, test_user_models 3, test_simulate 2, test_figures 2. Eight of the 15 come back with one helper line (`inf.tabs.widget(2)`, tests/test_settings_persistence.py:78).

**How the tests hold the window.**
1. By attribute name: about 230 widget attribute and method names across 26 classes, almost all on widgets built off-screen with no arguments. There is no accessibility-name or object-name lookup except one (`"iconButton"`, tests/test_user_models.py:1123).
2. By wording: 69 tests. The sharpest are the verbatim fix sentences that name tabs and screens ("Infer or Live simulation tab", "Artifacts screen", "FDT analysis or Sweep study cross-validation tab", "Model Builder screen").
3. By dialog mechanism: 38 tests read a `QMessageBox` instance off `SHOWN`; the session guard patches `QMessageBox.exec` (tests/conftest.py:185-188) and a scan forbids the static forms (tests/test_refusals.py:918-962).
4. By settings mechanism: 20 tests read or write QSettings groups and keys through `core.gui.settings`; two autouse fixtures depend on `ORG`, `APP`, `use_ini_file`, `_override_path` (tests/conftest.py:145-167).
5. By structure: 26 tests (tab index, nav stack index, splitter, QFormLayout rows, widget kind, stylesheet text, token names, font).
6. By source text: 10 tests in the window suites, plus 8 whole-tree scans elsewhere that walk `core/gui`.

**What a rebuild does to them.**
- New navigation: 3 tests rewritten (tests/test_artifact_browser.py:149, tests/test_simulate.py:202, tests/test_user_models.py:1110); the other MainWindow tests survive if its 12 names are kept.
- Renamed or regrouped tabs: every verbatim fix-sentence pin changes with `core/gui/fields.py`.
- New log or progress view: `PaneCapture` (tests/_fixtures.py:140-154) and the 10 widget tests in test_vt_progress need the same `append_line`/`append_lines`, `caption`, `overall`, `solver_label`, `spinner`, `_rows`, `_tick` surface or a rewrite.
- New way of remembering: the 20 settings tests and both fixtures.
- New look: stylesheet, token-name, font and view-geometry pins (5 tests).
- Moving `log_pane`, `progress_pane`, `base_panel`, `streams`, `vt` or `worker` breaks collection of five files at their top-level imports.

**Behaviour pins that protect a run and should carry over as they are:** science knobs open at config.py and are never written; consents and seeds never remembered; the budget travels as arguments; cancel is deferred, never lost, never reported as a clean stop; a refusal is told from a crash; delete and sweep confirm, default to No and are refused during a live run.

| file | lines | tests | touches the window | survive a relayout | pin wording | pin structure | pin source text |
|---|---|---|---|---|---|---|---|
| `tests/test_worker_dispatch.py` | 1879 | 41 | mostly | 38 | 13 | 3 | 1 |
| `tests/test_artifact_browser.py` | 1561 | 39 | wholly | 36 | 28 | 3 | 0 |
| `tests/test_settings_persistence.py` | 1347 | 31 | mostly | 16 | 9 | 13 | 9 |
| `tests/test_vt_progress.py` | 664 | 23 | mostly | 23 | 3 | 0 | 0 |
| `tests/test_simulate.py` | 507 | 17 | mostly | 15 | 4 | 2 | 0 |
| `tests/test_figures.py` | 470 | 19 | mostly | 17 | 3 | 2 | 0 |
| `tests/test_user_models.py` | 1264 | 46 | in a few places | 43 | 9 | 3 | 0 |
| `tests/test_docs.py` | 283 | 5 | in a few places | 5 | 0 | 0 | 0 |
| `tests/_fixtures.py` | 818 | 0 | in a few places | 0 | 0 | 0 | 0 |
| `tests/conftest.py` | 198 | 0 | mostly | 0 | 0 | 0 | 0 |
| `conftest.py` | 39 | 0 | in a few places | 0 | 0 | 0 | 0 |
| `tests/test_refusals.py` | 1372 | 33 | in a few places | 33 | 0 | 0 | 2 |
| `tests/test_fdt_user.py` | 3131 | 65 | in a few places | 65 | 1 | 0 | 0 |
| `tests/test_user_sbi.py` | 5442 | 136 | in a few places | 136 | 0 | 0 | 0 |
| `tests/test_tool.py` | 4364 | 107 | in a few places | 107 | 2 | 0 | 1 |
| `tests/test_tier1.py` | 636 | 24 | in a few places | 24 | 0 | 0 | 0 |
| `tests/test_artifact_store.py` | 5592 | 149 | in a few places | 149 | 0 | 0 | 3 |
| `tests/test_conditioning_repair.py` | 1443 | 44 | in a few places | 44 | 0 | 0 | 2 |
| `tests/test_source_hygiene.py` | 367 | 6 | in a few places | 6 | 0 | 0 | 1 |
| `tests/test_artifact_consistency.py` | 367 | 11 | not at all | 11 | 0 | 0 | 0 |

Notes per file:

- `tests/test_worker_dispatch.py`: No parametrisation. 22 tests build a widget; 19 do not (cancel/stream
  plumbing L138, L151, L252, L1163-L1610; pure table formatting L692-L847; L1215 imports no core.gui). Panels
  are built as `class P(BasePanel): pass; P()` (L56, L98, L198, L310, L376, L412, L458, L508, L617, L1684,
  L1798, L1858). Structure pins: L850-865 (table is a flat read-only QTreeWidget), L893-907 (DEFAULT_SORT == (1,
  1)), L1007-1033 (stylesheet text). Source pin: L593-594, negative, on ArtifactPicker.refresh. Wording: dialog
  titles 'Check your inputs' / 'Error' (L129, L469, L489, L531, L541, L633), the t_obs fix sentence verbatim
  (L472, L533, L623), 'Cancelling…' (L228), 'Run cancelled.' (L242, L1613), column headers (L699-707), cell
  texts (L781-790, L808-813, L819-822), 'library: ...' line format (L1122-1156). 9 tests read a QMessageBox.
  Sort tests L738, L910, L943, L1036 use column indices 0/1/4 as inputs and so depend on the column order pinned
  at L699-707. Top-level imports L31-39 pin six module paths.
- `tests/test_artifact_browser.py`: No parametrisation. 38 tests build ArtifactScreen through
  tests/_fixtures.artifact_screen; L639 calls `_delete_prompt` only. 4 build a full MainWindow (L168, L280,
  L1121, L1174). Structure pins: L71-106 (L84 selector items in KIND_DIRS order), L109-123 (KIND_LABELS keys and
  order, combo count), L149-190 (section order, nav stack index, Home tile button). Of the 28 wording tests, 13
  assert whole strings: L136, L179, L457, L491-493, L527, L602, L744-750, L899, L1060, L1190-1191, L1223-1224,
  L1310, L1523-1524; the rest assert quoted fragments of status and dialog text. 24 tests read a QMessageBox
  through SHOWN or `_answer` (L60-68). 3 tests read the settings store (L186-190, L218-232, L253-257). Helper
  `_show` (L20-23) selects a kind by KIND_DIRS index; `_show_kind` (L39-45) by item data;
  `_select`/`_select_ref` (L33-57) use the QTreeWidget item API.
- `tests/test_settings_persistence.py`: No parametrisation; one module-scoped fixture (L906-917). Helper
  `_budget_panel` (L68-78) returns `inf.tabs.widget(2)`, so eight tests need index 2 to be the Posterior tab
  (L81, L118, L143, L162, L190, L222, L279, L407); replacing that one line with the panel attribute restores
  them. L250 uses the helper but reads inf.tsnpe_panel / inf.posterior_panel. Other structure pins: L567
  (QFormLayout LabelRole rows, L622-631), L636 (splitter not collapsible, controls_scroll max width), L662
  (splitter sizes saved and restored), L688 (form growth policy, numeric field min width, `assert forms`), L716
  (LogPane wrap mode, cell_values wordWrap). Positive source pins on panel code: L394-405, L616-620, L846-871
  (whole test), L1239-1248. Negative or plumbing source pins: L774-779, L802-804, L1275-1280, L901-903,
  L948-951. 15 tests read or write the settings store by group and key. L808 and L942 touch no widget. Wording:
  budget lines verbatim L338-378; fragments at L105-112, L175-184, L210-220, L234-246, L270-276, L430, L630-634,
  L1219.
- `tests/test_vt_progress.py`: No parametrisation. 10 tests build LogPane or ProgressPane and read child widgets
  and private fields (L180, L214, L236, L266, L362, L384, L470, L534, L552, L572); 13 drive core/gui/vt.py and
  streams.py with no widget (L422 and L493 are core/Solvers/sdeint only). All 23 pass under a pure
  rearrangement, but the 10 widget tests are tied to the present log and progress widgets:
  LogPane.blockCount/toPlainText (L205, L230, L257, L289), ProgressPane._rows, caption,
  overall.maximum/value/toolTip, solver_label, spinner, _tick, _steps_at, _rate_at, _beat_at (L210-212,
  L375-381, L401-419, L481-490, L547-548, L567-569, L600). Wording: plus_meter glyphs L353-360; 'idle', '120.0k
  it/s', '+++++' L401-419; 'no output for' L486. L630-646 pins the log_batch payload order (text, level).
- `tests/test_simulate.py`: No parametrisation. 7 tests build a widget (L91, L152, L202, L216, L341, L396,
  L428); 10 call functions in core/gui/panels/simulate_runner.py, simulate_export.py and
  widgets/live_hair_bundle.gaussian_field, whose module paths the imports pin. Structure pins: L202-214
  (MainWindow.panel, 'Simulate' in _section_index, nav.go_to, nav.stack.currentIndex), L91-101 (view geometry:
  _margin > 0, _aspect > 1). Wording: 'Run cancelled.' L200; refusal sentence verbatim L391-393; 'Check your
  inputs' and ffmpeg text L420-424; three fix sentences verbatim L502-507. L216-239 round-trips the Simulate
  panel through save_settings without naming its settings group.
- `tests/test_figures.py`: No parametrisation. 6 tests build widgets (L138, L155, L182, L201, L223, L307). L92
  and L114 use WorkerSignals.figure and base_panel._png_fig_sink. L321-L456 use core/gui/mpl_theme.py and
  design.tokens. L248, L258, L267, L292 touch no core/gui. Structure pins: L138-153 (FigureStack used as a tab
  widget: count(), widget(0), a child QPushButton 'Pop out', `_fig_pickle` on the tab), L307-314 (help_label
  holder has a QLabel child). Wording: 'Pop out' L153; pretty_gui output L260-265; 'F<sub>0</sub>' L314.
  L405-412 and L431-433 read token keys 'window', 'base', 'text' by name. L358-361 and L395 set contrast floors
  on the dark and light palettes.
- `tests/test_user_models.py`: 23 of 46 tests touch core/gui, 15 build a widget; the other 23 are the model
  parser, forcing, registry and model_store. No parametrisation. One display-marked test, L1194-1223. Window
  blocks: L521-L879 the model builder (8 tests); L909 and L958 simulate_runner without a widget; L996-L1038
  MainWindow combo refresh and the builder's run guard; L1042-L1264 design tokens, font, icons, app icon,
  build_app. Structure pins: L1042-1056 (token names, exact on_accent colours, QPalette.Highlight, accent in the
  stylesheet), L1065-1072 (font family 'Inter'), L1110-1126 (NavShell buttons, object name 'iconButton',
  HelpBadge, _ChiProbeRow.btn_remove). L699 selects 'Sinusoidal' by combo index 1; L754, L758, L761, L799, L846
  use _FORCE_KINDS order. Wording: refusal fragments L577, L598, L625-626; 'Check your inputs' L666, L747, L807;
  fix sentences end 'on the Model Builder screen.' L705-707; status 'Saved ...' L870; 'task is running' L1035;
  window title 'PRISM' L1251.
- `tests/test_docs.py`: Imports no core.gui. docs/guide/window.md is one of ten required pages (L14-15,
  L129-130). L128-133 resolves every relative link in README.md and docs/guide/*.md, anchors included: 44 links
  from eight guide pages point at window.md, 38 of them with an anchor, and all ten headings of window.md are
  targets (docs/guide/window.md:11, 34, 67, 85, 115, 135, 224, 261, 285, 332). L136-145 requires each page to
  open with '# Title' then 'Checked against commit <hash>.'. Renaming a heading of the window page fails L128
  until every linking page is edited. L229 and L266 concern the command-line page only.
- `tests/_fixtures.py`: No tests. Window helpers: SHOWN (L60), qt_app (L63-67), pump (L70-75), code_only
  (L101-112), PaneCapture (L115-154: overwrites `panel.log_pane.append_line` and `.append_lines`),
  artifact_screen (L749-761: `ArtifactScreen(store=store)` then `refresh()`). CODE_ROOTS = ('core',) (L37) and
  CODE_FILES = ('conftest.py',) (L41) are what every source scan walks, so every file under core/gui is scanned.
  build_browse_store (L526-605) feeds the browser and the tool's artifacts tests.
- `tests/conftest.py`: No tests; 11 fixtures, 6 of them about Qt or the window. qapp (L12-16) is requested by no
  test. screen_run (L81-102) builds `InferenceScreen()`, calls install_config, sets session.inf_prior and
  session.posterior, calls refresh_gates; it is used only by tests/test_nav_and_gating.py:4100, 4151.
  _settings_home (L131-153) and _isolated_settings (L156-167) depend on core.gui.settings ORG, APP, use_ini_file
  and _override_path and on QSettings ini files. _no_modal_dialogs (L170-190) patches QMessageBox.exec for the
  session; _clear_shown (L193-198) empties SHOWN per test.
- `conftest.py`: No tests. Defaults QT_QPA_PLATFORM=offscreen and KMP_DUPLICATE_LIB_OK=TRUE before any import
  (L17-18), forces the Agg backend (L22), and turns the `display` marker into a skip when the platform is
  offscreen (L35-39). It is itself in CODE_FILES, so the source scans read it.
- `tests/test_refusals.py`: Read at the grep hits only. Three tests reach the window. L492 (L603-613) builds
  `SimulatePanel()` and requires frame_steps.text() and fps.text() to equal the registry defaults for
  'frame_steps' and 'fps'. L918-962 scans every .py under core for a static QMessageBox call. L965 onward
  requires every `field=`/rule key literal under core (core/gui included) to be a registered key and the tool's
  FLAG table to hold exactly the registry's keys (L995-997), so a new field key raised by the window needs a
  registry entry and a tool-table entry.
- `tests/test_fdt_user.py`: Read at the grep hits only. Two tests reach core/gui. L1118 (L1160-1170) builds
  IntField(60) and FloatField(0.05) and relies on a blank box reading as 0 / 0.0. L1248 (L1268, L1294-1308)
  requires gui fix_sentence(key) == '' for dt_nd, psd_t_obs_nd, t_obs_periods, burn_in_nd, freq_bounds: the
  window must keep having no control for them.
- `tests/test_user_sbi.py`: Read at the grep hits only. 136 `def test_`; parametrised at L1954 (3 cases) and
  L2317 (2 cases). Six tests import core.gui.streams or core.gui.worker and build no widget: L1003 (L1020),
  L1225 (L1229), L1337 (L1348), L5156 (L5172-5211: redirect_streams, WorkerSignals.log_batch/rows, CancelToken),
  L5396 and L5418 through the helper at L5320 (L5342: CancelToken, _PumpLogHandler). They pin the module path
  and names of the cancel and stream plumbing, not any screen.
- `tests/test_tool.py`: Read at the grep hits only. 107 `def test_`; parametrised at L1999-2000 (4 x 2). Three
  tests import core/gui. L2918 (L2928-2946): redirect_streams must not call setLevel (negative source pin).
  L3072 (L3085-3099): the tool's column list per kind must equal the window's `columns_for(kind)` lower-cased,
  and `_FINGERPRINT_DEPENDENT` must be the same string in both front ends. L3958 (L3966-3977): the tool's two
  'no log' sentences must be `artifact_screen._CACHE_NO_LOG` and `_NO_RUN_LOG`. A change to the Artifacts
  table's columns or these sentences must be made in core/tool/browse.py too.
- `tests/test_tier1.py`: Read at the grep hit only. 24 `def test_`; four parametrised over chi True/False (L160,
  L201, L221, L385). One test, L405-418, imports core.gui.panels.simulate_runner and pins plan_stream's fields
  rescale_idx, rescale_gt, x_scale and the call run_simulation_stream(cfg, t, frame_steps=, fps=, emit_chunk=).
  No widget.
- `tests/test_artifact_store.py`: Read at the grep hits only; imports no core.gui. Three whole-tree scans walk
  core/gui: L2116-2166 (no literal Resources/ or Artifacts/ path outside core/config.py), L2169-2198 (top-level
  directories holding Python must equal CODE_ROOTS, top-level .py files must equal CODE_FILES, so a window
  package outside core/ fails), L3751-3792 (the set of `@public_entry` functions is exact, so none may appear
  under core/gui). L3846 keeps control names such as 'Save box' out of a core message. Parametrised at
  L3795-3796.
- `tests/test_conditioning_repair.py`: Read at the grep hits only; imports no core.gui. Two whole-tree scans
  walk core/gui: L915-942 (only core/SBI/reparam.py may read `parts[0].M`) and L945-971 (no `input()` call
  anywhere under core).
- `tests/test_source_hygiene.py`: Imports no core.gui. L220-232 scans the comments, docstrings and string
  literals of every .py under core (core/gui included) and every page under docs/guide (window.md included) for
  references to working documents and their labels; scanned_files at L148-158. New window code and a rewritten
  window page must be clean of them. L310-315 expects at least ten reader pages.
- `tests/test_artifact_consistency.py`: Matched the PySide6/QSettings grep but imports nothing from core/gui.
  L300 requires a core refusal message not to contain 'PRISM.ini', 'QSettings' or 'Config tab'.

#### The names the tests reach for

Names are those used in my files only (tests/test_nav_and_gating.py excluded). Counts are distinct own names; inherited Qt calls are listed after "Qt:".

### Windows and screens
- **MainWindow** — 12: `artifact_screen`, `inference_screen`, `model_builder_screen`, `nav`, `_section_index`, `_all_panels()`, `panel(cls)`, `_save_state()`, `_refresh_store_pickers()`, `_on_user_models_changed()`, `_edit_user_model(name)`, `_delete_user_model(name)`. Qt: `close()`, `windowTitle()`. Built as `MainWindow()`. (tests/test_artifact_browser.py:168-190, 280-288, 1121-1147, 1174-1188; tests/test_simulate.py:209-214; tests/test_user_models.py:1002-1017.) `core.gui.app.build_app([])` returns `(app, window)` (tests/test_user_models.py:1247).
- **NavShell** — 4: `btn_back`, `btn_settings`, `stack` (`.widget(i)`, `.currentIndex()`), `go_to(idx)`. Built as `NavShell()`. (tests/test_user_models.py:1120-1121; tests/test_simulate.py:212-213; tests/test_artifact_browser.py:172-182.)
- **home_screen** — module constant `SECTIONS`; Home is `nav.stack.widget(0)` and holds a QPushButton with text "Artifacts" and property `accent` True. (tests/test_artifact_browser.py:163-181.)
- **InferenceScreen** — 10: `install_config(cfg)`, `session`, `refresh_gates()`, `tabs` (`.widget(2)`), `config_panel`, `prior_panel`, `posterior_panel`, `validate_panel`, `infer_panel`, `tsnpe_panel`. Built as `InferenceScreen()`. (tests/conftest.py:97-101; tests/test_settings_persistence.py:73-78, 264-268, 1102-1135, 1200-1234, 1283-1309; tests/test_worker_dispatch.py:293-297.)
- **SbiSession** — 3: `cfg`, `inf_prior`, `posterior`. (tests/test_settings_persistence.py:74-76; tests/conftest.py:99-100.)
- **ArtifactScreen** — 22: `kind_combo`, `kind()`, `table`, `status`, `detail`, `note_edit`, `btn_save`, `btn_note`, `btn_delete`, `btn_lineage`, `btn_sweep_all`, `store_changed` (signal), `refresh()`, `save_settings(qs)`, `_store`, `_resolved_store()`, `_sort`, `_set_status(text)`, `_set_note()`, `_delete()`, `_sweep(all_kinds=)`, `_lineage_report()`. Qt: `findChildren(QLabel)`. Built as `ArtifactScreen(store=store)`. Module names: `KIND_LABELS`, `LOG_MAX_BYTES`, `_delete_prompt`, `_FINGERPRINT_DEPENDENT`, `_CACHE_NO_LOG`, `_NO_RUN_LOG`. (tests/test_artifact_browser.py throughout; tests/test_tool.py:3099, 3972, 3977.)
- **ModelBuilderScreen** — 14: `vars_edit`, `name_edit`, `x_scale`, `t_scale`, `status`, `_var_rows`, `_param_fields`, `_set_variables()`, `_detect_params()`, `_validate()`, `_validate_clicked()`, `_save()`, `_assemble_doc()`, `load_existing(name)`. Built as `ModelBuilderScreen()`. Variable row — 6: `drift`, `noise`, `init`, `force_kind`, `forcing_fields()`, `_force_fields[kind][name]`. `_ParamRow(value)` — 5: `auto`, `lo`, `value`, `spec()`, `set_spec(...)`. Module: `_FORCE_KINDS`. (tests/test_user_models.py:524-549, 563-577, 652-702, 729-764, 848-876, 1026-1038.)

### Panels
- **BasePanel** — 15: `dispatch(fn, *args, on_result=, provide_stream=, on_chunk=)`, `_busy`, `_workers` (each `.signals.cancelled`, `.signals.error`), `_request_cancel()`, `_cancel` (`.requested`), `btn_cancel`, `progress_pane`, `log_pane`, `_on_error(exc, tb)`, `_refusal(exc)`, `splitter`, `controls_scroll`, `save_layout(qs)`, class attributes `_active_cancel` and `_running`. Module: `_png_fig_sink`, `show_refusal`. Subclassed with an empty body and built with no arguments. (tests/test_worker_dispatch.py:63-65, 215-250, 324, 628-655, 1693-1729; tests/test_simulate.py:179-200; tests/test_settings_persistence.py:650-686; tests/test_figures.py:97-104; tests/_fixtures.py:142-143.)
- **ConfigPanel** — 13: `model_combo`, `_build_config()`, `vram_ceiling`, `vram_note`, `chi_k`, `chi_pad`, `chi_f0`, `chi_range` (`.lo`, `.hi`), `chi_cycles`, `chi_check`, `rot_check`, `save_settings`, `restore_settings`. Built as `ConfigPanel(None)`. (tests/test_settings_persistence.py:753-785, 1001-1040, 1346-1347; tests/test_worker_dispatch.py:294-296.)
- **PriorPanel** — 12: `bounds_picker`, `prior_picker`, `_saved_bounds_key`, `bounds_source` (`.is_direct()`), `sweep_iters`, `sweep_batch`, `sweep_max_sets`, `sweep_step`, `sweep_units`, `cluster_size`, `cluster_samples`, `save_settings`. (tests/test_settings_persistence.py:1078-1083, 1123-1135; tests/test_worker_dispatch.py:297-299; tests/test_artifact_browser.py:1123.)
- **PosteriorPanel** — 20: `num_runs`, `run_size_cap`, `budget_total`, `budget_mem`, `budget_ckpt`, `_sync_budget()`, `_build_posterior()`, `_confirm_fresh_run` (source), `dispatch`, `_refusal`, `_hardware()`, `save_settings`, `restore_settings`, `flow_hidden`, `flow_transforms`, `flow_lr`, `flow_patience`, `fisher_m`, `fisher_dz`, `fisher_points`. (tests/test_settings_persistence.py:101-160, 203-248, 325-390, 421-444, 854-871, 1084-1088.)
- **ValidatePanel** — 3: `cal_n`, `cal_scales`, `save_settings`; must not define `restore_settings` (tests/test_settings_persistence.py:1089-1090, 1247).
- **InferPanel** — 3: `other_obs`, `save_settings`, `restore_settings` (tests/test_settings_persistence.py:1275-1309).
- **TSNPEPanel** — 12: `obs_picker`, `num_runs`, `run_size_cap`, `hpd`, `n_dirs`, `new_run`, `budget_total`, `budget_ckpt`, `_sync_budget()`, `_budget_checkpoint` (source), `save_settings`, `restore_settings` (tests/test_settings_persistence.py:264-276, 404, 1091-1092, 1194-1237, 1284).
- **FdtPanel** — 11: `n_freqs`, `skip_sanity`, `confirm_production`, `record_picker`, `seed`, `record_name`, `record_note`, `cell_picker`, `model_combo`, `save_settings`, `restore_settings`. Built as `FdtPanel()`. (tests/test_settings_persistence.py:458-477, 502-522, 1286-1312; tests/test_user_models.py:1003-1016.)
- **CrossValPanel** — 8: `s_grid` (`.hi`, `.points`), `f0`, `record_picker`, `seed`, `record_name`, `cell_values`, `save_settings`, `_build_controls` (source). Qt: `findChildren(QFormLayout)`. (tests/test_settings_persistence.py:548-565, 594-631, 726.)
- **ReductionPanel** — 3: `f0`, `cell_picker`, `save_settings` (tests/test_settings_persistence.py:452-471, 538-539, 916).
- **SimulatePanel** — 13: `tobs`, `fps`, `frame_steps`, `cell_picker` (`.selected_path`), `model_combo`, `btn_save_video`, `_on_chunk(arr)`, `_record`, `refresh_local_gates()`, `dispatch`, `_save_video()`, `_start()`, `save_settings`. Module names patched: `ffmpeg_available`, `build_stream_config`. (tests/test_simulate.py:221-239, 346-355, 410-499; tests/test_refusals.py:610-611.)
- **Inference module paths**: `core.gui.panels.inference_tabs` must expose `ConfigPanel`, `PosteriorPanel`, `_nvidia_smi_free_gib`, `_ChiProbeRow`; `core.gui.panels.inference.{config_tab, prior_tab, posterior_tab, validate_tab, infer_tab, tsnpe_tab}` the six panel classes; `core.gui.panels.inference.base._TrainingBudgetMixin`. (tests/test_settings_persistence.py:299-300, 746, 852, 1168-1173; tests/test_user_models.py:1116.)

### Widgets
- **ArtifactTable** — 5 own: `set_rows(kind, rows)`, `apply_sort_state(col, order)`, `sort_state()`, `current_summary()`, `selection_changed`. Qt (QTreeWidget): `topLevelItem(i).text(c)`, `topLevelItemCount()`, `columnCount()`, `headerItem()`, `header().setSortIndicator`, `setCurrentItem`, `clearSelection`, `rootIsDecorated`, `isSortingEnabled`, `selectionBehavior`, `selectionMode`, `editTriggers`. Module: `columns_for`, `cells_for`, `cell_key`, `NUMERIC_COLUMNS`, `DEFAULT_SORT`, `_EXTRA_COLUMNS`. (tests/test_worker_dispatch.py:699-1075; tests/test_artifact_browser.py:29-57, 250.)
- **ArtifactPicker** — 5: `refresh()`, `combo`, `key()`, `base_path`, `selected_path`; a child QPushButton with object name `iconButton`. Built as `ArtifactPicker(path, keep=)`. **StorePicker** — 6: `kind`, `_resolved_store` (patched on the class), `refresh()`, `restore_key(id)`, `key()`, `combo`. (tests/test_worker_dispatch.py:584-594; tests/test_user_models.py:1122-1124; tests/test_artifact_browser.py:281-289, 1117-1145; tests/test_settings_persistence.py:501, 593, 1189.)
- **LogPane** — 2 own: `append_line(text, level="info")`, `append_lines(batch)`. Qt (QPlainTextEdit): `toPlainText()`, `blockCount()`, `lineWrapMode()`. Built as `LogPane()`. (tests/test_vt_progress.py:183-212; tests/test_worker_dispatch.py:380-392; tests/test_settings_persistence.py:724.)
- **ProgressPane** — 13: `begin()`, `end()`, `set_rows(rows)`, `heartbeat()`, `_tick()`, `_rows`, `caption`, `overall`, `solver_label`, `spinner`, `_steps_at`, `_rate_at`, `_beat_at`. Module: `plus_meter`, `SOLVER_IDLE_S`, `STALL_S`. Built as `ProgressPane()`. (tests/test_vt_progress.py:183-212, 351-360, 368-420, 472-491, 540-600; tests/test_worker_dispatch.py:239.)
- **FigureStack** — 6: `add_figure(title, png, fig_pickle=)`, `add_png(title, path)`, `count()`, `widget(i)`, `_pop_out(container)`, `_windows`; the tab container carries `_fig_pickle`. Pop-out windows: `.canvas`, `._fig`; `ImageZoomWindow.view`. (tests/test_figures.py:145-244.)
- **LiveHairBundleView** — 3: `_margin`, `_cx(x)`, `_aspect`; module `gaussian_field`. (tests/test_simulate.py:71-101.)
- **FloatField / IntField** — `value()`, `value_or_none()`, `setText`, `text`, `minimumWidth()`; built with one default value. (tests/test_settings_persistence.py:710-711, 964-970; tests/test_fdt_user.py:1163-1170.)
- **help_badge** — `help_label(text, help)` returning a holder with a QLabel child; `HelpBadge(text).text()`. **_ChiProbeRow(callback)** — `btn_remove`. (tests/test_figures.py:312-314; tests/test_user_models.py:1125-1126.)
- **refusal_box** — `show_refusal(parent, exc)` (tests/test_worker_dispatch.py:611, 629).

### Plumbing modules under core/gui (no widget)
- **settings** — `settings()`, `use_ini_file(path)`, `_override_path`, `ORG`, `APP`, `get_int`, `get_bool` (tests/conftest.py:146-165; tests/test_settings_persistence.py:901-939, 1333-1340; tests/test_artifact_browser.py:256).
- **streams** — `redirect_streams(signals, token=None)`, `CancelToken` (`requested`, `fired`, `arm()`, `check()`), `WorkerCancelled`, `_SignalStream(pump, name, level, token)`, `_PumpLogHandler(pump, token)`. **worker** — `WorkerSignals` (`log`, `log_batch`, `rows`, `figure`, `cancelled`, `error`), `CANCEL_TOO_LATE_TEXT`. **vt** — `StreamRouter(name, sink, level=)`, `parse_bar(key, row, text)` with fields `total`, `informative`, `pct`, `rate`, `is_solver`, `desc`, `row`, `key`, `raw`. (tests/test_worker_dispatch.py:162-175, 1177-1212, 1793; tests/test_vt_progress.py:57, 174-177, 332-348, 621-663.)
- **app** — `build_app`, `_library_record_sink`. **design** — `build_qss(dark, accent=None)`, `tokens(dark, accent=None)` with keys `alt_base`, `text_2nd`, `accent`, `accent_hover`, `accent_press`, `on_accent`, `mid`, `window`, `base`, `text`; `build_palette`, `system_accent`. **mpl_theme** — `apply_mpl_theme(dark)`, `install(appearance)`; an appearance needs `is_dark()` and a `theme_changed` signal. **fonts** — `load_app_font(app, prefer_inter=)`. **icons** — `register`, `available`, `NAMES`, `glyph`, `apply_icon`, `_ICON_DIR`. **app_icon** — `app_icon`, `set_windows_app_user_model_id`, `set_windows_class_icon`, `_APP_DIR`. **fields** — `fix_sentence(key)`, `label(key)`. (tests/test_worker_dispatch.py:1017-1033, 1109; tests/test_user_models.py:1043-1107, 1143, 1158-1191, 1247; tests/test_figures.py:70-89, 403-433.)
- **simulate_runner** — `frame_time_grid`, `build_stream_config`, `plan_stream` (plan fields `subsample_factor`, `steady_steps`, `total_steps`, `n_obs`, `dt_nd`, `n_channels`, `state_dep_drift`, `x_scale`, `forcing_idx`, `user_spec`, `rescale_idx`, `rescale_gt`), `run_simulation_stream`. **simulate_export** — `export_stride`, `estimate_frame_count`, `export_animation`, `ffmpeg_available`, `gaussian_field`. (tests/test_simulate.py:56-149, 242-339; tests/test_tier1.py:408-416; tests/test_user_models.py:973-985.)

#### Tables and fixtures

### Tables the tests hold about the window (my files)

**The table that names each control.** The product table is `core/gui/fields.py` (`SCREENS` L45, `CONTROL` L55, `label` L190, `fix_sentence` L221). Its read-back against the built window (key set against the registry, places against tab titles, sentences verbatim) is in tests/test_nav_and_gating.py (grep hits at L1520-1640, L3990, L4036-4053, L5062-5063), which is the other reader's file and was not analysed. My files consume the table at:
- tests/test_worker_dispatch.py:472-473, 533, 623 — the `t_obs` sentence verbatim.
- tests/test_artifact_browser.py:491-493, 527, 750, 1224, 1524 — the `note` and `artifact` sentences verbatim.
- tests/test_simulate.py:473, 502-507 — `frame_steps`, `fps`, `t_obs` sentences verbatim.
- tests/test_user_models.py:668, 705-707, 748, 809 — seven Model Builder keys (`param_value`, `param_min`, `param_max`, `init`, `x_scale`, `t_scale`, `forcing_value`) must end "on the Model Builder screen."
- tests/test_settings_persistence.py:229-243, 390 — `label('num_runs')`, `label('run_size_cap')` build the live budget messages.
- tests/test_settings_persistence.py:616-634 — nine CrossVal rows (`cell`, `preset`, `s_grid`, `t_grid`, `n_freqs`, `ensemble_m`, `freqs_per_batch`, `f0`, `seed`) must be built from `label(key)`; two literal rows "Record name" and "Note" must match the words the `name` and `note` sentences quote.

**Settings with no control.** In my files only tests/test_fdt_user.py:1294-1308: `dt_nd`, `psd_t_obs_nd`, `t_obs_periods`, `burn_in_nd`, `freq_bounds` give an empty fix sentence in both front ends. The full pinned set of `None` entries: cannot tell from my files (core/gui/fields.py:23-25 describes it).

**Remembered keys.**
- tests/test_settings_persistence.py:1137-1145 — exact written sets: `inference_prior` {prior, bounds}; `inference_posterior` {posterior, num_runs, run_size_cap}; `inference_validate` {}; `inference_tsnpe` {observation, num_runs, run_size_cap}.
- tests/test_settings_persistence.py:1077-1093 — the science-knob table: (settings group, panel attribute on the screen, key = box attribute name) for 18 knobs on Prior, Posterior, Validate and TSNPE; none may be read or written.
- tests/test_settings_persistence.py:996-998, 1010-1013, 1045-1046 — six Config-tab chi keys never written; `inference_config` writes at least {model, units_mode, units_text, chi_mode, reparam_rotate}.
- tests/test_settings_persistence.py:515, 609 — `fdt` and `crossval` never hold seed, record_name, record_note; L559 `crossval/s_hi` absent; L532-534 `reduction` cell and f0.
- tests/test_settings_persistence.py:1275-1276, 1293-1298 — the consent table: TSNPEPanel `new_run`, InferPanel `other_obs`, FdtPanel `skip_sanity` and `confirm_production`; groups `inference_tsnpe`, `inference_infer`, `fdt` must write something and no consent.
- tests/test_settings_persistence.py:438-441, 1229-1232 — the budget is saved as the boxes' text ("" when blank).
- tests/test_settings_persistence.py:1241-1242 — the ten panel classes whose `__init__` must call `restore_settings`.
- tests/test_artifact_browser.py:222 — `artifacts` holds exactly {kind, sort_col, sort_order}; L229-232 a stale kind is ignored.
- tests/test_worker_dispatch.py:288-289 — `inference_config/model`.
- The Simulate panel's group name: cannot tell; tests/test_simulate.py:230-239 round-trips without naming it.

**Tab titles and screens.** My files hold no list of tab titles (that read-back is tests/test_nav_and_gating.py:339, 1533, 4036-4037). They hold:
- tests/test_settings_persistence.py:78 — the comment "Config Prior Posterior Validate Infer" and index 2 for Posterior.
- tests/test_artifact_browser.py:166, 170-171 and tests/test_simulate.py:211-213 — section names "Artifacts", "Simulate", "Settings" as `_section_index` keys; `SECTIONS[-1] == "Artifacts"`.
- Place names inside verbatim sentences: "Infer", "Live simulation", "Artifacts" screen, "FDT analysis", "Sweep study cross-validation", "Model Builder" screen (lines above).
- tests/test_artifact_browser.py:17, 84, 117-120, 899 — the kind selector follows `KIND_DIRS`; `KIND_LABELS["fdt"]` verbatim.

**Other tables.**
- tests/test_worker_dispatch.py:699-707 — the column headers per artifact kind; tests/test_tool.py:3089-3099 — the same columns lower-cased in the command-line tool.
- tests/test_artifact_browser.py:1190-1191 — the four model-dialog titles.
- tests/test_docs.py:14-15 — the ten guide pages, window.md among them.

### Fixtures and helpers that build windows, screens and panels off-screen

**Environment.** conftest.py:17-18 defaults `QT_QPA_PLATFORM=offscreen` and `KMP_DUPLICATE_LIB_OK=TRUE`; L22 forces Agg; L35-39 skips `display` tests offscreen. pytest.ini:8 registers the marker.

**tests/conftest.py.**
- `qapp` (session, L12-16) — the QApplication; requested by no test.
- `_sandbox_default_store` (session, autouse, L19-31) — temp ArtifactStore as the process default; asserts the real Artifacts/ gained nothing.
- `store` (function, L59-64) — a fresh store that is also the process default; handed to `artifact_screen(store)`.
- `tiny_run` (module, L67-78) and `screen_run` (module, L81-102) — a real tiny prior and posterior, then an `InferenceScreen()` with the config installed, `session.inf_prior` and `session.posterior` set, gates refreshed; returns `SimpleNamespace(screen, run, cell)`. Used only by tests/test_nav_and_gating.py:4100, 4151.
- `_settings_home` (session, autouse, L131-153) — points `core.gui.settings` at a temp ini before any panel is built; asserts the real user-scope file is byte-identical at teardown.
- `_isolated_settings` (function, autouse, L156-167) — a fresh ini per test; restores the session path, never the real file.
- `_no_modal_dialogs` (session, autouse, L170-190) — replaces `QMessageBox.exec` with record-into-`SHOWN`-and-return-0. `_clear_shown` (function, autouse, L193-198).
- `_checkpointing_off_unless_asked` (L105-128) and `_no_sbi_logs` (L34-56) — not window fixtures.

**tests/_fixtures.py.** `qt_app()` L63-67; `pump(app, seconds)` L70-75; `SHOWN` L60; `PaneCapture(panel)` L115-154 (stands in for both log-pane channels, must be installed before `dispatch`); `code_only(obj)` L101-112; `artifact_screen(store)` L749-761; `build_browse_store(root)` L526-605; `_nad_cfg` L157-166 and `_prior_artifact` L185-213 (real config and prior for the browser's delete and note tests); `backdate_tree` L43-54.

**Per-file builders.**
- tests/test_settings_persistence.py: `_budget_cfg` L41-62 (stub config), `_prior_stub` L63-67, `_budget_panel` L68-78 (an `InferenceScreen()` with a fresh `SbiSession`, returns the screen and tab index 2), `_panel_built_at_module_scope` L906-917 (module-scoped `ReductionPanel()`).
- tests/test_artifact_browser.py: `_show` L20-23, `_row_text` L26-30, `_select` L33-36, `_show_kind` L39-45, `_select_ref` L48-57, `_answer` L60-68 (chosen dialog answer), `_interrupted_fdt` L624-636, `_Proxy` L1243-1251 (store stand-in).
- tests/test_worker_dispatch.py: `class P(BasePanel): pass`; `_row` L668-678 and `_bad_row` L681-689 (hand-built Summary rows); `_run_leg` L1617-1631.
- tests/test_vt_progress.py: `_drive` L41-64, `_live` L65-74. tests/test_figures.py: `_tiny_fig` L40-44, `_png_bytes` L45-49, `_rcparams_guard` L60-69, `_fake_appearance` L70-89. tests/test_simulate.py: `_tiny_series` L41-45, `_export_kwargs` L46-49.

**What the tests stub on the window.**
- `panel.dispatch` (tests/test_settings_persistence.py:134, 154; tests/test_simulate.py:413, 457) and `panel._refusal` (tests/test_settings_persistence.py:155).
- `StorePicker._resolved_store` (tests/test_settings_persistence.py:501, 593, 1189; tests/test_artifact_browser.py:1117); the screen's `_resolved_store` (tests/test_artifact_browser.py:740, 1292, 1304, 1339, 1356, 1367) and `_store` (L142).
- `QFileDialog.getSaveFileName` (tests/test_artifact_browser.py:429, 435, 1470, 1503; tests/test_simulate.py:416, 494).
- `QMessageBox.exec` (tests/test_worker_dispatch.py:322, 355; tests/test_artifact_browser.py:68, 954) and the statics `warning`/`information` as a tripwire (tests/test_artifact_browser.py:1167-1169).
- `BasePanel._running` set directly (tests/test_artifact_browser.py:894, 1081, 1180, 1504; tests/test_user_models.py:1032).
- Module attributes: `base_panel.show_refusal` (tests/test_worker_dispatch.py:652), `artifact_picker.file_manager.list_dir` (L584), `simulate_panel.ffmpeg_available` and `build_stream_config` (tests/test_simulate.py:415, 459, 493), `orchestrator.training_preview` (tests/test_settings_persistence.py:317).

**Display-marked tests.** Exactly one in tests/ and core/Reduction/tests: tests/test_user_models.py:1194-1223. It needs `QT_QPA_PLATFORM=windows` and the windows platform plugin (it skips itself otherwise, L1208-1209), uses `ctypes` `user32`, creates a `QMainWindow` that is never shown, and checks that `app_icon.set_windows_class_icon` replaces the stock class icon. Its offscreen twin is L1168-1191, where the same call must return False.

#### Pins a redesign collides with

- tests/test_artifact_browser.py:166-182 — navigation: SECTIONS[-1] == 'Artifacts'; _section_index['Simulate'] <
  ['Artifacts'] < ['Settings']; nav.stack.widget(idx) is the artifact screen; Home is nav.stack.widget(0) and
  holds a QPushButton with text 'Artifacts' and property accent True whose click sets nav.stack.currentIndex()
  to idx.
- tests/test_simulate.py:209-214 — navigation: MainWindow.panel(SimulatePanel) is not None, 'Simulate' is a key
  of _section_index, nav.go_to(idx) makes nav.stack.currentIndex() == idx, and a SimulatePanel is in
  _all_panels().
- tests/test_user_models.py:1120-1126 — NavShell() has btn_back and btn_settings with non-empty text;
  ArtifactPicker has a child QPushButton with object name 'iconButton'; HelpBadge and _ChiProbeRow.btn_remove
  render text.
- tests/test_settings_persistence.py:78 — `inf.tabs.widget(2)` must be the Posterior tab; eight budget tests
  (L81, L118, L143, L162, L190, L222, L279, L407) go through it.
- tests/test_settings_persistence.py:636-660 — every BasePanel has `splitter` (children not collapsible, no
  zero-width pane at 1300x820) and `controls_scroll` with maximumWidth above 1000.
- tests/test_settings_persistence.py:662-686 — splitter sizes are saved by save_layout(qs) and restored in
  __init__ to the same sizes.
- tests/test_settings_persistence.py:688-714 — CrossVal, FDT and Reduction panels must contain QFormLayouts
  (`assert forms`), each with AllNonFixedFieldsGrow, and every FloatField/IntField at least 80 px wide.
- tests/test_settings_persistence.py:716-727 — LogPane is a QPlainTextEdit with lineWrapMode WidgetWidth;
  CrossValPanel.cell_values wraps.
- tests/test_settings_persistence.py:616-634 — CrossValPanel._build_controls source must contain label('<key>')
  for nine keys, and the form's LabelRole rows must show 'Record name' and 'Note', the words the name/note fix
  sentences quote together with 'Sweep study cross-validation'.
- tests/test_settings_persistence.py:1137-1145 and tests/test_artifact_browser.py:222 — the exact key sets each
  settings group writes (inference_prior, inference_posterior, inference_validate, inference_tsnpe, artifacts);
  any new remembered key or renamed group fails.
- tests/test_settings_persistence.py:1077-1093 — each science knob is found as getattr(getattr(screen,
  '<panel>'), '<key>') with the box attribute named after the settings key; moving a knob to another panel or
  renaming a box fails.
- tests/test_settings_persistence.py:1239-1248 — the literal text `self.restore_settings(settings.settings())`
  must appear in the __init__ source of every panel class that defines restore_settings (ten classes listed);
  ValidatePanel must not define one.
- tests/test_settings_persistence.py:854-871 — source shape of PosteriorPanel._build_posterior and
  _confirm_fresh_run: '_confirm_fresh_run' before 'self.dispatch(' with a 'return' between, 'new_run=' present,
  'True, False' at least three times, 'fresh_run_near_misses' present, three names banned.
- tests/test_settings_persistence.py:394-405 — the AST names of _TrainingBudgetMixin must include
  training_preview and none of 14 listed names; TSNPEPanel._budget_checkpoint must not name config,
  TRAINING_CHECKPOINT_EVERY or orchestrator.
- tests/test_settings_persistence.py:338-378 — the three budget lines verbatim, including '2,100 simulations = 3
  batches x 700 rows (capped from 2,048).' and all six checkpoint-state sentences.
- tests/conftest.py:145-167 and tests/test_settings_persistence.py:929-939 — remembering is a QSettings ini
  reached through core.gui.settings (ORG, APP, use_ini_file, _override_path, settings()); 20 tests read groups
  and keys with beginGroup/childKeys/value. A different store collides with all of them and with both autouse
  fixtures.
- tests/conftest.py:185-188 with tests/test_refusals.py:918-962 — every box the window shows must be a
  QMessageBox instance shown with exec(); a static call anywhere under core fails the scan, and 38 tests in my
  files read title, icon, text, informativeText, detailedText or buttons off SHOWN.
- tests/test_worker_dispatch.py:469-477, 531-535, 633-639 — the refusal box: title 'Check your inputs', Warning
  icon, text == the refusal message, informativeText == "Set it in the 'T_obs (s)' box on the Infer or Live
  simulation tab.", no details, one OK button that is the default; a bug: title 'Error', Critical icon,
  traceback in detailedText (L541-543).
- tests/test_worker_dispatch.py:228, 239, 242 — after a cancel request btn_cancel reads 'Cancelling…' and is
  disabled; afterwards panel.progress_pane._rows is empty and 'Run cancelled.' is in
  panel.log_pane.toPlainText().
- tests/_fixtures.py:140-154 with tests/test_worker_dispatch.py:380-392, 417-439 — a panel's log view is
  `panel.log_pane` with append_line(text, level) and append_lines(batch of (text, level)), both resolved by
  dispatch at connect time; tests/test_vt_progress.py:630-646 pins the batch payload order.
- tests/test_vt_progress.py:205-212, 375-381, 401-419, 481-490, 547-548, 567-569, 600 — the progress view:
  `overall` is a bar whose maximum 0 means indeterminate and whose value is the percent of the largest
  non-degenerate total; `caption` starts with that bar's description; overall.toolTip contains '1902/5000';
  solver_label shows 'idle', '120.0k it/s', '+++++'; spinner shows four distinct frames then 'no output for';
  end() empties _rows and caption.
- tests/test_vt_progress.py:353-360 — plus_meter output: '++++' for 10 000, '+++' for 1 000, '·' under 10, '—'
  for None.
- tests/test_worker_dispatch.py:31-39, tests/test_settings_persistence.py:30-35,
  tests/test_vt_progress.py:31-36, tests/test_simulate.py:31-36, tests/test_figures.py:30-35 — top-level imports
  of core.gui.panels.base_panel.BasePanel, core.gui.streams, core.gui.vt, core.gui.widgets.log_pane.LogPane,
  core.gui.widgets.progress_pane.ProgressPane, core.gui.worker.WorkerSignals; moving or renaming any of them
  stops the whole file from collecting.
- tests/test_worker_dispatch.py:699-707 with tests/test_tool.py:3094 — the Artifacts table's column headers per
  kind, verbatim and in order, and the command-line tool's column list must be the same words lower-cased.
- tests/test_worker_dispatch.py:860-865, 905-907 — the table is a QTreeWidget with rootIsDecorated False,
  sorting on, SelectRows, SingleSelection, NoEditTriggers; DEFAULT_SORT == (1, 1), i.e. column 1 'Created'
  descending.
- tests/test_worker_dispatch.py:1019-1033 — stylesheet text: 'QTreeView {', 'QTreeView::item',
  'QHeaderView::section', 'alternate-background-color: <alt_base>', 'background: <alt_base>; color: <text_2nd>',
  no '$' left, and 'selection-background-color: #AA3366' inside the QTreeView block under an accent override.
- tests/test_user_models.py:1044-1056 — design tokens: accent override sets 'accent' and distinct
  'accent_hover'/'accent_press'; on_accent is exactly '#1B1B1B' on a light accent and '#FFFFFF' on a dark one;
  QPalette.Highlight equals the accent; QPalette.Mid equals token 'mid'; the accent hex appears in build_qss.
- tests/test_user_models.py:1070 — fonts.load_app_font(app, prefer_inter=True) must return 'Inter'.
- tests/test_figures.py:405-412, 431-433 — design.tokens keys 'window', 'base', 'text' must exist and drive
  matplotlib figure/axes/savefig face colours and text colour; L358-361 the dark page corner must be darker than
  120; L391-396 table cells opaque with contrast at least 3:1 in both themes.
- tests/test_figures.py:150-153, 171-179, 194-197 — figures live in a FigureStack with count() and widget(0);
  each tab stores _fig_pickle and has a child QPushButton 'Pop out'; pop-out of a pickle is a window with a
  FigureCanvasQTAgg in `canvas`, without one an ImageZoomWindow whose `view` is a QGraphicsView; windows are
  tracked in _windows.
- tests/test_simulate.py:98-101 — LiveHairBundleView: _margin > 0, _cx stays inside the margins, _aspect > 1 (a
  wide rectangle).
- tests/test_artifact_browser.py:84, 117-123, 899 — the kind selector is `kind_combo`, its items carry the kind
  key as item data in KIND_DIRS order, KIND_LABELS has exactly the KIND_DIRS keys in that order, and
  KIND_LABELS['fdt'] == 'FDT measurements, sweeps and comparisons'.
- tests/test_artifact_browser.py:136, 457, 890, 1060, 1310 — verbatim screen text: status 'Nothing here yet.'; a
  QLabel 'Note'; btn_sweep_all tooltip contains 'legacy directory'; sweep dialog title 'Remove what no artifact
  accounts for'; status 'Deleted prior second.'.
- tests/test_artifact_browser.py:491-493, 527, 1223-1224 — fix sentences naming places: "Edit it in the Note box
  on the Artifacts screen, or in the 'Note' or 'Comparison note' box on the FDT analysis or Sweep study
  cross-validation tab." and 'Select an artifact in the list on the Artifacts screen.'; L1223 text 'No artifact
  is selected.'.
- tests/test_artifact_browser.py:1190-1196 — the model dialogs' titles in order: 'Cannot edit model', 'A task is
  running', 'Delete model', 'Delete failed', with their icons.
- tests/test_artifact_browser.py:174-176, 1093-1094 — ArtifactScreen must not be a BasePanel and must not be in
  _all_panels(); during a live run its note box and Delete stay enabled.
- tests/test_simulate.py:502-507 — "Set it in the 'Steps / frame' box on the Live simulation tab.", "Set it in
  the 'Max FPS' box on the Live simulation tab.", "Set it in the 'T_obs (s)' box on the Infer or Live simulation
  tab."
- tests/test_user_models.py:705-707 — seven builder keys' fix sentences must end 'on the Model Builder screen.';
  L699 the forcing-kind combo's index 1 must be the sinusoidal kind.
- tests/test_user_models.py:1247-1257 — build_app([]) returns (app, window); the app has a multi-size window
  icon; window.windowTitle() == 'PRISM'; it installs the default store from PRISM_ARTIFACTS and the root log
  handler.
- tests/test_refusals.py:607-613 — a freshly built SimulatePanel's frame_steps.text() and fps.text() must equal
  the registry defaults of 'frame_steps' and 'fps'.
- tests/test_tool.py:3099, 3972, 3977 — core.gui.screens.artifact_screen must keep the module constants
  _FINGERPRINT_DEPENDENT, _CACHE_NO_LOG and _NO_RUN_LOG, equal to the command-line tool's strings.
- tests/test_fdt_user.py:1308 — the window must offer no control for dt_nd, psd_t_obs_nd, t_obs_periods,
  burn_in_nd, freq_bounds (gui fix_sentence == '').
- tests/test_artifact_store.py:3791 — the set of @public_entry functions under core is exact; a window function
  carrying the decorator fails. tests/test_artifact_store.py:2192-2198 — window code must stay under core/ or
  the directory/file closure fails.
- tests/test_source_hygiene.py:220-232 — every comment, docstring and string under core/gui and every page under
  docs/guide (window.md included) must be free of working-document labels.
- tests/test_docs.py:128-133 with docs/guide/window.md:11, 34, 67, 85, 115, 135, 224, 261, 285, 332 — 38
  anchored links from eight guide pages target the window page's ten headings (screens,
  the-parameter-inference-tabs, one-run-at-a-time, the-dialogs, what-a-run-shows, the-inference-settings,
  settings-that-are-not-speed-dials, the-fisher-settings-on-a-resumed-run, what-the-window-remembers,
  supported-models); tests/test_docs.py:136-145 requires the page's first two non-blank lines to be '# Title'
  and 'Checked against commit <hash>.'.
- tests/conftest.py:97-101 — InferenceScreen() must build with no arguments and offer install_config(cfg),
  session.inf_prior, session.posterior and refresh_gates(); tests/test_settings_persistence.py:753 —
  ConfigPanel(None) must build alone.

