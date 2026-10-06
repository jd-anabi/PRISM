# Brief for the PRISM mock-ups (scratch; not part of the repository)

## What PRISM is

A research program for the biophysics of the hair bundle, the mechanical sensor of the inner ear's
hair cells. It simulates noisy models of a bundle's motion (the built-in models are called
Nadrowski, Hopf and BP; a user can define more), measures things on them, and infers a cell's
model parameters from recordings by simulation-based inference: it simulates millions of
recordings, trains a neural network on them (the "posterior"), checks that the network is honest
about its uncertainty, and then runs it on a simulated cell or on bench recordings. It is a
desktop program for Windows built with Qt (PySide6). Its window opens at 1300 x 820 px and may be
shrunk to 900 x 600.

## Who it is for, and what the owner asked for

The owner is the scientist who wrote it and runs it. It may be released with a paper, so strangers
will open it with no one to ask. The owner's words: "I require extreme clarity for this science
application." What the owner dislikes today:

- Home is "5 buttons just slapped in the center"; it should be intuitive and good to look at.
- The screens under each part are "very cluttered, unintuitive, and hard to use/read".
- Getting around the whole program is unintuitive.
- The progress bars are confusing: cluttered and not interpretable.
- The run's console text is shown by default; people "need to only see what they need to by
  default".
- The design language as a whole; the owner has no direction in mind and wants proposals.

Decided so far: the program stays on its desktop toolkit; the redesign also makes the window fit
for a training that runs for days (see "A long run"); everything else is open.

## The parts of the program (real content: use it, do not invent other features)

1. **Parameter inference**, a chain of steps, each needing the one before:
   1. *Model and options*: which model (Nadrowski by default), its units, and whether an
      observation is a passive recording plus driven recordings at several frequencies
      (the "frequency response", written χ(ω)) or a passive plus one driven recording.
   2. *Prior*: choose a bounds file (it declares which parameters are inferred and their ranges;
      e.g. `master_tier1.txt`, 13 parameters), then build a prior (about a minute and a half) or
      load a saved one (e.g. "retrain_prior").
   3. *Training*: simulate the training set, then fit the network. Main settings: Batches
      (10,000), rows per batch (2,048, automatic), network width (256) and depth (10).
      Rarely changed: learning rate 0.001, patience 20 epochs, an epoch limit, three settings of
      the rotation step, the graphics-card memory limit (6 GiB), how often the simulations are
      saved (every 50 batches), and whether the run must continue an existing set of simulations.
      A run can be given a name and a note before it starts.
   4. *Calibration check*: 2,000 test datasets at 200 operating points; ends in a verdict, PASS or
      FAIL, with one row per parameter.
   5. *Inference*: run the trained network on a simulated cell (pick a cell file) or on your own
      recordings (a passive recording, driven recordings each with its frequency in Hz, the
      drive amplitude in newtons, the recording length in seconds). Result: a table of each
      parameter's 5 %, 50 % and 95 % values and a corner plot.
   6. *Narrowing round*: retrain around one observation for a sharper answer (level 0.999,
      5 directions, its own simulation budget).
   Further checks exist that decide whether a trained network passes; today they can only be run
   from a command line and are to get screens: a check of the frequency band, a check of how many
   driven recordings are unusable, an identifiability check, and a check that the network really
   uses each part of its input.
2. **Effective temperature (FDT)**: a measurement on one cell, and a sweep study over two
   parameters; saved runs can be compared.
3. **Live simulation**: watch a cell's bundle move as it is simulated (a displacement trace and a
   top-down view); save a video.
4. **Reduction map**: an analytical map from the Nadrowski model to the Hopf normal form.
5. **Records**: everything the program writes is kept as a record with a name, a time, a note and
   its settings. Kinds: priors, simulation sets, posteriors, observations, calibrations,
   inferences, checks, effective-temperature runs. Records depend on each other (a posterior on
   its prior and its simulation set).
6. **Settings** (appearance: follow the system, light, dark), **your own models** (a model
   builder), and **help**.

## A long run (real figures from the last one; use them)

- Simulation: 10,000 batches of 2,048 = 20,480,000 simulated recordings; about 9.7 s per batch,
  about 27 hours; saved every 50 batches, so a stop loses nothing.
- Fit: about 63 minutes per epoch, about 130 epochs expected, so five to six days. It was at
  epoch 62 when stopped. **A fit cannot yet be resumed: stopping it loses the whole fit** (the
  simulations are kept).
- Memory on the 16 GB graphics card: the longest batch held 10.4 GiB with the limit at 6 GiB; with
  no limit the run died after 23 minutes.
- Health figures a person should be able to read: failed simulations 0.15 %; unusable driven
  recordings 40.1 % against 37.9 % expected (within 3 points is fine).
- Before starting, the window must say: how many simulations, whether this continues an existing
  set or starts a new one (and why), whether the worst batch fits in the card's free memory, and
  roughly how long it will take.
- During the run it must say: which stage, how far, how long so far, when it should end, and
  whether anything needs attention. The full console text is available on request and kept in a
  file; it is not the default view.
- Stopping must say what is kept and what is lost before it happens.
- Only one run at a time in the whole program. Settings are locked while it runs; moving around
  and reading results stay possible.

## Hard constraints on every mock-up

- **It must be buildable with a desktop toolkit's standard controls**: buttons, text and number
  fields, drop-downs, check boxes, radio buttons, lists, tables, trees, tabs, splitters, scroll
  areas, progress bars, dialogs, plus simple custom-drawn pieces (a step indicator, a status
  strip, charts). Flat fills, 1 px borders, rounded corners and simple icons are fine. No blur or
  glass, no decorative gradients, no drop shadows that carry meaning, no effect that appears only
  when the mouse hovers, no animation beyond a short fade or slide.
- **Fonts**: only ones installed on this machine (Segoe UI Variable, Bahnschrift, Cambria, Calibri,
  Candara, Corbel, Constantia, Consolas, Franklin Gothic Medium, Georgia, Palatino Linotype, Sitka,
  Trebuchet MS, Verdana, Tahoma, Arial) or Inter, which the program already bundles in two weights
  (load it with @font-face from
  `file:///C:/Users/J/PycharmProjects/PRISM/core/gui/assets/fonts/Inter-Regular.ttf` and
  `Inter-SemiBold.ttf`). You may NAME another open-licence typeface as the intended one, but the
  page must render with an installed one and its note must say so.
- **Nothing from the network**: no web fonts, no CDN, no remote image.
- **Light and dark** both, through the template's switch.
- **Words**: plain, sentence case, the user's terms and not the program's insides; numbers carry
  units; an action keeps one name everywhere; a destructive action says what is lost. No labels
  in capitals, no strings of facts joined by middle dots, no arrows glued to button text, no
  label sitting above every heading.
- **Clarity**: text contrast at least 4.5 to 1; a state is never shown by colour alone (add a word
  or an icon); keyboard focus is visible; at most about four choices compete for attention at any
  one decision.
- Icons: simple inline SVG or none. No emoji.

## Files

- Template to copy: `_template.html` in this folder (read its comments: lengths in rem, 1rem =
  10 px of the real window; no script needed).
- To take a picture of a screen at the real size:
  `powershell -NoProfile -File shot.ps1 -Html <page.html> -Out shots/<name>-<n>-<light|dark>.png -Screen <n> -Theme <light|dark>`
  then read the picture and look at it critically.
- Pictures of the window as it is today: `../shots-native/` (light-*.png and dark-*.png).
