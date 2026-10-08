"""The four named colours of every figure the engine draws.

A plotting function that gives a line, a band, a marker or a note a colour of its own takes it from
here, so a figure saved from the command line and one saved from the window carry the same ones.
They are the light values of the second to fifth of the window's line colours
(``core.gui.design.SERIES``); a test holds the two equal, and neither module imports the other. The
first of those colours, the ink, is not held: a plotting function reads it from matplotlib's settings
(``text.color``, ``axes.edgecolor``). A line that names no colour takes matplotlib's cycle.

This module imports nothing. The effective-temperature and reduction plotting files do not import
``core.Helpers.visualizers``, which imports torch, and a comparison of saved runs loads no inference
machinery; each can import this file at no cost.
"""
BLUE = "#0072B2"
VERMILION = "#C24E00"
GREEN = "#007A5A"
ORANGE = "#E69F00"
