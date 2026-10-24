"""The window's runs: everything about one background job, from the click that starts it until its end has been
seen.

A module here draws with the toolkit's own painting and never with the figure library: that library lays out its
text with one parser for the whole process and no lock, and a run builds its own figures on its own thread.
"""
