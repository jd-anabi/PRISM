"""``python -m core``: the PRISM command-line tool.

The prompt-driven menu that used to live here was retired with piece 2's D1. The tool's entry point
(``core/tool/``) lands with the next commit; until then this stub says so, rather than importing half
a CLI that no longer exists. The GUI is ``python -m core.gui``.
"""
if __name__ == "__main__":
    print("the command-line tool arrives with the next commits; the GUI is `python -m core.gui`")
    raise SystemExit(2)
