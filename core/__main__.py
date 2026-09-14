"""``python -m core <subcommand>``: the PRISM command-line tool (core/tool/). The GUI is ``python -m core.gui``."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # before torch: OMP Error #15 otherwise
import matplotlib
matplotlib.use("Agg")                                   # before any core import
if __name__ == "__main__":
    from core.tool import main
    raise SystemExit(main())
