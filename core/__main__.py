import sys

from core import registry
from core.cli import (select_mode, build_sim_config, build_fdt_config,
                      build_reduction_config, build_param_sweep_config, UnitParseError)
from core.orchestrator import run
from core.FDT.fdt_pipeline import run_fdt
from core.FDT.cross_validation import run_param_study_cli
from core.Reduction import run_reduction_map


def main() -> None:
    # User models appear in the CLI model menu too, but selecting one raises a clear "Simulate-only"
    # error in cli.select_model -- the CLI drives the SBI/FDT paths, which user models don't support.
    registry.load_user_models()
    mode = select_mode()
    if mode == "FDT":
        run_fdt(build_fdt_config())
    elif mode == "REDUCTION":
        run_reduction_map(build_reduction_config())
    elif mode == "CROSSVAL":
        cfg, s_grid, temp_grid = build_param_sweep_config()
        run_param_study_cli(cfg, s_grid, temp_grid)
    else:
        run(build_sim_config())


if __name__ == '__main__':
    try:
        main()
    except UnitParseError as e:
        # A bad unit string used to call exit() deep in the parser; now it raises and we exit cleanly here.
        print(f"Error: {e}")
        sys.exit(1)