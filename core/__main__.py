from core.cli import (select_mode, build_sim_config, build_fdt_config,
                      build_reduction_config, build_param_sweep_config)
from core.orchestrator import run
from core.FDT.fdt_pipeline import run_fdt
from core.FDT.cross_validation import run_param_study_cli
from core.Reduction import run_reduction_map

if __name__ == '__main__':
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