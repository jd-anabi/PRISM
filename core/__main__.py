from core.cli import select_mode, build_sim_config, build_fdt_config, build_reduction_config
from core.orchestrator import run
from core.FDT.fdt_pipeline import run_fdt
from core.Reduction import run_reduction_map

if __name__ == '__main__':
    mode = select_mode()
    if mode == "FDT":
        run_fdt(build_fdt_config())
    elif mode == "REDUCTION":
        run_reduction_map(build_reduction_config())
    else:
        run(build_sim_config())