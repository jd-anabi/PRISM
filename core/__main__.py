from core.cli import select_mode, build_sim_config, build_fdt_config
from core.orchestrator import run
from core.FDT.fdt_pipeline import run_fdt

if __name__ == '__main__':
    mode = select_mode()
    if mode == "FDT":
        run_fdt(build_fdt_config())
    else:
        run(build_sim_config())