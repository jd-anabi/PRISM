from core.cli import build_sim_config
from core.orchestrator import run

if __name__ == '__main__':
    cfg = build_sim_config()
    run(cfg)