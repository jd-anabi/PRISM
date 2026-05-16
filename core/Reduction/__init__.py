"""
NWK→Hopf analytical reduction map.

Pure analytical pipeline taking ND Nadrowski-NWK parameters and predicting the
ND Hopf normal-form parameters that describe the slow dynamics near the Hopf
bifurcation. No simulation, no SBI, no GPU — numpy/scipy on float64/complex128.

Public API:
  - reduce_nwk_to_hopf(...): single-point reduction returning a ReductionRecord
  - sweep_f_max(cfg, grid): vector of analytical predictions over an f_max sweep
  - run_reduction_map(cfg): CLI entry point (Part A + Part B)
"""
from .reduce import reduce_nwk_to_hopf, ReductionRecord, ReductionFailure
from .sweep import sweep_f_max, run_reduction_map

__all__ = [
    "reduce_nwk_to_hopf",
    "ReductionRecord",
    "ReductionFailure",
    "sweep_f_max",
    "run_reduction_map",
]
