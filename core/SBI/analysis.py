import torch

from core.SBI import pipeline

# === POSTERIOR PREDICTIVE CHECK ===
def posterior_predictive_check(s_obs: torch.Tensor, s_simulated: torch.Tensor) -> dict:
    """
    Performs a posterior predictive check by comparing observed statistics with simulated statistics,
    providing metrics such as z-scores, absolute z-score statistics, and coverage of observed values
    within a confidence interval.

    The method calculates normalized z-scores for observed statistics (`s_obs`) based on the mean and
    standard deviation of simulated statistics (`s_simulated`). It also computes diagnostic metrics,
    such as the fraction of observations within a 90% confidence interval derived from the simulated data,
    as well as counts of invalid statistics (zero variance) and those outside the confidence interval.

    :param s_obs: Observed statistics tensor.
    :param s_simulated: Simulated statistics tensor with samples along the first dimension.
    :return: A dictionary containing calculated z-scores, absolute z-statistics, coverage fraction, and counts of
        invalid observations or those outside the interval.
    """
    # per-statistic mean and std
    s_mean = s_simulated.mean(dim=0)
    s_std = s_simulated.std(dim=0)

    # handle zero variance statistics
    valid_mask = s_std > 1e-10
    z_scores = torch.full_like(s_obs, float('nan'))
    z_scores[valid_mask] = (s_obs[valid_mask] - s_mean[valid_mask]) / s_std[valid_mask]

    # compute metrics only on valid statistics
    valid_z = z_scores[valid_mask]

    lower = torch.quantile(s_simulated, 0.05, dim=0)
    upper = torch.quantile(s_simulated, 0.95, dim=0)
    within_interval = (s_obs >= lower) & (s_obs <= upper)
    coverage_fraction = within_interval.float().mean().item()

    return {
        "z_scores": z_scores,
        "mean_abs_z": valid_z.abs().mean().item(),
        "max_abs_z": valid_z.abs().max().item(),
        "coverage_90": coverage_fraction,
        "num_outside": (~within_interval).sum().item(),
        "num_invalid": (~valid_mask).sum().item()
    }

# === COVERAGE CHECKS ===
def gen_cal_data(model: str, prior: torch.distributions.Distribution, t: torch.Tensor,
                 n_segs: int, steady_idx: int, dt: float, n_cal: int,
                 dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates calibration data and filtered parameters for model training based on the provided input parameters.

    :param model: Name of the model to evaluate. Must be provided as a string.
    :param prior: Distribution object representing the prior over the model parameters.
    :param t: Time points for generating calibration data, provided as a tensor.
    :param n_segs: Number of distinct segments to use for simulation.
    :param steady_idx: Index defining the steady-state position in the simulation points.
    :param dt: Time step size used for numerical simulation.
    :param n_cal: Number of calibration data samples to generate.
    :param dtype: Data type for tensor computations (default is torch.float32).
    :param device: Device where computations will be performed (default is CPU).
    :return: A tuple containing filtered calibration data (torch.Tensor) and corresponding parameters
             (torch.Tensor) that exclude invalid simulations.
    """
    # generate calibration data and parameters
    cal_data, theta_star = pipeline.gen_training_data(model, prior, t, n_cal, 1, n_segs, steady_idx, dt, proposal=None, dtype=dtype, device=device)

    # filter out invalid simulations
    valid = torch.isfinite(cal_data).all(dim=1) & (torch.abs(cal_data) < 1e15).all(dim=1)

    return cal_data[valid], theta_star[valid]