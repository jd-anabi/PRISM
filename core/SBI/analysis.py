import torch

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