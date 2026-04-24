import torch
from sbi.inference.posteriors import DirectPosterior
from torch.distributions.transforms import Transform

from core.SBI import pipeline
from core.config import CAL_RUN_SIZE
from core.SBI.reparam import _transform_device

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
def gen_cal_data(model: str, prior: torch.distributions.Distribution,
                 forcing_prior: torch.distributions.Distribution,
                 t: torch.Tensor, steady_idx: int, dt_nd_min: float, n_cal: int,
                 nd_dim: int, forcing_idx: dict, rescale_idx: dict,
                 dt_exp: float = None, t_min_exp: float = None, t_max_exp: float = None,
                 t_scale_bounds: tuple[float, float] = None, theta_transform: Transform | None = None,
                 fixed_dict: dict = None, state_dep_drift: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates calibration data and filtered parameters for model training based on the provided input parameters.

    If theta_transform is provided, `prior` is LATENT. Internally gen_training_data samples
    z, simulates via T(z), and returns latent z as theta_star. This function then applies
    theta_transform to convert theta_star back to physical coordinates, so SBC/coverage
    can compare it directly against the physical-space TransformedPosterior.

    :param model: Name of the model to evaluate. Must be provided as a string.
    :param prior: Distribution object representing the prior over the model parameters.
    :param t: Pre-simulated ND time tensor at finest resolution, provided as a tensor.
    :param steady_idx: Index defining the steady-state position in the simulation points.
    :param dt_nd_min: Finest ND time step of the pre-simulated trajectory.
    :param n_cal: Number of calibration data samples to generate.
    :param dt_exp: Fixed experimental sampling interval (seconds).
    :param t_min_exp: Shortest experimental recording duration (seconds).
    :param t_max_exp: Longest experimental recording duration (seconds).
    :param t_scale_bounds: (lo, hi) bounds on the t_scale rescaling parameter.
    :param fixed_dict: Dictionary containing fixed parameter values for simulation (default is None).
    :param dtype: Data type for tensor computations (default is torch.float32).
    :param device: Device where computations will be performed (default is CPU).
    :return: A tuple containing filtered calibration data (torch.Tensor) and corresponding parameters
             (torch.Tensor) that exclude invalid simulations.
    """
    cal_run_size = min(CAL_RUN_SIZE, n_cal)
    cal_n_runs = max(1, n_cal // cal_run_size)

    cal_data, theta_star = pipeline.gen_training_data(
        model, prior, forcing_prior, t, cal_run_size, cal_n_runs,
        steady_idx, dt_nd_min, nd_dim, forcing_idx, rescale_idx,
        dt_exp=dt_exp, t_min_exp=t_min_exp, t_max_exp=t_max_exp,
        t_scale_bounds=t_scale_bounds, proposal=None,
        theta_transform=theta_transform,  # <-- NEW
        fixed_dict=fixed_dict, state_dep_drift=state_dep_drift,
        dtype=dtype, device=device,
    )

    valid = torch.isfinite(cal_data).all(dim=1) & (torch.abs(cal_data) < 1e15).all(dim=1)
    cal_data = cal_data[valid]
    theta_star_latent = theta_star[valid]

    if theta_transform is not None:
        # Convert latent theta_star to physical for downstream comparison.
        # The transform lives on cfg.hw.device; move cpu tensor there, apply, move back.
        t_device = _transform_device(theta_transform)
        theta_star_phys = theta_transform(theta_star_latent.to(t_device)).cpu()
        return cal_data, theta_star_phys
    else:
        return cal_data, theta_star_latent


def compute_sbc_ranks(posterior: DirectPosterior, theta_star: torch.Tensor, x_calibration: torch.Tensor,
                      m: int, chunk_size: int = 50, dtype: torch.dtype = torch.long, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Computes the Simulation-Based Calibration (SBC) ranks for a given posterior distribution. The SBC ranks
    are determined by comparing samples from the posterior to the true parameter values (`theta_star`).

    This function handles the computation in chunks, which allows for efficient memory usage, especially
    when large data sets or a high number of posterior samples (`m`) need to be processed.

    To determine the ranks, for each calibration input (`x_calibration`) in the provided chunk, samples are
    generated from the posterior conditioned on that input. The ranks of the ground-truth parameter values
    (`theta_star`) are then computed by counting the number of posterior samples less than the parameter values.

    :param posterior: The trained neural posterior object used to draw samples conditioned on calibration inputs.
    :type posterior: DirectPosterior
    :param theta_star: The ground truth parameter values corresponding to the calibration inputs.
                       Its shape is (n, d), where `n` is the number of calibration inputs and `d` is the
                       dimensionality of the parameter space.
    :type theta_star: torch.Tensor
    :param x_calibration: Calibration inputs to condition the posterior samples. Its shape is (n, n_stats),
                          where `n_stats` is the dimensionality of the input space.
    :type x_calibration: torch.Tensor
    :param m: The number of samples to draw from the posterior for each calibration input to estimate ranks.
    :type m: int
    :param chunk_size: The number of calibration inputs to process in each chunk to manage memory usage.
                       Defaults to 50.
    :type chunk_size: int
    :param dtype: The data type used for the computed ranks tensor. Defaults to `torch.long`.
    :type dtype: torch.dtype
    :param device: The device on which computations will be performed. Defaults to `torch.device('cpu')`.
    :type device: torch.device
    :return: A tensor of ranks with shape (n, d), where each entry represents the rank of the corresponding
             true parameter value (`theta_star`) among the posterior samples.
    :rtype: torch.Tensor
    """
    n, d = theta_star.shape
    ranks = torch.zeros((n, d), dtype=dtype)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        x_chunk = x_calibration[start:end].to(device)  # (chunk, n_stats)
        theta_chunk = theta_star[start:end].to(device)  # (chunk, d)

        samples = posterior.sample_batched((m,), x=x_chunk)  # (M, chunk, d)
        ranks[start:end] = (samples < theta_chunk[None]).sum(dim=0).cpu()

    return ranks

def compute_expected_coverage(posterior: DirectPosterior, theta_star: torch.Tensor, x_calibration: torch.Tensor,
                      m: int, chunk_size: int = 50, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Compute the expected coverage of a posterior distribution.

    This function estimates the expected coverage probability of a posterior
    distribution by comparing the log-probabilities of true parameter values
    (theta_star) with the log-probabilities of posterior samples. The function
    handles computations in chunks for efficiency.

    The coverage is computed as the fraction of posterior samples that have a lower
    density than the true parameter, averaged across all samples.

    :param posterior: Neural posterior model providing the `sample` and `log_prob` methods.
    :param theta_star: Tensor containing the true parameter values. Shape is (n, d),
                       where `n` is the number of samples, and `d` is the dimensionality
                       of the parameter space.
    :param x_calibration: Tensor containing the calibration data associated with
                          the parameters. Shape is (n, n_stats), where `n_stats`
                          represents the number of features/statistics.
    :param m: Number of posterior samples to draw per calibration data point.
    :param chunk_size: Optional; size of chunks to split the computation for memory
                       efficiency. Defaults to 50.
    :param dtype: Optional; data type to use for intermediate computations. Defaults
                  to torch.long.
    :param device: Optional; device on which computations are performed. Defaults
                   to CPU.

    :return: A tensor containing the expected coverage probabilities for each
             calibration point. Shape is (n,).
    """
    n = theta_star.shape[0]
    alphas = torch.zeros((n,), dtype=dtype, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        x_chunk = x_calibration[start:end].to(device)  # (chunk, n_stats)
        theta_chunk = theta_star[start:end].to(device)  # (chunk, d)

        samples = posterior.sample_batched((m,), x=x_chunk)  # (M, chunk, d)

        # log-prob of true parameters
        log_prob_true = posterior.log_prob_batched(theta_chunk.unsqueeze(0), x=x_chunk).squeeze(0)  # (chunk,)

        # log-prob of posterior samples
        log_prob_samples = posterior.log_prob_batched(samples, x=x_chunk)  # (M, chunk)

        # fraction of samples with lower density than true parameter
        alpha_chunk = (log_prob_samples < log_prob_true[None, :]).float().mean(dim=0)  # (chunk,)

        alphas[start:end] = alpha_chunk.cpu()

    return alphas