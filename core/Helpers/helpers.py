import os
import sys

import numpy as np
import torch


def get_even_ids(l: int, n: int) -> list:
    """
    Get evenly spaced indices from an array
    :param l: length of array
    :param n: number of evenly spaced indices
    :return: list of evenly spaced indices
    """
    # edge cases
    if n > l:
        raise ValueError('Number of evenly spaced indices cannot be greater than length of array')
    elif n <= 0:
        return []
    elif n == 1:
        return [0]
    ids = [round(i * (l - 1) / (n - 1)) for i in range(n)]
    ids[-1] = l
    return ids

def concat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Concatenate two arrays with the same number of rows
    :param x: first array
    :param y: second array
    :return: concatenated array
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('Both arrays must have same number of rows')
    return np.concatenate((x, y), axis=1)

def repeat2d_r(x: np.ndarray, ensemble_size: int, batch_size: int) -> np.ndarray:
    """
    Repeat a 2D array (n, m) to a new array (n * ensemble_size, m)
    :param x: array to repeat
    :param ensemble_size: the number of times to tile each element
    :param batch_size: total batch size
    :return: read-only repeated array
    """
    if x.shape[0] > batch_size:
        raise ValueError('Array length cannot be greater than batch size')
    elif batch_size % ensemble_size != 0:
        raise ValueError('Batch size must be divisible by ensemble size')
    expanded_x = x[:, np.newaxis, :]
    tiled_x_r = np.broadcast_to(expanded_x, (x.shape[0], ensemble_size, x.shape[1]))
    return tiled_x_r.reshape(x.shape[0] * ensemble_size, x.shape[1])

def clear_screen() -> None:
    """
    Clears the console screen depending on the operating system.

    This function checks the underlying operating system and executes the appropriate
    command to clear the terminal screen. It supports both Windows and Unix-based
    platforms.

    :return: None
    """
    if sys.platform == 'win32':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def condition_gmm_on_param(weights: torch.Tensor, means: torch.Tensor, covariance: torch.Tensor, pidx: int, pval: float) -> tuple:
    """
    Condition a GMM on one parameter taking a fixed value.

    :param weights: (K,) tensor of mixture weights
    :param means: (K, d) tensor of component means
    :param covariance: (K, d, d) tensor of component covariances
    :param pidx: index of the parameter to condition on
    :param pval: the fixed value

    :return: tuple of updated mixture weights (K,), conditional means (K, d-1), and conditional covariances (K, d-1, d-1)
    """
    K, d = means.shape
    keep = [i for i in range(d) if i != pidx]

    new_weights = torch.zeros(K)
    new_means = torch.zeros(K, d - 1)
    new_covs = torch.zeros(K, d - 1, d - 1)

    for k in range(K):
        mu_alpha = means[k, pidx]  # scalar
        mu_rest = means[k, keep]  # (d-1,)

        sigma_aa = covariance[k, pidx, pidx]  # scalar
        sigma_ra = covariance[k, keep, pidx]  # (d-1,)
        sigma_rr = covariance[k][:, keep][keep, :]  # (d-1, d-1)

        # conditional mean and covariance
        new_means[k] = mu_rest + sigma_ra * (pval - mu_alpha) / sigma_aa
        new_covs[k] = sigma_rr - torch.outer(sigma_ra, sigma_ra) / sigma_aa

        # updated weight: w_k * N(param_value | mu_alpha, sigma_aa)
        log_w = (torch.log(weights[k])
                 - 0.5 * torch.log(2 * torch.pi * sigma_aa)
                 - 0.5 * (pval - mu_alpha) ** 2 / sigma_aa)
        new_weights[k] = torch.exp(log_w)

    # normalize weights
    new_weights = new_weights / new_weights.sum()

    return new_weights, new_means, new_covs