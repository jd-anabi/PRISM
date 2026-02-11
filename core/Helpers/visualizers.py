from os import PathLike

import corner
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import stats

# === GENERAL VISUALIZERS ===
def plot(x: np.ndarray, y: np.ndarray, scatter: bool = False, title: str = None, labels: tuple = None, lims: list = None, hlines: tuple = None, tight: bool = True) -> None:
    if scatter:
        plt.scatter(x, y)
    else:
        plt.plot(x, y)

    if title is not None:
        plt.title(title)

    if labels is not None:
        plt.xlabel(labels[0])
        if len(labels) > 1:
            plt.ylabel(labels[1])

    if lims is not None:
        plt.xlim(*lims[0])
        if len(lims) > 1:
            plt.ylim(*lims[1])

    if hlines is not None:
        plt.hlines(*hlines, linestyle='--', color='r')

    if tight:
        plt.tight_layout()
    plt.show()

# === DISTRIBUTION VISUALIZERS ===
def visualize_dist(dist: torch.distributions.Distribution, labels: list, n_samples: int = 10000, save_path: str | PathLike[str] = 'prior_corner_plot.png') -> None:
    # sample from distribution
    samples = dist.sample((n_samples,)).cpu().numpy()

    # generate the corner plot
    figure = corner.corner(samples, labels=labels, show_titles=True, title_fmt=".2f", plot_datapoints=False, plot_density=True, fill_contours=True, smooth=1.0)

    # save distribution visualization and show it
    plt.savefig(save_path)
    plt.show()

# === POSTERIOR ANALYSIS VISUALIZERS ===
def plot_sbc(ranks: torch.Tensor, param_names: list, m: int, fig_size: tuple = None) -> plt.Figure:
    """
    Plot SBC rank histograms for each parameter.

    Under correct calibration, ranks are uniform over {0, ..., M},
    so histograms should be flat.
    """
    n, d = ranks.shape
    n_bins = 20  # number of histogram bins

    if fig_size is None:
        fig_size = (3 * d, 3)

    fig, axes = plt.subplots(1, d, figsize=fig_size)
    if d == 1:
        axes = [axes]

    # expected count per bin under uniformity
    expected = n / n_bins

    # 99% confidence band (binomial)
    # Each bin count ~ Binomial(N, 1/n_bins)
    ci_low, ci_high = stats.binom.interval(0.99, n, 1 / n_bins)

    for j, ax in enumerate(axes):
        ax.hist(ranks[:, j].numpy(), bins=n_bins, range=(0, m),
                density=False, alpha=0.7, edgecolor='black')
        ax.axhline(expected, color='k', linestyle='--', label='Expected')
        ax.axhspan(ci_low, ci_high, alpha=0.15, color='gray',
                   label='99% CI')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Count')
        ax.set_title(f'${param_names[j]}$')

    plt.tight_layout()
    return fig


def plot_expected_coverage(alpha_values: torch.Tensor, fig_size: tuple = (5, 5)) -> plt.Figure:
    """
    Plot empirical CDF of alpha values against the diagonal.

    Under correct calibration, this should follow the identity line.
    """
    n = len(alpha_values)
    sorted_alpha = np.sort(alpha_values.cpu().detach().numpy())
    empirical_cdf = np.arange(1, n + 1) / n

    # Kolmogorov-Smirnov confidence band
    from scipy.stats import kstwobign
    # 95% confidence bandwidth
    c_alpha = 1.36  # approximate critical value for alpha=0.05
    band_width = c_alpha / np.sqrt(n)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(sorted_alpha, empirical_cdf, 'r-', linewidth=2,
            label='Empirical')
    ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
    ax.fill_between(np.linspace(0, 1, 100),
                    np.linspace(0, 1, 100) - band_width,
                    np.linspace(0, 1, 100) + band_width,
                    alpha=0.15, color='gray', label='95% KS band')
    ax.set_xlabel('Nominal coverage')
    ax.set_ylabel('Empirical coverage')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.legend()

    return fig