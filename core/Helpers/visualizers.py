from os import PathLike

import corner
import numpy as np
import torch
from matplotlib import pyplot as plt

# === GENERAL VISUALIZERS ===
def plot(x: np.ndarray, y: np.ndarray, scatter: bool = False, title: str = None, labels: tuple = None, lims: list = None, hlines: tuple = None, tight: bool = True, sink=None) -> "plt.Figure":
    """Line/scatter plot. Builds an explicit Figure and returns it. ``sink``, if given, is a callable
    ``(title, fig) -> None`` that handles display (e.g. a GUI canvas); when None, falls back to the
    legacy blocking ``plt.show()`` so the CLI is unchanged."""
    fig, ax = plt.subplots()
    if scatter:
        ax.scatter(x, y)
    else:
        ax.plot(x, y)

    if title is not None:
        ax.set_title(title)

    if labels is not None:
        ax.set_xlabel(labels[0])
        if len(labels) > 1:
            ax.set_ylabel(labels[1])

    if lims is not None:
        ax.set_xlim(*lims[0])
        if len(lims) > 1:
            ax.set_ylim(*lims[1])

    if hlines is not None:
        ax.hlines(*hlines, linestyle='--', color='r')

    if tight:
        fig.tight_layout()
    if sink is not None:
        sink(title or "Plot", fig)
    else:
        plt.show()
    return fig

# === DISTRIBUTION VISUALIZERS ===
def visualize_dist(dist: torch.distributions.Distribution, labels: list, n_samples: int = 10000, save_path: str | PathLike[str] = None, title: str = "Distribution", sink=None) -> "plt.Figure":
    """Corner plot of a distribution. Returns the Figure. ``sink`` (title, fig)->None handles display
    for a GUI; when None, falls back to blocking ``plt.show()`` (CLI-unchanged)."""
    # sample from distribution
    samples = dist.sample((n_samples,)).cpu().numpy()

    # generate the corner plot
    figure = corner.corner(samples, labels=labels, show_titles=True, title_fmt=".2f", plot_datapoints=False, plot_density=True, fill_contours=True, smooth=1.0)

    # save distribution visualization, then display it (sink for GUI, else blocking show)
    if save_path is not None:
        figure.savefig(save_path)
    if sink is not None:
        sink(title, figure)
    else:
        plt.show()
    return figure

# === POSTERIOR ANALYSIS VISUALIZERS ===
def plot_ppc(ppc_results: dict, ground_truth: list = None, param_names: list = None,
             n_samples: int = None, fig_size: tuple = (16, 7)) -> plt.Figure:
    """
    Plot posterior predictive check z-scores.

    :param ppc_results: Dictionary returned by analysis.posterior_predictive_check().
    :param ground_truth: Ground truth parameter values (for subtitle display).
    :param param_names: LaTeX-formatted parameter names (for subtitle display).
    :param n_samples: Number of posterior samples used to generate simulated statistics.
    :param fig_size: Figure size.
    :return: matplotlib Figure.
    """
    z_scores = ppc_results["z_scores"]
    if isinstance(z_scores, torch.Tensor):
        z_scores = z_scores.cpu().detach().numpy()

    n_stats = len(z_scores)
    indices = np.arange(n_stats)

    valid_mask = np.isfinite(z_scores)
    valid_z = z_scores[valid_mask]
    abs_z = np.abs(valid_z)

    # classify points: blue (|z| <= 1), orange (1 < |z| <= 2), red (|z| > 2)
    outside_mask = valid_mask & (np.abs(z_scores) > 2)
    warning_mask = valid_mask & (np.abs(z_scores) > 1) & (np.abs(z_scores) <= 2)
    safe_mask = valid_mask & (np.abs(z_scores) <= 1)
    invalid_mask = ~valid_mask

    fig, ax = plt.subplots(figsize=fig_size)

    # shaded |z| < 2 region
    ax.axhspan(-2, 2, alpha=0.1, color='green', label=r'$|z| < 2$ region')

    # reference lines
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(2, color='red', linestyle='--', linewidth=0.8, label=r'$|z| = 2$')
    ax.axhline(-2, color='red', linestyle='--', linewidth=0.8)

    # plot points by category
    ax.scatter(indices[safe_mask], z_scores[safe_mask], c='steelblue',
               s=40, alpha=0.7, edgecolors='none', zorder=3)
    ax.scatter(indices[warning_mask], z_scores[warning_mask], c='orange',
               s=40, alpha=0.8, edgecolors='none', zorder=3)
    ax.scatter(indices[outside_mask], z_scores[outside_mask], c='red',
               s=50, alpha=0.9, edgecolors='none', zorder=4)
    if invalid_mask.any():
        ax.scatter(indices[invalid_mask], np.zeros(invalid_mask.sum()), c='gray',
                   s=50, marker='x', linewidths=1.5, zorder=4, label='Invalid (zero variance)')

    ax.set_xlabel('Statistic Index')
    ax.set_ylabel('Z-Score')
    ax.set_ylim(-3.5, 3.5)
    ax.legend(loc='upper left')

    # title
    title = "Posterior Predictive Check"
    if n_samples is not None:
        title += f" (N = {n_samples} samples)"
    subtitle_parts = []
    if ground_truth is not None and param_names is not None:
        pairs = [f"{name} = {val}" for name, val in zip(param_names, ground_truth)]
        subtitle_parts.append("Ground Truth: " + ", ".join(pairs))
    if subtitle_parts:
        title += "\n" + subtitle_parts[0]
    ax.set_title(title)

    # summary statistics box
    num_total = n_stats
    textstr = (
        f"Mean $|z|$: {abs_z.mean():.3f}\n"
        f"Max $|z|$: {abs_z.max():.3f}\n"
        f"Coverage (90%): {ppc_results['coverage_90'] * 100:.1f}%\n"
        f"Outside interval: {ppc_results['num_outside']}/{num_total}\n"
        f"Invalid stats: {ppc_results['num_invalid']}/{num_total}"
    )
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
    ax.text(0.99, 0.99, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props,
            family='monospace')

    # add "Summary Statistics" header above the box
    ax.text(0.99, 1.0, "Summary Statistics", transform=ax.transAxes, fontsize=10,
            fontweight='bold', verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

    plt.tight_layout()
    return fig


def plot_posterior_vs_truth(t: np.ndarray, x_true: np.ndarray,
                           x_mean: np.ndarray = None, x_median: np.ndarray = None,
                           x_samples: np.ndarray = None, n_show: int = 10,
                           fig_size: tuple = (14, 5)) -> plt.Figure:
    """
    Overlay posterior-simulated trajectories on top of ground truth data.

    :param t: Time array (steady-state portion), shape (T,).
    :param x_true: Ground truth x-position time series, shape (T,).
    :param x_mean: Posterior-mean-parameter trajectory, shape (T,). Optional.
    :param x_median: Posterior-median-parameter trajectory, shape (T,). Optional.
    :param x_samples: Posterior sample trajectories, shape (N, T).
    :param n_show: Number of individual sample trajectories to display.
    :param fig_size: Figure size.
    :return: matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    # confidence band from all samples
    if x_samples is not None and len(x_samples) > 1:
        mean = x_samples.mean(axis=0)
        std = x_samples.std(axis=0)
        ax.fill_between(t, mean - 2 * std, mean + 2 * std,
                        alpha=0.15, color='steelblue', label=r'Posterior $\pm 2\sigma$')

        # individual sample trajectories
        show_idx = np.random.choice(len(x_samples), size=min(n_show, len(x_samples)), replace=False)
        for i, idx in enumerate(show_idx):
            ax.plot(t, x_samples[idx], color='steelblue', alpha=0.25, linewidth=0.5,
                    label='Posterior samples' if i == 0 else None)

    ax.plot(t, x_true, color='black', linewidth=1.2, label='Ground truth')
    if x_median is not None:
        ax.plot(t, x_median, color='red', linewidth=1.0, linestyle='--', label='Posterior median')
    if x_mean is not None:
        ax.plot(t, x_mean, color='darkorange', linewidth=1.0, linestyle='-.', label='Posterior mean')

    ax.set_xlabel('Time')
    ax.set_ylabel('x(t)')
    ax.set_title('Posterior vs Ground Truth')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_training_loss(diagnostics: dict, save_path: str | PathLike[str] = None,
                       fig_size: tuple = (10, 5)) -> plt.Figure | None:
    """
    Plot the NPE per-epoch training/validation loss curve for a convergence check.

    Distinguishes "under-fit" (validation loss still descending near the best epoch ->
    train longer / raise capacity) from "converged" (clean plateau well before the best
    epoch -> remaining wide marginals are a data/identifiability limit, not under-fit).

    :param diagnostics: dict from train_nn carrying 'training_loss', 'validation_loss',
                        and optionally 'best_validation_loss', 'epochs_trained',
                        'stop_after_epochs'.
    :param save_path: optional path to save the figure.
    :param fig_size: figure size.
    :return: the matplotlib Figure, or None if no validation curve is present.
    """
    val = diagnostics.get("validation_loss") or []
    train = diagnostics.get("training_loss") or []
    if len(val) == 0:
        print("plot_training_loss: no validation_loss curve in diagnostics; nothing to plot.")
        return None

    fig, ax = plt.subplots(figsize=fig_size)
    if len(train):
        ax.plot(np.arange(1, len(train) + 1), train, color='steelblue', label='training loss')
    ax.plot(np.arange(1, len(val) + 1), val, color='darkorange', label='validation loss')

    best_epoch = int(np.argmin(val)) + 1
    ax.axvline(best_epoch, color='green', linestyle='--', linewidth=1.0,
               label=f'best epoch ({best_epoch})')
    sae = diagnostics.get("stop_after_epochs")
    if sae:
        ax.axvspan(best_epoch, len(val), alpha=0.08, color='red',
                   label=f'early-stop window ({sae}-epoch patience)')

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (lower = better)')
    title = 'NPE training / validation loss'
    et, bvl = diagnostics.get("epochs_trained"), diagnostics.get("best_validation_loss")
    if et is not None and bvl is not None:
        title += f'  (epochs={et}, best val={bvl:.4f})'
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return fig