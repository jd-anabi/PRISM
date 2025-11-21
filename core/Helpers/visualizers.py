from os import PathLike

import corner
import numpy as np
import torch
from matplotlib import pyplot as plt

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

def visualize_dist(dist: torch.distributions.Distribution, labels: list, n_samples: int = 10000, save_path: str | PathLike[str] = 'prior_corner_plot.png') -> None:
    # sample from distribution
    samples = dist.sample((n_samples,)).cpu().numpy()

    # generate the corner plot
    figure = corner.corner(samples, labels=labels, show_titles=True, title_fmt=".2f", plot_datapoints=False, plot_density=True, fill_contours=True, smooth=1.0)

    # save distribution visualization and show it
    plt.savefig(save_path)
    plt.show()