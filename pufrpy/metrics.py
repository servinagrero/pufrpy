#!/usr/bin/env python3

import pandas as pd
import numpy as np
import numpy.typing as npt
from numba import njit
from typing import Callable, Union, Tuple
from dataclasses import dataclass
from typing import Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from pufrpy import Responses, NDFloat, hamming_weight, hamming_dist, entropy_prob


class Metrics:
    """
    Attributes:
        uniformity:
    """

    def __init__(self, crps: Responses):
        if not crps.ndim in [2, 3]:
            raise ValueError("Expected 2D matrix or 3D array")

        if crps.ndim == 2:
            self.samples, (self.devices, self.challenges) = 1, crps.shape
        else:
            self.samples, self.devices, self.challenges = crps.shape

        if self.samples == 1:
            self.uniformity = uniformity(crps)
            self.bitaliasing = bitaliasing(crps)
            self.uniqueness = uniqueness(crps)
            self.reliability = np.nan
        else:
            self.uniformity = np.array(
                [uniformity(crps[s, :, :]) for s in range(self.samples)]
            )
            self.bitaliasing = np.array(
                [bitaliasing(crps[s, :, :]) for s in range(self.samples)]
            )
            self.uniqueness = np.array(
                [uniqueness(crps[s, :, :]) for s in range(self.samples)]
            )
            self.reliability = reliability(crps)

    def __repr__(self):
        def sumary(x):
            mu, sd = np.mean(x), np.std(x)
            return f"({mu:.4f}, {sd:.4f})"

        return (
            f"{'Uniformity:':<13} {sumary(self.uniformity)}\n"
            f"{'Bitaliasing:':<13} {sumary(self.bitaliasing)}\n"
            f"{'Uniqueness:':<13} {sumary(self.uniqueness)}\n"
            f"{'Reliability:':<13} {sumary(self.reliability.mean(0))}\n"
        )

    def _plot_single(self, *args, **kwargs):
        fig = plt.figure(layout="constrained")
        gs = GridSpec(2, 2, figure=fig)

        options = {"bins": 10, **kwargs}

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.uniformity, **options)
        ax1.set_title("Uniformity")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.bitaliasing, **options)
        ax2.set_title("Bitaliasing")

        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(self.uniqueness, **options)
        ax3.set_title("Uniqueness")

        fig.suptitle("Metric summary", fontsize=16)

        return fig, (ax1, ax2, ax3)

    def _plot_multiple(self, *args, **kwargs):
        fig, axs = plt.subplots(
            2, 2, sharey=False, tight_layout=True, gridspec_kw={"height_ratios": [1, 1]}
        )
        options = {"bins": 10, "alpha": 0.6, **kwargs}

        viridis = cm.get_cmap("viridis", 256)
        colors = viridis(np.linspace(0, 1, self.samples))

        # labels = [f'Sample {s}' for s in range(self.samples)]
        # axs[0, 0].hist(self.uniformity, label=labels, **options)

        for s in range(self.samples):
            axs[0, 0].hist(
                self.uniformity[s, :], label=f"Sample {s}", **options, color=colors[s]
            )
            axs[0, 1].hist(
                self.bitaliasing[s, :], label=f"Sample {s}", **options, color=colors[s]
            )
            axs[1, 0].hist(
                self.uniqueness[s, :], label=f"Sample {s}", **options, color=colors[s]
            )

        axs[0, 0].set_title("Uniformity")
        axs[0, 1].set_title("Bitaliasing")
        axs[1, 0].set_title("Uniqueness")
        axs[1, 0].legend()

        axs[1, 1].matshow(self.reliability, aspect="auto")
        axs[1, 1].set_title("Reliability")
        fig.suptitle("Metric summary", fontsize=16)

        return fig, axs

    def plot(self, *args, **kwargs):
        """
        https://towardsdatascience.com/creative-report-designed-only-with-matplotlib-without-office-software-9d8b5af4f9c2
        """
        if self.samples == 1:
            return self._plot_single(*args, **kwargs)
        return self._plot_multiple(*args, **kwargs)

    def summary(self) -> pd.DataFrame:
        def simple(x):
            return {
                "Min": [np.min(x)],
                "Mean": [np.mean(x)],
                "SD": [np.std(x)],
                "Max": [np.max(x)],
            }

        def multiple(m):
            return {
                "Sample": [s for s in range(self.samples)],
                "Min": [np.min(x) for x in m],
                "Mean": [np.mean(x) for x in m],
                "SD": [np.std(x) for x in m],
                "Max": [np.max(x) for x in m],
            }

        METRICS = {
            "Uniformity": self.uniformity,
            "Bitaliasing": self.bitaliasing,
            "Uniqueness": self.uniqueness,
        }
        if self.samples == 1:
            COLUMNS = ["Metric", "Min", "Mean", "SD", "Max"]
            df = pd.DataFrame(columns=COLUMNS)
            for name, metric in METRICS.items():
                metric_df = pd.DataFrame({"Metric": [name], **simple(metric)})
                df = pd.concat([df, metric_df], ignore_index=True)
            return df
        else:
            COLUMNS = ["Metric", "Sample", "Min", "Mean", "SD", "Max"]
            df = pd.DataFrame(
                {"Metric": ["Reliability"], "Sample": [0], **simple(self.reliability)},
                columns=COLUMNS,
            )
            for name, metric in METRICS.items():
                d = {"Metric": np.repeat(name, self.samples), **multiple(metric)}
                df = pd.concat([df, pd.DataFrame(d)], ignore_index=True)
            return df

    def with_entropy(self) -> "Metrics":
        """Convert the metrics to Entropy calculation"""
        from copy import deepcopy

        new = deepcopy(self)
        if self.samples == 1:
            new.uniformity = entropy_prob(self.uniformity)
            new.bitaliasing = entropy_prob(self.bitaliasing)
        else:
            new.uniformity = np.array([entropy_prob(row) for row in self.uniformity])
            new.bitaliasing = np.array([entropy_prob(row) for row in self.bitaliasing])
        return new


# FIXME: How to provide function pointers
# @njit("float64[:](int8[:,:], float64(int8[:], int8[:]))")
@njit
def compare_pairwise(
    crps: Responses,
    fn: Callable[[Responses, Responses], np.float64],
) -> NDFloat:
    """Compare a 2D matrix by pairs of rows

    Args:
        crps: A 2D CRP table
        fn: Function that takes 2 BitVec and returns a float

    Returns:
        An array containing the result of applying `fn` to every pair of rows.

    .todo: Change dtype of result depending on fn
    """
    assert crps.ndim == 2

    ndev, _ = crps.shape
    npairs = (ndev * (ndev - 1)) // 2
    res = np.zeros(npairs, dtype=np.float64)
    count = 0
    for i in range(0, ndev - 2):
        for j in range(i, ndev - 1):
            res[count] = fn(crps[i, :], crps[j, :])
            count += 1
    return res


@njit("float64[:](int8[:,:])")
def uniformity(crps: Responses) -> NDFloat:
    """Uniformity of a 2D matrix"""
    return np.array([hamming_weight(row, True) for row in crps])


@njit("float64[:](int8[:,:])")
def bitaliasing(crps: Responses):
    """Bitaliasing of a 2D matrix"""
    return np.array([hamming_weight(col, True) for col in crps.T])


def equal_to_idx(bv: Responses, ref: int) -> Responses:
    """ """
    hd = np.logical_xor(bv, ~bv[ref])
    hd = hd[~np.isnan(bv)]
    return np.int8(np.delete(hd, ref))


def intra_hd(
    crps: Responses, ref: int = 0
) -> Union[Responses, npt.NDArray[np.float64]]:
    """Intra Hamming Distance of a CRP Table

    Args:
        crps: A BitVec or 2D or 3D CRP table
        ref: Index of the reference sample. By default 0
    Returns:
        The Intra HD of the BitVec or the CRP table. If `crps` is a BitVec

    Raises:
        ValueError: If `crps` is not 1D, 2D, or 3D
    """
    if not crps.ndim in [1, 2, 3]:
        raise ValueError("crps should be a vector, 2D matrix or 3D array")

    if crps.ndim == 1:
        return equal_to_idx(crps, ref)

    if crps.ndim == 2:
        return np.array([equal_to_idx(col, ref) for col in crps.T]).T

    return np.array(
        [intra_hd(crps[:, d, :], ref).mean(0) for d in range(crps.shape[1])]
    )


def reliability(crps: Responses, ref: int = 0) -> Union[np.float64, NDFloat]:
    """Reliability of a CRP table

    Args:
        crps: A CRP table

    Returns:
        The reliability of the CRP table. If the table is 2D, returns a 1D array. If the table is 3D, returns a 2D array.
    """
    return 1 - intra_hd(crps, ref)


def uniqueness(crps: Responses):
    """Uniqueness or Inter Hamming Distance of a CRP table

    Args:
        crps: A CRP table

    Returns:
        The uniqueness of the CRP table
    """
    assert crps.ndim in [2, 3]

    @njit("float64(int8[:], int8[:])")
    def wrapper(x: Responses, y: Responses) -> np.float64:
        return hamming_dist(x, y, True)

    if crps.ndim == 2:
        return compare_pairwise(crps, wrapper)

    return np.array(
        [compare_pairwise(crps[s, :, :], wrapper) for s in range(crps.shape[0])],
        dtype=np.float64,
    )


inter_hd = uniqueness
