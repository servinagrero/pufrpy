#!/usr/bin/env python3

import pandas as pd
import numpy as np
import numpy.typing as npt
from numba import njit
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl

#: Multidimensional array of responses
Responses = npt.NDArray[np.int8]

#: Multidimensional array of floats
NDFloat = npt.NDArray[np.float64]


@njit("float64(int8[:], boolean)")
def hamming_weight(x: Responses, norm: bool = False) -> np.float64:
    """Hamming weight of a BitVec

    NAN values are ignored in the computation

    Args:
        x: A BitVec
        norm: If True, normalize the result

    Returns:
        The hamming weight of the BitVec
    """
    hw = np.float64(np.nansum(x))
    size = len(x[~np.isnan(x)])
    return hw / size if norm else hw


@njit("float64(int8[:], int8[:], boolean)")
def hamming_dist(x: Responses, y: Responses, norm: bool = False) -> np.float64:
    """Hamming distance between two BitVecs

    NAN values in any of the vectors are ignored.

    If `norm` is True, the distance is divided by the bit vector length

    Args:
        x: First BitVec
        y: Second BitVec
        norm: If True, normalize the result

    Returns:
        The hamming distance between the vectors
    """
    assert x.size == y.size
    hd = np.logical_xor(x, y)
    hd = hd[~np.isnan(x) | ~np.isnan(y)]
    return hd.sum(dtype=np.float64) / hd.size if norm else hd.sum(dtype=np.float64)


def ratio_bits(x: Responses) -> np.float64:
    """Ratio of 1s and 0s in a BitVec

    The ratio is positive if there are more 1s than 0s and negative in the opposite case.

    Args:
        x: A BitVec

    Returns:
        The ratio of 1s and 0s
    """
    return (2 * hamming_weight(x, True)) - 1


@njit("float64(int8[:])")
def entropy_bits(x: Responses) -> np.float64:
    """Shannon entropy of a BitVec

    Args:
        x: A BitVec

    Returns:
        The Shannon entropy
    """
    p = hamming_weight(x, True)
    q = 1 - p
    return -((p * np.log2(p)) + (q * np.log2(q)))


@njit("float64[:](float64[:])")
def entropy_prob(x: NDFloat) -> NDFloat:
    """Shannon entropy for a list of probabilities

    Vector is assumed to have the probability of 1

    Args:
        x: Vector of probabilities

    Returns:
        The array of computed entropies
    """
    q = 1 - x
    res = -((x * np.log2(x)) + (q * np.log2(q)))
    res[np.isnan(res)] = 0
    return res


def rbits(
    size: Union[int, Tuple[int, ...]], p: Union[float, NDFloat] = 0.5
) -> Responses:
    """Wrapper to generate random bits

    Args:
        size: The size of the resulting array
        p: Probability of obtaining 1

    Returns:
        The ndarry with the generated bits
    """
    if isinstance(p, float):
        return np.int8(np.random.choice([1, 0], size, p=[p, 1 - p]))
    assert len(p) == np.array(size).prod()
    return np.int8(np.random.binomial(1, p))


def responses_to_df(crps: Responses) -> pd.DataFrame:
    """Convert a CRP table into a DataFrame

    If the CRP table is 2D the columns 'device', 'challenge', 'response' are created.
    In the case of a 3D table, the column 'sample' is added to represent the 3rd dimension.

    Args:
        crps: 2D or 3D CRP table

    Returns:
        The DataFrame
    """
    assert crps.ndim in [2, 3]

    def to_df(arr, sample=None):
        df = pd.DataFrame(arr).reset_index().melt("index")
        df.columns = ["device", "challenge", "response"]
        if not sample is None:
            df["sample"] = sample
        return df

    if crps.ndim == 2:
        return to_df(crps)
    df_list = [to_df(crps[s, :, :], s) for s in range(crps.shape[0])]
    return pd.concat(df_list, ignore_index=True)


def df_to_responses(df: pd.DataFrame) -> Responses:
    """Convert a DataFrame into a 2D or 3D CRP table

    If the column 'sample' is present in the DataFrame, a 2D table is created for each sample of responses.

    Args:
        df: A DataFrame with the columns 'device', 'challenge', 'response' and the optional column

    Returns:
        The 2D or 3D CRP table
    """
    COLUMNS = ["device", "challenge", "response"]
    assert all(c in df.columns for c in COLUMNS)

    def to_matrix(df):
        ndev = len(df["device"].unique())
        nchallenge = len(df["challenge"].unique())
        return df["response"].to_numpy(dtype=np.int8).reshape((ndev, nchallenge))

    if "sample" in df.columns:
        arr_list = [to_matrix(df[df["sample"] == s]) for s in df["sample"].unique()]
        return np.stack(arr_list, axis=1)

    return to_matrix(df)


def crps_heatmap(crps: Responses, fig: Optional[mpl.figure.Figure] = None) -> plt.Axes:
    """Create a heatmap of CRPs
    
    Args:
        crps: 2D Matrix of responses
        fig: Optional matplotlib figure. If `None` a new one is created.

    Returns:
        The axes generated from calling `matshow`
    """
    fig = fig or plt.figure()
    return plt.matshow(crps, fignum=fig.number, aspect="auto")
