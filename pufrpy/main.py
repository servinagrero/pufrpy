#!/usr/bin/env python3

import pandas as pd
import numpy as np
import numpy.typing as npt
from numba import njit
from typing import Callable, Union, Tuple

#: 1D Vector of responses
BitVec = npt.NDArray[np.int8]


def rbits(size: Union[int, Tuple[int, ...]], p: float = 0.5) -> BitVec:
    """Wrapper to generate random bits

    Args:
        size: The size of the resulting array
        p: Probability of obtaining 1

    Returns:
        The ratio of bits in the bit vector
    """
    return np.random.choice([1, 0], size, p=[p, 1 - p])


@njit
def hamming_dist(x: BitVec, y: BitVec, norm: bool = False) -> np.float64:
    """Hamming distance between two BitVecs

    If `norm` is True, the distance is divided by the bit vector length

    Args:
        x: First BitVec
        y: Second BitVec
        norm: If True, normalize the result

    Returns:
        The hamming distance between the vectors
    """
    assert x.size == y.size
    hd = np.float64((x ^ y).sum())
    return hd / x.size if norm else hd


@njit
def hamming_weight(x: BitVec, norm: bool = False) -> np.float64:
    """Hamming weight of a BitVec

    Args:
        x: A BitVec
        norm: If True, normalize the result

    Returns:
        The hamming weight of the BitVec
    """
    hw = np.float64(x.sum())
    return hw / x.size if norm else hw


def ratio_bits(x: BitVec) -> np.float64:
    """Ratio of 1s and 0s in a BitVec

    The ratio is positive if there are more 1s than 0s and negative in the opposite case.

    Args:
        x: A BitVec

    Returns:
        The ratio of 1s and 0s
    """
    return (2 * hamming_weight(x, True)) - 1


@njit
def compare_pairwise(
    crps: npt.NDArray[np.int8],
    fn: Callable[[BitVec, BitVec], np.float64],
) -> npt.NDArray[np.float64]:
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


def crps_weight(
    crps: npt.NDArray[np.int8], by_row: bool = True
) -> Union[np.float64, npt.NDArray[np.float64]]:
    """Hamming weight of rows or columns of a 2D or 3D CRP table

    Args:
        crps: A 2D or 3D CRP table
        by_row: `True` to calculate the weight per row. `False` for columns.
            `True` corresponds to uniformity and `False` to bitaliasing.

    Raises:
        ValueError: If `crps` is not 1D, 2D or 3D
    """
    if not crps.ndim in [2, 3]:
        raise ValueError("crps should be 2D or 3D")

    if crps.ndim == 2:
        return np.apply_along_axis(lambda x: hamming_weight(x, True), int(by_row), crps)

    return np.array(
        [crps_weight(crps[s, :, :], by_row) for s in range(crps.shape[0])],
        dtype=np.float64,
    )


def intra_hd(
    crps: npt.NDArray[np.int8], ref: int = 0
) -> Union[np.float64, npt.NDArray[np.float64]]:
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
        raise ValueError("crps should be 1D, 2D or 3D")

    @njit
    def rel_fn(bv: BitVec, ref: int) -> np.float64:
        bv_new = np.delete(bv, ref)
        return (bv_new ^ bv[ref]).sum() / bv_new.size

    if crps.ndim == 1:
        return rel_fn(crps, ref)
    elif crps.ndim == 2:
        return np.apply_along_axis(rel_fn, 0, crps, ref)

    return np.array(
        [intra_hd(crps[s, :, :], ref) for s in range(crps.shape[0])],
        dtype=np.float64,
    )


def reliability(
    crps: npt.NDArray[np.int8], ref: int = 0
) -> Union[np.float64, npt.NDArray[np.float64]]:
    """Reliability of a CRP table

    Args:
        crps: A CRP table

    Returns:
        The reliability of the CRP table. If the table is 2D, returns a 1D array. If the table is 3D, returns a 2D array.
    """
    return 1 - intra_hd(crps, ref)


def uniqueness(crps: npt.NDArray[np.int8]):
    """Uniqueness or Inter Hamming Distance of a CRP table

    Args:
        crps: A CRP table

    Returns:
        The uniqueness of the CRP table
    """
    assert crps.ndim in [2, 3]

    @njit
    def wrapper(x: BitVec, y: BitVec) -> np.float64:
        return hamming_dist(x, y, True)

    if crps.ndim == 2:
        return compare_pairwise(crps, wrapper)
    return np.array(
        [compare_pairwise(crps[s, :, :], wrapper) for s in range(crps.shape[0])],
        dtype=np.float64,
    )
