# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as ss
import structlog
from numpy.typing import NDArray

from stripepy.data_structures import SparseMatrix
from stripepy.utils import zero_columns, zero_rows


def run(
    matrix: ss.csr_matrix,
    genomic_belt: int,
    resolution: int,
    roi: Optional[Dict] = None,
    logger=None,
) -> Tuple[ss.csr_matrix, Optional[ss.csr_matrix], Optional[ss.csr_matrix]]:
    """
    Preprocess the given input matrix.

    Parameters
    ----------
    matrix: ss.csr_matrix
        upper-triangular CSR matrix to be processed.
    genomic_belt: int
        belt expressed in base-pairs corresponding to the region around the matrix diagonal with
        the interactions to be processed.
        Interactions outside this region are dropped.
    resolution: int
        matrix resolution in base-pairs.
    roi: Optional[Dict]
        dictionary with the coordinates of the region of interest.
    logger:
        logger

    Returns
    -------
    ss.csr_matrix
        the pre-processed sparse matrix with genome-wide interactions spanning the upper-triangular region.
    Optional[ss.csr_matrix]
        the raw matrix with interactions spanning the region of interest.
        None if roi is None.
    Optional[ss.csr_matrix]
        the processed matrix with interactions spanning the region of interest.
        None if roi is None.

    All three matrices have the same shape as the input matrix.
    """
    assert isinstance(matrix, ss.csr_matrix)
    assert genomic_belt > 0
    assert resolution > 0

    if logger is None:
        logger = structlog.get_logger().bind(step="IO")

    # We need to extend the RoI to make sure we have all the data required to calculate the local pseudodistributions
    extension_window = genomic_belt // resolution
    if roi is None:
        extended_roi = None
    else:
        extended_roi = {
            "matrix": [
                max(0, roi["matrix"][0] - extension_window),
                min(matrix.shape[0], roi["matrix"][1] + extension_window),
            ]
        }

    roi_matrix_raw = _extract_region_of_interest(matrix, extended_roi)

    # TODO make madmax params tunable
    # TODO add flag to skip madmax filtering
    logger.bind(step=(1, 1)).info("looking for bad bins using the MAD-max filter...")
    bias = ss.csr_array(_madmax_filter(matrix))

    matrix = matrix.multiply(bias)
    matrix = matrix.transpose(copy=False)
    matrix = matrix.multiply(bias.T)
    matrix = matrix.transpose(copy=False)
    matrix.eliminate_zeros()  # TODO this may be redundant

    logger.bind(step=(1, 1)).info(f"MAD max filter masked {bias.shape[0] - bias.sum():.0f}/{bias.shape[0]} bins")

    logger.bind(step=(1, 2)).info("applying log-transformation")
    matrix_proc = _log_transform(matrix)

    logger.bind(step=(1, 3)).info("projecting interactions onto [1, 0]")
    scaling_factor = matrix_proc.max()
    matrix_proc /= scaling_factor

    if roi_matrix_raw is None:
        roi_matrix_proc = None
    else:
        roi_matrix_proc = _log_transform(roi_matrix_raw) / scaling_factor

    return matrix_proc, roi_matrix_raw, roi_matrix_proc  # noqa


def _madmax_filter(
    matrix: ss.csr_matrix,
    min_nnz: int = 10,
    min_count: int = 0,
    mad_max: float = 5.0,
) -> NDArray[float]:
    """
    TODO document
    https://github.com/open2c/cooler/blob/a981796113e3045267b600537ede7617629196ca/src/cooler/_balance.py#L374-L409
    """
    assert min_nnz >= 0
    assert min_count >= 0
    assert np.isfinite(mad_max)
    assert mad_max >= 0

    def mad(v: NDArray[float], axis: Optional[int] = None) -> NDArray[float]:
        return np.median(np.abs(v - np.median(v, axis)), axis)

    marg = matrix.sum(axis=0)
    if isinstance(marg, np.matrix):
        marg = marg.A1

    bias = matrix.count_nonzero(axis=0) >= min_nnz
    bias &= marg >= min_count

    logNzMarg = np.log(marg[marg > 0])
    med_logNzMarg = np.median(logNzMarg)
    dev_logNzMarg = mad(logNzMarg)
    cutoff = np.exp(med_logNzMarg - mad_max * dev_logNzMarg)
    bias[marg < cutoff] = False

    return bias.astype(float)


def _log_transform(matrix: SparseMatrix) -> SparseMatrix:
    """
    Apply a log-transform to a sparse matrix ignoring (i.e. dropping) NaNs.

    Parameters
    ----------
    matrix : SparseMatrix
        the sparse matrix to be transformed

    Returns
    -------
    SparseMatrix
        the log-transformed sparse matrix
    """

    matrix.data[np.isnan(matrix.data)] = 0
    matrix.eliminate_zeros()
    return matrix.log1p()  # noqa


def _extract_region_of_interest(
    ut_matrix: ss.csr_matrix,
    roi: Optional[Dict[str, List[int]]],
) -> Optional[ss.csr_matrix]:
    """
    Extract a region of interest (ROI) from the sparse matrix ut_matrix

    Parameters
    ----------
    ut_matrix: ss.csr_matrix
        the upper-triangular sparse matrix to be processed
    roi: Optional[Dict[str, List[int]]]
        dictionary with the region of interest in matrix ('matrix') and genomic ('genomic') coordinates

    Returns
    -------
    Optional[ss.csr_matrix]
        sparse matrix with the same shape as the input matrix, but without interactions outside the given roi.
        None if roi is None.
    """

    if roi is None:
        return None

    assert isinstance(ut_matrix, ss.csr_matrix)

    i0, i1 = roi["matrix"]
    idx = np.setdiff1d(np.arange(ut_matrix.shape[0]), np.arange(i0, i1 + 1))
    matrix_roi = zero_rows(ut_matrix, idx)
    matrix_roi = zero_columns(matrix_roi.tocsc(), idx)

    # make matrix symmetric
    return matrix_roi + ss.triu(matrix_roi, k=1, format="csc").T
