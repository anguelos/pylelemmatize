from typing import List, Tuple, Union
import numpy as np
from ..abstract_mapper import fast_str_to_numpy
from math import inf


def banded_edit_path(a: np.ndarray, b: np.ndarray, band: int, add_null_initial: bool = True) -> Tuple[np.ndarray, np.ndarray, Tuple[np.array, np.array, np.array, np.array]]:
    """
    Banded dynamic-programming alignment (edit distance with unit costs).
    
    Args:
        a: numpy array of single-character strings, shape (m,)
        b: numpy array of single-character strings, shape (n,)
        band: non-negative int, maximum |i - j| misalignment allowed
    
    Returns:
        path: A numpy array of (N+1, 2) containing the coordiantes of the optimal path with respect to a and b.
            The array begins with coordinates [-1, -1] to insdicate START.
        costs: A numpy array of the cumulative costs along the path.
        mat_ins_del_sub: Tuple of boolean arrays
    
    Raises:
        ValueError if alignment is impossible given the band (e.g., |m - n| > band).
    
    Notes:
        - Costs: match = 0, substitution = 1, insertion = 1, deletion = 1.
        - Memory: O((m+n)*band). Time: ~O((m+n)*band).
        - The returned path walks *cells* (i,j) of the DP grid (length m+n minus matches).
    """
    # Backpointer codes
    DIAG, UP, LEFT = 0, 1, 2
    m, n = len(a), len(b)
    if band < 0:
        raise ValueError("band must be non-negative")
    if abs(m - n) > band:
        # End point (m,n) lies outside the band reachable region
        raise ValueError(f"Strings differ in length by {abs(m-n)}, larger than band {band}.")

    # Per row storage (j-starts, costs, backpointers)
    row_starts: List[int] = []
    row_costs: List[np.ndarray] = []
    row_bp: List[np.ndarray] = []

    # Helper to allocate a row segment covering j in [jmin, jmax]
    def alloc_row(jmin: int, jmax: int):
        width = jmax - jmin + 1
        return (jmin, np.full(width, inf, dtype=float), np.full(width, -1, dtype=np.int8))

    # Row 0 initialization (i = 0): we can only do insertions
    jmin0 = max(0, 0 - band)
    jmax0 = min(n, 0 + band)
    j0, cost0, bp0 = alloc_row(jmin0, jmax0)
    # DP[0, j] = j, along the top row (only if within band)
    for j in range(j0, jmax0 + 1):
        cost0[j - j0] = j
        bp0[j - j0] = LEFT if j > 0 else -1  # from (0,j-1) unless at origin
    row_starts.append(j0); row_costs.append(cost0); row_bp.append(bp0)

    # Fill subsequent rows
    for i in range(1, m + 1):
        # Band-limited j-range for row i
        jmin = max(0, i - band)
        jmax = min(n, i + band)
        j_start, cost_row, bp_row = alloc_row(jmin, jmax)

        # Previous row context
        prev_start = row_starts[i - 1]
        prev_cost = row_costs[i - 1]

        for j in range(jmin, jmax + 1):
            k = j - j_start  # index in current row

            # Candidates: deletion (up), insertion (left), substitution/match (diag)
            best_cost = inf
            best_bp = -1

            # UP: from (i-1, j) if that col exists in prev row
            if prev_start <= j <= prev_start + len(prev_cost) - 1:
                c_up = prev_cost[j - prev_start] + 1  # deletion from a
                if c_up < best_cost:
                    best_cost, best_bp = c_up, UP

            # LEFT: from (i, j-1) if j-1 in current row band
            if j - 1 >= j_start:
                c_left = cost_row[k - 1] + 1  # insertion into a (gap in a)
                if c_left < best_cost:
                    best_cost, best_bp = c_left, LEFT

            # DIAG: from (i-1, j-1) if that exists in prev row
            if prev_start <= j - 1 <= prev_start + len(prev_cost) - 1:
                sub = 0 if a[i - 1] == b[j - 1] else 1
                c_diag = prev_cost[(j - 1) - prev_start] + sub
                if c_diag < best_cost:
                    best_cost, best_bp = c_diag, DIAG

            cost_row[k] = best_cost
            bp_row[k] = best_bp

        row_starts.append(j_start); row_costs.append(cost_row); row_bp.append(bp_row)

    # Verify end cell is inside band and finite
    if not (row_starts[m] <= n <= row_starts[m] + len(row_costs[m]) - 1):
        raise ValueError("End cell (m,n) falls outside the band — increase band.")
    end_idx = n - row_starts[m]
    if not np.isfinite(row_costs[m][end_idx]):
        raise ValueError("No valid path within band — increase band.")

    # Traceback from (m, n) to (0, 0)
    path: List[Tuple[int, int]] = []
    i, j = m, n
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        j_start = row_starts[i]
        k = j - j_start
        if k < 0 or k >= len(row_bp[i]):
            # If we ever step just outside due to band edge, it's invalid
            raise RuntimeError("Traceback stepped outside band; increase band.")
        bp = row_bp[i][k]
        if bp == DIAG:
            i, j = i - 1, j - 1
        elif bp == UP:
            i, j = i - 1, j
        elif bp == LEFT:
            i, j = i, j - 1
        else:
            raise RuntimeError("Invalid backpointer during traceback.")
    path.reverse()
    paths = np.array(path) - 1
    is_ins = paths[1:, 0] - paths[: -1, 0] == 0
    is_del = paths[1:, 1] - paths[: -1, 1] == 0
    is_diag = ~(is_del | is_ins)
    try:
        agree =  (a[paths[1:, 0]] == b[paths[1:, 1]])
    except IndexError:
        raise RuntimeError("Index error during agreement check — likely a bug in path construction. a: {}, b: {}, paths: {}".format(a, b, paths))
    is_sub = is_diag & ~agree
    is_match = is_diag & agree
    cost_1 = is_sub | is_del | is_ins
    return paths, cost_1.astype(np.int16), (is_match, is_ins, is_del, is_sub)


def compute_cer(seq1: Union[np.ndarray, str], seq2: Union[np.ndarray, str], band=-1, normalize=False) -> float:
    if isinstance(seq1, str):
        seq1 = fast_str_to_numpy(seq1)
    if isinstance(seq2, str):
        seq2 = fast_str_to_numpy(seq2)
    if seq1.ndim != 1 or seq2.ndim != 1:
        raise ValueError("Input sequences must be 1D arrays or strings.")
    if seq1.shape[0] == 0:
        return float(len(seq2)) if not normalize else 1.0
    if seq2.shape[0] == 0:
        return float(len(seq1)) if not normalize else 1.0
    if band<1:
        band = max(seq1.size, seq2.size)
    _, costs, (_, _, _, _) = banded_edit_path(seq1, seq2, band=band)
    if normalize:
        return float(costs.sum()) / max(len(seq1), len(seq2))
    else:
        return float(costs.sum())

