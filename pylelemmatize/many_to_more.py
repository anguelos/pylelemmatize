from collections import defaultdict
import sys
from typing import Dict, Optional, Union, Tuple, List
import re
import numpy as np
from math import inf


def pagexml_to_text(pagexml: str) -> str:
    """Convert PAGE XML to plain text.

    Args:
        pagexml (str): The PAGE XML content as a string.

    Returns:
        str: The extracted plain text.
    """
    from lxml import etree

    #root = etree.fromstring(pagexml)
    #texts = []

    # Iterate through all TextLine elements and extract their text
    #for text_line in root.findall('TextLine'):
    #    print(text_line, file=sys.stderr)
    #    line_text = text_line.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Unicode')
    #    if line_text is not None and line_text.text:
    #        texts.append(line_text.text)
    texts = re.findall(r"<Unicode>(.*?)</Unicode>", pagexml, re.DOTALL)
    res = '\n'.join(texts)
    #print(f"Extracted {len(res)} characters from PAGE XML original size: {len(pagexml)}", file=sys.stderr)
    return res





def banded_edit_path(a: np.ndarray, b: np.ndarray, band: int) -> List[Tuple[int, int]]:
    """
    Banded dynamic-programming alignment (edit distance with unit costs).
    
    Args:
        a: numpy array of single-character strings, shape (m,)
        b: numpy array of single-character strings, shape (n,)
        band: non-negative int, maximum |i - j| misalignment allowed
    
    Returns:
        path: list of (i, j) coordinates from (0,0) to (m,n) inclusive,
              following the optimal (minimal-cost) path within the band.
    
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
    return path




def many_to_more_main():
    import fargv
    def generate_pairs(pagexml_files, replace_names: Optional[Union[Tuple[str, str], Dict[str, str]]] = ("Abreviated", "Long")):
        pagexml_files = sorted(pagexml_files)
        by_filename = defaultdict(dict)
        for pagexml_file in pagexml_files:
            path_pieces = pagexml_file.split('/')
            by_filename[path_pieces[-1]][path_pieces[-2]] = pagexml_to_text(open(pagexml_file, 'r').read())

        #print(f"by_filename: {repr(by_filename)}", file=sys.stderr)
        categories_names = sorted(set([tuple(d.keys()) for d in by_filename.values()]))
        assert len(categories_names) == 1
        assert len(list(categories_names)[0]) == 2
        if replace_names is not None:
            if isinstance(replace_names, tuple):
                replace_names = {list(categories_names)[0][0]: replace_names[0], list(categories_names)[0][1]: replace_names[1]}
            assert sorted(replace_names.keys()) == sorted(list(categories_names[0]))
            new_by_filename = {}
            for document_id, category_to_content in by_filename.items():
                print(f"Category to content: {(category_to_content.keys())}", file=sys.stderr)
                new_by_filename[document_id] = {replace_names[k]: v for k, v in category_to_content.items()}
        new_by_filename = by_filename
        for document_id, category_to_content in new_by_filename.items():
            assert len(category_to_content) == 2
            yield (category_to_content[list(category_to_content.keys())[0]], category_to_content[list(category_to_content.keys())[1]])

    p = { "pagexml_files": set() }
    args, _ = fargv.fargv(p)
    res = []
    line_counter = 0
    all1 = []
    all2 = []
    for alligned1, aligned2 in generate_pairs(args.pagexml_files):
        lines1 = alligned1.split('\n')
        lines2 = aligned2.split('\n')
        assert len(lines1) == len(lines2)
        for l1, l2 in zip(lines1, lines2):
            all1.append(l1)
            all2.append(l2)
            res.append(f"{line_counter}\t{l1}\t{l2}")
            line_counter += 1
    print('\n'.join(res))
    alphabet1 = sorted(set(''.join(all1)))
    alphabet2 = sorted(set(''.join(all2)))
    print(f"# Abreviated Alphabet 1 ({len(alphabet1)}): {''.join(alphabet1)}")
    print(f"# Unabreviated Alphabet 2 ({len(alphabet2)}): {''.join(alphabet2)}")
