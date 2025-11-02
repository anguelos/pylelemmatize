from collections import defaultdict
import random
import sys
from typing import Dict, Optional, Union, Tuple, List
import re
import unicodedata
import numpy as np
from math import inf

import torch
import tqdm

from pylelemmatize.fast_mapper import LemmatizerBMP
from .abstract_mapper import AbstractLemmatizer, fast_str_to_numpy
from lxml import etree
from .mapper_ds import MapperDs

def pagexml_to_text(pagexml_path: str) -> str:
    """
    Converts a PAGE XML string to plain text.

    Parameters:
    pagexml (str): The PAGE XML content as a string.

    Returns:
    str: The extracted plain text.
    """
    pagexml = open(pagexml_path, "r").read()
    xml_bytes = pagexml.encode("utf-8")
    root = etree.fromstring(xml_bytes)
    texts = []
    for unicode_text in root.xpath(".//*[local-name()='Unicode']"):
        texts.append(unicode_text.text or "")
    return "\n".join(texts)


def get_textlines(filepath: str, assume_txt=False, strip_empty_lines=True) -> List[str]:
    if filepath.lower().endswith(".xml") or filepath.lower().endswith(".pagexml"):
        res = pagexml_to_text(filepath).split("\n")
    elif filepath.lower().endswith(".txt") or assume_txt:
        res = open(filepath, "r").read().split("\n")
    else:
        raise f"Can't open {filepath}"
    if strip_empty_lines:
        res = [line for line in res if len(line)]
    res = [unicodedata.normalize("NFC", s) for s in res]
    return res


def load_textline_pairs(filelist1: List[str], filelist2: List[str]) -> List[Tuple[str, str]]:
    assert len(filelist1) == len(filelist2)
    res = []
    for file1, file2 in zip(sorted(filelist1), sorted(filelist2)):
        lines1 = get_textlines(file1)
        lines2 = get_textlines(file2)
        if len(lines1) == len(lines2):
            for line1, line2 in zip(lines1, lines2):
                res.append((line1, line2))
        else:
            print(f"Unaligned {file1} {file2} with {len(lines1)} vs {len(lines2)} lines", file=sys.stderr)
    return res


def banded_edit_path(a: np.ndarray, b: np.ndarray, band: int) -> np.ndarray:
    """
    Banded dynamic-programming alignment (edit distance with unit costs).
    
    Args:
        a: numpy array of single-character strings, shape (m,)
        b: numpy array of single-character strings, shape (n,)
        band: non-negative int, maximum |i - j| misalignment allowed
    
    Returns:
        path: A numpy array of (i, j) coordinates from (0,0) to (m,n) inclusive,
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
    return np.array(path)


def unaligned_to_seq2seq2(src: str, dst: str, band=50) -> np.ndarray:
    #src='pħo đ pagano. tͥ Dopno Romano monacho sce̾ Marie đ cͥpta ⁊ ꝓposito monast'
    #dst='Philippo de Pagano tibi Dopno Romano monacho Sancte Marie de Cripta et proposito monast'

    path = banded_edit_path(fast_str_to_numpy(src), fast_str_to_numpy(dst), band=band)
    src+="@$"
    dst+="@$"
    src_pos = 0
    dst_pos = 0
    pairs = []
    #print(f"Path: {path.T}")
    for src_path, dst_path in path:
        #src_path = int(path[src_pos, 0] if src_pos < len(src) else -1)
        #dst_path = int(path[dst_pos, 1] if dst_pos < len(dst) else -1)
        src_move = src_path != (path[src_pos + 1, 0] if src_pos + 1 < len(path) else -1)
        dst_move = dst_path != (path[dst_pos + 1, 1] if dst_pos + 1 < len(path) else -1)
        if src_move and dst_move:
            pairs.append((src[src_path], dst[dst_path]))
            src_pos += 1
            dst_pos += 1
        elif src_move and not dst_move:
            pairs.append((src[src_path], ""))
            src_pos += 1
            dst_pos += 1
        elif not src_move and dst_move:
            if(len(pairs) == 0):
                pairs.append((src[0], dst[0]))
            else:
                pairs[-1] = (pairs[-1][0], pairs[-1][1] + dst[dst_path])
            dst_pos += 1
            src_pos += 1
        else:
            raise RuntimeError("Stuck")
    return pairs[:-1]



class ManyToMoreDs:
    def __init__(self, line_pairs: List[Tuple[str, str]], aligned_line_segment_pairs: Optional[List[List[Tuple[str, str]]]], max_target_lengths: Optional[List[int]], band: int=70):
        self.line_pairs = line_pairs
        self.band = band
        self.max_target_length = max(len(out) for _, out in line_pairs)
        input_alphabet = sorted(set(''.join(inp for inp, _ in line_pairs)))
        output_alphabet = sorted(set(''.join(out for _, out in line_pairs)))
        self.input_mapper = LemmatizerBMP.from_alphabet_mapping(input_alphabet)
        self.output_mapper = LemmatizerBMP.from_alphabet_mapping(output_alphabet)
        self.aligned_line_segment_pairs = []
        self.max_target_lengths = []
        if aligned_line_segment_pairs is None or max_target_lengths is None:
            aligned_line_segment_pairs = []
            max_target_lengths = []
            for inp, out in tqdm.tqdm(line_pairs):
                aligned_line_segment_pairs.append(unaligned_to_seq2seq2(inp, out, band))
                max_target_lengths.append(max(len(t) for _, t in aligned_line_segment_pairs[-1]))
        self.aligned_line_segment_pairs = aligned_line_segment_pairs
        self.max_target_lengths = max_target_lengths

    def __init__(self, line_pairs: List[Tuple[str, str]], band=70, onehot: bool=False):
        self.line_pairs = line_pairs
        self.band = band
        self.max_target_length = max(len(out) for _, out in line_pairs)
        input_alphabet = sorted(set(''.join(inp for inp, _ in line_pairs)))
        output_alphabet = sorted(set(''.join(out for _, out in line_pairs)))
        self.input_mapper = LemmatizerBMP.from_alphabet_mapping(input_alphabet)
        self.output_mapper = LemmatizerBMP.from_alphabet_mapping(output_alphabet)
        self.aligned_line_segment_pairs = []
        self.max_target_lengths = []
        self.onehot = onehot
        for inp, out in tqdm.tqdm(line_pairs):
            self.aligned_line_segment_pairs.append(unaligned_to_seq2seq2(inp, out, band))
            self.max_target_lengths.append(max(len(t) for _, t in self.aligned_line_segment_pairs[-1]))

    def __len__(self):
        return len(self.line_pairs)

    def getitem_onehot(self, idx: int) -> Tuple[str, str]:
        max_target_length = self.max_target_lengths[idx]
        inp, out = self.line_pairs[idx]
        if self.onehot:
            inp = MapperDs.onehot_encode(inp, len(self.input_mapper)+1)
            out = MapperDs.onehot_encode(out, len(self.input_mapper)+1)
        return inp, out
    

    def partition(self, frac: float) -> Tuple['ManyToMoreDs', 'ManyToMoreDs']:
        assert 0.0 < frac < 1.0
        cut = int(len(self) * frac)
        ds1 = ManyToMoreDs(self.line_pairs[:cut], self.aligned_line_segment_pairs[:cut], self.max_target_lengths[:cut], self.band)
        ds2 = ManyToMoreDs(self.line_pairs[cut:], self.aligned_line_segment_pairs[cut:], self.max_target_lengths[cut:], self.band)
        return ds1, ds2

    def shuffled(self, seed: Optional[int]=None) -> 'ManyToMoreDs':
        combined = list(zip(self.line_pairs, self.aligned_line_segment_pairs, self.max_target_lengths))
        if seed is not None:
            random.seed(seed)
        random.shuffle(combined)
        line_pairs, aligned_line_segment_pairs, max_target_lengths = zip(*combined)
        return ManyToMoreDs(line_pairs, aligned_line_segment_pairs, max_target_lengths, self.band)

    def __str__(self):
        res = f"ManyToMoreDs with {len(self)} items\n"
        res += f"  band={self.band}\n"
        res += f"  max_target_length={self.max_target_length}\n"
        res += f"  input_alphabet ({len(self.input_alphabet)}): {''.join(self.input_alphabet)}\n"
        res += f"  output_alphabet ({len(self.output_alphabet)}): {''.join(self.output_alphabet)}\n"
        return res
    
    def save(self, path):
        with open(path, 'wb') as f:
            torch.save([self.line_pairs, self.aligned_line_segment_pairs, self.max_target_lengths, self.band], f)

    @staticmethod
    def load(path):
        if Path(path).is_file() is False:
            return None
        else:
            with open(path, 'rb') as f:
                line_pairs, aligned_line_segment_pairs, max_target_lengths, band = torch.load(f)
            return ManyToMoreDs(line_pairs, aligned_line_segment_pairs, max_target_lengths, band)


def many_to_more_main():
    import fargv, glob
    p = {
        "inputs": set(glob.glob("/home/anguelos/data/corpora/maria_pia/abreviated/B*.xml")),
        "outputs": set(glob.glob("/home/anguelos/data/corpora/maria_pia/unabreviated/B*.xml")),
        "dataset_path": "many_to_more_ds.pt",
        "band": 70,
    }
    args, _ = fargv.fargv(p)
    dataset = ManyToMoreDs.load(args.dataset_path)
    if dataset is None:
        line_pairs = load_textline_pairs(sorted(args.inputs), sorted(args.outputs))
    dataset = ManyToMoreDs(line_pairs, band=args.band)
