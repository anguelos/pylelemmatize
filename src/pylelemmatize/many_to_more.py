from collections import defaultdict
import random
import sys
from typing import Dict, Optional, Union, Tuple, List
import re
import unicodedata
from pathlib import Path
import numpy as np
from math import inf

import torch
import tqdm

from pylelemmatize.fast_mapper import LemmatizerBMP, fast_numpy_to_str, fast_str_to_numpy
from .abstract_mapper import AbstractLemmatizer, fast_str_to_numpy
from lxml import etree


### In all contexts where we talk about insertions and deletions occuring in two strings A and B:
# - Insertion means inserting a character into A to obtain B (a gap in A) We insert in B to match A
# - Deletion means deleting a character from A to obtain B (a gap in B) We delete in B to match A
# The code should be read with this convention in mind. And variable namings like is_ins, is_del, etc. 
# MUST reflect this convention.
# This also applies to unitests about these fille


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
    #path = [[-1, -1]] + list(path)
    paths = np.array(path) - 1
    is_ins = paths[1:, 0] - paths[: -1, 0] == 0
    is_del = paths[1:, 1] - paths[: -1, 1] == 0
    is_diag = ~(is_del | is_ins)
    agree =  (a[paths[1:, 0]] == b[paths[1:, 1]])
    #agree =  (a[paths[:-1, 0] - 1] == b[paths[:-1, 1] -1])
    is_sub = is_diag & ~agree
    is_match = is_diag & agree
    #print(f"\nA:{a.tolist()}\nB:{b.tolist()}\nPATHS:\n", paths.T, "\nis_del: ", is_del, "\nis_ins: ", is_ins, "\nis_sub: ", is_sub, "\nis_mat: ", is_match)
    #is_sub = (a[paths[1:, 0]] != b[paths[1:, 1]])
    cost_1 = is_sub | is_del | is_ins
    return paths, cost_1.astype(np.int16), (is_match, is_ins, is_del, is_sub)


def align_sub_strings(a: str, b: str, band:int = 80) -> List[Tuple[str, str]]:
    np_a = fast_str_to_numpy(a)
    np_b = fast_str_to_numpy(b)
    paths, _, (is_match, is_ins, is_del, is_sub) = banded_edit_path(np_a, np_b, band=band)
    p = paths[1:, :].tolist()
    res = []
    n = 0
    while n < len(p):
        if is_ins[n]:
            start_a, start_b = p[n]
            start_a, start_b = max(start_a, 0), max(start_b, 0)
            while n < len(p) and is_ins[n]:
                n += 1
            if n == len(p):
                end_a, end_b = len(a), len(b)
            else:
                end_a, end_b = p[n]
                end_a, end_b = max(end_a, 0), max(end_b, 0)
            res.append((a[start_a: start_a], b[start_b: end_b]))
        elif is_match[n] or is_sub[n]:
            a_pos, b_pos = p[n] 
            res.append((a[a_pos: a_pos+1], b[b_pos: b_pos+1]))
            n+=1
        elif is_del[n]:
            a_pos, b_pos = p[n]
            res.append((a[a_pos: a_pos+1], b[b_pos: b_pos]))
            n+=1
        else:
            raise RuntimeError("Unknown cell type")

    merged_insertions = []
    prev_insertion = ""
    for a_substr, b_substr in res:
        if prev_insertion == "" and a_substr != "": # match / substitution / deletion no pending insertion
            merged_insertions.append((a_substr, b_substr))
        elif a_substr == "": # insertion
            prev_insertion += b_substr
        elif prev_insertion != "" and len(merged_insertions) == 0 and a_substr != "": # Starting with insertion(s) then match/sub
            merged_insertions.append((a_substr, prev_insertion + b_substr))
            prev_insertion = ""
        elif prev_insertion != "" and len(merged_insertions) > 0 and a_substr != "": # pending insertion then match/sub
            merged_insertions[-1] = (merged_insertions[-1][0], merged_insertions[-1][1] + prev_insertion)
            merged_insertions.append((a_substr, b_substr))
            prev_insertion = ""
        else:
            raise RuntimeError("Impossible state")
    if prev_insertion != "":
        merged_insertions[-1] = (merged_insertions[-1][0], merged_insertions[-1][1] + prev_insertion)
    return merged_insertions


class ManyToMoreDS:
    @staticmethod
    def create_from_aligned_textlines(line_pairs: List[Tuple[str, str]], min_src_len: int=10,
                          max_src_len: int=1000, min_tgt_len: int=10, max_tgt_len: int=1000, band: int = 70,
                          src_alphabet: Optional[str]=None, tgt_alphabet: Optional[str]=None,
                          onehot_input: bool = False, onehot_output: bool = False, allow_start_insertions: bool=False,
                          verbose: bool=False) -> List[Tuple[str, str]]:
        textlines_appropriate_sizes = []
        for inp, out in line_pairs:
            if not (min_src_len <= len(inp) <= max_src_len):
                continue
            if not (min_tgt_len <= len(out) <= max_tgt_len):
                continue
            textlines_appropriate_sizes.append((inp, out))
        inputs, outputs = zip(*textlines_appropriate_sizes)
        if src_alphabet is None:
            src_alphabet = set(''.join(inputs))
        if tgt_alphabet is None:
            tgt_alphabet = set(''.join(outputs))
        textlines_appropriate_alphabets = []
        aligned_substrings: List[List[Tuple[str, str]]] = []
        for inp, out in textlines_appropriate_sizes:
            if set(inp).issubset(src_alphabet) and set(out).issubset(tgt_alphabet):
                textlines_appropriate_alphabets.append((inp, out))
        if verbose:
            print(f"Read {len(line_pairs)} line pairs")
            print(f"After size filtering: {len(textlines_appropriate_sizes)} line pairs because of length constraints ({min_src_len}-{max_src_len} for source, {min_tgt_len}-{max_tgt_len} for target).", file=sys.stderr)
            print(f"After alphabet filtering: {len(textlines_appropriate_alphabets)} line pairs because of alphabet constraints.", file=sys.stderr)
            #print(f"Filtered {len(textlines_appropriate_sizes)} line pairs to {len(textlines_appropriate_alphabets)} line pairs because of length and alphabet constraints.", file=sys.stderr)
        if verbose:
            progress = tqdm.tqdm(total=len(textlines_appropriate_alphabets), desc="Processing line pairs")
        for inp, out in textlines_appropriate_alphabets:
            if verbose:
                progress.update(1)
            try:
                aligned_in_out = align_sub_strings(inp, out, band=band)
            except ValueError:
                if verbose:
                    print(f"Skipping pair due to alignment failure within band {band}: '{inp}' -> '{out}'", file=sys.stderr)
                continue
            if not allow_start_insertions and aligned_in_out[0][0][0]=="":
                continue
            aligned_substrings.append(aligned_in_out)
        if verbose:
            progress.close()
        src_alphabet = ''.join(sorted(src_alphabet))
        tgt_alphabet = ''.join(sorted(tgt_alphabet))
        res = ManyToMoreDS(aligned_substrings=aligned_substrings, src_alphabet=src_alphabet, tgt_alphabet=tgt_alphabet, 
                           onehot_input=onehot_input, onehot_output=onehot_output)
        if verbose:
            print(f"Created ManyToMoreDS with {len(res)} samples. aligned_substrings: {len(aligned_substrings)}", file=sys.stderr)
        return res
    
    def __init__(self, src_alphabet: str, tgt_alphabet: str,
                 aligned_substrings: List[List[Tuple[str, str]]],
                 onehot_input: bool=False, onehot_output: bool=False,
                 return_torch: bool=True, return_ctc: bool=False, max_unalignemet: int = 5):
        self.onehot_input = onehot_input
        self.onehot_output = onehot_output
        self.aligned_substrings = aligned_substrings
        self.src_lemmatizer = LemmatizerBMP(src_alphabet)
        self.tgt_lemmatizer = LemmatizerBMP(tgt_alphabet)
        self.return_torch = return_torch
        self.return_ctc = return_ctc
        self.max_unalignemet = max_unalignemet
        assert max_unalignemet < self.get_max_dst_len(), "max_unalignemet must be less than maximum target length"
    
    def get_max_dst_len(self) -> int:
        max_len = 0
        for aligned_substrings in self.aligned_substrings:
            tgt_len = sum([len(part[1]) for part in aligned_substrings])
            if tgt_len > max_len:
                max_len = tgt_len
        return max_len
    
    def __len__(self):
        return len(self.aligned_substrings)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        aligned_substrings = self.aligned_substrings[idx]
        src_parts, tgt_parts = zip(*aligned_substrings)

        # src is a list contianing single sequence of all input characters concatenated
        # tgt is a list containing [every target substring as a separate sequence]
        src: List[np.ndarray] = [self.src_lemmatizer.str_to_intlabel_seq(c) for c in src_parts]
        src = [np.concatenate(src, axis=0)]
        tgt: List[np.ndarray] = [self.tgt_lemmatizer.str_to_intlabel_seq(c) for c in tgt_parts]  

        if self.onehot_input:
            raise NotImplementedError("onehot_input is not implemented yet")

        if self.return_torch:            
            #print(f"input_lengths = {set([len(x) for x in src])}")
            #print(f"output_lengths = {set([len(x) for x in tgt])}")
            src = [torch.tensor(seq).long().reshape(-1) for seq in src]
            tgt = [torch.tensor(t).long() for t in tgt]
        else:
            src = np.array(src, dtype=np.int16).reshape(-1)
            tgt = [np.array(t, dtype=np.int16) for t in tgt]
        return src[0], tgt
    
    def save(self, dataset_path: str):
        data = {'aligned_substrings': self.aligned_substrings,
                'onehot_input': self.onehot_input,
                'onehot_output': self.onehot_output,
                'src_alphabet': self.src_lemmatizer.src_alphabet_str,
                'tgt_alphabet': self.tgt_lemmatizer.src_alphabet_str,
                'return_torch': self.return_torch,
                'return_ctc': self.return_ctc,
                }
        torch.save(data, dataset_path)
    
    @classmethod
    def load(cls, dataset_path: str, allow_start_insertions=True) -> Optional["ManyToMoreDS"]:
        try:
            data = torch.load(dataset_path)
            if allow_start_insertions:
                aligned_fragmenst = []
                for aligned_substrings in data['aligned_substrings']:
                    if aligned_substrings[0][0] == "":
                        continue
                    aligned_fragmenst.append(aligned_substrings)
                data['aligned_substrings'] = aligned_fragmenst
            dataset = ManyToMoreDS(
                src_alphabet=data['src_alphabet'],
                tgt_alphabet=data['tgt_alphabet'],
                aligned_substrings=data['aligned_substrings'],
                onehot_input=data['onehot_input'],
                onehot_output=data['onehot_output'],
                return_torch=data['return_torch'],
                return_ctc=data['return_ctc']
            )
            return dataset
        except FileNotFoundError:
            return None


def collate_many_to_more_seq2seq(batch: List[Tuple[Union[np.ndarray, torch.Tensor],
                                                   Union[np.ndarray, torch.Tensor]]],
                                                   max_unalignemet: int=-1) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(batch) == 1
    srcs, tgts = zip(*batch)
    assert len(srcs) == 1 and len(tgts) == 1    
    srcs, tgts = srcs[0], tgts[0]
    if max_unalignemet < 0:
        max_unalignemet = max([len(tgt) for tgt in tgts])
    tgts_tensor = torch.zeros((len(srcs), max_unalignemet + 1), dtype=torch.long)
    for i, tgt in enumerate(tgts):
        if len(tgt) > max_unalignemet:
            raise RuntimeError(f"Target length {len(tgt)} exceeds max_unalignemet {max_unalignemet}")
        tgts_tensor[i, :len(tgt)] = tgt
    return srcs.unsqueeze(0), tgts_tensor


def many_to_more_main():
    import fargv, glob
    p = {
        "inputs": set(glob.glob("/home/anguelos/data/corpora/maria_pia/abreviated/B*.xml")),
        "outputs": set(glob.glob("/home/anguelos/data/corpora/maria_pia/unabreviated/B*.xml")),
        "dataset_path": "./many_to_more_ds.pt",
        "allow_start_insertions": False,
        "band": 70,
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    if Path(args.dataset_path).is_file():
        print(f"Dataset {args.dataset_path} already exists. Loading existing dataset.", file=sys.stderr)
        dataset = ManyToMoreDS.load(args.dataset_path, allow_start_insertions=args.allow_start_insertions)
    else:
        print(f"Creating dataset at {args.dataset_path}", file=sys.stderr)
        line_pairs = load_textline_pairs(sorted(args.inputs), sorted(args.outputs))
        dataset = ManyToMoreDS.create_from_aligned_textlines(line_pairs=line_pairs, verbose=args.verbose, band=args.band, allow_start_insertions=args.allow_start_insertions)
        dataset.save(args.dataset_path)
    ds = ManyToMoreDS(src_alphabet="ACGT",
                         tgt_alphabet="ACGT",
                         aligned_substrings=[[("A", "A"), ("C", "C"), ("G", "G"), ("T", "T")],
                                             [("A", ""), ("C", "C"), ("G", "G"), ("T", "T")],
                                             [("A", "A"), ("C", "C"), ("G", "GGGG"), ("T", "T")]],
                                             max_unalignemet=4)
    #print(f"Dataset {args.dataset_path} loaded with {len(dataset)} samples.", file=sys.stderr)
    print(f"Dataset[0]: ", repr(ds[0]), file=sys.stderr)

    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                       collate_fn=lambda x: collate_many_to_more_seq2seq(x, max_unalignemet=ds.max_unalignemet))
    print("Dataloader[0]: ", file=sys.stderr)
    print(list(dataloader)[2], file=sys.stderr)