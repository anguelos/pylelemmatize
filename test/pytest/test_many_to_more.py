from typing import List, Tuple
import pytest
import torch
from pylelemmatize.many_to_more import ManyToMoreDS, align_sub_strings, banded_edit_path, collate_many_to_more_seq2seq
import sys
import numpy as np


# Genomic alphabets are prefered for testing as they have small size and are well known
# Normally everything should extend to any alphabet. TODO (anguelos): add tests for larger alphabets



debug_banded_edit_path = False  # enable this and run this tescase with pytest -s -x to debug output
skip_banded_edit_path = False  # enable this to skip banded_edit_path tests

@pytest.mark.parametrize(
    "a, b, ins_count, del_count, sub_count",
    [
        (np.array([11, 12, 13, 14, 15]), np.array([11, 12, 13, 14, 15]), 0, 0, 0), #  match

        (np.array([11, 12,13, 14, 15]), np.array([11, 12, 13, 14]), 0, 1, 0), # one deletion at end
        (np.array([11, 12, 13, 14, 15]), np.array([12, 13, 14, 15]), 0, 1, 0), # one deletion at beginning
        (np.array([11, 12, 13, 14, 15]), np.array([11, 12, 14, 15]), 0, 1, 0), # one deletion in midle
        (np.array([11, 12,13, 14, 15]), np.array([11, 12, 13]), 0, 2, 0), # two deletions at end
        (np.array([11, 12, 13, 14, 15]), np.array([13, 14, 15]), 0, 2, 0), # two deletion at beginning

        (np.array([11, 12, 13, 14, 15]), np.array([11, 12, 13, 14, 10]), 0, 0, 1), # one substitution at end
        (np.array([11, 12, 13, 14, 15]), np.array([10, 12, 13, 14, 15]), 0, 0, 1), # one substitution at beginning
        (np.array([11, 12, 13, 14, 15]), np.array([11, 12, 10, 14, 15]), 0, 0, 1), # one substitution in midle
        (np.array([11, 12, 13, 14, 15]), np.array([11, 10, 10, 10, 15]), 0, 0, 3), # three substitutions in midle
        (np.array([11, 12, 13, 14, 12, 13, 14, 15]), np.array([11, 12, 13, 14, 10, 10, 10, 15]), 0, 0, 3), # three substitutions in midle
        (np.array([11, 12, 13, 14, 12, 13, 14, 15]), np.array([11, 10, 10, 10, 12, 13, 14, 15]), 0, 0, 3), # three substitutions in midle
        (np.array([11, 12, 13, 14, 12, 13, 14, 15]), np.array([11, 12, 10, 10, 10, 13, 14, 15]), 0, 0, 3), # three substitutions in midle

        (np.array([11, 12, 13, 14, 15]), np.array([11, 12, 13, 14, 15, 0]), 1, 0, 0), # one insertion at end
        (np.array([11, 12, 13, 14, 15]), np.array([10, 11, 12, 13, 14, 15]), 1, 0, 0), # one insertion at beginning
        (np.array([11, 12, 13, 14, 15]), np.array([11, 12, 13, 10, 14, 15]), 1, 0, 0), # one insertion in midle
        (np.array([11, 12, 13, 14, 15]), np.array([11, 12, 13, 14, 15, 0, 0]), 2, 0, 0), # two insertions at end
        (np.array([11, 12, 13, 14, 15]), np.array([0, 0, 11, 12, 13, 14, 15]), 2, 0, 0), # two insertions at beginning

        (np.array([10, 10, 10, 11, 12, 13, 14, 15]), np.array([11, 12, 13, 14, 15, 10, 10, 10]), 3, 3, 0), # shifting 3 left
        (np.array([11, 12, 13, 14, 15, 10, 10, 10]), np.array([10, 10, 10, 11, 12, 13, 14, 15]), 3, 3, 0), # shifting 3 right
    ]
)
def test_banded_edit_path(a: np.ndarray, b: np.ndarray, ins_count: int, del_count: int, sub_count: int):
    if skip_banded_edit_path:
        return
    cost_count = ins_count + del_count + sub_count
    for band in [1, 3, 10]:
        if band > ins_count + del_count + sub_count:
            path, cost, (is_match, is_ins, is_del, is_sub) = banded_edit_path(a, b, 10)
            if debug_banded_edit_path:
                print(f"\nBand {band} Ins: {ins_count} Del: {del_count} Sub: {sub_count}")
                print('A: ', a.tolist())
                print('B: ', b.tolist())
                print(np.array(path).T)
                print(cost)
                print('Match: ', is_match.tolist())
                print('Ins  : ', is_ins.tolist())
                print('Del  :', is_del.tolist())
                print('Sub  :', is_sub.tolist())            
            assert path[0, 0] == -1 and path[0, 1] == -1
            assert path[-1, 0] + 1 == len(a) and path[-1, 1] + 1 == len(b)
            np.testing.assert_equal(path[1:, :] - path[:-1, :] <= 1, True)
            np.testing.assert_equal(is_ins + is_del + is_sub, cost)
            assert cost.sum() == cost_count
            assert ins_count == is_ins.sum()
            assert del_count == is_del.sum()
            assert sub_count == is_sub.sum()


debug_align_sub_strings = False  # enable this and run this tescase with pytest -s -x to debug output
skip_align_sub_strings = False  # enable this to skip banded_edit_path tests

@pytest.mark.parametrize(
    "a, b, tgt_aligned_substrings",
    [
        ("ACGT", "ACGT", [("A", "A"), ("C", "C"), ("G", "G"), ("T", "T")]),  # Match

        ("ACGT", "ZACGT", [("A", "ZA"), ("C", "C"), ("G", "G"), ("T", "T")]), # Insertion left
        ("ACGT", "ACGTZ", [("A", "A"), ("C", "C"), ("G", "G"), ("T", "TZ")]),  # Insertion right
        ("ACGT", "ACZGT", [("A", "A"), ("C", "CZ"), ("G", "G"), ("T", "T")]),  # Insertion midle
        ("ACGT", "ZZACGT", [("A", "ZZA"), ("C", "C"), ("G", "G"), ("T", "T")]), # Two Insertions left
        ("ACGT", "ACGTZZ", [("A", "A"), ("C", "C"), ("G", "G"), ("T", "TZZ")]),  # Two Insertions right

        ("ACGT", "CGT", [("A", ""), ("C", "C"), ("G", "G"), ("T", "T")]), # Deletion left
        ("ACGT", "ACG", [("A", "A"), ("C", "C"), ("G", "G"), ("T", "")]),  # Deletion right
        ("ACGT", "AGT", [("A", "A"), ("C", ""), ("G", "G"), ("T", "T")]),  # Deletion middle
        ("ACGT", "AC", [("A", "A"), ("C", "C"), ("G", ""), ("T", "")]),  # Two deletions left
        ("ACGT", "GT", [("A", ""), ("C", ""), ("G", "G"), ("T", "T")]),  # Two deletions left
        ("ACGT", "AT", [("A", "A"), ("C", ""), ("G", ""), ("T", "T")]),  # Two deletions middle

        ("AAACGT", "ACGTTT", [("A", ""), ("A", ""), ("A", "A"), ("C", "C"), ("G", "G"), ("T", "TTT")]),  # Shifting rigt by 2
    ]
)
def test_align_sub_strings(a, b, tgt_aligned_substrings):
    if skip_align_sub_strings:
        return
    aligned_substrings = align_sub_strings(a, b)
    if debug_align_sub_strings:
        print(f"\n\n\na: {a}\nb: {b}\nALIGNED:", aligned_substrings,"\nEXPECTED:", tgt_aligned_substrings)
    assert all([len(a) == 1 for a, b in aligned_substrings ]) # Each input must be one symbol
    reconstructed_a, reconstructed_b = zip(*aligned_substrings)
    assert ''.join(reconstructed_a) == a
    assert ''.join(reconstructed_b) == b
    
    tgt_aligned_substrings_no_deletions = [pair for pair in tgt_aligned_substrings if pair[1] != ""]
    aligned_substrings_no_deletions = [pair for pair in aligned_substrings if pair[1] != ""]
    if debug_align_sub_strings:
        print(f"ALIGNED NO DEL:", aligned_substrings_no_deletions,"\nEXPECTED NO DEL:", tgt_aligned_substrings_no_deletions)
    assert aligned_substrings_no_deletions == tgt_aligned_substrings_no_deletions

    # When a symbol is inserted There is ambiguity 
    # "AAACGT", "ACGTTT" -> [('A', ''), ('A', ''), ('A', 'A'), ('C', 'C'), ('G', 'G'), ('T', 'TTT')]
    # could also be
    # "AAACGT", "ACGTTT" -> [('A', ''), ('A', 'A'), ('A', ''), ('C', 'C'), ('G', 'G'), ('T', 'TTT')]
    # So we do not test for exact equality
    # TODO (anguelos): improve test to check for deletions not beeing 


@pytest.mark.parametrize("line_pairs, allow_start_insertions, band, sample_replication, expected_sample_count",
    [
        ([("ACGTACGT", "ACGTACGT"), ("AAGCT", "AGCT"), ("ACGT", "ZACGT"), ("ACGT", "ACGTZ"), ("ACGT", "ACZGT")], True, 10, 7, 5),
    ]
)
def test_create_from_aligned_textlines(line_pairs: List[Tuple[str, str]], allow_start_insertions, 
                                       band, sample_replication, expected_sample_count):
    line_pairs= [(inp* sample_replication, tgt* sample_replication) for inp, tgt in line_pairs]
    ds = ManyToMoreDS.create_from_aligned_textlines(line_pairs, allow_start_insertions=allow_start_insertions, band=band)
    assert len(ds) == expected_sample_count


@pytest.mark.parametrize("textlines, max_unalignement, batch_inputs, batch_outputs", [
        [[("ACGT", "ACGT"), ("ACGT", "AGT"), ("ACGT", "ACGAGAGT"), ], -1, # -1 means NULL terminated
            [[1, 2, 3, 4]] * 3,
            [[[1, 0], [2, 0], [3, 0], [4, 0]], 
             [[1, 0], [0, 0], [3, 0], [4, 0]],
             [[1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [3, 1, 3, 1, 3, 0], [4, 0, 0, 0, 0, 0]]],],

        [[("ACGT", "ACGT"), ("ACGT", "AGT"), ("ACGT", "ACGAGAGT"), ], 7,
            [[1, 2, 3, 4]] * 3,
            [[[1] + [0] * 7, [2] + [0] * 7, [3] + [0] * 7, [4] + [0] * 7], 
            [[1] + [0] * 7, [0] * 8, [3] + [0] * 7, [4] + [0] * 7],
            [[1] + [0] * 7, [2] + [0] * 7, [3, 1, 3, 1, 3, 0, 0, 0], [4] + [0] * 7]]],
])
def test_collate_many_to_more_seq2seq(textlines, max_unalignement, batch_inputs, batch_outputs):
    ds = ManyToMoreDS.create_from_aligned_textlines(line_pairs=textlines, min_src_len=3, min_tgt_len=3)
    assert len(ds) == len(textlines)  # Sanity check: All input output pairs became part of the dataset
    collator = lambda batch: collate_many_to_more_seq2seq(batch, max_unalignemet=max_unalignement)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)
    for i, batch in enumerate(dl):
        srcs_tensor, tgts_tensor = batch
        expected_srcs = torch.tensor(batch_inputs[i], dtype=torch.long).unsqueeze(0)
        expected_tgts = torch.tensor(batch_outputs[i], dtype=torch.long)
        assert torch.equal(srcs_tensor, expected_srcs)
        assert torch.equal(tgts_tensor, expected_tgts)