import pytest
import pylelemmatize
from pylelemmatize.substitution_augmenter import CharConfusionMatrix
from pylelemmatize.fast_mapper import LemmatizerBMP
import numpy as np


def test_confusion_matrix_initialization():
    alphabet_str = "ACGT"
    lemmatizer = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    cm = CharConfusionMatrix(lemmatizer)

    matrix = cm.get_matrix()
    assert matrix.shape == (len(alphabet_str) + 1, len(alphabet_str) + 1)
    assert (matrix == 0).all()  # Initially all zeros


def test_substitution_count():
    alphabet_str = "ACGT"
    lemmatizer = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    
    chars = "ACGTACGACA"
    cm = CharConfusionMatrix(lemmatizer)
    cm.ingest_textline_observation(chars, chars)
    matrix = cm.get_matrix()
    assert sum(matrix.reshape(-1)) == len(chars)  # All correct predictions
    assert matrix[0,0] == 0  # No nulls
    assert matrix[1,1] == 4  # 'A' correct
    assert matrix[2,2] == 3  # 'C' correct
    assert matrix[3,3] == 2  # 'G' correct
    assert matrix[4,4] == 1  # 'T' correct

    cm.ingest_textline_observation(chars, chars)
    matrix = cm.get_matrix()
    assert sum(matrix.reshape(-1)) == 2 * len(chars)  # All correct predictions
    assert matrix[0,0] == 0  # No nulls
    assert matrix[1,1] == 8  # 'A' correct
    assert matrix[2,2] == 6  # 'C' correct
    assert matrix[3,3] == 4  # 'G' correct
    assert matrix[4,4] == 2  # 'T' correct


def test_insertions_deletions():
    alphabet_str = "ACGT"
    lemmatizer = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    long = "AACAGATA"
    short = "ACGT"
    insertion_cm = CharConfusionMatrix(lemmatizer)
    insertion_cm.ingest_textline_observation(long, short)
    matrix = insertion_cm.get_matrix()
    assert sum(matrix.reshape(-1)) == len(long)  # All insertions
    assert matrix[0,0] == 0  # No nulls
    assert matrix[1,0] == 4  # 'A' inserted four times
    assert matrix[1,1] == 1  
    assert matrix[2,2] == 1  
    assert matrix[3,3] == 1  
    assert matrix[4,4] == 1
    deletion_cm = CharConfusionMatrix(lemmatizer)
    deletion_cm.ingest_textline_observation(short, long)
    assert np.allclose(deletion_cm.get_matrix(), insertion_cm.get_matrix().T)

    long = "ACGTAAAA"
    short = "ACGT"
    insertion_cm = CharConfusionMatrix(lemmatizer)
    insertion_cm.ingest_textline_observation(long, short)
    matrix = insertion_cm.get_matrix()
    assert sum(matrix.reshape(-1)) == len(long)  # All insertions
    assert matrix[0,0] == 0  # No nulls
    assert matrix[1,0] == 4  # 'A' inserted four times
    assert matrix[1,1] == 1  
    assert matrix[2,2] == 1  
    assert matrix[3,3] == 1  
    assert matrix[4,4] == 1
    assert np.allclose(deletion_cm.get_matrix().T, insertion_cm.get_matrix())

    long = "AAAAACGT"
    short = "ACGT"
    insertion_cm = CharConfusionMatrix(lemmatizer)
    insertion_cm.ingest_textline_observation(long, short)
    matrix = insertion_cm.get_matrix()
    assert sum(matrix.reshape(-1)) == len(long)  # All insertions
    assert matrix[0,0] == 0  # No nulls
    assert matrix[1,0] == 4  # 'A' inserted four times
    assert matrix[1,1] == 1  
    assert matrix[2,2] == 1  
    assert matrix[3,3] == 1  
    assert matrix[4,4] == 1
    assert np.allclose(deletion_cm.get_matrix().T, insertion_cm.get_matrix())


@pytest.mark.parametrize("s1,s2,expected_ed", [
    ("ACGT", "ACGT", 0),
    ("ACGTACGT", "AACGTACGT", 1),
    ("AGGTACGT", "AACGTACGT", 2),
    ("AGGTACGTT", "AACGTACGT", 3),
    ("A"+"ACCT"*7, "ACGT"*7, 8),
    ("ACCT"*7, "ACGT"*7+"G", 8),
    ("G"+"ACCT"*7, "ACGT"*7+"G", 9),
    ("G"*2+"AC"*5, "AC"*5+"T"*2, 4),
    ("AC"*1+"G"*2, "T"*2+"AC"*1, 4)
    ])
def test_ed(s1, s2, expected_ed):
    alphabet_str = "ACGT"
    lemmatizer = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    s1 = lemmatizer.str_to_intlabel_seq(s1)
    s2 = lemmatizer.str_to_intlabel_seq(s2)
    cm = CharConfusionMatrix(alphabet=alphabet_str)
    distance, dm = cm.edit_distance(s1, s2)
    assert distance == expected_ed
    assert dm.min() == 0
    assert dm.max() <= max(len(s1), len(s2))


@pytest.mark.parametrize("pred,gt,tst_subonly,expected_ed", [
    ("ACGT", "ACGT", "ACGT", 0), # PERFECT MATCH
    ("ACGTACGT", "AACGTACGT", "AACGTACGT", 1), # SIMPLE DELETION
    ("AACGTACGT", "ACGTACGT", "ACGTACGT", 1), # SIMPLE INSERTION
    ("ACACACAC", "ACACAGAG", "ACACACAC", 2), # TWO SUBSTITUTIONS
    ("ACACACAC", "TTTACACAGAG", "TTTACACACAC", 5), # THREE BEGIN DELETIONS + TWO SUBSTITUTIONS
    ("ACACACACATTT", "ACACAGAGA", "ACACACACA", 5), # TWO SUBSTITUTIONS + THREE INSERTIONS
    ("TTTACACACAC", "ACACAGAG", "ACACACAC", 5), # THREE BEGIN INSERTIONS + TWO SUBSTITUTIONS
    ("TTTTAAAAAGAAAAA", "AAAAACAAAAATTTT", "AAAAAGAAAAATTTT", 9), # FOUR BEGIN INSERTIONS + 1 SUBSTITUTIONS + FOUR END DELETIONS,
    ("AAAAAGAAAAATTTT", "TTTTAAAAACAAAAA", "TTTTAAAAAGAAAAA", 9) # FOUR BEGIN DELETIONS + 1 SUBSTITUTIONS + FOUR END INSERTIONS,
])
def test_backtrace_ed_matrix(pred, gt, tst_subonly, expected_ed):
    alphabet_str = "ACGT"
    lemmatizer = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    cm = CharConfusionMatrix(lemmatizer)
    pred = lemmatizer.str_to_intlabel_seq(pred)
    gt = lemmatizer.str_to_intlabel_seq(gt)
    edit_distance, dp = cm.edit_distance(pred, gt)
    path, op_type, np_subonly, _ = cm.backtrace_ed_matrix(pred, gt, dp)
    subonly = lemmatizer.intlabel_seq_to_str(np_subonly)
    assert edit_distance == expected_ed
    assert subonly == tst_subonly


@pytest.mark.parametrize("pred,gt,tst_subonly,tst_cm", [
    ("ACGT", "ACGT", "ACGT", [[0,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]),
    ("ACGTACGT", "AACGTACGT", "AACGTACGT", [[0,1,0,0,0],[0,2,0,0,0],[0,0,2,0,0],[0,0,0,2,0],[0,0,0,0,2]]),
    ("AACGTACGT", "ACGTACGT", "ACGTACGT", [[0,0,0,0,0],[1,2,0,0,0],[0,0,2,0,0],[0,0,0,2,0],[0,0,0,0,2]]),
    ("TTTACACACAC", "ACACAGAG", "ACACACAC", [[0,0,0,0,0],[0,4,0,0,0],[0,0,2,2,0],[0,0,0,0,0],[3,0,0,0,0]]),
    ("TTTTAAAAAGAAAAA", "AAAAACAAAAATTTT", "AAAAAGAAAAATTTT", [[0,0,0,0,4],[0,10,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[4,0,0,0,0]]),
])
def test_ingest_textline_observation(pred, gt, tst_subonly, tst_cm):
    alphabet_str = "ACGT"
    lemmatizer = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    cm = CharConfusionMatrix(lemmatizer)
    sub_only, _ = cm.ingest_textline_observation(pred, gt)
    assert sub_only == tst_subonly
    assert np.allclose(cm.get_matrix(), tst_cm)


def test_mutation_probability():
    alphabet_str = "ACGT"
    lemmatizer = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    cm = CharConfusionMatrix(lemmatizer)
    cm.cm[:,:] = [[0,  0,    0,   0,  0],
                  [0, 150,  42,  31, 50],
                  [0,  10, 200,  62, 5],
                  [0,   3,  11, 250, 10],
                  [0,   1,  21,   1, 300]]
    input_samples = 1 + np.arange(100000, dtype=np.int32) % 4
    mutated = cm.generate_random_substitution_sequences(input_samples)
    vals, counts = np.unique(mutated, return_counts=True)  # Just to ensure it runs
    cm_freq = cm.cm[1:, 1:].sum(axis=0)
    cm_freq = cm_freq / cm_freq.sum()
    output_freq = counts / counts.sum()

    assert vals.tolist() == [1, 2, 3, 4]
    assert np.allclose(output_freq, cm_freq, atol=0.1)


def test_self_supervision_textline():
    alphabet_str = "ACGT"
    lemmatizer = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    cm = CharConfusionMatrix(lemmatizer)
    cm.ingest_textline_observation("ACGT"* 10, "AGCT" * 10)
    cm.ingest_textline_observation("A"* 30, "GGGCCTTTTT" * 3)
    cm.ingest_textline_observation("C"* 30, "CCCCCCCCCG" * 3)
    cm.ingest_textline_observation("G"* 30, "CCCCCCCCCG" * 3)
    cm.ingest_textline_observation("T"* 30, "AAAAAAGGGG" * 3)
    input_line = "ACGT" * 2500
    mutated = cm.get_self_supervision_textline(input_line)
    vals, counts = np.unique(np.array(list(mutated)), return_counts=True)  # Just to ensure it runs
    cm_freq = cm.cm[1:, 1:].sum(axis=0)
    cm_freq = cm_freq / cm_freq.sum()
    output_freq = counts / counts.sum()
    assert np.allclose(output_freq, cm_freq, atol=0.01)
    assert False