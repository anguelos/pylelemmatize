import pylelemmatize
from pylelemmatize.substitution_augmenter import CharConfusionMatrix
from pylelemmatize.fast_mapper import LemmatizerBMP
from pylelemmatize.abstract_mapper import GenericLemmatizer, fast_str_to_numpy, fast_numpy_to_str, fast_cer
import numpy as np
import pytest


def test_nonbmp_mapper():
    mapping_dict = {"A": "a", "C": "c", "G": "g", "T": "t", "🧬": "🧬"}
    nonbmp_mapper = GenericLemmatizer(mapping_dict=mapping_dict)
    test_sequence = "ACGT🧬"
    expected_output = "acgt🧬"
    assert nonbmp_mapper(test_sequence) == expected_output
    with pytest.raises(ValueError):
        LemmatizerBMP(mapping_dict=mapping_dict)


def test_bmp_mapper():
    mapping_dict = {"A": "a", "C": "c", "G": "g", "T": "t", "a": "a", "c": "c", "g": "g", "t": "t"}
    test_sequence = "ACGTacgt"
    expected_output = "acgtacgt"
    bmp_mapper = LemmatizerBMP(mapping_dict=mapping_dict)
    assert bmp_mapper(test_sequence) == expected_output
    nonbmp_mapper = GenericLemmatizer(mapping_dict=mapping_dict)
    assert nonbmp_mapper(test_sequence) == expected_output

    # testing unknown characters
    test_sequence = "ACGTacgtZ"
    expected_output = "acgtacgt0"
    bmp_mapper = LemmatizerBMP(mapping_dict=mapping_dict, unknown_chr="0")
    assert bmp_mapper(test_sequence) == expected_output
    nonbmp_mapper = GenericLemmatizer(mapping_dict=mapping_dict, unknown_chr="0")
    assert nonbmp_mapper(test_sequence) == expected_output

@pytest.mark.parametrize("test_sequence", [
    "ACGTacgt",
])
def test_fast_str_to_array(test_sequence):
    sparse_test_array = np.array([ord(c) for c in test_sequence], dtype=np.uint32)
    np.testing.assert_array_equal(
        fast_str_to_numpy(test_sequence), sparse_test_array
    )
    np.testing.assert_array_equal(
        fast_numpy_to_str(sparse_test_array), test_sequence
    )
    assert fast_numpy_to_str(fast_str_to_numpy(test_sequence)) == test_sequence

@pytest.mark.parametrize("str1, str2, expected_cer", [
    ("ACGT", "ACGT", 0.0),
    ("ACGTT", "ACGTA", 0.2),
    ("ACGT", "AAAA", 0.75),
])
def test_fast_cer(str1, str2, expected_cer):
    np_str1 = fast_str_to_numpy(str1)
    np_str2 = fast_str_to_numpy(str2)
    cer = fast_cer(str1, str2)
    assert cer == expected_cer


def test_cer_raises():
    with pytest.raises(ValueError):
        fast_cer("ACGT", "ACGTA")  # different lengths