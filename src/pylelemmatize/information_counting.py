from typing import Tuple
from .abstract_mapper import fast_str_to_numpy
import numpy as np


def information_measurements(seq: str) -> Tuple[float, float, float, int]:
    """Calculate information measurements for a given sequence.
    This function computes various information-theoretic measurements for a given 
    sequence, including the Hartley measure, Shannon measure, alphabet bits, and 
    sequence size.
    Parameters
    ----------
    seq : str
        The input sequence for which information measurements are calculated.
    Returns
    -------
    hartley_measure : float
        The Hartley measure, representing the maximum information content of the sequence.
    shannon_measure : float
        The Shannon measure, representing the entropy-based information content of the sequence.
    alphabet_bits : float
        The number of bits required to represent the alphabet of the sequence.
    sequence_size : int
        The size of the input sequence.
    Notes
    -----
    - The input sequence is first converted to a numerical representation using `fast_str_to_numpy`.
    - The alphabet of the sequence is determined, and its size is used to calculate the alphabet bits.
    - The zero-order entropy is computed based on the probabilities of the symbols in the sequence.
    """

    int_seq = fast_str_to_numpy(seq)
    alphabet, alphabet_counts = np.unique(int_seq, return_counts=True)
    alphabet_bits = np.log2(alphabet.size)
    zero_order_entropy = -np.sum((alphabet_counts / alphabet_counts.sum()) * np.log2(alphabet_counts / alphabet_counts.sum()))
    hartley_measure = alphabet_bits * len(int_seq)
    shannon_measure = zero_order_entropy * len(int_seq)
    return hartley_measure, shannon_measure, alphabet_bits, int_seq.size