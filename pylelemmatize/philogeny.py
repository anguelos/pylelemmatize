from collections import defaultdict
from typing import Dict, List, Set, Tuple
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
from unidecode import unidecode
import unicodedata
from itertools import chain, combinations


def any_to_ascii(alphabet_str)-> Dict[str, str]:
    """
    Convert any string to ASCII using unidecode.
    
    Args:
        alphabet_str (str): The input string to convert.
        
    Returns:
        str: The ASCII representation of the input string.
    """
    return {chr: unidecode(chr) for chr in alphabet_str}


def composite_to_simpler(alphabet_str: str) -> Dict[str, str]:
    assert alphabet_str == unicodedata.normalize('NFC', alphabet_str), "Input string must be in NFC (compressed) form."
    compressed_to_decompressed_alphabet = {chr: list(unicodedata.normalize('NFD', chr)) for chr in alphabet_str}
    all_descendants = defaultdict(lambda: list())
    def chr_length(chr: str) -> int:
        return (chr, len(unicodedata.normalize('NFD', chr)))
    for full_chr, chr_list in compressed_to_decompressed_alphabet.items():
        if len(chr_list) == 1:
            all_descendants[chr_list[0]].append(chr_list[0])
            continue
        base_chr = ''.join(c for c in chr_list if unicodedata.combining(c) == 0)
        assert len(base_chr) == 1, "Base character must be a single character."
        for new_chr_list in combinations(chr_list, len(chr_list) - 1):
            try:                
                new_chr = ''.join(new_chr_list)
                _ = unicodedata.name(new_chr)
                all_descendants[new_chr].append(full_chr)
            except ValueError:
                continue  # skip invalid characters
    all_descendants = {k: list(sorted(set(v))) for k, v in all_descendants.items()}
    return all_descendants


