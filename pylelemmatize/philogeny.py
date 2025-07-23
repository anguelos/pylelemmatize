from collections import defaultdict
from typing import Dict, List, Set, Tuple
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
from .abstract_mapper import char_similarity
import numpy as np


def branch_name(n: List[str]) -> str:
    return f"{'+'.join(sorted(n))}"


def leaf_name(n:str)->str:
    return n


def get_dm(items:np.ndarray, similarity_f=char_similarity) -> np.ndarray:
    dm = np.zeros([len(items), len(items)])
    for n1, ch1 in enumerate(items):
        for n2, ch2 in enumerate(items):
            dm[n1, n2] = similarity_f(ch1, ch2)
    return dm



def get_branches(ax, linkage_matrix, symbols):
    locations = np.zeros()


def plot_dendrogram(DM,linkage_matrix, labels, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    def get_leaf_labels(id, n, _labels):
        if id < n:
            return [_labels[id]]
        else:
            left, right = int(linkage_matrix[id - n, 0]), int(linkage_matrix[id - n, 1])
            return get_leaf_labels(left, n, _labels) + get_leaf_labels(right, n, _labels)

    # Custom leaf labeling function
    def llf(id):
        leaf_labels = [leaf_name(n) for n in sorted(get_leaf_labels(id, len(DM), labels))]
        return ''.join(leaf_labels)

    # Plot the dendrogram with custom leaf labelsextended_keyboard_characters
    dendrogram(
        linkage_matrix,
        leaf_label_func=llf,  # Use the custom label function
        leaf_rotation=90,
        leaf_font_size=12,
        show_contracted=False  # Display intermediary branches
    )
    ax.set_yscale("symlog")
    #ax.yscale('symlog')
    ax.set_title('Dendrogram with Custom Node Labels')
    return fig, ax


def main_char_similarity_tree():
    import fargv
    import string
    from .charset_iso import get_encoding_dicts
    char_dicts = get_encoding_dicts()
    p = {
        "characters": char_dicts["iso8859_2"]
    }
    args, _ = fargv.fargv(p)
    characters = np.array(list(args.characters))
    dm = get_dm(list(characters))
    linkage_matrix = linkage(dm, "ward")
    print(linkage_matrix)
    print(linkage_matrix)
    #print(linkage.__doc__)
    f, ax = plot_dendrogram(dm, linkage_matrix=linkage_matrix, labels = list(args.characters))
    plt.show()