from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from .abstract_mapper import char_similarity


def branch_name(n: List[str]) -> str:
    return f"{'+'.join(sorted(n))}"


def leaf_name(n:str)->str:
    return n


def get_dm(items:np.ndarray, similarity_f=char_similarity) -> np.ndarray:
    dm = np.zeros([len(items), len(items)])
    for n1, ch1 in enumerate(items):
        for n2, ch2 in enumerate(items):
            dm[n1, n2] = similarity_f(ch1, ch2)
    return 1-dm


def make_branch_label(all_labels, indexes):
    """Concatenate original labels of given indexes."""
    return "+".join(sorted(all_labels[i] for i in indexes))

def adjust_linkage_distances(linkage_matrix, child_d=0.1, cousin_d=1.0):
    """Ensure min distance between parent and children is child_d, and max between cousins is cousin_d."""
    n = linkage_matrix.shape[0] + 1
    cluster_children = {}

    for i, (a, b, dist, _) in enumerate(linkage_matrix):
        idx = n + i
        a, b = int(a), int(b)
        cluster_children[idx] = set()

        for child in (a, b):
            if child < n:
                cluster_children[idx].add(child)
            else:
                cluster_children[idx].update(cluster_children[child])

    # Adjust distances
    new_linkage = linkage_matrix.copy()
    for i in range(linkage_matrix.shape[0]):
        parent_idx = n + i
        children = [int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1])]
        parent_dist = new_linkage[i, 2]

        # Ensure children are at least child_d below parent
        for j in range(i):
            child_idx = n + j
            if child_idx in children:
                if new_linkage[j, 2] >= parent_dist - child_d:
                    new_linkage[j, 2] = parent_dist - child_d

        # Ensure cousins (not direct children) are not too far apart
        for j in range(i):
            cousin_idx = n + j
            if cousin_idx not in children:
                if new_linkage[j, 2] > parent_dist - cousin_d:
                    new_linkage[j, 2] = parent_dist - cousin_d

        # Keep distances monotonic
        new_linkage[i, 2] = max(new_linkage[i, 2], 0.001)
    return new_linkage


def plot_annotated_dendrogram_sns(labels, dm, child_d=0.1, cousin_d=1.0, clustering_mode='ward', figsize=(10, 5)):
    from scipy.spatial.distance import squareform
    import seaborn as sns
    import pandas as pd
    from scipy.cluster.hierarchy import linkage
    """Plot dendrogram with concatenated leaf labels and adjusted distances."""
    condensed_dm = squareform(dm)
    # Step 1: Compute linkage
    p_dm = pd.DataFrame(dm, index=labels, columns=labels)
    #condensed_dm = squareform(dm)

    Z = linkage(dm, method=clustering_mode)  # or 'complete', 'single', etc.
    print("Linkage matrix:\n", Z)

    # Step 2: Adjust linkage distances
    #Z = adjust_linkage_distances(Z, child_d, cousin_d)
    #clustergrid = sns.clustermap(p_dm, row_linkage=Z, col_linkage=Z, figsize=figsize)
    clustergrid = sns.clustermap(p_dm, row_linkage=Z, col_linkage=Z, figsize=figsize)
    # Step 3: Plot
    fig = clustergrid.figure
    ax = clustergrid.ax_row_dendrogram
    #ddata = sns.dendrogram(Z, labels=labels, ax=ax)
    #ax.set_title("Annotated Dendrogram")
    return fig, [ax]


def plot_annotated_dendrogram_plt(labels, dm, child_d=0.1, cousin_d=1.0, clustering_mode='ward', figsize=(10, 5)):
    """Plot dendrogram with concatenated leaf labels and adjusted distances."""
    from scipy.spatial.distance import squareform
    import matplotlib.pyplot as plt
    import pandas as pd
    condensed_dm = squareform(dm)
    # Step 1: Compute linkage
    p_dm = pd.DataFrame(dm, index=labels, columns=labels)
    #condensed_dm = squareform(dm)

    Z = linkage(dm, method=clustering_mode)  # or 'complete', 'single', etc.
    print("Linkage matrix:\n", Z)
    fig, ax = plt.subplots(figsize=figsize)
    ddata = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=10)
    ax.set_title("Annotated Dendrogram")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Distance")
    ax.set_xticklabels([leaf_name(label) for label in labels], rotation=90)

    return fig, [ax]


def main_char_similarity_tree():
    import fargv
    import string
    from .charset_iso import get_encoding_dicts
    from matplotlib import pyplot as plt
    char_dicts = get_encoding_dicts()
    p = {
        "characters": char_dicts["iso8859_2"],
        "clustering_mode": ("ward", "single", "complete"),
        "rendering_mode": ("sns", "matplotlib"),
        "figure_size": "6,5",
        "output_name": ""

    }
    args, _ = fargv.fargv(p)
    characters = np.array(list(args.characters))
    dm = get_dm(list(characters))
    #print(linkage.__doc__)
    figsize = tuple(map(float, args.figure_size.split(",")))
    if args.rendering_mode == "sns":
        fig, ax = plot_annotated_dendrogram_sns(labels=list(args.characters), dm=dm, child_d=0.1, cousin_d=.01, clustering_mode=args.clustering_mode, figsize=figsize)
    else:
        fig, ax = plot_annotated_dendrogram_plt(labels=list(args.characters), dm=dm, child_d=0.1, cousin_d=.01, clustering_mode=args.clustering_mode, figsize=figsize)

    if args.output_name != "":
        fig.savefig(args.output_name)
    else:
        #plt.figure(fig.number)
        plt.show(block=True)