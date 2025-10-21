from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np


class SubstitutionMutator:
    """
    Distort integer labels y according to a target confusion-frequency matrix cm.
    Each row i of cm defines P(pred=j | true=i) after row-normalization.

    Also provides a static method to compute a confusion matrix from two label arrays.
    """

    def __init__(self, cm: np.ndarray, random_state: int | np.random.Generator | None = None):
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError("cm must be a square 2D array (K x K).")
        self.K = cm.shape[0]

        cm = np.asarray(cm, dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        probs = np.zeros_like(cm, dtype=float)
        nz = row_sums[:, 0] > 0
        probs[nz] = cm[nz] / row_sums[nz]
        if np.any(~nz):
            i0 = np.where(~nz)[0]
            probs[i0, i0] = 1.0

        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0

        self.probs = probs
        self.cdf = cdf

        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng(random_state)

    def distort(self, y: np.ndarray) -> np.ndarray:
        """Return a noisy version of y drawn according to the confusion model."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D integer array.")
        if (y < 0).any() or (y >= self.K).any():
            raise ValueError("labels in y must be in [0, K-1].")

        N = y.size
        if N == 0:
            return y.copy()

        cdf_rows = self.cdf[y]
        u = self.rng.random(N)
        y_noisy = (u[:, None] <= cdf_rows).argmax(axis=1).astype(int)
        return y_noisy

    # --- NEW STATIC METHOD ---
    @staticmethod
    def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
        """
        Compute a confusion matrix from two integer label arrays.

        Parameters
        ----------
        y_true : np.ndarray of shape (N,)
            True class labels.
        y_pred : np.ndarray of shape (N,)
            Predicted (or distorted) labels.
        num_classes : int | None
            Total number of classes. If None, inferred from max value in both arrays.

        Returns
        -------
        cm : np.ndarray of shape (K, K)
            Confusion frequency matrix where cm[i, j] counts how many times
            true class i was predicted as class j.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        if num_classes is None:
            num_classes = max(y_true.max(), y_pred.max()) + 1

        if (y_true < 0).any() or (y_pred < 0).any():
            raise ValueError("Labels must be nonnegative integers")

        # Flatten indices to 1D and accumulate counts efficiently
        cm = np.bincount(
            num_classes * y_true + y_pred,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        return cm


class SubstitutionAugmenter(SubstitutionMutator):
    @staticmethod

    @staticmethod
    def mappingdict_to_nparrays(alphabet_mapper: Dict[str, int], max_ord: int=-1) -> np.ndarray:
        np_symbols = len(alphabet_mapper)
        dense_to_sparse = np.zeros((np_symbols,), dtype=np.int32)
        if max_ord == -1:
            max_ord = max([ord(symbol) for symbol in alphabet_mapper.keys()])
        sparse_to_dense = np.zeros(max_ord + 1, dtype=np.int32)
        for symbol, dense_idx in alphabet_mapper.items():
            sparse_idx = ord(symbol)
            dense_to_sparse[dense_idx] = sparse_idx
            sparse_to_dense[sparse_idx] = dense_idx
        return dense_to_sparse, sparse_to_dense

    def __init__(self, parallel_corpus: List[Tuple[str, str]], alphabet_str = "", random_state: int | np.random.Generator | None = None):
        dense_to_sparse, sparse_to_dense = self.mappingdict_to_nparrays(alphabet_str = alphabet_str)
        self.parallel_corpus = parallel_corpus
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng(random_state)


    def augment(self, word: str) -> str:
        num_distortions = self.sample_distortion()
        word_chars = list(word)
        for _ in range(num_distortions):
            if len(word_chars) == 0:
                break
            pos = np.random.randint(0, len(word_chars))
            del word_chars[pos]
        return "".join(word_chars)
    

# if __name__ == "__main__":
#     cm_true = np.array([
#         [90,  5,  5],
#         [10, 80, 10],
#         [ 3, 3, 94]
#     ], dtype=int)
    
#     y = np.random.choice(3, 1000000, p=[0.7, 0.25, 0.05])
#     noiser = SubstitutionMutator(cm_true, random_state=0)
#     y_noisy = noiser.distort(y)

#     cm_est = SubstitutionMutator.compute_confusion(y, y_noisy)
#     print("Estimated confusion:\n", cm_est)
#     print("Row-normalized:\n", cm_est / cm_est.sum(axis=1, keepdims=True))


from typing import Tuple, List
import numpy as np

def edit_distance_with_confusion(
    s1: str,
    s2: str,
    alphabet: str,
    null_symbol: str = "∅",
) -> Tuple[int, np.ndarray, np.ndarray, str]:
    """
    Compute Levenshtein edit distance and a confusion matrix (with a null class).
    Also return a version of s1 where we apply only the non-substitution ops from
    the optimal path: insertions and deletions are realized, substitutions are NOT
    realized (the original s1 symbol is kept).

    Returns
    -------
    distance : int
    confusion : np.ndarray (K+1, K+1)
    labels : np.ndarray (K+1,)
    s1_no_subst : str
        s1 after applying only insertions/deletions from the optimal alignment;
        substitutions are ignored (keep the s1 symbol).
    """
    # Validate alphabet uniqueness & contents
    alpha_list = list(alphabet)
    if len(set(alpha_list)) != len(alpha_list):
        raise ValueError("Alphabet contains duplicate symbols.")
    if null_symbol in alpha_list:
        raise ValueError(f"null_symbol '{null_symbol}' must not be in the alphabet.")

    # Validate strings
    bad_s1 = {ch for ch in s1 if ch not in alpha_list}
    bad_s2 = {ch for ch in s2 if ch not in alpha_list}
    if bad_s1 or bad_s2:
        raise ValueError(
            "Input strings contain symbols not in the alphabet. "
            f"Bad in s1: {sorted(bad_s1)}; Bad in s2: {sorted(bad_s2)}"
        )

    # Labels and indices
    labels: List[str] = alpha_list + [null_symbol]
    label_to_idx = {ch: i for i, ch in enumerate(labels)}
    null_idx = label_to_idx[null_symbol]

    n, m = len(s1), len(s2)

    # DP + backpointers
    dp = np.zeros((n + 1, m + 1), dtype=int)
    back = np.empty((n + 1, m + 1), dtype=np.int8)  # 0=start, 1=diag, 2=up(del), 3=left(ins)

    for i in range(1, n + 1):
        dp[i, 0] = i
        back[i, 0] = 2
    for j in range(1, m + 1):
        dp[0, j] = j
        back[0, j] = 3
    back[0, 0] = 0

    for i in range(1, n + 1):
        s1c = s1[i - 1]
        for j in range(1, m + 1):
            s2c = s2[j - 1]
            cost_sub = 0 if s1c == s2c else 1
            diag = dp[i - 1, j - 1] + cost_sub
            up = dp[i - 1, j] + 1
            left = dp[i, j - 1] + 1
            best = min(diag, up, left)
            dp[i, j] = best
            if best == diag:
                back[i, j] = 1
            elif best == up:
                back[i, j] = 2
            else:
                back[i, j] = 3

    distance = int(dp[n, m])

    # Backtrace: fill confusion and build the "no-substitution-realization" string
    K = len(labels)
    confusion = np.zeros((K, K), dtype=int)

    i, j = n, m
    out_chars_rev: List[str] = []

    while not (i == 0 and j == 0):
        move = back[i, j]
        if move == 1:  # diag: match or substitution
            src = s1[i - 1]
            tgt = s2[j - 1]
            confusion[label_to_idx[src], label_to_idx[tgt]] += 1
            # KEY DIFFERENCE:
            # - If it's a match, appending src or tgt is identical.
            # - If it's a substitution, we DO NOT realize it; keep the source char.
            out_chars_rev.append(src)
            i -= 1
            j -= 1
        elif move == 2:  # deletion (src->null): remove src from output
            src = s1[i - 1]
            confusion[label_to_idx[src], null_idx] += 1
            # deletion => emit nothing
            i -= 1
        else:  # insertion (null->tgt): insert tgt into output
            tgt = s2[j - 1]
            confusion[null_idx, label_to_idx[tgt]] += 1
            out_chars_rev.append(tgt)
            j -= 1
    s1_no_subst = "".join(reversed(out_chars_rev))
    return distance, confusion, np.array(labels), s1_no_subst


def create_substitutiononly_parallel_corpus(textlines: List[Tuple[str, str]]):
    alphabet = "".join(sorted(set("".join([f"{p}{g}" for p, g in textlines]))))
    # Create a list to hold the modified text lines
    modified_lines = []
    for prediction, groundtruth in textlines:
        # Call the edit_distance_with_confusion function
        dist, conf, labels, no_sub = edit_distance_with_confusion(prediction, groundtruth, alphabet)
        # Append the no_substitution version of s1 to the modified lines
        modified_lines.append((no_sub, groundtruth))
    return modified_lines


def main_create_postcorrection_tsv():
    """creates a TSV where on substitutions are considered erros to train delemmatiser from arbitrary prediction-target pairs
    """
    import fargv
    import sys
    p={
        "ocr_prediction_target_tsv":"",
        "substion_only_tsv":"",
        "allow_overwrite": False,
        "min_line_length": 50,
        "max_edit_distance_tolerated": .2,
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    if args.ocr_prediction_target_tsv == "":
        input_fd = sys.stdin
    else:
        input_fd = open( args.ocr_prediction_target_tsv, "r")

    if args.substion_only_tsv == "":
        output_fd = sys.stdout
    else:
        if not Path(args.substion_only_tsv).exists() or args.allow_overwrite:
            output_fd = open( args.substion_only_tsv, "w")
        else:
            raise IOError(f" Could not write to {args.substion_only_tsv}")
    rejected = 0
    all_accepted = []
    for input_line in input_fd.readlines():
        input_line = input_line.strip().split("\t")
        if len(input_line) == 2 and len(input_line[1]) >= args.min_line_length:
            all_accepted.append(input_line)
        else:
            rejected+=1
    alphabet = "".join(sorted(set("".join([f"{p}{g}" for p, g in all_accepted]))))
    conf_acc = np.zeros([len(alphabet)+1, len(alphabet)+1])
    all_dist = 0
    for pred, gt in all_accepted:
        dist, conf, labels, no_sub = edit_distance_with_confusion(pred, gt, alphabet)
        all_dist+= dist
        conf_acc += conf
        if len(no_sub) != len(gt):
            raise ValueError(f" Length mismatch after substitution-only processing\nPred: {pred}\nGT: {gt}\nNoSub: {no_sub}")
        if (dist / len(gt)) > args.max_edit_distance_tolerated:
            rejected += 1
        else:
            print(f"{no_sub}\t{gt}", file=output_fd)
    output_fd.flush()
    if args.verbose:
        print(f"Read {rejected + len(all_accepted)} lines, kept {len(all_accepted)}, rejected {rejected} lines", file=sys.stderr)
        print(f"Observed alphabet: {repr(alphabet)}", file=sys.stderr)


def main_textline_full_cer():
    import fargv
    import sys
    import tqdm
    from pylelemmatize.demapper_lstm import DemapperLSTM
    p = {
        "src1_txt": "",
        "src2_txt": "",
        "ignore_lines_with_cer_above": 1.,
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    textlines1 = open(args.src1_txt,"r").readlines()
    textlines2 = open(args.src2_txt,"r").readlines()
    assert len(textlines1) == len(textlines2), "Input files must have the same number of lines."
    alphabet = "".join(sorted(set("".join(textlines1 + textlines2))))
    all_dist = 0
    conf_acc = np.zeros([len(alphabet)+1, len(alphabet)+1])
    dropped_lines = 0
    dropped_chars = 0
    total_chars = 0
    if args.verbose:
        progress = tqdm.tqdm(total=len(textlines1), desc="Processing lines")
    for l1, l2 in zip(textlines1, textlines2):
        dist, conf, labels, no_sub = edit_distance_with_confusion(l1, l2, alphabet)
        if (dist / len(l2.strip())) > args.ignore_lines_with_cer_above:
            dropped_lines += 1
            dropped_chars += len(l2)
            continue
        total_chars += len(l2)
        all_dist += dist
        conf_acc += conf
        if args.verbose:
            progress.update(1)
    if args.verbose:
        progress.close()
    cer = all_dist / total_chars
    insertions = conf_acc[:, -1].sum()
    deletions = conf_acc[-1, :].sum()
    substitutions = conf_acc.sum() - np.trace(conf_acc) - insertions - deletions
    print(f"Dropped lines: {dropped_lines}, Dropped characters: {dropped_chars}")
    print(f"Total characters (after dropping): {total_chars}")
    print(f"Total edit distance: {all_dist}")
    print(f"CER: {cer:.4f} (Insertions: {insertions}, Deletions: {deletions}, Substitutions: {substitutions})")