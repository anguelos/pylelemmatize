from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from .fast_mapper import LemmatizerBMP
import pickle


class CharConfusionMatrix:
    @staticmethod
    def edit_distance(s1: np.ndarray, s2: np.ndarray) -> Tuple[int, np.ndarray]:
        """Compute the Levenshtein edit distance between two sequences.

        This function calculates the minimum number of single-character edits 
        (insertions, deletions, or substitutions) required to change one sequence 
        into the other. It also returns the dynamic programming (DP) matrix used 
        to compute the distance.

        Parameters
        ----------
        s1 : np.ndarray
            The first sequence as a NumPy array.
        s2 : np.ndarray
            The second sequence as a NumPy array.

            The Levenshtein edit distance between `s1` and `s2`.
            The DP matrix used to compute the distance, where `dp[i, j]` represents 
            the edit distance between the first `i` characters of `s1` and the first 
            `j` characters of `s2`.

        Examples
        --------
        >>> import numpy as np
        >>> s1 = np.array(['a', 'b', 'c'])
        >>> s2 = np.array(['a', 'c', 'd'])
        >>> distance, dp = edit_distance(s1, s2)
        >>> distance
        2
        >>> dp
        array([[0, 1, 2, 3],
               [1, 0, 1, 2],
               [2, 1, 1, 2],
               [3, 2, 2, 2]])
        """
        n, m = len(s1), len(s2)
        dp = np.zeros((n + 1, m + 1), dtype=int)
        for i in range(n + 1):
            dp[i, 0] = i
        for j in range(m + 1):
            dp[0, j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost_sub = 0 if s1[i - 1] == s2[j - 1] else 1
                diag = dp[i - 1, j - 1] + cost_sub
                up = dp[i - 1, j] + 1
                left = dp[i, j - 1] + 1
                dp[i, j] = min(diag, up, left)
        distance = dp[n, m]
        return distance, dp


    def backtrace_ed_matrix(self, input_seq: np.ndarray, gt_seq: np.ndarray, dp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backtraces the edit distance matrix to compute the alignment path, operation types, 
        ground truth substitutions, and confusion matrix.
        Parameters
        ----------
        input_seq : np.ndarray
            The input sequence represented as an array of indices.
        gt_seq : np.ndarray
            The ground truth sequence represented as an array of indices.
        dp : np.ndarray
            The dynamic programming matrix containing the edit distances.
        Returns
        -------
        path : np.ndarray
            The alignment path as an array of (input_index, gt_index) pairs.
        operation_type : np.ndarray
            The sequence of operation types:
            - 0: Match
            - 1: Substitution
            - 2: Deletion
            - 3: Insertion
        gt_sub_input : np.ndarray
            The ground truth sequence with substitutions applied.
        cm : np.ndarray
            The confusion matrix representing the counts of matches, substitutions, 
            insertions, and deletions. The matrix has dimensions 
            (len(alphabet), len(alphabet)), where the first row/column represents 
            insertions/deletions.
        """
        cm = np.zeros((len(self.alphabet), len(self.alphabet)), dtype=np.int32)
        inp_idx = len(input_seq)
        gt_idx = len(gt_seq)
        path = []
        operation_type = []
        gt_sub_input = []

        while inp_idx > 0 and gt_idx > 0:
            choice = ((dp[inp_idx - 1, gt_idx - 1]-.00001, (-1, -1), 0), 
                      (dp[inp_idx - 1, gt_idx], (-1, 0), 2), 
                      (dp[inp_idx, gt_idx - 1], (0, -1), 3))

            _, (di, dj), op_type = min(choice, key=lambda x: x[0])
            inp_idx += di
            gt_idx += dj
            path.append((inp_idx, gt_idx))
            
            if op_type == 0:
                gt_sub_input.append(input_seq[inp_idx])
                op_type = 0 if input_seq[inp_idx] == gt_seq[gt_idx] else 1
                cm[input_seq[inp_idx], gt_seq[gt_idx]] += 1
            elif op_type == 3:  # Insertion
                gt_sub_input.append(gt_seq[gt_idx])
                cm[0, gt_seq[gt_idx]] += 1
            elif op_type == 2:  # Deletion
                cm[input_seq[inp_idx], 0] += 1
            operation_type.append(op_type)
        
        while gt_idx > 0:
            gt_idx -= 1
            path.append((inp_idx, gt_idx))
            operation_type.append(3)
            gt_sub_input.append(gt_seq[gt_idx])
            cm[0, gt_seq[gt_idx]] += 1

        while inp_idx > 0:
            inp_idx -= 1
            path.append((inp_idx, gt_idx))
            operation_type.append(2)
            cm[input_seq[inp_idx], 0] += 1
        return np.array(path)[::-1, :], np.array(operation_type)[::-1], np.array(gt_sub_input[::-1]), cm


    def ingest_textline_observation(self, pred_line: str, gt_line: str) -> Tuple[str, int]:
        """
        Processes a pair of predicted and ground truth text lines, computes the edit distance,
        and updates the confusion matrix.
                
        This method performs the following steps:
        1. Converts the predicted and ground truth text lines into dense integer label sequences.
        2. Computes the edit distance and dynamic programming matrix between the two sequences.
        3. Performs a backtrace on the edit distance matrix to generate the ground truth substitution input and updates the confusion matrix.
        4. Returns the ground truth substitution input and the computed edit distance.


        Parameters
        ----------
        pred_line : str
            The predicted text line as a string.
        gt_line : str
            The ground truth text line as a string.

        Returns
        -------
        Tuple[str, int]
            A tuple containing:
            - The ground truth substitution input as a string.
            - The edit distance between the predicted and ground truth text lines.

        """
        dense_pred = self.alphabet.str_to_intlabel_seq(pred_line)
        dense_gt = self.alphabet.str_to_intlabel_seq(gt_line)
        distance, dp = self.edit_distance(dense_pred, dense_gt)
        _, _, gt_sub_input, cm = self.backtrace_ed_matrix(dense_pred, dense_gt, dp)
        self.cm += cm
        return self.alphabet.intlabel_seq_to_str(gt_sub_input), distance


    def generate_random_substitution_sequences(self, seq) -> np.ndarray:
        """
        Generate random substitution sequences based on a conditional probability matrix.

        This method generates a sequence of random substitutions for the input
        sequence ``seq`` using the confusion matrix as conditional probability. Each
        output symbol is sampled from the conditional probabilities of the
        corresponding input symbol.

        Parameters
        ----------
        seq : np.ndarray
            Input sequence represented as a NumPy array of integers. Each integer
            corresponds to a symbol in the vocabulary.

        Returns
        -------
        np.ndarray
            A NumPy array of the same shape as ``seq``, where each element is a
            randomly substituted symbol based on the conditional probability matrix.

        Examples
        --------
        >>> import numpy as np
        >>> cm = np.array([[0, 0.5, 0.5],
        ...                [0, 0.7, 0.3],
        ...                [0, 0.4, 0.6]])
        >>> seq = np.array([1, 2, 1])
        >>> augmenter = SubstitutionAugmenter(cm)
        >>> out = augmenter.generate_random_substitution_sequences(seq)
        >>> out.shape == seq.shape
        True
        """
        
        e = 1e-8
        cm = self.cm.copy()
        cm[:, 0] = 0  # Zero out deletions
        cm[0, :] = 0  # Zero out insertions
        cm = cm / (cm.sum(axis=1, keepdims=True) + e)  # Normalize to probabilities
        all_cond_probabilities_cdf = np.cumsum(cm, axis=1)
        rnd_vals = np.random.rand(seq.shape[0])
        res = np.zeros_like(seq)
        for n in range(seq.shape[0]):
            input_symbol  = seq[n]
            res[n] = np.searchsorted(all_cond_probabilities_cdf[input_symbol, :], rnd_vals[n])
        return res
    
    def get_self_supervision_textline(self, input_line: str) -> str:
        dense_input = self.alphabet.str_to_intlabel_seq(input_line)
        mutated = self.generate_random_substitution_sequences(dense_input)
        return self.alphabet.intlabel_seq_to_str(mutated)

    def save(self, file_path: Union[str, Path]):
        pickle.dump([self.alphabet.dst_alphabet_str, self.cm], open(file_path, "wb"))


    @staticmethod
    def load(file_path: Union[str, Path]) -> "CharConfusionMatrix":
        dst_alphabet_str, cm = pickle.load(open(file_path, "rb"))
        lemmatizer = LemmatizerBMP.from_alphabet_mapping(dst_alphabet_str, dst_alphabet_str)
        char_cm = CharConfusionMatrix(lemmatizer)
        char_cm.cm = cm
        return char_cm


    def __init__(self, alphabet: Union[LemmatizerBMP, str]):
        if isinstance(alphabet, str):
            alphabet = LemmatizerBMP.from_alphabet_mapping(alphabet, alphabet)
        self.alphabet = alphabet
        self.cm = np.zeros((len(self.alphabet), len(self.alphabet)), dtype=int)


    def get_matrix(self) -> np.ndarray:
        return self.cm




# def create_substitutiononly_parallel_corpus(textlines: List[Tuple[str, str]]):
#     alphabet = "".join(sorted(set("".join([f"{p}{g}" for p, g in textlines]))))
#     # Create a list to hold the modified text lines
#     modified_lines = []
#     for prediction, groundtruth in textlines:
#         # Call the edit_distance_with_confusion function
#         dist, conf, labels, no_sub = edit_distance_with_confusion(prediction, groundtruth, alphabet)
#         # Append the no_substitution version of s1 to the modified lines
#         modified_lines.append((no_sub, groundtruth))
#     return modified_lines



def main_get_augmented_substitutiononly_parallel_corpus():
    import fargv
    p = {
        "gt_txt": "",
        "src_txt": "",
        "alphabet_str": "",
        "out_txt": "",
    }
    args, _ = fargv.fargv(p)
    if args.src_txt == "" and args.gt_txt == "":
        #all_lines = sys.stdin.readlines()
        #return
        pass



def main_textline_full_cer():
    """
    Compute the full CER (including substitutions) between two textline files.
    """
    import fargv
    import sys
    import tqdm
    from pylelemmatize.demapper_lstm import DemapperLSTM
    p = {
        "src_tsv": "",
        "src1_txt": "",
        "src2_txt": "",
        "ignore_lines_with_cer_above": 1.,
        "verbose": False,
    }
    args, _ = fargv.fargv(p)
    if args.src_tsv != "":
        assert args.src1_txt == "" and args.src2_txt == "", "If src_tsv is provided, src1_txt and src2_txt must be empty"
        textlines1, textlines2 = zip(*[line.split("\t")[:2] for line in open(args.src_tsv,"r").readlines()])
    elif args.src1_txt != "" and args.src2_txt != "":
        assert args.src_tsv == "", "If src1_txt and src2_txt are provided, src_tsv must be empty"
        textlines1 = open(args.src1_txt,"r").readlines()
        textlines2 = open(args.src2_txt,"r").readlines()
    elif args.src1_txt == "" and args.src2_txt == "" and args.src_tsv == "":
        textlines1 = []
        textlines2 = []
        for line in sys.stdin.readlines():
            p1, p2 = line.split("\t")[:2]
            textlines1.append(p1)
            textlines2.append(p2)
    assert len(textlines1) == len(textlines2), "Input files must have the same number of lines."
    alphabet_str ="".join(sorted(set("".join(textlines1 + textlines2))))
    alphabet = LemmatizerBMP.from_alphabet_mapping(alphabet_str, alphabet_str)
    all_dist = 0
    conf_acc = np.zeros([len(alphabet), len(alphabet)])
    dropped_lines = 0
    dropped_chars = 0
    total_chars = 0
    if args.verbose:
        progress = tqdm.tqdm(total=len(textlines1), desc="Processing lines")
    cm = CharConfusionMatrix(alphabet)
    for l1, l2 in zip(textlines1, textlines2):
        np_l1 = alphabet.str_to_intlabel_seq(l1)
        np_l2 = alphabet.str_to_intlabel_seq(l2)
        dist, dp = cm.edit_distance(np_l1, np_l2)
        _, _, _, conf = cm.backtrace_ed_matrix(np_l1, np_l2, dp)
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
    insertions = conf_acc[:, 0].sum()
    deletions = conf_acc[0, :].sum()
    invalids = conf_acc[0, 0]
    substitutions = conf_acc.sum() - np.trace(conf_acc) - (insertions + deletions + invalids)
    print(f"Dropped lines: {dropped_lines}, Dropped characters: {dropped_chars}")
    print(f"Total characters (after dropping): {total_chars}")
    print(f"Total edit distance: {all_dist}")
    print(f"CER: {cer:.4f} (Insertions: {insertions}, Deletions: {deletions}, Substitutions: {substitutions})")