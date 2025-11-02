from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from .fast_mapper import LemmatizerBMP
import pickle


# def edit_distance(s1: np.ndarray, s2: np.ndarray) -> Tuple[int, np.ndarray]:
#     """
#     Compute Levenshtein edit distance between two sequences s1 and s2.
#     Also returns the DP matrix used to compute the distance.
#     Retursns
#     -------
#     distance : int
#         The Levenshtein edit distance between s1 and s2.
#     dp : np.ndarray
#         The DP matrix used to compute the distance.
#     """
#     n, m = len(s1), len(s2)
#     dp = np.zeros((n + 1, m + 1), dtype=int)
#     for i in range(n + 1):
#         dp[i, 0] = i
#     for j in range(m + 1):
#         dp[0, j] = j
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             cost_sub = 0 if s1[i - 1] == s2[j - 1] else 1
#             diag = dp[i - 1, j - 1] + cost_sub
#             up = dp[i - 1, j] + 1
#             left = dp[i, j - 1] + 1
#             dp[i, j] = min(diag, up, left)
#     distance = dp[n, m]
#     return distance, dp


# def substitution_only_input(input_seq: np.ndarray, gt_seq: np.ndarray, dp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Given the edit distance DP matrix, backtrace to find the optimal path,
#     and create a version of input_seq where only substitutions are realized.
#     Returns
#     -------
#     path : np.ndarray
#         The optimal alignment path.
#     operation_type : np.ndarray
#         The types of operations for each step in the path.
#     gt_sub_input : np.ndarray
#         The ground truth input sequence after applying substitutions.
#     cm : np.ndarray
#         The confusion matrix for the substitutions.
#     """
#     cm = np.zeros((5, 5)), dtype=np.int32)
#     inp_idx = len(input_seq)
#     gt_idx = len(gt_seq)
#     res = []
#     path = []
#     operation_type = []
#     gt_sub_input = []

#     while inp_idx > 0 and gt_idx > 0:
#         choice = ((dp[inp_idx - 1, gt_idx - 1], (-1, -1), 0), (dp[inp_idx - 1, gt_idx], (-1, 0), 2), (dp[inp_idx, gt_idx - 1], (0, -1), 3))
#         _, (di, dj), op_type = min(choice, key=lambda x: x[0])
#         inp_idx += di
#         gt_idx += dj
#         path.append((inp_idx, gt_idx))
        
#         if op_type == 0:
#             gt_sub_input.append(input_seq[inp_idx])
#             op_type = 0 if input_seq[inp_idx] == gt_seq[gt_idx] else 1
#             cm[input_seq[inp_idx], gt_seq[gt_idx]] += 1
#         elif op_type == 3:
#             gt_sub_input.append(gt_seq[gt_idx])
#             cm[0, gt_seq[gt_idx]] += 1
#         elif op_type == 2:
#             cm[input_seq[inp_idx], 0] += 1
#         operation_type.append(op_type)
    
#     while gt_idx > 0:
#         gt_idx -= 1
#         path.append((inp_idx, gt_idx))
#         operation_type.append(3)
#         gt_sub_input.append(gt_seq[gt_idx])
#         cm[0, gt_seq[gt_idx]] += 1

#     while inp_idx > 0:
#         inp_idx -= 1
#         path.append((inp_idx, gt_idx))
#         operation_type.append(2)
#         gt_sub_input.append(input_seq[inp_idx])
#         cm[input_seq[inp_idx], 0] += 1

#     return np.array(path)[:,::-1], np.array(operation_type)[::-1], np.array(gt_sub_input[::-1]), cm



class CharConfusionMatrix:
    @staticmethod
    def edit_distance(s1: np.ndarray, s2: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Compute Levenshtein edit distance between two sequences s1 and s2.
        Also returns the DP matrix used to compute the distance.
        Returns
        -------
        distance : int
            The Levenshtein edit distance between s1 and s2.
        dp : np.ndarray
            The DP matrix used to compute the distance.
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
        Given the edit distance DP matrix, backtrace to find the optimal path,
        and create a version of input_seq where only substitutions are realized.
        Returns
        -------
        path : np.ndarray
            The optimal alignment path.
        operation_type : np.ndarray
            The types of operations for each step in the path.
        gt_sub_input : np.ndarray
            The ground truth input sequence after applying substitutions.
        cm : np.ndarray
            The confusion matrix for the substitutions.
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


    def ingest_textline_observation(self, pred_line: str, gt_line: str) -> str:
        """
        Ingest a pair of predicted and textlines.
        Updates the confusion matrix with the observations from the edit distance.
        Returns the input line after applying only the non-substitution operations
        thus aligning it to the ground truth.
        -------
        pred_line : str
            The predicted text line.
        gt_line : str
            The ground truth text line.
        Returns
        -------
        str
            The input line after applying only the non-substitution operations.
        """
        dense_pred = self.alphabet.str_to_intlabel_seq(pred_line)
        dense_gt = self.alphabet.str_to_intlabel_seq(gt_line)
        distance, dp = self.edit_distance(dense_pred, dense_gt)
        _, _, gt_sub_input, cm = self.backtrace_ed_matrix(dense_pred, dense_gt, dp)
        self.cm += cm
        return self.alphabet.intlabel_seq_to_str(gt_sub_input)
    

    def save(self, file_path: Union[str, Path]):
        pickle.dump([self.alphabet.dst_alphabet_str, self.cm], open(file_path, "wb"))
    
    @staticmethod
    def load(file_path: Union[str, Path]) -> "CharConfusionMatrix":
        dst_alphabet_str, cm = pickle.load(open(file_path, "rb"))
        lemmatizer = LemmatizerBMP.from_alphabet_mapping(dst_alphabet_str, dst_alphabet_str)
        char_cm = CharConfusionMatrix(lemmatizer)
        char_cm.cm = cm
        return char_cm

    # def edit_distance_with_confusion(
    #     self,
    #     s1: str,
    #     s2: str,
    #     distance_only: bool = False,
    # ) -> Tuple[int, Union[np.ndarray, None], Union[str, None]]:
    #     """
    #     Compute Levenshtein edit distance and a confusion matrix (with a null class).
    #     Also return a version of s1 where we apply only the non-substitution ops from
    #     the optimal path: insertions and deletions are realized, substitutions are NOT
    #     realized (the original s1 symbol is kept).

    #     Returns
    #     -------
    #     distance : int
    #     confusion : np.ndarray (K+1, K+1)
    #     labels : np.ndarray (K+1,)
    #     s1_no_subst : str
    #         s1 after applying only insertions/deletions from the optimal alignment;
    #         substitutions are ignored (keep the s1 symbol).
    #     """
    #     # Validate alphabet uniqueness & contents
    #     #alpha_list = list(alphabet)

    #     # Labels and indices
    #     #labels: List[str] = alpha_list + [null_symbol]
    #     #label_to_idx = {ch: i for i, ch in enumerate(labels)}
    #     null_idx = 0 #label_to_idx[null_symbol]

    #     dense_s1 = self.alphabet.str_to_intlabel_seq(s1)
    #     dense_s2 = self.alphabet.str_to_intlabel_seq(s2)

    #     #n, m = len(s1), len(s2)

    #     # DP + backpointers
    #     dp = np.zeros((dense_s1.size + 1, dense_s2.size + 1), dtype=int)
    #     back = np.empty((dense_s1.size + 1, dense_s2.size + 1), dtype=np.int8)  # 0=start, 1=diag, 2=up(del), 3=left(ins)
    #     back_up_del = 2
    #     back_left_ins = 3
    #     back_diag = 1
    #     back_start = 0


    #     for i in range(1, dense_s1.size + 1):
    #         dp[i, 0] = i
    #         back[i, 0] = back_up_del
    #     for j in range(1, dense_s2.size + 1):
    #         dp[0, j] = j
    #         back[0, j] = back_left_ins
    #     back[0, 0] = back_start

    #     for i in range(1, dense_s1.size + 1):
    #         s1c = dense_s1[i - 1]
    #         for j in range(1, dense_s2.size + 1):
    #             s2c = dense_s2[j - 1]
    #             cost_sub = 0 if s1c == s2c else 1
    #             diag = dp[i - 1, j - 1] + cost_sub
    #             up = dp[i - 1, j] + 1
    #             left = dp[i, j - 1] + 1
    #             best = min(diag, up, left)
    #             dp[i, j] = best
    #             if best == diag:
    #                 back[i, j] = back_diag
    #             elif best == up:
    #                 back[i, j] = back_up_del
    #             else:
    #                 back[i, j] = back_left_ins

    #     distance = int(dp[dense_s1.size, dense_s2.size])
    #     if distance_only:
    #         return distance, None, None

    #     # Backtrace: fill confusion and build the "no-substitution-realization" string
    #     #K = self.alphabet.size
    #     confusion = np.zeros((len(self.alphabet), len(self.alphabet)), dtype=int)

    #     i, j = dense_s1.size, dense_s2.size
    #     sub_only_rev: List[str] = []

    #     dbg = []

    #     while not (i == 0 and j == 0):
    #         move = back[i, j]
    #         if move == 1:  # diag: match or substitution
    #             src = dense_s1[i - 1]
    #             tgt = dense_s2[j - 1]
    #             confusion[src, tgt] += 1
    #             # KEY DIFFERENCE:
    #             # - If it's a match, appending src or tgt is identical.
    #             # - If it's a substitution, we DO NOT realize it; keep the source char.
    #             sub_only_rev.append(src)
    #             dbg.append(f"S: move{move}: src:{'0ACGT'[src]}->tgt:{'0ACGT'[tgt]}")
    #             i -= 1
    #             j -= 1
    #         elif move == 2:  # deletion (src->null): remove src from output
    #             src = dense_s1[i - 1]
    #             confusion[src, null_idx] += 1
    #             # deletion => emit nothing
    #             i -= 1
    #         else:  # insertion (null->tgt): insert tgt into output
    #             tgt = dense_s2[j - 1]
    #             confusion[null_idx, tgt] += 1
    #             sub_only_rev.append(tgt)
    #             j -= 1
    #             dbg.append(f"I: move{move}: src: ->tgt:{'0ACGT'[tgt]}")
    #     sub_only = np.array(sub_only_rev)[::-1]
    #     s1_no_subst =  self.alphabet.intlabel_seq_to_str(sub_only)
    #     print("Debug alignment trace:", file=sys.stderr)
    #     print(f"Pred: {s1}\nGT  : {s2}\nNo S: {s1_no_subst}", file=sys.stderr)
    #     print("\n".join(reversed(dbg)), file=sys.stderr)
    #     return distance, confusion, s1_no_subst


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