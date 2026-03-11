from abc import ABC, abstractmethod, abstractmethod
from collections import defaultdict
import glob
import random
import sys
import time
from typing import Any, Dict, Literal, Optional, Union, Tuple, List
import re
import unicodedata
from pathlib import Path
import numpy as np

import torch
from torch import Tensor
from tqdm import tqdm

from pylelemmatize.demapper_lstm import DemapperLSTM
from pylelemmatize.fast_mapper import LemmatizerBMP, fast_numpy_to_str, fast_str_to_numpy
from pylelemmatize.mapper_ds import Seq2SeqDs
from .abstract_mapper import AbstractLemmatizer, fast_str_to_numpy
from lxml import etree
from .util import banded_edit_path, compute_cer

### In all contexts where we talk about insertions and deletions occuring in two strings A and B:
# - Insertion means inserting a character into A to obtain B (a gap in A) We insert in B to match A
# - Deletion means deleting a character from A to obtain B (a gap in B) We delete in B to match A
# The code should be read with this convention in mind. And variable namings like is_ins, is_del, etc. 
# MUST reflect this convention.
# This also applies to unitests about these fille


def align_sub_strings(a: str, b: str, band:int = 80) -> List[Tuple[str, str]]:
    np_a = fast_str_to_numpy(a)
    np_b = fast_str_to_numpy(b)
    paths, _, (is_match, is_ins, is_del, is_sub) = banded_edit_path(np_a, np_b, band=band)
    p = paths[1:, :].tolist()
    res = []
    n = 0
    while n < len(p):
        if is_ins[n]:
            start_a, start_b = p[n]
            start_a, start_b = max(start_a, 0), max(start_b, 0)
            while n < len(p) and is_ins[n]:
                n += 1
            if n == len(p):
                end_a, end_b = len(a), len(b)
            else:
                end_a, end_b = p[n]
                end_a, end_b = max(end_a, 0), max(end_b, 0)
            res.append((a[start_a: start_a], b[start_b: end_b]))
        elif is_match[n] or is_sub[n]:
            a_pos, b_pos = p[n] 
            res.append((a[a_pos: a_pos+1], b[b_pos: b_pos+1]))
            n+=1
        elif is_del[n]:
            a_pos, b_pos = p[n]
            res.append((a[a_pos: a_pos+1], b[b_pos: b_pos]))
            n+=1
        else:
            raise RuntimeError("Unknown cell type")

    merged_insertions = []
    prev_insertion = ""
    for a_substr, b_substr in res:
        if prev_insertion == "" and a_substr != "": # match / substitution / deletion no pending insertion
            merged_insertions.append((a_substr, b_substr))
        elif a_substr == "": # insertion
            prev_insertion += b_substr
        elif prev_insertion != "" and len(merged_insertions) == 0 and a_substr != "": # Starting with insertion(s) then match/sub
            merged_insertions.append((a_substr, prev_insertion + b_substr))
            prev_insertion = ""
        elif prev_insertion != "" and len(merged_insertions) > 0 and a_substr != "": # pending insertion then match/sub
            merged_insertions[-1] = (merged_insertions[-1][0], merged_insertions[-1][1] + prev_insertion)
            merged_insertions.append((a_substr, b_substr))
            prev_insertion = ""
        else:
            raise RuntimeError("Impossible state")
    if prev_insertion != "":
        merged_insertions[-1] = (merged_insertions[-1][0], merged_insertions[-1][1] + prev_insertion)
    return merged_insertions


class ManyToMoreDS():
    @staticmethod
    def create_from_aligned_textlines(line_pairs: List[Tuple[str, str]], min_src_len: int=10,
                          max_src_len: int=1000, min_tgt_len: int=10, max_tgt_len: int=1000, band: int = 70,
                          src_alphabet: Optional[str]=None, tgt_alphabet: Optional[str]=None,
                          onehot_input: bool = False, onehot_output: bool = False, allow_start_insertions: bool=False,
                          verbose: bool=False) -> List[Tuple[str, str]]:
        textlines_appropriate_sizes = []
        for inp, out in line_pairs:
            if not (min_src_len <= len(inp) <= max_src_len):
                continue
            if not (min_tgt_len <= len(out) <= max_tgt_len):
                continue
            textlines_appropriate_sizes.append((inp, out))
        inputs, outputs = zip(*textlines_appropriate_sizes)
        if src_alphabet is None:
            src_alphabet = ''.join(sorted(set(''.join(inputs))))
        if tgt_alphabet is None:
            tgt_alphabet = ''.join(sorted(set(''.join(outputs))))
        alphabet_str = ''.join(sorted(set(src_alphabet + tgt_alphabet)))
        textlines_appropriate_alphabets = []
        aligned_substrings: List[List[Tuple[str, str]]] = []
        for inp, out in textlines_appropriate_sizes:
            if set(inp).issubset(src_alphabet) and set(out).issubset(tgt_alphabet):
                textlines_appropriate_alphabets.append((inp, out))
        if verbose:
            print(f"Read {len(line_pairs)} line pairs")
            print(f"After size filtering: {len(textlines_appropriate_sizes)} line pairs because of length constraints ({min_src_len}-{max_src_len} for source, {min_tgt_len}-{max_tgt_len} for target).", file=sys.stderr)
            print(f"After alphabet filtering: {len(textlines_appropriate_alphabets)} line pairs because of alphabet constraints.", file=sys.stderr)
            #print(f"Filtered {len(textlines_appropriate_sizes)} line pairs to {len(textlines_appropriate_alphabets)} line pairs because of length and alphabet constraints.", file=sys.stderr)
        if verbose:
            progress = tqdm(total=len(textlines_appropriate_alphabets), desc="Processing line pairs")
        for inp, out in textlines_appropriate_alphabets:
            if verbose:
                progress.update(1)
            try:
                aligned_in_out = align_sub_strings(inp, out, band=band)
            except ValueError:
                if verbose:
                    print(f"Skipping pair due to alignment failure within band {band}: '{inp}' -> '{out}'", file=sys.stderr)
                continue
            if not allow_start_insertions and aligned_in_out[0][0][0]=="":
                continue
            aligned_substrings.append(aligned_in_out)
        if verbose:
            progress.close()
        src_alphabet = ''.join(sorted(src_alphabet))
        tgt_alphabet = ''.join(sorted(tgt_alphabet))
        res = ManyToMoreDS(aligned_substrings=aligned_substrings, src_alphabet=alphabet_str, tgt_alphabet=alphabet_str, 
                           onehot_input=onehot_input, onehot_output=onehot_output)
        if verbose:
            print(f"Created ManyToMoreDS with {len(res)} samples. aligned_substrings: {len(aligned_substrings)}", file=sys.stderr)
        return res

    @staticmethod
    def create_correction_from_icdar19comp_data(filelist: Union[List[str], str], min_src_len: int=40,
                          max_src_len: int=5000, min_tgt_len: int=10, max_tgt_len: int=5000, band: int = 70,
                          alphabet: Optional[str]=None,
                          onehot_input: bool = False, onehot_output: bool = False, allow_start_insertions: bool=False,
                          max_ins_length: int = -1, verbose: bool=False, forbiten_characters: Optional[set]="=¥õ✓", random_crop_length: int = -1) -> List[Tuple[str, str]]:
        """
        2019 ICDAR Post-OCR Text Correction Competition dataset contains aligned OCR and GT textlines in a format where each line contains three parts: 
            the original input, the OCR-aligned string, and the GT-aligned string. The aligned strings use "@" to indicate gaps (insertions or deletions) 
            in the alignment. This function processes these aligned textlines to create a ManyToMoreDS dataset suitable for training correction models.
        
        Validation set CERs on the original aligned textlines (without splitting long insertions) for each language are as follows:
            'SL': (0.19628795782768657, 17873, 91055.0),
            'FI': (0.19785019016308902, 12277, 62052),
            'FR': (0.0906146358156801, 67786, 748069.0),
            'BG': (0.21622334724031594, 39336, 181923.0),
            'CZ': (0.09090669408631917, 6896, 75858),
            'DE': (0.27270782853211284, 182327, 668580.0),
            'ES': (0.4484517444320498, 58292, 129985),
            'NL': (0.32006913806604403, 67034, 209436),
            'PL': (0.26900244972369397, 23609, 87765),
            'EN': (0.19785019016308902, 12277, 62052)
        
        Parameters:
        -----------
        filelist: Union[List[str], str] 
            A list of file paths or a glob pattern string to read the aligned textlines from.
        min_src_len: int
            Minimum length of the source (OCR) textline to be included in the dataset.

        """
        if isinstance(filelist, str):
            filelist = glob.glob(filelist)
            assert len(filelist) > 0, f"No files found for pattern: {filelist}"
        textlines_appropriate_sizes = []
        all_texts = []
        aligned_input_outpus = []
        ins_lens = []
        del_lens = []
        rejected = 0
        for n, file in enumerate(filelist):
            with open(file, 'r', encoding='utf-8') as f:
                all_text = f.read().strip()
                for c in forbiten_characters:
                    all_text = all_text.replace(c, "")
                try:
                    original_input, ocr_aligned, gt_aligned = all_text.splitlines()
                except ValueError:
                    print(f"Skipping file {file} due to unexpected format. Expected 3 lines but got {len(all_text.splitlines())}. Content:\n{all_text}", file=sys.stderr)
                    continue
                original_input = original_input.split("[OCR_toInput]")[1]
                ocr_aligned = ocr_aligned.split("[OCR_aligned]")[1]
                gt_aligned = gt_aligned.split("[ GS_aligned]")[1]
            if ocr_aligned.replace("@","") != original_input:
                print(f"{rejected} from {n} Warning: OCR aligned string is different to original input after removing '@' characters in file {file}. ", file=sys.stderr)
                #print(f"{repr(ocr_aligned.replace('@',''))}\n{repr(original_input)}\n", file=sys.stderr)
                rejected += 1
                continue
            
            while len(gt_aligned) > 0 and gt_aligned[0] == "@":
                gt_aligned = gt_aligned[1:]
                ocr_aligned = ocr_aligned[1:]
            while len(gt_aligned) > 0 and gt_aligned[-1] == "@":
                gt_aligned = gt_aligned[:-1]
                ocr_aligned = ocr_aligned[:-1]
            aligned_input_outpus.append((ocr_aligned, gt_aligned))


        if verbose:
            print(f"Read {len(aligned_input_outpus)} aligned input-output pairs from {len(filelist)} files.", file=sys.stderr)
            print(f"Max insertion length in OCR aligned: {max(ins_lens) if ins_lens else 0}, Max deletion length in GT aligned: {max(del_lens) if del_lens else 0}", file=sys.stderr)
            print(f"Rejected {rejected} files due to OCR alignment issues.", file=sys.stderr)

        if max_ins_length > 0:
            ocr_aligned, gt_aligned = zip(*aligned_input_outpus)
            ocr_str = '\n'.join(ocr_aligned)
            gt_str = '\n'.join(gt_aligned)
            assert len(ocr_str) == len(gt_str)
            pieces = []
            last_end = 0

            for long_ins in re.finditer("@"* (max_ins_length -1) + "@+", gt_str):
                start, end = long_ins.span()
                pieces.append((ocr_str[last_end:start], gt_str[last_end:start]))
                last_end = end
            pieces.append((ocr_str[last_end:], gt_str[last_end:]))
            pieces = [p for p in pieces if len(p[0]) > min_src_len and len(p[1]) > min_tgt_len]
            
            ocr_str = '\n'.join([p[0] for p in pieces])
            gt_str = '\n'.join([p[1] for p in pieces])

            assert len(ocr_str) == len(gt_str)
            aligned_input_outpus = list(zip(ocr_str.splitlines(), gt_str.splitlines()))
        
        ocr_str = '\n'.join([pair[0] for pair in aligned_input_outpus])
        gt_str = '\n'.join([pair[1] for pair in aligned_input_outpus])

        occured_chars = set(ocr_str + gt_str) - set("@")

        if alphabet is None:
            alphabet = ''.join(sorted(occured_chars))
        else:
            assert set(alphabet).issuperset(occured_chars), f"Provided alphabet '{alphabet}' does not cover all characters in the data"
            assert "@" not in alphabet, "Alphabet cannot contain '@' character as it is used for alignment gaps"

        aligned_lines: List[List[Tuple[str, str]]] = []
        for ocr_aligned, gt_aligned in aligned_input_outpus:
            if len(ocr_aligned.replace("@", "")) == 0 or len(gt_aligned.replace("@", "")) == 0:
                continue
            if not allow_start_insertions and ocr_aligned[0] == "@":
                while ocr_aligned[0] == "@":
                    ocr_aligned = ocr_aligned[1:]
                    gt_aligned = gt_aligned[1:]
            aligned_substrings = [[ocr_aligned[0].replace("@", ""), gt_aligned[0].replace("@", "")]]
            pending_outputs = ""
            
            for n in range(1, len(ocr_aligned)):
                if ocr_aligned[n] == "@": # insertion in input
                    pending_outputs += gt_aligned[n]
                else:
                    if pending_outputs != "":
                        aligned_substrings[-1] = [aligned_substrings[-1][0], aligned_substrings[-1][1] + pending_outputs]
                        pending_outputs = ""
                    try:
                        aligned_substrings.append([ocr_aligned[n], gt_aligned[n]])
                    except IndexError:
                        print(f"Aligned substring:\n{repr(ocr_aligned)}\n{repr(gt_aligned)} Failing at '{n}'")
                        raise IndexError(f"Aligned substring:\n{repr(ocr_aligned)}\n{repr(gt_aligned)} Failing at '{n}'")
                    
            aligned_lines.append(aligned_substrings)
        breakpoint()
        print(f"Created {len(aligned_lines)} aligned line pairs after processing insertions. random_crop_length={random_crop_length}", file=sys.stderr)
        res = ManyToMoreDS(aligned_substrings=aligned_lines, src_alphabet=alphabet, tgt_alphabet=alphabet, 
                           onehot_input=onehot_input, onehot_output=onehot_output, random_crop_length=random_crop_length)
        return res


    def __init__(self, src_alphabet: str, tgt_alphabet: str,
                 aligned_substrings: List[List[Tuple[str, str]]],
                 onehot_input: bool=False, onehot_output: bool=False,
                 return_torch: bool=True, return_ctc: bool=False, max_unalignemet: int = 5, random_crop_length: int = -1):
        self.onehot_input = onehot_input
        self.onehot_output = onehot_output
        self.aligned_substrings = aligned_substrings
        self.input_mapper = LemmatizerBMP(src_alphabet)
        self.output_mapper = LemmatizerBMP(tgt_alphabet)
        self.return_torch = return_torch
        self.return_ctc = return_ctc
        self.max_unalignemet = max_unalignemet
        self.random_crop_length = random_crop_length
        assert max_unalignemet < self.get_max_dst_len(), "max_unalignemet must be less than maximum target length"
    
    def get_max_dst_len(self) -> int:
        max_len = 0
        for aligned_substrings in self.aligned_substrings:
            tgt_len = sum([len(part[1]) for part in aligned_substrings])
            if tgt_len > max_len:
                max_len = tgt_len
        return max_len
    
    def __len__(self):
        return len(self.aligned_substrings)

    def __getitem__(self, idx: int, return_info: bool = False) -> Tuple[str, str]:
        aligned_substrings = self.aligned_substrings[idx]
        src_parts, tgt_parts = zip(*aligned_substrings)

        # src is a list contianing single sequence of all input characters concatenated
        # tgt is a list containing [every target substring as a separate sequence]
        src: List[np.ndarray] = [self.input_mapper.str_to_intlabel_seq(c) for c in src_parts]
        src = [np.concatenate(src, axis=0)]
        tgt: List[np.ndarray] = [self.output_mapper.str_to_intlabel_seq(c) for c in tgt_parts]  

        if self.onehot_input:
            raise NotImplementedError("onehot_input is not implemented yet")

        if self.return_torch:            
            #print(f"input_lengths = {set([len(x) for x in src])}")
            #print(f"output_lengths = {set([len(x) for x in tgt])}")
            #src = [Tensor(seq).long().reshape(-1) for seq in src]
            src = [Tensor(seq.astype(np.int64)).long().reshape(-1) for seq in src]
            tgt = [Tensor(t.astype(np.int64)).long() for t in tgt]
        else:
            src = np.array(src, dtype=np.int16).reshape(-1)
            tgt = [np.array(t, dtype=np.int16) for t in tgt]
        
        src = src[0]
        if len(tgt) > self.random_crop_length and self.random_crop_length > 0:
            start_idx = random.randint(0, len(tgt) - self.random_crop_length)
            tgt = tgt[start_idx: start_idx + self.random_crop_length]
            src = src[start_idx: start_idx + self.random_crop_length]
        if not return_info:
            return src, tgt
        else:
            info = f"Sample {idx}:\nAligned substrings: {aligned_substrings}\nSource sequence (int labels): {src}\nTarget sequence (list of int label sequences): {torch.cat(tgt)}"
            return src, tgt, info
    
    def get_cer(self, idx: int=-1) -> Tuple[int, int]:
        if idx < 0:
            correct_chars = 0
            total_chars = 0
            for idx in range(len(self)):
                correct, total = self.get_cer(idx)
                correct_chars += correct
                total_chars += total
            return correct_chars, total_chars
        else:
            inp_out = self.aligned_substrings[idx]
            inp = ''.join([part[0] for part in inp_out])
            out = ''.join([part[1] for part in inp_out])
            errors = compute_cer(inp, out, normalize=False, band=-1)
            return int(errors), int(len(out))
    
    def save(self, dataset_path: str):
        data = {'aligned_substrings': self.aligned_substrings,
                'onehot_input': self.onehot_input,
                'onehot_output': self.onehot_output,
                'src_alphabet': self.input_mapper.src_alphabet_str,
                'tgt_alphabet': self.output_mapper.src_alphabet_str,
                'return_torch': self.return_torch,
                'return_ctc': self.return_ctc,
                'random_crop_length': self.random_crop_length
                }
        torch.save(data, dataset_path)
    
    @classmethod
    def load(cls, dataset_path: str, allow_start_insertions=True) -> Optional["ManyToMoreDS"]:
        try:
            data = torch.load(dataset_path)
            if allow_start_insertions:
                aligned_fragmenst = []
                for aligned_substrings in data['aligned_substrings']:
                    if aligned_substrings[0][0] == "":
                        continue
                    aligned_fragmenst.append(aligned_substrings)
                data['aligned_substrings'] = aligned_fragmenst
            dataset = ManyToMoreDS(
                src_alphabet=data['src_alphabet'],
                tgt_alphabet=data['tgt_alphabet'],
                aligned_substrings=data['aligned_substrings'],
                onehot_input=data['onehot_input'],
                onehot_output=data['onehot_output'],
                return_torch=data['return_torch'],
                return_ctc=data['return_ctc']
            )
            return dataset
        except FileNotFoundError:
            return None

    def shuffle(self) -> None:
        idx = list(range(len(self)))
        random.shuffle(idx)
        self.aligned_substrings = [self.aligned_substrings[i] for i in idx]

    def split(self, train_ratio: float = 0.8, shuffle: bool = True) -> Tuple['Seq2SeqDs', 'Seq2SeqDs']:
        assert 0 < train_ratio < 1, "Ratio must be between 0 and 1"
        if shuffle:
            self.shuffle()
        split_idx = int(len(self) * train_ratio)
        train_ds: ManyToMoreDS = ManyToMoreDS(
            src_alphabet=self.input_mapper.src_alphabet_str,
            tgt_alphabet=self.output_mapper.src_alphabet_str,
            aligned_substrings=self.aligned_substrings[:split_idx],
            onehot_input=self.onehot_input,
            onehot_output=self.onehot_output,
            return_torch=self.return_torch,
            return_ctc=self.return_ctc)
        val_ds: ManyToMoreDS = ManyToMoreDS(
            src_alphabet=self.input_mapper.src_alphabet_str,
            tgt_alphabet=self.output_mapper.src_alphabet_str,
            aligned_substrings=self.aligned_substrings[split_idx:],
            onehot_input=self.onehot_input,
            onehot_output=self.onehot_output,
            return_torch=self.return_torch,
            return_ctc=self.return_ctc)
        return train_ds, val_ds


class ManyToMoreCollator(ABC):
    @abstractmethod
    def run_srcs(self, batch: List[Tensor]) -> Tensor:
        pass

    @abstractmethod
    def run_srcs_and_tgts(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        pass

    def __call__(self, batch: Union[List[Tuple[Tensor, Tensor]], List[Tensor]], separate_output: Optional[List[Tensor]] = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        Collates a batch of ManyToMore data samples to batch tensors.
        Can be passed as a collation function to a PyTorch DataLoader.
        Essentially launches either `run_srcs_and_tgts` or `run_srcs` depending on the input.

        Parameters
        ----------
        batch : Union[List[Tuple[Tensor, Tensor]], List[Tensor]]
            Either a list of tuples (source, target) or a list of sources only.
        separate_output : Optional[List[Tensor]], optional
            Allows passing aligned input and output batches separately.

        Returns
        -------
        Union[Tuple[Tensor, Tensor], Tensor]
            If outputs were passed separately or `batch` is a list of tuples, returns a tuple 
            (sources, targets). Otherwise, returns sources only.
        """
        if separate_output is not None:
            assert isinstance(batch, list) and isinstance(batch[0], Tensor)
            batch = [tuple(src_tgt) for src_tgt in zip(batch, separate_output)]
        if isinstance(batch, list) and isinstance(batch[0], tuple) and len(batch[0]) == 2:  # sources ant targets
            return self.run_srcs_and_tgts(batch)  # type: ignore
        elif isinstance(batch[0], Tensor):  # sources only
            return self.run_srcs_only(batch)  # type: ignore
        else:
            raise ValueError(f"Batch must be a list of tuples or a list of tensors but got {type(batch)} with elements of type {type(batch[0])} batch:{repr(batch)}")


class ManyToMoreCollatorSeq2Seq2(ManyToMoreCollator):
    """
    A collation function for PyTorch DataLoader that processes batches of sequences 
    for sequence-to-sequence tasks. This collator is designed to handle cases where 
    the target sequences may have varying lengths, and it pads them to a uniform length.

    Attributes:
        max_unalignment (int): The maximum allowed length for target sequences. If set 
            to a negative value, the maximum length is determined dynamically based on 
            the longest target sequence in the batch.

    Methods:
        __call__(batch):
            Processes a batch of source and target sequences, ensuring that the target 
            sequences are padded to a uniform length. Returns the processed source and 
            target tensors.

    Args:
        max_unalignment (int, optional): The maximum allowed length for target sequences. 
            If set to a negative value (default: -1), the maximum length is determined 
            dynamically based on the longest target sequence in the batch.

    Raises:
        RuntimeError: If the length of any target sequence exceeds the specified 
            `max_unalignment`.

    Example:
        >>> collator = ManyToMoreCollatorSeq2Seq(max_unalignment=10)
        >>> srcs = [Tensor([1, 2, 3])]
        >>> tgts = [Tensor([4, 5])]
        >>> batch = [(srcs[0], tgts[0])]
        >>> src_tensor, tgt_tensor = collator(batch)
        >>> print(src_tensor.shape)  # torch.Size([1, 3])
        >>> print(tgt_tensor.shape)  # torch.Size([1, 11])
    """
    def __init__(self, max_unalignment: int = -1):
        self.max_unalignment = max_unalignment
    
    def run_srcs_and_tgts(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        assert len(batch) == 1
        srcs, tgts = zip(*batch)
        assert len(srcs) == 1 and len(tgts) == 1    
        srcs, tgts = srcs[0], tgts[0]
        if self.max_unalignment < 0:
            max_unalignment = max([len(tgt) for tgt in tgts])
        else:
            max_unalignment = self.max_unalignment
        tgts_tensor = torch.zeros((len(srcs), max_unalignment + 1), dtype=torch.long)
        for i, tgt in enumerate(tgts):
            if len(tgt) > max_unalignment:
                raise RuntimeError(f"Target length {len(tgt)} exceeds max_unalignment {max_unalignment}")
            tgts_tensor[i, :len(tgt)] = tgt
        return srcs.unsqueeze(0), tgts_tensor

    def __init__(self, src_alphabet: str, tgt_alphabet: str,
                 aligned_substrings: List[List[Tuple[str, str]]],
                 onehot_input: bool=False, onehot_output: bool=False,
                 return_torch: bool=True, return_ctc: bool=False, max_unalignemet: int = 5):
        self.onehot_input = onehot_input
        self.onehot_output = onehot_output
        self.aligned_substrings = aligned_substrings
        self.input_mapper = LemmatizerBMP(src_alphabet)
        self.output_mapper = LemmatizerBMP(tgt_alphabet)
        self.return_torch = return_torch
        self.return_ctc = return_ctc
        self.max_unalignemet = max_unalignemet
        assert max_unalignemet < self.get_max_dst_len(), "max_unalignemet must be less than maximum target length"
    
    def get_max_dst_len(self) -> int:
        max_len = 0
        for aligned_substrings in self.aligned_substrings:
            tgt_len = sum([len(part[1]) for part in aligned_substrings])
            if tgt_len > max_len:
                max_len = tgt_len
        return max_len
    
    def __len__(self):
        return len(self.aligned_substrings)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        aligned_substrings = self.aligned_substrings[idx]
        src_parts, tgt_parts = zip(*aligned_substrings)

        # src is a list contianing single sequence of all input characters concatenated
        # tgt is a list containing [every target substring as a separate sequence]
        src: List[np.ndarray] = [self.input_mapper.str_to_intlabel_seq(c) for c in src_parts]
        src = [np.concatenate(src, axis=0)]
        tgt: List[np.ndarray] = [self.output_mapper.str_to_intlabel_seq(c) for c in tgt_parts]  

        if self.onehot_input:
            raise NotImplementedError("onehot_input is not implemented yet")

        if self.return_torch:            
            #print(f"input_lengths = {set([len(x) for x in src])}")
            #print(f"output_lengths = {set([len(x) for x in tgt])}")
            #src = [Tensor(seq).long().reshape(-1) for seq in src]
            src = [Tensor(seq.astype(np.int64)).long().reshape(-1) for seq in src]


    def run_srcs(self, batch: List[Tensor]) -> Tensor:
        assert len(batch) == 1 and isinstance(batch[0], Tensor)
        return batch[0].unsqueeze(0)


