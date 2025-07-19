from typing import List, Literal, Optional, Tuple, Union
import torch
from .fast_mapper import LemmatizerBMP
import random
from .abstract_mapper import AbstractLemmatizer, fast_str_to_numpy, fast_numpy_to_str
from collections import defaultdict
from glob import glob
import numpy as np


class Seq2SeqDs:
    @staticmethod
    def __labels_to_onehot(labels: torch.Tensor, max_label: int) -> torch.Tensor:
        """Convert a sequence of labels to one-hot encoded tensor."""
        assert labels.ndim == 1, "Labels must be a 1D tensor."
        onehot = torch.zeros([labels.size(0), max_label + 1], dtype=torch.float32)
        onehot[torch.arange(len(labels)), labels] = 1.0
        return onehot
    
    @staticmethod
    def load_parallel_txt_corpus(input_glob: Union[str, List[str]], output_glob: Union[str, List[str]], check_integrity: Literal["cleanup", "raise", "ignore"] = "cleanup") -> List[Tuple[List[str], List[str]]]:
        if isinstance(input_glob, str):
            input_paths = list(glob(input_glob))
        elif isinstance(input_glob, list):
            input_paths = input_glob
        else:
            raise ValueError("input_glob must be a string or a list of strings")
        
        if isinstance(output_glob, str):
            output_paths = list(glob(output_glob))
        elif isinstance(output_glob, list):
            output_paths = output_glob
        else:
            raise ValueError("output_glob must be a string or a list of strings")

        data = defaultdict(list)
        for fname in input_paths + output_paths:
            k = fname.split("/")[-1].split(".")[0]
            lines = open(fname,"r").read().split("\n")
            lines = [line.strip() for line in lines if len(line.strip())]
            data[k].append(lines)
        
        if check_integrity == "ignore":
            values = list(data.values())
            inputs = [v[0] for v in values]
            outputs = [v[1] for v in values]
            return inputs, outputs

        remove_ids = []
        non_2 = 0
        non_eq = 0
        empty = 0
        for k, v in data.items():
            if len(v) != 2:
                print(f"No two {k}, found {len(v)}")
                non_2 += 1
                remove_ids.append(k)
                continue
            if len(v[0])!=len(v[1]):
                non_eq += 1
                print(f"Not equal lines {k}, found {len(v[0])} and {len(v[1])}")
                remove_ids.append(k)
                continue
            if len(v[0])==0:
                empty += 1
                #print(f"Empty lines {k}")
                remove_ids.append(k)
                continue
        if check_integrity == "raise":
            if non_2 > 0:
                raise ValueError(f"Found {non_2} files with not exactly two lines in the corpus")
            if non_eq > 0:
                raise ValueError(f"Found {non_eq} files with not equal number of lines in the corpus")
            if empty > 0:
                raise ValueError(f"Found {empty} files with empty lines in the corpus")
        elif check_integrity == "cleanup":
            print(f"Found {non_2} files with not exactly two lines in the corpus")
            print(f"Found {non_eq} files with not equal number of lines in the corpus")
            print(f"Found {empty} files with empty lines in the corpus")

            erassing = 0
            for k in remove_ids:
                erassing+= sum([len(v) for v in data[k]])
                del data[k]
            kept_lines = sum([len(v[0]) for v in data.values()])
            print(f"Kept: {kept_lines} double lines, erased {erassing} lines. Non 2 files: {non_2}, Not matching #: {non_eq}, Empty: {empty}")

        values = list(data.values())
        inputs = [v[0] for v in values]
        outputs = [v[1] for v in values]
        return inputs, outputs
    
    @staticmethod
    def from_parallel_txt_corpus(input_glob: Union[str, List[str]], output_glob: Union[str, List[str]], **kwargs) -> 'Seq2SeqDs':
        text_blocks = Seq2SeqDs.load_parallel_txt_corpus(input_glob, output_glob)
        return Seq2SeqDs(text_blocks, **kwargs)
    
    @staticmethod
    def create_selfsupervised_ds(corpus: List[str], mapper: LemmatizerBMP, mapped_is_input: bool = True, **kwargs) -> 'Seq2SeqDs':
        mapped_corpus = [mapper(text) for text in corpus]
        if mapped_is_input:
            text_blocks = (mapped_corpus, corpus)
            input_mapper = LemmatizerBMP.from_alphabet_str(mapper.dst_alphabet_str)
            output_mapper = LemmatizerBMP.from_alphabet_str(mapper.src_alphabet_str)
        else:
            text_blocks = (corpus, mapped_corpus)
            input_mapper = LemmatizerBMP.from_alphabet_str(mapper.src_alphabet_str)
            output_mapper = LemmatizerBMP.from_alphabet_str(mapper.dst_alphabet_str)
        return Seq2SeqDs(text_blocks, input_mapper=input_mapper, output_mapper=output_mapper, **kwargs)

    def __init__(self, text_blocks: Tuple[List[str], List[str]],  input_mapper: LemmatizerBMP, output_mapper: Optional[LemmatizerBMP], 
                 min_input_seqlen: int = 50, min_output_seqlen: int = 50, one2one_mapping: Optional[bool] = None, crop_to_seqlen: Optional[int] = None, 
                 input_is_onehot: bool = False, output_is_onehot: bool = False):
        self.src_text_blocks = []
        self.tgt_text_blocks = []
        
        for n in range(len(text_blocks[0])):
            if len(text_blocks[0][n]) >= min_input_seqlen and len(text_blocks[1][n]) >= min_output_seqlen:
                self.src_text_blocks.append(text_blocks[0][n])
                self.tgt_text_blocks.append(text_blocks[1][n])

        if input_mapper is None:
            self.input_mapper = AbstractLemmatizer.fast_alphabet_extraction(''.join(self.src_text_blocks))
        else:
            self.input_mapper = input_mapper

        if output_mapper is None:
            self.output_mapper = AbstractLemmatizer.fast_alphabet_extraction(''.join(self.tgt_text_blocks))
        else:
            self.output_mapper = output_mapper

        self.min_input_seqlen = min_input_seqlen
        self.max_output_seqlen = min_output_seqlen

        if one2one_mapping is None:
            self.one2one_mapping = all([len(src) == len(tgt) for src, tgt in zip(self.src_text_blocks, self.tgt_text_blocks)])
        elif isinstance(one2one_mapping, bool) and one2one_mapping:
            assert all([len(src) == len(tgt) for src, tgt in zip(self.src_text_blocks, self.tgt_text_blocks)])
            self.one2one_mapping = True
        else:
            self.one2one_mapping = False
        
        if crop_to_seqlen is not None:
            assert self.one2one_mapping, "Cannot crop to seqlen if one2one_mapping is False"
            self.crop_seqlen = crop_to_seqlen
        else:
            self.crop_seqlen = None
        self.input_is_onehot = input_is_onehot
        self.output_is_onehot = output_is_onehot
    
    def __len__(self) -> int:
        return len(self.src_text_blocks)

    def __getitem__(self, n: int) -> Tuple[torch.tensor, torch.tensor]:
        src_txt = self.src_text_blocks[n]
        tgt_txt = self.tgt_text_blocks[n]
        if self.crop_seqlen is not None:
            start_pos = random.randint(0, len(src_txt) - self.crop_seqlen - 1)           
            end_pos = start_pos + self.crop_seqlen
            src_txt = src_txt[start_pos:end_pos]
            tgt_txt = tgt_txt[start_pos:end_pos]
        src_dense_labels = self.input_mapper.str_to_intlabel_seq(src_txt)
        tgt_dense_labels = self.output_mapper.str_to_intlabel_seq(tgt_txt)
        src_dense_labels = torch.tensor(src_dense_labels.astype(np.int64), dtype=torch.int64)
        tgt_dense_labels = torch.tensor(tgt_dense_labels.astype(np.int64), dtype=torch.int64)
        if self.input_is_onehot:
            res_src = self.__labels_to_onehot(src_dense_labels, self.input_mapper.len()-1)
        else:
            res_src = src_dense_labels
        if self.output_is_onehot:
            res_tgt = self.__labels_to_onehot(tgt_dense_labels, self.output_mapper.len()-1)
        else:
            res_tgt = tgt_dense_labels
        return res_src, res_tgt
    
    def shuffle(self) -> None:
        idx = list(range(len(self)))
        random.shuffle(idx)
        self.src_text_blocks = [self.src_text_blocks[i] for i in idx]
        self.tgt_text_blocks = [self.tgt_text_blocks[i] for i in idx]

    def split(self, train_ratio: float = 0.8) -> Tuple['Seq2SeqDs', 'Seq2SeqDs']:
        assert 0 < train_ratio < 1, "Ratio must be between 0 and 1"
        split_idx = int(len(self) * train_ratio)
        train_ds = Seq2SeqDs((self.src_text_blocks[:split_idx], self.tgt_text_blocks[:split_idx]), 
                             input_mapper=self.input_mapper, output_mapper=self.output_mapper,
                             min_input_seqlen=self.min_input_seqlen, min_output_seqlen=self.max_output_seqlen,
                             one2one_mapping=self.one2one_mapping, crop_to_seqlen=self.crop_seqlen)
        val_ds = Seq2SeqDs((self.src_text_blocks[split_idx:], self.tgt_text_blocks[split_idx:]), 
                           input_mapper=self.input_mapper, output_mapper=self.output_mapper,
                           min_input_seqlen=self.min_input_seqlen, min_output_seqlen=self.max_output_seqlen,
                           one2one_mapping=self.one2one_mapping, crop_to_seqlen=self.crop_seqlen)
        return train_ds, val_ds
