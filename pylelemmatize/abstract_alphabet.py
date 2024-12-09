import sys
import time
from typing import Dict, Generator, Set, Tuple, Union
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod
import unicodedata


def fast_str_to_numpy(s: str, dtype=np.uint16) -> np.ndarray:
    if dtype == np.uint16:
        return np.frombuffer(s.encode('utf-16le'), dtype=dtype)
    elif dtype == np.uint32:
        return np.frombuffer(s.encode('utf-32le'), dtype=dtype)
    elif dtype == np.uint64:
        return np.frombuffer(s.encode('utf-64le'), dtype=dtype)
    elif dtype == np.uint8:
        return np.frombuffer(s.encode('utf-8'), dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def fast_numpy_to_str(np_arr: np.ndarray) -> str:
    if np_arr.dtype == np.uint16:
        return np_arr.tobytes().decode('utf-16le')
    elif np_arr.dtype == np.uint32:
        return np_arr.tobytes().decode('utf-32le')
    elif np_arr.dtype == np.uint64:
        return np_arr.tobytes().decode('utf-64le')
    elif np_arr.dtype == np.uint8:
        return np_arr.tobytes().decode('utf-8')
    else:
        raise ValueError(f"Unsupported dtype: {np_arr.dtype}")


def fast_cer(pred: str, true: str) -> float:
    np_pred = fast_str_to_numpy(pred)
    np_true = fast_str_to_numpy(true)
    return np.mean(np_pred != np_true)


class Alphabet(ABC):
    @property
    @abstractmethod
    def src_alphabet_str(self) -> str:
        pass

    @property
    @abstractmethod
    def dst_alphabet_str(self) -> str:
        pass

    @property
    @abstractmethod
    def unknown_chr(self) -> str:
        pass

    @property
    def alphabet_tsv(self) -> str:
        title = " #\tUnicode Number\tUnicode x10\tUnicode x16\tPython String"
        lines = [title]
        for n, c in enumerate(self.unknown_chr + self.dst_alphabet_str):
            lines.append(f"{n}\t{unicodedata.name(c)}\t{ord(c)}\t{ord(c):x}\t{repr(c)}")
        return "\n".join(lines)


class AlphabetBMP(Alphabet):
    @staticmethod
    def fast_alphabet_extraction(text: str) -> str:
        np_text = fast_str_to_numpy(text)
        uniq = np.unique(np_text)
        return fast_numpy_to_str(uniq)

    def __init__(self, sample: Union[str, None] = None, alphabet_str: Union[str, None] = None, unknown_chr: str = '�'):
        assert bool(sample is None) != bool(alphabet_str is None), "Either sample or alphabet_str must be provided, not both"
        if sample is not None:
            self.__src_alphabet_str = ''.join(sorted(set(sample)-set(unknown_chr)))
        else:
            assert len(alphabet_str) == len(set(alphabet_str)), "Alphabet string must not contain duplicates"
            #assert unknown_chr not in alphabet_str, "Alphabet string must not contain unknown character"
            if unknown_chr in alphabet_str:
                alphabet_str = alphabet_str.replace(unknown_chr, "")
            self.__src_alphabet_str = alphabet_str
        self.__unknown_chr = unknown_chr
        self.__chr2chr, self.__npint2chr, self.__int2chr, self.__chr2int, self.__npint2int = self.__create_mappers()

    def __create_mappers(self) -> Tuple[defaultdict, np.ndarray, Dict[int, str], Dict[str, int], np.ndarray]:
        chr2chr = defaultdict(lambda: self.__unknown_chr)
        chr2chr.update({a: a for a in self.__src_alphabet_str})
        full_str = self.__unknown_chr + self.__src_alphabet_str
        np_int2chr = np.array(list(full_str))
        int2chr = {i: a for i, a in enumerate(full_str)}
        chr2int = {a: i for i, a in int2chr.items()}

        np_int2int = np.zeros(256**2, dtype=np.uint16)
        for c in full_str:
            np_int2int[ord(c)] = ord(c)
        return chr2chr, np_int2chr, int2chr, chr2int, np_int2int

    @property
    def src_alphabet_str(self):
        return self.__src_alphabet_str

    @property
    def dst_alphabet_str(self):
        return self.__src_alphabet_str

    @property
    def unknown_chr(self):
        return self.__unknown_chr

    def __len__(self):
        return len(self.__src_alphabet_str)

    def __repr__(self):
        if self.__unknown_chr != '�':
            return f"Alphabet(alphabetstr={repr(self.__src_alphabet_str)}, unknown_chr={repr(self.__unknown_chr)})"
        return f"Alphabet(alphabetstr={repr(self.__src_alphabet_str)})"

    def __call__(self, text: str) -> str:
        #    return ''.join([self.__chr2chr[c] for c in text])
        #  Vectorized version is 10 x faster on large strings
        return fast_numpy_to_str(self.__npint2int[fast_str_to_numpy(text)])

    def __str__(self):
        return self.__src_alphabet_str

    def get_unigram(self, text: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        # adding all characters atleast once to make np.unique count zero counts
        np_text = fast_str_to_numpy(self.unknown_chr+self.__src_alphabet_str+text)
        mapped_np_text = self.__npint2int[np_text]
        values, counts = np.unique(mapped_np_text, return_counts=True)
        counts = counts - 1  # removing the counts of the added characters
        return values, counts, self.__int2chr.copy()

    def get_cer(self, pred: str, true: str) -> float:
        np_pred = fast_str_to_numpy(pred)
        np_true = fast_str_to_numpy(true)
        mapped_np_pred = self.__npint2int[np_pred]
        mapped_np_true = self.__npint2int[np_true]
        return np.mean(mapped_np_pred != mapped_np_true)

    def get_encoding_information_loss(self, text: str) -> float:
        np_text = fast_str_to_numpy(text)
        mapped_np_text = self.__npint2int[np_text]
        return np.mean(np_text != mapped_np_text)


def main_alphabet_extract_corpus_alphabet():
    import fargv, glob, time
    from .util import generate_corpus

    t = time.time()
    p = {
        "corpus_glob": "",
        "corpus_files": set([]),
        "dont_count_alphabet": False,
        "dont_show_alphabet": False,
        "dont_show_histogram": False,
        "verbose": False,
        "strip_xml": True,
        "all_is_xml": False,
        "output_tsv": ""
    }
    args, _ = fargv.fargv(p)

    if args.corpus_glob != "":
        glob_files = set(glob.glob(args.corpus_glob))
    else:
        glob_files = set([])
    if len(args.corpus_files) > 0:
        corpus_files = args.corpus_files
    else:
        corpus_files = set([])

    total_size = 0
    all_alphabets_strs = []
    all_corpus_strs = []
    for file_contnets in generate_corpus(glob_files | corpus_files, verbose=args.verbose,
                                         strip_xml=args.strip_xml, treat_all_file_as_xml=args.all_is_xml):
        total_size += len(file_contnets)
        all_corpus_strs.append(file_contnets)
        all_alphabets_strs.append(AlphabetBMP.fast_alphabet_extraction(file_contnets))
    found_alphabet_str = ''.join(sorted(set(''.join(all_alphabets_strs))))

    if not args.dont_show_alphabet:
        if args.output_tsv == "stdout":
            print(found_alphabet_str, file=sys.stderr)
        else:
            print(found_alphabet_str)

    if args.output_tsv != "":
        tsv_str = AlphabetBMP(alphabet_str=found_alphabet_str).alphabet_tsv
        if args.output_tsv == "stdout":
            print(tsv_str)
        elif args.output_tsv == "stderr":
            print(tsv_str, file=sys.stderr)
        else:
            with open(args.output_tsv, "w") as f:
                f.write(tsv_str)

    if not args.dont_count_alphabet:
        print(f"Alphabet Length: {len(found_alphabet_str)}", file=sys.stderr)

    if not args.dont_show_histogram:
        ht = time.time()
        corpus_str = ''.join(all_corpus_strs)
        nums, freqs, names = AlphabetBMP(alphabet_str=found_alphabet_str).get_unigram(corpus_str)
        most_frequent = np.argsort(freqs)
        print(f"\nUnigram model in reverced frequencies:", file=sys.stderr)
        for n in most_frequent:
            print(f"{names[n]}: {freqs[n]}", file=sys.stderr)
        print(f"Computed Histogram for {len(corpus_str)} in {time.time() - ht :.2f}", file=sys.stderr)

    if args.verbose:
        print(f"Computed {total_size} in {time.time() -t :.2f}", file=sys.stderr)


def main_alphabet_evaluate_merges():
    import fargv, glob, time
    from .util import generate_corpus

    t = time.time()
    p = {
        "corpus_glob": "",
        "corpus_files": set([]),
        "merges": "[('u', 'v'),  ('U', 'V')]",
        "verbose": False,
        "strip_xml": True,
        "all_is_xml": False,
        "output_tsv": ""
    }
    args, _ = fargv.fargv(p)

    if args.corpus_glob != "":
        glob_files = set(glob.glob(args.corpus_glob))
    else:
        glob_files = set([])
    if len(args.corpus_files) > 0:
        corpus_files = args.corpus_files
    else:
        corpus_files = set([])

    total_size = 0
    all_alphabets_strs = []
    all_corpus_strs = []
    for file_contnets in generate_corpus(glob_files | corpus_files, verbose=args.verbose,
                                         strip_xml=args.strip_xml, treat_all_file_as_xml=args.all_is_xml):
        total_size += len(file_contnets)
        all_corpus_strs.append(file_contnets)
        all_alphabets_strs.append(AlphabetBMP.fast_alphabet_extraction(file_contnets))
    found_alphabet_str = ''.join(sorted(set(''.join(all_alphabets_strs))))
    corpus_str = ''.join(all_corpus_strs)

    mapper = {k: k for k in found_alphabet_str}
    mapper.update(eval(args.merges))
    mapped_corpus_str = ''.join([mapper[c] for c in corpus_str])
    print(f"Mapping CER {fast_cer(corpus_str, mapped_corpus_str)}", file=sys.stderr)

    if args.verbose:
        print(f"Computed {total_size} in {time.time() -t :.2f}", file=sys.stderr)






class Lematizer(AlphabetBMP):
    def __init__(self, src_alphabet: str, mapping_dict: Dict[str, str], unknown_chr: str = '�'):
        super().__init__(alphabet_str=src_alphabet, unknown_chr=unknown_chr)
        self.__custom_mapping_dict = mapping_dict
        self.__dst_alphabet_str = ''.join(sorted(set(self.__custom_mapping_dict.values())-set(unknown_chr)))
        self.__chr2chr, self.__npint2chr, self.__int2chr, self.__chr2int, self.__npint2int = self.__create_mappers()

    def __create_mappers(self) -> Tuple[defaultdict, np.ndarray, Dict[int, str], Dict[str, int], np.ndarray]:
        chr2chr = defaultdict(lambda: self.__unknown_chr)
        chr2chr.update({a: a for a in self.__src_alphabet_str})
        chr2chr.update(self.__custom_mapping_dict)

        dst_full_str = self.__unknown_chr + self.__dst_alphabet_str
        np_int2chr = np.array(list(dst_full_str))
        int2chr = {i: a for i, a in enumerate(full_str)}
        chr2int = {a: i for i, a in int2chr.items()}

        np_int2int = np.zeros(256**2, dtype=np.uint16)
        for c in full_str:
            np_int2int[ord(c)] = ord(c)
        return chr2chr, np_int2chr, int2chr, chr2int, np_int2int

    def repr(self):
        if self.__unknown_chr != '�':
            unknown_repr = f", unknown_chr={repr(self.unknown_chr)}"
        else:
            unknown_repr = ""
        if self.__custom_mapping_dict != {}:
            mapping_repr = f", mapping_dict={repr(self.__custom_mapping_dict)}"
        else:
            mapping_repr = ""
        return f"Lematizer(src_alphabet={repr(self.src_alphabet_str)}{mapping_repr}{unknown_repr})"

