from collections import defaultdict
from typing import Dict, Optional, Tuple, Literal
import numpy as np
from unidecode import unidecode
from .abstract_mapper import AbstractLemmatizer, GenericLemmatizer, fast_numpy_to_str, fast_str_to_numpy


class LemmatizerBMP(GenericLemmatizer):
    @staticmethod
    def alphabet_in_bmp(alphabet: Optional[str]) -> bool:
        if alphabet is None:
            return True
        return (fast_str_to_numpy(alphabet) > 65535).sum() == 0

    @staticmethod
    def __create_mappers(mapping_dict, unknown_chr) -> Tuple[defaultdict, np.ndarray, Dict[int, str], Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
        assert all([ord(c)< 65536 for c in mapping_dict.keys()]), "All keys in mapping_dict must be BMP characters."
        assert all([ord(c)< 65536 for c in mapping_dict.values()]), "All values in mapping_dict must be BMP characters."
        assert ord(unknown_chr) < 65536, "unknown_chr must be a BMP character."
        if unknown_chr in mapping_dict:
            assert mapping_dict[unknown_chr] == unknown_chr, "unknown_chr must map to itself in the mapping_dict."
            del mapping_dict[unknown_chr]  # Remove the unknown character from the mapping to avoid confusion

        src_alphabet_str = ''.join(sorted(mapping_dict.keys()))
        dst_alphabet_str = ''.join(sorted(set(mapping_dict.values())))

        chr2chr = defaultdict(lambda: unknown_chr)
        chr2chr.update(mapping_dict)
        
        dense2src_dst = [(n + 1, (c, chr2chr[c])) for n, c in enumerate(src_alphabet_str)]

        src_str = [(s,s) for _, (s, _) in dense2src_dst]
        src_str = ''.join([s for _, s in sorted(src_str)])

        dst_str = [(s, d) for _, (s, d) in dense2src_dst]
        dst_str = [(s, d) for s, d in dst_str if d != unknown_chr]  # Remove unknown characters from the destination string
        dst_str = ''.join([d for _, d in sorted(dst_str)])
        
        src_full_str = unknown_chr + src_str
        
        srcchr2dense = {s: n for n, s in enumerate(src_full_str)}

        np_chrord2dense = np.zeros(65536, dtype=np.uint16)
        np_dense2chrord = np.zeros(65536, dtype=np.uint16)
        for c, n in srcchr2dense.items():
            np_chrord2dense[ord(c)] = n
            np_dense2chrord[n] = ord(chr2chr[c])
        return src_alphabet_str, dst_alphabet_str, np_chrord2dense, np_dense2chrord

    def __init__(self, mapping_dict={}, unknown_chr: str = "�", unicode_normalization: Literal["Dense", "Composite", None] = "Dense"):
        super().__init__(unicode_normalization=unicode_normalization, unknown_chr=unknown_chr, mapping_dict=mapping_dict.copy())
        self.__src_alphabet_str, self.__dst_alphabet_str, self.__np_chrord2dense, self.__np_dense2chrord = self.__create_mappers(self.mapping_dict, self.unknown_chr)
        self.__max_label = self.__np_dense2chrord.max(0)

    def __call__(self, text: str) -> str:
        label_seq = self.str_to_intlabel_seq(text)
        return self.intlabel_seq_to_str(label_seq)

    def str_to_intlabel_seq(self, text: str) -> np.ndarray:
        sparse_np_text = fast_str_to_numpy(text)
        dense_np_text = self.__np_chrord2dense[sparse_np_text]
        return dense_np_text

    def intlabel_seq_to_str(self, dense_np_text: np.ndarray) -> str:
        output_sparse_text = self.__np_dense2chrord[dense_np_text]
        output_sparse_text[output_sparse_text == 0] = ord(self.unknown_chr)  # Replace unknown characters with the unknown character ordinal
        return fast_numpy_to_str(output_sparse_text)

    def get_unigram(self, text: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        np_text = self.str_to_intlabel_seq(self.unknown_chr +self.src_alphabet_str+text)
        values, counts = np.unique(np_text, return_counts=True)
        counts = counts - 1  # removing the counts of the added characters
        labels = self.intlabel_seq_to_str(values)
        return values, counts, labels

    def str_to_onehot(self, text: str) -> np.ndarray:
        #raise NotImplementedError("str_to_btc is not implemented for LemmatizerBMP. Use str_to_intlabel_seq instead.")
        label_seq = self.str_to_intlabel_seq(text)
        return np.eye(len(self), dtype=np.uint8)[label_seq]
    
    def onehot_to_str(self, onehot: np.ndarray) -> str:
        #raise NotImplementedError("btc_to_str is not implemented for LemmatizerBMP. Use intlabel_seq_to_str instead.")
        if onehot.ndim == 1:
            onehot = onehot.reshape(1, -1)
        dense_np_text = np.argmax(onehot, axis=1)
        return self.intlabel_seq_to_str(dense_np_text)

    @property
    def dst_alphabet_str(self) -> str:
        return self.__dst_alphabet_str

    @property
    def src_alphabet_str(self) -> str:
        return self.__src_alphabet_str
    
    def __repr__(self):
        return f"LemmatizerBMP(mapping_dict={repr(self.mapping_dict)}, unknown_chr={repr(self.unknown_chr)})"


def main_remap_alphabet():
    import fargv
    from pylelemmatize.charsets import LemmatizerBMP, allbmp_encoding_alphabet_strings
    p = {
        "mode": "unigram",
        "inputs": set([]),
        "map": "U:V,u:v",
        "append_extention": ".remap.txt",
        "chars_beyond_bmp": False,
        "charset_name": "mes3a",
    }
    args, _ = fargv.fargv(p)
    if args.mode == "unigram":
        raise ValueError(f"Unknown mode: {args.mode}. Available: ['unigram', 'remap']")
    remap = {k: v for k, v in (m.split(':') for m in args.map.split(','))}
    fname_2txt = {}
    for fname in args.inputs:
        txt = open(fname, 'r', encoding='utf-8').read()
        fname_2txt[fname] = txt

    if args.charset_name == "":
        pass
    elif args.charset_name not in allbmp_encoding_alphabet_strings:
        raise ValueError(f"Unknown charset name: {args.charset_name}. Available: {list(allbmp_encoding_alphabet_strings.keys())}")
