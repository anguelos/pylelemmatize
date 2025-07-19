import sys
import time
from typing import Any, Dict, Generator, Literal, Optional, Set, Tuple, Union
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod
import unicodedata
import types

from unidecode import unidecode



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


def _densify_unicode(self: Any, text: str) -> str:
    """
    Convert a Unicode string to its composed (dense) form using NFC normalization.
    
    Args:
        text (str): The input Unicode string.
    
    Returns:
        str: The composed (dense) Unicode string.
    """
    return unicodedata.normalize('NFC', text)

def _decompose_unicode(self: Any, text: str) -> str:
    """
    Convert a Unicode string to its decomposed (sparse) form using NFD normalization.
    
    Args:
        text (str): The input Unicode string.
    
    Returns:
        str: The decomposed (sparse) Unicode string.
    """
    return unicodedata.normalize('NFD', text)


def _null_unicode(self: Any, text: str) -> str:
    """
    Normalize a Unicode string to its composed (dense) form using NFC normalization.
    
    Args:
        text (str): The input Unicode string.
    
    Returns:
        str: The normalized Unicode string.
    """
    return text


class AbstractLemmatizer(ABC):
    @classmethod
    def __create_mappers(cls, mapping_dict, unknown_chr) -> Tuple[defaultdict, np.ndarray, Dict[int, str], Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
        max_ord = max(ord(c) for c in mapping_dict.keys() | mapping_dict.values() | {unknown_chr})
        assert max_ord < 256**3, f"All characters must be whithin resoble size (256^3). found {max_ord}."

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

        np_chrord2dense = np.zeros(max_ord, dtype=np.uint16)
        np_dense2chrord = np.zeros(max_ord, dtype=np.uint16)
        for c, n in srcchr2dense.items():
            np_chrord2dense[ord(c)] = n
            np_dense2chrord[n] = ord(chr2chr[c])
        return src_alphabet_str, dst_alphabet_str, np_chrord2dense, np_dense2chrord

    @classmethod
    def fast_alphabet_extraction(cls, text: str) -> str:
        np_text = fast_str_to_numpy(text)
        uniq = np.unique(np_text)
        return fast_numpy_to_str(uniq)
    
    def __set_unicode_normalization(self, normalize_unicode: Literal["Dense", "Composite", None] = "Dense"):
        """
        np_text = fast_str_to_numpy(text)
        Set the Unicode normalization method.
        
        Args:
            normalize_unicode (Literal["Dense", "Composite", None]): The normalization type.
                - "Dense" for NFC normalization (composed form).
                - "Composite" for NFD normalization (decomposed form).
                - None for no normalization.
        """
        self.__unicode_normalization = normalize_unicode
        if normalize_unicode.lower() == "dense":
            self.normalize_unicode = types.MethodType(_densify_unicode, self)
            
        elif normalize_unicode == "composite":
            self.normalize_unicode = types.MethodType(_decompose_unicode, self)
        elif normalize_unicode is None:
            self.normalize_unicode = types.MethodType(_null_unicode, self)
        else:
            raise ValueError(f"Unknown normalization type: {normalize_unicode}")

    def __init__(self, unicode_normalization: Literal["Dense", "Composite", None] = "Dense", unknown_chr: str = "�"):
        super().__init__()
        self.__set_unicode_normalization(unicode_normalization)
        self.__unknown_chr = unknown_chr

    @property
    def unicode_normalization(self) -> Literal["Dense", "Composite", None]:
        return self.__unicode_normalization

    @abstractmethod
    def __call__(self, text: str) -> str:
        """Convert text to the alphabet representation."""
        pass

    @property
    @abstractmethod
    def src_alphabet_str(self) -> str:
        pass

    @property
    @abstractmethod
    def dst_alphabet_str(self) -> str:
        pass

    @property
    def unknown_chr(self) -> str:
        return self.__unknown_chr

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError("Subclasses must implement __repr__ method.")

    @property
    def alphabet_tsv(self) -> str:
        title = " #\tUnicode Number\tUnicode x10\tUnicode x16\tPython String"
        lines = [title]
        for n, c in enumerate(self.unknown_chr + self.dst_alphabet_str):
            lines.append(f"{n}\t{unicodedata.name(c)}\t{ord(c)}\t{ord(c):x}\t{repr(c)}")
        return "\n".join(lines)
    
    @property
    def mapping_tsv(self) -> str:
        title = "Source Character\tDestination Character"
        lines = [title]
        for src, dst in sorted(self.mapping_dict.items()):
            lines.append(f"{repr(src)}\t{repr(dst)}")
        return "\n".join(lines)
    
    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, AbstractLemmatizer):
            return True
        else:
            return self.mapping_tsv != other.mapping_tsv

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AbstractLemmatizer):
            return False
        else:
            return self.mapping_tsv == other.mapping_tsv
    
    def __len__(self) -> int:
        """Return the size of the destination alphabet."""
        return len(self.dst_alphabet_str) + 1

    def get_unigram(self, text: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        # adding all characters atleast once to make np.unique count zero counts
        src_alphabet_str, _, __np_chrord2dense, __np_dense2chrord = self.__create_mappers(self.mapping_dict, self.unknown_chr)
        np_text = fast_str_to_numpy(self.unknown_chr + src_alphabet_str + text)
        mapped_np_text = __np_chrord2dense[np_text]
        values, counts = np.unique(mapped_np_text, return_counts=True)
        counts = counts - 1  # removing the counts of the added characters
        labels = np.array([c for c in fast_numpy_to_str(__np_dense2chrord[values])], dtype=np.str_)
        return values, counts, labels

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

    def get_unigram(self, text: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        # adding all characters atleast once to make np.unique count zero counts
        src_alphabet_str, _, __np_chrord2dense, __np_dense2chrord = AbstractLemmatizer.__create_mappers(self.mapping_dict, self.unknown_chr)
        np_text = fast_str_to_numpy(self.unknown_chr + src_alphabet_str + text)
        mapped_np_text = __np_chrord2dense[np_text]
        values, counts = np.unique(mapped_np_text, return_counts=True)
        counts = counts - 1  # removing the counts of the added characters
        labels = np.array([c for c in fast_numpy_to_str(__np_dense2chrord[values])], dtype=np.str_)
        labels[0] = self.unknown_chr  # Ensure the unknown character is included
        print("\n\nLABELS:\n", labels, "\n\n\n")
        return values, counts, labels


class GenericLemmatizer(AbstractLemmatizer):
    @classmethod
    def from_alphabet_str(cls, alphabet_str, mapping_dict: Optional[Dict[str, str]] = None, unknown_chr: str = "�") -> 'GenericLemmatizer':
        """
        Create a GenericLemmatizer instance from a character mapping dictionary.
        
        Args:
            mapping_dict (Dict[str, str]): A dictionary mapping source characters to destination characters.
            unknown_chr (str): Character to use for unknown characters not in the mapping.
        
        Returns:
            GenericLemmatizer: An instance of GenericLemmatizer.
        """
        full_mapping_dict = {c: c for c in alphabet_str}
        if mapping_dict is not None:
            full_mapping_dict.update(mapping_dict)
        return cls(mapping_dict=full_mapping_dict, unknown_chr=unknown_chr)

    @classmethod
    def from_alphabet_mapping(cls, src_alphabet_str: str, dst_alphabet_str: Optional[str] = None, custom_map: Optional[Dict[str, str]] = None, unknown_chr: str = "�", guess_unidecode: bool = False) -> 'LemmatizerBMP':
        mapping_dict = {}
        if dst_alphabet_str is None:
            dst_alphabet_str = src_alphabet_str
        if custom_map is not None:
            src_alphabet_str = ''.join(sorted(set(src_alphabet_str + ''.join(custom_map.keys()))))
        for c in src_alphabet_str:
            c = _densify_unicode(None, c)  # Normalize the character to NFC)
            if custom_map and c in custom_map:
                print(f"CUSTOM Using custom mapping for character {repr(c)}: {repr(custom_map[c])}")
                mapping_dict[c] = custom_map[c]
            elif c in dst_alphabet_str and len(c) == 1:
                mapping_dict[c] = c
            elif c in dst_alphabet_str and len(c) >= 1:
                mapping_dict[c] = c[0]
            elif c in dst_alphabet_str and len(c) == 0:
                mapping_dict[c] = unknown_chr
            elif guess_unidecode:
                d = _densify_unicode(None, unidecode(c))
                if len(d) > 1:
                    #print(f"Warning: Unidecode for character {repr(c)} -> {repr(d)} is not a single character. Using unknown character instead.")
                    mapping_dict[c] = d[0]
                elif len(d) == 0:
                    #print(f"Warning: Unidecode for character {repr(c)} mapps to ''. Using unknown character instead.")
                    mapping_dict[c] = unknown_chr
                elif len(d) == 1:
                    mapping_dict[c] = d
                else:
                    raise ValueError(f"Unidecode for character {repr(c)} returned an unexpected length: {len(d)}. Expected 0, 1 or more than 1.")
        for k in list(mapping_dict):
            if len(k) != 1 or len(mapping_dict[k]) != 1:
                #print(f"Warning: Mapping for character {repr(k)} is not a single character. Removing it from the mapping.")
                #v = mapping_dict[k]
                #print(f"K({repr(k)}) Dense: {len(_densify_unicode(None, k))} Sparce: {len(_decompose_unicode(None, k))}, As is:{len(k)}")
                #print(f"V({repr(v)}) Dense: {len(_densify_unicode(None, v))} Sparce: {len(_decompose_unicode(None, v))}, As is:{len(v)}\n")
                del mapping_dict[k]
        for k in list(mapping_dict):
            if '' in (k, mapping_dict[k]):
                #print(f"Warning: Empty mapping for character {repr(k)} in {repr(mapping_dict[k])}. Removing it from the mapping.")
                del mapping_dict[k]
        return cls(mapping_dict, unknown_chr)

    def copy_removing_unused_inputs(self, txt: str) -> Any:
        txt = self.normalize_unicode(txt)
        txt = self.fast_alphabet_extraction(txt)
        mapping_dict = self.mapping_dict.copy()
        reduced_mapping_dict = {}
        for k in txt:
            if k in mapping_dict:
                reduced_mapping_dict[k] = mapping_dict[k]
        return self.__class__(mapping_dict=reduced_mapping_dict, unknown_chr=self.unknown_chr, unicode_normalization=self.unicode_normalization)


    def __init__(self, mapping_dict = {}, unknown_chr: str = "�", unicode_normalization: Literal["Dense", "Composed", None] = "Dense"):
        super().__init__(unicode_normalization=unicode_normalization, unknown_chr=unknown_chr)
        self.mapping_dict = mapping_dict.copy()
    
    def len(self) -> int:
        """Return the size of the destination alphabet."""
        return len(self.dst_alphabet_str) + 1

    def __call__(self, text: str) -> str:
        text = self.normalize_unicode(text)  #  This might be the bottleneck in a fast implementation.
        return ''.join(self.mapping_dict.get(c, self.unknown_chr) for c in text)

    @property
    def src_alphabet_str(self) -> str:
        return ''.join(sorted(self.mapping_dict.keys()))

    @property
    def dst_alphabet_str(self) -> str:
        return ''.join(sorted(set(self.mapping_dict.values())))
    
    def __repr__(self):
        return f"GenericLemmatizer(mapping_dict={repr(self.mapping_dict)}, unknown_chr={repr(self.unknown_chr)})"


