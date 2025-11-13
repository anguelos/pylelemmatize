from collections import defaultdict
from typing import Dict, Optional, Tuple, Literal, Union
import numpy as np
from unidecode import unidecode
from .abstract_mapper import AbstractLemmatizer, GenericLemmatizer, fast_numpy_to_str, fast_str_to_numpy


class LemmatizerBMP(GenericLemmatizer):
    @staticmethod
    def alphabet_in_bmp(alphabet: Optional[str]) -> bool:
        """
        Check if all characters in the given alphabet are within the BMP (Basic Multilingual Plane).

        Parameters
        ----------
        alphabet : Optional[str]
            A string containing the alphabet to check. If None, the method returns True.

        Returns
        -------
        bool
            True if all characters are within the BMP, False otherwise.
        """
        if alphabet is None:
            return True
        return (fast_str_to_numpy(alphabet) > 65535).sum() == 0

    @staticmethod
    def __create_mappers(mapping_dict, unknown_chr) -> Tuple[defaultdict, np.ndarray, Dict[int, str], Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
        """
        Create mappings and data structures for character transformation.

        This static method generates several mappings and arrays to facilitate
        the transformation of characters based on a given mapping dictionary.
        It ensures that all characters involved are within the BMP (Basic Multilingual Plane)
        and handles unknown characters by mapping them to a specified `unknown_chr`.

        Parameters
        ----------
        mapping_dict : dict
            A dictionary where keys are source characters and values are destination characters.
            Both keys and values must be BMP characters.
        unknown_chr : str
            A single BMP character used to represent unknown or unmapped characters.

        Returns
        -------
        Tuple[defaultdict, np.ndarray, Dict[int, str], Dict[str, int], np.ndarray, np.ndarray, np.ndarray]
            - src_alphabet_str : str
              A string containing all source characters sorted in ascending order.
            - dst_alphabet_str : str
              A string containing all unique destination characters sorted in ascending order.
            - np_chrord2dense : np.ndarray
              A NumPy array mapping Unicode code points to dense indices.
            - np_dense2chrord : np.ndarray
              A NumPy array mapping dense indices back to Unicode code points.

        Raises
        ------
        AssertionError
            If any character in `mapping_dict` keys or values, or `unknown_chr`, is not a BMP character.
            If `unknown_chr` is present in `mapping_dict` but does not map to itself.
        """
        if any([ord(c) >= 65536 for c in mapping_dict.keys()]) or \
                any([ord(c) >= 65536 for c in mapping_dict.values()]) or \
                ord(unknown_chr) >= 65536:
            raise ValueError("LemmatizerBMP can only handle BMP characters. Please use GenericLemmatizer for non-BMP characters.")

        if unknown_chr in mapping_dict:
            assert mapping_dict[unknown_chr] == unknown_chr, "unknown_chr must map to itself in the mapping_dict."
            del mapping_dict[unknown_chr]  # Remove the unknown character from the mapping to avoid confusion

        src_alphabet_str = ''.join(sorted(mapping_dict.keys()))
        dst_alphabet_str = ''.join(sorted(set(mapping_dict.values())))

        chr2chr = defaultdict(lambda: unknown_chr)
        chr2chr.update(mapping_dict)
        
        dense2src_dst = [(n + 1, (c, chr2chr[c])) for n, c in enumerate(src_alphabet_str)]

        src_str = [(s, s) for _, (s, _) in dense2src_dst]
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

    def __init__(self, mapping_dict: Union[Dict[str, str]] = {}, unknown_chr: str = "�", unicode_normalization: Literal["Dense", "Composite", None] = "Dense"):
        """
        Initialize the LemmatizerBMP instance.

        Parameters
        ----------
        mapping_dict : Union[Dict[str, str]], optional
            A dictionary mapping source characters to destination characters. 
            If a string is provided, it will be converted into a dictionary 
            where each character maps to itself. Defaults to an empty dictionary.
        unknown_chr : str, optional
            The character to use for unknown mappings. Defaults to "�".
        unicode_normalization : Literal["Dense", "Composite", None], optional
            The type of Unicode normalization to apply. 
            - "Dense": Use dense Unicode normalization.
            - "Composite": Use composite Unicode normalization.
            - None: No Unicode normalization is applied.
            Defaults to "Dense".

        Notes
        -----
        This constructor initializes the mapping dictionary, sets up Unicode 
        normalization, and creates internal mappings for efficient character 
        transformations.
        """
        if isinstance(mapping_dict, str):
            mapping_dict = {c: c for c in mapping_dict}
        super().__init__(unicode_normalization=unicode_normalization, unknown_chr=unknown_chr, mapping_dict=mapping_dict.copy())
        self.__src_alphabet_str, self.__dst_alphabet_str, self.__np_chrord2dense, self.__np_dense2chrord = self.__create_mappers(self.mapping_dict, self.unknown_chr)
        self.__max_label = self.__np_dense2chrord.max(0)

    def __call__(self, text: str) -> str:
        """
        Transform the input text using the lemmatizer.

        Parameters
        ----------
        text : str
            The input text to transform.

        Returns
        -------
        str
            The transformed text.
        """
        label_seq = self.str_to_intlabel_seq(text)
        return self.intlabel_seq_to_str(label_seq)

    def str_to_intlabel_seq(self, text: str) -> np.ndarray:
        """
        Convert a string to a sequence of integer labels.

        Parameters
        ----------
        text : str
            The input string to convert.

        Returns
        -------
        np.ndarray
            A NumPy array of integer labels representing the input string.
        """
        sparse_np_text = fast_str_to_numpy(text)
        dense_np_text = self.__np_chrord2dense[sparse_np_text]
        return dense_np_text

    def intlabel_seq_to_str(self, dense_np_text: np.ndarray) -> str:
        """
        Convert a sequence of integer labels back to a string.

        Parameters
        ----------
        dense_np_text : np.ndarray
            A NumPy array of integer labels to convert.

        Returns
        -------
        str
            The reconstructed string.
        """
        output_sparse_text = self.__np_dense2chrord[dense_np_text]
        output_sparse_text[output_sparse_text == 0] = ord(self.unknown_chr)  # Replace unknown characters with the unknown character ordinal
        return fast_numpy_to_str(output_sparse_text)

    def get_unigram(self, text: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Compute unigram statistics for the input text.

        Parameters
        ----------
        text : str
            The input text to analyze.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Dict[int, str]]
            - values : np.ndarray
              Unique integer labels in the text.
            - counts : np.ndarray
              Counts of each unique label.
            - labels : Dict[int, str]
              Mapping of integer labels to their corresponding characters.
        """
        np_text = self.str_to_intlabel_seq(self.unknown_chr + self.src_alphabet_str + text)
        values, counts = np.unique(np_text, return_counts=True)
        counts = counts - 1  # removing the counts of the added characters
        labels = self.intlabel_seq_to_str(values)
        return values, counts, labels

    def str_to_onehot(self, text: str) -> np.ndarray:
        """
        Convert a string to a one-hot encoded representation.

        Parameters
        ----------
        text : str
            The input string to convert.

        Returns
        -------
        np.ndarray
            A one-hot encoded NumPy array representing the input string.
        """
        label_seq = self.str_to_intlabel_seq(text)
        return np.eye(len(self), dtype=np.uint8)[label_seq]
    
    def onehot_to_str(self, onehot: np.ndarray) -> str:
        """
        Convert a one-hot encoded representation back to a string.

        Parameters
        ----------
        onehot : np.ndarray
            A one-hot encoded NumPy array to convert.

        Returns
        -------
        str
            The reconstructed string.
        """
        if onehot.ndim == 1:
            onehot = onehot.reshape(1, -1)
        dense_np_text = np.argmax(onehot, axis=1)
        return self.intlabel_seq_to_str(dense_np_text)

    @property
    def dst_alphabet_str(self) -> str:
        """
        Get the destination alphabet as a string.

        Returns
        -------
        str
            The destination alphabet string.
        """
        return self.__dst_alphabet_str

    @property
    def src_alphabet_str(self) -> str:
        """
        Get the source alphabet as a string.

        Returns
        -------
        str
            The source alphabet string.
        """
        return self.__src_alphabet_str
    
    def __repr__(self):
        """
        Return a string representation of the LemmatizerBMP instance.
        Can act as a string serialisation of the instance.

        Returns
        -------
        str
            A string representation of the instance.
        """
        return f"LemmatizerBMP(mapping_dict={repr(self.mapping_dict)}, unknown_chr={repr(self.unknown_chr)})"

