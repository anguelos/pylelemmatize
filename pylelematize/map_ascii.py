from unidecode import unidecode
import string


def encoding_to_ascii(alphabet, force_min_len_to_one: bool = True, force_max_len_to_one: bool = False):
    mapping = [(letter, unidecode(letter)) for letter in alphabet]
    if force_min_len_to_one:
        mapping = [(k, v) if len(v) >= 1 else (k, '�') for k, v in mapping]
    if force_max_len_to_one:
        mapping = [(k, v) if len(v) <= 1 else (k, v[0]) for k, v in mapping]
    return {k: v for k, v in mapping}
