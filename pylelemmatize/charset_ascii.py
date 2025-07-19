from typing import Dict, List
from unidecode import unidecode
import string


def encoding_to_ascii(alphabet, force_min_len_to_one: bool = True, force_max_len_to_one: bool = False):
    mapping = [(letter, unidecode(letter)) for letter in alphabet]
    if force_min_len_to_one:
        mapping = [(k, v) if len(v) >= 1 else (k, '�') for k, v in mapping]
    if force_max_len_to_one:
        mapping = [(k, v) if len(v) <= 1 else (k, v[0]) for k, v in mapping]
    return {k: v for k, v in mapping}


def get_charactermap_names() -> Dict[str, List[str]]:
    return {"ascii": ['ascii', 'ascii_lowercase', 'ascii_uppercase', 'ascii_letters', 'digits', 'hexdigits', 'octdigits', 'punctuation']}


def get_encoding_dicts() -> Dict[str, List[str]]:
    return {
        "ascii": string.printable,
        "ascii_lowercase": string.ascii_lowercase,
        "ascii_uppercase": string.ascii_uppercase,
        "ascii_letters": string.ascii_letters,
        "digits": string.digits,
        "hexdigits": string.hexdigits,
        "octdigits": string.octdigits,
        "punctuation": string.punctuation
    }


ascii_only_alphabet = get_encoding_dicts()
