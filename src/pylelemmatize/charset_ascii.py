from typing import Dict, List
import string

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

ascii_alphabets = ["ascii", "ascii_lowercase", "ascii_uppercase", "ascii_letters", "digits", "hexdigits", "octdigits", "punctuation"]
#ascii_only_alphabet = get_encoding_dicts()
