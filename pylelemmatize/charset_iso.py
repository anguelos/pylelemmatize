import codecs
import encodings
from typing import Dict, List
import encodings.aliases
from unidecode import unidecode


def get_characters_in_codepage(codepage):
    characters = []
    decoder = codecs.getdecoder(codepage)
    # Iterate over each byte in the possible range for single-byte encodings
    for byte in range(32, 256):
        try:
            char = decoder(bytes([byte]))[0]
            # Convert the byte to a bytes object and decode it
            characters.append(char)
        except UnicodeDecodeError:
            # If a byte cannot be decoded, append a placeholder or skip it
            characters.append('�')  # Placeholder for undecodable byte
        # quit hack to remove unpritable characters. they all seem to have the form '\x83'
        characters = sorted(set([c if len(repr(c)) <= 3 else '�' for c in characters]))
    return ''.join(characters)


def simplify_string(s):
    return ''.join(sorted(set(s)))


def get_charactermap_names() -> Dict[str, List[str]]:
    # Get all encoding aliases from the encodings.aliases module
    encoding_aliases = set(encodings.aliases.aliases.values())
    encoding_aliases = [a for a in encoding_aliases if a.startswith('iso8')]
    # Convert the set to a sorted list for better readability
    sorted_encodings = list(sorted(encoding_aliases))
    return {"iso-8859": sorted_encodings}


def get_encoding_dicts() -> Dict[str, List[str]]:
    return {k: get_characters_in_codepage(k) for k in get_charactermap_names()['iso-8859']}


iso_alphabets = ['iso_8859_10', 'iso_8859_11', 'iso_8859_13', 'iso_8859_14', 'iso_8859_15', 'iso_8859_16', 'iso_8859_2', 'iso_8859_3', 'iso_8859_4', 'iso_8859_5', 'iso_8859_6', 'iso_8859_7', 'iso_8859_8', 'iso_8859_9', 'iso_8859_1']

