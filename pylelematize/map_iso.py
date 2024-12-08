import codecs
import encodings
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


def get_encoding_dicts():
    # Get all encoding aliases from the encodings.aliases module
    encoding_aliases = set(encodings.aliases.aliases.values())
    encoding_aliases = [a for a in encoding_aliases if a.startswith('iso8')]
    # Convert the set to a sorted list for better readability
    sorted_encodings = sorted(encoding_aliases)
    return {k: get_characters_in_codepage(k) for k in sorted_encodings}


def encoding_to_ascii(alphabet, force_min_len_to_one: bool = True, force_max_len_to_one: bool = False):
    mapping = [(letter, unidecode(letter)) for letter in alphabet]
    if force_min_len_to_one:
        mapping = [(k, v) if len(v) >= 1 else (k, '�') for k, v in mapping]
    if force_max_len_to_one:
        mapping = [(k, v) if len(v) <= 1 else (k, v[0]) for k, v in mapping]
    return {k: v for k, v in mapping}


def simplify_string(s):
    return ''.join(sorted(set(s)))

codepage_alphabets = get_encoding_dicts()

print("codepage_alphabets = {")
for k, v in codepage_alphabets.items():
    print(f"{repr(k.split('_')[-1])}: {repr(v)} ,")
print("}")

#greek_to_ascii = encoding_to_ascii(codepage_alphabets['iso8859_7'], force_min_len_to_one=True, force_max_len_to_one=True)
#for k, v in greek_to_ascii.items():
#    print(f"{k}: {repr(v)}")


