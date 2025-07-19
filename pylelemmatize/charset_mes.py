# source for data https://www.evertype.com/standards/iso10646/pdf/cwa13873.pdf pages 9-13

from typing import Dict, List


mes1_ranges = {
    "BASIC LATIN": "0020-007E",
    "LATIN-1 SUPPLEMENT": "00A0-00FF",
    "LATIN EXTENDED-A": "0100-017F",
    "SPACING MODIFIER LETTERS": "02B0-02FF",
    "GENERAL PUNCTUATION": "2000-206F",
    "CURRENCY SYMBOLS": "20A0-20CF",
    "LETTERLIKE SYMBOLS": "2100-214F",
    "NUMBER FORMS": "2150-218F",
    "ARROWS": "2190-21FF",
    "MISCELLANEOUS SYMBOLS": "2600-26FF",
}

mes2_ranges = {
    "BASIC LATIN": "0020-007E",
    "LATIN-1 SUPPLEMENT": "00A0-00FF",
    "LATIN EXTENDED-A": "0100-017F",
    "LATIN EXTENDED-B": "0180-024F",
    "IPA EXTENSIONS": "0250-02AF",
    "SPACING MODIFIER LETTERS": "02B0-02FF",
    "BASIC GREEK": "0370-03CF",
    "GREEK SYMBOLS AND COPTIC": "03D0-03FF",
    "CYRILLIC": "0400-04FF",
    "LATIN EXTENDED ADDITIONAL": "1E00-1EFF",
    "GREEK EXTENDED": "1F00-1FFF",
    "GENERAL PUNCTUATION": "2000-206F",
    "SUPERSCRIPTS AND SUBSCRIPTS": "2070-209F",
    "CURRENCY SYMBOLS": "20A0-20CF",
    "LETTERLIKE SYMBOLS": "2100-214F",
    "NUMBER FORMS": "2150-218F",
    "ARROWS": "2190-21FF",
    "MATHEMATICAL OPERATORS": "2200-22FF",
    "MISCELLANEOUS TECHNICAL": "2300-23FF",
    "BOX DRAWING": "2500-257F",
    "BLOCK ELEMENTS": "2580-259F",
    "GEOMETRIC SHAPES": "25A0-25FF",
    "MISCELLANEOUS SYMBOLS": "2600-26FF",
    "LTR PRESENTATION FORMS": "FB00-FB4F",
    "SPECIALS": "FFF0-FFFD"
}

mes3a_ranges = {
    "BASIC LATIN": "0020-007E",
    "LATIN-1 SUPPLEMENT": "00A0-00FF",
    "LATIN EXTENDED-A": "0100-017F",
    "LATIN EXTENDED-B": "0180-024F",
    "IPA EXTENSIONS": "0250-02AF",
    "SPACING MODIFIER LETTERS": "02B0-02FF",
    "COMBINING DIACRITICAL MARKS": "0300-036F",
    "BASIC GREEK": "0370-03CF",
    "GREEK SYMBOLS AND COPTIC": "03D0-03FF",
    "CYRILLIC": "0400-04FF",
    "ARMENIAN": "0530-058F",
    "BASIC GEORGIAN": "10D0-10FF",
    "LATIN EXTENDED ADDITIONAL": "1E00-1EFF",
    "GREEK EXTENDED": "1F00-1FFF",
    "GENERAL PUNCTUATION": "2000-206F",
    "SUPERSCRIPTS AND SUBSCRIPTS": "2070-209F",
    "CURRENCY SYMBOLS": "20A0-20CF",
    "COMBINING DIACRITICAL MARKS FOR SYMBOLS": "20D0-20FF",
    "LETTERLIKE SYMBOLS": "2100-214F",
    "NUMBER FORMS": "2150-218F",
    "ARROWS": "2190-21FF",
    "MATHEMATICAL OPERATORS": "2200-22FF",
    "MISCELLANEOUS TECHNICAL": "2300-23FF",
    "OPTICAL CHARACTER RECOGNITION": "2440-245F",
    "BOX DRAWING": "2500-257F",
    "BLOCK ELEMENTS": "2580-259F",
    "GEOMETRIC SHAPES": "25A0-25FF",
    "MISCELLANEOUS SYMBOLS": "2600-26FF",
    "ALPHABETIC PRESENTATION FORMS": "FB00-FB4F",
    "COMBINING HALF MARKS": "FE20-FE2F",
    "SPECIALS": "FFF0-FFFD"
}

mes3b_ranges = {
    "BASIC LATIN": "0020-007E",
    "LATIN-1 SUPPLEMENT": "00A0-00FF",
    "LATIN EXTENDED-A": "0100-017F",
    "LATIN EXTENDED-B": "0180-024F",
    "IPA EXTENSIONS": "0250-02AF",
    "SPACING MODIFIER LETTERS": "02B0-02FF",
    "COMBINING DIACRITICAL MARKS": "0300-036F",
    "BASIC GREEK": "0370-03CF",
    "GREEK SYMBOLS AND COPTIC": "03D0-03FF",
    "CYRILLIC": "0400-04FF",
    "ARMENIAN": "0530-058F",
    "BASIC GEORGIAN": "10D0-10FF",
    "LATIN EXTENDED ADDITIONAL": "1E00-1EFF",
    "GREEK EXTENDED": "1F00-1FFF",
    "GENERAL PUNCTUATION": "2000-206F",
    "SUPERSCRIPTS AND SUBSCRIPTS": "2070-209F",
    "CURRENCY SYMBOLS": "20A0-20CF",
    "COMBINING DIACRITICAL MARKS FOR SYMBOLS": "20D0-20FF",
    "LETTERLIKE SYMBOLS": "2100-214F",
    "NUMBER FORMS": "2150-218F",
    "ARROWS": "2190-21FF",
    "MATHEMATICAL OPERATORS": "2200-22FF",
    "MISCELLANEOUS TECHNICAL": "2300-23FF",
    "OPTICAL CHARACTER RECOGNITION": "2440-245F",
    "BOX DRAWING": "2500-257F",
    "BLOCK ELEMENTS": "2580-259F",
    "GEOMETRIC SHAPES": "25A0-25FF",
    "MISCELLANEOUS SYMBOLS": "2600-26FF",
    "ALPHABETIC PRESENTATION FORMS": "FB00-FB4F",
    "COMBINING HALF MARKS": "FE20-FE2F",
    "SPECIALS": "FFF0-FFFD"
}

all_mes_ranges = {"mes1": mes1_ranges, "mes2": mes2_ranges, "mes3a": mes3a_ranges, "mes3b": mes3b_ranges}


def extract_ranges(rangestr: str) -> str:
    start, end = [int(v, 16) for v in rangestr.split('-')]
    return ''.join([chr(i) for i in range(start, end+1)])


def get_mes_alphabet_dict(ranges: Dict[str, str]) -> Dict[str, str]:
    return {k: extract_ranges(v) for k, v in ranges.items()}


def get_charactermap_names() -> Dict[str, List[str]]:
    return {"mes": ["mes1", "mes2", "mes3a", "mes3b"]}


def get_encoding_dicts() -> Dict[str, List[str]]:
    res = {}
    for name in get_charactermap_names()["mes"]:
        range_strs = []
        ranges = all_mes_ranges[name]
        for range in ranges.values():
            range_strs.append(extract_ranges(range))
        alphabet = ''.join(sorted(set(range_strs)))
        res[name] = alphabet
    return res


# if __name__ == "__main__":
#     print(f"MES1: {len(''.join(get_mes_alphabet_dict(mes1_ranges).values()))}")
#     for k, v in get_mes_alphabet_dict(mes1_ranges).items():
#         print(f"{k}: {repr(v)}")
#     print(f"\n\n\nMES2: {len(''.join(get_mes_alphabet_dict(mes2_ranges).values()))}")
#     for k, v in get_mes_alphabet_dict(mes2_ranges).items():
#         print(f"{k}: {repr(v)}")

mes_only_alphabet = get_encoding_dicts()
