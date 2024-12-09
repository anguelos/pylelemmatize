import re
from typing import Generator, Set, Union, List

import tqdm


def fast_extract_text_from_xml(xml_string: str, concatenate: bool = True) -> Union[str, List[str]]:
    # Regular expression to find text within tags
    # This regex avoids capturing empty spaces between tags and ensures capturing text
    text_parts = re.findall(r'>\s*([^<>]+?)\s*<', xml_string)
    if concatenate:
        return ' '.join(text_parts)
    else:
        return text_parts


def generate_corpus(filenames: Union[Set[str], List[str]], strip_xml: bool = True, treat_all_file_as_xml: bool = False, verbose: bool = False) -> Generator[str, None, None]:
    if verbose:
        progress = tqdm.tqdm(filenames)
    else:
        progress = filenames

    if strip_xml:
        def dexmlfy(xml_string: str) -> str:
            return fast_extract_text_from_xml(xml_string)
    else:
        def dexmlfy(xml_string: str) -> str:
            return xml_string

    for file in progress:
        data = open(file).read()
        if treat_all_file_as_xml or file.lower().endswith(".xml"):
            yield dexmlfy(data)
        else:
            yield data
            
