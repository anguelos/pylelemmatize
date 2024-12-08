import re
from typing import Union, List


def fast_extract_text_from_xml(xml_string: str, concatenate: bool = True) -> Union[str, List[str]]:
    # Regular expression to find text within tags
    # This regex avoids capturing empty spaces between tags and ensures capturing text
    text_parts = re.findall(r'>\s*([^<>]+?)\s*<', xml_string)
    if concatenate:
        return ' '.join(text_parts)
    else:
        return text_parts
