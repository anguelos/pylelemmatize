from pathlib import Path
import re
import sys
from typing import Dict, Generator, Iterator, List, Literal, Optional, Set, Tuple, Union, IO, Union
from lxml import etree
from abc import ABC, abstractmethod
import unicodedata

from tqdm import tqdm


def extract_text_from_htr_xml(htrxml_path: str, chop_first_words: int = 0, chop_last_words: int = 0) -> str:
    def chop_words(text: str) -> str:  # BAD data might need to reject some words from start/end
        if chop_first_words == 0 and chop_last_words == 0:
            return text
        words = text.split(" ")
        if chop_first_words + chop_last_words >= len(words):
            return ""
        return " ".join(words[chop_first_words: len(words) - chop_last_words])

    xml_str = open(htrxml_path, "r").read()
    if re.search(r'<\s*alto\b', xml_str, re.I):
        ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
        root = etree.fromstring(xml_str.encode("utf-8"))
        lines = []
        for textline in root.xpath(".//alto:TextLine", namespaces=ns):
            words = textline.xpath(".//alto:String/@CONTENT", namespaces=ns)
            line_text = chop_words(" ".join(words))
            lines.append(line_text)
        return "\n".join(lines)
    
    elif re.search(r'<\s*PcGts\b', xml_str, re.I):
        try:
            root = etree.fromstring(xml_str)
        except etree.ParseError as e:
            raise ValueError(f"Invalid XML content: {e}")
        ns = {'ns': root.tag.split('}')[0].strip('{')}
        lines = []
        for text_line in root.findall(".//ns:TextLine", ns):
            line_entries = []
            for text_equiv in text_line.findall("ns:TextEquiv", ns):
                unicode_el = text_equiv.find("ns:Unicode", ns)
                if unicode_el is not None and unicode_el.text:
                    choped_text = chop_words(unicode_el.text.strip())
                    line_entries.append(choped_text)
            if line_entries:
                lines.append("\t".join(line_entries))
        return "\n".join(lines)
    else:
        raise ValueError(f"File {htrxml_path} is neither ALTO nor PAGE XML.")


def get_textlines(filepath: str, assume_txt=False, chop_first_words: int = 0, chop_last_words: int = 0, strip_empty_lines=True) -> List[str]:
    if filepath.lower().endswith(".xml") or filepath.lower().endswith(".pagexml"):
        res = extract_text_from_htr_xml(filepath, chop_first_words=chop_first_words, chop_last_words=chop_last_words).split("\n")
    elif filepath.lower().endswith(".txt") or assume_txt:
        res = open(filepath, "r").read().split("\n")
    else:
        raise ValueError(f"Can't open {filepath}")
    if strip_empty_lines:
        res = [line for line in res if len(line)]
    res = [unicodedata.normalize("NFC", s) for s in res]
    return res


def load_textline_pairs(filelist1: List[str], filelist2: List[str], min_length: int = 10, chop_first_words: int = 0, chop_last_words: int = 0) -> List[Tuple[str, str]]:
    assert len(filelist1) == len(filelist2)
    res = []
    for file1, file2 in zip(sorted(filelist1), sorted(filelist2)):
        lines1 = get_textlines(file1, chop_first_words=chop_first_words, chop_last_words=chop_last_words)
        lines2 = get_textlines(file2, chop_first_words=chop_first_words, chop_last_words=chop_last_words)
        if len(lines1) == len(lines2):
            for line1, line2 in zip(lines1, lines2):
                if len(line1) >= min_length and len(line2) >= min_length:
                    res.append((line1, line2))
        else:
            print(f"Unaligned {file1} {file2} with {len(lines1)} vs {len(lines2)} lines", file=sys.stderr)
    return res


class XMLTextlines(ABC):
    def __init__(self, xml_data: Union[str, Path, bytes, IO]):
        super().__init__()
        if isinstance(xml_data, Path):
            with open(xml_data, "rb") as f:
                xml_bytes = f.read()
        elif hasattr(xml_data, "read"):
            xml_bytes = xml_data.read()
            if isinstance(xml_bytes, str):
                xml_bytes = xml_bytes.encode("utf-8")
        elif isinstance(xml_data, str):
            if Path(xml_data).is_file():
                with open(xml_data, "rb") as f:
                    xml_bytes = f.read()
            else:
                xml_bytes = xml_data.encode("utf-8")
        elif isinstance(xml_data, bytes):
            xml_bytes = xml_data
        else:
            raise ValueError("Unsupported type for xml_data")
        self._xml_bytes = xml_bytes

    @abstractmethod
    def _parse_xml(self):
        pass
        # self._xml_root = etree.fromstring(self._xml_bytes, remove_blank_text=True)

    @abstractmethod
    def __getitem__(self, idx: int) -> str:
        pass

    @abstractmethod
    def __setitem__(self, key: int, value: str):
        pass

    def __delattr__(self, name):
        raise NotImplementedError("Deletion of attributes is not supported.")

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[str]:
        for i in range(len(self)):
            yield self[i]

    def get_xml_str(self) -> str:
        # c14n returns bytes
        return etree.tostring(self._xml_root, method="c14n").decode("utf-8")


    @classmethod
    def from_tei(cls, xml_str: str) -> 'XMLMixedTextlines':
        raise NotImplementedError("TEI parsing not implemented yet.")
        ignored_children=["{http://www.tei-c.org/ns/1.0}teiHeader",
                          "{http://www.tei-c.org/ns/1.0}sourceDoc",
                          ]
        return XMLMixedTextlines(xml_str, mandatory_paranets = ['teiHeader', 'text'], ignored_children= ignored_children)
    
    @classmethod
    def from_pagexml(cls, xml_str: str) -> 'XMLMixedTextlines':
        ignored_children=["{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Metadata",
                          ]
        return XMLMixedTextlines(xml_str, mandatory_parents = ['PcGts', 'Page'], ignored_children= ignored_children)
    
    @classmethod
    def from_alto(cls, xml_str: str) -> 'XMLAltoTextlines':
        return XMLAltoTextlines(xml_str)


class XMLMixedTextlines(XMLTextlines):
    @staticmethod
    def parse_attributes(node, missing_mandatory: bool, mandatory_parents: List[str], ignored_children: List[str]) -> Iterator[Tuple[str, etree._Element]]:
        missing_mandatory = missing_mandatory and (mandatory_parents is not None and node.tag not in mandatory_parents)
        text = node.text
        if text is None:
            text = ""
        else:
            if text.strip() == "":
                text = ""
        if not missing_mandatory and text != "":
            yield(text, node)
        for child in list(node):
            if child.tail and not missing_mandatory:
                ctext = child.tail
                if ctext is None:
                    ctext = ""
                if ctext.strip() != "":
                    yield(ctext, child)
            if child.tag not in ignored_children:
                yield from XMLMixedTextlines.parse_attributes(child, missing_mandatory, mandatory_parents, ignored_children)

    def __init__(self, xml_str: str, mandatory_parents: Optional[List[str]] = None, ignored_children: Optional[List[str]] = None):
        super().__init__(xml_str)
        self.mandatory_parents = mandatory_parents or []
        self.ignored_children = ignored_children or []
        self.idx_to_txt_node = {}
        self._parse_xml()
    
    def _parse_xml(self):
        self._xml_root = etree.fromstring(self._xml_bytes)
        txt_nodes = list(XMLMixedTextlines.parse_attributes(self._xml_root, False, mandatory_parents = self.mandatory_parents, ignored_children = self.ignored_children))
        self.idx_to_txt_node = {i: (txt, node) for i, (txt, node) in enumerate(txt_nodes) if txt is not None}
    
    def __getitem__(self, idx: int) -> str:
        text, node = self.idx_to_txt_node[idx]
        return text

    def __setitem__(self, key: int, value: str):
        n = self.idx_to_txt_node[key][1]
        n.text = value
        self.idx_to_txt_node[key] = (value, n)
    
    def __len__(self) -> int:
        return len(self.idx_to_txt_node)


class XMLAltoTextlines(XMLTextlines):
    def __init__(self, xml_data: Union[str, Path, bytes, IO]):
        super().__init__(xml_data)
        self._parse_xml()
    
    def _parse_xml(self):
        self._xml_root = etree.fromstring(self._xml_bytes)
        self.text_elements = [el for el in self._xml_root.findall(".//{http://www.loc.gov/standards/alto/ns-v4#}String") if el.get("CONTENT") is not None]
        self.text_elements = [el for el in self.text_elements if el.get("CONTENT").strip() != ""]

    def __getitem__(self, idx: int) -> str:
        print(f"Getting item at index {idx}", file=sys.stderr)
        return self.text_elements[idx].get("CONTENT")
    
    def __setitem__(self, key: int, value: str):
        self.text_elements[key].set("CONTENT", value)

    def __len__(self) -> int:
        return len(self.text_elements)


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
        progress = tqdm(filenames)
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
