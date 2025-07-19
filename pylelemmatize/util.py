import re
from typing import Generator, Set, Union, List
import xml.etree.ElementTree as ET
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
            

def extract_transcription_from_page_xml(xml_content, line_separator="\n", linesegment_separator="\t", ignore_deleted=True):
    """
    Extracts transcription from a PAGE XML document string.
    
    Args:
        xml_content (str): The PAGE XML content as a string.
        ignore_deleted (bool): If True, text within <del> tags will be ignored.

    Returns:
        str: The full transcription with each <TextLine> stitched by tabs and lines separated by newlines.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML content: {e}")

    ns = {'ns': root.tag.split('}')[0].strip('{')}

    lines = []
    for text_line in root.findall(".//ns:TextLine", ns):
        line_entries = []

        for text_equiv in text_line.findall("ns:TextEquiv", ns):
            unicode_el = text_equiv.find("ns:Unicode", ns)
            if unicode_el is not None and unicode_el.text:
                # Parse the Unicode element's content to handle inner XML like <del>
                try:
                    unicode_inner = ET.fromstring(f"<root>{unicode_el.text}</root>")
                    parts = []
                    for node in unicode_inner.iter():
                        if node.tag == 'root':
                            continue
                        if node.tag == 'del' and ignore_deleted:
                            continue
                        if node.text:
                            parts.append(node.text.strip())
                    if unicode_inner.text and (not ignore_deleted or '<del>' not in unicode_el.text):
                        parts.insert(0, unicode_inner.text.strip())
                    if parts:
                        line_entries.append(" ".join(parts))
                except ET.ParseError:
                    # If no inner XML, treat as plain text
                    line_entries.append(unicode_el.text.strip())

        if line_entries:
            lines.append(linesegment_separator.join(line_entries))

    return line_separator.join(lines)


def main_extract_transcription_from_page_xml():
    import fargv
    import glob
    from pathlib import Path
    import sys
    p = {
        "corpus_glob": "",
        "corpus_files": set([]),
        "line_separator": "\n",
        "linesegment_separator": "\t",
        "include_deleted": False,
        "verbose": False,
        "output": "stdout",
        "output_postfix": ""
    }
    args, _ = fargv.fargv(p)
    if args.output == "stdout":
        assert not args.output_postfix, "Output postfix is not allowed for stdout"
        output_f = sys.stdout
    elif args.output == "stderr":
        assert not args.output_postfix, "Output postfix is not allowed for stderr"
        output_f = sys.stderr
    elif args.output:
        assert len(args.output_postfix) == 1, "Only one output postfix is allowed"
        output_f = open(args.output, 'w', encoding='utf-8')
    else:
        assert len(args.output_postfix) > 0, "Output postfix is required if output is not set to stdout or stderr or a file path"
        output_f = False
    for file in list(sorted(args.corpus_files))+ list(glob.glob(args.corpus_glob)):
        if not Path(file).is_file():
            continue
        with open(file, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        transcription = extract_transcription_from_page_xml(
            xml_content,
            line_separator=args.line_separator,
            linesegment_separator=args.linesegment_separator,
            ignore_deleted= not args.include_deleted
        )
        if output_f is False:
            with open(file + args.output_postfix, 'w', encoding='utf-8') as f:
                print(transcription, file=f)
        else:
            print(transcription, file=output_f)
        output_f.flush()
        if args.verbose:
            print(f"Processed {file} {len(transcription.split(args.line_separator))} lines, {len(transcription)} characters", file=sys.stderr)
