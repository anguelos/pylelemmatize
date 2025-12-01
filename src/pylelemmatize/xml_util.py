from pathlib import Path
import re
import sys
from typing import Iterator, Literal, Tuple, Union


class XMLTextIterator:
    """
    Iterate over an XML string as (text, xml) pairs such that:
        ''.join(text + xml for text, xml in XMLTextIterator(s, ...)) == s

    Modes
    -----
    - attributes_as_text=False (default): each tag is a single XML chunk.
      Example:
        s = '<xml> with some text <tag param1="AA" param2="BB">some more <smalltag/> text</tag></xml>'
        list(XMLTextIterator(s, attributes_as_text=False)) ->
          [
            ('', '<xml>'),
            (' with some text ', '<tag param1="AA" param2="BB">'),
            ('some more ', '<smalltag/>'),
            (' text', '</tag></xml>')
          ]

    - attributes_as_text=True: attribute *values* are surfaced as `text` items,
      while the surrounding tag syntax (including names, =, spaces, quotes, and >)
      stays on the `xml` side in between those values.
      Example (same s):
        [
          ('', '<xml>'),
          (' with some text ', '<tag param1="'),
          ('AA', '" param2="'),
          ('BB', '">'),
          ('some more ', '<smalltag/>'),
          (' text', '</tag></xml>')
        ]

    Notes
    -----
    * This is a lightweight tokenizer, not a validating XML parser.
    * It preserves the original bytes/characters (attribute order, spacing, quotes).
    * It treats comments (<!-- -->), CDATA (<![CDATA[ ]]>), DOCTYPE, and processing
      instructions (<? ?>) as indivisible tag-like chunks (never split attributes).
    """

    # Matches things we must treat as single “tag-like” chunks:
    _comment_start = "<!--"
    _cdata_start = "<![CDATA["
    _doctype_start = "<!DOCTYPE"
    _pi_start = "<?"

    def __init__(self, xml: str, attributes_as_text: bool = False):
        self.xml = xml
        self.attributes_as_text = attributes_as_text

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        s = self.xml
        n = len(s)
        i = 0
        pending_text = []

        def flush_pair(xml_chunk: str):
            nonlocal pending_text
            text_str = ''.join(pending_text)
            pending_text = []
            return (text_str, xml_chunk)

        while i < n:
            if s[i] != '<':
                # Accumulate text until next '<' or end
                j = s.find('<', i)
                if j == -1:
                    pending_text.append(s[i:])
                    i = n
                    break
                pending_text.append(s[i:j])
                i = j
                continue

            # Handle special sections (treat as indivisible)
            if s.startswith(self._comment_start, i):
                j = s.find("-->", i + 4)
                end = n if j == -1 else j + 3
                yield flush_pair(s[i:end])
                i = end
                continue

            if s.startswith(self._cdata_start, i):
                j = s.find("]]>", i + 9)
                end = n if j == -1 else j + 3
                yield flush_pair(s[i:end])
                i = end
                continue

            if s.startswith(self._doctype_start, i) or s.startswith(self._pi_start, i):
                # Find the next '>' that isn't inside quotes
                end = self._scan_tag_end(s, i)
                yield flush_pair(s[i:end])
                i = end
                continue

            # Regular tag (start, empty, or end tag). Grab full tag first.
            end = self._scan_tag_end(s, i)
            tag = s[i:end]

            # Closing tags and empty tags have no attribute values to split.
            is_closing = tag.startswith("</")
            if is_closing or (not self.attributes_as_text):
                yield flush_pair(tag)
                i = end
                continue

            # If self-closing or start tag: optionally split attribute *values* out.
            if self.attributes_as_text and not is_closing:
                # Build alternating [xml, text, xml, text, ..., xml] pieces for this tag
                pieces = self._split_attribute_values_as_pieces(tag)

                # Now emit as (text, xml) pairs:
                # first pair uses pending_text + pieces[0] (xml)
                yield flush_pair(pieces[0])
                # subsequent pairs pair pieces[1] (text) with pieces[2] (xml), etc.
                for k in range(1, len(pieces), 2):
                    text_seg = pieces[k]
                    xml_seg = pieces[k + 1] if k + 1 < len(pieces) else ""
                    # here pending_text is empty by construction
                    yield (text_seg, xml_seg)

                i = end
                continue

        # If there’s trailing text with no following tag, pair it with empty xml
        if pending_text:
            yield (''.join(pending_text), "")

    @staticmethod
    def _scan_tag_end(s: str, start: int) -> int:
        """
        Starting at '<', find the index just after the matching '>' while
        respecting quoted strings.
        """
        assert s[start] == '<'
        i = start + 1
        n = len(s)
        quote = None
        while i < n:
            c = s[i]
            if quote:
                if c == quote:
                    quote = None
            else:
                if c == '"' or c == "'":
                    quote = c
                elif c == '>':
                    return i + 1
            i += 1
        return n  # fallback (malformed: no closing '>')

    @staticmethod
    def _split_attribute_values_as_pieces(tag: str):
        """
        Return a list of alternating pieces starting with XML:
          [xml_before_first_value, value1_text, xml_between, value2_text, xml_after, ...]
        Only considers quoted attribute values inside a tag that starts with '<' and
        is not a closing tag.
        """
        pieces = []
        i = 0
        n = len(tag)

        # If it's a closing tag, just return as one piece.
        if tag.startswith("</"):
            return [tag]

        # Scan for quoted attribute values. We only treat content inside quotes as “text”.
        while i < n:
            if tag[i] == '"':
                # not expected at start unless malformed; treat as normal quoted value
                pass
            if tag[i] == "'":
                pass

            # Find next opening quote that is part of an attribute assignment.
            # We simply search for next quote and accept it as opening; then find its mate.
            open_pos = XMLTextIterator._find_next_quote(tag, i)
            if open_pos == -1:
                # No more values; append remainder as xml
                pieces.append(tag[i:])
                break

            # Emit xml up to and including the opening quote
            pieces.append(tag[i:open_pos + 1])

            # Find matching closing quote of the same kind
            q = tag[open_pos]
            close_pos = tag.find(q, open_pos + 1)
            if close_pos == -1:
                # Malformed: no closing quote; take rest as text then no trailing xml
                pieces.append(tag[open_pos + 1:])
                break

            # Emit the value (text segment)
            pieces.append(tag[open_pos + 1:close_pos])

            # Continue after the closing quote
            i = close_pos
            # The next loop iteration will append further xml/value alternations
        # Ensure we end on an XML piece (append empty if needed)
        if len(pieces) % 2 == 0:
            pieces.append("")
        return pieces

    @staticmethod
    def _find_next_quote(s: str, start: int) -> int:
        """
        Find the next quote character (single or double) from 'start'.
        We don't try to validate that it's after '=', because we only
        care about preserving exact bytes and splitting on quoted regions.
        """
        dq = s.find('"', start)
        sq = s.find("'", start)
        if dq == -1:
            return sq
        if sq == -1:
            return dq
        return min(dq, sq)



class TextExtractor():
    def __init__(self, input_path: Union[str, Path], output_path: Union[None, str, Path], write_mode: Literal['rewrite','w', 'a'] = 'w', create_output_dirs=False):
        if input_path == "stdin":
            self.input_f = sys.stdin
        elif Path(input_path).is_file():
            self.input_f = open(str(input_path), 'r', encoding='utf-8')
        else:
            raise IOError(f"Invalid input path: {input_path}")

        if output_path is None:
            self.output_f = None
        elif output_path == "stdout":
            self.output_f = sys.stdout
        elif output_path == "stderr":
            self.output_f = sys.stderr
        elif write_mode== 'w' and not Path(output_path).is_file(): # write mode does not allow overwrite
            try:
                self.output_f = open(str(output_path), "w", encoding='utf-8')
            except FileNotFoundError:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                self.output_f = open(str(output_path), "w", encoding='utf-8')
        elif write_mode == 'a':  # append mode allows creating file if not exists
            self.output_f = open(str(output_path), "a", encoding='utf-8')
        elif write_mode== 'rewrite': # write mode does not allow overwrite
            try:
                self.output_f = open(str(output_path), "w", encoding='utf-8')
            except FileNotFoundError:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                self.output_f = open(str(output_path), "w", encoding='utf-8')
        else:
            self.output_f = None



# --- quick demo ---
if __name__ == "__main__":
    s = '<xml> with some text <tag param1="AA" param2="BB">some more <smalltag/> text</tag></xml>'

    print("attributes_as_text=False")
    out0 = list(XMLTextIterator(s, attributes_as_text=False))
    print(out0)
    print("reconstructed ok:", ''.join(t + x for t, x in out0) == s)

    print("\nattributes_as_text=True")
    out1 = list(XMLTextIterator(s, attributes_as_text=True))
    print(out1)
    print("reconstructed ok:", ''.join(t + x for t, x in out1) == s)
