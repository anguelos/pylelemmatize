import re
from typing import IO, Generator, Literal, Optional, Set, Union, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import tqdm
import sys
import unicodedata
from lxml import etree
from math import inf
import xml.etree.ElementTree as ET

from pylelemmatize.abstract_mapper import fast_str_to_numpy


# Utility code from many to more START

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
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
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

# Utility code from many to more END


def banded_edit_path(a: np.ndarray, b: np.ndarray, band: int, add_null_initial: bool = True) -> Tuple[np.ndarray, np.ndarray, Tuple[np.array, np.array, np.array, np.array]]:
    """
    Banded dynamic-programming alignment (edit distance with unit costs).
    
    Args:
        a: numpy array of single-character strings, shape (m,)
        b: numpy array of single-character strings, shape (n,)
        band: non-negative int, maximum |i - j| misalignment allowed
    
    Returns:
        path: A numpy array of (N+1, 2) containing the coordiantes of the optimal path with respect to a and b.
            The array begins with coordinates [-1, -1] to insdicate START.
        costs: A numpy array of the cumulative costs along the path.
        mat_ins_del_sub: Tuple of boolean arrays
    
    Raises:
        ValueError if alignment is impossible given the band (e.g., |m - n| > band).
    
    Notes:
        - Costs: match = 0, substitution = 1, insertion = 1, deletion = 1.
        - Memory: O((m+n)*band). Time: ~O((m+n)*band).
        - The returned path walks *cells* (i,j) of the DP grid (length m+n minus matches).
    """
    # Backpointer codes
    DIAG, UP, LEFT = 0, 1, 2
    m, n = len(a), len(b)
    if band < 0:
        raise ValueError("band must be non-negative")
    if abs(m - n) > band:
        # End point (m,n) lies outside the band reachable region
        raise ValueError(f"Strings differ in length by {abs(m-n)}, larger than band {band}.")

    # Per row storage (j-starts, costs, backpointers)
    row_starts: List[int] = []
    row_costs: List[np.ndarray] = []
    row_bp: List[np.ndarray] = []

    # Helper to allocate a row segment covering j in [jmin, jmax]
    def alloc_row(jmin: int, jmax: int):
        width = jmax - jmin + 1
        return (jmin, np.full(width, inf, dtype=float), np.full(width, -1, dtype=np.int8))

    # Row 0 initialization (i = 0): we can only do insertions
    jmin0 = max(0, 0 - band)
    jmax0 = min(n, 0 + band)
    j0, cost0, bp0 = alloc_row(jmin0, jmax0)
    # DP[0, j] = j, along the top row (only if within band)
    for j in range(j0, jmax0 + 1):
        cost0[j - j0] = j
        bp0[j - j0] = LEFT if j > 0 else -1  # from (0,j-1) unless at origin
    row_starts.append(j0); row_costs.append(cost0); row_bp.append(bp0)

    # Fill subsequent rows
    for i in range(1, m + 1):
        # Band-limited j-range for row i
        jmin = max(0, i - band)
        jmax = min(n, i + band)
        j_start, cost_row, bp_row = alloc_row(jmin, jmax)

        # Previous row context
        prev_start = row_starts[i - 1]
        prev_cost = row_costs[i - 1]

        for j in range(jmin, jmax + 1):
            k = j - j_start  # index in current row

            # Candidates: deletion (up), insertion (left), substitution/match (diag)
            best_cost = inf
            best_bp = -1

            # UP: from (i-1, j) if that col exists in prev row
            if prev_start <= j <= prev_start + len(prev_cost) - 1:
                c_up = prev_cost[j - prev_start] + 1  # deletion from a
                if c_up < best_cost:
                    best_cost, best_bp = c_up, UP

            # LEFT: from (i, j-1) if j-1 in current row band
            if j - 1 >= j_start:
                c_left = cost_row[k - 1] + 1  # insertion into a (gap in a)
                if c_left < best_cost:
                    best_cost, best_bp = c_left, LEFT

            # DIAG: from (i-1, j-1) if that exists in prev row
            if prev_start <= j - 1 <= prev_start + len(prev_cost) - 1:
                sub = 0 if a[i - 1] == b[j - 1] else 1
                c_diag = prev_cost[(j - 1) - prev_start] + sub
                if c_diag < best_cost:
                    best_cost, best_bp = c_diag, DIAG

            cost_row[k] = best_cost
            bp_row[k] = best_bp

        row_starts.append(j_start); row_costs.append(cost_row); row_bp.append(bp_row)

    # Verify end cell is inside band and finite
    if not (row_starts[m] <= n <= row_starts[m] + len(row_costs[m]) - 1):
        raise ValueError("End cell (m,n) falls outside the band — increase band.")
    end_idx = n - row_starts[m]
    if not np.isfinite(row_costs[m][end_idx]):
        raise ValueError("No valid path within band — increase band.")

    # Traceback from (m, n) to (0, 0)
    path: List[Tuple[int, int]] = []
    i, j = m, n
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        j_start = row_starts[i]
        k = j - j_start
        if k < 0 or k >= len(row_bp[i]):
            # If we ever step just outside due to band edge, it's invalid
            raise RuntimeError("Traceback stepped outside band; increase band.")
        bp = row_bp[i][k]
        if bp == DIAG:
            i, j = i - 1, j - 1
        elif bp == UP:
            i, j = i - 1, j
        elif bp == LEFT:
            i, j = i, j - 1
        else:
            raise RuntimeError("Invalid backpointer during traceback.")
    path.reverse()
    paths = np.array(path) - 1
    is_ins = paths[1:, 0] - paths[: -1, 0] == 0
    is_del = paths[1:, 1] - paths[: -1, 1] == 0
    is_diag = ~(is_del | is_ins)
    agree =  (a[paths[1:, 0]] == b[paths[1:, 1]])
    is_sub = is_diag & ~agree
    is_match = is_diag & agree
    cost_1 = is_sub | is_del | is_ins
    return paths, cost_1.astype(np.int16), (is_match, is_ins, is_del, is_sub)


def compute_cer(seq1: Union[np.ndarray, str], seq2: Union[np.ndarray, str], band=-1, normalize=False) -> float:
    if isinstance(seq1, str):
        seq1 = fast_str_to_numpy(seq1)
    if isinstance(seq2, str):
        seq2 = fast_str_to_numpy(seq2)
    if band<1:
        band = max(seq1.size, seq2.size)
    _, costs, (_, _, _, _) = banded_edit_path(seq1, seq2, band=band)
    if normalize:
        return float(costs.sum()) / max(len(seq1), len(seq2))
    else:
        return float(costs.sum())


def fast_extract_text_from_xml(xml_string: str, concatenate: bool = True) -> Union[str, List[str]]:
    # Regular expression to find text within tags
    # This regex avoids capturing empty spaces between tags and ensures capturing text
    text_parts = re.findall(r'>\s*([^<>]+?)\s*<', xml_string)
    if concatenate:
        return ' '.join(text_parts)
    else:
        return text_parts


def print_err(txt: str ="Hello", correct: Optional[List[bool]]=None, confidence: Optional[List[float]]=None, file: Optional[IO[str]]=None) -> str:
    """
    Print text to stderr with color coding based on correctness and confidence.

    Each character in the input text is colorized using ANSI escape codes. The foreground
    color is green for correct characters and red for incorrect ones. The background color
    interpolates from black (high confidence) to white (low confidence).

    Parameters
    ----------
    txt : str, optional
        The text to be printed. Defaults to "Hello".
    correct : list of bool, optional
        A list indicating whether each character in `txt` is correct (True) or incorrect (False).
        If None, all characters are assumed to be correct. Defaults to None.
    confidence : list of float, optional
        A list of confidence values (between 0.0 and 1.0) for each character in `txt`.
        A value of 1.0 corresponds to high confidence (black background), and 0.0 corresponds
        to low confidence (white background). If None, all characters are assigned a confidence
        of 1.0. Defaults to None.
    file : file-like object, optional
        A file-like object to which the output will be written. If None, the output is printed
        to the standard error. Defaults to None.

    Notes
    -----
    This function uses ANSI escape codes for colorization, which may not be supported in all
    terminal environments.

    Examples
    --------
    >>> print_err("Test", correct=[True, False, True, True], confidence=[1.0, 0.5, 0.8, 1.0])
    (Outputs colorized text to the terminal)
    """
    def interpolate_color(c1, c2, t):
        """Linearly interpolate between two RGB colors c1 and c2 by t (0 to 1)."""
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def __colorize_char(char, correct=True, confidence=1.0):
        """
        Return a string with ANSI escape codes for a single character.
        Green foreground if correct, red if incorrect.
        Background from black (conf=1.0) to white (conf=0.0).
        """
        # Foreground colors
        fg = (0, 255, 0) if correct else (255, 0, 0)  # green or red

        # Background interpolation: black (1.0) → white (0.0)
        bg = interpolate_color((0, 0, 0), (255, 255, 255), 1 - confidence)

        return f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m{char}\x1b[0m"

    if correct is None:
        correct = [True] * len(txt)
    if confidence is None:
        confidence = [1.0] * len(txt)

    output = ''
    for c, corr, conf in zip(txt, correct, confidence):
        output += __colorize_char(c, corr, conf)
    if file is not None:
        print(output, file=file)
    return output


def print_colored_text(seq: Union[str, np.array],
                       fg_rgb: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None, 
                       bg_rgb: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None,
                       file: Optional[IO[str]] = None) -> str:
    if fg_rgb is None:
        fg_rgb = (255, 255, 255)
    if bg_rgb is None:
        bg_rgb = (0, 0, 0)
    if isinstance(fg_rgb, Tuple) and len(fg_rgb) == 3 and all(isinstance(c, int) for c in fg_rgb):
        fg_rgb = [fg_rgb] * len(seq)
    else:
        assert len(fg_rgb) == len(seq), "fg_rgb length must match sequence length"

    if isinstance(bg_rgb, Tuple) and len(bg_rgb) == 3 and all(isinstance(c, int) for c in bg_rgb):
        bg_rgb = [bg_rgb] * len(seq)
    else:
        assert len(bg_rgb) == len(seq), "bg_rgb length must match sequence length"
    res = []
    for n, character in enumerate(seq):
        fg = fg_rgb[n]
        bg = bg_rgb[n]
        colored_char = f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m{character}\x1b[0m"
        res.append(colored_char)
    res = ''.join(res)
    if file is not None:
        print(res, file=file)
    return res


def print_error_types(seq1: Union[str, np.array], seq2: Union[str, np.array], band: int = -1, 
                      correct_rgb: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((0, 255, 0), (0, 0, 0)), 
                      substitution_rgb: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((255, 0, 0), (0, 0, 0)),
                      insertion_rgb: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((255, 221, 0), (64, 64, 64)),
                      deletion_rgb: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((255, 0, 221), (64, 64, 64)), 
                      file: Optional[IO[str]] = None) -> str:
    """
    Visualize alignment errors between two sequences with color-coded output.

    This function computes the alignment between two sequences and prints the aligned text
    with color-coded error types. Matches, substitutions, insertions, and deletions are
    displayed with different foreground and background colors.

    Parameters
    ----------
    seq1 : Union[str, np.array]
        The first sequence (reference).
    seq2 : Union[str, np.array]
        The second sequence (hypothesis).
    band : int, optional
        The maximum allowed misalignment (band) for the dynamic programming alignment.
        If -1, the band is set to the maximum length of the two sequences. Default is -1.
    correct_rgb : Tuple[Tuple[int, int, int], Tuple[int, int, int]], optional
        RGB colors for matches. The first tuple is the foreground color, and the second
        tuple is the background color. Default is ((0, 255, 0), (0, 0, 0)).
    substitution_rgb : Tuple[Tuple[int, int, int], Tuple[int, int, int]], optional
        RGB colors for substitutions. The first tuple is the foreground color, and the second
        tuple is the background color. Default is ((255, 0, 0), (0, 0, 0)).
    insertion_rgb : Tuple[Tuple[int, int, int], Tuple[int, int, int]], optional
        RGB colors for insertions. The first tuple is the foreground color, and the second
        tuple is the background color. Default is ((255, 221, 0), (64, 64, 64)).
    deletion_rgb : Tuple[Tuple[int, int, int], Tuple[int, int, int]], optional
        RGB colors for deletions. The first tuple is the foreground color, and the second
        tuple is the background color. Default is ((255, 0, 221), (64, 64, 64)).
    file : Optional[IO[str]], optional
        A file-like object to which the output will be written. If None, the output is printed
        to the standard output. Default is None.

    Returns
    -------
    str
        The colorized string representation of the alignment.

    Raises
    ------
    RuntimeError
        If an invalid state is encountered during error type processing.

    Notes
    -----
    - This function uses ANSI escape codes for colorization, which may not be supported in all
      terminal environments.
    - The alignment is computed using a banded dynamic programming approach.

    Examples
    --------
    >>> print_error_types("hello", "h3llo", band=3)
    (Outputs colorized text to the terminal)
    """
    if isinstance(seq1, str):
        seq1_arr = fast_str_to_numpy(seq1)
    if isinstance(seq2, str):
        seq2_arr = fast_str_to_numpy(seq2)
    if band < 1:
        band = max(len(seq1_arr), len(seq2_arr))
    
    if len(seq1_arr) == 0 and len(seq2_arr) == 0:
        return print_colored_text("", file=file)
    if len(seq1_arr) == 0:
        fg_colors = [insertion_rgb[0]] * len(seq2_arr)
        bg_colors = [insertion_rgb[1]] * len(seq2_arr)
        return print_colored_text(seq2_arr, fg_rgb=fg_colors, bg_rgb=bg_colors, file=file)
    if len(seq2_arr) == 0:
        fg_colors = [deletion_rgb[0]] * len(seq1_arr)
        bg_colors = [deletion_rgb[1]] * len(seq1_arr)
        return print_colored_text(seq1_arr, fg_rgb=fg_colors, bg_rgb=bg_colors, file=file)
    
    paths, _, (is_match, is_ins, is_del, is_sub) = banded_edit_path(seq1_arr, seq2_arr, band=band)
    fg_colors = []
    bg_colors = []
    txt = []
    #print(f"seq1: {seq1}\nseq2: {seq2}\npaths:\n{paths}\n", file=sys.stderr)
    for n in range(len(paths)-1):
        if is_match[n]:
            fg, bg = correct_rgb
            fg_colors.append(fg)
            bg_colors.append(bg)
            txt.append(seq1[paths[n+1][0]])
        elif is_sub[n]:
            fg, bg = substitution_rgb
            fg_colors.append(fg)
            bg_colors.append(bg)
            txt.append(seq1[paths[n+1][0]])
        elif is_ins[n]:
            fg, bg = insertion_rgb
            fg_colors.append(fg)
            bg_colors.append(bg)
            txt.append(seq2[paths[n+1][1]])
        elif is_del[n]:
            fg, bg = deletion_rgb
            fg_colors.append(fg)
            bg_colors.append(bg)
            txt.append(seq1[paths[n+1][0]])
        else:
            raise RuntimeError("Invalid state in error type printing.")
    txt = ''.join(txt)
    #print(f"{txt} FG:{fg_colors} BG:{bg_colors} Is match{is_match}\n{paths}", file=sys.stderr)
    return print_colored_text(txt, fg_rgb=fg_colors, bg_rgb=bg_colors, file=file)


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
    import xml.etree.ElementTree as ET

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
