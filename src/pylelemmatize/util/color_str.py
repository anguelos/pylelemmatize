from typing import List, Optional, IO, Tuple, Union
import numpy as np
from .banded_editdistance import banded_edit_path
from ..abstract_mapper import fast_str_to_numpy

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
