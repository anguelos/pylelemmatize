from .main_scripts import main_alphabet_extract_corpus_alphabet, main_alphabet_evaluate_merges

from .fast_mapper import LemmatizerBMP
from .charset import allbmp_encoding_alphabet_strings, allnonbmp_encoding_alphabet_strings, main_map_test_corpus_on_alphabets
#from .charset import chr_iso, chr_mes, chr_ascii, chr_mufi_bmp, chr_mufi_nonbmp
from .util import extract_transcription_from_page_xml, main_extract_transcription_from_page_xml
import sys


try:
    import torch
    have_torch = True
except ImportError as e:
    have_torch = False

if have_torch:
    from .mapper_ds import Seq2SeqDs
    from .demapper_lstm import DemapperLSTM, main_train_one2one, main_infer_one2one
else:
    print("Warning: Torch is not installed. Seq2SeqDs will not be available.", file=sys.stderr)