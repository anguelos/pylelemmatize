from typing import Dict, Literal, Optional
from .char_distance import char_similarity
from .main_scripts import main_alphabet_extract_corpus_alphabet, main_alphabet_evaluate_merges
from .abstract_mapper import GenericLemmatizer, fast_cer, fast_numpy_to_str, fast_str_to_numpy
from .fast_mapper import LemmatizerBMP
#from .charset import allbmp_encoding_alphabet_strings, allnonbmp_encoding_alphabet_strings, main_map_test_corpus_on_alphabets

#from .charset import chr_iso, chr_mes, chr_ascii, chr_mufi_bmp, chr_mufi_nonbmp
from .charsets import Charsets
charset = Charsets()

from .util import extract_transcription_from_page_xml, main_extract_transcription_from_page_xml, print_err
import sys
from .philogeny import main_char_similarity_tree


default_unknown_chr = "�"

def create_lemmatizer(src_alphabet_str: str, dst_alphabet_str: Optional[str]=None,
                  mapper_type: Literal["fast", "generic", "guess"] = "guess", unknown_chr: str = default_unknown_chr,
                  override_map: Optional[Dict[str, str]] = None, min_similarity: float = .3) -> GenericLemmatizer:
    if mapper_type == "guess":
        if LemmatizerBMP.alphabet_in_bmp(src_alphabet_str)  and LemmatizerBMP.alphabet_in_bmp(dst_alphabet_str):
            mapper_type = "fast"
        else:
            mapper_type = "generic"
    if mapper_type == "fast":
        return LemmatizerBMP.from_alphabet_mapping(src_alphabet_str, dst_alphabet_str, unknown_chr=unknown_chr,
                                                   override_map=override_map, min_similarity=min_similarity)
    elif mapper_type == "generic":
        return GenericLemmatizer.from_alphabet_mapping(src_alphabet_str, dst_alphabet_str, unknown_chr=unknown_chr, 
                                                       override_map=override_map, min_similarity=min_similarity)
    else:
        raise ValueError(f"Unknown mapper type: {mapper_type}")


if "torch" in sys.modules:
    from .mapper_ds import Seq2SeqDs
    from .demapper_lstm import DemapperLSTM, main_train_one2one, main_infer_one2one, main_report_demapper
else:
    print("Warning: Torch is not loaded. Seq2SeqDs will not be available.", file=sys.stderr)
