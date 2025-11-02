from typing import Dict, Literal, Optional
from .char_distance import char_similarity
from .main_scripts import main_alphabet_extract_corpus_alphabet, main_alphabet_evaluate_merges
from .abstract_mapper import GenericLemmatizer, char_similarity

from .fast_mapper import LemmatizerBMP
from .charsets import Charsets as charset

from .util import extract_transcription_from_page_xml, main_extract_transcription_from_page_xml
import sys
from .philogeny import main_char_similarity_tree


default_unknown_chr = "�"

def create_mapper(input_alphabet: str, output_alphabet: Optional[str]=None,
                  mapper_type: Literal["fast", "generic", "guess"] = "guess", unknown_chr: str = default_unknown_chr,
                  override_map: Optional[Dict[str, str]] = None) -> GenericLemmatizer:
    if mapper_type == "guess":
        if LemmatizerBMP.alphabet_in_bmp(input_alphabet)  and LemmatizerBMP.alphabet_in_bmp(output_alphabet):
            mapper_type = "fast"
        else:
            mapper_type = "generic"
    if mapper_type == "fast":
        return LemmatizerBMP.from_alphabet_mapping(input_alphabet, output_alphabet, unknown_chr=unknown_chr, override_map=override_map)
    elif mapper_type == "generic":
        return GenericLemmatizer.from_alphabet_mapping(input_alphabet, output_alphabet, unknown_chr=unknown_chr, override_map=override_map)
    else:
        raise ValueError(f"Unknown mapper type: {mapper_type}")


from .mapper_ds import Seq2SeqDs
from .demapper_lstm import DemapperLSTM, main_train_one2one, main_infer_one2one, main_report_demapper
