import sys
import fargv
from typing import Dict, Tuple, Union, Optional
from collections import defaultdict
import numpy as np
from .abstract_mapper import AbstractLemmatizer, fast_str_to_numpy, fast_numpy_to_str, fast_cer, GenericLemmatizer
from .fast_mapper import LemmatizerBMP



def main_alphabet_extract_corpus_alphabet():
    import fargv, glob, time
    from .util import generate_corpus
    from unidecode import unidecode

    t = time.time()
    p = {
        "corpus_glob": "",
        "corpus_files": set([]),
        "dont_count_alphabet": False,
        "dont_show_alphabet": False,
        "dont_show_histogram": False,
        "verbose": False,
        "strip_xml": True,
        "all_is_xml": False,
        "unidecode": False,
        "output_tsv": ""
    }
    args, _ = fargv.fargv(p)

    if args.corpus_glob != "":
        glob_files = set(glob.glob(args.corpus_glob))
    else:
        glob_files = set([])
    if len(args.corpus_files) > 0:
        corpus_files = args.corpus_files
    else:
        corpus_files = set([])

    total_size = 0
    all_alphabets_strs = []
    all_corpus_strs = []
    for file_contnets in generate_corpus(glob_files | corpus_files, verbose=args.verbose,
                                         strip_xml=args.strip_xml, treat_all_file_as_xml=args.all_is_xml):
        total_size += len(file_contnets)
        all_corpus_strs.append(file_contnets)
        all_alphabets_strs.append(LemmatizerBMP.fast_alphabet_extraction(file_contnets))
    found_alphabet_str = ''.join(sorted(set(''.join(all_alphabets_strs))))

    if not args.dont_show_alphabet:
        if args.output_tsv == "stdout":
            print(found_alphabet_str, file=sys.stderr)
        else:
            print(found_alphabet_str)

    if args.output_tsv != "":
        tsv_str = LemmatizerBMP(alphabet_str=found_alphabet_str).alphabet_tsv
        if args.output_tsv == "stdout":
            print(tsv_str)
        elif args.output_tsv == "stderr":
            print(tsv_str, file=sys.stderr)
        else:
            with open(args.output_tsv, "w") as f:
                f.write(tsv_str)

    if not args.dont_count_alphabet:
        print(f"Alphabet Length: {len(found_alphabet_str)}", file=sys.stderr)

    if not args.dont_show_histogram:
        ht = time.time()
        corpus_str = ''.join(all_corpus_strs)
        nums, freqs, names = GenericLemmatizer.from_alphabet_str(alphabet_str=found_alphabet_str).get_unigram(corpus_str)
        most_frequent = np.argsort(freqs)
        print(f"\nUnigram model in reverced frequencies:", file=sys.stderr)
        for n in most_frequent:
            if args.unidecode:
                try:                    
                    udc_ord = ord(unidecode(names[n]))
                except Exception as e:
                    print(f"Error occurred while decoding {unidecode(names[n])}: {e}", file=sys.stderr)
                    udc_ord = -1
                print(f"CHR: {repr(names[n])}, # {ord(names[n])}, UDC: {repr(unidecode(names[n]))}, UDC # {udc_ord}: {freqs[n]}", file=sys.stderr)
            else:
                print(f"{repr(names[n])}, {ord(names[n])}: {freqs[n]}", file=sys.stderr)
        print(f"Computed Histogram for {len(corpus_str)} in {time.time() - ht :.2f}", file=sys.stderr)

    if args.verbose:
        print(f"Computed {total_size} in {time.time() -t :.2f}", file=sys.stderr)


def main_alphabet_evaluate_merges():
    import fargv, glob, time
    from .util import generate_corpus

    t = time.time()
    p = {
        "corpus_glob": "",
        "corpus_files": set([]),
        "merges": "[('u', 'v'),  ('U', 'V')]",
        "verbose": False,
        "strip_xml": True,
        "all_is_xml": False,
        "output_tsv": ""
    }
    args, _ = fargv.fargv(p)

    if args.corpus_glob != "":
        glob_files = set(glob.glob(args.corpus_glob))
    else:
        glob_files = set([])
    if len(args.corpus_files) > 0:
        corpus_files = args.corpus_files
    else:
        corpus_files = set([])

    total_size = 0
    all_alphabets_strs = []
    all_corpus_strs = []
    for file_contnets in generate_corpus(glob_files | corpus_files, verbose=args.verbose,
                                         strip_xml=args.strip_xml, treat_all_file_as_xml=args.all_is_xml):
        total_size += len(file_contnets)
        all_corpus_strs.append(file_contnets)
        all_alphabets_strs.append(AbstractLemmatizer.fast_alphabet_extraction(file_contnets))
    found_alphabet_str = ''.join(sorted(set(''.join(all_alphabets_strs))))
    corpus_str = ''.join(all_corpus_strs)

    mapper = {k: k for k in found_alphabet_str}
    mapper.update(eval(args.merges))
    mapped_corpus_str = ''.join([mapper[c] for c in corpus_str])
    print(f"Mapping CER {fast_cer(corpus_str, mapped_corpus_str)}", file=sys.stderr)

    if args.verbose:
        print(f"Computed {total_size} in {time.time() -t :.2f}", file=sys.stderr)
