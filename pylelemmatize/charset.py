from typing import Dict, List
from .charset_iso import iso_only_alphabet
from .charset_mes import mes_only_alphabet
from .charset_ascii import ascii_only_alphabet
from .charset_mufi import all_bmp_mufi, all_nonbmp_mufi


from .abstract_mapper import  fast_cer, GenericLemmatizer
from .fast_mapper import LemmatizerBMP

family_to_charmapnames = {}


allbmp_encoding_alphabet_strings = {}
allbmp_encoding_alphabet_strings.update(iso_only_alphabet)
allbmp_encoding_alphabet_strings.update(mes_only_alphabet)
allbmp_encoding_alphabet_strings.update(ascii_only_alphabet)
allbmp_encoding_alphabet_strings.update(all_bmp_mufi)

#nonbmp means the union of bmp and non-bmp characters.
allnonbmp_encoding_alphabet_strings = {k:v for k, v in allbmp_encoding_alphabet_strings.items()}
allnonbmp_encoding_alphabet_strings.update(all_nonbmp_mufi)


to_be_exported = {f"chr_{k}": v for k, v in allbmp_encoding_alphabet_strings.items()}
globals().update(to_be_exported)


def main_map_test_corpus_on_alphabets():
    import fargv, glob, time, sys, tqdm
    from matplotlib import pyplot as plt
    from .util import generate_corpus

    def plot_covverage(corpus_str, encoding_alphabet_strings, alphabet_to_cer: Dict[str, float], alphabet_to_missing: Dict[str, str], alphabet_to_unfound: Dict[str, str], save_plot_path: str, show_plot: bool):
        fig, ax = plt.subplots(3, 1, figsize=(15, 10))
        alphabet_to_missing_ratio = {k: 100*len(v)/len(corpus_str) for k, v in alphabet_to_missing.items() if len(v) > 0}
        alphabet_to_unfound_ratio = {k: len(v) for k, v in alphabet_to_unfound.items() if len(v) > 0}
        ax[0].bar(alphabet_to_cer.keys(), [100*v for v in alphabet_to_cer.values()])
        ax[0].set_xticklabels(alphabet_to_cer.keys(), rotation=45)
        ax[0].set_title("CER %")
        
        ax[1].bar(alphabet_to_missing_ratio.keys(), alphabet_to_missing_ratio.values())
        ax[1].set_xticklabels(alphabet_to_missing_ratio.keys(), rotation=45)
        ax[1].set_title("Character Recall %")

        ax[2].bar(alphabet_to_unfound_ratio.keys(), alphabet_to_unfound_ratio.values())
        ax[2].set_xticklabels(alphabet_to_unfound_ratio.keys(), rotation=45)
        ax[2].set_yscale('log')
        ax[2].set_title("Charset uncovered chars # (waste)")
        
        plt.subplots_adjust(hspace=.6)
        if save_plot_path != "":
            plt.savefig(save_plot_path)
        if show_plot:
            plt.show()

    t = time.time()
    p = {
        "alphabets": ["ascii", f"A comma separated listof encodings must be a subset of {repr(allbmp_encoding_alphabet_strings.keys())}. Or 'all'"],
        "corpus_glob": "",
        "corpus_files": set([]),
        "verbose": False,
        "strip_xml": True,
        "all_is_xml": False,
        "output_tsv": "",
        "hide_plot": False,
        "save_plot_path": ""
    }
    args, _ = fargv.fargv(p)

    if args.alphabets == "all":
        args.alphabets = list(allbmp_encoding_alphabet_strings.keys())
    else:
        args.alphabets = [a.strip() for a in args.alphabets.split(',')]

    if args.corpus_glob != "":
        glob_files = set(glob.glob(args.corpus_glob))
    else:
        glob_files = set([])
    if len(args.corpus_files) > 0:
        corpus_files = args.corpus_files
    else:
        corpus_files = set([])

    all_corpus_strs = []
    for file_contnets in generate_corpus(glob_files | corpus_files, verbose=args.verbose,
                                         strip_xml=args.strip_xml, treat_all_file_as_xml=args.all_is_xml):
        all_corpus_strs.append(file_contnets)
    found_corpus_str = ''.join(all_corpus_strs)

    if args.verbose:
        print(f"{time.time() - t :.2f}: Loaded corpus", file=sys.stderr)

    if args.verbose:
        alphabet_names = tqdm.tqdm(args.alphabets)
    else:
        alphabet_names = args.alphabets

    alphabet_to_cer = {}
    alphabet_to_missing = {}
    alphabet_to_unfound = {}
    for alphabet_name in alphabet_names:
        if alphabet_name not in allbmp_encoding_alphabet_strings:
            raise ValueError(f"Alphabet {alphabet_name} not in {allbmp_encoding_alphabet_strings.keys()}")

        alphabet_str = allbmp_encoding_alphabet_strings[alphabet_name]
        if max(alphabet_str) > '\uffff':
            #alphabet = Alphabet(alphabet_str=alphabet_str, vectorized_mapper_sz=max(256**2, 1 + ord(max(alphabet_str))))  #  This is a hack to ensure that the vectorized mapper is large enough for all characters in MUFI
            alphabet = GenericLemmatizer.from_alphabet_mapping(alphabet_str, mapping_dict=None)
        else:
            #alphabet = AlphabetBMP(alphabet_str=alphabet_str)  #  This is a hack to ensure that the vectorized mapper is large enough for all characters in MUFI
            alphabet = LemmatizerBMP.from_alphabet_mapping(alphabet_str)

        mapped_corpus_str = alphabet(found_corpus_str)

        if len(found_corpus_str) > 0:
            alphabet_to_cer[alphabet_name] = fast_cer(found_corpus_str, mapped_corpus_str)
            alphabet_to_missing[alphabet_name] = ''.join(sorted(set(found_corpus_str) - set(alphabet_str)))
            alphabet_to_unfound[alphabet_name] = ''.join(sorted(set(alphabet_str) - set(found_corpus_str)))

        else:
            alphabet_to_cer[alphabet_name] = 0
            alphabet_to_missing[alphabet_name] = ""
            alphabet_to_unfound[alphabet_name] = alphabet_str

    if args.verbose:
        for k, v in alphabet_to_cer.items():
            print(f"{k}: CER {100*v:.2f}% Missing: {len(alphabet_to_missing[k])}, Redundant: {len(alphabet_to_unfound[k])}  {repr(alphabet_to_missing[k])}", file=sys.stderr)
        print(f"Computed all for {len(found_corpus_str)} characters in {time.time() -t :.2f}", file=sys.stderr)

    if not args.hide_plot or args.save_plot_path != "":
        found_corpus_alphabet_str = ''.join(sorted(set(found_corpus_str)))
        plot_covverage(found_corpus_alphabet_str, allbmp_encoding_alphabet_strings, alphabet_to_cer, alphabet_to_missing, alphabet_to_unfound, args.save_plot_path, not args.hide_plot)
