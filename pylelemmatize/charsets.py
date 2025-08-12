from typing import Dict, List
#from .charset_iso import iso_only_alphabet
#from .charset_mes import mes_only_alphabet
#from .charset_ascii import ascii_only_alphabet
#from .charset_mufi import all_bmp_mufi, all_nonbmp_mufi


from .charset_iso import get_encoding_dicts as get_iso_encoding_dicts
from .charset_mes import get_encoding_dicts as get_mes_encoding_dicts
from .charset_ascii import get_encoding_dicts as get_ascii_encoding_dicts
from .charset_mufi import get_encoding_dicts as get_mufi_encoding_dicts



from .abstract_mapper import  fast_cer, GenericLemmatizer
from .fast_mapper import LemmatizerBMP


class Charsets:
    def __init__(self):
        self.lib_cache__ = {}
        self.alphabets = ["mes1", "mes2", "mes3a", "mes3b",
                     "iso_8859_1", "iso_8859_2", "iso_8859_3", "iso_8859_4",
                     "iso_8859_5", "iso_8859_6", "iso_8859_7", "iso_8859_8",
                     "iso_8859_9", "iso_8859_10", "iso_8859_11", "iso_8859_12", "iso_8859_13",
                     "iso_8859_14", "iso_8859_15", "iso_8859_16",
                     "ascii_lowercase", "ascii_uppercase", "ascii_letters", "digits",
                     "hexdigits", "octdigits", "punctuation",
                     "bmp_mufi", "bmp_pua_mufi", "nonbmp_mufi",
                     "nonbmp_pua_mufi"]

    @property
    def mes_charsets(self):
        if self.lib_cache__.get("mes") is None:
            self.lib_cache__["mes"] = get_mes_encoding_dicts()
        return self.lib_cache__["mes"]
    @property
    def mes1(self):
        return self.mes_charsets["mes1"]
    @property
    def mes2(self):
        return self.mes_charsets["mes2"]
    @property
    def mes3a(self):
        return self.mes_charsets["mes3a"]
    @property
    def mes3b(self):
        return self.mes_charsets["mes3b"]

    @property
    def iso_charsets(self):
        if self.lib_cache__.get("iso") is None:
            self.lib_cache__["iso"] = get_iso_encoding_dicts()
        return self.lib_cache__["iso"]
    @property
    def iso_8859_1(self):
        return self.iso_charsets["iso_8859_1"]
    @property
    def iso_8859_2(self):
        return self.iso_charsets["iso_8859_2"]
    @property
    def iso_8859_3(self):
        return self.iso_charsets["iso_8859_3"]
    @property
    def iso_8859_4(self):
        return self.iso_charsets["iso_8859_4"]
    @property
    def iso_8859_5(self):
        return self.iso_charsets["iso_8859_5"]
    @property
    def iso_8859_6(self):
        return self.iso_charsets["iso_8859_6"]
    @property
    def iso_8859_7(self):
        return self.iso_charsets["iso_8859_7"]
    @property
    def iso_8859_8(self):
        return self.iso_charsets["iso_8859_8"]
    @property
    def iso_8859_9(self):
        return self.iso_charsets["iso_8859_9"]
    @property
    def iso_8859_10(self):
        return self.iso_charsets["iso_8859_10"]
    @property
    def iso_8859_11(self):  
        return self.iso_charsets["iso_8859_11"]
    @property
    def iso_8859_12(self):
        return self.iso_charsets["iso_8859_12"]
    @property
    def iso_8859_13(self):
        return self.iso_charsets["iso_8859_13"]
    @property
    def iso_8859_14(self):
        return self.iso_charsets["iso_8859_14"]
    @property
    def iso_8859_15(self):
        return self.iso_charsets["iso_8859_15"]
    @property
    def iso_8859_16(self):
        return self.iso_charsets["iso_8859_16"]

    @property
    def ascii_charsets(self):
        if self.lib_cache__.get("ascii") is None:
            self.lib_cache__["ascii"] = get_ascii_encoding_dicts()
        return self.lib_cache__["ascii"]
    @property
    def ascii(self):
        return self.ascii_charsets["ascii"]
    @property
    def ascii_lowercase(self):
        return self.ascii_charsets["ascii_lowercase"]
    @property
    def ascii_uppercase(self):
        return self.ascii_charsets["ascii_uppercase"]
    @property
    def ascii_letters(self):
        return self.ascii_charsets["ascii_letters"]
    @property
    def digits(self):
        return self.ascii_charsets["digits"]
    @property
    def hexdigits(self):
        return self.ascii_charsets["hexdigits"]
    @property
    def octdigits(self):
        return self.ascii_charsets["octdigits"]
    @property
    def punctuation(self):
        return self.ascii_charsets["punctuation"]

    @property
    def mufibmp_charsets(self):
        if self.lib_cache__.get("mufibmp") is None:
            self.lib_cache__["mufibmp"], self.lib_cache__["mufinonbmp"] = get_mufi_encoding_dicts()
        return self.lib_cache__["mufibmp"]
    @property
    def mufinonbmp_charsets(self):
        if self.lib_cache__.get("mufinonbmp") is None:
            self.lib_cache__["mufibmp"], self.lib_cache__["mufinonbmp"] = get_mufi_encoding_dicts()
        return self.lib_cache__["mufinonbmp"]
    @property
    def mufibmp(self):
        return self.mufibmp_charsets["bmp_mufi"]
    @property
    def mufibmp_pua(self):
        return self.mufibmp_charsets["bmp_pua_mufi"]
    @property
    def mufinonbmp(self):
        return self.mufinonbmp_charsets["nonbmp_mufi"]
    @property
    def mufinonbmp_pua(self):
        return self.mufinonbmp_charsets["nonbmp_pua_mufi"]

    def __getitem__(self, item):
        return getattr(self, item)
    
    def __iter__(self):
        return iter(self.alphabets)

    def __len__(self):
        return len(self.alphabets)
    
    def __contains__(self, item):
        return item in self.alphabets
    
    def keys(self):
        return self.alphabets
    
    def items(self):
        return [(k, getattr(self, k)) for k in self.alphabets]


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
