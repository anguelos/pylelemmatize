from collections import defaultdict
from typing import Dict, List, Set, Tuple
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram

codepage_alphabets = {
'10': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~§°·ÁÂÃÄÅÆÉËÍÎÏÐÓÔÕÖØÚÛÜÝÞßáâãäåæéëíîïðóôõöøúûüýþĀāĄąČčĐđĒēĖėĘęĢģĨĩĪīĮįĶķĸĻļŅņŊŋŌōŠšŦŧŨũŪūŲųŽž―�' ,
'11': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืฺุู฿เแโใไๅๆ็่้๊๋์ํ๎๏๐๑๒๓๔๕๖๗๘๙๚๛�' ,
'13': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~¢£¤¦§©«¬®°±²³µ¶·¹»¼½¾ÄÅÆÉÓÕÖ×ØÜßäåæéóõö÷øüĀāĄąĆćČčĒēĖėĘęĢģĪīĮįĶķĻļŁłŃńŅņŌōŖŗŚśŠšŪūŲųŹźŻżŽž’“”„�' ,
'14': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~£§©®¶ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝßàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿĊċĠġŴŵŶŷŸḂḃḊḋḞḟṀṁṖṗṠṡṪṫẀẁẂẃẄẅỲỳ�' ,
'15': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¥§©ª«¬®¯°±²³µ¶·¹º»¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿŒœŠšŸŽž€�' ,
'16': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~§©«°±¶·»ÀÁÂÄÆÇÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜßàáâäæçèéêëìíîïòóôöùúûüÿĂăĄąĆćČčĐđĘęŁłŃńŐőŒœŚśŠšŰűŸŹźŻżŽžȘșȚț”„€�' ,
'2': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~¤§¨°´¸ÁÂÄÇÉËÍÎÓÔÖ×ÚÜÝßáâäçéëíîóôö÷úüýĂăĄąĆćČčĎďĐđĘęĚěĹĺĽľŁłŃńŇňŐőŔŕŘřŚśŞşŠšŢţŤťŮůŰűŹźŻżŽžˇ˘˙˛˝�' ,
'3': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~£¤§¨°²³´µ·¸½ÀÁÂÄÇÈÉÊËÌÍÎÏÑÒÓÔÖ×ÙÚÛÜßàáâäçèéêëìíîïñòóôö÷ùúûüĈĉĊċĜĝĞğĠġĤĥĦħİıĴĵŜŝŞşŬŭŻż˘˙�' ,
'4': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~¤§¨¯°´¸ÁÂÃÄÅÆÉËÍÎÔÕÖ×ØÚÛÜßáâãäåæéëíîôõö÷øúûüĀāĄąČčĐđĒēĖėĘęĢģĨĩĪīĮįĶķĸĻļŅņŊŋŌōŖŗŠšŦŧŨũŪūŲųŽžˇ˙˛�' ,
'5': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~§ЁЂЃЄЅІЇЈЉЊЋЌЎЏАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёђѓєѕіїјљњћќўџ№�' ,
'6': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~¤،؛؟ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيًٌٍَُِّْ�' ,
'7': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~£¦§¨©«¬°±²³·»½ͺ΄΅ΆΈΉΊΌΎΏΐΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΪΫάέήίΰαβγδεζηθικλμνξοπρςστυφχψωϊϋόύώ―‘’€₯�' ,
'8': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~¢£¤¥¦§¨©«¬®¯°±²³´µ¶·¸¹»¼½¾×÷אבגדהוזחטיךכלםמןנסעףפץצקרשת‗�' ,
'9': ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖ×ØÙÚÛÜßàáâãäåæçèéêëìíîïñòóôõö÷øùúûüÿĞğİıŞş�' ,
}


def get_common_characters(lang_to_chars: Dict[str, str]) -> Dict[Tuple[str, str], Set[str]]:
    common_characters = {}
    for lang1 in lang_to_chars:
        for lang2 in lang_to_chars:
            common_characters[(lang1, lang2)] = set(lang_to_chars[lang1]) & set(lang_to_chars[lang2])
    return common_characters


def common_characters_to_distance_matrix(common_characters: Dict[Tuple[str, str], Set[str]], language_alphabets: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    language_names = list(sorted(set([k[0] for k in common_characters.keys()])))
    distance_matrix = np.zeros((len(language_names), len(language_names)))
    for n1, lang1 in enumerate(language_names):
        for n2, lang2 in enumerate(language_names):
            #lang_intersection_length  = len(common_characters[(lang1, lang2)])
            #lang_union_length = len(set(extended_keyboard_characters[lang1])) + len(set(extended_keyboard_characters[lang2])) - lang_intersection_length
            #distance_matrix[n1, n2] = lang_intersection_length / lang_union_length
            largest_alphabet = max(len(language_alphabets[lang1]), len(language_alphabets[lang2]))
            distance_matrix[n1, n2] = (len(common_characters[(lang1, lang2)])//2) / (largest_alphabet//2)

    distance_matrix = distance_matrix.max() - distance_matrix
    print(f"LANGUAGES: {language_names} Mat{distance_matrix.shape}")
    return language_names, distance_matrix




def plot_dendrogram(DM,linkage_matrix, labels):
    
    def get_leaf_labels(id, n, _labels):
        if id < n:
            return [_labels[id]]
        else:
            left, right = int(linkage_matrix[id - n, 0]), int(linkage_matrix[id - n, 1])
            return get_leaf_labels(left, n, _labels) + get_leaf_labels(right, n, _labels)

    # Custom leaf labeling function
    def llf(id):
        leaf_labels = [leaf_name(n) for n in sorted(get_leaf_labels(id, len(DM), labels))]
        return ''.join(leaf_labels)

    # Plot the dendrogram with custom leaf labelsextended_keyboard_characters
    plt.figure(figsize=(10, 7))
    dendrogram(
        linkage_matrix,
        leaf_label_func=llf,  # Use the custom label function
        leaf_rotation=90,
        leaf_font_size=12,
        show_contracted=False  # Display intermediary branches
    )
    plt.title('Dendrogram with Custom Node Labels')
    plt.show()


def leaf_name(n:str)->str:
    return f"iso-8859-{n}"

def branch_name(n: List[str]) -> str:
    return f"iso({'+'.join(sorted(n))})"

if __name__ == "__main__":
    common_characters = get_common_characters(codepage_alphabets)
    language_names, distance_matrix = common_characters_to_distance_matrix(common_characters, codepage_alphabets)



    #make_tree(language_names, distance_matrix)
    unique_characters = {}
    character_occurences = defaultdict(lambda: [])
    for k in codepage_alphabets:
        others = set(codepage_alphabets.keys()) - {k}
        other_alphabet = set(''.join([codepage_alphabets[ok] for ok in others]))
        unique_characters[k] = set(codepage_alphabets[k]) - other_alphabet
        for c in codepage_alphabets[k]:
            character_occurences[c].append(k)

    # for k, v in unique_characters.items():
    #     print(f"{k}: {sorted(v)}")
    # print(f"ALL: {sorted([k for k, v in character_occurences.items() if len(v) == len(extended_keyboard_characters)])}")
    # print("\n\n")

    #common_characters = get_common_characters()
    for k in codepage_alphabets:
         print(f"{k} : {unique_characters[k]}", end="")
         common_count = list(reversed(sorted([(len(v), k_pair[1]) for k_pair, v in common_characters.items() if k_pair[0] == k])))[1:]
         for count, lang in common_count:
             print(f", {lang}({count})", end="")
         print(codepage_alphabets)
    linkage_matrix = linkage(distance_matrix, "ward")
    plot_dendrogram(distance_matrix, linkage_matrix,language_names)