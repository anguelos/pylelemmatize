from difflib import SequenceMatcher
import string
import unicodedata

import numpy as np
from unidecode import unidecode




def char_similarity(a: str, b: str, symmetric: bool = True) -> float:
    """Compute similarity score between two characters based on multiple heuristics."""
    score = 0.0

    # Basic identity
    if a == b:
        return 1.0

    # Unicode name similarity
    try:
        name_a = unicodedata.name(a)
    except ValueError:
        name_a = ""
    try:
        name_b = unicodedata.name(b)
    except ValueError:
        name_b = ""
    
    cat_1 = unicodedata.category(a)
    cat_2 = unicodedata.category(b)
    score += .1 * (cat_1[0] == cat_2[0])  # Same category class
    score += .2 * (cat_1 == cat_2)  # Same category


    if name_a and name_b:
        name_a_pieces = set(name_a.split())
        name_b_pieces = set(name_b.split())
        common_pieces = len(name_a_pieces.intersection(name_b_pieces))
        total_pieces = max(len(name_a_pieces), len(name_b_pieces))
        score += 0.3 * (common_pieces / (total_pieces + .00000000001))

        matcher = SequenceMatcher(None, name_a, name_b)
        score += 0.2 * matcher.ratio()
        matcher = SequenceMatcher(None, name_a[:len(name_a)//2], name_b[:len(name_b)//2])  #  δ("ά" - "έ") > δ("ά" - "a")
        score += 0.1 * matcher.ratio()

    # Lowercase/uppercase match
    if a.lower() == b.lower() and a.isalpha() and b.isalpha():
        score += 0.2

    # Whitespace match
    if a.isspace() and b.isspace():
        score += 0.2

    # Punctuation match
    if a in string.punctuation and b in string.punctuation:
        score += 0.2

    # Unidecode match
    unidecode_a = unidecode(a)
    unidecode_b = unidecode(b)
    #print(f"Unidecode: {unidecode_a} vs {unidecode_b}")
    if unidecode_a != "" and unidecode_b != "":
        #print(f"Unidecode comparison: {unidecode_a} {len(unidecode_a)} vs {unidecode_b} {len(unidecode_b)}")
        if len(unidecode_a) > 1 or len(unidecode_b) > 1:
            a_let = np.array(list(unidecode_a), dtype=str)[None, :]
            b_let = np.array(list(unidecode_b), dtype=str)[:, None]
            agreement = (a_let == b_let).astype(float)
            position_coefficient_a = np.linspace(1.5, 0.5, num=len(unidecode_a))[None, :]
            position_coefficient_b = np.linspace(1.5, 0.5, num=len(unidecode_b))[:, None]
            weighed_agreement = agreement * position_coefficient_a * position_coefficient_b
            #print(f"{a_let} vs {b_let} -> \nUnweighted:\n{agreement}\nWeighted:\n{weighed_agreement}\n")
            score += np.mean(weighed_agreement)
            #print(a_let, b_let, agreement)
        else:
            score += 0.7 * (unidecode_a == unidecode_b)
        
        unidecode_a, unidecode_b = unidecode_a.lower(), unidecode_b.lower()
        if len(unidecode_a) > 1 or len(unidecode_b) > 1:
            a_let = np.array(list(unidecode_a), dtype=str)[None, :]
            b_let = np.array(list(unidecode_b), dtype=str)[:, None]
            agreement = (a_let == b_let).astype(float)
            position_coefficient_a = np.linspace(1.5, 0.5, num=len(unidecode_a))[None, :]
            position_coefficient_b = np.linspace(1.5, 0.5, num=len(unidecode_b))[:, None]
            score += np.mean(agreement * position_coefficient_a * position_coefficient_b)
        else:
            score += 0.5 * (unidecode_a == unidecode_b)
    else:
        pass

    if unidecode_a == unidecode_b and unidecode_a != "":
        score += 0.25
    elif unidecode_a != "" and unidecode_b != "" and unidecode_a.lower() in "aeiouy" and unidecode_b.lower() in "aeiou": # Vowel
        score += 0.2
    elif unidecode_a != "" and unidecode_b != "" and unidecode_a.lower() not in "aeiouy" and unidecode_b.lower() not in "aeiou": # Consonant
        score += 0.1
    
    # Ordinal proximity this should matter a very little
    ord_proximity = 1/(1+np.exp(-np.abs(ord(a) - ord(b))))
    score += .05 * ord_proximity
    #print(f"score before: {score}", end="")
    score =  2*((1/(1+np.exp(-score))) - .5)  # Normalize score to be between 0 and 1
    #print(f"   sig(score: {score})")
    if symmetric:
        return score * .5 + char_similarity(b, a, symmetric=False) * .5  # Ensure symmetry
    else:
        return score
    
