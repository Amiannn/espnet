#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import unicodedata
import codecs
import argparse
from enum import Enum
from tqdm import tqdm
import os

spacelist = [" ", "\t", "\r", "\n"]
puncts = [
    "!",
    ",",
    "?",
    "、",
    "。",
    "！",
    "，",
    "；",
    "？",
    "：",
    "「",
    "」",
    "︰",
    "『",
    "』",
    "《",
    "》",
]

# Whether to remove tags like <...> from tokens:
remove_tag = False


class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


class WordError:
    """Holds the number of reference words and errors of each type
    for computing standard WER or a sub-category WER (e.g. U-WER, B-WER)."""

    def __init__(self):
        self.errors = {
            Code.substitution: 0,
            Code.insertion: 0,
            Code.deletion: 0,
        }
        self.ref_words = 0

    def get_wer(self):
        if self.ref_words == 0:
            return 0.0
        total_err = (
            self.errors[Code.substitution] +
            self.errors[Code.insertion] +
            self.errors[Code.deletion]
        )
        return 100.0 * total_err / self.ref_words

    def get_result_string(self):
        return (
            f"error_rate={self.get_wer():.4f}, "
            f"ref_words={self.ref_words}, "
            f"subs={self.errors[Code.substitution]}, "
            f"ins={self.errors[Code.insertion]}, "
            f"dels={self.errors[Code.deletion]}"
        )


def characterize(string):
    """Splits a string into character-like tokens, skipping punctuation/spaces."""
    res = []
    i = 0
    while i < len(string):
        char = string[i]
        if char in puncts:
            i += 1
            continue
        cat1 = unicodedata.category(char)
        # skip spaces or unassigned
        if cat1 == "Zs" or cat1 == "Cn" or char in spacelist:
            i += 1
            continue
        if cat1 == "Lo":  # letter-other
            res.append(char)
            i += 1
        else:
            # For tokens like <unk><noise>, we want to separate them properly
            sep = " "
            if char == "<":
                sep = ">"
            j = i + 1
            while j < len(string):
                c = string[j]
                if ord(c) >= 128 or (c in spacelist) or (c == sep):
                    break
                j += 1
            if j < len(string) and string[j] == ">":
                j += 1
            res.append(string[i:j])
            i = j
    return res


def stripoff_tags(x):
    """Removes <...> tags from a single token."""
    if not x:
        return ""
    chars = []
    i = 0
    T = len(x)
    while i < T:
        if x[i] == "<":
            while i < T and x[i] != ">":
                i += 1
            i += 1
        else:
            chars.append(x[i])
            i += 1
    return "".join(chars)


def normalize(sentence, ignore_words, cs, split=None):
    """Case normalization, removing tags if remove_tag=True, splitting if needed."""
    new_sentence = []
    for token in sentence:
        x = token
        if not cs:
            x = x.upper()
        if x in ignore_words:
            continue
        if remove_tag:
            x = stripoff_tags(x)
        if not x:
            continue
        if split and x in split:
            new_sentence += split[x]
        else:
            new_sentence.append(x)
    return new_sentence


class Calculator:
    """Levenshtein alignment + tracking per-token error counts."""

    def __init__(self):
        self.data = {}
        self.space = []
        self.cost = {
            "cor": 0,
            "sub": 1,
            "del": 1,
            "ins": 1,
        }

    def calculate(self, lab, rec):
        """Returns alignment results with stats on how many correct, sub, del, ins."""
        # Insert an empty token at beginning
        lab.insert(0, "")
        rec.insert(0, "")

        # Expand self.space to at least the needed size
        while len(self.space) < len(lab):
            self.space.append([])
        for row in self.space:
            while len(row) < len(rec):
                row.append({"dist": 0, "error": "non"})
            # Clear old distances/errors
            for cell in row:
                cell["dist"] = 0
                cell["error"] = "non"

        # Initialization
        for i in range(len(lab)):
            self.space[i][0]["dist"] = i
            self.space[i][0]["error"] = "del"
        for j in range(len(rec)):
            self.space[0][j]["dist"] = j
            self.space[0][j]["error"] = "ins"
        self.space[0][0]["error"] = "non"

        # Prepare data for tokens
        for token in lab + rec:
            if token and token not in self.data:
                self.data[token] = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}

        # Compute edit distance
        for i in range(1, len(lab)):
            for j in range(1, len(rec)):
                min_dist = sys.maxsize
                min_error = "none"

                # Deletion
                dist = self.space[i - 1][j]["dist"] + self.cost["del"]
                error = "del"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error

                # Insertion
                dist = self.space[i][j - 1]["dist"] + self.cost["ins"]
                error = "ins"
                if dist < min_dist:
                    min_dist = dist
                    min_error = error

                # Match/Substitution
                if lab[i] == rec[j].replace("<BIAS>", ""):
                    dist = self.space[i - 1][j - 1]["dist"] + self.cost["cor"]
                    error = "cor"
                else:
                    dist = self.space[i - 1][j - 1]["dist"] + self.cost["sub"]
                    error = "sub"

                if dist < min_dist:
                    min_dist = dist
                    min_error = error

                self.space[i][j]["dist"] = min_dist
                self.space[i][j]["error"] = min_error

        # Traceback
        result = {
            "lab": [],
            "rec": [],
            "code": [],
            "all": 0,
            "cor": 0,
            "sub": 0,
            "ins": 0,
            "del": 0,
        }
        i = len(lab) - 1
        j = len(rec) - 1
        while True:
            err = self.space[i][j]["error"]
            if err == "cor":
                # correct
                if lab[i]:
                    self.data[lab[i]]["all"] += 1
                    self.data[lab[i]]["cor"] += 1
                    result["all"] += 1
                    result["cor"] += 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, rec[j])
                result["code"].insert(0, Code.match)
                i -= 1
                j -= 1
            elif err == "sub":
                # substitution
                if lab[i]:
                    self.data[lab[i]]["all"] += 1
                    self.data[lab[i]]["sub"] += 1
                    result["all"] += 1
                    result["sub"] += 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, rec[j])
                result["code"].insert(0, Code.substitution)
                i -= 1
                j -= 1
            elif err == "del":
                # deletion
                if lab[i]:
                    self.data[lab[i]]["all"] += 1
                    self.data[lab[i]]["del"] += 1
                    result["all"] += 1
                    result["del"] += 1
                result["lab"].insert(0, lab[i])
                result["rec"].insert(0, "")
                result["code"].insert(0, Code.deletion)
                i -= 1
            elif err == "ins":
                # insertion
                if rec[j]:
                    self.data[rec[j]]["ins"] += 1
                    result["ins"] += 1
                result["lab"].insert(0, "")
                result["rec"].insert(0, rec[j])
                result["code"].insert(0, Code.insertion)
                j -= 1
            elif err == "non":
                # starting point
                break
            else:
                print(
                    f"This should not happen , i = {i} , j = {j} , error = {err}"
                )
                break

        return result

    def overall(self):
        """Aggregate over all tokens across all utterances."""
        result = {"all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0}
        for token in self.data:
            result["all"] += self.data[token]["all"]
            result["cor"] += self.data[token]["cor"]
            result["sub"] += self.data[token]["sub"]
            result["ins"] += self.data[token]["ins"]
            result["del"] += self.data[token]["del"]
        return result


def width(string):
    """Return 'visual width' of a string for alignment. CJK often double-width."""
    return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)


def default_cluster(word):
    """A simplistic 'cluster' approach for classifying tokens. Used in debug prints."""
    unicode_names = [unicodedata.name(char) for char in word]
    for i in reversed(range(len(unicode_names))):
        if unicode_names[i].startswith("DIGIT"):
            unicode_names[i] = "Number"
        elif unicode_names[i].startswith("CJK UNIFIED IDEOGRAPH") or \
             unicode_names[i].startswith("CJK COMPATIBILITY IDEOGRAPH"):
            unicode_names[i] = "Mandarin"
        elif unicode_names[i].startswith("LATIN CAPITAL LETTER") or \
             unicode_names[i].startswith("LATIN SMALL LETTER"):
            unicode_names[i] = "English"
        elif unicode_names[i].startswith("HIRAGANA LETTER"):
            unicode_names[i] = "Japanese"
        elif (
            unicode_names[i].startswith("AMPERSAND") or
            unicode_names[i].startswith("APOSTROPHE") or
            unicode_names[i].startswith("COMMERCIAL AT") or
            unicode_names[i].startswith("DEGREE CELSIUS") or
            unicode_names[i].startswith("EQUALS SIGN") or
            unicode_names[i].startswith("FULL STOP") or
            unicode_names[i].startswith("HYPHEN-MINUS") or
            unicode_names[i].startswith("LOW LINE") or
            unicode_names[i].startswith("NUMBER SIGN") or
            unicode_names[i].startswith("PLUS SIGN") or
            unicode_names[i].startswith("SEMICOLON")
        ):
            # We remove these small symbols entirely
            del unicode_names[i]
        else:
            return "Other"
    if len(unicode_names) == 0:
        return "Other"
    if len(unicode_names) == 1:
        return unicode_names[0]
    for i in range(len(unicode_names) - 1):
        if unicode_names[i] != unicode_names[i + 1]:
            return "Other"
    return unicode_names[0]


def get_args():
    parser = argparse.ArgumentParser(description="Compute WER, also track U-WER/B-WER.")
    parser.add_argument("--ref", type=str, help="Reference text input path")
    parser.add_argument("--ref_ocr", type=str, help="Reference OCR input path")
    parser.add_argument("--ref2session", type=str, default="", help="Map utterance to session")
    parser.add_argument("--rec_name", type=str, action="append", default=[], help="List of system names")
    parser.add_argument("--rec_file", type=str, action="append", default=[], help="List of system text files")
    parser.add_argument("--verbose", type=int, default=1, help="Set verbosity level")
    parser.add_argument("--char", type=bool, default=True, help="Character-based alignment (True) or word-based")
    return parser.parse_args()


def main(args):
    tochar = args.char
    verbose = args.verbose
    case_sensitive = False
    ignore_words = set()
    split = None

    if not case_sensitive:
        # unify ignore_words
        ignore_words = set(w.upper() for w in ignore_words)

    # Read reference OCR file
    ref_ocr_dict = {}
    with codecs.open(args.ref_ocr, "r", "utf-8") as fh:
        for line in fh:
            if "$" in line:
                line = line.replace("$", " ")
            array = characterize(line) if tochar else line.strip().split()
            if not array:
                continue
            fid = array[0]
            ref_ocr_dict[fid] = normalize(array[1:], ignore_words, case_sensitive, split)

    # Possibly load session info
    utt2session = None
    if args.ref2session:
        utt2session = {}
        with codecs.open(args.ref2session, "r", "utf-8") as fh:
            for line in fh:
                uttid, session = line.strip().split()
                utt2session[uttid] = session

    # Read hypothesis systems
    rec_files = args.rec_file
    rec_names = args.rec_name
    assert len(rec_files) == len(rec_names)

    rec_sets = {}
    calculators_dict = {}
    # For global (all-utt) U-WER/B-WER
    ub_wer_dict = {}
    # For session-level results
    multirec_session_result = {}
    # For hotword stats
    hotwords_related_dict = {}

    for i, hyp_file in enumerate(rec_files):
        system_name = rec_names[i]
        rec_sets[system_name] = {}
        with codecs.open(hyp_file, "r", "utf-8") as fh:
            for line in fh:
                array = characterize(line) if tochar else line.strip().split()
                if not array:
                    continue
                fid = array[0]
                rec_sets[system_name][fid] = normalize(array[1:], ignore_words, case_sensitive, split)

        calculators_dict[system_name] = Calculator()

        # Initialize global U/B/normal WER for each system
        ub_wer_dict[system_name] = {
            "u_wer": WordError(),
            "b_wer": WordError(),
            "wer": WordError(),
        }
        # For each system, keep session-level results
        if utt2session:
            multirec_session_result[system_name] = {}
        # Hotword stats
        hotwords_related_dict[system_name] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    # Stats: how many times label is in OCR but recognized incorrectly
    wrong_rec_but_in_ocr_dict = {name: 0 for name in rec_names}

    # Attempt to get the total lines for progress bar
    # (If you want to remove tqdm, you can simply loop without tqdm.)
    # Fallback if we can't get a line count
    try:
        from pathlib import Path
        _file_total_len = sum(1 for _ in open(args.ref, 'r', encoding='utf-8'))
    except:
        _file_total_len = None

    # Compute WER per line
    with open(args.ref, 'r', encoding='utf-8') as f:
        # If _file_total_len is None, tqdm won't show total
        it = tqdm(f, total=_file_total_len, desc="Processing") if verbose else f

        for line in it:
            array = characterize(line) if tochar else line.strip().split()
            if not array:
                continue
            fid = array[0]
            lab = normalize(array[1:], ignore_words, case_sensitive, split)

            if verbose:
                print(f"\nutt: {fid}")

            # label vs. OCR
            ocr_text = ref_ocr_dict.get(fid, [])
            print("ocr:", " ".join(ocr_text))

            ocr_set = set(ocr_text)
            list_match = []
            list_not_match = []
            for token in lab:
                if token in ocr_set:
                    list_match.append(token)
                else:
                    list_not_match.append(token)

            print("label in ocr:", " ".join(list_match))

            # Evaluate for each system
            base_wrong_ocr_wer = None
            ocr_wrong_ocr_wer = None

            for rec_name in rec_names:
                if fid not in rec_sets[rec_name]:
                    continue

                rec = rec_sets[rec_name][fid]
                # For debug cluster usage:
                for w in rec + lab:
                    if w not in ignore_words:
                        _ = default_cluster(w)  # we do not strictly use it, but you might

                result = calculators_dict[rec_name].calculate(lab.copy(), rec.copy())

                if result["all"] != 0:
                    wer_val = float(result["ins"] + result["sub"] + result["del"]) * 100.0 / result["all"]
                else:
                    wer_val = 0.0

                print(f"WER({rec_name}): {wer_val:4.2f} % ", end="")
                print("N={all} C={cor} S={sub} D={del_} I={ins}".format(
                    all=result["all"], cor=result["cor"],
                    sub=result["sub"], del_=result["del"], ins=result["ins"]))

                # Accumulate session-level standard WER stats:
                if utt2session:
                    session = utt2session[fid]
                    if session not in multirec_session_result[rec_name]:
                        multirec_session_result[rec_name][session] = {
                            "all": 0, "cor": 0, "sub": 0, "ins": 0, "del": 0,
                            # For session-level U-WER/B-WER:
                            "ub_wer": {
                                "u_wer": WordError(),
                                "b_wer": WordError(),
                                "wer": WordError()
                            }
                        }
                    ses_res = multirec_session_result[rec_name][session]
                    ses_res["all"] += result["all"]
                    ses_res["cor"] += result["cor"]
                    ses_res["sub"] += result["sub"]
                    ses_res["ins"] += result["ins"]
                    ses_res["del"] += result["del"]

                # Count how many label tokens are recognized wrongly while they're in OCR
                wrong_rec_but_in_ocr = []
                for idx, code in enumerate(result["code"]):
                    # If the reference token is not empty and is in OCR, but recognized incorrectly
                    if result["lab"][idx] and (result["lab"][idx] in list_match):
                        # mismatch
                        ref_word = result["lab"][idx]
                        hyp_word = result["rec"][idx].replace("<BIAS>", "")
                        if ref_word != hyp_word:
                            wrong_rec_but_in_ocr.append(ref_word)
                wrong_rec_but_in_ocr_dict[rec_name] += len(wrong_rec_but_in_ocr)
                print("wrong_rec_but_in_ocr:", " ".join(wrong_rec_but_in_ocr))

                # Compare to the 'base' system
                if rec_name == "base":
                    base_wrong_ocr_wer = len(wrong_rec_but_in_ocr)
                if "ocr" in rec_name or "hot" in rec_name:
                    ocr_wrong_ocr_wer = len(wrong_rec_but_in_ocr)
                    if base_wrong_ocr_wer is not None:
                        if ocr_wrong_ocr_wer < base_wrong_ocr_wer:
                            print(f"{fid} {rec_name} helps, {base_wrong_ocr_wer} -> {ocr_wrong_ocr_wer}")
                        elif ocr_wrong_ocr_wer > base_wrong_ocr_wer:
                            print(f"{fid} {rec_name} hurts, {base_wrong_ocr_wer} -> {ocr_wrong_ocr_wer}")

                # --- Hotword stats
                _rec_list = [w.replace("<BIAS>", "") for w in rec]
                hot_true_list = set(x for x in ocr_text if x in lab)   # "true" hotwords
                hot_bad_list = set(x for x in ocr_text if x not in lab)

                _tp = _tn = _fp = _fn = 0

                # Negative hotwords (in OCR, not in label)
                for bad in hot_bad_list:
                    count_in_rec = _rec_list.count(bad)
                    if count_in_rec == 0:
                        hotwords_related_dict[rec_name]['tn'] += 1
                        _tn += 1
                    else:
                        hotwords_related_dict[rec_name]['fp'] += count_in_rec
                        _fp += count_in_rec

                # Positive hotwords (in OCR, also in label)
                for hotword in hot_true_list:
                    count_label = lab.count(hotword)
                    count_rec = _rec_list.count(hotword)
                    if count_rec == count_label:
                        hotwords_related_dict[rec_name]['tp'] += count_label
                        _tp += count_label
                    elif count_rec > count_label:
                        hotwords_related_dict[rec_name]['tp'] += count_label
                        hotwords_related_dict[rec_name]['fp'] += (count_rec - count_label)
                        _tp += count_label
                        _fp += (count_rec - count_label)
                    else:
                        hotwords_related_dict[rec_name]['tp'] += count_rec
                        hotwords_related_dict[rec_name]['fn'] += (count_label - count_rec)
                        _tp += count_rec
                        _fn += (count_label - count_rec)

                all_hot = _tp + _tn + _fp + _fn
                recall_ = _tp / (_tp + _fn) * 100 if (_tp + _fn) != 0 else 0
                print(f"hotword: tp: {_tp}, tn: {_tn}, fp: {_fp}, fn: {_fn}, all: {all_hot}, recall: {recall_:.2f}%")

                # --- Accumulate global U-WER/B-WER/normal WER
                # Identify which references are "hot" (b-words) vs. "non-hot" (u-words).
                for idx, code in enumerate(result["code"]):
                    ref_word = result["lab"][idx]
                    hyp_word = result["rec"][idx].replace("<BIAS>", "")
                    # is this a "b-word" if it's in the intersection (hot_true_list)?
                    # (We define it as "hot" if the label's token is in hot_true_list.)
                    # The logic might differ depending on your exact definition.
                    is_b_word = (ref_word in hot_true_list)

                    # Increase counters
                    ub_wer_dict[rec_name]["wer"], wobj = ub_wer_dict[rec_name]["wer"], None
                    if is_b_word:
                        wobj = ub_wer_dict[rec_name]["b_wer"]
                    else:
                        wobj = ub_wer_dict[rec_name]["u_wer"]

                    if code == Code.match:
                        wobj.ref_words += 1
                        ub_wer_dict[rec_name]["wer"].ref_words += 1
                    elif code == Code.substitution:
                        wobj.ref_words += 1
                        wobj.errors[Code.substitution] += 1
                        ub_wer_dict[rec_name]["wer"].ref_words += 1
                        ub_wer_dict[rec_name]["wer"].errors[Code.substitution] += 1
                    elif code == Code.deletion:
                        wobj.ref_words += 1
                        wobj.errors[Code.deletion] += 1
                        ub_wer_dict[rec_name]["wer"].ref_words += 1
                        ub_wer_dict[rec_name]["wer"].errors[Code.deletion] += 1
                    elif code == Code.insertion:
                        # insertion does not increase ref_words, but we do increment insertion error
                        ub_wer_dict[rec_name]["wer"].errors[Code.insertion] += 1
                        if is_b_word:
                            ub_wer_dict[rec_name]["b_wer"].errors[Code.insertion] += 1
                        else:
                            ub_wer_dict[rec_name]["u_wer"].errors[Code.insertion] += 1

                    # Also update session-level U/B/normal WER if needed
                    if utt2session:
                        session = utt2session[fid]
                        s_ub = multirec_session_result[rec_name][session]["ub_wer"]

                        # The main "wer"
                        if code == Code.match:
                            s_ub["wer"].ref_words += 1
                            if is_b_word:
                                s_ub["b_wer"].ref_words += 1
                            else:
                                s_ub["u_wer"].ref_words += 1
                        elif code == Code.substitution:
                            s_ub["wer"].ref_words += 1
                            s_ub["wer"].errors[Code.substitution] += 1
                            if is_b_word:
                                s_ub["b_wer"].ref_words += 1
                                s_ub["b_wer"].errors[Code.substitution] += 1
                            else:
                                s_ub["u_wer"].ref_words += 1
                                s_ub["u_wer"].errors[Code.substitution] += 1
                        elif code == Code.deletion:
                            s_ub["wer"].ref_words += 1
                            s_ub["wer"].errors[Code.deletion] += 1
                            if is_b_word:
                                s_ub["b_wer"].ref_words += 1
                                s_ub["b_wer"].errors[Code.deletion] += 1
                            else:
                                s_ub["u_wer"].ref_words += 1
                                s_ub["u_wer"].errors[Code.deletion] += 1
                        elif code == Code.insertion:
                            s_ub["wer"].errors[Code.insertion] += 1
                            if is_b_word:
                                s_ub["b_wer"].errors[Code.insertion] += 1
                            else:
                                s_ub["u_wer"].errors[Code.insertion] += 1

                # Print alignment if verbose
                if verbose > 1:
                    padding_symbol = " "
                    max_words_per_line = sys.maxsize
                    space = {"lab": [], "rec": []}
                    for idx in range(len(result["lab"])):
                        len_lab = width(result["lab"][idx])
                        len_rec = width(result["rec"][idx])
                        length = max(len_lab, len_rec)
                        space["lab"].append(length - len_lab)
                        space["rec"].append(length - len_rec)

                    upper_lab = len(result["lab"])
                    upper_rec = len(result["rec"])
                    lab1 = rec1 = 0
                    while lab1 < upper_lab or rec1 < upper_rec:
                        print(f"lab({fid}):", end=" ")
                        lab2 = min(upper_lab, lab1 + max_words_per_line)
                        for idx in range(lab1, lab2):
                            token = result["lab"][idx]
                            print(token, end="")
                            print(padding_symbol * space["lab"][idx], end=" ")
                        print()

                        print(f"rec({fid}):", end=" ")
                        rec2 = min(upper_rec, rec1 + max_words_per_line)
                        for idx in range(rec1, rec2):
                            token = result["rec"][idx]
                            print(token, end="")
                            print(padding_symbol * space["rec"][idx], end=" ")
                        print()
                        lab1 = lab2
                        rec1 = rec2
                elif verbose == 1:
                    # Just minimal alignment prints
                    pass

            print()

    # Print session-level results (standard WER, plus U-WER/B-WER)
    if utt2session:
        print("=" * 75)
        print("Per-session results:")
        print()
        # session_results_zip structure:
        #  { session: [ (rec_name, result_dict), ... ], ... } is not directly built,
        #  but we can build it from multirec_session_result
        for rec_name, session_result in multirec_session_result.items():
            for session, sres in session_result.items():
                n_all = sres["all"]
                if n_all != 0:
                    wer_val = float(sres["ins"] + sres["sub"] + sres["del"]) * 100.0 / n_all
                else:
                    wer_val = 0.0
                print(f"{rec_name} session={session} -> {wer_val:4.2f} % ", end="")
                print(
                    f"N={n_all} C={sres['cor']} S={sres['sub']} "
                    f"D={sres['del']} I={sres['ins']}"
                )

                # Also print session-level U-WER/B-WER
                sess_wer_obj = sres["ub_wer"]["wer"]
                sess_u_wer_obj = sres["ub_wer"]["u_wer"]
                sess_b_wer_obj = sres["ub_wer"]["b_wer"]

                print(f"  WER:   {sess_wer_obj.get_result_string()}")
                print(f"  U-WER: {sess_u_wer_obj.get_result_string()}")
                print(f"  B-WER: {sess_b_wer_obj.get_result_string()}")
                print()
        print("=" * 75)
        print()

    # Print how many references (in OCR) got mis-recognized
    print("Counts of label-in-OCR but recognized incorrectly:", wrong_rec_but_in_ocr_dict)
    print()

    # Print overall results for each system
    for rec_name in rec_names:
        result = calculators_dict[rec_name].overall()
        n_all = result["all"]
        if n_all != 0:
            wer_val = float(result["ins"] + result["sub"] + result["del"]) * 100.0 / n_all
        else:
            wer_val = 0.0

        print(f"System {rec_name} Overall -> {wer_val:4.2f} % ", end="")
        print("N={all} C={cor} S={sub} D={del_} I={ins}".format(
            all=result["all"], cor=result["cor"], sub=result["sub"],
            del_=result["del"], ins=result["ins"])
        )

        # Print global WER, U-WER, B-WER
        w_wer = ub_wer_dict[rec_name]["wer"]
        u_wer = ub_wer_dict[rec_name]["u_wer"]
        b_wer = ub_wer_dict[rec_name]["b_wer"]
        print(f"  WER:   {w_wer.get_result_string()}")
        print(f"  U-WER: {u_wer.get_result_string()}")
        print(f"  B-WER: {b_wer.get_result_string()}")

        # hotword recall stats
        hw_stats = hotwords_related_dict[rec_name]
        tp = hw_stats['tp']
        tn = hw_stats['tn']
        fp = hw_stats['fp']
        fn = hw_stats['fn']
        total_hot = tp + tn + fp + fn
        recall = tp / (tp + fn) * 100 if (tp + fn) != 0 else 0
        print(f"  Hotword stats -> tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}, all: {total_hot}, recall: {recall:.2f}%")

        # Summarize the final line with WER, U-WER, B-WER, recall
        print(f"  => Final: {w_wer.get_wer():.3f} (WER); {u_wer.get_wer():.3f} (U-WER); "
              f"{b_wer.get_wer():.3f} (B-WER); {recall:.2f}% (hotword recall)")
        print()

    print("Done.")


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
