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
import csv

# By default, we are no longer removing punctuation or forced splitting.
# If you ever want to ignore certain tokens, you can do that in `normalize()`
# or with an --ignore-file, but don't forcibly remove them unless you know
# ESPnet does the same.

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
            self.errors[Code.substitution]
            + self.errors[Code.insertion]
            + self.errors[Code.deletion]
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

def normalize(sentence, ignore_words, case_sensitive, split=None):
    """Case normalization, removing tags if remove_tag=True, optionally splitting tokens."""
    new_sentence = []
    for token in sentence:
        x = token
        # Case normalization
        if not case_sensitive:
            x = x.upper()
        # Skip if the token is in ignore_words
        if x in ignore_words:
            continue
        # Remove <...> tags if remove_tag=True
        if remove_tag:
            x = stripoff_tags(x)
        if not x:
            continue
        # If you have a 'split' dict that maps certain tokens to expansions
        # (e.g., to handle certain compound tokens), apply it here:
        if split and x in split:
            new_sentence += split[x]
        else:
            new_sentence.append(x)
    return new_sentence

def tokenize_line(line, do_lowercase=False):
    """Simple whitespace-based tokenization. Includes punctuation as separate tokens."""
    if do_lowercase:
        line = line.lower()
    return line.strip().split()

class Calculator:
    """Levenshtein alignment + tracking per-token error counts."""

    def __init__(self):
        self.data = {}
        # 'space' is a 2D grid we reuse for dynamic programming
        self.space = []
        self.cost = {
            "cor": 0,
            "sub": 1,
            "del": 1,
            "ins": 1,
        }

    def calculate(self, lab, rec):
        """Returns alignment results with stats on how many correct, sub, del, ins."""
        # Insert an empty token at beginning to simplify indexing
        print(f'lab: {lab}')
        print(f'rec: {rec}')
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
                dist_del = self.space[i - 1][j]["dist"] + self.cost["del"]
                if dist_del < min_dist:
                    min_dist = dist_del
                    min_error = "del"

                # Insertion
                dist_ins = self.space[i][j - 1]["dist"] + self.cost["ins"]
                if dist_ins < min_dist:
                    min_dist = dist_ins
                    min_error = "ins"

                # Match/Substitution
                if lab[i] == rec[j].replace("<BIAS>", ""):
                    dist_cor = self.space[i - 1][j - 1]["dist"] + self.cost["cor"]
                    if dist_cor < min_dist:
                        min_dist = dist_cor
                        min_error = "cor"
                else:
                    dist_sub = self.space[i - 1][j - 1]["dist"] + self.cost["sub"]
                    if dist_sub < min_dist:
                        min_dist = dist_sub
                        min_error = "sub"

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
                print(f"This should not happen , i={i} , j={j} , error={err}")
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
    """Return 'visual width' of a string for alignment prints (optional)."""
    return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)

def default_cluster(word):
    """Dummy classification function (if you want it for debug prints)."""
    unicode_names = [unicodedata.name(char, "") for char in word]
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
    # Now default to word-based alignment:
    parser.add_argument("--char", action="store_true", help="Use character-based alignment if set")
    return parser.parse_args()

def main(args):
    # If args.char is True, you might do some alternate approach.
    # For standard word-based scoring, we do NOT break up punctuation, etc.
    tochar = args.char
    verbose = args.verbose

    # Typically official WER is case-insensitive, so set to False
    case_sensitive = False

    ignore_words = set()
    split = None

    if not case_sensitive:
        ignore_words = set(w.upper() for w in ignore_words)

    # --- 1) Read reference OCR file ---
    ref_ocr_dict = {}
    ref_ocr_name = os.path.basename(args.ref_ocr)
    with codecs.open(args.ref_ocr, "r", "utf-8") as fh:
        for line in fh:
            # Simple whitespace-based tokenization
            tokens = tokenize_line(line)
            if not tokens:
                continue
            fid = tokens[0]
            ref_ocr_dict[fid] = normalize(tokens[1:], ignore_words, case_sensitive, split)

    # Possibly load session info
    utt2session = None
    if args.ref2session:
        utt2session = {}
        with codecs.open(args.ref2session, "r", "utf-8") as fh:
            for line in fh:
                uttid, session = line.strip().split()
                utt2session[uttid] = session

    exp_folder = os.path.dirname(args.rec_file[0]) if args.rec_file else "."

    # --- 2) Read hypotheses for each system ---
    rec_sets = {}
    calculators_dict = {}
    ub_wer_dict = {}
    multirec_session_result = {}
    hotwords_related_dict = {}

    assert len(args.rec_file) == len(args.rec_name), \
        "Number of rec_file must match number of rec_name."

    for i, hyp_file in enumerate(args.rec_file):
        system_name = args.rec_name[i]
        rec_sets[system_name] = {}
        with codecs.open(hyp_file, "r", "utf-8") as fh:
            for line in fh:
                tokens = tokenize_line(line)
                if not tokens:
                    continue
                fid = tokens[0]
                rec_sets[system_name][fid] = normalize(tokens[1:], ignore_words, case_sensitive, split)

        calculators_dict[system_name] = Calculator()
        ub_wer_dict[system_name] = {
            "u_wer": WordError(),
            "b_wer": WordError(),
            "wer": WordError(),
        }

        if utt2session:
            multirec_session_result[system_name] = {}
        hotwords_related_dict[system_name] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    wrong_rec_but_in_ocr_dict = {name: 0 for name in args.rec_name}

    # Attempt to get total lines for progress bar
    try:
        from pathlib import Path
        _file_total_len = sum(1 for _ in open(args.ref, 'r', encoding='utf-8'))
    except:
        _file_total_len = None

    # Prepare CSV rows
    domain_result_rows = []
    final_result_rows = []

    # --- 3) Compute WER per line (reference vs. each system) ---
    with open(args.ref, 'r', encoding='utf-8') as f:
        it = tqdm(f, total=_file_total_len, desc="Processing") if verbose else f

        for line in it:
            tokens = tokenize_line(line)
            if not tokens:
                continue
            fid = tokens[0]
            lab = normalize(tokens[1:], ignore_words, case_sensitive, split)

            if verbose:
                print(f"\nutt: {fid}")

            # Show OCR text
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

            base_wrong_ocr_wer = None
            ocr_wrong_ocr_wer = None

            for rec_name in args.rec_name:
                if fid not in rec_sets[rec_name]:
                    continue

                rec = rec_sets[rec_name][fid]

                # If you need the 'debug cluster' approach:
                for w in rec + lab:
                    if w not in ignore_words:
                        _ = default_cluster(w)  # example usage only

                # Alignment
                result = calculators_dict[rec_name].calculate(lab.copy(), rec.copy())

                # Compute WER from alignment
                if result["all"] != 0:
                    wer_val = (
                        float(result["ins"] + result["sub"] + result["del"]) * 100.0
                        / result["all"]
                    )
                else:
                    wer_val = 0.0

                print(f"WER({rec_name}): {wer_val:4.2f} % ", end="")
                print(
                    "N={all} C={cor} S={sub} D={del_} I={ins}".format(
                        all=result["all"],
                        cor=result["cor"],
                        sub=result["sub"],
                        del_=result["del"],
                        ins=result["ins"],
                    )
                )

                # Session-level standard WER stats
                if utt2session:
                    session = utt2session[fid]
                    if session not in multirec_session_result[rec_name]:
                        multirec_session_result[rec_name][session] = {
                            "all": 0,
                            "cor": 0,
                            "sub": 0,
                            "ins": 0,
                            "del": 0,
                            "ub_wer": {
                                "u_wer": WordError(),
                                "b_wer": WordError(),
                                "wer": WordError(),
                            },
                        }
                    ses_res = multirec_session_result[rec_name][session]
                    ses_res["all"] += result["all"]
                    ses_res["cor"] += result["cor"]
                    ses_res["sub"] += result["sub"]
                    ses_res["ins"] += result["ins"]
                    ses_res["del"] += result["del"]

                # Count how many reference tokens were recognized wrongly while they're in OCR
                wrong_rec_but_in_ocr = []
                for idx, code in enumerate(result["code"]):
                    if result["lab"][idx] and (result["lab"][idx] in list_match):
                        ref_word = result["lab"][idx]
                        hyp_word = result["rec"][idx].replace("<BIAS>", "")
                        if ref_word != hyp_word:
                            wrong_rec_but_in_ocr.append(ref_word)

                wrong_rec_but_in_ocr_dict[rec_name] += len(wrong_rec_but_in_ocr)
                print("wrong_rec_but_in_ocr:", " ".join(wrong_rec_but_in_ocr))

                if rec_name == "base":
                    base_wrong_ocr_wer = len(wrong_rec_but_in_ocr)
                if "ocr" in rec_name or "hot" in rec_name:
                    ocr_wrong_ocr_wer = len(wrong_rec_but_in_ocr)
                    if base_wrong_ocr_wer is not None:
                        if ocr_wrong_ocr_wer < base_wrong_ocr_wer:
                            print(f"{fid} {rec_name} helps, {base_wrong_ocr_wer} -> {ocr_wrong_ocr_wer}")
                        elif ocr_wrong_ocr_wer > base_wrong_ocr_wer:
                            print(f"{fid} {rec_name} hurts, {base_wrong_ocr_wer} -> {ocr_wrong_ocr_wer}")

                # Hotword stats (example definition: "hotword" = any token that appears in both OCR and ref)
                _rec_list = [w.replace("<BIAS>", "") for w in rec]
                hot_true_list = set(x for x in ocr_text if x in lab)  # "true" hotwords
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

                # Accumulate global U-WER/B-WER/normal WER
                for idx, code in enumerate(result["code"]):
                    ref_word = result["lab"][idx]
                    hyp_word = result["rec"][idx].replace("<BIAS>", "")
                    # is this a "b-word"? We'll define it if it's in hot_true_list
                    is_b_word = (ref_word in hot_true_list)

                    w_wer = ub_wer_dict[rec_name]["wer"]
                    if is_b_word:
                        w_bwer = ub_wer_dict[rec_name]["b_wer"]
                    else:
                        w_bwer = ub_wer_dict[rec_name]["u_wer"]

                    if code == Code.match:
                        w_wer.ref_words += 1
                        w_bwer.ref_words += 1
                    elif code == Code.substitution:
                        w_wer.ref_words += 1
                        w_wer.errors[Code.substitution] += 1
                        w_bwer.ref_words += 1
                        w_bwer.errors[Code.substitution] += 1
                    elif code == Code.deletion:
                        w_wer.ref_words += 1
                        w_wer.errors[Code.deletion] += 1
                        w_bwer.ref_words += 1
                        w_bwer.errors[Code.deletion] += 1
                    elif code == Code.insertion:
                        w_wer.errors[Code.insertion] += 1
                        if is_b_word:
                            ub_wer_dict[rec_name]["b_wer"].errors[Code.insertion] += 1
                        else:
                            ub_wer_dict[rec_name]["u_wer"].errors[Code.insertion] += 1

                    # Session-level if needed
                    if utt2session:
                        session = utt2session[fid]
                        s_ub = multirec_session_result[rec_name][session]["ub_wer"]
                        s_wer = s_ub["wer"]
                        if code == Code.match:
                            s_wer.ref_words += 1
                            if is_b_word:
                                s_ub["b_wer"].ref_words += 1
                            else:
                                s_ub["u_wer"].ref_words += 1
                        elif code == Code.substitution:
                            s_wer.ref_words += 1
                            s_wer.errors[Code.substitution] += 1
                            if is_b_word:
                                s_ub["b_wer"].ref_words += 1
                                s_ub["b_wer"].errors[Code.substitution] += 1
                            else:
                                s_ub["u_wer"].ref_words += 1
                                s_ub["u_wer"].errors[Code.substitution] += 1
                        elif code == Code.deletion:
                            s_wer.ref_words += 1
                            s_wer.errors[Code.deletion] += 1
                            if is_b_word:
                                s_ub["b_wer"].ref_words += 1
                                s_ub["b_wer"].errors[Code.deletion] += 1
                            else:
                                s_ub["u_wer"].ref_words += 1
                                s_ub["u_wer"].errors[Code.deletion] += 1
                        elif code == Code.insertion:
                            s_wer.errors[Code.insertion] += 1
                            if is_b_word:
                                s_ub["b_wer"].errors[Code.insertion] += 1
                            else:
                                s_ub["u_wer"].errors[Code.insertion] += 1

                # Print alignment if verbose>1
                if verbose > 1:
                    padding_symbol = " "
                    max_words_per_line = sys.maxsize
                    space_ = {"lab": [], "rec": []}
                    for idx in range(len(result["lab"])):
                        len_lab = width(result["lab"][idx])
                        len_rec = width(result["rec"][idx])
                        length = max(len_lab, len_rec)
                        space_["lab"].append(length - len_lab)
                        space_["rec"].append(length - len_rec)

                    upper_lab = len(result["lab"])
                    upper_rec = len(result["rec"])
                    lab1 = rec1 = 0
                    while lab1 < upper_lab or rec1 < upper_rec:
                        print(f"lab({fid}):", end=" ")
                        lab2 = min(upper_lab, lab1 + max_words_per_line)
                        for idx in range(lab1, lab2):
                            token = result["lab"][idx]
                            print(token, end="")
                            print(padding_symbol * space_["lab"][idx], end=" ")
                        print()

                        print(f"rec({fid}):", end=" ")
                        rec2 = min(upper_rec, rec1 + max_words_per_line)
                        for idx in range(rec1, rec2):
                            token = result["rec"][idx]
                            print(token, end="")
                            print(padding_symbol * space_["rec"][idx], end=" ")
                        print()
                        lab1 = lab2
                        rec1 = rec2

            print()

    # --- 4) Print session-level results (if utt2session is used) ---
    if utt2session:
        print("=" * 75)
        print("Per-session (domain) results:")
        print()
        for rec_name, session_result in multirec_session_result.items():
            for session, sres in session_result.items():
                n_all = sres["all"]
                if n_all != 0:
                    wer_val = float(sres["ins"] + sres["sub"] + sres["del"]) * 100.0 / n_all
                else:
                    wer_val = 0.0
                print(f"{rec_name} session={session} -> {wer_val:4.2f} % ", end="")
                print(f"N={n_all} C={sres['cor']} S={sres['sub']} D={sres['del']} I={sres['ins']}")

                # Session-level U-WER/B-WER
                sess_wer_obj = sres["ub_wer"]["wer"]
                sess_u_wer_obj = sres["ub_wer"]["u_wer"]
                sess_b_wer_obj = sres["ub_wer"]["b_wer"]

                print(f"  WER:   {sess_wer_obj.get_result_string()}")
                print(f"  U-WER: {sess_u_wer_obj.get_result_string()}")
                print(f"  B-WER: {sess_b_wer_obj.get_result_string()}")
                print()

                domain_result_rows.append([
                    rec_name,
                    session,
                    f"{wer_val:.4f}",
                    sres["all"],
                    sres["cor"],
                    sres["sub"],
                    sres["del"],
                    sres["ins"],
                    f"{sess_u_wer_obj.get_wer():.4f}",
                    f"{sess_b_wer_obj.get_wer():.4f}",
                ])

        print("=" * 75)
        print()

    # --- 5) Print how many "label-in-OCR but recognized incorrectly" across references
    print("Counts of label-in-OCR but recognized incorrectly:", wrong_rec_but_in_ocr_dict)
    print()

    # --- 6) Print overall results for each system ---
    for rec_name in args.rec_name:
        result = calculators_dict[rec_name].overall()
        n_all = result["all"]
        if n_all != 0:
            wer_val = float(result["ins"] + result["sub"] + result["del"]) * 100.0 / n_all
        else:
            wer_val = 0.0

        print(f"System {rec_name} Overall -> {wer_val:4.2f} % ", end="")
        print(
            "N={all} C={cor} S={sub} D={del_} I={ins}".format(
                all=result["all"],
                cor=result["cor"],
                sub=result["sub"],
                del_=result["del"],
                ins=result["ins"],
            )
        )

        w_wer = ub_wer_dict[rec_name]["wer"]
        u_wer = ub_wer_dict[rec_name]["u_wer"]
        b_wer = ub_wer_dict[rec_name]["b_wer"]
        print(f"  WER:   {w_wer.get_result_string()}")
        print(f"  U-WER: {u_wer.get_result_string()}")
        print(f"  B-WER: {b_wer.get_result_string()}")

        hw_stats = hotwords_related_dict[rec_name]
        tp = hw_stats['tp']
        tn = hw_stats['tn']
        fp = hw_stats['fp']
        fn = hw_stats['fn']
        total_hot = tp + tn + fp + fn
        recall = tp / (tp + fn) * 100 if (tp + fn) != 0 else 0
        print(f"  Hotword stats -> tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}, all: {total_hot}, recall: {recall:.2f}%")

        final_wer_val = w_wer.get_wer()
        final_u_wer_val = u_wer.get_wer()
        final_b_wer_val = b_wer.get_wer()

        print(
            f"  => Final: {final_wer_val:.3f} (WER); "
            f"{final_u_wer_val:.3f} (U-WER); "
            f"{final_b_wer_val:.3f} (B-WER); "
            f"{recall:.2f}% (hotword recall)"
        )
        print()

        final_result_rows.append([
            rec_name,
            f"{final_wer_val:.3f}",
            result["all"],
            result["cor"],
            result["sub"],
            result["del"],
            result["ins"],
            f"{final_u_wer_val:.3f}",
            f"{final_b_wer_val:.3f}",
            f"{recall:.2f}",
            tp, tn, fp, fn
        ])

    # --- 7) Write domain-level results to CSV (if sessions exist) ---
    if utt2session:
        output_path = os.path.join(exp_folder, f"domain_result_{ref_ocr_name}.csv")
        with open(output_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "system",
                "session",
                "WER",
                "N",
                "C",
                "S",
                "D",
                "I",
                "U-WER",
                "B-WER",
            ])
            writer.writerows(domain_result_rows)

    # --- 8) Write final/overall results to CSV ---
    output_path = os.path.join(exp_folder, f"final_result_{ref_ocr_name}.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "system",
            "WER",
            "N",
            "C",
            "S",
            "D",
            "I",
            "U-WER",
            "B-WER",
            "HotwordRecall",
            "TP",
            "TN",
            "FP",
            "FN",
        ])
        writer.writerows(final_result_rows)

    print("Done.")

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
