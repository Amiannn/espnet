import os
import json 
import random
import logging
import numpy as np

from pathlib import Path
from typing import Collection, Dict, Iterable, List, Optional, Tuple, Union
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.hugging_face_token_id_converter import HuggingFaceTokenIDConverter
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

random.seed(0)

class TrieProcessor():
    def __init__(
        self, 
        tokenizer=None,
        token_id_converter=None,
        pad_value=-1,
        oov_value=500,
        for_transducer=True,
    ):
        self.tokenizer          = tokenizer
        self.token_id_converter = token_id_converter
        self.pad_value          = pad_value
        self.for_transducer     = for_transducer
        self.oov_value         = oov_value

    @classmethod
    def _get_cache_key(cls, element):
        return " ".join([str(e) for e in element])

    @classmethod
    def build_trie(cls, elements_list):
        tree  = {}
        cache = {}
        for elements in elements_list:
            now    = tree
            childs = []
            for element in elements:
                if element not in now:
                    now[element] = {}
                now = now[element]
                childs.append(now)
            key = cls._get_cache_key(elements)
            cache[key] = childs

        for key in cache:
            cache[key] = [
                sorted(
                    child.keys()
                ) for child in cache[key]
            ]
        return tree, cache

    @classmethod
    def search_trie_one_step(cls, previous_token_id, root, previous_node, token_list):
        ooKB_id        = len(token_list)
        previous_token = token_list[previous_token_id]
        mask      = []
        gate_mask = None
        if previous_token.endswith("▁") or "<blank>" in previous_token:
            now_node  = root
            mask      = list(now_node.keys()) + [ooKB_id]
            gate_mask = 0
        elif previous_token_id in previous_node:
            now_node  = previous_node[previous_token_id]
            mask      = list(now_node.keys()) + [ooKB_id]
            gate_mask = 0
        else:
            now_node  = {}
            mask      = [ooKB_id]
            gate_mask = 1
        
        mask      = np.array([mask])
        gate_mask = np.array([gate_mask]) 
        return (
            mask,
            gate_mask,
            now_node
        )

    def search_trie_sequencewise(self, words, tree, cache):
        masks      = []
        masks_gate = []
        tokens     = []

        first_level  = list(tree.keys())
        max_mask_len = len(first_level)
        for word in words:
            now = tree
            key = self._get_cache_key(word)
            subword_length = len(word)
            # cache
            if key in cache:
                mask     = cache[key]
                mask[-1] = first_level
                masks.extend(mask)
                masks_gate.extend([0] * len(mask))
                continue
            for i in range(subword_length):
                char = word[i]
                if (i + 1) == subword_length:
                    masks.append(first_level)
                elif char in now:
                    now = now[char]
                    masks.append(list(now.keys()))
                elif now != tree:
                    masks.extend([[self.oov_value]] * (subword_length - i - 1) + [first_level])
                    masks_gate.extend([1] * (subword_length - i))
                    break
                else:
                    masks.append(first_level)
                masks_gate.append(0)
                if max_mask_len < len(masks[-1]):
                    len(masks[-1])

        # insert <blank> at the first token
        if self.for_transducer:
            masks      = [first_level] + masks
            masks_gate = [0] + masks_gate
        return masks, masks_gate, max_mask_len
    
    def search_batch_trie_sequencewise(self, texts, tree, cache):
        batch_size = len(texts)
        max_batch_mask_len = 0
        max_batch_text_len = 0

        batch_masks              = []
        batch_masks_gate         = []
        batch_max_mask_len       = []
        batch_max_masks_gate_len = []

        for text in texts:
            masks, masks_gate, max_mask_len = self.search_trie_sequencewise(text, tree, cache)
            batch_masks.append(masks)
            batch_masks_gate.append(masks_gate)
            batch_max_mask_len.append(max_mask_len)
            batch_max_masks_gate_len.append(len(masks_gate))

            if max_batch_mask_len < max_mask_len:
                max_batch_mask_len = max_mask_len
            if max_batch_text_len < len(masks):
                max_batch_text_len = len(masks)

        batch_masks_mat = np.zeros(
            (batch_size, max_batch_text_len, max_batch_mask_len),
            dtype=np.int64
        ) + self.pad_value

        for i in range(batch_size):
            for j in range(len(batch_masks[i])):
                batch_masks_mat[i, j, :len(batch_masks[i][j])] = batch_masks[i][j]

        batch_masks_gate_mat = np.zeros(
            (batch_size, max_batch_text_len),
            dtype=np.int64
        ) + self.pad_value

        for i in range(batch_size):
            batch_masks_gate_mat[i, :len(batch_masks_gate[i])] = batch_masks_gate[i]
        
        return batch_masks_mat, batch_max_mask_len, batch_masks_gate_mat, batch_max_masks_gate_len


# _, tree, cache = self.build_batch_trie(uttblists_resolve)
# masks_mat, max_mask_len, masks_gate_mat, max_masks_gate_len = self.search_batch_trie_sequencewise(
#     texts, tree, cache
# )
# return masks_mat, max_mask_len, masks_gate_mat, max_masks_gate_len