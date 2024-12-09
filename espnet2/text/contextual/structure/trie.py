"""
TrieProcessor for Efficient Contextual ASR Matching

This module constructs and manages trie structures to facilitate token matching 
for contextual ASR systems. It supports batch and sequence-wise searches with 
features for efficient caching and context-based mask generation.

Key Features:
1. **Trie-based Token Matching:**  
   - Builds and searches trie structures for fast sequence matching.
2. **Batch and Sequence-wise Searches:**  
   - Supports efficient searches for multiple sequences with mask generation.
3. **Cache Management:**  
   - Caches subword paths to optimize search performance.
4. **Integration with Transducer Models:**  
   - Adds optional transducer-compatible <blank> tokens to the search.
"""

import numpy as np
from typing import Any, Dict, List, Tuple


class TrieProcessor:
    def __init__(
        self,
        pad_value: int = -1,
        no_context_token_value: int = 500,
    ):
        self.pad_value = pad_value
        self.no_context_token_value = no_context_token_value

    def build_trie(self, sequences_list: List[List[int]]):
        trie = [{}, [self.no_context_token_value]]
        for idx, sequence in enumerate(sequences_list):
            current_node = trie
            for token_id in sequence:
                if token_id == self.pad_value:
                    continue
                current_node[1].append(idx + 1)
                current_node = current_node[0].setdefault(
                    token_id, 
                    [{}, [self.no_context_token_value]]
                )
        return trie

    # @classmethod
    # def search_trie_one_step(
    #     cls,
    #     previous_token_id: int,
    #     root_node: Dict,
    #     current_node: Dict,
    #     vocab_size: int,
    # ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    #     unk_token_id = vocab_size
    #     mask = []
    #     gate_mask = None

    #     if previous_token_id == -1:  # Assuming -1 represents <blank> or start token
    #         next_node = root_node
    #         mask = list(next_node.keys()) + [unk_token_id]
    #         gate_mask = 0
    #     elif previous_token_id in current_node:
    #         next_node = current_node[previous_token_id]
    #         mask = list(next_node.keys()) + [unk_token_id]
    #         gate_mask = 0
    #     else:
    #         next_node = {}
    #         mask = [unk_token_id]
    #         gate_mask = 1

    #     mask = np.array([mask])
    #     gate_mask = np.array([gate_mask])
    #     return mask, gate_mask, next_node

    # def search_trie_sequencewise(
    #     self, sequences: List[List[int]], trie: Dict, cache: Dict
    # ) -> Tuple[List[List[int]], List[int], int]:
    #     masks = []
    #     gate_masks = []
    #     first_level_tokens = list(trie.keys())
    #     max_mask_length = len(first_level_tokens)

    #     for sequence in sequences:
    #         current_node = trie
    #         key = self._get_cache_key(sequence)
    #         sequence_length = len(sequence)

    #         if key in cache:
    #             mask = cache[key]
    #             mask[-1] = first_level_tokens
    #             masks.extend(mask)
    #             gate_masks.extend([0] * len(mask))
    #             continue

    #         for i, token_id in enumerate(sequence):
    #             if (i + 1) == sequence_length:
    #                 masks.append(first_level_tokens)
    #             elif token_id in current_node:
    #                 current_node = current_node[token_id]
    #                 masks.append(list(current_node.keys()))
    #             elif current_node != trie:
    #                 remaining_length = sequence_length - i - 1
    #                 masks.extend([[self.no_context_token_value]] * remaining_length + [first_level_tokens])
    #                 gate_masks.extend([1] * (remaining_length + 1))
    #                 break
    #             else:
    #                 masks.append(first_level_tokens)
    #             gate_masks.append(0)

    #             if max_mask_length < len(masks[-1]):
    #                 max_mask_length = len(masks[-1])

    #     if self.include_blank:
    #         masks = [first_level_tokens] + masks
    #         gate_masks = [0] + gate_masks
    #     return masks, gate_masks, max_mask_length

    # def search_batch_trie_sequencewise(
    #     self, sequences_batch: List[List[List[int]]], trie: Dict, cache: Dict
    # ) -> Tuple[np.ndarray, int, np.ndarray]:
    #     batch_size = len(sequences_batch)
    #     max_batch_mask_length = 0
    #     max_batch_sequence_length = 0

    #     batch_masks = []
    #     batch_gate_masks = []

    #     for sequences in sequences_batch:
    #         masks, gate_masks, max_mask_length = self.search_trie_sequencewise(
    #             sequences, trie, cache
    #         )
    #         batch_masks.append(masks)
    #         batch_gate_masks.append(gate_masks)

    #         max_batch_mask_length = max(max_batch_mask_length, max_mask_length)
    #         max_batch_sequence_length = max(max_batch_sequence_length, len(masks))

    #     batch_masks_mat = np.full(
    #         (batch_size, max_batch_sequence_length, max_batch_mask_length),
    #         fill_value=self.pad_value,
    #         dtype=np.int64,
    #     )

    #     for i, masks in enumerate(batch_masks):
    #         for j, mask in enumerate(masks):
    #             batch_masks_mat[i, j, : len(mask)] = mask

    #     batch_gate_masks_mat = np.full(
    #         (batch_size, max_batch_sequence_length),
    #         fill_value=self.pad_value,
    #         dtype=np.int64,
    #     )

    #     for i, gate_masks in enumerate(batch_gate_masks):
    #         batch_gate_masks_mat[i, : len(gate_masks)] = gate_masks

    #     return batch_masks_mat, max_batch_mask_length, batch_gate_masks_mat