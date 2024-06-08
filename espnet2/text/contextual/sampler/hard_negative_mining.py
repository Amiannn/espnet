import os
import math
import json
import time

import faiss
import torch
import random

import logging

from torch.nn.utils.rnn import pad_sequence

FAISS_GPU_ID = torch.cuda.current_device()
FAISS_RES    = faiss.StandardGpuResources()

print(f'Faiss GPU ID: {FAISS_GPU_ID}')

class HardNegativeSampler():
    def __init__(
        self,
        distractor_length,
        blist_idx,
        blist_words,
        blist_xphone=None,
        blist_xphone_indexis=None,
        pad_value=-1,
        contextualizer=None,
        device=None,
        use_gpu=True,
    ):
        self.distractor_length    = distractor_length
        self.blist_idx            = blist_idx
        self.blist_words          = blist_words
        self.blist_xphone         = blist_xphone
        self.blist_xphone_indexis = blist_xphone_indexis
        self.pad_value            = pad_value
        self.contextualizer       = contextualizer
        self.device               = device
        self.use_gpu              = use_gpu
        
        # TODO: Merge this two step into rareword processor
        # padding the biasing list
        self.blist_tensors = pad_sequence(
            [torch.tensor(e) for e in self.blist_idx], 
            batch_first=True, 
            padding_value=self.pad_value
        ).long()
        self.blist_tensor_ilens = (
            blist_tensors != self.pad_value
        ).sum(dim=-1)
        
        # extract xphone features
        self.blist_xphone_mean_tensors = None
        if self.blist_xphone is not None:
            # mean pooling
            element_xphone_idx = [self.blist_xphone_indexis[idx] for idx in element_idxs]
            blist_xphone_mean_tensors = torch.stack([
                torch.mean(
                    self.blist_xphone[start:end, :], dim=0
                ) for start, end in element_xphone_idx
            ])

    @torch.no_grad()
    def build_context_index(self):
        build_start_time = time_.time()
        
        blist_embeds, _, _ = self.contextualizer.forward_context_encoder(
            text_embed=self.blist_tensors,
            xphone_embed=self.blist_xphone_mean_tensors,
            ilens=self.blist_tensor_ilens,
        )
        C, D = blist_embeds.shape
        index = faiss.IndexFlatIP(D)
        if self.use_gpu:
            index = faiss.index_cpu_to_gpu(FAISS_RES, FAISS_GPU_ID, index)
        index.add(blist_embeds)
        build_time_elapsed  = time_.time() - build_start_time
        
        self.index = index
        print(f'Build index bembeds: {blist_embeds.shape}')
        print(f'Build biasing index done: {self.index}')
        print(f'Build index elapsed time: {build_time_elapsed:.4f}(s)')

    @torch.no_grad()
    def ann_method(self, gold_idx, K, unique_sorted=True):
        B = len(gold_idx)
        Gold_idx    = self.get_bword_idx(bwords)
        gold_embeds = self.bembed_keys[Gold_idx]
        G, D = gold_embeds.shape

        K_hardness_range = 20
        skip_sampling = random.random() <= self.sdrop
        if not skip_sampling:
            _, I = self.index.search(gold_embeds, K_hardness_range)
            I    = torch.from_numpy(I)
            rand_idx  = torch.randn((G, K_hardness_range)).argsort(dim=1)[:, :K]
            I_hat     = torch.gather(I, 1, rand_idx).reshape(-1)

            Q_idx  = torch.unique(I_hat, sorted=unique_sorted)
            Q_list = Q_idx.tolist()
        # Gold biasing word
        Gold_list = Gold_idx.tolist()
        # Random biasing word
        candiates = self.bindexis - set(Gold_list)
        K_rand    = self.maxlen - len(Gold_list)
        if not skip_sampling:
            candiates = candiates - set(Q_list)
            K_rand    = K_rand - K
        Rand_list = random.sample(candiates, K_rand)
        # combine all together
        B_list = Gold_list + Rand_list
        if not skip_sampling:
            B_list = B_list + Q_list
        distractors = [self.encodedlist[i] for i in B_list]
        logging.info(f'skip sampling: {skip_sampling}')
        logging.info(f'distractors  : {len(distractors)}')
        return distractors, B_list, None

if __name__ == '__main__':
    ...