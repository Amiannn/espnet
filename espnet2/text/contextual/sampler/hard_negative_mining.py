import os
import math
import json
import time as time_
import numpy as np
import torch
import random

import logging

from torch.nn.utils.rnn import pad_sequence

try:
    import faiss
except:
    logging.info(f'Warning: Cannot import faiss!')

class HardNegativeSampler():
    def __init__(
        self,
        sampling_method,
        hnwr_pre_gold_length,
        hardness_range,
        blist,
        blist_idxs,
        blist_words,
        blist_xphone=None,
        blist_xphone_indexis=None,
        sampler_drop=0.1,
        pad_value=-1,
        oov_value=0,
        asr_model=None,
        device=None,
        use_gpu=True,
    ):
        self.sampling_method      = sampling_method
        self.blist                = blist
        self.blist_idxs           = blist_idxs
        self.blist_words          = blist_words
        self.blist_xphone         = blist_xphone
        self.blist_xphone_indexis = blist_xphone_indexis
        self.sampler_drop         = sampler_drop
        self.pad_value            = pad_value
        self.oov_value            = oov_value
        self.asr_model            = asr_model
        self.device               = device
        
        self.hnwr_pre_gold_length = hnwr_pre_gold_length

        # TODO: Merge this two step into rareword processor
        # padding the biasing list
        elements = [[self.oov_value]] + self.blist
        blist_tensors = pad_sequence(
            [torch.tensor(e) for e in elements], 
            batch_first=True, 
            padding_value=self.pad_value
        ).long()
        blist_tensor_ilens = (
            blist_tensors != self.pad_value
        ).sum(dim=-1)
        self.blist_tensors      = blist_tensors
        self.blist_tensor_ilens = blist_tensor_ilens
        
        # extract xphone features
        self.blist_xphone_mean_tensors = None
        if self.blist_xphone is not None:
            # mean pooling
            element_xphone_idx = [self.blist_xphone_indexis[idx] for idx in self.blist_idxs]
            self.blist_xphone_mean_tensors = torch.stack([
                torch.mean(
                    self.blist_xphone[start:end, :], dim=0
                ) for start, end in element_xphone_idx
            ])
            self.blist_xphone_tensors = pad_sequence(
                [
                    self.blist_xphone[
                        start:end, :
                    ] for start, end in element_xphone_idx
                ],  
                batch_first=True
            )
            self.blist_xphone_tensor_ilens = torch.tensor(
                [(end - start) for start, end in element_xphone_idx]
            ).long()

        self.hardness_range = (
            hardness_range if self.hnwr_pre_gold_length < hardness_range else self.hnwr_pre_gold_length
        )

        # select sampler
        if self.sampling_method == "ann_hnw":
            self.sample = self.ann_hnw_method
        elif self.sampling_method == "qhnw":
            self.sample = self.qhnw_method

        # gpu acc
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.faiss_gpu_id = torch.cuda.current_device()
            self.faiss_res    = faiss.StandardGpuResources()
        self.index                = None
        self.blist_context_embeds = None

    def update_index(self):
        if self.sampling_method == "ann_hnw":
            logging.info(f"Updating ANN-HNW's index.")
            self.build_context_index()
        elif self.sampling_method == "qhnw":
            logging.info(f"Updating Q-HNW's index.")
            self.build_context_index(forward_key=True)

    @torch.no_grad()
    def build_context_index(self, forward_key=False):
        build_start_time = time_.time()
        kwargs = {
            'text_embed': self.blist_tensors,
            'ilens'     : self.blist_tensor_ilens,
        }
        if self.blist_xphone_mean_tensors is not None:
            device = next(self.asr_model.parameters()).device
            kwargs['xphone_embed']      = self.blist_xphone_tensors.to(device)
            kwargs['xphone_mean_embed'] = self.blist_xphone_mean_tensors.to(device)
            kwargs['xphone_ilens']      = self.blist_xphone_tensor_ilens.to(device)

        blist_context_embeds, _, _ = self.asr_model.contextualizer.forward_context_encoder(
            **kwargs
        )
        # remove oov
        blist_context_embeds = blist_context_embeds[1:, :]
        # check if needs to forward key
        if forward_key:
            blist_context_embeds = self.asr_model.contextualizer.adapter.norm_before_x2(
                blist_context_embeds.unsqueeze(0)
            )
            blist_context_embeds = self.asr_model.contextualizer.adapter.attention_layer.linear_k(
                blist_context_embeds
            ).squeeze(0)

        blist_context_embeds = blist_context_embeds.cpu()
        C, D  = blist_context_embeds.shape
        self.index = faiss.IndexFlatIP(D)
        # TODO: Move index to gpu
        # if self.use_gpu:
        #     self.index = faiss.index_cpu_to_gpu(self.faiss_res, self.faiss_gpu_id, self.index)
        self.index.add(blist_context_embeds)
        self.blist_context_embeds = blist_context_embeds
    
        build_time_elapsed  = time_.time() - build_start_time
        logging.info(f'Build index bembeds: {blist_context_embeds.shape}')
        logging.info(f'Build biasing index done: {self.index}')
        logging.info(f'Build index elapsed time: {build_time_elapsed:.4f}(s)')

    @torch.no_grad()
    def ann_hnw_method(self, gold_idx, speech=None, speech_lengths=None, unique_sort=True):
        skip_sampling   = random.random() <= self.sampler_drop
        ann_distractors = []
        if not skip_sampling:
            gold_embeds = self.blist_context_embeds[gold_idx]
            G, D     = gold_embeds.shape
            S, I     = self.index.search(gold_embeds, self.hardness_range)
            I        = torch.from_numpy(I)
            rand_idx = torch.randn((G, self.hardness_range)).argsort(dim=1)[:, :self.hnwr_pre_gold_length]
            I_hat    = torch.gather(I, 1, rand_idx).reshape(-1)
            # ann_distractors = torch.unique(I_hat, sorted=unique_sort)
            # ann_distractors = ann_distractors.tolist()
            ann_distractors = I_hat.tolist()
        return ann_distractors

    @torch.no_grad()
    def qhnw_method(self, gold_idx, speech=None, speech_lengths=None, unique_sort=True):
        skip_sampling    = random.random() <= self.sampler_drop
        acoustic_embeds, acoustic_embeds_olens = self.asr_model.encode(speech, speech_lengths)
        acoustic_flatten_embeds = []
        for embeds, olens in zip(acoustic_embeds, acoustic_embeds_olens):
            acoustic_flatten_embeds.append(embeds[:olens, :].unsqueeze(0))
        acoustic_flatten_embeds = torch.cat(acoustic_flatten_embeds, dim=0)
        acoustic_flatten_embeds = self.asr_model.contextualizer.adapter.norm_before_x1(
            acoustic_flatten_embeds
        )
        queries = self.asr_model.contextualizer.adapter.attention_layer.linear_q(
            acoustic_flatten_embeds
        ).squeeze(0)
        qhnw_distractors  = []
        if not skip_sampling:
            G, D = queries.shape
            _, I = self.index.search(queries, self.hardness_range)
            I    = torch.from_numpy(I)
            rand_idx = torch.randn((G, self.hardness_range)).argsort(dim=1)[:, :self.hnwr_pre_gold_length]
            I_hat    = torch.gather(I, 1, rand_idx).reshape(-1)
            qhnw_distractors = torch.unique(I_hat, sorted=unique_sort)
            qhnw_distractors = qhnw_distractors.tolist()
        return qhnw_distractors

if __name__ == '__main__':
    ...