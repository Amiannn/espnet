import os
import json 
import torch
import random
import logging
import numpy as np

from typing  import (
    Collection, 
    Dict, 
    Iterable, 
    List, 
    Optional, 
    Tuple, 
    Union
)

from dataclasses import (
    dataclass,
    asdict,
)

from ordered_set import OrderedSet
from torch.nn.utils.rnn import pad_sequence

from espnet2.text.contextual.sampler.hard_negative_mining import HardNegativeSampler
from espnet2.text.contextual.prompt.prompter              import WhisperPrompter
from espnet2.text.contextual.structure.trie               import TrieProcessor

from espnet2.text.abs_tokenizer              import AbsTokenizer
from espnet2.text.whisper_token_id_converter import OpenAIWhisperTokenIDConverter

seed=2022
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def read_file(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            if data == '':
                continue
            datas.append(data)
    return datas

@dataclass
class ContextSampleOutput:
    blist: torch.Tensor
    blist_idxs: list
    ilens: torch.Tensor
    trie: Optional[object] = None
    blist_xphone_mean: Optional[torch.Tensor] = None
    blist_xphone: Optional[torch.Tensor] = None
    blist_xphone_ilens: Optional[torch.Tensor] = None
    label_ctc: Optional[torch.Tensor] = None
    label_ctc_ilens: Optional[torch.Tensor] = None
    label_utterance_ctc: Optional[torch.Tensor] = None
    label_utterance_ctc_ilens: Optional[torch.Tensor] = None
    context_label: Optional[torch.Tensor] = None
    context_label_ilens: Optional[torch.Tensor] = None
    label_occurrence: Optional[torch.Tensor] = None
    label_occurrence_ilens: Optional[torch.Tensor] = None
    context_list: Optional[List[str]] = None
    context_list_idxs: Optional[List[int]] = None
    context_list_ints: Optional[List[int]] = None
    nlp_prompt: Optional[str] = None
    nlp_prompt_tensor: Optional[torch.Tensor] = None
    nlp_prompt_context_template: Optional[torch.Tensor] = None
    nlp_prompt_no_context_template: Optional[torch.Tensor] = None

class ContextSampler():
    def __init__(
        self,
        # tokenization settings
        tokenizer         : AbsTokenizer,
        token_id_converter: object,
        text_cleaner      : object,
        pad_token_value   : int,
        # prompt tokenization settings
        prompt_tokenizer         : AbsTokenizer,
        prompt_token_id_converter: object,
        prompt_text_cleaner      : object,
        # ASR model
        asr_model: object,
        # metadata for context list
        context_list_path           : str,
        context_phone_embedding_path: str,
        context_list_occurrence_path: str,
        # context settings
        use_no_context_token  : bool = True,
        no_context_token_value: int = 600,
        # sampling settings
        gold_context_dropout         : float = 0.3,
        sub_context_list_dropout     : float = 0.0,
        max_utterance_disrupt_context: int = 10,
        max_batch_disrupt_context    : int = 200,
        shuffle_sub_context_list     : bool = False,
        # hard negative context settings
        hnc_sampler_type    : str = None,
        hnc_per_gold_context: int = 1,
        hnc_dropout         : float = 0.2,
        hnc_hardness_range  : int = 20,
        hnc_use_gpu         : bool = False,
        # nlp prompt setting
        use_context_prompt                 : bool = False,
        context_prompt_has_context_template: str = "",
        context_prompt_no_context_template : str = "",
        # special structure type
        sub_context_list_structure_type: str = None,
        **kwargs,
    ):
        # tokenization settings
        self.tokenizer          = tokenizer
        self.token_id_converter = token_id_converter
        self.text_cleaner       = text_cleaner
        self.pad_token_value    = pad_token_value

        if isinstance(self.token_id_converter, OpenAIWhisperTokenIDConverter):
            self.token_id_converter_fn = self.token_id_converter.tokens2ids_withoutprompt
        else:
            self.token_id_converter_fn = self.token_id_converter.tokens2ids

        # prompt tokenization settings
        self.prompt_tokenizer          = prompt_tokenizer
        self.prompt_token_id_converter = prompt_token_id_converter
        self.prompt_text_cleaner       = prompt_text_cleaner

        if isinstance(self.prompt_token_id_converter, OpenAIWhisperTokenIDConverter):
            self.prompt_token_id_converter_fn = self.prompt_token_id_converter.tokens2ids_withoutprompt
        else:
            self.prompt_token_id_converter_fn = self.prompt_token_id_converter.tokens2ids

        # ASR model settings
        self.asr_model = asr_model

        # context metadatas
        (
            self.context_list, 
            self.context_idxs_list, 
            self.context_ints_list
        ) = self.load_context_list(context_list_path)

        # load context occurrence
        self.context_occurrence_list = None
        if context_list_occurrence_path is not None:
            self.context_occurrence_list = self.load_context_occurrence_list(context_list_occurrence_path)

        # load context embedding
        self.context_phone_embeddings = None
        if context_phone_embedding_path is not None:
            (
                self.context_phone_embeddings, 
                self.context_phone_embedding_indexis
            ) = self.load_context_phone_embedding(context_phone_embedding_path)

        # context settings
        self.use_no_context_token   = use_no_context_token
        self.no_context_token_value = no_context_token_value

        # sampling settings
        self.gold_context_dropout          = gold_context_dropout
        self.sub_context_list_dropout      = sub_context_list_dropout
        self.max_utterance_disrupt_context = max_utterance_disrupt_context
        self.max_batch_disrupt_context     = max_batch_disrupt_context
        self.shuffle_sub_context_list      = shuffle_sub_context_list

        # hard negative context settings
        self.hnc_sampler_type     = hnc_sampler_type
        self.hnc_per_gold_context = hnc_per_gold_context
        self.hnc_dropout          = hnc_dropout
        self.hnc_hardness_range   = hnc_hardness_range
        self.hnc_use_gpu          = hnc_use_gpu
        self.hnc_sampler          = None

        self.sampling_method = self.hnc_sampler_type
        if self.hnc_sampler_type is not None:
            self.hnc_sampler = HardNegativeSampler(
                sampling_method=self.hnc_sampler_type,
                hnwr_pre_gold_length=self.hnc_per_gold_context,
                hardness_range=self.hnc_hardness_range,
                blist=self.context_ints_list,
                blist_idxs=self.context_idxs_list,
                blist_words=self.context_list,
                sampler_drop=self.hnc_dropout,
                pad_value=self.pad_token_value,
                oov_value=self.no_context_token_value,
                asr_model=self.asr_model,
                # only cpu available for now
                device=None,
                use_gpu='cpu',
            )
            self.hn_sampler = self.hnc_sampler
        
        # nlp context prompt settings
        self.use_context_prompt                  = use_context_prompt
        self.context_prompt_has_context_template = context_prompt_has_context_template
        self.context_prompt_no_context_template  = context_prompt_no_context_template
        self.prompter                            = None
        
        if self.use_context_prompt:
            self.prompter = WhisperPrompter(
                id2context=self.context_list,
                prompt_template_context=self.context_prompt_has_context_template,
                prompt_template_no_context=self.context_prompt_no_context_template,
                do_context_shuffle=self.shuffle_sub_context_list,
            )

        # special structure type
        self.sub_context_list_structure_type = sub_context_list_structure_type
        if self.sub_context_list_structure_type is not None:
            self.context_constructurer = TrieProcessor(
                tokenizer=self.tokenizer,
                token_id_converter=self.token_id_converter,
                pad_value=self.pad_token_value,
                oov_value=self.no_context_token_value,
                for_transducer=True,
            )
        
        # output class
        self.output_class = ContextSampleOutput

    def text2int(self, text, text_clean=True):
        if text_clean:
            text = self.text_cleaner(text)
        tokens    = self.tokenizer.text2tokens(text)
        text_ints = self.token_id_converter_fn(tokens)
        return text_ints

    def prompt_text2int(self, text, text_clean=True):
        if text_clean:
            text = self.prompt_text_cleaner(text)
        tokens    = self.prompt_tokenizer.text2tokens(text)
        text_ints = self.prompt_token_id_converter_fn(tokens)
        return text_ints

    def load_context_list(self, path):
        # watch out! lowering every context words may cause some problem!
        context_list = [context.lower() for context in read_file(path)]
        context_idxs_list = [i for i in range(len(context_list))]
        context_ints_list = [self.text2int(context) for context in context_list]
        return context_list, context_idxs_list, context_ints_list 

    def load_context_occurrence_list(self, path):
        context_occurrence = read_file(path)
        context_occurrence = [int(occur) for occur in context_occurrence]
        return context_occurrence

    def load_context_phone_embedding(self, path):
        datas = torch.load(path)
        context_phone_embeddings        = datas['features']
        context_phone_embedding_indexis = datas['indexis']
        logging.info(f'Loaded conetxt phone embeddings ({context_phone_embeddings.shape})')
        return context_phone_embeddings, context_phone_embedding_indexis

    def tensorify(self, Xs):
        x_tensors = pad_sequence(
            [torch.tensor(x) for x in Xs], 
            batch_first=True, 
            padding_value=self.pad_token_value
        ).long()
        x_tensor_ilens = (
            x_tensors != self.pad_token_value
        ).sum(dim=-1)
        return x_tensors, x_tensor_ilens
    
    def construct_auxiliary_loss_label(
        self,
        utterance_wise_gold_contexts,
        utterance_wise_sub_context_lists, 
        batch_wise_sub_context_list,
        outputs: ContextSampleOutput,
    ):
        batch_size = len(utterance_wise_gold_contexts)
        
        if len(batch_wise_sub_context_list) == 0:
            return

        # build label for ctc objective
        utterance_wise_context_ctc_labels = []
        batch_wise_context_ctc_labels     = []
        for i in range(batch_size):
            # utterance-wise
            utterance_wise_context_ctc_label = [
                (
                    utterance_wise_sub_context_lists[i].index(idx) + (
                        1 if self.use_no_context_token else 0
                    )
                ) for idx in utterance_wise_gold_contexts[i] if idx in utterance_wise_sub_context_lists[i]
            ]
            utterance_wise_context_ctc_labels.append(utterance_wise_context_ctc_label)
            # batch-wise
            batch_wise_context_ctc_label = [
                (
                    batch_wise_sub_context_list.index(idx) + (
                        1 if self.use_no_context_token else 0
                    )
                ) for idx in utterance_wise_gold_contexts[i] if idx in batch_wise_sub_context_list
            ]
            batch_wise_context_ctc_labels.append(batch_wise_context_ctc_label)

        (
            utterance_wise_context_ctc_label_tensors, 
            utterance_wise_context_ctc_label_tensor_lens
        ) = self.tensorify(
            utterance_wise_context_ctc_labels
        )

        (
            batch_wise_context_ctc_label_tensors, 
            batch_wise_context_ctc_label_tensor_lens
        ) = self.tensorify(
            batch_wise_context_ctc_labels
        )

        outputs.label_ctc                 = batch_wise_context_ctc_label_tensors
        outputs.label_ctc_ilens           = batch_wise_context_ctc_label_tensor_lens
        outputs.label_utterance_ctc       = utterance_wise_context_ctc_label_tensors
        outputs.label_utterance_ctc_ilens = utterance_wise_context_ctc_label_tensor_lens

        # build label for context-balanced objective
        if self.context_occurrence_list is not None:
            context_occurrences_labels  = []
            no_context_occurrence_label = self.context_occurrence_list[-1]
            for i in range(batch_size):
                context_occurrences_label = [
                    (
                        self.context_occurrence_list[idx]
                    ) for idx in utterance_wise_gold_contexts[i]
                ]
                if self.use_no_context_token:
                    context_occurrences_label = [no_context_occurrence_label] + context_occurrences_label
                context_occurrences_labels.append(context_occurrences_label)

            (
                batch_wise_context_occurrences_label_tensors, 
                batch_wise_context_occurrences_label_tensor_lens
            ) = self.tensorify(
                context_occurrences_labels
            )
            outputs.label_occurrence       = batch_wise_context_occurrences_label_tensors
            outputs.label_occurrence_ilens = batch_wise_context_occurrences_label_tensor_lens

    def construct_prompt_labels(
        self,
        utterance_wise_sub_context_lists,
        outputs: ContextSampleOutput=None,
        has_confidence: bool=False
    ):
        context_prompts        = []
        context_prompt_tensors = []
        for i in range(len(utterance_wise_sub_context_lists)):
            if not has_confidence:
                elements = [{
                    "idx"       : idx,
                    "confidence": None,
                    "position"  : None, 
                } for idx in utterance_wise_sub_context_lists[i]]
            else:
                elements = [{
                    "idx"       : idx,
                    "confidence": score,
                    "position"  : pos, 
                } for idx, pos, score in utterance_wise_sub_context_lists[i]]

            context_prompt        = self.prompter.build_training_prompt(elements)
            context_prompt_tensor = torch.tensor(self.prompt_text2int(context_prompt, text_clean=False)).to(torch.int64)
            context_prompts.append(context_prompt)
            context_prompt_tensors.append(context_prompt_tensor)

        context_prompt_template, no_context_prompt_template = self.prompter.build_inference_prompt()
        context_prompt_template_tensor    = torch.tensor(self.prompt_text2int(context_prompt_template, text_clean=False)).to(torch.int64)
        no_context_prompt_template_tensor = torch.tensor(self.prompt_text2int(no_context_prompt_template, text_clean=False)).to(torch.int64)
        
        if outputs is not None:
            outputs.nlp_prompt                     = context_prompts
            outputs.nlp_prompt_tensor              = context_prompt_tensors
            outputs.nlp_prompt_context_template    = context_prompt_template_tensor
            outputs.nlp_prompt_no_context_template = no_context_prompt_template_tensor
        return (
            context_prompts,
            context_prompt_tensors,
            context_prompt_template_tensor,
            no_context_prompt_template_tensor,
        )
    
    def context_sampling(
        self, 
        utterance_wise_gold_contexts, 
        speechs=None, 
        speech_lengths=None
    ):
        batch_size = len(utterance_wise_gold_contexts)
        if self.sub_context_list_dropout > random.random():
            return [[] for _ in range(batch_size)], []

        utterance_wise_sub_context_lists = [
            list(OrderedSet(
                [context for context in contexts if random.random() > self.gold_context_dropout]
            )) for contexts in utterance_wise_gold_contexts
        ]
        batch_wise_sub_context_list      = []
        batch_wise_to_utterance_wise_ids = []

        for i, contexts in enumerate(utterance_wise_sub_context_lists):
            batch_wise_sub_context_list.extend(contexts)
            for _ in contexts:
                batch_wise_to_utterance_wise_ids.append(i)
            
        # hard negative context mining
        if self.hnc_sampler is not None:
            batch_wise_hnc_distractors = self.hnc_sampler.sample(
                gold_idx=batch_wise_sub_context_list,
                speech=speechs, 
                speech_lengths=speech_lengths,
            )
            # batch-wise
            batch_wise_sub_context_list = (
                batch_wise_sub_context_list + batch_wise_hnc_distractors
            )
            # map back to utterance-wise
            for i in range(len(batch_wise_hnc_distractors)):
                ids = batch_wise_to_utterance_wise_ids[i]
                for j in range(self.hnc_per_gold_context):
                    hnc_distractor = batch_wise_hnc_distractors[
                        (i * self.hnc_per_gold_context) + j
                    ]
                    utterance_wise_sub_context_lists[ids].append(hnc_distractor)

        # utterance-wise random sampling
        rand_distractor_lengths = (
            (self.max_utterance_disrupt_context * batch_size) - len(batch_wise_sub_context_list)
        )
        if rand_distractor_lengths > 0:
            rand_distractors = random.choices(
                self.context_idxs_list, 
                k=rand_distractor_lengths
            )
            start_idx = 0
            for i, contexts in enumerate(utterance_wise_sub_context_lists):
                distractor_length = self.max_utterance_disrupt_context - len(contexts)
                if distractor_length > 0:
                    contexts.extend(
                        rand_distractors[start_idx:start_idx + distractor_length]
                    )
                    start_idx += distractor_length
        
        # batch-wise random sampling
        rand_distractor_lengths = (
            self.max_batch_disrupt_context - len(batch_wise_sub_context_list)
        )
        if rand_distractor_lengths > 0:
            rand_distractors = random.choices(
                self.context_idxs_list, 
                k=rand_distractor_lengths
            )
            batch_wise_sub_context_list = (
                batch_wise_sub_context_list + rand_distractors
            )

        # remove repeated contexts
        utterance_wise_sub_context_lists = [
            list(OrderedSet(context)) for context in utterance_wise_sub_context_lists
        ]
        batch_wise_sub_context_list = list(OrderedSet(batch_wise_sub_context_list))

        return utterance_wise_sub_context_lists, batch_wise_sub_context_list

    def sample(
        self,
        batch_data,
        uttblists,
        **kwargs,
    ):
        speechs        = batch_data['speech']
        speech_lengths = batch_data['speech_lengths']

        (
            utterance_wise_sub_context_idxs_lists, 
            batch_wise_sub_context_idxs_list
        ) = self.context_sampling(
            utterance_wise_gold_contexts=uttblists,
            speechs=speechs,
            speech_lengths=speech_lengths
        )

        # idxs to tokens
        utterance_wise_sub_context_ints_lists = [
            [
                self.context_ints_list[idx] for idx in idxs
            ] for idxs in utterance_wise_sub_context_idxs_lists
        ]

        batch_wise_sub_context_ints_lists = [self.context_ints_list[idx] for idx in batch_wise_sub_context_idxs_list]

        # add <no-context> token
        if self.use_no_context_token:
            utterance_wise_sub_context_ints_lists = [
                ([[self.no_context_token_value]] + contexts) for contexts in utterance_wise_sub_context_ints_lists
            ]

            batch_wise_sub_context_ints_lists = (
                [[self.no_context_token_value]] + batch_wise_sub_context_ints_lists
            )
        # tensorify
        (
            batch_wise_sub_context_ints_tensors, 
            batch_wise_sub_context_ints_tensor_lens
        ) = self.tensorify(
            batch_wise_sub_context_ints_lists
        )

        outputs = self.output_class(
            blist=batch_wise_sub_context_ints_tensors,
            blist_idxs=batch_wise_sub_context_idxs_list,
            ilens=batch_wise_sub_context_ints_tensor_lens,
            context_list=[self.context_list[c] for c in batch_wise_sub_context_idxs_list],
        )
        outputs.context_list_ints=batch_wise_sub_context_ints_lists
        outputs.context_list_idxs=batch_wise_sub_context_idxs_list

        if self.use_no_context_token:
            outputs.context_list=['<no-context>'] + outputs.context_list
            outputs.context_list_idxs=[self.no_context_token_value] + outputs.context_list_idxs

        # build phone embeddings
        if self.context_phone_embeddings is not None:
            batch_wise_sub_context_embedding_phone_element_idx = [
                self.context_phone_embedding_indexis[idx] for idx in batch_wise_sub_context_idxs_list
            ]
            batch_wise_sub_context_phone_embeddings = pad_sequence(
                [
                    self.context_phone_embeddings[
                        start:end, :
                    ] for start, end in batch_wise_sub_context_embedding_phone_element_idx
                ],
                batch_first=True
            )
            batch_wise_sub_context_phone_embedding_ilens = torch.tensor(
                [(end - start) for start, end in batch_wise_sub_context_embedding_phone_element_idx]
            )
            outputs.blist_xphone       = batch_wise_sub_context_phone_embeddings
            outputs.blist_xphone_ilens = batch_wise_sub_context_phone_embedding_ilens

        # build auxiliary loss label
        self.construct_auxiliary_loss_label(
            utterance_wise_gold_contexts=uttblists,
            utterance_wise_sub_context_lists=utterance_wise_sub_context_idxs_lists, 
            batch_wise_sub_context_list=batch_wise_sub_context_idxs_list,
            outputs=outputs,
        )

        # construct prompt label
        if self.prompter is not None:
            self.construct_prompt_labels(
                utterance_wise_sub_context_lists=utterance_wise_sub_context_idxs_lists,
                outputs=outputs,
            )

        return asdict(outputs)
