import os
import json 
import torch
import random
import logging
import numpy as np

from pathlib import Path
from typing  import (
    Collection, 
    Dict, 
    Iterable, 
    List, 
    Optional, 
    Tuple, 
    Union
)
from torch.nn.utils.rnn import pad_sequence

from espnet2.text.build_tokenizer                 import build_tokenizer
from espnet2.text.token_id_converter              import TokenIDConverter
from espnet2.text.whisper_token_id_converter      import OpenAIWhisperTokenIDConverter
from espnet2.text.hugging_face_token_id_converter import HuggingFaceTokenIDConverter
from espnet2.text.cleaner                         import TextCleaner

from espnet2.text.contextual.sampler.hard_negative_mining import HardNegativeSampler
from espnet2.text.contextual.structure.trie               import TrieProcessor

from ordered_set import OrderedSet

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

class RarewordProcessor():
    def __init__(
        self, 
        blist_path, 
        blist_occurrence_path=None, 
        blist_xphonebert_path=None, 
        drop_out=0.1,
        blist_max=500,
        for_transducer=True,
        pad_value=-1,
        oov_value=500,
        token_type: str = None,
        token_list: Union[Path, str, Iterable[str]] = None,
        bpemodel: Union[Path, str, Iterable[str]] = None,
        g2p_type: str = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        nonsplit_symbol: Iterable[str] = None,
        # tokenization encode (text2token) args, e.g. BPE dropout, only applied in training
        encode_kwargs: Dict = None,
        # only use for whisper
        whisper_language: str = None,
        whisper_task: str = None,
        sot_asr: bool = False,
        # contextual asr
        structure_type: str = "none",
        sampling_method: str = "none",
        hnwr_pre_gold_length: int = 10,
        hardness_range: int = 20,
        sampler_drop: int = 0.1,
        asr_model: object = None,
        use_oov: bool = True,
        use_gpu: bool = False,
        text_cleaner: Collection[str] = None,
        prompt_template_context    : str = "THE TOPIC OF TODAY'S",
        prompt_template_no_context : str = "OKAY THEN I'LL CONTINUE.",
    ):
        self.tokenizer = build_tokenizer(
            token_type=token_type,
            bpemodel=bpemodel,
            delimiter=delimiter,
            space_symbol=space_symbol,
            non_linguistic_symbols=non_linguistic_symbols,
            g2p_type=g2p_type,
            nonsplit_symbol=nonsplit_symbol,
            whisper_language=whisper_language,
            whisper_task=whisper_task,
        )
        if token_type == "hugging_face":
            self.token_id_converter = HuggingFaceTokenIDConverter(
                model_name_or_path=bpemodel
            )
        elif bpemodel not in ["whisper_en", "whisper_multilingual"]:
            self.token_id_converter = TokenIDConverter(
                token_list=token_list,
                unk_symbol=unk_symbol,
            )
        else:
            print(f'bpemodel: {bpemodel}')
            self.token_id_converter = OpenAIWhisperTokenIDConverter(
                model_type=bpemodel,
                added_tokens_txt=non_linguistic_symbols,
                language=whisper_language or "en",
                task=whisper_task or "transcribe",
            )
        self.text_cleaner = TextCleaner(text_cleaner)
        if isinstance(self.token_id_converter, OpenAIWhisperTokenIDConverter):
            self.token_id_converter_fn = self.token_id_converter.tokens2ids_withoutprompt
        else:
            self.token_id_converter_fn = self.token_id_converter.tokens2ids

        # load rareword datas
        self.blist, self.blist_words = self.load_blist(blist_path)
        self.blist_idxs              = [i for i in range(len(self.blist))]
        self.blist_occurrence_path   = blist_occurrence_path
        self.blist_xphonebert_path   = blist_xphonebert_path
        self.blist_xphone            = None
        self.blist_xphone_indexis    = None

        if blist_occurrence_path is not None:
            logging.info(f'Loading blist occurrence datas...')
            self.blist_occurrence = None
            with open(blist_occurrence_path, 'r', encoding='utf-8') as fr:
                self.blist_occurrence = list(map(int, fr.read().split('\n')[:-1]))

        if blist_xphonebert_path is not None:
            logging.info(f'Loading XPhoneBERT features...')
            datas = torch.load(blist_xphonebert_path)
            self.blist_xphone         = datas['features']
            self.blist_xphone_indexis = datas['indexis']
            logging.info(f'xphone: {self.blist_xphone.shape}')

        self.drop_out       = drop_out
        self.blist_max      = blist_max
        self.pad_value      = pad_value
        self.for_transducer = for_transducer
        self.oov_value      = oov_value
        self.use_oov        = use_oov
        self.use_gpu        = use_gpu

        # prompt template
        self.prompt_template_context    = prompt_template_context
        self.prompt_template_no_context = prompt_template_no_context

        # asr model
        self.asr_model = asr_model

        # structure
        self.structure_type = structure_type
        self.trie_processor = TrieProcessor(
            tokenizer=self.tokenizer,
            token_id_converter=self.token_id_converter,
            pad_value=self.pad_value,
            oov_value=self.oov_value,
            for_transducer=self.for_transducer,
        )
        # sampling
        self.sampling_method      = sampling_method
        self.hnwr_pre_gold_length = hnwr_pre_gold_length
        self.hardness_range       = hardness_range
        self.sampler_drop         = sampler_drop
        if self.sampling_method is not None:
            self.hn_sampler = HardNegativeSampler(
                sampling_method=self.sampling_method,
                hnwr_pre_gold_length=self.hnwr_pre_gold_length,
                hardness_range=self.hardness_range,
                blist=self.blist,
                blist_idxs=self.blist_idxs,
                blist_words=self.blist_words,
                blist_xphone=self.blist_xphone,
                blist_xphone_indexis=self.blist_xphone_indexis,
                sampler_drop=self.sampler_drop,
                pad_value=self.pad_value,
                oov_value=self.oov_value,
                asr_model=self.asr_model,
                device=None,
                use_gpu=self.use_gpu,
            )

    def load_blist(self, blist_path):
        blist, blist_words = [], []
        with open(blist_path, 'r', encoding='utf-8') as frs:
            for fr in frs:
                bword = fr.replace('\n', '')
                if bword == '':
                    continue
                blist_words.append(bword)
                bword     = self.text_cleaner(bword)
                tokens    = self.tokenizer.text2tokens(bword)
                text_ints = self.token_id_converter_fn(tokens)
                blist.append(text_ints)
        return blist, blist_words

    def build_batch_contextual(self, batch_data, uttblist):
        droped_uttblist = list(OrderedSet([b for b in uttblist if random.random() > self.drop_out]))
        hnw_distractors = []
        # hard negative mining
        if self.sampling_method is not None:
            hnw_distractors = self.hn_sampler.sample(
                gold_idx=droped_uttblist,
                speech=batch_data['speech'], 
                speech_lengths=batch_data['speech_lengths'],
            )
            hnw_distractors = list(OrderedSet(hnw_distractors) - OrderedSet(droped_uttblist))
        blist = droped_uttblist + hnw_distractors
        rand_distractors = []
        logging.info(f'droped_uttblist: {len(droped_uttblist)}')
        logging.info(f'hnw_distractors: {len(hnw_distractors)}')
        # random sampling
        if self.blist_max > len(blist):
            rand_distractors = random.choices(
                self.blist_idxs, 
                k = (self.blist_max - len(blist))
            )
            rand_distractors = list(OrderedSet(rand_distractors) - OrderedSet(blist))
        blist = blist + rand_distractors
        logging.info(f'blist: {len(blist)}')
        # remove empty bword
        blist = [b for b in blist if len(self.blist[b]) > 0]
        return blist

    def build_batch_trie(self, elements):
        tree, cache = self.trie_processor.build_trie(elements)
        return tree

    def build_auxiliary_loss_label(
        self, 
        uttblist_batch,
        element_idxs, 
        pad_value,
        use_oov=True
    ):
        batch_size       = len(uttblist_batch)
        rareword_labels  = []
        # build ctc label
        for i in range(batch_size):
            rareword_label = [
                (
                    element_idxs.index(b_idx) + (1 if use_oov else 0)
                ) for b_idx in uttblist_batch[i] if b_idx in element_idxs
            ]
            rareword_labels.append(rareword_label)

        # to tensor
        label_ctc_tensors = pad_sequence(
            [torch.tensor(b) for b in rareword_labels], 
            batch_first=True, 
            padding_value=pad_value
        ).long()
        label_ctc_tensor_ilens = (
            label_ctc_tensors != pad_value
        ).sum(dim=-1)

        context_label_tensors = pad_sequence(
            [torch.tensor(b) for b in rareword_labels], 
            batch_first=True, 
            padding_value=-1
        ).long()
        context_label_tensor_ilens = (
            context_label_tensors != -1
        ).sum(dim=-1)

        # label occurrence
        label_occurrences = []
        label_occurrence_tensors      = None
        label_occurrence_tensor_ilens = None
        if self.blist_occurrence_path is not None:
            for i in range(batch_size):
                label_occurrence = [
                    (
                        self.blist_occurrence[b_idx]
                    ) for b_idx in uttblist_batch[i]
                ]
                if use_oov:
                    label_occurrence = [self.blist_occurrence[-1]] + label_occurrence
                label_occurrences.append(label_occurrence)

            # to tensor
            label_occurrence_tensors = pad_sequence(
                [torch.tensor(b) for b in label_occurrences], 
                batch_first=True, 
                padding_value=pad_value
            ).long()
            label_occurrence_tensor_ilens = (
                label_occurrence_tensors != pad_value
            ).sum(dim=-1)

        return (
            label_ctc_tensors, 
            label_ctc_tensor_ilens,
            context_label_tensors,
            context_label_tensor_ilens,
            label_occurrence_tensors,
            label_occurrence_tensor_ilens,
        )

    def build_context_prompt(self, elements):
        if len(elements) == 0:
            nlp_prompt = self.prompt_template_no_context
        else:
            shuffle_elements = elements.copy()
            random.shuffle(shuffle_elements)
            contexts   = ", ".join([self.blist_words[e] for e in shuffle_elements])
            nlp_prompt = f'{self.prompt_template_context} {contexts}. {self.prompt_template_no_context}'
        return self.build_prompt(nlp_prompt, inference_template=False)

    def build_inference_prompt(self):
        _, prompt_inference_context    = self.build_prompt(self.prompt_template_context, inference_template=True)
        _, prompt_inference_no_context = self.build_prompt(self.prompt_template_no_context, inference_template=True)
        return prompt_inference_context, prompt_inference_no_context

    def build_prompt(self, nlp_prompt, inference_template=False):
        prompt_tokens     = self.tokenizer.text2tokens(nlp_prompt)
        prompt_tokens_str = " ".join(prompt_tokens)
        if len(prompt_tokens_str.split()) > 1:
            prompt_ids    = self.token_id_converter.tokenizer.tokenizer.convert_tokens_to_ids(prompt_tokens_str.split())
            prompt_tensor = torch.tensor(prompt_ids).to(torch.int64)
        else:
            prompt_ids    = [self.token_id_converter.tokenizer.tokenizer.convert_tokens_to_ids(prompt_tokens_str)]
            prompt_tensor = torch.tensor(prompt_ids).to(torch.int64)
        # special tokens
        if isinstance(self.token_id_converter, OpenAIWhisperTokenIDConverter):
            lang_token              = self.token_id_converter.tokenizer.sot_sequence_including_notimestamps[1]
            no_time_stamp_token     = self.token_id_converter.tokenizer.sot_sequence_including_notimestamps[3]
            if not inference_template:
                prompt_tokens_special = [lang_token] + prompt_ids + [no_time_stamp_token]
            else:
                prompt_tokens_special = prompt_ids
            prompt_tensor= torch.tensor(prompt_tokens_special).to(torch.int64)
        return nlp_prompt, prompt_tensor
        
    def sample(
        self,
        batch_data,
        uttblists,
        pad_value=-1
    ):
        batch_size = len(uttblists)
        output     = {}
        
        uttblists_resolve       = []
        uttblists_batch_resolve = []
        for i in range(batch_size):
            uttblists_resolve.extend(uttblists[i])
            uttblists_batch_resolve.append(uttblists[i])
        
        element_idxs = self.build_batch_contextual(batch_data, uttblists_resolve)
        elements     = [self.blist[idx] for idx in element_idxs]
        # oov
        if self.use_oov:
            elements = [[self.oov_value]] + elements
        # tensorlize
        element_tensors = pad_sequence(
            [torch.tensor(e) for e in elements], 
            batch_first=True, 
            padding_value=pad_value
        ).long()
        element_tensor_ilens = (element_tensors != pad_value).sum(dim=-1)

        output = {
            'blist'     : element_tensors,
            'blist_idxs': element_idxs,
            'ilens'     : element_tensor_ilens,
        }
        # ssl features
        element_xphone_mean_tensors = None
        element_xphone_tensors      = None
        if self.blist_xphonebert_path is not None:
            # mean pooling
            element_xphone_idx = [self.blist_xphone_indexis[idx] for idx in element_idxs]
            element_xphone_mean_tensors = torch.stack([
                torch.mean(
                    self.blist_xphone[start:end, :], dim=0
                ) for start, end in element_xphone_idx
            ])
            element_xphone_tensors = pad_sequence(
                [
                    self.blist_xphone[
                        start:end, :
                    ] for start, end in element_xphone_idx
                ],
                batch_first=True
            )
        output['blist_xphone_mean'] = element_xphone_mean_tensors
        output['blist_xphone']      = element_xphone_tensors
        # structure blist
        if self.structure_type == "trie":
            tree           = self.build_batch_trie(elements)
            output['trie'] = tree
        # for attention guided auxiliary loss
        (
            label_ctc_tensors, 
            label_ctc_tensor_ilens, 
            context_label_tensors,
            context_label_tensors_ilens,
            label_occurrence_tensors,
            label_occurrence_tensor_ilens,
        ) = self.build_auxiliary_loss_label(
            uttblists_batch_resolve,
            element_idxs,
            pad_value,
            use_oov=self.use_oov,
        )
        
        output['label_ctc']              = label_ctc_tensors
        output['label_ctc_ilens']        = label_ctc_tensor_ilens
        output['context_label']          = context_label_tensors
        output['context_label_ilens']    = context_label_tensors_ilens
        output['label_occurrence']       = label_occurrence_tensors
        output['label_occurrence_ilens'] = label_occurrence_tensor_ilens
        output['context_list']           = [self.blist_words[e] for e in element_idxs]
        output['context_list_idxs']      = [self.blist[e] for e in element_idxs]
        
        # special tokens
        if isinstance(self.token_id_converter, OpenAIWhisperTokenIDConverter):
            # build text prompt
            nlp_prompt, nlp_prompt_tensor = self.build_context_prompt(element_idxs)
            prompt_inference_context, prompt_inference_no_context = self.build_inference_prompt()
            output['nlp_prompt']                     = nlp_prompt
            output['nlp_prompt_tensor']              = nlp_prompt_tensor
            output['nlp_prompt_context_template']    = prompt_inference_context
            output['nlp_prompt_no_context_template'] = prompt_inference_no_context
            if self.use_oov:
                output['context_list']      = ['<no-context>'] + output['context_list']
                output['context_list_idxs'] = [self.oov_value] + output['context_list_idxs']
        return output
