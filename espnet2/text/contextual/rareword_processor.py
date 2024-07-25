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
        # random sampling
        if self.blist_max > len(blist):
            rand_distractors = random.choices(
                self.blist_idxs, 
                k = (self.blist_max - len(blist))
            )
            rand_distractors = list(OrderedSet(rand_distractors) - OrderedSet(blist))
        blist = blist + rand_distractors
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
            contexts   = ", ".join([self.blist_words[e] for e in elements])
            nlp_prompt = f'{self.prompt_template_context} {contexts} {self.prompt_template_no_context}'
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
        if self.blist_xphonebert_path is not None:
            # mean pooling
            element_xphone_idx = [self.blist_xphone_indexis[idx] for idx in element_idxs]
            element_xphone_mean_tensors = torch.stack([
                torch.mean(
                    self.blist_xphone[start:end, :], dim=0
                ) for start, end in element_xphone_idx
            ])
        output['blist_xphone_mean'] = element_xphone_mean_tensors
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
        # build text prompt
        _, nlp_prompt_tensor = self.build_context_prompt(element_idxs)
        prompt_inference_context, prompt_inference_no_context = self.build_inference_prompt()
        output['label_ctc']              = label_ctc_tensors
        output['label_ctc_ilens']        = label_ctc_tensor_ilens
        output['context_label']          = context_label_tensors
        output['context_label_ilens']    = context_label_tensors_ilens
        output['label_occurrence']       = label_occurrence_tensors
        output['label_occurrence_ilens'] = label_occurrence_tensor_ilens
        output['context_list']           = [self.blist_words[e] for e in element_idxs]
        output['context_list_idxs']      = [self.blist[e] for e in element_idxs]
        output['nlp_prompt_tensor']              = nlp_prompt_tensor
        output['nlp_prompt_context_template']    = prompt_inference_context
        output['nlp_prompt_no_context_template'] = prompt_inference_no_context
        if self.use_oov:
            output['context_list']      = ['<no-context>'] + output['context_list']
            output['context_list_idxs'] = [self.oov_value] + output['context_list_idxs']
        return output

if __name__ == '__main__':
    print('hello')
    blist_path = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/local/rareword_f15.txt"
    batch_data = [{'text': np.array([465,  79, 125,   3,  84, 240,  13,   5, 364,  27, 347,  30, 317,
        65, 255, 150,  49,   6, 509, 183,  79, 124, 336,  13,  10,  75,
        11, 217,   4, 184,  18,  13,  56,  14,  54,  35,   4,  19,  98,
        22,  43,  74,  39, 420,   4, 337, 144,  22, 217,   4, 340,  13,
       159,  79, 189,  50,  10,   5, 124,  18, 375,  59,   5,   7, 328,
         8,   9, 102,  83,   8, 437,   5, 124,  14, 187, 259,   9, 534,
        22, 217]), 'uttblist': np.array([ 65, 255, 124, 336,  13, 184,  18,  13,  22,  43,  74,   5, 124,
        18, 375,  59,   5,   7,   9, 102,  83]), 'textsegment': np.array([[ 0,  1],
       [ 1,  2],
       [ 2,  7],
       [ 7, 10],
       [10, 11],
       [11, 12],
       [12, 13],
       [13, 15],
       [15, 16],
       [16, 17],
       [17, 18],
       [18, 19],
       [19, 20],
       [20, 21],
       [21, 24],
       [24, 25],
       [25, 28],
       [28, 29],
       [29, 32],
       [32, 33],
       [33, 36],
       [36, 37],
       [37, 39],
       [39, 42],
       [42, 44],
       [44, 45],
       [45, 47],
       [47, 49],
       [49, 50],
       [50, 52],
       [52, 53],
       [53, 54],
       [54, 55],
       [55, 56],
       [56, 57],
       [57, 64],
       [64, 65],
       [65, 66],
       [66, 69],
       [69, 70],
       [70, 71],
       [71, 75],
       [75, 76],
       [76, 78],
       [78, 80]]), 'uttblistsegment': np.array([[ 0,  2],
       [ 2,  5],
       [ 5,  8],
       [ 8, 11],
       [11, 18],
       [18, 21]])}, {'text': np.array([183,  86, 243,   3,  40,   6, 448,  62,  45,  12, 363, 229,   2,
        16,  37,  58, 285,  22,  24,  65,  19,  62,  27,  89,  65,  12,
       256,  30, 195,   3, 359,  13,  34, 370, 288,  15,  48,   2,  11,
       411, 223,  84,  46,  51,  21, 411, 122, 446,   2, 593,  23,  21,
       182, 200,  60,  26, 547, 105,  30, 388, 524,  44,  20, 117,  62,
        40, 106,   9, 184,   9,  77,  27,  52, 228,  42,  22,   5, 215,
        61,   2, 349, 244,  17, 186, 274, 120]), 'uttblist': np.array([  9, 184,   9,  77,  27]), 'textsegment': np.array([[ 0,  1],
       [ 1,  2],
       [ 2,  5],
       [ 5,  6],
       [ 6,  7],
       [ 7, 11],
       [11, 12],
       [12, 13],
       [13, 17],
       [17, 23],
       [23, 27],
       [27, 28],
       [28, 32],
       [32, 33],
       [33, 34],
       [34, 37],
       [37, 38],
       [38, 41],
       [41, 43],
       [43, 44],
       [44, 47],
       [47, 48],
       [48, 49],
       [49, 50],
       [50, 54],
       [54, 55],
       [55, 57],
       [57, 58],
       [58, 59],
       [59, 60],
       [60, 61],
       [61, 62],
       [62, 66],
       [66, 67],
       [67, 72],
       [72, 73],
       [73, 78],
       [78, 79],
       [79, 80],
       [80, 81],
       [81, 82],
       [82, 83],
       [83, 86]]), 'uttblistsegment': np.array([[0, 5]])}, {'text': np.array([183, 228,  25,   2,   9,  66,  18, 291,   6,   3,  45,   3,  74,
        60, 512,  12, 266, 310,  78, 528,  64, 156, 148, 280,  59,  11,
        15,  40,   4, 124, 112,  13,  61,   2,  65,  50,   6,  77,  23,
        10, 137, 106,  21,  23, 270,  17,   2,  37,  14,  10,   6,   2,
       339,   5, 151,  43,   7,  84,   9, 196,  45,  70,   3,   7,   4,
        17,  86,  90, 126,   5, 148, 130,  12,  18, 131,  37,  46,  89,
       100,   9, 307,  88, 272, 398,  64,  15,  92,  55,  27, 124,  37,
       129, 106,  91,  63,  16,  27, 186,  59,  22, 174,  48]), 'uttblist': np.array([  9,  66,  18, 291,   3,  45,   3,  74, 512,  12, 266, 310, 339,
         5, 151,  43,   7,  84,   9, 196,  45,  70,   3,   7,  90, 126,
         5, 148, 130,  12,  18, 131,  37,  46,  91,  63,  16,  27, 186,
        59,  22, 174,  48]), 'textsegment': np.array([[  0,   1],
       [  1,   3],
       [  3,   4],
       [  4,   8],
       [  8,   9],
       [  9,  13],
       [ 13,  14],
       [ 14,  18],
       [ 18,  19],
       [ 19,  20],
       [ 20,  23],
       [ 23,  24],
       [ 24,  28],
       [ 28,  29],
       [ 29,  32],
       [ 32,  33],
       [ 33,  34],
       [ 34,  36],
       [ 36,  37],
       [ 37,  40],
       [ 40,  41],
       [ 41,  42],
       [ 42,  45],
       [ 45,  46],
       [ 46,  47],
       [ 47,  50],
       [ 50,  51],
       [ 51,  52],
       [ 52,  57],
       [ 57,  60],
       [ 60,  64],
       [ 64,  65],
       [ 65,  66],
       [ 66,  67],
       [ 67,  71],
       [ 71,  77],
       [ 77,  81],
       [ 81,  83],
       [ 83,  84],
       [ 84,  89],
       [ 89,  92],
       [ 92,  93],
       [ 93,  97],
       [ 97, 102]]), 'uttblistsegment': np.array([[ 0,  4],
       [ 4,  8],
       [ 8, 12],
       [12, 17],
       [17, 20],
       [20, 24],
       [24, 28],
       [28, 34],
       [34, 38],
       [38, 43]])}, {'text': np.array([ 93,  51, 278,  10, 234, 358,  22, 383,  27,  49,   2, 188, 187,
         3,  68,   5,  13,  52,  98,  71,  38, 297,  12,  38, 127,  67,
        46, 107, 114,   9, 108, 125, 102,   9,  48, 239,   2,  67,  46,
         6,  10, 364,  83, 317, 587,  67,  16,   7, 203,  33,  58, 122,
         2,   3, 115,  40,   4,  30,  51, 565, 123,  35,   3,  90,   5,
       122, 110, 123, 127, 137, 238,  69, 104,   8,  64, 478, 429,   2,
        84, 130,  27,   6,   2, 166, 119,   8, 412,   2,  67,  46, 429]), 'uttblist': np.array([ 71,  38, 297,  12,  38, 127,  67,  46,  67,  46,  67,  16,   7,
       203,  33,  58, 122,  84, 130,  27,  67,  46]), 'textsegment': np.array([[ 0,  1],
       [ 1,  2],
       [ 2,  3],
       [ 3,  4],
       [ 4,  5],
       [ 5,  6],
       [ 6,  9],
       [ 9, 10],
       [10, 11],
       [11, 13],
       [13, 17],
       [17, 18],
       [18, 19],
       [19, 25],
       [25, 27],
       [27, 28],
       [28, 31],
       [31, 35],
       [35, 36],
       [36, 37],
       [37, 39],
       [39, 40],
       [40, 41],
       [41, 43],
       [43, 44],
       [44, 45],
       [45, 48],
       [48, 52],
       [52, 53],
       [53, 56],
       [56, 57],
       [57, 58],
       [58, 59],
       [59, 62],
       [62, 66],
       [66, 67],
       [67, 69],
       [69, 70],
       [70, 71],
       [71, 72],
       [72, 73],
       [73, 74],
       [74, 76],
       [76, 77],
       [77, 78],
       [78, 81],
       [81, 82],
       [82, 83],
       [83, 85],
       [85, 86],
       [86, 87],
       [87, 88],
       [88, 90],
       [90, 91]]), 'uttblistsegment': np.array([[ 0,  6],
       [ 6,  8],
       [ 8, 10],
       [10, 13],
       [13, 17],
       [17, 20],
       [20, 22]])}, {'text': np.array([158, 106,  23,   3, 170, 210,   6,  22,  18, 451,  27,   2, 203,
        33,  83,  80,   6, 149, 248,   9, 201, 143, 357, 120,  52, 143,
        89, 440,   8,  22,  18, 451,  28,   2, 203,  33,  83,  72, 158,
       238,  69, 124,  75,   9, 218,   8,  16, 101, 255,   2,   5, 535,
       145,   8, 224,  99,  36,  63,  16, 101,  29, 138,  95, 223,  23,
       261,  12,  26, 215,   8,  37,  15, 163, 340,  38,  41,  18,  39,
       223,  29,  66, 373]), 'uttblist': np.array([ 22,  18, 451,  27, 149, 248,   9, 201,  22,  18, 451,  28,  63,
        16, 101,  29, 138,  23, 261,  12,  26, 215]), 'textsegment': np.array([[ 0,  1],
       [ 1,  2],
       [ 2,  6],
       [ 6,  7],
       [ 7, 11],
       [11, 12],
       [12, 15],
       [15, 16],
       [16, 17],
       [17, 21],
       [21, 22],
       [22, 24],
       [24, 25],
       [25, 26],
       [26, 28],
       [28, 29],
       [29, 33],
       [33, 34],
       [34, 37],
       [37, 38],
       [38, 39],
       [39, 40],
       [40, 41],
       [41, 45],
       [45, 46],
       [46, 49],
       [49, 50],
       [50, 52],
       [52, 53],
       [53, 54],
       [54, 56],
       [56, 57],
       [57, 62],
       [62, 64],
       [64, 69],
       [69, 70],
       [70, 73],
       [73, 79],
       [79, 82]]), 'uttblistsegment': np.array([[ 0,  4],
       [ 4,  8],
       [ 8, 12],
       [12, 17],
       [17, 22]])}, {'text': np.array([245,  10, 184,   3, 312, 103,  11,  12,   3,  12,  98, 278,   2,
        88, 345,  70, 315,  18,  19,  23, 463,  28,  15, 333,  12,   3,
        13, 265, 435,  58,  22,  43,   7, 113,   3, 463,  28,   4, 467,
        49,  87,  52,  21,  42,  55,  16,  48,  11,  26,  45,   3,  40,
       107,   2, 444, 118,  15, 419,   2, 206,   2,   3,  19,  38, 286,
       193, 277, 191,  13,  49,   2,  29,  96, 305, 299,  87]), 'uttblist': np.array([103,  11,  12,   3,  12,  98,  15, 333,  12,   3,  13, 435,  58,
        22,  43,   7]), 'textsegment': np.array([[ 0,  1],
       [ 1,  2],
       [ 2,  5],
       [ 5, 11],
       [11, 12],
       [12, 13],
       [13, 17],
       [17, 22],
       [22, 27],
       [27, 28],
       [28, 33],
       [33, 37],
       [37, 38],
       [38, 39],
       [39, 40],
       [40, 41],
       [41, 42],
       [42, 47],
       [47, 52],
       [52, 53],
       [53, 54],
       [54, 56],
       [56, 58],
       [58, 59],
       [59, 60],
       [60, 61],
       [61, 65],
       [65, 66],
       [66, 69],
       [69, 70],
       [70, 71],
       [71, 74],
       [74, 75],
       [75, 76]]), 'uttblistsegment': np.array([[ 0,  6],
       [ 6, 11],
       [11, 16]])}, {'text': np.array([  3,  42,  13, 168,  22,  28,   9,  90,  38,  28,  58, 407,   6,
         2, 523,  37,  41, 165,  70,   7,  57, 278, 135,  59,  53,  23,
        43,  46,   8,  82,   2, 193,   6,  24, 261,  54,  26,  49,   2,
       141,  14, 122,  32, 149, 136,   3,   9,  48, 243,  20,   5, 405,
        13,   2, 280,  77, 175,  18, 307, 106,  15,  89, 204,   3, 108,
         8, 143,  65, 179, 218,  68,  16,  11,  40,  50,   2,  22,  64,
       140,  51, 164, 299, 150,  17,  89, 123,  41, 215,   8,   2,  84,
        29,   7, 169, 294,  42,  24,  43,  46]), 'uttblist': np.array([ 37,  41, 165,  59,  53,  23,  43,  46,  24, 261,  54,  26,  49,
       243,  20,   5, 405,  13,  77, 175,  18, 307,  15,  89, 204,   3,
       108,  68,  16,  11,  40,  89, 123,  41, 215, 169, 294,  42,  24,
        43,  46]), 'textsegment': np.array([[ 0,  3],
       [ 3,  4],
       [ 4,  6],
       [ 6, 10],
       [10, 12],
       [12, 13],
       [13, 14],
       [14, 15],
       [15, 18],
       [18, 20],
       [20, 21],
       [21, 22],
       [22, 23],
       [23, 28],
       [28, 29],
       [29, 30],
       [30, 31],
       [31, 32],
       [32, 33],
       [33, 38],
       [38, 39],
       [39, 42],
       [42, 43],
       [43, 48],
       [48, 53],
       [53, 54],
       [54, 55],
       [55, 59],
       [59, 60],
       [60, 65],
       [65, 66],
       [66, 67],
       [67, 70],
       [70, 74],
       [74, 75],
       [75, 76],
       [76, 79],
       [79, 80],
       [80, 81],
       [81, 82],
       [82, 83],
       [83, 84],
       [84, 88],
       [88, 89],
       [89, 90],
       [90, 93],
       [93, 99]]), 'uttblistsegment': np.array([[ 0,  3],
       [ 3,  8],
       [ 8, 13],
       [13, 18],
       [18, 22],
       [22, 27],
       [27, 31],
       [31, 35],
       [35, 41]])}, {'text': np.array([ 29,  92, 269, 168,  61,   2, 294,  42,  24,  40,   4,   2,  18,
       295,  12,  65,  40,  31,   9, 153,   2, 424,   4, 403,  43,  25,
        82, 239, 150,  93,  68,  99,   8,  29,  71,  83,  33, 358,  17,
       143, 289, 523,  66,  24,  40,   4,   8, 104,  14, 236,  54, 473,
       131, 255,   8, 150,  17,   2,  53,  15, 231,   4,   8,   5,  36,
        17,   2,  24, 196,  65,  92,   7,  17,   2,   5,  54, 221, 351,
       213,  40,   4,  60,  16,  23,  20, 140]), 'uttblist': np.array([294,  42,  24,  40,  18, 295,  12,  65,  40,  66,  24,  40,   5,
        54, 221, 351, 213,  40,  16,  23,  20, 140]), 'textsegment': np.array([[ 0,  3],
       [ 3,  4],
       [ 4,  5],
       [ 5,  6],
       [ 6, 10],
       [10, 11],
       [11, 12],
       [12, 17],
       [17, 18],
       [18, 20],
       [20, 21],
       [21, 22],
       [22, 23],
       [23, 26],
       [26, 27],
       [27, 28],
       [28, 29],
       [29, 30],
       [30, 32],
       [32, 33],
       [33, 36],
       [36, 38],
       [38, 39],
       [39, 40],
       [40, 41],
       [41, 42],
       [42, 45],
       [45, 46],
       [46, 47],
       [47, 48],
       [48, 52],
       [52, 54],
       [54, 55],
       [55, 56],
       [56, 57],
       [57, 58],
       [58, 61],
       [61, 62],
       [62, 63],
       [63, 65],
       [65, 66],
       [66, 67],
       [67, 69],
       [69, 72],
       [72, 73],
       [73, 74],
       [74, 80],
       [80, 81],
       [81, 82],
       [82, 86]]), 'uttblistsegment': np.array([[ 0,  4],
       [ 4,  9],
       [ 9, 12],
       [12, 18],
       [18, 22]])}]
    tokenizer_config = {
        'blist_max' : 10,
        'token_type': 'bpe', 
        'bpemodel': '/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/data/en_token_list/bpe_unigram600suffix/bpe.model', 
        'delimiter': None, 
        'space_symbol': '<space>', 
        'non_linguistic_symbols': None, 'g2p_type': None, 'nonsplit_symbol': None, 
        'token_list': ['<blank>', '<unk>', 'THE▁', 'C', 'AND▁', 'S', 'OF▁', 'S▁', 'TO▁', 'T', 'A▁', 'G', 'I', 'ED▁', 'E', 'RE', 'D', 'IN▁', 'P', 'R', 'N', 'F', 'O', 'IN', 'B', 'T▁', 'L', 'ING▁', '▁', 'W', 'I▁', 'HE▁', 'WAS▁', 'A', 'THAT▁', 'E▁', 'IT▁', 'AR', 'U', 'H', 'ES▁', 'M', 'RI', "'", 'HIS▁', 'AN', 'D▁', 'Y▁', 'LY▁', 'ON▁', 'AS▁', 'HAD▁', 'WITH▁', 'ST', 'Y', 'EN', 'HER▁', 'YOU▁', 'K', 'DE', 'AT▁', 'FOR▁', 'V', 'UN', 'TH', 'SE', 'RO', 'LI', 'LO', 'NOT▁', 'TI', 'AL', 'BUT▁', 'IS▁', 'ER▁', 'SI', 'OR', 'CH', 'ONE▁', 'SHE▁', 'OR▁', 'ME▁', 'BE▁', 'K▁', 'LA', 'LE', 'ALL▁', 'HIM▁', 'BE', 'CON', 'HO', 'PO', 'AT', 'THEY▁', 'MY▁', 'ME', 'ON', 'BY▁', 'AN▁', 'VE▁', 'DI', 'RA', 'AC', 'MA', 'HAVE▁', 'SO▁', 'WERE▁', 'WHICH▁', 'TED▁', 'AL▁', 'THIS▁', 'FROM▁', 'AD', 'SU', 'FI', 'AS', 'SAID▁', 'ER', 'TH▁', 'SE▁', 'RY▁', 'MO', 'EN▁', 'FOR', 'HE', 'EX', 'NE', 'M▁', 'VI', 'TS▁', 'SH', 'BO', 'COM', 'PRO', 'EL', 'ARE▁', 'FE', 'WE▁', 'N▁', 'NO▁', 'ERS▁', 'QU', 'THERE▁', 'THEIR▁', 'LE▁', 'WHEN▁', 'TE', 'TA', 'TY▁', 'PER', 'THEM▁', 'TER', 'WOULD▁', 'OLD▁', 'PA', 'CO', 'IR', 'IF▁', 'WHO▁', 'WHAT▁', 'TER▁', 'MAN▁', 'ATION▁', 'ST▁', 'BEEN▁', 'OUR▁', 'CA', 'UP▁', 'OUT▁', 'PRE', 'AP', 'TION▁', 'IT', 'FA', 'US', 'AM', 'VE', 'TUR', 'DO', 'PAR', 'PE', 'NO', 'LU', 'THEN▁', 'WI', 'SO', 'HI', 'P▁', 'TO', 'COULD▁', 'RE▁', 'Z', 'WILL▁', 'KING▁', 'EAR▁', 'DIS', 'EST▁', 'LL▁', 'SP', 'HA', 'ENCE▁', 'TING▁', 'IS', 'WE', 'DU', 'AND', 'MORE▁', 'SOME▁', 'US▁', 'PI', 'ABLE▁', 'NOW▁', 'VERY▁', 'GU', 'EM', 'ITY▁', 'WA', 'H▁', 'ATE▁', 'LL', 'DO▁', 'NA', 'DER', 'ANT▁', 'LEA', 'PLA', 'BU', 'SA', 'CU', 'INTO▁', 'OWN▁', 'ET▁', 'KE', 'PU', 'LITTLE▁', 'MENT▁', 'VER', 'TE▁', 'DID▁', 'LIKE▁', 'IM', 'ABOUT▁', 'OUR', 'TRA', 'TIME▁', 'THAN▁', 'YOUR▁', 'RED▁', 'MI', 'OTHER▁', 'HU', 'ION▁', 'ANCE▁', 'STR', 'WELL▁', 'W▁', 'L▁', 'ES', 'ANY▁', 'ITS▁', 'MIS', 'AB', 'AGE▁', 'MAR', 'UPON▁', 'OVER▁', 'TU', 'DAY▁', 'TEN', 'CH▁', 'ALLY▁', 'GRA', 'CAME▁', 'MEN▁', 'STO', 'LED▁', 'AM▁', 'GA', 'ONLY▁', 'COME▁', 'TWO▁', 'UG', 'HOW▁', 'VEN', 'INE▁', 'NESS▁', 'EL▁', 'HAS▁', 'BA', 'LONG▁', 'AFTER▁', 'IC▁', 'WAY▁', 'CAR', 'SC', 'HAR', 'MADE▁', 'MIN', 'STE', 'BEFORE▁', 'MOST▁', 'ILL', 'FO', 'GE', 'DOWN▁', 'DER▁', 'BL', 'IONS▁', 'SUCH▁', 'THESE▁', 'DE▁', 'MEN', 'KED▁', 'TRU', 'WHERE▁', 'FUL▁', 'BI', 'CAN▁', 'SEE▁', 'KNOW▁', 'GO▁', 'JE', 'GREAT▁', 'LOW▁', 'MUCH▁', 'NEVER▁', 'MISTER▁', 'GOOD▁', 'SHOULD▁', 'EVEN▁', 'ICE▁', 'STA', 'LESS▁', 'JO', 'BLE▁', 'MUST▁', 'AV', 'DA', 'ISH▁', 'MON', 'TRI', 'KE▁', 'BACK▁', 'YING▁', 'AIR▁', 'AU', 'IOUS▁', 'AGAIN▁', 'MU', 'FIRST▁', 'F▁', 'GO', 'EVER▁', 'VA', 'COR', 'OUS▁', 'ATED▁', 'COUNT', 'ROUND▁', 'OVER', 'LING▁', 'HERE▁', 'HIMSELF▁', 'SHED▁', 'MIL', 'G▁', 'THOUGH▁', 'SIDE▁', 'CL', 'MAY▁', 'JUST▁', 'WENT▁', 'SAY▁', 'NG▁', 'PASS', 'HER', 'NED▁', 'MIGHT▁', 'FR', 'MAN', 'HOUSE▁', 'JU', 'SON▁', 'PEN', 'THROUGH▁', 'EYES▁', 'MAKE▁', 'TOO▁', 'THOUGHT▁', 'WITHOUT▁', 'THINK▁', 'GEN', 'THOSE▁', 'MANY▁', 'SPEC', 'INTER', 'WHILE▁', 'AWAY▁', 'LIFE▁', 'HEAD▁', 'SUR', 'NTLY▁', 'RIGHT▁', 'DON', 'TAKE▁', 'PORT', 'EVERY▁', 'NIGHT▁', 'WARD▁', 'WAR', 'IMP', 'ALL', 'GET▁', 'STILL▁', 'BEING▁', 'FOUND▁', 'NOTHING▁', 'LES▁', 'LAST▁', 'TURNED▁', 'ILL▁', 'YOUNG▁', 'SURE▁', 'INGS▁', 'PEOPLE▁', 'YET▁', 'THREE▁', 'FACE▁', 'CUR', 'OFF▁', 'ROOM▁', 'OUT', 'ASKED▁', 'SAW▁', 'END▁', 'FER', 'MISSUS▁', 'EACH▁', 'SAME▁', 'SHA', 'SENT▁', 'OUL', 'LET▁', 'SOL', 'YOU', 'PLACE▁', 'UNDER▁', 'TOOK▁', 'LIGHT▁', 'LEFT▁', 'PER▁', 'PRESS', 'USE▁', 'ANOTHER▁', 'ONCE▁', 'TELL▁', 'SHALL▁', 'OFF', 'SEEMED▁', 'ALWAYS▁', 'NEW▁', 'ATIONS▁', 'J', 'CESS', 'USED▁', 'WHY▁', 'HEARD▁', 'LOOKED▁', 'GIVE▁', 'PUT▁', 'JA', 'BECAUSE▁', 'THINGS▁', 'BODY▁', 'FATHER▁', 'SOMETHING▁', 'OWING▁', 'LOOK▁', 'ROW▁', 'GOING▁', 'MOTHER▁', 'MIND▁', 'WORK▁', 'GOT▁', 'CENT', 'HAVING▁', 'SOON▁', 'KNEW▁', 'HEART▁', 'FAR▁', 'AGAINST▁', 'WORLD▁', 'FEW▁', 'ICAL▁', 'STOOD▁', 'BEGAN▁', 'SIR▁', 'BETTER▁', 'DOOR▁', 'CALLED▁', 'YEARS▁', 'MOMENT▁', 'ENOUGH▁', 'WOMAN▁', 'TOGETHER▁', 'LIGHT', 'OWED▁', 'READ▁', 'WHOLE▁', 'COURSE▁', 'BETWEEN▁', 'FELT▁', 'LONG', 'HALF▁', 'FULLY▁', 'MORNING▁', 'DENT', 'WOOD', 'HERSELF▁', 'OLD', 'DAYS▁', 'HOWEVER▁', 'WATER▁', 'WHITE▁', 'PERHAPS▁', 'REPLIED▁', 'GIRL▁', 'QUITE▁', 'HUNDRED▁', 'WORDS▁', 'MYSELF▁', 'VOICE▁', 'EARLY▁', 'OUGHT▁', 'AIL▁', 'WORD▁', 'WHOM▁', 'EITHER▁', 'AMONG▁', 'ENDED▁', 'TAKEN▁', 'UNTIL▁', 'ANYTHING▁', 'NEXT▁', 'POSSIBLE▁', 'KIND▁', 'BROUGHT▁', 'EAST▁', 'LOOKING▁', 'ROAD▁', 'SMALL▁', 'RATHER▁', 'BELIEVE▁', 'SINCE▁', 'MONEY▁', 'OPEN▁', 'INDEED▁', 'DOUBT', 'CERTAIN▁', 'TWENTY▁', 'MATTER▁', 'HELD▁', 'EXPECT', 'DIRECT', 'ANSWERED▁', 'THERE', 'WHOSE▁', 'SHIP▁', 'HIGH▁', 'THEMSELVES▁', 'APPEARED▁', 'BLACK▁', 'NATURE▁', 'BEHIND▁', 'POWER▁', 'IZED▁', 'CHILD▁', 'UNCLE▁', 'DEATH▁', 'KNOWN▁', 'OFTEN▁', 'LADY▁', 'POSITION▁', 'KEEP▁', 'CHILDREN▁', 'WIFE▁', 'JOHN▁', 'LARGE▁', 'GIVEN▁', 'EIGHT▁', 'SHORT▁', 'SAYS▁', 'EVERYTHING▁', 'GENERAL▁', 'DOCTOR▁', 'ABOVE▁', 'HAPPY▁', 'Q', 'X', '<sos/eos>'], 'unk_symbol': '<unk>'
    }
    
    contextual_processor = RarewordProcessor(blist_path, 0.0, **tokenizer_config)
    token_list  = tokenizer_config['token_list']
    
    # masks_mat, max_mask_len, masks_gate_mat, max_masks_gate_len = trie_search(batch_data)
    # print(masks_mat)
    # print(masks_mat.shape)

    output = contextual_processor.sample([batch_data[0]])
    print(f'blist:')
    for blist in output['blist']:
        print(blist)
    print(f'_' * 30)
    print(f'label:')
    for i, token in enumerate(batch_data[0]['text']):
        print(f'{token} ({output["label"][0][i]})')
    print(f'_' * 30)
    # text = [0] + batch_data[0]['text']

    # previous_node  = output['trie']
    # token_list     = contextual_processor.token_id_converter.token_list
    # token_list_oov = token_list + ['oov']

    # for token_ints in previous_node:
    #     tokens = token_list_oov[token_ints]
    #     print("".join(tokens))

    # for i, previous_token_id in enumerate(text):
    #     mask, gate_mask, now_node = TrieProcessor.search_trie_one_step(
    #         previous_token_id, 
    #         output['trie'], 
    #         previous_node,
    #         token_list
    #     )
    #     print(f'[{i}]' + '_' * 30)
    #     print(f'previous token: {previous_token_id}, token: {token_list[previous_token_id]}')
    #     print(f'gate: {gate_mask}')
    #     print(f'mask: {mask[0]}')
    #     print(f'mask: {[token_list_oov[m] for m in mask[0]]}')
    #     previous_node = now_node