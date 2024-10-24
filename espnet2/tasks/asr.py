import os
import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.hugging_face_transformers_decoder import (  # noqa: H301
    HuggingFaceTransformersDecoder,
)
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.avhubert_encoder import FairseqAVHubertEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,
    FairseqHubertPretrainEncoder,
    TorchAudioHuBERTPretrainEncoder,
)
from espnet2.asr.encoder.longformer_encoder import LongformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.transformer_encoder_multispkr import (
    TransformerEncoder as TransformerEncoderMultiSpkr,
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.whisper import WhisperFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.maskctc_model import MaskCTCModel
from espnet2.asr.pit_espnet_model import ESPnetASRModel as PITESPnetModel
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.postencoder.length_adaptor_postencoder import LengthAdaptorPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.joint_network import (
    JointNetwork,
    JointBiasingNetwork,
)
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import (
    CommonCollateFn,
    ContextualCollateFn
)
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    CommonPreprocessor,
    ContextualPreprocessor,
    CommonPreprocessor_multi,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none

from espnet2.text.contextual.rareword_processor import RarewordProcessor
from espnet2.text.contextual.context_sampler    import ContextSampler
from espnet2.asr.contextual_asr_espnet_model import ESPnetContextualASRModel

from espnet2.asr.contextualizer import CONTEXTUALIZERS
from espnet2.asr.contextualizer.component.utils import CustomLinear

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        whisper=WhisperFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetASRModel,
        maskctc=MaskCTCModel,
        pit_espnet=PITESPnetModel,
        contextual_asr_espnet=ESPnetContextualASRModel
    ),
    type_check=AbsESPnetModel,
    default="espnet",
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        transformer_multispkr=TransformerEncoderMultiSpkr,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        torchaudiohubert=TorchAudioHuBERTPretrainEncoder,
        longformer=LongformerEncoder,
        branchformer=BranchformerEncoder,
        whisper=OpenAIWhisperEncoder,
        e_branchformer=EBranchformerEncoder,
        avhubert=FairseqAVHubertEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
        length_adaptor=LengthAdaptorPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        transducer=TransducerDecoder,
        mlm=MLMDecoder,
        whisper=OpenAIWhisperDecoder,
        hugging_face_transformers=HuggingFaceTransformersDecoder,
        s4=S4Decoder,
    ),
    type_check=AbsDecoder,
    default=None,
    optional=True,
)
contextualizer_choices = ClassChoices(
    "contextualizer",
    classes=CONTEXTUALIZERS,
    default=list(CONTEXTUALIZERS.keys())[0],
)
contextual_choices = ClassChoices(
    "contextual",
    classes=dict(
        rareword_processor=RarewordProcessor,
        context_sampler=ContextSampler,
    ),
    default="rareword_processor",
)
preprocessor_choices = ClassChoices(
    "preprocessor",
    classes=dict(
        default=CommonPreprocessor,
        multi=CommonPreprocessor_multi,
        contextual=ContextualPreprocessor
    ),
    type_check=AbsPreprocessor,
    default="default",
)


class ASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --model and --model_conf
        model_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --contextualizer and --contextualizer_conf
        contextualizer_choices,
        # --contextual and --contextual_conf
        contextual_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--context_token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to context token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--joint_net_conf",
            action=NestedDictAction,
            default=None,
            help="The keyword arguments for joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--use_lang_prompt",
            type=str2bool,
            default=False,
            help="Use language id as prompt",
        )
        group.add_argument(
            "--use_nlp_prompt",
            type=str2bool,
            default=False,
            help="Use natural language phrases as prompt",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=[
                "bpe",
                "char",
                "word",
                "phn",
                "hugging_face",
                "whisper_en",
                "whisper_multilingual",
            ],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--context_token_type",
            type=str,
            default="bpe",
            choices=[
                "bpe",
                "char",
                "word",
                "phn",
                "hugging_face",
                "whisper_en",
                "whisper_multilingual",
            ],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        group.add_argument(
            "--context_bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[
                None,
                "tacotron",
                "jaconv",
                "vietnamese",
                "whisper_en",
                "whisper_basic",
            ],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--context_cleaner",
            type=str_or_none,
            choices=[
                None,
                "tacotron",
                "jaconv",
                "vietnamese",
                "whisper_en",
                "whisper_basic",
            ],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        group.add_argument(
            "--aux_ctc_tasks",
            type=str,
            nargs="+",
            default=[],
            help="Auxillary tasks to train on using CTC loss. ",
        )
        group.add_argument(
            "--collate_fn_type",
            type=str,
            choices=["default", "contextual"],
            default="default",
            help="Specify collate function type",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        collate_fn_type = getattr(args, "collate_fn_type", 'default')
        if collate_fn_type == "default":
            return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
        elif collate_fn_type == "contextual":
            return ContextualCollateFn(
                float_pad_value=0.0, 
                int_pad_value=-1, 
                contextual_processor=cls.contextual_processor
            )

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            try:
                _ = getattr(args, "preprocessor")
            except AttributeError:
                setattr(args, "preprocessor", "default")
                setattr(args, "preprocessor_conf", dict())
            except Exception as e:
                raise e

            preprocessor_class = preprocessor_choices.get_class(args.preprocessor)
            retval = preprocessor_class(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                short_noise_thres=args.short_noise_thres
                if hasattr(args, "short_noise_thres")
                else 0.5,
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
                aux_task_names=args.aux_ctc_tasks
                if hasattr(args, "aux_ctc_tasks")
                else None,
                use_lang_prompt=args.use_lang_prompt
                if hasattr(args, "use_lang_prompt")
                else None,
                **args.preprocessor_conf,
                use_nlp_prompt=args.use_nlp_prompt
                if hasattr(args, "use_nlp_prompt")
                else None,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_preprocess_for_context_sampler_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            try:
                _ = getattr(args, "preprocessor")
            except AttributeError:
                setattr(args, "preprocessor", "default")
                setattr(args, "preprocessor_conf", dict())
            except Exception as e:
                raise e
            logging.info(f'args.token_list: {args.context_token_list}')
            preprocessor_class = preprocessor_choices.get_class(args.preprocessor)
            retval = preprocessor_class(
                train=train,
                token_type=args.context_token_type,
                token_list=args.context_token_list,
                bpemodel=args.context_bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.context_cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                short_noise_thres=args.short_noise_thres
                if hasattr(args, "short_noise_thres")
                else 0.5,
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
                aux_task_names=args.aux_ctc_tasks
                if hasattr(args, "aux_ctc_tasks")
                else None,
                use_lang_prompt=args.use_lang_prompt
                if hasattr(args, "use_lang_prompt")
                else None,
                **args.preprocessor_conf,
                use_nlp_prompt=args.use_nlp_prompt
                if hasattr(args, "use_nlp_prompt")
                else None,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        MAX_REFERENCE_NUM = 4

        retval = ["text_spk{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval = retval + ["prompt"]
        retval = tuple(retval)

        logging.info(f"Optional Data Names: {retval }")
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_contextualizer(cls, vocab_size: int, args: argparse.Namespace):
        contextualizer_type  = args.contextualizer_conf.get("contextualizer_type", None)
        contextualizer_class = contextualizer_choices.get_class(contextualizer_type)
        contextualizer       = contextualizer_class(
            vocab_size=vocab_size,
            padding_idx=-1,
            use_oov=args.contextual_conf.get("use_oov", True),
            **args.contextualizer_conf, 
        )
        return contextualizer

    @classmethod
    def build_contextual_processor(cls, args: argparse.Namespace, model: object):
        contextual_type  = args.contextual_conf.get("contextual_type", None)
        contextual_class = contextual_choices.get_class(contextual_type)
        logging.info(f'args.contextual_conf: {args.contextual_conf}')
        if contextual_type == "rareword_processor":
            # TODO: remove rareword processor and replace by context sampler
            contextual_processor = contextual_class(
                blist_path=args.contextual_conf.get("blist_path", None), 
                blist_occurrence_path=args.contextual_conf.get("blist_occurrence_path", None), 
                blist_xphonebert_path=args.contextual_conf.get("blist_xphone_path", None),
                drop_out=args.contextual_conf.get("blist_drop_out", 0),
                full_drop_out=args.contextual_conf.get("full_drop_out", 0),
                blist_max=args.contextual_conf.get("blist_max", 500),
                pad_value=-1,
                oov_value=len(args.token_list),
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                g2p_type=args.g2p,
                non_linguistic_symbols=args.non_linguistic_symbols,
                structure_type=args.contextual_conf.get("structure_type", "none"),
                sampling_method=args.contextual_conf.get("sampling_method", "none"),
                hnwr_pre_gold_length=args.contextual_conf.get("hnwr_pre_gold_length", 5),
                hardness_range=args.contextual_conf.get("hardness_range", 20),
                sampler_drop=args.contextual_conf.get("sampler_drop", 0.5),
                asr_model=model,
                use_oov=args.contextual_conf.get("use_oov", True),
                use_gpu=args.contextual_conf.get("use_gpu", True),
                text_cleaner=args.cleaner,
                prompt_template_context=args.contextual_conf.get("prompt_template_context", "THE TOPIC OF TODAY'S"),
                prompt_template_no_context=args.contextual_conf.get("prompt_template_no_context", "OKAY THEN I'LL CONTINUE."),
                do_context_shuffle=args.contextual_conf.get("do_context_shuffle", False),
                **args.preprocessor_conf,
            )
        elif contextual_type == "context_sampler":
            preprocessor = cls.build_preprocess_fn(args, train=True)
            context_preprocessor = cls.build_preprocess_for_context_sampler_fn(args, train=True)
            contextual_processor = contextual_class(
                tokenizer=context_preprocessor.tokenizer,
                token_id_converter=context_preprocessor.token_id_converter,
                text_cleaner=context_preprocessor.text_cleaner,
                prompt_tokenizer=preprocessor.tokenizer,
                prompt_token_id_converter=preprocessor.token_id_converter,
                prompt_text_cleaner=preprocessor.text_cleaner,
                pad_token_value=-1,
                asr_model=model,
                no_context_token_value=len(args.context_token_list),
                **args.contextual_conf,
            )
        return contextual_processor

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRModel:
        assert check_argument_types()
        logging.info(f'args.token_list: {args.token_list}')
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")

        if hasattr(args, 'context_token_list'):
            if isinstance(args.context_token_list, str):
                with open(args.context_token_list, encoding="utf-8") as f:
                    context_token_list = [line.rstrip() for line in f]
                args.context_token_list = list(context_token_list)
            elif isinstance(args.context_token_list, (tuple, list)):
                context_token_list = list(args.context_token_list)
            else:
                raise RuntimeError("context_token_list must be str or list")
        else:
            args.context_token_list = args.token_list
            args.context_token_type = args.token_type
            args.context_bpemodel   = args.bpemodel
            args.context_cleaner    = args.cleaner
            context_token_list      = token_list

        # If use multi-blank transducer criterion,
        # big blank symbols are added just before the standard blank
        if args.model_conf.get("transducer_multi_blank_durations", None) is not None:
            sym_blank = args.model_conf.get("sym_blank", "<blank>")
            blank_idx = token_list.index(sym_blank)
            for dur in args.model_conf.get("transducer_multi_blank_durations"):
                if f"<blank{dur}>" not in token_list:  # avoid this during inference
                    token_list.insert(blank_idx, f"<blank{dur}>")
            args.token_list = token_list

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")
        context_vocab_size = len(context_token_list)
        logging.info(f"Context Vocabulary size: {context_vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 5. Decoder
        if getattr(args, "decoder", None) is not None:
            decoder_class = decoder_choices.get_class(args.decoder)

            if args.decoder == "transducer":
                decoder = decoder_class(
                    vocab_size,
                    embed_pad=0,
                    **args.decoder_conf,
                )
                joiner_type = args.joint_net_conf.get("joint_type", None)
                if joiner_type == "biasing":
                    joint_class = JointBiasingNetwork
                else:
                    joint_class = JointNetwork
                joint_network = joint_class(
                    vocab_size,
                    encoder.output_size(),
                    decoder.dunits,
                    **args.joint_net_conf,
                )
            else:
                decoder = decoder_class(
                    vocab_size=vocab_size,
                    encoder_output_size=encoder_output_size,
                    **args.decoder_conf,
                )
                joint_network = None
        else:
            decoder = None
            joint_network = None

        # ?. contextualizer methods
        ctc_lo_fn = None
        contextualizer_conf = getattr(args, "contextualizer_conf", {})
        if contextualizer_conf != {}:
            contextualizer = cls.build_contextualizer(context_vocab_size, args)
            if "embed_share_weight_ctc" in contextualizer_conf and contextualizer_conf['embed_share_weight_ctc']:
                ctc_lo_fn = CustomLinear(
                    embedding=contextualizer.context_encoder.embed,
                    no_context_embedding=contextualizer.context_encoder.oov_embed,
                )
        else:
            contextualizer = None
        if 'ctc_lo_fn' in args.ctc_conf:
            del args.ctc_conf['ctc_lo_fn']
        # 7. CTC
        logging.info(f'args.ctc_conf: {args.ctc_conf}')
        ctc = CTC(
            odim=vocab_size, encoder_output_size=encoder_output_size, ctc_lo_fn=ctc_lo_fn, **args.ctc_conf
        )

        # 8. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            contextualizer=contextualizer,
            contextualizer_conf=contextualizer_conf,
            ctc=ctc,
            joint_network=joint_network,
            token_list=token_list,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 9. Initialize
        if args.init is not None:
            initialize(model, args.init)
        
        # ?. contextual processor methods
        if getattr(args, "contextual_conf", {}) != {}:
            cls.contextual_processor = cls.build_contextual_processor(
                args,
                model
            )
            model.context_sampler = cls.contextual_processor
        else:
            cls.contextual_processor = None

        assert check_return_type(model)
        return model
