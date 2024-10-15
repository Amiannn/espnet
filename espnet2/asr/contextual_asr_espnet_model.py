import os
import torch
import logging
import torchaudio

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sop_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.contextualizer.func.contextual_adapter_func import forward_contextual_adapter
from espnet2.asr.contextualizer import (
    CONTEXTUAL_RETRIEVER,
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_HISTORY_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER
)

from espnet2.asr.contextualizer.func.contextual_retriever_func import (
    decode_ctc_predictions, 
    decode_topk_tokens,
    generate_prompt_from_hypotheses,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

try:
    from warprnnt_pytorch import RNNTLoss
except:
    logging.info(f'Warning: Cannot import warprnnt_pytorch!')

try:
    import optimized_transducer
except:
    logging.info(f'Warning: Cannot import optimized_transducer!')

class ESPnetContextualASRModel(ESPnetASRModel):
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        contextualizer: Optional[torch.nn.Module],
        contextualizer_conf: dict = {},
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        sym_sop: str = "<|startofprev|>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        context_sampler: object=None,
        **kwargs
    ):
        assert check_argument_types()
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            transducer_multi_blank_durations=transducer_multi_blank_durations,
            transducer_multi_blank_sigma=transducer_multi_blank_sigma,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
            **kwargs
        )
        self.epoch               = 1
        self.contextualizer      = contextualizer
        self.contextualizer_conf = contextualizer_conf
        self.warmup_epoch        = contextualizer_conf['warmup_epoch'] if 'warmup_epoch' in contextualizer_conf else 0

        if sym_sop in token_list:
            self.sop = token_list.index(sym_sop)
        else:
            self.sop = None

        if 'contextualizer_type' not in self.contextualizer_conf:
            self.contextualizer_conf['contextualizer_type'] = None

        # contextualizer loss
        self.contextualizer_weight = self.contextualizer_conf[
            'contextualizer_weight'
        ] if 'contextualizer_weight' in self.contextualizer_conf else 0.0
        self.contextualizer_losses = self.contextualizer_conf[
            'contextualizer_losses'
        ] if 'contextualizer_losses' in self.contextualizer_conf else {}
        
        if 'loss_contextualizer_ga_ctc' in self.contextualizer_losses:
            self.contextualizer_ctc_ga_loss = torch.nn.CTCLoss(
                reduction="mean", 
                zero_infinity=True, 
            )
        if 'loss_contextualizer_ga_rnnt' in self.contextualizer_losses:
            self.contextualizer_rnnt_ga_loss = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )
        if 'loss_contextualizer_ga_ctc_lp' in self.contextualizer_losses:
            self.contextualizer_ctc_ga_loss = torch.nn.CTCLoss(
                reduction="mean", 
                zero_infinity=True, 
            )
        # class balance cross-entropy
        if 'loss_contextualizer_ga_reweight_lp' in self.contextualizer_losses:
            self.lp_gamma = self.contextualizer_conf['lp_gamma'] if 'lp_gamma' in contextualizer_conf else 0.99
            self.loss_amp = 10

        self.context_sampler = context_sampler

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        contexts: object,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        utt_id = kwargs['utt_id']
        # logging.info(f'utt_id: {utt_id}')

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        loss_contextualizer = None
        stats = dict()

        # c1. Encoder contextualization
        enc_bias_vec     = None
        gate_prob        = None
        contexts_hyp     = None
        context_logit    = None
        encoder_out_proj = None

        # c1.1 Contextual Retriever
        if self.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_RETRIEVER:
            contexts_hyp, encoder_out_proj = self.contextualizer(
                query=encoder_out,
                query_ilens=encoder_out_lens,
                context_subword=contexts['blist'],
                context_subword_ilens=contexts['ilens'],
                context_phone=contexts['blist_xphone'],
                context_phone_ilens=contexts['blist_xphone_ilens'],
                return_model_proj=True
            )
        # c1.2 Contextual Adapter
        if self.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_ENCODER:
            enc_bias_vec, enc_attn = forward_contextual_adapter(
                contextualizer=self.contextualizer,
                model_embed=encoder_out,
                context_idxs=contexts['blist'],
                context_xphone_idxs=contexts['blist_xphone_mean'],
                ilens=contexts['ilens'],
                return_atten=True
            )
            # mean across attention heads
            contexts_hyp = torch.mean(enc_attn, dim=1)
            if (self.epoch >= self.warmup_epoch) and (not self.use_transducer_decoder):
                if encoder_out.shape[1] == enc_bias_vec.shape[1]:
                    encoder_out = encoder_out + enc_bias_vec
                else:
                    logging.info(f'Warning: the shape of enc_out: {encoder_out.shape} is different from bias_vec: {enc_bias_vec.shape} !')
            stats["contextualizer_warmup"] = (self.epoch >= self.warmup_epoch)
        
        # c1.3 Contextual History Adapter
        if self.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_HISTORY_ADAPTER_ENCODER:
            decoder_in, target, t_len, u_len = get_transducer_task_io(
                labels=contexts['context_label'],
                encoder_out_lens=encoder_out_lens,
                ignore_id=self.ignore_id,
                blank_id=self.blank_id,
            )
            
            enc_bias_vec, context_logit, contexts_hyp = self.contextualizer(
                model_embed=encoder_out,
                context_idxs=contexts['blist'],
                context_ilens=contexts['ilens'],
                context_xphone_idxs=contexts['blist_xphone_mean'],
                context_history_idx=decoder_in,
                context_history_ilens=(u_len + 1),
            )

            if (self.epoch >= self.warmup_epoch) and (not self.use_transducer_decoder):
                if encoder_out.shape[1] == enc_bias_vec.shape[1]:
                    encoder_out = encoder_out + enc_bias_vec
                else:
                    logging.info(f'Warning: the shape of enc_out: {encoder_out.shape} is different from bias_vec: {enc_bias_vec.shape} !')
            stats["contextualizer_warmup"] = (self.epoch >= self.warmup_epoch)

        # c1.4 Contextualizer Loss
        if len(self.contextualizer_losses) > 0:
            (
                loss_contextualizer, 
                losses_contextualizers
            ) = self._calc_contextualizer_loss(
                contexts,
                contexts_hyp,
                context_logit,
                encoder_out,
                encoder_out_lens,
                gate_prob,
            )
            for loss_name in losses_contextualizers:
                stats[loss_name] = losses_contextualizers[loss_name].detach()

            # Collect total adapter aux losses
            stats["loss_contextualizer"] = (
                loss_contextualizer.detach() if loss_contextualizer is not None else None
            )
                
        # 1. CTC branch
        if self.ctc_weight != 0.0:
            if encoder_out_proj is None:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
            else:
                logging.info(f'go encoder out proj route.')
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out_proj, encoder_out_lens, text, text_lengths
                )
            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
                contexts,
                enc_bias_vec
            )

            if loss_ctc is not None and loss_contextualizer is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc) + (self.contextualizer_weight * loss_contextualizer)
            elif loss_contextualizer is not None:
                loss = loss_transducer + (self.contextualizer_weight * loss_contextualizer)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths, contexts, contexts_hyp
                )
            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                if loss_contextualizer is not None:
                    loss = self.contextualizer_weight * loss_contextualizer + (1 - self.contextualizer_weight) * loss_att
                else:
                    loss = loss_att
            elif self.ctc_weight == 1.0:
                if loss_contextualizer is not None:
                    loss = self.contextualizer_weight * loss_contextualizer + (1 - self.contextualizer_weight) * loss_ctc
                else:
                    loss = loss_ctc
            else:
                if loss_contextualizer is not None:
                    loss = self.ctc_weight * loss_ctc + self.contextualizer_weight * loss_contextualizer + (1 - self.ctc_weight - self.contextualizer_weight) * loss_att
                else:
                    loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
                    
            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        contexts: object,
        contexts_hyp: object,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        # Add text prompt (Whisper style only)
        if "nlp_prompt_tensor" in contexts:
            # TODO: convert retriever's outputs into prompts 
            # for now, we use teacher forcing prefix tuning method
            prompts     = contexts['nlp_prompt_tensor']
            prompt_lens = torch.tensor([p.shape[0] for p in prompts]).to(ys_pad.device)
            prompts_nlp = "\n".join(contexts['nlp_prompt'])
            logging.info(f'\n{"_" * 30}\n{prompts_nlp}')

            if contexts_hyp is not None:
                nlp_prompt, nlp_prompt_tensor = generate_prompt_from_hypotheses(
                    context_hypotheses=contexts_hyp, 
                    contexts=contexts, 
                    construct_prompt_labels_fn=self.context_sampler.construct_prompt_labels,
                    blank_index=0,
                    top_k=10,
                    threshold=0.5,
                )
                prompts_nlp = "\n".join(nlp_prompt)
                logging.info(f'\n{"+" * 30}\n{prompts_nlp}')
                
                prompts     = [prompt.to(ys_pad) for prompt in nlp_prompt_tensor]
                prompt_lens = torch.tensor([p.shape[0] for p in prompts]).to(ys_pad.device)

            ys_in_pad, ys_out_pad = add_sop_sos_eos(ys_pad, prompts, self.sop, self.sos, self.eos, self.ignore_id)
            ys_in_lens = ys_pad_lens + prompt_lens + 2

            # for ys_in, ys_out in zip(ys_in_pad, ys_out_pad):
            #     ys_in  = ys_in.tolist()
            #     ys_out = ys_out.tolist()
            #     logging.info(f'ys_in: {ys_in}')
            #     logging.info(f'ys_out: {ys_out}')
            #     ys_in = self.context_sampler.prompt_token_id_converter.ids2tokens(
            #         ys_in, 
            #         skip_special_tokens=False
            #     )
            #     ys_in = self.context_sampler.prompt_tokenizer.tokens2text(ys_in)
            #     logging.info(f'ys_in text: {ys_in}')
            #     ys_out = self.context_sampler.prompt_token_id_converter.ids2tokens(
            #         [y if y != -1 else 0 for y in ys_out], 
            #         skip_special_tokens=False
            #     )
            #     ys_out = self.context_sampler.prompt_tokenizer.tokens2text(ys_out)
            #     logging.info(f'ys_out text: {ys_out}')
        else:
            ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_in_lens = ys_pad_lens + 1
        
        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
        contexts: dict,
        enc_bias_vec: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        # c1. Decoder contextualization
        dec_bias_vec = None
        if self.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_DECODER:
            logging.info(f'Decoder contextualize!')
            dec_bias_vec, dec_attn = forward_contextual_adapter(
                decoder=self.decoder,
                contextualizer=self.contextualizer,
                model_embed=decoder_out,
                context_idxs=contexts['blist'],
                context_xphone_idxs=contexts['blist_xphone'] if 'blist_xphone' in contexts else None,
                ilens=contexts['ilens'],
                return_atten=True
            )

        bias_vec = None
        if enc_bias_vec is not None and dec_bias_vec is not None:
            bias_vec = enc_bias_vec.unsqueeze(2) + dec_bias_vec.unsqueeze(1)
        elif enc_bias_vec is not None:
            bias_vec = enc_bias_vec.unsqueeze(2)
        elif dec_bias_vec is not None:
            bias_vec = dec_bias_vec.unsqueeze(1)
        # logging.info(f'encoder_out: {encoder_out.shape}')
        # logging.info(f'decoder_out: {decoder_out.shape}')
        # logging.info(f'bias_vec: {bias_vec.shape}')
        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), 
            decoder_out.unsqueeze(1),
            bias_out=bias_vec,
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _calc_contextualizer_loss(
        self, 
        contexts,
        contexts_hyp,
        context_logit,
        encoder_out,
        encoder_out_lens,
        gate_value=None,
    ):
        # use guilded attention ctc loss
        losses_contextualizers = {}
        if 'loss_contextualizer_ga_ctc' in self.contextualizer_losses:
            ga_ctc_input  = torch.log(contexts_hyp).transpose(0, 1)
            ga_ctc_target = contexts['label_ctc']
            ga_ctc_input_lengths  = encoder_out_lens
            ga_ctc_target_lengths = contexts['label_ctc_ilens']
            # logging.info(f'ga_ctc_target:\n{ga_ctc_target}')
            # logging.info(f'ga_ctc_target shape:\n{ga_ctc_target.shape}')
            # logging.info(f'ga_ctc_target_lengths:\n{ga_ctc_target_lengths}')
            # logging.info(f'ga_ctc_target_lengths shape:\n{ga_ctc_target_lengths.shape}')
            loss_ga_ctc = self.contextualizer_ctc_ga_loss(
                ga_ctc_input, 
                ga_ctc_target, 
                ga_ctc_input_lengths, 
                ga_ctc_target_lengths
            )
            losses_contextualizers['loss_contextualizer_ga_ctc'] = loss_ga_ctc
        # monotonic rnnt loss
        if 'loss_contextualizer_ga_monornnt' in self.contextualizer_losses:
            # TODO: fix mono loss problem and format
            logging.info(f'contexts_hyp: {contexts_hyp.shape}')
            ga_mrnnt_input  = torch.log(contexts_hyp).transpose(0, 1)
            ga_mrnnt_target = contexts['label_ctc'].to(torch.long)
            ga_mrnnt_input_lengths  = encoder_out_lens.to(torch.long)
            ga_mrnnt_target_lengths = contexts['label_ctc_ilens'].to(torch.long)
            logging.info(f'ga_mrnnt_target:\n{ga_mrnnt_target}')
            logging.info(f'ga_mrnnt_target shape:\n{ga_mrnnt_target.shape}')
            logging.info(f'ga_mrnnt_target_lengths:\n{ga_mrnnt_target_lengths}')
            logging.info(f'ga_mrnnt_target_lengths shape:\n{ga_mrnnt_target_lengths.shape}')
            loss_ga_monornnt = optimized_transducer.transducer_loss(
                logits=ga_mrnnt_input,
                targets=ga_mrnnt_target,
                logit_lengths=ga_mrnnt_input_lengths,
                target_lengths=ga_mrnnt_target_lengths,
                blank=self.blank_id,
                reduction="mean",
            )
            losses_contextualizers['loss_contextualizer_ga_monornnt'] = loss_ga_monornnt
        # rnnt loss
        if 'loss_contextualizer_ga_rnnt' in self.contextualizer_losses:
            _, target, t_len, u_len = get_transducer_task_io(
                labels=contexts['context_label'],
                encoder_out_lens=encoder_out_lens,
                ignore_id=self.ignore_id,
                blank_id=self.blank_id,
            )
            ga_rnnt_input = context_logit.to(torch.float)
            loss_ga_rnnt = self.contextualizer_rnnt_ga_loss(
                ga_rnnt_input,
                target,
                t_len,
                u_len,
            )
            logging.info(f'loss_ga_rnnt: {loss_ga_rnnt}')
            losses_contextualizers['loss_contextualizer_ga_rnnt'] = loss_ga_rnnt
        # ctc with label prior
        if 'loss_contextualizer_ga_ctc_lp' in self.contextualizer_losses:
            alpha     = 0.3
            lp_warmup = 1
            ga_ctc_input  = torch.log(contexts_hyp).transpose(0, 1)
            ga_ctc_target = contexts['label_ctc']
            ga_ctc_input_lengths  = encoder_out_lens
            ga_ctc_target_lengths = contexts['label_ctc_ilens']
            
            # warm-up (1 epoch)
            if self.epoch > lp_warmup:
                label_prior     = torch.mean(contexts_hyp, dim=1)
                label_prior_log = alpha * torch.log(label_prior)
                ga_ctc_input    = ga_ctc_input - label_prior_log

            loss_ga_ctc = self.contextualizer_ctc_ga_loss(
                ga_ctc_input, 
                ga_ctc_target, 
                ga_ctc_input_lengths, 
                ga_ctc_target_lengths
            )
            losses_contextualizers['loss_contextualizer_ga_ctc_lp'] = loss_ga_ctc
        # ctc with label prior
        if 'loss_contextualizer_ga_reweight_lp' in self.contextualizer_losses:
            label_prior     = torch.mean(contexts_hyp, dim=1)
            label_prior_log = torch.log(label_prior)

            label                  = contexts['label_ctc']
            label_occurrence       = contexts['label_occurrence']
            label_occurrence_ilens = contexts['label_occurrence_ilens']

            # add oov
            B, U     = label.shape
            idx      = (torch.arange(B).repeat(U + 1, 1).T).reshape(-1).to(label.device)
            label    = torch.cat([torch.zeros(B, 1).to(label.device), label], dim=-1).long()
            pred_log = label_prior_log[idx, label.reshape(-1)].reshape(B, U + 1)
            
            label_mask     = (label_occurrence == -1)
            weighted_label = (1 - self.lp_gamma) / (1 - torch.pow(self.lp_gamma, label_occurrence))
            weighted_label[label_mask] = 0.0
            loss_ga_rewieght = -1 * self.loss_amp * ((weighted_label * pred_log).sum(dim=-1)).mean()
            losses_contextualizers['loss_contextualizer_ga_reweight_lp'] = loss_ga_rewieght
        # combine the adapter aux loss
        loss_contextualizer = 0.0
        assert sum(list(self.contextualizer_losses.values())) == 1.0
        for loss_name in self.contextualizer_losses:
            loss_weight = self.contextualizer_losses[loss_name]
            loss_contextualizer += loss_weight * losses_contextualizers[loss_name]
        return loss_contextualizer, losses_contextualizers

if __name__ == '__main__':
    # test kl ga loss
    B, C, C_hat = 2, 10, 4

    C_pad   = C + 1
    x       = torch.zeros(B, C_pad)
    print(f'x: {x}')
    x      = x.view(-1)
    source = torch.ones(B * C_hat)
    index  = torch.randint(-1, C, (B, C_hat))
    step   = torch.arange(B).view(B, 1) * C_pad
    print(f'index: {index}')
    index[(index == -1)] = C
    index  = (index + step).view(-1)

    x.index_add_(0, index, source)
    x = x.view(B, C_pad)[:, :-1]
    x[:, 0] = 1

    print(f'x: {x}')
    x = torch.softmax(x, dim=1)
    print(f'x: {x}')
