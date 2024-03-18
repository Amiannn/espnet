import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
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
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.contextualizer.func.contextual_adapter_func   import forward_contextual_adapter
from espnet2.asr.contextualizer.func.contextualization_choices import (
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


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
        contextualizer_conf: dict,
        aux_ctc: dict = None,
        aux_ctc_ga: bool = False,
        ctc_ga_weight: float = 0.5,
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
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
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
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
        )
        self.contextualizer      = contextualizer
        self.contextualizer_conf = contextualizer_conf
        self.epoch               = 0

        if self.contextualizer is not None:
            self.use_contextual_methods = True
        else:
            self.use_contextual_methods = False
        
        # guild attention ctc loss
        self.aux_ctc_ga    = aux_ctc_ga
        self.ctc_ga_weight = ctc_ga_weight
        if self.aux_ctc_ga:
            self.aux_ctc_ga_loss = torch.nn.CTCLoss(
                reduction="mean", 
                zero_infinity=True, 
            )

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
        loss_ga_ctc = None
        stats = dict()

        # c1. Encoder contextualization
        enc_bias_vec = None
        if self.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_ENCODER:
            # logging.info(f'Encoder contextualize!')
            enc_bias_vec, enc_attn = forward_contextual_adapter(
                decoder=self.decoder,
                contextualizer=self.contextualizer,
                model_embed=encoder_out,
                context_idxs=contexts['blist'],
                context_xphone_idxs=contexts['blist_xphone'] if 'blist_xphone' in contexts else None,
                ilens=contexts['ilens'],
                return_atten=True
            )
            if not self.use_transducer_decoder:
                encoder_out = encoder_out + enc_bias_vec

            # use guilded attention ctc loss
            if self.aux_ctc_ga:
                enc_attn  = torch.mean(enc_attn, dim=1)
                ga_input  = torch.log(enc_attn).transpose(0, 1)
                ga_target = contexts['label']
                ga_input_lengths  = encoder_out_lens
                ga_target_lengths = contexts['label_ilens']
                
                loss_ga_ctc = self.aux_ctc_ga_loss(
                    ga_input, ga_target, ga_input_lengths, ga_target_lengths
                )
                # Collect CTC branch stats
                stats["loss_ga_ctc"] = loss_ga_ctc.detach() if loss_ga_ctc is not None else None

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
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

            if loss_ctc is not None and loss_ga_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc) + (self.ctc_ga_weight * loss_ga_ctc)
            elif loss_ga_ctc is not None:
                loss = loss_transducer + (self.ctc_ga_weight * loss_ga_ctc)
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
                    encoder_out, encoder_out_lens, text, text_lengths, contexts
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                if loss_ga_ctc is not None:
                    loss = self.ctc_ga_weight * loss_ga_ctc + (1 - self.ctc_ga_weight) * loss_att
                else:
                    loss = loss_att
            elif self.ctc_weight == 1.0:
                if loss_ga_ctc is not None:
                    loss = self.ctc_ga_weight * loss_ga_ctc + (1 - self.ctc_ga_weight) * loss_ctc
                else:
                    loss = loss_ctc
            else:
                if loss_ga_ctc is not None:
                    loss = self.ctc_weight * loss_ctc + self.ctc_ga_weight * loss_ga_ctc + (1 - self.ctc_weight - self.ctc_ga_weight) * loss_att
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
        contexts: dict,
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

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        outputs = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, return_hs=True
        )
        decoder_hs = outputs[0][1]
        
        # c1. Decoder contextualization
        if self.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_DECODER:
            logging.info(f'Decoder contextualize!')
            dec_bias_vec, dec_attn = forward_contextual_adapter(
                decoder=self.decoder,
                contextualizer=self.contextualizer,
                model_embed=decoder_hs,
                context_idxs=contexts['blist'],
                context_xphone_idxs=contexts['blist_xphone'] if 'blist_xphone' in contexts else None,
                ilens=contexts['ilens'],
                return_atten=True
            )
            decoder_hs = decoder_hs + dec_bias_vec

        decoder_out = self.decoder.output_layer(decoder_hs)

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
        # logging.info(f'bias_vec: {bias_vec.shape}')
        # logging.info(f'encoder_out: {encoder_out.unsqueeze(2).shape}')
        # logging.info(f'decoder_out: {decoder_out.unsqueeze(1).shape}')

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