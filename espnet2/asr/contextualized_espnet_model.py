"""
ESPnetContextualASRModel: Contextual Adaptation for Enhanced ASR Performance

This custom ASR model builds upon ESPnet's ASR architecture with added contextual biasing. It integrates contextual retrievers, adapters, and prompt generation to improve the recognition of rare and domain-specific terms in speech.

### Key Modifications:
1. **Contextual Adaptation:**
   - Introduces retrievers and adapters to bias recognition based on relevant subword and phoneme-level contexts.
   - Supports multiple contextualizer types (retrievers, encoder adapters, and decoder adapters).

2. **Loss Functions for Contextualization:**
   - Custom losses like contextual CTC, RNN-T, and reweighted label prior losses.
   - Dynamically adjusts contextualization losses through warm-up mechanisms and loss weighting.

3. **Contextual Prompts Handling:**
   - Uses retrieved context hypotheses to generate NLP-based prompts for improved decoding.
   - Updates contexts dynamically during decoding to reflect model predictions.

4. **Advanced Decoding and Loss Management:**
   - Combines contextualization loss with standard CTC and attention-based loss functions.
   - Applies contextualization at both encoder and decoder levels, with bias vectors influencing final predictions.

5. **Transducer Model Integration:**
   - Enhanced support for transducer models with contextual bias applied to joint networks.
   - Seamlessly combines bias vectors from encoder and decoder for optimized predictions.

6. **Prompt and Tokenization Support:**
   - Handles Whisper-style text prompts and manages auxiliary tasks for token handling.
   - Includes NLP prompt integration to steer predictions.

This model extends ESPnet to support contextual ASR, making it ideal for applications requiring high accuracy in recognizing domain-specific or rare vocabulary.
"""


import torch
import logging

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

from packaging.version import parse as V
from typeguard import check_argument_types

from torch.nn.utils.rnn import pad_sequence

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
from espnet.nets.pytorch_backend.transformer.add_sos_eos import (
    add_sos_eos,
    add_sop_sos_eos,
)
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
    LableSmoothingReWeightedLoss,
    LableSmoothingUtterLevelReWeightedLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.contextualizer.func.contextual_adapter_func import forward_contextual_adapter
from espnet2.asr.contextualizer import (
    CONTEXTUAL_RETRIEVER,
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_HISTORY_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER,
    CONTEXTUAL_PROTOTYPE,
)
from espnet2.asr.contextualizer.func.contextual_retriever_func import (
    decode_topk_tokens,
    generate_prompt_from_hypotheses,
)
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

try:
    from warprnnt_pytorch import RNNTLoss
except ImportError:
    logging.info("Warning: Cannot import warprnnt_pytorch!")

try:
    import optimized_transducer
except ImportError:
    logging.info("Warning: Cannot import optimized_transducer!")


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
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        sym_sop: str = "<|startofprev|>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        context_sampler: object = None,
        lsm_reweight_type: str = None,
        **kwargs,
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
            **kwargs,
        )

        self.epoch = 1
        self.contextualizer = contextualizer
        self.contextualizer_conf = contextualizer_conf
        self.warmup_epoch = self.contextualizer_conf.get("warmup_epoch", 0)

        self.sop = token_list.index(sym_sop) if sym_sop in token_list else None

        self.contextualizer_conf.setdefault("contextualizer_type", None)

        # Contextualizer loss configurations
        self.contextualizer_loss_weight = self.contextualizer_conf.get("contextualizer_weight", 0.0)
        self.contextualizer_losses      = self.contextualizer_conf.get("contextualizer_losses", {})

        # Initialize loss functions based on specified contextualizer losses
        if "loss_contextualizer_ga_ctc" in self.contextualizer_losses:
            self.contextualizer_ctc_ga_loss = torch.nn.CTCLoss(
                reduction="mean", zero_infinity=True
            )
        if "loss_contextualizer_ga_rnnt" in self.contextualizer_losses:
            self.contextualizer_rnnt_ga_loss = RNNTLoss(
                blank=self.blank_id, fastemit_lambda=0.0
            )
        if "loss_contextualizer_ga_reweight_lp" in self.contextualizer_losses:
            self.lp_gamma = self.contextualizer_conf.get("lp_gamma", 0.99)
            self.loss_amp = 10
        if "loss_contextualizer_ga_ce" in self.contextualizer_losses:
            self.contextualizer_ga_ce = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_id)
        if "loss_gate_ce" in self.contextualizer_losses:
            self.contextualizer_gate_ce = torch.nn.BCEWithLogitsLoss()
        self.context_sampler = context_sampler

        if not self.use_transducer_decoder:
            self.lsm_reweight_type = lsm_reweight_type
            if lsm_reweight_type is None:
                self.criterion_att = LabelSmoothingLoss(
                    size=vocab_size,
                    padding_idx=ignore_id,
                    smoothing=lsm_weight,
                    normalize_length=length_normalized_loss,
                )
            elif lsm_reweight_type == 'iw':
                self.criterion_att = LableSmoothingUtterLevelReWeightedLoss(
                    size=vocab_size,
                    padding_idx=ignore_id,
                    smoothing=lsm_weight,
                    normalize_length=length_normalized_loss,
                )
            elif lsm_reweight_type == 'ln':
                lp_gamma = self.contextualizer_conf.get("lp_gamma", 0.99)
                self.criterion_att = LableSmoothingReWeightedLoss(
                    size=vocab_size,
                    padding_idx=ignore_id,
                    smoothing=lsm_weight,
                    normalize_length=length_normalized_loss,
                    alpha=lp_gamma,
                )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        contexts: dict,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calculate loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            text: (Batch, Length)
            text_lengths: (Batch,)
            contexts: Contextual information
            kwargs: Additional arguments (e.g., "utt_id")
        """
        utt_id = kwargs.get("utt_id")
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id
        text = text[:, : text_lengths.max()]  # For data-parallel

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            encoder_out, intermediate_outs = encoder_out

        stats = dict()
        loss = None

        main_loss, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None

        # Apply contextualizer to the encoder output
        (
            encoder_out,
            encoder_bias_vector,
            contexts_hypotheses_encoder,
            context_logit_encoder,
            encoder_out_proj,
        ) = self._apply_contextualizer_encoder(encoder_out, encoder_out_lens, contexts, stats)
        
        # Here we can pass the retrieved contexts to the decoder
        if contexts_hypotheses_encoder is not None:
            self._update_contexts(contexts, contexts_hypotheses_encoder)

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out_proj if encoder_out_proj is not None else encoder_out,
                encoder_out_lens,
                text,
                text_lengths,
            )
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            loss_interctc = self._calc_intermediate_ctc_loss(
                intermediate_outs, encoder_out_lens, text, text_lengths, stats, kwargs
            )
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        # 2. Decoder
        if self.use_transducer_decoder:
            (
                main_loss,
                cer,
                wer,
                contexts_hypotheses_decoder,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
                contexts,
                encoder_bias_vector,
            )
            stats.update(
                {
                    "loss_transducer": main_loss.detach()
                    if main_loss is not None
                    else None,
                    "cer_transducer": cer,
                    "wer_transducer": wer,
                }
            )
        elif self.ctc_weight != 1.0:
            (
                main_loss,
                acc_att,
                cer_att,
                wer_att,
                decoder_out_length,
                contexts_hypotheses_decoder,
                gate_hypotheses_decoder,
            ) = self._calc_att_loss(
                encoder_out,
                encoder_out_lens,
                text,
                text_lengths,
                contexts,
                # contexts_hypotheses_encoder,
            )
            stats.update(
                {
                    "loss_att": main_loss.detach() if main_loss is not None else None,
                    "acc": acc_att,
                    "cer": cer_att,
                    "wer": wer_att,
                }
            )

        # 3. Contextualizer Loss
        loss_contextualizer = 0.0
        if len(self.contextualizer_losses) > 0:
            for hypotheses, gate_hypotheses, output_lengths, suffix in [
                (contexts_hypotheses_encoder, None, encoder_out_lens, "encoder"),
                (contexts_hypotheses_decoder, gate_hypotheses_decoder, decoder_out_length, "decoder"),
            ]:
                if hypotheses is not None:
                    loss, individual_losses = self._calc_contextualizer_loss(
                        contexts,
                        hypotheses,
                        context_logit_encoder,
                        gate_hypotheses,
                        output_lengths,
                        loss_suffix=suffix,
                    )
                    loss_contextualizer = loss_contextualizer + loss
                    stats.update(individual_losses)
            stats["loss_contextualizer"] = loss_contextualizer.detach()
        stats["contextualizer_warmup"] = self.epoch < self.warmup_epoch

        # Combine losses
        loss = self._combine_losses(
            main_loss, loss_ctc, loss_contextualizer, self.ctc_weight
        )

        stats["loss"] = loss.detach()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _apply_contextualizer_encoder(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        contexts: dict,
        stats: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply contextualizer to the encoder output."""
        encoder_bias_vector = None
        contexts_hypotheses = None
        context_logit = None
        encoder_out_proj = None

        contextualizer_type = self.contextualizer_conf["contextualizer_type"]

        if contextualizer_type in CONTEXTUAL_RETRIEVER:
            contexts_hypotheses, encoder_out_proj = self.contextualizer(
                query=encoder_out,
                query_ilens=encoder_out_lens,
                context_subword=contexts["blist"],
                context_subword_ilens=contexts["ilens"],
                context_phone=contexts["blist_xphone"],
                context_phone_ilens=contexts["blist_xphone_ilens"],
                return_model_proj=True,
            )
        elif contextualizer_type in CONTEXTUAL_ADAPTER_ENCODER:
            encoder_bias_vector, encoder_attention = self.contextualizer(
                model_embed=encoder_out,
                context_embed=contexts["blist"],
                context_xphone_idxs=contexts["blist_xphone_mean"],
                ilens=contexts["ilens"],
                return_atten=True,
            )
            contexts_hypotheses = torch.mean(encoder_attention, dim=1)
            if self.epoch >= self.warmup_epoch and not self.use_transducer_decoder:
                if encoder_out.shape[1] == encoder_bias_vector.shape[1]:
                    encoder_out = encoder_out + encoder_bias_vector
                else:
                    logging.info(
                        f"Warning: Shape mismatch between encoder_out {encoder_out.shape} and encoder_bias_vector {encoder_bias_vector.shape}!"
                    )
        elif contextualizer_type in CONTEXTUAL_HISTORY_ADAPTER_ENCODER:
            decoder_in, target, t_len, u_len = get_transducer_task_io(
                labels=contexts["context_label"],
                encoder_out_lens=encoder_out_lens,
                ignore_id=self.ignore_id,
                blank_id=self.blank_id,
            )
            (
                encoder_bias_vector,
                context_logit,
                contexts_hypotheses,
            ) = self.contextualizer(
                model_embed=encoder_out,
                context_idxs=contexts["blist"],
                context_ilens=contexts["ilens"],
                context_xphone_idxs=contexts["blist_xphone_mean"],
                context_history_idx=decoder_in,
                context_history_ilens=(u_len + 1),
            )
            if self.epoch >= self.warmup_epoch and not self.use_transducer_decoder:
                if encoder_out.shape[1] == encoder_bias_vector.shape[1]:
                    encoder_out = encoder_out + encoder_bias_vector
                else:
                    logging.info(
                        f"Warning: Shape mismatch between encoder_out {encoder_out.shape} and encoder_bias_vector {encoder_bias_vector.shape}!"
                    )
            stats["contextualizer_warmup"] = self.epoch >= self.warmup_epoch
        elif contextualizer_type in CONTEXTUAL_PROTOTYPE:
            contexts_hypotheses, encoder_out_proj = self.contextualizer.forward_at_encode(
                query=encoder_out,
                query_ilens=encoder_out_lens,
                context_subword=contexts["blist"],
                context_subword_ilens=contexts["ilens"],
                context_phone=contexts["blist_xphone"],
                context_phone_ilens=contexts["blist_xphone_ilens"],
                return_model_proj=True,
            )

        return encoder_out, encoder_bias_vector, contexts_hypotheses, context_logit, encoder_out_proj

    def _apply_contextualizer_decoder(
        self,
        decoder_embed: torch.Tensor,
        contexts: dict,
    ):
        """Apply contextualizer to the decoder output."""
        decoder_bias_vector = None
        contexts_hypotheses = None
        contextualizer_type = self.contextualizer_conf["contextualizer_type"]

        gate_value = None
        if contextualizer_type in CONTEXTUAL_ADAPTER_DECODER:
            decoder_bias_vector, decoder_attention = self.contextualizer(
                model_embed=decoder_embed,
                context_embed=contexts["blist"],
                ilens=contexts["ilens"],
                return_atten=True,
            )
            contexts_hypotheses = torch.mean(decoder_attention, dim=1)

            if decoder_bias_vector is not None and (self.epoch >= self.warmup_epoch):
                if hasattr(self.contextualizer, 'gate_layer'):
                    decoder_embed, gate_value = self.contextualizer.gate_layer(decoder_embed, decoder_bias_vector)
                    logging.info(f'gate_value: {torch.sum(gate_value)}')
                else:
                    decoder_embed = decoder_embed + decoder_bias_vector

        elif contextualizer_type in CONTEXTUAL_PROTOTYPE:
            blist_utterance_wise = contexts["blist_utterance_wise"]
            ilens_utterance_wise = contexts["ilens_utterance_wise"]
            decoder_bias_vectors = []
            decoder_attentions   = []
            for i in range(len(blist_utterance_wise)):
                decoder_bias_vector, decoder_attention = self.contextualizer.forward_at_decode(
                    model_embed=decoder_embed[i].unsqueeze(0),
                    context_embed=blist_utterance_wise[i],
                    ilens=ilens_utterance_wise[i],
                    return_atten=True,
                )
                decoder_attention = decoder_attention.squeeze(0).transpose(-1, 0)
                decoder_bias_vectors.append(decoder_bias_vector)
                decoder_attentions.append(decoder_attention)
            decoder_bias_vector = torch.stack(decoder_bias_vectors, dim=0)
            decoder_attentions = pad_sequence(
                decoder_attentions, 
                batch_first=True, 
                padding_value=0
            ).transpose(-1, 1)
            contexts_hypotheses = torch.mean(decoder_attentions, dim=1)
        decoder_out = self.decoder.output_layer(decoder_embed)
        return decoder_out, contexts_hypotheses, gate_value

    def _update_contexts(self, contexts, contexts_hypotheses):
        # Update the context prompt
        if contexts["nlp_prompt_tensor"] is not None:
            generated_prompts, generated_prompt_tensors = generate_prompt_from_hypotheses(
                contexts=contexts,
                context_hypotheses=contexts_hypotheses,
                construct_prompt_labels_fn=self.context_sampler.construct_prompt_labels,
                top_k=self.context_sampler.max_utterance_disrupt_context,
                blank_index=0,
                threshold=0.5,
            )
            prompts = [prompt.to(contexts_hypotheses.device) for prompt in generated_prompt_tensors]
            contexts.update(
                {
                    "nlp_prompt": generated_prompts,
                    "nlp_prompt_tensor": prompts,
                }
            )

    def _calc_intermediate_ctc_loss(
        self,
        intermediate_outs,
        encoder_out_lens,
        text,
        text_lengths,
        stats,
        kwargs,
    ):
        """Calculate intermediate CTC loss."""
        loss_interctc = 0.0
        for layer_idx, intermediate_out in intermediate_outs:
            # Use auxiliary CTC data if specified
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
                            "Auxiliary CTC tasks were specified but no data was found"
                        )
            if loss_ic is None:
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
            loss_interctc += loss_ic
            stats[f"loss_interctc_layer{layer_idx}"] = loss_ic.detach()
            stats[f"cer_interctc_layer{layer_idx}"] = cer_ic
        return loss_interctc / len(intermediate_outs)

    def _combine_losses(
        self,
        main_loss: torch.Tensor,
        ctc_loss: Optional[torch.Tensor],
        contextualizer_loss: Optional[torch.Tensor],
        ctc_weight: float,
    ) -> torch.Tensor:
        """Combine main loss with CTC and contextualizer losses.

        Args:
            main_loss (torch.Tensor): The primary loss (e.g., attention or transducer loss).
            ctc_loss (Optional[torch.Tensor]): The CTC loss component.
            contextualizer_loss (Optional[torch.Tensor]): The contextualizer loss component.
            ctc_weight (float): Weight for the CTC loss.

        Returns:
            torch.Tensor: The combined total loss.
        """
        # Compute the weights for each loss component
        main_loss_weight = 1.0 - ctc_weight
        total_weight = main_loss_weight + ctc_weight

        # Include contextualizer loss weight if applicable
        if contextualizer_loss is not None:
            total_weight += self.contextualizer_loss_weight  # Renamed for clarity

        # Normalize the weights to sum to 1
        main_loss_weight /= total_weight
        ctc_weight /= total_weight
        if contextualizer_loss is not None:
            contextualizer_weight = self.contextualizer_loss_weight / total_weight
        else:
            contextualizer_weight = 0.0

        # Combine the losses using the normalized weights
        loss = main_loss_weight * main_loss
        if ctc_loss is not None:
            loss += ctc_weight * ctc_loss
        if contextualizer_loss is not None:
            loss += contextualizer_weight * contextualizer_loss

        return loss

    def _calc_att_loss(
        self,
        encoder_output: torch.Tensor,
        encoder_output_lengths: torch.Tensor,
        target_sequences: torch.Tensor,
        target_lengths: torch.Tensor,
        contexts: dict,
    ) -> Tuple[torch.Tensor, float, Optional[float], Optional[float], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Calculate attention loss for the attention-based decoder.

        Args:
            encoder_output (torch.Tensor): Encoder output features.
            encoder_output_lengths (torch.Tensor): Lengths of encoder outputs.
            target_sequences (torch.Tensor): Padded target sequences (batch_size, seq_len).
            target_lengths (torch.Tensor): Lengths of target sequences (batch_size,).
            contexts (dict): Additional context information.
            encoder_context_hypotheses (Optional[torch.Tensor]): Context hypotheses from the encoder.

        Returns:
            loss_att (torch.Tensor): Attention loss value.
            acc_att (float): Attention accuracy.
            cer_att (Optional[float]): Character Error Rate.
            wer_att (Optional[float]): Word Error Rate.
            contextual_stats (Dict[str, torch.Tensor]): Contextual statistics.
            decoder_context_hypotheses (Optional[torch.Tensor]): Context hypotheses from the decoder.
        """
        # 1. Prepend language token if available
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            lang_token = self.lang_token_id.repeat(target_sequences.size(0), 1).to(target_sequences.device)
            target_sequences = torch.cat([lang_token, target_sequences], dim=1)
            target_lengths += 1

        # 2. Handle text prompts (Whisper-style)
        if contexts.get("nlp_prompt_tensor") is not None:
            prompts = contexts["nlp_prompt_tensor"]
            prompt_lengths = torch.tensor([p.shape[0] for p in prompts], device=target_sequences.device)
            prompt_text = "\n".join(contexts.get("nlp_prompt", []))
            logging.info(f'\n{"_" * 30}\n{prompt_text}')
            # Prepare input and output sequences with prompts
            ys_in_pad, ys_out_pad = add_sop_sos_eos(
                target_sequences, prompts, self.sop, self.sos, self.eos, self.ignore_id
            )
            ys_in_lengths = target_lengths + prompt_lengths + 2  # +2 for <sop> and <eos>
        else:
            # Add <sos> and <eos> tokens to the target sequences
            ys_in_pad, ys_out_pad = add_sos_eos(target_sequences, self.sos, self.eos, self.ignore_id)
            ys_in_lengths = target_lengths + 1  # +1 for <eos>

        # 3. Forward pass through the decoder
        if isinstance(self.decoder, OpenAIWhisperDecoder):
            (decoder_output, decoder_hidden_states), _ = self.decoder(
                encoder_output, encoder_output_lengths, ys_in_pad, ys_in_lengths, return_hs=True
            )
            decoder_embeddings = decoder_hidden_states
        else:
            (decoder_output, decoder_hidden_states), _ = self.decoder(
                encoder_output, encoder_output_lengths, ys_in_pad, ys_in_lengths, return_hs=True
            )
            decoder_embeddings = decoder_hidden_states

        # 4. Apply decoder contextualization
        decoder_output, decoder_context_hypotheses, decoder_gate_hypotheses = self._apply_contextualizer_decoder(
            decoder_embeddings, contexts
        )

        # 5. Compute attention loss
        if self.lsm_reweight_type is None:
            loss_att = self.criterion_att(
                decoder_output, 
                ys_out_pad, 
            )
        elif self.lsm_reweight_type == 'iw':
            label_importance_weight       = contexts['label_importance_weight']
            label_importance_weight_ilens = contexts['label_importance_weight_ilens']
            loss_att = self.criterion_att(
                decoder_output, 
                ys_out_pad, 
                label_importance_weight,
                label_importance_weight_ilens
            )
        elif self.lsm_reweight_type == 'ln':
            token_level_label_occurrence = contexts['token_level_label_occurrence']
            logging.info(f'token_level_label_occurrence: {token_level_label_occurrence}')
            loss_att = self.criterion_att(
                decoder_output, 
                ys_out_pad, 
                token_level_label_occurrence
            )
        
        acc_att = th_accuracy(
            decoder_output.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # 6. Compute CER/WER if applicable
        if not self.training and self.error_calculator is not None:
            ys_hat = decoder_output.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), target_sequences.cpu())
        else:
            cer_att, wer_att = None, None

        return (
            loss_att,
            acc_att,
            cer_att,
            wer_att,
            ys_in_lengths,
            decoder_context_hypotheses,
            decoder_gate_hypotheses,
        )

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
        contexts: dict,
        encoder_bias_vector: torch.Tensor,
    ):
        """Compute Transducer loss."""
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        # Apply decoder contextualization
        decoder_bias_vector, contexts_hypotheses_decoder = self._apply_contextualizer_decoder(
            decoder_out, contexts
        )

        # Combine bias vectors
        bias_vector = None
        if encoder_bias_vector is not None and decoder_bias_vector is not None:
            bias_vector = encoder_bias_vector.unsqueeze(2) + decoder_bias_vector.unsqueeze(1)
        elif encoder_bias_vector is not None:
            bias_vector = encoder_bias_vector.unsqueeze(2)
        elif decoder_bias_vector is not None:
            bias_vector = decoder_bias_vector.unsqueeze(1)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            bias_out=bias_vector,
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

        return loss_transducer, cer_transducer, wer_transducer, contexts_hypotheses_decoder

    def _calc_contextualizer_loss(
        self,
        contexts: dict,
        contextual_hypotheses: Optional[torch.Tensor],
        contextual_hypotheses_logits: Optional[torch.Tensor],
        gate_hypotheses: Optional[torch.Tensor],
        contextual_hypotheses_output_lengths: Optional[torch.Tensor],
        loss_suffix: str,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Calculate the contextualizer loss based on specified loss types.

        Args:
            contexts (dict): Dictionary containing context information.
            contextual_hypotheses (Optional[torch.Tensor]): Hypotheses from the contextualizer
                of shape (batch_size, seq_len, num_classes).
            contextual_hypotheses_logits (Optional[torch.Tensor]): Logits from the contextualizer (if applicable),
                of shape (batch_size, seq_len, num_classes).
            contextual_hypotheses_output_lengths (Optional[torch.Tensor]): Length of the hypotheses (if applicable),
                of shape (batch_size).

        Returns:
            total_contextualizer_loss (Optional[torch.Tensor]): Combined loss from all specified
                contextualizer losses.
            individual_losses (Dict[str, torch.Tensor]): Dictionary of individual contextualizer losses.
        """
        # Select hypotheses from encoder or decoder
        if contextual_hypotheses is None:
            logging.warning("No contextual hypotheses provided. Skipping contextualizer loss calculation.")
            return None, {}

        device = contextual_hypotheses.device
        # contexts = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in contexts.items()}
                
        individual_losses = {}
        epsilon = 1e-10  # To prevent log(0)
        log_contextual_hypotheses = torch.log(contextual_hypotheses + epsilon)

        batch_size, seq_len, _ = contextual_hypotheses.shape
        encoder_output_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

        contextualizer_losses        = {name: self.contextualizer_losses[name][0] for name in self.contextualizer_losses}
        contextualizer_losses_suffix = {name: self.contextualizer_losses[name][1] for name in self.contextualizer_losses}
        total_weight = sum(contextualizer_losses.values())
        if total_weight == 0.0:
            logging.warning("Total weight of contextualizer losses is zero. No losses will be calculated.")
            return None, {}
        elif abs(total_weight - 1.0) > 1e-6:
            logging.warning(
                f"The sum of contextualizer loss weights is {total_weight}, not 1.0. Normalizing weights."
            )
            normalized_weights = {k: v / total_weight for k, v in contextualizer_losses.items()}
        else:
            normalized_weights = contextualizer_losses

        # Compute individual losses
        if "loss_contextualizer_ga_ctc" in normalized_weights and loss_suffix == contextualizer_losses_suffix["loss_contextualizer_ga_ctc"]:
            # Ensure required context keys are available
            required_keys = ["label_ctc", "label_ctc_ilens"]
            for key in required_keys:
                if key not in contexts:
                    raise ValueError(f"Missing required context key: '{key}' for CTC loss.")

            # Prepare inputs
            ctc_input = log_contextual_hypotheses.transpose(0, 1)  # Shape: (T, N, C)
            ctc_targets = contexts["label_ctc"]  # Shape: (N, S)
            ctc_input_lengths = contextual_hypotheses_output_lengths  # Shape: (N,)
            ctc_target_lengths = contexts["label_ctc_ilens"]  # Shape: (N,)

            # Compute CTC loss
            loss_ctc = self.contextualizer_ctc_ga_loss(
                ctc_input,
                ctc_targets,
                ctc_input_lengths,
                ctc_target_lengths,
            )
            individual_losses[f"loss_contextualizer_ga_ctc_{loss_suffix}"] = loss_ctc

        if "loss_contextualizer_ga_rnnt" in normalized_weights and loss_suffix == contextualizer_losses_suffix["loss_contextualizer_ga_rnnt"]:
            if contextual_hypotheses_logits is None:
                raise ValueError("contextual_hypotheses_logits is required for 'loss_contextualizer_ga_rnnt'")

            # Prepare inputs
            labels = contexts["context_label"]
            if labels is None:
                raise ValueError("Missing 'context_label' in contexts for RNN-T loss")

            # Infer contextual_hypotheses_output_lengths from hypotheses
            time_lengths = contextual_hypotheses_output_lengths  # Shape: (N,)
            label_lengths = (labels != self.ignore_id).sum(dim=1).long()  # Assuming labels are padded with ignore_id

            rnnt_input = contextual_hypotheses_logits.float()

            # Compute RNN-T loss
            loss_rnnt = self.contextualizer_rnnt_ga_loss(
                rnnt_input,
                labels,
                time_lengths,
                label_lengths,
            )
            individual_losses["loss_contextualizer_ga_rnnt"] = loss_rnnt

        if "loss_contextualizer_ga_reweight_lp" in normalized_weights and loss_suffix == contextualizer_losses_suffix["loss_contextualizer_ga_reweight_lp"]:
            required_keys = ["label_ctc", "label_occurrence", "label_occurrence_ilens"]
            for key in required_keys:
                if key not in contexts:
                    raise ValueError(f"Missing required context key: '{key}' for Reweighted Label Prior loss.")

            label_prior = torch.mean(contextual_hypotheses, dim=1)  # Shape: (N, C)
            label_prior_log = torch.log(label_prior + epsilon)  # Shape: (N, C)

            labels = contexts["label_ctc"]  # Shape: (N, U)
            label_occurrences = contexts["label_occurrence"]  # Shape: (N, U)
            label_occurrence_lengths = contexts["label_occurrence_ilens"]  # Shape: (N,)

            batch_size, seq_length = labels.shape
            labels = torch.cat([torch.zeros(batch_size, 1).to(labels.device), labels], dim=-1).long() # add no-context ids
            seq_length += 1
            
            indices = (
                torch.arange(batch_size, device=labels.device).unsqueeze(1).repeat(1, seq_length).reshape(-1)
            )
            predicted_log_probs = label_prior_log[indices, labels.reshape(-1)].reshape(batch_size, seq_length)

            label_mask = label_occurrences == -1
            weighted_labels = (1 - self.lp_gamma) / (1 - torch.pow(self.lp_gamma, label_occurrences))
            weighted_labels[label_mask] = 0.0
            loss_reweighted_lp = -self.loss_amp * ((weighted_labels * predicted_log_probs).sum(dim=-1)).mean()
            individual_losses[f"loss_contextualizer_ga_reweight_lp_{loss_suffix}"] = loss_reweighted_lp

        if "loss_contextualizer_ga_ce" in normalized_weights and loss_suffix == contextualizer_losses_suffix["loss_contextualizer_ga_ce"]:
            ga_log_probs = log_contextual_hypotheses  # Shape: (batch_size, seq_len, num_classes)
            batch_size, seq_len, num_classes = ga_log_probs.shape
            label_ce = contexts['label_cross_entropy']
            # Flatten inputs and targets
            input_flat  = ga_log_probs[:, :-1, :].reshape(-1, num_classes)
            target_flat = label_ce.view(-1)
            # Compute Cross-Entropy loss
            loss_ce = self.contextualizer_ga_ce(input_flat, target_flat)
            individual_losses[f"loss_contextualizer_ga_ce_{loss_suffix}"] = loss_ce

        if "loss_gate_ce" in normalized_weights and loss_suffix == contextualizer_losses_suffix["loss_gate_ce"] and gate_hypotheses is not None:
            label_ce  = contexts['label_cross_entropy']
            label_bce = (label_ce != 0).float()
            # Create mask for valid positions
            mask = label_ce != -1  # Shape: [batch_size, seq_len]
            # Filter out ignored positions
            valid_gate_probs = (gate_hypotheses[:, :-1, :])[mask].squeeze(-1)      # Shape: [num_valid_positions]
            valid_gate_labels = label_bce[mask]    # Shape: [num_valid_positions]
            # Compute BCE loss
            loss_gate_ce = self.contextualizer_gate_ce(valid_gate_probs, valid_gate_labels)
            individual_losses[f"loss_gate_ce_{loss_suffix}"] = loss_gate_ce
        # Combine the individual losses into a total loss
        total_loss = 0.0
        for loss_name, loss_weight in normalized_weights.items():
            loss_name = f'{loss_name}_{loss_suffix}'
            if loss_name in individual_losses:
                total_loss += loss_weight * individual_losses[loss_name]
            else:
                logging.warning(f"Loss '{loss_name}' is specified but not calculated.")
        return total_loss, individual_losses

