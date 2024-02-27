"""Search algorithms for Transducer models."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.lm.transformer_lm import TransformerLM
from espnet.nets.pytorch_backend.transducer.utils import (
    is_prefix,
    recombine_hyps,
    select_k_expansions,
    subtract,
)

from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.transducer.beam_search_transducer import Hypothesis

@dataclass
class ContextualHypothesis(Hypothesis):
    """Default hypothesis definition for Transducer search algorithms."""
    ...

class ContextualBeamSearchTransducer(BeamSearchTransducer):
    """Beam search implementation for Contextual Transducer."""

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        contextualizer: object,
        contextualizer_conf: Dict,
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        expansion_gamma: int = 2.3,
        expansion_beta: int = 2,
        multi_blank_durations: List[int] = [],
        multi_blank_indices: List[int] = [],
        score_norm: bool = True,
        score_norm_during: bool = False,
        nbest: int = 1,
        token_list: Optional[List[str]] = None,
    ):
        super().__init__(
            decoder=decoder,
            joint_network=joint_network,
            beam_size=beam_size,
            lm=lm,
            lm_weight=lm_weight,
            search_type=search_type,
            max_sym_exp=max_sym_exp,
            u_max=u_max,
            nstep=nstep,
            prefix_alpha=prefix_alpha,
            expansion_gamma=expansion_gamma,
            expansion_beta=expansion_beta,
            multi_blank_durations=multi_blank_durations,
            multi_blank_indices=multi_blank_indices,
            score_norm=score_norm,
            score_norm_during=score_norm_during,
            nbest=nbest,
            token_list=token_list,
        )

        if search_type == "casr" and self.beam_size <= 1:
            # greedy contextual asr
            self.search_algorithm   = self.greedy_search_contextual
        elif search_type == "casr":
            # contextual asr
            self.search_algorithm   = self.beam_search_contextual
        else:
            raise NotImplementedError

        self.contextualizer      = contextualizer
        self.contextualizer_conf = contextualizer_conf

    def __call__(
        self,
        enc_out: torch.Tensor, 
        contexts: object={},
    ) -> Union[List[ContextualHypothesis]]:
        self.decoder.set_device(enc_out.device)
        nbest_hyps = self.search_algorithm(enc_out, contexts)

        return nbest_hyps

    def greedy_search_contextual(
        self, 
        enc_out: torch.Tensor, 
        contexts: object
    ) -> List[ContextualHypothesis]:
        """Greedy search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypotheses.

        """
        # Encoder contextualization
        if self.contextualizer_conf["contextualizer_type"] == "contextual_adapter_encoder":
            logging.info(f'Encoder contextualize!')
            enc_out = enc_out.unsqueeze(0)
            enc_out = self.forward_contextual_adapter_fusion(
                model_embed=enc_out,
                context_idxs=contexts['blist'],
            )
            enc_out = enc_out.squeeze(0)

        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)
        cache = {}

        dec_out, state, _ = self.decoder.score(hyp, cache)

        for enc_out_t in enc_out:
            # Decoder contextualization
            if self.contextualizer_conf["contextualizer_type"] == "contextual_adapter_decoder":
                logging.info(f'Decoder contextualize!')
                dec_out = dec_out.reshape(1, 1, -1)
                dec_out = self.forward_contextual_adapter_fusion(
                    model_embed=dec_out,
                    context_idxs=contexts['blist'],
                )
                dec_out = dec_out.reshape(-1)

            logp = torch.log_softmax(
                self.joint_network(enc_out_t, dec_out),
                dim=-1,
            )
            top_logp, pred = torch.max(logp, dim=-1)

            if pred != self.blank_id:
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)

                hyp.dec_state = state

                dec_out, state, _ = self.decoder.score(hyp, cache)

        return [hyp]

    def beam_search_contextual(
        self, 
        enc_out: torch.Tensor,
        contexts: object
    ) -> List[ContextualHypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        logging.info(f'Contextual ASR beam search!')
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [ContextualHypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)]
        cache = {}
        cache_lm = {}

        # Encoder contextualization
        if self.contextualizer_conf["contextualizer_type"] == "contextual_adapter_encoder":
            logging.info(f'Encoder contextualize!')
            enc_out = enc_out.unsqueeze(0)
            enc_out = self.forward_contextual_adapter_fusion(
                model_embed=enc_out,
                context_idxs=contexts['blist'],
            )
            enc_out = enc_out.squeeze(0)

        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            if self.token_list is not None:
                logging.debug(
                    "\n"
                    + "\n".join(
                        [
                            "hypo: "
                            + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                            + f", score: {round(float(hyp.score), 2)}"
                            for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                        ]
                    )
                )

            while True:
                if self.score_norm_during:
                    max_hyp = max(hyps, key=lambda x: x.score / len(x.yseq))
                else:
                    max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state, lm_tokens = self.decoder.score(max_hyp, cache)
                # Decoder contextualization
                if self.contextualizer_conf["contextualizer_type"] == "contextual_adapter_decoder":
                    logging.info(f'Decoder contextualize!')
                    dec_out = dec_out.reshape(1, 1, -1)
                    dec_out = self.forward_contextual_adapter_fusion(
                        model_embed=dec_out,
                        context_idxs=contexts['blist'],
                    )
                    dec_out = dec_out.reshape(-1)

                logp = torch.log_softmax(
                    self.joint_network(enc_out_t, dec_out),
                    dim=-1,
                )
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    ContextualHypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    if tuple(max_hyp.yseq) not in cache_lm:
                        lm_scores, lm_state = self.lm.score(
                            torch.LongTensor(
                                [self.sos] + max_hyp.yseq[1:],
                                device=self.decoder.device,
                            ),
                            max_hyp.lm_state,
                            None,
                        )
                        cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                    else:
                        lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        ContextualHypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                if self.score_norm_during:
                    hyps_max = float(
                        max(hyps, key=lambda x: x.score / len(x.yseq)).score
                    )
                else:
                    hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    # forward contextualizer
    def forward_contextual_adapter_fusion(
        self,
        model_embed,
        context_idxs,
    ):
        if isinstance(self.decoder, OpenAIWhisperDecoder):
            decoder_embed = self.decoder.decoders.token_embedding
        elif isinstance(self.decoder.embed, torch.nn.Sequential):
            decoder_embed = self.decoder.embed[0]
        else:
            decoder_embed = self.decoder.embed
            
        text_embed_matrix = torch.cat([
            decoder_embed.weight.data, 
            self.contextualizer.encoder.oovembed.weight.data,
        ], dim=0)

        logging.info(f'text_embed_matrix shape: {text_embed_matrix.shape}')
        context_embed = text_embed_matrix[context_idxs]
        logging.info(f'model_embed shape: {model_embed.shape}')
        logging.info(f'context_embed shape: {context_embed.shape}')
        
        out = self.contextualizer(
            model_embed=model_embed,
            context_embed=context_embed,
        )
        logging.info(f'fusion out shape: {out.shape}')
        return model_embed + out