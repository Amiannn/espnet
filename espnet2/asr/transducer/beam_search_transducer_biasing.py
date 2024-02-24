"""Search algorithms for Transducer models."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.asr.prototype.tcpgen_prototype import TCPGenPrototype
from espnet2.lm.transformer_lm import TransformerLM
from espnet.nets.pytorch_backend.transducer.utils import (
    is_prefix,
    recombine_hyps,
    select_k_expansions,
    subtract,
)

from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.transducer.beam_search_transducer import ExtendedHypothesis
from espnet2.asr.transducer.beam_search_transducer import Hypothesis

from espnet2.text.trie_processor import TrieProcessor

@dataclass
class BiasingHypothesis(Hypothesis):
    """Biasing hypothesis definition for trie beam search."""
    previous_node: Dict = None


class BeamSearchTransducerBiasing(BeamSearchTransducer):
    """Beam search implementation for Transducer."""

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        rareword: TCPGenPrototype,
        rareword_conf: dict,
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
        """Initialize Transducer search module.

        Args:
            decoder: Decoder module.
            joint_network: Joint network module.
            beam_size: Beam size.
            lm: LM class.
            lm_weight: LM weight for soft fusion.
            search_type: Search algorithm to use during inference.
            max_sym_exp: Number of maximum symbol expansions at each time step. (TSD)
            u_max: Maximum output sequence length. (ALSD)
            nstep: Number of maximum expansion steps at each time step. (NSC/mAES)
            prefix_alpha: Maximum prefix length in prefix search. (NSC/mAES)
            expansion_beta:
              Number of additional candidates for expanded hypotheses selection. (mAES)
            expansion_gamma: Allowed logp difference for prune-by-value method. (mAES)
            multi_blank_durations: The duration of each blank token. (MBG)
            multi_blank_indices: The index of each blank token in token_list. (MBG)
            score_norm: Normalize final scores by length. ("default")
            score_norm_during:
              Normalize scores by length during search. (default, TSD, ALSD)
            nbest: Number of final hypothesis.

        """
        logging.info('Biasing beam search!')
        self.decoder = decoder
        self.joint_network = joint_network
        self.rareword = rareword
        self.rareword_conf = rareword_conf
        self.beam_size = beam_size
        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim

        self.sos = self.vocab_size - 1
        self.token_list = token_list

        self.blank_id = decoder.blank_id

        if search_type == "mbg":
            self.beam_size = 1
            self.multi_blank_durations = multi_blank_durations
            self.multi_blank_indices = multi_blank_indices
            self.search_algorithm = self.multi_blank_greedy_search

        elif self.beam_size <= 1:
            self.search_algorithm = self.greedy_search_trie
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search_trie
        elif search_type == "tsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.nstep = nstep
            self.prefix_alpha = prefix_alpha

            self.search_algorithm = self.nsc_beam_search
        elif search_type == "maes":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.nstep = nstep if nstep > 1 else 2
            self.prefix_alpha = prefix_alpha
            self.expansion_gamma = expansion_gamma

            assert self.vocab_size >= beam_size + expansion_beta, (
                "beam_size (%d) + expansion_beta (%d) "
                "should be smaller or equal to vocabulary size (%d)."
                % (beam_size, expansion_beta, self.vocab_size)
            )
            self.max_candidates = beam_size + expansion_beta

            self.search_algorithm = self.modified_adaptive_expansion_search

        else:
            raise NotImplementedError

        self.use_lm = lm is not None
        self.lm = lm
        self.lm_weight = lm_weight

        if self.use_lm and self.beam_size == 1:
            logging.warning("LM is provided but not used, since this is greedy search.")

        self.score_norm = score_norm
        self.score_norm_during = score_norm_during
        self.nbest = nbest

    def __call__(
        self, enc_out: torch.Tensor, biasing_words: List, biasing_tries: Dict
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis], List[BiasingHypothesis]]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        # logging.info(f'beamsearch biasing_words -> {str(biasing_words)[:100]}')
        self.decoder.set_device(enc_out.device)

        nbest_hyps = self.search_algorithm(enc_out, biasing_tries)

        return nbest_hyps

    def transducer_with_tcpgen(self, tcpgen, enc_out_t, pervious_token, dec_out, decoder_embed, masks, masks_gate):
        (
            ptr_dist, 
            h_ptr, 
            dbias, 
        ) = tcpgen.forward_attention(
            enc_out_t, 
            pervious_token,
            decoder_embed,
            None, 
            masks,
            skip_dropout=True
        )
        if not self.rareword_conf['deepbiasing']:
            dbias = None
        joint_out, h_joint = self.joint_network(
            enc_out_t.unsqueeze(0), 
            dec_out.unsqueeze(0), 
            bias_out=dbias,
            return_h_joint=True
        )
        logging.info(f'enc_out_t shape: {enc_out_t.shape}')
        logging.info(f'dec_out shape  : {dec_out.shape}')
        logging.info(f'joint_out shape: {joint_out.shape}')
        logging.info(f'h_joint shape  : {h_joint.shape}')
        logging.info(f'h_ptr shape    : {h_ptr.shape}')
        gate       = tcpgen.forward_gate(h_joint, h_ptr, masks_gate)
        logging.info(f'gate: {gate}')
        model_dist = torch.nn.functional.softmax(joint_out, dim=-1)
        joint_out  = tcpgen.forward_copy_mechanism(model_dist, ptr_dist, gate)
        logp       = torch.log_softmax(joint_out, dim=-1)
        return logp[0, 0, 0, :]

    def greedy_search_trie(self, enc_out: torch.Tensor, biasing_tries: Dict) -> List[BiasingHypothesis]:
        """Greedy search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypotheses.

        """
        logging.info(f'greedy_search_trie biasing_tries -> {str(biasing_tries)[:10]}')

        dec_state = self.decoder.init_state(1)

        hyp = BiasingHypothesis(
            score=0.0, 
            yseq=[self.blank_id], 
            dec_state=dec_state, 
            previous_node=biasing_tries
        )
        cache = {}

        dec_out, state, _ = self.decoder.score(hyp, cache)

        for i, enc_out_t in enumerate(enc_out):
            vy = hyp.yseq[-1] if len(hyp.yseq) > 1 else self.blank_id
            mask, mask_gate, now_node = TrieProcessor.search_trie_one_step(
                previous_token_id=vy, 
                root=biasing_tries, 
                previous_node=hyp.previous_node, 
                id2token=self.token_list
            )
            
            logging.info(f'frame: {i}')
            logging.info(f'previous token: {self.token_list[vy]}')
            logging.info(f'gate: {mask_gate[0]}')
            # ookb_list = self.token_list + ['<OOL>']
            # logging.info(f'mask: {[ookb_list[m] for m in mask[0]]}')
            logging.info('_' * 30)
            
            enc_out_t      = enc_out_t.reshape(1, 1, -1) 
            dec_out        = dec_out.reshape(1, 1, -1)
            mask           = torch.from_numpy(mask).unsqueeze(0)
            mask_gate      = torch.from_numpy(mask_gate).unsqueeze(0)
            pervious_token = torch.tensor(vy).unsqueeze(0)

            # logp = torch.log_softmax(
            #     self.joint_network(enc_out_t, dec_out),
            #     dim=-1,
            # )
            # top_logp, pred = torch.max(logp, dim=-1)

            logp = self.transducer_with_tcpgen(
                tcpgen=self.rareword,
                enc_out_t=enc_out_t, 
                pervious_token=pervious_token, 
                dec_out=dec_out,
                decoder_embed=self.decoder.embed, 
                masks=mask,
                masks_gate=mask_gate
            )

            top_logp, pred = torch.max(logp, dim=-1)

            if pred != self.blank_id:
                hyp.previous_node = now_node
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)

                hyp.dec_state = state

                dec_out, state, _ = self.decoder.score(hyp, cache)

        return [hyp]

    def default_beam_search_trie(self, enc_out: torch.Tensor, biasing_tries: Dict) -> List[BiasingHypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        logging.info(f'beam_search_trie biasing_tries -> {str(biasing_tries)[:10]}')

        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [
            BiasingHypothesis(
                score=0.0, 
                yseq=[self.blank_id], 
                dec_state=dec_state,
                previous_node=biasing_tries
            )
        ]
        cache = {}
        cache_lm = {}

        for i, enc_out_t in enumerate(enc_out):
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
                
                vy = max_hyp.yseq[-1] if len(max_hyp.yseq) > 1 else self.blank_id
                mask, mask_gate, now_node = TrieProcessor.search_trie_one_step(
                    previous_token_id=vy, 
                    root=biasing_tries, 
                    previous_node=max_hyp.previous_node, 
                    id2token=self.token_list
                )
                
                logging.info(f'frame: {i}')
                logging.info(f'previous token: {self.token_list[vy]}')
                logging.info(f'gate: {mask_gate[0]}')
                ookb_list = self.token_list + ['<OOL>']
                logging.info(f'mask: {[ookb_list[m] for m in mask[0]]}')
                logging.info('_' * 30)

                enc_out_t      = enc_out_t.reshape(1, 1, -1) 
                dec_out        = dec_out.reshape(1, 1, -1)
                mask           = torch.from_numpy(mask).unsqueeze(0)
                mask_gate      = torch.from_numpy(mask_gate).unsqueeze(0)
                pervious_token = torch.tensor(vy).unsqueeze(0)

                logp = self.transducer_with_tcpgen(
                    tcpgen=self.rareword,
                    enc_out_t=enc_out_t, 
                    pervious_token=pervious_token, 
                    dec_out=dec_out,
                    decoder_embed=self.decoder.embed, 
                    masks=mask,
                    masks_gate=mask_gate
                )

                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    BiasingHypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                        previous_node=max_hyp.previous_node
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
                        BiasingHypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                            previous_node=now_node
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