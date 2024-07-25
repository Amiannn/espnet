"""Beam search module."""

import logging
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import torch
from espnet2.torch_utils.device_funcs import to_device
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface, ScorerInterface

from espnet.nets.beam_search       import Hypothesis
from espnet.nets.beam_search       import BeamSearch

from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder

from espnet2.asr.contextualizer.func.contextual_adapter_func   import forward_contextual_adapter
from espnet2.asr.contextualizer import (
    CONTEXTUAL_RETRIEVER,
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_ADAPTER_DECODER
)

from espnet2.asr.contextualizer.func.contextual_retriever_func import retrieve_greedy_decode

logger = logging.getLogger(__name__)


class ContextualHypothesis(Hypothesis):
    """Contextual hypothesis data type."""
    ...

class ContextualBeamSearch(BeamSearch):
    """Beam search implementation."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        contextualizer: object,
        contextualizer_conf: object,
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        token_list: List[str] = None,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
        return_hs: bool = False,
        hyp_primer: List[int] = None,
        normalize_length: bool = False,
    ):
        super().__init__(
            scorers=scorers,
            weights=weights,
            beam_size=beam_size,
            vocab_size=vocab_size,
            sos=sos,
            eos=eos,
            token_list=token_list,
            pre_beam_ratio=pre_beam_ratio,
            pre_beam_score_key=pre_beam_score_key,
            return_hs=return_hs,
            hyp_primer=hyp_primer,
            normalize_length=normalize_length,
        )

        # contextual asr
        self.contextualizer      = contextualizer
        self.contextualizer_conf = contextualizer_conf

        self.use_ctc_only_deocding = ('decoder' not in self.scorers)
        if not self.use_ctc_only_deocding:
            self.decoder = self.scorers['decoder']
        else:
            self.return_hs = False
        logging.info(f'Doing contextual asr beam search class!')

    def score_full(
        self, hyp: Hypothesis, x: torch.Tensor, pre_x: torch.Tensor = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            if "decoder" in k and self.return_hs:
                scores[k], hs, states[k] = d.score(
                    hyp.yseq, hyp.states[k], x, return_hs=self.return_hs
                )
                # logging.info(f'scores[k]: {scores[k].shape}')
                # logging.info(f'hs: {scores[k].shape}')
                # logging.info(f'states[k]: {states[k]}')

            elif pre_x is not None:
                raise NotImplementedError(
                    "Score pre_x for contextual asr is not supported."
                )
                # scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x, pre_x)
            else:
                raise NotImplementedError(
                    "Sore not return hs for contextual asr is not supported."
                )
                # scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)

        if self.return_hs:
            return hs, scores, states
        return scores, states
    
    def init_hyp(
            self, 
            x: torch.Tensor, 
            contexts: object = None, 
            pred_contexts: object = None
        ) -> List[Hypothesis]:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0

        # NOTE (Shih-Lun): added for OpenAI Whisper ASR
        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer

        if isinstance(self.scorers['decoder'], OpenAIWhisperDecoder):
            sot_lang_token     = primer[:2]
            no_timestamp_token = [primer[-1]]
            primer = sot_lang_token + contexts["nlp_prompt_context_template"].tolist() + pred_contexts + contexts["nlp_prompt_no_context_template"].tolist() + no_timestamp_token
        logging.info(f'primer: {primer}')

        return [
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                hs=[],
                yseq=torch.tensor(primer, device=x.device),
            )
        ]

    def forward(
        self,
        x: torch.Tensor,
        contexts: object,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_x: torch.Tensor = None,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.
                If minlenratio<0.0, its absolute value is interpreted
                as a constant min output length.
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        logging.info(f'In contextual AED beam search')
        # set length bounds
        if pre_x is not None:
            inp = pre_x
        else:
            inp = x
        if maxlenratio == 0:
            maxlen = inp.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * inp.size(0)))

        if minlenratio < 0:
            minlen = -1 * int(minlenratio)
        else:
            minlen = int(minlenratio * inp.size(0))
        
        # May cause some problem
        if not self.use_ctc_only_deocding and isinstance(self.scorers['decoder'], OpenAIWhisperDecoder):
            pos_len = self.scorers['decoder'].decoders.positional_embedding.shape[0]
            if maxlen > pos_len:
                # logging.info(f'original maxlen: {maxlen}, after: {pos_len}')
                # (4) -> special tokens
                maxlen = pos_len - 4
        logger.info("decoder input length: " + str(inp.shape[0]))
        logger.info("max output length: " + str(maxlen))
        logger.info("min output length: " + str(minlen))

        # Encoder contextualization
        pred_contexts = None
        if self.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_RETRIEVER:
            logging.info(f'Encoder contextualize, use retriever!')
            contexts     = to_device(contexts, device=x.device)
            context_prob = self.contextualizer(
                model_embed=x.unsqueeze(0),
                context_embed=contexts['blist'],
                context_xphone_embed=contexts['blist_xphone_mean'],
                ilens=contexts['ilens'],
            )
            # only for whisper model
            sep_toknes = [11, 220]
            pred_contexts = retrieve_greedy_decode(
                context_prob, 
                contexts['context_list_idxs'],
                sep_toknes
            )
        
        elif self.contextualizer_conf["contextualizer_type"] in CONTEXTUAL_ADAPTER_ENCODER:
            logging.info(f'Encoder contextualize!')
            contexts = to_device(contexts, device=x.device)
            x        = x.unsqueeze(0)
            bias_vec = forward_contextual_adapter(
                contextualizer=self.contextualizer,
                model_embed=x,
                context_idxs=contexts['blist'],
                context_xphone_idxs=contexts['blist_xphone_mean'],
                ilens=contexts['ilens']
            )
            x = x + bias_vec
            x = x.squeeze(0)

        # main loop of prefix search
        running_hyps = self.init_hyp(
            x if pre_x is None else pre_x,
            contexts=contexts,
            pred_contexts=(pred_contexts if pred_contexts is not None else None)
        )
        ended_hyps = []
        for i in range(maxlen):
            logger.debug("position " + str(i))
            best = self.search(running_hyps, x, pre_x=pre_x)
            # post process of one iteration
            running_hyps = self.post_process(
                i, maxlen, minlen, maxlenratio, best, ended_hyps
            )
            # end detection
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logger.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logger.info("no hypothesis. Finish decoding.")
                break
            else:
                logger.debug(f"remained hypotheses: {len(running_hyps)}")

        if self.normalize_length:
            # Note (Jinchuan): -1 since hyp starts with <sos> and
            # initially has score of 0.0
            nbest_hyps = sorted(
                ended_hyps, key=lambda x: x.score / (len(x.yseq) - 1), reverse=True
            )
        else:
            nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)

        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logger.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logger.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logger.info(f"total log probability: {best.score:.2f}")
        logger.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logger.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logger.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        if best.yseq[1:-1].shape[0] == maxlen:
            logger.warning(
                "best hypo length: {} == max output length: {}".format(
                    best.yseq[1:-1].shape[0], maxlen
                )
            )
            logger.warning(
                "decoding may be stopped by the max output length limitation, "
                + "please consider to increase the maxlenratio."
            )
        return nbest_hyps

def beam_search(
    x: torch.Tensor,
    sos: int,
    eos: int,
    beam_size: int,
    vocab_size: int,
    scorers: Dict[str, ScorerInterface],
    weights: Dict[str, float],
    token_list: List[str] = None,
    maxlenratio: float = 0.0,
    minlenratio: float = 0.0,
    pre_beam_ratio: float = 1.5,
    pre_beam_score_key: str = "full",
) -> list:
    """Perform beam search with scorers.

    Args:
        x (torch.Tensor): Encoded speech feature (T, D)
        sos (int): Start of sequence id
        eos (int): End of sequence id
        beam_size (int): The number of hypotheses kept during search
        vocab_size (int): The number of vocabulary
        scorers (dict[str, ScorerInterface]): Dict of decoder modules
            e.g., Decoder, CTCPrefixScorer, LM
            The scorer will be ignored if it is `None`
        weights (dict[str, float]): Dict of weights for each scorers
            The scorer will be ignored if its weight is 0
        token_list (list[str]): List of tokens for debug log
        maxlenratio (float): Input length ratio to obtain max output length.
            If maxlenratio=0.0 (default), it uses a end-detect function
            to automatically find maximum hypothesis lengths
        minlenratio (float): Input length ratio to obtain min output length.
        pre_beam_score_key (str): key of scores to perform pre-beam search
        pre_beam_ratio (float): beam size in the pre-beam search
            will be `int(pre_beam_ratio * beam_size)`

    Returns:
        list: N-best decoding results

    """
    ret = BeamSearch(
        scorers,
        weights,
        beam_size=beam_size,
        vocab_size=vocab_size,
        pre_beam_ratio=pre_beam_ratio,
        pre_beam_score_key=pre_beam_score_key,
        sos=sos,
        eos=eos,
        token_list=token_list,
    ).forward(x=x, maxlenratio=maxlenratio, minlenratio=minlenratio)
    return [h.asdict() for h in ret]
