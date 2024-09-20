import logging
from typing import Any, Dict, List, NamedTuple, Tuple

import torch
from packaging.version import parse as V
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search import Hypothesis
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.scorer_interface import ScorerInterface
from espnet.nets.e2e_asr_common import end_detect

# Import necessary modules for contextualization
from espnet2.torch_utils.device_funcs import to_device
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.contextualizer import (
    CONTEXTUAL_RETRIEVER,
    CONTEXTUAL_ADAPTER_ENCODER,
)
from espnet2.asr.contextualizer.func.contextual_retriever_func import (
    topk_decode,
)
from espnet2.asr.contextualizer.func.contextual_adapter_func import (
    forward_contextual_adapter,
)

logger = logging.getLogger(__name__)


class ContextualBatchHypothesis(NamedTuple):
    """Batchified/Vectorized contextual hypothesis data type."""

    yseq: torch.Tensor = torch.tensor([])  # (batch, maxlen)
    score: torch.Tensor = torch.tensor([])  # (batch,)
    length: torch.Tensor = torch.tensor([])  # (batch,)
    scores: Dict[str, torch.Tensor] = dict()  # values: (batch,)
    states: Dict[str, Dict] = dict()
    hs: List[torch.Tensor] = []  # (batch, maxlen, adim)
    context_yseq: torch.Tensor = torch.tensor([])  # (batch, context_maxlen)
    context_score: torch.Tensor = torch.tensor([])  # (batch,)
    context_scores: Dict[str, torch.Tensor] = dict()
    context_states: Dict[str, Dict] = dict()
    context_hs: List[torch.Tensor] = []  # (batch, context_maxlen, adim)

    def __len__(self) -> int:
        """Return a batch size."""
        return len(self.length)


class ContextualBatchBeamSearch(BatchBeamSearch):
    """Contextual batch beam search implementation."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        contextualizer: object,
        contextualizer_conf: Dict[str, Any],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        sop: int,
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

        # Contextual ASR attributes
        self.contextualizer = contextualizer
        self.contextualizer_conf = contextualizer_conf
        self.sop = sop
        self.use_ctc_only_decoding = 'decoder' not in self.scorers
        if not self.use_ctc_only_decoding:
            self.decoder = self.scorers['decoder']
        else:
            self.return_hs = False
        logger.info('Initialized ContextualBatchBeamSearch.')

    def batchfy(self, hyps: List[Hypothesis]) -> ContextualBatchHypothesis:
        """Convert list to batch of contextual hypotheses."""
        if len(hyps) == 0:
            return ContextualBatchHypothesis()

        if self.return_hs:
            hs = [h.hs for h in hyps]
        else:
            hs = []

        return ContextualBatchHypothesis(
            yseq=pad_sequence(
                [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
            ),
            length=torch.tensor([len(h.yseq) for h in hyps], dtype=torch.int64),
            score=torch.tensor([h.score for h in hyps]),
            scores={k: torch.tensor([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
            hs=hs,
            context_yseq=pad_sequence(
                [h.context_yseq for h in hyps], batch_first=True, padding_value=self.eos
            )
            if hasattr(hyps[0], 'context_yseq')
            else torch.tensor([]),
            context_score=torch.tensor(
                [h.context_score for h in hyps]
                if hasattr(hyps[0], 'context_score')
                else []
            ),
            context_scores={
                k: torch.tensor([h.context_scores[k] for h in hyps])
                for k in self.scorers
                if hasattr(hyps[0], 'context_scores')
            },
            context_states={
                k: [h.context_states[k] for h in hyps]
                for k in self.scorers
                if hasattr(hyps[0], 'context_states')
            },
            context_hs=[
                h.context_hs for h in hyps
            ]
            if self.return_hs and hasattr(hyps[0], 'context_hs')
            else [],
        )

    def _batch_select(self, hyps: ContextualBatchHypothesis, ids: List[int]) -> ContextualBatchHypothesis:
        if self.return_hs:
            hs = [hyps.hs[i] for i in ids]
        else:
            hs = []

        return ContextualBatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
            hs=hs,
            context_yseq=hyps.context_yseq[ids]
            if hyps.context_yseq.size(0) > 0
            else torch.tensor([]),
            context_score=hyps.context_score[ids]
            if hyps.context_score.size(0) > 0
            else torch.tensor([]),
            context_scores={k: v[ids] for k, v in hyps.context_scores.items()}
            if hyps.context_scores
            else {},
            context_states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.context_states.items()
            }
            if hyps.context_states
            else {},
            context_hs=[hyps.context_hs[i] for i in ids]
            if self.return_hs and hyps.context_hs
            else [],
        )

    def _select(self, hyps: ContextualBatchHypothesis, i: int) -> Hypothesis:
        hyp = Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
            hs=hyps.hs[i] if self.return_hs else [],
        )
        if hyps.context_yseq.size(0) > 0:
            hyp.context_yseq = hyps.context_yseq[i][: hyps.length[i]]
            hyp.context_score = hyps.context_score[i]
            hyp.context_scores = {k: v[i] for k, v in hyps.context_scores.items()}
            hyp.context_states = {
                k: self.scorers[k].select_state(v, i)
                for k, v in hyps.context_states.items()
            }
            hyp.context_hs = hyps.context_hs[i] if self.return_hs else []
        return hyp

    def unbatchfy(self, batch_hyps: ContextualBatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        hyps = []
        for i in range(len(batch_hyps.length)):
            hyp = Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={
                    k: self.scorers[k].select_state(batch_hyps.states[k], i)
                    for k in self.scorers
                },
                hs=batch_hyps.hs[i] if self.return_hs else [],
            )
            if batch_hyps.context_yseq.size(0) > 0:
                hyp.context_yseq = batch_hyps.context_yseq[i][: batch_hyps.length[i]]
                hyp.context_score = batch_hyps.context_score[i]
                hyp.context_scores = {
                    k: batch_hyps.context_scores[k][i] for k in self.scorers
                }
                hyp.context_states = {
                    k: self.scorers[k].select_state(batch_hyps.context_states[k], i)
                    for k in self.scorers
                }
                hyp.context_hs = batch_hyps.context_hs[i] if self.return_hs else []
            hyps.append(hyp)
        return hyps

    def init_hyp(
        self,
        x: torch.Tensor,
        contexts: Dict[str, Any] = None,
        pred_contexts: Any = None,
    ) -> ContextualBatchHypothesis:
        """Initialize hypotheses with contextual information."""
        batch_size = x.size(0)
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.batch_init_state(x)
            init_scores[k] = torch.zeros(batch_size, dtype=torch.float, device=x.device)

        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer

        yseq_list = []
        context_yseq_list = []
        for i in range(batch_size):
            sample_primer = primer.copy()
            if (
                'decoder' in self.scorers
                and isinstance(self.scorers['decoder'], OpenAIWhisperDecoder)
                and contexts is not None
            ):
                sample_contexts = {k: v[i] for k, v in contexts.items()}
                nlp_prompt_tensor = sample_contexts["nlp_prompt_tensor"][0].tolist()

                sample_primer = [self.sop] + nlp_prompt_tensor + sample_primer

            yseq_list.append(torch.tensor(sample_primer, device=x.device))
            context_yseq_list.append(torch.tensor(sample_primer, device=x.device))

        yseq_padded = pad_sequence(yseq_list, batch_first=True, padding_value=self.eos)
        lengths = torch.tensor([len(yseq) for yseq in yseq_list], dtype=torch.int64)
        context_yseq_padded = pad_sequence(
            context_yseq_list, batch_first=True, padding_value=self.eos
        )

        return ContextualBatchHypothesis(
            yseq=yseq_padded,
            length=lengths,
            score=torch.zeros(batch_size, dtype=torch.float, device=x.device),
            scores=init_scores,
            states=init_states,
            hs=[],
            context_yseq=context_yseq_padded,
            context_score=torch.zeros(batch_size, dtype=torch.float, device=x.device),
            context_scores=init_scores.copy(),
            context_states=init_states.copy(),
            context_hs=[],
        )

    def score_full(
        self,
        hyp: ContextualBatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypotheses by `self.full_scorers`.

        Args:
            hyp (ContextualBatchHypothesis): Batch of hypotheses to score.
            x (torch.Tensor): Encoded speech features.
            pre_x (torch.Tensor): Optional pre-encoded features for sequential attention.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
                Tuple of scores and states for the hypotheses.
        """
        scores = dict()
        states = dict()
        hs = None

        for k, d in self.full_scorers.items():
            if "decoder" in k and self.return_hs:
                (scores[k], hs), states[k] = d.batch_score(
                    hyp.yseq, hyp.states[k], x, return_hs=self.return_hs
                )
            elif "decoder" in k and pre_x is not None:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x, pre_x)
            else:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)

        if self.return_hs:
            return hs, scores, states
        return scores, states

    def search(
        self,
        running_hyps: ContextualBatchHypothesis,
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
    ) -> ContextualBatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x."""
        n_batch = len(running_hyps)
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores = torch.zeros(
            n_batch, self.n_vocab, dtype=x.dtype, device=x.device
        )
        if self.return_hs:
            hs, scores, states = self.score_full(
                running_hyps,
                x,
                pre_x=pre_x,
            )
        else:
            scores, states = self.score_full(
                running_hyps,
                x,
                pre_x=pre_x,
            )

        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]
        # partial scoring
        if self.do_pre_beam:
            pre_beam_scores = (
                weighted_scores
                if self.pre_beam_score_key == "full"
                else scores[self.pre_beam_score_key]
            )
            part_ids = torch.topk(pre_beam_scores, self.pre_beam_size, dim=-1)[1]
        # NOTE(takaaki-hori): Unlike BeamSearch, we assume that score_partial returns
        # full-size score matrices, which has non-zero scores for part_ids and zeros
        # for others.
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x, pre_x)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += running_hyps.score.to(
            dtype=x.dtype, device=x.device
        ).unsqueeze(1)

        # update hyps
        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in zip(*self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            if self.return_hs:
                new_hs = prev_hyp.hs + [hs[full_prev_hyp_id].squeeze(0)]
            else:
                new_hs = []
            new_hyp = Hypothesis(
                score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                scores=self.merge_scores(
                    prev_hyp.scores,
                    {k: v[full_prev_hyp_id] for k, v in scores.items()},
                    full_new_token_id,
                    {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                    part_new_token_id,
                ),
                states=self.merge_states(
                    {
                        k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                        for k, v in states.items()
                    },
                    {
                        k: self.part_scorers[k].select_state(
                            v, part_prev_hyp_id, part_new_token_id
                        )
                        for k, v in part_states.items()
                    },
                    part_new_token_id,
                ),
                hs=new_hs,
            )
            # Handle contextual information if necessary
            if hasattr(prev_hyp, 'context_yseq'):
                new_hyp.context_yseq = prev_hyp.context_yseq  # or update if needed
                new_hyp.context_score = prev_hyp.context_score
                new_hyp.context_scores = prev_hyp.context_scores
                new_hyp.context_states = prev_hyp.context_states
                new_hyp.context_hs = prev_hyp.context_hs
            best_hyps.append(new_hyp)
        return self.batchfy(best_hyps)

    def forward(
        self,
        x: torch.Tensor,
        contexts: Dict[str, Any],
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_x: torch.Tensor = None,
    ) -> List[Hypothesis]:
        """Perform batch beam search with contextual information."""
        logger.info('Starting contextual batch beam search.')
        if pre_x is not None:
            inp = pre_x
        else:
            inp = x
        batch_size = inp.size(0)
        if maxlenratio == 0:
            maxlen = inp.size(1)
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * inp.size(1)))

        if minlenratio < 0:
            minlen = -1 * int(minlenratio)
        else:
            minlen = int(minlenratio * inp.size(1))

        if (
            not self.use_ctc_only_decoding
            and 'decoder' in self.scorers
            and isinstance(self.scorers['decoder'], OpenAIWhisperDecoder)
            and hasattr(self.scorers['decoder'].decoders, 'positional_embedding')
        ):
            pos_len = self.scorers['decoder'].decoders.positional_embedding.shape[0]
            if maxlen > pos_len:
                maxlen = pos_len - 4
        logger.info(f"Decoder input length: {inp.size(1)}")
        logger.info(f"Max output length: {maxlen}")
        logger.info(f"Min output length: {minlen}")

        # Encoder contextualization
        pred_contexts = None
        context_preds = None
        if (
            self.contextualizer_conf["contextualizer_type"]
            in CONTEXTUAL_RETRIEVER
        ):
            logger.info('Using retriever for encoder contextualization.')
            contexts = to_device(contexts, device=x.device)
            context_prob, encoder_proj = self.contextualizer(
                query=x,
                query_ilens=None,
                context_subword=contexts['blist'],
                context_subword_ilens=contexts['ilens'],
                context_phone=contexts['blist_xphone'],
                context_phone_ilens=contexts['blist_xphone_ilens'],
                return_model_proj=True,
            )
            x = encoder_proj
            context_preds = topk_decode(
                context_prob,
                contexts['context_list_idxs'],
                idx_blank=0,
                top_k=5,
                threshold=0.6,
            )
        elif (
            self.contextualizer_conf["contextualizer_type"]
            in CONTEXTUAL_ADAPTER_ENCODER
        ):
            logger.info('Using adapter for encoder contextualization.')
            contexts = to_device(contexts, device=x.device)
            bias_vec = forward_contextual_adapter(
                contextualizer=self.contextualizer,
                model_embed=x,
                context_idxs=contexts['blist'],
                context_xphone_idxs=contexts['blist_xphone_mean'],
                ilens=contexts['ilens'],
            )
            x = x + bias_vec

        running_hyps = self.init_hyp(
            x if pre_x is None else pre_x,
            contexts=contexts,
            pred_contexts=(pred_contexts if pred_contexts is not None else None),
        )
        ended_hyps = []
        for i in range(maxlen):
            logger.debug(f"Position {i}")
            best = self.search(running_hyps, x, pre_x=pre_x)
            running_hyps = self.post_process(
                i, maxlen, minlen, maxlenratio, best, ended_hyps
            )
            if maxlenratio == 0.0 and end_detect(
                [h.asdict() for h in ended_hyps], i
            ):
                logger.info(f"End detected at {i}")
                break
            if len(running_hyps) == 0:
                logger.info("No hypothesis left. Finishing decoding.")
                break
            else:
                logger.debug(f"Remaining hypotheses: {len(running_hyps)}")

        if self.normalize_length:
            nbest_hyps = sorted(
                self.unbatchfy(ended_hyps),
                key=lambda x: x.score / (len(x.yseq) - 1),
                reverse=True,
            )
        else:
            nbest_hyps = sorted(
                self.unbatchfy(ended_hyps),
                key=lambda x: x.score,
                reverse=True,
            )

        if len(nbest_hyps) == 0:
            logger.warning(
                "No N-best results, performing recognition again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, contexts, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logger.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logger.info(f"Total log probability: {best.score:.2f}")
        logger.info(f"Normalized log probability: {best.score / len(best.yseq):.2f}")
        logger.info(f"Total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logger.info(
                "Best hypothesis: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        if best.yseq[1:-1].shape[0] == maxlen:
            logger.warning(
                f"Best hypothesis length {best.yseq[1:-1].shape[0]} equals max output length {maxlen}."
            )
            logger.warning(
                "Decoding may have been stopped by the max output length limitation. Consider increasing maxlenratio."
            )

        if context_preds is not None:
            context_yseq = []
            context_score = []
            for pred in context_preds:
                context_yseq.extend(pred[1])
                context_score.append(pred[2])
            context_yseq = [1] + context_yseq + [1]
        else:
            context_yseq = [1, 1]
            context_score = []

        for nbest_hyp in nbest_hyps:
            nbest_hyp.context_yseq = torch.tensor(
                context_yseq, device=nbest_hyp.yseq.device
            )
            nbest_hyp.context_score = sum(context_score)
        return nbest_hyps
