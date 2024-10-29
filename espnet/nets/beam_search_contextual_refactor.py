"""Beam search module with contextualization."""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import ScorerInterface, PartialScorerInterface
from espnet2.asr.contextualizer import (
    CONTEXTUAL_ADAPTER_DECODER,
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_RETRIEVER,
)
from espnet2.asr.contextualizer.func.contextual_adapter_func import (
    forward_contextual_adapter,
)
from espnet2.asr.contextualizer.func.contextual_retriever_func import (
    decode_topk_tokens,
    generate_prompt_from_hypotheses,
)
from espnet2.torch_utils.device_funcs import to_device
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder

logger = logging.getLogger(__name__)


class ContextualHypothesis(NamedTuple):
    """Hypothesis class with contextual information."""

    yseq: torch.Tensor
    context_yseq: torch.Tensor
    score: Union[float, torch.Tensor] = 0
    context_score: Union[float, torch.Tensor] = 0
    scores: Dict[str, Union[float, torch.Tensor]] = dict()
    context_scores: Dict[str, Union[float, torch.Tensor]] = dict()
    context_states: Dict[str, Any] = dict()
    states: Dict[str, Any] = dict()
    hs: List[torch.Tensor] = []
    context_hs: List[torch.Tensor] = []
    context_idxs: List[torch.Tensor] = []

    def asdict(self) -> dict:
        """Convert data to a JSON-friendly dictionary."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
            context_yseq=self.context_yseq.tolist(),
            context_score=float(self.context_score),
            context_scores={k: float(v) for k, v in self.context_scores.items()},
            context_idxs=[str(idx) for idx in self.context_idxs],
        )._asdict()

class ContextualizedDecoderScorer(ScorerInterface):
    """Wrapper scorer that adds contextualization to the decoder."""

    def __init__(
        self,
        decoder_scorer: ScorerInterface,
        contextualizer: Any,
        contextualizer_config: Dict[str, Any],
        context_data: Dict[str, Any],
    ):
        self.decoder_scorer = decoder_scorer
        self.contextualizer = contextualizer
        self.contextualizer_config = contextualizer_config
        self.context_data = context_data

    def init_state(self, x: torch.Tensor) -> Any:
        """Initialize the decoder state."""
        return self.decoder_scorer.init_state(x)

    def score(
        self,
        yseq: torch.Tensor,
        state: Any,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        """Score the hypothesis with decoder contextualization."""
        # Force return_hs=True to get the hidden state
        score, hidden_state, state = self.decoder_scorer.score(yseq, state, x, return_hs=True, *args, **kwargs)

        # Apply decoder contextualization if enabled
        if self.contextualizer_config["contextualizer_type"] in CONTEXTUAL_ADAPTER_DECODER:
            if hidden_state is None:
                # If hidden_state is not provided, we cannot apply contextualization
                logger.warning("Hidden state not available for contextualization.")
                return score, state

            decoder_embedding = hidden_state  # Shape: (D,)
            # Apply decoder contextualizer
            decoder_bias_vector, context_hypotheses = self.contextualizer(
                model_embed=decoder_embedding.unsqueeze(0),  # Shape: (1, D)
                context_embed=self.context_data["blist"],
                ilens=self.context_data["ilens"],
                return_atten=True,
            )
            # Mean across attention heads
            context_hypotheses = torch.mean(context_hypotheses, dim=1)
            context_predictions = decode_topk_tokens(
                token_probs=context_hypotheses,
                vocabulary=self.context_data["context_list"],
                blank_index=0,
                top_k=100,
                threshold=0.01,
            )
            pred_texts = ", ".join([pred[1] for pred in context_predictions])
            logging.info(f'pred_texts: {pred_texts}')

            # Bias the hidden state
            # hidden_state = hidden_state + decoder_bias_vector
            
            # Adjust the score
            adjusted_score = torch.log_softmax(self.decoder_scorer.output_layer(hidden_state), dim=-1)
            adjusted_score = adjusted_score.squeeze(0)
            return adjusted_score, state
        else:
            return score, state

class ContextualBeamSearch(BeamSearch):
    """Beam search implementation with contextualization."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        contextualizer: Any,
        contextualizer_config: Dict[str, Any],
        context_sampler: Any,
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        sop: int,
        token_list: List[str] = None,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
        return_hidden_states: bool = False,
        hypothesis_primer: List[int] = None,
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
            return_hs=return_hidden_states,
            hyp_primer=hypothesis_primer,
            normalize_length=normalize_length,
        )

        self.contextualizer = contextualizer
        self.contextualizer_config = contextualizer_config
        self.context_sampler = context_sampler
        self.sop = sop

        self.use_ctc_only_decoding = "decoder" not in self.scorers
        if not self.use_ctc_only_decoding:
            # Wrap the decoder scorer with ContextualizedDecoderScorer
            self.decoder = scorers["decoder"]
            self.scorers["decoder"] = ContextualizedDecoderScorer(
                decoder_scorer=self.decoder,
                contextualizer=self.contextualizer,
                contextualizer_config=self.contextualizer_config,
                context_data={},  # Placeholder, will be updated in forward()
            )
            # Update the scorer dictionaries
            if "decoder" in self.full_scorers:
                self.full_scorers["decoder"] = self.scorers["decoder"]
        else:
            self.return_hs = False

        logger.info("Initialized ContextualBeamSearch.")

    def init_hypothesis(
        self,
        encoder_output: torch.Tensor,
        context_data: Dict[str, Any] = None,
    ) -> List[Hypothesis]:
        """Initialize the hypothesis list."""
        init_states = {}
        init_scores = {}
        for key, scorer in self.scorers.items():
            init_states[key] = scorer.init_state(encoder_output)
            init_scores[key] = 0.0

        # Handling for OpenAI Whisper ASR
        primer = [self.sos] if self.hyp_primer is None else self.hyp_primer
        if (
            not self.use_ctc_only_decoding
            and isinstance(self.decoder, OpenAIWhisperDecoder)
            and context_data is not None
            and context_data["nlp_prompt_tensor"] is not None
        ):
            nlp_prompt_tensor = context_data["nlp_prompt_tensor"][0].tolist()
            logger.info(f"nlp_prompt_tensor: {nlp_prompt_tensor}")
            primer = [self.sop] + nlp_prompt_tensor + primer

        logger.info(f"Primer tokens: {primer}")

        initial_hypothesis = ContextualHypothesis(
            score=0.0,
            scores=init_scores,
            states=init_states,
            hs=[],
            yseq=torch.tensor(primer, device=encoder_output.device),
            context_score=0.0,
            context_scores=init_scores,
            context_states=init_states,
            context_hs=[],
            context_yseq=torch.tensor(primer, device=encoder_output.device),
            context_idxs=[],
        )
        return [initial_hypothesis]

    def forward(
        self,
        encoder_output: torch.Tensor,
        context_data: Dict[str, Any],
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_encoder_output: torch.Tensor = None,
    ) -> List[Hypothesis]:
        """Perform beam search with contextualization."""
        logger.info("Starting contextual beam search.")

        # Set length bounds
        input_feature = pre_encoder_output if pre_encoder_output is not None else encoder_output
        maxlen = self._get_max_length(maxlenratio, input_feature)
        minlen = self._get_min_length(minlenratio, input_feature)

        # Adjust maxlen for OpenAI Whisper model if necessary
        if (
            not self.use_ctc_only_decoding
            and isinstance(self.decoder, OpenAIWhisperDecoder)
        ):
            max_positional_embeddings = self.decoder.decoders.positional_embedding.shape[0]
            maxlen = min(maxlen, max_positional_embeddings - 4)

        logger.info(f"Input length: {input_feature.shape[0]}")
        logger.info(f"Max output length: {maxlen}")
        logger.info(f"Min output length: {minlen}")

        # Apply encoder contextualization
        encoder_output, context_hypotheses = self._apply_contextualizer_encoder(
            encoder_output, context_data
        )

        context_predictions = None

        # Here we can pass the retrieved context information to the decoder
        if context_hypotheses is not None:
            context_predictions = decode_topk_tokens(
                token_probs=context_hypotheses,
                vocabulary=context_data["context_list"],
                blank_index=0,
                top_k=100,
                threshold=0.01,
            )

            if context_data["nlp_prompt_tensor"] is not None:
                nlp_prompt, nlp_prompt_tensor = generate_prompt_from_hypotheses(
                    context_hypotheses=context_hypotheses,
                    contexts=context_data,
                    construct_prompt_labels_fn=self.context_sampler.construct_prompt_labels,
                    blank_index=0,
                    top_k=10,
                    threshold=0.5,
                )
                context_data.update(
                    {
                        "nlp_prompt": nlp_prompt,
                        "nlp_prompt_tensor": nlp_prompt_tensor,
                    }
                )

        # Initialize hypotheses
        running_hypotheses = self.init_hypothesis(
            encoder_output if pre_encoder_output is None else pre_encoder_output,
            context_data=context_data,
        )
        ended_hypotheses = []
        
        # Update context_data in the contextualized decoder scorer
        if "decoder" in self.scorers and isinstance(
            self.scorers["decoder"], ContextualizedDecoderScorer
        ):
            self.scorers["decoder"].context_data = context_data

        # Main beam search loop
        for i in range(maxlen):
            logger.debug(f"Beam search iteration {i}")
            best_hypotheses = self.search(
                running_hypotheses, encoder_output, pre_encoder_output
            )
            running_hypotheses = self.post_process(
                i, maxlen, minlen, maxlenratio, best_hypotheses, ended_hypotheses
            )

            if maxlenratio == 0.0 and end_detect(
                [hyp.asdict() for hyp in ended_hypotheses], i
            ):
                logger.info(f"End detected at iteration {i}")
                break

            if not running_hypotheses:
                logger.info("No hypotheses left. Ending decoding.")
                break
            else:
                logger.debug(f"Remaining hypotheses: {len(running_hypotheses)}")

        nbest_hypotheses = self._finalize_hypotheses(ended_hypotheses)

        # Handle case with no ended hypotheses
        if not nbest_hypotheses:
            logger.warning(
                "No N-best results found. Retrying with a smaller minlenratio."
            )
            if minlenratio < 0.1:
                return []
            else:
                return self.forward(
                    encoder_output, context_data, maxlenratio, max(0.0, minlenratio - 0.1)
                )

        # Log the best hypothesis
        self._log_best_hypothesis(nbest_hypotheses[0])

        # Add context predictions to the hypotheses
        contextual_nbest_hypotheses = self._add_context_predictions(
            nbest_hypotheses, context_predictions, context_data
        )

        return contextual_nbest_hypotheses

    def _get_max_length(self, maxlenratio: float, input_feature: torch.Tensor) -> int:
        """Calculate the maximum output length."""
        if maxlenratio == 0:
            return input_feature.shape[0]
        elif maxlenratio < 0:
            return -int(maxlenratio)
        else:
            return max(1, int(maxlenratio * input_feature.size(0)))

    def _get_min_length(self, minlenratio: float, input_feature: torch.Tensor) -> int:
        """Calculate the minimum output length."""
        if minlenratio < 0:
            return -int(minlenratio)
        else:
            return int(minlenratio * input_feature.size(0))

    def _apply_contextualizer_encoder(
        self, encoder_output: torch.Tensor, context_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[int, str, float]]]]:
        """Apply contextualizer to the encoder output."""
        logger.info("Applying encoder contextualization.")
        context_data = to_device(context_data, device=encoder_output.device)

        context_hypotheses = None
        contextualizer_type = self.contextualizer_config["contextualizer_type"]

        if contextualizer_type in CONTEXTUAL_RETRIEVER:
            context_hypotheses, encoder_output_proj = self.contextualizer(
                query=encoder_output.unsqueeze(0),
                query_ilens=None,
                context_subword=context_data["blist"],
                context_subword_ilens=context_data["ilens"],
                context_phone=context_data["blist_xphone"],
                context_phone_ilens=context_data["blist_xphone_ilens"],
                return_model_proj=True,
            )

            if self.use_ctc_only_decoding:
                encoder_output = encoder_output_proj

        elif contextualizer_type in CONTEXTUAL_ADAPTER_ENCODER:
            encoder_output = encoder_output.unsqueeze(0)
            encoder_bias_vector, context_hypotheses = self.contextualizer(
                model_embed=encoder_output,
                context_embed=context_data["blist"],
                context_xphone_idxs=context_data["blist_xphone_mean"],
                ilens=context_data["ilens"],
                return_atten=True,
            )
            context_hypotheses = torch.mean(context_hypotheses, dim=1)
            encoder_output = (encoder_output + encoder_bias_vector)

        return encoder_output, context_hypotheses

    def _add_context_predictions(
        self,
        nbest_hypotheses: List[Hypothesis],
        context_predictions: List[Tuple[int, str, float]],
        context_data: Dict[str, Any],
    ) -> List[Hypothesis]:
        """Add context predictions to the hypotheses."""
        if context_predictions is not None:
            pred_texts = " ".join([pred[1] for pred in context_predictions])
            context_yseq = self.context_sampler.prompt_text2int(pred_texts)
            context_score = [pred[2] for pred in context_predictions]
            context_yseq = [self.sos] + context_yseq + [self.eos]
            context_idxs = [
                context_data["context_list_idxs"][pred[0]] for pred in context_predictions
            ]
        else:
            context_yseq = [self.sos, self.eos]
            context_score = []
            context_idxs = []

        contextual_nbest_hypotheses = []
        for hyp in nbest_hypotheses:
            contextual_hyp = ContextualHypothesis(
                score=hyp.score,
                scores=hyp.scores,
                states=hyp.states,
                hs=hyp.hs,
                yseq=hyp.yseq,
                context_score=context_score,
                context_yseq=context_yseq,
                context_idxs=context_idxs,
            )
            contextual_nbest_hypotheses.append(contextual_hyp)
        return contextual_nbest_hypotheses

    def _finalize_hypotheses(self, ended_hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Finalize hypotheses after beam search."""
        if self.normalize_length:
            nbest_hypotheses = sorted(
                ended_hypotheses,
                key=lambda x: x.score / (len(x.yseq) - 1),
                reverse=True,
            )
        else:
            nbest_hypotheses = sorted(ended_hypotheses, key=lambda x: x.score, reverse=True)
        return nbest_hypotheses

    def _log_best_hypothesis(self, best_hypothesis: Hypothesis):
        """Log information about the best hypothesis."""
        for key, value in best_hypothesis.scores.items():
            logger.info(
                f"{value:6.2f} * {self.weights[key]:3} = {value * self.weights[key]:6.2f} for {key}"
            )
        logger.info(f"Total log probability: {best_hypothesis.score:.2f}")
        logger.info(
            f"Normalized log probability: {best_hypothesis.score / len(best_hypothesis.yseq):.2f}"
        )
        if self.token_list is not None:
            token_sequence = "".join(
                [self.token_list[idx] for idx in best_hypothesis.yseq[1:-1]]
            )
            logger.info(f"Best hypothesis: {token_sequence}\n")
        if len(best_hypothesis.yseq[1:-1]) == self._get_max_length(0, best_hypothesis.yseq):
            logger.warning(
                "Best hypothesis length equals max output length. "
                "Consider increasing maxlenratio."
            )


def beam_search(
    encoder_output: torch.Tensor,
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
    """Perform beam search with the provided scorers."""
    beam_search_instance = BeamSearch(
        scorers,
        weights,
        beam_size=beam_size,
        vocab_size=vocab_size,
        pre_beam_ratio=pre_beam_ratio,
        pre_beam_score_key=pre_beam_score_key,
        sos=sos,
        eos=eos,
        token_list=token_list,
    )
    result = beam_search_instance.forward(
        x=encoder_output, maxlenratio=maxlenratio, minlenratio=minlenratio
    )
    return [hyp.asdict() for hyp in result]
