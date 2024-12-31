"""
Contextualized Beam Search for ASR

This module extends the beam search algorithm by integrating contextual biasing mechanisms, improving ASR performance on domain-specific vocabulary. The contextualization affects both the encoder and decoder, dynamically influencing the scoring and token selection processes during decoding.

Key Features:
1. **Contextual Hypotheses Tracking:**
   - Introduces `ContextualHypothesis` to maintain contextual predictions along with standard ASR hypotheses.

2. **Contextualized Decoder Scorer:**
   - Wraps the decoder with contextual scoring logic using `ContextualizedDecoderScorer`.
   - Dynamically biases predictions based on context data and hidden state manipulation.

3. **Encoder and Decoder Contextualization:**
   - Applies contextualization at both the encoder and decoder stages through the adapter and retriever models.
   - Uses retrieval mechanisms (e.g., top-k tokens) to influence token probabilities during inference.

4. **Handling Whisper and NLP Prompts:**
   - Seamlessly integrates with OpenAI Whisper decoder and manages NLP prompts for initializing context-aware decoding.

5. **Flexible Beam Search Loop:**
   - Modifies the beam search loop to accommodate contextual information and generate hypotheses with added context predictions.
   - Includes token generation, scoring, and context adaptation for each search iteration.

6. **Support for Multiple Contextualization Strategies:**
   - Adapts to various contextual mechanisms, including retriever and adapter models, through `CONTEXTUAL_RETRIEVER` and `CONTEXTUAL_ADAPTER` options.

7. **Prompt Generation and Contextual Predictions:**
   - Generates NLP-based prompts from retrieved hypotheses and updates decoding states with relevant contextual predictions.

8. **Integration with Hard Negative Mining:**
   - Enables retrieval-based contextual scoring to incorporate hard negatives, improving robustness.
"""

import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from torch.nn.utils.rnn import pad_sequence

from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import ScorerInterface, PartialScorerInterface
from espnet2.asr.contextualizer import (
    CONTEXTUAL_ADAPTER_DECODER,
    CONTEXTUAL_ADAPTER_ENCODER,
    CONTEXTUAL_RETRIEVER,
    CONTEXTUAL_PROTOTYPE,
)
from espnet2.asr.contextualizer.func.contextual_retriever_func import (
    decode_topk_tokens,
    generate_prompt_from_hypotheses,
    select_max_predictions,
)
from espnet2.torch_utils.device_funcs import to_device
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder

logger = logging.getLogger(__name__)

def trie_search(trie, context_ints):
    if trie is None:
        return True
    no_context = [{}, [0]]
    now  = trie
    for y in context_ints:
        if y in now[0]:
            now = now[0][y]
        elif y != 220:
            now = no_context
        elif y == 220:
            now = trie
    context_node_idxs = now[1]
    logging.info(f'context_node_idxs: {len(context_node_idxs)}')
    return True if len(context_node_idxs) == 1 else False

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
        context_predictions: List,
        state: Any,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Any]:
        """Score the hypothesis with decoder contextualization."""
        # Force return_hs=True to get the hidden state
        score, hidden_state, state = self.decoder_scorer.score(yseq, state, x, return_hs=True, *args, **kwargs)

        if hidden_state is None:
            # If hidden_state is not provided, we cannot apply contextualization
            logger.warning("Hidden state not available for contextualization.")
            return score, state

        # Apply decoder contextualization if enabled
        logging.info(f'_' * 30)
        logging.info(f'yseq: {yseq}')
        score, context_hypotheses = self._apply_contextualizer_decoder(yseq, hidden_state, self.context_data)

        if context_hypotheses is not None:
            context_predictions_prior = self.context_data.get("context_predictions_prior", None)
            context_prediction = decode_topk_tokens(
                token_probs=context_hypotheses,
                vocabulary=self.context_data["context_list"],
                blank_index=0,
                top_k=1,
                threshold=0.01,
                priors=context_predictions_prior,
                combine_weight=0.5,
                retrieve_phrase=False,
            )
            logging.info(f'context_prediction: {[idx for idx, _, _ in context_prediction]}')
            context_predictions.extend(context_prediction)

        return score, state

    def _apply_contextualizer_decoder(
        self, yseq: torch.Tensor, decoder_output: torch.Tensor, context_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[int, str, float]]]]:
        """Apply contextualizer to the decoder output."""
        
        context_hypotheses = None
        # Apply decoder contextualization if enabled
        if self.contextualizer_config["contextualizer_type"] in CONTEXTUAL_ADAPTER_DECODER:
            ilens = context_data["ilens"]
            blist = context_data["blist"][:, :max(ilens)]

            decoder_embedding = decoder_output.reshape(1, 1, -1)  # Shape: (1, 1, D,)
            # Apply decoder contextualizer
            decoder_bias_vector, context_hypotheses = self.contextualizer(
                model_embed=decoder_embedding,
                context_embed=blist,
                ilens=ilens,
                return_atten=True,
            )
            # Mean across attention heads
            context_hypotheses = torch.mean(context_hypotheses, dim=1)
            # Bias the hidden state
            decoder_output = decoder_output + decoder_bias_vector
            decoder_output = torch.softmax(self.decoder_scorer.output_layer(decoder_output), dim=-1).reshape(-1)
            copy_style = False
            if copy_style:
                decoder_output = self._copy_context_decode_style(
                    model_probs=decoder_output,
                    context_probs=context_hypotheses.reshape(-1)[1:],
                    no_context_probs=context_hypotheses.reshape(-1)[:1],
                    threshold=0.9 if yseq[-1] == 220 else 1.0,
                    # threshold=0.8,
                )
            decoder_output = torch.log(decoder_output)
        elif self.contextualizer_config["contextualizer_type"] in CONTEXTUAL_PROTOTYPE:
            decoder_embedding    = decoder_output.reshape(1, 1, -1)
            blist_utterance_wise = context_data["blist_utterance_wise"][0]
            ilens_utterance_wise = context_data["ilens_utterance_wise"][0]

            decoder_bias_vector, decoder_attention = self.contextualizer.forward_at_decode(
                model_embed=decoder_embedding,
                context_embed=blist_utterance_wise,
                ilens=ilens_utterance_wise,
                return_atten=True,
            )
            # Mean across attention heads
            context_hypotheses = torch.mean(decoder_attention, dim=1)
            # Bias the hidden state
            # decoder_output = decoder_output + decoder_bias_vector
            # Adjust the score
            decoder_output = torch.log_softmax(self.decoder_scorer.output_layer(decoder_output), dim=-1)
            decoder_output = decoder_output.reshape(-1)
        return decoder_output, context_hypotheses
    
    def _copy_context_decode_style(self, model_probs, context_probs, no_context_probs, threshold):
        logging.info(f'torch.max(context_probs): {torch.max(context_probs)}')
        if torch.max(context_probs) < threshold:
            context_probs = torch.zeros_like(context_probs)
            no_context_probs = torch.ones_like(no_context_probs)

        model_probs = model_probs * no_context_probs
        probs = torch.cat([model_probs, context_probs], dim=-1)
        return probs

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
    ) -> List[ContextualHypothesis]:
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
        # Here we can pass the retrieved context information to the decoder
        context_predictions_encoder = self._update_contexts(
            contexts=context_data, 
            contexts_hypotheses=context_hypotheses
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
        context_predictions_decoder = []
        for i in range(maxlen):
            logger.debug(f"Beam search iteration {i}")
            best_hypotheses = self.search(
                running_hypotheses, encoder_output, pre_encoder_output, context_data, context_predictions_decoder
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

        # TODO: Change to save both
        # if len(context_predictions_decoder) > 0:
        context_predictions = select_max_predictions(context_predictions_decoder)
        # else:
        #     context_predictions = select_max_predictions(context_predictions_encoder)
        # Add context predictions to the hypotheses
        contextual_nbest_hypotheses = self._add_context_predictions(
            nbest_hypotheses, context_predictions, context_data
        )

        return contextual_nbest_hypotheses

    @staticmethod
    def append_token(
        xs: torch.Tensor, 
        x: int,
        n_vocab: int = None,
        context_vocab: List[int] = None, 
        trie: object = None,
    ) -> torch.Tensor:
        """Append new token to prefix tokens.

        Args:
            xs (torch.Tensor): The prefix token
            x (int): The new token to append

        Returns:
            torch.Tensor: New tensor contains: xs + [x] with xs.dtype and xs.device

        """
        if (n_vocab is None) or (x < n_vocab):
            logging.info(f'not copy!')
            x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        else:
            logging.info(f'doing copy!')
            x = x - n_vocab
            end_phrase = [220] if trie_search(trie, context_vocab[x]) else []
            logging.info(f'end_phrase: {end_phrase}')
            x = torch.tensor(context_vocab[x] + end_phrase, dtype=xs.dtype, device=xs.device)

            # roll back to last blank (space symbol)
            blank_index = (xs == 220).nonzero(as_tuple=True)[0]
            if len(blank_index) > 0:
                blank_index = blank_index[-1]
                logging.info(f'rolling back from {xs.shape[-1]} to {blank_index + 1}')

                xs = xs[:blank_index + 1] 
        return torch.cat((xs, x))

    def search(
        self,
        running_hyps: List[Hypothesis],
        x: torch.Tensor,
        pre_x: torch.Tensor = None,
        context_data: Dict[str, Any] = {},
        context_predictions: List = [],
    ) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        best_hyps = []
        # context_len = len(context_data['blist']) - 1
        context_len = 0
        part_ids = torch.arange(self.n_vocab + context_len, device=x.device)  # no pre-beam
        for hyp in running_hyps:
            # scoring
            weighted_scores = torch.zeros(self.n_vocab + context_len, dtype=x.dtype, device=x.device)
            if self.return_hs:
                hs, scores, states = self.score_full(hyp, x, pre_x=pre_x, context_predictions=context_predictions)
            else:
                scores, states = self.score_full(hyp, x, pre_x=pre_x, context_predictions=context_predictions)
            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]
            # partial scoring
            if self.do_pre_beam:
                pre_beam_scores = (
                    weighted_scores
                    if self.pre_beam_score_key == "full"
                    else scores[self.pre_beam_score_key]
                )
                part_ids = torch.topk(pre_beam_scores, self.pre_beam_size)[1]
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            # add previous hyp score
            weighted_scores += hyp.score

            # update hyps
            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                # will be (2 x beam at most)
                if self.return_hs:
                    new_hs = hyp.hs + [hs.squeeze(0)]
                else:
                    new_hs = []
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(
                            hyp.yseq, 
                            j,
                            self.n_vocab,
                            context_data['context_list_ints'][1:],
                            context_data['trie'],
                        ),
                        scores=self.merge_scores(
                            hyp.scores, scores, j, part_scores, part_j
                        ),
                        states=self.merge_states(states, part_states, part_j),
                        hs=new_hs,
                    )
                )

            # sort and prune 2 x beam -> beam
            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps

    def score_full(
        self, hyp: Hypothesis, x: torch.Tensor, pre_x: torch.Tensor = None, context_predictions: List = [],
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
                    hyp.yseq, hyp.states[k], context_predictions, x, return_hs=self.return_hs
                )
            elif pre_x is not None:
                scores[k], states[k] = d.score(hyp.yseq, context_predictions, hyp.states[k], x, pre_x)
            else:
                scores[k], states[k] = d.score(hyp.yseq, context_predictions, hyp.states[k], x)

        if self.return_hs:
            return hs, scores, states
        return scores, states

    def _update_contexts(self, contexts, contexts_hypotheses):
        context_predictions = None
        if contexts_hypotheses is None:
            return context_predictions
        
        context_predictions = decode_topk_tokens(
            token_probs=contexts_hypotheses,
            vocabulary=contexts["context_list"],
            blank_index=0,
            top_k=self.contextualizer_config.get('retrieve_top_k', 100),
            threshold=self.contextualizer_config.get('retrieve_threshold', 0.01),
        )
        # Update utterance wise contexts
        if self.contextualizer_config["contextualizer_type"] in CONTEXTUAL_PROTOTYPE:
            prediction_context_idxs_lists = [
                contexts['context_list_idxs'][idx] for idx, _, _ in context_predictions
            ]
        else:
            prediction_context_idxs_lists = contexts['context_list_idxs'][1:]
        (
            utterance_wise_sub_context_lists,
            utterance_wise_sub_context_ints_tensors,
            utterance_wise_sub_context_ints_tensor_lens
        ) = self.context_sampler.construct_utterance_wise_context(
            [prediction_context_idxs_lists],
        )
        context_predictions_prior = [prior for _, _, prior in context_predictions]
        contexts.update(
            {
                "context_list": utterance_wise_sub_context_lists[0],
                "blist_utterance_wise": utterance_wise_sub_context_ints_tensors,
                "ilens_utterance_wise": utterance_wise_sub_context_ints_tensor_lens,
                "context_predictions_prior": context_predictions_prior,
            }
        )
        # Update the context prompt
        if contexts["nlp_prompt_tensor"] is not None:
            nlp_prompt, nlp_prompt_tensor = generate_prompt_from_hypotheses(
                contexts=contexts,
                context_hypotheses=contexts_hypotheses,
                construct_prompt_labels_fn=self.context_sampler.construct_prompt_labels,
                top_k=self.context_sampler.max_utterance_disrupt_context,
                blank_index=0,
                threshold=0.5,
            )
            contexts.update(
                {
                    "nlp_prompt": nlp_prompt,
                    "nlp_prompt_tensor": nlp_prompt_tensor,
                }
            )
        return context_predictions

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
                encoder_output = encoder_output_proj.squeeze(0)

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

        elif contextualizer_type in CONTEXTUAL_PROTOTYPE:
            context_hypotheses, encoder_output_proj = self.contextualizer.forward_at_encode(
                query=encoder_output.unsqueeze(0),
                query_ilens=None,
                context_subword=context_data["blist"],
                context_subword_ilens=context_data["ilens"],
                context_phone=context_data["blist_xphone"],
                context_phone_ilens=context_data["blist_xphone_ilens"],
                return_model_proj=True,
            )
            if self.use_ctc_only_decoding:
                encoder_output = encoder_output_proj.squeeze(0)
        return encoder_output, context_hypotheses

    def _add_context_predictions(
        self,
        nbest_hypotheses: List[Hypothesis],
        context_predictions: List[Tuple[int, str, float]],
        context_data: Dict[str, Any],
    ) -> List[ContextualHypothesis]:
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
