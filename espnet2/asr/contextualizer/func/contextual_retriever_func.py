import torch
import logging
from typing import List, Tuple, Callable, Any, Dict
from itertools import groupby

from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

def decode_ctc_predictions(
    ctc_probs: torch.Tensor,
    vocabulary: List[str],
    blank_index: int = 0,
    threshold: float = 0.0,
    **kwargs,
) -> List[List[List[Any]]]:
    """
    Decodes CTC output probabilities to retrieve token sequences.

    Args:
        ctc_probs (torch.Tensor): The CTC probability tensor of shape (batch_size, seq_length, num_classes).
        vocabulary (List[str]): List of token strings corresponding to class indices.
        blank_index (int, optional): Index of the blank token in CTC. Defaults to 0.
        threshold (float, optional): Probability threshold for token inclusion. Defaults to 0.0.

    Returns:
        List[List[List[Any]]]: Decoded token sequences for each batch.
    """
    predicted_indices = ctc_probs.argmax(dim=-1).cpu()
    predicted_probs = ctc_probs.max(dim=-1).values.cpu()
    batch_sequences = []

    for b in range(predicted_indices.shape[0]):
        sequence = []
        prev_index = None
        for t in range(predicted_indices[b].shape[0]):
            idx = int(predicted_indices[b][t])
            token = vocabulary[idx]
            if idx != blank_index and idx != prev_index and predicted_probs[b][t] > threshold:
                sequence.append([t, token, predicted_probs[b][t].item()])
                prev_index = idx
            else:
                prev_index = idx if idx != blank_index else prev_index
        batch_sequences.append(sequence)
    return batch_sequences

def decode_topk_tokens(
    token_probs: torch.Tensor,
    vocabulary: List[str],
    blank_index: int = 0,
    top_k: int = 10,
    threshold: float = 0.6
) -> List[List[Any]]:
    """
    Decodes the top-k tokens from probabilities with optional thresholding.

    Args:
        token_probs (torch.Tensor): The probability tensor of shape (batch_size, seq_length, num_classes).
        vocabulary (List[str]): List of token strings corresponding to class indices.
        blank_index (int, optional): Index of the blank token. Defaults to 0.
        top_k (int, optional): Number of top tokens to consider. Defaults to 10.
        threshold (float, optional): Probability threshold for token inclusion. Defaults to 0.6.

    Returns:
        List[List[Any]]: List of decoded tokens with their positions and scores.
    """
    max_probs_info = token_probs.cpu().max(dim=-1)
    max_probs = max_probs_info.values[0]
    max_indices = max_probs_info.indices[0]

    vocab_size = len(vocabulary)
    average_probs = torch.zeros(vocab_size)
    token_counts = torch.zeros(vocab_size)

    average_probs.scatter_add_(0, max_indices, max_probs)
    token_counts.scatter_add_(0, max_indices, torch.ones_like(max_probs))

    # Avoid division by zero
    token_counts = token_counts.masked_fill(token_counts == 0, 1)
    average_probs = average_probs / token_counts
    average_probs[blank_index] = 0

    # Get indices sorted by average_probs values in descending order
    sorted_indices = torch.argsort(average_probs, descending=True)
    # Apply threshold filtering
    sorted_indices = sorted_indices[average_probs[sorted_indices] >= threshold][:top_k]
    topk_tokens = []
    for idx in sorted_indices:
        idx_int = idx.item()
        if idx_int == blank_index or idx_int == 0:
            continue
        token = vocabulary[idx_int]
        score = average_probs[idx].item()
        topk_tokens.append([idx_int, token, score])
    # Keep the position
    # ordered_topk_tokens = []
    # seen_indices = set()
    # predicted_indices = token_probs[0].argmax(dim=-1).cpu()
    # for t in range(predicted_indices.shape[0]):
    #     idx = int(predicted_indices[t])
    #     if idx != blank_index and idx not in seen_indices:
    #         for token_info in topk_tokens:
    #             if token_info[0] == idx:
    #                 ordered_topk_tokens.append(token_info)
    #                 seen_indices.add(idx)
    #                 break
    # return ordered_topk_tokens
    return topk_tokens

def generate_prompt_from_hypotheses(
    context_hypotheses: List[torch.Tensor],
    contexts: Dict[str, Any],
    construct_prompt_labels_fn: Callable[..., Tuple[Any, ...]],
    blank_index: int = 0,
    top_k: int = 10,
    threshold: float = 0.5,
) -> Tuple[Any, Any]:
    """
    Generates a prompt based on context hypotheses and constructs prompt labels.

    Args:
        context_hypotheses (List[torch.Tensor]): List of context hypothesis tensors.
        contexts (Dict[str, Any]): Dictionary containing context information.
        construct_prompt_labels_fn (Callable): Function to construct prompt labels.
        blank_index (int, optional): Index of the blank token. Defaults to 0.
        top_k (int, optional): Number of top tokens to consider. Defaults to 10.
        threshold (float, optional): Probability threshold for token inclusion. Defaults to 0.5.

    Returns:
        Tuple[Any, Any]: The NLP prompt and its tensor representation.
    """
    decoded_contexts = []
    for hypothesis in context_hypotheses:
        decoded_tokens = decode_topk_tokens(
            hypothesis.unsqueeze(0),
            contexts['context_list_ints'],
            blank_index=blank_index,
            top_k=top_k,
            threshold=threshold,
        )
        context_sequence = [
            [contexts['context_list_idxs'][idx], pos, score]
            for idx, pos, score in decoded_tokens
        ]
        decoded_contexts.append(context_sequence)

    nlp_prompt, nlp_prompt_tensor, _, _ = construct_prompt_labels_fn(
        decoded_contexts, has_confidence=True
    )
    return nlp_prompt, nlp_prompt_tensor

if __name__ == "__main__":
    # Sample vocabulary (including a blank token at index 0)
    vocabulary = ['<blank>', 'a', 'b', 'c', 'd']
    blank_index = 0  # Index of the blank token in the vocabulary

    # Sample CTC probabilities tensor
    ctc_probs = torch.tensor([
        [
            [0.1, 0.7, 0.2, 0.0, 0.0],  # t=0, predicts 'a'
            [0.1, 0.0, 0.8, 0.1, 0.0],  # t=1, predicts 'b'
            [0.1, 0.0, 0.0, 0.6, 0.3],  # t=2, predicts 'c'
            [0.1, 0.0, 0.0, 0.0, 0.9],  # t=3, predicts 'd'
            [0.9, 0.0, 0.0, 0.0, 0.0],  # t=4, predicts '<blank>'
        ]
    ])  # Shape: (1, 5, 5)

    # Test decode_ctc_predictions function
    decoded_sequences = decode_ctc_predictions(
        ctc_probs=ctc_probs,
        vocabulary=vocabulary,
        blank_index=blank_index,
        threshold=0.0
    )

    print("Decoded sequences from decode_ctc_predictions:")
    for batch_idx, sequence in enumerate(decoded_sequences):
        print(f"Batch {batch_idx}:")
        for t, token, prob in sequence:
            print(f"  Time {t}: Token '{token}', Probability {prob}")

    # Test decode_topk_tokens function
    topk_tokens = decode_topk_tokens(
        token_probs=ctc_probs,
        vocabulary=vocabulary,
        blank_index=blank_index,
        top_k=3,
        threshold=0.5
    )

    print("\nTop-k tokens from decode_topk_tokens:")
    for idx, token, avg_prob in topk_tokens:
        print(f"Token '{token}' (Index {idx}): Average Probability {avg_prob}")

    # Sample context hypotheses
    context_hypotheses = [ctc_probs[0]]  # List of tensors, each of shape (seq_length, num_classes)

    # Sample contexts dictionary
    contexts = {
        'context_list_ints': vocabulary, 
        'context_list_idxs': list(range(0, len(vocabulary)))  # [1, 2, 3, 4]
    }

    # Updated mock construct_prompt_labels_fn function
    def construct_prompt_labels_fn(decoded_contexts, has_confidence):
        """
        Mock function to construct prompt labels.
        Converts the decoded contexts to numerical indices for tensor creation.
        """
        nlp_prompt = decoded_contexts
        nlp_prompt_tensor = []

        for context_sequence in decoded_contexts:
            sequence_tensor = []
            for context_index, position, score in context_sequence:
                # context_index is already numerical (e.g., 1, 2, 3)
                # position and score are numerical as well
                sequence_tensor.append([context_index, position, score])
            nlp_prompt_tensor.append(sequence_tensor)

        # Convert to tensor
        nlp_prompt_tensor = torch.tensor([4, 5, 3], dtype=torch.float32)
        return nlp_prompt, nlp_prompt_tensor, None, None

    # Test generate_prompt_from_hypotheses function
    nlp_prompt, nlp_prompt_tensor = generate_prompt_from_hypotheses(
        context_hypotheses=context_hypotheses,
        contexts=contexts,
        construct_prompt_labels_fn=construct_prompt_labels_fn,
        blank_index=blank_index,
        top_k=3,
        threshold=0.5
    )

    print("\nGenerated prompt and tensor from generate_prompt_from_hypotheses:")
    print("NLP Prompt:")
    print(nlp_prompt)
    print("NLP Prompt Tensor:")
    print(nlp_prompt_tensor)
