import torch

from espnet2.asr.contextualizer.contextual_retriever import DotProductXPhoneContextualRetriever
from espnet2.asr.contextualizer.contextual_adapter   import ContextualAdapterTransformer

class RRAdapter(torch.nn.Module):
    def __init__(
        self,
        # retriever config
        retriever_query_input_dim     : int,
        retriever_context_input_dim   : int,
        retriever_proj_dim            : int,
        retriever_interaction_proj_dim: int,
        # adapter config
        adapter_context_embed_size: int,
        adapter_context_hidden_size: int,
        adapter_model_hidden_size: int,
        adapter_attndim: int,
        adapter_proj_hidden_size: int,
        adapter_num_blocks: int=2,
        adapter_linear_units: int=256,
        adapter_context_attention_heads: int=4,
        adapter_attention_heads: int=1,
        # global config
        vocab_size: int=6000,
        original_vocab_size: int=6000,
        dropout: float = 0.1,
        padding_idx: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.retriever = DotProductXPhoneContextualRetriever(
            vocab_size=vocab_size,
            query_input_dim=retriever_query_input_dim,
            context_input_dim=retriever_context_input_dim,
            proj_dim=retriever_proj_dim,
            interaction_proj_dim=retriever_interaction_proj_dim,
            dropout=dropout,
            pad_token_value=padding_idx,
        )
        self.adapter = ContextualAdapterTransformer(
            vocab_size=original_vocab_size,
            context_embed_size=adapter_context_embed_size,
            context_hidden_size=adapter_context_hidden_size,
            model_hidden_size=adapter_model_hidden_size,
            attndim=adapter_attndim,
            proj_hidden_size=adapter_proj_hidden_size,
            drop_out=dropout,
            num_blocks=adapter_num_blocks,
            linear_units=adapter_linear_units,
            context_attention_heads=adapter_context_attention_heads,
            adapter_attention_heads=adapter_attention_heads,
            padding_idx=padding_idx,
        )

        self.forward_at_encode = self.retriever.forward
        self.forward_at_decode = self.adapter.forward