import math
import torch
import random
import logging

from typing import Optional, Tuple

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

from espnet2.asr.contextualizer.component.context_encoder import (
    ContextEncoderBiLSTM,
    ContextEncoderTransformer,
    ContextEncoderXPhoneBiLSTM,
    ContextEncoderXPhone,
)

from espnet2.asr.contextualizer.component.similarity_based_retriever import (
    DotProductRetriever,
    Conv2DotProductRetriever,
)

class ContextualDotProductRetrieverPrototype(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        input_hidden_size: int,
        proj_hidden_size: int,
        num_blocks: int=1,
        drop_out: float = 0.1,
        use_value_norm: bool = False,
        padding_idx: int = -1,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.encoder = ContextEncoderBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_embed_size,
            output_size=context_hidden_size,
            num_blocks=num_blocks,
            drop_out=drop_out,
            padding_idx=padding_idx,
        )
        self.retriever = DotProductRetriever(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )

    def forward_context_encoder(
        self,
        text_embed: torch.Tensor,
        ilens     : torch.Tensor,
    ):
        return self.encoder(text_embed, ilens)

    def forward_retriever(
        self,
        model_embed        : torch.Tensor,
        context_embed      : torch.Tensor,
        context_embed_value: torch.Tensor = None,
        mask               : torch.Tensor = None,
        return_atten       : bool=False,
    ):
        return self.retriever(
            model_embed, 
            context_embed, 
        )

    def forward(
        self,
        model_embed  : torch.Tensor,
        context_embed: torch.Tensor,
        ilens        : torch.Tensor = None,
        **kwargs
    ):
        context_embed_mean, context_embed, ilens = self.forward_context_encoder(
            context_embed, 
            ilens
        )
        output = self.forward_retriever(
            model_embed=model_embed,
            context_embed=context_embed_mean,
        )
        return output

class ContextualConv2XPhoneDotProductRetriever(
    ContextualDotProductRetrieverPrototype
):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        input_hidden_size: int,
        proj_hidden_size: int,
        num_blocks: int=1,
        drop_out: float = 0.1,
        use_value_norm: bool = False,
        padding_idx: int = -1,
        temperature: float = 1.0,
        xphone_hidden_size: int = 768,
        merge_conv_kernel: int = 3,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            num_blocks=num_blocks,
            drop_out=drop_out,
            use_value_norm=use_value_norm,
            padding_idx=padding_idx,
            temperature=temperature,
            **kwargs
        )
        self.encoder = ContextEncoderXPhoneBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_hidden_size,
            output_size=input_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
        )
        self.retriever = Conv2DotProductRetriever(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )
    
    def forward_context_encoder(
        self,
        text_embed   : torch.Tensor,
        xphone_embed : torch.Tensor,
        ilens        : torch.Tensor,
    ):
        return self.encoder(text_embed, xphone_embed, ilens)

    def forward(
        self,
        model_embed         : torch.Tensor,
        context_embed       : torch.Tensor,
        context_xphone_embed: torch.Tensor,
        ilens               : torch.Tensor = None,
        **kwargs
    ):
        context_embed_merged, context_embed_mean, ilens = self.forward_context_encoder(
            context_embed,
            context_xphone_embed,
            ilens
        )
        output = self.forward_retriever(
            model_embed=model_embed,
            context_embed=context_embed_mean,
        )
        return output