import math
import torch
import random
import logging

from typing import Optional, Tuple

from espnet2.asr.contextualizer.component.context_encoder import (
    ContextEncoderBiLSTM,
    ContextEncoderTransformer
)

from espnet2.asr.contextualizer.component.attention_based_adapter import AttentionBasedAdapter

class ContextualAdapterPrototype(torch.nn.Module):
    def __init__(
        self,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        droup_out: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.encoder = ContextEncoderBiLSTM(
            hidden_size=context_embed_size,
            output_size=context_hidden_size,
            droup_out=droup_out,
        )
        self.adapter = AttentionBasedAdapter(
            model_hidden_size=model_hidden_size,
            context_hidden_size=context_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            droup_out=droup_out,
        )

    def forward_context_encoder(
        self,
        text_embed: torch.Tensor,
        ilens     : torch.Tensor = None,
    ):
        return self.encoder(text_embed, ilens)

    def forward_adapter(
        self,
        model_embed  : torch.Tensor,
        context_embed: torch.Tensor,
        mask         : torch.Tensor = None,
        return_atten : bool=False,
    ):
        return self.adapter(model_embed, context_embed, mask, return_atten)

    def forward(
        self,
        model_embed  : torch.Tensor,
        context_embed: torch.Tensor,
        ilens        : torch.Tensor = None,
        mask         : torch.Tensor = None,
        return_atten : bool=False,
    ):
        context_embed = self.forward_context_encoder(context_embed, ilens)
        output        = self.forward_adapter(
            model_embed=model_embed,
            context_embed=context_embed,
            mask=mask,
            return_atten=return_atten,
        )
        return output

class ContextualAdapterTransformer(ContextualAdapterPrototype):
    def __init__(
        self,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        droup_out: float = 0.1,
        num_blocks: int=2,
        linear_units: int=256,
        attention_heads: int=1,
        **kwargs
    ):
        super().__init__(
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            model_hidden_size=model_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            droup_out=droup_out,
        )
        self.encoder = ContextEncoderTransformer(
            hidden_size=context_embed_size,
            output_size=context_hidden_size,
            attention_heads=attention_heads,
            num_blocks=num_blocks,
            linear_units=linear_units,
            droup_out=droup_out,
        )

if __name__ == '__main__':
    B, U, T = 2, 3, 5

    vocab_size          = 5
    dropout             = 0.1
    encoder_hidden_size = 256
    context_hidden_size = 256
    joint_space_size    = 512
    proj_hidden_size    = 128
    context_embed_size  = 128

    context_adapter_prototype = ContextualAdapterPrototype(
        context_embed_size=context_embed_size,
        model_hidden_size=encoder_hidden_size,
        context_hidden_size=context_hidden_size,
        proj_hidden_size=proj_hidden_size,
        attndim=context_hidden_size,
        dropout=dropout,
    )

    model_out  = torch.rand(B, T, encoder_hidden_size)
    print(f'model_out: {model_out.shape}')
    text_embed = torch.rand(B, U, encoder_hidden_size)
    print(f'text_embed: {text_embed.shape}')
    
    bias = context_adapter_prototype(text_embed, model_out)
    print(f'contextual adapter prototype bias: {bias.shape}')

    ilens = torch.tensor([5, 3])

    context_adapter_transformer = ContextualAdapterTransformer(
        context_embed_size=context_embed_size,
        model_hidden_size=encoder_hidden_size,
        context_hidden_size=context_hidden_size,
        proj_hidden_size=proj_hidden_size,
        attndim=context_hidden_size,
        dropout=dropout,
    )

    bias, attn = context_adapter_transformer(text_embed, model_out, ilens, return_atten=True)
    print(f'contextual adapter transformer bias: {bias.shape}')
    print(f'contextual adapter attn: {attn.shape}')