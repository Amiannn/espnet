import math
import torch
import random
import logging

from typing import Optional, Tuple

from espnet2.asr.prototype.component.context_encoder         import ContextEncoder
from espnet2.asr.prototype.component.attention_based_adapter import AttentionBasedAdapter

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
        self.encoder = ContextEncoder(
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
    ):
        return self.encoder(text_embed)

    def forward_adapter(
        self,
        model_embed  : torch.Tensor,
        context_embed: torch.Tensor,
        mask         : torch.Tensor = None,
    ):
        return self.adapter(model_embed, context_embed)

    def forward(
        self,
        model_embed  : torch.Tensor,
        context_embed: torch.Tensor,
        mask         : torch.Tensor = None,
    ):
        context_embed = self.forward_context_encoder(context_embed)
        output        = self.forward_adapter(model_embed, context_embed)
        return output

if __name__ == '__main__':
    B, U, T = 2, 3, 5

    vocab_size          = 5
    dropout             = 0.1
    encoder_hidden_size = 256
    context_hidden_size = 256
    joint_space_size    = 512
    proj_hidden_size    = 128
    context_embed_size  = 128

    context_adapter = ContextualAdapterPrototype(
        context_embed_size=context_embed_size,
        model_hidden_size=encoder_hidden_size,
        context_hidden_size=context_hidden_size,
        proj_hidden_size=proj_hidden_size,
        dropout=dropout,
    )

    model_out  = torch.rand(B, T, encoder_hidden_size)
    print(f'model_out: {model_out.shape}')
    text_embed = torch.rand(B, U, encoder_hidden_size)
    print(f'text_embed: {text_embed.shape}')
    
    bias = context_adapter(text_embed, model_out)
    print(f'bias: {bias.shape}')