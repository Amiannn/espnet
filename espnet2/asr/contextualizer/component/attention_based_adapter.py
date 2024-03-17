import math
import torch
import random
import logging

from typing import Optional, Tuple

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.attention  import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)

class AttentionBasedAdapter(torch.nn.Module):
    def __init__(
        self,
        attention_heads: int,
        attndim: int,
        proj_hidden_size: int,
        droup_out: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.attndim         = attndim
        self.attention_heads = attention_heads
        self.attention_layer = MultiHeadedAttention(
            attention_heads, attndim, droup_out
        )
        self.proj  = torch.nn.Linear(self.attndim, proj_hidden_size)
        self.norm_before_x1 = LayerNorm(attndim)
        self.norm_before_x2 = LayerNorm(attndim)
        self.norm_before_x3 = LayerNorm(attndim)
        self.norm_after     = LayerNorm(attndim)

    def forward(
        self,
        model_embed,
        context_embed,
        context_embed_value=None,
        mask=None,
        return_atten=False,
    ):  
        # may cause some problems (softmax cross utterance)...
        B, T, D       = model_embed.shape
        model_embed   = self.norm_before_x1(model_embed)
        model_embed   = model_embed.reshape(1, B*T, D)

        C, D          = context_embed.shape
        context_embed = context_embed.unsqueeze(0)
        context_embed = self.norm_before_x2(context_embed)

        if context_embed_value is None:
            context_embed_value = context_embed
        else:
            context_embed_value = context_embed_value.unsqueeze(0)
            context_embed_value = self.norm_before_x3(context_embed_value)

        out = self.attention_layer(
            query=model_embed, 
            key=context_embed, 
            value=context_embed_value,
            mask=mask,
        )
        out = out.reshape(B, T, D)
        out = self.norm_after(out)
        out = self.proj(out)

        if return_atten:
            atten = self.attention_layer.attn
            atten = atten.reshape(B, -1, T, C)
            return out, atten
        return out
