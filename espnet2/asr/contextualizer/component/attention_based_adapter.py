import math
import torch
import random
import logging

from typing import Optional, Tuple

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.attention  import (
    CustomMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)

class AttentionBasedAdapter(torch.nn.Module):
    def __init__(
        self,
        attention_heads: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        use_value_norm: bool = False,
        atten_temperature: float = 1.0,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__()
        self.attndim         = attndim
        self.attention_heads = attention_heads
        self.attention_layer = CustomMultiHeadedAttention(
            attention_heads, attndim, drop_out
        )
        self.temperature    = atten_temperature
        self.proj           = torch.nn.Linear(self.attndim, proj_hidden_size)
        self.norm_before_x1 = LayerNorm(attndim)
        self.norm_before_x2 = LayerNorm(attndim)
        self.norm_after     = LayerNorm(attndim)

        if use_value_norm:
            self.norm_before_x3 = LayerNorm(attndim)
       
        self.use_local_attn_conv = use_local_attn_conv
        if self.use_local_attn_conv:
            self.local_attn_conv_1x3 = torch.nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(3, 1), 
                stride=(2, 1),
                padding=(1, 0),
            )
        
    def forward_local_attn_conv(self, query):
        query = query.unsqueeze(1)
        query = self.local_attn_conv_1x3(query)
        return query.squeeze(1)

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
        model_embed   = model_embed.reshape(1, B*T, D)
        model_embed   = self.norm_before_x1(model_embed)
        
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
            temperature=self.temperature,
        )
        out = out.reshape(B, T, D)
        out = self.norm_after(out)
        out = self.proj(out)

        if return_atten:
            attn = self.attention_layer.attn
            attn = attn.reshape(B, -1, T, C)
            if self.use_local_attn_conv:
                _, H, _, _ = attn.shape
                attn = attn.reshape(B * H, T, C)
                attn = self.forward_local_attn_conv(attn)
                attn = attn.reshape(B, H, -1, C)
                # TODO: without mask, this may cause some problem
                attn = torch.softmax(attn / self.temperature, dim=-1)
            return out, attn
        return out
    
class AttentionBasedLightAdapter(torch.nn.Module):
    def __init__(
        self,
        attention_heads: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        use_value_norm: bool = False,
        atten_temperature: float = 1.0,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__()
        self.attndim         = attndim
        self.attention_heads = attention_heads
        self.attention_layer = CustomMultiHeadedAttention(
            attention_heads, attndim, drop_out
        )
        self.temperature    = atten_temperature
        self.norm_before_x1 = LayerNorm(attndim)
        self.norm_before_x2 = LayerNorm(attndim)

        if use_value_norm:
            self.norm_before_x3 = LayerNorm(attndim)
       
    def forward(
        self,
        model_embed,
        context_embed,
        context_embed_value=None,
        mask=None,
        return_atten=False,
    ):  
        B, T, D       = model_embed.shape
        model_embed   = model_embed.reshape(1, B*T, D)
        model_embed   = self.norm_before_x1(model_embed)
        
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
            temperature=self.temperature,
        )
        out = out.reshape(B, T, D)
        if return_atten:
            attn = self.attention_layer.attn
            attn = attn.reshape(B, -1, T, C)
            return out, attn
        return out