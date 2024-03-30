import math
import torch
import random
import logging

from typing import Optional, Tuple

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.attention  import (
    CustomMultiHeadedAttention,
    ColbertAttention,
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
        droup_out: float = 0.1,
        use_value_norm: bool = False,
        atten_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.attndim         = attndim
        self.attention_heads = attention_heads
        self.attention_layer = CustomMultiHeadedAttention(
            attention_heads, attndim, droup_out
        )
        self.temperature    = atten_temperature
        self.proj           = torch.nn.Linear(self.attndim, proj_hidden_size)
        self.norm_before_x1 = LayerNorm(attndim)
        self.norm_before_x2 = LayerNorm(attndim)
        self.norm_after     = LayerNorm(attndim)

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
        )
        out = out.reshape(B, T, D)
        out = self.norm_after(out)
        out = self.proj(out)

        if return_atten:
            atten = torch.softmax(
                self.attention_layer.scores * self.temperature, 
                dim=-1
            )
            if mask is not None:
                mask  = mask.unsqueeze(1).eq(0)
                atten = atten.masked_fill(mask, 0.0)
            atten = atten.reshape(B, -1, T, C)
            return out, atten
        return out

class ConvAttentionAdapter(AttentionBasedAdapter):
    def __init__(
        self,
        attention_heads: int,
        attndim: int,
        proj_hidden_size: int,
        droup_out: float = 0.1,
        use_value_norm: bool = False,
        atten_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(
            attention_heads=attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            droup_out=droup_out,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            **kwargs,
        )
        self.query_conv = torch.nn.Conv1d(
            in_channels=attndim,
            out_channels=attndim, 
            kernel_size=3, 
            stride=2, 
            # stride=1, 
            padding=1
        )

    def forward(
        self,
        model_embed,
        context_embed,
        context_embed_value=None,
        mask=None,
        return_atten=False,
    ):  
        # extract local information
        query_embed = self.query_conv(
            model_embed.transpose(1, 2)
        ).transpose(1, 2)

        logging.info(f'model_embed shape: {model_embed.shape}')
        logging.info(f'query_embed shape: {query_embed.shape}')

        out = super().forward(
            model_embed=query_embed,
            context_embed=context_embed,
            context_embed_value=context_embed_value,
            mask=mask,
            return_atten=return_atten,
        )
        return out

class ColbertAdapter(AttentionBasedAdapter):
    def __init__(
        self,
        attention_heads: int,
        attndim: int,
        proj_hidden_size: int,
        droup_out: float = 0.1,
        atten_temperature: float = 1.0,
        **kwargs
    ):  
        super().__init__(
            attention_heads=attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            droup_out=droup_out,
            use_value_norm=True,
            atten_temperature=atten_temperature,
            **kwargs,
        )
        self.attention_layer = ColbertAttention(
            attention_heads, attndim, droup_out
        )

    def forward(
        self,
        model_embed,
        context_embed_key,
        context_embed_value,
        mask=None,
        return_atten=False,
    ):  
        # TODO: Add query mask and key mask
        # may cause some problems (softmax cross utterance)...
        B, T, D  = model_embed.shape
        query    = self.norm_before_x1(model_embed)

        C, U, D = context_embed_key.shape
        key     = self.norm_before_x2(context_embed_key)
        
        # entity-level
        C, D  = context_embed_value.shape
        value = self.norm_before_x3(context_embed_value)

        out = self.attention_layer(
            query=query, 
            key=key, 
            value=value,
            mask=mask,
        )
        out = self.norm_after(out)
        out = self.proj(out)
        
        if return_atten:
            atten = torch.softmax(
                self.attention_layer.scores / self.temperature, 
                dim=-1
            )
            if mask is not None:
                mask  = mask.unsqueeze(1).eq(0)
                atten = atten.masked_fill(mask, 0.0)
            return out, atten
        return out

if __name__ == '__main__':
    attention_heads  = 1 
    attndim          = 4
    proj_hidden_size = 3
    droup_out        = 0.1

    adapter = ColbertAdapter(
        attention_heads=attention_heads,
        attndim=attndim,
        proj_hidden_size=proj_hidden_size,
        droup_out=droup_out,
    )

    B, T, D = 2, 5, attndim
    C, U, D = 4, 3, attndim

    model_embed         = torch.randn(B, T, D)
    context_embed_key   = torch.randn(C, U, D)
    context_embed_value = torch.randn(C, D)

    output, attn = adapter(
        model_embed, 
        context_embed_key,
        context_embed_value,
        return_atten=True
    )

    print(f'output: {output.shape}')
    print(f'attn  : {attn.shape}')