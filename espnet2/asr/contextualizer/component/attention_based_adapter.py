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

class ConvAttentionAdapter(AttentionBasedAdapter):
    def __init__(
        self,
        attention_heads: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        use_value_norm: bool = False,
        atten_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(
            attention_heads=attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            **kwargs,
        )
        downproj_rate = 2
        self.conv_1x1 = torch.nn.Conv1d(
            in_channels=attndim,
            out_channels=(attndim // downproj_rate), 
            kernel_size=1, 
        )
        self.conv = torch.nn.Conv1d(
            in_channels=(attndim // downproj_rate),
            out_channels=attndim, 
            kernel_size=3, 
            stride=1, 
            # stride=1, 
            padding=1
        )
        
    def forward_conv(self, query):
        x = self.conv_1x1(query.transpose(1, 2))
        x = self.conv(x)
        return (query + x.transpose(1, 2))

    def forward(
        self,
        model_embed,
        context_embed,
        context_embed_value=None,
        mask=None,
        return_atten=False,
    ):  
        # extract local information
        self.query_embed = self.forward_conv(model_embed)
        out = super().forward(
            model_embed=self.query_embed,
            context_embed=context_embed,
            context_embed_value=context_embed_value,
            mask=mask,
            return_atten=return_atten,
        )
        return out

class Conv2AttentionAdapter(ConvAttentionAdapter):
    def __init__(
        self,
        attention_heads: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        use_value_norm: bool = False,
        atten_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(
            attention_heads=attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            **kwargs,
        )
        downproj_rate = 2
        self.conv_1x1 = torch.nn.Conv1d(
            in_channels=attndim,
            out_channels=(attndim // downproj_rate), 
            kernel_size=1, 
        )
        self.conv_1 = torch.nn.Conv1d(
            in_channels=(attndim // downproj_rate),
            out_channels=(attndim // downproj_rate), 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.conv_2 = torch.nn.Conv1d(
            in_channels=(attndim // downproj_rate),
            out_channels=attndim, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
    def forward_conv(self, query):
        x = self.conv_1x1(query.transpose(1, 2))
        x = self.conv_1(x)
        x = self.conv_2(x)
        return query + x.transpose(1, 2)

if __name__ == '__main__':
    attention_heads  = 1 
    attndim          = 4
    proj_hidden_size = 3
    drop_out        = 0.1

    # adapter = ColbertAdapter(
    #     attention_heads=attention_heads,
    #     attndim=attndim,
    #     proj_hidden_size=proj_hidden_size,
    #     drop_out=drop_out,
    # )

    B, T, D = 2, 5, attndim
    C, U, D = 4, 3, attndim

    model_embed         = torch.randn(B, T, D)
    context_embed_key   = torch.randn(C, U, D)
    context_embed_value = torch.randn(C, D)

    # output, attn = adapter(
    #     model_embed, 
    #     context_embed_key,
    #     context_embed_value,
    #     return_atten=True
    # )

    # print(f'output: {output.shape}')
    # print(f'attn  : {attn.shape}')
    # With square kernels and equal stride
    m = torch.nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(2, 1))
    input = torch.randn(20, 1, 50, 100)
    output = m(input)
    print(f'output: {output.shape}')