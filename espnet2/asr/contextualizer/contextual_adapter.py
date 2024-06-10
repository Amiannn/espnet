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

from espnet2.asr.contextualizer.component.attention_based_adapter import (
    AttentionBasedAdapter,
    ConvAttentionAdapter,
    Conv2AttentionAdapter,
    ColbertAdapter
)

class ContextualAdapterPrototype(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        num_blocks: int=1,
        drop_out: float = 0.1,
        attention_heads: int = 1,
        use_value_norm: bool = False,
        padding_idx: int = -1,
        atten_temperature: float = 1.0,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__()
        self.proj_hidden_size = proj_hidden_size
        self.encoder = ContextEncoderBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_embed_size,
            output_size=context_hidden_size,
            num_blocks=num_blocks,
            drop_out=drop_out,
            padding_idx=padding_idx,
        )
        self.adapter = AttentionBasedAdapter(
            attention_heads=attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
        )

    def forward_context_encoder(
        self,
        text_embed: torch.Tensor,
        ilens     : torch.Tensor,
    ):
        return self.encoder(text_embed, ilens)

    def forward_adapter(
        self,
        model_embed        : torch.Tensor,
        context_embed      : torch.Tensor,
        context_embed_value: torch.Tensor = None,
        mask               : torch.Tensor = None,
        return_atten       : bool=False,
    ):
        return self.adapter(
            model_embed, 
            context_embed, 
            context_embed_value, 
            mask, 
            return_atten
        )

    def forward(
        self,
        model_embed  : torch.Tensor,
        context_embed: torch.Tensor,
        ilens        : torch.Tensor = None,
        mask         : torch.Tensor = None,
        return_atten : bool = False,
        **kwargs
    ):
        context_embed_mean, context_embed, ilens = self.forward_context_encoder(
            context_embed, 
            ilens
        )
        output = self.forward_adapter(
            model_embed=model_embed,
            context_embed=context_embed_mean,
            context_embed_value=None,
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
        drop_out: float = 0.1,
        num_blocks: int=2,
        linear_units: int=256,
        context_attention_heads: int=4,
        adapter_attention_heads: int=1,
        padding_idx: int=-1,
        atten_temperature: float = 1.0,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__(
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            model_hidden_size=model_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            attention_heads=adapter_attention_heads,
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
        )
        self.encoder = ContextEncoderTransformer(
            hidden_size=context_embed_size,
            output_size=context_hidden_size,
            attention_heads=context_attention_heads,
            num_blocks=num_blocks,
            linear_units=linear_units,
            drop_out=drop_out,
            padding_idx=padding_idx,
        )

class ContextualColbertAdapter(ContextualAdapterPrototype):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        num_blocks: int=2,
        linear_units: int=256,
        context_attention_heads: int=4,
        adapter_attention_heads: int=1,
        padding_idx: int=-1,
        use_value_norm: bool=False,
        atten_temperature: float = 1.0,
        use_local_attn_conv: bool = False,
        contextual_list_size: int = 201,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            model_hidden_size=model_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            linear_units=linear_units,
            context_attention_heads=context_attention_heads,
            adapter_attention_heads=adapter_attention_heads,
            padding_idx=padding_idx,
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
            contextual_list_size=contextual_list_size,
            **kwargs
        )
        self.adapter = ColbertAdapter(
            attention_heads=adapter_attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_local_attn_conv=use_local_attn_conv,
            contextual_list_size=contextual_list_size,
        )

    def forward(
        self,
        model_embed  : torch.Tensor,
        context_embed: torch.Tensor,
        ilens        : torch.Tensor = None,
        mask         : torch.Tensor = None,
        return_atten : bool = False,
        **kwargs
    ):
        context_embed_mean, context_embed, ilens = self.forward_context_encoder(
            context_embed, 
            ilens
        )
        output = self.forward_adapter(
            model_embed=model_embed,
            context_embed=context_embed,
            context_embed_value=context_embed_mean,
            mask=mask,
            return_atten=return_atten,
        )
        return output

class ContextualXPhoneAdapter(ContextualAdapterPrototype):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        num_blocks: int=2,
        linear_units: int=256,
        context_attention_heads: int=4,
        adapter_attention_heads: int=1,
        padding_idx: int=-1,
        use_value_norm: bool=True,
        atten_temperature: float = 1.0,
        xphone_hidden_size: int = 768,
        merge_conv_kernel: int = 3,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            model_hidden_size=model_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            linear_units=linear_units,
            context_attention_heads=context_attention_heads,
            adapter_attention_heads=adapter_attention_heads,
            padding_idx=padding_idx,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
            **kwargs
        )
        self.encoder = ContextEncoderXPhoneBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_hidden_size,
            output_size=attndim,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
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
        mask                : torch.Tensor = None,
        return_atten        : bool = False,
        **kwargs
    ):
        context_embed_merged, context_embed_mean, ilens = self.forward_context_encoder(
            context_embed,
            context_xphone_embed,
            ilens
        )
        output = self.forward_adapter(
            model_embed=model_embed,
            context_embed=context_embed_merged,
            context_embed_value=context_embed_mean,
            mask=mask,
            return_atten=return_atten,
        )
        return output

class ContextualConvXPhoneAdapter(ContextualXPhoneAdapter):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        num_blocks: int=2,
        linear_units: int=256,
        context_attention_heads: int=4,
        adapter_attention_heads: int=1,
        padding_idx: int=-1,
        use_value_norm: bool=True,
        atten_temperature: float = 1.0,
        xphone_hidden_size: int = 768,
        merge_conv_kernel: int = 3,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            model_hidden_size=model_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            linear_units=linear_units,
            context_attention_heads=context_attention_heads,
            adapter_attention_heads=adapter_attention_heads,
            padding_idx=padding_idx,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
            use_local_attn_conv=use_local_attn_conv,
            **kwargs
        )
        self.adapter = ConvAttentionAdapter(
            attention_heads=adapter_attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_value_norm=use_value_norm,
            use_local_attn_conv=use_local_attn_conv,
            atten_temperature=atten_temperature,
        )

class ContextualConv2XPhoneAdapter(ContextualXPhoneAdapter):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        num_blocks: int=2,
        linear_units: int=256,
        context_attention_heads: int=4,
        adapter_attention_heads: int=1,
        padding_idx: int=-1,
        use_value_norm: bool=True,
        atten_temperature: float = 1.0,
        xphone_hidden_size: int = 768,
        merge_conv_kernel: int = 3,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            model_hidden_size=model_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            linear_units=linear_units,
            context_attention_heads=context_attention_heads,
            adapter_attention_heads=adapter_attention_heads,
            padding_idx=padding_idx,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
            use_local_attn_conv=use_local_attn_conv,
            **kwargs
        )
        self.adapter = Conv2AttentionAdapter(
            attention_heads=adapter_attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_value_norm=use_value_norm,
            use_local_attn_conv=use_local_attn_conv,
            atten_temperature=atten_temperature,
        )

class ContextualConvXPhoneGatedAdapter(ContextualConvXPhoneAdapter):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        num_blocks: int=2,
        linear_units: int=256,
        context_attention_heads: int=4,
        adapter_attention_heads: int=1,
        padding_idx: int=-1,
        use_value_norm: bool=True,
        atten_temperature: float = 1.0,
        xphone_hidden_size: int = 768,
        merge_conv_kernel: int = 3,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            model_hidden_size=model_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            linear_units=linear_units,
            context_attention_heads=context_attention_heads,
            adapter_attention_heads=adapter_attention_heads,
            padding_idx=padding_idx,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
            use_local_attn_conv=use_local_attn_conv,
            **kwargs
        )
        self.gate_drop_x1     = torch.nn.Dropout(0.1)
        # self.gate_drop_x2   = torch.nn.Dropout(0.1)
        self.gate_norm_x1     = LayerNorm(attndim)
        # self.gate_norm_x2   = LayerNorm(proj_hidden_size)
        self.gate_linear_x1   = torch.nn.Linear(model_hidden_size, model_hidden_size // 2)
        # self.gate_linear_x2 = torch.nn.Linear(proj_hidden_size, proj_hidden_size // 2)
        self.gate_linear      = torch.nn.Linear(model_hidden_size // 2, 1)

    def forward(
        self,
        model_embed  : torch.Tensor,
        context_embed: torch.Tensor,
        ilens        : torch.Tensor = None,
        mask         : torch.Tensor = None,
        return_atten : bool = False,
        **kwargs
    ):
        out = super().forward(
            model_embed=model_embed,
            context_embed=context_embed,
            ilens=ilens,
            mask=mask,
            return_atten=return_atten,
            **kwargs
        )
        if return_atten:
            out, atten = out

        query_embed = self.adapter.query_embed
        x1 = torch.tanh(self.gate_linear_x1(self.gate_drop_x1(self.gate_norm_x1(query_embed))))
        # x2 = self.gate_drop_x2(self.gate_linear_x2(self.gate_norm_x2(out)))
        self.gate_value = self.gate_linear(x1)
        # self.gate_value = x1 + x2
        self.gate_prob  = torch.sigmoid(self.gate_value)
        # residual gate
        out = out * self.gate_prob
        if return_atten:
            return out, atten
        return out

class ContextualConv2XPhoneGatedAdapter(ContextualConvXPhoneGatedAdapter):
    def __init__(
        self,
        vocab_size: int,
        context_embed_size: int,
        context_hidden_size: int,
        model_hidden_size: int,
        attndim: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        num_blocks: int=2,
        linear_units: int=256,
        context_attention_heads: int=4,
        adapter_attention_heads: int=1,
        padding_idx: int=-1,
        use_value_norm: bool=True,
        atten_temperature: float = 1.0,
        xphone_hidden_size: int = 768,
        merge_conv_kernel: int = 3,
        use_local_attn_conv: bool = False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            context_embed_size=context_embed_size,
            context_hidden_size=context_hidden_size,
            model_hidden_size=model_hidden_size,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            linear_units=linear_units,
            context_attention_heads=context_attention_heads,
            adapter_attention_heads=adapter_attention_heads,
            padding_idx=padding_idx,
            use_value_norm=use_value_norm,
            atten_temperature=atten_temperature,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
            use_local_attn_conv=use_local_attn_conv,
            **kwargs
        )
        self.adapter = Conv2AttentionAdapter(
            attention_heads=adapter_attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_value_norm=use_value_norm,
            use_local_attn_conv=use_local_attn_conv,
            atten_temperature=atten_temperature,
        )

if __name__ == '__main__':
    B, U, T, C = 2, 5, 10, 6

    vocab_size          = 5
    dropout             = 0.1
    encoder_hidden_size = 256
    context_hidden_size = 256
    joint_space_size    = 512
    proj_hidden_size    = 128
    context_embed_size  = 128

    context_adapter_prototype = ContextualColbertAttenAdapter(
        vocab_size=vocab_size,
        context_embed_size=context_embed_size,
        model_hidden_size=encoder_hidden_size,
        context_hidden_size=context_hidden_size,
        proj_hidden_size=proj_hidden_size,
        attndim=context_hidden_size,
        dropout=dropout,
    )

    model_out  = torch.rand(B, T, encoder_hidden_size)
    print(f'model_out: {model_out.shape}')
    text_embed = torch.rand(C, U, encoder_hidden_size)
    print(f'text_embed: {text_embed.shape}')
    
    ilens = torch.tensor([5, 3, 2, 4, 1, 5])

    bias = context_adapter_prototype(model_out, text_embed, ilens)
    print(f'contextual adapter prototype bias: {bias.shape}')