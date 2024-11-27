"""
ContextualAdapter Module for Context-Aware ASR

This module provides multiple contextual adapters to improve ASR performance through advanced contextual embedding and attention-based mechanisms. It integrates adapters with phoneme-aware components to bias recognition towards relevant context.

Key Features:
1. **Attention-based Contextual Adapters:**  
   - Supports transformer-based, BiLSTM, and phoneme-aware encoders.
2. **Gated Mechanisms for Context Control:**  
   - Uses gating to dynamically regulate context influence during inference.
3. **Support for Convolutional and Hybrid Attention Models:**  
   - Adapts to complex ASR pipelines with multiple attention mechanisms.
4. **Residual Gate Control:**  
   - Balances over-adaptation by regulating residual information flow.
"""


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
    AttentionBasedLightAdapter,
)

from espnet2.asr.contextualizer.component.utils import GatedAdditionWithDropout

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
        **kwargs
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
        atten_temperature: float = 1.0,
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
            attention_heads=adapter_attention_heads,
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
        )
        self.encoder = ContextEncoderTransformer(
            vocab_size=vocab_size,
            hidden_size=context_embed_size,
            output_size=context_hidden_size,
            attention_heads=context_attention_heads,
            num_blocks=num_blocks,
            linear_units=linear_units,
            drop_out=drop_out,
            padding_idx=padding_idx,
        )

class ContextualLightAdapterTransformer(ContextualAdapterPrototype):
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
        atten_temperature: float = 1.0,
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
            attention_heads=adapter_attention_heads,
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
        )
        self.encoder = ContextEncoderTransformer(
            vocab_size=vocab_size,
            hidden_size=context_embed_size,
            output_size=context_hidden_size,
            attention_heads=context_attention_heads,
            num_blocks=num_blocks,
            linear_units=linear_units,
            drop_out=drop_out,
            padding_idx=padding_idx,
        )
        self.adapter = AttentionBasedLightAdapter(
            attention_heads=adapter_attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_value_norm=False,
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
        )

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
        text_embed       : torch.Tensor,
        xphone_mean_embed: torch.Tensor,
        ilens            : torch.Tensor,
        **kwargs
    ):
        return self.encoder(text_embed, xphone_mean_embed, ilens)

    def forward(
        self,
        model_embed              : torch.Tensor,
        context_embed            : torch.Tensor,
        context_xphone_mean_embed: torch.Tensor,
        ilens                    : torch.Tensor = None,
        mask                     : torch.Tensor = None,
        return_atten             : bool = False,
        **kwargs
    ):
        context_embed_merged, context_embed_mean, ilens = self.forward_context_encoder(
            context_embed,
            context_xphone_mean_embed,
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

class GatedContextualAdapterTransformer(ContextualAdapterTransformer):
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
        atten_temperature: float = 1.0,
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
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
            **kwargs
        )
        self.adapter = AttentionBasedAdapter(
            attention_heads=adapter_attention_heads,
            attndim=attndim,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            use_value_norm=False,
            atten_temperature=atten_temperature,
            use_local_attn_conv=use_local_attn_conv,
        )
        self.gate_layer = GatedAdditionWithDropout(proj_hidden_size, 0.1)