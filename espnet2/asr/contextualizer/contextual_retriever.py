import math
import torch
import random
import logging
import torch.nn.functional as F

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
    LateInteractiveRetriever,
    LateMultiInteractiveRetriever,
    Conv2LateMultiInteractiveRetriever,
    Conv2LateMultiInteractiveDropoutRetriever,
    ConformerLateMultiInteractiveRetriever,
    MultiLateInteractiveRetriever,
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
        **kwargs
    ):
        return self.encoder(text_embed, ilens)

    def forward_retriever(
        self,
        model_embed        : torch.Tensor,
        context_embed      : torch.Tensor,
        context_embed_value: torch.Tensor = None,
        return_model_proj  : bool=False,
    ):
        return self.retriever(
            model_embed, 
            context_embed,
            return_model_proj=return_model_proj
        )

    def forward(
        self,
        model_embed      : torch.Tensor,
        context_embed    : torch.Tensor,
        ilens            : torch.Tensor = None,
        return_model_proj: bool=False,
        **kwargs
    ):
        context_embed_mean, context_embed, ilens = self.forward_context_encoder(
            context_embed, 
            ilens
        )
        output = self.forward_retriever(
            model_embed=model_embed,
            context_embed=context_embed_mean,
            return_model_proj=return_model_proj,
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
        return_model_proj        : bool=False,
        **kwargs
    ):
        context_embed_merged, context_embed_mean, ilens = self.forward_context_encoder(
            context_embed,
            context_xphone_mean_embed,
            ilens
        )
        output = self.forward_retriever(
            model_embed=model_embed,
            context_embed=context_embed_mean,
            return_model_proj=return_model_proj,
        )
        return output

class ContextualLateInteractiveRetriever(
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
        self.encoder = ContextEncoderBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_hidden_size,
            output_size=input_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
        )
        self.retriever = LateInteractiveRetriever(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )
    
    def forward_context_encoder(
        self,
        text_embed       : torch.Tensor,
        xphone_mean_embed: torch.Tensor,
        ilens            : torch.Tensor,
        **kwargs
    ):
        return self.encoder(text_embed, ilens)

    def forward(
        self,
        model_embed              : torch.Tensor,
        context_embed            : torch.Tensor,
        context_xphone_mean_embed: torch.Tensor,
        ilens                    : torch.Tensor = None,
        return_model_proj        : bool=False,
        **kwargs
    ):
        context_embed_mean, context_embeds, ilens = self.forward_context_encoder(
            context_embed,
            context_xphone_mean_embed,
            ilens
        )
        output = self.forward_retriever(
            model_embed=model_embed,
            context_embed=context_embeds,
            return_model_proj=return_model_proj
        )
        return output

class ContextualMultiLateInteractiveRetriever(
    ContextualLateInteractiveRetriever
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
        use_oov: bool = True,
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
        self.encoder = ContextEncoderBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_hidden_size,
            output_size=input_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
        )
        self.encoder_xphone = ContextEncoderXPhone(
            vocab_size=vocab_size,
            hidden_size=context_hidden_size,
            output_size=input_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
        )
        self.retriever = LateMultiInteractiveRetriever(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )
        self.use_oov = use_oov
    
    def build_embeddings(
        self,
        text_embed   : torch.Tensor,
        xphone_embed : torch.Tensor,
        ilens        : torch.Tensor,
        xphone_ilens : torch.Tensor,
        **kwargs
    ):
        context_embeds, xphone_embeds, ilens, xphone_ilens = self.forward_multi_context_encoder(
            text_embed,
            xphone_embed,
            ilens,
            xphone_ilens,
            use_oov=self.use_oov,
        )
        # flatten embedding
        flatten_context_embeds = []
        flatten_context_idxs   = []
        context_idx2flatten    = []

        start = 0
        for i in range(ilens.shape[0]):
            ilens_int = ilens[i].item()
            flatten_context_embeds.append(context_embeds[i, :ilens_int])
            flatten_context_idxs.extend(
                [i for _ in range(ilens_int)]
            )
            context_idx2flatten.append([start, start + ilens_int])
            start = start + ilens_int
        flatten_context_embeds = torch.cat(flatten_context_embeds, dim=0)

        flatten_xphone_embeds = []
        flatten_xphone_idxs   = []
        xphone_idx2flatten    = []

        start = 0
        for i in range(xphone_ilens.shape[0]):
            xphone_ilens_int = xphone_ilens[i].item()
            flatten_xphone_embeds.append(xphone_embeds[i, :xphone_ilens_int])
            flatten_xphone_idxs.extend(
                [i for _ in range(xphone_ilens_int)]
            )
            xphone_idx2flatten.append([start, start + xphone_ilens_int])
            start = start + xphone_ilens_int
        flatten_xphone_embeds = torch.cat(flatten_xphone_embeds, dim=0)
        return {
            'embeddings': [flatten_context_embeds, flatten_xphone_embeds],
            'flatten2id': [flatten_context_idxs, flatten_xphone_idxs],
            'id2flatten': [context_idx2flatten, xphone_idx2flatten]
        }
    
    def forward_context_encoder(
        self,
        text_embed       : torch.Tensor,
        xphone_embed     : torch.Tensor,
        xphone_mean_embed: torch.Tensor,
        ilens            : torch.Tensor,
        xphone_ilens     : torch.Tensor,
        **kwargs
    ):
        # TODO: Fix the collapse problems!
        # xphone_embeds, xphone_ilens = self.encoder_xphone(xphone_embed, xphone_ilens)
        
        # # append oov
        # xphone_ilens = xphone_ilens.to(xphone_embeds.device)
        # xphone_ilens = torch.cat([torch.ones(1).to(xphone_embeds.device), xphone_ilens])
        # xphone_embed_mean = torch.sum(xphone_embeds, dim=1) / xphone_ilens.unsqueeze(1)
        
        # For now, we use the same xphone embeddings
        _, D = xphone_mean_embed.shape
        xphone_embeds = xphone_embed

        # dummy oov
        xphone_embed_mean = torch.cat([
            torch.randn(1, D).to(xphone_embeds.device),
            xphone_mean_embed
        ])
        return xphone_embed_mean, xphone_embeds, ilens

    def forward_multi_context_encoder(
        self,
        text_embed   : torch.Tensor,
        xphone_embed : torch.Tensor,
        ilens        : torch.Tensor,
        xphone_ilens : torch.Tensor,
        use_oov      : bool=True,
        **kwargs
    ):
        _, context_embeds, ilens    = self.encoder(text_embed, ilens)
        xphone_embeds, xphone_ilens = self.encoder_xphone(xphone_embed, xphone_ilens, use_oov=use_oov)
        return context_embeds, xphone_embeds, ilens, xphone_ilens

    def forward_retriever(
        self,
        model_embed      : torch.Tensor,
        context_embed    : torch.Tensor,
        xphone_embed     : torch.Tensor,
        return_model_proj: bool=False,
        **kwargs
    ):
        return self.retriever(
            model_embed, 
            context_embed,
            xphone_embed,
            return_model_proj=return_model_proj,
        )

    def forward(
        self,
        model_embed         : torch.Tensor,
        context_embed       : torch.Tensor,
        context_xphone_embed: torch.Tensor,
        ilens               : torch.Tensor = None,
        xphone_ilens        : torch.Tensor = None,
        return_model_proj   : bool=False,
        **kwargs
    ):
        context_embeds, xphone_embeds, ilens, xphone_ilens = self.forward_multi_context_encoder(
            context_embed,
            context_xphone_embed,
            ilens,
            xphone_ilens,
            use_oov=self.use_oov,
        )
        output = self.forward_retriever(
            model_embed=model_embed,
            context_embed=context_embeds,
            xphone_embed=xphone_embeds,
            return_model_proj=return_model_proj
        )
        return output

class ContextualConv2MultiLateInteractiveRetriever(
    ContextualMultiLateInteractiveRetriever
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
        use_oov: bool = True,
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
            use_oov=use_oov,
            **kwargs
        )
        self.retriever = Conv2LateMultiInteractiveRetriever(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )

class ContextualConv2MultiLateInteractiveDropoutRetriever(
    ContextualMultiLateInteractiveRetriever
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
        use_oov: bool = True,
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
            use_oov=use_oov,
            **kwargs
        )
        self.retriever = Conv2LateMultiInteractiveDropoutRetriever(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )

class ContextualConformerMultiLateInteractiveRetriever(
    ContextualMultiLateInteractiveRetriever
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
        use_oov: bool = True,
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
            use_oov=use_oov,
            **kwargs
        )
        self.retriever = ConformerLateMultiInteractiveRetriever(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )

class MultiLateInteractiveContextRetriever(
    ContextualLateInteractiveRetriever
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
        use_oov: bool = True,
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
        self.encoder = ContextEncoderBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_hidden_size,
            output_size=input_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
        )
        self.encoder_xphone = ContextEncoderXPhone(
            vocab_size=vocab_size,
            hidden_size=context_hidden_size,
            output_size=input_hidden_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            xphone_hidden_size=xphone_hidden_size,
            merge_conv_kernel=merge_conv_kernel,
        )
        self.retriever = MultiLateInteractiveRetriever(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )
        self.use_oov = use_oov

    def forward_context_encoder(
        self,
        text_embed       : torch.Tensor,
        xphone_embed     : torch.Tensor,
        xphone_mean_embed: torch.Tensor,
        ilens            : torch.Tensor,
        xphone_ilens     : torch.Tensor,
        **kwargs
    ):
        # TODO: Fix the collapse problems!
        # For now, we use the same xphone embeddings
        _, D = xphone_mean_embed.shape
        xphone_embeds = xphone_embed

        # dummy oov
        xphone_embed_mean = torch.cat([
            torch.randn(1, D).to(xphone_embeds.device),
            xphone_mean_embed
        ])
        return xphone_embed_mean, xphone_embeds, ilens

    def forward_multi_context_encoder(
        self,
        text_embed   : torch.Tensor,
        xphone_embed : torch.Tensor,
        ilens        : torch.Tensor,
        xphone_ilens : torch.Tensor,
        use_oov      : bool=True,
        **kwargs
    ):
        _, context_embeds, ilens    = self.encoder(text_embed, ilens)
        xphone_embeds, xphone_ilens = self.encoder_xphone(xphone_embed, xphone_ilens, use_oov=use_oov)
        return context_embeds, xphone_embeds, ilens, xphone_ilens

    def forward_retriever(
        self,
        model_embed      : torch.Tensor,
        context_embed    : torch.Tensor,
        xphone_embed     : torch.Tensor,
        return_model_proj: bool=False,
        **kwargs
    ):
        return self.retriever(
            model_embed, 
            context_embed,
            xphone_embed,
            return_model_proj=return_model_proj,
        )

    def forward(
        self,
        model_embed         : torch.Tensor,
        context_embed       : torch.Tensor,
        context_xphone_embed: torch.Tensor,
        ilens               : torch.Tensor = None,
        xphone_ilens        : torch.Tensor = None,
        return_model_proj   : bool=False,
        **kwargs
    ):
        context_embeds, xphone_embeds, ilens, xphone_ilens = self.forward_multi_context_encoder(
            context_embed,
            context_xphone_embed,
            ilens,
            xphone_ilens,
            use_oov=self.use_oov,
        )
        output = self.forward_retriever(
            model_embed=model_embed,
            context_embed=context_embeds,
            xphone_embed=xphone_embeds,
            return_model_proj=return_model_proj
        )
        return output