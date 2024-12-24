import math
import torch
import random
import logging
import torch.nn.functional as F

from typing import List, Optional, Tuple
from espnet2.asr.encoder.rnn_encoder                    import RNNEncoder
from espnet2.asr.encoder.transformer_encoder            import TransformerEncoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.nets_utils             import make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding  import PositionalEncoding

class ContextEncoderBiLSTM(torch.nn.Module):
    def __init__(
        self,
        vocab_size  : int,
        hidden_size : int,
        output_size : int,
        drop_out   : float = 0.0,
        num_blocks  : int = 1,
        padding_idx : int = -1,
        **kwargs
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.padding_idx = padding_idx
        self.embed       = torch.nn.Embedding(
            vocab_size, 
            hidden_size, 
        )
        self.oov_embed = torch.nn.Linear(hidden_size, 1, bias=False)
        self.pad_embed = torch.nn.Linear(hidden_size, 1, bias=False)
        with torch.no_grad():
            self.pad_embed.weight.fill_(0)

        self.encoder = RNNEncoder(
            input_size=output_size,
            num_layers=num_blocks,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=0.0,
            subsample=None,
            use_projection=False,
        )

    def forward_embed(self, x: torch.Tensor):
        embedding_matrix = torch.cat([
            self.embed.weight, 
            self.oov_embed.weight,
            self.pad_embed.weight,
        ], dim=0)
        out = embedding_matrix[x]
        return out

    def forward(
        self,
        context_embed: torch.Tensor,
        ilens: torch.Tensor,
    ):
        context_embed           = self.forward_embed(context_embed)
        context_embed, ilens, _ = self.encoder(context_embed, ilens)
        
        ilens = ilens.to(context_embed.device)
        context_embed_mean = torch.sum(context_embed, dim=1) / ilens.unsqueeze(1)
        return context_embed_mean, context_embed, ilens

class ContextEncoderTransformer(ContextEncoderBiLSTM):
    def __init__(
        self,
        vocab_size: int,
        hidden_size : int,
        output_size : int,
        drop_out   : float = 0.1,
        attention_heads: int = 4,
        linear_units: int = 256,
        num_blocks: int = 2,
        padding_idx: int = -1,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            output_size=output_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            **kwargs
        )
        self.pos_enc = PositionalEncoding(output_size, 0.1)
        self.encoder = TransformerEncoder(
            input_size=output_size,
            output_size=output_size,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=drop_out,
            input_layer=None,
            padding_idx=padding_idx,
        )

    def forward(
        self,
        context_embed: torch.Tensor,
        ilens: torch.Tensor
    ):
        context_embed       = self.forward_embed(context_embed)
        context_embed       = self.pos_enc(context_embed)
        context_embed, _, _ = self.encoder(xs_pad=context_embed, ilens=ilens)

        masks = (~make_pad_mask(ilens)[:, None, :]).float().to(context_embed.device)
        context_embed_mean = (masks @ context_embed).squeeze(1) / ilens.unsqueeze(1)
        return context_embed_mean, context_embed, ilens

class ContextualRetriever(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
    
    def softmax(
        self,
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        return self.forward(
            query,
            query_ilens,
            context_subword,      
            context_subword_ilens,
            context_phone,        
            context_phone_ilens,  
            **kwargs
        )

    def log_softmax(
        self,
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        return torch.log(self.forward(
            query,
            query_ilens,
            context_subword,      
            context_subword_ilens,
            context_phone,        
            context_phone_ilens,  
            **kwargs
        ))

    def argmax(
        self, 
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        return torch.argmax(self.forward(
            query,
            query_ilens,
            context_subword,      
            context_subword_ilens,
            context_phone,        
            context_phone_ilens,  
            **kwargs
        ))

    @abc.abstractmethod
    def forward_query_encoder(
        self, 
        query: torch.Tensor, 
        ilens: torch.Tensor, 
        **kwargs
    ):
        """Encode the query input."""
        raise NotImplementedError("The method `forward_query_encoder` must be implemented by the subclass.")
    
    @abc.abstractmethod
    def forward_context_encoder(
        self, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        """Encode the context input."""
        raise NotImplementedError("The method `forward_context_encoder` must be implemented by the subclass.")
    
    @abc.abstractmethod
    def forward_hnc_encoder(
        self, 
        context: torch.Tensor, 
        ilens  : torch.Tensor, 
        **kwargs
    ):
        """Encode the context input for hard negative context mining."""
        raise NotImplementedError("The method `forward_hnc_encoder` must be implemented by the subclass.")

    @abc.abstractmethod
    def forward_retriever(
        self,
        query  : torch.Tensor, 
        context: torch.Tensor, 
        **kwargs
    ):
        """Compute the retrieval based on the encoded context."""
        raise NotImplementedError("The method `forward_retriever` must be implemented by the subclass.")
    
    def forward(
        self, 
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        return_model_proj    : bool = False, 
        **kwargs
    ):
        """Compute the forward pass."""
        query, query_ilens                     = self.forward_query_encoder(query, query_ilens)
        context_subword, context_subword_ilens = self.forward_context_encoder(
            context_subword, 
            context_subword_ilens,
            context_phone,
            context_phone_ilens,
        )

        scores = self.forward_retriever(query, query_ilens, context_subword, context_subword_ilens)
        probs  = torch.softmax(scores, dim=-1)

        if return_model_proj:
            return probs, query
        return probs    

class DotProductContextualRetriever(ContextualRetriever):
    def __init__(
        self,
        vocab_size          : int,
        query_input_dim     : int,
        context_input_dim   : int,
        proj_dim            : int,
        interaction_proj_dim: int,
        dropout             : float = 0.1,
        pad_token_value     : int = -1,
        **kwargs
    ):
        super().__init__()

        self.query_encoder = build_conformer_block(
            proj_hidden_size=query_input_dim,
            drop_out=dropout
        )
        self.context_encoder = ContextEncoderBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_input_dim,
            output_size=proj_dim,
            num_blocks=2,
            drop_out=dropout,
            padding_idx=pad_token_value,
        )
        self.retriever = DotProductInteractor(
            input_dim=proj_dim,
            proj_dim=interaction_proj_dim,
            dropout=dropout,
            **kwargs
        )

    def forward_query_encoder(
        self, 
        query: torch.Tensor, 
        ilens: torch.Tensor, 
        **kwargs
    ):
        query_hat, mask = self.query_encoder(query, mask=None)
        return (query + query_hat), ilens

    def forward_context_encoder(
        self, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        # TODO: move mean operation out of the context encoder
        context, _, ilens = self.context_encoder(context_subword, context_subword_ilens)
        return context, ilens

    def forward_hnc_encoder(
        self, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        return self.forward_context_encoder(
            context_subword, 
            context_subword_ilens,
            context_phone,
            context_phone_ilens,
        )

    def forward_retriever(
        self,
        query        : torch.Tensor,
        query_ilens  : torch.Tensor,
        context      : torch.Tensor, 
        context_ilens: torch.Tensor,
        **kwargs
    ):
        scores = self.retriever(query, context)
        return scores

class ContextEncoderXPhoneBiLSTM(ContextEncoderBiLSTM):
    def __init__(
        self,
        vocab_size        : int,
        hidden_size       : int,
        output_size       : int,
        drop_out         : float = 0.0,
        num_blocks        : int = 1,
        padding_idx       : int = -1,
        xphone_hidden_size: int = 768,
        merge_conv_kernel : int = 3,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            output_size=output_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            **kwargs,
        )
        self.drop_out = torch.nn.Dropout(p=drop_out)
        self.norm_x1  = LayerNorm(hidden_size)
        self.norm_x2  = LayerNorm(xphone_hidden_size)
        self.merge_conv_kernel = merge_conv_kernel

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            hidden_size + xphone_hidden_size,
            hidden_size + xphone_hidden_size,
            kernel_size=self.merge_conv_kernel,
            stride=1,
            padding=(self.merge_conv_kernel - 1) // 2,
            groups=hidden_size + xphone_hidden_size,
            bias=True,
        )
        self.proj = torch.nn.Linear(
            hidden_size + xphone_hidden_size, 
            hidden_size
        )
    
    def branch_merge(self, x1, x2):
        # Merge two branches
        x     = torch.cat([x1, x2], dim=-1).unsqueeze(0)
        x_tmp = x.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        return x + x_tmp

    def forward(
        self,
        context_embed: torch.Tensor,
        context_xphone_embed: torch.Tensor,
        ilens: torch.Tensor,
    ):
        context_embed_mean, context_embed, ilens = super().forward(
            context_embed=context_embed,
            ilens=ilens,
        )
        oov_embed = context_embed_mean[:1, :]
        x1_embed  = context_embed_mean[1:, :]
        x2_embed  = context_xphone_embed
        # layer normalize and dropout
        # x1_embed = self.norm_x1(self.drop_out(x1_embed))
        # x2_embed = self.norm_x2(self.drop_out(x2_embed))
        x1_embed = (self.drop_out(x1_embed))
        x2_embed = (self.drop_out(x2_embed))
        x_embed  = self.branch_merge(x1_embed, x2_embed)
        x_embed  = self.drop_out(self.proj(x_embed)).squeeze(0)
        merged_context_embed = torch.cat([oov_embed, x_embed], dim=0)
        return merged_context_embed, context_embed_mean, ilens

class ContextEncoderXPhone(torch.nn.Module):
    def __init__(
        self,
        vocab_size        : int,
        hidden_size       : int,
        drop_out          : float = 0.0,
        padding_idx       : int = -1,
        xphone_hidden_size: int = 768,
        **kwargs
    ):
        super().__init__()
        self.oov_embed = torch.nn.Linear(hidden_size, 1, bias=False)
        self.drop_out  = torch.nn.Dropout(p=drop_out)
        self.norm_x2   = LayerNorm(xphone_hidden_size)

        self.proj = torch.nn.Linear(
            xphone_hidden_size, 
            hidden_size
        )

    def forward(
        self,
        xphone_embeds: torch.Tensor,
        ilens: torch.Tensor,
        use_oov: bool=True,
    ):
        # xphone_embeds: C x S x D
        # x2_embed = self.drop_out(self.norm_x2(xphone_embeds))
        x2_embed = self.drop_out(xphone_embeds)
        x_embed  = self.drop_out(self.proj(x2_embed)).squeeze(0)
        C, S, D  = x_embed.shape
        if use_oov:
            oov_embed = (self.oov_embed.weight).unsqueeze(0)
            oov_embed = F.pad(oov_embed, (0, 0, 0, (S - 1), 0, 0))
            x_embed   = torch.cat([oov_embed, x_embed], dim=0)
            ilens     = torch.cat([torch.zeros(1).to(ilens.device), ilens])
        return x_embed, ilens

if __name__ == '__main__':

    encoder = ContextEncoderBiLSTM(
        hidden_size=128,
        output_size=256,
        num_blocks=2,
    )

    xs_pad = torch.randn(2, 3, 256)
    ilens  = torch.tensor([1, 3])
    context_embed = encoder(xs_pad, ilens)
    print(f'bilstm context_embed: {context_embed.shape}')

    encoder = ContextEncoderTransformer(
        hidden_size=128,
        output_size=256,
        attention_heads=1,
        num_blocks=2,
        linear_units=256,
    )

    xs_pad = torch.randn(2, 3, 256)
    ilens  = torch.tensor([1, 3])
    context_embed = encoder(xs_pad, ilens)
    print(f'transformer context_embed: {context_embed.shape}')