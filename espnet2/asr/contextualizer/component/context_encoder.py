import math
import torch
import random
import logging

from typing import List, Optional, Tuple
from espnet2.asr.encoder.rnn_encoder                   import RNNEncoder
from espnet2.asr.encoder.transformer_encoder           import TransformerEncoder
from espnet.nets.pytorch_backend.nets_utils            import make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding

class ContextEncoderBiLSTM(torch.nn.Module):
    def __init__(
        self,
        hidden_size : int,
        output_size : int,
        droup_out   : float = 0.1,
        num_blocks  : int = 1,
        **kwargs
    ):
        super().__init__()
        self.oovembed  = torch.nn.Embedding(1, hidden_size)
        self.droup_out = torch.nn.Dropout(droup_out)
        self.encoder   = RNNEncoder(
            input_size=output_size,
            num_layers=num_blocks,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=droup_out,
            subsample=None,
        )

    def forward(
        self,
        context_embed: torch.Tensor,
        ilens: torch.Tensor,
    ):
        context_embed, _, _ = self.encoder(context_embed, ilens)
        context_embed       = torch.mean(context_embed, dim=1)
        return context_embed

class ContextEncoderTransformer(torch.nn.Module):
    def __init__(
        self,
        hidden_size : int,
        output_size : int,
        droup_out   : float = 0.1,
        attention_heads: int = 4,
        linear_units: int = 256,
        num_blocks: int = 2,
        padding_idx: int = -1,
        **kwargs
    ):
        super().__init__()
        self.oovembed  = torch.nn.Embedding(1, hidden_size)
        self.pos_enc   = PositionalEncoding(output_size, 0.1)
        self.encoder   = TransformerEncoder(
            input_size=output_size,
            output_size=output_size,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            dropout_rate=droup_out,
            input_layer=None,
            padding_idx=padding_idx,
        )

    def forward(
        self,
        context_embed: torch.Tensor,
        ilens: torch.Tensor
    ):
        context_embed       = self.pos_enc(context_embed)
        context_embed, _, _ = self.encoder(xs_pad=context_embed, ilens=ilens)

        masks = (~make_pad_mask(ilens)[:, None, :]).float().to(context_embed.device)
        context_embed = masks @ context_embed
        return context_embed.squeeze(1)

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