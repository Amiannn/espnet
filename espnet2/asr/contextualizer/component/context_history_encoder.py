import math
import torch
import random
import logging

from typing import List, Optional, Tuple
from espnet2.asr.encoder.rnn_encoder                      import RNNEncoder
from espnet2.asr.encoder.transformer_encoder              import TransformerEncoder
from espnet.nets.pytorch_backend.transformer.layer_norm   import LayerNorm
from espnet.nets.pytorch_backend.nets_utils               import make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding    import PositionalEncoding

from espnet2.asr.contextualizer.component.context_encoder import ContextEncoderBiLSTM

class ContextHistoryEncoderLSTM(torch.nn.Module):
    def __init__(
        self,
        hidden_size    : int,
        output_size    : int,
        drop_out       : float = 0.0,
        num_blocks     : int = 1,
        padding_idx    : int = -1,
        **kwargs
    ):
        super().__init__()
        self.padding_idx      = padding_idx
        self.sequence_encoder = RNNEncoder(
            input_size=output_size,
            num_layers=num_blocks,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=drop_out,
            subsample=None,
            use_projection=False,
            bidirectional=False,
        )

    def forward(
        self,
        context_history_embed: torch.Tensor,
        ilens : torch.Tensor,
    ):
        context_history_embed, ilens, _ = self.sequence_encoder(context_history_embed, ilens)
        return context_history_embed

if __name__ == "__main__":
    encoder = ContextHistoryEncoderLSTM(
        hidden_size=256,
        output_size=256,
        num_blocks=1,
    )
    
    xs_pad = torch.randn(26, 11, 256)
    ilens  = torch.tensor([1,  3,  1,  2,  2,  1,  2,  3,  4,  3,  2,  7,  4,  3,  4,  2,  3,  3,
         8, 11,  4,  1,  3,  1,  6,  3])
    context_history_embed = encoder(xs_pad, ilens)
    print(f'bilstm context_history_embed: {context_history_embed.shape}')
