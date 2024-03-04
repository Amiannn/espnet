import math
import torch
import random
import logging

from typing import Optional, Tuple

class ContextEncoderBiLSTM(torch.nn.Module):
    def __init__(
        self,
        hidden_size : int,
        output_size : int,
        droup_out   : float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.oovembed  = torch.nn.Embedding(1, hidden_size)
        self.droup_out = torch.nn.Dropout(droup_out)
        self.encoder   = torch.nn.LSTM(
            hidden_size, 
            output_size // 2, 
            1, 
            batch_first=True, 
            bidirectional=True
        )

    def forward(
        self,
        context_embed: torch.Tensor,
    ):
        context_embed, _ = self.encoder(context_embed)
        context_embed    = torch.mean(context_embed, dim=1)
        return context_embed
    