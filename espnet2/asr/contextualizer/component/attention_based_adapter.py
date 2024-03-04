import math
import torch
import random
import logging

from typing import Optional, Tuple

class AttentionBasedAdapter(torch.nn.Module):
    def __init__(
        self,
        model_hidden_size: int,
        context_hidden_size: int,
        proj_hidden_size: int,
        attndim: int,
        droup_out: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.attndim = attndim
        self.Qproj   = torch.nn.Linear(model_hidden_size, self.attndim)
        self.Kproj   = torch.nn.Linear(context_hidden_size, self.attndim)
        self.Vproj   = torch.nn.Linear(context_hidden_size, self.attndim)
        self.proj    = torch.nn.Linear(self.attndim, proj_hidden_size)
        self.droup_out = torch.nn.Dropout(droup_out)

    def attention(
        self, 
        query, 
        key, 
        value, 
        mask=None, 
        return_atten=False
    ):
        query = self.Qproj(query)
        key   = self.Kproj(key)
        value = self.Vproj(value)

        # attention
        scores = torch.einsum("tk,ijk->ijt", key, query) / math.sqrt(query.size(-1))
        if mask is not None:
            mask      = mask.eq(1)
            min_value = torch.finfo(scores.dtype).min
            scores    = scores.masked_fill(mask, min_value)
            attn      = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.droup_out(attn)
        x      = torch.einsum("tk,ijt->ijk", value, p_attn)
        x      = self.proj(x)
        
        if return_atten:
            return x, attn
        return x

    def forward(
        self,
        model_embed,
        context_embed,
        mask=None,
        return_atten=False,
    ):  
        x = self.attention(
            query=model_embed, 
            key=context_embed, 
            value=context_embed,
            mask=mask,
        )
        return x
