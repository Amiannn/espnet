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

from espnet2.asr.contextualizer.component.context_history_encoder import (
    ContextHistoryEncoderLSTM,
)

from espnet2.asr.contextualizer.component.attention_based_adapter import (
    AttentionBasedAdapter,
    ConvAttentionAdapter,
    Conv2AttentionAdapter,
)

class ContextualHistoryAdapterPrototype(torch.nn.Module):
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
        self.encoder = ContextEncoderBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_embed_size,
            output_size=context_hidden_size,
            num_blocks=num_blocks,
            drop_out=drop_out,
            padding_idx=padding_idx,
        )
        self.history_encoder = ContextHistoryEncoderLSTM(
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
        self.acoustic_proj = torch.nn.Linear(proj_hidden_size, proj_hidden_size, bias=False) 

    def batch_greedy_mono_decoding(self, context_history_idx, prob, out):
        B, T, U, V = prob.shape
        o          = []
        u          = torch.zeros(B, dtype=torch.long).to(prob.device)
        idx        = torch.zeros(B, T, dtype=torch.long).to(prob.device) + (U - 1)
        b_idx      = torch.arange(B).to(prob.device)
        for t in range(T):
            blank_token_idx = 0
            # Ensure u does not exceed U - 1
            u       = torch.clamp(u, max=U-1)
            valid_u = torch.clamp(u + 1, max=U-1)
            # Gather token indices for the current position in the context history
            token_indices = context_history_idx[b_idx, valid_u]
            # Gather probabilities
            blank_prob = prob[b_idx, t, u, blank_token_idx]
            token_prob = prob[b_idx, t, u, token_indices]
            # Compare probabilities
            update_mask = token_prob > blank_prob
            # Update decoded sequences and u
            o.append(out[b_idx, t, u, :].unsqueeze(1))
            idx[:, t] = u
            u = u + update_mask.long()
            # Stop if all sequences have finished
            if (u + 1 >= U).all():
                break
        o = torch.cat(o, dim=1).to(prob.device)
        return idx, o

    def forward_encoder(
        self,
        text_embed: torch.Tensor,
        ilens     : torch.Tensor,
    ):
        return self.encoder(text_embed, ilens)

    def forward_history_encoder(
        self,
        text_embed: torch.Tensor,
        ilens     : torch.Tensor,
    ):
        return self.history_encoder(text_embed, ilens)

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
        model_embed          : torch.Tensor,
        context_idxs         : torch.Tensor,
        context_history_idx  : torch.Tensor, 
        context_ilens        : torch.Tensor,
        context_history_ilens: torch.Tensor,
        mask                 : torch.Tensor = None,
        **kwargs
    ):
        context_embed_mean, context_embed, ilens = self.forward_encoder(
            context_idxs, 
            context_ilens
        )

        C, D    = context_embed_mean.shape
        B, T, D = model_embed.shape 
        B, U    = context_history_idx.shape
        # add sos token (oov)
        context_history_idx_  = context_history_idx.reshape(B * U)
        context_history_embed = context_embed_mean[context_history_idx_]
        context_history_embed = context_history_embed.reshape(B, U, -1)
        context_history_embed = self.forward_history_encoder(
            context_history_embed, 
            context_history_ilens
        )
        model_proj_embed  = self.acoustic_proj(model_embed)
        model_fused_embed = model_proj_embed.unsqueeze(2) + context_history_embed.unsqueeze(1)
        model_fused_embed = model_fused_embed.reshape(B, T * U, D)
        out, attn = self.forward_adapter(
            model_embed=model_fused_embed,
            context_embed=context_embed_mean,
            context_embed_value=None,
            mask=mask,
            return_atten=True,
        )
        out   = out.reshape(B, T, U, D)
        prob  = attn.reshape(B, -1, T, U, C).mean(dim=1)
        logit = self.adapter.attention_layer.scores.reshape(B, -1, T, U, C).mean(dim=1)
        idx, out = self.batch_greedy_mono_decoding(context_history_idx, prob, out)
        logging.info(f'mono decoding idx:\n{idx}')
        return out, logit, prob

if __name__ == '__main__':
    B, U, T, C = 4, 5, 10, 10

    vocab_size          = 5
    dropout             = 0.0
    encoder_hidden_size = 128
    context_hidden_size = 128
    joint_space_size    = 128
    proj_hidden_size    = 128
    context_embed_size  = 128

    context_adapter_prototype = ContextualHistoryAdapterPrototype(
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
    text_embed = torch.randint(0, vocab_size, [C, U])
    print(f'text_embed: {text_embed.shape}')
    context_ilens = torch.tensor([5, 3, 2, 4, 1, 5, 5, 3, 2, 4])

    context_history_idx = torch.randint(0, C, [B, 5])
    print(f'context_history_idx: {context_history_idx.shape}')
    context_history_ilens = torch.tensor([5, 3, 4, 0])

    bias, prob = context_adapter_prototype(
        model_embed=model_out, 
        context_idxs=text_embed, 
        context_ilens=context_ilens,
        context_history_idx=context_history_idx,
        context_history_ilens=context_history_ilens,
    )
    print(f'bias: {bias.shape}')
    print(f'prob: {prob.shape}')