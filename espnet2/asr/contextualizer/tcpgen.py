import math
import torch
import random
import logging

from typing import Optional, Tuple

class TCPGenPrototype(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        joint_space_size: int,
        attndim: int,
        dropout: float = 0.1,
        pad_value: int = -1,
        **kwargs
    ):
        super().__init__()
        self.vocab_size     = vocab_size
        self.ookBid         = vocab_size
        self.attndim        = attndim
        self.pad_value      = pad_value

        self.ooKBemb        = torch.nn.Embedding(1, decoder_hidden_size)

        self.Qproj_acoustic = torch.nn.Linear(encoder_hidden_size, attndim)
        self.Qproj_semantic = torch.nn.Linear(decoder_hidden_size, attndim)
        self.Kproj          = torch.nn.Linear(decoder_hidden_size, attndim)
        self.Vproj          = torch.nn.Linear(decoder_hidden_size, attndim)
        
        self.gate           = torch.nn.Linear(joint_space_size + attndim, 1)
        self.dbias          = torch.nn.Linear(attndim, joint_space_size)
        self.dropout        = torch.nn.Dropout(dropout)

    def encode_query(self, acoustic: torch.Tensor, semantic: torch.Tensor):
        # B x T x 1 x attndim
        query_acoustic = self.Qproj_acoustic(acoustic).unsqueeze(2)
        # B x 1 x U x attndim
        query_semantic = self.Qproj_semantic(semantic).unsqueeze(1)
        # B x T x U x attndim
        query = query_acoustic + query_semantic
        return query

    def encode_key_value(self, key, value):
        # B x U x C x attndim
        key   = self.dropout(self.Kproj(key))
        # value = self.dropout(self.Vproj(value))
        return key, value

    def attention(self, queries, keys, values, masks):
        atten = torch.einsum("buca,btua->btuc", keys, queries)
        atten = atten / math.sqrt(self.attndim)
        atten.masked_fill_(masks, -1e9)
        atten = torch.nn.functional.softmax(atten, dim=-1)
        
        x = torch.einsum("bucd,btuc->btud", values, atten)
        return x, atten

    def forward_attention(
        self,
        encoder_out: torch.Tensor,
        decoder_in: torch.Tensor,
        dec_embed: torch.Tensor,
        dec_embed_dropout: torch.Tensor,
        masks_mat: torch.Tensor,
        skip_dropout: bool=False
    ):  
        # generate queries
        semantic = dec_embed_dropout(dec_embed(decoder_in)) if not skip_dropout else dec_embed(decoder_in)
        queries = self.encode_query(encoder_out, semantic)

        # generate keys and values
        embs   = torch.cat([dec_embed.weight.data, self.ooKBemb.weight], dim=0)
        keys   = embs[masks_mat]
        values = embs[masks_mat]
        keys, values = self.encode_key_value(keys, values)

        # attention
        masks = (masks_mat == self.pad_value)
        masks = masks.unsqueeze(1).repeat(1, queries.size(1), 1, 1)
        x, atten = self.attention(queries, keys, values, masks)

        # back transform
        B, T, U, C             = atten.shape
        vocab_add_ookB         = self.vocab_size + 1
        vocab_add_ookB_add_pad = vocab_add_ookB + 1
        masks_mat_index        = masks_mat.unsqueeze(1).repeat(1, queries.size(1), 1, 1)
        masks_mat_index[masks] = vocab_add_ookB
        # B x T x U x C -> B x T x U x vocab
        ptr_dist = torch.zeros(
            (B, T, U, vocab_add_ookB_add_pad), 
            dtype=atten.dtype
        ).to(atten.device).scatter_(
            3, masks_mat_index, atten
        )[:, :, :, :vocab_add_ookB]
        # B x T x U x d
        h_ptr = x
        # B x T x U x joint_dim
        dbias = self.dbias(x) 
        return ptr_dist, h_ptr, dbias

    def forward_gate(
        self,
        h_joint: torch.Tensor,
        h_ptr: torch.Tensor,
        masks_gate_mat: torch.Tensor
    ):
        # B x T x U x 1
        gate = torch.sigmoid(
            self.gate(torch.cat((h_joint, h_ptr), dim=-1))
        )
        gate = gate.masked_fill(
            masks_gate_mat.unsqueeze(1).unsqueeze(-1).bool(), 0
        )
        return gate

    def forward_copy_mechanism(
        self,
        model_dist: torch.Tensor,
        ptr_dist: torch.Tensor,
        gate: torch.Tensor,
    ):
        p_not_null         = 1.0 - model_dist[:, :, :, 0:1]
        ptr_dist_fact      = ptr_dist[:, :, :, 1:] * p_not_null
        ptr_gen_complement = (ptr_dist[:, :, :, -1:]) * gate
        p_partial          = ptr_dist_fact[:, :, :, :-1] * gate + model_dist[
            :, :, :, 1:
        ] * (1 - gate + ptr_gen_complement)
        p_final   = torch.cat([model_dist[:, :, :, 0:1], p_partial], dim=-1)
        joint_out = torch.log(p_final + 1e-12)
        return joint_out

if __name__ == '__main__':
    B, U, T = 2, 2, 3
    C = 4

    vocab_size=5
    encoder_hidden_size=256
    decoder_hidden_size=128
    joint_space_size=512
    attndim=64
    dropout=0.1
    pad_value=-1

    tcpgen = TCPGenPrototype(
        vocab_size=vocab_size,
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        joint_space_size=joint_space_size,
        attndim=attndim,
        dropout=dropout,
        pad_value=pad_value
    )

    encoder_out = torch.rand(B, T, encoder_hidden_size)
    decoder_in = torch.randint(0, vocab_size, (B, U))
    dec_embed = torch.nn.Embedding(vocab_size, decoder_hidden_size)
    dec_embed_dropout = torch.nn.Dropout(dropout)

    masks_mat = torch.randint(0, vocab_size + 1, (B, U, C))
    for b in range(B):
        for u in range(U):
            index = random.randint(1, C)
            masks_mat[b, u, index:] = pad_value

    masks_gate_mat = torch.randint(0, 2, (B, U))
    print(f'masks_mat:\n{masks_mat}')
    print()
    print(f'masks_gate_mat:\n{masks_gate_mat}')
    print()
    ptr_dist, h_ptr, dbias = tcpgen.forward_attention(
        encoder_out, 
        decoder_in, 
        dec_embed, 
        dec_embed_dropout,
        masks_mat
    )

    print(f'ptr_dist: {ptr_dist.shape}')
    print(f'h_ptr: {h_ptr.shape}')

    h_joint = torch.randn(B, T, U, joint_space_size)
    gate = tcpgen.forward_gate(h_joint, h_ptr, masks_gate_mat)

    print(f'gate: {gate}')
    print(f'gate: {gate.shape}')

    model_pos  = torch.randn(B, T, U, vocab_size)
    model_dist = torch.nn.functional.softmax(model_pos, dim=-1)
    print(f'model_dist: {model_dist}')
    print(f'model_dist: {model_dist.shape}')

    dist = tcpgen.forward_copy_mechanism(
        model_dist,
        ptr_dist,
        gate
    )

    print(f'dist: {dist}')
    print(f'dist: {dist.shape}')