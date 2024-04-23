import math
import torch
import random
import logging

from typing import Optional, Tuple

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

class DotProductRetriever(torch.nn.Module):
    def __init__(
        self,
        input_hidden_size: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.proj1   = torch.nn.Linear(input_hidden_size, proj_hidden_size)
        self.proj2   = torch.nn.Linear(input_hidden_size, proj_hidden_size)
        self.drop_x1 = torch.nn.Dropout(p=drop_out)
        self.drop_x2 = torch.nn.Dropout(p=drop_out)
        self.temperature = temperature
    
    def similarity(self, x1, x2):
        # B x T x D, C x D -> B x T x C
        x = torch.einsum('btd,cd->btc', x1, x2)
        return x

    def encode_x1(self, x1):
        return self.proj1(self.drop_x1(x1))

    def encode_x2(self, x2):
        return self.proj2(self.drop_x2(x2))

    def forward(
        self,
        model_embed,
        context_embed,
    ):  
        x1 = self.encode_x1(model_embed)
        x2 = self.encode_x2(context_embed)
        x  = self.similarity(x1, x2)
        x_prob = torch.softmax(x / self.temperature, dim=-1)
        return x_prob

class Conv2DotProductRetriever(DotProductRetriever):
    def __init__(
        self,
        input_hidden_size: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        temperature: float = 1.0,
        downproj_rate: int = 2,
        **kwargs
    ):
        super().__init__(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )
        self.conv_1x1 = torch.nn.Conv1d(
            in_channels=input_hidden_size,
            out_channels=(input_hidden_size // downproj_rate), 
            kernel_size=1, 
        )
        self.conv_1 = torch.nn.Conv1d(
            in_channels=(input_hidden_size // downproj_rate),
            out_channels=(input_hidden_size // downproj_rate), 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.conv_2 = torch.nn.Conv1d(
            in_channels=(input_hidden_size // downproj_rate),
            out_channels=input_hidden_size, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
    
    def forward_conv(self, x):
        x1 = self.conv_1x1(x.transpose(1, 2))
        x1 = self.conv_1(x1)
        x1 = self.conv_2(x1)
        return x + x1.transpose(1, 2)

    def encode_x1(self, x1):
        x1 = self.forward_conv(x1)
        x1 = super().encode_x1(x1)
        return x1

if __name__ == '__main__':
    attention_heads  = 1 
    attndim          = 4
    proj_hidden_size = 3
    drop_out        = 0.1

    # adapter = ColbertAdapter(
    #     attention_heads=attention_heads,
    #     attndim=attndim,
    #     proj_hidden_size=proj_hidden_size,
    #     drop_out=drop_out,
    # )

    B, T, D = 2, 5, attndim
    C, U, D = 4, 3, attndim

    model_embed         = torch.randn(B, T, D)
    context_embed_key   = torch.randn(C, U, D)
    context_embed_value = torch.randn(C, D)

    # output, attn = adapter(
    #     model_embed, 
    #     context_embed_key,
    #     context_embed_value,
    #     return_atten=True
    # )

    # print(f'output: {output.shape}')
    # print(f'attn  : {attn.shape}')
    # With square kernels and equal stride
    m = torch.nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(2, 1))
    input = torch.randn(20, 1, 50, 100)
    output = m(input)
    print(f'output: {output.shape}')