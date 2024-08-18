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
        return_model_proj=False,
    ):  
        x1 = self.encode_x1(model_embed)
        x2 = self.encode_x2(context_embed)
        x  = self.similarity(x1, x2)
        x_prob = torch.softmax(x / self.temperature, dim=-1)
        if return_model_proj:
            return x_prob, x1
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

class LateInteractiveRetriever(DotProductRetriever):
    def __init__(
        self,
        input_hidden_size: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )

    def similarity(self, x1, x2):
        # B x T x D, C x J x D -> B x T x C x J
        x = torch.einsum('btd,cjd->btcj', x1, x2)
        # B x T x C x J -> B x T x C
        x = torch.max(x, dim=-1).values
        return x

class LateMultiInteractiveRetriever(LateInteractiveRetriever):
    def __init__(
        self,
        input_hidden_size: int,
        proj_hidden_size: int,
        drop_out: float = 0.1,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(
            input_hidden_size=input_hidden_size,
            proj_hidden_size=proj_hidden_size,
            drop_out=drop_out,
            temperature=temperature,
            **kwargs
        )
        self.query   = torch.nn.Linear(input_hidden_size, proj_hidden_size)
        self.drop    = torch.nn.Dropout(p=drop_out)
        self.proj3   = torch.nn.Linear(input_hidden_size, proj_hidden_size)
        self.drop_x3 = torch.nn.Dropout(p=drop_out)
        self.proj4   = torch.nn.Linear(input_hidden_size, proj_hidden_size)
        self.drop_x4 = torch.nn.Dropout(p=drop_out)
        self.gate    = torch.nn.Parameter(torch.ones(1))

    def encode_query(self, x):
        return self.query(self.drop(x))
    
    def encode_x3(self, x):
        return self.proj3(self.drop_x3(x))

    def encode_x4(self, x):
        return self.proj4(self.drop_x4(x))

    def forward(
        self,
        model_embed,
        context_embed,
        xphone_embed,
        return_model_proj=False,
    ):
        logging.info(f'model_embed  : {model_embed.shape}')
        logging.info(f'context_embed: {context_embed.shape}')
        logging.info(f'xphone_embed : {xphone_embed.shape}')
        
        query     = self.encode_query(model_embed)
        query_sw  = self.encode_x1(query)
        query_pho = self.encode_x3(query)

        key_sw  = self.encode_x2(context_embed)
        key_pho = self.encode_x4(xphone_embed)

        maxsim_sw  = self.similarity(query_sw, key_sw)
        maxsim_pho = self.similarity(query_pho, key_pho)
        # combine
        logging.info(f'maxsim_sw : {maxsim_sw.max()}')
        logging.info(f'maxsim_pho: {maxsim_pho.max()}')
        maxsim = maxsim_sw + maxsim_pho
        prob   = torch.softmax(maxsim, dim=-1)
        if return_model_proj:
            return prob, query
        return prob

class Conv2LateMultiInteractiveRetriever(LateMultiInteractiveRetriever):
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
        # self.bn1x1 = torch.nn.BatchNorm1d(input_hidden_size // downproj_rate)
        # self.bn1   = torch.nn.BatchNorm1d(input_hidden_size // downproj_rate)
        # self.bn2   = torch.nn.BatchNorm1d(input_hidden_size)
    
    def encode_query(self, x):
        # x1 = self.bn1x1(self.conv_1x1(x.transpose(1, 2)))
        # x1 = self.bn1(self.conv_1(x1))
        # x1 = self.bn2(self.conv_2(x1))
        x1 = (self.conv_1x1(x.transpose(1, 2)))
        x1 = (self.conv_1(x1))
        x1 = (self.conv_2(x1))
        x1 = x1.transpose(1, 2)  # Ensuring the dimensions match for residual addition
        return x + x1

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