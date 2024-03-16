import torch
import logging

from espnet2.asr.decoder.whisper_decoder    import OpenAIWhisperDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from espnet2.asr.contextualizer.component.context_encoder import (
    ContextEncoderEmbedBiLSTM,
    ContextEncoderEmbedTransformer,
)

def get_embedding_matrix(
    decoder,
    contextualizer,
):
    if isinstance(
        contextualizer.encoder, 
        (ContextEncoderEmbedBiLSTM, ContextEncoderEmbedTransformer)
    ):
        decoder_embed = contextualizer.encoder.embed
    elif isinstance(decoder, OpenAIWhisperDecoder):
        decoder_embed = decoder.decoders.token_embedding
    elif isinstance(decoder.embed, torch.nn.Sequential):
        decoder_embed = decoder.embed[0]
    else:
        decoder_embed = decoder.embed
        
    text_embed_matrix = torch.cat([
        decoder_embed.weight, 
        contextualizer.encoder.oovembed.weight,
    ], dim=0)

    return text_embed_matrix

def forward_contextual_adapter(
    decoder,
    contextualizer,
    model_embed,
    context_idxs,
    ilens,
    mask=None,
    return_atten=False,
):
    text_embed_matrix = get_embedding_matrix(decoder, contextualizer)
    context_embed     = text_embed_matrix[context_idxs]
    
    context_embed.masked_fill_(
        make_pad_mask(
            ilens, context_embed, 1
        ), 0.0
    )

    out = contextualizer(
        model_embed=model_embed,
        context_embed=context_embed,
        ilens=ilens,
        mask=mask,
        return_atten=return_atten,
    )
    return out

if __name__ == '__main__':
    B, T, D = 3, 4, 2
    context_embed = torch.randn(B, T, D)
    print(context_embed)
    ilens = torch.tensor([2, 3, 4]).long()
    context_embed.masked_fill_(
        make_pad_mask(
            ilens, context_embed, 1
        ), 0.0
    )
    print(context_embed)
