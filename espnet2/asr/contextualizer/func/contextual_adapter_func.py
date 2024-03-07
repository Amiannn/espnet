import torch

from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder

def get_embedding_matrix(
    decoder,
    contextualizer,
):
    if isinstance(decoder, OpenAIWhisperDecoder):
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
    
    out = contextualizer(
        model_embed=model_embed,
        context_embed=context_embed,
        ilens=ilens,
        mask=mask,
        return_atten=return_atten,
    )
    return out