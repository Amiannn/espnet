import torch
import logging

from espnet2.asr.decoder.whisper_decoder    import OpenAIWhisperDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

def forward_contextual_adapter(
    contextualizer,
    model_embed,
    context_idxs,
    ilens,
    context_xphone_idxs=None,
    mask=None,
    return_atten=False,
):
    out = contextualizer(
        model_embed=model_embed,
        context_embed=context_idxs,
        context_xphone_embed=context_xphone_idxs,
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
