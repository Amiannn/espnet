"""Utility functions for Transducer models."""

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


def get_transducer_task_io(
    labels: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ignore_id: int = -1,
    blank_id: int = 0,
):
    """Get Transducer loss I/O.

    Args:
        labels:
        encoder_out_lens:
        ignore_id: Padding symbol ID.
        blank_id: Blank symbol ID.

    Return:
        target:
        t_len:
        u_len:

    """
    device = labels.device

    ys = [y[y != ignore_id] for y in labels]
    target = pad_list(ys, blank_id).type(torch.int32).to(device)

    if encoder_out_lens.dim() > 1:
        enc_mask = [m[m != 0] for m in encoder_out_lens]
        encoder_out_lens = list(map(int, [m.size(0) for m in enc_mask]))
    else:
        encoder_out_lens = list(map(int, encoder_out_lens))

    t_len = torch.IntTensor(encoder_out_lens).to(device)
    u_len = torch.IntTensor([y.size(0) for y in ys]).to(device)

    return target, t_len, u_len
