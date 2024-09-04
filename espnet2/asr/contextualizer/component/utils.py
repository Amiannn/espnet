import math
import torch

from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.conformer.convolution   import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.attention   import (
    MultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.nets_utils import (
    get_activation,
    make_pad_mask,
    trim_by_ctc_posterior,
)

def build_conformer_block(
    proj_hidden_size,
    drop_out
):
    activation = get_activation("swish")
    # encoder_selfattn_layer
    encoder_selfattn_layer = MultiHeadedAttention
    encoder_selfattn_layer_args = (
        4,
        proj_hidden_size,
        drop_out,
    )
    # positionwise_layer
    positionwise_layer = PositionwiseFeedForward
    positionwise_layer_args = (
        proj_hidden_size,
        2048,
        drop_out,
        activation,
    )
    # convolution_layer
    convolution_layer = ConvolutionModule
    convolution_layer_args = (
        proj_hidden_size, 
        31, 
        activation,
    )
    return EncoderLayer(
        proj_hidden_size,
        encoder_selfattn_layer(*encoder_selfattn_layer_args),
        positionwise_layer(*positionwise_layer_args),
        positionwise_layer(*positionwise_layer_args),
        convolution_layer(*convolution_layer_args),
        drop_out,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    )

class CustomLinear(torch.nn.Module):
    def __init__(self, embedding, no_context_embedding):
        super(CustomLinear, self).__init__()
        # Embedding layers
        self.weight1 = embedding
        self.weight2 = no_context_embedding
        # Calculate the output size based on the layers
        output_size = self.weight1.num_embeddings - 1 + self.weight2.out_features
        # Bias parameter
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))
        # Initialize weights and bias
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.kaiming_uniform_(self.weight1.weight, a=math.sqrt(5))
        # Only initialize weight for weight2 if it is an Embedding
        if isinstance(self.weight2, torch.nn.Embedding):
            torch.nn.init.kaiming_uniform_(self.weight2.weight, a=math.sqrt(5))
        # Bias initialization
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight1.weight)
        bound = 1 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float))
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Concatenate the weights along dimension 1
        combined_weight = torch.cat(
            [
                self.weight2.weight.T, 
                self.weight1.weight[1:, :].T
            ], 
            dim=1
        )  # Shape: (hidden_size, vocab_size)
        # Perform matrix multiplication with the combined weight
        out = torch.matmul(x, combined_weight)
        # Add the bias
        out += self.bias
        return out

class InteractionDropout(torch.nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(InteractionDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, A, B):
        # Decide randomly whether to drop A or B, or keep both
        if self.training:  # only apply this during training
            rand_val = torch.rand(1).item()  # generate a random value between 0 and 1
            if rand_val < self.dropout_prob / 2:  # drop A
                A = torch.zeros_like(A)
            elif rand_val < self.dropout_prob:  # drop B
                B = torch.zeros_like(B)
        return A, B

