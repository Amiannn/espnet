import abc
import math
import torch


from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

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

class ContextEncoderBiLSTM(torch.nn.Module):
    def __init__(
        self,
        vocab_size  : int,
        hidden_size : int,
        output_size : int,
        drop_out   : float = 0.0,
        num_blocks  : int = 1,
        padding_idx : int = -1,
        **kwargs
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.padding_idx = padding_idx
        self.embed       = torch.nn.Embedding(
            vocab_size, 
            hidden_size, 
        )
        self.oov_embed = torch.nn.Linear(hidden_size, 1, bias=False)
        self.pad_embed = torch.nn.Linear(hidden_size, 1, bias=False)
        with torch.no_grad():
            self.pad_embed.weight.fill_(0)

        self.encoder = RNNEncoder(
            input_size=output_size,
            num_layers=num_blocks,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=0.0,
            subsample=None,
            use_projection=False,
        )

    def forward_embed(self, x: torch.Tensor):
        embedding_matrix = torch.cat([
            self.embed.weight, 
            self.oov_embed.weight,
            self.pad_embed.weight,
        ], dim=0)
        out = embedding_matrix[x]
        return out

    def forward(
        self,
        context_embed: torch.Tensor,
        ilens: torch.Tensor,
    ):
        context_embed           = self.forward_embed(context_embed)
        context_embed, ilens, _ = self.encoder(context_embed, ilens)
        
        ilens = ilens.to(context_embed.device)
        context_embed_mean = torch.sum(context_embed, dim=1) / ilens.unsqueeze(1)
        return context_embed_mean, context_embed, ilens

class ContextEncoderXPhoneBiLSTM(ContextEncoderBiLSTM):
    def __init__(
        self,
        vocab_size        : int,
        hidden_size       : int,
        output_size       : int,
        drop_out         : float = 0.0,
        num_blocks        : int = 1,
        padding_idx       : int = -1,
        xphone_hidden_size: int = 768,
        merge_conv_kernel : int = 3,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            output_size=output_size,
            drop_out=drop_out,
            num_blocks=num_blocks,
            padding_idx=padding_idx,
            **kwargs,
        )
        self.drop_out = torch.nn.Dropout(p=drop_out)
        self.norm_x1  = LayerNorm(hidden_size)
        self.norm_x2  = LayerNorm(xphone_hidden_size)
        self.merge_conv_kernel = merge_conv_kernel

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            hidden_size + xphone_hidden_size,
            hidden_size + xphone_hidden_size,
            kernel_size=self.merge_conv_kernel,
            stride=1,
            padding=(self.merge_conv_kernel - 1) // 2,
            groups=hidden_size + xphone_hidden_size,
            bias=True,
        )
        self.proj = torch.nn.Linear(
            hidden_size + xphone_hidden_size, 
            hidden_size
        )
    
    def branch_merge(self, x1, x2):
        # Merge two branches
        x     = torch.cat([x1, x2], dim=-1).unsqueeze(0)
        x_tmp = x.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        return x + x_tmp

    def forward(
        self,
        context_embed: torch.Tensor,
        context_xphone_embed: torch.Tensor,
        ilens: torch.Tensor,
    ):
        context_embed_mean, context_embed, ilens = super().forward(
            context_embed=context_embed,
            ilens=ilens,
        )
        oov_embed = context_embed_mean[:1, :]
        x1_embed  = context_embed_mean[1:, :]
        x2_embed  = context_xphone_embed
        # layer normalize and dropout
        # x1_embed = self.norm_x1(self.drop_out(x1_embed))
        # x2_embed = self.norm_x2(self.drop_out(x2_embed))
        x1_embed = (self.drop_out(x1_embed))
        x2_embed = (self.drop_out(x2_embed))
        x_embed  = self.branch_merge(x1_embed, x2_embed)
        x_embed  = self.drop_out(self.proj(x_embed)).squeeze(0)
        merged_context_embed = torch.cat([oov_embed, x_embed], dim=0)
        return merged_context_embed, context_embed_mean, ilens

class RepresentationBasedInteractorABC(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def similarity(self, query, context):
        """Compute similarity between query and context."""
        raise NotImplementedError("The method `similarity` must be implemented by the subclass.")
    
    @abc.abstractmethod
    def encode_query(self, query):
        """Encode the query input."""
        raise NotImplementedError("The method `encode_query` must be implemented by the subclass.")
    
    @abc.abstractmethod
    def encode_context(self, context):
        """Encode the context input."""
        raise NotImplementedError("The method `encode_context` must be implemented by the subclass.")
    
    def forward(self, query, context):
        query   = self.encode_query(query)
        context = self.encode_context(context)
        scores  = self.similarity(query, context)
        return scores

class DotProductInteractor(RepresentationBasedInteractorABC):
    def __init__(
            self,
            input_dim,
            proj_dim,
            dropout=0.1,
            **kwargs
        ):
        super().__init__()

        self.query_proj   = torch.nn.Linear(input_dim, proj_dim)
        self.context_proj = torch.nn.Linear(input_dim, proj_dim)
        self.dropout      = torch.nn.Dropout(p=dropout)

    def encode_query(self, query):
        query = self.dropout(self.query_proj(query))
        return query
    
    def encode_context(self, context):
        context = self.dropout(self.context_proj(context))
        return context
    
    def similarity(self, query, context):
        # (B x T x D), (C x D) -> (B x T x C)
        scores = torch.einsum('btd,cd->btc', query, context)
        return scores

class ContextualRetriever(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
    
    def softmax(
        self,
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        return self.forward(
            query,
            query_ilens,
            context_subword,      
            context_subword_ilens,
            context_phone,        
            context_phone_ilens,  
            **kwargs
        )

    def log_softmax(
        self,
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        return torch.log(self.forward(
            query,
            query_ilens,
            context_subword,      
            context_subword_ilens,
            context_phone,        
            context_phone_ilens,  
            **kwargs
        ))

    def argmax(
        self, 
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        return torch.argmax(self.forward(
            query,
            query_ilens,
            context_subword,      
            context_subword_ilens,
            context_phone,        
            context_phone_ilens,  
            **kwargs
        ))

    @abc.abstractmethod
    def forward_query_encoder(
        self, 
        query: torch.Tensor, 
        ilens: torch.Tensor, 
        **kwargs
    ):
        """Encode the query input."""
        raise NotImplementedError("The method `forward_query_encoder` must be implemented by the subclass.")
    
    @abc.abstractmethod
    def forward_context_encoder(
        self, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        """Encode the context input."""
        raise NotImplementedError("The method `forward_context_encoder` must be implemented by the subclass.")
    
    @abc.abstractmethod
    def forward_hnc_encoder(
        self, 
        context: torch.Tensor, 
        ilens  : torch.Tensor, 
        **kwargs
    ):
        """Encode the context input for hard negative context mining."""
        raise NotImplementedError("The method `forward_hnc_encoder` must be implemented by the subclass.")

    @abc.abstractmethod
    def forward_retriever(
        self,
        query  : torch.Tensor, 
        context: torch.Tensor, 
        **kwargs
    ):
        """Compute the retrieval based on the encoded context."""
        raise NotImplementedError("The method `forward_retriever` must be implemented by the subclass.")
    
    def forward(
        self, 
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        return_model_proj    : bool = False, 
        **kwargs
    ):
        """Compute the forward pass."""
        query, query_ilens                     = self.forward_query_encoder(query, query_ilens)
        context_subword, context_subword_ilens = self.forward_context_encoder(
            context_subword, 
            context_subword_ilens,
            context_phone,
            context_phone_ilens,
        )

        scores = self.forward_retriever(query, query_ilens, context_subword, context_subword_ilens)
        probs  = torch.softmax(scores, dim=-1)

        if return_model_proj:
            return probs, query
        return probs    

class DotProductContextualRetriever(ContextualRetriever):
    def __init__(
        self,
        vocab_size          : int,
        query_input_dim     : int,
        context_input_dim   : int,
        proj_dim            : int,
        interaction_proj_dim: int,
        dropout             : float = 0.1,
        pad_token_value     : int = -1,
        **kwargs
    ):
        super().__init__()

        self.query_encoder = build_conformer_block(
            proj_hidden_size=query_input_dim,
            drop_out=dropout
        )
        self.context_encoder = ContextEncoderBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_input_dim,
            output_size=proj_dim,
            num_blocks=2,
            drop_out=dropout,
            padding_idx=pad_token_value,
        )
        self.retriever = DotProductInteractor(
            input_dim=proj_dim,
            proj_dim=interaction_proj_dim,
            dropout=dropout,
            **kwargs
        )

    def forward_query_encoder(
        self, 
        query: torch.Tensor, 
        ilens: torch.Tensor, 
        **kwargs
    ):
        query_hat, mask = self.query_encoder(query, mask=None)
        return (query + query_hat), ilens

    def forward_context_encoder(
        self, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        # TODO: move mean operation out of the context encoder
        context, _, ilens = self.context_encoder(context_subword, context_subword_ilens)
        return context, ilens

    def forward_hnc_encoder(
        self, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        return self.forward_context_encoder(
            context_subword, 
            context_subword_ilens,
            context_phone,
            context_phone_ilens,
        )

    def forward_retriever(
        self,
        query        : torch.Tensor,
        query_ilens  : torch.Tensor,
        context      : torch.Tensor, 
        context_ilens: torch.Tensor,
        **kwargs
    ):
        scores = self.retriever(query, context)
        return scores

class DotProductXPhoneContextualRetriever(DotProductContextualRetriever):
    def __init__(
        self,
        vocab_size          : int,
        query_input_dim     : int,
        context_input_dim   : int,
        proj_dim            : int,
        interaction_proj_dim: int,
        dropout             : float = 0.1,
        pad_token_value     : int = -1,
        **kwargs
    ):
        super().__init__(
            vocab_size           = vocab_size,
            query_input_dim      = query_input_dim,
            context_input_dim    = context_input_dim,
            proj_dim             = proj_dim,
            interaction_proj_dim = interaction_proj_dim,
            dropout              = dropout,
            pad_token_value      = pad_token_value,
            **kwargs
        )

        self.context_encoder = ContextEncoderXPhoneBiLSTM(
            vocab_size=vocab_size,
            hidden_size=context_input_dim,
            output_size=proj_dim,
            num_blocks=2,
            drop_out=dropout,
            padding_idx=pad_token_value,
        )

    def forward_context_encoder(
        self, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        **kwargs
    ):
        # TODO: move mean operation out of the context encoder
        context_phone_mean = torch.sum(context_phone, dim=1) / context_phone_ilens.unsqueeze(1)
        
        context, _, ilens = self.context_encoder(
            context_subword, 
            context_phone_mean,
            context_subword_ilens,
        )
        return context, ilens

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

class CTC(torch.nn.Module):
    def __init__(self, embed, oov_embed):
        super(CTC, self).__init__()
        self.ctc_lo = CustomLinear(embed, oov_embed)
    
    def forward(self, x):
        return self.ctc_lo(x)
    
class Contextualizer(torch.nn.Module):
    def __init__(
            self,
            vocab_size=100,
            query_input_dim=80,
            context_input_dim=80,
            proj_dim=80,
            interaction_proj_dim=80,
            dropout=0.1,
            pad_token_value=-1,
        ):
        super(Contextualizer, self).__init__()
        self.contextualizer = DotProductXPhoneContextualRetriever(
            vocab_size=vocab_size,
            query_input_dim=query_input_dim,
            context_input_dim=context_input_dim,
            proj_dim=proj_dim,
            interaction_proj_dim=interaction_proj_dim,
            dropout=dropout,
            pad_token_value=pad_token_value,
        )
        self.ctc = CTC(
            self.contextualizer.context_encoder.embed, 
            self.contextualizer.context_encoder.oov_embed
        )

    def forward(
        self, 
        query                : torch.Tensor, 
        query_ilens          : torch.Tensor, 
        context_subword      : torch.Tensor, 
        context_subword_ilens: torch.Tensor,
        context_phone        : torch.Tensor, 
        context_phone_ilens  : torch.Tensor,
        return_model_proj    : bool = True, 
        **kwargs
    ):
        """Compute the forward pass."""
        probs, query = self.contextualizer(
            query,
            query_ilens,
            context_subword,      
            context_subword_ilens,
            context_phone,        
            context_phone_ilens,  
            return_model_proj=return_model_proj,
            **kwargs
        )
        ctc_out = self.ctc(query)
        return ctc_out, probs
    
    def forward_ctc(self, 
        query      : torch.Tensor, 
        query_ilens: torch.Tensor
    ):
        query, query_ilens = self.forward_query_encoder(query, query_ilens)
        return self.ctc(query)
    
if __name__ == "__main__":
    ckpt_path = "./exp/asr_whisper/run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8_suffix/xdotretriever.pt"
    
    retriever = Contextualizer(
        vocab_size=5000,
        query_input_dim=1024,
        context_input_dim=1024,
        proj_dim=1024,
        interaction_proj_dim=1024,
        dropout=0.0,
        pad_token_value=-1,
    )

    retriever.load_state_dict(torch.load(ckpt_path))
    retriever.eval()

    print(retriever)