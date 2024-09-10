import abc
import torch
import logging

from espnet2.asr.contextualizer.component.context_encoder import (
    ContextEncoderBiLSTM,
    ContextEncoderTransformer,
    ContextEncoderXPhoneBiLSTM,
    ContextEncoderXPhone,
)

from espnet2.asr.contextualizer.component.interactions import (
    DotProductInteractor,
    MaxSimInteractor,
    MultiMaxSimInteractor,
)

from espnet2.asr.contextualizer.component.utils import build_conformer_block

class ContextualRetriever(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
    
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

class LateInteractionContextualRetriever(ContextualRetriever):
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
        self.retriever = MaxSimInteractor(
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
        return_mean          : bool=False,
        **kwargs
    ):
        # TODO: move mean operation out of the context encoder
        context_mean, context, ilens = self.context_encoder(
            context_subword, 
            context_subword_ilens
        )
        if return_mean:
            context_mean, ilens
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
            return_mean=True,
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