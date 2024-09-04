import torch

from espnet2.asr.contextualizer.component.abs_interaction import (
    RepresentationBasedInteractorABC,
    LateInteractionBasedInteractorABC,
    MultiLateInteractionBasedInteractorABC,
)

from espnet2.asr.contextualizer.component.utils import InteractionDropout

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

class MaxSimInteractor(LateInteractionBasedInteractorABC):
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
        # (B x T x D), (C x J x D) -> (B x T x C x J)
        scores = torch.einsum('btd,cjd->btcj', query, context)
        # (B x T x C x J) -> (B x T x C)
        scores = torch.max(scores, dim=-1).values
        return scores

class MultiMaxSimInteractor(MultiLateInteractionBasedInteractorABC):
    def __init__(
            self,
            input_dim,
            proj_dim,
            dropout=0.1,
            interaction_dropout=0.5,
            **kwargs
        ):
        super().__init__()

        self.query_proj           = torch.nn.Linear(input_dim, proj_dim)
        self.context_subwrod_proj = torch.nn.Linear(input_dim, proj_dim)
        self.context_phoneme_proj = torch.nn.Linear(input_dim, proj_dim)
        self.dropout              = torch.nn.Dropout(p=dropout)
        self.interaction_dropout  = InteractionDropout(dropout_prob=interaction_dropout)

    def encode_query(self, query):
        query = self.dropout(self.query_proj(query))
        return query
    
    def encode_context_subword_level(self, context):
        context = self.dropout(self.context_subwrod_proj(context))
        return context

    def encode_context_phoneme_level(self, context):
        context = self.dropout(self.context_phoneme_proj(context))
        return context

    def interaction_dropout(self, subword_scores, context_phoneme):
        return self.interaction_dropout(subword_scores, context_phoneme)
    
    def similarity(self, query, context):
        # (B x T x D), (C x J x D) -> (B x T x C x J)
        scores = torch.einsum('btd,cjd->btcj', query, context)
        # (B x T x C x J) -> (B x T x C)
        scores = torch.max(scores, dim=-1).values
        return scores