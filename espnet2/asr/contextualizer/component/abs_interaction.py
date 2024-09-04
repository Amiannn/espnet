import torch
import abc

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

class LateInteractionBasedInteractorABC(torch.nn.Module, abc.ABC):
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

class MultiLateInteractionBasedInteractorABC(torch.nn.Module, abc.ABC):
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
    def encode_context_subword_level(self, context):
        """Encode the context (subword-level) input."""
        raise NotImplementedError("The method `encode_context_subword_level` must be implemented by the subclass.")
    
    @abc.abstractmethod
    def encode_context_phoneme_level(self, context):
        """Encode the context (phoneme-level) input."""
        raise NotImplementedError("The method `encode_context_phoneme_level` must be implemented by the subclass.")
    
    @abc.abstractmethod
    def interaction_dropout(self, subword_scores, context_phoneme):
        """Apply dropout based on interaction between subword scores and context phonemes."""
        raise NotImplementedError("The method `interaction_dropout` must be implemented by the subclass.")
    
    def forward(self, query, context_subword, context_phoneme):
        query           = self.encode_query(query)
        context_subword = self.encode_context_subword_level(context_subword)
        context_phoneme = self.encode_context_phoneme_level(context_phoneme)
        
        self.subword_scores  = self.similarity(query, context_subword)
        self.phoneme_scores  = self.similarity(query, context_phoneme)

        # interaction dropout
        subword_scores, phoneme_scores = self.interaction_dropout(
            self.subword_scores, 
            self.phoneme_scores
        )
        return subword_scores + phoneme_scores