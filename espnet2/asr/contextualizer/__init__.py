from espnet2.asr.contextualizer.contextual_adapter import (
    ContextualAdapterPrototype,
    ContextualAdapterTransformer,
    ContextualConvAttenAdapter,
    ContextualColbertAdapter,
)

CONTEXTUAL_ADAPTER_ENCODER = {
    "contextual_adapter_encoder"            : ContextualAdapterPrototype,
    "contextual_adapter_transformer_encoder": ContextualAdapterTransformer,
    "contextual_convatten_adapter_encoder"  : ContextualConvAttenAdapter,
    "contextual_colbert_adapter_encoder"    : ContextualColbertAdapter,
}

CONTEXTUAL_ADAPTER_DECODER = {
    "contextual_adapter_decoder"            : ContextualAdapterPrototype,
    "contextual_adapter_transformer_decoder": ContextualAdapterTransformer,
    "contextual_convatten_adapter_decoder"  : ContextualConvAttenAdapter,
    "contextual_colbert_adapter_decoder"    : ContextualColbertAdapter,
}

CONTEXTUALIZERS = {}
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_ENCODER)
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_DECODER)