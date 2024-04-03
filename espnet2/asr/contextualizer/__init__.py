from espnet2.asr.contextualizer.contextual_adapter import (
    ContextualAdapterPrototype,
    ContextualAdapterTransformer,
    ContextualConvXPhoneAdapter,
    ContextualColbertAdapter,
    ContextualXPhoneAdapter,
)

CONTEXTUAL_ADAPTER_ENCODER = {
    "contextual_adapter_encoder"             : ContextualAdapterPrototype,
    "contextual_adapter_transformer_encoder" : ContextualAdapterTransformer,
    "contextual_colbert_adapter_encoder"     : ContextualColbertAdapter,
    "contextual_xphone_adapter_encoder"      : ContextualXPhoneAdapter,
    "contextual_conv_xphone_adapter_encoder" : ContextualConvXPhoneAdapter,
}

CONTEXTUAL_ADAPTER_DECODER = {
    "contextual_adapter_decoder"             : ContextualAdapterPrototype,
    "contextual_adapter_transformer_decoder" : ContextualAdapterTransformer,
    "contextual_colbert_adapter_decoder"     : ContextualColbertAdapter,
    "contextual_xphone_adapter_decoder"      : ContextualXPhoneAdapter,
    "contextual_conv_xphone_adapter_decoder" : ContextualConvXPhoneAdapter,
}

CONTEXTUALIZERS = {}
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_ENCODER)
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_DECODER)