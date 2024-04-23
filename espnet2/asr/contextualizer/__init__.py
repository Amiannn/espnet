from espnet2.asr.contextualizer.contextual_adapter import (
    ContextualAdapterPrototype,
    ContextualAdapterTransformer,
    ContextualConvXPhoneAdapter,
    ContextualConv2XPhoneAdapter,
    ContextualColbertAdapter,
    ContextualXPhoneAdapter,
    ContextualConvXPhoneGatedAdapter,
    ContextualConv2XPhoneGatedAdapter,
)

from espnet2.asr.contextualizer.contextual_retriever import (
    ContextualDotProductRetrieverPrototype,
    ContextualConv2XPhoneDotProductRetriever,
)

CONTEXTUAL_RETRIEVER = {
    "contextual_dotproduct_retriever"             : ContextualDotProductRetrieverPrototype,
    "contextual_conv2_xphone_dotproduct_retriever": ContextualConv2XPhoneDotProductRetriever,
}

CONTEXTUAL_ADAPTER_ENCODER = {
    "contextual_adapter_encoder"                   : ContextualAdapterPrototype,
    "contextual_adapter_transformer_encoder"       : ContextualAdapterTransformer,
    "contextual_colbert_adapter_encoder"           : ContextualColbertAdapter,
    "contextual_xphone_adapter_encoder"            : ContextualXPhoneAdapter,
    "contextual_conv_xphone_adapter_encoder"       : ContextualConvXPhoneAdapter,
    "contextual_conv2_xphone_adapter_encoder"      : ContextualConv2XPhoneAdapter,
    "contextual_conv2_xphone_gated_adapter_encoder": ContextualConv2XPhoneGatedAdapter,
}

CONTEXTUAL_ADAPTER_DECODER = {
    "contextual_adapter_decoder"                   : ContextualAdapterPrototype,
    "contextual_adapter_transformer_decoder"       : ContextualAdapterTransformer,
    "contextual_colbert_adapter_decoder"           : ContextualColbertAdapter,
    "contextual_xphone_adapter_decoder"            : ContextualXPhoneAdapter,
    "contextual_conv_xphone_adapter_decoder"       : ContextualConvXPhoneAdapter,
    "contextual_conv2_xphone_adapter_decoder"      : ContextualConv2XPhoneAdapter,
    "contextual_conv2_xphone_gated_adapter_decoder": ContextualConv2XPhoneGatedAdapter,
}

CONTEXTUALIZERS = {}
CONTEXTUALIZERS.update(CONTEXTUAL_RETRIEVER)
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_ENCODER)
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_DECODER)