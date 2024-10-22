from espnet2.asr.contextualizer.contextual_adapter import (
    ContextualAdapterPrototype,
    ContextualAdapterTransformer,
    ContextualConvXPhoneAdapter,
    ContextualConv2XPhoneAdapter,
    ContextualXPhoneAdapter,
    ContextualConvXPhoneGatedAdapter,
    ContextualConv2XPhoneGatedAdapter,
)

from espnet2.asr.contextualizer.contextual_history_adapter import (
    ContextualHistoryAdapterPrototype,
)

from espnet2.asr.contextualizer.contextual_retriever import (
    DotProductContextualRetriever,
    DotProductXPhoneContextualRetriever,
    LateInteractionContextualRetriever,
    xLateInteractionContextualRetriever,
    MultiLateInteractionContextualRetriever,
)

CONTEXTUAL_RETRIEVER = {
    "dotproduct_contextual_retriever"          : DotProductContextualRetriever,
    "dotproduct_xphone_contextual_retriever"   : DotProductXPhoneContextualRetriever,
    "lateinteraction_contextual_retriever"     : LateInteractionContextualRetriever,
    "xlateinteraction_contextual_retriever"    : xLateInteractionContextualRetriever,
    "multilateinteraction_contextual_retriever": MultiLateInteractionContextualRetriever,
}

CONTEXTUAL_HISTORY_ADAPTER_ENCODER = {
    "contextual_history_adapter_encoder": ContextualHistoryAdapterPrototype,
}

CONTEXTUAL_ADAPTER_ENCODER = {
    "contextual_adapter_encoder"                   : ContextualAdapterPrototype,
    "contextual_adapter_transformer_encoder"       : ContextualAdapterTransformer,
    "contextual_xphone_adapter_encoder"            : ContextualXPhoneAdapter,
    "contextual_conv_xphone_adapter_encoder"       : ContextualConvXPhoneAdapter,
    "contextual_conv2_xphone_adapter_encoder"      : ContextualConv2XPhoneAdapter,
    "contextual_conv2_xphone_gated_adapter_encoder": ContextualConv2XPhoneGatedAdapter,
}

CONTEXTUAL_ADAPTER_DECODER = {
    "contextual_adapter_decoder"                   : ContextualAdapterPrototype,
    "contextual_adapter_transformer_decoder"       : ContextualAdapterTransformer,
    "contextual_xphone_adapter_decoder"            : ContextualXPhoneAdapter,
    "contextual_conv_xphone_adapter_decoder"       : ContextualConvXPhoneAdapter,
    "contextual_conv2_xphone_adapter_decoder"      : ContextualConv2XPhoneAdapter,
    "contextual_conv2_xphone_gated_adapter_decoder": ContextualConv2XPhoneGatedAdapter,
}

CONTEXTUALIZERS = {}
CONTEXTUALIZERS.update(CONTEXTUAL_RETRIEVER)
CONTEXTUALIZERS.update(CONTEXTUAL_HISTORY_ADAPTER_ENCODER)
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_ENCODER)
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_DECODER)