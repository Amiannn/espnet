from espnet2.asr.contextualizer.contextual_adapter import (
    ContextualAdapterPrototype,
    ContextualAdapterTransformer,
    ContextualConvXPhoneAdapter,
    ContextualConv2XPhoneAdapter,
    ContextualXPhoneAdapter,
    ContextualConvXPhoneGatedAdapter,
    ContextualConv2XPhoneGatedAdapter,
)

from espnet2.asr.contextualizer.contextual_retriever import (
    ContextualDotProductRetrieverPrototype,
    ContextualConv2XPhoneDotProductRetriever,
    ContextualLateInteractiveRetriever,
    ContextualMultiLateInteractiveRetriever,
    ContextualConv2MultiLateInteractiveRetriever,
    ContextualConv2MultiLateInteractiveDropoutRetriever,
    ContextualConformerMultiLateInteractiveRetriever,
    MultiLateInteractiveContextRetriever,
)

from espnet2.asr.contextualizer.contextual_history_adapter import (
    ContextualHistoryAdapterPrototype,
)

from espnet2.asr.contextualizer.contextual_retriever_new import (
    DotProductContextualRetriever,
    DotProductXPhoneContextualRetriever,
    LateInteractionContextualRetriever,
    LateInteractionXPhoneContextualRetriever,
)

CONTEXTUAL_RETRIEVER = {
    "contextual_dotproduct_retriever"                        : ContextualDotProductRetrieverPrototype,
    "contextual_conv2_xphone_dotproduct_retriever"           : ContextualConv2XPhoneDotProductRetriever,
    "contextual_lateinteractive_retriever"                   : ContextualLateInteractiveRetriever,
    "contextual_multilateinteractive_retriever"              : ContextualMultiLateInteractiveRetriever,
    "contextual_conv2_multilateinteractive_retriever"        : ContextualConv2MultiLateInteractiveRetriever,
    "contextual_conv2_multilateinteractive_dropout_retriever": ContextualConv2MultiLateInteractiveDropoutRetriever,
    "contextual_conformer_multilateinteractive_retriever"    : ContextualConformerMultiLateInteractiveRetriever,
    "multilateinteractive_context_retriever"                 : MultiLateInteractiveContextRetriever,

    # new contextual retriever
    "dotproduct_contextual_retriever"          : DotProductContextualRetriever,
    "dotproduct_xphone_contextual_retriever"   : DotProductXPhoneContextualRetriever,
    "lateinteraction_contextual_retriever"     : LateInteractionContextualRetriever,
    "multilateinteraction_contextual_retriever": LateInteractionXPhoneContextualRetriever,
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