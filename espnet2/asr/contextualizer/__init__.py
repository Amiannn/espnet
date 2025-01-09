from espnet2.asr.contextualizer.contextual_adapter import (
    ContextualAdapterPrototype,
    ContextualAdapterTransformer,
    ContextualLightAdapterTransformer,
    ContextualXPhoneAdapter,
    GatedContextualAdapterTransformer,
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

from espnet2.asr.contextualizer.prototype.rradapter import RRAdapter

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
    "contextual_adapter_light_transformer_encoder" : ContextualLightAdapterTransformer,
}

CONTEXTUAL_ADAPTER_DECODER = {
    "contextual_adapter_decoder"                   : ContextualAdapterPrototype,
    "contextual_adapter_transformer_decoder"       : ContextualAdapterTransformer,
    "contextual_adapter_light_transformer_decoder" : ContextualLightAdapterTransformer,
    "contextual_xphone_adapter_decoder"            : ContextualXPhoneAdapter,
    "contextual_gated_adapter_transformer_decoder" : GatedContextualAdapterTransformer,
}

CONTEXTUAL_PROTOTYPE = {
    "rradapter": RRAdapter,
}

CONTEXTUALIZERS = {}
CONTEXTUALIZERS.update(CONTEXTUAL_RETRIEVER)
CONTEXTUALIZERS.update(CONTEXTUAL_HISTORY_ADAPTER_ENCODER)
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_ENCODER)
CONTEXTUALIZERS.update(CONTEXTUAL_ADAPTER_DECODER)
CONTEXTUALIZERS.update(CONTEXTUAL_PROTOTYPE)