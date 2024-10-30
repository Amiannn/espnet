# Enhanced ASR with Contextualization

This project is a fork of an existing Automatic Speech Recognition (ASR) system, enhanced with contextualization capabilities to improve recognition performance, especially for rare and domain-specific terms. The modifications introduce context-aware components that adapt the ASR model to better handle out-of-vocabulary words and provide more accurate transcriptions in specialized domains.

## Table of Contents

- [Key Features](#key-features)
- [Modified Components](#modified-components)
  - [Contextual ASR Model](#contextual-asr-model)
  - [Contextual Adapters](#contextual-adapters)
  - [Contextual Retrievers](#contextual-retrievers)
  - [Context Sampler](#context-sampler)
  - [Hard Negative Mining](#hard-negative-mining)
  - [Whisper Prompter](#whisper-prompter)
  - [Trie Processor](#trie-processor)
  - [Beam Search Refactoring](#beam-search-refactoring)

## Key Features

- **Contextual Adaptation:** Enhances the ASR model with context-aware components to recognize rare and domain-specific terms.
- **Advanced Contextualizers:** Includes retrievers, adapters, and prompt generators to incorporate context at various stages of the ASR pipeline.
- **Hard Negative Sampling:** Implements hard negative sampling to introduce challenging distractors during training, improving model robustness.
- **Prompt Generation:** Generates NLP-based prompts to bias predictions towards relevant contexts dynamically.
- **Trie-Based Token Matching:** Uses trie structures for efficient token matching and context-based sequence searches.
- **Support for Multiple Models:** Integrates seamlessly with different ASR models, including transformer and transducer architectures.

## Modified Components

### Contextual ASR Model

**File:** `espnet2/asr/contextualized_espnet_model.py`

This custom ASR model builds upon the ESPnet ASR architecture with added contextual biasing. It integrates contextual retrievers, adapters, and prompt generation to improve the recognition of rare and domain-specific terms in speech.

#### Key Modifications:

1. **Contextual Adaptation:**
   - Introduces retrievers and adapters to bias recognition based on relevant subword and phoneme-level contexts.
   - Supports multiple contextualizer types, including retrievers, encoder adapters, and decoder adapters.

2. **Loss Functions for Contextualization:**
   - Implements custom losses like contextual CTC, RNN-T, and reweighted label prior losses.
   - Dynamically adjusts contextualization losses through warm-up mechanisms and loss weighting.

3. **Contextual Prompts Handling:**
   - Utilizes retrieved context hypotheses to generate NLP-based prompts for improved decoding.
   - Updates contexts dynamically during decoding to reflect model predictions.

4. **Advanced Decoding and Loss Management:**
   - Combines contextualization loss with standard CTC and attention-based loss functions.
   - Applies contextualization at both encoder and decoder levels, with bias vectors influencing final predictions.

5. **Transducer Model Integration:**
   - Enhances support for transducer models with contextual bias applied to joint networks.
   - Seamlessly combines bias vectors from encoder and decoder for optimized predictions.

6. **Prompt and Tokenization Support:**
   - Handles Whisper-style text prompts and manages auxiliary tasks for token handling.
   - Includes NLP prompt integration to steer predictions.

### Contextual Adapters

**File:** `espnet2/asr/contextualizer/contextual_adapter.py`

This module provides multiple contextual adapters to improve ASR performance through advanced contextual embedding and attention-based mechanisms. It integrates adapters with phoneme-aware components to bias recognition towards relevant context.

#### Key Features:

1. **Attention-Based Contextual Adapters:**
   - Supports transformer-based, BiLSTM, and phoneme-aware encoders.

2. **Gated Mechanisms for Context Control:**
   - Uses gating to dynamically regulate context influence during inference.

3. **Support for Convolutional and Hybrid Attention Models:**
   - Adapts to complex ASR pipelines with multiple attention mechanisms.

4. **Residual Gate Control:**
   - Balances over-adaptation by regulating residual information flow.

### Contextual Retrievers

**File:** `espnet2/asr/contextualizer/contextual_retriever.py`

This module provides multiple retriever models that enhance ASR systems with context-aware embeddings. It integrates various retrieval strategies to bias recognition towards relevant subword and phoneme contexts.

#### Key Features:

1. **Multi-Strategy Retrieval:**
   - Supports dot-product and late interaction models for efficient context retrieval.

2. **Phoneme-Aware Context Encoding:**
   - Encodes both subword and phoneme-based representations.

3. **Hard Negative Context Mining:**
   - Integrates mining to improve retrieval accuracy through distractors.

4. **Adaptive Query Encoding:**
   - Uses conformer and BiLSTM-based encoders for flexible query projection.

### Context Sampler

**File:** `espnet2/text/contextual/context_sampler.py`

The `ContextSampler` class enhances ASR models by incorporating contextual information during training and inference. It facilitates the sampling and management of context data, including gold contexts, hard negative distractors, and context prompts.

#### Key Features:

1. **Context Sampling:**
   - Samples relevant contexts (gold contexts) and introduces distractors.
   - Supports dropout for variability and integrates hard negative sampling through `HardNegativeSampler`.

2. **Auxiliary Loss Labels Construction:**
   - Generates CTC-based and occurrence-based auxiliary labels for utterance-wise and batch-wise operations.

3. **Prompt Construction:**
   - Creates NLP-based prompts using `WhisperPrompter` to bias predictions toward relevant contexts.
   - Supports both context-present and context-absent prompt templates.

4. **Context Embedding Management:**
   - Loads and processes context phone embeddings for use as model inputs or auxiliary tasks.
   - Aligns and pads embeddings for efficient batch processing.

5. **Special Structure Support:**
   - Manages trie structures via `TrieProcessor` for fast token matching and sequence searches.

### Hard Negative Mining

**File:** `espnet2/text/contextual/sampler/hard_negative_mining.py`

This module implements hard negative sampling to enhance ASR systems by introducing challenging distractors during training and inference. It supports various sampling strategies to improve model robustness for rare word recognition.

#### Key Features:

1. **ANN and Q-HNW Sampling Methods:**
   - Uses Approximate Nearest Neighbors (ANN) and Query-based Hard Negative Word (Q-HNW) techniques.

2. **Phoneme-Aware Mean Pooling:**
   - Extracts phoneme embeddings for enhanced context biasing.

3. **Adaptive Indexing:**
   - Builds FAISS-based indices for efficient contextual retrieval.

4. **GPU Support:**
   - Optional GPU acceleration for faster processing.

### Whisper Prompter

**File:** `espnet2/text/contextual/prompt/prompter.py`

This module generates dynamic prompts for contextualized ASR tasks using Whisper token converters. It builds both training and inference prompts based on context elements and templates.

#### Key Features:

1. **Training Prompt Generation:**
   - Constructs prompts with context elements, optionally including confidence scores and positions.

2. **Inference Prompt Templates:**
   - Provides templates for both context-aware and context-free scenarios.

3. **Context Shuffling:**
   - Supports random shuffling of context elements to diversify prompts.

### Trie Processor

**File:** `espnet2/text/contextual/structure/trie.py`

This module constructs and manages trie structures to facilitate token matching for contextual ASR systems. It supports batch and sequence-wise searches with features for efficient caching and context-based mask generation.

#### Key Features:

1. **Trie-Based Token Matching:**
   - Builds and searches trie structures for fast sequence matching.

2. **Batch and Sequence-Wise Searches:**
   - Supports efficient searches for multiple sequences with mask generation.

3. **Cache Management:**
   - Caches subword paths to optimize search performance.

4. **Integration with Transducer Models:**
   - Adds optional transducer-compatible `<blank>` tokens to the search.

### Beam Search Refactoring

**File:** `espnet/nets/beam_search_contextual_refactor.py`

This module refactors the beam search algorithm to integrate with the trie processor for efficient contextual ASR matching.

#### Key Features:

1. **Trie-Based Beam Search:**
   - Incorporates trie structures into the beam search to guide decoding with context.

2. **Context-Aware Decoding:**
   - Adjusts beam scoring based on context matches to improve recognition accuracy.

3. **Performance Optimization:**
   - Enhances decoding speed and efficiency through optimized search strategies.