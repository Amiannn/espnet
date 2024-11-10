# Enhanced ASR with Contextualization

## Introduction

**Enhanced ASR with Contextualization** is a fork from[ ESPNet project](README_espnet.md), augmented with advanced contextualization capabilities. This project aims to improve recognition performance, especially for rare and domain-specific terms, by integrating context-aware components. These enhancements enable the ASR model to better handle out-of-vocabulary words and provide more accurate transcriptions in specialized domains.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Performing Inference](#performing-inference)
- [Modified Components](#modified-components)
  - [Contextual ASR Model](#contextual-asr-model)
  - [Contextual Adapters](#contextual-adapters)
  - [Contextual Retrievers](#contextual-retrievers)
  - [Context Sampler](#context-sampler)
  - [Hard Negative Mining](#hard-negative-mining)
  - [Whisper Prompter](#whisper-prompter)
  - [Trie Processor](#trie-processor)
  - [Contextualized Beam Search for ASR](#contextualized-beam-search-for-asr)
- [Module Interactions](#module-interactions)
  - [Training Time Interactions](#training-time-interactions)
  - [Decoding Time Interactions](#decoding-time-interactions)

## Key Features

- **Contextual Adaptation:** Enhances the ASR model with context-aware components to recognize rare and domain-specific terms more effectively.
- **Advanced Contextualizers:** Incorporates retrievers, adapters, and prompt generators to introduce context at various stages of the ASR pipeline.
- **Hard Negative Sampling:** Implements hard negative sampling to introduce challenging distractors during training, improving model robustness against confusing terms.
- **Dynamic Prompt Generation:** Generates NLP-based prompts in real-time to bias predictions towards relevant contexts.
- **Efficient Token Matching:** Utilizes trie structures for efficient token matching and context-based sequence searches.
- **Multi-Model Support:** Seamlessly integrates with different ASR models, including transformer and transducer architectures, offering flexibility in deployment.

## Usage

### Training the Model
Fill in examples.

## Modified Components

### Contextual ASR Model

**File:** [`espnet2/asr/contextualized_espnet_model.py`](espnet2/asr/contextualized_espnet_model.py)

A custom ASR model built upon the ESPnet architecture with added contextual biasing. It integrates contextual retrievers, adapters, and prompt generation mechanisms to improve recognition of rare and domain-specific terms.

#### Key Modifications:

1. **Contextual Adaptation:**
   - Integrates retrievers and adapters for biasing recognition based on relevant subword and phoneme-level contexts.
   - Supports multiple contextualizer types, including retrievers, encoder adapters, and decoder adapters.

2. **Enhanced Loss Functions:**
   - Implements custom losses such as contextual Connectionist Temporal Classification (CTC), Recurrent Neural Network Transducer (RNN-T), and reweighted label prior losses.
   - Employs dynamic adjustment of contextualization losses through warm-up mechanisms and loss weighting strategies.

3. **Dynamic Context Handling:**
   - Utilizes retrieved context hypotheses to generate NLP-based prompts for improved decoding.
   - Updates contexts dynamically during decoding to reflect evolving model predictions.

4. **Advanced Decoding and Integration:**
   - Combines contextualization loss with standard CTC and attention-based loss functions.
   - Applies contextualization at both encoder and decoder levels, influencing final predictions with bias vectors.

5. **Transducer Model Integration:**
   - Enhances support for transducer models by applying contextual bias to joint networks.
   - Seamlessly merges bias vectors from both encoder and decoder for optimized predictions.

6. **Prompt and Tokenization Support:**
   - Handles Whisper-style text prompts and manages auxiliary tasks for token handling.
   - Integrates NLP prompts to steer predictions towards relevant contexts.

### Contextual Adapters

**File:** [`espnet2/asr/contextualizer/contextual_adapter.py`](espnet2/asr/contextualizer/contextual_adapter.py)

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

**File:** [`espnet2/asr/contextualizer/contextual_retriever.py`](espnet2/asr/contextualizer/contextual_retriever.py)

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

**File:** [`espnet2/text/contextual/context_sampler.py`](espnet2/text/contextual/context_sampler.py)

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

**File:** [`espnet2/text/contextual/sampler/hard_negative_mining.py`](espnet2/text/contextual/sampler/hard_negative_mining.py)

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

**File:** [`espnet2/text/contextual/prompt/prompter.py`](espnet2/text/contextual/prompt/prompter.py)

This module generates dynamic prompts for contextualized ASR tasks using Whisper token converters. It builds both training and inference prompts based on context elements and templates.

#### Key Features:

1. **Training Prompt Generation:**
   - Constructs prompts with context elements, optionally including confidence scores and positions.

2. **Inference Prompt Templates:**
   - Provides templates for both context-aware and context-free scenarios.

3. **Context Shuffling:**
   - Supports random shuffling of context elements to diversify prompts.

### Trie Processor

**File:** [`espnet2/text/contextual/structure/trie.py`](espnet2/text/contextual/structure/trie.py)

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

### Contextualized Beam Search for ASR

**File:** [`espnet/nets/beam_search_contextual_refactor.py`](espnet/nets/beam_search_contextual_refactor.py)

This module extends the beam search algorithm by integrating contextual biasing mechanisms.

#### Key Features:

1. **Contextual Hypotheses Tracking:**
   - Maintains contextual predictions alongside standard ASR hypotheses.

2. **Contextualized Decoder Scorer:**
   - Wraps the decoder with contextual scoring logic using `ContextualizedDecoderScorer`.

3. **Encoder and Decoder Contextualization:**
   - Applies contextualization at both the encoder and decoder stages.

4. **Handling Whisper and NLP Prompts:**
   - Integrates with OpenAI Whisper decoder and manages NLP prompts for initializing context-aware decoding.

5. **Flexible Beam Search Loop:**
   - Modifies the beam search loop to accommodate contextual information.

6. **Support for Multiple Contextualization Strategies:**
   - Adapts to various contextual mechanisms, including retriever and adapter models.

7. **Prompt Generation and Contextual Predictions:**
   - Generates NLP-based prompts from retrieved hypotheses.

8. **Integration with Hard Negative Mining:**
   - Incorporates hard negatives into the retrieval-based contextual scoring.

## Module Interactions

- **Contextual ASR Model:** Serves as the central component, integrating both **Contextual Adapters** and **Contextual Retrievers** to enhance performance using contextual information.
- **Contextual Adapters** & **Contextual Retrievers:** Work in tandem within the model to incorporate context into the recognition process.
- **Context Sampler:** Supplies necessary context data to the model, interacting with:
  - **Hard Negative Sampler:** Introduces challenging distractors during training to improve model robustness.
  - **Whisper Prompter:** Generates NLP-based prompts to bias predictions.
  - **Trie Processor:** Facilitates efficient token matching and context-based sequence searches.

### Additional Notes

- **Hard Negative Sampler**, **Whisper Prompter**, and **Trie Processor** primarily interact with the **Context Sampler**.
- The **Context Sampler** is responsible for supplying context to both the **Contextual ASR Model** during training and the **Contextualized Beam Search** during decoding.
