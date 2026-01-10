# GASTON: Graph-Aware Social Transformer for Online Networks

**Abstract:** Online communities are "digital third places" with unique social norms.Current approaches to modeling these spaces often fail to capture this nuance because they treat communities either as simple buckets of text (text-only models) or as static structural nodes (structure-only models).

This repository contains the implementation of **GASTON**, a heterogeneous graph learning framework designed to capture the essence of online social networks. GASTON models connections between **Users**, **Communities**, and **Text** as distinct entities. Crucially, it employs a **Contrastive Initialization** strategy to pre-train community representations based on user membership patterns, allowing the model to distinguish between communities (e.g., a support group vs. a hate group) based on their social signature before processing any text.

---

## Architecture

GASTON is built upon a **Heterogeneous Graph Transformer (HGT)** backbone. It addresses the "context blindness" of standard language models by fusing semantic content with structural signals.

### Key Components
1.  **Heterogeneous Graph Schema:**
    * **Nodes:** Users, Text (Posts/Comments), Communities.
    * **Edges:** `posted_by`, `posted_in`, `active_in`.
2.  **Contrastive Initialization :**
    * Unlike prior models that initialize community nodes by averaging text embeddings, GASTON initializes them using **Bayesian Personalized Ranking (BPR)**.
    * This optimizes the embedding space by pulling connected user-community pairs closer while pushing disjoint pairs apart.
3.  **Text Encoding:**
    * Text nodes are initialized using **EmbeddingGemma** to capture high-quality semantic representations.
4.  **Dynamic User Aggregation:**
    * User embeddings are dynamically generated during training by aggregating the representations of the communities they are active in.

---

## Repository Structure

The codebase is organized into core architectural components and downstream task evaluations.

### 1. Core Architecture & Pre-training
* `gaston_pretrain_with_presaved_batches.py`: The main entry point for the graph pre-training loop. Implements the multi-task objective: Text Reconstruction + Edge Generation.
* `contrastive_init_embedding_generator.py`: Implements the BPR-based initialization for community nodes.
* `contrastive_init_pretrain_graph_generator.py`: Handles the construction of the heterogeneous graph from raw data.
* `gaston_split_and_save_subgraphs.py`: Utilities for subgraph sampling (NeighborLoader) to handle large-scale graphs.

### 2. Downstream Tasks (Fine-Tuning)
Scripts for fine-tuning the pre-trained GASTON model on specific social tasks:

* **Norm Violation:** `contr_init_normvio_finetune.py` (Detecting rule-breaking comments).
* **Hate Speech:** `contr_init_hateful_finetune.py` (Detecting hateful content in context).
* **Mental Health:** `contr_init_dreaddit_finetune.py` (Stress detection).
* **Toxicity Scoring:** `contr_init_ruddit_finetune.py` (Regression task for toxicity scores).
* **Recommendation:** `contr_init_recommendation_finetune.py` (Link prediction for user-community recommendation).

---

## Setup & Installation

### Environment
This project is managed with `uv` (or standard pip). Key dependencies include PyTorch, PyTorch Geometric, and PyTorch Lightning.

```bash
# Install dependencies
pip install -r pyproject.toml
# OR
uv sync