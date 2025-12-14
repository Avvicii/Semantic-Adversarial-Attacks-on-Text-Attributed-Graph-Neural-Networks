# Semantic-Preserving Adversarial Attacks on Text-Attributed Graph Neural Networks

This repository contains the code and experiments for analyzing the robustness of **text-attributed Graph Neural Networks (GNNs)** under **semantic-preserving adversarial attacks**. We study how sentence-level paraphrasing and back-translation perturb node text while preserving semantic meaning, and evaluate their impact on downstream node classification performance.

All attacks are applied **exclusively at test time**, ensuring that training and validation data remain unmodified. This protocol isolates robustness issues arising from embedding-space perturbations rather than model retraining effects.

---

## Task and Dataset

- **Task:** Node classification  
- **Dataset:** OGBN-ArXiv (Hu et al., 2020)  
- **Nodes:** 169,343 Computer Science papers  
- **Edges:** Directed citation links  
- **Classes:** 40 subject areas  
- **Official splits:**
  - Train: 90,941 nodes  
  - Validation: 29,799 nodes  
  - Test: 48,603 nodes  

Each node is associated with textual content consisting of the paper title and abstract.

---

## Text Encoding and GNN Model

Node text is encoded using pretrained sentence embedding models. These embeddings are used as **fixed node features** for a Graph Neural Network trained under clean conditions.

### Sentence Encoders
- `all-MiniLM-L6-v2` (384 dimensions)  
- `all-mpnet-base-v2` (768 dimensions)

### Graph Model
- GNN trained only on clean embeddings  
- No retraining or fine-tuning under adversarial settings  

---

## Adversarial Attacks

We evaluate **semantic-preserving text perturbations** that induce distributional shifts in the embedding space without modifying graph structure or labels.

### Direct Paraphrasing Attack
- Test node text is rewritten using a neural paraphrasing model
- Two attack strengths:
  - **Single-step paraphrasing**
  - **Two-step paraphrasing** (paraphrase of a paraphrase)

### Back-Translation Attack
- Text is translated to a pivot language and then translated back to English
- Pivot languages evaluated:
  - Chinese (zh)
  - Hindi (hi)
  - German (de)

---

## Attack Application Protocol

- Attacks are applied **exclusively to test nodes**
- Training and validation texts remain unchanged
- Attack generation is **deterministic**, using node IDs as random seeds
- After text modification:
  - All 169,343 documents are re-encoded using the same sentence encoder
  - A **hybrid feature matrix** is constructed:
    - Clean embeddings for training and validation nodes
    - Attacked embeddings for test nodes
- The pre-trained GNN is evaluated directly on this hybrid feature matrix

---

## Evaluation Metrics

Robustness under adversarial conditions is evaluated using the following metrics:

- **Clean Test Accuracy:** Accuracy on the unmodified test set  
- **Attacked Test Accuracy:** Accuracy after applying adversarial perturbations  
- **Accuracy Drop (ΔAcc):** Absolute difference between clean and attacked accuracy  
- **Prediction Flip Rate:** Percentage of test nodes whose predictions change from correct to incorrect under attack  


