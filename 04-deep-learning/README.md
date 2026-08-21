# 🧠 Deep Learning

Deep Learning is a subset of machine learning based on artificial neural networks with representation learning. It uses multiple layers of non-linear processing units to extract and transform features from raw inputs, achieving state-of-the-art results in computer vision, natural language processing, speech recognition, and generative AI.

## Subcategories in This Section

This section is split into two phases to build a solid transition from core architectures to advanced model design:

### 🧱 1. [Deep Learning: Foundations](foundations/) — *Phase 5*
Focuses on the bedrock architectures of deep learning. These are the fundamental building blocks for almost all modern models.

| # | Algorithm / Architecture | Key Focus | Difficulty |
|---|--------------------------|-----------|------------|
| 1 | [Multi-Layer Perceptron (MLP)](foundations/01-mlp/) | Feedforward networks, activations, backpropagation | ⭐ Beginner |
| 2 | [Convolutional Neural Network (CNN)](foundations/02-cnn/) | Spatial feature extraction, kernels, pooling, receptive fields | ⭐⭐ Intermediate |
| 3 | [Recurrent Neural Network (RNN)](foundations/03-rnn/) | Sequential processing, hidden state tracking, BPTT | ⭐⭐ Intermediate |
| 4 | [Long Short-Term Memory (LSTM)](foundations/04-lstm/) | Long-term dependencies, gating mechanism, cell state | ⭐⭐ Intermediate |
| 5 | [Gated Recurrent Unit (GRU)](foundations/05-gru/) | Parameter-efficient sequential modeling, reset/update gates | ⭐⭐ Intermediate |
| 6 | [Autoencoder (AE)](foundations/06-autoencoder/) | Bottleneck representations, reconstruction loss, dimensionality reduction | ⭐⭐ Intermediate |

### 🚀 2. Deep Learning: Advanced — *Phase 6 (Coming Soon)*
Covers advanced structures designed for complex modalities like sequence-to-sequence translation, graph-structured data, and probabilistic generative modeling.

- **Transformers** — Multi-head self-attention, positional encoding, encoder-decoder
- **BERT** — Masked language modeling, bidirectional context
- **GPT** — Autoregressive causal language modeling
- **Graph Neural Networks (GNNs)** — Message passing, node/edge embeddings
- **Variational Autoencoders (VAEs)** — Generative latent variables, reparameterization trick

---

## ⚡ Core Concepts to Master for Interviews

1. **Backpropagation**: Understand the chain rule derivation of gradients with respect to weights and biases.
2. **Vanishing and Exploding Gradients**: Know the mathematical causes (e.g., repeating matrix multiplications, saturating activations like sigmoid/tanh) and solutions (residual connections, gating, gradient clipping, proper initialization).
3. **Weight Initialization**: Why symmetric weights fail, and how Xavier (Glorot) and He (Kaiming) initializations prevent activations/gradients from collapsing or exploding.
4. **Regularization**: Dropout (scaling at train vs test), L1/L2 weight decay, Batch Normalization (internal covariate shift mitigation, train vs test behavior).
