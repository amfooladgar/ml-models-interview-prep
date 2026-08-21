# 🧱 Deep Learning: Foundations

Welcome to the Foundations subcategory of Deep Learning. This section covers the core building blocks of deep artificial neural networks. Mastering these topics is essential for technical interviews, as they test your deep understanding of forward passes, backpropagation, and sequence/spatial modeling.

## Algorithms & Architectures in This Section

| # | Topic | Key Focus | Mathematical Concepts | Difficulty |
|---|-------|-----------|-----------------------|------------|
| 1 | [Multi-Layer Perceptron (MLP)](01-mlp/) | Dense Feedforward Layers | Fully connected layers, Chain rule backpropagation, Activations (ReLU, Sigmoid), Cross-Entropy loss | ⭐ Beginner |
| 2 | [Convolutional Neural Network (CNN)](02-cnn/) | Spatial/Grid Data Processing | 2D Convolution, padding, stride, parameter sharing, translation invariance, Max/Avg Pooling | ⭐⭐ Intermediate |
| 3 | [Recurrent Neural Network (RNN)](03-rnn/) | Sequential/Time-Series Modeling | Recurrent hidden state updates, Backpropagation Through Time (BPTT), vanishing/exploding gradients | ⭐⭐ Intermediate |
| 4 | [Long Short-Term Memory (LSTM)](04-lstm/) | Long-term dependencies | Forget, Input, Output gating, Cell state ($C_t$) vs. Hidden state ($h_t$), additive cell gradient flow | ⭐⭐ Intermediate |
| 5 | [Gated Recurrent Unit (GRU)](05-gru/) | Parameter-efficient sequences | Update and Reset gates, hidden state combination, parameter/gating complexity vs. LSTM | ⭐⭐ Intermediate |
| 6 | [Autoencoder (AE)](06-autoencoder/) | Latent space representations | Encoder-decoder bottleneck, Reconstruction Loss, comparison to PCA (Principal Component Analysis) | ⭐⭐ Intermediate |

---

## 🔑 Key Interview Math & Formulas

### 1. Multi-Layer Perceptron (MLP)
- **Forward Pass**: $a^{[l]} = g^{[l]}(W^{[l]} a^{[l-1]} + b^{[l]})$ where $g$ is the activation.
- **Cross-Entropy Loss (Binary)**: $L = - \frac{1}{m} \sum_{i=1}^m [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]$
- **Chain Rule (Gradient with respect to weights)**: $\frac{\partial L}{\partial W^{[l]}} = dZ^{[l]} (a^{[l-1]})^T$ where $dZ^{[l]} = da^{[l]} * g'^{[l]}(Z^{[l]})$.

### 2. Convolutional Neural Network (CNN)
- **Output Dimension Formula**:
  $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
  where $W$ is input size, $K$ is kernel size, $P$ is padding, and $S$ is stride.
- **Receptive Field ($RF$)**:
  $$RF_{l} = RF_{l-1} + (K_l - 1) \cdot J_{l-1}$$
  where $J_{l-1}$ is the cumulative stride up to layer $l-1$ ($J_{l-1} = \prod_{i=1}^{l-1} S_i$).

### 3. Recurrent Architectures (RNN/LSTM/GRU)
- **Vanilla RNN Hidden State**: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
- **LSTM Forget Gate**: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
- **LSTM Cell State Update**: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
- **GRU Update Gate**: $z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$

### 4. Autoencoder
- **Reconstruction MSE Loss**: $L = \frac{1}{m} \sum_{i=1}^m \|x^{(i)} - \hat{x}^{(i)}\|_2^2$ where $\hat{x} = g(\text{Decoder}(f(\text{Encoder}(x))))$.
