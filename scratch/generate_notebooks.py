import os
import json

def make_markdown_cell(lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines]
    }

def make_code_cell(lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines]
    }

def save_notebook(filepath, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Generated {filepath}")

# ----------------- MLP Cells -----------------
def get_mlp_cells(is_solution):
    title = "# 🧱 Multi-Layer Perceptron (MLP) — " + ("Solutions" if is_solution else "Practice") + " Notebook"
    desc = "**This notebook contains " + ("complete, verified solutions.**" if is_solution else "guided exercises — implement the # TODO blocks.**")
    difficulty = "**Difficulty**: ⭐ Beginner  \n**Time**: ~45 minutes"
    
    cells = [
        make_markdown_cell([title, "", desc, "", difficulty, "", "---"]),
        make_markdown_cell([
            "## 🎯 Section 1: Overview",
            "",
            "A **Multi-Layer Perceptron (MLP)** is a class of feedforward artificial neural network. It consists of an input layer, one or more hidden layers, and an output layer. Except for the input nodes, each node is a neuron that uses a non-linear activation function. MLP utilizes backpropagation for training the network.",
            "",
            "### Applications",
            "- Tabular data classification and regression",
            "- Simple image classification (e.g. MNIST digit classification)",
            "- Function approximation"
        ]),
        make_markdown_cell([
            "## 📐 Section 2: Math & Intuition",
            "",
            "### Layer Equations",
            "For a layer $l$ with weights $W^{[l]}$, bias $b^{[l]}$, and activation function $g^{[l]}$:",
            "$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$",
            "$$A^{[l]} = g^{[l]}(Z^{[l]})$$",
            "where $A^{[0]} = X$ (input data).",
            "",
            "### Activations",
            "- **Sigmoid**: $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ with derivative $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$",
            "- **ReLU**: $\\text{ReLU}(z) = \\max(0, z)$ with derivative $\\text{ReLU}'(z) = \\mathbb{1}(z > 0)$",
            "",
            "### Binary Cross-Entropy Loss",
            "$$J = -\\frac{1}{m} \\sum_{i=1}^m \\left[ y^{(i)} \\log(a_2^{(i)}) + (1 - y^{(i)}) \\log(1 - a_2^{(i)}) \\right]$$",
            "",
            "### Backpropagation (2-layer MLP)",
            "We want to compute gradients of cost $J$ with respect to parameters $W^{[2]}, b^{[2]}, W^{[1]}, b^{[1]}$:",
            "- Output layer error: $dZ^{[2]} = A^{[2]} - Y$ (assuming sigmoid activation + binary cross-entropy loss)",
            "- Gradients: ",
            "  $$dW^{[2]} = \\frac{1}{m} (A^{[1]})^T dZ^{[2]}$$",
            "  $$db^{[2]} = \\frac{1}{m} \\sum dZ^{[2]}$$",
            "- Hidden layer error:",
            "  $$dZ^{[1]} = (dZ^{[2]} (W^{[2]})^T) * g^{[1]}' (Z^{[1]})$$",
            "- Gradients:",
            "  $$dW^{[1]} = \\frac{1}{m} X^T dZ^{[1]}$$",
            "  $$db^{[1]} = \\frac{1}{m} \\sum dZ^{[1]}$$"
        ]),
        make_markdown_cell(["## 🔧 Section 3: Implementation from Scratch"]),
        make_code_cell([
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from sklearn.datasets import make_moons",
            "from sklearn.model_selection import train_test_split",
            "",
            "np.random.seed(42)",
            "plt.style.use('seaborn-v0_8-whitegrid')",
            "sns.set_palette('husl')",
            "print('MLP Setup complete! ✅')"
        ]),
        make_markdown_cell(["### 3.1 Generate Synthetic Non-linear Data"]),
        make_code_cell([
            "X, y = make_moons(n_samples=500, noise=0.2, random_state=42)",
            "y = y.reshape(-1, 1)",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "",
            "plt.figure(figsize=(8, 6))",
            "plt.scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1], label='Class 0', alpha=0.8)",
            "plt.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1], label='Class 1', alpha=0.8)",
            "plt.title('Synthetic Moons Dataset')",
            "plt.legend()",
            "plt.show()"
        ]),
        make_markdown_cell(["### 3.2 MLP Implementation"]),
        make_code_cell([
            "class MLPFromScratch:",
            "    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):",
            "        self.lr = lr",
            "        # Initialize weights randomly, biases to zero",
            "        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1",
            "        self.b1 = np.zeros((1, hidden_dim))",
            "        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1",
            "        self.b2 = np.zeros((1, output_dim))",
            "        ",
            "    def sigmoid(self, z):",
            "        # TODO: Implement sigmoid" if not is_solution else "        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))",
            "        " if not is_solution else "",
            "    def sigmoid_derivative(self, a):",
            "        # Note: input is already activated 'a'",
            "        # TODO: Implement sigmoid derivative" if not is_solution else "        return a * (1 - a)",
            "        " if not is_solution else "",
            "    def relu(self, z):",
            "        # TODO: Implement relu" if not is_solution else "        return np.maximum(0, z)",
            "        " if not is_solution else "",
            "    def relu_derivative(self, z):",
            "        # Note: input is pre-activated 'z'",
            "        # TODO: Implement relu derivative" if not is_solution else "        return (z > 0).astype(float)",
            "        " if not is_solution else "",
            "    def forward(self, X):",
            "        # TODO: Implement forward pass. Return cache dict.",
            "        # Cache should store: Z1, A1, Z2, A2" if not is_solution else "        Z1 = X @ self.W1 + self.b1\n        A1 = self.relu(Z1)\n        Z2 = A1 @ self.W2 + self.b2\n        A2 = self.sigmoid(Z2)\n        return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}",
            "        " if not is_solution else "",
            "    def backward(self, X, y, cache):",
            "        # TODO: Implement backward pass. Return gradients dict.",
            "        # Gradients: dW2, db2, dW1, db1" if not is_solution else "        m = X.shape[0]\n        A1 = cache['A1']\n        A2 = cache['A2']\n        Z1 = cache['Z1']\n        \n        dZ2 = A2 - y\n        dW2 = (A1.T @ dZ2) / m\n        db2 = np.sum(dZ2, axis=0, keepdims=True) / m\n        \n        dA1 = dZ2 @ self.W2.T\n        dZ1 = dA1 * self.relu_derivative(Z1)\n        dW1 = (X.T @ dZ1) / m\n        db1 = np.sum(dZ1, axis=0, keepdims=True) / m\n        \n        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}",
            "        " if not is_solution else "",
            "    def update_params(self, grads):",
            "        # TODO: Apply gradient descent updates" if not is_solution else "        self.W1 -= self.lr * grads['dW1']\n        self.b1 -= self.lr * grads['db1']\n        self.W2 -= self.lr * grads['dW2']\n        self.b2 -= self.lr * grads['db2']",
            "        " if not is_solution else "",
            "    def compute_loss(self, y, a2):",
            "        # Binary cross entropy loss with clipping to avoid log(0)",
            "        a2 = np.clip(a2, 1e-15, 1 - 1e-15)",
            "        return -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))",
            "        ",
            "    def fit(self, X, y, epochs=1000):",
            "        history = []",
            "        for epoch in range(epochs):",
            "            cache = self.forward(X)",
            "            loss = self.compute_loss(y, cache['A2'])",
            "            grads = self.backward(X, y, cache)",
            "            self.update_params(grads)",
            "            history.append(loss)",
            "        return history",
            "        ",
            "    def predict(self, X):",
            "        cache = self.forward(X)",
            "        return (cache['A2'] >= 0.5).astype(int)"
        ]),
        make_markdown_cell(["### 3.3 Verify Implementation"]),
        make_code_cell([
            "mlp = MLPFromScratch(input_dim=2, hidden_dim=8, output_dim=1, lr=0.1)",
            "if 'TODO' not in mlp.forward.__code__.co_consts:",
            "    loss_history = mlp.fit(X_train, y_train, epochs=2000)",
            "    preds = mlp.predict(X_test)",
            "    acc = np.mean(preds == y_test)",
            "    print(f'Training Final Loss: {loss_history[-1]:.4f}')",
            "    print(f'Test Accuracy: {acc * 100:.2f}%')",
            "    assert acc >= 0.8, 'Accuracy should be at least 80%'",
            "    ",
            "    # Plot loss",
            "    plt.figure(figsize=(6, 4))",
            "    plt.plot(loss_history)",
            "    plt.title('NumPy MLP Convergence')",
            "    plt.xlabel('Epoch')",
            "    plt.ylabel('Loss')",
            "    plt.show()",
            "else:",
            "    print('Skipping test - class functions not yet implemented')"
        ]),
        make_markdown_cell(["## 📦 Section 4: Library Implementation"]),
        make_markdown_cell([
            "Now let's build the equivalent MLP using PyTorch. We will use `nn.Module` to define our model structure, `optim.Adam` to optimize parameters, and write the training loop."
        ]),
        make_code_cell([
            "import torch",
            "import torch.nn as nn",
            "import torch.optim as optim",
            "",
            "class PyTorchMLP(nn.Module):",
            "    def __init__(self, input_dim, hidden_dim, output_dim):",
            "        super().__init__()",
            "        # TODO: Define layers: Linear -> ReLU -> Linear -> Sigmoid" if not is_solution else "        self.net = nn.Sequential(\n            nn.Linear(input_dim, hidden_dim),\n            nn.ReLU(),\n            nn.Linear(hidden_dim, output_dim),\n            nn.Sigmoid()\n        )",
            "        " if not is_solution else "",
            "    def forward(self, x):",
            "        # TODO: Implement forward" if not is_solution else "        return self.net(x)",
            "        " if not is_solution else ""
        ]),
        make_markdown_cell(["### 4.1 PyTorch Training Loop"]),
        make_code_cell([
            "X_train_t = torch.FloatTensor(X_train)",
            "y_train_t = torch.FloatTensor(y_train)",
            "X_test_t = torch.FloatTensor(X_test)",
            "y_test_t = torch.FloatTensor(y_test)",
            "",
            "torch.manual_seed(42)",
            "model = PyTorchMLP(input_dim=2, hidden_dim=8, output_dim=1)",
            "criterion = nn.BCELoss()",
            "optimizer = optim.Adam(model.parameters(), lr=0.1)",
            "",
            "# TODO: Write PyTorch training loop for 200 epochs" if not is_solution else "# Solution training loop\nepochs = 200\nfor epoch in range(epochs):\n    optimizer.zero_grad()\n    outputs = model(X_train_t)\n    loss = criterion(outputs, y_train_t)\n    loss.backward()\n    optimizer.step()",
            "",
            "with torch.no_grad():",
            "    test_outputs = model(X_test_t)",
            "    test_preds = (test_outputs >= 0.5).float()",
            "    test_acc = (test_preds == y_test_t).float().mean().item()",
            "    print(f'PyTorch Test Accuracy: {test_acc * 100:.2f}%')",
            "    assert test_acc >= 0.8, 'PyTorch accuracy should be at least 80%'"
        ]),
        make_markdown_cell(["## 🧪 Section 5: Experiments"]),
        make_markdown_cell([
            "Compare the impact of different activation functions on training convergence."
        ]),
        make_code_cell([
            "# Experiment: ReLU vs Sigmoid vs Tanh on MLP",
            "class PyTorchFlexMLP(nn.Module):",
            "    def __init__(self, act_fn):",
            "        super().__init__()",
            "        self.net = nn.Sequential(",
            "            nn.Linear(2, 16),",
            "            act_fn,",
            "            nn.Linear(16, 1),",
            "            nn.Sigmoid()",
            "        )",
            "    def forward(self, x):",
            "        return self.net(x)",
            "",
            "activations = {",
            "    'ReLU': nn.ReLU(),",
            "    'Sigmoid': nn.Sigmoid(),",
            "    'Tanh': nn.Tanh()",
            "}",
            "",
            "plt.figure(figsize=(10, 6))",
            "for name, act in activations.items():",
            "    torch.manual_seed(42)",
            "    m = PyTorchFlexMLP(act)",
            "    opt = optim.Adam(m.parameters(), lr=0.01)",
            "    crit = nn.BCELoss()",
            "    losses = []",
            "    for epoch in range(100):",
            "        opt.zero_grad()",
            "        out = m(X_train_t)",
            "        l = crit(out, y_train_t)",
            "        l.backward()",
            "        opt.step()",
            "        losses.append(l.item())",
            "    plt.plot(losses, label=name)",
            "",
            "plt.title('Effect of Activation Functions on Convergence')",
            "plt.xlabel('Epoch')",
            "plt.ylabel('Loss')",
            "plt.legend()",
            "plt.show()"
        ]),
        make_markdown_cell(["## ❓ Section 6: Interview Questions"]),
        make_markdown_cell([
            "### Q1: Explain backpropagation and its reliance on the chain rule.",
            "**Answer**:",
            "Backpropagation is the algorithm used to calculate parameters' gradients in neural networks. It works by computing the gradient of the loss function with respect to each weight and bias by applying the chain rule of calculus. The computation starts at the final layer (output) and propagates backwards through the network layers to calculate gradients layer-by-layer.",
            "",
            "### Q2: What is the Vanishing Gradient problem? How do activation functions like ReLU help mitigate it?",
            "**Answer**:",
            "The vanishing gradient problem occurs when gradients of the loss function approach zero as they are backpropagated through many layers, causing the early layers to update very slowly (or not at all). Activation functions like Sigmoid and Tanh saturate for very large positive or negative values, meaning their derivatives go to zero. ReLU mitigates this because its derivative is constant ($1$) for all positive inputs, allowing gradient flow to remain strong through deep layers.",
            "",
            "### Q3: What is a 'dead ReLU' and how do you prevent it?",
            "**Answer**:",
            "A 'dead ReLU' occurs when a neuron gets stuck in the inactive state (outputting $0$) because the input to it is always negative, leading to a gradient of $0$ during backpropagation. Since the gradient is zero, the weights will never update, and the neuron remains 'dead'. This can be prevented by: using Leaky ReLU (which has a small non-zero slope for negative inputs), lower learning rates, or proper weight initialization (such as He initialization).",
            "",
            "### Q4: Explain the difference between Xavier (Glorot) and He (Kaiming) initialization.",
            "**Answer**:",
            "- **Xavier Glorot Initialization** is designed for symmetric activation functions (like Sigmoid and Tanh). It sets weights drawn from a distribution with variance $\\text{Var}(W) = \\frac{2}{N_{\\text{in}} + N_{\\text{out}}}$.",
            "- **He Kaiming Initialization** is designed specifically for non-symmetric, rectified activations (like ReLU). Since ReLU discards half the input variance (inputs $<0$), He initialization accounts for this by drawing weights with variance $\\text{Var}(W) = \\frac{2}{N_{\\text{in}}}$, scaling the weights larger to maintain signal strength.",
            "",
            "### Q5: Why do we need non-linear activation functions in deep networks?",
            "**Answer**:",
            "Without non-linear activation functions, a multi-layer neural network would simply represent a composition of linear transformations. Because a composition of linear transformations is mathematically equivalent to a single linear transformation (i.e. $W_2(W_1 x + b_1) + b_2 = W_{new} x + b_{new}$), a network of any depth without non-linearities would only be able to learn linear decision boundaries, defeating the purpose of deep representations."
        ]),
        make_markdown_cell(["## 🏆 Section 7: Challenge — Multiclass MLP"]),
        make_markdown_cell([
            "**Challenge**: Implement the Softmax activation function and Categorical Cross-Entropy Loss from scratch, and test it on a multi-class dataset."
        ]),
        make_code_cell([
            "def softmax(z):",
            "    # TODO: Implement softmax securely" if not is_solution else "    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))\n    return exp_z / np.sum(exp_z, axis=1, keepdims=True)",
            "    ",
            "def categorical_cross_entropy(y_true, y_pred):",
            "    # TODO: Implement categorical cross entropy loss" if not is_solution else "    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)\n    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]",
            "    ",
            "# Simple check",
            "z_test = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])",
            "s = softmax(z_test)",
            "print('Softmax output (sums to 1 per row):\\n', s)",
            "assert np.allclose(s.sum(axis=1), 1.0)",
            "",
            "y_t = np.array([[0, 0, 1], [1, 0, 0]])",
            "loss_val = categorical_cross_entropy(y_t, s)",
            "print('Categorical CE Loss:', loss_val)",
            "assert loss_val > 0"
        ])
    ]
    return cells

# ----------------- CNN Cells -----------------
def get_cnn_cells(is_solution):
    title = "# 👁️ Convolutional Neural Network (CNN) — " + ("Solutions" if is_solution else "Practice") + " Notebook"
    desc = "**This notebook contains " + ("complete, verified solutions.**" if is_solution else "guided exercises — implement the # TODO blocks.**")
    difficulty = "**Difficulty**: ⭐⭐ Intermediate  \n**Time**: ~60 minutes"
    
    cells = [
        make_markdown_cell([title, "", desc, "", difficulty, "", "---"]),
        make_markdown_cell([
            "## 🎯 Section 1: Overview",
            "",
            "A **Convolutional Neural Network (CNN)** is a class of deep neural network most commonly applied to analyzing visual imagery. Unlike fully connected layers, CNN layers use **convolution operations** that share parameters across space, enabling shift-invariance and spatial hierarchies.",
            "",
            "### Receptive Fields and Architecture",
            "By stacking convolutional layers, the network learns increasingly complex features — from simple edges in early layers to full shapes and objects in deeper layers."
        ]),
        make_markdown_cell([
            "## 📐 Section 2: Math & Intuition",
            "",
            "### Output Dimension Formula",
            "Given an input image of size $W$, kernel size $K$, padding $P$, and stride $S$:",
            "$$O = \\left\\lfloor \\frac{W - K + 2P}{S} \\right\\rfloor + 1$$",
            "",
            "### 2D Convolution Operation (Single Channel)",
            "$$Y_{i, j} = \\sum_{m=0}^{K_h - 1} \\sum_{n=0}^{K_w - 1} X_{i \\cdot S + m, j \\cdot S + n} W_{m, n} + b$$",
            "",
            "### Receptive Field Calculation",
            "$$RF_l = RF_{l-1} + (K_l - 1) \\cdot J_{l-1}$$",
            "where $J_{l-1}$ is the jump/stride up to the previous layer: $J_{l-1} = \\prod_{i=1}^{l-1} S_i$."
        ]),
        make_markdown_cell(["## 🔧 Section 3: Implementation from Scratch"]),
        make_code_cell([
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "np.random.seed(42)",
            "plt.style.use('seaborn-v0_8-whitegrid')",
            "print('CNN Setup complete! ✅')"
        ]),
        make_markdown_cell(["### 3.1 2D Convolution Forward Pass in NumPy"]),
        make_code_cell([
            "def conv2d_forward(X, W, b, stride=1, padding=0):",
            "    \"\"\"",
            "    Forward pass for a 2D convolution layer (single channel).",
            "    X: input matrix of shape (H, W)",
            "    W: kernel weights of shape (Kh, Kw)",
            "    b: bias (scalar)",
            "    \"\"\"",
            "    H, W_in = X.shape",
            "    Kh, Kw = W.shape",
            "    ",
            "    # Apply padding",
            "    if padding > 0:",
            "        X_pad = np.pad(X, padding, mode='constant', constant_values=0)",
            "    else:",
            "        X_pad = X",
            "        ",
            "    H_pad, W_pad = X_pad.shape",
            "    ",
            "    # Compute output dimensions",
            "    H_out = int((H_pad - Kh) / stride) + 1",
            "    W_out = int((W_pad - Kw) / stride) + 1",
            "    ",
            "    out = np.zeros((H_out, W_out))",
            "    ",
            "    # TODO: Implement convolution slicing loop" if not is_solution else "    for i in range(H_out):\n        for j in range(W_out):\n            h_start = i * stride\n            h_end = h_start + Kh\n            w_start = j * stride\n            w_end = w_start + Kw\n            \n            slice_X = X_pad[h_start:h_end, w_start:w_end]\n            out[i, j] = np.sum(slice_X * W) + b",
            "    ",
            "    return out"
        ]),
        make_markdown_cell(["### 3.2 Max Pooling Forward Pass in NumPy"]),
        make_code_cell([
            "def maxpool_forward(X, pool_size=2, stride=2):",
            "    \"\"\"",
            "    Forward pass for Max Pooling 2D layer.",
            "    \"\"\"",
            "    H, W = X.shape",
            "    H_out = int((H - pool_size) / stride) + 1",
            "    W_out = int((W - pool_size) / stride) + 1",
            "    ",
            "    out = np.zeros((H_out, W_out))",
            "    ",
            "    # TODO: Implement Max Pooling slicing loop" if not is_solution else "    for i in range(H_out):\n        for j in range(W_out):\n            h_start = i * stride\n            h_end = h_start + pool_size\n            w_start = j * stride\n            w_end = w_start + pool_size\n            \n            slice_X = X[h_start:h_end, w_start:w_end]\n            out[i, j] = np.max(slice_X)",
            "            ",
            "    return out"
        ]),
        make_markdown_cell(["### 3.3 Verify NumPy Implementations"]),
        make_code_cell([
            "X_test = np.array([",
            "    [1, 2, 3, 0],",
            "    [0, 1, 2, 1],",
            "    [2, 1, 1, 0],",
            "    [0, 1, 0, 1]",
            "])",
            "W_test = np.array([",
            "    [1, 0],",
            "    [0, 1]",
            "])",
            "bias = 1.0",
            "",
            "c_out = conv2d_forward(X_test, W_test, bias, stride=1, padding=0)",
            "print('Conv Output:\\n', c_out)",
            "if 'TODO' not in conv2d_forward.__code__.co_consts:",
            "    assert c_out.shape == (3, 3)",
            "    assert c_out[0, 0] == 3.0  # (1*1 + 2*0 + 0*0 + 1*1) + 1",
            "    print('Convolution verification passed! ✅')",
            "",
            "p_out = maxpool_forward(X_test, pool_size=2, stride=2)",
            "print('Pooling Output:\\n', p_out)",
            "if 'TODO' not in maxpool_forward.__code__.co_consts:",
            "    assert p_out.shape == (2, 2)",
            "    assert p_out[0, 0] == 2.0",
            "    print('Max Pooling verification passed! ✅')"
        ]),
        make_markdown_cell(["## 📦 Section 4: Library Implementation"]),
        make_markdown_cell([
            "We will build a convolutional neural network (CNN) in PyTorch to classify simulated simple image representations."
        ]),
        make_code_cell([
            "import torch",
            "import torch.nn as nn",
            "",
            "class PyTorchCNN(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        # TODO: Define Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear" if not is_solution else "        self.features = nn.Sequential(\n            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),\n            nn.ReLU(),\n            nn.MaxPool2d(kernel_size=2, stride=2)\n        )\n        self.classifier = nn.Linear(4 * 14 * 14, 10)",
            "        ",
            "    def forward(self, x):",
            "        # TODO: Implement forward pass" if not is_solution else "        x = self.features(x)\n        x = x.view(x.size(0), -1)\n        x = self.classifier(x)\n        return x",
            "        "
        ]),
        make_markdown_cell(["### 4.1 Verification on Dummy Batch"]),
        make_code_cell([
            "model = PyTorchCNN()",
            "dummy_batch = torch.randn(8, 1, 28, 28)  # batch size of 8, single-channel 28x28 images",
            "outputs = model(dummy_batch)",
            "print('Output shape (should be [8, 10]):', list(outputs.shape))",
            "assert list(outputs.shape) == [8, 10]",
            "print('PyTorch CNN compilation check successful! ✅')"
        ]),
        make_markdown_cell(["## 🧪 Section 5: Experiments"]),
        make_markdown_cell([
            "Visualizing the output of a convolution filter reveals how it extracts edge features."
        ]),
        make_code_cell([
            "# Generate dummy image with vertical edge",
            "img = np.zeros((28, 28))",
            "img[:, 14:] = 1.0",
            "",
            "# Vertical edge filter",
            "v_kernel = np.array([",
            "    [-1, 0, 1],",
            "    [-1, 0, 1],",
            "    [-1, 0, 1]",
            "])",
            "",
            "feat_map = conv2d_forward(img, v_kernel, 0.0, stride=1, padding=1)",
            "",
            "fig, axes = plt.subplots(1, 2, figsize=(10, 5))",
            "axes[0].imshow(img, cmap='gray')",
            "axes[0].set_title('Original Image')",
            "axes[1].imshow(feat_map, cmap='coolwarm')",
            "axes[1].set_title('Filtered Image (Edge Filter)')",
            "plt.show()"
        ]),
        make_markdown_cell(["## ❓ Section 6: Interview Questions"]),
        make_markdown_cell([
            "### Q1: How do CNNs achieve translational invariance and parameter efficiency?",
            "**Answer**:",
            "- **Parameter Efficiency**: In a convolutional layer, the same kernel weights are reused (slid) across the entire input grid. A 3x3 conv layer has just $9$ weights regardless of the image dimensions, compared to $H \\times W$ weights for fully connected layers.",
            "- **Translational Invariance**: Parameter sharing combined with Pooling (like Max Pooling) ensures that if a feature shifts slightly in position, the activation map still detects it, resulting in robust representations that are invariant to shifts.",
            "",
            "### Q2: Write the formula for receptive field size.",
            "**Answer**:",
            "The receptive field $RF_l$ of layer $l$ is calculated recursively from input to output:",
            "$$RF_l = RF_{l-1} + (K_l - 1) \\cdot J_{l-1}$$",
            "where $K_l$ is the kernel size of layer $l$, and $J_{l-1}$ is the cumulative stride of all preceding layers: $J_{l-1} = \\prod_{i=1}^{l-1} S_i$.",
            "",
            "### Q3: Calculate the parameter count of a convolutional layer.",
            "**Answer**:",
            "Given a layer with $C_{\\text{in}}$ input channels, $C_{\\text{out}}$ output channels, kernel dimensions $K_h \\times K_w$, and bias enabled:",
            "$$\\text{Parameters} = C_{\\text{out}} \\times (C_{\\text{in}} \\times K_h \\times K_w + 1)$$",
            "For example, a layer with $3$ input channels, $16$ output channels, and a $3 \\times 3$ kernel has: $16 \\times (3 \\times 3 \\times 3 + 1) = 16 \\times 28 = 448$ parameters.",
            "",
            "### Q4: What is the difference between Dilated Convolution and Standard Convolution?",
            "**Answer**:",
            "Dilated convolutions introduce 'holes' (dilation rate $D$) in the kernel, skipping pixels. For a dilation rate $D=2$, the kernel elements are spaced 1 pixel apart. This increases the kernel's receptive field size without adding any parameter count or computational cost."
        ]),
        make_markdown_cell(["## 🏆 Section 7: Challenge — Conv2D Backward Pass"]),
        make_markdown_cell([
            "**Challenge**: Implement the gradient calculation of a 2D convolution layer with respect to its weights ($dW$)."
        ]),
        make_code_cell([
            "def conv2d_backward_W(X, dZ, W_shape):",
            "    \"\"\"",
            "    Compute gradient of loss with respect to weights dW.",
            "    X: Input feature map of shape (H, W)",
            "    dZ: Gradient of loss with respect to layer output of shape (H_out, W_out)",
            "    W_shape: tuple (Kh, Kw) representing kernel shape",
            "    \"\"\"",
            "    Kh, Kw = W_shape",
            "    dW = np.zeros(W_shape)",
            "    ",
            "    # TODO: Implement dW computation" if not is_solution else "    H_out, W_out = dZ.shape\n    for m in range(Kh):\n        for n in range(Kw):\n            # Slide across input matching the window slice\n            dW[m, n] = np.sum(X[m:m+H_out, n:n+W_out] * dZ)",
            "            ",
            "    return dW",
            "",
            "# Check",
            "X_val = np.ones((5, 5))",
            "dZ_val = np.ones((3, 3))",
            "dW = conv2d_backward_W(X_val, dZ_val, (3, 3))",
            "print('dW matrix:\\n', dW)",
            "if 'TODO' not in conv2d_backward_W.__code__.co_consts:",
            "    assert dW.shape == (3, 3)",
            "    assert np.all(dW == 9.0)  # Each cell is sum of 3x3 ones (9.0)"
        ])
    ]
    return cells

# ----------------- RNN Cells -----------------
def get_rnn_cells(is_solution):
    title = "# 🔄 Recurrent Neural Network (RNN) — " + ("Solutions" if is_solution else "Practice") + " Notebook"
    desc = "**This notebook contains " + ("complete, verified solutions.**" if is_solution else "guided exercises — implement the # TODO blocks.**")
    difficulty = "**Difficulty**: ⭐⭐ Intermediate  \n**Time**: ~45 minutes"
    
    cells = [
        make_markdown_cell([title, "", desc, "", difficulty, "", "---"]),
        make_markdown_cell([
            "## 🎯 Section 1: Overview",
            "",
            "A **Recurrent Neural Network (RNN)** is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior, keeping a memory (hidden state) of previous sequence tokens.",
            "",
            "### Applications",
            "- Sentiment analysis",
            "- Time-series forecasting",
            "- Language translation (sequence-to-sequence)"
        ]),
        make_markdown_cell([
            "## 📐 Section 2: Math & Intuition",
            "",
            "### RNN Cell Forward Formulation",
            "At time step $t$, given input $x_t$ and previous hidden state $h_{t-1}$:",
            "$$a_t = x_t W_x + h_{t-1} W_h + b$$",
            "$$h_t = \\tanh(a_t)$$",
            "where $h_0$ is typically initialized to zero.",
            "",
            "### Backpropagation Through Time (BPTT)",
            "RNNs backpropagate through time by unrolling the model across sequence length $T$. Gradients are accumulated across all time steps. However, multiplying by $W_h$ repeatedly at each step leads to gradients shrinking (vanishing) or growing exponentially (exploding)."
        ]),
        make_markdown_cell(["## 🔧 Section 3: Implementation from Scratch"]),
        make_code_cell([
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "np.random.seed(42)",
            "plt.style.use('seaborn-v0_8-whitegrid')",
            "print('RNN Setup complete! ✅')"
        ]),
        make_markdown_cell(["### 3.1 RNN Single-Step Forward Pass"]),
        make_code_cell([
            "def rnn_cell_forward(xt, h_prev, Wx, Wh, b):",
            "    \"\"\"",
            "    xt: input vector at time t of shape (batch_size, input_dim)",
            "    h_prev: hidden state of previous step of shape (batch_size, hidden_dim)",
            "    Wx: weight matrix for inputs of shape (input_dim, hidden_dim)",
            "    Wh: weight matrix for hidden state of shape (hidden_dim, hidden_dim)",
            "    b: bias vector of shape (1, hidden_dim)",
            "    \"\"\"",
            "    # TODO: Implement the tanh update rule" if not is_solution else "    h_next = np.tanh(xt @ Wx + h_prev @ Wh + b)",
            "    return h_next"
        ]),
        make_markdown_cell(["### 3.2 RNN Full Sequence Forward Pass"]),
        make_code_cell([
            "def rnn_forward(X, h0, Wx, Wh, b):",
            "    \"\"\"",
            "    X: sequence matrix of shape (seq_len, batch_size, input_dim)",
            "    h0: initial hidden state of shape (batch_size, hidden_dim)",
            "    \"\"\"",
            "    seq_len, batch_size, input_dim = X.shape",
            "    hidden_dim = h0.shape[1]",
            "    ",
            "    h_states = np.zeros((seq_len, batch_size, hidden_dim))",
            "    h_curr = h0",
            "    ",
            "    # TODO: Loop over sequence length and perform recurrent updates" if not is_solution else "    for t in range(seq_len):\n        h_curr = rnn_cell_forward(X[t], h_curr, Wx, Wh, b)\n        h_states[t] = h_curr",
            "        ",
            "    return h_states"
        ]),
        make_markdown_cell(["### 3.3 Verify Forward Passes"]),
        make_code_cell([
            "X_dummy = np.random.randn(5, 2, 3)  # seq_len=5, batch_size=2, input_dim=3",
            "h0_dummy = np.zeros((2, 4))         # hidden_dim=4",
            "Wx_dummy = np.random.randn(3, 4)",
            "Wh_dummy = np.random.randn(4, 4)",
            "b_dummy = np.zeros((1, 4))",
            "",
            "h_all = rnn_forward(X_dummy, h0_dummy, Wx_dummy, Wh_dummy, b_dummy)",
            "print('RNN states output shape (should be [5, 2, 4]):', list(h_all.shape))",
            "if 'TODO' not in rnn_cell_forward.__code__.co_consts and 'TODO' not in rnn_forward.__code__.co_consts:",
            "    assert list(h_all.shape) == [5, 2, 4]",
            "    print('RNN verification passed! ✅')"
        ]),
        make_markdown_cell(["## 📦 Section 4: Library Implementation"]),
        make_markdown_cell([
            "We will define a sequence forecasting model using PyTorch's native `nn.RNN` module."
        ]),
        make_code_cell([
            "import torch",
            "import torch.nn as nn",
            "",
            "class PyTorchRNN(nn.Module):",
            "    def __init__(self, input_dim, hidden_dim, output_dim):",
            "        super().__init__()",
            "        # TODO: Define nn.RNN and classification layer" if not is_solution else "        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)\n        self.fc = nn.Linear(hidden_dim, output_dim)",
            "        ",
            "    def forward(self, x):",
            "        # x shape: (batch, seq_len, input_dim)",
            "        # TODO: Forward through rnn, take last hidden state, predict" if not is_solution else "        out, h_n = self.rnn(x)\n        # Take the output of the last sequence step\n        last_step = out[:, -1, :]\n        return self.fc(last_step)",
            "        "
        ]),
        make_code_cell([
            "model = PyTorchRNN(input_dim=5, hidden_dim=10, output_dim=2)",
            "dummy_seq = torch.randn(8, 15, 5)  # batch size of 8, sequence length of 15, input dimensions of 5",
            "preds = model(dummy_seq)",
            "print('Output shape (should be [8, 2]):', list(preds.shape))",
            "assert list(preds.shape) == [8, 2]",
            "print('PyTorch RNN compilation check passed! ✅')"
        ]),
        make_markdown_cell(["## 🧪 Section 5: Experiments"]),
        make_markdown_cell([
            "Demonstrate the exploding gradient problem: when training an RNN on extremely long sequences, track the gradient norm."
        ]),
        make_code_cell([
            "# Exploding gradient simulation",
            "seq_len = 150",
            "torch.manual_seed(0)",
            "rnn = nn.RNN(input_size=1, hidden_size=1, num_layers=1, bias=False)",
            "with torch.no_grad():",
            "    # Set weight to a value > 1.0",
            "    rnn.weight_hh_l0.fill_(1.5)",
            "    rnn.weight_ih_l0.fill_(1.0)",
            "",
            "x = torch.ones(seq_len, 1, 1, requires_grad=True)",
            "h0 = torch.zeros(1, 1, 1)",
            "out, hn = rnn(x, h0)",
            "",
            "# Gradient of final hidden state w.r.t initial input",
            "hn.backward()",
            "grad_norm = x.grad.clone().squeeze()",
            "",
            "plt.figure(figsize=(8, 4))",
            "plt.plot(grad_norm.numpy(), color='red')",
            "plt.yscale('log')",
            "plt.title('Exploding Gradients in Vanilla RNN (Log Scale)')",
            "plt.xlabel('Sequence Step')",
            "plt.ylabel('Gradient Norm')",
            "plt.show()",
            "print(f'Gradient value at earliest sequence step: {grad_norm[0].item():.4e}')"
        ]),
        make_markdown_cell(["## ❓ Section 6: Interview Questions"]),
        make_markdown_cell([
            "### Q1: Why do standard RNNs suffer from vanishing and exploding gradients?",
            "**Answer**:",
            "During Backpropagation Through Time (BPTT), the loss gradient at step $T$ is backpropagated to step $0$ by repeatedly multiplying by the recurrent weight matrix transpose $(W_{hh})^T$. If the largest eigenvalue of $W_{hh}$ is $> 1.0$, the gradient grows exponentially ($1.5^{150}$), causing exploding gradients. If the largest eigenvalue is $< 1.0$, the gradient decays exponentially to zero, preventing the weights from learning long-term dependencies.",
            "",
            "### Q2: Explain Backpropagation Through Time (BPTT).",
            "**Answer**:",
            "BPTT is the standard backpropagation algorithm applied to sequential data. The RNN architecture is unrolled through all time steps of the sequence. The forward pass is computed for all steps, storing outputs and hidden states. In the backward pass, gradients are computed starting from the loss at the final time step and backpropagated backward through time, accumulating weight adjustments across all temporal steps.",
            "",
            "### Q3: What is Teacher Forcing and when is it used?",
            "**Answer**:",
            "Teacher Forcing is a training method for recurrent networks where the model receives the ground-truth target sequence token as input at the next step, rather than feeding its own predicted output back into itself. This speeds up training and keeps the model stable early on, but can lead to 'exposure bias' during inference when the ground truth is unavailable.",
            "",
            "### Q4: What is the difference between autoregressive and non-autoregressive decoding?",
            "**Answer**:",
            "- **Autoregressive decoding** generates tokens sequentially, where each output token relies on the previously generated tokens. Standard RNNs and GPT-style models use this.",
            "- **Non-autoregressive decoding** attempts to predict all target outputs in parallel, which is faster but struggles to capture dependency correlations between output tokens."
        ]),
        make_markdown_cell(["## 🏆 Section 7: Challenge — BPTT Cell Gradient"]),
        make_markdown_cell([
            "**Challenge**: Derive and implement the parameter gradient updates for a single RNN cell backward step."
        ]),
        make_code_cell([
            "def rnn_cell_backward(dnext, xt, h_prev, h_curr, Wx, Wh):",
            "    \"\"\"",
            "    Compute gradients for a single step of the RNN cell.",
            "    dnext: gradient of loss w.r.t current hidden state (batch_size, hidden_dim)",
            "    xt: input at current step (batch_size, input_dim)",
            "    h_prev: hidden state of previous step (batch_size, hidden_dim)",
            "    h_curr: activated current hidden state (batch_size, hidden_dim)",
            "    \"\"\"",
            "    # Gradient of tanh: dtanh = (1 - tanh²)",
            "    # TODO: Implement gradient calculations w.r.t weights" if not is_solution else "    dtanh = (1 - h_curr ** 2) * dnext\n    \n    dWx = xt.T @ dtanh\n    dWh = h_prev.T @ dtanh\n    db = np.sum(dtanh, axis=0, keepdims=True)\n    dprev = dtanh @ Wh.T",
            "    ",
            "    return dWx, dWh, db, dprev",
            "",
            "# Check shapes",
            "dnext_v = np.ones((1, 4))",
            "xt_v = np.ones((1, 3))",
            "h_prev_v = np.zeros((1, 4))",
            "h_curr_v = np.ones((1, 4))",
            "Wx_v = np.ones((3, 4))",
            "Wh_v = np.ones((4, 4))",
            "",
            "dWx, dWh, db, dprev = rnn_cell_backward(dnext_v, xt_v, h_prev_v, h_curr_v, Wx_v, Wh_v)",
            "if 'TODO' not in rnn_cell_backward.__code__.co_consts:",
            "    assert dWx.shape == (3, 4)",
            "    assert dWh.shape == (4, 4)",
            "    assert db.shape == (1, 4)",
            "    assert dprev.shape == (1, 4)",
            "    print('BPTT cell backward pass implementation verified! ✅')"
        ])
    ]
    return cells

# ----------------- LSTM Cells -----------------
def get_lstm_cells(is_solution):
    title = "# 🧬 Long Short-Term Memory (LSTM) — " + ("Solutions" if is_solution else "Practice") + " Notebook"
    desc = "**This notebook contains " + ("complete, verified solutions.**" if is_solution else "guided exercises — implement the # TODO blocks.**")
    difficulty = "**Difficulty**: ⭐⭐ Intermediate  \n**Time**: ~60 minutes"
    
    cells = [
        make_markdown_cell([title, "", desc, "", difficulty, "", "---"]),
        make_markdown_cell([
            "## 🎯 Section 1: Overview",
            "",
            "A **Long Short-Term Memory (LSTM)** network is an improved variant of recurrent neural networks. It introduces a **cell state** ($C_t$) that acts as a linear conveyor belt, allowing gradient signals to propagate over long sequences without scaling exponentially.",
            "",
            "### Gating Mechanism",
            "LSTM uses three gates to regulate input/output flows: the forget gate, the input gate, and the output gate."
        ]),
        make_markdown_cell([
            "## 📐 Section 2: Math & Intuition",
            "",
            "### LSTM Gate Equations",
            "Given input $x_t$ and previous hidden state $h_{t-1}$:",
            "1. **Forget Gate**: $f_t = \\sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f)$",
            "2. **Input Gate**: $i_t = \\sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i)$",
            "3. **Candidate Cell State**: $\\tilde{C}_t = \\tanh(x_t W_{xc} + h_{t-1} W_{hc} + b_c)$",
            "4. **Cell State Update**: $C_t = f_t * C_{t-1} + i_t * \\tilde{C}_t$",
            "5. **Output Gate**: $o_t = \\sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o)$",
            "6. **Hidden State**: $h_t = o_t * \\tanh(C_t)$",
            "",
            "where $\\sigma$ is the element-wise sigmoid activation."
        ]),
        make_markdown_cell(["## 🔧 Section 3: Implementation from Scratch"]),
        make_code_cell([
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "np.random.seed(42)",
            "plt.style.use('seaborn-v0_8-whitegrid')",
            "print('LSTM Setup complete! ✅')"
        ]),
        make_markdown_cell(["### 3.1 LSTM Cell Forward Pass"]),
        make_code_cell([
            "def sigmoid(x):",
            "    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))",
            "",
            "def lstm_cell_forward(xt, h_prev, c_prev, W, b):",
            "    \"\"\"",
            "    xt: input vector (batch_size, input_dim)",
            "    h_prev: previous hidden state (batch_size, hidden_dim)",
            "    c_prev: previous cell state (batch_size, hidden_dim)",
            "    W: unified weights containing stacked weights for [f, i, c, o]",
            "       shape is (input_dim + hidden_dim, 4 * hidden_dim)",
            "    b: bias vector of shape (1, 4 * hidden_dim)",
            "    \"\"\"",
            "    batch_size, hidden_dim = h_prev.shape",
            "    ",
            "    # Concatenate inputs",
            "    concat = np.hstack((xt, h_prev))  # shape: (batch_size, input_dim + hidden_dim)",
            "    ",
            "    # Calculate linear combination",
            "    A = concat @ W + b  # shape: (batch_size, 4 * hidden_dim)",
            "    ",
            "    # Slice gates",
            "    f_gate = A[:, :hidden_dim]",
            "    i_gate = A[:, hidden_dim:2*hidden_dim]",
            "    c_cand = A[:, 2*hidden_dim:3*hidden_dim]",
            "    o_gate = A[:, 3*hidden_dim:]",
            "    ",
            "    # TODO: Apply activation functions and compute next cell state & hidden state" if not is_solution else "    f = sigmoid(f_gate)\n    i = sigmoid(i_gate)\n    c_bar = np.tanh(c_cand)\n    o = sigmoid(o_gate)\n    \n    c_next = f * c_prev + i * c_bar\n    h_next = o * np.tanh(c_next)",
            "    ",
            "    return h_next, c_next"
        ]),
        make_markdown_cell(["### 3.2 Verify LSTM Cell"]),
        make_code_cell([
            "xt_v = np.random.randn(2, 3)     # batch_size=2, input_dim=3",
            "h_v = np.zeros((2, 4))           # hidden_dim=4",
            "c_v = np.zeros((2, 4))",
            "W_v = np.random.randn(7, 16)     # (3+4) input+hidden, 4*4 gates",
            "b_v = np.zeros((1, 16))",
            "",
            "h_n, c_n = lstm_cell_forward(xt_v, h_v, c_v, W_v, b_v)",
            "print('Hidden state output shape (should be [2, 4]):', list(h_n.shape))",
            "if 'TODO' not in lstm_cell_forward.__code__.co_consts:",
            "    assert list(h_n.shape) == [2, 4]",
            "    assert list(c_n.shape) == [2, 4]",
            "    print('LSTM Cell forward pass verified! ✅')"
        ]),
        make_markdown_cell(["## 📦 Section 4: Library Implementation"]),
        make_code_cell([
            "import torch",
            "import torch.nn as nn",
            "",
            "class PyTorchLSTM(nn.Module):",
            "    def __init__(self, input_dim, hidden_dim, output_dim):",
            "        super().__init__()",
            "        # TODO: Define LSTM layer and classification layer" if not is_solution else "        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)\n        self.fc = nn.Linear(hidden_dim, output_dim)",
            "        ",
            "    def forward(self, x):",
            "        # TODO: Forward pass" if not is_solution else "        out, (h_n, c_n) = self.lstm(x)\n        # Take the output of the last time step\n        return self.fc(out[:, -1, :])",
            "        "
        ]),
        make_code_cell([
            "model = PyTorchLSTM(input_dim=4, hidden_dim=8, output_dim=1)",
            "dummy_batch = torch.randn(5, 10, 4)  # batch size of 5, seq len of 10, input dims of 4",
            "out = model(dummy_batch)",
            "print('Output shape (should be [5, 1]):', list(out.shape))",
            "assert list(out.shape) == [5, 1]",
            "print('PyTorch LSTM check passed! ✅')"
        ]),
        make_markdown_cell(["## 🧪 Section 5: Experiments"]),
        make_markdown_cell([
            "Compare LSTM vs. standard RNN on a simple memory task. The target is simply the first character in the sequence."
        ]),
        make_code_cell([
            "# Generate memory data: sequences of length 40, classes based on index 0",
            "N = 200",
            "seq_len = 40",
            "X_mem = np.random.randn(N, seq_len, 1)",
            "# Class 1 if first step > 0, else Class 0",
            "y_mem = (X_mem[:, 0, 0] > 0).astype(int)",
            "",
            "X_mem_t = torch.FloatTensor(X_mem)",
            "y_mem_t = torch.FloatTensor(y_mem).view(-1, 1)",
            "",
            "# Training LSTM vs RNN",
            "class SimpleRNN(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        self.rnn = nn.RNN(1, 8, batch_first=True)",
            "        self.fc = nn.Linear(8, 1)",
            "    def forward(self, x):",
            "        out, _ = self.rnn(x)",
            "        return torch.sigmoid(self.fc(out[:, -1, :]))",
            "",
            "class SimpleLSTM(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        self.lstm = nn.LSTM(1, 8, batch_first=True)",
            "        self.fc = nn.Linear(8, 1)",
            "    def forward(self, x):",
            "        out, _ = self.lstm(x)",
            "        return torch.sigmoid(self.fc(out[:, -1, :]))",
            "",
            "models = {'RNN': SimpleRNN(), 'LSTM': SimpleLSTM()}",
            "histories = {}",
            "",
            "for name, m in models.items():",
            "    opt = torch.optim.Adam(m.parameters(), lr=0.02)",
            "    crit = nn.BCELoss()",
            "    losses = []",
            "    for epoch in range(120):",
            "        opt.zero_grad()",
            "        pred = m(X_mem_t)",
            "        l = crit(pred, y_mem_t)",
            "        l.backward()",
            "        opt.step()",
            "        losses.append(l.item())",
            "    histories[name] = losses",
            "",
            "plt.figure(figsize=(8, 5))",
            "for name, losses in histories.items():",
            "    plt.plot(losses, label=name)",
            "plt.title('Memory Task Convergence (Sequence Length = 40)')",
            "plt.xlabel('Epoch')",
            "plt.ylabel('BCE Loss')",
            "plt.legend()",
            "plt.show()"
        ]),
        make_markdown_cell(["## ❓ Section 6: Interview Questions"]),
        make_markdown_cell([
            "### Q1: Explain the role of the cell state ($C_t$) in LSTMs.",
            "**Answer**:",
            "The cell state $C_t$ serves as the memory pathway of the LSTM cell. It has only linear operations (multiplication by forget gate $f_t$ and addition of input candidate $i_t * \\tilde{C}_t$). Because this path is linear, error signals backpropagating along it can flow backward across long sequence steps without undergoing exponential growth or decay, resolving vanishing gradients.",
            "",
            "### Q2: Walk through each of the gates in an LSTM cell.",
            "**Answer**:",
            "- **Forget Gate ($f_t$)**: Outputs values in $[0, 1]$ via Sigmoid, determining how much of the historical cell state $C_{t-1}$ to retain.",
            "- **Input Gate ($i_t$)**: Decides which new input coordinates to update in our memory.",
            "- **Candidate Cell State ($\\tilde{C}_t$)**: Outputs values in $[-1, 1]$ via Tanh, creating candidate values to append to the cell state.",
            "- **Output Gate ($o_t$)**: Determines which parts of the updated cell state to write into the output hidden state $h_t$.",
            "",
            "### Q3: Why does LSTM avoid the vanishing gradient problem?",
            "**Answer**:",
            "In Vanilla RNNs, backpropagating through time requires multiplying by $W_{hh}^T$ at each step. In LSTMs, the gradient flow along the cell state $C_t$ path is controlled by addition and scaling by $f_t$. If the network learns to keep the forget gate $f_t \\approx 1.0$, the gradient is propagated back through time nearly unimpeded, preventing the signal from vanishing.",
            "",
            "### Q4: What is the difference between LSTM cell state and hidden state?",
            "**Answer**:",
            "- **Cell State ($C_t$)**: Represents the internal long-term memory of the unit. It is not exposed to other layers directly.",
            "- **Hidden State ($h_t$)**: Represents the external short-term memory / output of the cell at step $t$. It is computed by taking a non-linear activation of the cell state gated by the output gate."
        ]),
        make_markdown_cell(["## 🏆 Section 7: Challenge — Gating Calculus"]),
        make_markdown_cell([
            "**Challenge**: Derive the gradient of the next cell state ($C_t$) with respect to the forget gate ($f_t$)."
        ]),
        make_markdown_cell([
            "**Mathematical Solution**:",
            "The LSTM update equation for the cell state is:",
            "$$C_t = f_t * C_{t-1} + i_t * \\tilde{C}_t$$",
            "Taking the partial derivative of $C_t$ with respect to $f_t$ (treating other variables as constant local variables at step $t$):",
            "$$\\frac{\\partial C_t}{\\partial f_t} = C_{t-1}$$",
            "This simple derivative shows that the gradient scale directly corresponds to the magnitude of the previous cell state."
        ])
    ]
    return cells

# ----------------- GRU Cells -----------------
def get_gru_cells(is_solution):
    title = "# 🚀 Gated Recurrent Unit (GRU) — " + ("Solutions" if is_solution else "Practice") + " Notebook"
    desc = "**This notebook contains " + ("complete, verified solutions.**" if is_solution else "guided exercises — implement the # TODO blocks.**")
    difficulty = "**Difficulty**: ⭐⭐ Intermediate  \n**Time**: ~45 minutes"
    
    cells = [
        make_markdown_cell([title, "", desc, "", difficulty, "", "---"]),
        make_markdown_cell([
            "## 🎯 Section 1: Overview",
            "",
            "The **Gated Recurrent Unit (GRU)** is a simplified gating mechanism variant of the LSTM. It has fewer parameters because it merges the cell state and hidden state, and combines the forget and input gates into a single **update gate**.",
            "",
            "### Why use GRU?",
            "- Fewer parameters -> less prone to overfitting",
            "- Faster training speed"
        ]),
        make_markdown_cell([
            "## 📐 Section 2: Math & Intuition",
            "",
            "### GRU Cell Equations",
            "Given input $x_t$ and previous hidden state $h_{t-1}$:",
            "1. **Reset Gate**: $r_t = \\sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r)$",
            "2. **Update Gate**: $z_t = \\sigma(x_t W_{xz} + h_{t-1} W_{hz} + b_z)$",
            "3. **Candidate Hidden State**: $\\tilde{h}_t = \\tanh(x_t W_{xh} + (r_t * h_{t-1}) W_{hh} + b_h)$",
            "4. **Hidden State Update**: $h_t = (1 - z_t) * h_{t-1} + z_t * \\tilde{h}_t$",
            "",
            "The reset gate determines how to combine new input with past memory, while the update gate decides how much of the past state to keep."
        ]),
        make_markdown_cell(["## 🔧 Section 3: Implementation from Scratch"]),
        make_code_cell([
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "np.random.seed(42)",
            "plt.style.use('seaborn-v0_8-whitegrid')",
            "print('GRU Setup complete! ✅')"
        ]),
        make_markdown_cell(["### 3.1 GRU Cell Forward Pass"]),
        make_code_cell([
            "def sigmoid(x):",
            "    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))",
            "",
            "def gru_cell_forward(xt, h_prev, Wx, Wh, bx, bh):",
            "    \"\"\"",
            "    xt: input at step t of shape (batch_size, input_dim)",
            "    h_prev: previous hidden state of shape (batch_size, hidden_dim)",
            "    Wx: input weights for [r, z, h] of shape (input_dim, 3 * hidden_dim)",
            "    Wh: hidden weights for [r, z, h] of shape (hidden_dim, 3 * hidden_dim)",
            "    bx: input biases of shape (1, 3 * hidden_dim)",
            "    bh: hidden biases of shape (1, 3 * hidden_dim)",
            "    \"\"\"",
            "    hidden_dim = h_prev.shape[1]",
            "    ",
            "    # Project inputs and hidden states",
            "    X_proj = xt @ Wx + bx",
            "    H_proj = h_prev @ Wh + bh",
            "    ",
            "    # Split projects",
            "    xr, xz, xh = np.split(X_proj, 3, axis=1)",
            "    hr, hz, hh = np.split(H_proj, 3, axis=1)",
            "    ",
            "    # TODO: Implement update equations" if not is_solution else "    r = sigmoid(xr + hr)\n    z = sigmoid(xz + hz)\n    \n    # Candidate hidden state uses reset gate to mask past state\n    h_cand = np.tanh(xh + r * hh)\n    \n    # Update hidden state\n    h_next = (1 - z) * h_prev + z * h_cand",
            "    ",
            "    return h_next"
        ]),
        make_markdown_cell(["### 3.2 Verify GRU Cell"]),
        make_code_cell([
            "xt_v = np.random.randn(2, 3)",
            "h_v = np.zeros((2, 4))",
            "Wx_v = np.random.randn(3, 12)",
            "Wh_v = np.random.randn(4, 12)",
            "bx_v = np.zeros((1, 12))",
            "bh_v = np.zeros((1, 12))",
            "",
            "h_n = gru_cell_forward(xt_v, h_v, Wx_v, Wh_v, bx_v, bh_v)",
            "print('GRU state shape (should be [2, 4]):', list(h_n.shape))",
            "if 'TODO' not in gru_cell_forward.__code__.co_consts:",
            "    assert list(h_n.shape) == [2, 4]",
            "    print('GRU forward pass verified! ✅')"
        ]),
        make_markdown_cell(["## 📦 Section 4: Library Implementation"]),
        make_code_cell([
            "import torch",
            "import torch.nn as nn",
            "",
            "class PyTorchGRU(nn.Module):",
            "    def __init__(self, input_dim, hidden_dim, output_dim):",
            "        super().__init__()",
            "        # TODO: Define GRU and classification layers" if not is_solution else "        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)\n        self.fc = nn.Linear(hidden_dim, output_dim)",
            "        ",
            "    def forward(self, x):",
            "        # TODO: Implement forward pass" if not is_solution else "        out, h_n = self.gru(x)\n        return self.fc(out[:, -1, :])",
            "        "
        ]),
        make_code_cell([
            "model = PyTorchGRU(input_dim=4, hidden_dim=8, output_dim=1)",
            "dummy_batch = torch.randn(5, 10, 4)",
            "out = model(dummy_batch)",
            "print('Output shape (should be [5, 1]):', list(out.shape))",
            "assert list(out.shape) == [5, 1]",
            "print('PyTorch GRU check passed! ✅')"
        ]),
        make_markdown_cell(["## 🧪 Section 5: Experiments"]),
        make_markdown_cell([
            "Compare the parameter efficiency of RNN, LSTM, and GRU."
        ]),
        make_code_cell([
            "input_dim = 10",
            "hidden_dim = 20",
            "output_dim = 2",
            "",
            "rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)",
            "lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)",
            "gru = nn.GRU(input_dim, hidden_dim, batch_first=True)",
            "",
            "def count_params(model):",
            "    return sum(p.numel() for p in model.parameters())",
            "",
            "print('Parameter Counts:')",
            "print('Vanilla RNN:', count_params(rnn))",
            "print('LSTM:       ', count_params(lstm))",
            "print('GRU:        ', count_params(gru))",
            "",
            "assert count_params(gru) < count_params(lstm)",
            "print('Verification: GRU has fewer parameters than LSTM! ✅')"
        ]),
        make_markdown_cell(["## ❓ Section 6: Interview Questions"]),
        make_markdown_cell([
            "### Q1: How does a GRU differ from an LSTM in terms of gates and states?",
            "**Answer**:",
            "- **States**: LSTM maintains two states: hidden state $h_t$ and cell state $C_t$. GRU maintains only a single hidden state $h_t$.",
            "- **Gates**: LSTM has $3$ gates (forget, input, output). GRU has only $2$ gates (reset, update). This reduction makes GRU computationally lighter.",
            "",
            "### Q2: Why might you choose a GRU over an LSTM?",
            "**Answer**:",
            "GRU models have fewer weights, making them faster to compute per iteration and less likely to overfit on smaller sequential datasets. If training latency or storage size is a constraint, GRUs often perform comparably to LSTMs with reduced overhead.",
            "",
            "### Q3: Explain the function of the Reset Gate in GRU.",
            "**Answer**:",
            "The Reset Gate $r_t$ determines how much of the past hidden state $h_{t-1}$ to write into the candidate hidden state calculation $\\tilde{h}_t$. If $r_t \\approx 0.0$, the candidate drops the historical hidden state entirely and acts as if processing a new sequence, which is useful for segments that change context suddenly."
        ]),
        make_markdown_cell(["## 🏆 Section 7: Challenge — Bidirectional GRU"]),
        make_markdown_cell([
            "**Challenge**: Implement a Bidirectional GRU layer wrapper using PyTorch's native `nn.GRU` layer."
        ]),
        make_code_cell([
            "class BiGRU(nn.Module):",
            "    def __init__(self, input_dim, hidden_dim, output_dim):",
            "        super().__init__()",
            "        # TODO: Define a bidirectional GRU layer (set bidirectional=True)" if not is_solution else "        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)\n        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 because output is concat of forward & backward",
            "        ",
            "    def forward(self, x):",
            "        # TODO: Forward pass and classification step" if not is_solution else "        out, h_n = self.gru(x)\n        # Take the output of the last sequence step (forward + backward concats)\n        last_step = out[:, -1, :]\n        return self.fc(last_step)",
            "",
            "model_bi = BiGRU(5, 8, 2)",
            "dummy_batch = torch.randn(4, 12, 5)",
            "out_bi = model_bi(dummy_batch)",
            "print('BiGRU output shape (should be [4, 2]):', list(out_bi.shape))",
            "if 'TODO' not in BiGRU.__init__.__code__.co_consts:",
            "    assert list(out_bi.shape) == [4, 2]",
            "    print('BiGRU wrapper verified! ✅')"
        ])
    ]
    return cells

# ----------------- Autoencoder Cells -----------------
def get_autoencoder_cells(is_solution):
    title = "# 🔍 Autoencoder (AE) — " + ("Solutions" if is_solution else "Practice") + " Notebook"
    desc = "**This notebook contains " + ("complete, verified solutions.**" if is_solution else "guided exercises — implement the # TODO blocks.**")
    difficulty = "**Difficulty**: ⭐⭐ Intermediate  \n**Time**: ~60 minutes"
    
    cells = [
        make_markdown_cell([title, "", desc, "", difficulty, "", "---"]),
        make_markdown_cell([
            "## 🎯 Section 1: Overview",
            "",
            "An **Autoencoder** is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal noise."
        ]),
        make_markdown_cell([
            "## 📐 Section 2: Math & Intuition",
            "",
            "### Subspace Projection",
            "An Autoencoder consists of two functions:",
            "- Encoder: $z = f(x) = \\sigma(W_e x + b_e)$",
            "- Decoder: $\\hat{x} = g(z) = \\sigma(W_d z + b_d)$",
            "",
            "The network is optimized using Mean Squared Error (MSE) reconstruction loss:",
            "$$L = \\frac{1}{m} \\sum_{i=1}^m \\|x^{(i)} - \\hat{x}^{(i)}\\|_2^2$$",
            "",
            "### Relationship to PCA",
            "If the encoder and decoder activations are linear, the bottleneck space spans the same subspace as **Principal Component Analysis (PCA)**."
        ]),
        make_markdown_cell(["## 🔧 Section 3: Implementation from Scratch"]),
        make_code_cell([
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import torch",
            "import torch.nn as nn",
            "import torch.optim as optim",
            "",
            "torch.manual_seed(42)",
            "print('Autoencoder Setup complete! ✅')"
        ]),
        make_markdown_cell(["### 3.1 Linear Autoencoder vs PCA"]),
        make_code_cell([
            "class LinearAutoencoder(nn.Module):",
            "    def __init__(self, input_dim, latent_dim):",
            "        super().__init__()",
            "        # TODO: Define linear encoder and linear decoder layers" if not is_solution else "        self.encoder = nn.Linear(input_dim, latent_dim)\n        self.decoder = nn.Linear(latent_dim, input_dim)",
            "        ",
            "    def forward(self, x):",
            "        # TODO: Implement forward pass" if not is_solution else "        z = self.encoder(x)\n        x_hat = self.decoder(z)\n        return x_hat",
            "        "
        ]),
        make_code_cell([
            "# Generate simple correlated 2D data",
            "np.random.seed(42)",
            "x1 = np.random.randn(300)",
            "x2 = x1 * 2.0 + np.random.randn(300) * 0.2",
            "data = np.column_stack((x1, x2))",
            "data_t = torch.FloatTensor(data)",
            "",
            "model = LinearAutoencoder(input_dim=2, latent_dim=1)",
            "criterion = nn.MSELoss()",
            "optimizer = optim.Adam(model.parameters(), lr=0.05)",
            "",
            "if 'TODO' not in LinearAutoencoder.__init__.__code__.co_consts:",
            "    # Train",
            "    for epoch in range(150):",
            "        optimizer.zero_grad()",
            "        reconstructed = model(data_t)",
            "        loss = criterion(reconstructed, data_t)",
            "        loss.backward()",
            "        optimizer.step()",
            "        ",
            "    # Get reconstructed coordinates",
            "    with torch.no_grad():",
            "        recon = model(data_t).numpy()",
            "        ",
            "    # Plot original vs. reconstructed",
            "    plt.figure(figsize=(8, 5))",
            "    plt.scatter(data[:, 0], data[:, 1], label='Original', alpha=0.5)",
            "    plt.scatter(recon[:, 0], recon[:, 1], label='Reconstructed', color='red', alpha=0.5)",
            "    plt.title('Linear Autoencoder Projection')",
            "    plt.legend()",
            "    plt.show()",
            "    print('Subspace projection completed! ✅')"
        ]),
        make_markdown_cell(["## 📦 Section 4: Library Implementation"]),
        make_markdown_cell([
            "We will build a Denoising Autoencoder using non-linear layers. Denoising autoencoders learn to reconstruct the clean input from a corrupted version."
        ]),
        make_code_cell([
            "class DenoisingAutoencoder(nn.Module):",
            "    def __init__(self, input_dim, hidden_dim, latent_dim):",
            "        super().__init__()",
            "        # TODO: Define encoder (Linear -> ReLU -> Linear -> ReLU) and decoder" if not is_solution else "        self.encoder = nn.Sequential(\n            nn.Linear(input_dim, hidden_dim),\n            nn.ReLU(),\n            nn.Linear(hidden_dim, latent_dim),\n            nn.ReLU()\n        )\n        self.decoder = nn.Sequential(\n            nn.Linear(latent_dim, hidden_dim),\n            nn.ReLU(),\n            nn.Linear(hidden_dim, input_dim)\n        )",
            "        ",
            "    def forward(self, x):",
            "        # TODO: Implement forward pass" if not is_solution else "        z = self.encoder(x)\n        x_hat = self.decoder(z)\n        return x_hat",
            "        "
        ]),
        make_code_cell([
            "dae = DenoisingAutoencoder(input_dim=10, hidden_dim=6, latent_dim=2)",
            "dummy_x = torch.randn(4, 10)",
            "out = dae(dummy_x)",
            "print('DAE output shape (should be [4, 10]):', list(out.shape))",
            "assert list(out.shape) == [4, 10]",
            "print('Denoising Autoencoder structure verified! ✅')"
        ]),
        make_markdown_cell(["## 🧪 Section 5: Experiments"]),
        make_markdown_cell([
            "Let's train our Denoising Autoencoder to clean up synthetic patterns with Gaussian noise."
        ]),
        make_code_cell([
            "# Generate synthetic target pattern",
            "np.random.seed(42)",
            "clean_patterns = np.sin(np.linspace(0, 2*np.pi, 100))[None, :] * np.random.uniform(0.5, 2.0, (200, 1))",
            "clean_t = torch.FloatTensor(clean_patterns)",
            "",
            "# Add noise to inputs",
            "noisy_patterns = clean_patterns + np.random.randn(*clean_patterns.shape) * 0.3",
            "noisy_t = torch.FloatTensor(noisy_patterns)",
            "",
            "dae = DenoisingAutoencoder(input_dim=100, hidden_dim=32, latent_dim=8)",
            "opt = optim.Adam(dae.parameters(), lr=0.01)",
            "crit = nn.MSELoss()",
            "",
            "if 'TODO' not in DenoisingAutoencoder.__init__.__code__.co_consts:",
            "    for epoch in range(250):",
            "        opt.zero_grad()",
            "        recon = dae(noisy_t)",
            "        l = crit(recon, clean_t)  # Learn to output clean patterns",
            "        l.backward()",
            "        opt.step()",
            "        ",
            "    with torch.no_grad():",
            "        cleaned = dae(noisy_t).numpy()",
            "        ",
            "    # Plot one sample comparison",
            "    plt.figure(figsize=(10, 5))",
            "    plt.plot(clean_patterns[0], label='Original Clean', color='black', linewidth=2)",
            "    plt.plot(noisy_patterns[0], label='Noisy Input', alpha=0.5, linestyle='--')",
            "    plt.plot(cleaned[0], label='Denoised Output', color='green', linewidth=2)",
            "    plt.title('Denoising Autoencoder Verification')",
            "    plt.legend()",
            "    plt.show()"
        ]),
        make_markdown_cell(["## ❓ Section 6: Interview Questions"]),
        make_markdown_cell([
            "### Q1: What is the relationship between a linear autoencoder and Principal Component Analysis (PCA)?",
            "**Answer**:",
            "Under linear activations and an MSE loss function, the hidden representation of a bottleneck autoencoder learns to project input vectors into a subspace spanned by the first $K$ principal components. However, unlike standard PCA, the autoencoder weights are not required to be orthogonal, so it learns a scaled and rotated representation of the same subspace.",
            "",
            "### Q2: Why is a non-linear activation necessary in autoencoders?",
            "**Answer**:",
            "If all activation functions are linear, the network can only represent projections into linear subspaces, meaning the autoencoder is functionally identical to PCA. Non-linear activation functions (such as ReLU or Sigmoid) allow the network to learn complex non-linear coordinate projections, capturing highly non-linear manifolds in data.",
            "",
            "### Q3: What is a Denoising Autoencoder (DAE) and how does it prevent the model from learning the identity function?",
            "**Answer**:",
            "A Denoising Autoencoder is trained to reconstruct the original clean input $x$ from a deliberately corrupted input $\\tilde{x} = x + \\epsilon$. Because the input is distorted, the model cannot simply learn the identity function (copying input to output). Instead, it must capture the underlying distribution structure of the clean data in its latent space to clean up the inputs.",
            "",
            "### Q4: Explain the reconstruction threshold trick for anomaly detection.",
            "**Answer**:",
            "To detect anomalies, an Autoencoder is trained strictly on normal data instances. Because it only learns representations for normal instances, it will reconstruct normal validation items with low MSE. When presented with an anomalous instance, the reconstruction error will be significantly higher because it falls outside the learned manifold. We choose an MSE threshold (e.g. 95th percentile of normal training loss) and flag any item exceeding it as an anomaly."
        ]),
        make_markdown_cell(["## 🏆 Section 7: Challenge — Anomaly Detection"]),
        make_markdown_cell([
            "**Challenge**: Implement the anomaly detection decision logic: compute reconstruction MSE per instance and flag inputs as anomalous if they exceed a specific threshold."
        ]),
        make_code_cell([
            "def detect_anomalies(original, reconstructed, threshold):",
            "    \"\"\"",
            "    original: numpy array of shape (N, D)",
            "    reconstructed: numpy array of shape (N, D)",
            "    threshold: scalar representation of maximum allowed MSE",
            "    Return boolean array of shape (N,) where True = Anomaly",
            "    \"\"\"",
            "    # TODO: Implement anomaly classification logic" if not is_solution else "    mse = np.mean((original - reconstructed) ** 2, axis=1)\n    return mse > threshold",
            "    ",
            "# Test",
            "orig = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 10.0]])  # Last one is anomalous",
            "recon = np.array([[1.0, 1.1], [1.0, 0.9], [1.0, 1.0]])",
            "anom = detect_anomalies(orig, recon, 0.5)",
            "print('Anomaly Flags:', anom)",
            "if 'TODO' not in detect_anomalies.__code__.co_consts:",
            "    assert list(anom) == [False, False, True]",
            "    print('Anomaly Detection logic works! ✅')"
        ])
    ]
    return cells

# ----------------- Write Loop -----------------
base_dir = "/Users/alifouladgar/ml-models-interview-prep/04-deep-learning/foundations"

generators = {
    "01-mlp/mlp": get_mlp_cells,
    "02-cnn/cnn": get_cnn_cells,
    "03-rnn/rnn": get_rnn_cells,
    "04-lstm/lstm": get_lstm_cells,
    "05-gru/gru": get_gru_cells,
    "06-autoencoder/autoencoder": get_autoencoder_cells
}

for rel_path, gen in generators.items():
    # Practice Notebook
    practice_path = os.path.join(base_dir, f"{rel_path}_practice.ipynb")
    practice_cells = gen(is_solution=False)
    save_notebook(practice_path, practice_cells)
    
    # Solutions Notebook
    solution_path = os.path.join(base_dir, f"{rel_path}_solutions.ipynb")
    solution_cells = gen(is_solution=True)
    save_notebook(solution_path, solution_cells)

print("Notebook generation complete!")
