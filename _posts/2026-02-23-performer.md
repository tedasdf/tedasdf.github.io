---
layout: post
title: "Understanding the Performer: From Softmax to Linear FAVOR+"
date: 2026-02-23
categories: machine-learning
---

## Why the Performer?
The classic Transformer bottleneck is the $O(N^2)$ attention matrix. When the sequence length $N$ grows (e.g., long documents or high-res images), the memory and compute requirements explode. The **Performer** solves this by approximating the attention mechanism to achieve **linear complexity** $O(N \cdot d^2)$.

### 1. The Core Intuition: The Kernel Trick in Reverse
In standard attention:
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

The Performer uses the **FAVOR+** (Fast Attention Via Positive Orthogonal Random Features) framework. It decomposes the kernel function (Softmax) into a product of two smaller feature maps:
$$K(q, k) \approx \phi(q)^T \phi(k)$$

By using the associative property of matrix multiplication, we change the order of operations:
1. **Classic:** $(Q K^T) V \rightarrow [N \times N] \times [N \times d] = O(N^2)$
2. **Performer:** $\phi(Q) (\phi(K)^T V) \rightarrow [N \times M] \times [M \times d] = O(N)$



---

### 2. The Problem with Previous FAVOR (Sine/Cosine)
Earlier methods used **Random Fourier Features** (RFF) with $\sin$ and $\cos$ to approximate kernels. While mathematically sound, they fail for Transformers because:
* **Negativity:** Softmax is always positive. Trig functions can yield negative values.
* **Instability:** Negative values in the attention denominator lead to "anti-attention" or division-by-zero, causing gradients to explode or vanish.

### 3. The FAVOR+ Solution: Positive Random Features
To fix this, the Performer uses a **Positive Random Feature (PRF)** map:
$$\phi(x) = \frac{h(x)}{\sqrt{m}} \exp(W^T x)$$
Where $h(x) = \exp(-\frac{\|x\|^2}{2})$.

When you take the inner product $\phi(q)^T \phi(k)$, you get:
$$\frac{e^{-\|q\|^2/2} e^{-\|k\|^2/2}}{m} \sum_{i=1}^m e^{w_i^T(q+k)}$$

This is an approximation of the **Gaussian Kernel**. Because it uses the exponential function, the output is guaranteed to be positive, ensuring training stability.

---

### 4. Improvements: Orthogonality & SMREG
The Performer doesn't just use standard Gaussian noise for the projection matrix $W$. It uses two key refinements:

1. **Orthogonal Random Features (ORF):** Instead of independent random vectors, $W$ is kept orthogonal (via QR decomposition). This "covers" the space more efficiently, significantly reducing the variance of the approximation.
   
2. **SMREG (Softmax Regularization):**
   To further stabilize training, the authors proposed a regularized version of the kernel to prevent numerical precision issues during 16-bit (FP16) training.

---

### 5. Implementation Summary (PyTorch Logic)

```python
# The "Linear" Causal Trick
# 1. Map Q and K to feature space
q_prime = feature_map(q) # [B, H, T, M]
k_prime = feature_map(k) # [B, H, T, M]

# 2. Compute the "Context" (Accumulated KV pairs)
# Using cumsum instead of a mask for O(N) causality
kv = torch.einsum('bhtm, bhtd -> bhtmd', k_prime, v)
kv_cumsum = torch.cumsum(kv, dim=2)

# 3. Apply Query to the Context
num = torch.einsum('bhtm, bhtmd -> bhtd', q_prime, kv_cumsum)
# (Normalize by the denominator sum...)