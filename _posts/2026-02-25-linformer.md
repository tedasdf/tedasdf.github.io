---
title: "Why Linformer Works: Low-Rank Self-Attention"
date: 2026-02-25
categories: [Transformers, Efficient Attention]
---

# Why Self-Attention Can Be Linear

The standard Transformer introduced in *Attention Is All You Need* (Vaswani et al., 2017) computes self-attention as:

\[
P = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right), \quad Y = PV
\]

where:

- \( Q, K, V \in \mathbb{R}^{n \times d} \)
- \( n \) = sequence length
- \( d \) = hidden dimension

This requires:

- **Time:** \( O(n^2 d) \)
- **Memory:** \( O(n^2) \)

The quadratic dependence on sequence length makes long-context modeling expensive.

---

# Theorem 1: Self-Attention is Low Rank

For any \( Q, K, V \in \mathbb{R}^{n \times d} \), there exists a low-rank matrix  
\( \tilde{P} \in \mathbb{R}^{n \times n} \) such that:

\[
\| \tilde{P}w - Pw \| \le \epsilon \| Pw \|
\]

with high probability, and:

\[
\text{rank}(\tilde{P}) = \Theta(\log n)
\]

## Interpretation

Although the attention matrix \( P \) is \( n \times n \), it is **approximately low-rank**.

This means:

\[
P \approx AB^T
\]

where:

- \( A \in \mathbb{R}^{n \times k} \)
- \( B \in \mathbb{R}^{n \times k} \)
- \( k \ll n \), typically \( k = O(\log n) \)

So the quadratic matrix can be approximated using only \( O(nk) \) parameters.

---

# Theorem 2: Linear Self-Attention

If

\[
k = \min\left( \Theta\left(\frac{d \log d}{\epsilon^2}\right), 
\Theta\left(\frac{\log n}{\epsilon^2}\right) \right)
\]

then there exist matrices:

\[
E, F \in \mathbb{R}^{n \times k}
\]

such that:

\[
\text{softmax}(QK^T)V 
\approx 
\text{softmax}(Q E^T) F V
\]

with bounded approximation error.

---

# Why Linformer Adds Two Linear Projections

The key idea is to project **along the sequence dimension**.

Instead of computing:

\[
(QK^T)V
\]

Linformer computes:

\[
Q(K'^T)V'
\]

where:

\[
K' = E^T K \in \mathbb{R}^{k \times d}
\]

\[
V' = F^T V \in \mathbb{R}^{k \times d}
\]

Now:

- \( QK'^T \) is \( n \times k \)
- \( V' \) is \( k \times d \)

The complexity becomes:

\[
O(nkd)
\]

If \( k = O(\log n) \), we obtain **linear complexity in sequence length**.

---

# Intuition

Self-attention does not use all pairwise interactions independently.

The attention matrix exhibits:

- Redundant structure  
- Strong row correlations  
- Low effective dimensionality  

Therefore, projecting keys and values into a lower-dimensional subspace preserves most of the contextual information.

---

# Final Takeaway

The reason Linformer works is:

> The attention matrix is approximately low-rank.

Thus, we can avoid constructing the full \( n \times n \) matrix and instead operate in a compressed space of dimension \( k \ll n \), achieving linear time and memory complexity while maintaining approximation guarantees.