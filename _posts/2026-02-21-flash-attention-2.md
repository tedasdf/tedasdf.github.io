---
layout: post
title: "Paper Review: FlashAttention-2"
date: 2026-02-21
categories: research
---

# FlashAttention-2: Faster Attention with Better Parallelism

## 1. Algorithmic Refinement (The "Matmul-First" Strategy)
The primary bottleneck in FlashAttention-1 wasn't the total number of operations, but the type. GPUs are "Matmul-Monsters" ($312\text{ TFLOPs/s}$ on A100) but are relatively slow at non-matmul operations like exp or softmax ($19.5\text{ TFLOPs/s}$).

* **The Change:** Instead of rescaling the output $O$ at every iteration, FlashAttention-2 delays the scaling.
* **The Math:** It stores the LogSumExp ($L$) and only performs the final normalization $\text{diag}(L)^{-1}$ at the end.
* **Result:** This shifts the workload toward Matmuls, increasing GPU throughput.

## 2. Parallelism (Sequence Length Scaling)
FlashAttention-1 mostly parallelized across Batch Size and Number of Heads. 



* **Forward Pass:** Parallelizes across the Sequence Length ($N$). Each worker handles specific Rows of the $Q$ matrix.
* **Backward Pass:** Parallelizes across the Sequence Length of $K$ and $V$ (Columns). This avoids "Atomic Adds" for $dK$ and $dV$.

## 3. Warp-Level Work Partitioning
Within a single Thread Block, work is split among Warps.

* **The Solution:** FlashAttention-2 splits the $Q$ matrix across warps while keeping $K$ and $V$ accessible to all.
* **Why it's better:** This reduces "bank conflicts." Warps work in parallel on assigned rows of $Q$ and pull $K/V$ from shared memory as a fast read-only pattern.