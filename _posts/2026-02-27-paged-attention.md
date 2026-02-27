---
layout: post
title: "Efficient Memory Management for LLM Serving: PagedAttention & vLLM"
date: 2026-02-27
categories: [AI, Systems, LLM]
link: https://arxiv.org/
---

## Introduction: The KV Cache Bottleneck

Large Language Model (LLM) inference is primarily limited by GPU memory rather than compute.  
The dominant memory consumer is the **Key–Value (KV) cache**, which stores past token representations during autoregressive decoding.

Traditional KV cache allocation relies on contiguous memory reservation for the maximum possible sequence length. Since generation length is unpredictable, this leads to severe memory inefficiency.

In practice, **60–80% of VRAM can be wasted**, limiting batch size and throughput.

PagedAttention addresses this bottleneck by virtualizing the KV cache.

---

### 1. The Memory Waste Problem

During decoding, attention at time step \( t \) requires:

$$
\text{Attention}(Q_t, K_{1:t}, V_{1:t})
$$

To avoid recomputation, all past keys and values are stored.

Traditional systems allocate memory as:

- One contiguous block per sequence  
- Reserved for maximum sequence length  
- Fixed for the duration of the request  

This introduces three types of waste.

#### Internal Fragmentation

Unused reserved memory within a sequence’s allocation.

#### External Fragmentation

Small unusable memory gaps between allocations.

#### Reservation Waste

Memory reserved for future tokens that may never be generated.

The result is poor GPU utilization and limited concurrency.

---

### 2. PagedAttention: Block-Based KV Storage

PagedAttention divides the KV cache into fixed-size blocks (e.g., 16 tokens per block).

Instead of storing tokens contiguously, the cache is treated as a collection of pages.

Logical token layout:

```
Token positions: 0 1 2 3 4 5 6 7 ...
Logical blocks: [B0] [B1] [B2]
```

Physical memory layout:

```
B0 → 0xA92F
B1 → 0xF120
B2 → 0x712C
```


Blocks do not need to be adjacent in memory.

---

#### The Block Table

Each sequence maintains a mapping:

```
Logical Block → Physical Block
```


This **Block Table** decouples:

- Logical token order  
- Physical memory location  

The KV cache becomes virtually contiguous while physically fragmented.

---

### 3. Decoding with PagedAttention

Autoregressive decoding proceeds token by token.

---

#### Step 1: Query Computation

At time step \( t \):

$$
Q_t = \text{Projection}(x_t)
$$

---

#### Step 2: Block-Wise Attention

Instead of loading a contiguous KV tensor, the attention kernel:

1. Iterates through the block table  
2. Fetches each physical block  
3. Computes partial attention  
4. Accumulates results  

Mathematically:

$$
\sum_{\text{blocks}} \text{Softmax}(Q_t K_{\text{block}}^T) V_{\text{block}}
$$

This requires specialized **block-aware CUDA kernels** capable of gathering non-contiguous memory efficiently.

---

#### Step 3: Appending New Tokens

After generating token \( t+1 \):

- Its K and V vectors are written to the last block  
- If the block is full:
  - A new block is allocated  
  - The block table is updated  

Memory grows only when needed.

No worst-case preallocation is required.

---

### 4. Advanced Decoding and Sharing

The flexibility of the block table enables efficient multi-path decoding.

---

#### Shared Prefix Caching

If multiple requests share the same prompt:

```
"Summarize this document..."
```


They reference the same physical prompt blocks.

The KV cache for the prompt is computed once and reused.

---

#### Parallel Sampling

When generating multiple outputs from the same prompt:

- Prompt blocks are shared  
- New blocks are allocated only for divergent tokens  

Memory grows with divergence rather than duplication.

---

#### Beam Search and Copy-on-Write

Beam search repeatedly branches sequences.

Before divergence:

```
Beam A → shared blocks
Beam B → shared blocks
```


When a beam generates a unique token:

1. A new block is allocated  
2. Only necessary data is copied  
3. The block table is updated  

This is equivalent to operating system **Copy-on-Write (CoW)** memory.

Memory complexity becomes:

$$
O(\text{divergent tokens})
$$

instead of

$$
O(\text{sequence length} \times \text{beam width})
$$

---

### 5. Why Waste Drops Below 4%

Because allocation occurs in small fixed blocks:

Worst-case waste per sequence:

$$
< \text{block size}
$$

As sequence length increases:

$$
\text{Waste ratio} \rightarrow 0
$$

Compared to 60–80% waste in traditional systems, this is near-optimal utilization.

---

### 6. The vLLM Execution Engine

PagedAttention is implemented within the **vLLM** serving system.

vLLM provides:

- Centralized scheduling  
- Memory management  
- Preemption policies  
- Custom CUDA kernels  

---

#### Scheduler

vLLM uses a First-Come-First-Served (FCFS) policy.

Sequences belonging to the same request (e.g., beams) are scheduled as a group to avoid deadlocks.

---

#### Preemption

If GPU memory becomes full, vLLM can:

- Swap KV blocks to CPU memory  
- Discard and later recompute them  

This ensures high GPU utilization under heavy load.

---

### Final Thoughts

PagedAttention reframes GPU memory management for LLM serving as a virtual memory problem.

By decoupling logical token order from physical memory placement, it eliminates:

- Internal fragmentation  
- External fragmentation  
- Over-reservation  

Combined with prefix sharing and Copy-on-Write branching, this enables **2×–4× higher throughput** compared to traditional serving systems.

As LLMs continue to scale, memory virtualization techniques like PagedAttention are becoming foundational infrastructure for efficient inference.

---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']]
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>s