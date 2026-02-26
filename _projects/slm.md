<!-- ---
layout: page
title: "Project: Building a Synthetic-Data SLM"
description: "Training a 1B parameter model using high-quality synthetic data."
---

## 🎯 The Objective
To replicate the 'Textbooks Are All You Need' methodology to see if a small model can achieve high reasoning capabilities with < 5B tokens of data.

## 🛠️ The Roadmap
1. **Data Generation:** Using GPT-4o to generate "Gold" synthetic textbooks.
2. **Filtering:** Implementing the 'Self-Instruct' heuristics to prune low-quality data.
3. **Training:** Fine-tuning a base TinyLlama or Phi model.

## 📓 Research Deep Dives
*Related Journal Entries:*
* [The Power of Textbook Data (Phi Series)]({% post_url 2026-02-24-textbooks-need %})
* [Data Mixtures & Scaling (TinyLlama)]({% post_url 2026-02-25-tinyllama-mixture %}) -->