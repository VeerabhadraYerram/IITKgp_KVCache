# When Attention Becomes Memory
**The Hidden Duality Between KV Caches, Fast Weights, and Synapses**

*DataForge 2026 — Pathway Track: Explain the Frontier (IIT Kharagpur)*

---

## Overview

An interactive visual essay exploring how Transformer Key-Value caches, fast-weight associative memory, and biological Hebbian synaptic plasticity are mathematically connected — and what happens when memory fills up.

**Central Claim:** Fixed-state associative memory can replace sequence-length-growing token storage, but its finite capacity creates an interference trade-off: sparse, non-negative representations reduce expected overlap between stored associations (E[overlap] ∝ p²), while excessive compression or dense overlap causes retrieval failures.

## Quick Start

```bash
# Serve the interactive essay locally
cd web/
python3 -m http.server 8765
# Open http://localhost:8765 in your browser
```

No build step, no npm, no dependencies. The entire web application is vanilla HTML, CSS, and JavaScript with ES modules. KaTeX is loaded from CDN for math rendering.

## Project Structure

```
├── web/                             # Interactive Visual Essay
│   ├── index.html                   # 5-act semantic HTML5 structure
│   ├── style.css                    # Publication-grade glassmorphic dark-mode design
│   ├── app.js                       # Main controller
│   ├── engine/
│   │   ├── micro_transformer.js     # Causal softmax attention + KV-cache tracking
│   │   ├── micro_bdh.js             # Educational Micro-BDH (Hebbian update, sparse projection)
│   │   └── memory_experiment.js     # Live associative recall, crime scene, density sweep
│   └── visualizers/
│       ├── memory_wall_canvas.js    # KV buffer growth vs constant state
│       ├── synaptic_matrix_view.js  # Live synaptic grid + Crime Scene inspector
│       └── surgery_hud.js           # 3-knob sandbox + live Recall–Interference curve
├── summary/
│   └── concept_summary.md           # One-page concept summary (500–950 words)
├── notebook/
│   ├── bdh_verification.ipynb       # PyTorch verification of all mathematical claims
│   └── requirements.txt             # Python dependencies
└── README.md                        # This file
```

## The 5-Act Interactive Journey

| Act | Title | Core Interaction |
|-----|-------|-----------------|
| **I** | The Memory Wall & Re-Association | Interactive bracket drag: Σ(qkᵀ)v → q(Σkᵀv) = qS |
| **II** | When Attention Becomes Synapses | Stream tokens; watch Hebbian outer-product writes |
| **III** | BREAK IT! | Store dense associations; inspect the Memory Crime Scene |
| **IV** | Memory Surgery (Crown Jewel) | 3 knobs + live Recall–Interference curve |
| **V** | The Frontier: BDH & BDH-CQ | Real bdh.py vs Micro-BDH + published BDH-CQ effort tiers |

## Honesty Badging System

Every equation, chart, and interactive component displays an explicit provenance badge:

- 🟢 **ESTABLISHED** — Peer-reviewed primary result
- 🟡 **EDUCATIONAL TOY** — Our transparent Micro-BDH abstraction
- 🔵 **LIVE EMPIRICAL** — Dynamically computed from user inputs/seeds
- 🟣 **PRECOMPUTED / AUDITED** — Published benchmark (e.g., BDH-CQ)
- 🔴 **NOT CLAIMED** — Explicit limitations and boundaries

## Primary Sources

1. Kosowski, A. et al. (2025). *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
2. Pathway Research (2026). *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* [arXiv:2608.09888](https://arxiv.org/abs/2608.09888)
3. Katharopoulos, A. et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020.
4. Schlag, I. et al. (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021.
5. Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention.* NeurIPS 2022.
6. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
7. Sun, Y. et al. (2023). *Retentive Network.* arXiv:2307.08621.
8. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.

## Disclosures

- The interactive web demonstration uses a clearly labeled educational **Micro-BDH** engine (D=32, N=128) that isolates the Hebbian outer-product update principle. It is **not** a full reproduction of the BDH architecture (which includes RoPE, graph dynamics, gated sparse MLP, and operates at N=8192).
- BDH-CQ effort tier scores (Low: ~21%, Medium: ~27%, High: ~30% pass@2 on ARC-AGI-1) are published results from arXiv:2608.09888. The inference cost ($0.0007/task) is a developer-computed estimate. Internal recurrence iteration counts are proprietary.
- All live empirical results (Recall–Interference curves, memory comparisons) are computed dynamically in the browser from user-controlled parameters and random seeds.

## License

MIT License. See individual source citations for referenced works.
