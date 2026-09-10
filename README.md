# When Attention Becomes Memory
**The Hidden Duality Between KV Caches, Fast Weights, and Synapses**

*DataForge 2026 — Pathway Track: Explain the Frontier (IIT Kharagpur)*

---

## Overview

An interactive visual essay and live client-side computational substrate exploring how Transformer Key-Value caches, memory bandwidth bottlenecks, fast-weight associative memory, and biological Hebbian synaptic plasticity are mathematically connected — and what happens when memory fills up.

**Central Thesis:**
Transformers "remember" by explicitly caching every historical Key-Value token in VRAM — an exact but memory-bandwidth-choking strategy scaling as \(\mathcal{O}(T \cdot d)\). Replacing this with a fixed-state associative memory (\(\mathcal{O}(d^2)\) constant in \(T\)) achieves constant memory and fast recurrence, but introduces a fundamental physical trade-off: sparse non-negative representations reduce expected crosstalk (\(\mathbb{E}[\text{overlap}] \propto p^2\)), while finite capacity bounds verbatim precision.

## Quick Start

```bash
# Serve the interactive essay locally
cd web/
python3 -m http.server 8765
# Open http://localhost:8765 in your browser
```

No build step, no npm, no external server dependencies. Pure vanilla HTML5, CSS, and ES Modules with client-side execution and KaTeX math rendering.

## Project Structure

```
├── web/                             # Interactive Visual Essay
│   ├── index.html                   # 6-chapter semantic HTML5 structure + Epilogue
│   ├── style.css                    # Publication-grade monochrome black & white design
│   ├── app.js                       # Main controller wiring benchmarks & interactions
│   ├── engine/
│   │   ├── micro_transformer.js     # Live timed benchmark: naive recompute vs KV-cache
│   │   ├── micro_bdh.js             # Micro-BDH Hebbian associative memory substrate
│   │   ├── gpu_calculator.js        # Exact GPU VRAM allocator equation across architectures
│   │   └── memory_experiment.js     # Associative recall, crime scene, density sweep
│   └── visualizers/
│       ├── memory_wall_canvas.js    # KV buffer growth vs constant synaptic state
│       ├── bandwidth_canvas.js      # GPU Roofline (Prefill vs Decode) & Batch collapse
│       ├── synaptic_matrix_view.js  # Live synaptic grid + Crime Scene inspector
│       └── surgery_hud.js           # 3-knob sandbox + live Recall–Interference curve
├── summary/
│   └── concept_summary.md           # One-page concept summary (500–950 words)
├── notebook/
│   ├── bdh_verification.ipynb       # PyTorch verification of all mathematical claims
│   └── requirements.txt             # Python dependencies
└── README.md                        # This file
```

## The 6-Chapter Interactive Journey + Epilogue

| Chapter | Title | Core Interaction |
|---|---|---|
| **1** | What does an AI mean by "remembering"? | Forced-choice dilemma: Literal Legal Audit vs High-Throughput Agent |
| **2** | Naive Recompute vs. The Cache | Live client-side timed benchmark: \(\mathcal{O}(n^2)\) vs \(\mathcal{O}(n)\) with `performance.now()` |
| **3** | The Cost: What You're Actually Storing | Interactive GPU VRAM equation sandbox (Llama-3, GPT-4, DeepSeek MLA) + BDH Thread |
| **4** | Limitations in Production | GPU Roofline model (Prefill Compute vs Decode Bandwidth) & Concurrency Collapse |
| **5** | Alternate Approaches, Compared Honestly | Taxonomy comparison matrix: Eviction vs Low-Rank vs SSM vs Linear Attention vs BDH |
| **6** | BDH as an Integrated Redesign | Kosowski's 5 interlocking differences table, BDH-CQ ARC-AGI benchmarks + Surgery HUD |
| **Epilogue** | The Sixty-Second Test | Self-assessment synthesis quiz closing the loop on the opening thesis |

## Provenance Badging System

Every equation, chart, and interactive component displays an explicit provenance badge:

- ● **ESTABLISHED** — Peer-reviewed primary literature (Vaswani 2017, Katharopoulos 2020, Dao 2022)
- ⬡ **EDUCATIONAL TOY** — Transparent Micro-BDH abstraction (D=32, N=128)
- ◈ **LIVE EMPIRICAL** — Dynamically computed client-side from user inputs/seeds
- ◼ **PRECOMPUTED / AUDITED** — Published benchmark (BDH-CQ 29.5% pass@2 on ARC-AGI-1)
- ⊘ **NOT CLAIMED / CAVEATS** — Disclosed boundaries (Mean-field approximation, unnormalized attention)

## Primary Sources

1. Kosowski, A. et al. (2025). *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
2. Pathway Research (2026). *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* [arXiv:2608.09888](https://arxiv.org/abs/2608.09888)
3. Katharopoulos, A. et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020.
4. Schlag, I. et al. (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021.
5. Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention.* NeurIPS 2022.
6. Pope, R. et al. (2023). *Efficiently Scaling Transformer Inference on TPU v4.* MLSys 2023.
7. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
8. DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* arXiv:2405.04434.
9. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.

## Disclosures & Honest Technical Boundaries

- **Integrated Architecture:** BDH is presented as an integrated 5-part system (Hebbian synapses, ~5% sparse projection, gated MLP, RoPE phases, recurrent latent state) per author Adrian Kosowski's primary-source clarification.
- **Mean-Field Approximation Disclosed:** The GPU implementation (`bdh.py`) uses a mean-field continuous approximation to discrete spiking neuron-particle dynamics.
- **BDH-CQ Disambiguation:** BDH-CQ is a distinct reasoning system evaluated at 29.5% pass@2 on ARC-AGI-1 at $0.0007/task. Frontier dense models with Chain-of-Thought achieve higher raw accuracy (>40–70%), highlighting a Pareto cost-efficiency trade-off rather than raw ceiling accuracy.

## License

MIT License.
