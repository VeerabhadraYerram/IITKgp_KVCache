# When Attention Becomes Memory: KV-Caches, Fast Weights, and Synapses

*DataForge 2026 — Pathway Track: Explain the Frontier (IIT Kharagpur)*

---

## 1. The Central Thesis

Transformers "remember" by verbatim caching every historical Key-Value token in VRAM — an exact but memory-bandwidth-choking strategy scaling as \(\mathcal{O}(T \cdot d)\). Replacing this with fixed-state associative memory (\(\mathcal{O}(d^2)\) constant in \(T\)) achieves constant memory and fast recurrence, but introduces a fundamental physical trade-off: **sparse non-negative representations reduce expected crosstalk (\(\mathbb{E}[\text{overlap}] \propto p^2\)), while finite capacity bounds verbatim precision.**

---

## 2. Naive Recompute vs. KV-Caching (\(\mathcal{O}(n^2)\) vs \(\mathcal{O}(n)\))

Autoregressive inference predicts token \(n\) conditioned on tokens \(1 \dots n-1\):
- **Without caching (Naive Full Recompute):** Generating token \(n\) recomputes Key and Value projections for all past tokens from scratch: \(\sum_{t=1}^n \mathcal{O}(t \cdot d^2) = \mathcal{O}(n^2 \cdot d^2)\).
- **With KV-Caching:** Key and Value vectors are computed once per token and stored in VRAM. Marginal step compute drops to \(\mathcal{O}(1 \cdot d^2 + n \cdot d)\), resulting in \(\mathcal{O}(n)\) total projection compute.

---

## 3. The Physical Cost: GPU VRAM Scaling

The exact memory footprint required to hold the KV cache across layers and heads is:
$$\text{VRAM}_{\text{KV}} = 2 \times L \times n_{\text{KV}} \times d_{\text{head}} \times T \times B \times \text{bytes\_per\_element}$$

For an 80-layer model (e.g. Llama-3-70B) running at context length \(T=128\text{k}\) with batch size \(B=16\) in FP16, the KV cache alone demands **~160 GB of VRAM** — exceeding two full NVIDIA A100 (80GB) GPUs purely for caching past tokens.

---

## 4. Production Bottleneck: Prefill vs. Decode

Serving systems (vLLM, DistServe) split execution into two distinct hardware regimes:
1. **Prefill (Compute-Bound):** Prompt tokens are processed in parallel via GEMM with high arithmetic intensity (\(>200\text{ FLOPs/Byte}\)), saturating GPU Tensor Cores.
2. **Decode (Memory-Bandwidth Bound):** Autoregressive token generation executes sequential GEMV with arithmetic intensity collapsed to \(\approx 1\text{–}2\text{ FLOPs/Byte}\). For every single token generated, multi-gigabyte KV caches must be repeatedly transferred across high-bandwidth memory (HBM).

---

## 5. Alternate Approaches, Compared Honestly

| Approach | Mechanism | Memory Scaling | Trade-offs & Limitations |
|---|---|---|---|
| **Standard KV-Cache** | Explicit past token buffer | \(\mathcal{O}(T \cdot d)\) | Unbounded VRAM growth; memory bandwidth decode bottleneck |
| **Eviction / Sinks (StreamingLLM)** | Retain sink tokens + sliding window | \(\mathcal{O}(W)\) constant | **Zero recall for evicted middle context**; cannot perform long-range retrieval |
| **Low-Rank Latents (DeepSeek MLA)** | Compress K,V into latent vector \(c_t\) | \(\mathcal{O}(T \cdot d_c)\) | Reduces memory per token by 4–8×, but still scales linearly with \(T\) |
| **Linear Attention (Katharopoulos 2020)** | Unnormalized associative regrouping \(q(k^\top v) = qS\) | \(\mathcal{O}(d^2)\) constant | Lacks sharp softmax attention focus; high associative interference |
| **State-Space Models (Mamba)** | Continuous selective recurrence \(h_t = Ah_{t-1} + Bx_t\) | \(\mathcal{O}(d \cdot N)\) constant | Fixed state saturates on precise verbatim associative retrieval |
| **Dragon Hatchling (BDH, Kosowski 2025)** | Hebbian edge-reweighting on sparse graph | \(\mathcal{O}(N \cdot D)\) constant | Bounded capacity; mean-field GPU approximation vs discrete theoretical particles |

---

## 6. BDH as an Integrated Redesign (The 5 Interlocking Differences)

Primary author Adrian Kosowski clarified that BDH cannot be reduced to isolated tweaks ("removed softmax, added ReLU"): BDH has **five interlocking differences** that function only as an integrated architecture (arXiv:2509.26507):
1. **Synaptic Plasticity vs Buffer Storage:** Inference working memory is dynamic edge-reweighting (\(\Delta S = k^\top v\)) on a fixed graph topology.
2. **High-Dimensional Sparse Projection:** Non-negative Top-\(k\) / ReLU projection (\(D \rightarrow N\), \(N=8192\)) enforcing ~5% active sparsity, driving \(\mathbb{E}[\text{overlap}] \propto p^2\).
3. **Bilinear Gated Sparse MLP:** Direct elementwise gating (\(xy\_sparse = x\_sparse \odot y\_sparse\)) without dense LayerNorm barriers.
4. **RoPE Phases on Sparse Latents:** Rotary position frequency modulation applied directly to sparse neuron particle activations.
5. **Recurrent In-Place State:** State evolves continuously in \(\mathbf{S}_t\), enabling multi-step latent reasoning without emitting verbal tokens.

### Disclosed Technical Caveats
- **Mean-Field Approximation:** Theoretical BDH defines discrete spiking neuron particles with strict local dynamics; the baseline GPU implementation (`bdh.py` / BDH-GPU) uses a continuous mean-field approximation for Tensor Core efficiency.
- **BDH-CQ Disambiguation & Evaluated Numbers:** BDH-CQ is a separate reasoning model built on BDH for in-context latent reasoning. A 150M-parameter BDH-CQ achieves **29.5% pass@2 on ARC-AGI-1** at **~$0.0007 per task** ($3/H200-hr). Frontier massive dense models with extensive verbal Chain-of-Thought (e.g. Claude 3.5 Sonnet / o1) achieve higher raw accuracy (>40–70%), reflecting a Pareto cost-efficiency trade-off rather than raw ceiling accuracy.

---

## 7. The Epilogue: Closing the Thesis Loop

When an AI stops remembering explicitly:
- **You Gain:** Constant \(\mathcal{O}(1)\) memory footprint w.r.t. sequence length \(T\), elimination of the decoding memory-bandwidth transfer bottleneck, and high concurrent serving batch size.
- **You Surrender:** Literal verbatim replay of historical token strings, trading exact storage for bounded associative capacity subject to \(\mathbb{E}[\text{overlap}] \propto p^2\) interference.
