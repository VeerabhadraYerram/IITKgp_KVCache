# When Attention Becomes Memory: KV-Caches, Fast Weights, and Synapses

## The Claim

Fixed-state associative memory can replace sequence-length-growing token storage, but its finite capacity creates an interference trade-off: sparse, non-negative representations reduce expected overlap between stored associations (E[overlap] ∝ p²), while excessive compression or dense overlap causes retrieval failures.

## The Problem

Standard Transformers retain historical context via a Key-Value (KV) cache. During autoregressive inference, every past token's key and value vectors are stored explicitly, creating memory and bandwidth costs that grow proportionally with sequence length *T*: O(*T* · *d*). At 100K+ tokens, this becomes a dominant infrastructure bottleneck (Dao et al., 2022).

## The Mechanism

Matrix multiplication is associative. In unnormalized (linear) attention, the retrieval operation Σ(q · kᵀ) · v can be regrouped as q · Σ(kᵀ · v) = q · S (Katharopoulos et al., 2020). The state matrix S ∈ ℝ^(N×D) accumulates all historical key-value associations into a single structure whose dimensions depend on model width — not sequence length. This eliminates the linear memory dependency on *T*.

The state update S ← S + kᵀv takes the exact mathematical form of a Hebbian outer-product learning rule: ΔW_ij ∝ pre_i × post_j (Hebb, 1949; Schmidhuber, 1992). This formal correspondence connects attention mechanisms to fast-weight memory systems from the neural network literature.

**Critical caveat:** This re-association applies to unnormalized/linear attention. Standard softmax attention couples queries and keys non-linearly, preventing exact regrouping.

## The Trade-Off

| Dimension | Transformer KV-Cache | Fixed-State Associative Memory |
|---|---|---|
| Memory scaling | O(T · d) — grows with sequence length | O(d² or N·d) — constant w.r.t. *T* |
| Recall fidelity | Exact token-level retrieval | Approximate; subject to interference |
| Capacity | Unbounded (limited by hardware) | Bounded by state dimensions |
| Failure mode | Memory/bandwidth exhaustion | Associative crosstalk (interference) |

Dense fixed-state memory suffers catastrophic associative interference: when multiple stored keys share overlapping active dimensions, retrieval produces corrupted blends of stored values. For random non-negative sparse vectors with activation probability *p*, expected pairwise overlap scales as E[⟨k_i, k_j⟩] ∝ p², meaning that reducing density quadratically suppresses interference — but at the cost of reduced representational capacity at extremely low density.

## BDH Connection

Pathway's Dragon Hatchling (BDH) architecture (Kosowski et al., 2025, arXiv:2509.26507) implements these principles at scale. BDH uses a GPU-friendly state-space formulation with sparse non-negative activations (observed at ~5% average activity, varying with predictability) and RoPE-modulated linear attention. The architecture matches GPT-2-scale Transformers on language and translation tasks at 10M–1B parameters while maintaining interpretable, sparse activations.

BDH-CQ (Pathway, 2026, arXiv:2608.09888) extends this recurrent memory to enable latent reasoning without verbal Chain-of-Thought token generation. A 150M-parameter configuration achieves 29.5% pass@2 on ARC-AGI-1 at a developer-computed inference cost of $0.0007/task. This score was independently audited by Bielik/NYU co-authors on the public evaluation set.

## Evidence Classification

| Claim | Classification |
|---|---|
| Associativity of linear attention | 🟢 Mathematically proven (Katharopoulos 2020) |
| Hebbian correspondence of outer-product update | 🟢 Formal equivalence (Hebb 1949, Schmidhuber 1992) |
| BDH ~5% sparsity, Transformer-scale performance | 🟢 Peer-reviewed (Kosowski et al. 2025) |
| BDH-CQ 29.5% ARC-AGI score | 🟣 Developer-reported, independently audited score |
| BDH-CQ $0.0007/task cost | 🟣 Developer-computed estimate ($3/H200-hr) |
| E[overlap] ∝ p² for sparse non-negative codes | 🟡 Derivable analytical property, verified in our toy |
| Latent recurrence counts in BDH-CQ | 🔴 Proprietary; not disclosed |

## Limitations

- Linear attention ≠ softmax attention: the normalization constant is absent, changing retrieval semantics.
- Finite state capacity implies interference under high load (T ≫ N); selective forgetting/decay mechanisms remain an open research question.
- BDH-CQ architectural internals (recurrence depth, update dynamics) are proprietary.
- Real hardware speedups with sparse Hebbian updates may require specialized sparse kernels or neuromorphic substrates beyond standard GPU GEMM operations.

## Primary Sources

[1] Kosowski, A. et al. (2025). *The Dragon Hatchling.* arXiv:2509.26507.
[2] Pathway (2026). *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* arXiv:2608.09888.
[3] Katharopoulos, A. et al. (2020). *Transformers are RNNs.* ICML 2020.
[4] Schlag, I. et al. (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021.
[5] Dao, T. et al. (2022). *FlashAttention.* NeurIPS 2022.
[6] Gu, A. & Dao, T. (2023). *Mamba.* arXiv:2312.00752.
[7] Sun, Y. et al. (2023). *Retentive Network.* arXiv:2307.08621.
[8] Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
