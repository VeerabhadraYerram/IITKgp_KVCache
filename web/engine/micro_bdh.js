// micro_bdh.js
// 🟡 EDUCATIONAL TOY — Transparent Micro-BDH engine isolating the Hebbian outer-product
// update and sparse non-negative projection. D=32 by default for real-time browser use.
// This is NOT the full BDH architecture (which includes RoPE, graph dynamics, gated MLP).
// It demonstrates the core mathematical principle: S_{t+1} = S_t + k^T v (Hebbian update).

export class MicroBDH {
  /**
   * @param {number} D     - Embedding dimension (default 32)
   * @param {number} N     - Expanded neuron dimension (default 128)
   * @param {number} density - Fraction of neurons active after ReLU (0.0–1.0, default 0.05)
   */
  constructor(D = 32, N = 128, density = 0.05) {
    this.D = D;
    this.N = N;
    this.density = density;

    // Encoder: projects from D → N (random Gaussian, fixed seed for reproducibility)
    this.encoder = MicroBDH.randomMatrix(D, N, 42);

    // Synaptic state matrix S ∈ R^(N × D) — THE recurrent working memory
    this.S = new Float64Array(N * D); // Initialized to zero

    // Write history for crime-scene analysis
    this.writeHistory = [];
  }

  /** Reset synaptic state and write history */
  reset() {
    this.S.fill(0);
    this.writeHistory = [];
  }

  /**
   * Sparse projection: x → ReLU(x @ E) with density-based thresholding.
   * @param {Float64Array} x - Input vector (length D)
   * @returns {Float64Array} Sparse non-negative activation (length N)
   */
  sparseProject(x) {
    const { D, N, density } = this;
    const latent = new Float64Array(N);

    // Project: latent = x @ encoder
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let i = 0; i < D; i++) {
        sum += x[i] * this.encoder[i * N + j];
      }
      latent[j] = sum;
    }

    // ReLU
    for (let j = 0; j < N; j++) {
      if (latent[j] < 0) latent[j] = 0;
    }

    // Top-k thresholding to enforce target density
    const k = Math.max(1, Math.round(N * density));
    const indices = [];
    for (let j = 0; j < N; j++) {
      if (latent[j] > 0) indices.push({ idx: j, val: latent[j] });
    }
    indices.sort((a, b) => b.val - a.val);

    const sparse = new Float64Array(N);
    const activeIndices = [];
    for (let i = 0; i < Math.min(k, indices.length); i++) {
      sparse[indices[i].idx] = indices[i].val;
      activeIndices.push(indices[i].idx);
    }

    return { sparse, activeIndices };
  }

  /**
   * Hebbian write: S ← S + k^T v (outer product update)
   * @param {Float64Array} key   - Sparse key activation (length N)
   * @param {Float64Array} value - Dense value vector (length D)
   * @param {string} label       - Human-readable label for this association
   * @param {number[]} activeIndices - Indices of active neurons in the key
   */
  hebbianWrite(key, value, label = '', activeIndices = []) {
    const { N, D } = this;

    // ΔS_ij = k_i · v_j  (Hebbian outer product)
    for (let i = 0; i < N; i++) {
      if (key[i] === 0) continue; // Skip zero activations for efficiency
      for (let j = 0; j < D; j++) {
        this.S[i * D + j] += key[i] * value[j];
      }
    }

    // Record write event for crime scene analysis
    this.writeHistory.push({
      label,
      activeIndices: [...activeIndices],
      keyNorm: Math.sqrt(key.reduce((s, v) => s + v * v, 0)),
      timestamp: this.writeHistory.length
    });
  }

  /**
   * Readout: y = q @ S (query the synaptic state)
   * @param {Float64Array} query - Sparse query activation (length N)
   * @returns {Float64Array} Retrieved value estimate (length D)
   */
  readout(query) {
    const { N, D } = this;
    const output = new Float64Array(D);

    for (let i = 0; i < N; i++) {
      if (query[i] === 0) continue;
      for (let j = 0; j < D; j++) {
        output[j] += query[i] * this.S[i * D + j];
      }
    }

    return output;
  }

  /**
   * Full store operation: project key, write Hebbian update.
   * @param {Float64Array} keyInput   - Dense key input (length D)
   * @param {Float64Array} valueInput - Dense value vector (length D)
   * @param {string} label            - Human-readable label
   * @returns {{ sparse: Float64Array, activeIndices: number[] }}
   */
  store(keyInput, valueInput, label = '') {
    const { sparse, activeIndices } = this.sparseProject(keyInput);
    this.hebbianWrite(sparse, valueInput, label, activeIndices);
    return { sparse, activeIndices };
  }

  /**
   * Full retrieve operation: project query, read from synaptic state.
   * @param {Float64Array} queryInput - Dense query input (length D)
   * @returns {{ output: Float64Array, queryActivation: Float64Array, queryActiveIndices: number[] }}
   */
  retrieve(queryInput) {
    const { sparse: queryActivation, activeIndices: queryActiveIndices } = this.sparseProject(queryInput);
    const output = this.readout(queryActivation);
    return { output, queryActivation, queryActiveIndices };
  }

  /**
   * Analyze overlap between a query's active neurons and all stored keys.
   * Used for the Memory Crime Scene inspector.
   * @param {number[]} queryActiveIndices - Active neuron indices of the query
   * @returns {Array<{ label: string, overlapFraction: number, sharedIndices: number[] }>}
   */
  analyzeCrimeScene(queryActiveIndices) {
    const querySet = new Set(queryActiveIndices);
    return this.writeHistory.map(write => {
      const shared = write.activeIndices.filter(idx => querySet.has(idx));
      return {
        label: write.label,
        overlapFraction: write.activeIndices.length > 0
          ? shared.length / Math.max(queryActiveIndices.length, write.activeIndices.length)
          : 0,
        sharedIndices: shared,
        storedActiveCount: write.activeIndices.length,
        timestamp: write.timestamp
      };
    });
  }

  /** Get current memory footprint in bytes (constant w.r.t. sequence length T) */
  getMemoryBytes() {
    return this.N * this.D * 8; // S matrix: N×D × 8 bytes per float64
  }

  /** Get sparsity statistics for the current density setting */
  getSparsityStats() {
    const k = Math.max(1, Math.round(this.N * this.density));
    return {
      totalNeurons: this.N,
      activeNeurons: k,
      activeFraction: k / this.N,
      expectedOverlap: this.density * this.density // E[<k_i, k_j>] ∝ p²
    };
  }

  // ─── Utilities ──────────────────────────────────────────────────

  /** Seeded pseudo-random Gaussian matrix (Box-Muller transform) */
  static randomMatrix(rows, cols, seed = 42) {
    const mat = new Float64Array(rows * cols);
    let s = seed;
    const nextRand = () => {
      s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
      return (s >>> 0) / 4294967296;
    };
    for (let i = 0; i < rows * cols; i += 2) {
      const u1 = nextRand() || 1e-10;
      const u2 = nextRand();
      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 2 * Math.PI * u2;
      mat[i] = r * Math.cos(theta) * 0.1; // std = 0.1
      if (i + 1 < rows * cols) mat[i + 1] = r * Math.sin(theta) * 0.1;
    }
    return mat;
  }

  /** Generate a random dense vector (seeded) */
  static randomVector(D, seed = 0) {
    const vec = new Float64Array(D);
    let s = seed;
    const nextRand = () => {
      s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
      return (s >>> 0) / 4294967296;
    };
    for (let i = 0; i < D; i++) {
      const u1 = nextRand() || 1e-10;
      const u2 = nextRand();
      vec[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return vec;
  }
}
