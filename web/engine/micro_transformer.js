// micro_transformer.js
// 🟡 REAL CLIENT-SIDE NUMERICAL ENGINE
// Pure JavaScript TypedArray implementation of Transformer K/V Projections & Scaled Dot-Product Attention
// Independent numerical source of truth: Float64Array, D=32, deterministic weights and inputs.

export class MicroTransformerEngine {
  constructor(D = 32) {
    this.D = D;

    // Weight matrices for linear projections: W_Q, W_K, W_V in R^{D x D}
    this.W_q = MicroTransformerEngine.createDeterministicMatrix(D, D, 101);
    this.W_k = MicroTransformerEngine.createDeterministicMatrix(D, D, 102);
    this.W_v = MicroTransformerEngine.createDeterministicMatrix(D, D, 103);

    // Fixed, deterministic projection matrix P in R^{D x 3} for visualizing 32D in 3D
    this.P = MicroTransformerEngine.createProjectionMatrix(D, 3, 2026);
  }

  /**
   * Deterministic pseudo-random matrix initialization (seeded LCG)
   */
  static createDeterministicMatrix(rows, cols, seed = 42) {
    const mat = new Float64Array(rows * cols);
    let s = seed;
    const nextRand = () => {
      s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
      return (s >>> 0) / 4294967296;
    };
    for (let i = 0; i < rows * cols; i++) {
      mat[i] = (nextRand() - 0.5) * (2.0 / Math.sqrt(rows));
    }
    return mat;
  }

  /**
   * Fixed orthonormalized projection matrix P in R^{D x 3}
   */
  static createProjectionMatrix(inDim, outDim = 3, seed = 2026) {
    const P = new Float64Array(inDim * outDim);
    let s = seed;
    const nextRand = () => {
      s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
      return (s >>> 0) / 4294967296;
    };

    // Initialize random Gaussian-like columns
    for (let j = 0; j < outDim; j++) {
      for (let i = 0; i < inDim; i++) {
        P[i * outDim + j] = nextRand() - 0.5;
      }
    }

    // Gram-Schmidt orthogonalization for stable 3D visual projection
    for (let j = 0; j < outDim; j++) {
      for (let prev = 0; prev < j; prev++) {
        let dot = 0;
        for (let i = 0; i < inDim; i++) {
          dot += P[i * outDim + j] * P[i * outDim + prev];
        }
        for (let i = 0; i < inDim; i++) {
          P[i * outDim + j] -= dot * P[i * outDim + prev];
        }
      }
      let norm = 0;
      for (let i = 0; i < inDim; i++) {
        norm += P[i * outDim + j] * P[i * outDim + j];
      }
      norm = Math.sqrt(norm);
      if (norm > 1e-8) {
        for (let i = 0; i < inDim; i++) {
          P[i * outDim + j] /= norm;
        }
      }
    }
    return P;
  }

  /**
   * Matrix-vector product: y = x @ W
   * x: Float64Array(inDim), W: Float64Array(inDim * outDim)
   * Cost: 2 * inDim * outDim FLOPs (inDim * outDim multiplications + inDim * outDim additions)
   */
  static matVecMul(x, W, inDim, outDim) {
    const y = new Float64Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let sum = 0.0;
      for (let i = 0; i < inDim; i++) {
        sum += x[i] * W[i * outDim + j];
      }
      y[j] = sum;
    }
    return y;
  }

  /**
   * Project D-dimensional vector to 3D via deterministic matrix P: y_3d = x @ P
   */
  projectTo3D(x) {
    const D = this.D;
    const P = this.P;
    const out = new Float32Array(3);
    for (let j = 0; j < 3; j++) {
      let sum = 0.0;
      for (let i = 0; i < D; i++) {
        sum += x[i] * P[i * 3 + j];
      }
      // Scale for comfortable 3D coordinate space (~[-15, 15])
      out[j] = sum * 12.0;
    }
    return out;
  }

  /**
   * Run the comparative benchmark and extract numerical state snapshots.
   * Execution order:
   * 1. Run numerical benchmark isolated from rendering.
   * 2. Measure performance.now() elapsed times for No-Cache and With-Cache passes.
   * 3. Compute analytical FLOP accounting for included operations.
   * 4. Generate deterministic state snapshots for the 3D Attention Microscope.
   *
   * @param {number} N - Sequence length (number of generated tokens)
   * @param {number} repeats - Workload multiplier to ensure reliable timing above browser clock jitter
   */
  runBenchmark(N = 80, repeats = 25) {
    const D = this.D;
    const scale = 1.0 / Math.sqrt(D);

    // Generate N deterministic input vectors x_1 ... x_N
    const tokens = [];
    let s = 999;
    const nextRand = () => {
      s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
      return (s >>> 0) / 4294967296;
    };
    for (let t = 0; t < N; t++) {
      const vec = new Float64Array(D);
      for (let i = 0; i < D; i++) {
        vec[i] = (nextRand() - 0.5) * 2.0;
      }
      tokens.push(vec);
    }

    // ─────────────────────────────────────────────────────────────
    // PASS A: NO CACHE (Naive Recompute at every step)
    // ─────────────────────────────────────────────────────────────
    const noCacheStepTimes = new Float64Array(N);
    const noCacheCumTimes = new Float64Array(N);

    // Warm-up JIT
    for (let i = 0; i < 5; i++) {
      MicroTransformerEngine.matVecMul(tokens[0], this.W_k, D, D);
    }

    let naiveTotalElapsed = 0;
    for (let t = 1; t <= N; t++) {
      const t0 = performance.now();

      for (let rep = 0; rep < repeats; rep++) {
        // 1. Timestep-dependent query: q_t = x_t @ W_Q
        const q_t = MicroTransformerEngine.matVecMul(tokens[t - 1], this.W_q, D, D);

        // 2. Recompute all K_1...K_t and V_1...V_t from scratch
        const K_hist = [];
        const V_hist = [];
        for (let i = 0; i < t; i++) {
          K_hist.push(MicroTransformerEngine.matVecMul(tokens[i], this.W_k, D, D));
          V_hist.push(MicroTransformerEngine.matVecMul(tokens[i], this.W_v, D, D));
        }

        // 3. Compute attention output over K_hist and V_hist
        const scores = new Float64Array(t);
        for (let i = 0; i < t; i++) {
          let dot = 0;
          for (let d = 0; d < D; d++) {
            dot += q_t[d] * K_hist[i][d];
          }
          scores[i] = dot * scale;
        }
        // Softmax
        let maxS = -Infinity;
        for (let i = 0; i < t; i++) if (scores[i] > maxS) maxS = scores[i];
        let sumExp = 0;
        for (let i = 0; i < t; i++) {
          scores[i] = Math.exp(scores[i] - maxS);
          sumExp += scores[i];
        }
        for (let i = 0; i < t; i++) scores[i] /= sumExp;

        // Weighted sum
        const out = new Float64Array(D);
        for (let i = 0; i < t; i++) {
          const w = scores[i];
          for (let d = 0; d < D; d++) {
            out[d] += w * V_hist[i][d];
          }
        }
      }

      const stepTime = (performance.now() - t0) / repeats;
      noCacheStepTimes[t - 1] = stepTime;
      naiveTotalElapsed += stepTime;
      noCacheCumTimes[t - 1] = naiveTotalElapsed;
    }

    // ─────────────────────────────────────────────────────────────
    // PASS B: WITH KV CACHE (Store past projections, compute only new)
    // ─────────────────────────────────────────────────────────────
    const cachedStepTimes = new Float64Array(N);
    const cachedCumTimes = new Float64Array(N);

    const cacheK = [];
    const cacheV = [];

    let cachedTotalElapsed = 0;
    for (let t = 1; t <= N; t++) {
      const t0 = performance.now();

      for (let rep = 0; rep < repeats; rep++) {
        // 1. Timestep-dependent query: q_t = x_t @ W_Q
        const q_t = MicroTransformerEngine.matVecMul(tokens[t - 1], this.W_q, D, D);

        // 2. Compute ONLY the new token's K_t and V_t
        const new_k = MicroTransformerEngine.matVecMul(tokens[t - 1], this.W_k, D, D);
        const new_v = MicroTransformerEngine.matVecMul(tokens[t - 1], this.W_v, D, D);

        // Append to cache on first repeat pass
        if (rep === 0) {
          cacheK.push(new_k);
          cacheV.push(new_v);
        }

        // 3. Compute identical attention operation over cached K_{1:t}, V_{1:t}
        const scores = new Float64Array(t);
        for (let i = 0; i < t; i++) {
          let dot = 0;
          for (let d = 0; d < D; d++) {
            dot += q_t[d] * cacheK[i][d];
          }
          scores[i] = dot * scale;
        }
        let maxS = -Infinity;
        for (let i = 0; i < t; i++) if (scores[i] > maxS) maxS = scores[i];
        let sumExp = 0;
        for (let i = 0; i < t; i++) {
          scores[i] = Math.exp(scores[i] - maxS);
          sumExp += scores[i];
        }
        for (let i = 0; i < t; i++) scores[i] /= sumExp;

        const out = new Float64Array(D);
        for (let i = 0; i < t; i++) {
          const w = scores[i];
          for (let d = 0; d < D; d++) {
            out[d] += w * cacheV[i][d];
          }
        }
      }

      const stepTime = (performance.now() - t0) / repeats;
      cachedStepTimes[t - 1] = stepTime;
      cachedTotalElapsed += stepTime;
      cachedCumTimes[t - 1] = cachedTotalElapsed;
    }

    // ─────────────────────────────────────────────────────────────
    // ANALYTICAL FLOP ACCOUNTING FOR INCLUDED OPERATIONS
    // ─────────────────────────────────────────────────────────────
    // Naive K/V Proj: 4 * D^2 * sum(t=1..N, t) = 2 * N * (N + 1) * D^2
    const naiveProjFlops = 2 * N * (N + 1) * D * D;
    // Cached K/V Proj: 4 * N * D^2
    const cachedProjFlops = 4 * N * D * D;
    // Attention Interaction: 4 * D * sum(t=1..N, t) = 2 * N * (N + 1) * D
    const attnFlops = 2 * N * (N + 1) * D;
    // Query Projection: 2 * N * D^2 (computed per step in both passes)
    const queryProjFlops = 2 * N * D * D;

    const naiveTotalFlops = naiveProjFlops + attnFlops;
    const cachedTotalFlops = cachedProjFlops + attnFlops;
    const theoreticalSpeedup = naiveTotalFlops / (cachedTotalFlops || 1);

    // ─────────────────────────────────────────────────────────────
    // GENERATE STATE SNAPSHOTS FOR 3D ATTENTION MICROSCOPE
    // (Generated after benchmark timing completes — zero interference)
    // ─────────────────────────────────────────────────────────────
    const snapshots = [];
    const allKeys = [];
    const allValues = [];
    const allProjectedKeys = [];
    const allProjectedValues = [];

    // Precompute all projections deterministically
    for (let i = 0; i < N; i++) {
      const k = MicroTransformerEngine.matVecMul(tokens[i], this.W_k, D, D);
      const v = MicroTransformerEngine.matVecMul(tokens[i], this.W_v, D, D);
      allKeys.push(k);
      allValues.push(v);
      allProjectedKeys.push(this.projectTo3D(k));
      allProjectedValues.push(this.projectTo3D(v));
    }

    for (let t = 1; t <= N; t++) {
      // Step-specific query: q_t = x_t @ W_Q
      const q_t = MicroTransformerEngine.matVecMul(tokens[t - 1], this.W_q, D, D);
      const proj_q_t = this.projectTo3D(q_t);

      const rawDotProducts = new Float64Array(t);
      const scaledScores = new Float64Array(t);
      const attentionWeights = new Float64Array(t);

      for (let i = 0; i < t; i++) {
        let dot = 0;
        for (let d = 0; d < D; d++) {
          dot += q_t[d] * allKeys[i][d];
        }
        rawDotProducts[i] = dot;
        scaledScores[i] = dot * scale;
      }

      let maxS = -Infinity;
      for (let i = 0; i < t; i++) if (scaledScores[i] > maxS) maxS = scaledScores[i];
      let sumExp = 0;
      for (let i = 0; i < t; i++) {
        attentionWeights[i] = Math.exp(scaledScores[i] - maxS);
        sumExp += attentionWeights[i];
      }
      for (let i = 0; i < t; i++) attentionWeights[i] /= sumExp;

      // True output vector: o_t = sum(alpha_i * v_i) in R^D
      const outputVector = new Float64Array(D);
      for (let i = 0; i < t; i++) {
        const w = attentionWeights[i];
        for (let d = 0; d < D; d++) {
          outputVector[d] += w * allValues[i][d];
        }
      }
      const projectedOutput = this.projectTo3D(outputVector);

      snapshots.push({
        step: t,
        D,
        query: q_t,
        projectedQuery: proj_q_t,
        keys: allKeys.slice(0, t),
        values: allValues.slice(0, t),
        projectedKeys: allProjectedKeys.slice(0, t),
        projectedValues: allProjectedValues.slice(0, t),
        rawDotProducts,
        scaledScores,
        attentionWeights,
        outputVector,
        projectedOutput,
        cacheStats: {
          noCacheRecomputed: t,
          cachedReused: t - 1,
          cachedNew: 1,
          totalStoredVectors: 2 * t
        }
      });
    }

    return {
      measured: {
        numTokens: N,
        D,
        noCacheTotalMs: naiveTotalElapsed,
        cachedTotalMs: cachedTotalElapsed,
        measuredSpeedup: naiveTotalElapsed / (cachedTotalElapsed || 0.001),
        noCacheStepTimes,
        noCacheCumTimes,
        cachedStepTimes,
        cachedCumTimes
      },
      analytical: {
        numTokens: N,
        D,
        naiveProjFlops,
        cachedProjFlops,
        attnFlops,
        queryProjFlops,
        naiveTotalFlops,
        cachedTotalFlops,
        theoreticalSpeedup
      },
      snapshots
    };
  }
}

export { MicroTransformerEngine as MicroTransformer };
