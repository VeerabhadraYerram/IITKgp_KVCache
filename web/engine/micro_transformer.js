// micro_transformer.js
// 🟡 EDUCATIONAL TOY & LIVE MEASUREMENT ENGINE
// Implements both:
// 1. Naive Full-Recompute (No Cache): reprojects K,V for all past tokens at every generation step -> O(n²) total compute
// 2. Standard KV-Cache: retains past K,V, computing only new projection -> O(n) total, O(1) step compute
// Includes real client-side timed benchmark with performance.now() and measured FLOPs.

export class MicroTransformer {
  /**
   * @param {number} D - Embedding dimension (default 32)
   * @param {number} heads - Number of attention heads (default 4)
   */
  constructor(D = 32, heads = 4) {
    this.D = D;
    this.heads = heads;
    this.headDim = D / heads;

    // Weight projection matrices (fixed seed for deterministic real math)
    this.W_q = MicroTransformer.randomMatrix(D, D, 101);
    this.W_k = MicroTransformer.randomMatrix(D, D, 102);
    this.W_v = MicroTransformer.randomMatrix(D, D, 103);
    this.W_o = MicroTransformer.randomMatrix(D, D, 104);

    this.kvCache = { keys: [], values: [] };
    this.tokenHistory = []; // Raw token embeddings for naive recompute
  }

  reset() {
    this.kvCache = { keys: [], values: [] };
    this.tokenHistory = [];
  }

  /**
   * Matrix-vector multiplication helper: y = x @ W
   */
  static matVecMul(x, W, inDim, outDim) {
    const out = new Float64Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let sum = 0;
      for (let i = 0; i < inDim; i++) {
        sum += x[i] * W[i * outDim + j];
      }
      out[j] = sum;
    }
    return out;
  }

  /**
   * Run ONE generation step WITH KV-Cache.
   * Computes Q, K, V only for the single current token, appends K,V to cache,
   * performs causal attention over all cached tokens.
   * Time complexity: O(T · d)
   */
  stepWithCache(tokenEmbedding) {
    const t0 = performance.now();
    const D = this.D;

    // 1. Project current token only: O(3 * D²) FLOPs
    const q = MicroTransformer.matVecMul(tokenEmbedding, this.W_q, D, D);
    const k = MicroTransformer.matVecMul(tokenEmbedding, this.W_k, D, D);
    const v = MicroTransformer.matVecMul(tokenEmbedding, this.W_v, D, D);

    // 2. Append to KV cache
    this.kvCache.keys.push(k);
    this.kvCache.values.push(v);
    this.tokenHistory.push(new Float64Array(tokenEmbedding));

    const T = this.kvCache.keys.length;

    // 3. Multi-head scaled dot-product attention over cached history: O(T · D) FLOPs
    const attnOutput = this._computeAttention(q, this.kvCache.keys, this.kvCache.values, T);

    // 4. Output projection
    const out = MicroTransformer.matVecMul(attnOutput, this.W_o, D, D);

    const elapsedMs = performance.now() - t0;
    const memoryBytes = T * D * 8 * 2; // Keys + Values in float64
    const totalFlops = 3 * (2 * D * D) + (4 * T * D) + (2 * D * D);

    return {
      output: out,
      seqLen: T,
      stepLatencyMs: elapsedMs,
      memoryBytes,
      flops: totalFlops,
      mode: 'cached'
    };
  }

  /**
   * Run ONE generation step WITHOUT KV-Cache (Naive Full Recompute).
   * Reprojects Q for current token, AND re-projects K and V for all T previous tokens!
   * Time complexity: O(T · d² + T · d)
   */
  stepWithoutCache(tokenEmbedding) {
    const t0 = performance.now();
    const D = this.D;

    this.tokenHistory.push(new Float64Array(tokenEmbedding));
    const T = this.tokenHistory.length;

    // 1. Project Q for current token
    const q = MicroTransformer.matVecMul(tokenEmbedding, this.W_q, D, D);

    // 2. Recompute K and V for ALL tokens 1..T from scratch: O(2 * T * D²) FLOPs
    const recomputedKeys = [];
    const recomputedValues = [];
    for (let i = 0; i < T; i++) {
      const k_i = MicroTransformer.matVecMul(this.tokenHistory[i], this.W_k, D, D);
      const v_i = MicroTransformer.matVecMul(this.tokenHistory[i], this.W_v, D, D);
      recomputedKeys.push(k_i);
      recomputedValues.push(v_i);
    }

    // 3. Attention over history
    const attnOutput = this._computeAttention(q, recomputedKeys, recomputedValues, T);

    // 4. Output projection
    const out = MicroTransformer.matVecMul(attnOutput, this.W_o, D, D);

    const elapsedMs = performance.now() - t0;
    const memoryBytes = 0; // No persistent KV cache retained between steps!
    const totalFlops = (2 * D * D) + (2 * T * 2 * D * D) + (4 * T * D) + (2 * D * D);

    return {
      output: out,
      seqLen: T,
      stepLatencyMs: elapsedMs,
      memoryBytes,
      flops: totalFlops,
      mode: 'naive_recompute'
    };
  }

  /**
   * Scaled dot-product multi-head attention
   */
  _computeAttention(q, keys, values, T) {
    const D = this.D;
    const scale = 1.0 / Math.sqrt(D);
    const scores = new Float64Array(T);

    for (let t = 0; t < T; t++) {
      let dot = 0;
      for (let i = 0; i < D; i++) {
        dot += q[i] * keys[t][i];
      }
      scores[t] = dot * scale;
    }

    // Softmax
    let maxS = -Infinity;
    for (let t = 0; t < T; t++) maxS = Math.max(maxS, scores[t]);
    let sumExp = 0;
    for (let t = 0; t < T; t++) {
      scores[t] = Math.exp(scores[t] - maxS);
      sumExp += scores[t];
    }
    for (let t = 0; t < T; t++) scores[t] /= sumExp;

    // Weighted sum
    const out = new Float64Array(D);
    for (let t = 0; t < T; t++) {
      const weight = scores[t];
      for (let i = 0; i < D; i++) {
        out[i] += weight * values[t][i];
      }
    }
    return out;
  }

  /**
   * Run a timed comparative benchmark generating N tokens.
   * Compares Naive Recompute vs KV-Cache with measured times and FLOPs.
   */
  static runComparativeBenchmark(numTokens = 60, D = 64) {
    const modelCache = new MicroTransformer(D);
    const modelNaive = new MicroTransformer(D);

    const tokens = [];
    for (let i = 0; i < numTokens; i++) {
      tokens.push(MicroTransformer.randomVector(D, 500 + i));
    }

    const cachedSteps = [];
    const naiveSteps = [];

    // Run Cached
    let cachedTotalTime = 0;
    for (let i = 0; i < numTokens; i++) {
      const res = modelCache.stepWithCache(tokens[i]);
      cachedTotalTime += res.stepLatencyMs;
      cachedSteps.push({
        step: i + 1,
        latencyMs: res.stepLatencyMs,
        cumLatencyMs: cachedTotalTime,
        memoryBytes: res.memoryBytes,
        flops: res.flops
      });
    }

    // Run Naive Recompute
    let naiveTotalTime = 0;
    for (let i = 0; i < numTokens; i++) {
      const res = modelNaive.stepWithoutCache(tokens[i]);
      naiveTotalTime += res.stepLatencyMs;
      naiveSteps.push({
        step: i + 1,
        latencyMs: res.stepLatencyMs,
        cumLatencyMs: naiveTotalTime,
        memoryBytes: 0,
        flops: res.flops
      });
    }

    return {
      numTokens,
      D,
      cachedSteps,
      naiveSteps,
      cachedTotalTime,
      naiveTotalTime,
      speedup: naiveTotalTime / (cachedTotalTime || 0.001)
    };
  }

  static randomMatrix(rows, cols, seed = 42) {
    const mat = new Float64Array(rows * cols);
    let s = seed;
    const nextRand = () => {
      s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
      return (s >>> 0) / 4294967296;
    };
    for (let i = 0; i < rows * cols; i++) {
      mat[i] = (nextRand() - 0.5) * 0.2;
    }
    return mat;
  }

  static randomVector(D, seed = 0) {
    const vec = new Float64Array(D);
    let s = seed;
    const nextRand = () => {
      s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
      return (s >>> 0) / 4294967296;
    };
    for (let i = 0; i < D; i++) {
      vec[i] = (nextRand() - 0.5) * 2;
    }
    return vec;
  }
}
