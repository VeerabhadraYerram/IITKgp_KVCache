// micro_transformer.js
// 🟡 EDUCATIONAL TOY — Exact causal softmax attention with explicit KV-cache memory tracking.
// Demonstrates O(T·d) memory growth of standard Transformer KV caching.

export class MicroTransformer {
  /**
   * @param {number} D - Embedding dimension (default 32)
   */
  constructor(D = 32) {
    this.D = D;
    this.kvCache = { keys: [], values: [] }; // Explicit KV-cache storage
    this.memoryLog = [];                      // Track memory bytes over time
  }

  /** Reset KV cache and memory log */
  reset() {
    this.kvCache = { keys: [], values: [] };
    this.memoryLog = [];
  }

  /**
   * Append a key-value pair to the KV cache and return attention-weighted retrieval.
   * @param {Float64Array} query  - Query vector (length D)
   * @param {Float64Array} key    - Key vector (length D)
   * @param {Float64Array} value  - Value vector (length D)
   * @returns {{ output: Float64Array, memoryBytes: number, cacheLength: number }}
   */
  step(query, key, value) {
    const D = this.D;
    // Append to KV cache — this is the O(T) growth
    this.kvCache.keys.push(new Float64Array(key));
    this.kvCache.values.push(new Float64Array(value));

    const T = this.kvCache.keys.length;

    // Compute scaled dot-product scores: score_t = q · k_t / sqrt(D)
    const scores = new Float64Array(T);
    const scale = 1.0 / Math.sqrt(D);
    for (let t = 0; t < T; t++) {
      let dot = 0;
      for (let i = 0; i < D; i++) {
        dot += query[i] * this.kvCache.keys[t][i];
      }
      scores[t] = dot * scale;
    }

    // Softmax over scores
    let maxScore = -Infinity;
    for (let t = 0; t < T; t++) maxScore = Math.max(maxScore, scores[t]);
    let sumExp = 0;
    for (let t = 0; t < T; t++) {
      scores[t] = Math.exp(scores[t] - maxScore);
      sumExp += scores[t];
    }
    for (let t = 0; t < T; t++) scores[t] /= sumExp;

    // Weighted sum of values
    const output = new Float64Array(D);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < D; i++) {
        output[i] += scores[t] * this.kvCache.values[t][i];
      }
    }

    // Memory footprint: T keys × D floats × 8 bytes + T values × D floats × 8 bytes
    const memoryBytes = T * D * 8 * 2;
    this.memoryLog.push({ T, memoryBytes });

    return { output, memoryBytes, cacheLength: T };
  }

  /** Run a full sequence of key-value associations and track memory */
  runSequence(keys, values) {
    this.reset();
    const results = [];
    for (let i = 0; i < keys.length; i++) {
      // In standard autoregressive inference, query = current token embedding
      const result = this.step(keys[i], keys[i], values[i]);
      results.push(result);
    }
    return results;
  }

  /**
   * Query the KV cache for retrieval of a specific key.
   * @param {Float64Array} query - The query vector
   * @returns {{ output: Float64Array, attentionWeights: Float64Array, memoryBytes: number }}
   */
  query(query) {
    const D = this.D;
    const T = this.kvCache.keys.length;
    if (T === 0) return { output: new Float64Array(D), attentionWeights: new Float64Array(0), memoryBytes: 0 };

    const scores = new Float64Array(T);
    const scale = 1.0 / Math.sqrt(D);
    for (let t = 0; t < T; t++) {
      let dot = 0;
      for (let i = 0; i < D; i++) {
        dot += query[i] * this.kvCache.keys[t][i];
      }
      scores[t] = dot * scale;
    }

    let maxScore = -Infinity;
    for (let t = 0; t < T; t++) maxScore = Math.max(maxScore, scores[t]);
    let sumExp = 0;
    for (let t = 0; t < T; t++) {
      scores[t] = Math.exp(scores[t] - maxScore);
      sumExp += scores[t];
    }
    for (let t = 0; t < T; t++) scores[t] /= sumExp;

    const output = new Float64Array(D);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < D; i++) {
        output[i] += scores[t] * this.kvCache.values[t][i];
      }
    }

    const memoryBytes = T * D * 8 * 2;
    return { output, attentionWeights: scores, memoryBytes };
  }

  /** Get current memory footprint in bytes */
  getMemoryBytes() {
    return this.kvCache.keys.length * this.D * 8 * 2;
  }
}
