// memory_experiment.js
// 🔵 LIVE EMPIRICAL — Associative recall benchmark, crime scene analysis,
// and Recall–Interference curve generator with analytical E[overlap] ∝ p² overlay.
// All computations run live in the browser from user-controlled parameters and seeds.

import { MicroBDH } from './micro_bdh.js';
import { MicroTransformerEngine } from './micro_transformer.js';

/**
 * Generate a set of key-value associations for the experiment.
 * Keys are random D-dimensional vectors; values encode a scalar ID.
 */
export function generateAssociations(count, D, seed = 12345, keySimilarity = 0.0) {
  const associations = [];
  const baseVectors = [];

  // Generate base key vectors
  for (let i = 0; i < count; i++) {
    baseVectors.push(MicroBDH.randomVector(D, seed + i * 7 + 3));
  }

  // Optionally blend keys to increase similarity
  for (let i = 0; i < count; i++) {
    const key = new Float64Array(D);
    for (let d = 0; d < D; d++) {
      // Mix own vector with a shared direction to control overlap
      const shared = baseVectors[0][d]; // reference direction
      key[d] = (1 - keySimilarity) * baseVectors[i][d] + keySimilarity * shared;
    }

    // Normalize the key
    let norm = 0;
    for (let d = 0; d < D; d++) norm += key[d] * key[d];
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < D; d++) key[d] /= norm;

    // Value encodes a unique "ground truth" scalar spread across the vector
    const value = new Float64Array(D);
    const groundTruth = (i + 1) * 10; // 10, 20, 30, ...
    for (let d = 0; d < D; d++) {
      value[d] = groundTruth / D; // Uniform encoding
    }

    const labels = [
      'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon',
      'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa',
      'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron',
      'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon'
    ];

    associations.push({
      key,
      value,
      label: labels[i % labels.length],
      groundTruth,
    });
  }

  return associations;
}

/**
 * Run a single associative recall experiment.
 * Stores all associations, then queries each one and measures retrieval accuracy.
 * @returns {{ recalls: Array, overallAccuracy: number, crimeScenes: Array }}
 */
export function runRecallExperiment({ D = 32, N = 128, density = 0.05, count = 8, seed = 12345, keySimilarity = 0.0 }) {
  const bdh = new MicroBDH(D, N, density);
  const associations = generateAssociations(count, D, seed, keySimilarity);

  // Store all associations
  const storeResults = [];
  for (const assoc of associations) {
    const result = bdh.store(assoc.key, assoc.value, assoc.label);
    storeResults.push(result);
  }

  // Query each association and measure retrieval quality
  const recalls = [];
  const crimeScenes = [];

  for (let i = 0; i < associations.length; i++) {
    const assoc = associations[i];
    const { output, queryActivation, queryActiveIndices } = bdh.retrieve(assoc.key);

    // Compute retrieval accuracy via cosine similarity with ground truth value
    const cosSim = cosineSimilarity(output, assoc.value);
    const retrievedScalar = output.reduce((s, v) => s + v, 0); // Sum as proxy for encoded value
    const expectedScalar = assoc.groundTruth;

    // Crime scene: which stored keys overlap with this query?
    const crimeScene = bdh.analyzeCrimeScene(queryActiveIndices);

    recalls.push({
      label: assoc.label,
      groundTruth: expectedScalar,
      retrieved: Math.round(retrievedScalar * 10) / 10,
      cosineSimilarity: Math.round(cosSim * 1000) / 1000,
      match: cosSim > 0.8,
      queryActiveCount: queryActiveIndices.length,
    });

    crimeScenes.push({
      queryLabel: assoc.label,
      overlaps: crimeScene
        .filter(cs => cs.label !== assoc.label)
        .sort((a, b) => b.overlapFraction - a.overlapFraction)
        .slice(0, 3), // Top 3 interfering keys
    });
  }

  const overallAccuracy = recalls.filter(r => r.match).length / recalls.length;

  return {
    recalls,
    overallAccuracy,
    crimeScenes,
    memoryBytes: bdh.getMemoryBytes(),
    sparsityStats: bdh.getSparsityStats(),
  };
}

/**
 * Sweep density from pMin to pMax and measure recall accuracy at each point.
 * This generates the live Recall–Interference curve.
 * @returns {{ densities: number[], accuracies: number[], analyticalOverlaps: number[], interferences: number[] }}
 */
export function sweepDensityCurve({ D = 32, N = 128, count = 8, seed = 12345, keySimilarity = 0.0, steps = 20 }) {
  const densities = [];
  const accuracies = [];
  const analyticalOverlaps = [];
  const interferences = [];

  for (let s = 0; s < steps; s++) {
    // Density from 2% to 100%
    const p = 0.02 + (s / (steps - 1)) * 0.98;
    densities.push(p);

    const result = runRecallExperiment({ D, N, density: p, count, seed, keySimilarity });
    accuracies.push(result.overallAccuracy);

    // Analytical expected overlap: E[<k_i, k_j>] ∝ p² for non-negative sparse vectors
    const analyticalOverlap = p * p;
    analyticalOverlaps.push(analyticalOverlap);

    // Mean interference from crime scenes (average max overlap fraction)
    const meanInterference = result.crimeScenes.reduce((sum, cs) => {
      const maxOverlap = cs.overlaps.length > 0 ? cs.overlaps[0].overlapFraction : 0;
      return sum + maxOverlap;
    }, 0) / result.crimeScenes.length;
    interferences.push(meanInterference);
  }

  return { densities, accuracies, analyticalOverlaps, interferences };
}

/**
 * Compare memory footprint of MicroTransformer vs MicroBDH across sequence lengths.
 * @returns {{ lengths: number[], transformerMemory: number[], bdhMemory: number[] }}
 */
export function compareMemoryScaling({ D = 32, N = 128, maxT = 100, steps = 20 }) {
  const lengths = [];
  const transformerMemory = [];
  const bdhMemory = [];

  const bdh = new MicroBDH(D, N, 0.05);
  const bdhFixed = bdh.getMemoryBytes();

  for (let s = 0; s < steps; s++) {
    const T = Math.max(1, Math.round((s + 1) * maxT / steps));
    lengths.push(T);

    // Transformer: memory = T × D × 8 bytes × 2 (keys + values)
    transformerMemory.push(T * D * 8 * 2);

    // BDH: memory = N × D × 8 bytes (constant)
    bdhMemory.push(bdhFixed);
  }

  return { lengths, transformerMemory, bdhMemory };
}

// ─── Utility ──────────────────────────────────────────────────────

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}
