// gpu_calculator.js
// 🟡 EXACT ARCHITECTURE-AWARE VRAM CALCULATOR ENGINE
// Implements conventional MHA/GQA formula (Pope et al. 2023) AND DeepSeek-V2 MLA formula.
// All binary calculations are performed in exact bytes and binary GiB (1 GiB = 2^30 bytes).

export const MODEL_PRESETS = {
  'llama-3-8b': {
    name: 'Llama-3-8B',
    architectureType: 'conventional',
    archName: 'GQA (8 KV heads, 4:1 ratio)',
    layers: 32,
    qHeads: 32,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 4096,
    weightsGiB: 14.96, // 8.03B params * 2 bytes / 2^30
    citation: 'Meta AI (2024), Llama 3 Model Card'
  },
  'llama-3-70b': {
    name: 'Llama-3-70B',
    architectureType: 'conventional',
    archName: 'GQA (8 KV heads, 8:1 ratio)',
    layers: 80,
    qHeads: 64,
    kvHeads: 8,
    headDim: 128,
    hiddenDim: 8192,
    weightsGiB: 131.5, // 70.6B params * 2 bytes / 2^30
    citation: 'Meta AI (2024), Llama 3 Model Card'
  },
  'gpt3-175b': {
    name: 'GPT-3 (175B Scale)',
    architectureType: 'conventional',
    archName: 'MHA (96 KV heads, 1:1 ratio)',
    layers: 96,
    qHeads: 96,
    kvHeads: 96,
    headDim: 128,
    hiddenDim: 12288,
    weightsGiB: 325.96, // 175B params * 2 bytes / 2^30
    citation: 'Brown et al. (2020), Language Models are Few-Shot Learners'
  },
  'deepseek-v2': {
    name: 'DeepSeek-V2',
    architectureType: 'mla',
    archName: 'MLA: 512D KV latent + 64D decoupled RoPE key, 60 layers',
    layers: 60,
    qHeads: 128,
    latentDim: 512,  // d_c: compressed latent KV dimension
    ropeKeyDim: 64,  // d_r: decoupled RoPE key dimension
    hiddenDim: 5120,
    weightsGiB: 44.0, // 21B active params * 2 bytes / 2^30 (approx active footprint)
    citation: 'DeepSeek-AI (2024), DeepSeek-V2 Technical Report'
  }
};

export const GPU_SPECS = {
  'rtx-4090': {
    name: 'NVIDIA RTX 4090 (24 GB)',
    vramGiB: 24,
    isAggregate: false,
    note: 'Single-GPU consumer flagship'
  },
  'a100-80gb': {
    name: 'NVIDIA A100 (80 GB)',
    vramGiB: 80,
    isAggregate: false,
    note: 'Single-GPU datacenter accelerator'
  },
  '8x-h100': {
    name: '8× H100 — 640 GB aggregate HBM',
    vramGiB: 640,
    isAggregate: true,
    note: 'Aggregate memory across eight GPUs, not a single contiguous 640 GB memory pool. Actual usable capacity depends on the parallelism/sharding strategy.'
  }
};

export const PRECISIONS = {
  'fp16': { name: 'FP16 / BF16 (16-bit)', bytes: 2.0 },
  'fp8': { name: 'FP8 (8-bit)', bytes: 1.0 },
  'int4': { name: 'INT4 (4-bit)', bytes: 0.5 }
};

/**
 * Architecture-aware calculation of KV Cache memory in Bytes and binary GiB.
 *
 * Conventional MHA/GQA: M_KV = 2 * L * n_KV * d_head * T * B * s (Pope et al. 2023)
 * Multi-Head Latent Attention (MLA): M_MLA = L * (d_c + d_r) * T * B * s (DeepSeek-AI 2024)
 *
 * @param {Object} params
 * @param {string} params.presetKey - 'llama-3-8b' | 'llama-3-70b' | 'gpt3-175b' | 'deepseek-v2'
 * @param {number} params.seqLen - Context sequence length T
 * @param {number} params.batchSize - Batch size B
 * @param {string} params.precisionKey - 'fp16' | 'fp8' | 'int4'
 * @param {string} params.gpuKey - 'rtx-4090' | 'a100-80gb' | '8x-h100'
 */
export function calculateKVCacheMemory({
  presetKey = 'llama-3-8b',
  seqLen = 4096,
  batchSize = 1,
  precisionKey = 'fp16',
  gpuKey = 'a100-80gb'
}) {
  const model = MODEL_PRESETS[presetKey] || MODEL_PRESETS['llama-3-8b'];
  const prec = PRECISIONS[precisionKey] || PRECISIONS['fp16'];
  const gpu = GPU_SPECS[gpuKey] || GPU_SPECS['a100-80gb'];

  const T = Number(seqLen);
  const B = Number(batchSize);
  const s = prec.bytes;

  let bytesPerTokenPerLayer = 0;
  let totalKVBytes = 0;
  let formulaUsed = '';
  let formulaMath = '';

  if (model.architectureType === 'conventional') {
    // Factor of 2 accounts for separate Key and Value tensors (Pope et al. 2023)
    // bytesPerTokenPerLayer = 2 * n_KV * d_head * s
    const L = model.layers;
    const nKV = model.kvHeads;
    const dHead = model.headDim;

    bytesPerTokenPerLayer = 2 * nKV * dHead * s;
    totalKVBytes = 2 * L * nKV * dHead * T * B * s;
    formulaUsed = 'Conventional MHA/GQA (Pope et al. 2023): M_KV = 2 · L · n_KV · d_head · T · B · s';
    formulaMath = `2 × ${L} × ${nKV} × ${dHead} × ${T} × ${B} × ${s} bytes`;
  } else if (model.architectureType === 'mla') {
    // Multi-head Latent Attention compresses KV into latent vector c_t + decoupled RoPE key k_t^R
    // Stored representation = (d_c + d_r) elements per token per layer. NO factor of 2!
    const L = model.layers;
    const dc = model.latentDim;   // 512
    const dr = model.ropeKeyDim;  // 64
    const totalElements = dc + dr; // 576

    bytesPerTokenPerLayer = totalElements * s;
    totalKVBytes = L * totalElements * T * B * s;
    formulaUsed = 'DeepSeek-V2 MLA (DeepSeek-AI 2024): M_MLA = L · (d_c + d_r) · T · B · s';
    formulaMath = `${L} × (${dc} + ${dr}) × ${T} × ${B} × ${s} bytes`;
  }

  // Convert to binary GiB (1 GiB = 2^30 bytes)
  const totalKVGiB = totalKVBytes / 1073741824;

  // Scale weights by precision relative to FP16
  const weightsGiB = model.weightsGiB * (s / 2.0);
  const totalRequiredGiB = weightsGiB + totalKVGiB;

  // Simplified capacity threshold test
  const isExceeded = totalRequiredGiB > gpu.vramGiB;
  const kvPercentOfGpu = (totalKVGiB / gpu.vramGiB) * 100;
  const weightsPercentOfGpu = (weightsGiB / gpu.vramGiB) * 100;
  const totalPercentOfGpu = (totalRequiredGiB / gpu.vramGiB) * 100;

  return {
    model,
    precision: prec,
    gpu,
    seqLen: T,
    batchSize: B,
    architectureType: model.architectureType,
    formulaUsed,
    formulaMath,
    bytesPerTokenPerLayer,
    totalKVBytes,
    totalKVGiB,
    weightsGiB,
    totalRequiredGiB,
    gpuCeilingGiB: gpu.vramGiB,
    isExceeded,
    kvPercentOfGpu,
    weightsPercentOfGpu,
    totalPercentOfGpu,
    isAggregate: gpu.isAggregate,
    gpuNote: gpu.note
  };
}
