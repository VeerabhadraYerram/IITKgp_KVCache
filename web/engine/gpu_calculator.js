// gpu_calculator.js
// Exact GPU VRAM calculation engine for KV-Cache across architectures and precisions.
// Formula: Memory = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element

export const MODEL_PRESETS = {
  'llama-3-8b': {
    name: 'Llama-3-8B',
    layers: 32,
    qHeads: 32,
    kvHeads: 8,       // GQA (Grouped Query Attention: 4:1 ratio)
    headDim: 128,
    hiddenDim: 4096,
    weightsGB: 16.0,  // FP16 weights
    arch: 'GQA (8 KV heads)'
  },
  'llama-3-70b': {
    name: 'Llama-3-70B',
    layers: 80,
    qHeads: 64,
    kvHeads: 8,       // GQA (8:1 ratio)
    headDim: 128,
    hiddenDim: 8192,
    weightsGB: 140.0, // FP16 weights
    arch: 'GQA (8 KV heads)'
  },
  'gpt4-estimate': {
    name: 'GPT-4 Scale (Dense Est.)',
    layers: 120,
    qHeads: 96,
    kvHeads: 96,      // MHA (Multi-Head Attention: 1:1 ratio)
    headDim: 128,
    hiddenDim: 12288,
    weightsGB: 440.0,
    arch: 'MHA (96 KV heads)'
  },
  'deepseek-v2': {
    name: 'DeepSeek-V2/V3 (MLA)',
    layers: 60,
    qHeads: 128,
    kvHeads: 1,       // MLA compresses KV into single latent c_t of dim 512
    headDim: 512,     // Latent KV dimension
    hiddenDim: 5120,
    weightsGB: 47.0,
    arch: 'MLA (Low-Rank Latent)'
  },
  'mistral-7b': {
    name: 'Mistral-7B',
    layers: 32,
    qHeads: 32,
    kvHeads: 8,       // GQA
    headDim: 128,
    hiddenDim: 4096,
    weightsGB: 14.5,
    arch: 'GQA (8 KV heads)'
  }
};

export const GPU_SPECS = {
  'rtx-4090': { name: 'NVIDIA RTX 4090', vramGB: 24, bandwidthGBs: 1008 },
  'a100-80gb': { name: 'NVIDIA A100 (80GB)', vramGB: 80, bandwidthGBs: 2039 },
  'h100-80gb': { name: 'NVIDIA H100 SXM (80GB)', vramGB: 80, bandwidthGBs: 3350 },
  '8x-h100': { name: '8× NVIDIA H100 Node (640GB)', vramGB: 640, bandwidthGBs: 26800 },
};

export const PRECISIONS = {
  'fp16': { name: 'FP16 / BF16 (16-bit)', bytes: 2 },
  'fp8': { name: 'FP8 (8-bit)', bytes: 1 },
  'int4': { name: 'INT4 (4-bit)', bytes: 0.5 },
};

/**
 * Compute exact KV Cache memory in Bytes and Gigabytes.
 */
export function calculateKVCacheMemory({
  presetKey = 'llama-3-8b',
  seqLen = 4096,
  batchSize = 1,
  precisionKey = 'fp16'
}) {
  const model = MODEL_PRESETS[presetKey] || MODEL_PRESETS['llama-3-8b'];
  const prec = PRECISIONS[precisionKey] || PRECISIONS['fp16'];

  // KV Cache stores both Keys and Values: factor of 2
  // Bytes = 2 × Layers × KV_Heads × Head_Dim × Seq_Len × Batch × Bytes_Per_Element
  const bytesPerToken = 2 * model.layers * model.kvHeads * model.headDim * prec.bytes;
  const totalKVBytes = bytesPerToken * seqLen * batchSize;
  const totalKVGB = totalKVBytes / (1024 * 1024 * 1024);

  // Model weights (assuming precision applies or standard weights)
  const weightsGB = model.weightsGB * (prec.bytes / 2);

  // BDH constant memory comparison (S matrix: N × D, N=8192, D=2048 at scale)
  // For standard scale BDH, S is constant in T:
  const bdhN = 8192;
  const bdhD = model.hiddenDim || 4096;
  const bdhBytesPerLayer = bdhN * bdhD * prec.bytes;
  const bdhTotalBytes = bdhBytesPerLayer * model.layers * batchSize;
  const bdhTotalGB = bdhTotalBytes / (1024 * 1024 * 1024);

  return {
    model,
    precision: prec,
    seqLen,
    batchSize,
    bytesPerToken,
    totalKVBytes,
    totalKVGB,
    weightsGB,
    totalRequiredGB: weightsGB + totalKVGB,
    bdhTotalGB,
  };
}
