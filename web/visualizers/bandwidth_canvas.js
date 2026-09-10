// bandwidth_canvas.js
// Visualizer for Chapter 4: Prefill vs Decode & Memory Bandwidth Roofline Bottleneck.
// Renders Arithmetic Intensity (FLOPs/Byte) and Max Serving Concurrency (Batch Size B) vs Sequence Length.

export class BandwidthCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.currentMode = 'roofline'; // 'roofline' or 'concurrency'
    this.seqLen = 4096;
    this.batchSize = 16;
    this.gpuBandwidth = 2039; // A100: 2039 GB/s
    this.gpuComputeTFLOPS = 312; // A100 FP16 Tensor Core: 312 TFLOPs

    this._resizeHandler = () => this._resize();
    window.addEventListener('resize', this._resizeHandler);
    this._resize();
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = (rect.height || 300) * dpr;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = (rect.height || 300) + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = rect.width;
    this.height = rect.height || 300;
    this.draw();
  }

  updateParams(seqLen, batchSize) {
    this.seqLen = seqLen;
    this.batchSize = batchSize;
    this.draw();
  }

  setMode(mode) {
    this.currentMode = mode;
    this.draw();
  }

  draw() {
    if (this.currentMode === 'roofline') {
      this.drawRoofline();
    } else {
      this.drawConcurrency();
    }
  }

  drawRoofline() {
    const { ctx, width, height, seqLen, batchSize } = this;
    ctx.clearRect(0, 0, width, height);

    // Pure black background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    const pad = { top: 40, right: 30, bottom: 50, left: 70 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Theoretical Roofline line:
    // Memory bound slope = GPU Bandwidth (e.g. 2039 GB/s = 2.039 TB/s)
    // Compute peak ceiling = 312 TFLOPs
    // Knee point Arithmetic Intensity = Peak TFLOPs / Bandwidth = ~153 FLOPs/Byte
    const kneeAI = 150; // FLOPs / byte

    const minLogAI = -1; // 0.1 FLOPs/Byte
    const maxLogAI = 3.5; // ~3000 FLOPs/Byte

    const xLogScale = (ai) => {
      const log = Math.log10(Math.max(0.1, ai));
      return pad.left + ((log - minLogAI) / (maxLogAI - minLogAI)) * plotW;
    };

    const yLogScale = (perfTFlops) => {
      const minLogPerf = -1;
      const maxLogPerf = 3;
      const log = Math.log10(Math.max(0.1, perfTFlops));
      return pad.top + plotH - ((log - minLogPerf) / (maxLogPerf - minLogPerf)) * plotH;
    };

    // Draw Roofline Curve
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    // Slope (Memory bound)
    ctx.moveTo(xLogScale(0.1), yLogScale(0.1 * 2.039));
    ctx.lineTo(xLogScale(kneeAI), yLogScale(312));
    // Ceiling (Compute bound)
    ctx.lineTo(xLogScale(3000), yLogScale(312));
    ctx.stroke();

    // Prefill Operating Point:
    // In Prefill, GEMM arithmetic intensity = O(T_prompt) ~ 200 to 800 FLOPs/Byte
    const prefillAI = Math.min(800, 50 + seqLen * 0.1);
    const prefillPerf = Math.min(312, prefillAI * 2.039 * 0.85);

    // Decode Operating Point:
    // In Decode, GEMV arithmetic intensity is strictly 1 to 2 FLOPs/Byte per token!
    const decodeAI = Math.min(4, 1 + batchSize * 0.1);
    const decodePerf = decodeAI * 2.039 * 0.8;

    // Plot Prefill Point (Square marker)
    const px = xLogScale(prefillAI);
    const py = yLogScale(prefillPerf);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(px - 6, py - 6, 12, 12);
    ctx.font = 'bold 11px "Inter", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('⚡ PREFILL (Compute-Bound)', px, py - 12);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(`${prefillAI.toFixed(0)} FLOP/B · ${prefillPerf.toFixed(0)} TFLOPS`, px, py + 18);

    // Plot Decode Point (Circle marker)
    const dx = xLogScale(decodeAI);
    const dy = yLogScale(decodePerf);
    ctx.beginPath();
    ctx.arc(dx, dy, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText('🐢 DECODE (Memory-Bound)', dx + 60, dy - 12);
    ctx.fillText(`${decodeAI.toFixed(1)} FLOP/B · ${decodePerf.toFixed(1)} TFLOPS`, dx + 60, dy + 18);

    // Draw connecting dashed line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(dx, dy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.font = '500 12px "Inter", system-ui, sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText('Arithmetic Intensity (FLOPs / Byte transferred)', pad.left + plotW / 2, height - 8);

    ctx.save();
    ctx.translate(16, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Attained Throughput (TFLOPs)', 0, 0);
    ctx.restore();

    // Title / Legend
    ctx.font = 'bold 12px "Inter", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('A100 GPU Roofline: Prefill vs Decode Arithmetic Intensity Gap', pad.left, pad.top - 12);
  }

  drawConcurrency() {
    const { ctx, width, height, seqLen } = this;
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    const pad = { top: 40, right: 30, bottom: 50, left: 70 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Sequence lengths from 1k to 128k
    const seqs = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
    // Max batch size on A100 (80GB) with Llama-3-8B (16GB weights, 64GB KV headroom)
    // KV per token = 0.5 MB -> at 4k: 2GB/stream -> Max Batch = 32
    // at 32k: 16GB/stream -> Max Batch = 4
    // at 128k: 64GB/stream -> Max Batch = 1 (OOM risk)
    const maxBatches = seqs.map(s => {
      const bytesPerSeq = 2 * 32 * 8 * 128 * 2 * s; // 8B model FP16
      const gbPerSeq = bytesPerSeq / (1024 * 1024 * 1024);
      return Math.max(0.5, Math.floor(60 / gbPerSeq));
    });

    const xScale = (i) => pad.left + (i / (seqs.length - 1)) * plotW;
    const maxB = 64;
    const yScale = (b) => pad.top + plotH - (Math.min(b, maxB) / maxB) * plotH;

    // Draw Max Batch Line (Solid White)
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i < seqs.length; i++) {
      const x = xScale(i);
      const y = yScale(maxBatches[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under curve
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.beginPath();
    ctx.moveTo(xScale(0), yScale(0));
    for (let i = 0; i < seqs.length; i++) {
      ctx.lineTo(xScale(i), yScale(maxBatches[i]));
    }
    ctx.lineTo(xScale(seqs.length - 1), yScale(0));
    ctx.closePath();
    ctx.fill();

    // Draw BDH Constant Concurrency Line (Dashed Line at B=64)
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, yScale(64));
    ctx.lineTo(pad.left + plotW, yScale(64));
    ctx.stroke();
    ctx.setLineDash([]);

    // Labels & Ticks
    ctx.font = '500 12px "Inter", sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.fillText('Context Length (T tokens)', pad.left + plotW / 2, height - 8);

    ctx.save();
    ctx.translate(16, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Max Concurrency (Batch Size B)', 0, 0);
    ctx.restore();

    // X Axis Ticks
    ctx.font = '10px "JetBrains Mono", monospace';
    for (let i = 0; i < seqs.length; i++) {
      const label = seqs[i] >= 1024 ? `${seqs[i] / 1024}k` : `${seqs[i]}`;
      ctx.fillText(label, xScale(i), height - pad.bottom + 18);
    }

    // Legend
    ctx.font = 'bold 11px "Inter", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Transformer: Max Batch Collapses as T Grows [Solid]', pad.left, pad.top - 16);
    ctx.fillText('BDH: Constant Batch Concurrency w.r.t. T [Dashed]', pad.left + 320, pad.top - 16);
  }

  destroy() {
    window.removeEventListener('resize', this._resizeHandler);
  }
}
