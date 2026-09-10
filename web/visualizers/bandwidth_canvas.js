// bandwidth_canvas.js
// Visualizer for Chapter 4: Prefill vs Decode & Memory Bandwidth Roofline Bottleneck.
// Renders the analytical Roofline Model (Williams et al. 2009) for NVIDIA A100 80GB SXM.
// Ground truth specs:
//   - Peak Dense FP16 Tensor Core: 312 TFLOPS (312e12 FLOP/s)
//   - HBM2e Memory Bandwidth: 2,039 GB/s (2.039e12 B/s)
//   - Knee Point (Ridge Point): I* = 312 / 2.039 ≈ 153.016 FLOP/byte

export class BandwidthCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    // Hardware specifications
    this.gpuBandwidth = 2039; // GB/s (2.039 TB/s)
    this.gpuComputeTFLOPS = 312; // Dense FP16 Tensor Core peak TFLOPS
    this.kneeAI = this.gpuComputeTFLOPS / (this.gpuBandwidth / 1000); // ≈ 153.016 FLOP/B

    // Dynamic operating state
    this.batchSize = 1;
    this.contextLen = 1024;
    this.promptLen = 1024;
    
    // Model parameters: Llama-3-8B baseline
    this.modelParams = {
      P: 8.03e9,
      L: 32,
      n_Q: 32,
      n_KV: 8,
      d_head: 128,
      s: 2 // FP16 (2 bytes/element)
    };

    this._resizeHandler = () => this._resize();
    window.addEventListener('resize', this._resizeHandler);
    this._resize();
  }

  _resize() {
    if (!this.canvas) return;
    const parent = this.canvas.parentElement;
    const rect = parent ? parent.getBoundingClientRect() : { width: 700, height: 380 };
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width > 0 ? rect.width : 700;
    const h = rect.height > 100 ? rect.height : 380;

    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = w;
    this.height = h;
    this.draw();
  }

  updateState(batchSize, contextLen, promptLen = 1024) {
    this.batchSize = batchSize;
    this.contextLen = contextLen;
    this.promptLen = promptLen;
    this.draw();
  }

  // Analytical Prefill Arithmetic Intensity (FLOPs / Bytes transferred)
  getPrefillMetrics() {
    const { P, L, n_Q, d_head, s, n_KV } = this.modelParams;
    const T = this.promptLen;
    const k_token = 2 * L * n_KV * d_head * s; // 131,072 bytes/token

    // Weight projections + attention interactions
    const flops = 2 * P * T + 2 * L * n_Q * d_head * (T * T);
    // Idealized memory traffic: weights read once + KV cache written
    const bytes = 2 * P + k_token * T;
    const ai = flops / bytes;
    const perfTFlops = Math.min(this.gpuComputeTFLOPS, ai * (this.gpuBandwidth / 1000));
    return { ai, perfTFlops, flops, bytes };
  }

  // Analytical Decode Arithmetic Intensity (FLOPs / Bytes transferred)
  getDecodeMetrics() {
    const { P, L, n_Q, d_head, s, n_KV } = this.modelParams;
    const B = this.batchSize;
    const T = this.contextLen;
    const k_token = 2 * L * n_KV * d_head * s; // 131,072 bytes/token

    // Simplified analytical decode FLOP model:
    // 2PB (dense weights) + 4 B L n_Q d_head T (attention QK^T + SV)
    const flops = 2 * P * B + 4 * B * L * n_Q * d_head * T;
    // Idealized memory traffic: weights streamed once + full KV cache
    const bytes = 2 * P + B * k_token * T;
    const ai = flops / bytes;
    const perfTFlops = Math.min(this.gpuComputeTFLOPS, ai * (this.gpuBandwidth / 1000));
    return { ai, perfTFlops, flops, bytes };
  }

  draw() {
    if (!this.ctx) return;
    const { ctx, width, height } = this;
    ctx.clearRect(0, 0, width, height);

    // Clean white background (Monochromatic Light Theme)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const pad = { top: 40, right: 35, bottom: 55, left: 65 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    if (plotW <= 20 || plotH <= 20) return;

    // Log scales for Arithmetic Intensity (0.1 to 3,500 FLOP/Byte)
    const minLogAI = -1;   // log10(0.1) = -1
    const maxLogAI = 3.6;  // log10(~4000) ≈ 3.6

    // Log scales for Performance (0.1 to 400 TFLOPS)
    const minLogPerf = -1;  // log10(0.1) = -1
    const maxLogPerf = 2.65; // log10(~450) ≈ 2.65

    const xLogScale = (ai) => {
      const clamped = Math.max(0.1, Math.min(4000, ai));
      const log = Math.log10(clamped);
      return pad.left + ((log - minLogAI) / (maxLogAI - minLogAI)) * plotW;
    };

    const yLogScale = (perf) => {
      const clamped = Math.max(0.1, Math.min(450, perf));
      const log = Math.log10(clamped);
      return pad.top + plotH - ((log - minLogPerf) / (maxLogPerf - minLogPerf)) * plotH;
    };

    // Draw Grid Lines & Ticks
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.setLineDash([]);

    // Horizontal grid lines (0.1, 1, 10, 100, 312 TFLOPS)
    const yGridVals = [0.1, 1, 10, 100, 312];
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'right';

    yGridVals.forEach(val => {
      const y = yLogScale(val);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();

      const label = val === 312 ? '312' : `${val}`;
      ctx.fillText(label, pad.left - 8, y + 3);
    });

    // Vertical grid lines (0.1, 1, 10, 100, 153, 1000 FLOP/B)
    const xGridVals = [0.1, 1, 10, 100, 1000];
    ctx.textAlign = 'center';
    xGridVals.forEach(val => {
      const x = xLogScale(val);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();

      ctx.fillText(`${val}`, x, pad.top + plotH + 18);
    });

    // Fill under the roofline curve
    const kneeX = xLogScale(this.kneeAI);
    const kneeY = yLogScale(this.gpuComputeTFLOPS);
    const startX = xLogScale(0.1);
    const startY = yLogScale(0.1 * (this.gpuBandwidth / 1000));
    const endX = xLogScale(3500);
    const baseY = pad.top + plotH;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.03)';
    ctx.beginPath();
    ctx.moveTo(startX, baseY);
    ctx.lineTo(startX, startY);
    ctx.lineTo(kneeX, kneeY);
    ctx.lineTo(endX, kneeY);
    ctx.lineTo(endX, baseY);
    ctx.closePath();
    ctx.fill();

    // Draw Roofline Curve
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    // Memory-bound slope
    ctx.moveTo(startX, startY);
    ctx.lineTo(kneeX, kneeY);
    // Compute-bound flat ceiling
    ctx.lineTo(endX, kneeY);
    ctx.stroke();

    // Knee Point (Ridge Point) annotation
    ctx.fillStyle = '#000000';
    ctx.beginPath();
    ctx.arc(kneeX, kneeY, 4.5, 0, 2 * Math.PI);
    ctx.fill();

    // Knee dashed vertical line
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(kneeX, kneeY);
    ctx.lineTo(kneeX, baseY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Knee marker label
    ctx.font = '600 10px "Inter", sans-serif';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'left';
    ctx.fillText(`Knee: 153.0 FLOP/B`, kneeX + 8, kneeY + 14);
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = '#4b5563';
    ctx.fillText(`(312 TFLOPS)`, kneeX + 8, kneeY + 26);

    // Compute Dynamic Points
    const prefill = this.getPrefillMetrics();
    const decode = this.getDecodeMetrics();

    const px = xLogScale(prefill.ai);
    const py = yLogScale(prefill.perfTFlops);
    const dx = xLogScale(decode.ai);
    const dy = yLogScale(decode.perfTFlops);

    // Dashed trajectory connector between prefill and decode
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(dx, dy);
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Prefill Operating Point (Square Marker) ─────────────────
    ctx.fillStyle = '#000000';
    ctx.fillRect(px - 6, py - 6, 12, 12);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(px - 6, py - 6, 12, 12);

    ctx.font = 'bold 11px "Inter", sans-serif';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'center';
    ctx.fillText('⚡ PREFILL (Prompt Ingestion)', px, py - 18);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = '#374151';
    ctx.fillText(`${prefill.ai.toFixed(0)} FLOP/B · ${prefill.perfTFlops.toFixed(0)} TFLOPS`, px, py - 6);

    // ── Decode Operating Point (Circle Marker) ──────────────────
    ctx.fillStyle = '#000000';
    ctx.beginPath();
    ctx.arc(dx, dy, 7, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.font = 'bold 11px "Inter", sans-serif';
    ctx.fillStyle = '#000000';
    // Position decode text carefully so it doesn't clip
    const decodeAlignLeft = dx < pad.left + plotW * 0.45;
    if (decodeAlignLeft) {
      ctx.textAlign = 'left';
      ctx.fillText('🐢 DECODE (Token Generation)', dx + 12, dy - 6);
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#374151';
      ctx.fillText(`${decode.ai.toFixed(2)} FLOP/B · ${decode.perfTFlops.toFixed(2)} TFLOPS`, dx + 12, dy + 8);
    } else {
      ctx.textAlign = 'center';
      ctx.fillText('🐢 DECODE (Token Generation)', dx, dy - 16);
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#374151';
      ctx.fillText(`${decode.ai.toFixed(2)} FLOP/B · ${decode.perfTFlops.toFixed(2)} TFLOPS`, dx, dy - 4);
    }

    // Regime Labels
    ctx.font = '600 11px "Inter", sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'left';
    ctx.fillText('◄ MEMORY-BANDWIDTH BOUND (Slope = 2,039 GB/s)', pad.left + 15, pad.top + 55);

    ctx.textAlign = 'right';
    ctx.fillText('COMPUTE BOUND (Peak = 312 TFLOPS) ►', pad.left + plotW - 10, pad.top + 20);

    // Axis Titles
    ctx.font = '600 11px "Inter", sans-serif';
    ctx.fillStyle = '#111827';
    ctx.textAlign = 'center';
    ctx.fillText('Arithmetic Intensity I (FLOPs / Byte transferred, log scale)', pad.left + plotW / 2, height - 12);

    ctx.save();
    ctx.translate(18, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Attained Performance P (TFLOPS, log scale)', 0, 0);
    ctx.restore();

    // Chart Title
    ctx.font = 'bold 12px "Inter", sans-serif';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'left';
    ctx.fillText('A100 SXM4 Roofline: Analytical Prefill vs. Decode Operating Regimes', pad.left, pad.top - 15);
  }

  destroy() {
    window.removeEventListener('resize', this._resizeHandler);
  }
}
