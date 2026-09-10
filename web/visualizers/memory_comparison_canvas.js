// memory_comparison_canvas.js
// Visualizer for Chapter 5: Alternate Memory Approaches, Compared Honestly
// Compares memory footprint (MB/GB decimal and MiB/GiB binary) across sequence length T in [512, 131,072].
// Implements dual-scale rendering (Logarithmic Y-axis by default, Linear Y-axis toggleable).

export const APPROACHES = {
  standard_kv: {
    id: 'standard_kv',
    name: 'Standard KV-Cache',
    shortName: 'Standard KV',
    badge: 'O(T) Linear',
    badgeClass: 'linear',
    stateCategory: 'Explicit Token Memory (Token-Addressable)',
    tokenAddressable: 'Yes (all positions)',
    complexityStr: 'O(T) linear with respect to sequence length T for fixed model dimensions',
    formulaStr: 'M = 2 · L · n_KV · d_head · s · T = 131,072 · T bytes',
    normalizedBytes: (T) => 131072 * T,
    assumptions: 'Normalized L=32, n_KV=8, d_head=128, FP16 (s=2 bytes). Represents cache-state memory scaling only, not total model inference footprint.',
    informationRetained: 'Full, uncompressed past Key and Value activation projections for every token in the context window.',
    accessMechanism: 'Exact query-key dot-product attention with softmax normalization over all historical positions.',
    limitations: 'Unbounded linear growth with context length; acute memory-bandwidth saturation during autoregressive decode (Chapter 4).',
    citation: 'Vaswani et al. 2017 · Pope et al. 2023'
  },
  streaming_llm: {
    id: 'streaming_llm',
    name: 'Eviction / Attention Sinks (StreamingLLM & H2O)',
    shortName: 'Attention Sinks',
    badge: 'O(1) Fixed Window',
    badgeClass: 'constant',
    stateCategory: 'Explicit Token Memory (Windowed / Subsampled)',
    tokenAddressable: 'Yes (retained subset only)',
    complexityStr: 'O(1) with respect to sequence length T for fixed cache budget W',
    formulaStr: 'M = 131,072 · min(T, 1028) bytes [StreamingLLM pedagogical config]',
    normalizedBytes: (T) => 131072 * Math.min(T, 1028),
    assumptions: 'StreamingLLM pedagogical configuration: 4 initial sink tokens + 1024 local sliding window (W=1028). H2O uses a configurable budget W_H2O based on cumulative attention scores.',
    informationRetained: 'Initial anchor/sink tokens + recent local sliding window. Middle historical tokens are evicted.',
    accessMechanism: 'Standard softmax attention restricted to retained initial sinks and local window.',
    limitations: 'Once a token\'s KV state is evicted, this cache mechanism no longer has direct access to that token\'s stored K/V representation. Exact recovery is therefore unavailable unless information was preserved elsewhere or can be reconstructed.',
    citation: 'StreamingLLM: Xiao et al. 2023 · H2O: Zhang et al. 2023'
  },
  deepseek_mla: {
    id: 'deepseek_mla',
    name: 'Low-Rank / Latent Compression (DeepSeek MLA)',
    shortName: 'DeepSeek MLA',
    badge: 'O(T) Compressed',
    badgeClass: 'linear',
    stateCategory: 'Compressed Token Memory (Token-Addressable Latents)',
    tokenAddressable: 'Yes (via latent vector)',
    complexityStr: 'O(T) linear with respect to sequence length T, but with compressed latent dimension',
    formulaStr: 'M = L · (d_c + d_R) · s · T = 36,864 · T bytes [Normalized to L=32]',
    normalizedBytes: (T) => 36864 * T,
    assumptions: 'DeepSeek-V2 MLA mechanism normalized to L=32: 512D compressed KV latent + 64D decoupled RoPE (576 elements/token/layer). Not the literal DeepSeek-V2 model footprint (which has 60 layers).',
    informationRetained: 'Low-rank compressed latent representation (c_t^KV ∈ ℝ⁵¹²) plus decoupled rotary key (k_t^R ∈ ℝ⁶⁴).',
    accessMechanism: 'Latent vectors are projected on-the-fly or mathematically absorbed into projection weight matrices.',
    limitations: 'Reduces footprint per token by 71.875% vs standard KV, but does NOT make memory constant in T. Memory still grows linearly without bound over long contexts.',
    citation: 'DeepSeek-AI 2024 (DeepSeek-V2)'
  },
  linear_attention: {
    id: 'linear_attention',
    name: 'Linear Attention (Fast Weight Programmers)',
    shortName: 'Linear Attention',
    badge: 'O(1) Recurrent State',
    badgeClass: 'constant',
    stateCategory: 'Recurrent Associative State (Fixed Matrix Memory)',
    tokenAddressable: 'No (folded into associative state)',
    complexityStr: 'O(1) with respect to sequence length T for fixed feature dim r and value dim d_v',
    formulaStr: 'M = L · n_head · r · d_v · s = 33,554,432 bytes [r = d_v = 128]',
    normalizedBytes: (T) => 33554432,
    assumptions: 'Normalized to L=32, n_head=32, feature dim r=128, value dim d_v=128, FP16 (s=2). Fixed recurrent state; size depends on feature dimension r and value dimension d_v.',
    informationRetained: 'Cumulative outer-product associative matrix: S_t = S_{t-1} + φ(k_t) v_tᵀ.',
    accessMechanism: 'Recurrent matrix-vector query projection q_t S_t exploiting associativity of matrix multiplication.',
    limitations: 'Replaces exact softmax attention with a kernelized/linearized attention mechanism, changing the retrieval behavior and introducing capacity/interference trade-offs in its finite recurrent state.',
    citation: 'Katharopoulos et al. 2020 · Schlag, Irie, & Schmidhuber 2021'
  },
  mamba_ssm: {
    id: 'mamba_ssm',
    name: 'State-Space Models (Mamba)',
    shortName: 'Mamba SSM',
    badge: 'O(1) Dynamical State',
    badgeClass: 'constant',
    stateCategory: 'Dynamical Recurrent State (Hidden Vector Memory)',
    tokenAddressable: 'No (folded into continuous state)',
    complexityStr: 'O(1) with respect to sequence length T for fixed model and state dimensions',
    formulaStr: 'M = L · d_model · d_state · s = 4,194,304 bytes [d_state = 16]',
    normalizedBytes: (T) => 4194304,
    assumptions: 'Mamba-style SSM recurrent state — simplified state-only estimate: L=32, d_model=4096, d_state=16 per channel, FP16 (s=2). Excludes implementation and convolutional buffers.',
    informationRetained: 'Continuous-time hidden state evolving via input-dependent selective discretization: h_t = Ā_t h_{t-1} + B̄_t x_t.',
    accessMechanism: 'Linear output projection y_t = C̄_t h_t + D x_t.',
    limitations: 'Fixed-size recurrent state can make exact long-range retrieval and associative recall more difficult than explicit token-addressable memory; performance depends strongly on architecture, training, and task.',
    citation: 'Gu & Dao 2023 (Mamba)'
  },
  bdh_graph: {
    id: 'bdh_graph',
    name: 'Dragon Hatchling (BDH)',
    shortName: 'BDH Graph',
    badge: 'O(1) Graph Memory',
    badgeClass: 'constant',
    stateCategory: 'Learned Graph Memory (Dynamical Graph State)',
    tokenAddressable: 'No (distributed synaptic edge weights)',
    complexityStr: 'O(1) with respect to sequence length T for fixed graph architecture',
    formulaStr: 'M = Constant graph state footprint [See Chapter 6 for full treatment]',
    normalizedBytes: (T) => 8388608, // 8.0 MiB illustrative fixed graph state
    assumptions: 'Fixed graph state; sequence-length-independent state footprint under a fixed architecture. See Chapter 6 for full treatment.',
    informationRetained: 'Synaptic edge connection weights and dynamic activations on a learned bipartite network graph.',
    accessMechanism: 'Hebbian edge reweighting and sparse graph activation propagation.',
    limitations: 'Explored fully in Chapter 6; trades explicit token buffers for a dynamical graph memory structure.',
    citation: 'Teaser for Chapter 6'
  }
};

export class MemoryComparisonCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.approachA = 'standard_kv';
    this.approachB = 'mamba_ssm';
    this.scaleType = 'log'; // 'log' or 'linear'
    this.activeT = 65536;

    this.seqRange = [512, 131072];

    this._resizeHandler = () => this._resize();
    window.addEventListener('resize', this._resizeHandler);

    this._setupInteractions();
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

  _setupInteractions() {
    const handleMove = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const x = clientX - rect.left;
      const pad = { top: 40, right: 40, bottom: 50, left: 75 };
      const plotW = this.width - pad.left - pad.right;

      if (x >= pad.left && x <= pad.left + plotW) {
        const ratio = (x - pad.left) / plotW;
        // Map ratio to sequence length T (log or linear)
        const minLog = Math.log10(this.seqRange[0]);
        const maxLog = Math.log10(this.seqRange[1]);
        const logT = minLog + ratio * (maxLog - minLog);
        this.activeT = Math.round(Math.pow(10, logT));
        this.draw();
        if (this.onHoverT) this.onHoverT(this.activeT);
      }
    };

    this.canvas.addEventListener('mousemove', handleMove);
    this.canvas.addEventListener('touchmove', handleMove);
  }

  setApproaches(idA, idB) {
    if (APPROACHES[idA]) this.approachA = idA;
    if (APPROACHES[idB]) this.approachB = idB;
    this.draw();
  }

  setScaleType(scale) {
    this.scaleType = scale === 'linear' ? 'linear' : 'log';
    this.draw();
  }

  draw() {
    if (!this.ctx) return;
    const { ctx, width, height } = this;
    ctx.clearRect(0, 0, width, height);

    // Clean white background (Swiss Monochromatic Light Theme)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const pad = { top: 40, right: 40, bottom: 50, left: 80 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    if (plotW <= 20 || plotH <= 20) return;

    const [minT, maxT] = this.seqRange;
    const minLogT = Math.log10(minT);
    const maxLogT = Math.log10(maxT);

    const xLogScale = (t) => {
      const clamped = Math.max(minT, Math.min(maxT, t));
      const log = Math.log10(clamped);
      return pad.left + ((log - minLogT) / (maxLogT - minLogT)) * plotW;
    };

    // Range bounds for Y axis (bytes)
    // In log scale: from 1 MB (1e6 bytes) to 32 GB (32e9 bytes)
    const minLogY = 6;     // 10^6 = 1 MB
    const maxLogY = 10.55; // 10^10.55 ≈ 35.5 GB
    const maxLinearY = 20 * 1e9; // 20 GB decimal

    const yLogScale = (bytes) => {
      const clamped = Math.max(1e6, Math.min(3.5e10, bytes));
      const log = Math.log10(clamped);
      return pad.top + plotH - ((log - minLogY) / (maxLogY - minLogY)) * plotH;
    };

    const yLinearScale = (bytes) => {
      const clamped = Math.max(0, Math.min(maxLinearY, bytes));
      return pad.top + plotH - (clamped / maxLinearY) * plotH;
    };

    const yScale = this.scaleType === 'log' ? yLogScale : yLinearScale;

    // Grid lines & Ticks
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.setLineDash([]);

    // Y Grid lines
    if (this.scaleType === 'log') {
      const yLogGrid = [
        { bytes: 1e6, label: '1 MB' },
        { bytes: 10e6, label: '10 MB' },
        { bytes: 100e6, label: '100 MB' },
        { bytes: 1e9, label: '1 GB' },
        { bytes: 10e9, label: '10 GB' },
        { bytes: 20e9, label: '20 GB' }
      ];

      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#6b7280';
      ctx.textAlign = 'right';

      yLogGrid.forEach(item => {
        const y = yScale(item.bytes);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + plotW, y);
        ctx.stroke();
        ctx.fillText(item.label, pad.left - 8, y + 3);
      });
    } else {
      const yLinearGrid = [
        { bytes: 0, label: '0 GB' },
        { bytes: 5e9, label: '5 GB' },
        { bytes: 10e9, label: '10 GB' },
        { bytes: 15e9, label: '15 GB' },
        { bytes: 20e9, label: '20 GB' }
      ];

      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#6b7280';
      ctx.textAlign = 'right';

      yLinearGrid.forEach(item => {
        const y = yScale(item.bytes);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + plotW, y);
        ctx.stroke();
        ctx.fillText(item.label, pad.left - 8, y + 3);
      });
    }

    // X Grid lines (512, 2k, 8k, 32k, 128k)
    const xGridVals = [512, 2048, 8192, 32768, 131072];
    ctx.textAlign = 'center';
    xGridVals.forEach(val => {
      const x = xLogScale(val);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();

      const label = val >= 1024 ? `${val / 1024}k` : `${val}`;
      ctx.fillText(label, x, pad.top + plotH + 18);
    });

    // Sample points for curve generation
    const sampleTs = [];
    let curr = minT;
    while (curr <= maxT) {
      sampleTs.push(curr);
      curr = Math.round(curr * 1.15);
    }
    if (sampleTs[sampleTs.length - 1] !== maxT) sampleTs.push(maxT);

    const appA = APPROACHES[this.approachA];
    const appB = APPROACHES[this.approachB];

    // Helper to draw an approach curve
    const drawCurve = (app, isPrimary) => {
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = isPrimary ? 3 : 2;
      ctx.setLineDash(isPrimary ? [] : [6, 4]);

      ctx.beginPath();
      sampleTs.forEach((t, idx) => {
        const bytes = app.normalizedBytes(t);
        const x = xLogScale(t);
        const y = yScale(bytes);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // Draw curves
    if (appB) drawCurve(appB, false);
    if (appA) drawCurve(appA, true);

    // Active T Scrubber Line
    const scrubX = xLogScale(this.activeT);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(scrubX, pad.top);
    ctx.lineTo(scrubX, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw active point markers on the curves
    const drawActivePoint = (app, isPrimary) => {
      const bytes = app.normalizedBytes(this.activeT);
      const px = xLogScale(this.activeT);
      const py = yScale(bytes);

      if (isPrimary) {
        // Solid square
        ctx.fillStyle = '#000000';
        ctx.fillRect(px - 5, py - 5, 10, 10);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(px - 5, py - 5, 10, 10);
      } else {
        // Hollow circle
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }

      // Value Callout Pill
      const mbDecimal = (bytes / 1e6).toFixed(bytes >= 1e9 ? 2 : 1);
      const gibBinary = (bytes / (1024 * 1024 * 1024)).toFixed(2);
      const mibBinary = (bytes / (1024 * 1024)).toFixed(1);
      const text = bytes >= 1e9 
        ? `${(bytes / 1e9).toFixed(2)} GB (${gibBinary} GiB)`
        : `${mbDecimal} MB (${mibBinary} MiB)`;

      ctx.font = 'bold 10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#000000';
      ctx.textAlign = px > pad.left + plotW * 0.6 ? 'right' : 'left';
      const offsetX = px > pad.left + plotW * 0.6 ? -12 : 12;
      ctx.fillText(text, px + offsetX, py + 3);
    };

    if (appB) drawActivePoint(appB, false);
    if (appA) drawActivePoint(appA, true);

    // Axis Titles
    ctx.font = '600 11px "Inter", sans-serif';
    ctx.fillStyle = '#111827';
    ctx.textAlign = 'center';
    ctx.fillText('Context Sequence Length T (tokens, log scale)', pad.left + plotW / 2, height - 12);

    ctx.save();
    ctx.translate(18, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    const yAxisTitle = this.scaleType === 'log'
      ? 'Normalized State Footprint (Log Scale: MB / GB)'
      : 'Normalized State Footprint (Linear Scale: 0 to 20 GB)';
    ctx.fillText(yAxisTitle, 0, 0);
    ctx.restore();

    // Chart Legend Header
    ctx.font = 'bold 12px "Inter", sans-serif';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'left';
    ctx.fillText('Analytical State Memory Scaling vs. Context Length T', pad.left, pad.top - 15);

    // Dynamic Legend Indicator
    ctx.font = '600 11px "Inter", sans-serif';
    ctx.textAlign = 'right';
    
    // Approach A
    ctx.fillStyle = '#000000';
    ctx.fillRect(pad.left + plotW - 220, pad.top - 20, 12, 12);
    ctx.fillText(`A: ${appA.shortName}`, pad.left + plotW - 130, pad.top - 10);

    // Approach B
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pad.left + plotW - 110, pad.top - 14, 5, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillText(`B: ${appB.shortName}`, pad.left + plotW - 15, pad.top - 10);
  }

  destroy() {
    window.removeEventListener('resize', this._resizeHandler);
  }
}
