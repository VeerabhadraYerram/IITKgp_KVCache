// benchmark_canvas.js
// 🟡 2D QUANTITATIVE RUNTIME MEASUREMENT VISUALIZER
// Renders actual measured browser execution time (ms via performance.now())
// Strict monochromatic Light Theme: #ffffff background, solid/dashed black lines.

export class BenchmarkCanvas {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = null;
    this.hoverStep = null;

    this._setupEvents();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = rect.height;
    this.render();
  }

  setData(measuredData) {
    this.data = measuredData;
    this.render();
  }

  _setupEvents() {
    this.canvas.addEventListener('mousemove', (e) => {
      if (!this.data || !this.data.noCacheCumTimes) return;
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const padding = { top: 40, right: 30, bottom: 50, left: 65 };
      const chartWidth = this.width - padding.left - padding.right;

      if (mouseX >= padding.left && mouseX <= this.width - padding.right) {
        const frac = (mouseX - padding.left) / chartWidth;
        const step = Math.max(1, Math.min(this.data.numTokens, Math.round(frac * (this.data.numTokens - 1)) + 1));
        this.hoverStep = step;
        this.render();
      } else {
        if (this.hoverStep !== null) {
          this.hoverStep = null;
          this.render();
        }
      }
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.hoverStep = null;
      this.render();
    });
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    if (!ctx || !w || !h) return;

    // Reset transform & clear
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const dpr = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, w * dpr, h * dpr);
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    const padding = { top: 40, right: 30, bottom: 50, left: 65 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    if (!this.data || !this.data.noCacheCumTimes || this.data.numTokens === 0) {
      // Empty state placeholder
      ctx.fillStyle = '#888888';
      ctx.font = '13px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Click "Run Live Browser Benchmark" to record real timings', w / 2, h / 2);
      ctx.restore();
      return;
    }

    const N = this.data.numTokens;
    const maxTime = Math.max(
      this.data.noCacheCumTimes[N - 1] || 1,
      this.data.cachedCumTimes[N - 1] || 1,
      0.01
    ) * 1.1; // 10% headroom

    // Coordinates mapper
    const getX = (step) => padding.left + ((step - 1) / (N - 1 || 1)) * chartW;
    const getY = (timeMs) => padding.top + chartH - (timeMs / maxTime) * chartH;

    // Grid lines (horizontal)
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#eeeeee';
    const numYGrid = 5;
    for (let i = 0; i <= numYGrid; i++) {
      const val = (maxTime / numYGrid) * i;
      const y = getY(val);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartW, y);
      ctx.stroke();

      // Label
      ctx.fillStyle = '#666666';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(val >= 1 ? val.toFixed(1) + ' ms' : val.toFixed(2) + ' ms', padding.left - 10, y);
    }

    // Grid lines (vertical)
    const numXGrid = Math.min(8, N);
    const stepInterval = Math.max(1, Math.floor(N / numXGrid));
    for (let step = 1; step <= N; step += stepInterval) {
      const x = getX(step);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartH);
      ctx.stroke();

      ctx.fillStyle = '#666666';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText('t=' + step, x, padding.top + chartH + 8);
    }

    // Axis lines
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartH);
    ctx.lineTo(padding.left + chartW, padding.top + chartH);
    ctx.stroke();

    // Axis titles
    ctx.fillStyle = '#000000';
    ctx.font = '11px "Inter", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Sequence Step (t = 1 … N)', padding.left + chartW / 2, padding.top + chartH + 34);

    ctx.save();
    ctx.translate(16, padding.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Measured Cumulative Time (ms)', 0, 0);
    ctx.restore();

    // ─────────────────────────────────────────────────────────────
    // CURVE 1: NO CACHE (Solid Black Line)
    // ─────────────────────────────────────────────────────────────
    ctx.save();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let t = 1; t <= N; t++) {
      const x = getX(t);
      const y = getY(this.data.noCacheCumTimes[t - 1]);
      if (t === 1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // ─────────────────────────────────────────────────────────────
    // CURVE 2: WITH KV CACHE (Dashed Black Line)
    // ─────────────────────────────────────────────────────────────
    ctx.save();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    for (let t = 1; t <= N; t++) {
      const x = getX(t);
      const y = getY(this.data.cachedCumTimes[t - 1]);
      if (t === 1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // ─────────────────────────────────────────────────────────────
    // LEGEND
    // ─────────────────────────────────────────────────────────────
    const legendX = padding.left + 16;
    const legendY = padding.top + 16;

    // Solid
    ctx.save();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY);
    ctx.lineTo(legendX + 28, legendY);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = '#000000';
    ctx.font = '12px "Inter", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('No Cache — Cumulative Runtime (ms)', legendX + 36, legendY);

    // Dashed
    ctx.save();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(legendX, legendY + 20);
    ctx.lineTo(legendX + 28, legendY + 20);
    ctx.stroke();
    ctx.restore();

    ctx.fillText('With KV Cache — Cumulative Runtime (ms)', legendX + 36, legendY + 20);

    // ─────────────────────────────────────────────────────────────
    // HOVER TOOLTIP / INSPECTION
    // ─────────────────────────────────────────────────────────────
    if (this.hoverStep !== null && this.hoverStep >= 1 && this.hoverStep <= N) {
      const step = this.hoverStep;
      const x = getX(step);
      const yNaive = getY(this.data.noCacheCumTimes[step - 1]);
      const yCached = getY(this.data.cachedCumTimes[step - 1]);

      // Vertical guide
      ctx.save();
      ctx.strokeStyle = '#aaaaaa';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartH);
      ctx.stroke();
      ctx.restore();

      // Points on curves
      ctx.fillStyle = '#000000';
      ctx.beginPath();
      ctx.arc(x, yNaive, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, yCached, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Tooltip Card
      const naiveVal = this.data.noCacheCumTimes[step - 1].toFixed(2);
      const cachedVal = this.data.cachedCumTimes[step - 1].toFixed(2);
      const speedupVal = (this.data.noCacheCumTimes[step - 1] / Math.max(0.001, this.data.cachedCumTimes[step - 1])).toFixed(1);

      const tipW = 190;
      const tipH = 76;
      let tipX = x + 12;
      if (tipX + tipW > padding.left + chartW) tipX = x - tipW - 12;
      const tipY = Math.min(padding.top + chartH - tipH - 5, Math.max(padding.top + 5, yCached - 30));

      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.fillRect(tipX, tipY, tipW, tipH);
      ctx.strokeRect(tipX, tipY, tipW, tipH);

      ctx.fillStyle = '#000000';
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(`Step t = ${step} / ${N}`, tipX + 10, tipY + 8);

      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(`No Cache:  ${naiveVal} ms`, tipX + 10, tipY + 26);
      ctx.fillText(`KV Cache:  ${cachedVal} ms`, tipX + 10, tipY + 42);
      ctx.fillText(`Ratio:     ${speedupVal}× faster`, tipX + 10, tipY + 58);
    }

    ctx.restore();
  }
}
