// memory_wall_canvas.js
// Canvas visualizer comparing KV-cache buffer growth vs fixed synaptic state.
// Used in Act I to demonstrate O(T·d) vs O(d²) memory scaling.

export class MemoryWallCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.data = { lengths: [], transformerMemory: [], bdhMemory: [] };
    this.animationProgress = 0;
    this.animating = false;
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

  setData(data) {
    this.data = data;
    this.draw();
  }

  animate(duration = 1500) {
    this.animationProgress = 0;
    this.animating = true;
    const start = performance.now();
    const tick = (now) => {
      this.animationProgress = Math.min(1, (now - start) / duration);
      this.draw();
      if (this.animationProgress < 1) requestAnimationFrame(tick);
      else this.animating = false;
    };
    requestAnimationFrame(tick);
  }

  draw() {
    const { ctx, width, height, data } = this;
    if (!data.lengths || data.lengths.length === 0) return;

    const pad = { top: 40, right: 30, bottom: 50, left: 70 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = 'rgba(15, 15, 25, 0.6)';
    ctx.beginPath();
    ctx.roundRect(0, 0, width, height, 12);
    ctx.fill();

    const maxT = Math.max(...data.lengths);
    const maxMem = Math.max(...data.transformerMemory);

    const xScale = (t) => pad.left + (t / maxT) * plotW;
    const yScale = (m) => pad.top + plotH - (m / maxMem) * plotH;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    const progress = this.animationProgress;
    const visibleCount = Math.ceil(data.lengths.length * progress);

    // Transformer memory line (red/orange gradient)
    ctx.strokeStyle = '#ff6b6b';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#ff6b6b';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    for (let i = 0; i < visibleCount; i++) {
      const x = xScale(data.lengths[i]);
      const y = yScale(data.transformerMemory[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // BDH memory line (green, flat)
    ctx.strokeStyle = '#51cf66';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#51cf66';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    for (let i = 0; i < visibleCount; i++) {
      const x = xScale(data.lengths[i]);
      const y = yScale(data.bdhMemory[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Fill under Transformer line
    ctx.fillStyle = 'rgba(255, 107, 107, 0.08)';
    ctx.beginPath();
    ctx.moveTo(xScale(data.lengths[0]), yScale(0));
    for (let i = 0; i < visibleCount; i++) {
      ctx.lineTo(xScale(data.lengths[i]), yScale(data.transformerMemory[i]));
    }
    if (visibleCount > 0) {
      ctx.lineTo(xScale(data.lengths[visibleCount - 1]), yScale(0));
    }
    ctx.closePath();
    ctx.fill();

    // Labels
    ctx.font = '13px "Inter", system-ui, sans-serif';
    ctx.fillStyle = '#adb5bd';
    ctx.textAlign = 'center';
    ctx.fillText('Sequence Length (T)', pad.left + plotW / 2, height - 8);

    ctx.save();
    ctx.translate(16, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Memory (bytes)', 0, 0);
    ctx.restore();

    // Legend
    const legendY = pad.top - 12;
    ctx.font = '12px "Inter", system-ui, sans-serif';

    ctx.fillStyle = '#ff6b6b';
    ctx.fillRect(pad.left, legendY - 8, 14, 3);
    ctx.fillStyle = '#e9ecef';
    ctx.textAlign = 'left';
    ctx.fillText('Transformer KV-Cache: O(T · d)', pad.left + 20, legendY - 3);

    ctx.fillStyle = '#51cf66';
    ctx.fillRect(pad.left + 230, legendY - 8, 14, 3);
    ctx.fillStyle = '#e9ecef';
    ctx.fillText('BDH Synaptic State: O(d²)', pad.left + 250, legendY - 3);

    // Axis ticks
    ctx.fillStyle = '#868e96';
    ctx.font = '11px "Inter", system-ui, sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
      const t = Math.round(maxT * i / 4);
      ctx.fillText(t, xScale(t), height - pad.bottom + 18);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const m = maxMem * (4 - i) / 4;
      const label = m >= 1024 ? (m / 1024).toFixed(1) + ' KB' : Math.round(m) + ' B';
      ctx.fillText(label, pad.left - 8, pad.top + (i / 4) * plotH + 4);
    }

    // Live memory counters
    if (visibleCount > 0) {
      const lastIdx = visibleCount - 1;
      const tMem = data.transformerMemory[lastIdx];
      const bMem = data.bdhMemory[lastIdx];

      ctx.font = 'bold 13px "Inter", system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillStyle = '#ff6b6b';
      ctx.fillText(
        (tMem / 1024).toFixed(1) + ' KB',
        xScale(data.lengths[lastIdx]) + 2,
        yScale(tMem) - 10
      );
      ctx.fillStyle = '#51cf66';
      ctx.fillText(
        (bMem / 1024).toFixed(1) + ' KB',
        xScale(data.lengths[lastIdx]) + 2,
        yScale(bMem) - 10
      );
    }
  }

  destroy() {
    window.removeEventListener('resize', this._resizeHandler);
  }
}
