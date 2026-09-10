// surgery_hud.js
// The Crown Jewel: 3-knob interactive sandbox (Density, Capacity, Overlap)
// with live Recall–Interference curve and analytical E[overlap] ∝ p² overlay.

import { sweepDensityCurve, runRecallExperiment } from '../engine/memory_experiment.js';

export class SurgeryHUD {
  constructor(canvasId, controlsContainerId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.controlsContainer = document.getElementById(controlsContainerId);

    // Default parameters
    this.params = {
      density: 0.05,
      capacity: 128,  // N dimension
      keySimilarity: 0.0,
      D: 32,
      count: 8,
      seed: 12345,
    };

    // Cached curve data
    this.curveData = null;
    // Current single experiment result
    this.currentResult = null;

    this._resizeHandler = () => this._resize();
    window.addEventListener('resize', this._resizeHandler);
    this._resize();
    this._buildControls();
    this.runExperiment();
    this.runCurve();
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = (rect.height || 320) * dpr;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = (rect.height || 320) + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = rect.width;
    this.height = rect.height || 320;
    this.drawCurve();
  }

  _buildControls() {
    this.controlsContainer.innerHTML = '';
    this.controlsContainer.className = 'surgery-controls';

    const knobs = [
      {
        id: 'density', label: 'Representation Density',
        min: 0.02, max: 1.0, step: 0.01, value: this.params.density,
        format: v => `${(v * 100).toFixed(0)}% active`,
        labelLeft: 'Sparse', labelRight: 'Dense',
      },
      {
        id: 'capacity', label: 'Memory Capacity (N)',
        min: 16, max: 256, step: 16, value: this.params.capacity,
        format: v => `N = ${v}`,
        labelLeft: 'Small', labelRight: 'Large',
      },
      {
        id: 'keySimilarity', label: 'Key Similarity / Overlap',
        min: 0.0, max: 0.9, step: 0.05, value: this.params.keySimilarity,
        format: v => `${(v * 100).toFixed(0)}% similar`,
        labelLeft: 'Orthogonal', labelRight: 'Correlated',
      },
    ];

    for (const knob of knobs) {
      const group = document.createElement('div');
      group.className = 'knob-group';

      const labelRow = document.createElement('div');
      labelRow.className = 'knob-label-row';
      const labelEl = document.createElement('span');
      labelEl.className = 'knob-label';
      labelEl.textContent = knob.label;
      const valueEl = document.createElement('span');
      valueEl.className = 'knob-value';
      valueEl.id = `knob-val-${knob.id}`;
      valueEl.textContent = knob.format(knob.value);
      labelRow.appendChild(labelEl);
      labelRow.appendChild(valueEl);

      const sliderRow = document.createElement('div');
      sliderRow.className = 'knob-slider-row';
      const leftLabel = document.createElement('span');
      leftLabel.className = 'knob-range-label';
      leftLabel.textContent = knob.labelLeft;
      const input = document.createElement('input');
      input.type = 'range';
      input.min = knob.min;
      input.max = knob.max;
      input.step = knob.step;
      input.value = knob.value;
      input.className = 'knob-slider';
      input.id = `knob-${knob.id}`;
      const rightLabel = document.createElement('span');
      rightLabel.className = 'knob-range-label';
      rightLabel.textContent = knob.labelRight;
      sliderRow.appendChild(leftLabel);
      sliderRow.appendChild(input);
      sliderRow.appendChild(rightLabel);

      group.appendChild(labelRow);
      group.appendChild(sliderRow);
      this.controlsContainer.appendChild(group);

      // Event listener
      input.addEventListener('input', () => {
        const val = knob.id === 'capacity' ? parseInt(input.value) : parseFloat(input.value);
        this.params[knob.id] = val;
        valueEl.textContent = knob.format(val);
        this.runExperiment();
        this.runCurve();
      });
    }

    // HUD metrics display
    const hudEl = document.createElement('div');
    hudEl.className = 'surgery-hud-metrics';
    hudEl.id = 'surgery-hud-metrics';
    this.controlsContainer.appendChild(hudEl);
  }

  runExperiment() {
    this.currentResult = runRecallExperiment({
      D: this.params.D,
      N: this.params.capacity,
      density: this.params.density,
      count: this.params.count,
      seed: this.params.seed,
      keySimilarity: this.params.keySimilarity,
    });
    this.updateHUD();
    this.dispatchResultEvent();
  }

  runCurve() {
    this.curveData = sweepDensityCurve({
      D: this.params.D,
      N: this.params.capacity,
      count: this.params.count,
      seed: this.params.seed,
      keySimilarity: this.params.keySimilarity,
      steps: 25,
    });
    this.drawCurve();
  }

  updateHUD() {
    const el = document.getElementById('surgery-hud-metrics');
    if (!el || !this.currentResult) return;

    const r = this.currentResult;
    const accuracyPct = (r.overallAccuracy * 100).toFixed(0);
    const load = (this.params.count / this.params.capacity * 100).toFixed(0);
    const avgInterference = r.crimeScenes.reduce((sum, cs) => {
      return sum + (cs.overlaps.length > 0 ? cs.overlaps[0].overlapFraction : 0);
    }, 0) / r.crimeScenes.length;

    const accuracyColor = r.overallAccuracy >= 0.8 ? '#51cf66' : r.overallAccuracy >= 0.5 ? '#ffd43b' : '#ff6b6b';
    const interferenceColor = avgInterference < 0.2 ? '#51cf66' : avgInterference < 0.5 ? '#ffd43b' : '#ff6b6b';

    el.innerHTML = `
      <div class="hud-metric">
        <span class="hud-metric-label">Capacity Load</span>
        <div class="hud-bar-track"><div class="hud-bar-fill" style="width:${load}%;background:${accuracyColor}"></div></div>
        <span class="hud-metric-value">${load}%</span>
      </div>
      <div class="hud-metric">
        <span class="hud-metric-label">Avg Interference</span>
        <div class="hud-bar-track"><div class="hud-bar-fill" style="width:${(avgInterference*100).toFixed(0)}%;background:${interferenceColor}"></div></div>
        <span class="hud-metric-value">${(avgInterference*100).toFixed(0)}%</span>
      </div>
      <div class="hud-metric">
        <span class="hud-metric-label">Recall Accuracy</span>
        <div class="hud-bar-track"><div class="hud-bar-fill" style="width:${accuracyPct}%;background:${accuracyColor}"></div></div>
        <span class="hud-metric-value" style="color:${accuracyColor}">${accuracyPct}%</span>
      </div>
      <div class="hud-metric">
        <span class="hud-metric-label">Memory</span>
        <span class="hud-metric-value" style="color:#74c0fc">${(r.memoryBytes / 1024).toFixed(1)} KB (constant in T)</span>
      </div>
    `;
  }

  drawCurve() {
    const { ctx, width, height, curveData } = this;
    if (!curveData) return;

    const pad = { top: 35, right: 20, bottom: 50, left: 60 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(15, 15, 25, 0.6)';
    ctx.beginPath();
    ctx.roundRect(0, 0, width, height, 12);
    ctx.fill();

    const { densities, accuracies, analyticalOverlaps, interferences } = curveData;

    // Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    const xScale = (p) => pad.left + (p / 1.0) * plotW;
    const yScale = (v) => pad.top + plotH - v * plotH;

    // Recall accuracy curve (green)
    ctx.strokeStyle = '#51cf66';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = '#51cf66';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    for (let i = 0; i < densities.length; i++) {
      const x = xScale(densities[i]);
      const y = yScale(accuracies[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Interference curve (red)
    ctx.strokeStyle = '#ff6b6b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < densities.length; i++) {
      const x = xScale(densities[i]);
      const y = yScale(interferences[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Analytical p² overlap (dashed white)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    for (let i = 0; i < densities.length; i++) {
      const x = xScale(densities[i]);
      const y = yScale(analyticalOverlaps[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Current density marker (vertical line)
    const curX = xScale(this.params.density);
    ctx.strokeStyle = 'rgba(255, 224, 102, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(curX, pad.top);
    ctx.lineTo(curX, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Current density label
    ctx.font = 'bold 11px "Inter", system-ui, sans-serif';
    ctx.fillStyle = '#ffe066';
    ctx.textAlign = 'center';
    ctx.fillText(`p = ${(this.params.density * 100).toFixed(0)}%`, curX, pad.top - 5);

    // BDH observed region (shaded band around 5%)
    const bdhLeft = xScale(0.03);
    const bdhRight = xScale(0.08);
    ctx.fillStyle = 'rgba(116, 192, 252, 0.08)';
    ctx.fillRect(bdhLeft, pad.top, bdhRight - bdhLeft, plotH);
    ctx.font = '10px "Inter", system-ui, sans-serif';
    ctx.fillStyle = 'rgba(116, 192, 252, 0.6)';
    ctx.textAlign = 'center';
    ctx.fillText('Observed BDH (~5%)', (bdhLeft + bdhRight) / 2, pad.top + plotH - 6);

    // Axis labels
    ctx.font = '13px "Inter", system-ui, sans-serif';
    ctx.fillStyle = '#adb5bd';
    ctx.textAlign = 'center';
    ctx.fillText('Active Neuron Density (p)', pad.left + plotW / 2, height - 8);
    ctx.save();
    ctx.translate(14, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Fraction', 0, 0);
    ctx.restore();

    // Axis ticks
    ctx.font = '11px "Inter", system-ui, sans-serif';
    ctx.fillStyle = '#868e96';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const p = i * 0.2;
      ctx.fillText(`${(p * 100).toFixed(0)}%`, xScale(p), height - pad.bottom + 18);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      ctx.fillText(`${((4 - i) * 25)}%`, pad.left - 8, pad.top + (i / 4) * plotH + 4);
    }

    // Legend
    ctx.font = '11px "Inter", system-ui, sans-serif';
    ctx.textAlign = 'left';
    const lx = pad.left + 10;
    let ly = pad.top + 14;

    ctx.fillStyle = '#51cf66';
    ctx.fillRect(lx, ly - 6, 12, 3);
    ctx.fillText('Recall Accuracy', lx + 18, ly);
    ly += 16;
    ctx.fillStyle = '#ff6b6b';
    ctx.fillRect(lx, ly - 6, 12, 3);
    ctx.fillText('Interference', lx + 18, ly);
    ly += 16;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(lx, ly - 5);
    ctx.lineTo(lx + 12, ly - 5);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText('Analytical E[overlap] ∝ p²', lx + 18, ly);

    // 🔵 Badge
    ctx.font = 'bold 11px "Inter", system-ui, sans-serif';
    ctx.fillStyle = '#74c0fc';
    ctx.textAlign = 'right';
    ctx.fillText('🔵 LIVE EMPIRICAL', width - pad.right - 5, pad.top + 12);
  }

  /** Dispatch event so other components can react to parameter changes */
  dispatchResultEvent() {
    if (this.currentResult) {
      window.dispatchEvent(new CustomEvent('surgery-result', {
        detail: {
          result: this.currentResult,
          params: { ...this.params }
        }
      }));
    }
  }

  destroy() {
    window.removeEventListener('resize', this._resizeHandler);
  }
}
