// synaptic_matrix_view.js
// Canvas visualizer for the live synaptic weight matrix S ∈ R^(N × D).
// Also provides the Memory Crime Scene inspector highlighting colliding neurons.

export class SynapticMatrixView {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.matrixData = null;    // Float64Array (N × D)
    this.N = 0;
    this.D = 0;
    this.writeHistory = [];
    this.highlightedCells = new Set();   // Cells to highlight for crime scene
    this.activeWriteIndices = [];        // Currently glowing write neurons
    this.crimeSceneMode = false;
    this.crimeOverlaps = [];
    this._resizeHandler = () => this._resize();
    window.addEventListener('resize', this._resizeHandler);
    this._resize();
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = (rect.height || 280) * dpr;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = (rect.height || 280) + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = rect.width;
    this.height = rect.height || 280;
    this.draw();
  }

  /**
   * Update the matrix data for rendering.
   * @param {Float64Array} S - Synaptic state matrix (N × D, row-major)
   * @param {number} N
   * @param {number} D
   */
  setMatrix(S, N, D) {
    this.matrixData = S;
    this.N = N;
    this.D = D;
    this.draw();
  }

  /**
   * Highlight a Hebbian write event (neurons that just fired).
   * @param {number[]} activeIndices - Row indices that were written to
   */
  flashWrite(activeIndices) {
    this.activeWriteIndices = activeIndices;
    this.draw();
    // Fade out after 600ms
    setTimeout(() => {
      this.activeWriteIndices = [];
      this.draw();
    }, 600);
  }

  /**
   * Activate Crime Scene mode: highlight overlapping neurons between query and stored keys.
   * @param {Array<{ label: string, overlapFraction: number, sharedIndices: number[] }>} overlaps
   * @param {number[]} queryIndices - Active indices of the query
   */
  showCrimeScene(overlaps, queryIndices) {
    this.crimeSceneMode = true;
    this.crimeOverlaps = overlaps;
    this.highlightedCells.clear();
    // Highlight all shared indices
    for (const overlap of overlaps) {
      for (const idx of overlap.sharedIndices) {
        this.highlightedCells.add(idx);
      }
    }
    // Also highlight query indices
    for (const idx of queryIndices) {
      this.highlightedCells.add(idx);
    }
    this.draw();
  }

  clearCrimeScene() {
    this.crimeSceneMode = false;
    this.crimeOverlaps = [];
    this.highlightedCells.clear();
    this.draw();
  }

  draw() {
    const { ctx, width, height, matrixData, N, D } = this;
    if (!matrixData || N === 0 || D === 0) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'rgba(15, 15, 25, 0.6)';
      ctx.beginPath();
      ctx.roundRect(0, 0, width, height, 12);
      ctx.fill();
      ctx.fillStyle = '#868e96';
      ctx.font = '14px "Inter", system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Synaptic state will appear here', width / 2, height / 2);
      return;
    }

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(15, 15, 25, 0.6)';
    ctx.beginPath();
    ctx.roundRect(0, 0, width, height, 12);
    ctx.fill();

    const pad = { top: 30, right: 15, bottom: 25, left: 50 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Sample rows/columns if matrix is too large for pixel-level rendering
    const maxRows = Math.min(N, 64);
    const maxCols = Math.min(D, 32);
    const rowStep = Math.max(1, Math.floor(N / maxRows));
    const colStep = Math.max(1, Math.floor(D / maxCols));
    const displayRows = Math.ceil(N / rowStep);
    const displayCols = Math.ceil(D / colStep);

    const cellW = plotW / displayCols;
    const cellH = plotH / displayRows;

    // Find max absolute value for normalization
    let maxVal = 0;
    for (let i = 0; i < matrixData.length; i++) {
      maxVal = Math.max(maxVal, Math.abs(matrixData[i]));
    }
    if (maxVal === 0) maxVal = 1;

    const activeWriteSet = new Set(this.activeWriteIndices);

    // Draw cells
    for (let ri = 0; ri < displayRows; ri++) {
      const row = ri * rowStep;
      if (row >= N) break;
      for (let ci = 0; ci < displayCols; ci++) {
        const col = ci * colStep;
        if (col >= D) break;

        const val = matrixData[row * D + col];
        const norm = val / maxVal;
        const x = pad.left + ci * cellW;
        const y = pad.top + ri * cellH;

        // Base color: blue for positive, red for negative, intensity by magnitude
        if (norm > 0.01) {
          const intensity = Math.min(1, norm);
          ctx.fillStyle = `rgba(81, 207, 102, ${intensity * 0.8})`;
        } else if (norm < -0.01) {
          const intensity = Math.min(1, -norm);
          ctx.fillStyle = `rgba(255, 107, 107, ${intensity * 0.8})`;
        } else {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
        }

        ctx.fillRect(x, y, cellW - 0.5, cellH - 0.5);

        // Active write glow
        if (activeWriteSet.has(row)) {
          ctx.fillStyle = 'rgba(255, 224, 102, 0.5)';
          ctx.fillRect(x, y, cellW - 0.5, cellH - 0.5);
        }

        // Crime scene highlight
        if (this.crimeSceneMode && this.highlightedCells.has(row)) {
          ctx.strokeStyle = 'rgba(255, 107, 107, 0.9)';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x, y, cellW - 0.5, cellH - 0.5);
        }
      }
    }

    // Labels
    ctx.font = '12px "Inter", system-ui, sans-serif';
    ctx.fillStyle = '#adb5bd';
    ctx.textAlign = 'center';
    ctx.fillText(`S ∈ ℝ^(${N}×${D})`, pad.left + plotW / 2, height - 5);

    ctx.save();
    ctx.translate(14, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Neurons (N)', 0, 0);
    ctx.restore();

    // Title
    ctx.font = 'bold 12px "Inter", system-ui, sans-serif';
    ctx.fillStyle = this.crimeSceneMode ? '#ff6b6b' : '#e9ecef';
    ctx.textAlign = 'left';
    ctx.fillText(
      this.crimeSceneMode ? '⚠ MEMORY CRIME SCENE' : '🧠 Synaptic State Matrix',
      pad.left, pad.top - 10
    );

    // Crime scene legend
    if (this.crimeSceneMode && this.crimeOverlaps.length > 0) {
      ctx.font = '11px "Inter", system-ui, sans-serif';
      ctx.fillStyle = '#ff6b6b';
      const legendX = width - pad.right - 200;
      let legendY = pad.top + 5;
      ctx.fillText('Interfering keys:', legendX, legendY);
      for (const overlap of this.crimeOverlaps.slice(0, 3)) {
        legendY += 16;
        const pct = (overlap.overlapFraction * 100).toFixed(0);
        ctx.fillText(`${overlap.label}: ${pct}% overlap`, legendX + 8, legendY);
      }
    }
  }

  destroy() {
    window.removeEventListener('resize', this._resizeHandler);
  }
}
