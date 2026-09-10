// app.js — Main controller wiring all engines and visualizers together.
// DataForge 2026: "When Attention Becomes Memory"

import { MicroBDH } from './engine/micro_bdh.js';
import { MicroTransformer } from './engine/micro_transformer.js';
import { compareMemoryScaling, runRecallExperiment, generateAssociations } from './engine/memory_experiment.js';
import { MemoryWallCanvas } from './visualizers/memory_wall_canvas.js';
import { SynapticMatrixView } from './visualizers/synaptic_matrix_view.js';
import { SurgeryHUD } from './visualizers/surgery_hud.js';

// ─── State ──────────────────────────────────────────────────────
const D = 32;
const N = 128;
let microBDH = new MicroBDH(D, N, 0.05);
let tokenCounter = 0;

// ─── Initialize KaTeX auto-render ──────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Wait for KaTeX to load
  const initKaTeX = () => {
    if (typeof renderMathInElement === 'function') {
      renderMathInElement(document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true },
        ],
        throwOnError: false,
      });
    } else {
      setTimeout(initKaTeX, 100);
    }
  };
  initKaTeX();
});

// ─── ACT I: Memory Wall Canvas ─────────────────────────────────
let memoryWallViz;
function initMemoryWall() {
  memoryWallViz = new MemoryWallCanvas('memory-wall-canvas');
  const data = compareMemoryScaling({ D, N, maxT: 200, steps: 30 });
  memoryWallViz.setData(data);

  // Animate when scrolled into view
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        memoryWallViz.animate(2000);
        observer.disconnect();
      }
    }
  }, { threshold: 0.3 });
  observer.observe(document.getElementById('memory-wall-container'));
}

// ─── ACT I: Bracket Drag Interaction ───────────────────────────
function initBracketDrag() {
  const eqBefore = document.getElementById('eq-before');
  const eqAfter = document.getElementById('eq-after');
  const stateLabel = document.getElementById('bracket-state-label');
  const instruction = document.getElementById('bracket-instruction');
  let reassociated = false;

  // Click interaction (simpler and more reliable than drag)
  const parenOpen = document.getElementById('paren-open');
  const parenClose = document.getElementById('paren-close');

  const toggle = () => {
    reassociated = !reassociated;
    if (reassociated) {
      eqBefore.style.display = 'none';
      eqAfter.style.display = 'inline';
      stateLabel.className = 'bracket-state-label bracket-state-after';
      stateLabel.textContent = 'Accumulated into fixed state S_t · Constant memory w.r.t. T';
      instruction.textContent = '✅ History is now accumulated into a single state matrix S — click again to revert';
      // Re-render KaTeX for the new equation
      if (typeof renderMathInElement === 'function') {
        renderMathInElement(document.getElementById('bracket-drag'), {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\(', right: '\\)', display: false },
          ],
          throwOnError: false,
        });
      }
    } else {
      eqBefore.style.display = 'inline';
      eqAfter.style.display = 'none';
      stateLabel.className = 'bracket-state-label bracket-state-before';
      stateLabel.textContent = 'Token-by-token pairwise comparison · O(T · d) memory';
      instruction.textContent = '👆 Click the yellow brackets to re-associate the multiplication';
      if (typeof renderMathInElement === 'function') {
        renderMathInElement(document.getElementById('bracket-drag'), {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\(', right: '\\)', display: false },
          ],
          throwOnError: false,
        });
      }
    }
  };

  if (parenOpen) parenOpen.addEventListener('click', toggle);
  if (parenClose) parenClose.addEventListener('click', toggle);
}

// ─── ACT II: Synaptic Matrix ──────────────────────────────────
let synapticViz;
function initSynapticMatrix() {
  synapticViz = new SynapticMatrixView('synapse-canvas');

  // Stream token button
  document.getElementById('btn-stream-token').addEventListener('click', () => {
    tokenCounter++;
    const key = MicroBDH.randomVector(D, tokenCounter * 13 + 7);
    const value = MicroBDH.randomVector(D, tokenCounter * 17 + 3);
    const labels = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta'];
    const label = labels[(tokenCounter - 1) % labels.length];

    const { activeIndices } = microBDH.store(key, value, label);
    synapticViz.setMatrix(microBDH.S, N, D);
    synapticViz.flashWrite(activeIndices);
  });

  // Reset button
  document.getElementById('btn-reset-synapse').addEventListener('click', () => {
    microBDH.reset();
    tokenCounter = 0;
    synapticViz.setMatrix(microBDH.S, N, D);
    synapticViz.clearCrimeScene();
  });

  // Initial empty state
  synapticViz.setMatrix(microBDH.S, N, D);
}

// ─── ACT III: Break It! Crime Scene ────────────────────────────
let crimeSceneBDH = null;
let crimeAssociations = null;

function initCrimeScene() {
  const resultEl = document.getElementById('crime-scene-result');
  const inspectorEl = document.getElementById('crime-scene-inspector');
  const expectedEl = document.getElementById('truth-expected');
  const retrievedEl = document.getElementById('truth-retrieved');
  const overlapListEl = document.getElementById('crime-overlap-list');

  // Break Memory button — store 8 dense associations
  document.getElementById('btn-break-memory').addEventListener('click', () => {
    crimeSceneBDH = new MicroBDH(D, N, 1.0); // 100% dense = maximum overlap
    crimeAssociations = generateAssociations(8, D, 42, 0.0);

    for (const assoc of crimeAssociations) {
      crimeSceneBDH.store(assoc.key, assoc.value, assoc.label);
    }

    // Update synaptic view to show the crime scene state
    synapticViz.setMatrix(crimeSceneBDH.S, N, D);
    synapticViz.clearCrimeScene();

    resultEl.style.display = 'none';
    inspectorEl.style.display = 'none';
  });

  // Query button — try to retrieve Alpha
  document.getElementById('btn-query-memory').addEventListener('click', () => {
    if (!crimeSceneBDH || !crimeAssociations) return;

    const alpha = crimeAssociations[0]; // Alpha
    const { output, queryActiveIndices } = crimeSceneBDH.retrieve(alpha.key);

    // Compute retrieved scalar (sum of output)
    const retrievedScalar = output.reduce((s, v) => s + v, 0);
    const expectedScalar = alpha.groundTruth;

    // Cosine similarity
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < D; i++) {
      dot += output[i] * alpha.value[i];
      normA += output[i] * output[i];
      normB += alpha.value[i] * alpha.value[i];
    }
    const cosSim = dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
    const isCorrect = cosSim > 0.8;

    expectedEl.textContent = expectedScalar;
    retrievedEl.textContent = Math.round(retrievedScalar * 10) / 10;
    retrievedEl.className = `truth-box-value ${isCorrect ? 'correct' : 'incorrect'}`;

    resultEl.style.display = 'grid';
  });

  // Crime scene button — why did it forget?
  document.getElementById('btn-crime-scene').addEventListener('click', () => {
    if (!crimeSceneBDH || !crimeAssociations) return;

    const alpha = crimeAssociations[0];
    const { queryActiveIndices } = crimeSceneBDH.retrieve(alpha.key);
    const crimeScene = crimeSceneBDH.analyzeCrimeScene(queryActiveIndices);

    // Filter out self, sort by overlap
    const overlaps = crimeScene
      .filter(cs => cs.label !== alpha.label)
      .sort((a, b) => b.overlapFraction - a.overlapFraction)
      .slice(0, 3);

    overlapListEl.innerHTML = '';
    for (const overlap of overlaps) {
      const pct = (overlap.overlapFraction * 100).toFixed(0);
      const li = document.createElement('li');
      li.className = 'crime-overlap-item';
      li.innerHTML = `
        <span style="min-width:70px;font-weight:600;color:var(--text-primary)">${overlap.label}</span>
        <div class="crime-overlap-bar">
          <div class="crime-overlap-fill" style="width:${pct}%"></div>
        </div>
        <span class="text-mono" style="min-width:70px;text-align:right;color:var(--red)">${pct}% overlap</span>
      `;
      overlapListEl.appendChild(li);
    }

    // Show in synaptic matrix
    synapticViz.showCrimeScene(overlaps, queryActiveIndices);

    inspectorEl.style.display = 'block';
  });
}

// ─── ACT IV: Surgery HUD ──────────────────────────────────────
let surgeryHUD;
function initSurgeryHUD() {
  surgeryHUD = new SurgeryHUD('surgery-canvas', 'surgery-controls');
}

// ─── ACT V: Effort Tier Interaction ────────────────────────────
function initEffortTiers() {
  const tiers = document.querySelectorAll('.effort-tier');
  tiers.forEach(tier => {
    tier.addEventListener('click', () => {
      tiers.forEach(t => t.classList.remove('active'));
      tier.classList.add('active');
    });
  });
}

// ─── Initialize Everything ─────────────────────────────────────
function init() {
  initMemoryWall();
  initBracketDrag();
  initSynapticMatrix();
  initCrimeScene();
  initSurgeryHUD();
  initEffortTiers();
}

// Start when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
