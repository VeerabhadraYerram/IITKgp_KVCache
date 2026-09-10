// app.js — Chapters 1 & 2 Controller
// Chapter 1: Typewriter hero + Dilemma widget
// Chapter 2: Live TypedArray Benchmark + 2D Runtime Canvas + 3D Attention Microscope

import { MicroTransformerEngine } from './engine/micro_transformer.js';
import { BenchmarkCanvas } from './visualizers/benchmark_canvas.js';
import { AttentionMicroscope3D } from './visualizers/benchmark_3d.js';

// ─── Chapter 1: Typewriter Animation ───────────────────────────
function initTypewriter() {
  const titleEl = document.getElementById('typewriter-title');
  const subEl = document.getElementById('typewriter-sub');
  const promptEl = document.getElementById('scroll-prompt');

  if (!titleEl || !subEl) return;

  const titleText = `What does an AI actually mean by "remembering"?`;
  const subText = `Does it keep a perfect recording of everything you said — or let the past quietly reshape it?`;

  let titleIdx = 0;
  let subIdx = 0;

  const renderCaret = () => '<span class="caret"></span>';

  function typeTitle() {
    if (titleIdx < titleText.length) {
      titleEl.innerHTML = titleText.slice(0, titleIdx + 1) + renderCaret();
      titleIdx++;
      setTimeout(typeTitle, 40);
    } else {
      titleEl.innerHTML = titleText;
      subEl.innerHTML = renderCaret();
      setTimeout(typeSub, 300);
    }
  }

  function typeSub() {
    if (subIdx < subText.length) {
      subEl.innerHTML = subText.slice(0, subIdx + 1) + renderCaret();
      subIdx++;
      setTimeout(typeSub, 30);
    } else {
      subEl.innerHTML = subText;
      if (promptEl) {
        promptEl.style.opacity = '1';
      }
    }
  }

  setTimeout(typeTitle, 400);
}

// ─── Chapter 1: Dilemma Widget State Handler ───────────────────
function initDilemmaWidget() {
  const choices = { legal: null, chat: null };

  const buttons = document.querySelectorAll('.choice-btn');
  const summaryBox = document.getElementById('summary-box');
  const summaryLine = document.getElementById('summary-line-text');
  const summaryExplanation = document.getElementById('summary-explanation-text');
  const cardVerbatim = document.getElementById('card-verbatim');
  const cardCompressed = document.getElementById('card-compressed');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const scenario = btn.dataset.scenario;
      const choice = btn.dataset.choice;

      const siblingBtns = document.querySelectorAll(`.choice-btn[data-scenario="${scenario}"]`);
      siblingBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      choices[scenario] = choice;
      updateSummary();
    });
  });

  function updateSummary() {
    if (!choices.legal && !choices.chat) return;

    summaryBox.style.display = 'block';
    const legalLabel = choices.legal === 'verbatim' ? 'Verbatim Store' : choices.legal === 'compressed' ? 'Compressed State' : 'Not selected';
    const chatLabel = choices.chat === 'verbatim' ? 'Verbatim Store' : choices.chat === 'compressed' ? 'Compressed State' : 'Not selected';

    summaryLine.textContent = `Your Deployment Choices: Legal Audit → ${legalLabel} | Real-Time Chat → ${chatLabel}`;

    let explanation = '';
    if (choices.legal === 'verbatim' && choices.chat === 'compressed') {
      explanation = `You chose exact literal recall for high-stakes legal compliance (where a lost number is a critical failure), but opted for semantic compression in real-time chat (where general gist is enough and memory overhead must stay minimal). You intuitively navigated the exact trade-off at the heart of AI memory.`;
    } else if (choices.legal === 'verbatim' && choices.chat === 'verbatim') {
      explanation = `You prioritized 100% literal accuracy across all workloads, accepting that memory consumption will grow continuously over long conversations.`;
    } else if (choices.legal === 'compressed' && choices.chat === 'compressed') {
      explanation = `You prioritized constant, bounded memory across all workloads, accepting that exact numbers like "$1,450,000" might be recalled only approximately as "around $1.45M".`;
    } else if (choices.legal === 'compressed' && choices.chat === 'verbatim') {
      explanation = `You accepted approximate recall in legal auditing and demanded exact retention in casual chat. This shows how differing task demands force different compromises.`;
    } else {
      explanation = `Make a selection for both scenarios above to see how task requirements dictate memory trade-offs.`;
    }

    summaryExplanation.textContent = explanation;

    if (choices.legal === 'verbatim' || choices.chat === 'verbatim') {
      cardVerbatim.style.borderColor = 'rgba(0, 0, 0, 0.6)';
    } else {
      cardVerbatim.style.borderColor = 'var(--border-glass)';
    }

    if (choices.legal === 'compressed' || choices.chat === 'compressed') {
      cardCompressed.style.borderColor = 'rgba(0, 0, 0, 0.6)';
    } else {
      cardCompressed.style.borderColor = 'var(--border-glass)';
    }
  }
}

// ─── Chapter 2: Live Benchmark & 3D Microscope ─────────────────
function initChapter2() {
  const engine = new MicroTransformerEngine(32);

  // DOM Elements
  const seqLenSlider = document.getElementById('benchmark-seq-len');
  const seqLenVal = document.getElementById('seq-len-val');
  const btnRunBenchmark = document.getElementById('btn-run-benchmark');

  // Metrics
  const metricNoCacheMs = document.getElementById('metric-nocache-ms');
  const metricCachedMs = document.getElementById('metric-cached-ms');
  const metricSpeedupX = document.getElementById('metric-speedup-x');

  // Analytical FLOP fields
  const flopNaiveProj = document.getElementById('flop-naive-proj');
  const flopCachedProj = document.getElementById('flop-cached-proj');
  const flopAttn = document.getElementById('flop-attn');
  const flopNaiveTotal = document.getElementById('flop-naive-total');
  const flopCachedTotal = document.getElementById('flop-cached-total');

  // 2D Canvas
  const canvasEl = document.getElementById('benchmark-canvas');
  let benchmarkCanvas = null;
  if (canvasEl) {
    benchmarkCanvas = new BenchmarkCanvas(canvasEl);
  }

  // 3D Microscope
  const microscopeContainer = document.getElementById('microscope-3d-container');
  const sliderTimestep = document.getElementById('slider-timestep');
  const dispTimestep = document.getElementById('disp-timestep');
  const dispTimestepMax = document.getElementById('disp-timestep-max');
  const btnModeCached = document.getElementById('btn-mode-cached');
  const btnModeNoCache = document.getElementById('btn-mode-nocache');
  const btnResetCam = document.getElementById('btn-reset-cam');

  // Badges
  const pillCachedTokens = document.getElementById('pill-cached-tokens');
  const pillStoredVectors = document.getElementById('pill-stored-vectors');
  const pillProjBreakdown = document.getElementById('pill-proj-breakdown');

  // Inspector Elements
  const inspectTokenTag = document.getElementById('inspect-token-tag');
  const inspectDot = document.getElementById('inspect-dot');
  const inspectLogit = document.getElementById('inspect-logit');
  const inspectWeight = document.getElementById('inspect-weight');
  const inspectValContrib = document.getElementById('inspect-val-contrib');
  const inspectOutputVec = document.getElementById('inspect-output-vec');

  let currentMode = 'kv_cache';
  let latestBenchmarkResult = null;

  // Inspector callback when token is selected in 3D
  const onTokenSelected = (info) => {
    if (!info) return;
    inspectTokenTag.textContent = `Token k_${info.tokenIndex}`;
    inspectDot.textContent = `${info.rawDotProduct >= 0 ? '+' : ''}${info.rawDotProduct.toFixed(3)}`;
    inspectLogit.textContent = `${info.scaledLogit >= 0 ? '+' : ''}${info.scaledLogit.toFixed(3)}`;
    const pct = (info.attentionWeight * 100).toFixed(2);
    inspectWeight.textContent = `${pct}%  (α = ${info.attentionWeight.toFixed(4)})`;
    inspectValContrib.textContent = `Weighted by α_${info.tokenIndex} (${pct}%) in ℝ³²`;

    const po = info.projectedOutput;
    inspectOutputVec.textContent = `Projected 3D: [${po[0].toFixed(1)}, ${po[1].toFixed(1)}, ${po[2].toFixed(1)}]`;
  };

  let microscope3D = null;
  if (microscopeContainer) {
    microscope3D = new AttentionMicroscope3D(microscopeContainer, onTokenSelected);
  }

  // Format integer with commas
  const fmt = (num) => Number(num).toLocaleString('en-US');

  function updateStatusBadges(t) {
    if (pillCachedTokens) pillCachedTokens.textContent = t;
    if (pillStoredVectors) pillStoredVectors.textContent = 2 * t;

    if (pillProjBreakdown) {
      if (currentMode === 'kv_cache') {
        pillProjBreakdown.textContent = `Historical K/V reused: ${t - 1} · New: 1`;
      } else {
        pillProjBreakdown.textContent = `Historical K/V recomputed: ${t}`;
      }
    }
  }

  function runBenchmarkExecution() {
    const N = parseInt(seqLenSlider.value, 10) || 80;

    btnRunBenchmark.textContent = '⏳ Computing matrix operations…';
    btnRunBenchmark.disabled = true;

    // Isolate numerical benchmark in setTimeout so browser UI updates
    setTimeout(() => {
      // 1. Run live TypedArray benchmark (isolated from rendering)
      const res = engine.runBenchmark(N, 20);
      latestBenchmarkResult = res;

      // 2. Update Wall-Clock Measured Metrics
      metricNoCacheMs.textContent = `${res.measured.noCacheTotalMs.toFixed(2)} ms`;
      metricCachedMs.textContent = `${res.measured.cachedTotalMs.toFixed(2)} ms`;
      metricSpeedupX.textContent = `${res.measured.measuredSpeedup.toFixed(1)}×`;

      // 3. Update Analytical FLOP Table
      flopNaiveProj.textContent = fmt(res.analytical.naiveProjFlops);
      flopCachedProj.textContent = fmt(res.analytical.cachedProjFlops);
      flopAttn.textContent = fmt(res.analytical.attnFlops);
      flopNaiveTotal.textContent = fmt(res.analytical.naiveTotalFlops);
      flopCachedTotal.textContent = fmt(res.analytical.cachedTotalFlops);

      // 4. Update 2D Canvas Chart
      if (benchmarkCanvas) {
        benchmarkCanvas.setData(res.measured);
      }

      // 5. Update 3D Attention Microscope
      if (microscope3D) {
        microscope3D.setSnapshots(res.snapshots);
      }

      // Configure Timestep slider
      sliderTimestep.max = N;
      sliderTimestep.value = N;
      dispTimestep.textContent = N;
      dispTimestepMax.textContent = N;

      updateStatusBadges(N);

      btnRunBenchmark.textContent = '▶ Run Live Browser Benchmark';
      btnRunBenchmark.disabled = false;
    }, 40);
  }

  // Sequence Length Slider Listener
  seqLenSlider.addEventListener('input', () => {
    seqLenVal.textContent = seqLenSlider.value;
  });

  // Run Benchmark Button Listener
  btnRunBenchmark.addEventListener('click', runBenchmarkExecution);

  // Timestep Slider Listener
  sliderTimestep.addEventListener('input', () => {
    const t = parseInt(sliderTimestep.value, 10);
    dispTimestep.textContent = t;
    if (microscope3D) {
      microscope3D.setStep(t);
    }
    updateStatusBadges(t);
  });

  // Mode Toggle Listeners
  btnModeCached.addEventListener('click', () => {
    currentMode = 'kv_cache';
    btnModeCached.classList.add('active');
    btnModeNoCache.classList.remove('active');
    if (microscope3D) microscope3D.setMode('kv_cache');
    const t = parseInt(sliderTimestep.value, 10);
    updateStatusBadges(t);
  });

  btnModeNoCache.addEventListener('click', () => {
    currentMode = 'no_cache';
    btnModeNoCache.classList.add('active');
    btnModeCached.classList.remove('active');
    if (microscope3D) microscope3D.setMode('no_cache');
    const t = parseInt(sliderTimestep.value, 10);
    updateStatusBadges(t);
  });

  // Reset Camera Button Listener
  if (btnResetCam) {
    btnResetCam.addEventListener('click', () => {
      if (microscope3D) microscope3D.resetCamera();
    });
  }

  // Initial Run on Page Load
  runBenchmarkExecution();
}

// ─── Main Initialization ───────────────────────────────────────
function init() {
  initTypewriter();
  initDilemmaWidget();
  initChapter2();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
