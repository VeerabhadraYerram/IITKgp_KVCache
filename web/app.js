// app.js — Chapters 1, 2, 3, 4, & 5 Controller
// Chapter 1: Typewriter hero + Dilemma widget
// Chapter 2: Live TypedArray Benchmark + 2D Runtime Canvas + 3D Attention Microscope
// Chapter 3: Architecture-Aware VRAM Sandbox + 3D Structural KV Cache Tensor Visualizer
// Chapter 4: Prefill vs. Decode Analytical Roofline + Serving Concurrency Sandbox
// Chapter 5: Alternate Memory Approaches Side-by-Side Comparison

import { MicroTransformerEngine } from './engine/micro_transformer.js';
import { BenchmarkCanvas } from './visualizers/benchmark_canvas.js';
import { AttentionMicroscope3D } from './visualizers/benchmark_3d.js';
import { calculateKVCacheMemory, MODEL_PRESETS, GPU_SPECS, PRECISIONS } from './engine/gpu_calculator.js';
import { KV3DVisualizer } from './visualizers/kv_3d_visualizer.js';
import { BandwidthCanvas } from './visualizers/bandwidth_canvas.js';
import { MemoryComparisonCanvas, APPROACHES } from './visualizers/memory_comparison_canvas.js';
import { SurgeryHUD } from './visualizers/surgery_hud.js';
import { runRecallExperiment } from './engine/memory_experiment.js';

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

// ─── Chapter 3: VRAM Sandbox & 3D Structural Visualizer ───────
function initChapter3() {
  const modelPresetSelect = document.getElementById('select-model-preset');
  const presetHelpText = document.getElementById('preset-help-text');
  const contextSlider = document.getElementById('slider-context-len');
  const badgeContextLen = document.getElementById('badge-context-len');
  const batchSlider = document.getElementById('slider-batch-size');
  const badgeBatchSize = document.getElementById('badge-batch-size');
  const precButtons = document.querySelectorAll('.prec-btn');
  const gpuSelect = document.getElementById('select-gpu-target');
  const btnResetCam = document.getElementById('btn-kv3d-reset');

  // Breakdown DOM
  const allocTotalText = document.getElementById('alloc-total-text');
  const vramBarWeights = document.getElementById('vram-bar-weights');
  const vramBarKv = document.getElementById('vram-bar-kv');
  const vramBarOverflow = document.getElementById('vram-bar-overflow');
  const capacityWarningCard = document.getElementById('capacity-warning-card');
  const aggregateNoteCard = document.getElementById('aggregate-note-card');

  const valWeightsGiB = document.getElementById('val-weights-gib');
  const valWeightsSub = document.getElementById('val-weights-sub');
  const valKvGiB = document.getElementById('val-kv-gib');
  const valKvSub = document.getElementById('val-kv-sub');
  const valTotalGiB = document.getElementById('val-total-gib');
  const valTotalSub = document.getElementById('val-total-sub');
  const valFormulaText = document.getElementById('val-formula-text');

  // 3D Visualizer
  const kv3dContainer = document.getElementById('kv-3d-container');
  let kv3dVisualizer = null;
  if (kv3dContainer) {
    kv3dVisualizer = new KV3DVisualizer(kv3dContainer);
  }

  const contextLengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
  let activePrecision = 'fp16';

  function updateSandbox() {
    if (!modelPresetSelect || !contextSlider || !batchSlider || !gpuSelect) return;

    const presetKey = modelPresetSelect.value;
    const seqLen = contextLengths[parseInt(contextSlider.value, 10)] || 8192;
    const batchSize = parseInt(batchSlider.value, 10) || 1;
    const precisionKey = activePrecision;
    const gpuKey = gpuSelect.value;

    // Run exact arithmetic calculation
    const res = calculateKVCacheMemory({
      presetKey,
      seqLen,
      batchSize,
      precisionKey,
      gpuKey
    });

    // Update help text
    if (presetHelpText) {
      presetHelpText.textContent = `${res.model.archName} · ${res.weightsGiB.toFixed(2)} GiB weights (${res.precision.name})`;
    }

    // Update Badges
    if (badgeContextLen) badgeContextLen.textContent = `${seqLen.toLocaleString('en-US')} tokens`;
    if (badgeBatchSize) badgeBatchSize.textContent = batchSize;

    // Update Breakdown numbers (in GiB)
    if (valWeightsGiB) valWeightsGiB.textContent = `${res.weightsGiB.toFixed(2)} GiB`;
    if (valWeightsSub) valWeightsSub.textContent = `${res.precision.name} (${res.precision.bytes} B/elem)`;

    if (valKvGiB) valKvGiB.textContent = `${res.totalKVGiB.toFixed(2)} GiB`;
    if (valKvSub) valKvSub.textContent = `${res.bytesPerTokenPerLayer} B/token/layer · ${res.totalKVBytes.toLocaleString('en-US')} B`;

    if (valTotalGiB) valTotalGiB.textContent = `${res.totalRequiredGiB.toFixed(2)} GiB`;
    if (valTotalSub) valTotalSub.textContent = `${res.totalPercentOfGpu.toFixed(1)}% of ${res.gpuCeilingGiB} GiB GPU ceiling`;

    if (allocTotalText) allocTotalText.textContent = `${res.totalRequiredGiB.toFixed(2)} GiB / ${res.gpuCeilingGiB} GiB (${res.gpu.name})`;
    if (valFormulaText) valFormulaText.textContent = `${res.formulaUsed} → ${res.formulaMath}`;

    // Update VRAM Bar
    if (vramBarWeights && vramBarKv) {
      const weightsPct = Math.min(100, res.weightsPercentOfGpu);
      const kvPct = Math.min(Math.max(0, 100 - weightsPct), res.kvPercentOfGpu);

      vramBarWeights.style.width = `${weightsPct}%`;
      vramBarKv.style.width = `${kvPct}%`;
    }

    if (capacityWarningCard && vramBarOverflow) {
      if (res.isExceeded) {
        const overflowPct = Math.min(50, Math.max(5, res.totalPercentOfGpu - 100));
        vramBarOverflow.style.display = 'flex';
        vramBarOverflow.style.width = `${overflowPct}%`;
        capacityWarningCard.style.display = 'block';
      } else {
        vramBarOverflow.style.display = 'none';
        capacityWarningCard.style.display = 'none';
      }
    }

    if (aggregateNoteCard) {
      if (res.isAggregate) {
        aggregateNoteCard.style.display = 'block';
      } else {
        aggregateNoteCard.style.display = 'none';
      }
    }

    // Update 3D Visualizer
    if (kv3dVisualizer) {
      kv3dVisualizer.updateState(res);
    }
  }

  // Listeners
  if (modelPresetSelect) modelPresetSelect.addEventListener('change', updateSandbox);
  if (contextSlider) contextSlider.addEventListener('input', updateSandbox);
  if (batchSlider) batchSlider.addEventListener('input', updateSandbox);
  if (gpuSelect) gpuSelect.addEventListener('change', updateSandbox);

  precButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      precButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activePrecision = btn.dataset.prec;
      updateSandbox();
    });
  });

  if (btnResetCam) {
    btnResetCam.addEventListener('click', () => {
      if (kv3dVisualizer) kv3dVisualizer.resetCamera();
    });
  }

  // Initial update
  updateSandbox();
}

// ─── Chapter 4: Prefill vs. Decode & Serving Sandbox ────────────
function initChapter4() {
  const canvasEl = document.getElementById('bandwidth-canvas');
  if (!canvasEl) return;

  const bandwidthCanvas = new BandwidthCanvas('bandwidth-canvas');

  const batchSlider = document.getElementById('slider-serving-batch');
  const contextSlider = document.getElementById('slider-serving-context');
  const batchBadge = document.getElementById('badge-serving-batch');
  const contextBadge = document.getElementById('badge-serving-context');

  const valMemTraffic = document.getElementById('val-serving-mem-traffic');
  const subMemTraffic = document.getElementById('sub-serving-mem-traffic');
  const valAI = document.getElementById('val-serving-ai');
  const valPerf = document.getElementById('val-serving-perf');
  const subPerf = document.getElementById('sub-serving-perf');
  const valLatency = document.getElementById('val-serving-latency');
  const valUserRate = document.getElementById('val-serving-user-rate');
  const valAggRate = document.getElementById('val-serving-agg-rate');
  const subAggRate = document.getElementById('sub-serving-agg-rate');

  function updateServingSandbox() {
    const B = parseInt(batchSlider ? batchSlider.value : 1, 10);
    const T = parseInt(contextSlider ? contextSlider.value : 1024, 10);

    if (batchBadge) batchBadge.textContent = `${B} request${B > 1 ? 's' : ''}`;
    if (contextBadge) contextBadge.textContent = `${T.toLocaleString()} tokens`;

    // Update Roofline canvas operating state
    bandwidthCanvas.updateState(B, T, 1024);

    // Compute dynamic metrics
    const decode = bandwidthCanvas.getDecodeMetrics();
    const memBytes = decode.bytes;
    const memGB = memBytes / 1e9; // Decimal GB consistent with 2039 GB/s

    // Time per step
    const gpuBandwidthBytesSec = 2039 * 1e9; // 2.039e12 B/s
    const gpuPeakFlopsSec = 312 * 1e12; // 312e12 FLOP/s
    const tMem = memBytes / gpuBandwidthBytesSec;
    const tCompute = decode.flops / gpuPeakFlopsSec;
    const tStepSec = Math.max(tMem, tCompute);
    const tStepMs = tStepSec * 1000;

    // Token rates
    const userRate = 1 / tStepSec;
    const aggRate = B / tStepSec;

    // Asymptotic saturation limit: B_mem / (k_token * T)
    const kToken = 2 * 32 * 8 * 128 * 2; // 131,072 bytes
    const maxAggRate = gpuBandwidthBytesSec / (kToken * T);

    // Update DOM readouts
    if (valMemTraffic) valMemTraffic.textContent = `${memGB.toFixed(2)} GB`;
    if (subMemTraffic) {
      const kvGB = (B * kToken * T) / 1e9;
      subMemTraffic.textContent = `Weights: 16.06 GB · KV: ${kvGB.toFixed(2)} GB`;
    }

    if (valAI) valAI.textContent = `${decode.ai.toFixed(2)} FLOP/B`;
    if (valPerf) valPerf.textContent = `${decode.perfTFlops.toFixed(2)} TFLOPS`;
    if (subPerf) {
      const pct = (decode.perfTFlops / 312) * 100;
      subPerf.textContent = `${pct.toFixed(1)}% of 312 TFLOPS dense peak`;
    }

    if (valLatency) valLatency.textContent = `${tStepMs.toFixed(1)} ms`;
    if (valUserRate) valUserRate.textContent = `${userRate.toFixed(1)} tok/s`;
    if (valAggRate) valAggRate.textContent = `${Math.round(aggRate).toLocaleString()} tok/s`;
    if (subAggRate) {
      subAggRate.textContent = `Bandwidth limit ceiling: ~${Math.round(maxAggRate).toLocaleString()} tok/s`;
    }
  }

  if (batchSlider) batchSlider.addEventListener('input', updateServingSandbox);
  if (contextSlider) contextSlider.addEventListener('input', updateServingSandbox);

  updateServingSandbox();
}

// ─── Chapter 5: Alternate Memory Approaches Controller ──────────
function initChapter5() {
  const canvasEl = document.getElementById('memory-comparison-canvas');
  if (!canvasEl) return;

  const compCanvas = new MemoryComparisonCanvas('memory-comparison-canvas');

  let activeIdA = 'standard_kv';
  let activeIdB = 'mamba_ssm';

  const btnScaleLog = document.getElementById('btn-scale-log');
  const btnScaleLinear = document.getElementById('btn-scale-linear');

  const scrubberTVal = document.getElementById('scrubber-t-val');
  const scrubberValA = document.getElementById('scrubber-val-a');
  const scrubberValB = document.getElementById('scrubber-val-b');

  // Table element bindings
  const tableHeadA = document.getElementById('table-head-a');
  const tableHeadB = document.getElementById('table-head-b');
  const tdCategoryA = document.getElementById('td-category-a');
  const tdCategoryB = document.getElementById('td-category-b');
  const tdComplexityA = document.getElementById('td-complexity-a');
  const tdComplexityB = document.getElementById('td-complexity-b');
  const tdMem128A = document.getElementById('td-mem128-a');
  const tdMem128B = document.getElementById('td-mem128-b');
  const tdAddressA = document.getElementById('td-address-a');
  const tdAddressB = document.getElementById('td-address-b');
  const tdInfoA = document.getElementById('td-info-a');
  const tdInfoB = document.getElementById('td-info-b');
  const tdLimitA = document.getElementById('td-limit-a');
  const tdLimitB = document.getElementById('td-limit-b');
  const tdCiteA = document.getElementById('td-cite-a');
  const tdCiteB = document.getElementById('td-cite-b');

  function formatMemoryExact(bytes) {
    if (bytes >= 1e9) {
      const gbDec = (bytes / 1e9).toFixed(2);
      const gibBin = (bytes / (1024 * 1024 * 1024)).toFixed(2);
      return `${gbDec} GB decimal (${gibBin} GiB binary)`;
    } else {
      const mbDec = (bytes / 1e6).toFixed(2);
      const mibBin = (bytes / (1024 * 1024)).toFixed(2);
      return `${mbDec} MB decimal (${mibBin} MiB binary)`;
    }
  }

  function updateScrubberDisplay(t) {
    if (scrubberTVal) scrubberTVal.textContent = `${t.toLocaleString()} tokens`;
    const appA = APPROACHES[activeIdA];
    const appB = APPROACHES[activeIdB];
    if (appA && scrubberValA) {
      const bytesA = appA.normalizedBytes(t);
      scrubberValA.textContent = `A (${appA.shortName}): ${formatMemoryExact(bytesA)}`;
    }
    if (appB && scrubberValB) {
      const bytesB = appB.normalizedBytes(t);
      scrubberValB.textContent = `B (${appB.shortName}): ${formatMemoryExact(bytesB)}`;
    }
  }

  function updateComparisonView() {
    const appA = APPROACHES[activeIdA];
    const appB = APPROACHES[activeIdB];
    if (!appA || !appB) return;

    // Update canvas
    compCanvas.setApproaches(activeIdA, activeIdB);

    // Update table headers
    if (tableHeadA) tableHeadA.textContent = `Approach A: ${appA.name}`;
    if (tableHeadB) tableHeadB.textContent = `Approach B: ${appB.name}`;

    // Update table rows
    if (tdCategoryA) tdCategoryA.textContent = appA.stateCategory;
    if (tdCategoryB) tdCategoryB.textContent = appB.stateCategory;

    if (tdComplexityA) tdComplexityA.textContent = appA.complexityStr;
    if (tdComplexityB) tdComplexityB.textContent = appB.complexityStr;

    if (tdMem128A) tdMem128A.textContent = formatMemoryExact(appA.normalizedBytes(131072));
    if (tdMem128B) tdMem128B.textContent = formatMemoryExact(appB.normalizedBytes(131072));

    if (tdAddressA) tdAddressA.textContent = appA.tokenAddressable;
    if (tdAddressB) tdAddressB.textContent = appB.tokenAddressable;

    if (tdInfoA) tdInfoA.textContent = appA.informationRetained;
    if (tdInfoB) tdInfoB.textContent = appB.informationRetained;

    if (tdLimitA) tdLimitA.textContent = appA.limitations;
    if (tdLimitB) tdLimitB.textContent = appB.limitations;

    if (tdCiteA) tdCiteA.textContent = appA.citation;
    if (tdCiteB) tdCiteB.textContent = appB.citation;

    // Update cards visual state
    document.querySelectorAll('.approach-card').forEach(card => {
      const id = card.dataset.approach;
      card.classList.toggle('selected-a', id === activeIdA);
      card.classList.toggle('selected-b', id === activeIdB);
    });

    document.querySelectorAll('.approach-select-btn').forEach(btn => {
      const id = btn.dataset.id;
      const action = btn.dataset.action;
      if (action === 'set-a') {
        btn.classList.toggle('active-a', id === activeIdA);
      } else if (action === 'set-b') {
        btn.classList.toggle('active-b', id === activeIdB);
      }
    });

    updateScrubberDisplay(compCanvas.activeT);
  }

  // Card select button event listeners
  document.querySelectorAll('.approach-select-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const action = btn.dataset.action;
      if (action === 'set-a') {
        activeIdA = id;
      } else if (action === 'set-b') {
        activeIdB = id;
      }
      updateComparisonView();
    });
  });

  // Scale toggle event listeners
  if (btnScaleLog && btnScaleLinear) {
    btnScaleLog.addEventListener('click', () => {
      btnScaleLog.classList.add('active');
      btnScaleLinear.classList.remove('active');
      compCanvas.setScaleType('log');
    });

    btnScaleLinear.addEventListener('click', () => {
      btnScaleLinear.classList.add('active');
      btnScaleLog.classList.remove('active');
      compCanvas.setScaleType('linear');
    });
  }

  // Scrubber hover callback
  compCanvas.onHoverT = (t) => {
    updateScrubberDisplay(t);
  };

  updateComparisonView();
}

// ─── Chapter 6: What Changes Stepper ───────────────────────────
function initChapter6() {
  const contentEl = document.getElementById('stepper-content');
  const indicatorEl = document.getElementById('stepper-step-indicator');
  const btnPrev = document.getElementById('btn-step-prev');
  const btnNext = document.getElementById('btn-step-next');
  const stepBtns = document.querySelectorAll('.step-pill-btn');

  if (!contentEl) return;

  let currentStep = 0;

  const STEPS = [
    {
      tag: 'Step 01 · Transformer Architecture',
      badgeClass: 'badge-primary',
      badgeText: '[PRIMARY SOURCE]',
      badgeSub: 'Architecture',
      heading: '1. Transformer: Explicit Token-Addressable History',
      body: 'In standard Transformer attention, every historical token generates an independent Key and Value representation stored directly in GPU High-Bandwidth Memory (HBM). When attending, the model directly evaluates pairwise dot-product similarities against every past position.',
      diagramHtml: `
        <div><strong>Input Sequence:</strong> [x₁, x₂, ..., x_T]</div>
        <div><strong>VRAM Storage:</strong> &nbsp; [k₁, v₁], [k₂, v₂], ..., [k_T, v_T] &nbsp;&rarr;&nbsp; <em>T distinct token memory vectors in VRAM</em></div>
        <div><strong>Attention:</strong> &nbsp;&nbsp;&nbsp;&nbsp; A_t = softmax(Q_t K_{1:t}ᵀ / &radic;d) V_{1:t}</div>
      `,
      footerNote: 'Characteristic: Every historical token preserves its exact individual position and vector representation.'
    },
    {
      tag: 'Step 02 · The Physical Problem',
      badgeClass: 'badge-derived',
      badgeText: '[DERIVED]',
      badgeSub: 'Mathematical Consequence',
      heading: '2. The Bottleneck: State Size Scales with Context',
      body: 'Because every token requires dedicated storage across all L layers and n_KV heads, the memory footprint expands linearly with sequence length T. As T grows from thousands to tens of thousands of tokens, multi-gigabyte KV caches must be repeatedly streamed across the memory bus for every single decode step.',
      diagramHtml: `
        <div><strong>Memory Footprint:</strong> VRAM_KV = 2 &times; L &times; n_KV &times; d_head &times; T &times; B &times; bytes_per_element</div>
        <div><strong>Scaling:</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; T &uarr; &nbsp;&implies;&nbsp; VRAM_KV &prop; O(T) &uarr; &nbsp;&implies;&nbsp; <em>Memory-Bandwidth Saturation</em></div>
        <div><strong>At T=128k:</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Llama-3-70B, FP16) &sim;160 GB VRAM consumed purely by past token cache.</div>
      `,
      footerNote: 'Consequence: The GPU Tensor Cores sit idle waiting for memory transfers, collapsing decode arithmetic intensity to ~1.5 FLOP/byte (Chapter 4).'
    },
    {
      tag: 'Step 03 · BDH Architectural Shift',
      badgeClass: 'badge-primary',
      badgeText: '[PRIMARY SOURCE]',
      badgeSub: 'Architecture',
      heading: '3. BDH: History Compressed into Recurrent Synaptic State',
      body: 'BDH replaces external token buffers with an evolving internal synaptic state. Input activations are projected into a high-dimensional sparse non-negative neuron space, and history is repeatedly accumulated into the synaptic connections S_t via associative plasticity.',
      diagramHtml: `
        <div><strong>Input Token:</strong> &nbsp;&nbsp;&nbsp;&nbsp; x_t &nbsp;&rarr;&nbsp; Sparse Positive Activation: a_t &ge; 0 (High dimension N)</div>
        <div><strong>Synaptic Update:</strong> S_t = Update(S_{t-1}, k_t, v_t)</div>
        <div><strong>Memory Scaling:</strong> &nbsp; Fixed size O(1) with respect to sequence length T</div>
      `,
      footerNote: 'Characteristic: No external KV cache is retained. The physical state footprint is constant whether T=100 or T=100,000.'
    },
    {
      tag: 'Step 04 · The Fundamental Trade-off',
      badgeClass: 'badge-derived',
      badgeText: '[DERIVED]',
      badgeSub: 'Architectural Trade-off',
      heading: '4. The Trade-off: What We Gain vs. What We Give Up',
      body: 'BDH removes the requirement to allocate one persistent KV entry per token, but it does not create infinite information capacity. Memory is transformed from an externalized lookup table into an evolving internal state.',
      isTradeoff: true,
      tradeoffGain: [
        'Sequence-length-independent state size (O(1) w.r.t. T)',
        'Elimination of decode-time multi-gigabyte KV cache streaming transfers',
        'Substantially higher concurrent serving throughput under fixed GPU VRAM',
        'Inherent recurrent working memory updated in place'
      ],
      tradeoffCost: [
        'Finite information capacity: cannot store unbounded historical precision',
        'Associative interference: concurrent concepts share high-dimensional synaptic pathways',
        'No exact token-addressable retrieval of arbitrary past states',
        'System behavior is governed by learned recurrent dynamics rather than verbatim lookup'
      ],
      footerNote: 'Conclusion: BDH does not make memory disappear—it changes what memory means.'
    }
  ];

  function renderStep(idx) {
    currentStep = idx;
    const s = STEPS[idx];

    let html = `
      <div class="step-badge-row">
        <span class="step-num-tag">${s.tag}</span>
        <div style="display: flex; align-items: center; gap: 6px;">
          <span class="evidence-badge ${s.badgeClass}">${s.badgeText}</span>
          <span style="font-size: 0.72rem; color: #6b7280; font-family: var(--font-mono);">${s.badgeSub}</span>
        </div>
      </div>
      <h4 class="step-heading">${s.heading}</h4>
      <p class="step-body-text">${s.body}</p>
    `;

    if (s.isTradeoff) {
      html += `
        <div class="step-tradeoff-grid">
          <div class="tradeoff-col tradeoff-gain">
            <h5 class="tradeoff-title">What We Gain &uarr;</h5>
            <ul class="tradeoff-list">
              ${s.tradeoffGain.map(g => `<li>${g}</li>`).join('')}
            </ul>
          </div>
          <div class="tradeoff-col tradeoff-cost">
            <h5 class="tradeoff-title">What We Give Up &darr;</h5>
            <ul class="tradeoff-list">
              ${s.tradeoffCost.map(c => `<li>${c}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
    } else {
      html += `
        <div class="step-diagram-box">
          ${s.diagramHtml}
        </div>
      `;
    }

    html += `
      <div style="font-size: 0.78rem; color: #6b7280; font-family: var(--font-mono); margin-top: 14px; border-top: 1px dashed #e5e7eb; padding-top: 10px;">
        ${s.footerNote}
      </div>
    `;

    contentEl.innerHTML = html;

    // Update nav pills
    stepBtns.forEach((btn, bIdx) => {
      btn.classList.toggle('active', bIdx === idx);
    });

    // Update controls
    if (indicatorEl) {
      indicatorEl.textContent = `Step ${idx + 1} of ${STEPS.length}`;
    }
    if (btnPrev) btnPrev.disabled = (idx === 0);
    if (btnNext) btnNext.disabled = (idx === STEPS.length - 1);
  }

  stepBtns.forEach((btn, idx) => {
    btn.addEventListener('click', () => renderStep(idx));
  });

  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      if (currentStep > 0) renderStep(currentStep - 1);
    });
  }

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      if (currentStep < STEPS.length - 1) renderStep(currentStep + 1);
    });
  }

  renderStep(0);
}

// ─── Main Initialization ───────────────────────────────────────
function init() {
  initTypewriter();
  initDilemmaWidget();
  initChapter2();
  initChapter3();
  initChapter4();
  initChapter5();
  initChapter6();
  // New features
  initChapterNav();      // ③ Floating chapter nav rail
  initSurgeryHUD();      // ① Surgery HUD + ② Crime Scene
  initParetoChart();     // ⑤ Pareto scatter plot
  initEpilogueQuiz();    // ④ Epilogue quiz
  initScrollTrigger();   // ⑦ Auto-run benchmark on scroll
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// ══════════════════════════════════════════════════════════════
// ③ CHAPTER NAV RAIL
// ══════════════════════════════════════════════════════════════
function initChapterNav() {
  const nav = document.getElementById('chapter-nav');
  if (!nav) return;

  const pills = nav.querySelectorAll('.nav-pill');
  const sections = ['hero', 'chapter-1', 'chapter-2', 'chapter-3', 'chapter-4', 'chapter-5', 'chapter-6', 'quiz-section'];

  // Click → smooth scroll
  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      const target = document.getElementById(pill.dataset.target);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  // IntersectionObserver → highlight active pill
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        pills.forEach(p => p.classList.toggle('active', p.dataset.target === id));
      }
    });
  }, { threshold: 0.3, rootMargin: '-10% 0px -50% 0px' });

  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}

// ══════════════════════════════════════════════════════════════
// ① SURGERY HUD + ② CRIME SCENE VISUALIZER
// ══════════════════════════════════════════════════════════════
function initSurgeryHUD() {
  const canvas = document.getElementById('surgery-hud-canvas');
  const controls = document.getElementById('surgery-controls');
  if (!canvas || !controls) return;

  // Instantiate the Surgery HUD — it builds its own controls and draws the curve
  const hud = new SurgeryHUD('surgery-hud-canvas', 'surgery-controls');

  // ② Listen for result events to populate the Crime Scene table
  window.addEventListener('surgery-result', (e) => {
    renderCrimeScene(e.detail.result);
  });

  // Trigger initial crime scene render once HUD has data
  hud.runExperiment();
}

function renderCrimeScene(result) {
  const tbody = document.getElementById('crime-scene-tbody');
  if (!tbody || !result) return;

  const { recalls, crimeScenes } = result;

  if (!recalls || recalls.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:16px;">No data yet…</td></tr>';
    return;
  }

  const overlapCell = (overlap) => {
    if (!overlap) return '<td style="color:var(--text-muted)">—</td>';
    const pct = Math.round(overlap.overlapFraction * 100);
    const cls = pct >= 40 ? 'overlap-high' : pct >= 15 ? 'overlap-med' : 'overlap-low';
    const barW = Math.max(2, Math.round(pct * 0.6));
    return `<td class="${cls}"><span class="overlap-bar" style="width:${barW}px"></span>${overlap.label} <span style="opacity:.6">(${pct}%)</span></td>`;
  };

  tbody.innerHTML = recalls.map((r, i) => {
    const scene = crimeScenes[i] || { overlaps: [] };
    const matchClass = r.match ? 'match-yes' : 'match-no';
    const matchIcon = r.match ? '✓ Match' : '✗ Miss';
    return `
      <tr>
        <td><strong>${r.label}</strong></td>
        <td class="${matchClass}">${matchIcon}</td>
        ${overlapCell(scene.overlaps[0])}
        ${overlapCell(scene.overlaps[1])}
        ${overlapCell(scene.overlaps[2])}
      </tr>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════
// ⑤ BDH-CQ PARETO SCATTER CHART
// ══════════════════════════════════════════════════════════════
function initParetoChart() {
  const canvas = document.getElementById('pareto-canvas');
  if (!canvas) return;

  // Set canvas physical resolution
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = (rect.width || 600) * dpr;
  canvas.height = 320 * dpr;
  canvas.style.width = (rect.width || 600) + 'px';
  canvas.style.height = '320px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width || 600;
  const H = 320;
  const pad = { top: 40, right: 40, bottom: 55, left: 70 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Data: [label, cost_per_task_usd, arc_agi_accuracy_pct, isBDH]
  const models = [
    ['BDH-CQ (150M)', 0.0007, 29.5, true],
    ['Gemini Flash', 0.003, 32, false],
    ['o1-mini', 0.08, 62, false],
    ['GPT-4o', 0.12, 55, false],
    ['Claude 3.5 Sonnet', 0.15, 68, false],
  ];

  // Log scale for cost: from 0.0001 to 1.0
  const costMin = Math.log10(0.0005);
  const costMax = Math.log10(0.5);
  const accMin = 0;
  const accMax = 80;

  const xScale = (cost) => pad.left + ((Math.log10(cost) - costMin) / (costMax - costMin)) * plotW;
  const yScale = (acc) => pad.top + plotH - ((acc - accMin) / (accMax - accMin)) * plotH;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(0,0,0,0.08)';
  ctx.lineWidth = 1;
  [20, 40, 60, 80].forEach(acc => {
    const y = yScale(acc);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(acc + '%', pad.left - 6, y + 4);
  });
  [-3, -2, -1, 0].forEach(exp => {
    const x = xScale(Math.pow(10, exp));
    if (x < pad.left || x > pad.left + plotW) return;
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('$' + Math.pow(10, exp).toFixed(exp < 0 ? Math.abs(exp) : 0), x, pad.top + plotH + 18);
  });

  // Axes
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Cost per Task (USD, log scale) →', pad.left + plotW / 2, H - 8);
  ctx.save();
  ctx.translate(14, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('ARC-AGI-1 Accuracy (%) →', 0, 0);
  ctx.restore();

  // Title
  ctx.font = 'bold 12px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Cost–Efficiency Pareto Frontier', pad.left + plotW / 2, 20);

  // Pareto frontier line (connect BDH-CQ to best accuracy model going right)
  const sorted = [...models].sort((a, b) => a[1] - b[1]);
  ctx.strokeStyle = 'rgba(0,0,0,0.15)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  sorted.forEach(([, cost, acc], i) => {
    const x = xScale(cost), y = yScale(acc);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // Plot each model
  models.forEach(([label, cost, acc, isBDH]) => {
    const x = xScale(cost);
    const y = yScale(acc);
    const r = isBDH ? 9 : 6;

    // Outer ring for BDH
    if (isBDH) {
      ctx.beginPath();
      ctx.arc(x, y, 16, 0, Math.PI * 2);
      ctx.strokeStyle = '#065f46';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Dot
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = isBDH ? '#065f46' : '#000000';
    ctx.fill();

    // Label
    ctx.font = isBDH ? 'bold 11px Inter' : '10px Inter';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'center';
    const labelY = isBDH ? y - 22 : (acc > 50 ? y - 14 : y + 20);
    ctx.fillText(label, x, labelY);

    if (isBDH) {
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillStyle = '#065f46';
      ctx.fillText('← Cost-Efficiency Leader', x + 60, y + 4);
    }
  });
}

// ══════════════════════════════════════════════════════════════
// ④ EPILOGUE QUIZ
// ══════════════════════════════════════════════════════════════
function initEpilogueQuiz() {
  const body = document.getElementById('quiz-body');
  const timerEl = document.getElementById('quiz-timer');
  const progressBar = document.getElementById('quiz-progress-bar');
  if (!body) return;

  const QUESTIONS = [
    {
      q: 'In Chapter 1, a legal compliance AI asks for the exact dollar figure from Turn 3. Which memory strategy guarantees it can retrieve "$1,450,000" verbatim 50 turns later?',
      options: ['A. Compressed associative state (BDH)', 'B. Verbatim KV cache store', 'C. Sliding window with attention sinks', 'D. It doesn\'t matter — all approaches recall this equally'],
      correct: 1,
      explanation: 'Only the verbatim KV cache preserves exact token-level recall. BDH and SSMs trade this for constant memory.'
    },
    {
      q: 'What is the memory complexity of a standard KV cache as context length T grows?',
      options: ['A. O(1) — constant, independent of T', 'B. O(log T) — logarithmic growth', 'C. O(T) — linear growth with sequence length', 'D. O(T²) — quadratic growth'],
      correct: 2,
      explanation: 'VRAM_KV = 2 × L × n_KV × d_head × T × B × bytes — linear in T. This is the core bottleneck.'
    },
    {
      q: 'The GPU Roofline model in Chapter 4 shows that autoregressive decode has arithmetic intensity of ~1–2 FLOP/byte. This means decode is:',
      options: ['A. Compute-bound — Tensor Cores are the bottleneck', 'B. Memory-bandwidth bound — HBM transfers are the bottleneck', 'C. Neither — the GPU is perfectly balanced', 'D. I/O bound — disk access is the bottleneck'],
      correct: 1,
      explanation: 'At 1–2 FLOP/byte, decode is far below the Roofline knee (~153 FLOP/byte on A100). The HBM memory bus is the bottleneck, not compute.'
    },
    {
      q: 'BDH\'s key architectural property is that its inference-time state footprint is:',
      options: ['A. O(T) — grows linearly with context length', 'B. O(d²) — grows with the square of embedding dimension', 'C. O(1) — constant regardless of sequence length T', 'D. O(log T) — grows logarithmically'],
      correct: 2,
      explanation: 'BDH stores history in a fixed synaptic matrix S ∈ R^(N×D). This state is constant in T — the memory footprint does not grow as the conversation lengthens.'
    },
    {
      q: 'When two concepts "Alpha" and "Beta" share active neurons in the BDH sparse representation, the expected cross-talk (interference) scales as:',
      options: ['A. Proportional to p (density)', 'B. Proportional to p² (density squared)', 'C. Proportional to N (total neuron count)', 'D. Independent of density — always constant'],
      correct: 1,
      explanation: 'E[overlap] ∝ p² — this is the key mathematical result. Sparse representations (small p) drive interference down quadratically, which is why BDH targets ~5% active density.'
    }
  ];

  let current = 0;
  let answers = new Array(QUESTIONS.length).fill(null);
  let timerSec = 60;
  let timerInterval = null;
  let quizStarted = false;

  function startTimer() {
    if (timerInterval) return;
    timerInterval = setInterval(() => {
      timerSec--;
      if (timerEl) {
        timerEl.textContent = timerSec + 's';
        timerEl.classList.toggle('urgent', timerSec <= 15);
      }
      if (timerSec <= 0) {
        clearInterval(timerInterval);
        showResult();
      }
    }, 1000);
  }

  function renderQuestion(idx) {
    if (idx >= QUESTIONS.length) { showResult(); return; }
    const q = QUESTIONS[idx];
    if (progressBar) progressBar.style.width = `${(idx / QUESTIONS.length) * 100}%`;

    const answered = answers[idx] !== null;
    body.innerHTML = `
      <div class="quiz-question-card">
        <div class="quiz-q-number">Question ${idx + 1} of ${QUESTIONS.length}</div>
        <p class="quiz-q-text">${q.q}</p>
        <div class="quiz-options">
          ${q.options.map((opt, i) => {
            let cls = 'quiz-option-btn';
            if (answered) {
              if (i === q.correct) cls += ' correct';
              else if (i === answers[idx]) cls += ' wrong';
            } else if (i === answers[idx]) cls += ' selected';
            return `<button class="${cls}" data-idx="${i}" ${answered ? 'disabled' : ''}>
              <span class="quiz-option-letter">${String.fromCharCode(65 + i)}</span>${opt.substring(3)}
            </button>`;
          }).join('')}
        </div>
        ${answered ? `<p style="margin-top:12px;font-size:0.76rem;color:var(--text-muted);border-left:2px solid rgba(0,0,0,0.2);padding-left:8px;">${q.explanation}</p>` : ''}
      </div>
      <div class="quiz-nav">
        <button class="quiz-nav-btn" id="quiz-prev" ${idx === 0 ? 'disabled' : ''}>← Prev</button>
        <span class="quiz-q-status">${answers.filter(a => a !== null).length}/${QUESTIONS.length} answered</span>
        <button class="quiz-nav-btn" id="quiz-next">${idx === QUESTIONS.length - 1 ? 'See Results →' : 'Next →'}</button>
      </div>`;

    // Option click
    body.querySelectorAll('.quiz-option-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (!quizStarted) { quizStarted = true; startTimer(); }
        answers[idx] = parseInt(btn.dataset.idx);
        renderQuestion(idx);
      });
    });

    document.getElementById('quiz-prev')?.addEventListener('click', () => renderQuestion(idx - 1));
    document.getElementById('quiz-next')?.addEventListener('click', () => {
      if (idx === QUESTIONS.length - 1) showResult();
      else renderQuestion(idx + 1);
    });
  }

  function showResult() {
    clearInterval(timerInterval);
    const score = answers.reduce((s, a, i) => s + (a === QUESTIONS[i].correct ? 1 : 0), 0);
    const pct = Math.round((score / QUESTIONS.length) * 100);
    if (progressBar) progressBar.style.width = '100%';
    const grades = [
      [100, '🏆 Perfect — You understand the frontier!', 'You nailed every concept from KV cache complexity to BDH synaptic interference.'],
      [80, '🎯 Excellent — Deep understanding demonstrated.', 'You grasp the core trade-offs between verbatim recall and constant-memory architectures.'],
      [60, '✓ Good — Solid grasp of the fundamentals.', 'Review the Roofline model (Ch 4) and the E[overlap] ∝ p² interference result (Ch 6).'],
      [40, '📖 Partial — A few concepts to revisit.', 'Re-read Chapters 3 and 6 on VRAM scaling and BDH\'s synaptic state properties.'],
      [0, '🔁 Keep exploring — the essay is your guide.', 'Scroll back to Chapter 1 and follow the narrative from KV cache to synaptic memory.'],
    ];
    const [, grade, gradeSub] = grades.find(([min]) => pct >= min);
    body.innerHTML = `
      <div class="quiz-result-card">
        <div class="quiz-score-big">${score}/${QUESTIONS.length}</div>
        <div class="quiz-score-label">${pct}% correct</div>
        <div class="quiz-grade">${grade}</div>
        <div class="quiz-grade-sub">${gradeSub}</div>
        <button class="quiz-restart-btn" id="quiz-restart">↺ Try Again</button>
      </div>`;
    document.getElementById('quiz-restart')?.addEventListener('click', () => {
      answers = new Array(QUESTIONS.length).fill(null);
      timerSec = 60; quizStarted = false; current = 0;
      timerInterval = null;
      if (timerEl) { timerEl.textContent = '60s'; timerEl.classList.remove('urgent'); }
      renderQuestion(0);
    });
  }

  renderQuestion(0);
}

// ══════════════════════════════════════════════════════════════
// ⑦ SCROLL-TRIGGERED BENCHMARK AUTO-RUN
// ══════════════════════════════════════════════════════════════
function initScrollTrigger() {
  const chapter2 = document.getElementById('chapter-2');
  const banner = document.getElementById('auto-run-banner');
  const btn = document.getElementById('btn-run-benchmark');
  if (!chapter2 || !btn) return;

  let hasAutoRun = false;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !hasAutoRun) {
        hasAutoRun = true;
        observer.disconnect();
        if (banner) banner.classList.remove('hidden');
        setTimeout(() => {
          btn.click();
          setTimeout(() => { if (banner) banner.classList.add('hidden'); }, 3000);
        }, 500);
      }
    });
  }, { threshold: 0.3 });

  observer.observe(chapter2);
}




