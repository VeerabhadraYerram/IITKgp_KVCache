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
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}



