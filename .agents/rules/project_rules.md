# Global Rules for "When Attention Becomes Memory"

These rules apply globally across all chapters, scripts, visualizers, and documentation.

## 1. Visual & Animation Honesty
Any animation, chart, or 3D visualization must be one of exactly two things:
- **(a) Driven by real computation** happening in the browser/backend right now — actual numbers from actual math, actual measured timings (`performance.now()`), actual tensor shapes.
- **(b) Explicitly labeled in the UI, visibly, as:** `"Teaching simplification — not a live computation."`
- **Never silently animate a fake process as if it were real.**

## 2. No Invented Numbers, Formulas, or Quotes
- Never invent a statistic, benchmark number, formula, or attributed quote.
- If not certain a number is real, state the claim qualitatively without numbers, or insert a visible marker: `[NEEDS SOURCE - do not ship]`.

## 3. Strict & Verifiable Citations
- Every named formula or technique must be attributed to a real, correctly matched citation.
- Do not co-cite multiple papers for one claim unless both independently derived it.
- If unsure which paper a formula belongs to, mark `[VERIFY ATTRIBUTION]`.

## 4. Grounded Geometry & Diagrams
- For any 3D or diagram rendering of transformer/attention/neuron-graph: geometry and motion must be generated FROM real underlying data (e.g., actual token embedding vectors, actual computed attention weights, actual K/V tensor shapes) — never random decorative values pretending to be data.
- If real-time client-side calculation is infeasible, use small real model precomputed outputs exported as JSON and label as precomputed.

## 5. Design Consistency
- Maintain the visual design system established in Chapter 1: monochromatic high-contrast light theme (`#ffffff` background, `#000000` text, crisp 1px borders, Inter & JetBrains Mono typography, technical architectural background elements).
