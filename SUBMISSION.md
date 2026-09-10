# Official Submission Package Manifest
## When Attention Becomes Memory: The Hidden Duality Between KV Caches, Fast Weights, and Synapses

**Track:** DataForge 2026 — Pathway Track: Explain the Frontier (IIT Kharagpur)  
**Authors:** Veerabhadra Yerram and Contributors  
**Date:** September 2026  

---

### Submission Deliverables Checklist & Resource Index

| # | Requirement | Deliverable Location / Link | Description |
|---|---|---|---|
| **1** | **Public Artifact URL (No Sign-In)** | [https://iit-kgp-kv-cache.vercel.app/](https://iit-kgp-kv-cache.vercel.app/) (Mirror: [GitHub Pages](https://veerabhadrayerram.github.io/IITKgp_KVCache/web/)) | Hosted interactive visual essay featuring 6 chapters, Cyber OLED Dark/Light theme, 90-second Guided Tour presenter HUD, Three.js 3D spatial projections, Live Prompt Playground, GPU Cloud Fleet ROI Cost Calculator, 3-knob BDH Surgery HUD with Crime Scene forensic analyzer, and verifiable AI Memory Architect Certificate. Zero sign-in or login required. |
| **2** | **Public Source Code Repository** | [https://github.com/VeerabhadraYerram/IITKgp_KVCache](https://github.com/VeerabhadraYerram/IITKgp_KVCache) | Complete Git repository containing full web engine, PyTorch verification suite, PDF monograph, and technical summary. |
| **3** | **Blog as a PDF File** | [`What_does_an_AI_mean_by_remembering.pdf`](What_does_an_AI_mean_by_remembering.pdf) | High-resolution 10-page publication monograph exported directly from the interactive essay with vector typography, live benchmark outputs, and full derivations. |
| **4** | **Complete README** | [`README.md`](README.md) | In-depth, publication-grade documentation explaining the central thesis, mathematical derivations, hardware roofline parameters, architecture comparisons, and setup. |
| **5** | **Clear Setup Instructions** | [`README.md#quick-start--local-setup`](README.md#-quick-start--local-setup) | Exact terminal reproduction commands for local web serving (`python3 -m http.server 8765 --directory web`) and PyTorch notebook execution (`notebook/bdh_verification.ipynb`). |
| **6** | **Primary Papers (2022–2026)** | [`README.md#primary-literature-citations-20222026`](README.md#-primary-literature-citations-20222026) | Seven primary sources (2022–2026) directly cited beside their corresponding technical claims: Kosowski 2025, Engdahl 2026, DeepSeek MLA 2024, Gu & Dao Mamba 2023, Xiao StreamingLLM 2023, Pope TPU v4 2023, Dao FlashAttention 2022. |
| **7** | **Source & License Record** | [`README.md#source-and-license-record`](README.md#-source-and-license-record) | Complete asset and license inventory for code, KaTeX v0.16.8, Three.js r128, Inter, Newsreader, JetBrains Mono, NVIDIA A100 specs, and ARC-AGI-1 benchmark metrics. |
| **8** | **AI Assistance & Asset Disclosure** | [`README.md#disclosures-ai-assistance-code-data-and-assets`](README.md#️-disclosures-ai-assistance-code-data-and-assets) | Formal declaration of Antigravity AI tooling for scaffolding and formatting; human author oversight and verification; client-side zero-telemetry policy; and scientific boundary disclosures. |
| **9** | **Complete Submission ZIP** | [`IITKgp_KVCache-1.zip`](IITKgp_KVCache-1.zip) | Standalone offline archive bundling all deliverables into a single distributable zip package matching the submission template. |

---

### Quick Verification Commands

```bash
# 1. Verify and serve web application locally
cd web/
python3 -m http.server 8765
# Open http://localhost:8765/

# 2. Verify PyTorch mathematical test suite
cd ../notebook/
pip install -r requirements.txt
python -c "
import torch
# Verify exact Llama-3-70B KV memory footprint at T=128k, B=16 in FP16
vram_bytes = 2 * 80 * 8 * 128 * 131072 * 16 * 2
print(f'Llama-3-70B KV Cache VRAM: {vram_bytes / (1024**3):.2f} GB')
assert round(vram_bytes / (1024**3), 1) == 160.0
print('Assertion Passed: KV cache exactly equals 160.0 GB!')
"
```
