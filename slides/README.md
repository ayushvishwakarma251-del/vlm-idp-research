# Bolt Slides — VLM-IDP Research Presentation Deck

Interactive, web-native presentation deck for:

**Vision-Language Models for Intelligent Document Processing under Real-World Visual Degradation**

Built with Vite, React, Framer Motion, and the **Bolt Slides** presentation engine.

---

## Quick Start

Launch the interactive presentation with one click using:
```cmd
.\slides.bat
```
Or manually:
```bash
cd slides
npm run dev
```
Then open `http://localhost:5173` in your browser.

---

## Keyboard Controls & Presenter Mode

| Key | Action |
| :--- | :--- |
| `→` / `Space` / `PageDown` | Next step or advance slide |
| `←` / `PageUp` | Previous step or previous slide |
| `P` | **Open Synchronized Presenter View** (Dual-screen mode with speaker notes, timers, and upcoming slide preview) |
| `F` | Toggle fullscreen mode |
| `S` | Toggle thumbnail navigation sidebar |
| `G` | Toggle grid overview |
| `A` | Toggle drawing annotator mode (live pen drawing over slides) |
| `H` | Hide / show dock chrome |
| `Home` / `End` | Jump to First / Last slide |

---

## Slide Structure (12 Slides)

1. **Title / Cover**: Research title, affiliations, hardware platform (RTX 3050 6GB), and Phase 14.5 audit status.
2. **The Problem Statement**: Contrast panel illustrating the fragility of static VLMs vs our adaptive framework under visual degradation.
3. **Core Research Pillars**: Bento grid covering Adaptive Routing, Evidence Grounding, Calibrated Abstention, and Edge GPU viability.
4. **End-to-End Pipeline**: High-resolution 3D visual diagram and data-flow walk-through.
5. **Physical Hardware Wall (7B OOM)**: 16 GB allocation negative control demonstrating the physical memory limit on 6GB GPUs.
6. **Physical Quantization Ladder**: Formatted data table of FP16, INT8, and INT4 NF4 benchmarks on CUDA 12.6.
7. **Hero Metric (52.2% VRAM Reduction)**: Animated `CountUp` highlighting memory savings with 16.5 tok/s throughput.
8. **Multi-Page Retrieval Page Pruning**: Context scaling plot, 60% page reduction, and 50% latency drop.
9. **Hallucination Suppression & Grounding**: StatGrid cards showcasing Grounding IoU (0.799) and 92% reduction in Unsupported Answer Rate.
10. **IEEE Scientific Rigor & Forensic Audit**: 4-step timeline of 1,100 traces, 10,000 bootstrap resamples, and 21,462 frozen baseline files.
11. **Interactive Document Studio**: Browser mockup showcasing real-time PDF extraction and tabular matching on `localhost:8896`.
12. **Conclusion & Publication Readiness**: Key scientific takeaways and reproducibility deliverables for IEEE peer review.
