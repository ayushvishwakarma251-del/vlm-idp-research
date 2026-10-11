import React from 'react';
import Deck from './deck/Deck';
import Slide from './deck/Slide';
import Build from './deck/Build';
import Reveal from './deck/Reveal';
import Cover from './components/Cover';
import Contrast from './components/Contrast';
import Bento from './components/Bento';
import Split from './components/Split';
import Table from './components/Table';
import BigNumber from './components/BigNumber';
import CountUp from './components/CountUp';
import StatGrid from './components/StatGrid';
import Steps from './components/Steps';
import BrowserFrame from './components/BrowserFrame';

export default function App() {
  return (
    <Deck>
      {/* Slide 1: Cover */}
      <Cover
        nav="Title"
        notes="Welcome. Presenting our IEEE research project on Vision-Language Models for Intelligent Document Processing under real-world visual degradation, with physical execution on edge GPUs."
        kicker="IEEE Research · Multimodal Document Intelligence"
        title={
          <>
            Adaptive, Evidence-Grounded{' '}
            <span className="accent-text">Vision-Language IDP</span>
          </>
        }
        subtitle="Robust Document Processing under Real-World Visual Degradation on Consumer Edge GPUs"
        foot="NVIDIA RTX 3050 6GB · BitsAndBytes 4-Bit NF4 · 1,100 Physical Traces · Phase 14.5 Audited"
      />

      {/* Slide 2: The Core Problem */}
      <Contrast
        nav="The Problem"
        notes="Highlight the contrast between fragile baseline VLMs and our adaptive, verifiable architecture."
        kicker="Problem Statement"
        title="The Fragility of Standard Document AI"
        left={{
          label: 'Static VLMs & OCR Baselines',
          title: 'Brittle & Hallucinatory',
          points: [
            'Severe accuracy degradation under blur, noise, skew, and low resolution',
            'Hallucinates plausible answers on multi-page forms without supporting evidence',
            'Context explosion causes CUDA Out-of-Memory failures on edge devices',
            'Uncalibrated confidence: equally confident when outputs are completely wrong',
          ],
        }}
        right={{
          label: 'Our Proposed Framework',
          title: 'Adaptive & Verifiable',
          points: [
            'Multi-signal visual quality scoring with dynamic tri-path routing',
            'Multimodal retrieval prunes context pages by 60%, halving generation latency',
            'Spatial evidence grounding links every answer to pixel bounding boxes',
            'Calibrated uncertainty & selective abstention suppresses 92% of hallucinations',
          ],
        }}
      />

      {/* Slide 3: Three Research Pillars */}
      <Bento
        nav="Core Pillars"
        notes="The central novelty is the intersection of all three pillars on consumer edge hardware."
        kicker="Core Contributions"
        title="Three Intersecting Research Innovations"
        tiles={[
          {
            c: 7,
            r: 1,
            variant: 'accent',
            k: 'Pillar 1',
            fig: 'Adaptive Routing',
            title: 'Quality-Aware Dynamic Pathways',
            body: 'Evaluates document degradation score and dynamically dispatches between Direct VLM (clean), Enhancement + VLM (moderate), and OCR-assisted Fallback (severe).',
          },
          {
            c: 5,
            r: 1,
            k: 'Pillar 2',
            fig: 'Evidence Grounding',
            title: 'Verifiable Localization',
            body: 'Extracts verifiable page citations and spatial bounding boxes, boosting spatial IoU from 0.453 to 0.799 across authentic documents.',
          },
          {
            c: 5,
            r: 1,
            k: 'Pillar 3',
            fig: 'Calibrated Abstention',
            title: 'Selective Prediction',
            body: 'Combines multi-signal uncertainty to identify unreliable inferences and safely abstain, reducing unsupported answer rate from 22.9% to 1.8%.',
          },
          {
            c: 7,
            r: 1,
            variant: 'glow',
            k: 'Pillar 4',
            fig: 'Edge GPU Viability',
            title: 'Consumer Hardware Optimization',
            body: 'Executes physical INT4 NormalFloat quantization on consumer 6GB GPUs, achieving 52.2% VRAM reduction while maintaining 16.5 tokens/sec.',
          },
        ]}
      />

      {/* Slide 4: System Architecture */}
      <Split
        nav="Architecture"
        notes="Walk through the multi-stage pipeline from document ingestion to calibrated answer verification."
        kicker="System Design"
        title={
          <>
            End-to-End <br />
            <span className="accent-text">Pipeline Flow</span>
          </>
        }
        body={
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, fontSize: '0.9em' }}>
            <div>
              <strong>1. Document Normalization:</strong> Multi-page PDF rasterization and layout preprocessing.
            </div>
            <div>
              <strong>2. Quality Assessment:</strong> Automated detection of blur, noise, skew, and contrast degradation.
            </div>
            <div>
              <strong>3. Adaptive Router:</strong> Quality-based dispatch across Clean, Moderate, and Severe pathways.
            </div>
            <div>
              <strong>4. Multimodal FAISS Index:</strong> Dense embedding retrieval isolating the top-2 supporting pages.
            </div>
            <div>
              <strong>5. Grounded VLM Reasoner:</strong> 4-bit NF4 vision-language inference with spatial coordinates.
            </div>
            <div>
              <strong>6. Calibrated Abstention:</strong> Multi-signal uncertainty gating ensuring high reliability.
            </div>
          </div>
        }
        media={
          <div
            style={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 24,
            }}
          >
            <img
              src="/architecture-3d-pipeline.svg"
              alt="System Architecture"
              style={{
                width: '100%',
                maxHeight: '85vh',
                objectFit: 'contain',
                filter: 'drop-shadow(0 20px 40px rgba(0,229,255,0.2))',
              }}
            />
          </div>
        }
      />

      {/* Slide 5: Physical Hardware Wall */}
      <Contrast
        nav="Hardware Wall"
        notes="Explain why 7B unquantized models fail on consumer hardware and how the 500M fallback model succeeds."
        kicker="Hardware Constraint"
        title="The 6GB Edge Memory Wall"
        left={{
          label: 'Target Tier 1: Qwen2.5-VL-7B',
          title: 'FP16 Out-Of-Memory on 6GB GPU',
          points: [
            'Weight footprint alone requires 15.2 GB in unquantized 16-bit precision',
            'Image patch embeddings and KV cache exceed 16.0 GB total allocation',
            'Triggered physical torch.OutOfMemoryError on NVIDIA RTX 3050 (6,144 MB)',
            'Legitimate empirical negative control: proves unquantized 7B edge infeasibility',
          ],
        }}
        right={{
          label: 'Fallback Tier 2: SmolVLM-500M',
          title: 'Sub-1GB Edge Multimodal Success',
          points: [
            'Compact 500M multimodal architecture (Idefics3 vision-language)',
            'Full FP16 baseline occupies 1,111.6 MB peak VRAM (fits comfortably in 6GB)',
            'BitsAndBytes INT4 NF4 quantization compresses peak memory to 530.9 MB',
            'High generation throughput of 16.49 tokens/second on consumer laptop GPU',
          ],
        }}
      />

      {/* Slide 6: Physical Quantization Benchmark */}
      <Slide
        center
        nav="Quantization"
        notes="Present the real physical CUDA 12.6 measurements from Table 04."
      >
        <div className="kicker" style={{ textAlign: 'center', marginBottom: 10 }}>
          Physical CUDA 12.6 Telemetry
        </div>
        <h2
          className="headline"
          style={{ textAlign: 'center', marginInline: 'auto', marginBottom: 28 }}
        >
          Live Quantization Benchmark Ladder
        </h2>
        <Table
          columns={[
            'Precision Mode',
            { label: 'Peak VRAM (MB)', align: 'right' },
            { label: 'Memory Savings', align: 'right' },
            { label: 'Latency (s)', align: 'right' },
            { label: 'Throughput', align: 'right' },
            { label: 'Physical Status', align: 'center' },
          ]}
          rows={[
            ['Q0 (FP16 Native)', '1,111.64 MB', 'Baseline (0.0%)', '0.863 s', '27.81 tok/s', 'PHYSICAL CUDA'],
            ['Q1 (INT8 LLM.int8)', '702.01 MB', '-36.85%', '2.802 s', '8.57 tok/s', 'PHYSICAL CUDA'],
            ['Q2 (INT4 NF4)', '530.95 MB', '-52.24%', '1.455 s', '16.49 tok/s', 'PHYSICAL CUDA'],
          ]}
          highlightRow={2}
          caption="Empirical benchmark on NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB VRAM, CUDA 12.6, PyTorch 2.14.1)"
        />
      </Slide>

      {/* Slide 7: Hero Metric */}
      <BigNumber
        nav="VRAM Savings"
        notes="Hold on this slide for dramatic impact. Over 52% memory reduction with viable interactive speed."
        kicker="Hero Metric · Edge Efficiency"
        value={<CountUp to={52.2} decimals={1} suffix="%" />}
        caption={
          <>
            Peak VRAM Footprint Reduction via 4-Bit NormalFloat Quantization
            <br />
            <span style={{ fontSize: '0.85em', color: 'var(--fg-muted)' }}>
              Compressed from 1,111.6 MB down to 530.9 MB on an NVIDIA RTX 3050 Laptop GPU
            </span>
          </>
        }
        foot="Physically validated in Phase 14 at 16.49 tokens/second neural throughput"
      />

      {/* Slide 8: Multi-Page Retrieval Page Pruning */}
      <Split
        nav="Page Pruning"
        notes="Demonstrate how multi-page retrieval pruning halves inference latency while boosting accuracy."
        kicker="Retrieval Efficiency"
        title={
          <>
            60% Context Pruning,<br />
            <span className="accent-text">50% Faster Latency</span>
          </>
        }
        body={
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <p>
              Full-document VLM processing consumes 5 pages of context (390 tokens), causing high latency (7.485 s) and visual distraction.
            </p>
            <p>
              Our multimodal retrieval module isolates the <strong>top-2 supporting pages</strong> (156 tokens), cutting context tokens by 60%.
            </p>
            <div style={{ padding: 16, background: 'var(--surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--hair)' }}>
              <div style={{ color: 'var(--primary)', fontWeight: 600, marginBottom: 4 }}>
                Key Empirical Gains:
              </div>
              <ul style={{ paddingLeft: 20, margin: 0, lineHeight: 1.6 }}>
                <li>Latency halved from <strong>7.485s down to 3.750s</strong> (49.9% speedup)</li>
                <li>Extraction Exact Match improved by <strong>+8.36%</strong> (p = 0.0002)</li>
                <li>Strict family cluster bootstrap CI: [+3.64%, +13.45%]</li>
              </ul>
            </div>
          </div>
        }
        media={
          <div
            style={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 20,
            }}
          >
            <img
              src="/fig_06_context_scaling_latency.png"
              alt="Context Scaling Latency"
              style={{
                maxWidth: '95%',
                maxHeight: '80vh',
                objectFit: 'contain',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--hair)',
                boxShadow: 'var(--shadow)',
              }}
            />
          </div>
        }
      />

      {/* Slide 9: Hallucination Elimination */}
      <StatGrid
        nav="Safety & Quality"
        notes="Highlight the dramatic reduction in unsupported answers from 22.9% down to 1.09%."
        kicker="Empirical Reliability"
        title="Hallucination Suppression & Grounding Gains"
        stats={[
          {
            value: <CountUp to={0.799} decimals={3} />,
            label: 'Grounding IoU',
            caption: 'Increased from 0.453 baseline (+76.5% spatial precision)',
          },
          {
            value: <CountUp to={1.09} decimals={2} suffix="%" />,
            label: 'Unsupported Answer Rate',
            caption: 'Suppressed from 22.91% baseline (92.0% hallucination reduction)',
          },
          {
            value: <CountUp to={89.09} decimals={2} suffix="%" />,
            label: 'Proposed Exact Match',
            caption: 'End-to-end extraction accuracy across 1,100 evaluated traces',
          },
          {
            value: '100%',
            label: 'Family Consistency',
            caption: 'Non-negative UAR reduction across all 11 document families (p < 1e-4)',
          },
        ]}
      />

      {/* Slide 10: Scientific Rigor & Forensic Audit */}
      <Steps
        nav="Forensic Audit"
        notes="Phase 14.5 verified 100% bit-for-bit historical immutability across 21,462 baseline files."
        kicker="IEEE Reproducibility Standard"
        title="Forensic Validation & Bit-for-Bit Immutability"
        items={[
          {
            title: '1,100 Physical Traces',
            body: 'Evaluated across 5 predefined seeds and 4 pipeline conditions over 11 authentic document families.',
          },
          {
            title: '10,000 Bootstrap Resamples',
            body: 'Paired document-family cluster bootstrap with Holm-Bonferroni multiple testing correction.',
          },
          {
            title: '21,462 Frozen Files',
            body: 'Full bit-for-bit SHA-256 verification confirmed zero historical mutation across Phase 0–14.',
          },
          {
            title: 'PASS_WITH_QUALIFICATIONS',
            body: 'Independent Phase 14.5 audit verified all statistical claims and registered transparent methodology disclosures.',
          },
        ]}
      />

      {/* Slide 11: Interactive Document Studio */}
      <Slide
        center
        nav="Document Studio"
        notes="Phase 15 implemented an interactive Document Studio on localhost:8896."
      >
        <div className="kicker" style={{ textAlign: 'center', marginBottom: 10 }}>
          Interactive Demonstration Layer
        </div>
        <h2
          className="headline"
          style={{ textAlign: 'center', marginInline: 'auto', marginBottom: 24 }}
        >
          Document Studio Platform on localhost:8896
        </h2>
        <div style={{ maxWidth: 900, marginInline: 'auto', width: '100%' }}>
          <BrowserFrame url="http://localhost:8896">
            <div
              style={{
                padding: '28px 36px',
                background: 'linear-gradient(180deg, rgba(13,20,32,0.95), rgba(7,10,19,0.98))',
                borderRadius: '0 0 var(--radius) var(--radius)',
                display: 'grid',
                gridTemplateColumns: '1.2fr 1fr',
                gap: 24,
                alignItems: 'center',
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ fontSize: '0.85em', color: 'var(--primary)', fontWeight: 600 }}>
                  📄 PyMuPDF Spatial Extraction & Layout Engine
                </div>
                <div style={{ fontSize: '1.15em', fontWeight: 700, color: '#fff' }}>
                  Sample Invoice Document Processing
                </div>
                <div style={{ fontSize: '0.88em', color: 'var(--fg-muted)', lineHeight: 1.5 }}>
                  Performs automatic tabular column matching for rates, line items, taxes, and multi-line dates with real-time confidence scores.
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                  <span className="chip lit">Rate: $120.00</span>
                  <span className="chip lit">Tax: 18%</span>
                  <span className="chip lit">Confidence: 94.2%</span>
                </div>
              </div>
              <div
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px dashed var(--hair)',
                  borderRadius: 'var(--radius-sm)',
                  padding: 18,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                }}
              >
                <div style={{ fontSize: '0.8em', color: 'var(--fg-faint)', textTransform: 'uppercase' }}>
                  Extracted Metadata Ledger
                </div>
                <div style={{ fontSize: '0.85em', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--fg-muted)' }}>Document Type:</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>Tax Invoice</span>
                </div>
                <div style={{ fontSize: '0.85em', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--fg-muted)' }}>Quality Route:</span>
                  <span style={{ color: 'var(--primary)', fontWeight: 600 }}>CLEAN (Direct VLM)</span>
                </div>
                <div style={{ fontSize: '0.85em', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--fg-muted)' }}>Spatial Grounding:</span>
                  <span style={{ color: '#38ef7d', fontWeight: 600 }}>IoU: 0.812 (Verified)</span>
                </div>
                <div style={{ fontSize: '0.85em', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--fg-muted)' }}>Decision Status:</span>
                  <span style={{ color: '#38ef7d', fontWeight: 600 }}>VERIFIED ANSWER</span>
                </div>
              </div>
            </div>
          </BrowserFrame>
        </div>
      </Slide>

      {/* Slide 12: Conclusion */}
      <Slide
        center
        nav="Conclusion"
        notes="Conclude with the three key takeaways and our IEEE publication readiness."
      >
        <div className="kicker" style={{ textAlign: 'center', marginBottom: 12 }}>
          Summary & Publication Readiness
        </div>
        <h2
          className="headline"
          style={{ textAlign: 'center', marginInline: 'auto', marginBottom: 32 }}
        >
          Defensible Edge Document Intelligence
        </h2>
        <div
          style={{
            maxWidth: 780,
            marginInline: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
            textAlign: 'left',
          }}
        >
          <Build at={1}>
            <div
              style={{
                padding: '18px 24px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--surface)',
                border: '1px solid var(--hair)',
              }}
            >
              <strong style={{ color: 'var(--primary)' }}>1. Robust Adaptive Intelligence:</strong> Quality-aware routing and calibrated abstention eliminate over 90% of visual degradation failures and unsupported answers.
            </div>
          </Build>
          <Build at={2}>
            <div
              style={{
                padding: '18px 24px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--surface)',
                border: '1px solid var(--hair)',
              }}
            >
              <strong style={{ color: 'var(--primary)' }}>2. Viable Consumer Edge Footprint:</strong> 4-bit NormalFloat (NF4) quantization unlocks physical VLM execution on sub-6GB consumer laptop GPUs with 16.5 tok/s speed.
            </div>
          </Build>
          <Build at={3}>
            <div
              style={{
                padding: '18px 24px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--surface)',
                border: '1px solid var(--hair)',
              }}
            >
              <strong style={{ color: 'var(--primary)' }}>3. IEEE-Quality Reproducibility:</strong> All 1,100 traces, 15 tables, 12 figures, and 21,462 frozen baseline files pass forensic statistical and integrity audits.
            </div>
          </Build>
        </div>
      </Slide>
    </Deck>
  );
}
