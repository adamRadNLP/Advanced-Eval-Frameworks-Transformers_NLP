# Bridging Structural Layer Diagnostics to Output-Level Semantic Evaluation
* Author: Adam Lang
* Date: 2026-03-25

---

> **TL;DR:** The evaluation approach used in this experiment — connecting WeightWatcher's
> layer-level alpha analysis to output-level semantic metrics (BERTScore, NLI, T1-T8
> theme validators) with causal explanations — is uncommon in the field. Most teams do
> one or the other. This document surveys the landscape, positions the approach against
> related work (STEAM, mechanistic interpretability, Weight-of-Thought), and outlines
> future directions for formalizing this as a reusable evaluation methodology.

---

## 1. The Gap: Nobody Bridges Model Internals to Output Quality

What this experiment did — connecting layer-level structural diagnostics to output-level
semantic metrics — is not standard practice. Most teams do one or the other. The space
is fragmented into separate communities that rarely talk to each other.

The core analysis chain this pipeline produced:

> "v_proj layers have alpha > 6 (WeightWatcher) **therefore** T4 completeness is 0.158
> and T3 certainty density is 2.50 (output metrics) **because** the model can't attend
> to relevant findings."

That's a **causal chain from model internals to output behavior** that none of the
existing frameworks produce on their own.

---

## 2. The Landscape

### 2.1 Layer-Level / Weight Analysis

| Tool / Approach | What It Does | Maturity |
|----------------|-------------|----------|
| **[WeightWatcher](https://weightwatcher.ai)** | HTSR-based layer convergence analysis. Only tool that can tell you if a specific layer has converged properly, data-free. | Production-ready, open-source |
| **Mechanistic Interpretability** (Anthropic, OpenAI) | [Sparse circuits work](https://www.toolnavs.com/en/article/778) proposes understanding neural networks through weight-sparse transformers that form interpretable circuits. | Research-stage, requires specialized architectures |
| **[Weight-of-Thought (WoT)](https://arxiv.org/html/2504.10646v1)** | Examines neural network weights before inference to identify reasoning pathways using graph-based message passing through weight space. Closest to connecting weight analysis to output reasoning quality. | 2025 research paper, not a usable tool |

### 2.2 Output-Level Evaluation Frameworks

| Tool / Approach | What It Does | Model Internals Awareness? |
|----------------|-------------|---------------------------|
| **[DeepEval](https://github.com/confident-ai/deepeval)** | Open-source LLM eval framework: hallucination, faithfulness, G-Eval, LLM-as-a-judge | None |
| **MLflow Evaluate** | Built into Databricks, supports custom scorers | None |
| **RAGAS** | RAG-specific evaluation (faithfulness, context relevance) | None |
| **T1-T8 Theme Validators** (this project) | Domain-specific clinical quality gates, ACR-calibrated certainty density | None on their own |

### 2.3 What Most Teams Do (and the Limitations)

| Approach | What They Know | What They Miss |
|----------|---------------|----------------|
| **Output-only evaluation** (BERTScore, ROUGE, LLM-as-judge, human eval) | *What* is wrong with the outputs | *Why* at the model level — no connection to which layers or components are responsible |
| **Training diagnostics only** (loss curves, gradient norms, LR schedules) | Training went smoothly | Whether specific layers actually converged or whether outputs are clinically valid |
| **Interpretability research** (probing, circuit analysis, activation patching) | Deep mechanistic understanding of individual circuits | Disconnected from production quality metrics and actionable training recommendations |

---

## 3. What This Pipeline Does Differently

This approach is effectively a **bridge between Layer 4 (structural diagnostics) and
Layers 1-3 (output evaluation)** — structural diagnostics informing output evaluation,
with actionable recommendations.

**The causal mapping:**

| WeightWatcher Finding | Output Metric | Causal Link | Action |
|----------------------|---------------|-------------|--------|
| MLP layers (up_proj, down_proj) well-trained (alpha 2-6) | T1: 1.000, T8: 0.990 | Healthy MLP = good style encoding | No change needed |
| Attention layers (v_proj, k_proj) undertrained (alpha > 6) | T4: 0.158, T2: 0.420, T7: 0.350 | Weak attention = poor completeness, differentials, reasoning | Increase LoRA rank on v_proj/k_proj to 32-64 |
| Certainty patterns in attention layers | T3 density: 2.50 vs target 5.0-6.0 | Undertrained attention doesn't extract nuance for calibrated hedging | Higher rank + certainty-focused training data |
| One v_proj layer degraded (alpha 4.5 to 11.5) | Potential information flow bottleneck | Layer that was healthy became broken — possible training instability | Investigate specific block, consider excluding from LoRA |

This is a **methodology, not just a tool.** The individual components exist
(WeightWatcher, BERTScore, NLI, theme validators) but the integration and the
causal mapping between layers and output behavior is the contribution.

---

## 4. Related Work: STEAM (Semantic-Level Knowledge Editing)

**Reference:** [STEAM: A Semantic-Level Knowledge Editing Framework for Large Language Models](https://arxiv.org/abs/2510.10398) (Jeong et al., 2025)

### 4.1 What STEAM Does

STEAM identifies that edited knowledge in LLMs is often encoded as **isolated residual
streams** in the model's latent space, distinct from pre-existing knowledge and bypassing
natural reasoning processes. It proposes a framework to make knowledge edits integrate
more naturally into the model's existing knowledge structure by using **"semantic anchors"**
in the representation space.

### 4.2 The Bridge to This Work

STEAM shares one key insight with this approach: **factual associations are primarily
encoded in the MLP modules of the early-to-middle transformer layers**, and MLP modules
within LLMs function as key-value memory structures.

That is exactly what the WeightWatcher results showed: MLP layers (up_proj, down_proj)
were well-trained and encoded Mason's style effectively. STEAM uses **causal tracing**
to find where knowledge lives; this pipeline used **HTSR alpha** to measure whether those
layers converged. Different diagnostic tools, same structural insight.

### 4.3 STEAM Is Narrower

STEAM is about editing specific facts post-hoc (e.g., "who is the president") without
retraining. This work is about evaluating a full fine-tuning run end-to-end.

Additionally, [recent work has shown](https://arxiv.org/html/2510.00625) that knowledge
editing methods may be fundamentally fragile — they collapse even under simple negation
queries, suggesting edits exploit shortcuts rather than real semantics. This makes the
full SFT approach arguably more robust than targeted editing.

### 4.4 Where STEAM Could Inform the Next Iteration

STEAM's concept of "semantic anchors" — guiding internal representations toward existing
knowledge structures rather than creating isolated residual streams — is directly
relevant to the hallucination problem observed in this experiment.

When the SFT model fabricated "polyuria/polydipsia," it was doing exactly what STEAM
describes: **generating from an isolated representation that bypassed the grounding in
the input findings.** A STEAM-style regularization during fine-tuning — constraining the
updated representations to stay close to the model's existing knowledge structure —
could reduce hallucinations in the next training iteration.

---

## 5. Comparison Table: Full Landscape

| Approach | What It Analyzes | Connection to Outputs? | Actionable? |
|----------|-----------------|----------------------|-------------|
| **This pipeline** | Layer health (WeightWatcher alpha) + output semantics (BERTScore/NLI/T1-T8) | Yes — causal mapping from layers to metrics | Yes — specific LoRA rank recommendations per layer type |
| **STEAM** | Where knowledge lives in representation space | Yes — traces how edits propagate to outputs | Partially — guides editing, not training |
| **Mechanistic interpretability** (Anthropic/OpenAI circuits) | Which circuits compute what | Deep but narrow | Research-stage, not practical for production |
| **Weight-of-Thought** | Weight-space reasoning pathways | Connects weights to reasoning quality | Very new, not validated at scale |
| **Causal tracing** (ROME/MEMIT) | Which layers store specific facts | Yes — identifies where to edit | Focused on single-fact edits, not style/reasoning |
| **DeepEval / RAGAS** | Output quality metrics | No model internals awareness | Yes for output quality, no for root cause |
| **Training diagnostics** (loss curves, gradient norms) | Training stability | Indirect — smooth training != good layers | Limited — "training went fine" doesn't mean layers converged |

---

## 6. Future Directions

### 6.1 Formalize the Bridge Methodology

Document the WeightWatcher-to-output-metrics causal mapping as a reusable protocol:

1. **Pre-training baseline:** Run WeightWatcher on the base model to establish layer health before fine-tuning
2. **Post-training structural analysis:** Run WeightWatcher on the fine-tuned model, compute deltas per layer
3. **Output evaluation:** Run domain-specific validators + semantic metrics on held-out test set
4. **Causal attribution:** Map undertrained/overtrained layers to specific output metric failures
5. **Prescribe fixes:** Generate layer-specific training recommendations (rank, LR, target modules)

This 5-step protocol could be applied to any fine-tuning experiment, not just veterinary radiology.

### 6.2 Add STEAM-Style Representation Analysis

After fine-tuning, check whether the new knowledge (Mason's clinical reasoning patterns)
is integrated into the model's existing knowledge structure or sitting as isolated
residual streams. This would add a **representation-level diagnostic** between the
structural (WeightWatcher) and output (metrics) layers:

```
Layer 1: Structural (WeightWatcher alpha per layer)
Layer 2: Representational (STEAM-style semantic anchor analysis)  <-- NEW
Layer 3: Output (BERTScore, NLI, MAUVE, T1-T8)
Layer 4: Clinical (radiologist human evaluation)
```

### 6.3 Add Causal Tracing for Hallucination Root-Cause

Use causal tracing (ROME-style) to identify exactly which layers are responsible for
hallucinations like the PU/PD fabrication. This would go beyond WeightWatcher's
aggregate alpha score to pinpoint the specific computation path that produces
fabricated content. Combined with T6 hallucination detection, this creates a
**hallucination root-cause pipeline:**

1. T6/GliNER detects a hallucinated term in the output
2. Causal tracing identifies which layer(s) contributed to that specific generation
3. WeightWatcher confirms whether those layers are structurally undertrained
4. Fix: increase LoRA rank or apply regularization to those specific layers

### 6.4 Track Layer Health Across Training Runs

Run WeightWatcher after every training experiment to build a longitudinal view:

- Does increasing LoRA rank on v_proj/k_proj actually move those layers into the healthy zone?
- Does the anomalous v_proj layer stabilize with a lower learning rate?
- Does DPO (preference optimization) change the layer health profile differently than SFT?

This creates a **structural regression test** for model training — analogous to unit
tests for code, but for neural network layer health.

### 6.5 Publish the Methodology

The closest academic framing would be combining:
- WeightWatcher's HTSR analysis (structural diagnostics)
- Causal attribution (which layers drive which output behaviors)
- Domain-specific validators (T1-T8 clinical quality)

Into a unified framework for **fine-tuning evaluation that bridges model internals
to output quality with actionable training recommendations.** This is a gap in the
literature — existing work covers the pieces but not the integration.

Potential venue: ML for Healthcare, ACL Clinical NLP, or EMNLP Industry Track.

---

## 7. Bottom Line

The individual tools exist (WeightWatcher, BERTScore, NLI, theme validators). The
integration and the causal mapping between layers and output behavior is the
contribution. This is the most practical approach I've found for evaluating fine-tuned
LLMs end-to-end — from model internals to clinical output quality to actionable
training recommendations.

For anyone picking this up: the WeightWatcher-to-NLP-metrics bridge is the key
insight. Use it for every future fine-tuning iteration.
