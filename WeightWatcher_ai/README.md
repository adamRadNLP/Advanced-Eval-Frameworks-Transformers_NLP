# WeightWatcher.ai evaluation 


---
# What WeightWatcher Measures
* WeightWatcher is a data-free, open-source diagnostic tool that uses Heavy-Tailed Self-Regularization (HTSR) theory from Random Matrix Theory to analyze the weight matrices of neural networks without needing any test data.
* I originally learned about this framework by using it to evaluate CLIP models on text-image pairs in Veterinary Radiology. However, it can be leveraged for other open source deep learning models as you see below. 
* It computes an alpha value for each layer that indicates structural health -- whether the layer has learned meaningful representations.
* For background on WeightWatcher analysis of Mistral models and fine-tuned LLMs, see:
  * Mistral model analysis: https://weightwatcher.ai/models/Mistral-summary.html
  * Fine-tuned LLM analysis guide: https://www.weightwatcher.ai/fine_tuned.html


 ---
 # To Run WeightWatcher Eval
 * I originally ran this on GCP Vertex AI, so I ran the script by opening a terminal in Vertex and running:

```
pip install weightwatcher
python sft_weightwatcher_analysis.py
```
* NOTE: Direct adapter analysis requires loading as a PyTorch model. WeightWatcher cannot analyze raw `.safetensors` files. The base vs merged comparison in Phases 2-3 captures the same insights. Thus, I commented out Phase 1 in the .py file.

---

# WeightWatcher Layer Analysis: Results & Recommendations
* Author: Adam Lang
* Original Date I ran this (before I was at Rad AI): 3/25/2026
* Tool: WeightWatcher v0.7.7 (HTSR / Random Matrix Theory)
* Note to Research team:
  - I ran this experiment to evaluate a fine-tuned Mistral-7b model that I had fine-tuned on a veterinary radiologists personal reports. You will likely have to adapt the code to evaluate our internal Mistral models.
  - The notes you see below are from the original experiment I ran. 

---

> **TL;DR:** Fine-tuning improved the model's structural health significantly — well-trained
> layers increased from 72% to 85%, and the worst layers saw dramatic improvement. The
> remaining weak layers are concentrated in `v_proj` and `k_proj` (attention key/value
> projections), which are the layers responsible for what the model attends to. This
> aligns with the clinical reasoning gap observed during radiologist testing. Future
> fine-tuning should increase LoRA rank specifically on these layers.

---

## 1. What WeightWatcher Measures

[WeightWatcher](https://weightwatcher.ai) is a data-free, open-source diagnostic tool
that uses Heavy-Tailed Self-Regularization (HTSR) theory from Random Matrix Theory to
analyze the weight matrices of neural networks without needing any test data. It computes
an **alpha** value for each layer that indicates structural health -- whether the layer
has learned meaningful representations.

For background on WeightWatcher analysis of Mistral models and fine-tuned LLMs, see:
- Mistral model analysis: https://weightwatcher.ai/models/Mistral-summary.html
- Fine-tuned LLM analysis guide: https://www.weightwatcher.ai/fine_tuned.html

| Alpha Range | Interpretation | Action |
|-------------|---------------|--------|
| 2 <= alpha <= 6 | Well-trained — healthy spectral properties | None needed |
| alpha > 6 | Undertrained — layer hasn't learned enough structure | Increase capacity (higher LoRA rank, more epochs, or full fine-tuning) |
| alpha < 2 | Overtrained — layer is over-specialized | Reduce learning rate or add regularization |

**Caveat:** Our LoRA rank is 16, which is in the small-n regime where individual alpha
estimates are noisier. The trends across layers are reliable; individual values should
be interpreted with a grain of salt.

---

## 2. Results Summary

### 2.1 Overall Comparison: Base vs Fine-Tuned

| Metric | Base Model | Fine-Tuned | Delta |
|--------|-----------|------------|-------|
| Mean alpha | 5.629 | 4.672 | -0.957 |
| Median alpha | 4.806 | 4.369 | -0.438 |
| Well-trained (2-6) | 163/226 (72.1%) | 192/226 (85.0%) | +29 layers |
| Undertrained (>6) | 61/226 (27.0%) | 32/226 (14.2%) | -29 layers |
| Overtrained (<2) | 2/226 (0.9%) | 2/226 (0.9%) | No change |
| Max alpha | 34.598 | 11.500 | -23.098 |

**Key finding:** Fine-tuning moved 29 layers from undertrained into the well-trained
zone. The worst layer improved from alpha=34.6 to alpha=8.8. Mean alpha moved firmly
into the healthy band. The fine-tuning structurally improved the model.

### 2.2 Layers Most Changed by Fine-Tuning (Top 10)

| Layer Type | Base Alpha | FT Alpha | Delta | Interpretation |
|-----------|-----------|---------|-------|----------------|
| v_proj | 34.60 | 8.79 | -25.81 | Massive improvement, still slightly undertrained |
| v_proj | 24.84 | 9.32 | -15.52 | Large improvement, still undertrained |
| v_proj | 22.27 | 9.11 | -13.16 | Large improvement, still undertrained |
| o_proj | 15.47 | 3.28 | -12.19 | Fully repaired -- now well-trained |
| v_proj | 16.35 | 7.43 | -8.92 | Improved but still undertrained |
| v_proj | 4.52 | 11.50 | +6.98 | ANOMALY: was healthy, became undertrained |
| up_proj | 9.63 | 2.78 | -6.85 | Fully repaired -- now well-trained |
| up_proj | 8.92 | 2.62 | -6.30 | Fully repaired -- now well-trained |
| v_proj | 14.13 | 8.55 | -5.58 | Improved but still undertrained |
| up_proj | 10.22 | 5.25 | -4.97 | Fully repaired -- now well-trained |

**Pattern:** `v_proj` layers saw the largest absolute improvements but many remain
above alpha=6. `up_proj` (MLP) layers were fully repaired at LoRA rank 16.

### 2.3 Remaining Weak Layers After Fine-Tuning

**32 undertrained layers (alpha > 6):**

The majority are `v_proj` (value projection) and `k_proj` (key projection) in the
attention mechanism. These are the layers that determine **what the model attends to**
when processing input text.

| Layer Type | Count Above Alpha 6 | Interpretation |
|-----------|---------------------|----------------|
| v_proj | ~15 layers | Value projections -- what information to extract |
| k_proj | ~10 layers | Key projections -- what to match queries against |
| Other | ~7 layers | Scattered across other projection types |

**Why this matters clinically:** The attention key and value projections are responsible
for the model's ability to identify which findings are relevant to which conclusions.
Weak `v_proj` and `k_proj` layers directly explain the clinical reasoning gap observed
during radiologist testing -- the model can produce Mason's writing style (learned by
MLP layers which are healthy) but struggles with deep reasoning about which findings
connect to which diagnoses (attention layers still undertrained).

**2 overtrained layers (alpha < 2):**

| Layer | Alpha | Location |
|-------|-------|----------|
| q_proj | 1.42 | First transformer block |
| k_proj | 1.44 | First transformer block |

These were already slightly overtrained in the base model. Fine-tuning did not change
them. Not concerning -- first-block overtraining is common in pretrained LLMs and
has minimal impact on output quality.

### 2.4 One Anomaly

One `v_proj` layer moved from alpha=4.52 (well-trained) to alpha=11.50 (undertrained)
after fine-tuning. This layer **got worse**. This could indicate:
- A distribution shift in the training data that destabilized this specific layer
- The learning rate was too high for this layer
- The LoRA rank was insufficient to capture the update needed

This layer should be investigated in future training runs. Possible mitigations:
layer-specific learning rate scheduling or excluding this layer from LoRA targeting
and using full fine-tuning instead.

---

## 3. Visualizations

All plots saved to `./weightwatcher_results/` and backed up to GCS at
`gs://overlook-data/hackathon/sft_models/weightwatcher_results/`.

| Plot | What It Shows |
|------|--------------|
| `alpha_histogram_comparison.png` | Distribution of alpha values: base (left) vs fine-tuned (right). FT histogram is tighter and shifted left toward the healthy 2-6 zone. |
| `correlation_flow_comparison.png` | Alpha vs layer depth. Shows how information flows through the model. FT model has more consistent flow with fewer spikes. |
| `alpha_vs_alpha.png` | Scatterplot of base alpha (x) vs FT alpha (y) per layer. Points below the diagonal improved; points above got worse. Most points are below. |

---

## 4. Recommendations for Future Fine-Tuning

### 4.1 Increase LoRA Rank on Attention Layers

The `v_proj` and `k_proj` layers are the bottleneck. Current rank=16 was sufficient
for MLP layers but not for attention. **Recommended configuration for next iteration:**

```python
# Layer-specific LoRA rank (if supported by PEFT version)
# Otherwise, increase global rank to 32 or 64

lora_config = LoraConfig(
    r=32,                    # Increase from 16 to 32
    lora_alpha=64,           # Keep 2x rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"        # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

Alternatively, if PEFT supports per-module rank configuration:
- `v_proj`, `k_proj`: r=64 (these need the most capacity)
- `q_proj`, `o_proj`: r=32
- `gate_proj`, `up_proj`, `down_proj`: r=16 (already healthy at this rank)

### 4.2 Investigate the Anomalous v_proj Layer

The layer that went from alpha=4.5 to alpha=11.5 needs investigation:
- Identify which transformer block it belongs to (check `layer_comparison.csv`)
- Consider excluding it from LoRA and using full fine-tuning instead
- Or apply a lower layer-specific learning rate

### 4.3 Monitor with WeightWatcher Across Training Runs

Run WeightWatcher after each training experiment to track:
- Are undertrained layers improving?
- Are any well-trained layers degrading (like the anomalous v_proj)?
- Is the overall alpha distribution tightening?

This gives a structural quality gate that complements output-based metrics.

### 4.4 Consider Larger Base Model

With 32 layers still undertrained at rank 16, there may be a capacity ceiling
for Mistral 7B on this task. A larger base model (Mistral-22B, Llama-70B) would
have more parametric capacity for clinical reasoning, potentially requiring less
aggressive LoRA rank increases to achieve similar structural health.

---

## 5. Connection to Observed Behavior

| Observation from Radiologist Testing | WeightWatcher Explanation |
|-------------------------------------|--------------------------|
| Style transfer was excellent | MLP layers (`up_proj`, `down_proj`, `gate_proj`) are well-trained -- these encode style and pattern |
| Clinical reasoning was weaker | Attention layers (`v_proj`, `k_proj`) are still undertrained -- these encode what to attend to and how to reason about relationships |
| Hallucinations on unseen data | Undertrained attention layers default to base model priors rather than grounding in input findings |
| Low temperature worked best | Narrow well-trained MLP layers produce good outputs when sampling stays on-distribution; undertrained attention layers add noise at higher temperatures |

---

## 6. Files

| File | Description |
|------|-------------|
| `weightwatcher_results/base_model_layer_details.csv` | Per-layer alpha values for base Mistral-7B-Instruct-v0.3 |
| `weightwatcher_results/ft_model_layer_details.csv` | Per-layer alpha values for fine-tuned model |
| `weightwatcher_results/layer_comparison.csv` | Side-by-side base vs FT alpha with delta, sorted by largest change |
| `weightwatcher_results/alpha_histogram_comparison.png` | Alpha distribution histograms |
| `weightwatcher_results/correlation_flow_comparison.png` | Alpha vs layer depth (information flow) |
| `weightwatcher_results/alpha_vs_alpha.png` | Base alpha vs FT alpha scatterplot |
| `weightwatcher_results/weightwatcher_summary.json` | Machine-readable summary statistics |
