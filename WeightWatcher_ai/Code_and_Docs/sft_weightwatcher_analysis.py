"""
WeightWatcher Layer Analysis: Fine-Tuned Mistral-7B
=====================================================
Author: Adam Lang
Date: 3-25-2026
Platform: Vertex AI Workbench (A100 40GB)

Purpose: Use WeightWatcher (HTSR / Random Matrix Theory) to analyze
the weight matrices of the fine-tuned model and identify:
  1. Which layers the LoRA fine-tuning actually changed
  2. Which layers are well-trained vs undertrained vs overtrained
  3. Where to focus future fine-tuning efforts (e.g., increase rank, target specific layers)

Reference: https://weightwatcher.ai/fine_tuned.html

Caveat: Our LoRA rank is 16, which is in the small-n regime where
alpha estimates are noisier. Look at trends across layers, not individual values.

Usage: pip install weightwatcher && python sft_weightwatcher_analysis.py
       Or run as cells in Jupyter Lab.
"""

# =============================================================================
# 1. Setup
# =============================================================================

import os
import gc
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ADAPTER_PATH = "./sft_output/mistral-7b_20260324_1714/final"
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./weightwatcher_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# 2. Install WeightWatcher
# =============================================================================

# !pip install weightwatcher -q

import weightwatcher as ww
print(f"WeightWatcher version: {ww.__version__}")

# =============================================================================
# 3. Analyze LoRA Adapter Directly
# =============================================================================
# NOTE: Direct adapter analysis requires loading as a PyTorch model.
# WeightWatcher cannot analyze raw .safetensors files.
# The base vs merged comparison in Phases 2-3 captures the same insights.

# print("=" * 80)
# print("PHASE 1: ANALYZE LORA ADAPTER WEIGHTS")
# print("=" * 80)
# print("Analyzing adapter_model.safetensors directly.")
# print("Caveat: LoRA rank=16 is small-n regime -- alpha estimates are noisier.")
# print("Look at trends across layers, not individual values.")
# print()

# # WeightWatcher can analyze the adapter file directly
# adapter_file = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
# print(f"Adapter file: {adapter_file}")
# print(f"Adapter size: {os.path.getsize(adapter_file) / 1e6:.1f} MB")

# watcher = ww.WeightWatcher()
# adapter_details = watcher.analyze(model=adapter_file)

# print(f"\nAdapter analysis complete: {len(adapter_details)} layers analyzed")
# print(f"\nAdapter alpha summary:")
# print(f"  Mean alpha:   {adapter_details['alpha'].mean():.3f}")
# print(f"  Median alpha: {adapter_details['alpha'].median():.3f}")
# print(f"  Std alpha:    {adapter_details['alpha'].std():.3f}")
# print(f"  Min alpha:    {adapter_details['alpha'].min():.3f}")
# print(f"  Max alpha:    {adapter_details['alpha'].max():.3f}")

# # Save adapter details
# adapter_details.to_csv(f"{OUTPUT_DIR}/adapter_layer_details.csv", index=False)
# print(f"\nSaved to {OUTPUT_DIR}/adapter_layer_details.csv")

# =============================================================================
# 4. Analyze Base Model
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 2: ANALYZE BASE MODEL (Mistral-7B-Instruct-v0.3)")
print("=" * 80)
print("This takes a few minutes -- analyzing all weight matrices in the base model.")

from transformers import AutoModelForCausalLM

print("Loading base model (float16 for analysis)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",  # WW works on CPU
)

watcher_base = ww.WeightWatcher(model=base_model)
base_details = watcher_base.analyze()

print(f"\nBase model analysis complete: {len(base_details)} layers analyzed")
print(f"\nBase model alpha summary:")
print(f"  Mean alpha:   {base_details['alpha'].mean():.3f}")
print(f"  Median alpha: {base_details['alpha'].median():.3f}")
print(f"  Std alpha:    {base_details['alpha'].std():.3f}")
print(f"  Min alpha:    {base_details['alpha'].min():.3f}")
print(f"  Max alpha:    {base_details['alpha'].max():.3f}")

# Count layers by quality zone
well_trained = ((base_details['alpha'] >= 2) & (base_details['alpha'] <= 6)).sum()
undertrained = (base_details['alpha'] > 6).sum()
overtrained = (base_details['alpha'] < 2).sum()
total_layers = len(base_details)

print(f"\n  Layer quality zones (HTSR):")
print(f"    Well-trained (2 <= alpha <= 6): {well_trained}/{total_layers} ({100*well_trained/total_layers:.1f}%)")
print(f"    Undertrained (alpha > 6):       {undertrained}/{total_layers} ({100*undertrained/total_layers:.1f}%)")
print(f"    Overtrained (alpha < 2):        {overtrained}/{total_layers} ({100*overtrained/total_layers:.1f}%)")

base_details.to_csv(f"{OUTPUT_DIR}/base_model_layer_details.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR}/base_model_layer_details.csv")

# Free memory
del base_model
gc.collect()

# =============================================================================
# 5. Analyze Merged Fine-Tuned Model
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 3: ANALYZE MERGED FINE-TUNED MODEL")
print("=" * 80)

from peft import PeftModel

print("Loading base + merging LoRA adapter...")
base_model_2 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
)
ft_model = PeftModel.from_pretrained(base_model_2, ADAPTER_PATH)
ft_model = ft_model.merge_and_unload()
print("Merged model ready.")

watcher_ft = ww.WeightWatcher(model=ft_model)
ft_details = watcher_ft.analyze()

print(f"\nFine-tuned model analysis complete: {len(ft_details)} layers analyzed")
print(f"\nFine-tuned model alpha summary:")
print(f"  Mean alpha:   {ft_details['alpha'].mean():.3f}")
print(f"  Median alpha: {ft_details['alpha'].median():.3f}")
print(f"  Std alpha:    {ft_details['alpha'].std():.3f}")
print(f"  Min alpha:    {ft_details['alpha'].min():.3f}")
print(f"  Max alpha:    {ft_details['alpha'].max():.3f}")

# Count layers by quality zone
well_trained_ft = ((ft_details['alpha'] >= 2) & (ft_details['alpha'] <= 6)).sum()
undertrained_ft = (ft_details['alpha'] > 6).sum()
overtrained_ft = (ft_details['alpha'] < 2).sum()
total_layers_ft = len(ft_details)

print(f"\n  Layer quality zones (HTSR):")
print(f"    Well-trained (2 <= alpha <= 6): {well_trained_ft}/{total_layers_ft} ({100*well_trained_ft/total_layers_ft:.1f}%)")
print(f"    Undertrained (alpha > 6):       {undertrained_ft}/{total_layers_ft} ({100*undertrained_ft/total_layers_ft:.1f}%)")
print(f"    Overtrained (alpha < 2):        {overtrained_ft}/{total_layers_ft} ({100*overtrained_ft/total_layers_ft:.1f}%)")

ft_details.to_csv(f"{OUTPUT_DIR}/ft_model_layer_details.csv", index=False)

# =============================================================================
# 6. Comparison: Base vs Fine-Tuned
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 4: BASE vs FINE-TUNED COMPARISON")
print("=" * 80)

print(f"\n{'Metric':<30} {'Base':>10} {'Fine-Tuned':>12} {'Delta':>10}")
print("-" * 65)
print(f"{'Mean alpha':<30} {base_details['alpha'].mean():>10.3f} {ft_details['alpha'].mean():>12.3f} {ft_details['alpha'].mean() - base_details['alpha'].mean():>+10.3f}")
print(f"{'Median alpha':<30} {base_details['alpha'].median():>10.3f} {ft_details['alpha'].median():>12.3f} {ft_details['alpha'].median() - base_details['alpha'].median():>+10.3f}")
print(f"{'Well-trained (2-6)':<30} {well_trained:>10} {well_trained_ft:>12} {well_trained_ft - well_trained:>+10}")
print(f"{'Undertrained (>6)':<30} {undertrained:>10} {undertrained_ft:>12} {undertrained_ft - undertrained:>+10}")
print(f"{'Overtrained (<2)':<30} {overtrained:>10} {overtrained_ft:>12} {overtrained_ft - overtrained:>+10}")

# =============================================================================
# 7. Visualizations
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 5: VISUALIZATIONS")
print("=" * 80)

# 7.1 Alpha histogram comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(base_details['alpha'].dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(x=2, color='green', linestyle='--', label='Well-trained zone')
axes[0].axvline(x=6, color='green', linestyle='--')
axes[0].set_title('Base Model (Mistral-7B-Instruct-v0.3)')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Layer Count')
axes[0].legend()

axes[1].hist(ft_details['alpha'].dropna(), bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[1].axvline(x=2, color='green', linestyle='--', label='Well-trained zone')
axes[1].axvline(x=6, color='green', linestyle='--')
axes[1].set_title('Fine-Tuned (Mason Savage SFT)')
axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('Layer Count')
axes[1].legend()

plt.suptitle('WeightWatcher Alpha Distribution: Base vs Fine-Tuned', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/alpha_histogram_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/alpha_histogram_comparison.png")
plt.close()

# 7.2 Correlation flow (alpha vs layer depth)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(range(len(base_details)), base_details['alpha'], s=8, alpha=0.6, color='steelblue')
axes[0].axhline(y=2, color='green', linestyle='--', alpha=0.5)
axes[0].axhline(y=6, color='green', linestyle='--', alpha=0.5)
axes[0].set_title('Base Model: Correlation Flow')
axes[0].set_xlabel('Layer Index')
axes[0].set_ylabel('Alpha')
axes[0].set_ylim(0, max(base_details['alpha'].max(), ft_details['alpha'].max()) * 1.1)

axes[1].scatter(range(len(ft_details)), ft_details['alpha'], s=8, alpha=0.6, color='coral')
axes[1].axhline(y=2, color='green', linestyle='--', alpha=0.5)
axes[1].axhline(y=6, color='green', linestyle='--', alpha=0.5)
axes[1].set_title('Fine-Tuned: Correlation Flow')
axes[1].set_xlabel('Layer Index')
axes[1].set_ylabel('Alpha')
axes[1].set_ylim(0, max(base_details['alpha'].max(), ft_details['alpha'].max()) * 1.1)

plt.suptitle('WeightWatcher Correlation Flow: Base vs Fine-Tuned', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_flow_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/correlation_flow_comparison.png")
plt.close()

# 7.3 Alpha vs Alpha scatterplot (base vs FT per layer)
if len(base_details) == len(ft_details):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(base_details['alpha'], ft_details['alpha'], s=12, alpha=0.5, color='purple')
    
    # Diagonal line (no change)
    max_val = max(base_details['alpha'].max(), ft_details['alpha'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No change')
    
    # Well-trained zone
    ax.axhline(y=2, color='green', linestyle=':', alpha=0.3)
    ax.axhline(y=6, color='green', linestyle=':', alpha=0.3)
    ax.axvline(x=2, color='green', linestyle=':', alpha=0.3)
    ax.axvline(x=6, color='green', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Base Model Alpha')
    ax.set_ylabel('Fine-Tuned Alpha')
    ax.set_title('Alpha vs Alpha: How Each Layer Changed After SFT')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/alpha_vs_alpha.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/alpha_vs_alpha.png")
    plt.close()
else:
    print("Layer count mismatch between base and FT -- skipping alpha-vs-alpha plot.")

# =============================================================================
# 8. Identify Weak Layers for Future Fine-Tuning
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 6: WEAK LAYER IDENTIFICATION")
print("=" * 80)
print("Layers that remain undertrained (alpha > 6) or overtrained (alpha < 2)")
print("after fine-tuning are candidates for targeted improvement.\n")

# Undertrained layers (alpha > 6)
undertrained_layers = ft_details[ft_details['alpha'] > 6].copy()
if len(undertrained_layers) > 0:
    print(f"UNDERTRAINED LAYERS (alpha > 6): {len(undertrained_layers)}")
    print("These layers may benefit from higher LoRA rank or full fine-tuning:")
    for _, row in undertrained_layers.head(20).iterrows():
        layer_name = row.get('name', row.get('layer_id', 'unknown'))
        print(f"  Layer {layer_name}: alpha={row['alpha']:.2f}")
else:
    print("No undertrained layers (alpha > 6) -- fine-tuning covered all layers well.")

print()

# Overtrained layers (alpha < 2)
overtrained_layers = ft_details[ft_details['alpha'] < 2].copy()
if len(overtrained_layers) > 0:
    print(f"OVERTRAINED LAYERS (alpha < 2): {len(overtrained_layers)}")
    print("These layers may be over-specialized -- consider lower learning rate or regularization:")
    for _, row in overtrained_layers.head(20).iterrows():
        layer_name = row.get('name', row.get('layer_id', 'unknown'))
        print(f"  Layer {layer_name}: alpha={row['alpha']:.2f}")
else:
    print("No overtrained layers (alpha < 2).")

# =============================================================================
# 9. Layers Most Changed by Fine-Tuning
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 7: LAYERS MOST CHANGED BY FINE-TUNING")
print("=" * 80)

if len(base_details) == len(ft_details):
    delta_alpha = ft_details['alpha'].values - base_details['alpha'].values
    
    comparison_df = pd.DataFrame({
        'layer_idx': range(len(delta_alpha)),
        'base_alpha': base_details['alpha'].values,
        'ft_alpha': ft_details['alpha'].values,
        'delta_alpha': delta_alpha,
        'abs_delta': np.abs(delta_alpha),
    })
    
    # Add layer names if available
    if 'name' in ft_details.columns:
        comparison_df['layer_name'] = ft_details['name'].values
    
    comparison_df = comparison_df.sort_values('abs_delta', ascending=False)
    
    print("Top 20 layers with largest alpha change:")
    print(f"{'Layer':<50} {'Base':>8} {'FT':>8} {'Delta':>8}")
    print("-" * 75)
    for _, row in comparison_df.head(20).iterrows():
        name = row.get('layer_name', f"layer_{row['layer_idx']}")
        print(f"{str(name)[:50]:<50} {row['base_alpha']:>8.2f} {row['ft_alpha']:>8.2f} {row['delta_alpha']:>+8.2f}")
    
    comparison_df.to_csv(f"{OUTPUT_DIR}/layer_comparison.csv", index=False)
    print(f"\nFull comparison saved to {OUTPUT_DIR}/layer_comparison.csv")

# =============================================================================
# 10. Save Summary and Push to GCS
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 8: SAVE AND BACKUP")
print("=" * 80)

summary = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3 + QLoRA (r=16)",
    "adapter_path": ADAPTER_PATH,
    "analysis_date": pd.Timestamp.now().isoformat(),
    "base_model": {
        "mean_alpha": float(base_details['alpha'].mean()),
        "median_alpha": float(base_details['alpha'].median()),
        "well_trained_pct": float(100 * well_trained / total_layers),
        "undertrained_pct": float(100 * undertrained / total_layers),
        "overtrained_pct": float(100 * overtrained / total_layers),
        "total_layers": int(total_layers),
    },
    "fine_tuned_model": {
        "mean_alpha": float(ft_details['alpha'].mean()),
        "median_alpha": float(ft_details['alpha'].median()),
        "well_trained_pct": float(100 * well_trained_ft / total_layers_ft),
        "undertrained_pct": float(100 * undertrained_ft / total_layers_ft),
        "overtrained_pct": float(100 * overtrained_ft / total_layers_ft),
        "total_layers": int(total_layers_ft),
    },
    "lora_config": {
        "rank": 16,
        "alpha_note": "Rank 16 is small-n regime -- alpha estimates noisier. Look at trends, not individual values."
    }
}

with open(f"{OUTPUT_DIR}/weightwatcher_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to {OUTPUT_DIR}/weightwatcher_summary.json")
print(f"Plots saved to {OUTPUT_DIR}/")

# Push to GCS
print("\nPushing to GCS...")
os.system(f"gsutil cp -r {OUTPUT_DIR}/* gs://overlook-data/hackathon/sft_models/weightwatcher_results/")
print("Done.")

print("\n" + "=" * 80)
print("WEIGHTWATCHER ANALYSIS COMPLETE")
print("=" * 80)
print("\nKey outputs:")
print(f"  {OUTPUT_DIR}/alpha_histogram_comparison.png")
print(f"  {OUTPUT_DIR}/correlation_flow_comparison.png")
print(f"  {OUTPUT_DIR}/alpha_vs_alpha.png")
print(f"  {OUTPUT_DIR}/layer_comparison.csv")
print(f"  {OUTPUT_DIR}/weightwatcher_summary.json")
print("\nNext steps:")
print("  - Layers with alpha > 6 after FT: increase LoRA rank or target with full FT")
print("  - Layers with alpha < 2 after FT: reduce LR or add regularization")
print("  - Compare correlation flow to identify structural bottlenecks")
