# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2-LEAN: Simplified Entity Extraction for Association Rules
# MAGIC **Notebook by:** Adam Lang  
# MAGIC **Date:** 2/2/2026
# MAGIC **Purpose:** LEAN entity extraction for alpha model association rules mining
# MAGIC
# MAGIC ## Configuration
# MAGIC - **Instance:** `g5.48xlarge` (8× NVIDIA A10G GPUs, 768GB RAM)
# MAGIC - **Stack:** Ray + Polars + GLiNER-BioMed
# MAGIC - **Model:** Ihor/gliner-biomed-large-v1.0
# MAGIC - **Enhanced Entity Labels:** 13 standard medical labels (see below) -- this will enhance coverage.
# MAGIC - **Sections:** FINDINGS + CONCLUSIONS + RECOMMENDATIONS
# MAGIC
# MAGIC ## LEAN Approach - What We're Skipping:
# MAGIC - x NO reference table building (deferred to March VLM work)
# MAGIC - x NO comprehensive categorization against ontologies
# MAGIC - x NO metadata extraction
# MAGIC - x NO entity vocabulary analysis
# MAGIC - --> ONLY extract entities for association rules mining
# MAGIC
# MAGIC ## Expected Performance
# MAGIC - **Reports:** 66,791
# MAGIC - **Chunks:** 34 chunks (2,000 reports per chunk)
# MAGIC - **Estimated Total time:** ~20-30 minutes
# MAGIC - **Estimated Entities extracted:** ~100K-200K total instances
# MAGIC
# MAGIC ## Next Step
# MAGIC - NB3-LEAN: Association Rules Mining

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Cluster Configuration
# MAGIC **Required:**
# MAGIC - Databricks Runtime: 16.4 LTS ML (GPU)
# MAGIC - Worker Type: `g5.48xlarge` (8× A10G GPUs)
# MAGIC - Workers: 0 (single-node cluster)
# MAGIC - Driver Type: `g5.48xlarge`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install ray[default]==2.50.1 polars==0.20.31 gliner torch --quiet
# MAGIC %pip install deltalake --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

## 1. Verify installations
import ray
import polars as pl
import torch
from gliner import GLiNER

print("   Installations verified:")
print(f"   Ray: {ray.__version__}")
print(f"   Polars: {pl.__version__}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA devices: {torch.cuda.device_count()}")

# Test installations
try:
    ray.init(ignore_reinit_error=True)
    print(" Ray: Working")
    ray.shutdown()
except Exception as e:
    print(f" Ray: {e}")

try:
    assert torch.cuda.is_available()
    print(f" PyTorch + CUDA: Working ({torch.cuda.device_count()} GPUs)")
except Exception as e:
    print(f" PyTorch: {e}")

print("\n All core dependencies working!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

## 2. Configuration
#Tables
SOURCE_TABLE = "llm_sandbox.gamuts.canine_reports_3rad_with_templates_final"
TARGET_TABLE = "llm_sandbox.gamuts.rag_entities_lean_66k"

# Model
MODEL_NAME = "Ihor/gliner-biomed-large-v1.0"

# # Entity labels - SIMPLIFIED for LEAN approach (3 labels only!)
# ENTITY_LABELS = [
#     "finding",    # Clinical observations (opacity, pattern, mass, nodule)
#     "disease",    # Diagnoses (pneumonia, cardiomegaly, CHF)
#     "anatomy",    # Body parts (heart, lung, liver, kidney)
# ]
# Entity labels (13 labels - medical standard compliant)
ENTITY_LABELS = [
    "anatomy",              # Body structures (heart, liver, lungs, kidney)
    "observation",          # Radiographic findings (cardiomegaly, mass, effusion)
    "pattern",              # Tissue patterns (alveolar, bronchial, nodular)
    "measurement",          # Quantitative values (VHS, 4cm, opacity)
    "characteristics",      # Descriptive features (hyperechoic, irregular, diffuse)
    "severity",             # Intensity (mild, moderate, severe)
    "change",               # Temporal (progressive, stable, improved)
    "laterality",           # Left, right, bilateral, unilateral (SNOMED-CT)
    "anatomical_position",  # Cranial, caudal, dorsal, ventral, medial, lateral (RadLex)
    "disease",              # Diagnoses (MVD, pancreatitis, pneumonia)
    "procedure",            # Recommendations (echo, ultrasound, recheck)
    "certainty",            # Confidence (suspect, consistent, diagnostic)
    "negation"              # Negative findings (no evidence, absence)
]




# Processing (proven config from 1.65M report pipeline)
NUM_GPUS = 8  # g5.48xlarge has 8× A10G
BATCH_SIZE = 1024  # Per GPU batch size
CHUNK_SIZE = 2000  # Reports per chunk

# Sections to extract from
SECTIONS = ["history","findings", "conclusions", "recommendations"]

print("="*80)
print("LEAN CONFIGURATION")
print("="*80)
print(f"Source: {SOURCE_TABLE}")
print(f"Target: {TARGET_TABLE}")
print(f"Model: {MODEL_NAME}")
print(f"Entity labels: {ENTITY_LABELS}")
print(f"GPUs: {NUM_GPUS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Sections: {SECTIONS}")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Ray

# COMMAND ----------

## 3. Init Ray
import ray
import torch
import json

# Initialize Ray with all resources
ray.init(
    num_gpus=NUM_GPUS,
    num_cpus=192,  # g5.48xlarge has 192 vCPUs
    object_store_memory=100 * 1024 * 1024 * 1024,  # 100 GB
    ignore_reinit_error=True,
    _system_config={
        "automatic_object_spilling_enabled": True,
        "object_spilling_config": json.dumps({
            "type": "filesystem",
            "params": {"directory_path": "/tmp/spill"}
        })
    }
)

print(f"   Ray initialized")
print(f"   Ray version: {ray.__version__}")
print(f"   GPUs: {ray.cluster_resources().get('GPU', 0)}")
print(f"   CPUs: {ray.cluster_resources().get('CPU', 0)}")
print(f"   Memory: {ray.cluster_resources().get('memory', 0) / 1e9:.1f} GB")

# Verify GPU detection
print(f"\nDetected GPUs:")
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Ray Worker Class

# COMMAND ----------

@ray.remote(num_gpus=1, num_cpus=24)
class A10GWorker:
    """
    Worker for processing reports on A10G GPU.
    Extracts entities from findings, conclusions, and recommendations.
    LEAN version: Simplified entity labels, no categorization.
    """
    
    def __init__(self):
        from gliner import GLiNER
        import torch
        import os
        
        # Get GPU ID from CUDA_VISIBLE_DEVICES (set by Ray)
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        self.gpu_id = int(cuda_visible.split(',')[0])
        
        # Always use cuda:0 from worker's perspective (Ray handles mapping)
        self.device = torch.device('cuda:0')
        
        print(f"[Worker GPU {self.gpu_id}] Initializing...")
        
        # Load model
        self.model = GLiNER.from_pretrained(MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model = self.model.half()  # FP16 for speed
        self.model.eval()
        
        # Try to compile (PyTorch 2.0+ optimization)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print(f"[Worker GPU {self.gpu_id}] Model compiled")
        except Exception:
            print(f"[Worker GPU {self.gpu_id}] Compile not available (OK)")
        
        # Configuration
        self.entity_labels = ENTITY_LABELS
        self.batch_size = BATCH_SIZE
        
        print(f"[Worker GPU {self.gpu_id}] Ready! Batch size: {self.batch_size}")
        print(f"[Worker GPU {self.gpu_id}] Entity labels: {self.entity_labels}")
    
    def process_chunk(self, chunk_data):
        """
        Process a chunk of reports.
        
        Args:
            chunk_data: List of (case_id, findings, conclusions, recommendations, reader)
        
        Returns:
            List of results with extracted entities
        """
        import time
        
        start_time = time.time()
        results = []
        
        # Process each report in chunk
        for case_id, history, findings, conclusions, recommendations, reader in chunk_data:
            report_entities = []
            
            # Extract from each section
            for section_name, section_text in [
                ("history", history),
                ("findings", findings),
                ("conclusions", conclusions),
                ("recommendations", recommendations)
            ]:
                if section_text and len(section_text.strip()) > 0:
                    # Extract entities
                    entities = self.model.predict_entities(
                        section_text,
                        self.entity_labels,
                        threshold=0.5  # Standard threshold
                    )
                    
                    # Format entities with section source
                    for ent in entities:
                        report_entities.append({
                            "text": ent["text"],
                            "label": ent["label"],
                            "score": float(ent["score"]),
                            "section": section_name
                        })
            
            results.append({
                "case_id": case_id,
                "reader": reader,
                "entities": report_entities,
                "num_entities": len(report_entities)
            })
        
        elapsed = time.time() - start_time
        reports_per_sec = len(chunk_data) / elapsed if elapsed > 0 else 0
        
        print(f"[Worker GPU {self.gpu_id}] Processed {len(chunk_data)} reports "
              f"in {elapsed:.1f}s ({reports_per_sec:.1f} reports/sec)")
        
        return results

print(" Worker class defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load Data with Polars

# COMMAND ----------

## 5. Load data with Polars
import polars as pl
from pyspark.sql import functions as F

# Load reports from Spark
spark_df = spark.table(SOURCE_TABLE).select(
    "case_id",
    "history",
    "findings",
    "conclusions", 
    "recommendations",
    "reader"
).filter(
    # Ensure at least one section has content
    (F.col("findings").isNotNull()) | 
    (F.col("conclusions").isNotNull()) | 
    (F.col("recommendations").isNotNull())
)

total_reports = spark_df.count()
print(f" Loaded {total_reports:,} reports from {SOURCE_TABLE}")

# Convert to Polars for fast processing
pdf = spark_df.toPandas()
polars_df = pl.from_pandas(pdf)

print(f"   Converted to Polars DataFrame")
print(f"   Shape: {polars_df.shape}")
print(f"   Memory usage: ~{polars_df.estimated_size() / 1e6:.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Worker Pool and Process Reports

# COMMAND ----------

## 6. Create worker pool and process reports
import numpy as np
import time

# Create worker pool (one worker per GPU)
print(f"Creating {NUM_GPUS} workers...")
workers = [A10GWorker.remote() for _ in range(NUM_GPUS)]
print(f"  {NUM_GPUS} workers created")

# Prepare data for processing
data_tuples = [
    (
        row["case_id"],
        row["history"] if row["history"] else "", 
        row["findings"] if row["findings"] else "",
        row["conclusions"] if row["conclusions"] else "",
        row["recommendations"] if row["recommendations"] else "",
        row["reader"]
    )
    for row in polars_df.iter_rows(named=True)
]

print(f" Prepared {len(data_tuples):,} reports for processing")

# Split into chunks
num_chunks = int(np.ceil(len(data_tuples) / CHUNK_SIZE))
chunks = [
    data_tuples[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
    for i in range(num_chunks)
]

print(f"   Split into {num_chunks} chunks ({CHUNK_SIZE} reports per chunk)")
print(f"   Chunks per GPU: {num_chunks / NUM_GPUS:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Process chunks in parallel

# COMMAND ----------

## Process chunks in parallel
print(f"\n{'='*80}")
print(f"STARTING ENTITY EXTRACTION (LEAN)")
print(f"{'='*80}")
print(f"Reports: {len(data_tuples):,}")
print(f"Chunks: {num_chunks}")
print(f"Workers: {NUM_GPUS}")
print(f"Entity labels: {ENTITY_LABELS}")
print(f"{'='*80}\n")

start_time = time.time()

# Distribute chunks across workers using round-robin
futures = []
for i, chunk in enumerate(chunks):
    worker_idx = i % NUM_GPUS  # Round-robin assignment
    future = workers[worker_idx].process_chunk.remote(chunk)
    futures.append(future)
    
    if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
        print(f"Submitted {i + 1}/{num_chunks} chunks...")

print(f"\n All chunks submitted, waiting for results...")

# Collect results
all_results = []
for i, future in enumerate(futures):
    chunk_results = ray.get(future)
    all_results.extend(chunk_results)
    
    if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
        elapsed = time.time() - start_time
        progress = (i + 1) / num_chunks * 100
        eta = elapsed / (i + 1) * num_chunks - elapsed
        print(f"Progress: {i + 1}/{num_chunks} chunks ({progress:.1f}%) | "
              f"Elapsed: {elapsed / 60:.1f}m | ETA: {eta / 60:.1f}m")

elapsed_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"EXTRACTION COMPLETE!")
print(f"{'='*80}")
print(f"Time: {elapsed_time / 60:.2f} minutes")
print(f"Reports: {len(all_results):,}")
print(f"Throughput: {len(all_results) / (elapsed_time / 60):.1f} reports/min")
print(f"{'='*80}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Convert Results to Spark DataFrame and Save

# COMMAND ----------

## 7. Convert results to Polars first (faster)
results_polars = pl.DataFrame(all_results)

print(f"   Results converted to Polars")
print(f"   Shape: {results_polars.shape}")

# Convert to Pandas then Spark
results_pandas = results_polars.to_pandas()
results_spark = spark.createDataFrame(results_pandas)

print(f" Results converted to Spark DataFrame")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save to Delta Table

# COMMAND ----------

# Save to Delta table
results_spark.write.mode("overwrite").saveAsTable(TARGET_TABLE)

print(f" Results saved to {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ZORDER Table Optimization

# COMMAND ----------

# Optimize table
spark.sql(f"OPTIMIZE {TARGET_TABLE} ZORDER BY (case_id)")
print(f" Table optimized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Basic Validation & Statistics

# COMMAND ----------

## 8. Load results for validation
entities_df = spark.table(TARGET_TABLE)

print("="*80)
print("LEAN ENTITY EXTRACTION - VALIDATION")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overall Stats

# COMMAND ----------

# Overall statistics
from pyspark.sql.functions import explode, col, size

total_reports = entities_df.count()
total_entities_sum = entities_df.agg(F.sum("num_entities")).collect()[0][0]

print(f"\n Reports processed: {total_reports:,}")
print(f"  Total entity instances: {total_entities_sum:,}")
print(f"  Average entities per report: {total_entities_sum / total_reports:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explode entities for analysis

# COMMAND ----------

### Explode entities for analysis
entities_exploded = entities_df.select(
    col("case_id"),
    col("reader"),
    explode(col("entities")).alias("entity")
).select(
    col("case_id"),
    col("reader"),
    col("entity.text").alias("entity_text"),
    col("entity.label").alias("label"),
    col("entity.score").alias("score"),
    col("entity.section").alias("section")
)

print(f" Entities exploded for analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entity label distribution

# COMMAND ----------

# Entity label distribution
print("\n Entity Label Distribution:")
display(
    entities_exploded.groupBy("label")
    .count()
    .withColumn("percentage", F.round(col("count") / total_entities_sum * 100, 1))
    .orderBy("count", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Section Distribution

# COMMAND ----------

# Section distribution
print("\n Section Distribution:")
display(
    entities_exploded.groupBy("section")
    .count()
    .withColumn("percentage", F.round(col("count") / total_entities_sum * 100, 1))
    .orderBy("count", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Radiologist Distribution

# COMMAND ----------

# Radiologist distribution
print("\n Entities by Radiologist:")
display(
    entities_exploded.groupBy("reader")
    .agg(
        F.count("*").alias("total_entities"),
        F.countDistinct("entity_text").alias("unique_entities")
    )
    .orderBy("total_entities", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top 20 most frequent entities

# COMMAND ----------

# Top 20 most frequent entities
print("\n Top 20 Most Frequent Entities:")
display(
    entities_exploded.groupBy("entity_text", "label")
    .count()
    .orderBy("count", ascending=False)
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample entities for quality check

# COMMAND ----------

# Sample entities for quality check
print("\n Sample Entities (Quality Check):")
display(
    entities_exploded
    .filter(col("score") >= 0.7)  # High confidence only
    .sample(fraction=0.001)
    .limit(50)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Cleanup

# COMMAND ----------

## 9. Shutdown Ray to free resources
ray.shutdown()
print(" Ray shutdown")

# Clear Spark cache
spark.catalog.clearCache()
print(" Cache cleared")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary & Next Steps
# MAGIC
# MAGIC ** LEAN ENTITY EXTRACTION COMPLETE!**
# MAGIC
# MAGIC **Tables created:**
# MAGIC - `llm_sandbox.gamuts.rag_entities_lean_66k` (entity extraction results)
# MAGIC
# MAGIC **Performance:**
# MAGIC - Reports processed: 66,791
# MAGIC - Total time: ~20-30 minutes
# MAGIC - Throughput: ~3,000+ reports/min
# MAGIC
# MAGIC **Entity Labels Used (LEAN):**
# MAGIC - `finding` (clinical observations)
# MAGIC - `disease` (diagnoses)
# MAGIC - `anatomy` (body parts)
# MAGIC
# MAGIC **Section Tracking:**
# MAGIC -  Findings
# MAGIC -  Conclusions
# MAGIC -  Recommendations
# MAGIC
# MAGIC **Next Steps:**
# MAGIC 1. **Tuesday (Feb 3):** Run NB3-LEAN for association rules mining
# MAGIC 2. **Wednesday (Feb 4):** Package deliverables for alpha model A/B testing
# MAGIC 3. **March:** Resume comprehensive EDA for VLM work (see JIRA story)
# MAGIC
# MAGIC **Deferred to March (VLM Work):**
# MAGIC - Comprehensive entity vocabulary (2K-5K entities)
# MAGIC - Reference table categorization (CLIP, DICOM, ontologies)
# MAGIC - Metadata extraction (measurements, severity, spatial)
# MAGIC - Knowledge graph construction
# MAGIC - Normal vs abnormal pattern analysis

# COMMAND ----------

print("\n" + "="*80)
print(" NB2-LEAN COMPLETE - ENTITY EXTRACTION DONE")
print("="*80)
print(f"\nReady for NB3-LEAN: Association Rules Mining (Tuesday)")
print(f"Total entities extracted: {total_entities_sum:,}")
print(f"Unique entities: {entities_exploded.select('entity_text').distinct().count():,}")
