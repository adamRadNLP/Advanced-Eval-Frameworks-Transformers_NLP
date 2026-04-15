# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1: NER Pipeline - g5.48xlarge (8× A10G GPUs)
# MAGIC * Notebook by Adam Lang
# MAGIC * Date: 10/27/2025
# MAGIC
# MAGIC # Configuration
# MAGIC * **Instance**: `g5.48xlarge` (8× NVIDIA A10G, 768GB RAM)
# MAGIC * **Stack**: Ray + Polars + GLiNER-BioMed
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook processes 1.7M veterinary radiology reports to extract medical entities using:
# MAGIC - **Ray**: Distributed computing (better than Spark for ML)
# MAGIC - **Polars**: Fast data loading/processing (10-100× faster than Pandas)
# MAGIC - **GLiNER-BioMed**: Biomedical entity extraction (no API costs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Cluster Configuration
# MAGIC * I created cluster manually in the UI. But if you want to try doing it with the config below here it is.
# MAGIC
# MAGIC **Create cluster with these settings:**
# MAGIC ```
# MAGIC Cluster Name: <name your cluster>
# MAGIC Databricks Runtime: 16.4 LTS ML (GPU)
# MAGIC Worker Type: `g5.48xlarge` (8× A10G GPUs)
# MAGIC Workers: 0 (Python single-node cluster)
# MAGIC Driver Type: g5.48xlarge
# MAGIC ```
# MAGIC
# MAGIC **OR use this JSON config below to create:**

# COMMAND ----------

# # Run this to create cluster via API (optional)
# cluster_config = {
#     "cluster_name": "ner-g5-48xlarge",
#     "spark_version": "14.3.x-gpu-ml-scala2.12",
#     "aws_attributes": {
#         "availability": "ON_DEMAND",
#         "zone_id": "us-east-1a",
#         "instance_profile_arn": "arn:aws:iam::YOUR_ACCOUNT:instance-profile/YOUR_PROFILE",
#         "ebs_volume_type": "GENERAL_PURPOSE_SSD",
#         "ebs_volume_count": 1,
#         "ebs_volume_size": 100
#     },
#     "node_type_id": "g5.48xlarge",
#     "driver_node_type_id": "g5.48xlarge",
#     "num_workers": 0,  # Single-node cluster
#     "autotermination_minutes": 120,
#     "spark_conf": {
#         "spark.master": "local[*]",
#         "spark.databricks.delta.preview.enabled": "true"
#     }
# }

# Uncomment to create:
# import requests
# DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl")
# DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# response = requests.post(
#     f"https://{DATABRICKS_HOST}/api/2.0/clusters/create",
#     headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
#     json=cluster_config
# )
# print(response.json())

# print("   Cluster configuration ready")
# print("   Create cluster manually in UI or use API above")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install ray[default]==2.50.1 polars==0.20.31 gliner torch --quiet
# MAGIC %pip install deltalake --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Versions

# COMMAND ----------

# Verify versions
import ray
import polars as pl
import torch
from gliner import GLiNER

print("    Installations verified:")
print(f"   Ray: {ray.__version__}")
print(f"   Polars: {pl.__version__}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA devices: {torch.cuda.device_count()}")

# Test that everything works despite the warning
import ray
import torch
from gliner import GLiNER
import polars as pl

print("Testing installations...")

# Test Ray
try:
    ray.init(ignore_reinit_error=True)
    print(" Ray: Working")
    ray.shutdown()
except Exception as e:
    print(f" Ray: {e}")

# Test PyTorch + CUDA
try:
    assert torch.cuda.is_available()
    print(f" PyTorch + CUDA: Working ({torch.cuda.device_count()} GPUs)")
except Exception as e:
    print(f" PyTorch: {e}")

# Test GLiNER (quick load test)
try:
    # Just verify it can import, don't actually load model yet
    print(" GLiNER: Importable")
except Exception as e:
    print(f" GLiNER: {e}")

# Test Polars
try:
    test_df = pl.DataFrame({"a": [1, 2, 3]})
    assert len(test_df) == 3
    print(" Polars: Working")
except Exception as e:
    print(f" Polars: {e}")

print("\n All core dependencies working!")
print("   The protobuf warning can be safely ignored.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# Tables
SOURCE_TABLE = "llm_sandbox.gamuts.canine_radiographs_metadata_enriched_v1"
TARGET_TABLE = "llm_sandbox.gamuts.rag_entities_g5_final_v1"

# Model
MODEL_NAME = "Ihor/gliner-biomed-large-v1.0"
ENTITY_TYPES = ["anatomy", "finding", "disease"]

# Processing
NUM_GPUS = 8  # g5.48xlarge has 8× A10G
BATCH_SIZE = 1024  # Per GPU batch size
CHUNK_SIZE = 2000  # Reports per chunk submitted to workers

print("Configuration:")
print(f"  Source: {SOURCE_TABLE}")
print(f"  Target: {TARGET_TABLE}")
print(f"  Model: {MODEL_NAME}")
print(f"  GPUs: {NUM_GPUS}")
print(f"  Batch size: {BATCH_SIZE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Ray

# COMMAND ----------

## 3. Initialize Ray
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
# MAGIC ## 4. Define Worker Class

# COMMAND ----------

## 4. Define Worker Class (FIXED)
@ray.remote(num_gpus=1, num_cpus=24)
class A10GWorker:
    """
    Worker for processing reports on A10G GPU.
    Let Ray auto-assign GPU IDs instead of manual assignment.
    """
    
    def __init__(self):  # REMOVED gpu_id parameter
        from gliner import GLiNER
        import torch
        import os
        
        # Get GPU ID from CUDA_VISIBLE_DEVICES (set by Ray)
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        self.gpu_id = int(cuda_visible.split(',')[0])  # Ray sets this
        
        # Always use cuda:0 from worker's perspective (Ray handles mapping)
        self.device = torch.device('cuda:0')  # ← ALWAYS cuda:0!
        
        print(f"[Worker] Initializing on GPU {self.gpu_id} "
              f"(PyTorch sees: {torch.cuda.get_device_name(0)})...")
        
        # Load model
        self.model = GLiNER.from_pretrained(MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model = self.model.half()  # FP16
        self.model.eval()
        
        # Try to compile
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print(f"[Worker GPU {self.gpu_id}] Model compiled")
        except Exception as e:
            print(f"[Worker GPU {self.gpu_id}] Compile not available")
        
        # Configuration
        self.entity_types = ENTITY_TYPES
        self.batch_size = BATCH_SIZE
        
        # Critical findings
        self.critical_findings = self._get_critical_findings()
        
        print(f"[Worker GPU {self.gpu_id}] Ready! Batch size: {self.batch_size}")
    
    def _get_critical_findings(self):
        """Return set of critical finding terms"""
        findings = {
            "thorax": [
                "pneumothorax", "pleural effusion", "pulmonary nodule", "lung mass",
                "pulmonary metastases", "congestive heart failure", "CHF", "cardiogenic edema",
                "pulmonary thromboembolism", "PTE", "pericardial effusion", "heart-based mass",
                "fungal pneumonia", "aspiration pneumonia", "noncardiogenic edema"
            ],
            "abdomen": [
                "foreign body", "gastric foreign body", "intestinal foreign body",
                "gastric dilatation-volvulus", "GDV", "hemoabdomen", "free abdominal fluid",
                "urinary bladder rupture", "intestinal obstruction", "mesenteric volvulus",
                "splenic torsion", "renal lymphoma", "pancreatic mass", "pancreatitis"
            ],
            "msk": [
                "fracture", "aggressive bone lesion", "osteosarcoma", "bone lysis",
                "pathologic fracture", "joint luxation", "spinal fracture"
            ],
            "pelvis": ["pelvic fracture", "hip luxation", "urinary bladder rupture"],
            "spine": ["spinal fracture", "vertebral luxation", "disk extrusion", "ischemic myelopathy"],
            "brain": ["thalamic astrocytoma"]
        }
        return {term.lower() for terms in findings.values() for term in terms}
    
    def process_batch(self, batch_data):
        """
        Process batch of reports.
        Args:
            batch_data: List of {'case_id': str, 'text': str}
        Returns:
            List of results with entities
        """
        import torch
        import time
        
        start = time.time()
        
        case_ids = [item['case_id'] for item in batch_data]
        texts = [item['text'] for item in batch_data]
        
        results = []
        
        # Process in sub-batches
        for i in range(0, len(texts), self.batch_size):
            sub_texts = texts[i:i + self.batch_size]
            sub_case_ids = case_ids[i:i + self.batch_size]
            
            # Run inference
            with torch.no_grad():
                predictions = self.model.run(
                    sub_texts,
                    labels=self.entity_types,
                    threshold=0.25,
                    flat_ner=True
                )
            
            # Post-process each report
            for case_id, preds, text in zip(sub_case_ids, predictions, sub_texts):
                entities = []
                
                for entity in preds:
                    entity_text = entity['text']
                    entity_lower = entity_text.lower()
                    
                    # Assertion detection
                    assertion = "present"
                    text_lower = text.lower()
                    pos = text_lower.find(entity_lower)
                    
                    if pos > 0:
                        before = text_lower[max(0, pos-100):pos]
                        if any(neg in before for neg in ['no ', 'without ', 'absence of', 'ruled out', 'normal ', 'unremarkable']):
                            assertion = "absent"
                        elif any(poss in before for poss in ['possible', 'suspected', 'likely', 'questionable', 'may represent', 'consider']):
                            assertion = "possible"
                        elif any(past in before for past in ['previous', 'prior', 'history of', 'old', 'chronic']):
                            assertion = "past"
                    
                    # Critical finding check
                    is_critical = any(crit in entity_lower for crit in self.critical_findings)
                    
                    # Section detection
                    section = "findings"
                    if "history:" in text_lower[:pos] if pos > 0 else False:
                        section = "history"
                    elif "conclusions:" in text_lower[:pos] if pos > 0 else False:
                        section = "conclusions"
                    elif "recommendations:" in text_lower[:pos] if pos > 0 else False:
                        section = "recommendations"
                    
                    entities.append({
                        'text': entity_text,
                        'category': entity['label'].upper(),
                        'assertion': assertion,
                        'section': section,
                        'is_critical': is_critical,
                        'score': float(entity['score'])
                    })
                
                results.append({
                    'case_id': str(case_id),
                    'entities': entities,
                    'num_entities': len(entities),
                    'error': None ## NOTE: this creates 'void' value -> if all reports succeed, then all values will be of type 'None'. Delta Lake creates a void type when all values are null. This can cause display issues. in future just omit error field if not needed for tracking. 

                })
        
        elapsed = time.time() - start
        rate = len(batch_data) / elapsed * 60
        print(f"[Worker GPU {self.gpu_id}] Processed {len(batch_data)} reports in {elapsed:.1f}s ({rate:.0f} reports/min)")
        
        return results
    
    def get_memory_stats(self):
        """Return GPU memory statistics"""
        import torch
        return {
            'gpu_id': self.gpu_id,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(self.device) / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1e9
        }

print(" Worker class defined (GPU auto-assignment)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test on Small Sample (800 reports)

# COMMAND ----------

## 5. test on sample 800 reports
import polars as pl
import time

print("="*80)
print("TEST: 8 GPUs × 100 reports = 800 reports")
print("="*80)

# Load test data
start = time.time()
spark_df = spark.table(SOURCE_TABLE) \
    .filter("findings_length > 50") \
    .limit(800) \
    .select("case_id", "history_clean", "findings_clean", 
            "conclusions_clean", "recommendations_clean")

test_df = pl.from_pandas(spark_df.toPandas())
load_time = time.time() - start
print(f" Loaded 800 reports in {load_time:.2f}s")

# Prepare text
test_df = test_df.with_columns([
    pl.col("case_id").cast(pl.Utf8),
    pl.concat_str([
        pl.lit("History: "), pl.col("history_clean").fill_null(""),
        pl.lit("\nFindings: "), pl.col("findings_clean").fill_null(""),
        pl.lit("\nConclusions: "), pl.col("conclusions_clean").fill_null(""),
        pl.lit("\nRecommendations: "), pl.col("recommendations_clean").fill_null("")
    ]).str.slice(0, 4000).alias("text")
])

# Create workers - NO gpu_id parameter!
print(f"\nCreating {NUM_GPUS} workers...")
workers = [A10GWorker.remote() for _ in range(NUM_GPUS)]  # ← FIXED!
print(f" Workers created")

# chunk data
data = test_df.select(['case_id', 'text']).to_dicts()
chunk_size = len(data) // NUM_GPUS
chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(NUM_GPUS)]

print(f"Split into {NUM_GPUS} chunks of ~{chunk_size} reports")

print("\nProcessing...")
start = time.time()
futures = [worker.process_batch.remote(chunk) for worker, chunk in zip(workers, chunks)]
results = ray.get(futures)
elapsed = time.time() - start

all_results = [item for batch in results for item in batch]
total_entities = sum(r['num_entities'] for r in all_results)

print(f"\n{'='*80}")
print("TEST RESULTS")
print(f"{'='*80}")
print(f"  Total time: {elapsed:.1f}s")
print(f"  Throughput: {48000/elapsed:.0f} reports/min")
print(f"  Total entities: {total_entities:,}")
print(f"  Avg entities: {total_entities/800:.1f}")

projected_hours = (1.7e6 * elapsed / 800) / 3600
print(f"\n Projected time for 1.7M: {projected_hours:.1f} hours")
print(f"   Projected cost: ${16 * projected_hours:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5b. Pre-warm workers 

# COMMAND ----------

## 5B. Pre-warm Workers (Optional but Recommended)
print("Pre-warming workers with dummy batch...")
import time

# Send small dummy batch to all workers to trigger model loading
dummy_data = [{'case_id': 'test', 'text': 'test report'}] * 10

start = time.time()
warmup_futures = [worker.process_batch.remote(dummy_data) for worker in workers]
ray.get(warmup_futures)
warmup_time = time.time() - start

print(f"  Workers pre-warmed in {warmup_time:.1f}s")
print("   Models loaded on all GPUs")
print("   Ready for production run!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Production Run - Full 1.7M Reports

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a. Test Production Run - 10K Reports

# COMMAND ----------

## 6a. TEST Production Run - 10K Reports
from ray.util import ActorPool
from datetime import datetime, timedelta
import time

print("="*80)
print("TEST PRODUCTION: 10K REPORTS")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load 10K reports with SPARK
print("\nLoading data with Spark...")
start = time.time()

spark_df = spark.table(SOURCE_TABLE) \
    .filter("findings_length > 50") \
    .limit(10000) \
    .select("case_id", "history_clean", "findings_clean", 
            "conclusions_clean", "recommendations_clean")

# Convert to Polars
full_df = pl.from_pandas(spark_df.toPandas())

load_time = time.time() - start
total_reports = len(full_df)

print(f" Loaded {total_reports:,} reports in {load_time:.1f}s")

# Prepare text with Polars
print("Preparing text...")
start = time.time()

full_df = full_df.with_columns([
    pl.col("case_id").cast(pl.Utf8),
    pl.concat_str([
        pl.lit("History: "), pl.col("history_clean").fill_null(""),
        pl.lit("\nFindings: "), pl.col("findings_clean").fill_null(""),
        pl.lit("\nConclusions: "), pl.col("conclusions_clean").fill_null(""),
        pl.lit("\nRecommendations: "), pl.col("recommendations_clean").fill_null("")
    ]).str.slice(0, 4000).alias("text")
])

prep_time = time.time() - start
print(f" Text preparation complete in {prep_time:.1f}s")

# Convert to list of dicts
data = full_df.select(['case_id', 'text']).to_dicts()

# Split into chunks for processing
chunks = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
print(f"\nSplit into {len(chunks)} chunks of ~{CHUNK_SIZE} reports")

# Create worker pool (reuse pre-warmed workers!)
pool = ActorPool(workers)

# Process with progress tracking
print(f"\nProcessing {total_reports:,} reports...")
all_results = []
start_time = time.time()

for i, chunk_results in enumerate(pool.map(lambda w, c: w.process_batch.remote(c), chunks)):
    all_results.extend(chunk_results)
    
    # Progress update
    elapsed = time.time() - start_time
    processed = len(all_results)
    rate = processed / elapsed * 60 if elapsed > 0 else 0
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"Progress: {processed:,}/{total_reports:,} ({100*processed/total_reports:.1f}%) | "
          f"{rate:.0f} reports/min | "
          f"Chunks: {i+1}/{len(chunks)}")

total_time = time.time() - start_time
throughput = total_reports / total_time * 60

print(f"\n{'='*80}")
print("PROCESSING COMPLETE!")
print(f"{'='*80}")
print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
print(f"  Throughput: {throughput:.0f} reports/min")
print(f"  Reports processed: {len(all_results):,}")
print(f"  Total entities: {sum(r['num_entities'] for r in all_results):,}")
print(f"  Avg entities/report: {sum(r['num_entities'] for r in all_results) / len(all_results):.1f}")
print(f"  Errors: {sum(1 for r in all_results if r.get('error'))}")

# Calculate projections for full 1.7M
full_time_hours = (1.7e6 / throughput) / 60
full_cost = 16 * full_time_hours

print(f"\n{'='*80}")
print("PROJECTIONS FOR FULL 1.7M DATASET")
print(f"{'='*80}")
print(f"  Estimated time: {full_time_hours:.1f} hours ({full_time_hours*60:.0f} minutes)")
print(f"  Estimated cost: ${full_cost:.2f}")
print(f"  Estimated completion: {datetime.now() + timedelta(hours=full_time_hours)}")

# Save test results
print(f"\nSaving test results...")
start = time.time()

# Convert to Polars DataFrame
results_df = pl.DataFrame(all_results)
print(f"Results shape: {results_df.shape}")

# Write to test table
test_table = TARGET_TABLE + "_test_10k"
spark_results = spark.createDataFrame(results_df.to_pandas())

spark_results.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(test_table)

save_time = time.time() - start
print(f" Saved {len(results_df):,} results in {save_time:.1f}s")
print(f" Test table created: {test_table}")

print(f"\n{'='*80}")
print("TEST COMPLETE - READY FOR FULL RUN")
print(f"{'='*80}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validation after 10K

# COMMAND ----------

# Quick check
display(spark.sql(f"SELECT COUNT(*), AVG(num_entities) FROM {TARGET_TABLE}_test_10k").show())

# Sample entities
sample_df = spark.sql(f"""
    SELECT case_id, slice(entities, 1, 5) as sample_entities 
    FROM {TARGET_TABLE}_test_10k 
    LIMIT 3
""")
sample_pd = sample_df.toPandas()
sample_pd.head()

# COMMAND ----------

for i in sample_pd["sample_entities"]: 
    print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. FULL Production Run - 1.7M Reports
# MAGIC

# COMMAND ----------

## 6b. FULL Production Run - 1.7M Reports
from ray.util import ActorPool
from datetime import datetime, timedelta
import time
import polars as pl

print("="*80)
print("FULL PRODUCTION: 1.7M REPORTS")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Expected completion: ~6 hours")
print(f"Expected cost: ~$95")

# Load ALL data
print("\nLoading data with Spark...")
start = time.time()

spark_df = spark.table(SOURCE_TABLE) \
    .filter("findings_length > 50") \
    .select("case_id", "history_clean", "findings_clean", 
            "conclusions_clean", "recommendations_clean")

full_df = pl.from_pandas(spark_df.toPandas())

load_time = time.time() - start
total_reports = len(full_df)

print(f" Loaded {total_reports:,} reports in {load_time:.1f}s")

# Prepare text
print("Preparing text...")
start = time.time()

full_df = full_df.with_columns([
    pl.col("case_id").cast(pl.Utf8),
    pl.concat_str([
        pl.lit("History: "), pl.col("history_clean").fill_null(""),
        pl.lit("\nFindings: "), pl.col("findings_clean").fill_null(""),
        pl.lit("\nConclusions: "), pl.col("conclusions_clean").fill_null(""),
        pl.lit("\nRecommendations: "), pl.col("recommendations_clean").fill_null("")
    ]).str.slice(0, 4000).alias("text")
])

prep_time = time.time() - start
print(f" Text preparation complete in {prep_time:.1f}s")

# Convert to list
data = full_df.select(['case_id', 'text']).to_dicts()
chunks = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]

print(f"\nProcessing Configuration:")
print(f"  Total reports: {total_reports:,}")
print(f"  Chunks: {len(chunks)}")
print(f"  Chunk size: {CHUNK_SIZE}")
print(f"  GPUs: {NUM_GPUS}")
print(f"  Batch size per GPU: {BATCH_SIZE}")

# Process
pool = ActorPool(workers)
print(f"\n{'='*80}")
print("PROCESSING STARTED")
print(f"{'='*80}")
print("Monitor progress below. This will take ~6 hours.")
print("You can check back periodically or let it run overnight.\n")

all_results = []
start_time = time.time()
last_update = start_time

for i, chunk_results in enumerate(pool.map(lambda w, c: w.process_batch.remote(c), chunks)):
    all_results.extend(chunk_results)
    
    # Progress every 10 chunks or 60 seconds
    current_time = time.time()
    if (i + 1) % 10 == 0 or (current_time - last_update) > 60:
        elapsed = current_time - start_time
        processed = len(all_results)
        rate = processed / elapsed * 60
        remaining = total_reports - processed
        eta_hours = (remaining / rate) / 60 if rate > 0 else 0
        eta_time = datetime.now() + timedelta(hours=eta_hours)
        pct_complete = 100 * processed / total_reports
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"{processed:,}/{total_reports:,} ({pct_complete:.1f}%) | "
              f"{rate:.0f}/min | "
              f"ETA: {eta_hours:.1f}h ({eta_time.strftime('%m/%d %H:%M')}) | "
              f"Chunk {i+1}/{len(chunks)}")
        
        last_update = current_time

total_time = time.time() - start_time
throughput = total_reports / total_time * 60
cost = 16 * (total_time / 3600)

print(f"\n{'='*80}")
print(" PROCESSING COMPLETE!")
print(f"{'='*80}")
print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Total time: {total_time/3600:.2f} hours ({total_time/60:.0f} minutes)")
print(f"  Throughput: {throughput:.0f} reports/min")
print(f"  Actual cost: ${cost:.2f}")
print(f"  Reports processed: {len(all_results):,}")
print(f"  Total entities: {sum(r['num_entities'] for r in all_results):,}")
print(f"  Avg entities/report: {sum(r['num_entities'] for r in all_results) / len(all_results):.1f}")
print(f"  Errors: {sum(1 for r in all_results if r.get('error'))}")

### Save results to TARGET_TABLE
print(f"\nSaving results to {TARGET_TABLE}...")
start = time.time()

results_df = pl.DataFrame(all_results)
print(f"  Results dataframe: {results_df.shape}")

spark_results = spark.createDataFrame(results_df.to_pandas())

spark_results.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TARGET_TABLE)

save_time = time.time() - start
print(f" Saved in {save_time:.1f}s")

# Optimize table
print("Optimizing Delta table...")
spark.sql(f"OPTIMIZE {TARGET_TABLE} ZORDER BY (case_id)")
print(" Table optimized")

print(f"\n{'='*80}")
print(" PASS 1 COMPLETE!")
print(f"{'='*80}")
print(f"Results table: {TARGET_TABLE}")
print(f"\nNext steps:")
print(f"  1. Validate results: SELECT * FROM {TARGET_TABLE} LIMIT 10")
print(f"  2. Design Pass 2: LLM classification + RadGraph")
print(f"\nQuery example:")
print(f"  SELECT case_id, num_entities, entities[0:3] FROM {TARGET_TABLE} LIMIT 5")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Table Validation checks

# COMMAND ----------

# Check if table exists
spark.sql("SHOW TABLES IN llm_sandbox.gamuts LIKE '*rag_entities*'").show(truncate=False)

# COMMAND ----------

# Try to query the table
try:
    result = spark.sql("SELECT COUNT(*) as cnt FROM llm_sandbox.gamuts.rag_entities_g5_final_v1").collect()[0]['cnt']
    print(f" Table found with {result:,} rows!")
except Exception as e:
    print(f" Error: {e}")

# COMMAND ----------

# If table exists, show sample
spark.sql("""
    SELECT 
        case_id, 
        num_entities,
        entities[0] as first_entity
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1 
    LIMIT 5
""").show(truncate=False)

# COMMAND ----------

# Refresh catalog
spark.sql("REFRESH TABLE llm_sandbox.gamuts.rag_entities_g5_final_v1")

# Or
spark.catalog.refreshTable("llm_sandbox.gamuts.rag_entities_g5_final_v1")

# Check again
spark.sql("SELECT COUNT(*) FROM llm_sandbox.gamuts.rag_entities_g5_final_v1").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Validate Results
# MAGIC * Processed 1.65M radiology reports
# MAGIC * Extracted 51.6M medical entities
# MAGIC * Completed in 3.3 hours (vs estimated 175 hours)
# MAGIC * Cost $53 (vs estimated $11,100)
# MAGIC * 208× cost reduction
# MAGIC * 53× speed improvement
# MAGIC
# MAGIC

# COMMAND ----------

## Pass 1 Validation --> Overall statistics
spark.sql("""
    SELECT 
        COUNT(*) as total_reports,
        SUM(num_entities) as total_entities,
        AVG(num_entities) as avg_entities,
        MIN(num_entities) as min_entities,
        MAX(num_entities) as max_entities,
        PERCENTILE(num_entities, 0.5) as median_entities
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
""").show()

# COMMAND ----------

## Pass 2
# Entity category distribution
spark.sql("""
    SELECT 
        entity.category,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    LATERAL VIEW explode(entities) as entity
    GROUP BY entity.category
    ORDER BY count DESC
""").show()

# COMMAND ----------

# Assertion distribution
spark.sql("""
    SELECT 
        entity.assertion,
        COUNT(*) as count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    LATERAL VIEW explode(entities) as entity
    GROUP BY entity.assertion
    ORDER BY count DESC
""").show()

# COMMAND ----------

# Critical findings
spark.sql("""
    SELECT 
        COUNT(DISTINCT case_id) as reports_with_critical,
        ROUND(100.0 * COUNT(DISTINCT case_id) / (SELECT COUNT(*) FROM llm_sandbox.gamuts.rag_entities_g5_final_v1), 1) as pct_reports
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    LATERAL VIEW explode(entities) as entity
    WHERE entity.is_critical = true
""").show()

# COMMAND ----------

# Top entities
spark.sql("""
    SELECT 
        entity.text,
        entity.category,
        COUNT(*) as frequency
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    LATERAL VIEW explode(entities) as entity
    GROUP BY entity.text, entity.category
    ORDER BY frequency DESC
    LIMIT 20
""").show(truncate=False)

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE llm_sandbox.gamuts.rag_entities_g5_final_v1

# COMMAND ----------

# Sample reports with entities
spark.sql("""
    SELECT 
        case_id,
        num_entities,
        transform(
            slice(entities, 1, 5),
            e -> concat(e.text, ' (', e.category, ', ', e.assertion, ')')
        ) as sample_entities
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    ORDER BY RAND()
    LIMIT 10
""").show(truncate=False)

# COMMAND ----------

# Check critical findings breakdown
spark.sql("""
    SELECT 
        entity.text,
        COUNT(*) as frequency,
        COUNT(DISTINCT case_id) as unique_reports
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    LATERAL VIEW explode(entities) as entity
    WHERE entity.is_critical = true
    GROUP BY entity.text
    ORDER BY frequency DESC
    LIMIT 30
""").show(truncate=False)

# COMMAND ----------

# Find reports with 0 entities
spark.sql("""
    SELECT case_id, num_entities
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    WHERE num_entities = 0
    LIMIT 10
""").show()

# Count them
spark.sql("""
    SELECT COUNT(*) as zero_entity_reports
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    WHERE num_entities = 0
""").show()

# COMMAND ----------

# Look at the zero-entity report
spark.sql("""
    SELECT *
    FROM llm_sandbox.gamuts.canine_radiographs_metadata_enriched_v1
    WHERE case_id = '6078190'
""").show(truncate=False)

# Check if it has text
spark.sql("""
    SELECT 
        case_id,
        length(history_clean) as history_len,
        length(findings_clean) as findings_len,
        length(conclusions_clean) as conclusions_len,
        length(recommendations_clean) as recommendations_len,
        findings_clean
    FROM llm_sandbox.gamuts.canine_radiographs_metadata_enriched_v1
    WHERE case_id = '6078190'
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7b. Validate GPU usage

# COMMAND ----------

## 7b. Validate GPU usage -- Check Ray worker logs for GPU usage
import ray

# Get Ray cluster info
ray.init(ignore_reinit_error=True)

# Check resources
resources = ray.cluster_resources()
print("Ray Cluster Resources:")
print(f"  GPUs: {resources.get('GPU', 0)}")
print(f"  CPUs: {resources.get('CPU', 0)}")
print(f"  Memory: {resources.get('memory', 0) / 1e9:.1f} GB")

# Check worker nodes
nodes = ray.nodes()
print(f"\nRay Nodes: {len(nodes)}")
for i, node in enumerate(nodes):
    print(f"\n  Node {i}:")
    print(f"    Alive: {node['Alive']}")
    print(f"    Resources: {node['Resources']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7c. Drop Error column -- no errors
# MAGIC * NOTE: I had to drop the "error" column because there were no errors and it was causing an issue with table display in databricks. It also had all null values and was not needed so I dropped it. 

# COMMAND ----------

# 7c. Clean solution: Drop error column permanently
spark.sql("""
    CREATE OR REPLACE TABLE llm_sandbox.gamuts.rag_entities_g5_final_v1 AS
    SELECT 
        case_id,
        entities,
        num_entities
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
""")

# Optimize after recreation
spark.sql("OPTIMIZE llm_sandbox.gamuts.rag_entities_g5_final_v1 ZORDER BY (case_id)")

print(" Table cleaned and optimized")

# Verify
print("\nNew schema:")
spark.sql("DESCRIBE llm_sandbox.gamuts.rag_entities_g5_final_v1").show()

print("\nRow count check:")
spark.sql("SELECT COUNT(*) FROM llm_sandbox.gamuts.rag_entities_g5_final_v1").show()

# COMMAND ----------

## test queries after error col drop 
spark.sql("""
    SELECT 
        case_id,
        num_entities,
        entities[0] as first_entity
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
    LIMIT 5
""").show(truncate=False)

# Test aggregations
spark.sql("""
    SELECT 
        COUNT(*) as reports,
        AVG(num_entities) as avg_entities,
        SUM(num_entities) as total_entities
    FROM llm_sandbox.gamuts.rag_entities_g5_final_v1
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Cleanup
# MAGIC * Why shut down:
# MAGIC     - Free up GPU memory (8× A10Gs are expensive!)
# MAGIC     - Allow cluster auto-termination
# MAGIC     - Clean slate for Pass 2
# MAGIC * We can always restart Ray later if needed!

# COMMAND ----------

# Shutdown Ray to free resources
ray.shutdown()
print("Ray shutdown")

# Clear Spark cache
spark.catalog.clearCache()
print("Cache cleared")

print("\n" + "="*80)
print("PASS 1 COMPLETE!")
print("="*80)
#print(f"Results: {TARGET_TABLE}")
print(f"Entities: 51.6M extracted")
print(f"Ready for Pass 2!")
