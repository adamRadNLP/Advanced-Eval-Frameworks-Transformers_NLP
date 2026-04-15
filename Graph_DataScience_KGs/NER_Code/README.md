# NER Code
- Named Entity Recognition (NER) code.
- Various approaches to NER models. 

---
## Details on Each File

### 1. `NER_pipeline_ray_polars.py`
   - This is a notebook that was originally run on a Databricks instance. Databricks turns all .py files into a jupyter notebook if they have a `# COMMAND ----` in the file which you will see. What you can do to run this in Sagemaker or locally, is ask Claude to transform it to a jupyter notebook format.
   - This is just 1 example of a NER pipeline that I ran to perform batch entity extraction using a GliNER-BioMed. I used Ray for GPU acceleration and Polars for efficient data processing as its built in Rust and more efficient than PySpark UDFs.
   - I originally ran this on a `g5.48xlarge` GPU so it was (8 x A10G GPUs). This config could be changed depending upon the GPU configurations.
   - The following configurations are also important to know and you may have to have Claude help you configure them as you go:
     1. `NUM_GPUS = 8` -- adjust depending upon number of GPUs you have
     2. `BATCH_SIZE = 1024` -- this is the per GPU batch size, you may have to toggle up and down depending upon your configuration and dataset size.
     3. `CHUNK_SIZE = 2000` -- reports per chunk submitted to workers --> I ran this on Databricks so in spark there are driver and worker nodes. Depending upon your config, the `CHUNK_SIZE` may have to be adjusted for GPU efficiency.

    - Polars is similar to Pandas but more efficent for BIG DATA processing at scale. The syntax is very similar.
    - You will see this config in the file, it is very minimal number of entities that I extracted, I will upload other code/notebooks that have more. In addition, this is the specific GliNER-BioMed model from hugging face that I used. 
```
MODEL_NAME = "Ihor/gliner-biomed-large-v1.0"
ENTITY_TYPES = ["anatomy", "finding", "disease"]
```
---
### 2. `NB2_LEAN_Ray_Entity_Extraction.py`
   - This again was run on Databricks so its a .py file that loads as a jupyter notebook. Again, you may have to ask Claude to transform it to a jupyter notebook to run on Sagemaker.
   - The main difference with this version is that the number of entity labels is 13. This was determined based on my own research and experimentation. Some of these entity labels were taken from RadGraph and from some other clinical NLP models.

```
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

```
