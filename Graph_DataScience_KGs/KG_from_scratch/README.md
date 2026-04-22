# Knowledge Graphs from Scratch
- The code in this file was used to build knowledge graphs from scratch.
- I will describe what each file does and feel free to experiment with various implementations on your own.


---
# Basic Code Framework
- The files listed under here are experiments that I did with various ways for zero or few shot entity-relationship extraction.
- **Again all are .py files because they were run on a Databricks instance so you will have to convert them to jupyter notebook files to run on Sagemaker.**

## 1. Relation_Extraction_Phase_1_Experiments.py
- This was a battery of relation extraction experiments that I did using various ontologies (e.g. UMLS, SNOMED-CT).
- There is a lot of code in this notebook and some of it is useful, some is not. The general concept though is using GliNER-BioMed for entity extraction and GliREL for relation extraction. The idea is then to check these against standard ontologies. I'll admit alot of this code file is probably useless garbage but thats what you have to get when you are experimenting. If I had to do this again I would pick maybe 2 or 3 appproaches and make it more modular. I'll put it here for historical purposes but just know its a mess. 

---
## 2. Polars_GraphFrames_on_Patient_Records.py
- This was another experiment using GraphFrames which is a spark native library (recently revived) for building massive property graphs using Spark.
- It was done with veterinary patient records which is why you will see veterinary terminology.
- The basic flow is this:
  1. Ingests a patient JSON record from a delta table on Databricks
  2. Flatten the JSON
  3. Uses DSPy for zero-shot entity-relationship extraction and then entity resolution (uses `claude-4-sonnet` which at the time was a Databricks hosted model)
  4. Moves the extracted entities and relationships into a Polars dataframe.
  5. Inserts data into GraphFrames (Spark native property graph)
  6. Optional -- can visualize with "Graphistry" libary which again is spark native, however you will need to get your own tokens-keys by making an account. Otherwise you could just use NetworkX to visualize but NetworkX has its limitatations (static, not interactive).
- Important Notes
  - A UDF dataframe was tried in another notebook but known issues with LLM on worker vs. driver nodes was blocker (Databricks specific issues). 
  - Polars is a high-performance, open-source DataFrame library that provides fast and efficient data processing capabilities, especially valuable when dealing with large datasets (BIG DATA). It was built with Rust, a language known for its speed and safety, and offers Python, R, and NodeJS wrappers. For more about Polars see documention.

---
## 3. POC_Veterinary_RadGraph_pipeline.ipynb
- This was a more modular graph pipeline that I built to adopt the human RadGraph model for veterinary medical records.
- **Each component is grounded in my knowledge and experience in clinical NLP and Graph Data Science. I built this with purpose, it was not vibe coded. The individual components were carefully constructed.** You don't need all the components to be successful, but I will tell you that I chose each component with purpose after extensive research and experimentation.
- These are the key components:
```
Entity Extraction: GliNER-BioMed-large (biomedical-specialized, +5.96% over baselines)
Tokenization: spaCy en_core_web_sm (token-level mapping for GliREL)
Relation Extraction: GliREL (state-of-the-art, typed relationships with pre-extracted entities)
Semantic Resolution: DSPy framework (Russell Jurney-inspired entity deduplication with Radiology specific embedding model: IAMJB/RadEvalModernBERT)
Graph Construction: NetworkX directed graphs with typed edges
Graph Algorithms: 9 RadGraph-inspired structural analysis algorithms
Metrics: 24 comprehensive quality metrics
Storage: GraphFrames → Delta tables for Phase 2

```
- **Critical Integration Note:** spaCy tokenization is required for GliREL to work with GliNER entities. GliREL operates at the token level (not character level), so spaCy converts GliNER's character-based entity spans to token indices. This is the proven approach from my `Relation_Extraction` experiments notebook.

---
## 4. Diagnostic KG Graph Agentic Rad System POC
- These are notes from an agentic KG system an colleague and I built for Veterinary medicine. It was built with human radiology model components so it could be potentially useful.
- However, not it was built with CLIP models in mind (they were using CLIP encoder models for image classification for FINDINGS text). CLIP is a model that Open AI released that embeds images and text in the same embedding space.
---
## 5. Custom RadGraph Architecture 
- Document describing an approach to create a RadGraph-inspired veterinary radiology report evaluation using entity-relationship graphs.

---
## 6. radiology_knowledge_graph_notebook.py
- What This Notebook Does -- Extracts a **knowledge graph** from radiology report text using:
1. **GliNER-BioMed** -- extracts clinical entities (anatomy, observations, diseases)
2. **GliREL** -- extracts typed relationships between entities (located_at, suggestive_of, modify)
3. **NetworkX** -- builds a directed graph from the entities and relationships
4. **pyvis** -- creates an interactive HTML visualization you can explore in your browser

### Pipeline 
```
Report Text --> GliNER-BioMed (entities) --> spaCy (tokenization) --> GliREL (relationships)
            --> NetworkX (graph) --> pyvis (interactive visualization)
```
### RadGraph Schema (default labels)
- **Entities:**
  - Anatomy definitely present (ANAT-DP)
  - Observation definitely present (OBS-DP)
  - Observation definitely absent (OBS-DA) -- negation!
  - Observation uncertain (OBS-U)

- **Relationships:**
  - `suggestive_of`: OBS --> OBS (diagnostic inference)
  - `located_at`: OBS --> ANAT (anatomical localization)
  - `modify`: Any --> Any (descriptor modification)

### Important Notes
  - spaCy tokenization is **required** for GliREL. GliREL operates at the token level, so spaCy converts GliNER's character-based entity spans to token indices.
  - GliREL's `ner=` parameter (not `entities=`) is the correct API.
  - GliREL returns token LISTS for head/tail text, not strings. Must join them.
  - Threshold of 0.05 works well for radiology. Default 0.3 is too strict.
