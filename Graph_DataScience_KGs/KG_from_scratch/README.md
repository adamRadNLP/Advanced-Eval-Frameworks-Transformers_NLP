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

