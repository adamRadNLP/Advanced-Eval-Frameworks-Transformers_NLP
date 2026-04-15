# Knowledge Graphs from Scratch
- The code in this file was used to build knowledge graphs from scratch.
- I will describe what each file does and feel free to experiment with various implementations on your own.


---
# Basic Code Framework
- The files listed under here are experiments that I did with various ways for zero or few shot entity-relationship extraction. Again all are .py files because they were run on a Databricks instance so you will have to convert them to jupyter notebook files to run on Sagemaker.

## 1. Relation_Extraction_Phase_1_Experiments.py
- This was a battery of relation extraction experiments that I did using various ontologies (e.g. UMLS, SNOMED-CT).
- There is a lot of code in this notebook and some of it is useful, some is not. The general concept though is using GliNER-BioMed for entity extraction and GliREL for relation extraction. The idea is then to check these against standard ontologies. I'll admit alot of this code file is probably useless garbage but thats what you have to get when you are experimenting. If I had to do this again I would pick maybe 2 or 3 appproaches and make it more modular. I'll put it here for historical purposes but just know its a mess. 
