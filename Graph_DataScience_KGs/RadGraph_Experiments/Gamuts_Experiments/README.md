# Radiology GAMUTS
- These are previous experiments I had done with the Gamuts ontology (differential diagnosis ontology of human radiology). 
- I downloaded the CSV version of Gamuts via the bioportal: https://bioportal.bioontology.org/ontologies/GAMUTS
- It is also accessible via the website: https://www.gamuts.net/

## Other related ontologies to try in future (Anatomy related)
- Anatomy:
  - https://bioportal.bioontology.org/ontologies/FMA
  - http://sig.biostr.washington.edu/projects/fm/AboutFM.html
  - https://www.anatomiclocations.org/hierarchy/

---
# Gamuts Download notes
- The CSV export stripped all the causal relationships — 0% populated for `may_be_caused_by` and `may_cause`. The hierarchy was nearly flat: 90% of entries are children of root (rgo:00000). BioPortal CSV exports often lose relationship properties.
- As a result, I had to switch to using the RDF/XML download from the bioportal which as the full `may_be_caused_by` and `may_cause triples`.
- Here's what I ended up doing to use Gamuts:

```
1. Downloaded RDF/XML Gamuts from BioPortal
2. Uploaded to DBFS on Databricks
3. Parseed with rdflib (pure Python package, no GPU needed)
```
