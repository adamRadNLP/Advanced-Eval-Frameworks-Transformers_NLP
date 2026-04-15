# Entity Resolution with Graphs
* This repo contains all things related to entity resolution with graphs.


# Workflow 
* These are my notes from the Neo4j Nodes 2024 Conference lecture: [A Graph Entity Resolution Playbook](https://www.youtube.com/watch?v=MfZR_ZrLSDw)

## Entity Resolution Flow
* Build Graph
* Analyze Graph:

<img width="1051" height="561" alt="Screenshot 2025-09-03 203851" src="https://github.com/user-attachments/assets/30e851c2-d787-4725-8bda-398158cee707" />
* Source: Neo4j Nodes - A Graph Entity Resolution Playbook


1. **Algorithm: Weakly connected components (WCC)**
  * narrow down list of candidate matches --> if this works then move on
  * WCC -- This answers the question: which nodes share identifiers?
    * if nodes have no identifiers in common, there's no reason to compare them!
    * Example: If nodes on left aren't duplicate of right. 
  * if WCC doesn't work, then go to another algorithm....

1b. Node Similarity
  *  how similar are nodes sets of neighbors?
  * a) Important metrics
    * Jaccard --> good as baseline
    * Overlap --> good if you have missing data
    * Cosine --> good with weighted relationships 
  
  * b) Weighted/unweighted relationships
    * Treat some relationships as more significant than others. 


1c. or FastRP with KNN
  * a) Translate graph topology to vector space with FastRP embedding algorithm
  * b) Fast RP + K-Nearest Neighbors 
    * FastRP builds an embedding vector for each node by summarizing information from the node's neighborhood.
    * Use few iteration weights for a local neighborhood. 
    * Filtered K Nearest neighbors checks each node and finds the k other nodes in the graph with the most similar property values. 
      * 1 to 2 to 3 local hops vs. 6 or 7 longer hops 
  * c) Use random walk initial neighbor sampling. 
      * This approximate approach can be much faster than nodeSimilarity for large graphs

2. **Calculate String similarity**
```
Calculate string similarity with apoc.txt
1. levenshteinDistance 
2. distance
3. levenshteinSimilarity
4. fuzzyMatch
5. jaroWinklerDistance
6. sorensenDiceSimilarity
7. hammingDistance
8. phonetic.delta
9. compareCleaned
```

4. **Calculate Descriptor penalties**
 * Penalize conflicting descriptors
 * matching descriptors doesn't make nodes much more likely to be duplicates.
 * but unmatched descriptors should decrease the probability of nodes being duplicates. 

6. **Calculate final similarity score**
  * Find the final formula
  * Make a weighted combination of the following:
    * node similarity or KNN score for common neighbors
    * string similarity scores for text properties
    * unmatched descriptor penalites
    * adjust the weights based on domain knowledge, or if you have a de-duplicated dataset you are confident in, train a machine learning model to find right weights for inputs.

# Summary 
- Design a graph data model that shows shared identifiers
- Standardize data and flag placeholders


---
# Entity Resolved Knowledge Graphs - Another Approach 
1. Node deduplication
2. Edge deduplication
3. Semantic Blocking -- group similar pairs/records together
4. Filtering 
5. Matching
6. Clustering
7. Merging (Canonicalization)

## Simplified Flow 
1. Entity collection
2. Blocking
3. Matching
4. Clustering
5. Resolution 


## References
1. Araújo, Tiago Brasileiro, Vasilis Efthymiou, and Kostas Stefanidis. "Fairness and Explanations in Entity Resolution: An Overview." IEEE Access (2025).
2. Evensen, 2024. Entity Resolution — An Introduction. Retrieved from: https://medium.com/@adev94/entity-resolution-an-introduction-fb2394d9a04e
3. Jurney, 2025. The Rise of Semantic Entity Resolution. Retrieved from: https://blog.graphlet.ai/the-rise-of-semantic-entity-resolution-45c48d5eb00a


---
# Two-Step Approach
1. Blocking
2. Similarity/Matching 



## References
1. KEJRIWAL, 2023. NamedEntity Resolution in Personal Knowledge Graphs. arXiv:2307.12173v1 22 Jul 2023. Link: https://arxiv.org/abs/2307.12173


