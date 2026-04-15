# Graph Machine Learning
* This repo is dedicated to machine learning with Graphs.
* An excellent resource is the open source course CS224W from Stanford which can be found on YouTube.
* These are my notes from that open source course as well as some other resources. 


# Types of Graphs and Networks 

1. Networks (aka Natural Graphs)
    1. Social networks
    2. Communication and transactions
    3. Biomedicine —> relationships between genes/proteins
    4. Brain connections —> neurons and their hidden connections


2. Graphs (as a representation)
    1. Information/knowledge organized and linked
    2. Software can be a graph
    3. Similarity networks —> similar data points are connected (eg. Patient networks, social networks)
    4. Relational structures 
        1. Molecules
        2. Scene graphs
        3. 3D shapes
        4. Particle-based physics 


* **Note: The distinction between networks & graphs is often blurred…!!!!!**

* **The Big Question:**
  * **How do we take advantage of relational structures to make more accurate predictions?**
    - **Domains that are complex will have a rich relational structure and can thus be represented as a relational graph.**
    - **Modeling rich relationships  == better predictions!!!!!**


## Modern Deep Learning/Machine Learning 
* Mostly designed for and pretty much biased for simple linear sequences & grids
* For example: images, text


## Networks are more complex to process - Why????
1. Changing size and complex structures (e.g. no spatial locality like a grid in deep learning)
2. No fixed node order or reference point
3. Very dynamic with multimodal features

* Oh and by the way, **Networks are != images and text**!!!!! 


## Feature Engineering
* Traditional machine learning is all about Feature Engineering to get the best predictions.
* **Graph ML is the opposite**.
  * There is no “feature engineering”.
  * With data represented as a Graph, you have **“Representation Learning”** that automatically learns the features of a graph or network that can be applied to an algorithm  and a downstream prediction task. 


## Wait, what is Representation Learning?
* Maps nodes to d-dimensional embeddings.
* Similar nodes in a network are embedded closer together.



## Graph Machine Learning -- high-level overview
* With Graphs, you can get a high-level to very granular representation of your data:
    * Graph-level (entire graph/network)
    * Node level
    * Community/subgraph level (collection of similar nodes)
    * Edge-level (similar relationships)

---
# Applications of Graph ML

## Node level tasks
1. **Node classification**
    1. predict property of a node
    2. Example: categorize online users/items
2. **Link prediction**
    1. Predict whether there are missing links or edges between 2 or more nodes.
    2. Example: Completing a knowledge graph
    3. Example: Entity resolution 
3. **Graph classification**
    1. Categorize graphs —> predict properties of drug molecules
4. **Clustering**
    1. Do nodes in a graph form a community?
    2. Example: Social network “social circle” detection
5. **Graph Generation** —> discover and predict new drugs
6. **Graph Evolution** —> physical simulation 


### Real-World Examples of Node level tasks
1. Protein Folding
    1. Predict protein 3D structure based on amino acid sequence.
    2. Graph representation:
        1. Nodes: amino acids of protein sequence 
        2. Edges: amino acids that are similar to each other (residues)



## Edge Level Examples
1. **Recommendation Systems**
    1. This is the **most common example of a bipartite graph**.
    2. User-Item interactions (e.g. movies, music, merchandise)
    3. **Nodes: Users + Items**
    4. **Edges: User-Item interactions** 
    5. Goal: Recommend user preferences given node-edge relationships/interactions with items 
    6. Prediction Task: Nodes should be closer together that are more similar based on their edge interactions. 
2. **Drug Side Effects in healthcare**
    1. Patients take many drugs to treat complex or co-existing conditions.
    2. Task: given a pair of drugs —> predict adverse side effects 
    3. Goal: Biomedical Graph Link Prediction
        1. Nodes: Drugs & Proteins
        2. Edges: Interactions
            1. Protein-Protein interactions 
        3. Query: How likely will Simvastatin & Cipro when taken together lead to muscle breakdown?
    4. New drug side-effects can be discovered



## Subgraph-Level Machine Learning Tasks
    1. **Traffic Prediction in Google Maps** -- this is an example that we use just about everyday!
        1. Nodes: Road segments 
        2. Edges: Connectivity between road segments
        3. GNN predicts travel time given data



## Graph Level Machine Learning Tasks
1. Drug Discovery
    1. Nodes: Atoms
    2. Edges: Chemical Bonds
    3. Which molecules should be prioritized?
    4. GNN classification can predict optimal combinations.
2. Graph Generation
    1. Generate new molecular structures or pathways that have not been seen before
    2. Optimize existing drug molecules

---
# Choice of Graph Representation 

1. Objects: nodes, vertices —> N
2. Interactions: links, edges —> E 
3. System: network, graph —> G(N,E)


## How to build a graph
1. What are the nodes?
2. What are the edges?
3. Choice of network representation of a given domain/problem  determines the success of your predictions. 


## Directed vs. Undirected Graphs

1. **Undirected**
    1. Links: undirected (symmetrical, reciprocal)
    2. Examples: 
        1. Collaborations
        2. Social network friends
2. **Directed**
    1. Links: directed (arcs)
    2. Examples:
        1. Phone calls
        2. Twitter followers
    3. Connectivity of Directed Graphs
        1. Strongly connected
            1. Path from each node to every other node and vice versa (e.g. A-B, B-A)
        2. Weakly connected 
            1. Connected if we ignore edge directions
    4. Strongly Connected Components (SCCs)
        1. Not every node is part of a nontrivial strongly connected component.
            1. In-component: nodes that can reach SCC
            2. Out-component: nodes that can be reached from SCC
3. **Undirected Node degrees, k**
    1. Number of edges adjacent to node
    2. Avg node degree = sum of all nodes in network —> 2*E/N
        1. 2 times the number of edges divided by number of Nodes
4. **Directed Node degrees**
    1. In-degree
    2. Out-degree
    3. Total degree of a node is the sum of in + out degrees
5. **Bipartite Graphs**
    1. Nodes can be divided into 2 disjointed sets U and V
    2. Every link connects node U to one in V
    3. U and V are independent sets
    4. Examples:
        1. Authors to papers they authored
        2. Actors to movies they appeared in 
        3. Users to movies they rated
        4. Recipes to ingredients they contain
6. **Folded Networks**
        1. Author collaboration networks
        2. Movie co-rating networks



# Representing Graphs
1. **Adjacency Matrix**
    1. Graph is represented as a matrix.
    2. 1 == if link from node to node
    3. 0 == if no links/connections
    4. Symmetry
        1. Undirected graphs are symmetric 
        2. Directed graphs are NOT symmetric 
    5. Node Degrees
        1. Summation across row and/or columns
    6. Adjacency Matrices are Sparse
        1. Most real-world networks are sparse matrices
      
![image](https://github.com/user-attachments/assets/831936bf-cf08-4b7c-8e49-2b3c25500b43)

Source: [Stanford CS224W: Machine Learning with Graphs | 2021 | Lecture 1.3 - Choice of Graph Representation​](https://www.youtube.com/watch?v=P-m1Qv6-8cI&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=3)

2. **Edge list**
    1. List of edges is another representation of graphs but not ideal.
    2. This represents all possible connections between nodes and edges.
  
![image](https://github.com/user-attachments/assets/4e688566-e1b7-480f-833b-b210dc93f57e)
Source: Same as above

3. **Adjacency list**
    1. Easier to work with if networks are:
        1. Large
        2. Sparse
    2. Allows quick retrieval of all neighbors of a given node
        1. Example: 
            1. 1:
            2. 2: 3, 4
            3. 3: 2, 4
            4. 4:5
            5. 5:1, 2
4. Node and Edge Attributes, possible options:
    1. Weight (e.g. frequency of communication)
    2. Ranking (e.g. best friend, second best friend, etc.)
    3. Type (e.g. friend, relative, co-worker)
    4. Sign (e.g. friend vs. foe, trust vs. distrust)
    5. Properties depend on structure of rest of graph
        1. e.g. number of common friends


* This is a great reference for visualizing graphs via the mathematical representations above: [visualalgo.net](https://visualgo.net/en/graphds)

![image](https://github.com/user-attachments/assets/9dbda41d-0c8e-440e-b350-e8a7321b05aa)


## Other types of graphs
1. Unweighted (undirected)
2. Weighted (undirected)
    1. Instead of 0 or 1, we can use higher values with specific weights.
3. Self-edges (self-loops, undirected)
    1. Entries on diagonal of adjacency matrix
4. Multigraph (undirected)
    1. Same links but different meaning 
5. Connected (undirected) graph
    1. Any 2 vertices can be joined by a path. 
    2. Disconnected graph is made up by 2 or more connected parts
    3. A graph is connected if there exists a (finite) path between each pair of nodes.
    4. A graph is disconnected if there does not exist a path connecting every pair of nodes

![image](https://github.com/user-attachments/assets/6db6f997-26c6-4593-818c-8d3eb584f455)

* [Source](https://olizardo.github.io/networks-textbook/lesson-graphs-connectivity.html)

---
# Traditional Methods for ML in Graphs

## Different levels of tasks for ML in Graphs
1. Node-level predictions (vertices)
2. Link-level predictions (edges)
3. Graph-level predictions (entire graph or sub-graphs or communities)



## Traditional Machine Learning Pipelines
* These clasically invovle:

1. Train ML model: 
    * Random Forest
    * SVM
    * Neural network

2. Apply Model
    * Given node-link-graph, obtain features + make predictions 



## Feature-Design for Graphs on Undirected Graphs
* **The goal here is: How do we make predictions for a set of objects?**

### Design choices:
1. Features
    1. d-dimensional vectors
2. Objects
    1. Nodes
    2. Edges
    3. Set of nodes
    4. Entire graphs
3. Objective function
    1. What task are we trying to solve


### Mathematical Function
* The goal is to learn a function:

```
- Given: G ‎ =  (V, E)
- Learn function: f : V -> R (e.g. for given node V)
```


## Node-level Tasks
* Semi-supervised case (Node Classification)
* Given a few nodes that are labeled how can we predict the nodes that are not labeled? 
* We need Machine learning features!


### Goal: Describe the structure and position of a Node in the network:

#### 1. **Node degree (structure of network around node)**
		* **Degree k of node v is number of edges (neighboring nodes) the node has.**
		* **All neighboring nodes are treated equally which is a downside because they are unique!!!**
		* This only tells us how many nodes are connected based on their similar features/links/edges but **not their weighted or ranked level of importance to the entire network!!**
		* As example: we could have 4 nodes of patients with heart disease which are similar because of this feature. However, related to other nodes and edges, how important or impactful is each node to the entire group?

  ![image](https://github.com/user-attachments/assets/6bc92235-831c-4da1-b042-67de46c7bd9f)

  Source: Stanford CS224W: Machine Learning for Graphs https://cs224w.stanford.edu


#### 2. **Node centrality** 
		* Node degree counts neighboring nodes without capturing each nodes importance in the network.
		* Centrality c takes node importance in the entire graph into account
		* how we model importance:
			1. **Eigenvector centrality**

   ![image](https://github.com/user-attachments/assets/9562325f-53ca-4ff1-9fa8-4e26ee64b9f6)
   Source: Stanford CS224W: Machine Learning for Graphs https://cs224w.stanford.edu

			2. **Betweenness centrality** — node lies on many shortest paths between other nodes (e.g. a “transit hub” connector for other nodes)

   ![image](https://github.com/user-attachments/assets/65dfc103-2611-4515-bfba-b689d0bc2886)
   [Source](https://www.youtube.com/watch?v=3IS7UhNMQ3U&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=4)
   

			3. **Closeness centrality** — node is important if it has small shortest path lengths to all other nodes

   ![image](https://github.com/user-attachments/assets/76809f61-5da8-4ce9-bfe7-0c38941418a8)
   [Source](https://www.youtube.com/watch?v=3IS7UhNMQ3U&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=4)

			* ….more…

#### 3. **Clustering coefficient**
		* measures how connected v’s neighboring nodes are
		* # edges of neighbor nodes / kv2
		* Metric is between 0 and 1


  ![image](https://github.com/user-attachments/assets/30e98ecc-e7c1-4b9e-a3f1-8d8fef2f6760)
  [Source](https://www.youtube.com/watch?v=3IS7UhNMQ3U&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=4)

  
#### 4. **Graphlets**
		* Clustering coefficient counts **num of triangles** in ego-network of node
		* We can generalize this by counting # of pre-specified subgraphs or graphlets
		* We can quantify the number of possible graph lets 
		* 2-node, 3-node, 4-node, 5-node graphlets 
    * Source: same as above

![image](https://github.com/user-attachments/assets/1cc343d0-ecf9-45b6-afd3-ed3fa036e4b4)


![image](https://github.com/user-attachments/assets/6491ac7e-f838-4276-9e59-6d61ac27624f)



### Graphlet Degree Vector (GDV)
* Graphlet degree vector counts the number of graphlets that a node touches at a particular orbit position.
* If we only consider graphlets of 2 to 5 nodes we get a total of **73 orbit positions.**
* Therefore, GDV will be a vector of 73 coordinates.
  * It is a signature of a given node that describes the topology of node's neighborhood. Captures its interconnectivities out to a distance of 4 hops. You can compare GDV of two nodes and get a highly constraining measure of local topological similarity between them. Maybe you can use it as your node feature vector??
  * [Source](https://wandb.ai/syllogismos/machine-learning-with-graphs/reports/3-Motifs-and-Structural-Roles-in-Networks--VmlldzozNzU1NTg)

### Automorphism Orbits
* For a node `uuu of graph G`, the automorphism orbit of `uuu` is `Orb(u)={v∈V(G);v=f(u)` for some `f=Aut(G)}Orb(u) = \lbrace v \in V(G)`; v = f(u) \text{ for some } f = Aut(G)\rbraceOrb(u)={v∈V(G);v=f(u) for some f=Aut(G)}

![image](https://github.com/user-attachments/assets/d5b7800d-877e-46c7-a62a-779663a78720)


* The Aut denotes an automorphism group of G, i.e., an isomorphism from G to itself. This is a bit confusing, you can look at the image below to see how a,b,c,da, b, c, da,b,c,d are orbit positions for the node vvv in graph GGG. You assume vvv in those positions and count the number of instances that graphlet touches node vvv
* The node vvv can assume any of the positions a,b,da, b, da,b,d in the graphlets. But not ccc Hence in the graphlet degree vector you put in 0 for orbit position ccc

![image](https://github.com/user-attachments/assets/9018780e-2fb3-4681-8b9b-db5fb9e51875)


### Graphlet Node Features
* If we only consider graphlets 2 to 5 nodes we will have:

1. Vector of 73 coordinates is a signature of a node that describes the topology of a node’s neighborhood 
2. Distance of 4 hops captures interconnectivity — a chain of 4 edges has 5 nodes
3. Graphlet degree vector provides measure of node’s local network topology
    1. By comparing vectors of 2 nodes this provides a more detailed measure of local topological similarity than node degrees or clustering coefficients
  
---
# Structural Roles in networks
* Roles are functions of nodes in a Network.
* Roles are measured by structural behaviors.
* **Note roles are different from groups/communities!!!!**
  * Roles are based on the **similarity of ties between subsets of nodes**.
  * Nodes with the **same role have similar structural properties**, however they don’t need to be in direct or even indirect interaction with each other.
* A **Group/community** is formed based on **adjacency, proximity or reachability of nodes in the same community** and are well-connected to each other.


## Structural equivalence
* In the diagram below, nodes u and v are structurally equivalent if they have the **same relationships to each other.**
* **Structurally equivalent nodes are likely to be similar in different ways.**
* A an example, node u and v are structurally equivalent in the figure below because they connect other nodes **in the same way.**
* [Source](https://snap-stanford.github.io/cs224w-notes/preliminaries/motifs-and-structral-roles_lecture)

![image](https://github.com/user-attachments/assets/b51d0e6a-325e-497a-bfbf-eaf803a8ed82)


## Discovering Structural Roles in Networks
* These are a few problems where determining structural roles in a network are useful
* [Source](https://wandb.ai/syllogismos/machine-learning-with-graphs/reports/3-Motifs-and-Structural-Roles-in-Networks--VmlldzozNzU1NTg)

![image](https://github.com/user-attachments/assets/5324ad30-a3e7-43e1-aa6b-daa80da52be0)

* Another summary of role discovery from Rossi et al. 2016:

![image](https://github.com/user-attachments/assets/2a69188a-112f-4292-b611-64bb2b7e2baf)


## RoIX
* Roles allow us to identify different properties of nodes in network. Here we will introduce an automatic structural roles discovery method called RolX. It’s an unsupervised learning approach without prior knowledge.
* [More about RoIX here](https://snap-stanford.github.io/cs224w-notes/preliminaries/motifs-and-structral-roles_lecture)

---
# Link-Level Prediction Tasks
* Predict **new links** between existing network links.
* How this is done:
    1) All node pairs with no existing links are ranked.
    2) Top K node pairs are predicted
 
* Key: Design features for a pair of nodes


## Link-Prediction Task: 2 main methods
1. Links missing at random
   * Randomly remove sets of network links --> goal is then to predict them.

2. Links over **time**
   * Given `G[to, t'o]` graph on edges up to time `to'`
   * Goal: Output ranked liks `L` of links that are not in the given graph.

### Biological Networks over time - Real-World use case
* Link prediction in biological networks aims to identify potential or missing interactions between biological entities (like proteins, genes, or diseases) by analyzing existing network data.
* When considering time-varying biological networks, link prediction becomes a task of predicting future interactions based on past and present connections.
* This involves analyzing the dynamics of the network to uncover patterns that can be used to forecast future relationships.
* As an example, a paper by Aziz et al. in 2021 [Multimorbidity prediction using link prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC8360941/#:~:text=Predicting%20the%20likelihood%20of%20a,Recall%2C%20and%20F%2DScore.), the authors used a network-based approach to analyze multimorbidity data and develop methods for predicting diseases that a patient is likely to develop. 


## Link Prediction via Proximity
* Methods:
  * Given a pair of nodes `(x,y)`, we will compute score `c(x,y)`.
  * `c(x,y)` could be the number of most common neighbors to nodex `(x,y)`
  * We then sort node pairs `(x,y)` in decreasing order by their common neighbors `c(x,y)`
* Goal:
  * **Predict the top `n` pairs of nodes as NEW LINKS**.
  * Analyze which of these links **actually appear** in `G[t1,t1']` the new temporal graph nodes.
 
## Link-Level Features - 3 Main Types
1. **Distance-based features**
   * Shortest-path distance between 2 nodes.
   * **Problem: this metric DOES NOT capture the degree of overlap between common or nearest neighborhoods.** 
2. **Local neighborhood overlap**
   * Analyze the number of neighboring nodes that are shared between 2 nodes `v1` and `v2`.
   * These are the methods:
     * **Common neighbors**
     * **Jaccard's Coefficient**
     * **Adamic-Adar index** --> this actually works better than the 2 previous methods.
     * **Problem: Becomes zero when no neighbor nodes are shared or there are nodes that are more than 2 hops apart from one another.**
3. **Global neighborhood overlap**
   * Local neighborhood overlap features have significant limitations:
     1) If 2 nodes **do not have any common neighbors, metric is ALWAYS ZERO.**
     2) If 2 nodes to not have any common neighbors, it does not mean they won't overlap or be connected in the future.
   * **Global overlap** resolves the local neighbor problem by utilizing the **entire graph**.

### Global Neighborhood Overlap Metrics
* **Katz index**
  * Number of paths of all lengths between a given pair of nodes is counted.
  * How do we calculate the number of paths and their lengths between 2 nodes?
    * **Graph Adjacency Matrix!!!**

* **Computing number of paths between 2 nodes**
  * The simple intuitive explanation is that if there is a path we input the value 1, if there is no path we input the value 0.
  * See this slide from Professor Jure Leskovec's lecture:

![image](https://github.com/user-attachments/assets/eef3d1fe-787c-4e2f-947e-7a3e187c93a0)

* **How to compute number of paths for 2 given nodes?**
  1) Compute num of paths of length 1 between each of node `u` neighbor and `v`
  2) Add these paths across `u` neighbors
 * We can see this again from Professor Jure Leskovec's lecture, [source](https://snap.stanford.edu/class/cs224w-2021/slides/02-tradition-ml.pdf)

![image](https://github.com/user-attachments/assets/08c1ad36-4780-4bd7-a402-ba65111b009e)

* How do we do this for ALL possible paths?
  * use adjacency matrix powers
  * Possible paths goes from 1 to infinity!
 

---
# Graph-Level Prediction Tasks
* Goal: To find features that give us the characteristics of an **ENTIRE GRAPH**
* **Kernel methods**
  * These are commonly used for machine learning graph-level prediction tasks **instead of feature vectors.**
  * **A Kernel matrix measures similarity between two graphs.**
* There are 2 common Graph Kernels:
  1) **Graphlet Kernel**
  2) **Weisfeiler-Lehman Kernel**

* Other less common kernels:
  * Random-walk kernel
  * Shortest-path graph kernel
  * ...others...
 
## Graph Kernel Concept
* Goal: Design graph feature vectors
* [GraKeL](https://github.com/ysig/GraKeL?tab=readme-ov-file) is a sklearn implementation of many Graph kernel methods. 
* Idea: **Bag-of-Words (BoW)**
  * BoW intuition:
    * **BoW counts the number of occurrences of words in a document without order.** 
    * Nodes as words --> naive extension to graphs.
    * Both graphs have 4 red nodes --> same feature vector for 2 different graphs.
    * From Professor Jure Leskovec's lecture:
   
![image](https://github.com/user-attachments/assets/56f6d077-1b9a-4509-8b58-fcd089f01d97)

* **Bag of node degrees**
  * Deg1, Deg2, Deg3
  * Both the Graphlet kernel and Weisfeiler-Lehman kernel use `Bag-of-*`

### 1. Graphlet Kernel Features
* Count number of **different graphlets** in a graph.
* Graphlets here is NOT the same as **node-level features**, the differences are:
  1) Graphlet nodes **DO NOT NEED TO BE CONNECTED** -- allows for isolated nodes.
  2) Graphlets are **NOT ROOTED**.
* Example from Professor Jure Leskovec's lecture:
  * Notice that for k=3 you have 4 graphlets
  * This is because you can have 3, 2, 1, 0 connections.

![image](https://github.com/user-attachments/assets/f9f084f5-6b6b-4638-b72f-a5b6e6e32227)

#### Limitation of Graphlet kernels
* If you count size-k graphlets for a graph with size `n` this takes a long time and is exponential in a graph.

### 2. Weisfeiler-Lehman Graph Kernel
* MORE efficient graph kernel!
* Concept
  * Using graph neighborhood structure to iterate and enrich node vocab.
  * Generalizes the Bag-of-node degrees because node degrees are 1 hop neighborhoods.
  * Algorithm: **Color Refinement**
 
#### Color Refinement Algorithm
* Given graph `G` with nodes `V`
  * Assign initial color to each node.
  * Refine node colors iteratively using HASH maps of various colors.
  * **After K steps of refinement, summarize structure of k-hop neighborhood.**
* WL kernel is **more computationally efficient** due to neighborhood aggregation.
  * Total time complexity is linear in number of edges.
* Example from Professor Jure Leskovec's lecture, given 2 similar graphs:
  * 2 graphs that are similar but slightly different.
  * Iterate and aggregate neighboring colors into HASH table.
  * [Source below](https://snap.stanford.edu/class/cs224w-2021/slides/02-tradition-ml.pdf)

![image](https://github.com/user-attachments/assets/059ab2c3-e115-42d1-8246-ac8f642a1da1)

![image](https://github.com/user-attachments/assets/5a8eb6ec-90e2-45a7-bca8-5c3212083d05)

![image](https://github.com/user-attachments/assets/39f3f6eb-7235-4df0-93cd-6457fa2d9c49)

![image](https://github.com/user-attachments/assets/2bb89c51-f10b-4301-a028-9fb3074ff2ee)

![image](https://github.com/user-attachments/assets/c9a0f884-9790-4608-8ab9-660a5b4bdc5d)

![image](https://github.com/user-attachments/assets/27bcf567-d6ee-45ad-b5ea-0a7ba2132560)

---
# Node Embeddings
* Traditional machine learning on graphs usually consists of:
  1. Given an input graph
  2. extract node --> link --> graph-level features
  3. learn a model (e.g. SVM, neural network, etc.) that maps
features to labels.

* **Graph Representation Learning** removes the need for manual feature engineering to extract the structured features.
  * **Representation Learning automatically learns the features of a graph**

## Goal of Graph Representation Learning?
* Perform efficient task-independent feature learning.
* This image from [Stanford Professor Jure Leskovec's lecture on Node Embeddings](https://snap.stanford.edu/class/cs224w-2021/slides/03-nodeemb.pdf) demonstrates this task below. You are taking a node --> learning a function --> representing this as vector embeddings.
* The resulting vectors should capture the semantic representations of the graph network. 

![image](https://github.com/user-attachments/assets/cc63112b-9749-458a-9c19-586e12ffec99)


## Wait....Why would you want to create Node Embeddings?
* Task is to map graph nodes into an embedding space.
* Reasoning:
  1. **Node similarity represented by embeddings == node similarity in a graph network.**
  	* As an example:
    	  * Two nodes that are closer to one another connected by an edge **should* be embedded in the same vector space (theoretically). 
  2. Encoding network information
  3. Downstream prediction and/or recommendation tasks (see image below from Professor Jure Leskovec's lecture)

![image](https://github.com/user-attachments/assets/fdc9d35f-f481-45a6-a3d9-4065c4ddb5b5)


* The original paper on this subject was published by Perozzi et al. in 2014 entitled [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
  * The example below is an input graph is embedded and the nodes are mapped to a 2-dimensional vector space. 

![image](https://github.com/user-attachments/assets/811cc4a3-8e1f-41e9-baa0-5473b3fe26cc)

## How do you create Node (or Graph) Embeddings?
* Going back to Graph Representation techniques above, we use an **Adjacency Matrix**.
* This is depicted as:
  * V is vertex set
  * A is adjacency matrix (assumes binary)
  * No node features or extra information is used (for simplicity this is an undirected graph). See representation below from Professor Jure Leskovec's lecture:

![image](https://github.com/user-attachments/assets/704c6655-05e3-4160-8753-0a40381dbfd3)

* Embedding space should be the similarity (e.g. dot product) of every node encoded.

## How do you define a node similarity function?
1. An encoder maps nodes to embeddings (low-dimensional vector)
   * Simplest encoding method: embedding lookup in matrix
     * Simple Examples: DeepWalk, node2vec
   * Complex encoder + decoder methods: GNNs are a more complex method to learn node embeddings.
2. Similarity function defined --> original network relationships mapped to vector space relationships.
3. Decoder maps embeddings to similarity score.
4. Final optimization of encoder parameters.

# Node Similarity using Random Walks
* Random Walks is a common approach to this.
* Random Walks are **unsupervised/self-supervised** method to learn node embeddings.
* Important distinctions:
  1. NO node labels
  2. NO node features (e.g. attributes)
  3. Goal is to estimate embedding coordinates of a node so that the input network structure and relationships are preserved.
* Embeddings are **Task Independent**
  * There is no specific task the embeddings are trained on, they can be used for any task. 

## Random Walk Notation
* The standard notation for Random Walks is as follows [Source](https://snap.stanford.edu/class/cs224w-2021/slides/03-nodeemb.pdf)

![image](https://github.com/user-attachments/assets/43e70859-32bb-45e4-a6a8-7e83058f80ba)

* A Random Walk is a sequence of graph points (nodes) visited via random probability then optimization of the resulting embeddings to encode. [Source](https://snap.stanford.edu/class/cs224w-2021/slides/03-nodeemb.pdf)


![image](https://github.com/user-attachments/assets/46448a50-b39f-4874-a5cb-f3ac81aaab47)


## Why would you use a Random Walk?
1. **Expressive**
   * Demonstrates local AND higher-order graph neighborhood information for node similarity.
   * Math definition: "if a random walk starting from node u visits v with high probability, thus u and v are similar via high-order multi-hop information"

2. **Efficiency**
   * Eliminates need for having to consider ALL node pairs during training graph representation model.
   * **ONLY need for considering pairs that co-occur on a random walk**

## Feature Learning Optimization
* Given node u, we learn feature representations that will be **predictive of nodes within its random walk neighborhood `NR(u)`.**
* A node can be visited multiple times.
* Softmax Parameterization
  * Node v should be most similar to node u out of all nodes n.
  * Softmax is used to paramterize the exponential dot products.
 
* This can be summarized as follows:
  1. Run short-fixed length random walks beginning from each node on a graph.
  2. For each node u we collect NR(u) the multiset of nodes visited on random walks starting from u.
  3. Optimize node embeddings using Stochastic Gradient Descent algorithm (approximated using negative sampling)

### Negative Sampling
* Negative Sampling is used as it is a form of Noise Contrastive Estimation (NCE) which approximately maximizes the log probability of the softmax.
* Random walks are used to generate sequences of nodes that capture the graph's structural information. These sequences are treated like "sentences" in natural language processing. 
* Skip-gram with Negative Sampling:
  * The skip-gram model (a component of word2vec) is then used to learn node embeddings.
  * It predicts the context nodes given a central node. 

* Softmax Normalization:
  * In the original skip-gram formulation, the probability of a context node is calculated using softmax, which involves summing over all nodes in the graph, making it computationally expensive for large graphs. 

* **Negative Sampling Solution:**
  * Negative sampling avoids this by sampling a small number of "negative" nodes (nodes not in the context) and only calculating the probability distribution over this subset.
  * This significantly speeds up the training process. 

* Efficiency and Bias
  * **We sample k negative nodes each with probability to its degree.**
  * There are 2 general considerations for k (e.g. number of negative samples):
    1. Higher k gives more robust estimates
    2. Higher k leads to higher bias on negative events -- **A k between 5 and 20 is most commonly used**

* Node2vec's Biased Random Walks
  * Node2vec further optimizes random walks by introducing parameters p and q to control the exploration of the graph, allowing for both local and global exploration. 

* Hub-Aware Methods
  * Some methods, like HuGE, attempt to improve random walk generation by considering node importance (e.g., degree) to create more informative walks.
  * Fang et al. 2024. [Distributed Graph Embedding with Information-Oriented Random Walks](https://arxiv.org/pdf/2303.15702#:~:text=These%20graph%20embedding%20algorithms%20are,embeddings%20from%20the%20sampled%20walks.&text=%F0%9D%91%A2%F0%9D%91%97+%F0%9D%91%96%20denotes%20a%20context,with%20negative%20sampling%20%5B34%5D.&text=%2Dall%20strategy%20cannot%20meet%20the,representation%20in%20the%20sampling%20procedure.)

# node2vec
* Goal is to create embeddings of nodes with similar network neighborhoods close in a feature space.
* This is a **maximum likelihood optimization problem** which is independent to a downstream prediction task.
* Flexible notation of network neighborhood NR(u) of node u leads to "richer" node embeddings.
* Develop a biased 2nd order random walk R to generate the neighborhood network NR(u) of node u.
* **Key differences between DeepWalk and node2vec**:
  1. how to define the set of number of neighboring nodes
  2. how the random walk is defined

## Biased Walks in node2vec
* The concept is to use "flexible, biased" random walks that alternate between local and global network views.
* This was first introduced in the Grover and Leskovec paper in 2016, [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)
* This is similar to doing BFS and DFS traversal in a graph. We can see this depicted in the figure below from the 2016 paper:
  1. DFS - entire network exploration
     * Recall that DFS explores an entire network first going down the left side of a tree then beginning at the last node it left off at and exploring the right side of a tree (consider this similar to a hierarchical tree structure).

![image](https://github.com/user-attachments/assets/6cbcefe9-33b8-4e12-b070-08caef7856a4)


  2. BFS - local network exploration
     * Recall that BFS goes in parallel down both sides of a tree. So it will go out 1 hop left and 1 hop right and proceed in parallel down the hierarchical tree structure.

![image](https://github.com/user-attachments/assets/a0b1ba40-0241-4174-ad8f-cd18172770ac)

 
![image](https://github.com/user-attachments/assets/261a5431-e155-4915-9515-d6bbe68d3392)

* A more granular view of DFS vs. BFS in biased walks is seen below as from Professor Jure Leskovec's lecture:

![image](https://github.com/user-attachments/assets/bbaf39e8-c3e6-49d2-b463-41507ebbece6)


![image](https://github.com/user-attachments/assets/805a2886-e4e5-4d2c-8f36-b9beac6e1f63)

# Which Node Embedding method(s) should you use?
* Just as with any embedding model or technique, there is "no one size fits all" approach.
* For example, node2vec can perform better on node classification tasks and other graph embedding techniques may perform better on link prediction and other algorithms.
* **Random Walk approaches are usually more efficient because you can define a small subset of walks to sample, but random walks generally don't scale to large networks as well as other methods.** 
* Generally speaking you choose the algorithmic approach that best fits your data and goals of your application.
* There are various papers on this subject including:
  1. Wu et al, 2023. [A Survey on Graph Embedding Techniques for Biomedical Data: Methods and Applications](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4374504)
  2. Goyal et al, 2019. [Benchmarks for Graph Embedding Evaluation](https://arxiv.org/abs/1908.06543)
  3. Madushanka et al. 2024. [Negative Sampling in Knowledge Graph Representation Learning: A Review](https://arxiv.org/html/2402.19195v2)

## Biomedical Applications of Embedding Methods
* The paper by Wu et al. in 2023 break this down in to homogenous, heterogenous, and dynamic graph based methods as seen below in Figure 4.

![image](https://github.com/user-attachments/assets/ae5db481-149a-4a68-a6d5-83b5d0d4baa9)

---
# Graph Embeddings
* Embedding entire graphs!
* Goal: embed entire subgraph or entire graph `G` into embedding space.
* Tasks:
  1. Classification
  2. Anomaly detection
 
## Approach 1 - Sum/Avg node embeddings
* Run standard graph embedding technique over subgraph `G`
* Sum or Average **node embeddings** in the subgraph `G`
* In plain english: Graph embeddings are equal to the sum or average of the nodes in that graph.
* Original paper: [Duvenaud et al. 2016](https://arxiv.org/abs/1509.09292) -- classifying molecules based on graph structures.

<img width="714" height="260" alt="image" src="https://github.com/user-attachments/assets/85387270-954f-4d5e-884c-24872ea6b934" />


## Approach 2 - "Virtual Node" via node2vec
* Add a "virtual node" to represent the "subgraph" and run standard embedding technique.
* In plain english: the "virtual node" represents the embeddings for the graph via the nod2vec algorithm run on the entire graph. 
* [Li et al. 2016](https://arxiv.org/abs/1511.05493)

<img width="668" height="204" alt="image" src="https://github.com/user-attachments/assets/13503618-cbc4-4fc5-a433-486d43b9384e" />

* [Image sources above](https://snap-stanford.github.io/cs224w-notes/machine-learning-with-networks/node-representation-learning)

## Approach 3 - Anonymous Walk Embeddings
* States in anonymous walks will correspond to the index of the first time you visit a node in a random walk.
* You get essentially a temporal sequence of when the nodes were visited.
* These are "agnostic" to the identity of the nodes that were visited (e.g. anonymous!)
* The number of anonymous walks will increase EXPONENTIALLY!

<img width="624" height="618" alt="image" src="https://github.com/user-attachments/assets/80639469-88b6-4854-9a95-dba4a7169e61" />

* Source: [Ivanov et al. 2016](https://arxiv.org/pdf/1805.11921)

### Using Anonymous Walks
* Simulate anonymous walks.
* Represent graph as probability distribution of these walks.

### How many walks do you need?
* Formula: We want distribution to have error of more than epsilon with probability less than delta. 

## Approach 4 - Learn Walk Embeddings
* Besides representing each walk by a fraction of times it occurs, learn graph embeddings together with all anonymous walk embeddings.
* If you embed walks --> the next walk can be predicted!
* Objective: As you sample random walks, the goal is to learn to predict walks that co-occur in a certain window size.
  * Predict w2 given w1, w3 if delta is 1.
 
## What can you use graph embeddings for?
1. Clustering/community detection
2. Node classification
3. Link prediction
4. Graph classification

## Summary of Embedding approaches
1. Encoder-decoder framework
   * encoder: embedding lookup
   * decoder: predict score based on embedding to match node similarity
  
2. Node similarity measure
   * biased random walk
   * examples:
     * DeepWalk
     * Node2Vec

3. Extension to Graph Embeddings
   * Node embedding aggregation
   * Anonymous walk embeddings

---
# References
1. [Stanford CS224W: ML with Graphs | 2021](https://www.youtube.com/watch?v=4dVwlE9jYxY&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=5)
2. [Network comparison using directed graphlets](https://arxiv.org/abs/1511.01964)
3. [Encoding edge type information in graphlets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273609)
4. [Traditional ML for Graphs](https://medium.com/@taunkdhaval08/traditional-ml-for-graphs-ca7bdbef7544)
5. [Motifs and Structural Roles in Networks](https://wandb.ai/syllogismos/machine-learning-with-graphs/reports/3-Motifs-and-Structural-Roles-in-Networks--VmlldzozNzU1NTg)
6. [Biological network comparison using graphlet degree distribution](https://academic.oup.com/bioinformatics/article/23/2/e177/202080?login=false)
7. [Stanford SNAP - Motifs and Structral Rules in Network](https://snap-stanford.github.io/cs224w-notes/preliminaries/motifs-and-structral-roles_lecture)
8. [Stanford CS224W Class Notes](https://snap-stanford.github.io/cs224w-notes/)
9. [Rossi et al. Role Discovery in Networks, 2016](https://arxiv.org/pdf/1405.7134)
10. Perozzi et al, 2014. [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
