# Graph-Data-Science-and-Machine-Learning
* A repo devoted to all things Graph Data Science and Machine Learning.
* Including but not limited to:
1. Knowledge Graphs
2. Graph Databases
3. Graph Analytics and Algorithms
4. Graph Machine Learning and Neural Networks
5. GraphRAG and AI applications


# What are Graphs?
* Graphs are a way to model a network of interconnected objects, topics, people, things, the list goes on.
* Graphs are commonly used in computer science as data structures consisting of:
  * sets of nodes (vertices/points)
  * sets of relationships (edges/links)
 

# What are Graph Databases?
* Graph DBs are designed to store graph based structures for data modeling and semantic queries of the existing nodes, relationships and properties (attributes).
* Popular Graph database formats include:
  1. RDF
  2. LPG (labeled property graph)
 

# What Graphs are NOT
* Graphs are not charts or plots. Although the term is used interchangebly it is not the same thing in Data Science, Machine Learning and Computer Science. 

# Types of Graphs
1. Undirected
   * Edges do not have orientation.
2. Directed
   * Edges have orientation.
3. Weighted
   * Edges have assigned numeric values. 
4. Multigraph
   * Two nodes may be connected by more than one edge. 
5. Bipartite
    * Nodes are divided into two disjoint sets, and all relationships connect nodes from different sets. Bipartite graphs are often used to model relationships between two distinct types of entities, such as users and products, students and courses, or employees and skills. They are particularly useful in applications like recommendation systems or matching problems.
6. Cyclic
    * Have at least one cycle, which is a path that starts and ends at the same node.
7. Acyclic
    * No cycles are present. 
8. ....there are a lot more....

* Image source below [Memgraph](https://memgraph.com/blog/graph-search-algorithms-developers-guide)

![image](https://github.com/user-attachments/assets/260ac07c-2c64-4108-82e1-d36e8dddd653)



# Graph Network Analysis
* **Data that is highly connected requires a graph based approach leveraging network analytic algorithms.**
* These algorithms allow us to analyze nodes, edges and property relationships to surface insights and relevant information in our data.
* Graph Algorithms are the cornerstone of graph/network analysis, they can be used to solve such problems as:
1. Shortest Path
2. Community Detection
3. Influential Nodes/Relationships
4. Cycle detection
5. Graph coloring
6. ...etc..

## Graph Algorithms
* [Source](https://memgraph.com/docs/advanced-algorithms/deep-path-traversal#breadth-first-search)
* Most common algorithms are:
1. **Traversals**
   * Visit EVERY NODE in a graph.
   * This is similar to the traveling salesman or postman who visits every house on your street to deliver mail.
   * In Graph traversals, some nodes can be visited multiple times.
   * The key is to remember which nodes were previously "visited" so you don't duplicate node visits too many times if not necessary.
   * The main Graph Traversal Algorithms are:
     * **Breadth-First-Search (BFS)**
       * BFS is best for locating the shortest path between two nodes in an unweighted graph or between the start node and any other node in the graph. Since BFS traverses all nodes at a given depth before moving to the next level, it ensures that the **shortest path** is found.


     * **Depth-First-Search (DFS)**
       * DFS returns all paths between nodes and is most useful for determining path existence between two nodes in your graph.
       * If the output of the algorithm is null, there are no available paths between the nodes.
       * If the output is NOT null, the DFS algorithm is going to provide all possible paths as a result.
2. **Centrality**
   * This answers the question: **What node(s) in the graph are most important?**
   * **Ranking** of node importance is the output of this algorithm.
   * Most common use cases:
     1. Social Networks --> who is most influential user?
     2. Urban Networks --> infrastructure
     3. Disease Networks
     4. Patient Networks
    * Most common centrality algorithms:
      * **PageRank**
        * Originated from Google's web search.
        * Output is a probability distribution based on the ranking of importance of webpages in a network related to a search. 
      * **Katz**
      * **Betweenness**
        * num of instances node is a "bridge" on shortest path between 2 other nodes.
        * higher betweenness == MOST INFLUENTIAL
      * **Closeness**
      * **Degree**
3. **Pathfinding**
   * Search graph starting at 1 node of interest and search all adjacent nodes until final destination.
   * Goal is to find the **cheapest route**.
   * Most similar to the **shortest path problem** --> what is the shortest path to travel?
   * Two most common algorithms:
     * **Dijkstra's**
       * Shortest path between nodes in weighted graph.
     * **A***
       * Weighted version of Dijkstras.
      

 # Real-World Use Cases of Graph Analytics
 1. Google --> PageRank for similarity search
 2. Facebook --> Community Detection
 3. Amazon --> Collaborative Filtering for recommendations
 4. Pinterest --> Random Walks + Graph ML for recommendations
 5. Uber Eats --> Graph ML for recommendations


## Social Networks
* Most common use case of Graph Analytics/Network Analysis
* Below is an example of the Louvain Algorithm from this recent paper: [Comparative Analysis of Community Detection Algorithms on the SNAP Social Circles Dataset](https://arxiv.org/html/2502.04341v1)
  * github from the paper: [Comparative Analysis of Community Detection Algorithms](https://github.com/YashM246/Comparative_Analysis_of_Community_Detection_Algorithm)
  * [Stanford SNAP Large Network Datasets](https://snap.stanford.edu/data/index.html)

![image](https://github.com/user-attachments/assets/cc964a5e-4cda-4801-9004-c0aae85ec0f2)


## Recommendation Engines
* Example of link prediction from [AWS](https://aws.amazon.com/blogs/machine-learning/graph-based-recommendation-system-with-neptune-ml-an-illustration-on-social-network-link-prediction-challenges/)

![image](https://github.com/user-attachments/assets/4c947be1-4701-4028-9a0f-9307ebe16ab3)


## Other Examples
* Fraud Detection
* Supply Chains
* Patient and Disease networks

# Graph Analytics Tools
* This is a great diagram from Memgraph of all the Tools available for Graph Analytics.
* NetworkX is the most popular Python tool. 

![image](https://github.com/user-attachments/assets/879c865a-b6bb-45fd-a7df-d60563a720bb)


## NetworkX
* Great for getting quick analysis on graphs and networks.
* Supports multiple data formats.
* Data structure analysis for:
  * graphs
  * digraphs
  * multigraphs
  * multidigraphs
  * standard graph algorithms
 
* Example of all types from [Memgraph](https://memgraph.github.io/networkx-guide/basics/):

![image](https://github.com/user-attachments/assets/81c191b2-f62d-4639-af4a-39d19bd491ed)


### Reading Graphs
* Data can be imported from multiple data types including:
1. Adjacency list
2. Multiline adjacency list
3. Edge list
4. GEXF
5. GML
6. Pickle
7. GraphML
8. JSON
9. LEDA
10. SparseGraph6
11. Pajek
12. GIS Shapefile
13. Matrix Market
14. CSV

* Example code:
```
graph_type = nx.Graph()
df = pd.read_csv('graph.csv')
G = nx.from_pandas_edgelist(df, create_using=graph_type)
```

### Add Nodes and Edges
* hashable Python objects to add new nodes and edges to NetworkX.

### Graph Algorithms in NetworkX
1. Community detection (e.g. Girvan-Newma, Fluid communities, Label Prop)
2. Traversals (e.g. BFS, DFS)
3. Pathfinding (e.g. Dijkstra, A* search)
4. Connected components (e.g. Weakly connected components, Strongly connected)
5. Centrality (e.g. Degree centrality, Closeness, Betweenness, Katz, PageRank)

### Graph Drawing
* Basic drawing with Matplotlib and interfae for Graphviz.

### Writing/Exporting Graphs
* Can export graphs in many formats including JSON and Adjacency Lists. 

