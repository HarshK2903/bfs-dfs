# Graph Implementations and Applications in Python

This repository is to showcase Graph implementations
and its use-cases in Python.

## Contents
<hr>

### graph.py
#### Classes
- AdjMatrixGraph: Implementation of a base class for Undirected & Directed graphs
- AdjMatrixUndiGraph: Implementation of **Undirected Graphs** using adjacency matrix.
- AdjMatrixDiGraph: Implementation of **Directed Graphs** using adjacency matrix.
- DFS: Implementation of **Depth First Search**
- BFS: Implementation of **Breadth First Search**
- GraphUtil: Implementation of a base class for Undirected & Directed Graph Util
- UndirectedGraphUtil: A utility class for undirected graphs.
- DirectedGraphUtil: A utility class for directed graphs.
#### Methods
- visualize: Visualizes the given graph(s) using **networkx**
- samples: Visualizes the sample datasets provided within the repository.

#### Screenshots
Here are some screenshots of the visualization results of the samples.
<details>
  <summary>Click to expand images</summary>
  
  ![tinyG.png](datasets%2Ffigures%2FtinyG.png)
  ![mediumG.png](datasets%2Ffigures%2FmediumG.png)
  ![mediumG2.png](datasets%2Ffigures%2FmediumG2.png)
  ![mediumG3.png](datasets%2Ffigures%2FmediumG3.png)
  ![largeG2.png](datasets%2Ffigures%2FlargeG2.png)

</details>

### maze.py
maze.py is a great application of the Graph data structure. It's used to
find the shortest path in a 10x10 maze. An Undirected Graph and BFS Algorithm are used in this application.

Here is a YouTube video ,recorded by me, showing how it's used.
<details>
  <summary>Click to see the video</summary>
  
  [![Maze Solver](https://img.youtube.com/vi/WSXSdzSjFzc/0.jpg)](https://www.youtube.com/shorts/WSXSdzSjFzc)
</details>
