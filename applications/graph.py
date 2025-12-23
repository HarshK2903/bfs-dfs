from queue import PriorityQueue

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class AdjMatrixGraph:
    """
    Base class for both undirected and directed graphs using an adjacency matrix.
    """

    def __init__(self, v: int = None, filename: str = None, matrix: np.ndarray[bool] = None):
        """
        Initializes a Graph object.

        Parameters:
        - v: int, optional, number of vertices.
        - filename: str, optional, name of the file to read the graph from.
        """

        if filename is None and v is None and matrix is None:
            raise ValueError("All 'filename', 'v' and 'matrix' cannot be None!")

        if filename is None:
            if matrix is None:
                self.V = v
                self.matrix = np.zeros((self.V, self.V), dtype=bool)
            elif v is None:
                self.matrix = matrix
                self.V = matrix.shape[0]
        else:
            with open(filename) as file:
                lines = file.readlines()
                self.V = int(lines[0])
                self.matrix = np.zeros((self.V, self.V), dtype=bool)
                for line in lines[2:]:
                    pairs = line.strip().split(' ')
                    self.add_edge(int(pairs[0]), int(pairs[1]))

    def add_edge(self, v: int, w: int) -> None:
        """
        Adds an edge between vertices v and w.

        Parameters:
        - v: int, vertex index.
        - w: int, vertex index.
        """
        self.matrix[v][w] = True

    def remove_edge(self, v: int, w: int) -> None:
        """
        Removes the edge between vertices v and w.

        Parameters:
        - v: int, vertex index.
        - w: int, vertex index.
        """
        self.matrix[v][w] = False

    def adj(self, v: int) -> np.ndarray:
        """
        Returns an array of adjacent vertices to vertex v.

        Parameters:
        - v: int, vertex index.

        Returns:
        - np.ndarray: Array of adjacent vertices.
        """
        return np.argwhere(self.matrix[v]).flatten()

    def clear_edges_of_vertex(self, v: int) -> None:
        """
        Abstract method to clear edges of a vertex.

        Parameters:
        - v: vertex v
        """
        pass


class AdjMatrixUndiGraph(AdjMatrixGraph):
    """
    Represents an undirected graph using an adjacency matrix.
    """

    def __init__(self, v: int = None, filename: str = None, matrix: np.ndarray[bool] = None):
        """
        Initializes an AdjMatrixUndiGraph object.

        Parameters:
        - v: int, optional, number of vertices.
        - filename: str, optional, name of the file to read the graph from.
        """
        super().__init__(v, filename, matrix)

    def add_edge(self, v: int, w: int) -> None:
        """
        Adds an edge between vertices v and w.

        Parameters:
        - v: int, vertex index.
        - w: int, vertex index.
        """
        super().add_edge(v, w)
        super().add_edge(w, v)

    def remove_edge(self, v: int, w: int) -> None:
        """
        Removes the edge between vertices v and w.

        Parameters:
        - v: int, vertex index.
        - w: int, vertex index.
        """
        super().remove_edge(v, w)
        super().remove_edge(w, v)

    def clear_edges_of_vertex(self, v: int) -> None:
        """
        Clears all edges originating from vertex v or connected to vertex v.

        Parameters:
        - v: int, vertex index.
        """
        self.matrix[v, :] = False
        self.matrix[:, v] = False


class AdjMatrixDiGraph(AdjMatrixGraph):
    """
    Represents a directed graph using an adjacency matrix.
    """

    def __init__(self, v: int = None, filename: str = None, matrix: np.ndarray = None):
        """
        Initializes an AdjMatrixUndiGraph object.

        Parameters:
        - v: int, optional, number of vertices.
        - filename: str, optional, name of the file to read the graph from.
        """
        super().__init__(v, filename, matrix)

    def clear_edges_of_vertex(self, v: int) -> None:
        """
        Clears all edges originating from vertex v.

        Parameters:
        - v: int, vertex index.
        """
        self.matrix[v, :] = False

    def reverse(self):
        return AdjMatrixDiGraph(matrix=self.matrix.T)


class EdgeWeightedGraph:
    class Edge:
        def __init__(self, v: int, w: int, weight: float):
            self.v = v
            self.w = w
            self.weight = weight

        def either(self) -> int:
            return self.v

        def other(self, vertex: int) -> int:
            if vertex == self.v:
                return self.w
            if vertex == self.w:
                return self.v

    def __init__(self, V: int):
        self.V = V
        self.adj_list = [[] for _ in range(self.V)]
        self.has_negative = False

    def add_edge(self, v: int, w: int, weight: float) -> None:

        if weight < 0:
            self.has_negative = True

        edge1 = self.Edge(v, w, weight)
        edge2 = self.Edge(w, v, weight)
        self.adj_list[v].append(edge1)
        self.adj_list[w].append(edge2)

    def remove_edge(self, v: int, w: int):
        for edge in self.adj_list[v]:
            if edge.other(v) == w:
                self.adj_list[v].remove(edge)
                self.adj_list[w].remove(edge)
                return

    def adj(self, v: int) -> list[Edge]:
        return self.adj_list[v]

    def edges(self):
        all_edges = []
        for edges in self.adj_list:
            all_edges.extend(edges)
        return all_edges

    def visualize(self):
        G = nx.Graph()

        for v, edges in enumerate(self.adj_list):
            for edge in edges:
                G.add_edge(str(v), str(edge.other(v)), weight=edge.weight)

        pos = nx.spring_layout(G)  # Using spring layout for better visualization
        labels = {node: node for node in G.nodes()}  # Node labels for display
        edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}  # Edge labels for weights

        nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', font_weight='bold', node_size=1000)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Edge Weighted Graph")
        plt.show()


class EdgeWeightedGraphUtil:
    def __init__(self, g: EdgeWeightedGraph):
        self.g = g

    def dijkstra_shortest(self, s: int) -> list[tuple]:
        """
        Finds the shortest distance to every other vertex using dijkstra's algorithm.
        :param s: Source vertex
        :return: A list of shortest distance to every other vertex
        """

        if self.g.has_negative:
            raise ValueError('Graphs that contain negative weights are not suitable for Dijkstra\'s Algorithm')

        pq = PriorityQueue()
        visited = [False] * self.g.V
        dist = [(float('inf'), vertex) for vertex in range(self.g.V)]
        dist[s] = (0, None)
        pq.put((0, s))
        while pq.qsize() != 0:
            min_dist, vertex = pq.get()
            visited[vertex] = True
            for edge in self.g.adj(vertex):
                w = edge.other(vertex)
                if visited[w]:
                    continue

                new_dist = dist[vertex][0] + edge.weight
                if dist[w][0] > new_dist:
                    dist[w] = (new_dist, vertex)
                    pq.put((new_dist, w))
        return dist

    def bellman_ford_shortest(self, s: int) -> list[tuple]:
        """
        Finds the shortest distance from vertex s to every other vertex using the bellman ford algorithm.
        :param s: source vertex
        :return: A list of tuples consisting of
        0 - shortest distance to every other vertex
        1 - previous vertex that goes to the specific vertex
        """
        dist = [(float('inf'), vertex) for vertex in range(self.g.V)]
        dist[s] = (0, None)

        for i in range(self.g.V - 1):
            relaxed = False
            for edge in self.g.edges():
                new_dist = dist[edge.v][0] + edge.weight
                if dist[edge.w][0] > new_dist:
                    dist[edge.w] = new_dist, edge.v
                    relaxed = True

            if not relaxed:
                break
        return dist

    def shortest_path(self, s: int, d: int) -> tuple[float, list]:
        """
        Finds the shortest path using either dijkstra's algorithm or bellman ford's algorithm; with backtracking.
        If the graph contains negative weights, the bellman ford's algorithm is used.
        Else, dijkstra's algorithm is used.
        :param s: source vertex
        :param d: destination vertex
        :return: a tuple consisting of
        0 - distance
        1 - a list of vertices representing the path
        """

        if self.g.has_negative:
            dist = self.bellman_ford_shortest(s)
        else:
            dist = self.dijkstra_shortest(s)
        cost = dist[d][0]
        path = [d]
        current = dist[d][1]
        while current:
            path.append(current)
            current = dist[current][1]
        path.append(s)
        return cost, list(reversed(path))


class DFS:
    """
    Depth-First Search implementation for both undirected and directed graphs.
    """

    def __init__(self, g: AdjMatrixGraph):
        """
        Initializes a DFS object.

        Parameters:
        - graph: AdjMatrixUndiGraph or AdjMatrixDiGraph, the graph to perform DFS on.
        """
        self.g = g

    def has_path_to(self, s: int, v: int) -> bool:
        """
        Checks if there is a path from vertex s to vertex v.

        Parameters:
        - s: int, source vertex index.
        - v: int, destination vertex index.

        Returns:
        - bool: True if there is a path, False otherwise.
        """
        marked, _ = self.dfs_search(s)
        return marked[v]

    def dfs_search(self, s: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs DFS starting from vertex s.

        Parameters:
        - s: int, starting vertex index.

        Returns:
        - tuple: Tuple of _marked and edge_to np.ndarray s.
        """

        marked = np.zeros(self.g.V, dtype=bool)
        edge_to = np.zeros(self.g.V, dtype=int)
        self._recursive_dfs_search(s, marked, edge_to)
        return marked, edge_to

    def _recursive_dfs_search(self, s: int, marked: np.ndarray, edge_to: np.ndarray):
        """
        Helper function for recursive DFS search.

        Parameters:
        - s: int, current vertex index.
        """
        marked[s] = True
        for n in self.g.adj(s):
            if not marked[n]:
                edge_to[n] = s
                self._recursive_dfs_search(n, marked, edge_to)

    def path_to(self, s: int, v: int) -> list:
        """
        Finds a path from vertex s to vertex v.

        Parameters:
        - s: int, source vertex index.
        - v: int, destination vertex index.

        Returns:
        - list: List of vertices representing the path from s to v.
        """
        marked, edge_to = self.dfs_search(s)
        path = []
        if not marked[v]:
            return path

        i = v
        while i != s:
            path.append(i)
            i = edge_to[i]

        path.append(s)
        return list(reversed(path))


class BFS:
    """
    Breadth-First Search implementation for both undirected and directed graphs.
    """

    def __init__(self, g: AdjMatrixUndiGraph | AdjMatrixDiGraph):
        """
        Initializes a BFS object.

        Parameters:
        - graph: AdjMatrixUndiGraph or AdjMatrixDiGraph, the graph to perform BFS on.
        """
        self.g = g

    def bfs_search(self, s: int) -> np.ndarray:
        """
        Performs BFS starting from vertex s.

        Parameters:
        - s: int, starting vertex index.

        Returns:
        - np.ndarray: Array of _marked vertices after BFS.
        """
        marked = np.zeros(self.g.V, dtype=bool)
        queue = [s]

        while len(queue) != 0:
            v = queue.pop(0)
            for n in self.g.adj(v):
                if not marked[n]:
                    queue.append(n)
            marked[v] = True
        return marked

    def has_path_to(self, s: int, v: int) -> bool:
        """
        Checks if there is a path from vertex s to vertex v.

        Parameters:
        - s: int, source vertex index.
        - v: int, destination vertex index.

        Returns:
        - bool: True if there is a path, False otherwise.
        """
        marked, _ = self._bfs_search_path(s, v)
        return marked[v]

    def path_to(self, s: int, w: int) -> list:
        """
        Finds a path from vertex s to vertex w.

        Parameters:
        - s: int, source vertex index.
        - w: int, destination vertex index.

        Returns:
        - list: List of vertices representing the path from s to w.
        """
        marked, edge_to = self._bfs_search_path(s, w)
        path = []
        if not marked[w]:
            return path
        i = w
        while i != s:
            path.append(i)
            i = edge_to[i]
        path.append(s)
        return list(reversed(path))

    def _bfs_search_path(self, s: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper function for BFS search for a specific destination vertex.

        Parameters:
        - s: int, source vertex index.
        - w: int, destination vertex index.

        Returns:
            - tuple: Tuple of _marked and edge_to np.ndarray s.
        """
        marked = np.zeros(self.g.V, dtype=bool)
        edge_to = np.zeros(self.g.V, dtype=int)
        queue = [s]

        while len(queue) != 0:
            v = queue.pop(0)
            for n in self.g.adj(v):
                if not marked[n]:
                    queue.append(n)
                    edge_to[n] = v
            marked[v] = True
            if v == w:
                return marked, edge_to

        return marked, edge_to


class GraphUtil(DFS, BFS):
    def __init__(self, g: AdjMatrixGraph):
        DFS.__init__(self, g)
        BFS.__init__(self, g)
        self.g = g
        self._has_cycle = False

    def has_cycle(self):
        marked = np.zeros(self.g.V, dtype=bool)
        for i in range(self.g.V):
            if not marked[i]:
                self._dfs(i, marked)
        return self._has_cycle

    def _dfs(self, s: int, marked):
        self._recursive_dfs(s, s, marked)

    def _recursive_dfs(self, s: int, parent: int, marked):
        marked[s] = True
        for n in self.g.adj(s):
            if not marked[n]:
                self._recursive_dfs(n, s, marked)
            elif n != parent:
                self._has_cycle = True


class UndirectedGraphUtil(GraphUtil):
    def __init__(self, g: AdjMatrixUndiGraph):
        GraphUtil.__init__(self, g)
        self.g = g


class DirectedGraphUtil(GraphUtil):

    def __init__(self, g: AdjMatrixDiGraph):
        GraphUtil.__init__(self, g)
        self.g = g

    def topological_sort(self):

        if self.has_cycle():
            raise TypeError('Topological sort can only be performed to ACYCLIC graphs')

        marked = np.zeros(self.g.V, dtype=bool)
        s = []

        for v in range(self.g.V):
            if not marked[v]:
                self._reverse_post_order(v, s, marked)
        return s

    def _reverse_post_order(self, v, s, marked):
        marked[v] = True
        for n in self.g.adj(v):
            if not marked[n]:
                self._reverse_post_order(n, s, marked)
        s.append(v)


def visualize(g: AdjMatrixGraph | list[AdjMatrixGraph], node_size=500,
              with_labels=True) -> None:
    """
    Visualizes the graph.

    Parameters:
    - g: AdjMatrixUndiGraph or AdjMatrixDiGraph or a list of these graph types.
    - node_size: int, optional, size of the nodes in the visualization.
    - with_labels: bool, optional, whether to display node labels.
    """

    def _visualize_undirected(graph: AdjMatrixGraph) -> None:
        """
        Visualizes an undirected graph.

        Parameters:
        - g: AdjMatrixUndiGraph or AdjMatrixDiGraph
        - node_size: int, optional, size of the nodes in the visualization.
        - with_labels: bool, optional, whether to display node labels.
        """
        gr = nx.Graph()
        gr.add_nodes_from(range(graph.V))
        gr.add_edges_from(np.argwhere(graph.matrix))
        nx.draw(gr, node_size=node_size, with_labels=with_labels)
        plt.show()

    def _visualize_directed(graph: AdjMatrixGraph) -> None:
        """
        Visualizes a directed graph.

        Parameters:
        - g: AdjMatrixUndiGraph or AdjMatrixDiGraph
        - node_size: int, optional, size of the nodes in the visualization.
        - with_labels: bool, optional, whether to display node labels.
        """
        gr = nx.DiGraph()
        gr.add_nodes_from(range(graph.V))
        gr.add_edges_from(np.argwhere(graph.matrix))
        nx.draw(gr, node_size=node_size, with_labels=with_labels)
        plt.show()

    if type(g) is not list:
        g = [g]
    for graph in g:
        if isinstance(graph, AdjMatrixUndiGraph):
            _visualize_undirected(graph)
        elif isinstance(graph, AdjMatrixDiGraph):
            _visualize_directed(graph)


def samples():
    tiny = AdjMatrixUndiGraph(filename='datasets/tinyG.txt')
    medium = AdjMatrixUndiGraph(filename='datasets/mediumG.txt')
    medium2 = AdjMatrixUndiGraph(filename='datasets/mediumG2.txt')
    medium3 = AdjMatrixDiGraph(filename='datasets/mediumG3.txt')
    large = AdjMatrixUndiGraph(filename='datasets/largeG2.txt')
    visualize(tiny, 350)
    visualize([medium, medium2, medium3, large], 100, False)


g = AdjMatrixUndiGraph(7)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 4)
g.add_edge(4, 5)
g.add_edge(5, 6)
g.add_edge(6, 3)
visualize(g)
util = UndirectedGraphUtil(g)
print(util.has_cycle())
