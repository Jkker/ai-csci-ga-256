import argparse
import sys
from math import sqrt
import queue as lib_queue


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action='store_true', help="verbose mode")
    parser.add_argument("-mode",
                        help="Program's mode",
                        choices=['cnf', 'dpll', 'solver'])
    parser.add_argument(
        'input_file',
        help='A mode-dependent input file',
        default='ex1.txt',
    )
    return parser.parse_args()


def path_to_str(*path):
    return ' -> '.join(path)


class Graph:
    def __init__(self, vertices=dict(), edges=dict(), adj=dict()) -> None:
        self.vertices = vertices
        self.edges = edges
        self.adj = adj

    def euclidean_distance(self, vertex_1, vertex_2):
        (px, py) = self.vertices[vertex_1]
        (qx, qy) = self.vertices[vertex_2]
        return sqrt((px - qx)**2 + (py - qy)**2)

    def _store_edge_cost(self, edge):
        i = iter(edge)
        return self.euclidean_distance(next(i), next(i))

    def add_vertex(self, *vertices):
        for v in vertices:
            self.vertices[v[0]] = [int(x) for x in v[1:]]
            # Initialize adjacency matrix
            self.adj[v[0]] = set()

    def add_edge(self, *edges):
        for e in edges:
            self.edges[e] = self._store_edge_cost(e)
            # Update adjacency matrix
            p, q = tuple(e)
            self.adj[p].update(q)
            self.adj[q].update(p)

    def get_edge_cost(self, *vertices):
        return self.edges[frozenset(vertices)]

    def __str__(self) -> str:
        return f'Vertices: {self.vertices}\n\nEdges: {self.edges}\n\nAdjacency Matrix: {self.adj}'


def parse_graph_file(filename):
    G = Graph()
    with open(filename) as f:
        lines = f.readlines()
        # Temp set to store edges
        edges = set()
        for line in lines:
            line = line.strip()
            # Skip empty lines & comments
            if not line or line[0] == '#' or len(line) < 3:
                continue
            else:
                items = line.split(' ')
                if len(items) == 2:  # Edge
                    edges.add(frozenset(items))
                if len(items) == 3:  # Vertex
                    G.add_vertex(tuple(items))
        # Add edges after vertices
        G.add_edge(*edges)
    return G


if __name__ == '__main__':
    args = get_args()
    G = parse_graph_file(args.graph_file)

    if args.alg == 'BFS':
        res = G.BFS(args.start, args.end, verbose=args.v)
    elif args.alg == 'ID':
        res = G.ID(args.start,
                   args.end,
                   initial_depth=args.depth,
                   verbose=args.v)
    elif args.alg == 'ASTAR':
        res = G.ASTAR(args.start, args.end, verbose=args.v)