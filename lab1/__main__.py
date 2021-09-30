import argparse
from math import sqrt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action='store_true', help="verbose mode")
    parser.add_argument("-start", help="Name of the node to start from")
    parser.add_argument("-end", help="Name of the node which is the goal")
    parser.add_argument("-alg",
                        help="Algorithm to use",
                        choices=['BFS', 'ID', 'ASTAR'])
    parser.add_argument(
        "-depth",
        help=
        "Used only for iterative-deepening, that indicates the initial search depth, with a default increase of 1 after that"
    )
    parser.add_argument('graph_file', help='Graph File Input')
    return parser.parse_args()


class Graph:
    def __init__(self, vertices=dict(), edges=dict(), adj=dict()) -> None:
        self.vertices = vertices
        self.edges = edges
        self.adj = adj

    def compute_edge_length(self, edge):
        i = iter(edge)
        (px, py) = self.vertices[next(i)]
        (qx, qy) = self.vertices[next(i)]
        return sqrt((px - qx)**2 + (py - qy)**2)

    def add_vertex(self, *vertices):
        for v in vertices:
            self.vertices[v[0]] = [int(x) for x in v[1:]]
            self.adj[v[0]] = set()

    def add_edge(self, *edges):
        for e in edges:
            self.edges[e] = self.compute_edge_length(e)
            p, q = tuple(e)
            self.adj[p].update(q)
            self.adj[q].update(p)

    def get_edge_weight(self, *vertices):
        return self.edges[frozenset(vertices)]

    def __str__(self) -> str:
        return f'Vertices: {self.vertices}\n\nEdges: {self.edges}\n\nAdjacency Matrix: {self.adj}'

    def BFS(self, startVertex, endVertex):
        visited = dict()
        queue = []
        visited[startVertex] = 0
        queue.append(startVertex)

        while queue:
            path = queue.pop(0)
            vertex = path[-1]

            if vertex == endVertex:  # Search complete
                print('Solution:', ' -> '.join(path))
                return path

            print(f'Expanding: {vertex}')

            for adjacent_vertex in sorted(self.adj[vertex]):
                if adjacent_vertex not in visited:
                    weight = self.get_edge_weight(vertex, adjacent_vertex)
                    queue.append([*path, adjacent_vertex])
                    visited[adjacent_vertex] = weight

        return visited

    def ID(self):
        pass

    def ASTAR(self):
        pass


def parse_graph_file(filename):
    G = Graph()
    with open(filename) as f:
        lines = f.readlines()
        edges = set()
        for line in lines:
            line = line.strip()
            if not line or line[0] == '#' or len(line) < 3:
                continue
            else:
                items = line.split(' ')
                if len(items) == 2:
                    edges.add(frozenset(items))
                if len(items) == 3:
                    G.add_vertex(tuple(items))
        G.add_edge(*edges)
    return G


if __name__ == '__main__':
    args = get_args()
    G = parse_graph_file(args.graph_file)
    print(args)
    # start, end, alg, depth = args
    # start, end, alg,
    if args.alg == 'BFS':
        res = G.BFS(args.start, args.end)