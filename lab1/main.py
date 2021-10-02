import argparse
import sys
from math import sqrt
import queue as lib_queue


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
        type=int,
        help=
        "Used only for iterative-deepening, that indicates the initial search depth, with a default increase of 1 after that"
    )
    parser.add_argument(
        'graph_file',
        help='Graph File Input',
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

    def BFS(self, start_vertex, end_vertex, verbose=False):
        visited = dict()
        queue = lib_queue.Queue()
        visited[start_vertex] = 0
        queue.put([start_vertex])

        while queue:
            path = queue.get()
            vertex = path[-1]

            if vertex == end_vertex:  # Search complete
                print('Solution:', path_to_str(*path))
                return path

            if verbose: print(f'Expanding: {vertex}')

            for next_vertex in sorted(self.adj[vertex]):
                if next_vertex not in visited:
                    weight = self.get_edge_cost(vertex, next_vertex)
                    queue.put([*path, next_vertex])
                    visited[next_vertex] = weight

        return visited

    def ID(self, start_vertex, end_vertex, initial_depth, verbose=False):
        def depth_limited_search(depth):
            stack = [[start_vertex]]
            visited = set({start_vertex})

            while stack:
                path = stack.pop()
                vertex = path[-1]

                if vertex == end_vertex:
                    return path

                visited.add(vertex)

                # hit depth -> skip expansion
                if len(path) > depth:
                    print(f'hit depth={len(path)-1}: {vertex}')
                    continue

                if verbose:
                    print(f'Expanding: {vertex}')
                # reversed due to LIFO stack
                for next_vertex in reversed(sorted(self.adj[vertex])):
                    if next_vertex not in visited:
                        stack.append([*path, next_vertex])

        for depth in range(initial_depth, sys.maxsize):
            result_path = depth_limited_search(depth)
            if result_path:
                print('Solution:', path_to_str(*result_path))
                return result_path

    def ASTAR(self, start_vertex, end_vertex, verbose=False):

        visited_cost = {start_vertex: 0}
        pq = lib_queue.PriorityQueue()

        pq.put((0, [start_vertex]))

        while not pq.empty():
            (_, curr_path) = pq.get()
            print(f'Adding {path_to_str(*curr_path)}')
            curr_vertex = curr_path[-1]

            for next_vertex in sorted(self.adj[curr_vertex]):

                # Loop detection
                if next_vertex in curr_path:
                    continue

                # cost to visit next
                g = visited_cost[curr_vertex] + self.get_edge_cost(
                    curr_vertex, next_vertex)

                # heuristic cost
                h = self.euclidean_distance(next_vertex, end_vertex)

                path = [*curr_path, next_vertex]

                if verbose:
                    print(
                        f'{path_to_str(*path)}; g={g:.2f} h={h:.2f} = {g+h:.2f}'
                    )

                if next_vertex == end_vertex:
                    print('Solution:', path_to_str(*path))
                    return path

                if g not in visited_cost or visited_cost[next_vertex] > g:
                    visited_cost[next_vertex] = g
                    pq.put((h + g, path))


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