import argparse
from copy import deepcopy
import random
import numpy as np
import pandas as pd


def manhatton(x, y):
    return np.sum(np.abs(x - y))


def euclidean(x, y):
    return np.linalg.norm(x - y)


def parse_input(input_file):
    df = pd.read_csv(input_file, header=None)
    values = np.asarray(df.iloc[:, :-1].values)
    names = np.asarray(df.iloc[:, -1].values)
    return values, names


class KMeans:
    def __init__(self, data, init_centroids, dist_func="e2", verbose=False):
        self.centroids = np.asarray(init_centroids)
        self.values = data[0]
        self.labels = data[1]
        self.K = self.centroids.shape[0]
        self.N = self.values.shape[0]
        self.dist_func = manhatton if dist_func == 'manh' else euclidean
        self.clusters = None
        self.verbose = verbose

    def fit(self, max_iters=20):
        clusters = np.zeros(self.N)
        centroids = self.centroids
        for i in range(max_iters):
            dist = self.compute_distances()
            clusters = np.argmin(dist, axis=1)
            centroids = self.compute_centroids(clusters)
            if (centroids == self.centroids).all():
                if self.verbose: print('Converged at i =', i)
                clusters = clusters.astype(int)
                self.clusters = clusters
                self.centroids = centroids
                return clusters
            self.centroids = centroids
        if self.verbose: print('Max iteration count reached')

        clusters = clusters.astype(int)
        self.clusters = clusters
        self.centroids = centroids
        return clusters

    def compute_centroids(self, clusters):
        centroids = np.zeros((self.K, self.values.shape[1]))
        n = dict((i, 0) for i in range(self.K))
        for point, label in zip(self.values, clusters):
            centroids[label] += point
            n[label] += 1

        for k in range(self.K):
            if n[k] != 0:
                centroids[k] /= n[k]
        return centroids

    def compute_distances(self):
        dist = np.ndarray((self.N, self.K))
        for n in range(self.N):
            for k in range(self.K):
                dist[n, k] = self.dist_func(self.values[n], self.centroids[k])
        return dist

    def print_clusters(self):
        if self.verbose: print('Clusters:')
        for i in range(self.K):
            print(f'C{i} = ' + '{' +
                  ','.join(self.labels[self.clusters == i]) + '}')
        if self.verbose: print()

    def print_centroids(self):
        if self.verbose: print('Centroids:')
        print(
            np.array2string(self.centroids,
                            precision=4,
                            separator=' ',
                            suppress_small=True))
        if self.verbose: print()

    def print_params(self):
        print('\nDistance function:', self.dist_func.__name__)
        print('K =', self.K)
        print('N =', self.N)
        print()

    def print(self):
        if self.verbose: self.print_params()
        self.print_clusters()
        self.print_centroids()
        print()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--distance-function',
                        type=str,
                        dest='d',
                        default='e2',
                        help='Distance function to use',
                        choices=['e2', 'manh'])

    parser.add_argument('-v', action='store_true', default=False)

    parser.add_argument(
        '-data',
        help='input data file',
        default='ex1.txt',
    )
    parser.add_argument('init_centroids',
                        nargs='+',
                        type=str,
                        help='initial centroids')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    init = [tuple(map(float, x.split(','))) for x in args.init_centroids]
    data = parse_input(args.data)

    model = KMeans(data, init, dist_func=args.d, verbose=args.v)
    model.fit()
    model.print()
