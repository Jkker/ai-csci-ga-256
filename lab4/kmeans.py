import argparse
import numpy as np
from utils import manhatton, euclidean, parse_input


class kMeans:
    def __init__(self, dist_func="e2", verbose=False):
        self.dist_func = manhatton if dist_func == 'manh' else euclidean
        self.clusters = None
        self.K = 0
        self.verbose = verbose

    def fit(self, data, init_centroids, max_iters=20):
        self.centroids = np.asarray(init_centroids)
        self.K = self.centroids.shape[0]
        self.data = data

        clusters = np.zeros(data.shape[0])
        for i in range(max_iters):
            dist = self.compute_distances()
            clusters = np.argmin(dist, axis=1)
            centroids = self.compute_centroids(clusters)

            # no change in centroids
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
        centroids = np.zeros((self.K, self.data.shape[1]))
        n = dict((i, 0) for i in range(self.K))
        for point, label in zip(self.data, clusters):
            centroids[label] += point
            n[label] += 1

        for k in range(self.K):
            if n[k] != 0:
                centroids[k] /= n[k]
        return centroids

    def compute_distances(self):
        N = self.data.shape[0]
        dist = np.ndarray((N, self.K))
        for n in range(N):
            for k in range(self.K):
                dist[n, k] = self.dist_func(self.data[n], self.centroids[k])
        return dist

    def print_clusters(self, labels):
        if self.verbose: print('Clusters:')
        for i in range(self.K):
            print(f'C{i} = ' + '{' +
                  ','.join(labels[self.clusters == i]) + '}')
        if self.verbose: print()

    def print_centroids(self):
        if self.verbose: print('Centroids:')
        print(
            np.array2string(self.centroids,
                            precision=4,
                            separator=' ',
                            suppress_small=True))
        if self.verbose: print()

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
    init_centroids = [tuple(map(float, x.split(','))) for x in args.init_centroids]
    data, data_labels = parse_input(args.data)

    model = kMeans(dist_func=args.d, verbose=args.v)
    model.fit(data, init_centroids)
    model.print_clusters(data_labels)
    model.print_centroids()
    print()
