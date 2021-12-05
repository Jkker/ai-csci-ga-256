import argparse
import numpy as np
import pandas as pd
from utils import manhatton, euclidean, parse_input
import matplotlib.pyplot as plt

# Argument & Input File Parser
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        type=str,
                        default='e2',
                        help='Distance function to use',
                        choices=['e2', 'manh'])

    parser.add_argument('-train',
                        type=str,
                        help='a file containing training data')

    parser.add_argument('-test',
                        type=str,
                        help='a file containing points to test')

    parser.add_argument(
        '-unitw',
        action='store_true',
        default=False,
        help='whether to use unit voting weights; if not, set use 1/d weights')

    parser.add_argument('-k',
                        type=int,
                        default=3,
                        help='number of nearest neighbors to use')

    parser.add_argument('-v',
                        action='store_true',
                        default=False,
                        help='verbose')

    args = parser.parse_args()

    return args

class KNN:
    def __init__(self, k, train, test, dist_func='e2', unitw=False):
        self.k = k
        self.unitw = unitw
        self.dist_func = manhatton if dist_func == 'manh' else euclidean
        self.train = parse_input(train)
        self.test = parse_input(test)
        self.all_labels = set(self.train[1]) | set(self.test[1])

    def _vote(self, k_labels, k_dists):
        record = dict()

        for dist, label in zip(k_dists, k_labels):
            if self.unitw:
                w = 1
            else:
                w = 1 / dist if dist != 0 else 1/0.0001

            record[label] = record.get(label, 0) + w

        return max(record, key=record.get)

    def predict(self):
        train = self.train
        test = self.test
        n_train = train[0].shape[0]
        y_pred = []

        for X_test in test[0]:
            dist = np.ndarray((n_train,))

            for i, X_train in enumerate(train[0]):
                dist[i] = self.dist_func(X_test, X_train)

            k_indices = dist.argsort()[:self.k]
            k_labels = train[1][k_indices]
            k_dists = dist[k_indices]

            pred = self._vote(k_labels, k_dists)

            y_pred.append(pred)

        return y_pred

    def print_metrics(self, y_pred):
        y_test= self.test[1]
        for l in sorted(list(self.all_labels)):
            n_correct = sum(1 if pred == test == l else 0
                        for pred,test in zip(y_pred, y_test))
            n_predicted = list(y_pred).count(l)
            n_actual = list(y_test).count(l)

            print(
                f'Label={l} Precision={n_correct}/{n_predicted} Recall={n_correct}/{n_actual}'
            )

    def eval(self):
        y_pred = self.predict()
        for p, t in zip(y_pred, self.test[1]):
            print(f'want={t} got={p}')
        self.print_metrics(y_pred)


def knn(X_trainset,
        y_trainset,
        X_testset,
        y_testset,
        k,
        dist_func='e2',
        unitw=False,
        verbose=False):

    d = manhatton if dist_func == 'manh' else euclidean

    y_pred = []
    k_nearests = []

    for i_test, X_test in enumerate(X_testset):
        # Compute distances to all training points
        distances = []
        for X_train, y_train in zip(X_trainset, y_trainset):
            # for index, p_train in enumerate(X_train):
            distances.append(d(X_test, X_train), X_test, X_train, y_train)
            # train index, distance, train label
            # distances.append((index, d(X_test, p_train), y_trainset[index]))

        # Sort by distance
        # k_nearest = sorted(distances, key=lambda x: x[1])[:k]
        k_nearest = sorted(distances, key=lambda x: x[0])[:k]

        if verbose:
            print('🔰', X_test)
            for dist, label in k_nearest:
                print(f'{X_trainset[idx]} {y_trainset[idx]} {np.round(dist, 2)}')


            for idx, dist, label in k_nearest:
                print(f'{X_trainset[idx]} {y_trainset[idx]} {np.round(dist, 2)}')

        # res = np.array([
        #     np.argmax(np.bincount(y_train[neighbor])) for (neighbor, _, __) in k_nearest
        # ])
        # print(res)

        # Vote
        vote = dict()

        for idx, dist, label in k_nearest:
            if not unitw: vote[label] = vote.get(label, 0) + 1 / (0.0001 + dist)
            else: vote[label] = vote.get(label, 0) + 1
        y_pred.append(max(vote, key=vote.get))
        # print(k_nearest)
        # print(vote)

        print(f'want={y_testset[i_test]} got={y_pred[i_test]}')
        k_nearests.append(k_nearest)

    print_metrics(y_testset, y_pred, y_trainset)
    # print()

    return y_pred, k_nearests


def print_metrics(y_test, y_pred, y_train):
    labels = set(y_train)

    n = len(y_test)
    for l in sorted(list(labels)):
        correct = sum(1 if y_pred[i] == y_test[i] == l else 0
                      for i in range(n))
        print(
            f'Label={l} Precision={correct}/{list(y_pred).count(l)} Recall={correct}/{list(y_test).count(l)}'
        )

colors = {'A': 'red', 'B': 'orange', 'C': 'blue'}


if __name__ == '__main__':
    args = get_args()

    # print(args)
    X_train, y_train = parse_input(args.train)
    X_test, y_test = parse_input(args.test)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=[colors[y] for y in y_train], alpha=0.3, marker='s')
    # plt.show()
    # y_pred, k_nearests = knn(X_train,
    #                          y_train,
    #                          X_test,
    #                          y_test,
    #                          k=args.k,
    #                          dist_func=args.d,
    #                          verbose=args.v)

    model = KNN(args.k, args.train, args.test, args.d, args.unitw)
    model.eval()

    # XX = np.asarray([X_train[i] for (i, _, __) in k_nearests[0]])
    # i = 3
    # KNN
    # plt.scatter(XX[:, 0],
    #             XX[:, 1],
    #             edgecolors='black',
    #             marker='X')
    # # # ALL TEST
    # # plt.scatter(
    # #     X_test[:, 0],
    # #     X_test[:, 1],
    # #     c=[colors[y] for y in y_pred],
    # #     alpha=0.5,
    # # )
    # # CURRENT TEST
    # plt.scatter(X_test[i, 0],
    #             X_test[i, 1],
    #             c='red',
    #             edgecolors=colors[y_test[i]],
    #             alpha=0.5,
    #             s=100)
    # plt.show()
