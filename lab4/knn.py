import argparse
import numpy as np
import pandas as pd
from utils import manhatton, euclidean, parse_input


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

def knn(X_train,
        y_train,
        X_test,
        y_test,
        k,
        dist_func='e2',
        unitw=False,
        verbose=False):
    d = manhatton if dist_func == 'manh' else euclidean
    pred=[]
    for i, p_test in enumerate(X_test):
        distances = []
        for j, p_train in enumerate(X_train):
            distances.append((j, d(p_test, p_train)))

        k_nearest = sorted(distances, key=lambda x: x[1])[:k]
        # print('k_nearest', k_nearest)
        k_nearest_labels = [y_train[i] for i, _ in k_nearest]
        if verbose: print(p_test)
        for l, (ii, dd) in zip(k_nearest_labels, k_nearest):
            if verbose: print(l, np.round(dd, 3))

        pred.append(max(k_nearest_labels, key=k_nearest_labels.count))

        print(f'want={y_test[i]} got={pred[i]}')

    return pred



if __name__ == '__main__':
    args = get_args()

    # print(args)
    X_train, y_train = parse_input(args.train)
    X_test, y_test = parse_input(args.test)

    labels = set(y_train)

    y_pred = knn(X_train,
                 y_train,
                 X_test,
                 y_test,
                 k=args.k,
                 dist_func=args.d,
                 verbose=args.v)
    n = len(y_test)
    for l in sorted(list(labels)):
        correct = sum(1 if y_pred[i] == y_test[i] == l else 0
                      for i in range(n))
        print(
            f'Label={l} Precision={correct}/{list(y_pred).count(l)} Recall={correct}/{list(y_test).count(l)}'
        )
