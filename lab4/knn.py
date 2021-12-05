import argparse
import numpy as np
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
        y_pred = []

        for X_test in test[0]:
            dist = np.ndarray((train[0].shape[0],))

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



if __name__ == '__main__':
    args = get_args()
    X_train, y_train = parse_input(args.train)
    X_test, y_test = parse_input(args.test)
    model = KNN(args.k, args.train, args.test, args.d, args.unitw)
    model.eval()