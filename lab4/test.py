import argparse
import os
from utils import parse_input
from knn import knn, print_metrics
from io import StringIO
import sys


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def test_knn(path, v=False, n=None):
    l = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    l = [f for f in l if 'knn' in f]
    train_sets = [f for f in l if 'train' in f]
    test_sets = [f for f in l if 'test' in f]

    proj = ['knn1', 'knn2', 'knn3']
    # proj = ['knn1', 'knn2']
    if n is not None:
        proj = [proj[n-1]]
    for p in proj:
        failed = []
        passed = []
        ptrain = [f for f in train_sets if p in f][0]
        ptest = [f for f in test_sets if p in f][0]
        X_train, y_train = parse_input(os.path.join('data', ptrain))
        X_test, y_test = parse_input(os.path.join('data', ptest))

        tasks = [f for f in l if 'out' in f and p in f]

        for task in tasks:
            if v: print('\n\n🔰 ' + task + '\n')

            dist_func = task.split('.')[1]
            unitw = 'unit' in task
            k = int(task.split('.')[3] if unitw else task.split('.')[2])

            y_pred, logs = run(X_train, y_train, X_test, y_test, k, dist_func,
                               unitw, False)
            test_passed = diff(os.path.join('data', task), logs, v)
            if not test_passed:
                failed.append(task)
            else:
                passed.append(task)

        if failed: print(f'\n🌋 {p.upper()} FAILED ({len(failed)}/{len(tasks)}): ' +
              ', '.join(failed) + '\n')
        else: print(f'\n✅ {p.upper()} PASSED')


def diff(filename, logs, v=False):
    same = True
    with open(filename, 'r') as f:
        for a, b in zip(f.readlines(), logs):
            if a.strip() != b.strip():
                same = False
                if v:
                    print(b.replace('\n', ''), '❗Actual:', a.replace('\n', ''))
            else:
                if v: print(a, end='')
    return same


def run(*args):
    with Capturing() as output:
        ret = knn(*args)
    return ret, output


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-n', type=int, default=None)


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    test_knn('data', args.v, args.n)