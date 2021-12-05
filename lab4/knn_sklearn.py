from sklearn.neighbors import KNeighborsClassifier
from utils import parse_input


def print_metrics(y_pred, y_test, y_train):
    labels = set(y_train)

    n = len(y_test)
    for l in sorted(list(labels)):
        correct = sum(1 if y_pred[i] == y_test[i] == l else 0
                      for i in range(n))
        print(
            f'Label={l} Precision={correct}/{list(y_pred).count(l)} Recall={correct}/{list(y_test).count(l)}'
        )


def eval(y_pred, y_test):
    for p, t in zip(y_pred, y_test):
        print(f'want={t} got={p}')


def sklearn_knn(k, train, test, dist_func='e2', unitw=False):
    model = KNeighborsClassifier(
        n_neighbors=k,
        algorithm='brute',
        metric='euclidean' if dist_func == 'e2' else 'manhattan',
        weights='uniform' if unitw else 'distance')

    d_train = parse_input(train)
    d_test = parse_input(test)

    model.fit(d_train[0], d_train[1])
    y_pred = model.predict(d_test[0])
    eval(y_pred, d_test[1])
    print_metrics(y_pred, d_test[1], d_train[1])
    return y_pred
