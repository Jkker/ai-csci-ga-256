import numpy as np
import csv
import sys


def manhatton(x, y):
    return np.sum(np.abs(x - y))


def euclidean(x, y):
    return np.linalg.norm(x - y)


def parse_input(input_file):
    X = []
    y = []
    with open(input_file, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            X.append(lines[:-1])
            y.append(lines[-1])

    return np.asarray(X, dtype=np.float64), np.asarray(y)