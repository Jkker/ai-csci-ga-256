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
