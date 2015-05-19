#!/usr/bin/env python3
from time import time
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize


__author__ = 'Andrew Kozlov'
__copyright__ = 'Copyright 2015, SPbAU'


def read_data(file_name, norm=True):
    with open(file_name) as file:
        array = np.asarray([list(map(float, line.split(','))) for line in file.readlines()[15:]])
        x = array[:, :-1]
        return normalize(x) if norm else x, array[:, -1]


class NormalLR:
    def __init__(self):
        self.weights = np.zeros(0)

    def fit(self, x, y):
        self.weights = np.dot(np.linalg.pinv(x), y)
        return self

    def predict(self, x):
        return np.dot(x, self.weights)


def sample(size, *, weights):
    x = np.ones((size, 2))
    x[:, 1] = np.random.gamma(4., 2., size)
    y = x.dot(np.asarray(weights))
    y += np.random.normal(0, 1, size)
    return x[:, 1:], y


def plot(lr, size=100):
    x, y_actual = sample(size, weights=[24., 42.])
    lr.fit(x, y_actual)
    plt.scatter(x, y_actual)
    plt.plot(x, lr.predict(x), color="red")
    plt.show()


class GradientLR(NormalLR):
    def __init__(self, *, alpha=1e-3):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        self.alpha = alpha
        self.threshold = alpha / 100
        self.max_iterations_count = 10 ** 3

    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        for _ in range(self.max_iterations_count):
            weights = self.weights.copy()
            self.weights = weights - self.alpha * np.dot(np.transpose(np.dot(x, weights) - y), x) / x.shape[0]

            if np.linalg.norm(self.weights - weights) < self.threshold:
                break
        return self


def mse(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)


def performance_test(constructor, iterations_count=100):
    def mean(xs):
        return np.mean(np.array(xs, dtype=float))

    for size in [2 ** power for power in range(7, 11)]:
        timestamps = []
        mses = []
        for i in range(iterations_count):
            x, y = sample(size, weights=[24., 42.])
            lr = constructor()

            timestamp = time()
            lr.fit(x, y)
            mses.append(mse(y, lr.predict(x)))
            timestamps.append(time() - timestamp)

        print('size: %d, mse: %.2f, time: %.6f μs' % (size, mean(mses), mean(timestamps) * 10 ** 6))


def calculate_mse(constructor, x, y, train_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
    lr = constructor()

    lr.fit(x_train, y_train)
    return mse(y_test, lr.predict(x_test))


def calculate_mean_mse(constructor, x, y, iterations_count=100):
    mses = np.zeros(iterations_count)
    for i in range(iterations_count):
        mses[i] = calculate_mse(constructor, x, y)
    return np.mean(mses)


if __name__ == '__main__':
    # data = read_data(sys.argv[1], False)
    # nlr = NormalLR()
    # nlr.fit(*data)
    # print(nlr.weights)

    data = read_data(sys.argv[1])

    def run_tests(is_normal):
        print('normal' if is_normal else 'gradient')
        cons = NormalLR if is_normal else GradientLR
        performance_test(cons)
        print('boston data mse: %.f' % calculate_mean_mse(cons, *data))

    run_tests(True)
    run_tests(False)