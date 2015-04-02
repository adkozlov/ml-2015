#!/usr/bin/env python3

__author__ = 'Andrew Kozlov'
__copyright__ = 'Copyright 2015, SPbAU'

from matplotlib.mlab import find
import numpy as np
import matplotlib.pyplot as pl
from sklearn.datasets import make_blobs
from cvxopt import matrix, solvers


def visualize(clf, x, y):
    border = .5
    h = .02

    x_min, x_max = x[:, 0].min() - border, x[:, 0].max() + border
    y_min, y_max = x[:, 1].min() - border, x[:, 1].max() + border

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    z_class = clf.predict(mesh).reshape(xx.shape)

    # Put the result into a color plot
    pl.figure(1, figsize=(8, 6))
    pl.pcolormesh(xx, yy, z_class, cmap=pl.cm.summer, alpha=0.3)

    # Plot hyperplane and margin
    z_dist = clf.decision_function(mesh).reshape(xx.shape)
    pl.contour(xx, yy, z_dist, [0.0], colors='black')
    pl.contour(xx, yy, z_dist, [-1.0, 1.0], colors='black', linestyles='dashed')

    # Plot also the training points
    y_pred = clf.predict(x)

    ind_support = clf.support_
    ind_correct = list(set(find(y == y_pred)) - set(ind_support))
    ind_incorrect = list(set(find(y != y_pred)) - set(ind_support))

    pl.scatter(x[ind_correct, 0], x[ind_correct, 1], c=y[ind_correct], cmap=pl.cm.summer, alpha=0.9)
    pl.scatter(x[ind_incorrect, 0], x[ind_incorrect, 1], c=y[ind_incorrect], cmap=pl.cm.summer, alpha=0.9, marker='*',
               s=50)
    pl.scatter(x[ind_support, 0], x[ind_support, 1], c=y[ind_support], cmap=pl.cm.summer, alpha=0.9, linewidths=1.8,
               s=40)

    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())


class LinearSVM:
    def __init__(self, c, eps=1e-4):
        self.c = c
        self.eps = eps

        self.w = np.zeros(1)
        self.b = 0

        self.support_ = []

    def fit(self, x, y):
        a = self.__solve(x, y)

        self.support_ = np.arange(len(a))[a > self.eps]
        self.w = np.sum(x * np.expand_dims(a * y, 1), axis=0)

        for (i, row) in enumerate(x):
            if self.eps < a[i] < self.c:
                self.b = y[i] - np.dot(self.w, row)
                break

    def __solve(self, x, y):
        solution = solvers.qp(*(self.__create_matrices(x, y)))
        return np.reshape(np.array(solution['x']), (len(x),))

    def __create_matrices(self, x, y):
        p = x * np.expand_dims(y, 1)
        p = np.dot(p, np.transpose(p))

        n = len(x)
        return [matrix(m) for m in [p,
                                    -np.ones((n, 1)),
                                    np.vstack((np.eye(n), -np.eye(n))),
                                    np.vstack((np.ones((n, 1)) * self.c, np.zeros((n, 1)))),
                                    np.reshape(y, (1, n)),
                                    np.zeros(1)]]

    def decision_function(self, x):
        return (np.dot(x, self.w) + self.b) / np.linalg.norm(self.w)

    def predict(self, x):
        return np.sign(self.decision_function(x))


class KernelSVM:
    def __init__(self, c, kernel=None, sigma=1.0, degree=2, eps=1e-4):
        self.c = c
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.eps = eps

        self.w = np.zeros(1)
        self.b = 0
        self.kernel_is_linear_ = kernel == linear_kernel

        self.support_ = []
        self.positive_a_ = []
        self.positive_y_ = []
        self.positive_x_ = []

    def fit(self, x, y):
        k = self.__create_kernel(x)
        a = self.__solve(x, y, k)

        positive_filter = a > self.eps
        self.support_ = np.arange(len(a))[positive_filter]
        self.positive_a_ = a[positive_filter]
        self.positive_y_ = y[positive_filter]
        self.positive_x_ = x[positive_filter]

        if self.kernel_is_linear_:
            self.w = np.zeros(x.shape[1])
            for (a_value, y_value, x_value) in self.__get_zipped():
                self.w += a_value * y_value * x_value

        self.b = np.mean(self.positive_y_) - np.sum(self.positive_a_ * self.positive_y_ *
                                                    [k[row, positive_filter] for row in self.support_]) / len(
            self.positive_a_)

    def __get_zipped(self):
        return zip(self.positive_a_, self.positive_y_, self.positive_x_)

    def __solve(self, x, y, k):
        solution = solvers.qp(*(self.__create_matrices(x, y, k)))
        return np.reshape(np.array(solution['x']), (len(x),))

    def __create_matrices(self, x, y, k):
        n = len(x)
        return [matrix(m) for m in [np.outer(y, y) * k,
                                    -np.ones((n, 1)),
                                    np.vstack((np.diag(-np.ones(n)), np.identity(n))),
                                    np.hstack((np.zeros(n), np.ones(n) * self.c)),
                                    np.reshape(y, (1, n)),
                                    np.zeros(1)]]

    def __create_kernel(self, x):
        if self.kernel == gaussian_kernel:
            n = len(x)
            result = np.zeros((n, n))
            for (i, a) in enumerate(x):
                for (j, b) in enumerate(x):
                    result[i, j] = self.__get_value(a, b)
            return result
        else:
            return self.__get_value(x, x.T)

    def decision_function(self, x):
        if self.kernel_is_linear_:
            return np.dot(x, self.w) + self.b
        else:
            result = np.zeros(x.shape[0])
            for (i, row) in enumerate(x):
                for (a_value, y_value, x_value) in self.__get_zipped():
                    result[i] += a_value * y_value * self.__get_value(row, x_value)

            return result + self.b

    def __get_value(self, x, y):
        if self.kernel == gaussian_kernel:
            return gaussian_kernel(x, y, self.sigma)
        elif self.kernel == polynomial_kernel:
            return polynomial_kernel(x, y, self.degree)
        else:
            return self.kernel(x, y)

    def predict(self, x):
        return np.sign(self.decision_function(x))


def linear_kernel(x, y):
    return np.dot(x, y)


def gaussian_kernel(x, y, sigma):
    return np.exp((-np.linalg.norm(x - y) ** 2) / (2 * (sigma ** 2)))


def polynomial_kernel(x, y, degree):
    return (np.dot(x, y) + 1) ** degree


def generate_blobs(n_features=2, random_state=0):
    x, y = make_blobs(n_features=n_features, centers=2, random_state=random_state)
    y[y == 0] = -1
    y = y.astype(float, copy=False)
    return x, y


def run_test(kernel, train_set, test_set, file_name=None, **kwargs):
    svm = KernelSVM(2, kernel, **kwargs) if kernel else LinearSVM(2)
    svm.fit(*train_set)

    visualize(svm, test_set[0], test_set[1])
    if file_name:
        pl.savefig(file_name)
    else:
        pl.show()


if __name__ == '__main__':
    train = generate_blobs()
    test = generate_blobs()

    # run_test(None, train, test, 'linear_svm')
    # run_test(linear_kernel, train, test, 'kernel_svm-linear')
    # run_test(polynomial_kernel, train, test, 'kernel_svm-polynomial_degree_2')
    # run_test(polynomial_kernel, train, test, 'kernel_svm-polynomial', degree=3)
    run_test(gaussian_kernel, train, test, 'kernel_svm-gaussian_sigma_1')
    # run_test(gaussian_kernel, train, test, 'kernel_svm-gaussian_sigma_3', sigma=3)