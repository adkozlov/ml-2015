#!/usr/bin/env python3

__author__ = 'Andrew Kozlov'
__copyright__ = 'Copyright 2015, SPbAU'

import sys
import random
import math
import operator


def read_data(file_name):
    X = []
    y = []
    with open(file_name) as file:
        for line in file.readlines():
            if line.startswith('#'):
                continue

            numbers = line.strip('\n').split(',')
            X.append(tuple(map(float, numbers[1:])))
            y.append(int(numbers[0]))

    return X, y


def train_test_split(X, y, ratio):
    data = list(zip(X, y))
    random.shuffle(data)

    bound = int(ratio * len(data))
    train = data[:bound]
    test = data[bound:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return list(X_train), list(y_train), list(X_test), list(y_test)


def sum_values(first, second, func):
    return sum([func(x, y) for (x, y) in list(zip(first, second))])


def euclidean_distance(first, second):
    return math.sqrt(sum_values(first, second, lambda x, y: (x - y) ** 2))


def cosine_similarity(first, second):
    func = lambda x, y: x * y

    first_norm = sum_values(first, first, func)
    second_norm = sum_values(second, second, func)
    return sum_values(first, second, func) / math.sqrt(first_norm * second_norm)


def knn(X_train, y_train, X_test, k=1, dist=euclidean_distance):
    train = list(zip(X_train, y_train))

    def inner_knn(test):
        neighbors = [((x, y), dist(x, test)) for (x, y) in train]

        neighbors.sort(key=operator.itemgetter(1))
        result, _ = zip(*neighbors[:k])

        return result

    result = []
    for test in X_test:
        votes = {}
        for neighbor in inner_knn(test):
            vote = neighbor[-1]
            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1

        result.append(sorted(votes.items(), key=operator.itemgetter(1), reverse=True)[0][0])

    return result


def print_precision_recall(y_pred, y_test):
    values = list(zip(y_pred, y_test))

    def calculate_precision_recall(class_index):
        tp = 0
        fp = 0
        fn = 0
        for (predicted, actual) in values:
            if predicted == class_index and actual == class_index:
                tp += 1
            if predicted == class_index and actual != class_index:
                fp += 1
            if predicted != class_index and actual == class_index:
                fn += 1

        def calculate_fraction(denominator):
            return tp / denominator if denominator != 0 else 0

        return calculate_fraction(tp + fp), calculate_fraction(tp + fn)

    for clazz in set(y_test):
        precision, recall = calculate_precision_recall(clazz)
        print('%d %.2f %.2f' % (clazz, precision, recall))


def loocv(X_train, y_train, X_test, y_test, dist=euclidean_distance):
    X_data = [] + X_train + X_test
    y_data = [] + y_train + y_test
    length = len(X_data)

    def remove_element(list, i):
        return list[:i] + list[i + 1:]

    def count_errors(k):
        return sum([1 for i in range(length) if knn(
            remove_element(X_data, i), remove_element(y_data, i), [X_data[i]], k, dist)[0] != y_data[i]])

    errors = [(k, float(count_errors(k)) / length) for k in range(1, length)]
    errors.sort(key=operator.itemgetter(1))

    return errors[0][0]


if __name__ == '__main__':
    X, y = read_data(sys.argv[1])
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.67)

    print('euclidean distance:')
    print_precision_recall(knn(X_train, y_train, X_test, loocv(X_train, y_train, X_test, y_test)), y_test)

    print('cosine similarity:')
    print_precision_recall(knn(X_train, y_train, X_test, loocv(X_train, y_train, X_test, y_test), cosine_similarity), y_test)