#!/usr/bin/env python3

__author__ = 'Andrew Kozlov'
__copyright__ = 'Copyright 2015, SPbAU'

import operator


def read_data(file_name):
    with open(file_name) as file:
        return [tuple(map(float, pair)) for pair in [line.strip('\n').split('\t') for line in file.readlines()]]


def get_range(bound_inclusive, origin_inclusive=0):
    return range(origin_inclusive, bound_inclusive + 1)


def create_system(data, power):
    def create_matrix():
        x_data, _ = zip(*data)

        def create_row(i):
            return [sum([x ** (i + j) for x in x_data]) for j in get_range(power)]

        return [create_row(i) for i in get_range(power)]

    def create_free_coefficients():
        return [sum([(x ** i) * y for (x, y) in data]) for i in get_range(power)]

    matrix = create_matrix()
    free_coefficients = create_free_coefficients()

    for i in get_range(power):
        if matrix[i][i] == 0:
            for j in get_range(power):
                if j == i:
                    continue

                if matrix[j][i] != 0 and matrix[i][j] != 0:
                    for k in get_range(power):
                        matrix[i][k], matrix[j][k] = matrix[j][k], matrix[i][k]

                    free_coefficients[i], free_coefficients[j] = free_coefficients[j], free_coefficients[i]
                    break

    return matrix, free_coefficients, power


def gauss(system):
    matrix, free_coefficients, power = system

    for k in get_range(power):
        for i in get_range(power, k + 1):
            m = matrix[i][k] / matrix[k][k]

            for j in get_range(power, k):
                matrix[i][j] -= m * matrix[k][j]

            free_coefficients[i] -= m * free_coefficients[k]

    return matrix, free_coefficients, power


def calculate_vector(system):
    matrix, free_coefficients, power = system

    result = [0.0 for _ in get_range(power)]
    for i in reversed(get_range(power)):
        result[i] = (free_coefficients[i] - sum([matrix[i][j] * result[j] for j in get_range(power, i)])) / matrix[i][i]

    return result


def calculate_value_of_polynomial(coefficients, arg):
    return sum([c * (arg ** i) for (i, c) in enumerate(coefficients)])


def calculate_error(data, coefficients):
    return sum([(y - calculate_value_of_polynomial(coefficients, x)) ** 2 for (x, y) in data])


def polynomial_string(coefficients):
    def monomial_string(coefficient, power):
        return ('+' if coefficient >= 0 else '-') + ' ' + str(abs(coefficient)) + ' x^' + str(power)

    result = 'P(x) = '
    for (i, c) in enumerate(coefficients):
        result += str(c) if i == 0 else ' ' + monomial_string(c, i)

    return result


if __name__ == '__main__':
    print('== LEARNING ==')
    learn_data = read_data('learn.txt')

    vectors = []
    errors = []
    for power in get_range(50):
        vector = calculate_vector(gauss(create_system(learn_data, power)))
        vectors.append(vector)

        error = calculate_error(learn_data, vector)
        errors.append(error)

        print('power = %d, E = %.5f' % (power, error))

    optimal_power, min_error = min(enumerate(errors), key=operator.itemgetter(1))
    print('optimal power = %d, min E = %.5f' % (optimal_power, min_error))

    print('== TESTING ==')
    test_data = read_data('test.txt')

    vector = vectors[optimal_power]
    print('power = %d, E = %.5f' % (optimal_power, calculate_error(test_data, vector)))
    print(polynomial_string(vector))