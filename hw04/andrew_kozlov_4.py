#!/usr/bin/env python3

__author__ = 'Andrew Kozlov'
__copyright__ = 'Copyright 2015, SPbAU'

import sys
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import operator


def read_image(path, reshape=True):
    image = misc.imread(path)

    if reshape:
        old_shape = image.shape
        return np.reshape(image, (old_shape[0] * old_shape[1], old_shape[2]))
    else:
        return image


def matrix_row_to_tuple(matrix):
    return tuple(matrix.tolist()[0])


def k_means(x, n_clusters, distance_metric=lambda x, y: np.linalg.norm(x - y)):
    def create_random_centroids():
        unique_rows = set()

        while len(unique_rows) != n_clusters:
            unique_rows.add(tuple(x[np.random.choice(x.shape[0])].tolist()))

        result = [row for row in unique_rows]
        result = np.matrix(result)
        result.sort(axis=0)
        return result

    cached_distances = {}
    centroids = create_random_centroids()
    old_centroids = create_random_centroids()
    labels = []

    def get_distance(array, matrix):
        array_tuple = tuple(array.tolist())
        matrix_tuple = matrix_row_to_tuple(matrix)

        pair = (array_tuple, matrix_tuple)
        if pair in cached_distances:
            return cached_distances[pair]

        result = distance_metric(array, matrix)
        cached_distances[pair] = result
        cached_distances[(matrix_tuple, array_tuple)] = result

        return result

    def converged(ratio=0.05, testing=False):
        if centroids.shape != old_centroids.shape:
            return False

        delta = centroids - old_centroids
        norm = np.linalg.norm(delta)
        if testing:
            print(delta)

        return norm / np.linalg.norm(centroids) < ratio and norm / np.linalg.norm(old_centroids) < ratio

    while not converged():
        old_centroids = centroids

        clusters = {}
        labels = []
        for x_row in x:
            distances = [get_distance(x_row, centroid) for centroid in centroids]
            centroid = np.argmin(np.array(distances))

            if centroid in clusters:
                clusters[centroid].append(x_row)
            else:
                clusters[centroid] = [x_row]

            labels.append(centroid)

        centroids = np.matrix([np.mean(clusters[c], axis=0) for c in sorted(clusters.keys())])
        centroids = centroids.astype(int)
        centroids.sort(axis=0)

    return labels, centroids


def centroid_histogram(labels):
    pixels = len(labels)
    n_colors = np.max(labels) + 1
    bars = [labels.count(i) / pixels for i in range(n_colors)]

    plt.bar(np.arange(n_colors), bars)
    plt.xlabel('Clusters')
    plt.ylabel('Number of pixels')
    plt.title('Number of pixels in each cluster')
    plt.show()

    return bars


def plot_colors(hist, centroids):
    centroids = [matrix_row_to_tuple(centroid) for centroid in centroids]

    left = 0.0
    white_color = (256, 256, 256)
    paths = []
    for (percent, color) in zip(hist, centroids):
        vertices = [(left, 0.0), (left, 1.0), (left + percent, 1.0), (left + percent, 0.0), (left, 0.0)]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        left += percent

        paths.append((Path(vertices, codes), tuple(map(operator.truediv, color, white_color))))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for (path, color) in paths:
        patch = patches.PathPatch(path, facecolor=color, linewidth=0.0)
        ax.add_patch(patch)
    plt.title('Percent of pixels of each color')
    plt.show()


def recolor(image, k_means_result, n_colors):
    labels, centroids = k_means_result

    shape = image.shape
    labels = np.array(labels).reshape((shape[0], shape[1]))
    centroids = [np.array(centroid).tolist()[0] for centroid in centroids]

    for (i, row) in enumerate(labels):
        for (j, centroid) in enumerate(row):
            image[i][j] = np.array(centroids[centroid])

    return image


if __name__ == '__main__':
    image_path = sys.argv[1]
    colors_count = 16

    colors, mu = k_means(read_image(image_path), colors_count)
    plot_colors(centroid_histogram(colors), mu)
    misc.imsave('out_' + image_path, recolor(read_image(image_path, False), (colors, mu), colors_count))