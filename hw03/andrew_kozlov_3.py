#!/usr/bin/env python3

__author__ = 'Andrew Kozlov'
__copyright__ = 'Copyright 2015, SPbAU'

from sys import argv
from collections import Counter
import operator

from scipy.cluster.hierarchy import dendrogram, from_mlab_linkage
from matplotlib import pyplot as plt
import numpy as np


def read_fasta(path):
    result = []
    with open(path) as file:
        fasta_id = None
        fasta_string = ''

        def add_pair():
            if fasta_id:
                result.append((fasta_id, fasta_string))

        for line in file.readlines():
            if line.startswith('>'):
                add_pair()

                fasta_id = line.lstrip('>').strip()
                fasta_string = ''
            else:
                fasta_string += line.strip()

        add_pair()

    return result


def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    if len(target) == 0:
        return len(source)

    result = range(len(target) + 1)
    for (i, s) in enumerate(source):
        new_result = [i + 1]
        for (j, t) in enumerate(target):
            new_result.append(min(result[j + 1] + 1, new_result[j] + 1, result[j] + (s != t)))

        result = new_result

    return result[-1]


def jaccard(source, target, n=8):
    def create_counter(string):
        return Counter([string[i:i + n] for i in range(len(string) - n + 1)])

    source_counter = create_counter(source)
    target_counter = create_counter(target)

    intersection = {}
    union = dict(target_counter)
    for (key, value) in source_counter.items():
        if key in target_counter:
            intersection[key] = min(value, target_counter[key])
            union[key] = max(value, target_counter[key])
        else:
            union[key] = value

    def multiset_size(multiset):
        return sum(multiset.values())

    return 1 - multiset_size(intersection) / multiset_size(union)


def lance_williams(x, dist):
    y = [pair for pair in enumerate(x)]
    n = len(y)
    z = np.zeros((n - 1, 3))

    def calculate_element_distances():
        result = {}
        for (i, (u_id, u_string)) in enumerate(y):
            for (j, (v_id, v_string)) in enumerate(y):
                if i != j:
                    result[(u_id, v_id)] = dist(u_string, v_string)

        return result

    elements_distances = calculate_element_distances()

    def calculate_clusters_distance(first_cluster, second_cluster):
        return sum([elements_distances[(first_id, second_id)]
                    for (first_id, _) in first_cluster
                    for (second_id, _) in second_cluster
                    if first_id != second_id]) / len(first_cluster) / len(second_cluster)

    all_clusters = [[pair] for pair in y]
    current_clusters = all_clusters.copy()
    for t in range(n - 1):
        distances = [(u, v, calculate_clusters_distance(u, v))
                     for (i, u) in enumerate(current_clusters)
                     for (j, v) in enumerate(current_clusters)
                     if i != j]

        (u, v, d) = min(distances, key=operator.itemgetter(2))

        def get_index(cluster):
            return all_clusters.index(cluster) + 1

        z[t, 0] = get_index(u)
        z[t, 1] = get_index(v)
        z[t, 2] = d

        current_clusters.remove(u)
        current_clusters.remove(v)
        current_clusters.append(u + v)
        all_clusters.append(u + v)

    return z


def show_dendrogram(z, **kwargs):
    dendrogram(from_mlab_linkage(z), **kwargs)
    plt.show()


if __name__ == '__main__':
    data = read_fasta(argv[1])
    split = list(map(list, zip(*data)))
    show_dendrogram(lance_williams(split[1], jaccard), labels=split[0])