"""
Provides various commonly used loss functions used as distance metrics
to calculate the fitness on candidate programs.
"""
import math

def hamming_distance(x: list, y: list) -> int:
    """
    Calculate the Hamming distance between two vectors.

    :param x: Set of (pseudo-boolean) values
    :param y: Set of (pseudo-boolean) values
    :return: hamming distance
    """
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0
    for xi, yi in zip(x, y):
        if xi != yi:
            dist += 1
    return dist


def euclidean_distance(x: list, y: list) -> float:
    """
    Calculate the Euclidean distance between two sets of values.

    :param x: Set of integers or real values
    :param y: Set of integers or real values
    :return: euclidean distance
    """
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0.0
    for xi, yi in zip(x, y):
        dist += math.pow(xi - yi, 2)
    return math.sqrt(dist)
