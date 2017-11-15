import numpy as np


def euclides(matrix, vector):
    return np.sqrt(np.sum((matrix - vector) ** 2, axis=1))
