import numpy as np


def phase_to_line(v):
    return v * np.sqrt(3)

def clarke_transform(a, b, c):
    alpha = 2 / 3 * (a - b / 2 - c / 2)
    beta = 2 / 3 * (b - c) * np.sqrt(3) / 2
    return alpha, beta

def park_transform(alpha, beta, theta):
    d = alpha * np.cos(theta) + beta * np.sin(theta)
    q = -alpha * np.sin(theta) + beta * np.cos(theta)
    return d, q
