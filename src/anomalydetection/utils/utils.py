import numpy as np

def l1shrink(x, eps):
    return np.sign(x) * np.clip(np.abs(x) - eps, a_min=0, a_max=None)


def l21shrink(x, eps):
    return x - (eps * x) / np.linalg.norm(x, ord=2, axis=0)


def nuclear_prox(x, eps):
    U, s, V = np.linalg.svd(x, full_matrices=False)
    T = l1shrink(s, eps)
    return U.dot(np.diag(T)).dot(V)


def l2norm(x):
    return np.linalg.norm(x, ord=2)


def fnorm(x):
    return np.linalg.norm(x, ord='fro')