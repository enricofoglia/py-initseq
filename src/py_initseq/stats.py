import numpy as np


def running_mean(x: np.ndarray) -> np.ndarray:
    csum = np.cumsum(x)
    norm = np.arange(1, len(x) + 1)
    return csum / norm


def running_std(x: np.ndarray) -> np.ndarray:
    m = np.mean(x)
    return np.sqrt(running_mean((x - m) ** 2))


def running_moment(x: np.ndarray, order: int) -> np.ndarray:
    m = np.mean(x)
    s = np.std(x)
    return s ** (-order) * running_mean((x - m) ** order)
