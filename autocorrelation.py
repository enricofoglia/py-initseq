import logging

import numpy as np
from scipy.signal import correlate

logger = logging.getLogger(__name__)

def greatest_convex_minorant(f:np.ndarray, x:np.ndarray=None)->tuple[np.ndarray,np.ndarray]:
    """Greatest convex minorant of samples f at sorted abscissae x.

    Returns (g, hull): g evaluated on x, and the indices of the hull vertices.
    """
    f = np.asarray(f, dtype=float)
    x = np.arange(len(f), dtype=float) if x is None else np.asarray(x, dtype=float)

    hull = []
    for i in range(len(f)):
        # Pop the last vertex while it lies on or above the segment
        # from the second-to-last vertex to the new point.
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (x[b] - x[a]) * (f[i] - f[a]) - (f[b] - f[a]) * (x[i] - x[a])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)

    hull = np.asarray(hull)
    g = np.interp(x, x[hull], f[hull])
    return g, hull

def _empirical_correlation(x:np.ndarray)->np.ndarray:
    r"""Compute the empirical autocorrelation function:
    
    .. math::

         \widehat\gamma_k = \frac{1}{T}\sum_{t=1}^{T-k}(X_t-\widehat\mu)(X_{t+k}-\widehat\mu)

    where :math:`\widehat{\mu}` is the empirical average.
    """
    
    x_ = x - x.mean()
    conv = correlate(x_ ,x_, mode="full", method="fft")
    norm = np.arange(1, len(x)+1)[::-1]
    logger.debug(f"conv.shape = {conv.shape}")
    return conv[len(conv)//2:] / norm

def ipse(x:np.ndarray)->np.ndarray:
    """Compute the initial positive sequence estimator"""
    gam = _empirical_correlation(x)
    return gam[0::2] + gam[1::2]

def imse(x:np.ndarray)->np.ndarray:
    """Compute the initial monotone sequence estimator"""
    gam = ipse(x)
    gamm = np.zeros_like(gam)
    gamm[0] = gam[0]
    gamm[1:] = np.minimum(gam[0:-1], gam[1:])
    return gamm

def icse(x:np.ndarray)->np.ndarray:
    """Compute the initial convex sequence estimator"""
    gam = ipse(x)
    return greatest_convex_minorant(gam)[0]
    