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

def _empirical_correlation(x:np.ndarray, unbiased:bool=False)->np.ndarray:
    r"""Compute the empirical autocorrelation function:
    
    .. math::

         \widehat\gamma_k = \frac{1}{T}\sum_{t=1}^{T-k}(X_t-\widehat\mu)(X_{t+k}-\widehat\mu)

    where :math:`\widehat{\mu}` is the empirical average.
    """
    
    x_ = x - x.mean()
    conv = correlate(x_ ,x_, mode="full", method="fft")
    if unbiased:
        norm = np.arange(1, len(x)+1)[::-1]
    else:
        norm = len(x)

    logger.debug(f"conv.shape = {conv.shape}")
    return conv[len(conv)//2:] / norm

def _gamma_pairs(x:np.ndarray)->np.ndarray:
    r"""Sums of adjacent autocovariances, :math:`\widehat\Gamma_k = \widehat\gamma_{2k} + \widehat\gamma_{2k+1}`.

    The last lag is dropped when the number of lags is odd.
    """
    gam = _empirical_correlation(x)
    n = len(gam) - len(gam) % 2
    return gam[0:n:2] + gam[1:n:2]

def ipse(x:np.ndarray)->np.ndarray:
    r"""Compute the initial positive sequence estimator.

    Returns :math:`\widehat\Gamma_0,\dots,\widehat\Gamma_m`, where :math:`m` is the
    last index before the first non-positive :math:`\widehat\Gamma_k`.
    """
    gam = _gamma_pairs(x)
    nonpos = np.flatnonzero(gam <= 0)
    m = nonpos[0] if nonpos.size else len(gam)
    return gam[:m]

def imse(x:np.ndarray)->np.ndarray:
    r"""Compute the initial monotone sequence estimator, the running minimum of the
    initial positive sequence: :math:`\min(\widehat\Gamma_0,\dots,\widehat\Gamma_k)`.
    """
    return np.minimum.accumulate(ipse(x))

def icse(x:np.ndarray)->np.ndarray:
    """Compute the initial convex sequence estimator, the greatest convex minorant
    of the initial monotone sequence.
    """
    return greatest_convex_minorant(imse(x))[0]

_ESTIMATORS = {"positive": ipse, "monotone": imse, "convex": icse}

def integrated_time(x:np.ndarray, method:str="convex")->tuple[float,int]:
    r"""Integrated autocorrelation time, in samples [Geyer1992]:

    .. math::

        \widehat\tau_{\mathrm{int}} = \frac{-\widehat\gamma_0 + 2\sum_{k=0}^{m}\widehat\Gamma_k}{\widehat\gamma_0}

    where :math:`\widehat\Gamma_k` is the initial ``"positive"``, ``"monotone"`` or
    ``"convex"`` sequence estimator. The variance of the empirical mean is then
    :math:`\mathrm{Var}(\widehat\mu)\approx\widehat\gamma_0\,\widehat\tau_{\mathrm{int}}/T`.

    Assumes a non-oscillating autocorrelation: for signals with a periodic component
    (e.g. vortex shedding) cross-check with :func:`batch_means`.

    Returns (tau, lag): the estimate and the last lag :math:`2m+1` in the sum.
    """
    gam = _ESTIMATORS[method](x)
    gam0 = np.var(x)
    tau = (-gam0 + 2 * gam.sum()) / gam0
    return tau, 2 * len(gam) - 1

def correlation_time(x:np.ndarray, dt:float=1.0, method:str="convex")->float:
    r"""Correlation time (integral time scale), in the units of the time step ``dt``:

    .. math::

        \widehat\tau_c = \frac{\Delta t}{2}\,\widehat\tau_{\mathrm{int}}

    i.e. the trapezoidal-rule approximation of :math:`\int_0^\infty\varrho_X(s)\,\mathrm ds`,
    with :math:`\widehat\tau_{\mathrm{int}}` from :func:`integrated_time`.
    """
    return 0.5 * dt * integrated_time(x, method)[0]

def batch_means(x:np.ndarray, block_sizes:np.ndarray=None, min_batches:int=10)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    r"""Batch-means estimate of the integrated autocorrelation time as a function of
    the block size :math:`b`:

    .. math::

        \widehat\tau(b) = \frac{b\,\widehat{\mathrm{Var}}(\bar X^{(b)})}{\widehat\gamma_0}

    where :math:`\bar X^{(b)}` are the means of the :math:`n_b = \lfloor T/b\rfloor`
    non-overlapping blocks of length :math:`b`. :math:`\widehat\tau(b)` reaches a
    plateau at :math:`\tau_{\mathrm{int}}` once :math:`b \gg \tau_{\mathrm{int}}`.
    No assumption is made on the shape of the autocorrelation.

    By default, block sizes are log-spaced so that there are at least
    ``min_batches`` blocks. For a single estimate, pass e.g. ``[len(x) // 30]``.

    Returns (b, tau, err), with :math:`\mathrm{err}\approx\widehat\tau(b)\sqrt{2/(n_b-1)}`
    the standard error for approximately Gaussian block means.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if block_sizes is None:
        block_sizes = np.logspace(0, np.log10(T // min_batches), 50)
    b = np.unique(np.asarray(block_sizes, dtype=int))
    nb = T // b

    tau = np.empty(len(b))
    for i, (bi, ni) in enumerate(zip(b, nb)):
        means = x[:ni * bi].reshape(ni, bi).mean(axis=1)
        tau[i] = bi * means.var(ddof=1) / x.var()
    err = tau * np.sqrt(2 / (nb - 1))
    return b, tau, err
