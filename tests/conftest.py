import numpy as np
import pytest
from scipy.signal import lfilter


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def ar1(rng):
    """Stationary AR(1) generator, x_t = phi x_{t-1} + e_t, with tau_int = (1+phi)/(1-phi)."""
    def _ar1(phi: float, n: int) -> np.ndarray:
        e = rng.normal(size=n)
        e[0] /= np.sqrt(1 - phi**2)
        return lfilter([1], [1, -phi], e)
    return _ar1
