import numpy as np
import pytest
import scipy.stats

from py_initseq import stats


@pytest.fixture
def x(rng):
    return rng.gamma(2.0, size=1000)


def test_running_mean(x):
    rm = stats.running_mean(x)
    for t in (1, 10, len(x)):
        assert rm[t - 1] == pytest.approx(x[:t].mean())


def test_running_std(x):
    assert stats.running_std(x)[-1] == pytest.approx(x.std())


def test_running_moment(x):
    assert stats.running_moment(x, 2)[-1] == pytest.approx(1)
    assert stats.running_moment(x, 3)[-1] == pytest.approx(scipy.stats.skew(x))
    assert stats.running_moment(x, 4)[-1] == pytest.approx(scipy.stats.kurtosis(x, fisher=False))
