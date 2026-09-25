import numpy as np
import pytest

from py_initseq import autocorrelation as ac


def test_empirical_correlation_matches_direct_sum(rng):
    x = rng.normal(size=11)
    xc = x - x.mean()
    direct = np.array([xc[:len(x) - k] @ xc[k:] for k in range(len(x))])
    np.testing.assert_allclose(ac._empirical_correlation(x), direct / len(x))
    np.testing.assert_allclose(ac._empirical_correlation(x, unbiased=True), direct / (len(x) - np.arange(len(x))))


def test_sum_over_all_lags_is_zero(rng):
    gam = ac._empirical_correlation(rng.normal(size=100))
    assert gam[0] + 2 * gam[1:].sum() == pytest.approx(0, abs=1e-12)


def test_greatest_convex_minorant():
    f = np.array([3, 1, 2, 0.5, 4, 0.2, 1.0])
    g, hull = ac.greatest_convex_minorant(f)
    np.testing.assert_array_equal(hull, [0, 1, 3, 5, 6])
    np.testing.assert_allclose(g[hull], f[hull])
    assert np.all(g <= f + 1e-12)
    assert np.all(np.diff(g, 2) >= -1e-12)


@pytest.mark.parametrize("n", [1000, 1001])
def test_initial_sequences(ar1, n):
    x = ar1(0.9, n)
    gi, gm, gc = ac.ipse(x), ac.imse(x), ac.icse(x)
    assert len(gi) == len(gm) == len(gc) > 0
    assert np.all(gi > 0)
    assert np.all(np.diff(gm) <= 0) and np.all(gm <= gi)
    assert np.all(np.diff(gc, 2) >= -1e-12) and np.all(gc <= gm + 1e-12)


@pytest.mark.parametrize("method", ["positive", "monotone", "convex"])
@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_integrated_time_ar1(ar1, phi, method):
    tau, _ = ac.integrated_time(ar1(phi, 100_000), method)
    # worst case over 200 seeds is ~25% (positive, phi=0.9)
    assert tau == pytest.approx((1 + phi) / (1 - phi), rel=0.3)


def test_integrated_time_positive_is_truncated_sum(ar1):
    x = ar1(0.5, 10_000)
    tau, lag = ac.integrated_time(x, "positive")
    rho = ac._empirical_correlation(x) / x.var()
    assert tau == pytest.approx(1 + 2 * rho[1:lag + 1].sum())


def test_correlation_time(ar1):
    x = ar1(0.5, 10_000)
    assert ac.correlation_time(x, dt=1e-3) == pytest.approx(0.5e-3 * ac.integrated_time(x)[0])


def test_batch_means(ar1):
    n, phi = 200_000, 0.9
    x = ar1(phi, n)

    b, tau, err = ac.batch_means(x)
    assert b[0] == 1 and tau[0] == pytest.approx(n / (n - 1))
    assert np.all(n // b >= 10)
    assert np.all(err > 0)

    # 4 standard errors, sqrt(2/(n_b-1)), evaluated at the exact tau_int
    tau_exact = (1 + phi) / (1 - phi)
    _, tau, _ = ac.batch_means(x, [n // 50])
    assert abs(tau[0] - tau_exact) < 4 * tau_exact * np.sqrt(2 / 49)
