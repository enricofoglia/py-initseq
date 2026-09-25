import logging
import numpy as np
import matplotlib.pyplot as plt

from py_initseq import autocorrelation

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def ar1(rho: float, tau: float, n: int) -> np.ndarray:
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.random.normal(0, tau**2)
    return x


def main():
    T = 10_000
    rho = 0.5
    t = np.arange(T)
    x = ar1(rho=rho, tau=0.1, n=T)
    conv = autocorrelation._empirical_correlation(x)
    gami = autocorrelation.ipse(x)
    gamm = autocorrelation.imse(x)
    gamc = autocorrelation.icse(x)

    logger.info(f"exact     : tau_int = {(1 + rho) / (1 - rho):.3f}")
    taus = {}
    for method in ("positive", "monotone", "convex"):
        taus[method], lag = autocorrelation.integrated_time(x, method)
        logger.info(f"{method:10s}: tau_int = {taus[method]:.3f} (summed up to lag {lag})")
    b, tau_b, err_b = autocorrelation.batch_means(x)

    fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(11, 4))
    ax.plot(t, conv / x.var(), label=r"$\widehat{\gamma_k}\;/\;\mathrm{Var}(X_t)$")
    ax.plot(t[: 2 * len(gami) : 2], 0.5 * gami / x.var(), label=r"$\widehat{\Gamma}_k\;/\;\mathrm{Var}(X_t)$")
    ax.plot(
        t[: 2 * len(gamm) : 2],
        0.5 * gamm / x.var(),
        label=r"$\widehat{\Gamma}_{\mathrm{mono},k}\;/\;\mathrm{Var}(X_t)$",
    )
    ax.plot(
        t[: 2 * len(gamc) : 2],
        0.5 * gamc / x.var(),
        label=r"$\widehat{\Gamma}_{\mathrm{conv},k}\;/\;\mathrm{Var}(X_t)$",
    )
    ax.set_xlim(0, 4 * lag)
    ax.set_ylabel(r"Correlation")
    ax.set_xlabel(r"$k$")
    ax.legend()

    ax_b.errorbar(b, tau_b, yerr=err_b, fmt="o", ms=3, lw=1, color="k", label="batch means")
    for (method, tau), color in zip(taus.items(), ("C1", "C2", "C3")):
        ax_b.axhline(tau, ls="--", lw=1, color=color, label=f"Geyer, {method}")
    ax_b.set_xscale("log")
    ax_b.set_ylabel(r"$\widehat\tau_{\mathrm{int}}$")
    ax_b.set_xlabel(r"block size $b$")
    ax_b.legend(loc="upper left")
    fig.tight_layout()
    plt.show()

    fig.savefig("ar1_example.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
