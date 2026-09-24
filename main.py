import logging 
import numpy as np
import matplotlib.pyplot as plt

import autocorrelation

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def ar1(rho:float, tau:float, len:int)->np.ndarray:
    x = np.zeros(len)
    for i in range(1,len):
        x[i] = rho * x[i-1] + np.random.normal(0, tau**2)
    return x

def main():
    T = 100
    t = np.arange(T)
    x = ar1(rho=0.5, tau=0.1, len= T)
    conv = autocorrelation._empirical_correlation(x)
    gami = autocorrelation.ipse(x)
    gamm = autocorrelation.imse(x)
    gamc = autocorrelation.icse(x)
    logger.info(type(gamc))
    logger.info(conv.shape)


    fig, ax = plt.subplots()
    ax.plot(t, conv / x.var(), label=r"$\widehat{\gamma_k}\;/\;\mathrm{Var}(X_t)$")
    ax.plot(t[::2], 0.5 * gami / x.var(), label=r"$\widehat{\Gamma}_k\;/\;\mathrm{Var}(X_t)$")
    ax.plot(t[::2], 0.5 * gamm / x.var(), label=r"$\widehat{\Gamma}_{\mathrm{mono},k}\;/\;\mathrm{Var}(X_t)$")
    ax.plot(t[::2], 0.5 * gamc / x.var(), label=r"$\widehat{\Gamma}_{\mathrm{conv},k}\;/\;\mathrm{Var}(X_t)$")
    ax.set_ylabel(r"Correlation")
    ax.set_xlabel(r"$k$")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
