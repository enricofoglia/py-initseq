# Wind tunnel data analysis

Analyse wind tunnel data to compute statistical convergence and correlation times.

## Autocorrelation

Let $\{X_t\}$ be a discrete time random process (in this case, $X$ is either the lift or the drag coefficient). The autocorrelation $\varrho_X(\tau)$ is defined as:

$$
    \varrho_X(\tau) = \mathbb E[(X_t-\mu)(X_{t+\tau}-\mu)]
$$

Where $\mu = \mathbb E(X_t)$. Notice that, for weak-sense stationary random processes, the average and autocorrelation does not depend on the reference $t$.  

To estimate the $\varrho_X$, we follow [Geyer1992][^1]. Let $\{X_1,\dots, X_T\}$ be a set of $T$ samples of the random process $X_t$. Define the empirical average as:

$$
\widehat \mu = \frac{1}{T}\sum_{t=1}^T X_t
$$

Define the empirical estimator of the covariance as:

$$
    \widehat\gamma_k = \frac{1}{T}\sum_{t=1}^{T-k}(X_t-\widehat\mu)(X_{t+k}-\widehat\mu)
$$

where dividing by $T$ instead of $T-k$ helps reduce the variance of the estimators for large lags $k$. Define the *initial positive sequence estimator*:

$$
\widehat \Gamma_k = \widehat\gamma_{2k} + \widehat\gamma_{2k+1}
$$

The *autocorrelation time* can be taken to be the last index $m$ for which $\widehat\Gamma_k>0$. This can be refined by using the *initial monotone sequence estimator*:

$$
\widehat\Gamma_{\mathrm{mono},k} = \min(\widehat\Gamma_{k-1}, \widehat\Gamma_k), \quad \widehat\Gamma_{\mathrm{mono},0} = \widehat\Gamma_0
$$

Finally, the *initial convex sequence estimator* works by using the convex hull (or greatest convex minorant) of the sequence $\{\widehat\Gamma_k\}$. Theoretically, this last estimator is the most precise.


[^1]: Geyer, Charles J. "Practical markov chain monte carlo." Statistical science (1992): 473-483.


## Convergence

Compute the convergence up to fourth order by computing the first 4 standardized moments:
 * **average** $\mu_1 = \mathbb E(X)$
 * **standard deviation** $\sigma = \mu_2^{1/2} = \sqrt{\mathbb E[(X-\mu_1)^2}]$ 
 * **skewness** $\alpha_3 = \mu_3 / \mu_2^{3/2} = \frac{\mathbb E[(X-\mu_1)^3]}{\mathbb E^{3/2}[(X-\mu_1)^2]}$
 * **flatness** $\alpha_4 = \mu_4 / \mu_2^{4/2} = \frac{\mathbb E[(X-\mu_1)^4]}{\mathbb E^{4/2}[(X-\mu_1)^2]}$

The expected values are computed in time as a running average to show the convergence, as:

$$
    \widehat\mu_1(t) = \frac{1}{t}\sum_{k=1}^t X_k
$$