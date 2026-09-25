# Wind tunnel data analysis

Analyse wind tunnel data to compute statistical convergence and correlation times.

## Autocorrelation

Let $\{X_t\}$ be a discrete time random process sampled with time step $\Delta t = 1/f_s$ (in this case, $X$ is either the lift or the drag coefficient). The autocovariance $\gamma_X(s)$ at lag $s$ is defined as:

$$
    \gamma_X(s) = \mathbb E[(X_t-\mu)(X_{t+s}-\mu)]
$$

Where $\mu = \mathbb E(X_t)$. Notice that, for weak-sense stationary random processes, the average and autocovariance do not depend on the reference $t$. The autocorrelation function is defined as $\varrho_X(s) = \gamma_X(s) / \gamma_X(0) = \gamma_X(s) / \mathrm{Var}(X)$.

### Correlation time

The correlation time (or integral time scale) is defined as:

$$
\tau_c = \int_0^\infty \mathrm d s\; \varrho_X(s) =  \frac{1}{\gamma_X(0)}\int_0^\infty \mathrm d s\; \gamma_X(s)
$$

It sets the convergence rate of the empirical average $\widehat\mu$ of $T$ samples, i.e. over a duration $T\Delta t$:

$$
\mathrm{Var}(\widehat\mu) \approx \frac{2\tau_c}{T\Delta t}\,\mathrm{Var}(X)
$$

For sampled signals, it is convenient to work with the *integrated autocorrelation time*, measured in samples:

$$
\tau_{\mathrm{int}} = \sum_{k=-\infty}^{\infty}\varrho_X(k) = 1 + 2\sum_{k=1}^{\infty}\varrho_X(k)
$$

so that $\mathrm{Var}(\widehat\mu) \approx \tau_{\mathrm{int}}\mathrm{Var}(X)/T$: the $T$ correlated samples are worth $N_{\mathrm{eff}} = T/\tau_{\mathrm{int}}$ independent ones. Approximating the integral with the trapezoidal rule, the two are related by:

$$
\tau_c \approx \Delta t\left(\frac{1}{2} + \sum_{k=1}^{\infty}\varrho_X(k)\right) = \frac{\Delta t}{2}\,\tau_{\mathrm{int}}
$$

### Estimation

To estimate $\tau_{\mathrm{int}}$, we follow [Geyer1992][^1]. Let $\{X_1,\dots, X_T\}$ be a set of $T$ samples of the random process $X_t$. Define the empirical average as:

$$
\widehat \mu = \frac{1}{T}\sum_{t=1}^T X_t
$$

Define the empirical estimator of the covariance as:

$$
    \widehat\gamma_k = \frac{1}{T}\sum_{t=1}^{T-k}(X_t-\widehat\mu)(X_{t+k}-\widehat\mu)
$$

where dividing by $T$ instead of $T-k$ reduces the variance of the estimators for large lags $k$ and makes the sequence $\{\widehat\gamma_k\}$ positive semi-definite. The empirical autocorrelation is $\widehat\varrho_k = \widehat\gamma_k/\widehat\gamma_0$.

The sum of *all* the empirical lags cannot be used: since the empirical average is subtracted, $\widehat\gamma_0 + 2\sum_{k=1}^{T-1}\widehat\gamma_k = 0$ exactly. The sum must therefore be truncated, and the estimators below differ in how they choose the truncation. They are built on the sums of adjacent pairs of autocovariances:

$$
\widehat \Gamma_k = \widehat\gamma_{2k} + \widehat\gamma_{2k+1}
$$

For a reversible Markov chain, the exact $\Gamma_k$ are positive, decreasing and convex in $k$ [Geyer1992][^1]. The estimators enforce these properties on $\widehat\Gamma_k$:

* **initial positive sequence estimator**: keep $\widehat\Gamma_{0:m}$, where $m$ is the end of the first run of positive terms: $\widehat\Gamma_k > 0$ for all $k \le m$ and $\widehat\Gamma_{m+1} \le 0$. Positive terms at larger lags are due to noise and are discarded. Here $\widehat\Gamma_{0:m}$ is a short hand notation for the sequence $\{\widehat\Gamma_0,\dots,\widehat\Gamma_{m}\}$.
* **initial monotone sequence estimator**: the running minimum of the initial positive sequence,

$$
\widehat\Gamma_{\mathrm{mono},k} = \min \widehat\Gamma_{0:k}, \quad k = 0,\dots,m
$$

* **initial convex sequence estimator**: $\widehat\Gamma_{\mathrm{conv},0:m}$ is the greatest convex minorant (lower convex hull) of $\widehat\Gamma_{\mathrm{mono},0:m}$.

For any of the three estimators, the empirical integrated autocorrelation time and correlation time are:

$$
\widehat\tau_{\mathrm{int}} = \frac{1}{\widehat\gamma_0}\left(-\widehat\gamma_0 + 2\sum_{k=0}^{m}\widehat\Gamma_k\right), \qquad \widehat\tau_c = \frac{\Delta t}{2}\,\widehat\tau_{\mathrm{int}}
$$

For the initial positive sequence estimator, this reduces to $\widehat\tau_{\mathrm{int}} = 1 + 2\sum_{k=1}^{2m+1}\widehat\varrho_k$. All three estimators are asymptotically conservative (they overestimate $\tau_{\mathrm{int}}$); the monotone and convex ones usually have a lower variance.

**Caveat.** The properties of $\Gamma_k$ are proven for reversible Markov chains, not for physical signals. When the signal has a periodic component (e.g. vortex shedding), $\varrho_X$ oscillates: the truncation keeps the first positive lobe of $\varrho_X$ but discards the negative lobe that compensates it, and $\widehat\tau_{\mathrm{int}}$ can be overestimated significantly (by more than 50% on a synthetic test signal). In this case, rely on batch means.

### Batch means

Split the $T$ samples into $n_b = \lfloor T/b\rfloor$ non-overlapping blocks of length $b$, and call $\bar X^{(b)}_j$ the average of block $j$. Since $\mathrm{Var}(\bar X^{(b)}) \approx \tau_{\mathrm{int}}\mathrm{Var}(X)/b$ when $b \gg \tau_{\mathrm{int}}$, the estimator:

$$
\widehat\tau(b) = \frac{b\,\widehat{\mathrm{Var}}(\bar X^{(b)})}{\widehat\gamma_0}
$$

reaches a plateau at $\tau_{\mathrm{int}}$ for large enough $b$. It makes no assumption on the shape of $\varrho_X$, so it remains valid for oscillating signals, at the cost of a larger variance: for approximately Gaussian block averages, the standard error of $\widehat\tau(b)$ is $\widehat\tau(b)\sqrt{2/(n_b-1)}$.

[^1]: Geyer, Charles J. "Practical markov chain monte carlo." Statistical science (1992): 473-483.


## Convergence

Compute the convergence up to fourth order by computing the first 4 standardized moments:
 * **average** $\mu = \mu_1 = \mathbb E(X)$
 * **standard deviation** $\sigma = \mu_2^{1/2} = \sqrt{\mathbb E[(X-\mu_1)^2]}$ 
 * **skewness** $\alpha_3 = \mu_3 / \mu_2^{3/2} = \frac{\mathbb E[(X-\mu_1)^3]}{\mathbb E^{3/2}[(X-\mu_1)^2]}$
 * **flatness** $\alpha_4 = \mu_4 / \mu_2^{4/2} = \frac{\mathbb E[(X-\mu_1)^4]}{\mathbb E^{4/2}[(X-\mu_1)^2]}$

The expected values are computed in time as a running average to show the convergence:

$$
    \widehat\mu_1(t) = \frac{1}{t}\sum_{k=1}^t X_k\\
    \widehat\sigma(t) = \sqrt{\frac{1}{t}\sum_{k=1}^t(X_k-\widehat\mu(T))^2}\\
    \widehat\alpha_m(t) = \frac{1}{\widehat\sigma(T)^m}\frac{1}{t}\sum_{k=1}^t(X_k-\widehat\mu(T))^m
$$
