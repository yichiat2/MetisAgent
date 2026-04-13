# pMCMC Calibration — Design Plan

## 1. Overview

A new method `pmcmc` is added to `StochasticProcessBase` in `stochastic.py` alongside the existing `calibrate` (CMA-ES).  It calibrates process parameters via **Particle Marginal Metropolis–Hastings (PMMH)** enhanced with the **Correlated Pseudo-Marginal (CPM)** technique.

The three key components are:

| Component | Role |
|---|---|
| APF (`loglikelihood`) | Unbiased particle estimate $\hat{\ell}(\theta)$ of $\log p(y_{1:T}\mid\theta)$ |
| PMMH | MH accept/reject using $\hat{\ell}(\theta)$ in place of the true likelihood |
| CPM | Correlates auxiliary noise across proposals to reduce log-likelihood variance |

---

## 2. Model and Notation

Let $\theta \in \mathbb{R}^d$ denote the **unconstrained** parameter vector (the same space used by CMA-ES).  The constrained parameters $\phi = \mathcal{T}(\theta)$ are obtained via `unconstrained_to_params`.

Each process defines `PARAM_TRANSFORMS`: a dict mapping each parameter name to `(transform_type, lo, hi)`.

---

## 3. Prior Specification

### 3.1 Uniform prior on the constrained space

The simplest and most natural choice is a **uniform prior** over the constrained support $(lo_i, hi_i)$ for each parameter $\phi_i$.  Working in unconstrained coordinates, this induces a prior on $\theta$ with log-density equal to the **log-Jacobian** of the transform:

$$
\log \pi(\theta) = \sum_{i=1}^{d} \log \left|\frac{d\phi_i}{d\theta_i}\right|
$$

Per-transform Jacobian log-determinants:

| `PARAM_TRANSFORMS` type | Constrained map $\phi(\theta)$ | $\log\lvert d\phi/d\theta\rvert$ |
|---|---|---|
| `sigmoid_ab` | $\phi = lo + (hi - lo)\,\sigma(\theta)$ | $\log(hi - lo) + \log\sigma(\theta) + \log(1{-}\sigma(\theta))$ |
| `tanh` | $\phi = \tanh(\theta)$ (then clipped) | $\log(1 - \tanh^2(\theta))$ |
| `softplus` | $\phi = \log(1 + e^\theta)$ | $-\log(1 + e^{-\theta})$ |

The total log-prior under the uniform-in-constrained-space assumption is therefore the sum of these terms. This ensures the sampler does not need knowledge beyond what is already in `PARAM_TRANSFORMS`.

### 3.2 Optional named priors (per parameter)

An optional `PARAM_PRIORS` dict may be defined on each process class, mapping parameter names to `(distribution_type, *hyperparams)`.  Supported types in constrained space:

| Type | Hyperparams | Log-density (constrained $\phi$) |
|---|---|---|
| `"uniform"` | *(uses `lo`, `hi` from transform)* | $0$ (constant) |
| `"beta"` | $(\alpha, \beta)$ | $(\alpha{-}1)\log\hat\phi + (\beta{-}1)\log(1{-}\hat\phi)$, where $\hat\phi = (\phi - lo)/(hi - lo)$ |
| `"normal"` | $(\mu, \sigma)$ | $-\tfrac{(\phi-\mu)^2}{2\sigma^2}$ |
| `"log_normal"` | $(\mu, \sigma)$ | $-\log\phi - \tfrac{(\log\phi - \mu)^2}{2\sigma^2}$ |

If `PARAM_PRIORS` is absent, the uniform-in-constrained-space default (§3.1) is used.

The **log-posterior** evaluated at unconstrained $\theta$ is:

$$
\log \tilde{p}(\theta \mid y_{1:T}) = \hat{\ell}(\theta, u) + \log \pi(\theta)
$$

where $\hat{\ell}(\theta, u)$ is the particle filter log-likelihood estimate and $u$ collects all auxiliary random numbers used by the APF.

---

## 4. Correlated Pseudo-Marginal (CPM) Noise Sampling

### 4.1 Problem with standard PMMH

Standard PMMH draws a fresh set of auxiliary variables $u \sim p(u)$ (i.e. fresh PRNG seeds for the particle filter) at every proposal.  When $N$ particles is large the estimator variance is low, but at moderate $N$ the log-likelihood estimates $\hat{\ell}(\theta, u)$ and $\hat{\ell}(\theta', u')$ are nearly independent, causing the chain to mix poorly — the acceptance ratio amplifies noise.

### 4.2 CPM correlation scheme

The CPM approach (Deligiannidis, Doucet & Pitt 2018) maintains a persistent auxiliary state $u \in \mathbb{R}^M$ and refreshes it via an AR(1) update:

$$
\boxed{u' = \rho_{\mathrm{cpm}}\,u + \sqrt{1 - \rho_{\mathrm{cpm}}^2}\;\xi, \qquad \xi \sim \mathcal{N}(0, I_M)}
$$

with **CPM correlation** $\rho_{\mathrm{cpm}} \in [0, 1)$.  When $\rho_{\mathrm{cpm}} = 0$ this reduces to independent PMMH; when $\rho_{\mathrm{cpm}} \to 1$ the noise is nearly frozen.

Since the APF uses JAX PRNG keys (not explicit Gaussian vectors), the correlation is realized as follows:

1. Store the **base random seed** $s \in \mathbb{Z}$ for the current particle filter run.
2. Generate a fresh seed $s_\xi$ for the innovation.
3. Produce $s'$ by mixing:

$$
s' = \text{hash}\!\left(\lfloor \rho_{\mathrm{cpm}} \cdot 2^{32} \rfloor \;\oplus\; s \;\oplus\; s_\xi\right)
$$

In practice, this is implemented by drawing the $N \times T$ Gaussian noise array $U$ explicitly, applying the AR(1) update element-wise, and passing the resulting array to the particle filter via a deterministic key derived from the mixed array. This requires a version of `loglikelihood` that accepts a pre-drawn noise array instead of deriving noise internally; that adaptation will be done when implementing.

**Optimal CPM correlation** (from theory) is approximately:

$$
\rho_{\mathrm{cpm}}^* \approx \exp\!\left(-\frac{1}{2\,\hat{\sigma}^2_{\log\hat\ell}}\right)
$$

where $\hat{\sigma}^2_{\log\hat\ell}$ is the estimated variance of $\log\hat{\ell}(\theta, u)$ over noise draws $u$ at the current $\theta$.  A good default is $\rho_{\mathrm{cpm}} = 0.99$ for $N \ge 200$ particles; this can be tuned during burn-in by estimating $\hat\sigma^2_{\log\hat\ell}$ from repeated noise draws at the initial $\theta$.

---

## 5. MH Kernel and Proposal

### 5.1 Accept/reject step

At iteration $n$:

1. Propose $\theta' = \theta_n + \epsilon\,L_n\,z$, $z \sim \mathcal{N}(0, I_d)$, where $\epsilon$ is the step size and $L_n$ is a Cholesky factor updated during burn-in.
2. Refresh noise: $u' \leftarrow \rho_{\mathrm{cpm}}\,u_n + \sqrt{1{-}\rho_{\mathrm{cpm}}^2}\,\xi$.
3. Run APF with $(\theta', u')$ to get $\hat{\ell}(\theta', u')$.
4. Compute log acceptance ratio:

$$
\log \alpha = \underbrace{\hat{\ell}(\theta', u') - \hat{\ell}(\theta_n, u_n)}_{\text{log-lik ratio}} + \underbrace{\log\pi(\theta') - \log\pi(\theta_n)}_{\text{log-prior ratio}}
$$

5. Draw $v \sim \mathcal{U}(0,1)$.  If $\log v < \log\alpha$: accept, set $(\theta_{n+1}, u_{n+1}) = (\theta', u')$; else: reject, set $(\theta_{n+1}, u_{n+1}) = (\theta_n, u_n)$.

**Note**: the CPM update to $u$ is always applied (even on rejection the chain state reverts to $u_n$, *not* to the pre-AR(1) state).  This is the standard CPM convention and preserves detailed balance.

### 5.2 Proposal covariance adaptation (burn-in)

During the burn-in period (first $B$ iterations), the proposal scale $\epsilon$ and covariance $\Sigma_n$ are adapted.  After the burn-in the parameters are frozen.

**Covariance adaptation** (AM algorithm, Haario et al. 2001):

$$
\Sigma_n = \frac{2.38^2}{d}\,\mathrm{Cov}(\theta_0, \ldots, \theta_{n-1}) + \frac{2.38^2}{d}\,\varepsilon_0\,I_d
$$

with a small jitter $\varepsilon_0 = 10^{-6}$ to prevent degeneracy.  The Cholesky $L_n = \mathrm{chol}(\Sigma_n)$ is updated every `adapt_interval` steps via rank-1 updates.

**Step-size adaptation** (Robbins–Monro, targeting $\alpha^* = 0.30$):

$$
\log\epsilon_{n+1} = \log\epsilon_n + \gamma_n\,\bigl(\bar{\alpha}_n - \alpha^*\bigr)
$$

$$
\gamma_n = \frac{C}{n^{0.6}}, \qquad C = 1
$$

where $\bar{\alpha}_n$ is the empirical acceptance rate over the last `adapt_interval` steps.  After burn-in, $\epsilon$ is fixed at its final value.

---

## 6. Burn-in and Post-Burn-in Phase

| Phase | Iterations | Actions |
|---|---|---|
| Burn-in | $[0, B)$ | Adapt $\epsilon$ and $\Sigma_n$ every `adapt_interval` steps; discard samples |
| Sampling | $[B, B+N_{\mathrm{iter}})$ | Fixed proposal; collect $\theta_n$, $\hat\ell_n$ |

The chain state (current $\theta$, current $u$, current $\hat\ell$) is carried across both phases without interruption; the only difference is whether adaptation is active.

**Diagnostics during burn-in** still printed and plotted live (with a visual separator at iteration $B$) so convergence can be monitored.

---

## 7. Live Dash Dashboard

A separate thread runs a [Dash](https://dash.plotly.com/) server that reads a shared in-memory buffer (list of dicts) written by the main MCMC loop.

### 7.1 Layout

```
┌─────────────────────────────────────────────────────────┐
│  Header: process name, #iter, acceptance rate, log-lik  │
├──────────────────────┬──────────────────────────────────┤
│  Trace panel         │  Histogram panel                  │
│  One line per param  │  One bar chart per param          │
│  x-axis = iteration  │  x-axis = constrained value       │
│  y-axis = constrained│  (burn-in samples shown greyed)   │
│  value               │                                   │
│  Burn-in region      │                                   │
│  shaded grey         │                                   │
└──────────────────────┴──────────────────────────────────┘
│  Log-likelihood trace (bottom panel, full width)         │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Update mechanism

- `dcc.Interval` component triggers a callback every `dash_update_interval_ms` milliseconds (default 2000 ms).
- Callback reads from the shared buffer; renders traces and histograms via Plotly `go.Scatter` and `go.Histogram`.
- Constrained parameter values $\phi_n = \mathcal{T}(\theta_n)$ are stored in the buffer, so plots are in interpretable units.

### 7.3 Ports and threading

- Dash app starts in a `daemon=True` thread on `localhost:8050` (configurable).
- The MCMC loop writes to the shared buffer under a `threading.Lock`.
- The Dash callback acquires the same lock for reads.

---

## 8. Console Output per Iteration

Each iteration prints:

```
[   42 /  5000]  acc=28.3%  loglik= -12345.67
  θ (unconstrained): [ 0.123  -0.456  1.789  0.234  0.567  -0.012]
  φ (constrained):   v0=0.024  rho=-0.42  kappa=3.12  theta=0.023  sigma=0.18  r=-0.001
  Δloglik=  +1.23  accepted=True
```

During burn-in: prefix with `[BURN-IN]`.

Printed every `print_interval` iterations (default 10).

---

## 9. `pmcmc` Signature and Return Value

```python
def pmcmc(
    self,
    key: jax.Array,
    setting: Setting,
    dsetting: DynSetting,
    *,
    num_iterations: int = 5000,
    burn_in: int = 1000,
    target_acceptance: float = 0.30,
    cpm_rho: float = 0.99,
    adapt_interval: int = 50,
    initial_step_size: float = 0.1,
    print_interval: int = 10,
    dash_port: int = 8050,
    dash_update_interval_ms: int = 2000,
) -> MCMCResult:
```

### `MCMCResult` (new NamedTuple in `helper.py`)

```python
class MCMCResult(NamedTuple):
    samples: np.ndarray          # (num_iterations, d)  unconstrained
    constrained_samples: np.ndarray  # (num_iterations, d)  constrained
    log_likelihoods: np.ndarray  # (num_iterations,)
    acceptance_rate: float
    best_params: np.ndarray      # constrained, shape (d,)
    best_loglik: float
```

Burn-in samples are **not** included in `samples`.

---

## 10. Integration with Existing Architecture

| Existing item | Role in pMCMC |
|---|---|
| `PARAM_NAMES` | Parameter ordering, dashboard labels |
| `PARAM_TRANSFORMS` | Prior log-Jacobian; constrained display values |
| `unconstrained_to_params` | Converting chain state to constrained for prior/print/plot |
| `params_to_unconstrained` | Initialising chain from `dsetting.initial_guess` |
| `loglikelihood` | Called inside the MH kernel; APF provides unbiased $\hat\ell$ |
| `get_default_param` | Provides `setting`, `dsetting`, and initial guess for chain start |

The CPM noise correlation only requires a thin wrapper around `loglikelihood` that accepts an explicit noise seed rather than a JAX key; this will be implemented as a helper that splits the key from the correlated seed.

---

## 11. References

1. Andrieu, C. & Roberts, G.O. (2009). *The pseudo-marginal approach for efficient Monte Carlo computations.* Ann. Statist.
2. Deligiannidis, G., Doucet, A. & Pitt, M.K. (2018). *The correlated pseudo-marginal method.* J. R. Statist. Soc. B.
3. Haario, H., Saksman, E. & Tamminen, J. (2001). *An adaptive Metropolis algorithm.* Bernoulli.
4. Dahlin, J. & Schön, T.B. (2015). *Getting started with particle Metropolis-Hastings for inference in nonlinear stochastic differential equations.* J. Statist. Softw.
