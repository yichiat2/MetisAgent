# CPAM-Heston: Time-Varying Alpha/Beta Particle Filter and Portfolio Design

## Overview

This document specifies a **Conditional CAPM–Heston** (CPAM-Heston) model where
the market exposure ($\beta_t$) and idiosyncratic excess return ($\alpha_t$) of a
single stock $X$ are latent, time-varying, and must be estimated online from
observed log-returns. The idiosyncratic volatility follows a **CIR** (Cox–Ingersoll–Ross)
process. A **Particle Filter (Sequential Monte Carlo)** is designed to recursively
estimate the joint posterior over $(\alpha_t, \beta_t, V_t)$.

A second major section designs optimal portfolio and trading strategies that
consume the filtered state estimates downstream.

---

## Part I — Model Specification

### 1.1 Observation Equation

At each discrete time step $t$ (e.g. one minute or one day), the log-return of stock $X$ is

$$
\boxed{
r_t^X = \alpha_t + \beta_t\, r_t^{\mathrm{SPX}} + \sqrt{V_t \,\Delta t}\;\varepsilon_t,
\qquad \varepsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0,1)
}
\tag{1}
$$

where $r_t^{\mathrm{SPX}}$ is the observed SPX log-return (a **known** driving input, not a
latent state), and $V_t$ is the **instantaneous idiosyncratic variance** of $X$.
The noise $\varepsilon_t$ is independent of the SPX shock.

**Interpretation.**  
- $\alpha_t$: Jensen's alpha — the component of return that cannot be explained by the
  market factor. Time-varying alpha represents changing manager skill, earnings drift,
  or supply/demand imbalance in the stock.  
- $\beta_t$: Factor loading — how sensitive $X$ is to market moves at time $t$. It can
  vary with macro regimes, liquidity, or investor positioning.  
- $\sqrt{V_t}$: Idiosyncratic volatility modulated by a Heston-like CIR process.

---

### 1.2 Latent State Dynamics

#### Alpha (AR(1) with mean zero)

$$
\alpha_t = \phi_a\,\alpha_{t-1} + \sigma_a\,\xi_t^a,
\qquad \xi_t^a \overset{\text{iid}}{\sim} \mathcal{N}(0,1),
\quad |\phi_a| < 1
\tag{2}
$$

The unconditional distribution is $\alpha_t \sim \mathcal{N}\!\bigl(0,\,\sigma_a^2/(1-\phi_a^2)\bigr)$.
Setting $\phi_a$ close to 1 gives persistent alpha; small $\phi_a$ gives fast mean-reversion.

#### Beta (AR(1) around a long-run mean $\bar\beta$)

$$
\beta_t = \bar\beta + \phi_b(\beta_{t-1} - \bar\beta) + \sigma_b\,\xi_t^b,
\qquad \xi_t^b \overset{\text{iid}}{\sim} \mathcal{N}(0,1),
\quad |\phi_b| < 1
\tag{3}
$$

Reparametrising $\tilde\beta_t = \beta_t - \bar\beta$ gives a zero-mean AR(1) identical
to the alpha recursion. In the simplest form $\bar\beta = 1$.

#### Idiosyncratic Variance — CIR Process

$$
dV_t = \kappa(\theta - V_t)\,dt + \sigma_V\sqrt{V_t}\,dW_t^V
\tag{4}
$$

Euler–Maruyama discretisation at step size $\Delta t$:

$$
V_t = \max\!\Bigl(V_{t-1} + \kappa(\theta - V_{t-1})\Delta t
                  + \sigma_V\sqrt{V_{t-1}\,\Delta t}\;\xi_t^V,\;\delta\Bigr),
\qquad \xi_t^V \overset{\text{iid}}{\sim} \mathcal{N}(0,1)
\tag{5}
$$

where $\delta > 0$ is a small variance floor (e.g. $10^{-6}$). The Feller condition
$2\kappa\theta \geq \sigma_V^2$ ensures $V_t > 0$ in continuous time; the floor
enforces this numerically.

**Noise independence.**  
$\xi_t^a$, $\xi_t^b$, $\xi_t^V$, and $\varepsilon_t$ are mutually independent. A
generalisation can allow $\mathrm{corr}(\varepsilon_t, \xi_t^V) = \rho$ (leverage
effect), at the cost of more complex weights — see §2.3.

---

### 1.3 State-Space Summary

| Component | Symbol | Dynamics | Parameters |
|-----------|--------|----------|------------|
| Idiosyncratic alpha | $\alpha_t$ | AR(1) | $\phi_a, \sigma_a$ |
| Market beta | $\beta_t$ | AR(1) around $\bar\beta$ | $\phi_b, \sigma_b, \bar\beta$ |
| Idiosyncratic variance | $V_t$ | CIR | $\kappa, \theta, \sigma_V, V_0$ |

Joint latent state: $\mathbf{x}_t = (\alpha_t, \beta_t, V_t)^\top \in \mathbb{R}^3$.

Full parameter vector:
$$
\boldsymbol\psi = (\phi_a, \sigma_a,\; \phi_b, \sigma_b, \bar\beta,\; \kappa, \theta, \sigma_V, V_0)
$$

---

## Part II — Particle Filter Design

### 2.1 Bayesian Filtering Objective

At each step $t$, track the filtering distribution:

$$
p(\mathbf{x}_t \mid y_{1:t})
\propto
p(y_t \mid \mathbf{x}_t)\;
\int p(\mathbf{x}_t \mid \mathbf{x}_{t-1})\,
p(\mathbf{x}_{t-1} \mid y_{1:t-1})\,d\mathbf{x}_{t-1}
\tag{6}
$$

The model is nonlinear (the observation variance $V_t \Delta t$ depends on the latent
state), making a Kalman filter exact only if $V_t$ is fixed. In general, a
**Sequential Monte Carlo (SMC)** method is required.

---

### 2.2 Bootstrap Particle Filter (BPF)

The simplest baseline. At step $t$, given particles $\{\mathbf{x}_{t-1}^{(i)}\}_{i=1}^N$
with uniform weights:

**Step 1 — Propagate** (sample from prior transition)

$$
\alpha_t^{(i)} = \phi_a\,\alpha_{t-1}^{(i)} + \sigma_a\,\xi^{a,(i)},
\quad
\tilde\beta_t^{(i)} = \phi_b\,\tilde\beta_{t-1}^{(i)} + \sigma_b\,\xi^{b,(i)},
\quad
V_t^{(i)} = \text{CIR-step}(V_{t-1}^{(i)})
\tag{7}
$$

**Step 2 — Weigh** by observation likelihood

$$
\tilde w_t^{(i)}
= p\!\left(y_t \mid \mathbf{x}_t^{(i)}\right)
= \mathcal{N}\!\left(y_t;\;\alpha_t^{(i)} + \beta_t^{(i)}\,r_t^{\mathrm{SPX}},\;V_t^{(i)}\Delta t\right)
\tag{8}
$$

Normalise: $w_t^{(i)} = \tilde w_t^{(i)} / \sum_j \tilde w_t^{(j)}$.

**Step 3 — Resample** (systematic resampling) whenever $\mathrm{ESS} < N/2$:

$$
\mathrm{ESS}_t = \frac{1}{\sum_i (w_t^{(i)})^2}
$$

**Log-likelihood contribution:**

$$
\log \hat p(y_t \mid y_{1:t-1}) = \log \frac{1}{N} \sum_i \tilde w_t^{(i)}
\tag{9}
$$

**Pros:** Simple to implement; only one resampling step per step.  
**Cons:** High variance weights when the observation is informative relative to the prior
(especially for large $|\alpha|$ or large $|\beta|$ moves); $O(N \log N)$ resample cost
(though $O(N)$ with systematic resampling).

---

### 2.3 Auxiliary Particle Filter (APF)

The APF uses a **pilot look-ahead weight** $g_t^{(i)}$ before propagation to pre-select
particles that are more likely to be consistent with $y_t$.

**Stage 1 — Auxiliary index selection**

Choose a pilot statistic, e.g. the conditional mean of the observation:

$$
\mu_t^{(i)} = \phi_a\,\alpha_{t-1}^{(i)} + \beta_t^{(i)}\cdot r_t^{\mathrm{SPX}},
\quad
\tilde V_t^{(i)} = V_{t-1}^{(i)} + \kappa(\theta - V_{t-1}^{(i)})\Delta t
\tag{10}
$$

Pilot weight:

$$
g_t^{(i)} \propto p\!\left(y_t \mid \mu_t^{(i)}, \tilde V_t^{(i)}\right) \cdot w_{t-1}^{(i)}
\tag{11}
$$

Resample indices $\{a^{(i)}\}$ from the pilot weights.

**Stage 2 — Propagate and correct**

Propagate ancestor particles $\mathbf{x}_{t-1}^{(a^{(i)})}$ exactly as in the BPF, then
apply a correction weight:

$$
w_t^{(i)} \propto
\frac{p\!\left(y_t \mid \mathbf{x}_t^{(i)}\right)}{g_t^{(a^{(i)})}}
\tag{12}
$$

The incremental log-likelihood is:

$$
\log \hat p(y_t \mid y_{1:t-1})
= \log \frac{1}{N}\sum_i g_t^{(i)}
+ \log \frac{1}{N}\sum_i \frac{p(y_t \mid \mathbf{x}_t^{(i)})}{g_t^{(a^{(i)})}}
\tag{13}
$$

**Pros:** Lower variance weights; significantly more efficient per particle than BPF in
informative observation regimes.  
**Cons:** Requires two resampling passes per step; the pilot statistic must be a
**tight** approximation to the true predictive to yield gains (poor pilot → worse than BPF).

---

### 2.4 Optimal Proposal (Rao-Blackwellisation)

Because $\alpha_t$ and $\beta_t$ enter the observation **linearly**, conditional on $V_t$
the observation equation is

$$
y_t \mid V_t,\,r_t^{\mathrm{SPX}} \sim \mathcal{N}\!\bigl(\alpha_t + \beta_t\,r_t^{\mathrm{SPX}},\;V_t\Delta t\bigr)
\tag{14}
$$

This is a **linear-Gaussian** sub-model in $(\alpha_t, \beta_t)$ given $V_t$. One can
maintain a conditional **Kalman filter** over $(\alpha_t, \beta_t)$ for each particle
index in $V_t$, and only use SMC for $V_t$.

#### Rao-Blackwellised Particle Filter (RBPF)

| Component | Treated by |
|-----------|-----------|
| $V_t^{(i)}$ | Particle (CIR, non-Gaussian) |
| $(\alpha_t^{(i)}, \beta_t^{(i)})$ | Kalman filter conditioned on $V_t^{(i)}$ |

**Kalman update per particle $i$:**

State mean and covariance before step $t$:

$$
\mathbf{m}_{t-1}^{(i)} = \begin{pmatrix}\hat\alpha_{t-1}^{(i)}\\\hat\beta_{t-1}^{(i)}\end{pmatrix},
\quad
\mathbf{P}_{t-1}^{(i)} = \begin{pmatrix}p_{\alpha\alpha} & p_{\alpha\beta}\\p_{\alpha\beta} & p_{\beta\beta}\end{pmatrix}^{(i)}
$$

**Predict:**

$$
\mathbf{F} = \begin{pmatrix}\phi_a & 0 \\ 0 & \phi_b\end{pmatrix},
\quad
\mathbf{Q} = \begin{pmatrix}\sigma_a^2 & 0 \\ 0 & \sigma_b^2\end{pmatrix}
$$

$$
\mathbf{m}_{t|t-1}^{(i)} = \mathbf{F}\,\mathbf{m}_{t-1}^{(i)},
\quad
\mathbf{P}_{t|t-1}^{(i)} = \mathbf{F}\,\mathbf{P}_{t-1}^{(i)}\mathbf{F}^\top + \mathbf{Q}
\tag{15}
$$

**Observation matrix and noise** (conditioned on $V_t^{(i)}$):

$$
\mathbf{H}_t = \begin{pmatrix}1 & r_t^{\mathrm{SPX}}\end{pmatrix},
\quad
R_t^{(i)} = V_t^{(i)}\,\Delta t
$$

**Marginal likelihood** (used as particle weight):

$$
S_t^{(i)} = \mathbf{H}_t\,\mathbf{P}_{t|t-1}^{(i)}\,\mathbf{H}_t^\top + R_t^{(i)},
\quad
\nu_t^{(i)} = y_t - \mathbf{H}_t\,\mathbf{m}_{t|t-1}^{(i)}
$$

$$
\log \tilde w_t^{(i)} = -\tfrac{1}{2}\log(2\pi S_t^{(i)}) - \frac{(\nu_t^{(i)})^2}{2 S_t^{(i)}}
\tag{16}
$$

**Update:**

$$
\mathbf{K}_t^{(i)} = \mathbf{P}_{t|t-1}^{(i)}\,\mathbf{H}_t^\top / S_t^{(i)}
$$

$$
\mathbf{m}_t^{(i)} = \mathbf{m}_{t|t-1}^{(i)} + \mathbf{K}_t^{(i)}\,\nu_t^{(i)},
\quad
\mathbf{P}_t^{(i)} = \bigl(\mathbf{I} - \mathbf{K}_t^{(i)}\,\mathbf{H}_t\bigr)\,\mathbf{P}_{t|t-1}^{(i)}
\tag{17}
$$

The RBPF uses **only $N$ particles for the CIR dimension**, while the alpha/beta
posteriors are exact Gaussians. This yields dramatically lower Monte Carlo error for
the same $N$.

**Pros:** Optimal variance reduction; exact Gaussian posterior for $(\alpha_t, \beta_t)$
given $V_t$; well-suited to JAX via `vmap` over $i$.  
**Cons:** $O(N \times 4)$ additional memory for Kalman mean/covariance matrices; the
conditional Kalman gain must be computed for each particle; slight implementation
complexity.

---

### 2.5 Extended Kalman Filter (EKF) — Baseline

If $N$ is expensive, an **EKF** over the joint 3-dimensional state can serve as a
fast (but approximate) baseline. The nonlinearity lies only in the observation
variance: linearise $\sqrt{V_t}$ around the prior mean.

$$
h(\mathbf{x}_t) = \alpha_t + \beta_t\,r_t^{\mathrm{SPX}},
\quad
\frac{\partial h}{\partial \mathbf{x}_t} = (1,\; r_t^{\mathrm{SPX}},\; 0)
$$

The observation noise is state-dependent: $R_t = \hat V_t^{-}\,\Delta t$. The EKF is
consistent with the true model only at first order; it can diverge when $V_t$ is
unobservable (e.g. on days with very small $|r_t^{\mathrm{SPX}}|$). **Do not use as
the production filter**; use for parameter initialisation.

---

### 2.6 Parameter Estimation

The marginal log-likelihood from the particle filter is:

$$
\log \hat{\mathcal{L}}(\boldsymbol\psi) = \sum_{t=1}^T \log \hat p(y_t \mid y_{1:t-1};\boldsymbol\psi)
\tag{18}
$$

Three strategies:

| Method | Description | Notes |
|--------|-------------|-------|
| **MLE via gradient-free optimizer** | Evaluate $\hat{\mathcal{L}}$ for candidate $\boldsymbol\psi$; use CMA-ES or Nelder–Mead | Noisy objective; run multiple seeds |
| **PMCMC (particle MCMC)** | Use the particle filter inside an MCMC kernel as an unbiased estimator of the likelihood; see `docs/pmcmc_design.md` | Full Bayesian; expensive |
| **EM / online EM** | Sufficient statistics accumulated from particle smoother | Cleanest for AR(1) sub-models; complex for CIR |

For real-time use, fix $\boldsymbol\psi$ (estimated offline) and run only the particle
filter online.

---

## Part III — Pros, Cons, and Alternatives

### 3.1 Comparison Table

| Method | Statistical Efficiency | Computational Cost | Exact? | JAX-Friendly |
|--------|----------------------|-------------------|--------|-------------|
| BPF | Low–moderate ($O(N^{-1/2})$ error) | 1 resample / step | No | Yes (scan) |
| APF | Moderate–high | 2 resamples / step | No | Yes (scan) |
| RBPF | High (only variance is SMC) | 1 resample + $N$ Kalman updates | Exact in $(\alpha,\beta)$ | Yes (vmap over $i$) |
| EKF | Low (linearisation bias) | $O(d^2)$ per step | No | Yes |
| UKF | Moderate (better nonlinearity) | $O(d^3)$ per step | No | Yes |
| Particle MCMC | Very high | $O(N \times T)$ per iteration | Asymptotically | Yes |
| Variational SMC | High with amortisation | Moderate | No | JAX/normalising flows |

### 3.2 Model Assumptions and Risks

**Pros:**
- AR(1) dynamics for $\alpha_t, \beta_t$ are parsimonious and have well-understood
  stationary distributions; the unconditional variances are identifiable.
- CIR for $V_t$ ensures non-negative variance with analytical transition density
  (noncentral chi-squared), enabling exact simulation (as opposed to Euler
  approximation) via the Broadie–Kaya algorithm or the quadratic-exponential scheme.
- The RBPF exploits linear-Gaussian structure exactly, achieving the best possible
  efficiency for this model class.
- The model decomposes total return variance into systematic ($\beta_t^2\,\mathrm{Var}(r^{\mathrm{SPX}})$)
  and idiosyncratic ($V_t\,\Delta t$) components — useful for risk attribution.

**Cons / Risks:**
- **Factor incompleteness:** A single SPX factor leaves multi-sector stocks
  under-explained. Consider adding sector ETF factors or Fama-French factors as
  additional regressors $\{r_t^{f_k}\}$ with corresponding $\beta_t^{(k)}$.
- **Leverage effect ignored:** The model has $\mathrm{corr}(\varepsilon_t, \xi_t^V) = 0$.
  In practice equities exhibit a strong leverage effect ($V$ spikes when price falls);
  ignoring it biases $\hat\alpha_t$ downward after drawdowns.
- **Jump risk in alpha/beta:** Real alpha/beta can have sudden structural breaks
  (earnings, index rebalancing). The AR(1) diffusion model is slow to react. Consider
  a **regime-switching** extension or a **heavy-tailed noise** model (Student-$t$)
  for $\xi_t^a, \xi_t^b$.
- **Parameter drift:** $\boldsymbol\psi$ itself may change on the timescale of months.
  Online parameter estimation (Liu-West filter with parameter jitter) or periodic
  re-calibration windows are required in production.
- **Variance identifiability:** Without direct variance observations (VIX, options),
  $V_t$ is identified solely through the squared residuals from the factor model.
  Short windows give noisy estimates; this is ameliorated by the RBPF's exact
  integration over $(\alpha, \beta)$.

### 3.3 Alternative Model Extensions

| Extension | Modification | When to Use |
|-----------|-------------|-------------|
| Multi-factor CAPM | Add $K$ factors; $\boldsymbol\beta_t \in \mathbb{R}^K$ with VAR(1) dynamics | Sector/style exposure |
| Regime-switching alpha | $\alpha_t$ follows a hidden Markov model over regimes | Earnings, M&A events |
| Heston with jumps | Add Merton/Kou jump term to $V_t$ equation (5) | Fat-tailed IV, overnight gaps |
| Non-Gaussian state noise | Replace $\xi_t^a, \xi_t^b$ with Student-$t(\nu)$ | Tail robustness |
| Correlated alpha-vol | $\mathrm{corr}(\varepsilon_t, \xi_t^V) = \rho_{\varepsilon V}$ | Leverage effect |
| Realized variance measurement | Augment $y_t$ with 5-min realized variance as a noisy $V_t$ proxy | Faster variance identification |

---

## Part IV — Portfolio Optimization Using Filtered States

### 4.1 Extracted Signals

At each time $t$, the particle filter produces posterior moments:

$$
\hat\alpha_t = \mathbb{E}[\alpha_t \mid y_{1:t}],\quad
\hat\beta_t = \mathbb{E}[\beta_t \mid y_{1:t}],\quad
\hat V_t = \mathbb{E}[V_t \mid y_{1:t}]
$$

plus posterior uncertainty $\mathrm{Var}(\alpha_t \mid y_{1:t})$,
$\mathrm{Var}(\beta_t \mid y_{1:t})$, enabling **uncertainty-aware** position sizing.

For an $M$-asset universe, each asset $m$ has its own particle filter yielding
$\hat\alpha_t^{(m)}, \hat\beta_t^{(m)}, \hat V_t^{(m)}$ (independently or jointly via
a shared SPX factor).

---

### 4.2 Expected Return and Covariance Reconstruction

**One-step-ahead expected return** for asset $m$:

$$
\hat\mu_t^{(m)} = \hat\alpha_t^{(m)} + \hat\beta_t^{(m)}\,\mathbb{E}[r_{t+1}^{\mathrm{SPX}}]
\tag{19}
$$

If $\mathbb{E}[r^{\mathrm{SPX}}] \approx 0$ over the prediction horizon (or is set to a
prior estimate), then $\hat\mu_t^{(m)} \approx \hat\alpha_t^{(m)}$.

**Covariance matrix** of the $M$-asset return vector:

$$
\boldsymbol\Sigma_t = \underbrace{\hat{\boldsymbol\beta}_t\,\hat{\boldsymbol\beta}_t^\top
  \cdot \hat\sigma_{\mathrm{SPX}}^2}_{\text{systematic risk}}
+ \underbrace{\mathrm{diag}\!\bigl(\hat V_t^{(1)},\ldots,\hat V_t^{(M)}\bigr)\cdot\Delta t}_{\text{idiosyncratic risk}}
\tag{20}
$$

where $\hat{\boldsymbol\beta}_t = (\hat\beta_t^{(1)},\ldots,\hat\beta_t^{(M)})^\top$
and $\hat\sigma_{\mathrm{SPX}}^2$ is a short-window SPX variance estimate.

This is a **one-factor structure** with time-varying loadings. For $M \gg 1$, the
Woodbury identity reduces $\boldsymbol\Sigma_t^{-1}$ to $O(M)$ operations:

$$
\boldsymbol\Sigma_t^{-1} = \mathbf{D}_t^{-1}
- \frac{\mathbf{D}_t^{-1}\hat{\boldsymbol\beta}_t\,\hat{\boldsymbol\beta}_t^\top \mathbf{D}_t^{-1}}
       {\hat\sigma_{\mathrm{SPX}}^{-2} + \hat{\boldsymbol\beta}_t^\top \mathbf{D}_t^{-1}\hat{\boldsymbol\beta}_t}
\tag{21}
$$

with $\mathbf{D}_t = \mathrm{diag}(\hat V_t^{(1)}\Delta t,\ldots,\hat V_t^{(M)}\Delta t)$.

---

### 4.3 Classical Mean-Variance Optimization (Markowitz)

Given $\hat{\boldsymbol\mu}_t \in \mathbb{R}^M$ and $\boldsymbol\Sigma_t \in \mathbb{R}^{M\times M}$,
solve the Markowitz program:

$$
\max_{\mathbf{w} \in \mathcal{C}}\;
\mathbf{w}^\top \hat{\boldsymbol\mu}_t
- \frac{\gamma}{2}\,\mathbf{w}^\top\boldsymbol\Sigma_t\,\mathbf{w}
\tag{22}
$$

Common constraint sets $\mathcal{C}$:

| Constraint | Expression | Purpose |
|------------|-----------|---------|
| Budget | $\mathbf{1}^\top\mathbf{w} = 1$ | Fully invested |
| Long-only | $\mathbf{w} \geq \mathbf{0}$ | No short-selling |
| Long-short dollar-neutral | $\mathbf{1}^\top\mathbf{w} = 0$ | Market-neutral |
| Beta-neutral | $\hat{\boldsymbol\beta}_t^\top\mathbf{w} = 0$ | Factor-neutral book |
| Position limits | $-c \leq w_m \leq c$ | Risk limits |
| Turnover | $\|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1 \leq \tau$ | Transaction cost control |

The **beta-neutral** constraint is particularly natural here: since $\hat\beta_t^{(m)}$
is continuously estimated, one can maintain $\hat{\boldsymbol\beta}_t^\top\mathbf{w}_t = 0$
to trade **only the idiosyncratic alpha**, with zero net exposure to the SPX.

**Analytical solution** (no inequality constraints, budget only):

$$
\mathbf{w}_t^* = \frac{1}{\gamma}\boldsymbol\Sigma_t^{-1}
\Bigl(\hat{\boldsymbol\mu}_t - \lambda_t\,\mathbf{1}\Bigr),
\quad
\lambda_t = \frac{\mathbf{1}^\top\boldsymbol\Sigma_t^{-1}\hat{\boldsymbol\mu}_t - \gamma}
                 {\mathbf{1}^\top\boldsymbol\Sigma_t^{-1}\mathbf{1}}
\tag{23}
$$

With the Woodbury formula (21) this is $O(M)$ per step.

---

### 4.4 Uncertainty-Adjusted Position Sizing

The filtered alpha carries posterior variance $\hat\sigma_{\alpha,t}^{(m)2}$. A
conservative signal-to-noise adjusted sizing:

$$
\hat\mu_t^{(m),\text{adj}}
= \frac{\hat\alpha_t^{(m)}}
       {1 + \hat\sigma_{\alpha,t}^{(m)} / |\hat\alpha_t^{(m)}|}
\tag{24}
$$

This shrinks positions automatically when the filter is uncertain (high variance →
shrinkage towards zero). Combined with $\boldsymbol\Sigma_t$ from (20), this is a
**Bayesian plug-in Markowitz** rule.

---

### 4.5 Black-Litterman with Filtered Views

The Black-Litterman model treats $\hat{\boldsymbol\alpha}_t$ as **investor views**
overlaid on a prior return $\boldsymbol\Pi_t = \gamma\boldsymbol\Sigma_t\,\mathbf{w}^{\mathrm{mkt}}$
(equilibrium returns from market-cap weights $\mathbf{w}^{\mathrm{mkt}}$):

$$
\boldsymbol\mu_t^{\mathrm{BL}}
= \bigl[(\tau\boldsymbol\Sigma_t)^{-1} + \mathbf{P}^\top\boldsymbol\Omega_t^{-1}\mathbf{P}\bigr]^{-1}
  \bigl[(\tau\boldsymbol\Sigma_t)^{-1}\boldsymbol\Pi_t + \mathbf{P}^\top\boldsymbol\Omega_t^{-1}\mathbf{q}_t\bigr]
\tag{25}
$$

Here $\mathbf{P} = \mathbf{I}$ (absolute views), $\mathbf{q}_t = \hat{\boldsymbol\alpha}_t$,
and $\boldsymbol\Omega_t = \mathrm{diag}(\hat\sigma_{\alpha,t}^{(m)2})$ — the filtered
posterior variance directly becomes the view uncertainty matrix, making the integration
end-to-end coherent.

---

### 4.6 When Classical Methods Break Down

Classical Markowitz is **static** within each step and ignores:

1. **Transaction costs and market impact** — myopic MVO re-optimises each step,
   ignoring the cost of moving between consecutive optimal portfolios.
2. **Holding period** — MVO gives an instantaneous allocation; it does not decide
   *when to enter* and *when to exit* a trade.
3. **PATH DEPENDENCY** — a signal that is strong for 3 days and then reverses should
   be traded differently from one that is persistent. MVO treats all signals as
   interchangeable.
4. **Non-convex constraints** — minimum lot sizes, discrete positions, shorting fees,
   margin calls.
5. **Distribution shift / overfitting** — if $\boldsymbol\psi$ is re-estimated too
   frequently from in-sample data, MVO can over-trade and fit noise.

Reinforcement Learning — specifically **PPO** — can address all five, at the cost of
greater implementation complexity and the risk of overfitting the reward during training.

---

## Part V — PPO-Based Trading Strategy

### 5.1 MDP Formulation

| MDP Element | Definition |
|-------------|-----------|
| State $s_t$ | $(\hat{\boldsymbol\alpha}_t, \hat{\boldsymbol\beta}_t, \hat{\mathbf{V}}_t, \mathbf{w}_{t-1}, r_{t-1}^{\mathrm{SPX}}, \mathbf{r}_{t-1}^X)$ |
| Action $a_t$ | Target portfolio weights $\mathbf{w}_t \in \Delta^M$ (or $\Delta^{M,\mathrm{L/S}}$) |
| Transition | Filtered state update from particle filter step; price simulation or live data |
| Reward $R_t$ | Risk-adjusted P&L (see §5.3) |
| Episode length | $T_{\mathrm{ep}}$ trading steps (e.g. 1 day, 1 week) |
| Terminal | End of episode or drawdown stop |

Including the **current portfolio weights $\mathbf{w}_{t-1}$** in the state is critical:
it allows the agent to learn optimal rebalancing frequency (acting implies turnover cost).

---

### 5.2 Action Space Design

**Option A — Continuous allocation:**

$$
a_t \in [-1, 1]^M, \quad \text{then projected/normalised to } \mathcal{C}
$$

Use a softmax for long-only or a tanh-normalised map for long-short. PPO's clipped
surrogate objective handles continuous actions via a diagonal Gaussian policy.

**Option B — Discretised "trade/hold/exit":**

Define three actions per asset: increase position, hold, decrease. This scales poorly
to large $M$ (action space explodes) but is intuitive for a single asset.

**Option C — Hierarchical action** (recommended for larger $M$):

1. High-level policy: decide which assets to include (subset selection).
2. Low-level policy: given the active set, optimise weights via a differentiable
   Markowitz layer (as in the SPO framework — embed a QP solver inside the network).

---

### 5.3 Reward Function Design

A poor reward leads to "reward hacking" — the agent finds degenerate strategies
(e.g. zero risk → zero return). Recommended layered reward:

$$
R_t = \underbrace{\mathbf{w}_t^\top \mathbf{r}_{t+1}^X}_{\text{P\&L}}
     - \underbrace{\lambda_c\,\|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1}_{\text{turnover cost}}
     - \underbrace{\lambda_r\,\mathbf{w}_t^\top\boldsymbol\Sigma_t\,\mathbf{w}_t}_{\text{risk penalty}}
     - \underbrace{\lambda_d\,D_t^+}_{\text{drawdown penalty}}
\tag{26}
$$

where $D_t^+ = \max(0, \mathrm{peak value} - \mathrm{current value})$. The drawdown
penalty discourages the agent from "riding" losses.

**Sharpe-penalised variant** (penalise each episode):

$$
R^{\mathrm{ep}} = \frac{\bar{R}}{\hat\sigma_R} - \lambda_c\,\sum_t\|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1
\tag{27}
$$

where $\bar{R}$ and $\hat\sigma_R$ are the mean and std of step returns within the
episode. This directly maximises the Sharpe ratio but introduces a non-Markovian
episode-level reward — must be handled via advantage estimation with careful
baselines.

---

### 5.4 PPO Architecture

```
Observation encoder:
  s_t = [alpha_hat (M), beta_hat (M), V_hat (M), w_{t-1} (M), r_SPX, r_X (M)]
        ↓  LayerNorm
        Linear(state_dim → 256) → GELU
        Linear(256 → 256) → GELU

Policy head (actor):
        Linear(256 → M) → softmax  [long-only]
        or
        Linear(256 → M) → tanh × w_max  [long-short]

Value head (critic):
        Linear(256 → 128) → GELU → Linear(128 → 1)
```

The encoder can optionally include an LSTM/GRU layer to capture temporal structure
beyond what the Markovian state provides (e.g. multi-step momentum):

```
GRU(input_dim=state_dim, hidden_dim=256)
        ↓
shared_features
```

This is especially relevant when the AR(1) dynamics of $\alpha_t, \beta_t$ should
be inferred implicitly from the raw observation stream rather than the pre-filtered
state.

---

### 5.5 What PPO Learns vs Classical MVO

| Capability | MVO | PPO |
|------------|-----|-----|
| Mean-variance efficient allocation | ✓ (optimal) | ✓ (approximates) |
| Transaction cost awareness | Penalty term only | ✓ (directly in reward) |
| When to enter/exit a trade | ✗ | ✓ |
| How long to hold | ✗ | ✓ |
| Drawdown control | ✗ | ✓ (with reward shaping) |
| Non-convex constraints | ✗ | ✓ |
| Regime adaptation (no re-calibration) | ✗ | ✓ (if trained on diverse regimes) |
| Interpretability | ✓ (analytical weights) | ✗ |
| Overfitting risk | Moderate (estimation error) | High (distributional shift) |

---

### 5.6 Anti-Overfitting Strategies for the PPO Agent

Overfitting in RL trading is **qualitatively different** from supervised learning: the
agent may find spurious patterns in the replay distribution that do not hold
out-of-sample. Mitigations:

1. **Curriculum via synthetic data:** Train on particle filter rollouts from the
   generative model (§1) with randomised $\boldsymbol\psi$. The agent sees many
   market regimes without overfitting to one historical path.

2. **Randomised transaction costs and slippage:** Randomly sample $\lambda_c$ and
   bid-ask spread from plausible ranges during training. Prevents the agent from
   exploiting zero-cost assumptions.

3. **Data augmentation via permuted episodes:** Randomly permute the ordering of
   non-overlapping sub-windows when constructing training episodes. This breaks
   spurious serial correlations.

4. **Regularisation in the policy network:** L2 weight decay on the actor;
   entropy bonus $+\lambda_H\,\mathcal{H}[\pi_\theta(\cdot|s_t)]$ promotes
   exploration and prevents degenerate deterministic policies.

5. **Walk-forward validation:** Use a strict expanding-window regime: train only on
   $[0, t_{\mathrm{train}}]$, validate on $[t_{\mathrm{train}}, t_{\mathrm{val}}]$,
   test on $[t_{\mathrm{val}}, T]$. Never use test data for hyperparameter tuning.

6. **Behavioural cloning warm-start:** Pre-train the policy network to imitate the
   MVO solution (§4.3) via supervised loss before RL fine-tuning. This anchors the
   agent near a known-good baseline, shortening the exploration phase and reducing
   spurious early exploration.

---

### 5.7 Hybrid Architecture: MVO-Residual PPO

A practical and interpretable hybrid:

$$
\mathbf{w}_t = \underbrace{\mathbf{w}_t^{\mathrm{MVO}}}_{\text{analytical baseline}}
             + \underbrace{\pi_\theta(s_t)}_{\text{PPO residual (small)}}
\tag{28}
$$

The PPO agent only learns a **residual adjustment** from the MVO solution, bounded by
$\|\pi_\theta\|_\infty \leq \delta_w$ (e.g. $\delta_w = 0.05$). This enforces that
the strategy remains close to the well-understood classical solution while learning
to correct for its blind spots (timing, transaction costs, drawdown avoidance).
The residual is much easier to train and much less likely to overfit.

---

## Part VI — Final Design: Leverage Effect and Online Parameter Estimation

This section assembles all components into the production filter. It (i) extends the
model with a leverage correlation $\rho_V$, (ii) verifies that Rao-Blackwellisation
remains analytically exact despite the correlation, and (iii) integrates Liu-West
parameter jitter to enable online estimation of the full parameter vector
$\boldsymbol\psi$.

---

### 6.1 Extended Model — Leverage Effect

Introduce a correlation $\rho_V \in (-1, 1)$ between the idiosyncratic price shock
$\varepsilon_t$ and the CIR variance noise $\xi_t^V$. Decompose:

$$
\varepsilon_t = \rho_V\,\xi_t^V + \sqrt{1-\rho_V^2}\,\xi_t^\perp,
\qquad \xi_t^\perp \overset{\text{iid}}{\sim}\mathcal{N}(0,1),
\quad \xi_t^\perp \perp \xi_t^V,\;\xi_t^a,\;\xi_t^b
\tag{29}
$$

Substituting into (1) and using the Euler–Maruyama convention of evaluating the
diffusion coefficient at $V_{t-1}$:

$$
\boxed{
r_t^X = \alpha_t + \beta_t\,r_t^{\mathrm{SPX}}
      + \rho_V\sqrt{V_{t-1}\,\Delta t}\;\xi_t^V
      + \sqrt{(1-\rho_V^2)\,V_{t-1}\,\Delta t}\;\xi_t^\perp
}
\tag{30}
$$

The extended parameter vector becomes:

$$
\boldsymbol\psi = (\phi_a,\,\sigma_a,\;\phi_b,\,\sigma_b,\,\bar\beta,\;
                   \kappa,\,\theta,\,\sigma_V,\,V_0,\,\rho_V)
$$

**Economic interpretation.**
Negative $\rho_V$ captures the standard equity leverage effect: a large downward
idiosyncratic shock ($\varepsilon_t < 0$) is correlated with a jump in variance
($\xi_t^V > 0$). Without this term, the filter systematically underestimates $V_t$
after sharp drawdowns and biases $\hat\alpha_t$ negatively in those periods.

---

### 6.2 RBPF Tractability Check under Leverage

**Central question.** Does $\mathrm{corr}(\varepsilon_t, \xi_t^V) = \rho_V \neq 0$ break
the conditional linear-Gaussian structure that makes the Rao-Blackwellised Particle
Filter exact?

**Answer: No.** The key is that the CIR propagation step explicitly samples
$\xi_t^{V,(i)}$, making it a known quantity at Stage 2. Once $\xi_t^{V,(i)}$ is in
hand, isolate the unknown part of (30) by defining the
**leverage-corrected observation** per particle $i$:

$$
\tilde y_t^{(i)}
\;\equiv\;
r_t^X - \rho_V\sqrt{V_{t-1}^{(i)}\,\Delta t}\;\xi_t^{V,(i)}
\tag{31}
$$

Substituting into (30):

$$
\tilde y_t^{(i)} = \alpha_t + \beta_t\,r_t^{\mathrm{SPX}}
                  + \sqrt{(1-\rho_V^2)\,V_{t-1}^{(i)}\,\Delta t}\;\xi_t^\perp
\tag{32}
$$

This is **linear-Gaussian in $(\alpha_t, \beta_t)$** with the same observation matrix
$\mathbf{H}_t = (1,\;r_t^{\mathrm{SPX}})$ as in the no-leverage case, but with a
reduced observation noise:

$$
\tilde R_t^{(i)} = (1 - \rho_V^2)\,V_{t-1}^{(i)}\,\Delta t
\tag{33}
$$

The Kalman update (15)–(17) applies exactly with the substitutions
$y_t \to \tilde y_t^{(i)}$ and $R_t^{(i)} \to \tilde R_t^{(i)}$. The RBPF is
unchanged in structure; leverage only introduces a particle-specific shift in the
observation and a tighter observation variance.

#### APF Pilot Weight is Unchanged

In Stage 1, $\xi_t^{V,(i)}$ has not yet been sampled, so we must marginalise (30)
over $\xi_t^V \sim \mathcal{N}(0,1)$. The $\rho_V$ and $\sqrt{1-\rho_V^2}$ terms
correspond to two independent zero-mean Gaussian contributions to $r_t^X$; their
variances sum to $V_{t-1}^{(i)}\,\Delta t$ regardless of $\rho_V$:

$$
\mathrm{Var}(r_t^X \mid \mathbf{m}_{t|t-1}^{(i)}, V_{t-1}^{(i)})
= \mathbf{H}_t\,\mathbf{P}_{t|t-1}^{(i)}\,\mathbf{H}_t^\top
+ \underbrace{\rho_V^2\,V_{t-1}^{(i)}\,\Delta t
  + (1-\rho_V^2)\,V_{t-1}^{(i)}\,\Delta t}_{= V_{t-1}^{(i)}\,\Delta t}
= S_t^{(i)}
\tag{34}
$$

The $\rho_V$ contributions cancel exactly. The APF pilot weight
$g_t^{(i)} \propto \mathcal{N}(r_t^X;\,\mathbf{H}_t\mathbf{m}_{t|t-1}^{(i)},\,S_t^{(i)})$
is **identical** to the no-leverage formula — no code change is required in Stage 1.

#### Summary of Changes vs No-Leverage RBPF

| RBPF step | Without leverage ($\rho_V = 0$) | With leverage ($\rho_V \neq 0$) |
|-----------|--------------------------------|--------------------------------|
| APF Stage 1 pilot $g_t^{(i)}$ | $\mathcal{N}(r_t^X;\,\mu_t^{(i)},\,S_t^{(i)})$ | **Same** |
| Effective observation | $y_t$ | $\tilde y_t^{(i)} = y_t - \rho_V\sqrt{V_{t-1}^{(i)}\Delta t}\,\xi_t^{V,(i)}$ |
| Observation noise $R_t^{(i)}$ | $V_{t-1}^{(i)}\,\Delta t$ | $(1-\rho_V^2)\,V_{t-1}^{(i)}\,\Delta t$ |
| Innovation $\tilde\nu_t^{(i)}$ | $y_t - \mathbf{H}_t\mathbf{m}_{t|t-1}^{(i)}$ | $\tilde y_t^{(i)} - \mathbf{H}_t\mathbf{m}_{t|t-1}^{(i)}$ |
| Weight $\tilde S_t^{(i)}$ | $\mathbf{H}_t\mathbf{P}_{t|t-1}^{(i)}\mathbf{H}_t^\top + V_{t-1}^{(i)}\Delta t$ | $\mathbf{H}_t\mathbf{P}_{t|t-1}^{(i)}\mathbf{H}_t^\top + (1-\rho_V^2)V_{t-1}^{(i)}\Delta t$ |

Leverage makes the particle weights sharper (smaller $\tilde S_t^{(i)}$), which can
increase weight variance. The APF pilot pre-selects against this by concentrating
ancestors near high-likelihood regions before the sharp Kalman update.

---

### 6.3 Online Parameter Estimation — Liu-West Augmentation

Augment each particle's state with the full parameter vector so the filter jointly
estimtes states and parameters online:

$$
z_t^{(i)} = \bigl(V_t^{(i)},\;\mathbf{m}_t^{(i)},\;\mathbf{P}_t^{(i)},\;\boldsymbol\phi^{(i)}\bigr)
$$

where $\boldsymbol\phi^{(i)} = T(\boldsymbol\psi^{(i)})$ stores parameters in
**unconstrained space** to prevent jitter from violating bounds.

#### Unconstrained Transforms

| Parameter | Transform $T(\cdot)$ | Inverse | Constraint |
|-----------|-------------------|---------|------------|
| $\sigma_a,\,\sigma_b,\,\sigma_V,\,\theta,\,V_0$ | $\log(\cdot)$ | $\exp(\cdot)$ | $> 0$ |
| $\phi_a,\,\phi_b$ | $\tanh^{-1}(\cdot)$ | $\tanh(\cdot)$ | $(-1,1)$ |
| $\rho_V$ | $\tanh^{-1}(\cdot)$ | $\tanh(\cdot)$ | $(-1,1)$ |
| $\kappa$ | $\log(\cdot)$ | $\exp(\cdot)$ | $> 0$ |
| $\bar\beta$ | identity | identity | $\mathbb{R}$ |

#### Liu-West Shrink-and-Jitter (in Unconstrained Space)

After APF Stage 1 resample with ancestor indices $\{a^{(i)}\}$, apply:

$$
\boldsymbol\phi^{(i),\text{new}}
= a\,\boldsymbol\phi^{(a^{(i)})}
+ (1-a)\,\bar{\boldsymbol\phi}_t
+ h\,\boldsymbol\eta^{(i)},
\qquad \boldsymbol\eta^{(i)} \sim \mathcal{N}(\mathbf{0},\,\mathbf{V}_t^{\boldsymbol\phi})
\tag{35}
$$

where
$$
\bar{\boldsymbol\phi}_t = \frac{1}{N}\sum_i\boldsymbol\phi^{(i)},
\quad
\mathbf{V}_t^{\boldsymbol\phi} = \frac{1}{N}\sum_i
  (\boldsymbol\phi^{(i)}-\bar{\boldsymbol\phi}_t)(\boldsymbol\phi^{(i)}-\bar{\boldsymbol\phi}_t)^\top,
\quad a = \sqrt{1-h^2}
$$

Transform back: $\boldsymbol\psi^{(i),\text{new}} = T^{-1}(\boldsymbol\phi^{(i),\text{new}})$.
Constraints are automatically satisfied by construction.

**Recommended default:** $h = 0.1$–$0.2$ (jitter 1–4% extra variance; $a \approx
0.995$–$0.98$). Use a smaller $h$ for fast-moving market data to prevent parameter
wandering; larger $h$ when the filter is deployed on longer (weekly/monthly) intervals
where parameter drift is expected.

#### Particle-Specific Kalman Matrices

With $\boldsymbol\psi^{(i)}$ varying across particles, the Kalman prediction matrices
become particle-specific:

$$
\mathbf{F}^{(i)} = \begin{pmatrix}\phi_a^{(i)} & 0 \\ 0 & \phi_b^{(i)}\end{pmatrix},
\qquad
\mathbf{Q}^{(i)} = \begin{pmatrix}(\sigma_a^{(i)})^2 & 0 \\ 0 & (\sigma_b^{(i)})^2\end{pmatrix}
\tag{36}
$$

The Kalman predict step (15) becomes:

$$
\mathbf{m}_{t|t-1}^{(i)} = \mathbf{F}^{(i)}\,\mathbf{m}_{t-1}^{(i)},
\qquad
\mathbf{P}_{t|t-1}^{(i)} =
  \mathbf{F}^{(i)}\,\mathbf{P}_{t-1}^{(i)}\,(\mathbf{F}^{(i)})^\top + \mathbf{Q}^{(i)}
$$

This is a valid conditional Kalman filter. Rao-Blackwellisation holds because the
conditioning set now includes $(V_t^{(i)}, \xi_t^{V,(i)}, \boldsymbol\psi^{(i)})$,
and the observation (32) remains linear-Gaussian in $(\alpha_t, \beta_t)$ given
this enriched conditioning set.

---

### 6.4 Complete Filter Step — RBPF + Leverage + Liu-West

**Carry at step $t-1$** (per particle $i = 1,\ldots,N$):

$$
z_{t-1}^{(i)} = \bigl(V_{t-1}^{(i)},\;\mathbf{m}_{t-1}^{(i)},\;\mathbf{P}_{t-1}^{(i)},\;\boldsymbol\phi^{(i)}\bigr)
$$

**Inputs at step $t$:** $(r_t^X,\;r_t^{\mathrm{SPX}},\;\Delta t)$.

---

**Step 1 — Cloud statistics and pilot parameters.**

$$
\bar{\boldsymbol\phi}_t = \tfrac{1}{N}\textstyle\sum_i\boldsymbol\phi^{(i)},
\quad
\mathbf{V}_t^{\boldsymbol\phi} = \tfrac{1}{N}\textstyle\sum_i
  (\boldsymbol\phi^{(i)}-\bar{\boldsymbol\phi}_t)(\cdots)^\top
$$

Shrunk pilot parameters (pre-jitter look-ahead for APF):

$$
\tilde{\boldsymbol\phi}^{(i)} = a\,\boldsymbol\phi^{(i)} + (1-a)\,\bar{\boldsymbol\phi}_t,
\qquad
\tilde{\boldsymbol\psi}^{(i)} = T^{-1}(\tilde{\boldsymbol\phi}^{(i)})
$$

**Step 2 — Kalman predict with pilot parameters.**

$$
\mathbf{m}_{t|t-1}^{(i)} = \tilde{\mathbf{F}}^{(i)}\,\mathbf{m}_{t-1}^{(i)},
\qquad
\mathbf{P}_{t|t-1}^{(i)} =
  \tilde{\mathbf{F}}^{(i)}\mathbf{P}_{t-1}^{(i)}(\tilde{\mathbf{F}}^{(i)})^\top
  + \tilde{\mathbf{Q}}^{(i)}
$$

**Step 3 — APF Stage 1: pilot weights and resample.**

$$
S_t^{(i)} = \mathbf{H}_t\,\mathbf{P}_{t|t-1}^{(i)}\,\mathbf{H}_t^\top
             + \tilde V_{t-1}^{(i)}\Delta t,
\qquad
\nu^{(i)} = r_t^X - \mathbf{H}_t\,\mathbf{m}_{t|t-1}^{(i)}
$$

$$
\log g_t^{(i)} = -\tfrac{1}{2}\log(2\pi S_t^{(i)}) - \frac{(\nu^{(i)})^2}{2 S_t^{(i)}}
$$

Draw ancestor indices $\{a^{(i)}\}$ from normalised $\{g_t^{(i)}\}$ via systematic
resampling.

**Step 4 — Liu-West jitter.**

$$
\boldsymbol\phi^{(i),\mathrm{new}}
= a\,\boldsymbol\phi^{(a^{(i)})} + (1-a)\bar{\boldsymbol\phi}_t + h\,\boldsymbol\eta^{(i)},
\qquad \boldsymbol\eta^{(i)} \sim \mathcal{N}(\mathbf{0},\mathbf{V}_t^{\boldsymbol\phi})
$$

$$
\boldsymbol\psi^{(i),\mathrm{new}} = T^{-1}(\boldsymbol\phi^{(i),\mathrm{new}})
$$

**Step 5 — Kalman predict with jittered parameters** (reusing ancestors).

$$
\mathbf{m}_{t|t-1}^{(i)} =
  \mathbf{F}^{(i),\mathrm{new}}\,\mathbf{m}_{t-1}^{(a^{(i)})},
\qquad
\mathbf{P}_{t|t-1}^{(i)} =
  \mathbf{F}^{(i),\mathrm{new}}\mathbf{P}_{t-1}^{(a^{(i)})}(\mathbf{F}^{(i),\mathrm{new}})^\top
  + \mathbf{Q}^{(i),\mathrm{new}}
$$

**Step 6 — Propagate CIR variance.**

$$
\xi_t^{V,(i)} \sim \mathcal{N}(0,1)
$$

$$
V_t^{(i)} = \max\!\Bigl(
  V_{t-1}^{(a^{(i)})}
  + \kappa^{(i)}\bigl(\theta^{(i)} - V_{t-1}^{(a^{(i)})}\bigr)\Delta t
  + \sigma_V^{(i)}\sqrt{V_{t-1}^{(a^{(i)})}\,\Delta t}\;\xi_t^{V,(i)},\;
  \delta\Bigr)
$$

**Step 7 — Leverage-corrected Kalman update.**

$$
\tilde y_t^{(i)} = r_t^X - \rho_V^{(i)}\sqrt{V_{t-1}^{(a^{(i)})}\,\Delta t}\;\xi_t^{V,(i)},
\qquad
\tilde R_t^{(i)} = \bigl(1-(\rho_V^{(i)})^2\bigr)\,V_{t-1}^{(a^{(i)})}\,\Delta t
$$

$$
\tilde S_t^{(i)} = \mathbf{H}_t\,\mathbf{P}_{t|t-1}^{(i)}\,\mathbf{H}_t^\top + \tilde R_t^{(i)},
\qquad
\tilde\nu_t^{(i)} = \tilde y_t^{(i)} - \mathbf{H}_t\,\mathbf{m}_{t|t-1}^{(i)}
$$

$$
\mathbf{K}_t^{(i)} = \mathbf{P}_{t|t-1}^{(i)}\,\mathbf{H}_t^\top / \tilde S_t^{(i)}
$$

$$
\mathbf{m}_t^{(i)} = \mathbf{m}_{t|t-1}^{(i)} + \mathbf{K}_t^{(i)}\,\tilde\nu_t^{(i)},
\qquad
\mathbf{P}_t^{(i)} = \bigl(\mathbf{I} - \mathbf{K}_t^{(i)}\,\mathbf{H}_t\bigr)\,\mathbf{P}_{t|t-1}^{(i)}
$$

**Step 8 — Correction weights, log-likelihood, and ESS.**

$$
\log\tilde w_t^{(i)}
= -\tfrac{1}{2}\log(2\pi\tilde S_t^{(i)})
  - \frac{(\tilde\nu_t^{(i)})^2}{2\tilde S_t^{(i)}}
  - \log g_t^{(a^{(i)})}
$$

$$
\log\hat p(y_t \mid y_{1:t-1})
= \mathrm{LSE}\bigl(\log g_t^{(i)}\bigr) - \log N
+ \mathrm{LSE}\bigl(\log\tilde w_t^{(i)}\bigr) - \log N
$$

Normalise $w_t^{(i)}$; compute ESS; resample if $\mathrm{ESS} < N/2$.

---

### 6.5 JAX Carry Structure

```python
carry = (
    V,         # (N,)           CIR variance particles
    m,         # (N, 2)         Kalman means [alpha_hat, beta_hat]
    P,         # (N, 2, 2)      Kalman covariances
    phi_unc,   # (N, D_psi)     unconstrained parameters, D_psi = 10
    loglik,    # scalar         accumulated log-likelihood
)
```

- Cloud statistics $\bar{\boldsymbol\phi}_t$, $\mathbf{V}_t^{\boldsymbol\phi}$ are
  computed with `jnp.mean` / `jnp.cov` over axis 0 **before** the per-particle
  `vmap` (these are `jnp` reductions over $N$, compatible with `scan`).
- Steps 2–8 are `vmap`'d over particle index $i$, each taking its own slice of the
  carry and the shared $(r_t^X,\,r_t^{\mathrm{SPX}},\,\Delta t,\,\bar{\boldsymbol\phi}_t,\,\mathbf{V}_t^{\boldsymbol\phi})$.
- The systematic resample in Step 8 breaks the per-particle independence (it needs
  the full weight vector); this is handled outside the `vmap`, identically to the
  existing `InhomoHestonProcess` pattern.

---

### 6.6 Posterior Extraction

At each step the filter yields:

| Quantity | Formula | Notes |
|----------|---------|-------|
| $\hat\alpha_t$ | $\sum_i w_t^{(i)}\,\mathbf{m}_t^{(i)}[0]$ | Particle-weighted Kalman mean |
| $\hat\beta_t$ | $\sum_i w_t^{(i)}\,\mathbf{m}_t^{(i)}[1]$ | |
| $\hat V_t$ | $\sum_i w_t^{(i)}\,V_t^{(i)}$ | |
| $\mathrm{Var}(\alpha_t)$ | $\sum_i w_t^{(i)}\bigl[\mathbf{P}_t^{(i)}[0,0] + (\mathbf{m}_t^{(i)}[0]-\hat\alpha_t)^2\bigr]$ | Law of total variance |
| $\hat{\boldsymbol\psi}_t$ | $\sum_i w_t^{(i)}\,T^{-1}(\boldsymbol\phi^{(i)})$ | Online parameter estimate |

The online parameter estimate $\hat{\boldsymbol\psi}_t$ converges to a neighbourhood
of the true $\boldsymbol\psi$ as $T \to \infty$ (for fixed $\boldsymbol\psi$) or
tracks a slowly drifting $\boldsymbol\psi_t$ in the non-stationary regime. Monitoring
$\mathrm{Var}(\psi_k \mid y_{1:t})$ over time provides a real-time alarm for
parameter instability.

---

## Part VII — Implementation Roadmap

### Phase 1 — Particle Filter (Weeks 1–2)

1. Implement single-asset RBPF in JAX: `vmap` Kalman update over $N$ CIR particles.
2. Validate on synthetic data from the generative model with known $\boldsymbol\psi$.
3. Calibrate $\boldsymbol\psi$ via CMA-ES on historical SPX/stock minute data.
4. Extend to $M$ assets independently (trivially parallelisable).

### Phase 2 — Classical Portfolio Layer (Week 3)

5. Implement Woodbury-efficient covariance (21) and MVO solver (23).
6. Add turnover and beta-neutral constraints via QP (e.g. `jaxopt`).
7. Backtest MVO strategy as the baseline.

### Phase 3 — PPO Strategy Layer (Weeks 4–6)

8. Build MDP environment wrapping the live particle filter.
9. Implement PPO with Gaussian actor and value baseline in JAX/Flax.
10. Train on synthetic rollouts first (curriculum), then fine-tune on historical data.
11. Implement MVO-residual variant (28) for the production hybrid.

### Phase 4 — Evaluation (Week 7)

12. Walk-forward backtest: Sharpe, Calmar, max drawdown, turnover.
13. Compare: EKF+MVO, RBPF+MVO, RBPF+PPO, RBPF+Hybrid.
14. Stress-test on high-volatility regimes (2020 COVID, 2022 rate shock).

---

## Appendix — Notation Summary

| Symbol | Meaning |
|--------|---------|
| $r_t^X$ | Log-return of stock $X$ at step $t$ |
| $r_t^{\mathrm{SPX}}$ | Observed SPX log-return (known input) |
| $\alpha_t$ | Idiosyncratic excess return (Jensen's alpha) |
| $\beta_t$ | Market factor loading |
| $V_t$ | Instantaneous idiosyncratic variance |
| $\phi_a, \phi_b$ | AR(1) persistence coefficients |
| $\sigma_a, \sigma_b$ | AR(1) innovation standard deviations |
| $\kappa, \theta, \sigma_V$ | CIR speed, long-run mean, vol-of-vol |
| $N$ | Number of particles |
| $\delta$ | Variance floor |
| $\mathbf{w}_t$ | Portfolio weight vector |
| $\boldsymbol\Sigma_t$ | Reconstructed return covariance matrix |
| $\gamma$ | Risk aversion coefficient |
| $\rho_V$ | Leverage correlation: $\mathrm{corr}(\varepsilon_t, \xi_t^V)$ |
| $\boldsymbol\phi$ | Unconstrained parameter vector $T(\boldsymbol\psi)$ |
| $h, a$ | Liu-West jitter scale and shrinkage ($a = \sqrt{1-h^2}$) |
| $\tilde y_t^{(i)}$ | Leverage-corrected observation per particle |
| $\tilde R_t^{(i)}$ | Reduced observation noise after leverage correction |
