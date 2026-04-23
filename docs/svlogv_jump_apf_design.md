# Stochastic Volatility Model with Log-Variance OU Process and Correlated Gaussian Jumps

## Design Document for Auxiliary Particle Filter Implementation

---

## 1. Model Specification

### 1.1 State Variables

| Symbol | Description |
|--------|-------------|
| $S_t$ | Asset price at time $t$ |
| $y_t = \log(S_t / S_{t-1})$ | Log-return at step $t$ |
| $\ell_t = \log(V_t)$ | Latent log-variance (OU process) |
| $V_t = e^{\ell_t}$ | Instantaneous variance at time $t$ |
| $I_t \in \{0, 1\}$ | Jump indicator, shared by both equations |

---

### 1.2 Discretised SDEs

The model is discretised under the Euler–Maruyama scheme at a uniform resolution of $dt = 1\text{ min}$ for **diffusive** terms. Jump probability, however, scales with the actual inter-observation interval $dt_t$ (from `dt_seq`) to correctly absorb overnight and weekend gaps.

#### Log-return

$$y_t = \underbrace{\left(r - \tfrac{1}{2}V_t\right)dt}_{\text{risk-neutral drift}} - \underbrace{c_t}_{\text{jump compensator}} + \underbrace{\sqrt{V_t \cdot dt}\,\epsilon_t}_{\text{diffusion}} + \underbrace{I_t J_t^S}_{\text{jump}}$$

where $\epsilon_t \sim \mathcal{N}(0, 1)$.

> **Key design choice:** $y_t$ depends on $V_t$ (the *current* variance), not $V_{t-1}$. This reflects the contemporaneous nature of the within-period variance.

#### Log-variance (OU process)

$$\ell_t = \ell_{t-1} + \kappa(\theta - \ell_{t-1})dt + \sigma_v\sqrt{dt}\,\eta_t + I_t J_t^V$$

where $\eta_t \sim \mathcal{N}(0,1)$.

#### Correlated diffusion innovations

$$\begin{pmatrix} \epsilon_t \\ \eta_t \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix} \begin{pmatrix} z_{1,t} \\ z_{2,t} \end{pmatrix}, \quad z_{1,t}, z_{2,t} \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$$

This gives $\text{Cov}(\epsilon_t, \eta_t) = \rho$.

#### Bivariate Gaussian jump sizes

$$\begin{pmatrix} J_t^S \\ J_t^V \end{pmatrix} \sim \mathcal{N}\!\left( \begin{pmatrix} \mu_{JS} \\ \mu_{JV} \end{pmatrix},\; \begin{pmatrix} \sigma_{JS}^2 & \rho_J \sigma_{JS}\sigma_{JV} \\ \rho_J \sigma_{JS}\sigma_{JV} & \sigma_{JV}^2 \end{pmatrix} \right)$$

Equivalently via Cholesky:

$$J_t^S = \mu_{JS} + \sigma_{JS}\, z_{j1,t}$$

$$J_t^V = \mu_{JV} + \sigma_{JV}\!\left(\rho_J\, z_{j1,t} + \sqrt{1-\rho_J^2}\, z_{j2,t}\right)$$

with $z_{j1,t}, z_{j2,t} \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$ independent of $(z_{1,t}, z_{2,t})$.

---

### 1.3 Jump Arrival Probability

The jump indicator $I_t$ is shared between both equations:

$$I_t \sim \text{Bernoulli}(p_t), \quad p_t = \lambda_J \cdot dt_t$$

where $dt_t$ is the actual calendar interval (from `dt_seq`). Proportionality to $dt_t$ ensures that overnight and weekend gaps accumulate the correct expected number of jumps.

> **Note:** A simple linear approximation $p_t = \lambda_J \cdot dt_t$ is used. For intraday steps $dt_t = dt_{\min}$, this is accurate as long as $\lambda_J \cdot dt_{\min} \ll 1$ (e.g. $\lambda_J = 100$ jumps/year, $dt_{\min} \approx 1/98\,280$ year gives $p \approx 0.001$). For overnight/weekend intervals the approximation may slightly exceed 1 for very large $\lambda_J$; users should bound $\lambda_J$ accordingly or switch to the exact Poisson formula if needed.

---

### 1.4 Risk-Neutral Compensator

Under the risk-neutral measure $\mathbb{Q}$, we require:

$$\mathbb{E}_{\mathbb{Q}}\!\left[e^{y_t} \,\Big|\, \mathcal{F}_{t-1}\right] = e^{r \cdot dt}$$

**Derivation:**

Taking the expectation of $e^{y_t}$ using the discretised log-return:

$$e^{y_t} = \exp\!\left[(r - \tfrac{1}{2}V_t)dt - c_t + \sqrt{V_t\,dt}\,\epsilon_t + I_t J_t^S\right]$$

Since $\epsilon_t$ and $J_t^S$ are independent of each other and of $\mathcal{F}_{t-1}$ (conditional on $V_t$), and noting $\mathbb{E}[e^{\sqrt{V_t\,dt}\,\epsilon_t}] = e^{V_t\,dt/2}$ (log-normal MGF):

$$\mathbb{E}_{\mathbb{Q}}\!\left[e^{y_t}\right] = e^{(r - \frac{1}{2}V_t)dt - c_t + \frac{1}{2}V_t\,dt} \cdot \mathbb{E}\!\left[e^{I_t J_t^S}\right]$$

$$= e^{r\,dt - c_t} \cdot \mathbb{E}\!\left[e^{I_t J_t^S}\right]$$

Now compute $\mathbb{E}[e^{I_t J_t^S}]$ using the law of total expectation:

$$\mathbb{E}\!\left[e^{I_t J_t^S}\right] = (1-p_t)\cdot 1 + p_t \cdot \mathbb{E}\!\left[e^{J_t^S}\right]$$

The moment generating function of a Gaussian jump $J_t^S \sim \mathcal{N}(\mu_{JS}, \sigma_{JS}^2)$ at argument 1:

$$m_r \triangleq \mathbb{E}\!\left[e^{J_t^S}\right] = e^{\mu_{JS} + \frac{1}{2}\sigma_{JS}^2}$$

Therefore:

$$\mathbb{E}\!\left[e^{I_t J_t^S}\right] = (1-p_t) + p_t\, m_r$$

Setting $\mathbb{E}_{\mathbb{Q}}[e^{y_t}] = e^{r\,dt}$ requires:

$$e^{r\,dt - c_t}\left[(1-p_t) + p_t\, m_r\right] = e^{r\,dt}$$

$$\Rightarrow \quad c_t = \log\!\left[(1-p_t) + p_t\, m_r\right]$$

Since $m_r > 0$ and $p_t \in (0,1)$, the argument is always positive.

> **Summary:**
> $$\boxed{c_t = \log\!\left[(1 - \lambda_J dt_t) + \lambda_J dt_t \cdot e^{\mu_{JS} + \frac{1}{2}\sigma_{JS}^2}\right]}$$

---

## 2. Parameter Table

| Index | Name | Description | Transform |
|-------|------|-------------|-----------|
| 0 | $\ell_0$ = `lnv0` | Initial log-variance | `sigmoid_ab(-10, 0)` |
| 1 | $\kappa$ | OU mean-reversion speed (yr$^{-1}$) | `softplus(0.01, 30)` |
| 2 | $\theta$ | OU long-run log-variance level | `sigmoid_ab(-10, 0)` |
| 3 | $\sigma_v$ | OU vol-of-log-variance (yr$^{-1/2}$) | `softplus(0.01, 3)` |
| 4 | $\rho$ | Diffusion correlation | `tanh(-0.99, 0.99)` |
| 5 | $r$ | Risk-free rate (yr$^{-1}$) | `sigmoid_ab(-1e-4, 1e-4)` |
| 6 | $\lambda_J$ | Jump intensity (jumps yr$^{-1}$) | `softplus(0.1, 200)` |
| 7 | $\mu_{JS}$ | Mean log-return jump size | `sigmoid_ab(-0.2, 0.2)` |
| 8 | $\sigma_{JS}$ | Std of log-return jump | `softplus(1e-3, 0.3)` |
| 9 | $\mu_{JV}$ | Mean log-variance jump size | `sigmoid_ab(-0.3, 0.3)` |
| 10 | $\sigma_{JV}$ | Std of log-variance jump | `softplus(1e-3, 0.1)` |
| 11 | $\rho_J$ | Jump size correlation | `tanh(-0.99, 0.99)` |

---

## 3. Auxiliary Particle Filter (APF)

### 3.1 Notation

- $N$: number of particles per parameter candidate.
- $P$: population size (number of ES candidates evaluated in parallel).
- $\ell_{t-1}^{(i)}$: particle $i$'s log-variance **before** the update at time $t$.
- $\hat{V}_{t|t-1}^{(i)}$: predictive mean of $V_t$ given $\ell_{t-1}^{(i)}$, used in pilot stage.

---

### 3.2 Pilot Stage — Computing $g_t^{(i)}$

The APF introduces an auxiliary variable to resample particles **before** propagation, using an approximation to the one-step predictive likelihood.

We define the pilot weight (un-normalised):

$$g_t^{(i)} \propto p\!\left(y_t \,\Big|\, \hat{V}_t^{(i)}\right)$$

where $\hat{V}_t^{(i)}$ is the **mean** of $V_t$ conditioned on $\ell_{t-1}^{(i)}$:

$$\hat{V}_t^{(i)} \triangleq \mathbb{E}\!\left[V_t \,\Big|\, \ell_{t-1}^{(i)}\right] \approx e^{\hat{\ell}_t^{(i)}}$$

with $\hat{\ell}_t^{(i)} = \ell_{t-1}^{(i)} + \kappa(\theta - \ell_{t-1}^{(i)})dt$.

> We approximate $\mathbb{E}[e^{\ell_t}] \approx e^{\mathbb{E}[\ell_t]}$ (Jensen approximation), so $\hat{V}_t^{(i)} = e^{\hat{\ell}_t^{(i)}}$.

#### Rao-Blackwellisation of the pilot likelihood

Because $I_t$ is unobserved, we marginalise it analytically:

$$g_t^{(i)} = p\!\left(y_t \,\Big|\, \hat{V}_t^{(i)}\right) = (1-p_t)\,f_0^{(i)} + p_t\,f_1^{(i)}$$

where:

$$f_0^{(i)} = \mathcal{N}\!\left(y_t;\; \mu_y^{(i)},\; \hat{V}_t^{(i)} dt\right)$$

$$f_1^{(i)} = \mathcal{N}\!\left(y_t;\; \mu_y^{(i)} + \mu_{JS},\; \hat{V}_t^{(i)} dt + \sigma_{JS}^2\right)$$

$$\mu_y^{(i)} = \left(r - \tfrac{1}{2}\hat{V}_t^{(i)}\right)dt - c_t$$

This is a **2-component Gaussian mixture** over the jump indicator. The Rao-Blackwellised likelihood is exact given $\hat{V}_t^{(i)}$.

#### Log-likelihood increment

The marginal likelihood contribution at time $t$ is:

$$\hat{p}(y_t \mid y_{1:t-1}) = \frac{1}{N}\sum_{i=1}^N g_t^{(i)}$$

In log-domain:

$$\log \hat{p}(y_t \mid y_{1:t-1}) = \log\!\sum_{i=1}^N g_t^{(i)} - \log N$$

The total log-likelihood is:

$$\log \mathcal{L} = \sum_{t=1}^T \left(\log\!\sum_{i=1}^N g_t^{(i)} - \log N\right)$$

---

### 3.3 Ancestor Resampling (Pilot)

Normalise the pilot weights:

$$\tilde{w}_t^{(i)} = \frac{g_t^{(i)}}{\sum_j g_t^{(j)}}$$

Draw ancestor indices $a_1, \ldots, a_N$ by **systematic resampling** with a single pre-drawn uniform $u_{t} \sim \text{Uniform}[0,1)$:

For $j = 1, \ldots, N$:

$$a_j = \min\left\{k : \sum_{i=1}^k \tilde{w}_t^{(i)} \geq \frac{u_t + j - 1}{N}\right\}$$

The resampled particles are $\ell_{t-1}^{*(j)} = \ell_{t-1}^{(a_j)}$.

---

### 3.4 Propagation Stage

For each resampled particle $j$, propagate using the **true transition kernel** $p(\ell_t \mid y_t, \ell_{t-1}^{*(j)})$. We do **not** use Kalman updates here; instead we sample explicitly.

**Step 1: Sample jump indicator**

$$I_t^{(j)} \sim \text{Bernoulli}\!\left(\pi_t^{(j)}\right)$$

where $\pi_t^{(j)}$ is the posterior jump probability given the observation. Using Bayes:

$$\pi_t^{(j)} = \frac{p_t\, f_1^{*(j)}}{(1-p_t)\, f_0^{*(j)} + p_t\, f_1^{*(j)}}$$

with $f_0^{*(j)}, f_1^{*(j)}$ evaluated at the resampled particle $\ell_{t-1}^{*(j)}$ (carrying same predictive mean $\hat{V}$ used in pilot). In log-odds form (numerically stable):

$$\log \frac{\pi_t^{(j)}}{1 - \pi_t^{(j)}} = (\log p_t - \log(1-p_t)) + \log f_1^{*(j)} - \log f_0^{*(j)}$$

In practice we sample:

$$b^{(j)} = \mathbf{1}\!\left[u_{\text{mix}}^{(j)} < \pi_t^{(j)}\right], \quad u_{\text{mix}}^{(j)} \sim \text{Uniform}[0,1)$$

**Step 2: Sample jump sizes (conditional on $I_t^{(j)}$)**

If $b^{(j)} = 1$:

$$z_{j1}^{(j)} \sim \mathcal{N}(0,1)$$

$$J_t^{S,(j)} = \mu_{JS} + \sigma_{JS}\, z_{j1}^{(j)}$$

$$J_t^{V,(j)} = \mu_{JV} + \sigma_{JV}\!\left(\rho_J\, z_{j1}^{(j)} + \sqrt{1-\rho_J^2}\, z_{j2}^{(j)}\right), \quad z_{j2}^{(j)} \sim \mathcal{N}(0,1)$$

Otherwise $J_t^{S,(j)} = J_t^{V,(j)} = 0$.

**Step 3: Propagate log-variance**

$$\ell_t^{(j)} = \ell_{t-1}^{*(j)} + \kappa(\theta - \ell_{t-1}^{*(j)})dt + \sigma_v\sqrt{dt}\,z_{\eta}^{(j)} + b^{(j)} J_t^{V,(j)}$$

where $z_{\eta}^{(j)} \sim \mathcal{N}(0,1)$.

> The diffusion noise $z_{\eta}^{(j)}$ is the $z_2$ column of the CRN noise array, tiled across populations.

---

### 3.5 Correction Weights

In the standard APF, the correction importance weight is:

$$w_t^{(j)} \propto \frac{p(y_t \mid \ell_t^{(j)})\, p(\ell_t^{(j)} \mid \ell_{t-1}^{*(j)})}{q(\ell_t^{(j)} \mid y_t, \ell_{t-1}^{*(j)}) \cdot g_t^{(a_j)}}$$

Since the propagation kernel is the exact prior transition $q = p(\ell_t \mid \ell_{t-1})$ and the observation model is $p(y_t \mid \ell_t)$:

$$w_t^{(j)} \propto \frac{p(y_t \mid \ell_t^{(j)})}{g_t^{(a_j)}}$$

These correction weights are **not** identically 1 (unlike a fully-adapted filter). A second systematic resampling step is required when $\text{ESS} = \left(\sum_j (w_t^{(j)})^2\right)^{-1} / N$ falls below a threshold.

> In this implementation we perform the correction-weight resampling at **every** step for simplicity, or the weights can be accumulated and resampled adaptively. The log-likelihood accumulator already uses the pilot $g_t$ terms; the correction weights ensure the particle cloud remains calibrated.

---

### 3.6 Full APF Algorithm

```
Initialise:  ell^(i)_0 ~ p(ell_0) for i = 1,...,N

For t = 1, ..., T:

  // ---- PILOT STAGE ----
  1. Compute hat_V^(i) = exp(ell^(i)_{t-1} + kappa*(theta - ell^(i)_{t-1})*dt)
  2. Compute p_t = lambda_J * dt_t
  3. Compute c_t = log[(1 - p_t) + p_t * m_r]
  4. Compute mu_y^(i) = (r - 0.5*hat_V^(i))*dt - c_t
  5. Compute f0^(i) = N(y_t; mu_y^(i), hat_V^(i)*dt)
     Compute f1^(i) = N(y_t; mu_y^(i) + mu_JS, hat_V^(i)*dt + sigma_JS^2)
  6. Compute g^(i) = (1-p_t)*f0^(i) + p_t*f1^(i)
  7. Log-likelihood += log(mean_i g^(i))

  // ---- ANCESTOR RESAMPLING ----
  8. Resample a_1,...,a_N ~ Multinomial(tilde_w) via systematic resampling
     ell^*(j) = ell^{(a_j)}_{t-1}

  // ---- PROPAGATION STAGE ----
  9.  Compute pi^(j) = p_t*f1^*(j) / [(1-p_t)*f0^*(j) + p_t*f1^*(j)]
  10. Sample b^(j) = 1[u_mix^(j) < pi^(j)]   (jump indicator)
  11. If b^(j)=1: sample z_j1, z_j2 ~ N(0,1)
      J^{S,(j)} = mu_JS + sigma_JS * z_j1
      J^{V,(j)} = mu_JV + sigma_JV*(rho_J*z_j1 + sqrt(1-rho_J^2)*z_j2)
  12. Sample z_eta^(j) ~ N(0,1)
      ell^(j)_t = ell^*(j) + kappa*(theta - ell^*(j))*dt
                 + sigma_v*sqrt(dt)*z_eta^(j)
                 + b^(j)*J^{V,(j)}

  // ---- CORRECTION WEIGHTS (optional second resample) ----
  13. Compute w^(j) = p(y_t | ell^(j)_t) / g^(a_j)
      Normalise: tilde_w^(j) = w^(j) / sum_k w^(k)
      Optionally resample if ESS < N/2
```

---

## 4. CRN Noise Layout

Pre-generate all random draws for Common Random Numbers (shape `(T, 4N + 1)`):

| Columns | Distribution | Use |
|---------|-------------|-----|
| `0 .. N-1` | $\mathcal{N}(0,1)$ | $z_1$ — diffusion (log-return direction) |
| `N .. 2N-1` | $\mathcal{N}(0,1)$ | $z_\eta$ — OU diffusion noise |
| `2N .. 3N-1` | $\mathcal{N}(0,1)$ | $z_{j1}$ — jump factor 1 |
| `3N .. 4N-1` | $\mathcal{N}(0,1)$ | $z_{j2}$ — jump factor 2 |
| `4N` | $\text{Uniform}[0,1)$ | $u_{\text{res}}$ — systematic resampling |
| `4N+1 .. 5N` | $\text{Uniform}[0,1)$ | $u_{\text{mix}}$ — per-particle Bernoulli |

> **Total width:** $5N + 1$ columns per time step.

---

## 5. Predictive Distribution

Given the propagated particles $\{\ell_t^{(j)}\}$ with (approximately) equal weights, the one-step-ahead predictive moments for $y_{t+1}$ are:

$$\bar{y}_{t+1} = \frac{1}{N}\sum_j \left[\left(r - \tfrac{1}{2}e^{\ell_t^{(j)}}\right)dt - c_{t+1} + p_{t+1}\,\mu_{JS}\right]$$

$$\text{Var}(y_{t+1}) = \frac{1}{N}\sum_j \left[e^{\ell_t^{(j)}}dt + p_{t+1}\sigma_{JS}^2 + p_{t+1}(1-p_{t+1})\mu_{JS}^2\right] + \text{Var}_j\!\left[\bar{y}_{t+1}^{(j)}\right]$$

---

## 6. Implementation Notes

- **`dt` vs `dt_t`:** Diffusive terms always use `dt = _DT_MIN` (1 min). Jump arrival probability uses `dt_t` from `dt_seq` which captures overnight/weekend gaps.
- **Parameter bounds:** $\lambda_J \leq 200$ ensures $p_t = \lambda_J \cdot dt_{\min} \approx 0.002 \ll 1$ for intraday steps.
- **Variance floor:** $V_t \geq \varepsilon = 10^{-8}$ prevents division by zero and log underflow.
- **Log-variance clipping:** $\ell_t \in [-20, 5]$ contains variance in $[2 \times 10^{-9}, 148]$.
- **CRN / CPM:** The `calibrate` loop in `StochasticProcessBase` updates the noise array via AR(1) CPM before each generation, preserving gradient signal across evolution steps.
