# Stochastic Volatility Model with CIR Variance Process and Correlated Gaussian Jumps

## Design Document for Auxiliary Particle Filter Implementation

---

## 1. Model Specification

### 1.1 State Variables

| Symbol | Description |
|--------|-------------|
| $S_t$ | Asset price at time $t$ |
| $y_t = \log(S_t / S_{t-1})$ | Log-return at step $t$ |
| $V_t$ | Instantaneous variance (CIR latent state) |
| $I_t \in \{0, 1\}$ | Jump indicator, shared by both equations |

---

### 1.2 Discretised SDEs

The model is discretised under the Euler–Maruyama scheme at a uniform resolution of $dt = 1\text{ min}$ for **diffusive** terms. Jump probability scales with the actual inter-observation interval $dt_t$ (from `dt_seq`) to correctly absorb overnight and weekend gaps.

#### Log-return (active formulation)

$$y_t = r\,dt + \sqrt{V_t\,dt}\,z_{1,t} + I_t\,J_t^S$$

> **Note on risk-neutral drift:** The full risk-neutral drift $(r - \tfrac{1}{2}V_t)dt - c_t$ and the jump compensator $c_t$ are **currently disabled** in the implementation. Only $r\,dt$ is active. The complete derivation of $c_t$ is preserved in Section 1.4 for future re-enabling.

> **Key design choice:** $y_t$ uses the *current* variance $V_t$ (not $V_{t-1}$), reflecting the contemporaneous nature of within-period variance.

#### CIR variance process

$$V_t = V_{t-1} + \kappa(\theta - V_{t-1})dt + \sigma_v\sqrt{V_{t-1}\,dt}\,\eta_t + I_t\,J_t^V$$

$$V_t \leftarrow \text{clip}(V_t,\;\varepsilon,\;5.0)$$

where $\varepsilon = 10^{-8}$ is the variance floor and 5.0 is the ceiling (annual vol $\leq 224\%$).

> **CIR vs OU:** The diffusion term scales with $\sqrt{V_{t-1}}$, not a constant. This dampens noise as variance approaches zero (Feller condition), preventing the variance from going negative without requiring a log-transformation.

#### Correlated diffusion innovations

$$\eta_t = \rho\,z_{1,t} + \sqrt{1-\rho^2}\,z_{2,t}, \qquad z_{1,t},z_{2,t} \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$$

This gives $\text{Cov}(z_{1,t}, \eta_t) = \rho$. In the implementation, $z_{\eta,t} \equiv \eta_t$ is drawn directly as the CRN column; $z_{1,t}$ is never drawn explicitly (it is recovered analytically in the correction weight via the conditional $z_1 \mid z_\eta$).

#### Bivariate Gaussian jump sizes

$$\begin{pmatrix} J_t^S \\ J_t^V \end{pmatrix} \sim \mathcal{N}\!\left( \begin{pmatrix} \mu_{JS} \\ \mu_{JV} \end{pmatrix},\; \begin{pmatrix} \sigma_{JS}^2 & \rho_J \sigma_{JS}\sigma_{JV} \\ \rho_J \sigma_{JS}\sigma_{JV} & \sigma_{JV}^2 \end{pmatrix} \right)$$

The **marginal** distribution of $J_t^V$ is:

$$J_t^V \sim \mathcal{N}(\mu_{JV},\,\sigma_{JV}^2)$$

The **conditional** distribution of $J_t^S$ given $J_t^V$ is:

$$J_t^S \mid J_t^V \sim \mathcal{N}\!\left(\mu_{JS|V},\;\sigma_{JS|V}^2\right)$$

$$\mu_{JS|V} = \mu_{JS} + \frac{\sigma_{JS}\,\rho_J}{\sigma_{JV}}\,(J_t^V - \mu_{JV}), \qquad \sigma_{JS|V}^2 = \sigma_{JS}^2\,(1 - \rho_J^2)$$

This factorisation $p(J^S, J^V) = p(J^V)\,p(J^S \mid J^V)$ is the basis for Rao-Blackwellising $J_t^S$ out of the APF (Section 3.2). Only $J_t^V$ is sampled explicitly; the $J_t^S$–$J_t^V$ correlation $\rho_J$ enters analytically through $\mu_{JS|V}$.

---

### 1.3 Jump Arrival Probability

The jump indicator $I_t$ is shared between both equations:

$$I_t \sim \text{Bernoulli}(p_t), \qquad p_t = 1 - e^{-\lambda_J\,dt_t}$$

where $dt_t$ is the actual calendar interval (from `dt_seq`). This is the **exact** first-arrival probability of a Poisson process with rate $\lambda_J$, ensuring correct accumulation over overnight and weekend gaps.

For intraday steps ($dt_t = dt_{\min} \approx 1/98{,}280\,\text{yr}$) with $\lambda_J \leq 200$: $p_t \approx \lambda_J\,dt_{\min} \leq 0.002 \ll 1$.

---

### 1.4 Risk-Neutral Compensator (currently disabled)

Under the risk-neutral measure $\mathbb{Q}$, we require:

$$\mathbb{E}_{\mathbb{Q}}\!\left[e^{y_t} \,\Big|\, \mathcal{F}_{t-1}\right] = e^{r \cdot dt}$$

**Derivation:**

Using $y_t = (r - \tfrac{1}{2}V_t)dt - c_t + \sqrt{V_t\,dt}\,z_1 + I_t\,J_t^S$ and the log-normal MGF $\mathbb{E}[e^{\sqrt{V\,dt}\,z_1}] = e^{V\,dt/2}$:

$$\mathbb{E}_{\mathbb{Q}}\!\left[e^{y_t}\right] = e^{r\,dt - c_t} \cdot \mathbb{E}\!\left[e^{I_t J_t^S}\right]$$

With $m_r \triangleq \mathbb{E}[e^{J_t^S}] = e^{\mu_{JS} + \frac{1}{2}\sigma_{JS}^2}$:

$$\mathbb{E}\!\left[e^{I_t J_t^S}\right] = (1-p_t) + p_t\,m_r$$

Setting this equal to $e^{r\,dt}$ gives:

$$\boxed{c_t = \log\!\left[(1-p_t) + p_t\,e^{\mu_{JS} + \frac{1}{2}\sigma_{JS}^2}\right]}$$

To re-enable: uncomment the `(r - 0.5*hat_V)*dt - comp_v` lines in `_pilot_one`, `_obs_loglik_one`, and `generator`.

---

## 2. Parameter Table

| Index | Name | Description | Transform |
|-------|------|-------------|-----------|
| 0 | $v_0$ | Initial variance | `sigmoid_ab(1e-4, 1.0)` |
| 1 | $\kappa$ | CIR mean-reversion speed (yr$^{-1}$) | `softplus(0.01, 30)` |
| 2 | $\theta$ | CIR long-run variance level | `sigmoid_ab(1e-4, 1.0)` |
| 3 | $\sigma_v$ | CIR vol-of-variance (yr$^{-1/2}$) | `softplus(0.01, 30)` |
| 4 | $\rho$ | Diffusion correlation | `tanh(-0.99, 0.99)` |
| 5 | $r$ | Risk-free rate (yr$^{-1}$) | `sigmoid_ab(-1e-4, 1e-4)` |
| 6 | $\lambda_J$ | Jump intensity (jumps yr$^{-1}$) | `softplus(0.1, 200)` |
| 7 | $\mu_{JS}$ | Mean log-return jump size | `sigmoid_ab(-0.2, 0.2)` |
| 8 | $\sigma_{JS}$ | Std of log-return jump | `softplus(1e-3, 0.3)` |
| 9 | $\mu_{JV}$ | Mean variance jump size | `sigmoid_ab(-0.3, 0.3)` |
| 10 | $\sigma_{JV}$ | Std of variance jump | `softplus(1e-3, 5.0)` |
| 11 | $\rho_J$ | Jump size correlation | `tanh(-0.99, 0.99)` |

---

## 3. Auxiliary Particle Filter (APF)

### 3.1 Notation

- $N$: number of particles per parameter candidate.
- $P$: population size (number of ES candidates evaluated in parallel).
- $V_{t-1}^{(i)}$: particle $i$'s variance **before** the update at time $t$.
- $\hat{V}_t^{(i)}$: one-step predictive mean of $V_t$ given $V_{t-1}^{(i)}$, used in pilot stage.

---

### 3.2 Why Only $J_t^V$ Is Sampled: Rao-Blackwellisation of $J_t^S$

**Core insight:** $J_t^S$ appears **only** in the observation equation for $y_t$. It does **not** enter the CIR state transition for $V_t$. This makes $J_t^S$ a pure observation-level nuisance variable.

In a standard particle filter one would sample $(b^{(j)}, J_t^{V,(j)}, J_t^{S,(j)})$ jointly and evaluate $p(y_t \mid V_t^{(j)}, J_t^{S,(j)})$. By instead integrating $J_t^S$ out **analytically**, the correction weight becomes a deterministic function of the sampled $(V_t^{(j)}, b^{(j)}, J_t^{V,(j)}, z_\eta^{(j)})$:

$$p(y_t \mid V_t^{(j)}, b^{(j)}, J_t^{V,(j)}, z_\eta^{(j)}) = \int p(y_t \mid V_t, b, J_t^S, z_\eta)\,p(J_t^S \mid J_t^{V,(j)}, b^{(j)})\,dJ_t^S$$

**Derivation of the integrated likelihood:**

Conditioning on $z_\eta$ separates the residual variance in $y_t$. From the correlation structure $z_1 = \rho\,z_\eta + \sqrt{1-\rho^2}\,w$ where $w \perp z_\eta$:

$$z_1 \mid z_\eta \sim \mathcal{N}(\rho\,z_\eta,\;1-\rho^2)$$

Therefore:

$$y_t \mid (V_t, b, J_t^S, z_\eta) \sim \mathcal{N}\!\left(\underbrace{r\,dt + \sqrt{V_t\,dt}\,\rho\,z_\eta}_{\text{known given }z_\eta} + b\,J_t^S,\;\; V_t\,dt\,(1-\rho^2)\right)$$

Now substitute $J_t^S \mid (J_t^V, b=1) \sim \mathcal{N}(\mu_{JS|V}, \sigma_{JS|V}^2)$ and integrate (Gaussian convolution):

$$p(y_t \mid V_t, b=1, J_t^V, z_\eta) = \mathcal{N}\!\left(y_t;\; r\,dt + \sqrt{V_t\,dt}\,\rho\,z_\eta + \mu_{JS|V},\; V_t\,dt\,(1-\rho^2) + \sigma_{JS|V}^2\right)$$

For $b=0$: $J_t^S = 0$ so the $\mu_{JS|V}$ and $\sigma_{JS|V}^2$ terms vanish. Both cases unify as:

$$\boxed{\mu_y = r\,dt + \sqrt{V_t\,dt}\,\rho\,z_\eta + b\,\mu_{JS|V}, \qquad \sigma_y^2 = V_t\,dt\,(1-\rho^2) + b\,\sigma_{JS|V}^2}$$

**Benefits of this approach:**
1. **Lower MC variance:** the exact integral replaces a noisy sample of $J_t^S$, reducing correction-weight variance without additional particles.
2. **Smaller CRN array:** no $z_{j1}, z_{j2}$ columns needed; width drops from $5N+1$ to $3N+2$.
3. **Analytic correlation:** $\rho_J$ is handled through $\mu_{JS|V}$, never through a sampled $z_{j1}$ shared between $J^S$ and $J^V$.

---

### 3.3 Pilot Stage — Computing $g_t^{(i)}$

The APF resamples particles **before** propagation using an approximation to the one-step predictive likelihood.

**Predictive variance** (CIR mean step, floored):

$$\hat{V}_t^{(i)} = \max\!\left(V_{t-1}^{(i)} + \kappa(\theta - V_{t-1}^{(i)})dt,\;\varepsilon\right)$$

**Rao-Blackwellised pilot likelihood** (marginalise over $I_t \in \{0,1\}$):

$$g_t^{(i)} = (1-p_t)\,f_0^{(i)} + p_t\,f_1^{(i)}$$

$$f_0^{(i)} = \mathcal{N}\!\left(y_t;\; r\,dt,\; \hat{V}_t^{(i)}\,dt\right)$$

$$f_1^{(i)} = \mathcal{N}\!\left(y_t;\; r\,dt + \mu_{JS},\; \hat{V}_t^{(i)}\,dt + \sigma_{JS}^2\right)$$

> The jump-present density $f_1$ uses the **marginal** variance $\sigma_{JS}^2$ (not the conditional $\sigma_{JS|V}^2$), because $J_t^V$ has not yet been sampled at the pilot stage. Both the jump indicator $I_t$ and the jump size $J_t^S$ are marginalised out — this is a **2-component Gaussian mixture** in the observation $y_t$.

#### Log-likelihood increment

$$\log \hat{p}(y_t \mid y_{1:t-1}) = \log\!\sum_{i=1}^N g_t^{(i)} - \log N$$

$$\log \mathcal{L} = \sum_{t=1}^T \left(\log\!\sum_{i=1}^N g_t^{(i)} - \log N\right)$$

---

### 3.4 Ancestor Resampling (Pilot)

Normalise the pilot weights:

$$\tilde{w}_t^{(i)} = \frac{g_t^{(i)}}{\sum_j g_t^{(j)}}$$

Draw ancestor indices via **systematic resampling** with a single pre-drawn $u_{\text{res}} \sim \text{Uniform}[0,1)$:

$$a_j = \min\!\left\{k : \sum_{i=1}^k \tilde{w}_t^{(i)} \geq \frac{u_{\text{res}} + j - 1}{N}\right\}, \quad j = 1,\ldots,N$$

Resampled particles: $V_{t-1}^{*(j)} = V_{t-1}^{(a_j)}$. The pilot densities $f_0^{*(j)}, f_1^{*(j)}, g_t^{*(j)}$ are gathered at the same ancestor indices for use in the propagation and correction stages.

---

### 3.5 Propagation Stage

For each resampled particle $j$:

**Step 1: Posterior jump probability**

Using the pilot densities at the ancestor, apply Bayes (numerically stable log-odds form):

$$\log \frac{\pi_t^{(j)}}{1-\pi_t^{(j)}} = \log\frac{p_t}{1-p_t} + \log f_1^{*(j)} - \log f_0^{*(j)}$$

$$b^{(j)} = \mathbf{1}\!\left[u_{\text{mix}}^{(j)} < \pi_t^{(j)}\right]$$

This uses the **same** $f_0^*$ and $f_1^*$ computed in the pilot, so no additional likelihood evaluations are needed.

**Step 2: Sample $J_t^{V,(j)}$ from the marginal prior**

$$J_t^{V,(j)} = \mu_{JV} + \sigma_{JV}\,z_{jv}^{(j)}, \qquad z_{jv}^{(j)} \sim \mathcal{N}(0,1)$$

$J_t^S$ is **not sampled here** — it will be marginalised out analytically in Step 3 of the correction weight.

**Step 3: Propagate CIR variance**

$$V_t^{(j)} = \text{clip}\!\left(V_{t-1}^{*(j)} + \kappa(\theta - V_{t-1}^{*(j)})dt + \sigma_v\sqrt{V_{t-1}^{*(j)}\,dt}\,z_\eta^{(j)} + b^{(j)}\,J_t^{V,(j)},\;\varepsilon,\;5.0\right)$$

where $z_\eta^{(j)}$ is drawn from the CRN `z_eta` column (tiled across $P$ populations). Note $V_{t-1}^{*(j)}$ is floored to $\varepsilon$ before taking the square root in the diffusion term.

---

### 3.6 Correction Weights (Rao-Blackwellised over $J_t^S$)

Compute the conditional mean and variance that integrates out $J_t^S$:

$$\mu_{JS|V}^{(j)} = \mu_{JS} + \frac{\sigma_{JS}\,\rho_J}{\sigma_{JV}}\!\left(J_t^{V,(j)} - \mu_{JV}\right)$$

$$\sigma_{JS|V}^2 = \sigma_{JS}^2\,(1-\rho_J^2) \quad \text{(constant across particles)}$$

Observation log-likelihood (derived in Section 3.2):

$$\mu_y^{(j)} = r\,dt + \sqrt{V_t^{(j)}\,dt}\,\rho\,z_\eta^{(j)} + b^{(j)}\,\mu_{JS|V}^{(j)}$$

$$\sigma_y^{2,(j)} = V_t^{(j)}\,dt\,(1-\rho^2) + b^{(j)}\,\sigma_{JS|V}^2$$

Correction log-weight:

$$\log w_t^{(j)} = \log \mathcal{N}(y_t;\,\mu_y^{(j)},\,\sigma_y^{2,(j)}) - \log g_t^{*(j)}$$

Second systematic resample using $\log w_t^{(j)}$ and independent $u_{\text{res2}} \sim \text{Uniform}[0,1)$. This corrects for the pilot approximation and ensures the particle cloud remains calibrated.

---

### 3.7 Full APF Algorithm

```
Initialise:  V^(i)_0 = v0,  i = 1,...,N  (tiled across P populations)

For t = 1, ..., T:

  // ---- PILOT STAGE ----
  1. Compute p_t    = 1 - exp(-lambda_J * dt_t)
  2. Compute hat_V^(i) = max(V^(i)_{t-1} + kappa*(theta - V^(i)_{t-1})*dt, eps)
  3. Compute f0^(i) = N(y_t; r*dt,           hat_V^(i)*dt)
     Compute f1^(i) = N(y_t; r*dt + mu_JS,   hat_V^(i)*dt + sigma_JS^2)
  4. Compute log_g^(i) = logaddexp(log(1-p_t) + log f0^(i),  log p_t + log f1^(i))
  5. Log-likelihood += logsumexp(log_g) - log N

  // ---- ANCESTOR RESAMPLING (pilot weights) ----
  6. Resample a_1,...,a_N via systematic(log_g, u_res)
     V^*(j) = V^{(a_j)}_{t-1}
     Gather f0^*(j), f1^*(j), log_g^*(j) = log_g^{(a_j)}

  // ---- PROPAGATION STAGE ----
  7.  logit(pi^(j)) = log(p_t/(1-p_t)) + log f1^*(j) - log f0^*(j)
      b^(j) = 1[ u_mix^(j) < sigmoid(logit(pi^(j))) ]

  8.  J_V^(j) = mu_JV + sigma_JV * z_jv^(j)          // marginal prior; J_S NOT sampled

  9.  V_t^(j) = clip(V^*(j) + kappa*(theta-V^*(j))*dt
                     + sigma_v*sqrt(V^*(j)*dt)*z_eta^(j)
                     + b^(j)*J_V^(j),   eps, 5.0)

  // ---- CORRECTION WEIGHTS (RB over J_S | J_V, b) ----
  10. mu_JS|V^(j)   = mu_JS + (sigma_JS*rho_J/sigma_JV)*(J_V^(j) - mu_JV)
      sigma2_JS|V   = sigma_JS^2 * (1 - rho_J^2)

      mu_y^(j)  = r*dt + sqrt(V_t^(j)*dt)*rho*z_eta^(j) + b^(j)*mu_JS|V^(j)
      sig2^(j)  = V_t^(j)*dt*(1-rho^2)                  + b^(j)*sigma2_JS|V

  11. log_w^(j) = log N(y_t; mu_y^(j), sig2^(j)) - log_g^*(j)
  12. Resample (second pass) via systematic(log_w, u_res2)
```

---

## 4. CRN Noise Layout

Pre-generate all random draws for Common Random Numbers, shape `(T, 3N + 2)`:

| Columns | Distribution | Use |
|---------|-------------|-----|
| `0 .. N-1` | $\mathcal{N}(0,1)$ | $z_\eta$ — CIR diffusion (also encodes leverage via $\rho$ in correction) |
| `N .. 2N-1` | $\mathcal{N}(0,1)$ | $z_{jv}$ — marginal $J_t^V$ sample |
| `2N` | $\text{Uniform}[0,1)$ | $u_{\text{res}}$ — pilot systematic resample |
| `2N+1` | $\text{Uniform}[0,1)$ | $u_{\text{res2}}$ — correction systematic resample |
| `2N+2 .. 3N+1` | $\text{Uniform}[0,1)$ | $u_{\text{mix}}$ — per-particle Bernoulli threshold |

**Total width:** $3N + 2$ columns per time step.

> **Why no $z_{j1}$, $z_{j2}$ columns:** The original design required drawing $z_{j1}$ to generate correlated $(J^S, J^V)$ jointly (Section 1.2 Cholesky form). Since $J^S$ is Rao-Blackwellised out, only the marginal $J^V \sim \mathcal{N}(\mu_{JV}, \sigma_{JV}^2)$ is sampled via $z_{jv}$. The $z_{j1}$ column, which coupled the two jump sizes, is no longer needed — the correlation $\rho_J$ enters analytically through $\mu_{JS|V}$.

---

## 5. Predictive Distribution

Given propagated particles $\{V_t^{(j)}\}$ with approximately equal weights after the second resample:

$$\bar{y}_{t+1} = \frac{1}{N}\sum_j \left[r\,dt + p_{t+1}\,\mu_{JS}\right]$$

$$\text{Var}(y_{t+1}) = \frac{1}{N}\sum_j \left[V_t^{(j)}\,dt + p_{t+1}\,\sigma_{JS}^2 + p_{t+1}(1-p_{t+1})\,\mu_{JS}^2\right] + \text{Var}_j\!\left[\bar{y}_{t+1}^{(j)}\right]$$

---

## 6. Implementation Notes

- **`dt` vs `dt_t`:** Diffusive terms always use `dt = _DT_MIN` (1 min). Jump arrival probability uses `dt_t` from `dt_seq` to capture overnight/weekend gaps.
- **Variance bounds:** $V_t \in [\varepsilon,\,5.0]$ where $\varepsilon = 10^{-8}$. The upper bound 5.0 limits instantaneous variance (annual vol $\leq 224\%$).
- **CIR Feller condition:** $2\kappa\theta > \sigma_v^2$ guarantees strict positivity in the continuous-time limit. This is **not** enforced explicitly in the parameter transforms; the variance floor provides a safety net.
- **Risk-neutral drift (disabled):** The full drift $(r - \tfrac{1}{2}V_t)dt - c_t$ is commented out. Only $r\,dt$ is active. Re-enabling requires uncommenting three locations: `_pilot_one`, `_obs_loglik_one`, and `generator`.
- **CRN / CPM:** The `calibrate` loop in `StochasticProcessBase` updates the noise array via AR(1) CPM before each generation, preserving gradient signal across ES evolution steps.
- **$P$ populations:** Each ES candidate occupies a block of $N$ contiguous particles in the `(P*N,)` flat arrays. Parameters are broadcast with `jnp.repeat(param, N)`, and noises are tiled with `jnp.tile(z, P)`.

---

## 7. Regime Change Estimation: AR Autocorrelation Extension (Plan)

### 7.1 Motivation

The base model assumes log-returns are serially uncorrelated conditional on $(V_t, I_t)$. Empirical intraday data shows persistent autocorrelation structure driven by:

- **Mean-reverting regime** ($R_t = 0$): negative autocorrelation from bid-ask bounce, market-making, short-term over-reaction.
- **Trending regime** ($R_t = 1$): positive autocorrelation from momentum, order-flow clustering, herding.

Adding a hidden Markov regime state $R_t \in \{0,1\}$ with an AR(1) component in the observation equation allows the APF to filter $P(R_t = 1 \mid y_{1:t})$ in real time alongside the variance $V_t$.

---

### 7.2 Extended Model

**Augmented observation equation:**

$$y_t = \phi_{R_t}\,y_{t-1} + r\,dt + \sqrt{V_t\,dt}\,z_{1,t} + I_t\,J_t^S$$

The lagged return $y_{t-1}$ is available at time $t$ from the data, so this adds no additional latent variable — only the **regime** $R_t$ is new.

**Regime dynamics (first-order Markov chain):**

$$P(R_t \mid R_{t-1}) = \mathbf{Q}, \qquad \mathbf{Q} = \begin{pmatrix} 1-q_{01} & q_{01} \\ q_{10} & 1-q_{10} \end{pmatrix}$$

where $q_{01} = P(R_t=1 \mid R_{t-1}=0)$ and $q_{10} = P(R_t=0 \mid R_{t-1}=1)$. Expected regime durations are $\bar{\tau}_0 = 1/q_{01}$ and $\bar{\tau}_1 = 1/q_{10}$.

**State transition:** The CIR dynamics for $V_t$ are unchanged. The regime $R_t$ evolves independently of $V_t$ (given $R_{t-1}$).

**Augmented particle state:** $(V_t^{(i)}, R_t^{(i)}) \in \mathbb{R}_{+} \times \{0,1\}$.

**New parameters (4 additions, total 16):**

| Index | Name | Description | Transform |
|-------|------|-------------|-----------|
| 12 | $\phi_0$ | AR(1) coefficient, mean-reverting regime | `tanh(-0.99, 0.0)` |
| 13 | $\phi_1$ | AR(1) coefficient, trending regime | `sigmoid_ab(0.0, 0.99)` |
| 14 | $q_{01}$ | Transition prob: 0→1 | `sigmoid_ab(1e-6, 1)` |
| 15 | $q_{10}$ | Transition prob: 1→0 | `sigmoid_ab(1e-6, 1)` |

---

### 7.3 APF Extension

#### Pilot stage — Rao-Blackwellise $R_t$ given $R_{t-1}^{(i)}$

Given particle $i$ with state $(V_{t-1}^{(i)}, R_{t-1}^{(i)} = r_{\text{prev}})$, marginalise over the two possible next regimes:

$$g_t^{(i)} = \sum_{r=0}^{1} \mathbf{Q}[r_{\text{prev}}, r]\, g_t^{(i,r)}$$

where each regime-conditional pilot mixture is:

$$g_t^{(i,r)} = (1-p_t)\,\mathcal{N}\!\left(y_t;\; \phi_r\,y_{t-1} + r\,dt,\; \hat{V}_t^{(i)}\,dt\right) + p_t\,\mathcal{N}\!\left(y_t;\; \phi_r\,y_{t-1} + r\,dt + \mu_{JS},\; \hat{V}_t^{(i)}\,dt + \sigma_{JS}^2\right)$$

This is a **4-component Gaussian mixture** (2 regimes × 2 jump states). All components are evaluated analytically — no additional samples needed at this stage.

#### Propagation stage

After ancestor resampling:

**1. Sample regime $R_t^{(j)}$ from posterior:**

$$\tilde{\pi}_{t,r}^{(j)} = \frac{\mathbf{Q}[r_{\text{prev}}^{(j)}, r]\;g_t^{*(j,r)}}{\sum_{r'} \mathbf{Q}[r_{\text{prev}}^{(j)}, r']\;g_t^{*(j,r')}}, \quad r \in \{0,1\}$$

$$R_t^{(j)} \sim \text{Bernoulli}\!\left(\tilde{\pi}_{t,1}^{(j)}\right), \quad \text{using pre-drawn } u_{\text{reg}}^{(j)}$$

**2. Sample jump indicator** (conditioned on the now-known $R_t^{(j)}$):

$$\pi_{\text{jump}}^{(j)} = \frac{p_t\,f_1^{*(j,\,R_t^{(j)})}}{g_t^{*(j,\,R_t^{(j)})}}$$

$$b^{(j)} = \mathbf{1}\!\left[u_{\text{mix}}^{(j)} < \pi_{\text{jump}}^{(j)}\right]$$

**3–4.** Sample $J_t^{V,(j)}$ from marginal prior; propagate $V_t^{(j)}$ via CIR (unchanged).

#### Correction weights

The correction is identical to the base model except $\phi_{R_t^{(j)}}\,y_{t-1}$ shifts the mean:

$$\mu_y^{(j)} = \phi_{R_t^{(j)}}\,y_{t-1} + r\,dt + \sqrt{V_t^{(j)}\,dt}\,\rho\,z_\eta^{(j)} + b^{(j)}\,\mu_{JS|V}^{(j)}$$

$$\sigma_y^{2,(j)} = V_t^{(j)}\,dt\,(1-\rho^2) + b^{(j)}\,\sigma_{JS|V}^2$$

$J_t^S$ remains Rao-Blackwellised exactly as before. The pilot approximation now uses the marginal $\sigma_{JS}^2$ rather than $\sigma_{JS|V}^2$, so the correction weight still accounts for the residual.

---

### 7.4 CRN Noise Update

Add one block of per-particle Bernoulli uniforms for regime sampling:

| Columns | Use |
|---------|-----|
| `0 .. N-1` | $z_\eta$ |
| `N .. 2N-1` | $z_{jv}$ |
| `2N` | $u_{\text{res}}$ |
| `2N+1` | $u_{\text{res2}}$ |
| `2N+2 .. 3N+1` | $u_{\text{mix}}$ (jump Bernoulli) |
| `3N+2 .. 4N+1` | $u_{\text{reg}}$ (regime Bernoulli) |

**Total width:** $4N + 2$ columns per time step.

---

### 7.5 Scan Input Change

The scan currently receives `(log_returns, dt_seq, noises_seq)`. Add `y_prev` as a fourth input (lagged by one step):

```python
y_prev_seq = jnp.concatenate([jnp.array([0.0]), log_returns[:-1]])  # shape (T,)
# scan over (log_returns, dt_seq, noises_seq, y_prev_seq)
```

The initial lag $y_0 = 0$ is a natural neutral prior; alternatively one can initialise it to the sample mean.

---

### 7.6 Regime Diagnostics Output

Extend `FilterInfo` with:

- `regime_prob`: $(T,)$ array of $\hat{P}(R_t=1 \mid y_{1:t}) = \frac{1}{N}\sum_j \mathbf{1}[R_t^{(j)} = 1]$, averaged over $P$ candidates.

This provides a real-time regime probability trace alongside the filtered variance.

---

### 7.7 Implementation Steps (ordered)

1. **Parameters:** append `phi0, phi1, q01, q10` to `PARAM_NAMES` and `PARAM_TRANSFORMS`.
2. **Particle state:** keep two arrays `v_pn (P*N,)` and `r_pn (P*N, int32)` in the scan carry.
3. **Scan inputs:** add `y_prev_seq` to the tuple passed to `jax.lax.scan`.
4. **CRN:** add `u_reg = jax.random.uniform(k6, shape=(T, N))` to `get_noises`; concatenate as `[..., u_reg]`, total $4N+2$.
5. **Pilot (`_pilot_one`):** accept `r_prev` and compute the 4-component mixture $g_t^{(i)} = \sum_r \mathbf{Q}[r_{\text{prev}}, r]\,g_t^{(i,r)}$; return per-regime components for propagation.
6. **Propagation (`_propagate_one`):** accept `u_reg`; compute $\tilde{\pi}_{t,1}^{(j)}$ and sample $R_t^{(j)}$; then sample $b^{(j)}$ conditioned on $R_t^{(j)}$.
7. **Correction (`_obs_loglik_one`):** accept `r_new` and `y_prev`; add $\phi_{r_{\text{new}}} y_{\text{prev}}$ to `mu_y`.
8. **`FilterInfo`:** add `regime_prob` field; populate from particle mean of `r_pn`.
9. **Generator:** add Markov chain simulation for $R_t$ and include $\phi_{R_t}\,y_{t-1}$ in the `y_t` equation.

