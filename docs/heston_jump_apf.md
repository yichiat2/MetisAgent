# Heston–Jump Model: Equations and APF Design

## 1. Model

### 1.1 Continuous-time specification

The Heston–Jump model augments the standard stochastic-volatility
dynamics with a common jump process shared by both the log-price and the
variance.  The variance jump is **multiplicative**: a jump event scales
$V_t$ by a log-normal factor, keeping variance strictly positive and
capturing the empirical asymmetry (large upward spikes, bounded
downward).

$$
dV_t = \kappa(\theta - V_t)\,dt + \sigma\sqrt{V_t}\,dW_t^V + V_{t^-}\bigl(e^{dZ_t^V}-1\bigr)
$$

$$
d\ln S_t = \!\left(r - \tfrac{1}{2}V_t\right)dt
            + \sqrt{V_t}\left(\rho\,dW_t^V + \sqrt{1{-}\rho^2}\,dW_t^S\right)
            + dZ_t^r
$$

where $W_t^V, W_t^S$ are independent Brownian motions, and
$Z_t^V, Z_t^r$ are compound jump processes that share a single arrival
stream (at most one jump per interval after discretisation).

---

### 1.2 Euler–Maruyama discretisation

With a non-uniform time step $\Delta t_t$ (step index $t = 1, \ldots, T$):

**Variance transition**

Define the continuous (diffusion-only) update first:

$$
V_t^\mathrm{cont} = \max\!\Bigl(
  V_{t-1} + \kappa(\theta - V_{t-1})\Delta t_t
  + \sigma\sqrt{V_{t-1}\,\Delta t_t}\;\varepsilon_V^t,\;\delta
\Bigr)
$$

The jump then **multiplies** the floored continuous value:

$$
\boxed{
V_t = V_t^\mathrm{cont} \cdot \exp\!\bigl(I_t\,J_V^t\bigr)
}
\tag{1}
$$

Because $\exp(\cdot) > 0$ always, $V_t \ge \delta$ without any additional
floor on the jump itself.

**Log-return**

$$
\boxed{
r_t = \!\left(r - \tfrac{1}{2}V_t\right)\Delta t_t
      + \sqrt{V_t\,\Delta t_t}
        \!\left(\rho\,\varepsilon_V^t + \sqrt{1{-}\rho^2}\,\varepsilon_S^t\right)
      + I_t\,J_r^t
}
\tag{2}
$$

where $\delta = 10^{-8}$ is the variance floor, and independent noise
draws:

$$
\varepsilon_V^t,\;\varepsilon_S^t \sim \mathcal{N}(0,1)
$$

---

### 1.3 Jump specification

**Shared indicator** (at most one jump per interval):

$$
\boxed{
I_t \sim \mathrm{Bernoulli}(p_t),
\qquad
p_t =
\begin{cases}
  1   & \text{if } \Delta t_t > 1.5\,\Delta t \quad\text{(overnight / weekend gap)} \\
  p_J & \text{otherwise (intraday)}
\end{cases}
}
\tag{3}
$$

**Correlated jump sizes** — parameterised through a shared Gaussian
factor $Z_1^t$ and an independent component $Z_2^t$:

$$
Z_1^t,\;Z_2^t \overset{\text{iid}}{\sim}\mathcal{N}(0,1),
\qquad Z_1^t \perp Z_2^t
$$

$J_V^t$ is the **log** of the variance multiplier (Gaussian):

$$
\boxed{
J_V^t = \mu_J^V + \sigma_J^V\,Z_1^t
}
\tag{4a}
$$

so that $\exp(J_V^t) \sim \mathrm{LogNormal}(\mu_J^V, (\sigma_J^V)^2)$.
Positive $\mu_J^V$ biases jumps upward; the exponential form prevents
the variance from turning negative regardless of the sign of $J_V^t$.

$$
\boxed{
J_r^t = \mu_J^r + \sigma_J^r\!\left(\rho_J\,Z_1^t + \sqrt{1{-}\rho_J^2}\,Z_2^t\right)
}
\tag{4b}
$$

**Marginal distributions and correlation:**

$$
\exp(J_V^t) \sim \mathrm{LogNormal}\!\left(\mu_J^V,\,(\sigma_J^V)^2\right),
\quad E[\exp(J_V^t)] = \exp\!\bigl(\mu_J^V + \tfrac{1}{2}(\sigma_J^V)^2\bigr) \eqqcolon m_V
\tag{5a}
$$

$$
J_r^t \sim \mathcal{N}\!\left(\mu_J^r,\,(\sigma_J^r)^2\right),
\quad
\mathrm{Corr}(J_V^t, J_r^t) = \rho_J
\tag{5b}
$$

$\rho_J$ is the linear correlation between the **log** variance-jump
$J_V^t$ and the log-return jump $J_r^t$; a large positive $\rho_J$ means
big variance jumps tend to coincide with large positive return jumps.

---

### 1.4 Parameter vector

| Index | Symbol | Description | Suggested bounds |
|------:|--------|-------------|-----------------|
| 0 | $v_0$ | initial variance | $(10^{-2},\;1)$ |
| 1 | $\rho$ | price–vol diffusion correlation | $(-0.99,\;0.99)$ |
| 2 | $\kappa$ | mean-reversion speed | $(0.1,\;10)$ |
| 3 | $\theta$ | long-run variance | $(10^{-2},\;1)$ |
| 4 | $\sigma$ | vol of vol | $(0.01,\;1)$ |
| 5 | $r$ | risk-free drift | $(-0.05,\;0.05)$ |
| 6 | $p_J$ | intraday jump probability | $(0,\;0.5)$ |
| 7 | $\mu_J^r$ | mean log-return jump size | $(-0.5,\;0.5)$ |
| 8 | $\sigma_J^r$ | std dev log-return jump | $(10^{-3},\;0.3)$ |
| 9 | $\mu_J^V$ | mean of log-variance multiplier ($m_V = e^{\mu_J^V+\frac{1}{2}(\sigma_J^V)^2}$) | $(-1,\;2)$ |
| 10 | $\sigma_J^V$ | std dev of log-variance multiplier | $(10^{-3},\;1)$ |
| 11 | $\rho_J$ | jump-size correlation | $(-0.99,\;0.99)$ |

---

## 2. Conditional likelihood — key identity for the APF

Given the resampled variance $V_{t-1}$, the sampled diffusion shock
$\varepsilon_V^t$, the jump indicator $I_t$, and the shared jump factor
$Z_1^t$, the independent component $Z_2^t$ can be **marginalised
analytically**.  The resulting conditional log-return distribution is:

$$
r_t \;\Big|\; V_{t-1},\,\varepsilon_V^t,\,I_t,\,Z_1^t
\;\sim\;
\mathcal{N}\!\left(\mu_r,\;\sigma_r^2\right)
\tag{6}
$$

with

$$
\mu_r = \!\left(r - \tfrac{1}{2}V_t\right)\Delta t_t
         + \rho\sqrt{V_t\,\Delta t_t}\;\varepsilon_V^t
         + I_t\!\left(\mu_J^r + \sigma_J^r\rho_J Z_1^t\right)
\tag{7a}
$$

$$
\sigma_r^2 = V_t(1-\rho^2)\Delta t_t
             + I_t\,(\sigma_J^r)^2(1-\rho_J^2)
\tag{7b}
$$

where $V_t = V_t^\mathrm{cont}\cdot\exp(I_t J_V^t)$ is the propagated
variance from eq. (1).  The $\rho_J Z_1^t$ term in the mean captures
the portion of the log-return jump explained by the (log) variance-jump
factor; the residual $\sqrt{1-\rho_J^2}\,Z_2^t$ contributes only to the
variance $\sigma_r^2$.

---

## 3. Auxiliary Particle Filter (APF) with jumps

The state is $V_{t-1}$; the observation is $r_t$.  The filter carries
$N$ particles $\{V_{t-1}^{(i)}, \log w_{t-1}^{(i)}\}_{i=1}^N$.

---

### Step 1 — Jump probability for the current interval

$$
p_t = \mathbf{1}\!\left[\,\Delta t_t > 1.5\,\Delta t\,\right]
    + p_J\;\mathbf{1}\!\left[\,\Delta t_t \le 1.5\,\Delta t\,\right]
\tag{8}
$$

Computed once per step before the particle loop; broadcasts over all
particles.

---

### Step 2 — First-stage (pilot) weights

**Pilot base variance** (drift-only, no noise, no jump):

$$
v_{\mathrm{pilot},0}^{(i)}
= \max\!\bigl(V_{t-1}^{(i)} + \kappa(\theta - V_{t-1}^{(i)})\Delta t_t,\;\delta\bigr)
\tag{9}
$$

**Pilot jump-branch variance** — replace the unknown $\exp(J_V^t)$ by
its mean $m_V = \exp(\mu_J^V + \frac{1}{2}(\sigma_J^V)^2)$:

$$
v_{\mathrm{pilot},1}^{(i)} = v_{\mathrm{pilot},0}^{(i)} \cdot m_V
\tag{9b}
$$

**Pilot observation densities** (marginalised over $\varepsilon_V, Z_1, Z_2$):

$$
\mu_0^{(i)} = \!\left(r - \tfrac{1}{2}v_{\mathrm{pilot},0}^{(i)}\right)\Delta t_t,
\qquad
(\sigma_0^{(i)})^2 = v_{\mathrm{pilot},0}^{(i)}\,\Delta t_t
\tag{10a}
$$

$$
\mu_1^{(i)} = \!\left(r - \tfrac{1}{2}v_{\mathrm{pilot},1}^{(i)}\right)\Delta t_t + \mu_J^r,
\qquad
(\sigma_1^{(i)})^2 = v_{\mathrm{pilot},1}^{(i)}\,\Delta t_t + (\sigma_J^r)^2
\tag{10b}
$$

Using the mean-inflated pilot variance $v_{\mathrm{pilot},1}^{(i)}$ in
both the mean and the diffusion variance of the jump branch ensures the
pilot is calibrated to the expected post-jump return distribution.

**Mixture pilot log-likelihood:**

$$
\log g^{(i)} = \log\!\Bigl[
  (1-p_t)\,\mathcal{N}\!\left(r_t;\mu_0^{(i)},(\sigma_0^{(i)})^2\right)
  + p_t\,\mathcal{N}\!\left(r_t;\mu_1^{(i)},(\sigma_1^{(i)})^2\right)
\Bigr]
\tag{11}
$$

Numerically stable form using `logaddexp`:

$$
\log g^{(i)}
= \mathrm{logaddexp}\!\Bigl(
    \log(1-p_t) + \log \mathcal{N}(r_t;\mu_0^{(i)},(\sigma_0^{(i)})^2),\;
    \log p_t    + \log \mathcal{N}(r_t;\mu_1^{(i)},(\sigma_1^{(i)})^2)
  \Bigr)
\tag{12}
$$

**First-stage importance scores and marginal likelihood contribution:**

$$
\log \xi^{(i)} = \log w_{t-1}^{(i)} + \log g^{(i)}
\tag{13}
$$

$$
\log Z_1 = \mathrm{logsumexp}_i\!\left(\log \xi^{(i)}\right)
\tag{14}
$$

**First-stage resampling** (systematic): draw ancestors
$\{a^{(i)}\}$ from $\{{\xi^{(i)}}/{\sum_j \xi^{(j)}}\}$.

---

### Step 3 — Propagation (second stage)

For each particle $i$, starting from the resampled ancestor $V_{t-1}^{(a^{(i)})}$:

1. **Sample noise:**
$$
\varepsilon_V^{(i)},\;Z_1^{(i)} \sim \mathcal{N}(0,1),
\qquad
u^{(i)} \sim \mathcal{U}(0,1)
$$

2. **Realise jump indicator:**
$$
I_t^{(i)} = \mathbf{1}\!\left[u^{(i)} < p_t\right]
$$

3. **Log-variance jump size:**
$$
J_V^{(i)} = \mu_J^V + \sigma_J^V\,Z_1^{(i)}
$$

4. **Propagate variance (eq. 1) — floor diffusion, then multiply by exponential jump:**
$$
V_t^{\mathrm{cont},(i)} = \max\!\Bigl(
  V_{t-1}^{(a^{(i)})}
  + \kappa\!\left(\theta - V_{t-1}^{(a^{(i)})}\right)\Delta t_t
  + \sigma\sqrt{V_{t-1}^{(a^{(i)})}\,\Delta t_t}\;\varepsilon_V^{(i)},\;\delta
\Bigr)
$$
$$
V_t^{(i)} = V_t^{\mathrm{cont},(i)} \cdot \exp\!\bigl(I_t^{(i)}\,J_V^{(i)}\bigr)
$$

5. **Conditional log-return distribution via eqs. (7a–7b):**
$$
\mu_r^{(i)}
= \!\left(r - \tfrac{1}{2}V_t^{(i)}\right)\Delta t_t
  + \rho\sqrt{V_t^{(i)}\,\Delta t_t}\;\varepsilon_V^{(i)}
  + I_t^{(i)}\!\left(\mu_J^r + \sigma_J^r\rho_J Z_1^{(i)}\right)
$$

$$
(\sigma_r^{(i)})^2
= V_t^{(i)}(1-\rho^2)\Delta t_t
  + I_t^{(i)}(\sigma_J^r)^2(1-\rho_J^2)
$$

6. **True conditional log-likelihood:**
$$
\log p^{(i)} = \log \mathcal{N}\!\left(r_t;\;\mu_r^{(i)},\;(\sigma_r^{(i)})^2\right)
$$

---

### Step 4 — Second-stage importance weights

$$
\log \alpha^{(i)} = \log p^{(i)} - \log g^{(a^{(i)})}
\tag{15}
$$

**Second-stage marginal likelihood contribution:**

$$
\log Z_2 = \mathrm{logsumexp}_i\!\left(\log \alpha^{(i)}\right) - \log N
\tag{16}
$$

**Total log-likelihood increment at step $t$:**

$$
\boxed{\log p(r_t \mid r_{1:t-1}) = \log Z_1 + \log Z_2}
\tag{17}
$$

---

### Step 5 — Posterior resampling and diagnostics

**Resampling:** draw ancestors from $\{\alpha^{(i)}/\sum_j\alpha^{(j)}\}$
(systematic); reset all log-weights to $-\log N$.

**Filtered mean and std (variance posterior):**

$$
\bar{v}_t = \sum_i \tilde\alpha^{(i)}\,V_t^{(i)},
\qquad
\tilde\alpha^{(i)} = \frac{\alpha^{(i)}}{\sum_j \alpha^{(j)}}
$$

$$
\hat\sigma_t = \sqrt{\sum_i \tilde\alpha^{(i)}\!\left(V_t^{(i)}-\bar{v}_t\right)^2}
$$

**ESS:**

$$
\mathrm{ESS}_t = \exp\!\left(-\mathrm{logsumexp}_i(2\tilde{l}^{(i)})\right),
\qquad
\tilde{l}^{(i)} = \log\alpha^{(i)} - \mathrm{logsumexp}_j(\log\alpha^{(j)})
$$

---

### Step 6 — One-step-ahead log-return prediction

Using the resampled particles (approximate draws from
$p(V_t \mid r_{1:t})$), predict $r_{t+1}$ via the pilot approximation.
Let $m_V = \exp(\mu_J^V + \frac{1}{2}(\sigma_J^V)^2)$ and define the
two-branch pilot variances:

$$
v_{\mathrm{pred},0}^{(i)}
= \max\!\bigl(
    V_t^{(i)} + \kappa(\theta - V_t^{(i)})\Delta t_t,\;\delta
  \bigr),
\qquad
v_{\mathrm{pred},1}^{(i)} = v_{\mathrm{pred},0}^{(i)} \cdot m_V
\tag{18}
$$

**Per-particle branch means:**
$$
\mu_{\mathrm{pred},0}^{(i)} = \!\left(r - \tfrac{1}{2}v_{\mathrm{pred},0}^{(i)}\right)\Delta t_t,
\qquad
\mu_{\mathrm{pred},1}^{(i)} = \!\left(r - \tfrac{1}{2}v_{\mathrm{pred},1}^{(i)}\right)\Delta t_t + \mu_J^r
\tag{19}
$$

**Per-particle mixture mean (law of total expectation):**
$$
\bar{r}_{\mathrm{pred}}^{(i)}
= (1-p_t)\,\mu_{\mathrm{pred},0}^{(i)} + p_t\,\mu_{\mathrm{pred},1}^{(i)}
\tag{20}
$$

**Per-particle mixture variance (law of total variance):**
$$
\overline{v_{\mathrm{pred}}^r}^{\,(i)}
= (1-p_t)\,v_{\mathrm{pred},0}^{(i)}\Delta t_t
  + p_t\!\left[v_{\mathrm{pred},1}^{(i)}\Delta t_t + (\sigma_J^r)^2\right]
  + p_t(1-p_t)\!\left[\mu_{\mathrm{pred},1}^{(i)}-\mu_{\mathrm{pred},0}^{(i)}\right]^2
\tag{21}
$$

The three terms in eq. (21) are: no-jump within-regime variance, jump
within-regime variance (diffusion inflated by $m_V$ plus return-jump
variance), and the between-regime variance from the mean shift.

Aggregate across particles (law of total variance):

$$
\widehat{\mu}_{r,t+1}
= \frac{1}{N}\sum_i \bar{r}_{\mathrm{pred}}^{(i)}
$$

$$
\widehat{\sigma}_{r,t+1}
= \sqrt{
    \frac{1}{N}\sum_i \overline{v_{\mathrm{pred}}^r}^{\,(i)}
    + \frac{1}{N}\sum_i \!\left(\bar{r}_{\mathrm{pred}}^{(i)} - \widehat{\mu}_{r,t+1}\right)^2
  }
$$

---

## 4. Implementation notes

### 4.1 Overnight / weekend gate

The condition `dt_i > 1.5 * base_dt` forces $p_t = 1$ for any step
whose calendar gap exceeds 1.5× the nominal intraday step.  In JAX:

```python
p_t = jnp.where(dt_i > 1.5 * base_dt, 1.0, p_J)
```

The pilot mixture (eq. 12) degenerates gracefully: when $p_t = 1$ the
no-jump term vanishes ($\log(1-p_t) \to -\infty$), so only the jump
density contributes to $\log g^{(i)}$.

### 4.2 Sampling the jump indicator inside `jax.lax.scan`

Binary sampling is done via a uniform threshold:
```python
key, jump_key, eps_key, z1_key = jax.random.split(key, 4)
u   = jax.random.uniform(jump_key, shape=(N,))
I_t = (u < p_t).astype(jnp.float32)          # (N,)  0.0 or 1.0
eps_v = jax.random.normal(eps_key, shape=(N,))
Z1    = jax.random.normal(z1_key,  shape=(N,))
```

### 4.3 Variance-floor interaction with jump

The variance floor is applied to the **diffusion part only**; the
exponential jump is then applied on top:
```python
J_V    = mu_JV + sigma_JV * Z1            # log-multiplier, shape (N,)
v_cont = jnp.maximum(drift + diffusion, VARIANCE_FLOOR)
v_next = v_cont * jnp.exp(I_t * J_V)     # always > 0
```
Because $\exp(\cdot) > 0$, $v_\mathrm{next} \ge \delta$ automatically;
no additional floor on the jump term is needed.  Large negative $J_V$
(rare outlier) shrinks $V_t$ toward $\delta \cdot e^{J_V}$, which stays
positive.

### 4.4 Numerical stability

- The mixture pilot (eq. 12) is evaluated entirely in log-space to avoid
  underflow when one mixture component has negligible weight.
- The correction $\log\alpha^{(i)} = \log p^{(i)} - \log g^{(a^{(i)})}$
  is unbounded; consider clipping to $[-50, 50]$ if numerical issues arise.
- When $p_t = 1$ (guaranteed jump), the pilot is a single Gaussian
  ($k=1$ branch only), which keeps the correction weights well-behaved.

### 4.5 Generator update

`HestonProcess.generator` must be updated to:

1. Sample $I_t \sim \mathrm{Bernoulli}(p_t)$ via a uniform draw.
2. Sample $Z_1, Z_2 \sim \mathcal{N}(0,1)$ independently per step.
3. Apply eqs. (1) and (2): floor the diffusion, then multiply variance by $\exp(I_t J_V^t)$.
4. Variance floor is already respected by eq. (1); no extra clamp needed.

The returned `(log_returns, variances)` signature is unchanged.

---

## 5. Summary of changes to `heston_process.py`

| Component | Change |
|-----------|--------|
| `loglikelihood` parameter unpacking | add indices 6–11: `p_J, mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J` |
| First-stage pilot `log_g` | `logaddexp` mixture; jump branch uses `v_pilot_1 = v_pilot_0 * m_V` in both mean and diffusion variance (eqs. 9b, 10b) |
| Propagation keys | split one extra key for $Z_1$ and one for the uniform $u$ |
| Variance propagation | `v_cont = max(diffusion, δ)`; then `v_next = v_cont * exp(I_t * (mu_JV + sigma_JV * Z1))` |
| Conditional mean `mu_cond` | add `I_t * (mu_Jr + sigma_Jr * rho_J * Z1)` jump term |
| Conditional variance `sig_cond_sq` | add `I_t * sigma_Jr**2 * (1 - rho_J**2)` jump term |
| One-step-ahead prediction | include expected jump in `pred_mean_i` and `pred_var_i` (eqs. 18–20) |
| `get_default_param` | extend bounds/guess arrays to length 12 |
| `generator` | add Bernoulli draw and correlated jump sampling |
