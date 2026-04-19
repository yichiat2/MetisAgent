# Semivariance Heston Model for Scalping: Design, Estimation, and RL

---

## 1. Skewed Returns and Optimal TP/SL Design

### Return Distribution Under Asymmetric Volatility

Under separated upside/downside CIR processes with long-run levels $\theta^- > \theta^+$ and
Heston leverage $\rho < 0$, the single-bar log-return is a **variance-mean mixture**:

$$
r_t \mid v_t \;\sim\; \mathcal{N}\!\left(\mu\Delta t - \tfrac{1}{2}v_t\Delta t,\; v_t\Delta t\right)
$$

Integrating over the marginal (approximately Gamma-distributed) $v_t$ yields a **left-skewed,
fat-tailed unconditional distribution** — empirically a Normal-Inverse Gaussian (NIG) or
Variance-Gamma family. The third cumulant (skewness contribution) has two additive sources:

$$
\text{Skew}[r_t] \approx \underbrace{-\frac{\rho\,\xi\,v_t^{3/2}\Delta t^{1/2}}{(v_t\Delta t)^{3/2}}}_{\text{leverage effect}} + \underbrace{f(\theta^- - \theta^+)}_{\text{semivariance asymmetry}}
$$

Both terms are negative under empirically realistic parameters, meaning **down-moves tend to
be larger than up-moves** at any given Vol level.

### Touch Probabilities for TP and SL

For a **long** entry with TP at $+\delta^+$ ticks and SL at $-\delta^-$ ticks, the barrier
touch probabilities (reflection principle, GBM, zero drift) are:

$$
p_{\text{TP}}(\delta^+) \approx 2\,\mathbb{E}_{v^+}\!\left[\Phi\!\left(\frac{-\delta^+}{\sqrt{v_t^+\,\Delta t}}\right)\right], \qquad
p_{\text{SL}}(\delta^-) \approx 2\,\mathbb{E}_{v^-}\!\left[\Phi\!\left(\frac{-\delta^-}{\sqrt{v_t^-\,\Delta t}}\right)\right]
$$

In practice these expectations are computed as weighted sums over APF particles:

$$
p_{\text{TP}} = 2\sum_i w_t^{(i)}\,\Phi\!\left(\frac{-\delta^+}{\sqrt{v_t^{+,(i)}\,\Delta t}}\right), \qquad
p_{\text{SL}} = 2\sum_i w_t^{(i)}\,\Phi\!\left(\frac{-\delta^-}{\sqrt{v_t^{-,(i)}\,\Delta t}}\right)
$$

### Expected Value and Optimal Ratio

Expected P&L per trade (ticks, net of one-way transaction cost $c$):

$$
\text{EV}_{\text{long}} = \delta^+\,p_{\text{TP}}(\delta^+) - \delta^-\,p_{\text{SL}}(\delta^-) - c
$$

Maximising $\text{EV}$ over $\delta^+$ at fixed $\delta^- = D$ requires:

$$
\frac{\partial}{\partial\delta^+}\!\left[\delta^+\cdot 2\Phi\!\left(\frac{-\delta^+}{\sqrt{v^+\Delta t}}\right)\right] = 0
\implies
\Phi(-z^*) - z^*\phi(z^*) = 0, \quad z^* \approx 0.76
$$

So the **optimal tick distances** are:

$$
\boxed{\delta^{+*} \approx 0.76\sqrt{v_t^+\,\Delta t}, \qquad \delta^{-*} \approx 0.76\sqrt{v_t^-\,\Delta t}}
$$

and the **optimal TP/SL ratio** is simply the square root of the directional variance ratio:

$$
\frac{\delta^{+*}}{\delta^{-*}} = \sqrt{\frac{v_t^+}{v_t^-}}
$$

The asymmetry is the signal: when $v_t^- \gg v_t^+$, a **short entry** has a favourable TP/SL
ratio because the SL (driven by $v^+$) is cheap while the TP (driven by $v^-$) is reachable.

### Regime–Action–TP/SL Table

| Regime | $\hat{v}^+$ vs $\hat{v}^-$ | Optimal direction | TP/SL ratio |
|---|---|---|---|
| Upside vol dominance | $v^+ \gg v^-$ | Long | $> 1$ (TP wide) |
| Downside vol dominance | $v^- \gg v^+$ | Short | $> 1$ (TP wide) |
| Balanced high vol | $v^+ \approx v^- \gg \theta$ | Both (trend) | $\approx 1$ |
| Balanced low vol | $v^+ \approx v^- \ll \theta$ | MM limit both | $\approx 1$ (tight TP) |

---

## 2. Semivariance Heston Processes and OHLC Observations

### 2.1 Latent State: Two Independent CIR Processes

The **upside** and **downside** instantaneous variances each follow a Cox-Ingersoll-Ross (CIR)
process:

$$
dv_t^+ = \kappa^+(\theta^+ - v_t^+)\,dt + \xi^+\sqrt{v_t^+}\,dW_t^{v+}
$$

$$
dv_t^- = \kappa^-(\theta^- - v_t^-)\,dt + \xi^-\sqrt{v_t^-}\,dW_t^{v-}
$$

with Feller conditions $2\kappa^\pm\theta^\pm > (\xi^\pm)^2$ ensuring $v_t^\pm > 0$ a.s., and
correlation structure:

$$
\langle dW_t^S,\,dW_t^{v+}\rangle = \rho^+\,dt, \quad
\langle dW_t^S,\,dW_t^{v-}\rangle = \rho^-\,dt, \quad
\langle dW_t^{v+},\,dW_t^{v-}\rangle = \rho^{+-}\,dt
$$

The log-price process selects between them based on the sign of the driving Brownian motion. Because the shock is no longer strictly mean-zero given the sign-selected volatility, substituting the standard symmetric $d\log S_t \approx (r - 0.5\sigma_t^2)dt + \sigma_t dW_t^S$ drift is mathematically invalid. To enforce the risk-neutral martingale condition $\mathbb{E}^{\mathbb{Q}}[S_t] = S_0 e^{rt}$, the exact drift compensator $c_t$ requires integrating over both branches of the standard normal constraint:

$$
d\log S_t = c_t\,dt + \sqrt{v_t^+\,\phi_t^+ + v_t^-\,\phi_t^-}\;dW_t^S, \qquad
\phi_t^\pm = \mathbf{1}(\pm dW_t^S > 0)
$$

$$
1 = \mathbb{E}^{\mathbb{Q}}\!\left[\exp\Big(c_t \Delta t + \sigma_t(Z)\sqrt{\Delta t} Z \Big)\right]e^{-r\Delta t} \implies c_t \Delta t = r\Delta t - \ln\left( e^{\frac{1}{2}v_t^- \Delta t}\Phi(-\sqrt{v_t^-\Delta t}) + e^{\frac{1}{2}v_t^+ \Delta t}\Phi(\sqrt{v_t^+\Delta t}) \right)
$$

### 2.2 Discrete-Time Euler Scheme (1-Minute Bars)

At each bar $t$ with calendar duration $\Delta t$, propagate each process:

$$
v_t^\pm = v_{t-1}^\pm + \kappa^\pm(\theta^\pm - v_{t-1}^\pm)\Delta t + \xi^\pm\sqrt{v_{t-1}^\pm\,\Delta t}\;\epsilon_t^\pm, \qquad \epsilon_t^\pm \overset{\text{iid}}{\sim}\mathcal{N}(0,1)
$$

with reflection: $v_t^\pm \leftarrow \max(v_t^\pm,\, \varepsilon_{\min})$. The observed return is then constructed using the exact drift $c_t$:

$$
r_t^{OC} = c_t \Delta t + \sqrt{\sigma_t^2 \Delta t}\; \epsilon_t^S \qquad \sigma_t^2 = \begin{cases} v_t^+ & \text{if } \epsilon_t^S \ge 0 \\ v_t^- & \text{if } \epsilon_t^S < 0 \end{cases}
$$

Because the volatility is strictly positive, the sign of the residual precisely matches the sign of the unobserved true shock $\epsilon_t^S$. For the particle filter viewing raw returns, we deduce the active direction dynamically without ambiguity:
$$ \text{Up-bar} \iff r_t^{OC} \ge c_t \Delta t $$

### 2.3 Bivariate OHLC Observation

Each intraday bar provides two near-independent observations of the latent state.

**Observation 1 — Intrabar log-return** ($r_t^{OC}$ not $r_t^{CC}$, to avoid gap contamination):

$$
r_t^{OC} = \ln\frac{C_t}{O_t}
$$

Emission: $r_t^{OC} \mid v_t^\pm \sim \mathcal{N}\!\left(c_t\Delta t,\; \sigma_t^2\Delta t\right)$
*(Note: Conditioning in the APF additionally requires exact cross-correlation bivariate constraints from the joint CIR noises. See section 2.5.)*

**Observation 2 — Rogers-Satchell (RS) intrabar variance estimate** (drift-free, ~8× more
efficient than close-to-close squared return):

$$
\hat{v}_{\text{RS},t} = \ln\!\frac{H_t}{O_t}\ln\!\frac{H_t}{C_t} + \ln\!\frac{L_t}{O_t}\ln\!\frac{L_t}{C_t}
$$

with $\mathbb{E}[\hat{v}_{\text{RS},t}\mid v_t] = v_t\,\Delta t$ exactly for any drift $\mu$.

Emission: $\hat{v}_{\text{RS},t} \mid \sigma_t^2 \sim \text{Gamma}\!\left(\alpha,\;\frac{\sigma_t^2\Delta t}{\alpha}\right)$

where shape parameter $\alpha$ is calibrated offline from historical RS variance via
$\text{Var}[\hat{v}_{\text{RS},t}] = \sigma_t^4\Delta t^2/\alpha$.

**Directional RS** (used for the asymmetric model — only the active leg is observed):

$$
\hat{v}_{\text{RS},t}^+ = \hat{v}_{\text{RS},t}\;\mathbf{1}(r_t^{OC} \ge c_t\Delta t), \qquad
\hat{v}_{\text{RS},t}^- = \hat{v}_{\text{RS},t}\;\mathbf{1}(r_t^{OC} < c_t\Delta t)
$$

### 2.4 Joint Log-Likelihood (APF Emission)

For the **up-bar** case ($r_t^{OC} \ge c_t\Delta t$):

$$
\log p(\mathbf{y}_t \mid \dots) =
\underbrace{\log\mathcal{N}\!\left(r_t^{OC}\;\Big|\;\text{cond. mean}, \text{cond. var}\right)}_{\text{return, conditioned on CIR noises, uses }v^+}
+\underbrace{\log\text{Gamma}\!\left(\hat{v}_{\text{RS},t}^+\;\Big|\;\alpha^+,\;\frac{v_t^+\Delta t}{\alpha^+}\right)}_{\text{range, uses }v^+}
$$

For the **down-bar** case ($r_t^{OC} < c_t\Delta t$), replace $v^+ \to v^-$ and $\hat{v}_{\text{RS}}^+ \to \hat{v}_{\text{RS}}^-$.

The inactive process propagates under its prior (no likelihood term); its uncertainty grows
between directional observations, which is the Bayesian analogue of the Kinlay carry-forward.

### 2.5 APF Particle Structure

Each particle carries the full state:

$$
\mathbf{x}_t^{(i)} = (v_t^{+,(i)},\; v_t^{-,(i)}) \in \mathbb{R}_{>0}^2
$$

At each bar the APF auxiliary weight pre-selects on the **active** component only:

$$
\tilde{w}_t^{(i)} \propto w_{t-1}^{(i)}\;p\!\left(\mathbf{y}_t \;\big|\; \mu_t^{(i)}\right)
$$

where $\mu_t^{(i)}$ is the deterministic pilot mean of the active CIR. After resampling,
both $v^+$ and $v^-$ are propagated forward. However, $\epsilon_S$ is correlated with *both* variance processes ($\rho^+$ and $\rho^-$). To share PRNG streams efficiently across particles, we construct the variance noises structurally mapped onto common standard basis normals ($\epsilon_{vp}, \eta_{vm}$):

$$ \epsilon_{vp} \sim \mathcal{N}(0,1), \qquad \epsilon_{vm} = \rho^{+-} \epsilon_{vp} + \sqrt{1-(\rho^{+-})^2} \eta_{vm} $$

To evaluate the true exact likelihood of $r_t^{OC} \mid v_t^{\pm,(i)}$ given that the variance shocks are now known in the correction stage, we perform **bivariate Gaussian conditioning** on $\epsilon_S$ against the joint drawn plane $(\epsilon_{vp}, \eta_{vm})$:

$$
\beta = \frac{\rho^- - \rho^+ \rho^{+-}}{\sqrt{1 - (\rho^{+-})^2}} \qquad\implies\qquad \mathbb{E}[\epsilon_S \mid \epsilon_{vp}, \eta_{vm}] = \rho^+ \epsilon_{vp} + \beta \eta_{vm}, \qquad \text{Var}(\epsilon_S \mid \epsilon_{vp}, \eta_{vm}) = 1 - (\rho^+)^2 - \beta^2
$$

The corrected un-normalised emission weight evaluated at exactly matching variance constraints is:

$$ r_t^{OC,(i)} \sim \mathcal{N}\!\left(c_t \Delta t + \sqrt{v_t^{\pm,(i)} \Delta t} \, \mathbb{E}[\epsilon_S \mid \epsilon_{vp}, \eta_{vm}], \;\;\;v_t^{\pm,(i)} \Delta t \big(1 - (\rho^+)^2 - \beta^2\big) \right) $$

$$
w_t^{(i)} \propto \frac{p(\mathbf{y}_t \mid \dots)}{p(\mathbf{y}_t \mid \text{pilot})}
$$

### 2.6 Parameter Vector (13 parameters)

$$
\boldsymbol{\psi} = \left(v_0^+,\; v_0^-,\; \kappa^+,\; \kappa^-,\; \theta^+,\; \theta^-,\; \xi^+,\; \xi^-,\; \rho^+,\; \rho^-,\; \rho^{+-},\; \mu,\; \lambda_{\text{ov}}\right)
$$

Calibration is via CMA-ES maximising the APF log-likelihood with Common Random Numbers
(CRN), identical to the existing `InhomoHestonProcess` infrastructure. The overnight step
propagates both CIR processes under a sub-stepped prior with intensity $\lambda_{\text{ov}}$.

---

## 3. Comparison with the Double Heston Model

### 3.1 Double Heston (Christoffersen, Heston, Jacobs 2009)

Double Heston uses **two symmetric variance components** summed to drive all returns:

$$
d\log S_t = \mu\,dt + \sqrt{v_t^{(1)} + v_t^{(2)}}\,dW_t^S
$$

$$
dv_t^{(j)} = \kappa_j(\theta_j - v_t^{(j)})\,dt + \xi_j\sqrt{v_t^{(j)}}\,dW_t^{v_j}, \quad j=1,2
$$

with $\langle dW_t^S, dW_t^{v_j}\rangle = \rho_j\,dt$ and the two components typically
interpreted as a **fast mean-reverting** ($\kappa_1 \gg \kappa_2$) and a **slow persistent**
component.

### 3.2 Structural Differences

| Property | Double Heston | Semivariance Heston |
|---|---|---|
| Number of variance processes | 2 | 2 |
| How components combine | $v^{(1)} + v^{(2)}$ always | $v^+$ XOR $v^-$ (sign switch) |
| Component interpretation | Fast vs. slow mean-reversion | Upside vs. downside diffusion |
| Same-bar direction signal | None — both active simultaneously | Direct — only active-sign process drives bar |
| Leverage per component | $\rho_1, \rho_2$ (both $< 0$) | $\rho^+, \rho^-$ asymmetric |
| RS semivariance | Not separately identified | Directly observed per direction |
| Optimal for vol smile fitting | Yes (two humps in term structure) | Weaker (designed for time-series) |
| Optimal for bar-level touch probability | No — $v^{(1)}+v^{(2)}$ is undirected | Yes — directional $v^\pm$ maps to $p_\text{touch}^\pm$ |

### 3.3 Can Double Heston Replace the Semivariance Model?

**No, for two fundamental reasons.**

**Reason 1 — Direction identification.** In Double Heston the total variance $v_t^{(1)} + v_t^{(2)}$ is the same regardless of whether the bar goes up or down; the model has no mechanism to assign a different diffusion magnitude to up-bars vs. down-bars within the same time step. The semivariance model does this by construction: $\sigma_t^2 = v_t^+$ on up-bars and $\sigma_t^2 = v_t^-$ on down-bars.

**Reason 2 — Observation structure.** Double Heston is identified from option prices or close-to-close returns; neither observation separately constrains $v^{(1)}$ vs. $v^{(2)}$ without strong prior assumptions. The semivariance model is identified from OHLC data through directional RS: $\hat{v}_{\text{RS}}^+$ constrains $v^+$ on up-bars and $\hat{v}_{\text{RS}}^-$ constrains $v^-$ on down-bars. The observations and the state are matched by direction.

**Where Double Heston is superior:** fitting the term structure of implied vol and the volatility smile across strikes and maturities. Double Heston's two components independently price short-dated and long-dated options. The semivariance model does not improve option pricing.

**Where Semivariance Heston is superior:** real-time sequential filtering for directional scalping, where the key question at each bar is "which direction is volatility elevated in?" — a question Double Heston cannot answer.

**Conclusion:** the two models are **complementary, not substitutes**. For a scalping RL agent, the semivariance model is the appropriate choice. For options desk risk management, Double Heston is more relevant.

---

## 4. Parsimonious RL Agent Design Using the Semivariance Model

### 4.1 Core Principle: Compress Everything into the State

The APF serves as a **sufficient-statistic compressor**: it maps the raw OHLCV stream into a
small, information-rich state vector. The RL agent never sees raw prices — only PF posteriors.
This decouples the **perception problem** (what is the vol regime?) from the **decision
problem** (what action to take?), making both easier to solve.

### 4.2 Minimal Sufficient State

At bar $t$, after the APF update, extract:

$$
s_t = \left(
\underbrace{\hat{v}_t^+,\; \sigma_{v,t}^+}_{\text{upside posterior}},\;
\underbrace{\hat{v}_t^-,\; \sigma_{v,t}^-}_{\text{downside posterior}},\;
\underbrace{P_{\text{HV},t}^+,\; P_{\text{LV},t}^+,\; P_{\text{HV},t}^-,\; P_{\text{LV},t}^-}_{\text{soft regime probs}},\;
\underbrace{\tilde{p}_{\text{touch},t}^{\text{TP}},\; \tilde{p}_{\text{touch},t}^{\text{SL}}}_{\text{barrier probs}},\;
\underbrace{r_{t-1}^{OC}}_{\text{last return}},\;
\underbrace{\tau_t}_{\text{time-of-day}},\;
\underbrace{q_t}_{\text{position}}
\right) \in \mathbb{R}^{13}
$$

where the soft regime probabilities are posterior tail masses:

$$
P_{\text{HV},t}^\pm = \sum_i w_t^{(i)}\,\mathbf{1}(v_t^{\pm,(i)} > \tau_H), \qquad
P_{\text{LV},t}^\pm = \sum_i w_t^{(i)}\,\mathbf{1}(v_t^{\pm,(i)} < \tau_L)
$$

and the touch probabilities use the proposed TP/SL from the previous action $a_{t-1}$:

$$
\tilde{p}_{\text{touch},t}^{\text{TP}} = 2\sum_i w_t^{(i)}\,\Phi\!\left(\frac{-\delta_{t-1}^+}{\sqrt{v_t^{+,(i)}\,\Delta t}}\right), \qquad
\tilde{p}_{\text{touch},t}^{\text{SL}} = 2\sum_i w_t^{(i)}\,\Phi\!\left(\frac{-\delta_{t-1}^-}{\sqrt{v_t^{-,(i)}\,\Delta t}}\right)
$$

### 4.3 Factored Discrete Action Space

Use three independent categorical heads sharing a policy trunk $h_t = f_\theta(s_t)$:

$$
\log\pi_\theta(a_t \mid s_t) = \log\pi_\theta^{\text{dir}}(d_t \mid h_t) + \log\pi_\theta^{\text{type}}(o_t \mid h_t) + \log\pi_\theta^{\text{risk}}(k_t \mid h_t)
$$

| Head | Space | Values |
|---|---|---|
| Direction $d_t$ | $\{+1, 0, -1\}$ | Long, Flat, Short |
| Order type $o_t$ | $\{\text{Mkt}, \text{Lmt}\}$ | Market, Limit |
| TP/SL tier $k_t$ | $\{T, M, W\}$ | Tight, Medium, Wide |

Predefined TP/SL tier lookup (in ticks, calibrated to $z^* \approx 0.76\sqrt{\hat{v}\,\Delta t}$):

| Tier | TP ($\delta^+$) | SL ($\delta^-$) | Intended regime |
|---|---|---|---|
| Tight $T$ | 2 | 30 | Low-vol mean-reversion |
| Medium $M$ | 10 | 10 | Ambiguous / transitional |
| Wide $W$ | 30 | 2 | High-vol trend-following |

This gives $3 \times 2 \times 3 = 18$ combinations; with Flat action collapsing order type and
tier, the effective distinct actions are 13.

### 4.4 Reward Function

$$
r_t = \underbrace{\Delta\text{PnL}_t^{\text{realized}}}_{\text{ticks, on close}} - \underbrace{\lambda\,|\Delta q_t|}_{\text{transaction cost}} - \underbrace{\mu_{\text{ov}}\,\mathbf{1}[\text{open overnight}]}_{\text{overnight penalty}}
$$

Use **realized** (not mark-to-market) P&L only: reward is credited when a TP or SL is hit
or at session close via `setexitonclose`. This avoids shaping the agent toward holding
losers to avoid unrealized loss crystallisation.

### 4.5 Training Pipeline

**Step 1 — Offline PF calibration.** Calibrate $\boldsymbol{\psi}$ via CMA-ES on
historical OHLCV using the APF log-likelihood. Freeze parameters.

**Step 2 — State dataset generation.** Run the calibrated APF forward on the training
window, saving $(s_t, a_t^*, r_t^*)$ tuples where $a_t^*$ is the Kinlay rule-based action
and $r_t^*$ is the realized P&L following that action.

**Step 3 — Behavioural cloning warm-start.** Pre-train the policy via supervised imitation:

$$
\mathcal{L}_{\text{BC}}(\theta) = -\mathbb{E}_{(s_t, a_t^*)\sim\mathcal{D}}\!\left[\log\pi_\theta(a_t^* \mid s_t)\right]
$$

This initialises the policy near the known rule-based solution, eliminating cold-start
exploration failures in a sparse reward scalping environment.

**Step 4 — PPO fine-tuning.** Run PPO with entropy regularisation per head. The policy
explores away from the rule-based baseline toward higher-EV actions — particularly
discovering when Medium tier outperforms Tight/Wide in ambiguous regime states where
$\sigma_{v,t}^\pm$ (posterior uncertainty) is large.

**Step 5 — Optional joint fine-tuning.** If an EnKF (differentiable filter) replaces the
bootstrap APF, gradients can flow from the policy loss through $\hat{v}_t^\pm$ back into
$\boldsymbol{\psi}$, end-to-end. This is more complex but allows the filter parameters to
adapt to the reward objective rather than purely the likelihood.

### 4.6 Why This Design Is Parsimonious

- **State dimension is 13**, regardless of history length — all temporal information is
  absorbed into the PF posterior.
- **A small MLP (2–3 hidden layers, 64–128 units) is sufficient** — the state is already
  engineered to be information-dense and nearly Markovian.
- **No sequence model (LSTM) is needed** unless the user suspects PF posterior is an
  insufficient statistic, which would indicate model misspecification rather than an
  architecture gap.
- **The Kinlay strategy is recoverable** as a degenerate deterministic policy: Wide tier
  when $P_\text{HV} > 0.5$, Tight tier when $P_\text{LV} > 0.5$. The RL agent generalises
  this to the continuous-probability regime, discovering non-trivial behaviour in the
  intermediate ($P \approx 0.5$) region that the original fixed-threshold rule ignores.

---

## 5. Alternative Design: Asymmetric Leverage Effect

### 5.1 Model Specification

Rather than introducing two independent CIR processes, an alternative approach keeps a
**single variance process** and makes the leverage correlation sign-dependent:

$$
\rho_t = \rho_0 + \rho_1\,\mathbf{1}_{dS_t < 0}, \qquad \rho_0 \in (-1,1),\quad \rho_0 + \rho_1 \in (-1,1)
$$

The full model is then:

$$
d\log S_t = \mu\,dt + \sqrt{v_t}\,dW_t^S
$$

$$
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \qquad
\langle dW_t^S,\,dW_t^v\rangle = \rho_t\,dt
$$

Decomposing the Brownian motion: $dW_t^v = \rho_t\,dW_t^S + \sqrt{1-\rho_t^2}\,dW_t^\perp$,
the variance innovation on a down-bar ($dS_t < 0$) receives a leverage coefficient
$\rho_0 + \rho_1$ while an up-bar receives only $\rho_0$. Under the empirically realistic
choice $\rho_1 < 0$, the total leverage for down-moves is more negative, so
**downward price shocks pump variance more strongly** than upward shocks of equal size.

The Euler–Maruyama discretisation at bar $t$ is:

$$
r_t^{OC} = \left(\mu - \tfrac{1}{2}v_t\right)\Delta t + \sqrt{v_t\,\Delta t}\;\epsilon_t^S
$$

$$
v_{t+1} = v_t + \kappa(\theta - v_t)\Delta t + \xi\sqrt{v_t\,\Delta t}
\left(\rho_t\,\epsilon_t^S + \sqrt{1-\rho_t^2}\;\epsilon_t^\perp\right)
$$

$$
\rho_t = \rho_0 + \rho_1\,\mathbf{1}_{r_t^{OC} < 0}
$$

with $v_{t+1} \leftarrow \max(v_{t+1}, \varepsilon_{\min})$. The parameter vector is:

$$
\boldsymbol{\psi}_{\text{alt}} = (v_0,\;\kappa,\;\theta,\;\xi,\;\rho_0,\;\rho_1,\;\mu,\;\lambda_{\text{ov}})
\in \mathbb{R}^8
$$

versus 13 parameters for the two-CIR design.

---

### 5.2 Effective Directional Volatilities via Leverage Impulse

Although $v_t$ is scalar, the asymmetric leverage induces direction-conditional predictive
variances for bar $t{+}1$. Taking the conditional expectation of the CIR update and using
$\mathbb{E}[|\epsilon^S|] = \sqrt{2/\pi}$:

$$
v_{\mathrm{pred},t}^+ \;\equiv\; \mathbb{E}[v_{t+1} \mid r_t^{OC} > 0,\, v_t]
= v_t + \kappa(\theta - v_t)\Delta t + \xi\sqrt{v_t\,\Delta t}\;\rho_0\sqrt{\tfrac{2}{\pi}}
$$

$$
v_{\mathrm{pred},t}^- \;\equiv\; \mathbb{E}[v_{t+1} \mid r_t^{OC} < 0,\, v_t]
= v_t + \kappa(\theta - v_t)\Delta t - \xi\sqrt{v_t\,\Delta t}\;(\rho_0+\rho_1)\sqrt{\tfrac{2}{\pi}}
$$

(The sign difference arises because $\mathbb{E}[\epsilon^S \mid r^{OC}<0] = -\sqrt{2/\pi}$.)
The **asymmetric leverage impulse** that separates the two predictions is:

$$
\Lambda_t \;\equiv\; v_{\mathrm{pred},t}^- - v_{\mathrm{pred},t}^+
= -\xi\sqrt{v_t\,\Delta t}\;\rho_1\sqrt{\tfrac{2}{\pi}} > 0
\quad (\text{since }\rho_1 < 0)
$$

For $\rho_1 = -0.5$, $\xi = 0.4$, $v_t = 2\times10^{-4}$, $\Delta t = 1/390$:
$\Lambda_t \approx 0.4\cdot\sqrt{2\times10^{-4}/390}\cdot0.5\cdot0.798 \approx 7.2\times10^{-5}$,
roughly a 36 % spread over $v_t$ itself — substantial next-bar directional signal.

The APF particle estimate of these predictive vols is:

$$
\hat{v}_{\mathrm{pred},t}^\pm = \sum_i w_t^{(i)}\,v_{\mathrm{pred},t}^{\pm,(i)}
$$

These play the role of $\hat{v}_t^\pm$ from the two-CIR model but are **one bar ahead**
rather than simultaneous.

---

### 5.3 TP/SL Optimisation Under Asymmetric Leverage

#### Single-Bar Trade (Intrabar TP/SL)

Within bar $t$ the price diffuses under the **same** $v_t$ regardless of direction, because
$v_t$ is the variance determined at bar $t{-}1$ before the current bar opens. The barrier
touch probabilities for a long position are therefore symmetric in vol:

$$
p_{\text{TP}}(\delta^+) = 2\,\Phi\!\left(\frac{-\delta^+}{\sqrt{v_t\,\Delta t}}\right), \qquad
p_{\text{SL}}(\delta^-) = 2\,\Phi\!\left(\frac{-\delta^-}{\sqrt{v_t\,\Delta t}}\right)
$$

Maximising $\text{EV} = \delta^+\,p_{\text{TP}} - \delta^-\,p_{\text{SL}} - c$ over $\delta^\pm$
independently yields the same $z^* \approx 0.76$ condition for both legs:

$$
\delta^{+*} = \delta^{-*} = 0.76\sqrt{v_t\,\Delta t}
$$

**Single-bar trades are therefore symmetric in the asymmetric leverage model** — no
directional edge is available from the TP/SL ratio alone within the bar.

#### Multi-Bar Trade (Deferred Leverage Feedback)

For a position held over $K$ bars, the adverse path accumulates variance faster. A first-order
approximation for the average running variance along each path from the leverage impulse is:

$$
\bar{v}^{+}(K) \;\approx\; \theta + (v_t - \theta)\,\frac{1-(1-\kappa\Delta t)^K}{K\kappa\Delta t}
+ \xi\sqrt{v_t\,\Delta t}\;\rho_0\sqrt{\tfrac{2}{\pi}}\cdot\frac{K-1}{2K}
$$

$$
\bar{v}^{-}(K) \;\approx\; \theta + (v_t - \theta)\,\frac{1-(1-\kappa\Delta t)^K}{K\kappa\Delta t}
- \xi\sqrt{v_t\,\Delta t}\;(\rho_0+\rho_1)\sqrt{\tfrac{2}{\pi}}\cdot\frac{K-1}{2K}
$$

The directional accumulated diffusion widths entering the $z^*$ formula are
$\sqrt{\bar{v}^\pm(K)\cdot K\cdot\Delta t}$. Applying the same EV-maximisation:

$$
\delta^{+*}(K) = 0.76\sqrt{\bar{v}^+(K)\cdot K\,\Delta t}, \qquad
\delta^{-*}(K) = 0.76\sqrt{\bar{v}^-(K)\cdot K\,\Delta t}
$$

Since $\bar{v}^-(K) > \bar{v}^+(K)$ for $\rho_1 < 0$, the **optimal TP/SL ratio for a long
multi-bar position is less than one**:

$$
\frac{\delta^{+*}(K)}{\delta^{-*}(K)} = \sqrt{\frac{\bar{v}^+(K)}{\bar{v}^-(K)}} < 1
\quad\Longrightarrow\quad \delta^{-*} > \delta^{+*}
$$

The SL must be set wider than the TP to account for the vol expansion on adverse down-moves.

#### Directional EV at the Optimum

Using $\text{EV}^* = 2z^*\phi(z^*)\bigl(\delta^{+*} - \delta^{-*}\bigr) - c$ and substituting:

$$
\text{EV}^*_{\text{long}}(K) = 2z^*\phi(z^*)\sqrt{K\,\Delta t}\!\left(\sqrt{\bar{v}^+(K)} - \sqrt{\bar{v}^-(K)}\right) - c \;<\; 0
$$

$$
\text{EV}^*_{\text{short}}(K) = 2z^*\phi(z^*)\sqrt{K\,\Delta t}\!\left(\sqrt{\bar{v}^-(K)} - \sqrt{\bar{v}^+(K)}\right) - c \;>\; 0
$$

This is precisely the scalping signal: **when $\rho_1 < 0$, a short position has positive
optimal EV; a long position has negative optimal EV**, consistent with the two-CIR analysis
but derived one bar later via the leverage channel. The direction signal is:

$$
\text{go short if}\quad \Lambda_t = -\xi\sqrt{v_t\,\Delta t}\;\rho_1\sqrt{\tfrac{2}{\pi}} > \frac{c}{2z^*\phi(z^*)\sqrt{K\,\Delta t}}
$$

i.e.\ whenever the leverage-induced vol asymmetry exceeds the round-trip breakeven.

---

### 5.4 Pros and Cons vs. the Two-CIR Semivariance Model

| Property | Two-CIR Semivariance | Asymmetric Leverage |
|---|---|---|
| **Parameter count** | 13 | 8 |
| **State dimension (particle)** | 2D $(v^+, v^-)$ | 1D $(v)$ |
| **Same-bar directionality** | Yes — $v^+$ XOR $v^-$ drives current bar | No — single $v_t$ applies to both TP and SL |
| **When asymmetry manifests** | Instantaneous (current bar) | Deferred (next-bar predictive) |
| **RS observation** | Directional split $\hat{v}_\text{RS}^+$, $\hat{v}_\text{RS}^-$ | Single $\hat{v}_\text{RS}$ constrains $v_t$ directly |
| **Identification** | From directional OHLC | From leverage correlation (return sign × subsequent vol change) |
| **Feller condition** | Two conditions $2\kappa^\pm\theta^\pm>(\xi^\pm)^2$ | One condition $2\kappa\theta>\xi^2$ |
| **APF complexity** | Bivariate resampling | Standard scalar APF |
| **Option pricing** | Weak | Closer to original Heston |
| **Regime expressivity** | Four regimes: $v^+ \times v^-$ grid | Two: high/low overall $v$ (no directional split) |
| **TP/SL asymmetry onset** | Immediate — exploitable in bar $t$ | Deferred — exploitable from bar $t{+}1$ |
| **Calibration sample requirement** | Moderate (RS observations per direction) | Larger (leverage correlation estimated from return sign × vol cross-section) |

**Key advantages of asymmetric leverage:**

1. **Parsimony.** Eight parameters vs. 13; a single Feller condition; standard CIR
   infrastructure reused without modification. The APF is a straightforward scalar
   bootstrap filter identical to `InhomoHestonProcess` with one additional scalar parameter.

2. **Non-directional RS.** The Rogers–Satchell estimator constrains $v_t$ without
   requiring knowledge of which direction was dominant, increasing observation efficiency
   on flat or ambiguous bars.

3. **Analytical tractability.** The undirected characteristic function of the model is
   nearly Heston-standard, facilitating semi-analytical option pricing as a consistency
   check and for risk-management overlay.

**Key disadvantages of asymmetric leverage:**

1. **No same-bar directional signal.** The most valuable property of the two-CIR model —
   that $v_t^+ \ne v_t^-$ drives the *current* bar's diffusion asymmetrically — is absent.
   For a 1-minute scalper whose TP/SL is intrabar, the asymmetric leverage model yields no
   EV advantage from TP/SL sizing in the bar being traded.

2. **Deferred identification.** The asymmetry is identified only via the cross-sectional
   covariance of $\text{sign}(r_t^{OC})$ and $v_{t+1} - v_t$, requiring a longer history
   to pin down $\rho_1$ compared to the directional RS approach that provides an
   observation at every bar.

3. **Limited regime expressivity.** The single scalar $v_t$ cannot simultaneously
   represent "$v^+$ elevated, $v^-$ subdued" — a state that the two-CIR model identifies
   as a bullish vol regime and immediately exploits. The asymmetric leverage model
   conflates upside and downside vol into one number.

4. **Leverage stationarity assumption.** The model assumes $\rho_1$ is constant over the
   estimation window. In practice, leverage asymmetry varies by regime (stronger in
   distressed markets); misspecification here creates the kind of parameter
   non-stationarity that the two-CIR model's separate mean-reversion levels $\theta^\pm$
   partially absorb.

---

### 5.5 Hybrid Design: Asymmetric Leverage on Top of Two-CIR

The two approaches are not mutually exclusive. A **hybrid** model retains two CIR
processes but allows each to have its own asymmetric leverage:

$$
\rho_t^+ = \rho_0^+ + \rho_1^+\,\mathbf{1}_{r_t^{OC} < 0}, \qquad
\rho_t^- = \rho_0^- + \rho_1^-\,\mathbf{1}_{r_t^{OC} < 0}
$$

This adds 2 parameters ($\rho_1^+, \rho_1^-$) to the base two-CIR model (15 total), giving:

- **Same-bar directionality** from the CIR XOR structure (unchanged)
- **Next-bar vol feedback** from the asymmetric leverage on each arm

The second-stage APF weight then uses the full joint emission with direction-dependent
leverage, and the posterior predictive directional vols become:

$$
v_{\mathrm{pred}}^{+,\pm} = \sum_i w_t^{(i)}
\!\left[v_t^{+,(i)} + \kappa^+(\theta^+-v_t^{+,(i)})\Delta t \pm \xi^+\sqrt{v_t^{+,(i)}\Delta t}\,\rho_0^+\sqrt{\tfrac{2}{\pi}}\right]
$$

In practice the incremental expressivity is modest for 1-bar scalping relative to the
added calibration burden; the hybrid is most useful when the holding horizon $K \ge 5$
bars, where deferred leverage feedback is material.

---

### 5.6 Model Selection Guidance

| Use case | Recommended model |
|---|---|
| Intrabar 1-bar TP/SL scalping | Two-CIR (instantaneous directional vol) |
| Multi-bar swing scalping ($K \ge 5$) | Hybrid or Two-CIR + leverage asymmetry |
| Options risk overlay / smile fitting | Asymmetric leverage (closer to Heston) |
| Limited data (< 6 months OHLCV) | Asymmetric leverage (fewer parameters) |
| Full expressive semivariance filtering | Two-CIR |
| Fast calibration / production deployment | Asymmetric leverage (scalar APF, 8 params) |

**Bottom line.** The asymmetric leverage model is a **lighter, faster, analytically
cleaner** alternative that recovers the directional EV insight of the two-CIR model
**one bar later**. For high-frequency intrabar scalping the latency cost is prohibitive;
for multi-bar strategies the deferred signal is acceptable and the parsimony gains are
substantial. The two-CIR semivariance model remains the preferred design for the primary
use-case of 1-minute bar scalping.

---

## 6. State-Conditioned TP/SL Sizing, Entry Thresholds, and Trailing Stops

This section derives full-posterior TP/SL optimisation rules from the APF state,
establishes an analytically tractable entry threshold, and develops both a classical
Bayesian and an RL-based trailing-stop framework. A head-to-head comparison closes
the section.

---

### 6.1 Full-Posterior TP/SL Optimisation

#### EV Objective and Decoupled Legs

Given the APF posterior $\{w_t^{(i)},\, v_t^{+,(i)},\, v_t^{-,(i)}\}_{i=1}^N$ at bar $t$,
the expected value for a **long** entry is:

$$
\text{EV}_{\text{long}}(\delta^+, \delta^-) =
\delta^+\, p_{\text{TP}}(\delta^+;\,\{w^{(i)},v^{+,(i)}\})
- \delta^-\, p_{\text{SL}}(\delta^-;\,\{w^{(i)},v^{-,(i)}\})
- c
$$

where the weighted barrier-touch probabilities are (as in §1):

$$
p_{\text{TP}}(\delta^+) = 2\sum_i w_t^{(i)}\,\Phi\!\left(\frac{-\delta^+}{\sqrt{v_t^{+,(i)}\,\Delta t}}\right), \qquad
p_{\text{SL}}(\delta^-) = 2\sum_i w_t^{(i)}\,\Phi\!\left(\frac{-\delta^-}{\sqrt{v_t^{-,(i)}\,\Delta t}}\right)
$$

The two legs decouple: maximise $\delta^+ p_{\text{TP}}(\delta^+)$ and $\delta^- p_{\text{SL}}(\delta^-)$
independently. The first-order condition for $\delta^+$:

$$
\frac{\partial}{\partial\delta^+}\!\left[\delta^+\,p_{\text{TP}}(\delta^+)\right] = 0
\;\Longrightarrow\;
\sum_i w_t^{(i)}\!\left[\Phi\!\left(-z_t^{+,(i)}\right) - z_t^{+,(i)}\phi\!\left(z_t^{+,(i)}\right)\right] = 0
\tag{6.1}
$$

where $z_t^{+,(i)} \equiv \delta^+/\sqrt{v_t^{+,(i)}\,\Delta t}$ is the normalised distance
for particle $i$. This is a particle-weighted average of the scalar FOC $\Phi(-z) = z\phi(z)$,
whose root is $z^* \approx 0.758$. The aggregate root of (6.1) is equivalent to solving:

$$
G^+(\delta^+) \;\equiv\;
\sum_i w_t^{(i)}\!\left[
\Phi\!\left(\frac{-\delta^+}{\sqrt{v_t^{+,(i)}\Delta t}}\right)
- \frac{\delta^+\,\phi\!\left(\frac{\delta^+}{\sqrt{v_t^{+,(i)}\Delta t}}\right)}{\sqrt{v_t^{+,(i)}\Delta t}}
\right] = 0
$$

which is solved numerically in $O(N)$ operations via bisection on the monotone function $G^+$.

**Mean-field (first-order) approximation.** Using the posterior mean $\hat{v}_t^\pm \equiv \sum_i w_t^{(i)} v_t^{\pm,(i)}$:

$$
\boxed{\delta^{+*} \approx z^*\sqrt{\hat{v}_t^+\,\Delta t}, \qquad
\delta^{-*} \approx z^*\sqrt{\hat{v}_t^-\,\Delta t}}
\tag{6.2}
$$

**Second-order correction.** Expanding $G^+$ around $\hat{v}_t^+$ to second order in the
posterior variance $(\sigma_{v,t}^+)^2 \equiv \sum_i w_t^{(i)}(v_t^{+,(i)} - \hat{v}_t^+)^2$:

$$
\delta^{+*} \approx z^*\sqrt{\hat{v}_t^+\,\Delta t}\cdot
\left[1 + \frac{z^{*2}-1}{4}\cdot\frac{(\sigma_{v,t}^+)^2}{(\hat{v}_t^+)^2}\right]
\tag{6.3}
$$

Since $z^{*2}-1 \approx -0.43 < 0$, **higher posterior uncertainty shrinks the optimal TP**
— the filter automatically enforces conservative sizing when the vol signal is ambiguous.

#### Risk-Adjusted Sizing

Define the risk-penalised objective (subtract a multiple of touch-probability variance):

$$
\text{EV}_\beta(\delta^+) = \delta^+\,\mathbb{E}[p_{\text{TP}}] - \beta\,\delta^{+2}\,\text{Var}[p_{\text{TP}}] - (\text{SL leg}) - c
$$

where the variance is taken across the particle posterior:

$$
\text{Var}[p_{\text{TP}}] = 4\!\left[
\sum_i w_t^{(i)}\Phi^2\!\left(\frac{-\delta^+}{\sqrt{v_t^{+,(i)}\Delta t}}\right)
- \left(\frac{p_{\text{TP}}}{2}\right)^{\!2}
\right]
$$

The risk-adjusted FOC gives, to leading order:

$$
\delta^{+*}_\beta \approx \frac{z^*\sqrt{\hat{v}_t^+\,\Delta t}}{1 + \beta\,A_t^+}, \qquad
A_t^+ \;\equiv\; \frac{4\,\text{Var}[p_{\text{TP}}]}{z^*\phi(z^*)}
\tag{6.4}
$$

The factor $\beta A_t^+$ is zero when all particles agree (point mass posterior) and grows
with vol spread across particles. **Larger $\beta$: tighter TP; $\beta=0$: pure EV maximisation.**

---

### 6.2 Entry Threshold: When to Open a Position

#### Optimal EV as a Function of Posterior Means

Substituting the mean-field distances (6.2) into the EV objective and using
$\Phi(-z^*) = z^*\phi(z^*)$ (which follows directly from the FOC):

$$
\delta^{+*}\,p_{\text{TP}}(\delta^{+*}) \approx z^*\sqrt{\hat{v}_t^+\,\Delta t}\cdot 2z^*\phi(z^*)
= 2z^{*2}\phi(z^*)\sqrt{\hat{v}_t^+\,\Delta t}
$$

The net optimal EV for a long position is therefore:

$$
\text{EV}^*_{\text{long}} = 2z^{*2}\phi(z^*)\sqrt{\Delta t}
\!\left(\sqrt{\hat{v}_t^+} - \sqrt{\hat{v}_t^-}\right) - c
\tag{6.5}
$$

Setting $\text{EV}^*_{\text{long}} > 0$ gives the **long entry condition**:

$$
\boxed{
\sqrt{\hat{v}_t^+} - \sqrt{\hat{v}_t^-}
\;>\; \varepsilon_{\text{entry}} \;\equiv\; \frac{c}{2z^{*2}\phi(z^*)\sqrt{\Delta t}}
}
\tag{6.6}
$$

Symmetrically, enter **short** when $\sqrt{\hat{v}_t^-} - \sqrt{\hat{v}_t^+} > \varepsilon_{\text{entry}}$.

**Numerical values.** At $z^* = 0.758$: $2z^{*2}\phi(z^*) \approx 0.345$.
For 1-minute bars ($\Delta t = 1/390$, $\sqrt{\Delta t} \approx 0.0506$):

$$
\varepsilon_{\text{entry}} = \frac{c}{0.0175} \approx 57.1\,c
$$

So if transaction cost is $c = 0.01\%$ (in vol units), the entry requires at least a $0.57\%$ gap
between upside and downside vol square roots — a quantitative, calibration-free threshold.

#### Uncertainty-Adjusted Entry Threshold

Under posterior uncertainty, the realised EV is stochastic. A conservative entry requires the
$\alpha$-lower-confidence EV to be positive:

$$
\text{EV}^*_{\text{long}} - z_\alpha\,\sigma_{\text{EV},t} > 0, \qquad
\sigma_{\text{EV},t} \approx 2z^{*2}\phi(z^*)\sqrt{\Delta t}
\left[\frac{(\sigma_{v,t}^+)^2}{4\hat{v}_t^+} + \frac{(\sigma_{v,t}^-)^2}{4\hat{v}_t^-}\right]^{1/2}
\tag{6.7}
$$

This inflates the effective threshold when the filter is uncertain, naturally delaying entry
until a cleaner APF signal is available. With $\alpha = 1.28$ (90 % confidence):

$$
\sqrt{\hat{v}_t^+} - \sqrt{\hat{v}_t^-}
> \varepsilon_{\text{entry}} + \frac{z_\alpha}{2}
\sqrt{\frac{(\sigma_{v,t}^+)^2}{\hat{v}_t^+} + \frac{(\sigma_{v,t}^-)^2}{\hat{v}_t^-}}
\tag{6.8}
$$

#### Entry Rule Summary

| Signal | Classical condition | RL equivalent |
|---|---|---|
| Long | $\sqrt{\hat{v}^+} - \sqrt{\hat{v}^-} > \varepsilon_{\text{entry}}$ | $\pi^{\text{dir}}(\text{Long}\mid s_t) > 0.5$ |
| Short | $\sqrt{\hat{v}^-} - \sqrt{\hat{v}^+} > \varepsilon_{\text{entry}}$ | $\pi^{\text{dir}}(\text{Short}\mid s_t) > 0.5$ |
| Flat (no edge) | $\bigl|\sqrt{\hat{v}^+} - \sqrt{\hat{v}^-}\bigr| \le \varepsilon_{\text{entry}}$ | $\pi^{\text{dir}}(\text{Flat}\mid s_t) > 0.5$ |
| Conservative flat | Above OR uncertainty (6.8) not met | State includes $\sigma_{v,t}^\pm$; learned implicitly |

---

### 6.3 Classical Trailing Stop: Volatility-Adaptive and Drift-Aggressive

#### The Trailing Problem

Once a position is entered at bar $t_0$, the optimal SL from §6.1 should be recomputed
at every subsequent bar as the APF posterior updates. Let:

- $\Delta_t \equiv P_t - P_{t_0}$: cumulative tick PnL (positive = favourable for long)
- $\delta^{+*}_0 = z^*\sqrt{\hat{v}_{t_0}^+\,\Delta t}$: TP distance at entry
- $f_t \equiv \Delta_t / \delta^{+*}_0$: **profit fraction** ($f_t = 1$ when TP price is reached)

#### Base Bayesian Trailing Rule

At each bar $t > t_0$, re-apply the EV maximisation to the updated posterior:

$$
\delta^{-*}_t = z^*\sqrt{\hat{v}_t^-\,\Delta t}
$$

The **trailing SL price** enforces monotonicity (can only tighten, never widen):

$$
\text{SL}_t = \max\!\left(\text{SL}_{t-1},\; P_t - \delta^{-*}_t\right)
\tag{6.9}
$$

This rule tightens automatically when $\hat{v}_t^-$ falls (downside vol contracted after a
regime shift), and follows price up in a rising market. No hand-tuned trailing percentage is
needed; the model calibration fully determines the stop.

#### Drift-Triggered Aggressive Tightening

The base rule (6.9) does not exploit the *magnitude* of the drift. When the price has moved
significantly toward the TP, the remaining upside $\delta^{+*}_0 - \Delta_t$ shrinks while
the bare vol-optimal SL distance may still be wide. Define the **lock-in schedule**
$\eta: \mathbb{R} \to (0,1]$:

$$
\eta(f) = \exp\!\bigl(-\alpha^{\text{trail}}\,\max(0,\,f - f_0)\bigr), \qquad
f_0 \in (0,1),\quad \alpha^{\text{trail}} > 0
\tag{6.10}
$$

The **drift-adjusted SL distance** applies this tightening multiplier to the vol-optimal level:

$$
\delta^{-\text{trail}}_t = z^*\sqrt{\hat{v}_t^-\,\Delta t} \cdot \eta(f_t)
\tag{6.11}
$$

The trailing level becomes:

$$
\boxed{\text{SL}_t = \max\!\left(\text{SL}_{t-1},\; P_t - z^*\sqrt{\hat{v}_t^-\,\Delta t}\cdot
\exp\!\bigl(-\alpha^{\text{trail}}\,\max(0,f_t - f_0)\bigr)\right)}
\tag{6.12}
$$

**Worked example.** Parameters $f_0 = 0.5$, $\alpha^{\text{trail}} = 3$:

| $f_t$ | $\eta(f_t)$ | SL tightness vs. vol-optimal |
|---|---|---|
| $0.0$ | $1.00$ | Normal — no tightening |
| $0.5$ | $1.00$ | Threshold not yet crossed |
| $0.7$ | $e^{-0.6} \approx 0.55$ | SL is 45 % tighter |
| $0.9$ | $e^{-1.2} \approx 0.30$ | SL locks in 70 % of drift |
| $1.0$ | $e^{-1.5} \approx 0.22$ | SL very tight near TP price |

A linear variant avoids the exponential but adds a floor $\eta_{\min}$ to prevent the SL
collapsing to zero prematurely:

$$
\eta(f) = \max\!\left(\eta_{\min},\; 1 - \beta^{\text{trail}}\,\max(0,\,f - f_0)\right)
\tag{6.13}
$$

#### Inactive-Leg Uncertainty (Stale Downside Vol)

After $K$ consecutive up-bars, the downside-vol process has received no directional
observations (see §2.5); its posterior uncertainty $\sigma_{v,t}^-$ grows. When the filter
can no longer reliably price the SL, the prudent rule is to exit rather than hold an
SL anchored to a stale estimate:

$$
\text{Force exit if}\quad \sigma_{v,t}^- > \sigma_{\text{exit}} \quad\text{and}\quad f_t > f_{\min}
\tag{6.14}
$$

This embodies the principle: **when you no longer know the downside vol, exit and re-enter
once the filter refreshes.** The minimum profit fraction $f_{\min}$ prevents forced exits
on positions that have barely moved.

---

### 6.4 RL Approach to Dynamic Stop Management

#### Extended State Vector

Augment the base state $s_t \in \mathbb{R}^{13}$ (§4.2) with four position-side features:

$$
s_t^{\text{ext}} = \!\left(
s_t,\;
\underbrace{f_t}_{\substack{\text{profit}\\\text{fraction}}},\;
\underbrace{\dfrac{\Delta_t}{\sqrt{\hat{v}_t\,\Delta t}}}_{\substack{\text{normalised}\\\text{drift}}},\;
\underbrace{\dfrac{\sigma_{v,t}^-}{\hat{v}_t^-}}_{\substack{\text{rel. downside}\\\text{uncertainty}}},\;
\underbrace{\tau_{\text{rem},t}}_{\substack{\text{time to}\\\text{close}}}
\right) \in \mathbb{R}^{17}
\tag{6.15}
$$

The normalised drift $\Delta_t/\!\sqrt{\hat{v}_t\,\Delta t}$ measures how many **instantaneous
vol units** the price has moved since entry — precisely the quantity that drives $\eta$ in the
classical rule (6.10). By exposing it as a feature, the RL agent can learn a generalised,
data-driven analogue of $\eta$ without committing to an exponential functional form.

#### Trailing Action Head

Add a fourth policy head to the factored action space of §4.3:

$$
\log\pi_\theta(a_t \mid s_t^{\text{ext}}) =
\log\pi^{\text{dir}} + \log\pi^{\text{type}} + \log\pi^{\text{risk}} + \log\pi^{\text{trail}}
$$

The **trailing head** outputs a tightening action:

$$
\pi^{\text{trail}}\;:\; h_t \;\mapsto\; \Delta\delta^-_t \in \{0,\;{-1},\;{-3},\;{-10},\;-\infty\}
$$

corresponding to \{hold SL, tighten 1 tick, tighten 3 ticks, tighten 10 ticks, exit now\}.
The SL can only move inward; the hard constraint $\delta^-_t \ge \delta^-_{\min}$ is enforced
in the environment step.

A continuous variant parameterises the SL directly as a learnable fraction of the vol-optimal
distance, so the policy learns to deviate from the classical baseline rather than learning
absolute tick magnitudes:

$$
\delta^{-}_t = \sigma\!\left(u_t^{\text{trail}}\right)\cdot z^*\!\sqrt{\hat{v}_t^-\,\Delta t}, \qquad
u_t^{\text{trail}} = W_{\text{trail}}\,h_t + b_{\text{trail}},\quad \sigma(\cdot)\in(0,1]
\tag{6.16}
$$

Here $\sigma(u) = e^u/(1+e^u)$ ensures the output is always a contraction of the classical rule.

#### Reward and Value Objective

The reward for trailing-stop management includes a penalty for premature stop-outs:

$$
r_t^{\text{trail}} =
\Delta\text{PnL}_t^{\text{realised}}
- \lambda_c\,|\Delta q_t|
- \mu_{\text{ov}}\,\mathbf{1}[\text{overnight}]
- \lambda_t\,\mathbf{1}[\text{stopped out with }f_t < f_{\min}]
\tag{6.17}
$$

The last term prevents the agent from excessively tightening the SL when it has barely moved
into profit, a common failure mode of aggressive trailing in low-vol regimes.

The value function learns to balance three competing pressures:

1. **Lock in profits** when $f_t$ is large (tighten SL aggressively)
2. **Give the trade room** when posterior uncertainty $\sigma_{v,t}^-/\hat{v}_t^-$ is high
3. **Avoid premature exit** before the position has earned its minimum profit fraction

Point 2 is structurally invisible to the classical rule (6.12), which uses $\eta$ based on
$f_t$ only. The RL agent learns the interaction: when the inactive-leg uncertainty is elevated,
a wider SL is warranted even at high $f_t$.

#### Warm-Starting the Trailing Head

Mirroring the BC warm-start in §4.5, the trailing head is initialised by supervised imitation
of the classical rule (6.12):

$$
\mathcal{L}_{\text{BC}}^{\text{trail}}(\theta) =
-\mathbb{E}_{(s_t^{\text{ext}}, \delta^{-\text{clsscl}}_t)}\!
\left[\log\pi^{\text{trail}}_\theta\!\left(\text{nearest discrete action}\,\big|\,s_t^{\text{ext}}\right)\right]
$$

This gives the agent a solid baseline from day one and constrains early PPO exploration to
meaningful deviations from the analytically justified rule.

---

### 6.5 Classical vs. RL: Full Comparison

#### Equation Summary

| Quantity | Formula |
|---|---|
| Optimal TP (long, mean-field) | $\delta^{+*} = z^*\!\sqrt{\hat{v}_t^+\,\Delta t}\!\left[1 + \tfrac{z^{*2}-1}{4}\tfrac{(\sigma_{v,t}^+)^2}{(\hat{v}_t^+)^2}\right]$ |
| Optimal SL (long, mean-field) | $\delta^{-*} = z^*\!\sqrt{\hat{v}_t^-\,\Delta t}$ |
| Long entry condition | $\sqrt{\hat{v}_t^+} - \sqrt{\hat{v}_t^-} > c\,/\,(2z^{*2}\phi(z^*)\sqrt{\Delta t})$ |
| Short entry condition | $\sqrt{\hat{v}_t^-} - \sqrt{\hat{v}_t^+} > c\,/\,(2z^{*2}\phi(z^*)\sqrt{\Delta t})$ |
| Base trailing SL (long) | $\text{SL}_t = \max(\text{SL}_{t-1},\; P_t - z^*\!\sqrt{\hat{v}_t^-\,\Delta t})$ |
| Drift-aggressive trailing SL | $\text{SL}_t = \max\!\bigl(\text{SL}_{t-1},\; P_t - z^*\!\sqrt{\hat{v}_t^-\,\Delta t}\cdot e^{-\alpha(f_t-f_0)^+}\bigr)$ |
| RL SL distance | $\delta^-_t = \sigma(u_t^{\text{trail}})\cdot z^*\!\sqrt{\hat{v}_t^-\,\Delta t}$ |

#### Property-by-Property Comparison

| Dimension | Classical Bayesian | RL |
|---|---|---|
| **TP/SL expression** | Closed-form (6.2)–(6.3) from EV maximisation | Learned fractions (6.16) anchored to vol-optimal distance |
| **Entry threshold** | Fixed $\varepsilon_{\text{entry}}$ from breakeven (6.6); optionally inflated by (6.8) | Policy head $\pi^{\text{dir}}$ over full 13-dim state; threshold implicit in learned weights |
| **Posterior uncertainty** | Second-order correction (6.3); inflated entry threshold (6.8) | Direct feature $\sigma_{v,t}^\pm / \hat{v}_t^\pm$; agent learns non-linear response |
| **Trailing schedule** | Fixed $\eta(f_t)$ with parameters $\alpha^{\text{trail}}, f_0$ (2 hyperparams) | Trained from reward; adapts schedule per regime automatically |
| **Drift-aggressive tightening** | Triggered above fixed $f_0$ via exponential $\eta$ | Triggered by normalised-drift feature $\Delta_t/\!\sqrt{\hat{v}_t\,\Delta t}$ |
| **Inactive-leg stale vol** | Hard exit rule (6.14) when $\sigma_{v,t}^- > \sigma_{\text{exit}}$ | Soft: relative uncertainty feature $\sigma_{v,t}^-/\hat{v}_t^-$ in state; learned exit |
| **Time-of-day** | Fixed session open/close rules | $\tau_{\text{rem},t}$ in state; non-uniform schedule learned from data |
| **Regime transitions** | Piecewise threshold rules; discontinuous | Smooth policy interpolation across states |
| **Interpretability** | Full — every rule has closed-form EV or posterior justification | Opaque; inspectable via SHAP on $s_t^{\text{ext}}$ |
| **Hyperparameters** | 4–6 ($\alpha^{\text{trail}}, f_0, \eta_{\min}, f_{\min}, \sigma_{\text{exit}}, \beta$) | Many (network weights), but most absorbed by BC warm-start |
| **Sample efficiency** | High — calibrated analytically from APF outputs | Low — requires $O(10^5)$–$O(10^6)$ environment steps for fine-tuning |
| **Best use case** | Cold-start, limited data, production without re-training | Post BC warm-start on large dataset; regime-rich historical record |

#### Key Insight: Why the RL Agent Adds Value

The classical rule (6.12) conditions the tightening schedule only on $f_t$ (profit fraction).
The RL agent additionally conditions on $\sigma_{v,t}^-/\hat{v}_t^-$ (downside vol
uncertainty) and $\tau_{\text{rem},t}$ (time to session close), enabling two behaviours the
classical rule cannot represent:

1. **Wide SL at high $f_t$ when downside vol is uncertain.** After a long run of up-bars,
   $v^-$ has not been observed recently and $\sigma_{v,t}^-$ is elevated. The classical rule
   would tighten aggressively because $f_t$ is large. The RL agent learns that this is
   precisely when a sudden reversal may be vol-amplified — and holds a wider SL.

2. **Aggressive exit near session close.** With $\tau_{\text{rem},t}$ small, overnight
   penalty $\mu_{\text{ov}}$ dominates. The agent learns to exit all open positions on a
   tighter schedule than the vol-optimal SL would mandate, absorbing a small EV loss to
   avoid the overnight penalty.

These deviations are structurally inaccessible to the classical rule without adding more
hand-crafted conditions. The RL agent discovers them automatically from the reward signal.

---

### 6.6 Multi-Bar Hold Window, Timeout, and Re-Entry Logic

#### Clarification: 1-Bar vs. Intrabar vs. Multi-Bar

The reflection-principle formula

$$
p_{\text{TP/SL}}(\delta) = 2\,\Phi\!\left(\frac{-\delta}{\sqrt{v\,\Delta t}}\right)
$$

is a **first-passage probability**, not an end-of-bar comparison. It answers: "given that the
price diffuses with variance $v\,\Delta t$ over interval $\Delta t$, what is the probability
the path crosses $\pm\delta$ at any point during that interval?"

When $\Delta t$ is a **1-minute bar**, this is evaluated from OpenHigh/Low/Close data: the TP
is deemed touched if $H_t \ge \text{TP price}$ and the SL if $L_t \le \text{SL price}$,
regardless of where the bar closes. So the 1-bar formula is already intrabar in the OHLC sense.

For a **$K$-bar hold window** (successive checks each bar, up to $K$ bars $= K$ minutes), the
position remains open as long as neither barrier is struck. This is a sequential first-passage
problem with horizon $T = K\,\Delta t$.

---

#### Active-Leg Touch Probabilities Over $K$ Bars

For a long position, the TP is driven by $v^+$ and the SL by $v^-$.
Define the per-direction running diffusion widths after $K$ bars:

$$
\sigma_K^+ \;\equiv\; \sqrt{\hat{v}_t^+\,K\,\Delta t}, \qquad
\sigma_K^- \;\equiv\; \sqrt{\hat{v}_t^-\,K\,\Delta t}
\tag{6.18}
$$

Under the assumption that $\hat{v}_t^\pm$ remain approximately constant over $K \le 15$ bars
(reasonable since CIR mean-reversion over 15 minutes at rates $\kappa \sim 5$–$20$ per year
is negligible), the leading-order single-barrier touch probabilities are:

$$
p_{\text{TP}}^{(K)}(\delta^+) = 2\,\Phi\!\left(\frac{-\delta^+}{\sigma_K^+}\right), \qquad
p_{\text{SL}}^{(K)}(\delta^-) = 2\,\Phi\!\left(\frac{-\delta^-}{\sigma_K^-}\right)
\tag{6.19}
$$

These are **identical in form** to the 1-bar formula with $\Delta t \to K\,\Delta t$.
A two-barrier image correction (first reflection) keeps the probabilities consistent:

$$
p_{\text{TP}}^{(K)} \approx 2\,\Phi\!\left(\frac{-\delta^+}{\sigma_K^+}\right)
- 2\,\Phi\!\left(\frac{-(\delta^+ + 2\delta^-)}{\sigma_K}\right), \quad
\sigma_K \equiv \tfrac{1}{2}(\sigma_K^+ + \sigma_K^-)
\tag{6.20}
$$

The second term is the probability that the path overshoots the SL first and then artificially
reaches the TP — small when $(\delta^+ + 2\delta^-)/\sigma_K \gg 1$. At the optimal sizing
$z^* \approx 0.76$, $(\delta^{+*} + 2\delta^{-*})/\sigma_K \approx 3z^* \approx 2.3$,
so the correction is roughly $2\Phi(-2.3) \approx 2\%$ — negligible for most purposes.

**Particle-weighted version** (full posterior):

$$
p_{\text{TP}}^{(K)} = 2\sum_i w_t^{(i)}\,\Phi\!\left(\frac{-\delta^+}{\sqrt{v_t^{+,(i)}\,K\,\Delta t}}\right)
\tag{6.21}
$$

---

#### Three-Outcome EV Under Finite Horizon $K$

With a hard timeout at bar $K$, there are three mutually exclusive outcomes for a long:

| Outcome | Probability | P&L |
|---|---|---|
| TP touched (at any bar $k \le K$) | $p_{\text{TP}}^{(K)}$ | $+\delta^+$ |
| SL touched (at any bar $k \le K$) | $p_{\text{SL}}^{(K)}$ | $-\delta^-$ |
| Neither touched — timeout | $p_{\text{to}}^{(K)} = 1 - p_{\text{TP}}^{(K)} - p_{\text{SL}}^{(K)}$ | $\Delta_{t_0+K}$ (mark at close) |

The expected P&L on timeout, conditioning on the process staying within $(-\delta^-, +\delta^+)$
through bar $K$:

$$
\mathbb{E}\!\left[\Delta_{t_0+K} \;\Big|\; \text{no touch}\right]
= \tilde{\mu}\,K\,\Delta t + 0 + O\!\left(\frac{\sigma_K^2}{\delta^+{+}\delta^-}\right)
\tag{6.22}
$$

where $\tilde{\mu} \approx \mu K\Delta t$ is the drift term, and the $O(\cdot)$ correction accounts
for the asymmetric barrier reflecting the process toward zero. On 1-minute bars
$\mu K\Delta t \ll \delta^\pm$ in most equity applications, so the timeout expected P&L is
approximately zero.

The full EV becomes:

$$
\text{EV}^{(K)}_{\text{long}}(\delta^+, \delta^-)
= \delta^+\,p_{\text{TP}}^{(K)} - \delta^-\,p_{\text{SL}}^{(K)}
+ \mu\,K\,\Delta t\,\cdot p_{\text{to}}^{(K)} - c
\tag{6.23}
$$

For practical purposes with $\mu \approx 0$ on short intraday horizons, equation (6.23) reduces
to the same form as the 1-bar EV (§6.1), with $K\Delta t$ substituting $\Delta t$:

$$
\text{EV}^{(K)}_{\text{long}} \approx \delta^+\,p_{\text{TP}}^{(K)} - \delta^-\,p_{\text{SL}}^{(K)} - c
$$

---

#### Optimal Sizing Under Finite Horizon $K$

The FOC is identical to (6.1), and the mean-field solution is:

$$
\boxed{\delta^{\pm*}(K) = z^*\sqrt{\hat{v}_t^\pm\,K\,\Delta t}}
\tag{6.24}
$$

TP and SL **scale as $\sqrt{K}$** with the hold horizon. The optimal EV at the optimum:

$$
\text{EV}^{*}(K) = 2z^{*2}\phi(z^*)\sqrt{K\,\Delta t}\,\bigl(\sqrt{\hat{v}_t^+} - \sqrt{\hat{v}_t^-}\bigr) - c
\tag{6.25}
$$

also grows as $\sqrt{K}$, so **a wider hold window amplifies the edge in vol asymmetry**.
The minimum viable hold that breaks even is:

$$
K_{\min} = \frac{c^2}{4z^{*4}\phi^2(z^*)\,(\sqrt{\hat{v}_t^+} - \sqrt{\hat{v}_t^-})^2\,\Delta t}
\tag{6.26}
$$

For $K \ge K_{\min}$ the expected return is positive; for $K < K_{\min}$ the barrier is too
narrow to recoup costs. This gives a **hold-time lower bound driven purely by APF posteriors
and transaction cost**.

**Vol uncertainty correction (2nd order in $K$).** Over $K$ bars, the posterior
$\hat{v}_t^\pm$ drifts as the CIR reverts to $\theta^\pm$. The holding vol below uses the
mean-reverting expected path:

$$
\bar{v}^{\pm}(K) = \theta^\pm + (\hat{v}_t^\pm - \theta^\pm)\,\frac{1-(1-\kappa^\pm\Delta t)^K}{K\kappa^\pm\Delta t}
\tag{6.27}
$$

Replacing $\hat{v}_t^\pm$ with $\bar{v}^\pm(K)$ in (6.24) gives vol-drift-corrected sizing:

$$
\delta^{\pm*}(K) = z^*\sqrt{\bar{v}^\pm(K)\,K\,\Delta t}
$$

For $K \le 15$ and typical $\kappa \le 20$/year, the correction is below 1 % and safely
ignored. It matters for $K > 30$ bars (multi-hour holds).

---

#### What Happens at Timeout?

When neither TP nor SL is touched by bar $t_0 + K$:

**Step 1 — Close the position.** Mark at the bar-$K$ close:
$$\text{PnL}_{\text{timeout}} = P_{t_0+K} - P_{t_0} - c$$

**Step 2 — APF has been running throughout.** All $K$ bars of OHLC have been fed to the
filter during the hold, so the posterior at $t_0 + K$ is fully updated and no information
is wasted. The state $s_{t_0+K}$ reflects any regime shifts that occurred during the hold.

**Step 3 — Re-evaluate the entry condition.**

$$
\text{edge}_{t_0+K} = \sqrt{\hat{v}_{t_0+K}^+} - \sqrt{\hat{v}_{t_0+K}^-}
$$

Four decisions based on the updated state:

| Condition | Action |
|---|---|
| $\text{edge} > +\varepsilon_{\text{entry}}$ and same direction | Re-enter long (regime persists) |
| $\text{edge} < -\varepsilon_{\text{entry}}$ | Enter short (regime reversed during hold) |
| $|\text{edge}| \le \varepsilon_{\text{entry}}$ | Stay flat |
| $\sigma_{v,t_0+K}^+ > \sigma_{\text{exit}}$ or $\sigma_{v,t_0+K}^- > \sigma_{\text{exit}}$ | Stay flat (posterior too wide) |

**Step 4 — Cool-down guard.** To prevent transaction cost churn from rapid consecutive
timeouts, impose a minimum wait of $\tau_{\text{cool}}$ bars before re-entry:

$$
\text{Re-entry allowed only if}\quad t - t_{\text{last exit}} \ge \tau_{\text{cool}}
\tag{6.28}
$$

A conservative choice is $\tau_{\text{cool}} = 3$–$5$ bars (3–5 minutes); below this the
filter has barely integrated new information.

---

#### Hold Duration as a Policy Variable

Instead of fixing $K$ in advance, the hold horizon can be made adaptive. Define the
**maximum expected remaining EV** at bar $k$ into the hold ($0 \le k \le K_{\max}$):

$$
\text{EV}_{\text{rem}}(k) = 2z^{*2}\phi(z^*)\sqrt{(K_{\max}-k)\,\Delta t}\,
\bigl(\sqrt{\hat{v}_{t_0+k}^+} - \sqrt{\hat{v}_{t_0+k}^-}\bigr) - c_{\text{hold}}
\tag{6.29}
$$

where $c_{\text{hold}}$ is a per-bar holding cost (bid-ask spread leakage, margin, etc.).
The classical **adaptive timeout rule**: exit at bar $k^*$ where remaining EV first falls below
the immediate exit EV (zero):

$$
k^* = \min\left\{k \;\Big|\; \text{EV}_{\text{rem}}(k) \le 0\right\}
\tag{6.30}
$$

If the vol edge decays during the hold (posterior converges as $\hat{v}^+ \to \hat{v}^-$), the
position is exited early without waiting for $K_{\max}$. If the edge strengthens (a regime
event during the hold), the position is held longer.

In the **RL framework**, $k^*$ is implicit: the trailing-head action "exit now" is selected
naturally when the remaining $\text{EV}_{\text{rem}}$ (encoded via state features
$\hat{v}_{t_0+k}^\pm$, $f_k$, $\tau_{\text{rem},k}$) falls below the re-entry value
estimated by the critic.

---

#### Summary: 1-Bar vs. K-Bar Key Differences

| Quantity | 1-Bar $(K=1)$ | K-Bar $(K > 1)$ |
|---|---|---|
| Touch probability | $2\Phi(-\delta/\sqrt{v\Delta t})$ | $2\Phi(-\delta/\sqrt{v K\Delta t})$ (first-passage) |
| Optimal TP/SL | $z^*\sqrt{\hat v^\pm \Delta t}$ | $z^*\sqrt{\hat v^\pm K\Delta t}$ |
| Sizes grow with $K$ | — | $\propto \sqrt{K}$ |
| Optimal EV | $\propto \sqrt{\Delta t}(\sqrt{v^+}-\sqrt{v^-})$ | $\propto \sqrt{K\Delta t}(\sqrt{v^+}-\sqrt{v^-})$ |
| Timeout outcome | Not applicable — bar either triggers or not | Positive probability; mark at close |
| Vol drift during hold | Negligible | Material for $K > 30$; use $\bar v^\pm(K)$ (6.27) |
| Re-entry decision | N/A | Re-evaluate at timeout via updated APF state |
| Hold-time lower bound $K_{\min}$ | N/A (single bar) | From (6.26); driven by cost/vol-edge ratio |

---

## 7. Synthetic OHLC Simulation and Model Verification

Simulation of artificial bars serves two purposes: (1) **trajectory generation** for RL environment
training without relying solely on historical data, and (2) **model fit verification** by comparing
statistical properties of simulated vs. real bars.

---

### 7.1 Innovation Structure and the Sign-Selection Problem

The core simulation challenge is that the active variance $\sigma_t^2 \in \{v_t^+, v_t^-\}$
depends on the *sign* of the Brownian increment $dW_t^S$, which in turn is driven by
$\sigma_t^2$. This is resolved by noting that $\phi_t^\pm = \mathbf{1}(\pm\epsilon_t^S > 0)$
where $\epsilon_t^S$ is a standard normal draw — the sign is determined before conditioning on
the variance level. The discretised innovation structure is:

$$
\epsilon_t^S \sim \mathcal{N}(0,1), \qquad \epsilon_t^{v+}, \epsilon_t^{v-} \sim \mathcal{N}(0,1)
\tag{7.1}
$$

with the correlated decomposition:

$$
\epsilon_t^{v+} = \rho^+\,\epsilon_t^S + \sqrt{1-(\rho^+)^2}\;\eta_t^+, \qquad
\epsilon_t^{v-} = \rho^-\,\epsilon_t^S + \sqrt{1-(\rho^-)^2}\;\eta_t^-
\tag{7.2}
$$

and the cross-process correlation $\langle\eta_t^+, \eta_t^-\rangle = \tilde\rho^{+-}$ where
$\tilde\rho^{+-}$ is chosen so that $\text{Cov}(\epsilon_t^{v+}, \epsilon_t^{v-}) = \rho^{+-}$:

$$
\tilde\rho^{+-} = \frac{\rho^{+-} - \rho^+\rho^-}{\sqrt{(1-(\rho^+)^2)(1-(\rho^-)^2)}}
\tag{7.3}
$$

The three-dimensional normal draw $(\epsilon_t^S, \eta_t^+, \eta_t^-)$ is therefore sampled from
$\mathcal{N}(\mathbf{0}, \Sigma)$ with:

$$
\Sigma = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & \tilde\rho^{+-} \\ 0 & \tilde\rho^{+-} & 1 \end{pmatrix}
\quad\Longrightarrow\quad
L = \text{Cholesky}(\Sigma), \quad \begin{pmatrix}\epsilon^S \\ \eta^+ \\ \eta^-\end{pmatrix} = L\,\mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0},I_3)
\tag{7.4}
$$

---

### 7.2 Per-Bar Simulation Algorithm

Given parameters $\boldsymbol{\psi}$, Cholesky factor $L$, and state $(v_{t-1}^+, v_{t-1}^-, P_{t-1})$:

**Step 1 — Draw correlated innovations.**

$$
(\epsilon_t^S,\; \eta_t^+,\; \eta_t^-) = L\,\mathbf{z}_t, \qquad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, I_3)
\tag{7.5}
$$

**Step 2 — Propagate both CIR processes.**

$$
v_t^+ = v_{t-1}^+ + \kappa^+(\theta^+ - v_{t-1}^+)\Delta t
+ \xi^+\sqrt{v_{t-1}^+\,\Delta t}\;\epsilon_t^{v+}, \qquad v_t^+ \leftarrow \max(v_t^+, \varepsilon_{\min})
\tag{7.6}
$$

$$
v_t^- = v_{t-1}^- + \kappa^-(\theta^- - v_{t-1}^-)\Delta t
+ \xi^-\sqrt{v_{t-1}^-\,\Delta t}\;\epsilon_t^{v-}, \qquad v_t^- \leftarrow \max(v_t^-, \varepsilon_{\min})
\tag{7.7}
$$

where $\epsilon_t^{v\pm} = \rho^\pm\,\epsilon_t^S + \sqrt{1-(\rho^\pm)^2}\;\eta_t^\pm$.

**Step 3 — Select active variance by sign of $\epsilon_t^S$.**

$$
\sigma_t^2 = \begin{cases} v_t^+ & \text{if } \epsilon_t^S > 0 \\ v_t^- & \text{if } \epsilon_t^S < 0 \end{cases}
\tag{7.8}
$$

**Step 4 — Generate the intrabar log-return.**

$$
r_t^{OC} = \left(\mu - \tfrac{1}{2}\sigma_t^2\right)\Delta t + \sqrt{\sigma_t^2\,\Delta t}\;\epsilon_t^S
\tag{7.9}
$$

**Step 5 — Advance price.**

$$
O_t = P_{t-1}, \qquad C_t = O_t\,\exp(r_t^{OC})
\tag{7.10}
$$

---

### 7.3 OHLC Construction via Brownian Bridge

Given $(O_t, C_t, \sigma_t^2)$ from Step 5, the High and Low are derived from the conditional
distribution of the running maximum and minimum of a Brownian bridge.

Let $b \equiv r_t^{OC}/(\sigma_t\sqrt{\Delta t})$ be the normalised endpoint and $B_t$ a
standard Brownian bridge on $[0, T]$ with $B_0 = 0$, $B_T = b\sqrt{T}$.

#### High (Running Maximum)

For $m \ge \max(0, b)$, the CDF of the running maximum $M = \max_{0\le s\le T} B_s/\sqrt{T}$ is:

$$
P(M \le m \mid B_T/\sqrt{T} = b) = 1 - \exp\!\left(-2m(m - b)\right)
\tag{7.11}
$$

Inversion via $U_H \sim \text{Uniform}(0,1)_{> \Phi(b)}$: solve $U_H = P(M \le m)$ for $m$:

$$
m_H = \tfrac{1}{2}\!\left[b + \sqrt{b^2 - 2\ln(1-U_H)}\right], \quad U_H \sim \text{Uniform}(0,1)
\tag{7.12}
$$

**Note:** when $b > 0$ (up-bar), $m_H \ge b$ always; when $b < 0$, $m_H \ge 0$ and a separate
draw $U_H$ below which $m_H = b$ corresponds to no new high.

$$
H_t = O_t \exp\!\bigl(\sigma_t\sqrt{\Delta t}\cdot m_H\bigr)
\tag{7.13}
$$

#### Low (Running Minimum)

Mirror symmetry: apply (7.12) with $b' = -b$ and $U_L \sim \text{Uniform}(0,1)$ to get $m_L \ge \max(0,-b)$, then

$$
L_t = O_t \exp\!\bigl(-\sigma_t\sqrt{\Delta t}\cdot m_L\bigr)
\tag{7.14}
$$

#### Constraint Enforcement

After drawing, enforce the OHLC ordering constraint:

$$
H_t = \max(H_t, O_t, C_t), \qquad L_t = \min(L_t, O_t, C_t)
\tag{7.15}
$$

This is automatically satisfied almost surely by the bridge formulae; the clamp (7.15) handles
rare floating-point edge cases.

#### Rogers-Satchell Estimator (Cross-Check)

The simulated RS variance should recover $\sigma_t^2\Delta t$ in expectation:

$$
\hat{v}_{\text{RS},t}^{\text{sim}} = \ln\!\frac{H_t}{O_t}\ln\!\frac{H_t}{C_t}
+ \ln\!\frac{L_t}{O_t}\ln\!\frac{L_t}{C_t}
\tag{7.16}
$$

$\mathbb{E}[\hat{v}_{\text{RS},t}^{\text{sim}} \mid \sigma_t^2] = \sigma_t^2\Delta t$ exactly
(drift-free estimator, independent of $\mu$). **This provides the first internal consistency
check** — if the simulation code is correct, the ensemble mean of (7.16) must equal the
ensemble mean of $\sigma_t^2\cdot\Delta t$.

---

### 7.4 Overnight Bar Handling

The overnight gap ($C_{\text{close}} \to O_{\text{open}}$, duration $\Delta t_{\text{ov}} \approx 16.5\,\text{h}$)
is longer than any intraday bar. Both CIR processes propagate under their prior for the full
overnight window; the price gap is simulated under the total variance accumulated overnight.

**Sub-stepping strategy.** To avoid CIR negativity bias over large $\Delta t_{\text{ov}}$,
divide the overnight into $n_{\text{ov}} = \lceil\Delta t_{\text{ov}}/\Delta t\rceil$ sub-steps:

$$
n_{\text{ov}} = 990 \qquad (\text{for a 16.5 h overnight at 1-min } \Delta t)
$$

At each sub-step $j = 1, \ldots, n_{\text{ov}}$, propagate both CIR processes using (7.6)–(7.7)
with $\Delta t$. The total log-return is accumulated:

$$
r_{\text{ov}} = \sum_{j=1}^{n_{\text{ov}}} r_j^{OC}
\tag{7.17}
$$

The overnight OHLC is computed on the sub-path. **Only C of the previous session and O of the
next session are observable**; H and L of the overnight sub-path are latent and discarded.

Alternatively, for speed: propagate the CIR process analytically via its exact non-central
chi-squared transition (no Euler discretisation bias) and draw a single overnight return:

$$
v_{\text{ov}} \sim p_{\text{CIR}}(v\mid v_{t_{\text{close}}}, \Delta t_{\text{ov}}, \kappa^\pm, \theta^\pm, \xi^\pm)
\tag{7.18}
$$

using the Gamma approximation $v_{\text{ov}} \approx \text{Gamma}(2\kappa\bar v / \xi^2,\; \xi^2/(2\kappa))$
where $\bar v = \theta + (v_{t_{\text{close}}} - \theta)e^{-\kappa\Delta t_{\text{ov}}}$.

---

### 7.5 Full Simulation Pseudocode

```
Input:  ψ = (v0+, v0-, κ+, κ-, θ+, θ-, ξ+, ξ-, ρ+, ρ-, ρ+-, μ, λ_ov)
        T bars, session schedule, P0 (initial price)

Precompute: L = Cholesky of Σ in (7.4), ε_min

Initialise: v+ ← v0+, v- ← v0-, P ← P0

For t = 1, ..., T:

    If t is session open (overnight gap since t-1):
        Propagate v+, v- through overnight via CIR exact draw (7.18)
        Draw r_ov ~ N((μ - σ²/2)Δt_ov, σ²·Δt_ov), σ² = (v+ + v-)/2
        P ← P · exp(r_ov)   [open price of new session]

    # Intraday bar
    z ~ N(0, I_3)
    (ε^S, η+, η-) = L · z                               # (7.5)
    ε^{v+} = ρ+ · ε^S + sqrt(1-(ρ+)²) · η+             # (7.2)
    ε^{v-} = ρ- · ε^S + sqrt(1-(ρ-)²) · η-

    v+ ← max(v+ + κ+(θ+-v+)·Δt + ξ+·sqrt(v+·Δt)·ε^{v+}, ε_min)  # (7.6)
    v- ← max(v- + κ-(θ--v-)·Δt + ξ-·sqrt(v-·Δt)·ε^{v-}, ε_min)  # (7.7)

    σ² ← v+ if ε^S > 0 else v-                          # (7.8)
    r   ← (μ - σ²/2)·Δt + sqrt(σ²·Δt)·ε^S             # (7.9)

    O ← P
    C ← O · exp(r)                                       # (7.10)
    H ← O · exp(σ·sqrt(Δt)·m_H)  via (7.12)–(7.13)
    L ← O · exp(-σ·sqrt(Δt)·m_L) via (7.14)
    Enforce OHLC ordering (7.15)

    Output bar: (O, H, L, C, v+, v-, σ²)
    P ← C
```

---

### 7.6 Verification Tests

#### Test 1 — Return Distribution Moments

Simulate $T_{\text{sim}} = 100\,000$ bars. Compare simulated vs. real first four moments per
session-hour bucket (to account for intraday seasonality):

| Statistic | Simulated | Real (target) | Pass criterion |
|---|---|---|---|
| Mean $\hat\mu_r$ | $\approx \mu\Delta t$ | Sample mean | $\|t\text{-stat}\| < 2$ |
| Variance $\hat\sigma_r^2$ | $\approx (\hat v^+\!+\!\hat v^-)/2\cdot\Delta t$ | Sample var | $<5\%$ relative error |
| Skewness | Computed from sim | Real sample skew | Sign match |
| Excess kurtosis | Computed from sim | Real sample kurt | Order of magnitude match |

#### Test 2 — Directional RS Asymmetry

On the simulated bars, compute:

$$
R_{\text{asym}} = \frac{\mathbb{E}[\hat v_{\text{RS}}^- \mid r^{OC}<0]}{\mathbb{E}[\hat v_{\text{RS}}^+ \mid r^{OC}>0]}
$$

**Expected value** (from model):

$$
R_{\text{asym}}^{\text{model}} = \frac{\hat\theta^-}{\hat\theta^+}
$$

at stationarity, since $\mathbb{E}[v_t^\pm] = \theta^\pm$.
The simulated ratio must match this to within Monte Carlo noise:

$$
\left| R_{\text{asym}}^{\text{sim}} - \frac{\theta^-}{\theta^+} \right| < 3\,\frac{\sigma_{\text{MC}}}{\sqrt{T_{\text{sim}}}}
\tag{7.19}
$$

This is the **primary diagnostic**: if it fails, the sign-selection step (7.8) is bugged.

#### Test 3 — RS Unbiasedness

For each bar, compute $\hat v_{\text{RS},t}^{\text{sim}}$ using (7.16). Regress it on the true
simulated $\sigma_t^2$:

$$
\hat v_{\text{RS},t}^{\text{sim}} = a + b\,\sigma_t^2\,\Delta t + \varepsilon_t
$$

**Expected:** $a = 0$, $b = 1$. Any significant deviation indicates a OHLC-construction bug in
the Brownian bridge sampling (7.12)–(7.14).

#### Test 4 — CIR Marginal Distribution Check

Collect the full simulated path $\{v_t^+\}_{t=1}^T$. The stationary distribution is:

$$
v^\pm \sim \text{Gamma}\!\left(\frac{2\kappa^\pm\theta^\pm}{(\xi^\pm)^2},\; \frac{(\xi^\pm)^2}{2\kappa^\pm}\right)
\tag{7.20}
$$

Apply a Kolmogorov-Smirnov or chi-squared goodness-of-fit test comparing the empirical CDF of
$\{v_t^+\}$ against the theoretical Gamma CDF. **Pass criterion:** $p$-value $> 0.05$.

#### Test 5 — ACF of Squared Returns (Volatility Clustering)

The squared return series $\{(r_t^{OC})^2\}$ from a Heston-type model should show significant
positive autocorrelation at lags 1–30. Compute:

$$
\text{ACF}_{k}^{\text{sim}} = \text{Corr}\!\left[(r_t^{OC})^2, (r_{t-k}^{OC})^2\right]
$$

Compare the shape (decay rate) against the real data ACF. The CIR half-life $\ln 2 / \kappa^\pm$
should match the empirical lag at which $\text{ACF}_k$ drops to half its lag-1 value. This is
**the primary check that $\kappa^\pm$ are correctly calibrated**.

#### Test 6 — Leverage Effect Check

Compute the **realised leverage** from the simulated path:

$$
\rho_{\text{realised}}^\pm \approx \text{Corr}\!\left[r_t^{OC},\; v_{t+1}^\pm - v_t^\pm\right]
$$

separately on up-bars and down-bars. This should recover $\rho^+$ and $\rho^-$ respectively.
Significant deviation indicates a bug in the innovation coupling (7.2).

#### Test 7 — Parameter Recovery (Round-Trip Test)

**This is the most comprehensive verification.** It confirms that the APF + CMA-ES calibration
pipeline is consistent with the model:

1. Simulate $T_{\text{sim}} = 50\,000$ bars with known $\boldsymbol{\psi}_{\text{true}}$.
2. Run the APF + CMA-ES calibration on the simulated OHLC.
3. Compare recovered $\hat{\boldsymbol{\psi}}$ against $\boldsymbol{\psi}_{\text{true}}$.

**Pass criterion** per parameter $\psi_j$:

$$
\left|\hat\psi_j - \psi_j^{\text{true}}\right| < 2\,\text{SE}_j
\tag{7.21}
$$

where $\text{SE}_j$ is the numerical standard error from the CMA-ES covariance estimate.
Parameters that consistently fail recovery (large bias) indicate either model unidentifiability
or APF numerical issues.

**Typical identifiability ranking** (easiest → hardest to recover):

| Parameter | Expected recovery | Reason |
|---|---|---|
| $\theta^\pm$ | Easy | Constrained by sample mean of $\hat v_{\text{RS}}^\pm$ |
| $\kappa^\pm$ | Moderate | ACF shape of $\hat v_{\text{RS}}^\pm$ series |
| $\xi^\pm$ | Moderate | Variance of $\hat v_{\text{RS}}^\pm$ |
| $\rho^\pm$ | Harder | Requires cross-correlation $r_t \times \Delta v_{t+1}$ |
| $\rho^{+-}$ | Hardest | Cross-correlation of two latent processes, rarely updates simultaneously |
| $v_0^\pm$ | Trivially | Dominated by initialisation; converges in first ~100 bars |
| $\mu$ | Hard | Drift dwarfed by variance over short horizons; weakly identified |

---

### 7.7 Simulation Diagnostics Plot Plan

Four-panel figure per simulation run:

**Panel 1 — Price trajectory with vol overlay.**
Plot $P_t$ (black), $\hat v_t^+$ (green), $\hat v_t^-$ (red) on the same axis with normalised
secondary $y$-axis. Visual check: are up-regime transitions correlated with elevated $v^+$?

**Panel 2 — Directional RS scatter.**
Scatter plot: $x$-axis = bar index or $v_t^+$ (up-bars) / $v_t^-$ (down-bars), $y$-axis = $\hat v_{\text{RS},t}$.
Should lie along the diagonal $\hat v_{\text{RS}} = \sigma_t^2\Delta t$.

**Panel 3 — ACF of squaredreturns.**
Bar chart of $\text{ACF}_k$ at lags $k=1,\ldots,60$, with theoretical CIR-implied ACF overlay:

$$
\text{ACF}_k^{\text{CIR}} \approx A\,e^{-\bar\kappa k\Delta t}, \qquad
\bar\kappa = \tfrac{1}{2}(\kappa^+ + \kappa^-)
\tag{7.22}
$$

**Panel 4 — QQ plot of $r_t^{OC}$.**
Standardised return $(r_t^{OC} - \mu\Delta t)/(\sigma_t\sqrt{\Delta t})$ vs. standard normal
quantiles. Under the model this should be exactly $\mathcal{N}(0,1)$ when conditioning on
the true $\sigma_t^2$. With estimated $\hat\sigma_t^2$ from the APF there will be a modest
fat-tail residual; if the tails are heavier than a $t_5$ distribution, the model is
under-dispersed.

---

## 8. Auxiliary Particle Filter — Equations and Per-Step Algorithm

This section gives a self-contained mathematical account of every step executed inside
`SemivarianceHestonProcess.loglikelihood`. The implementation runs $P$ CMA-ES candidates
in parallel; each candidate owns $N$ particles. All shapes below are quoted for a single
candidate ($P = 1$).

---

### 8.1 State, Noise, and Identification Constraints

**Particle state.** Each of the $N$ particles carries the bivariate latent variance:

$$
\mathbf{x}_t^{(i)} = \bigl(v_t^{+,(i)},\; v_t^{-,(i)}\bigr) \in \mathbb{R}_{>0}^2, \qquad i = 1,\ldots,N
$$

**Pre-generated Common Random Numbers (CRNs).** All randomness is drawn once before the scan
and stored in `noises[t]` $\in \mathbb{R}^{2N+2}$:

| Columns | Symbol | Distribution | Role |
|---|---|---|---|
| $0,\ldots,N{-}1$ | $\varepsilon_{vp,t}^{(i)}$ | $\mathcal{N}(0,1)$ | $v^+$ CIR innovation |
| $N,\ldots,2N{-}1$ | $\eta_{vm,t}^{(i)}$ | $\mathcal{N}(0,1)$ | orthogonal component for $v^-$ |
| $2N$ | $u_{1,t}$ | $\text{Uniform}[0,1)$ | first-stage systematic resample |
| $2N{+}1$ | $u_{2,t}$ | $\text{Uniform}[0,1)$ | second-stage systematic resample |

**Identification constraint.** Before the filter runs, the parameter vector is sorted to enforce
$\theta^- \geq \theta^+$ (downside long-run variance $\geq$ upside long-run variance):

$$
\theta^+ \leftarrow \min(\theta^+, \theta^-), \qquad \theta^- \leftarrow \max(\theta^+, \theta^-)
$$

This breaks the discrete label-swap symmetry so the optimizer has a unique mode.

**Effective time step.** For intraday bars, $\Delta t_{\text{eff}} = \Delta t = 1\,\text{min}$.
For the overnight step ($\Delta t_{\text{raw}} = 1050\,\text{min}$):

$$
n_{\text{sub}} = \max\!\left(\text{round}(1050 \cdot \lambda_{ov}),\, 1\right), \qquad
\Delta t_{\text{eff}} = n_{\text{sub}} \cdot \Delta t
$$

This sub-steps the overnight CIR propagation using a single pre-drawn innovation while keeping
the effective vol-of-vol per sub-step at scale $\sqrt{\Delta t}$.

---

### 8.2 Initialization

$$
v_0^{+,(i)} = v_0^+, \qquad v_0^{-,(i)} = v_0^-, \qquad i = 1,\ldots,N
$$

All particles start at the point mass $v_0^\pm$; the filter spreads them out as observations
arrive.

---

### 8.3 Per-Step Algorithm (scanned over $t = 1,\ldots,T$)

Let $r_t = \log(S_t / S_{t-1})$ be the close-to-close log-return and let
$\mathbb{1}_t^+ = [r_t \geq 0]$ be the up-bar indicator.

---

#### Step 1 — First-Stage Pilot Weights

Compute a **deterministic pilot propagation** (drift only, no stochastic term) for each
particle to obtain a predictive variance:

$$
\tilde{v}^{+,(i)} = \max\!\Bigl(v_{t-1}^{+,(i)} + \kappa^+\!\left(\theta^+ - v_{t-1}^{+,(i)}\right)\Delta t_{\text{eff}},\;\varepsilon_{\min}\Bigr)
$$

$$
\tilde{v}^{-,(i)} = \max\!\Bigl(v_{t-1}^{-,(i)} + \kappa^-\!\left(\theta^- - v_{t-1}^{-,(i)}\right)\Delta t_{\text{eff}},\;\varepsilon_{\min}\Bigr)
$$

Select the active leg by bar direction and evaluate the Gaussian pilot log-weight:

$$
\tilde{\sigma}_t^{2,(i)} = \begin{cases} \tilde{v}^{+,(i)} & \text{if } \mathbb{1}_t^+ \\ \tilde{v}^{-,(i)} & \text{otherwise} \end{cases}
$$

$$
\log g^{(i)} = \log \mathcal{N}\!\left(r_t \;\Big|\; \left(\mu - \tfrac{1}{2}\tilde{\sigma}_t^{2,(i)}\right)\Delta t_{\text{eff}},\;\tilde{\sigma}_t^{2,(i)}\Delta t_{\text{eff}}\right)
$$

---

#### Step 2 — First-Stage Log-Normalizer and Resample

$$
\log Z_1 = \log\!\sum_{i=1}^N \exp\!\left(\log g^{(i)}\right) - \log N
$$

Draw ancestor indices $\{a^{(i)}\}_{i=1}^N$ by **systematic resampling** with scalar seed
$u_{1,t}$ and unnormalized log-weights $\{\log g^{(i)}\}$:

$$
v_{\text{sel}}^{+,(i)} = v_{t-1}^{+,\,a^{(i)}}, \qquad
v_{\text{sel}}^{-,(i)} = v_{t-1}^{-,\,a^{(i)}}, \qquad
\log g_{\text{sel}}^{(i)} = \log g^{a^{(i)}}
$$

---

#### Step 3 — Correlated Innovation Decomposition

Enforce $\text{Cov}(\varepsilon_{vp}, \varepsilon_{vm}) = \rho_{pm}$ without a Cholesky factor
by deriving $\varepsilon_{vm}$ from the two pre-drawn independent normals:

$$
\varepsilon_{vm,t}^{(i)} = \rho_{pm}\,\varepsilon_{vp,t}^{(i)}
+ \sqrt{\max\!\left(1 - \rho_{pm}^2,\,\varepsilon_{\min}\right)}\cdot\eta_{vm,t}^{(i)}
$$

---

#### Step 4 — Propagate Both CIR Processes

Apply the full Euler–Maruyama update to both variance legs simultaneously:

$$
v_t^{+,(i)} = \max\!\left(
v_{\text{sel}}^{+,(i)}
+ \kappa^+\!\left(\theta^+ - v_{\text{sel}}^{+,(i)}\right)\Delta t_{\text{eff}}
+ \xi^+\sqrt{\max\!\left(v_{\text{sel}}^{+,(i)},\,\varepsilon_{\min}\right)\Delta t_{\text{eff}}}\;\varepsilon_{vp,t}^{(i)},\;
\varepsilon_{\min}
\right)
$$

$$
v_t^{-,(i)} = \max\!\left(
v_{\text{sel}}^{-,(i)}
+ \kappa^-\!\left(\theta^- - v_{\text{sel}}^{-,(i)}\right)\Delta t_{\text{eff}}
+ \xi^-\sqrt{\max\!\left(v_{\text{sel}}^{-,(i)},\,\varepsilon_{\min}\right)\Delta t_{\text{eff}}}\;\varepsilon_{vm,t}^{(i)},\;
\varepsilon_{\min}
\right)
$$

The inactive leg (the one not selected by $\mathbb{1}_t^+$) propagates under its pure prior —
its uncertainty accumulates between directional observations.

---

#### Step 5 — Second-Stage Correction Weights

The price-vol correlation ($\rho^+$ or $\rho^-$) enters here. Conditioning on the drawn
innovation $\varepsilon_{vp}$ (or $\varepsilon_{vm}$) for the active process, the return
distribution becomes a Gaussian with correlation-adjusted mean and variance:

**Up-bar** ($\mathbb{1}_t^+ = 1$, active process $v^+$, innovation $\varepsilon_{vp,t}^{(i)}$):

$$
\mu_t^{+,(i)} = \left(\mu - \tfrac{1}{2}v_t^{+,(i)}\right)\Delta t_{\text{eff}}
+ \rho^+\sqrt{\max\!\left(v_t^{+,(i)}\Delta t_{\text{eff}},\,\varepsilon_{\min}\right)}\;\varepsilon_{vp,t}^{(i)}
$$

$$
s_t^{+2,(i)} = \max\!\left(v_t^{+,(i)}\left(1 - (\rho^+)^2\right)\Delta t_{\text{eff}},\;\varepsilon_{\min}\right)
$$

$$
\log p^{(i)} = \log \mathcal{N}\!\left(r_t \;\Big|\; \mu_t^{+,(i)},\; s_t^{+2,(i)}\right)
$$

**Down-bar** ($\mathbb{1}_t^+ = 0$, active process $v^-$, innovation $\varepsilon_{vm,t}^{(i)}$):

$$
\mu_t^{-,(i)} = \left(\mu - \tfrac{1}{2}v_t^{-,(i)}\right)\Delta t_{\text{eff}}
+ \rho^-\sqrt{\max\!\left(v_t^{-,(i)}\Delta t_{\text{eff}},\,\varepsilon_{\min}\right)}\;\varepsilon_{vm,t}^{(i)}
$$

$$
s_t^{-2,(i)} = \max\!\left(v_t^{-,(i)}\left(1 - (\rho^-)^2\right)\Delta t_{\text{eff}},\;\varepsilon_{\min}\right)
$$

$$
\log p^{(i)} = \log \mathcal{N}\!\left(r_t \;\Big|\; \mu_t^{-,(i)},\; s_t^{-2,(i)}\right)
$$

The log correction ratio subtracts the pilot log-weight used to pre-select that particle:

$$
\log \alpha^{(i)} = \log p^{(i)} - \log g_{\text{sel}}^{(i)}
$$

---

#### Step 6 — Second-Stage Log-Normalizer and Log-Likelihood Increment

$$
\log Z_2 = \log\!\sum_{i=1}^N \exp\!\left(\log \alpha^{(i)}\right) - \log N
$$

$$
\boxed{\Delta \log \mathcal{L}_t = \log Z_1 + \log Z_2}
$$

The total APF log-likelihood is accumulated: $\log \mathcal{L} = \sum_{t=1}^T \Delta \log \mathcal{L}_t$.

---

#### Step 7 — Posterior Summaries

Normalize the correction weights:

$$
w_t^{(i)} = \frac{\exp\!\left(\log \alpha^{(i)}\right)}{\sum_j \exp\!\left(\log \alpha^{(j)}\right)}
$$

Extract posterior means, standard deviations, and effective sample size:

$$
\hat{v}_t^+ = \sum_i w_t^{(i)} v_t^{+,(i)}, \qquad
\hat{\sigma}_{v,t}^+ = \sqrt{\sum_i w_t^{(i)}\!\left(v_t^{+,(i)} - \hat{v}_t^+\right)^2}
$$

$$
\hat{v}_t^- = \sum_i w_t^{(i)} v_t^{-,(i)}, \qquad
\hat{\sigma}_{v,t}^- = \sqrt{\sum_i w_t^{(i)}\!\left(v_t^{-,(i)} - \hat{v}_t^-\right)^2}
$$

$$
\text{ESS}_t = \exp\!\left(-\log\!\sum_i \left(w_t^{(i)}\right)^2\right)
$$

---

#### Step 8 — Second Resample

Draw ancestor indices $\{b^{(i)}\}$ by systematic resampling using seed $u_{2,t}$ and
log-weights $\{\log \alpha^{(i)}\}$:

$$
v_t^{+,(i)} \leftarrow v_t^{+,\,b^{(i)}}, \qquad v_t^{-,(i)} \leftarrow v_t^{-,\,b^{(i)}}
$$

These resampled particles form the initial state for step $t+1$.

---

#### Step 9 — One-Step-Ahead Predictive Statistics

Apply another deterministic pilot propagation to the resampled particles to form the
predictive distribution for $r_{t+1}$:

$$
\hat{v}^{+,(i)}_{\text{pred}} = \max\!\left(v_t^{+,(i)} + \kappa^+\!\left(\theta^+ - v_t^{+,(i)}\right)\Delta t_{\text{eff}},\;\varepsilon_{\min}\right)
$$

$$
\hat{v}^{-,(i)}_{\text{pred}} = \max\!\left(v_t^{-,(i)} + \kappa^-\!\left(\theta^- - v_t^{-,(i)}\right)\Delta t_{\text{eff}},\;\varepsilon_{\min}\right)
$$

The active predicted variance $\hat{\sigma}^{2,(i)}_{\text{pred}}$ is selected by $\mathbb{1}_t^+$.
Predictive mean and std are:

$$
\bar{\mu}_{t+1} = \frac{1}{N}\sum_i \left(\mu - \tfrac{1}{2}\hat{\sigma}^{2,(i)}_{\text{pred}}\right)\Delta t_{\text{eff}}
$$

$$
\bar{\sigma}_{t+1} = \sqrt{
\frac{1}{N}\sum_i \hat{\sigma}^{2,(i)}_{\text{pred}}\,\Delta t_{\text{eff}}
+ \frac{1}{N}\sum_i \left(\hat{\mu}^{(i)}_{\text{pred}} - \bar{\mu}_{t+1}\right)^2
}
$$

where the second term is the between-particle variance of the predictive mean (law of total
variance).

---

### 8.4 Summary Table

| Step | Operation | Key equation | Output |
|---|---|---|---|
| 1 | Pilot propagation + log-weight | $\log g^{(i)} = \log\mathcal{N}(r_t \mid \tilde{\mu}^{(i)}, \tilde{\sigma}^{2,(i)}\Delta t_{\text{eff}})$ | $\log g^{(i)}$, shape $(N,)$ |
| 2 | First resample | systematic, seed $u_{1,t}$ | ancestors $a^{(i)}$; $\log Z_1$ |
| 3 | Innovation coupling | $\varepsilon_{vm} = \rho_{pm}\varepsilon_{vp} + \sqrt{1-\rho_{pm}^2}\,\eta_{vm}$ | $\varepsilon_{vm}^{(i)}$ |
| 4 | CIR propagation | Euler–Maruyama for both $v^+$ and $v^-$ | $v_t^{+,(i)},\,v_t^{-,(i)}$ |
| 5 | Correction log-weights | $\log\alpha^{(i)} = \log p^{(i)} - \log g^{a^{(i)}}$ | $\log\alpha^{(i)}$ |
| 6 | Log-likelihood increment | $\Delta\log\mathcal{L}_t = \log Z_1 + \log Z_2$ | scalar accumulated |
| 7 | Posterior summaries | weighted mean/std, ESS | $\hat{v}^\pm_t$, $\text{ESS}_t$ |
| 8 | Second resample | systematic, seed $u_{2,t}$ | refreshed particles |
| 9 | Predictive stats | law of total variance | $\bar{\mu}_{t+1}$, $\bar{\sigma}_{t+1}$ |

### 8.5 Log-Likelihood Decomposition

The APF factorises $\log p(r_{1:T} \mid \boldsymbol{\psi})$ as:

$$
\log \mathcal{L}(\boldsymbol{\psi}) = \sum_{t=1}^T \underbrace{\log Z_{1,t}}_{\text{pilot normalizer}} + \underbrace{\log Z_{2,t}}_{\text{correction normalizer}}
$$

$\log Z_1$ measures how well the deterministic pilot predicts the observation direction and
magnitude. $\log Z_2$ measures the extra likelihood gained from the stochastic propagation
after conditioning on the drawn innovations, including the price-vol correlation adjustment.
Monitoring the ratio $\log Z_2 / \log Z_1$ across training can diagnose whether the
price-vol correlations $\rho^+, \rho^-$ are contributing information beyond the pilot
(if $\log Z_2 \approx 0$ consistently, the pilot is near-perfect and $\rho^\pm$ may be
weakly identified).
