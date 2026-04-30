# PPO Design for Intraday Volatility Scalping Market Making

## 1. Objective and Scope

This document specifies the training and inference design for a PPO agent that scalp-trades intraday volatility by continuously placing bid and ask limit orders around the current close price. The implementation target is the [rl/walkthrough.ipynb](/home/metis/Projects/SV-MM/rl/walkthrough.ipynb) JAX/Flax style: fold-local environment parameter containers are transferred to GPU once per walk-forward fold, and grouped multi-environment rollout, minibatching, PPO updates, and inference are executed on device with `jit`, `vmap`, and `lax.scan`.

The design below is intentionally implementation-free. It defines the environment, the feature pipeline, the PPO loss functions, the walk-forward schedule, and the metrics/logging contract needed before coding.

## 2. Data Contract

### 2.1 Source bars

Bars are loaded from the database API over a user-selected root and date range with 1-minute frequency. Each bar contains:

$$
(t_i, O_i, H_i, L_i, C_i, V_i, AP_i, N_i)
$$

where:

- $t_i$ is the beginning timestamp of the minute bar
- $O_i, H_i, L_i, C_i$ are open, high, low, close
- $V_i$ is volume
- $AP_i$ is average price
- $N_i$ is trade count

The available date range is from $20200101$ to $20251231$.

### 2.2 Daily segmentation

Let day $d$ contain $n_d$ bars indexed by $k = 0, 1, \dots, n_d - 1$. The normalized time-of-day feature is

$$
\tau_{d,k} =
\begin{cases}
0 & n_d = 1 \\
\dfrac{k}{n_d - 1} & n_d > 1
\end{cases}
$$

so the first bar is always $0$ and the last bar is always $1$, even when the session length varies.

### 2.3 Recommended preprocessing output

The preprocessing stage should produce aligned `float32` arrays for:

- OHLC used by the execution simulator
- feature matrix used by the policy and value functions
- day identifiers, bar-in-day indices, and session-end masks used to flatten inventory and cancel working orders at day boundaries

The preprocessing should be done once on CPU with pandas, then converted into contiguous arrays before walk-forward slicing.

### 2.4 Fold-local environment parameters

For each walk-forward fold $m$, materialize fold-local environment parameter records

$$
\operatorname{EnvParam}^{\text{train}}_m,
\qquad
\operatorname{EnvParam}^{\text{infer}}_m,
$$

where each `EnvParam` stores:

- the fold-local `PreprocessedArrays` slice used by that rollout
- environment constants such as episode length $H$, episode stride $S$, maximum inventory $Q_{\max}$, quote-size multiplier, and end-of-day flatten flag
- the number of parallel environments `NUM_ENV = E`

This means the environment reads all data through its `EnvParam` instead of receiving per-environment data windows.

## 3. Feature Engineering

Let the close-to-close log-return be

$$
r_i = \log\left(\frac{C_i}{C_{i-1}}\right)
$$

and let the EMA decay for length $L$ be

$$
\alpha(L) = \frac{2}{L+1}.
$$

All rolling lengths below are hyperparameters.

### 3.1 EMA variance

Use squared log-return as the instantaneous variance proxy:

$$
\nu_i = r_i^2.
$$

Its EMA is

$$
\widehat{\nu}_i^{(L_{\nu})} = \alpha(L_{\nu})\,\nu_i + \left(1-\alpha(L_{\nu})\right)\widehat{\nu}_{i-1}^{(L_{\nu})}.
$$

With the requested default, $L_{\nu}=9$.

### 3.2 Price EMAs

Fast and slow EMAs of close are

$$
\operatorname{EMA}^{\text{fast}}_i = \operatorname{EMA}(C_i; L_f),
\qquad
\operatorname{EMA}^{\text{slow}}_i = \operatorname{EMA}(C_i; L_s)
$$

with defaults $L_f=8$ and $L_s=30$.

### 3.3 Average true range

Define true range as

$$
\operatorname{TR}_i = \max\left(H_i-L_i,\; |H_i-C_{i-1}|,\; |L_i-C_{i-1}|\right)
$$

and ATR as

$$
\operatorname{ATR}_i^{(L_{\text{atr}})} = \operatorname{EMA}(\operatorname{TR}_i; L_{\text{atr}})
$$

with default $L_{\text{atr}}=14$.

### 3.4 Signed realized variance imbalance

A compact rolling-window definition consistent with your latest specification is

$$
\operatorname{SRVI}_i = \frac{\sum_{j=i-L_{\text{srvi}}+1}^{i} r_j^2\,\operatorname{sign}(r_j)}{\sum_{j=i-L_{\text{srvi}}+1}^{i} r_j^2 + \varepsilon}
$$

with default $L_{\text{srvi}}=9$.

Because the numerator is bounded in magnitude by the denominator, this feature lies approximately in $[-1,1]$.

Interpretation:

- positive values indicate upside variance dominance
- negative values indicate downside variance dominance
- larger magnitude indicates more directional variance imbalance

### 3.5 ATR-scaled EMA slope

Convert return variance into a one-sigma price-range proxy using the current close:

$$
\sigma^{\text{px}}_i = C_i\left(\exp\left(\sqrt{\widehat{\nu}_i^{(L_{\nu})} + \varepsilon}\right) - 1\right)
$$

with a small numerical floor $\varepsilon > 0$.

This is the quote-distance form. The quantity $C_i\exp(\sqrt{\widehat{\nu}_i})$ is the one-sigma up-move price level, while the market-making offset should use the distance from the current close, hence the subtraction of $1$.

Define the slow-EMA slope feature as

$$
\operatorname{VSlope}_i = \frac{\operatorname{EMA}^{\text{slow}}_i - \operatorname{EMA}^{\text{slow}}_{i-1}}{\operatorname{ATR}_i^{(L_{\text{atr}})} + \varepsilon}.
$$

This keeps the slope on the same intrabar range scale as the requested variance-scaled MACD feature.

### 3.6 Variance-scaled MACD and MACD slope

The requested variance-scaled MACD is

$$
\operatorname{VMACD}_i = \frac{\operatorname{EMA}^{\text{fast}}_i - \operatorname{EMA}^{\text{slow}}_i}{\operatorname{ATR}_i^{(L_{\text{atr}})} + \varepsilon}.
$$

Its slope is the first difference

$$
\Delta\operatorname{VMACD}_i = \operatorname{VMACD}_i - \operatorname{VMACD}_{i-1}.
$$

### 3.7 Inventory feature

Let inventory in units be $q_i$ and let the maximum inventory be $Q_{\max}$. The normalized inventory is

$$
\bar q_i = \frac{q_i}{Q_{\max}} \in [-1,1].
$$

### 3.8 Final state vector

At each time step $i$, the PPO state is

$$
s_i = \left[
\tau_i,
r_i,
\widehat{\nu}_i^{(L_{\nu})},
\operatorname{SRVI}_i,
\operatorname{VSlope}_i,
\operatorname{VMACD}_i,
\Delta\operatorname{VMACD}_i,
\bar q_i
\right].
$$

This gives an 8-dimensional feature vector.

## 4. Action Space and Execution Model

### 4.1 Action definition

The action is

$$
a_i = (A_{1,i}, A_{2,i}, A_{3,i})
$$

with bounds

$$
A_{1,i} \in [0,2],
\qquad
A_{2,i} \in [0,2],
\qquad
A_{3,i} \in [0,1].
$$

Interpretation:

- $A_1$ controls bid distance in volatility units
- $A_2$ controls ask distance in volatility units
- $A_3$ controls quote size intensity

### 4.2 Quote construction

Using the close price and the price-standard-deviation proxy,

$$
b_i = C_i - A_{1,i}\sigma_i^{\text{px}},
\qquad
a_i^{\text{ask}} = C_i + A_{2,i}\sigma_i^{\text{px}}.
$$

The quote size per side is

$$
n_i^{\text{raw}} = 100 A_{3,i},
\qquad
n_i = \operatorname{round}(n_i^{\text{raw}}) \in \{0,1,\dots,100\}.
$$

### 4.3 Inventory-constrained executable size

To enforce $q_i \in [-Q_{\max},Q_{\max}]$, the executable buy and sell sizes should be capped by remaining inventory headroom:

$$
n_i^{\text{bid}} = \min\left(n_i,\; Q_{\max} - q_i\right),
\qquad
n_i^{\text{ask}} = \min\left(n_i,\; Q_{\max} + q_i\right).
$$

This is preferable to clipping inventory after the fact because it preserves a causal action-to-fill mapping.

Both $n_i^{\text{bid}}$ and $n_i^{\text{ask}}$ are therefore integer lot sizes.

### 4.4 Fill rules from next OHLC

Quotes decided at time $i$ are evaluated against the next bar $(O_{i+1},H_{i+1},L_{i+1},C_{i+1})$.

Assume 100% fill rate when touched.

Bid fill:

$$
f_i^{\text{bid}} = \mathbf{1}\{L_{i+1} \le b_i\}
$$

Ask fill:

$$
f_i^{\text{ask}} = \mathbf{1}\{H_{i+1} \ge a_i^{\text{ask}}\}
$$

Executed signed trade size:

$$
\Delta q_i^{\text{trade}} = f_i^{\text{bid}} n_i^{\text{bid}} - f_i^{\text{ask}} n_i^{\text{ask}}.
$$

Inventory update:

$$
q_{i+1} = q_i + \Delta q_i^{\text{trade}}.
$$

Cash update:

$$
\text{cash}_{i+1} = \text{cash}_i - f_i^{\text{bid}} n_i^{\text{bid}} b_i + f_i^{\text{ask}} n_i^{\text{ask}} a_i^{\text{ask}}.
$$

### 4.5 Both-side touch assumption

When both $L_{i+1} \le b_i$ and $H_{i+1} \ge a_i^{\text{ask}}$, the minute bar does not reveal which side traded first. For the first implementation, the cleanest assumption is:

$$
f_i^{\text{bid}} = 1,
\qquad
f_i^{\text{ask}} = 1
$$

which means both orders fill in that bar if both levels were touched. This is optimistic but internally consistent with the current 100% fill-rate simplification.

This document adopts that both-fill assumption as the default simulator behavior.

If a later robustness check is needed, a conservative tie-break rule means resolving the ambiguous bar in the least favorable direction for the current inventory and quote placement instead of granting both fills. A practical definition is:

- if inventory is long, assume the ask fills first and the bid does not fill on that bar
- if inventory is short, assume the bid fills first and the ask does not fill on that bar
- if inventory is flat, choose the side farther from the open as unfilled, or simply mark the bar as no-fill if a stricter convention is desired

This rule reduces optimistic spread capture on bars where the intrabar path is unknown.

### 4.6 End-of-day reset inside long episodes

Episodes are allowed to cross day boundaries. To avoid overnight jumps while preserving long contiguous slices, the environment applies a hard reset at the end of each trading day:

- cancel all working bid and ask orders
- flatten inventory to zero at the session close
- carry realized cash PnL forward into the next day

If $i$ is the last bar of a session, then after marking to the close $C_i$:

$$
cash_i^{\mathrm{flat}} = cash_i + q_i C_i,
\qquad
q_i^{\text{flat}} = 0.
$$

Because flattening occurs at the same close used for mark-to-market valuation, it changes the state carried into the next day without creating an extra artificial PnL jump.

## 5. Reward Design

### 5.1 Mark-to-close portfolio PnL

Define marked portfolio value at the close as

$$
\mathcal V_i = \text{cash}_i + q_i C_i.
$$

Initialize each training episode and each inference window with

$$
cash_0 = 0,
\qquad
q_0 = 0.
$$

The one-step PnL is

$$
\operatorname{PnL}_i = \mathcal V_{i+1} - \mathcal V_i.
$$

This uses close-to-close marking, consistent with the requirement that each step uses the close price as reference. With zero initial cash and zero initial inventory, cumulative PnL over an episode or inference window is simply the current marked portfolio value.

### 5.2 Damped PnL

The damped PnL term is

$$
\operatorname{DPnL}_i = \operatorname{PnL}_i - \max\left(0,\eta_{\text{dp}}\operatorname{PnL}_i\right).
$$

Equivalently,

$$
\operatorname{DPnL}_i =
\begin{cases}
(1-\eta_{\text{dp}})\operatorname{PnL}_i & \operatorname{PnL}_i > 0 \\
\operatorname{PnL}_i & \operatorname{PnL}_i \le 0
\end{cases}
$$

so positive PnL is damped while negative PnL is left unchanged.

### 5.3 Trading PnL

Using the next close $C_{i+1}$ as the bar reference for fills,

$$
\operatorname{TPnL}_i = f_i^{\text{bid}} n_i^{\text{bid}} (C_{i+1} - b_i) + f_i^{\text{ask}} n_i^{\text{ask}} (a_i^{\text{ask}} - C_{i+1}).
$$

This is equivalent to the signed-size form requested by the spec.

### 5.4 Inventory penalty

The inventory penalty is

$$
\operatorname{IPen}_i = \eta_{\text{ip}}\,\bar q_{i+1}^2.
$$

### 5.5 Total reward

The per-step reward is

$$
R_i = \operatorname{DPnL}_i + \operatorname{TPnL}_i - \operatorname{IPen}_i.
$$

This reward encourages spread capture and mark-to-close profitability while discouraging persistent inventory accumulation.

### 5.6 Reward normalization

Raw PnL-based reward can become much more volatile in high-volatility regimes, which can make PPO updates noisier and distort the relative weight of the inventory penalty. The chosen design is to normalize both damped PnL and trading PnL by $\operatorname{ATR}_i$ while keeping the normalized reward linear.

Define the step risk scale as

$$
S_i^{\text{reward}} = \operatorname{ATR}_i^{(L_{\text{atr}})} + \varepsilon.
$$

Then the normalized damped and trading PnL terms are

$$
\widetilde{\operatorname{DPnL}}_i = \frac{\operatorname{DPnL}_i}{S_i^{\text{reward}}},
\qquad
\widetilde{\operatorname{TPnL}}_i = \frac{\operatorname{TPnL}_i}{S_i^{\text{reward}}}.
$$

The training reward is therefore

$$
\widetilde R_i = \widetilde{\operatorname{DPnL}}_i + \widetilde{\operatorname{TPnL}}_i - \eta_{\text{ip}}\bar q_{i+1}^2.
$$

This keeps the reward scale adaptive to intraday range conditions without adding an extra nonlinear squashing layer. Using ATR alone instead of $Q_{\max}\operatorname{ATR}$ also avoids shrinking the PPO reward magnitude too aggressively when $Q_{\max}$ is large.

This training normalization is intentionally separate from the risk-adjusted reporting normalization in Section 11.

## 6. Episode Construction and Walk-Forward Validation

### 6.1 Walk-forward folds

Let:

- training window length be $N_{\text{train}} = 80{,}000$
- inference window length be $N_{\text{infer}} = 20{,}000$
- fold stride be $\Delta = 40{,}000$

All three are hyperparameters.

For fold $m$, define

$$
\mathcal I_m^{\text{train}} = [m\Delta,\; m\Delta + N_{\text{train}})
$$

$$
\mathcal I_m^{\text{infer}} = [m\Delta + N_{\text{train}},\; m\Delta + N_{\text{train}} + N_{\text{infer}}).
$$

The number of valid folds over a dataset of length $T$ is

$$
M = 1 + \left\lfloor \frac{T - (N_{\text{train}} + N_{\text{infer}})}{\Delta} \right\rfloor.
$$

### 6.2 Continual training across folds

At the end of fold $m$, carry forward both network parameters and optimizer states into fold $m+1$. This matches the requirement to continuously train the networks rather than reinitializing them on each walk-forward split.

### 6.3 Parallel environment indexing inside each training fold

Each episode has horizon $H = 128$ and episode-start stride $S = 64$, both hyperparameters. Let `NUM_ENV = E` denote the number of parallel environment instances carried in one grouped rollout.

For a training window of length $N_{\text{train}}$, the valid episode-start count is

$$
N_{\text{ep}} = 1 + \left\lfloor\frac{N_{\text{train}} - H}{S}\right\rfloor.
$$

Instead of slicing a separate training window per environment, maintain one `EnvState` per environment lane. Each `EnvState` stores at least:

- inventory and cash carry
- a local step index $\ell$, starting from $0$ within each episode rollout
- a fold-local global step index $g$, used to fetch bars and features from `EnvParam`

For environment lane $e \in \{0,1,\dots,E-1\}$ and grouped rollout step $n$, initialize the lane with

$$
\ell^{(0)}_{n,e} = 0,
\qquad
g^{\text{train},(0)}_{n,e} = eS + nES.
$$

Then, for inner episode step $h \in \{0,1,\dots,H-1\}$,

$$
\ell^{(h)}_{n,e} = h,
\qquad
g^{\text{train},(h)}_{n,e} = eS + nES + h.
$$

The corresponding episode slice is therefore

$$
\mathcal E_{n,e} = [g^{\text{train},(0)}_{n,e},\; g^{\text{train},(0)}_{n,e} + H).
$$

This keeps all environments on the same fold-local arrays while giving each environment lane its own progression state. A rollout lane is valid when

$$
g^{\text{train},(0)}_{n,e} + H \le N_{\text{train}}.
$$

The grouped rollout scan length is therefore

$$
N_{\text{scan}} = \left\lceil\frac{N_{\text{ep}}}{E}\right\rceil.
$$

### 6.4 Session-boundary handling

Episodes may cross day boundaries. The overnight discontinuity is handled inside the environment by the end-of-day reset described in Section 4.6 rather than by filtering out cross-day episodes.

This keeps more training data available while still removing overnight inventory risk and stale quotes.

## 7. PPO Architecture

## 7.1 Network choice

Use shallow actor and critic networks implemented in Flax. A practical default is two hidden layers per network with widths in the 32 to 128 range. The actor and critic should be separate networks rather than a shared trunk in the first version because:

- the actor and critic use different regularization pressures
- value gradients can distort action features in a small-data financial setting
- separate losses are easier to inspect with `jax.debug.print`

### 7.2 Actor parameterization with Sigmoid-Normal distributions

Because the action space is continuous and bounded, use a diagonal Normal policy in latent space followed by a logistic sigmoid transform and affine range mapping.

Let the actor output a mean vector and log-standard-deviation vector

$$
\mu_\theta(s_i) = \left[\mu_{1,i}, \mu_{2,i}, \mu_{3,i}\right],
\qquad
\ell_\theta(s_i) = \left[\ell_{1,i}, \ell_{2,i}, \ell_{3,i}\right].
$$

Define bounded standard deviations by clipping log-scales to a configured interval:

$$
\sigma_{k,i} = \exp\left(\operatorname{clip}(\ell_{k,i}, \ell_{\min}, \ell_{\max})\right).
$$

Sample latent actions

$$
u_i \sim \mathcal N\!\left(\mu_\theta(s_i), \operatorname{diag}(\sigma_{1,i}^2, \sigma_{2,i}^2, \sigma_{3,i}^2)\right)
$$

and apply the sigmoid squashing transform componentwise:

$$
y_i = \operatorname{sigmoid}(u_i) = [y_{1,i}, y_{2,i}, y_{3,i}] \in (0,1)^3.
$$

Map the bounded latent coordinates to the environment action ranges as

$$
A_{1,i} = 2 y_{1,i},
\qquad
A_{2,i} = 2 y_{2,i},
\qquad
A_{3,i} = y_{3,i}.
$$

For PPO, define the inverse normalized coordinates

$$
y_{1,i} = \frac{A_{1,i}}{2},
\qquad
y_{2,i} = \frac{A_{2,i}}{2},
\qquad
y_{3,i} = A_{3,i},
$$

and

$$
u_{k,i} = \operatorname{logit}(y_{k,i}) = \log\left(\frac{y_{k,i}}{1-y_{k,i}}\right).
$$

The action log-density is then

$$
\log \pi_\theta(a_i \mid s_i)
= \sum_{k=1}^{3}\left[
\log \mathcal N\!\left(u_{k,i}\,\middle|\,\mu_{k,i}, \sigma_{k,i}^2\right)
- \log\left(y_{k,i}(1-y_{k,i})\right)
\right]
- 2\log 2,
$$

where the final $-2\log 2$ term is the Jacobian correction from rescaling the first two channels from $(0,1)$ to $(0,2)$.

During inference, use deterministic actions obtained by transforming the actor mean directly:

$$
A_{1,i}^{\text{mean}} = 2\operatorname{sigmoid}(\mu_{1,i}),
\quad
A_{2,i}^{\text{mean}} = 2\operatorname{sigmoid}(\mu_{2,i}),
\quad
A_{3,i}^{\text{mean}} = \operatorname{sigmoid}(\mu_{3,i}).
$$

### 7.3 Critic

The critic outputs a scalar state value

$$
V_\phi(s_i).
$$

### 7.4 Generalized advantage estimation

Use GAE with discount $\gamma$ and trace parameter $\lambda$:

$$
\delta_i = R_i + \gamma V_\phi(s_{i+1}) - V_\phi(s_i)
$$

$$
\hat A_i = \delta_i + \gamma\lambda\hat A_{i+1}
$$

and targets

$$
\hat G_i = \hat A_i + V_\phi(s_i).
$$

## 8. PPO Objective with Entropy and L1 Regularization

Let the importance ratio be

$$
\rho_i(\theta) = \frac{\pi_\theta(a_i \mid s_i)}{\pi_{\theta_{\text{old}}}(a_i \mid s_i)}.
$$

### 8.1 Actor loss

The clipped PPO actor objective is

$$
\mathcal L_{\text{clip}}(\theta) = \mathbb E\left[
\min\left(
\rho_i(\theta)\hat A_i,
\operatorname{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon)\hat A_i
\right)
\right].
$$

Let the policy entropy bonus be computed from the pre-sigmoid diagonal Normal,

$$
\mathcal H_{\text{base}}(s_i) = \sum_{k=1}^{3} \frac{1}{2}\log\left(2\pi e\,\sigma_{k,i}^2\right),
$$

which is the stable default regularizer for the Sigmoid-Normal policy. The actor loss to minimize is

$$
\mathcal J_{\text{actor}} = -\mathcal L_{\text{clip}}(\theta)
- \beta_{\text{ent}}\,\mathbb E\left[\mathcal H_{\text{base}}(s_i)\right]
+ \lambda_{1,a}\|\theta\|_1.
$$

This satisfies the constraints that the actor uses entropy regularization for exploration and L1 regularization for overfitting control.

### 8.2 Critic loss

The critic loss is

$$
\mathcal J_{\text{critic}} = c_v\,\mathbb E\left[(V_\phi(s_i) - \hat G_i)^2\right] + \lambda_{1,c}\|\phi\|_1.
$$

If needed, the squared error can be replaced by clipped value loss later, but the unclipped form is the cleanest starting point.

### 8.3 Total objective

With separate networks, optimize actor and critic independently using separate `TrainState`s and optimizers.

## 9. GPU-Centric Training and Inference Plan

### 9.1 Fold-level transfer strategy

For each walk-forward fold:

1. Slice one training window and one inference window from the fully preprocessed arrays on CPU.
2. Build `EnvParam_train` and `EnvParam_infer` for that fold, then transfer both to GPU once.
3. Maintain a bank of `NUM_ENV` training `EnvState`s, each with its own local step index and fold-local global step index into `EnvParam_train`.
4. Run grouped rollout, PPO updates, and inference directly from those device-resident `EnvParam` and `EnvState` containers.
4. Copy back only scalar metrics needed for JSON logging.

This avoids repetitive host-device transfers.

### 9.2 Device-side `EnvParam` and `EnvState` layout

After moving a fold to device, keep the fold-local arrays inside `EnvParam` and represent parallel rollout by an `EnvState` bank of width $E = \text{NUM_ENV}$.

Each `EnvState` lane stores inventory, cash, local step index, and fold-local global step index. At grouped rollout step $n$, each lane is initialized with local step index $0$ and its lane-specific fold-local global start index, then the inner horizon scan advances both indices through $H$ steps.

For one grouped rollout step, the total number of transition rows is

$$
H E.
$$

After scanning over all grouped rollout steps in one update, the collected training batch size is

$$
N_{\text{batch}} = N_{\text{scan}} H E.
$$

This rollout tensor is then flattened and reshaped into minibatches of size $B$, with either exact divisibility or a final masked tail batch.

### 9.3 PPO update structure

The training loop should mirror the JAX scan/vmap style from [rl/walkthrough.ipynb](/home/metis/Projects/SV-MM/rl/walkthrough.ipynb), but with grouped parallel environments rather than an explicit episode-start buffer.

For fold $m$, the outer structure is:

$$
	ext{for fold } m
\rightarrow \text{for update } u = 1,\dots,\text{NUM\_UPDATES}
\rightarrow \text{scan over } N_{\text{scan}} \text{ grouped rollout steps}
\rightarrow \text{collect } H E \text{ transitions per grouped step}
\rightarrow \text{flatten to } N_{\text{batch}} \text{ rows}
\rightarrow \text{reshape into minibatches}
\rightarrow \text{for PPO epoch } k
\rightarrow \text{for minibatch } b.
$$

Operationally:

- `lax.scan` over grouped rollout steps advances the `NUM_ENV` `EnvState`s together
- `vmap` over environment lanes handles the parallel episode launches inside each grouped step
- a second `lax.scan` over the episode horizon handles the path-dependent environment dynamics within each environment lane
- the resulting batch is flattened and reshaped into PPO minibatches based on minibatch size
- `lax.scan` over PPO epochs and then over minibatches performs the actor and critic updates
- `jit` should wrap the full fold-local update path

Minibatch size is $B = 32$ by specification, but this remains a hyperparameter.

### 9.4 Inference after each epoch

After each training epoch, run one inference pass over the fold's inference `EnvParam` using the latest actor and critic states.

Inference mode is fixed as follows:

- use deterministic actor actions from the transformed policy mean rather than sampled actions
- preserve the same environment logic as training
- initialize with zero inventory and zero cash
- flatten inventory and cancel all working orders at every day boundary inside inference

Because inference is path-dependent through inventory, the natural structure is one `lax.scan` over the full inference `EnvParam`, typically with a single inference `EnvState` that advances over the entire inference window.

## 10. Epoch-Level Logging and JSON Output

### 10.1 `jax.debug.print` diagnostics

After each training epoch, print at least:

- actor loss
- critic loss
- entropy term
- actor L1 penalty
- critic L1 penalty
- inference cumulative PnL
- inference risk-adjusted Sharpe ratio
- inference raw Sharpe ratio
- inference risk-adjusted Sortino ratio
- inference raw Sortino ratio
- inference maximum drawdown

These diagnostics should be emitted inside the jitted update/evaluation path with `jax.debug.print` so the printed values match the actual device execution.

### 10.2 Suggested JSON record schema

The JSON file should be appended or overwritten after each epoch with a fold-and-epoch record containing, at minimum:

```json
{
    "fold_id": "...",
    "epoch": 0,
    "actor_loss": 0.0,
    "critic_loss": 0.0,
    "entropy": 0.0,
    "actor_l1": 0.0,
    "critic_l1": 0.0,
    "cumulative_pnl": 0.0,
    "sharpe_risk_adj": 0.0,
    "sharpe_raw": 0.0,
    "sortino_risk_adj": 0.0,
    "sortino_raw": 0.0,
    "max_drawdown": 0.0
}
```

Recommended file location: a timestamped JSON file under the RL logging area so each walk-forward experiment remains reproducible.

## 11. Evaluation Metrics

Let the inference step PnL series be $p_i$.

### 11.1 Cumulative PnL

$$
\operatorname{CumPnL}_t = \sum_{i=1}^{t} p_i.
$$

### 11.2 Two return series for reporting

Report both a risk-adjusted series and a raw series.

Recommended risk-adjusted per-step return:

$$
g_i^{\text{risk}} = \frac{p_i}{Q_{\max}(\sigma_i^{\text{px}} + \varepsilon)}.
$$

This scales PnL by the maximum-inventory one-sigma price range and makes folds more comparable across volatility regimes.

Raw series:

$$
g_i^{\text{raw}} = p_i.
$$

Because this design does not impose a fixed capital base and starts from zero cash, the raw series is best interpreted as a raw step-PnL sequence rather than a percentage return.

### 11.3 Sharpe ratio

If $g_i$ denotes either $g_i^{\text{risk}}$ or $g_i^{\text{raw}}$, and $B_{\text{year}}$ is the number of tradable 1-minute bars per year used for annualization, then

$$
\operatorname{Sharpe} = \sqrt{B_{\text{year}}}\;\frac{\mathbb E[g_i]}{\operatorname{Std}(g_i) + \varepsilon}.
$$

Report both $\operatorname{Sharpe}^{\text{risk}}$ and $\operatorname{Sharpe}^{\text{raw}}$.

### 11.4 Sortino ratio

Let downside deviation be

$$
\sigma^-_g = \sqrt{\mathbb E\left[\min(g_i,0)^2\right]}.
$$

Then

$$
\operatorname{Sortino} = \sqrt{B_{\text{year}}}\;\frac{\mathbb E[g_i]}{\sigma^-_g + \varepsilon}.
$$

Report both $\operatorname{Sortino}^{\text{risk}}$ and $\operatorname{Sortino}^{\text{raw}}$.

### 11.5 Maximum drawdown

From the cumulative PnL curve,

$$
\operatorname{DD}_t = \max_{u \le t} \operatorname{CumPnL}_u - \operatorname{CumPnL}_t
$$

and

$$
\operatorname{MDD} = \max_t \operatorname{DD}_t.
$$

## 12. Design Trees and Architectures

### 12.1 End-to-end design tree

```text
RL volatility scalping system
├── Data layer
│   ├── Database OHLCV bars
│   ├── Pandas preprocessing
│   ├── Feature matrix
│   ├── Walk-forward fold slicing
│   └── Fold-local EnvParam materialization
├── Environment layer
│   ├── EnvParam = fold-local arrays + environment constants
│   ├── EnvState = carry + local/global step indices
│   ├── State = 8 features + normalized inventory
│   ├── Sigmoid-Normal policy actions
│   ├── Quote construction from close and sigma_px
│   ├── Next-bar OHLC execution simulator
│   ├── End-of-day flatten and order cancel
│   └── Reward from ATR-normalized DPnL, ATR-normalized TPnL, inventory penalty
├── Learning layer
│   ├── Actor network
│   ├── Critic network
│   ├── GAE
│   ├── PPO clipped objective
│   ├── Entropy regularization
│   └── L1 regularization
├── Acceleration layer
│   ├── GPU fold transfer
│   ├── Device-resident EnvParam per fold
│   ├── Vectorized EnvState bank
│   ├── vmap over environment lanes/minibatches
│   ├── lax.scan over time and epochs
│   └── jit over update and inference steps
└── Evaluation layer
    ├── Epoch inference pass
    ├── Cumulative PnL
    ├── Sharpe risk-adjusted and raw
    ├── Sortino risk-adjusted and raw
    ├── Maximum drawdown
    └── JSON logging + jax.debug.print
```

### 12.2 Actor architecture

Recommended shallow actor architecture:

```text
Input state (8)
-> Dense(h1)
-> activation
-> Dense(h2)
-> activation
-> Dense(6)
-> split into 3 means and 3 log-stds
-> clip log-stds and exponentiate
-> sample diagonal Normal latent action
-> sigmoid squash
-> affine map to A1, A2, A3
```

Suggested default widths are two hidden layers in the 32 to 128 range, for example $(64, 64)$.

### 12.3 Critic architecture

Recommended shallow critic architecture:

```text
Input state (8)
-> Dense(h1)
-> activation
-> Dense(h2)
-> activation
-> Dense(1)
-> scalar state value
```

Keeping the critic separate from the actor avoids shared-feature interference and makes actor/critic diagnostics easier to interpret.

### 12.4 Training loop architecture

```text
Walk-forward fold
-> slice train + inference windows and build fold-local EnvParam records
-> move EnvParam_train + EnvParam_infer to GPU
-> repeat for each update:
    initialize or refresh NUM_ENV training EnvStates
    scan over grouped rollout steps
    run episode-horizon scans inside each environment lane
    flatten collected rollout rows
    reshape by minibatch size
    repeat for each PPO epoch:
        iterate minibatches
        compute GAE and PPO losses
        update actor and critic TrainStates
-> run full-window inference scan from EnvParam_infer
-> print metrics and persist JSON
-> warm-start next fold from current TrainStates
```

## 13. Suggested Config Structure

Per the requirement, hyperparameters should live in a `config.py` and not be supplied through CLI arguments. A clean grouping is:

- data: root, start date, end date, train/infer lengths, fold stride
- features: all EMA and ATR lengths, epsilon floor
- environment: episode length, episode stride, `NUM_ENV`, max inventory, quote size multiplier, end-of-day flatten flag, integer-lot rounding rule
- reward: $\eta_{\text{dp}}$, $\eta_{\text{ip}}$, reward epsilon
- PPO: learning rates, discount, GAE lambda, clip epsilon, entropy coefficient, actor L1, critic L1, minibatch size, epochs
- model: hidden sizes, activation, log-std bounds
- logging: JSON path, print cadence, evaluation annualization factor, raw and risk-adjusted metric toggles

## 14. Important Design Choices to Confirm Before Coding

The following choices remain important enough to resolve explicitly before implementation:

1. Entropy regularization for the Sigmoid-Normal actor: keep using the pre-sigmoid Normal entropy as the stable default.
2. Both-side touched bars: keep the optimistic both-fill simulator assumption.
3. Risk-adjusted reporting scale: keep the adopted $Q_{\max}\sigma_i^{\text{px}}$ normalization.

## 15. Recommended First Implementation Order

To keep the build-out controlled, the implementation should proceed in this order:

1. pandas preprocessing and fold slicing
2. fold-local `EnvParam` materialization and `EnvState` definition
3. JAX execution environment with inventory and next-OHLC fills
4. actor and critic Flax modules plus `TrainState`
5. grouped `NUM_ENV` rollout and PPO update with GAE, entropy, and L1 losses
6. device-side inference scan and metric computation
7. epoch-level `jax.debug.print` and JSON logging
8. walk-forward orchestration with warm starts across folds

This order makes validation easier because each layer can be checked before the next one is added.

## 16. Dependency-Light Implementation Steps for Fixed Context Windows

The ordering in Section 15 is the coarse build sequence. For actual coding with a fixed context window, it is better to split the design into smaller implementation steps whose inputs and outputs are explicitly frozen. The goal is that each step depends on only a small, already-stable contract from the previous steps.

The recommended rule is:

- each step should introduce one main abstraction only
- each step should produce a small runnable or testable artifact
- each later step should consume a frozen output contract rather than reach back into earlier implementation details
- feature engineering, environment accounting, PPO math, and orchestration should stay separated until each is individually validated

### 16.1 Frozen intermediate contracts

Before implementing the full pipeline, treat the following interfaces as fixed handoff points between steps:

- `raw_bars_df`: canonical pandas DataFrame with timestamps, OHLCV fields, average price, trade count, and day segmentation columns
- `preprocessed_arrays`: contiguous `float32` arrays for OHLC, feature matrix, ATR, sigma proxy, day ids, bar-in-day indices, and session-end masks
- `env_param_contract`: fold-local `EnvParam` records for training and inference, each bundling preprocessed arrays with environment constants
- `env_state_contract`: per-lane environment carry containing inventory, cash, local step index, and fold-local global step index
- `env_step_contract`: pure transition from current `EnvState`, action, and `EnvParam` to next carry state, reward terms, and diagnostic fields
- `rollout_contract`: batched trajectory tensors produced by grouped `lax.scan` over `NUM_ENV` environment lanes and either episode-horizon or full-window scans
- `policy_value_contract`: actor mean and log-std outputs, bounded action transform, log-probability, entropy, and critic value
- `epoch_metrics_contract`: scalar losses and inference metrics copied back to host for logging

If these contracts are kept stable, the implementation can proceed step-by-step without reopening already-finished components.

### 16.2 Recommended step breakdown

1. Config and container definitions.

    Implement the config groups and the minimal typed containers or dictionaries that represent the contracts above.

    Depends on: none.

    Output: a stable config layout and named payloads for preprocessing output, `EnvParam`, `EnvState`, rollout outputs, and epoch metrics.

    Validation: instantiate defaults and verify all required hyperparameter groups are present.

2. Bar loading and session segmentation.

    Implement database loading into a canonical pandas DataFrame and add day ids, bar-in-day indices, normalized time-of-day, and session-end flags.

    Depends on: Step 1.

    Output: `raw_bars_df`.

    Validation: first and last bar of each day map to $0$ and $1$ in $	au$, and session-end masks align with day boundaries.

3. Feature engineering on CPU.

    Implement returns, EMA variance, ATR, SRVI, VSlope, VMACD, and VMACD slope, then convert everything into contiguous `float32` arrays.

    Depends on: Step 2.

    Output: `preprocessed_arrays` without any PPO or environment logic.

    Validation: check shapes, dtype, warmup handling, and absence of unintended NaNs or infs.

4. Walk-forward slicing and fold-local `EnvParam` materialization.

    Implement fold construction, train and inference ranges, and the train/inference `EnvParam` records for each fold.

    Depends on: Step 3.

    Output: `env_param_contract` for each fold.

    Validation: verify fold count, exact slice lengths, and that the grouped rollout indexing implied by `NUM_ENV`, episode length, and episode stride stays within the fold-local training arrays.

5. One-step execution kernel.

    Implement the pure environment step that takes the current `EnvState`, `EnvParam`, and action, then computes quote levels, fills, inventory updates, cash updates, global-step advancement, flatten-at-close handling, and reward terms.

    Depends on: Step 3.

    Output: `env_step_contract`.

    Validation: hand-check deterministic edge cases for no fill, bid-only fill, ask-only fill, both-fill, inventory headroom caps, end-of-day flatten, and local/global step-index advancement.

6. Rollout scans for training and inference.

    Lift the one-step kernel into grouped `lax.scan` so it can produce `NUM_ENV`-wide training trajectories and a full-window path-dependent rollout for inference.

    Depends on: Step 5.

    Output: `rollout_contract`.

    Validation: compare a short grouped rollout against a manual unroll on a tiny synthetic example and verify the `EnvState` bank advances the expected fold-local global indices.

7. Actor and critic modules.

    Implement separate Flax actor and critic modules, bounded action transforms, inverse transform logic for PPO log-probability, entropy term, and critic forward pass.

    Depends on: Step 1 and the fixed state dimension from Step 3.

    Output: `policy_value_contract`.

    Validation: confirm action bounds, finite log-probabilities, stable entropy values, and scalar critic outputs.

8. GAE and PPO loss functions.

    Implement advantage computation, return targets, clipped PPO actor loss, critic regression loss, entropy bonus, and L1 penalties using precomputed rollout tensors.

    Depends on: Steps 6 and 7.

    Output: differentiable actor and critic loss functions plus auxiliary diagnostics.

    Validation: finite losses on a toy batch, correct zero-advantage behavior, and consistent tensor shapes.

9. Minibatch update step on device.

    Implement grouped rollout collection from the `EnvState` bank, flattening and minibatch reshaping, optimizer updates, and the jitted epoch update path.

    Depends on: Steps 4, 6, 7, and 8.

    Output: one train-epoch function that updates actor and critic `TrainState`s and returns diagnostics.

    Validation: run one tiny epoch and confirm parameters change, losses remain finite, grouped rollout batch shapes are consistent, and minibatch reshaping matches minibatch size.

10. Inference metrics and JSON logging.

     Implement deterministic inference rollout, cumulative PnL, raw and risk-adjusted Sharpe and Sortino, maximum drawdown, `jax.debug.print`, and JSON record emission.

     Depends on: Steps 6, 7, and 9.

     Output: `epoch_metrics_contract` and persisted epoch records.

     Validation: check metric formulas on small synthetic PnL sequences where the expected result is easy to verify.

11. Walk-forward driver with warm starts.

    Implement the full fold loop that transfers each fold-local `EnvParam` to device once, trains over updates and epochs, evaluates after each epoch, logs results, and carries actor, critic, and optimizer states into the next fold.

     Depends on: Steps 4 through 10.

     Output: complete end-to-end training and inference orchestration.

     Validation: run one fold first, then multiple folds, and verify state carry-over and logging continuity.

### 16.3 Minimal dependency graph

The recommended dependency graph is intentionally narrow:

```text
Step 1 config/contracts
-> Step 2 bar loading
-> Step 3 feature arrays
-> Step 4 fold slicing and EnvParam materialization

Step 3 feature arrays
-> Step 5 env step
-> Step 6 rollout scan

Step 1 config/contracts + Step 3 state dimension
-> Step 7 actor/critic

Step 6 rollout scan + Step 7 actor/critic
-> Step 8 PPO losses

Step 4 fold slicing/EnvParam + Step 8 PPO losses
-> Step 9 train epoch

Step 6 rollout scan + Step 7 actor/critic + Step 9 train epoch
-> Step 10 inference metrics/logging

Step 4 through Step 10
-> Step 11 walk-forward driver
```

This decomposition is intended to keep each coding session local. A later step should be able to assume the previous contract is correct and avoid reloading the full design into context.