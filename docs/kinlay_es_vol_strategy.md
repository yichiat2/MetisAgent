# Kinlay ES 1-Minute Volatility Regime Strategy

**Author:** Jonathan Kinlay  
**Date:** May 19, 2014  
**Instrument:** @ES (E-mini S&P 500), 1-minute chart

---

## Overview

This strategy trades directionally on the ES futures contract by conditioning entry decisions on a **one-step-ahead forecast of realized volatility**, derived from directional True Range contributions. It distinguishes two regimes — **high volatility** and **low volatility** — and applies opposite logic to each:

- **High-vol regime → trend-following** entries with wide profit targets and tight stops.
- **Low-vol regime → mean-reversion** entries with tight profit targets and wide stops.

---

## Volatility Decomposition

True Range (TR) is split into upside and downside components based on the direction of the close:

$$
\text{plusTR}_t =
\begin{cases}
\text{TR}_t & \text{if } C_t > C_{t-1} \\
\text{unchanged} & \text{otherwise}
\end{cases}
$$

$$
\text{minusTR}_t =
\begin{cases}
\text{TR}_t & \text{if } C_t < C_{t-1} \\
\text{unchanged} & \text{otherwise}
\end{cases}
$$

---

## Volatility Forecast

A simple **linear extrapolation** (momentum-based forecast) is applied to each directional TR series. On each bar $t$:

$$
\widehat{\sigma}^+_t = \text{plusTR}_t + 0.5 \cdot \left(\text{plusTR}_t - \text{plusTR}_{t-1}\right)
= 1.5 \cdot \text{plusTR}_t - 0.5 \cdot \text{plusTR}_{t-1}
$$

$$
\widehat{\sigma}^-_t = \text{minusTR}_t + 0.5 \cdot \left(\text{minusTR}_t - \text{minusTR}_{t-1}\right)
= 1.5 \cdot \text{minusTR}_t - 0.5 \cdot \text{minusTR}_{t-1}
$$

This is equivalent to a first-order linear extrapolation with a dampening coefficient of 0.5, anticipating whether upside or downside volatility is accelerating or decelerating.

---

## Entry Logic

Entries are only permitted after a warm-up period of $\text{Len} - 1 = 9$ bars within the trading session, and only during session hours $[\text{StartTime},\, \text{EndTime}] = [08{:}00,\, 15{:}15]$ (exchange time).

### Regime Thresholds

| Parameter | Symbol | Default Value |
|---|---|---|
| `upperVolThreshold` | $\tau_H$ | 3.0 |
| `lowerVolThreshold` | $\tau_L$ | 0.25 |

### High-Volatility Regime (Trend-Following)

$$
\widehat{\sigma}^+_t > \tau_H \implies \text{Buy at Market} \quad \text{(Long Entry, LV)}
$$

$$
\widehat{\sigma}^-_t > \tau_H \implies \text{Sell Short at Market} \quad \text{(Short Entry, LV)}
$$

Rationale: When directional volatility is forecasted to be large, momentum/trend is likely to persist. Enter aggressively at market; target a large move and use a tight stop.

### Low-Volatility Regime (Mean-Reversion)

$$
\widehat{\sigma}^+_t < \tau_L \implies \text{Sell Short at InsideAsk (Limit)} \quad \text{(Short Entry, SV)}
$$

$$
\widehat{\sigma}^-_t < \tau_L \implies \text{Buy at InsideBid (Limit)} \quad \text{(Long Entry, SV)}
$$

Rationale: When directional volatility is forecasted to be very small, the market is in a tight range. Fade the last move using limit orders; target a small reversion and protect against a breakout with a wide stop.

---

## Risk Management

Profit targets and stop losses are defined in **ticks** and converted to dollar values:

$$
\text{PTv} = \text{PT} \cdot \frac{\text{MinMove}}{\text{PriceScale}}, \quad \text{SLv} = \text{SL} \cdot \frac{\text{MinMove}}{\text{PriceScale}}
$$

$$
\text{ProfitTarget (\$)} = \text{PTv} \cdot \text{BigPointValue}, \quad \text{StopLoss (\$)} = \text{SLv} \cdot \text{BigPointValue}
$$

### Parameter Table

| Parameter | Long/Short | Regime | Default (ticks) |
|---|---|---|---|
| `PTLVticks` | Both | High-vol | 30 |
| `SLLVticks` | Both | High-vol | 2 |
| `PTSVticks` | Both | Low-vol | 2 |
| `SLSVticks` | Both | Low-vol | 30 |

Note the **asymmetry**: the high-vol regime uses a wide PT and tight SL (reward-to-risk = 15:1), while the low-vol regime uses a tight PT and wide SL (reward-to-risk = 1:15). This reflects the statistical nature of each regime — trend trades are allowed to run; mean-reversion trades are expected to snap back quickly.

Additionally, `setexitonclose` forces all open positions to close at the end of each session, preventing overnight exposure.

---

## Session Bar Counter

A bar counter $\text{BN}$ resets to 1 at the start of each trading day and increments each bar:

$$
\text{BN}_t =
\begin{cases}
1 & \text{if } \text{date}_t \neq \text{date}_{t-1} \\
\text{BN}_{t-1} + 1 & \text{otherwise}
\end{cases}
$$

Entries are gated by $\text{BN} \geq \text{Len} - 1$ to ensure the volatility forecast has at least one prior directional TR observation.

---

## Strategy Summary Table

| Condition | Direction | Order Type | PT (ticks) | SL (ticks) |
|---|---|---|---|---|
| $\widehat{\sigma}^+_t > \tau_H$ | Long | Market | 30 | 2 |
| $\widehat{\sigma}^-_t > \tau_H$ | Short | Market | 30 | 2 |
| $\widehat{\sigma}^+_t < \tau_L$ | Short | Limit (InsideAsk) | 2 | 30 |
| $\widehat{\sigma}^-_t < \tau_L$ | Long | Limit (InsideBid) | 2 | 30 |

---

## TradingView Pine Script

The indicator below plots $\widehat{\sigma}^+$ and $\widehat{\sigma}^-$ in a separate pane and draws horizontal lines for both thresholds. Background shading highlights active regime signals. Both thresholds are exposed as user-adjustable inputs.

```pine
//@version=5
indicator("Kinlay Directional Vol Forecast", overlay=false)

// ─── User Inputs ───────────────────────────────────────────────────────────
upperVolThreshold = input.float(3.0,  title="Upper Vol Threshold (τ_H)",
     minval=0.0, step=0.25,
     tooltip="High-vol regime trigger. Above this → trend-following entry.")
lowerVolThreshold = input.float(0.25, title="Lower Vol Threshold (τ_L)",
     minval=0.0, step=0.05,
     tooltip="Low-vol regime trigger. Below this → mean-reversion entry.")

// ─── True Range ────────────────────────────────────────────────────────────
tr = ta.tr(true)

// ─── Directional TR (carry forward on neutral bars) ────────────────────────
// plusTR  updates only when close > close[1] (upside bar)
// minusTR updates only when close < close[1] (downside bar)
var float plusTR  = na
var float minusTR = na
plusTR  := close > close[1] ? tr : nz(plusTR)
minusTR := close < close[1] ? tr : nz(minusTR)

// ─── One-step-ahead Volatility Forecasts ───────────────────────────────────
// σ̂⁺ = 1.5·plusTR  − 0.5·plusTR[1]   (linear extrapolation, λ = 0.5)
// σ̂⁻ = 1.5·minusTR − 0.5·minusTR[1]
sigmaPlusHat  = 1.5 * plusTR  - 0.5 * nz(plusTR[1])
sigmaMinusHat = 1.5 * minusTR - 0.5 * nz(minusTR[1])

// ─── Plots ─────────────────────────────────────────────────────────────────
plot(sigmaPlusHat,  title="σ̂⁺  (Upside Vol Forecast)",
     color=color.new(color.green, 0), linewidth=2)
plot(sigmaMinusHat, title="σ̂⁻  (Downside Vol Forecast)",
     color=color.new(color.red, 0),   linewidth=2)

hline(upperVolThreshold, "Upper Threshold (τ_H)",
      color=color.orange, linestyle=hline.style_dashed, linewidth=2)
hline(lowerVolThreshold, "Lower Threshold (τ_L)",
      color=color.blue,   linestyle=hline.style_dashed, linewidth=2)

// ─── Regime Background Highlights ──────────────────────────────────────────
// Dark green  → high-vol long signal  (σ̂⁺  > τ_H)
// Dark red    → high-vol short signal (σ̂⁻  > τ_H)
// Light red   → low-vol short signal  (σ̂⁺  < τ_L)
// Light green → low-vol long signal   (σ̂⁻  < τ_L)
bgcolor(sigmaPlusHat  > upperVolThreshold ? color.new(color.green, 80) : na,
        title="High-Vol Long Signal")
bgcolor(sigmaMinusHat > upperVolThreshold ? color.new(color.red,   80) : na,
        title="High-Vol Short Signal")
bgcolor(sigmaPlusHat  < lowerVolThreshold ? color.new(color.red,   90) : na,
        title="Low-Vol Short Signal")
bgcolor(sigmaMinusHat < lowerVolThreshold ? color.new(color.green, 90) : na,
        title="Low-Vol Long Signal")
```

### Notes

- **Carry-forward logic** — `var` declarations make `plusTR` / `minusTR` persist across bars; each updates only on its respective directional bar, matching the TradeStation behaviour in the original strategy.
- **Threshold inputs** — Change `τ_H` and `τ_L` directly in the indicator settings panel. All plots and backgrounds update automatically.
- **Regime shading** — Solid backgrounds use 80 % transparency (high-vol) and 90 % transparency (low-vol) to remain readable on any chart theme.
- **Overlay** — Set `overlay=false` keeps the forecast in a sub-pane so price action is unobstructed; change to `overlay=true` if plotting on the price chart is preferred.
