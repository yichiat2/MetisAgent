from __future__ import annotations

import threading
from typing import Any, Mapping

import numpy as np

from .contracts import STATIC_FEATURE_NAMES


DEFAULT_DASH_PORT = 8051

_state: dict[str, Any] = {
    "lock": threading.Lock(),
    "payload": None,
}

_server_started = False
_server_lock = threading.Lock()
ATR_OVER_CLOSE_MILLI_INDEX = STATIC_FEATURE_NAMES.index("atr_over_close_milli")
ATR_IMBALANCE_INDEX = STATIC_FEATURE_NAMES.index("atr_imbalance")


def prepare_inference_payload(
    metrics: Mapping[str, Any],
    ohlc: Any,
    static_features: Any,
    fold_id: int,
    update_step: int,
    num_updates: int,
) -> dict[str, Any]:
    host_metrics = {
        key: _normalize_metric_value(value) for key, value in metrics.items()
    }
    aligned_ohlc = np.asarray(ohlc, dtype=np.float32)
    aligned_static_features = np.asarray(static_features, dtype=np.float32)
    if aligned_ohlc.ndim != 2 or aligned_ohlc.shape[1] < 4:
        raise ValueError("Expected OHLC input with shape (num_bars, 4)")
    if aligned_static_features.ndim != 2:
        raise ValueError("Expected static_features input with shape (num_bars, num_features)")

    bid_price = np.asarray(host_metrics["bid_price"], dtype=np.float32)
    ask_price = np.asarray(host_metrics["ask_price"], dtype=np.float32)
    num_steps = bid_price.shape[0]
    if ask_price.shape[0] != num_steps:
        raise ValueError("Bid and ask price series must have the same length")
    if aligned_ohlc.shape[0] != num_steps + 1:
        raise ValueError("Expected one extra OHLC bar so prices at t align with OHLC at t + 1")
    if aligned_static_features.shape[0] != num_steps + 1:
        raise ValueError(
            "Expected one extra static feature row so values at t align with the inference horizon"
        )
    if aligned_static_features.shape[1] <= ATR_OVER_CLOSE_MILLI_INDEX:
        raise ValueError("Static features are missing atr_over_close_milli")
    if aligned_static_features.shape[1] <= ATR_IMBALANCE_INDEX:
        raise ValueError("Static features are missing atr_imbalance")

    actions = np.asarray(host_metrics["actions"], dtype=np.int32)
    if actions.shape[0] != num_steps:
        raise ValueError("Action series must align with the inference horizon")

    reservation_price = np.asarray(host_metrics["reservation_price"], dtype=np.float32)
    spread = np.asarray(host_metrics["spread"], dtype=np.float32)
    atr_over_ema = aligned_static_features[:num_steps, ATR_OVER_CLOSE_MILLI_INDEX] / 1000.0
    atr_imbalance = aligned_static_features[:num_steps, ATR_IMBALANCE_INDEX]

    return {
        "fold_id": int(fold_id),
        "update_step": int(update_step),
        "num_updates": int(num_updates),
        "transaction_count": int(host_metrics["transaction_count"]),
        "total_pnl": float(host_metrics["total_pnl"]),
        "final_cumulative_return": float(host_metrics["final_cumulative_return"]),
        "sharpe_ratio": float(host_metrics["sharpe_ratio"]),
        "sortino_ratio": float(host_metrics["sortino_ratio"]),
        "max_drawdown": float(host_metrics["max_drawdown"]),
        "bankruptcy": bool(host_metrics["bankruptcy"]),
        "metrics": host_metrics,
        "ohlc": aligned_ohlc[1:, :4],
        "reservation_price": reservation_price,
        "spread": spread,
        "atr_imbalance": atr_imbalance,
        "atr_over_ema": atr_over_ema,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "inventory": np.asarray(host_metrics["inventory"], dtype=np.float32),
        "pnl": np.asarray(host_metrics["pnl"], dtype=np.float32),
        "cumulative_pnl": np.asarray(host_metrics["cumulative_pnl"], dtype=np.float32),
        "actions": np.asarray(host_metrics["actions"], dtype=np.float32),
    }


def push_inference_metrics(payload: Mapping[str, Any]) -> None:
    with _state["lock"]:
        _state["payload"] = {key: _copy_value(value) for key, value in payload.items()}


def start_dashboard_server(port: int = DEFAULT_DASH_PORT) -> None:
    global _server_started

    with _server_lock:
        if _server_started:
            return
        app = _build_app()
        _server_started = True

    def _run() -> None:
        import logging

        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="ppo-vol-scalping-dashboard")
    thread.start()
    print(f"[Dashboard] Serving PPO vol scalping dashboard at http://localhost:{port}/")


def _build_app():
    try:
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "The PPO vol scalping dashboard requires dash and plotly. "
            "Install them with: pip install dash plotly"
        ) from exc

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "padding": "12px"},
        children=[
            html.H2("PPO Vol Scalping Dashboard"),
            html.Div(id="stats-bar", style={"marginBottom": "10px", "fontWeight": "bold"}),
            dcc.Graph(id="inference-plot", style={"height": "1240px"}),
            dcc.Interval(id="interval", interval=2_000, n_intervals=0),
        ],
    )

    @app.callback(
        Output("stats-bar", "children"),
        Output("inference-plot", "figure"),
        Input("interval", "n_intervals"),
    )
    def _refresh(_n):
        with _state["lock"]:
            payload = _state["payload"]

        if payload is None:
            return "Waiting for final inference metrics...", go.Figure()

        return _build_stats_bar(payload, html), build_inference_figure(payload)

    return app


def build_inference_figure(payload: Mapping[str, Any]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ohlc = np.asarray(payload["ohlc"], dtype=np.float32)
    reservation_price = np.asarray(payload["reservation_price"], dtype=np.float32)
    bid_price = np.asarray(payload["bid_price"], dtype=np.float32)
    ask_price = np.asarray(payload["ask_price"], dtype=np.float32)
    inventory = np.asarray(payload["inventory"], dtype=np.float32)
    pnl = np.asarray(payload["pnl"], dtype=np.float32)
    cumulative_pnl = np.asarray(payload["cumulative_pnl"], dtype=np.float32)
    atr_imbalance = np.asarray(payload["atr_imbalance"], dtype=np.float32)
    atr_over_ema = np.asarray(payload["atr_over_ema"], dtype=np.float32)
    actions = np.asarray(payload["actions"], dtype=np.float32)
    timestep = np.arange(ohlc.shape[0], dtype=np.int32)
    if atr_imbalance.shape[0] != timestep.shape[0]:
        raise ValueError("Expected ATR imbalance series to align with the inference horizon")
    if atr_over_ema.shape[0] != timestep.shape[0]:
        raise ValueError("Expected ATR / EMA series to align with the inference horizon")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.42, 0.22, 0.14, 0.22],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{}],
            [{"secondary_y": True}],
        ],
        subplot_titles=(
            "OHLC (t + 1) with reservation price and bid/ask quotes (t)",
            "Cumulative PnL, step PnL, and ATR imbalance",
            "ATR / EMA(close) at t",
            "A1 inventory skew and A2 spread multiplier",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=timestep,
            open=ohlc[:, 0],
            high=ohlc[:, 1],
            low=ohlc[:, 2],
            close=ohlc[:, 3],
            name="OHLC t+1",
            increasing_line_color="#0b8f55",
            decreasing_line_color="#d1495b",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=reservation_price,
            mode="lines",
            name="Reservation price t",
            line={"color": "#111827", "width": 1.2, "dash": "dash"},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=bid_price,
            mode="lines",
            name="Bid price t",
            line={"color": "#1f77b4", "width": 1.4},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=ask_price,
            mode="lines",
            name="Ask price t",
            line={"color": "#ff7f0e", "width": 1.4},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=inventory,
            mode="lines",
            name="Inventory t+1",
            line={"color": "#6b7280", "width": 1.3, "dash": "dot"},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=cumulative_pnl,
            mode="lines",
            name="Cumulative PnL",
            line={"color": "#111827", "width": 1.8},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=timestep,
            y=pnl,
            name="PnL per step",
            marker_color="#9ca3af",
            opacity=0.7,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=atr_imbalance,
            mode="lines",
            name="ATR imbalance t",
            line={"color": "#7c3aed", "width": 1.4},
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=atr_over_ema,
            mode="lines",
            name="ATR / EMA(close) t",
            line={"color": "#0f766e", "width": 1.6},
        ),
        row=3,
        col=1,
    )

    action_names = ("A1 inventory skew", "A2 spread multiplier")
    action_colors = ("#2563eb", "#dc2626")
    if actions.ndim != 2 or actions.shape[1] != len(action_names):
        raise ValueError("Expected action series with shape (num_steps, 2)")
    for index, (name, color) in enumerate(zip(action_names, action_colors, strict=True)):
        fig.add_trace(
            go.Scatter(
                x=timestep,
                y=actions[:, index],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 1.4, "shape": "hv"},
                marker={"size": 5},
            ),
            row=4,
            col=1,
            secondary_y=bool(index),
        )

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Inventory", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Cumulative PnL", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Step PnL / ATR imbalance", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="ATR / EMA(close)", row=3, col=1)
    fig.update_yaxes(title_text="A1 inventory skew", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="A2 spread multiplier", row=4, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Timestep", row=4, col=1)
    fig.update_layout(
        template="plotly_white",
        height=1240,
        hovermode="x unified",
        bargap=0.05,
        margin={"t": 70, "b": 40, "l": 60, "r": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        title=(
            f"Fold {payload['fold_id']} inference | "
            f"update {payload['update_step']}/{payload['num_updates']}"
        ),
        xaxis_rangeslider_visible=False,
    )
    return fig


def _build_stats_bar(payload: Mapping[str, Any], html: Any):
    sharpe = _format_float(payload["sharpe_ratio"])
    sortino = _format_float(payload["sortino_ratio"])
    total_pnl = _format_float(payload["total_pnl"])
    total_return = _format_float(payload["final_cumulative_return"])
    max_drawdown = _format_float(payload["max_drawdown"])
    return html.Div(
        (
            f"Fold {payload['fold_id']} | Update {payload['update_step']}/{payload['num_updates']} | "
            f"Transactions {payload['transaction_count']} | Total PnL {total_pnl} | "
            f"Cumulative return {total_return} | Sharpe {sharpe} | Sortino {sortino} | "
            f"Max drawdown {max_drawdown} | Bankruptcy {payload['bankruptcy']}"
        )
    )


def _format_float(value: Any) -> str:
    scalar = float(value)
    if np.isnan(scalar):
        return "nan"
    return f"{scalar:.6f}"


def _normalize_metric_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.copy()


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


__all__ = [
    "DEFAULT_DASH_PORT",
    "build_inference_figure",
    "prepare_inference_payload",
    "push_inference_metrics",
    "start_dashboard_server",
]