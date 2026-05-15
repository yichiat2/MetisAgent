from __future__ import annotations

import threading
from typing import Any, Mapping

import numpy as np


DEFAULT_DASH_PORT = 8051

state_store: dict[str, Any] = {
    "lock": threading.Lock(),
    "payload": None,
}

server_started = False
server_lock = threading.Lock()


def prepare_inference_payload(
    metrics: Mapping[str, Any],
    ohlc: Any,
    atr: Any,
    atr_up: Any,
    atr_down: Any,
    atr_imbalance: Any,
    fold_id: int,
    dataset_label: str,
) -> dict[str, Any]:
    host_metrics = {key: normalize_metric_value(value) for key, value in metrics.items()}
    aligned_ohlc = np.asarray(ohlc, dtype=np.float32)
    aligned_atr = np.asarray(atr, dtype=np.float32)
    aligned_atr_up = np.asarray(atr_up, dtype=np.float32)
    aligned_atr_down = np.asarray(atr_down, dtype=np.float32)
    aligned_atr_imbalance = np.asarray(atr_imbalance, dtype=np.float32)
    num_steps = np.asarray(host_metrics["pnl"], dtype=np.float32).shape[0]
    if aligned_ohlc.shape[0] != num_steps + 1:
        raise ValueError("Expected one extra OHLC bar so step metrics align to t + 1 bars")
    if aligned_atr.shape[0] != num_steps + 1:
        raise ValueError("Expected one extra ATR value so step metrics align to t + 1 bars")
    if aligned_atr_up.shape[0] != num_steps + 1:
        raise ValueError("Expected one extra ATR_Up value so step metrics align to t + 1 bars")
    if aligned_atr_down.shape[0] != num_steps + 1:
        raise ValueError("Expected one extra ATR_Down value so step metrics align to t + 1 bars")
    if aligned_atr_imbalance.shape[0] != num_steps + 1:
        raise ValueError("Expected one extra ATR imbalance value so step metrics align to t + 1 bars")
    return {
        "fold_id": int(fold_id),
        "dataset_label": dataset_label,
        "metrics": host_metrics,
        "ohlc": aligned_ohlc[1:, :4],
        "atr": aligned_atr[:-1],
        "atr_up": aligned_atr_up[:-1],
        "atr_down": aligned_atr_down[:-1],
        "atr_imbalance": aligned_atr_imbalance[:-1],
        "entry_long_price": np.asarray(host_metrics["entry_long_price"], dtype=np.float32),
        "entry_short_price": np.asarray(host_metrics["entry_short_price"], dtype=np.float32),
        "exit_long_price": np.asarray(host_metrics["exit_long_price"], dtype=np.float32),
        "exit_short_price": np.asarray(host_metrics["exit_short_price"], dtype=np.float32),
        "step_pnl": np.asarray(host_metrics["pnl"], dtype=np.float32),
        "cumulative_pnl": np.asarray(host_metrics["cumulative_pnl"], dtype=np.float32),
        "entry_long_threshold": float(host_metrics["entry_long"]),
        "entry_short_threshold": float(host_metrics["entry_short"]),
        "k_init": float(host_metrics["k_init"]),
        "a_tp": float(host_metrics["a_tp"]),
        "k_act": float(host_metrics["k_act"]),
    }


def push_inference_metrics(payload: Mapping[str, Any]) -> None:
    with state_store["lock"]:
        state_store["payload"] = {key: copy_value(value) for key, value in payload.items()}


def start_dashboard_server(port: int = DEFAULT_DASH_PORT) -> None:
    global server_started

    with server_lock:
        if server_started:
            return
        app = build_app()
        server_started = True

    def run_server() -> None:
        import logging

        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run_server, daemon=True, name="ppo-vol-scalping-dashboard")
    thread.start()
    print(f"[Dashboard] Serving PPO vol scalping dashboard at http://localhost:{port}/")


def build_app():
    try:
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "The PPO vol scalping dashboard requires dash and plotly. Install them with: pip install dash plotly"
        ) from exc

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "padding": "12px"},
        children=[
            html.H2("ATR Imbalance Grid Search Dashboard"),
            html.Div(id="stats-bar", style={"marginBottom": "10px", "fontWeight": "bold"}),
            dcc.Graph(id="inference-plot", style={"height": "1180px"}),
            dcc.Interval(id="interval", interval=2_000, n_intervals=0),
        ],
    )

    @app.callback(
        Output("stats-bar", "children"),
        Output("inference-plot", "figure"),
        Input("interval", "n_intervals"),
    )
    def refresh(_n):
        with state_store["lock"]:
            payload = state_store["payload"]

        if payload is None:
            return "Waiting for inference metrics...", go.Figure()

        return build_stats_bar(payload, html), build_inference_figure(payload)

    return app


def build_inference_figure(payload: Mapping[str, Any]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ohlc = np.asarray(payload["ohlc"], dtype=np.float32)
    entry_long_price = np.asarray(payload["entry_long_price"], dtype=np.float32)
    entry_short_price = np.asarray(payload["entry_short_price"], dtype=np.float32)
    exit_long_price = np.asarray(payload["exit_long_price"], dtype=np.float32)
    exit_short_price = np.asarray(payload["exit_short_price"], dtype=np.float32)
    step_pnl = np.asarray(payload["step_pnl"], dtype=np.float32)
    cumulative_pnl = np.asarray(payload["cumulative_pnl"], dtype=np.float32)
    atr = np.asarray(payload["atr"], dtype=np.float32)
    atr_up = np.asarray(payload["atr_up"], dtype=np.float32)
    atr_down = np.asarray(payload["atr_down"], dtype=np.float32)
    atr_imbalance = np.asarray(payload["atr_imbalance"], dtype=np.float32)
    timestep = np.arange(ohlc.shape[0], dtype=np.int32)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.46, 0.22, 0.16, 0.16],
        specs=[[{}], [{"secondary_y": True}], [{}], [{}]],
        subplot_titles=(
            "OHLC with entry and exit markers",
            "Cumulative PnL and step PnL",
            "ATR imbalance with entry thresholds",
            "ATR_Up, ATR_Down, and ATR",
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
    )
    fig.add_trace(
        go.Scatter(
            x=timestep[np.isfinite(entry_long_price)],
            y=entry_long_price[np.isfinite(entry_long_price)],
            mode="markers",
            name="Long entry",
            marker={"symbol": "triangle-up", "size": 11, "color": "#15803d"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep[np.isfinite(entry_short_price)],
            y=entry_short_price[np.isfinite(entry_short_price)],
            mode="markers",
            name="Short entry",
            marker={"symbol": "triangle-down", "size": 11, "color": "#b91c1c"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep[np.isfinite(exit_long_price)],
            y=exit_long_price[np.isfinite(exit_long_price)],
            mode="markers",
            name="Long exit / TP-SL",
            marker={"symbol": "triangle-down", "size": 10, "color": "#065f46"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep[np.isfinite(exit_short_price)],
            y=exit_short_price[np.isfinite(exit_short_price)],
            mode="markers",
            name="Short exit / TP-SL",
            marker={"symbol": "triangle-up", "size": 10, "color": "#7f1d1d"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=cumulative_pnl,
            mode="lines",
            name="Cumulative PnL",
            line={"color": "#111827", "width": 2.0},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=timestep,
            y=step_pnl,
            name="Step PnL",
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
            name="ATR imbalance",
            line={"color": "#1d4ed8", "width": 1.8},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=np.full_like(atr_imbalance, payload["entry_long_threshold"]),
            mode="lines",
            name="EntryLong",
            line={"color": "#6b7280", "width": 1.2, "dash": "dash"},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=np.full_like(atr_imbalance, payload["entry_short_threshold"]),
            mode="lines",
            name="EntryShort",
            line={"color": "#6b7280", "width": 1.2, "dash": "dash"},
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=atr_up,
            mode="lines",
            name="ATR_Up",
            line={"color": "#15803d", "width": 1.8},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=atr_down,
            mode="lines",
            name="ATR_Down",
            line={"color": "#b91c1c", "width": 1.8},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestep,
            y=atr,
            mode="lines",
            name="ATR",
            line={"color": "#0f766e", "width": 2.0},
        ),
        row=4,
        col=1,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Step PnL", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="ATR imbalance", row=3, col=1)
    fig.update_yaxes(title_text="ATR signals", row=4, col=1)
    fig.update_xaxes(title_text="Timestep", row=4, col=1)
    fig.update_layout(
        template="plotly_white",
        height=1180,
        hovermode="x unified",
        margin={"t": 70, "b": 40, "l": 60, "r": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        title=(
            f"Fold {payload['fold_id']} {payload['dataset_label']} | "
            f"k_init={payload['k_init']:.2f}, a_tp={payload['a_tp']:.2f}, k_act={payload['k_act']:.2f}, "
            f"EntryLong={payload['entry_long_threshold']:.2f}, "
            f"EntryShort={payload['entry_short_threshold']:.2f}"
        ),
        xaxis_rangeslider_visible=False,
    )
    return fig


def build_stats_bar(payload: Mapping[str, Any], html: Any):
    metrics = payload["metrics"]
    return html.Div(
        (
            f"Fold {payload['fold_id']} {payload['dataset_label']} | "
            f"k_init {payload['k_init']:.2f} | a_tp {payload['a_tp']:.2f} | "
            f"k_act {payload['k_act']:.2f} | EntryLong {payload['entry_long_threshold']:.2f} | "
            f"EntryShort {payload['entry_short_threshold']:.2f} | "
            f"Sharpe {format_float(metrics['sharpe_ratio'])} | "
            f"Sortino {format_float(metrics['sortino_ratio'])} | "
            f"Cumulative return {format_float(metrics['final_cumulative_return'])} | "
            f"Max drawdown {format_float(metrics['max_drawdown'])} | "
            f"Trades {int(metrics['trade_count'])} | "
            f"Win rate {format_float(metrics['win_rate'])} | "
            f"Expectation {format_float(metrics['expectation_per_trade'])} | "
            f"Bankruptcy {bool(metrics['bankruptcy'])}"
        )
    )


def format_float(value: Any) -> str:
    scalar = float(value)
    if np.isnan(scalar):
        return "nan"
    return f"{scalar:.6f}"


def normalize_metric_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.copy()


def copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: copy_value(item) for key, item in value.items()}
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