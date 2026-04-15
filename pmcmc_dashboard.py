"""Live PMMH-CPM dashboard.

Provides a Dash-based dashboard that shows live MCMC traces and
marginal histograms updated every two seconds.

Usage inside pmcmc (called automatically when dashboard=True)::

    from pmcmc_dashboard import start_dashboard_server, push_sample
    start_dashboard_server(param_names, port=8050)
    push_sample(theta, loglik, n_accepted, step, is_burnin=False)

The Dash server runs in a background daemon thread and does not block
the MCMC loop.  Open http://localhost:<port>/ in a browser after the
server prints its startup message.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Shared state (written by the MCMC loop; read by Dash callbacks)
# ---------------------------------------------------------------------------

_MAX_HISTORY = 5000  # cap in-memory history to avoid unbounded growth

_state: dict = {
    "lock":          threading.Lock(),
    "param_names":   [],
    "chain":         [],        # list of np.ndarray (D,)
    "log_likelihoods": [],      # list of float
    "n_accepted":    0,
    "n_total":       0,
    "is_burnin":     True,
}

_server_started = False
_server_lock    = threading.Lock()


def push_sample(
    theta_constrained: np.ndarray,
    loglik: float,
    n_accepted: int,
    step: int,
    is_burnin: bool = False,
) -> None:
    """Push a new sample into the dashboard state.

    Thread-safe; designed to be called from inside the MCMC loop.
    """
    with _state["lock"]:
        _state["chain"].append(np.asarray(theta_constrained, dtype=np.float32))
        _state["log_likelihoods"].append(float(loglik))
        _state["n_accepted"] = n_accepted
        _state["n_total"]    = step
        _state["is_burnin"]  = is_burnin
        # Cap history to avoid unbounded growth
        if len(_state["chain"]) > _MAX_HISTORY:
            _state["chain"].pop(0)
            _state["log_likelihoods"].pop(0)


def start_dashboard_server(
    param_names: Sequence[str],
    port: int = 8050,
) -> None:
    """Launch the Dash server in a background daemon thread (idempotent)."""
    global _server_started

    with _server_lock:
        if _server_started:
            return
        _server_started = True

    with _state["lock"]:
        _state["param_names"] = list(param_names)

    app = _build_app(param_names)

    def _run() -> None:
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name="pmmh-dashboard")
    t.start()
    print(f"[Dashboard] Serving at http://localhost:{port}/")


# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------

def _build_app(param_names: Sequence[str]):
    """Construct and return the Dash application object."""
    try:
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "The dashboard requires dash and plotly.  "
            "Install them with:  pip install dash plotly"
        ) from exc

    D    = len(param_names)
    cols = min(4, D)
    rows = (D + cols - 1) // cols

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "padding": "10px"},
        children=[
            html.H2("PMMH-CPM Live Dashboard"),
            # ── Statistics bar ──────────────────────────────────────────
            html.Div(id="stats-bar", style={"marginBottom": "8px"}),
            # ── Trace plots ─────────────────────────────────────────────
            html.H4("Trace"),
            dcc.Graph(id="trace-plot", style={"height": f"{180 * rows + 60}px"}),
            # ── Histogram plots ─────────────────────────────────────────
            html.H4("Marginal posteriors"),
            dcc.Graph(id="hist-plot", style={"height": f"{180 * rows + 60}px"}),
            # ── Log-likelihood trace ─────────────────────────────────────
            html.H4("Log-likelihood"),
            dcc.Graph(id="loglik-plot", style={"height": "200px"}),
            # ── Refresh interval ────────────────────────────────────────
            dcc.Interval(id="interval", interval=2_000, n_intervals=0),
        ],
    )

    # ── Callbacks ────────────────────────────────────────────────────────

    @app.callback(
        Output("stats-bar",   "children"),
        Output("trace-plot",  "figure"),
        Output("hist-plot",   "figure"),
        Output("loglik-plot", "figure"),
        Input("interval",     "n_intervals"),
    )
    def _refresh(_n):
        with _state["lock"]:
            chain  = list(_state["chain"])
            logliks = list(_state["log_likelihoods"])
            n_acc  = _state["n_accepted"]
            n_tot  = _state["n_total"]
            burnin = _state["is_burnin"]

        phase = "burn-in" if burnin else "sampling"
        ar    = n_acc / n_tot if n_tot > 0 else 0.0
        stats = html.Span(
            f"Step {n_tot} | Phase: {phase} | "
            f"Acceptance rate: {ar:.2%} | Samples stored: {len(chain)}",
            style={"fontWeight": "bold"},
        )

        if not chain:
            _empty = go.Figure()
            return stats, _empty, _empty, _empty

        arr = np.stack(chain, axis=0)  # (M, D)

        # — Trace plots —
        trace_fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=param_names,
            shared_xaxes=False,
        )
        for j, name in enumerate(param_names):
            r, c = divmod(j, cols)
            trace_fig.add_trace(
                go.Scatter(
                    y=arr[:, j].tolist(),
                    mode="lines",
                    line={"width": 1},
                    name=name,
                    showlegend=False,
                ),
                row=r + 1, col=c + 1,
            )
        trace_fig.update_layout(margin={"t": 40, "b": 20, "l": 40, "r": 10})

        # — Histogram plots —
        hist_fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=param_names,
        )
        for j, name in enumerate(param_names):
            r, c = divmod(j, cols)
            hist_fig.add_trace(
                go.Histogram(
                    x=arr[:, j].tolist(),
                    nbinsx=40,
                    name=name,
                    showlegend=False,
                ),
                row=r + 1, col=c + 1,
            )
        hist_fig.update_layout(margin={"t": 40, "b": 20, "l": 40, "r": 10})

        # — Log-likelihood —
        ll_fig = go.Figure(
            go.Scatter(
                y=logliks,
                mode="lines",
                line={"width": 1},
            )
        )
        ll_fig.update_layout(
            yaxis_title="log p̂(y|θ,z)",
            margin={"t": 20, "b": 20, "l": 60, "r": 10},
        )

        return stats, trace_fig, hist_fig, ll_fig

    return app
