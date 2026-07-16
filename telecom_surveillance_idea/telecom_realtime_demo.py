"""
Real-Time Visualization Demo - Italian Telecom Dataset (Milan, Nov 2013)

Replays the telecom activity data as if it's happening live.
Uses Dash + Plotly to render a heatmap grid of Milan showing
SMS, Call, and Internet activity over time.

Usage:
    python telecom_realtime_demo.py

Then open http://127.0.0.1:8050 in your browser.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "ocanaydin", "italian-telecom-data-2013-1week", "versions", "2"
)

# Milan grid is 100x100 (CellIDs 1-10000)
GRID_SIZE = 100

# How many real-data minutes pass per tick (10 min = 1 data interval)
DATA_STEP_MINUTES = 10

# ---------------------------------------------------------------------------
# Data Loading (sample for performance - aggregate by cell, ignore country code)
# ---------------------------------------------------------------------------
print("Loading dataset (this may take a moment)...")

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "sms-call-internet-mi-*.csv")))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

# Load a subset for demo performance - first 2 days
# Each file is ~350MB so we sample to keep memory reasonable
frames = []
for f in csv_files[:2]:
    print(f"  Loading {os.path.basename(f)}...")
    df = pd.read_csv(
        f,
        usecols=["CellID", "datetime", "smsin", "smsout", "callin", "callout", "internet"],
        dtype={"CellID": "int32"},
    )
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
del frames

# Convert epoch ms to datetime
data["timestamp"] = pd.to_datetime(data["datetime"], unit="ms")
data.drop(columns=["datetime"], inplace=True)

# Aggregate by CellID and timestamp (sum across country codes)
print("Aggregating data...")
agg_data = data.groupby(["CellID", "timestamp"], as_index=False).agg({
    "smsin": "sum",
    "smsout": "sum",
    "callin": "sum",
    "callout": "sum",
    "internet": "sum",
})
del data

# Fill NaN with 0
agg_data.fillna(0, inplace=True)

# Get sorted unique timestamps
timestamps = sorted(agg_data["timestamp"].unique())
print(f"Loaded {len(timestamps)} time steps across {len(agg_data)} records.")
print(f"Time range: {timestamps[0]} to {timestamps[-1]}")

# Pre-compute grids for each timestamp for fast lookup
print("Pre-computing grids...")
grid_cache = {}
for ts in timestamps:
    slice_df = agg_data[agg_data["timestamp"] == ts]
    grid_cache[ts] = slice_df.set_index("CellID")

print("Ready! Starting server...\n")

# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("Milan Telecom Activity - Real-Time Replay", style={"textAlign": "center"}),
        html.P(
            "Replaying SMS, Call, and Internet activity across Milan's 100x100 grid (Nov 2013)",
            style={"textAlign": "center", "color": "#666"},
        ),

        # Controls row
        html.Div(
            style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "20px", "marginBottom": "20px"},
            children=[
                html.Button("Play", id="play-btn", n_clicks=0,
                            style={"fontSize": "16px", "padding": "10px 20px", "cursor": "pointer"}),
                html.Button("Pause", id="pause-btn", n_clicks=0,
                            style={"fontSize": "16px", "padding": "10px 20px", "cursor": "pointer"}),
                html.Button("Reset", id="reset-btn", n_clicks=0,
                            style={"fontSize": "16px", "padding": "10px 20px", "cursor": "pointer"}),
                html.Div([
                    html.Label("Speed (ms/frame): "),
                    dcc.Input(id="speed-input", type="number", value=500, min=100, max=5000, step=100,
                              style={"width": "80px"}),
                ]),
                html.Div([
                    html.Label("Metric: "),
                    dcc.Dropdown(
                        id="metric-dropdown",
                        options=[
                            {"label": "Internet", "value": "internet"},
                            {"label": "SMS In", "value": "smsin"},
                            {"label": "SMS Out", "value": "smsout"},
                            {"label": "Calls In", "value": "callin"},
                            {"label": "Calls Out", "value": "callout"},
                            {"label": "Total Activity", "value": "total"},
                        ],
                        value="internet",
                        clearable=False,
                        style={"width": "160px"},
                    ),
                ]),
            ],
        ),

        # Time display
        html.Div(id="time-display", style={"textAlign": "center", "fontSize": "20px", "fontWeight": "bold", "marginBottom": "10px"}),

        # Progress slider
        dcc.Slider(
            id="time-slider",
            min=0,
            max=len(timestamps) - 1,
            value=0,
            step=1,
            marks={i: "" for i in range(0, len(timestamps), max(1, len(timestamps) // 10))},
            tooltip={"placement": "bottom"},
        ),

        # Heatmap
        dcc.Graph(id="heatmap", style={"height": "700px"}),

        # Interval for auto-play
        dcc.Interval(id="interval", interval=500, n_intervals=0, disabled=True),

        # Store for play state
        dcc.Store(id="play-state", data={"playing": False, "step": 0}),
    ],
)


def build_grid(step_idx, metric):
    """Build a 100x100 grid for the given time step and metric."""
    ts = timestamps[step_idx]
    slice_df = grid_cache.get(ts)

    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    if slice_df is not None and len(slice_df) > 0:
        if metric == "total":
            values = slice_df[["smsin", "smsout", "callin", "callout", "internet"]].sum(axis=1)
        else:
            values = slice_df[metric] if metric in slice_df.columns else pd.Series(0, index=slice_df.index)

        for cell_id, val in values.items():
            if 1 <= cell_id <= 10000:
                row = (cell_id - 1) // GRID_SIZE
                col = (cell_id - 1) % GRID_SIZE
                grid[row, col] = val

    return grid


@app.callback(
    Output("play-state", "data"),
    Output("interval", "disabled"),
    Output("interval", "interval"),
    Input("play-btn", "n_clicks"),
    Input("pause-btn", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    Input("speed-input", "value"),
    State("play-state", "data"),
    prevent_initial_call=True,
)
def control_playback(play_clicks, pause_clicks, reset_clicks, speed, state):
    triggered = callback_context.triggered[0]["prop_id"]
    interval_ms = speed if speed and speed >= 100 else 500

    if "play-btn" in triggered:
        state["playing"] = True
        return state, False, interval_ms
    elif "pause-btn" in triggered:
        state["playing"] = False
        return state, True, interval_ms
    elif "reset-btn" in triggered:
        state["playing"] = False
        state["step"] = 0
        return state, True, interval_ms
    elif "speed-input" in triggered:
        return state, not state["playing"], interval_ms

    return state, True, interval_ms


@app.callback(
    Output("time-slider", "value"),
    Output("play-state", "data", allow_duplicate=True),
    Input("interval", "n_intervals"),
    State("play-state", "data"),
    prevent_initial_call=True,
)
def advance_step(n_intervals, state):
    if state["playing"]:
        state["step"] = (state["step"] + 1) % len(timestamps)
    return state["step"], state


@app.callback(
    Output("heatmap", "figure"),
    Output("time-display", "children"),
    Input("time-slider", "value"),
    Input("metric-dropdown", "value"),
)
def update_heatmap(step_idx, metric):
    grid = build_grid(step_idx, metric)
    ts = timestamps[step_idx]

    # Format timestamp
    ts_dt = pd.Timestamp(ts)
    time_str = ts_dt.strftime("%A, %B %d, %Y  %H:%M")

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            colorscale="Hot",
            zmin=0,
            zmax=np.percentile(grid[grid > 0], 95) if np.any(grid > 0) else 1,
            colorbar=dict(title=metric.capitalize()),
        )
    )
    fig.update_layout(
        title=f"Milan Grid - {metric.capitalize()} Activity",
        xaxis=dict(title="Column", showgrid=False, zeroline=False),
        yaxis=dict(title="Row", showgrid=False, zeroline=False, autorange="reversed"),
        margin=dict(l=50, r=50, t=60, b=50),
    )

    return fig, f"[{time_str}]"


if __name__ == "__main__":
    app.run(debug=False, port=8050)
