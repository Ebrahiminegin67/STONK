"""
Infrastructure Health Monitoring MVP
=====================================
Adaptive path model using GPS + accelerometer data to detect road issues.

Supports:
- PVS (Passive Vehicular Sensors) dataset from Kaggle (auto-download)
- Phyphox CSV exports for live demos
- Synthetic demo data

Features:
- Adaptive expected-path model (EMA-based spatial grid)
- Route deviation detection (cars avoiding segments)
- Pothole/obstruction inference (accelerometer spikes)
- Dual time-interval comparison (day vs week vs month)
- Interactive map canvas with highlighted anomalies
- Click-to-inspect anomaly details

Usage:
    python infra_health_mvp.py                    # synthetic demo
    python infra_health_mvp.py --kaggle           # download PVS from Kaggle
    python infra_health_mvp.py --data-dir PATH    # local CSV directory
    python infra_health_mvp.py --phyphox FILE     # single phyphox CSV

Then open http://127.0.0.1:8051 in your browser.
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import signal

import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRID_RESOLUTION = 0.0005  # ~55m grid cells (lat/lon degrees)
EMA_ALPHA = 0.05  # Exponential moving average weight for new observations
ACCEL_SPIKE_THRESHOLD = 2.5  # g-force threshold for pothole detection
DEVIATION_Z_THRESHOLD = 2.0  # z-score for route deviation
MIN_OBSERVATIONS = 5  # Minimum obs before a cell is considered "established"
PORT = 8051

# ---------------------------------------------------------------------------
# Data Schema Mappings
# ---------------------------------------------------------------------------
PVS_COLUMNS = {
    "timestamp": "timestamp",
    "latitude": "latitude",
    "longitude": "longitude",
    "speed": "speed",
    "acc_x": "acc_x",
    "acc_y": "acc_y",
    "acc_z": "acc_z",
    "gps_lat": "latitude",
    "gps_lon": "longitude",
    "gps_long": "longitude",
    "accel_x": "acc_x",
    "accel_y": "acc_y",
    "accel_z": "acc_z",
    "accelerometer_x": "acc_x",
    "accelerometer_y": "acc_y",
    "accelerometer_z": "acc_z",
    "lat": "latitude",
    "lon": "longitude",
    "long": "longitude",
}

PHYPHOX_COLUMNS = {
    "Time (s)": "timestamp",
    "Linear Acceleration x (m/s^2)": "acc_x",
    "Linear Acceleration y (m/s^2)": "acc_y",
    "Linear Acceleration z (m/s^2)": "acc_z",
    "Location Latitude (deg)": "latitude",
    "Location Longitude (deg)": "longitude",
    "Location Speed (m/s)": "speed",
    "time": "timestamp",
    "accX": "acc_x",
    "accY": "acc_y",
    "accZ": "acc_z",
    "lat": "latitude",
    "lon": "longitude",
}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def lat_lon_to_grid(lat, lon, resolution=GRID_RESOLUTION):
    """Convert lat/lon to grid cell indices."""
    row = int(np.floor(lat / resolution))
    col = int(np.floor(lon / resolution))
    return (row, col)


def grid_to_lat_lon(row, col, resolution=GRID_RESOLUTION):
    """Convert grid cell back to center lat/lon."""
    lat = (row + 0.5) * resolution
    lon = (col + 0.5) * resolution
    return lat, lon


def compute_rms_acceleration(acc_x, acc_y, acc_z):
    """Compute RMS of 3-axis acceleration."""
    return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)



# ---------------------------------------------------------------------------
# Kaggle Dataset Download
# ---------------------------------------------------------------------------
def download_pvs_kaggle():
    """Download the PVS dataset from Kaggle using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    print("Downloading PVS dataset from Kaggle...")
    print("  (This may take a while on first run)")
    path = kagglehub.dataset_download("jefmenegazzo/pvs-passive-vehicular-sensors-datasets")
    print(f"  Path to dataset files: {path}")
    return path


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def detect_format(df):
    """Detect whether a CSV is PVS format or Phyphox format."""
    cols_lower = set(c.lower().strip() for c in df.columns)
    if "linear acceleration x (m/s^2)" in cols_lower or "accx" in cols_lower:
        return "phyphox"
    if "time (s)" in cols_lower:
        return "phyphox"
    return "pvs"


def normalize_columns(df, fmt):
    """Rename columns to standard internal names."""
    mapping = PHYPHOX_COLUMNS if fmt == "phyphox" else PVS_COLUMNS

    col_map = {}
    for orig_col in df.columns:
        for pattern, target in mapping.items():
            if orig_col.strip().lower() == pattern.lower():
                col_map[orig_col] = target
                break

    df = df.rename(columns=col_map)
    return df


def load_csv_file(filepath, max_rows=None):
    """Load a single CSV file and normalize it."""
    try:
        df = pd.read_csv(filepath, nrows=max_rows)
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return None

    if len(df) == 0:
        return None

    fmt = detect_format(df)
    df = normalize_columns(df, fmt)

    # Ensure required columns exist
    required = ["latitude", "longitude"]
    for col in required:
        if col not in df.columns:
            # Try to find columns that contain lat/lon in name
            for c in df.columns:
                if "lat" in c.lower() and "latitude" not in df.columns:
                    df = df.rename(columns={c: "latitude"})
                elif "lon" in c.lower() and "longitude" not in df.columns:
                    df = df.rename(columns={c: "longitude"})

    for col in required:
        if col not in df.columns:
            return None

    # Fill missing accel with 0
    for col in ["acc_x", "acc_y", "acc_z"]:
        if col not in df.columns:
            df[col] = 0.0

    if "speed" not in df.columns:
        df["speed"] = 0.0

    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))

    # Convert timestamp to datetime if numeric
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        if df["timestamp"].max() > 1e12:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        elif df["timestamp"].max() > 1e9:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        else:
            df["datetime"] = pd.Timestamp.now() + pd.to_timedelta(df["timestamp"], unit="s")
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows with invalid GPS
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df = df[(df["latitude"].abs() <= 90) & (df["longitude"].abs() <= 180)]

    if len(df) == 0:
        return None

    # Convert accel to numeric
    for col in ["acc_x", "acc_y", "acc_z", "speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Add RMS acceleration
    df["acc_rms"] = compute_rms_acceleration(df["acc_x"], df["acc_y"], df["acc_z"])

    # Source file tag
    df["source"] = os.path.basename(filepath)

    return df


def load_data_directory(data_dir, max_files=50, max_rows_per_file=50000):
    """Load CSV files from a directory (with limits for performance)."""
    frames = []
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame()

    csv_files = sorted(csv_files)[:max_files]
    print(f"  Found {len(csv_files)} CSV files (loading up to {max_files})")

    for filepath in csv_files:
        print(f"  Loading {os.path.basename(filepath)}...")
        df = load_csv_file(filepath, max_rows=max_rows_per_file)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    print(f"  Total rows loaded: {len(result):,}")
    return result



# ---------------------------------------------------------------------------
# Adaptive Path Model
# ---------------------------------------------------------------------------
class AdaptivePathModel:
    """
    Maintains an adaptive expected-path model using spatial grid cells.
    Each cell tracks average traffic and acceleration via EMA.
    """

    def __init__(self, resolution=GRID_RESOLUTION, alpha=EMA_ALPHA):
        self.resolution = resolution
        self.alpha = alpha
        self.cells = defaultdict(lambda: {
            "count": 0,
            "avg_acc_rms": 0.0,
            "acc_history": [],
            "observations": 0,
            "last_seen": None,
        })

    def update(self, lat, lon, acc_rms, timestamp=None):
        """Update a cell with a new observation."""
        cell = lat_lon_to_grid(lat, lon, self.resolution)
        state = self.cells[cell]
        state["observations"] += 1
        state["count"] += 1
        state["last_seen"] = timestamp

        if state["observations"] == 1:
            state["avg_acc_rms"] = acc_rms
        else:
            state["avg_acc_rms"] = (
                self.alpha * acc_rms + (1 - self.alpha) * state["avg_acc_rms"]
            )

        state["acc_history"].append(acc_rms)
        if len(state["acc_history"]) > 100:
            state["acc_history"] = state["acc_history"][-100:]

        return cell

    def update_batch(self, df):
        """Process an entire dataframe of observations."""
        for _, row in df.iterrows():
            self.update(row["latitude"], row["longitude"], row["acc_rms"], row.get("datetime"))

    def get_cell_stats(self):
        """Return all cells with their statistics as a DataFrame."""
        results = []
        for cell_key, state in self.cells.items():
            lat, lon = grid_to_lat_lon(cell_key[0], cell_key[1], self.resolution)
            results.append({
                "cell": cell_key,
                "latitude": lat,
                "longitude": lon,
                "observations": state["observations"],
                "avg_acc_rms": state["avg_acc_rms"],
                "acc_std": np.std(state["acc_history"]) if len(state["acc_history"]) > 1 else 0,
                "last_seen": state["last_seen"],
            })
        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Detects road anomalies from the adaptive path model and raw data."""

    def __init__(self, spike_threshold=ACCEL_SPIKE_THRESHOLD,
                 deviation_threshold=DEVIATION_Z_THRESHOLD):
        self.spike_threshold = spike_threshold
        self.deviation_threshold = deviation_threshold

    def detect_accel_spikes(self, df):
        """Find accelerometer spikes indicating potholes/bumps."""
        z_abs = df["acc_z"].abs()
        rms = df["acc_rms"]
        threshold = self.spike_threshold * 9.81

        spike_mask = (z_abs > threshold) | (rms > threshold)
        spikes = df[spike_mask].copy()
        spikes["anomaly_type"] = "pothole_candidate"
        spikes["severity"] = rms[spike_mask] / 9.81
        return spikes

    def detect_route_deviations(self, model, recent_df, historical_df):
        """Detect segments where recent traffic deviates from historical."""
        deviations = []

        recent_cells = defaultdict(int)
        for _, row in recent_df.iterrows():
            cell = lat_lon_to_grid(row["latitude"], row["longitude"])
            recent_cells[cell] += 1

        hist_cells = defaultdict(int)
        for _, row in historical_df.iterrows():
            cell = lat_lon_to_grid(row["latitude"], row["longitude"])
            hist_cells[cell] += 1

        recent_total = max(len(recent_df), 1)
        hist_total = max(len(historical_df), 1)

        for cell, hist_count in hist_cells.items():
            if hist_count < MIN_OBSERVATIONS:
                continue
            hist_rate = hist_count / hist_total
            recent_rate = recent_cells.get(cell, 0) / recent_total
            if hist_rate > 0:
                drop_ratio = (hist_rate - recent_rate) / hist_rate
                if drop_ratio > 0.5:
                    lat, lon = grid_to_lat_lon(cell[0], cell[1])
                    deviations.append({
                        "latitude": lat,
                        "longitude": lon,
                        "drop_ratio": drop_ratio,
                        "anomaly_type": "route_deviation",
                        "severity": drop_ratio,
                    })

        return pd.DataFrame(deviations) if deviations else pd.DataFrame()

    def detect_all(self, model, df, time_split_ratio=0.7):
        """Run all detection methods and return combined anomalies."""
        if len(df) == 0:
            return pd.DataFrame(columns=["latitude", "longitude", "anomaly_type", "severity", "datetime"])

        split_idx = int(len(df) * time_split_ratio)
        historical = df.iloc[:split_idx]
        recent = df.iloc[split_idx:]

        spikes = self.detect_accel_spikes(df)
        deviations = self.detect_route_deviations(model, recent, historical)

        all_anomalies = []
        if len(spikes) > 0:
            all_anomalies.append(
                spikes[["latitude", "longitude", "anomaly_type", "severity", "datetime"]].copy()
            )
        if len(deviations) > 0:
            deviations["datetime"] = pd.NaT
            all_anomalies.append(
                deviations[["latitude", "longitude", "anomaly_type", "severity", "datetime"]].copy()
            )

        if all_anomalies:
            return pd.concat(all_anomalies, ignore_index=True)
        return pd.DataFrame(columns=["latitude", "longitude", "anomaly_type", "severity", "datetime"])



# ---------------------------------------------------------------------------
# Demo Data Generator
# ---------------------------------------------------------------------------
def generate_demo_data(num_trips=50, points_per_trip=200):
    """Generate synthetic GPS + accelerometer data with potholes and deviations."""
    np.random.seed(42)
    base_lat, base_lon = 59.33, 18.07
    all_data = []
    start_time = datetime(2024, 1, 1, 8, 0, 0)

    for trip_id in range(num_trips):
        trip_time = start_time + timedelta(hours=trip_id * 0.5)
        t = np.linspace(0, 1, points_per_trip)

        route_lat = base_lat + 0.01 * t + np.random.normal(0, 0.0001, points_per_trip)
        route_lon = base_lon + 0.015 * t + np.random.normal(0, 0.0001, points_per_trip)

        speed = np.clip(30 + 10 * np.sin(2 * np.pi * t) + np.random.normal(0, 2, points_per_trip), 5, 60) / 3.6
        acc_x = np.random.normal(0, 0.5, points_per_trip)
        acc_y = np.random.normal(0, 0.5, points_per_trip)
        acc_z = np.random.normal(0, 0.8, points_per_trip)

        if trip_id > 30:
            idx = int(points_per_trip * 0.4)
            acc_z[idx:idx+3] += np.array([15, 25, 10])
            acc_x[idx:idx+3] += np.array([5, 8, 3])
            idx2 = int(points_per_trip * 0.7)
            acc_z[idx2:idx2+2] += np.array([20, 12])

        if trip_id > 35:
            ds, de = int(points_per_trip * 0.5), int(points_per_trip * 0.6)
            route_lat[ds:de] += 0.002
            route_lon[ds:de] -= 0.001

        timestamps = [trip_time + timedelta(seconds=i * 0.5) for i in range(points_per_trip)]
        trip_df = pd.DataFrame({
            "timestamp": [ts.timestamp() for ts in timestamps],
            "latitude": route_lat, "longitude": route_lon, "speed": speed,
            "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z, "trip_id": trip_id,
        })
        all_data.append(trip_df)

    df = pd.concat(all_data, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["acc_rms"] = compute_rms_acceleration(df["acc_x"], df["acc_y"], df["acc_z"])
    df["source"] = "demo_synthetic"
    print(f"Generated {num_trips} demo trips, {len(df)} data points total.")
    return df


# ---------------------------------------------------------------------------
# Time Interval Analysis
# ---------------------------------------------------------------------------
def compute_interval_stats(df, interval="day"):
    """Compute statistics grouped by time interval."""
    if len(df) == 0 or "datetime" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    if interval == "day":
        df["interval"] = df["datetime"].dt.strftime("%Y-%m-%d")
    elif interval == "week":
        df["interval"] = df["datetime"].dt.strftime("W%U-%Y")
    elif interval == "month":
        df["interval"] = df["datetime"].dt.strftime("%Y-%m")
    elif interval == "hour":
        df["interval"] = df["datetime"].dt.strftime("%H:00")
    else:
        df["interval"] = df["datetime"].dt.strftime("%Y-%m-%d")

    threshold = ACCEL_SPIKE_THRESHOLD * 9.81
    stats = df.groupby("interval", sort=True).agg(
        avg_acc_rms=("acc_rms", "mean"),
        max_acc_rms=("acc_rms", "max"),
        avg_speed=("speed", "mean"),
        num_points=("latitude", "count"),
        spike_count=("acc_rms", lambda x: int((x > threshold).sum())),
    ).reset_index()

    return stats



# ---------------------------------------------------------------------------
# Dash Application - Layout
# ---------------------------------------------------------------------------
def create_app(df, model, anomalies):
    """Create the Dash visualization app with interactive dual panels."""

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "20px",
               "maxWidth": "1500px", "margin": "0 auto", "backgroundColor": "#fafafa"},
        children=[
            # Header
            html.Div(style={"textAlign": "center", "marginBottom": "25px"}, children=[
                html.H1("Infrastructure Health Monitor",
                         style={"color": "#2c3e50", "marginBottom": "5px"}),
                html.P("Adaptive path model | GPS + Accelerometer anomaly detection",
                       style={"color": "#7f8c8d", "fontSize": "16px"}),
            ]),

            # Stats cards
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                       "gap": "15px", "marginBottom": "25px"},
                children=[
                    _stat_card("Data Points", f"{len(df):,}", "#2980b9"),
                    _stat_card("Grid Cells", f"{len(model.cells):,}", "#27ae60"),
                    _stat_card("Anomalies", f"{len(anomalies):,}", "#e74c3c"),
                    _stat_card("Potholes",
                               f"{len(anomalies[anomalies['anomaly_type']=='pothole_candidate']) if len(anomalies) > 0 else 0}",
                               "#f39c12"),
                ],
            ),

            # Controls
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 2fr",
                       "gap": "15px", "marginBottom": "25px",
                       "padding": "15px", "backgroundColor": "white",
                       "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
                children=[
                    html.Div([
                        html.Label("Map Layer:", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="map-layer",
                            options=[
                                {"label": "All Tracks + Anomalies", "value": "all"},
                                {"label": "Anomalies Only", "value": "anomalies"},
                                {"label": "Grid Heatmap", "value": "heatmap"},
                                {"label": "Acceleration Heatmap", "value": "acc_heatmap"},
                            ],
                            value="all", clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("Left Panel:", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="left-interval",
                            options=[
                                {"label": "Hourly", "value": "hour"},
                                {"label": "Daily", "value": "day"},
                                {"label": "Weekly", "value": "week"},
                                {"label": "Monthly", "value": "month"},
                            ],
                            value="day", clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("Right Panel:", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="right-interval",
                            options=[
                                {"label": "Hourly", "value": "hour"},
                                {"label": "Daily", "value": "day"},
                                {"label": "Weekly", "value": "week"},
                                {"label": "Monthly", "value": "month"},
                            ],
                            value="week", clearable=False,
                        ),
                    ]),
                    html.Div([
                        html.Label("Min Severity (g-force):", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Slider(
                            id="severity-slider",
                            min=0, max=5, step=0.25, value=0,
                            marks={i: f"{i}g" for i in range(6)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]),
                ],
            ),

            # Main Map
            html.Div(style={"backgroundColor": "white", "borderRadius": "8px",
                           "padding": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                           "marginBottom": "25px"}, children=[
                html.H3("Road Network Canvas", style={"color": "#2c3e50", "margin": "5px 0 10px 10px"}),
                dcc.Graph(id="main-map", style={"height": "550px"},
                          config={"scrollZoom": True, "displayModeBar": True}),
            ]),

            # Click info
            html.Div(id="click-info",
                     style={"padding": "10px", "backgroundColor": "#eaf2f8",
                            "borderRadius": "6px", "marginBottom": "25px",
                            "display": "none"}),

            # Dual Panels - side by side
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                           "gap": "20px", "marginBottom": "25px"}, children=[
                html.Div(style={"backgroundColor": "white", "borderRadius": "8px",
                               "padding": "15px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
                    html.H4("Left: Acceleration & Spikes", style={"color": "#2c3e50", "marginTop": "0"}),
                    dcc.Graph(id="left-panel", style={"height": "380px"},
                              config={"displayModeBar": False}),
                ]),
                html.Div(style={"backgroundColor": "white", "borderRadius": "8px",
                               "padding": "15px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
                    html.H4("Right: Speed & Traffic Volume", style={"color": "#2c3e50", "marginTop": "0"}),
                    dcc.Graph(id="right-panel", style={"height": "380px"},
                              config={"displayModeBar": False}),
                ]),
            ]),

            # Acceleration time series
            html.Div(style={"backgroundColor": "white", "borderRadius": "8px",
                           "padding": "15px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                           "marginBottom": "25px"}, children=[
                html.H4("Accelerometer Signal (sample)", style={"color": "#2c3e50", "marginTop": "0"}),
                dcc.Graph(id="accel-timeline", style={"height": "250px"},
                          config={"displayModeBar": True}),
            ]),

            # Anomaly table
            html.Div(style={"backgroundColor": "white", "borderRadius": "8px",
                           "padding": "15px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
                html.H4("Detected Anomalies", style={"color": "#2c3e50", "marginTop": "0"}),
                html.Div(id="anomaly-table", style={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"}),
            ]),

            # Hidden stores
            dcc.Store(id="data-store", data={"center_lat": center_lat, "center_lon": center_lon}),
        ],
    )

    return app


def _stat_card(title, value, color):
    """Create a stats card element."""
    return html.Div(
        style={"backgroundColor": "white", "padding": "20px", "borderRadius": "8px",
               "textAlign": "center", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
               "borderLeft": f"4px solid {color}"},
        children=[
            html.H2(value, style={"margin": "0", "color": color, "fontSize": "28px"}),
            html.P(title, style={"margin": "5px 0 0 0", "color": "#7f8c8d", "fontSize": "14px"}),
        ],
    )



# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def register_callbacks(app, df, model, anomalies):
    """Register all Dash callbacks for interactivity."""

    cell_stats = model.get_cell_stats()

    # --- Main Map ---
    @app.callback(
        Output("main-map", "figure"),
        Input("map-layer", "value"),
        Input("severity-slider", "value"),
        State("data-store", "data"),
    )
    def update_map(layer, min_severity, store):
        center_lat = store["center_lat"]
        center_lon = store["center_lon"]
        fig = go.Figure()

        if layer == "all":
            sample_size = min(8000, len(df))
            sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
            fig.add_trace(go.Scattermapbox(
                lat=sample["latitude"], lon=sample["longitude"],
                mode="markers",
                marker=dict(size=3, color="#3498db", opacity=0.3),
                name="GPS Tracks",
                hovertemplate="Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra>Track</extra>",
            ))

        elif layer == "heatmap":
            if len(cell_stats) > 0:
                fig.add_trace(go.Densitymapbox(
                    lat=cell_stats["latitude"], lon=cell_stats["longitude"],
                    z=cell_stats["observations"],
                    radius=15, colorscale="YlOrRd",
                    name="Traffic Density",
                    hovertemplate="Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<br>Observations: %{z}<extra></extra>",
                ))

        elif layer == "acc_heatmap":
            if len(cell_stats) > 0:
                fig.add_trace(go.Densitymapbox(
                    lat=cell_stats["latitude"], lon=cell_stats["longitude"],
                    z=cell_stats["avg_acc_rms"],
                    radius=15, colorscale="Inferno",
                    name="Avg Acceleration",
                    hovertemplate="Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<br>Avg RMS: %{z:.2f} m/s2<extra></extra>",
                ))

        # Always show anomalies (except in pure heatmap mode)
        if layer in ("all", "anomalies", "acc_heatmap") and len(anomalies) > 0:
            filtered = anomalies[anomalies["severity"] >= min_severity]

            potholes = filtered[filtered["anomaly_type"] == "pothole_candidate"]
            if len(potholes) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=potholes["latitude"], lon=potholes["longitude"],
                    mode="markers",
                    marker=dict(
                        size=np.clip(potholes["severity"].values * 3, 8, 22).tolist(),
                        color="#e74c3c", opacity=0.85,
                    ),
                    name="Pothole Candidates",
                    customdata=potholes["severity"].values,
                    hovertemplate="POTHOLE<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<br>Severity: %{customdata:.1f}g<extra></extra>",
                ))

            deviations = filtered[filtered["anomaly_type"] == "route_deviation"]
            if len(deviations) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=deviations["latitude"], lon=deviations["longitude"],
                    mode="markers",
                    marker=dict(size=14, color="#f39c12", opacity=0.85, symbol="diamond"),
                    name="Route Deviations",
                    customdata=deviations["severity"].values,
                    hovertemplate="DEVIATION<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<br>Drop: %{customdata:.0%}<extra></extra>",
                ))

        fig.update_layout(
            mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=13),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)",
                       bordercolor="#ddd", borderwidth=1),
            clickmode="event+select",
        )
        return fig

    # --- Click info ---
    @app.callback(
        Output("click-info", "children"),
        Output("click-info", "style"),
        Input("main-map", "clickData"),
        prevent_initial_call=True,
    )
    def show_click_info(click_data):
        if not click_data:
            return "", {"display": "none"}

        point = click_data["points"][0]
        lat = point.get("lat", "N/A")
        lon = point.get("lon", "N/A")
        trace_name = point.get("curveNumber", 0)

        info_style = {"padding": "12px", "backgroundColor": "#eaf2f8",
                      "borderRadius": "6px", "marginBottom": "25px",
                      "borderLeft": "4px solid #2980b9"}

        content = [
            html.Strong("Clicked Point: "),
            html.Span(f"Lat {lat:.5f}, Lon {lon:.5f}  |  "),
        ]

        if "customdata" in point and point["customdata"] is not None:
            content.append(html.Span(f"Severity: {point['customdata']:.2f}g",
                                    style={"color": "#e74c3c", "fontWeight": "bold"}))

        # Find nearby anomalies
        if len(anomalies) > 0:
            nearby = anomalies[
                (anomalies["latitude"].between(lat - 0.001, lat + 0.001)) &
                (anomalies["longitude"].between(lon - 0.001, lon + 0.001))
            ]
            if len(nearby) > 0:
                content.append(html.Br())
                content.append(html.Span(
                    f"Nearby anomalies: {len(nearby)} ({len(nearby[nearby['anomaly_type']=='pothole_candidate'])} potholes, "
                    f"{len(nearby[nearby['anomaly_type']=='route_deviation'])} deviations)",
                    style={"fontSize": "13px", "color": "#555"}
                ))

        return content, info_style


    # --- Left Panel ---
    @app.callback(
        Output("left-panel", "figure"),
        Input("left-interval", "value"),
    )
    def update_left_panel(interval):
        stats = compute_interval_stats(df, interval)
        if len(stats) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data for this interval", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False, font_size=16)
            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
            return fig

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=["Avg Acceleration RMS (m/s2)", "Spike Count (potential issues)"],
        )

        # Acceleration bars with color gradient
        colors = ["#e74c3c" if v > ACCEL_SPIKE_THRESHOLD * 9.81 * 0.5 else "#3498db"
                  for v in stats["avg_acc_rms"]]
        fig.add_trace(
            go.Bar(x=stats["interval"], y=stats["avg_acc_rms"],
                   marker_color=colors, name="Avg RMS",
                   hovertemplate="%{x}<br>RMS: %{y:.2f} m/s2<extra></extra>"),
            row=1, col=1,
        )

        # Spike count with warning colors
        spike_colors = ["#e74c3c" if v > 0 else "#95a5a6" for v in stats["spike_count"]]
        fig.add_trace(
            go.Bar(x=stats["interval"], y=stats["spike_count"],
                   marker_color=spike_colors, name="Spikes",
                   hovertemplate="%{x}<br>Spikes: %{y}<extra></extra>"),
            row=2, col=1,
        )

        fig.update_layout(
            showlegend=False,
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor="white",
        )
        fig.update_xaxes(tickangle=45, tickfont_size=10)
        fig.update_yaxes(gridcolor="#ecf0f1")
        return fig

    # --- Right Panel ---
    @app.callback(
        Output("right-panel", "figure"),
        Input("right-interval", "value"),
    )
    def update_right_panel(interval):
        stats = compute_interval_stats(df, interval)
        if len(stats) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data for this interval", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False, font_size=16)
            fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
            return fig

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=["Avg Speed (m/s)", "Data Volume (points)"],
        )

        fig.add_trace(
            go.Scatter(x=stats["interval"], y=stats["avg_speed"],
                       mode="lines+markers", name="Speed",
                       line=dict(color="#27ae60", width=2),
                       marker=dict(size=6),
                       hovertemplate="%{x}<br>Speed: %{y:.1f} m/s<extra></extra>"),
            row=1, col=1,
        )

        fig.add_trace(
            go.Bar(x=stats["interval"], y=stats["num_points"],
                   marker_color="#9b59b6", name="Volume",
                   hovertemplate="%{x}<br>Points: %{y:,}<extra></extra>"),
            row=2, col=1,
        )

        fig.update_layout(
            showlegend=False,
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor="white",
        )
        fig.update_xaxes(tickangle=45, tickfont_size=10)
        fig.update_yaxes(gridcolor="#ecf0f1")
        return fig

    # --- Acceleration Timeline ---
    @app.callback(
        Output("accel-timeline", "figure"),
        Input("severity-slider", "value"),
    )
    def update_accel_timeline(min_severity):
        # Show a sample of the raw accelerometer signal
        sample_size = min(2000, len(df))
        sample = df.head(sample_size)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=sample["acc_x"], mode="lines", name="X-axis",
            line=dict(color="#3498db", width=1), opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            y=sample["acc_y"], mode="lines", name="Y-axis",
            line=dict(color="#27ae60", width=1), opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            y=sample["acc_z"], mode="lines", name="Z-axis",
            line=dict(color="#e74c3c", width=1), opacity=0.9,
        ))

        # Add threshold line
        threshold = ACCEL_SPIKE_THRESHOLD * 9.81
        fig.add_hline(y=threshold, line_dash="dash", line_color="#f39c12",
                      annotation_text=f"Spike threshold ({ACCEL_SPIKE_THRESHOLD}g)")
        fig.add_hline(y=-threshold, line_dash="dash", line_color="#f39c12")

        fig.update_layout(
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis_title="Sample Index",
            yaxis_title="Acceleration (m/s2)",
            legend=dict(orientation="h", x=0, y=1.1),
            plot_bgcolor="white",
            hovermode="x unified",
        )
        fig.update_yaxes(gridcolor="#ecf0f1")
        return fig

    # --- Anomaly Table ---
    @app.callback(
        Output("anomaly-table", "children"),
        Input("severity-slider", "value"),
    )
    def update_table(min_severity):
        if len(anomalies) == 0:
            return html.P("No anomalies detected.", style={"color": "#7f8c8d", "padding": "20px"})

        filtered = anomalies[anomalies["severity"] >= min_severity].sort_values("severity", ascending=False).head(100)
        if len(filtered) == 0:
            return html.P("No anomalies above this severity threshold.",
                         style={"color": "#7f8c8d", "padding": "20px"})

        header = html.Tr([
            html.Th("Type", style=_th_style()),
            html.Th("Latitude", style=_th_style()),
            html.Th("Longitude", style=_th_style()),
            html.Th("Severity", style=_th_style()),
            html.Th("Time", style=_th_style()),
        ])

        rows = []
        for i, (_, row) in enumerate(filtered.iterrows()):
            color = "#e74c3c" if row["anomaly_type"] == "pothole_candidate" else "#f39c12"
            bg = "#fff" if i % 2 == 0 else "#f8f9fa"
            rows.append(html.Tr([
                html.Td(row["anomaly_type"].replace("_", " ").title(),
                        style={"color": color, "fontWeight": "bold", "padding": "8px"}),
                html.Td(f"{row['latitude']:.5f}", style={"padding": "8px"}),
                html.Td(f"{row['longitude']:.5f}", style={"padding": "8px"}),
                html.Td(f"{row['severity']:.2f}g", style={"padding": "8px", "fontWeight": "bold"}),
                html.Td(str(row["datetime"])[:19] if pd.notna(row.get("datetime")) else "-",
                        style={"padding": "8px", "color": "#7f8c8d"}),
            ], style={"backgroundColor": bg}))

        return html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "13px"},
        )


def _th_style():
    return {"padding": "10px 8px", "backgroundColor": "#2c3e50",
            "color": "white", "textAlign": "left", "fontSize": "13px"}



# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Infrastructure Health Monitoring MVP")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing PVS or phyphox CSV files")
    parser.add_argument("--phyphox", type=str, default=None,
                        help="Single phyphox CSV file to load")
    parser.add_argument("--kaggle", action="store_true",
                        help="Download PVS dataset from Kaggle and use it")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data (no real data needed)")
    parser.add_argument("--port", type=int, default=PORT,
                        help=f"Port to serve on (default: {PORT})")
    args = parser.parse_args()

    print("=" * 60)
    print("  Infrastructure Health Monitoring MVP")
    print("=" * 60)

    # Determine data source
    if args.kaggle:
        data_path = download_pvs_kaggle()
        print(f"\nLoading PVS data from: {data_path}")
        df = load_data_directory(data_path)
        if len(df) == 0:
            print("ERROR: No valid GPS data found in Kaggle dataset.")
            print("Falling back to demo data...")
            df = generate_demo_data()
    elif args.phyphox:
        print(f"\nLoading phyphox file: {args.phyphox}")
        df = load_csv_file(args.phyphox)
        if df is None or len(df) == 0:
            print("ERROR: Could not load phyphox file.")
            sys.exit(1)
    elif args.data_dir:
        print(f"\nLoading data from: {args.data_dir}")
        df = load_data_directory(args.data_dir)
        if len(df) == 0:
            print("ERROR: No valid data found in directory.")
            sys.exit(1)
    else:
        print("\nNo data source specified. Using synthetic demo data.")
        print("  Options: --kaggle | --data-dir PATH | --phyphox FILE")
        df = generate_demo_data(num_trips=50, points_per_trip=200)

    print(f"\nData loaded: {len(df):,} points")
    print(f"  Lat range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"  Lon range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    if "datetime" in df.columns:
        print(f"  Time range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Build adaptive path model
    print("\nBuilding adaptive path model...")
    model = AdaptivePathModel()
    model.update_batch(df)
    print(f"  Grid cells populated: {len(model.cells):,}")

    # Detect anomalies
    print("\nRunning anomaly detection...")
    detector = AnomalyDetector()
    anomalies = detector.detect_all(model, df)
    if len(anomalies) > 0:
        pothole_count = len(anomalies[anomalies["anomaly_type"] == "pothole_candidate"])
        deviation_count = len(anomalies[anomalies["anomaly_type"] == "route_deviation"])
        print(f"  Pothole candidates: {pothole_count}")
        print(f"  Route deviations: {deviation_count}")
    else:
        print("  No anomalies detected.")
        anomalies = pd.DataFrame(columns=["latitude", "longitude", "anomaly_type", "severity", "datetime"])

    # Create and run app
    print(f"\nStarting dashboard on http://127.0.0.1:{args.port}/")
    print("Press Ctrl+C to stop.\n")

    app = create_app(df, model, anomalies)
    register_callbacks(app, df, model, anomalies)
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
