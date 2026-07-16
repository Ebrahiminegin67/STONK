"""
Generate a realistic single vehicle trace dataset.
- GPS at 1 Hz with realistic noise/jitter (~3-5m CEP)
- Accelerometer at 100 Hz with road texture vibration + engine harmonics
- Realistic speed profile with acceleration, braking, stops
- Potholes modeled as sharp impulse + damped oscillation
- Route follows realistic Stockholm-area coordinates
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DURATION_S = 120  # 2 minutes of driving
ACCEL_RATE_HZ = 100
GPS_RATE_HZ = 1

# Route: Stockholm area, heading roughly NE
START_LAT = 59.3327
START_LON = 18.0649
HEADING_DEG = 35  # degrees from north

# ---------------------------------------------------------------------------
# 1. Generate realistic speed profile (city driving)
# ---------------------------------------------------------------------------
t_accel = np.arange(0, DURATION_S, 1.0 / ACCEL_RATE_HZ)
n_samples = len(t_accel)

# Build speed profile in segments (km/h)
# Accelerate -> cruise -> slow for turn -> cruise -> brake to stop -> accelerate -> cruise
segments = [
    ("accelerate", 0, 8, 0, 42),      # 0-8s: start from ~15 to 42 km/h
    ("cruise", 8, 30, 42, 42),          # 8-30s: cruise at ~42
    ("brake", 30, 35, 42, 25),          # 30-35s: slow for a turn
    ("accelerate", 35, 42, 25, 50),     # 35-42s: speed up
    ("cruise", 42, 70, 50, 50),         # 42-70s: cruise at 50
    ("brake", 70, 78, 50, 0),           # 70-78s: stop at red light
    ("idle", 78, 88, 0, 0),             # 78-88s: waiting at light
    ("accelerate", 88, 97, 0, 45),      # 88-97s: green light, go
    ("cruise", 97, 120, 45, 45),        # 97-120s: cruise
]

speed_kmh = np.zeros(n_samples)
for seg_type, t_start, t_end, v_start, v_end in segments:
    mask = (t_accel >= t_start) & (t_accel < t_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        continue
    if seg_type == "idle":
        speed_kmh[idx] = 0
    else:
        # Smooth interpolation
        frac = (t_accel[idx] - t_start) / (t_end - t_start)
        # Use sigmoid-like curve for natural acceleration
        frac_smooth = 0.5 * (1 - np.cos(np.pi * frac))
        speed_kmh[idx] = v_start + (v_end - v_start) * frac_smooth

# Add small speed noise (speedometer jitter)
speed_noise = np.random.normal(0, 0.3, n_samples)
speed_kmh = np.clip(speed_kmh + speed_noise, 0, None)

speed_ms = speed_kmh / 3.6

# ---------------------------------------------------------------------------
# 2. Generate realistic accelerometer data
# ---------------------------------------------------------------------------
# Base: gravity component (phone mounted in car, slight tilt)
# Assume phone is ~5 deg tilted forward, ~2 deg sideways
gravity_x = np.sin(np.radians(5))   # ~0.087g forward component
gravity_y = np.sin(np.radians(2))   # ~0.035g sideways
gravity_z = -1.0  # pointing down (but we'll report deviation from 1g)

# Road surface vibration: broadband noise filtered to 5-30 Hz range
def bandpass_noise(n, fs, low_hz, high_hz, amplitude):
    """Generate band-limited noise simulating road texture."""
    noise = np.random.normal(0, 1, n)
    # Simple FIR bandpass approximation
    from scipy.signal import butter, filtfilt
    nyq = fs / 2
    b, a = butter(3, [low_hz/nyq, high_hz/nyq], btype='band')
    filtered = filtfilt(b, a, noise)
    return filtered * amplitude / np.std(filtered)

# Road vibration (scales with speed)
road_vib_x = bandpass_noise(n_samples, ACCEL_RATE_HZ, 4, 25, 1.0)
road_vib_y = bandpass_noise(n_samples, ACCEL_RATE_HZ, 4, 25, 0.7)
road_vib_z = bandpass_noise(n_samples, ACCEL_RATE_HZ, 5, 30, 1.2)

# Scale vibration by speed (no vibration when stopped)
speed_factor = np.clip(speed_ms / 12.0, 0, 1)  # normalized, full at ~43 km/h
road_vib_x *= speed_factor
road_vib_y *= speed_factor
road_vib_z *= speed_factor

# Engine vibration: harmonic at ~25-40 Hz depending on speed/RPM
engine_freq = 28 + 8 * speed_factor  # Hz, varies with speed
engine_phase = np.cumsum(2 * np.pi * engine_freq / ACCEL_RATE_HZ)
engine_vib = 0.015 * np.sin(engine_phase) * speed_factor

# Driving dynamics: longitudinal accel from speed changes
longitudinal_accel_g = np.gradient(speed_ms, 1.0/ACCEL_RATE_HZ) / 9.81
# Smooth it (car doesn't have infinite jerk)
from scipy.signal import savgol_filter
longitudinal_accel_g = savgol_filter(longitudinal_accel_g, 51, 3)

# Lateral acceleration (simulate a turn around t=32s)
lateral_accel_g = np.zeros(n_samples)
turn_mask = (t_accel >= 30) & (t_accel < 36)
turn_t = t_accel[turn_mask] - 30
lateral_accel_g[turn_mask] = 0.15 * np.sin(np.pi * turn_t / 6)  # gentle turn ~0.15g

# Combine accelerometer signals (in g)
accel_x = gravity_x + longitudinal_accel_g + road_vib_x * 0.03 + np.random.normal(0, 0.005, n_samples)
accel_y = gravity_y + lateral_accel_g + road_vib_y * 0.02 + np.random.normal(0, 0.004, n_samples)
accel_z = road_vib_z * 0.04 + engine_vib + np.random.normal(0, 0.008, n_samples)

# ---------------------------------------------------------------------------
# 3. Add potholes (impulse + damped oscillation)
# ---------------------------------------------------------------------------
def add_pothole(accel_z, t_accel, time_s, severity=1.0):
    """Add a pothole impact: sharp spike + damped spring response."""
    idx = int(time_s * ACCEL_RATE_HZ)
    if idx >= len(accel_z) - 50:
        return

    # Impact duration ~30-80ms
    impact_samples = int(0.04 * ACCEL_RATE_HZ)  # 40ms
    
    # Sharp downward then upward (wheel drops then rebounds)
    t_impact = np.arange(0, 50) / ACCEL_RATE_HZ
    damping = 15.0
    freq = 8.0  # suspension natural frequency ~8 Hz
    response = severity * 0.8 * np.exp(-damping * t_impact) * np.sin(2 * np.pi * freq * t_impact)
    
    # Initial sharp spike
    response[0:3] = [-0.3 * severity, -0.6 * severity, 0.4 * severity]
    
    end_idx = min(idx + len(response), len(accel_z))
    accel_z[idx:end_idx] += response[:end_idx - idx]
    
    # Also affects x and y slightly
    accel_x[idx:idx+5] += np.random.normal(0, 0.05 * severity, min(5, len(accel_x) - idx))
    accel_y[idx:idx+5] += np.random.normal(0, 0.03 * severity, min(5, len(accel_y) - idx))

# Place potholes at specific times
add_pothole(accel_z, t_accel, 18.5, severity=0.6)   # mild pothole
add_pothole(accel_z, t_accel, 45.2, severity=1.2)   # moderate pothole
add_pothole(accel_z, t_accel, 103.7, severity=0.8)  # another one

# Speed bump at t=62s (wider, gentler)
bump_mask = (t_accel >= 61.8) & (t_accel < 62.5)
bump_t = t_accel[bump_mask] - 61.8
accel_z[bump_mask] += 0.25 * np.sin(np.pi * bump_t / 0.7)

# ---------------------------------------------------------------------------
# 4. Generate GPS data at 1 Hz with realistic noise
# ---------------------------------------------------------------------------
t_gps = np.arange(0, DURATION_S, 1.0 / GPS_RATE_HZ)
n_gps = len(t_gps)

# Integrate speed to get distance
distance_m = np.cumsum(speed_ms) / ACCEL_RATE_HZ

# Sample distance at GPS timestamps
gps_indices = (t_gps * ACCEL_RATE_HZ).astype(int)
gps_indices = np.clip(gps_indices, 0, n_samples - 1)
distance_at_gps = distance_m[gps_indices]
speed_at_gps = speed_kmh[gps_indices]

# Convert distance to lat/lon along heading
heading_rad = np.radians(HEADING_DEG)
meters_per_deg_lat = 111320
meters_per_deg_lon = 111320 * np.cos(np.radians(START_LAT))

# Add a turn at the appropriate time (t=30-36s)
# Build cumulative heading changes
heading_changes = np.zeros(n_gps)
for i in range(n_gps):
    t = t_gps[i]
    if 30 <= t < 36:
        heading_changes[i] = 15 / 6  # 15 degrees over 6 seconds (right turn)

cumulative_heading = HEADING_DEG + np.cumsum(heading_changes)
heading_rads = np.radians(cumulative_heading)

# Compute positions from speed + heading
dt_gps = 1.0 / GPS_RATE_HZ
dx = speed_at_gps / 3.6 * dt_gps * np.sin(heading_rads)  # east
dy = speed_at_gps / 3.6 * dt_gps * np.cos(heading_rads)  # north

gps_lat_clean = START_LAT + np.cumsum(dy) / meters_per_deg_lat
gps_lon_clean = START_LON + np.cumsum(dx) / meters_per_deg_lon

# Add GPS noise (CEP ~3-5m, so std ~2.5m per axis)
gps_noise_m = 2.5
gps_lat_noisy = gps_lat_clean + np.random.normal(0, gps_noise_m / meters_per_deg_lat, n_gps)
gps_lon_noisy = gps_lon_clean + np.random.normal(0, gps_noise_m / meters_per_deg_lon, n_gps)

# Occasional GPS glitch (one point with larger error)
glitch_idx = 55
gps_lat_noisy[glitch_idx] += 8.0 / meters_per_deg_lat  # 8m offset
gps_lon_noisy[glitch_idx] -= 6.0 / meters_per_deg_lon

# GPS speed (derived, with noise)
gps_speed = speed_at_gps + np.random.normal(0, 1.0, n_gps)
gps_speed = np.clip(gps_speed, 0, None)

# ---------------------------------------------------------------------------
# 5. Build output DataFrame (accelerometer at 100 Hz, GPS at 1 Hz)
# ---------------------------------------------------------------------------
# Main output: merged timeline at 100 Hz, GPS columns filled only every 100 samples
# This mimics how phyphox exports data

df = pd.DataFrame({
    "timestamp_s": t_accel,
    "accel_x_g": np.round(accel_x, 5),
    "accel_y_g": np.round(accel_y, 5),
    "accel_z_g": np.round(accel_z, 5),
    "speed_kmh": np.round(speed_kmh, 2),
    "gps_lat": np.nan,
    "gps_lon": np.nan,
    "gps_speed_kmh": np.nan,
})

# Fill GPS at 1 Hz (every 100th sample)
for i, gps_idx in enumerate(gps_indices):
    if i < n_gps:
        df.loc[gps_idx, "gps_lat"] = round(gps_lat_noisy[i], 7)
        df.loc[gps_idx, "gps_lon"] = round(gps_lon_noisy[i], 7)
        df.loc[gps_idx, "gps_speed_kmh"] = round(gps_speed[i], 1)

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
output_path = r"C:\Users\eyorbdt\STONK\infrastructure_health_network\dataset_single_vehicle_trace.csv"
df.to_csv(output_path, index=False)

print(f"Generated realistic vehicle trace:")
print(f"  Duration: {DURATION_S}s")
print(f"  Accelerometer: {n_samples} samples at {ACCEL_RATE_HZ} Hz")
print(f"  GPS: {n_gps} fixes at {GPS_RATE_HZ} Hz")
print(f"  Speed profile: stop-and-go city driving (0-50 km/h)")
print(f"  Potholes: 3 events + 1 speed bump")
print(f"  GPS noise: ~2.5m std per axis + 1 glitch")
print(f"  Saved to: {output_path}")
print(f"  File size: {os.path.getsize(output_path) / 1024:.0f} KB")
