#!/usr/bin/env python3
"""
Generate synthetic time series mimicking 8 LOTSA real datasets.
Each generator is tuned based on FFT analysis + domain knowledge.
Output: /workspace/data/mimic_test/{dataset}/
  - synthetic_h96.pt
  - comparison.png  (8 real vs 8 synthetic futures)
  - psd_comparison.png  (mean power spectrum)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REAL_BASE = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "real_eval_lot_ett"
OUTPUT_BASE = Path(os.environ.get("DATA_ROOT", "/workspace/data")) / "mimic_test"
CONTEXT_LEN = 512
HORIZON = 96
N_SYNTH = 500
N_PLOT = 8
RNG_SEED = 42

# ─── Real data loader ────────────────────────────────────────────────────────

def load_real_futures(name: str) -> np.ndarray:
    """Load normalized future arrays for each dataset. Returns (N, 96) np.ndarray."""
    base = REAL_BASE
    paths = {
        'elecdemand':       base / 'energy/elecdemand/futures_c512_half_hourly_h96_lotsa_test.pt',
        'ETTh1':            base / 'energy/ETTh1/cache/futures_c512_H_h96_ett.pt',
        'oikolab_weather':  base / 'climate/oikolab_weather/futures_c512_hourly_h96_lotsa_test.pt',
        'saugeenday':       base / 'nature/saugeenday/futures_c512_daily_h96_lotsa_test.pt',
        'us_births':        base / 'healthcare/us_births/futures_c512_daily_h96_lotsa_test.pt',
        'PEMS03':           base / 'transport/PEMS03/futures_c512_5_minutes_h96_lotsa.pt',
        'pedestrian_counts':base / 'econfin/pedestrian_counts/futures_c512_hourly_h96_lotsa.pt',
        'alibaba_cluster_trace_2018': base / 'cloudops/alibaba_cluster_trace_2018/futures_c512_5_minutes_h96_lotsa.pt',
    }
    t = torch.load(paths[name], map_location='cpu', weights_only=False)
    fn = t['futures_n'].float().numpy()
    # Filter invalid
    valid = np.isfinite(fn).all(axis=1) & (np.abs(fn).max(axis=1) < 50)
    return fn[valid]


# ─── Generator utilities ─────────────────────────────────────────────────────

def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Simple Gaussian smoothing via convolution (no scipy)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r+1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode='same')


def colored_noise(L: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f^alpha colored noise via FFT shaping.
    alpha=0: white noise; alpha=1: pink; alpha=2: Brownian/red.
    """
    white = rng.normal(0, 1, L)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(L)
    freqs[0] = 1e-6
    fft  *= freqs ** (-alpha / 2)
    fft[0] = 0.0   # zero DC
    out = np.fft.irfft(fft, n=L)
    std = out.std()
    if std > 0:
        out /= std
    return out


def am_envelope(L: int, period_scale: float, depth: float,
                rng: np.random.Generator) -> np.ndarray:
    """Smooth multiplicative amplitude envelope (slow variation).
    depth: fraction of variation (0=none, 0.5 = ±50% variation).
    """
    mod = colored_noise(L, alpha=2.0, rng=rng)  # very smooth (red noise)
    mod = (mod - mod.min()) / (mod.max() - mod.min() + 1e-8)  # 0-1
    return 1.0 - depth + depth * mod   # range [1-depth, 1]


def sample_windows(
    ts_long: np.ndarray,
    context_len: int,
    horizon: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n_samples (context, future) windows from a long series."""
    total = context_len + horizon
    max_start = len(ts_long) - total
    if max_start <= 0:
        raise ValueError(f"Series length {len(ts_long)} too short for window {total}")
    starts = rng.integers(0, max_start, n_samples)
    contexts, futures = [], []
    for s in starts:
        ctx = ts_long[s:s + context_len]
        fut = ts_long[s + context_len:s + total]
        mu, sig = ctx.mean(), ctx.std() + 1e-8
        contexts.append((ctx - mu) / sig)
        futures.append((fut - mu) / sig)
    return np.array(contexts, dtype=np.float32), np.array(futures, dtype=np.float32)


# ─── Per-dataset generators (each returns a long np.ndarray) ─────────────────

def gen_elecdemand(L: int, rng: np.random.Generator) -> np.ndarray:
    """Half-hourly energy demand: strong daily (bimodal) + weekly + AM envelope.

    Real data is smoother than initially thought but has day-to-day amplitude
    variation and mid-frequency energy from non-stationary load patterns.
    """
    t = np.arange(L, dtype=float)
    ppd, ppw = 48, 336

    phi_d = rng.uniform(0, 2*np.pi)
    A_d   = rng.uniform(0.6, 1.1)
    A_w   = rng.uniform(0.10, 0.35)
    phi_w = rng.uniform(0, 2*np.pi)

    # Per-day amplitude jitter (day-to-day demand variation: weekends, weather)
    n_days  = L // ppd + 2
    day_amp = rng.uniform(0.5, 1.5, n_days)
    day_amp = np.repeat(day_amp, ppd)[:L]

    # Bimodal daily: k=1 broad arch, k=2 creates asymmetric bimodal split
    daily_shape = (
        0.55 * np.sin(2*np.pi*1*t/ppd + phi_d) +
        0.35 * np.sin(2*np.pi*2*t/ppd + phi_d + 1.20) +
        0.10 * np.sin(2*np.pi*3*t/ppd + phi_d - 0.50)
    )
    daily = A_d * day_amp * daily_shape

    weekly = A_w * (
        0.80 * np.sin(2*np.pi*t/ppw + phi_w) +
        0.20 * np.sin(2*np.pi*2*t/ppw + phi_w + 0.4)
    )
    # Slow linear drift
    trend = rng.uniform(-0.25, 0.25) * t / L
    # Short noise + slow correlated component
    noise = 0.07 * rng.normal(0, 1, L) + 0.06 * colored_noise(L, alpha=1.5, rng=rng)
    return trend + daily + weekly + noise


def gen_etth1(L: int, rng: np.random.Generator) -> np.ndarray:
    """Hourly transformer oil temp: yearly dominant (slow drift) + noisy daily.

    Real ETTh1 looks like daily oscillation with HEAVY day-to-day amplitude
    variation and short-timescale noise — not a clean sinusoid.
    """
    t = np.arange(L, dtype=float)
    ppd = 24
    ppY = 365.25 * ppd

    phi_y = rng.uniform(0, 2*np.pi)
    phi_d = rng.uniform(0, 2*np.pi)
    A_y   = rng.uniform(0.7, 1.2)
    A_d   = rng.uniform(0.40, 0.70)

    yearly = A_y * (
        0.62 * np.sin(2*np.pi*1*t/ppY + phi_y) +
        0.24 * np.sin(2*np.pi*2*t/ppY + phi_y + 0.7) +
        0.10 * np.sin(2*np.pi*4*t/ppY + phi_y + 1.5) +
        0.04 * np.sin(2*np.pi*6*t/ppY + phi_y + 2.0)
    )

    # Per-day amplitude jitter: each day independently scaled
    n_days  = L // ppd + 2
    day_amp = rng.uniform(0.4, 1.6, n_days)     # day-to-day amplitude scatter
    day_amp = np.repeat(day_amp, ppd)[:L]        # expand to sample level

    daily_shape = (
        0.72 * np.sin(2*np.pi*t/ppd + phi_d) +
        0.28 * np.sin(2*np.pi*2*t/ppd + phi_d + 0.8)
    )
    daily = A_d * day_amp * daily_shape

    # Short noise (makes daily cycles look irregular, as in real ETTh1)
    noise = 0.16 * rng.normal(0, 1, L)
    # Slow drift (seasonal + weather systems)
    slow_drift = 0.12 * colored_noise(L, alpha=2.0, rng=rng)
    return yearly + daily + noise + slow_drift


def gen_oikolab(L: int, rng: np.random.Generator) -> np.ndarray:
    """Hourly weather: daily + yearly drift + correlated weather noise."""
    t = np.arange(L, dtype=float)
    ppd = 24
    ppw = 7 * ppd
    ppY = 365.25 * ppd

    phi_d = rng.uniform(0, 2*np.pi)
    phi_w = rng.uniform(0, 2*np.pi)
    phi_y = rng.uniform(0, 2*np.pi)

    A_d = rng.uniform(0.5, 1.0)
    A_w = rng.uniform(0.05, 0.18)
    A_y = rng.uniform(0.4, 0.9)

    # Per-day amplitude jitter (weather: each day has different temperature range)
    n_days  = L // ppd + 2
    day_amp = rng.uniform(0.5, 1.5, n_days)
    day_amp = np.repeat(day_amp, ppd)[:L]

    daily_shape = (
        0.65 * np.sin(2*np.pi*t/ppd + phi_d) +
        0.25 * np.sin(2*np.pi*2*t/ppd + phi_d + 0.6) +
        0.10 * np.sin(2*np.pi*3*t/ppd + phi_d - 0.4)
    )
    daily = A_d * day_amp * daily_shape

    weekly = A_w * np.sin(2*np.pi*t/ppw + phi_w)
    yearly = A_y * (
        0.70 * np.sin(2*np.pi*t/ppY + phi_y) +
        0.30 * np.sin(2*np.pi*2*t/ppY + phi_y + 1.0)
    )
    # Correlated weather-system noise (hours to days) + short white noise
    noise = (0.15 * colored_noise(L, alpha=1.8, rng=rng)
             + 0.12 * rng.normal(0, 1, L))
    return daily + weekly + yearly + noise


def gen_saugeenday(L: int, rng: np.random.Generator) -> np.ndarray:
    """Daily Saugeen River flow: asymmetric annual spring freshet + occasional storms.

    The dominant pattern is an annual spring flood (snowmelt): predictable
    timing ~day 100-150 of year, fast rise (2-3 weeks), slow exponential
    recession (2-3 months). Low-flow baseline in summer/fall/winter.
    This creates spike-heavy patterns in every 96-day window depending on phase.
    """
    t = np.arange(L, dtype=float)
    ppY = 365.25

    # Slow yearly baseline (symmetric sinusoid — winter-summer contrast)
    phi_y_base = rng.uniform(0, 2*np.pi)
    A_y_base   = rng.uniform(0.15, 0.35)
    yearly_base = A_y_base * np.sin(2*np.pi*t/ppY + phi_y_base)

    # Annual spring freshet: asymmetric pulse train (once per year)
    spring = np.zeros(L)
    A_spring = rng.uniform(1.0, 2.2)   # peak flow amplitude
    rise_t   = rng.integers(10, 25)    # 10-25 days rise (snowmelt)
    decay_t  = rng.uniform(40.0, 90.0) # 40-90 days slow recession

    # Random spring peak day within the year (day 90-160 = Apr-Jun)
    first_peak = rng.uniform(90, 160)
    curr_peak  = first_peak
    while curr_peak < L + rise_t:
        pos = int(curr_peak - rise_t)
        # Year-to-year amplitude variation
        amp = A_spring * rng.uniform(0.5, 1.5)
        # Year-to-year timing variation ±3 weeks
        timing_jitter = rng.integers(-21, 22)
        start = max(0, pos + timing_jitter)
        max_dur = int(rise_t + decay_t * 4)
        for i in range(min(max_dur, L - start)):
            if i < rise_t:
                spring[start + i] += amp * (i + 1) / rise_t
            else:
                spring[start + i] += amp * np.exp(-(i - rise_t) / decay_t)
        curr_peak += ppY * rng.uniform(0.92, 1.08)  # next year (slight variation)

    # Occasional summer/fall storms (smaller, shorter)
    storm_spikes = np.zeros(L)
    n_storms = rng.poisson(max(1, L // 120))
    for _ in range(n_storms):
        pos   = rng.integers(0, L)
        amp   = rng.exponential(0.4)
        rise  = rng.integers(2, 8)
        decay = rng.uniform(10.0, 25.0)
        dur   = int(rise + decay * 3)
        for i in range(min(dur, L - pos)):
            if i < rise:
                storm_spikes[pos + i] += amp * (i + 1) / rise
            else:
                storm_spikes[pos + i] += amp * np.exp(-(i - rise) / decay)

    noise = rng.normal(0, 0.05, L)
    return yearly_base + spring + storm_spikes + noise


def gen_us_births(L: int, rng: np.random.Generator) -> np.ndarray:
    """Daily births: dominant weekly (weekday/weekend) + annual cycle.

    Real us_births shows: jagged square-wave-like weekly oscillation with
    noticeable day-to-day random variation WITHIN each week.
    Occasional holiday outliers (very low counts on holidays like Christmas).
    Annual baseline drift (more births in summer/fall).
    """
    t = np.arange(L, dtype=float)
    ppW = 7.0
    ppY = 365.25

    phi_w = rng.uniform(0, 2*np.pi)
    phi_y = rng.uniform(0, 2*np.pi)
    A_w   = rng.uniform(0.6, 1.1)
    A_y   = rng.uniform(0.20, 0.55)

    # Weekly base: square-wave-like via higher harmonics
    # US births: Mon-Fri ≈ high, Sat-Sun ≈ lower (elective procedures on weekdays)
    # Model as offset + sinusoid + harmonics (creates notch on weekends)
    weekly_shape = (
        0.60 * np.sin(2*np.pi*1*t/ppW + phi_w) +
        0.25 * np.sin(2*np.pi*2*t/ppW + phi_w + 0.20) +
        0.10 * np.sin(2*np.pi*3*t/ppW + phi_w - 0.40) +
        0.05 * np.sin(2*np.pi*4*t/ppW + phi_w + 0.70)
    )

    # Per-day random multiplier: captures within-week and week-to-week variation
    n_days  = L + 1
    day_jitter = rng.normal(1.0, 0.18, n_days)   # ~18% random day variation
    day_jitter = np.clip(day_jitter, 0.3, 2.0)[:L]

    weekly = A_w * weekly_shape * day_jitter

    # Annual baseline (seasonal trend)
    yearly = A_y * (
        0.72 * np.sin(2*np.pi*t/ppY + phi_y) +
        0.28 * np.sin(2*np.pi*2*t/ppY + phi_y + 0.9)
    )

    # Holiday dips: occasional very-low days (Christmas, New Year, etc.)
    n_holidays = rng.integers(max(1, L // 60), max(2, L // 30))
    holiday_pos = rng.integers(0, L, n_holidays)
    holiday_dip = np.zeros(L)
    for pos in holiday_pos:
        holiday_dip[pos] -= rng.uniform(0.8, 2.0)

    # Mild colored noise
    noise = 0.07 * colored_noise(L, alpha=0.8, rng=rng)
    return weekly + yearly + holiday_dip + noise


def gen_pems03(L: int, rng: np.random.Generator) -> np.ndarray:
    """5-minute traffic: bimodal daily (rush hours) + weekly weekend dip.

    Key constraint: context=512 steps = 42.7 hours ≈ 1.78 days.
    Future=96 steps = 8 hours (partial daily cycle).
    Normalized windows show partial daily arches, not multiple oscillations.
    High-frequency harmonics must be suppressed.
    """
    t = np.arange(L, dtype=float)
    ppd = 288   # 288 pts per day at 5-min
    ppw = 7 * ppd

    A_d   = rng.uniform(0.6, 1.1)
    A_w   = rng.uniform(0.08, 0.22)
    phi_w = rng.uniform(0, 2*np.pi)

    # Bimodal daily shape via k=1 and k=2 only (suppress k=3,4 to avoid
    # visible sub-period oscillations in 8-hour future windows)
    # Peak at ~12.5h midday between morning (8h=96) and evening (17h=204)
    morning_center = 96
    evening_center = 204
    midpoint = (morning_center + evening_center) / 2   # ~150 steps

    phi_1 = np.pi/2 - 2*np.pi * midpoint / ppd
    phi_2 = np.pi/2 - 2*np.pi * 2 * morning_center / ppd

    daily = A_d * (
        0.58 * np.sin(2*np.pi*1*t/ppd + phi_1) +
        0.35 * np.sin(2*np.pi*2*t/ppd + phi_2) +
        0.05 * np.sin(2*np.pi*3*t/ppd + phi_2 + 0.3) +
        0.02 * np.sin(2*np.pi*4*t/ppd + phi_2 + 0.6)
    )

    # Nighttime floor: traffic goes near-zero at night
    # Shift so minimum is near 0, then re-center by subtracting global mean
    daily = daily - np.percentile(daily, 10)  # lift floor to 0
    daily = daily - daily.mean()              # re-center for normalization

    weekly = A_w * (
        0.80 * np.sin(2*np.pi*t/ppw + phi_w) +
        0.20 * np.sin(2*np.pi*2*t/ppw + phi_w + 0.5)
    )

    # Traffic noise: heteroskedastic (more noise during rush hours)
    noise_base = rng.normal(0, 0.08, L)
    # Add occasional incident spikes
    incident_mask = rng.random(L) < 0.004
    incident_amp  = rng.exponential(1.0, L)
    incidents = gaussian_smooth(incident_mask.astype(float) * incident_amp, sigma=2.0)

    return daily + weekly + noise_base + incidents


def gen_pedestrian(L: int, rng: np.random.Generator) -> np.ndarray:
    """Hourly pedestrian: near-zero at night, strong afternoon peak + weekly + spikes.

    Key features:
    - Very asymmetric daily shape (almost zero 0-7h, peak 12-17h, decline)
    - Day-to-day amplitude variation (some days much busier)
    - Occasional event spikes (concerts, markets, weekend festivals)
    - Weekly: some day-of-week structure
    """
    t = np.arange(L, dtype=float)
    ppd = 24
    ppw = 7 * ppd

    peak_hour = rng.uniform(13.0, 16.0)   # afternoon peak hour
    A_base    = rng.uniform(0.9, 1.4)
    A_w       = rng.uniform(0.06, 0.20)
    phi_w     = rng.uniform(0, 2*np.pi)

    # --- Build day-varying amplitude envelope ---
    n_days = L // ppd + 2
    # Day amplitude varies: weekdays slightly higher, weekends can be higher for some sites
    day_amps = np.ones(n_days)
    for d in range(n_days):
        dow = d % 7
        # Random day-to-day variation
        day_amps[d] = rng.uniform(0.4, 1.6)

    # Expand to sample-level
    amp_env = np.repeat(day_amps, ppd)[:L]

    # --- Rectified daily shape ---
    phase = np.pi/2 - 2*np.pi * peak_hour / ppd
    raw   = np.sin(2*np.pi*t/ppd + phase)
    # Power 2 gives sharper peak; clip negative to zero (night = no pedestrians)
    daily_shape = np.maximum(0.0, raw) ** 2.0
    daily = A_base * amp_env * daily_shape

    # --- Weekly slow modulation ---
    weekly = A_w * np.sin(2*np.pi*t/ppw + phi_w)

    # --- Event spikes (festivals, markets) ---
    spikes = np.zeros(L)
    n_ev = rng.poisson(max(1, L // (ppd * 15)))  # ~1 per 15 days
    for _ in range(n_ev):
        s = rng.integers(ppd // 4, L - ppd // 4)   # daytime events
        amp = rng.exponential(1.5)
        dur = rng.integers(3, 7)
        end = min(s + dur, L)
        spikes[s:end] += amp
    if spikes.any():
        spikes = gaussian_smooth(spikes, sigma=1.5)

    noise = rng.normal(0, 0.06, L)
    return daily + weekly + spikes + noise


def gen_alibaba(L: int, rng: np.random.Generator) -> np.ndarray:
    """5-minute Alibaba server cluster metrics: irregular, bursty, weakly periodic.

    Real data characteristics:
    - Some samples show gradual rise/fall (load shifting)
    - Some samples have large sustained spikes (job launches)
    - Mild daily pattern in background (lower at night)
    - Much noisier and more irregular than traffic data
    - std after normalization is high (~1.23 for futures vs ~1.0 for others)
    """
    t = np.arange(L, dtype=float)
    ppd = 288

    # Very mild daily pattern (background rhythmic load)
    phi_d = rng.uniform(0, 2*np.pi)
    A_d   = rng.uniform(0.04, 0.12)   # very weak — mostly noise
    daily = A_d * np.sin(2*np.pi*t/ppd + phi_d)

    # Random walk baseline (smooth trend shifts over hours/days)
    walk_speed = rng.uniform(0.001, 0.006)
    walk = np.cumsum(rng.normal(0, walk_speed, L))
    walk = gaussian_smooth(walk, sigma=30.0)   # smooth to slow drift

    # Bursty load spikes: Poisson-driven, varying duration
    spikes = np.zeros(L)
    burst_rate = rng.uniform(0.005, 0.025)   # more frequent than before
    for i in range(L):
        if rng.random() < burst_rate:
            amp = rng.exponential(1.2)
            dur = rng.integers(5, 60)   # 5-60 minutes wide
            end = min(i + dur, L)
            spikes[i:end] += amp * np.exp(-np.arange(end - i) / (dur * 0.4))

    spikes = gaussian_smooth(spikes, sigma=2.0)

    # Background noise (servers are always somewhat noisy)
    noise = rng.normal(0, 0.18, L)

    raw = daily + walk + spikes + noise
    # Zero-center the walk so normalization works correctly
    raw = raw - raw.mean()
    return raw


# ─── Dataset registry ────────────────────────────────────────────────────────

DATASETS = {
    'elecdemand': {
        'generator': gen_elecdemand,
        'L': 15000,          # ~312 days at half-hourly
        'ppd': 48,
        'label': 'Energy Demand (half-hourly)',
    },
    'ETTh1': {
        'generator': gen_etth1,
        'L': 25000,          # ~2.85 years at hourly — needs yearly pattern
        'ppd': 24,
        'label': 'ETTh1 Oil Temp (hourly)',
    },
    'oikolab_weather': {
        'generator': gen_oikolab,
        'L': 25000,
        'ppd': 24,
        'label': 'Weather (hourly)',
    },
    'saugeenday': {
        'generator': gen_saugeenday,
        'L': 4000,           # ~11 years at daily
        'ppd': 1,
        'label': 'Saugeen River (daily)',
    },
    'us_births': {
        'generator': gen_us_births,
        'L': 4000,
        'ppd': 1,
        'label': 'US Births (daily)',
    },
    'PEMS03': {
        'generator': gen_pems03,
        'L': 20000,          # ~69 days at 5-minute
        'ppd': 288,
        'label': 'PEMS03 Traffic (5-min)',
    },
    'pedestrian_counts': {
        'generator': gen_pedestrian,
        'L': 15000,
        'ppd': 24,
        'label': 'Pedestrian Counts (hourly)',
    },
    'alibaba_cluster_trace_2018': {
        'generator': gen_alibaba,
        'L': 15000,
        'ppd': 288,
        'label': 'Alibaba Cluster (5-min)',
    },
}


# ─── Plotting ────────────────────────────────────────────────────────────────

COLORS = {
    'real':  '#2166ac',
    'synth': '#d73027',
}


def plot_comparison(name: str, real_fn: np.ndarray, synth_fn: np.ndarray, label: str) -> None:
    """2×8 grid: top = real samples, bottom = synthetic samples (futures only)."""
    H = real_fn.shape[1]
    n_real  = min(N_PLOT, len(real_fn))
    n_synth = min(N_PLOT, len(synth_fn))

    # Randomly pick samples
    rng  = np.random.default_rng(0)
    ridx = rng.choice(len(real_fn),  n_real,  replace=False)
    sidx = rng.choice(len(synth_fn), n_synth, replace=False)

    fig, axes = plt.subplots(2, N_PLOT, figsize=(3*N_PLOT, 5), sharex=True)
    fig.suptitle(f'{label}\nTop: Real   Bottom: Synthetic', fontsize=12, fontweight='bold')

    xticks = np.linspace(0, H-1, 5, dtype=int)

    for col in range(N_PLOT):
        ax_r = axes[0, col]
        ax_s = axes[1, col]

        ax_r.plot(real_fn[ridx[col]], color=COLORS['real'], lw=0.9)
        ax_s.plot(synth_fn[sidx[col]], color=COLORS['synth'], lw=0.9)

        for ax in (ax_r, ax_s):
            ax.set_xticks(xticks)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, lw=0.5)

        if col == 0:
            ax_r.set_ylabel('Real', fontsize=8, color=COLORS['real'])
            ax_s.set_ylabel('Synthetic', fontsize=8, color=COLORS['synth'])

    fig.text(0.5, 0.01, f'Forecast step (H={H})', ha='center', fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = OUTPUT_BASE / name / 'comparison.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → saved {out}')


def plot_psd(name: str, real_fn: np.ndarray, synth_fn: np.ndarray, label: str) -> None:
    """Mean power spectrum: real vs synthetic."""
    H = real_fn.shape[1]
    freqs = np.fft.rfftfreq(H)[1:]  # skip DC

    real_psd  = np.abs(np.fft.rfft(real_fn, axis=1))[:, 1:].mean(axis=0)
    synth_psd = np.abs(np.fft.rfft(synth_fn, axis=1))[:, 1:].mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(freqs, real_psd,  color=COLORS['real'],  lw=1.2, label='Real (mean)')
    ax.semilogy(freqs, synth_psd, color=COLORS['synth'], lw=1.2, label='Synthetic (mean)', alpha=0.85)
    ax.set_xlabel('Normalized frequency (cycles/step)', fontsize=10)
    ax.set_ylabel('Mean amplitude', fontsize=10)
    ax.set_title(f'{label} — Power Spectrum Comparison', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()

    out = OUTPUT_BASE / name / 'psd_comparison.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → saved {out}')


def plot_long_context(
    name: str, real_fn: np.ndarray, synth_ctx: np.ndarray, synth_fut: np.ndarray, label: str
) -> None:
    """Show synthetic context+future (full 608 steps) vs real futures only."""
    rng  = np.random.default_rng(1)
    sidx = rng.choice(len(synth_ctx), min(4, len(synth_ctx)), replace=False)
    ridx = rng.choice(len(real_fn),   min(4, len(real_fn)),   replace=False)
    C = synth_ctx.shape[1]
    H = synth_fut.shape[1]

    fig, axes = plt.subplots(2, 4, figsize=(16, 5))
    fig.suptitle(f'{label}\nTop: Synthetic (context+future)   Bottom: Real futures', fontsize=11)

    for col in range(4):
        ax = axes[0, col]
        full = np.concatenate([synth_ctx[sidx[col]], synth_fut[sidx[col]]])
        ax.plot(np.arange(C), synth_ctx[sidx[col]], color='#aaa', lw=0.8)
        ax.plot(np.arange(C, C+H), synth_fut[sidx[col]], color=COLORS['synth'], lw=1.1)
        ax.axvline(C, color='k', lw=0.7, ls='--')
        ax.set_title(f'synth[{sidx[col]}]', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

        ax = axes[1, col]
        ax.plot(real_fn[ridx[col]], color=COLORS['real'], lw=0.9)
        ax.set_title(f'real[{ridx[col]}]', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    axes[0, 0].set_ylabel('Synthetic', fontsize=8)
    axes[1, 0].set_ylabel('Real', fontsize=8)
    plt.tight_layout()

    out = OUTPUT_BASE / name / 'context_comparison.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → saved {out}')


# ─── Main pipeline ───────────────────────────────────────────────────────────

def generate_dataset(name: str, cfg: dict, rng_global: np.random.Generator) -> None:
    out_dir = OUTPUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    label = cfg['label']
    print(f'\n[{name}]')

    # 1. Load real futures
    try:
        real_fn = load_real_futures(name)
        print(f'  real: {real_fn.shape}')
    except Exception as e:
        print(f'  WARNING: could not load real data: {e}')
        real_fn = None

    # 2. Generate one long series, sample N_SYNTH windows
    gen_fn = cfg['generator']
    L = cfg['L']

    # Use per-dataset seed for reproducibility
    ds_seed = int(abs(hash(name))) % (2**31)
    rng_ds = np.random.default_rng(ds_seed)

    ts_long = gen_fn(L, rng_ds)
    synth_ctx, synth_fut = sample_windows(ts_long, CONTEXT_LEN, HORIZON, N_SYNTH, rng_ds)

    print(f'  synth: ctx={synth_ctx.shape}, fut={synth_fut.shape}')
    print(f'  synth fut: mean={synth_fut.mean():.3f}, std={synth_fut.std():.3f}')
    if real_fn is not None:
        print(f'  real  fut: mean={real_fn.mean():.3f}, std={real_fn.std():.3f}')

    # 3. Save
    torch.save(
        {
            'contexts_n': torch.from_numpy(synth_ctx),
            'futures_n':  torch.from_numpy(synth_fut),
            'context_len': CONTEXT_LEN,
            'horizon': HORIZON,
            'n_samples': N_SYNTH,
            'generator': name,
        },
        out_dir / 'synthetic_h96.pt',
    )

    # 4. Plots
    if real_fn is not None:
        plot_comparison(name, real_fn, synth_fut, label)
        plot_psd(name, real_fn, synth_fut, label)
        plot_long_context(name, real_fn, synth_ctx, synth_fut, label)
    else:
        print('  skipping comparison plots (no real data)')


def main() -> None:
    print(f'Output: {OUTPUT_BASE}')
    rng = np.random.default_rng(RNG_SEED)

    for name, cfg in DATASETS.items():
        generate_dataset(name, cfg, rng)

    print('\nDone. Check /workspace/data/mimic_test/ for plots.')


if __name__ == '__main__':
    main()
