"""
Generate synthetic light curves for demo cases.
"""

import numpy as np


def inject_transit(time, flux, period, depth, duration, t0=None):
    if t0 is None:
        t0 = period / 4
    flux_with_transit = flux.copy()
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    half_dur = (duration / period) / 2
    in_transit = np.abs(phase) < half_dur
    ingress_egress = (np.abs(phase) >= half_dur * 0.7) & (np.abs(phase) < half_dur)
    flux_with_transit[in_transit] -= depth
    if np.sum(ingress_egress) > 0:
        edge_phase = (np.abs(phase[ingress_egress]) - half_dur * 0.7) / (half_dur * 0.3 + 1e-10)
        flux_with_transit[ingress_egress] += depth * (1 - 0.5 * (1 - np.cos(np.pi * edge_phase)))
    return flux_with_transit


def stellar_background(n, variability=0.0005):
    t = np.linspace(0, 1, n)
    bg = 1.0 + variability * np.sin(2*np.pi*3*t + 0.7)
    bg += variability * 0.5 * np.sin(2*np.pi*7*t + 1.2)
    return bg


def case1_clean_planet():
    np.random.seed(42)
    n, time = 1440, np.linspace(0, 30, 1440)
    flux = stellar_background(n, 0.0003) + np.random.normal(0, 0.0008, n)
    flux = inject_transit(time, flux, period=10.0, depth=0.008, duration=3/24, t0=2.5)
    flux = 0.95 * flux + 0.05 * 1.0
    return {"time": time, "flux": flux, "flux_err": np.ones(n)*0.0008,
            "ra": 285.68, "dec": 50.24, "candidate_id": "DEMO_001_PLANET",
            "true_label": "Planet Transit", "description": "Clean planet, period=10d, depth=0.8%"}


def case2_noisy_blended():
    np.random.seed(123)
    n, time = 1440, np.linspace(0, 30, 1440)
    flux = stellar_background(n, 0.001) + np.random.normal(0, 0.003, n)
    flux += 0.001 * np.linspace(0, 1, n)
    flux = inject_transit(time, flux, period=7.0, depth=0.012, duration=2.5/24, t0=1.5)
    contaminant = stellar_background(n, 0.0002) + np.random.normal(0, 0.001, n)
    flux = 0.65 * flux + 0.35 * contaminant
    outlier_idx = np.random.choice(n, 5, replace=False)
    flux[outlier_idx] += np.random.choice([-1,1], 5) * 0.05
    return {"time": time, "flux": flux, "flux_err": np.ones(n)*0.003,
            "ra": 290.12, "dec": 45.68, "candidate_id": "DEMO_002_NOISY_BLEND",
            "true_label": "Planet Transit", "description": "Noisy+blended, period=7d, depth=1.2%"}


def case3_eclipsing_binary():
    np.random.seed(99)
    n, time = 1440, np.linspace(0, 30, 1440)
    flux = stellar_background(n, 0.0005) + np.random.normal(0, 0.001, n)
    flux = inject_transit(time, flux, period=5.0, depth=0.03, duration=4/24, t0=1.0)
    flux = inject_transit(time, flux, period=5.0, depth=0.015, duration=3.5/24, t0=3.5)
    return {"time": time, "flux": flux, "flux_err": np.ones(n)*0.001,
            "ra": 270.99, "dec": 38.43, "candidate_id": "DEMO_003_BINARY",
            "true_label": "Eclipsing Binary", "description": "Binary with secondary eclipse, period=5d"}


def case4_astrosat_like():
    np.random.seed(77)
    time_segs, flux_segs = [], []
    for i in range(200):
        t_start = i * (97/60/24)
        t_seg = t_start + np.linspace(0, 37/60/24, 25)
        f_seg = stellar_background(25, 0.002) + np.random.normal(0, 0.005, 25)
        f_seg += 0.003 * np.sin(2*np.pi*t_seg*15)
        time_segs.append(t_seg); flux_segs.append(f_seg)
    time = np.concatenate(time_segs)
    flux = np.concatenate(flux_segs)
    n = len(time)
    flux = inject_transit(time, flux, period=10.0, depth=0.01, duration=3/24, t0=2.0)
    return {"time": time, "flux": flux, "flux_err": np.ones(n)*0.005,
            "ra": 180.54, "dec": 22.12, "candidate_id": "DEMO_004_ASTROSAT",
            "true_label": "Planet Transit", "description": "AstroSat-like cadence with gaps"}


def case5_kepler7b():
    """
    Case 5: Kepler-7b — Real confirmed exoplanet parameters.
    Period:   4.8854892 days (NASA confirmed)
    Depth:    0.96%         (NASA confirmed)
    Duration: 4.6 hours     (NASA confirmed)
    Expected: Planet Transit — HIGH confidence
    """
    np.random.seed(99)
    n    = 720
    time = np.linspace(0, 30, n)

    # Realistic ground-based photometric noise
    flux = stellar_background(n, variability=0.0005)
    flux += np.random.normal(0, 0.001, n)

    # Inject REAL Kepler-7b parameters
    flux = inject_transit(time, flux,
                          period=4.8854892,
                          depth=0.0096,
                          duration=4.6/24,
                          t0=1.2)

    # Mild blending (realistic crowded field)
    flux = 0.92 * flux + 0.08 * 1.0

    return {
        "time":        time,
        "flux":        flux,
        "flux_err":    np.ones(n) * 0.001,
        "ra":          291.3836,
        "dec":         41.0882,
        "candidate_id": "KEPLER-7b (Real Parameters)",
        "true_label":  "Planet Transit",
        "description": "Real NASA confirmed planet: period=4.885d, depth=0.96%, 6 transits in 30d"
    }