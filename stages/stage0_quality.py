"""
Stage 0: Data Quality Assessment
Outputs: Q_data in [0,1] + quality breakdown
"""

import numpy as np


def assess_quality(time, flux, flux_err=None):
    scores = {}

    # 1. Completeness
    nan_mask = np.isnan(flux)
    completeness = 1.0 - np.sum(nan_mask) / len(flux)
    scores["completeness"] = float(completeness)

    time_clean = time[~nan_mask]
    flux_clean = flux[~nan_mask]

    # 2. SNR estimate
    if len(flux_clean) > 10:
        window = max(5, len(flux_clean) // 20)
        noise = np.median([
            np.std(flux_clean[i:i+window])
            for i in range(0, len(flux_clean) - window, window)
        ])
        signal_range = np.percentile(flux_clean, 95) - np.percentile(flux_clean, 5)
        snr = signal_range / (noise + 1e-10)
        snr_score = float(np.clip((snr - 3) / 17, 0, 1))
    else:
        snr_score = 0.0
    scores["snr_score"] = snr_score

    # 3. Gap penalty
    if len(time_clean) > 1:
        gaps = np.diff(np.sort(time_clean))
        median_cadence = np.median(gaps)
        large_gaps = gaps[gaps > 5 * median_cadence]
        gap_fraction = np.sum(large_gaps) / (time_clean[-1] - time_clean[0] + 1e-10)
        gap_score = float(np.clip(1.0 - gap_fraction, 0, 1))
    else:
        gap_score = 0.0
    scores["gap_score"] = gap_score

    # 4. Saturation / outlier flag
    if len(flux_clean) > 0:
        median_flux = np.median(flux_clean)
        mad = np.median(np.abs(flux_clean - median_flux))
        outlier_fraction = np.sum(np.abs(flux_clean - median_flux) > 10 * mad) / len(flux_clean)
        saturation_score = float(np.clip(1.0 - 10 * outlier_fraction, 0, 1))
    else:
        saturation_score = 0.0
    scores["saturation_score"] = saturation_score

    # 5. Systematics flag
    if len(time_clean) > 10:
        try:
            t_norm = (time_clean - np.mean(time_clean)) / (np.std(time_clean) + 1e-10)
            coeffs = np.polyfit(t_norm, flux_clean, 2)
            trend_amplitude = np.abs(coeffs[0]) + np.abs(coeffs[1])
            flux_std = np.std(flux_clean) + 1e-10
            systematics_score = float(np.clip(1.0 - trend_amplitude / flux_std, 0, 1))
        except Exception:
            systematics_score = 0.5
    else:
        systematics_score = 0.5
    scores["systematics_score"] = systematics_score

    weights = {
        "completeness":      0.25,
        "snr_score":         0.30,
        "gap_score":         0.20,
        "saturation_score":  0.10,
        "systematics_score": 0.15,
    }
    Q_data = float(sum(weights[k] * scores[k] for k in weights))

    report = {
        "Q_data": round(Q_data, 4),
        "n_points": int(len(flux)),
        "n_valid": int(np.sum(~nan_mask)),
        "metrics": {k: round(v, 4) for k, v in scores.items()},
        "flag": _quality_flag(Q_data),
    }
    return Q_data, report


def _quality_flag(Q):
    if Q >= 0.80:
        return "GOOD"
    elif Q >= 0.60:
        return "ACCEPTABLE"
    elif Q >= 0.40:
        return "POOR"
    else:
        return "UNUSABLE"
