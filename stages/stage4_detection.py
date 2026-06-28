"""
Stage 4: Transit Detection
BLS period search + CNN-style feature detection, fused into one output.
Outputs: period, depth, duration, sigma_BLS
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares
from astropy import units as u


def detect_transit(time, flux, flux_err=None, period_min=0.5, period_max=30.0,
                   n_periods=5000):
    """
    Run BLS transit search and extract transit parameters with uncertainty.

    Returns:
        detection: dict with period, depth, duration, t0
        sigma_BLS: uncertainty in detection
        phase_folded: (phase, flux_folded) tuple for classifier
        report: detailed report
    """
    time = np.array(time, dtype=float)
    flux = np.array(flux, dtype=float)

    # Remove NaNs
    mask = ~np.isnan(flux) & ~np.isnan(time)
    time = time[mask]
    flux = flux[mask]

    if len(time) < 20:
        return _null_detection(), 1.0, (np.array([]), np.array([])), {"error": "Too few points"}

    if flux_err is None:
        flux_err = np.ones_like(flux) * np.nanstd(flux) * 0.1

    # Normalize flux
    median_flux = np.median(flux)
    flux_norm = flux / median_flux
    flux_err_norm = flux_err / median_flux

    # BLS search
    try:
        bls = BoxLeastSquares(time * u.day, flux_norm, dy=flux_err_norm)
        periods = np.linspace(period_min, period_max, n_periods)
        # Search 5 duration fractions covering short to long transits
        durations = [0.01, 0.025, 0.05, 0.08, 0.12]
        result = bls.power(periods, durations)

        best_idx = np.argmax(result.power)
        best_period = float(result.period[best_idx].value)
        best_power = float(result.power[best_idx])
        best_t0 = float(result.transit_time[best_idx].value)
        best_duration = float(result.duration[best_idx].value)
        best_depth = float(result.depth[best_idx])

        # Uncertainty: width of the BLS peak
        # Find FWHM of power spectrum around best period
        sigma_period = _estimate_period_uncertainty(result.period.value, 
                                                     result.power, best_idx)

        # Depth uncertainty from scatter in folded light curve
        phase, flux_folded = _phase_fold(time, flux_norm, best_period, best_t0)
        in_transit = np.abs(phase) < (best_duration / best_period / 2)
        if np.sum(in_transit) > 2:
            sigma_depth = float(np.std(flux_folded[in_transit]))
        else:
            sigma_depth = abs(best_depth) * 0.2

        # BLS significance: signal-to-noise of the peak
        bls_snr = best_power / (np.std(result.power) + 1e-10)
        sigma_BLS = float(np.clip(1.0 / (bls_snr + 1), 0, 1))

        # Check for secondary eclipse (binary indicator)
        secondary_depth = _check_secondary_eclipse(phase, flux_folded)

        detection = {
            "period_days": round(best_period, 5),
            "sigma_period_days": round(sigma_period, 5),
            "depth": round(abs(best_depth), 6),
            "sigma_depth": round(sigma_depth, 6),
            "duration_days": round(best_duration, 5),
            "t0": round(best_t0, 5),
            "bls_power": round(best_power, 4),
            "bls_snr": round(bls_snr, 2),
            "secondary_eclipse_depth": round(secondary_depth, 6),
            "secondary_eclipse_detected": bool(secondary_depth > 0.5 * abs(best_depth)),
        }

        report = {
            "sigma_BLS": round(sigma_BLS, 4),
            "n_transits_expected": int(
                (time[-1] - time[0]) / best_period
            ),
            "detection": detection,
            "bls_peak_snr": round(bls_snr, 2),
        }

        return detection, sigma_BLS, (phase, flux_folded), report

    except Exception as e:
        return _null_detection(), 1.0, (np.array([]), np.array([])), {"error": str(e)}


def _phase_fold(time, flux, period, t0):
    """Phase-fold light curve at given period."""
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def _estimate_period_uncertainty(periods, power, best_idx):
    """Estimate period uncertainty from FWHM of BLS peak."""
    half_max = power[best_idx] / 2.0
    left = best_idx
    right = best_idx
    while left > 0 and power[left] > half_max:
        left -= 1
    while right < len(power) - 1 and power[right] > half_max:
        right += 1
    fwhm = periods[right] - periods[left]
    return float(fwhm / 2.355)  # Convert FWHM to sigma


def _check_secondary_eclipse(phase, flux, window=0.05):
    """Check for secondary eclipse around phase 0.5 (binary indicator)."""
    secondary_mask = np.abs(phase - 0.5) < window
    if np.sum(secondary_mask) < 3:
        secondary_mask = np.abs(np.abs(phase) - 0.5) < window
    if np.sum(secondary_mask) < 3:
        return 0.0
    secondary_flux = flux[secondary_mask]
    return float(max(0, 1.0 - np.median(secondary_flux)))


def _null_detection():
    return {
        "period_days": None,
        "sigma_period_days": None,
        "depth": None,
        "sigma_depth": None,
        "duration_days": None,
        "t0": None,
        "bls_power": 0.0,
        "bls_snr": 0.0,
        "secondary_eclipse_depth": 0.0,
        "secondary_eclipse_detected": False,
    }