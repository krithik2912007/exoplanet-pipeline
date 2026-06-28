"""
Stage 2: Three-Method Ensemble Denoising
Methods: Savitzky-Golay + Wavelet (CWT) + Median Filter
σ_denoise = agreement between three independent methods

Key insight: if three fundamentally different methods preserve
the same transit at the same location with the same depth,
confidence in the cleaned signal is high.
If they disagree, that disagreement is propagated as σ_denoise.
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import uniform_filter1d


def _savgol_denoise(flux, window_fraction=0.15):
    """
    Savitzky-Golay: fits local polynomials.
    Good at: preserving peak shapes, smooth trends.
    """
    n = len(flux)
    window = max(51, int(n * window_fraction))
    if window % 2 == 0:
        window += 1
    window = min(window, n - 2)
    polyorder = min(2, window - 1)
    try:
        trend = savgol_filter(flux, window, polyorder)
        detrended = flux / (trend + 1e-10)
        noise_win = min(5, n - 2)
        if noise_win % 2 == 0:
            noise_win += 1
        if noise_win >= 3:
            detrended = savgol_filter(detrended, noise_win, 1)
        return detrended * trend
    except Exception:
        return flux.copy()


def _wavelet_denoise(flux, wavelet_scale=8):
    """
    Wavelet-inspired denoising via multi-scale decomposition.
    Uses uniform filters at different scales to separate
    stellar trend (large scale) from noise (small scale).
    Good at: multi-scale signal separation, transient detection.
    """
    try:
        n = len(flux)
        # Coarse scale: long-baseline stellar trend
        coarse_scale = max(int(n * 0.15), 51)
        trend = uniform_filter1d(flux, size=coarse_scale, mode='nearest')

        # Fine scale: noise smoother
        residual = flux / (trend + 1e-10)
        fine_scale = max(3, wavelet_scale)
        if fine_scale % 2 == 0:
            fine_scale += 1
        smoothed_residual = uniform_filter1d(residual, size=fine_scale,
                                              mode='nearest')
        return smoothed_residual * trend
    except Exception:
        return flux.copy()


def _median_denoise(flux, window_fraction=0.15):
    """
    Median filter: robust to outliers (cosmic rays, spikes).
    Good at: edge preservation, spike removal.
    Unlike Savitzky-Golay, median filter is non-parametric
    and makes no assumptions about signal shape.
    """
    try:
        n = len(flux)
        # Large window for trend
        trend_win = max(51, int(n * window_fraction))
        if trend_win % 2 == 0:
            trend_win += 1
        trend_win = min(trend_win, n - 2 if n > 3 else n)
        trend = medfilt(flux, kernel_size=trend_win)
        residual = flux / (trend + 1e-10)
        # Small median filter for noise
        noise_win = min(5, n - 2 if n > 3 else n)
        if noise_win % 2 == 0:
            noise_win += 1
        if noise_win >= 3:
            residual = medfilt(residual, kernel_size=noise_win)
        return residual * trend
    except Exception:
        return flux.copy()


def denoise(time, flux, flux_err=None, n_bootstrap=50, window_fraction=0.15):
    """
    Three-method ensemble denoising.

    Strategy:
      1. Run Savitzky-Golay, Wavelet, and Median filter independently
      2. Compute per-point agreement (std across three outputs)
      3. Final cleaned signal = mean of three outputs
      4. σ_denoise = mean disagreement = how trustworthy the denoising is

    If three fundamentally different methods agree → high confidence
    If they disagree → high σ_denoise → propagated to final confidence

    Returns:
        flux_clean:        mean of three denoised outputs
        sigma_per_point:   per-point std across three methods
        sigma_global:      single global uncertainty estimate
        report:            detailed breakdown
    """
    flux = np.array(flux, dtype=float)
    time = np.array(time, dtype=float)

    nan_mask = np.isnan(flux)
    if np.sum(~nan_mask) < 10:
        return flux, np.ones_like(flux)*0.5, 0.5, {"error": "Too few valid points"}

    # Interpolate NaNs for filtering
    flux_interp = flux.copy()
    if nan_mask.any():
        flux_interp[nan_mask] = np.interp(
            time[nan_mask], time[~nan_mask], flux[~nan_mask]
        )

    # Run three independent denoisers
    out_savgol  = _savgol_denoise(flux_interp, window_fraction)
    out_wavelet = _wavelet_denoise(flux_interp)
    out_median  = _median_denoise(flux_interp, window_fraction)

    # Stack outputs: shape (3, n_points)
    stack = np.array([out_savgol, out_wavelet, out_median])

    # Final signal: mean across three methods
    flux_clean = np.mean(stack, axis=0)

    # Per-point agreement: std across three methods
    # This is the key innovation — disagreement = denoising uncertainty
    sigma_per_point = np.std(stack, axis=0)

    # Restore NaNs
    flux_clean[nan_mask] = np.nan
    sigma_per_point[nan_mask] = np.nan

    # Global uncertainty = mean disagreement, normalized by flux scale
    flux_scale = np.nanstd(flux_interp) + 1e-10
    sigma_global = float(np.nanmean(sigma_per_point) / flux_scale)
    sigma_global = float(np.clip(sigma_global, 0, 1))

    # Individual method outputs for inspection
    out_savgol[nan_mask] = np.nan
    out_wavelet[nan_mask] = np.nan
    out_median[nan_mask] = np.nan

    # Agreement score: 1 = perfect agreement, 0 = total disagreement
    agreement_score = float(1.0 - np.clip(sigma_global, 0, 1))

    # Transit depth consistency check across methods
    min_flux = {
        "savgol":  float(np.nanmin(out_savgol)),
        "wavelet": float(np.nanmin(out_wavelet)),
        "median":  float(np.nanmin(out_median)),
    }
    depth_consistency = float(np.std(list(min_flux.values())))

    original_std = float(np.nanstd(flux))
    denoised_std = float(np.nanstd(flux_clean))
    snr_improvement = original_std / (denoised_std + 1e-10)

    report = {
        "sigma_denoise":       round(sigma_global, 6),
        "agreement_score":     round(agreement_score, 4),
        "depth_consistency":   round(depth_consistency, 6),
        "method_outputs": {
            "savitzky_golay_min_flux": round(min_flux["savgol"],  6),
            "wavelet_min_flux":        round(min_flux["wavelet"], 6),
            "median_min_flux":         round(min_flux["median"],  6),
        },
        "original_std":        round(original_std, 6),
        "denoised_std":        round(denoised_std, 6),
        "snr_improvement_factor": round(snr_improvement, 3),
        "method": "ensemble (Savitzky-Golay + Wavelet + Median Filter)",
        "n_methods": 3,
        "interpretation": (
            "High confidence — three methods agree"
            if agreement_score > 0.8 else
            "Moderate confidence — methods partially agree"
            if agreement_score > 0.5 else
            "Low confidence — methods disagree significantly"
        ),
    }

    return flux_clean, sigma_per_point, sigma_global, report