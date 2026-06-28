"""
Refinement Engine
Diagnosis-driven, runs exactly once.
Identifies the dominant uncertainty source and re-runs only that stage.
"""

import numpy as np
from stages.stage2_denoise import denoise
from stages.stage4_detection import detect_transit
from stages.stage6_uncertainty import aggregate_uncertainty


def run_refinement(time, flux, flux_err, ra, dec,
                   detection, phase, flux_folded,
                   sigma_denoise, sigma_blend, sigma_BLS, sigma_classifier,
                   T, planet_probability, uncertainty_report):
    """
    Targeted refinement based on dominant uncertainty source.
    Runs exactly once. Returns updated values.
    """
    dominant = uncertainty_report.get("dominant_uncertainty_source", "classifier")
    action_taken = "none"
    notes = []

    # --- Diagnose and act ---

    if dominant == "denoise" and sigma_denoise > 0.15:
        # Stronger denoising: smaller window for sharper transit preservation
        action_taken = "Stronger denoising (reduced window)"
        flux_new, sigma_per_point, sigma_denoise_new, _ = denoise(
            time, flux, flux_err, n_bootstrap=100, window_fraction=0.03
        )
        notes.append(f"sigma_denoise: {sigma_denoise:.4f} → {sigma_denoise_new:.4f}")
        sigma_denoise = sigma_denoise_new
        flux = flux_new

    elif dominant == "BLS" and sigma_BLS > 0.15:
        # Finer period grid search
        action_taken = "Finer BLS period grid search"
        if detection.get("period_days"):
            center = detection["period_days"]
            p_min = max(0.5, center * 0.8)
            p_max = center * 1.2
        else:
            p_min, p_max = 0.5, 30.0

        det_new, sigma_BLS_new, (phase_new, ff_new), _ = detect_transit(
            time, flux, flux_err,
            period_min=p_min, period_max=p_max, n_periods=20000
        )
        notes.append(f"sigma_BLS: {sigma_BLS:.4f} → {sigma_BLS_new:.4f}")

        if sigma_BLS_new < sigma_BLS:
            detection = det_new
            sigma_BLS = sigma_BLS_new
            phase = phase_new
            flux_folded = ff_new

    elif dominant == "blend" and sigma_blend > 0.15:
        # Can't re-query Gaia in refinement, but we can widen uncertainty bounds
        action_taken = "Conservative blending uncertainty acknowledged"
        notes.append("Blend correction uncertainty is catalog-limited")
        notes.append("Recommend high-resolution follow-up imaging")

    elif dominant == "classifier" and sigma_classifier > 0.15:
        # Flag for secondary eclipse re-examination
        action_taken = "Secondary eclipse region re-examined"
        sec_dep = detection.get("secondary_eclipse_depth", 0)
        depth = detection.get("depth", 0)
        notes.append(f"Secondary/primary depth ratio: {sec_dep/(depth+1e-10):.3f}")
        if sec_dep > 0.5 * depth:
            notes.append("Strong secondary eclipse → likely Eclipsing Binary")
        else:
            notes.append("No strong secondary → planet interpretation supported")

    # Recompute uncertainty with updated values
    confidence_new, sigma_total_new, report_new, thresholds_new = aggregate_uncertainty(
        sigma_denoise, sigma_blend, sigma_BLS, sigma_classifier, T, planet_probability
    )

    return {
        "action_taken": action_taken,
        "notes": notes,
        "updated_detection": detection,
        "updated_phase": phase,
        "updated_flux_folded": flux_folded,
        "updated_sigmas": {
            "sigma_denoise":    sigma_denoise,
            "sigma_blend":      sigma_blend,
            "sigma_BLS":        sigma_BLS,
            "sigma_classifier": sigma_classifier,
        },
        "updated_confidence": round(confidence_new, 4),
        "updated_sigma_total": round(sigma_total_new, 4),
        "updated_uncertainty_report": report_new,
        "updated_thresholds": thresholds_new,
    }
