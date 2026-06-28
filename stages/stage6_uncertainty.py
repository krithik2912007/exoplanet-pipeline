"""
Stage 6: Trust-Adjusted Uncertainty Aggregation
Implements the hierarchical uncertainty framework:
  Level 1: Domain Trust T
  Level 2: Stage uncertainties (inflated by T)
  Level 3: Final confidence score
"""

import numpy as np
from stages.stage1_domain import compute_inflation_factor, adjust_thresholds


# Base weights learned from domain knowledge
# These would ideally be learned from validation data
BASE_WEIGHTS = {
    "denoise":    0.20,  # ML-based, domain-sensitive
    "blend":      0.35,  # Catalog-based, less domain-sensitive
    "BLS":        0.25,  # Physics-based, moderately domain-sensitive
    "classifier": 0.20,  # ML-based, domain-sensitive
}

# Domain sensitivity per stage (controls inflation factor)
DOMAIN_SENSITIVITY = {
    "denoise":    "high",
    "blend":      "low",
    "BLS":        "medium",
    "classifier": "high",
}


def aggregate_uncertainty(sigma_denoise, sigma_blend, sigma_BLS,
                          sigma_classifier, T, planet_probability):
    """
    Compute trust-adjusted total uncertainty and final confidence score.

    Args:
        sigma_denoise: uncertainty from denoising stage
        sigma_blend: uncertainty from blending correction
        sigma_BLS: uncertainty from transit detection
        sigma_classifier: uncertainty from ensemble classifier
        T: domain trust score in [0,1]
        planet_probability: P(planet) from classifier

    Returns:
        confidence: final confidence score in [0,1]
        sigma_total: aggregated uncertainty
        report: full uncertainty breakdown
        thresholds: T-adjusted decision thresholds
    """
    sigma_domain = 1.0 - T

    # Stage-specific inflation factors
    IF = {
        stage: compute_inflation_factor(sigma_domain, DOMAIN_SENSITIVITY[stage])
        for stage in DOMAIN_SENSITIVITY
    }

    # Effective (inflated) uncertainties
    sigma_eff = {
        "denoise":    sigma_denoise    * IF["denoise"],
        "blend":      sigma_blend      * IF["blend"],
        "BLS":        sigma_BLS        * IF["BLS"],
        "classifier": sigma_classifier * IF["classifier"],
    }

    # Weighted quadrature sum
    sigma_total_sq = sum(
        BASE_WEIGHTS[k] * sigma_eff[k]**2
        for k in BASE_WEIGHTS
    )
    sigma_total = float(np.sqrt(sigma_total_sq))

    # Final confidence: planet probability penalized by total uncertainty
    raw_confidence = planet_probability * (1.0 - sigma_total)
    confidence = float(np.clip(raw_confidence, 0, 1))

    # Confidence interval
    ci_lower = float(np.clip(planet_probability - 2 * sigma_total, 0, 1))
    ci_upper = float(np.clip(planet_probability + 2 * sigma_total, 0, 1))

    # T-adjusted decision thresholds
    thresholds = adjust_thresholds(T)

    # Dominant uncertainty source
    weighted_sigmas = {k: BASE_WEIGHTS[k] * sigma_eff[k]**2 for k in sigma_eff}
    dominant = max(weighted_sigmas, key=weighted_sigmas.get)

    report = {
        "confidence": round(confidence, 4),
        "sigma_total": round(sigma_total, 4),
        "confidence_interval": {
            "lower": round(ci_lower, 4),
            "upper": round(ci_upper, 4),
        },
        "domain_trust": round(T, 4),
        "sigma_domain": round(sigma_domain, 4),
        "inflation_factors": {k: round(v, 4) for k, v in IF.items()},
        "stage_uncertainties": {
            "raw": {
                "sigma_denoise":    round(sigma_denoise, 4),
                "sigma_blend":      round(sigma_blend, 4),
                "sigma_BLS":        round(sigma_BLS, 4),
                "sigma_classifier": round(sigma_classifier, 4),
            },
            "effective": {k: round(v, 4) for k, v in sigma_eff.items()},
        },
        "dominant_uncertainty_source": dominant,
        "thresholds": thresholds,
    }

    return confidence, sigma_total, report, thresholds
