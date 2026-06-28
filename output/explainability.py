"""
Explainability Module
Generates human-readable, scientifically meaningful output reports.
Includes ruling-out section, audit trail, and recommendations.
"""

import json
import numpy as np
from datetime import datetime


def generate_report(candidate_id, detection, classification, uncertainty_report,
                    quality_report, domain_report, blending_report,
                    refinement_result=None, final_confidence=None,
                    final_decision=None):
    """
    Generate complete explainability report.
    """
    thresholds = uncertainty_report.get("thresholds", {})
    confidence = final_confidence or uncertainty_report.get("confidence", 0)
    decision = final_decision or _make_decision(confidence, thresholds)

    # Ruling-out section
    ruling_out = _generate_ruling_out(detection, classification, blending_report)

    # Follow-up recommendation
    followup = _recommend_followup(
        uncertainty_report.get("dominant_uncertainty_source", "unknown"),
        blending_report, domain_report, confidence
    )

    # Audit trail
    audit = _build_audit_trail(
        quality_report, domain_report, blending_report,
        uncertainty_report, refinement_result
    )

    report = {
        "candidate_id": candidate_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "final_decision": decision,

        # Transit parameters
        "transit_parameters": {
            "period_days": detection.get("period_days"),
            "sigma_period_days": detection.get("sigma_period_days"),
            "depth_percent": round((detection.get("depth") or 0) * 100, 4),
            "duration_hours": round((detection.get("duration_days") or 0) * 24, 3),
            "bls_snr": detection.get("bls_snr"),
        },

        # Classification
        "classification": {
            "predicted_class": classification.get("predicted_class"),
            "probabilities": classification.get("probabilities"),
            "ensemble_agreement": round(
                1.0 - classification.get("sigma_classifier", 0.5), 3
            ),
        },

        # Confidence
        "confidence": {
            "score": round(confidence, 4),
            "interval": uncertainty_report.get("confidence_interval"),
            "sigma_total": uncertainty_report.get("sigma_total"),
            "domain_trust": domain_report.get("T"),
            "trust_level": domain_report.get("trust_level"),
        },

        # Why we ruled out alternatives
        "ruling_out": ruling_out,

        # Dominant uncertainty
        "dominant_uncertainty": {
            "source": uncertainty_report.get("dominant_uncertainty_source"),
            "stage_uncertainties": uncertainty_report.get("stage_uncertainties", {}).get("effective"),
        },

        # Data quality
        "data_quality": {
            "Q_data": quality_report.get("Q_data"),
            "flag": quality_report.get("flag"),
            "n_valid_points": quality_report.get("n_valid"),
        },

        # Blending
        "blending": {
            "contamination_ratio": blending_report.get("contamination_ratio"),
            "blending_flag": blending_report.get("blending_flag"),
            "centroid_shift_detected": blending_report.get("centroid_shift_detected"),
        },

        # Recommendation
        "recommended_followup": followup,

        # Refinement
        "refinement": {
            "triggered": refinement_result is not None,
            "action": refinement_result.get("action_taken") if refinement_result else None,
            "notes": refinement_result.get("notes") if refinement_result else [],
        },

        # Full audit trail
        "audit_trail": audit,
    }

    return report


def _make_decision(confidence, thresholds):
    accept_thresh = thresholds.get("accept", 0.85)
    review_thresh = thresholds.get("human_review", 0.50)

    if confidence >= accept_thresh:
        return "EXOPLANET CANDIDATE — AUTO-ACCEPTED"
    elif confidence >= review_thresh:
        return "AMBIGUOUS — REFINEMENT TRIGGERED"
    else:
        return "FLAGGED FOR HUMAN REVIEW"


def _generate_ruling_out(detection, classification, blending_report):
    ruling_out = {}

    # Eclipsing Binary
    sec_flag = detection.get("secondary_eclipse_detected", False)
    sec_depth = detection.get("secondary_eclipse_depth", 0)
    depth = detection.get("depth", 0)

    if not sec_flag:
        ruling_out["eclipsing_binary"] = (
            f"No secondary eclipse detected at phase ~0.5. "
            f"Secondary/primary depth ratio: {sec_depth/(depth+1e-10):.3f} (threshold: 0.5)"
        )
    else:
        ruling_out["eclipsing_binary"] = (
            f"WARNING: Secondary eclipse detected. "
            f"Depth ratio {sec_depth/(depth+1e-10):.3f} suggests possible binary system."
        )

    # Starspot
    probs = classification.get("probabilities", {})
    starspot_prob = probs.get("Starspot", 0)
    planet_prob = probs.get("Planet Transit", 0)

    if starspot_prob < 0.2:
        ruling_out["starspot"] = (
            f"Starspot probability low ({starspot_prob:.2%}). "
            f"Transit morphology inconsistent with rotational modulation "
            f"(localized dip, not sinusoidal)."
        )
    else:
        ruling_out["starspot"] = (
            f"Moderate starspot probability ({starspot_prob:.2%}). "
            f"Cannot fully rule out rotational modulation."
        )

    # Instrumental
    bls_snr = detection.get("bls_snr", 0)
    if bls_snr > 7:
        ruling_out["instrumental"] = (
            f"BLS SNR = {bls_snr:.1f} (>7). "
            f"Signal significance argues against pure instrumental artifact. "
            f"Transit consistent across multiple observation windows."
        )
    else:
        ruling_out["instrumental"] = (
            f"BLS SNR = {bls_snr:.1f} (low). "
            f"Cannot confidently rule out instrumental artifact."
        )

    # Blending
    centroid = blending_report.get("centroid_shift_detected", False)
    cont_ratio = blending_report.get("contamination_ratio", 1.0)
    if not centroid:
        ruling_out["stellar_blend"] = (
            f"No centroid shift detected during transit. "
            f"Contamination ratio: {cont_ratio:.2f}. "
            f"Transit depth corrected for blending."
        )
    else:
        ruling_out["stellar_blend"] = (
            f"WARNING: Centroid shift detected during transit. "
            f"Transit may originate from a neighboring star."
        )

    return ruling_out


def _recommend_followup(dominant_uncertainty, blending_report,
                         domain_report, confidence):
    recommendations = []

    if blending_report.get("blending_flag") in ["HIGH", "SEVERE"]:
        recommendations.append(
            "High-resolution imaging to resolve stellar blends in the aperture."
        )

    if blending_report.get("centroid_shift_detected"):
        recommendations.append(
            "Precise astrometric follow-up to identify which star hosts the transit."
        )

    if domain_report.get("T", 1.0) < 0.70:
        recommendations.append(
            "Pipeline operating outside training distribution. "
            "Validate results with instrument-specific analysis."
        )

    if dominant_uncertainty == "BLS":
        recommendations.append(
            "Additional photometric monitoring to confirm period and rule out aliases."
        )

    if dominant_uncertainty == "classifier":
        recommendations.append(
            "Radial velocity follow-up to confirm planetary mass and rule out binary."
        )

    if confidence > 0.80:
        recommendations.append(
            "High-confidence candidate. Recommend priority spectroscopic follow-up."
        )

    return recommendations if recommendations else ["Standard follow-up photometry recommended."]


def _build_audit_trail(quality_report, domain_report, blending_report,
                        uncertainty_report, refinement_result):
    audit = {
        "stage_0_quality": {
            "Q_data": quality_report.get("Q_data"),
            "flag": quality_report.get("flag"),
        },
        "stage_1_domain": {
            "T": domain_report.get("T"),
            "sigma_domain": domain_report.get("sigma_domain"),
            "trust_level": domain_report.get("trust_level"),
        },
        "stage_2_denoise": {
            "sigma_denoise": uncertainty_report.get(
                "stage_uncertainties", {}
            ).get("raw", {}).get("sigma_denoise"),
        },
        "stage_3_blend": {
            "contamination_ratio": blending_report.get("contamination_ratio"),
            "sigma_blend": blending_report.get("sigma_blend"),
            "blending_flag": blending_report.get("blending_flag"),
        },
        "stage_4_detection": {
            "sigma_BLS": uncertainty_report.get(
                "stage_uncertainties", {}
            ).get("raw", {}).get("sigma_BLS"),
        },
        "stage_5_classifier": {
            "sigma_classifier": uncertainty_report.get(
                "stage_uncertainties", {}
            ).get("raw", {}).get("sigma_classifier"),
        },
        "stage_6_aggregation": {
            "sigma_total": uncertainty_report.get("sigma_total"),
            "dominant_uncertainty": uncertainty_report.get("dominant_uncertainty_source"),
            "inflation_factors": uncertainty_report.get("inflation_factors"),
        },
        "refinement": {
            "triggered": refinement_result is not None,
            "action": refinement_result.get("action_taken") if refinement_result else None,
        },
    }
    return audit


def print_report(report):
    """Pretty print the report to console."""
    print("\n" + "="*60)
    print(f"  EXOPLANET DETECTION REPORT")
    print(f"  Candidate: {report['candidate_id']}")
    print("="*60)
    print(f"\n  DECISION: {report['final_decision']}")
    print(f"\n  CONFIDENCE: {report['confidence']['score']:.1%}")
    print(f"  Interval:   [{report['confidence']['interval']['lower']:.1%}, "
          f"{report['confidence']['interval']['upper']:.1%}]")
    print(f"  Domain Trust (T): {report['confidence']['domain_trust']} "
          f"({report['confidence']['trust_level']})")

    tp = report["transit_parameters"]
    print(f"\n  TRANSIT PARAMETERS:")
    print(f"    Period:    {tp['period_days']} ± {tp['sigma_period_days']} days")
    print(f"    Depth:     {tp['depth_percent']}%")
    print(f"    Duration:  {tp['duration_hours']} hours")
    print(f"    BLS SNR:   {tp['bls_snr']}")

    cls = report["classification"]
    print(f"\n  CLASSIFICATION: {cls['predicted_class']}")
    for c, p in cls["probabilities"].items():
        bar = "█" * int(p * 20)
        print(f"    {c:<20} {p:.1%}  {bar}")

    print(f"\n  RULING OUT ALTERNATIVES:")
    for alt, reason in report["ruling_out"].items():
        print(f"    [{alt}]: {reason[:80]}...")

    print(f"\n  DOMINANT UNCERTAINTY: {report['dominant_uncertainty']['source']}")
    print(f"  DATA QUALITY: {report['data_quality']['Q_data']} ({report['data_quality']['flag']})")
    print(f"  BLENDING: {report['blending']['blending_flag']} "
          f"(contamination ratio: {report['blending']['contamination_ratio']})")

    print(f"\n  RECOMMENDED FOLLOW-UP:")
    for rec in report["recommended_followup"]:
        print(f"    • {rec}")

    if report["refinement"]["triggered"]:
        print(f"\n  REFINEMENT: {report['refinement']['action']}")

    print("\n" + "="*60 + "\n")
