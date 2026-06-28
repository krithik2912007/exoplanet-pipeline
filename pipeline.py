"""
Main Pipeline Orchestrator
Connects all stages in order, handles uncertainty propagation,
triggers refinement when needed, generates final report.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stages.stage0_quality import assess_quality
from stages.stage1_domain import DomainAssessor, adjust_thresholds
from stages.stage2_denoise import denoise
from stages.stage3_blending import correct_blending
from stages.stage4_detection import detect_transit
from stages.stage5_classifier import EnsembleClassifier
from stages.stage6_uncertainty import aggregate_uncertainty
from refinement.refinement_engine import run_refinement
from output.explainability import generate_report, print_report


class ExoplanetPipeline:
    def __init__(self, verbose=True):
        self.domain_assessor = DomainAssessor(n_components=10)
        self.classifier = EnsembleClassifier(n_estimators=200)
        self.verbose = verbose
        self._log("Pipeline initialized.")

    def _log(self, msg):
        if self.verbose:
            print(f"  [Pipeline] {msg}")

    def fit_domain_assessor(self, flux_list):
        """Fit domain assessor on training light curves."""
        self._log(f"Fitting domain assessor on {len(flux_list)} light curves...")
        self.domain_assessor.fit(flux_list)
        self._log("Domain assessor fitted.")

    def fit_classifier(self, X, y):
        """Fit ensemble classifier on feature matrix."""
        self._log(f"Fitting classifier on {len(X)} examples...")
        self.classifier.fit(X, y)
        self._log("Classifier fitted.")

    def run(self, time, flux, flux_err=None, ra=0.0, dec=0.0,
            candidate_id="UNKNOWN", use_gaia=False):
        """
        Run the full pipeline on a single light curve.

        Args:
            time: array of timestamps (days)
            flux: array of normalized flux values
            flux_err: optional flux errors
            ra, dec: sky coordinates (degrees) for Gaia query
            candidate_id: identifier string
            use_gaia: whether to query Gaia (requires internet)

        Returns:
            report: complete JSON-serializable report dict
        """
        self._log(f"Starting pipeline for {candidate_id}")
        time = np.array(time, dtype=float)
        flux = np.array(flux, dtype=float)

        # ── Stage 0: Data Quality ──────────────────────────────────
        self._log("Stage 0: Data Quality Assessment")
        Q_data, quality_report = assess_quality(time, flux, flux_err)
        self._log(f"  Q_data = {Q_data:.3f} ({quality_report['flag']})")

        if quality_report["flag"] == "UNUSABLE":
            self._log("  Data quality too poor. Flagging for human review.")
            return self._unusable_report(candidate_id, quality_report)

        # ── Stage 1: Domain Assessment ─────────────────────────────
        self._log("Stage 1: Domain Assessment")
        T, domain_report = self.domain_assessor.compute_trust(flux)
        self._log(f"  T = {T:.3f} ({domain_report['trust_level']})")

        # ── Stage 2: Denoising ─────────────────────────────────────
        self._log("Stage 2: Denoising")
        flux_clean, sigma_per_point, sigma_denoise, denoise_report = denoise(
            time, flux, flux_err
        )
        self._log(f"  sigma_denoise = {sigma_denoise:.5f}, "
                  f"SNR improvement = {denoise_report.get('snr_improvement_factor', '?'):.2f}x")

        # ── Stage 3: Blending Correction ───────────────────────────
        self._log("Stage 3: Blending Correction")
        flux_corrected, cont_ratio, sigma_blend, blending_report = correct_blending(
            time, flux_clean, ra, dec, use_gaia=use_gaia
        )
        self._log(f"  contamination_ratio = {cont_ratio:.3f}, "
                  f"sigma_blend = {sigma_blend:.4f} ({blending_report['blending_flag']})")

        # ── Stage 4: Transit Detection ─────────────────────────────
        self._log("Stage 4: Transit Detection (BLS)")
        detection, sigma_BLS, (phase, flux_folded), detection_report = detect_transit(
            time, flux_corrected, flux_err
        )
        self._log(f"  Period = {detection.get('period_days')} days, "
                  f"Depth = {(detection.get('depth') or 0)*100:.4f}%, "
                  f"BLS SNR = {detection.get('bls_snr'):.2f}")

        # ── Stage 5: Classification ────────────────────────────────
        self._log("Stage 5: Ensemble Classification")
        classification = self.classifier.predict(
            detection, phase, flux_folded,
            sigma_denoise, sigma_blend, Q_data
        )
        planet_prob = classification["probabilities"].get("Planet Transit", 0.0)
        sigma_classifier = classification["sigma_classifier"]
        self._log(f"  {classification['predicted_class']} "
                  f"(P={planet_prob:.2%}), sigma_classifier = {sigma_classifier:.4f}")

        # ── Stage 6: Uncertainty Aggregation ──────────────────────
        self._log("Stage 6: Uncertainty Aggregation")
        confidence, sigma_total, uncertainty_report, thresholds = aggregate_uncertainty(
            sigma_denoise, sigma_blend, sigma_BLS, sigma_classifier, T, planet_prob
        )
        self._log(f"  Confidence = {confidence:.3f}, sigma_total = {sigma_total:.4f}")
        self._log(f"  Thresholds → Accept: {thresholds['accept']:.2f}, "
                  f"Review: {thresholds['human_review']:.2f}")

        # ── Decision & Refinement ──────────────────────────────────
        refinement_result = None
        final_confidence = confidence
        final_decision = None

        if confidence >= thresholds["accept"]:
            final_decision = "EXOPLANET CANDIDATE — AUTO-ACCEPTED"
            self._log(f"  Decision: AUTO-ACCEPTED")

        elif confidence >= thresholds["human_review"]:
            self._log(f"  Confidence in ambiguous zone. Triggering refinement...")
            refinement_result = run_refinement(
                time, flux_corrected, flux_err, ra, dec,
                detection, phase, flux_folded,
                sigma_denoise, sigma_blend, sigma_BLS, sigma_classifier,
                T, planet_prob, uncertainty_report
            )
            final_confidence = refinement_result["updated_confidence"]
            uncertainty_report = refinement_result["updated_uncertainty_report"]
            detection = refinement_result["updated_detection"]

            self._log(f"  Refinement: {refinement_result['action_taken']}")
            self._log(f"  Confidence: {confidence:.3f} → {final_confidence:.3f}")

            if final_confidence >= thresholds["accept"]:
                final_decision = "EXOPLANET CANDIDATE — ACCEPTED AFTER REFINEMENT"
            elif final_confidence >= thresholds["human_review"]:
                final_decision = "AMBIGUOUS — FLAGGED FOR HUMAN REVIEW (MEDIUM PRIORITY)"
            else:
                final_decision = "FLAGGED FOR HUMAN REVIEW (LOW PRIORITY)"

        else:
            final_decision = "FLAGGED FOR HUMAN REVIEW (LOW CONFIDENCE)"
            self._log(f"  Low confidence. Routing to human review.")

        # ── Explainability Report ──────────────────────────────────
        self._log("Generating explainability report...")
        report = generate_report(
            candidate_id=candidate_id,
            detection=detection,
            classification=classification,
            uncertainty_report=uncertainty_report,
            quality_report=quality_report,
            domain_report=domain_report,
            blending_report=blending_report,
            refinement_result=refinement_result,
            final_confidence=final_confidence,
            final_decision=final_decision,
        )

        if self.verbose:
            print_report(report)

        return report

    def _unusable_report(self, candidate_id, quality_report):
        return {
            "candidate_id": candidate_id,
            "final_decision": "REJECTED — DATA QUALITY TOO POOR",
            "data_quality": quality_report,
            "confidence": {"score": 0.0},
        }
