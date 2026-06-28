"""
Generate all presentation plots.
Run: python generate_plots.py
Output: plots/ directory
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.generate_demo_data import (case1_clean_planet, case2_noisy_blended,
                                      case3_eclipsing_binary, case4_astrosat_like,
                                      inject_transit, stellar_background)
from demo.run_demo import build_training_data
from pipeline import ExoplanetPipeline
from stages.stage2_denoise import denoise
from stages.stage4_detection import detect_transit
from stages.stage5_classifier import EnsembleClassifier
from visualize import (plot_denoising, plot_phase_folded,
                        plot_uncertainty_budget, plot_all_cases_summary,
                        plot_pipeline_diagram)

os.makedirs("plots", exist_ok=True)


def run_pipeline_verbose(pipeline, case):
    """Run pipeline and collect intermediate outputs for plotting."""
    import warnings; warnings.filterwarnings("ignore")
    from stages.stage0_quality import assess_quality
    from stages.stage1_domain import adjust_thresholds
    from stages.stage2_denoise import denoise
    from stages.stage3_blending import correct_blending
    from stages.stage4_detection import detect_transit
    from stages.stage6_uncertainty import aggregate_uncertainty

    time     = np.array(case["time"])
    flux_raw = np.array(case["flux"])
    flux_err = np.array(case["flux_err"])

    Q_data, quality_rep  = assess_quality(time, flux_raw, flux_err)
    T, domain_rep        = pipeline.domain_assessor.compute_trust(flux_raw)
    flux_clean, _, sd, _ = denoise(time, flux_raw, flux_err)
    flux_corr, cr, sb, blend_rep = correct_blending(
        time, flux_clean, case["ra"], case["dec"], use_gaia=False)
    detection, sigma_BLS, (phase, ff), _ = detect_transit(time, flux_corr, flux_err)
    classification = pipeline.classifier.predict(
        detection, phase, ff, sd, sb, Q_data)
    planet_prob = classification["probabilities"].get("Planet Transit", 0)
    sc = classification["sigma_classifier"]
    confidence, sigma_total, unc_rep, thresholds = aggregate_uncertainty(
        sd, sb, sigma_BLS, sc, T, planet_prob)

    return {
        "flux_raw":          flux_raw,
        "flux_clean":        flux_clean,
        "phase":             phase,
        "flux_folded":       ff,
        "detection":         detection,
        "classification":    classification,
        "uncertainty_report": unc_rep,
        "domain_report":     domain_rep,
        "quality_report":    {"Q_data": Q_data, "flag": "ACCEPTABLE"},
        "blend_report":      blend_rep,
        "confidence":        confidence,
        "T":                 T,
    }


def main():
    print("\n=== Generating All Pipeline Plots ===\n")

    # Train pipeline
    print("[1/3] Training pipeline...")
    X_train, y_train, training_fluxes = build_training_data(n_each=100)
    pipeline = ExoplanetPipeline(verbose=False)
    pipeline.fit_domain_assessor(training_fluxes[:200])
    pipeline.classifier.fit(X_train, y_train)
    print("      Done.\n")

    # Collect all cases
    cases = [
        case1_clean_planet(),
        case2_noisy_blended(),
        case3_eclipsing_binary(),
        case4_astrosat_like(),
    ]

    print("[2/3] Running pipeline on all cases...")
    results = []
    for case in cases:
        r = run_pipeline_verbose(pipeline, case)
        r["label"]      = case["candidate_id"].replace("DEMO_", "")
        r["true_label"] = case["true_label"]
        r["time"]       = np.array(case["time"])
        r["decision"]   = pipeline.run(
            case["time"], case["flux"], case["flux_err"],
            case["ra"], case["dec"], case["candidate_id"],
            use_gaia=False
        ).get("final_decision", "")
        results.append(r)
    print("      Done.\n")

    print("[3/3] Generating plots...\n")

    # Plot 1: Denoising comparison (Case 1)
    print("  Plot 1: Denoising comparison")
    plot_denoising(
        results[0]["time"], results[0]["flux_raw"], results[0]["flux_clean"],
        results[0]["detection"], "DEMO_001_PLANET",
        "plots/01_denoising.png"
    )

    # Plot 2: Phase-folded — Planet (Case 1)
    print("  Plot 2: Phase-folded — Planet")
    plot_phase_folded(
        results[0]["phase"], results[0]["flux_folded"],
        results[0]["detection"], results[0]["classification"],
        "plots/02_phase_folded_planet.png"
    )

    # Plot 3: Phase-folded — Binary (Case 3)
    print("  Plot 3: Phase-folded — Binary")
    plot_phase_folded(
        results[2]["phase"], results[2]["flux_folded"],
        results[2]["detection"], results[2]["classification"],
        "plots/03_phase_folded_binary.png"
    )

    # Plot 4: Uncertainty budget — Case 1
    print("  Plot 4: Uncertainty budget (clean planet)")
    plot_uncertainty_budget(
        results[0]["uncertainty_report"],
        results[0]["domain_report"],
        "DEMO_001_PLANET",
        "plots/04_uncertainty_budget_planet.png"
    )

    # Plot 5: Uncertainty budget — Case 4 (AstroSat / OOD)
    print("  Plot 5: Uncertainty budget (AstroSat OOD)")
    plot_uncertainty_budget(
        results[3]["uncertainty_report"],
        results[3]["domain_report"],
        "DEMO_004_ASTROSAT",
        "plots/05_uncertainty_budget_astrosat.png"
    )

    # Plot 6: All 4 cases summary
    print("  Plot 6: All cases summary")
    plot_all_cases_summary(results, "plots/06_all_cases_summary.png")

    # Plot 7: Pipeline architecture
    print("  Plot 7: Pipeline architecture diagram")
    plot_pipeline_diagram("plots/07_pipeline_architecture.png")

    print("\n=== All plots saved to ./plots/ ===")
    print("\nFiles:")
    for f in sorted(os.listdir("plots")):
        size = os.path.getsize(f"plots/{f}") // 1024
        print(f"  plots/{f}  ({size} KB)")


if __name__ == "__main__":
    main()
