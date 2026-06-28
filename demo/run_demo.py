"""
Complete Demo Runner
Trains domain assessor and classifier on synthetic data,
then runs all 4 demo cases through the full pipeline.
"""

import numpy as np
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.generate_demo_data import (case1_clean_planet, case2_noisy_blended,
                                      case3_eclipsing_binary, case4_astrosat_like,
                                      inject_transit, stellar_background)
from pipeline import ExoplanetPipeline
from stages.stage4_detection import detect_transit
from stages.stage5_classifier import EnsembleClassifier


def build_training_data(n_each=150):
    """
    Generate synthetic training data.
    n_each=150 gives enough diversity for all three models
    without being too slow.
    """
    rng = np.random.default_rng(42)
    X, y, training_fluxes, training_curves = [], [], [], []

    # ── Planet Transits ───────────────────────────────────────
    for i in range(n_each):
        p   = rng.uniform(1, 25)
        d   = rng.uniform(0.001, 0.015)
        dur = rng.uniform(1.5, 5)
        noise = rng.uniform(0.0005, 0.002)
        time = np.linspace(0, 30, 720)
        flux = stellar_background(720, rng.uniform(0.0001, 0.001))
        flux += rng.normal(0, noise, 720)
        t0   = rng.uniform(0, p)
        flux = inject_transit(time, flux, p, d, dur/24, t0=t0)

        # Vary blending slightly
        if rng.random() > 0.5:
            blend = rng.uniform(0.7, 1.0)
            contaminant = stellar_background(720, 0.0002) + rng.normal(0, noise*0.5, 720)
            flux = blend * flux + (1-blend) * contaminant

        det, _, (ph, ff), _ = detect_transit(time, flux)
        from stages.stage5_classifier import _extract_features
        feat = _extract_features(det, ph, ff, noise, 0.05, 0.75)
        X.append(feat)
        y.append(0)
        training_fluxes.append(flux)
        training_curves.append(ff if len(ff) > 5 else np.ones(64))

    # ── Eclipsing Binaries ────────────────────────────────────
    for i in range(n_each):
        p    = rng.uniform(1, 20)
        d    = rng.uniform(0.01, 0.05)
        dur  = rng.uniform(2, 6)
        sec  = rng.uniform(0.005, d * 0.8)
        noise = rng.uniform(0.0005, 0.002)
        time = np.linspace(0, 30, 720)
        flux = stellar_background(720, rng.uniform(0.0001, 0.001))
        flux += rng.normal(0, noise, 720)
        t0   = rng.uniform(0, p)
        flux = inject_transit(time, flux, p, d, dur/24, t0=t0)
        flux = inject_transit(time, flux, p, sec, (dur*0.9)/24,
                              t0=t0 + p/2)

        det, _, (ph, ff), _ = detect_transit(time, flux)
        from stages.stage5_classifier import _extract_features
        feat = _extract_features(det, ph, ff, noise, 0.05, 0.75)
        X.append(feat)
        y.append(1)
        training_fluxes.append(flux)
        training_curves.append(ff if len(ff) > 5 else np.ones(64))

    # ── Starspots ─────────────────────────────────────────────
    for i in range(n_each):
        time  = np.linspace(0, 30, 720)
        Prot  = rng.uniform(5, 25)
        amp   = rng.uniform(0.002, 0.01)
        phase = rng.uniform(0, 2*np.pi)
        flux  = (1.0
                 + amp * np.sin(2*np.pi*time/Prot + phase)
                 + rng.normal(0, 0.001, 720))

        # Occasionally add second spot
        if rng.random() > 0.6:
            Prot2 = Prot * rng.uniform(0.9, 1.1)
            flux += amp * 0.5 * np.sin(2*np.pi*time/Prot2 + rng.uniform(0, 2*np.pi))

        det, _, (ph, ff), _ = detect_transit(time, flux)
        from stages.stage5_classifier import _extract_features
        feat = _extract_features(det, ph, ff, 0.002, 0.05, 0.70)
        X.append(feat)
        y.append(2)
        training_fluxes.append(flux)
        training_curves.append(ff if len(ff) > 5 else np.ones(64))

    # ── Instrumental ──────────────────────────────────────────
    for i in range(n_each):
        time = np.linspace(0, 30, 720)
        # Vary noise character
        noise_type = rng.integers(0, 3)
        if noise_type == 0:
            # Pure Gaussian
            flux = 1.0 + rng.normal(0, rng.uniform(0.005, 0.02), 720)
        elif noise_type == 1:
            # Systematic trend + noise
            flux = (1.0
                    + 0.01 * np.linspace(0, 1, 720)
                    + rng.normal(0, rng.uniform(0.005, 0.015), 720))
        else:
            # Periodic systematic (thermal)
            t = np.linspace(0, 30, 720)
            flux = (1.0
                    + 0.008 * np.sin(2*np.pi*t/0.065)
                    + rng.normal(0, 0.005, 720))

        det, _, (ph, ff), _ = detect_transit(time, flux)
        from stages.stage5_classifier import _extract_features
        feat = _extract_features(det, ph, ff, 0.01, 0.05, 0.40)
        X.append(feat)
        y.append(3)
        training_fluxes.append(flux)
        training_curves.append(ff if len(ff) > 5 else np.ones(64))

    return (np.nan_to_num(np.array(X)),
            np.array(y),
            training_fluxes,
            training_curves)


def main():
    print("\n" + "="*65)
    print("  EXOPLANET AI DETECTION PIPELINE — ISRO HACKATHON DEMO")
    print("="*65)

    print("\n[1/3] Building synthetic training dataset (n=150 per class)...")
    X_train, y_train, training_fluxes, training_curves = build_training_data(n_each=150)
    labels = ["Planet Transit", "Eclipsing Binary", "Starspot", "Instrumental"]
    for i, lbl in enumerate(labels):
        print(f"       {lbl}: {sum(y_train == i)} samples")

    print("\n[2/3] Training domain assessor and classifier...")
    pipeline = ExoplanetPipeline(verbose=False)

    # Train domain assessor on ALL training types — not just planets
    pipeline.fit_domain_assessor(training_fluxes)

    # Train classifier with phase-folded curves for CNN
    pipeline.classifier.fit(X_train, y_train, X_curves=training_curves)
    print("       Domain assessor fitted on all 4 classes.")
    print("       RF + CNN + GB classifier fitted.")

    cases = [
        case1_clean_planet(),
        case2_noisy_blended(),
        case3_eclipsing_binary(),
        case4_astrosat_like(),
    ]

    print("\n[3/3] Running all demo cases through pipeline...\n")

    all_reports = []
    for case in cases:
        print(f"  ─── {case['candidate_id']} ───")
        print(f"  Description: {case['description']}")
        print(f"  True label:  {case['true_label']}")

        report = pipeline.run(
            time=case['time'], flux=case['flux'], flux_err=case['flux_err'],
            ra=case['ra'], dec=case['dec'],
            candidate_id=case['candidate_id'],
            use_gaia=False,
        )
        all_reports.append((case, report))

    print("\n" + "="*65)
    print("  FINAL SUMMARY")
    print("="*65)
    print(f"  {'Candidate':<35} {'Conf':>6}  {'T':>5}  {'Decision'}")
    print(f"  {'-'*35} {'-'*6}  {'-'*5}  {'-'*25}")
    for case, report in all_reports:
        conf = report.get('confidence', {}).get('score', 0) \
               if isinstance(report.get('confidence'), dict) else 0
        T    = report.get('confidence', {}).get('domain_trust', '?') \
               if isinstance(report.get('confidence'), dict) else '?'
        dec  = report.get('final_decision', '')[:40]
        print(f"  {case['candidate_id']:<35} {conf:>5.1%}  {str(T):>5}  {dec}")

    print()
    output_path = os.path.join(os.path.dirname(__file__), "demo_reports.json")
    serializable = []
    for case, report in all_reports:
        r = {}
        for k, v in report.items():
            try:
                json.dumps(v)
                r[k] = v
            except Exception:
                r[k] = str(v)
        r['true_label'] = case['true_label']
        serializable.append(r)
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Full reports saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()