# Exoplanet AI Detection Pipeline
### ISRO Hackathon Submission

---

## What This Is

An **uncertainty-aware adaptive pipeline** that detects exoplanet transit signals 
from noisy, contaminated astronomical light curves.

Unlike a standard ML classifier, this pipeline:
- Tracks uncertainty at every stage
- Uses Domain Trust T to modulate all downstream confidences
- Performs diagnosis-driven refinement (runs once only)
- Generates a complete audit trail for scientific verification

---

## Pipeline Architecture

```
Raw Light Curve + Metadata
        ↓
[Stage 0] Data Quality Assessment      → Q_data ∈ [0,1]
        ↓
[Stage 1] Domain Assessment            → T ∈ [0,1], σ_domain
        ↓
[Stage 2] Transit-Aware Denoising      → μ_clean, σ_denoise
        ↓
[Stage 3] Blending Correction          → μ_corrected, σ_blend (Gaia catalog)
        ↓
[Stage 4] BLS Transit Detection        → period, depth, σ_BLS
        ↓
[Stage 5] Ensemble Classifier          → class probs, σ_classifier
        ↓
[Stage 6] Trust-Adjusted Aggregation   → σ_total, confidence
        ↓
   Confidence Zone Check
 >T-thresh: Accept | 50-85%: Refine | <50%: Human Review
        ↓
[Refinement] Diagnosis-driven (once)
        ↓
[Explainability] ruling_out + audit trail
        ↓
Verified Candidate / Human Review Queue
```

---

## Key Innovation: Hierarchical Uncertainty

σ_domain (Domain Trust T) is a **meta-uncertainty** that governs all stage uncertainties:

```
σ_total = √(w1·σ_denoise_eff² + w2·σ_blend_eff² + w3·σ_BLS_eff² + w4·σ_classifier_eff²)

where σ_stage_eff = σ_stage × exp(α_stage × σ_domain)
```

ML-heavy stages (denoise, classifier) receive higher inflation than catalog-based stages (blend).

---

## Installation

```bash
pip install lightkurve astroquery astropy scikit-learn matplotlib scipy numpy
```

---

## Quick Start

```python
from pipeline import ExoplanetPipeline
from demo.run_demo import build_training_data

# Train
X_train, y_train, training_fluxes = build_training_data(n_each=100)
pipeline = ExoplanetPipeline(verbose=True)
pipeline.fit_domain_assessor(training_fluxes)
pipeline.classifier.fit(X_train, y_train)

# Run on your light curve
report = pipeline.run(
    time=time_array,
    flux=flux_array,
    flux_err=flux_err_array,
    ra=285.68, dec=50.24,
    candidate_id="MY_STAR_001",
    use_gaia=True   # Set True to query Gaia DR3
)
```

---

## Demo Cases

Run all 4 demo cases:
```bash
cd exoplanet_pipeline
python demo/run_demo.py
```

| Case | Description | True Label |
|------|------------|------------|
| DEMO_001 | Clean planet transit, period=10d | Planet Transit |
| DEMO_002 | Noisy + heavily blended field | Planet Transit |
| DEMO_003 | Eclipsing binary with secondary eclipse | Eclipsing Binary |
| DEMO_004 | AstroSat-like cadence (OOD test) | Planet Transit |

---

## Output Format

```json
{
  "candidate_id": "DEMO_001_PLANET",
  "final_decision": "EXOPLANET CANDIDATE — AUTO-ACCEPTED",
  "transit_parameters": {
    "period_days": 10.013,
    "depth_percent": 0.78,
    "duration_hours": 3.2,
    "bls_snr": 7.96
  },
  "confidence": {
    "score": 0.69,
    "interval": {"lower": 0.57, "upper": 0.95},
    "domain_trust": 1.0,
    "trust_level": "HIGH"
  },
  "classification": {
    "predicted_class": "Planet Transit",
    "probabilities": {
      "Planet Transit": 0.761,
      "Eclipsing Binary": 0.029,
      "Starspot": 0.172,
      "Instrumental": 0.038
    }
  },
  "ruling_out": {
    "eclipsing_binary": "No secondary eclipse at phase 0.5",
    "starspot": "Morphology inconsistent with rotational modulation",
    "instrumental": "BLS SNR > 7, signal persistent"
  },
  "dominant_uncertainty": {"source": "blend"},
  "recommended_followup": ["High-resolution imaging to resolve blend"],
  "audit_trail": {...}
}
```

---

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| F2 Score | Penalizes missed planets more than false alarms |
| AUC-PR | Handles class imbalance correctly |
| Period Recovery Rate | Physically meaningful |
| Expected Calibration Error | Tests if uncertainty estimates are honest |
| Detection Efficiency Curve | Standard in real mission pipelines |

---

## Project Structure

```
exoplanet_pipeline/
├── pipeline.py              # Main orchestrator
├── stages/
│   ├── stage0_quality.py    # Data quality assessment
│   ├── stage1_domain.py     # Domain trust T
│   ├── stage2_denoise.py    # Transit-aware denoising
│   ├── stage3_blending.py   # Gaia catalog + correction
│   ├── stage4_detection.py  # BLS transit search
│   ├── stage5_classifier.py # Ensemble classifier
│   └── stage6_uncertainty.py# Trust-adjusted aggregation
├── refinement/
│   └── refinement_engine.py # Diagnosis-driven refinement
├── output/
│   └── explainability.py    # Audit trail + report generator
└── demo/
    ├── generate_demo_data.py # Synthetic light curve generator
    └── run_demo.py           # Full demo runner
```
