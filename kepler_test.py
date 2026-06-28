"""
Real Kepler Data Test
Downloads confirmed Kepler planet light curves and runs the pipeline.
Run: python kepler_test.py
Requires: pip install lightkurve
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.run_demo import build_training_data
from pipeline import ExoplanetPipeline

import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize import _dark_style, COLORS, plot_phase_folded
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Known Kepler planets for testing
KEPLER_TARGETS = [
    {
        "name":          "Kepler-7b",
        "star":          "Kepler-7",
        "true_period":   4.8854892,
        "true_depth":    0.0096,
        "true_label":    "Planet Transit",
        "ra":            291.3836,
        "dec":           41.0882,
        "description":   "Hot Jupiter, very clear transit signal",
    },
    {
        "name":          "Kepler-10b",
        "star":          "Kepler-10",
        "true_period":   0.8374907,
        "true_depth":    0.000365,
        "true_label":    "Planet Transit",
        "ra":            285.6789,
        "dec":           50.2415,
        "description":   "Rocky super-Earth, very shallow transit",
    },
]


def download_kepler(star_name, quarter=3):
    """Download Kepler light curve using lightkurve."""
    try:
        import lightkurve as lk
        print(f"  Searching for {star_name}...")
        search = lk.search_lightcurve(star_name, mission="Kepler", author="Kepler")
        if len(search) == 0:
            print(f"  No results found for {star_name}")
            return None, None, None
        lc = search[0].download()
        lc = lc.remove_nans().normalize()
        time     = np.array(lc.time.value, dtype=float)
        flux     = np.array(lc.flux.value, dtype=float)
        flux_err = (np.array(lc.flux_err.value, dtype=float)
                    if lc.flux_err is not None
                    else np.ones_like(flux) * np.nanstd(flux) * 0.1)
        print(f"  Downloaded: {len(time)} data points, "
              f"baseline {time[-1]-time[0]:.1f} days")
        return time, flux, flux_err
    except ImportError:
        print("  lightkurve not installed. Run: pip install lightkurve")
        return None, None, None
    except Exception as e:
        print(f"  Download failed: {e}")
        return None, None, None


def plot_kepler_result(time, flux_raw, flux_clean, phase, flux_folded,
                        detection, classification, target, save_path):
    if not HAS_MPL:
        return
    _dark_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle(f"Real Kepler Data — {target['name']}\n{target['description']}",
                 color=COLORS["text"], fontsize=13, fontweight="bold", y=0.99)

    pred    = classification.get("predicted_class", "?")
    planet_p = classification.get("probabilities", {}).get("Planet Transit", 0)
    col = COLORS["planet"] if "Planet" in pred else COLORS["binary"]

    # Raw + denoised
    ax = axes[0]
    ax.plot(time, flux_raw,   color=COLORS["raw"],   lw=0.4, alpha=0.6, label="Raw Kepler flux")
    ax.plot(time, flux_clean, color=col,              lw=0.8, alpha=0.9, label="Denoised")
    ax.set_ylabel("Normalized Flux", fontsize=10)
    ax.set_title("Full Light Curve", fontsize=10, color=COLORS["subtext"])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # Phase-folded
    ax = axes[1]
    if len(phase) > 10:
        ax.scatter(phase, flux_folded, s=3, alpha=0.3, color=COLORS["raw"])
        n_bins = 60
        bins   = np.linspace(-0.5, 0.5, n_bins+1)
        bc     = 0.5*(bins[:-1]+bins[1:])
        bm     = np.array([
            np.mean(flux_folded[(phase>=bins[i])&(phase<bins[i+1])])
            if np.sum((phase>=bins[i])&(phase<bins[i+1]))>0 else np.nan
            for i in range(n_bins)])
        ax.plot(bc, bm, color=col, lw=2.5, label="Binned")
        ax.axhline(1.0, color=COLORS["subtext"], lw=1, ls="--", alpha=0.5)
    ax.set_ylabel("Normalized Flux", fontsize=10)
    ax.set_title(f"Phase-Folded at P={detection.get('period_days','?')} days",
                 fontsize=10, color=COLORS["subtext"])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5); ax.tick_params(labelsize=8)

    # Classification bar
    ax = axes[2]
    classes = ["Planet Transit", "Eclipsing Binary", "Starspot", "Instrumental"]
    short   = ["Planet", "Binary", "Spot", "Instr."]
    probs   = classification.get("probabilities", {})
    vals    = [probs.get(c, 0) for c in classes]
    bar_colors = [COLORS["planet"], COLORS["binary"],
                  COLORS["starspot"], COLORS["instrument"]]
    bars = ax.barh(short, vals, color=bar_colors, alpha=0.85,
                   edgecolor=COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color=COLORS["subtext"], lw=1, ls="--", alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(min(v+0.02, 0.92), bar.get_y()+bar.get_height()/2,
                f"{v:.1%}", va="center", fontsize=10, color=COLORS["text"])
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title(f"Classification: {pred}  (P(Planet)={planet_p:.1%})",
                 fontsize=10, color=col, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3); ax.tick_params(labelsize=9)

    # Metrics annotation
    tp  = detection.get("period_days")
    td  = (detection.get("depth") or 0)*100
    snr = detection.get("bls_snr", 0)
    true_p = target["true_period"]
    period_err = abs(tp-true_p)/true_p*100 if tp else None
    info = (f"Detected period: {tp} days  (true: {true_p})\n"
            f"Period error: {period_err:.2f}%\n"
            f"Detected depth: {td:.4f}%  (true: {target['true_depth']*100:.4f}%)\n"
            f"BLS SNR: {snr:.1f}")
    fig.text(0.5, 0.01, info, ha="center", fontsize=9,
             color=COLORS["subtext"])

    plt.tight_layout(rect=[0,0.05,1,0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Plot saved: {save_path}")


def main():
    print("\n" + "="*60)
    print("  REAL KEPLER DATA TEST")
    print("="*60)

    # Train pipeline
    print("\n[1/3] Training pipeline...")
    X_train, y_train, training_fluxes = build_training_data(n_each=100)
    pipeline = ExoplanetPipeline(verbose=False)
    pipeline.fit_domain_assessor(training_fluxes[:200])
    pipeline.classifier.fit(X_train, y_train)
    print("      Done.")

    os.makedirs("plots", exist_ok=True)
    results = []

    print("\n[2/3] Downloading and testing Kepler targets...")
    for target in KEPLER_TARGETS:
        print(f"\n  Target: {target['name']} — {target['description']}")
        time, flux, flux_err = download_kepler(target["star"])

        if time is None:
            print(f"  Skipping {target['name']} — download failed")
            continue

        print(f"  Running pipeline...")
        report = pipeline.run(
            time=time, flux=flux, flux_err=flux_err,
            ra=target["ra"], dec=target["dec"],
            candidate_id=target["name"],
            use_gaia=False,
            verbose=False,
        )

        det = report.get("transit_parameters", {})
        clf = report.get("classification", {})
        conf = report.get("confidence", {}).get("score", 0)
        pred = clf.get("predicted_class", "?")
        planet_p = clf.get("probabilities", {}).get("Planet Transit", 0)
        T = report.get("confidence", {}).get("domain_trust", "?")

        # Period accuracy
        rec_p = det.get("period_days")
        true_p = target["true_period"]
        if rec_p:
            period_err = abs(rec_p - true_p) / true_p * 100
            period_str = f"{rec_p:.4f}d (err={period_err:.2f}%)"
        else:
            period_err = None
            period_str = "Not detected"

        print(f"  Result:")
        print(f"    Decision:       {report.get('final_decision','?')}")
        print(f"    Predicted:      {pred} (P(planet)={planet_p:.1%})")
        print(f"    Confidence:     {conf:.1%}  |  T={T}")
        print(f"    Period:         {period_str}")
        print(f"    True label:     {target['true_label']}")
        correct = "✓ CORRECT" if "Planet" in pred else "✗ INCORRECT"
        print(f"    Classification: {correct}")

        results.append({
            "target": target,
            "report": report,
            "period_err": period_err,
            "correct": "Planet" in pred,
        })

    # Summary
    print("\n[3/3] Summary")
    print("="*60)
    print(f"  {'Target':<15} {'Predicted':<18} {'Conf':>6}  {'Period Err':>10}  {'Correct?'}")
    print(f"  {'-'*15} {'-'*18} {'-'*6}  {'-'*10}  {'-'*8}")
    for r in results:
        pred = r["report"].get("classification",{}).get("predicted_class","?")
        conf = r["report"].get("confidence",{}).get("score",0)
        perr = f"{r['period_err']:.2f}%" if r["period_err"] else "N/A"
        ok   = "✓" if r["correct"] else "✗"
        print(f"  {r['target']['name']:<15} {pred:<18} {conf:>5.1%}  {perr:>10}  {ok}")
    print()


if __name__ == "__main__":
    main()


# ── Fallback: Simulated Kepler-7b ─────────────────────────────
def simulate_kepler7b_and_run(pipeline):
    """
    If download fails, simulate Kepler-7b using its exact known parameters.
    Kepler-7b: period=4.8854892d, depth=0.96%, duration=4.6hr
    This is scientifically accurate — same parameters, simulated noise.
    """
    print("\n  [Fallback] Simulating Kepler-7b with real parameters...")
    from demo.generate_demo_data import inject_transit, stellar_background

    # Kepler has 30-min cadence over ~90 days per quarter
    n      = 4320   # 90 days × 48 points/day
    time   = np.linspace(0, 90, n)
    # Kepler photometric noise: ~100 ppm = 0.0001
    flux   = stellar_background(n, variability=0.00008)
    flux  += np.random.normal(0, 0.0001, n)
    # Inject Kepler-7b transit
    flux   = inject_transit(time, flux,
                             period=4.8854892,
                             depth=0.0096,
                             duration=4.6/24,
                             t0=1.2)
    flux_err = np.ones(n) * 0.0001

    target = KEPLER_TARGETS[0]
    report = pipeline.run(
        time=time, flux=flux, flux_err=flux_err,
        ra=target["ra"], dec=target["dec"],
        candidate_id="Kepler-7b (simulated)",
        use_gaia=False,
    )

    det  = report.get("transit_parameters", {})
    clf  = report.get("classification", {})
    conf = report.get("confidence", {}).get("score", 0)
    pred = clf.get("predicted_class", "?")
    pp   = clf.get("probabilities", {}).get("Planet Transit", 0)
    rp   = det.get("period_days")
    true_p = target["true_period"]
    perr = abs(rp - true_p)/true_p*100 if rp else None

    print(f"  Result (simulated Kepler-7b):")
    print(f"    Predicted:   {pred}  (P(planet)={pp:.1%})")
    print(f"    Confidence:  {conf:.1%}")
    print(f"    Period:      {rp} days  (true: {true_p}d)")
    if perr: print(f"    Period err:  {perr:.3f}%")
    print(f"    Decision:    {report.get('final_decision','?')}")
    ok = '✓ CORRECT' if 'Planet' in pred else '✗ INCORRECT'
    print(f"    Classification: {ok}")
    return report
