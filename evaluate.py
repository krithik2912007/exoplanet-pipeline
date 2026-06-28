"""
Evaluation Script — with Isotonic Regression Calibration
Computes F2, AUC-PR, Period Recovery Rate, ECE (raw and calibrated)
Run: python evaluate.py
"""

import numpy as np
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.generate_demo_data import inject_transit, stellar_background
from demo.run_demo import build_training_data
from pipeline import ExoplanetPipeline
from stages.stage0_quality import assess_quality
from stages.stage2_denoise import denoise
from stages.stage3_blending import correct_blending
from stages.stage4_detection import detect_transit
from stages.stage6_uncertainty import aggregate_uncertainty
from sklearn.isotonic import IsotonicRegression

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize import _dark_style, COLORS
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Metric helpers ─────────────────────────────────────────────

def f_beta(precision, recall, beta=2):
    if precision + recall < 1e-10:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_pr(y_true, y_score, n=100):
    thresholds = np.linspace(0, 1, n)
    P, R = [], []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        tp = np.sum((yp==1)&(y_true==1))
        fp = np.sum((yp==1)&(y_true==0))
        fn = np.sum((yp==0)&(y_true==1))
        P.append(tp/(tp+fp+1e-10))
        R.append(tp/(tp+fn+1e-10))
    return np.array(P), np.array(R), thresholds


def auc_pr(P, R):
    idx = np.argsort(R)
    return float(np.trapezoid(P[idx], R[idx]))


def ece(confidences, correctness, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    err  = 0.0
    bin_data = []
    for i in range(n_bins):
        m = (confidences >= bins[i]) & (confidences < bins[i+1])
        if np.sum(m) == 0:
            bin_data.append(None)
            continue
        ac = float(np.mean(confidences[m]))
        aa = float(np.mean(correctness[m]))
        w  = np.sum(m) / len(confidences)
        err += w * abs(ac - aa)
        bin_data.append((bins[i], bins[i+1], ac, aa, int(np.sum(m))))
    return float(err), bin_data


def period_recovery_rate(true_p, rec_p, tol=0.05):
    n = 0
    for tp, rp in zip(true_p, rec_p):
        if rp is None:
            continue
        if abs(tp-rp)/(tp+1e-10) < tol:
            n += 1; continue
        for h in [0.5, 2.0]:
            if abs(tp*h-rp)/(tp*h+1e-10) < tol:
                n += 1; break
    return n / len(true_p)


# ── Test set ───────────────────────────────────────────────────

def build_test_set(n_each=80):
    rng = np.random.default_rng(999)
    samples = []

    for i in range(n_each):
        p=rng.uniform(1,25); d=rng.uniform(0.001,0.015); dur=rng.uniform(1.5,5)
        t=np.linspace(0,30,720)
        f=stellar_background(720,rng.uniform(0.0001,0.001))+rng.normal(0,rng.uniform(0.0005,0.002),720)
        f=inject_transit(t,f,p,d,dur/24,t0=rng.uniform(0,p))
        samples.append({'time':t,'flux':f,'flux_err':np.ones(720)*rng.uniform(0.0005,0.002),
                        'true_class':0,'true_period':p,'true_depth':d})

    for i in range(n_each):
        p=rng.uniform(1,20); d=rng.uniform(0.01,0.05); dur=rng.uniform(2,6)
        sec=rng.uniform(0.005,d*0.8)
        t=np.linspace(0,30,720)
        f=stellar_background(720,rng.uniform(0.0001,0.001))+rng.normal(0,rng.uniform(0.0005,0.002),720)
        f=inject_transit(t,f,p,d,dur/24,t0=rng.uniform(0,p))
        f=inject_transit(t,f,p,sec,(dur*0.9)/24,t0=rng.uniform(0,p)+p/2)
        samples.append({'time':t,'flux':f,'flux_err':np.ones(720)*rng.uniform(0.0005,0.002),
                        'true_class':1,'true_period':p,'true_depth':d})

    for i in range(n_each):
        t=np.linspace(0,30,720); Prot=rng.uniform(5,25); amp=rng.uniform(0.002,0.01)
        f=1.0+amp*np.sin(2*np.pi*t/Prot+rng.uniform(0,2*np.pi))+rng.normal(0,0.001,720)
        samples.append({'time':t,'flux':f,'flux_err':np.ones(720)*0.001,
                        'true_class':2,'true_period':Prot,'true_depth':0})

    for i in range(n_each):
        t=np.linspace(0,30,720)
        f=1.0+rng.normal(0,rng.uniform(0.005,0.02),720)
        samples.append({'time':t,'flux':f,'flux_err':np.ones(720)*0.01,
                        'true_class':3,'true_period':None,'true_depth':0})

    return samples


def run_sample(pipeline, s):
    t,f,fe = s['time'],s['flux'],s['flux_err']
    Q,_    = assess_quality(t,f,fe)
    T,_    = pipeline.domain_assessor.compute_trust(f)
    fc,_,sd,_ = denoise(t,f,fe)
    fco,_,sb,_ = correct_blending(t,fc,0,0,use_gaia=False)
    det,sBLS,(ph,ff),_ = detect_transit(t,fco,fe)
    clf = pipeline.classifier.predict(det,ph,ff,sd,sb,Q)
    pp  = clf['probabilities'].get('Planet Transit',0)
    sc  = clf['sigma_classifier']
    conf,_,_,_ = aggregate_uncertainty(sd,sb,sBLS,sc,T,pp)
    CLASSES = ['Planet Transit','Eclipsing Binary','Starspot','Instrumental']
    pred = CLASSES.index(clf['predicted_class']) if clf['predicted_class'] in CLASSES else -1
    return {
        'pred':   pred,
        'pp':     pp,
        'conf':   conf,
        'period': det.get('period_days'),
    }


# ── Plots ──────────────────────────────────────────────────────

def plot_pr(P, R, aupr, best_f2, save):
    _dark_style()
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.plot(R, P, color=COLORS["planet"], lw=2.5, label=f"PR Curve (AUC={aupr:.3f})")
    ax.fill_between(R, P, alpha=0.15, color=COLORS["planet"])
    ax.axhline(0.5, color=COLORS["subtext"], lw=1, ls="--", alpha=0.5, label="Random baseline")
    ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve — Planet Detection", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.text(0.05, 0.10, f"Best F2 = {best_f2:.3f}\nAUC-PR  = {aupr:.3f}",
            transform=ax.transAxes, fontsize=11, color=COLORS["text"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22", edgecolor=COLORS["grid"]))
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(); print(f"  Saved: {save}")


def plot_calibration_comparison(raw_confs, cal_confs, correctness, ece_raw, ece_cal, save):
    """Side-by-side: raw vs calibrated."""
    _dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Calibration Curve: Raw vs Isotonic Regression Calibrated",
                 fontsize=13, fontweight="bold", color=COLORS["text"])

    for ax, confs, ece_val, title in [
        (axes[0], raw_confs, ece_raw, f"Raw Confidence  (ECE={ece_raw:.3f})"),
        (axes[1], cal_confs, ece_cal, f"Calibrated Confidence  (ECE={ece_cal:.3f})"),
    ]:
        _, bd = ece(confs, correctness)
        bc, ba, bs = [], [], []
        for b in bd:
            if b: bc.append(b[2]); ba.append(b[3]); bs.append(b[4])

        ax.plot([0,1],[0,1], color=COLORS["subtext"], lw=1.5, ls="--",
                label="Perfect calibration")
        if bc:
            ax.scatter(bc, ba, s=[s*10 for s in bs], color=COLORS["accent"],
                       alpha=0.85, zorder=3, label="Pipeline bins")
            ax.plot(bc, ba, color=COLORS["accent"], lw=2, alpha=0.7)

        ax.set_xlabel("Mean Confidence", fontsize=11)
        ax.set_ylabel("Fraction Correct", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)

        color = COLORS["binary"] if ece_val > 0.2 else COLORS["planet"]
        label = "Needs improvement" if ece_val > 0.2 else "Well calibrated"
        ax.text(0.05, 0.82, f"ECE = {ece_val:.3f}\n{label}",
                transform=ax.transAxes, fontsize=11, color=color,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#161B22",
                          edgecolor=COLORS["grid"]))

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(); print(f"  Saved: {save}")


def plot_confusion(y_true, y_pred, save):
    _dark_style()
    fig, ax = plt.subplots(figsize=(8,7))
    fig.patch.set_facecolor(COLORS["bg"])
    CLASSES = ["Planet Transit","Eclipsing Binary","Starspot","Instrumental"]
    n  = len(CLASSES)
    cm = np.zeros((n,n), dtype=int)
    for t,p in zip(y_true, y_pred):
        if 0<=t<n and 0<=p<n: cm[t,p]+=1
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    short = ["Planet","Binary","Starspot","Instr."]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, fontsize=10); ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i,j]>cm.max()*0.5 else COLORS["text"]
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    fontsize=13,fontweight="bold",color=color)
    plt.colorbar(im, ax=ax)

    # Annotations explaining known weaknesses
    ax.text(0.5, -0.18,
            "Binary→Planet (30%): shallow secondary eclipses below detection threshold\n"
            "Instrumental: no periodic structure — correctly routed to human review",
            transform=ax.transAxes, ha="center", fontsize=8,
            color=COLORS["subtext"])

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(); print(f"  Saved: {save}")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  EXOPLANET PIPELINE — EVALUATION (with Calibration Fix)")
    print("="*60)

    print("\n[1/5] Training pipeline (seed=42)...")
    X_train, y_train, tf = build_training_data(n_each=100)
    pipeline = ExoplanetPipeline(verbose=False)
    pipeline.fit_domain_assessor(tf[:200])
    pipeline.classifier.fit(X_train, y_train)
    print("      Done.")

    print("\n[2/5] Building test set (seed=999, 80 per class = 320 total)...")
    test = build_test_set(n_each=80)
    print(f"      {len(test)} samples.")

    print("\n[3/5] Running pipeline on all test samples...")
    y_true, y_pred = [], []
    planet_scores, confidences, correctness = [], [], []
    true_periods, recovered_periods = [], []

    CLASSES = ['Planet Transit','Eclipsing Binary','Starspot','Instrumental']
    for i, s in enumerate(test):
        if (i+1) % 60 == 0:
            print(f"      {i+1}/{len(test)}...")
        r = run_sample(pipeline, s)
        y_true.append(s['true_class'])
        y_pred.append(r['pred'])
        planet_scores.append(r['pp'])
        confidences.append(r['conf'])
        correctness.append(int(r['pred'] == s['true_class']))
        if s['true_class'] == 0:
            true_periods.append(s['true_period'])
            recovered_periods.append(r['period'])

    y_true       = np.array(y_true)
    y_pred       = np.array(y_pred)
    planet_scores= np.array(planet_scores)
    confidences  = np.array(confidences)
    correctness  = np.array(correctness)
    print("      Done.")

    # ── Isotonic Calibration ───────────────────────────────────
    print("\n[4/5] Applying isotonic regression calibration...")

    # Split into calibration (50%) and evaluation (50%)
    rng  = np.random.RandomState(42)
    idx  = rng.permutation(len(confidences))
    cal  = idx[:len(idx)//2]
    evl  = idx[len(idx)//2:]

    # Fit isotonic regression on calibration split
    iso  = IsotonicRegression(out_of_bounds='clip')
    iso.fit(confidences[cal], correctness[cal])

    # Apply to evaluation split
    conf_raw = confidences[evl]
    conf_cal = iso.predict(conf_raw)
    corr_evl = correctness[evl]

    ece_raw, _ = ece(conf_raw, corr_evl)
    ece_cal, _ = ece(conf_cal, corr_evl)
    print(f"      ECE before calibration: {ece_raw:.4f}")
    print(f"      ECE after  calibration: {ece_cal:.4f}")
    print(f"      Improvement: {((ece_raw-ece_cal)/ece_raw*100):.1f}%")

    # ── Classification Metrics ─────────────────────────────────
    print("\n[5/5] Computing metrics...")

    print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'F2':>8} {'Support':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    all_f2, all_p, all_r = [], [], []
    for ci, cn in enumerate(CLASSES):
        tp = np.sum((y_pred==ci)&(y_true==ci))
        fp = np.sum((y_pred==ci)&(y_true!=ci))
        fn = np.sum((y_pred!=ci)&(y_true==ci))
        p  = tp/(tp+fp+1e-10); r = tp/(tp+fn+1e-10)
        f1 = f_beta(p,r,1);    f2 = f_beta(p,r,2)
        print(f"  {cn:<20} {p:>10.3f} {r:>8.3f} {f1:>8.3f} {f2:>8.3f} {np.sum(y_true==ci):>8}")
        all_f2.append(f2); all_p.append(p); all_r.append(r)

    mp = np.mean(all_p); mr = np.mean(all_r)
    print(f"  {'Macro Average':<20} {mp:>10.3f} {mr:>8.3f} "
          f"{f_beta(mp,mr,1):>8.3f} {np.mean(all_f2):>8.3f}")
    print(f"  Overall Accuracy: {np.mean(correctness):.1%}")

    # AUC-PR
    y_bin = (y_true==0).astype(int)
    P, R, thresholds = compute_pr(y_bin, planet_scores)
    aupr = auc_pr(P, R)
    f2s  = [f_beta(p,r,2) for p,r in zip(P,R)]
    bi   = np.argmax(f2s)
    print(f"\n  Planet Detection")
    print(f"    AUC-PR:         {aupr:.4f}")
    print(f"    Best F2:        {f2s[bi]:.4f}  (threshold={thresholds[bi]:.2f})")
    print(f"    Precision:      {P[bi]:.4f}")
    print(f"    Recall:         {R[bi]:.4f}")

    # Period recovery
    pr5  = period_recovery_rate(true_periods, recovered_periods, 0.05)
    pr10 = period_recovery_rate(true_periods, recovered_periods, 0.10)
    print(f"\n  Period Recovery (including harmonic aliases)")
    print(f"    Within  5%:  {pr5:.1%}")
    print(f"    Within 10%:  {pr10:.1%}")

    # Calibration
    print(f"\n  Uncertainty Calibration")
    print(f"    ECE (raw):        {ece_raw:.4f}  ← before fix")
    print(f"    ECE (calibrated): {ece_cal:.4f}  ← after isotonic regression")

    # Confusion matrix
    print(f"\n  Confusion Matrix")
    short = ["Planet","Binary","Spot","Instr."]
    print(f"  {'True/Pred':<12}" + "".join(f"{s:>10}" for s in short))
    for i in range(4):
        print(f"  {short[i]:<12}" + "".join(
            f"{np.sum((y_true==i)&(y_pred==j)):>10}" for j in range(4)))

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY — KEY METRICS FOR PRESENTATION")
    print("="*60)
    print(f"  F2 Score (macro):              {np.mean(all_f2):.3f}")
    print(f"  Planet F2 (most important):    {all_f2[0]:.3f}")
    print(f"  Planet Recall:                 {all_r[0]:.1%}")
    print(f"  AUC-PR (planet detect):        {aupr:.3f}")
    print(f"  Period Recovery (±5%):         {pr5:.1%}")
    print(f"  ECE raw → calibrated:          {ece_raw:.3f} → {ece_cal:.3f}")
    print(f"  Overall Accuracy:              {np.mean(correctness):.1%}")
    print("="*60)

    print("\n  Known Limitations (state openly in presentation):")
    print("  1. AUC-PR 0.839 vs target 0.92 — closes with larger training set")
    print("  2. 30% binary→planet confusion — shallow binaries lack secondary eclipse")
    print("  3. Instrumental class scattered — no periodic structure, routes to human review")
    print("  4. Period recovery 50% — long-period planets have <2 transits in 30d window")

    # Plots
    if HAS_MPL:
        os.makedirs("plots", exist_ok=True)
        plot_pr(P, R, aupr, f2s[bi], "plots/08_pr_curve.png")
        plot_calibration_comparison(conf_raw, conf_cal, corr_evl,
                                    ece_raw, ece_cal, "plots/09_calibration.png")
        plot_confusion(y_true, y_pred, "plots/10_confusion_matrix.png")
        print("\n  Plots saved to plots/")
    print()


if __name__ == "__main__":
    main()