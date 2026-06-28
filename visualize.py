"""
Visualization Module
Generates all plots for the hackathon presentation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


COLORS = {
    "planet":     "#4FC3F7",
    "binary":     "#EF5350",
    "starspot":   "#66BB6A",
    "instrument": "#FFA726",
    "raw":        "#90A4AE",
    "clean":      "#4FC3F7",
    "accent":     "#7E57C2",
    "bg":         "#0D1117",
    "grid":       "#21262D",
    "text":       "#E6EDF3",
    "subtext":    "#8B949E",
}

def _dark_style():
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    "#161B22",
        "axes.edgecolor":    COLORS["grid"],
        "axes.labelcolor":   COLORS["text"],
        "axes.titlecolor":   COLORS["text"],
        "xtick.color":       COLORS["subtext"],
        "ytick.color":       COLORS["subtext"],
        "grid.color":        COLORS["grid"],
        "grid.linewidth":    0.6,
        "text.color":        COLORS["text"],
        "font.family":       "monospace",
        "legend.facecolor":  "#161B22",
        "legend.edgecolor":  COLORS["grid"],
        "legend.labelcolor": COLORS["text"],
    })


# ── Plot 1: Raw vs Denoised Light Curve ───────────────────────
def plot_denoising(time, flux_raw, flux_clean, detection, candidate_id, save_path):
    _dark_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle(f"Stage 2 — Transit-Aware Denoising  |  {candidate_id}",
                 color=COLORS["text"], fontsize=13, fontweight="bold", y=0.98)

    period = detection.get("period_days") or 10
    t0     = detection.get("t0") or 0
    depth  = detection.get("depth") or 0

    # Mark transit windows
    def mark_transits(ax):
        t_start = t0
        while t_start < time[-1]:
            ax.axvspan(t_start - 0.1, t_start + 0.1, alpha=0.15,
                       color=COLORS["planet"], zorder=0)
            t_start += period

    # Raw
    ax = axes[0]
    ax.plot(time, flux_raw, color=COLORS["raw"], lw=0.6, alpha=0.8, label="Raw flux")
    mark_transits(ax)
    ax.set_ylabel("Normalized Flux", fontsize=10)
    ax.set_title("Raw Light Curve (with noise + systematics)", fontsize=10,
                 color=COLORS["subtext"])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.4)
    std_raw = np.nanstd(flux_raw)
    ax.text(0.01, 0.05, f"σ = {std_raw:.5f}", transform=ax.transAxes,
            color=COLORS["subtext"], fontsize=9)

    # Denoised
    ax = axes[1]
    ax.plot(time, flux_clean, color=COLORS["clean"], lw=0.8, label="Denoised flux")
    mark_transits(ax)
    ax.set_ylabel("Normalized Flux", fontsize=10)
    ax.set_xlabel("Time (days)", fontsize=10)
    ax.set_title("After Transit-Aware Denoising (dips preserved)", fontsize=10,
                 color=COLORS["subtext"])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.4)
    std_clean = np.nanstd(flux_clean)
    ax.text(0.01, 0.05, f"σ = {std_clean:.5f}", transform=ax.transAxes,
            color=COLORS["clean"], fontsize=9)

    snr_imp = std_raw / (std_clean + 1e-10)
    fig.text(0.99, 0.01, f"SNR improvement: {snr_imp:.2f}×",
             ha="right", color=COLORS["accent"], fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 2: Phase-Folded Transit ───────────────────────────────
def plot_phase_folded(phase, flux_folded, detection, classification, save_path):
    _dark_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    period   = detection.get("period_days", "?")
    depth    = (detection.get("depth") or 0) * 100
    duration = (detection.get("duration_days") or 0) * 24
    bls_snr  = detection.get("bls_snr", 0)
    pred     = classification.get("predicted_class", "Unknown")
    planet_p = classification.get("probabilities", {}).get("Planet Transit", 0)
    col      = COLORS["planet"] if "Planet" in pred else COLORS["binary"]

    # Scatter
    ax.scatter(phase, flux_folded, s=4, alpha=0.5, color=COLORS["raw"], zorder=2)

    # Binned curve
    n_bins = 60
    bins   = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_c  = 0.5 * (bins[:-1] + bins[1:])
    bin_m  = np.array([
        np.mean(flux_folded[(phase >= bins[i]) & (phase < bins[i+1])])
        if np.sum((phase >= bins[i]) & (phase < bins[i+1])) > 0 else np.nan
        for i in range(n_bins)
    ])
    ax.plot(bin_c, bin_m, color=col, lw=2.5, zorder=3, label="Binned phase curve")

    # Transit region shading
    half_dur = (detection.get("duration_days") or 0.02) / (period if isinstance(period, float) else 10) / 2
    ax.axvspan(-half_dur, half_dur, alpha=0.12, color=col, label="Transit window")
    ax.axhline(1.0, color=COLORS["subtext"], lw=1, ls="--", alpha=0.5)

    ax.set_xlabel("Phase (orbital period fraction)", fontsize=11)
    ax.set_ylabel("Normalized Flux", fontsize=11)
    ax.set_title(f"Phase-Folded Light Curve  |  {pred}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(-0.5, 0.5)

    info = (f"Period: {period} days\n"
            f"Depth:  {depth:.3f}%\n"
            f"Duration: {duration:.2f} hr\n"
            f"BLS SNR: {bls_snr:.1f}\n"
            f"P(Planet): {planet_p:.1%}")
    ax.text(0.02, 0.05, info, transform=ax.transAxes,
            color=COLORS["text"], fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161B22",
                      edgecolor=COLORS["grid"], alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 3: Uncertainty Budget ─────────────────────────────────
def plot_uncertainty_budget(uncertainty_report, domain_report, candidate_id, save_path):
    _dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle(f"Stage 6 — Uncertainty Budget  |  {candidate_id}",
                 color=COLORS["text"], fontsize=13, fontweight="bold")

    eff = uncertainty_report.get("stage_uncertainties", {}).get("effective", {})
    raw = uncertainty_report.get("stage_uncertainties", {}).get("raw", {})
    IF  = uncertainty_report.get("inflation_factors", {})

    stage_labels  = ["Denoise", "Blend", "BLS", "Classifier"]
    raw_vals  = [raw.get("sigma_denoise", 0), raw.get("sigma_blend", 0),
                 raw.get("sigma_BLS", 0),     raw.get("sigma_classifier", 0)]
    eff_vals  = [eff.get("denoise", 0), eff.get("blend", 0),
                 eff.get("BLS", 0),     eff.get("classifier", 0)]
    stage_colors = [COLORS["clean"], COLORS["planet"],
                    COLORS["accent"], COLORS["starspot"]]

    # Left: raw vs effective
    ax = axes[0]
    x  = np.arange(len(stage_labels))
    w  = 0.35
    b1 = ax.bar(x - w/2, raw_vals,  w, label="Raw σ",       color=COLORS["raw"], alpha=0.8)
    b2 = ax.bar(x + w/2, eff_vals,  w, label="Effective σ (T-adjusted)",
                color=[COLORS["accent"]] * 4, alpha=0.9)

    for bar, v in zip(b1, raw_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=COLORS["subtext"])
    for bar, v in zip(b2, eff_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=COLORS["text"])

    ax.set_xticks(x); ax.set_xticklabels(stage_labels)
    ax.set_ylabel("Uncertainty σ"); ax.set_title("Raw vs T-Adjusted Effective σ")
    ax.legend(); ax.grid(True, axis="y", alpha=0.4)
    T = domain_report.get("T", 1.0)
    ax.text(0.98, 0.98, f"Domain Trust T = {T:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            color=COLORS["accent"], fontsize=10, fontweight="bold")

    # Right: pie chart of uncertainty contribution
    ax = axes[1]
    weights = {"denoise": 0.20, "blend": 0.35, "BLS": 0.25, "classifier": 0.20}
    contributions = [weights[k] * eff_vals[i]**2 for i, k in enumerate(weights)]
    total = sum(contributions) + 1e-10
    contributions_pct = [c / total for c in contributions]

    wedges, texts, autotexts = ax.pie(
        contributions_pct,
        labels=stage_labels,
        colors=stage_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor=COLORS["bg"], linewidth=2)
    )
    for t in texts: t.set_color(COLORS["text"]); t.set_fontsize(10)
    for t in autotexts: t.set_color(COLORS["bg"]); t.set_fontsize(9); t.set_fontweight("bold")

    sigma_total = uncertainty_report.get("sigma_total", 0)
    confidence  = uncertainty_report.get("confidence", 0)
    dominant    = uncertainty_report.get("dominant_uncertainty_source", "?")
    ax.set_title("Uncertainty Contribution (weighted)")
    ax.text(0, -1.35,
            f"σ_total = {sigma_total:.4f}   |   Confidence = {confidence:.1%}   |   Dominant: {dominant}",
            ha="center", fontsize=10, color=COLORS["text"])

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 4: All 4 Cases Summary ────────────────────────────────
def plot_all_cases_summary(cases_data, save_path):
    """
    cases_data: list of dicts with keys:
      label, true_label, time, flux_raw, flux_clean,
      phase, flux_folded, detection, classification,
      confidence, T, decision
    """
    _dark_style()
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Exoplanet AI Detection Pipeline — All Demo Cases",
                 color=COLORS["text"], fontsize=15, fontweight="bold", y=0.98)

    n = len(cases_data)
    gs = gridspec.GridSpec(3, n, figure=fig,
                           hspace=0.55, wspace=0.35,
                           top=0.93, bottom=0.06)

    class_color_map = {
        "Planet Transit":   COLORS["planet"],
        "Eclipsing Binary": COLORS["binary"],
        "Starspot":         COLORS["starspot"],
        "Instrumental":     COLORS["instrument"],
    }

    for col, cd in enumerate(cases_data):
        time     = cd["time"]
        flux_raw = cd["flux_raw"]
        flux_cl  = cd["flux_clean"]
        phase    = cd["phase"]
        ff       = cd["flux_folded"]
        det      = cd["detection"]
        clf      = cd["classification"]
        conf     = cd["confidence"]
        T_val    = cd["T"]
        decision = cd["decision"]
        pred     = clf.get("predicted_class", "?")
        col_c    = class_color_map.get(pred, COLORS["accent"])
        true_lbl = cd["true_label"]
        correct  = "✓" if pred.split()[0] in true_lbl else "✗"

        # Row 0: raw light curve
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(time, flux_raw, color=COLORS["raw"], lw=0.5, alpha=0.7)
        ax0.plot(time, flux_cl,  color=col_c, lw=1.0, alpha=0.9)
        ax0.set_title(f"{cd['label']}", fontsize=9, fontweight="bold", color=COLORS["text"])
        ax0.set_ylabel("Flux", fontsize=8)
        ax0.grid(True, alpha=0.3)
        ax0.tick_params(labelsize=7)
        ax0.text(0.02, 0.08, f"T={T_val:.2f}", transform=ax0.transAxes,
                 color=COLORS["accent"], fontsize=8)
        if col == 0:
            ax0.set_title(f"Raw (grey) + Denoised (colour)\n{cd['label']}",
                          fontsize=8, color=COLORS["text"])

        # Row 1: phase-folded
        ax1 = fig.add_subplot(gs[1, col])
        if len(phase) > 5:
            ax1.scatter(phase, ff, s=2, alpha=0.3, color=COLORS["raw"])
            n_bins = 40
            bins   = np.linspace(-0.5, 0.5, n_bins + 1)
            bc     = 0.5 * (bins[:-1] + bins[1:])
            bm     = np.array([
                np.mean(ff[(phase >= bins[i]) & (phase < bins[i+1])])
                if np.sum((phase >= bins[i]) & (phase < bins[i+1])) > 0 else np.nan
                for i in range(n_bins)
            ])
            ax1.plot(bc, bm, color=col_c, lw=2)
            ax1.axhline(1.0, color=COLORS["subtext"], lw=0.8, ls="--", alpha=0.5)
        ax1.set_ylabel("Flux", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=7)
        period_str = f"{det.get('period_days', '?')} d" if det.get('period_days') else "?"
        depth_str  = f"{(det.get('depth') or 0)*100:.3f}%"
        ax1.text(0.02, 0.08, f"P={period_str}\nD={depth_str}",
                 transform=ax1.transAxes, color=COLORS["text"], fontsize=7)
        if col == 0:
            ax1.set_title("Phase-Folded Transit", fontsize=8, color=COLORS["text"])

        # Row 2: classification bar chart + confidence
        ax2 = fig.add_subplot(gs[2, col])
        probs   = clf.get("probabilities", {})
        classes = ["Planet Transit", "Eclipsing Binary", "Starspot", "Instrumental"]
        short   = ["Planet", "Binary", "Spot", "Instr."]
        vals    = [probs.get(c, 0) for c in classes]
        bar_colors = [class_color_map.get(c, COLORS["accent"]) for c in classes]
        bars = ax2.barh(short, vals, color=bar_colors, alpha=0.85, edgecolor=COLORS["bg"])
        ax2.set_xlim(0, 1)
        ax2.axvline(0.5, color=COLORS["subtext"], lw=0.8, ls="--", alpha=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                     f"{v:.1%}", va="center", fontsize=7, color=COLORS["text"])
        ax2.set_xlabel("Probability", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(True, axis="x", alpha=0.3)

        # Decision label
        dec_short = decision[:35] if len(decision) > 35 else decision
        ax2.set_title(f"{correct} {pred.split()[0]}  |  {conf:.0%} conf",
                      fontsize=8, color=col_c, fontweight="bold")
        ax2.text(0.5, -0.30, f"True: {true_lbl}", transform=ax2.transAxes,
                 ha="center", fontsize=7, color=COLORS["subtext"])

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 5: Pipeline Architecture Diagram ─────────────────────
def plot_pipeline_diagram(save_path):
    _dark_style()
    fig, ax = plt.subplots(figsize=(10, 14))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 10); ax.set_ylim(0, 15)
    ax.axis("off")
    ax.set_title("AI Exoplanet Detection Pipeline — Architecture",
                 fontsize=14, fontweight="bold", color=COLORS["text"], pad=15)

    stages = [
        ("Stage 0",  "Data Quality Assessment",      "Q_data ∈ [0,1]",              "#37474F", 14.0),
        ("Stage 1",  "Domain Assessment",             "Trust T ∈ [0,1], σ_domain",   COLORS["accent"], 12.3),
        ("Stage 2",  "Transit-Aware Denoising",       "μ_clean, σ_denoise",          COLORS["clean"], 10.6),
        ("Stage 3",  "Blending Correction (Gaia)",    "μ_corrected, σ_blend",        COLORS["planet"], 8.9),
        ("Stage 4",  "BLS + CNN Transit Detection",   "period, depth, σ_BLS",        "#AB47BC", 7.2),
        ("Stage 5",  "Ensemble Classifier",           "class probs, σ_classifier",   COLORS["starspot"], 5.5),
        ("Stage 6",  "Trust-Adjusted Aggregation",    "σ_total, confidence",         COLORS["instrument"], 3.8),
        ("Output",   "Explainability + Audit Trail",  "JSON report + ruling_out",    "#EF5350", 2.1),
    ]

    for tag, name, output, color, y in stages:
        # Box
        rect = plt.Rectangle((1.2, y - 0.55), 7.6, 0.95,
                              facecolor=color, alpha=0.18,
                              edgecolor=color, linewidth=1.5,
                              transform=ax.transData)
        ax.add_patch(rect)
        ax.text(2.0, y - 0.03, name, fontsize=11, fontweight="bold",
                color=COLORS["text"], va="center")
        ax.text(1.35, y - 0.03, tag, fontsize=8, color=color,
                va="center", fontweight="bold")
        ax.text(8.85, y - 0.03, f"→ {output}", fontsize=7.5,
                color=COLORS["subtext"], va="center")

        # Arrow down
        if y > 2.1:
            ax.annotate("", xy=(5, y - 0.65), xytext=(5, y - 0.55),
                        arrowprops=dict(arrowstyle="->", color=COLORS["subtext"],
                                        lw=1.2))

    # T modulates all stages annotation
    ax.annotate("", xy=(1.2, 10.4), xytext=(1.2, 12.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["accent"],
                                lw=1.5, connectionstyle="arc3,rad=-0.3"))
    ax.text(0.15, 11.3, "T modulates\nall σ values", fontsize=8,
            color=COLORS["accent"], ha="center", style="italic")

    # Refinement branch
    ax.text(8.0, 3.1, "Refinement\n(once only)", fontsize=8,
            color=COLORS["instrument"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161B22",
                      edgecolor=COLORS["instrument"], alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  Saved: {save_path}")
