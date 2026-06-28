"""
Data Loader Utility
Downloads and prepares light curves from Kepler/TESS missions.
"""

import numpy as np
import lightkurve as lk
import warnings
warnings.filterwarnings('ignore')


# Known test cases with labels
KNOWN_SOURCES = {
    # Confirmed planets
    "Kepler-22b":    {"target": "Kepler-22",    "mission": "Kepler", "label": "planet"},
    "Kepler-7b":     {"target": "Kepler-7",     "mission": "Kepler", "label": "planet"},
    "Kepler-10b":    {"target": "Kepler-10",    "mission": "Kepler", "label": "planet"},

    # Eclipsing binaries
    "KIC-4851217":   {"target": "KIC 4851217",  "mission": "Kepler", "label": "eclipsing_binary"},
    "KIC-9602595":   {"target": "KIC 9602595",  "mission": "Kepler", "label": "eclipsing_binary"},

    # Quiet stars (no transit)
    "KIC-3733346":   {"target": "KIC 3733346",  "mission": "Kepler", "label": "negative"},
    "KIC-6106415":   {"target": "KIC 6106415",  "mission": "Kepler", "label": "negative"},
}


def download_light_curve(target: str, mission: str = "Kepler",
                          quarter: int = None, sector: int = None) -> dict:
    """
    Download and normalize a light curve.

    Returns dict with:
        time:   np.array of timestamps (days)
        flux:   np.array of normalized flux
        flux_err: np.array of flux uncertainties
        meta:   dict of metadata
    """
    try:
        search = lk.search_lightcurve(target, mission=mission)
        if len(search) == 0:
            raise ValueError(f"No data found for {target}")

        # Download first available
        lc = search[0].download()
        lc = lc.remove_nans().normalize()

        time     = np.array(lc.time.value, dtype=np.float64)
        flux     = np.array(lc.flux.value, dtype=np.float64)
        flux_err = np.array(lc.flux_err.value, dtype=np.float64)

        # Replace any remaining NaNs
        mask     = np.isfinite(flux) & np.isfinite(flux_err)
        time     = time[mask]
        flux     = flux[mask]
        flux_err = flux_err[mask]

        return {
            "time":      time,
            "flux":      flux,
            "flux_err":  flux_err,
            "meta": {
                "target":    target,
                "mission":   mission,
                "n_points":  len(time),
                "baseline":  float(time[-1] - time[0]),
                "cadence":   float(np.median(np.diff(time))),
                "mean_flux": float(np.mean(flux)),
            }
        }

    except Exception as e:
        raise RuntimeError(f"Failed to download {target}: {e}")


def inject_transit(lc_dict: dict, period: float, depth: float,
                   duration: float, t0: float = None) -> dict:
    """
    Inject a synthetic box transit into a light curve.

    Args:
        period:   orbital period in days
        depth:    transit depth as fraction (e.g. 0.008 for 0.8%)
        duration: transit duration in days
        t0:       time of first transit center (default: random)

    Returns modified lc_dict with injected transit and injection params.
    """
    time = lc_dict["time"]
    flux = lc_dict["flux"].copy()

    if t0 is None:
        t0 = time[0] + np.random.uniform(0, period)

    # Compute phase
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    half_dur_phase = (duration / 2.0) / period
    in_transit = np.abs(phase) < half_dur_phase
    flux[in_transit] -= depth

    result = lc_dict.copy()
    result["flux"] = flux
    result["injected_transit"] = {
        "period":   period,
        "depth":    depth,
        "duration": duration,
        "t0":       t0,
        "label":    "planet"
    }
    return result


def simulate_astrosat_noise(lc_dict: dict,
                             noise_scale: float = 2.5,
                             gap_probability: float = 0.05) -> dict:
    """
    Simulate AstroSat-like noise characteristics on a Kepler light curve.
    Based on published AstroSat SXT/UVIT photometric precision specs.

    Args:
        noise_scale:     multiplicative factor on existing noise (~2.5x Kepler)
        gap_probability: fraction of points to randomly remove (data gaps)
    """
    time     = lc_dict["time"].copy()
    flux     = lc_dict["flux"].copy()
    flux_err = lc_dict["flux_err"].copy()

    # Add extra noise
    extra_noise = np.random.normal(0, noise_scale * np.std(flux), size=len(flux))
    flux += extra_noise
    flux_err = np.sqrt(flux_err**2 + (noise_scale * np.std(flux_err))**2)

    # Simulate data gaps
    keep = np.random.random(len(time)) > gap_probability
    time     = time[keep]
    flux     = flux[keep]
    flux_err = flux_err[keep]

    result = lc_dict.copy()
    result["time"]     = time
    result["flux"]     = flux
    result["flux_err"] = flux_err
    result["meta"]["simulated_domain"] = "AstroSat"
    result["meta"]["n_points"]         = len(time)

    return result


def load_synthetic_dataset(n_planet: int = 20, n_negative: int = 20,
                            n_binary: int = 10) -> list:
    """
    Build a small synthetic labeled dataset using transit injection.
    Used for testing the pipeline without downloading many files.
    """
    dataset = []

    # Generate synthetic light curves with Gaussian noise
    def make_base_lc(n_days=30, cadence_min=30, noise_level=0.001):
        n_points = int(n_days * 24 * 60 / cadence_min)
        time = np.linspace(0, n_days, n_points)
        flux = 1.0 + np.random.normal(0, noise_level, n_points)
        flux_err = np.full(n_points, noise_level)
        return {
            "time": time, "flux": flux, "flux_err": flux_err,
            "meta": {"target": "synthetic", "mission": "synthetic",
                     "n_points": n_points, "baseline": n_days,
                     "cadence": cadence_min / (24 * 60), "mean_flux": 1.0}
        }

    # Planet cases
    for i in range(n_planet):
        lc = make_base_lc()
        period   = np.random.uniform(3, 15)
        depth    = np.random.uniform(0.003, 0.015)
        duration = np.random.uniform(0.05, 0.2)
        lc = inject_transit(lc, period, depth, duration)
        lc["label"] = "planet"
        dataset.append(lc)

    # Negative cases
    for i in range(n_negative):
        lc = make_base_lc()
        lc["label"] = "negative"
        dataset.append(lc)

    # Simulated binary (deeper V-shaped dips)
    for i in range(n_binary):
        lc = make_base_lc()
        period   = np.random.uniform(2, 10)
        depth    = np.random.uniform(0.02, 0.15)
        duration = np.random.uniform(0.05, 0.3)
        lc = inject_transit(lc, period, depth, duration)
        lc["label"] = "eclipsing_binary"
        dataset.append(lc)

    return dataset


if __name__ == "__main__":
    print("Testing synthetic dataset generation...")
    dataset = load_synthetic_dataset(n_planet=5, n_negative=5, n_binary=3)
    print(f"Generated {len(dataset)} light curves")
    print(f"Example keys: {list(dataset[0].keys())}")
    print(f"Example meta: {dataset[0]['meta']}")
    print("Data loader OK")
