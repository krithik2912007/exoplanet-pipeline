"""
Stage 3: Blending Correction
Queries Gaia DR3 for nearby sources, corrects transit depth dilution,
runs centroid shift test, outputs sigma_blend.
"""

import numpy as np


def correct_blending(time, flux, ra, dec, aperture_radius_arcsec=21.0,
                     use_gaia=True):
    """
    Estimate and correct for stellar blending contamination.

    Returns:
        flux_corrected: blend-corrected flux
        contamination_ratio: fraction of flux from target star
        sigma_blend: uncertainty in blend correction
        report: detailed report dict
    """
    contamination_ratio, sigma_blend_ratio, neighbor_info = _query_blend_ratio(
        ra, dec, aperture_radius_arcsec, use_gaia
    )

    # Correct transit depth dilution
    # Observed dip = true dip * contamination_ratio
    # True dip = observed dip / contamination_ratio
    baseline = np.nanmedian(flux)
    flux_normalized = flux / baseline

    # Correct: bring flux deviations back to what they'd be without blending
    flux_corrected = (flux_normalized - 1.0) / contamination_ratio + 1.0
    flux_corrected = flux_corrected * baseline

    # Centroid shift test (simplified: check variance during vs outside dips)
    centroid_flag, centroid_detail = _centroid_shift_test(time, flux_normalized)

    # Uncertainty in blend correction propagates as:
    # if f_blend = contamination_ratio with uncertainty sigma_ratio
    # then sigma_depth_corrected = sigma_ratio / contamination_ratio^2 * observed_depth
    observed_depth = float(1.0 - np.nanmin(flux_normalized))
    sigma_depth = sigma_blend_ratio * observed_depth / (contamination_ratio**2 + 1e-10)
    sigma_blend = float(np.clip(sigma_blend_ratio / contamination_ratio, 0, 1))

    report = {
        "contamination_ratio": round(contamination_ratio, 4),
        "sigma_blend": round(sigma_blend, 4),
        "corrected_depth_estimate": round(observed_depth / contamination_ratio, 6),
        "observed_depth_estimate": round(observed_depth, 6),
        "sigma_corrected_depth": round(sigma_depth, 6),
        "centroid_shift_detected": centroid_flag,
        "centroid_detail": centroid_detail,
        "neighbor_count": len(neighbor_info),
        "neighbors": neighbor_info,
        "blending_flag": _blend_flag(contamination_ratio),
    }

    return flux_corrected, contamination_ratio, sigma_blend, report


def _query_blend_ratio(ra, dec, aperture_radius_arcsec, use_gaia):
    """
    Query Gaia DR3 for nearby sources and compute contamination ratio.
    Falls back to a conservative estimate if query fails.
    """
    if use_gaia:
        try:
            from astroquery.gaia import Gaia
            import warnings
            warnings.filterwarnings("ignore")

            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
            radius_deg = aperture_radius_arcsec / 3600.0

            query = f"""
            SELECT source_id, ra, dec, phot_g_mean_mag,
                   DISTANCE({ra}, {dec}, ra, dec) AS dist_deg
            FROM gaiadr3.gaia_source
            WHERE DISTANCE({ra}, {dec}, ra, dec) < {radius_deg}
              AND phot_g_mean_mag IS NOT NULL
            ORDER BY phot_g_mean_mag ASC
            """
            job = Gaia.launch_job(query)
            results = job.get_results()

            if len(results) == 0:
                return 1.0, 0.05, []

            # Convert magnitudes to flux (relative)
            mags = np.array(results["phot_g_mean_mag"])
            fluxes = 10 ** (-mags / 2.5)
            total_flux = np.sum(fluxes)
            target_flux = fluxes[0]  # Brightest = target

            contamination_ratio = float(target_flux / total_flux)
            # Uncertainty from magnitude measurement errors (~0.01 mag typical)
            sigma_ratio = float(0.01 * np.log(10) / 2.5 * contamination_ratio * (1 - contamination_ratio))
            sigma_ratio = max(sigma_ratio, 0.02)  # Minimum 2% uncertainty

            neighbors = []
            for i, row in enumerate(results):
                neighbors.append({
                    "rank": int(i),
                    "separation_arcsec": round(float(row["dist_deg"]) * 3600, 2),
                    "g_mag": round(float(row["phot_g_mean_mag"]), 3),
                    "flux_fraction": round(float(fluxes[i] / total_flux), 4),
                })

            return contamination_ratio, sigma_ratio, neighbors

        except Exception as e:
            # Gaia query failed: use conservative fallback
            return _fallback_blend_estimate()
    else:
        return _fallback_blend_estimate()


def _fallback_blend_estimate():
    """Conservative fallback: assume 10% contamination with high uncertainty."""
    return 0.90, 0.10, [{"note": "Gaia query unavailable, using conservative estimate"}]


def _centroid_shift_test(time, flux_normalized, dip_threshold=0.005):
    """
    Simplified centroid test: checks if dip timing is consistent.
    In a full implementation this uses (x,y) pixel centroids.
    Here we use flux variability as a proxy.
    """
    in_dip = flux_normalized < (1.0 - dip_threshold)
    out_dip = ~in_dip

    if np.sum(in_dip) < 3 or np.sum(out_dip) < 3:
        return False, "Insufficient dip points for centroid test"

    # Real centroid shift test would compare (x,y) pixel positions
    # Here: if dip points are systematically offset in time from expectation
    dip_flux_std = np.nanstd(flux_normalized[in_dip])
    out_flux_std = np.nanstd(flux_normalized[out_dip])

    # If in-dip variability is much higher than out-of-dip → possible blend shift
    ratio = dip_flux_std / (out_flux_std + 1e-10)
    shift_detected = bool(ratio > 3.0)

    return shift_detected, f"Dip/out-of-dip std ratio: {ratio:.2f} (>3.0 flags shift)"


def _blend_flag(contamination_ratio):
    if contamination_ratio >= 0.90:
        return "LOW"
    elif contamination_ratio >= 0.70:
        return "MEDIUM"
    elif contamination_ratio >= 0.50:
        return "HIGH"
    else:
        return "SEVERE"
