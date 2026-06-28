"""
Stage 1: Domain Assessment
Computes Trust Score T in [0,1] using PCA embedding distance.
T governs uncertainty inflation for all downstream stages.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DomainAssessor:
    def __init__(self, n_components=10):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.train_embeddings = None
        self.fitted = False

    def _extract_features(self, flux):
        """Extract statistical features from flux array."""
        flux = np.array(flux, dtype=float)
        flux = flux[~np.isnan(flux)]
        if len(flux) < 10:
            return np.zeros(20)

        features = [
            np.mean(flux),
            np.std(flux),
            np.median(flux),
            np.percentile(flux, 5),
            np.percentile(flux, 95),
            np.percentile(flux, 95) - np.percentile(flux, 5),  # range
            float(np.sum(flux < np.mean(flux) - 2*np.std(flux))),  # dip count
            np.mean(np.abs(np.diff(flux))),                     # roughness
            np.std(np.diff(flux)),                              # roughness std
            float(np.sum(np.abs(np.diff(flux)) > 3*np.std(np.diff(flux)))),
            np.max(flux),
            np.min(flux),
            np.max(flux) - np.min(flux),
            float(np.sum(np.isnan(flux))) / max(len(flux), 1),
            np.percentile(flux, 25),
            np.percentile(flux, 75),
            np.percentile(flux, 75) - np.percentile(flux, 25),  # IQR
            np.mean(flux**2),
            np.std(flux**2),
            float(len(flux)),
        ]
        return np.array(features)

    def fit(self, flux_list):
        """Fit on training light curves."""
        features = np.array([self._extract_features(f) for f in flux_list])
        features = np.nan_to_num(features)
        scaled = self.scaler.fit_transform(features)
        n_components = min(self.pca.n_components, scaled.shape[0], scaled.shape[1])
        self.pca.n_components = n_components
        self.train_embeddings = self.pca.fit_transform(scaled)
        self.fitted = True
        return self

    def compute_trust(self, flux):
        """
        Compute domain trust score T for a new light curve.
        T=1 means in-distribution, T=0 means far out-of-distribution.
        """
        if not self.fitted:
            # Not fitted yet: return neutral trust
            return 0.75, {"T": 0.75, "sigma_domain": 0.25,
                          "embedding_distance": None,
                          "trust_level": "UNKNOWN (assessor not fitted)"}

        features = self._extract_features(flux).reshape(1, -1)
        features = np.nan_to_num(features)
        scaled = self.scaler.transform(features)
        embedding = self.pca.transform(scaled)

        # Mahalanobis-like distance: distance from training centroid
        train_mean = np.mean(self.train_embeddings, axis=0)
        train_std = np.std(self.train_embeddings, axis=0) + 1e-10
        distance = float(np.sqrt(np.sum(((embedding[0] - train_mean) / train_std) ** 2)))

        # Normalize distance to [0,1] sigma_domain
        # distance < 2 = in-distribution, > 6 = strongly OOD
        sigma_domain = float(np.clip((distance - 2) / 4, 0, 1))
        T = float(1.0 - sigma_domain)

        report = {
            "T": round(T, 4),
            "sigma_domain": round(sigma_domain, 4),
            "embedding_distance": round(distance, 4),
            "trust_level": _trust_level(T),
        }
        return T, report


def compute_inflation_factor(sigma_domain, stage_sensitivity="high"):
    """
    Compute domain inflation factor for a given stage.
    stage_sensitivity: 'high' (ML stages), 'medium' (BLS), 'low' (catalog-based)
    """
    alpha = {"high": 1.5, "medium": 0.8, "low": 0.3}[stage_sensitivity]
    return float(np.exp(alpha * sigma_domain))


def adjust_thresholds(T):
    """
    Compute T-adjusted confidence thresholds.
    As T decreases, pipeline becomes more conservative.
    """
    accept_threshold = 0.85 + 0.15 * (1 - T)
    review_threshold = 0.50 + 0.20 * (1 - T)
    return {
        "accept":       round(min(accept_threshold, 0.97), 4),
        "refine_lower": round(review_threshold, 4),
        "human_review": round(review_threshold, 4),
    }


def _trust_level(T):
    if T >= 0.85:
        return "HIGH"
    elif T >= 0.60:
        return "MODERATE"
    elif T >= 0.30:
        return "LOW"
    else:
        return "MINIMAL"
