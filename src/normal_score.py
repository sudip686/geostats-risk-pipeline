"""
05_normal_score.py - Normal Score Transform

Performs Normal Score Transform (NST) for geostatistical analysis.
Uses declustered weights in CDF construction.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class NormalScoreTransform:
    """Normal Score Transform for Gaussian anamorphosis."""

    def __init__(self):
        self.data = None
        self.sorted_data = None
        self.norm_scores = None
        self.weights = None

    def fit(self, data, weights=None):
        """
        Fit the normal score transform.

        Args:
            data (np.array): Original data values
            weights (np.array): Optional declustering weights

        Returns:
            self
        """
        data = np.array(data)
        self.data = data[~np.isnan(data)]

        if weights is not None:
            weights = np.array(weights)[~np.isnan(data)]
        else:
            weights = np.ones(len(self.data))

        # Sort by data value
        sort_idx = np.argsort(self.data)
        self.sorted_data = self.data[sort_idx]
        sorted_weights = weights[sort_idx]

        # Compute cumulative weights (CDF) with centered probabilities
        cum_weights = np.cumsum(sorted_weights)
        total_w = cum_weights[-1]
        centered = (cum_weights - 0.5 * sorted_weights) / total_w
        centered = np.clip(centered, 1e-6, 1 - 1e-6)

        # Compute normal quantiles
        self.norm_scores = norm.ppf(centered)

        # Store weights for transform
        self.weights = sorted_weights

        logger.info(f"NST fitted: {len(self.data)} samples, range [{self.sorted_data.min():.2f}, {self.sorted_data.max():.2f}]")

        return self

    def transform(self, data):
        """
        Transform data to normal scores.

        Args:
            data (np.array): Original values

        Returns:
            np.array: Normal scores
        """
        if self.sorted_data is None:
            raise ValueError("NST not fitted. Call fit() first.")

        return np.interp(data, self.sorted_data, self.norm_scores)

    def fit_transform(self, data, weights=None):
        """Fit and transform in one step."""
        self.fit(data, weights)
        return self.transform(data)

    def back_transform(self, scores):
        """
        Back-transform normal scores to original units.

        Args:
            scores (np.array): Normal scores

        Returns:
            np.array: Original values
        """
        if self.sorted_data is None:
            raise ValueError("NST not fitted. Call fit() first.")

        # Clip scores to valid range
        scores = np.clip(scores, self.norm_scores.min(), self.norm_scores.max())
        return np.interp(scores, self.norm_scores, self.sorted_data)

    def save(self, path):
        """Save transform parameters to file."""
        import json
        data = {
            'sorted_data': self.sorted_data.tolist(),
            'norm_scores': self.norm_scores.tolist()
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path):
        """Load transform parameters from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.sorted_data = np.array(data['sorted_data'])
        self.norm_scores = np.array(data['norm_scores'])


def run(data_path=None, data_dir='data', grade_field='tgc_pct', output_path=None, config=None):
    """
    Run normal score transform.

    Args:
        data_path: Path to declustered CSV
        data_dir: Input data directory
        grade_field: Grade column name
        output_path: Optional output path

    Returns:
        tuple: (nst object, transformed data)
    """
    if data_path:
        df = pd.read_csv(data_path)
    else:
        from .declustering import run as run_decluster
        df, _ = run_decluster(data_dir=data_dir, grade_field=grade_field)

    # Optional detrending
    trend_cfg = config.get('trend', {}) if config else {}
    trend_columns = trend_cfg.get('columns')
    if trend_cfg.get('enabled') and trend_columns:
        from .trend import fit_linear_trend, apply_linear_trend
        coeffs = fit_linear_trend(df, trend_columns, grade_field)
        df = df.copy()
        df['trend'] = apply_linear_trend(df, trend_columns, coeffs)
        df[grade_field] = df[grade_field] - df['trend']
        trend_cfg['coeffs'] = coeffs.tolist()
    
    # Get values and weights
    values = df[grade_field].values
    weights = df['decluster_weight'].values if 'decluster_weight' in df.columns else None

    # Fit NST
    nst = NormalScoreTransform()
    norm_values = nst.fit_transform(values, weights)

    # Add to dataframe
    result = df.copy()
    result['tgc_ns'] = norm_values
    if trend_cfg.get('enabled') and trend_columns:
        result['trend'] = df['trend']

    # Verify back transformation
    back_transformed = nst.back_transform(nst.transform(values))
    error = np.abs(back_transformed - values).max()
    logger.info(f"Back-transform verification max error: {error:.6f}")

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved NST data to {output_path}")

    return nst, result
