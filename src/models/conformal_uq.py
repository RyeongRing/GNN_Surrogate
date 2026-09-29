"""
Conformal Prediction 래퍼.
Split Conformal — regime별 q_hat 분리 저장.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    Empirical residual-quantile intervals inspired by split conformal prediction.
    calibration set에서 nonconformity score 계산 후 q_hat 저장.
    regime별 분리 저장으로 alternate-regime coverage 실험 지원.
    """

    def __init__(self, alpha: float = 0.10):
        """
        alpha: 1 - target_coverage (default 0.10 → 90% coverage)
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.q_hat_dict: Dict[str, np.ndarray] = {}  # regime_id -> q_hat [T] for final (graph*node, T) inputs

    def calibrate(self, y_pred: np.ndarray, y_true: np.ndarray,
                  regime_id: str = "B") -> None:
        """
        Calibration set에서 nonconformity score 계산 후 q_hat 저장.
        Final inputs: [calibration_graphs * nodes, T]. Pool axis 0 at each timestep.
        Correlated residuals and linear quantile interpolation yield empirical
        intervals; no formal distribution-free coverage guarantee is asserted.
        """
        if y_pred.shape != y_true.shape or y_pred.ndim < 2 or y_pred.shape[0] == 0:
            raise ValueError("Calibration arrays must have equal, nonempty shapes")
        scores = np.abs(y_true - y_pred)
        if not np.isfinite(scores).all():
            raise ValueError("Calibration residuals must be finite")

        # Finite-sample-adjusted empirical quantile level; NumPy linear interpolation.
        n = scores.shape[0]
        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)

        q_hat = np.quantile(scores, level, axis=0, method="linear")

        self.q_hat_dict[regime_id] = q_hat
        logger.info(f"Calibrated regime={regime_id}: q_hat shape={q_hat.shape}, "
                    f"mean={q_hat.mean():.4f}")

    def predict(self, y_pred: np.ndarray,
                regime_id: str = "B") -> Dict[str, np.ndarray]:
        """
        예측 구간 반환.
        y_pred: [N_nodes, T] 또는 [N_nodes]
        """
        if regime_id not in self.q_hat_dict:
            raise RuntimeError(f"Regime '{regime_id}' not calibrated. "
                               f"Available: {list(self.q_hat_dict.keys())}")
        q_hat = self.q_hat_dict[regime_id]
        return {
            "y_pred": y_pred,
            "lower": y_pred - q_hat,
            "upper": y_pred + q_hat,
            "interval_width": 2 * q_hat,
        }

    def evaluate_coverage(self, y_pred: np.ndarray, y_true: np.ndarray,
                          regime_id: str = "B") -> Dict[str, float]:
        """PICR (prediction interval coverage rate) 및 sharpness 계산."""
        interval = self.predict(y_pred, regime_id)
        covered = ((y_true >= interval["lower"]) &
                   (y_true <= interval["upper"]))
        picr = float(covered.mean())
        sharpness = float(interval["interval_width"].mean())

        logger.info(f"Coverage [{regime_id}]: PICR={picr:.4f}, "
                    f"sharpness={sharpness:.4f} (target: {1 - self.alpha:.2f})")
        return {
            f"PICR_{int((1-self.alpha)*100)}": picr,
            "sharpness": sharpness,
        }

    def save_calibration(self, path: str) -> None:
        save_data = {k: v.tolist() for k, v in self.q_hat_dict.items()}
        save_data["alpha"] = self.alpha
        with open(path, "w") as f:
            json.dump(save_data, f)
        logger.info(f"Calibration saved → {path}")

    def load_calibration(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.alpha = data.pop("alpha", self.alpha)
        self.q_hat_dict = {k: np.array(v) for k, v in data.items()}
        logger.info(f"Calibration loaded from {path}: "
                    f"regimes={list(self.q_hat_dict.keys())}")


def compute_nonconformity_scores(y_pred: np.ndarray,
                                  y_true: np.ndarray) -> np.ndarray:
    """절대 잔차 기반 nonconformity score."""
    return np.abs(y_true - y_pred)
