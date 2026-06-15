"""
Ridge 회귀 모델.

- alpha: 교차검증으로 선택 (val set IC 기준)
- expanding window 학습 (train+val 누적)
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    return spearmanr(y_true[mask], y_pred[mask]).statistic


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
) -> tuple[Ridge, StandardScaler, float, float]:
    """
    Returns:
        (model, scaler, best_alpha, val_ic)
    """
    X_tr = X_train[feature_cols].fillna(0).values
    y_tr = y_train.values

    X_vl = X_val[feature_cols].fillna(0).values
    y_vl = y_val.values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    best_alpha, best_ic = None, -np.inf
    for alpha in ALPHA_CANDIDATES:
        model = Ridge(alpha=alpha)
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_vl_s)
        ic = spearman_ic(y_vl, pred)
        logger.info(f"  alpha={alpha:>8.2f}  val IC={ic:.4f}")
        if not np.isnan(ic) and ic > best_ic:
            best_ic = ic
            best_alpha = alpha

    logger.info(f"  → 최적 alpha={best_alpha}  val IC={best_ic:.4f}")

    # train+val 합쳐서 최종 학습
    X_full = np.vstack([X_tr_s, X_vl_s])
    y_full = np.concatenate([y_tr, y_vl])
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_full, y_full)

    return final_model, scaler, best_alpha, best_ic


def predict_ridge(
    model: Ridge,
    scaler: StandardScaler,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    X_arr = X[feature_cols].fillna(0).values
    X_s = scaler.transform(X_arr)
    pred = model.predict(X_s)
    return pd.Series(pred, index=X.index, name="pred")
