"""
Elastic Net 모델.

Ridge(L2) + Lasso(L1) 조합:
  - L1: 불필요한 피처를 0으로 → 자동 피처 선택
  - L2: 다중공선성 완화, 안정적 수렴

수급 피처 중 진짜 알파 기여 피처만 살아남는 효과.
"""
from __future__ import annotations
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)

logger = logging.getLogger(__name__)

L1_RATIO_CANDIDATES = [0.1, 0.3, 0.5, 0.7, 0.9]
ALPHA_CANDIDATES    = [0.001, 0.01, 0.1, 1.0, 10.0]
MAX_TRAIN_ROWS      = 200_000   # OOM 방지


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spearmanr(y_true[mask], y_pred[mask]).statistic


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    index: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    """시간 기준 최근 max_rows 사용."""
    if len(X) <= max_rows:
        return X, y
    # 인덱스 날짜 기준 최근 우선
    dates = index.get_level_values("date")
    sorted_pos = np.argsort(dates)[::-1][:max_rows]
    sorted_pos = np.sort(sorted_pos)
    logger.info(f"  서브샘플: {len(X):,} → {max_rows:,} rows (최근 우선)")
    return X[sorted_pos], y[sorted_pos]


def train_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
) -> tuple[ElasticNet, StandardScaler, dict, float]:
    """
    Returns:
        (model, scaler, best_params, val_ic)
    """
    X_tr_raw = X_train[feature_cols].fillna(0).values
    y_tr_raw = y_train.values
    X_vl     = X_val[feature_cols].fillna(0).values
    y_vl     = y_val.values

    # 서브샘플
    X_tr_raw, y_tr_raw = _subsample(
        X_tr_raw, y_tr_raw, MAX_TRAIN_ROWS, X_train.index
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_raw)
    X_vl_s = scaler.transform(X_vl)

    best_ic     = -np.inf
    best_params = {"alpha": 0.01, "l1_ratio": 0.1}

    for l1_ratio in L1_RATIO_CANDIDATES:
        for alpha in ALPHA_CANDIDATES:
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=2000,
                random_state=42,
            )
            model.fit(X_tr_s, y_tr_raw)
            pred = model.predict(X_vl_s)
            ic = spearman_ic(y_vl, pred)
            if not np.isnan(ic) and ic > best_ic:
                best_ic     = ic
                best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

    logger.info(f"  → 최적 params={best_params}  val IC={best_ic:.4f}")

    # train+val 합쳐서 최종 학습
    X_vl_orig = X_val[feature_cols].fillna(0).values
    X_full = np.vstack([X_tr_s, scaler.transform(X_vl_orig)])
    y_full = np.concatenate([y_tr_raw, y_vl])

    final = ElasticNet(
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
        max_iter=2000,
        random_state=42,
    )
    final.fit(X_full, y_full)

    n_nonzero = (final.coef_ != 0).sum()
    logger.info(f"  비zero 피처: {n_nonzero}/{len(feature_cols)} "
                f"(L1이 {len(feature_cols)-n_nonzero}개 제거)")

    return final, scaler, best_params, best_ic


def predict_elasticnet(
    model: ElasticNet,
    scaler: StandardScaler,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    X_arr = X[feature_cols].fillna(0).values
    X_s   = scaler.transform(X_arr)
    pred  = model.predict(X_s)
    return pd.Series(pred, index=X.index, name="pred")


def get_feature_importance(
    model: ElasticNet,
    feature_cols: list[str],
) -> pd.Series:
    """계수 절대값 기준 — 0이면 L1에 의해 제거된 피처."""
    return pd.Series(
        np.abs(model.coef_), index=feature_cols
    ).sort_values(ascending=False)
