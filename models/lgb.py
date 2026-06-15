"""
LightGBM 모델.

메모리 절약:
- 학습 데이터 시간 기준 서브샘플 (최근 N일)
- LGB histogram bin 축소
- early stopping
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "objective": "regression",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,           # 63→31 (메모리 절약)
    "min_child_samples": 100,   # 50→100 (과적합 방지)
    "subsample": 0.5,           # 0.8→0.5
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,          # 정규화 강화
    "max_bin": 63,              # 255→63 (메모리 절약)
    "random_state": 42,
}

# 학습에 사용할 최대 행 수 (메모리 제한)
MAX_TRAIN_ROWS = 300_000


def _spearman_ic_eval(y_pred, dataset):
    y_true = dataset.get_label()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return "IC", 0.0, True
    ic = spearmanr(y_true[mask], y_pred[mask]).statistic
    return "IC", ic, True


def _subsample_train(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    시간 기준 최근 max_rows 개 사용.
    최신 데이터가 더 유용하다는 가정.
    """
    if len(X) <= max_rows:
        return X, y
    # 날짜 내림차순 정렬 후 최근 max_rows
    dates = X.index.get_level_values("date")
    sorted_idx = np.argsort(dates)[::-1][:max_rows]
    sorted_idx = np.sort(sorted_idx)  # 시간순 복원
    logger.info(f"  학습 데이터 서브샘플: {len(X):,} → {max_rows:,} rows (최근 우선)")
    return X.iloc[sorted_idx], y.iloc[sorted_idx]


def train_lgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
    params: dict | None = None,
    max_train_rows: int = MAX_TRAIN_ROWS,
) -> tuple[lgb.Booster, float]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators")

    # 학습 데이터 서브샘플
    X_tr, y_tr = _subsample_train(X_train, y_train, max_train_rows)

    X_tr_arr = X_tr[feature_cols].fillna(0).values
    X_vl_arr = X_val[feature_cols].fillna(0).values
    y_tr_arr = y_tr.values
    y_vl_arr = y_val.values

    dtrain = lgb.Dataset(X_tr_arr, label=y_tr_arr,
                         feature_name=feature_cols, free_raw_data=True)
    dval   = lgb.Dataset(X_vl_arr, label=y_vl_arr,
                         feature_name=feature_cols, reference=dtrain,
                         free_raw_data=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    booster = lgb.train(
        p,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        valid_names=["val"],
        feval=_spearman_ic_eval,
        callbacks=callbacks,
    )

    val_pred = booster.predict(X_vl_arr)
    val_ic = spearmanr(y_vl_arr, val_pred).statistic
    best_iter = booster.best_iteration
    logger.info(f"  best iter={best_iter}  val IC={val_ic:.4f}")

    # train+val 합쳐서 재학습
    X_full_arr = np.vstack([X_tr_arr, X_vl_arr])
    y_full_arr = np.concatenate([y_tr_arr, y_vl_arr])
    dfull = lgb.Dataset(X_full_arr, label=y_full_arr,
                        feature_name=feature_cols, free_raw_data=True)
    final = lgb.train(p, dfull, num_boost_round=best_iter)

    return final, val_ic


def predict_lgb(
    booster: lgb.Booster,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    X_arr = X[feature_cols].fillna(0).values
    pred = booster.predict(X_arr)
    return pd.Series(pred, index=X.index, name="pred")


def get_feature_importance(
    booster: lgb.Booster,
    feature_cols: list[str],
    importance_type: str = "gain",
) -> pd.Series:
    imp = booster.feature_importance(importance_type=importance_type)
    return pd.Series(imp, index=feature_cols).sort_values(ascending=False)
