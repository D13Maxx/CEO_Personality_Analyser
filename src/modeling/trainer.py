"""
XGBoost trainer with Optuna hyperparameter tuning.

For each Big Five trait:
  1. Bayesian search over XGBoost hyperparameters (Optuna).
  2. k-fold CV evaluation with the best params.
  3. Final model trained on all data and saved.

Falls back to config defaults if Optuna is not installed.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    MODELS_DIR, TRAITS,
    XGB_MAX_DEPTH, XGB_N_ESTIMATORS, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE, XGB_REG_ALPHA, XGB_REG_LAMBDA,
    XGB_CV_FOLDS,
    OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_TIMEOUT,
)
from src.modeling.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed; using default hyperparameters.")


class ModelTrainer:

    def __init__(self, use_optuna=True):
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.engineer = FeatureEngineer()
        self.models: Dict[str, XGBRegressor] = {}
        self.use_optuna = use_optuna and HAS_OPTUNA

    @staticmethod
    def _default_params():
        return {
            "max_depth": XGB_MAX_DEPTH,
            "n_estimators": XGB_N_ESTIMATORS,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "reg_alpha": XGB_REG_ALPHA,
            "reg_lambda": XGB_REG_LAMBDA,
        }

    def _objective(self, trial, X, y, folds):
        """Optuna objective: maximize mean CV R2."""
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        return cross_val_score(model, X, y, cv=folds, scoring="r2").mean()

    def _find_best_params(self, trait, X, y):
        logger.info("  Tuning %s (%d trials) ...", trait, OPTUNA_N_TRIALS)
        folds = KFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=42)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: self._objective(t, X, y, folds),
            n_trials=OPTUNA_N_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            show_progress_bar=True,
        )
        logger.info("  Best R2: %.3f | Params: %s", study.best_value, study.best_params)
        return study.best_params

    def train_and_evaluate(self):
        """Run the full pipeline: prepare data, tune, cross-validate, train, save."""
        logger.info("Preparing feature matrix and labels...")
        X, Y, feature_cols = self.engineer.prepare_training_data()

        all_metrics = {}
        all_params = {}

        for trait in TRAITS:
            logger.info("--- Training %s ---", trait.upper())
            X_arr = X[feature_cols].values
            y_arr = Y[trait].values

            params = self._find_best_params(trait, X_arr, y_arr) if self.use_optuna else self._default_params()
            all_params[trait] = params

            folds = KFold(n_splits=XGB_CV_FOLDS, shuffle=True, random_state=42)
            val_maes, val_r2s, train_maes, train_r2s = [], [], [], []

            for train_idx, val_idx in folds.split(X_arr):
                model = XGBRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(X_arr[train_idx], y_arr[train_idx],
                          eval_set=[(X_arr[val_idx], y_arr[val_idx])], verbose=False)

                vp = model.predict(X_arr[val_idx])
                val_maes.append(mean_absolute_error(y_arr[val_idx], vp))
                val_r2s.append(r2_score(y_arr[val_idx], vp))

                tp = model.predict(X_arr[train_idx])
                train_maes.append(mean_absolute_error(y_arr[train_idx], tp))
                train_r2s.append(r2_score(y_arr[train_idx], tp))

            avg_val_mae = np.mean(val_maes)
            avg_val_r2 = np.mean(val_r2s)
            avg_train_mae = np.mean(train_maes)
            avg_train_r2 = np.mean(train_r2s)

            all_metrics[trait] = {
                "val_mae_mean": avg_val_mae, "val_mae_std": np.std(val_maes),
                "val_r2_mean": avg_val_r2, "val_r2_std": np.std(val_r2s),
                "train_mae_mean": avg_train_mae, "train_mae_std": np.std(train_maes),
                "train_r2_mean": avg_train_r2, "train_r2_std": np.std(train_r2s),
            }

            logger.info("CV Train -- MAE: %.3f, R2: %.3f", avg_train_mae, avg_train_r2)
            logger.info("CV Val   -- MAE: %.3f, R2: %.3f", avg_val_mae, avg_val_r2)

            if avg_train_mae < avg_val_mae * 0.5:
                logger.warning("Potential overfitting for %s: train MAE %.3f vs val MAE %.3f",
                               trait.upper(), avg_train_mae, avg_val_mae)

            logger.info("Training final %s model on full dataset...", trait)
            final_model = XGBRegressor(**params, random_state=42, n_jobs=-1)
            final_model.fit(X_arr, y_arr)
            self.models[trait] = final_model
            final_model.save_model(str(self.models_dir / f"{trait}_xgb.json"))

        def as_native(d):
            return {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in d.items()}

        with open(self.models_dir / "cv_metrics.json", "w") as f:
            json.dump({t: as_native(m) for t, m in all_metrics.items()}, f, indent=2)

        with open(self.models_dir / "best_hyperparams.json", "w") as f:
            json.dump({t: as_native(p) for t, p in all_params.items()}, f, indent=2)

        logger.info("Training complete. All models saved.")
        return all_metrics


def train_models(use_optuna=True):
    return ModelTrainer(use_optuna=use_optuna).train_and_evaluate()
