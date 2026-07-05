from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


@dataclass
class XGBConfig:
    n_estimators: int = 100
    max_depth: int = 3
    lr: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 5.0
    reg_alpha: float = 1.0
    min_child_weight: int = 10


@dataclass
class XGBTrainerData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class CopperXGBTrainer:
    def __init__(self, cfg: XGBConfig, data: XGBTrainerData, seed: int = 42):
        self.cfg = cfg
        self.data = data
        self.seed = seed

    def train(self):
        cfg = self.cfg
        data = self.data

        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.lr,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            min_child_weight=cfg.min_child_weight,
            random_state=self.seed,
        )

        model.fit(data.X_train, data.y_train)

        return model


def xgb_test_score(data: XGBTrainerData, fit_model: XGBClassifier):
    y_pred = fit_model.predict(data.X_test)
    y_proba = fit_model.predict_proba(data.X_test)[:, 1]

    print(f"Accuracy score: {float(accuracy_score(data.y_test, y_pred)):.2%}")
    print(f"ROC AUC score: {float(roc_auc_score(data.y_test, y_proba)):.2%}")
