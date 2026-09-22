"""Classifier definitions: SVM (RBF), Random Forest, XGBoost.

For each model we expose a base estimator and a hyperparameter search space.

Mathematical reminders (used in the methodology write-up):

SVM with kernel K solves
    min_{w,b,xi}   (1/2) ||w||^2 + C * sum_i xi_i
    s.t. y_i (<phi(x_i), w> + b) >= 1 - xi_i, xi_i >= 0
With RBF kernel K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2).

Random Forest aggregates B decorrelated trees:
    yhat(x) = majority_vote{ T_b(x) : b = 1..B }
Each tree splits to maximize Gini reduction:
    Gini(node) = 1 - sum_k p_k^2

XGBoost minimises a regularised objective with second-order Taylor expansion:
    L(phi) = sum_i l(y_i, yhat_i) + sum_t Omega(f_t)
    Omega(f) = gamma * T + (1/2) * lambda * ||w||^2
At iteration t the optimal leaf weight given gradient g_i and hessian h_i is
    w_j^* = - sum_{i in I_j} g_i / ( sum_{i in I_j} h_i + lambda )
"""
from __future__ import annotations
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import RANDOM_SEED


def make_svm() -> SVC:
    return SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED, class_weight="balanced")


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"
    )


def make_xgb(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


# Hyperparameter search spaces (kept compact so grid search is tractable).
SEARCH_SPACES = {
    "SVM": {
        "clf__C":      [0.5, 1.0, 2.0, 4.0],
        "clf__gamma":  ["scale", 0.1, 0.5],
    },
    "RF": {
        "clf__n_estimators":     [200, 400],
        "clf__max_depth":        [None, 20, 40],
        "clf__min_samples_split":[2, 5],
    },
    "XGB": {
        "clf__n_estimators":  [200, 400],
        "clf__max_depth":     [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1, 0.2],
    },
}
