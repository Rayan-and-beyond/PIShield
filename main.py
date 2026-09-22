"""End-to-end training / evaluation pipeline for prompt-injection detection.

Uses the official train/test CSV files prepared from
``rogue-security/prompt-injections-benchmark``:
    data/train_dataset_70.csv   (3500 prompts, stratified)
    data/test_dataset_30.csv    (1500 prompts, stratified)
"""
from __future__ import annotations
import logging
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

from src.config import RANDOM_SEED, CV_FOLDS, MODELS_DIR
from src.data_loading import load_train_test
from src.preprocessing import preprocess
from src.feature_engineering import make_tfidf, make_bow
from src.models import make_svm, make_rf, make_xgb, SEARCH_SPACES
from src.imbalance import make_plain_pipeline
from src.evaluate import evaluate_model


LOG_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("main")


def step(msg: str):
    log.info("=" * 8 + " " + msg + " " + "=" * 8)


def main():
    np.random.seed(RANDOM_SEED)

    # ---- 1. Load + preprocess ----
    step("Loading official train/test CSVs")
    train_raw, test_raw = load_train_test()

    step("Preprocessing")
    train = preprocess(train_raw)
    test  = preprocess(test_raw)
    log.info("After preprocessing -> train=%d, test=%d", len(train), len(test))

    X_train, y_train = train["text"].values, train["label"].values
    X_test,  y_test  = test["text"].values,  test["label"].values

    # class-imbalance ratio for XGBoost (N_benign / N_injection)
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    log.info("XGBoost scale_pos_weight = %.4f", spw)

    # ---- 2. Baseline cross-validation (no tuning) ----
    step("Baseline cross-validation (no tuning)")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    feature_makers = {"TFIDF": make_tfidf, "BoW": make_bow}
    model_makers   = {"SVM": make_svm, "RF": make_rf, "XGB": lambda: make_xgb(scale_pos_weight=spw)}

    for feat_name, vec_make in feature_makers.items():
        for mdl_name, mdl_make in model_makers.items():
            pipe = make_plain_pipeline(vec_make(), mdl_make())
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
            log.info("  CV %-5s + %-3s -> F1=%.4f ± %.4f",
                     feat_name, mdl_name, scores.mean(), scores.std())

    # ---- 3. Hyperparameter optimisation on TF-IDF ----
    step("Hyperparameter optimisation (GridSearchCV, TF-IDF, F1-scoring)")
    best_models = {}
    train_times = {}
    for mdl_name, mdl_make in model_makers.items():
        pipe = make_plain_pipeline(make_tfidf(), mdl_make())
        grid = GridSearchCV(
            pipe, SEARCH_SPACES[mdl_name], scoring="f1",
            cv=cv, n_jobs=-1, refit=True, verbose=0,
        )
        t0 = time.time()
        grid.fit(X_train, y_train)
        elapsed = time.time() - t0
        log.info("  %-3s best F1=%.4f params=%s (%.1fs)",
                 mdl_name, grid.best_score_, grid.best_params_, elapsed)
        best_models[mdl_name] = grid.best_estimator_
        train_times[mdl_name] = elapsed
        joblib.dump(grid.best_estimator_, MODELS_DIR / f"best_{mdl_name}_TFIDF.joblib")

    # ---- 4. Final evaluation on held-out test set ----
    # GridSearchCV(refit=True) already re-fits the best estimator on the full
    # training set, so no additional fit() call is needed here.
    step("Final evaluation on held-out test set")
    eval_rows = []
    for mdl_name, model in best_models.items():
        m = evaluate_model(mdl_name, model, X_test, y_test,
                           train_time=train_times[mdl_name], feature_name="TFIDF")
        eval_rows.append(m)

    step("DONE")
    print("\n=== TEST METRICS (held-out) ===")
    cols = ["model", "features", "accuracy", "precision", "recall", "f1", "roc_auc"]
    df = pd.DataFrame([{k: v for k, v in r.items() if k in cols} for r in eval_rows])
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

