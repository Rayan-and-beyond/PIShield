"""
PIShield: Prompt Injection Detection for LLM-Powered Code Assistants

This package contains all modules for loading, preprocessing, feature engineering,
modeling, and evaluation of prompt injection detection using interpretable ML classifiers.

Modules:
    config              # Paths, constants, and random seed
    data_loading        # CSV loading and label normalization
    preprocessing       # Text normalization and deduplication
    feature_engineering # TF-IDF and Bag-of-Words vectorizer factories
    models              # SVM, Random Forest, XGBoost model factories
    imbalance           # Imbalance diagnostics and pipelines
    evaluate            # Classification metrics
"""
