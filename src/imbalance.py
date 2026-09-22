"""Imbalance handling: plain and SMOTE pipelines for ablation comparison.

Production pipeline uses `class_weight='balanced'` on each classifier rather
than SMOTE, since SMOTE in TF-IDF space synthesizes samples with no semantic
meaning for tokens (Chawla et al., 2002, was developed for low-dimensional
features). The SMOTE pipeline is retained for the ablation reported in Table 3
of the methodology.

Note: imblearn.pipeline.Pipeline must be used (not sklearn's) so that SMOTE
is applied only to training folds during cross-validation, never to the
held-out fold.
"""
from __future__ import annotations
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .config import RANDOM_SEED


def make_plain_pipeline(vectorizer, classifier) -> SkPipeline:
    """Vectorizer → Classifier (no sampler). Used in the production pipeline."""
    return SkPipeline([("vec", vectorizer), ("clf", classifier)])


def make_smote_pipeline(vectorizer, classifier) -> ImbPipeline:
    """Vectorizer → SMOTE → Classifier. Used only for the imbalance ablation."""
    return ImbPipeline([
        ("vec", vectorizer),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("clf", classifier),
    ])
