"""Project-wide configuration: paths, seeds, dataset file locations.

The official experimental data come from
``rogue-security/prompt-injections-benchmark`` (Hugging Face), saved locally
as a stratified 70/30 train/test pair.  No alternative dataset is supported.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

for p in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Official train / test files (stratified 70/30, prepared from
# rogue-security/prompt-injections-benchmark).
TRAIN_CSV = DATA_DIR / "train_dataset_70.csv"
TEST_CSV  = DATA_DIR / "test_dataset_30.csv"

RANDOM_SEED = 42
CV_FOLDS = 5
