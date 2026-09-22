"""Load the official train/test CSVs prepared from
``rogue-security/prompt-injections-benchmark``.

Output for each split:
    text  : str, the raw prompt
    label : int, 0 = benign, 1 = jailbreak (injection / malicious)
"""
from __future__ import annotations
import logging
import pandas as pd

from .config import TRAIN_CSV, TEST_CSV

log = logging.getLogger(__name__)


def _normalize_label(value) -> int:
    """Map the ``benign`` / ``jailbreak`` strings to {0, 1}."""
    if isinstance(value, (int, bool)):
        return int(bool(value))
    s = str(value).strip().lower()
    if s in ("0", "benign", "legit", "legitimate", "normal", "safe"):
        return 0
    if s in ("1", "jailbreak", "injection", "malicious", "attack",
             "prompt_injection", "unsafe", "harmful", "adversarial"):
        return 1
    raise ValueError(f"Unrecognised label value: {value!r}")


def _load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path}: expected columns ['text', 'label'], got {list(df.columns)}")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].map(_normalize_label).astype(int)
    return df.reset_index(drop=True)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, test_df)`` from the official CSV files."""
    log.info("Loading official training file: %s", TRAIN_CSV)
    train = _load_csv(TRAIN_CSV)
    log.info("Loading official test file: %s", TEST_CSV)
    test  = _load_csv(TEST_CSV)
    log.info("Train rows: %d  (benign=%d, jailbreak=%d)",
             len(train), int((train["label"] == 0).sum()), int((train["label"] == 1).sum()))
    log.info("Test  rows: %d  (benign=%d, jailbreak=%d)",
             len(test),  int((test["label"] == 0).sum()),  int((test["label"] == 1).sum()))
    return train, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tr, te = load_train_test()
    print(tr.shape, te.shape)
    print(tr.head())
