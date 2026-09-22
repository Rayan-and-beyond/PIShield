"""Text preprocessing for prompt-injection classification.

Design choices (justified in the methodology):
- Normalize whitespace and Unicode but PRESERVE punctuation/case-aware tokens
  weakly. We only lowercase because injection patterns often rely on tokens like
  "ignore previous instructions"; case alone is not a reliable discriminator and
  many prior works (Jain et al. [5], Rahman et al. [13]) lowercase prior to TF-IDF.
- We do NOT strip stopwords aggressively: prompt-injection signature phrases such
  as "ignore the above" or "you are now" rely heavily on stopwords. Removing them
  destroys the very pattern we want to detect.
- We DO drop duplicates and empty rows: duplicates are a known issue in prompt
  datasets (Jain et al. [5]) and inflate evaluation metrics.
- We keep punctuation but collapse runs of whitespace.
"""
from __future__ import annotations
import re
import unicodedata
import pandas as pd

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _CTRL_RE.sub(" ", s)
    s = s.replace("\r", " ").replace("\t", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s.lower()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[preprocess] removed {before - after} duplicate (text,label) rows")
    return df
