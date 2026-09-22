"""Build TF-IDF and Bag-of-Words feature representations.

Mathematical definitions
========================
Bag-of-Words: for term t in document d,
    BoW(t, d) = count(t, d)

TF-IDF: for term t in document d, document collection D,
    tf(t, d)   = count(t, d) / sum_t' count(t', d)
    idf(t, D)  = log( (1 + |D|) / (1 + df(t)) ) + 1     (sklearn convention)
    tfidf(t,d) = tf(t,d) * idf(t, D)
The vectors are L2-normalized so cosine similarity is well-defined.

We use word 1-2 grams (motivated by Jain et al. [5]: bigrams capture characteristic
phrases like "ignore previous", "you are now", "disregard above").

Feature dimensionality is capped via max_features and min_df to control overfitting.
"""
from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Conservative defaults; refined further during hyperparameter search.
TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20000,
                    sublinear_tf=True, strip_accents="unicode")
BOW_PARAMS   = dict(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20000,
                    strip_accents="unicode")


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(**TFIDF_PARAMS)


def make_bow() -> CountVectorizer:
    return CountVectorizer(**BOW_PARAMS)
