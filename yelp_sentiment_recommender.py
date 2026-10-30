#!/usr/bin/env python3
"""
Sentiment-aware restaurant recommender based on:
"Design of a Recommendation Framework Incorporating Sentiments from User Reviews"

Dataset:
    Yelp Open Dataset (JSON Lines)
    https://www.yelp.com/dataset

Scope:
    Restaurants in Toronto, Canada and Pittsburgh, Pennsylvania, USA.
    Each city is processed independently.

Implemented:
  1. Yelp business/review loading and city + restaurant filtering
  2. Review-level emotional offsets (lexicon-based, with negation handling)
  3. UBEB and CBEB emotional-bias equations from the paper
  4. Revised ratings from equations (6) and (7)
  5. Four collaborative-filtering predictors:
       - Truncated-SVD matrix factorization
       - Non-negative matrix factorization
       - User-based cosine CF
       - Item-based Pearson/correlation CF
  6. MAE, RMSE, Precision@5, Recall@5 and F1@5
  7. A dependency-based aspect extraction baseline and AFINN sentiment scores

Important reproducibility notes:
  * The paper uses SentiWordNet for emotional offsets and AFINN for aspect
    sentiment. This script uses AFINN scores for both by default so that it
    can run with a single lexicon. Set --emotion-lexicon sentiwordnet to use
    NLTK's SentiWordNet corpus for the emotional-offset stage.
  * The paper's exact phrase-dependency rules and its full multi-criteria
    aggregation implementation are not fully specified in the article.
    The aspect module below is therefore an executable approximation:
    spaCy noun chunks are aspect candidates; AFINN scores the containing
    sentence; WordNet Wu-Palmer similarity optionally groups non-top aspects.
  * The paper reports five-fold cross-validation but does not specify the
    exact fold assignment. This implementation uses shuffled interaction-level
    folds. For very large city subsets, use --max-reviews-per-city to control
    runtime and memory.
  * The Yelp dataset is not downloaded automatically. Download and extract
    it from Yelp, then pass the directory containing business.json and
    review.json (or the academic-dataset filenames) with --data-dir.

Example:
    pip install -r requirements.txt
    python yelp_sentiment_recommender.py --data-dir ./yelp_dataset

Optional NLTK corpora for SentiWordNet / WordNet:
    python -m nltk.downloader punkt averaged_perceptron_tagger
    python -m nltk.downloader sentiwordnet wordnet omw-1.4

Optional spaCy model:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

try:
    from afinn import Afinn
except ImportError:
    Afinn = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

try:
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import wordnet as wn
except ImportError:
    nltk = None
    swn = None
    wn = None

try:
    import spacy
except ImportError:
    spacy = None


LOGGER = logging.getLogger("yelp_sentiment_recommender")
CITY_CONFIG = {
    "Toronto": {"state": "ON", "country": "Canada"},
    "Pittsburgh": {"state": "PA", "country": "USA"},
}


# ---------------------------------------------------------------------------
# 1. Dataset loading and filtering
# ---------------------------------------------------------------------------

def find_json_file(data_dir: Path, stem: str) -> Path:
    candidates = [
        data_dir / f"{stem}.json",
        data_dir / f"yelp_academic_dataset_{stem}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {stem}.json or yelp_academic_dataset_{stem}.json "
        f"inside {data_dir}"
    )


def read_json_lines(path: Path, chunksize: int = 250_000) -> pd.DataFrame:
    """Read Yelp's one-JSON-object-per-line format."""
    LOGGER.info("Reading %s", path)
    frames = []
    for chunk in pd.read_json(path, lines=True, chunksize=chunksize):
        frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def is_restaurant(categories) -> bool:
    """Yelp categories may be a comma-separated string or null."""
    if not isinstance(categories, str):
        return False
    return any(c.strip().lower() == "restaurants"
               for c in categories.split(","))


def load_city_data(
    data_dir: Path,
    city: str,
    max_reviews: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    business_path = find_json_file(data_dir, "business")
    review_path = find_json_file(data_dir, "review")

    businesses = read_json_lines(business_path)
    required_business = {"business_id", "name", "city", "categories"}
    missing = required_business - set(businesses.columns)
    if missing:
        raise ValueError(f"business.json is missing columns: {sorted(missing)}")

    city_businesses = businesses[
        businesses["city"].fillna("").str.strip().str.casefold()
        == city.casefold()
    ].copy()
    city_businesses = city_businesses[
        city_businesses["categories"].apply(is_restaurant)
    ].copy()

    if city_businesses.empty:
        raise ValueError(
            f"No businesses classified as Restaurants found for {city}. "
            "Check the city spelling and the dataset version."
        )

    business_ids = set(city_businesses["business_id"])
    LOGGER.info("%s: %d restaurant businesses", city, len(business_ids))

    # Read review data in chunks and keep only reviews for the selected city.
    LOGGER.info("Filtering review.json for %s", city)
    selected_reviews = []
    for chunk in pd.read_json(review_path, lines=True, chunksize=250_000):
        part = chunk[chunk["business_id"].isin(business_ids)]
        if not part.empty:
            selected_reviews.append(part)

    if not selected_reviews:
        raise ValueError(f"No reviews found for restaurants in {city}")

    reviews = pd.concat(selected_reviews, ignore_index=True)
    if max_reviews and len(reviews) > max_reviews:
        reviews = reviews.sample(
            n=max_reviews, random_state=random_state
        ).reset_index(drop=True)
        LOGGER.warning(
            "%s: sampled %d reviews for runtime control",
            city, max_reviews
        )

    reviews["stars"] = pd.to_numeric(reviews["stars"], errors="coerce")
    reviews = reviews.dropna(subset=["user_id", "business_id", "stars", "text"])
    reviews = reviews[
        reviews["stars"].between(1, 5)
    ].copy()

    city_businesses = city_businesses.drop_duplicates("business_id")
    return city_businesses, reviews


# ---------------------------------------------------------------------------
# 2. Sentiment lexicons and paper equations (1)-(7)
# ---------------------------------------------------------------------------

class ReviewSentiment:
    def __init__(self, emotion_lexicon: str = "afinn"):
        self.emotion_lexicon = emotion_lexicon
        self.afinn = Afinn() if Afinn is not None else None
        self.vader = (
            SentimentIntensityAnalyzer()
            if SentimentIntensityAnalyzer is not None else None
        )
        self._senti_cache = {}

        if emotion_lexicon == "sentiwordnet":
            if nltk is None or swn is None:
                raise ImportError(
                    "Install nltk and download SentiWordNet: "
                    "python -m nltk.downloader sentiwordnet wordnet omw-1.4"
                )
            try:
                swn.senti_synsets("good")
            except LookupError as exc:
                raise LookupError(
                    "SentiWordNet corpus is missing. Run: "
                    "python -m nltk.downloader sentiwordnet wordnet omw-1.4"
                ) from exc

        if emotion_lexicon == "afinn" and self.afinn is None:
            raise ImportError("Install afinn: pip install afinn")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z']+", str(text).lower())

    def _sentiwordnet_score(self, token: str) -> float:
        if token in self._senti_cache:
            return self._senti_cache[token]
        score = 0.0
        try:
            synsets = list(swn.senti_synsets(token))
            if synsets:
                # Use the first available WordNet sense, as a transparent
                # deterministic approximation when context disambiguation
                # is not specified by the paper.
                score = synsets[0].pos_score() - synsets[0].neg_score()
        except (LookupError, Exception):
            score = 0.0
        self._senti_cache[token] = float(score)
        return float(score)

    def token_score(self, token: str) -> float:
        if self.emotion_lexicon == "sentiwordnet":
            return self._sentiwordnet_score(token)
        return float(self.afinn.score(token))

    def review_offset(self, text: str) -> float:
        """
        Paper equations (1)-(3):
          polarity(sub) = +1 for an even number of negative words,
                          -1 for an odd number.
          sentiment(sub) = polarity(sub) * sum(word sentiment scores).
          offset(review) = sum(sub-sentence sentiments) / number of
                           sentiment-bearing words.

        Sentence splitting is used as the executable sub-sentence boundary.
        AFINN/SentiWordNet scores of zero are not counted as sentiment words.
        """
        sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
        total_sentiment = 0.0
        total_sentiment_words = 0

        for sentence in sentences:
            tokens = self.tokenize(sentence)
            scored = [(t, self.token_score(t)) for t in tokens]
            scored = [(t, s) for t, s in scored if s != 0]
            if not scored:
                continue

            # Count explicit negation words for the paper's polarity parity
            # rule (e.g. no, not, never, neither, hardly). A negative
            # sentiment word such as "bad" is not itself a negator.
            negators = {"no", "not", "never", "neither", "nor",
                        "none", "nobody", "nothing", "nowhere",
                        "hardly", "scarcely", "barely", "n't"}
            negative_count = sum(1 for token in self.tokenize(sentence)
                                 if token in negators or token.endswith("n't"))
            polarity = 1 if negative_count % 2 == 0 else -1
            total_sentiment += polarity * sum(score for _, score in scored)
            total_sentiment_words += len(scored)

        if total_sentiment_words == 0:
            return 0.0
        return total_sentiment / total_sentiment_words


def add_emotional_offsets(
    reviews: pd.DataFrame,
    sentiment: ReviewSentiment,
) -> pd.DataFrame:
    result = reviews.copy()
    LOGGER.info("Calculating review emotional offsets...")
    result["emotional_offset"] = result["text"].fillna("").map(
        sentiment.review_offset
    )
    return result


def calculate_ubeb_cbeb(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Equations (4)-(5):
      UBEB(i,j) = offset(i,j) - mean(offset(i,k) for restaurants k visited by i)
      CBEB(i,j) = UBEB(i,j) - mean(offset(k,j) for users k who visited j)

    Each Yelp review is an interaction. If a user has multiple reviews for
    the same restaurant, offsets are averaged first to form one user-item row.
    """
    interactions = (
        reviews.groupby(["user_id", "business_id"], as_index=False)
        .agg(
            stars=("stars", "mean"),
            emotional_offset=("emotional_offset", "mean"),
            review_count=("review_id", "count")
        )
    )

    user_mean = interactions.groupby("user_id")[
        "emotional_offset"
    ].transform("mean")
    interactions["ubeb_bias"] = (
        interactions["emotional_offset"] - user_mean
    )

    restaurant_mean = interactions.groupby("business_id")[
        "emotional_offset"
    ].transform("mean")
    interactions["cbeb_bias"] = (
        interactions["ubeb_bias"] - restaurant_mean
    )

    # Equations (6)-(7): revised rating = original rating - emotional bias.
    interactions["rating_original"] = interactions["stars"]
    interactions["rating_ubeb"] = (
        interactions["rating_original"] - interactions["ubeb_bias"]
    ).clip(1, 5)
    interactions["rating_cbeb"] = (
        interactions["rating_original"] - interactions["cbeb_bias"]
    ).clip(1, 5)
    return interactions


# ---------------------------------------------------------------------------
# 3. Rating predictors
# ---------------------------------------------------------------------------

class SparseRatingModel:
    """Common interface for sparse rating prediction models."""

    def fit(self, train: pd.DataFrame):
        self.train = train.copy()
        self.users = pd.Index(train["user_id"].unique())
        self.items = pd.Index(train["business_id"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.item_to_idx = {b: i for i, b in enumerate(self.items)}

        rows = train["user_id"].map(self.user_to_idx).to_numpy()
        cols = train["business_id"].map(self.item_to_idx).to_numpy()
        vals = train["rating"].astype(float).to_numpy()
        self.matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.users), len(self.items)),
            dtype=np.float32,
        )
        self.global_mean = float(np.mean(vals))
        self.user_means = np.asarray(self.matrix.sum(axis=1)).ravel()
        counts = np.diff(self.matrix.indptr)
        self.user_means = np.divide(
            self.user_means, counts,
            out=np.full_like(self.user_means, self.global_mean),
            where=counts > 0
        )
        self.item_means = np.asarray(self.matrix.sum(axis=0)).ravel()
        item_counts = np.diff(self.matrix.tocsc().indptr)
        self.item_means = np.divide(
            self.item_means, item_counts,
            out=np.full_like(self.item_means, self.global_mean),
            where=item_counts > 0
        )
        return self

    def predict_one(self, user_id, business_id) -> float:
        raise NotImplementedError

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.array([
            self.predict_one(u, b)
            for u, b in zip(test["user_id"], test["business_id"])
        ], dtype=float)

    def fallback(self, user_id, business_id) -> float:
        ui = self.user_to_idx.get(user_id)
        ii = self.item_to_idx.get(business_id)
        if ui is not None and ii is not None:
            return float(
                0.5 * self.user_means[ui] + 0.5 * self.item_means[ii]
            )
        if ui is not None:
            return float(self.user_means[ui])
        if ii is not None:
            return float(self.item_means[ii])
        return self.global_mean


class SVDModel(SparseRatingModel):
    def __init__(self, factors=5, random_state=42):
        self.factors = factors
        self.random_state = random_state

    def fit(self, train):
        super().fit(train)
        n_components = max(
            1, min(self.factors, self.matrix.shape[0] - 1,
                   self.matrix.shape[1] - 1)
        )
        self.model = TruncatedSVD(
            n_components=n_components, random_state=self.random_state
        )
        self.user_factors = self.model.fit_transform(self.matrix)
        self.item_factors = self.model.components_.T
        return self

    def predict_one(self, user_id, business_id):
        ui = self.user_to_idx.get(user_id)
        ii = self.item_to_idx.get(business_id)
        if ui is None or ii is None:
            return self.fallback(user_id, business_id)
        # Paper's SVD prediction uses the user's row mean plus latent-factor
        # product. TruncatedSVD is used here as a sparse scalable solver.
        return float(
            self.user_means[ui] +
            np.dot(self.user_factors[ui], self.item_factors[ii])
        )


class NMFModel(SparseRatingModel):
    def __init__(self, factors=5, random_state=42, max_iter=150):
        self.factors = factors
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, train):
        super().fit(train)
        n_components = max(
            1, min(self.factors, self.matrix.shape[0] - 1,
                   self.matrix.shape[1] - 1)
        )
        self.model = NMF(
            n_components=n_components,
            init="nndsvda",
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.user_factors = self.model.fit_transform(self.matrix)
        self.item_factors = self.model.components_.T
        return self

    def predict_one(self, user_id, business_id):
        ui = self.user_to_idx.get(user_id)
        ii = self.item_to_idx.get(business_id)
        if ui is None or ii is None:
            return self.fallback(user_id, business_id)
        return float(np.dot(self.user_factors[ui], self.item_factors[ii]))


class UserBasedCF(SparseRatingModel):
    def __init__(self, neighbors=40):
        self.neighbors = neighbors

    def fit(self, train):
        super().fit(train)
        n = min(self.neighbors + 1, self.matrix.shape[0])
        self.nn = NearestNeighbors(
            n_neighbors=n, metric="cosine", algorithm="brute"
        ).fit(self.matrix)
        self.distances, self.indices = self.nn.kneighbors(self.matrix)
        return self

    def predict_one(self, user_id, business_id):
        ui = self.user_to_idx.get(user_id)
        ii = self.item_to_idx.get(business_id)
        if ui is None or ii is None:
            return self.fallback(user_id, business_id)
        neighbor_ids = self.indices[ui][1:]
        similarities = 1.0 - self.distances[ui][1:]
        numerator, denominator = 0.0, 0.0
        for neighbor, sim in zip(neighbor_ids, similarities):
            rating = self.matrix[neighbor, ii]
            if rating != 0 and sim > 0:
                numerator += sim * (float(rating) - self.user_means[neighbor])
                denominator += abs(sim)
        if denominator == 0:
            return self.fallback(user_id, business_id)
        return float(self.user_means[ui] + numerator / denominator)


class ItemBasedCF(SparseRatingModel):
    def __init__(self, neighbors=40):
        self.neighbors = neighbors

    def fit(self, train):
        super().fit(train)
        # Pearson-style item similarity: center each user's ratings by that
        # user's mean, then compute cosine similarity between item columns.
        # This is a sparse approximation to pairwise Pearson correlation.
        centered = self.matrix.copy().tolil()
        for user_idx in range(centered.shape[0]):
            row = centered.rows[user_idx]
            if row:
                mean = self.user_means[user_idx]
                centered.data[user_idx] = [
                    float(value) - mean for value in centered.data[user_idx]
                ]
        item_matrix = centered.tocsr().T.tocsr()
        n = min(self.neighbors + 1, item_matrix.shape[0])
        self.nn = NearestNeighbors(
            n_neighbors=n, metric="cosine", algorithm="brute"
        ).fit(item_matrix)
        self.distances, self.indices = self.nn.kneighbors(item_matrix)
        return self

    def predict_one(self, user_id, business_id):
        ui = self.user_to_idx.get(user_id)
        ii = self.item_to_idx.get(business_id)
        if ui is None or ii is None:
            return self.fallback(user_id, business_id)
        neighbor_ids = self.indices[ii][1:]
        similarities = 1.0 - self.distances[ii][1:]
        numerator, denominator = 0.0, 0.0
        for neighbor, sim in zip(neighbor_ids, similarities):
            rating = self.matrix[ui, neighbor]
            if rating != 0 and sim > 0:
                numerator += sim * (float(rating) - self.item_means[neighbor])
                denominator += abs(sim)
        if denominator == 0:
            return self.fallback(user_id, business_id)
        return float(self.item_means[ii] + numerator / denominator)


MODEL_FACTORIES = {
    "SVD": lambda: SVDModel(factors=5),
    "NMF": lambda: NMFModel(factors=5),
    "UserBasedCF": lambda: UserBasedCF(neighbors=40),
    "ItemBasedCF": lambda: ItemBasedCF(neighbors=40),
}


# ---------------------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean(
        (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    )))


def precision_recall_f1_at_k(
    test: pd.DataFrame,
    predictions: np.ndarray,
    train: pd.DataFrame,
    k: int = 5,
    relevance_threshold: float = 3.0,
) -> Tuple[float, float, float]:
    """
    For each test user, recommend the top-k predicted restaurants not already
    rated in that user's training data. A test rating > 3 is relevant,
    matching the paper's threshold.
    """
    test = test.copy()
    test["prediction"] = predictions
    seen = train.groupby("user_id")["business_id"].apply(set).to_dict()

    precisions, recalls = [], []
    for user_id, group in test.groupby("user_id"):
        relevant = set(
            group.loc[group["stars"] > relevance_threshold, "business_id"]
        )
        if not relevant:
            continue
        candidates = group[~group["business_id"].isin(seen.get(user_id, set()))]
        if candidates.empty:
            continue
        top_k = set(
            candidates.nlargest(min(k, len(candidates)), "prediction")[
                "business_id"
            ]
        )
        hits = len(top_k & relevant)
        precisions.append(hits / max(len(top_k), 1))
        recalls.append(hits / len(relevant))

    if not precisions:
        return 0.0, 0.0, 0.0
    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return precision, recall, f1


def evaluate_one_city(
    interactions: pd.DataFrame,
    city: str,
    folds: int = 5,
    min_user_reviews: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Five-fold split over user-business interactions. Users can therefore
    have interactions in both train and test folds, as in conventional
    collaborative-filtering evaluation. The article states five disjoint
    folds but does not specify a user-grouped split.
    """
    data = interactions.copy()
    user_counts = data.groupby("user_id")["business_id"].nunique()
    eligible_users = user_counts[user_counts >= min_user_reviews].index
    data = data[data["user_id"].isin(eligible_users)].reset_index(drop=True)

    if data["user_id"].nunique() < folds:
        raise ValueError(
            f"{city}: need at least {folds} users for {folds}-fold CV; "
            f"found {data['user_id'].nunique()}"
        )

    kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    results = []

    rating_columns = {
        "Original": "rating_original",
        "UBEB": "rating_ubeb",
        "CBEB": "rating_cbeb",
    }

    for fold, (train_idx, test_idx) in enumerate(
        kfold.split(data), start=1
    ):
        base_train = data.iloc[train_idx].copy()
        base_test = data.iloc[test_idx].copy()

        for rating_name, rating_col in rating_columns.items():
            train = base_train.rename(columns={rating_col: "rating"})
            test = base_test.rename(columns={rating_col: "rating"})
            # Evaluation ground truth remains the observed Yelp star rating.
            # Models are trained with the selected rating representation.
            for model_name, factory in MODEL_FACTORIES.items():
                LOGGER.info(
                    "%s | fold %d/%d | %s | %s",
                    city, fold, folds, rating_name, model_name
                )
                model = factory().fit(train)
                predictions = np.clip(model.predict(test), 1, 5)
                p, r, f = precision_recall_f1_at_k(
                    test, predictions, train, k=5,
                    relevance_threshold=3.0
                )
                results.append({
                    "city": city,
                    "fold": fold,
                    "rating_method": rating_name,
                    "model": model_name,
                    "MAE": mae(test["stars"], predictions),
                    "RMSE": rmse(test["stars"], predictions),
                    "Precision@5": p,
                    "Recall@5": r,
                    "F1@5": f,
                    "n_train": len(train),
                    "n_test": len(test),
                })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Aspect extraction baseline (AFINN + spaCy + WordNet)
# ---------------------------------------------------------------------------

def load_spacy_pipeline():
    if spacy is None:
        raise ImportError(
            "Aspect extraction requires spaCy. Install it and run: "
            "python -m spacy download en_core_web_sm"
        )
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise OSError(
            "spaCy model en_core_web_sm is missing. Run: "
            "python -m spacy download en_core_web_sm"
        ) from exc


def wordnet_similarity(word_a: str, word_b: str) -> float:
    if wn is None:
        return 0.0
    try:
        syn_a = wn.synsets(word_a, pos=wn.NOUN)
        syn_b = wn.synsets(word_b, pos=wn.NOUN)
        if not syn_a or not syn_b:
            return 0.0
        scores = [
            a.wup_similarity(b) or 0.0
            for a in syn_a[:3] for b in syn_b[:3]
        ]
        return max(scores, default=0.0)
    except LookupError:
        return 0.0


def extract_aspect_sentiments(
    reviews: pd.DataFrame,
    top_n: int = 10,
    similarity_threshold: float = 0.90,
    max_reviews: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Approximation of paper Section 3.2:
      - spaCy noun chunks supply aspect candidates;
      - AFINN scores the sentence containing each aspect;
      - top-N aspects are selected by frequency;
      - remaining aspects may be mapped to top aspects by WordNet WUP;
      - scores are averaged within user/business/aspect-class.

    Returns review-level aspect observations and aggregated aspect ratings.
    """
    if Afinn is None:
        raise ImportError("Aspect extraction requires: pip install afinn")
    nlp = load_spacy_pipeline()
    afinn = Afinn()

    work = reviews
    if max_reviews and len(work) > max_reviews:
        work = work.sample(max_reviews, random_state=42)

    observations = []
    for row in work[["review_id", "user_id", "business_id", "text"]].itertuples(
        index=False
    ):
        doc = nlp(str(row.text)[:10000])
        for chunk in doc.noun_chunks:
            aspect = " ".join(
                token.lemma_.lower()
                for token in chunk
                if token.pos_ in {"NOUN", "PROPN"}
            ).strip()
            if not aspect or len(aspect) < 2:
                continue
            sentence = chunk.root.sent.text
            score = float(afinn.score(sentence))
            observations.append({
                "review_id": row.review_id,
                "user_id": row.user_id,
                "business_id": row.business_id,
                "aspect": aspect,
                "sentiment_score": score,
            })

    aspect_df = pd.DataFrame(observations)
    if aspect_df.empty:
        return aspect_df, pd.DataFrame()

    frequency = aspect_df["aspect"].value_counts()
    main_aspects = list(frequency.head(top_n).index)

    def assign_class(aspect):
        if aspect in main_aspects:
            return aspect
        best_aspect, best_score = None, 0.0
        for main in main_aspects:
            score = wordnet_similarity(aspect, main)
            if score > best_score:
                best_aspect, best_score = main, score
        if best_aspect is not None and best_score >= similarity_threshold:
            return best_aspect
        return aspect  # preserve unmatched aspect rather than discard it

    aspect_df["aspect_class"] = aspect_df["aspect"].map(assign_class)
    aggregated = (
        aspect_df.groupby(
            ["user_id", "business_id", "aspect_class"], as_index=False
        )["sentiment_score"].mean()
    )
    return aspect_df, aggregated


# ---------------------------------------------------------------------------
# 6. Main execution and outputs
# ---------------------------------------------------------------------------

def run_city(
    data_dir: Path,
    output_dir: Path,
    city: str,
    args,
    sentiment: ReviewSentiment,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    businesses, reviews = load_city_data(
        data_dir, city, args.max_reviews_per_city, args.random_state
    )

    LOGGER.info(
        "%s: %d reviews, %d restaurants, %d unique users",
        city, len(reviews), businesses["business_id"].nunique(),
        reviews["user_id"].nunique()
    )

    reviews = add_emotional_offsets(reviews, sentiment)

    # The paper keeps users who have written at least five reviews for its
    # emotional-bias experiment. Apply this criterion to review records
    # before collapsing multiple reviews for a user-business pair.
    if args.emotion_min_reviews > 1:
        review_counts = reviews.groupby("user_id").size()
        eligible_users = review_counts[
            review_counts >= args.emotion_min_reviews
        ].index
        emotion_reviews = reviews[
            reviews["user_id"].isin(eligible_users)
        ].copy()
    else:
        emotion_reviews = reviews
    interactions = calculate_ubeb_cbeb(emotion_reviews)
    emotion_interactions = interactions

    interactions.to_csv(
        output_dir / f"{city.lower()}_emotional_bias_interactions.csv",
        index=False
    )
    businesses.to_csv(
        output_dir / f"{city.lower()}_restaurants.csv", index=False
    )

    # Run the four CF methods for original, UBEB and CBEB rating matrices.
    # This is the paper's emotional-bias experiment.
    metrics = evaluate_one_city(
        emotion_interactions,
        city=city,
        folds=args.folds,
        min_user_reviews=1,
        random_state=args.random_state,
    )
    metrics.to_csv(
        output_dir / f"{city.lower()}_emotional_bias_metrics_by_fold.csv",
        index=False
    )
    summary = (
        metrics.groupby(["city", "rating_method", "model"], as_index=False)
        [["MAE", "RMSE", "Precision@5", "Recall@5", "F1@5"]].mean()
    )
    summary.to_csv(
        output_dir / f"{city.lower()}_emotional_bias_metrics_summary.csv",
        index=False
    )

    if args.run_aspects:
        aspect_reviews, aspect_scores = extract_aspect_sentiments(
            reviews,
            top_n=args.top_aspects,
            similarity_threshold=args.aspect_similarity,
            max_reviews=args.max_aspect_reviews,
        )
        aspect_reviews.to_csv(
            output_dir / f"{city.lower()}_aspect_observations.csv",
            index=False
        )
        aspect_scores.to_csv(
            output_dir / f"{city.lower()}_aspect_scores.csv",
            index=False
        )

    LOGGER.info("Finished %s. Results saved under %s", city, output_dir)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Yelp sentiment-aware restaurant recommender"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing Yelp business.json and review.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("yelp_results")
    )
    parser.add_argument("--cities", nargs="+",
                        choices=list(CITY_CONFIG), default=list(CITY_CONFIG))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--emotion-min-reviews", type=int, default=5,
                        help="Minimum distinct restaurant reviews per user "
                             "for the emotional-bias experiment (paper: 5)")
    parser.add_argument("--emotion-lexicon",
                        choices=["afinn", "sentiwordnet"], default="afinn")
    parser.add_argument("--max-reviews-per-city", type=int, default=None,
                        help="Optional cap for quick experiments; default uses "
                             "all city reviews")
    parser.add_argument("--run-aspects", action="store_true",
                        help="Run spaCy/AFINN aspect extraction")
    parser.add_argument("--top-aspects", type=int, default=10)
    parser.add_argument("--aspect-similarity", type=float, default=0.90)
    parser.add_argument("--max-aspect-reviews", type=int, default=50000,
                        help="Aspect extraction is expensive; default cap 50k")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)

    sentiment = ReviewSentiment(args.emotion_lexicon)
    summaries = []

    for city in args.cities:
        summary = run_city(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            city=city,
            args=args,
            sentiment=sentiment,
        )
        summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(
        args.output_dir / "toronto_pittsburgh_metrics_summary.csv",
        index=False
    )
    print("\n=== Mean cross-validation results ===")
    print(combined.to_string(index=False))
    print(f"\nAll outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
