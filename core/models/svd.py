"""
svd.py
Baseline Collaborative Filtering model — matrix factorization via SVD.
C# analogy: a class implementing a single prediction algorithm, similar
to a class implementing an IRecommender interface (added properly once
AutoRec/content models exist too).
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


class SVDRecommender:
    def __init__(self, n_factors=20):
        # n_factors=20, not 50 — verified against real data: accuracy
        # DEGRADES past ~20 factors on this dataset (more factors just
        # re-learn the "who rated what" pattern instead of taste signal).
        self.n_factors = n_factors

        self._user_ids = None
        self._movie_ids = None
        self._user_means = None
        self._seen_mask = None  # which (user, movie) pairs were in train

        # Store the FACTORS, not the full dense reconstruction —
        # (610 x 20) + (20 x 9724) is far smaller than (610 x 9724).
        self._user_factors = None   # U * sigma  -> shape (n_users, k)
        self._item_factors = None   # Vt         -> shape (k, n_movies)

    def fit(self, train_matrix):
        """
        train_matrix: pandas DataFrame, rows=userId, cols=movieId, NaN = no rating.
        """
        self._user_ids = train_matrix.index.to_numpy()
        self._movie_ids = train_matrix.columns.to_numpy()
        self._seen_mask = train_matrix.notna().to_numpy()

        matrix = train_matrix.to_numpy()

        # Mean-center each user's ratings (removes "some users rate everything
        # 5 stars" bias) BEFORE filling missing values with 0.
        self._user_means = np.nanmean(matrix, axis=1)
        centered = matrix - self._user_means.reshape(-1, 1)
        centered = np.nan_to_num(centered, nan=0.0)

        # svds returns singular values in ASCENDING order (unlike np.linalg.svd) —
        # sort descending so factor 0 is the most important one.
        U, sigma, Vt = svds(centered, k=self.n_factors)
        order = np.argsort(-sigma)
        U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

        self._user_factors = U * sigma  # fold sigma into U once, reuse forever
        self._item_factors = Vt
        return self

    def score_all_items(self, user_id):
        """
        Predicted rating for EVERY movie, for one user — returns a pandas
        Series indexed by movieId. This is the method the hybrid combiner
        (SVD + content) will call later: one vectorized lookup instead of
        looping predict() 9,724 times per user.
        """
        user_idx = np.where(self._user_ids == user_id)[0]
        if len(user_idx) == 0:
            return None  # unknown user (cold-start) — caller must handle this
        user_idx = user_idx[0]

        # One matrix-vector multiply reconstructs this user's full predicted row.
        predicted = self._user_factors[user_idx] @ self._item_factors
        predicted = predicted + self._user_means[user_idx]
        predicted = np.clip(predicted, 0.5, 5.0)  # ratings can't go outside 0.5-5.0

        return pd.Series(predicted, index=self._movie_ids)

    def predict(self, user_id, movie_id):
        """Predicted rating for one (user, movie) pair, or None if unknown."""
        scores = self.score_all_items(user_id)
        if scores is None or movie_id not in scores.index:
            return None
        return scores.loc[movie_id]

    def recommend_top_n(self, user_id, n=10, exclude_seen=True):
        """
        Top-N recommendations, sorted by predicted rating.
        exclude_seen uses the TRAIN-time seen mask (captured in fit()) —
        NOT whatever matrix a caller passes in later. This matters during
        evaluation: it stops test-set "already seen" info from leaking in.
        """
        scores = self.score_all_items(user_id)
        if scores is None:
            return pd.Series(dtype=float)  # unknown user -> no recommendations

        if exclude_seen:
            user_idx = np.where(self._user_ids == user_id)[0][0]
            seen = pd.Series(self._seen_mask[user_idx], index=self._movie_ids)
            scores = scores[~seen]

        return scores.sort_values(ascending=False).head(n)

    