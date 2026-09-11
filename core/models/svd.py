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
    def __init__(self, n_factors=10, center="user+item", damping=5):
        # n_factors=10, damping=5 — selected together by a 2-D validation
        # grid search (k in {5,10,15,20,30,50} x damping in {1,5,10,25,50}),
        # not by looking at test performance. (k=10, damping=5) minimized
        # validation RMSE (0.8789); the earlier damping=10 guess was inside
        # noise but damping=5 is a real, reproducible improvement.
        self.n_factors = n_factors

        # damping: how much to shrink an item's bias toward 0 when it has
        # few ratings — bias = sum_of_residuals / (count + damping). Must
        # be > 0: damping=0 divides by zero for any item with 0 train
        # ratings (common — 1,641 of 9,724 items have none in the 60% split).
        if damping <= 0:
            raise ValueError("damping must be > 0 (damping=0 divides by zero for unrated items)")
        self.damping = damping

        # "user+item"  -> best RMSE (0.8723) — use for the rating-error table.
        # "user"       -> best ranking (NDCG@10 0.1060 vs 0.0398, validated in
        #                 notebook Step 4) — use as the SVD feed into the
        #                 hybrid blend later. Item bias is a strong error-
        #                 reducer but it dominates and flattens personalized
        #                 ranking (measured, see project notes).
        if center not in ("user", "user+item"):
            raise ValueError('center must be "user" or "user+item"')
        self.center = center

        self._user_ids = None
        self._movie_ids = None
        self._user_means = None
        self._seen_mask = None  # which (user, movie) pairs were in train

        # Store the FACTORS, not the full dense reconstruction —
        # (610 x 20) + (20 x 9724) is far smaller than (610 x 9724).
        self._user_factors = None   # U * sigma  -> shape (n_users, k)
        self._item_factors = None   # Vt         -> shape (k, n_movies)
        self._item_bias = None

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
        user_centered = matrix - self._user_means.reshape(-1, 1)

        if self.center == "user+item":
            # Item bias: average deviation from each rater's own mean, damped
            # by (count + self.damping) so an item with only 1-2 ratings
            # doesn't get an extreme bias from noise (a single 5-star
            # shouldn't imply +5).
            residual_sum = np.nansum(user_centered, axis=0)
            rating_count = np.sum(~np.isnan(user_centered), axis=0)
            self._item_bias = residual_sum / (rating_count + self.damping)
        else:
            # "user" mode: no item bias term — kept as a same-shaped zero
            # array so score_all_items() never needs an if-branch.
            self._item_bias = np.zeros(matrix.shape[1])

        # Remove item bias too (a no-op when it's all zeros), before SVD —
        # factors only need to explain whatever signal is left.
        residual = user_centered - self._item_bias.reshape(1, -1)
        residual = np.nan_to_num(residual, nan=0.0)
        U, sigma, Vt = svds(residual, k=self.n_factors)

        # svds returns singular values ASCENDING — resort so factor 0 is the
        # strongest. Predictions are unaffected (the reconstruction sums over
        # all k), but ordered factors are what let us talk about "the top
        # latent factors" in the report, and let us truncate k meaningfully.
        order = np.argsort(-sigma)
        U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

        self._user_factors = U * sigma  # fold sigma into U once, reuse forever
        self._item_factors = Vt
        return self

    def score_all_items(self, user_id, clip=True):
        """
        Predicted rating for EVERY movie, for one user — returns a pandas
        Series indexed by movieId. This is the method the hybrid combiner
        (SVD + content) will call later: one vectorized lookup instead of
        looping predict() 9,724 times per user.

        clip=True  -> predicted RATINGS, bounded to 0.5-5.0. Use for RMSE/MAE.
        clip=False -> RAW scores, unbounded. Use for ranking and for the
                      hybrid blend. Clipping is right for rating error but
                      wrong for ordering: it pins the strongest predictions
                      to a flat 5.0 ceiling, turning real differences into
                      ties, and it truncates the distribution the hybrid's
                      per-user z-score is computed over.
        """
        user_idx = np.where(self._user_ids == user_id)[0]
        if len(user_idx) == 0:
            return None  # unknown user (cold-start) — caller must handle this
        user_idx = user_idx[0]

        # One matrix-vector multiply reconstructs this user's full predicted row.
        predicted = self._user_factors[user_idx] @ self._item_factors
        predicted = predicted + self._user_means[user_idx] + self._item_bias
        if clip:
            predicted = np.clip(predicted, 0.5, 5.0)

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
        scores = self.score_all_items(user_id, clip=False)
        if scores is None:
            return pd.Series(dtype=float)  # unknown user -> no recommendations

        if exclude_seen:
            user_idx = np.where(self._user_ids == user_id)[0][0]
            seen = pd.Series(self._seen_mask[user_idx], index=self._movie_ids)
            scores = scores[~seen]

        # kind="mergesort" is a STABLE sort — ties keep their original order,
        # and `scores` starts out indexed in ascending movieId order, so ties
        # resolve to ascending movieId. Without this, the default quicksort
        # is not stable and the same inputs can print a different top-10 on
        # different runs whenever two items tie on score.
        return scores.sort_values(ascending=False, kind="mergesort").head(n)
