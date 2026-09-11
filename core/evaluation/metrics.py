"""
metrics.py
Rating-error evaluation, mirroring the evaluate(y_true, y_pred) -> dict
helper pattern from COMP813 Assignment 1 — one function, reused identically
for validation (hyperparameter selection) and test (final reporting), so
there is exactly one place that defines what "error" means.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


def evaluate(y_true, y_pred):
    """
    Returns MSE, RMSE, MAE for a set of true vs predicted ratings.

    Assignment 1 also reported R^2 (variance explained) — appropriate for
    continuous house-price regression. Ratings are bounded ordinal scores
    on a 0.5-5.0 scale, where R^2 is a much less standard/interpretable
    metric in the recommender-systems literature; RMSE/MAE are the metrics
    actually used throughout this project's report, so R^2 is intentionally
    left out here rather than included just to mirror Assignment 1 exactly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    mse = float(np.mean(errors ** 2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(errors))),
    }


@dataclass
class EvalResult:
    """
    What evaluate_model() hands back.

    n_valid alone was misleading: SVD returns a number for EVERY item,
    including items with zero train ratings, where the "prediction" is
    just the user's own mean (verified: identical to the user-mean
    baseline to 0 decimal places). So a 19940/19940 coverage line made a
    silent baseline fallback look like full collaborative coverage.
    n_supported separates the two.
    """
    metrics: dict
    n_valid: int        # rows the model returned a number for
    n_total: int        # rows asked about
    n_supported: int    # subset of n_valid the model had real evidence for

    def __iter__(self):
        """
        Lets existing callers keep writing
            metrics, n_valid, n_total = evaluate_model(...)
        unchanged. C# analogy: a record's Deconstruct method — the object
        gains fields without breaking code that destructures it.
        """
        return iter((self.metrics, self.n_valid, self.n_total))

    @property
    def n_fallback(self):
        """Rows that got a number but no real evidence behind it."""
        return self.n_valid - self.n_supported

    def coverage(self):
        """One-line coverage string for printing under a metrics table."""
        return (
            f"{self.n_valid}/{self.n_total} rows predicted; "
            f"{self.n_supported} backed by real evidence, "
            f"{self.n_fallback} fell back to the user-mean baseline"
        )


def evaluate_model(model, df, rating_col="rating", allow_non_rating=False):
    """
    Evaluates a model over a (userId, movieId, rating) DataFrame.

    Scores ONE USER AT A TIME via model.score_all_items(), not one ROW at a
    time via model.predict(). predict() internally rebuilds the user's whole
    score vector on every call, so the old row-wise df.apply() rebuilt it
    once per rating — ~20k times for a single test pass. Grouping by user
    does that work once per user instead (610 times), which matters most in
    the Step 2 grid search, where a full validation pass runs 30 times.

    allow_non_rating: guard against a real trap. ContentBasedRecommender
    outputs cosine similarity on a 0-1 scale, NOT a star rating — feeding it
    here silently produced "RMSE 3.71", a meaningless number that would look
    perfectly at home in a comparison table. Models declare produces_ratings;
    pass allow_non_rating=True only if you deliberately want that comparison.
    """
    if not getattr(model, "produces_ratings", True) and not allow_non_rating:
        raise ValueError(
            f"{type(model).__name__} does not output star ratings, so RMSE/MAE "
            f"against a 0.5-5.0 rating column is not meaningful. Evaluate it with "
            f"core.evaluation.ranking instead, or pass allow_non_rating=True if "
            f"you really intend this."
        )

    # Work in POSITIONS (numpy), not index labels. An earlier version of this
    # function assigned with predicted.loc[rows.index], which needs the frame's
    # index to be unique — it raised "cannot set using a list-like indexer with
    # a different length than the value" on any df built by concatenating
    # slices (e.g. pd.concat([train_df, val_df]) without reset_index). The
    # split() frames are all reset_index'd so the notebook never hit it, but
    # evaluation frames get concatenated often enough that this must not care.
    user_ids = df["userId"].to_numpy()
    movie_ids = df["movieId"].to_numpy()
    predicted = np.full(len(df), np.nan)
    can_batch = hasattr(model, "score_all_items")

    for user_id in pd.unique(user_ids):
        pos = np.flatnonzero(user_ids == user_id)   # row POSITIONS for this user
        if can_batch:
            scores = model.score_all_items(user_id)
            if scores is None:
                continue  # unknown user (cold-start) — leave as NaN
            # reindex() maps this user's movieIds onto the score Series in one
            # lookup; movieIds the model doesn't know about come back as NaN.
            predicted[pos] = scores.reindex(movie_ids[pos]).to_numpy()
        else:
            predicted[pos] = [model.predict(user_id, mid) for mid in movie_ids[pos]]

    valid = ~np.isnan(predicted)
    metrics = evaluate(df[rating_col].to_numpy()[valid], predicted[valid])

    # supported_items: movies the model has actual evidence for (>=1 train
    # rating for SVD, plot text for the content model). A model that doesn't
    # declare it is assumed to support everything it predicted.
    supported = getattr(model, "supported_items", None)
    if supported is None:
        n_supported = int(valid.sum())
    else:
        n_supported = int((valid & np.isin(movie_ids, np.asarray(supported))).sum())

    return EvalResult(metrics, int(valid.sum()), len(df), n_supported)
