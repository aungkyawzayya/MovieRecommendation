"""
metrics.py
Rating-error evaluation, mirroring the evaluate(y_true, y_pred) -> dict
helper pattern from COMP813 Assignment 1 — one function, reused identically
for validation (hyperparameter selection) and test (final reporting), so
there is exactly one place that defines what "error" means.
"""

import numpy as np
import pandas as pd


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


def evaluate_model(model, df, rating_col="rating", user_col="userId", movie_col="movieId"):
    """
    Convenience wrapper: predicts every (userId, movieId) row in df and
    evaluates the result. Rows the model can't predict (cold-start user, or
    movie missing from the model's index) are dropped — same behaviour the
    notebook cells have used from the start.

    Grouped by user for speed: when the model exposes score_all_items(user_id)
    (SVDRecommender and ContentBasedRecommender both do), that full-vector
    call is made ONCE per user and then indexed for every row that user has
    in df — instead of recomputing the whole vector from scratch inside
    predict() for every single row. Falls back to row-by-row predict() for
    any model that only implements predict().

    Returns (metrics_dict, n_valid, n_total) so the caller can report
    coverage alongside the error numbers.

    NOTE ON COVERAGE: n_valid counts every row the model returned SOME
    number for. For SVDRecommender that is effectively every row —
    INCLUDING items with zero TRAIN ratings, where it silently falls back
    to the user-mean (+ zero item-bias), not a real collaborative-filtering
    prediction. A high n_valid/n_total is not the same claim as "the model
    has real signal for these items" — see cold_item_rate() below to report
    that distinction honestly alongside this function's output.
    """
    if hasattr(model, "score_all_items"):
        pieces = []
        for user_id, group in df.groupby(user_col, sort=False):
            scores = model.score_all_items(user_id)
            if scores is None:
                pieces.append(pd.Series(np.nan, index=group.index))
            else:
                pieces.append(scores.reindex(group[movie_col]).set_axis(group.index))
        predicted = pd.concat(pieces).reindex(df.index)
    else:
        predicted = df.apply(
            lambda row: model.predict(row[user_col], row[movie_col]), axis=1
        )

    valid = predicted.notna()
    metrics = evaluate(df.loc[valid, rating_col], predicted[valid])
    return metrics, int(valid.sum()), len(df)


def cold_item_rate(df, train_matrix, movie_col="movieId"):
    """
    Fraction of df's rows whose item has ZERO ratings in train_matrix — the
    rows where a collaborative model like SVDRecommender has no real signal
    and silently falls back to a user-only baseline instead of an actual
    collaborative-filtering prediction.

    Report this ALONGSIDE evaluate_model()'s coverage (n_valid/n_total), not
    instead of it: a model can show 100% coverage (a number for every row)
    while a real chunk of that "coverage" is this fallback, not genuine
    signal — reporting coverage alone overstates what the model is doing.

    Returns (rate, n_cold, n_total).
    """
    rated_counts = train_matrix.notna().sum(axis=0)
    cold_items = set(rated_counts.index[rated_counts == 0])
    is_cold = df[movie_col].isin(cold_items)
    return float(is_cold.mean()), int(is_cold.sum()), len(df)
