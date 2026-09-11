"""
metrics.py
Rating-error evaluation, mirroring the evaluate(y_true, y_pred) -> dict
helper pattern from COMP813 Assignment 1 — one function, reused identically
for validation (hyperparameter selection) and test (final reporting), so
there is exactly one place that defines what "error" means.
"""

import numpy as np


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


def evaluate_model(model, df, rating_col="rating"):
    """
    Convenience wrapper: runs model.predict(userId, movieId) row-by-row
    over a (userId, movieId, rating) DataFrame and evaluates the result.
    Rows the model can't predict (cold-start user/movie) are dropped —
    same behaviour the notebook cells have used from the start.

    Returns (metrics_dict, n_valid, n_total) so the caller can report
    coverage alongside the error numbers.
    """
    predicted = df.apply(
        lambda row: model.predict(row["userId"], row["movieId"]), axis=1
    )
    valid = predicted.notna()
    metrics = evaluate(df.loc[valid, rating_col], predicted[valid])
    return metrics, int(valid.sum()), len(df)
