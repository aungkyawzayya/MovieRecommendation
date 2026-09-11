"""
ranking.py
Top-N ranking quality metrics — Precision@K, Recall@K, NDCG@K — separate
from metrics.py's rating-error metrics (RMSE/MAE) because they answer a
different question: RMSE asks "how close was the predicted number", these
ask "did the right movies end up near the top of the list". A model can
win on one and lose on the other (measured earlier: the RMSE-best SVD
variant scores much worse on NDCG@10 than the user-centered variant).
"""

import numpy as np
import pandas as pd


def build_relevant_items(df, threshold=4.0, user_col="userId", movie_col="movieId", rating_col="rating"):
    """
    Groups a long-form ratings DataFrame into {userId: set of movieIds
    rated >= threshold} — the "ground truth" ranking metrics are scored
    against. threshold=4.0 matches the earlier design-consult convention:
    a rating of 4 or 5 counts as something the user actually wanted
    recommended, not just tolerated.
    """
    relevant = df[df[rating_col] >= threshold].groupby(user_col)[movie_col].apply(set)
    return relevant.to_dict()


def precision_at_k(recommended_ids, relevant_ids, k):
    """Fraction of the top-k recommended items that are relevant."""
    top_k = recommended_ids[:k]
    if len(top_k) == 0:
        return None
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / len(top_k)


def recall_at_k(recommended_ids, relevant_ids, k):
    """Fraction of the user's relevant items that appear in the top-k."""
    if len(relevant_ids) == 0:
        return None
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids, relevant_ids, k):
    """
    Normalized Discounted Cumulative Gain at k, binary relevance (1 if the
    item is in relevant_ids, else 0). Unlike Precision/Recall, this cares
    about ORDER — a relevant item at rank 1 counts more than the same item
    at rank 10, via the 1/log2(rank+1) discount.
    """
    top_k = recommended_ids[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant_ids)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return None
    return dcg / idcg


def evaluate_ranking(recommend_fn, relevant_by_user, k=10):
    """
    Runs precision/recall/NDCG @k for every user with at least one
    relevant item, and averages the results.

    recommend_fn: a function user_id -> list of recommended movieIds,
                  already ranked best-first and already excluding
                  whatever "seen" set the caller decided on. Kept as a
                  plain callable (not tied to one model's interface) so
                  the SAME function evaluates SVDRecommender,
                  ContentBasedRecommender, or a future hybrid identically
                  — each has a different recommend_top_n() signature
                  (SVD carries its own seen mask, content needs seen_ids
                  passed in), so the caller adapts that, not this function.
    relevant_by_user: output of build_relevant_items().

    Returns a dict of mean Precision@K / Recall@K / NDCG@K plus how many
    users were actually scored (n_users) — coverage matters here because
    a user with a great model but nothing relevant in their held-out set
    contributes nothing and is silently skipped, same as sklearn's
    ranking metrics do.
    """
    precisions, recalls, ndcgs = [], [], []
    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue
        rec_ids = recommend_fn(user_id)
        if not rec_ids:
            continue
        p = precision_at_k(rec_ids, relevant, k)
        r = recall_at_k(rec_ids, relevant, k)
        n = ndcg_at_k(rec_ids, relevant, k)
        if p is not None:
            precisions.append(p)
        if r is not None:
            recalls.append(r)
        if n is not None:
            ndcgs.append(n)

    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else None,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else None,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else None,
        "n_users": len(precisions),
    }
