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
    """
    Fraction of the top-k recommended items that are relevant.

    Denominator is k, NOT len(top_k). Dividing by the list length rewards a
    model for returning a SHORT list: 1 hit out of 3 returned items scored
    0.3333 instead of the correct 0.1000 — a 3.3x overstatement. Short lists
    happen in practice (the content model can only rank the 36% of movies
    that have plot text, then drops the ones the user already rated), so
    unfilled slots must count against the model, not be quietly excluded.
    """
    if k <= 0:
        return None
    top_k = recommended_ids[:k]
    if len(top_k) == 0:
        return None  # model had no opinion at all — a coverage problem, not a score
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / k


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
    n_scored = 0          # users that produced a ranked list at all
    n_no_recs = 0         # users the model returned nothing for
    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue
        rec_ids = recommend_fn(user_id)
        if not rec_ids:
            n_no_recs += 1
            continue
        n_scored += 1
        p = precision_at_k(rec_ids, relevant, k)
        r = recall_at_k(rec_ids, relevant, k)
        n = ndcg_at_k(rec_ids, relevant, k)
        if p is not None:
            precisions.append(p)
        if r is not None:
            recalls.append(r)
        if n is not None:
            ndcgs.append(n)

    # Each metric can individually return None, so the three means are not
    # guaranteed to be averaged over the same users. Reporting one
    # "n_users": len(precisions) for all three hid that. n_users is now the
    # number of users actually scored, and n_* records what each mean was
    # really computed over — if those three disagree, the metrics are not
    # directly comparable and the report needs to say so.
    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else None,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else None,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else None,
        "n_users": n_scored,
        "n_precision": len(precisions),
        "n_recall": len(recalls),
        "n_ndcg": len(ndcgs),
        "n_no_recs": n_no_recs,
    }


def evaluate_ranking_at_ks(recommend_fn, relevant_by_user, ks):
    """
    Same metrics as evaluate_ranking(), but for SEVERAL cutoffs in one pass.

    evaluate_ranking() called in a `for k in Ks` loop re-runs recommend_fn from
    scratch on every iteration — measured: 1,806 recommendation builds for 602
    users across Ks=[5,10,20], 3x more work than needed. The ranked list does
    not depend on k, only the cutoff does, so this builds it once per user and
    slices it at each k.

    recommend_fn must return at least max(ks) items per user (the caller asks
    its model for n=max(Ks)); shorter lists are handled but count unfilled
    slots against the model, exactly as precision_at_k() does.

    Returns {k: result_dict}, each result_dict identical in shape to what
    evaluate_ranking() returns for that k.
    """
    ks = sorted(ks)
    acc = {k: {"p": [], "r": [], "n": [], "scored": 0} for k in ks}
    n_no_recs = 0

    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue
        rec_ids = recommend_fn(user_id)   # built ONCE per user, sliced per k below
        if not rec_ids:
            n_no_recs += 1
            continue
        for k in ks:
            a = acc[k]
            a["scored"] += 1
            for key, fn in (("p", precision_at_k), ("r", recall_at_k), ("n", ndcg_at_k)):
                value = fn(rec_ids, relevant, k)
                if value is not None:
                    a[key].append(value)

    return {
        k: {
            f"Precision@{k}": float(np.mean(a["p"])) if a["p"] else None,
            f"Recall@{k}": float(np.mean(a["r"])) if a["r"] else None,
            f"NDCG@{k}": float(np.mean(a["n"])) if a["n"] else None,
            "n_users": a["scored"],
            "n_precision": len(a["p"]),
            "n_recall": len(a["r"]),
            "n_ndcg": len(a["n"]),
            "n_no_recs": n_no_recs,
        }
        for k, a in acc.items()
    }
