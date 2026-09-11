"""
hybrid.py
Combines SVD (collaborative filtering) and content-based (plot-text) scores
into a single ranked list. Neither model alone is enough: SVD collapses to
the user-mean baseline for items with zero train ratings (Step 6's
cold-start evidence), and the content model only has plot text for 36% of
the catalogue (Step 5) — this class exists to cover what either one misses
on its own.
"""

import numpy as np
import pandas as pd


class HybridRecommender:
    # Hybrid score is a blended z-score, not a star rating — same guard
    # pattern as ContentBasedRecommender.produces_ratings, so
    # metrics.evaluate_model() refuses a meaningless RMSE against it.
    produces_ratings = False

    def __init__(self, svd_model, content_model, alpha=0.5, fallback="neutral", mode="weighted"):
        # Composition, not inheritance — C# constructor injection. A Hybrid
        # USES an SVD model and a content model, it isn't a subtype of
        # either one (has-a, not is-a).
        self._svd = svd_model
        self._content = content_model
        self.alpha = alpha

        # fallback only matters when mode="weighted" — see score_all_items().
        # "neutral": a text-less item's missing content_z counts as 0 (mean),
        #   so its blended score is just alpha*svd_z.
        # "renormalise": a text-less item gets its FULL weight handed to SVD
        #   (svd_z alone), which looks more "natural" but is a real bias:
        #   content_z has mean 0 only over items WITH text, so an
        #   average-similarity item with text shrinks toward 0 while every
        #   text-less item keeps its full svd_z — text-less items end up
        #   over-represented near the top for reasons that have nothing to
        #   do with the user liking them. Kept as an option so both can be
        #   measured side by side as an ablation, not because it's the
        #   better default.
        if fallback not in ("neutral", "renormalise"):
            raise ValueError('fallback must be "neutral" or "renormalise"')
        self.fallback = fallback

        # mode="weighted"  -> alpha*svd_z + (1-alpha)*content_z everywhere.
        # mode="switching"  -> per item, trust whichever model has REAL
        #   evidence instead of blending: SVD's own collaborative signal for
        #   any item with >=1 train rating, content similarity for a
        #   train-cold item that has plot text, and no opinion at all (NaN,
        #   simply not ranked) for an item neither model can speak to. This
        #   targets cold-start directly and sidesteps the weighted blend's
        #   text-less-item bias entirely — a second ablation, not a
        #   replacement for "weighted".
        if mode not in ("weighted", "switching"):
            raise ValueError('mode must be "weighted" or "switching"')
        self.mode = mode

    @staticmethod
    def _zscore(scores):
        """
        Per-user z-score over the given candidate set. ddof=0 (population
        std): the candidates ARE the full population being ranked here, not
        a sample drawn from something larger.
        """
        sd = scores.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            # Every candidate score is identical — e.g. a user whose content
            # profile is empty (all-NaN scores) or whose SVD row has no
            # variation at all. Dividing by 0 would blow up; 0 says "no
            # discriminating signal here" instead of crashing or faking one.
            return pd.Series(0.0, index=scores.index)
        return (scores - scores.mean()) / sd

    def _content_z_full(self, user_id, index):
        """Content z-scores reindexed onto `index` — NaN wherever there's no plot text."""
        content_scores = self._content.score_all_items(user_id)
        if content_scores is None:
            return pd.Series(np.nan, index=index)

        content_scores = content_scores.reindex(index)
        text_mask = content_scores.notna()
        if not text_mask.any():
            return pd.Series(np.nan, index=index)

        # z-score computed ONLY over items that actually have text — an item
        # with no text was never a candidate for this model in the first
        # place, so it shouldn't pull the mean/std around.
        content_z = self._zscore(content_scores[text_mask])
        return content_z.reindex(index)

    def score_all_items(self, user_id):
        """
        Blended score for every item SVD has an opinion on. SVD always
        scores the full catalogue (even cold items, via the user-mean
        fallback — see SVDRecommender.score_all_items's clip=False note),
        so its index is the frame everything else lines up against.
        clip=False: ranking needs the raw distribution, not ratings pinned
        to the 5.0 ceiling — see svd.py's docstring on why clipping would
        flatten exactly the differences z-scoring depends on.
        """
        svd_scores = self._svd.score_all_items(user_id, clip=False)
        if svd_scores is None:
            return None  # unknown user (cold-start) — caller must handle this

        svd_z = self._zscore(svd_scores)
        content_z_full = self._content_z_full(user_id, svd_scores.index)
        has_content = content_z_full.notna()

        if self.mode == "switching":
            has_train_ratings = svd_scores.index.isin(self._svd.supported_items)
            blended = pd.Series(np.nan, index=svd_scores.index)
            blended[has_train_ratings] = svd_z[has_train_ratings]
            use_content = (~has_train_ratings) & has_content
            blended[use_content] = content_z_full[use_content]
            return blended

        # mode == "weighted"
        if self.fallback == "neutral":
            return self.alpha * svd_z + (1 - self.alpha) * content_z_full.fillna(0.0)

        # fallback == "renormalise"
        blended = svd_z.copy()
        blended[has_content] = (
            self.alpha * svd_z[has_content] + (1 - self.alpha) * content_z_full[has_content]
        )
        return blended

    @property
    def supported_items(self):
        """Either model having real evidence counts as this hybrid having evidence."""
        return np.union1d(self._svd.supported_items, self._content.supported_items)

    def seen_items(self, user_id):
        """Delegates to the SVD model's train-time seen mask (same grid as content's)."""
        return self._svd.seen_items(user_id)

    def predict(self, user_id, movie_id):
        """Blended score for one (user, movie) pair, or None if unknown."""
        scores = self.score_all_items(user_id)
        if scores is None or movie_id not in scores.index:
            return None
        return scores.loc[movie_id]

    def recommend_top_n(self, user_id, n=10, exclude_seen=True):
        """
        Top-N by blended score. Same interface as SVDRecommender /
        ContentBasedRecommender on purpose — evaluate_ranking_at_ks() and
        friends don't need to change to evaluate this model, and neither
        will they when AutoRec is added later.
        """
        scores = self.score_all_items(user_id)
        if scores is None:
            return pd.Series(dtype=float)

        scores = scores.dropna()  # "switching" mode can leave items with no opinion at all

        if exclude_seen:
            seen = self.seen_items(user_id)
            scores = scores[~scores.index.isin(seen)]

        # kind="mergesort": stable sort, matching the convention used
        # everywhere else in this project (SVDRecommender, ContentBasedRecommender,
        # Step 6's popularity ranking) — ties resolve deterministically.
        return scores.sort_values(ascending=False, kind="mergesort").head(n)
