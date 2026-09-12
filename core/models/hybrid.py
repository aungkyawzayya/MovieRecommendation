"""
hybrid.py
Combines SVD (collaborative filtering) and content-based (plot-text) scores
into a single ranked list. Neither model alone is enough: SVD collapses to
the user-mean baseline for items with zero train ratings (Step 6's
cold-start evidence), and the content model only has plot text for 36% of
the catalogue (Step 5) — this class exists to cover what either one misses
on its own.

Also designed to NEST: a HybridRecommender can itself sit in either slot of
another HybridRecommender (e.g. combining SVD+AutoRec first, then blending
that with content — see notebook Step 11). That needs HybridRecommender to
satisfy the exact same contract SVDRecommender/AutoRecRecommender already
do (score_all_items(user_id, clip=...), item_ids, supported_items,
seen_items) — the two small additions below (the clip parameter and the
item_ids property) exist for that reason, not for anything used inside
this file itself.
"""

import numpy as np
import pandas as pd


class RawScoreAdapter:
    """
    Lets a rating-scale model (SVDRecommender, AutoRecRecommender) sit in a
    HybridRecommender's content_model slot.

    That slot is always called as score_all_items(user_id) — no clip
    argument — because ContentBasedRecommender's own similarity scores have
    no such parameter. SVD and AutoRec DO have one, defaulting to clip=True,
    so plugging either straight into the content slot would silently score
    on the clipped 0.5-5.0 scale right where z-scoring needs the raw one
    (the same reason the svd_model slot is always called with clip=False —
    see svd.py's docstring). This adapter forwards every call with
    clip=False pinned, so no caller can get that wrong by omission.

    Composition, not inheritance: this is not a subtype of the wrapped
    model, just a thin forwarding shim exposing the interface
    HybridRecommender's content-slot code actually calls.
    """

    def __init__(self, model):
        self._model = model

    def score_all_items(self, user_id):
        return self._model.score_all_items(user_id, clip=False)

    @property
    def supported_items(self):
        return self._model.supported_items

    @property
    def item_ids(self):
        return self._model.item_ids

    def seen_items(self, user_id):
        return self._model.seen_items(user_id)

    def predict(self, user_id, movie_id):
        return self._model.predict(user_id, movie_id)

    def recommend_top_n(self, user_id, n=10, exclude_seen=True):
        return self._model.recommend_top_n(user_id, n=n, exclude_seen=exclude_seen)


class HybridRecommender:
    # Hybrid score is a blended z-score, not a star rating — same guard
    # pattern as ContentBasedRecommender.produces_ratings, so
    # metrics.evaluate_model() refuses a meaningless RMSE against it.
    produces_ratings = False

    def __init__(self, svd_model, content_model, alpha=0.5, fallback="neutral", mode="weighted",
                 cache_components=False):
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

        # svd_z and content_z do NOT depend on alpha — only the blend does.
        # An alpha sweep therefore recomputes the same two z-score vectors
        # once per (alpha, user): 11 alphas x 598 users = 6,578 rebuilds of
        # work that only changes 598 times. Setting cache_components=True
        # memoizes them per user so a sweep can reuse one instance and just
        # reassign .alpha between runs. C# analogy: a private Dictionary
        # backing a lazily-computed property.
        # Off by default — the cache holds 2 float64 Series per user
        # (~95 MB for all 610 users), which is only worth paying in a sweep.
        self._cache_components = cache_components
        self._component_cache = {}

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

    def _components(self, user_id):
        """
        The alpha-INDEPENDENT half of the blend: (svd_z, content_z_full).
        Split out from score_all_items() so an alpha sweep can compute it
        once per user instead of once per (alpha, user) — see
        cache_components in __init__.

        NOTE on scope: both z-scores are computed over the FULL catalogue
        for this user, and already-rated items are removed later, in
        recommend_top_n(). For a single model that choice cannot change the
        ranking (z is a monotonic transform), but in a BLEND it shifts what
        alpha means, since each model's std comes from the same full set
        rather than the post-exclusion candidate set. alpha is selected on
        validation, so the tuning absorbs it — but the report's Methods
        section should state which set the z-scores are taken over.
        """
        if self._cache_components and user_id in self._component_cache:
            return self._component_cache[user_id]

        svd_scores = self._svd.score_all_items(user_id, clip=False)
        if svd_scores is None:
            result = (None, None)   # unknown user (cold-start)
        else:
            result = (
                self._zscore(svd_scores),
                self._content_z_full(user_id, svd_scores.index),
            )

        if self._cache_components:
            self._component_cache[user_id] = result
        return result

    def score_all_items(self, user_id, clip=True):
        """
        Blended score for every item SVD has an opinion on. SVD always
        scores the full catalogue (even cold items, via the user-mean
        fallback — see SVDRecommender.score_all_items's clip=False note),
        so its index is the frame everything else lines up against.
        clip=False: ranking needs the raw distribution, not ratings pinned
        to the 5.0 ceiling — see svd.py's docstring on why clipping would
        flatten exactly the differences z-scoring depends on.

        clip IS ACCEPTED BUT IGNORED: the blended output is a z-score, not
        a 0.5-5.0 rating, so "clip to the rating range" has no meaning here.
        The parameter exists purely so a HybridRecommender can be nested
        inside another one's svd_model slot, which always calls
        score_all_items(user_id, clip=False) — without accepting the
        keyword this would raise TypeError the moment nesting was tried.
        """
        svd_z, content_z_full = self._components(user_id)
        if svd_z is None:
            return None  # unknown user (cold-start) — caller must handle this

        has_content = content_z_full.notna()

        if self.mode == "switching":
            has_train_ratings = svd_z.index.isin(self._svd.supported_items)
            blended = pd.Series(np.nan, index=svd_z.index)
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

    @property
    def item_ids(self):
        """
        Every movieId this hybrid can score over — delegates to the
        svd_model slot's own grid, since that slot always scores the full
        catalogue (see score_all_items's docstring). Same purpose as
        SVDRecommender.item_ids / AutoRecRecommender.item_ids: lets this
        object sit in ANOTHER HybridRecommender's svd_model slot, whose
        rankable_items property (weighted mode) reads self._svd.item_ids.
        """
        return self._svd.item_ids

    @property
    def rankable_items(self):
        """
        Items this model can actually place in a ranked list — the honest
        denominator for a catalogue-coverage figure.

        In "weighted" mode with fallback="neutral" every item gets a score
        (a text-less item simply gets alpha*svd_z), so this is the whole
        catalogue. In "switching" mode an item with neither train ratings
        nor plot text gets NaN and is dropped before sorting, so quoting
        coverage against the full catalogue would understate it.
        """
        if self.mode == "switching":
            return self.supported_items
        return self._svd.item_ids

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
        did they when AutoRec was added, or when a HybridRecommender was
        nested inside another one.
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
