"""
content.py
Content-based recommender — uses each movie's plot text (TF-IDF + cosine
similarity), not other users' ratings. This is what fills the gap SVD
can't: a movie with ZERO train ratings still has plot text, so this
model can still score it. SVD collapses to the user-mean baseline for
those items — that's the exact problem this class exists to solve.

COVERAGE, measured: overview_plot.csv was expanded from 4,800 to 10,884
rows (scripts/fetch_overviews.py), taking text coverage of the rated
catalogue from 3,536/9,724 (36.4%) to 9,603/9,724 (98.8%), and of items
with zero train ratings from 18% to 98%. The 121 still uncovered are
movies TMDB itself has no synopsis for (113) or that carry no tmdbId at
all (8).

What that bought, measured rather than assumed: this model's cold-start
reach roughly tripled (163 -> 481 of 6,100 recommended slots) and its
catalogue coverage rose 17.8% -> 24.9%, but its ranking quality FELL
(NDCG@10 0.0469 -> 0.0359). Cold items make up only 2.92% of the relevant
held-out ratings, so every obscure item promoted into a top-10 displaces a
popular one with a far higher chance of being a hit. That is the
accuracy-coverage trade-off, not a regression to fix.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    # score_all_items() returns COSINE SIMILARITY (roughly 0-1), not a star
    # rating. metrics.evaluate_model() checks this flag and refuses to compute
    # RMSE/MAE against a 0.5-5.0 rating column — without the guard it happily
    # returned "RMSE 3.71", a meaningless number that looks like a real result.
    produces_ratings = False

    def __init__(self, use_genres=False):
        # use_genres=False by default — verified in an earlier design
        # consult that adding genre tokens HURTS top-N ranking quality,
        # even though it slightly helps rating-order correlation.
        self.use_genres = use_genres

        self._vectorizer = None
        self._tfidf_matrix = None      # (n_text_movies, n_terms) sparse
        self._movie_ids = None         # FULL rated-movie index (9,724) — same
                                        # grid as SVDRecommender, so the hybrid
                                        # combiner can line the two up directly
        self._text_movie_ids = None    # subset of movie_ids that HAVE text
        self._user_ids = None
        self._user_profiles = None     # (n_users, n_terms) — one row per user
        self._seen_mask = None         # which (user, movie) pairs were in train


    def fit(self, train_matrix, text_corpus):
        """
        train_matrix: same shape/grid as SVDRecommender.fit() gets — rows are
                      users, columns are ALL 9,724 rated movies, NaN = unrated.
        text_corpus:  preprocessor's text_corpus DataFrame (movieId, title,
                      genres, overview) — already filtered to movies WITH text.
        """
        self._movie_ids = train_matrix.columns.to_numpy()
        self._user_ids = train_matrix.index.to_numpy()
        # Capture the train-time seen mask here, exactly like SVDRecommender
        # does. recommend_top_n() used to depend on the caller passing the
        # seen set in, and silently recommended nothing but already-rated
        # movies when they didn't.
        self._seen_mask = train_matrix.notna().to_numpy()

        # text_corpus can include movies nobody ever rated (they exist in
        # movies.csv but never appear in ratings.csv) — those have no
        # column in train_matrix, so drop them before anything else touches
        # this DataFrame. Without this, a handful of never-rated movies
        # with text silently crash the fit().
        text_corpus = text_corpus[text_corpus["movieId"].isin(self._movie_ids)].reset_index(drop=True)
        self._text_movie_ids = text_corpus["movieId"].to_numpy()

        # Build the raw text each movie is vectorized from. Genres in
        # movies.csv look like "Adventure|Children|Fantasy" — replace "|"
        # with a space so they tokenize as separate words, only used if
        # use_genres=True.
        if self.use_genres:
            genre_text = text_corpus["genres"].str.replace("|", " ", regex=False)
            corpus_text = text_corpus["overview"] + " " + genre_text
        else:
            corpus_text = text_corpus["overview"]

        # ngram_range=(1,1): unigrams only. Bigrams were tried and added
        # noise on this short-text corpus (min_df=2 lets too few real
        # bigrams survive to be useful).
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 1), min_df=2, sublinear_tf=False
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus_text)

        # Build each user's taste profile as a WEIGHTED SUM of the TF-IDF
        # vectors of movies they rated — weight = rating minus their OWN
        # mean ("centered"). A movie rated BELOW a user's own average gets
        # a NEGATIVE weight, actively pushing the profile away from it —
        # this beat both a binary-positive and a raw-rating-weighted
        # profile when compared directly on this dataset.
        user_means = train_matrix.mean(axis=1)  # pandas .mean skips NaN by default
        centered = train_matrix[self._text_movie_ids].sub(user_means, axis=0)
        centered = centered.fillna(0.0).to_numpy()  # unrated -> 0 contribution

        self._user_profiles = centered @ self._tfidf_matrix
        return self

    def score_all_items(self, user_id):
        """
        Content-similarity score for EVERY movie, for one user — a pandas
        Series over the FULL movie index (same shape as SVDRecommender's
        score_all_items), so the hybrid combiner can line the two up
        directly. Text-less movies get NaN, not 0 — 0 would silently claim
        "this user actively dislikes this movie", which we have no basis
        for. NaN honestly says "this model has no opinion here".
        """
        user_idx = np.where(self._user_ids == user_id)[0]
        if len(user_idx) == 0:
            return None  # unknown user (cold-start) — caller must handle this
        user_idx = user_idx[0]

        profile = self._user_profiles[user_idx]

        # Guard: a user whose every rated movie landed exactly at their own
        # mean (centered weight = 0 everywhere) gets an all-zero profile.
        # cosine_similarity would divide by a zero norm — return "no
        # opinion" (all-NaN) instead of letting that blow up or silently
        # return all-zero similarity (which looks like real signal but isn't).
        if not np.any(profile):
            return pd.Series(np.nan, index=self._movie_ids)

        sims = cosine_similarity(profile.reshape(1, -1), self._tfidf_matrix).flatten()

        result = pd.Series(np.nan, index=self._movie_ids, dtype=float)
        result.loc[self._text_movie_ids] = sims
        return result

    @property
    def supported_items(self):
        """movieIds this model has evidence for — i.e. the ones with plot text."""
        return self._text_movie_ids

    def seen_items(self, user_id):
        """movieIds this user rated in the matrix passed to fit()."""
        user_idx = np.where(self._user_ids == user_id)[0]
        if len(user_idx) == 0:
            return np.array([], dtype=self._movie_ids.dtype)
        return self._movie_ids[self._seen_mask[user_idx[0]]]

    def predict(self, user_id, movie_id):
        """Content-similarity score for one (user, movie) pair, or None."""
        scores = self.score_all_items(user_id)
        if scores is None or movie_id not in scores.index:
            return None
        return scores.loc[movie_id]

    def recommend_top_n(self, user_id, n=10, exclude_seen=True, seen_movie_ids=None):
        """
        Top-N by content similarity, excluding what the user already rated.

        BUG FIX: the old signature was
            (..., exclude_seen=True, seen_mask=None, seen_movie_ids=None)
        with the body guarded by `if exclude_seen and seen_movie_ids is not
        None`. seen_movie_ids defaulted to None, so the documented default
        exclude_seen=True did NOTHING. Measured over the first 100 users:
        871 of 990 recommended items (88%) were movies the user had already
        rated, and 61 of those users got a top-10 that was 100% already-seen.
        Precision@K on that output is meaningless but looks excellent.

        Now the train-time mask captured in fit() is used by default, the same
        way SVDRecommender.recommend_top_n() does it. seen_movie_ids stays as
        an explicit override for callers that want to exclude a different set
        (e.g. the hybrid, which may pool train+val); the unused seen_mask
        parameter is gone.
        """
        scores = self.score_all_items(user_id)
        if scores is None:
            return pd.Series(dtype=float)

        scores = scores.dropna()  # drop text-less movies — nothing to rank them on

        if exclude_seen:
            if seen_movie_ids is None:
                seen_movie_ids = self.seen_items(user_id)
            scores = scores[~scores.index.isin(seen_movie_ids)]

        return scores.sort_values(ascending=False, kind="mergesort").head(n)