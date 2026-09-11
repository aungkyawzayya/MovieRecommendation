"""
preprocess.py
Encapsulates the MovieLens + TMDB preprocessing pipeline as a class —
DataFrames live as private fields (self._ratings, etc.) instead of
being passed around as loose function parameters.
Similar to a C# class: private fields + public methods.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"
OVERVIEW_PATH = PROJECT_ROOT / "data" / "overview_plot.csv"

# COMP813 convention (matches the lecturer's Assignment 1 seeding): seed
# every random split with the student ID, so the split is reproducible
# AND identifiably "this student's" run if anyone re-executes the notebook.
STUDENT_ID = 24265298


@dataclass
class DataSplit:
    """
    Bundles what a 3-way train/validation/test split produces. A plain
    tuple return here would be easy to unpack in the wrong order — a
    dataclass makes each piece self-naming, like a C# result/DTO class.
    """
    train_matrix: pd.DataFrame       # 60% — fit models here while TUNING
    trainval_matrix: pd.DataFrame    # 80% (train+val) — final refit, once tuning is done
    train_df: pd.DataFrame
    val_df: pd.DataFrame             # 20% — pick hyperparameters (k, alpha, ...) here
    test_df: pd.DataFrame            # 20% — touch ONCE, only to report final numbers


class MovieDataPreprocessor:
    """
    Python has no 'private' keyword like C# — the convention is a
    leading underscore (self._ratings) to signal "internal use only,
    don't touch from outside the class".
    """

    def __init__(self):
        # Empty until load_raw_data() runs — like fields set in a C# constructor.
        self._ratings = None
        self._movies = None
        self._links = None
        self._overview = None

        # Public results other code (models, notebooks) is meant to read.
        self.movies_full = None
        self.text_corpus = None
        self.user_item_matrix = None

    def load_raw_data(self):
        """Loads the 4 raw CSVs into private fields."""
        self._ratings = pd.read_csv(DATA_DIR / "ratings.csv")
        self._movies = pd.read_csv(DATA_DIR / "movies.csv")
        self._links = pd.read_csv(DATA_DIR / "links.csv")
        self._overview = pd.read_csv(OVERVIEW_PATH)
        return self  # enables method chaining, like a C# fluent builder

    def build_movies_full(self):
        """Merges movies -> links (tmdbId) -> overview (plot text)."""
        movies_links = self._movies.merge(
            self._links[["movieId", "tmdbId"]], on="movieId", how="left"
        )
        movies_links["tmdbId"] = movies_links["tmdbId"].astype("Int64")

        overview = self._overview.copy()  # don't mutate the caller's data
        overview["tmdbId"] = overview["tmdbId"].astype("Int64")

        self.movies_full = movies_links.merge(
            overview[["tmdbId", "overview"]], on="tmdbId", how="left"
        )
        return self

    def build_text_corpus(self):
        """Filters movies_full down to rows with usable plot text."""
        self.text_corpus = self.movies_full[
            self.movies_full["overview"].notnull()
        ][["movieId", "title", "genres", "overview"]].reset_index(drop=True)
        return self

    def build_user_item_matrix(self):
        """Pivots ratings into the (610 x 9724) User-Item matrix."""
        self.user_item_matrix = self._ratings.pivot(
            index="userId", columns="movieId", values="rating"
        )
        return self

    @property
    def ratings(self):
        """
        Public read-only access to the raw ratings DataFrame.
        C# analogy: public DataFrame Ratings => _ratings;
        (auto-property with a private backing field, read-only from outside)
        """
        return self._ratings

    def _pivot(self, ratings_df):
        """
        Pivot a long-form ratings DataFrame into the FULL 610 x 9724 grid —
        same user/item index every time, regardless of which subset of rows
        is passed in, so train/val/test matrices always line up column-for-
        column with each other and with SVDRecommender/ContentBasedRecommender.
        """
        user_index = self._ratings["userId"].sort_values().unique()
        item_index = self._ratings["movieId"].sort_values().unique()
        return (
            ratings_df.pivot(index="userId", columns="movieId", values="rating")
            .reindex(index=user_index, columns=item_index)
        )

    def split(self, val_frac=0.2, test_frac=0.2, seed=STUDENT_ID):
        """
        Per-user 3-way split — train_test_split() below is a thin wrapper
        around this with val_frac=0.0. The test slice is always
        perm[:n_test], computed BEFORE the validation slice is carved out,
        so split(val_frac=0.2, test_frac=0.2, seed=S) and
        split(val_frac=0.0, test_frac=0.2, seed=S) select the identical
        test rows — adding validation never disturbs the test set.

        Use .train_matrix while TUNING hyperparameters (k, alpha, ...)
        against .val_df. Once a choice is locked in, refit on
        .trainval_matrix and report metrics against .test_df exactly once —
        never use test performance to make a decision.
        """
        rng = np.random.default_rng(seed)
        train_parts, val_parts, test_parts = [], [], []

        for _, group in self._ratings.groupby("userId"):
            n = len(group)
            perm = rng.permutation(n)
            # Cap n_test at n-1 so at least one rating is always left for train.
            # Without the cap, a user with a single rating got n_test=1 and
            # n_val=-1, and perm[n_test + n_val:] == perm[0:] handed that SAME
            # row to BOTH test and train — a silent train/test leak. Unreachable
            # on ml-latest-small (min 20 ratings/user) but one user filter away.
            n_test = min(max(1, int(n * test_frac)), n - 1)
            # val_frac=0.0 (train_test_split's case) -> n_val=0, no validation
            # slice at all, reproducing the old 2-way behaviour exactly.
            n_val = max(1, int(n * val_frac)) if val_frac > 0 else 0
            n_val = max(0, min(n_val, n - n_test - 1))  # always leave >=1 train rating

            test_parts.append(group.iloc[perm[:n_test]])
            val_parts.append(group.iloc[perm[n_test:n_test + n_val]])
            train_parts.append(group.iloc[perm[n_test + n_val:]])

        train_df = pd.concat(train_parts).reset_index(drop=True)
        val_df = pd.concat(val_parts).reset_index(drop=True)
        test_df = pd.concat(test_parts).reset_index(drop=True)

        return DataSplit(
            train_matrix=self._pivot(train_df),
            trainval_matrix=self._pivot(pd.concat([train_df, val_df])),
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    def train_test_split(self, test_frac=0.2, seed=STUDENT_ID):
        """
        Kept for existing notebook cells written against this 2-way
        signature. A thin wrapper around split(val_frac=0.0, ...) — no
        validation slice, same behaviour as the original implementation.

        Returns:
            train_matrix: pivoted on the FULL dataset's user/movie grid.
            test_df: held-out ratings in long form (userId, movieId, rating)
        """
        result = self.split(val_frac=0.0, test_frac=test_frac, seed=seed)
        return result.train_matrix, result.test_df


# Quick self-test when this file is run directly (python preprocess.py)
# rather than imported — same idea as a C# console app's Main() method
# guarded so a class library doesn't auto-run on import.
if __name__ == "__main__":
    pipeline = (
        MovieDataPreprocessor()
        .load_raw_data()
        .build_movies_full()
        .build_text_corpus()
        .build_user_item_matrix()
    )
    print("User-item matrix:", pipeline.user_item_matrix.shape)
    print("Text corpus:", pipeline.text_corpus.shape)