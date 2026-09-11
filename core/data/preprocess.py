"""
preprocess.py
Encapsulates the MovieLens + TMDB preprocessing pipeline as a class —
DataFrames live as private fields (self._ratings, etc.) instead of
being passed around as loose function parameters.
Similar to a C# class: private fields + public methods.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"
OVERVIEW_PATH = PROJECT_ROOT / "data" / "overview_plot.csv"


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