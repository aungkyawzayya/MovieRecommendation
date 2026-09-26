"""
qualitative_snapshot.py  --  ONE-OFF, safe to delete after running.

Produces the qualitative side-by-side the report is missing: for a few users,
the top-5 each model actually recommends. No model is retrained beyond SVD and
Content (both seconds); the Nested Hybrid's list is read from the export the
batch job already wrote, so PyTorch is never imported.

Run from the repo root with the project venv active:
    python qualitative_snapshot.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.data.preprocess import MovieDataPreprocessor
from core.models.svd import SVDRecommender
from core.models.content import ContentBasedRecommender

CANDIDATE_USERS = [1, 23, 414, 599]   # a few profiles to choose between
TOP_N = 5

pipe = (MovieDataPreprocessor()
        .load_raw_data().build_movies_full()
        .build_text_corpus().build_user_item_matrix())
ds = pipe.split()
titles = pipe.movies_full.set_index("movieId")["title"]

# center="user" is the ranking variant (svd.py: item bias flattens ranking)
svd = SVDRecommender(n_factors=10, center="user", damping=5).fit(ds.trainval_matrix)
content = ContentBasedRecommender().fit(ds.trainval_matrix, pipe.text_corpus)

export = json.loads((ROOT / "data" / "recommendations_export.json").read_text())
nested = export["recommendations"]

# how many train+val ratings each film has -- shows how "popular" a pick is
support = ds.trainval_matrix.notna().sum(axis=0)

def name(mid):
    return titles.get(mid, f"movie {mid}")

for uid in CANDIDATE_USERS:
    if str(uid) not in nested:
        continue
    print("=" * 78)
    print(f"USER {uid}")
    seen = svd.seen_items(uid)
    print(f"  rated {len(seen)} films in train+val")

    own = (ds.trainval_matrix.loc[uid].dropna().sort_values(ascending=False).head(4))
    print("  their own highest-rated:")
    for mid, r in own.items():
        print(f"     {r:.1f}  {name(mid)}")

    print(f"\n  {'SVD':<38} {'Content (TF-IDF)':<38}")
    s_top = svd.recommend_top_n(uid, n=TOP_N, exclude_seen=True).index.tolist()
    c_top = content.recommend_top_n(uid, n=TOP_N, exclude_seen=True,
                                    seen_movie_ids=seen).index.tolist()
    n_top = [r["movieId"] for r in nested[str(uid)][:TOP_N]]
    for a, b in zip(s_top, c_top):
        print(f"   {name(a)[:34]:<36} {support.get(a,0):>4}   {name(b)[:34]:<36} {support.get(b,0):>4}")

    print(f"\n  {'Nested Hybrid (deployed)':<38}")
    for m in n_top:
        print(f"   {name(m)[:34]:<36} {support.get(m,0):>4}")

    print(f"\n  median train+val ratings per recommended film:")
    print(f"     SVD      {pd.Series([support.get(m,0) for m in s_top]).median():>6.0f}")
    print(f"     Content  {pd.Series([support.get(m,0) for m in c_top]).median():>6.0f}")
    print(f"     Nested   {pd.Series([support.get(m,0) for m in n_top]).median():>6.0f}")
    print()
