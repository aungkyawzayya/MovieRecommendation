"""
export_recommendations.py
Batch-scoring / offline export step for the deployed Nested Hybrid model
(notebook Step 12: SVD + AutoRec + Content composition, alpha1=0.4,
alpha2=0.6 — the best of all seven models measured in this project, test
NDCG@10=0.1643).

WHY THIS EXISTS AS ITS OWN SCRIPT: nobody in a real deployment runs a
Jupyter notebook to get a recommendation, and a request-serving process
has no business importing torch/sklearn and re-fitting a model on every
call either. The standard split in industry is: a batch job (this script)
periodically fits/refits the model and writes its predictions to a plain
lookup table, and the serving layer (api/main.py) is a thin, fast reader
of that table with zero training dependency at request time. Re-run this
whenever the underlying data or a notebook hyperparameter choice changes —
api/main.py only ever needs the JSON it writes.

All hyperparameters below are hardcoded to the values notebooks/01_eda.ipynb
already selected on VALIDATION (Steps 2/4/5/7/9/11) — this script does not
re-run any grid search, exactly like the notebook's own "deploy" cells
(Steps 3/4/5/8/10/12) don't either.

Run from the repo root, with the project's real venv active (needs torch):
    python scripts/export_recommendations.py
"""
import json
import time
from pathlib import Path

from core.data.preprocess import MovieDataPreprocessor, STUDENT_ID
from core.models.svd import SVDRecommender
from core.models.content import ContentBasedRecommender
from core.models.autorec import AutoRecRecommender
from core.models.hybrid import HybridRecommender, RawScoreAdapter

# More than any evaluation cutoff the notebook used (Ks=[5,10,20]), so the
# web demo can show a deeper list than was ever needed for a metric.
TOP_N = 20


def main():
    t0 = time.time()

    print("Loading and preprocessing data...")
    pre = (
        MovieDataPreprocessor()
        .load_raw_data()
        .build_movies_full()
        .build_text_corpus()
        .build_user_item_matrix()
    )
    ds = pre.split(val_frac=0.2, test_frac=0.2, seed=STUDENT_ID)

    # --- Step 4: SVD, ranking-selected (k=10, damping=5, center="user") ---
    print("Fitting SVD (k=10, damping=5, center='user')...")
    rank_model = SVDRecommender(n_factors=10, damping=5, center="user").fit(ds.trainval_matrix)

    # --- Step 5: Content, validation-selected (use_genres=False) ---
    print("Fitting Content model (use_genres=False)...")
    content_model = ContentBasedRecommender(use_genres=False).fit(ds.trainval_matrix, pre.text_corpus)

    # --- Step 10: AutoRec, NDCG-selected (k=25, weight_decay=1e-2), refit
    # on trainval for exactly best_epoch+1=414 epochs, no val_df (nothing
    # left to early-stop on once trainval IS train+val) ---
    print("Fitting AutoRec (hidden_dim=25, weight_decay=1e-2, epochs=414) — the slow step...")
    autorec_deploy = AutoRecRecommender(
        hidden_dim=25, weight_decay=1e-2, lr=1e-2, epochs=414, patience=1, seed=STUDENT_ID,
    ).fit(ds.trainval_matrix)

    # --- Step 12: Nested Hybrid at the interior alphas (a1=0.4, a2=0.6) ---
    print("Composing Nested Hybrid (a1=0.4, a2=0.6)...")
    nested_inner = HybridRecommender(rank_model, RawScoreAdapter(autorec_deploy), alpha=0.4)
    nested_hybrid = HybridRecommender(nested_inner, content_model, alpha=0.6, fallback="neutral")

    # --- Batch-score every user in the dataset ---
    movie_lookup = pre.movies_full.set_index("movieId")[["title", "genres"]]
    all_user_ids = sorted(pre.ratings["userId"].unique())
    print(f"Scoring {len(all_user_ids)} users, top {TOP_N} each...")

    recommendations = {}
    for uid in all_user_ids:
        top = nested_hybrid.recommend_top_n(int(uid), n=TOP_N, exclude_seen=True)
        rows = []
        for rank, (movie_id, score) in enumerate(top.items(), start=1):
            meta = movie_lookup.loc[movie_id] if movie_id in movie_lookup.index else None
            rows.append({
                "rank": rank,
                "movieId": int(movie_id),
                "title": str(meta["title"]) if meta is not None else f"Movie {movie_id}",
                "genres": str(meta["genres"]) if meta is not None else "",
                "score": round(float(score), 4),
            })
        recommendations[str(int(uid))] = rows

    out = {
        "model": "Nested Hybrid (a1=0.4, a2=0.6)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_users": len(recommendations),
        "top_n_per_user": TOP_N,
        "recommendations": recommendations,
    }

    out_path = Path(__file__).resolve().parents[1] / "data" / "recommendations_export.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
