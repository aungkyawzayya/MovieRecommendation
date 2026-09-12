"""
autorec_sweep.py
Diagnostic: is AutoRec's hyperparameter grid measuring CAPACITY, or just
measuring how many gradient steps each config happened to get before early
stopping fired?

Runs the same k sweep twice — once full-batch (one update per epoch, the
original behaviour) and once mini-batched — and prints, for each config,
how many epochs ran and which epoch won. If the full-batch column shows
best_epoch pinned near 0 while mini-batch does not, the original grid was
reporting optimization noise and needs re-running before its numbers go in
the report.

Run from the repo root with the venv active:
    python scripts/autorec_sweep.py
"""

import sys
import time
from pathlib import Path

# Make `core` importable when run as scripts/autorec_sweep.py from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from core.data.preprocess import MovieDataPreprocessor, STUDENT_ID
from core.models.autorec import AutoRecRecommender
from core.evaluation.metrics import evaluate_model
from core.evaluation.ranking import build_relevant_items, evaluate_ranking

# Keep the grid small on purpose — this is a diagnostic, not the real sweep.
# One weight_decay, the k values that behaved most strangely in the notebook.
KS = [10, 50, 200, 500]
WEIGHT_DECAY = 1e-4
LR = 1e-2
EPOCHS = 80
PATIENCE = 10
BATCH_SIZES = [None, 256]      # None = full batch (original behaviour)


def main():
    print("Loading data ...", flush=True)
    pipeline = (
        MovieDataPreprocessor()
        .load_raw_data()
        .build_movies_full()
        .build_text_corpus()
        .build_user_item_matrix()
    )
    ds = pipeline.split()
    relevant_val = build_relevant_items(ds.val_df, threshold=4.0)
    print(f"train {len(ds.train_df)} / val {len(ds.val_df)} rows\n", flush=True)

    rows = []
    for batch_size in BATCH_SIZES:
        label = "full-batch" if batch_size is None else f"batch={batch_size}"
        for k in KS:
            t0 = time.time()
            model = AutoRecRecommender(
                hidden_dim=k,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                epochs=EPOCHS,
                patience=PATIENCE,
                batch_size=batch_size,
                seed=STUDENT_ID,
            ).fit(ds.train_matrix, val_df=ds.val_df)

            val_rmse = evaluate_model(model, ds.val_df).metrics["RMSE"]
            recommend_fn = lambda uid, m=model: m.recommend_top_n(
                uid, n=10, exclude_seen=True
            ).index.tolist()
            val_ndcg = evaluate_ranking(recommend_fn, relevant_val, k=10)["NDCG@10"]

            epochs_run = len(model.train_losses_)
            # updates per epoch: 1 when full batch, ceil(n_items / batch) otherwise
            n_items = ds.train_matrix.shape[1]
            per_epoch = 1 if batch_size is None else -(-n_items // batch_size)

            rows.append({
                "mode": label,
                "k": k,
                "val_RMSE": val_rmse,
                "val_NDCG@10": val_ndcg,
                "epochs_run": epochs_run,
                "best_epoch": model.best_epoch_,
                "updates": epochs_run * per_epoch,
                "secs": time.time() - t0,
            })
            print(
                f"  {label:11s} k={k:4d}  RMSE={val_rmse:.4f}  NDCG={val_ndcg:.4f}  "
                f"epochs={epochs_run:3d}  best={model.best_epoch_:3d}  "
                f"updates={rows[-1]['updates']:5d}  {rows[-1]['secs']:5.1f}s",
                flush=True,
            )
        print(flush=True)

    grid = pd.DataFrame(rows)
    print("=" * 78)
    print(grid.to_string(index=False, formatters={
        "val_RMSE": "{:.4f}".format,
        "val_NDCG@10": "{:.4f}".format,
        "secs": "{:.1f}".format,
    }))
    print("=" * 78)

    # The diagnosis, stated rather than left to the reader.
    for label in grid["mode"].unique():
        sub = grid[grid["mode"] == label]
        stopped_early = sub[sub["epochs_run"] < EPOCHS]
        starved = sub[sub["best_epoch"] < PATIENCE * 2]
        print(f"\n{label}:")
        print(f"  best val_RMSE : {sub['val_RMSE'].min():.4f} (k={sub.loc[sub['val_RMSE'].idxmin(), 'k']})")
        print(f"  configs that early-stopped      : {len(stopped_early)}/{len(sub)}")
        print(f"  configs whose best epoch was <{PATIENCE * 2:<3d}: {len(starved)}/{len(sub)}"
              f"   <-- these never trained")

    print("\nRead it like this: a config whose best epoch is in single digits did not")
    print("lose on capacity, it lost on not having been trained yet. If that column")
    print("empties out under mini-batching, the original grid's k ordering was an")
    print("artefact and the full 6x4 sweep needs re-running before the report uses it.")


if __name__ == "__main__":
    main()
