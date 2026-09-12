"""
autorec_sweep2.py
Follow-up diagnostic. Sweep 1 showed two things:

  full-batch k=200 : best_epoch=67 of epochs_run=78, cap=80
                     -> still improving when the epoch cap stopped it
  batch=256        : best_epoch 1-3 at lr=1e-2
                     -> 38x more updates per epoch at the same lr overshoots
                        immediately; the problem was never the batch size,
                        it was the step size

So the real unknowns are LEARNING RATE and EPOCH BUDGET, not batching. This
sweeps both, with a budget big enough that a config can actually converge,
and reports whether each run stopped because it peaked or because it ran out
of epochs — the distinction sweep 1 could not make.

Run from the repo root with the venv active:
    python scripts/autorec_sweep2.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from core.data.preprocess import MovieDataPreprocessor, STUDENT_ID
from core.models.autorec import AutoRecRecommender
from core.evaluation.metrics import evaluate_model
from core.evaluation.ranking import build_relevant_items, evaluate_ranking

KS = [10, 200]
WEIGHT_DECAY = 1e-4

# (batch_size, lr, epochs, patience)
# Full batch: 1 update/epoch, so it needs thousands of epochs and a patience
# measured in hundreds. Mini-batch: ~38 updates/epoch, so a comparable number
# of UPDATES is reached in ~1/38 the epochs — and the lr has to drop to match.
CONFIGS = [
    (None, 1e-2, 2000, 200),
    (None, 3e-3, 2000, 200),
    (256,  1e-3,  200,  20),
    (256,  3e-4,  200,  20),
]


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
    n_items = ds.train_matrix.shape[1]
    print(f"train {len(ds.train_df)} / val {len(ds.val_df)} rows\n", flush=True)

    rows = []
    for batch_size, lr, epochs, patience in CONFIGS:
        label = "full-batch" if batch_size is None else f"batch={batch_size}"
        per_epoch = 1 if batch_size is None else -(-n_items // batch_size)
        for k in KS:
            t0 = time.time()
            model = AutoRecRecommender(
                hidden_dim=k,
                lr=lr,
                weight_decay=WEIGHT_DECAY,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                seed=STUDENT_ID,
            ).fit(ds.train_matrix, val_df=ds.val_df)

            val_rmse = evaluate_model(model, ds.val_df).metrics["RMSE"]
            recommend_fn = lambda uid, m=model: m.recommend_top_n(
                uid, n=10, exclude_seen=True
            ).index.tolist()
            val_ndcg = evaluate_ranking(recommend_fn, relevant_val, k=10)["NDCG@10"]

            epochs_run = len(model.train_losses_)
            # Why did this run stop? "cap" means it was still improving and the
            # epoch budget cut it off — that config's number is a floor, not a
            # result. "peak" means validation genuinely stopped improving.
            hit_cap = epochs_run >= epochs
            stop_reason = "cap" if hit_cap else "peak"

            rows.append({
                "mode": label,
                "lr": lr,
                "k": k,
                "val_RMSE": val_rmse,
                "val_NDCG@10": val_ndcg,
                "epochs_run": epochs_run,
                "best_epoch": model.best_epoch_,
                "updates": epochs_run * per_epoch,
                "stopped_by": stop_reason,
                "secs": time.time() - t0,
            })
            print(
                f"  {label:11s} lr={lr:<7.0e} k={k:4d}  RMSE={val_rmse:.4f}  "
                f"NDCG={val_ndcg:.4f}  epochs={epochs_run:5d}/{epochs}  "
                f"best={model.best_epoch_:5d}  updates={rows[-1]['updates']:6d}  "
                f"[{stop_reason}]  {rows[-1]['secs']:5.1f}s",
                flush=True,
            )
        print(flush=True)

    grid = pd.DataFrame(rows)
    print("=" * 96)
    print(grid.to_string(index=False, formatters={
        "val_RMSE": "{:.4f}".format,
        "val_NDCG@10": "{:.4f}".format,
        "lr": "{:.0e}".format,
        "secs": "{:.1f}".format,
    }))
    print("=" * 96)

    best = grid.loc[grid["val_RMSE"].idxmin()]
    print(f"\nBest validation RMSE here : {best['val_RMSE']:.4f}"
          f"  ({best['mode']}, lr={best['lr']:.0e}, k={best['k']})")
    print(f"Sweep 1's best was        : 0.8616  (full-batch, lr=1e-2, k=200, epochs capped at 80)")

    capped = grid[grid["stopped_by"] == "cap"]
    if len(capped):
        print(f"\n{len(capped)}/{len(grid)} configs were still improving when the epoch budget ran out:")
        for _, r in capped.iterrows():
            print(f"  {r['mode']}, lr={r['lr']:.0e}, k={r['k']} -> raise epochs and re-run this one")
    else:
        print("\nEvery config stopped on its own validation peak, not on the epoch cap —")
        print("the budget is finally big enough for the numbers to mean what they say.")

    print("\nOnce one (mode, lr) setting converges cleanly for both k values, THAT is the")
    print("setting to run the full 6x4 k x weight_decay grid at in notebook Step 9.")


if __name__ == "__main__":
    main()
