"""
seed_stability.py
How much of the reported comparison is the models, and how much is the split?

WHY THIS EXISTS: every number in this project comes from ONE 60/20/20 split,
seeded with the student ID as the COMP813 convention requires. That makes the
results reproducible, but reproducible is not the same as stable: a different
split would move every figure, and where two models sit 0.003 apart the
ordering between them could simply flip. Step 11 already showed this happening
inside the project - the degenerate corner beat the best interior blend by
0.0006 on validation and then LOST to it on test by 0.0211.

So this script re-runs the final seven-model comparison at several splits and
reports, per model, how far each metric moves - and, more usefully, WHICH
CLAIMS in the report survive every split and which do not.

WHAT IT DOES NOT DO, stated plainly because it bounds the conclusion:

  * It does not re-run any hyperparameter search. k, damping, hidden_dim,
    weight_decay, the epoch budget and both alphas stay fixed at the values
    the notebook selected on the STUDENT-ID validation split. So this
    measures SPLIT variance with the model held fixed - the dominant term,
    and a well-posed question on its own - not selection variance.
  * Because those hyperparameters were chosen using the student-ID validation
    set, and some of those ratings land in ANOTHER seed's test set, there is
    mild optimistic leakage at the non-student-ID seeds. The effect is small
    (the choices are coarse) but it is real, and the report should say so
    rather than present these as clean held-out numbers.
  * The primary reported results remain the student-ID ones. This is a
    robustness check appended to them, not a replacement for them.

The seeds are fixed in the SEEDS constant below and were chosen before any of
this was run, so there is no room to pick the split that flatters the result.

Run from the repo root with the venv active (needs torch). Roughly 4-8 minutes
per seed, almost all of it AutoRec:

    python scripts/seed_stability.py
    python scripts/seed_stability.py --seeds 24265298,1,42
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data.preprocess import MovieDataPreprocessor, STUDENT_ID
from core.models.svd import SVDRecommender
from core.models.content import ContentBasedRecommender
from core.models.autorec import AutoRecRecommender
from core.models.hybrid import HybridRecommender, RawScoreAdapter
from core.evaluation.metrics import evaluate_model
from core.evaluation.ranking import (
    build_relevant_items,
    evaluate_ranking_at_ks,
    PopularityProfile,
    evaluate_beyond_accuracy,
)

# Fixed in advance. The first is the reported seed; the rest are arbitrary.
SEEDS = [STUDENT_ID, 1, 42, 12345]
K = 10

# What notebook Step 12 reports at the student-ID split. The student-ID run
# here rebuilds those models from the same configs, so it MUST land on these
# numbers - if it does not, this script is not measuring the same thing the
# report describes and its spread figures mean nothing. Checked automatically
# below rather than eyeballed.
REPORTED_NDCG10 = {
    "SVD": 0.1502, "Content": 0.0359, "Hybrid (weighted)": 0.1605,
    "Hybrid (switching)": 0.1474, "Most-Popular": 0.1549, "AutoRec": 0.1432,
    "Nested Hybrid": 0.1643,
}
# Checked too. The first version of this script checked only NDCG and so
# silently reported AutoRec's RMSE as 1.0792 - the NDCG-selected model's
# number - against a reported 0.8517, and concluded the claim "AutoRec has
# the lowest RMSE" failed. It did not; the script was measuring a different
# model. A reproduction check that covers one metric family only catches
# bugs in that one metric family.
REPORTED_RMSE = {"SVD": 0.8734, "AutoRec": 0.8517}

# The head-to-head comparisons the report actually makes. Compared PAIRWISE on
# each split rather than by overlapping error bars - see the note where these
# are printed.
PAIRS = [
    ("Nested Hybrid", "Hybrid (weighted)"),
    ("Nested Hybrid", "Most-Popular"),
    ("Hybrid (weighted)", "SVD"),
    ("Hybrid (switching)", "SVD"),
    ("SVD", "Most-Popular"),
    ("AutoRec", "Most-Popular"),
]

# Every one of these was selected on the STUDENT-ID validation split (notebook
# Steps 2/4/5/7/9/11) and is held fixed here - see the docstring.
SVD_RMSE_CFG = dict(n_factors=10, damping=5)                      # center defaults to user+item
SVD_RANK_CFG = dict(n_factors=10, damping=5, center="user")
CONTENT_CFG = dict(use_genres=False)
# TWO AutoRec configs, because the validation grid (Step 9) selected different
# ones for the two metric families and the notebook deploys both. Collapsing
# them into one is a real error, not a shortcut: k=25/wd=1e-2 was chosen to
# maximise NDCG and its validation RMSE is 1.0819 - worse than the user-mean
# baseline. Reporting that as "AutoRec's RMSE" would fabricate a finding.
AUTOREC_NDCG_CFG = dict(hidden_dim=25, weight_decay=1e-2, lr=1e-2, epochs=414, patience=1)
AUTOREC_RMSE_CFG = dict(hidden_dim=200, weight_decay=1e-4, lr=1e-2, epochs=68, patience=1)
HYBRID_ALPHA, HYBRID_FALLBACK = 0.5, "neutral"
NESTED_A1, NESTED_A2 = 0.4, 0.6


def popular_recommender(trainval_matrix):
    """Most-Popular baseline, rebuilt per split - popularity is a train-set fact."""
    from itertools import islice

    popularity = trainval_matrix.notna().sum(axis=0)
    # Stable sort: popularity counts are almost all ties, and an unstable sort
    # reshuffles the baseline's top-N between runs, drifting its NDCG in the
    # 4th decimal - which is exactly the magnitude this script is measuring.
    ranked = popularity.sort_values(ascending=False, kind="mergesort")
    seen = {
        uid: set(trainval_matrix.columns[trainval_matrix.loc[uid].notna()])
        for uid in trainval_matrix.index
    }

    def recommend(user_id, n=10):
        if user_id not in trainval_matrix.index:
            return []
        return list(islice((m for m in ranked.index if m not in seen[user_id]), n))

    return popularity, recommend


def run_one_seed(pre, seed):
    """Fit and evaluate all seven models on the split this seed produces."""
    t0 = time.time()
    ds = pre.split(val_frac=0.2, test_frac=0.2, seed=seed)

    svd_rmse = SVDRecommender(**SVD_RMSE_CFG).fit(ds.trainval_matrix)
    svd_rank = SVDRecommender(**SVD_RANK_CFG).fit(ds.trainval_matrix)
    content = ContentBasedRecommender(**CONTENT_CFG).fit(ds.trainval_matrix, pre.text_corpus)
    # seed passed through: in the real pipeline the split seed and AutoRec's
    # init seed are the same STUDENT_ID, so varying them together is what
    # actually reproduces "a different run of this project".
    autorec = AutoRecRecommender(seed=seed, **AUTOREC_NDCG_CFG).fit(ds.trainval_matrix)
    autorec_rmse = AutoRecRecommender(seed=seed, **AUTOREC_RMSE_CFG).fit(ds.trainval_matrix)

    hybrid = HybridRecommender(svd_rank, content, alpha=HYBRID_ALPHA, fallback=HYBRID_FALLBACK)
    switching = HybridRecommender(svd_rank, content, mode="switching")
    nested = HybridRecommender(
        HybridRecommender(svd_rank, RawScoreAdapter(autorec), alpha=NESTED_A1),
        content, alpha=NESTED_A2, fallback=HYBRID_FALLBACK,
    )

    popularity, recommend_popular = popular_recommender(ds.trainval_matrix)
    all_users = ds.trainval_matrix.index
    cold_items = set(popularity.index[popularity == 0])
    relevant_test = build_relevant_items(ds.test_df)

    rankers = {
        "SVD":                lambda uid: svd_rank.recommend_top_n(uid, n=K, exclude_seen=True).index.tolist(),
        "Content":            lambda uid: content.recommend_top_n(uid, n=K).index.tolist(),
        "Hybrid (weighted)":  lambda uid: hybrid.recommend_top_n(uid, n=K).index.tolist(),
        "Hybrid (switching)": lambda uid: switching.recommend_top_n(uid, n=K).index.tolist(),
        "Most-Popular":       lambda uid: recommend_popular(uid, n=K),
        "AutoRec":            lambda uid: autorec.recommend_top_n(uid, n=K, exclude_seen=True).index.tolist(),
        "Nested Hybrid":      lambda uid: nested.recommend_top_n(uid, n=K).index.tolist(),
    }

    # Only SVD and AutoRec produce calibrated ratings; evaluate_model() refuses
    # the rest by design, so RMSE is left as NaN for them rather than faked.
    rmse = {
        "SVD": evaluate_model(svd_rmse, ds.test_df).metrics["RMSE"],
        "AutoRec": evaluate_model(autorec_rmse, ds.test_df).metrics["RMSE"],
    }
    user_mean_rmse = float(np.sqrt(np.mean(
        (ds.test_df["rating"] - ds.test_df["userId"].map(ds.trainval_matrix.mean(axis=1))) ** 2
    )))

    pop_profile = PopularityProfile(popularity, n_users=len(all_users))
    rows = []
    for name, fn in rankers.items():
        top = {uid: fn(uid) for uid in all_users}
        ranking = evaluate_ranking_at_ks(lambda uid, t=top: t[uid], relevant_test, [K])
        beyond = evaluate_beyond_accuracy(
            lambda uid, t=top: t[uid], pop_profile, all_users, relevant_test, k=K
        )
        items = {m for lst in top.values() for m in lst[:K]}
        rows.append({
            "seed": seed,
            "model": name,
            f"NDCG@{K}": ranking[K][f"NDCG@{K}"],
            f"Precision@{K}": ranking[K][f"Precision@{K}"],
            "RMSE": rmse.get(name, np.nan),
            f"Novelty@{K}": beyond[f"Novelty@{K}"],
            f"Serendipity@{K}": beyond[f"Serendipity@{K}"],
            "Coverage": len(items) / len(popularity),
            "ColdSlots": sum(len(set(top[u][:K]) & cold_items) for u in all_users),
        })

    # AutoRec's cold-item degeneracy is a MATHEMATICAL claim (an all-zero input
    # row carries no signal), so it should hold at every seed by construction.
    # Checking it anyway is what separates "we argued it" from "we measured it".
    # clip=False and .std() to match notebook cell 42 exactly - clipping to
    # [0.5, 5.0] is a presentation step and has no business inside a check on
    # whether the DECODER distinguishes cold items.
    cold_list = sorted(cold_items)
    cold_std = 0.0
    if cold_list:
        spreads = []
        for uid in list(all_users)[:5]:
            scores = autorec.score_all_items(uid, clip=False)
            spreads.append(float(scores.loc[cold_list].std()))
        cold_std = max(spreads)

    meta = {
        "seed": seed,
        "n_cold_items": len(cold_items),
        "user_mean_RMSE": user_mean_rmse,
        "autorec_max_cold_std": cold_std,
        "seconds": time.time() - t0,
    }
    return pd.DataFrame(rows), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                    help="comma-separated split seeds; the first is treated as the reported one")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("Loading data (seed-independent, done once)...")
    pre = (
        MovieDataPreprocessor()
        .load_raw_data().build_movies_full().build_text_corpus().build_user_item_matrix()
    )

    frames, metas = [], []
    for i, seed in enumerate(seeds, 1):
        tag = " (reported)" if seed == seeds[0] else ""
        print(f"\n[{i}/{len(seeds)}] seed={seed}{tag} - fitting seven models...", flush=True)
        df, meta = run_one_seed(pre, seed)
        frames.append(df)
        metas.append(meta)
        print(f"      done in {meta['seconds']:.0f}s | cold items {meta['n_cold_items']} | "
              f"AutoRec max cold-item std {meta['autorec_max_cold_std']:.2e}")

    results = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------- reproduction check
    if seeds[0] == STUDENT_ID:
        ref = results[results.seed == STUDENT_ID].set_index("model")[f"NDCG@{K}"]
        ref_rmse = results[results.seed == STUDENT_ID].set_index("model")["RMSE"]
        drift = {f"{m} NDCG": ref[m] - v for m, v in REPORTED_NDCG10.items() if m in ref.index}
        drift.update({f"{m} RMSE": ref_rmse[m] - v for m, v in REPORTED_RMSE.items()
                      if m in ref_rmse.index and pd.notna(ref_rmse[m])})
        worst = max(drift.items(), key=lambda kv: abs(kv[1]))
        print(f"\nReproduction check against the notebook (student-ID split):")
        print(f"  largest difference: {worst[0]} {worst[1]:+.4f}")
        if abs(worst[1]) > 5e-4:
            print("  WARNING: this script does not reproduce the reported numbers.")
            print("  The spread figures below describe a DIFFERENT setup than the")
            print("  report does - fix this before quoting any of them.")
            for m, d in sorted(drift.items(), key=lambda kv: -abs(kv[1]))[:6]:
                print(f"    {m:25s} off by {d:+.4f}")
        else:
            print("  OK - reproduces the reported seven-model comparison.")

    out = REPO_ROOT / "data" / "seed_stability.csv"
    results.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")

    order = ["SVD", "Content", "Hybrid (weighted)", "Hybrid (switching)",
             "Most-Popular", "AutoRec", "Nested Hybrid"]

    # ---------------------------------------------------------------- spread
    print(f"\n=== NDCG@{K} across {len(seeds)} splits ===")
    pivot = results.pivot(index="model", columns="seed", values=f"NDCG@{K}").loc[order]
    pivot["mean"] = pivot.mean(axis=1)
    pivot["std"] = pivot[seeds].std(axis=1)
    pivot["range"] = pivot[seeds].max(axis=1) - pivot[seeds].min(axis=1)
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\n=== Coverage / Novelty@{K} across splits (mean +- std) ===")
    for col, fmt in (("Coverage", "{:.1%}"), (f"Novelty@{K}", "{:.2f}"), ("RMSE", "{:.4f}")):
        g = results.groupby("model")[col]
        line = pd.DataFrame({"mean": g.mean(), "std": g.std()}).reindex(order).dropna(how="all")
        print(f"\n{col}:")
        for name, r in line.iterrows():
            print(f"  {name:20s} {fmt.format(r['mean'])}  +- {fmt.format(r['std'])}")

    # ------------------------------------------------- claim-by-claim check
    # Each entry is a claim the report makes. A claim that holds at every seed
    # can be stated plainly; one that does not must be stated as within noise.
    print("\n=== Does each reported claim survive every split? ===")

    def per_seed(metric):
        return {s: results[results.seed == s].set_index("model")[metric] for s in seeds}

    ndcg, cov, nov, rm = (per_seed(f"NDCG@{K}"), per_seed("Coverage"),
                          per_seed(f"Novelty@{K}"), per_seed("RMSE"))
    claims = {
        "Nested Hybrid is the best ranker":
            [ndcg[s].idxmax() == "Nested Hybrid" for s in seeds],
        "SVD does NOT beat Most-Popular on NDCG":
            [ndcg[s]["SVD"] <= ndcg[s]["Most-Popular"] for s in seeds],
        # Deliberately NOT "every hybrid": the switching hybrid is a separate
        # claim below, and it goes the other way. Wording a check more broadly
        # than what it tests is how an overclaim reaches the report.
        "Both WEIGHTED hybrids beat plain SVD on NDCG":
            [min(ndcg[s]["Hybrid (weighted)"], ndcg[s]["Nested Hybrid"]) > ndcg[s]["SVD"] for s in seeds],
        "The SWITCHING hybrid also beats plain SVD on NDCG":
            [ndcg[s]["Hybrid (switching)"] > ndcg[s]["SVD"] for s in seeds],
        "AutoRec has the lowest test RMSE":
            [rm[s].idxmin() == "AutoRec" for s in seeds],
        "Content has the widest catalogue coverage":
            [cov[s].idxmax() == "Content" for s in seeds],
        "Content has the highest novelty":
            [nov[s].idxmax() == "Content" for s in seeds],
        "Content is the WORST ranker":
            [ndcg[s].idxmin() == "Content" for s in seeds],
        "AutoRec is as narrow as Most-Popular (coverage within 1pp)":
            [abs(cov[s]["AutoRec"] - cov[s]["Most-Popular"]) < 0.01 for s in seeds],
        # 1e-6, not 1e-9: torch computes in float32, whose epsilon is ~1.2e-7,
        # so identical inputs through a matmul can differ in the last bit or
        # two purely from memory layout. Against scores of order 3.5 anything
        # below 1e-6 IS zero. A 1e-9 threshold tests the floating-point unit,
        # not the model.
        "AutoRec gives cold items zero-variance predictions":
            [m["autorec_max_cold_std"] < 1e-6 for m in metas],
    }
    for claim, holds in claims.items():
        n = sum(holds)
        mark = "HOLDS " if n == len(seeds) else "FAILS "
        print(f"  [{mark}] {n}/{len(seeds)} splits  -  {claim}")

    # ------------------------------------------------- paired comparisons
    # Comparing two models by whether their mean +- std overlap is the WRONG
    # test here, and it is too harsh. Every model is scored on the SAME split,
    # so a split that happens to be generous lifts all seven together - that
    # shared movement is most of each model's individual std. Differencing the
    # pair on each split cancels it. A pair whose difference keeps its sign on
    # every split is separable even when its error bars overlap heavily.
    print("\n=== Head-to-head, differenced within each split ===")
    print(f"  {'ordering that holds on every split':47s} {'|diff|':>8s} {'std':>8s}  verdict")
    for a, b in PAIRS:
        diffs = np.array([ndcg[s_][a] - ndcg[s_][b] for s_ in seeds])
        wins, loses = bool(np.all(diffs > 0)), bool(np.all(diffs < 0))
        # Print the direction the data actually shows. Hard-coding "a > b"
        # printed a consistent LOSS as though it were a win.
        if wins:
            label, verdict = f"{a} > {b}", "separable"
        elif loses:
            label, verdict = f"{b} > {a}", "separable"
        else:
            label, verdict = f"{a} vs {b}", "NOT separable - sign flips"
        # ddof=1 (sample std, /n-1) to match every other std this script
        # reports (the per-model spread above, and the coverage/novelty/RMSE
        # mean+-std later) - a bare .std() on this numpy array defaults to
        # ddof=0 (population std, /n), which silently used a DIFFERENT
        # convention for the same n=4 sample and is a bug, not a choice.
        print(f"  {label:47s} {abs(diffs.mean()):8.4f} {diffs.std(ddof=1):8.4f}  {verdict}")
    print("\n  'separable' = the ordering is the same on all four splits, so it can")
    print("  be reported as a result. A pair whose sign flips must not be, however")
    print("  large the gap looks on the student-ID split alone.")

    fragile = [c for c, h in claims.items() if not all(h)]
    print("\n" + "-" * 72)
    if fragile:
        print("Claims that did NOT hold at every split - state these as within")
        print("selection noise in the report, not as findings:")
        for c in fragile:
            print(f"  * {c}")
    else:
        print("Every claim above held at every split tested.")
    print(f"\nWidest NDCG@{K} swing for any single model across splits: "
          f"{pivot['range'].max():.4f}")
    print("Two models closer together than that on the reported split should")
    print("not be presented as separable.")


if __name__ == "__main__":
    main()
