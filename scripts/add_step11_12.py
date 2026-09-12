"""
add_step11_12.py
Appends Step 11 (Nested Hybrid: SVD + AutoRec, then + Content, via
composition) and Step 12 (seven-model final comparison) to the notebook.

Run from the repo root, with the notebook CLOSED:
    python scripts/add_step11_12.py
"""

import json
import pathlib

repo = pathlib.Path(__file__).resolve().parents[1]
nbp = repo / "notebooks/01_eda.ipynb"
nb = json.loads(nbp.read_text())


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


step11_md = md(
"""## Step 11 — Nesting AutoRec Into the Hybrid (Composition, Not a Rewrite)

`HybridRecommender` was built around `score_all_items`/`predict`/`recommend_top_n`/
`supported_items`/`item_ids`/`seen_items` from the very start (Step 7) so that
adding a third model would not mean touching its code. That claim gets
tested here, not just asserted:

```python
inner = HybridRecommender(svd, autorec, alpha=a1)   # two CF models first
full  = HybridRecommender(inner, content, alpha=a2) # then blend in content
```

Checking this against `hybrid.py` line by line surfaced two real gaps, not
zero:

1. `full`'s `_components()` calls `self._svd.score_all_items(user_id,
   clip=False)` — but `HybridRecommender.score_all_items` took no `clip`
   argument at all, so `full._svd = inner` would raise `TypeError` the
   first time it ran.
2. `full.rankable_items` (weighted mode) reads `self._svd.item_ids` —
   `HybridRecommender` never defined that property, so `full.rankable_items`
   would raise `AttributeError`.
3. A third, quieter one: `full`'s `_content_z_full` calls
   `self._content.score_all_items(user_id)` with **no** `clip` argument,
   because `ContentBasedRecommender` doesn't have one. `AutoRecRecommender`
   does, defaulting to `clip=True` — so putting AutoRec straight into the
   `content_model` slot would silently score it on the clipped 0.5-5.0
   scale exactly where the SVD slot's `clip=False` note says clipping
   flattens the differences z-scoring needs.

Two small, targeted fixes in `hybrid.py` close these — not a rewrite:
`score_all_items` now accepts (and ignores) `clip`, `item_ids` delegates to
`self._svd.item_ids`, and a new `RawScoreAdapter` class forwards to any
clip-supporting model with `clip=False` pinned, for use in the
`content_model` slot. Everything else in the file — the blend logic, both
ablations, the coverage properties — is untouched.

Fitting is free here: `svd_t`/`content_t` (Step 7) and `best_ndcg["model"]`
(Step 9, the NDCG-selected AutoRec) are already-fit, train-only tuning
objects sitting in memory. Nesting them costs one 2-D validation sweep over
alpha1 (SVD vs AutoRec) and alpha2 (that blend vs Content) — pure z-score
arithmetic, no retraining."""
)

step11_code = code(
"""from core.models.hybrid import RawScoreAdapter

# Reuses TRAIN-only tuning objects already fit for earlier sweeps - svd_t
# and content_t (Step 7), and best_ndcg["model"] (Step 9's NDCG-selected
# AutoRec, since this hybrid only ever ranks - it has no rating-error use,
# same reason the original Hybrid was always built from the NDCG-selected
# SVD centering variant, not the RMSE-selected one).
autorec_t = best_ndcg["model"]

# alpha1 blends SVD with AutoRec (both collaborative filtering); alpha2
# then blends that combination with Content. A coarser 6x6 grid (step 0.2)
# keeps the printed table readable - each cell is cached z-score
# arithmetic over already-fit models, not a retrain, so this is not a
# compute-time compromise.
alphas_nested = np.round(np.arange(0.0, 1.01, 0.2), 2)
nested_rows = []
best_nested = {"val_NDCG@10": -float("inf")}

for a1 in alphas_nested:
    # RawScoreAdapter forces clip=False on AutoRec's scores - see Step 11's
    # markdown and hybrid.py's docstring for why the bare interface would
    # silently clip otherwise. cache_components=True: inner_t's own
    # (svd_z, autorec_z) pair doesn't depend on alpha2, so it is computed
    # once per user and reused across the whole inner a2 loop below.
    inner_t = HybridRecommender(svd_t, RawScoreAdapter(autorec_t), alpha=a1, cache_components=True)
    for a2 in alphas_nested:
        full_t = HybridRecommender(inner_t, content_t, alpha=a2, fallback=best_fallback)
        # m=full_t freezes THIS instance in the lambda's default arg - the
        # same closure-over-loop-variable guard as Step 7's alpha sweep.
        fn = lambda uid, m=full_t: m.recommend_top_n(uid, n=10, exclude_seen=True).index.tolist()
        val_ndcg = evaluate_ranking(fn, relevant_val, k=10)["NDCG@10"]
        row = {"alpha1_svd_vs_autorec": a1, "alpha2_cf_vs_content": a2, "val_NDCG@10": val_ndcg}
        nested_rows.append(row)
        if val_ndcg > best_nested["val_NDCG@10"]:
            best_nested = dict(row)

nested_grid = pd.DataFrame(nested_rows)
nested_pivot = nested_grid.pivot(index="alpha1_svd_vs_autorec", columns="alpha2_cf_vs_content",
                                  values="val_NDCG@10")
print("=== Nested Hybrid validation NDCG@10: alpha1 (SVD vs AutoRec) x alpha2 (that blend vs Content) ===")
print(nested_pivot.to_string(float_format=lambda v: f"{v:.4f}"))

best_a1 = best_nested["alpha1_svd_vs_autorec"]
best_a2 = best_nested["alpha2_cf_vs_content"]
print(f"\\nValidation-selected: alpha1={best_a1}, alpha2={best_a2}  "
      f"(NDCG@10={best_nested['val_NDCG@10']:.4f})")
print(f"For reference, the two-model weighted Hybrid selected alpha={best_alpha} "
      f"(NDCG@10={alpha_results[best_fallback][best_alpha]:.4f}) on the same validation split."
)"""
)

step11_graph_md = md("### Step 11's Graph — Validation NDCG@10 Across (alpha1, alpha2)")

step11_graph_code = code(
"""plt.figure(figsize=(7, 5.5))
im = plt.imshow(nested_pivot.to_numpy(), cmap="viridis", origin="lower", aspect="auto")
plt.colorbar(im, label="Validation NDCG@10")
plt.xticks(range(len(nested_pivot.columns)), nested_pivot.columns)
plt.yticks(range(len(nested_pivot.index)), nested_pivot.index)
plt.xlabel("alpha2 (weight on SVD+AutoRec; 1-alpha2 on content)")
plt.ylabel("alpha1 (weight on SVD; 1-alpha1 on AutoRec)")
best_row = list(nested_pivot.index).index(best_a1)
best_col = list(nested_pivot.columns).index(best_a2)
plt.scatter([best_col], [best_row], marker="*", s=300, color="white", edgecolor="black",
            label=f"selected: alpha1={best_a1}, alpha2={best_a2}")
plt.title("Step 11 - Nested Hybrid alpha selection")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()"""
)

step12_md = md(
"""## Step 12 — Seven-Model Final Comparison

`nested_hybrid_deploy` is built the same way `hybrid_model` was in Step 8:
compose the already-fit DEPLOYMENT objects (`rank_model`, `content_model`
from Steps 4/5, `autorec_ndcg_deploy` from Step 10) at the validation-
selected alpha1/alpha2 - nothing refit here either.

`produces_ratings = False` (inherited from `HybridRecommender`, unchanged
by the two Step 11 additions), so this model joins the ranking tables
only, the same way both earlier Hybrids did. `final_rankers`,
`final_reachable_pool`, `final_ndcg` and `final_precision` are extended in
place - the exact pattern Step 10 used to add AutoRec to Step 8's tables."""
)

step12_code = code(
"""nested_inner_deploy = HybridRecommender(
    rank_model, RawScoreAdapter(autorec_ndcg_deploy), alpha=best_a1,
)
nested_hybrid_deploy = HybridRecommender(
    nested_inner_deploy, content_model, alpha=best_a2, fallback=best_fallback,
)

final_rankers["Nested Hybrid"] = (
    lambda uid: nested_hybrid_deploy.recommend_top_n(uid, n=max(Ks), exclude_seen=True).index.tolist()
)
final_reachable_pool["Nested Hybrid"] = len(popularity)  # scores the whole catalogue, like SVD/AutoRec

nested_ranking = evaluate_ranking_at_ks(final_rankers["Nested Hybrid"], relevant_test, Ks)
final_ndcg["Nested Hybrid"] = pd.Series({k: nested_ranking[k][f"NDCG@{k}"] for k in Ks})
final_precision["Nested Hybrid"] = pd.Series({k: nested_ranking[k][f"Precision@{k}"] for k in Ks})

print("=== NDCG@K on TEST - seven models (Nested Hybrid added) ===")
print(final_ndcg.to_string(float_format=lambda v: f"{v:.4f}"))
print("\\n=== Precision@K on TEST ===")
print(final_precision.to_string(float_format=lambda v: f"{v:.4f}"))

nested_wins = int((final_ndcg["Nested Hybrid"] > final_ndcg["Most-Popular"]).sum())
weighted_hybrid_col = f"Hybrid (weighted, a={best_alpha})"
print(f"\\nNested Hybrid beats Most-Popular on NDCG at {nested_wins}/{len(Ks)} cutoffs "
      f"(weighted SVD+Content Hybrid: {hybrid_wins}/{len(Ks)}, AutoRec alone: {autorec_wins}/{len(Ks)})."
)"""
)

step12_graph_md = md(
"""### Step 12's Graph — Seven-Model Comparison

Step 10's figure was drawn before the Nested Hybrid existed, so it shows
six models while this step's tables show seven - recomputed here rather
than leaving a stale figure beside a wider table, the same reason Step 10
redrew Step 8's three-panel figure when AutoRec joined."""
)

step12_graph_code = code(
"""final_top10 = {
    name: {uid: fn(uid)[:10] for uid in all_users}
    for name, fn in final_rankers.items()
}

seven_bias_rows = []
for name in final_rankers:
    items, pops = set(), []
    for uid in all_users:
        for m in final_top10[name][uid]:
            items.add(m)
            pops.append(popularity.get(m, 0))
    seven_bias_rows.append({
        "Recommender": name,
        "Catalog coverage": len(items) / len(popularity),
        "Coverage of reachable pool": len(items) / final_reachable_pool[name],
        "Mean popularity of recs": float(np.mean(pops)) if pops else np.nan,
    })
seven_bias_df = pd.DataFrame(seven_bias_rows).set_index("Recommender")

seven_cold_hits = {
    name: sum(len(set(final_top10[name][uid]) & cold_items) for uid in all_users)
    for name in final_rankers
}

print("=== Coverage and popularity bias - all seven models ===")
print(seven_bias_df.to_string(formatters={
    "Catalog coverage": "{:.1%}".format,
    "Coverage of reachable pool": "{:.1%}".format,
    "Mean popularity of recs": "{:.1f}".format,
}))
print(f"\\n=== Cold-start items in top-10 (out of {len(all_users) * 10} slots) ===")
for name, hits in seven_cold_hits.items():
    print(f"{name:35s}: {hits:5d}  ({100 * hits / (len(all_users) * 10):.2f}% of slots)")

fig, axes = plt.subplots(1, 3, figsize=(20, 4.8))

axes[0].bar(seven_bias_df.index, seven_bias_df["Catalog coverage"] * 100, color="#4C72B0")
axes[0].set_ylabel("Catalogue coverage (%)")
axes[0].set_title("Step 12a - catalogue coverage")
axes[0].tick_params(axis="x", rotation=30)

palette7 = dict(palette)
palette7["Nested Hybrid"] = "#DD8452"
for name in final_ndcg.columns:
    axes[1].plot(final_ndcg.index, final_ndcg[name], marker="o",
                 linestyle="--" if name == "Most-Popular" else "-",
                 color=palette7.get(name), label=name)
axes[1].set_xlabel("K"); axes[1].set_ylabel("NDCG@K (test)")
axes[1].set_xticks(Ks)
axes[1].set_title("Step 12b - ranking quality, all seven models")
axes[1].legend(fontsize=7)

axes[2].bar(list(seven_cold_hits.keys()), list(seven_cold_hits.values()),
            color=[palette7.get(n, "#4C72B0") for n in seven_cold_hits])
axes[2].set_ylabel("Cold-start items in top-10")
axes[2].set_title("Step 12c - cold-start reach")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()"""
)

nb["cells"].extend([
    step11_md, step11_code, step11_graph_md, step11_graph_code,
    step12_md, step12_code, step12_graph_md, step12_graph_code,
])

nbp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Appended Step 11 and Step 12. Total cells now: {len(nb['cells'])}")
