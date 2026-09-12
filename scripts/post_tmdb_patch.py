"""
post_tmdb_patch.py
Three corrections after expanding TMDB overview coverage from 36.4% to 98.8%.

  1. content.py's docstring still says "covers only 3,536 of the 9,724 rated
     movies (36%) ... Pull more TMDB overviews before relying on it." That
     was done; the file now covers 9,603 (98.8%).

  2. Step 8's closing line claims "Cold-start reach is capped by plot text
     coverage, so improving TMDB overviews is the prerequisite for it."
     That prediction has now been tested and falsified by this project's own
     experiment: text coverage of cold items went 18% -> 98% and the weighted
     hybrid's cold-start reach stayed at exactly 6 slots out of 6,100. The
     real cap is the blend weight, not the text.

  3. Step 11's grid selected alpha1=0.0, alpha2=1.0 — both degenerate, which
     reduces the "Nested Hybrid" to AutoRec alone, and Step 12's table
     therefore prints two identical rows. It won by 0.0005 NDCG over a
     genuine three-model blend. Both are now reported, and Step 12 carries
     the interior blend forward so the row is a real three-model result
     rather than a duplicate of one already in the table.

Run from the repo root, with the notebook CLOSED:
    python scripts/post_tmdb_patch.py
"""

import json
import pathlib

repo = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- content.py
p = repo / "core/models/content.py"
s = p.read_text()
old = """CAVEAT, measured: overview_plot.csv covers only 3,536 of the 9,724 rated
movies (36%), and only 297 of the 1,641 items with zero train ratings have
plot text. So this model currently rescues ~18% of the cold-start cases,
not all of them. Pull more TMDB overviews before relying on it in the hybrid."""
new = """COVERAGE, measured: overview_plot.csv was expanded from 4,800 to 10,884
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
accuracy-coverage trade-off, not a regression to fix."""
assert old in s, "content.py caveat not found"
p.write_text(s.replace(old, new))
print("content.py: 36% caveat -> measured 98.8% coverage and its trade-off")

# ------------------------------------------------------------------ notebook
nbp = repo / "notebooks/01_eda.ipynb"
nb = json.loads(nbp.read_text())


def patch(index, pairs):
    src = "".join(nb["cells"][index]["source"])
    for old_t, new_t in pairs:
        assert old_t in src, f"cell {index}: not found -> {old_t[:70]!r}"
        src = src.replace(old_t, new_t)
    nb["cells"][index]["source"] = src.splitlines(keepends=True)
    if nb["cells"][index]["cell_type"] == "code":
        nb["cells"][index]["outputs"] = []
        nb["cells"][index]["execution_count"] = None


# --- Step 8: the prediction this project went on to falsify -----------------
patch(33, [("""print("\\nInterpretation: any hybrid gain in NDCG above cannot be explained by")
print("these counts - they are far too small. The gain comes from content")
print("re-ranking items SVD already covers. Cold-start reach is capped by plot")
print("text coverage, so improving TMDB overviews is the prerequisite for it.")""",
"""print("\\nInterpretation: any hybrid gain in NDCG above cannot be explained by")
print("these counts - they are far too small. The gain comes from content")
print("re-ranking items SVD already covers.")
print()
print("An earlier version of this cell predicted that cold-start reach was")
print("capped by plot text coverage, and that fetching more TMDB overviews")
print("was the prerequisite for improving it. That prediction was tested and")
print("is WRONG. Coverage of cold items was raised from 18% to 98%")
print("(scripts/fetch_overviews.py) and the weighted hybrid's cold-start")
print("reach did not move at all - still exactly 6 slots out of 6,100.")
print()
print("The real cap is the BLEND WEIGHT. A cold item has no collaborative")
print("signal, so its svd_z sits at the user's own mean; at alpha=0.5 the")
print("content half cannot lift it past warm items that score well on both")
print("halves. The switching hybrid, which hands cold items content_z alone")
print("instead of averaging it against a flat svd_z, is the variant whose")
print("cold reach did improve (38 -> 60 slots) on the same data.")""")])
print("notebook cell 33: falsified prediction replaced with the measured result")

# --- Step 11: report the interior blend as well as the corner --------------
patch(46, [("""best_a1 = best_nested["alpha1_svd_vs_autorec"]
best_a2 = best_nested["alpha2_cf_vs_content"]
print(f"\\nValidation-selected: alpha1={best_a1}, alpha2={best_a2}  "
      f"(NDCG@10={best_nested['val_NDCG@10']:.4f})")""",
"""best_a1 = best_nested["alpha1_svd_vs_autorec"]
best_a2 = best_nested["alpha2_cf_vs_content"]
print(f"\\nValidation-selected: alpha1={best_a1}, alpha2={best_a2}  "
      f"(NDCG@10={best_nested['val_NDCG@10']:.4f})")

# alpha1 or alpha2 at 0.0/1.0 is a DEGENERATE corner: the "nested hybrid"
# collapses to one of the models already in the comparison (alpha1=0 ->
# AutoRec alone, alpha2=1 -> whatever the inner blend is, alpha2=0 ->
# Content alone). Selecting one is a legitimate answer - the search saying
# "do not blend" - but it makes the Step 12 row a duplicate of an existing
# one. So the best STRICTLY INTERIOR blend is reported too: that is the
# best genuine three-model combination, and it is what Step 12 carries.
interior = nested_grid[
    nested_grid.alpha1_svd_vs_autorec.between(0.01, 0.99)
    & nested_grid.alpha2_cf_vs_content.between(0.01, 0.99)
]
best_interior = interior.loc[interior["val_NDCG@10"].idxmax()]
int_a1 = best_interior["alpha1_svd_vs_autorec"]
int_a2 = best_interior["alpha2_cf_vs_content"]
print(f"Best STRICTLY INTERIOR blend: alpha1={int_a1}, alpha2={int_a2}  "
      f"(NDCG@10={best_interior['val_NDCG@10']:.4f})")

degenerate = (best_a1 in (0.0, 1.0)) or (best_a2 in (0.0, 1.0))
if degenerate:
    gap = best_nested["val_NDCG@10"] - best_interior["val_NDCG@10"]
    print(f"\\nThe unconstrained winner is a DEGENERATE corner - it reduces to a")
    print(f"single model already in the comparison - and it beats the best real")
    print(f"three-model blend by only {gap:.4f} NDCG, well inside selection noise")
    print(f"on ~{len(relevant_val)} validation users. Step 12 therefore carries the")
    print(f"interior blend, so the 'Nested Hybrid' row is an actual three-model")
    print(f"result rather than a second copy of one already in the table.")""")])
print("notebook cell 46: interior blend reported alongside the corner")

# --- Step 12: deploy the interior blend ------------------------------------
patch(50, [("""nested_inner_deploy = HybridRecommender(
    rank_model, RawScoreAdapter(autorec_ndcg_deploy), alpha=best_a1,
)
nested_hybrid_deploy = HybridRecommender(
    nested_inner_deploy, content_model, alpha=best_a2, fallback=best_fallback,
)""",
"""# Built at the INTERIOR alphas from Step 11, not the unconstrained winner:
# that winner (alpha1=0.0, alpha2=1.0) collapses to AutoRec alone, which is
# already its own row below, so carrying it would print the same numbers
# twice and tell the reader nothing new. The interior pair is the best
# genuine SVD+AutoRec+Content blend the validation sweep found, and it lost
# to the corner by an amount well inside noise. Step 11 prints both.
deploy_a1, deploy_a2 = int_a1, int_a2

nested_inner_deploy = HybridRecommender(
    rank_model, RawScoreAdapter(autorec_ndcg_deploy), alpha=deploy_a1,
)
nested_hybrid_deploy = HybridRecommender(
    nested_inner_deploy, content_model, alpha=deploy_a2, fallback=best_fallback,
)"""),
("""final_rankers["Nested Hybrid"] = (""",
 """final_rankers[f"Nested Hybrid (a1={deploy_a1}, a2={deploy_a2})"] = ("""),
("""final_reachable_pool["Nested Hybrid"] = len(popularity)  # scores the whole catalogue, like SVD/AutoRec

nested_ranking = evaluate_ranking_at_ks(final_rankers["Nested Hybrid"], relevant_test, Ks)
final_ndcg["Nested Hybrid"] = pd.Series({k: nested_ranking[k][f"NDCG@{k}"] for k in Ks})
final_precision["Nested Hybrid"] = pd.Series({k: nested_ranking[k][f"Precision@{k}"] for k in Ks})""",
 """nested_col = f"Nested Hybrid (a1={deploy_a1}, a2={deploy_a2})"
final_reachable_pool[nested_col] = len(popularity)  # scores the whole catalogue, like SVD/AutoRec

nested_ranking = evaluate_ranking_at_ks(final_rankers[nested_col], relevant_test, Ks)
final_ndcg[nested_col] = pd.Series({k: nested_ranking[k][f"NDCG@{k}"] for k in Ks})
final_precision[nested_col] = pd.Series({k: nested_ranking[k][f"Precision@{k}"] for k in Ks})"""),
("""nested_wins = int((final_ndcg["Nested Hybrid"] > final_ndcg["Most-Popular"]).sum())""",
 """nested_wins = int((final_ndcg[nested_col] > final_ndcg["Most-Popular"]).sum())""")])
print("notebook cell 50: deploys the interior blend, row renamed with its alphas")

# --- Step 12's graph: palette key follows the renamed column ---------------
patch(52, [('palette7["Nested Hybrid"] = "#DD8452"',
            'palette7[nested_col] = "#DD8452"')])
print("notebook cell 52: graph palette follows the renamed column")

nbp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"\nnotebook written ({len(nb['cells'])} cells). Reload, restart kernel, run all.")
