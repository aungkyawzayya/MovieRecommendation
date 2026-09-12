# Project Timeline — Hybrid Movie Recommendation System

COMP813 final project (individual, 60% of grade). Due **Friday 23 Oct 2026**.
This file tracks progress against the timeline submitted in
`docs/Project_Proposal.docx` (Section 8) and gets updated as work lands —
treat it as a running log, not a one-time plan.

## Originally proposed timeline (from Project_Proposal.docx)

| Week | Dates | Milestone |
|---|---|---|
| 1 | 8 – 14 Sep | Proposal review with tutor; environment setup; data preprocessing |
| 2 | 15 – 21 Sep | Baseline SVD model implemented and evaluated |
| 3 | 22 – 28 Sep | TF-IDF content-based model implemented |
| 4 | 29 Sep – 5 Oct | AutoRec (PyTorch) implemented and tuned |
| 5 | 6 – 12 Oct | Hybrid combination, full evaluation, ablations |
| 6 | 13 – 19 Oct | Report writing, video recording, code cleanup |
| 7 | 20 – 23 Oct | Final review and submission (due Fri 23 Oct 2026) |

## Actual progress

| Date | What landed | Commit(s) |
|---|---|---|
| ~9 Sep | Data preprocessing (MovieLens + TMDB overviews), environment setup | `02d2076` and earlier |
| 9–10 Sep | SVD baseline: item-bias term, clip/ranking split, 2-D grid search (k × damping) | `02d2076`, `633c48f`, `fcf3fc6` |
| 10–11 Sep | **Methodology fix**: rewrote the split from 2-way to a proper train/val/test 3-way split (matching the lecturer's own Assignment 1/2 code), to stop hyperparameter tuning from leaking off the test set. Content-based model (TF-IDF) built. Evaluation module (RMSE/MAE) added. | `897c236` |
| 11 Sep | Notebook restructured into the Step N / Step N's Graph pattern (matches Assignment 2's style) | `a598d76` |
| 11 Sep | Ranking metrics module (Precision@K / Recall@K / NDCG@K) + notebook Step 4 — SVD centering variant re-selected on validation NDCG (not RMSE) | `451c09f`, `b3c92f1` |
| 11 Sep | Code review pass (bug fixes): content model's seen-item exclusion, precision@k denominator, coverage-reporting honesty, split() edge case, ~40x eval speedup | `552e593`, `08e8611` |
| 11 Sep | Notebook Step 5 — content-based model exercised end to end (coverage, use_genres selection, test evaluation, SVD comparison) | `08e8611` |
| 11 Sep | Notebook Step 6 — Responsible AI: popularity bias (SVD vs content vs naive baseline) and cold-start fairness | `c26b848` |
| 11 Sep | Hybrid model (SVD + Content, per-user z-score blend) — notebook Steps 7/8: weighted hybrid beats the Most-Popular baseline on NDCG at 3/3 cutoffs (SVD alone: 0/3); a switching-mode ablation reaches cold-start items far more often than the weighted blend | `83c5ff7` |
| 12 Sep | AutoRec (I-AutoRec item-based autoencoder, PyTorch) — notebook Steps 9/10: 24-combo grid (k x weight_decay) tuned on validation, RMSE-best (k=200, wd=1e-4) and NDCG-best (k=25, wd=1e-2) diverge again (same lesson as Step 4's SVD centering choice); AutoRec wins on test RMSE (0.8517 vs SVD's 0.8734, user-mean's 0.9448) but loses on ranking (NDCG@10 0.1432 vs SVD 0.1502, weighted Hybrid 0.1616, Most-Popular 0.1549) because its recommendations concentrate on popular items (0.6% catalogue coverage - identical to Most-Popular's own 0.6%, the clearest sign the ranking gain came from picking better-sellers, not from personalising); cold-item degeneracy confirmed exactly as with SVD (std=0.00e+00 across 732 untrained items per user). **Fixed a real bug along the way**: the first grid (epochs=80, patience=10) under-trained 4 of 6 k values, since full-batch training makes one epoch a single gradient step - patience=200 let every config reach its own validation peak (`stopped_by` column added to catch this class of bug again), which flipped the capacity-vs-data-size conclusion (RMSE now improves monotonically with k, converged k=10 alone moved 0.9681 -> 0.9028) and reproduced bit-for-bit between the cloud verification run and Aung's own Mac | `fa6697e`, `d8b3beb`, `8f6e14c` |
| 12 Sep | Nested Hybrid (SVD + AutoRec, then + Content, via composition) — notebook Steps 11/12: `inner=Hybrid(svd,autorec)` then `full=Hybrid(inner,content)`, exactly as planned, but tracing the actual interface calls found the plan's "no code changes needed" claim was wrong on two counts fixed in `hybrid.py` (`score_all_items` had no `clip` parameter, so nesting raised `TypeError`; `item_ids` was missing, so `rankable_items` raised `AttributeError`) plus a third, quieter one (the content slot calls with no `clip` arg, so AutoRec's own `clip=True` default would have silently flattened its scores) fixed with a new `RawScoreAdapter` (`produces_ratings = False` added defensively too) — no change to the blend logic or either ablation. Three independent boundary checks confirm the composition computes real arithmetic rather than silently collapsing to one input: alpha2=0.0 reproduces Content-only (0.0400, Step 7), alpha1=alpha2=1.0 reproduces SVD-only (0.1060, Step 7), and alpha1=0/alpha2=1 reproduces AutoRec's own validation NDCG@10 exactly (0.1196, Step 9). **Honest result: this did not improve on the two-model Hybrid.** Validation NDCG@10 rose (0.1192 -> 0.1244 at alpha1=0.2, alpha2=0.6) but test NDCG@10 fell and the sign flipped (0.1616 -> 0.1590), and the Most-Popular win-count dropped (3/3 -> 2/3) - consistent with validation overfitting, not a real gain: the (alpha1, alpha2) grid (36 cells) is a second round of hyperparameter selection stacked on top of AutoRec's own 24-config selection, both against the same ~598 validation users. **The two-model Hybrid (SVD+Content) remains the best-performing ranking model** and is what the report should recommend; the nested composition is kept as a documented ablation for the requirements' "ablation studies" / "critically analyze findings" asks, not as an improvement. It does show a real, separate trade-off worth reporting: best-of-all-seven catalogue coverage (6.0% vs plain Hybrid's 5.8%) and lowest popularity bias (mean pop. 122.5 vs 131.8), but at the cost of losing the plain Hybrid's cold-start reach entirely (0 vs 6 cold items in top-10) - AutoRec's own popularity collapse (0.6% coverage) cancels out content's cold-item boost once blended in | `e30c53b`, `1b32d5d`, `18c16a5` |
| 13 Sep | TMDB overview coverage expanded 36.4% -> 98.8% (`scripts/fetch_overviews.py`, checkpointed/rate-limited fetch of the 6,084 missing plot overviews) and every downstream number re-measured rather than assumed. Content model: cold-start reach roughly tripled (163 -> 481 of 6,100 slots) and catalogue coverage rose (17.8% -> 24.9%), but ranking quality FELL (NDCG@10 0.0469 -> 0.0359) - cold items are only 2.92% of relevant held-out ratings, so surfacing more of them displaces popular items far more likely to be hits; documented in `content.py`'s docstring as the accuracy-coverage trade-off it is, not a bug. Weighted Hybrid's cold-start reach did NOT move at all (6/6,100 both before and after) even though cold-item text coverage went 18% -> 98% - this **falsifies** Step 8's own earlier prediction that text coverage was the binding constraint; the real cap is the blend weight (a cold item's flat svd_z, averaged at alpha=0.5, can't outscore a warm item that's strong on both halves) - the switching-mode ablation, which hands cold items to content alone instead of averaging, is what actually moved (38 -> 60 slots) on the same data. Cell 33's interpretation was rewritten to report this rather than leave a tested-and-wrong prediction standing. **Nested Hybrid methodology fix**: the unconstrained alpha1/alpha2 grid's validation winner (alpha1=0.0, alpha2=1.0) is a degenerate corner that collapses to AutoRec alone, beating the best genuine three-model blend by only 0.0006 NDCG - noise on ~598 validation users, not a real result. Both are now reported, and Step 12 deploys the best *strictly interior* blend (alpha1=0.4, alpha2=0.6) instead. That reverses the previous round's conclusion: this interior Nested Hybrid scores test NDCG@10=0.1643, actually beating the two-model Hybrid's own 0.1605 (itself re-selected at alpha=0.5 now that content has changed) - **the Nested Hybrid (a1=0.4, a2=0.6) is now the best-performing ranking model of all seven**, and beats Most-Popular on NDCG at 3/3 cutoffs | `630b54f`, `5fa5c82`, `5bd40e8` |

## Where that leaves us (as of 11 Sep, still inside proposal Week 1)

Both **Week 2's milestone (SVD, evaluated)** and **Week 3's milestone
(content-based model)** are done, plus the ranking-metrics and
responsible-AI evaluation work that wasn't tied to a single week in the
proposal. Roughly a week and a half ahead of the submitted plan on the
modelling side — the trade a proper 3-way split (not in the original plan)
bought back by catching test-leakage early rather than during the report.

Now the hybrid (originally Week 5) is done too, same day. AutoRec (Week 4) is also done — non-linear model built, tuned, and proven to inherit the same cold-start ceiling as SVD rather than solving it, which is itself a report-worthy finding. The nested three-model nested_hybrid_deploy composition from the AutoRec guide's own Step 6 is also done, ahead of Week 5's "full evaluation, ablations" milestone — interface parity across all three base models (SVD/Content/AutoRec) meant only two small additions to HybridRecommender were needed, not a refactor. That composition's first alpha sweep looked like validation overfitting (a degenerate corner selection beat the real blend by noise) until the selection rule itself was fixed to exclude degenerate corners and the TMDB coverage gap was closed - with both fixed, the genuine three-model Nested Hybrid (alpha1=0.4, alpha2=0.6) is the best ranking model measured so far. Two of this project's own predictions were tested against real data and found wrong along the way (the capacity-vs-k argument in the AutoRec grid, and Step 8's text-coverage-caps-cold-start claim) - both are left in the report as tested-and-corrected rather than quietly fixed, since a lecturer grading "critically analyze findings" should be able to see the actual before/after.

## Suggested revised plan (adjust as needed — the Oct 23 date is fixed, everything before it is not)

| Target dates | Milestone |
|---|---|
| 6 – 12 Oct | Full evaluation across all four models, ablations, polish |
| 13 – 19 Oct | Report writing, video recording, code freeze |
| 20 – 23 Oct | Final review and submission |

## How to keep this file current

Add a row to "Actual progress" whenever a milestone lands (worth doing in
the same commit, or right after). If a target date slips, update the
"Suggested revised plan" table rather than rewriting history in the
progress log above.
