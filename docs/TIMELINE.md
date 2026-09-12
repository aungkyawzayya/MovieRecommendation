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
| 12 Sep | AutoRec (I-AutoRec item-based autoencoder, PyTorch) — notebook Steps 9/10: 24-combo grid (k x weight_decay) tuned on validation, RMSE-best (k=200, wd=1e-4) and NDCG-best (k=25, wd=1e-2) diverge again (same lesson as Step 4's SVD centering choice); AutoRec wins on test RMSE (0.8517 vs SVD's 0.8734, user-mean's 0.9448) but loses on ranking (NDCG@10 0.1432 vs SVD 0.1502, weighted Hybrid 0.1616, Most-Popular 0.1549) because its recommendations concentrate on popular items (0.6% catalogue coverage - identical to Most-Popular's own 0.6%, the clearest sign the ranking gain came from picking better-sellers, not from personalising); cold-item degeneracy confirmed exactly as with SVD (std=0.00e+00 across 732 untrained items per user). **Fixed a real bug along the way**: the first grid (epochs=80, patience=10) under-trained 4 of 6 k values, since full-batch training makes one epoch a single gradient step - patience=200 let every config reach its own validation peak (`stopped_by` column added to catch this class of bug again), which flipped the capacity-vs-data-size conclusion (RMSE now improves monotonically with k, converged k=10 alone moved 0.9681 -> 0.9028) and reproduced bit-for-bit between the cloud verification run and Aung's own Mac | `fa6697e`, `d8b3beb`, `<pending>` |

## Where that leaves us (as of 11 Sep, still inside proposal Week 1)

Both **Week 2's milestone (SVD, evaluated)** and **Week 3's milestone
(content-based model)** are done, plus the ranking-metrics and
responsible-AI evaluation work that wasn't tied to a single week in the
proposal. Roughly a week and a half ahead of the submitted plan on the
modelling side — the trade a proper 3-way split (not in the original plan)
bought back by catching test-leakage early rather than during the report.

Now the hybrid (originally Week 5) is done too, same day. AutoRec (Week 4) is also done — non-linear model built, tuned, and proven to inherit the same cold-start ceiling as SVD rather than solving it, which is itself a report-worthy finding.

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
