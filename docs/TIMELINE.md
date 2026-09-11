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

## Where that leaves us (as of 11 Sep, still inside proposal Week 1)

Both **Week 2's milestone (SVD, evaluated)** and **Week 3's milestone
(content-based model)** are done, plus the ranking-metrics and
responsible-AI evaluation work that wasn't tied to a single week in the
proposal. Roughly a week and a half ahead of the submitted plan on the
modelling side — the trade a proper 3-way split (not in the original plan)
bought back by catching test-leakage early rather than during the report.

Now the hybrid (originally Week 5) is done too, same day. Not started yet: AutoRec (Week 4).

## Suggested revised plan (adjust as needed — the Oct 23 date is fixed, everything before it is not)

| Target dates | Milestone |
|---|---|
| now – ~5 Oct | AutoRec (PyTorch, I-AutoRec) — can start immediately since the hybrid landed early too; kept a generous window since this is new territory (first PyTorch model) |
| 6 – 12 Oct | Full evaluation across all four models, ablations, polish |
| 13 – 19 Oct | Report writing, video recording, code freeze |
| 20 – 23 Oct | Final review and submission |

## How to keep this file current

Add a row to "Actual progress" whenever a milestone lands (worth doing in
the same commit, or right after). If a target date slips, update the
"Suggested revised plan" table rather than rewriting history in the
progress log above.
