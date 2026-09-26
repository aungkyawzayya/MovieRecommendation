# Hybrid Movie Recommendation System

COMP813 Artificial Intelligence 2026 — Final Project
Aung Kyaw Zay Ya (24265298), Auckland University of Technology

A hybrid movie recommender combining collaborative filtering (SVD), a deep
model (I-AutoRec), and content-based NLP (TF-IDF over plot overviews),
evaluated on MovieLens `ml-latest-small` with TMDB plot text.

Seven models are built and compared on the same held-out test split, on
both rating-error and ranking metrics, plus coverage and popularity-bias
measures. The headline result is that no single model wins everything:

| Model | Test NDCG@10 | Test RMSE | Catalogue coverage | Novelty@10 | Cold-start reach |
|---|---|---|---|---|---|
| User-mean baseline | — | 0.9448 | — | — | — |
| Most-Popular (no personalization) | 0.1549 | — | 0.6% | 1.65 | 0 |
| SVD (collaborative filtering) | 0.1502 | 0.8734 | 4.0% | 2.34 | 0 |
| Content-based (TF-IDF) | 0.0359 | — | **24.9%** | **6.76** | **481** |
| AutoRec (I-AutoRec) | 0.1432 | **0.8517** | 0.6% | 1.86 | 0 |
| Hybrid (SVD + Content, α=0.5) | 0.1605 | — | 5.5% | 2.49 | 6 |
| Hybrid (switching) | 0.1474 | — | 4.2% | 2.40 | 60 |
| **Nested Hybrid (SVD + AutoRec + Content)** | **0.1643** | — | 5.1% | 2.42 | 1 |

Cold-start reach counts items with zero train+val ratings appearing in a
top-10, summed over 6,100 recommended slots. Novelty@10 is mean
`-log2 P(item was rated)`: **1.19** for the single most-rated film, **7.13**
for a uniformly random recommender, **9.26** for an item nobody rated. Six of
the seven models sit between 1.65 and 2.49 — far below what picking at random
would score.

Two findings worth stating up front, because they shaped the project:

- **SVD alone is not reliably better than a non-personalized Most-Popular
  baseline on ranking**, even though it clearly beats the user-mean baseline
  on RMSE. On the reported split SVD loses at all 3 cutoffs; across four
  splits the gap averages +0.0012 NDCG@10 and **changes sign twice**
  (`scripts/seed_stability.py`), so the honest claim is that collaborative
  filtering alone cannot be separated from recommending best-sellers — not
  that it is strictly worse. The weighted hybrids, by contrast, beat plain
  SVD on every split (+0.0103 ± 0.0022).
- **AutoRec achieves near-CF ranking quality by collapsing onto
  popularity** — its catalogue coverage (0.6%) is identical to the naive
  baseline's, and Step 13's novelty measure sharpens that from "as narrow as"
  to "at the same end of the distribution": AutoRec scores 1.86 against
  Most-Popular's 1.65, on a scale where a random recommender scores 7.13. Its
  low RMSE is bought by predicting well on films almost everyone has already
  rated. Ranking metrics alone hide this entirely.

---

## Repository layout

```
core/                     Reusable model and evaluation classes
  data/preprocess.py      MovieLens + TMDB loading, 3-way per-user split
  models/svd.py           SVDRecommender  — matrix factorization baseline
  models/content.py       ContentBasedRecommender — TF-IDF over plot text
  models/autorec.py       AutoRecRecommender — I-AutoRec (PyTorch)
  models/hybrid.py        HybridRecommender — per-user z-score blending,
                          composable (a hybrid can nest inside a hybrid)
  evaluation/metrics.py   RMSE/MAE with honest coverage reporting
  evaluation/ranking.py   Precision@K / Recall@K / NDCG@K

notebooks/01_eda.ipynb    The experimental record: every grid search,
                          ablation, figure and results table in the report
                          comes from here. Steps 1-12.

scripts/                  Data fetching and batch scoring (see below)
api/main.py               FastAPI serving layer (reads precomputed JSON)
web/index.html            Single-page demo UI
docs/TIMELINE.md          Running progress log against the proposal
data/                     MovieLens ml-latest-small + TMDB overviews
```

The notebook is the deliverable for the *experiments*; `core/` holds the
implementations it calls; `api/` and `web/` are a serving demonstration
built on the selected model.

---

## Setup

Python 3.9 (developed on 3.9.6, macOS arm64).

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To run the serving layer as well:

```bash
pip install -r api/requirements.txt
```

---

## Running the project

### 1. Reproduce the experiments

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Restart the kernel and run all cells. Takes roughly 20-30 minutes, most of
it in the AutoRec hyperparameter grid (Step 9) and the alpha sweeps
(Steps 7 and 11). Every number in the report is produced here.

All randomness is seeded with the student ID (24265298), so the split and
all reported figures reproduce exactly.

### 2. Batch-score the selected model

```bash
python scripts/export_recommendations.py
```

Fits the deployed Nested Hybrid at the hyperparameters the notebook
selected on validation, scores all 610 users, and writes
`data/recommendations_export.json` (~2.5 MB). Takes a few minutes — this
is the only step that needs PyTorch.

### 3. Serve it

```bash
uvicorn api.main:app --reload
```

Then open <http://127.0.0.1:8000/>.

| Endpoint | Returns |
|---|---|
| `GET /` | The demo page |
| `GET /health` | Model name, generation time, user count |
| `GET /users` | All available userIds |
| `GET /recommend/{user_id}?n=10` | Top-n recommendations. An unknown userId returns 200 with `is_fallback: true` and the Most-Popular baseline, not 404 |
| `GET /profile/{user_id}?n=8` | That user's own highest-rated movies |

The serving process never imports PyTorch or scikit-learn — it only reads
the JSON that step 2 wrote. That split is deliberate: batch job fits and
predicts, serving tier is a thin, fast reader with no training framework
in the request path.

The demo page also takes a free-text userId, so entering one the batch job
never scored (611, say) shows the cold-start fallback and the banner
explaining what is being served and why.

The demo shows each user's own top-rated movies beside the model's
recommendations. Where a recommended movie carries a "they rated this ★"
badge, the user really did rate it — in the held-out test split the model
was never fit on.

---

## Scripts

| Script | Purpose |
|---|---|
| `export_recommendations.py` | Batch-score the deployed model to JSON (step 2 above) |
| `fetch_overviews.py` | Fetch TMDB plot overviews. Needs `TMDB_API_KEY` in the environment. Raised text coverage from 36.4% to 98.8% of the rated catalogue |
| `seed_stability.py` | Re-run the seven-model comparison across four splits and report which claims survive all of them (~80s) |
| `qualitative_snapshot.py` | Print each model's top-5 for four sample users with the rating count of every recommended film — the source of Table 10 in the report (~40s) |

`fetch_overviews.py` only needs re-running to rebuild
`data/overview_plot.csv` from scratch; the file it produces is already
included. It is resumable and skips overviews already present.

---

## Data

- **MovieLens `ml-latest-small`** — 100,836 ratings, 610 users, 9,724
  rated movies, 98.3% sparse. Included under `data/ml-latest-small/`; see
  its own `README.txt` for the usage licence.
- **TMDB plot overviews** — `data/overview_plot.csv`, 10,884 rows covering
  9,603 of the 9,724 rated movies (98.8%). Fetched with
  `scripts/fetch_overviews.py`.
- **TMDB plot overviews, pre-expansion** — `data/overview_plot_before_expansion.csv`,
  the original fetch, covering 3,536 of the 9,724 rated movies (36.4%).
  Kept as a tracked input because Appendix A of the report measures the
  corpus expansion against it. It cannot be regenerated: re-running the
  fetcher returns today's TMDB data, not the September snapshot.

> This product uses the TMDB API but is not endorsed or certified by TMDB.

Splitting is 60/20/20 per user: hyperparameters are selected on
validation, models are refit on train+val for deployment, and the test set
is evaluated exactly once.

---

## Dependencies

See `requirements.txt` (core project) and `api/requirements.txt` (serving
layer only). Principal versions used:

| Package | Version |
|---|---|
| python | 3.9.6 |
| numpy | 2.0.2 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| scikit-learn | 1.6.1 |
| torch | 2.8.0 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |

`venv/` is intentionally not included in the submitted archive.

---

## Limitations and future work

These are the boundaries of what this project measured, stated so the
results are read at the strength they actually support.

### Serving

- **No orchestration or refresh layer.** `data/recommendations_export.json`
  is a static snapshot. Nothing schedules a refit, versions successive
  exports, or detects a stale one — `/health` reports `generated_at` but no
  consumer acts on it. The batch/serving split itself is in place; the
  scheduler and model registry that would sit above it are out of scope.
- **Cold users get the baseline, not personalization.** The export covers
  exactly the 610 users in `ml-latest-small`. `/recommend/611` no longer
  returns 404 — it returns 200 with `is_fallback: true` and the
  non-personalized Most-Popular list, which is the *same* baseline the
  seven-model comparison measures, so the fallback's quality is a reported
  number (test NDCG@10 0.1549) rather than an unquantified guess. What is
  still missing is the other half: nothing scores a genuinely new user on
  demand, so that user stays on the baseline until the next batch run. The
  route that would close this gap — a near-line service building a TF-IDF
  profile from a new user's first few interactions and blending it through
  the existing z-score machinery with the collaborative weight pinned to
  zero — is sketched in "Next steps" below, not implemented.

### Data

- **121 of the 9,724 rated movies still have no plot text** — 113 have no
  synopsis on TMDB at all, and 8 carry no `tmdbId`. Coverage is 98.8%, not
  100%, and the remainder is not fetchable.
- **One dataset, one domain.** Everything here is MovieLens
  `ml-latest-small`: 610 users, 100,836 ratings, 98.3% sparse. Conclusions
  about model capacity in particular are tied to that scale — the AutoRec
  grid showed validation RMSE improving with hidden size up to k=200, which
  is a statement about this dataset, not about I-AutoRec in general.

### Evaluation

- **Offline only, and that shapes the results.** Expanding text coverage
  from 36.4% to 98.8% tripled the content model's cold-start reach
  (163 → 481 recommended slots) while its NDCG@10 *fell* (0.0469 → 0.0359).
  Cold items make up only 2.92% of the relevant held-out ratings, so
  promoting one costs a slot a popular item would more often have
  converted. Offline ranking metrics can only credit items someone already
  rated; they systematically penalise discovery. An online test is the only
  way to tell whether that trade is worth making.
- **Coverage is a blunt instrument, so novelty is measured too.** Catalogue
  coverage counts how many *different* items a model reached; it scores
  recommending the 3rd-most-rated film and an unrated obscurity identically.
  Step 13 adds Novelty@10 (mean `-log2 P(item was rated)`) and Serendipity@10
  (Precision@10 discounted by item popularity) so "reached wide" and "reached
  *deep* into the tail" stop being the same claim. Both are reported against
  two reference lines — the novelty a uniformly random recommender would score,
  and the ceiling an unrated item scores — so the numbers read on a stated
  scale instead of as bare figures. Measured: the content model scores 6.76,
  every other model 1.65–2.49, against a random-recommender line of 7.13. The
  six accuracy-oriented models are not merely *somewhat* popularity-biased;
  they occupy the bottom quarter of the scale. Serendipity@10 tells the same
  story from the accuracy side — the share of each model's Precision@10 that
  survives a popularity discount runs from 0.678 (Most-Popular) through 0.702
  (AutoRec) to 0.851 (content).
- **Coverage and accuracy point at different winners.** Across six measures —
  lowest RMSE, highest NDCG@10, highest Novelty@10, highest Serendipity@10,
  widest coverage, most cold-start reach — **three different models** hold the
  six titles. AutoRec has the lowest rating error (0.8517) while recommending
  from 0.6% of the catalogue; the Nested Hybrid ranks best (0.1643) and is the
  most serendipitous (0.0900); the content model is the most diverse (24.9%),
  the most novel (6.76), reaches the most cold items (481) and is the worst
  ranker (0.0359). Reporting a single "best model" would hide all of it.
- **The noise floor is measured, not assumed.**
  `scripts/seed_stability.py` re-runs the whole seven-model comparison on four
  different 60/20/20 splits (the student-ID seed plus 1, 42 and 12345, fixed
  in advance). Any single model's NDCG@10 moves by up to **0.0123** between
  splits, so two models closer together than that on one split cannot be
  called separable from that split alone.

  Comparing *pairwise within each split* is the sharper test, because all
  seven models share the split and a generous split lifts them together —
  differencing cancels that shared movement. On that test:

  | Ordering | mean diff | std | Verdict |
  |---|---|---|---|
  | Nested Hybrid > Hybrid (weighted) | 0.0035 | 0.0008 | holds on 4/4 |
  | Hybrid (weighted) > SVD | 0.0103 | 0.0022 | holds on 4/4 |
  | Nested Hybrid > Most-Popular | 0.0150 | 0.0076 | holds on 4/4 |
  | Most-Popular > AutoRec | 0.0073 | 0.0050 | holds on 4/4 |
  | SVD > Hybrid (switching) | 0.0025 | 0.0006 | holds on 4/4 |
  | SVD vs Most-Popular | 0.0012 | 0.0060 | **sign flips — not separable** |

  Two things follow. The switching hybrid *consistently* ranks below plain
  SVD while reaching 60 cold-start slots to SVD's 0 — the accuracy/coverage
  trade-off in a single pair, separable in both directions. And the coverage
  and novelty findings are far more stable than the ranking ones: Content's
  coverage is 24.9% ± 0.2% and its novelty 6.74 ± 0.03 across all four
  splits, while AutoRec's zero-variance predictions on cold items hold
  exactly, as the mathematics requires.

  Caveat: hyperparameters are held at the values selected on the student-ID
  validation split, so this measures split variance with the model fixed, not
  selection variance as well.
- **Selection noise is visible at this scale.** The nested-hybrid grid's
  unconstrained winner was a degenerate corner (alpha1=0, alpha2=1, i.e.
  AutoRec alone) that beat the best genuine three-model blend by 0.0006
  NDCG on ~598 validation users — and then lost to it on test, 0.1432
  against 0.1643. With grids this fine and a validation set this small,
  differences in the fourth decimal should not be read as real.

### Next steps

1. An online or interleaved evaluation, to test whether the coverage the
   content model buys is worth the offline ranking it costs. Novelty and
   serendipity are offline proxies for it, not a substitute: they measure
   how far into the tail a list reaches, not whether anyone watched.
2. **Personalized** cold-start, not just the baseline fallback now in place.
   A near-line service could build a temporary TF-IDF profile from a new
   user's first few interactions and push it through the existing hybrid
   with the collaborative weight pinned to zero — reusing the components
   already built and measured here, rather than adding a model.
3. Repeated splits for *hyperparameter selection*, not just for reporting.
   `scripts/seed_stability.py` now measures how far the final comparison
   moves across splits, but the hyperparameters themselves are still chosen
   on one validation set; re-running the grids per split would capture
   selection variance too, and would remove the mild optimism that comes
   from having chosen them on data that lands in another split's test set.
4. Feeding content features into the deep model itself rather than blending
   them afterwards — a two-tower design, or injecting the text cosine
   similarities as extra inputs to AutoRec's hidden layer. The cold items
   here are not featureless: all 732 have plot text, and AutoRec simply
   cannot see it, which is why its predictions on them have zero variance.
   The trade-off is real in both directions, though: late fusion is what
   makes each component separately measurable, and it is what let this
   project show that AutoRec collapses onto popularity while the content
   model does not. An early-fusion model would rank better or worse as one
   number, with no way to attribute the difference.
