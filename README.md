# Hybrid Movie Recommendation System

COMP813 Artificial Intelligence 2026 — Final Project
Aung Kyaw Zay Ya (24265298), Auckland University of Technology

A hybrid movie recommender combining collaborative filtering (SVD), a deep
model (I-AutoRec), and content-based NLP (TF-IDF over plot overviews),
evaluated on MovieLens `ml-latest-small` with TMDB plot text.

Seven models are built and compared on the same held-out test split, on
both rating-error and ranking metrics, plus coverage and popularity-bias
measures. The headline result is that no single model wins everything:

| Model | Test NDCG@10 | Test RMSE | Catalogue coverage | Cold-start reach |
|---|---|---|---|---|
| User-mean baseline | — | 0.9448 | — | — |
| Most-Popular (no personalization) | 0.1549 | — | 0.6% | 0 |
| SVD (collaborative filtering) | 0.1502 | 0.8734 | 4.0% | 0 |
| Content-based (TF-IDF) | 0.0359 | — | **24.9%** | **481** |
| AutoRec (I-AutoRec) | 0.1432 | **0.8517** | 0.6% | 0 |
| Hybrid (SVD + Content, α=0.5) | 0.1605 | — | 5.5% | 6 |
| Hybrid (switching) | 0.1474 | — | 4.2% | 60 |
| **Nested Hybrid (SVD + AutoRec + Content)** | **0.1643** | — | 5.1% | 1 |

Cold-start reach counts items with zero train+val ratings appearing in a
top-10, summed over 6,100 recommended slots.

Two findings worth stating up front, because they shaped the project:

- **SVD alone does not beat a non-personalized Most-Popular baseline on
  ranking** (0/3 cutoffs), even though it clearly beats the user-mean
  baseline on RMSE. Only the hybrids beat Most-Popular on ranking.
- **AutoRec achieves near-CF ranking quality by collapsing onto
  popularity** — its catalogue coverage (0.6%) is identical to the naive
  baseline's. Ranking metrics alone hide this; it is visible only because
  coverage was measured alongside them.

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

scripts/                  One-off utilities (see "Scripts" below)
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
| `GET /recommend/{user_id}?n=10` | Top-n recommendations |
| `GET /profile/{user_id}?n=8` | That user's own highest-rated movies |

The serving process never imports PyTorch or scikit-learn — it only reads
the JSON that step 2 wrote. That split is deliberate: batch job fits and
predicts, serving tier is a thin, fast reader with no training framework
in the request path.

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
| `autorec_sweep.py`, `autorec_sweep2.py` | Diagnostics that established AutoRec's training budget |

The remaining `scripts/*_patch.py` files are one-off migrations applied
during development; they are kept for provenance and do not need to be
re-run.

---

## Data

- **MovieLens `ml-latest-small`** — 100,836 ratings, 610 users, 9,724
  rated movies, 98.3% sparse. Included under `data/ml-latest-small/`; see
  its own `README.txt` for the usage licence.
- **TMDB plot overviews** — `data/overview_plot.csv`, 10,884 rows covering
  9,603 of the 9,724 rated movies (98.8%). Fetched with
  `scripts/fetch_overviews.py`.

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
