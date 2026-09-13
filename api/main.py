"""
api/main.py
Serving layer for the batch-precomputed Nested Hybrid recommendations.

scripts/export_recommendations.py does the actual model fitting/scoring
(needs torch, sklearn, a few minutes); this process never imports either —
it only ever reads the plain JSON that script writes. That split mirrors
how a real recommender system is usually deployed: a batch/offline job
periodically (re)fits the model and writes predictions to a lookup table,
and the serving tier is a thin, fast reader of that table with no training
framework loaded into the request path at all. Nobody's end user runs a
notebook — this is what they'd actually hit.

Run (from the repo root, with fastapi+uvicorn installed — see requirements
below — and AFTER running scripts/export_recommendations.py at least once):
    pip install fastapi uvicorn
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/ for the demo page, or call
GET /recommend/{user_id}?n=10 directly for JSON.
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_PATH = PROJECT_ROOT / "data" / "recommendations_export.json"
WEB_DIR = PROJECT_ROOT / "web"


class RecommendationService:
    """
    Owns the precomputed recommendations table and answers lookups against
    it — a private dict behind a small public interface, same encapsulation
    convention as the core/ models (C# analogy: a singleton service class
    with a private backing field, injected into route handlers below
    instead of each one re-opening the JSON file itself).
    """

    def __init__(self, export_path: Path):
        if not export_path.exists():
            raise FileNotFoundError(
                f"{export_path} not found — run scripts/export_recommendations.py "
                f"first (needs the project's real venv, with torch) to generate it."
            )
        with open(export_path, encoding="utf-8") as f:
            data = json.load(f)
        self._model_name = data["model"]
        self._generated_at = data["generated_at"]
        self._top_n_per_user = data["top_n_per_user"]
        self._recommendations = data["recommendations"]  # {userId_str: [rows]}

    @property
    def model_name(self):
        return self._model_name

    @property
    def generated_at(self):
        return self._generated_at

    @property
    def max_n(self):
        """The deepest list export_recommendations.py actually computed."""
        return self._top_n_per_user

    @property
    def user_ids(self):
        """Sorted int userIds this export has recommendations for."""
        return sorted(int(uid) for uid in self._recommendations)

    def get_top_n(self, user_id: int, n: int = 10):
        """Top-n rows for one user, or None if this user isn't in the export."""
        rows = self._recommendations.get(str(user_id))
        if rows is None:
            return None
        return rows[:n]


service = RecommendationService(EXPORT_PATH)

app = FastAPI(
    title="Movie Recommendation API",
    description=(
        f"Serves precomputed recommendations from '{service.model_name}' "
        f"(generated {service.generated_at})."
    ),
)


@app.get("/health")
def health():
    """Liveness/readiness check — also reports which model this instance is serving."""
    return {
        "status": "ok",
        "model": service.model_name,
        "generated_at": service.generated_at,
        "n_users": len(service.user_ids),
    }


@app.get("/users")
def list_users():
    """All userIds the demo page can offer in its dropdown."""
    return {"user_ids": service.user_ids}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    """
    Top-n recommended movies for one MovieLens user.
    n is capped at the export's own top_n_per_user — asking for more than
    was ever batch-scored would silently return fewer rows than requested,
    so that is refused explicitly instead.
    """
    if n < 1:
        raise HTTPException(400, "n must be >= 1")
    if n > service.max_n:
        raise HTTPException(
            400,
            f"n={n} exceeds this export's top_n_per_user={service.max_n} — "
            f"re-run scripts/export_recommendations.py with a higher TOP_N first.",
        )
    rows = service.get_top_n(user_id, n)
    if rows is None:
        raise HTTPException(
            404,
            f"No recommendations for userId={user_id} — this export only covers "
            f"the {len(service.user_ids)} users in ml-latest-small.",
        )
    return {"userId": user_id, "model": service.model_name, "recommendations": rows}


# Serves web/index.html at "/" — same-origin, so the page's fetch("/recommend/...")
# calls need no CORS setup at all. Mounted LAST: StaticFiles(html=True) would
# otherwise swallow every path (including /recommend/...) before it reaches
# the routes above.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
