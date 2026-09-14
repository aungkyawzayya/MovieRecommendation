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
        # {userId_str: {"n_ratings", "mean_rating", "top_rated": [rows]}} — a
        # user's OWN highest-rated movies, from the full ratings.csv. Absent
        # from exports written before this field existed (older
        # recommendations_export.json on disk); .get() below handles that.
        self._profiles = data.get("profiles", {})
        # Most-Popular top-N, served to users this export never scored. .get()
        # with a default rather than data["..."]: an export written before this
        # field existed must still load, and has_cold_start_fallback below tells
        # the routes which behaviour they can offer.
        self._cold_start_fallback = data.get("cold_start_fallback", [])
        self._cold_start_model = data.get("cold_start_model", "Most-Popular (non-personalized baseline)")

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

    @property
    def cold_start_model(self):
        return self._cold_start_model

    @property
    def has_cold_start_fallback(self):
        """False for exports written before the fallback list existed."""
        return bool(self._cold_start_fallback)

    def knows_user(self, user_id: int) -> bool:
        """Whether this user was scored by the batch job at all."""
        return str(user_id) in self._recommendations

    def get_top_n(self, user_id: int, n: int = 10):
        """Top-n rows for one user, or None if this user isn't in the export."""
        rows = self._recommendations.get(str(user_id))
        if rows is None:
            return None
        return rows[:n]

    def get_cold_start(self, n: int = 10):
        """
        Top-n of the non-personalized Most-Popular list — what a user the
        batch job never scored gets served.

        This is the standing cost of precomputing: lookups are a dict hit, but
        only for users scored in advance. Returning 404 to everyone else made
        that cost look like a crash. Serving the baseline instead keeps the
        request path free of any model while still answering, and because the
        fallback is the SAME Most-Popular baseline the notebook evaluates, its
        quality is a measured number (test NDCG@10 0.1549) rather than a guess.
        """
        return self._cold_start_fallback[:n]

    def get_profile(self, user_id: int, n: int = 8):
        """
        This user's own top-n highest-rated movies (real ratings, not model
        output), or None if this user isn't in the export. Lets the web demo
        show "what they actually loved" next to "what the model recommends" —
        the difference between users is otherwise invisible from the
        recommendation list alone.
        """
        profile = self._profiles.get(str(user_id))
        if profile is None:
            return None
        return {**profile, "top_rated": profile["top_rated"][:n]}


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
        "cold_start_fallback": service.cold_start_model if service.has_cold_start_fallback else None,
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
    if rows is not None:
        return {
            "userId": user_id,
            "model": service.model_name,
            "is_fallback": False,
            "recommendations": rows,
        }

    # Unknown user: serve the cold-start baseline rather than an error. 200 with
    # is_fallback=True, not 404 and not a silent substitution — a caller that
    # ignores the flag still gets a usable list, and one that reads it can tell
    # the two apart. Personalization returns for this user on the next batch run.
    if not service.has_cold_start_fallback:
        raise HTTPException(
            404,
            f"No recommendations for userId={user_id}, and this export predates "
            f"the cold-start fallback — re-run scripts/export_recommendations.py.",
        )
    return {
        "userId": user_id,
        "model": service.cold_start_model,
        "is_fallback": True,
        "fallback_reason": (
            f"userId={user_id} was not scored by the last batch run "
            f"(generated {service.generated_at}, covering "
            f"{len(service.user_ids)} users). Serving the non-personalized "
            f"Most-Popular baseline until the next run scores this user."
        ),
        "recommendations": service.get_cold_start(n),
    }


@app.get("/profile/{user_id}")
def profile(user_id: int, n: int = 8):
    """
    This user's own top-n highest-rated movies — real ratings from
    ratings.csv, not a model prediction. The web demo shows this beside
    /recommend's output so the personalization is visible: two users with
    very different tastes get very different top-10 lists, and this is the
    real data explaining why.
    """
    if n < 1:
        raise HTTPException(400, "n must be >= 1")
    result = service.get_profile(user_id, n)
    if result is None:
        raise HTTPException(
            404,
            f"No rating history for userId={user_id} — this export only covers "
            f"the {len(service.user_ids)} users in ml-latest-small. (/recommend "
            f"still answers for unknown users, with the cold-start fallback; a "
            f"profile cannot be faked the same way — a new user genuinely has "
            f"no ratings yet.)",
        )
    return {"userId": user_id, **result}


# Serves web/index.html at "/" — same-origin, so the page's fetch("/recommend/...")
# calls need no CORS setup at all. Mounted LAST: StaticFiles(html=True) would
# otherwise swallow every path (including /recommend/...) before it reaches
# the routes above.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
