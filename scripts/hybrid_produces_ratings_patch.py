"""
hybrid_produces_ratings_patch.py
Adds RawScoreAdapter.produces_ratings = False for defensive interface
completeness (evaluate_model() falls back to True via getattr when the
attribute is absent, which would be wrong for an adapter's raw/unclipped
scores if it were ever passed there directly - not currently the case,
but cheap to guard against).

Run from the repo root:
    python scripts/hybrid_produces_ratings_patch.py
"""
import pathlib

path = pathlib.Path(__file__).resolve().parents[1] / "core/models/hybrid.py"
text = path.read_text(encoding="utf-8")

old = '''    Composition, not inheritance: this is not a subtype of the wrapped
    model, just a thin forwarding shim exposing the interface
    HybridRecommender's content-slot code actually calls.
    """

    def __init__(self, model):
        self._model = model'''

new = '''    Composition, not inheritance: this is not a subtype of the wrapped
    model, just a thin forwarding shim exposing the interface
    HybridRecommender's content-slot code actually calls.
    """

    # Not read by anything in this file (only score_all_items/supported_items/
    # etc. are), but evaluate_model() does getattr(model, "produces_ratings",
    # True) - without this, an adapter passed there by mistake would default
    # to True and report a meaningless RMSE against its own unclipped scores.
    produces_ratings = False

    def __init__(self, model):
        self._model = model'''

assert text.count(old) == 1, "pattern not found or not unique"
text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print("patched:", path)
