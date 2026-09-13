"""
docstring_coverage_refresh.py
Self-directed code-quality pass (no behaviour change): after the TMDB
overview expansion (36.4% -> 98.8% plot-text coverage, commits 630b54f/
5fa5c82/5bd40e8), two docstrings elsewhere in core/ still quoted the old
"36%" figure as if it were still current, which now reads as wrong to
anyone opening those files rather than the notebook/TIMELINE. This updates
both to state the CURRENT coverage while keeping each comment's actual
point (why the class exists / why precision must divide by k) intact,
and adds a short "measured, not assumed" pointer to content.py's own
docstring instead of duplicating its numbers.

Run from the repo root:
    python scripts/docstring_coverage_refresh.py
"""
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]

# --- hybrid.py: module docstring's "36%" was the pre-expansion figure ---
hybrid_path = root / "core/models/hybrid.py"
hybrid_text = hybrid_path.read_text(encoding="utf-8")

old_hybrid = (
    '''Combines SVD (collaborative filtering) and content-based (plot-text) scores
into a single ranked list. Neither model alone is enough: SVD collapses to
the user-mean baseline for items with zero train ratings (Step 6's
cold-start evidence), and the content model only has plot text for 36% of
the catalogue (Step 5) — this class exists to cover what either one misses
on its own.'''
)
assert hybrid_text.count(old_hybrid) == 1, "hybrid.py module docstring not found or not unique"

new_hybrid = (
    '''Combines SVD (collaborative filtering) and content-based (plot-text) scores
into a single ranked list. Neither model alone is enough: SVD collapses to
the user-mean baseline for items with zero train ratings (Step 6's
cold-start evidence), and even after the TMDB overview expansion the
content model still has no opinion on 121 movies with no plot text at all
(content.py's own docstring has the coverage numbers) — this class exists
to cover what either one misses on its own.'''
)
hybrid_text = hybrid_text.replace(old_hybrid, new_hybrid)
hybrid_path.write_text(hybrid_text, encoding="utf-8")
print("patched:", hybrid_path)

# --- ranking.py: precision_at_k docstring's "36%" example was also stale ---
ranking_path = root / "core/evaluation/ranking.py"
ranking_text = ranking_path.read_text(encoding="utf-8")

old_ranking = (
    '''Denominator is k, NOT len(top_k). Dividing by the list length rewards a
    model for returning a SHORT list: 1 hit out of 3 returned items scored
    0.3333 instead of the correct 0.1000 — a 3.3x overstatement. Short lists
    happen in practice (the content model can only rank the 36% of movies
    that have plot text, then drops the ones the user already rated), so
    unfilled slots must count against the model, not be quietly excluded.'''
)
assert ranking_text.count(old_ranking) == 1, "ranking.py docstring not found or not unique"

new_ranking = (
    '''Denominator is k, NOT len(top_k). Dividing by the list length rewards a
    model for returning a SHORT list: 1 hit out of 3 returned items scored
    0.3333 instead of the correct 0.1000 — a 3.3x overstatement. Short lists
    happen in practice (the content model can only rank movies with plot
    text — 98.8% of the catalogue now, but never all of it — then drops the
    ones the user already rated), so unfilled slots must count against the
    model, not be quietly excluded.'''
)
ranking_text = ranking_text.replace(old_ranking, new_ranking)
ranking_path.write_text(ranking_text, encoding="utf-8")
print("patched:", ranking_path)
