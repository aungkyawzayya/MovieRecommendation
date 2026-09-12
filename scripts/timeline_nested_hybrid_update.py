"""
timeline_nested_hybrid_update.py
Rewrites the Nested Hybrid TIMELINE.md row to the honest conclusion after
independent review: validation NDCG improved but test NDCG regressed and
sign-flipped (0.1192->0.1244 val, 0.1616->0.1590 test) - consistent with
validation overfitting from a second round of hyperparameter selection
(36 alpha combos) stacked on AutoRec's own 24-config selection, both
against the same ~598-user validation split. The two-model Hybrid remains
the best-performing ranking model; the nested composition is kept as a
documented ablation/critical-analysis result, not presented as a win.

Run from the repo root:
    python scripts/timeline_nested_hybrid_update.py
"""
import pathlib

path = pathlib.Path(__file__).resolve().parents[1] / "docs/TIMELINE.md"
text = path.read_text(encoding="utf-8")

old = (
    "| 12 Sep | Nested Hybrid (SVD + AutoRec, then + Content, via composition) — "
    "notebook Steps 11/12: `inner=Hybrid(svd,autorec)` then `full=Hybrid(inner,content)`, "
    "exactly as planned, but tracing the actual interface calls found the plan's "
    '"no code changes needed" claim was wrong on two counts fixed in `hybrid.py` '
    "(`score_all_items` had no `clip` parameter, so nesting raised `TypeError`; "
    "`item_ids` was missing, so `rankable_items` raised `AttributeError`) plus a "
    "third, quieter one (the content slot calls with no `clip` arg, so AutoRec's own "
    "`clip=True` default would have silently flattened its scores) fixed with a new "
    "`RawScoreAdapter` — no change to the blend logic or either ablation. "
    "Validation-selected alpha1=0.2, alpha2=0.6 beats the two-model Hybrid's own "
    "validation NDCG@10 (0.1244 vs 0.1192), and at alpha1=0/alpha2=1 the nested model "
    "reproduces AutoRec's own validation NDCG@10 exactly (0.1196) - confirming the "
    "composition is doing real arithmetic, not silently falling back to one "
    "component. On test: NDCG@10 0.1590, between AutoRec alone (0.1432) and the "
    "plain weighted Hybrid (0.1616); catalogue coverage 6.0%, the best of all seven "
    "models (plain Hybrid: 5.8%), with mean recommendation popularity down to 122.5 "
    "(plain Hybrid: 131.8) - adding AutoRec measurably reduces popularity bias "
    "without beating the simpler Hybrid on ranking | `e30c53b` |\n"
)
assert text.count(old) == 1, "row not found or not unique"

new = (
    "| 12 Sep | Nested Hybrid (SVD + AutoRec, then + Content, via composition) — "
    "notebook Steps 11/12: `inner=Hybrid(svd,autorec)` then `full=Hybrid(inner,content)`, "
    "exactly as planned, but tracing the actual interface calls found the plan's "
    '"no code changes needed" claim was wrong on two counts fixed in `hybrid.py` '
    "(`score_all_items` had no `clip` parameter, so nesting raised `TypeError`; "
    "`item_ids` was missing, so `rankable_items` raised `AttributeError`) plus a "
    "third, quieter one (the content slot calls with no `clip` arg, so AutoRec's own "
    "`clip=True` default would have silently flattened its scores) fixed with a new "
    "`RawScoreAdapter` (`produces_ratings = False` added defensively too) — no "
    "change to the blend logic or either ablation. Three independent boundary "
    "checks confirm the composition computes real arithmetic rather than silently "
    "collapsing to one input: alpha2=0.0 reproduces Content-only (0.0400, Step 7), "
    "alpha1=alpha2=1.0 reproduces SVD-only (0.1060, Step 7), and alpha1=0/alpha2=1 "
    "reproduces AutoRec's own validation NDCG@10 exactly (0.1196, Step 9). **Honest "
    "result: this did not improve on the two-model Hybrid.** Validation NDCG@10 rose "
    "(0.1192 -> 0.1244 at alpha1=0.2, alpha2=0.6) but test NDCG@10 fell and the sign "
    "flipped (0.1616 -> 0.1590), and the Most-Popular win-count dropped (3/3 -> 2/3) "
    "- consistent with validation overfitting, not a real gain: the (alpha1, alpha2) "
    "grid (36 cells) is a second round of hyperparameter selection stacked on top of "
    "AutoRec's own 24-config selection, both against the same ~598 validation users. "
    "**The two-model Hybrid (SVD+Content) remains the best-performing ranking "
    "model** and is what the report should recommend; the nested composition is kept "
    "as a documented ablation for the requirements' \"ablation studies\" / "
    "\"critically analyze findings\" asks, not as an improvement. It does show a real, "
    "separate trade-off worth reporting: best-of-all-seven catalogue coverage (6.0% "
    "vs plain Hybrid's 5.8%) and lowest popularity bias (mean pop. 122.5 vs 131.8), "
    "but at the cost of losing the plain Hybrid's cold-start reach entirely (0 vs 6 "
    "cold items in top-10) - AutoRec's own popularity collapse (0.6% coverage) "
    "cancels out content's cold-item boost once blended in | "
    "`e30c53b`, `1b32d5d`, `<pending>` |\n"
)

text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print("patched:", path)
