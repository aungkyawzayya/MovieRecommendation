"""
Figure 2 - batch / serving split.

The point of the figure is the dashed line down the middle: the request path
never imports a training framework. That is the claim the old proposal diagram
could not make, because it drew a live feedback loop and a periodic retraining
trigger that were never built.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def tint(hex_colour, amount=0.88):
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(c + (1 - c) * amount for c in (r, g, b))

OFF, STORE, ON, COLD = "#8A6BA8", "#5B7C99", "#3E8E7E", "#C08A2E"

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(2, 98); ax.axis("off")

def box(x, y, w, h, title, sub=None, colour="#555555", fs=8.0, subfs=6.5,
        dy=1.3, dash=False, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.5",
                                facecolor=tint(colour), edgecolor=colour,
                                linewidth=1.15, zorder=z,
                                linestyle="--" if dash else "-"))
    ax.text(x + w / 2, y + h / 2 + (dy if sub else 0), title, ha="center",
            va="center", fontsize=fs, fontweight="bold", color="#1a1a1a", zorder=z + 1)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                fontsize=subfs, color="#333", zorder=z + 1, linespacing=1.4)

def arrow(x1, y1, x2, y2, colour="#666666", rad=0.0, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=9, linewidth=lw,
                                 color=colour, linestyle=ls, zorder=8))

# ------------------------------------------------------- the dividing line
ax.plot([50, 50], [6, 92], linestyle=(0, (5, 4)), color="#999", linewidth=1.3, zorder=1)
ax.text(50, 95.5, "no PyTorch or scikit-learn past this line",
        ha="center", va="center", fontsize=7.2, style="italic", color="#555",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", lw=0.8), zorder=9)
ax.text(24, 88.5, "OFFLINE  ·  batch", ha="center", fontsize=8.4,
        fontweight="bold", color=OFF)
ax.text(76, 88.5, "ONLINE  ·  request path", ha="center", fontsize=8.4,
        fontweight="bold", color=ON)

# ------------------------------------------------------------------ offline
box(3, 70, 42, 12, "notebooks/01_eda.ipynb",
    "Steps 1-13: grid searches, ablations,\nevery figure and number in the report",
    OFF, dy=1.9)
box(3, 52, 42, 12, "scripts/export_recommendations.py",
    "refits the Nested Hybrid on train+val,\nscores all 610 users (needs torch)",
    OFF, dy=1.9)
arrow(24, 70, 24, 64.6, colour=OFF)
ax.text(26.5, 67.3, "selected hyperparameters", ha="left", va="center",
        fontsize=6.3, color="#555")

box(3, 32, 42, 13, "data/recommendations_export.json",
    "top-20 per user  ·  each user's own ratings\nMost-Popular cold-start fallback list",
    STORE, dy=2.1)
arrow(24, 52, 24, 45.6, colour=OFF)

# ------------------------------------------------------------------- online
box(55, 64, 42, 12, "api/main.py   (FastAPI)",
    "RecommendationService loads the JSON once;\na lookup is a dict hit",
    ON, dy=1.9)
box(55, 46, 42, 11, "web/index.html",
    "user picker, held-out-hit badges,\nfallback banner", ON, dy=1.9)
arrow(76, 64, 76, 57.6, colour=ON)

arrow(45, 40, 54.3, 71, colour=STORE, rad=-0.20, lw=1.3)
# ---------------------------------------------------------- cold-start path
box(55, 10, 42, 22, "Unknown user   (e.g. userId 611)",
    "never scored by the batch job, so there is nothing\n"
    "to look up. Returns 200 with is_fallback: true and\n"
    "the Most-Popular list - a baseline whose quality is\n"
    "already measured (test NDCG@10 0.1549), not a 404.",
    COLD, dy=6.0, subfs=6.4)
arrow(76, 46, 76, 32.6, colour=COLD, ls="--")

plt.tight_layout(pad=0.3)
plt.savefig("fig2_serving.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("fig2_serving.pdf", bbox_inches="tight", facecolor="white")
print("rendered fig2")
