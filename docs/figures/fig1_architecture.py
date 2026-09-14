"""
Figure 1 - experimental architecture of the hybrid recommender.

Colours come from the notebook's own model palette so a reader who has seen
the results charts recognises SVD as blue, Content as green and AutoRec as
tan in both places. Fills are light tints of the same hue, which keeps the
figure readable when the report is printed in greyscale.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {"svd": "#4C72B0", "auto": "#B08E3C", "content": "#55A868",
     "nested": "#DD8452", "base": "#8C8C8C", "data": "#5B7C99",
     "eval": "#6B6B8C", "plain": "#555555"}


def tint(hex_colour, amount=0.88):
    """Lighten toward white. Accepts #abc and #aabbcc."""
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(c + (1 - c) * amount for c in (r, g, b))


fig, ax = plt.subplots(figsize=(7.2, 5.9))
ax.set_xlim(0, 100); ax.set_ylim(8, 99); ax.axis("off")


def box(x, y, w, h, title, sub=None, colour="#555555", lw=1.1, fs=8.0,
        subfs=6.6, z=2, dy=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.6",
                                facecolor=tint(colour), edgecolor=colour,
                                linewidth=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2 + (dy if sub else 0), title, ha="center",
            va="center", fontsize=fs, fontweight="bold", color="#1a1a1a", zorder=z + 1)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.1, sub, ha="center", va="center",
                fontsize=subfs, color="#333", zorder=z + 1, linespacing=1.4)


def arrow(x1, y1, x2, y2, colour="#666666", rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=colour, zorder=6))


# ------------------------------------------------------------------ data
box(4, 88, 42, 9.5, "MovieLens ml-latest-small",
    "610 users x 9,724 rated movies\n100,836 ratings, 98.3% sparse", C["data"])
box(54, 88, 42, 9.5, "TMDB plot overviews",
    "9,603 of 9,724 movies (98.8%)\njoined via links.csv tmdbId", C["data"])

# ------------------------------------------------------------ preprocess
box(20, 76, 60, 7, "Preprocessing",
    "user-item rating matrix  +  text corpus", C["plain"], dy=1.4)
arrow(25, 88, 40, 83.3)
arrow(75, 88, 60, 83.3)

# ----------------------------------------------------------------- split
box(20, 64, 60, 7.5, "Per-user 60 / 20 / 20 split     (seed = student ID)",
    "train: fit        validation: select hyperparameters        test: report ONCE",
    C["plain"], dy=1.5)
arrow(50, 76, 50, 71.7)

# ----------------------------------------------------------- base models
box(3, 50, 29, 9, "SVD", "scipy svds, k = 10\ndamped item bias", C["svd"])
box(35.5, 50, 29, 9, "I-AutoRec", "PyTorch autoencoder\nmasked MSE loss", C["auto"])
box(68, 50, 29, 9, "Content", "TF-IDF over plot text\n+ cosine similarity", C["content"])
arrow(40, 64, 17.5, 59.3, rad=0.10)
arrow(50, 64, 50, 59.3)
arrow(60, 64, 82.5, 59.3, rad=-0.10)

# -------------------------------------------- composition: literal nesting
ax.add_patch(FancyBboxPatch((3, 25.5), 61, 18.5,
                            boxstyle="round,pad=0.3,rounding_size=1.6",
                            facecolor=tint(C["nested"], 0.93), edgecolor=C["nested"],
                            linewidth=1.6, zorder=2))
ax.text(33.5, 41.5, "Nested Hybrid      (deployed model)", ha="center",
        va="center", fontsize=8.3, fontweight="bold", color="#1a1a1a", zorder=4)
ax.text(33.5, 38.6, r"per-user z-score blend,  $\alpha_2 = 0.6$", ha="center",
        va="center", fontsize=6.7, color="#333", zorder=4)

ax.add_patch(FancyBboxPatch((6, 27.5), 30, 8.5,
                            boxstyle="round,pad=0.25,rounding_size=1.3",
                            facecolor="white", edgecolor="#7a7a7a",
                            linewidth=1.1, linestyle="--", zorder=3))
ax.text(21, 33.9, "Inner Hybrid", ha="center", va="center", fontsize=7.5,
        fontweight="bold", color="#1a1a1a", zorder=4)
ax.text(21, 30.9, r"SVD $\oplus$ AutoRec,   $\alpha_1 = 0.4$", ha="center",
        va="center", fontsize=6.7, color="#333", zorder=4)

box(40, 27.5, 21, 8.5, "Content", "second signal", C["content"], fs=7.5,
    subfs=6.4, z=3, dy=1.1)

arrow(15, 49.2, 13, 44.4, colour=C["svd"], lw=1.3)
arrow(46, 49.2, 30, 44.4, colour=C["auto"], lw=1.3, rad=0.08)
arrow(78, 49.2, 52, 44.4, colour=C["content"], lw=1.3, rad=0.10)

box(68, 25.5, 29, 18.5, "Also compared",
    "Hybrid (weighted)\n"
    r"SVD $\oplus$ Content,  $\alpha = 0.5$"
    "\n\nHybrid (switching)\nSVD if warm, else Content"
    "\n\nMost-Popular baseline\nno personalization",
    C["base"], fs=7.7, subfs=6.3, dy=6.6)

# ------------------------------------------------------------ evaluation
arrow(33.5, 25.5, 33.5, 19.8, lw=1.3)
arrow(82.5, 25.5, 66, 19.8, rad=0.14)
box(3, 10, 94, 9.8, "Evaluation on the held-out test split",
    "Accuracy  RMSE / MAE          Ranking  Precision@K, Recall@K, NDCG@K          "
    "Beyond-accuracy  coverage, novelty, serendipity\n"
    "Responsible AI  popularity bias, cold-start reach          "
    "Robustness  the whole comparison repeated across 4 splits",
    C["eval"], fs=8.2, subfs=6.5, dy=1.6)

plt.tight_layout(pad=0.3)
plt.savefig("fig1_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("fig1_architecture.pdf", bbox_inches="tight", facecolor="white")
print("rendered")
