"""
step9_patch.py
Applies the findings from scripts/autorec_sweep.py and autorec_sweep2.py.

What the sweeps established:
  * full-batch, lr=1e-2, k=200 peaked at epoch 67 and is genuinely converged
    -> the notebook's 0.8616 was right all along.
  * patience=10 with ONE update per epoch is far too tight. 4 of the 6 k
    values in Step 9's grid stopped before converging: k=10 and k=100 hit the
    80-epoch cap outright, k=25 and k=50 early-stopped at best_epoch 17 and 8.
    With patience=200, k=10 keeps improving to epoch 347 and lands at
    val_RMSE 0.9028 instead of 0.9681, NDCG 0.0915 instead of 0.0382.
    The grid's k ordering — the whole capacity-vs-data-size argument — rests
    on those distorted rows.
  * mini-batching did NOT help (best 0.8815 vs full-batch 0.8616), so
    batch_size stays None here and the comparison becomes an ablation to
    report rather than a change to adopt.

Run from the repo root:
    python scripts/step9_patch.py
"""

import json
import pathlib

repo = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- autorec.py
p = repo / "core/models/autorec.py"
s = p.read_text()
old = 'epochs=300, patience=20, input_fill="zero", batch_size=256,'
new = 'epochs=300, patience=20, input_fill="zero", batch_size=None,'
assert old in s, "autorec.py: batch_size default not found"
s = s.replace(old, new)

old = "        # cost, so the SAME wall-clock budget buys ~38x more updates."
new = (
    "        # cost, so the SAME wall-clock budget buys ~38x more updates.\n"
    "        # Measured outcome: at this dataset's scale mini-batching did NOT\n"
    "        # help (best val RMSE 0.8815 at batch=256/lr=1e-3 vs 0.8616\n"
    "        # full-batch/lr=1e-2), because 38x more updates at the same lr\n"
    "        # overshoots into overfitting within 1-3 epochs. Default is None\n"
    "        # (full batch) for that reason; the option stays so the comparison\n"
    "        # can be reported as an ablation."
)
assert old in s, "autorec.py: batch comment not found"
s = s.replace(old, new)
p.write_text(s)
print("core/models/autorec.py: batch_size default -> None")

# ------------------------------------------------------------------ notebook
nbp = repo / "notebooks/01_eda.ipynb"
nb = json.loads(nbp.read_text())


def patch_cell(index, pairs):
    src = "".join(nb["cells"][index]["source"])
    for old_text, new_text in pairs:
        assert old_text in src, f"cell {index}: not found -> {old_text[:60]!r}"
        src = src.replace(old_text, new_text)
    nb["cells"][index]["source"] = src.splitlines(keepends=True)
    nb["cells"][index]["outputs"] = []
    nb["cells"][index]["execution_count"] = None


# --- Step 9 grid: give every config room to converge -------------------------
patch_cell(37, [(
    """        m = AutoRecRecommender(hidden_dim=k, lr=1e-2, weight_decay=wd,
                                epochs=80, patience=10, seed=STUDENT_ID)""",
    """        # epochs=2000, patience=200: with full-batch training an "epoch"
        # is a SINGLE gradient update, so epochs=80/patience=10 meant at most
        # 80 updates and a stop after 10 flat ones. Four of the six k values
        # never converged under that budget (k=10 and k=100 hit the cap with
        # best_epoch 79/80; k=25 and k=50 stopped at best_epoch 17 and 8) —
        # so the grid was ranking how far each config got, not its capacity.
        # Verified separately: k=10 keeps improving to epoch 347 and reaches
        # val_RMSE 0.9028 (not 0.9681), while the k=200 winner peaks at epoch
        # 67 either way — its 0.8616 was already correct.
        # batch_size=None (full batch) is deliberate: mini-batching was tested
        # and lost (0.8815 vs 0.8616); see scripts/autorec_sweep2.py.
        m = AutoRecRecommender(hidden_dim=k, lr=1e-2, weight_decay=wd,
                                epochs=2000, patience=200, batch_size=None,
                                seed=STUDENT_ID)"""
), (
    """        row = {"k": k, "weight_decay": wd, "val_RMSE": val_rmse, "val_NDCG@10": val_ndcg,
               "epochs_run": len(m.train_losses_), "best_epoch": m.best_epoch_}""",
    """        # stopped_by records WHY each run ended. "cap" means the epoch
        # budget cut it off while validation was still improving, so that
        # row's RMSE is a floor, not a result — exactly the failure that
        # distorted the first version of this grid. Any "cap" row below means
        # epochs needs raising before these numbers are quoted anywhere.
        epochs_run = len(m.train_losses_)
        row = {"k": k, "weight_decay": wd, "val_RMSE": val_rmse, "val_NDCG@10": val_ndcg,
               "epochs_run": epochs_run, "best_epoch": m.best_epoch_,
               "stopped_by": "cap" if epochs_run >= m.epochs else "peak"}"""
), (
    """print(f"Best by validation NDCG@10: k={best_ndcg['k']:4d}, weight_decay={best_ndcg['weight_decay']:.0e}  "
      f"(NDCG@10={best_ndcg['val_NDCG@10']:.4f})")""",
    """print(f"Best by validation NDCG@10: k={best_ndcg['k']:4d}, weight_decay={best_ndcg['weight_decay']:.0e}  "
      f"(NDCG@10={best_ndcg['val_NDCG@10']:.4f})")

n_capped = int((autorec_grid["stopped_by"] == "cap").sum())
print(f"\\nConfigs stopped by the epoch cap rather than a validation peak: "
      f"{n_capped}/{len(autorec_grid)}"
      + ("  <-- raise epochs before quoting these" if n_capped else "  (all converged)"))"""
)])
print("notebook cell 37: epochs 80 -> 2000, patience 10 -> 200, stopped_by column")

# --- Step 10: assign by index, not position ----------------------------------
patch_cell(41, [(
    """final_ndcg["AutoRec"] = [autorec_ranking[k][f"NDCG@{k}"] for k in Ks]
final_precision["AutoRec"] = [autorec_ranking[k][f"Precision@{k}"] for k in Ks]""",
    """# Assign as a Series keyed by K, not a bare list: a list lines up by
# POSITION, so it would silently mis-file the values if Ks were ever
# reordered relative to the table's index.
final_ndcg["AutoRec"] = pd.Series({k: autorec_ranking[k][f"NDCG@{k}"] for k in Ks})
final_precision["AutoRec"] = pd.Series({k: autorec_ranking[k][f"Precision@{k}"] for k in Ks})"""
)])
nb["cells"][42]["outputs"] = []
nb["cells"][42]["execution_count"] = None
print("notebook cell 41: positional list assignment -> indexed Series")

# --- Step 10 graph: the final figure has to show all six models --------------
graph_md = """### Step 10's Graph — Six-Model Final Comparison

Step 8's figure was drawn before AutoRec existed, so it shows five models
while Step 10's tables show six. This is the figure the report should use:
same three panels, every model included."""

graph_code = '''# Rebuild coverage for ALL models including AutoRec. final_rankers and
# final_reachable_pool both gained an "AutoRec" entry in the cell above, so
# Step 8's versions of these tables are now one model short — recompute here
# rather than leaving the report with a 5-model figure beside a 6-model table.
final_top10 = {
    name: {uid: fn(uid)[:10] for uid in all_users}
    for name, fn in final_rankers.items()
}

all_bias_rows = []
for name in final_rankers:
    items, pops = set(), []
    for uid in all_users:
        for m in final_top10[name][uid]:
            items.add(m)
            pops.append(popularity.get(m, 0))
    all_bias_rows.append({
        "Recommender": name,
        "Catalog coverage": len(items) / len(popularity),
        "Coverage of reachable pool": len(items) / final_reachable_pool[name],
        "Mean popularity of recs": float(np.mean(pops)) if pops else np.nan,
    })
all_bias_df = pd.DataFrame(all_bias_rows).set_index("Recommender")

all_cold_hits = {
    name: sum(len(set(final_top10[name][uid]) & cold_items) for uid in all_users)
    for name in final_rankers
}

print("=== Coverage and popularity bias - all six models ===")
print(all_bias_df.to_string(formatters={
    "Catalog coverage": "{:.1%}".format,
    "Coverage of reachable pool": "{:.1%}".format,
    "Mean popularity of recs": "{:.1f}".format,
}))
print(f"\\n=== Cold-start items in top-10 (out of {len(all_users) * 10} slots) ===")
for name, hits in all_cold_hits.items():
    print(f"{name:35s}: {hits:5d}  ({100 * hits / (len(all_users) * 10):.2f}% of slots)")

fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))

axes[0].bar(all_bias_df.index, all_bias_df["Catalog coverage"] * 100, color="#4C72B0")
axes[0].set_ylabel("Catalogue coverage (%)")
axes[0].set_title("Step 10a - catalogue coverage")
axes[0].tick_params(axis="x", rotation=30)

palette = {"SVD": "#4C72B0", "Content": "#55A868",
           f"Hybrid (weighted, a={best_alpha})": "#C44E52",
           "Hybrid (switching)": "#8172B2", "Most-Popular": "#8C8C8C",
           "AutoRec": "#CCB974"}
for name in final_ndcg.columns:
    axes[1].plot(final_ndcg.index, final_ndcg[name], marker="o",
                 linestyle="--" if name == "Most-Popular" else "-",
                 color=palette.get(name), label=name)
axes[1].set_xlabel("K"); axes[1].set_ylabel("NDCG@K (test)")
axes[1].set_xticks(Ks)
axes[1].set_title("Step 10b - ranking quality, all six models")
axes[1].legend(fontsize=7)

axes[2].bar(list(all_cold_hits.keys()), list(all_cold_hits.values()),
            color=[palette.get(n, "#4C72B0") for n in all_cold_hits])
axes[2].set_ylabel("Cold-start items in top-10")
axes[2].set_title("Step 10c - cold-start reach")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()'''

nb["cells"].append({"cell_type": "markdown", "metadata": {},
                    "source": graph_md.splitlines(keepends=True)})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                    "outputs": [], "source": graph_code.splitlines(keepends=True)})
print("notebook: Step 10's Graph added (6-model coverage + NDCG + cold-start)")

nbp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"\nnotebook now has {len(nb['cells'])} cells. Restart kernel and run all.")
