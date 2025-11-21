# run.py
# run.py
"""
Run the full agent-based model:
- Baseline simulation
- Parameter sweep
- Route analysis
Outputs saved into ./output_abm/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import WaspModel, distance

OUTDIR = "output_abm"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- Baseline run ---------------- #
print("Running baseline...")

model = WaspModel(
    layout_csv="ED_FL_3nests1noC2.csv",
    n_wasps=25,
    n_foragers=3,
    n_unloaders=6,
    perception_radius=1.2,
    forward_bias=0.7
)

model.run(steps=800)

# Feed events
feed_rows = []
for c in model.larvae:
    for (t, w) in c.feed_history:
        feed_rows.append({"time": t, "wasp_id": w, "cell_id": c.cell_id})
feed_df = pd.DataFrame(feed_rows)
feed_df.to_csv(os.path.join(OUTDIR, "baseline_feed_events.csv"), index=False)

# Larval counts
larval_df = pd.DataFrame([
    {"cell_id": c.cell_id, "x": c.pos[0], "y": c.pos[1], "feeds": c.feed_count}
    for c in model.larvae
])
larval_df.to_csv(os.path.join(OUTDIR, "baseline_larval_counts.csv"), index=False)

# Heatmap
plt.scatter(larval_df["x"], larval_df["y"], c=larval_df["feeds"], cmap="magma", s=60)
plt.colorbar(label="Feed count")
plt.title("Baseline Feeding Heatmap")
plt.savefig(os.path.join(OUTDIR, "baseline_heatmap.png"))
plt.close()

print("Baseline complete. Files saved.")

# ---------------- Parameter Sweep ---------------- #
print("Running parameter sweep...")

results = []

for foragers in [1,2,3,4]:
    for pr in [0.8, 1.2, 2.0]:
        for fb in [0.3, 0.7, 0.95]:
            for r in range(4):
                m = WaspModel(
                    layout_csv="ED_FL_3nests1noC2.csv",
                    n_wasps=25,
                    n_foragers=foragers,
                    n_unloaders=6,
                    perception_radius=pr,
                    forward_bias=fb
                )
                m.run(steps=500)

                counts = np.array([c.feed_count for c in m.larvae])
                if counts.sum() == 0:
                    gini = 0
                else:
                    s = np.sort(counts)
                    n = len(s)
                    idx = np.arange(1, n+1)
                    gini = (2*np.sum(idx*s)/(n*s.sum())) - (n+1)/n

                results.append({
                    "foragers": foragers,
                    "perception": pr,
                    "forward_bias": fb,
                    "gini": gini
                })

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTDIR, "param_sweep_results.csv"), index=False)

print("Parameter sweep done.")

# ---------------- Route analysis ---------------- #
print("Running route analysis...")

feed_df = feed_df.sort_values("time")
seq = feed_df["cell_id"].astype(str).unique().tolist()

id2pos = {c.cell_id: c.pos for c in model.larvae}

def path_length(order):
    if len(order) < 2:
        return 0
    L = 0
    for i in range(len(order)-1):
        L += distance(id2pos[order[i]], id2pos[order[i+1]])
    return L

obs = path_length(seq)

# NN heuristic
nn_seq = [seq[0]]
remaining = set(seq[1:])
cur = nn_seq[0]

while remaining:
    nxt = min(remaining, key=lambda c: distance(id2pos[cur], id2pos[c]))
    nn_seq.append(nxt)
    remaining.remove(nxt)
    cur = nxt

nn_len = path_length(nn_seq)

# Random permutations
perms = []
import random
for i in range(200):
    tmp = seq[:]
    random.shuffle(tmp)
    perms.append(path_length(tmp))

summary = pd.DataFrame([{
    "observed": obs,
    "nn": nn_len,
    "random_mean": np.mean(perms)
}])

summary.to_csv(os.path.join(OUTDIR, "route_comparison.csv"), index=False)

print("Route analysis complete.")
print("All outputs saved in:", OUTDIR)

