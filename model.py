# model.py
# model.py
import math
import random
import pandas as pd
import numpy as np
from agent import WaspAgent

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

class LarvalCell:
    def __init__(self, cid, x, y, stage=None):
        self.cell_id = str(cid)
        self.pos = (float(x), float(y))
        self.stage = stage
        self.feed_count = 0
        self.last_fed = -1
        self.feed_history = []

    def feed(self, t, wasp):
        self.feed_count += 1
        self.last_fed = t
        self.feed_history.append((t, wasp))


class WaspModel:
    def __init__(self, layout_csv,
                 n_wasps=25, n_foragers=3, n_unloaders=6,
                 perception_radius=1.2, feed_radius=0.4,
                 step_size=0.5, forward_bias=0.7,
                 forager_mean_return=135.0,
                 nest_id=None):

        self.layout_csv = layout_csv
        self.time = 0

        self.n_wasps = n_wasps
        self.n_foragers = n_foragers
        self.n_unloaders = n_unloaders

        self.perception_radius = perception_radius
        self.feed_radius = feed_radius
        self.step_size = step_size
        self.forward_bias = forward_bias
        self.forager_mean_return = forager_mean_return
        self.forager_transfer_prob = 0.7

        df = pd.read_csv(layout_csv)
        if nest_id and 'nest' in df.columns:
            df = df[df['nest'] == nest_id]

        xcol = [c for c in df.columns if c.lower().startswith("x")][0]
        ycol = [c for c in df.columns if c.lower().startswith("y")][0]

        larvae = []
        for _, r in df.iterrows():
            content_str = str(r).lower()
            if 'l' in content_str or 'larva' in content_str:
                larvae.append(r)

        if not larvae:
            larvae = df.iloc[:30].to_dict('records')

        xs = [float(r[xcol]) for r in larvae]
        ys = [float(r[ycol]) for r in larvae]

        self.xmin, self.xmax = min(xs) - 1, max(xs) + 1
        self.ymin, self.ymax = min(ys) - 1, max(ys) + 1

        self.larvae = [
            LarvalCell(
                r.get("cell.no.", i),
                r[xcol], r[ycol],
                r.get("stages", None)
            )
            for i, r in enumerate(larvae)
        ]

        self.entrance_pos = ((self.xmin + self.xmax) / 2,
                             (self.ymin + self.ymax) / 2)

        roles = (
            ['forager'] * n_foragers +
            ['unloader'] * n_unloaders +
            ['idle'] * (n_wasps - (n_foragers + n_unloaders))
        )
        random.shuffle(roles)

        self.agents = []
        for uid, r in enumerate(roles):
            a = WaspAgent(uid, self, role=r)
            a.pos = (
                self.entrance_pos[0] + random.uniform(-0.6, 0.6),
                self.entrance_pos[1] + random.uniform(-0.6, 0.6)
            )
            self.agents.append(a)

        self.global_feed_events = []
        self.global_transfers = []
        self.food_arrivals = []

    # ---------------- Agent helpers ---------------- #
    def agents_in_radius(self, pos, rad, exclude=None):
        return [
            a for a in self.agents
            if a is not exclude and distance(a.pos, pos) <= rad
        ]

    def distance(self, a, b):
        return distance(a, b)

    # ---------------- Larva selection ---------------- #
    def select_larva_for(self, wasp):
        t = self.time
        scored = []

        for c in self.larvae:
            time_since = t - c.last_fed if c.last_fed >= 0 else t + 999
            dist = distance(wasp.pos, c.pos)
            score = time_since - 0.6 * dist
            scored.append((score, c))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1] if scored else None

    # ---------------- Simulation loop ---------------- #
    def step(self):
        for a in self.agents:
            a.step()
        self.time += 1

    def run(self, steps=600):
        for _ in range(steps):
            self.step()

