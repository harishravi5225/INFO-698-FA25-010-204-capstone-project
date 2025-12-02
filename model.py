# model.py
"""
WaspModel (continuous) — final, commented.
Key features:
 - reads layout CSV (x,y) and chooses contiguous cluster of 'nest_size' cells
 - identifies centroid and periphery; entrance_pos biased to periphery
 - creates agents (forager/primary/unloader/feeder/idle)
 - occupancy bookkeeping (rounded grid cells) to limit agents per cell to 2
 - agents_in_radius, distance helper used by agent.py
 - register_arrival / log_transfer / log_feed to build bout logs
 - larvae with hunger timers; workers have eating intervals
 - compute_gini and route_efficiency helper
"""

import math, random
import numpy as np
import pandas as pd
from collections import defaultdict
from agent import make_agent

def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

class LarvalCell:
    def __init__(self, cid, x, y, hunger_min=80, hunger_max=180):
        self.cell_id = str(cid)
        self.pos = (float(x), float(y))
        self.feed_count = 0
        self.last_fed = -1
        self.feed_history = []
        self.hunger_timer = random.randint(hunger_min, hunger_max)
        self.hunger_min = hunger_min
        self.hunger_max = hunger_max
        self.checked_by_wasps = set()  # track which wasps checked but didn't feed
        self.distance_from_center = None

    def feed(self, t, wasp_id=None, amount=1, bout_id=None):
        self.feed_count += amount
        self.last_fed = t
        self.feed_history.append((t, wasp_id, amount, bout_id))
        self.hunger_timer = random.randint(self.hunger_min, self.hunger_max)
        # clear checked set when successfully fed
        self.checked_by_wasps.discard(wasp_id) if wasp_id else None

class WaspModel:
    def __init__(self,
                 layout_csv="ED_FL_3nests1noC2.csv",
                 alt_csv="ALL_FL_minmaj_final3noC2.csv",
                 nest_size=30,
                 wasp_to_cell_ratio=0.9,
                 forager_frac=0.06,
                 receiver_capacity=5,
                 perception_radius=1.2,
                 feed_radius=0.4,
                 step_size=0.5,
                 forward_bias=0.7,
                 forager_mean_return=40,
                 forager_transfer_prob=0.7,
                 bout_extend_prob=0.05,
                 model_seed=None):
        if model_seed is not None:
            random.seed(model_seed)
            np.random.seed(model_seed)

        # params
        self.layout_csv = layout_csv
        self.alt_csv = alt_csv
        self.nest_size = int(nest_size)
        self.wasp_to_cell_ratio = wasp_to_cell_ratio
        self.forager_frac = forager_frac
        self.receiver_capacity = receiver_capacity
        self.perception_radius = perception_radius
        self.feed_radius = feed_radius
        self.step_size = step_size
        self.forward_bias = forward_bias
        self.forager_mean_return = forager_mean_return
        self.forager_transfer_prob = forager_transfer_prob
        self.bout_extend_prob = bout_extend_prob

        # runtime
        self.model_time = 0
        self.global_feed_events = []
        self.global_transfers = []
        self.food_arrivals = []
        self.bouts_log = []
        self.active_bouts = {}
        self.bout_counter = 0
        self.hungry_ignored_events = []  # (bout_id, larva_id, wasp_id, time)
        self.redundant_feed_events = []  # (bout_id, larva_id, wasp1_id, wasp2_id, time)

        # read CSV and pick contiguous cluster of cells
        df = pd.read_csv(self.layout_csv)
        xcol = [c for c in df.columns if c.lower().startswith("x")][0]
        ycol = [c for c in df.columns if c.lower().startswith("y")][0]

        if 'type' in df.columns:
            larrows = df[df['type'].str.contains('larval', case=False, na=False)]
            if len(larrows) == 0:
                larrows = df.copy()
        else:
            larrows = df.copy()

        if 'ID' in larrows.columns:
            ids = larrows['ID'].astype(str).tolist()
        else:
            ids = [str(i) for i in larrows.index.tolist()]

        xs = larrows[xcol].tolist(); ys = larrows[ycol].tolist()
        pts = list(zip(ids, xs, ys))

        # supplement with alt CSV if needed
        if len(pts) < self.nest_size and self.alt_csv:
            alt = pd.read_csv(self.alt_csv)
            ax = [c for c in alt.columns if c.lower().startswith("x")][0]
            ay = [c for c in alt.columns if c.lower().startswith("y")][0]
            if 'ID' in alt.columns:
                aid = alt['ID'].astype(str).tolist()
            else:
                aid = [str(i) for i in alt.index.tolist()]
            extra = list(zip(aid, alt[ax].tolist(), alt[ay].tolist()))
            pts += extra

        if len(pts) < self.nest_size:
            raise ValueError("Not enough coordinates to build requested nest_size")

        # choose contiguous cluster by distance to seed
        pts_sorted = sorted(pts, key=lambda r: euclid((r[1], r[2]), (pts[0][1], pts[0][2])))
        chosen = pts_sorted[:self.nest_size]
        self.larvae = [LarvalCell(cid, x, y) for cid, x, y in chosen]

        # bounding box & centroid
        xs = [c.pos[0] for c in self.larvae]; ys = [c.pos[1] for c in self.larvae]
        self.xmin, self.xmax = min(xs)-1, max(xs)+1
        self.ymin, self.ymax = min(ys)-1, max(ys)+1
        self.centroid = (np.mean(xs), np.mean(ys))

        # identify periphery (~outer 20% by distance)
        d = {c.cell_id: euclid(self.centroid, c.pos) for c in self.larvae}
        sd = sorted(d.items(), key=lambda x: x[1])
        cut = int(0.8*len(sd))
        self.periphery_ids = set([cid for cid,_ in sd[cut:]])
        per_cells = [c for c in self.larvae if c.cell_id in self.periphery_ids]
        self.entrance_pos = per_cells[0].pos if per_cells else self.centroid

        # occupancy bookkeeping (rounded grid cell -> list of agent ids)
        self.cell_occupancy = defaultdict(list)
        self.max_per_cell = 2  # PROF requirement: up to 2 wasps per cell

        # compute distance from center for all larvae
        for larva in self.larvae:
            larva.distance_from_center = euclid(self.centroid, larva.pos)

        # create agents: approximate 90% of larvae
        total_wasps = int(max(6, round(self.wasp_to_cell_ratio * len(self.larvae))))
        n_foragers = max(1, int(round(total_wasps * self.forager_frac)))
        n_primary = max(1, int(round(0.18 * (total_wasps - n_foragers))))
        n_secondary = max(1, int(round(0.35 * (total_wasps - n_foragers))))
        n_feeders = max(0, total_wasps - n_foragers - n_primary - n_secondary)
        roles = (['forager']*n_foragers + ['primary']*n_primary + ['secondary']*n_secondary + ['feeder']*n_feeders)
        while len(roles) < total_wasps:
            roles.append('idle')
        random.shuffle(roles)

        self.agents = []
        for uid, r in enumerate(roles):
            ag = make_agent(uid, self, r)
            if r == 'forager':
                # place foragers toward entrance (so when they "arrive" they are already near periphery)
                ag.pos = (self.entrance_pos[0] + random.uniform(-0.2,0.2), self.entrance_pos[1] + random.uniform(-0.2,0.2))
            elif r == 'primary':
                ag.pos = (self.entrance_pos[0] + random.uniform(-0.4,0.4), self.entrance_pos[1] + random.uniform(-0.4,0.4))
            else:
                ag.pos = (self.centroid[0] + random.uniform(-0.6,0.6), self.centroid[1] + random.uniform(-0.6,0.6))
            key = (round(ag.pos[0],2), round(ag.pos[1],2))
            self.cell_occupancy[key].append(ag.unique_id)
            self.agents.append(ag)

    # ---------------- helpers used by agent.py ----------------
    def agents_in_radius(self, pos, rad, exclude=None):
        out = []
        for a in self.agents:
            if exclude is not None and hasattr(exclude,'unique_id') and a.unique_id == exclude.unique_id:
                continue
            try:
                d = euclid(a.pos, pos)
            except Exception:
                continue
            if d <= rad:
                out.append(a)
        return out

    def distance(self, a, b):
        # accept tuples or objects with .pos
        if isinstance(a, tuple) and isinstance(b, tuple):
            return euclid(a,b)
        if hasattr(a,'pos') and hasattr(b,'pos'):
            return euclid(a.pos, b.pos)
        # fallback for ids
        apos = None; bpos = None
        try:
            apos = next(c.pos for c in self.larvae if c.cell_id == str(a))
            bpos = next(c.pos for c in self.larvae if c.cell_id == str(b))
        except StopIteration:
            pass
        if apos and bpos:
            return euclid(apos, bpos)
        raise ValueError("Unsupported types for distance()")

    # occupancy helpers
    def cell_has_space(self, pos):
        key = (round(pos[0],2), round(pos[1],2))
        return len(self.cell_occupancy.get(key, [])) < self.max_per_cell

    def register_occupant(self, pos, uid):
        key = (round(pos[0],2), round(pos[1],2))
        self.cell_occupancy[key].append(uid)

    def unregister_occupant(self, pos, uid):
        key = (round(pos[0],2), round(pos[1],2))
        if uid in self.cell_occupancy.get(key, []):
            self.cell_occupancy[key].remove(uid)
            if len(self.cell_occupancy[key])==0:
                del self.cell_occupancy[key]

    # prey drawing & peripheral position helper
    def draw_prey_amount(self):
        base = random.choice([3,5,8,10,12])
        if random.random() < 0.03:
            base *= random.randint(2,3)
        return base

    # bouts / logging
    def register_arrival(self, forager_agent, qty):
        # join existing bout with small prob (bout_extend_prob) else new bout
        join = None
        for bid, info in self.active_bouts.items():
            if not info.get('ended', False) and random.random() < self.bout_extend_prob:
                join = bid
                break
        if join is None:
            self.bout_counter += 1
            bid = self.bout_counter
            self.active_bouts[bid] = {'start': self.model_time, 'arrivals': [], 'transfers': [], 'feeds': [], 'ended': False}
            join = bid
        self.active_bouts[join]['arrivals'].append((forager_agent.unique_id, self.model_time, qty))
        self.food_arrivals.append((forager_agent.unique_id, self.model_time, qty))
        return join

    def log_transfer(self, giver, receiver, amt=None):
        if not self.active_bouts:
            return
        bid = max(self.active_bouts.keys())
        self.active_bouts[bid]['transfers'].append((giver, receiver, amt, self.model_time))
        self.global_transfers.append((giver, receiver, amt, self.model_time, bid))

    def log_feed(self, wasp_id, cell_id, amt=1):
        if not self.active_bouts:
            return
        bid = max(self.active_bouts.keys())
        self.active_bouts[bid]['feeds'].append((wasp_id, cell_id, amt, self.model_time))
        self.global_feed_events.append((wasp_id, cell_id, amt, self.model_time, bid))

    def record_larva_checked_but_not_fed(self, bout_id, larva_cell_id, wasp_id, time):
        """Track when a wasp checks a larva but doesn't feed it"""
        if bout_id:
            larva_obj = next((c for c in self.larvae if c.cell_id == str(larva_cell_id)), None)
            if larva_obj:
                larva_obj.checked_by_wasps.add(wasp_id)
                self.hungry_ignored_events.append((bout_id, larva_cell_id, wasp_id, time))

    def record_redundant_feed(self, bout_id, larva_cell_id, wasp1_id, wasp2_id, time):
        """Track when two wasps unnecessarily feed the same larva"""
        self.redundant_feed_events.append((bout_id, larva_cell_id, wasp1_id, wasp2_id, time))

    # hunger tick for larvae
    def _hunger_tick(self):
        for lar in self.larvae:
            if lar.hunger_timer > 0:
                lar.hunger_timer -= 1

    def should_promote_idle_to_receiver(self):
        window = 80
        recent = [a for a in self.food_arrivals if a[1] >= (self.model_time - window)]
        threshold = max(1, int(0.02 * len(self.agents)))
        if len(recent) > threshold:
            return random.random() < 0.06
        return False

    def select_larva_for(self, wasp_agent):
        """Select next larva for a feeder to visit (nearest hungry larva)"""
        # Find nearby hungry larvae
        nearby = self.agents_in_radius(wasp_agent.pos, self.perception_radius * 3, exclude=wasp_agent)
        larvae_near = [a for a in nearby if hasattr(a, 'cell_id')]  # filter for larva cells
        
        # If no larvae nearby, find nearest hungry larva globally
        hungry_larvae = [c for c in self.larvae if c.hunger_timer <= 0]
        if not hungry_larvae:
            hungry_larvae = self.larvae  # if none hungry, pick any
        
        if not hungry_larvae:
            return None
        
        # Pick nearest hungry larva
        target = min(hungry_larvae, key=lambda c: euclid(wasp_agent.pos, c.pos))
        return target

    # core loop
    def step(self):
        random.shuffle(self.agents)
        for a in self.agents:
            a.step()
        self._hunger_tick()
        self.model_time += 1
        # close old bouts (no arrivals for a while)
        for bid, info in list(self.active_bouts.items()):
            if info.get('ended', False):
                continue
            last_act = info['arrivals'][-1][1] if info['arrivals'] else info['start']
            if self.model_time - last_act > 80:  # shorter window to close bouts
                info['ended'] = True
                info['end'] = self.model_time
                self.bouts_log.append((bid, info))

    def run(self, steps=600):
        for _ in range(steps):
            self.step()

    # metrics
    def compute_gini(self):
        arr = np.array([c.feed_count for c in self.larvae], dtype=float)
        if arr.sum()==0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        idx = np.arange(1, n+1)
        return float((2.0*np.sum(idx*arr)/(n*arr.sum())) - (n+1)/n)

    def compute_feeding_by_distance(self):
        """Return feeding rates stratified by distance from center"""
        if not self.larvae:
            return {}
        distances = [c.distance_from_center for c in self.larvae if c.distance_from_center is not None]
        if not distances:
            return {}
        d_min, d_max = min(distances), max(distances)
        bins = np.linspace(d_min, d_max, 6)  # 5 distance bins
        result = {}
        for i in range(len(bins)-1):
            larvae_in_bin = [c for c in self.larvae if bins[i] <= c.distance_from_center < bins[i+1]]
            if larvae_in_bin:
                avg_feeds = np.mean([c.feed_count for c in larvae_in_bin])
                result[f"dist_{i}_{bins[i]:.2f}_{bins[i+1]:.2f}"] = avg_feeds
        return result

    def compute_center_vs_periphery_feeding(self):
        """Compare feeding rates: center vs periphery"""
        center_larvae = [c for c in self.larvae if c.cell_id not in self.periphery_ids]
        periphery_larvae = [c for c in self.larvae if c.cell_id in self.periphery_ids]
        center_avg = np.mean([c.feed_count for c in center_larvae]) if center_larvae else 0.0
        periph_avg = np.mean([c.feed_count for c in periphery_larvae]) if periphery_larvae else 0.0
        return {"center_avg": float(center_avg), "periphery_avg": float(periph_avg), 
                "center_count": len(center_larvae), "periph_count": len(periphery_larvae)}

    def compute_bout_metrics(self, bout_info):
        """Compute comprehensive metrics for a single bout"""
        arrivals = bout_info.get('arrivals', [])
        transfers = bout_info.get('transfers', [])
        feeds = bout_info.get('feeds', [])
        
        total_food_arrived = sum(a[2] for a in arrivals)
        unique_larvae_fed = set(f[1] for f in feeds)
        total_food_delivered = sum(f[2] for f in feeds)
        larvae_fed_multiple = defaultdict(int)
        for f in feeds:
            larvae_fed_multiple[f[1]] += 1
        
        unfed_larvae = set(c.cell_id for c in self.larvae) - unique_larvae_fed
        
        # track which feeders fed which larvae
        feeder_larva_map = defaultdict(list)
        for wasp_id, cell_id, amt, t in feeds:
            feeder_larva_map[wasp_id].append((cell_id, amt, t))
        
        # build route from feed sequence
        route = [f[1] for f in feeds if f[2] > 0]  # cells visited during feeding
        
        return {
            "bout_start": bout_info.get('start'),
            "bout_end": bout_info.get('end'),
            "duration": bout_info.get('end', self.model_time) - bout_info.get('start', 0),
            "arrivals": len(arrivals),
            "total_food_arrived": total_food_arrived,
            "total_food_delivered": total_food_delivered,
            "unique_larvae_fed": len(unique_larvae_fed),
            "larvae_fed_multiple": dict(larvae_fed_multiple),
            "unfed_larvae_count": len(unfed_larvae),
            "unfed_larvae_ids": list(unfed_larvae),
            "route": route,
            "route_length": len(route),
            "transfers": len(transfers),
            "feeder_coverage": dict(feeder_larva_map),
            "feeds": feeds
        }

    def get_all_bouts_metrics(self):
        """Get metrics for all completed bouts"""
        metrics = []
        for bid, bout in self.bouts_log:
            metrics.append(self.compute_bout_metrics(bout))
        return metrics

    def compute_efficiency_metrics(self):
        """Compute overall efficiency and inequality metrics"""
        return {
            "gini": self.compute_gini(),
            "center_vs_periphery": self.compute_center_vs_periphery_feeding(),
            "feeding_by_distance": self.compute_feeding_by_distance(),
            "total_feeds": sum(c.feed_count for c in self.larvae),
            "avg_feeds_per_larva": np.mean([c.feed_count for c in self.larvae]),
            "hungry_ignored_count": len(self.hungry_ignored_events),
            "redundant_feed_count": len(self.redundant_feed_events),
            "total_bouts_completed": len(self.bouts_log)
        }

    def route_efficiency(self, route_seq):
        """Compute efficiency metrics: path length and directional bias score"""
        if len(route_seq) < 2:
            return {"path_length": 0.0, "direction_score": 0.0, "turn_analysis": {}}
        
        coords = {c.cell_id: c.pos for c in self.larvae}
        L = 0.0
        forward_moves = 0
        side_moves = 0
        back_moves = 0
        
        for i in range(len(route_seq)-1):
            try:
                L += euclid(coords[route_seq[i]], coords[route_seq[i+1]])
            except KeyError:
                continue
        
        # direction score uses forward=+1, side=+0.5, back=-1 penalties
        score = 0.0
        for i in range(1, len(route_seq)-1):
            try:
                p0 = coords[route_seq[i-1]]
                p1 = coords[route_seq[i]]
                p2 = coords[route_seq[i+1]]
                v1 = (p1[0]-p0[0], p1[1]-p0[1])
                v2 = (p2[0]-p1[0], p2[1]-p1[1])
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                mag = (math.hypot(*v1)*math.hypot(*v2)) + 1e-9
                cosang = dot/mag
                
                if cosang > 0.8:
                    score += 1.0
                    forward_moves += 1
                elif cosang > 0.0:
                    score += 0.5
                    side_moves += 1
                elif cosang > -0.3:
                    score += -0.5
                    side_moves += 1
                else:
                    score += -1.0
                    back_moves += 1
            except (KeyError, ValueError):
                continue
        
        norm = score / (len(route_seq)-2) if len(route_seq)>2 else score
        return {
            "path_length": L,
            "direction_score": norm,
            "turn_analysis": {
                "forward": forward_moves,
                "side": side_moves,
                "back": back_moves
            }
        }

    def greedy_route(self, start_cell_id, cell_list):
        """Compute greedy nearest-neighbor route from starting cell"""
        coords = {c.cell_id: c.pos for c in self.larvae}
        route = [start_cell_id]
        remaining = set(cell_list) - {start_cell_id}
        current = start_cell_id
        
        while remaining:
            next_cell = min(remaining, key=lambda c: euclid(coords.get(current, (0,0)), coords.get(c, (0,0))))
            route.append(next_cell)
            remaining.remove(next_cell)
            current = next_cell
        
        return route

    def random_walk_route(self, start_cell_id, cell_list, steps=None):
        """Simulate random walk through cells"""
        if steps is None:
            steps = len(cell_list)
        route = [start_cell_id]
        remaining = set(cell_list) - {start_cell_id}
        
        for _ in range(steps):
            if not remaining:
                break
            next_cell = random.choice(list(remaining))
            route.append(next_cell)
            remaining.discard(next_cell)
        
        return route

    def biased_random_walk_route(self, start_cell_id, cell_list, bias_strength=0.6):
        """Simulate forward-biased random walk (prefer nearest)"""
        coords = {c.cell_id: c.pos for c in self.larvae}
        route = [start_cell_id]
        remaining = set(cell_list) - {start_cell_id}
        current = start_cell_id
        
        while remaining:
            if random.random() < bias_strength:
                # biased toward nearest
                next_cell = min(remaining, key=lambda c: euclid(coords.get(current, (0,0)), coords.get(c, (0,0))))
            else:
                # random
                next_cell = random.choice(list(remaining))
            route.append(next_cell)
            remaining.remove(next_cell)
            current = next_cell
        
        return route

    def compare_to_baselines(self, route_seq, nperm=200):
        coords = {c.cell_id: c.pos for c in self.larvae}
        def path_len(seq):
            L = 0.0
            for i in range(len(seq)-1):
                L += euclid(coords[seq[i]], coords[seq[i+1]])
            return L
        obs = path_len(route_seq)
        start = route_seq[0]
        remaining = set(route_seq[1:])
        seq = [start]; cur = start
        while remaining:
            nxt = min(remaining, key=lambda c: euclid(coords[cur], coords[c]))
            seq.append(nxt); remaining.remove(nxt); cur = nxt
        nn_len = path_len(seq)
        import random as _rnd
        perms = []
        for _ in range(nperm):
            tmp = route_seq[:]; _rnd.shuffle(tmp); perms.append(path_len(tmp))
        rand_mean = float(np.mean(perms))
        # TSP optional if python-tsp installed (heuristic)
        tsp_len = None
        try:
            from python_tsp.heuristics import solve_tsp_local_search
            uniq = list(dict.fromkeys(route_seq))
            coords_list = [coords[c] for c in uniq]
            n = len(coords_list)
            import numpy as _np
            dist_mat = _np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    dist_mat[i,j] = euclid(coords_list[i], coords_list[j])
            perm, d = solve_tsp_local_search(dist_mat)
            tsp_len = float(d)
        except Exception:
            pass
        return {"obs": obs, "nn": nn_len, "random_mean": rand_mean, "tsp": tsp_len}
