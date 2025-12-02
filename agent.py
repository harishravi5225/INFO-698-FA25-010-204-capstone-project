# agent.py
"""
WaspAgent (continuous-space). Compatible with model.py.
Exports:
 - make_agent(uid, model, role) -> WaspAgent

Key points:
 - roles: 'forager','primary','secondary','feeder','idle'
 - .carry : int food units held
 - uses model.agents_in_radius, model.distance, model.register_arrival, model.log_feed, model.log_transfer
 - enforces light occupancy courtesy checks via model.cell_has_space (model tracks rounded cells)
"""

import math, random

class AgentBase:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model

class WaspAgent(AgentBase):
    def __init__(self, unique_id, model, role='idle'):
        super().__init__(unique_id, model)
        self.role = role
        self.carry = 0
        self.orientation = None
        self.path = []
        self.feed_events = []
        self.transfer_events = []
        self.last_role = role
        self.role_switch_count = 0
        # Foragers go out and return after an interval (exponential)
        if role == 'forager':
            mean = getattr(model, 'forager_mean_return', 40)
            self.time_out = max(1, int(random.expovariate(1.0 / max(1.0, mean))))
        else:
            self.time_out = 0
        # worker hunger/digestion state (adults)
        self.hunger_timer = random.randint(400, 2000)  # long interval
        self.eating = False
        # continuous-space position
        self._pos = (0.0, 0.0)
        self.current_bout_id = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, v):
        self._pos = (float(v[0]), float(v[1]))

    # small helper to attempt to move while respecting occupancy limit
    def _attempt_move(self, new_pos):
        # check occupancy in model; if too crowded, don't move
        if hasattr(self.model, 'cell_has_space'):
            if not self.model.cell_has_space(new_pos):
                # try small jitter instead
                jitter = 0.05
                alt = (self.pos[0] + random.uniform(-jitter, jitter), self.pos[1] + random.uniform(-jitter, jitter))
                if self.model.cell_has_space(alt):
                    self.model.unregister_occupant(self.pos, self.unique_id)
                    self.pos = alt
                    self.model.register_occupant(self.pos, self.unique_id)
                return
        # perform move
        self.model.unregister_occupant(self.pos, self.unique_id)
        self.pos = new_pos
        self.model.register_occupant(self.pos, self.unique_id)

    def step(self):
        # record path
        try:
            self.path.append(self.pos)
        except Exception:
            pass

        # adult hunger handling: occasionally eat (interrupt task)
        if self.hunger_timer <= 0 and not self.eating:
            # eat for a while
            self.eating = True
            self.hunger_timer = random.randint(80, 220)
        if self.eating:
            # simple eating cycle: skip heavy actions but decrement eating
            self.hunger_timer -= 1
            if self.hunger_timer <= 0:
                self.eating = False
                self.hunger_timer = random.randint(400, 2000)
            return
        else:
            # decrement passive hunger
            self.hunger_timer -= 1

        if self.role == 'forager':
            self._forager_step()
        elif self.role in ('primary','secondary','feeder','unloader'):
            self._worker_step()
        else:
            self._idle_step()

    # movement: forward-biased continuous random walk
    def _move_forward_biased(self, step_size=None):
        if step_size is None:
            step_size = getattr(self.model, 'step_size', 0.5)
        forward_bias = getattr(self.model, 'forward_bias', 0.7)
        if self.orientation is None or random.random() > forward_bias:
            ang = random.random() * 2 * math.pi
        else:
            ox, oy = self.orientation
            jitter = random.gauss(0, 0.6)
            ang = math.atan2(oy, ox) + jitter
        dx, dy = math.cos(ang)*step_size, math.sin(ang)*step_size
        nx = min(max(self.pos[0] + dx, self.model.xmin), self.model.xmax)
        ny = min(max(self.pos[1] + dy, self.model.ymin), self.model.ymax)
        new = (nx, ny)
        self.orientation = (dx, dy)
        self._attempt_move(new)

    def _move_towards(self, target_pos, step_size=None):
        if step_size is None:
            step_size = getattr(self.model, 'step_size', 0.5)
        tx, ty = target_pos
        x, y = self.pos
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        ux, uy = dx/dist, dy/dist
        nx = min(max(x + ux*step_size, self.model.xmin), self.model.xmax)
        ny = min(max(y + uy*step_size, self.model.ymin), self.model.ymax)
        new = (nx, ny)
        self.orientation = (ux, uy)
        self._attempt_move(new)

    # FORAGER behavior: leave/return, carry variable qty, transfer to receivers
    def _forager_step(self):
        # Check if should go out foraging (not carrying and ready)
        if self.carry == 0:
            if self.time_out > 0:
                self.time_out -= 1
                # reduce timeout faster to make foragers more active
                if random.random() < 0.15:  # 15% chance to return early with food
                    self.time_out = 0
                return
            else:
                # Arrived at entrance - return with food
                ent = getattr(self.model, 'entrance_pos', self.model.centroid)
                jitter = 0.25
                newpos = (ent[0] + random.uniform(-jitter, jitter), ent[1] + random.uniform(-jitter, jitter))
                self._attempt_move(newpos)
                # pick prey amount and register bout
                qty = self.model.draw_prey_amount()
                self.carry = int(qty)
                if hasattr(self.model, 'register_arrival'):
                    bout_id = self.model.register_arrival(self, qty)
                    self.current_bout_id = bout_id
                if hasattr(self.model, 'food_arrivals'):
                    self.model.food_arrivals.append((self.unique_id, self.model.model_time, qty))
                return

        # carrying > 0: try to hand off to nearby receivers
        nearby = self.model.agents_in_radius(self.pos, getattr(self.model, 'perception_radius', 1.2), exclude=self)
        # receiver priority order: primary > unloader > secondary > feeder
        candidates = [a for a in nearby if a.role in ('primary','unloader','secondary','feeder') and getattr(a, 'carry', 0) < getattr(self.model, 'receiver_capacity', 5)]
        
        if candidates and random.random() < getattr(self.model, 'forager_transfer_prob', 0.7):
            rec = random.choice(candidates)
            cap = getattr(self.model, 'receiver_capacity', 5) - getattr(rec, 'carry', 0)
            give = min(self.carry, max(1, min(2, cap)))
            if give > 0:
                rec.carry += give
                self.carry -= give
                self.transfer_events.append((self.model.model_time, rec.unique_id, give))
                if hasattr(self.model, 'global_transfers'):
                    self.model.global_transfers.append((self.unique_id, rec.unique_id, give, self.model.model_time))
                if hasattr(self.model, 'log_transfer'):
                    self.model.log_transfer(self.unique_id, rec.unique_id, give)
                # if emptied, go back out
                if self.carry <= 0:
                    mean = getattr(self.model, 'forager_mean_return', 40)  # shorter return time
                    self.time_out = max(1, int(random.expovariate(1.0 / max(1.0, mean))))
            return
        else:
            # No available receivers - move back towards entrance to look for them
            ent = getattr(self.model, 'entrance_pos', self.model.centroid)
            if random.random() < 0.6:
                self._move_towards(ent, step_size=getattr(self.model, 'step_size', 0.5))
            else:
                self._move_forward_biased(step_size=getattr(self.model, 'step_size', 0.5))

    # WORKER behavior (primary/secondary/feeder)
    def _worker_step(self):
        # dynamic role switching based on congestion/food arrival
        if self._should_switch_role():
            self._switch_role()
        
        # if carrying food -> select larva & feed
        if self.carry > 0:
            target = self.model.select_larva_for(self)
            if target:
                self._move_towards(target.pos)
                # feed if close enough
                if self.model.distance(self.pos, target.pos) < getattr(self.model, 'feed_radius', 0.4):
                    amt = min(self.carry, 2)
                    try:
                        target.feed(self.model.model_time, self.unique_id, amt, self.current_bout_id)
                    except TypeError:
                        target.feed(self.model.model_time)
                    self.carry -= amt
                    self.feed_events.append((self.model.model_time, target.cell_id, amt))
                    if hasattr(self.model, 'global_feed_events'):
                        self.model.global_feed_events.append((self.unique_id, target.cell_id, self.model.model_time, amt, self.current_bout_id))
                    if hasattr(self.model, 'log_feed'):
                        self.model.log_feed(self.unique_id, target.cell_id, amt)
                    return
                else:
                    # check but don't feed - record inefficiency
                    if self.current_bout_id and hasattr(self.model, 'record_larva_checked_but_not_fed'):
                        self.model.record_larva_checked_but_not_fed(self.current_bout_id, target.cell_id, self.unique_id, self.model.model_time)
            # otherwise wander
            self._move_forward_biased(step_size=getattr(self.model, 'step_size', 0.5)*0.8)
        else:
            # not carrying -> look for donors (forager/primary)
            if self.role == 'primary':
                # primary stay near entrance with higher probability
                ent = self.model.entrance_pos
                if random.random() < 0.15:
                    self._attempt_move((ent[0] + random.uniform(-0.2,0.2), ent[1] + random.uniform(-0.2,0.2)))
            near = self.model.agents_in_radius(self.pos, getattr(self.model, 'perception_radius', 1.2), exclude=self)
            donors = [a for a in near if a.role in ('forager','primary') and getattr(a, 'carry', 0) > 0]
            if donors and random.random() < 0.6:
                donor = random.choice(donors)
                cap_rem = getattr(self.model, 'receiver_capacity', 5) - getattr(self, 'carry', 0)
                if cap_rem > 0:
                    give = min(cap_rem, getattr(donor, 'carry', 0), 2)
                    if give > 0:
                        donor.carry -= give
                        self.carry += give
                        self.transfer_events.append((self.model.model_time, donor.unique_id, give))
                        if hasattr(self.model, 'global_transfers'):
                            self.model.global_transfers.append((donor.unique_id, self.unique_id, give, self.model.model_time))
                        if hasattr(self.model, 'log_transfer'):
                            self.model.log_transfer(donor.unique_id, self.unique_id, give)
                        return
            # drift
            if random.random() < 0.25:
                self._move_forward_biased(step_size=getattr(self.model, 'step_size', 0.5)*0.6)

    def _should_switch_role(self):
        """Determine if agent should switch roles based on congestion/food availability"""
        if self.role == 'idle':
            # idle can become primary if food is arriving
            if hasattr(self.model, 'should_promote_idle_to_receiver') and self.model.should_promote_idle_to_receiver():
                return True
        elif self.role == 'secondary':
            # secondary can become feeder if carrying food
            if self.carry > 0 and random.random() < 0.05:
                return True
        elif self.role == 'feeder':
            # feeder can become idle if no food found for a while
            if len(self.feed_events) == 0 and random.random() < 0.02:
                return True
        return False

    def _switch_role(self):
        """Switch role and track the event"""
        old_role = self.role
        if old_role == 'idle':
            self.role = 'primary'
        elif old_role == 'secondary':
            self.role = 'feeder'
        elif old_role == 'feeder':
            self.role = 'idle'
        self.role_switch_count += 1

    def _idle_step(self):
        if random.random() < 0.08:
            self._move_forward_biased(step_size=getattr(self.model, 'step_size', 0.5)*0.4)
        if hasattr(self.model, 'should_promote_idle_to_receiver') and self.model.should_promote_idle_to_receiver():
            self.role = 'primary'

def make_agent(uid, model, role='idle'):
    return WaspAgent(uid, model, role)
