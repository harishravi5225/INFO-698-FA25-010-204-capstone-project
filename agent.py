# agent.py
# agent.py
import math
import random
from mesa import Agent

class WaspAgent(Agent):
    """
    Wasp agent roles:
    - forager: leaves nest, returns with food
    - unloader: receives food & delivers to larvae
    - idle: slow movement, may convert to unloader
    """

    def __init__(self, unique_id, model, role='idle'):
        super().__init__(unique_id, model)
        self.role = role
        self.carrying = 0
        self.orientation = None
        self.path = []
        self.feed_events = []
        self.transfer_events = []

        if role == 'forager':
            self.time_out = max(1, int(random.expovariate(
                1.0 / model.forager_mean_return
            )))
        else:
            self.time_out = 0

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, v):
        self._pos = v

    def step(self):
        self.path.append(self.pos)

        if self.role == 'forager':
            self.handle_forager()

        elif self.role == 'unloader':
            self.handle_unloader()

        else:
            self.handle_idle()

    # ------------------ Movement helpers ------------------ #
    def move_forward_biased(self, step_size=0.5):
        if self.orientation is None or random.random() > self.model.forward_bias:
            ang = random.random() * 2 * math.pi
            dx, dy = math.cos(ang), math.sin(ang)
        else:
            ox, oy = self.orientation
            jitter = random.gauss(0, 0.6)
            ang = math.atan2(oy, ox) + jitter
            dx, dy = math.cos(ang), math.sin(ang)

        nx = min(max(self.pos[0] + dx * step_size, self.model.xmin), self.model.xmax)
        ny = min(max(self.pos[1] + dy * step_size, self.model.ymin), self.model.ymax)

        self.orientation = (dx, dy)
        self.pos = (nx, ny)

    def move_towards(self, target, step_size=0.6):
        tx, ty = target
        x, y = self.pos
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return

        ux, uy = dx / dist, dy / dist
        nx = min(max(x + ux * step_size, self.model.xmin), self.model.xmax)
        ny = min(max(y + uy * step_size, self.model.ymin), self.model.ymax)
        self.orientation = (ux, uy)
        self.pos = (nx, ny)

    # ------------------ Role behaviors ------------------ #
    def handle_forager(self):
        if self.carrying == 0 and self.time_out > 0:
            self.time_out -= 1
            return

        if self.carrying == 0 and self.time_out <= 0:
            self.pos = self.model.entrance_pos
            self.carrying = 1
            self.model.food_arrivals.append((self.unique_id, self.model.time, self.pos))
            return

        nearby = self.model.agents_in_radius(self.pos, self.model.perception_radius, exclude=self)
        receivers = [a for a in nearby if a.role in ('unloader', 'idle') and a.carrying == 0]

        if receivers and random.random() < self.model.forager_transfer_prob:
            rec = random.choice(receivers)
            rec.carrying += 1
            self.carrying -= 1
            self.transfer_events.append((self.model.time, rec.unique_id))
            self.model.global_transfers.append((self.unique_id, rec.unique_id, self.model.time))
            return

        self.move_forward_biased(self.model.step_size)

    def handle_unloader(self):
        if self.carrying > 0:
            larva = self.model.select_larva_for(self)
            if larva:
                self.move_towards(larva.pos, self.model.step_size)
                if self.model.distance(self.pos, larva.pos) < self.model.feed_radius:
                    larva.feed(self.model.time, self.unique_id)
                    self.feed_events.append((self.model.time, larva.cell_id))
                    self.model.global_feed_events.append(
                        (self.unique_id, larva.cell_id, self.model.time)
                    )
                    self.carrying -= 1
            else:
                self.move_forward_biased(self.model.step_size * 0.6)
        else:
            self.move_forward_biased(self.model.step_size * 0.5)

    def handle_idle(self):
        if random.random() < 0.12:
            self.move_forward_biased(self.model.step_size * 0.4)

        busy = any(a.carrying > 0 for a in self.model.agents_in_radius(self.pos, self.model.perception_radius, exclude=self))
        if busy and random.random() < 0.03:
            self.role = 'unloader'

