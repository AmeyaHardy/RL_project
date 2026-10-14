from __future__ import annotations

import functools
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

UP, DOWN, LEFT, RIGHT, STAY, PICK, DROP = range(7)
N_ACTIONS = 7
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}  # (dy, dx)
PERSIST, TEMP, MAL = 0, 1, 2
TYPE_NAMES = ("persistent", "temporary", "malicious")
N_LOCAL_CH = 6   # obstacle, persistent, temporary(remaining life), malicious, depot, other agents
N_TEAM_CH = 6    # known, known-obstacle, persistent, temporary(remaining life), malicious, agents
N_SCALARS = 12
HANDLING_TIME = 2  # pick + drop steps, used in the slack estimate


@dataclass
class EnvConfig:
    height: int = 20
    width: int = 20
    n_agents: int = 4
    obstacle_density: tuple = (0.10, 0.25)
    min_free_frac: float = 0.55             
    view_radius: int = 3                    # local window is (2r+1) x (2r+1)
    start_radius: int = 4                   # agents start within this path distance of depot
    # items
    n_persistent: int = 10
    n_malicious: int = 3
    temp_spawn_prob: float = 0.04           # per step
    temp_life_range: tuple = (40, 80)       # lifetime in steps
    max_temp_active: int = 5
    forbid_malicious_pickup: bool = False   # True -> hard action mask instead of penalty
    # episode
    max_steps: int = 300
    target_deliveries: Optional[int] = None  # terminate early after X valid deliveries
    # rewards
    v_low: float = 1.0        # persistent delivery
    v_high: float = 5.0       # temporary delivery (before expiry)
    p_mal: float = 2.0        # penalty for picking up malicious item
    c_time: float = 0.01      # per-step cost (team)
    c_collision: float = 0.05
    c_invalid: float = 0.02
    alpha_map: float = 0.5    # x fraction of newly revealed map cells
    b_discover: tuple = (0.05, 0.20, 0.0)  # first-discovery bonus by item type
    coord_penalty: float = 0.1             # crowding on the same item (see _coordination)
    crowd_radius: int = 2


class SwarmSearchEnv(ParallelEnv):
    metadata = {"name": "swarm_search_v0", "render_modes": ["ansi", "rgb_array"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.possible_agents = [f"robot_{i}" for i in range(self.cfg.n_agents)]
        self.agents = []
        self.rng = np.random.default_rng()
        self._life_max = float(self.cfg.temp_life_range[1])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        c = self.cfg
        w = 2 * c.view_radius + 1
        return spaces.Dict({
            "local": spaces.Box(0.0, 1.0, (N_LOCAL_CH, w, w), np.float32),
            "team_map": spaces.Box(0.0, 1.0, (N_TEAM_CH, c.height, c.width), np.float32),
            "scalars": spaces.Box(-1.0, 1.0, (N_SCALARS,), np.float32),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(N_ACTIONS)

    @property
    def state_space(self):
        c = self.cfg
        n = c.height * c.width * (1 + 1 + 3 + 1 + 1) + c.n_agents * 3 + 2
        return spaces.Box(-1.0, 1.0, (n,), np.float32)

    @staticmethod
    def _bfs(grid: np.ndarray, start) -> np.ndarray:
        H, W = grid.shape
        dist = -np.ones((H, W), dtype=np.int32)
        dist[start] = 0
        q = deque([start])
        while q:
            y, x = q.popleft()
            for dy, dx in MOVES.values():
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not grid[ny, nx] and dist[ny, nx] < 0:
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((ny, nx))
        return dist

    def _generate_map(self):
        c = self.cfg
        need = c.n_agents + c.n_persistent + c.n_malicious + c.max_temp_active + 1
        for _ in range(1000):
            dens = self.rng.uniform(*c.obstacle_density)
            grid = self.rng.random((c.height, c.width)) < dens
            free = np.argwhere(~grid)
            if len(free) == 0:
                continue
            depot = tuple(int(v) for v in free[self.rng.integers(len(free))])
            dist = self._bfs(grid, depot)
            reach = dist >= 0
            if reach.sum() < max(c.min_free_frac * c.height * c.width, need):
                continue
            self.grid = grid | ~reach          # wall off unreachable pockets
            self.depot = depot
            self.depot_dist = dist
            self.free_cells = np.argwhere(~self.grid)
            return
        raise RuntimeError("Could not generate a valid map; relax obstacle_density/min_free_frac.")

    # ------------------------------------------------------------------ items
    def _new_item(self, typ, pos, expiry=np.inf):
        return {"type": typ, "pos": pos, "spawn": self.t, "expiry": expiry, "seen": False}

    def _random_free_cell(self, exclude):
        for _ in range(200):
            y, x = self.free_cells[self.rng.integers(len(self.free_cells))]
            p = (int(y), int(x))
            if p != self.depot and p not in exclude:
                return p
        return None

    def _spawn_temp(self):
        pos = self._random_free_cell(self.active)
        if pos is None:
            return
        lo, hi = self.cfg.temp_life_range
        life = int(self.rng.integers(lo, hi + 1))
        self.active[pos] = self._new_item(TEMP, pos, self.t + life)

    def _n_temp_alive(self):
        n = sum(1 for it in self.active.values() if it["type"] == TEMP)
        return n + sum(1 for it in self.carry if it is not None and it["type"] == TEMP)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        c = self.cfg
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._generate_map()
        self.t = 0
        self.known = np.zeros((c.height, c.width), dtype=bool)
        self.carry = [None] * c.n_agents
        self.active = {}
        self.stats = {
            "deliveries": 0, "delivered_value": 0.0,
            "delivered_by_type": [0, 0, 0], "expired_by_type": [0, 0, 0],
            "malicious_picked": 0, "temp_spawned": 0, "collisions": 0,
            "discovered_by_type": [0, 0, 0], "ep_return": 0.0,
        }

        # agent starts near depot (not on it)
        R = c.start_radius
        d = self.depot_dist
        cand = np.argwhere((d > 0) & (d <= R))
        while len(cand) < c.n_agents:
            R += 1
            cand = np.argwhere((d > 0) & (d <= R))
        idx = self.rng.choice(len(cand), c.n_agents, replace=False)
        self.pos = [tuple(int(v) for v in cand[i]) for i in idx]

        taken = set(self.pos)
        for typ, n in ((PERSIST, c.n_persistent), (MAL, c.n_malicious)):
            for _ in range(n):
                p = self._random_free_cell(taken | set(self.active))
                if p is not None:
                    self.active[p] = self._new_item(typ, p)

        self._update_visibility()  # initial discoveries are free
        return self._observations(), self._infos({})

    # ------------------------------------------------------------- visibility
    def _update_visibility(self):
        """Reveal windows around every robot (team shares the map).
        Returns (fraction of newly revealed cells, list of newly discovered items)."""
        c = self.cfg
        before = int(self.known.sum())
        r = c.view_radius
        for (y, x) in self.pos:
            self.known[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1] = True
        new_frac = (int(self.known.sum()) - before) / (c.height * c.width)
        new_items = []
        for p, it in self.active.items():
            if not it["seen"] and self.known[p]:
                it["seen"] = True
                new_items.append(it)
        return new_frac, new_items

    # ------------------------------------------------------------------- step
    def step(self, actions):
        c = self.cfg
        assert self.agents, "call reset() first"
        comp = dict(time=-c.c_time, collision=0.0, invalid=0.0, map=0.0,
                    discover=0.0, delivery=0.0, coordination=0.0)
        n_inv = 0

        acts = [int(actions.get(a, STAY)) for a in self.possible_agents]

        # 1) pick / drop
        for i, a in enumerate(acts):
            if a == PICK:
                if not self._can_pick(i):
                    n_inv += 1
                    continue
                it = self.active.pop(self.pos[i])
                if it["type"] == MAL:
                    comp["delivery"] -= c.p_mal           # item is destroyed
                    self.stats["malicious_picked"] += 1
                else:
                    self.carry[i] = it
            elif a == DROP:
                if not self._can_drop(i):
                    n_inv += 1
                    continue
                it, self.carry[i] = self.carry[i], None
                val = c.v_high if it["type"] == TEMP else c.v_low
                comp["delivery"] += val
                self.stats["deliveries"] += 1
                self.stats["delivered_value"] += val
                self.stats["delivered_by_type"][it["type"]] += 1

        # 2) movement with conflict resolution
        n_col = self._move(acts)
        comp["collision"] = -c.c_collision * n_col
        comp["invalid"] = -c.c_invalid * n_inv
        self.stats["collisions"] += n_col

        # 3) time, expiry, spawning
        self.t += 1
        for p in [p for p, it in self.active.items() if it["expiry"] <= self.t]:
            self.stats["expired_by_type"][self.active.pop(p)["type"]] += 1
        for i, it in enumerate(self.carry):
            if it is not None and it["expiry"] <= self.t:      # expired while carried
                self.stats["expired_by_type"][it["type"]] += 1
                self.carry[i] = None
        if self._n_temp_alive() < c.max_temp_active and self.rng.random() < c.temp_spawn_prob:
            self._spawn_temp()
            self.stats["temp_spawned"] += 1

        # 4) discovery (counted once per cell / item for the whole team)
        new_frac, new_items = self._update_visibility()
        comp["map"] = c.alpha_map * new_frac
        for it in new_items:
            comp["discover"] += c.b_discover[it["type"]]
            self.stats["discovered_by_type"][it["type"]] += 1

        # 5) coordination
        comp["coordination"] = self._coordination()

        reward = float(sum(comp.values()))
        self.stats["ep_return"] += reward

        terminated = c.target_deliveries is not None and self.stats["deliveries"] >= c.target_deliveries
        truncated = (self.t >= c.max_steps) and not terminated

        obs = self._observations()
        infos = self._infos(comp)
        agents_now = list(self.agents)
        rewards = {a: reward for a in agents_now}
        terms = {a: terminated for a in agents_now}
        truncs = {a: truncated for a in agents_now}
        if terminated or truncated:
            for a in agents_now:
                infos[a]["episode_stats"] = dict(self.stats)
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def _can_pick(self, i):
        if self.carry[i] is not None:
            return False
        it = self.active.get(self.pos[i])
        if it is None:
            return False
        return not (it["type"] == MAL and self.cfg.forbid_malicious_pickup)

    def _can_drop(self, i):
        return self.carry[i] is not None and self.pos[i] == self.depot

    def _in_free(self, y, x):
        return 0 <= y < self.cfg.height and 0 <= x < self.cfg.width and not self.grid[y, x]

    def _move(self, acts):
        n = self.cfg.n_agents
        cur = list(self.pos)
        des = list(cur)
        col = [False] * n
        for i, a in enumerate(acts):
            if a in MOVES:
                dy, dx = MOVES[a]
                ty, tx = cur[i][0] + dy, cur[i][1] + dx
                if self._in_free(ty, tx):
                    des[i] = (ty, tx)
                else:
                    col[i] = True  # bumped a wall
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if des[i] == cur[i]:
                    continue
                for j in range(n):
                    if j == i:
                        continue
                    same_target = des[j] == des[i]                       # includes a staying robot
                    swap = des[i] == cur[j] and des[j] == cur[i]
                    if same_target or swap:
                        des[i] = cur[i]
                        col[i] = True
                        changed = True
                        break
        self.pos = des
        return int(sum(col))

    def _coordination(self):
        c = self.cfg
        pen = 0.0
        free_agents = [self.pos[i] for i in range(c.n_agents) if self.carry[i] is None]
        for p, it in self.active.items():
            if not it["seen"] or it["type"] == MAL:
                continue
            k = sum(1 for q in free_agents
                    if max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= c.crowd_radius)
            if k >= 2:
                pen -= c.coord_penalty * (k - 1)
        return pen

    def action_mask(self, agent):
        i = self.possible_agents.index(agent)
        m = np.zeros(N_ACTIONS, dtype=bool)
        y, x = self.pos[i]
        for a, (dy, dx) in MOVES.items():
            m[a] = self._in_free(y + dy, x + dx)
        m[STAY] = True
        m[PICK] = self._can_pick(i)
        m[DROP] = self._can_drop(i)
        return m

    def _remaining_frac(self, it):
        return float(np.clip((it["expiry"] - self.t) / self._life_max, 0.0, 1.0))

    def _item_channels(self, seen_only):
        ch = np.zeros((3, self.cfg.height, self.cfg.width), dtype=np.float32)
        for (y, x), it in self.active.items():
            if seen_only and not it["seen"]:
                continue
            ch[it["type"], y, x] = self._remaining_frac(it) if it["type"] == TEMP else 1.0
        return ch

    def _agent_map(self):
        m = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        for (y, x) in self.pos:
            m[y, x] = 1.0
        return m

    def _observations(self):
        c = self.cfg
        r, H, W = c.view_radius, c.height, c.width
        w = 2 * r + 1

        items = self._item_channels(seen_only=False)
        agents = self._agent_map()
        depot = np.zeros((H, W), dtype=np.float32)
        depot[self.depot] = 1.0
        world = np.concatenate([
            self.grid[None].astype(np.float32), items, depot[None], agents[None]], axis=0)
        padded = np.pad(world, ((0, 0), (r, r), (r, r)))
        padded[0, :r, :] = padded[0, -r:, :] = 1.0  
        padded[0, :, :r] = padded[0, :, -r:] = 1.0

        titems = self._item_channels(seen_only=True)
        team = np.concatenate([
            self.known[None].astype(np.float32),
            (self.known & self.grid)[None].astype(np.float32),
            titems, agents[None]], axis=0)

        known_val = [it for it in self.active.values() if it["seen"] and it["type"] != MAL]
        obs = {}
        for i, a in enumerate(self.possible_agents):
            y, x = self.pos[i]
            local = padded[:, y: y + w, x: x + w].copy()
            local[5, r, r] = 0.0  # remove self from "other agents"
            obs[a] = {"local": local, "team_map": team.copy(),
                      "scalars": self._scalars(i, known_val)}
        return obs

    def _slack(self, i, it):
        d_item = abs(self.pos[i][0] - it["pos"][0]) + abs(self.pos[i][1] - it["pos"][1])
        d_dep = self.depot_dist[it["pos"]]
        return it["expiry"] - self.t - (d_item + d_dep + HANDLING_TIME)

    def _scalars(self, i, known_val):
        c = self.cfg
        y, x = self.pos[i]
        carried = self.carry[i]
        s = np.zeros(N_SCALARS, dtype=np.float32)
        s[0], s[1] = y / c.height, x / c.width
        s[2] = float(carried is not None)
        s[3] = self._remaining_frac(carried) if (carried and carried["type"] == TEMP) else 0.0
        s[4] = self.t / c.max_steps
        s[5] = min(self.stats["deliveries"] / (c.target_deliveries or 20), 1.0)
        s[6], s[7] = (self.depot[0] - y) / c.height, (self.depot[1] - x) / c.width
        slack = 1.0
        if carried is not None:
            if carried["type"] == TEMP:
                slack = (carried["expiry"] - self.t - self.depot_dist[y, x] - 1) / self._life_max
        elif known_val:
            near = min(known_val, key=lambda it: abs(it["pos"][0] - y) + abs(it["pos"][1] - x))
            s[8] = (near["pos"][0] - y) / c.height
            s[9] = (near["pos"][1] - x) / c.width
            temps = [it for it in known_val if it["type"] == TEMP]
            if temps:
                slack = min(self._slack(i, it) for it in temps) / self._life_max
        s[10] = float(np.clip(slack, -1.0, 1.0))
        s[11] = min(len(known_val) / 10.0, 1.0)
        return s

    def _infos(self, comp):
        infos = {}
        for a in self.agents:
            infos[a] = {
                "action_mask": self.action_mask(a),
                "reward_components": dict(comp),
                "deliveries": self.stats["deliveries"],
            }
        return infos

    def state(self):
        """Full global state for a centralized critic (CTDE)."""
        c = self.cfg
        items = self._item_channels(seen_only=False)
        depot = np.zeros((c.height, c.width), dtype=np.float32)
        depot[self.depot] = 1.0
        agent_feats = []
        for i in range(c.n_agents):
            y, x = self.pos[i]
            carried = self.carry[i]
            agent_feats += [y / c.height, x / c.width, float(carried is not None)]
        return np.concatenate([
            self.grid.astype(np.float32).ravel(), depot.ravel(), items.ravel(),
            self.known.astype(np.float32).ravel(), self._agent_map().ravel(),
            np.array(agent_feats + [self.t / c.max_steps,
                                    min(self.stats["deliveries"] / (c.target_deliveries or 20), 1.0)],
                     dtype=np.float32)]).astype(np.float32)

    def render(self, cell_px: int = 24):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb(cell_px)
        return None

    def _render_ansi(self):
        c = self.cfg
        rows = [["#" if self.grid[y, x] else "." for x in range(c.width)] for y in range(c.height)]
        sym = {PERSIST: "p", TEMP: "T", MAL: "m"}
        for (y, x), it in self.active.items():
            rows[y][x] = sym[it["type"]]
        rows[self.depot[0]][self.depot[1]] = "D"
        for i, (y, x) in enumerate(self.pos):
            rows[y][x] = str(i) if self.carry[i] is None else chr(ord("a") + i)
        return "\n".join("".join(r) for r in rows)

    def _render_rgb(self, px):
        c = self.cfg
        img = np.ones((c.height, c.width, 3), dtype=np.float32)
        img[self.grid] = (0.25, 0.25, 0.28)
        cols = {PERSIST: (0.95, 0.85, 0.2), TEMP: (1.0, 0.5, 0.0), MAL: (0.85, 0.1, 0.1)}
        for (y, x), it in self.active.items():
            img[y, x] = cols[it["type"]]
        img[self.depot] = (0.1, 0.65, 0.3)
        for i, (y, x) in enumerate(self.pos):
            img[y, x] = (0.0, 0.8, 0.9) if self.carry[i] is not None else (0.15, 0.3, 0.9)
        img[~self.known] *= 0.45   # fog of war
        return (np.kron(img, np.ones((px, px, 1))) * 255).astype(np.uint8)