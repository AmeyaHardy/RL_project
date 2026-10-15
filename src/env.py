from __future__ import annotations

import copy
import functools
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

UP, DOWN, LEFT, RIGHT, STAY, PICK, DROP = range(7)
N_ACTIONS = 7
MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
PERSIST, TEMP, MAL = 0, 1, 2
TYPE_NAMES = ("persistent", "temporary", "malicious")
N_LOCAL_CH = 6
N_TEAM_CH = 7
N_SCALARS = 23
HANDLING_TIME = 2


class Status(IntEnum):
    AVAILABLE = 0
    CLAIMED = 1
    CARRIED = 2
    DELIVERED = 3
    EXPIRED = 4
    DISPOSED = 5


class Mode(IntEnum):
    EXPLORE = 0
    PURSUE = 1
    RETURN = 2


@dataclass
class Item:
    id: int
    type: int
    pos: tuple
    spawn: int
    expiry: float
    value: float
    status: Status = Status.AVAILABLE
    seen: bool = False
    discovered_by: Optional[int] = None
    discovered_t: Optional[int] = None
    claimed_by: Optional[int] = None
    collected_by: Optional[int] = None
    pickup_t: Optional[int] = None
    end_t: Optional[int] = None


@dataclass
class EnvConfig:
    height: int = 20
    width: int = 20
    n_agents: int = 4
    obstacle_density: tuple = (0.10, 0.25)
    min_free_frac: float = 0.55
    view_radius: int = 3
    start_radius: int = 4
    checkpoint_xy: Optional[tuple] = None
    checkpoint_stacking: bool = True
    checkpoint_service_rate: Optional[int] = None
    service_priority: str = "urgency"
    n_persistent: int = 10
    n_malicious: int = 3
    n_temp_initial: int = 0
    temp_spawn_prob: float = 0.04
    persist_spawn_prob: float = 0.0
    mal_spawn_prob: float = 0.0
    temp_life_range: tuple = (40, 80)
    max_temp_active: int = 5
    max_persist_active: int = 30
    max_mal_active: int = 10
    forbid_malicious_pickup: bool = False
    mal_on_pickup: str = "destroy"
    temp_expires_while_carried: bool = True
    tiebreak: str = "index"
    claim_blocks_pickup: bool = False
    slack_uses_true_path: bool = False
    info_mode: str = "shared"
    max_steps: int = 300
    target_deliveries: Optional[int] = None
    v_low: float = 1.0
    v_high: float = 5.0
    p_mal: float = 2.0
    c_time: float = 0.01
    c_collision: float = 0.05
    c_invalid: float = 0.02
    alpha_map: float = 0.5
    b_discover: tuple = (0.05, 0.20, 0.0)
    coord_penalty: float = 0.1
    crowd_radius: int = 2
    count_initial_discovery: bool = False

    def __post_init__(self):
        """Validate configuration settings and checkpoint positioning."""
        assert self.mal_on_pickup in ("destroy", "carry")
        assert self.service_priority in ("urgency", "index", "random")
        assert self.tiebreak in ("index", "random")
        assert self.info_mode in ("shared", "private", "window")
        assert self.checkpoint_service_rate is None or self.checkpoint_service_rate >= 1
        if self.checkpoint_xy is not None:
            x, y = self.checkpoint_xy
            assert 0 <= x < self.width and 0 <= y < self.height, "checkpoint_xy outside the grid"


_C_OBST = (0.25, 0.25, 0.28)
_C_UNKNOWN = (0.12, 0.12, 0.14)
_C_CP = (0.1, 0.65, 0.3)
_C_CLAIM = (0.8, 0.0, 0.8)
_C_SELF = (1.0, 0.9, 0.0)
_C_AGENT_OTHER = (0.15, 0.3, 0.9)
_ITEM_COL = {PERSIST: (0.95, 0.85, 0.2), TEMP: (1.0, 0.5, 0.0), MAL: (0.85, 0.1, 0.1)}
_AGENT_COLS = [
    (0.15, 0.3, 0.9), (0.0, 0.6, 0.6), (0.6, 0.2, 0.8), (0.1, 0.5, 0.1),
    (0.5, 0.3, 0.0), (0.9, 0.3, 0.6), (0.3, 0.3, 0.3), (0.0, 0.0, 0.6)
]


class SwarmSearchEnv(ParallelEnv):
    metadata = {"name": "swarm_search_v1", "render_modes": ["ansi", "rgb_array"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        """Initialize environment configuration, agent identifiers, and RNG."""
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.possible_agents = [f"robot_{i}" for i in range(self.cfg.n_agents)]
        self._idx = {a: i for i, a in enumerate(self.possible_agents)}
        self.agents = []
        self.rng = np.random.default_rng()
        self._life_max = float(self.cfg.temp_life_range[1])

    @property
    def checkpoint_xy(self):
        """Return checkpoint location as (x, y)."""
        return (self.checkpoint[1], self.checkpoint[0])

    @property
    def depot(self):
        """Return checkpoint location as (y, x)."""
        return self.checkpoint

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Return Dict observation space for agent."""
        c = self.cfg
        w = 2 * c.view_radius + 1
        return spaces.Dict({
            "local": spaces.Box(0.0, 1.0, (N_LOCAL_CH, w, w), np.float32),
            "team_map": spaces.Box(0.0, 1.0, (N_TEAM_CH, c.height, c.width), np.float32),
            "scalars": spaces.Box(-1.0, 1.0, (N_SCALARS,), np.float32),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Return discrete action space for agent."""
        return spaces.Discrete(N_ACTIONS)

    @property
    def state_space(self):
        """Return centralized state space box."""
        c = self.cfg
        n = c.height * c.width * 7 + c.n_agents * 4 + 2
        return spaces.Box(-1.0, 1.0, (n,), np.float32)

    @staticmethod
    def _bfs(grid: np.ndarray, start) -> np.ndarray:
        """Compute 2D shortest-path distance array from start."""
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
        """Generate connected random grid map satisfying density and reachability."""
        c = self.cfg
        need = (c.n_agents + c.n_persistent + c.n_malicious + max(c.n_temp_initial, c.max_temp_active) + 1)
        fixed = None if c.checkpoint_xy is None else (int(c.checkpoint_xy[1]), int(c.checkpoint_xy[0]))
        for _ in range(1000):
            dens = self.rng.uniform(*c.obstacle_density)
            grid = self.rng.random((c.height, c.width)) < dens
            if fixed is not None:
                grid[fixed] = False
                cp = fixed
            else:
                free = np.argwhere(~grid)
                if len(free) == 0:
                    continue
                cp = tuple(int(v) for v in free[self.rng.integers(len(free))])
            dist = self._bfs(grid, cp)
            reach = dist >= 0
            if reach.sum() < max(c.min_free_frac * c.height * c.width, need):
                continue
            self.grid = grid | ~reach
            self.checkpoint = cp
            self.checkpoint_dist = dist
            self.free_cells = np.argwhere(~self.grid)
            return
        raise RuntimeError("Could not generate a valid map; relax obstacle_density/min_free_frac.")

    def _new_item(self, typ, pos, expiry=np.inf) -> Item:
        """Create and register a new item."""
        c = self.cfg
        value = {PERSIST: c.v_low, TEMP: c.v_high, MAL: -c.p_mal}[typ]
        it = Item(id=self._next_id, type=typ, pos=pos, spawn=self.t, expiry=expiry, value=value)
        self._next_id += 1
        self.objects[it.id] = it
        return it

    def _random_free_cell(self, exclude):
        """Sample an unoccupied free cell not in exclude."""
        for _ in range(200):
            y, x = self.free_cells[self.rng.integers(len(self.free_cells))]
            p = (int(y), int(x))
            if p != self.checkpoint and p not in exclude:
                return p
        return None

    def _spawn_item(self, typ) -> bool:
        """Spawn a single item of typ at a random free cell."""
        pos = self._random_free_cell(set(self.active) | set(self.pos))
        if pos is None:
            return False
        if typ == TEMP:
            lo, hi = self.cfg.temp_life_range
            life = int(self.rng.integers(lo, hi + 1))
            self.active[pos] = self._new_item(TEMP, pos, self.t + life)
        else:
            self.active[pos] = self._new_item(typ, pos)
        return True

    def _spawn_temp(self) -> bool:
        """Spawn a temporary item."""
        return self._spawn_item(TEMP)

    def _count_type(self, typ) -> int:
        """Count active ground and carried items of typ."""
        n = sum(1 for it in self.active.values() if it.type == typ)
        return n + sum(1 for it in self.carry if it is not None and it.type == typ)

    def _n_temp_alive(self):
        """Count active temporary items."""
        return self._count_type(TEMP)

    def reset(self, seed=None, options=None):
        """Reset environment state, generate map, and place agents."""
        c = self.cfg
        n = c.n_agents
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._generate_map()
        self.t = 0
        self._next_id = 0
        self.objects = {}
        self.known = np.zeros((c.height, c.width), dtype=bool)
        self._agent_known = [np.zeros((c.height, c.width), dtype=bool) for _ in range(n)]
        self._seen_by = [set() for _ in range(n)]
        self.carry = [None] * n
        self.target = [None] * n
        self.active = {}
        self.stats = {
            "deliveries": 0, "delivered_value": 0.0,
            "delivered_by_type": [0, 0, 0], "expired_by_type": [0, 0, 0],
            "expired_after_discovery": 0, "expired_while_carried": 0,
            "malicious_picked": 0, "disposed": 0,
            "pickups": 0, "pickup_by_type": [0, 0, 0],
            "temp_spawned": 0, "persist_spawned": 0, "mal_spawned": 0,
            "collisions": 0,
            "discovered_by_type": [0, 0, 0], "discovered_by_agent": [0] * n,
            "personal_discovered_by_agent": [0] * n,
            "pickup_conflicts": 0, "service_denied": 0,
            "claim_conflicts": 0, "invalid_targets": 0,
            "delivery_log": [],
            "carrying_per_step": [],
            "steps_returning": [0] * n,
            "steps_waiting": [0] * n,
            "distance": [0] * n,
            "ep_return": 0.0,
        }
        R = c.start_radius
        d = self.checkpoint_dist
        cand = np.argwhere((d > 0) & (d <= R))
        while len(cand) < n:
            R += 1
            cand = np.argwhere((d > 0) & (d <= R))
        idx = self.rng.choice(len(cand), n, replace=False)
        self.pos = [tuple(int(v) for v in cand[i]) for i in idx]

        taken = set(self.pos)
        for typ, k in ((PERSIST, c.n_persistent), (MAL, c.n_malicious)):
            for _ in range(k):
                p = self._random_free_cell(taken | set(self.active))
                if p is not None:
                    self.active[p] = self._new_item(typ, p)
        for _ in range(c.n_temp_initial):
            self._spawn_item(TEMP)

        _, init_items = self._update_visibility(count_personal=c.count_initial_discovery)
        if c.count_initial_discovery:
            for it in init_items:
                self.stats["discovered_by_type"][it.type] += 1
                if it.discovered_by is not None:
                    self.stats["discovered_by_agent"][it.discovered_by] += 1

        return self._observations(), self._infos({})

    def _update_visibility(self, count_personal: bool = True):
        """Reveal Chebyshev sensing windows and register newly discovered items."""
        c = self.cfg
        before = int(self.known.sum())
        r = c.view_radius
        for i, (y, x) in enumerate(self.pos):
            slc = (slice(max(0, y - r), y + r + 1), slice(max(0, x - r), x + r + 1))
            self.known[slc] = True
            self._agent_known[i][slc] = True
        new_frac = (int(self.known.sum()) - before) / (c.height * c.width)
        new_items = []
        for i, (y, x) in enumerate(self.pos):
            for p, it in self.active.items():
                if max(abs(p[0] - y), abs(p[1] - x)) <= r:
                    if it.id not in self._seen_by[i]:
                        self._seen_by[i].add(it.id)
                        if count_personal:
                            self.stats["personal_discovered_by_agent"][i] += 1
                    if not it.seen:
                        it.seen, it.discovered_by, it.discovered_t = True, i, self.t
                        new_items.append(it)
        return new_frac, new_items

    def _order(self, idxs, mode):
        """Order agent indices deterministically or via random permutation."""
        idxs = sorted(idxs)
        if mode == "random" and len(idxs) > 1:
            return [int(v) for v in self.rng.permutation(idxs)]
        return idxs

    def _clear_claim(self, it: Item):
        """Clear claiming agent assignment from item."""
        j = it.claimed_by
        if j is not None:
            if self.target[j] == it.id:
                self.target[j] = None
            it.claimed_by = None
            if it.status == Status.CLAIMED:
                it.status = Status.AVAILABLE

    def _release(self, i):
        """Release claim on target assigned to agent i."""
        tgt = self.target[i]
        if tgt is not None:
            self._clear_claim(self.objects[tgt])
        self.target[i] = None

    def _resolve_target(self, v) -> Optional[Item]:
        """Resolve item ID or coordinate tuple into Item."""
        if isinstance(v, (int, np.integer)):
            return self.objects.get(int(v))
        y, x = v
        return self.active.get((int(y), int(x)))

    def _apply_targets(self, targets):
        """Process item target claims submitted by agents."""
        reqs = []
        for a, v in targets.items():
            if a not in self._idx:
                continue
            if v is None:
                self._release(self._idx[a])
            else:
                reqs.append(self._idx[a])
        for i in self._order(reqs, self.cfg.tiebreak):
            it = self._resolve_target(targets[self.possible_agents[i]])
            if (it is None or self.carry[i] is not None or it.type == MAL
                    or not self._is_item_visible_to(i, it)
                    or it.status not in (Status.AVAILABLE, Status.CLAIMED)):
                self.stats["invalid_targets"] += 1
                continue
            if it.claimed_by == i:
                continue
            if it.claimed_by is not None:
                self.stats["claim_conflicts"] += 1
                continue
            self._release(i)
            it.status, it.claimed_by, self.target[i] = Status.CLAIMED, i, it.id

    def step(self, actions, targets=None):
        """Advance simulation by executing actions and handling events."""
        c = self.cfg
        assert self.agents, "call reset() first"
        comp = dict(time=-c.c_time, collision=0.0, invalid=0.0, map=0.0,
                    discover=0.0, delivery=0.0, coordination=0.0)
        n_inv = 0

        acts = []
        for a in self.possible_agents:
            v = int(actions.get(a, STAY))
            if not 0 <= v < N_ACTIONS:
                n_inv += 1
                v = STAY
            acts.append(v)

        self.stats["carrying_per_step"].append(sum(it is not None for it in self.carry))
        for i, it in enumerate(self.carry):
            if it is not None:
                self.stats["steps_returning"][i] += 1
        if targets:
            self._apply_targets(targets)

        pickers = {}
        for i, a in enumerate(acts):
            if a != PICK:
                continue
            if self._can_pick(i):
                pickers.setdefault(self.pos[i], []).append(i)
            else:
                n_inv += 1
        for pos, idxs in pickers.items():
            winner = idxs[0] if len(idxs) == 1 else self._order(idxs, c.tiebreak)[0]
            self.stats["pickup_conflicts"] += len(idxs) - 1
            comp["delivery"] += self._do_pickup(winner, self.active.pop(pos))

        droppers = []
        for i, a in enumerate(acts):
            if a != DROP:
                continue
            if self._can_drop(i):
                droppers.append(i)
            else:
                n_inv += 1
        served = self._select_served(droppers)
        for i in droppers:
            if i not in served:
                self.stats["service_denied"] += 1
                self.stats["steps_waiting"][i] += 1
        for i in served:
            comp["delivery"] += self._do_delivery(i)

        n_col = self._move(acts)
        comp["collision"] = -c.c_collision * n_col
        comp["invalid"] = -c.c_invalid * n_inv
        self.stats["collisions"] += n_col

        self.t += 1
        if not c.temp_expires_while_carried:
            for it in self.carry:
                if it is not None and it.type == TEMP:
                    it.expiry += 1
        for p in [p for p, it in self.active.items() if it.expiry <= self.t]:
            it = self.active.pop(p)
            self._clear_claim(it)
            self._expire(it)
        for i, it in enumerate(self.carry):
            if it is not None and it.expiry <= self.t:
                self.stats["expired_while_carried"] += 1
                self._expire(it)
                self.carry[i] = None

        if self._count_type(TEMP) < c.max_temp_active and self.rng.random() < c.temp_spawn_prob:
            if self._spawn_item(TEMP):
                self.stats["temp_spawned"] += 1
        if c.persist_spawn_prob > 0 and self._count_type(PERSIST) < c.max_persist_active:
            if self.rng.random() < c.persist_spawn_prob:
                if self._spawn_item(PERSIST):
                    self.stats["persist_spawned"] += 1
        if c.mal_spawn_prob > 0 and self._count_type(MAL) < c.max_mal_active:
            if self.rng.random() < c.mal_spawn_prob:
                if self._spawn_item(MAL):
                    self.stats["mal_spawned"] += 1

        new_frac, new_items = self._update_visibility(count_personal=True)
        if c.info_mode == "window":
            win_mask = self._union_view()
            for i, tgt_id in enumerate(self.target):
                if tgt_id is not None:
                    it = self.objects[tgt_id]
                    if not win_mask[it.pos]:
                        self._release(i)
        comp["map"] = c.alpha_map * new_frac
        for it in new_items:
            comp["discover"] += c.b_discover[it.type]
            self.stats["discovered_by_type"][it.type] += 1
            if it.discovered_by is not None:
                self.stats["discovered_by_agent"][it.discovered_by] += 1

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
                infos[a]["episode_stats"] = copy.deepcopy(self.stats)
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def _can_pick(self, i):
        """Check if agent i can pick an item at its current position."""
        if self.carry[i] is not None:
            return False
        it = self.active.get(self.pos[i])
        if it is None:
            return False
        if it.type == MAL and self.cfg.forbid_malicious_pickup:
            return False
        if self.cfg.claim_blocks_pickup and it.claimed_by not in (None, i):
            return False
        return True

    def _can_drop(self, i):
        """Check if agent i can drop an item at checkpoint."""
        return self.carry[i] is not None and self.pos[i] == self.checkpoint

    def _do_pickup(self, i, it: Item) -> float:
        """Handle item pickup by agent i."""
        c = self.cfg
        self._clear_claim(it)
        it.collected_by, it.pickup_t = i, self.t
        self.stats["pickups"] += 1
        self.stats["pickup_by_type"][it.type] += 1
        pen = 0.0
        if it.type == MAL:
            self.stats["malicious_picked"] += 1
            pen = -c.p_mal
            if c.mal_on_pickup == "destroy":
                it.status, it.end_t = Status.DISPOSED, self.t
                self.stats["disposed"] += 1
                self._release(i)
                return pen
        self._release(i)
        it.status = Status.CARRIED
        self.carry[i] = it
        return pen

    def _select_served(self, idxs):
        """Select dropping agents served up to checkpoint service rate."""
        k = self.cfg.checkpoint_service_rate
        if k is None or len(idxs) <= k:
            return list(idxs)
        mode = self.cfg.service_priority
        if mode == "random":
            order = self._order(idxs, "random")
        elif mode == "index":
            order = sorted(idxs)
        else:
            order = sorted(idxs, key=lambda i: (self.carry[i].expiry - self.t if self.carry[i].type == TEMP else float("inf"), i))
        return order[:k]

    def _do_delivery(self, i) -> float:
        """Process item delivery at checkpoint by agent i."""
        it = self.carry[i]
        assert it is not None and it.status == Status.CARRIED, "double delivery"
        self.carry[i] = None
        it.end_t = self.t
        if it.type == MAL:
            it.status = Status.DISPOSED
            self.stats["disposed"] += 1
            return 0.0
        it.status = Status.DELIVERED
        self.stats["deliveries"] += 1
        self.stats["delivered_value"] += it.value
        self.stats["delivered_by_type"][it.type] += 1
        self.stats["delivery_log"].append({
            "object_id": it.id, "type": it.type, "agent": i, "value": it.value,
            "discovered_t": it.discovered_t, "pickup_t": it.pickup_t, "delivery_t": self.t,
            "pickup_to_delivery": self.t - it.pickup_t,
            "life_left": (it.expiry - self.t) if it.type == TEMP else None,
        })
        return it.value

    def _expire(self, it: Item):
        """Mark item expired and update statistics."""
        it.status, it.end_t = Status.EXPIRED, self.t
        self.stats["expired_by_type"][it.type] += 1
        if it.seen:
            self.stats["expired_after_discovery"] += 1

    def _in_free(self, y, x):
        """Check if (y, x) is an in-bounds free grid cell."""
        return 0 <= y < self.cfg.height and 0 <= x < self.cfg.width and not self.grid[y, x]

    def _move(self, acts):
        """Simulate agent movements and resolve collisions."""
        n = self.cfg.n_agents
        stack = self.cfg.checkpoint_stacking
        cp = self.checkpoint
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
                    col[i] = True
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if des[i] == cur[i]:
                    continue
                for j in range(n):
                    if j == i:
                        continue
                    same_target = des[j] == des[i] and not (stack and des[i] == cp)
                    swap = des[i] == cur[j] and des[j] == cur[i]
                    if same_target or swap:
                        des[i] = cur[i]
                        col[i] = True
                        changed = True
                        break
        for i in range(n):
            if des[i] != cur[i]:
                self.stats["distance"][i] += 1
        self.pos = des
        return int(sum(col))

    def _coordination(self):
        """Calculate crowding penalty around visible valuable items."""
        c = self.cfg
        pen = 0.0
        free_agents = [self.pos[i] for i in range(c.n_agents) if self.carry[i] is None]
        for p, it in self.active.items():
            if not it.seen or it.type == MAL:
                continue
            k = sum(1 for q in free_agents if max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= c.crowd_radius)
            if k >= 2:
                pen -= c.coord_penalty * (k - 1)
        return pen

    def action_mask(self, agent):
        """Return legal action boolean mask for agent."""
        i = self._idx[agent]
        m = np.zeros(N_ACTIONS, dtype=bool)
        y, x = self.pos[i]
        for a, (dy, dx) in MOVES.items():
            m[a] = self._in_free(y + dy, x + dx)
        m[STAY] = True
        m[PICK] = self._can_pick(i)
        m[DROP] = self._can_drop(i)
        return m

    def _mode(self, i) -> Mode:
        """Return agent task Mode (RETURN, PURSUE, or EXPLORE)."""
        if self.carry[i] is not None:
            return Mode.RETURN
        if self.target[i] is not None:
            return Mode.PURSUE
        return Mode.EXPLORE

    def agent_state(self, agent):
        """Return dictionary summary of agent state."""
        i = self._idx[agent]
        y, x = self.pos[i]
        carried, mode = self.carry[i], self._mode(i)
        if mode == Mode.RETURN:
            kind, tpos = "checkpoint", self.checkpoint
        elif mode == Mode.PURSUE:
            kind, tpos = "item", self.objects[self.target[i]].pos
        else:
            kind, tpos = None, None
        return {
            "position_xy": (x, y),
            "carrying": carried is not None,
            "carried_object_id": None if carried is None else carried.id,
            "carried_object_type": None if carried is None else TYPE_NAMES[carried.type],
            "target_kind": kind,
            "target_object_id": self.target[i],
            "target_xy": None if tpos is None else (tpos[1], tpos[0]),
            "task_mode": mode.name,
        }

    def agent_states(self):
        """Return dictionary of agent state summaries for all agents."""
        return {a: self.agent_state(a) for a in self.possible_agents}

    def _remaining_frac(self, it):
        """Return remaining life fraction of item."""
        return float(np.clip((it.expiry - self.t) / self._life_max, 0.0, 1.0))

    def _item_channels(self, seen_only):
        """Return 3-channel spatial map of items."""
        ch = np.zeros((3, self.cfg.height, self.cfg.width), dtype=np.float32)
        for (y, x), it in self.active.items():
            if seen_only and not it.seen:
                continue
            ch[it.type, y, x] = self._remaining_frac(it) if it.type == TEMP else 1.0
        return ch

    def _agent_map(self):
        """Return 2D occupancy grid of agent locations."""
        m = np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32)
        for (y, x) in self.pos:
            m[y, x] = 1.0
        return m

    def _is_item_visible_to(self, i: int, it: Item) -> bool:
        """Check whether item is known or visible to agent i under info_mode."""
        mode = self.cfg.info_mode
        if mode == "shared":
            return it.seen
        if mode == "private":
            return it.id in self._seen_by[i]
        r = self.cfg.view_radius
        py, px = it.pos
        return any(max(abs(py - y), abs(px - x)) <= r for y, x in self.pos)

    def _belief_dist(self):
        """Compute team belief distance matrix to checkpoint."""
        if self.cfg.slack_uses_true_path:
            return self.checkpoint_dist
        return self._bfs(self.known & self.grid, self.checkpoint)

    def _belief_dist_for(self, agent_known):
        """Compute belief distance matrix to checkpoint using specified known grid."""
        if self.cfg.slack_uses_true_path:
            return self.checkpoint_dist
        return self._bfs(agent_known & self.grid, self.checkpoint)

    def _plen(self, bdist, p):
        """Return distance to checkpoint or grid dimension upper bound."""
        v = int(bdist[p])
        return v if v >= 0 else self.cfg.height + self.cfg.width

    def _union_view(self):
        """Compute collective sensing window union of all agents."""
        c = self.cfg
        r = c.view_radius
        k = np.zeros((c.height, c.width), dtype=bool)
        for (y, x) in self.pos:
            k[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        return k

    def _effective_known(self, i):
        """Return accumulated known grid for agent i."""
        return self._agent_known[i]

    def _visible_item_ids(self, i):
        """Return set of item IDs observed by agent i."""
        return self._seen_by[i]

    def _observations(self):
        """Assemble observation dictionaries for all agents."""
        c = self.cfg
        r, H, W = c.view_radius, c.height, c.width
        w = 2 * r + 1
        mode = c.info_mode

        items_all = self._item_channels(seen_only=False)
        agents_all = self._agent_map()
        cp = np.zeros((H, W), dtype=np.float32)
        cp[self.checkpoint] = 1.0
        world = np.concatenate([self.grid[None].astype(np.float32), items_all, cp[None], agents_all[None]], axis=0)
        padded = np.pad(world, ((0, 0), (r, r), (r, r)))
        if r > 0:
            padded[0, :r, :] = padded[0, -r:, :] = 1.0
            padded[0, :, :r] = padded[0, :, -r:] = 1.0

        if mode in ("shared", "window"):
            if mode == "shared":
                kn = self.known
                vis_items = [it for it in self.active.values() if it.seen]
            else:
                kn = self._union_view()
                vis_items = [it for p, it in self.active.items() if kn[p]]
            titems_s = np.zeros((3, H, W), dtype=np.float32)
            claimed_s = np.zeros((H, W), dtype=np.float32)
            for it in vis_items:
                titems_s[it.type, it.pos[0], it.pos[1]] = self._remaining_frac(it) if it.type == TEMP else 1.0
                if it.status == Status.CLAIMED:
                    claimed_s[it.pos] = 1.0
            team_s = np.concatenate([
                kn[None].astype(np.float32),
                (kn & self.grid)[None].astype(np.float32),
                titems_s, agents_all[None], claimed_s[None]
            ], axis=0)
            kval_s = [it for it in vis_items if it.type != MAL]
            bdist_s = self._belief_dist_for(kn)

        obs = {}
        for i, a in enumerate(self.possible_agents):
            y, x = self.pos[i]
            local = padded[:, y: y + w, x: x + w].copy()
            local[5, r, r] = 0.0

            if mode in ("shared", "window"):
                team = team_s.copy()
                known_val = kval_s
                bdist = bdist_s
            else:
                ak = self._effective_known(i)
                vis = self._visible_item_ids(i)
                titems_i = np.zeros((3, H, W), dtype=np.float32)
                claimed_i = np.zeros((H, W), dtype=np.float32)
                known_val = []
                for p, it in self.active.items():
                    if it.id not in vis:
                        continue
                    titems_i[it.type, p[0], p[1]] = self._remaining_frac(it) if it.type == TEMP else 1.0
                    if it.status == Status.CLAIMED:
                        claimed_i[p] = 1.0
                    if it.type != MAL:
                        known_val.append(it)

                agents_vis = np.zeros((H, W), dtype=np.float32)
                for j, (jy, jx) in enumerate(self.pos):
                    if max(abs(jy - y), abs(jx - x)) <= r:
                        agents_vis[jy, jx] = 1.0

                team = np.concatenate([
                    ak[None].astype(np.float32),
                    (ak & self.grid)[None].astype(np.float32),
                    titems_i, agents_vis[None], claimed_i[None]
                ], axis=0)
                bdist = self._belief_dist_for(ak)

            obs[a] = {"local": local, "team_map": team, "scalars": self._scalars(i, known_val, bdist)}
        return obs

    def _slack(self, i, it, bdist):
        """Compute deadline slack for temporary item pickup and delivery."""
        d_item = abs(self.pos[i][0] - it.pos[0]) + abs(self.pos[i][1] - it.pos[1])
        return it.expiry - self.t - (d_item + self._plen(bdist, it.pos) + HANDLING_TIME)

    def _scalars(self, i, known_val, bdist):
        """Construct scalar observation vector for agent i."""
        c = self.cfg
        y, x = self.pos[i]
        carried = self.carry[i]
        cy, cx = self.checkpoint
        s = np.zeros(N_SCALARS, dtype=np.float32)
        s[0], s[1] = y / c.height, x / c.width
        s[2] = float(carried is not None)
        s[3] = self._remaining_frac(carried) if (carried and carried.type == TEMP) else 0.0
        s[4] = self.t / c.max_steps
        s[5] = min(self.stats["deliveries"] / (c.target_deliveries or 20), 1.0)
        s[6], s[7] = (cy - y) / c.height, (cx - x) / c.width
        to_cp = self._plen(bdist, (y, x))
        slack = 1.0
        if carried is not None:
            if carried.type == TEMP:
                slack = (carried.expiry - self.t - to_cp - 1) / self._life_max
        else:
            cands = [it for it in known_val if it.claimed_by in (None, i)]
            if cands:
                near = min(cands, key=lambda it: abs(it.pos[0] - y) + abs(it.pos[1] - x))
                s[8] = (near.pos[0] - y) / c.height
                s[9] = (near.pos[1] - x) / c.width
                temps = [it for it in cands if it.type == TEMP]
                if temps:
                    slack = min(self._slack(i, it, bdist) for it in temps) / self._life_max
        s[10] = slack
        s[11] = min(len(known_val) / 10.0, 1.0)
        s[12] = float(carried is not None and carried.type == TEMP)
        s[13] = float(carried is not None and carried.type == MAL)
        if self.target[i] is not None:
            tp = self.objects[self.target[i]].pos
            s[14] = 1.0
            s[15], s[16] = (tp[0] - y) / c.height, (tp[1] - x) / c.width
        s[17 + int(self._mode(i))] = 1.0
        s[20] = sum(it is not None for it in self.carry) / c.n_agents
        s[21] = (self.t - carried.pickup_t) / self._life_max if carried is not None else 0.0
        s[22] = to_cp / (c.height + c.width)
        return np.clip(s, -1.0, 1.0)

    def _infos(self, comp):
        """Build info dictionary for current step."""
        infos = {}
        n_carry = sum(it is not None for it in self.carry)
        for a in self.agents:
            infos[a] = {
                "action_mask": self.action_mask(a),
                "reward_components": dict(comp),
                "deliveries": self.stats["deliveries"],
                "n_carrying": n_carry,
                "agent_state": self.agent_state(a),
            }
        return infos

    def state(self):
        """Construct centralized critic global state vector."""
        c = self.cfg
        items = self._item_channels(seen_only=False)
        cp = np.zeros((c.height, c.width), dtype=np.float32)
        cp[self.checkpoint] = 1.0
        agent_feats = []
        for i in range(c.n_agents):
            y, x = self.pos[i]
            carried = self.carry[i]
            agent_feats += [
                y / c.height, x / c.width, float(carried is not None),
                self._remaining_frac(carried) if (carried and carried.type == TEMP) else 0.0
            ]
        return np.concatenate([
            self.grid.astype(np.float32).ravel(), cp.ravel(), items.ravel(),
            self.known.astype(np.float32).ravel(), self._agent_map().ravel(),
            np.array(agent_feats + [self.t / c.max_steps, min(self.stats["deliveries"] / (c.target_deliveries or 20), 1.0)], dtype=np.float32)
        ]).astype(np.float32)

    def check_invariants(self):
        """Assert environment and item state consistency."""
        on_world = set()
        for p, it in self.active.items():
            assert it.status in (Status.AVAILABLE, Status.CLAIMED) and it.pos == p
            assert (it.status == Status.CLAIMED) == (it.claimed_by is not None)
            assert it.id not in on_world
            on_world.add(it.id)
        for i, it in enumerate(self.carry):
            if it is not None:
                assert it.status == Status.CARRIED and it.collected_by == i and it.claimed_by is None
                assert it.id not in on_world, "object both on the ground and carried"
                on_world.add(it.id)
        live = (Status.AVAILABLE, Status.CLAIMED, Status.CARRIED)
        for it in self.objects.values():
            assert (it.id in on_world) == (it.status in live), f"object {it.id} {it.status.name} misplaced"
        assert self.stats["deliveries"] == sum(it.status == Status.DELIVERED for it in self.objects.values())
        assert len({d["object_id"] for d in self.stats["delivery_log"]}) == self.stats["deliveries"]
        for i, t in enumerate(self.target):
            if t is not None:
                it = self.objects[t]
                assert it.status == Status.CLAIMED and it.claimed_by == i and self.carry[i] is None
        cells = [p for p in self.pos if not (self.cfg.checkpoint_stacking and p == self.checkpoint)]
        assert len(cells) == len(set(cells)), "two agents share a non-checkpoint cell"
        assert all(not self.grid[p] for p in self.pos), "agent inside an obstacle"

    def render(self, cell_px: int = 24, view: str = "both", agent=0):
        """Render environment frame or ANSI text."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self.render_view(view, cell_px, agent)
        return None

    def close(self):
        """Clean up environment resources."""
        pass

    def _render_ansi(self):
        """Render ANSI text string of grid and agents."""
        c = self.cfg
        rows = [["#" if self.grid[y, x] else "." for x in range(c.width)] for y in range(c.height)]
        sym = {PERSIST: "p", TEMP: "T", MAL: "m"}
        for (y, x), it in self.active.items():
            rows[y][x] = sym[it.type]
        rows[self.checkpoint[0]][self.checkpoint[1]] = "C"
        for i, (y, x) in enumerate(self.pos):
            rows[y][x] = str(i) if self.carry[i] is None else chr(ord("a") + i)
        return "\n".join("".join(r) for r in rows)

    @staticmethod
    def _kron(base, px):
        """Scale pixel grid by px using Kronecker product."""
        return (np.kron(base, np.ones((px, px, 1))) * 255).astype(np.uint8)

    @staticmethod
    def _paint(img, y, x, px, color, inset=0):
        """Fill cell rectangle with color."""
        img[y * px + inset:(y + 1) * px - inset, x * px + inset:(x + 1) * px - inset] = (np.asarray(color) * 255).astype(np.uint8)

    @staticmethod
    def _frame(img, y0, x0, y1, x1, px, color, t=2):
        """Draw outline frame around bounding coordinates."""
        col = (np.asarray(color) * 255).astype(np.uint8)
        Y0, Y1, X0, X1 = y0 * px, (y1 + 1) * px, x0 * px, (x1 + 1) * px
        img[Y0:Y0 + t, X0:X1] = col
        img[Y1 - t:Y1, X0:X1] = col
        img[Y0:Y1, X0:X0 + t] = col
        img[Y0:Y1, X1 - t:X1] = col

    @staticmethod
    def _hcat(imgs, gap=10):
        """Concatenate image arrays horizontally."""
        h = max(im.shape[0] for im in imgs)
        out = []
        for k, im in enumerate(imgs):
            pad = np.full((h, im.shape[1], 3), 255, np.uint8)
            pad[:im.shape[0]] = im
            out.append(pad)
            if k < len(imgs) - 1:
                out.append(np.full((h, gap, 3), 255, np.uint8))
        return np.concatenate(out, axis=1)

    def render_view(self, view="both", px=24, agent=0, fog=True):
        """Render specified view mode."""
        i = self._idx[agent] if isinstance(agent, str) else int(agent)
        if view == "true":
            return self._render_true(px, fog=fog)
        obs = self._observations()[self.possible_agents[i]]
        if view == "agent":
            return self.render_observation(obs, px)
        return self._hcat([self._render_true(px, fog=fog), self.render_observation(obs, px)])

    def _render_true(self, px=24, fog=True):
        """Render ground-truth view with optional fog of war."""
        c = self.cfg
        H, W, r = c.height, c.width, c.view_radius
        win_mask = self._union_view() if (fog and c.info_mode == "window") else None
        if fog:
            vis_mask = win_mask if win_mask is not None else self.known
            base = np.full((H, W, 3), _C_UNKNOWN, dtype=np.float32)
            base[vis_mask & ~self.grid] = 1.0
            base[vis_mask & self.grid] = _C_OBST
            base[self.checkpoint] = _C_CP
        else:
            base = np.ones((H, W, 3), np.float32)
            base[self.grid] = _C_OBST
            base[self.checkpoint] = _C_CP
        img = self._kron(base, px)
        for i, (y, x) in enumerate(self.pos):
            self._frame(img, max(0, y - r), max(0, x - r), min(H - 1, y + r), min(W - 1, x + r), px, _AGENT_COLS[i % len(_AGENT_COLS)], 2)
        for (y, x), it in self.active.items():
            is_vis = (not fog) or (bool(win_mask[y, x]) if win_mask is not None else it.seen)
            if is_vis:
                self._paint(img, y, x, px, _ITEM_COL[it.type], 3)
                if it.status == Status.CLAIMED:
                    self._frame(img, y, x, y, x, px, _C_CLAIM, 1)
        for i, (y, x) in enumerate(self.pos):
            self._paint(img, y, x, px, _AGENT_COLS[i % len(_AGENT_COLS)], px // 6)
            if self.carry[i] is not None:
                self._paint(img, y, x, px, _ITEM_COL[self.carry[i].type], px // 3)
        return img

    def render_true_nofog(self, px=24):
        """Render full map without fog of war."""
        return self._render_true(px, fog=False)

    def _render_rgb(self, px=24, fog=True):
        """Render true map for backwards compatibility."""
        return self._render_true(px, fog=fog)

    def render_observation(self, obs, px=24):
        """Render agent observation map and local field of view."""
        c = self.cfg
        H, W, r = c.height, c.width, c.view_radius
        team, local, sc = obs["team_map"], obs["local"], obs["scalars"]
        ay, ax = int(round(float(sc[0]) * H)), int(round(float(sc[1]) * W))
        cy, cx = ay + int(round(float(sc[6]) * H)), ax + int(round(float(sc[7]) * W))
        base = np.empty((H, W, 3), np.float32)
        base[:] = _C_UNKNOWN
        base[team[0] > 0.5] = 1.0
        base[team[1] > 0.5] = _C_OBST
        base[cy, cx] = _C_CP
        img = self._kron(base, px)
        for typ in (PERSIST, TEMP, MAL):
            for y, x in np.argwhere(team[2 + typ] > 0):
                self._paint(img, y, x, px, _ITEM_COL[typ], 3)
        for y, x in np.argwhere(team[6] > 0.5):
            self._frame(img, y, x, y, x, px, _C_CLAIM, 1)
        for y, x in np.argwhere(team[5] > 0.5):
            self._paint(img, y, x, px, _C_AGENT_OTHER, px // 6)
        self._paint(img, ay, ax, px, _C_SELF, px // 6)
        self._frame(img, max(0, ay - r), max(0, ax - r), min(H - 1, ay + r), min(W - 1, ax + r), px, _C_SELF, 2)

        w = 2 * r + 1
        lp = np.ones((w, w, 3), np.float32)
        lp[local[0] > 0.5] = _C_OBST
        lp[local[4] > 0.5] = _C_CP
        panel = self._kron(lp, px)
        for typ in (PERSIST, TEMP, MAL):
            for y, x in np.argwhere(local[1 + typ] > 0):
                self._paint(panel, y, x, px, _ITEM_COL[typ], 3)
        for y, x in np.argwhere(local[5] > 0.5):
            self._paint(panel, y, x, px, _C_AGENT_OTHER, px // 6)
        self._paint(panel, r, r, px, _C_SELF, px // 6)
        return self._hcat([img, panel])