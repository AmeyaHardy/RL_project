"""Hierarchical interface: high-level goal selection with low-level deterministic BFS executor and yielding."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import DOWN, DROP, LEFT, PICK, RIGHT, STAY, UP, EnvConfig, HANDLING_TIME

MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
KI, KR = 20, 20                  # max item / region candidates
FI, FR, FC = 9, 9, 14            # feature sizes (item, region, context)
N_CAND = 1 + KI + KR
AGENT_BLOCK = 7
STAY_TICKS = 4                   # an idle robot re-decides after this many steps
REGION_TTL = 12                  # region goals are interruptible after this many steps
AGE_CAP = 50.0


def critic_dim(n_agents: int) -> int:
    return FC + AGENT_BLOCK * n_agents + 4


class HPlanner:
    def __init__(self, cfg: EnvConfig, seed: int = 0, memory: bool = True):
        assert cfg.info_mode in ("shared", "window"), "HPlanner supports info_mode 'shared' or 'window'"
        self.memory = memory
        self.cfg, self.N = cfg, cfg.n_agents
        self.H, self.W = cfg.height, cfg.width
        self.D = float(self.H + self.W)
        self.life_max = float(cfg.temp_life_range[1])
        self.by, self.bx = -(-self.H // 4), -(-self.W // 4)       # 4x4 region grid for every stage
        self.rng = np.random.default_rng(seed)
        self.names: List[str] = []
        self.reset(seed)

    # ------------------------------------------------------------------ state
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        N = self.N
        self.t, self._t_seen = 0, -1
        self.last_seen = np.full((self.H, self.W), -1e9, np.float32)
        self.known = np.zeros((self.H, self.W), bool)          # policy-side map memory (window mode)
        self.obst = np.zeros((self.H, self.W), bool)
        self.item_mem: Dict[tuple, Tuple[int, int]] = {}       # pos -> (type, absolute expiry step)
        self.goal: List[Optional[tuple]] = [None] * N          # ("item", pos) | ("region", cell) | ("stay", until)
        self.goal_t = [0] * N
        self.claims: Dict[tuple, int] = {}
        self.ban: List[Dict[tuple, int]] = [dict() for _ in range(N)]
        self.last_act = [STAY] * N
        self.last_pos: List[Optional[tuple]] = [None] * N
        self.blocked = [0] * N
        self.stall = [0] * N                                   # consecutive steps a carrier made no progress
        self.retreat = [-1] * N                                # carrier backs out of a jam until this step
        self.prev_items: set = set()
        self.new_item = False
        self._cache: dict = {}

    # ------------------------------------------------------------------ perception
    def perceive(self, obs: Dict[str, dict]):
        c, H, W = self.cfg, self.H, self.W
        self.names = list(obs.keys())
        sc = np.stack([obs[a]["scalars"] for a in self.names])
        team = obs[self.names[0]]["team_map"]
        t = int(round(float(sc[0, 4]) * c.max_steps))
        if t < self._t_seen:
            self.reset()
        self._t_seen, self.t = t, t
        self._cache = {}
        self.pos = [(int(np.clip(round(float(s[0]) * H), 0, H - 1)), int(np.clip(round(float(s[1]) * W), 0, W - 1)))
                    for s in sc]
        self.carrying = [bool(s[2] > 0.5) for s in sc]
        self.carry_rem = [float(s[3]) for s in sc]
        self.carry_val = [(c.v_high if s[12] > 0.5 else c.v_low) if s[2] > 0.5 else 0.0 for s in sc]
        y0, x0 = self.pos[0]
        self.depot = (int(np.clip(round(y0 + float(sc[0, 6]) * H), 0, H - 1)),
                      int(np.clip(round(x0 + float(sc[0, 7]) * W), 0, W - 1)))
        self.deliv_frac = float(sc[0, 5])
        vis, vis_obst = team[0] > 0.5, team[1] > 0.5
        window = c.info_mode == "window"
        # items visible in this observation (malicious are ignored: never picked); temp expiry is absolute
        seen: Dict[tuple, Tuple[int, int]] = {}
        for y, x in np.argwhere(team[2] > 0.5):
            seen[(int(y), int(x))] = (0, 10 ** 9)
        for y, x in np.argwhere(team[3] > 0):
            seen[(int(y), int(x))] = (1, int(round(t + float(team[3][y, x]) * self.life_max)))
        if window and self.memory:
            self.known |= vis
            self.obst |= vis_obst
            for p in [p for p in self.item_mem if vis[p] and p not in seen]:
                del self.item_mem[p]                      # cell is in view and empty: picked or expired
            self.item_mem.update(seen)
            for p in [p for p, (ty, ex) in self.item_mem.items() if ty == 1 and ex <= t]:
                del self.item_mem[p]                      # remembered temp item reached its deadline
        else:                                              # shared mode (env accumulates) or memoryless ablation
            self.known, self.obst, self.item_mem = vis, vis_obst, seen
        self.free = self.known & ~self.obst
        self.free[self.depot] = True
        self.items: Dict[tuple, Tuple[int, float]] = {
            p: (ty, 1e9 if ty == 0 else float(ex - t)) for p, (ty, ex) in self.item_mem.items()}
        r = c.view_radius
        if window and not self.memory:
            self.last_seen[:] = -1e9
        for (y, x) in self.pos:
            self.last_seen[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = t
        cur = set(self.items)
        self.new_item = bool(cur - self.prev_items)
        self.prev_items = cur
        for p in list(self.claims):                    # drop claims on items that are gone
            if p not in self.items:
                del self.claims[p]
        for i in range(self.N):
            lp, la = self.last_pos[i], self.last_act[i]
            self.blocked[i] = self.blocked[i] + 1 if (la in MOVES and lp == self.pos[i]) else 0
            self.stall[i] = self.stall[i] + 1 if (self.carrying[i] and lp == self.pos[i] and self.pos[i] != self.depot) else 0
            self.last_pos[i] = self.pos[i]
            if self.carrying[i]:
                self._release(i)

    # ------------------------------------------------------------------ graph helpers
    def _bfs(self, src: tuple, extra_block=None, stop: Optional[tuple] = None):
        """Distance dict-free BFS over known free cells. Returns (dist [H,W] int, first_step dict or None)."""
        H, W, free = self.H, self.W, self.free
        dist = np.full((H, W), -1, np.int32)
        if not free[src]:
            return dist, {}
        dist[src] = 0
        q = deque([src])
        first: Dict[tuple, int] = {}
        while q:
            y, x = q.popleft()
            if stop is not None and (y, x) == stop:
                break
            d = dist[y, x] + 1
            for a, (dy, dx) in MOVES.items():
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and dist[ny, nx] < 0:
                    if extra_block is not None and (ny, nx) in extra_block:
                        continue
                    dist[ny, nx] = d
                    first[(ny, nx)] = a if (y, x) == src else first[(y, x)]
                    q.append((ny, nx))
        return dist, first

    def dist_from(self, src: tuple) -> np.ndarray:
        key = ("d", src)
        if key not in self._cache:
            self._cache[key] = self._bfs(src)[0]
        return self._cache[key]

    # ------------------------------------------------------------------ goals / claims
    def _release(self, i: int):
        g = self.goal[i]
        if g is not None and g[0] == "item" and self.claims.get(g[1]) == i:
            del self.claims[g[1]]
        self.goal[i] = None

    def _feasible(self, i: int, p: tuple) -> bool:
        typ, rem = self.items[p]
        dd, dc = self.dist_from(self.pos[i])[p], self.dist_from(self.depot)[p]
        if dd < 0 or dc < 0:
            return False
        need = dd + dc + HANDLING_TIME
        return need <= self.cfg.max_steps - self.t and (typ == 0 or need <= rem)

    def need_decision(self) -> List[int]:
        """Agents that must choose a new goal this step, in random order (random order breaks symmetry)."""
        t, out = self.t, []
        for i in range(self.N):
            if self.carrying[i]:
                continue
            g = self.goal[i]
            need = False
            if g is None:
                need = True
            elif g[0] == "stay":
                need = t >= g[1] or self.new_item
            elif g[0] == "item":
                p = g[1]
                if p not in self.items or self.claims.get(p) != i or not self._feasible(i, p):
                    self._release(i); need = True
            else:
                need = (self.pos[i] == g[1] or t - self.goal_t[i] >= REGION_TTL or self.new_item)
            if self.blocked[i] >= 3 and g is not None and g[0] != "stay":
                self.ban[i][g[:2]] = t + 10
                self._release(i); self.blocked[i] = 0; need = True
            if need:
                self._release(i)
                out.append(i)
        return [int(v) for v in self.rng.permutation(out)] if len(out) > 1 else out

    def apply_choice(self, i: int, k: int, cand: dict):
        self._release(i)
        self.goal_t[i] = self.t
        if k == 0:
            self.goal[i] = ("stay", self.t + STAY_TICKS)
        elif k <= KI:
            p = cand["item_pos"][k - 1]
            self.goal[i] = ("item", p)
            self.claims[p] = i
        else:
            self.goal[i] = ("region", cand["reg_cell"][k - 1 - KI])

    # ------------------------------------------------------------------ candidates (features for the learned chooser)
    def _stale_gain(self) -> np.ndarray:
        if "G" not in self._cache:
            r = self.cfg.view_radius
            age = np.clip(self.t - self.last_seen, 0.0, AGE_CAP) / AGE_CAP
            self._cache["age"] = age
            pad = np.pad(age, ((r + 1, r), (r + 1, r)))
            ii = pad.cumsum(0).cumsum(1)
            k = 2 * r + 1
            box = ii[k:k + self.H, k:k + self.W] - ii[:self.H, k:k + self.W] - ii[k:k + self.H, :self.W] + ii[:self.H, :self.W]
            self._cache["G"] = box / float(k * k)
        return self._cache["G"]

    def _free_mates(self, i: int):
        return [j for j in range(self.N) if j != i and not self.carrying[j]]

    def ctx(self, i: int) -> np.ndarray:
        c, N = self.cfg, self.N
        self._stale_gain()
        age = self._cache["age"]
        unclaimed = [p for p in self.items if p not in self.claims]
        n_t = sum(1 for p in unclaimed if self.items[p][0] == 1)
        free_pos = [self.pos[j] for j in range(N) if not self.carrying[j]]
        spread = 0.0
        if len(free_pos) >= 2:
            spread = float(np.mean([max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                                    for ii, a in enumerate(free_pos) for b in free_pos[ii + 1:]])) / self.D
        dd = self.dist_from(self.depot)[self.pos[i]]
        f = np.zeros(FC, np.float32)
        f[:] = [self.t / c.max_steps, self.deliv_frac, float(self.known.mean()), float(age.mean()),
                len(free_pos) / N, sum(self.carrying) / N, min(len(unclaimed) / 10.0, 1.0), min(n_t / 5.0, 1.0),
                (dd if dd >= 0 else self.D) / self.D, spread, self.pos[i][0] / self.H, self.pos[i][1] / self.W,
                min(sum(1 for v in self.items.values() if v[0] == 1) / 5.0, 1.0), N / 4.0]
        return f

    def candidates(self, i: int) -> dict:
        c, N, D = self.cfg, self.N, self.D
        t, T_left = self.t, c.max_steps - self.t
        d_me = self.dist_from(self.pos[i])
        d_dep = self.dist_from(self.depot)
        mates = self._free_mates(i)
        mate_d = [self.dist_from(self.pos[j]) for j in mates]
        banned = {k for k, until in self.ban[i].items() if until > t}

        # ---- items
        rows = []
        for p, (typ, rem) in self.items.items():
            if self.claims.get(p, i) != i or ("item", p) in banned:
                continue
            dd, dc = int(d_me[p]), int(d_dep[p])
            if dd < 0 or dc < 0:
                continue
            need = dd + dc + HANDLING_TIME
            if need > T_left or (typ == 1 and need > rem):
                continue
            val = c.v_high if typ == 1 else c.v_low
            rows.append((val / need, p, typ, rem, dd, dc, val, need))
        rows.sort(key=lambda r: -r[0])
        items = np.zeros((KI, FI), np.float32)
        imask = np.zeros(KI, bool)
        ipos: List[tuple] = []
        for k, (dens, p, typ, rem, dd, dc, val, need) in enumerate(rows[:KI]):
            n_closer = sum(1 for md in mate_d if 0 <= md[p] < dd)
            slack = ((min(rem, T_left) if typ == 1 else T_left) - need) / (self.life_max if typ == 1 else c.max_steps)
            items[k] = [val / c.v_high, float(typ), min(rem, self.life_max) / self.life_max if typ == 1 else 1.0,
                        np.clip(slack, -1, 1), dd / D, dc / D, min(dens / c.v_high * 4.0, 1.0),
                        n_closer / max(N - 1, 1), float(self.goal[i] == ("item", p))]
            imask[k] = True
            ipos.append(p)

        # ---- regions
        G, age = self._stale_gain(), self._cache["age"]
        score = np.where((d_me >= 0), G - 0.003 * d_me, -1e9)
        regs = np.zeros((KR, FR), np.float32)
        rmask = np.zeros(KR, bool)
        rcell: List[tuple] = []
        mate_pts = []
        for j in mates:
            g = self.goal[j]
            mate_pts.append(g[1] if (g is not None and g[0] == "region") else self.pos[j])
        mate_cur = [self.pos[j] for j in mates]
        k = 0
        for ry in range(4):
            for rx in range(4):
                if k >= KR:
                    break
                ys, xs = slice(ry * self.by, min((ry + 1) * self.by, self.H)), slice(rx * self.bx, min((rx + 1) * self.bx, self.W))
                sub = score[ys, xs]
                if sub.size == 0 or sub.max() < -1e8:
                    k += 1
                    continue
                a = np.unravel_index(int(sub.argmax()), sub.shape)
                cell = (ys.start + int(a[0]), xs.start + int(a[1]))
                gain = float(G[cell])
                if gain < 0.02 or cell == self.pos[i] or ("region", cell) in banned:
                    k += 1
                    continue
                cheb = lambda q: max(abs(q[0] - cell[0]), abs(q[1] - cell[1])) / D
                regs[k] = [gain, d_me[cell] / D, max(int(d_dep[cell]), 0) / D, float(age[ys, xs].mean()),
                           float((~self.known[ys, xs]).mean()),
                           min([cheb(q) for q in mate_pts], default=1.0),
                           min([cheb(q) for q in mate_cur], default=1.0),
                           float(self.goal[i] == ("region", cell)), sub.size / float(self.by * self.bx)]
                rmask[k] = True
                rcell.append((k, cell))
                k += 1
        reg_cell = [None] * KR
        for kk, cell in rcell:
            reg_cell[kk] = cell
        return dict(items=items, item_mask=imask, item_pos=ipos, regs=regs, reg_mask=rmask,
                    reg_cell=reg_cell, ctx=self.ctx(i))

    def critic_features(self, i: int) -> np.ndarray:
        """Global features for the centralised critic, ego robot first."""
        c, N = self.cfg, self.N
        order = [i] + [j for j in range(N) if j != i]
        blocks = []
        for j in order:
            g = self.goal[j]
            gd = 0.0
            if g is not None and g[0] in ("item", "region"):
                gd = max(int(self.dist_from(self.pos[j])[g[1]]), 0) / self.D
            blocks += [self.pos[j][0] / self.H, self.pos[j][1] / self.W, self.carry_val[j] / c.v_high,
                       self.carry_rem[j] if self.carrying[j] else 0.0,
                       float(g is not None and g[0] == "item"), float(g is not None and g[0] == "region"), gd]
        unclaimed = [p for p in self.items if p not in self.claims]
        n_p = sum(1 for p in unclaimed if self.items[p][0] == 0)
        n_t = sum(1 for p in unclaimed if self.items[p][0] == 1)
        dd = self.dist_from(self.depot)
        md = np.mean([max(int(dd[p]), 0) for p in self.items]) / self.D if self.items else 0.0
        extra = [min(n_p / 10.0, 1.0), min(n_t / 5.0, 1.0), len(self.claims) / N, float(md)]
        return np.concatenate([self.ctx(i), np.asarray(blocks + extra, np.float32)]).astype(np.float32)

    # ------------------------------------------------------------------ low-level control
    def actions(self) -> Dict[str, int]:
        c, N = self.cfg, self.N
        acts = [STAY] * N
        fixed = set()
        tgt: List[Optional[tuple]] = [None] * N
        d_dep = self.dist_from(self.depot)
        for i in range(N):
            p = self.pos[i]
            if self.carrying[i]:
                if p == self.depot:
                    acts[i] = DROP; fixed.add(i)
                else:
                    tgt[i] = self.depot
                continue
            it = self.items.get(p)
            if it is not None and self.claims.get(p, i) == i:
                need = max(int(d_dep[p]), 0) + HANDLING_TIME
                if need <= c.max_steps - self.t and (it[0] == 0 or need <= it[1]):
                    acts[i] = PICK; fixed.add(i)       # take whatever valuable, feasible item we stand on
                    continue
            g = self.goal[i]
            if g is not None and g[0] in ("item", "region"):
                tgt[i] = g[1]

        # Temporarily back out carriers stalled for >= 4 steps to resolve corridor jams.
        for i in range(N):
            if self.carrying[i] and self.stall[i] >= 4 and self.t >= self.retreat[i]:
                self.retreat[i], self.stall[i] = self.t + 6, 0
        retreating = {i for i in range(N) if self.carrying[i] and self.t < self.retreat[i] and i not in fixed}

        def prio(i):
            if i in retreating:
                return (4, 0, i)
            kind = 0 if self.carrying[i] else (1 if (self.goal[i] and self.goal[i][0] == "item") else
                                                 2 if (self.goal[i] and self.goal[i][0] == "region") else 3)
            d = d_dep[self.pos[i]] if self.carrying[i] else 0
            return (kind, d, i)

        order = sorted(range(N), key=prio)
        rank = {i: r for r, i in enumerate(order)}
        dest: Dict[int, tuple] = {i: self.pos[i] for i in fixed}
        yield_req: set = set()
        for i in order:
            if i in fixed:
                continue
            p = self.pos[i]
            hard = {dest[k] for k in dest if k != i and dest[k] != self.depot}
            if i in yield_req or i in retreating:
                a, d = self._sidestep(i, hard)
            elif tgt[i] is None or tgt[i] == p:
                a, d = STAY, p
            else:
                block = set(hard)
                for j in range(N):
                    if j == i or j in dest:
                        continue
                    if rank[j] < rank[i] or self.blocked[i] >= 1:
                        if self.pos[j] != self.depot:
                            block.add(self.pos[j])
                for k, dk in dest.items():
                    if dk == p and self.pos[k] != p:
                        block.add(self.pos[k])
                _, first = self._bfs(p, extra_block=block, stop=tgt[i])
                a = first.get(tgt[i], STAY)
                d = p if a == STAY else (p[0] + MOVES[a][0], p[1] + MOVES[a][1])
                for j in range(N):
                    if j != i and self.pos[j] == d and rank[j] > rank[i] and j not in fixed and d != self.depot:
                        yield_req.add(j)
            acts[i], dest[i] = a, d
        self.last_act = list(acts)
        return {self.names[i]: int(acts[i]) for i in range(N)}

    def _sidestep(self, i: int, hard: set) -> Tuple[int, tuple]:
        p = self.pos[i]
        occupied = {self.pos[j] for j in range(self.N) if j != i}
        best, best_score = (STAY, p), -1e9
        dd = self.dist_from(self.depot)
        for a, (dy, dx) in MOVES.items():
            q = (p[0] + dy, p[1] + dx)
            if not (0 <= q[0] < self.H and 0 <= q[1] < self.W) or not self.free[q] or q in hard or q in occupied:
                continue
            s = float(dd[q]) + self.rng.random() * 0.1
            if s > best_score:
                best, best_score = (a, q), s
        return best


def scripted_choice(cand: dict, D: float, vh: float = 5.0) -> int:
    """Heuristic goal selector over candidate items and regions."""
    best, best_s = 0, 0.0
    for k in range(KI):
        if cand["item_mask"][k]:
            f = cand["items"][k]
            need = (f[4] + f[5]) * D + 2
            s = f[0] * vh / need * (1.0 - 0.6 * f[7]) + 0.05 * f[8]
            if s > best_s:
                best, best_s = 1 + k, s
    if best == 0:
        for k in range(KR):
            if cand["reg_mask"][k]:
                f = cand["regs"][k]
                s = 0.6 * f[0] / (f[1] * D + 3.0) * min(1.0, f[5] * D / 6.0) + 0.002 * f[8]
                if s > best_s:
                    best, best_s = 1 + KI + k, s
    return best


def random_choice(cand: dict, rng: np.random.Generator) -> int:
    """Uniform selection over legal candidate items, regions, or stay."""
    valid = [0] + [1 + k for k in range(KI) if cand["item_mask"][k]] + [1 + KI + k for k in range(KR) if cand["reg_mask"][k]]
    return int(rng.choice(valid))


def nearest_choice(cand: dict) -> int:
    """Select nearest feasible item, falling back to nearest stale region."""
    best, best_d = 0, 1e9
    for k in range(KI):
        if cand["item_mask"][k] and cand["items"][k][4] < best_d:
            best, best_d = 1 + k, cand["items"][k][4]
    if best == 0:
        for k in range(KR):
            if cand["reg_mask"][k] and cand["regs"][k][1] < best_d:
                best, best_d = 1 + KI + k, cand["regs"][k][1]
    return best


class ChooserHierPolicy:
    """Combines HPlanner low-level execution with a goal chooser ('scripted' | 'random' | 'nearest')."""

    def __init__(self, env, kind: str = "scripted", memory: bool = True):
        self.env, self.kind, self.pl = env, kind, HPlanner(env.cfg, memory=memory)
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int):
        self.pl.reset(seed)
        self.rng = np.random.default_rng(seed + 12345)

    def act(self, obs, infos=None):
        pl = self.pl
        pl.perceive(obs)
        for i in pl.need_decision():
            cand = pl.candidates(i)
            if self.kind == "scripted":
                k = scripted_choice(cand, pl.D)
            elif self.kind == "random":
                k = random_choice(cand, self.rng)
            else:
                k = nearest_choice(cand)
            pl.apply_choice(i, k, cand)
        return pl.actions()


class ScriptedHierPolicy(ChooserHierPolicy):
    """Scripted goal chooser with HPlanner executor."""

    def __init__(self, env, memory: bool = True):
        super().__init__(env, kind="scripted", memory=memory)