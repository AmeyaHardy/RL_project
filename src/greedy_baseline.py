from __future__ import annotations

import argparse
from collections import deque
from typing import Dict, Optional, Set, Tuple

import numpy as np

try:
    from env import (DOWN, DROP, LEFT, PICK, RIGHT, STAY, UP,
                     EnvConfig, SwarmSearchEnv, HANDLING_TIME, PERSIST, TEMP, MAL)
except ImportError:
    from swarm_env import (DOWN, DROP, LEFT, PICK, RIGHT, STAY, UP,
                           EnvConfig, SwarmSearchEnv, HANDLING_TIME, PERSIST, TEMP, MAL)

DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


def bfs(free: np.ndarray, start: Tuple[int, int]) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """Compute shortest path distances and first-step actions on a 2D grid."""
    H, W = free.shape
    dist = -np.ones((H, W), dtype=np.int32)
    first = {}
    if not (0 <= start[0] < H and 0 <= start[1] < W) or not free[start]:
        return dist, first

    dist[start] = 0
    q = deque([start])
    while q:
        y, x = q.popleft()
        for a, (dy, dx) in DIRS.items():
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and free[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                first[(ny, nx)] = first.get((y, x), a) if (y, x) != start else a
                q.append((ny, nx))
    return dist, first


def observation_action_mask(obs_a: dict, cfg: EnvConfig) -> np.ndarray:
    """Derive legal action mask strictly from agent observation and configuration."""
    H, W = cfg.height, cfg.width
    team = obs_a["team_map"]
    sc = obs_a["scalars"]

    pos = (
        int(np.clip(round(float(sc[0]) * H), 0, H - 1)),
        int(np.clip(round(float(sc[1]) * W), 0, W - 1)),
    )
    depot = (
        int(np.clip(round(float(pos[0] + sc[6] * H)), 0, H - 1)),
        int(np.clip(round(float(pos[1] + sc[7] * W)), 0, W - 1)),
    )
    carrying = bool(sc[2] > 0.5)

    known = team[0] > 0.5
    kobs = team[1] > 0.5
    mask = np.zeros(7, dtype=bool)

    for a, (dy, dx) in DIRS.items():
        ny, nx = pos[0] + dy, pos[1] + dx
        if 0 <= ny < H and 0 <= nx < W:
            if not (known[ny, nx] and kobs[ny, nx]):
                mask[a] = True

    mask[STAY] = True

    if not carrying:
        has_persist = team[2, pos[0], pos[1]] > 0
        has_temp = team[3, pos[0], pos[1]] > 0
        has_mal = team[4, pos[0], pos[1]] > 0
        if has_persist or has_temp or has_mal:
            is_claimed = team[6, pos[0], pos[1]] > 0.5
            if cfg.claim_blocks_pickup and is_claimed:
                mask[PICK] = False
            elif has_mal and not has_persist and not has_temp and cfg.forbid_malicious_pickup:
                mask[PICK] = False
            else:
                mask[PICK] = True

    if carrying:
        at_depot = (pos == depot) or (abs(float(sc[6])) < 1e-4 and abs(float(sc[7])) < 1e-4)
        if at_depot:
            mask[DROP] = True

    return mask


class GreedyPolicy:
    """Observation-only scripted baseline with target coordination and exploration."""

    def __init__(self, env: SwarmSearchEnv, seed: int = 0,
                 window_memory: Optional[bool] = None,
                 momentum: Optional[bool] = None,
                 optimistic_planning: Optional[bool] = None,
                 dedupe: Optional[bool] = None,
                 turn_prob: float = 0.08):
        """Initialize policy settings, tracking structures, and random generator."""
        self.env = env
        self.cfg = env.cfg
        self.life_max = float(env.cfg.temp_life_range[1])
        self.rng = np.random.default_rng(seed)
        win = env.cfg.info_mode == "window"
        self._use_memory = win and bool(window_memory)
        self._use_momentum = win and (True if momentum is None else momentum)
        self._optimistic = win and (True if optimistic_planning is None else optimistic_planning)
        self._dedupe = win and (True if dedupe is None else dedupe)
        self.turn_prob = turn_prob
        self.heading: Dict[str, int] = {}
        self.last_pos: Dict[str, Tuple[int, int]] = {}
        self.prev_pos: Dict[str, Tuple[int, int]] = {}
        self.mem: Dict[str, dict] = {}
        self._last_t = -1
        self.targets: Dict[str, Tuple[int, int]] = {}
        self.target_type: Dict[str, str] = {}
        self.visited_frontiers: Set[Tuple[int, int]] = set()
        self.agent_visited_frontiers: Dict[str, Set[Tuple[int, int]]] = {}

    def reset(self):
        """Reset policy coordination and per-episode tracking state."""
        self.targets.clear()
        self.target_type.clear()
        self.visited_frontiers.clear()
        self.agent_visited_frontiers.clear()
        self.heading.clear()
        self.last_pos.clear()
        self.prev_pos.clear()
        self.mem.clear()
        self._last_t = -1

    def observation_action_mask(self, obs_a: dict) -> np.ndarray:
        """Derive legal action mask for agent observation."""
        return observation_action_mask(obs_a, self.cfg)

    def _wiggle(self, mask: np.ndarray, avoid_dir: Optional[int] = None) -> int:
        """Select a random legal move, avoiding reversal if alternatives exist."""
        valid = np.flatnonzero(mask[:4])
        if len(valid) == 0:
            return STAY
        if avoid_dir is not None and len(valid) > 1:
            non_rev = [m for m in valid if m != avoid_dir]
            if non_rev:
                return int(self.rng.choice(non_rev))
        return int(self.rng.choice(valid))

    def _toward(self, mask: np.ndarray, dy: int, dx: int, avoid_dir: Optional[int] = None) -> int:
        """Select preferred legal move towards delta offsets."""
        if abs(dy) >= abs(dx):
            prefs = [DOWN if dy > 0 else UP, RIGHT if dx > 0 else LEFT]
        else:
            prefs = [RIGHT if dx > 0 else LEFT, DOWN if dy > 0 else UP]
        for p in prefs:
            if mask[p] and p != avoid_dir and (dy if p in (UP, DOWN) else dx) != 0:
                return p
        return self._wiggle(mask, avoid_dir=avoid_dir)

    def _momentum(self, a: str, mask: np.ndarray, pos: Tuple[int, int]) -> int:
        """Select action maintaining heading with occasional turns."""
        valid = [m for m in range(4) if mask[m]]
        if not valid:
            return STAY
        h = self.heading.get(a)
        stuck = self.last_pos.get(a) == pos and h is not None
        self.last_pos[a] = pos
        if h is None or stuck or not mask[h] or self.rng.random() < self.turn_prob:
            opts = [m for m in valid if h is None or m != OPPOSITE[h]] or valid
            h = int(self.rng.choice(opts))
        self.heading[a] = h
        return h

    def _remember(self, obs: dict) -> dict:
        """Accumulate shared views into policy-side memory buffer."""
        cfg = self.cfg
        H, W = cfg.height, cfg.width
        first = next(iter(obs.values()))
        t = int(round(float(first["scalars"][4]) * cfg.max_steps))
        if t < self._last_t:
            self.mem.clear()
            self.heading.clear()
            self.last_pos.clear()
            self.prev_pos.clear()
        self._last_t = t
        out = {}
        for a, o in obs.items():
            team = o["team_map"]
            m = self.mem.get(a)
            if m is None:
                m = {"known": np.zeros((H, W), bool), "kobs": np.zeros((H, W), bool),
                     "type": -np.ones((H, W), np.int8), "exp": np.full((H, W), np.inf, np.float32)}
                self.mem[a] = m
            win = team[0] > 0.5
            m["known"] |= win
            m["kobs"] |= team[1] > 0.5
            m["type"][win] = -1
            m["exp"][win] = np.inf
            m["type"][team[2] > 0.5] = PERSIST
            tm_ = team[3] > 0
            m["type"][tm_] = TEMP
            m["exp"][tm_] = t + team[3][tm_] * self.life_max
            m["type"][team[4] > 0.5] = MAL
            mt, me = m["type"], m["exp"]
            new = np.zeros_like(team)
            new[0], new[1] = m["known"], m["kobs"]
            new[2] = mt == PERSIST
            alive = (mt == TEMP) & ((me - t) > 0.5)
            new[3] = np.where(alive, np.clip((me - t) / self.life_max, 0.0, 1.0), 0.0)
            new[4] = mt == MAL
            new[5:] = team[5:]
            out[a] = {**o, "team_map": new}
        return out

    def act(self, obs: dict) -> Dict[str, int]:
        """Compute coordinated actions for all agents from observations."""
        if self._use_memory:
            obs = self._remember(obs)
        cfg = self.cfg
        H, W = cfg.height, cfg.width
        shared = (cfg.info_mode == "shared")
        actions: Dict[str, int] = {}

        carrying_positions = set()
        if len(obs) > 1:
            for ag, o_ag in obs.items():
                sc_ag = o_ag["scalars"]
                if sc_ag[2] > 0.5:
                    cy_ag = int(np.clip(round(float(sc_ag[0]) * H), 0, H - 1))
                    cx_ag = int(np.clip(round(float(sc_ag[1]) * W), 0, W - 1))
                    carrying_positions.add((cy_ag, cx_ag))

        for a, o in obs.items():
            team = o["team_map"]
            sc = o["scalars"]
            known = team[0] > 0.5
            kobs = team[1] > 0.5
            free = known & ~kobs

            pos = (
                int(np.clip(round(float(sc[0]) * H), 0, H - 1)),
                int(np.clip(round(float(sc[1]) * W), 0, W - 1)),
            )
            depot = (
                int(np.clip(round(float(pos[0] + sc[6] * H)), 0, H - 1)),
                int(np.clip(round(float(pos[1] + sc[7] * W)), 0, W - 1)),
            )
            carrying = bool(sc[2] > 0.5)
            mask = observation_action_mask(o, cfg)

            prev = self.prev_pos.get(a)
            avoid_dir = None
            if prev is not None and prev != pos:
                for d_act, (dy, dx) in DIRS.items():
                    if (pos[0] + dy, pos[1] + dx) == prev:
                        avoid_dir = d_act
                        break
            self.prev_pos[a] = pos

            visited = self.visited_frontiers if shared else self.agent_visited_frontiers.setdefault(a, set())

            if self.target_type.get(a) == "frontier":
                prev_tgt = self.targets.get(a)
                if prev_tgt is not None and pos == prev_tgt:
                    visited.add(pos)
                    self.targets.pop(a, None)
                    self.target_type.pop(a, None)

            teammate_claims = (
                {tgt for ag, tgt in self.targets.items() if ag != a}
                if (shared or self._dedupe) else set()
            )

            items = {}
            for t in (PERSIST, TEMP):
                for y, x in np.argwhere(team[2 + t] > 0):
                    p = (int(y), int(x))
                    if cfg.claim_blocks_pickup and team[6, p[0], p[1]] > 0.5:
                        continue
                    items[p] = (t, float(team[2 + t, y, x]))

            base_free = (~kobs).copy() if self._optimistic else free.copy()
            plan_free = base_free.copy()
            for q in np.argwhere(team[5] > 0.5):
                qy, qx = int(q[0]), int(q[1])
                if (qy, qx) != pos and (qy, qx) != depot:
                    if max(abs(qy - pos[0]), abs(qx - pos[1])) <= 2:
                        plan_free[qy, qx] = False

            if self._optimistic and prev is not None and prev != pos:
                plan_no_rev = plan_free.copy()
                plan_no_rev[prev] = False
                dist_nr, first_nr = bfs(plan_no_rev, pos)
            else:
                dist_nr, first_nr = None, None

            dist, first = bfs(plan_free, pos)
            base_dist, base_first = bfs(base_free, pos)
            dep_dist, _ = bfs(free, depot)
            dep_opt = bfs(~kobs, depot)[0] if self._optimistic else None

            def dd(p):
                v = int(dep_dist[p])
                return v if (v >= 0 or dep_opt is None) else int(dep_opt[p])

            if carrying:
                self.targets.pop(a, None)
                self.target_type.pop(a, None)
                if (pos == depot or (abs(float(sc[6])) < 1e-4 and abs(float(sc[7])) < 1e-4)) and mask[DROP]:
                    actions[a] = DROP
                elif first_nr is not None and depot in first_nr and mask[first_nr[depot]]:
                    actions[a] = first_nr[depot]
                elif depot in first and mask[first[depot]]:
                    actions[a] = first[depot]
                elif depot in base_first and mask[base_first[depot]]:
                    actions[a] = base_first[depot]
                elif free[depot]:
                    actions[a] = STAY
                else:
                    actions[a] = self._toward(mask, depot[0] - pos[0], depot[1] - pos[1], avoid_dir=avoid_dir)
                if actions[a] in DIRS:      # only real moves define a heading (not DROP/PICK/STAY)
                    self.heading[a] = actions[a]
                continue

            if mask[PICK] and pos in items:
                actions[a] = PICK
                self.targets.pop(a, None)
                self.target_type.pop(a, None)
                continue

            curr_tgt = self.targets.get(a)
            curr_type = self.target_type.get(a)
            best_item = None
            best_score = -1.0

            def item_score(p, t, rem, d_val):
                val = cfg.v_high if t == TEMP else cfg.v_low
                if t == TEMP:
                    val = val + 0.5 * (1.0 - np.clip(rem, 0.0, 1.0))
                return val / (d_val + 1)

            if curr_type == "item" and curr_tgt is not None and curr_tgt in items:
                d_curr = dist[curr_tgt] if dist[curr_tgt] >= 0 else base_dist[curr_tgt]
                if curr_tgt in items and curr_tgt not in teammate_claims and d_curr >= 0:
                    t, rem = items[curr_tgt]
                    is_feasible = True
                    if t == TEMP:
                        if dd(curr_tgt) < 0:
                            is_feasible = False
                        else:
                            needed = d_curr + 1 + dd(curr_tgt) + 1
                            if rem * self.life_max < needed:
                                is_feasible = False
                    if is_feasible:
                        best_item = curr_tgt
                        best_score = item_score(curr_tgt, t, rem, d_curr)

            others = []
            if self._dedupe:
                others = [
                    (int(q[0]), int(q[1])) for q in np.argwhere(team[5] > 0.5)
                    if (int(q[0]), int(q[1])) != pos and (int(q[0]), int(q[1])) not in carrying_positions
                ]

            def owned_by_other(p):
                if not others:
                    return False
                p_dist, _ = bfs(~kobs, p)
                my_d = int(p_dist[pos]) if p_dist[pos] >= 0 else abs(pos[0] - p[0]) + abs(pos[1] - p[1])
                me = (my_d, pos)
                for q in others:
                    qd = int(p_dist[q]) if p_dist[q] >= 0 else abs(q[0] - p[0]) + abs(q[1] - p[1])
                    if p_dist[q] >= 0 and (qd, q) < me:
                        return True
                return False

            if best_item is None:
                for p, (t, rem) in items.items():
                    d_p = dist[p] if dist[p] >= 0 else base_dist[p]
                    if p in teammate_claims or d_p < 0 or owned_by_other(p):
                        continue
                    if t == TEMP:
                        if dd(p) < 0:
                            continue
                        needed = d_p + 1 + dd(p) + 1
                        if rem * self.life_max < needed:
                            continue
                    score = item_score(p, t, rem, d_p)
                    if score > best_score:
                        best_score = score
                        best_item = p

            if best_item is not None:
                self.targets[a] = best_item
                self.target_type[a] = "item"
                if first_nr is not None and best_item in first_nr and mask[first_nr[best_item]]:
                    act_choice = first_nr[best_item]
                elif best_item in first and mask[first[best_item]]:
                    act_choice = first[best_item]
                elif best_item in base_first and mask[base_first[best_item]]:
                    act_choice = base_first[best_item]
                else:
                    act_choice = self._wiggle(mask, avoid_dir=avoid_dir)
                actions[a] = act_choice
                if act_choice in DIRS:
                    self.heading[a] = act_choice
                continue

            if self._use_momentum:
                self.targets.pop(a, None)
                self.target_type.pop(a, None)
                actions[a] = self._momentum(a, mask, pos)
                continue

            unk = ~known
            nu = np.zeros_like(known)
            nu[1:] |= unk[:-1]
            nu[:-1] |= unk[1:]
            nu[:, 1:] |= unk[:, :-1]
            nu[:, :-1] |= unk[:, 1:]
            frontier = free & nu

            chosen_frontier = None
            if curr_type == "frontier" and curr_tgt is not None:
                if 0 <= curr_tgt[0] < H and 0 <= curr_tgt[1] < W:
                    d_f = dist[curr_tgt] if dist[curr_tgt] > 0 else base_dist[curr_tgt]
                    if frontier[curr_tgt] and d_f > 0 and curr_tgt not in teammate_claims:
                        chosen_frontier = curr_tgt

            if chosen_frontier is None:
                cands = []
                for p in np.argwhere(frontier):
                    pt = (int(p[0]), int(p[1]))
                    d_pt = dist[pt] if dist[pt] > 0 else base_dist[pt]
                    if d_pt > 0 and pt not in teammate_claims:
                        cands.append((pt in visited, d_pt, pt[0], pt[1], pt))

                if cands:
                    cands.sort()
                    chosen_frontier = cands[0][4]

            if chosen_frontier is not None:
                self.targets[a] = chosen_frontier
                self.target_type[a] = "frontier"
                if chosen_frontier in first and mask[first[chosen_frontier]]:
                    actions[a] = first[chosen_frontier]
                elif chosen_frontier in base_first and mask[base_first[chosen_frontier]]:
                    actions[a] = base_first[chosen_frontier]
                else:
                    actions[a] = self._wiggle(mask, avoid_dir=avoid_dir)
            else:
                self.targets.pop(a, None)
                self.target_type.pop(a, None)
                actions[a] = self._wiggle(mask, avoid_dir=avoid_dir)

        return actions


def evaluate(n_episodes: int = 20, seed: int = 0, cfg: Optional[EnvConfig] = None,
             verbose: bool = False, policy_kwargs: Optional[dict] = None):
    """Evaluate greedy policy over multiple episodes and return statistics list."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1")
    cfg = cfg or EnvConfig()
    env = SwarmSearchEnv(cfg)
    rows = []
    pol = GreedyPolicy(env, seed=seed, **(policy_kwargs or {}))

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        pol.reset()
        ep_len = 0
        while env.agents:
            obs, r, term, trunc, info = env.step(pol.act(obs))
            ep_len += 1

        stat = env.stats.copy()
        stat["ep_length"] = ep_len
        rows.append(stat)
        if verbose:
            print(f"  ep {ep:3d}  return {stat['ep_return']:6.1f}  "
                  f"deliveries {stat['deliveries']}  steps {ep_len}")

    return rows


def _print_summary(mode: str, rows: list):
    """Print aggregate metrics summary for evaluation episodes."""
    if not rows:
        print(f"{mode:8s}  No episodes evaluated.")
        return
    m = lambda k: float(np.mean([r[k] for r in rows]))
    d = np.mean([r["delivered_by_type"] for r in rows], axis=0).round(1)
    e = np.mean([r["expired_by_type"] for r in rows], axis=0).round(1)
    lens = [r["ep_length"] for r in rows]
    print(f"{mode:8s}  return {m('ep_return'):6.1f}  value {m('delivered_value'):5.1f}  "
          f"deliv P/T/M {d[0]}/{d[1]}/{d[2]} (tot {m('deliveries'):.1f})  "
          f"temp expired {e[1]}  collisions {m('collisions'):.0f}  "
          f"malicious picked {m('malicious_picked'):.1f}  avg steps {np.mean(lens):.1f}")


def main():
    """CLI entrypoint for running greedy baseline evaluations."""
    p = argparse.ArgumentParser(description="Observation-only Greedy Baseline evaluation")
    p.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes (>=1)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--modes", nargs="+", default=None,
                   help="Info modes to test (default: all supported ['shared', 'private', 'window'])")
    p.add_argument("--ablate-window", action="store_true",
                   help="Compare window-mode improvements one by one")
    args = p.parse_args()

    if args.episodes < 1:
        p.error("--episodes must be at least 1.")

    if args.ablate_window:
        off = dict(window_memory=False, momentum=False, optimistic_planning=False, dedupe=False)
        variants = {
            "strict (known cells only)":  dict(off),
            "+optimistic planning":       {**off, "optimistic_planning": True},
            "+dedupe":                    {**off, "optimistic_planning": True, "dedupe": True},
            "+momentum (default)":        {**off, "optimistic_planning": True, "dedupe": True, "momentum": True},
            "[ref] +policy memory":       {**off, "optimistic_planning": True, "dedupe": True, "window_memory": True},
        }
        cfg = EnvConfig(info_mode="window")
        for name, kw in variants.items():
            rows = evaluate(args.episodes, seed=args.seed, cfg=cfg, policy_kwargs=kw)
            _print_summary(f"{name}", rows)
        return

    modes = args.modes or ["shared", "private", "window"]
    for mode in modes:
        print(f"\n=== info_mode={mode} ===")
        cfg = EnvConfig(info_mode=mode)
        rows = evaluate(args.episodes, seed=args.seed, cfg=cfg, verbose=False)
        _print_summary(mode, rows)


if __name__ == "__main__":
    main()