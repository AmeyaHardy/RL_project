"""
Scripted baseline: frontier exploration + greedy retrieval, using ONLY
observations (team map + scalars). This is the performance floor
RL should beat.

Usage:
    python greedy_baseline.py
    python greedy_baseline.py --episodes 20 --seed 42
"""
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


def bfs(free: np.ndarray, start: Tuple[int, int]) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """Breadth-first search on a 2D grid of passable cells.

    Returns:
        dist: 2D array of shortest-path distances (-1 for unreachable)
        first: mapping from reachable (y, x) to the first action on the shortest path from start
    """
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
    """Derive legal action mask strictly from the agent's observation and public config.

    Actions: [UP, DOWN, LEFT, RIGHT, STAY, PICK, DROP]
    Zero hidden-state leakage: does NOT inspect env.grid, env.active, or env.carry.
    """
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

    # 1. Movement: check grid boundaries and known obstacles
    for a, (dy, dx) in DIRS.items():
        ny, nx = pos[0] + dy, pos[1] + dx
        if 0 <= ny < H and 0 <= nx < W:
            # Movement into unknown in-bounds cells is permitted.
            # Only known obstacles are blocked.
            if not (known[ny, nx] and kobs[ny, nx]):
                mask[a] = True

    # 2. STAY is always legal
    mask[STAY] = True

    # 3. PICK: legal only if not carrying and an item is observed at pos
    if not carrying:
        has_persist = team[2, pos[0], pos[1]] > 0
        has_temp = team[3, pos[0], pos[1]] > 0
        has_mal = team[4, pos[0], pos[1]] > 0
        has_item = has_persist or has_temp or has_mal

        if has_item:
            is_claimed = team[6, pos[0], pos[1]] > 0.5
            # When claim_blocks_pickup is True, respect channel 6 conservatively
            if cfg.claim_blocks_pickup and is_claimed:
                mask[PICK] = False
            elif has_mal and not has_persist and not has_temp and cfg.forbid_malicious_pickup:
                mask[PICK] = False
            else:
                mask[PICK] = True

    # 4. DROP: legal only if carrying and at the depot
    if carrying:
        at_depot = (pos == depot) or (abs(float(sc[6])) < 1e-4 and abs(float(sc[7])) < 1e-4)
        if at_depot:
            mask[DROP] = True

    return mask


class GreedyPolicy:
    """Observation-only scripted baseline with persistent target coordination.

    Features:
    - Zero hidden-state leakage: action mask and path planning use only observed maps.
    - Respects claim_blocks_pickup and claimed-items channel (channel 6) conservatively.
    - Separate visited-frontier tracking per agent in private mode, shared in shared/window modes.
    - Frontiers marked visited only upon arrival or when no longer a frontier.
    - Strict deadline feasibility: temporary items require a reachable route to depot (no Manhattan fallback).
    - Seeded, lazy stochastic choices for tie-breaking and wiggling.
    - Clean reset clearing all per-episode tracking state.
    """

    def __init__(self, env: SwarmSearchEnv, seed: int = 0):
        self.env = env
        self.cfg = env.cfg
        self.life_max = float(env.cfg.temp_life_range[1])
        self.rng = np.random.default_rng(seed)
        self.targets: Dict[str, Tuple[int, int]] = {}
        self.target_type: Dict[str, str] = {}
        self.visited_frontiers: Set[Tuple[int, int]] = set()  # shared / window modes
        self.agent_visited_frontiers: Dict[str, Set[Tuple[int, int]]] = {}  # private mode

    def reset(self):
        """Reset all episode-specific coordination and exploration state."""
        self.targets.clear()
        self.target_type.clear()
        self.visited_frontiers.clear()
        self.agent_visited_frontiers.clear()

    def observation_action_mask(self, obs_a: dict) -> np.ndarray:
        return observation_action_mask(obs_a, self.cfg)

    def _wiggle(self, mask: np.ndarray) -> int:
        valid = np.flatnonzero(mask[:4])
        if len(valid) > 0:
            return int(self.rng.choice(valid))
        return STAY

    def _toward(self, mask: np.ndarray, dy: int, dx: int) -> int:
        if abs(dy) >= abs(dx):
            prefs = [DOWN if dy > 0 else UP, RIGHT if dx > 0 else LEFT]
        else:
            prefs = [RIGHT if dx > 0 else LEFT, DOWN if dy > 0 else UP]
        for p in prefs:
            if mask[p] and (dy if p in (UP, DOWN) else dx) != 0:
                return p
        return self._wiggle(mask)

    def act(self, obs: dict) -> Dict[str, int]:
        cfg = self.cfg
        H, W = cfg.height, cfg.width
        # Both 'shared' and 'window' communicate live observations across the team
        shared = cfg.info_mode in ("shared", "window")
        window_mode = (cfg.info_mode == "window")
        actions: Dict[str, int] = {}

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

            # Visited frontiers set: shared for shared/window modes, per-agent for private mode
            if shared:
                visited = self.visited_frontiers
            else:
                if a not in self.agent_visited_frontiers:
                    self.agent_visited_frontiers[a] = set()
                visited = self.agent_visited_frontiers[a]

            # Mark frontier as visited only when agent reaches it
            if self.target_type.get(a) == "frontier":
                prev_tgt = self.targets.get(a)
                if prev_tgt is not None and pos == prev_tgt:
                    visited.add(pos)
                    self.targets.pop(a, None)
                    self.target_type.pop(a, None)

            # Avoid teammate claims when communicating (shared or window mode)
            teammate_claims = {tgt for ag, tgt in self.targets.items() if ag != a} if shared else set()

            # Observable valuable items: persistent (ch 2) and temporary (ch 3)
            # Avoid malicious items (ch 4) as target candidates
            items = {}
            for t in (PERSIST, TEMP):
                for y, x in np.argwhere(team[2 + t] > 0):
                    p = (int(y), int(x))
                    # Conservative claim handling: if claim_blocks_pickup is enabled and claimed, skip
                    if cfg.claim_blocks_pickup and team[6, p[0], p[1]] > 0.5:
                        continue
                    items[p] = (t, float(team[2 + t, y, x]))

            # Plan paths around other visible robots to minimize congestion
            plan_free = free.copy()
            for q in np.argwhere(team[5] > 0.5):
                qy, qx = int(q[0]), int(q[1])
                if (qy, qx) != pos and (qy, qx) != depot:
                    plan_free[qy, qx] = False

            # Distance from current position through known free space
            dist, first = bfs(plan_free, pos)

            # Distance to depot: in memoryless window mode, cells outside current collective
            # line-of-sight are unknown, so route through non-obstacle cells ~(known & kobs)
            # while still blocking known obstacles.
            if window_mode:
                opt_free = ~(known & kobs)
                dep_dist, _ = bfs(opt_free, depot)
            else:
                dep_dist, _ = bfs(free, depot)

            # CASE 1: Carrying an item -> return to depot
            if carrying:
                self.targets.pop(a, None)
                self.target_type.pop(a, None)
                if (pos == depot or (abs(float(sc[6])) < 1e-4 and abs(float(sc[7])) < 1e-4)) and mask[DROP]:
                    actions[a] = DROP
                elif depot in first:
                    actions[a] = first[depot]
                elif window_mode:
                    # In memoryless window mode, navigate around currently visible obstacles toward depot
                    opt_plan = ~(known & kobs)
                    for q in np.argwhere(team[5] > 0.5):
                        qy, qx = int(q[0]), int(q[1])
                        if (qy, qx) != pos and (qy, qx) != depot:
                            opt_plan[qy, qx] = False
                    _, opt_first = bfs(opt_plan, pos)
                    if depot in opt_first and mask[opt_first[depot]]:
                        actions[a] = opt_first[depot]
                    else:
                        actions[a] = self._toward(mask, depot[0] - pos[0], depot[1] - pos[1])
                elif free[depot]:
                    actions[a] = STAY
                else:
                    actions[a] = self._toward(mask, depot[0] - pos[0], depot[1] - pos[1])
                continue

            # CASE 2: At item position and can pick
            if mask[PICK] and pos in items:
                actions[a] = PICK
                self.targets.pop(a, None)
                self.target_type.pop(a, None)
                continue

            # CASE 3: Pursue an item target
            curr_tgt = self.targets.get(a)
            curr_type = self.target_type.get(a)
            best_item = None
            best_score = -1.0

            # Verify existing persistent item target
            if curr_type == "item" and curr_tgt is not None:
                if curr_tgt in items and dist[curr_tgt] >= 0:
                    t, rem = items[curr_tgt]
                    is_feasible = True
                    if t == TEMP:
                        # Feasibility requires a reachable route to depot (no Manhattan distance fallback)
                        if dep_dist[curr_tgt] < 0:
                            is_feasible = False
                        else:
                            needed = dist[curr_tgt] + 1 + dep_dist[curr_tgt] + 1
                            if rem * self.life_max < needed:
                                is_feasible = False
                    if is_feasible:
                        best_item = curr_tgt
                        val = cfg.v_high if t == TEMP else cfg.v_low
                        best_score = val / (dist[curr_tgt] + 1)

            # If no valid active target, select best available item
            if best_item is None:
                for p, (t, rem) in items.items():
                    if p in teammate_claims or dist[p] < 0:
                        continue
                    if t == TEMP:
                        # Must have established reachable route to depot
                        if dep_dist[p] < 0:
                            continue
                        needed = dist[p] + 1 + dep_dist[p] + 1
                        if rem * self.life_max < needed:
                            continue
                    val = cfg.v_high if t == TEMP else cfg.v_low
                    score = val / (dist[p] + 1)
                    if score > best_score:
                        best_score = score
                        best_item = p

            if best_item is not None:
                self.targets[a] = best_item
                self.target_type[a] = "item"
                if best_item in first:
                    actions[a] = first[best_item]
                else:
                    actions[a] = self._wiggle(mask)
                continue

            # CASE 4: Frontier exploration
            unk = ~known
            nu = np.zeros_like(known)
            nu[1:] |= unk[:-1]
            nu[:-1] |= unk[1:]
            nu[:, 1:] |= unk[:, :-1]
            nu[:, :-1] |= unk[:, 1:]
            frontier = free & nu

            chosen_frontier = None
            # Retain existing valid frontier target if still reachable and unobserved border exists
            if curr_type == "frontier" and curr_tgt is not None:
                if 0 <= curr_tgt[0] < H and 0 <= curr_tgt[1] < W:
                    if frontier[curr_tgt] and dist[curr_tgt] > 0:
                        chosen_frontier = curr_tgt

            if chosen_frontier is None:
                cands = []
                for p in np.argwhere(frontier):
                    pt = (int(p[0]), int(p[1]))
                    if dist[pt] > 0 and pt not in teammate_claims:
                        is_vis = pt in visited
                        cands.append((is_vis, dist[pt], pt[0], pt[1], pt))

                if cands:
                    cands.sort()
                    _, _, _, _, chosen_frontier = cands[0]

            if chosen_frontier is not None:
                self.targets[a] = chosen_frontier
                self.target_type[a] = "frontier"
                # Do NOT mark visited here; marked only upon arrival (or when no longer frontier)
                if chosen_frontier in first:
                    actions[a] = first[chosen_frontier]
                else:
                    actions[a] = self._wiggle(mask)
            else:
                self.targets.pop(a, None)
                self.target_type.pop(a, None)
                actions[a] = self._wiggle(mask)

        return actions


def evaluate(n_episodes: int = 20, seed: int = 0, cfg: Optional[EnvConfig] = None, verbose: bool = False):
    """Run greedy baseline evaluation across multiple episodes and record full metrics."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1")
    cfg = cfg or EnvConfig()
    env = SwarmSearchEnv(cfg)
    rows = []
    pol = GreedyPolicy(env, seed=seed)

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
    p = argparse.ArgumentParser(description="Observation-only Greedy Baseline evaluation")
    p.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes (>=1)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--modes", nargs="+", default=None,
                   help="Info modes to test (default: all supported ['shared', 'private', 'window'])")
    args = p.parse_args()

    if args.episodes < 1:
        p.error("--episodes must be at least 1.")

    modes = args.modes or ["shared", "private", "window"]
    for mode in modes:
        print(f"\n=== info_mode={mode} ===")
        cfg = EnvConfig(info_mode=mode)
        rows = evaluate(args.episodes, seed=args.seed, cfg=cfg, verbose=False)
        _print_summary(mode, rows)


if __name__ == "__main__":
    main()