"""Training reward shaping: individual credit assignment, coordination bonuses, and BFS potentials."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class RewardConfig:
    lam_ind: float = 0.3          # weight for individual reward block
    own_delivery: float = 0.3     # fraction of delivered value credited to deliverer
    own_discover: float = 0.1     # discovery bonus for first spotting valuable item
    own_cover: float = 0.1        # novel coverage reward per cell normalized by window area
    own_collision: float = 0.05   # collision penalty for individual agent
    w_stale_cov: float = 0.30     # team reward weight for revisiting stale areas
    age_cap: float = 50.0         # max staleness step cap
    w_overlap: float = 0.02       # team penalty for redundant sensing among free agents
    w_sync: float = 0.25          # team bonus for simultaneous agent pickups
    sync_window: int = 8          # step window to qualify for sync pickup
    psi: float = 0.30             # potential scale on carried value
    gamma: float = 0.995          # discount factor for potential shaping


class RewardShaper:
    """Stateful environment reward shaper for multi-agent coordination."""

    def __init__(self, cfg: RewardConfig, n_agents: int, height: int, width: int, view_radius: int,
                 v_high: float):
        self.c, self.N, self.H, self.W, self.r, self.v_high = cfg, n_agents, height, width, view_radius, v_high
        self.win_area = float((2 * view_radius + 1) ** 2)

    def _windows(self, pos) -> List[np.ndarray]:
        out = []
        for (y, x) in pos:
            m = np.zeros((self.H, self.W), bool)
            m[max(0, y - self.r):y + self.r + 1, max(0, x - self.r):x + self.r + 1] = True
            out.append(m)
        return out

    def _phi(self, env) -> np.ndarray:
        p = np.zeros(self.N, np.float32)
        for i, it in enumerate(env.carry):
            if it is not None and it.type != 2:
                p[i] = self.c.psi * it.value
        return p

    def reset(self, env):
        self.last_seen = np.full((self.H, self.W), -1e9, np.float32)
        self.ever = np.zeros((self.H, self.W), bool)
        self.last_pick = np.full(self.N, -10 ** 6)
        self.prev_phi = self._phi(env)
        self._observe(env.pos, 0, update_only=True)
        self.prev_discovered = 0

    def _observe(self, pos, t, update_only=False):
        """Update staleness bookkeeping; return (stale gain per agent, novel cells per agent, overlap cells)."""
        wins = self._windows(pos)
        cnt = np.sum(wins, axis=0)
        age = np.clip(t - self.last_seen, 0.0, self.c.age_cap) / self.c.age_cap          # 1 if never seen
        gain_i = np.zeros(self.N, np.float32)
        nov_i = np.zeros(self.N, np.float32)
        for i, w in enumerate(wins):
            share = np.where(w, 1.0 / np.maximum(cnt, 1), 0.0)                            # split simultaneous observers
            gain_i[i] = float((age * share).sum())
            nov_i[i] = float(((~self.ever) * share).sum())
        self.last_seen[cnt > 0] = t
        self.ever |= cnt > 0
        overlap = float(np.maximum(cnt - 1, 0).sum())
        return (gain_i, nov_i, overlap, wins) if not update_only else None

    def step(self, env, team_reward: float, comp: Dict[str, float], deliveries: List[dict],
             collided: np.ndarray, new_item_owner: List[int], picked: List[int], done: bool,
             scale: float) -> np.ndarray:

        c, N = self.c, self.N
        t = env.t
        gain_i, nov_i, overlap, wins = self._observe(env.pos, t)

        ind = np.zeros(N, np.float32)
        for d in deliveries:
            ind[d["agent"]] += c.own_delivery * d["value"]
        for i in new_item_owner:
            ind[i] += c.own_discover
        ind += c.own_cover * nov_i / self.win_area
        ind -= c.own_collision * collided.astype(np.float32)

        free = [i for i in range(N) if env.carry[i] is None]
        stale = c.w_stale_cov * float(gain_i.sum()) / (self.H * self.W)
        if len(free) >= 2:
            cnt_f = np.sum([wins[i] for i in free], axis=0)
            ov = float(np.maximum(cnt_f - 1, 0).sum()) / self.win_area
        else:
            ov = 0.0
        sync = 0.0
        for i in picked:
            others = [j for j in range(N) if j != i and t - self.last_pick[j] <= c.sync_window]
            if others and N > 1:
                sync += c.w_sync * len(others) / (N - 1)
            self.last_pick[i] = t
        coord = stale - c.w_overlap * ov + sync

        phi = np.zeros(N, np.float32) if done else self._phi(env)
        F = c.gamma * phi - self.prev_phi
        self.prev_phi = phi

        return (team_reward + c.lam_ind * ind + scale * coord + F).astype(np.float32)
