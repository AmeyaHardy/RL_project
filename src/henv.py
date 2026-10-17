"""One SwarmSearchEnv + HPlanner + RewardShaper with auto-reset (torch-free, so it can be tested anywhere)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from env import SwarmSearchEnv
from evaluate import episode_metrics
from hier import HPlanner, MOVES
from rewards import RewardConfig, RewardShaper
from stages import TRAIN_SEED_BASE, cfg_from_dict


class HEnv:
    def __init__(self, cfg_dict: dict, rank: int, base_seed: int, rcfg: RewardConfig,
                 map_pool: Optional[int] = None, memory: bool = True):
        self.cfg = cfg_from_dict(cfg_dict)
        self.env = SwarmSearchEnv(self.cfg)
        self.N = self.cfg.n_agents
        self.rng = np.random.default_rng(base_seed * 1009 + rank)
        self.pl = HPlanner(self.cfg, seed=base_seed * 7 + rank, memory=memory)
        self.shaper = RewardShaper(rcfg, self.N, self.cfg.height, self.cfg.width, self.cfg.view_radius,
                                   self.cfg.v_high)
        self.map_pool = map_pool

    def _seed(self) -> int:
        hi = self.map_pool if self.map_pool else 10 ** 8
        return TRAIN_SEED_BASE + int(self.rng.integers(0, hi))

    def reset(self):
        obs, _ = self.env.reset(seed=self._seed())
        self.pl.reset()
        self.pl.perceive(obs)
        self.shaper.reset(self.env)

    def step(self, scale: float):
        """Execute the planner's primitive actions. Returns (reward[N], done, episode_metrics|None)."""
        env, N = self.env, self.N
        actions = self.pl.actions()
        acts = [actions[n] for n in self.pl.names]
        prev_pos, n_log = list(env.pos), len(env.stats["delivery_log"])
        obs, rew, te, tr, infos = env.step(actions)
        a0 = self.pl.names[0]
        comp = infos[a0]["reward_components"]
        done = bool(te[a0] or tr[a0])
        collided = np.array([(acts[i] in MOVES) and env.pos[i] == prev_pos[i] for i in range(N)])
        owners = [it.discovered_by for it in env.objects.values()
                  if it.discovered_t == env.t and it.type != 2 and it.discovered_by is not None]
        picked = [i for i in range(N) if env.carry[i] is not None and env.carry[i].pickup_t == env.t - 1]
        r = self.shaper.step(env, float(rew[a0]), comp, env.stats["delivery_log"][n_log:], collided,
                             owners, picked, done, scale)
        ep = None
        if done:
            ep = episode_metrics(env)
            self.reset()
        else:
            self.pl.perceive(obs)
        return r, done, ep