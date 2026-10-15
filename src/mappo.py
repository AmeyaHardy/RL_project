"""
MAPPO (multi-agent PPO, centralized training / decentralized execution) for SwarmSearchEnv.

  * Shared decentralised Actor: uses local 7x7 window, team map + reconstructed checkpoint +
    separated self/teammate channels, scalar features, and action-mask sanitization.
  * Spatial Agent-Conditioned Centralised Critic V(s, o_i): 2D CNN over the 7 ground-truth state
    grids + global state scalars + per-agent scalar observations -> per-agent value [E, N].
  * Per-agent credit assignment & potential-based BFS reward shaping in EnvRunner.
  * Compatible with all unit tests in test_mappo.py and cross-stage transfer in train.py.
"""
from __future__ import annotations

import multiprocessing as mp
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from env import (SwarmSearchEnv, N_ACTIONS, N_LOCAL_CH, N_TEAM_CH, N_SCALARS,
                 PERSIST, TEMP, MAL, PICK, DROP)
from evaluate import episode_metrics
from stages import TRAIN_SEED_BASE, cfg_from_dict


# =============================================================== environments
def pack(env: SwarmSearchEnv, obs: dict, infos: dict):
    names = env.possible_agents
    return dict(
        local=np.stack([obs[a]["local"] for a in names]).astype(np.float32),
        team=np.stack([obs[a]["team_map"] for a in names]).astype(np.float32),
        scal=np.stack([obs[a]["scalars"] for a in names]).astype(np.float32),
        mask=np.stack([infos[a]["action_mask"] for a in names]).astype(bool),
        state=env.state().astype(np.float32),
    )


class EnvRunner:
    """One environment with auto-reset, per-agent credit assignment, and BFS potential shaping."""

    def __init__(self, cfg_dict: dict, rank: int, base_seed: int, map_pool: Optional[int] = None):
        self.cfg = cfg_from_dict(cfg_dict)
        self.env = SwarmSearchEnv(self.cfg)
        self.names = self.env.possible_agents
        self.rng = np.random.default_rng(base_seed * 1009 + rank)
        self.map_pool = map_pool
        self.prev_pot = np.zeros(self.cfg.n_agents, dtype=np.float32)

    def _seed(self) -> int:
        hi = self.map_pool if self.map_pool else 10 ** 8
        return TRAIN_SEED_BASE + int(self.rng.integers(0, hi))

    def _potentials(self) -> np.ndarray:
        """Compute per-agent potential Phi(s_i) using true BFS checkpoint distances."""
        env, cfg = self.env, self.cfg
        max_d = float(cfg.height + cfg.width)
        pots = np.zeros(cfg.n_agents, dtype=np.float32)
        vis_items = [it for it in env.active.values() if it.seen and it.type != MAL]
        for i in range(cfg.n_agents):
            y, x = env.pos[i]
            carried = env.carry[i]
            if carried is not None and carried.type != MAL:
                d_cp = float(env.checkpoint_dist[y, x])
                val = cfg.v_high if carried.type == TEMP else cfg.v_low
                pots[i] = 0.4 * val * (1.0 + (1.0 - d_cp / max_d))
            elif vis_items:
                d_min = min(abs(it.pos[0] - y) + abs(it.pos[1] - x) for it in vis_items)
                pots[i] = 0.25 * (1.0 - d_min / max_d)
        return pots

    def reset(self):
        obs, infos = self.env.reset(seed=self._seed())
        self.prev_pot = self._potentials()
        return pack(self.env, obs, infos)

    def step(self, actions: np.ndarray):
        env, cfg = self.env, self.cfg
        N = cfg.n_agents
        prev_pos = list(env.pos)
        prev_carry = [it.type if it is not None else -1 for it in env.carry]
        prev_deliv_len = len(env.stats["delivery_log"])

        obs, rew, te, tr, infos = env.step({a: int(actions[i]) for i, a in enumerate(self.names)})
        a0 = self.names[0]
        comp = infos[a0]["reward_components"]

        # 1. Shared exploration & time reward split evenly across agents
        shared_r = (comp["time"] + comp["map"] + comp["discover"] + comp["coordination"]) / N
        agent_r = np.full(N, shared_r, dtype=np.float32)

        # 2. Individual credit for deliveries via delivery_log
        for d_entry in env.stats["delivery_log"][prev_deliv_len:]:
            agent_r[d_entry["agent"]] += float(d_entry["value"])

        # 3. Individual credit for pickups, collisions, and carried expirations
        for i in range(N):
            act_i = int(actions[i])
            if act_i < 4 and env.pos[i] == prev_pos[i]:
                agent_r[i] -= cfg.c_collision
            if act_i == PICK and prev_carry[i] == -1:
                if env.carry[i] is not None:
                    if env.carry[i].type in (PERSIST, TEMP):
                        val = cfg.v_high if env.carry[i].type == TEMP else cfg.v_low
                        agent_r[i] += 0.2 * val
                    elif env.carry[i].type == MAL:
                        agent_r[i] -= cfg.p_mal
                elif comp["delivery"] < 0:
                    # Malicious item destroyed on pickup
                    agent_r[i] -= cfg.p_mal

        # 4. Potential-based shaping (gamma * Phi(s') - Phi(s))
        term, trunc = bool(te[a0]), bool(tr[a0])
        new_pot = np.zeros(N, dtype=np.float32) if term else self._potentials()
        agent_r += 0.99 * new_pot - self.prev_pot
        self.prev_pot = new_pot

        out = dict(
            reward=agent_r,
            term=term,
            trunc=trunc,
            final_state=None,
            final_scal=None,
            ep=None,
        )
        if term or trunc:
            out["final_state"] = env.state().astype(np.float32)
            out["final_scal"] = np.stack([obs[a]["scalars"] for a in self.names]).astype(np.float32)
            out["ep"] = episode_metrics(env)
            obs, infos = env.reset(seed=self._seed())
            self.prev_pot = self._potentials()
        out.update(pack(env, obs, infos))
        return out


def _worker(conn, cfg_dict, rank, base_seed, map_pool):
    runner = EnvRunner(cfg_dict, rank, base_seed, map_pool)
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            conn.send(runner.reset())
        elif cmd == "step":
            conn.send(runner.step(data))
        elif cmd == "close":
            conn.close()
            break


def _stack(items: List[dict]) -> dict:
    return {k: np.stack([it[k] for it in items]) for k in ("local", "team", "scal", "mask", "state")}


class VecEnv:
    """workers=0: in-process sequential. workers>0: one subprocess per env (spawn-safe)."""

    def __init__(self, cfg_dict: dict, n_envs: int, base_seed: int = 0, workers: int = 0,
                 map_pool: Optional[int] = None):
        self.n_envs, self.workers = n_envs, workers
        if workers:
            ctx = mp.get_context("spawn")
            self.conns, self.procs = [], []
            for r in range(n_envs):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, cfg_dict, r, base_seed, map_pool), daemon=True)
                p.start()
                self.conns.append(parent)
                self.procs.append(p)
        else:
            self.runners = [EnvRunner(cfg_dict, r, base_seed, map_pool) for r in range(n_envs)]

    def reset(self) -> dict:
        if self.workers:
            for c in self.conns:
                c.send(("reset", None))
            return _stack([c.recv() for c in self.conns])
        return _stack([r.reset() for r in self.runners])

    def step(self, actions: np.ndarray):
        if self.workers:
            for c, a in zip(self.conns, actions):
                c.send(("step", a))
            outs = [c.recv() for c in self.conns]
        else:
            outs = [r.step(a) for r, a in zip(self.runners, actions)]
        batch = _stack(outs)
        extra = dict(
            reward=np.stack([o["reward"] for o in outs]).astype(np.float32),
            term=np.array([o["term"] for o in outs]),
            trunc=np.array([o["trunc"] for o in outs]),
            final_state=[o["final_state"] for o in outs],
            final_scal=[o["final_scal"] for o in outs],
            ep=[o["ep"] for o in outs],
        )
        return batch, extra

    def close(self):
        if self.workers:
            for c in self.conns:
                try:
                    c.send(("close", None))
                except Exception:
                    pass


# =============================================================== networks
def _init(m, gain=np.sqrt(2)):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)
    return m


class Actor(nn.Module):
    """Decentralised policy. Input: robot's own observation. Output: masked action logits."""

    def __init__(self, H: int, W: int, view_k: int = 7, hidden: int = 128, recurrent: bool = False):
        super().__init__()
        self.H, self.W, self.hidden, self.recurrent = H, W, hidden, recurrent
        self.local = nn.Sequential(
            _init(nn.Conv2d(N_LOCAL_CH, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 32, 3, padding=1)), nn.ReLU(), nn.Flatten(),
            _init(nn.Linear(32 * view_k * view_k, hidden)), nn.ReLU()
        )
        # team map (7) + [self-pos (1)] + [checkpoint (1)] + [y, x coords (2)] = 11 channels
        # AdaptiveMaxPool2d(6) preserves sharp point features (items/depot) AND keeps shape
        # identical across 12x12, 16x16, and 20x20 stages for 100% weight transfer!
        self.team = nn.Sequential(
            _init(nn.Conv2d(N_TEAM_CH + 4, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            nn.AdaptiveMaxPool2d((6, 6)),
            _init(nn.Conv2d(64, 32, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(32 * 36, hidden)), nn.ReLU()
        )
        self.scal = nn.Sequential(
            _init(nn.Linear(N_SCALARS, 64)), nn.ReLU(),
            _init(nn.Linear(64, 64)), nn.ReLU()
        )
        self.trunk = nn.Sequential(
            _init(nn.Linear(2 * hidden + 64, hidden)), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.rnn = nn.GRUCell(hidden, hidden) if recurrent else None
        self.pi = _init(nn.Linear(hidden, N_ACTIONS), 0.01)
        yy = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        xx = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
        self.register_buffer("coords", torch.cat([yy, xx], 1))

    def sanitize_mask(self, local: torch.Tensor, scal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forbid picking up malicious items or doomed temporary items using local/scalar obs."""
        clean = mask.clone()
        r = local.shape[-1] // 2
        has_persist = local[:, 1, r, r] > 0.0
        has_temp = local[:, 2, r, r] > 0.0
        has_mal = local[:, 3, r, r] > 0.0

        # 1. Block PICK on malicious items
        mal_only = has_mal & ~has_persist & ~has_temp
        clean[:, PICK] &= ~mal_only

        # 2. Block PICK on temporary items that cannot survive the BFS path to checkpoint
        rem_steps = local[:, 2, r, r] * 80.0
        min_dist_to_cp = scal[:, 22] * (self.H + self.W)
        doomed_temp = has_temp & ~has_persist & (rem_steps < (min_dist_to_cp + 1.0))
        clean[:, PICK] &= ~doomed_temp

        # Safety fallback: if a synthetic test masks everything except PICK, keep original mask
        empty = ~clean.any(dim=-1, keepdim=True)
        return torch.where(empty, mask, clean)

    def encode(self, local, team, scal):
        B = local.shape[0]
        idx = torch.arange(B, device=local.device)
        y = (scal[:, 0] * self.H).round().long().clamp(0, self.H - 1)
        x = (scal[:, 1] * self.W).round().long().clamp(0, self.W - 1)
        cy = ((scal[:, 0] + scal[:, 6]) * self.H).round().long().clamp(0, self.H - 1)
        cx = ((scal[:, 1] + scal[:, 7]) * self.W).round().long().clamp(0, self.W - 1)

        me = torch.zeros(B, 1, self.H, self.W, device=local.device)
        cp = torch.zeros(B, 1, self.H, self.W, device=local.device)
        me[idx, 0, y, x] = 1.0
        cp[idx, 0, cy, cx] = 1.0

        team_clean = team.clone()
        team_clean[:, 5:6] = (team_clean[:, 5:6] - me).clamp(0.0, 1.0)
        team_in = torch.cat([team_clean, me, cp, self.coords.expand(B, -1, -1, -1)], 1)
        z = torch.cat([self.local(local), self.team(team_in), self.scal(scal)], 1)
        return self.trunk(z)

    def init_hidden(self, B: int, device=None):
        return torch.zeros(B, self.hidden, device=device or self.coords.device)

    def forward_step(self, local, team, scal, mask, h=None):
        """One timestep. h is the hidden state AFTER episode-start resets were applied."""
        clean_mask = self.sanitize_mask(local, scal, mask)
        x = self.encode(local, team, scal)
        if self.rnn is not None:
            h = self.rnn(x, h)
            x = h
        return self.pi(x).masked_fill(~clean_mask, -1e10), h

    def forward_seq(self, local, team, scal, mask, h0=None, reset=None):
        """Sequence version. Inputs [L, B, ...]; reset [L, B] bool zeroes h before step t."""
        L, B = local.shape[:2]
        flat_local, flat_team, flat_scal = local.flatten(0, 1), team.flatten(0, 1), scal.flatten(0, 1)
        clean_mask = self.sanitize_mask(flat_local, flat_scal, mask.flatten(0, 1)).view(L, B, -1)
        x = self.encode(flat_local, flat_team, flat_scal).view(L, B, -1)
        if self.rnn is not None:
            h, outs = h0, []
            for t in range(L):
                if reset is not None:
                    h = h * (~reset[t]).float().unsqueeze(-1)
                h = self.rnn(x[t], h)
                outs.append(h)
            x = torch.stack(outs)
        return self.pi(x).masked_fill(~clean_mask, -1e10)


class Critic(nn.Module):
    """Spatial Agent-Conditioned Centralised Critic V(s, o_i) -> [B, N]."""

    def __init__(self, state_dim: int, H: int = 20, W: int = 20, n_agents: int = 4, hidden: int = 128):
        super().__init__()
        self.H, self.W, self.N = H, W, n_agents
        self.grid_dim = 7 * H * W
        self.tail_dim = state_dim - self.grid_dim
        self.conv = nn.Sequential(
            _init(nn.Conv2d(7, 32, 3, padding=1)), nn.ReLU(),
            nn.AdaptiveMaxPool2d((6, 6)),
            _init(nn.Conv2d(32, 32, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(32 * 36, hidden)), nn.ReLU()
        )
        self.head = nn.Sequential(
            _init(nn.Linear(hidden + self.tail_dim + N_SCALARS, hidden)), nn.ReLU(), nn.LayerNorm(hidden),
            _init(nn.Linear(hidden, hidden)), nn.ReLU(), nn.LayerNorm(hidden),
            _init(nn.Linear(hidden, 1), 1.0)
        )

    def forward(self, s: torch.Tensor, scal: Optional[torch.Tensor] = None):
        orig_shape = s.shape[:-1]
        s_flat = s.reshape(-1, s.shape[-1])
        B = s_flat.shape[0]
        grids = s_flat[:, :self.grid_dim].view(B, 7, self.H, self.W)
        tail = s_flat[:, self.grid_dim:]
        g_emb = torch.cat([self.conv(grids), tail], dim=-1)
        g_exp = g_emb.unsqueeze(1).expand(B, self.N, -1)
        if scal is None:
            scal_flat = torch.zeros(B, self.N, N_SCALARS, device=s.device, dtype=s.dtype)
        else:
            scal_flat = scal.reshape(B, self.N, N_SCALARS)
        v = self.head(torch.cat([g_exp, scal_flat], dim=-1)).squeeze(-1)  # [B, N]
        return v.view(*orig_shape, self.N)


class ValueNorm:
    """Running mean/std of return targets; critic learns normalised values."""

    def __init__(self, device="cpu"):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: torch.Tensor):
        x = x.detach().flatten()
        bm, bv, bc = x.mean().item(), x.var(unbiased=False).item(), x.numel()
        d, tot = bm - self.mean, self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / (self.var ** 0.5 + 1e-8)

    def denorm(self, x):
        return x * (self.var ** 0.5 + 1e-8) + self.mean


# =============================================================== algorithm
@dataclass
class PPOConfig:
    n_envs: int = 8
    rollout: int = 512           # steps per env per update (must be a multiple of chunk)
    chunk: int = 1                # BPTT chunk length; use ~16 with recurrent=True
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    lr_final_frac: float = 0.1
    ent_coef: float = 0.015
    ent_coef_final: float = 0.003
    max_grad: float = 10.0
    target_kl: float = 0.03
    hidden: int = 128
    recurrent: bool = False
    value_norm: bool = True


def compute_gae(rewards, values, dones, boot, last_value, gamma, lam):
    """Supports both [T, E] (unit tests) and [T, E, N] (per-agent GAE)."""
    T = rewards.shape[0]
    if dones.dim() < rewards.dim():
        dones = dones.unsqueeze(-1)
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + boot[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


class MAPPO:
    def __init__(self, env_cfg_dict: dict, ppo: PPOConfig, workers: int = 0, seed: int = 0,
                 device: str = "cpu", map_pool: Optional[int] = None):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cfg_dict, self.ppo, self.device = env_cfg_dict, ppo, torch.device(device)
        assert ppo.rollout % ppo.chunk == 0, "rollout must be a multiple of chunk"
        self.venv = VecEnv(env_cfg_dict, ppo.n_envs, seed, workers, map_pool)
        cfg = cfg_from_dict(env_cfg_dict)
        self.N, self.E = cfg.n_agents, ppo.n_envs
        self.net_cfg = dict(H=cfg.height, W=cfg.width, view_k=2 * cfg.view_radius + 1,
                            hidden=ppo.hidden, recurrent=ppo.recurrent)
        probe = EnvRunner(env_cfg_dict, 0, seed).reset()
        self.actor = Actor(**self.net_cfg).to(self.device)
        self.critic = Critic(probe["state"].shape[0], H=cfg.height, W=cfg.width,
                             n_agents=self.N, hidden=ppo.hidden).to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=ppo.lr_actor, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=ppo.lr_critic, eps=1e-5)
        self.vnorm = ValueNorm()
        self.obs = self.venv.reset()
        self.h = torch.zeros(self.E, self.N, ppo.hidden, device=self.device)
        self.first = torch.ones(self.E, dtype=torch.bool, device=self.device)
        self.recent: deque = deque(maxlen=100)
        self.env_steps = 0

    # ---------------------------------------------------------------- helpers
    def _t(self, x, dtype=None):
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def _flat(self, b):  # [E,N,...] -> [E*N,...]
        return (self._t(b["local"]).flatten(0, 1), self._t(b["team"]).flatten(0, 1),
                self._t(b["scal"]).flatten(0, 1), self._t(b["mask"]).flatten(0, 1))

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect(self) -> dict:
        p, T, E, N = self.ppo, self.ppo.rollout, self.E, self.N
        buf = {k: [] for k in ("local", "team", "scal", "mask", "state", "act", "logp", "value",
                               "reward", "done", "boot", "h", "first")}
        for _ in range(T):
            b = self.obs
            local, team, scal, mask = self._flat(b)
            state = self._t(b["state"])
            scal_en = scal.view(E, N, -1)
            v_raw = self.critic(state, scal_en)                                  # [E, N]
            value = self.vnorm.denorm(v_raw) if p.value_norm else v_raw
            h_in = self.h * (~self.first).float().view(E, 1, 1)                  # zero hidden at episode start
            logits, h_new = self.actor.forward_step(local, team, scal, mask,
                                                    h_in.view(E * N, -1) if p.recurrent else None)
            dist = Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)
            batch, ex = self.venv.step(act.view(E, N).cpu().numpy())

            done = ex["term"] | ex["trunc"]
            boot = np.zeros((E, N), np.float32)
            for e in range(E):
                if ex["trunc"][e] and not ex["term"][e]:                         # time limit -> bootstrap
                    fs = self._t(ex["final_state"][e]).unsqueeze(0)
                    fsc = self._t(ex["final_scal"][e]).unsqueeze(0)
                    v = self.critic(fs, fsc)
                    boot[e] = p.gamma * (self.vnorm.denorm(v) if p.value_norm else v).squeeze(0).cpu().numpy()
                if ex["ep"][e] is not None:
                    self.recent.append(ex["ep"][e])
            for k, v in (("local", local.view(E, N, *local.shape[1:])),
                         ("team", team.view(E, N, *team.shape[1:])),
                         ("scal", scal_en),
                         ("mask", mask.view(E, N, -1)),
                         ("state", state),
                         ("act", act.view(E, N)),
                         ("logp", logp.view(E, N)),
                         ("value", value),
                         ("reward", self._t(ex["reward"])),
                         ("done", self._t(done, torch.float32)),
                         ("boot", self._t(boot)),
                         ("h", h_in),
                         ("first", self.first.clone())):
                buf[k].append(v)
            if p.recurrent:
                self.h = h_new.view(E, N, -1)
            self.first = self._t(done)
            self.obs = batch
            self.env_steps += E
        out = {k: torch.stack(v) for k, v in buf.items()}
        last_state = self._t(self.obs["state"])
        last_scal = self._t(self.obs["scal"])
        lv = self.critic(last_state, last_scal)
        out["last_value"] = self.vnorm.denorm(lv) if p.value_norm else lv
        return out

    # ---------------------------------------------------------------- update
    def update(self, buf: dict, ent_coef: float, lr_frac: float) -> dict:
        p, N, E, L = self.ppo, self.N, self.E, self.ppo.chunk
        T = p.rollout
        adv, ret = compute_gae(buf["reward"], buf["value"], buf["done"], buf["boot"],
                               buf["last_value"], p.gamma, p.lam)                # [T, E, N]
        if p.value_norm:
            self.vnorm.update(ret)
        ret_n = self.vnorm.norm(ret) if p.value_norm else ret
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        ev = 1.0 - ((ret - buf["value"]).var() / (ret.var() + 1e-8)).item()

        for opt, lr0 in ((self.opt_a, p.lr_actor), (self.opt_c, p.lr_critic)):
            for g in opt.param_groups:
                g["lr"] = lr0 * lr_frac

        C = T // L
        cs, es = np.repeat(np.arange(C), E), np.tile(np.arange(E), C)
        U = C * E
        stats = dict(pl=[], vl=[], ent=[], kl=[], clipfrac=[])
        for epoch in range(p.epochs):
            perm = self.rng.permutation(U)
            ep_kl = []
            for mb in np.array_split(perm, p.minibatches):
                if len(mb) == 0:
                    continue
                ts = self._t(cs[mb][:, None] * L + np.arange(L)[None, :], torch.long)   # [B, L]
                ee = self._t(es[mb][:, None], torch.long)                               # [B, 1]
                B = len(mb)

                def seq(x):  # [B, L, N, ...] -> [L, B*N, ...]
                    return x.permute(1, 0, 2, *range(3, x.dim())).reshape(L, B * N, *x.shape[3:])

                local, team, scal, mask = (seq(buf[k][ts, ee]) for k in ("local", "team", "scal", "mask"))
                act, old_logp = buf["act"][ts, ee], buf["logp"][ts, ee]                  # [B, L, N]
                h0 = buf["h"][ts[:, 0], ee[:, 0]].reshape(B * N, -1)
                reset = buf["first"][ts, ee].t().unsqueeze(-1).expand(L, B, N).reshape(L, B * N)
                logits = self.actor.forward_seq(local, team, scal, mask, h0 if p.recurrent else None, reset)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act.permute(1, 0, 2).reshape(L, B * N))
                logp = logp.view(L, B, N).permute(1, 0, 2)
                log_ratio = logp - old_logp
                ratio = torch.exp(log_ratio)
                a = adv_n[ts, ee]                                                        # [B, L, N]
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - p.clip, 1 + p.clip) * a).mean()
                ent = dist.entropy().mean()
                self.opt_a.zero_grad()
                (pl - ent_coef * ent).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), p.max_grad)
                self.opt_a.step()

                v = self.critic(buf["state"][ts, ee], buf["scal"][ts, ee])               # [B, L, N]
                vl = F.smooth_l1_loss(v, ret_n[ts, ee])
                self.opt_c.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), p.max_grad)
                self.opt_c.step()

                with torch.no_grad():
                    kl = ((ratio - 1.0) - log_ratio).mean().item()
                    stats["pl"].append(pl.item())
                    stats["vl"].append(vl.item())
                    stats["ent"].append(ent.item())
                    stats["kl"].append(kl)
                    ep_kl.append(kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > p.clip).float().mean().item())
            if np.mean(ep_kl) > 1.5 * p.target_kl:
                break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["explained_var"] = ev
        out["epochs_run"] = epoch + 1
        return out

    # ---------------------------------------------------------------- io
    def state_dict(self) -> dict:
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    net=self.net_cfg, env_cfg=self.cfg_dict, ppo=self.ppo.__dict__,
                    vnorm=(self.vnorm.mean, self.vnorm.var, self.vnorm.count), env_steps=self.env_steps)

    def close(self):
        self.venv.close()


# =============================================================== inference policy
class MappoPolicy:
    """Evaluation-time policy (decentralised: own observation + action mask only)."""

    def __init__(self, actor: Actor, deterministic: bool = True, device: str = "cpu"):
        self.actor, self.deterministic, self.device = actor.to(device).eval(), deterministic, torch.device(device)
        self.h = None
        self.gen = torch.Generator().manual_seed(0)

    @classmethod
    def from_checkpoint(cls, path: str, deterministic: bool = True, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        actor = Actor(**ck["net"])
        actor.load_state_dict(ck["actor"])
        return cls(actor, deterministic, device)

    def reset(self, seed: int):
        self.h = None
        self.gen = torch.Generator().manual_seed(int(seed))

    @torch.no_grad()
    def act(self, obs: dict, infos: dict) -> Dict[str, int]:
        names = list(obs.keys())
        t = lambda xs, dt=torch.float32: torch.as_tensor(np.stack(xs), dtype=dt, device=self.device)
        local = t([obs[a]["local"] for a in names])
        team = t([obs[a]["team_map"] for a in names])
        scal = t([obs[a]["scalars"] for a in names])
        mask = t([infos[a]["action_mask"] for a in names], torch.bool)
        if self.h is None:
            self.h = self.actor.init_hidden(len(names), self.device)
        logits, self.h = self.actor.forward_step(local, team, scal, mask,
                                                 self.h if self.actor.rnn is not None else None)
        if self.deterministic:
            act = logits.argmax(-1)
        else:
            probs = F.softmax(logits, -1).cpu()
            act = torch.multinomial(probs, 1, generator=self.gen).squeeze(-1)
        return {a: int(act[i]) for i, a in enumerate(names)}