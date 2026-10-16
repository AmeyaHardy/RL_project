"""
Hierarchical MAPPO (goal-level) for SwarmSearchEnv.

  Actor  : permutation-equivariant pointer network over candidate goals {stay, items(<=10), regions(<=16)};
           one self-attention layer lets every candidate see the others (relative comparisons), parameters
           are independent of map size / #robots -> curriculum transfer is 100%.
  Critic : centralised V(s, i) on compact global features (all robots' goals/carry state, claims, coverage...).
  Action : robots decide SEQUENTIALLY in a random order each time they need a goal; each robot's candidate
           features include the earlier robots' claims -> duplicate targets are impossible, and the joint
           log-prob factorises as the product of the per-robot log-probs that PPO already uses.
  PPO    : per-step reward + GAE over all env steps; the actor loss is taken only on decision steps
           (between decisions the low-level executor acts, the policy gradient there is zero).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from henv import HEnv
from hier import FC, FI, FR, KI, KR, HPlanner, critic_dim
from rewards import RewardConfig
from stages import cfg_from_dict


# ======================================================================== networks
def mlp(i: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())


class HActor(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.f_ctx, self.f_item, self.f_reg = mlp(FC, hidden), mlp(FI, hidden), mlp(FR, hidden)
        self.stay = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_emb = nn.Parameter(0.02 * torch.randn(3, hidden))
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.score = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ctx, items, imask, regs, rmask):
        """ctx [B,FC], items [B,KI,FI], imask [B,KI] bool, regs [B,KR,FR], rmask [B,KR] bool -> logits [B,1+KI+KR]."""
        B = ctx.shape[0]
        c = self.f_ctx(ctx).unsqueeze(1)                                           # [B,1,h]
        tok = torch.cat([self.stay.expand(B, 1, -1) + self.type_emb[2],
                         self.f_item(items) + self.type_emb[0],
                         self.f_reg(regs) + self.type_emb[1]], 1) + c              # [B,27,h]
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=ctx.device), imask, rmask], 1)
        a, _ = self.attn(tok, tok, tok, key_padding_mask=~valid, need_weights=False)
        tok = self.norm(tok + a)
        z = torch.cat([tok, c.expand_as(tok), tok * c], -1)
        return self.score(z).squeeze(-1).masked_fill(~valid, -1e9)               # "stay" is always valid


class HCritic(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ValueNorm:
    def __init__(self):
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


def cands_to_tensors(cands, device):
    f = lambda k, dt=torch.float32: torch.as_tensor(np.stack([c[k] for c in cands]), dtype=dt, device=device)
    return f("ctx"), f("items"), f("item_mask", torch.bool), f("regs"), f("reg_mask", torch.bool)


# ======================================================================== PPO
@dataclass
class HPPOConfig:
    n_envs: int = 8
    rollout: int = 512            # env steps per env per update
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_final_frac: float = 0.1
    ent_coef: float = 0.02
    ent_coef_final: float = 0.003
    max_grad: float = 5.0
    target_kl: float = 0.03
    hidden: int = 128


def compute_gae(rew, val, done, last_val, gamma, lam):
    """rew,val [T,E,N]; done [T,E]; last_val [E,N]. Episodes end at a fixed horizon and the critic sees t/T,
    so a time-limit end is treated as terminal (finite-horizon objective)."""
    T = rew.shape[0]
    adv, last = torch.zeros_like(rew), torch.zeros_like(last_val)
    nd = 1.0 - done.unsqueeze(-1)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * nv * nd[t] - val[t]
        last = delta + gamma * lam * nd[t] * last
        adv[t] = last
    return adv, adv + val


class HMAPPO:
    def __init__(self, env_cfg_dict: dict, ppo: HPPOConfig, rcfg: RewardConfig, seed: int = 0,
                 device: str = "cpu", map_pool: Optional[int] = None):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cfg_dict, self.ppo, self.rcfg, self.device = env_cfg_dict, ppo, rcfg, torch.device(device)
        cfg = cfg_from_dict(env_cfg_dict)
        self.N, self.E = cfg.n_agents, ppo.n_envs
        self.envs = [HEnv(env_cfg_dict, r, seed, rcfg, map_pool) for r in range(self.E)]
        for h in self.envs:
            h.reset()
        self.net_cfg = dict(hidden=ppo.hidden)
        self.actor = HActor(**self.net_cfg).to(self.device)
        self.critic = HCritic(critic_dim(self.N), ppo.hidden).to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=ppo.lr_actor, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=ppo.lr_critic, eps=1e-5)
        self.vnorm = ValueNorm()
        self.recent: deque = deque(maxlen=100)
        self.env_steps = 0

    def _t(self, x, dt=torch.float32):
        return torch.as_tensor(x, dtype=dt, device=self.device)

    def _crit_feats(self) -> np.ndarray:
        return np.stack([np.stack([h.pl.critic_features(i) for i in range(self.N)]) for h in self.envs]).astype(np.float32)

    # -------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect(self, shape_scale: float) -> dict:
        T, E, N = self.ppo.rollout, self.E, self.N
        cd = critic_dim(N)
        B = dict(ctx=np.zeros((T, E, N, FC), np.float32), items=np.zeros((T, E, N, KI, FI), np.float32),
                 imask=np.zeros((T, E, N, KI), bool), regs=np.zeros((T, E, N, KR, FR), np.float32),
                 rmask=np.zeros((T, E, N, KR), bool), act=np.zeros((T, E, N), np.int64),
                 logp=np.zeros((T, E, N), np.float32), dec=np.zeros((T, E, N), bool),
                 cf=np.zeros((T, E, N, cd), np.float32), rew=np.zeros((T, E, N), np.float32),
                 done=np.zeros((T, E), np.float32), val=np.zeros((T, E, N), np.float32))
        for t in range(T):
            cf = self._crit_feats()
            B["cf"][t] = cf
            v = self.critic(self._t(cf))
            B["val"][t] = self.vnorm.denorm(v).cpu().numpy()
            orders = [h.pl.need_decision() for h in self.envs]
            for r in range(max((len(o) for o in orders), default=0)):
                act_envs = [e for e, o in enumerate(orders) if len(o) > r]
                cands = [self.envs[e].pl.candidates(orders[e][r]) for e in act_envs]
                dist = Categorical(logits=self.actor(*cands_to_tensors(cands, self.device)))
                a = dist.sample()
                lp = dist.log_prob(a).cpu().numpy()
                for k, e in enumerate(act_envs):
                    i, c = orders[e][r], cands[k]
                    self.envs[e].pl.apply_choice(i, int(a[k]), c)
                    B["ctx"][t, e, i], B["items"][t, e, i], B["imask"][t, e, i] = c["ctx"], c["items"], c["item_mask"]
                    B["regs"][t, e, i], B["rmask"][t, e, i] = c["regs"], c["reg_mask"]
                    B["act"][t, e, i], B["logp"][t, e, i], B["dec"][t, e, i] = int(a[k]), lp[k], True
            for e, h in enumerate(self.envs):
                r, done, ep = h.step(shape_scale)
                B["rew"][t, e], B["done"][t, e] = r, float(done)
                if ep is not None:
                    self.recent.append(ep)
            self.env_steps += E
        out = {k: torch.as_tensor(v, device=self.device) for k, v in B.items()}
        out["last_val"] = self.vnorm.denorm(self.critic(self._t(self._crit_feats())))
        return out

    # -------------------------------------------------------------- update
    def update(self, buf: dict, ent_coef: float, lr_frac: float) -> dict:
        p = self.ppo
        adv, ret = compute_gae(buf["rew"], buf["val"], buf["done"], buf["last_val"], p.gamma, p.lam)
        ev = 1.0 - ((ret - buf["val"]).var() / (ret.var() + 1e-8)).item()
        self.vnorm.update(ret)
        ret_n = self.vnorm.norm(ret)
        for opt, lr0 in ((self.opt_a, p.lr_actor), (self.opt_c, p.lr_critic)):
            for g in opt.param_groups:
                g["lr"] = lr0 * lr_frac

        dec = buf["dec"]
        M = int(dec.sum())
        ctx, items, imask, regs, rmask = (buf[k][dec] for k in ("ctx", "items", "imask", "regs", "rmask"))
        act, old_lp, a_dec = buf["act"][dec], buf["logp"][dec], adv[dec]
        if M > 1:
            a_dec = (a_dec - a_dec.mean()) / (a_dec.std() + 1e-8)
        cf, ret_flat = buf["cf"].reshape(-1, buf["cf"].shape[-1]), ret_n.reshape(-1)
        S = cf.shape[0]
        st = dict(pl=[], vl=[], ent=[], kl=[], clipfrac=[])
        ran = 0
        for epoch in range(p.epochs):
            ep_kl = []
            if M >= 16:
                for mb in np.array_split(self.rng.permutation(M), p.minibatches):
                    mb = torch.as_tensor(mb, device=self.device)
                    dist = Categorical(logits=self.actor(ctx[mb], items[mb], imask[mb], regs[mb], rmask[mb]))
                    lp = dist.log_prob(act[mb])
                    lr_ = lp - old_lp[mb]
                    ratio = lr_.exp()
                    a = a_dec[mb]
                    pl = -torch.min(ratio * a, ratio.clamp(1 - p.clip, 1 + p.clip) * a).mean()
                    ent = dist.entropy().mean()
                    self.opt_a.zero_grad()
                    (pl - ent_coef * ent).backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), p.max_grad)
                    self.opt_a.step()
                    with torch.no_grad():
                        kl = ((ratio - 1.0) - lr_).mean().item()
                        ep_kl.append(kl)
                        st["pl"].append(pl.item()); st["ent"].append(ent.item()); st["kl"].append(kl)
                        st["clipfrac"].append(((ratio - 1).abs() > p.clip).float().mean().item())
            for mb in np.array_split(self.rng.permutation(S), p.minibatches * 2):
                mb = torch.as_tensor(mb, device=self.device)
                vl = F.smooth_l1_loss(self.critic(cf[mb]), ret_flat[mb])
                self.opt_c.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), p.max_grad)
                self.opt_c.step()
                st["vl"].append(vl.item())
            ran = epoch + 1
            if ep_kl and np.mean(ep_kl) > 1.5 * p.target_kl:
                break
        out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in st.items()}
        out.update(explained_var=ev, epochs_run=ran, n_decisions=M)
        return out

    def state_dict(self) -> dict:
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), net=self.net_cfg,
                    env_cfg=self.cfg_dict, ppo=self.ppo.__dict__, rcfg=self.rcfg.__dict__,
                    vnorm=(self.vnorm.mean, self.vnorm.var, self.vnorm.count), env_steps=self.env_steps)


# ======================================================================== evaluation-time policy
class HMappoPolicy:
    """evaluate.py-compatible: policy.reset(seed); policy.act(obs, infos) -> {agent: action}."""

    def __init__(self, actor: HActor, cfg, deterministic: bool = True, device: str = "cpu"):
        self.actor, self.cfg, self.det, self.device = actor.to(device).eval(), cfg, deterministic, torch.device(device)
        self.pl = HPlanner(cfg)
        self.gen = torch.Generator().manual_seed(0)

    @classmethod
    def from_checkpoint(cls, path: str, cfg=None, deterministic: bool = True, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        actor = HActor(**ck["net"])
        actor.load_state_dict(ck["actor"])
        return cls(actor, cfg or cfg_from_dict(ck["env_cfg"]), deterministic, device)

    def reset(self, seed: int):
        self.pl.reset(int(seed))
        self.gen.manual_seed(int(seed))

    @torch.no_grad()
    def act(self, obs: dict, infos: dict) -> Dict[str, int]:
        pl = self.pl
        pl.perceive(obs)
        for i in pl.need_decision():
            cand = pl.candidates(i)
            logits = self.actor(*cands_to_tensors([cand], self.device))[0]
            k = int(logits.argmax()) if self.det else int(torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=self.gen))
            pl.apply_choice(i, k, cand)
        return pl.actions()
