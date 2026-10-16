from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from env import SwarmSearchEnv, EnvConfig
from stages import (VARIANTS, cfg_from_dict, cfg_to_dict, make_env_config, split_seeds)

METRICS = ["return", "value", "deliveries", "persist_delivered", "temp_delivered",
           "temp_expired", "temp_capture", "collisions", "malicious_picked", "length",
           "coverage", "mean_carrying", "overlap"]      # last three: coordination diagnostics
PRIMARY = "return"

class GreedyAdapter:
    def __init__(self, env: SwarmSearchEnv, **kw):
        self.env, self.kw, self.pol = env, kw, None

    def reset(self, seed: int):
        from greedy_baseline import GreedyPolicy
        self.pol = GreedyPolicy(self.env, seed=seed, **self.kw)

    def act(self, obs, infos):
        return self.pol.act(obs)


class RandomAdapter:
    """Uniform over legal actions (the lower reference every learner must clear)."""

    def __init__(self, env: SwarmSearchEnv):
        self.env, self.rng = env, np.random.default_rng(0)

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, infos):
        return {a: int(self.rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in obs}

def episode_metrics(env: SwarmSearchEnv) -> Dict[str, float]:
    s = env.stats
    dbt, ebt = s["delivered_by_type"], s["expired_by_type"]
    spawned = s.get("temp_spawned", 0) + env.cfg.n_temp_initial
    return {
        "return": float(s["ep_return"]),
        "value": float(s["delivered_value"]),
        "deliveries": float(s["deliveries"]),
        "persist_delivered": float(dbt[0]),
        "temp_delivered": float(dbt[1]),
        "temp_expired": float(ebt[1]),
        "temp_capture": float(dbt[1]) / spawned if spawned else float("nan"),
        "collisions": float(s["collisions"]),
        "malicious_picked": float(s.get("malicious_picked", 0)),
        "length": float(s.get("ep_length", env.t)),
        "coverage": float(env.known.mean()),                                   # fraction of map ever observed
        "mean_carrying": float(np.mean(s["carrying_per_step"])) if s["carrying_per_step"] else 0.0,
    }


def window_overlap(env: SwarmSearchEnv) -> float:
    """Redundant sensing among robots that are not carrying: cells seen by >=2 windows, in window-areas."""
    r, H, W = env.cfg.view_radius, env.cfg.height, env.cfg.width
    cnt = np.zeros((H, W), int)
    free = [i for i in range(env.cfg.n_agents) if env.carry[i] is None]
    if len(free) < 2:
        return 0.0
    for i in free:
        y, x = env.pos[i]
        cnt[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] += 1
    return float(np.maximum(cnt - 1, 0).sum()) / (2 * r + 1) ** 2


def run_episode(env: SwarmSearchEnv, policy, seed: int) -> Dict[str, float]:
    obs, infos = env.reset(seed=seed)
    policy.reset(seed)
    ov, n = 0.0, 0
    while env.agents:
        actions = policy.act(obs, infos)
        obs, _, _, _, infos = env.step(actions)
        ov += window_overlap(env)
        n += 1
    m = episode_metrics(env)
    m["overlap"] = ov / max(n, 1)
    return m


def evaluate_policy(cfg: EnvConfig, make_policy: Callable[[SwarmSearchEnv], object],
                    seeds: Sequence[int], label: str = "", split: str = "",
                    variant: str = "default", verbose: bool = True) -> dict:
    env = SwarmSearchEnv(cfg)
    policy = make_policy(env)
    rows: List[Dict[str, float]] = []
    t0 = time.time()
    for i, sd in enumerate(seeds):
        rows.append(run_episode(env, policy, int(sd)))
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{label}] {i + 1}/{len(seeds)} episodes  ({time.time() - t0:.0f}s)", flush=True)
    metrics = {k: [r[k] for r in rows] for k in METRICS}
    return {"label": label, "split": split, "variant": variant, "seeds": [int(s) for s in seeds],
            "env_cfg": cfg_to_dict(cfg), "metrics": metrics, "summary": summarize(metrics)}


def bootstrap_ci(x: Sequence[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(metrics: Dict[str, list]) -> Dict[str, dict]:
    out = {}
    for k, v in metrics.items():
        m, lo, hi = bootstrap_ci(v)
        arr = np.asarray(v, dtype=float)
        arr = arr[~np.isnan(arr)]
        out[k] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
    return out


def paired_compare(a: dict, b: dict, metric: str = PRIMARY, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap of (a - b) over identical seeds."""
    assert a["seeds"] == b["seeds"], "paired comparison needs identical seeds (same split & #episodes)"
    d = np.asarray(a["metrics"][metric], float) - np.asarray(b["metrics"][metric], float)
    m, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {"metric": metric, "diff": m, "ci_lo": lo, "ci_hi": hi, "n": int(len(d)),
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
            "significant": bool(lo > 0 or hi < 0)}


def episodes_needed(std: float, halfwidth: float) -> int:
    """Episodes for a 95% CI of +-halfwidth on a mean, given the per-episode std."""
    return int(np.ceil((1.96 * std / halfwidth) ** 2))


def format_summary(res: dict) -> str:
    lines = [f"{res['label']}  |  split={res['split']} variant={res['variant']} "
             f"episodes={len(res['seeds'])} info_mode={res['env_cfg']['info_mode']}"]
    for k in METRICS:
        if k not in res["summary"]:                  # JSONs written before the coordination metrics existed
            continue
        s = res["summary"][k]
        lines.append(f"  {k:18s} {s['mean']:8.2f}   95% CI [{s['ci_lo']:8.2f}, {s['ci_hi']:8.2f}]   std {s['std']:6.2f}")
    return "\n".join(lines)

def build_policy_factory(args, ckpt_cfg: Optional[dict]):
    if args.policy == "greedy":
        return lambda env: GreedyAdapter(env)
    if args.policy == "random":
        return lambda env: RandomAdapter(env)
    if args.policy == "scripted":                    # hand-written goal chooser + the same executor (no torch)
        from hier import ScriptedHierPolicy
        return lambda env: ScriptedHierPolicy(env)
    if args.policy == "hmappo":
        from hmappo import HMappoPolicy
        return lambda env: HMappoPolicy.from_checkpoint(args.checkpoint, env.cfg, deterministic=not args.stochastic)
    from mappo import MappoPolicy                    # lazy: only needs torch for learned policies
    return lambda env: MappoPolicy.from_checkpoint(args.checkpoint, deterministic=not args.stochastic)


def cmd_run(args):
    ckpt_cfg = None
    if args.policy in ("mappo", "hmappo"):
        import torch
        ckpt_cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["env_cfg"]
        cfg = cfg_from_dict(ckpt_cfg, variant=args.variant)
    else:
        cfg = make_env_config(args.stage, args.info_mode)
        for k, v in VARIANTS[args.variant].items():
            setattr(cfg, k, v)
    seeds = split_seeds(args.split, args.episodes)
    label = args.label or (args.policy if args.policy not in ("mappo", "hmappo") else os.path.basename(args.checkpoint))
    res = evaluate_policy(cfg, build_policy_factory(args, ckpt_cfg), seeds, label, args.split, args.variant)
    print(format_summary(res))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f)
        print(f"saved {args.out}")


def cmd_compare(args):
    a, b = (json.load(open(p)) for p in (args.a, args.b))
    print(format_summary(a)); print(); print(format_summary(b)); print()
    print(f"Paired comparison  A={a['label']}  minus  B={b['label']}  (same {len(a['seeds'])} seeds)")
    for m in METRICS:
        if m not in a["metrics"] or m not in b["metrics"]:
            continue
        if any(np.isnan(a["metrics"][m])) or any(np.isnan(b["metrics"][m])):
            continue
        r = paired_compare(a, b, m)
        flag = "  <-- significant" if r["significant"] else ""
        print(f"  {m:18s} diff {r['diff']:+8.2f}  95% CI [{r['ci_lo']:+8.2f}, {r['ci_hi']:+8.2f}]  "
              f"W/T/L {r['wins']}/{r['ties']}/{r['losses']}{flag}")


def cmd_plan(args):
    res = json.load(open(args.pilot))
    std = res["summary"][PRIMARY]["std"]
    n = episodes_needed(std, args.halfwidth)
    print(f"per-episode std of '{PRIMARY}' = {std:.2f}")
    print(f"episodes for a 95% CI of +-{args.halfwidth}: {n}   (unpaired; paired comparisons need fewer)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--policy", choices=["greedy", "random", "mappo", "hmappo", "scripted"], required=True)
    r.add_argument("--checkpoint", type=str, default=None)
    r.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax (mappo)")
    r.add_argument("--stage", type=int, default=4)
    r.add_argument("--info-mode", choices=["shared", "private", "window"], default="shared")
    r.add_argument("--split", choices=["val", "test"], default="val")
    r.add_argument("--episodes", type=int, default=None)
    r.add_argument("--variant", choices=list(VARIANTS), default="default")
    r.add_argument("--label", type=str, default=None)
    r.add_argument("--out", type=str, default=None)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    pl = sub.add_parser("plan"); pl.add_argument("pilot"); pl.add_argument("--halfwidth", type=float, default=2.0)
    args = p.parse_args()
    if args.cmd == "run" and args.policy in ("mappo", "hmappo") and not args.checkpoint:
        p.error("--checkpoint is required for --policy mappo / hmappo")
    {"run": cmd_run, "compare": cmd_compare, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    main()