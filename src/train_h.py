"""
Train hierarchical MAPPO (goal-level policy + BFS executor + team/individual/coordination reward).

  python train_h.py --selftest                                   # 1-2 minutes: shapes, one PPO update, eval hook
  python train_h.py --stage 2 --total-steps 300000 --out runs/h2
  python train_h.py --stage 4 --total-steps 1000000 --init-actor-from runs/h2/best.pt --out runs/h4 --ref greedy_s4.json
  python train_h.py --stage 4 --info-mode window --total-steps 1000000 --out runs/hw4 --ref res/greedy_window_mem.json

Evaluate:
  python evaluate.py run --policy hmappo --checkpoint runs/h4/best.pt --split val --episodes 50 --out res/h4.json
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from evaluate import evaluate_policy
from hmappo import HMAPPO, HMappoPolicy, HPPOConfig
from rewards import RewardConfig
from stages import cfg_to_dict, make_env_config, split_seeds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument("--total-steps", type=int, default=1_000_000, help="env steps summed over envs")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--rollout", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--ent-final", type=float, default=0.003)
    p.add_argument("--hidden", type=int, default=128)
    # reward
    p.add_argument("--lam-ind", type=float, default=0.3, help="weight of the individual reward block")
    p.add_argument("--w-stale-cov", type=float, default=0.30, help="team stale-coverage (patrol) weight")
    p.add_argument("--w-overlap", type=float, default=0.02, help="team redundant-window penalty")
    p.add_argument("--w-sync", type=float, default=0.25, help="team simultaneous-pickup bonus")
    p.add_argument("--shape-final", type=float, default=0.2, help="final scale of coordination terms (anneal 1 -> this)")
    p.add_argument("--anneal-frac", type=float, default=0.7, help="fraction of training over which to anneal")
    # misc
    p.add_argument("--map-pool", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=5, help="updates between validation runs")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-stochastic", action="store_true")
    p.add_argument("--ref", type=str, default=None, help="evaluate.py JSON (e.g. greedy_s4.json) printed as reference")
    p.add_argument("--init-actor-from", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=str, default="runs/h")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--info-mode", choices=["shared", "window"], default="shared")
    p.add_argument("--no-memory", action="store_true", help="window mode: strictly memoryless planner (ablation)")
    a = p.parse_args()
    if a.selftest:
        a.stage, a.n_envs, a.rollout, a.total_steps, a.eval_every, a.eval_episodes = 3, 2, 64, 128, 1, 2
        a.out = "runs/_selftest"

    ppo = HPPOConfig(n_envs=a.n_envs, rollout=a.rollout, epochs=a.epochs, minibatches=a.minibatches, gamma=a.gamma,
                     lam=a.lam, clip=a.clip, lr_actor=a.lr, ent_coef=a.ent_coef, ent_coef_final=a.ent_final,
                     hidden=a.hidden)
    rcfg = RewardConfig(lam_ind=a.lam_ind, w_stale_cov=a.w_stale_cov, w_overlap=a.w_overlap, w_sync=a.w_sync,
                        gamma=a.gamma)
    cfg = make_env_config(a.stage, a.info_mode)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"args": vars(a), "ppo": ppo.__dict__, "reward": rcfg.__dict__, "env_cfg": cfg_to_dict(cfg)}, f, indent=1)

    algo = HMAPPO(cfg_to_dict(cfg), ppo, rcfg, seed=a.seed, device=a.device, map_pool=a.map_pool,
                 memory=not a.no_memory)
    if a.init_actor_from:
        ck = torch.load(a.init_actor_from, map_location=a.device, weights_only=False)
        algo.actor.load_state_dict(ck["actor"])
        print(f"actor initialised from {a.init_actor_from}")
    if a.ref and os.path.exists(a.ref):
        ref = json.load(open(a.ref))
        s = ref["summary"]["return"]
        print(f"reference '{ref['label']}': return {s['mean']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]")

    steps_per_update = ppo.n_envs * ppo.rollout
    n_updates = max(1, a.total_steps // steps_per_update)
    print(f"stage {a.stage} | {cfg.n_agents} robots | {cfg.height}x{cfg.width} | actor params "
          f"{sum(x.numel() for x in algo.actor.parameters()):,} | {n_updates} updates x {steps_per_update} env-steps", flush=True)

    val_seeds = split_seeds("val", a.eval_episodes)
    fields = ["update", "env_steps", "fps", "train_return", "train_deliveries", "val_return", "val_lo", "val_hi",
              "entropy", "kl", "clipfrac", "value_loss", "explained_var", "epochs_run", "shape_scale", "n_decisions"]
    log_path = os.path.join(a.out, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)
    best, t0 = -np.inf, time.time()
    for u in range(1, n_updates + 1):
        frac = (u - 1) / n_updates
        ent = a.ent_coef + (a.ent_final - a.ent_coef) * frac
        lr_frac = 1.0 + (ppo.lr_final_frac - 1.0) * frac
        scale = 1.0 + (a.shape_final - 1.0) * min(1.0, frac / max(a.anneal_frac, 1e-6))
        t_u = time.time()
        buf = algo.collect(scale)
        st = algo.update(buf, ent, lr_frac)
        fps = steps_per_update / max(time.time() - t_u, 1e-6)
        rec = list(algo.recent)
        tr = float(np.mean([e["return"] for e in rec])) if rec else float("nan")
        td = float(np.mean([e["deliveries"] for e in rec])) if rec else float("nan")
        row = dict(update=u, env_steps=algo.env_steps, fps=fps, train_return=tr, train_deliveries=td,
                   val_return=float("nan"), val_lo=float("nan"), val_hi=float("nan"), entropy=st["ent"], kl=st["kl"],
                   clipfrac=st["clipfrac"], value_loss=st["vl"], explained_var=st["explained_var"],
                   epochs_run=st["epochs_run"], shape_scale=scale, n_decisions=st["n_decisions"])
        msg = (f"upd {u:4d}/{n_updates} steps {algo.env_steps:8d} fps {fps:4.0f} | train ret {tr:7.2f} deliv {td:5.1f} "
               f"| ent {st['ent']:.2f} kl {st['kl']:.3f} ev {st['explained_var']:.2f} dec {st['n_decisions']} shape {scale:.2f}")
        if u % a.eval_every == 0 or u == n_updates:
            algo.actor.eval()
            res = evaluate_policy(cfg, lambda env: HMappoPolicy(algo.actor, env.cfg, not a.eval_stochastic, a.device, not a.no_memory),
                                  val_seeds, "hmappo", "val", verbose=False)
            algo.actor.train()
            s = res["summary"]["return"]
            row.update(val_return=s["mean"], val_lo=s["ci_lo"], val_hi=s["ci_hi"])
            msg += f" | VAL {s['mean']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}] (deliv {res['summary']['deliveries']['mean']:.1f})"
            if s["mean"] > best:
                best = s["mean"]
                torch.save(algo.state_dict(), os.path.join(a.out, "best.pt"))
                msg += " *best*"
        torch.save(algo.state_dict(), os.path.join(a.out, "latest.pt"))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in fields])
        print(msg, flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min. best val return {best:.2f}. checkpoints in {a.out}/")
    if a.selftest:
        pol = HMappoPolicy.from_checkpoint(os.path.join(a.out, "best.pt"))
        r = evaluate_policy(cfg, lambda env: pol, split_seeds("val", 1), "reload", "val", verbose=False)
        print("SELFTEST OK  (checkpoint reload + eval:", round(r["summary"]["return"]["mean"], 2), ")")


if __name__ == "__main__":
    main()