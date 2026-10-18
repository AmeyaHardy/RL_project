"""Training pipeline for MAPPO on SwarmSearchEnv curriculum stages."""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from evaluate import evaluate_policy
from mappo import MAPPO, MappoPolicy, PPOConfig
from stages import cfg_to_dict, make_env_config, split_seeds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--info-mode", choices=["shared", "private", "window"], default="shared")
    p.add_argument("--total-steps", type=int, default=1_000_000, help="environment steps (summed over envs)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--workers", type=int, default=0, help="0 = in-process; N = one subprocess per env")
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--recurrent", action="store_true", help="GRU actor (use for --info-mode window)")
    p.add_argument("--chunk", type=int, default=None, help="BPTT length (default 16 if recurrent else 1)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.015)
    p.add_argument("--ent-final", type=float, default=0.003)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--map-pool", type=int, default=None, help="train on only K distinct maps (generalisation studies)")
    p.add_argument("--eval-every", type=int, default=10, help="updates between validation runs")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-deterministic", action="store_true",
                   help="validate with argmax actions (default: stochastic sampling)")
    p.add_argument("--ref", type=str, default=None, help="evaluate.py JSON (e.g. greedy) to print as a reference line")
    p.add_argument("--init-actor-from", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=str, default="runs/run")
    a = p.parse_args()

    chunk = a.chunk or (16 if a.recurrent else 1)
    rollout = (a.rollout // chunk) * chunk
    ppo = PPOConfig(
        n_envs=a.n_envs, rollout=rollout, chunk=chunk, epochs=a.epochs, minibatches=a.minibatches,
        gamma=a.gamma, lam=a.lam, clip=a.clip, lr_actor=a.lr, lr_critic=a.lr, ent_coef=a.ent_coef,
        ent_coef_final=a.ent_final, hidden=a.hidden, recurrent=a.recurrent
    )
    cfg = make_env_config(a.stage, a.info_mode)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"args": vars(a), "ppo": ppo.__dict__, "env_cfg": cfg_to_dict(cfg)}, f, indent=1)

    algo = MAPPO(cfg_to_dict(cfg), ppo, workers=a.workers, seed=a.seed, device=a.device, map_pool=a.map_pool)
    if a.init_actor_from:
        ck = torch.load(a.init_actor_from, map_location=a.device, weights_only=False)
        own = algo.actor.state_dict()
        ok = {k: v for k, v in ck["actor"].items()
              if k != "coords" and k in own and own[k].shape == v.shape}
        algo.actor.load_state_dict(ok, strict=False)
        print(f"initialised actor from {a.init_actor_from}: {len(ok)}/{len(own) - 1} learnable tensors transferred")

    if a.ref and os.path.exists(a.ref):
        ref = json.load(open(a.ref))
        print(f"reference '{ref['label']}': return {ref['summary']['return']['mean']:.2f} "
              f"[{ref['summary']['return']['ci_lo']:.2f}, {ref['summary']['return']['ci_hi']:.2f}]")

    n_params = sum(x.numel() for x in algo.actor.parameters())
    steps_per_update = ppo.n_envs * ppo.rollout
    n_updates = max(1, a.total_steps // steps_per_update)
    print(f"stage {a.stage} | {a.info_mode} | {cfg.n_agents} robots | {cfg.height}x{cfg.width} | "
          f"actor params {n_params:,} | {n_updates} updates x {steps_per_update} env-steps | "
          f"recurrent={a.recurrent}", flush=True)

    val_seeds = split_seeds("val", a.eval_episodes)
    log_path = os.path.join(a.out, "log.csv")
    fields = ["update", "env_steps", "fps", "train_return", "train_deliveries", "val_return", "val_lo", "val_hi",
              "entropy", "kl", "clipfrac", "value_loss", "explained_var", "epochs_run"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)
    best, t0 = -np.inf, time.time()

    for u in range(1, n_updates + 1):
        frac = (u - 1) / n_updates
        ent = a.ent_coef + (a.ent_final - a.ent_coef) * frac
        lr_frac = 1.0 + (ppo.lr_final_frac - 1.0) * frac
        t_u = time.time()
        algo.actor.train()
        buf = algo.collect()
        st = algo.update(buf, ent, lr_frac)
        fps = steps_per_update / max(time.time() - t_u, 1e-6)
        rec = list(algo.recent)
        tr = float(np.mean([e["return"] for e in rec])) if rec else float("nan")
        td = float(np.mean([e["deliveries"] for e in rec])) if rec else float("nan")
        row = dict(update=u, env_steps=algo.env_steps, fps=fps, train_return=tr, train_deliveries=td,
                   val_return=float("nan"), val_lo=float("nan"), val_hi=float("nan"), entropy=st["ent"],
                   kl=st["kl"], clipfrac=st["clipfrac"], value_loss=st["vl"], explained_var=st["explained_var"],
                   epochs_run=st["epochs_run"])
        msg = (f"upd {u:4d}/{n_updates} steps {algo.env_steps:8d} fps {fps:5.0f} | train ret {tr:7.2f} "
               f"deliv {td:5.1f} | ent {st['ent']:.2f} kl {st['kl']:.3f} ev {st['explained_var']:.2f}")
        if u % a.eval_every == 0 or u == n_updates:
            algo.actor.eval()
            res = evaluate_policy(
                cfg,
                lambda env: MappoPolicy(algo.actor, a.eval_deterministic, device=a.device),
                val_seeds, "mappo", "val", verbose=False
            )
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
    algo.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min. best val return {best:.2f}. checkpoints in {a.out}/")


if __name__ == "__main__":
    main()