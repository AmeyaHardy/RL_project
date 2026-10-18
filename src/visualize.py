"""Visualisation tools: live interactive rendering and GIF export for SwarmSearchEnv."""
from __future__ import annotations

import argparse
import sys

import matplotlib
import numpy as np

from env import (DOWN, DROP, LEFT, PICK, RIGHT, STAY, UP,
                 EnvConfig, SwarmSearchEnv, PERSIST, TEMP, MAL)
from greedy_baseline import GreedyPolicy, observation_action_mask
from stages import cfg_from_dict, make_env_config

PX = 24
ROBOT_COLORS = ["#00e5ff", "#ff2bd6", "#39ff14", "#ffffff", "#ff9100", "#b388ff", "#ffeb3b", "#ff5252"]


class HierActor:
    """Callable wrapper for hierarchical policies exposing current goals for overlay rendering."""

    def __init__(self, pol):
        self.pol, self.pl = pol, pol.pl

    def __call__(self, obs):
        return self.pol.act(obs, None)

    def overlay(self, env):
        pl, segs, marks = self.pl, [], []
        for i, (y, x) in enumerate(env.pos):
            c = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            g = pl.goal[i] if i < len(pl.goal) else None
            if env.carry[i] is not None:
                tgt, shape = pl.depot, "D"                       # delivering
            elif g is not None and g[0] in ("item", "region"):
                tgt, shape = g[1], ("o" if g[0] == "item" else "s")
            else:
                continue
            segs.append((x, y, tgt[1], tgt[0], c))
            marks.append((tgt[1], tgt[0], c, shape))
        return segs, marks


def make_policy(name: str, env: SwarmSearchEnv, seed: int = 0, checkpoint=None, memory: bool = True):
    """Create policy action function: greedy / random, or a planner+executor policy with a goal chooser."""
    if name in ("scripted", "nearest_goal", "random_goal", "hmappo"):
        from hier import ChooserHierPolicy, ScriptedHierPolicy
        if name == "scripted":
            pol = ScriptedHierPolicy(env, memory=memory)
        elif name == "hmappo":
            from hmappo import HMappoPolicy
            pol = HMappoPolicy.from_checkpoint(checkpoint, env.cfg, deterministic=True)
        else:
            pol = ChooserHierPolicy(env, name.split("_")[0], memory=memory)
        pol.reset(seed)
        return HierActor(pol)
    if name == "greedy":
        return GreedyPolicy(env, seed=seed).act
    if name == "random":
        rng = np.random.default_rng(seed)

        def random_act(obs):
            actions = {}
            for a in env.agents:
                if a in obs:
                    mask = observation_action_mask(obs[a], env.cfg)
                    valid = np.flatnonzero(mask)
                    actions[a] = int(rng.choice(valid)) if len(valid) > 0 else STAY
            return actions

        return random_act
    raise ValueError(f"Unknown policy '{name}'.")


def render_step_frame(env: SwarmSearchEnv, fog: bool, view: str = "true", agent: int = 0, px: int = PX):
    """Render frame image, labels, and title statistics without leaking hidden state."""
    if view == "agent":
        agent_key = env.possible_agents[agent % len(env.possible_agents)]
        obs = env._observations()[agent_key]
        img = env.render_observation(obs, px=px)
        labels = []
    elif view == "both":
        img = env.render_view(view="both", px=px, agent=agent, fog=fog)
        labels = []
    else:
        img = env.render_view(view="true", px=px, fog=True) if fog else env.render_true_nofog(px=px)
        labels = []
        for i, (y, x) in enumerate(env.pos):
            labels.append((x, y, str(i), "white"))

        win_mask = env._union_view() if (fog and env.cfg.info_mode == "window") else None
        for (y, x), it in env.active.items():
            is_vis = (not fog) or (bool(win_mask[y, x]) if win_mask is not None else it.seen)
            if it.type == TEMP and is_vis:
                remaining = max(0, int(round(it.expiry - env.t)))
                labels.append((x, y, str(remaining), "black"))

    s = env.stats
    p_del = s["delivered_by_type"][PERSIST]
    t_del = s["delivered_by_type"][TEMP]
    m_del = s["delivered_by_type"][MAL]
    exp_t = s["expired_by_type"][TEMP]
    mal_pick = s.get("malicious_picked", 0)
    ret = s.get("ep_return", 0.0)

    title = (f"[{env.cfg.info_mode}] t={env.t} | Deliv: P={p_del} T={t_del} M={m_del} "
             f"(tot {s['deliveries']}) | Exp T={exp_t} | Mal={mal_pick} | Ret={ret:.1f}")
    return img, labels, title


def setup_figure(img_shape, px: int = PX):
    """Set up Matplotlib figure, axes, and color legend."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7, 7.6))
    ax.set_xticks([])
    ax.set_yticks([])

    legend = [Patch(color=c, label=l) for c, l in [
        ("#2650e6", "robot"),
        ("#1aa64d", "checkpoint"),
        ("#f2d933", "persistent"),
        ("#ff8000", "temp (num = steps left)"),
        ("#d91a1a", "malicious"),
        ("#404047", "obstacle"),
    ]]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=3, fontsize=8, frameon=False)
    return fig, ax


def get_overlay(policy, env):
    return policy.overlay(env) if hasattr(policy, "overlay") else None


def update_ax(ax, im, texts, img, labels, title, px: int = PX, overlay=None):
    """Update imshow canvas, text annotations and (optional) goal overlay."""
    im.set_data(img)
    for t in texts:
        t.remove()
    texts.clear()
    if overlay is not None:
        segs, marks = overlay
        for x0, y0, x1, y1, c in segs:
            texts.extend(ax.plot([x0 * px + px / 2, x1 * px + px / 2], [y0 * px + px / 2, y1 * px + px / 2],
                                 color=c, lw=1.6, alpha=0.9, zorder=5))
        for x, y, c, m in marks:
            texts.append(ax.scatter([x * px + px / 2], [y * px + px / 2], s=90, marker=m, facecolors="none",
                                    edgecolors=c, linewidths=2, zorder=6))
    for x, y, s, col in labels:
        texts.append(ax.text(x * px + px / 2, y * px + px / 2, s,
                             color=col, ha="center", va="center",
                             fontsize=9, fontweight="bold"))
    ax.set_title(title, fontsize=10)


def stream_to_gif(env: SwarmSearchEnv, policy, seed: int, steps: int, fog: bool,
                  fps: int, save_path: str, view: str = "true", agent: int = 0, px: int = PX, show_goals: bool = True):
    """Stream environment simulation frames directly to GIF file."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = None
    writer = None

    try:
        obs, _ = env.reset(seed=seed)
        img, labels, title = render_step_frame(env, fog=fog, view=view, agent=agent, px=px)

        fig, ax = setup_figure(img.shape, px=px)
        im = ax.imshow(img)
        ax.autoscale(False)
        texts = []
        update_ax(ax, im, texts, img, labels, title, px=px)

        writer = animation.PillowWriter(fps=fps)
        writer.setup(fig, save_path, dpi=fig.dpi)

        frame_count = 0
        writer.grab_frame()
        frame_count += 1

        while env.agents and env.t < steps:
            actions = policy(obs)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            img, labels, title = render_step_frame(env, fog=fog, view=view, agent=agent, px=px)
            update_ax(ax, im, texts, img, labels, title, px=px,
                      overlay=get_overlay(policy, env) if (show_goals and view == "true") else None)
            writer.grab_frame()
            frame_count += 1

            if all(terminations.values()) or all(truncations.values()):
                break

        print(f"Saved {save_path} ({frame_count} frames)")
    finally:
        if writer is not None:
            try:
                writer.finish()
            except Exception:
                pass
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass


def live_animation(env: SwarmSearchEnv, policy, seed: int, steps: int, fog: bool,
                   fps: int, view: str = "true", agent: int = 0, px: int = PX, show_goals: bool = True):
    """Execute live interactive Matplotlib animation of episode."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = None
    try:
        obs, _ = env.reset(seed=seed)
        frames = []
        ov = lambda: get_overlay(policy, env) if (show_goals and view == "true") else None
        frames.append(render_step_frame(env, fog=fog, view=view, agent=agent, px=px) + (None,))

        while env.agents and env.t < steps:
            actions = policy(obs)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            frames.append(render_step_frame(env, fog=fog, view=view, agent=agent, px=px) + (ov(),))
            if all(terminations.values()) or all(truncations.values()):
                break

        if not frames:
            print("No frames recorded.")
            return

        fig, ax = setup_figure(frames[0][0].shape, px=px)
        im = ax.imshow(frames[0][0])
        ax.autoscale(False)
        texts = []

        def draw(k):
            img, labels, title, overlay = frames[k]
            update_ax(ax, im, texts, img, labels, title, px=px, overlay=overlay)
            return [im]

        anim = animation.FuncAnimation(
            fig, draw, frames=len(frames),
            interval=1000 / fps, blit=False, repeat=False
        )
        plt.show()
    finally:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass


def main():
    """Parse command-line arguments and run episode visualization."""
    p = argparse.ArgumentParser(description="Visualize / simulate SwarmSearchEnv.")
    p.add_argument("--policy", choices=["greedy", "random", "scripted", "nearest_goal", "random_goal", "hmappo"],
                   default="greedy", help="greedy/random, or planner+executor with a goal chooser (hmappo = trained)")
    p.add_argument("--checkpoint", type=str, default=None, help="checkpoint for --policy hmappo (e.g. runs/hw4/best.pt)")
    p.add_argument("--stage", type=int, default=4, choices=[1, 2, 3, 4], help="env stage (ignored for hmappo: uses checkpoint cfg)")
    p.add_argument("--no-memory", action="store_true", help="planner without window memory (ablation)")
    p.add_argument("--no-goals", action="store_true", help="do not draw goal lines/markers")
    p.add_argument("--info-mode", choices=["shared", "private", "window"], default="shared",
                   help="Information sharing mode ('shared', 'private', or 'window')")
    p.add_argument("--seed", type=int, default=0, help="Random seed for environment and policy")
    p.add_argument("--steps", type=int, default=None, help="Maximum episode steps (default: the env config's max_steps)")
    p.add_argument("--fps", type=int, default=10, help="Animation frames per second")
    p.add_argument("--fog", choices=["on", "off"], default="on", help="Enable or disable fog of war in visualization")
    p.add_argument("--view", choices=["true", "agent", "both"], default="true",
                   help="View mode: 'true' (global), 'agent' (single agent obs), 'both'")
    p.add_argument("--agent", type=int, default=0, help="Agent index for 'agent' or 'both' view mode")
    p.add_argument("--save", type=str, default=None, help="Filepath to export GIF (e.g. episode.gif)")
    args = p.parse_args()

    if args.fps <= 0:
        p.error("--fps must be a positive integer.")
    if args.steps is not None and args.steps <= 0:
        p.error("--steps must be a positive integer.")
    if args.policy == "hmappo" and not args.checkpoint:
        p.error("--checkpoint is required for --policy hmappo")
    if args.seed < 0:
        p.error("--seed must be a non-negative integer.")

    if args.policy == "hmappo":
        import torch
        cfg = cfg_from_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["env_cfg"])
        print(f"using the checkpoint's env config: info_mode={cfg.info_mode}, {cfg.n_agents} robots, {cfg.height}x{cfg.width}")
    else:
        cfg = make_env_config(args.stage, args.info_mode)
    if args.steps is not None:
        cfg.max_steps = args.steps
    steps = cfg.max_steps
    env = SwarmSearchEnv(cfg, render_mode="rgb_array")

    if args.agent < 0 or args.agent >= env.cfg.n_agents:
        p.error(f"--agent must be between 0 and {env.cfg.n_agents - 1}.")

    if args.save:
        matplotlib.use("Agg")

    fog_enabled = (args.fog == "on")
    policy = make_policy(args.policy, env, seed=args.seed, checkpoint=args.checkpoint, memory=not args.no_memory)

    if args.save:
        stream_to_gif(env, policy, args.seed, steps, fog_enabled, args.fps,
                      args.save, view=args.view, agent=args.agent, px=PX, show_goals=not args.no_goals)
    else:
        live_animation(env, policy, args.seed, steps, fog_enabled, args.fps,
                       view=args.view, agent=args.agent, px=PX, show_goals=not args.no_goals)


if __name__ == "__main__":
    main()