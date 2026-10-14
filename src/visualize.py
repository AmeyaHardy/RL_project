"""
Visualise / simulate SwarmSearchEnv.

Usage:
  python visualize.py                                   # live window, greedy baseline (shared)
  python visualize.py --info-mode window                # live window, collective LOS mode
  python visualize.py --info-mode window --view both    # side-by-side global + agent obs in window mode
  python visualize.py --info-mode window --save win.gif # export GIF of window mode
  python visualize.py --policy random                   # random (masked) policy
  python visualize.py --seed 5 --steps 200 --fps 8 --fog off
  python visualize.py --view agent --agent 0            # inspect robot 0's observation view
"""
import argparse
import sys

import matplotlib
import numpy as np

try:
    from env import (DOWN, DROP, LEFT, PICK, RIGHT, STAY, UP,
                     EnvConfig, SwarmSearchEnv, PERSIST, TEMP, MAL)
except ImportError:
    from swarm_env import (DOWN, DROP, LEFT, PICK, RIGHT, STAY, UP,
                           EnvConfig, SwarmSearchEnv, PERSIST, TEMP, MAL)

try:
    from greedy_baseline import GreedyPolicy, observation_action_mask
except ImportError:
    from baseline_greedy import GreedyPolicy, observation_action_mask

PX = 24  # pixels per cell in the rendered frame


def make_policy(name: str, env: SwarmSearchEnv, seed: int = 0):
    """Factory for policies.

    A policy is a callable: obs_dict -> {agent_id: action_int}.
    Uses observation-derived action masks to prevent hidden-state leakage.
    """
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
    raise ValueError(f"Unknown policy '{name}'. Must be 'greedy' or 'random'.")


def render_step_frame(env: SwarmSearchEnv, fog: bool, view: str = "true", agent: int = 0, px: int = PX):
    """Render image, text labels, and episode stats title without leaking info."""
    if view == "agent":
        agent_key = env.possible_agents[agent % len(env.possible_agents)]
        obs = env._observations()[agent_key]
        img = env.render_observation(obs, px=px)
        labels = []
    elif view == "both":
        img = env.render_view(view="both", px=px, agent=agent, fog=fog)
        labels = []
    else:  # view == "true"
        if fog:
            img = env.render_view(view="true", px=px, fog=True)
        else:
            img = env.render_true_nofog(px=px)

        labels = []
        # Agent index labels
        for i, (y, x) in enumerate(env.pos):
            labels.append((x, y, str(i), "white"))

        # Temporary item countdown labels
        vis_items = env._union_view() if (fog and env.cfg.info_mode == "window") else None
        for (y, x), it in env.active.items():
            is_vis = (not fog) or (it.id in vis_items if vis_items is not None else it.seen)
            if it.type == TEMP and is_vis:
                remaining = max(0, int(round(it.expiry - env.t)))
                labels.append((x, y, str(remaining), "black"))

    # Episode statistics
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
    """Set up matplotlib figure, axes, and legend."""
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


def update_ax(ax, im, texts, img, labels, title, px: int = PX):
    """Update imshow and text labels in ax."""
    im.set_data(img)
    for t in texts:
        t.remove()
    texts.clear()
    for x, y, s, col in labels:
        texts.append(ax.text(x * px + px / 2, y * px + px / 2, s,
                             color=col, ha="center", va="center",
                             fontsize=9, fontweight="bold"))
    ax.set_title(title, fontsize=10)


def stream_to_gif(env: SwarmSearchEnv, policy, seed: int, steps: int, fog: bool,
                  fps: int, save_path: str, view: str = "true", agent: int = 0, px: int = PX):
    """Memory-efficient streaming GIF writer. Grabs frames directly without storing all in RAM."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = None
    writer = None

    try:
        obs, _ = env.reset(seed=seed)
        img, labels, title = render_step_frame(env, fog=fog, view=view, agent=agent, px=px)

        fig, ax = setup_figure(img.shape, px=px)
        im = ax.imshow(img)
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
            update_ax(ax, im, texts, img, labels, title, px=px)
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
                   fps: int, view: str = "true", agent: int = 0, px: int = PX):
    """Interactive live Matplotlib animation."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = None
    try:
        obs, _ = env.reset(seed=seed)
        frames = []
        frames.append(render_step_frame(env, fog=fog, view=view, agent=agent, px=px))

        while env.agents and env.t < steps:
            actions = policy(obs)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            frames.append(render_step_frame(env, fog=fog, view=view, agent=agent, px=px))
            if all(terminations.values()) or all(truncations.values()):
                break

        if not frames:
            print("No frames recorded.")
            return

        fig, ax = setup_figure(frames[0][0].shape, px=px)
        im = ax.imshow(frames[0][0])
        texts = []

        def draw(k):
            img, labels, title = frames[k]
            update_ax(ax, im, texts, img, labels, title, px=px)
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
    p = argparse.ArgumentParser(description="Visualize / simulate SwarmSearchEnv.")
    p.add_argument("--policy", choices=["greedy", "random"], default="greedy",
                   help="Policy to evaluate ('greedy' or 'random')")
    p.add_argument("--info-mode", choices=["shared", "private", "window"], default="shared",
                   help="Information sharing mode ('shared', 'private', or 'window')")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for environment and policy")
    p.add_argument("--steps", type=int, default=300,
                   help="Maximum episode steps")
    p.add_argument("--fps", type=int, default=10,
                   help="Animation frames per second")
    p.add_argument("--fog", choices=["on", "off"], default="on",
                   help="Enable or disable fog of war in visualization")
    p.add_argument("--view", choices=["true", "agent", "both"], default="true",
                   help="View mode: 'true' (global), 'agent' (single agent obs), 'both'")
    p.add_argument("--agent", type=int, default=0,
                   help="Agent index for 'agent' or 'both' view mode")
    p.add_argument("--save", type=str, default=None,
                   help="Filepath to export GIF (e.g. episode.gif)")
    args = p.parse_args()

    # Validation
    if args.fps <= 0:
        p.error("--fps must be a positive integer.")
    if args.steps <= 0:
        p.error("--steps must be a positive integer.")
    if args.seed < 0:
        p.error("--seed must be a non-negative integer.")

    env = SwarmSearchEnv(
        EnvConfig(max_steps=args.steps, info_mode=args.info_mode),
        render_mode="rgb_array"
    )

    # Validate agent selection against configured number of agents
    if args.agent < 0 or args.agent >= env.cfg.n_agents:
        p.error(f"--agent must be between 0 and {env.cfg.n_agents - 1}.")

    if args.save:
        matplotlib.use("Agg")

    fog_enabled = (args.fog == "on")
    policy = make_policy(args.policy, env, seed=args.seed)

    if args.save:
        stream_to_gif(env, policy, args.seed, args.steps, fog_enabled, args.fps,
                      args.save, view=args.view, agent=args.agent, px=PX)
    else:
        live_animation(env, policy, args.seed, args.steps, fog_enabled, args.fps,
                       view=args.view, agent=args.agent, px=PX)


if __name__ == "__main__":
    main()