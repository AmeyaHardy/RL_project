"""
Visualise / simulate SwarmSearchEnv.

  python visualize.py                         # live window, greedy baseline
  python visualize.py --policy random         # random (masked) policy
  python visualize.py --save episode.gif      # save animation (works headless)
  python visualize.py --seed 5 --steps 200 --fps 8 --fog off

Later, plug in a trained policy by replacing `make_policy` (it must map
obs dict -> {agent: action}).
"""
import argparse

import matplotlib
import numpy as np

from swarm_env import EnvConfig, SwarmSearchEnv, TEMP
from baseline_greedy import GreedyPolicy

PX = 24  # pixels per cell in the rendered frame


def make_policy(name, env):
    if name == "greedy":
        return GreedyPolicy(env).act
    rng = np.random.default_rng(0)
    return lambda obs: {a: int(rng.choice(np.flatnonzero(env.action_mask(a))))
                        for a in env.agents}


def record(env, policy, seed, steps, fog):
    """Run one episode and capture a frame (image + labels + title) per step."""
    obs, _ = env.reset(seed=seed)
    frames = []

    def snap():
        img = env._render_rgb(PX)
        if not fog:  # show the whole world, ignore the fog of war
            known = env.known.copy()
            env.known[:] = True
            img = env._render_rgb(PX)
            env.known[:] = known
        labels = [(x, y, str(i), "white") for i, (y, x) in enumerate(env.pos)]
        for (y, x), it in env.active.items():
            if it["type"] == TEMP and (fog is False or it["seen"]):
                labels.append((x, y, str(int(it["expiry"] - env.t)), "black"))
        s = env.stats
        title = (f"t={env.t}   delivered={s['deliveries']} "
                 f"(P {s['delivered_by_type'][0]} / T {s['delivered_by_type'][1]})   "
                 f"expired T={s['expired_by_type'][1]}   return={s['ep_return']:.1f}")
        frames.append((img, labels, title))

    snap()
    while env.agents and env.t < steps:
        obs, *_ = env.step(policy(obs))
        snap()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["greedy", "random"], default="greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--fog", choices=["on", "off"], default="on")
    p.add_argument("--save", type=str, default=None, help="e.g. episode.gif")
    a = p.parse_args()

    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Patch

    env = SwarmSearchEnv(EnvConfig(max_steps=a.steps), render_mode="rgb_array")
    frames = record(env, make_policy(a.policy, env), a.seed, a.steps, a.fog == "on")

    fig, ax = plt.subplots(figsize=(7, 7.6))
    im = ax.imshow(frames[0][0])
    ax.set_xticks([]); ax.set_yticks([])
    legend = [Patch(color=c, label=l) for c, l in [
        ("#2650e6", "robot"), ("#00ccd9", "robot carrying"), ("#1aa64d", "depot"),
        ("#f2d933", "persistent"), ("#ff8000", "temporary (number = steps left)"),
        ("#d91a1a", "malicious"), ("#404047", "obstacle")]]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=3, fontsize=8, frameon=False)
    texts = []

    def draw(k):
        img, labels, title = frames[k]
        im.set_data(img)
        for t in texts:
            t.remove()
        texts.clear()
        for x, y, s, col in labels:
            texts.append(ax.text(x * PX + PX / 2, y * PX + PX / 2, s, color=col, ha="center",
                                 va="center", fontsize=9, fontweight="bold"))
        ax.set_title(title, fontsize=10)
        return [im]

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / a.fps, blit=False)
    if a.save:
        anim.save(a.save, writer=animation.PillowWriter(fps=a.fps))
        print(f"saved {a.save} ({len(frames)} frames)")
    else:
        plt.show()


if __name__ == "__main__":
    main()