"""Curriculum stages, environment stress variants, and disjoint seed splits (train/val/test)."""
from dataclasses import asdict, fields
from typing import Optional

from env import EnvConfig

TRAIN_SEED_BASE = 100_000
SPLITS = {"val": (10_000, 50), "test": (20_000, 100)}   # (first seed, default #episodes)

# Overrides applied on top of the default EnvConfig (20x20, 4 robots, 300 steps).
STAGES = {
    1: dict(height=12, width=12, n_agents=1, n_persistent=6, n_malicious=0,
            n_temp_initial=0, temp_spawn_prob=0.0, max_steps=150),          # explore + collect
    2: dict(height=12, width=12, n_agents=1, n_persistent=6, n_malicious=0,
            n_temp_initial=0, temp_spawn_prob=0.04, max_steps=150),         # + expiring items
    3: dict(height=16, width=16, n_agents=2, n_persistent=8, n_malicious=2,
            temp_spawn_prob=0.04, max_steps=200),                           # + 2 robots, malicious
    4: dict(),
    5: dict(height=30, width=30, n_agents=5, n_persistent=20, n_malicious=5,
            n_temp_initial=3, temp_spawn_prob=0.08, max_persist_active=50,
            max_temp_active=10, max_steps=400),                                                               # full problem
}

# Stress variants for generalisation tests (applied on top of the trained config).
VARIANTS = {
    "default": dict(),
    "dense":   dict(obstacle_density=(0.25, 0.35)),     # much more clutter
    "sparse":  dict(obstacle_density=(0.0, 0.05)),      # open maps
    "rushed":  dict(temp_life_range=(25, 45)),          # tighter deadlines
    "scarce":  dict(n_persistent=5, temp_spawn_prob=0.02),
}


def make_env_config(stage: Optional[int] = 4, info_mode: str = "shared", **overrides) -> EnvConfig:
    kw = dict(STAGES[stage]) if stage is not None else {}
    kw["info_mode"] = info_mode
    kw.update(overrides)
    return EnvConfig(**kw)


def cfg_to_dict(cfg: EnvConfig) -> dict:
    return asdict(cfg)


def cfg_from_dict(d: dict, variant: str = "default") -> EnvConfig:
    names = {f.name for f in fields(EnvConfig)}
    d = {k: v for k, v in d.items() if k in names}
    for k in ("obstacle_density", "temp_life_range", "b_discover", "checkpoint_xy"):
        if isinstance(d.get(k), list):
            d[k] = tuple(d[k])
    d.update(VARIANTS[variant])
    return EnvConfig(**d)


def split_seeds(split: str, n: Optional[int] = None):
    base, default_n = SPLITS[split]
    n = default_n if n is None else n
    assert n <= 10_000, "splits are 10k seeds wide"
    return list(range(base, base + n))