# SwarmSearch: Hierarchical Multi-Agent Reinforcement Learning for Cooperative Search, Pickup, and Delivery

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PettingZoo](https://img.shields.io/badge/API-PettingZoo%20ParallelEnv-green.svg)](https://pettingzoo.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen.svg)]()

## Overview

**SwarmSearch** is a multi-agent reinforcement learning (MARL) framework tackling complex cooperative exploration, pickup, and delivery under partial observability, dynamic deadlines, and inter-agent coordination challenges.

The environment implements a high-throughput **PettingZoo ParallelEnv** simulating a swarm of autonomous ground robots operating in an unknown grid world. Robots must coordinate under limited sensory range (fog of war) to:
- Discover hidden items across unknown terrain
- Prioritize perishable/temporary items with expiration timers over persistent items
- Identify and avoid malicious items
- Avoid intra-swarm collisions and coordinate drop-offs at designated delivery zones
- Communicate over realistic communication models (centralized shared state, private observations, or local communication windows)

### Key Method: Hierarchical MAPPO (H-MAPPO)
To overcome the credit assignment and sample complexity bottlenecks of flat MARL on long-horizon grid worlds, we develop a **Hierarchical MAPPO** architecture:
1. **High-Level Centralized Training with Decentralized Execution (CTDE)**: High-level policy assigns macro-goals (candidate items, search frontiers, or patrol regions) to individual agents using spatial attention and multi-agent coordination rewards (stale-coverage patrol bonuses, window overlap penalties, and synchronous pickup incentives).
2. **Low-Level Collision-Free Planner**: Deterministic path planner and executor resolves local pathfinding and guarantees collision-free navigation between macro-goals.

---

## Performance Summary

Across paired benchmark evaluations on held-out validation and test seed splits (identical maps, seeds `10,000`–`10,049`), H-MAPPO outperforms strong greedy and heuristic baselines with statistical significance (95% bootstrap confidence intervals):

| Metric | H-MAPPO (`hw4/best.pt`) | Greedy Baseline | Delta (Paired Comparison) | Win/Tie/Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Episode Return** | **56.72** | 40.16 | **+16.56** [95% CI: +13.20, +20.21] | **48 / 0 / 2** |
| **Deliveries** | **19.44** | 15.92 | **+3.52** [95% CI: +2.58, +4.60] | **45 / 3 / 2** |
| **Temp Items Captured** | **84%** | 57% | **+27%** [95% CI: +22%, +32%] | **47 / 2 / 1** |
| **Expired Items** | **1.02** | 3.28 | **-2.26** [95% CI: -2.84, -1.72] | **40 / 9 / 1** |
| **Collisions** | **0.06** | 8.22 | **-8.16** [95% CI: -9.38, -6.98] | **49 / 0 / 1** |
| **Redundant Overlap** | **0.24** | 0.46 | **-0.22** [95% CI: -0.27, -0.17] | **46 / 0 / 4** |

---

## Repository Structure

```
├── swarm_search/               # Core Python package
│   ├── __init__.py             # Public package exports
│   ├── env.py                  # PettingZoo ParallelEnv with fog-of-war, items, drop zones
│   ├── hier.py                 # Hierarchical planner & macro-action abstraction
│   ├── henv.py                 # Hierarchical environment wrapper
│   ├── hmappo.py               # Hierarchical MAPPO policy & value network architectures
│   ├── mappo.py                # Flat MAPPO baseline implementation
│   ├── rewards.py              # Reward shaping (coverage, overlap penalty, sync pickup)
│   ├── stages.py               # Curriculum stages & disjoint seed splits (train/val/test)
│   ├── greedy_baseline.py      # BFS greedy heuristic benchmark policy
│   ├── evaluate.py             # Evaluation protocol, bootstrap CI, paired statistical test
│   ├── visualize.py            # Headless & interactive GIF rendering tools
│   ├── train.py                # Flat MAPPO training CLI
│   └── train_h.py              # Hierarchical MAPPO training CLI
├── tests/
│   ├── test_suite.py           # 33 unit tests (environment, masks, invariants, determinism)
│   └── test_hier.py            # Hierarchical policy validation & comparison script
├── runs/                       # Trained model checkpoints & logs
│   ├── hw4/best.pt             # Best trained Stage 4 H-MAPPO model (window mode)
│   ├── h5/best.pt              # Best trained Stage 5 H-MAPPO model (30x30, 5 robots)
│   ├── hw2/best.pt             # Stage 2 checkpoint
│   └── hw4_up/                 # Stage 4 checkpoint with updated coordination weights
├── res/                        # Evaluation metric outputs (JSON) across all test runs
├── viz/                        # Pre-rendered evaluation episodes (GIFs)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Installation & Environment Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/RL_project.git
cd RL_project
```

### 3. Create and Activate Virtual Environment
```bash
# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt / PowerShell):
python -m venv .venv
.\.venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Reproduction & Execution Guide

### 1. Run the Complete Test Suite
Verify environment mechanics, observation masks, determinism, and hierarchical planners:
```bash
python -m unittest discover tests
```
*All 33 test suites should pass cleanly with `OK`.*

---

### 2. Evaluate Trained Policies

#### A. Compare Trained H-MAPPO with Greedy Baseline (Paired Test)
Run paired statistical evaluation comparing the trained model against the greedy baseline on 50 validation seeds:
```bash
python -m swarm_search.evaluate compare res/hw4.json res/greedy_s4.json
```

#### B. Evaluate Checkpoint on Validation or Test Splits
Evaluate a trained model directly on the held-out validation split (`val`, seeds 10,000–10,049):
```bash
python -m swarm_search.evaluate run \
  --policy hmappo \
  --checkpoint runs/hw4/best.pt \
  --split val \
  --episodes 50 \
  --out res/my_eval_hw4.json
```

#### C. Evaluate the Heuristic / Greedy Baselines
```bash
python -m swarm_search.evaluate run \
  --policy greedy \
  --stage 4 \
  --info-mode window \
  --split val \
  --episodes 50 \
  --out res/my_eval_greedy.json
```

#### D. Quick Hierarchical vs. Greedy Benchmark Script
Run hierarchical evaluation across $N$ seeds on stage 4:
```bash
# Usage: python tests/test_hier.py <stage> <episodes>
python tests/test_hier.py 4 10
```

---

### 3. Visualize & Generate Episode GIFs

Generate and inspect an animated GIF of a trained agent or baseline solving an episode:

```bash
# Render trained H-MAPPO agent on seed 20000
python -m swarm_search.visualize \
  --policy hmappo \
  --checkpoint runs/hw4/best.pt \
  --seed 20000 \
  --fps 10 \
  --save viz/sample_hmappo.gif

# Render greedy baseline on seed 20000
python -m swarm_search.visualize \
  --policy greedy \
  --stage 4 \
  --info-mode window \
  --seed 20000 \
  --fps 10 \
  --save viz/sample_greedy.gif
```

---

### 4. Training

#### Train Hierarchical MAPPO (H-MAPPO)
```bash
python -m swarm_search.train_h \
  --stage 4 \
  --info-mode window \
  --total-steps 300000 \
  --n-envs 8 \
  --rollout 128 \
  --lr 3e-4 \
  --w-stale-cov 0.1 \
  --w-overlap 0.2 \
  --out runs/my_run
```

#### Train Flat MAPPO
```bash
python -m swarm_search.train \
  --stage 4 \
  --total-steps 300000 \
  --n-envs 8 \
  --rollout 128 \
  --lr 3e-4 \
  --out runs/my_flat_run
```

---

## Seed Protocol & Reproducibility Guarantee

To prevent data contamination and ensure rigorous scientific evaluation:
- **Disjoint Seed Splits**:
  - `val`: Seeds `10,000` to `10,049` (held-out for model selection & hyperparameter tuning)
  - `test`: Seeds `20,000` to `20,099` (held-out for final test benchmarks)
  - `train`: Dynamically generated starting from seed `100,000`+
- **Bootstrap 95% Confidence Intervals**: Every evaluation computes non-parametric bootstrap confidence intervals across paired episodes ($A - B$) so map variation is controlled.

---

## Dependencies

- `gymnasium >= 1.0.0`
- `pettingzoo >= 1.24.0`
- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `pillow >= 10.0.0`
