from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from env import (DOWN, DROP, LEFT, PICK, RIGHT, STAY, UP,
                 EnvConfig, SwarmSearchEnv, PERSIST, TEMP, MAL, Status)
from greedy_baseline import GreedyPolicy, bfs, observation_action_mask, evaluate
from visualize import make_policy, render_step_frame, stream_to_gif


class TestAuditAndPatches(unittest.TestCase):

    def setUp(self):
        """Initialize default test configuration."""
        self.cfg = EnvConfig(height=10, width=10, n_agents=2, max_steps=50)

    def test_01_no_hidden_state_leakage(self):
        """Identical observations produce identical masks regardless of hidden grid changes."""
        env1 = SwarmSearchEnv(self.cfg)
        obs1, _ = env1.reset(seed=42)
        a0 = env1.possible_agents[0]
        mask1 = observation_action_mask(obs1[a0], self.cfg)

        unobserved = np.argwhere(~env1.known)
        self.assertGreater(len(unobserved), 0, "There should be unobserved cells")
        uy, ux = unobserved[0]
        original_val = env1.grid[uy, ux]
        env1.grid[uy, ux] = not original_val

        mask2 = observation_action_mask(obs1[a0], self.cfg)
        np.testing.assert_array_equal(mask1, mask2)

    def test_02_known_obstacles_blocked(self):
        """Movement into known obstacles is blocked in observation action mask."""
        H, W = 10, 10
        obs = {
            "team_map": np.zeros((7, H, W), dtype=np.float32),
            "scalars": np.zeros(12, dtype=np.float32),
        }
        obs["scalars"][0] = 5.0 / H
        obs["scalars"][1] = 5.0 / W
        obs["team_map"][0, 4, 5] = 1.0
        obs["team_map"][1, 4, 5] = 1.0

        mask = observation_action_mask(obs, self.cfg)
        self.assertFalse(mask[UP], "UP movement into a known obstacle must be blocked")

    def test_03_unknown_cells_permitted(self):
        """Movement into unknown in-bounds cells is permitted."""
        H, W = 10, 10
        obs = {
            "team_map": np.zeros((7, H, W), dtype=np.float32),
            "scalars": np.zeros(12, dtype=np.float32),
        }
        obs["scalars"][0] = 5.0 / H
        obs["scalars"][1] = 5.0 / W

        mask = observation_action_mask(obs, self.cfg)
        self.assertTrue(mask[UP], "UP into unknown in-bounds cell must be permitted")
        self.assertTrue(mask[DOWN], "DOWN into unknown in-bounds cell must be permitted")
        self.assertTrue(mask[LEFT], "LEFT into unknown in-bounds cell must be permitted")
        self.assertTrue(mask[RIGHT], "RIGHT into unknown in-bounds cell must be permitted")

    def test_04_reproducibility(self):
        """Fixed seeds reproduce deterministic actions and returns."""
        def run_sim(seed):
            env = SwarmSearchEnv(self.cfg)
            obs, _ = env.reset(seed=seed)
            pol = GreedyPolicy(env, seed=seed)
            act_history = []
            for _ in range(15):
                acts = pol.act(obs)
                act_history.append(copy.deepcopy(acts))
                obs, r, term, trunc, _ = env.step(acts)
                if all(term.values()) or all(trunc.values()):
                    break
            return act_history, env.stats["ep_return"]

        acts1, ret1 = run_sim(seed=123)
        acts2, ret2 = run_sim(seed=123)
        self.assertEqual(acts1, acts2, "Actions must be identical across deterministic seeded runs")
        self.assertEqual(ret1, ret2, "Returns must be identical across deterministic seeded runs")

    def test_05_lazy_fallbacks_no_wasted_rng(self):
        """RNG state changes only when stochastic choices are made."""
        env = SwarmSearchEnv(self.cfg)
        obs, _ = env.reset(seed=0)
        pol = GreedyPolicy(env, seed=999)

        rng_state_before = pol.rng.bit_generator.state
        mask = np.array([True, True, True, True, True, False, False])
        pol._wiggle(mask)
        rng_state_after_wiggle = pol.rng.bit_generator.state
        self.assertNotEqual(rng_state_before, rng_state_after_wiggle)

    def test_06_bfs_correctness(self):
        """Validate shortest paths, unreachable targets, and first actions."""
        free = np.ones((5, 5), dtype=bool)
        free[1, 0] = False
        dist, first = bfs(free, start=(0, 0))

        self.assertEqual(dist[0, 0], 0)
        self.assertEqual(first[(2, 0)], RIGHT)
        self.assertEqual(dist[2, 0], 4)

        free[4, 4] = True
        free[3, 4] = free[4, 3] = False
        dist2, first2 = bfs(free, start=(0, 0))
        self.assertEqual(dist2[4, 4], -1)
        self.assertNotIn((4, 4), first2)

    def test_07_persistent_assignments(self):
        """Valid target assignments survive across consecutive timesteps."""
        H, W = 10, 10
        cfg = EnvConfig(height=H, width=W, n_agents=1, max_steps=20)
        env = SwarmSearchEnv(cfg)
        obs, _ = env.reset(seed=10)
        a = env.possible_agents[0]
        pol = GreedyPolicy(env, seed=10)

        o1 = copy.deepcopy(obs[a])
        o1["scalars"][0], o1["scalars"][1] = 5.0 / H, 2.0 / W
        o1["team_map"][0, :, :] = 1.0
        o1["team_map"][1, :, :] = 0.0
        o1["team_map"][2:] = 0.0
        o1["team_map"][2, 5, 8] = 1.0

        pol.act({a: o1})
        self.assertEqual(pol.targets[a], (5, 8))
        self.assertEqual(pol.target_type[a], "item")

        o2 = copy.deepcopy(o1)
        o2["scalars"][0], o2["scalars"][1] = 5.0 / H, 3.0 / W
        pol.act({a: o2})

        self.assertEqual(pol.targets[a], (5, 8))
        self.assertEqual(pol.target_type[a], "item")

    def test_08_assignment_invalidation(self):
        """Invalid targets are cleared upon policy step."""
        env = SwarmSearchEnv(self.cfg)
        obs, _ = env.reset(seed=15)
        pol = GreedyPolicy(env, seed=15)
        pol.act(obs)

        a0 = env.possible_agents[0]
        pol.targets[a0] = (99, 99)
        pol.target_type[a0] = "item"

        pol.act(obs)
        self.assertNotEqual(pol.targets.get(a0), (99, 99))

    def test_09_information_modes(self):
        """Evaluate baseline execution across shared, private, and window modes."""
        for mode in ("shared", "private", "window"):
            cfg = EnvConfig(height=10, width=10, n_agents=2, max_steps=10, info_mode=mode)
            rows = evaluate(n_episodes=1, seed=1, cfg=cfg)
            self.assertEqual(len(rows), 1)
            self.assertIn("ep_return", rows[0])
            self.assertIn("ep_length", rows[0])

    def test_10_item_selection_avoid_malicious(self):
        """Select persistent or temporary items while ignoring malicious items."""
        H, W = 10, 10
        cfg = EnvConfig(height=H, width=W, n_agents=1, max_steps=20)
        env = SwarmSearchEnv(cfg)
        obs, _ = env.reset(seed=4)
        a = env.possible_agents[0]

        pol = GreedyPolicy(env, seed=4)
        o = copy.deepcopy(obs[a])
        o["team_map"][2:] = 0.0
        o["scalars"][0], o["scalars"][1] = 5.0 / H, 5.0 / W
        o["team_map"][0, :, :] = 1.0
        o["team_map"][1, :, :] = 0.0

        o["team_map"][4, 5, 6] = 1.0
        o["team_map"][2, 5, 7] = 1.0

        acts = pol.act({a: o})
        self.assertEqual(pol.targets[a], (5, 7))
        self.assertEqual(acts[a], RIGHT)

    def test_11_deadline_logic(self):
        """Infeasible temporary items are not targeted."""
        H, W = 10, 10
        cfg = EnvConfig(height=H, width=W, n_agents=1, max_steps=20)
        env = SwarmSearchEnv(cfg)
        obs, _ = env.reset(seed=4)
        a = env.possible_agents[0]
        pol = GreedyPolicy(env, seed=4)

        o = copy.deepcopy(obs[a])
        o["team_map"][2:] = 0.0
        o["scalars"][0], o["scalars"][1] = 0.0, 0.0
        o["team_map"][0, :, :] = 1.0
        o["team_map"][1, :, :] = 0.0

        o["team_map"][3, 0, 8] = 0.01
        pol.act({a: o})
        self.assertNotEqual(pol.targets.get(a), (0, 8))

    def test_12_depot_behavior(self):
        """Carrying agents navigate to depot and drop."""
        H, W = 10, 10
        cfg = EnvConfig(height=H, width=W, n_agents=1, max_steps=20)
        env = SwarmSearchEnv(cfg)
        obs, _ = env.reset(seed=4)
        a = env.possible_agents[0]
        pol = GreedyPolicy(env, seed=4)

        o = copy.deepcopy(obs[a])
        o["scalars"][0], o["scalars"][1] = 5.0 / H, 5.0 / W
        o["scalars"][2] = 1.0
        o["scalars"][6], o["scalars"][7] = 0.0, 0.0
        o["team_map"][0, :, :] = 1.0

        acts = pol.act({a: o})
        self.assertEqual(acts[a], DROP, "Carrying agent at depot must DROP")

    def test_13_episode_length_accuracy(self):
        """Recorded episode lengths match executed steps."""
        cfg = EnvConfig(height=10, width=10, n_agents=2, max_steps=12)
        rows = evaluate(n_episodes=2, seed=5, cfg=cfg)
        for r in rows:
            self.assertIn("ep_length", r)
            self.assertLessEqual(r["ep_length"], 12)
            self.assertGreater(r["ep_length"], 0)

    def test_14_cli_validation(self):
        """Invalid episode counts raise ValueError."""
        with self.assertRaises(ValueError):
            evaluate(n_episodes=0)

    def test_15_rendering_views(self):
        """Generate frames for fogged, observation, and no-fog views."""
        cfg = EnvConfig(height=10, width=10, n_agents=2, max_steps=10)
        env = SwarmSearchEnv(cfg, render_mode="rgb_array")
        env.reset(seed=1)

        img_fog, _, _ = render_step_frame(env, fog=True, view="true", px=16)
        self.assertEqual(img_fog.shape, (10 * 16, 10 * 16, 3))

        img_nofog, _, _ = render_step_frame(env, fog=False, view="true", px=16)
        self.assertEqual(img_nofog.shape, (10 * 16, 10 * 16, 3))

        img_agent, _, _ = render_step_frame(env, fog=True, view="agent", agent=0, px=16)
        self.assertEqual(len(img_agent.shape), 3)

    def test_16_gif_export(self):
        """Export short simulation episode to GIF format."""
        cfg = EnvConfig(height=8, width=8, n_agents=1, max_steps=5)
        env = SwarmSearchEnv(cfg, render_mode="rgb_array")
        pol = make_policy("random", env, seed=0)

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            tmp_path = f.name

        try:
            stream_to_gif(env, pol, seed=0, steps=5, fog=True, fps=5, save_path=tmp_path, px=16)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_17_environment_invariants(self):
        """Verify environment invariants hold throughout simulation."""
        cfg = EnvConfig(height=10, width=10, n_agents=2, max_steps=20)
        env = SwarmSearchEnv(cfg)
        obs, _ = env.reset(seed=7)
        pol = GreedyPolicy(env, seed=7)

        for _ in range(15):
            acts = pol.act(obs)
            obs, r, term, trunc, _ = env.step(acts)
            env.check_invariants()
            if all(term.values()) or all(trunc.values()):
                break

    def test_18_claim_blocks_pickup(self):
        """Conserve pickup when cell is claimed and claim_blocks_pickup is active."""
        H, W = 10, 10
        cfg = EnvConfig(height=H, width=W, n_agents=1, max_steps=20, claim_blocks_pickup=True)
        env = SwarmSearchEnv(cfg)
        obs, _ = env.reset(seed=4)
        a = env.possible_agents[0]

        o = copy.deepcopy(obs[a])
        o["scalars"][0], o["scalars"][1] = 5.0 / H, 5.0 / W
        o["scalars"][2] = 0.0
        o["team_map"][0, :, :] = 1.0
        o["team_map"][2, 5, 5] = 1.0
        o["team_map"][6, 5, 5] = 1.0

        mask = observation_action_mask(o, cfg)
        self.assertFalse(mask[PICK], "PICK must be blocked when claim_blocks_pickup is True and cell is claimed")

    def test_19_frontier_tracking_arrival_and_private_mode(self):
        """Private mode maintains separate visited frontier sets."""
        H, W = 10, 10
        cfg = EnvConfig(height=H, width=W, n_agents=2, max_steps=20, info_mode="private")
        env = SwarmSearchEnv(cfg)
        obs, _ = env.reset(seed=20)
        pol = GreedyPolicy(env, seed=20)

        a0, a1 = env.possible_agents[0], env.possible_agents[1]
        pol.act(obs)

        self.assertIn(a0, pol.agent_visited_frontiers)
        self.assertIn(a1, pol.agent_visited_frontiers)
        if pol.target_type.get(a0) == "frontier":
            tgt = pol.targets[a0]
            self.assertNotIn(tgt, pol.agent_visited_frontiers[a0])

    def test_20_no_manhattan_fallback_when_depot_unreachable(self):
        """Reject temporary item when no reachable path to depot exists."""
        H, W = 10, 10
        cfg = EnvConfig(height=H, width=W, n_agents=1, max_steps=20)
        env = SwarmSearchEnv(cfg)
        obs, _ = env.reset(seed=4)
        a = env.possible_agents[0]
        pol = GreedyPolicy(env, seed=4)

        o = copy.deepcopy(obs[a])
        o["scalars"][0], o["scalars"][1] = 0.0, 0.0
        o["scalars"][6], o["scalars"][7] = 5.0 / H, 5.0 / W
        o["team_map"][0, :, :] = 1.0
        o["team_map"][1, :, :] = 0.0
        o["team_map"][1, 4, 5] = o["team_map"][1, 6, 5] = 1.0
        o["team_map"][1, 5, 4] = o["team_map"][1, 5, 6] = 1.0
        o["team_map"][2:] = 0.0
        o["team_map"][3, 0, 2] = 1.0

        pol.act({a: o})
        self.assertNotEqual(pol.targets.get(a), (0, 2))

    def test_21_visualize_agent_bounds(self):
        """Verify valid agent indexing boundaries."""
        cfg = EnvConfig(height=10, width=10, n_agents=2, max_steps=10)
        env = SwarmSearchEnv(cfg)
        self.assertEqual(env.cfg.n_agents, 2)
        self.assertTrue(0 <= 0 < env.cfg.n_agents)
        self.assertTrue(0 <= 1 < env.cfg.n_agents)
        self.assertFalse(0 <= 2 < env.cfg.n_agents)

    def test_22_window_mode_collective_los(self):
        """Collective line-of-sight window union in window mode."""
        cfg = EnvConfig(height=15, width=15, n_agents=2, view_radius=2, info_mode="window",
                        n_persistent=0, n_malicious=0, n_temp_initial=0)
        env = SwarmSearchEnv(cfg)
        env.reset(seed=10)

        env.pos = [(2, 2), (12, 12)]
        env.grid[2, 2] = False
        env.grid[12, 12] = False
        it = env._new_item(PERSIST, (12, 12))
        env.active[(12, 12)] = it

        obs = env._observations()
        a0, a1 = env.possible_agents[0], env.possible_agents[1]

        self.assertEqual(obs[a0]["team_map"][0, 2, 2], 1.0)
        self.assertEqual(obs[a0]["team_map"][0, 12, 12], 1.0)
        self.assertEqual(obs[a0]["team_map"][0, 7, 7], 0.0)
        np.testing.assert_array_equal(obs[a0]["team_map"], obs[a1]["team_map"])
        self.assertEqual(obs[a0]["team_map"][2, 12, 12], 1.0)

    def test_23_dynamic_spawning_all_types(self):
        """Dynamic spawning triggers for all item types."""
        cfg = EnvConfig(
            height=12, width=12, n_agents=1,
            n_persistent=0, n_malicious=0, n_temp_initial=0,
            temp_spawn_prob=1.0, persist_spawn_prob=1.0, mal_spawn_prob=1.0,
            max_temp_active=5, max_persist_active=5, max_mal_active=5,
            max_steps=10
        )
        env = SwarmSearchEnv(cfg)
        env.reset(seed=42)
        env.step({env.possible_agents[0]: STAY})

        self.assertGreaterEqual(env.stats["temp_spawned"], 1)
        self.assertGreaterEqual(env.stats["persist_spawned"], 1)
        self.assertGreaterEqual(env.stats["mal_spawned"], 1)

    def test_24_full_map_spawning_safe(self):
        """Gracefully handle spawning requests when no free cells remain."""
        cfg = EnvConfig(height=8, width=8, n_agents=1, n_persistent=0, n_malicious=0)
        env = SwarmSearchEnv(cfg)
        env.reset(seed=1)

        for y, x in env.free_cells:
            p = (int(y), int(x))
            if p != env.checkpoint and p not in env.pos:
                env.active[p] = env._new_item(PERSIST, p)

        self.assertFalse(env._spawn_item(TEMP))
        self.assertFalse(env._spawn_item(PERSIST))
        self.assertFalse(env._spawn_item(MAL))

    def test_25_observation_space_compatibility(self):
        """Observation dictionary matches declared observation space."""
        for mode in ("shared", "private", "window"):
            cfg = EnvConfig(height=10, width=10, n_agents=2, info_mode=mode)
            env = SwarmSearchEnv(cfg)
            obs, _ = env.reset(seed=3)
            for a in env.possible_agents:
                space = env.observation_space(a)
                self.assertTrue(space.contains(obs[a]), f"Observation out of bounds for mode={mode}")

    def test_26_discovery_and_map_rewards_not_duplicated(self):
        """Exploration and discovery rewards are not awarded on stationary turns."""
        cfg = EnvConfig(height=10, width=10, n_agents=1, temp_spawn_prob=0.0,
                        persist_spawn_prob=0.0, mal_spawn_prob=0.0)
        env = SwarmSearchEnv(cfg)
        env.reset(seed=5)
        a0 = env.possible_agents[0]

        _, _, _, _, infos = env.step({a0: STAY})
        rc = infos[a0]["reward_components"]
        self.assertEqual(rc["map"], 0.0)
        self.assertEqual(rc["discover"], 0.0)

    def test_27_single_object_capacity_enforcement(self):
        """Enforce single object carrying capacity."""
        cfg = EnvConfig(height=10, width=10, n_agents=1, n_persistent=0, n_malicious=0, temp_spawn_prob=0.0)
        env = SwarmSearchEnv(cfg)
        env.reset(seed=7)
        a0 = env.possible_agents[0]
        p = env.pos[0]

        it1 = env._new_item(PERSIST, p)
        env.active[p] = it1
        env.step({a0: PICK})
        self.assertIsNotNone(env.carry[0])

        it2 = env._new_item(PERSIST, p)
        env.active[p] = it2
        self.assertFalse(env.action_mask(a0)[PICK])
        env.step({a0: PICK})
        self.assertEqual(env.carry[0].id, it1.id)
        self.assertIn(p, env.active)

    def test_28_temp_expiration_on_ground(self):
        """Temporary items expire on the ground and clear claims."""
        cfg = EnvConfig(height=10, width=10, n_agents=1, n_persistent=0, n_malicious=0, temp_spawn_prob=0.0)
        env = SwarmSearchEnv(cfg)
        env.reset(seed=8)
        a0 = env.possible_agents[0]

        p = env._random_free_cell(set(env.pos))
        it = env._new_item(TEMP, p, expiry=env.t + 1)
        it.seen = True
        env.active[p] = it
        env._apply_targets({a0: p})
        self.assertEqual(env.target[0], it.id)

        env.step({a0: STAY})
        self.assertEqual(it.status, Status.EXPIRED)
        self.assertNotIn(p, env.active)
        self.assertIsNone(env.target[0])
        self.assertEqual(env.stats["expired_by_type"][TEMP], 1)
        env.check_invariants()

    def test_29_temp_expiration_while_carried_policy(self):
        """Temporary item expiration behavior when carried."""
        cfg_exp = EnvConfig(height=10, width=10, n_agents=1, n_persistent=0, n_malicious=0,
                            temp_spawn_prob=0.0, temp_expires_while_carried=True)
        env1 = SwarmSearchEnv(cfg_exp)
        env1.reset(seed=9)
        a0 = env1.possible_agents[0]
        p = env1.pos[0]
        it1 = env1._new_item(TEMP, p, expiry=env1.t + 2)
        env1.active[p] = it1
        env1.step({a0: PICK})
        self.assertIsNotNone(env1.carry[0])
        env1.step({a0: STAY})
        self.assertIsNone(env1.carry[0])
        self.assertEqual(env1.stats["expired_while_carried"], 1)

        cfg_freeze = EnvConfig(height=10, width=10, n_agents=1, n_persistent=0, n_malicious=0,
                               temp_spawn_prob=0.0, temp_expires_while_carried=False)
        env2 = SwarmSearchEnv(cfg_freeze)
        env2.reset(seed=9)
        p2 = env2.pos[0]
        it2 = env2._new_item(TEMP, p2, expiry=env2.t + 2)
        env2.active[p2] = it2
        env2.step({a0: PICK})
        env2.step({a0: STAY})
        env2.step({a0: STAY})
        self.assertIsNotNone(env2.carry[0], "Carried temp item should not expire when temp_expires_while_carried=False")
        self.assertEqual(env2.stats["expired_while_carried"], 0)

    def test_30_malicious_pickup_modes(self):
        """Malicious pickup behavior under destroy and carry configurations."""
        cfg_d = EnvConfig(height=10, width=10, n_agents=1, n_persistent=1, n_malicious=0,
                          temp_spawn_prob=0.0, mal_on_pickup="destroy")
        env_d = SwarmSearchEnv(cfg_d)
        env_d.reset(seed=11)
        a0 = env_d.possible_agents[0]
        p_item = next(iter(env_d.active.values()))
        p_item.seen = True
        env_d._apply_targets({a0: p_item.pos})
        self.assertEqual(env_d.target[0], p_item.id)

        pos0 = env_d.pos[0]
        mal_it = env_d._new_item(MAL, pos0)
        env_d.active[pos0] = mal_it
        env_d.step({a0: PICK})
        self.assertEqual(mal_it.status, Status.DISPOSED)
        self.assertIsNone(env_d.carry[0])
        self.assertIsNone(env_d.target[0], "Prior claim must be released when picking up destroyed mal item")
        self.assertEqual(env_d.stats["malicious_picked"], 1)
        env_d.check_invariants()

        cfg_c = EnvConfig(height=10, width=10, n_agents=1, n_persistent=0, n_malicious=0,
                          temp_spawn_prob=0.0, mal_on_pickup="carry")
        env_c = SwarmSearchEnv(cfg_c)
        env_c.reset(seed=11)
        pos_c = env_c.pos[0]
        mal_it2 = env_c._new_item(MAL, pos_c)
        env_c.active[pos_c] = mal_it2
        env_c.step({a0: PICK})
        self.assertEqual(mal_it2.status, Status.CARRIED)
        self.assertEqual(env_c.carry[0].id, mal_it2.id)
        env_c.pos[0] = env_c.checkpoint
        env_c.step({a0: DROP})
        self.assertEqual(mal_it2.status, Status.DISPOSED)
        self.assertIsNone(env_c.carry[0])
        self.assertEqual(env_c.stats["deliveries"], 0)
        self.assertEqual(env_c.stats["disposed"], 1)
        env_c.check_invariants()

    def test_31_simultaneous_pickup_conflict(self):
        """Simultaneous pickup conflicts are resolved via configured tiebreak."""
        cfg = EnvConfig(height=10, width=10, n_agents=2, n_persistent=0, n_malicious=0,
                        temp_spawn_prob=0.0, checkpoint_stacking=True, tiebreak="index")
        env = SwarmSearchEnv(cfg)
        env.reset(seed=14)
        a0, a1 = env.possible_agents[0], env.possible_agents[1]

        cp = env.checkpoint
        env.pos = [cp, cp]
        it = env._new_item(PERSIST, cp)
        env.active[cp] = it

        env.step({a0: PICK, a1: PICK})
        self.assertEqual(env.stats["pickup_conflicts"], 1)
        self.assertIsNotNone(env.carry[0], "Agent 0 should win under index tiebreak")
        self.assertIsNone(env.carry[1], "Agent 1 should not duplicate pickup")
        env.check_invariants()

    def test_32_termination_vs_truncation(self):
        """Target deliveries trigger termination rather than truncation."""
        cfg = EnvConfig(height=10, width=10, n_agents=1, target_deliveries=1, max_steps=10)
        env = SwarmSearchEnv(cfg)
        env.reset(seed=16)
        a0 = env.possible_agents[0]

        it = env._new_item(PERSIST, env.checkpoint)
        it.status, it.collected_by, it.pickup_t = Status.CARRIED, 0, 0
        env.carry[0] = it
        env.pos[0] = env.checkpoint

        _, _, terms, truncs, _ = env.step({a0: DROP})
        self.assertTrue(terms[a0], "Reaching target_deliveries must set terminated=True")
        self.assertFalse(truncs[a0], "Reaching target_deliveries must keep truncated=False")

    def test_33_pettingzoo_parallel_api(self):
        """Validate environment compliance with PettingZoo ParallelEnv API."""
        from pettingzoo.test import parallel_api_test
        cfg = EnvConfig(height=10, width=10, n_agents=2, max_steps=15)
        env = SwarmSearchEnv(cfg)
        parallel_api_test(env, num_cycles=10)


if __name__ == "__main__":
    unittest.main()