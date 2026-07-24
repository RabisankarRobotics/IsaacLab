# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class H1_2FlatLegsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the H1_2 legs-only walk (arm-motion + EE-payload DR)."""

    # Asymmetric actor-critic: the actor reads the deploy-safe `policy` group (no
    # base_lin_vel), the critic reads the privileged `critic` group. Without this
    # mapping RSL-RL feeds `policy` to both, ignoring the critic group.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    num_steps_per_env = 24
    # Harder than a bare flat walk (omnidirectional + live arm-motion DR + payload +
    # pushes), so it needs samples to converge AND to harden against the arm
    # disturbance. A basic gait appears by ~2-3k iters; robustness keeps improving.
    # Watch TensorBoard and stop at the reward/episode-length plateau — an upper bound.
    max_iterations = 10000
    save_interval = 200
    experiment_name = "h1_2_flat_legs"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
