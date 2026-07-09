# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..symmetry import compute_symmetric_states


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 200
    experiment_name = "g1_rough"
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


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]


@configclass
class G1FlatLegs29DofPPORunnerCfg(G1FlatPPORunnerCfg):
    # Asymmetric actor-critic: the actor reads the deploy-safe `policy` group
    # (no base_lin_vel), the critic reads the privileged `critic` group.
    # Without this mapping RSL-RL feeds `policy` to both, ignoring the critic
    # group and defeating the asymmetric setup.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_legs_29dof"
        # Harder than stock flat G1 (omnidirectional + heavy upper-body /
        # push DR), so it needs more samples to converge AND to harden against
        # the arm-motion disturbance. A basic gait appears by ~2k iters;
        # robustness keeps improving past that. Watch TensorBoard and stop at
        # the reward/episode-length plateau — this is an upper bound, not a fixed run.
        self.max_iterations = 8000


@configclass
class G1FlatLegs29DofCleanPPORunnerCfg(G1FlatLegs29DofPPORunnerCfg):
    """Runner for the clean-slate legs-only walk (stock reward recipe, no gait
    shaping, forward-biased commands, DR off). Same asymmetric actor-critic and
    policy dims as the sibling; only the experiment name + iteration budget
    differ so its checkpoints/logs stay separate and it can seed the later
    DR/omnidirectional hardening run."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "g1_flat_legs_29dof_clean"
        # No DR and forward-biased commands — this is close to the stock flat G1
        # walk, which converges fast. A clean gait usually appears well before
        # 4k iters; stop at the reward/episode-length plateau.
        self.max_iterations = 4000


@configclass
class G1FlatLegs29DofCleanSymmetryPPORunnerCfg(G1FlatLegs29DofCleanPPORunnerCfg):
    """Clean legs-only walk WITH left-right symmetry augmentation.

    Identical env/reward/obs/action and network to the sibling — the ONLY change is
    that every PPO minibatch is augmented with its left-right mirror (2x), forcing the
    actor to be left-right symmetric. This cures the random "handedness" symmetry-break
    (turns/strafes one way but not the other; the working side flipping between runs)
    that no reward tweak could fix, and it evens out the "one leg more active" step
    asymmetry. Since only the algorithm changes, this WARM-STARTS from a clean-task
    checkpoint (``--resume``) and the exported policy is byte-for-byte deploy-identical
    (same 81-dim obs, same manifest). The mirror map is validated in ``symmetry.py``.
    """

    def __post_init__(self):
        super().__post_init__()

        # Same experiment_name as the sibling so `--resume` warm-starts from the
        # existing clean run; distinguish sym vs non-sym runs by timestamp.
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_symmetric_states,
        )
