from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TahitiC1VelocityFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 24 rollout steps × 4096 envs × 20k iters = ~2B env-steps — plenty for a
    # 12-DoF flat-ground bipedal walker with mild DR to converge.
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "tahiti_c1_velocity_flat"

    policy = RslRlPpoActorCriticCfg(
        # 0.5 initial exploration std. Enough range to find a stepping gait
        # under Phase-1 zero-command standing without letting action_std grow
        # into the runaway regime we saw on HV1.2.
        init_noise_std=0.5,
        # Observation normalization on both actor and critic — canonical PPO
        # stability mechanism, essential when joint_vel obs can spike during
        # near-fall / recovery transients.
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.002 entropy_coef — dropped from 0.005 after the 15k-iter fresh
        # run showed action_std GROWING from 0.5 → 1.10 (PPO widening its
        # distribution because no deterministic policy scored above noise).
        # DR is now narrower (±10 % actuator, ±0.7 m/s pushes) so the extra
        # exploration that justified 0.005 isn't needed; 0.002 pressures the
        # policy to commit once tracking finds a peak.
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 1e-3 — the standard locomotion starting LR. Adaptive schedule will
        # dial it up or down based on the observed KL.
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
