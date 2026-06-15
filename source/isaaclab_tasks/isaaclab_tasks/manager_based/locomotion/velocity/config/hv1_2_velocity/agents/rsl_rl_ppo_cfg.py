from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HV1_2VelocityFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Walking needs more samples and iterations than standing.
    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 200
    experiment_name = "hv1_2_velocity_flat"

    policy = RslRlPpoActorCriticCfg(
        # Softer exploration: previous run hit action_std=2.66 at iter 6673 with
        # asymmetric walking and no yaw tracking. Starting at 0.5 (vs 0.8) gives
        # PPO less std-headroom to grow into, paired with reduced entropy_coef
        # below to discourage the runaway-exploration regime.
        init_noise_std=0.5,
        # Observation normalization on actor + critic is the canonical PPO
        # stability mechanism for high-dim obs. Was OFF in the run that diverged;
        # raw joint_vel spikes during near-falls fed unnormalized into the actor's
        # first linear layer → activation blow-up → late-game collapse.
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
        # Halved 0.01 → 0.005. Previous run let action_std grow to 2.66 because
        # the entropy bonus made high-noise policies look attractive even when
        # they tracked velocity poorly. Lower coef means PPO drops std as soon
        # as a more deterministic policy gives better velocity reward.
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Halved from 1e-3 — slower per-step gradient drift, reduces the chance
        # of a single bad batch pushing the actor past safe weight ranges.
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        # Tighter target KL → adaptive LR scheme is less likely to crank lr up
        # mid-training in response to noisy KL estimates.
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
