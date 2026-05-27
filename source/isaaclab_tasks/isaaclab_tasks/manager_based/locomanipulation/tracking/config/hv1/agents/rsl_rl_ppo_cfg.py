from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HV1LocoManipPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 200
    experiment_name = "hv1_locomanip_flat"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
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
        entropy_coef=0.001,
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
class HV1LocoManipV2PPORunnerCfg(HV1LocoManipPPORunnerCfg):
    """Stage-5 runner: asymmetric actor-critic.

    The V2 env exposes two observation groups (`policy`, `critic`);
    `obs_groups` tells RSL-RL which one feeds the actor and which feeds
    the critic. Without this mapping, RSL-RL would fall back to feeding
    `policy` to both, defeating the asymmetric setup.
    """

    experiment_name = "hv1_locomanip_v2_flat"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
