from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HV1LocoManipPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
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


@configclass
class HV1LocoManipV3PPORunnerCfg(HV1LocoManipV2PPORunnerCfg):
    """V3 (HiWET Stage-1 robustification) — same PPO knobs as V2.

    The architectural changes are in the env config: observation history
    on the policy group (history_length=5, auto-flattened) and two new scalar
    commands (body_height, waist_regularization). No custom ActorCritic
    needed — RSL-RL sees the wider flat observation tensor and the MLP
    handles it.
    """

    experiment_name = "hv1_locomanip_v3_flat"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}


@configclass
class HV1LocoManipV4PPORunnerCfg(HV1LocoManipV3PPORunnerCfg):
    """V4 (HiWET KMP-residual) — same PPO knobs as V3, shorter horizon.

    The actor now outputs a residual on top of a frozen KMP MLP, not a full
    joint posture. Expected convergence is ~5-8k iters (vs V3's 30-40k from
    scratch) because the kinematic search has already been solved offline.
    """

    experiment_name = "hv1_locomanip_v4_flat"
    max_iterations = 15000
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}


@configclass
class HV1LocoManipV5HPPORunnerCfg(HV1LocoManipV2PPORunnerCfg):
    """V5-H (HiWET hierarchical) — Stage 2 commander, V4 frozen inside.

    Stage 2 trains FROM SCRATCH (V4 weights stay frozen). Action space is
    only 19-D (paper Eq. 3), observation is small (proprio + world pose +
    world EE + mask). A small MLP suffices.

    Architecture override: smaller hidden dims [256, 128, 64] since the
    output is 19-D and the input dimension is ~70 (vs V4's 121-per-step ×
    5-history = 605). Same PPO knobs as V2 elsewhere.

    No `obs_groups` change vs V2 — `policy` → actor, `critic` → critic.
    The `v4_actor` group exists for the env's internal use (read by the
    Stage 2 action class to run frozen V4 inference) and is NOT mapped to
    actor or critic; RSL-RL doesn't see it.
    """

    experiment_name = "hv1_locomanip_v5h_flat"
    max_iterations = 10000
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class HV1LocoManipV5PPORunnerCfg(HV1LocoManipV4PPORunnerCfg):
    """V5 (world-frame EE tracking) — warm-start from V4 24k.

    Same PPO knobs as V4. The architectural changes are env-side: world-frame
    EE commands, distance curriculum, navigation-progress reward, new
    pelvis-to-target observation terms. Existing 28-D actor output (KMP
    residual) is unchanged; the observation dimension grows by 6 (two arms ×
    3 = Δx_b, Δy_b, ‖xy‖), which the MLP absorbs automatically — but a fresh
    actor head must be initialized at warm-start since the input dim shifted.

    Use rsl-rl's `resume_path` + `--checkpoint model_24000.pt` from the V4
    run. RSL-RL drops the optimizer state and re-inits the first-layer
    weights if the obs dim doesn't match — this works because the V4 policy
    is good enough that the actor's residual + the body-frame KMP arm pose
    will still produce sensible postures while PPO re-trains the navigation
    skill on top.

    `max_iterations` 20000: V4 converged at 24k, V5 needs ~8-15k extra for
    long-range navigation per the V5 plan memory.
    """

    experiment_name = "hv1_locomanip_v5_flat"
    max_iterations = 20000
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
