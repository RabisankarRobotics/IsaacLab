"""HV1 body-frame locomanipulation task: walk + per-hand EE pose tracking."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Tracking-LocoManip-HV1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:HV1LocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManip-HV1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:HV1LocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipPPORunnerCfg",
    },
)

# V2 (Stage 5): waist in action, torso IMU in obs, asymmetric critic.
gym.register(
    id="Isaac-Tracking-LocoManipV2-HV1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v2_env_cfg:HV1LocoManipV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManipV2-HV1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v2_env_cfg:HV1LocoManipV2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV2PPORunnerCfg",
    },
)

# V3 (HiWET Stage-1 robustification): obs history + body_height + α_t commands.
gym.register(
    id="Isaac-Tracking-LocoManipV3-HV1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v3_env_cfg:HV1LocoManipV3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManipV3-HV1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v3_env_cfg:HV1LocoManipV3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV3PPORunnerCfg",
    },
)

# V4 (HiWET KMP-residual): actor outputs residual on top of frozen KMP MLP.
gym.register(
    id="Isaac-Tracking-LocoManipV4-HV1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v4_env_cfg:HV1LocoManipV4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV4PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManipV4-HV1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v4_env_cfg:HV1LocoManipV4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV4PPORunnerCfg",
    },
)

# V5 (HiWET world-frame): EE targets sampled in env-local world, episode-static.
# Couples gait + reach — robot must navigate pelvis to reach far targets.
# Reuses V4's KMP via runtime world→body transform inside the action class.
gym.register(
    id="Isaac-Tracking-LocoManipV5-HV1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v5_env_cfg:HV1LocoManipV5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV5PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManipV5-HV1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v5_env_cfg:HV1LocoManipV5EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV5PPORunnerCfg",
    },
)

# V5-H (HiWET hierarchical): Stage 2 commander trains on world-frame EE
# targets, outputs Stage 1's command vector. Stage 1 (V4) is loaded JIT
# inside the action class and frozen.
gym.register(
    id="Isaac-Tracking-LocoManipV5H-HV1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v5h_env_cfg:HV1LocoManipV5HEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV5HPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManipV5H-HV1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_v5h_env_cfg:HV1LocoManipV5HEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1LocoManipV5HPPORunnerCfg",
    },
)
