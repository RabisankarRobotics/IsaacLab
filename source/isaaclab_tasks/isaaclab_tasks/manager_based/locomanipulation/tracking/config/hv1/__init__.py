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
