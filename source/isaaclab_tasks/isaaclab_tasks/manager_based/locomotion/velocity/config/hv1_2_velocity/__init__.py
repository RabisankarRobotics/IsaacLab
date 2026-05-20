"""HV1.2 walking task (Phase 2: leg-only policy, upper body pinned, non-zero velocity commands)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-HV1_2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HV1_2VelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1_2VelocityFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-HV1_2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:HV1_2VelocityFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HV1_2VelocityFlatPPORunnerCfg",
    },
)
