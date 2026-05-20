"""H1_2 standing task with randomized arm pose (Isaac Lab manager-based)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Stand-H1_2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1_2StandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1_2StandFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Stand-H1_2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1_2StandFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1_2StandFlatPPORunnerCfg",
    },
)
