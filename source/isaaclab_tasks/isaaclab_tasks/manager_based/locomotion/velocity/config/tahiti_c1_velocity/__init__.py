"""Tahiti C1 velocity-tracking (walking) task on flat ground."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Tahiti_C1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TahitiC1VelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TahitiC1VelocityFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tahiti_C1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TahitiC1VelocityFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TahitiC1VelocityFlatPPORunnerCfg",
    },
)
