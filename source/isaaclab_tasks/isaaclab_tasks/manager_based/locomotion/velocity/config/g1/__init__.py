# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Legs-G1-29Dof-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_legs_29dof_env_cfg:G1FlatLegs29DofEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLegs29DofPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Legs-G1-29Dof-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_legs_29dof_env_cfg:G1FlatLegs29DofEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLegs29DofPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Legs-G1-29Dof-Clean-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_legs_29dof_clean_env_cfg:G1FlatLegs29DofCleanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLegs29DofCleanPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Legs-G1-29Dof-Clean-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_legs_29dof_clean_env_cfg:G1FlatLegs29DofCleanEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLegs29DofCleanPPORunnerCfg",
    },
)


# Same clean env as above, but trained WITH left-right symmetry augmentation to fix the
# random one-sided-turn / one-sided-strafe handedness. Warm-startable from the clean run
# (only the algorithm differs); the exported policy is deploy-identical (Play task and the
# deploy manifest are unchanged — reuse the -Clean-Play-v0 task to visualize it).
gym.register(
    id="Isaac-Velocity-Flat-Legs-G1-29Dof-Clean-Sym-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_legs_29dof_clean_env_cfg:G1FlatLegs29DofCleanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLegs29DofCleanSymmetryPPORunnerCfg",
    },
)
