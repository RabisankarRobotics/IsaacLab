# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legs-only omnidirectional flat walk for the Unitree H1_2 (fingerless 27-DoF).

Trained with arm-motion domain randomization + a +3 kg EE-payload mass DR so the
locomotion policy stays balanced under the CoM shifts a later loco-manipulation
policy will create. See ``flat_legs_env_cfg.py`` for the full design notes.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Legs-H1_2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_legs_env_cfg:H1_2FlatLegsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1_2FlatLegsPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Legs-H1_2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_legs_env_cfg:H1_2FlatLegsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1_2FlatLegsPPORunnerCfg",
    },
)
