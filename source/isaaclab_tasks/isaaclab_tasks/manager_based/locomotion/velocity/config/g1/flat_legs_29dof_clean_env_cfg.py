# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clean-slate flat locomotion for the Unitree G1 29-DOF, legs-only.

This is a deliberate reset of ``flat_legs_29dof_env_cfg.py``. That sibling
config accumulated ~8 custom gait-shaping reward terms (``knee_too_straight``,
``knee_stance_extension``, ``base_height``, ``foot_clearance``,
``feet_airtime_variance``, ``feet_lateral_clearance``, two ``stand_still``
terms) that directly clamp the knee and pelvis height. They fight each other and
fight the natural gait, so every retrain traded one artifact for another — a
Groucho crouch (knee too bent) for a locked-straight knee (ankle-only shuffle)
and back again.

The stock IsaacLab G1 flat walk (``Isaac-Velocity-Flat-G1-v0``) produces a
natural, human-like gait with **none** of those terms. The knee bend simply
EMERGES: it is the lowest-torque way to track the velocity command without
hitting a joint limit or falling. So this file keeps the deploy-critical
architecture from the sibling config — **legs-only 12-joint action**,
**asymmetric (deploy-safe) actor-critic**, **29-DOF model**, **arm-aware
observation** — but adopts the STOCK minimal reward recipe and adds nothing.

First-run scope ("clean walk first"):
* Commands are **forward-biased** (vx 0→1, small strafe, turn), not full
  omnidirectional — a forward-biased command set yields the cleanest gait.
* Domain randomization is **OFF** (no arm motion, no pushes, no payload/CoM).
  The arms stay parked at their default by their stiff PD actuators; the
  arm-aware observation is still present (just ≈0), so the observation is the
  same 79-dim vector as the sibling task. A follow-up run can re-enable
  omnidirectional commands + upper-body/base DR, **warm-started from this
  checkpoint** (identical obs/action dims).

Task id: ``Isaac-Velocity-Flat-Legs-G1-29Dof-Clean-v0``.
"""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# Reuse the deploy-critical action + observation architecture from the sibling
# config (single source of truth for the 12-dim legs-only action and the 79-dim
# asymmetric, arm-aware observation). We do NOT import its rewards — those are
# the gait-shaping terms this reset drops.
from .flat_legs_29dof_env_cfg import G1Legs29DofActionsCfg, G1Legs29DofObservationsCfg
from .rough_env_cfg import G1Rewards, G1RoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_29DOF_CFG  # isort: skip


@configclass
class G1FlatLegs29DofCleanEnvCfg(G1RoughEnvCfg):
    """Legs-only G1 29-DOF flat walk using the STOCK reward recipe (no custom
    gait shaping). Natural knee bend is left to emerge on its own."""

    # rewards stays the plain stock G1Rewards (minimal-model-only terms are
    # disabled in __post_init__ below). actions/observations bring the
    # legs-only + deploy-safe asymmetric architecture.
    rewards: G1Rewards = G1Rewards()
    actions: G1Legs29DofActionsCfg = G1Legs29DofActionsCfg()
    observations: G1Legs29DofObservationsCfg = G1Legs29DofObservationsCfg()

    def __post_init__(self):
        # post init of parent (G1RoughEnvCfg) — rough velocity env on the
        # minimal model; we swap the model + flatten the terrain below.
        super().__post_init__()

        # ----- 29-DOF model (instead of the minimal model) -----
        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # G1_29DOF_CFG ships with contact sensors disabled (manipulation focus);
        # the velocity env attaches a foot contact sensor, so enable the API.
        self.scene.robot.spawn.activate_contact_sensors = True

        # ----- Flatten the terrain (mirror of G1FlatEnvCfg) -----
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # ----- Drop the minimal-model-only reward terms -----
        # These reference joint names that don't exist on the 29-DOF model
        # (fingers ``*_zero.._six_joint``, ``torso_joint``, minimal
        # ``*_elbow_pitch/_roll_joint``). The arms/waist are held at their
        # default by stiff PD, so no deviation reward is needed for them.
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None
        self.rewards.joint_deviation_fingers = None
        # Keep joint_deviation_hip — its names (``*_hip_yaw/_roll_joint``) are
        # valid on the 29-DOF model and it keeps the hips square for a straight,
        # natural walk.

        # ----- Stock FLAT reward tuning (mirror of G1FlatEnvCfg) -----
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # NOTE: intentionally NO knee / base_height / foot_clearance /
        # air-time-variance / lateral-clearance / stand-still terms. The natural
        # knee bend emerges from velocity tracking + the torque/limit penalties,
        # exactly as in the stock walk. If a SPECIFIC defect appears later (feet
        # scissoring, scuffing, idle drift), add back ONE targeted term at a
        # time and retrain — do not front-load the gait shaping again.

        # ----- Commands: forward-biased (stock flat), NOT omnidirectional -----
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ----- Terminations: kill the legs-only "collapse and survive" optimum -----
        # A legs-only walker can fold into an L-shape (legs flat on the floor,
        # torso upright) where the torso never contacts the ground, so the stock
        # base_contact termination never fires and the policy farms reward while
        # collapsed. Terminate when the pelvis drops well below nominal or the
        # base tilts too far.
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.5},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.0},  # ~57 deg of base tilt
        )

        # ----- Domain randomization: OFF for this clean run -----
        # The rough parent already disables push_robot / add_base_mass /
        # base_com; we do NOT re-enable them, and add NO arm-motion DR. The arms
        # stay parked at default (arm-aware obs still present, just ≈0). Harden
        # with DR + omnidirectional commands in a follow-up run warm-started from
        # this checkpoint.


@configclass
class G1FlatLegs29DofCleanEnvCfg_PLAY(G1FlatLegs29DofCleanEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable observation noise for a clean demo
        self.observations.policy.enable_corruption = False
