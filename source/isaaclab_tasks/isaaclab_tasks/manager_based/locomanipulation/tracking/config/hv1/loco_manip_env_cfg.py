"""HV1 body-frame loco-manipulation task.

Single policy that simultaneously:
  * Tracks a base velocity command (locomotion) — inherited from Stage 3.
  * Tracks a per-hand SE(3) target pose in the **robot base (pelvis) frame**
    for the left and right `wrist_yaw_link` — Stage 4 addition.

Action space: 12 leg joints + 14 arm joints  =  26 (waist + neck stay pinned).
Command space: [v_x, v_y, w_z, EE_L(7), EE_R(7)]
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.flat_env_cfg import (
    ARM_JOINT_NAMES,
    HV1ActionsCfg,
    HV1EventCfg,
    HV1ObservationsCfg,
    HV1RewardsCfg,
    HV1VelocityFlatEnvCfg,
    LEG_JOINTS,
)


# ---- end-effector links ----------------------------------------------------
LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"


# ---- actions: legs + arms in one PD-target action -------------------------
@configclass
class HV1LocoManipActionsCfg(HV1ActionsCfg):
    """Override Stage-3 action to include arms.

    Single `JointPositionActionCfg` so the policy emits one Δ-vector that is
    re-ordered to match the robot's actuated joints under the hood.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS + ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


# ---- events: keep waist + neck pinned, drop the arm pin in __post_init__ --
@configclass
class HV1LocoManipEventCfg(HV1EventCfg):
    """Inherits Stage-3 events; `pin_arms_target_reset` is removed at runtime
    in `HV1LocoManipEnvCfg.__post_init__` since the policy now controls the
    arms directly."""

    pass


# ---- observations: add per-hand EE pose commands --------------------------
@configclass
class HV1LocoManipObservationsCfg(HV1ObservationsCfg):
    @configclass
    class PolicyCfg(HV1ObservationsCfg.PolicyCfg):
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_ee_pose"}
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_ee_pose"}
        )

    policy: PolicyCfg = PolicyCfg()


# ---- rewards: keep all Stage-3 rewards, add EE tracking -------------------
@configclass
class HV1LocoManipRewardsCfg(HV1RewardsCfg):
    # ---- left hand --------------------------------------------------------
    left_ee_pos_tracking = RewTerm(
        func=reach_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE_BODY),
            "command_name": "left_ee_pose",
        },
    )
    left_ee_pos_tracking_fine = RewTerm(
        func=reach_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE_BODY),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )
    left_ee_orient_tracking = RewTerm(
        func=reach_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE_BODY),
            "command_name": "left_ee_pose",
        },
    )

    # ---- right hand -------------------------------------------------------
    right_ee_pos_tracking = RewTerm(
        func=reach_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE_BODY),
            "command_name": "right_ee_pose",
        },
    )
    right_ee_pos_tracking_fine = RewTerm(
        func=reach_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE_BODY),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )
    right_ee_orient_tracking = RewTerm(
        func=reach_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE_BODY),
            "command_name": "right_ee_pose",
        },
    )


# ---- env config ------------------------------------------------------------
@configclass
class HV1LocoManipEnvCfg(HV1VelocityFlatEnvCfg):
    actions: HV1LocoManipActionsCfg = HV1LocoManipActionsCfg()
    observations: HV1LocoManipObservationsCfg = HV1LocoManipObservationsCfg()
    rewards: HV1LocoManipRewardsCfg = HV1LocoManipRewardsCfg()
    events: HV1LocoManipEventCfg = HV1LocoManipEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # Arms are now part of the action space — drop the Stage-3 PD pin.
        self.events.pin_arms_target_reset = None

        # Body-frame EE workspace boxes (pelvis frame), derived from HV1 URDF:
        #   shoulder origin (pelvis frame) ≈ (0, ±0.19, 0.55)
        #   max kinematic reach            = 0.79 m
        #   default wrist (joints=0)       = (0.30, ±0.26, 0.24)
        # Box corners stay ≤77% of max reach → safe dexterous workspace.
        self.commands.left_ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=LEFT_EE_BODY,
            resampling_time_range=(2.0, 4.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.10, 0.50),
                pos_y=(0.05, 0.45),
                pos_z=(0.00, 0.55),
                roll=(-0.2, 0.2),
                pitch=(-0.2, 0.2),
                yaw=(-0.2, 0.2),
            ),
        )
        self.commands.right_ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=RIGHT_EE_BODY,
            resampling_time_range=(2.0, 4.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.10, 0.50),
                pos_y=(-0.45, -0.05),
                pos_z=(0.00, 0.55),
                roll=(-0.2, 0.2),
                pitch=(-0.2, 0.2),
                yaw=(-0.2, 0.2),
            ),
        )

        # Boost locomotion so EE rewards don't dominate it 2:1.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # Soften EE tanh: wider sweet spot, lower peak → kills hand vibration.
        self.rewards.left_ee_pos_tracking_fine.weight = 1.5
        self.rewards.left_ee_pos_tracking_fine.params["std"] = 0.10
        self.rewards.right_ee_pos_tracking_fine.weight = 1.5
        self.rewards.right_ee_pos_tracking_fine.params["std"] = 0.10

        # Wake feet back up — air-time reward only fires above threshold.
        self.rewards.feet_air_time.params["threshold"] = 0.2

        # ---------------- Stage-4 upper-body domain randomization -----------
        # Unitree G1/H1 pattern: ALL body-level DR on `torso_link`, none on
        # pelvis. The upper body is where COM lives and where real disturbances
        # actually act (chest pushes, drag, top-mounted equipment). Retargeting
        # the inherited `base_*` events overrides the Stage-3 pelvis defaults.

        # Mass: models battery / sensor / payload variance on the torso.
        # Softened ±2 → ±1 kg for resume-from-checkpoint stability.
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)

        # COM: models uneven internal mass distribution of the torso.
        # Softened ±2 → ±1 cm (less destabilizing balance origin shift).
        self.events.base_com.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com.params["com_range"] = {
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01),
        }

        # External wrench: sustained per-episode push/drag on the chest.
        # Force already softened to ±5 N. Torque softened ±2 → ±1 Nm.
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)

        # Wrist payload mass: softened 0.5 → 0.2 kg. Keeps a small amount of
        # payload variance for sim2real without destroying Stage-4 EE tracking.
        # Bump back to 0.5 when starting Stage 5 (where payload is the point).
        self.events.add_wrist_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_BODY, RIGHT_EE_BODY]),
                "mass_distribution_params": (0.0, 0.2),
                "operation": "add",
                "distribution": "uniform",
            },
        )

        # Slightly longer episode to give EE rewards time to accumulate
        # signal across multiple command samples.
        self.episode_length_s = 14.0


@configclass
class HV1LocoManipEnvCfg_PLAY(HV1LocoManipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Mix all behaviors so playback shows walking + reaching together.
        self.commands.base_velocity.rel_standing_envs = 0.30
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)

        # Slower EE resampling so each target is held long enough to see reach.
        self.commands.left_ee_pose.resampling_time_range = (4.0, 4.0)
        self.commands.right_ee_pose.resampling_time_range = (4.0, 4.0)

        # Visible disturbances during play so external-force effect is obvious.
        # push_robot = sudden velocity kick (visible as instant slide/sway).
        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}
        }

        # Big sustained chest wrench per episode → robot visibly leans into it.
        self.events.base_external_force_torque.params["force_range"] = (-5.0, 5.0)
        self.events.base_external_force_torque.params["torque_range"] = (-1.0, 1.0)

        # Shorter episodes so resets (= new wrench sample) happen more often.
        # self.episode_length_s = 8.0
