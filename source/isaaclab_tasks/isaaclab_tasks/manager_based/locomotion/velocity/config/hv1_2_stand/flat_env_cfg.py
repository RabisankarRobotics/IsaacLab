"""HV1.2 standing task on flat ground (Isaac Lab manager-based).

Phase 1: policy controls ONLY the 12 leg joints via JointPositionAction. The
upper body (3-DoF waist + 14-DoF arms + 3-DoF head = 20 DoF) is pinned to
its default pose at reset; the implicit-actuator PD holds it there.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
)

from isaaclab_assets import HV1_2_CFG  # isort: skip

from . import mdp as custom_mdp


# ---- joint-name regexes -------------------------------------------------
LEG_JOINTS = [
    "^(left|right)_hip_(yaw|pitch|roll)_joint$",
    "^(left|right)_knee_joint$",
    "^(left|right)_ankle_(pitch|roll)_joint$",
]
WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
HEAD_JOINT_NAMES = ["head_pitch_joint", "head_yaw_joint"]  # 2-DoF head (no head_roll in without_ee URDF)
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Default upper-body pose: zero everywhere (matches HV1_2_CFG.init_state for
# waist/wrist/head; arms also pinned to 0 for Phase 1 simplicity).
WAIST_TARGETS = {n: (0.0, 0.0) for n in WAIST_JOINT_NAMES}
HEAD_TARGETS = {n: (0.0, 0.0) for n in HEAD_JOINT_NAMES}
ARM_TARGETS_PIN = {
    "left_shoulder_pitch_joint": (0.4, 0.4),
    "left_shoulder_roll_joint":  (0.0, 0.0),
    "left_shoulder_yaw_joint":   (0.0, 0.0),
    "left_elbow_joint":          (0.3, 0.3),
    "left_wrist_roll_joint":     (0.0, 0.0),
    "left_wrist_pitch_joint":    (0.0, 0.0),
    "left_wrist_yaw_joint":      (0.0, 0.0),
    "right_shoulder_pitch_joint": (0.4, 0.4),
    "right_shoulder_roll_joint":  (0.0, 0.0),
    "right_shoulder_yaw_joint":   (0.0, 0.0),
    "right_elbow_joint":          (0.3, 0.3),
    "right_wrist_roll_joint":     (0.0, 0.0),
    "right_wrist_pitch_joint":    (0.0, 0.0),
    "right_wrist_yaw_joint":      (0.0, 0.0),
}


@configclass
class HV1_2StandActionsCfg:
    """Policy actions act ONLY on legs (12 of 32 joints)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class HV1_2StandObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HV1_2StandEventCfg(EventCfg):
    """Inherits velocity-task events and adds reset-time pinning for waist /
    arms / head joints (since the policy no longer drives them)."""

    pin_waist_target_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": WAIST_TARGETS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINT_NAMES, preserve_order=True),
        },
    )
    pin_arms_target_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": ARM_TARGETS_PIN,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_TARGETS_PIN.keys()), preserve_order=True),
        },
    )
    pin_head_target_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": HEAD_TARGETS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=HEAD_JOINT_NAMES, preserve_order=True),
        },
    )


@configclass
class HV1_2StandRewardsCfg:
    """Rewards: tracking velocity commands of zero == standing still."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    is_alive = RewTerm(func=mdp.is_alive, weight=0.2)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    # Keep hip yaw/roll near defaults so the legs don't cross inward
    # (without this the policy converges to a feet-touching stance).
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["^(left|right)_hip_(yaw|roll)_joint$"],
            )
        },
    )


@configclass
class HV1_2StandFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: HV1_2StandActionsCfg = HV1_2StandActionsCfg()
    observations: HV1_2StandObservationsCfg = HV1_2StandObservationsCfg()
    rewards: HV1_2StandRewardsCfg = HV1_2StandRewardsCfg()
    events: HV1_2StandEventCfg = HV1_2StandEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------------- scene: flat plane, no height scanner ----------------
        self.scene.robot = HV1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # ---------------- commands: all zero (= stand still) ------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False

        # ---------------- events inherited from parent: trim ------------------
        # parent expects a "base" body; HV1.2's root is "pelvis" — disable
        # these terms to avoid lookup failures.
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None
        self.events.base_com = None

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.push_robot.interval_range_s = (5.0, 8.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}}

        # ---------------- terminations: only "pelvis" contact -----------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"

        # ---------------- runtime ---------------------------------------------
        self.episode_length_s = 20.0
        self.decimation = 4  # policy at 50 Hz when sim.dt = 0.005


@configclass
class HV1_2StandFlatEnvCfg_PLAY(HV1_2StandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
