"""H1_2 standing task on flat ground (Isaac Lab manager-based).

* Policy controls only LEGS (12 of 21 DoF) via JointPositionAction. Torso is
  pinned to 0 via PD (its target written at reset by `pin_torso_target_reset`).
  Earlier runs let the policy act on torso, which caused unprompted yaw drift.
* Arm DoFs are PD-driven to a per-episode-randomized target, written into the
  articulation buffer by a custom event term. Sample ranges restrict arms to
  forward/down/up + small left/right + small twist + elbow bend — no backward.
* All velocity commands are zero, so the inherited tracking rewards peak when
  the robot stands still.
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

from isaaclab_assets import H1_2_STAND_CFG  # isort: skip

from . import mdp as custom_mdp


# ---- joint-name regexes used in multiple places ----
LEG_JOINTS = [
    "^(left|right)_hip_(yaw|pitch|roll)_joint$",
    "^(left|right)_knee_joint$",
    "^(left|right)_ankle_(pitch|roll)_joint$",
]
TORSO_JOINT = ["torso_joint"]
LEG_TORSO_JOINTS = LEG_JOINTS + ["^torso_joint$"]  # kept only for the dof_pos_limits reward
ARM_JOINTS = [
    "^(left|right)_shoulder_(pitch|roll|yaw)_joint$",
    "^(left|right)_elbow_pitch_joint$",
]
ARM_JOINT_NAMES_FLAT = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
]

# Arm-target sample ranges (lo, hi) per joint. Joint-angle conventions for H1_2:
#   shoulder_pitch: 0 = arm straight down, negative = arm forward/up, positive = arm BACKWARD.
#                  Cap upper bound at the default (0.4) so the policy never sees a backward pose.
#   shoulder_roll:  left  -> positive = arm out to the left side (away from torso)
#                   right -> negative = arm out to the right side (away from torso)
#                  Restricted to "small" lateral motion.
#   shoulder_yaw:   upper-arm twist. Restricted to small twist either way.
#   elbow_pitch:    0 = straight, positive = bent. Natural bend allowed.
ARM_TARGET_RANGES: dict[str, tuple[float, float]] = {
    # left arm
    "left_shoulder_pitch_joint":  (-2.5,  0.4),    # up / forward / down, no backward
    "left_shoulder_roll_joint":   (-0.2,  0.8),    # small left
    "left_shoulder_yaw_joint":    (-0.5,  0.5),    # small twist
    "left_elbow_pitch_joint":     ( 0.0,  2.0),    # straight to bent
    # right arm (shoulder_roll mirrored)
    "right_shoulder_pitch_joint": (-2.5,  0.4),
    "right_shoulder_roll_joint":  (-0.8,  0.2),    # small right
    "right_shoulder_yaw_joint":   (-0.5,  0.5),
    "right_elbow_pitch_joint":    ( 0.0,  2.0),
}

# Fixed arm pose for playback when "no hand movement" is desired.
# Values match the default_joint_pos in H1_2_STAND_CFG (slightly-forward rest pose).
# Edit these to pose the arms differently while the legs still hold balance.
FIXED_ARM_POSE: dict[str, tuple[float, float]] = {
    "left_shoulder_pitch_joint":  (0.4, 0.4),
    "left_shoulder_roll_joint":   (0.0, 0.0),
    "left_shoulder_yaw_joint":    (0.0, 0.0),
    "left_elbow_pitch_joint":     (0.3, 0.3),
    "right_shoulder_pitch_joint": (0.4, 0.4),
    "right_shoulder_roll_joint":  (0.0, 0.0),
    "right_shoulder_yaw_joint":   (0.0, 0.0),
    "right_elbow_pitch_joint":    (0.3, 0.3),
}


@configclass
class H1_2StandActionsCfg:
    """Policy actions act ONLY on legs (12 of 21 joints). Torso is pinned at 0."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class H1_2StandObservationsCfg:
    """Observations match the layout used in the Isaac Gym version where reasonable."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        arm_target_delta = ObsTerm(
            func=custom_mdp.arm_target_delta,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES_FLAT, preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class H1_2StandEventCfg(EventCfg):
    """Inherits the velocity-task events (physics_material, reset_base,
    reset_robot_joints, push_robot, etc.) and adds:
      * `pin_torso_target_reset` — writes torso joint_pos_target = 0 on reset
        (since the policy no longer drives this joint).
      * `randomize_arm_targets_reset` / `randomize_arm_targets_interval` —
        per-episode + mid-episode random arm pose.
    """

    pin_torso_target_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": {"torso_joint": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot", joint_names=TORSO_JOINT, preserve_order=True),
        },
    )
    randomize_arm_targets_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": ARM_TARGET_RANGES,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_TARGET_RANGES.keys()), preserve_order=True),
        },
    )
    randomize_arm_targets_interval = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "position_range": ARM_TARGET_RANGES,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_TARGET_RANGES.keys()), preserve_order=True),
        },
    )


@configclass
class H1_2StandRewardsCfg:
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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_TORSO_JOINTS)},
    )


@configclass
class H1_2StandFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: H1_2StandActionsCfg = H1_2StandActionsCfg()
    observations: H1_2StandObservationsCfg = H1_2StandObservationsCfg()
    rewards: H1_2StandRewardsCfg = H1_2StandRewardsCfg()
    events: H1_2StandEventCfg = H1_2StandEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------------- scene: flat plane, no height scanner ----------------
        self.scene.robot = H1_2_STAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # disable terrain curriculum (we override curriculum to None below)
        self.curriculum.terrain_levels = None

        # ---------------- commands: ALL ZERO (= stand still) ------------------
        # tracking_lin_vel and tracking_ang_vel rewards then peak at base_vel=0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False

        # ---------------- events inherited from parent: keep, but trim --------
        # base_external_force_torque only references "base" which doesn't exist
        # on H1_2 (the root link is "pelvis"). Disable it.
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None
        self.events.base_com = None
        # tighten reset: smaller initial position perturbation
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        # mid-episode pushes still useful for robustness
        self.events.push_robot.interval_range_s = (5.0, 8.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}}

        # ---------------- terminations: only "pelvis" contact -----------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"

        # ---------------- runtime ---------------------------------------------
        self.episode_length_s = 20.0
        self.decimation = 4   # policy at 50 Hz when sim.dt = 0.005


@configclass
class H1_2StandFlatEnvCfg_PLAY(H1_2StandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # No hand movement during playback:
        #   * `randomize_arm_targets_interval` disabled -> no mid-episode resampling.
        #   * `randomize_arm_targets_reset` uses degenerate (v, v) ranges so the arms
        #     hold a fixed pose every episode. Edit FIXED_ARM_POSE to change it.
        self.events.randomize_arm_targets_interval = None
        self.events.randomize_arm_targets_reset.params["position_range"] = FIXED_ARM_POSE
