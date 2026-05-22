"""HV1 unified standing + walking task on flat ground.

A single env teaches both behaviors:
  * `rel_standing_envs = 0.25` → 25% of envs get a zero velocity command and
    must hold the default pose (gated by `stand_still_joint_deviation_l1`).
  * The remaining 75% get a non-zero command and must walk to track it.

Policy controls only the 12 leg joints. Upper body (3-DoF waist + 14-DoF arms
+ 2-DoF neck) is pinned at default via PD with reset-time target events.

Differences vs. the HV1.2 walking config (calibrated to HV1's actual geometry):
  * Foot-contact link: `*_ankle_pitch_link` (HV1) vs `*_ankle_roll_link` (HV1.2).
    HV1's ankle chain is knee → ankle_roll → ankle_pitch (leaf), so ankle_pitch
    is the foot proper. HV1.2's chain is inverted (knee → ankle_pitch → ankle_roll,
    leaf), so ankle_roll is its foot. Targeting the wrong link makes
    feet_air_time and feet_slide silently read zero.
  * `base_height_below.target_height`: 0.92 → 0.89 m  (HV1 spawns 3 cm lower)
  * `feet_lateral_clearance.min_distance`: 0.18 → 0.30 m  (HV1 stance 1.84× wider)
  * `flat_orientation_l2.weight`: -1.0 → -2.0  (HV1 torso is +70% heavier)
  * `push_robot.velocity_range`: ±0.5 → ±0.3 m/s  (smaller impulses on top-heavy body)
  * 2-DoF neck pin (yaw + pitch) instead of HV1.2's 3-DoF head.
  * Added `stand_still_joint_deviation_l1` reward + `rel_standing_envs = 0.05`.
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

from isaaclab_assets import HV1_CFG  # isort: skip

from . import mdp as custom_mdp


# ---- joint-name regexes -------------------------------------------------
LEG_JOINTS = [
    "^(left|right)_hip_(yaw|pitch|roll)_joint$",
    "^(left|right)_knee_joint$",
    "^(left|right)_ankle_(pitch|roll)_joint$",
]
WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
NECK_JOINT_NAMES = ["neck_yaw_joint", "neck_pitch_joint"]
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Default upper-body pose (matches HV1_CFG.init_state.joint_pos).
WAIST_TARGETS = {n: (0.0, 0.0) for n in WAIST_JOINT_NAMES}
NECK_TARGETS = {n: (0.0, 0.0) for n in NECK_JOINT_NAMES}
ARM_TARGETS_PIN = {
    "left_shoulder_pitch_joint": (0.3, 0.3),
    "left_shoulder_roll_joint":  (0.0, 0.0),
    "left_shoulder_yaw_joint":   (0.0, 0.0),
    "left_elbow_joint":          (0.3, 0.3),
    "left_wrist_roll_joint":     (0.0, 0.0),
    "left_wrist_pitch_joint":    (0.0, 0.0),
    "left_wrist_yaw_joint":      (0.0, 0.0),
    "right_shoulder_pitch_joint": (0.3, 0.3),
    "right_shoulder_roll_joint":  (0.0, 0.0),
    "right_shoulder_yaw_joint":   (0.0, 0.0),
    "right_elbow_joint":          (0.3, 0.3),
    "right_wrist_roll_joint":     (0.0, 0.0),
    "right_wrist_pitch_joint":    (0.0, 0.0),
    "right_wrist_yaw_joint":      (0.0, 0.0),
}


@configclass
class HV1ActionsCfg:
    """Policy acts on the 12 leg joints. Upper body is pinned via PD targets."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class HV1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
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
class HV1EventCfg(EventCfg):
    """Inherits velocity-task events and pins waist / arms / neck at reset."""

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
    pin_neck_target_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": NECK_TARGETS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=NECK_JOINT_NAMES, preserve_order=True),
        },
    )


@configclass
class HV1RewardsCfg:
    """Unified rewards: velocity tracking handles walking, stand_still handles
    standing. Both share stability / safety penalties."""

    # ---- velocity tracking ---------------------------------------------
    # At zero command, exp(-||vel||^2) peaks when robot is stationary → also
    # the "stand still" objective. No mode switch needed.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ---- gait (active only when ||cmd_xy|| > 0.1) ----------------------
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_pitch_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_pitch_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_pitch_link"),
        },
    )

    # ---- stand-still (active only when ||cmd_xy|| < 0.06) --------------
    # Penalizes leg joints deviating from default when commanded to stand.
    stand_still_legs = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
        },
    )

    # ---- stability (always-on) -----------------------------------------
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # HV1 torso is +70% heavier than HV1.2 → stronger tilt penalty.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # HV1 spawn 0.95 m; preserve 6 cm headroom → target 0.89 m.
    base_height_below = RewTerm(
        func=custom_mdp.base_height_below_target_l1,
        weight=-10.0,
        params={"target_height": 0.89},
    )

    # ---- effort / smoothness -------------------------------------------
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.003)

    # ---- safety --------------------------------------------------------
    # Smaller alive bonus so "stand still 20 s" isn't a sufficient strategy.
    is_alive = RewTerm(func=mdp.is_alive, weight=0.05)
    # Bounded termination penalty — -200 was a value-loss blow-up risk
    # once action_std grew and noisy rollouts hit fall states.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_(pitch|roll)_joint$"])},
    )
    # HV1 default sep 0.42 m → min_distance 0.30 m leaves 12 cm of compression room.
    feet_lateral_clearance = RewTerm(
        func=custom_mdp.feet_lateral_distance_clearance,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_pitch_link"),
            "min_distance": 0.30,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_(yaw|roll)_joint$"])},
    )


@configclass
class HV1VelocityFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: HV1ActionsCfg = HV1ActionsCfg()
    observations: HV1ObservationsCfg = HV1ObservationsCfg()
    rewards: HV1RewardsCfg = HV1RewardsCfg()
    events: HV1EventCfg = HV1EventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------------- scene: flat plane, no height scanner --------------
        self.scene.robot = HV1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # ---------------- commands: walk + stand mix ------------------------
        # Only 5% of envs get a zero-velocity command. Keep some standing
        # examples so the policy learns the cmd≈0 gating, but make walking
        # the dominant training distribution — otherwise the policy finds a
        # local optimum of "always stand and collect alive bonus".
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)

        # ---------------- domain randomization on pelvis --------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)

        self.events.base_com.params["asset_cfg"].body_names = "pelvis"
        self.events.base_com.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.01, 0.01),
        }

        self.events.base_external_force_torque.params["asset_cfg"].body_names = "pelvis"

        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        # Top-heavy body → gentler pushes than HV1.2 (±0.5 → ±0.3).
        self.events.push_robot.interval_range_s = (8.0, 12.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}

        # ---------------- terminations --------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"

        # ---------------- runtime -------------------------------------------
        self.episode_length_s = 20.0
        self.decimation = 4  # policy at 50 Hz with sim.dt = 0.005


@configclass
class HV1VelocityFlatEnvCfg_PLAY(HV1VelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # Inspect with a steady forward walk; toggle rel_standing_envs to see
        # standing behavior.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
