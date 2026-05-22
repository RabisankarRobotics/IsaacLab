"""HV1.2 walking task on flat ground (Isaac Lab manager-based).

Phase 2: policy controls ONLY the 12 leg joints. Upper body (3-DoF waist +
14-DoF arms + 3-DoF head) is held at the default pose by the implicit-actuator
PD, with targets re-asserted at every reset via custom event terms.

Differences vs. hv1_2_stand:
  * Non-zero velocity command ranges (forward 0..1 m/s, yaw ±1 rad/s).
  * Feet-air-time and feet-slide rewards (foot contact sensor on ankle_roll_link).
  * Stronger joint_deviation penalties on upper body to actively resist torque
    feedback from leg swings.
  * Larger action scale (0.5) so the policy has range to lift the feet.
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
HEAD_JOINT_NAMES = ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Default upper-body pose (matches HV1_2_CFG.init_state).
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
class HV1_2VelocityActionsCfg:
    """Policy actions on legs only (12 of 32 joints), with a larger scale than
    the standing task so it can actually lift its feet."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class HV1_2VelocityObservationsCfg:
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
class HV1_2VelocityEventCfg(EventCfg):
    """Inherits velocity-task events and adds reset-time pinning for
    waist / arms / head joints (since the policy no longer drives them)."""

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
class HV1_2VelocityRewardsCfg:
    """Velocity-tracking rewards (walking)."""

    # ---- tracking (restored to 1.0 — strong velocity command pulls the
    #      robot forward, which is impossible while balancing on one leg) ----
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
    # ---- gait: keep the bipedal single-stance reward (it got the robot
    # stepping) but dial weights down to G1-style values and ADD gait-shape
    # terms — variance penalty (anti-asymmetry) and foot clearance reward
    # (smooth swing). Reference: Unitree G1 rough_env_cfg uses weight=0.25.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,  # was 1.0 — let velocity tracking shape the gait, this
                      # is now a tiebreaker for single-stance.
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,  # was -1.0; G1 uses -0.1. Still anti-shuffle.
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    # NEW: direct asymmetric-gait penalty (the yoga-walk fix).
    # Penalizes variance of last_air_time and last_contact_time across the
    # two feet. Zero when both feet step in the same rhythm.
    feet_airtime_variance = RewTerm(
        func=custom_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )
    # NEW: rewards swing-foot clearance — encourages a clean foot-lift arc
    # rather than dragging. Target = 0.10 m above ground (HV1.2 foot
    # thickness is ~0.04 m, so this is ~6 cm of clearance). Only active
    # while the foot is moving (tanh gate on xy-velocity).
    foot_clearance = RewTerm(
        func=custom_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "target_height": 0.10,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    # ---- stability ----
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # One-sided L1 "no-squat" penalty: zero when pelvis is at or above 0.92 m,
    # linear in shortfall below. L1 (vs the previous L2 at weight -50) avoids
    # the value-function blow-up when a falling env hits shortfall ≈ 0.8 m.
    #   normal walk at 0.93 → 0 penalty
    #   crouch at 0.84    → shortfall 0.08, penalty -0.8 (bigger than tracking gain)
    #   fall at 0.10      → shortfall 0.82, penalty -8.2 (large but bounded)
    base_height_below = RewTerm(
        func=custom_mdp.base_height_below_target_l1,
        weight=-10.0,
        params={"target_height": 0.92},
    )
    # ---- effort / smoothness ----
    # Backed off from -0.005 — paired with the height-penalty rework, that
    # was too many tight new constraints at once and PPO went chaotic.
    # -0.003 is between original (-0.002) and the over-aggressive (-0.005).
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    # ---- safety ----
    is_alive = RewTerm(func=mdp.is_alive, weight=0.15)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_(pitch|roll)_joint$"])},
    )
    # Penalize the OUTCOME (feet too close in yaw-frame Y) rather than the
    # MEANS (hip deviation). Lets the policy use hip_roll freely for balance
    # while strictly preventing leg crossing. One-sided: zero when clear.
    feet_lateral_clearance = RewTerm(
        func=custom_mdp.feet_lateral_distance_clearance,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "min_distance": 0.18,  # ~half of standing 0.34 m separation
        },
    )
    # Softer hip-deviation now that feet_lateral_clearance does the heavy
    # lifting — kept at low weight as a secondary nudge.
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_(yaw|roll)_joint$"])},
    )


@configclass
class HV1_2VelocityFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: HV1_2VelocityActionsCfg = HV1_2VelocityActionsCfg()
    observations: HV1_2VelocityObservationsCfg = HV1_2VelocityObservationsCfg()
    rewards: HV1_2VelocityRewardsCfg = HV1_2VelocityRewardsCfg()
    events: HV1_2VelocityEventCfg = HV1_2VelocityEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------------- scene: flat plane, no height scanner ----------------
        self.scene.robot = HV1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # ---------------- commands: walking ranges ---------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)

        # ---------------- domain randomization on the base link --------------
        # Parent points these at a body named "base" which doesn't exist on
        # HV1.2 (root is "pelvis"). Re-target them and dial the ranges down
        # to "small but useful" so the policy is robust across mass / COM
        # variation without being asked to track an impossible target.

        # ±3 kg around the 83 kg total — ≈ 3.6% body mass.
        self.events.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)

        # ±3 cm horizontal, ±1 cm vertical pelvis COM offset.
        self.events.base_com.params["asset_cfg"].body_names = "pelvis"
        self.events.base_com.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.01, 0.01),
        }

        # Re-target the reset-time external force/torque term (default ranges
        # are 0/0, so this just keeps it from erroring out — push_robot does
        # the actual mid-episode perturbation).
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "pelvis"

        # Widen ground friction randomization — single-bucket friction is too
        # narrow for "stable across environments". Static 0.4..1.2 covers
        # smooth indoor floor to grippy rubber; dynamic 0.3..1.0 similarly.
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
        # push robot every 8-12 s for robustness
        self.events.push_robot.interval_range_s = (8.0, 12.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}

        # ---------------- terminations: only "pelvis" contact -----------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"

        # ---------------- runtime ---------------------------------------------
        self.episode_length_s = 20.0
        self.decimation = 4  # policy at 50 Hz when sim.dt = 0.005


@configclass
class HV1_2VelocityFlatEnvCfg_PLAY(HV1_2VelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # For inspection, hold the command at a steady forward walk.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
