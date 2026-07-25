# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legs-only omnidirectional flat-ground WALK for the Unitree H1_2 (fingerless 27-DoF).

This is the locomotion half of an H1_2 loco-manipulation stack. It fuses three
proven ingredients:

1. **The G1 "clean" legs-only walk recipe** (``config/g1/flat_legs_29dof_clean_env_cfg.py``):
   phase-clock gait shaping (``feet_gait`` + ``gait_phase``, frozen at idle),
   the velocity-command curriculum, ``stand_still`` anti-vibration, the
   feet-lateral-distance fall fix, and swing-knee shaping. This is what makes the
   gait clean, rhythmic, omnidirectional, and quiet-when-standing — exactly the
   two behaviors the user asked for (walk every direction; no idle vibration).

2. **Arm-motion domain randomization** (``config/h1_2_stand``): the 14 arm joints
   are PD-driven to a per-episode-randomized pose AND re-randomized mid-episode,
   so the legs learn to hold balance under the CoM shifts a manipulation policy
   will later create (both hands forward, out to the side, bent, etc.).

3. **A +3 kg EE-payload mass DR**: up to +3 kg is added to each wrist/hand link at
   startup, so the walk stays stable when the robot is carrying a picked object.

Design (same deploy-safe contract as the G1 clean task):

* **Legs-only action** (12 joints). The torso + 14 arm joints are NEVER actioned;
  they are held by their implicit PD — the torso at its default, the arms at the
  randomized target. A manipulation policy owns the arms at deploy.
* **Asymmetric, deploy-safe actor-critic**: the actor reads only hardware-measurable
  terms (IMU gyro, IMU tilt, command, gait clock, the 12 leg encoders, last action,
  and — arm-aware — the 15 upper-body encoders). ``base_lin_vel`` is critic-only.
* **No observation history** — the policy phase-locks to the 2-value gait clock.

Policy observation layout (77), in concat order — a deploy runner must rebuild
this exactly:
    base_ang_vel (3) | projected_gravity (3) | velocity_commands (3) |
    gait_phase (2) |
    joint_pos legs (12) | joint_vel legs (12) |
    last_action (12) | upper_body_joint_pos (15) | upper_body_joint_vel (15)

Action: ``target_q_leg = raw_action * 0.25 + default_q_leg`` (scale 0.25).

Task id: ``Isaac-Velocity-Flat-Legs-H1_2-v0`` (+ ``-Play-v0``).

NOTE — several geometry-dependent constants below are FIRST-CUT ESTIMATES for
H1_2 and are flagged ``# TUNE``; verify each from a short PLAY rollout before a
long training run (H1_2 is much taller/heavier than G1):
    base_height target, termination minimum_height, feet_lateral_distance
    min_distance, feet_clearance target_height.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from . import mdp as custom_mdp

##
# Pre-defined configs
##
from isaaclab_assets import H1_2_CFG  # isort: skip


# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------

# Joints driven by the policy (legs only, 12).
LEG_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

# Torso — 1 joint, held at default by its PD (never actioned).
TORSO_JOINT_NAMES = ["torso_joint"]

# Arm joints — 14 (7 per arm). PD-driven to a randomized target here; owned by a
# manipulation policy at deploy. Fingers are fixed in the walk URDF (fingerless 27-DoF).
ARM_JOINT_NAMES = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]

# Upper body the walker OBSERVES (arm-aware): torso + arms = 15 joints.
UPPER_BODY_JOINT_NAMES = TORSO_JOINT_NAMES + ARM_JOINT_NAMES

# Gait clock period (s). Shared by the feet_gait reward and the gait_phase obs.
GAIT_PERIOD = 0.8

# Arm-target sample ranges (lo, hi) per joint — the poses the arms are randomized
# to at reset + on interval. Same conventions/values as the validated h1_2_stand
# task, EXTENDED to the 3 extra distal joints this fingerless-27DoF asset keeps
# articulated (elbow_roll, wrist_pitch, wrist_yaw).
#   shoulder_pitch: 0 = arm straight down, negative = forward/up, positive = BACKWARD.
#                   Upper bound capped at the default (0.4) so no backward pose.
#   shoulder_roll:  left  -> positive = out to the left; right -> negative = out to the right.
#   shoulder_yaw:   small twist either way.
#   elbow_pitch:    0 = straight, positive = bent.
#   elbow_roll / wrist_*: small ranges — enough to shift the hand CoM, not to flail.
ARM_TARGET_RANGES: dict[str, tuple[float, float]] = {
    "left_shoulder_pitch_joint":  (-2.5,  0.4),
    "left_shoulder_roll_joint":   (-0.2,  0.8),
    "left_shoulder_yaw_joint":    (-0.5,  0.5),
    "left_elbow_pitch_joint":     ( 0.0,  2.0),
    "left_elbow_roll_joint":      (-0.5,  0.5),
    "left_wrist_pitch_joint":     (-0.3,  0.3),
    "left_wrist_yaw_joint":       (-0.3,  0.3),
    "right_shoulder_pitch_joint": (-2.5,  0.4),
    "right_shoulder_roll_joint":  (-0.8,  0.2),
    "right_shoulder_yaw_joint":   (-0.5,  0.5),
    "right_elbow_pitch_joint":    ( 0.0,  2.0),
    "right_elbow_roll_joint":     (-0.5,  0.5),
    "right_wrist_pitch_joint":    (-0.3,  0.3),
    "right_wrist_yaw_joint":      (-0.3,  0.3),
}


# ---------------------------------------------------------------------------
# Command: uniform velocity + a `limit_ranges` field holding the full (Phase-3) range
# ---------------------------------------------------------------------------
@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """``UniformVelocityCommandCfg`` plus ``limit_ranges``, the full (Phase-3)
    command range. ``mdp.stand_to_walk_command_curriculum`` steps ``ranges`` up to
    these values (its ``*_full`` params mirror them), and PLAY sets ``ranges`` to
    ``limit_ranges`` directly to demo at full speed."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class H1_2WalkSceneCfg(InteractiveSceneCfg):
    """Flat ground + H1_2 27-DoF + a full-body contact sensor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Full H1_2 articulation (leg + soft upper-body gains, bent-arm pose, contact
    # sensors on). Only prim_path is overridden.
    robot: ArticulationCfg = H1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Full-body contact sensor: feet gate the gait/slide/clearance terms, the rest
    # gate the undesired-contact penalty. track_air_time is required by feet_gait.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@configclass
class CommandsCfg:
    base_velocity = UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        # START tiny — the curriculum grows these toward limit_ranges.
        ranges=UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2)
        ),
        # Caps the curriculum. Omnidirectional, forward-biased for a clean walk.
        limit_ranges=UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5)
        ),
    )


# ---------------------------------------------------------------------------
# Actions — legs only
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINT_NAMES, scale=0.25, use_default_offset=True
    )


# ---------------------------------------------------------------------------
# Observations — asymmetric, deploy-safe, arm-aware, gait-clock (no history)
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Actor group — only hardware-measurable terms. ORDER IS DEPLOY-CRITICAL
        (a deploy runner rebuilds this exact concatenation)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 2-value sin/cos gait clock (phase-lock reference for feet_gait). FROZEN AT
        # IDLE (custom_mdp._gait_phase_scalar): constant while standing, so it does
        # not drive the "parade" march. A deploy runner mirrors this frozen clock.
        gait_phase = ObsTerm(func=custom_mdp.gait_phase, params={"period": GAIT_PERIOD})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        # Arm-aware: observe the live upper-body joint state so the legs can
        # anticipate CoM shifts from arm motion (this is the whole reason the arms
        # are randomized). At deploy, feed the real arm/torso encoders here.
        upper_body_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        upper_body_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # OPTIONAL feed-forward: custom_mdp.arm_target_delta (commanded arm target minus
        # default) could be added here so the legs see the arm's intent BEFORE it moves.
        # Left OUT to keep the obs deploy-safe with just encoders; add if lookahead is needed.

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Privileged group: the actor's terms plus the true base_lin_vel."""

        def __post_init__(self):
            super().__post_init__()
            self.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards — the G1 clean recipe, adapted to H1_2 legs-only
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # -- task
    # weight raised 1.0 -> 1.5 (2026-07-25): translation must clearly out-reward the
    # safe "stand still" optimum. See the feet_air_time / feet_clearance change below.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base / smoothness (joint-wise terms scoped to the actuated legs so we never
    #    charge the policy for the PD holding the parked torso/arms)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2, weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    # )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    energy = RewTerm(
        func=custom_mdp.energy, weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )

    # -- posture
    # hip_roll strongly penalized (stops leg splay) but not so hard it chokes strafing;
    # hip_yaw lightly penalized so the policy can still use it to steer.
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    # TUNE: H1_2 pelvis rides ~1.0 m (init z 1.05, drops a little with the bent-knee
    # stance). Verify from a PLAY rollout — read the actual pelvis height and set this.
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 1.0})

    # -- feet / gait (the natural-walk drivers)
    gait = RewTerm(
        func=custom_mdp.feet_gait,
        weight=0.5,
        params={
            "period": GAIT_PERIOD,
            "offset": [0.0, 0.5],  # anti-phase (alternating steps)
            "threshold": 0.55,  # stance duty
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    # THE step-driver (added 2026-07-25, from the official H1/G1 recipe). Rewards
    # SINGLE-SUPPORT air time — it is exactly ZERO while both feet are planted
    # (single_stance requires one foot in contact) and only pays once the robot lifts
    # a foot, up to `threshold` s. This is what breaks the "stand perfectly still"
    # optimum: unlike feet_gait (which hands out ~55% just for matching the stance
    # phase) and the old feet_clearance (which paid MAX for a planted foot), a stander
    # earns 0 here. Command-gated, so it never forces stepping while idle.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    # L/R symmetry nudge: penalize variance of air/contact time across the two feet.
    air_time_variance = RewTerm(
        func=custom_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    # DISABLED 2026-07-25 — this reward multiplies the foot-height error by
    # tanh(foot_horizontal_speed), so a PLANTED foot (zero horizontal speed) zeroes
    # the penalty and collects the MAXIMUM reward (exp(0) = 1.0). It therefore paid
    # the robot full marks for NEVER lifting a foot, which is a direct driver of the
    # "stand perfectly still" collapse. The official H1/G1 configs don't use it at all
    # — feet_air_time (above) is the foot-lift driver. Re-enable ONLY if, once walking,
    # a PLAY rollout shows the feet dragging/scuffing (too little swing clearance), and
    # then prefer a swing-gated variant or swing_knee_flexion so a stander can't farm it.
    # feet_clearance = RewTerm(
    #     func=custom_mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.13,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     },
    # )

    # -- feet must not converge (the SIDEWAYS-WALK FALL fix). Self-collisions are OFF
    # and undesired_contacts excludes ankles, so nothing else stops the feet from
    # interpenetrating in Isaac. A one-sided lateral-distance hinge catches the whole
    # approach (a contact penalty would be flat until the feet already touch).
    feet_lateral_distance = RewTerm(
        func=custom_mdp.feet_lateral_distance_clearance,
        weight=-10.0,  # authority lives in the WEIGHT; keep min_distance near half the
        # natural stance. Move the weight (toward -20 if feet still converge, toward -5
        # if the strafe turns timid), NOT min_distance.
        params={
            # TUNE: ~half H1_2's natural stance (wider hips than G1). 0.20 is a first
            # cut; measure the true foot separation at hip_roll=0 in PLAY and set to ~0.5x.
            "min_distance": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # -- knee: NO explicit knee-flexion reward (deliberate, per design choice).
    # The knee angle is left to emerge NATURALLY from the gait clock (feet_gait),
    # velocity tracking, and the energy/smoothness penalties — so the bend scales
    # with the commanded speed instead of being pinned to a fixed "bent enough"
    # pose at every velocity. Forcing a swing-knee bend tends toward a high-stepping
    # / parade gait rather than a natural foot step, which is the opposite of the
    # goal here. The swing_knee_flexion_reward function is kept in mdp.py and can be
    # re-enabled ONLY as a last resort if a PLAY rollout shows hip circumduction (a
    # straight leg swung out in an arc to clear the foot — the "rounded step"); if
    # that appears, prefer first nudging feet_clearance / hip_roll deviation.
    # swing_knee_flexion = RewTerm(
    #     func=custom_mdp.swing_knee_flexion_reward,
    #     weight=0.6,
    #     params={
    #         "scale": 0.5,
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
    #             preserve_order=True,
    #         ),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["left_knee_joint", "right_knee_joint"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    # -- SOFT-LANDING / low ground-force pair (ported from the tuned tahiti_c1 recipe).
    # Two terms constraining DIFFERENT axes so a quiet, low-impact walk is learned:
    #
    #   (1) foot_contact_force  — REACTIVE cap on peak vertical GRF. mdp.contact_forces
    #       penalizes only the net foot force ABOVE `threshold`, so it never charges the
    #       policy for the force needed to SUPPORT the robot — only for slam/impact spikes.
    #       Physics anchor (H1_2 ≈ 67 kg ⇒ weight ≈ 660 N = 1 BW):
    #         * double-support standing ≈ 330 N / foot,
    #         * single-support stance   ≈ 660 N / foot (1 BW),
    #         * healthy human walk peaks ≈ 1.2–1.5 BW,
    #       so a 1000 N threshold (≈1.5 BW) sits ABOVE normal stance/walk (pays ~0) and
    #       bites only hard 2–4 BW stomps. TUNE: verify actual peak GRF in a PLAY/MuJoCo
    #       rollout and lower the threshold toward ~900 N if landings are still loud.
    foot_contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-2.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 1000.0,  # ≈1.5x body weight (TUNE from measured peak GRF)
        },
    )
    #   (2) foot_contact_velocity — UPSTREAM soft-landing shaper. Peak GRF ∝ m·Δv_z/Δt,
    #       so penalizing the foot's downward speed at touchdown cuts the impulse at its
    #       cause (a quiet landing), rather than only paying for it after impact like (1).
    #       Small weight so it shapes touchdown without freezing stance micro-motion.
    foot_contact_velocity = RewTerm(
        func=custom_mdp.foot_contact_velocity_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "force_threshold": 5.0,  # contact DETECTOR only (not a force cap)
        },
    )

    # -- idle: kill the "parade" march / standing vibration. Penalize leg deviation
    #    from the default stance while standing (command ~ 0). This is the primary
    #    driver of the "no vibration when standing" requirement.
    stand_still = RewTerm(
        func=custom_mdp.stand_still_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )

    # -- safety: penalize any non-foot body touching the ground
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"),
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Legs-only can fold into an L-shape (torso never hits the floor), so gate on
    # pelvis height + tilt rather than torso contact.
    # TUNE: minimum pelvis height. 0.6 is a first cut for H1_2 (init ~1.05); raise if
    # the robot survives in a deep crouch, lower if legit deep steps trip it.
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.6})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


# ---------------------------------------------------------------------------
# Events — physics DR + arm-motion DR + the +3 kg EE-payload DR
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    # +3 kg EE PAYLOAD DR — the "stable pick-and-place without affecting balance"
    # requirement. Up to +3 kg is added to EACH wrist/hand link at startup (the
    # fingerless walk asset merged the hand + finger inertia into *_wrist_yaw_link,
    # so this is the single rigid EE body). Modeling it per-hand covers the worst
    # case (a heavy object in each hand); halve the upper bound if only one shared
    # object is ever carried.
    add_ee_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_yaw_link"),
            "mass_distribution_params": (0.0, 3.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-0.5, 0.5)},
    )
    # Pin the TORSO target to its default at reset (the legs-only action term never
    # writes it, and the framework inits every target to 0). The arms are handled
    # separately by the randomizer below.
    hold_torso_target = EventTerm(
        func=custom_mdp.hold_joint_targets_at_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TORSO_JOINT_NAMES)},
    )
    # ARM-MOTION DR (reset): sample a fresh arm pose each episode. Declared AFTER
    # hold_torso_target so it owns the arm targets. This is what forces the legs to
    # balance a variety of static arm poses (both hands forward, out to the side, etc.).
    randomize_arm_targets_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": ARM_TARGET_RANGES,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )
    # ARM-MOTION DR (interval): re-randomize the arm target mid-episode, so the legs
    # must reject a LIVE, moving CoM shift (not just a static hold) — the disturbance
    # a manipulation policy actually creates while reaching.
    randomize_arm_targets_interval = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "position_range": ARM_TARGET_RANGES,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )
    # PUSH DR — starts at ZERO and is grown by ``custom_mdp.push_velocity_levels``
    # only once the robot survives full episodes.
    #
    # This must NOT be at full strength from iteration 0. push_by_setting_velocity
    # writes the root velocity directly, and a v m/s kick displaces the capture point
    # by v / sqrt(g / h_com). H1_2's CoM rides ~0.85 m, so 0.5 m/s => ~0.147 m — past
    # the foot edge, i.e. only a STEP recovers it. The same 0.5 m/s is survivable on
    # G1 (h_com ~0.6 m => 0.124 m) which is where this value was copied from, and
    # H1_2 also needs ~2.7x G1's ankle torque for the same tilt while running the
    # same 40 N·m/rad ankle stiffness. Measured on the 10k-iteration run: 100% of
    # terminations were bad_orientation, 79% in the 5-8 s window (pushes fire at
    # t = 5/10/15 s), mean pelvis tilt spiking 0.10 -> 0.27 rad within 0.8 s of a
    # push. Walk first, harden second.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


# ---------------------------------------------------------------------------
# Curriculum — grow the command range as tracking improves
# ---------------------------------------------------------------------------
@configclass
class CurriculumCfg:
    # FIXED 3-phase command schedule (ported from the tuned tahiti_c1 recipe), NOT the
    # performance-gated lin/ang_vel_cmd_levels. Those gates deadlocked the 10 000-iter
    # run: a robot that falls early can never clear a per-episode tracking-quality
    # threshold, so the command range stayed pinned at lin=0.1/ang=0.2 forever. A
    # fixed schedule can't stall — the command grows on a known iteration timeline, so
    # the robot always gets a reason to start stepping.
    #   Phase 1 (iter 0 .. stand):  zero command, 100% standing envs (learn to stand).
    #   Phase 2 (stand .. slow):    command x 0.3, 30% standing (slow walk).
    #   Phase 3 (>= slow):          full command, 10% standing.
    # *_full ranges MUST equal commands.base_velocity.limit_ranges below.
    command_phase = CurrTerm(
        func=custom_mdp.stand_to_walk_command_curriculum,
        params={
            "stand_until_iters": 2000,  # TUNE — user proposed 3000; 2000 is tahiti-proven
            "slow_until_iters": 5000,   # TUNE — Phase 2 spans iters 2000..5000
            "slow_scale": 0.3,
            "lin_vel_x_full": (-0.5, 1.0),  # == limit_ranges.lin_vel_x
            "lin_vel_y_full": (-0.5, 0.5),  # == limit_ranges.lin_vel_y
            "ang_vel_z_full": (-0.5, 0.5),  # == limit_ranges.ang_vel_z
            "rel_standing_envs_phase1": 1.0,
            "rel_standing_envs_phase2": 0.3,
            "rel_standing_envs_phase3": 0.1,
        },
    )
    # PUSH DISABLED 2026-07-25 — get the robot WALKING first; push is a disturbance to
    # reject AFTER it has a gait, and while it can't walk it just causes falls (it was
    # ~87% of terminations at 10k). push_robot stays at (0,0) (a true no-op: the event
    # does vel += 0), so no kick is ever applied. RE-ENABLE this term once a PLAY
    # rollout shows a stable walk, to harden disturbance rejection.
    # push_velocity_levels = CurrTerm(
    #     func=custom_mdp.push_velocity_levels,
    #     params={
    #         "term_name": "push_robot",
    #         "step": 0.05,
    #         "max_velocity": 0.5,
    #         "min_alive_frac": 0.8,
    #         "start_after_iters": 2000,  # == command_phase stand_until_iters
    #     },
    # )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@configclass
class H1_2FlatLegsEnvCfg(ManagerBasedRLEnvCfg):
    """Legs-only H1_2 27-DoF flat walk with arm-motion + EE-payload DR, deploy-safe."""

    scene: H1_2WalkSceneCfg = H1_2WalkSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 50 Hz control (decimation 4 x sim dt 0.005) — matches the 0.8 s gait period
        # (= 40 control steps) and a typical deploy CONTROL_DT.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # H1_2_CFG enables contact sensors; tick the sensor at the physics rate for
        # correct air-time.
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class H1_2FlatLegsEnvCfg_PLAY(H1_2FlatLegsEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # clean demo: no observation noise
        self.observations.policy.enable_corruption = False
        # keep arm randomization + EE payload ON so the demo shows the walk rejecting
        # arm motion and payload — the whole point of this task. Disable pushes only
        # (raise velocity_range here to whatever push level training actually reached
        # if you want to demo disturbance rejection).
        self.events.push_robot = None
        # play at the fully-grown command range (skip the curriculum ramp)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.command_phase = None
        # push_velocity_levels is disabled in the base cfg (push off until it walks)
