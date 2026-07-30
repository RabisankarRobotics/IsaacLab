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
  terms (IMU gyro, IMU tilt, command, the 12 leg encoders, last action, and — arm-aware
  — the 15 upper-body encoders). ``base_lin_vel`` is critic-only.
* **No observation history and NO gait clock** (removed 2026-07-27 in the full alignment to
  the user's own proven MLP walker ``hv1_2_velocity``, which walks with plain instantaneous
  obs and no clock). Stepping is driven SOLELY by ``feet_air_time_positive_biped`` (the
  positive "swing carrot" both proven walkers use). The 5.6k/7.6k server runs proved the
  gait-clock ``feet_gait`` reward was a STANDING SUBSIDY (a planted foot farms ~55% of it for
  free), which — with a high ``alive`` — made standing the optimum and left the carrot (0
  until a step exists) unable to compete. The fix aligns the stepping-critical knobs to the
  walker: no gait reward/obs, ``alive`` 0.15->0.05, ``termination_penalty`` -100 restored,
  ``action_rate`` -0.01->-0.003, ``track_lin_vel`` ->1.5 (out-weighting the farmable yaw).

Policy observation layout (75), in concat order — a deploy runner must rebuild
this exactly:
    base_ang_vel (3) | projected_gravity (3) | velocity_commands (3) |
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
# manipulation policy at deploy. Official handless URDF: no hands, 7-DoF arm named
# shoulder_pitch/roll/yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw.
ARM_JOINT_NAMES = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]

# Upper body the walker OBSERVES (arm-aware): torso + arms = 15 joints.
UPPER_BODY_JOINT_NAMES = TORSO_JOINT_NAMES + ARM_JOINT_NAMES

# Gait clock period (s). Shared by the feet_gait/`contact` reward and the gait_phase obs
# (both restored 2026-07-26 for the official-recipe switch). 0.8 s = official h1_2 period.
GAIT_PERIOD = 0.8

# Arm-target sample ranges (lo, hi) per joint — the poses the arms are randomized
# to at reset + on interval. Same conventions/values as the validated h1_2_stand
# task, EXTENDED to the 3 extra distal joints this handless-27DoF asset keeps
# articulated (wrist_roll, wrist_pitch, wrist_yaw).
#   shoulder_pitch: 0 = arm straight down, negative = forward/up, positive = BACKWARD.
#                   Upper bound capped at the default (0.4) so no backward pose.
#   shoulder_roll:  left  -> positive = out to the left; right -> negative = out to the right.
#   shoulder_yaw:   small twist either way.
#   elbow:          0 = straight, positive = bent.
#   wrist_roll / wrist_*: small ranges — enough to shift the hand CoM, not to flail.
ARM_TARGET_RANGES: dict[str, tuple[float, float]] = {
    "left_shoulder_pitch_joint":  (-2.5,  0.4),
    "left_shoulder_roll_joint":   (-0.2,  0.8),
    "left_shoulder_yaw_joint":    (-0.5,  0.5),
    "left_elbow_joint":           ( 0.0,  2.0),
    "left_wrist_roll_joint":      (-0.5,  0.5),
    "left_wrist_pitch_joint":     (-0.3,  0.3),
    "left_wrist_yaw_joint":       (-0.3,  0.3),
    "right_shoulder_pitch_joint": (-2.5,  0.4),
    "right_shoulder_roll_joint":  (-0.8,  0.2),
    "right_shoulder_yaw_joint":   (-0.5,  0.5),
    "right_elbow_joint":          ( 0.0,  2.0),
    "right_wrist_roll_joint":     (-0.5,  0.5),
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
        # Draw the two velocity arrows (like tahiti/hv1_2, which inherit
        # debug_vis=True from the base CommandsCfg): GREEN = commanded velocity
        # (goal), BLUE = the base's actual velocity (tracking). Lets you SEE how
        # well the walk tracks the command during PLAY. Cheap; safe to leave on
        # for headless training too (markers only render with a viewer).
        debug_vis=True,
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
    # scale=0.5 matches the proven in-repo MLP walker (hv1_2). At the old 0.25,
    # joint_target = default + 0.25*action, so commanding a swing step (hip ~0.5,
    # knee ~0.8 rad off default) needed actions of 2-4 — the far tail of the policy
    # distribution and only rarer as std collapses to 0.44. A step was mechanically
    # out of exploration range regardless of reward => the robot always stood.
    # 0.5 halves the action needed for the same joint excursion, putting a step
    # inside the distribution. DEPLOY: the runtime must apply the SAME 0.5 scale.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINT_NAMES, scale=0.5, use_default_offset=True
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
        # GAIT-CLOCK OBSERVATION REMOVED 2026-07-27 — full alignment to the user's OWN proven
        # MLP walker (hv1_2_velocity), which has NO gait clock (obs = ang_vel, gravity, cmd,
        # joint_pos, joint_vel, actions) and DOES step. The clock was paired with the `contact`
        # gait reward, and that reward turned out to SUBSIDIZE standing (a planted foot already
        # matches the stance window ~55% of the cycle, so a stander farms ~half the term for
        # free) — the 7.6k server run proved it: pure stander, feet_air_time pinned at 0.0000,
        # gait ~0.95 raw = the stander level. With the gait reward gone (below) this obs feeds
        # a clock no term references, so it is removed too. Actor obs 77 -> 75 (critic 80 -> 78).
        # gait_phase = ObsTerm(func=custom_mdp.gait_phase, params={"period": GAIT_PERIOD})
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
# Rewards — STAGE 1 "walk-first": aligned to the proven official h1/g1 biped recipe
# (2026-07-26). Stripped from 22 active terms to ~16 by removing polish / guessed-param
# shaping (base_height, the soft-landing pair, air-time-variance) and de-escalating
# stability weights that were 3-10x the official values; ADDED the -200 termination
# penalty every official biped config uses. The robot already WALKS (error_vel_xy 0.13)
# but FALLS (bad_orientation) — this round targets stability, not stepping. Re-add the
# removed terms in STAGE 2, ONE at a time, with constants MEASURED from a PLAY rollout
# of the first working walker (never guessed).
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # -- task (weights + std matched to the OFFICIAL Unitree h1_2 config, 2026-07-26)
    # weight -> 1.0, std -> sqrt(0.25)=0.5 (official tracking_lin_vel=1.0, tracking_sigma=0.25).
    # LOOSENING std back to 0.5 is a deliberate fix: at std 0.3 a barely-moving robot got
    # essentially the SAME ~0 tracking reward as a standing one (exp(-0.87^2/0.09)~0 either
    # way), so there was NO gradient pulling it from "still" toward "moving". At std 0.5 the
    # same error 0.87->0.5 jumps the reward 0.05->0.37 — a real slope to climb toward walking.
    # track_lin 1.0 -> 1.5 (2026-07-27) = the hv1_2 walker's value, and it now OUT-WEIGHTS
    # track_ang (1.0) on purpose: the 7.6k run farmed track_ang_vel_z (0.75, the single
    # biggest reward) by PIVOTING the base in place to satisfy yaw commands while never
    # translating (error_vel_xy stuck at 0.88). Making linear tracking pay 1.5x the yaw kills
    # that shortcut — forward progress is now the cheaper way to earn tracking reward.
    # STD TIGHTENED 0.5 -> 0.4 (2026-07-29, "slow-motion" fix). The robot now WALKS
    # robustly (ep len 1000, bad_orient 0.15%) but LAGS the command: error_vel_xy 0.28,
    # error_vel_yaw 0.47, while track_lin sat at 1.36 (~90% of max) — proof the loose
    # std=0.5 let a 0.28 m/s lag keep most of the reward, so there was no pull to full
    # speed. std 0.5->0.4 (sigma^2 0.25->0.16) steepens the slope around the exact
    # command (at err 0.28: reward 0.73->0.61), pulling the achieved velocity UP toward
    # the commanded value. The earlier "loosen std or it stands" concern is retired —
    # that applied when it couldn't step; now stepping is solved, sharpen tracking.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,  # kept BELOW track_lin (was farmed by pivot-in-place); hv1_2 uses 1.5
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},  # 0.5->0.4: yaw error 0.47 is the worst-tracked axis
    )
    # alive 0.15 -> 0.05 (2026-07-27) = hv1_2 value. At 0.15 this was a big STANDING SUBSIDY:
    # a robot that just balances collects it in full with zero risk, so standing beat stepping.
    # Paired with the -100 termination penalty below (as hv1_2 does), the incentive flips —
    # survival still matters, but not enough to make standing-forever the optimum.
    alive = RewTerm(func=mdp.is_alive, weight=0.05)
    # TERMINATION PENALTY RESTORED 2026-07-27 (-100 = hv1_2/tahiti). Removing it (to copy the
    # official LSTM recipe) left falling nearly free; combined with the alive+gait standing
    # subsidy, the robot's safest high-reward strategy was to stand. Both of the user's OWN
    # MLP walkers pair a LOW alive (0.05) with a -100 termination penalty — restored here.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    # -- base / smoothness (joint-wise terms scoped to the actuated legs so we never
    #    charge the policy for the PD holding the parked torso/arms). Weights = official h1_2.
    # -2.0 = official lin_vel_z. (Earlier lowered to -0.2 fearing it flattened the step's
    # vertical bob; the official proves -2.0 walks — it is NOT the stepping blocker.)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # dof_vel -1e-3 = official (restored). Scoped to the legs.
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7,  # dof_acc = official
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    # action_rate -0.01 -> -0.003 (2026-07-27) = hv1_2 value. At -0.01 (3.3x heavier) this
    # penalized exactly the fast, alternating leg-action changes a real step requires, so a
    # smooth stand was cheaper than a step. Both proven MLP walkers use -0.003.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0,  # -5.0 = official h1_2
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    energy = RewTerm(
        func=custom_mdp.energy, weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )

    # -- posture
    # hip_roll strongly penalized (stops leg splay) but not so hard it chokes strafing;
    # hip_yaw lightly penalized so the policy can still use it to steer.
    # weights -0.2 / -0.1 = official h1/g1 (were -1.0 / -0.5, i.e. 5x too strong). At 5x
    # the policy couldn't freely use its hips to catch a stumble / recover a push.
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)  # -1.0 = official (orientation)
    # ANTI-CROUCH 2026-07-30. iter-3028 log decoded: base_height reward -0.0114 = -10*(h-1.0)^2
    # => walking pelvis ~0.97 m (stance knee ~40deg = deep crouch). At -10 the term was
    # negligible (-0.01 vs track_lin +1.33) => policy picked the stable knee-bend crouch.
    # URDF FORWARD KINEMATICS (thigh .40 + shank .40 + ankle .02, pelvis->hip .163, foot sole
    # .045 below ankle origin): straight-leg pelvis CEILING = 1.028 m; default pose (knee 21deg)
    # = 1.015 m. So an earlier 1.05 target was ABOVE the physical ceiling (unreachable). Correct
    # target = a natural WALKING height with a slight athletic bend: 1.00 m (~3 cm above the 0.97
    # crouch, safely below the 1.028 stiff-knee ceiling). Weight -20 for authority. Targets
    # PELVIS HEIGHT (stance property) not knee angle -> extends the stance leg without blocking
    # swing-knee bend (keeps the "no explicit knee-flexion reward" constraint). NOTE: L2 over this
    # narrow 6 cm window is a WEAK lever; the RELIABLE de-crouch is straightening the DEFAULT pose
    # (Tier 2: knee 0.36->0.24 in H1_2_CFG) since the policy anchors to its default and currently
    # walks BELOW even its own static default height. Reward-only: NO deploy change.
    base_height = RewTerm(func=mdp.base_height_l2, weight=-20.0, params={"target_height": 1.00})

    # -- feet / gait
    #
    # GAIT CLOCK REWARD REMOVED 2026-07-27 — this was THE standing subsidy. `feet_gait` pays
    # +1 per foot whose contact matches the phase-clock schedule; a PLANTED foot already
    # matches the stance window (leg_phase < 0.55) ~55% of every cycle, so a robot that never
    # lifts a foot still farms ~half the term for free. The 7.6k server run is the proof:
    # pure stander, feet_air_time == 0.0000, yet `gait` sat at ~0.95 raw = exactly the free
    # stander level. That free reward, on top of alive 0.15, made standing the optimum and
    # the air-time carrot (0 until a step exists) could never out-earn it. Neither of the
    # user's OWN MLP walkers (tahiti, hv1_2) uses a gait clock at all — feet_air_time is the
    # SOLE stepping driver there, and they step. Matching that. (gait_phase OBS removed too.)
    # gait = RewTerm(
    #     func=custom_mdp.feet_gait,
    #     weight=0.18,
    #     params={
    #         "period": GAIT_PERIOD,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #     },
    # )
    # THE MISSING CARROT — feet_air_time_positive_biped, RE-ADDED 2026-07-27.
    # Diagnosis from the 5.6k-iter SERVER run (4096 env, obs-normalization ON): the robot now
    # BALANCES well (ep len 938/1000, bad_orientation only 16%) but STILL won't step —
    # feet_swing_height logged at EXACTLY -0.0000 (the feet literally never leave the ground)
    # and error_vel_xy stuck at 0.80 at full command. Root cause: not one stepping term gives a
    # STANDING robot a positive reward it is MISSING. `gait` pays a stander ~55% for free
    # (stance-window match); `feet_swing_height` is a PURE PENALTY — 0 while planted, and it only
    # ever COSTS the instant a foot lifts, so it is a BARRIER to the first tentative step, not a
    # driver (the 0.0 in the log is the proof: the policy simply avoids it by never lifting). A
    # well-balanced robot then has every incentive to stay planted. This term is the carrot our
    # OWN two MLP walkers (tahiti_c1 w=2.5, hv1_2 w=1.0) use: it is EXACTLY 0 in double support
    # and grows with real single-support swing TIME, so a stander is visibly leaving reward on
    # the table and the gradient points OUT of standing. Command-gated (|cmd|>0.1) so it can't be
    # farmed by marching at idle. The earlier "air_time never worked" rounds were ALL
    # pre-obs-normalization — obs were unlearnable-conditioned, so no reward could step it; with
    # obs-norm ON and balance already solved, the carrot can finally do its job.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,  # strong on purpose — must beat a DEEP standing basin. tahiti 2.5 / hv1_2 1.0.
        params={
            "command_name": "base_velocity",
            # 0.4 -> 0.3 (2026-07-29, cadence). The term caps single-support credit at
            # `threshold`; at 0.4 s it paid to PROLONG each swing (~1.25 Hz, the slow-mo
            # gait). Capping at 0.3 s removes the incentive to hold a long swing, so the
            # policy ends the step sooner => quicker turnover. Still well above a foot-tap.
            "threshold": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    # feet_swing_height DISABLED 2026-07-27 — the 5.6k run logged it at EXACTLY -0.0000, i.e. it
    # forced NOTHING: as a penalty that only applies to an AIRBORNE foot it is inert for a planted
    # robot and becomes a BARRIER to the exploratory first lift (any height != 0.08 m costs
    # -20*(z-0.08)^2). It shapes swing HEIGHT of an ALREADY-stepping gait (which is how the
    # official LSTM recipe can afford -20). Re-add SMALL (~ -1) for clearance polish AFTER a
    # confirmed walk; for now it only fights the air_time carrot above.
    # feet_swing_height = RewTerm(
    #     func=custom_mdp.feet_swing_height,
    #     weight=-20.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #         "target_height": 0.08,
    #     },
    # )
    # L/R symmetry — RE-ENABLED 2026-07-29 at -0.5 (was disabled walk-first). This is the
    # "one-leg-stand / limp" fix: it penalizes var(air_time)+var(contact_time) across the two
    # feet, so a gait where one foot stays up much longer than the other (the yoga-walk the
    # user is seeing) costs reward. Held at -0.5 (not the old -1.0, which suppresses the
    # naturally-uneven FIRST steps) because a stable walk now exists — STAGE-2 conditions met.
    air_time_variance = RewTerm(
        func=custom_mdp.air_time_variance_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    # -0.2 -> -0.5 (2026-07-26, matching tahiti/hv1_2). Closes the "slide/drag a planted
    # foot to translate" loophole: to satisfy velocity tracking the base must actually be
    # carried forward by an AIRBORNE foot (a step), not dragged. Both custom walkers use -0.5.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
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
        # -2 -> -5 (2026-07-29, "feet-close" fix). At -2 the hinge was toothless
        # (logged -0.0147) so the policy freely narrowed its stance / scissored. -5 gives
        # it real authority WITHOUT reaching the -10 that could force a toed-out waddle.
        weight=-5.0,
        params={
            # MEASURED from the handless URDF (not guessed): each foot hangs at Y=±0.163 m
            # under its hip at hip_roll=0, so the NATURAL stance is 0.326 m. min_distance 0.22
            # (0.20 -> 0.22) sets the penalty floor well below natural (so normal swing-through
            # near the midline isn't over-taxed) but high enough to stop the narrow/crossed
            # stance. Raise toward ~0.25 only if PLAY still shows the feet too close.
            "min_distance": 0.22,
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

    # -- SOFT-LANDING / low ground-force pair — REMOVED for the walk-first stage (2026-07-26).
    # Both are polish (a quiet, low-impact walk) and no official biped config uses them.
    # foot_contact_velocity (-0.5) in particular penalizes the foot's vertical speed IN
    # contact — the fast foot motion a push-off / fall-catch needs — so it works against
    # learning a robust walk. STAGE 2: re-add BOTH once a stable walk exists, with the force
    # threshold MEASURED from a PLAY/MuJoCo peak-GRF rollout (not the 1000 N guess). Physics
    # anchor kept for that step: H1_2 ≈ 67 kg ⇒ ~660 N = 1 BW; walk peaks ≈ 1.2-1.5 BW.
    # foot_contact_force = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-2.0e-3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 1000.0,  # TUNE from measured peak GRF
    #     },
    # )
    # foot_contact_velocity = RewTerm(
    #     func=custom_mdp.foot_contact_velocity_penalty,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #         "force_threshold": 5.0,
    #     },
    # )

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
    # +3 kg EE PAYLOAD DR — REMOVED for the walk-first stage (2026-07-26, user request:
    # "remove hand payload for now, first walk then manipulation DR"). Carrying a heavy,
    # arm-swung object while ALSO trying to discover a gait was one disturbance too many.
    # Re-add this (staged via payload_mass_levels) only AFTER a confirmed stable walk.
    # add_ee_payload = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_yaw_link"),
    #         "mass_distribution_params": (0.0, 0.0),  # grown to (0, 3) by payload_mass_levels
    #         "operation": "add",
    #     },
    # )
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
    # WALK-FIRST 2026-07-26: pin the ENTIRE upper body (torso + arms) at its default
    # ready-pose, PD-held. The arm-motion DR and the +3 kg EE payload were BOTH removed
    # for now (user: "remove hand payload for now, first walk then manipulation DR") so
    # the robot learns a clean legs-only walk with zero manipulation disturbance — this
    # is exactly the official Unitree h1_2 setup (arms held at a fixed default, no arm
    # randomization). The arm-motion DR + payload are RE-ADDED (staged) only AFTER a
    # confirmed stable walk. The two randomize_arm_joint_targets events below are kept,
    # commented out, for that step.
    hold_upper_body_target = EventTerm(
        func=custom_mdp.hold_joint_targets_at_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES)},
    )
    # ARM-MOTION DR — DISABLED for the walk-first stage (re-enable after a stable walk).
    # randomize_arm_targets_reset = EventTerm(
    #     func=custom_mdp.randomize_arm_joint_targets,
    #     mode="reset",
    #     params={
    #         "position_range": ARM_TARGET_RANGES,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
    #     },
    # )
    # randomize_arm_targets_interval = EventTerm(
    #     func=custom_mdp.randomize_arm_joint_targets,
    #     mode="interval",
    #     interval_range_s=(3.0, 5.0),
    #     params={
    #         "position_range": ARM_TARGET_RANGES,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
    #     },
    # )
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
    # PURE-STAND PHASE REMOVED 2026-07-26 — this was the standing ATTRACTOR. The official
    # Unitree h1_2 recipe has NO stand-first curriculum: it demands walking (full command
    # + push) from iter 0, so the robot never learns "stand still" as a deep optimum. Our
    # 2000-iter zero-command Phase 1 (100% standing envs) trained a strong standing prior,
    # and by the time full command arrived the MLP was stuck in that basin and never
    # stepped (feet_air_time ~0 for the whole run, even at full command in Phase 3). Fix:
    # start command immediately. stand_until_iters 2000 -> 0 (skip Phase 1 entirely) and a
    # short slow ramp slow_until_iters 5000 -> 1000, so the robot is commanded to slow-walk
    # from iter 0 and reaches full command by iter 1000. It WILL fall a lot early (like the
    # official does) — that is how it discovers stepping. A fixed schedule still can't stall.
    #   Phase 2 (iter 0 .. 1000):   command x 0.3, 30% standing (slow walk from the start).
    #   Phase 3 (>= 1000):          full command, 10% standing.
    # *_full ranges MUST equal commands.base_velocity.limit_ranges below.
    command_phase = CurrTerm(
        func=custom_mdp.stand_to_walk_command_curriculum,
        params={
            "stand_until_iters": 0,     # no pure-stand phase (was 2000 — the standing attractor)
            "slow_until_iters": 1000,   # short slow ramp; full command from iter 1000 (was 5000)
            "slow_scale": 0.3,
            "lin_vel_x_full": (-0.5, 1.0),  # == limit_ranges.lin_vel_x
            "lin_vel_y_full": (-0.5, 0.5),  # == limit_ranges.lin_vel_y
            "ang_vel_z_full": (-0.5, 0.5),  # == limit_ranges.ang_vel_z
            "rel_standing_envs_phase1": 1.0,  # unused now (Phase 1 skipped)
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
    # EE-PAYLOAD curriculum REMOVED 2026-07-26 — the payload event itself is disabled for
    # the walk-first stage (see EventCfg.add_ee_payload). Re-add both together, staged, once
    # a stable walk is confirmed.
    # payload_mass_levels = CurrTerm(
    #     func=custom_mdp.payload_mass_levels,
    #     params={
    #         "term_name": "add_ee_payload",
    #         "start_iters": 5000,
    #         "full_iters": 9000,
    #         "max_mass": 3.0,
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
        # Walk-first stage: arms are held at default and there is no EE payload (both the
        # arm-motion DR and payload are disabled in the base cfg), so PLAY just demos the
        # clean legs-only walk. Disable pushes for the demo.
        self.events.push_robot = None
        # play at the fully-grown command range (skip the curriculum ramp)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.command_phase = None
        # push_velocity_levels / payload_mass_levels / add_ee_payload are all disabled in
        # the base cfg for the walk-first stage — nothing to override here.
