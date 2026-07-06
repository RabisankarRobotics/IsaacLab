# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained flat-ground walk for the Unitree G1 29-DOF, legs-only.

This config does NOT inherit the IsaacLab G1 velocity parents. It is a
standalone ``ManagerBasedRLEnvCfg`` that ports the proven *unitree_rl_lab* G1
velocity recipe — the reference config that produces a clean, rhythmic,
human-like walk — while keeping this project's deploy design.

What makes the gait natural (ported from unitree_rl_lab, NOT in the IsaacLab
base config):

* ``gait`` (``feet_gait``): a **phase-based periodic gait reward**. A fixed clock
  tells each foot when to be in stance vs swing (anti-phase, 0.8 s period). The
  natural knee/hip motion emerges to satisfy the rhythm — instead of clamping
  the knee angle directly (which caused the crouch↔locked-knee oscillation).
* **Velocity-command curriculum**: commands start tiny (±0.1) and grow only as
  tracking improves, up to ``limit_ranges``. The robot masters slow clean
  walking first.
* ``energy`` penalty and stronger posture regularization (hip deviation −1.0,
  orientation −5.0, ``dof_pos_limits`` −5.0) for an efficient, upright gait.

What is kept from this project's deploy design (differs from unitree_rl_lab):

* **Legs-only action** (12 joints). Arms/waist/wrists are held at their default
  by their stiff PD actuators (and are free for the arm-DR loco-manip work
  later). unitree_rl_lab drives all 29 joints.
* **Asymmetric, deploy-safe actor-critic**: the actor (``policy`` group) reads
  only hardware-measurable quantities (IMU gyro, IMU tilt, command, the 12 leg
  encoders, last action, the gait clock, and — arm-aware — the upper-body
  encoders). The unmeasurable ``base_lin_vel`` is critic-only.
* **No observation history.** Instead of unitree_rl_lab's history length 5, the
  policy gets a **2-value sin/cos gait clock** (:func:`~.mdp.gait_phase`) so it
  can phase-lock to the ``feet_gait`` reward. At deploy this is just a free-
  running timer — no encoder history to buffer.
* IsaacLab ``G1_29DOF_CFG`` (matches the MuJoCo deploy asset / gains) on **flat
  ground**.

Policy observation layout (81), in concat order — the MuJoCo deploy runner must
rebuild this exactly:
    base_ang_vel (3) | projected_gravity (3) | velocity_commands (3) |
    gait_phase (2) | joint_pos legs (12) | joint_vel legs (12) |
    last_action (12) | upper_body_joint_pos (17) | upper_body_joint_vel (17)

Action: ``target_q_leg = raw_action * 0.25 + default_q_leg`` (scale 0.25).

Task id: ``Isaac-Velocity-Flat-Legs-G1-29Dof-Clean-v0``.
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
from isaaclab_assets import G1_29DOF_CLEAN_CFG  # isort: skip


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

# Arm joints — held at default here; owned by a manipulation policy at deploy.
ARM_JOINT_NAMES = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]
WAIST_JOINT_NAMES = ["waist_.*_joint"]
# Upper body the walker OBSERVES (arm-aware). Fingers excluded (negligible mass).
UPPER_BODY_JOINT_NAMES = WAIST_JOINT_NAMES + ARM_JOINT_NAMES

# Gait clock period (s). Shared by the feet_gait reward and the gait_phase obs.
GAIT_PERIOD = 0.8


# ---------------------------------------------------------------------------
# Command: uniform velocity + a `limit_ranges` field the curriculum grows toward
# ---------------------------------------------------------------------------
@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """``UniformVelocityCommandCfg`` plus the ``limit_ranges`` cap the
    command-level curriculum (``mdp.lin_vel_cmd_levels``) grows toward."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class G1CleanSceneCfg(InteractiveSceneCfg):
    """Flat ground + G1 29-DOF + a full-body contact sensor."""

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

    # Full unitree_rl_lab articulation (leg + soft upper-body gains, bent-arm pose,
    # contact sensors on). Only prim_path is overridden.
    robot: ArticulationCfg = G1_29DOF_CLEAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Full-body contact sensor: feet gate the gait/slide/clearance terms, the
    # rest gate the undesired-contact penalty. track_air_time is required by
    # feet_gait's contact timing.
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
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        # Caps the curriculum. Forward-biased, modest strafe/yaw for a clean walk.
        limit_ranges=UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
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
        (the MuJoCo runner rebuilds this exact concatenation)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 2-value sin/cos gait clock — lets the policy phase-lock to feet_gait
        # without an observation history. Deploy = a free-running timer.
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
        # Arm-aware: observe the live upper-body joint state (≈0 while parked; at
        # deploy feed real arm/waist encoders) so the legs can anticipate CoM
        # shifts from arm motion.
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
# Rewards — the unitree_rl_lab recipe, adapted to legs-only
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base / smoothness (joint-wise terms scoped to the actuated legs so we
    #    never charge the policy for the PD holding the parked arms/waist)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
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
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})

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
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_clearance = RewTerm(
        func=custom_mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
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
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


# ---------------------------------------------------------------------------
# Events — modest DR (curriculum makes it tractable). Arms stay parked: no arm
# randomization in this clean run; add it in the loco-manip hardening run.
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
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# ---------------------------------------------------------------------------
# Curriculum — grow the command range as tracking improves
# ---------------------------------------------------------------------------
@configclass
class CurriculumCfg:
    lin_vel_cmd_levels = CurrTerm(func=custom_mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=custom_mdp.ang_vel_cmd_levels)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@configclass
class G1FlatLegs29DofCleanEnvCfg(ManagerBasedRLEnvCfg):
    """Legs-only G1 29-DOF flat walk, unitree_rl_lab recipe, deploy-safe."""

    scene: G1CleanSceneCfg = G1CleanSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 50 Hz control (decimation 4 x sim dt 0.005) — matches the MuJoCo deploy
        # CONTROL_DT and the 0.8 s gait period (= 40 control steps).
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # G1_29DOF_CLEAN_CFG already enables contact sensors; tick the sensor at the
        # physics rate for correct air-time.
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1FlatLegs29DofCleanEnvCfg_PLAY(G1FlatLegs29DofCleanEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # clean demo: no observation noise, no pushes
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # play at the fully-grown command range (skip the curriculum ramp)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
