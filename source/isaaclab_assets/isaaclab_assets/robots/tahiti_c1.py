# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tahiti C1 — 12-DoF bipedal robot articulation config.

Per-leg chain: hip_yaw → hip_pitch → hip_roll → knee → ankle_pitch → ankle_roll.
Total 12 actuated joints (6 per leg). Base link is ``base_link`` (no separate
pelvis). No arms / no upper body — this is a locomotion-only platform.

Motor spec supplied by the user:
    X12-320  → ``.*hip.*`` and ``.*knee.*``  (8 joints)  85 Nm, Kp 250, Kd 25
    X6-60    → ``.*ankle.*``                 (4 joints)  20 Nm, Kp 200, Kd 20

    Kd values updated 2026-07-17 to match the robo_control deploy base yaml
    (hips/knees kd=25, ankles kd=20 — inverted vs earlier training). Match
    is required for sim-to-real parity.

Actuators default to :class:`DelayedPDActuatorCfg` (0-6 physics-step command
lag, ~30 ms at sim_dt=0.005 — the sim-to-real target). If early training gets
stuck under the added delay, swap the two ``DelayedPDActuatorCfg`` blocks for
``ImplicitActuatorCfg`` (drop ``min_delay`` / ``max_delay``, keep everything
else) for a simpler zero-lag first pass.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg  # noqa: F401 (ImplicitActuatorCfg re-exported for easy swap)
from isaaclab.assets.articulation import ArticulationCfg


TAHITI_C1_URDF_PATH = (
    "/home/rabisankar/IsaacLab/source/isaaclab_assets/data/custom_robot/"
    "urdf_mesh/tahiti_c1/tahiti_c1.urdf"
)


TAHITI_C1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=TAHITI_C1_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                # Match the per-actuator-group stiffness/damping below.
                stiffness={
                    ".*_hip_yaw_joint": 250.0,
                    ".*_hip_pitch_joint": 250.0,
                    ".*_hip_roll_joint": 250.0,
                    ".*_knee_joint": 250.0,
                    ".*_ankle_pitch_joint": 200.0,
                    ".*_ankle_roll_joint": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 25.0,
                    ".*_hip_pitch_joint": 25.0,
                    ".*_hip_roll_joint": 25.0,
                    ".*_knee_joint": 25.0,
                    ".*_ankle_pitch_joint": 20.0,
                    ".*_ankle_roll_joint": 20.0,
                },
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Default pose updated 2026-07-17 to match the robo_control deploy base
        # yaml (hip_pitch=-0.10, knee=0.30, ankle_pitch=+0.20). Less-bent knee
        # + more-upright hip raises the settled height by ~1-2 cm vs the earlier
        # 0.36/-0.16 pose, so spawn bumped 0.92 → 0.95 m to keep the "settle-in
        # from above" behavior at t=0.
        # NOTE ankle_pitch SIGN FLIPPED from -0.20 to +0.20. Verify in Play
        # viewer that the foot is flat (not heel-tilted with toe raised). If
        # heel-tilted, the deploy encoder sign is opposite the URDF axis and a
        # sign inversion must be added in the deploy path.
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.10,
            ".*_hip_roll_joint": 0.0,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": 0.20,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # 2026-07-18: 0.9 → 0.85. mujoco V3@30k showed policy pinning ankle_roll
    # at ±0.194 (= 0.9 × ±0.216 hardware limit) every stance phase → structural
    # L/R asymmetry from L/R hardware DR spread at the wall. Moving the soft
    # limit inward to 0.85 pushes dof_pos_limits penalty earlier so the policy
    # stops using the limit as a stability crutch.
    soft_joint_pos_limit_factor=0.85,
    actuators={
        # X12-320 group: 3 hip joints + knee, per leg × 2 = 8 joints.
        "legs_x12_320": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit=85.0,
            velocity_limit=10.0,
            stiffness=250.0,
            damping=25.0,
            armature=0.326938,
            friction=1.694307,
            viscous_friction=0.350134,
            min_delay=0,
            max_delay=6,
        ),
        # X6-60 group: ankle pitch + roll, per leg × 2 = 4 joints.
        "ankles_x6_60": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=20.0,
            velocity_limit=16.0,
            stiffness=200.0,
            damping=20.0,
            armature=0.019603,
            friction=0.321816,
            viscous_friction=0.165640,
            min_delay=0,
            max_delay=6,
        ),
    },
)
"""Tahiti C1 bipedal robot — 12 DoF, ~53.5 kg total, base_link root."""
