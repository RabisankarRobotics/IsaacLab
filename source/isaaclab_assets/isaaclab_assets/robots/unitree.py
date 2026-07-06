# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`UNITREE_GO1_CFG`: Unitree Go1 robot with actuator net model for the legs
* :obj:`UNITREE_GO2_CFG`: Unitree Go2 robot with DC motor model for the legs
* :obj:`H1_CFG`: H1 humanoid robot
* :obj:`H1_MINIMAL_CFG`: H1 humanoid robot with minimal collision bodies
* :obj:`G1_CFG`: G1 humanoid robot
* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies
* :obj:`G1_29DOF_CFG`: G1 humanoid robot configured for locomanipulation tasks
* :obj:`G1_INSPIRE_FTP_CFG`: G1 29DOF humanoid robot with Inspire 5-finger hand

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##

GO1_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/Unitree/unitree_go1.pt",
    pos_scale=-1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=23.7,  # taken from spec sheet
    velocity_limit=30.0,  # taken from spec sheet
    saturation_effort=23.7,  # same as effort limit
)
"""Configuration of Go1 actuators using MLP model.

Actuator specifications: https://shop.unitree.com/products/go1-motor

This model is taken from: https://github.com/Improbable-AI/walk-these-ways
"""


##
# Configuration
##


UNITREE_A1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/A1/a1.usd",
        activate_contact_sensors=True,
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree A1 using DC motor.

Note: Specifications taken from: https://www.trossenrobotics.com/a1-quadruped#specifications
"""


UNITREE_GO1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd",
        activate_contact_sensors=True,
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": GO1_ACTUATOR_CFG,
    },
)
"""Configuration of Unitree Go1 using MLP-based actuator model."""


UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 using DC-Motor actuator model."""


H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
        activate_contact_sensors=True,
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.28,  # -16 degrees
            ".*_knee": 0.79,  # 45 degrees
            ".*_ankle": -0.52,  # -30 degrees
            "torso": 0.0,
            ".*_shoulder_pitch": 0.28,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw": 150.0,
                ".*_hip_roll": 150.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 200.0,
                "torso": 200.0,
            },
            damping={
                ".*_hip_yaw": 5.0,
                ".*_hip_roll": 5.0,
                ".*_hip_pitch": 5.0,
                ".*_knee": 5.0,
                "torso": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit_sim=100,
            stiffness={".*_ankle": 20.0},
            damping={".*_ankle": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit_sim=300,
            stiffness={
                ".*_shoulder_pitch": 40.0,
                ".*_shoulder_roll": 40.0,
                ".*_shoulder_yaw": 40.0,
                ".*_elbow": 40.0,
            },
            damping={
                ".*_shoulder_pitch": 10.0,
                ".*_shoulder_roll": 10.0,
                ".*_shoulder_yaw": 10.0,
                ".*_elbow": 10.0,
            },
        ),
    },
)
"""Configuration for the Unitree H1 Humanoid robot."""


H1_MINIMAL_CFG = H1_CFG.copy()
H1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1_minimal.usd"
"""Configuration for the Unitree H1 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""


# ---------------------------------------------------------------------------
# H1_2 standing task (21 DoF: legs + torso + shoulder/elbow_pitch per arm).
# The URDF is generated by:
#   resources/robots/h1_2/generate_stand_urdf.py
# in the unitree_rl_gym repo. Isaac Lab converts URDF -> USD on first launch.
# ---------------------------------------------------------------------------
H1_2_STAND_URDF_PATH = "/home/rabisankar/unitree_rl_gym/resources/robots/h1_2/h1_2_stand.urdf"

H1_2_STAND_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=H1_2_STAND_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    ".*_hip_yaw_joint":   200.0,
                    ".*_hip_roll_joint":  200.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint":      300.0,
                    ".*_ankle_pitch_joint": 40.0,
                    ".*_ankle_roll_joint":  40.0,
                    "torso_joint":        200.0,
                    ".*_shoulder_pitch_joint": 100.0,
                    ".*_shoulder_roll_joint":  100.0,
                    ".*_shoulder_yaw_joint":   100.0,
                    ".*_elbow_pitch_joint":     80.0,
                },
                damping={
                    ".*_hip_yaw_joint":   2.5,
                    ".*_hip_roll_joint":  2.5,
                    ".*_hip_pitch_joint": 2.5,
                    ".*_knee_joint":      4.0,
                    ".*_ankle_pitch_joint": 2.0,
                    ".*_ankle_roll_joint":  2.0,
                    "torso_joint":        5.0,
                    ".*_shoulder_pitch_joint": 2.0,
                    ".*_shoulder_roll_joint":  2.0,
                    ".*_shoulder_yaw_joint":   2.0,
                    ".*_elbow_pitch_joint":    2.0,
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint":     0.0,
            ".*_hip_pitch_joint":  -0.16,
            ".*_hip_roll_joint":    0.0,
            ".*_knee_joint":        0.36,
            ".*_ankle_pitch_joint":-0.2,
            ".*_ankle_roll_joint":  0.0,
            "torso_joint":          0.0,
            ".*_shoulder_pitch_joint": 0.4,
            ".*_shoulder_roll_joint":  0.0,
            ".*_shoulder_yaw_joint":   0.0,
            ".*_elbow_pitch_joint":    0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Effort limits are inherited from the URDF; explicit actuator wrappers are
    # not needed because the joint drive (stiffness/damping) is already declared
    # in the spawn config above. Isaac Lab automatically creates one implicit
    # actuator group per joint when no `actuators=` entry is provided here.
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
)
"""H1_2 21-DoF (legs + torso + shoulder/elbow_pitch) for the standing task."""


G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit_sim=300,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""


G1_MINIMAL_CFG = G1_CFG.copy()
G1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd"
"""Configuration for the Unitree G1 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""


G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=False,
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
            fix_root_link=False,  # Configurable - can be set to True for fixed base
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        rot=(0.7071, 0, 0, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=180.0,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 5000.0,
                "waist_roll_joint": 5000.0,
                "waist_pitch_joint": 5000.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit=300,
            velocity_limit=100,
            stiffness=3000.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.001,
                ".*_elbow_.*": 0.001,
                ".*_wrist_.*_joint": 0.001,
            },
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_.*",
                ".*_middle_.*",
                ".*_thumb_.*",
            ],
            effort_limit=300,
            velocity_limit=100,
            stiffness=20,
            damping=2,
            armature=0.001,
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)
"""Configuration for the Unitree G1 Humanoid robot for locomanipulation tasks.

This configuration sets up the G1 humanoid robot for locomanipulation tasks,
allowing both locomotion and manipulation capabilities. The robot can be configured
for either fixed base or mobile scenarios by modifying the fix_root_link parameter.

Key features:
- Configurable base (fixed or mobile) via fix_root_link parameter
- Optimized actuator parameters for locomanipulation tasks
- Enhanced hand and arm configurations for manipulation

Usage examples:
    # For fixed base scenarios (upper body manipulation only)
    fixed_base_cfg = G1_29DOF_CFG.copy()
    fixed_base_cfg.spawn.articulation_props.fix_root_link = True

    # For mobile scenarios (locomotion + manipulation)
    mobile_cfg = G1_29DOF_CFG.copy()
    mobile_cfg.spawn.articulation_props.fix_root_link = False
"""


G1_29DOF_CLEAN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Kept from IsaacLab's g1.usd (NOT unitree_rl_lab's g1_29dof_rev_1_0.usd):
        # switching USD can change IsaacLab's joint sort order, which would break the
        # LEG_DDS/UP_DDS mapping the MuJoCo deploy relies on.
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
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
            # unitree_rl_lab uses enabled_self_collisions=True. Kept False here: the
            # IsaacLab g1.usd collision meshes are not validated for self-contact and
            # can cause spawn-time explosions. Flip to True only after checking.
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        # g1.usd is authored with a 90-deg yaw; keep this so "forward" (+x command)
        # matches the robot's facing. Do NOT drop it for this USD.
        rot=(0.7071, 0, 0, 0.7071),
        # unitree_rl_lab UNITREE_G1_29DOF_CFG default pose — legs + a bent-arm ready
        # pose. The soft upper-body PD (below) holds the arms here; they comply under
        # gravity. Parked (no policy action) during legs-only, ready for loco-manip.
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Faithful port of unitree_rl_lab UNITREE_G1_29DOF_CFG actuators (the
    # manufacturer's own gains). ImplicitActuator (PD + effort clamp) matches the
    # MuJoCo software-PD deploy. Upper body is intentionally SOFT (KP 40) so it is
    # compliant for the upcoming loco-manipulation stage, not held rigid.
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)
"""G1 29-DOF — faithful port of unitree_rl_lab ``UNITREE_G1_29DOF_CFG``.

Uses the manufacturer's own gains and default pose for ALL joints, including a
SOFT (KP 40) upper body so the arms are already compliant for the coming
loco-manipulation stage (legs-only for now: no action is sent to the upper body,
but it is not held rigid). Deliberate deviations from unitree_rl_lab:

* USD + 90-deg spawn yaw kept from IsaacLab's ``g1.usd`` (not unitree's USD),
  to preserve IsaacLab's joint sort order and the MuJoCo LEG_DDS/UP_DDS mapping.
* enabled_self_collisions=False (unitree uses True) — enable only after the
  g1.usd collision meshes are validated for self-contact.

The gains AND the default arm pose here MUST stay in sync with the deploy PD and
DEFAULT_Q in ``unitree_mujoco/custom_policy/run_policy_mujoco.py``.
"""

"""
Configuration for the Unitree G1 Humanoid robot with Inspire 5fingers hand.
The Unitree G1 URDF can be found here: https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf
The Inspire hand URDF is available at: https://github.com/unitreerobotics/xr_teleoperate/tree/main/assets/inspire_hand
The merging code for the hand and robot can be found here: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/merge_g1_29dof_and_inspire_hand.ipynb,
Necessary modifications should be made to ensure the correct parent–child relationship.
"""
# Inherit PD settings from G1_29DOF_CFG, with minor adjustments for grasping task
G1_INSPIRE_FTP_CFG = G1_29DOF_CFG.copy()
G1_INSPIRE_FTP_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof_inspire_hand.usd"
G1_INSPIRE_FTP_CFG.spawn.activate_contact_sensors = True
G1_INSPIRE_FTP_CFG.spawn.rigid_props.disable_gravity = True
G1_INSPIRE_FTP_CFG.spawn.articulation_props.fix_root_link = True
G1_INSPIRE_FTP_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)
# Actuator configuration for arms (stability focused for manipulation)
# Increased damping improves stability of arm movements
G1_INSPIRE_FTP_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_.*_joint",
    ],
    effort_limit=300,
    velocity_limit=100,
    stiffness=3000.0,
    damping=100.0,
    armature={
        ".*_shoulder_.*": 0.001,
        ".*_elbow_.*": 0.001,
        ".*_wrist_.*_joint": 0.001,
    },
)
# Actuator configuration for hands (flexibility focused for grasping)
# Lower stiffness and damping to improve finger flexibility when grasping objects
G1_INSPIRE_FTP_CFG.actuators["hands"] = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_index_.*",
        ".*_middle_.*",
        ".*_thumb_.*",
        ".*_ring_.*",
        ".*_pinky_.*",
    ],
    effort_limit_sim=30.0,
    velocity_limit_sim=10.0,
    stiffness=10.0,
    damping=0.2,
    armature=0.001,
)


# ---------------------------------------------------------------------------
# HV1.2 custom humanoid (32 DoF). Source: source/isaaclab_assets/data/custom_robot/
# Structurally similar to H1-2 but with 3-DoF waist, 3-DoF wrists, and 3-DoF head.
# Per-motor specs supplied by the user: X4-36, X6-60, X8-120, X12-320.
#
# We spawn from the URDF (not the pre-generated USD) so Isaac Lab regenerates a
# clean self-contained USD on first launch. The provided _robot.usd in
# data/custom_robot/usd/hv1_2/configuration/ is a thin wrapper that references
# _base.usd; those references don't resolve correctly under
# /World/envs/env_*/Robot, so spawning _robot.usd directly fails with
# "no rigid bodies are present under this prim".
# ---------------------------------------------------------------------------
HV1_2_URDF_PATH = (
    "/home/rabisankar/IsaacLab/source/isaaclab_assets/data/custom_robot/"
    "urdf_mesh/hv1_2/hv1_2_without_ee_ankle_modified/hv1_2_without_arm_2.urdf"
)

HV1_2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=HV1_2_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                # Mirror the per-actuator-type stiffness/damping below.
                # These get baked into the joint drives in the generated USD.
                stiffness={
                    ".*_hip_yaw_joint": 250.0,
                    ".*_hip_roll_joint": 250.0,
                    ".*_hip_pitch_joint": 250.0,
                    ".*_knee_joint": 250.0,
                    ".*_ankle_pitch_joint": 200.0,
                    ".*_ankle_roll_joint": 200.0,
                    ".*_wrist_roll_joint": 200.0,
                    "waist_yaw_joint": 200.0,
                    "waist_roll_joint": 200.0,
                    "waist_pitch_joint": 200.0,
                    ".*_shoulder_pitch_joint": 200.0,
                    ".*_shoulder_roll_joint": 200.0,
                    ".*_shoulder_yaw_joint": 200.0,
                    ".*_elbow_joint": 200.0,
                    ".*_wrist_pitch_joint": 80.0,
                    ".*_wrist_yaw_joint": 80.0,
                    "head_pitch_joint": 80.0,
                    "head_yaw_joint": 80.0,
                },
                damping={
                    ".*_hip_yaw_joint": 15.0,
                    ".*_hip_roll_joint": 15.0,
                    ".*_hip_pitch_joint": 15.0,
                    ".*_knee_joint": 15.0,
                    ".*_ankle_pitch_joint": 20.0,
                    ".*_ankle_roll_joint": 20.0,
                    ".*_wrist_roll_joint": 25.0,
                    "waist_yaw_joint": 20.0,
                    "waist_roll_joint": 20.0,
                    "waist_pitch_joint": 20.0,
                    ".*_shoulder_pitch_joint": 20.0,
                    ".*_shoulder_roll_joint": 20.0,
                    ".*_shoulder_yaw_joint": 20.0,
                    ".*_elbow_joint": 20.0,
                    ".*_wrist_pitch_joint": 3.0,
                    ".*_wrist_yaw_joint": 3.0,
                    "head_pitch_joint": 3.0,
                    "head_yaw_joint": 3.0,
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
        pos=(0.0, 0.0, 0.96),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.16,
            ".*_knee_joint": 0.36,
            ".*_ankle_pitch_joint": -0.2,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # shoulder_pitch 0.0 (arms straight down). Eliminates ~7 cm
            # forward CoM bias vs the original 0.4 pose.
            # shoulder_roll ±0.16 (Unitree G1 convention) — abducts both arms
            # slightly outward to prevent arm-thigh/waist collision during
            # hip rotation. Per-side values overridden after the regex.
            # Keep these in sync with ARM_TARGETS_PIN in
            # hv1_2_velocity/flat_env_cfg.py — frame-1 PD snap destabilizes
            # init balance if they disagree.
            ".*_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_roll_joint": -0.16,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.3,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # ---------------------------------------------------------------------
    # All actuator groups switched ImplicitActuatorCfg → DelayedPDActuatorCfg
    # for sim-to-real training. The delay buffer adds 0–6 physics steps of
    # command lag per env (re-sampled each reset), modeling real-hardware
    # CAN-bus + motor controller response latency. At sim_dt=0.005s, max_delay=6
    # corresponds to ~30 ms of round-trip command delay — representative of
    # mid-tier industrial motor drives.
    # `effort_limit` / `velocity_limit` (no _sim suffix) — for explicit PD
    # actuators these clip the Python PD output (matches Spot's convention,
    # see spot.py:163-170). All other params (Kp, Kd, armature, friction,
    # viscous_friction) carry over unchanged from the implicit configuration.
    # ---------------------------------------------------------------------
    actuators={
        # X12-320: hip pitch/roll/yaw + knee (both sides) = 8 joints
        "legs_x12_320": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=85.0,
            velocity_limit=10.0,
            stiffness=250.0,
            damping=15.0,
            armature=0.326938,
            friction=1.694307,
            viscous_friction=0.350134,
            min_delay=0,
            max_delay=6,
        ),
        # X8-120: ankle pitch/roll (both sides) = 4 joints.
        # Upgraded from X6 because X6-60 (20 Nm) is undersized for real-robot
        # ankle torque on an 83 kg body during walking/balance. All motor
        # params (Kp, Kd, armature, friction, viscous_friction) match
        # waist_shoulder_elbow_x8_120 — same physical motor spec.
        "ankles_x8_120": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit=43.0,
            velocity_limit=13.0,
            stiffness=200.0,
            damping=20.0,
            armature=0.065893,
            friction=0.849291,
            viscous_friction=0.379237,
            min_delay=0,
            max_delay=6,
        ),
        # X6-60: wrist_roll only (both sides) = 2 joints
        "wristroll_x6_60": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_roll_joint",
            ],
            effort_limit=20.0,
            velocity_limit=16.0,
            stiffness=200.0,
            damping=25.0,
            armature=0.019603,
            friction=0.321816,
            viscous_friction=0.165640,
            min_delay=0,
            max_delay=6,
        ),
        # X8-120: waist (3) + shoulder pitch/roll/yaw + elbow (both sides) = 11 joints
        "waist_shoulder_elbow_x8_120": DelayedPDActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit=43.0,
            velocity_limit=13.0,
            stiffness=200.0,
            damping=20.0,
            armature=0.065893,
            friction=0.849291,
            viscous_friction=0.379237,
            min_delay=0,
            max_delay=6,
        ),
        # X4-36: wrist pitch/yaw + head pitch/roll/yaw = 7 joints
        "wrist_head_x4_36": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
                "head_pitch_joint",
                "head_yaw_joint",
            ],
            effort_limit=10.5,
            velocity_limit=8.6,
            stiffness=80.0,
            damping=3.0,
            armature=0.045213,
            friction=0.388995,
            viscous_friction=0.154954,
            min_delay=0,
            max_delay=6,
        ),
    },
)
"""HV1.2 custom humanoid (32 DoF) for the standing/walking task."""


# ---------------------------------------------------------------------------
# HV1 custom humanoid (31 DoF). Source: source/isaaclab_assets/data/custom_robot/
# Same body plan as HV1.2 except: 2-DoF neck (yaw + pitch) instead of 3-DoF head.
# End-effector links: left_wrist_yaw_link, right_wrist_yaw_link.
# Spawned from URDF (not the layered USD) for the same reason as HV1.2 — the
# Isaac-Sim-importer layered USD wrapper does not resolve cleanly under
# /World/envs/env_*/Robot. Isaac Lab regenerates a self-contained USD on first
# launch.
# ---------------------------------------------------------------------------
HV1_URDF_PATH = (
    "/home/rabisankar/IsaacLab/source/isaaclab_assets/data/custom_robot/"
    "urdf_mesh/hv1/hv1.urdf"
)

HV1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=HV1_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    ".*_hip_yaw_joint": 250.0,
                    ".*_hip_roll_joint": 250.0,
                    ".*_hip_pitch_joint": 250.0,
                    ".*_knee_joint": 250.0,
                    ".*_ankle_pitch_joint": 200.0,
                    ".*_ankle_roll_joint": 200.0,
                    ".*_wrist_roll_joint": 200.0,
                    "waist_yaw_joint": 250.0,
                    "waist_roll_joint": 250.0,
                    "waist_pitch_joint": 250.0,
                    ".*_shoulder_pitch_joint": 200.0,
                    ".*_shoulder_roll_joint": 200.0,
                    ".*_shoulder_yaw_joint": 200.0,
                    ".*_elbow_joint": 200.0,
                    ".*_wrist_pitch_joint": 80.0,
                    ".*_wrist_yaw_joint": 80.0,
                    "neck_yaw_joint": 80.0,
                    "neck_pitch_joint": 80.0,
                },
                damping={
                    ".*_hip_yaw_joint": 15.0,
                    ".*_hip_roll_joint": 15.0,
                    ".*_hip_pitch_joint": 15.0,
                    ".*_knee_joint": 15.0,
                    ".*_ankle_pitch_joint": 25.0,
                    ".*_ankle_roll_joint": 25.0,
                    ".*_wrist_roll_joint": 25.0,
                    "waist_yaw_joint": 15.0,
                    "waist_roll_joint": 15.0,
                    "waist_pitch_joint": 15.0,
                    ".*_shoulder_pitch_joint": 20.0,
                    ".*_shoulder_roll_joint": 20.0,
                    ".*_shoulder_yaw_joint": 20.0,
                    ".*_elbow_joint": 20.0,
                    ".*_wrist_pitch_joint": 3.0,
                    ".*_wrist_yaw_joint": 3.0,
                    "neck_yaw_joint": 3.0,
                    "neck_pitch_joint": 3.0,
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
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.40,
            ".*_ankle_pitch_joint": -0.20,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.30,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.30,
            ".*_wrist_roll_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_yaw_joint": 0.0,
            "neck_yaw_joint": 0.0,
            "neck_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # X12-320 class: hip pitch/roll/yaw + knee + waist  (both sides) = 8 joints, 85 Nm.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=85.0,
            velocity_limit_sim=10.0,
            stiffness=250.0,
            damping=15.0,
            armature=0.326938,
            friction=1.694307,
            viscous_friction=0.350134,
        ),
        # X6-60 class: ankle pitch/roll + wrist_roll (both sides) = 6 joints, 20 Nm.
        "ankle_wristroll": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=16.0,
            stiffness=200.0,
            damping=25.0,
            armature=0.019603,
            friction=0.321816,
            viscous_friction=0.165640,
        ),
        # X8-120 class: waist (3) + shoulder pitch/roll/yaw + elbow (both sides) = 11 joints, 43 Nm.
        "shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=43.0,
            velocity_limit_sim=13.0,
            stiffness=200.0,
            damping=20.0,
            armature=0.065893,
            friction=0.849291,
            viscous_friction=0.379237,
        ),
        # X4-36 class: wrist pitch/yaw + neck pitch/yaw = 6 joints, 10.5 Nm.
        "wrist_neck": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
                "neck_yaw_joint",
                "neck_pitch_joint",
            ],
            effort_limit_sim=10.5,
            velocity_limit_sim=8.6,
            stiffness=80.0,
            damping=3.0,
            armature=0.045213,
            friction=0.388995,
            viscous_friction=0.154954,
        ),
    },
)
"""HV1 custom humanoid (31 DoF) for the standing/walking task."""
