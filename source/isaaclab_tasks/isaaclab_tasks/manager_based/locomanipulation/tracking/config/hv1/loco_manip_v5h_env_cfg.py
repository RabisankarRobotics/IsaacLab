"""HV1 V5-H — paper-aligned hierarchical world-frame loco-manipulation.

Implements HiWET's two-stage hierarchy (paper §III, IV, V):

  Stage 1 (V4, FROZEN) — body-frame tracker. Takes
      u_t = [v_b^des(3), h^des(1), bT_L^des(7), bT_R^des(7), α_t(1)]
      and produces 28-D joint residual; KMP + PD downstream.
      Loaded from JIT-exported `deploy/model/.../policy.pt` (V4 24k ckpt).

  Stage 2 (THIS env) — world-frame commander. Trained from scratch.
      Observation: proprio + world base pose + world EE targets + mask
      Action:     u_t (19-D, exactly Stage 1's input vector)
      Reward:     world EE pos error (masked, L2 + tanh-fine) + workspace
                  distance + base-heading alignment.

The Stage 2 action class (`Stage2WrappedAction` in `stage2_action.py`)
writes the action into V4's command buffers each step, runs V4 inference
to produce the 28-D residual, runs KMP for q_prior, sets joint targets.

Why this matches the paper (and V5 monolithic does not):
  * Stage 1 stays untouched -> V4 MuJoCo deploy still works as-is.
  * Stage 2 has a 19-D action and a small observation -> tiny MLP, fast
    training, ~thousands of iters rather than tens of thousands.
  * Success-rate curriculum (Eq. 17) follows the paper.

Differences from V5 (monolithic Option B):
  * V5 dropped `base_velocity` entirely; V5-H keeps it because Stage 2's
    output PROVIDES it (paper's u_t includes v_b^des). V4 still consumes
    it normally during inference.
  * V5 retrained the whole stack; V5-H freezes V4.
  * V5 used a single observation set; V5-H exposes two — `policy` /
    `critic` (for Stage 2's actor / critic) and `v4_actor` (the obs Stage
    1 needs to run inference, replicated from V4's training-time layout).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity import mdp as custom_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.flat_env_cfg import (
    ARM_JOINT_NAMES,
    LEG_JOINTS,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.stage2_action import (
    Stage2WrappedActionCfg,
)

from .loco_manip_env_cfg import LEFT_EE_BODY, RIGHT_EE_BODY
from .loco_manip_v2_env_cfg import (
    HV1LocoManipV2EnvCfg,
    HV1LocoManipV2RewardsCfg,
    TORSO_BODY,
    WAIST_ACTUATED_JOINTS,
)
from .loco_manip_v3_env_cfg import HV1LocoManipV3ObservationsCfg
from .loco_manip_v4_env_cfg import KMP_CKPT, _KMP_RESIDUAL_SCALE


# Frozen V4 actor — JIT-exported by play.py.
V4_JIT_POLICY = "/home/rabisankar/IsaacLab/deploy/model/kmp_base_height_ee_tracking/policy.pt"


# ---- per-arm goal markers (left = red, right = blue, 6 cm spheres) ---------
# Each command term needs its OWN prim_path or the second instance overwrites
# the first inside USD and only one marker is visible at runtime.
_LEFT_GOAL_MARKER = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/world_left_ee_goal",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
        ),
    },
)
_RIGHT_GOAL_MARKER = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/world_right_ee_goal",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
        ),
    },
)


# ---- actions: Stage 2 wraps frozen V4 + KMP --------------------------------
@configclass
class HV1LocoManipV5HActionsCfg:
    """Single action term — Stage 2 19-D command wired to frozen V4 inside."""

    joint_pos = Stage2WrappedActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS + ARM_JOINT_NAMES + WAIST_ACTUATED_JOINTS,
        preserve_order=True,
        v4_jit_policy=V4_JIT_POLICY,
        v4_obs_group_name="v4_actor",
        kmp_checkpoint=KMP_CKPT,
        scale=_KMP_RESIDUAL_SCALE,
        residual_scale=None,
        use_default_offset=False,
    )


# ---- observations: 3 groups — policy / critic / v4_actor --------------------
# - `policy`  : Stage 2 actor obs (proprio + world base + world EE + mask)
# - `critic`  : Stage 2 critic obs (= policy + privileged base_lin_vel + world EE error)
# - `v4_actor`: REPLICA of V4's training-time PolicyCfg, used inside the action
#               class to feed the frozen V4 policy. Must match V4's obs order
#               exactly (history_length=5, 121 features/step -> 605 total).
@configclass
class HV1LocoManipV5HObservationsCfg:
    """3 obs groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception — actor-only, minimal IMU + joint state.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        projected_gravity_torso = ObsTerm(
            func=custom_mdp.projected_gravity_body,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        torso_ang_vel = ObsTerm(
            func=custom_mdp.body_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        # World-frame components per paper Eq. 13:  w T_b + world EE + mask
        world_base_pose = ObsTerm(func=custom_mdp.world_base_pose_obs)
        world_left_ee_pose = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "world_left_ee_pose"}
        )
        world_right_ee_pose = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "world_right_ee_pose"}
        )
        ee_active_mask = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_active_mask"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # Stage 2 critic — same as policy + privileged base_lin_vel.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        projected_gravity_torso = ObsTerm(
            func=custom_mdp.projected_gravity_body,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        torso_ang_vel = ObsTerm(
            func=custom_mdp.body_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        world_base_pose = ObsTerm(func=custom_mdp.world_base_pose_obs)
        world_left_ee_pose = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "world_left_ee_pose"}
        )
        world_right_ee_pose = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "world_right_ee_pose"}
        )
        ee_active_mask = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_active_mask"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class V4ActorCfg(ObsGroup):
        """REPLICA of V3 PolicyCfg = V4's actor input. DO NOT modify ordering —
        the JIT-loaded V4 actor expects this exact layout (605-D after history).
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        projected_gravity_torso = ObsTerm(
            func=custom_mdp.projected_gravity_body,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        torso_ang_vel = ObsTerm(
            func=custom_mdp.body_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_ee_pose"}
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_ee_pose"}
        )
        body_height_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "body_height"}
        )
        waist_alpha_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "waist_regularization"}
        )

        def __post_init__(self):
            # Same 5-step history V4 was trained with.
            self.history_length = 5
            # No corruption — V4 saw raw obs at deploy time via play.py.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    v4_actor: V4ActorCfg = V4ActorCfg()


# ---- rewards: workspace + alignment + masked EE tracking --------------------
@configclass
class HV1LocoManipV5HRewardsCfg(HV1LocoManipV2RewardsCfg):
    """Stage 2 rewards (paper §V.C).

    Inherits V2's stability + torso shaping. World EE tracking + workspace
    distance + base alignment are the V5-H-specific signals.
    The V2 body-frame EE rewards stay declared but get zeroed in __post_init__
    (they reference body-frame `left_ee_pose` which is now Stage 2's *output*,
    not a sampled target — no tracking sense there).
    """

    # Workspace: planar distance pelvis→(closer target). Drives gross
    # navigation. Modest weight; the nav-progress kernel from V5 is more
    # informative per step, but workspace gives a steady baseline gradient
    # even when nav-progress saturates.
    workspace_distance = RewTerm(
        func=custom_mdp.workspace_distance_reward,
        weight=-1.0,
        params={"command_names": ["world_left_ee_pose", "world_right_ee_pose"]},
    )

    # Base heading alignment (paper §V.C "active steering").
    base_alignment = RewTerm(
        func=custom_mdp.base_heading_alignment_reward,
        weight=1.0,
        params={
            "command_names": ["world_left_ee_pose", "world_right_ee_pose"],
            "distance_threshold": 0.5,
        },
    )

    # Per-arm L2 + tanh-fine, multiplied by mask (= 1.0 for V5-H first run).
    left_world_ee_pos = RewTerm(
        func=custom_mdp.world_ee_position_error_masked,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE_BODY),
            "command_name": "world_left_ee_pose",
            "mask_command_name": "ee_active_mask",
            "mask_slot": 0,
        },
    )
    right_world_ee_pos = RewTerm(
        func=custom_mdp.world_ee_position_error_masked,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE_BODY),
            "command_name": "world_right_ee_pose",
            "mask_command_name": "ee_active_mask",
            "mask_slot": 1,
        },
    )
    left_world_ee_pos_fine = RewTerm(
        func=custom_mdp.world_ee_position_error_masked_tanh,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE_BODY),
            "command_name": "world_left_ee_pose",
            "std": 0.15,
            "mask_command_name": "ee_active_mask",
            "mask_slot": 0,
        },
    )
    right_world_ee_pos_fine = RewTerm(
        func=custom_mdp.world_ee_position_error_masked_tanh,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE_BODY),
            "command_name": "world_right_ee_pose",
            "std": 0.15,
            "mask_command_name": "ee_active_mask",
            "mask_slot": 1,
        },
    )


# ---- curriculum: success-rate gated radius expansion ------------------------
@configclass
class HV1LocoManipV5HCurriculumCfg:
    """Paper Eq. 17 — expand world-EE sampling range only when the policy is
    actually solving the current range. `error_threshold` = 0.10 m matches
    a "reachable" target; `delta_r` = 0.25 m per success-check.
    """

    world_ee_radius = CurrTerm(
        func=custom_mdp.modify_world_ee_distance_cap_on_success,
        params={
            "command_names": ["world_left_ee_pose", "world_right_ee_pose"],
            "error_metric_name": "position_error",
            "error_threshold": 0.10,
            "delta_r": 0.25,
            "r_max_ceiling": 5.0,
            "check_every_n_steps": 2400,  # ~iter 100
            "min_envs_below_threshold_frac": 0.6,
        },
    )


# ---- env --------------------------------------------------------------------
@configclass
class HV1LocoManipV5HEnvCfg(HV1LocoManipV2EnvCfg):
    """V5-H Stage 2 env. Inherits V2 (the cleanest pre-V3 base) and replaces
    actions/observations/rewards with the hierarchical Stage 2 set."""

    actions: HV1LocoManipV5HActionsCfg = HV1LocoManipV5HActionsCfg()
    observations: HV1LocoManipV5HObservationsCfg = HV1LocoManipV5HObservationsCfg()
    rewards: HV1LocoManipV5HRewardsCfg = HV1LocoManipV5HRewardsCfg()
    curriculum: HV1LocoManipV5HCurriculumCfg = HV1LocoManipV5HCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- Replace V2 body-frame EE commands with world-frame --------------
        # We KEEP `left_ee_pose` / `right_ee_pose` because V4's obs replica
        # needs them — but they become Stage 2 *outputs*, not random samples.
        # The Stage 2 action class writes into their buffers each step.
        # Curate their sampling: zero range, never resample so values stay
        # exactly what Stage 2 wrote (UniformPoseCommand resamples on a
        # timer regardless).
        _RESAMPLE_INFTY = (1e6, 1e6)
        if hasattr(self.commands, "left_ee_pose") and self.commands.left_ee_pose is not None:
            self.commands.left_ee_pose.resampling_time_range = _RESAMPLE_INFTY
        if hasattr(self.commands, "right_ee_pose") and self.commands.right_ee_pose is not None:
            self.commands.right_ee_pose.resampling_time_range = _RESAMPLE_INFTY

        # Velocity command: stop random sampling — Stage 2 writes it.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = _RESAMPLE_INFTY

        # body_height and waist_regularization: declared in V3 normally —
        # the V2 base does not have them. We add them here so V4's obs has
        # something to read. Stage 2 will overwrite each step.
        self.commands.body_height = custom_mdp.UniformScalarCommandCfg(
            resampling_time_range=_RESAMPLE_INFTY,
            range=(0.85, 0.95),
            log_uniform=False,
            metric_source="root_pos_z",
            debug_vis=False,
        )
        self.commands.waist_regularization = custom_mdp.UniformScalarCommandCfg(
            resampling_time_range=_RESAMPLE_INFTY,
            range=(0.1, 3.0),
            log_uniform=True,
        )

        # --- New world-frame EE commands (per V5 design, episode-static) -----
        # debug_vis = True draws a colored sphere at the target world position
        # in every env. Left = RED, right = BLUE. Each command term gets its
        # own USD prim path so they don't overwrite each other's marker.
        self.commands.world_left_ee_pose = custom_mdp.WorldFramePoseCommandCfg(
            asset_name="robot",
            body_name=LEFT_EE_BODY,
            resampling_time_range=_RESAMPLE_INFTY,
            debug_vis=True,
            anchor_xy=(0.0, 0.2),
            anchor_z=0.94,
            ranges=custom_mdp.WorldFramePoseCommandCfg.Ranges(
                r=(0.1, 0.5),  # Stage A — curriculum widens
                theta=(-math.pi / 4, 3 * math.pi / 4),
                phi=(-math.pi / 6, math.pi / 3),
                roll=(-0.3, 0.3),
                pitch=(-0.3, 0.3),
                yaw=(-0.3, 0.3),
            ),
            goal_pose_visualizer_cfg=_LEFT_GOAL_MARKER,
        )
        self.commands.world_right_ee_pose = custom_mdp.WorldFramePoseCommandCfg(
            asset_name="robot",
            body_name=RIGHT_EE_BODY,
            resampling_time_range=_RESAMPLE_INFTY,
            debug_vis=True,
            anchor_xy=(0.0, -0.2),
            anchor_z=0.94,
            ranges=custom_mdp.WorldFramePoseCommandCfg.Ranges(
                r=(0.1, 0.5),
                theta=(-3 * math.pi / 4, math.pi / 4),
                phi=(-math.pi / 6, math.pi / 3),
                roll=(-0.3, 0.3),
                pitch=(-0.3, 0.3),
                yaw=(-0.3, 0.3),
            ),
            goal_pose_visualizer_cfg=_RIGHT_GOAL_MARKER,
        )

        # --- Mask command (always [1, 1] for V5-H first run) -----------------
        self.commands.ee_active_mask = custom_mdp.ConstantVectorCommandCfg(
            resampling_time_range=_RESAMPLE_INFTY,
            value=(1.0, 1.0),
        )

        # --- Zero V2 body-frame EE rewards (no longer a tracking target) -----
        for _name in (
            "left_ee_pos_tracking",
            "right_ee_pos_tracking",
            "left_ee_pos_tracking_fine",
            "right_ee_pos_tracking_fine",
            "left_ee_orient_tracking",
            "right_ee_orient_tracking",
        ):
            if hasattr(self.rewards, _name):
                getattr(self.rewards, _name).weight = 0.0

        # Stage 2 has no velocity reward — V4 figures velocity out internally.
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 0.0
        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Episode length 15s (walk + reach budget)
        self.episode_length_s = 15.0

        print("\n=== HV1 V5-H effective reward weights ===")
        for _name in sorted(vars(self.rewards)):
            _term = getattr(self.rewards, _name)
            _w = getattr(_term, "weight", None)
            if _w is not None:
                print(f"  {_name:36s} = {_w:+.4f}")
        print()


@configclass
class HV1LocoManipV5HEnvCfg_PLAY(HV1LocoManipV5HEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        # PLAY uses Stage-C-ish bounds so the policy is exercised.
        self.commands.world_left_ee_pose.ranges.r = (0.3, 3.0)
        self.commands.world_right_ee_pose.ranges.r = (0.3, 3.0)
        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
        }
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
