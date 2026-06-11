"""HV1 V5 world-frame loco-manipulation — pure EE-driven navigation (Option B).

V5 differences vs V4 (`loco_manip_v4_env_cfg.py`):
  * EE pose commands are **world-frame**, **episode-static**, sampled
    spherically around a per-arm anchor. The robot can no longer "stand
    still in KMP pose" — the world target stays put while the pelvis must
    navigate to within reach.
  * Action: `KMPResidualJointPositionActionV5Cfg` — same frozen KMP MLP as V4
    (`kmp_v1.pt`); `process_actions` runs a world→body transform on the EE
    target before packing the 16-D KMP input. See `kmp_action_v5.py`.
  * **Velocity command REMOVED.** No `base_velocity` reward, no
    `velocity_commands` observation. Walking direction and speed are
    EMERGENT from the world EE rewards + the navigation-progress reward —
    the policy decides for itself when to walk and where, based purely on
    the EE target. When the target is within reach without walking, the
    EE-gated stand-still reward pushes the policy to a clean stand.
  * Rewards (new in V5):
      - world_ee_position / orient + tanh-fine (per arm) — REPLACE V4 body
        EE rewards (V4 ones are zeroed in __post_init__).
      - navigation_progress — Δ(pelvis-to-target distance) per step. With
        velocity tracking gone, this is the primary "walk toward target"
        signal.
      - feet_air_time (EE-gated): biped single-stance bonus, fires only
        when the closer EE target is > 0.5 m from pelvis.
      - stand_still_legs (EE-gated): leg joint-deviation penalty, fires
        only when the closer EE target is within 0.5 m of pelvis.
      - facing_target (EE-gated): small reward for pelvis yaw aligned with
        the bearing to the closer target. "Turn-before-walking" gradient.
  * Observations: dropped `velocity_commands` from both groups; added
    `pelvis_to_target_xy_b` per arm (Δx, Δy, ‖xy‖ in pelvis-yaw frame).
    Critic still sees privileged `base_lin_vel`.
  * Curriculum: 3-stage distance cap — Stage A r ≤ 0.5 m (target within
    reach, no walking), Stage B r ≤ 2.0 m (a few steps), Stage C r ≤ 5.0 m
    (long-range navigation). Triggered by `common_step_counter` thresholds.
  * Episode length 14 → 15 s (worst-case 5 m at ~0.5 m/s + reach).

V4-side things kept identical:
  * KMP MLP, residual scale dict (legs 0.25 / arms 0.10 / waist 0.10).
  * `body_height` and `waist_regularization` scalar commands.
  * Asymmetric actor / critic obs split, 5-step policy history.
  * Shaping rewards (action_rate, joint_deviation, base_height_tracking,
    base_height_above, torso flat orientation, etc.).

Warm-start: train resumed from a converged V4 checkpoint. Leg gait and arm
posture transfer; PPO learns WHERE to put the pelvis. Obs dim shrinks by
3 (velocity_commands removed) and grows by 6 (pelvis_to_target × 2), net +3.
RSL-RL re-inits the first MLP layer on the dim mismatch.

Why "base_velocity" command still exists (with zero ranges):
  Multiple inherited reward terms call `env.command_manager.get_command(
  "base_velocity")` and would crash if the command were deleted. Setting
  the ranges to (0, 0, 0) makes the command a no-op (always zero) and
  setting the affected reward weights to 0 short-circuits the gradient.
  Cheaper than rewriting every inherited reward.
"""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity import mdp as custom_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.flat_env_cfg import (
    ARM_JOINT_NAMES,
    LEG_JOINTS,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.kmp_action_v5 import (
    KMPResidualJointPositionActionV5Cfg,
)

from .loco_manip_env_cfg import LEFT_EE_BODY, RIGHT_EE_BODY
from .loco_manip_v2_env_cfg import WAIST_ACTUATED_JOINTS
from .loco_manip_v4_env_cfg import (
    HV1LocoManipV4ActionsCfg,
    HV1LocoManipV4EnvCfg,
    HV1LocoManipV4RewardsCfg,
    _KMP_RESIDUAL_SCALE,
    KMP_CKPT,
)
from .loco_manip_v3_env_cfg import (
    HV1LocoManipV3CurriculumCfg,
    HV1LocoManipV3ObservationsCfg,
)


# ---- actions: V5 KMP-residual (world→body transform inside) -----------------
@configclass
class HV1LocoManipV5ActionsCfg(HV1LocoManipV4ActionsCfg):
    joint_pos = KMPResidualJointPositionActionV5Cfg(
        asset_name="robot",
        joint_names=LEG_JOINTS + ARM_JOINT_NAMES + WAIST_ACTUATED_JOINTS,
        preserve_order=True,
        kmp_checkpoint=KMP_CKPT,
        scale=_KMP_RESIDUAL_SCALE,
        residual_scale=None,
        use_default_offset=False,
    )


# ---- observations: V3 base − velocity_commands + pelvis-to-target XY ----------
@configclass
class HV1LocoManipV5ObservationsCfg(HV1LocoManipV3ObservationsCfg):
    @configclass
    class PolicyCfg(HV1LocoManipV3ObservationsCfg.PolicyCfg):
        # Drop the inherited velocity-command observation. Setting to None on
        # a configclass field removes the term from the ObservationManager.
        velocity_commands = None
        # Navigation gradient — see custom_mdp.pelvis_to_target_xy_b docstring.
        left_pelvis_to_target = ObsTerm(
            func=custom_mdp.pelvis_to_target_xy_b,
            params={"command_name": "world_left_ee_pose"},
        )
        right_pelvis_to_target = ObsTerm(
            func=custom_mdp.pelvis_to_target_xy_b,
            params={"command_name": "world_right_ee_pose"},
        )

    @configclass
    class CriticCfg(HV1LocoManipV3ObservationsCfg.CriticCfg):
        velocity_commands = None
        left_pelvis_to_target = ObsTerm(
            func=custom_mdp.pelvis_to_target_xy_b,
            params={"command_name": "world_left_ee_pose"},
        )
        right_pelvis_to_target = ObsTerm(
            func=custom_mdp.pelvis_to_target_xy_b,
            params={"command_name": "world_right_ee_pose"},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---- rewards: V4 rewards + world EE replacements + nav progress -------------
@configclass
class HV1LocoManipV5RewardsCfg(HV1LocoManipV4RewardsCfg):
    """V4 rewards with body-EE replaced by world-EE and a nav-progress bonus.

    The V4 body-frame EE rewards (`left_ee_pos_tracking*`, `right_ee_*`) are
    zeroed in __post_init__ so the policy stops reading them — they're still
    declared in the cfg chain (from V1's `HV1LocoManipRewardsCfg`) but their
    weight is 0 in V5. The world-frame equivalents replace them.
    """

    # Coarse L2 — keeps a gradient even when the target is far (tanh saturates).
    left_world_ee_pos = RewTerm(
        func=custom_mdp.world_ee_position_command_error,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE_BODY),
            "command_name": "world_left_ee_pose",
        },
    )
    right_world_ee_pos = RewTerm(
        func=custom_mdp.world_ee_position_command_error,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE_BODY),
            "command_name": "world_right_ee_pose",
        },
    )

    # Fine `1 - tanh(err/std)` — peaks at err=0, dies off after ~3·std.
    # std=0.15 matches V3 stage-3 fine std so the close-range gradient is
    # similar to what the V4 ckpt was trained with.
    left_world_ee_pos_fine = RewTerm(
        func=custom_mdp.world_ee_position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE_BODY),
            "command_name": "world_left_ee_pose",
            "std": 0.15,
        },
    )
    right_world_ee_pos_fine = RewTerm(
        func=custom_mdp.world_ee_position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE_BODY),
            "command_name": "world_right_ee_pose",
            "std": 0.15,
        },
    )

    # Orientation — only matters once the EE is close. Smaller weight than pos
    # since orient is a secondary objective for general reach.
    left_world_ee_orient = RewTerm(
        func=custom_mdp.world_ee_orientation_command_error,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE_BODY),
            "command_name": "world_left_ee_pose",
        },
    )
    right_world_ee_orient = RewTerm(
        func=custom_mdp.world_ee_orientation_command_error,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE_BODY),
            "command_name": "world_right_ee_pose",
        },
    )

    # Navigation progress — rewards Δ(distance to target) per step, summed
    # over both arms. With base_velocity removed this is the PRIMARY "walk
    # toward target" signal. Weight bumped from 10 → 20 to compensate for
    # losing track_lin_vel_xy_exp (=4.0).
    navigation_progress = RewTerm(
        func=custom_mdp.navigation_progress_reward,
        weight=20.0,
        params={"command_names": ["world_left_ee_pose", "world_right_ee_pose"]},
    )

    # EE-distance-gated gait/stand-still — replace the velocity-gated
    # versions inherited from V1/V4 (those are zeroed in __post_init__).
    # `distance_threshold = 0.5 m` matches Stage-A r_max so Stage-A targets
    # are mostly "within reach → stand mode", Stage-B/C → walk mode.

    feet_air_time_ee = RewTerm(
        func=custom_mdp.feet_air_time_world_ee_positive_biped,
        weight=4.0,
        params={
            "command_names": ["world_left_ee_pose", "world_right_ee_pose"],
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_pitch_link"
            ),
            "threshold": 0.3,
            "distance_threshold": 0.5,
        },
    )
    stand_still_legs_ee = RewTerm(
        func=custom_mdp.stand_still_world_ee_joint_deviation_l1,
        weight=-4.0,
        params={
            "command_names": ["world_left_ee_pose", "world_right_ee_pose"],
            "distance_threshold": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "^(left|right)_hip_(yaw|pitch|roll)_joint$",
                    "^(left|right)_ankle_(pitch|roll)_joint$",
                ],
            ),
        },
    )

    # Heading-toward-target reward. Small positive weight; gives a
    # gradient for "turn before you walk" so the policy can start
    # rotating before translation begins.
    facing_target = RewTerm(
        func=custom_mdp.facing_target_world_ee,
        weight=0.5,
        params={
            "command_names": ["world_left_ee_pose", "world_right_ee_pose"],
            "distance_threshold": 0.5,
        },
    )


# ---- curriculum: distance cap unlock + V3 height curriculum ------------------
@configclass
class HV1LocoManipV5CurriculumCfg(HV1LocoManipV3CurriculumCfg):
    """Two extra curricula on top of V3's height-tracking unlock.

    Stage A (default at init): r_max = 0.5 m for both arms — target is within
    reach without walking. Lets the actor first re-learn the world EE reward
    shape via the existing KMP+arm machinery (V4 warm-start = arm reach is
    already solved).

    Stage B (iter 1500 = 36000 steps): r_max = 2.0 m. Robot must walk a few
    steps to reach. The navigation-progress reward dominates here.

    Stage C (iter 6000 = 144000 steps): r_max = 5.0 m. Long-range navigation.

    `num_steps_per_env` = 24, so iter N ≈ 24*N steps.
    """

    world_ee_r_stage_b = CurrTerm(
        func=custom_mdp.modify_world_ee_distance_cap,
        params={
            "command_names": ["world_left_ee_pose", "world_right_ee_pose"],
            "r_min": 0.3,
            "r_max": 2.0,
            "num_steps": 36000,
        },
    )
    world_ee_r_stage_c = CurrTerm(
        func=custom_mdp.modify_world_ee_distance_cap,
        params={
            "command_names": ["world_left_ee_pose", "world_right_ee_pose"],
            "r_min": 0.3,
            "r_max": 5.0,
            "num_steps": 144000,
        },
    )


# ---- env --------------------------------------------------------------------
@configclass
class HV1LocoManipV5EnvCfg(HV1LocoManipV4EnvCfg):
    """V5 = V4 + world-frame EE + distance curriculum."""

    actions: HV1LocoManipV5ActionsCfg = HV1LocoManipV5ActionsCfg()
    observations: HV1LocoManipV5ObservationsCfg = HV1LocoManipV5ObservationsCfg()
    rewards: HV1LocoManipV5RewardsCfg = HV1LocoManipV5RewardsCfg()
    curriculum: HV1LocoManipV5CurriculumCfg = HV1LocoManipV5CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- Replace body-frame EE commands with world-frame --------------
        # The V4 chain registered `left_ee_pose` / `right_ee_pose` as
        # UniformPoseCommands (body-frame, ~4s resample). Drop them and
        # register the world-frame versions instead.
        self.commands.left_ee_pose = None
        self.commands.right_ee_pose = None

        # Episode-static lifetime: resample only on reset (no per-step or
        # mid-episode resample). Set the resampling time range >= episode
        # length so the internal timer never fires mid-episode.
        _RESAMPLE_INFTY = (1e6, 1e6)

        # Spherical sampling defaults — Stage A bounds. Curriculum overrides
        # r at iter thresholds. anchor_xy splits the two arms left/right so
        # the robot doesn't have to walk to the same side for both.
        # Hemispheres in front of the robot (theta ∈ [-π/2, π/2]) — backward
        # reach is unusual for manipulation. Left arm tilts to +y, right -y.
        self.commands.world_left_ee_pose = custom_mdp.WorldFramePoseCommandCfg(
            asset_name="robot",
            body_name=LEFT_EE_BODY,
            resampling_time_range=_RESAMPLE_INFTY,
            debug_vis=True,
            anchor_xy=(0.0, 0.2),
            anchor_z=0.94,
            ranges=custom_mdp.WorldFramePoseCommandCfg.Ranges(
                r=(0.1, 0.5),                # Stage A: within reach
                theta=(-math.pi / 4, 3 * math.pi / 4),  # front + left
                phi=(-math.pi / 6, math.pi / 3),         # mostly horizontal/upper
                roll=(-0.3, 0.3),
                pitch=(-0.3, 0.3),
                yaw=(-0.3, 0.3),
            ),
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
                theta=(-3 * math.pi / 4, math.pi / 4),    # front + right
                phi=(-math.pi / 6, math.pi / 3),
                roll=(-0.3, 0.3),
                pitch=(-0.3, 0.3),
                yaw=(-0.3, 0.3),
            ),
        )

        # --- Zero out V4 body-frame EE rewards ----------------------------
        # Declared in HV1LocoManipRewardsCfg (V1) and carried through the
        # chain. Setting weight=0 silences them — RewardManager still
        # constructs them but they contribute nothing to the loss.
        # NOTE: must not touch params (the named command is gone, so calling
        # them would crash). Setting weight=0 short-circuits evaluation.
        for _name in (
            "left_ee_pos_tracking",
            "right_ee_pos_tracking",
            "left_ee_pos_tracking_fine",
            "right_ee_pos_tracking_fine",
            "left_ee_orient_tracking",
            "right_ee_orient_tracking",
        ):
            if hasattr(self.rewards, _name):
                # Reroute the dead term to a world EE command so RewardManager
                # can still look up the command name — weight=0 means the
                # output is multiplied to zero. Cheaper than full removal.
                _term = getattr(self.rewards, _name)
                _term.weight = 0.0
                if "command_name" in _term.params:
                    if "left" in _name:
                        _term.params["command_name"] = "world_left_ee_pose"
                    else:
                        _term.params["command_name"] = "world_right_ee_pose"

        # --- Episode length bump -----------------------------------------
        # Stage C (5 m at 0.5 m/s) needs ~10 s of pure walking before reach.
        # 15 s gives headroom for the reach + a few course corrections.
        self.episode_length_s = 15.0

        # --- Option B: neutralize base_velocity command + velocity rewards
        # Force all velocity command ranges to (0, 0). Inherited reward
        # terms that read `base_velocity` (track_lin_vel_xy_exp,
        # track_ang_vel_z_exp, inherited feet_air_time, inherited
        # stand_still_legs) still execute but always see a zero command.
        # Their weights are zeroed below so their contribution to the total
        # reward is identically zero, but the function calls themselves
        # don't crash since the command term still exists.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        # Zero the velocity-tracking rewards.
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 0.0
        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Zero the velocity-gated gait/stand terms inherited from V4. The
        # V5 RewardsCfg declares EE-distance-gated replacements
        # (`feet_air_time_ee`, `stand_still_legs_ee`) at static cfg time.
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.weight = 0.0
        if hasattr(self.rewards, "stand_still_legs"):
            self.rewards.stand_still_legs.weight = 0.0

        # --- Reprint effective reward weights -----------------------------
        print("\n=== HV1 V5 effective reward weights (post-inheritance) ===")
        for _name in sorted(vars(self.rewards)):
            _term = getattr(self.rewards, _name)
            _w = getattr(_term, "weight", None)
            if _w is not None:
                print(f"  {_name:36s} = {_w:+.4f}")
        print()


@configclass
class HV1LocoManipV5EnvCfg_PLAY(HV1LocoManipV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 6.0  # wider so 5-m targets fit
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        # Option B: velocity command is a stub — PLAY also leaves it at zero.
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)

        # PLAY uses Stage C bounds (full range) so the policy is exercised.
        self.commands.world_left_ee_pose.ranges.r = (0.3, 3.0)
        self.commands.world_right_ee_pose.ranges.r = (0.3, 3.0)

        self.commands.body_height.range = (0.85, 0.95)
        self.commands.waist_regularization.range = (0.5, 2.0)
        self.commands.waist_regularization.log_uniform = False

        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
        }
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
