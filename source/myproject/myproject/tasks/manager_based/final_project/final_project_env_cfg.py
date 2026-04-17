import json
import os
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1Rewards, H1RoughEnvCfg

from myproject.assets.unitree_humanoid_config import UNITREE_H1_MINIMAL_CFG

from . import mdp as custom_mdp

_ASSETS_ENV_DIR = Path(__file__).resolve().parents[3] / "assets" / "environments"
_CURRICULUM_ARENA_DIR = _ASSETS_ENV_DIR / "curriculum_arenas"

FINAL_MAP_USD_PATH = str(_ASSETS_ENV_DIR / "final_map_2.usd")
FINAL_MAP_USD_PRIM_PATH = "/World/ground"
CURRICULUM_ARENA_STABILITY_USD_PATH = str(_CURRICULUM_ARENA_DIR / "stability_arena.usda")
CURRICULUM_ARENA_CROSSING_USD_PATH = str(_CURRICULUM_ARENA_DIR / "crossing_arena.usda")
CURRICULUM_ARENA_STABILITY_LAYOUT_PATH = str(_CURRICULUM_ARENA_DIR / "stability_arena.layout.json")
CURRICULUM_ARENA_CROSSING_LAYOUT_PATH = str(_CURRICULUM_ARENA_DIR / "crossing_arena.layout.json")
# Real bounds measured from final_map_2.usd:
#   min = (-7.6, -3.85, -1.0)
#   max = (7.6, 3.85, 0.19)
#   size = (15.2, 7.7, 1.19)
# We keep the curriculum terrain tiles at the same horizontal footprint as the map so obstacle
# spacing and traversal distance are closer before transferring to the real arena.
REAL_MAP_SIZE_XY = (15.2, 7.7)
GOAL_X = 6.8
MAP_ENV_SPACING = 0
SPAWN_Z_CLEARANCE = 0.08
MAP_START_POS = (-6.5, 0, 1.05 + SPAWN_Z_CLEARANCE)
MAP_START_ROT = (1.0, 0.0, 0.0, 0.0)
MAP_START_POS_JITTER_XY = (0.3, 0.5)
CURRICULUM_ARENA_START_POS_JITTER_XY = (0.4, 0.4)
BASELINE_GOAL_PROGRESS_WEIGHT = 4.0
BASELINE_FEET_AIR_TIME_WEIGHT = 4.0
BASELINE_FEET_AIR_TIME_MIN = 0.04
BASELINE_FEET_AIR_TIME_TARGET = 0.14
BASELINE_FEET_STANCE_TIME_MIN = 0.04
BASELINE_STEP_FORWARD_SPEED_MIN = 0.05
BASELINE_TRACK_LIN_VEL_WEIGHT = 0.25
BASELINE_UPRIGHT_SURVIVAL_WEIGHT = 0.5
CURRICULUM_PROXY_GOAL_X = 2.75
CURRICULUM_GOAL_PROGRESS_WEIGHT = 2.5
CURRICULUM_FEET_AIR_TIME_WEIGHT = 18.0
CURRICULUM_TRACK_LIN_VEL_WEIGHT = 0.015
CURRICULUM_UPRIGHT_SURVIVAL_WEIGHT = 1.5
CURRICULUM_STAND_UP_HEIGHT_WEIGHT = 4.0
ROUGH_GOAL_BASELINE_TRACK_ANG_VEL_WEIGHT = 0.25
ROUGH_GOAL_BASELINE_TERMINATION_PENALTY = -25.0
ROUGH_GOAL_BASELINE_GOAL_PROGRESS_WEIGHT = 2.0
ROUGH_GOAL_BASELINE_GOAL_REACHED_BONUS_WEIGHT = 50.0
ROUGH_GOAL_BASELINE_LIN_VEL_X_RANGE = (0.25, 0.5)
MINIMAL_REWARD_GOAL_PROGRESS_WEIGHT = 2.0
MINIMAL_REWARD_GOAL_REACHED_BONUS_WEIGHT = 50.0
MINIMAL_REWARD_LIN_VEL_X_RANGE = (0.25, 0.5)
SPEEDRUN_LIN_VEL_X_RANGE = (1.0, 1.5)
SPEEDRUN_TERMINATION_PENALTY = -5.0
FASTWALK_LIN_VEL_X_RANGE = (1.0, 1.5)
AUTOPILOT_PROFILE_ENV_VAR = "FINAL_PROJECT_AUTOPILOT_PROFILE"
STABILITY_WARMUP_PROXY_GOAL_X = 4.0


def _load_autopilot_profile() -> dict:
    """Load bounded autopilot tuning overrides from a JSON file when present."""

    profile_path = os.environ.get(AUTOPILOT_PROFILE_ENV_VAR)
    if not profile_path:
        return {}
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return profile if isinstance(profile, dict) else {}


def _get_autopilot_override(section: str) -> dict:
    overrides = _load_autopilot_profile().get("sections", {})
    selected = overrides.get(section, {})
    return selected if isinstance(selected, dict) else {}


def _tuple2(value, default):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return default


def _tune_shared_arena_physx_buffers_for_training_headless(env_cfg) -> None:
    """Use large GPU PhysX capacities for dense headless humanoid-on-mesh training."""

    env_cfg.sim.physx.gpu_collision_stack_size = 2**28
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**24
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**23
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**23
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
    env_cfg.sim.physx.gpu_heap_capacity = 2**27
    env_cfg.sim.physx.gpu_temp_buffer_capacity = 2**25


def _tune_shared_arena_physx_buffers_for_training_gui(env_cfg) -> None:
    """Use reduced GPU PhysX capacities so GUI training fits limited VRAM."""

    env_cfg.sim.physx.gpu_collision_stack_size = 2**26
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**22
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**20
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**21
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**23
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21
    env_cfg.sim.physx.gpu_heap_capacity = 2**25
    env_cfg.sim.physx.gpu_temp_buffer_capacity = 2**23


def _tune_shared_arena_physx_buffers_for_play(env_cfg) -> None:
    """Use smaller GPU PhysX capacities so single-env PLAY fits limited VRAM."""

    env_cfg.sim.physx.gpu_collision_stack_size = 2**24
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**19
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**21
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**19
    env_cfg.sim.physx.gpu_heap_capacity = 2**24
    env_cfg.sim.physx.gpu_temp_buffer_capacity = 2**22


def _tune_shared_arena_physx_buffers_for_training(env_cfg) -> None:
    """Choose a shared-arena training PhysX profile based on execution mode."""

    profile = getattr(env_cfg, "shared_arena_physx_profile", "training_headless")
    if profile == "training_gui":
        _tune_shared_arena_physx_buffers_for_training_gui(env_cfg)
    elif profile == "training_headless":
        _tune_shared_arena_physx_buffers_for_training_headless(env_cfg)
    else:
        raise ValueError(
            f"Unsupported shared_arena_physx_profile='{profile}'. "
            "Expected one of {'training_headless', 'training_gui'}."
        )


def _make_usd_terrain_cfg(usd_path: str) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=usd_path,
    )


def _make_final_map_terrain_cfg(num_envs: int, env_spacing: float) -> TerrainImporterCfg:
    return _make_usd_terrain_cfg(FINAL_MAP_USD_PATH)


def _load_arena_layout(layout_path: str) -> dict:
    with open(layout_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_spawn_z_clearance(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """Raise spawn poses slightly so feet do not start inside the terrain mesh."""

    return (pos[0], pos[1], pos[2] + SPAWN_Z_CLEARANCE)


FINAL_PROJECT_CURRICULUM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=REAL_MAP_SIZE_XY,
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Old aggressive proportions for reference:
        # flat=0.20, rough_easy=0.20, rough_hard=0.15, gaps=0.15, stepping_stones=0.15, star_obstacles=0.15
        # Phase-1 warm-up should be dominated by walkable terrain.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.45),
        "rough_easy": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.0, 0.02), noise_step=0.01, border_width=0.25
        ),
        "rough_hard": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.01, 0.04), noise_step=0.01, border_width=0.25
        ),
        # Keep obstacle terrains rare at the beginning of curriculum training.
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.04,
            gap_width_range=(0.10, 0.35),
            platform_width=1.6,
        ),
        # Stepping and elevated blocks.
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.04,
            stone_height_max=0.10,
            stone_width_range=(0.30, 0.80),
            stone_distance_range=(0.05, 0.20),
            holes_depth=-2.0,
            platform_width=1.8,
        ),
        # Fan/star-like obstacles.
        "star_obstacles": terrain_gen.MeshStarTerrainCfg(
            proportion=0.02,
            num_bars=8,
            bar_width_range=(0.10, 0.28),
            bar_height_range=(0.04, 0.10),
            platform_width=1.6,
        ),
    },
)


@configclass
class FinalProjectRewards(H1Rewards):
    """Use stock H1 rough rewards unchanged."""

    pass

    # Previous custom reward mix kept for reference:
    # forward_velocity = RewTerm(func=custom_mdp.forward_velocity_toward_goal, weight=0.75, params={"goal_x": GOAL_X})
    # goal_progress = RewTerm(func=custom_mdp.goal_distance_progress, weight=0.25, params={"goal_x": GOAL_X})
    # goal_reached_bonus = RewTerm(
    #     func=custom_mdp.goal_reached_bonus,
    #     weight=25.0,
    #     params={"goal_x": GOAL_X, "bonus": 1.0},
    # )
    # obstacle_zone_crossing = RewTerm(
    #     func=custom_mdp.zone_crossing_bonus,
    #     weight=0.1,
    #     params={"zone_positions": (1.5, 3.0, 4.5, 6.0), "sigma": 0.35},
    # )
    # low_height_penalty = RewTerm(func=custom_mdp.base_height_penalty, weight=-2.0, params={"min_height": 0.55})
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.75,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
    #         "threshold": 0.5,
    #     },
    # )
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
    #     },
    # )


@configclass
class FinalProjectBaselineRewards(H1Rewards):
    """First-passage baseline aligned with minimizing time-to-goal."""

    termination_penalty = None
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    dof_pos_limits = None
    joint_deviation_hip = None
    joint_deviation_arms = None
    joint_deviation_torso = None
    ang_vel_xy_l2 = None
    lin_vel_z_l2 = None

    time_cost = RewTerm(func=custom_mdp.time_penalty, weight=-1.0)
    goal_progress = RewTerm(
        func=custom_mdp.gated_goal_progress_delta,
        weight=BASELINE_GOAL_PROGRESS_WEIGHT,
        params={
            "goal_x": GOAL_X,
            "start_x": MAP_START_POS[0],
            "normalize_by_goal": False,
            "min_height": 0.42,
            "safe_height": 0.70,
            "min_upright": 0.30,
            "safe_upright": 0.80,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    track_lin_vel_xy_exp = RewTerm(
        func=custom_mdp.gated_track_lin_vel_xy_command,
        weight=BASELINE_TRACK_LIN_VEL_WEIGHT,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "min_height": 0.42,
            "safe_height": 0.70,
            "min_upright": 0.25,
            "safe_upright": 0.75,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    forward_speed = None
    goal_reached_bonus = RewTerm(
        func=custom_mdp.goal_reached_bonus,
        weight=25.0,
        params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "bonus": 1.0},
    )
    stand_up_height = RewTerm(
        func=custom_mdp.stand_up_height_reward,
        weight=0.0,
        params={
            "min_height": 0.46,
            "target_height": 0.70,
            "min_upright": 0.3,
            "safe_upright": 0.75,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    stance_leg_extension = RewTerm(
        func=custom_mdp.stance_leg_extension_reward,
        weight=0.0,
        params={
            "target_hip_pitch": -0.28,
            "target_knee": 0.79,
            "target_ankle": -0.52,
            "posture_sigma": 0.20,
            "stance_contact_time": 0.03,
            "min_height": 0.44,
            "safe_height": 0.70,
            "min_upright": 0.25,
            "safe_upright": 0.75,
            "contact_force_threshold": 1.0,
            "hip_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch"),
            "knee_cfg": SceneEntityCfg("robot", joint_names=".*_knee"),
            "ankle_cfg": SceneEntityCfg("robot", joint_names=".*_ankle"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "torso_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    failure_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    upright_survival = RewTerm(
        func=custom_mdp.upright_survival_reward,
        weight=BASELINE_UPRIGHT_SURVIVAL_WEIGHT,
        params={
            "min_height": 0.40,
            "safe_height": 0.68,
            "min_upright": 0.20,
            "safe_upright": 0.70,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    low_height_penalty = RewTerm(func=custom_mdp.base_height_penalty, weight=-0.2, params={"min_height": 0.50})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    feet_air_time = RewTerm(
        func=custom_mdp.gated_single_swing_step_reward,
        weight=BASELINE_FEET_AIR_TIME_WEIGHT,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "min_air_time": BASELINE_FEET_AIR_TIME_MIN,
            "target_air_time": BASELINE_FEET_AIR_TIME_TARGET,
            "min_stance_time": BASELINE_FEET_STANCE_TIME_MIN,
            "min_forward_speed": BASELINE_STEP_FORWARD_SPEED_MIN,
            "min_height": 0.42,
            "safe_height": 0.70,
            "min_upright": 0.18,
            "safe_upright": 0.68,
            "contact_force_threshold": 1.0,
            "torso_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )


@configclass
class FinalProjectCurriculumRewards(FinalProjectBaselineRewards):
    """Stepping-first curriculum rewards on the final-map geometry."""

    goal_progress = RewTerm(
        func=custom_mdp.gated_goal_progress_delta,
        weight=CURRICULUM_GOAL_PROGRESS_WEIGHT,
        params={
            "goal_x": CURRICULUM_PROXY_GOAL_X,
            "start_x": MAP_START_POS[0],
            "normalize_by_goal": False,
            "min_height": 0.38,
            "safe_height": 0.66,
            "min_upright": 0.22,
            "safe_upright": 0.72,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    track_lin_vel_xy_exp = RewTerm(
        func=custom_mdp.gated_track_lin_vel_xy_command,
        weight=CURRICULUM_TRACK_LIN_VEL_WEIGHT,
        params={
            "command_name": "base_velocity",
            "std": 0.35,
            "min_height": 0.38,
            "safe_height": 0.66,
            "min_upright": 0.18,
            "safe_upright": 0.68,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    goal_reached_bonus = RewTerm(
        func=custom_mdp.goal_reached_bonus,
        weight=20.0,
        params={"goal_x": CURRICULUM_PROXY_GOAL_X, "start_x": MAP_START_POS[0], "bonus": 1.0},
    )
    stand_up_height = RewTerm(
        func=custom_mdp.stand_up_height_reward,
        weight=CURRICULUM_STAND_UP_HEIGHT_WEIGHT,
        params={
            "min_height": 0.42,
            "target_height": 0.68,
            "min_upright": 0.22,
            "safe_upright": 0.68,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    stance_leg_extension = RewTerm(
        func=custom_mdp.stance_leg_extension_reward,
        weight=3.0,
        params={
            "target_hip_pitch": -0.28,
            "target_knee": 0.79,
            "target_ankle": -0.52,
            "posture_sigma": 0.20,
            "stance_contact_time": 0.025,
            "min_height": 0.40,
            "safe_height": 0.68,
            "min_upright": 0.18,
            "safe_upright": 0.66,
            "contact_force_threshold": 1.0,
            "hip_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch"),
            "knee_cfg": SceneEntityCfg("robot", joint_names=".*_knee"),
            "ankle_cfg": SceneEntityCfg("robot", joint_names=".*_ankle"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "torso_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    upright_survival = RewTerm(
        func=custom_mdp.upright_survival_reward,
        weight=CURRICULUM_UPRIGHT_SURVIVAL_WEIGHT,
        params={
            "min_height": 0.38,
            "safe_height": 0.64,
            "min_upright": 0.18,
            "safe_upright": 0.66,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    low_height_penalty = RewTerm(func=custom_mdp.base_height_penalty, weight=-0.35, params={"min_height": 0.46})
    feet_air_time = RewTerm(
        func=custom_mdp.gated_early_step_reward,
        weight=CURRICULUM_FEET_AIR_TIME_WEIGHT,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "min_air_time": 0.002,
            "target_air_time": 0.024,
            "min_stance_time": 0.004,
            "min_forward_speed": 0.001,
            "min_height": 0.40,
            "safe_height": 0.68,
            "min_upright": 0.16,
            "safe_upright": 0.66,
            "contact_force_threshold": 1.0,
            "torso_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )


@configclass
class FinalProjectRoughGoalBaselineRewards(H1Rewards):
    """Stock H1 locomotion rewards plus goal terms for final-map traversal."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=ROUGH_GOAL_BASELINE_TERMINATION_PENALTY)
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=ROUGH_GOAL_BASELINE_TRACK_ANG_VEL_WEIGHT,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    time_cost = RewTerm(func=custom_mdp.time_penalty, weight=-2.0)
    goal_progress = RewTerm(
        func=custom_mdp.gated_goal_progress_delta,
        weight=3.0,
        params={
            "goal_x": GOAL_X,
            "start_x": MAP_START_POS[0],
            "normalize_by_goal": False,
            "min_height": 0.42,
            "safe_height": 0.70,
            "min_upright": 0.30,
            "safe_upright": 0.80,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    goal_reached_bonus = RewTerm(
        func=custom_mdp.goal_reached_bonus,
        weight=200.0,
        params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "bonus": 1.0},
    )


@configclass
class FinalProjectSpeedRunRewards(H1Rewards):
    """Rewards for unlimited-retry fastest-total-time goal reaching.

    Fall-forward friendly: goal_progress is ungated so any forward displacement earns
    reward regardless of posture. Time-remaining bonus directly rewards faster arrivals.
    """

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=SPEEDRUN_TERMINATION_PENALTY)
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.1,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    time_cost = RewTerm(func=custom_mdp.time_penalty, weight=-2.0)
    goal_progress = RewTerm(
        func=custom_mdp.goal_progress_delta,
        weight=5.0,
        params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "normalize_by_goal": False},
    )
    goal_reached_bonus = RewTerm(
        func=custom_mdp.time_remaining_goal_bonus,
        weight=1.0,
        params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "base_bonus": 500.0},
    )
    forward_speed = RewTerm(
        func=custom_mdp.forward_velocity_toward_goal,
        weight=0.3,
        params={"goal_x": GOAL_X},
    )


@configclass
class FinalProjectFastWalkRewards(H1Rewards):
    """Rewards for fastest upright goal reaching.

    Blocks fall-forward via speed-gated goal_progress: only earns when upright AND moving
    above min_forward_speed. Time-remaining bonus rewards faster arrivals.
    """

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-25.0)
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.25,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    time_cost = RewTerm(func=custom_mdp.time_penalty, weight=-2.0)
    goal_progress = RewTerm(
        func=custom_mdp.speed_gated_goal_progress_delta,
        weight=5.0,
        params={
            "goal_x": GOAL_X,
            "start_x": MAP_START_POS[0],
            "normalize_by_goal": False,
            "min_forward_speed": 0.4,
            "min_height": 0.42,
            "safe_height": 0.70,
            "min_upright": 0.30,
            "safe_upright": 0.80,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    goal_reached_bonus = RewTerm(
        func=custom_mdp.time_remaining_goal_bonus,
        weight=1.0,
        params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "base_bonus": 500.0},
    )


@configclass
class FinalProjectMinimalRewardRewards(H1Rewards):
    """Diagnostic final-map reward stack with only one stock tracking term plus goal terms."""

    track_ang_vel_z_exp = None
    feet_air_time = None
    feet_slide = None
    dof_pos_limits = None
    joint_deviation_hip = None
    joint_deviation_arms = None
    joint_deviation_torso = None

    goal_progress = RewTerm(
        func=custom_mdp.gated_goal_progress_delta,
        weight=MINIMAL_REWARD_GOAL_PROGRESS_WEIGHT,
        params={
            "goal_x": GOAL_X,
            "start_x": MAP_START_POS[0],
            "normalize_by_goal": False,
            "min_height": 0.42,
            "safe_height": 0.70,
            "min_upright": 0.30,
            "safe_upright": 0.80,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        },
    )
    goal_reached_bonus = RewTerm(
        func=custom_mdp.goal_reached_bonus,
        weight=MINIMAL_REWARD_GOAL_REACHED_BONUS_WEIGHT,
        params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "bonus": 1.0},
    )


@configclass
class FinalProjectCurriculum:
    """Map-aware curriculum uses explicit phase overrides instead of inherited terrain speed ramps."""

    terrain_levels = None
    command_speed = None
    push_strength = None


@configclass
class FinalProjectUnitreeH1EnvCfg(H1FlatEnvCfg):
    """Stage A: stepping-first curriculum on the shared final-map geometry."""

    rewards: FinalProjectCurriculumRewards = FinalProjectCurriculumRewards()
    curriculum: FinalProjectCurriculum = FinalProjectCurriculum()

    def _apply_curriculum_reward_overrides(self):
        self.rewards.feet_air_time.func = custom_mdp.gated_early_step_reward
        self.rewards.feet_air_time.weight = CURRICULUM_FEET_AIR_TIME_WEIGHT
        self.rewards.feet_air_time.params = {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "min_air_time": 0.002,
            "target_air_time": 0.024,
            "min_stance_time": 0.004,
            "min_forward_speed": 0.001,
            "min_height": 0.40,
            "safe_height": 0.68,
            "min_upright": 0.16,
            "safe_upright": 0.66,
            "contact_force_threshold": 1.0,
            "torso_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.goal_progress.weight = CURRICULUM_GOAL_PROGRESS_WEIGHT
        self.rewards.goal_progress.params = {
            "goal_x": CURRICULUM_PROXY_GOAL_X,
            "start_x": MAP_START_POS[0],
            "normalize_by_goal": False,
            "min_height": 0.38,
            "safe_height": 0.66,
            "min_upright": 0.22,
            "safe_upright": 0.72,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.track_lin_vel_xy_exp.weight = CURRICULUM_TRACK_LIN_VEL_WEIGHT
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity",
            "std": 0.35,
            "min_height": 0.38,
            "safe_height": 0.66,
            "min_upright": 0.18,
            "safe_upright": 0.68,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.goal_reached_bonus.weight = 20.0
        self.rewards.goal_reached_bonus.params = {
            "goal_x": CURRICULUM_PROXY_GOAL_X,
            "start_x": MAP_START_POS[0],
            "bonus": 1.0,
        }
        self.rewards.stand_up_height.weight = CURRICULUM_STAND_UP_HEIGHT_WEIGHT
        self.rewards.stand_up_height.params = {
            "min_height": 0.42,
            "target_height": 0.68,
            "min_upright": 0.22,
            "safe_upright": 0.68,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.stance_leg_extension.weight = 3.0
        self.rewards.stance_leg_extension.params = {
            "target_hip_pitch": -0.28,
            "target_knee": 0.79,
            "target_ankle": -0.52,
            "posture_sigma": 0.20,
            "stance_contact_time": 0.025,
            "min_height": 0.40,
            "safe_height": 0.68,
            "min_upright": 0.18,
            "safe_upright": 0.66,
            "contact_force_threshold": 1.0,
            "hip_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch"),
            "knee_cfg": SceneEntityCfg("robot", joint_names=".*_knee"),
            "ankle_cfg": SceneEntityCfg("robot", joint_names=".*_ankle"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "torso_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.upright_survival.weight = CURRICULUM_UPRIGHT_SURVIVAL_WEIGHT
        self.rewards.upright_survival.params = {
            "min_height": 0.38,
            "safe_height": 0.64,
            "min_upright": 0.18,
            "safe_upright": 0.66,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.low_height_penalty.weight = -0.35
        self.rewards.low_height_penalty.params = {"min_height": 0.46}
        self.rewards.forward_speed = None
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.001

    def _apply_curriculum_autopilot_overrides(self):
        overrides = _get_autopilot_override("curriculum")
        if not overrides:
            return

        if "num_envs" in overrides:
            self.scene.num_envs = int(overrides["num_envs"])
        if "action_scale" in overrides:
            self.actions.joint_pos.scale = float(overrides["action_scale"])
        if "goal_x" in overrides:
            goal_x = float(overrides["goal_x"])
            self.rewards.goal_progress.params["goal_x"] = goal_x
            self.rewards.goal_reached_bonus.params["goal_x"] = goal_x
            self.terminations.goal_reached.params["goal_x"] = goal_x
        if "lin_vel_x" in overrides:
            self.commands.base_velocity.ranges.lin_vel_x = _tuple2(
                overrides["lin_vel_x"], self.commands.base_velocity.ranges.lin_vel_x
            )
        if "track_lin_vel_xy_weight" in overrides:
            self.rewards.track_lin_vel_xy_exp.weight = float(overrides["track_lin_vel_xy_weight"])
        if "track_lin_vel_xy_std" in overrides:
            self.rewards.track_lin_vel_xy_exp.params["std"] = float(overrides["track_lin_vel_xy_std"])
        if "goal_progress_weight" in overrides:
            self.rewards.goal_progress.weight = float(overrides["goal_progress_weight"])
        if "goal_progress_min_height" in overrides:
            self.rewards.goal_progress.params["min_height"] = float(overrides["goal_progress_min_height"])
        if "goal_progress_safe_height" in overrides:
            self.rewards.goal_progress.params["safe_height"] = float(overrides["goal_progress_safe_height"])
        if "goal_progress_min_upright" in overrides:
            self.rewards.goal_progress.params["min_upright"] = float(overrides["goal_progress_min_upright"])
        if "goal_progress_safe_upright" in overrides:
            self.rewards.goal_progress.params["safe_upright"] = float(overrides["goal_progress_safe_upright"])
        if "goal_reached_bonus_weight" in overrides:
            self.rewards.goal_reached_bonus.weight = float(overrides["goal_reached_bonus_weight"])
        if "stand_up_height_weight" in overrides:
            self.rewards.stand_up_height.weight = float(overrides["stand_up_height_weight"])
        if "stand_up_min_height" in overrides:
            self.rewards.stand_up_height.params["min_height"] = float(overrides["stand_up_min_height"])
        if "stand_up_target_height" in overrides:
            self.rewards.stand_up_height.params["target_height"] = float(overrides["stand_up_target_height"])
        if "stand_up_min_upright" in overrides:
            self.rewards.stand_up_height.params["min_upright"] = float(overrides["stand_up_min_upright"])
        if "stand_up_safe_upright" in overrides:
            self.rewards.stand_up_height.params["safe_upright"] = float(overrides["stand_up_safe_upright"])
        if "upright_survival_weight" in overrides:
            self.rewards.upright_survival.weight = float(overrides["upright_survival_weight"])
        if "upright_survival_min_height" in overrides:
            self.rewards.upright_survival.params["min_height"] = float(overrides["upright_survival_min_height"])
        if "upright_survival_safe_height" in overrides:
            self.rewards.upright_survival.params["safe_height"] = float(overrides["upright_survival_safe_height"])
        if "upright_survival_min_upright" in overrides:
            self.rewards.upright_survival.params["min_upright"] = float(overrides["upright_survival_min_upright"])
        if "upright_survival_safe_upright" in overrides:
            self.rewards.upright_survival.params["safe_upright"] = float(overrides["upright_survival_safe_upright"])
        if "feet_air_time_weight" in overrides:
            self.rewards.feet_air_time.weight = float(overrides["feet_air_time_weight"])
        if "stance_leg_extension_weight" in overrides:
            self.rewards.stance_leg_extension.weight = float(overrides["stance_leg_extension_weight"])
        if "stance_leg_extension_sigma" in overrides:
            self.rewards.stance_leg_extension.params["posture_sigma"] = float(overrides["stance_leg_extension_sigma"])
        if "stance_leg_extension_contact_time" in overrides:
            self.rewards.stance_leg_extension.params["stance_contact_time"] = float(
                overrides["stance_leg_extension_contact_time"]
            )
        if "stance_leg_extension_min_height" in overrides:
            self.rewards.stance_leg_extension.params["min_height"] = float(overrides["stance_leg_extension_min_height"])
        if "stance_leg_extension_safe_height" in overrides:
            self.rewards.stance_leg_extension.params["safe_height"] = float(overrides["stance_leg_extension_safe_height"])
        if "stance_leg_extension_min_upright" in overrides:
            self.rewards.stance_leg_extension.params["min_upright"] = float(
                overrides["stance_leg_extension_min_upright"]
            )
        if "stance_leg_extension_safe_upright" in overrides:
            self.rewards.stance_leg_extension.params["safe_upright"] = float(
                overrides["stance_leg_extension_safe_upright"]
            )
        if "low_height_penalty_weight" in overrides:
            self.rewards.low_height_penalty.weight = float(overrides["low_height_penalty_weight"])
        if "low_height_penalty_minimum" in overrides:
            self.rewards.low_height_penalty.params["min_height"] = float(overrides["low_height_penalty_minimum"])
        if "step_reward" in overrides and isinstance(overrides["step_reward"], dict):
            step_reward = overrides["step_reward"]
            step_reward_mode = str(step_reward.get("mode", "early_step"))
            if step_reward_mode == "single_swing":
                self.rewards.feet_air_time.func = custom_mdp.gated_single_swing_step_reward
            else:
                self.rewards.feet_air_time.func = custom_mdp.gated_early_step_reward
            if "min_air_time" in step_reward:
                self.rewards.feet_air_time.params["min_air_time"] = float(step_reward["min_air_time"])
            if "target_air_time" in step_reward:
                self.rewards.feet_air_time.params["target_air_time"] = float(step_reward["target_air_time"])
            if "min_stance_time" in step_reward:
                self.rewards.feet_air_time.params["min_stance_time"] = float(step_reward["min_stance_time"])
            if "min_forward_speed" in step_reward:
                self.rewards.feet_air_time.params["min_forward_speed"] = float(step_reward["min_forward_speed"])
            if "min_height" in step_reward:
                self.rewards.feet_air_time.params["min_height"] = float(step_reward["min_height"])
            if "safe_height" in step_reward:
                self.rewards.feet_air_time.params["safe_height"] = float(step_reward["safe_height"])
            if "min_upright" in step_reward:
                self.rewards.feet_air_time.params["min_upright"] = float(step_reward["min_upright"])
            if "safe_upright" in step_reward:
                self.rewards.feet_air_time.params["safe_upright"] = float(step_reward["safe_upright"])
        if "reset_joints_position_range" in overrides:
            self.events.reset_robot_joints.params["position_range"] = _tuple2(
                overrides["reset_joints_position_range"],
                self.events.reset_robot_joints.params["position_range"],
            )
        if "low_height_termination_minimum" in overrides:
            minimum_height = float(overrides["low_height_termination_minimum"])
            self.terminations.low_height.params["minimum_height"] = minimum_height
        if "goal_reached_min_height" in overrides:
            self.terminations.goal_reached.params["min_height"] = float(overrides["goal_reached_min_height"])
        if "goal_reached_min_upright" in overrides:
            self.terminations.goal_reached.params["min_upright"] = float(overrides["goal_reached_min_upright"])
        if "goal_reached_contact_force_threshold" in overrides:
            self.terminations.goal_reached.params["contact_force_threshold"] = float(
                overrides["goal_reached_contact_force_threshold"]
            )

    def __post_init__(self):
        super().__post_init__()
        self._apply_curriculum_reward_overrides()

        self.scene.robot = UNITREE_H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.final_map = None
        self.scene.num_envs = 32
        self.scene.env_spacing = MAP_ENV_SPACING
        self.actions.joint_pos.scale = 0.28
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.04, 0.12)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        self.observations.policy.goal_distance = ObsTerm(
            func=custom_mdp.goal_distance_x,
            params={"goal_x": CURRICULUM_PROXY_GOAL_X, "start_x": MAP_START_POS[0], "normalize": True},
        )
        self.events.reset_robot_joints.params["position_range"] = (0.99, 1.01)
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached_upright,
            params={
                "goal_x": CURRICULUM_PROXY_GOAL_X,
                "start_x": MAP_START_POS[0],
                "min_height": 0.50,
                "min_upright": 0.72,
                "contact_force_threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
            },
        )
        self.terminations.low_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.42, "asset_cfg": SceneEntityCfg("robot")},
        )
        self._apply_curriculum_autopilot_overrides()
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class FinalProjectUnitreeH1EnvCfg_PLAY(FinalProjectUnitreeH1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class FinalProjectUnitreeH1MapEnvCfg(FinalProjectUnitreeH1EnvCfg):
    """Stage B: final-map fine-tuning for competition optimization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.22)
        self.rewards.goal_progress.params["goal_x"] = GOAL_X
        self.rewards.goal_reached_bonus.params["goal_x"] = GOAL_X
        self.terminations.goal_reached.params["goal_x"] = GOAL_X
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class FinalProjectUnitreeH1MapEnvCfg_PLAY(FinalProjectUnitreeH1MapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class FinalProjectUnitreeH1BaselineEnvCfg(H1FlatEnvCfg):
    """Baseline: first-passage training on the final map from scratch."""

    rewards: FinalProjectBaselineRewards = FinalProjectBaselineRewards()

    def _apply_baseline_reward_overrides(self):
        """Re-apply baseline reward values after inherited H1 post-init mutations."""
        self.rewards.feet_air_time.func = custom_mdp.gated_single_swing_step_reward
        self.rewards.feet_air_time.weight = BASELINE_FEET_AIR_TIME_WEIGHT
        self.rewards.feet_air_time.params = {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "min_air_time": BASELINE_FEET_AIR_TIME_MIN,
            "target_air_time": BASELINE_FEET_AIR_TIME_TARGET,
            "min_stance_time": BASELINE_FEET_STANCE_TIME_MIN,
            "min_forward_speed": BASELINE_STEP_FORWARD_SPEED_MIN,
            "min_height": 0.45,
            "safe_height": 0.72,
            "min_upright": 0.25,
            "safe_upright": 0.75,
            "contact_force_threshold": 1.0,
            "torso_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.goal_progress.weight = BASELINE_GOAL_PROGRESS_WEIGHT
        self.rewards.goal_progress.params["normalize_by_goal"] = False
        self.rewards.track_lin_vel_xy_exp.weight = BASELINE_TRACK_LIN_VEL_WEIGHT
        self.rewards.goal_reached_bonus.weight = 40.0
        self.rewards.stand_up_height.weight = 0.0
        self.rewards.stand_up_height.params = {
            "min_height": 0.46,
            "target_height": 0.70,
            "min_upright": 0.3,
            "safe_upright": 0.75,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.stance_leg_extension.weight = 0.0
        self.rewards.stance_leg_extension.params = {
            "target_hip_pitch": -0.28,
            "target_knee": 0.79,
            "target_ankle": -0.52,
            "posture_sigma": 0.20,
            "stance_contact_time": 0.03,
            "min_height": 0.44,
            "safe_height": 0.70,
            "min_upright": 0.25,
            "safe_upright": 0.75,
            "contact_force_threshold": 1.0,
            "hip_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch"),
            "knee_cfg": SceneEntityCfg("robot", joint_names=".*_knee"),
            "ankle_cfg": SceneEntityCfg("robot", joint_names=".*_ankle"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "torso_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.upright_survival.weight = 0.75
        self.rewards.upright_survival.params = {
            "min_height": 0.42,
            "safe_height": 0.70,
            "min_upright": 0.25,
            "safe_upright": 0.75,
            "contact_force_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
        }
        self.rewards.forward_speed = None
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.001

    def _validate_baseline_reward_overrides(self):
        """Fail fast if inherited configs silently override the intended baseline reward setup."""
        if self.rewards.feet_air_time.func is not custom_mdp.gated_single_swing_step_reward:
            raise ValueError("Baseline feet_air_time reward func was overridden unexpectedly.")
        if self.rewards.feet_air_time.weight != BASELINE_FEET_AIR_TIME_WEIGHT:
            raise ValueError(
                f"Baseline feet_air_time weight={self.rewards.feet_air_time.weight} "
                f"expected {BASELINE_FEET_AIR_TIME_WEIGHT}."
            )
        if self.rewards.feet_air_time.params.get("min_air_time") != BASELINE_FEET_AIR_TIME_MIN:
            raise ValueError(
                f"Baseline feet_air_time min_air_time={self.rewards.feet_air_time.params.get('min_air_time')} "
                f"expected {BASELINE_FEET_AIR_TIME_MIN}."
            )
        if self.rewards.goal_progress.weight != BASELINE_GOAL_PROGRESS_WEIGHT:
            raise ValueError(
                f"Baseline goal_progress weight={self.rewards.goal_progress.weight} "
                f"expected {BASELINE_GOAL_PROGRESS_WEIGHT}."
            )
        if self.rewards.goal_progress.params.get("normalize_by_goal") is not False:
            raise ValueError("Baseline goal_progress must use raw per-step x-delta.")
        if self.rewards.goal_reached_bonus.weight != 40.0:
            raise ValueError(
                f"Baseline goal_reached_bonus weight={self.rewards.goal_reached_bonus.weight} expected 40.0."
            )
        if self.rewards.upright_survival.weight != 0.75:
            raise ValueError(
                f"Baseline upright_survival weight={self.rewards.upright_survival.weight} expected 0.75."
            )
        if self.rewards.upright_survival.params.get("min_height") != 0.42:
            raise ValueError(
                f"Baseline upright_survival min_height={self.rewards.upright_survival.params.get('min_height')} "
                "expected 0.42."
            )
        if self.rewards.forward_speed is not None:
            raise ValueError("Baseline forward_speed reward must stay disabled.")

    def _apply_baseline_autopilot_overrides(self):
        """Allow the autopilot loop to tune a narrow set of baseline knobs."""

        overrides = _get_autopilot_override("baseline")
        if not overrides:
            return

        if "num_envs" in overrides:
            self.scene.num_envs = int(overrides["num_envs"])
        if "action_scale" in overrides:
            self.actions.joint_pos.scale = float(overrides["action_scale"])
        if "lin_vel_x" in overrides:
            self.commands.base_velocity.ranges.lin_vel_x = _tuple2(overrides["lin_vel_x"], self.commands.base_velocity.ranges.lin_vel_x)
        if "track_lin_vel_xy_weight" in overrides:
            self.rewards.track_lin_vel_xy_exp.weight = float(overrides["track_lin_vel_xy_weight"])
        if "track_lin_vel_xy_std" in overrides:
            self.rewards.track_lin_vel_xy_exp.params["std"] = float(overrides["track_lin_vel_xy_std"])
        if "goal_progress_weight" in overrides:
            self.rewards.goal_progress.weight = float(overrides["goal_progress_weight"])
        if "goal_progress_min_height" in overrides:
            self.rewards.goal_progress.params["min_height"] = float(overrides["goal_progress_min_height"])
        if "goal_progress_safe_height" in overrides:
            self.rewards.goal_progress.params["safe_height"] = float(overrides["goal_progress_safe_height"])
        if "goal_progress_min_upright" in overrides:
            self.rewards.goal_progress.params["min_upright"] = float(overrides["goal_progress_min_upright"])
        if "goal_progress_safe_upright" in overrides:
            self.rewards.goal_progress.params["safe_upright"] = float(overrides["goal_progress_safe_upright"])
        if "goal_progress_contact_force_threshold" in overrides:
            self.rewards.goal_progress.params["contact_force_threshold"] = float(
                overrides["goal_progress_contact_force_threshold"]
            )
        if "goal_reached_bonus_weight" in overrides:
            self.rewards.goal_reached_bonus.weight = float(overrides["goal_reached_bonus_weight"])
        if "stand_up_height_weight" in overrides:
            self.rewards.stand_up_height.weight = float(overrides["stand_up_height_weight"])
        if "stand_up_min_height" in overrides:
            self.rewards.stand_up_height.params["min_height"] = float(overrides["stand_up_min_height"])
        if "stand_up_target_height" in overrides:
            self.rewards.stand_up_height.params["target_height"] = float(overrides["stand_up_target_height"])
        if "stand_up_min_upright" in overrides:
            self.rewards.stand_up_height.params["min_upright"] = float(overrides["stand_up_min_upright"])
        if "stand_up_safe_upright" in overrides:
            self.rewards.stand_up_height.params["safe_upright"] = float(overrides["stand_up_safe_upright"])
        if "stand_up_contact_force_threshold" in overrides:
            self.rewards.stand_up_height.params["contact_force_threshold"] = float(
                overrides["stand_up_contact_force_threshold"]
            )
        if "upright_survival_weight" in overrides:
            self.rewards.upright_survival.weight = float(overrides["upright_survival_weight"])
        if "upright_survival_min_height" in overrides:
            self.rewards.upright_survival.params["min_height"] = float(overrides["upright_survival_min_height"])
        if "upright_survival_safe_height" in overrides:
            self.rewards.upright_survival.params["safe_height"] = float(overrides["upright_survival_safe_height"])
        if "upright_survival_min_upright" in overrides:
            self.rewards.upright_survival.params["min_upright"] = float(overrides["upright_survival_min_upright"])
        if "upright_survival_safe_upright" in overrides:
            self.rewards.upright_survival.params["safe_upright"] = float(overrides["upright_survival_safe_upright"])
        if "upright_survival_contact_force_threshold" in overrides:
            self.rewards.upright_survival.params["contact_force_threshold"] = float(
                overrides["upright_survival_contact_force_threshold"]
            )
        if "feet_air_time_weight" in overrides:
            self.rewards.feet_air_time.weight = float(overrides["feet_air_time_weight"])
        if "stance_leg_extension_weight" in overrides:
            self.rewards.stance_leg_extension.weight = float(overrides["stance_leg_extension_weight"])
        if "stance_leg_extension_sigma" in overrides:
            self.rewards.stance_leg_extension.params["posture_sigma"] = float(overrides["stance_leg_extension_sigma"])
        if "stance_leg_extension_contact_time" in overrides:
            self.rewards.stance_leg_extension.params["stance_contact_time"] = float(
                overrides["stance_leg_extension_contact_time"]
            )
        if "stance_leg_extension_min_height" in overrides:
            self.rewards.stance_leg_extension.params["min_height"] = float(overrides["stance_leg_extension_min_height"])
        if "stance_leg_extension_safe_height" in overrides:
            self.rewards.stance_leg_extension.params["safe_height"] = float(overrides["stance_leg_extension_safe_height"])
        if "stance_leg_extension_min_upright" in overrides:
            self.rewards.stance_leg_extension.params["min_upright"] = float(
                overrides["stance_leg_extension_min_upright"]
            )
        if "stance_leg_extension_safe_upright" in overrides:
            self.rewards.stance_leg_extension.params["safe_upright"] = float(
                overrides["stance_leg_extension_safe_upright"]
            )
        if "low_height_penalty_weight" in overrides:
            self.rewards.low_height_penalty.weight = float(overrides["low_height_penalty_weight"])
        if "low_height_penalty_minimum" in overrides:
            self.rewards.low_height_penalty.params["min_height"] = float(overrides["low_height_penalty_minimum"])
        if "step_reward" in overrides and isinstance(overrides["step_reward"], dict):
            step_reward = overrides["step_reward"]
            step_reward_mode = str(step_reward.get("mode", "single_swing"))
            if step_reward_mode == "early_step":
                self.rewards.feet_air_time.func = custom_mdp.gated_early_step_reward
            else:
                self.rewards.feet_air_time.func = custom_mdp.gated_single_swing_step_reward
            if "min_air_time" in step_reward:
                self.rewards.feet_air_time.params["min_air_time"] = float(step_reward["min_air_time"])
            if "target_air_time" in step_reward:
                self.rewards.feet_air_time.params["target_air_time"] = float(step_reward["target_air_time"])
            if "min_stance_time" in step_reward:
                self.rewards.feet_air_time.params["min_stance_time"] = float(step_reward["min_stance_time"])
            if "min_forward_speed" in step_reward:
                self.rewards.feet_air_time.params["min_forward_speed"] = float(step_reward["min_forward_speed"])
        if "reset_joints_position_range" in overrides:
            self.events.reset_robot_joints.params["position_range"] = _tuple2(
                overrides["reset_joints_position_range"],
                self.events.reset_robot_joints.params["position_range"],
            )
        if "low_height_termination_minimum" in overrides:
            minimum_height = float(overrides["low_height_termination_minimum"])
            if getattr(self.terminations, "low_height", None) is None:
                self.terminations.low_height = DoneTerm(
                    func=mdp.root_height_below_minimum,
                    params={"minimum_height": minimum_height, "asset_cfg": SceneEntityCfg("robot")},
                )
            else:
                self.terminations.low_height.params["minimum_height"] = minimum_height
        if bool(overrides.get("use_goal_reached_upright")):
            self.terminations.goal_reached = DoneTerm(
                func=custom_mdp.goal_reached_upright,
                params={
                    "goal_x": GOAL_X,
                    "start_x": MAP_START_POS[0],
                    "min_height": float(overrides.get("goal_reached_min_height", 0.55)),
                    "min_upright": float(overrides.get("goal_reached_min_upright", 0.75)),
                    "contact_force_threshold": float(overrides.get("goal_reached_contact_force_threshold", 1.0)),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
                },
            )

    def __post_init__(self):
        super().__post_init__()

        self._apply_baseline_reward_overrides()
        self._validate_baseline_reward_overrides()

        self.scene.robot = UNITREE_H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.final_map = None
        self.scene.num_envs = 64
        self.scene.env_spacing = MAP_ENV_SPACING
        self.actions.joint_pos.scale = 0.4
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        # Keep each episode on one forward command so completion pressure lines
        # up with the first-passage objective instead of switching targets mid-run.
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        self.observations.policy.goal_distance = ObsTerm(
            func=custom_mdp.goal_distance_x,
            params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "normalize": True},
        )
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached,
            params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0]},
        )
        if self.commands.base_velocity.resampling_time_range != (20.0, 20.0):
            raise ValueError(
                "Baseline base_velocity resampling_time_range was overridden unexpectedly. "
                f"Got {self.commands.base_velocity.resampling_time_range}."
            )
        if self.commands.base_velocity.ranges.lin_vel_x != (0.35, 0.6):
            raise ValueError(
                f"Baseline lin_vel_x range={self.commands.base_velocity.ranges.lin_vel_x} expected (0.35, 0.6)."
            )
        self._apply_baseline_autopilot_overrides()
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class FinalProjectUnitreeH1BaselineEnvCfg_PLAY(FinalProjectUnitreeH1BaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class FinalProjectUnitreeH1RoughGoalBaselineEnvCfg(H1RoughEnvCfg):
    """Sibling baseline: stock H1 rough rewards plus explicit goal terms on the final map."""

    rewards: FinalProjectRoughGoalBaselineRewards = FinalProjectRoughGoalBaselineRewards()

    def _validate_rough_goal_baseline_cfg(self):
        """Fail fast if the intended rough-goal sibling config gets overridden."""
        if self.scene.terrain.terrain_type != "usd":
            raise ValueError(f"Rough-goal baseline terrain_type={self.scene.terrain.terrain_type} expected 'usd'.")
        if self.commands.base_velocity.resampling_time_range != (20.0, 20.0):
            raise ValueError(
                "Rough-goal baseline base_velocity resampling_time_range was overridden unexpectedly. "
                f"Got {self.commands.base_velocity.resampling_time_range}."
            )
        if self.rewards.track_ang_vel_z_exp.weight != ROUGH_GOAL_BASELINE_TRACK_ANG_VEL_WEIGHT:
            raise ValueError(
                "Rough-goal baseline track_ang_vel_z_exp weight="
                f"{self.rewards.track_ang_vel_z_exp.weight} expected {ROUGH_GOAL_BASELINE_TRACK_ANG_VEL_WEIGHT}."
            )
        if self.rewards.termination_penalty.weight != ROUGH_GOAL_BASELINE_TERMINATION_PENALTY:
            raise ValueError(
                "Rough-goal baseline termination_penalty weight="
                f"{self.rewards.termination_penalty.weight} expected {ROUGH_GOAL_BASELINE_TERMINATION_PENALTY}."
            )
        if self.terminations.goal_reached.func is not custom_mdp.goal_reached_upright:
            raise ValueError("Rough-goal baseline goal_reached must use goal_reached_upright.")

    def __post_init__(self):
        super().__post_init__()

        self.rewards.flat_orientation_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.dof_acc_l2 = None
        self.scene.robot = UNITREE_H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.final_map = None
        self.curriculum.terrain_levels = None
        self.scene.num_envs = 64
        self.scene.env_spacing = MAP_ENV_SPACING
        self.actions.joint_pos.scale = 0.4
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        self.observations.policy.goal_distance = ObsTerm(
            func=custom_mdp.goal_distance_x,
            params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "normalize": True},
        )
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.episode_length_s = 20.0
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached_upright,
            params={
                "goal_x": GOAL_X,
                "start_x": MAP_START_POS[0],
                "min_height": 0.55,
                "min_upright": 0.72,
                "contact_force_threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
            },
        )
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self._validate_rough_goal_baseline_cfg()


@configclass
class FinalProjectUnitreeH1RoughGoalBaselineEnvCfg_PLAY(FinalProjectUnitreeH1RoughGoalBaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self._validate_rough_goal_baseline_cfg()


@configclass
class FinalProjectUnitreeH1SpeedRunEnvCfg(FinalProjectUnitreeH1RoughGoalBaselineEnvCfg):
    """Speed-run: optimized for unlimited-retry fastest expected total time to goal.

    Fall-forward is intentionally permitted — goal_progress is ungated and the episode
    terminates on any goal crossing (not just upright). Time-remaining bonus rewards
    faster arrivals directly.
    """

    rewards: FinalProjectSpeedRunRewards = FinalProjectSpeedRunRewards()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = SPEEDRUN_LIN_VEL_X_RANGE
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached,
            params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0]},
        )


@configclass
class FinalProjectUnitreeH1SpeedRunEnvCfg_PLAY(FinalProjectUnitreeH1SpeedRunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class FinalProjectUnitreeH1FastWalkEnvCfg(FinalProjectUnitreeH1RoughGoalBaselineEnvCfg):
    """Fast-walk: fastest upright goal reaching; fall-forward blocked by speed-gated progress.

    Keeps goal_reached_upright termination. Speed-gated goal_progress requires the robot
    to be upright AND above min_forward_speed to earn shaping reward, pushing toward
    genuinely fast bipedal locomotion.
    """

    rewards: FinalProjectFastWalkRewards = FinalProjectFastWalkRewards()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = FASTWALK_LIN_VEL_X_RANGE


@configclass
class FinalProjectUnitreeH1FastWalkEnvCfg_PLAY(FinalProjectUnitreeH1FastWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }


@configclass
class FinalProjectUnitreeH1MinimalRewardEnvCfg(H1RoughEnvCfg):
    """Diagnostic sibling: keep the final-map task wiring and reduce rewards to the minimum set."""

    rewards: FinalProjectMinimalRewardRewards = FinalProjectMinimalRewardRewards()

    def _apply_minimal_reward_autopilot_overrides(self):
        overrides = _get_autopilot_override("minimal_reward")
        if not overrides:
            return

        if "num_envs" in overrides:
            self.scene.num_envs = int(overrides["num_envs"])
        if "lin_vel_x" in overrides:
            self.commands.base_velocity.ranges.lin_vel_x = _tuple2(overrides["lin_vel_x"], self.commands.base_velocity.ranges.lin_vel_x)
        if "goal_progress_weight" in overrides:
            self.rewards.goal_progress.weight = float(overrides["goal_progress_weight"])
        if "goal_reached_bonus_weight" in overrides:
            self.rewards.goal_reached_bonus.weight = float(overrides["goal_reached_bonus_weight"])
        if "low_height_termination_minimum" in overrides:
            self.terminations.low_height.params["minimum_height"] = float(overrides["low_height_termination_minimum"])
        if "bad_orientation_limit" in overrides:
            self.terminations.bad_orientation.params["limit_angle"] = float(overrides["bad_orientation_limit"])

    def _validate_minimal_reward_cfg(self):
        """Fail fast if the intended minimal-reward config gets overridden."""
        if self.scene.terrain.terrain_type != "usd":
            raise ValueError(f"Minimal-reward terrain_type={self.scene.terrain.terrain_type} expected 'usd'.")
        if self.commands.base_velocity.resampling_time_range != (20.0, 20.0):
            raise ValueError(
                "Minimal-reward base_velocity resampling_time_range was overridden unexpectedly. "
                f"Got {self.commands.base_velocity.resampling_time_range}."
            )
        if self.commands.base_velocity.ranges.lin_vel_x != MINIMAL_REWARD_LIN_VEL_X_RANGE:
            raise ValueError(
                f"Minimal-reward lin_vel_x range={self.commands.base_velocity.ranges.lin_vel_x} "
                f"expected {MINIMAL_REWARD_LIN_VEL_X_RANGE}."
            )
        if self.rewards.termination_penalty.weight != -200.0:
            raise ValueError(
                f"Minimal-reward termination_penalty weight={self.rewards.termination_penalty.weight} expected -200.0."
            )
        if self.rewards.track_lin_vel_xy_exp.weight != 1.0:
            raise ValueError(
                f"Minimal-reward track_lin_vel_xy_exp weight={self.rewards.track_lin_vel_xy_exp.weight} expected 1.0."
            )
        for reward_name in (
            "ang_vel_xy_l2",
            "track_ang_vel_z_exp",
            "feet_air_time",
            "feet_slide",
            "flat_orientation_l2",
            "dof_pos_limits",
            "joint_deviation_hip",
            "joint_deviation_arms",
            "joint_deviation_torso",
            "action_rate_l2",
            "dof_acc_l2",
            "dof_torques_l2",
        ):
            if getattr(self.rewards, reward_name) is not None:
                raise ValueError(f"Minimal-reward {reward_name} must stay disabled.")
        if self.rewards.goal_progress.weight != MINIMAL_REWARD_GOAL_PROGRESS_WEIGHT:
            raise ValueError(
                f"Minimal-reward goal_progress weight={self.rewards.goal_progress.weight} "
                f"expected {MINIMAL_REWARD_GOAL_PROGRESS_WEIGHT}."
            )
        if self.rewards.goal_reached_bonus.weight != MINIMAL_REWARD_GOAL_REACHED_BONUS_WEIGHT:
            raise ValueError(
                f"Minimal-reward goal_reached_bonus weight={self.rewards.goal_reached_bonus.weight} "
                f"expected {MINIMAL_REWARD_GOAL_REACHED_BONUS_WEIGHT}."
            )
        if self.terminations.goal_reached.func is not custom_mdp.goal_reached_upright:
            raise ValueError("Minimal-reward goal_reached must use custom_mdp.goal_reached_upright.")
        if getattr(self.terminations, "low_height", None) is None:
            raise ValueError("Minimal-reward low_height termination must stay enabled.")
        if getattr(self.terminations, "bad_orientation", None) is None:
            raise ValueError("Minimal-reward bad_orientation termination must stay enabled.")
        if self.terminations.bad_orientation.func is not custom_mdp.bad_orientation_safe:
            raise ValueError("Minimal-reward bad_orientation must use custom_mdp.bad_orientation_safe.")
        if self.terminations.low_height.params["minimum_height"] != 0.45:
            raise ValueError("Minimal-reward low_height minimum_height must stay at 0.45.")
        if self.terminations.bad_orientation.params["limit_angle"] != 1.15:
            raise ValueError("Minimal-reward bad_orientation limit_angle must stay at 1.15.")

    def __post_init__(self):
        super().__post_init__()

        self.rewards.ang_vel_xy_l2 = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.dof_acc_l2 = None
        self.scene.robot = UNITREE_H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.final_map = None
        self.curriculum.terrain_levels = None
        self.scene.num_envs = 64
        self.scene.env_spacing = MAP_ENV_SPACING
        self.actions.joint_pos.scale = 0.4
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.ranges.lin_vel_x = MINIMAL_REWARD_LIN_VEL_X_RANGE
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        self.observations.policy.goal_distance = ObsTerm(
            func=custom_mdp.goal_distance_x,
            params={"goal_x": GOAL_X, "start_x": MAP_START_POS[0], "normalize": True},
        )
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.episode_length_s = 20.0
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached_upright,
            params={
                "goal_x": GOAL_X,
                "start_x": MAP_START_POS[0],
                "min_height": 0.55,
                "min_upright": 0.75,
                "contact_force_threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso_link"),
            },
        )
        self.terminations.low_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=custom_mdp.bad_orientation_safe,
            params={"limit_angle": 1.15, "asset_cfg": SceneEntityCfg("robot")},
        )
        self._apply_minimal_reward_autopilot_overrides()
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self._validate_minimal_reward_cfg()


@configclass
class FinalProjectUnitreeH1MinimalRewardEnvCfg_PLAY(FinalProjectUnitreeH1MinimalRewardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_final_map_terrain_cfg(self.scene.num_envs, self.scene.env_spacing)
        self.scene.robot.init_state.pos = MAP_START_POS
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": MAP_START_POS,
            "base_rot": MAP_START_ROT,
            "xy_range": MAP_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self._validate_minimal_reward_cfg()


@configclass
class FinalProjectUnitreeH1StabilityArenaEnvCfg(FinalProjectUnitreeH1EnvCfg):
    """Stage A1: warm-up on curated easy terrain arena."""

    def _apply_stability_warmup_autopilot_overrides(self):
        overrides = _get_autopilot_override("stability_warmup")
        if not overrides:
            return

        if "num_envs" in overrides:
            self.scene.num_envs = int(overrides["num_envs"])
        if "lin_vel_x" in overrides:
            self.commands.base_velocity.ranges.lin_vel_x = _tuple2(overrides["lin_vel_x"], self.commands.base_velocity.ranges.lin_vel_x)
        if "lin_vel_y" in overrides:
            self.commands.base_velocity.ranges.lin_vel_y = _tuple2(overrides["lin_vel_y"], self.commands.base_velocity.ranges.lin_vel_y)
        if "ang_vel_z" in overrides:
            self.commands.base_velocity.ranges.ang_vel_z = _tuple2(overrides["ang_vel_z"], self.commands.base_velocity.ranges.ang_vel_z)
        if "resampling_time_range" in overrides:
            self.commands.base_velocity.resampling_time_range = _tuple2(
                overrides["resampling_time_range"], self.commands.base_velocity.resampling_time_range
            )
        if "rel_standing_envs" in overrides:
            self.commands.base_velocity.rel_standing_envs = float(overrides["rel_standing_envs"])
        if "rel_heading_envs" in overrides:
            self.commands.base_velocity.rel_heading_envs = float(overrides["rel_heading_envs"])

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None
        self.curriculum.command_speed = None
        self.curriculum.push_strength = None
        self.scene.num_envs = 64
        self.scene.env_spacing = 0
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self._apply_stability_warmup_autopilot_overrides()
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        layout = _load_arena_layout(CURRICULUM_ARENA_STABILITY_LAYOUT_PATH)
        start_pos = _apply_spawn_z_clearance(
            (
                float(layout["suggested_start"]["x"]),
                float(layout["suggested_start"]["y"]),
                float(layout["suggested_start"]["z"]),
            )
        )
        overrides = _get_autopilot_override("stability_warmup")
        goal_x = float(overrides.get("goal_x", STABILITY_WARMUP_PROXY_GOAL_X))
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_usd_terrain_cfg(CURRICULUM_ARENA_STABILITY_USD_PATH)
        self.scene.robot.init_state.pos = start_pos
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": start_pos,
            "base_rot": MAP_START_ROT,
            "xy_range": CURRICULUM_ARENA_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached,
            params={"goal_x": goal_x, "start_x": start_pos[0]},
        )


@configclass
class FinalProjectUnitreeH1StabilityArenaEnvCfg_PLAY(FinalProjectUnitreeH1StabilityArenaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        layout = _load_arena_layout(CURRICULUM_ARENA_STABILITY_LAYOUT_PATH)
        start_pos = _apply_spawn_z_clearance(
            (
                float(layout["suggested_start"]["x"]),
                float(layout["suggested_start"]["y"]),
                float(layout["suggested_start"]["z"]),
            )
        )
        overrides = _get_autopilot_override("stability_warmup")
        goal_x = float(overrides.get("goal_x", STABILITY_WARMUP_PROXY_GOAL_X))
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_usd_terrain_cfg(CURRICULUM_ARENA_STABILITY_USD_PATH)
        self.scene.robot.init_state.pos = start_pos
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": start_pos,
            "base_rot": MAP_START_ROT,
            "xy_range": CURRICULUM_ARENA_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached,
            params={"goal_x": goal_x, "start_x": start_pos[0]},
        )


@configclass
class FinalProjectUnitreeH1CrossingArenaEnvCfg(FinalProjectUnitreeH1EnvCfg):
    """Stage A2: mixed curated terrain arena for crossing transfer."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None
        self.scene.num_envs = 64
        self.scene.env_spacing = 0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        layout = _load_arena_layout(CURRICULUM_ARENA_CROSSING_LAYOUT_PATH)
        start_pos = _apply_spawn_z_clearance(
            (
                float(layout["suggested_start"]["x"]),
                float(layout["suggested_start"]["y"]),
                float(layout["suggested_start"]["z"]),
            )
        )
        goal_x = float(layout["suggested_goal_x"])
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_training(self)
        self.scene.terrain = _make_usd_terrain_cfg(CURRICULUM_ARENA_CROSSING_USD_PATH)
        self.scene.robot.init_state.pos = start_pos
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": start_pos,
            "base_rot": MAP_START_ROT,
            "xy_range": CURRICULUM_ARENA_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached,
            params={"goal_x": goal_x},
        )


@configclass
class FinalProjectUnitreeH1CrossingArenaEnvCfg_PLAY(FinalProjectUnitreeH1CrossingArenaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.finalize_after_overrides()

    def finalize_after_overrides(self):
        layout = _load_arena_layout(CURRICULUM_ARENA_CROSSING_LAYOUT_PATH)
        start_pos = _apply_spawn_z_clearance(
            (
                float(layout["suggested_start"]["x"]),
                float(layout["suggested_start"]["y"]),
                float(layout["suggested_start"]["z"]),
            )
        )
        goal_x = float(layout["suggested_goal_x"])
        _validate_shared_map_env_cfg(self.scene.num_envs, self.scene.env_spacing)
        _tune_shared_arena_physx_buffers_for_play(self)
        self.scene.terrain = _make_usd_terrain_cfg(CURRICULUM_ARENA_CROSSING_USD_PATH)
        self.scene.robot.init_state.pos = start_pos
        self.scene.robot.init_state.rot = MAP_START_ROT
        self.events.reset_base.func = custom_mdp.reset_root_state_on_shared_map
        self.events.reset_base.params = {
            "base_pos": start_pos,
            "base_rot": MAP_START_ROT,
            "xy_range": CURRICULUM_ARENA_START_POS_JITTER_XY,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self.terminations.goal_reached = DoneTerm(
            func=custom_mdp.goal_reached,
            params={"goal_x": goal_x},
        )


@configclass
class FinalProjectH1RoughWalkerEnvCfg(H1RoughEnvCfg):
    """Stage-1B warm-start: H1 rough locomotion with obs that exactly match the RoughGoal Stage-2 env.

    Trains on procedural rough terrain (not the USD map) so the robot generalises across varied
    surfaces. The observation space — including height_scan, velocity_commands and goal_distance —
    is intentionally identical to FinalProjectUnitreeH1RoughGoalBaselineEnvCfg so that every
    parameter layer transfers cleanly when Stage-2 resumes from this checkpoint.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Override height scanner to match Stage-2 geometry exactly.
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # Mirror Stage-2 observation overrides so obs dims match exactly.
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        self.observations.policy.velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # goal_distance: use the same 13.3 m range; start_x=None resolves per env origin.
        # This 1-dim obs will be near 1.0 at episode start and shrinks as the robot walks —
        # even on procedural terrain it teaches the policy to read forward progress.
        self.observations.policy.goal_distance = ObsTerm(
            func=custom_mdp.goal_distance_x,
            params={"goal_x": GOAL_X, "start_x": None, "normalize": True},
        )

        # Match Stage-2 action and command settings.
        self.actions.joint_pos.scale = 0.4
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = None

        self.scene.num_envs = 4096


@configclass
class FinalProjectH1RoughWalkerEnvCfg_PLAY(FinalProjectH1RoughWalkerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


def _validate_shared_map_env_cfg(num_envs: int, env_spacing: float) -> None:
    """Reject unsupported shared-map layouts before training starts."""

    if env_spacing != 0:
        raise ValueError(
            "Final-map tasks use one shared USD arena, so scene.env_spacing must stay at 0. "
            "The current setup does not clone the map per environment."
        )

    if num_envs < 1:
        raise ValueError("scene.num_envs must be at least 1.")
