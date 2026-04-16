import json
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
class FinalProjectCurriculum:
    """Difficulty progression for command speed and disturbances."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    command_speed = CurrTerm(
        func=custom_mdp.increase_command_range,
        # Old aggressive setting for reference:
        # params={"interval_steps": 5000, "max_lin_vel_x": 2.0, "step_size": 0.15},
        params={"interval_steps": 10000, "max_lin_vel_x": 1.2, "step_size": 0.1},
    )
    push_strength = CurrTerm(
        func=custom_mdp.increase_push_disturbance,
        params={"interval_steps": 8000, "max_push_vel": 0.8, "step_size": 0.05},
    )


@configclass
class FinalProjectUnitreeH1EnvCfg(H1RoughEnvCfg):
    """Stage A: curriculum training on procedural terrains."""

    rewards: FinalProjectRewards = FinalProjectRewards()
    curriculum: FinalProjectCurriculum = FinalProjectCurriculum()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=FINAL_PROJECT_CURRICULUM_TERRAINS_CFG,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        # Old aggressive settings for reference:
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        # Start with straight-ahead walking and grow speed only after stable gait appears.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.heading_command = False

        self.episode_length_s = 20.0
        self.terminations.goal_reached = DoneTerm(func=custom_mdp.goal_reached, params={"goal_x": GOAL_X})


@configclass
class FinalProjectUnitreeH1EnvCfg_PLAY(FinalProjectUnitreeH1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class FinalProjectUnitreeH1MapEnvCfg(FinalProjectUnitreeH1EnvCfg):
    """Stage B: final-map fine-tuning for competition optimization."""

    def __post_init__(self):
        super().__post_init__()

        # Load the arena map through the stock terrain importer instead of as a separate scene asset.
        self.scene.final_map = None
        self.curriculum.terrain_levels = None

        # Preserve the observation width from the curriculum stage without depending on map mesh parsing.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = ObsTerm(
            func=custom_mdp.zero_height_scan,
            params={"num_rays": 187},
        )

        # Map stage is more expensive, so start smaller by default.
        self.scene.num_envs = 64
        self.scene.env_spacing = MAP_ENV_SPACING
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
class FinalProjectUnitreeH1StabilityArenaEnvCfg(FinalProjectUnitreeH1EnvCfg):
    """Stage A1: warm-up on curated easy terrain arena."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None
        self.scene.num_envs = 64
        self.scene.env_spacing = 0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
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
        goal_x = float(layout["suggested_goal_x"])
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
            params={"goal_x": goal_x},
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
        goal_x = float(layout["suggested_goal_x"])
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
            params={"goal_x": goal_x},
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


def _validate_shared_map_env_cfg(num_envs: int, env_spacing: float) -> None:
    """Reject unsupported shared-map layouts before training starts."""

    if env_spacing != 0:
        raise ValueError(
            "Final-map tasks use one shared USD arena, so scene.env_spacing must stay at 0. "
            "The current setup does not clone the map per environment."
        )

    if num_envs < 1:
        raise ValueError("scene.num_envs must be at least 1.")
