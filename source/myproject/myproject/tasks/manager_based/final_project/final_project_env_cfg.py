import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1Rewards, H1RoughEnvCfg

from myproject.assets.unitree_humanoid_config import UNITREE_H1_MINIMAL_CFG
from myproject.assets.usd_prim_reference_cfg import UsdFileWithPrimPathCfg

from . import mdp as custom_mdp

FINAL_MAP_USD_PATH = "/home/pongsaky/university/sim-for-humaniod/final_map.usd"
FINAL_MAP_USD_PRIM_PATH = "/World/ground"
GOAL_X = 18.0

FINAL_PROJECT_CURRICULUM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # Phase-1 warm-up and easy locomotion.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "rough_easy": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.0, 0.03), noise_step=0.01, border_width=0.25
        ),
        "rough_hard": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.02, 0.06), noise_step=0.01, border_width=0.25
        ),
        # Gap-like trench challenge.
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.10, 0.60),
            platform_width=1.6,
        ),
        # Stepping and elevated blocks.
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_height_max=0.15,
            stone_width_range=(0.30, 0.80),
            stone_distance_range=(0.05, 0.20),
            holes_depth=-2.0,
            platform_width=1.8,
        ),
        # Fan/star-like obstacles.
        "star_obstacles": terrain_gen.MeshStarTerrainCfg(
            proportion=0.15,
            num_bars=8,
            bar_width_range=(0.10, 0.28),
            bar_height_range=(0.06, 0.14),
            platform_width=1.6,
        ),
    },
)


@configclass
class FinalProjectRewards(H1Rewards):
    """Reward mix tuned for speed-to-goal locomotion."""

    forward_velocity = RewTerm(func=custom_mdp.forward_velocity_toward_goal, weight=2.0, params={"goal_x": GOAL_X})
    goal_progress = RewTerm(func=custom_mdp.goal_distance_progress, weight=1.0, params={"goal_x": GOAL_X})
    goal_reached_bonus = RewTerm(
        func=custom_mdp.goal_reached_bonus,
        weight=100.0,
        params={"goal_x": GOAL_X, "bonus": 1.0},
    )
    obstacle_zone_crossing = RewTerm(
        func=custom_mdp.zone_crossing_bonus,
        weight=0.75,
        params={"zone_positions": (3.0, 6.5, 10.0, 13.5, 16.0), "sigma": 0.35},
    )
    low_height_penalty = RewTerm(func=custom_mdp.base_height_penalty, weight=-2.0, params={"min_height": 0.55})

    # Keep regularizers present but light to prioritize speed.
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)


@configclass
class FinalProjectCurriculum:
    """Difficulty progression for command speed and disturbances."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    command_speed = CurrTerm(
        func=custom_mdp.increase_command_range,
        params={"interval_steps": 5000, "max_lin_vel_x": 2.0, "step_size": 0.15},
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

        # Start with conservative commands and expand via curriculum.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)

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

        # Use a flat plane under the map and disable terrain-level curriculum for map adaptation.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # Duplicate map per environment (smaller env count recommended for this stage).
        self.scene.final_map = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/final_map",
            spawn=UsdFileWithPrimPathCfg(
                usd_path=FINAL_MAP_USD_PATH,
                usd_prim_path=FINAL_MAP_USD_PRIM_PATH,
            ),
        )

        # Preserve the observation width from the curriculum stage without depending on map mesh parsing.
        self.scene.height_scanner = None
        self.observations.policy.height_scan = ObsTerm(
            func=custom_mdp.zero_height_scan,
            params={"num_rays": 187},
        )

        # Map stage is more expensive, so start smaller by default.
        self.scene.num_envs = 256
        self.scene.env_spacing = 8.0


@configclass
class FinalProjectUnitreeH1MapEnvCfg_PLAY(FinalProjectUnitreeH1MapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class FinalProjectUnitreeH1BaselineEnvCfg(H1FlatEnvCfg):
    """Baseline: stock H1 flat locomotion setup, trained from scratch on the final map."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # The map needs to exist in every cloned environment. A single USD terrain import only creates one global map.
        self.scene.terrain = None
        self.scene.final_map = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/final_map",
            spawn=UsdFileWithPrimPathCfg(
                usd_path=FINAL_MAP_USD_PATH,
                usd_prim_path=FINAL_MAP_USD_PRIM_PATH,
            ),
        )

        # Map duplication is heavy enough that a smaller default is more realistic.
        self.scene.num_envs = 256
        self.scene.env_spacing = 8.0


@configclass
class FinalProjectUnitreeH1BaselineEnvCfg_PLAY(FinalProjectUnitreeH1BaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
