import json
import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import (
    H1FlatPPORunnerCfg,
    H1RoughPPORunnerCfg,
)

AUTOPILOT_PROFILE_ENV_VAR = "FINAL_PROJECT_AUTOPILOT_PROFILE"


def _load_autopilot_profile() -> dict:
    profile_path = os.environ.get(AUTOPILOT_PROFILE_ENV_VAR)
    if not profile_path:
        return {}
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return profile if isinstance(profile, dict) else {}


def _get_runner_override(section: str) -> dict:
    overrides = _load_autopilot_profile().get("runner_sections", {})
    selected = overrides.get(section, {})
    return selected if isinstance(selected, dict) else {}


def _apply_runner_override(cfg, section: str) -> None:
    overrides = _get_runner_override(section)
    if not overrides:
        return

    if "max_iterations" in overrides:
        cfg.max_iterations = int(overrides["max_iterations"])
    if "save_interval" in overrides:
        cfg.save_interval = int(overrides["save_interval"])
    if "num_steps_per_env" in overrides:
        cfg.num_steps_per_env = int(overrides["num_steps_per_env"])
    if "init_noise_std" in overrides:
        cfg.policy.init_noise_std = float(overrides["init_noise_std"])
    if "noise_std_type" in overrides:
        cfg.policy.noise_std_type = str(overrides["noise_std_type"])
    if "actor_hidden_dims" in overrides:
        cfg.policy.actor_hidden_dims = [int(v) for v in overrides["actor_hidden_dims"]]
    if "critic_hidden_dims" in overrides:
        cfg.policy.critic_hidden_dims = [int(v) for v in overrides["critic_hidden_dims"]]
    if "learning_rate" in overrides:
        cfg.algorithm.learning_rate = float(overrides["learning_rate"])
    if "entropy_coef" in overrides:
        cfg.algorithm.entropy_coef = float(overrides["entropy_coef"])
    if "desired_kl" in overrides:
        cfg.algorithm.desired_kl = float(overrides["desired_kl"])


@configclass
class FinalProjectUnitreeH1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    def __post_init__(self):
        self.num_steps_per_env = 24
        self.max_iterations = 4000
        self.clip_actions = 1.0
        self.save_interval = 50
        self.experiment_name = "final_project_unitree_h1_curriculum"
        self.run_name = ""
        self.resume = False
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.06,
            noise_std_type="scalar",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=5.0e-5,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-5,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.0015,
            max_grad_norm=1.0,
        )
        _apply_runner_override(self, "curriculum")


@configclass
class FinalProjectUnitreeH1MapPPORunnerCfg(FinalProjectUnitreeH1PPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2500
    save_interval = 50
    experiment_name = "final_project_unitree_h1_map_finetune"
    run_name = ""
    resume = False

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FinalProjectUnitreeH1BaselinePPORunnerCfg(H1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.clip_actions = 1.0
        self.save_interval = 50
        self.experiment_name = "final_project_unitree_h1_baseline"
        self.run_name = ""
        self.resume = False
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.25,
            noise_std_type="scalar",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-5,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.005,
            max_grad_norm=1.0,
        )
        _apply_runner_override(self, "baseline")


@configclass
class FinalProjectUnitreeH1StabilityWarmupPPORunnerCfg(FinalProjectUnitreeH1BaselinePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1600
        self.save_interval = 50
        self.experiment_name = "final_project_unitree_h1_stability_warmup"
        _apply_runner_override(self, "stability_warmup")


@configclass
class FinalProjectH1RoughWalkerPPORunnerCfg(H1RoughPPORunnerCfg):
    """Stage-1B warm-start PPO runner.

    Uses stock H1 rough exploration params (noise=1.0, entropy=0.01, lr=1e-3) and explicitly
    enables obs normalisation to match Stage-2, so the running-mean state transfers too.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 50
        self.experiment_name = "final_project_unitree_h1_rough_warmup"
        self.run_name = ""
        # Enable obs normalisation to match Stage-2 FinalProjectUnitreeH1RoughGoalBaselinePPORunnerCfg.
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            noise_std_type="scalar",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        # Keep all other stock H1 rough algorithm params: entropy=0.01, lr=1e-3, kl=0.01


@configclass
class FinalProjectH1FlatWalkerPPORunnerCfg(H1FlatPPORunnerCfg):
    """Stock H1 flat locomotion config used as Stage-1 warm-start for the goal task."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.experiment_name = "final_project_unitree_h1_flat_warmup"
        # Intentionally keep all stock H1 defaults:
        #   init_noise_std=1.0, entropy_coef=0.01, lr=1e-3, desired_kl=0.01
        # The project's custom configs used values 4–200x too conservative, preventing walking.


@configclass
class FinalProjectUnitreeH1RoughGoalBaselinePPORunnerCfg(H1RoughPPORunnerCfg):
    """Stage-2 goal fine-tune: stock H1 exploration params.

    Option A (default): resume=False — start from scratch with better hyperparams.
    Option B (warm-start): set resume=True and pass --load_experiment final_project_unitree_h1_rough_warmup
      so that ALL layers transfer cleanly from a matching-obs-space Stage-1B checkpoint.
    """

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000
        self.save_interval = 50
        self.experiment_name = "final_project_unitree_h1_roughgoal_baseline"
        self.run_name = ""
        # Set resume=True and --load_experiment final_project_unitree_h1_rough_warmup on CLI for Option B.
        self.resume = False
        self.load_run = "2026-04-17_12-29-19"
        self.load_checkpoint = "model_5058.pt"
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.5,
            noise_std_type="scalar",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )


@configclass
class FinalProjectUnitreeH1MinimalRewardPPORunnerCfg(FinalProjectUnitreeH1BaselinePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "final_project_unitree_h1_minimal_reward"
        _apply_runner_override(self, "minimal_reward")


@configclass
class FinalProjectUnitreeH1SpeedRunPPORunnerCfg(FinalProjectUnitreeH1RoughGoalBaselinePPORunnerCfg):
    """Speed-run PPO runner: starts fresh, same arch as rough-goal baseline."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "final_project_unitree_h1_speedrun"
        self.run_name = ""
        self.resume = False


@configclass
class FinalProjectUnitreeH1FastWalkPPORunnerCfg(FinalProjectUnitreeH1RoughGoalBaselinePPORunnerCfg):
    """Fast-walk PPO runner: starts fresh, same arch as rough-goal baseline."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "final_project_unitree_h1_fastwalk"
        self.run_name = ""
        self.resume = False
