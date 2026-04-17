import gymnasium as gym

from . import agents

gym.register(
    id="Template-Final-Project-Unitree-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1PPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1PPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1BaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1BaselinePPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-Baseline-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1BaselineEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1BaselinePPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-RoughGoal-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1RoughGoalBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1RoughGoalBaselinePPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-RoughGoal-Baseline-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1RoughGoalBaselineEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1RoughGoalBaselinePPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-SpeedRun-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1SpeedRunEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1SpeedRunPPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-SpeedRun-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1SpeedRunEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1SpeedRunPPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-FastWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1FastWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1FastWalkPPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-FastWalk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1FastWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1FastWalkPPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-Stability-Arena-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1StabilityArenaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1StabilityWarmupPPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-Stability-Arena-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1StabilityArenaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1StabilityWarmupPPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-Crossing-Arena-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1CrossingArenaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1PPORunnerCfg",
    },
)

gym.register(
    id="Template-Final-Project-Unitree-H1-Crossing-Arena-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.final_project_env_cfg:FinalProjectUnitreeH1CrossingArenaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FinalProjectUnitreeH1PPORunnerCfg",
    },
)
