# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--num_envs", "--num_env", dest="num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime
from types import MethodType

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import myproject.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _ensure_tensor_finite(name: str, tensor: torch.Tensor, iteration: int | None = None) -> None:
    """Raise a clear error as soon as NaN/Inf appears in training state."""

    if not torch.is_tensor(tensor):
        return
    if tensor.numel() == 0:
        return
    if torch.isfinite(tensor).all():
        return

    prefix = f"[iteration {iteration}] " if iteration is not None else ""
    finite_mask = torch.isfinite(tensor)
    non_finite = (~finite_mask).sum().item()
    finite_values = tensor[finite_mask]
    if finite_values.numel() > 0:
        finite_min = finite_values.min().item()
        finite_max = finite_values.max().item()
    else:
        finite_min = float("nan")
        finite_max = float("nan")
    raise RuntimeError(
        f"{prefix}Non-finite values detected in {name}: {non_finite} / {tensor.numel()} values are NaN/Inf "
        f"(finite range: [{finite_min}, {finite_max}])."
    )


def _ensure_module_parameters_finite(module: torch.nn.Module, name: str, iteration: int | None = None) -> None:
    """Validate every parameter tensor in a module."""

    for param_name, param in module.named_parameters():
        _ensure_tensor_finite(f"{name}.{param_name}", param.data, iteration=iteration)


def _install_training_diagnostics(runner) -> None:
    """Attach local numeric guards without patching upstream packages."""

    policy = runner.alg.policy
    state = {"iteration": 0}

    original_update_distribution = policy._update_distribution
    original_update = runner.alg.update

    def guarded_update_distribution(self, obs):
        if hasattr(obs, "items"):
            for obs_name, obs_tensor in obs.items():
                _ensure_tensor_finite(f"policy_obs.{obs_name}", obs_tensor, iteration=state["iteration"])
        else:
            _ensure_tensor_finite("policy_obs", obs, iteration=state["iteration"])

        original_update_distribution(obs)

        _ensure_tensor_finite("policy.action_mean", self.action_mean, iteration=state["iteration"])
        _ensure_tensor_finite("policy.action_std", self.action_std, iteration=state["iteration"])

    def guarded_update(self):
        state["iteration"] += 1
        iteration = state["iteration"]
        _ensure_module_parameters_finite(self.policy, "policy", iteration=iteration)
        if hasattr(self.policy, "log_std"):
            _ensure_tensor_finite("policy.log_std", self.policy.log_std.data, iteration=iteration)
        elif hasattr(self.policy, "std"):
            _ensure_tensor_finite("policy.std", self.policy.std.data, iteration=iteration)

        loss_dict = original_update()

        _ensure_module_parameters_finite(self.policy, "policy", iteration=iteration)
        if hasattr(self.policy, "action_std"):
            _ensure_tensor_finite("policy.action_std", self.policy.action_std, iteration=iteration)
        if hasattr(self.policy, "action_mean"):
            _ensure_tensor_finite("policy.action_mean", self.policy.action_mean, iteration=iteration)

        return loss_dict

    policy._update_distribution = MethodType(guarded_update_distribution, policy)
    runner.alg.update = MethodType(guarded_update, runner.alg)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    execution_mode = "headless" if app_launcher._headless else "gui"
    env_cfg.shared_arena_physx_profile = "training_headless" if app_launcher._headless else "training_gui"
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    finalize_env_cfg = getattr(env_cfg, "finalize_after_overrides", None)
    if callable(finalize_env_cfg):
        finalize_env_cfg()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_experiment_name = (
            args_cli.load_experiment if args_cli.load_experiment is not None else agent_cfg.experiment_name
        )
        resume_root_path = os.path.join("logs", "rsl_rl", resume_experiment_name)
        resume_root_path = os.path.abspath(resume_root_path)
        resume_path = get_checkpoint_path(resume_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(
        "[INFO] Resolved runner config: "
        f"task={args_cli.task}, num_envs={env.num_envs}, device={agent_cfg.device}, "
        f"execution_mode={execution_mode}, shared_arena_physx_profile={env_cfg.shared_arena_physx_profile}, "
        f"noise_std_type={agent_cfg.policy.noise_std_type}, init_noise_std={agent_cfg.policy.init_noise_std}, "
        f"lr={agent_cfg.algorithm.learning_rate}, desired_kl={agent_cfg.algorithm.desired_kl}, "
        f"entropy_coef={agent_cfg.algorithm.entropy_coef}"
    )
    if not app_launcher._headless and "Template-Final-Project-Unitree-H1" in args_cli.task:
        print(
            "[WARN] GUI mode uses reduced PhysX capacities for shared-arena tasks. "
            "Prefer low env counts in non-headless runs to avoid GPU memory pressure."
        )

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    _ensure_module_parameters_finite(runner.alg.policy, "policy")
    if hasattr(runner.alg.policy, "log_std"):
        _ensure_tensor_finite("policy.log_std", runner.alg.policy.log_std.data)
    elif hasattr(runner.alg.policy, "std"):
        _ensure_tensor_finite("policy.std", runner.alg.policy.std.data)
    _install_training_diagnostics(runner)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
