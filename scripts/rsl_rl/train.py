# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
from math import isfinite
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--shutdown-eval",
    action="store_true",
    default=False,
    help="After graceful shutdown, run a short headless evaluation from the saved shutdown checkpoint.",
)
parser.add_argument(
    "--shutdown-video",
    action="store_true",
    default=False,
    help="Record a video during the post-shutdown evaluation run.",
)
parser.add_argument(
    "--shutdown-video-length",
    type=int,
    default=300,
    help="Video length in steps for the post-shutdown evaluation run.",
)
parser.add_argument(
    "--shutdown-eval-steps",
    type=int,
    default=300,
    help="Maximum number of steps for the post-shutdown evaluation run.",
)
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
import time
from datetime import datetime

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


class _GracefulShutdownRequested(RuntimeError):
    """Raised at a safe iteration boundary after a shutdown signal is received."""


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


def _tensor_to_float(value) -> float | None:
    """Convert scalars/tensors to a JSON-safe float when possible."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().mean().item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize_episode_infos(ep_infos: list[dict]) -> dict[str, float]:
    """Average episode info metrics into a compact JSON-safe dictionary."""

    if not ep_infos:
        return {}

    summary: dict[str, float] = {}
    for key in ep_infos[0]:
        values = []
        for ep_info in ep_infos:
            if key not in ep_info:
                continue
            value = _tensor_to_float(ep_info[key])
            if value is not None:
                values.append(value)
        if values:
            summary[key] = float(sum(values) / len(values))
    return summary


def _build_recent_metrics_snapshot(runner, locs: dict) -> dict:
    """Create a compact metrics snapshot from the latest completed iteration."""

    snapshot = {
        "iteration": int(locs["it"]),
        "total_iterations": int(locs["tot_iter"]),
        "total_timesteps": int(runner.tot_timesteps),
        "total_time_s": float(runner.tot_time),
        "collection_time_s": float(locs["collection_time"]),
        "learning_time_s": float(locs["learn_time"]),
        "policy_mean_noise_std": float(runner.alg.policy.action_std.mean().item()),
        "losses": {key: float(value) for key, value in locs["loss_dict"].items()},
        "episode_metrics": _summarize_episode_infos(locs["ep_infos"]),
    }

    if len(locs["rewbuffer"]) > 0:
        snapshot["mean_reward"] = float(statistics.mean(locs["rewbuffer"]))
        snapshot["mean_episode_length"] = float(statistics.mean(locs["lenbuffer"]))
    return snapshot


def _write_shutdown_summary(log_dir: str, summary: dict) -> str:
    """Persist the shutdown metadata next to the run artifacts."""

    summary_path = os.path.join(log_dir, "shutdown_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    _write_run_summary(log_dir, summary)
    return summary_path


def _write_run_summary(log_dir: str, summary: dict) -> str:
    """Persist a generic run summary for both completed and interrupted runs."""

    summary_path = os.path.join(log_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary_path


def _safe_metric(summary: dict, key: str):
    """Read a scalar metric from the latest metrics snapshot when available."""

    latest_metrics = summary.get("latest_metrics") or {}
    episode_metrics = latest_metrics.get("episode_metrics") or {}
    value = episode_metrics.get(key)
    if value is None:
        value = latest_metrics.get(key)
    if isinstance(value, (int, float)) and isfinite(value):
        return float(value)
    return None


def _format_metric(value: float | None, precision: int = 4) -> str:
    """Render a compact numeric value for markdown output."""

    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _find_previous_run_summary(log_dir: str, experiment_name: str) -> dict | None:
    """Locate the most recent prior summary for the same experiment."""

    experiment_root = Path(log_dir).parent
    current_run = Path(log_dir).name
    candidate_runs = sorted(path for path in experiment_root.iterdir() if path.is_dir() and path.name != current_run)
    for run_dir in reversed(candidate_runs):
        for summary_name in ("run_summary.json", "shutdown_summary.json"):
            summary_path = run_dir / summary_name
            if not summary_path.is_file():
                continue
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if summary.get("experiment_name") == experiment_name:
                return summary
    return None


def _build_improvement_note(summary: dict, previous_summary: dict | None) -> str:
    """Summarize the most relevant improvement signal for the run."""

    goal_progress = _safe_metric(summary, "Episode_Reward/goal_progress")
    goal_reached = _safe_metric(summary, "Episode_Termination/goal_reached")
    base_contact = _safe_metric(summary, "Episode_Termination/base_contact")
    mean_reward = _safe_metric(summary, "mean_reward")

    if previous_summary is None:
        if goal_reached is not None and goal_reached > 0.01:
            return "Goal-reaching is active in this run, but the conversion rate is still too low to call the task stable."
        if goal_progress is not None and goal_progress > 0.001:
            return "Goal progress is alive, but the policy is still failing before it can turn that progress into reliable finishes."
        if base_contact is not None and base_contact > 0.5:
            return "The run is still dominated by base-contact failures, so survival remains the main bottleneck."
        if mean_reward is not None:
            return "This run produced a usable baseline snapshot, but it still needs a clearer task-level improvement signal."
        return "This run created the first logged result entry for the experiment."

    prev_goal_progress = _safe_metric(previous_summary, "Episode_Reward/goal_progress")
    prev_goal_reached = _safe_metric(previous_summary, "Episode_Termination/goal_reached")
    prev_base_contact = _safe_metric(previous_summary, "Episode_Termination/base_contact")
    prev_mean_reward = _safe_metric(previous_summary, "mean_reward")

    if (
        goal_reached is not None
        and prev_goal_reached is not None
        and goal_reached > prev_goal_reached + 0.002
    ):
        return (
            f"Goal completion improved ({_format_metric(prev_goal_reached)} -> {_format_metric(goal_reached)}), "
            "which is the strongest task-level gain in this run."
        )
    if (
        goal_progress is not None
        and prev_goal_progress is not None
        and goal_progress > prev_goal_progress + 0.0005
    ):
        if (
            base_contact is not None
            and prev_base_contact is not None
            and base_contact < prev_base_contact - 0.05
        ):
            return (
                f"Goal progress improved ({_format_metric(prev_goal_progress)} -> {_format_metric(goal_progress)}) "
                f"while base-contact failures also dropped ({_format_metric(prev_base_contact)} -> {_format_metric(base_contact)})."
            )
        return (
            f"Goal progress improved ({_format_metric(prev_goal_progress)} -> {_format_metric(goal_progress)}), "
            "but falls are still capping the gain."
        )
    if (
        base_contact is not None
        and prev_base_contact is not None
        and base_contact < prev_base_contact - 0.05
    ):
        return (
            f"Base-contact terminations dropped ({_format_metric(prev_base_contact)} -> {_format_metric(base_contact)}), "
            "so stability improved even though task completion is still weak."
        )
    if mean_reward is not None and prev_mean_reward is not None and mean_reward > prev_mean_reward + 0.02:
        return (
            f"Mean reward improved ({_format_metric(prev_mean_reward, 3)} -> {_format_metric(mean_reward, 3)}), "
            "but the task terms still need a cleaner win signal."
        )
    return "No clear task-level improvement over the previous logged run; the current bottleneck is still the same."


def _build_next_improvement_note(summary: dict) -> str:
    """Suggest the next concrete improvement target from the latest metrics."""

    goal_progress = _safe_metric(summary, "Episode_Reward/goal_progress")
    goal_reached = _safe_metric(summary, "Episode_Termination/goal_reached")
    base_contact = _safe_metric(summary, "Episode_Termination/base_contact")
    noise_std = _safe_metric(summary, "policy_mean_noise_std")

    if noise_std is not None and noise_std > 1.5:
        return "Stabilize PPO first: reduce exploration pressure so the policy stops diffusing before more reward tuning."
    if base_contact is not None and base_contact > 0.5:
        return "Focus on early-stance survival: tighten reset/support shaping and verify `base_contact` drops before other reward changes."
    if goal_progress is not None and goal_progress < 0.001:
        return "Re-check the resolved config and reward wiring so `goal_progress` is materially alive in the run, not just enabled on paper."
    if goal_reached is not None and goal_reached < 0.01:
        return "Keep the dense progress reward, but strengthen the bridge from partial progress to actual goal completion."
    return "Run a narrow ablation on the current best setup and change one bottleneck at a time so the gain stays attributable."


def _append_results_entry(summary: dict):
    """Append a concise experiment-level markdown entry for the finished run."""

    log_dir = summary["log_dir"]
    experiment_name = summary["experiment_name"]
    experiment_root = Path(log_dir).parent
    results_path = experiment_root / "RESULTS.md"
    run_name = Path(log_dir).name

    file_exists = results_path.is_file()
    if file_exists:
        existing_text = results_path.read_text(encoding="utf-8")
        if f"- Run: `{run_name}`" in existing_text:
            return
    else:
        existing_text = "# Results\n\n"

    previous_summary = _find_previous_run_summary(log_dir, experiment_name)
    latest_metrics = summary.get("latest_metrics") or {}
    finished_at = summary.get("finished_at") or datetime.now().astimezone().isoformat(timespec="seconds")

    status = "completed"
    if summary.get("requested_shutdown"):
        reason = summary.get("reason") or "requested stop"
        status = f"stopped via {reason}"

    entry_lines = [
        f"## {run_name}",
        f"- Finished: `{finished_at}`",
        f"- Run: `{run_name}`",
        f"- Task: `{summary.get('task', 'n/a')}`",
        f"- Status: {status}",
        (
            "- Progress: "
            f"{summary.get('completed_iteration', 0)} / {latest_metrics.get('total_iterations', 'n/a')} iterations, "
            f"{summary.get('total_timesteps', 0)} timesteps"
        ),
        (
            "- Snapshot: "
            f"reward={_format_metric(_safe_metric(summary, 'mean_reward'), 3)}, "
            f"ep_len={_format_metric(_safe_metric(summary, 'mean_episode_length'), 2)}, "
            f"goal_progress={_format_metric(_safe_metric(summary, 'Episode_Reward/goal_progress'))}, "
            f"goal_reached={_format_metric(_safe_metric(summary, 'Episode_Termination/goal_reached'))}, "
            f"base_contact={_format_metric(_safe_metric(summary, 'Episode_Termination/base_contact'))}"
        ),
        f"- Improvement: {_build_improvement_note(summary, previous_summary)}",
        f"- Next: {_build_next_improvement_note(summary)}",
    ]
    entry = "\n".join(entry_lines)

    if file_exists and existing_text.rstrip():
        new_text = existing_text.rstrip() + "\n\n---\n\n" + entry + "\n"
    else:
        new_text = existing_text + entry + "\n"
    results_path.write_text(new_text, encoding="utf-8")


def _finalize_shutdown_eval(shutdown_summary: dict, eval_result: subprocess.CompletedProcess, log_dir: str):
    """Attach post-shutdown evaluation results and rewrite the shutdown summary."""

    eval_summary_path = os.path.join(log_dir, "shutdown_eval_summary.json")
    shutdown_summary["shutdown_eval"] = {
        "summary_path": eval_summary_path,
        "returncode": int(eval_result.returncode),
        "stdout_tail": eval_result.stdout[-4000:],
        "stderr_tail": eval_result.stderr[-4000:],
    }
    summary_path = _write_shutdown_summary(log_dir, shutdown_summary)
    if eval_result.returncode != 0:
        print("[WARN] Post-shutdown evaluation failed.")
    else:
        print(f"[INFO] Post-shutdown evaluation artifacts written under: {log_dir}")
    print(f"[INFO] Graceful shutdown summary: {summary_path}")


def _run_shutdown_eval(
    task_name: str,
    checkpoint_path: str,
    eval_steps: int,
    video: bool,
    video_length: int,
    summary_path: str,
) -> subprocess.CompletedProcess:
    """Launch a short isolated play run from the saved shutdown checkpoint."""

    play_script = Path(__file__).with_name("play.py")
    cmd = [
        sys.executable,
        str(play_script),
        "--task",
        task_name,
        "--checkpoint",
        checkpoint_path,
        "--headless",
        "--num_envs",
        "1",
        "--max_steps",
        str(eval_steps),
        "--summary-json",
        summary_path,
    ]
    if video:
        cmd.extend(["--video", "--video_length", str(video_length)])

    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _install_graceful_shutdown(runner, log_dir: str, task_name: str):
    """Install signal and iteration-boundary shutdown handling on the runner."""

    state = {
        "requested": False,
        "reason": None,
        "signal_count": 0,
        "checkpoint_path": None,
        "latest_metrics": None,
    }

    original_log = runner.log

    def _signal_handler(signum, _frame):
        state["signal_count"] += 1
        signal_name = signal.Signals(signum).name
        if state["requested"]:
            raise KeyboardInterrupt(f"Forced shutdown requested by {signal_name}.")
        state["requested"] = True
        state["reason"] = signal_name
        print(
            f"[INFO] Received {signal_name}. Finishing the current iteration, "
            "saving a shutdown checkpoint, and writing run artifacts."
        )

    def guarded_log(self, locs: dict, width: int = 80, pad: int = 35):
        state["latest_metrics"] = _build_recent_metrics_snapshot(self, locs)
        original_log(locs, width=width, pad=pad)
        if state["requested"]:
            checkpoint_path = os.path.join(log_dir, "model_shutdown.pt")
            infos = {
                "shutdown_reason": state["reason"],
                "task": task_name,
                "total_timesteps": self.tot_timesteps,
            }
            self.save(checkpoint_path, infos=infos)
            state["checkpoint_path"] = checkpoint_path
            raise _GracefulShutdownRequested(f"Graceful shutdown requested via {state['reason']}.")

    runner.log = MethodType(guarded_log, runner)

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    def restore_handlers():
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    return state, restore_handlers


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
    shutdown_state, restore_signal_handlers = _install_graceful_shutdown(runner, log_dir, args_cli.task)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    shutdown_summary = {
        "task": args_cli.task,
        "experiment_name": agent_cfg.experiment_name,
        "log_dir": log_dir,
        "requested_shutdown": False,
        "reason": None,
        "total_timesteps": int(runner.tot_timesteps),
        "completed_iteration": int(runner.current_learning_iteration),
        "checkpoint_path": None,
        "latest_metrics": None,
        "shutdown_eval": None,
    }
    deferred_shutdown_eval = None
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        shutdown_summary["reason"] = "completed"
        shutdown_summary["latest_metrics"] = shutdown_state["latest_metrics"]
        shutdown_summary["total_timesteps"] = int(runner.tot_timesteps)
        shutdown_summary["completed_iteration"] = int(runner.current_learning_iteration)
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    except _GracefulShutdownRequested as exc:
        print(f"[INFO] {exc}")
        shutdown_summary["requested_shutdown"] = True
        shutdown_summary["reason"] = shutdown_state["reason"]
        shutdown_summary["checkpoint_path"] = shutdown_state["checkpoint_path"]
        shutdown_summary["latest_metrics"] = shutdown_state["latest_metrics"]
        shutdown_summary["total_timesteps"] = int(runner.tot_timesteps)
        shutdown_summary["completed_iteration"] = int(runner.current_learning_iteration)

        summary_path = _write_shutdown_summary(log_dir, shutdown_summary)
        print(f"[INFO] Graceful shutdown checkpoint: {shutdown_state['checkpoint_path']}")
        print(f"[INFO] Graceful shutdown summary: {summary_path}")

        if args_cli.shutdown_eval and shutdown_state["checkpoint_path"] is not None:
            shutdown_summary["shutdown_eval"] = {
                "scheduled": True,
                "summary_path": os.path.join(log_dir, "shutdown_eval_summary.json"),
            }
            summary_path = _write_shutdown_summary(log_dir, shutdown_summary)
            deferred_shutdown_eval = {
                "task_name": args_cli.task,
                "checkpoint_path": shutdown_state["checkpoint_path"],
                "eval_steps": args_cli.shutdown_eval_steps,
                "video": args_cli.shutdown_video,
                "video_length": args_cli.shutdown_video_length,
                "summary_path": os.path.join(log_dir, "shutdown_eval_summary.json"),
                "log_dir": log_dir,
                "shutdown_summary": shutdown_summary,
            }
            print("[INFO] Shutdown eval scheduled after simulator teardown.")
            print(f"[INFO] Graceful shutdown summary: {summary_path}")
        else:
            shutdown_summary["shutdown_eval"] = {"skipped": True}
            summary_path = _write_shutdown_summary(log_dir, shutdown_summary)
            print(f"[INFO] Graceful shutdown summary: {summary_path}")
    finally:
        restore_signal_handlers()
        env.close()

    simulation_app.close()
    if deferred_shutdown_eval is not None:
        result = _run_shutdown_eval(
            task_name=deferred_shutdown_eval["task_name"],
            checkpoint_path=deferred_shutdown_eval["checkpoint_path"],
            eval_steps=deferred_shutdown_eval["eval_steps"],
            video=deferred_shutdown_eval["video"],
            video_length=deferred_shutdown_eval["video_length"],
            summary_path=deferred_shutdown_eval["summary_path"],
        )
        _finalize_shutdown_eval(
            shutdown_summary=deferred_shutdown_eval["shutdown_summary"],
            eval_result=result,
            log_dir=deferred_shutdown_eval["log_dir"],
        )
        deferred_shutdown_eval["shutdown_summary"]["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_run_summary(deferred_shutdown_eval["log_dir"], deferred_shutdown_eval["shutdown_summary"])
    else:
        shutdown_summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_run_summary(log_dir, shutdown_summary)


if __name__ == "__main__":
    # run the main function
    main()
