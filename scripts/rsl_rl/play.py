# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of evaluation steps before exit.")
parser.add_argument(
    "--summary-json",
    type=str,
    default=None,
    help="Optional path to write a compact evaluation summary as JSON.",
)
parser.add_argument(
    "--disable-completion-hud",
    action="store_true",
    default=False,
    help="Disable the Isaac Sim HUD window that shows current and best completion time during play.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs", "--num_env", dest="num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
from collections import defaultdict

import gymnasium as gym
import myproject.tasks  # noqa: F401
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _scalarize(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_play_task_name(task_name: str | None) -> str | None:
    """Prefer the registered *-Play task when using this script."""

    if task_name is None or "-Play-" in task_name:
        return task_name
    if "-v" not in task_name:
        return task_name
    play_task_name = task_name.replace("-v", "-Play-v", 1)
    if play_task_name in gym.registry:
        print(f"[INFO] Replacing task '{task_name}' with play task '{play_task_name}'.")
        return play_task_name
    return task_name


def _mask_to_list(mask_like, expected_len: int) -> list[bool]:
    """Convert a scalar/list/tensor done mask to a flat Python bool list."""

    if isinstance(mask_like, torch.Tensor):
        flat = mask_like.detach().reshape(-1).to(device="cpu", dtype=torch.bool).tolist()
    elif isinstance(mask_like, (list, tuple)):
        flat = [bool(v) for v in mask_like]
    else:
        flat = [bool(mask_like)]

    if len(flat) == expected_len:
        return flat
    if len(flat) == 1 and expected_len > 1:
        return flat * expected_len
    return (flat + [False] * expected_len)[:expected_len]


class _CompletionHud:
    """Small Isaac Sim HUD for current and best wall-clock completion time."""

    def __init__(self):
        self._window = None
        self._round_label = None
        self._status_label = None
        self._current_label = None
        self._best_label = None

        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window(
            "Completion Time",
            width=300,
            height=130,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self._window.frame:
            with ui.VStack(spacing=4):
                self._round_label = ui.Label("Round: 1")
                self._status_label = ui.Label("Status: waiting")
                self._current_label = ui.Label("Current Time: 0.00s")
                self._best_label = ui.Label("Best Wall Time: --")

    @property
    def enabled(self) -> bool:
        return self._window is not None

    def update(self, round_idx: int, status: str, current_s: float | None, best_s: float | None) -> None:
        if not self.enabled:
            return
        self._round_label.text = f"Round: {round_idx}"
        self._status_label.text = f"Status: {status}"
        self._current_label.text = f"Current Time: {0.0 if current_s is None else current_s:.2f}s"
        self._best_label.text = f"Best Wall Time: {'--' if best_s is None else f'{best_s:.2f}s'}"

    def close(self) -> None:
        if self._window is not None:
            self._window.visible = False
            self._window = None


args_cli.task = _resolve_play_task_name(args_cli.task)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if os.path.basename(resume_path) == "policy.pt" and os.path.basename(os.path.dirname(resume_path)) == "exported":
        raise ValueError(
            "Received an exported TorchScript policy at "
            f"'{resume_path}'. scripts/rsl_rl/play.py expects an RSL-RL training checkpoint "
            "(for example 'model_1000.pt'), not the exported 'policy.pt' artifact."
        )

    log_dir = os.path.dirname(resume_path)

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

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    base_env = env.unwrapped
    dt = base_env.step_dt
    episode_round = 1
    best_completion_time_s: float | None = None
    round_wall_start_time = time.perf_counter()
    hud = None if args_cli.disable_completion_hud or base_env.num_envs != 1 else _CompletionHud()
    if hud is not None and hud.enabled:
        hud.update(round_idx=episode_round, status="running", current_s=0.0, best_s=best_completion_time_s)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    episode_metric_values: dict[str, list[float]] = defaultdict(list)
    completion_times: list[float] = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            done_mask = _mask_to_list(dones, base_env.num_envs)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if "episode" in extras:
                for key, value in extras["episode"].items():
                    scalar = _scalarize(value)
                    if scalar is not None:
                        episode_metric_values[key].append(scalar)
            elif "log" in extras:
                for key, value in extras["log"].items():
                    scalar = _scalarize(value)
                    if scalar is not None:
                        episode_metric_values[key].append(scalar)
            # Collect per-episode goal completion times written by completion_time_metric.
            for t in extras.get("completion_times_s", []):
                completion_times.append(float(t))
            if base_env.num_envs == 1:
                current_time_s = time.perf_counter() - round_wall_start_time
                goal_events = extras.get("completion_times_s", [])
                if goal_events:
                    completion_time_s = current_time_s
                    best_completion_time_s = (
                        completion_time_s
                        if best_completion_time_s is None
                        else min(best_completion_time_s, completion_time_s)
                    )
                    if hud is not None and hud.enabled:
                        hud.update(
                            round_idx=episode_round,
                            status="goal reached",
                            current_s=completion_time_s,
                            best_s=best_completion_time_s,
                        )
                elif hud is not None and hud.enabled:
                    hud.update(
                        round_idx=episode_round,
                        status="running",
                        current_s=current_time_s,
                        best_s=best_completion_time_s,
                    )

                if done_mask[0]:
                    episode_round += 1
                    round_wall_start_time = time.perf_counter()
                    if hud is not None and hud.enabled:
                        hud.update(
                            round_idx=episode_round,
                            status="running",
                            current_s=0.0,
                            best_s=best_completion_time_s,
                        )
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if completion_times:
        print(
            f"[Completion Time] n={len(completion_times)}"
            f"  best={min(completion_times):.2f}s"
            f"  mean={sum(completion_times)/len(completion_times):.2f}s"
            f"  worst={max(completion_times):.2f}s"
        )

    if args_cli.summary_json:
        summary = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "steps": int(timestep),
            "completed_episode_count": int(max((len(v) for v in episode_metric_values.values()), default=0)),
            "episode_metrics": {
                key: float(sum(values) / len(values)) for key, values in episode_metric_values.items() if values
            },
        }
        if completion_times:
            summary["completion_time_s"] = {
                "best": float(min(completion_times)),
                "mean": float(sum(completion_times) / len(completion_times)),
                "worst": float(max(completion_times)),
                "n": len(completion_times),
            }
        with open(args_cli.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    if hud is not None:
        hud.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
