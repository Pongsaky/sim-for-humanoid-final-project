from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def increase_command_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    interval_steps: int,
    max_lin_vel_x: float,
    step_size: float = 0.1,
) -> float:
    """Increase forward command range over training to push speed progressively."""
    del env_ids
    command_cfg = env.command_manager.get_term("base_velocity").cfg
    current_min, current_max = command_cfg.ranges.lin_vel_x

    if env.common_step_counter > 0 and env.common_step_counter % interval_steps == 0:
        current_max = float(np.clip(current_max + step_size, 0.6, max_lin_vel_x))
        command_cfg.ranges.lin_vel_x = (0.0, current_max)

    return command_cfg.ranges.lin_vel_x[1]


def increase_push_disturbance(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    interval_steps: int,
    max_push_vel: float,
    step_size: float = 0.05,
) -> float:
    """Progressively increase push disturbance magnitude."""
    del env_ids
    try:
        term_cfg = env.event_manager.get_term_cfg("push_robot")
    except Exception:
        return 0.0

    if env.common_step_counter > 0 and env.common_step_counter % interval_steps == 0:
        current = float(term_cfg.params["velocity_range"]["x"][1])
        current = float(np.clip(current + step_size, 0.0, max_push_vel))
        term_cfg.params["velocity_range"]["x"] = (-current, current)
        term_cfg.params["velocity_range"]["y"] = (-current, current)
        env.event_manager.set_term_cfg("push_robot", term_cfg)

    return float(term_cfg.params["velocity_range"]["x"][1])


def success_rate_proxy(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    goal_x: float,
) -> torch.Tensor:
    """Simple logging curriculum term: fraction of envs past goal line at reset."""
    robot = env.scene["robot"]
    env_x = env.scene.env_origins[env_ids, 0]
    reached = robot.data.root_pos_w[env_ids, 0] >= (env_x + goal_x)
    return reached.float().mean()


def anneal_init_lin_vel_x(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_max: float = 1.2,
    start_min: float = 0.6,
    end_iter: int = 2000,
    iters_per_step: int = 24,
    reset_term_name: str = "reset_base",
) -> float:
    """Linearly anneal reset_base.lin_vel_x_range from (start_min, start_max) to (0, 0).

    Returns the current upper bound for tensorboard logging.
    """
    del env_ids
    iter_idx = env.common_step_counter // max(iters_per_step, 1)
    progress = float(np.clip(iter_idx / max(end_iter, 1), 0.0, 1.0))
    cur_max = float(start_max * (1.0 - progress))
    cur_min = float(start_min * (1.0 - progress))
    try:
        term_cfg = env.event_manager.get_term_cfg(reset_term_name)
    except Exception:
        return cur_max
    term_cfg.params["lin_vel_x_range"] = (cur_min, cur_max)
    env.event_manager.set_term_cfg(reset_term_name, term_cfg)
    return cur_max
