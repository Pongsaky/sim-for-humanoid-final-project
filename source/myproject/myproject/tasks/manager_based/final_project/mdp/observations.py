from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_height_scan(env: ManagerBasedRLEnv, num_rays: int) -> torch.Tensor:
    """Return a zero height-scan vector to preserve observation size during map fine-tuning."""
    return torch.zeros((env.num_envs, num_rays), device=env.device, dtype=torch.float32)


def goal_distance_axis(
    env: ManagerBasedRLEnv,
    goal: float,
    start: float | None = None,
    axis: str = "x",
    normalize: bool = True,
) -> torch.Tensor:
    """Return signed remaining distance to a goal line along the selected world axis."""
    robot = env.scene["robot"]
    axis_idx = _axis_index(axis)
    if start is None:
        goal_line = env.scene.env_origins[:, axis_idx] + goal
    else:
        goal_line = robot.data.root_pos_w[:, axis_idx].new_full((env.num_envs,), float(start + goal))
    remaining = goal_line - robot.data.root_pos_w[:, axis_idx]
    if normalize:
        remaining = remaining / max(abs(goal), 1e-6)
    return remaining.unsqueeze(-1)


def goal_distance_x(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Return signed remaining distance to the goal line along the environment x-axis."""
    return goal_distance_axis(env=env, goal=goal_x, start=start_x, axis="x", normalize=normalize)


def _axis_index(axis: str) -> int:
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    raise ValueError(f"Unsupported axis '{axis}'. Expected 'x' or 'y'.")
