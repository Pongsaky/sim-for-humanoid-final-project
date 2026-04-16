from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_height_scan(env: ManagerBasedRLEnv, num_rays: int) -> torch.Tensor:
    """Return a zero height-scan vector to preserve observation size during map fine-tuning."""
    return torch.zeros((env.num_envs, num_rays), device=env.device, dtype=torch.float32)


def goal_distance_x(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Return signed remaining distance to the goal line along the environment x-axis."""
    robot = env.scene["robot"]
    if start_x is None:
        goal_line_x = env.scene.env_origins[:, 0] + goal_x
    else:
        goal_line_x = robot.data.root_pos_w[:, 0].new_full((env.num_envs,), float(start_x + goal_x))
    remaining = goal_line_x - robot.data.root_pos_w[:, 0]
    if normalize:
        remaining = remaining / max(goal_x, 1e-6)
    return remaining.unsqueeze(-1)
