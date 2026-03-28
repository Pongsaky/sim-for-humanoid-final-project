from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_velocity_toward_goal(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward world-frame velocity toward the goal line."""
    asset = env.scene[asset_cfg.name]
    goal_dir_w = torch.zeros_like(asset.data.root_lin_vel_w[:, :2])
    goal_dir_w[:, 0] = 1.0 if goal_x >= start_x else -1.0
    return torch.sum(asset.data.root_lin_vel_w[:, :2] * goal_dir_w, dim=1)


def goal_distance_progress(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense normalized progress reward along the arena x-axis."""
    asset = env.scene[asset_cfg.name]
    env_x = env.scene.env_origins[:, 0]
    start_x_w = env_x + start_x
    goal_x_w = env_x + goal_x
    denom = torch.full_like(start_x_w, goal_x - start_x).clamp(min=-1.0e6, max=1.0e6)
    denom = torch.where(torch.abs(denom) < 1.0e-6, torch.ones_like(denom), denom)
    progress = (asset.data.root_pos_w[:, 0] - start_x_w) / denom
    return torch.clamp(progress, min=0.0, max=1.0)


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float = 0.0,
    bonus: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-step bonus whenever robot has crossed the goal x-line."""
    asset = env.scene[asset_cfg.name]
    env_x = env.scene.env_origins[:, 0]
    goal_x_w = env_x + goal_x
    if goal_x >= start_x:
        reached = asset.data.root_pos_w[:, 0] >= goal_x_w
    else:
        reached = asset.data.root_pos_w[:, 0] <= goal_x_w
    return reached.float() * bonus


def base_height_penalty(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize low base height before terminal fall happens."""
    asset = env.scene[asset_cfg.name]
    deficit = torch.clamp(min_height - asset.data.root_pos_w[:, 2], min=0.0)
    return deficit


def zone_crossing_bonus(
    env: ManagerBasedRLEnv,
    zone_positions: tuple[float, ...],
    sigma: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense local bonus near obstacle-zone lines in x-direction."""
    asset = env.scene[asset_cfg.name]
    env_x = env.scene.env_origins[:, 0].unsqueeze(1)
    robot_x = asset.data.root_pos_w[:, 0].unsqueeze(1)
    thresholds = env_x + torch.tensor(zone_positions, device=robot_x.device).unsqueeze(0)
    return torch.exp(-((robot_x - thresholds) ** 2) / (2.0 * sigma**2)).sum(dim=1)
