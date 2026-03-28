from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_velocity_toward_goal(
    env: ManagerBasedRLEnv,
    goal_x: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward world-frame velocity toward a goal line located at +x in each env frame."""
    asset = env.scene[asset_cfg.name]
    goal_dir_w = torch.zeros_like(asset.data.root_lin_vel_w[:, :2])
    goal_dir_w[:, 0] = 1.0
    return torch.sum(asset.data.root_lin_vel_w[:, :2] * goal_dir_w, dim=1)


def goal_distance_progress(
    env: ManagerBasedRLEnv,
    goal_x: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense progress reward from normalized x-position in the environment frame."""
    asset = env.scene[asset_cfg.name]
    env_x = env.scene.env_origins[:, 0]
    progress = (asset.data.root_pos_w[:, 0] - env_x) / max(goal_x, 1e-6)
    return torch.clamp(progress, min=0.0, max=1.0)


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    goal_x: float,
    bonus: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-step bonus whenever robot has crossed the goal x-line."""
    asset = env.scene[asset_cfg.name]
    env_x = env.scene.env_origins[:, 0]
    reached = asset.data.root_pos_w[:, 0] >= (env_x + goal_x)
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
