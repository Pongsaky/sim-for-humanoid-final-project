from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> "torch.Tensor":
    """Terminate episode when robot base crosses the goal line."""
    asset = env.scene[asset_cfg.name]
    env_x = env.scene.env_origins[:, 0]
    goal_x_w = env_x + goal_x
    if goal_x >= start_x:
        return asset.data.root_pos_w[:, 0] >= goal_x_w
    return asset.data.root_pos_w[:, 0] <= goal_x_w
