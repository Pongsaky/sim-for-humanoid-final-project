from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: ManagerBasedRLEnv,
    goal_x: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> "torch.Tensor":
    """Terminate episode when robot base crosses the goal line in +x for each env."""
    asset = env.scene[asset_cfg.name]
    env_x = env.scene.env_origins[:, 0]
    return asset.data.root_pos_w[:, 0] >= (env_x + goal_x)
