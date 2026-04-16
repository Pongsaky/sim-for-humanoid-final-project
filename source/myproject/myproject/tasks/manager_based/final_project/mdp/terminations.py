from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> "torch.Tensor":
    """Terminate episode when robot base crosses the goal line in +x for each env."""
    asset = env.scene[asset_cfg.name]
    if start_x is None:
        goal_line_x = env.scene.env_origins[:, 0] + goal_x
    else:
        goal_line_x = asset.data.root_pos_w[:, 0].new_full((env.num_envs,), float(start_x + goal_x))
    return asset.data.root_pos_w[:, 0] >= goal_line_x
