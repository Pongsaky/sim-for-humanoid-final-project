from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_root_state_from_spawn_grid(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_pos: tuple[float, float, float],
    base_rot: tuple[float, float, float, float],
    spacing_xy: tuple[float, float],
    num_cols: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robots onto deterministic slots on one shared map."""

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    env_ids_float = env_ids.to(dtype=torch.float32)

    num_cols_float = float(num_cols)
    row_ids = torch.floor(env_ids_float / num_cols_float)
    col_ids = torch.remainder(env_ids_float, num_cols_float)

    positions = torch.zeros((len(env_ids), 3), device=asset.device, dtype=torch.float32)
    positions[:, 0] = base_pos[0] + row_ids * spacing_xy[0]
    positions[:, 1] = base_pos[1] + col_ids * spacing_xy[1]
    positions[:, 2] = base_pos[2]

    orientations = torch.tensor(base_rot, device=asset.device, dtype=torch.float32).repeat(len(env_ids), 1)
    velocities = torch.zeros((len(env_ids), 6), device=asset.device, dtype=torch.float32)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
