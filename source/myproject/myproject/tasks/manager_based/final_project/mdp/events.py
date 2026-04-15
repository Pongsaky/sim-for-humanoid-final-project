from __future__ import annotations

import math
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
    if num_cols < 1:
        raise ValueError("reset_root_state_from_spawn_grid requires num_cols >= 1.")
    if spacing_xy[0] <= 0.0 or spacing_xy[1] <= 0.0:
        raise ValueError("reset_root_state_from_spawn_grid requires strictly positive spacing on both axes.")

    env_ids_float = env_ids.to(dtype=torch.float32)
    num_rows = math.ceil(env.scene.num_envs / num_cols)

    num_cols_float = float(num_cols)
    num_rows_float = float(num_rows)
    row_ids = torch.floor(env_ids_float / num_cols_float)
    col_ids = torch.remainder(env_ids_float, num_cols_float)

    positions = torch.zeros((len(env_ids), 3), device=asset.device, dtype=torch.float32)
    positions[:, 0] = base_pos[0] + (row_ids - (num_rows_float - 1.0) / 2.0) * spacing_xy[0]
    positions[:, 1] = base_pos[1] + (col_ids - (num_cols_float - 1.0) / 2.0) * spacing_xy[1]
    positions[:, 2] = base_pos[2]

    orientations = torch.tensor(base_rot, device=asset.device, dtype=torch.float32).repeat(len(env_ids), 1)
    velocities = torch.zeros((len(env_ids), 6), device=asset.device, dtype=torch.float32)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_near_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_pos: tuple[float, float, float],
    base_rot: tuple[float, float, float, float],
    xy_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robots near a shared start pose with uniform XY jitter."""

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if xy_range[0] < 0.0 or xy_range[1] < 0.0:
        raise ValueError("reset_root_state_near_start requires non-negative XY jitter ranges.")

    positions = torch.zeros((len(env_ids), 3), device=asset.device, dtype=torch.float32)
    positions[:] = env.scene.env_origins[env_ids]
    positions[:, 0] += base_pos[0]
    positions[:, 1] += base_pos[1]
    positions[:, 2] += base_pos[2]

    if xy_range[0] > 0.0:
        positions[:, 0] += torch.empty(len(env_ids), device=asset.device, dtype=torch.float32).uniform_(
            -xy_range[0], xy_range[0]
        )
    if xy_range[1] > 0.0:
        positions[:, 1] += torch.empty(len(env_ids), device=asset.device, dtype=torch.float32).uniform_(
            -xy_range[1], xy_range[1]
        )

    orientations = torch.tensor(base_rot, device=asset.device, dtype=torch.float32).repeat(len(env_ids), 1)
    velocities = torch.zeros((len(env_ids), 6), device=asset.device, dtype=torch.float32)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_on_shared_map(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_pos: tuple[float, float, float],
    base_rot: tuple[float, float, float, float],
    xy_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robots on a shared map using absolute world coordinates.

    Shared-map tasks use one arena for every replicated env, so adding `env_origins`
    can move some robots off the valid ground region.
    """

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if xy_range[0] < 0.0 or xy_range[1] < 0.0:
        raise ValueError("reset_root_state_on_shared_map requires non-negative XY jitter ranges.")

    positions = torch.zeros((len(env_ids), 3), device=asset.device, dtype=torch.float32)
    positions[:, 0] = base_pos[0]
    positions[:, 1] = base_pos[1]
    positions[:, 2] = base_pos[2]

    if xy_range[0] > 0.0:
        positions[:, 0] += torch.empty(len(env_ids), device=asset.device, dtype=torch.float32).uniform_(
            -xy_range[0], xy_range[0]
        )
    if xy_range[1] > 0.0:
        positions[:, 1] += torch.empty(len(env_ids), device=asset.device, dtype=torch.float32).uniform_(
            -xy_range[1], xy_range[1]
        )

    orientations = torch.tensor(base_rot, device=asset.device, dtype=torch.float32).repeat(len(env_ids), 1)
    velocities = torch.zeros((len(env_ids), 6), device=asset.device, dtype=torch.float32)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
