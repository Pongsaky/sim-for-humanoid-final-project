from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
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


def goal_reached_upright(
    env: ManagerBasedRLEnv,
    goal_x: float,
    start_x: float | None = None,
    min_height: float = 0.55,
    min_upright: float = 0.75,
    contact_force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*torso_link"),
) -> torch.Tensor:
    """Terminate with success only when the robot crosses the goal line in a valid upright pose."""
    asset = env.scene[asset_cfg.name]
    if start_x is None:
        goal_line_x = env.scene.env_origins[:, 0] + goal_x
    else:
        goal_line_x = asset.data.root_pos_w[:, 0].new_full((env.num_envs,), float(start_x + goal_x))

    crossed_goal = asset.data.root_pos_w[:, 0] >= goal_line_x
    tall_enough = asset.data.root_pos_w[:, 2] >= min_height
    upright_enough = -asset.data.projected_gravity_b[:, 2] >= min_upright

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    torso_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    torso_contact = torch.any(torso_contact > contact_force_threshold, dim=1)

    return crossed_goal & tall_enough & upright_enough & ~torso_contact


def bad_orientation_safe(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when tilt exceeds the limit without using acos on noisy GPU values."""
    asset = env.scene[asset_cfg.name]
    upright_cos = torch.clamp(-asset.data.projected_gravity_b[:, 2], min=-1.0, max=1.0)
    min_allowed_cos = math.cos(limit_angle)
    return upright_cos < min_allowed_cos
