from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_height_scan(env: ManagerBasedRLEnv, num_rays: int) -> torch.Tensor:
    """Return a zero height-scan vector to preserve observation size during map fine-tuning."""
    return torch.zeros((env.num_envs, num_rays), device=env.device, dtype=torch.float32)
