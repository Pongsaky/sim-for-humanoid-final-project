from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import MISSING

from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
from isaaclab.sim.utils import bind_visual_material, clone, create_prim, get_current_stage, select_usd_variants
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_usd_path_with_timeout

logger = logging.getLogger(__name__)


@clone
def spawn_from_usd_with_prim_path(
    prim_path: str,
    cfg: "UsdFileWithPrimPathCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD reference from a specific prim path inside the source USD."""

    if not check_usd_path_with_timeout(cfg.usd_path):
        raise FileNotFoundError(f"USD file not found at path: '{cfg.usd_path}'.")

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = create_prim(
            prim_path,
            prim_type="Xform",
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )
        success = prim.GetReferences().AddReference(cfg.usd_path, primPath=cfg.usd_prim_path)
        if not success:
            raise RuntimeError(
                f"Unable to add USD reference to '{prim_path}' from '{cfg.usd_path}' prim '{cfg.usd_prim_path}'."
            )
    else:
        logger.warning(f"A prim already exists at prim path: '{prim_path}'.")

    if cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    if cfg.spatial_tendons_props is not None:
        schemas.modify_spatial_tendon_properties(prim_path, cfg.spatial_tendons_props)
    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)

    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(prim_path, material_path, stage=stage)

    return stage.GetPrimAtPath(prim_path)


@configclass
class UsdFileWithPrimPathCfg(FileCfg):
    """Spawn a USD by referencing an explicit prim path inside the USD file."""

    func: Callable = spawn_from_usd_with_prim_path

    usd_path: str = MISSING
    usd_prim_path: str = MISSING
    variants: object | dict[str, str] | None = None
