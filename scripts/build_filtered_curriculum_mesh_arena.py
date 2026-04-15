#!/usr/bin/env python3
"""Build merged single-mesh curriculum arenas from curated recentered terrain patches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSETS_ENV_DIR = PROJECT_DIR / "source" / "myproject" / "myproject" / "assets" / "environments"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout-path",
        type=Path,
        required=True,
        help="Layout JSON produced by compose_filtered_curriculum_arena.py",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=ASSETS_ENV_DIR / "test_segment_filtered_recentered",
        help="Directory containing recentered patch USDA files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to write merged arena USDA.",
    )
    return parser


def _copy_authored_attributes(src_prim, dst_prim) -> None:
    for attr in src_prim.GetAttributes():
        if not attr.HasAuthoredValueOpinion():
            continue
        dst_attr = dst_prim.CreateAttribute(attr.GetName(), attr.GetTypeName(), attr.IsCustom())
        dst_attr.Set(attr.Get())


def _remap_targets(targets, mapping: dict[str, str]):
    remapped = []
    for target in targets:
        text = str(target)
        replaced = text
        for old, new in mapping.items():
            if text == old or text.startswith(old + "/"):
                replaced = new + text[len(old) :]
                break
        remapped.append(replaced)
    return remapped


def _copy_relationships(src_prim, dst_prim, mapping: dict[str, str]) -> None:
    for rel in src_prim.GetRelationships():
        dst_rel = dst_prim.CreateRelationship(rel.GetName(), custom=rel.IsCustom())
        targets = rel.GetTargets()
        if targets:
            dst_rel.SetTargets(_remap_targets(targets, mapping))


def _copy_subtree(stage, src_prim, dst_path: str, mapping: dict[str, str]) -> None:
    dst_prim = stage.DefinePrim(dst_path, src_prim.GetTypeName())
    _copy_authored_attributes(src_prim, dst_prim)
    _copy_relationships(src_prim, dst_prim, mapping)
    for child in src_prim.GetChildren():
        _copy_subtree(stage, child, dst_path + "/" + child.GetName(), mapping)


def _transform_point(point, rotate_z_deg: float, translate_x: float, translate_y: float):
    x = float(point[0])
    y = float(point[1])
    z = float(point[2])
    theta = math.radians(float(rotate_z_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xr = x * cos_t - y * sin_t
    yr = x * sin_t + y * cos_t
    return (xr + translate_x, yr + translate_y, z)


def _copy_material_binding(src_prim, dst_prim) -> None:
    from pxr import Sdf

    src_rel = src_prim.GetRelationship("material:binding")
    if not src_rel or not src_rel.GetTargets():
        return
    dst_rel = dst_prim.CreateRelationship("material:binding")
    dst_rel.SetTargets([Sdf.Path("/terrain/visualMaterial")])


def _author_physics_material(stage, src_material_prim, material_path: str):
    from pxr import UsdPhysics, UsdShade, PhysxSchema

    material = UsdShade.Material.Define(stage, material_path)
    material_prim = material.GetPrim()
    UsdPhysics.MaterialAPI.Apply(material_prim)
    PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    _copy_authored_attributes(src_material_prim, material_prim)
    return material


def _author_visual_material(stage, src_material_prim, material_path: str) -> None:
    src_shader = None
    for child in src_material_prim.GetChildren():
        if child.GetTypeName() == "Shader":
            src_shader = child
            break

    if src_shader is None:
        return

    from pxr import UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/" + src_shader.GetName())
    _copy_authored_attributes(src_shader, shader.GetPrim())

    for output in src_shader.GetOutputs():
        shader.CreateOutput(output.GetBaseName(), output.GetTypeName())
    for output in material.GetOutputs():
        pass
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")


def build_mesh_arena(args: argparse.Namespace) -> dict:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    layout = json.loads(args.layout_path.read_text())
    tiles = layout["tiles"]
    if not tiles:
        raise RuntimeError("Layout contains no tiles.")

    first_patch_path = args.patch_dir / tiles[0]["file"]
    first_stage = Usd.Stage.Open(str(first_patch_path))
    if first_stage is None:
        raise RuntimeError(f"Failed to open patch stage: {first_patch_path}")

    src_mesh = first_stage.GetPrimAtPath("/World/ground/terrain/mesh")
    src_visual_material = first_stage.GetPrimAtPath("/World/ground/terrain/visualMaterial")
    src_physics_material = first_stage.GetPrimAtPath("/World/ground/terrain/physicsMaterial")
    if not src_mesh.IsValid() or not src_visual_material.IsValid() or not src_physics_material.IsValid():
        raise RuntimeError(f"Unexpected patch structure in: {first_patch_path}")

    merged_points = []
    merged_face_counts = []
    merged_face_indices = []

    for tile in tiles:
        patch_path = args.patch_dir / tile["file"]
        patch_stage = Usd.Stage.Open(str(patch_path))
        patch_mesh = patch_stage.GetPrimAtPath("/World/ground/terrain/mesh")
        points = patch_mesh.GetAttribute("points").Get() or []
        face_counts = patch_mesh.GetAttribute("faceVertexCounts").Get() or []
        face_indices = patch_mesh.GetAttribute("faceVertexIndices").Get() or []
        base_idx = len(merged_points)

        for point in points:
            merged_points.append(
                _transform_point(point, tile.get("rotate_z_deg", 0.0), float(tile["x"]), float(tile["y"]))
            )

        merged_face_counts.extend(int(v) for v in face_counts)
        merged_face_indices.extend(base_idx + int(v) for v in face_indices)

    xs = [p[0] for p in merged_points]
    ys = [p[1] for p in merged_points]
    zs = [p[2] for p in merged_points]
    extent = [(min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))]

    if args.output_path.exists():
        args.output_path.unlink()
    stage = Usd.Stage.CreateNew(str(args.output_path))
    stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    terrain = stage.DefinePrim("/terrain", "Xform")
    stage.SetDefaultPrim(terrain)
    _copy_authored_attributes(first_stage.GetPrimAtPath("/World/ground/terrain"), terrain)

    out_mesh = UsdGeom.Mesh.Define(stage, "/terrain/mesh")
    out_mesh_prim = out_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(out_mesh_prim)
    PhysxSchema.PhysxCollisionAPI.Apply(out_mesh_prim)
    UsdShade.MaterialBindingAPI.Apply(out_mesh_prim)
    _copy_authored_attributes(src_mesh, out_mesh_prim)
    _copy_material_binding(src_mesh, out_mesh_prim)
    out_mesh.GetPointsAttr().Set(merged_points)
    out_mesh.GetFaceVertexCountsAttr().Set(merged_face_counts)
    out_mesh.GetFaceVertexIndicesAttr().Set(merged_face_indices)
    out_mesh.GetExtentAttr().Set(extent)

    _author_visual_material(stage, src_visual_material, "/terrain/visualMaterial")
    _author_physics_material(stage, src_physics_material, "/terrain/physicsMaterial")
    out_mesh_prim.GetRelationship("material:binding:physics").SetTargets([Sdf.Path("/terrain/physicsMaterial")])

    stage.GetRootLayer().Export(str(args.output_path))

    return {
        "output_path": str(args.output_path),
        "point_count": len(merged_points),
        "triangle_count": len(merged_face_counts),
        "extent": extent,
    }


def main() -> None:
    args = build_parser().parse_args()
    simulation_app = SimulationApp({"headless": True})
    try:
        result = build_mesh_arena(args)
        print(json.dumps(result, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
