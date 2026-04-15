#!/usr/bin/env python3
"""Recenter curated terrain segments and attach manual curriculum levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSETS_ENV_DIR = PROJECT_DIR / "source" / "myproject" / "myproject" / "assets" / "environments"


DEFAULT_LEVELS = {
    "a.usd": 2,
    "a_2.usd": 2,
    "b.usd": 1,
    "b_deep.usd": 1,
    "c.usd": 1,
    "d_0.usd": 0,
    "d_1.usd": 0,
    "d_2.usd": 0,
    "d_3.usd": 0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_DIR / "test_segment_filtered",
        help="Directory containing curated source USD patches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_ENV_DIR / "test_segment_filtered_recentered",
        help="Directory where recentered USDA files will be written.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Filename of the manifest written inside output-dir.",
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
        child_dst = dst_path + "/" + child.GetName()
        _copy_subtree(stage, child, child_dst, mapping)


def _apply_required_schemas(mesh_prim, physics_material_prim) -> None:
    from pxr import UsdPhysics, UsdShade, PhysxSchema

    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim)

    UsdPhysics.MaterialAPI.Apply(physics_material_prim)
    PhysxSchema.PhysxMaterialAPI.Apply(physics_material_prim)


def _compute_bounds(mesh_prim) -> dict[str, float]:
    points = mesh_prim.GetAttribute("points").Get() or []
    if not points:
        raise RuntimeError(f"Mesh prim has no points: {mesh_prim.GetPath()}")
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "z_min": min(zs),
        "z_max": max(zs),
        "x_center": 0.5 * (min(xs) + max(xs)),
        "y_center": 0.5 * (min(ys) + max(ys)),
        "point_count": len(points),
    }


def _shift_mesh(mesh_prim, x_shift: float, y_shift: float) -> None:
    points = mesh_prim.GetAttribute("points").Get() or []
    shifted_points = [(float(p[0]) + x_shift, float(p[1]) + y_shift, float(p[2])) for p in points]
    mesh_prim.GetAttribute("points").Set(shifted_points)

    extent_attr = mesh_prim.GetAttribute("extent")
    extent = extent_attr.Get()
    if extent:
        shifted_extent = [
            (float(extent[0][0]) + x_shift, float(extent[0][1]) + y_shift, float(extent[0][2])),
            (float(extent[1][0]) + x_shift, float(extent[1][1]) + y_shift, float(extent[1][2])),
        ]
        extent_attr.Set(shifted_extent)


def export_recentered(args: argparse.Namespace) -> dict:
    from pxr import Usd, UsdGeom

    src_files = sorted(args.source_dir.glob("*.usd"))
    if not src_files:
        raise RuntimeError(f"No USD files found in {args.source_dir}")

    missing_levels = [path.name for path in src_files if path.name not in DEFAULT_LEVELS]
    if missing_levels:
        raise RuntimeError(f"Missing manual level mapping for: {missing_levels}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "segment_count": len(src_files),
        "levels": {},
        "segments": [],
    }

    for path in src_files:
        src_stage = Usd.Stage.Open(str(path))
        if src_stage is None:
            raise RuntimeError(f"Failed to open source stage: {path}")

        default_root = src_stage.GetDefaultPrim()
        if not default_root:
            raise RuntimeError(f"Stage has no default prim: {path}")
        seg_roots = default_root.GetChildren()
        if len(seg_roots) != 1:
            raise RuntimeError(f"Expected one segment child under {default_root.GetPath()} in {path}")
        src_segment = seg_roots[0]
        src_mesh = src_stage.GetPrimAtPath(str(src_segment.GetPath()) + "/ground/terrain/mesh")
        if not src_mesh.IsValid():
            raise RuntimeError(f"Mesh not found in {path}")

        bounds = _compute_bounds(src_mesh)
        x_shift = -bounds["x_center"]
        y_shift = -bounds["y_center"]

        stem = path.stem
        out_path = args.output_dir / f"{stem}.usda"
        if out_path.exists():
            out_path.unlink()
        out_stage = Usd.Stage.CreateNew(str(out_path))
        out_stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
        out_stage.SetMetadata("metersPerUnit", 1.0)

        world = out_stage.DefinePrim("/World", "Xform")
        out_stage.SetDefaultPrim(world)

        mapping = {str(src_segment.GetPath()): "/World"}
        _copy_subtree(out_stage, src_segment, "/World", mapping)

        out_mesh = out_stage.GetPrimAtPath("/World/ground/terrain/mesh")
        if not out_mesh.IsValid():
            raise RuntimeError(f"Exported mesh missing in {out_path}")
        _shift_mesh(out_mesh, x_shift, y_shift)
        out_physics_material = out_stage.GetPrimAtPath("/World/ground/terrain/physicsMaterial")
        if not out_physics_material.IsValid():
            raise RuntimeError(f"Exported physics material missing in {out_path}")
        _apply_required_schemas(out_mesh, out_physics_material)

        out_stage.GetRootLayer().Save()

        manifest["segments"].append(
            {
                "source_file": path.name,
                "output_file": out_path.name,
                "level": DEFAULT_LEVELS[path.name],
                "prim_path": "/World/ground",
                "terrain_mesh_path": "/World/ground/terrain/mesh",
                "source_segment_prim": str(src_segment.GetPath()),
                "source_bounds": {
                    "x_min": bounds["x_min"],
                    "x_max": bounds["x_max"],
                    "y_min": bounds["y_min"],
                    "y_max": bounds["y_max"],
                    "z_min": bounds["z_min"],
                    "z_max": bounds["z_max"],
                },
                "recenter_shift": {
                    "x": x_shift,
                    "y": y_shift,
                    "z": 0.0,
                },
                "local_bounds_after_shift": {
                    "x_min": bounds["x_min"] + x_shift,
                    "x_max": bounds["x_max"] + x_shift,
                    "y_min": bounds["y_min"] + y_shift,
                    "y_max": bounds["y_max"] + y_shift,
                    "z_min": bounds["z_min"],
                    "z_max": bounds["z_max"],
                },
                "point_count": bounds["point_count"],
            }
        )

    level_groups: dict[str, list[str]] = {}
    for segment in manifest["segments"]:
        key = str(segment["level"])
        level_groups.setdefault(key, []).append(segment["output_file"])
    manifest["levels"] = dict(sorted(level_groups.items(), key=lambda item: int(item[0])))

    manifest_path = args.output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    simulation_app = SimulationApp({"headless": True})
    try:
        manifest = export_recentered(args)
        print(json.dumps({"segment_count": manifest["segment_count"], "output_dir": manifest["output_dir"]}, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
