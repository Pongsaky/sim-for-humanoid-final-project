#!/usr/bin/env python3
"""Extract clean Y-axis segments from test_map.usd into standalone USDA scenes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from isaacsim import SimulationApp


def _round6(value: float) -> float:
    return round(float(value), 6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-path", type=Path, required=True, help="Source USD terrain stage.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("/tmp/test_map_segment_report.json"),
        help="JSON report produced by analyze_test_map_segments.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where extracted USDA scenes will be written.",
    )
    parser.add_argument(
        "--segment-ids",
        type=int,
        nargs="*",
        default=None,
        help="Subset of segment ids to export. Default exports all candidate segments in the report.",
    )
    parser.add_argument(
        "--use-aligned-patches",
        action="store_true",
        help="Export using aligned_patches from the report instead of candidate_segments.",
    )
    parser.add_argument(
        "--patch-ids",
        type=int,
        nargs="*",
        default=None,
        help="Subset of aligned patch ids to export when --use-aligned-patches is set.",
    )
    parser.add_argument(
        "--mesh-path",
        default="/World/Terrain/terrain/mesh",
        help="Prim path of the source terrain mesh.",
    )
    parser.add_argument(
        "--recenter-y",
        action="store_true",
        default=False,
        help="Shift each segment so its center is near y=0 in the exported scene.",
    )
    parser.add_argument(
        "--keep-world-light",
        action="store_true",
        default=True,
        help="Copy the source dome light into the exported scenes.",
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


def _analyze_source_rows(points, corridor_x_min: float, corridor_x_max: float):
    xy_to_zs: dict[tuple[float, float], list[float]] = defaultdict(list)
    for point in points:
        x = _round6(point[0])
        y = _round6(point[1])
        z = float(point[2])
        if corridor_x_min - 1e-6 <= x <= corridor_x_max + 1e-6:
            xy_to_zs[(x, y)].append(z)
    return xy_to_zs


def _subset_mesh(points, face_counts, face_indices, y_min: float, y_max: float, x_min: float, x_max: float, recenter_y: bool):
    kept_triangles = []
    used_vertex_ids = set()
    tri_count = 0
    cursor = 0
    for count in face_counts:
        tri = face_indices[cursor : cursor + count]
        cursor += count
        if count != 3:
            continue
        verts = [points[i] for i in tri]
        if all(y_min - 1e-6 <= float(v[1]) <= y_max + 1e-6 and x_min - 1e-6 <= float(v[0]) <= x_max + 1e-6 for v in verts):
            kept_triangles.append(tuple(tri))
            used_vertex_ids.update(tri)
        tri_count += 1

    if not kept_triangles:
        raise RuntimeError(f"No triangles found in range y=[{y_min}, {y_max}]")

    center_y = 0.5 * (y_min + y_max)
    sorted_vertex_ids = sorted(used_vertex_ids)
    old_to_new = {old: new for new, old in enumerate(sorted_vertex_ids)}

    new_points = []
    for old in sorted_vertex_ids:
        point = points[old]
        x = float(point[0])
        y = float(point[1]) - center_y if recenter_y else float(point[1])
        z = float(point[2])
        new_points.append((x, y, z))

    new_face_counts = []
    new_face_indices = []
    for tri in kept_triangles:
        new_face_counts.append(3)
        new_face_indices.extend(old_to_new[i] for i in tri)

    return {
        "points": new_points,
        "face_counts": new_face_counts,
        "face_indices": new_face_indices,
        "center_y": center_y,
        "triangle_count": len(new_face_counts),
    }


def _make_visual_material(stage, material_path: str):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def export_segments(args: argparse.Namespace) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

    report = json.loads(args.report_path.read_text())
    if args.use_aligned_patches:
        selected_patch_ids = set(args.patch_ids) if args.patch_ids else None
        candidate_segments = [
            {
                "segment_id": patch["patch_id"],
                "y_min": patch.get("export_y_min", patch["y_min"]),
                "y_max": patch.get("export_y_max", patch["y_max"]),
                "nominal_y_min": patch["y_min"],
                "nominal_y_max": patch["y_max"],
                "length": patch.get("export_y_max", patch["y_max"]) - patch.get("export_y_min", patch["y_min"]),
                "row_count": patch["row_count"],
                "x_range": report["corridor_x_range"],
                "mean_row_z_span": patch["mean_row_z_span"],
                "max_row_z_span": patch["max_row_z_span"],
                "clean_fraction": patch["clean_fraction"],
                "usable": patch["usable"],
            }
            for patch in report.get("aligned_patches", [])
            if selected_patch_ids is None or patch["patch_id"] in selected_patch_ids
        ]
        if not candidate_segments:
            raise RuntimeError("No aligned patches selected for export.")
    else:
        selected_ids = set(args.segment_ids) if args.segment_ids else None
        candidate_segments = [
            segment
            for segment in report["candidate_segments"]
            if selected_ids is None or segment["segment_id"] in selected_ids
        ]
        if not candidate_segments:
            raise RuntimeError("No candidate segments selected for export.")

    stage = Usd.Stage.Open(str(args.usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open source stage: {args.usd_path}")

    mesh_prim = stage.GetPrimAtPath(args.mesh_path)
    terrain_prim = stage.GetPrimAtPath("/World/Terrain/terrain")
    light_prim = stage.GetPrimAtPath("/World/DomeLight")
    physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
    default_material_prim = stage.GetPrimAtPath("/physicsScene/defaultMaterial")
    physics_material_prim = stage.GetPrimAtPath("/World/Terrain/terrain/physicsMaterial")

    points = mesh_prim.GetAttribute("points").Get() or []
    face_counts = mesh_prim.GetAttribute("faceVertexCounts").Get() or []
    face_indices = mesh_prim.GetAttribute("faceVertexIndices").Get() or []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_usd": str(args.usd_path),
        "segment_count": len(candidate_segments),
        "segments": [],
    }

    path_mapping = {
        "/World/Terrain/terrain/visualMaterial": "/World/ground/terrain/visualMaterial",
        "/World/Terrain/terrain/physicsMaterial": "/World/ground/terrain/physicsMaterial",
        "/World/Terrain/terrain/mesh": "/World/ground/terrain/mesh",
        "/World/Terrain/terrain": "/World/ground/terrain",
        "/World/Terrain": "/World/ground",
        "/World/DomeLight": "/World/Light",
        "/physicsScene/defaultMaterial": "/physicsScene/defaultMaterial",
        "/physicsScene": "/physicsScene",
    }

    for ordinal, segment in enumerate(candidate_segments):
        subset = _subset_mesh(
            points=points,
            face_counts=face_counts,
            face_indices=face_indices,
            y_min=segment["y_min"],
            y_max=segment["y_max"],
            x_min=segment["x_range"][0],
            x_max=segment["x_range"][1],
            recenter_y=args.recenter_y,
        )

        slug = f"seg_{ordinal:02d}_id_{segment['segment_id']:02d}"
        out_path = args.output_dir / f"{slug}.usda"
        out_stage = Usd.Stage.CreateNew(str(out_path))
        out_stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
        out_stage.SetMetadata("metersPerUnit", 1.0)

        world = out_stage.DefinePrim("/World", "Xform")
        out_stage.SetDefaultPrim(world)
        ground = out_stage.DefinePrim("/World/ground", "Xform")
        terrain = out_stage.DefinePrim("/World/ground/terrain", "Xform")
        _copy_authored_attributes(terrain_prim, terrain)

        mesh = UsdGeom.Mesh.Define(out_stage, "/World/ground/terrain/mesh")
        _copy_authored_attributes(mesh_prim, mesh.GetPrim())
        mesh.GetPointsAttr().Set(subset["points"])
        mesh.GetFaceVertexCountsAttr().Set(subset["face_counts"])
        mesh.GetFaceVertexIndicesAttr().Set(subset["face_indices"])
        mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)
        mesh.CreateDoubleSidedAttr(False)

        visual_material = _make_visual_material(out_stage, "/World/ground/terrain/visualMaterial")

        physics_material = UsdShade.Material.Define(out_stage, "/World/ground/terrain/physicsMaterial")
        _copy_authored_attributes(physics_material_prim, physics_material.GetPrim())

        default_material = UsdShade.Material.Define(out_stage, "/physicsScene/defaultMaterial")
        _copy_authored_attributes(default_material_prim, default_material.GetPrim())

        physics_scene = UsdPhysics.Scene.Define(out_stage, "/physicsScene")
        _copy_authored_attributes(physics_scene_prim, physics_scene.GetPrim())
        _copy_relationships(physics_scene_prim, physics_scene.GetPrim(), path_mapping)

        if args.keep_world_light:
            light = UsdLux.DomeLight.Define(out_stage, "/World/Light")
            _copy_authored_attributes(light_prim, light.GetPrim())
            _copy_relationships(light_prim, light.GetPrim(), path_mapping)

        _copy_relationships(mesh_prim, mesh.GetPrim(), path_mapping)
        UsdShade.MaterialBindingAPI(mesh).Bind(visual_material)
        mesh.GetPrim().GetRelationship("material:binding:physics").SetTargets(
            [Sdf.Path("/World/ground/terrain/physicsMaterial")]
        )

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(mesh.GetPrim()).GetRange()
        mesh.GetExtentAttr().Set([bbox.GetMin(), bbox.GetMax()])

        out_stage.GetRootLayer().Save()

        manifest["segments"].append(
            {
                "segment_id": segment["segment_id"],
                "output": str(out_path),
                "y_min": segment["y_min"],
                "y_max": segment["y_max"],
                "nominal_y_min": segment.get("nominal_y_min", segment["y_min"]),
                "nominal_y_max": segment.get("nominal_y_max", segment["y_max"]),
                "center_y_removed": subset["center_y"] if args.recenter_y else 0.0,
                "point_count": len(subset["points"]),
                "triangle_count": subset["triangle_count"],
                "bounds_min": list(bbox.GetMin()),
                "bounds_max": list(bbox.GetMax()),
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    simulation_app = SimulationApp({"headless": True})
    try:
        manifest = export_segments(args)
        print(
            f"Exported {manifest['segment_count']} segment scenes to {args.output_dir}",
            flush=True,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
