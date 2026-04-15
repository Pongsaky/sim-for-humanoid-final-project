#!/usr/bin/env python3
"""Compose tiled curriculum arenas from curated recentered terrain patches."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from isaacsim import SimulationApp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSETS_ENV_DIR = PROJECT_DIR / "source" / "myproject" / "myproject" / "assets" / "environments"


PRESETS = {
    "stability": {
        "level_weights": {0: 1.0},
        "seed": 7,
        "gap": 0.0,
        "rotate_z_deg": 90.0,
    },
    "crossing": {
        "level_weights": {0: 0.5, 1: 0.35, 2: 0.15},
        "seed": 19,
        "gap": 0.5,
        "rotate_z_deg": 90.0,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ASSETS_ENV_DIR / "test_segment_filtered_recentered" / "manifest.json",
        help="Manifest produced by recenter_filtered_segments.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_ENV_DIR / "curriculum_arenas",
        help="Directory where composed arena USDs and layout manifests will be written.",
    )
    parser.add_argument("--rows", type=int, default=3, help="Number of patch rows along Y.")
    parser.add_argument("--cols", type=int, default=6, help="Number of patch columns along X.")
    parser.add_argument("--gap", type=float, default=0.5, help="Gap between neighboring patches in meters.")
    parser.add_argument(
        "--preset",
        choices=["stability", "crossing", "both"],
        default="both",
        help="Arena composition preset to generate.",
    )
    return parser


def _compute_tile_metrics(manifest: dict) -> tuple[float, float]:
    first = manifest["segments"][0]["local_bounds_after_shift"]
    width_x = float(first["x_max"]) - float(first["x_min"])
    width_y = float(first["y_max"]) - float(first["y_min"])
    if width_x <= 0.0 or width_y <= 0.0:
        raise RuntimeError("Invalid local bounds in manifest.")
    return width_x, width_y


def _build_level_pools(manifest: dict) -> dict[int, list[str]]:
    pools: dict[int, list[str]] = {}
    for key, files in manifest["levels"].items():
        pools[int(key)] = list(files)
    return pools


def _pick_tiles(level_pools: dict[int, list[str]], level_weights: dict[int, float], slots: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    levels = [level for level, weight in sorted(level_weights.items()) if weight > 0.0]
    weights = [level_weights[level] for level in levels]
    if not levels:
        raise RuntimeError("No positive level weights configured.")

    shuffled_by_level: dict[int, list[str]] = {}
    cursor_by_level: dict[int, int] = {}
    for level in levels:
        files = list(level_pools.get(level, []))
        if not files:
            raise RuntimeError(f"No files available for requested level {level}.")
        rng.shuffle(files)
        shuffled_by_level[level] = files
        cursor_by_level[level] = 0

    picks = []
    for slot_idx in range(slots):
        level = rng.choices(levels, weights=weights, k=1)[0]
        files = shuffled_by_level[level]
        cursor = cursor_by_level[level]
        picked = files[cursor % len(files)]
        cursor_by_level[level] = cursor + 1
        picks.append({"slot": slot_idx, "level": level, "file": picked})
    return picks


def _make_arena_stage(
    output_path: Path,
    patch_dir: Path,
    tile_width_x: float,
    tile_width_y: float,
    rows: int,
    cols: int,
    gap: float,
    rotate_z_deg: float,
    picks: list[dict],
) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom

    if output_path.exists():
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    ground = stage.DefinePrim("/World/ground", "Xform")

    step_x = tile_width_x + gap
    step_y = tile_width_y + gap
    half_cols = (cols - 1) / 2.0
    half_rows = (rows - 1) / 2.0

    layout_tiles = []
    for slot_idx, pick in enumerate(picks):
        row = slot_idx // cols
        col = slot_idx % cols
        x = (col - half_cols) * step_x
        y = (row - half_rows) * step_y
        tile_path = f"/World/ground/tile_r{row}_c{col}"
        tile_prim = stage.DefinePrim(tile_path, "Xform")
        xformable = UsdGeom.Xformable(tile_prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(rotate_z_deg)))

        ref_path = Path(pick["file"])
        asset_rel = os.path.relpath(patch_dir / ref_path, output_path.parent)
        tile_prim.GetReferences().AddReference(asset_rel, Sdf.Path("/World/ground"))

        layout_tiles.append(
            {
                "slot": slot_idx,
                "row": row,
                "col": col,
                "x": x,
                "y": y,
                "level": pick["level"],
                "file": pick["file"],
                "prim_path": tile_path,
                "rotate_z_deg": rotate_z_deg,
            }
        )

    stage.GetRootLayer().Save()

    left_edge = -0.5 * (cols * tile_width_x + (cols - 1) * gap)
    right_edge = 0.5 * (cols * tile_width_x + (cols - 1) * gap)
    lower_edge = -0.5 * (rows * tile_width_y + (rows - 1) * gap)
    upper_edge = 0.5 * (rows * tile_width_y + (rows - 1) * gap)

    return {
        "usd_path": str(output_path),
        "rows": rows,
        "cols": cols,
        "gap": gap,
        "rotate_z_deg": rotate_z_deg,
        "tile_width_x": tile_width_x,
        "tile_width_y": tile_width_y,
        "bounds": {
            "x_min": left_edge,
            "x_max": right_edge,
            "y_min": lower_edge,
            "y_max": upper_edge,
        },
        "suggested_start": {
            "x": left_edge + 1.5,
            "y": 0.0,
            "z": 1.05,
        },
        "suggested_goal_x": right_edge - 1.5,
        "tiles": layout_tiles,
    }


def compose(args: argparse.Namespace) -> list[Path]:
    manifest = json.loads(args.manifest_path.read_text())
    patch_dir = args.manifest_path.parent
    tile_width_x, tile_width_y = _compute_tile_metrics(manifest)
    level_pools = _build_level_pools(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preset_names = ["stability", "crossing"] if args.preset == "both" else [args.preset]
    produced = []

    for preset_name in preset_names:
        preset = PRESETS[preset_name]
        picks = _pick_tiles(
            level_pools=level_pools,
            level_weights=preset["level_weights"],
            slots=args.rows * args.cols,
            seed=preset["seed"],
        )
        usd_path = args.output_dir / f"{preset_name}_arena.usda"
        layout = _make_arena_stage(
            output_path=usd_path,
            patch_dir=patch_dir,
            tile_width_x=tile_width_x,
            tile_width_y=tile_width_y,
            rows=args.rows,
            cols=args.cols,
            gap=float(preset.get("gap", args.gap)),
            rotate_z_deg=float(preset.get("rotate_z_deg", 0.0)),
            picks=picks,
        )
        layout["preset"] = preset_name
        layout["level_weights"] = preset["level_weights"]
        layout_path = args.output_dir / f"{preset_name}_arena.layout.json"
        layout_path.write_text(json.dumps(layout, indent=2))
        produced.extend([usd_path, layout_path])

    return produced


def main() -> None:
    args = build_parser().parse_args()
    simulation_app = SimulationApp({"headless": True})
    try:
        produced = compose(args)
        print(json.dumps({"produced": [str(path) for path in produced]}, indent=2))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
