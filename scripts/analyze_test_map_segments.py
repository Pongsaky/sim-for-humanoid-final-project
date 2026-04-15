#!/usr/bin/env python3
"""Analyze test_map.usd and report candidate terrain segments along the Y axis.

This script is intentionally read-only. It inspects the terrain mesh inside
``test_map.usd`` and finds contiguous Y-bands that look regular enough to be
split into standalone scenes later.

Run it from the Isaac Sim / IsaacLab Python environment, for example:

    conda activate <isaaclab-env>
    python myproject/scripts/analyze_test_map_segments.py --usd-path test_map.usd
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from isaacsim import SimulationApp


def _round6(value: float) -> float:
    return round(float(value), 6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-path", type=Path, required=True, help="Path to the source terrain USD.")
    parser.add_argument(
        "--mesh-path",
        default="/World/Terrain/terrain/mesh",
        help="Prim path of the terrain mesh inside the stage.",
    )
    parser.add_argument(
        "--corridor-x-min",
        type=float,
        default=-4.0,
        help="Expected lower X bound for the main walkable corridor.",
    )
    parser.add_argument(
        "--corridor-x-max",
        type=float,
        default=4.0,
        help="Expected upper X bound for the main walkable corridor.",
    )
    parser.add_argument(
        "--nominal-row-width",
        type=int,
        default=81,
        help="Expected number of X samples in a clean corridor row.",
    )
    parser.add_argument(
        "--row-step",
        type=float,
        default=0.1,
        help="Nominal spacing between consecutive Y rows.",
    )
    parser.add_argument(
        "--row-step-tol",
        type=float,
        default=0.02,
        help="Tolerance used when deciding whether two Y rows are contiguous.",
    )
    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=6.0,
        help="Discard contiguous row groups shorter than this Y length.",
    )
    parser.add_argument(
        "--aligned-y-min",
        type=float,
        default=-120.0,
        help="Lower Y bound of the user-defined patch grid.",
    )
    parser.add_argument(
        "--aligned-y-max",
        type=float,
        default=120.0,
        help="Upper Y bound of the user-defined patch grid.",
    )
    parser.add_argument(
        "--patch-length",
        type=float,
        default=8.0,
        help="Patch length along Y for the user-defined grid.",
    )
    parser.add_argument(
        "--edge-trim",
        type=float,
        default=0.0,
        help="Trim this much from each patch edge when suggesting extraction bounds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/test_map_segment_report.json"),
        help="Where to write the JSON report.",
    )
    return parser


def analyze(args: argparse.Namespace) -> dict:
    from pxr import Usd

    stage = Usd.Stage.Open(str(args.usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {args.usd_path}")

    mesh = stage.GetPrimAtPath(args.mesh_path)
    if not mesh.IsValid():
        raise RuntimeError(f"Mesh prim not found: {args.mesh_path}")

    points = mesh.GetAttribute("points").Get() or []
    if not points:
        raise RuntimeError("Mesh has no points.")

    xy_to_zs: dict[tuple[float, float], list[float]] = defaultdict(list)
    for point in points:
        x = _round6(point[0])
        y = _round6(point[1])
        z = float(point[2])
        if args.corridor_x_min - 1e-6 <= x <= args.corridor_x_max + 1e-6:
            xy_to_zs[(x, y)].append(z)

    rows: dict[float, list[tuple[float, list[float]]]] = defaultdict(list)
    for (x, y), zs in xy_to_zs.items():
        rows[y].append((x, zs))

    ys = sorted(rows)
    if not ys:
        raise RuntimeError("No rows found in the requested corridor range.")

    row_infos = []
    for y in ys:
        xs = sorted(x for x, _ in rows[y])
        z_values = [zs[0] for _, zs in rows[y] if zs]
        multi_layer = sum(1 for _, zs in rows[y] if len(zs) > 1)
        row_infos.append(
            {
                "y": y,
                "x_count": len(xs),
                "x_min": xs[0],
                "x_max": xs[-1],
                "multi_layer_xy": multi_layer,
                "z_min": min(z_values),
                "z_max": max(z_values),
                "z_span": max(z_values) - min(z_values),
                "clean": (
                    len(xs) == args.nominal_row_width
                    and abs(xs[0] - args.corridor_x_min) < 1e-6
                    and abs(xs[-1] - args.corridor_x_max) < 1e-6
                    and multi_layer == 0
                ),
            }
        )

    contiguous_segments = []
    current = []
    nominal = args.row_step
    tol = args.row_step_tol

    for row in row_infos:
        if not row["clean"]:
            if current:
                contiguous_segments.append(current)
                current = []
            continue

        if not current:
            current = [row]
            continue

        step = row["y"] - current[-1]["y"]
        if abs(step - nominal) <= tol:
            current.append(row)
        else:
            contiguous_segments.append(current)
            current = [row]

    if current:
        contiguous_segments.append(current)

    candidate_segments = []
    for idx, segment in enumerate(contiguous_segments):
        y_min = segment[0]["y"]
        y_max = segment[-1]["y"]
        length = y_max - y_min
        if length < args.min_segment_length:
            continue
        z_spans = [row["z_span"] for row in segment]
        candidate_segments.append(
            {
                "segment_id": idx,
                "y_min": y_min,
                "y_max": y_max,
                "length": length,
                "row_count": len(segment),
                "x_range": [segment[0]["x_min"], segment[0]["x_max"]],
                "mean_row_z_span": sum(z_spans) / len(z_spans),
                "max_row_z_span": max(z_spans),
            }
        )

    row_width_counts = Counter(row["x_count"] for row in row_infos)
    multi_layer_rows = sum(1 for row in row_infos if row["multi_layer_xy"] > 0)
    clean_rows = sum(1 for row in row_infos if row["clean"])

    aligned_patches = []
    patch_idx = 0
    y0 = args.aligned_y_min
    while y0 < args.aligned_y_max - 1e-9:
        y1 = min(y0 + args.patch_length, args.aligned_y_max)
        patch_rows = [row for row in row_infos if y0 - 1e-6 <= row["y"] <= y1 + 1e-6]
        clean_patch_rows = [row for row in patch_rows if row["clean"]]
        if patch_rows:
            z_spans = [row["z_span"] for row in patch_rows]
            aligned_patches.append(
                {
                    "patch_id": patch_idx,
                    "y_min": y0,
                    "y_max": y1,
                    "trimmed_y_min": y0 + args.edge_trim,
                    "trimmed_y_max": y1 - args.edge_trim,
                    "export_y_min": y0,
                    "export_y_max": y1,
                    "row_count": len(patch_rows),
                    "clean_row_count": len(clean_patch_rows),
                    "clean_fraction": len(clean_patch_rows) / len(patch_rows),
                    "mean_row_z_span": sum(z_spans) / len(z_spans),
                    "max_row_z_span": max(z_spans),
                    "usable": len(clean_patch_rows) >= max(1, int(0.75 * len(patch_rows))),
                }
            )
        patch_idx += 1
        y0 = y1

    report = {
        "usd_path": str(args.usd_path),
        "mesh_path": args.mesh_path,
        "corridor_x_range": [args.corridor_x_min, args.corridor_x_max],
        "nominal_row_width": args.nominal_row_width,
        "row_step": args.row_step,
        "row_step_tol": args.row_step_tol,
        "min_segment_length": args.min_segment_length,
        "summary": {
            "corridor_row_count": len(row_infos),
            "clean_row_count": clean_rows,
            "multi_layer_row_count": multi_layer_rows,
            "row_width_counts": row_width_counts.most_common(10),
            "candidate_segment_count": len(candidate_segments),
            "aligned_patch_count": len(aligned_patches),
            "usable_aligned_patch_count": sum(1 for patch in aligned_patches if patch["usable"]),
        },
        "candidate_segments": candidate_segments,
        "aligned_patches": aligned_patches,
        "sample_rows": row_infos[:8] + row_infos[-8:],
    }
    return report


def main() -> None:
    args = build_parser().parse_args()
    simulation_app = SimulationApp({"headless": True})
    try:
        report = analyze(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.out}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
