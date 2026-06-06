from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "repro_outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def lawnmower_route(width_m: float, height_m: float, spacing_m: float) -> list[dict]:
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    rows = max(2, math.ceil(height_m / spacing_m) + 1)
    route = []
    for i in range(rows):
        y = min(i * spacing_m, height_m)
        if i % 2 == 0:
            route.append({"x": 0.0, "y": y})
            route.append({"x": width_m, "y": y})
        else:
            route.append({"x": width_m, "y": y})
            route.append({"x": 0.0, "y": y})
    return route


def estimate_spacing(flight_height_m: float, camera_fov_deg: float, side_overlap: float) -> float:
    footprint = 2 * flight_height_m * math.tan(math.radians(camera_fov_deg / 2))
    return footprint * (1 - side_overlap)


def route_length(route: list[dict]) -> float:
    total = 0.0
    for a, b in zip(route, route[1:]):
        total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    return total


def plot_route(width_m: float, height_m: float, route: list[dict], output_path: Path) -> None:
    xs = [p["x"] for p in route]
    ys = [p["y"] for p in route]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 7), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, ys, "-o", color="#0f766e", markersize=3, linewidth=1.8)
    ax.add_patch(plt.Rectangle((0, 0), width_m, height_m, fill=False, linewidth=2, edgecolor="#111827"))
    ax.scatter([0], [0], marker="*", s=160, color="#dc2626", label="home / takeoff")
    ax.scatter([xs[-1]], [ys[-1]], marker="s", s=70, color="#2563eb", label="route end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("UAV lawnmower route for corn tassel image acquisition")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="repro_outputs")
    parser.add_argument("--field-width-m", type=float, default=120.0)
    parser.add_argument("--field-height-m", type=float, default=80.0)
    parser.add_argument("--flight-height-m", type=float, default=25.0)
    parser.add_argument("--camera-fov-deg", type=float, default=72.0)
    parser.add_argument("--side-overlap", type=float, default=0.7)
    parser.add_argument("--speed-mps", type=float, default=4.0)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    output_dir = project_root / args.output_dir
    spacing = estimate_spacing(args.flight_height_m, args.camera_fov_deg, args.side_overlap)
    route = lawnmower_route(args.field_width_m, args.field_height_m, spacing)
    length = route_length(route)
    duration = length / args.speed_mps
    plan = {
        "field_width_m": args.field_width_m,
        "field_height_m": args.field_height_m,
        "flight_height_m": args.flight_height_m,
        "camera_fov_deg": args.camera_fov_deg,
        "side_overlap": args.side_overlap,
        "line_spacing_m": round(spacing, 3),
        "speed_mps": args.speed_mps,
        "waypoint_count": len(route),
        "route_length_m": round(length, 3),
        "estimated_duration_s": round(duration, 3),
        "route": route,
    }
    report_path = output_dir / "reports" / "uav_mission_plan.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_route(args.field_width_m, args.field_height_m, route, output_dir / "screenshots" / "uav_route_plan.png")
    print(json.dumps({k: v for k, v in plan.items() if k != "route"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
