from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "repro_outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import gridspec
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from data.util import read_image
from utils.config import opt


PALETTE = {
    "blue": "#2563eb",
    "green": "#16a34a",
    "orange": "#f97316",
    "red": "#dc2626",
    "slate": "#334155",
    "gray": "#64748b",
    "light": "#f8fafc",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_annotation(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default=f"{xml_path.stem}.jpg")
    size = root.find("size")
    width = int(size.findtext("width", default="0")) if size is not None else 0
    height = int(size.findtext("height", default="0")) if size is not None else 0
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        boxes.append([
            int(box.findtext("ymin")) - 1,
            int(box.findtext("xmin")) - 1,
            int(box.findtext("ymax")) - 1,
            int(box.findtext("xmax")) - 1,
        ])
    return {
        "sample_id": xml_path.stem,
        "filename": filename,
        "width": width,
        "height": height,
        "boxes": np.asarray(boxes, dtype=np.float32),
        "count": len(boxes),
    }


def load_annotations(voc_dir: Path) -> list[dict]:
    def sort_key(path: Path):
        return int(path.stem) if path.stem.isdigit() else path.stem

    return [parse_annotation(path) for path in sorted((voc_dir / "Annotations").glob("*.xml"), key=sort_key)]


def choose_representative_samples(items: list[dict], n: int = 9) -> list[dict]:
    ordered = sorted(items, key=lambda item: item["count"])
    positions = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    selected = []
    seen = set()
    for pos in positions:
        item = ordered[int(pos)]
        if item["sample_id"] not in seen:
            selected.append(item)
            seen.add(item["sample_id"])
    for item in ordered:
        if len(selected) >= n:
            break
        if item["sample_id"] not in seen:
            selected.append(item)
            seen.add(item["sample_id"])
    return selected[:n]


def image_hwc(voc_dir: Path, item: dict) -> np.ndarray:
    img = read_image(str(voc_dir / "JPEGImages" / item["filename"]), color=True)
    return img.transpose(1, 2, 0).astype(np.uint8)


def draw_boxes(ax, boxes: np.ndarray, limit: int = 45, color: str = PALETTE["red"], linewidth: float = 1.2) -> None:
    for bb in boxes[:limit]:
        y_min, x_min, y_max, x_max = bb
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor=color, linewidth=linewidth))


def save_annotation_gallery(voc_dir: Path, items: list[dict], output_path: Path) -> None:
    selected = choose_representative_samples(items, 9)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=150)
    for ax, item in zip(axes.ravel(), selected):
        img = image_hwc(voc_dir, item)
        ax.imshow(img)
        draw_boxes(ax, item["boxes"], limit=50, linewidth=1.0)
        ax.text(
            0.02,
            0.96,
            f"ID {item['sample_id']} | GT {item['count']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#0f172a",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 3},
        )
        ax.axis("off")
    fig.suptitle("Corn tassel annotation gallery: 9 representative field images", fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output_path)
    plt.close(fig)


def flatten_box_stats(items: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    widths, heights, areas = [], [], []
    for item in items:
        boxes = item["boxes"]
        if len(boxes) == 0:
            continue
        box_w = boxes[:, 3] - boxes[:, 1]
        box_h = boxes[:, 2] - boxes[:, 0]
        widths.extend(box_w.tolist())
        heights.extend(box_h.tolist())
        areas.extend((box_w * box_h).tolist())
    return np.asarray(widths), np.asarray(heights), np.asarray(areas)


def save_dataset_dashboard(items: list[dict], stats: dict, output_path: Path) -> None:
    counts = np.asarray([item["count"] for item in items], dtype=np.float32)
    widths, heights, areas = flatten_box_stats(items)

    fig = plt.figure(figsize=(14, 8.5), dpi=150)
    spec = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.15, 1.15, 0.95], height_ratios=[1, 1])

    ax1 = fig.add_subplot(spec[0, 0])
    ax1.hist(counts, bins=18, color=PALETTE["blue"], alpha=0.85)
    ax1.axvline(counts.mean(), color=PALETTE["orange"], linewidth=2, label=f"mean {counts.mean():.1f}")
    ax1.set_title("Objects per image")
    ax1.set_xlabel("GT boxes")
    ax1.set_ylabel("image count")
    ax1.grid(axis="y", linestyle="--", alpha=0.28)
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.scatter(widths, heights, s=8, alpha=0.18, color=PALETTE["green"], edgecolors="none")
    ax2.set_title("Bounding box shape distribution")
    ax2.set_xlabel("box width / px")
    ax2.set_ylabel("box height / px")
    ax2.grid(True, linestyle="--", alpha=0.22)

    ax3 = fig.add_subplot(spec[1, 0])
    ax3.hist(np.sqrt(areas), bins=28, color=PALETTE["orange"], alpha=0.85)
    ax3.set_title("Box scale distribution")
    ax3.set_xlabel("sqrt(area) / px")
    ax3.set_ylabel("box count")
    ax3.grid(axis="y", linestyle="--", alpha=0.28)

    ax4 = fig.add_subplot(spec[1, 1])
    split_names = ["trainval", "test"]
    split_values = [stats["trainval_count"], stats["test_count"]]
    ax4.bar(split_names, split_values, color=[PALETTE["blue"], PALETTE["green"]], width=0.55)
    ax4.set_title("Dataset split")
    ax4.set_ylabel("images")
    ax4.grid(axis="y", linestyle="--", alpha=0.28)
    for idx, value in enumerate(split_values):
        ax4.text(idx, value + 3, str(value), ha="center", va="bottom", fontsize=11)

    ax5 = fig.add_subplot(spec[:, 2])
    ax5.axis("off")
    summary_lines = [
        ("Images", stats["image_files"]),
        ("Annotations", stats["annotation_files"]),
        ("GT boxes", stats["objects_total"]),
        ("Avg boxes/image", f"{stats['objects_per_image_mean']:.2f}"),
        ("Max boxes/image", stats["objects_per_image_max"]),
        ("Image width", f"{stats['image_width_min']} - {stats['image_width_max']}"),
        ("Image height", f"{stats['image_height_min']} - {stats['image_height_max']}"),
    ]
    y = 0.92
    ax5.text(0.0, y, "Dataset summary", fontsize=16, fontweight="bold", color="#0f172a")
    y -= 0.09
    for label, value in summary_lines:
        ax5.add_patch(Rectangle((0, y - 0.035), 0.98, 0.058, facecolor=PALETTE["light"], edgecolor="#e2e8f0", linewidth=0.8))
        ax5.text(0.04, y, label, ha="left", va="center", fontsize=10, color=PALETTE["gray"])
        ax5.text(0.94, y, str(value), ha="right", va="center", fontsize=11, color="#0f172a", fontweight="bold")
        y -= 0.078

    fig.suptitle("Dataset and annotation statistics", fontsize=17, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output_path)
    plt.close(fig)


def save_training_dashboard(metrics: dict, output_path: Path) -> None:
    train = metrics["train_metrics"]
    epochs = np.asarray([item["epoch"] for item in train])
    losses = np.asarray([item["mean_total_loss"] for item in train])
    first_loss = float(losses[0])
    last_loss = float(losses[-1])
    reduction = (first_loss - last_loss) / first_loss * 100.0

    fig = plt.figure(figsize=(14, 8.5), dpi=150)
    spec = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.35, 1.0, 0.95], height_ratios=[1.0, 0.9])

    ax1 = fig.add_subplot(spec[:, 0])
    ax1.plot(epochs, losses, marker="o", color=PALETTE["blue"], linewidth=2.6)
    ax1.fill_between(epochs, losses, losses.min() * 0.96, color=PALETTE["blue"], alpha=0.12)
    ax1.set_title("10 epoch training loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("mean total loss")
    ax1.grid(True, linestyle="--", alpha=0.28)
    for x, y in zip(epochs, losses):
        ax1.text(x, y + 0.035, f"{y:.2f}", ha="center", fontsize=8, color=PALETTE["slate"])

    ax2 = fig.add_subplot(spec[0, 1])
    bars = ax2.bar(["epoch 1", "epoch 10"], [first_loss, last_loss], color=[PALETTE["orange"], PALETTE["green"]], width=0.5)
    ax2.set_title("Loss comparison")
    ax2.set_ylabel("mean total loss")
    ax2.grid(axis="y", linestyle="--", alpha=0.28)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{bar.get_height():.3f}", ha="center", fontsize=10)

    ax3 = fig.add_subplot(spec[1, 1])
    ax3.axis("off")
    cards = [
        ("GPU world size", metrics["world_size"]),
        ("Training epochs", metrics["epochs"]),
        ("Loss reduction", f"{reduction:.1f}%"),
        ("mAP", f"{metrics['eval']['map']:.6f}"),
        ("Eval images", metrics["eval"]["eval_images"]),
        ("Elapsed", f"{metrics['elapsed_seconds']:.1f}s"),
    ]
    y = 0.88
    for label, value in cards:
        ax3.add_patch(Rectangle((0.03, y - 0.05), 0.92, 0.075, facecolor=PALETTE["light"], edgecolor="#e2e8f0", linewidth=0.8))
        ax3.text(0.08, y, label, ha="left", va="center", fontsize=10, color=PALETTE["gray"])
        ax3.text(0.90, y, str(value), ha="right", va="center", fontsize=11, color="#0f172a", fontweight="bold")
        y -= 0.13

    ax4 = fig.add_subplot(spec[:, 2])
    ax4.set_title("Distributed ranks")
    ax4.set_xlim(-0.5, 3.5)
    ax4.set_ylim(-0.5, 3.5)
    ax4.set_aspect("equal")
    ax4.axis("off")
    rank_positions = [(0, 2.2), (2, 2.2), (0, 0.6), (2, 0.6)]
    for rank, (x, y0) in enumerate(rank_positions):
        ax4.add_patch(Rectangle((x - 0.42, y0 - 0.35), 0.84, 0.7, facecolor="#dbeafe", edgecolor=PALETTE["blue"], linewidth=1.4))
        ax4.text(x, y0 + 0.08, f"rank {rank}", ha="center", va="center", fontsize=12, fontweight="bold", color="#1e3a8a")
        ax4.text(x, y0 - 0.16, "A100 80GB", ha="center", va="center", fontsize=8, color=PALETTE["slate"])
    ax4.plot([0, 2], [2.2, 2.2], color=PALETTE["gray"], linewidth=1.1)
    ax4.plot([0, 2], [0.6, 0.6], color=PALETTE["gray"], linewidth=1.1)
    ax4.plot([0, 0], [0.6, 2.2], color=PALETTE["gray"], linewidth=1.1)
    ax4.plot([2, 2], [0.6, 2.2], color=PALETTE["gray"], linewidth=1.1)

    fig.suptitle("4GPU training result dashboard", fontsize=17, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output_path)
    plt.close(fig)


def save_uav_dashboard(plan: dict, output_path: Path) -> None:
    route = np.asarray([[p["x"], p["y"]] for p in plan["route"]], dtype=np.float32)
    field_w = plan["field_width_m"]
    field_h = plan["field_height_m"]
    spacing = plan["line_spacing_m"]
    footprint = spacing / max(1.0 - plan["side_overlap"], 1e-6)

    fig = plt.figure(figsize=(14, 8.5), dpi=150)
    spec = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.35, 1.0, 0.95], height_ratios=[1.0, 0.9])

    ax1 = fig.add_subplot(spec[:, 0])
    ax1.add_patch(Rectangle((0, 0), field_w, field_h, facecolor="#ecfdf5", edgecolor="#166534", linewidth=1.5))
    y_values = np.unique(np.round(route[:, 1], 6))
    for y in y_values:
        ax1.add_patch(Rectangle((0, y - footprint / 2), field_w, footprint, facecolor="#bbf7d0", edgecolor="none", alpha=0.28))
    ax1.plot(route[:, 0], route[:, 1], color=PALETTE["blue"], linewidth=2.2, marker="o", markersize=4)
    for idx, (x, y) in enumerate(route):
        if idx in (0, len(route) - 1):
            ax1.text(x, y + 2.5, "START" if idx == 0 else "END", fontsize=9, color=PALETTE["slate"], ha="center")
    ax1.set_xlim(-5, field_w + 5)
    ax1.set_ylim(-5, field_h + 5)
    ax1.set_aspect("equal")
    ax1.set_title("Coverage route and camera swaths")
    ax1.set_xlabel("x / m")
    ax1.set_ylabel("y / m")
    ax1.grid(True, linestyle="--", alpha=0.22)

    ax2 = fig.add_subplot(spec[0, 1])
    waypoint_idx = np.arange(1, len(route) + 1)
    ax2.step(waypoint_idx, np.full_like(waypoint_idx, plan["flight_height_m"], dtype=np.float32), where="mid", color=PALETTE["green"], linewidth=2.2)
    ax2.scatter(waypoint_idx, np.full_like(waypoint_idx, plan["flight_height_m"], dtype=np.float32), color=PALETTE["green"], s=28)
    ax2.set_title("Waypoint altitude profile")
    ax2.set_xlabel("waypoint")
    ax2.set_ylabel("height / m")
    ax2.set_ylim(0, plan["flight_height_m"] * 1.45)
    ax2.grid(True, linestyle="--", alpha=0.25)

    ax3 = fig.add_subplot(spec[1, 1])
    ax3.axis("off")
    cards = [
        ("Field", f"{field_w:.0f}m x {field_h:.0f}m"),
        ("Flight height", f"{plan['flight_height_m']:.1f}m"),
        ("FOV", f"{plan['camera_fov_deg']:.1f} deg"),
        ("Line spacing", f"{spacing:.3f}m"),
        ("Waypoints", plan["waypoint_count"]),
        ("Route length", f"{plan['route_length_m']:.1f}m"),
        ("Duration", f"{plan['estimated_duration_s']:.1f}s"),
    ]
    y = 0.90
    for label, value in cards:
        ax3.text(0.03, y, label, ha="left", va="center", fontsize=10, color=PALETTE["gray"])
        ax3.text(0.94, y, str(value), ha="right", va="center", fontsize=11, color="#0f172a", fontweight="bold")
        ax3.plot([0.03, 0.94], [y - 0.045, y - 0.045], color="#e2e8f0", linewidth=0.8)
        y -= 0.12

    ax4 = fig.add_subplot(spec[:, 2])
    ax4.set_title("Mission workflow")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")
    steps = [
        ("Boundary", "field model"),
        ("Route", "lawnmower scan"),
        ("Image", "low-altitude capture"),
        ("Detect", "tassel boxes"),
        ("Action", "mapped task points"),
    ]
    y_positions = np.linspace(0.86, 0.16, len(steps))
    for idx, ((title, desc), y) in enumerate(zip(steps, y_positions)):
        ax4.add_patch(Circle((0.18, y), 0.055, facecolor="#dbeafe", edgecolor=PALETTE["blue"], linewidth=1.2))
        ax4.text(0.18, y, str(idx + 1), ha="center", va="center", fontsize=11, color="#1e3a8a", fontweight="bold")
        ax4.text(0.32, y + 0.018, title, ha="left", va="center", fontsize=11, color="#0f172a", fontweight="bold")
        ax4.text(0.32, y - 0.030, desc, ha="left", va="center", fontsize=9, color=PALETTE["gray"])
        if idx < len(steps) - 1:
            ax4.plot([0.18, 0.18], [y - 0.07, y_positions[idx + 1] + 0.07], color="#94a3b8", linewidth=1.0)

    fig.suptitle("UAV corn tassel inspection mission plan", fontsize=17, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output_path)
    plt.close(fig)


def save_montage(image_paths: list[tuple[str, Path]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=150)
    for ax, (title, path) in zip(axes.ravel(), image_paths):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")
    for ax in axes.ravel()[len(image_paths):]:
        ax.axis("off")
    fig.suptitle("Faster R-ViT + UAV project visual summary", fontsize=18, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="repro_outputs/4gpu_train")
    args = parser.parse_args()

    train_dir = PROJECT_ROOT / args.train_dir
    screenshots_dir = train_dir / "screenshots"
    reports_dir = train_dir / "reports"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    voc_dir = Path(opt.voc_data_dir)
    annotations = load_annotations(voc_dir)
    dataset_stats = load_json(PROJECT_ROOT / "repro_outputs" / "reports" / "dataset_stats.json")
    train_metrics = load_json(train_dir / "reports" / "train_metrics.json")
    uav_plan = load_json(train_dir / "reports" / "uav_mission_plan.json")

    outputs = {
        "annotation_gallery_9": screenshots_dir / "annotation_gallery_9.png",
        "dataset_dashboard": screenshots_dir / "dataset_dashboard.png",
        "training_dashboard": screenshots_dir / "training_dashboard.png",
        "uav_mission_dashboard": screenshots_dir / "uav_mission_dashboard.png",
        "teacher_display_montage": screenshots_dir / "teacher_display_montage.png",
    }
    save_annotation_gallery(voc_dir, annotations, outputs["annotation_gallery_9"])
    save_dataset_dashboard(annotations, dataset_stats, outputs["dataset_dashboard"])
    save_training_dashboard(train_metrics, outputs["training_dashboard"])
    save_uav_dashboard(uav_plan, outputs["uav_mission_dashboard"])
    save_montage(
        [
            ("Dataset dashboard", outputs["dataset_dashboard"]),
            ("9-sample annotation gallery", outputs["annotation_gallery_9"]),
            ("4GPU training dashboard", outputs["training_dashboard"]),
            ("Detection sample", train_dir / "screenshots" / "eval_detection_sample.png"),
            ("UAV mission dashboard", outputs["uav_mission_dashboard"]),
            ("UAV route plan", train_dir / "screenshots" / "uav_route_plan.png"),
        ],
        outputs["teacher_display_montage"],
    )

    visual_assets = {name: str(path) for name, path in outputs.items()}
    (reports_dir / "visual_assets.json").write_text(json.dumps(visual_assets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(visual_assets, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
