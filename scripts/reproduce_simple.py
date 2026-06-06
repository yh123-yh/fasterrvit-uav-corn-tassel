from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "repro_outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision

from data.dataset import Dataset
from data.util import read_image
from data.voc_dataset import VOC_BBOX_LABEL_NAMES
from model import FasterRCNNVGG16
from utils.config import opt
from utils.vis_tool import vis_bbox


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fasterrvit_repro")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)
    return logger


def read_ids(voc_dir: Path, split: str) -> list[str]:
    path = voc_dir / "ImageSets" / "Main" / f"{split}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_dataset_stats(voc_dir: Path) -> dict:
    annotation_dir = voc_dir / "Annotations"
    image_dir = voc_dir / "JPEGImages"
    label_counts: dict[str, int] = {}
    object_counts = []
    image_sizes = []
    missing_images = []
    for xml_path in sorted(annotation_dir.glob("*.xml"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename", default=f"{xml_path.stem}.jpg")
        if not (image_dir / filename).exists():
            missing_images.append(filename)
        size = root.find("size")
        if size is not None:
            image_sizes.append({
                "width": int(size.findtext("width", default="0")),
                "height": int(size.findtext("height", default="0")),
            })
        n_objects = 0
        for obj in root.findall("object"):
            label = obj.findtext("name", default="unknown").strip().lower()
            label_counts[label] = label_counts.get(label, 0) + 1
            n_objects += 1
        object_counts.append(n_objects)

    trainval = read_ids(voc_dir, "trainval")
    test = read_ids(voc_dir, "test")
    widths = [item["width"] for item in image_sizes]
    heights = [item["height"] for item in image_sizes]
    return {
        "voc_dir": str(voc_dir),
        "annotation_files": len(list(annotation_dir.glob("*.xml"))),
        "image_files": len(list(image_dir.glob("*.jpg"))),
        "trainval_count": len(trainval),
        "test_count": len(test),
        "label_names": list(VOC_BBOX_LABEL_NAMES),
        "label_counts": label_counts,
        "objects_total": int(sum(object_counts)),
        "objects_per_image_min": int(min(object_counts)) if object_counts else 0,
        "objects_per_image_mean": float(np.mean(object_counts)) if object_counts else 0.0,
        "objects_per_image_max": int(max(object_counts)) if object_counts else 0,
        "image_width_min": int(min(widths)) if widths else 0,
        "image_width_max": int(max(widths)) if widths else 0,
        "image_height_min": int(min(heights)) if heights else 0,
        "image_height_max": int(max(heights)) if heights else 0,
        "missing_images": missing_images,
    }


def load_raw_example(voc_dir: Path, sample_id: str):
    xml_path = voc_dir / "Annotations" / f"{sample_id}.xml"
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default=f"{sample_id}.jpg")
    img = read_image(str(voc_dir / "JPEGImages" / filename), color=True)
    bboxes = []
    labels = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        bboxes.append([
            int(box.findtext("ymin")) - 1,
            int(box.findtext("xmin")) - 1,
            int(box.findtext("ymax")) - 1,
            int(box.findtext("xmax")) - 1,
        ])
        labels.append(VOC_BBOX_LABEL_NAMES.index(obj.findtext("name").strip().lower()))
    return img, np.asarray(bboxes, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def save_sample_visual(voc_dir: Path, output_path: Path, sample_id: str) -> dict:
    img, bboxes, labels = load_raw_example(voc_dir, sample_id)
    fig = plt.figure(figsize=(12, 8), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    vis_bbox(img, bboxes, labels, ax=ax)
    ax.set_title(f"VOC sample {sample_id}: corn tassel GT boxes")
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return {"sample_id": sample_id, "gt_boxes": int(len(bboxes)), "path": str(output_path)}


def run_model_smoke(device: torch.device, logger: logging.Logger) -> dict:
    opt.min_size = 256
    opt.max_size = 256
    dataset = Dataset(opt)
    img, bbox, label, scale = dataset[0]
    logger.info("Loaded transformed sample: img=%s bbox=%s labels=%s scale=%.4f", img.shape, bbox.shape, label.tolist(), scale)

    model = FasterRCNNVGG16(n_fg_class=opt.n_fg_class).to(device)
    model.eval()
    tensor = torch.from_numpy(img[None]).float().to(device)
    with torch.no_grad():
        features = model.extractor(tensor)
        rpn_locs, rpn_scores, rois, roi_indices, anchors = model.rpn(features, tensor.shape[2:], scale)
    result = {
        "device": str(device),
        "input_shape": list(tensor.shape),
        "feature_shape": list(features.shape),
        "rpn_locs_shape": list(rpn_locs.shape),
        "rpn_scores_shape": list(rpn_scores.shape),
        "rois_shape": list(rois.shape),
        "roi_indices_shape": list(roi_indices.shape),
        "anchors_shape": list(anchors.shape),
        "n_fg_class": opt.n_fg_class,
    }
    logger.info("Model smoke result: %s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="repro_outputs")
    parser.add_argument("--sample-id", default=None)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    output_dir = project_root / args.output_dir
    logger = setup_logger(output_dir / "logs" / "run_reproduction.log")
    started = time.time()

    voc_dir = Path(opt.voc_data_dir)
    sample_id = args.sample_id or read_ids(voc_dir, "trainval")[0]
    logger.info("Project root: %s", project_root)
    logger.info("VOC dir: %s", voc_dir)
    logger.info("Sample id: %s", sample_id)

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
    }
    logger.info("Environment: %s", env)

    stats = collect_dataset_stats(voc_dir)
    stats_path = output_dir / "reports" / "dataset_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Dataset stats written to %s", stats_path)

    visual = save_sample_visual(voc_dir, output_dir / "screenshots" / "sample_gt_boxes.png", sample_id)
    logger.info("Sample visual written to %s", visual["path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = run_model_smoke(device, logger)

    summary = {
        "environment": env,
        "dataset_stats": stats,
        "sample_visual": visual,
        "model_smoke": smoke,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = output_dir / "reports" / "reproduction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)
    logger.info("Done in %.2fs", summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
