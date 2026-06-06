from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "repro_outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch

from data.dataset import TestDataset, preprocess
from model import FasterRCNNVGG16
from utils.config import opt
from utils.vis_tool import vis_bbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="repro_outputs/4gpu_train")
    parser.add_argument("--max-images", type=int, default=40)
    args = parser.parse_args()

    train_dir = PROJECT_ROOT / args.train_dir
    metrics_path = train_dir / "reports" / "train_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    opt.min_size = 256
    opt.max_size = 256
    opt.n_fg_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FasterRCNNVGG16(n_fg_class=1).to(device)
    state = torch.load(train_dir / "checkpoints" / "fasterrcnn_4gpu_10epoch.pth", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    model.use_preset("evaluate")

    dataset = TestDataset(opt)
    best = None
    for idx in range(min(args.max_images, len(dataset))):
        raw_img, bbox, label, difficult = dataset.db.get_example(idx)
        img = preprocess(raw_img, opt.min_size, opt.max_size)
        pred_bbox, pred_label, pred_score = model.predict([img], [raw_img.shape[1:]])
        count = len(pred_bbox[0])
        max_score = float(pred_score[0].max()) if count else 0.0
        if best is None or count > best["count"] or max_score > best["max_score"]:
            best = {
                "idx": idx,
                "raw_img": raw_img,
                "pred_bbox": pred_bbox[0],
                "pred_label": pred_label[0],
                "pred_score": pred_score[0],
                "count": count,
                "max_score": max_score,
            }
        if count >= 5:
            break

    output_path = train_dir / "screenshots" / "eval_detection_sample.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = best["pred_score"].argsort()[::-1]
    keep = order[:5]
    boxes = best["pred_bbox"][keep]
    fig = plt.figure(figsize=(12, 8), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    vis_bbox(best["raw_img"], boxes, ax=ax)
    ax.set_title("Evaluation sample: predicted corn tassel boxes", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    metrics["detection_sample"] = {
        "sample_index": int(best["idx"]),
        "predicted_boxes_shown": int(len(boxes)),
        "total_predicted_boxes": int(best["count"]),
        "max_score": float(best["max_score"]),
    }
    metrics["eval_detection_sample"] = str(output_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["detection_sample"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
