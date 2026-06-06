from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "repro_outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.config import opt
from utils.eval_tool import eval_detection_voc
from utils.vis_tool import vis_bbox


class NullVisualizer:
    env = "offline"

    def __getattr__(self, _name):
        def no_op(*_args, **_kwargs):
            return None
        return no_op

    def state_dict(self):
        return {}


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_4gpu_short")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


def run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"command failed: {' '.join(cmd)}\n{exc}"


def average_gradients(model: torch.nn.Module, world_size: int) -> None:
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def reduce_epoch_stats(loss_sum: float, steps: int, device: torch.device) -> tuple[float, int]:
    tensor = torch.tensor([loss_sum, float(steps)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item()), int(tensor[1].item())


def evaluate(model: FasterRCNNVGG16, max_images: int) -> dict:
    dataset = TestDataset(opt)
    pred_bboxes, pred_labels, pred_scores = [], [], []
    gt_bboxes, gt_labels, gt_difficults = [], [], []
    count = min(max_images, len(dataset))
    model.eval()
    for idx in range(count):
        img, size, gt_bbox, gt_label, gt_difficult = dataset[idx]
        pred_bbox, pred_label, pred_score = model.predict([img], [size])
        pred_bboxes += pred_bbox
        pred_labels += pred_label
        pred_scores += pred_score
        gt_bboxes.append(gt_bbox)
        gt_labels.append(gt_label)
        gt_difficults.append(gt_difficult)
    result = eval_detection_voc(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults,
        use_07_metric=True,
    )
    return {
        "map": float(result["map"]),
        "ap": [None if np.isnan(v) else float(v) for v in result["ap"]],
        "eval_images": count,
    }


def save_loss_curve(metrics: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in metrics]
    losses = [item["mean_total_loss"] for item in metrics]
    fig = plt.figure(figsize=(9, 5.4), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, losses, marker="o", linewidth=2, color="#2563eb")
    ax.set_title("4GPU short training loss curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean total loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_detection_sample(model: FasterRCNNVGG16, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = TestDataset(opt)
    raw_img = dataset.db.get_example(0)[0]
    pred_bbox, pred_label, pred_score = model.predict([raw_img], [raw_img.shape[1:]], visualize=True)
    boxes = pred_bbox[0][:10]
    labels = pred_label[0][:10]
    scores = pred_score[0][:10]

    fig = plt.figure(figsize=(12, 8), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    vis_bbox(raw_img, boxes, labels, scores, ax=ax)
    ax.set_title("Evaluation sample: predicted corn tassel boxes")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return {"predicted_boxes_shown": int(len(boxes)), "max_score": float(scores.max()) if len(scores) else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output-dir", default="repro_outputs/4gpu_train")
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--eval-images", type=int, default=40)
    args = parser.parse_args()

    if args.epochs > 20:
        raise ValueError("--epochs must not exceed 20")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    output_dir = PROJECT_ROOT / args.output_dir
    log_path = output_dir / "logs" / "train_4gpu_10epoch.log"
    logger = setup_logger(log_path) if rank == 0 else None

    opt.min_size = args.min_size
    opt.max_size = args.max_size
    opt.num_workers = 0
    opt.test_num_workers = 0
    opt.n_fg_class = 1
    opt.device = "cuda"

    if rank == 0:
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        before = run_cmd(["nvidia-smi"])
        (output_dir / "logs" / "nvidia_smi_before.txt").write_text(before, encoding="utf-8")
        logger.info("4GPU training started")
        logger.info("world_size=%s epochs=%s min_size=%s max_size=%s", world_size, args.epochs, opt.min_size, opt.max_size)
        logger.info("torch=%s cuda_available=%s cuda_device_count=%s", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())

    dataset = Dataset(opt)
    indices = list(range(rank, len(dataset), world_size))
    model = FasterRCNNVGG16(n_fg_class=opt.n_fg_class).to(device)
    trainer = FasterRCNNTrainer(model).to(device)
    trainer.vis = NullVisualizer()

    if rank == 0:
        logger.info("dataset_size=%s per_rank_steps=%s", len(dataset), len(indices))
    logger_rank_path = output_dir / "logs" / f"rank_{rank}.log"
    logger_rank_path.parent.mkdir(parents=True, exist_ok=True)
    logger_rank_path.write_text(
        f"rank={rank}\nlocal_rank={local_rank}\nworld_size={world_size}\ncuda_device={torch.cuda.get_device_name(local_rank)}\nsteps_per_epoch={len(indices)}\n",
        encoding="utf-8",
    )

    metrics = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        trainer.reset_meters()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for idx in indices:
            img, bbox, label, scale = dataset[idx]
            img_t = torch.from_numpy(img[None]).float().to(device)
            bbox_t = torch.from_numpy(bbox[None]).to(device)
            label_t = torch.from_numpy(label[None]).to(device)
            trainer.optimizer.zero_grad()
            losses = trainer.forward(img_t, bbox_t, label_t, float(scale))
            losses.total_loss.backward()
            average_gradients(trainer.faster_rcnn, world_size)
            trainer.optimizer.step()
            trainer.update_meters(losses)
            epoch_loss_sum += float(losses.total_loss.detach().cpu())
            epoch_steps += 1

        reduced_loss_sum, reduced_steps = reduce_epoch_stats(epoch_loss_sum, epoch_steps, device)
        mean_total_loss = reduced_loss_sum / max(reduced_steps, 1)
        if rank == 0:
            item = {
                "epoch": epoch,
                "mean_total_loss": mean_total_loss,
                "global_steps": reduced_steps,
                "lr": float(trainer.faster_rcnn.optimizer.param_groups[0]["lr"]),
            }
            metrics.append(item)
            logger.info("epoch=%s mean_total_loss=%.6f global_steps=%s", epoch, mean_total_loss, reduced_steps)
        dist.barrier()

    if rank == 0:
        eval_result = evaluate(trainer.faster_rcnn, args.eval_images)
        ckpt_path = output_dir / "checkpoints" / "fasterrcnn_4gpu_10epoch.pth"
        torch.save(
            {
                "model": trainer.faster_rcnn.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "metrics": metrics,
                "eval": eval_result,
                "config": opt._state_dict(),
            },
            ckpt_path,
        )
        loss_curve_path = output_dir / "screenshots" / "loss_curve.png"
        save_loss_curve(metrics, loss_curve_path)
        sample_path = output_dir / "screenshots" / "eval_detection_sample.png"
        sample_result = save_detection_sample(trainer.faster_rcnn, sample_path)
        after = run_cmd(["nvidia-smi"])
        (output_dir / "logs" / "nvidia_smi_after.txt").write_text(after, encoding="utf-8")

        summary = {
            "epochs": args.epochs,
            "world_size": world_size,
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "train_metrics": metrics,
            "eval": eval_result,
            "detection_sample": sample_result,
            "checkpoint": str(ckpt_path),
            "loss_curve": str(loss_curve_path),
            "eval_detection_sample": str(sample_path),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        metrics_path = output_dir / "reports" / "train_metrics.json"
        metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("eval=%s", eval_result)
        logger.info("checkpoint=%s", ckpt_path)
        logger.info("metrics=%s", metrics_path)
        logger.info("done elapsed_seconds=%.3f", summary["elapsed_seconds"])

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
