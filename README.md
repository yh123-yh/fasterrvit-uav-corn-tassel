# Faster R-ViT UAV Corn Tassel Detection Reproduction

本仓库整理了基于 Faster R-ViT/Faster R-CNN 思路的玉米雄穗检测复现实验，并在原检测流程基础上加入无人机巡检航线规划。仓库包含可运行源码、4GPU 10 epoch 复现实验记录、教师展示版 Markdown 报告、可视化图表和加入无人机规划内容的立项书版本。

## 项目内容

- 玉米雄穗 VOC 数据读取与标注统计
- Faster R-CNN/Faster R-ViT 风格检测模型代码修复与复现
- 4GPU 分布式短训练脚本，默认 10 epoch
- 训练日志、loss 曲线、检测结果可视化
- 无人机田块覆盖式航线规划
- 教师展示版报告与可视化总拼图

## 目录结构

```text
.
├── data/                         # VOC 数据集读取与预处理
├── model/                        # Faster R-CNN / ViT 相关模型代码
├── scripts/                      # 复现、训练、可视化和报告生成脚本
├── utils/                        # 配置、评估、可视化工具
├── VOCdevkit/                    # VOC 标注、ImageSets 和 VOC 工具代码，不包含原始 JPG 大图
├── repro_outputs/                # 已生成的报告、日志和展示图
├── requirements.txt
└── README.md
```

## 重要说明

为控制 GitHub 仓库体积，以下大文件未上传：

- 原始训练图像：`VOCdevkit/VOC2007/JPEGImages/`
- 训练 checkpoint：`repro_outputs/4gpu_train/checkpoints/fasterrcnn_4gpu_10epoch.pth`
- ViT 预训练权重：`imagenet21k+imagenet2012_ViT-B_16.pth`
- Python 虚拟环境：`.venv_fasterrvit/`

如果需要重新训练，需要将上述数据和权重按原路径放回项目目录。

## 环境创建

示例环境命令如下：

```bash
python -m venv .venv_fasterrvit
source .venv_fasterrvit/bin/activate
pip install -r requirements.txt
```

本次复现实验使用 PyTorch CUDA 环境，训练时检测到 4 张 `NVIDIA A100 80GB PCIe`。

## 运行流程

1. 检查 4GPU 环境：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_fasterrvit/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

2. 数据集检查与基础复现：

```bash
.venv_fasterrvit/bin/python scripts/reproduce_simple.py --output-dir repro_outputs
```

3. 4GPU 10 epoch 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv_fasterrvit/bin/python -m torch.distributed.run \
  --nproc_per_node=4 \
  scripts/train_4gpu_short.py \
  --epochs 10 \
  --output-dir repro_outputs/4gpu_train
```

4. 重新渲染检测样例：

```bash
CUDA_VISIBLE_DEVICES=0 .venv_fasterrvit/bin/python scripts/render_trained_detection.py \
  --train-dir repro_outputs/4gpu_train \
  --max-images 40
```

5. 生成无人机航线规划：

```bash
.venv_fasterrvit/bin/python scripts/plan_uav_mission.py --output-dir repro_outputs/4gpu_train
```

6. 生成展示图和教师报告：

```bash
.venv_fasterrvit/bin/python scripts/create_teacher_visuals.py --train-dir repro_outputs/4gpu_train
.venv_fasterrvit/bin/python scripts/create_teacher_report.py \
  --train-dir repro_outputs/4gpu_train \
  --output repro_outputs/reports/FasterRVIT_UAV_教师展示版复现报告.md
```

## 复现结果

本次 4GPU 短训练完成 10 epoch，主要结果如下：

| 指标 | 数值 |
|---|---:|
| GPU 数量 | 4 |
| 训练轮数 | 10 epoch |
| 初始 mean total loss | 3.242924 |
| 最终 mean total loss | 1.521377 |
| 评估图像数量 | 40 |
| mAP | 0.005682 |
| 训练耗时 | 140.906 秒 |

## 展示图

教师展示总拼图：

![教师展示总拼图](repro_outputs/4gpu_train/screenshots/teacher_display_montage.png)

9 张代表性标注样例：

![9宫格标注样例](repro_outputs/4gpu_train/screenshots/annotation_gallery_9.png)

无人机任务规划面板：

![无人机任务规划面板](repro_outputs/4gpu_train/screenshots/uav_mission_dashboard.png)

## 报告与文档

- 教师展示版报告：`repro_outputs/reports/FasterRVIT_UAV_教师展示版复现报告.md`
- 博客式复现记录：`repro_outputs/reports/FasterRVIT_UAV_复现与无人机规划博客.md`
- 加入无人机规划的立项书：`repro_outputs/reports/立项书_加入无人机航线规划版.docx`

## 输出文件来源

| 输出 | 生成脚本 |
|---|---|
| `dataset_stats.json` | `scripts/reproduce_simple.py` |
| `sample_gt_boxes.png` | `scripts/reproduce_simple.py` |
| `train_metrics.json` | `scripts/train_4gpu_short.py` |
| `loss_curve.png` | `scripts/train_4gpu_short.py` |
| `eval_detection_sample.png` | `scripts/render_trained_detection.py` |
| `uav_route_plan.png` | `scripts/plan_uav_mission.py` |
| `teacher_display_montage.png` | `scripts/create_teacher_visuals.py` |
| `FasterRVIT_UAV_教师展示版复现报告.md` | `scripts/create_teacher_report.py` |

## 备注

本仓库中的训练结果用于展示复现流程、4GPU 运行记录、检测推理流程和无人机规划扩展。由于训练轮数较短，检测精度不是最终优化结果；后续可在完整数据和更长训练周期下继续提升模型效果。
