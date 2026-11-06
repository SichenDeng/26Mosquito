# 26Mosquito — 项目记忆

> 这个文件是跨聊天、跨设备的共享记忆。Claude Code 每次开新对话会自动读取它。
> 每次有重要进展、决定或踩坑后更新这里，然后 `git commit` + `git push`，其他设备 `git pull` 即可同步。

## 项目概况
- JHU Mechanical Engineering Senior Design（2026），仓库：https://github.com/SichenDeng/26Mosquito
- 目标：用于 mosquito **mark–release–recapture (MRR)** 的自动捕捉 + 标记装置
- Sichen 负责 **imaging + computer vision**（`Sichen/` 是个人工作目录）；其他成员分工待补充
- 相机：Raspberry Pi 相机拍照 → 传到电脑**离线**处理（不需要实时推理，也不需要部署到 Pi）
- 完整 imaging pipeline 最终包括：检测/计数、mosquito vs non-mosquito、marked vs unmarked、species

## 当前阶段：species classification（二分类）
- 类别：*Aedes albopictus*（Asian tiger mosquito）vs *Aedes aegypti*（yellow fever mosquito）
- 输入：一张只含单只蚊子的图（或裁剪后的单只图）；输出：物种 + 分数
- 先用公开数据训练 baseline，先不考虑标记粉；等装置有照片后再测真实表现，必要时用自己的数据 fine-tune
- 原则：**公开数据上的表现 ≠ 装置上的表现**，要分开报告；softmax 分数不能当作"未知物种"检测
- 用户偏好：一步一步带着做、讲清楚原理，不要一次生成一整套复杂代码
- 设备：Mac（M3）用来学习/整理数据/调试；Windows 笔记本（RTX 4070 Laptop）用来训练

## 数据集调研结论（2026-11-03）
- **首选：Ong et al. 2021 Sci Rep（Kaggle `pradeepisawasan/aedes-mosquitos`）**。用 Raspberry Pi NoIR v2 相机拍摄，距离 2.5 cm，白色背景，LED 照明，雌蚊 >12 天龄。CC BY-NC-SA 4.0，总共 20.4 GB，4,803 个 jpg（单张约 4 MB）
  - 结构：`{sp}/{sp}/{sp}_{0,90,180,270}[_testing]/{sp}{N}{a-d}.jpg`。90/180/270 文件夹是同一张图的旋转副本
  - 实际独立图片：每个物种约 150 个编号 N × 4 个视角（a–d），约 600 张，两类合计约 1,200 张
  - ⚠️ 官方的 testing 文件夹会泄漏：同一个 N 的其他视角在 train 里，而且不同旋转文件夹的划分不一致。**不要用官方划分**，只用 `_0` 文件夹，按 N 分组重新划分
  - 看图结果（2026-11-03，aegypti 的 12/57/100/101 号）：同一个 N 的 a–d 很可能是同一只死蚊子，被翻动或旋转后拍的（腿的形态、碎屑位置都对得上），所以**必须按 N 分组划分**
  - 图像特点：3280×2464 分辨率；背景偏蓝紫色、有横向条纹（NoIR 相机的色偏）；蚊子大致在画面中央，但只占宽度的约 15–20%。原图下胸部白色鳞片清晰；如果整张图直接缩到 224，细节会丢失，**必须先裁剪蚊子所在区域**
  - 下载脚本：`Sichen/species_classification/scripts/download_ong2021.py`（Kaggle 会限流，脚本用 2 个线程并带重试）；数据存放在 `Sichen/species_classification/data/raw/ong2021/`（已 gitignore）
- 备用的外部测试集（domain shift 检查）：Ong & Ahmad 2022 Sci Data，"Mosquito-on-human-skin"（Mendeley `zw4p9kj6nt/2`，CC BY 4.0），每类 250 张 landing + 250 张 smashed，拍的是人手皮肤背景
- 不采用：Mosquito Alert 2023（训练集里 aegypti 只有 38 张）；AMID v1（物种和数据来源高度相关，而且其中 WHO 部分就是上面的 skin 数据集，合并会重复）；Pise 2022（没有 albopictus）
- Baseline 模型计划：torchvision ResNet-50（ImageNet 预训练），transfer learning，用纯 PyTorch 训练循环来学原理；YOLO-cls 之后作为可选对比

## 目录结构
- `Sichen/yolo_learning/` — 学习 YOLO（ultralytics）的练习目录，使用本地 `.venv`（Python 3.9，含 torch、ultralytics）

## 环境配置
- Python 虚拟环境：`Sichen/yolo_learning/.venv`（不提交到 git，换设备需重建）
- 重建方式：`python3 -m venv .venv && source .venv/bin/activate && pip install ultralytics`
- Windows（RTX 4070）训练环境：
  1. `cd Sichen\species_classification`，`python -m venv .venv`，`.venv\Scripts\activate`
  2. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`，`pip install -r requirements.txt`
  3. 检查：`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  4. 把 Mac 上的 `data/processed/ong2021_800/`（108 MB，gitignore 了）手动拷到同一路径；`splits.csv` 通过 git 同步
  5. `python train.py`；如果报 CUDA out of memory，就改用 `python train.py --batch-size 8`

## 已做的决定
<!-- 格式：- YYYY-MM-DD：决定内容（原因） -->

## 当前进度 / 下一步
<!-- 格式：- [ ] 待办 / - [x] 已完成 -->
- [x] 数据集调研，选定 Ong 2021
- [x] 下载 `_0` 和 `_0_testing` 四个文件夹，共 1,203 张，全部能正常打开，尺寸都是 3280×2464；albopictus 看图后也确认同编号 = 同一只蚊子
- [x] 按编号分组划分：`scripts/make_splits.py` → `data/splits.csv`（seed=42，各物种分层，比例 70/15/15）
  - 最终 1,200 张 / 301 只蚊子：train 211 只（840 张），val 45 只（180 张），test 45 只（180 张）
  - 跳过了 3 个 `- Copy.jpg` 文件（aegypti29b/30d/32b，md5 与原图相同）。全数据集 md5 检查后没有其他完全重复的文件
  - albopictus_61 只有 1 张，albopictus_89 只有 3 张，其余每只都是 4 张
  - `data/raw/` 已 gitignore，`splits.csv` 会提交，保证不同设备用同一份划分
- [x] 决定：baseline **先不裁剪**（2026-11-03，用户看到不少蚊子不在画面中央）。改为保持 4:3 比例，模型输入 512×384，不缩成 224 的正方形。裁剪 vs 不裁剪留作之后的对照实验
- [x] `scripts/resize_images.py` → `data/processed/ong2021_800/`（800×600，108 MB，原图不动）
- [x] `dataset.py`：`MosquitoDataset(split, transform)`，`CLASSES = ["aegypti", "albopictus"]`。训练时做水平/垂直翻转 + 轻微的亮度和对比度扰动，用 ImageNet 的 mean/std 归一化。已可视化一个 batch 检查过
- [x] `train.py`：ResNet-50 两阶段训练（head 5 epoch，lr 1e-3 → 全部解冻 finetune 15 epoch，lr 1e-4），按 val loss 保存 `runs/<时间>/best.pt`（里面同时存了 classes、input_size、mean、std）和 `history.csv`。Mac 上的 smoke test 已通过（mps）
  - 教训：smoke test 时 `--limit-batches` 只跑前几个 val batch，而 val 是按物种排序的，所以准确率会虚高到 100%。类别不平衡时只看 accuracy 会被误导
- [x] `evaluate.py <ckpt> [--split test]` → `eval_<split>/`：metrics.txt（图片级和蚊子级 accuracy、每类 P/R/F1、混淆矩阵）、confusion_matrix.png、misclassified.jpg、predictions.csv。已用 smoke checkpoint 测试能跑通
- [x] `predict.py <ckpt> <图片或文件夹...> [--out x.csv]`
- [ ] **在 Windows 4070 上跑完整训练 + evaluate**（用户自己跑，2026-11-04）

## 踩坑记录
<!-- 遇到的问题和解决办法 -->

## 更新日志
- 2026-11-01：创建本文件
- 2026-11-03：补充项目背景，完成数据集调研
