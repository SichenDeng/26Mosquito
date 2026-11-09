# 26Mosquito — 项目记忆

> 这个文件是跨聊天、跨设备的共享记忆。Claude Code 每次开新对话会自动读取它。
> 每次有重要进展、决定或踩坑后更新这里，然后 `git commit` + `git push`，其他设备 `git pull` 即可同步。

## ⭐ 新对话先看这里（最后更新：2026-11-09）
- **当前状态**：代码和数据都已 push 到 GitHub（commit `b762e6b`）。用户正在 **Windows 笔记本（RTX 4070 Laptop）** 上第一次跑完整训练
- **在 Windows 上要做的事**：配好环境（见下方"环境配置 → Windows"）→ `python train.py` → `python evaluate.py runs\<运行文件夹>\best.pt`
- **跑完后 Claude 要做的**：帮用户分析训练日志、`eval_test/metrics.txt`、`misclassified.jpg`（具体看什么见"下一步"里的 Step 3）
- **和用户的协作方式**：
  - 用中文回答（用户要英文时再用英文）
  - 一步一步带着做，讲清楚原理；不要一次生成一整套复杂代码
  - 每做完一步都说明下一步要做什么，等用户同意后再继续
  - 有进展就更新本文件；commit/push 之前先问用户

## 项目概况
- JHU Mechanical Engineering Senior Design（2026），仓库：https://github.com/SichenDeng/26Mosquito（**公开仓库**）
- 目标：用于 mosquito **mark–release–recapture (MRR)** 的自动捕捉 + 标记装置
- Sichen 负责 **imaging + computer vision**（`Sichen/` 是个人工作目录）；其他成员分工待补充
- 相机：Raspberry Pi 相机拍照 → 传到电脑**离线**处理（不需要实时推理，也不需要部署到 Pi）
- 完整 imaging pipeline 最终包括：检测/计数、mosquito vs non-mosquito、marked vs unmarked、species
- 装置目前**还没有拍出照片**

## 当前阶段：species classification（二分类）
- 类别：*Aedes aegypti*（yellow fever mosquito）vs *Aedes albopictus*（Asian tiger mosquito）
- 输入：一张只含单只蚊子的图（或裁剪后的单只图）；输出：物种 + 分数
- 先用公开数据训练 baseline，先不考虑标记粉；等装置有照片后再测真实表现，必要时用自己的数据 fine-tune
- 原则：
  - **公开数据上的表现 ≠ 装置上的表现**，两者要分开报告
  - softmax 分数不能当作"未知物种 / 非蚊子"的检测依据
  - 每次只改一个变量做对照实验
- 设备：Mac（M3）用来学习/整理数据/调试；Windows 笔记本（RTX 4070 Laptop，8 GB 显存）用来训练

## 目录结构（`Sichen/species_classification/`）
| 路径 | 作用 |
|---|---|
| `scripts/download_ong2021.py` | 从 Kaggle 下载原图（需要 Kaggle token；会被限流，所以用 2 个线程并带重试） |
| `scripts/make_splits.py` | 按蚊子编号分组划分 → `data/splits.csv` |
| `scripts/resize_images.py` | 原图缩到 800×600 → `data/processed/ong2021_800/` |
| `dataset.py` | `MosquitoDataset(split, transform)`、`build_transforms(train)`、`eval_transform(...)`、`CLASSES` |
| `train.py` | 训练，输出到 `runs/<时间>/best.pt` 和 `history.csv` |
| `evaluate.py` | `python evaluate.py <ckpt> [--split test]` → `runs/<时间>/eval_<split>/` |
| `predict.py` | `python predict.py <ckpt> <图片或文件夹...> [--out x.csv]` |
| `data/splits.csv` | 列为 path, species, specimen_id, split（在 git 里） |
| `data/processed/ong2021_800/` | 1,200 张 800×600 的图，108 MB（在 git 里，带 README 写明来源和许可） |
| `data/raw/ong2021/` | 原图，4.5 GB（**不在 git 里**，只在 Mac 上有） |
| `runs/` | 训练输出（不在 git 里） |

另外还有 `Sichen/yolo_learning/`：早期学习 YOLO 的练习目录，只有 Mac 上的 `.venv`（Python 3.9、torch 2.8、ultralytics）。Mac 上的 species_classification 目前借用这个 venv。

## 环境配置
- **Mac**：`Sichen/species_classification` 下运行 `../yolo_learning/.venv/bin/python <脚本>`（device = mps）
- **Windows（RTX 4070）**，在 `Sichen\species_classification` 下：
  1. `git pull`
  2. `python -m venv .venv`，然后 `.venv\Scripts\activate`
  3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`，然后 `pip install -r requirements.txt`
  4. 检查：`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`，应该输出 `True` 和 4070 的型号。如果是 False，先排查驱动和 CUDA 版本，不要直接训练
  5. `python train.py`；如果报 CUDA out of memory，就改用 `python train.py --batch-size 8`
  6. `python evaluate.py runs\<运行文件夹>\best.pt`

## 数据：Ong et al. 2021（Sci Rep 11:9908）
- Kaggle `pradeepisawasan/aedes-mosquitos`，CC BY-NC-SA 4.0（使用时需注明出处，非商业，相同方式共享）
- 用 **Raspberry Pi NoIR v2 相机**拍摄，距离 2.5 cm，白色背景，LED 照明，雌蚊 >12 天龄（和我们的装置最接近）
- Kaggle 原始结构：`{sp}_{0,90,180,270}[_testing]/{sp}{N}{a-d}.jpg`。90/180/270 是旋转副本，**只下载了 `_0` 和 `_0_testing`**
- **同一个编号 N 的 a–d 是同一只死蚊子被翻动后拍的 4 张**（两个物种都看图确认过：腿的形态、碎屑位置都对得上）
- ⚠️ 官方的 testing 划分有泄漏（同一只蚊子的其他照片在 train 里），所以**不使用官方划分**
- 我们的划分：seed=42，按蚊子分组，各物种分层，比例 70/15/15。共 1,200 张 / 301 只蚊子：
  - train：211 只，840 张（每类 420）
  - val：45 只，180 张（aegypti 88 / albopictus 92）
  - test：45 只，180 张（aegypti 92 / albopictus 88）
  - val/test 都很小：错 1 只蚊子（4 张）准确率就差约 2%，1–2 个百分点的差别没有意义
- 图像特点：
  - 3280×2464 分辨率；蚊子只占画面宽度的 15–20%
  - **不少蚊子不在画面中央**
  - 背景偏蓝紫色、有横向条纹（NoIR 色偏），两个物种都一样，所以模型不会靠颜色区分物种；但这是将来和装置照片之间 domain shift 的来源之一
- 备用外部测试集：Ong & Ahmad 2022 Sci Data "Mosquito-on-human-skin"（Mendeley `zw4p9kj6nt/2`，CC BY 4.0），每类 250 张 landing + 250 张 smashed，背景是人手皮肤，用来做 domain shift 检查
- 不采用的数据集：Mosquito Alert 2023（训练集里 aegypti 只有 38 张）；AMID v1（物种和来源高度相关，而且其中 WHO 部分就是 skin 数据集，合并会重复）；Pise 2022（没有 albopictus）

## 模型和训练设置（baseline）
- torchvision **ResNet-50**，ImageNet 预训练权重（IMAGENET1K_V2），最后一层换成 2 类
- `CLASSES = ["aegypti", "albopictus"]`（输出 0 = aegypti，1 = albopictus）
- 输入 **512×384**（宽×高，保持 4:3），不裁剪，用 ImageNet 的 mean/std 归一化
- 训练增强：水平/垂直翻转，亮度和对比度各 ±20%（不做强烈颜色变化或模糊，以免破坏白色鳞片特征）
- 两个阶段：
  1. 冻结 backbone，只训练 fc：5 epoch，AdamW，lr 1e-3
  2. 全部解冻做 finetune：15 epoch，lr 1e-4
- batch size 16；按 **val loss 最低**保存 `best.pt`（里面同时存了 arch、classes、input_size、mean、std、epoch）
- evaluate 输出：图片级和蚊子级 accuracy（蚊子级 = 同一只蚊子各张图的分数取平均）、每类 P/R/F1、混淆矩阵、错分图片

## 已做的决定
- 2026-11-03：数据选 Ong 2021（唯一一个两类平衡、用 Pi 相机拍、有完整物种标签的数据集）
- 2026-11-03：baseline **先不裁剪**（用户发现很多蚊子不在画面中央）。用 512×384 来尽量保留细节；裁剪留作之后的对照实验
- 2026-11-04：**先跑 baseline，暂不做色差校正**（两个物种色偏相同，校正对公开数据的结果影响不大；它主要影响 domain shift，留到 Step 5 做对照实验）
- 2026-11-04：模型用 ResNet-50 + 纯 PyTorch 训练循环（为了学原理）；YOLO-cls 和 Transformer 暂不考虑
- 2026-11-09：把 `data/processed/` push 到 GitHub（108 MB，加了来源和许可 README），方便 Windows 直接 pull

## 当前进度 / 下一步
- [x] 数据集调研 → 下载 → 去重 → 按蚊子分组划分 → 缩图
- [x] `dataset.py` / `train.py` / `evaluate.py` / `predict.py` 写好，并在 Mac 上用 smoke test 跑通
- [ ] **Step 2：在 Windows 上跑完整训练** ← 当前
- [ ] **Step 3：分析 test 结果**
  - 先看 train/val loss 曲线：有没有过拟合（train 一直降、val 回升）？
  - 看每类的 recall 是否平衡；看混淆矩阵
  - 看错分图片有没有规律（姿态？模糊？蚊子太小或在边缘？）
  - 比较图片级和蚊子级 accuracy
  - 注意 test 只有 45 只蚊子，结果的波动会比较大
- [ ] Step 4：用 `predict.py` 对单张图和文件夹做预测，确认能用
- [ ] Step 5：对照实验（每次只改一个变量，在 val 上比较）：裁剪 vs 不裁剪；色差校正（如 gray-world 白平衡）或 hue/saturation 增强；更高的输入分辨率
- [ ] Step 6：domain shift 检查：在 skin 数据集上测；装置有照片后单独测试并单独报告，需要的话标注一些装置照片来 fine-tune
- 色差问题的长期方案（需要和团队讨论）：装置上固定光照和白平衡，用中性背景；考虑不用 NoIR，改用带红外滤片的普通 Pi 相机

## 踩坑记录
- `.gitignore` 里中间带斜杠的模式（如 `data/raw/`）**只匹配仓库根目录**，要写成 `**/data/raw/`
- Kaggle 单文件下载并发太多会触发 HTTP 429；它返回的 Retry-After（5–10 秒）不够，要用指数退避
- 用 `--limit-batches` 做 smoke test 时，val 只跑了前几个 batch，而 `splits.csv` 按物种排序，所以只评估到了 aegypti，准确率会虚高到 100%。**类别不平衡时只看 accuracy 会被误导**，要看每类的 recall 和混淆矩阵
- Kaggle 原数据里有 3 个 `- Copy.jpg` 重复文件（aegypti29b/30d/32b），`make_splits.py` 会跳过它们

## 更新日志
- 2026-11-01：创建本文件
- 2026-11-03：补充项目背景；完成数据集调研、下载、划分
- 2026-11-04：写好训练、评估、预测脚本，Mac 上 smoke test 通过
- 2026-11-09：数据 push 到 GitHub；重新整理本文件，方便在 Windows 上开新对话时接上
