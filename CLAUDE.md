# 26Mosquito — 项目记忆

> 这个文件是跨聊天、跨设备的共享记忆。Claude Code 每次开新对话会自动读取它。
> 每次有重要进展、决定或踩坑后更新这里，然后 `git commit` + `git push`，其他设备 `git pull` 即可同步。

## ⭐ 新对话先看这里（最后更新：2026-11-10）
- **当前状态**：species 分类做了三个模型，最终候选是**紧边框的“只用蚊子”模型**（`runs\20261110_160414`，只在 Windows 笔记本上）：val acc 0.972，背景泄漏已基本消除。**test 还没用**
- **本周汇报材料**：`Sichen/species_classification/results/weekly_summary_2026-11-10.md`（英文，一页总结 + 图的位置）；详细数字在 `results/README.md`
- **下一步（等用户决定）**：① 用最终模型在 test 上只跑一次；② Step 6：skin 数据集外部测试（要先抠图，背景是人手，抠图方法可能要调）
- **和用户的协作方式**：
  - 用中文回答（用户要英文时再用英文）；项目文档（README、汇报材料等）用英文
  - 一步一步带着做，讲清楚原理；不要一次生成一整套复杂代码
  - 每做完一步都说明下一步要做什么，等用户同意后再继续
  - 有进展就更新本文件；commit/push 之前先问用户

## 项目概况
- JHU Mechanical Engineering Senior Design（2026），仓库：https://github.com/SichenDeng/26Mosquito（**公开仓库**）
- 目标：用于 mosquito **mark–release–recapture (MRR)** 的自动捕捉 + 标记装置；需求文档见仓库根目录 `README.md`（SR-5：物种准确率 ≥70%，要在装置拍的图上验证）
- Sichen 负责 **imaging + computer vision**（`Sichen/` 是个人工作目录）；其他成员分工待补充
- 相机：Raspberry Pi 相机拍照 → 传到电脑**离线**处理（不需要实时推理，也不需要部署到 Pi）
- 完整 imaging pipeline 最终包括：检测/计数、mosquito vs non-mosquito、marked vs unmarked、species
- 装置目前**还没有拍出照片**

## 当前阶段：species classification（二分类）
- 类别：*Aedes aegypti*（yellow fever mosquito）vs *Aedes albopictus*（Asian tiger mosquito）
- 输入：一张只含单只蚊子的图；**现在的模型要求先抠图**（背景换成固定颜色），输出：物种 + 分数
- 先用公开数据训练，先不考虑标记粉；等装置有照片后再测真实表现，必要时用自己的数据 fine-tune
- 原则：
  - **公开数据上的表现 ≠ 装置上的表现**，两者要分开报告
  - softmax 分数不能当作"未知物种 / 非蚊子"的检测依据
  - 每次只改一个变量做对照实验
  - **test 只在最终模型上跑一次**；选模型、调参都只看 val
  - 准确率高不代表模型在看蚊子：要用 Grad-CAM 和去背景实验检查
- 设备：Mac（M3）用来学习/整理数据/调试；Windows 笔记本（RTX 4070 Laptop，8 GB 显存）用来训练

## 目录结构（`Sichen/species_classification/`）
| 路径 | 作用 |
|---|---|
| `scripts/download_ong2021.py` | 从 Kaggle 下载原图（需要 Kaggle token；会被限流，所以用 2 个线程并带重试） |
| `scripts/make_splits.py` | 按蚊子编号分组划分 → `data/splits.csv` |
| `scripts/resize_images.py` | 原图缩到 800×600 → `data/processed/ong2021_800/` |
| `scripts/make_masked_images.py` | 抠图，背景换成固定颜色 → `data/processed/ong2021_800_masked_tight/`（不在 git 里，Windows 上约 6 分钟） |
| `dataset.py` | `MosquitoDataset(split, transform, image_dir)`、`build_transforms(train)`、`eval_transform(...)`、`CLASSES` |
| `train.py` | `python train.py [--image-dir <相对路径>]`，输出 `runs/<时间>/best.pt`（里面记录了 `image_dir`）和 `history.csv` |
| `evaluate.py` | `python evaluate.py <ckpt> [--split test]` → `runs/<时间>/eval_<split>/`；默认用 checkpoint 训练时的图片目录 |
| `predict.py` | `python predict.py <ckpt> <图片或文件夹...> [--out x.csv]`（对抠图模型要先传抠好的图） |
| `gradcam.py` | `python gradcam.py <ckpt> [--split test] [--n 8] [--layer layer4]` → `runs/<时间>/gradcam_<split>_<layer>/`（需要先跑 evaluate.py） |
| `background_test.py` | 抠图函数（`segment`、`keep_mosquito` 等）+ 去背景实验：`python background_test.py <ckpt> [--split val]` → `runs/<时间>/background_test_<split>/`，6 种改图条件下的 acc/recall/AUC、分数分布、抠图检查图（GPU 上约 2 分钟） |
| `results/` | 从 `runs/` 挑出来的结果（不含 `best.pt`），在 git 里；`README.md` 是完整数字，`weekly_summary_2026-11-10.md` 是汇报用总结 |
| `data/splits.csv` | 列为 path, species, specimen_id, split（在 git 里） |
| `data/processed/ong2021_800/` | 1,200 张 800×600 的图，108 MB（在 git 里，带 README 写明来源和许可） |
| `data/processed/ong2021_800_masked*/` | 抠图后的图（不在 git 里，可以用脚本重新生成）。`_masked` 是旧版 8 px 边框，`_masked_tight` 是现在用的 |
| `data/raw/ong2021/` | 原图，4.5 GB（**不在 git 里**，只在 Mac 上有） |
| `runs/` | 训练输出（不在 git 里，`best.pt` 约 94 MB，只在 Windows 上） |

另外还有 `Sichen/yolo_learning/`：早期学习 YOLO 的练习目录，只有 Mac 上的 `.venv`（Python 3.9、torch 2.8、ultralytics）。Mac 上的 species_classification 目前借用这个 venv。

## 环境配置
- **Mac**：`Sichen/species_classification` 下运行 `../yolo_learning/.venv/bin/python <脚本>`（device = mps）
- **Windows（RTX 4070）**：用 conda 环境 `mosquito`（Python 3.11，位置 `C:\anaconda3\envs\mosquito`）。conda 不在 PowerShell 的 PATH 里，要从开始菜单打开 **Anaconda Prompt**；不要用系统默认的 `python`（3.8.6）。首次配置：
  1. `git pull`
  2. `conda create -n mosquito python=3.11 -y`，然后 `conda activate mosquito`，再 `cd /d` 到 `Sichen\species_classification`
  3. `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`，然后 `python -m pip install -r requirements.txt`（驱动支持到 CUDA 12.5，如果下一步不通过，换成 `cu124` 重装）
  4. 检查：`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`，应该输出 `True` 和 4070 的型号。如果是 False，先排查驱动和 CUDA 版本，不要直接训练
  5. 训练最终版本：先 `python scripts\make_masked_images.py`，再 `python train.py --image-dir data/processed/ong2021_800_masked_tight`；如果报 CUDA out of memory，加 `--batch-size 8`
  6. `python evaluate.py runs\<运行文件夹>\best.pt --split val`
  - 环境里没有 scipy / OpenCV，图像处理都用 numpy + PIL + torch 写的

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
  - 3280×2464 分辨率；蚊子只占画面宽度的 15–20%；**不少蚊子不在画面中央**
  - 背景偏蓝紫色、有横向条纹（NoIR 色偏）
  - ⚠️ **两个物种的背景有系统差异**（很可能是分批拍的）：只看背景就能完全分开两类（AUC 1.000）。整体亮度、颜色这些简单统计单项只有 AUC 0.3–0.7，所以差异更可能在纹理、碎屑或底板上。按蚊子分组也防不住这种泄漏
- 备用外部测试集：Ong & Ahmad 2022 Sci Data "Mosquito-on-human-skin"（Mendeley `zw4p9kj6nt/2`，CC BY 4.0），每类 250 张 landing + 250 张 smashed，背景是人手皮肤，用来做 domain shift 检查
- 不采用的数据集：Mosquito Alert 2023（训练集里 aegypti 只有 38 张）；AMID v1（物种和来源高度相关，而且其中 WHO 部分就是 skin 数据集，合并会重复）；Pise 2022（没有 albopictus）

## 模型和训练设置
- torchvision **ResNet-50**，ImageNet 预训练权重（IMAGENET1K_V2），最后一层换成 2 类
- `CLASSES = ["aegypti", "albopictus"]`（输出 0 = aegypti，1 = albopictus）
- 输入 **512×384**（宽×高，保持 4:3），用 ImageNet 的 mean/std 归一化
- 训练增强：水平/垂直翻转，亮度和对比度各 ±20%（不做强烈颜色变化或模糊，以免破坏白色鳞片特征）
- 两个阶段：冻结 backbone 只训练 fc 5 epoch（AdamW，lr 1e-3）→ 全部解冻 finetune 15 epoch（lr 1e-4）
- batch size 16；按 **val loss 最低**保存 `best.pt`（存了 arch、classes、input_size、mean、std、image_dir、epoch）
- 抠图（`background_test.py` 的 `segment`）：用大半径模糊估计局部背景亮度 → 比背景暗 30 以上的像素找出蚊子（取最大连通块）→ 向周围比背景暗 12 以上、且相连的像素生长（能跨过 6 px 的白环）→ 只保留暗像素并补洞 → 外扩 2 px。背景统一填 `FILL_COLOR = (99, 126, 212)`

## 三个模型（2026-11-09/10，详细数字见 `results/README.md`）
| 运行 | 训练图片 | val acc（图 / 蚊子） | 只留背景时 AUC | 说明 |
|---|---|---|---|---|
| `20261109_181015` | 原图 | 1.000 / 1.000（test 0.978） | **1.000** | 主要靠背景判断，结果不能当作物种识别能力 |
| `20261109_203751` | 抠图 + 8 px 背景边框 | 0.983 / 0.978 | 0.953 | 热区移到蚊子上，但边框里的背景还在泄漏 |
| **`20261110_160414`** | 抠图 + 2 px | **0.972 / 0.978** | **0.584** | **最终候选**，泄漏基本消除 |

- 发现过程：baseline test 0.978 → Grad-CAM 热区多在背景上 → 去背景实验：只留背景 acc 0.939 / AUC 1.000，只留蚊子时 albopictus recall 只有 0.30 → 确认靠背景
- 8 px 模型的边框测试：只留边框（纯背景）AUC 0.858，只留边框形状 0.614 → 边框纹理在泄漏；去掉边框 acc 0.972 → 0.928
- 紧边框模型：错 5 张（aegypti_139 两张，它在 8 px 模型里也错；aegypti57d、albopictus26c、albopictus47a）；热区在虫体或腿上；**原图输入只有 0.594**，所以用在装置上必须先抠图；训练时 val 波动较大（epoch 10 掉到 0.65）
- 2 px 下的边框测试没有意义（边框紧贴蚊子，测到的是蚊子自己的边缘和轮廓），判断泄漏要看“只留背景”那一项

## 已做的决定
- 2026-11-03：数据选 Ong 2021（唯一一个两类平衡、用 Pi 相机拍、有完整物种标签的数据集）
- 2026-11-03：baseline **先不裁剪**（用户发现很多蚊子不在画面中央）。用 512×384 来尽量保留细节；裁剪留作之后的对照实验
- 2026-11-04：**先跑 baseline，暂不做色差校正**（两个物种色偏相同，校正对公开数据的结果影响不大；它主要影响 domain shift，留到 Step 5 做对照实验）
- 2026-11-04：模型用 ResNet-50 + 纯 PyTorch 训练循环（为了学原理）；YOLO-cls 和 Transformer 暂不考虑
- 2026-11-09：把 `data/processed/` push 到 GitHub（108 MB，加了来源和许可 README），方便 Windows 直接 pull
- 2026-11-09：Windows 用 conda 环境（系统 Python 是 3.8，太旧）
- 2026-11-09：发现背景泄漏后，**改为只用抠好的蚊子训练**；抠图背景用固定颜色（每张图自己的背景色带一点物种信息）
- 2026-11-10：抠图边框从 8 px 缩到 2 px（8 px 边框仍有泄漏）
- 2026-11-10：结果挑选后放进 `results/` 进 git；`best.pt` 太大（约 94 MB）、又是公开仓库，不放
- test 集保留到最终模型只跑一次；之前的诊断实验都在 val 上做

## 当前进度 / 下一步
- [x] 数据集调研 → 下载 → 去重 → 按蚊子分组划分 → 缩图
- [x] `dataset.py` / `train.py` / `evaluate.py` / `predict.py`，Mac 上 smoke test 通过
- [x] Step 2：Windows 上完整训练 baseline
- [x] Step 3：分析结果 → Grad-CAM + 去背景实验发现背景泄漏 → 抠图重新训练两次（见“三个模型”）
- [x] Step 4：`predict.py` 单张、多张、文件夹预测都确认能用（2026-11-09）
- [ ] **用最终模型（`runs\20261110_160414`）在 test 上只跑一次** ← 等用户同意
- [ ] Step 5：对照实验（每次只改一个变量，在 val 上比较）：按外框裁剪放大蚊子；色差校正（如 gray-world 白平衡）或 hue/saturation 增强；更高的输入分辨率
- [ ] Step 6：domain shift 检查：在 skin 数据集上测（先抠图）；装置有照片后单独测试并单独报告，需要的话标注一些装置照片来 fine-tune
- 需要和团队讨论：
  - 采集自己的数据时，两种蚊子要在**同一批次、同一背景下交替拍摄**，否则模型会学到拍摄批次而不是物种
  - 模型依赖抠图，而现在的抠图方法要求亮且均匀的背景 → 装置的成像台背景要配合
  - 色差问题的长期方案：装置上固定光照和白平衡，用中性背景；考虑不用 NoIR，改用带红外滤片的普通 Pi 相机

## 踩坑记录
- `.gitignore` 里中间带斜杠的模式（如 `data/raw/`）**只匹配仓库根目录**，要写成 `**/data/raw/`
- Kaggle 单文件下载并发太多会触发 HTTP 429；它返回的 Retry-After（5–10 秒）不够，要用指数退避
- 用 `--limit-batches` 做 smoke test 时，val 只跑了前几个 batch，而 `splits.csv` 按物种排序，所以只评估到了 aegypti，准确率会虚高到 100%。**类别不平衡时只看 accuracy 会被误导**，要看每类的 recall 和混淆矩阵
- Kaggle 原数据里有 3 个 `- Copy.jpg` 重复文件（aegypti29b/30d/32b），`make_splits.py` 会跳过它们
- **准确率高 ≠ 模型在看对的地方**：baseline 的 test 0.978 主要来自背景。按个体分组划分防不住“整个类别共有”的背景差异
- 对没见过的图（比如没有蚊子的图），模型可能全判成同一类：除了 accuracy，还要看 AUC 和每类 recall
- Grad-CAM 的 layer4 一格约 32 px，看不出 8 px 边框这种细的泄漏，要靠改输入的实验来测
- 边框测试第一版把浅色腿节、白环也算进了“边框”，结果偏高；做这类测试时要先看图，确认改出来的图里真的只剩想测的东西
- PIL 不能对浮点（`F` 模式）的图做高斯模糊；纯 Python 逐像素做形态学操作太慢（val 要 10 分钟），改用 torch 的 max_pool / avg_pool 在 GPU 上算
- 按名字排序找最新的 `runs/` 子文件夹时，`background_test_val` 这种非时间戳的名字会排到最后

## 更新日志
- 2026-11-01：创建本文件
- 2026-11-03：补充项目背景；完成数据集调研、下载、划分
- 2026-11-04：写好训练、评估、预测脚本，Mac 上 smoke test 通过
- 2026-11-09：数据 push 到 GitHub；重新整理本文件，方便在 Windows 上开新对话时接上
- 2026-11-09：Windows 改用 conda 环境；完成 baseline 训练和 test 评估
- 2026-11-09/10：Grad-CAM 和去背景实验发现 baseline 靠背景判断；只用蚊子重新训练两次（8 px、2 px 边框）；结果整理到 `results/`
- 2026-11-10：重写本文件的进度部分；写本周汇报总结 `results/weekly_summary_2026-11-10.md`
