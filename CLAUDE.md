# 26Mosquito — 项目记忆

> 这个文件是跨聊天、跨设备的共享记忆。Claude Code 每次开新对话会自动读取它。
> 每次有重要进展、决定或踩坑后更新这里，然后 `git commit` + `git push`，其他设备 `git pull` 即可同步。

## ⭐ 新对话先看这里（最后更新：2026-11-09）
- **当前状态**：baseline 已在 **Windows 笔记本（RTX 4070 Laptop）** 上训练并评估完（运行文件夹 `runs\20261109_181015`，只在这台电脑上）；结果见"当前进度"里的 Step 3
- **下一步**：只用蚊子训练的模型（`runs\20261109_203751`）在 val 上 0.983，热区在蚊子身上，但“只留背景”仍有 AUC 0.953。下一步是检查抠图边框是否泄漏（待用户同意），排除后再在 test 上跑一次，之后做 Step 6 的 skin 数据集检查
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
| `gradcam.py` | `python gradcam.py <ckpt> [--split test] [--n 8] [--layer layer4]` → `runs/<时间>/gradcam_<split>_<layer>/`（需要先跑 evaluate.py） |
| `background_test.py` | `python background_test.py <ckpt> [--split val]` → `runs/<时间>/background_test_<split>/`：分别在原图、只留蚊子、只留背景三种图上测 acc/recall/AUC，并输出抠图检查图（双阈值 + 生长，GPU 上 val 约 1.5 分钟） |
| `data/splits.csv` | 列为 path, species, specimen_id, split（在 git 里） |
| `data/processed/ong2021_800/` | 1,200 张 800×600 的图，108 MB（在 git 里，带 README 写明来源和许可） |
| `data/raw/ong2021/` | 原图，4.5 GB（**不在 git 里**，只在 Mac 上有） |
| `runs/` | 训练输出（不在 git 里） |

另外还有 `Sichen/yolo_learning/`：早期学习 YOLO 的练习目录，只有 Mac 上的 `.venv`（Python 3.9、torch 2.8、ultralytics）。Mac 上的 species_classification 目前借用这个 venv。

## 环境配置
- **Mac**：`Sichen/species_classification` 下运行 `../yolo_learning/.venv/bin/python <脚本>`（device = mps）
- **Windows（RTX 4070）**：用 conda 环境 `mosquito`（Python 3.11，位置 `C:\anaconda3\envs\mosquito`）。conda 不在 PowerShell 的 PATH 里，要从开始菜单打开 **Anaconda Prompt**；不要用系统默认的 `python`（3.8.6）。首次配置：
  1. `git pull`
  2. `conda create -n mosquito python=3.11 -y`，然后 `conda activate mosquito`，再 `cd /d` 到 `Sichen\species_classification`
  3. `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`，然后 `python -m pip install -r requirements.txt`（驱动支持到 CUDA 12.5，如果下一步不通过，换成 `cu124` 重装）
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
- [x] Step 2：在 Windows 上跑完整训练（`runs\20261109_181015`，20 epoch，约 18 分钟；`best.pt` 来自 epoch 11，val loss 0.0065）
- [x] Step 3：分析 test 结果（2026-11-09，只代表公开数据上的表现）
  - 图片级 accuracy 0.978（176/180），蚊子级 1.000（45/45，95% CI 约 92–100%）
  - 混淆矩阵：aegypti 88 对 / 4 错，albopictus 88 对 / 0 错
  - 4 张错图全是 aegypti 判成 albopictus，来自 2 只蚊子（aegypti_144 的 a、d；aegypti_150 的 a、c），其中一张错得很自信（0.91）。样本太少，不能认定模型有方向性偏差
  - train/val loss 没有持续的过拟合迹象；从 epoch 11 之后 val loss 只是在小范围波动
  - ⚠️ Grad-CAM（2026-11-09，test 抽样，layer4，人工看图）：判对的 aegypti 约 5/8 热区在蚊子身上（另有 1 张混合、2 张在背景）；判对的 albopictus 0/8；4 张错图 0/4。layer3 噪声太大，难以解读。可能是模型依赖背景，也可能是 Grad-CAM 对“缺少 aegypti 特征”这类依据显示不出来，**还没有定论**，在排除之前 97.8% 要谨慎引用
  - ❌ **去背景实验（2026-11-09，val，`background_test.py`）确认了背景泄漏**：
    - 原图：acc 1.000，AUC 1.000
    - 只留蚊子（背景换纯色）：acc 0.656，AUC 0.947。aegypti recall 1.000，albopictus 只有 0.326，大多被判成 aegypti
    - 只留背景（蚊子抹掉）：**acc 0.939，AUC 1.000**，92 张 albopictus 背景全部判对
    - 结论：baseline 主要靠背景区分物种，蚊子本身只提供部分信息。test 的 97.8% **不能**当作物种识别能力来引用。按蚊子分组划分防不住这种泄漏，因为背景差异是整个物种共有的（很可能每个物种是分开拍的）
    - 全图的简单统计（亮度、RGB 中位数、横纹、暗角）单个 AUC 只有 0.3–0.7，所以泄漏的线索不是简单的整体偏色，更可能是纹理、碎屑或底板的特征
  - 给装置采数据的教训：两种蚊子要在同一批次、同一背景下交替拍摄，否则模型会学到“拍摄批次”而不是物种
- [x] **只用蚊子重新训练**（2026-11-09，`runs\20261109_203751`，`--image-dir data/processed/ong2021_800_masked`，其他设置和 baseline 相同；best = epoch 18）
  - 去背景的图由 `scripts/make_masked_images.py` 生成，背景统一填成固定颜色 `FILL_COLOR`（不用每张图自己的中位色，因为背景颜色本身带一点物种信息）；不进 git
  - val（去背景）：acc 0.983，蚊子级 0.978。3 张错图全来自 aegypti_139，这只蚊子在早期版本里就是最难抠的
  - Grad-CAM：抽查的 16 张判对的图，热区全部在蚊子身上（baseline 是 5/16）
  - 去背景实验：只留蚊子 acc 0.972 / AUC 0.998（baseline 是 0.656 / 0.947）；**只留背景仍有 acc 0.789 / AUC 0.953**（baseline 是 0.939 / 1.000）
  - 未解决：模型从没见过外框以外的背景，“只留背景”的图却仍能排序。可能的原因：(1) 抠图时留下的约 8 px 背景边框里的纹理；(2) B 图里残留的腿尖或补过的平滑轮廓。layer4 的 Grad-CAM 分辨率区分不了虫体和这圈边框
  - **test 还没用**：等泄漏问题排除后只跑一次
- [ ] Step 4：用 `predict.py` 对单张图和文件夹做预测，确认能用 ← 下一步
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
- 2026-11-09：Windows 改用 conda 环境；完成 baseline 训练和 test 评估
