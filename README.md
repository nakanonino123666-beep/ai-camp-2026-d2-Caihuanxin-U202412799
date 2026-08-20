# D2：混凝土裂缝图像筛查器

为设施维护团队做一个照片初筛工具：比较"总是预测训练集多数类别"的简单基线与一个小型卷积神经网络（SmallCNN），在真实混凝土裂缝照片上重点检查**漏检裂缝**（假阴性）和错误图像。输出只用于安排人工复核顺序，不替代现场检查、工程师判断或安全决策。

## 目录结构

```text
train.py            # 主程序：真实数据检查、固定子集与划分、基线/CNN 训练与评估、错误图像网格
models.py           # SmallCNN（学生完成的核心改动：TODO 1 与 TODO 2）
analyze_errors.py   # 错误分析：同一划分上重训，导出全部 56 个错误的图像统计
transfer_experiment.py  # 扩展：ResNet18 迁移学习 + 漏检优先阈值选择（见 EXTENSION.md）
EXTENSION.md        # 扩展实验说明
tests/              # 单元测试（不依赖真实数据：张量形状、划分逻辑、混淆计数、阈值选择）
requirements.txt    # 依赖：torch>=2.5、torchvision>=0.20、matplotlib>=3.9
data/raw/           # 真实数据（需从 Kaggle 下载，不入库，被 .gitignore 排除）
runs/               # 运行后生成：baseline.json / cnn.json 指标、errors.png 错误网格、error-analysis.csv（不入库）
```

## 环境要求

- Windows + Python 3（本机验证：Python 3.13.5）
- 依赖安装（CPU 版即可，CUDA 不是必需）：

```powershell
python -m pip install -r requirements.txt
```

## 从零运行

```powershell
# 1. 进入本目录
cd student-work\day-02-concrete

# 2. 下载真实数据（见下），解压后把 Positive/Negative 文件夹放到 data\raw\ 下

# 3. 验证真实数据
python train.py --check-data
# 预期：REAL DATA CHECK PASSED；counts: {'Negative': 20000, 'Positive': 20000}

# 4. 安装依赖
python -m pip install -r requirements.txt

# 5. 运行多数类基线
$env:KMP_DUPLICATE_LIB_OK="TRUE"   # 见下方"常见问题"
python train.py --model baseline
# 输出：runs/baseline.json（含准确率、裂缝召回率、混淆矩阵、前 12 个错误）+ runs/baseline-errors.png

# 6. 训练并评估 SmallCNN（与基线同数据、同划分、同种子）
python train.py --model cnn
# 输出：runs/cnn.json + runs/cnn-errors.png

# 7. 错误分析（同一划分重训并统计全部错误）
python analyze_errors.py
# 输出：runs/error-analysis.csv（56 个错误的路径、错误类型与图像统计）

# 8. 运行单元测试
python -m unittest discover -s tests -v
# 预期：6 个测试全部 OK
```

## 常见问题

- **报错 `OMP: Error #15 ... libiomp5md.dll already initialized`**：Windows 上 torch 与其它包的 OpenMP 运行时冲突。在命令前加 `$env:KMP_DUPLICATE_LIB_OK="TRUE"`（PowerShell）或 `set KMP_DUPLICATE_LIB_OK=TRUE`（cmd）即可，本机全部训练命令都带上了它。
- **`FileNotFoundError: Real crack image folder not found at data\raw`**：数据没放对位置。确认 `data/raw/Positive` 与 `data/raw/Negative` 两个文件夹名大小写正确、里面是图像文件而不是套了一层解压目录。

## 真实数据

- 页面：https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
- 内容：混凝土表面照片，`Positive`（有裂缝）与 `Negative`（无裂缝）**各 20,000 张**
- 放置位置：`data/raw/Positive` 与 `data/raw/Negative`（已被 `.gitignore` 排除，不会提交）
- 检查命令：`python train.py --check-data`（看到 `REAL DATA CHECK PASSED` 才能进入模型步骤）
- 不允许：生成或编造替代数据、从陌生镜像下载、把 40,000 张原始图像提交到仓库
- 下载失败时：保留完整报错，联系教师取得同一来源的缓存副本

## 方法与评估

- **输入**：一张 227×227 的混凝土照片，被缩放到 64×64×3 的图像张量后进入模型
- **基线**：多数类基线——总是预测训练子集中数量最多的类别，完全不读取图像内容
- **候选**：`SmallCNN`（models.py）——两层卷积 + 池化 + 全连接的二分类网络：
  `Conv2d(3,8,k=3,p=1) → ReLU → MaxPool2d(2) → Conv2d(8,16,k=3,p=1) → ReLU → MaxPool2d(2) → Flatten → Linear(4096,2)`
- **数据范围与划分**：从 40,000 张中按类别各取 `--max-per-class` 张（默认 600），每类 75/25 划分训练/测试，`--seed 2026` 固定；基线与候选使用**同一划分**
- **训练**：Adam（lr=0.001）+ CrossEntropyLoss，默认 2 个 epoch（`--epochs`）
- **指标**：准确率、裂缝精确率、**裂缝召回率**（与使用者最相关的指标：真实裂缝里被发现的比例）、混淆矩阵、漏检裂缝数（假阴性）
- **错误分析**：每个模型导出前 12 个错误样本路径与真实/预测标签，并用前 6 个生成错误图像网格（`runs/*-errors.png`）

## 输出位置

- `runs/baseline.json`：基线指标、混淆矩阵、前 12 个错误（真实标签与预测标签）
- `runs/cnn.json`：CNN 指标、每个 epoch 的训练损失、前 12 个错误
- `runs/baseline-errors.png` / `runs/cnn-errors.png`：错误图像网格
- `runs/error-analysis.csv`：全部 56 个 CNN 错误的路径、错误类型（漏检/误报）与亮度、对比度、边缘密度统计

## 本机运行结果（seed 2026，每类 600 张，75/25 划分，300 张测试图）

| 指标 | 多数类基线 | SmallCNN | 说明 |
| --- | ---: | ---: | --- |
| 准确率 | 0.500 | 0.813 | 两类各 450 张训练图完全均衡，基线"多数类"平票，退化为永远猜 crack |
| 裂缝精确率 | 0.500 | 0.892 | 被标"有裂缝"的图里真正有裂缝的比例 |
| 裂缝召回率 | 1.000 | 0.713 | 真实裂缝中被发现的比例；基线全猜 crack 所以不漏，但 150 张误报等于没有筛选 |
| 漏检裂缝（假阴性） | 0 | 43 | 与使用者最相关的错误：需要人工查看却没被优先发现 |
| 误报（假阳性） | 150 | 13 | 基线把一半照片全标成裂缝，CNN 大幅减少误报 |
| 混淆矩阵 [TN,FP;FN,TP] | [[0,150],[0,150]] | [[137,13],[43,107]] | 行=真实，列=预测，标签顺序 no_crack/crack |

CNN 训练损失：[0.6818, 0.6186]（2 个 epoch，CPU 约 11 秒）。

## 扩展实验：迁移学习与漏检优先阈值

在课程要求的 Baseline + SmallCNN 之外，`transfer_experiment.py` 增加一个可复查的迁移学习对照，回答两个额外问题：

1. 相同真实数据与测试集上，ImageNet 预训练的 ResNet18 是否比从零学习更有效？
2. 当业务更怕漏检裂缝时，能否用验证集选择决策阈值，提高裂缝召回率，并明确展示增加的误报代价？

做法：保持课程原始 300 张 test 不变；把原 900 张训练图再拆成 fit / validation，用 validation 选择"召回率 ≥ 目标值"的最高阈值；训练只用轻度增强（翻转、±8° 旋转、亮度/对比度扰动），验证/测试不做随机增强。

运行：

```powershell
python transfer_experiment.py                  # ImageNet 预训练 ResNet18
python transfer_experiment.py --no-pretrained  # 同结构、随机初始化（消融对照）
```

首次运行会自动下载 ResNet18 官方权重（需联网）。输出见 `runs/resnet18_pretrained.json`、`runs/resnet18_pretrained-errors.png`、`runs/resnet18_pretrained-validation-recall-threshold.png`（scratch 对应 `resnet18_scratch_*`）。详细说明见 [EXTENSION.md](EXTENSION.md)。

| 模型 | Accuracy | Crack Precision | Crack Recall | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.500 | 0.500 | 1.000 | 0 | 150 |
| SmallCNN | 0.813 | 0.892 | 0.713 | 43 | 13 |
| ResNet18 pretrained @ 0.5 | 0.997 | 0.993 | 1.000 | 0 | 1 |
| ResNet18 pretrained @ 阈值 0.9996 | 0.953 | 1.000 | 0.907 | 14 | 0 |
| ResNet18 scratch @ 0.5 | 0.983 | 0.993 | 0.973 | 4 | 1 |

本机实测（CPU，默认 1 冻结 epoch + 1 微调 epoch）：

- **消融结论**：同结构 ResNet18，pretrained（漏检 0）明显好于 scratch（漏检 4）——提升来自 ImageNet 预训练知识，而不是单纯把网络变大。
- **阈值实验**：验证集按"召回率 ≥ 0.90"选出的阈值高达 0.9996（模型置信度太高），它在测试集上漏检 14 张（recall 0.907）却换来 0 误报——对"漏检裂缝最贵"的业务，默认 0.5 反而是漏检最少的工作点。这正好说明：模型本身和"如何使用模型输出"是两回事。
- **边界**：子集结果仍受相关图像泄漏风险影响，不推广到全部 40,000 张；输出仍只用于安排人工复核顺序。

## 限制

这是照片初筛工具，不是结构安全鉴定。模型输出只用于把"可能有裂缝"的照片优先交给人工复核；低置信度、异常材质和不同拍摄环境的照片仍必须由现场人员或工程师判断。只报准确率会掩盖漏检风险——必须同时看裂缝召回率和错误图像；随机拆分高度相似的图像块可能造成数据泄漏，使测试分数过于乐观。
