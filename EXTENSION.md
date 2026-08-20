# Day 2 扩展实验：预训练迁移学习 + 漏检优先阈值

这个扩展不修改课程原始 `SmallCNN` 实验，而是在其基础上增加一个可复查的迁移学习对照。

## 为什么做这个扩展

原始 SmallCNN 只有 9,586 个参数，输入缩放为 64×64，并且只使用每类 600 张中的训练部分。它适合教学，但从实际视觉任务角度看，存在两个明显问题：

1. 从零训练的小网络只能依赖当前很小的数据子集学习边缘、纹理等特征。
2. 原程序使用 `argmax`，等价于固定决策边界；但本题真正关心的是漏检裂缝，而不是单独最大化 accuracy。

因此扩展提出两个问题：

- ImageNet 预训练的 ResNet18 是否能在相同真实数据和相同测试集上得到更好的裂缝识别能力？
- 如果维护团队更加害怕漏检，能否利用验证集调节“判定为裂缝”的概率阈值，主动换取更高 recall？

## 扩展内容

### 1. ResNet18 迁移学习

使用 torchvision 官方 `ResNet18_Weights.DEFAULT`。

训练分两阶段：

- 冻结 backbone，只训练最后的两分类全连接层；
- 再用更小学习率解冻整个网络微调。

这可以与 `--no-pretrained` 模式形成消融实验：

```powershell
python transfer_experiment.py
python transfer_experiment.py --no-pretrained
```

二者结构完全相同，主要区别是是否利用 ImageNet 预训练知识，因此比“换一个更大模型”更有解释力。

### 2. 只对训练集做数据增强

训练图像增加：

- 随机水平翻转；
- ±8° 小角度旋转；
- 轻度亮度与对比度扰动。

测试和验证图像绝不做随机增强。

目的不是虚构新数据，而是让模型减少对固定拍摄角度、亮度和局部纹理的依赖。

### 3. 单独验证集

课程原始训练/测试划分保持不变。

扩展只把原训练集再拆成：

- fit：用于更新模型参数；
- validation：用于选择概率阈值；
- test：仍然使用课程原来那一批测试图片。

因此没有根据 test 分数反复调阈值，避免测试集泄漏。

### 4. 漏检优先阈值

默认目标：

```text
validation crack recall >= 0.90
```

程序会寻找满足这个条件的最高阈值。

为什么选“最高”？

因为在达到目标召回率的前提下，阈值越高通常越能减少把正常表面误报成裂缝的问题。

最终同时报告：

- 固定阈值 0.5 的测试结果；
- 验证集选择阈值后的测试结果。

这会展示 precision / recall 的实际权衡。

## 运行

首先仍然执行原课程数据检查：

```powershell
python train.py --check-data
```

运行预训练 ResNet18：

```powershell
python transfer_experiment.py
```

如果 CPU 比较慢，可以只做冻结阶段：

```powershell
python transfer_experiment.py --frozen-epochs 1 --finetune-epochs 0
```

运行从零初始化的 ResNet18 对照：

```powershell
python transfer_experiment.py --no-pretrained
```

提高目标召回率：

```powershell
python transfer_experiment.py --target-recall 0.95
```

测试：

```powershell
python -m unittest discover -s tests -v
```

## 输出

运行后新增：

```text
runs/resnet18_pretrained.json
runs/resnet18_pretrained-errors.png
runs/resnet18_pretrained-validation-recall-threshold.png
```

从零初始化对照则为：

```text
runs/resnet18_scratch.json
runs/resnet18_scratch-errors.png
runs/resnet18_scratch-validation-recall-threshold.png
```

## 最适合放进 PPT 的比较表

实际运行后填写：

| 模型 | Accuracy | Crack Precision | Crack Recall | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.500 | 0.500 | 1.000 | 0 | 150 |
| SmallCNN | 0.813 | 0.892 | 0.713 | 43 | 13 |
| ResNet18 pretrained @ 0.5 | 实测 | 实测 | 实测 | 实测 | 实测 |
| ResNet18 pretrained @ selected threshold | 实测 | 实测 | 实测 | 实测 | 实测 |
| ResNet18 scratch（可选） | 实测 | 实测 | 实测 | 实测 | 实测 |

不要在未运行之前填写 ResNet18 数字。

## 可以在答辩里说什么

这个扩展不是单纯为了得到更高准确率。

它增加了三个更接近真实机器学习实验的概念：

1. **迁移学习**：利用大规模自然图像预训练得到的通用视觉特征；
2. **消融实验**：相同 ResNet18 比较 pretrained 与 scratch，隔离预训练本身的贡献；
3. **工作点选择**：分类器不是只能用固定阈值，根据维护任务对漏检的成本，可以选择更偏向 recall 的决策阈值。

同时保留原课程的边界：模型只用于安排人工复核顺序，不能把预测 `no_crack` 当成结构安全结论。

## 进一步可扩展但不建议现在继续堆的方向

如果老师还要求继续发挥，可以再做：

- Grad-CAM：查看模型到底在看裂缝还是背景纹理；
- 按亮度/边缘密度分组比较 recall；
- 近重复图像检测，进一步研究课程强调的数据泄漏问题；
- MobileNetV3 与 ResNet18 的“准确率—速度—参数量”对比。

优先级上，Grad-CAM 是下一项最值得做的，因为它直接增强失败案例的可解释性。
