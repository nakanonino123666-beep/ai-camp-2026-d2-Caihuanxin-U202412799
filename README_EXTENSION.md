## 可追加到原 README 的“扩展实验”章节

### 额外实验：迁移学习与漏检优先阈值

在课程要求的 Majority Baseline + SmallCNN 之外，我额外实现了 `transfer_experiment.py`。

扩展使用 ImageNet 预训练 ResNet18，并保持课程原始测试集不变；原训练集内部再拆出 validation，用于选择满足目标裂缝召回率的概率阈值。训练集使用轻度旋转、翻转和亮度/对比度增强，验证/测试集不做随机增强。

运行：

```powershell
python transfer_experiment.py
```

可选消融实验：

```powershell
python transfer_experiment.py --no-pretrained
```

输出：

```text
runs/resnet18_pretrained.json
runs/resnet18_pretrained-errors.png
runs/resnet18_pretrained-validation-recall-threshold.png
```

实验主要回答两个额外问题：

1. 在相同真实数据与测试集上，ImageNet 预训练是否比从零学习图像特征更有效？
2. 当业务更重视漏检风险时，是否可以通过验证集选择决策阈值，提高裂缝 recall，并明确展示由此增加的误报代价？

详细说明见 `EXTENSION.md`。
