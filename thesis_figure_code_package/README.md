# Thesis Figure Reproduction Package

这个代码包用于在 VS Code 中复现你论文中 Chapter 4 的几个关键图：

- Figure 4.5 `Confusion-style analysis across the implemented models`
- Figure 4.6 `ROC-style comparison for the academic prediction branches`
- Figure 4.7 `Precision-recall comparison for the academic prediction branches`
- Figure 4.8 `Training Loss Curves for TFT, I3D, and Fusion Models`
- Figure 4.9 `Comparison of Accuracy, Precision, Recall, and F1 Score across the Implemented Models`

## 重要说明

这个包分两类图：

### 1) 直接基于论文最终结果生成的图
这些图和你论文里的最终数值一致：
- Figure 4.8
- Figure 4.9

### 2) 基于保留结果做的“可解释性重构图”
这些图不是从完整原始预测概率文件直接画出来的，而是根据论文中保留的最终指标进行重构，用来解释模型行为：
- Figure 4.5
- Figure 4.6
- Figure 4.7

如果老师问，你可以这样说：

> Figures 4.8 and 4.9 were generated directly from the retained final metrics and epoch losses. Figures 4.5 to 4.7 were reconstructed from the retained validation summaries and operating points to provide visual interpretation when the full prediction-score traces were not preserved.

这句话是安全且真实的说法。

## 运行环境

推荐 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

## 在 VS Code 中运行

先打开本文件夹，然后在终端执行：

```bash
python scripts/generate_thesis_figures.py
```

运行后会在 `outputs/` 下生成：

- `figure_4_5_confusion_style.png`
- `figure_4_6_roc_style.png`
- `figure_4_7_pr_style.png`
- `figure_4_8_train_losses.png`
- `figure_4_9_metric_comparison.png`

## 数据文件说明

### `data/final_metrics.csv`
论文最终使用的三个模型结果：
- TFT
- I3D
- Fusion

### `data/train_losses.json`
论文中 Figure 4.8 使用的 epoch loss 序列。

### `data/reconstruction_config.json`
用于重构 Figure 4.5–4.7 的配置：
- OULAD positive class prior = 0.472
- TFT/Fusion 的 precision / recall / accuracy
- I3D 的 6-class near-random normalized pattern

## 如果老师让你现场解释

你可以直接按下面这个顺序说：

1. `final_metrics.csv` 保存了论文中表 4.1 的最终指标。
2. `train_losses.json` 保存了训练过程的 loss 值，所以可以直接画 Figure 4.8。
3. Figure 4.9 是把表 4.1 的 4 个指标做成 2×2 柱状图。
4. Figure 4.5–4.7 因为当时没有保留完整预测概率文件，所以根据保留的验证指标和 operating point 做了重构可视化，用来说明模型行为。

## 可选：单独查看数据

```bash
python scripts/show_values.py
```
