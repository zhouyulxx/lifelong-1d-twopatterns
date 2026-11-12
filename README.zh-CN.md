# Lifelong Learning：UCR TwoPatterns 一维序列最小基线

> 目标：先验证普通神经网络在 **Task A → Task B** 顺序训练时，旧任务 A 的准确率是否下降。此阶段没有实现四维晶格。

## 已经存入仓库的真实数据

- `data/TwoPatterns/TwoPatterns_TRAIN.ts`：官方 TRAIN 1,000 条。
- `data/TwoPatterns/TwoPatterns_TEST.ts`：官方 TEST 4,000 条。
- 单通道，每条 128 个数值；4 类；TwoPatterns 是人工生成的标准**时间序列分类**数据，不是展平图像。
- `data/TwoPatterns/SOURCE.json`：官方来源网址、文件 SHA-256 及检查记录。
- `scripts/fetch_twopatterns.py`：以后可重新下载、验证并生成同样的数据文件。不生成替代样本。

上面的数据通过 GitHub Actions 下载、逐条验证后提交。官方来源：[Time Series Classification Archive / TwoPatterns](https://timeseriesclassification.com/description.php?Dataset=TwoPatterns)。

## 测试什么

| 阶段 | 训练数据 | 验证 |
|---|---|---|
| A | 原始标签的前两个类别（映射 0、1） | A 的测试准确率 |
| B | 原始标签的后两个类别（映射 2、3） | A 和 B 的测试准确率 |

- 一个共享的 **128 → 128 → 128 → 4 MLP**。
- Task B 直接继续更新 Task A 的模型权重；**不重置模型**。
- Task B 不含 A 的训练样本；**无 experience replay**。
- 测试时四个类别共同竞争，**没有 Task ID**。
- 采用每条序列独立 z-normalization，不从测试集学习全局标准化参数。
- Task-aware 两类限制输出的分数**仅供诊断**，不是主要指标。

## 运行

要求 Python 3.10+，在仓库根目录：

```bash
python -m pip install -r requirements.txt
python run_baseline.py --epochs-a 20 --epochs-b 20 --seed 42
```

如果数据文件意外丢失：

```bash
python scripts/fetch_twopatterns.py
```

CPU 运行也可以：

```bash
python run_baseline.py --device cpu --epochs-a 20 --epochs-b 20
```

## 结果

默认保存于 `results/baseline_seed42/`：

- `metrics.json`：A 学完后、B 学完后的两任务准确率与遗忘量。
- `accuracy_matrix.csv`：准确率矩阵。
- `train_history.csv`：逐轮训练损失。
- `confusion_after_B.csv`：学习 B 后的四类混淆矩阵。
- `model_after_A.pt` / `model_after_B.pt`：两阶段参数检查点。

主要比较：

```
Forgetting_A = Acc(A immediately after training A) - Acc(A after training B)
```

这个数据集和任务划分属于本项目的**实验协议**。相关持续学习论文使用过 TwoPatterns，但我们**不宣称**在没有复现实验之前已经得到某篇论文的具体准确率。先跑实际基线，再确定未来的四维切片结构与对照指标。

## 数据学术引用

Dau et al. (2019), *The UCR Time Series Archive*, IEEE/CAA Journal of Automatica Sinica, DOI: 10.1109/JAS.2019.1911747。

## 后续准备讨论

1. 是否要先训练固定种子的普通 MLP 对照并量化 A 的遗忘程度？
2. 四维结构是每个 Task 一张切片，还是每个 **Feature** 一个位置？
3. 是否冻结旧表征？如何测量跨切片特征复用？

**尚未运行正式训练，不填写任何猜测的最终准确率。**
