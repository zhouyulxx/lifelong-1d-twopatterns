# 当前实验：原生整数 EEG 序列（2026-11-12）

**已经切换当前主线**：原先的 UCR TwoPatterns 是连续浮点序列，不能直接验证整数 Index–Position Embedding。旧实验和结果保留用于历史对照，但不是新的晶格实验基线。

新数据：UCI Epileptic Seizure Recognition，11,500 条、每条 178 个原生整数数值、五类；已逐项验证 2,047,000 个值均为整数，原 CSV gzip 无损保存于 `data/eeg_integer/data.csv.gz`。

新模型：**每个输出类只有一张二维 Index–Position 矩阵（窗口 30）**，两个类别共两张矩阵，没有额外 MLP 或 Linear Head。先训练任务 A，再训练 B，最后测试 A。

**三个种子实测：A 平均从 96.88% 降至 59.82%，遗忘 37.07 个百分点。** B 的平均准确率 65.98%；B-only 对照为 65.00%，因此 B 本身对当前极简模型较难。

- [完整中文基线报告](docs/INTEGER_EEG_TABLE_BASELINE_2026-11-12.zh-CN.md)
- [整数版训练脚本](run_integer_eeg_tables.py)
- [原生整数数据来源与校验](data/eeg_integer/SOURCE.json)
- [Seed 42 实测结果](results/eeg_integer_table_seed42/metrics.json)

---
## 历史实验：UCR TwoPatterns（连续浮点输入，不适用于原生整数模型）


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
