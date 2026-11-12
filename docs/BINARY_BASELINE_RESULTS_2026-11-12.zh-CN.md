# 修正版基线：A → B → 再测 A（2026-11-12）

## 为什么需要这个实验

最初的四分类增量实验，学习 B 后 A 的 **global 4-way accuracy = 0%**。但仅在 A 的两个旧输出间比较时还有 **72.63%**，所以无法将 0% 直接解释为旧特征全部丢失。

修正版完全复用同一个 MLP 和同一个 **2 输出**分类头，重新训练并检测 A 旧任务的二分类准确率。

## 实验数据

UCR TwoPatterns 官方 TRAIN=1,000、TEST=4,000；每样本序列长度=128；官方原始类标签 `1,2,3,4` 映射为内部 `0,1,2,3`。

- Task A：内部类别 `0,1`，输出标签 `0,1`；TRAIN=508、TEST=2,046。
- Task B：内部类别 `2,3`，重映射输出标签 `0,1`；TRAIN=492、TEST=1,954。
- 网络：MLP，128→128→128→2（ReLU）。
- 固定种子 42；A 和 B 各 20 epochs；Adam，lr=0.001；batch_size=64。
- 训练 B 时不重置模型参数、不给模型 A 训练数据。两个阶段均使用**相同的 2 个输出神经元**，任务标签只用于选择测试集和事先映射标签。
- 不存在额外的 Task Router、Replay 或训练后的任务专属 head。
- 所有测试数据来自原始 TEST split；每个序列独立 z-normalization。

## GitHub Actions 的实测结果

| 训练完成后 | 测试 A | 测试 B |
|---|---:|---:|
| A | 94.3304% | 67.1955% |
| B | 78.4457% | 93.1423% |

**A 的遗忘：15.8847 个百分点**。不是旧四分类实验的 93.65%→0%。

### A 的二分类混淆矩阵（按行真实标签、按列预测标签）

学完 A：

```text
             pred 0  pred 1
actual 0        990      45
actual 1         71     940
```

学完 B：

```text
             pred 0  pred 1
actual 0        988      47
actual 1        394     617
```

可见 A 的 label 1 更容易被识别为 0，说明存在 Task B 训练带来的干扰。

## 怎么复现

```bash
pip install -r requirements.txt
python run_binary_baseline.py --device cpu --epochs-a 20 --epochs-b 20 --seed 42
```

结果位于 `results/binary_seed42/`：

- `metrics.json`：所有实测数值及混淆矩阵
- `accuracy_matrix.csv`：A、B 两个阶段的完整测试矩阵
- `train_history.csv`、`train.log`：训练日志
- `model_after_A.pt`、`model_after_B.pt`：训练保存的模型检查点（在 Actions Artifacts；未必长期保留）

训练运行：[GitHub Actions Run 43177711356](https://github.com/zhouyulxx/lifelong-1d-twopatterns/actions/runs/43177711356)。

## 对结果的限制

1. 这是一组固定种子 42 的实测结果，并不是多种子均值。
2. 采用 **task-wise binary** 目标映射；它不是无需任务背景的原始四类增量分类指标，不能和先前的 class-incremental 0% 直接比较。
3. A 准确率下降证明任务干扰，但并不能凭这个数字单独证明底层特征具体丢失了多少。
4. 此实验尚未使用 4D 切片结构，也尚未进行结构学习/特征复用对照。
