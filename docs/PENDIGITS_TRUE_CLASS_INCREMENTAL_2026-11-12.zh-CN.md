# 当前 Lifelong Learning 主实验：UCI PenDigits 原生整数序列

**日期：2026-11-12**

## 研究问题

我们要区分三个不同问题：

1. 当训练数据只含单个 Class 时，一张整数 Index–Position 2D Matrix 能否学习有用的特征？
2. A 的类别已学会、B 的类别也可学会的前提下，A→B 无回放的 Class-Incremental 学习是否发生遗忘？
3. 把每个类别的 2D Matrix 旋转进 3D 晶格，是否真正获得知识复用收益，还是只是隔离存储？

第 1、2 项本次取得了实测数据。第 3 项仅验证了**无交叠旋转空间映射的功能等价性**，尚未证实有益的共享/迁移。

## 原始数据和实验限制

公开 [UCI Pen-Based Recognition of Handwritten Digits (dataset 81)](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)。

- 7494 个原始 TRAIN 样本、3498 个原始 TEST 样本（原始 UCI writer-separated split），10 个数字类 0..9。
- 每样本为 **16 个原生整数**（8 个按序排列的 `(x,y)` 手写轨迹点）；取值 0..100。
- **没有数值分箱、取整、插值或重新划分 TEST。**
- 每个位置的含义不同：交替为 x、y；这不是单通道的 packet-size 时间序列，因此只能作为有序整数特征基准，不能声称它与 IoT 单通道数据同构。
- 原始 ZIP checksum 及 TRAIN/TEST checksum：`data/pendigits/SOURCE.json`。
- 原始训练和测试文件以 gzip 无损储存 `data/pendigits/pendigits.tra.gz`、`pendigits.tes.gz`。

## I. 单类别学习：每类一张 2D 矩阵

对于每个类别 `c`，学习一张 \(W_c\)：

```
W_c.shape = [101 integer symbols, 16 positions]
```

单类别训练只需要该类别的正样本，记录每个位置的原始整数值的平滑频率：

\[
W_c[v,r] = \log\frac{N_c(v,r)+1}{N_c(r)+101}
\]

其输出为类条件分布的每位置对数似然；预测时在**所有已见类别**之间比较 `sum_r W_c[x_r,r]`，不需要 Task ID。

注意：这是从论文的 Index–Position 查表计算借鉴出来的**单类别生成式改版**，而非论文原始的 Softmax 分类 Cross-Entropy 训练。

学习顺序 \(0\to1\to2\to\dots\to9\)，每次加入新类，只新增该类的一张表，旧表不修改。

**第一阶段只有 0 这一类时，closed-set 识别率没有意义**（任何模型都能输出 0 达到 100%）。改用 0 和从未见过的类 1 TEST 样本作 OOD 排序诊断：AUROC = **0.9524**（此指标不用于训练或挑选模型）。

| 已学类 | 所有已见类别的测试准确率 |
|---|---:|
| 0,1 | 96.70% |
| 0,1,2 | 84.05% |
| 0,1,2,3 | 85.91% |
| 0..4 | 88.16% |
| 0..5 | 83.77% |
| 0..6 | 84.12% |
| 0..7 | 83.62% |
| 0..8 | 83.71% |
| 0..9 | **82.45%** |

结果源：`results/pendigits_generative_seed42/metrics.json`。

对于最初的 A = {0,1}：在只有这两个类别时，A 测试准确率 96.70%；学完 B = {2,3} 以后，在全 4 类竞争中 A 正确数 = 553/727，即 **76.07%**，下降 **20.63 个百分点**。

同一时候 B = {2,3} 的正确数 = 673/700，正确率 **96.14%**。全 4 类总正确率 = 1226/1427，**85.91%**。

**尽管旧类别表的参数一个都没修改，A 的准确率仍下降：原因是新类别得分参与了全局竞争。这不是旧参数被覆盖引起的遗忘，但确实是新增类别后的功能性性能退化。**

这两次所谓 seed 42/43 对生成式结果完全一样，因为它是固定官方划分、无随机训练的确定性统计法；**绝不能当成多随机种子独立重复实验**。

## II. 判别式 class-incremental 标准基线

同一数据，严格保持每个输出类一张 \(W_c\in R^{101\times16}\)：

- Task A = 原始 digits {0,1}
- Task B = 原始 digits {2,3}
- 第一阶段只激活两个已见类别输出；第二阶段激活四个输出。
- B 阶段不回放 A 的训练样本；也不重置旧模型权重。
- 使用统一 4 类的 global argmax 预测；**不提供 Task ID**。
- 训练方式：两阶段各 30 epochs、Adam、lr=0.01、Cross-Entropy。
- 另测试 B-only (仅学习 2,3) 和联合训练 A+B 作为能力控制。

| 实验结果 | Accuracy |
|---|---:|
| 学完 A，测 A | **97.25%** |
| 学完 B，重新测 A (四类竞争) | **11.28%** |
| 学完 B，测 B (四类竞争) | **99.57%** |
| B-only，测 B (两类竞争) | **99.57%** |
| A+B joint，同步学习四类，测全部四类 | **92.01%** |
| A+B sequential，学完 B 测全部四类 | **54.59%** |

遗忘量：

\[
F_A = 97.25\% - 11.28\% = 85.97 \text{ pp}
\]

A 的 TEST 原有 727 条（0 为 363，1 为 364），A 在 after B 只有 82 条正确；多数旧样本被预测成新类。说明没有 Task ID 的增量分类在正常训练下发生严重失效。

因为 A-only 97.25%、B-only 99.57%、joint 92.01%，**这个类别组合满足我们要求的两任务都可以学习的测试条件**。比之前 EEG B-only ~65% 的组合更合适。

结果文件：

- `results/pendigits_sequential_seed42/metrics.json`
- `results/pendigits_B_only_seed42/metrics.json`
- `results/pendigits_joint_seed42/metrics.json`

这些是 Seed 42 的首次实测；判别式结果还需要多随机种子验证。

## III. 3D 旋转矩阵（空间布局实验）

物理空间坐标 `(v,y,z)`：

- `v` 始终为整数索引；`r` 始终为序列位置，不交换含义。
- 每新学一个 class，位置轴在 `y-z` 平面旋转 90°。
- 每个新类的平面中心做足够远的空间平移，**保证不同类别的表不相交**。
- 每张 2D Table 的所有 101×16 个值都实存到有精确 `(v,y,z)` 位置的稀疏 `geometry_store`，并断言从空间坐标读取可完全还原原矩阵。
- 推理必须比较当前全部已见类别的 Table 得分，而不能预先选择真实 class 平面。

因此当前旋转版与独立保存全部 class 矩阵**预测完全等价**。这证实了空间存储编码正确，但**不能把性能提升归因于旋转，也没有证实正向特征迁移**。

真正的后续任务是让不同空间表之间允许受控相交、共享参数或演化式复用，并与独立表在相同训练/推理目标下比较。

## 为什么两套结果不宜直接称为“旋转优于基线”？

判别式基线使用 CE 的标签竞争梯度；单类别模型使用每类单独拟合的对数频数。这是**不同的学习目标**，因此 85.91% vs 54.59% 不能单独支持“空间旋转带来 31 个百分点提升”。

必须增加生成式独立存储 vs 生成式旋转存储（已知二者等价）和未来相交共享 vs 不共享的匹配控制。

## 复现

```bash
python scripts/import_uci_pendigits.py
python run_pendigits_incremental.py --mode generative --seed 42
python run_pendigits_incremental.py --mode sequential --seed 42 --epochs 30
python run_pendigits_incremental.py --mode B_only --seed 42 --epochs 30
python run_pendigits_incremental.py --mode joint --seed 42 --epochs 30
```

GitHub Actions 记录：https://github.com/zhouyulxx/lifelong-1d-twopatterns/actions/runs/43195787385

## 建议的下轮实验

1. 基线 seed 42/43/44 等多随机初始化复测，报告平均与标准差。
2. 选择更简单的 4-class A/B 作为晶格机制初期试验，但保留 10-class 全流程用于扩展性测试。
3. 同一学习目标下比较 3D 平行隔离、旋转隔离和受控交点共享，明确对 A retention、B learning 和存储量的影响。
4. 研究**从 A 的单类别表中抽取可共享信息，而不改写 A 对其自身的类别得分**，防止新类竞争导致旧类混淆。

## 追加核验：三个随机种子的全局四分类遗忘（2026-11-12）

复核了标准判别式 A={0,1}→B={2,3} 无 replay，全局四类输出的三个随机初始化：

| Seed | A 学完测 A | B 学完再测 A | B 学完测 B | A 遗忘 (pp) |
|---|---:|---:|---:|---:|
| 42 | 97.25% | 11.28% | 99.57% | 85.97 |
| 43 | 97.25% | 13.07% | 99.43% | 84.18 |
| 44 | 97.11% | 11.14% | 99.57% | 85.97 |
| **平均** | **97.20%** | **11.83%** | **99.52%** | **85.37** |

三个种子都出现强烈旧类遗忘，而 B 始终学得很好。样本划分沿用官方固定 TRAIN/TEST，所以种子变化仅来自网络参数初始化、batch shuffle 等随机训练过程。完整机器生成数值见 `results/pendigits_sequential_seed42/metrics.json`、`results/pendigits_sequential_seed43/metrics.json`、`results/pendigits_sequential_seed44/metrics.json`。

GitHub Actions：https://github.com/zhouyulxx/lifelong-1d-twopatterns/actions/runs/43196301376
