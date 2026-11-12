# 原生整数序列 Lifelong Learning 基线（2026-11-12）

## 研究目标

在一维**整数序列**上验证：只使用每个二分类输出类别一张 2D Embedding Matrix 的模型，学习 Task A → 学习 Task B → 再次测试 Task A，会不会发生遗忘？

这是用于后续 4D 晶格研究的**基线**，还没有加入 4D 结构。

## 数据核验

UCI *Epileptic Seizure Recognition*，公开镜像的原始 CSV 经逐值解析检查：

- 共 11,500 条，178 个时间点/序列，5 类，每类 2,300 条。
- 2,047,000 个 EEG 数值均满足整数性质；fractional_value_count = 0。
- 值域 -1885 至 2047；**没有分箱、取整或归一化**。
- 只对原始整数建立 index vocabulary，未出现过的测试值贡献 0。
- 来源、镜像 URL、原始字节 SHA-256：`data/eeg_integer/SOURCE.json`。
- 原始 CSV 使用 gzip **无损压缩**储存：`data/eeg_integer/data.csv.gz`。

> 注意：这份 EEG 数据本身是经过公开数据集预处理、分段的信号，而非原始模拟电极模拟电压。整数值是所下载来源中实际存储的数值，不是本项目量化得到的。

## 实验任务

- Task A：原始标签 1（癫痫发作）vs 5（睁眼）。
- Task B：原始标签 2（病灶区域）vs 4（闭眼）。
- 每任务分别映射输出标签 0/1，因此 **测试任务需要知道当前采用哪组标签语义**；这属于 task-wise binary continual learning，不是四类统一识别。
- 两任务在原始五类中没有重叠样本；训练 Task B 不使用 Task A 的训练集。
- 研究者自定义 A/B 配对；不是 UCI 发布的官方持续学习任务划分。
- 每个原始类按 seed 随机 80/20 分割，A 和 B 均为训练 3,680，测试 920。
- 目前 **不是按受试者划分**，同一原始记录的不同 EEG 分段可能同时存在 train/test，不能作为临床诊断验证。

## 精简模型

借鉴所附 2026 年 IoT Index–Position 论文 **查表聚合** 的核心，但这是刻意简化的架构：

- 每个输出类别只有一张 \(W_c\)，形状 \(|V| \\times 30\)。
- 无 MLP、CNN、Transformer、额外线性分类头。
- 输入整数 \(x\) 在训练词表中映射为 \(v(x)\)；每个重叠窗口提取每个位置的 \(W_c[v_r,r]\) 并求和，输出分类 logit。
- 对每窗口计算 Cross-Entropy；推理时平均每个窗口的 Softmax 概率。
- 使用 \(\\tanh(W_c)\) 稳定有效权重；每个窗口长度 30，步长 1。
- A 学习完后保留两张矩阵。学习 B 时，两张矩阵**继续更新**，新增整数只扩展新的词表行，旧整数索引不重排。
- Adam、lr=0.005、batch size=128，A/B 各 20 epochs。

Seed 42：A 词表 3,227 个整数，B 补充 41 个新整数，总词表 3,268；参数量 \(2 \\times 3268 \\times 30 + 2 = 196,082\)。

## GitHub Actions 的真实结果

| Seed | A→测A | A→B→测A | A→B→测B | A 遗忘 (pp) |
|---|---:|---:|---:|---:|
| 42 | 97.17% | 61.96% | 67.17% | 35.22 |
| 43 | 96.30% | 59.35% | 67.17% | 36.96 |
| 44 | 97.17% | 58.15% | 63.59% | 39.02 |
| **Mean** | **96.88%** | **59.82%** | **65.98%** | **37.07** |

标准差（样本）：遗忘量约 1.90 个百分点，三个种子都发生显著遗忘。

- Task B **单独从随机初始化训练**（seed 42）→ B 测试 65.00%。
- 对照说明 B 的低准确率主要是这个简单模型的判别限制，而不是只有 A→B 顺序训练才发生。
- 原始 Task A 学习效果足够强（约 97%），适合作为观测遗忘的初始条件。
- 目前不能说它实现了强 Task B 学习，也不能声称它证明 4D 晶格有效。

## 文件与复现

源数据与校验：`data/eeg_integer/SOURCE.json`、`data/eeg_integer/data.csv.gz`

主实验：

```bash
pip install -r requirements.txt
python run_integer_eeg_tables.py --seed 42 --device cpu --epochs-a 20 --epochs-b 20 --window 30 --lr 0.005
```

B 单任务对照：

```bash
python run_integer_eeg_controls.py --seed 42 --epochs 20
```

结果文件：
- `results/eeg_integer_table_seed42/metrics.json`
- `results/eeg_integer_table_seed43/metrics.json`
- `results/eeg_integer_table_seed44/metrics.json`
- `results/eeg_integer_B_only_seed42/metrics.json`

训练记录：
- https://github.com/zhouyulxx/lifelong-1d-twopatterns/actions/runs/43183984306
- https://github.com/zhouyulxx/lifelong-1d-twopatterns/actions/runs/43184359894

## 下一步设计方向（未实现）

1. 使用同一组输入和每类矩阵作为共同起点，建立新旧表征存储/共享方式的公平对照。
2. 不把「冻结全部旧矩阵」当作自动获得特征复用；必须另测 B 是否利用旧矩阵。
3. 如果需要更强的 B→分类准确率，应选另一个更可分的真实整数任务配对，并在**只用训练/验证数据**选定后锁定测试集。
4. 晶格架构必须同时报告 A retention、B learning、内存/参数增长量、计算开销。
