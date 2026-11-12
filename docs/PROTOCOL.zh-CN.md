# 实验协议：两任务一维序列持续学习

数据：UCR TwoPatterns 原始训练/测试切分（1,000/4,000），长度 128，类别 4。

## Task
- A：排序后前两个原始类别 → 全局标签 0 和 1
- B：后两个原始类别 → 全局标签 2 和 3
- 序列均作逐样本 z-normalization
- 原始测试集只用于评价，从不参与训练

## Baseline
- MLP 128 → 128 → 128 → 4，ReLU，Adam
- A 训练后保存 checkpoint，直接在同一权重上接着训练 B
- B 训练不提供 A 的样本；无 replay，无 task-specific head
- 两任务评估时统一在四个输出类别做 argmax，不提供 Task ID
- 另外报告 task-aware restricted argmax 仅作诊断

## 指标
- A_after_A：学 A 后对 A 测试集的四分类准确率
- A_after_B：学 B 后对 A 测试集的四分类准确率
- B_after_B：学 B 后对 B 测试集的四分类准确率
- Forgetting_A = A_after_A - A_after_B

注意：如果只报告统一四分类遗忘，不能据此区分特征遗忘与最后层输出竞争。必须同时检查诊断分数、混淆矩阵和模型权重。

## 数据可追溯
完整 TRAIN/TEST 数据已入库；原始文件来源与 SHA-256 见 data/TwoPatterns/SOURCE.json。
数据经脚本核对：训练 1,000 条、测试 4,000 条、长度 128、四个类别。

## 未做的事
尚未在原始数据上运行正式对照、未对参数调优、未实现四维晶格结构。
