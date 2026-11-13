# N1 实验结果说明

本目录记录 N1 实验：在尽量保持 E2a 训练与评估设置不变的前提下，将基座模型从 **Qwen2.5-7B-Instruct** 替换为 **Qwen2.5-14B-Instruct**，检验扩大模型规模能否稳定提升 Amazon ESCI 四分类任务的表现。

## 当前结论

N1 的三个正式种子均已完成训练、tune 偏置选择和 test 评估：

| Seed | Test raw Macro F1 | Test prior-adjusted Macro F1 |
| ---: | ---: | ---: |
| 42 | 0.542281 | 0.564195 |
| 43 | 0.523131 | 0.564639 |
| 44 | 0.554430 | 0.561737 |
| **Mean** | **0.539947** | **0.563523** |

Prior-adjusted Macro F1 的三种子范围为 **0.561737–0.564639**，三个种子的输出格式有效率均为 100%。结果初步显示 14B 版本表现较好且种子间较稳定。

**统一协议后的对比（2026-11-13）**：E2a 三个 adapter 已在 CCDS 上用与 N1 相同的 NF4、bf16 compute、double quantization 协议重新打分（报告为本目录的 `e2a-dq-seed*-prior.json`，过程见 DEVLOG S27 结果）。

| 指标（加先验调整） | 7B E2a-dq（42 / 43 / 44） | 14B N1（42 / 43 / 44） | 平均差 | 种子范围 |
| --- | --- | --- | ---: | --- |
| full Macro F1 | 0.5468 / 0.5523 / 0.5443 | 0.5642 / 0.5646 / 0.5617 | +1.57 点 | 不重叠 |
| merged Macro F1（E+S 合并） | 0.5834 / 0.5907 / 0.5774 | 0.6063 / 0.6041 / 0.6050 | +2.13 点 | 不重叠 |
| accuracy | 0.7307 / 0.7288 / 0.7297 | 0.7330 / 0.7426 / 0.7491 | +1.18 点 | 不重叠 |
| raw full Macro F1（不加偏置） | 0.5320 / 0.5188 / 0.5318 | 0.5423 / 0.5231 / 0.5544 | +1.24 点 | 重叠 |

按 DEVLOG S27 的预注册主标准（tune 上选偏置后的 test Macro F1，三种子范围不重叠），**14B 稳定优于 7B**，并且在 merged 口径上同样成立；提升主要来自 complement（每类 F1 平均 0.337 → 0.387）。不加偏置时两组范围重叠。clean 口径、query-level bootstrap 和 ranking 指标仍待补，需要 N1 的逐条 label scores（见下文）。

## 实验设置与实际偏差

- 代码 commit：`b6db723e7547a769385e297318b4de4214573628`
- 模型：`Qwen/Qwen2.5-14B-Instruct`
- 模型 commit：`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`
- 训练种子：42、43、44
- 训练轮数：1 epoch，每个种子 625 optimization steps
- 有效 batch size：32
- 评估集：6,000 条 tune、10,000 条 test
- 量化评估：bitsandbytes 4-bit NF4、bf16 compute、double quantization enabled
- 实际 GPU：NVIDIA GeForce RTX 5090 32 GB

原计划使用 A100 80 GB，但实际服务器为 RTX 5090 32 GB。为适配显存，micro-batch 调整为 1，gradient accumulation 调整为 32，有效 batch size 仍保持 32。这个硬件和 micro-batch 偏差需要在最终报告中披露。

三次训练合计记录约 **7.06 GPU-hours**。训练与评估均正常完成，日志中未发现 OOM、NaN、Traceback 或失败运行。

## 目录内容

| 文件 | 用途 |
| --- | --- |
| [`RESULTS.md`](RESULTS.md) | 完整实验摘要、训练记录、指标解释和结论边界 |
| [`training_summary.csv`](training_summary.csv) | 便于查看和导入表格的三种子训练摘要 |
| [`training_summary.json`](training_summary.json) | 机器可读的三种子训练摘要 |
| [`n1-seed42-prior.json`](n1-seed42-prior.json) | Seed 42 的完整指标、per-class P/R/F1、混淆矩阵和偏置曲线 |
| [`n1-seed43-prior.json`](n1-seed43-prior.json) | Seed 43 的完整指标、per-class P/R/F1、混淆矩阵和偏置曲线 |
| [`n1-seed44-prior.json`](n1-seed44-prior.json) | Seed 44 的完整指标、per-class P/R/F1、混淆矩阵和偏置曲线 |
| [`data-manifest.json`](data-manifest.json) | 数据规模、标签分布和 SHA256 provenance |
| [`model-revision.json`](model-revision.json) | Hugging Face 模型版本 provenance |
| [`e2a-dq-seed42-prior.json`](e2a-dq-seed42-prior.json) 等 3 个 | 7B E2a 按同一 double-quant 协议重打分后的报告（CCDS，格式同上） |
| [`TEMPLATE.md`](TEMPLATE.md) | 最初预注册的 N1 报告结构与判定标准 |

逐样本 label scores、预测缓存、完整训练日志和 adapter 权重没有提交到 Git：这些文件体积较大或可由现有脚本重新生成。原始结果归档应由实验负责人另行保留。

## 与旧 E2a 结果的关系

旧 E2a 三种子（未开 double quantization）的 prior-adjusted Macro F1 为 0.549287 / 0.553800 / 0.540541；开启 double quantization 后为 0.5468 / 0.5523 / 0.5443，三种子平均几乎不变（0.5479 → 0.5478），预测一致率 raw 约 98.8%、prior 约 96.5%。所以统一协议没有改变 7B 的水平，上表的差距来自模型规模。

## 还缺什么

1. **N1 的逐条 label scores**（`n1-seed{42,43,44}-{tune,test}.jsonl`）：有了它们才能做 clean 口径、query-level bootstrap 95% CI 和 ranking（AUC / AP）分析。7B 一侧的逐条打分已在 CCDS 上齐备。
2. N0 gate 或团队批准的 waiver，以及 N1 smoke-test 的原始证据。

## 快速复核

```bash
# 查看核心汇总
cat reports/n1/training_summary.csv

# 提取三个种子的 test 指标
for report in reports/n1/n1-seed*-prior.json; do
  jq '{raw: .test_argmax_no_bias.macro_f1, adjusted: .test_adjusted.macro_f1}' "$report"
done

# 验证所有 JSON 均可解析
for file in reports/n1/*.json; do
  jq empty "$file" || exit 1
done
```

如需引用结果，请优先引用 `RESULTS.md` 中的实验身份和结论边界，不要只复制单个最高 seed 的结果。
