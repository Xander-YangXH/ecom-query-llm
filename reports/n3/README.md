# N3：7B 全参数微调（NSCC ASPIRE2A）

过程、问题与结论见 [DEVLOG S32](../../docs/DEVLOG.md)；对照组是 NSCC 上的 E2a 三个种子（[`reports/nscc/`](../nscc/)，DEVLOG S31）。

- 设定：Qwen2.5-7B-Instruct 全参数 bf16，4 × A100 40GB + DeepSpeed ZeRO-3；数据、轮数、全局 batch 32、按验证损失选 checkpoint 与 E2a 相同；打分用 bf16 完整模型，先验调整的偏置在调参集上选。
- 学习率在 seed 42 上扫 1e-6 / 2e-6 / 5e-6 / 1e-5 / 2e-5，按调参集宏 F1 选 5e-6，再跑 seed 43、44。
- S34 又加了种子 45、46、47（同一学习率），用于集成实验；S33/S34 的集成结果见 `reports/s33-*.json`、`reports/s34-*.json`。
- 结果（先验调整后，均值 [范围]）：宏 F1 0.5524 [0.5481, 0.5569]，E2a 为 0.5487 [0.5372, 0.5568]；merged 0.5910 vs 0.5871；准确率 0.7317 vs 0.7284。范围都重叠，不是稳定提升。

文件：`n3-full-lr<学习率>-s<种子>-prior.json`，每个是 `ecomq.eval.prior_adjust` 的报告（选出的偏置、调参集前后宏 F1、测试集 raw 与先验调整后的逐类指标、混淆矩阵与偏置曲线）。

复现：

```bash
LR=5e-6 SEED=42 bash scripts/pbs/run_n3.sh   # NSCC 登录节点，只提交作业
REMOTE_HOST=<user>@<host> REMOTE_ROOT=scratch/ecomq bash scripts/local_cache.sh n3-full-lr5e-6-s42
```
