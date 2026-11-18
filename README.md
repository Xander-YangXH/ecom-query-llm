# ecom-query-llm

电商搜索相关性判断的大模型微调：在 Amazon ESCI 数据集上，用 QLoRA 微调 Qwen2.5-7B/14B-Instruct，判断 query 与商品之间的关系（exact / substitute / complement / irrelevant 四分类），并系统地检查每一项改进在统计上和标注噪声面前是否站得住。7B 主线实验运行于单卡 L40S；N1 的 14B 三种子实验运行于单卡 RTX 5090 32 GB。

每一步的动机、做法、结果和遇到的问题都记录在 [docs/DEVLOG.md](docs/DEVLOG.md)；N1 的运行方法和结果分别见 [AutoDL 指南](docs/N1_AUTODL_GUIDE.md)与 [N1 结果说明](reports/n1/README.md)。完整的课程报告（全部实验结果、结论与局限）见 [REPORT.md](REPORT.md)。

## 任务

- 数据：[Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data)，美国英文、`large_version=1`。按 query 分组抽样，训练集 5,000 个 query（2 万条），测试集 2,500 个 query（1 万条），另有 1,500 个 query 的调参集；三者没有共同的 query（官方 train/test 之间唯一共享的 query 也已剔除，见 P13）。
- 输入：query 与商品标题。输出：一个 JSON，例如 `{"relevance": "complement"}`。指令、标签定义与解析规则见 [src/ecomq/schema.py](src/ecomq/schema.py)。
- 主指标：宏 F1（exact 约占 67%，只看准确率会被它主导）；同时报告准确率。所有差异都用按 query 分组的 bootstrap 给出 95% 置信区间，关键结论至少 3 个随机种子。

## 主要结果

2,500 个 query 的测试集：

| 方案 | 准确率 | 宏 F1 | 说明 |
|---|---|---|---|
| 基座零样本 | 0.550 | 0.384 | |
| E1：QLoRA，2 个 epoch | 0.727 | 0.534 | |
| **E2a：1 个 epoch，按验证损失选 checkpoint** | **0.739** | 0.535 | 默认训练设定，训练时间减半 |
| E2b：训练时重采样 C/I | 0.721 | 0.545 | 少数类召回升、精确率降 |
| **E2a + 推理时先验调整** | 0.730 | **0.549** | 推荐方案：4 个标签的对数概率加类别偏置，偏置只在调参集上搜索；宏 F1 +1.45（显著），不用重训 |

其余实验（数据量曲线、丰富商品输入、追加 C/I 真实样本、软标签、只追加 complement、先推理再分类）和它们的结论见 DEVLOG 的 S13–S24。简要地说：

- **加数据、加商品文本带来的准确率提升，几乎都发生在标注有争议的样本上**；换成“去掉争议样本”（clean）和“E/S 合并”（merged）两种口径后基本消失（S18）。
- **改训练集的类别比例主要是在改变先验倾向**，与推理时调偏置大部分可以互相替代；只加 complement 会伤害 irrelevant，C/I 各加 3,000 会伤害 E/S 区分（S19、S20）。C/I 各追加 1,500 条（E10）是唯一让两类的排序质量同时提高、E/S 不受损的设定（complement AP 0.322 → 0.347，irrelevant 0.473 → 0.480），但加先验调整后的宏 F1（0.551）仍在 E2a 的种子范围内（S24）。
- **用 7B 模型自己写的理由做“先推理再分类”的 SFT 明显有害**（宏 F1 0.363），因此没有继续做 GRPO（S21–S23）。
- **D1–D5（S26）**：微调后 MMLU 没有净变化（0.735 → 0.735）；LoRA 默认设定（r=16，全部线性层）合适，只训练 q、v 稳定变差 3.2 个百分点；蒸馏到 3B（人工标签）宏 F1 0.517，配合 vLLM 与“只看第一个不同 token”的打分每秒 399 条，是教师现行打分的 47 倍；中文 Multi-CPR 重排 MRR@10 从零样本 0.256 提高到 0.346；意图 / 属性 / 改写多任务在 7B 自标注下没有带来提升。
- **标注审计**：E↔S 之间约 6.8% 的测试样本标注有争议（S17）；complement / irrelevant 方面，盲评估计约 4% 的测试样本粗粒度标注（相关 / 搭配 / 无关）有误，其中标注为 irrelevant 的样本约四分之一更像是相关的（S24）。两项都是 Claude 的盲评，人工复核表在 `analysis/`。

### N1：14B 容量实验

N1 仅把 E2a 的基座替换为 Qwen2.5-14B-Instruct，并尽量冻结数据、prompt、QLoRA 超参数、有效 batch 和评估方法。三个正式种子的 test prior-adjusted Macro F1 为 **0.564195、0.564639、0.561737**，平均 **0.563523**，范围 **0.561737–0.564639**。

E2a 的三个 adapter 已在 CCDS 上按 N1 相同的协议（NF4、bf16 compute、double quantization）重新打分（DEVLOG S27 结果）。double quantization 对 7B 本身几乎没有影响（三种子平均 0.5479 → 0.5478）。同一协议下：

| 指标（加先验调整） | 7B E2a-dq 范围 | 14B N1 范围 | 平均差 |
| --- | --- | --- | ---: |
| full 宏 F1 | [0.5443, 0.5523] | [0.5617, 0.5646] | +1.57 |
| merged 宏 F1 | [0.5774, 0.5907] | [0.6041, 0.6063] | +2.13 |
| accuracy | [0.7288, 0.7307] | [0.7330, 0.7491] | +1.18 |

三项的种子范围都不重叠，按预注册标准算稳定提升，主要来自 complement（F1 +5 点）。不加偏置（raw）时两组范围重叠。N1 只交回了汇总报告、没有逐条打分，因此这一对比没有 clean 口径、bootstrap 置信区间和排序指标，结论只覆盖上表三项。详细指标、实验偏差和文件导航见 [`reports/n1/README.md`](reports/n1/README.md)。

**N3：7B 全参数微调（NSCC，DEVLOG S32）。** 4 × A100 40GB + DeepSpeed ZeRO-3，数据、轮数、全局 batch 与 checkpoint 选择同 E2a，只把 QLoRA 换成全参数 bf16 训练。学习率在 seed 42 上扫 1e-6 到 2e-5 五档，按调参集选出 5e-6，再跑 seed 43、44。对照组是 NSCC 上的 E2a 三个种子（S31）：

| 指标（加先验调整） | 7B QLoRA E2a（NSCC） | 7B 全参数 N3（lr 5e-6） | 平均差 |
| --- | --- | --- | ---: |
| full 宏 F1 | 0.5487 [0.5372, 0.5568] | 0.5524 [0.5481, 0.5569] | +0.37 |
| merged 宏 F1 | 0.5871 [0.5724, 0.5967] | 0.5910 [0.5864, 0.5971] | +0.39 |
| accuracy | 0.7284 [0.7251, 0.7340] | 0.7317 [0.7290, 0.7360] | +0.33 |

三项的种子范围都重叠，按预注册标准**不是稳定提升**；全参数微调只让种子之间的波动变小。7B 全参数（训练全部 76 亿参数）仍低于 14B QLoRA（N1，[0.5617, 0.5646]），和 D2（LoRA 秩加到 64 无提升）、N2（bf16 LoRA 无稳定提升）一致：这个任务上 7B 的瓶颈不在可训练参数量，换更大的基座更有效。报告见 [`reports/n3/`](reports/n3/)。

**S33：种子集成与 bagged 偏置（离线）。** 调参集只有 88 条 complement，单次搜出的偏置在种子之间波动很大，是种子差距的主要来源之一。改为按 query 重采样调参集 50 次后平均偏置（`prior_adjust --n-bag 50`），E2a 单个种子的宏 F1 范围从 1.96 个百分点缩到 0.96。在此基础上，E2a 的 3 种子集成只高 0.2 个百分点，不值得 3 倍推理成本；N3 的 3 种子集成高 1.6（配对 bootstrap CI 不含 0），宏 F1 0.569、merged 0.607。报告见 `reports/s33-ensembles*.json`。

**S34：N3 再加 3 个种子确认。** 新的 3 个种子（45–47）组成的集成也比成员高 1.3（CI 不含 0），提升复现。N3 与 E2a 各 6 个种子、各 20 种 3 种子组合：N3 均值 0.5660，E2a 均值 0.5507。按预注册标准（bagged 偏置下 N3 最小值 > E2a 最大值）以 0.5555 对 0.5551 **勉强成立**，单次搜偏置下不成立。6 个 N3 种子全部集成达到宏 F1 0.5703、merged 0.6066，是项目最好结果，代价是 6 倍的 7B 推理。报告见 `reports/s34-n3-pool*.json`。

## 方法要点

- **训练**：LLaMA-Factory 0.9.5，QLoRA（4bit NF4，r=16，alpha=32，全部线性层，可训练参数 0.53%），只对回复计损失；训练与推理的提示词由同一个函数生成，并有单测锁定（P1）。
- **评测时打分而不是生成**：对 4 个候选回复分别计算 log p，取最大即预测；同一份打分支持先验调整、召回–精确率曲线和与偏置无关的排序指标（[src/ecomq/eval/score_labels.py](src/ecomq/eval/score_labels.py)）。QLoRA 适配器必须配 4bit 基座评测（P8）。
- **推理时先验调整**：`argmax(log p + b)`，`b_exact = 0`，其余三类的偏置在调参集上用坐标上升最大化宏 F1（[src/ecomq/eval/prior_adjust.py](src/ecomq/eval/prior_adjust.py)）。
- **噪声感知的评测**：clean 口径的争议集由**不参与比较的模型**组成的委员会定义，避免把被比较模型自己的共同错误从考卷上删掉（[src/ecomq/eval/noise_aware.py](src/ecomq/eval/noise_aware.py)）。
- **与偏置无关的排序指标**：每一类直接用 log p(c) 排序计算 AUC / AP；softmax 后的概率并不与偏置无关（P15，[src/ecomq/eval/ranking.py](src/ecomq/eval/ranking.py)）。
- **事先写下判断标准**：从 E8 起，每个实验在运行前把“什么结果算成功”写进 DEVLOG。

## 目录

```
configs/sft/          LLaMA-Factory 配置：e2_relevance_qlora_1ep_best.yaml 为默认（E2a），e9a_* 为推理格式
src/ecomq/schema.py   标签、指令、输出格式与解析、校验
src/ecomq/data/       prepare_esci（抽样与划分）、enrich_esci（商品附加字段）、build_e7 / build_e9（各实验的数据）、
                      make_rationales（推理理由）、to_alpaca（转 LLaMA-Factory 格式）
src/ecomq/eval/       score_labels（标签打分）、score_labels_cot、prior_adjust、compare（bootstrap）、
                      noise_aware、ranking、metrics、generate_hf
src/ecomq/analysis/   es_audit（标注审计：抽样、生成 Excel 盲评表、评分）
scripts/slurm/        CCDS 作业脚本与各实验的调度脚本（run_e*.sh）
scripts/pbs/          NSCC ASPIRE2A 的 PBS 作业（单卡训练、4 卡全参数训练、打分）与 run_n0.sh / run_n3.sh
scripts/autodl/       N1 单卡 AutoDL：环境检查、数据审计、模型下载、smoke/N0 gate、三种子训练与统一评估
scripts/              环境安装、模型与数据下载、代码同步（remote.sh）、本地分析缓存（local_cache.sh）、提交前检查（check_leaks.sh）
analysis/             es_audit（E↔S）、ci_audit（complement / irrelevant）两份盲评材料
reports/              各实验的指标与对比 JSON（小文件，进 git）
reports/n1/           N1 三种子结果、训练摘要、数据/模型 provenance 与组员阅读入口
reports/nscc/         NSCC 上 E2a 三个种子（N0 复现 + 7B 对照组）
reports/n3/           N3 7B 全参数微调：五档学习率（seed 42）与 lr 5e-6 的三个种子
docs/                 DEVLOG、LABELS、WORKFLOW、N1 AutoDL 指南
demo/                 前端演示（静态网页与数据，build_data.py 生成数据）
REPORT.md             课程报告：全部实验结果、结论与局限
PRESENTATION.md       15–20 分钟展示的 PPT 要点建议
EXPERIMENT_RESULTS.md 组员的 N0 / N2 实验结果
B3_RESULTS.md, B3/    组员的 B1–B3 人工复核结果与评分文件
tests/                单测（当前 72 个）
```

数据、模型和训练产出只放在服务器上，路径通过 `.env` 配置（CCDS 见 `.env.ccds.example`，AutoDL/N1 见 `.env.autodl.example`），不进 git。

## 复现

本地：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test
```

CCDS（SLURM）上的主结果，按顺序在登录节点用 `setsid nohup` 运行调度脚本（它们只提交和等待作业，不在登录节点计算）：

| 步骤 | 脚本 | 产出 |
|---|---|---|
| 环境、模型、数据 | `scripts/setup_env_ccds.sh`、`scripts/download_model.sh`、`scripts/download_data.sh` | conda 环境、`$MODEL_ROOT`、`$DATA_ROOT/raw/esci` |
| E1 数据 | `sbatch scripts/slurm/prepare_data.sbatch`（`prepare_esci` 默认参数） | `$DATA_ROOT/e1_relevance`（E2a 也用这份数据） |
| E1 训练与评测 | `scripts/slurm/chain_e1.sh` | 适配器 `e1-qlora-r16` |
| E2a / E2b | `scripts/slurm/run_e2.sh` | 适配器 `e2a-1ep-best`、`e2b-rebal-1ep-best` |
| 2,500 query 测试集 | `scripts/slurm/run_r1.sh` | `$DATA_ROOT/esci_test2500` |
| 调参集与先验调整 | `scripts/slurm/run_e3.sh` | `$DATA_ROOT/esci_tune`、标签打分 `e2a-{tune,test}` |

本地分析：`REMOTE_HOST=<user>@<host> bash scripts/local_cache.sh e2a` 会拉取打分并重新生成先验调整的预测与报告（应与 `reports/` 中已提交的报告一致）；之后用 `ecomq.eval.compare`、`noise_aware`、`ranking` 做对比。每个实验的具体命令见 DEVLOG 对应章节。

### N1 / AutoDL

N1 的完整命令、断点续跑规则和门禁条件见 [`docs/N1_AUTODL_GUIDE.md`](docs/N1_AUTODL_GUIDE.md)。如果只需了解已经得到的结果，从 [`reports/n1/README.md`](reports/n1/README.md) 开始；机器可读指标和 provenance 均在同一目录。

E2a 三个 adapter（`e2a-1ep-best`、`e5-20k-s43`、`e5-20k-s44`）在 CCDS 上的统一协议重打分用 `bash scripts/slurm/run_e2a_dq.sh`（`label_scores.sbatch` 的 `DOUBLE_QUANT=1`）；在 AutoDL 上对应 `scripts/autodl/rescore_e2a_for_n1.sh`。结果报告为 `reports/n1/e2a-dq-seed*-prior.json`。

## 项目状态

已结项（2026-11-13）。CCDS 单卡 7B 主线（E1–E10、D1–D5、N0/N2）和 AutoDL 单卡 14B（N1）全部完成并写入 DEVLOG；不再安排后续实验。完整的课程报告见 [REPORT.md](REPORT.md)，展示要点见 [PRESENTATION.md](PRESENTATION.md)，前端演示在 `demo/`（`cd demo && python3 -m http.server 8000`，再打开 http://localhost:8000）。


### NSCC ASPIRE2A

2026-11-17 起后续实验在 NSCC 上运行（DEVLOG S30）。环境、PBS 脚本和复现结果：

```bash
cp .env.nscc.example .env               # 填好 NSCC_PROJECT
bash scripts/setup_env_nscc.sh          # 登录节点：conda 环境，版本与 CCDS 锁定文件一致
bash scripts/download_model.sh Qwen/Qwen2.5-7B-Instruct
ESCI_ONLY=1 RAW_DIR=$DATA_ROOT/raw bash scripts/download_data.sh
scripts/pbs/submit.sh gpu_check
scripts/pbs/submit.sh prepare_data      # 生成的数据 sha256 与 CCDS 完全相同
bash scripts/pbs/run_n0.sh              # E2a seed 42：训练 → 调参集 / 测试集打分
SEED=43 bash scripts/pbs/run_n0.sh      # 其他种子
LR=5e-6 SEED=42 bash scripts/pbs/run_n3.sh   # N3：4 卡全参数微调 → 打分（需要 .env 里的 CUDA_HOME 和 DeepSpeed）
```

E2a 三个种子在 NSCC 上全部重跑（DEVLOG S30、S31）：先验调整后宏 F1 均值 0.5487，范围 [0.5372, 0.5568]；CCDS 为 0.5479，范围 [0.5405, 0.5538]。两边均值只差 0.08 个百分点，单个种子的差别与种子之间的波动同一量级。NSCC 上的新实验以这三个种子为 7B 对照组。报告见 `reports/nscc/`。

## 数据说明

只使用公开的 ESCI 数据，不含任何真实线上日志。理由数据（E9）由开源模型生成，只用于实验。
