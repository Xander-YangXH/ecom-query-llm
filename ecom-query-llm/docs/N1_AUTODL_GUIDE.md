# N1 AutoDL 单卡工程与运行指南

本文只覆盖 N1（Qwen2.5-14B-Instruct QLoRA）及其必要的 N0 环境门禁，不实现 N2/N3/N4。N1 三种子训练与标准 full-test 评估已经完成；统一协议的 E2a 复评和稳健性分析仍待完成。

## 1. 审查结论

### 仓库与协作状态（更新于 2026-11-12）

- 最初工作目录不是 Git 仓库，而是 AI6130 Assignment 1 的旧产物。本项目已单独克隆为 `ecom-query-llm`。
- 实验仓库上游：`Xander-YangXH/ecom-query-llm`；N1 工作位于个人 fork `Pat7ryk/ecom-query-llm` 的 `feat/n1-autodl-qwen14b`。
- 最终提交仓库 `Xander-YangXH/NTU-AI6130-LLM-Project` 可访问。其 `main` 只有项目 README；成员通过 `liyouran`、`zhangchen` 远端分支交付结果。
- `liyouran` 分支记录 N0 seed 42 在 L40S 上通过：prior Macro F1 0.5493，落在历史 E2a 三种子 0.5405–0.5538 内。这证明原 CCDS 链路可复现，但不是 RTX 5090 环境的独立 N0 证明。
- N1 代码、结果和组员 README 已推送到个人 fork。组长仓库拒绝个人账号直接 push（HTTP 403），因此应通过 Pull Request 合并，或由组长先授予 collaborator 权限。

### 项目任务与架构

任务输入英文 query 与商品标题，输出 exact / substitute / complement / irrelevant 四分类 JSON。数据由 `prepare_esci` 按 query 抽样，再由 `to_alpaca` 转成 LlamaFactory 数据；训练由 LlamaFactory 做 QLoRA；评估不是自由生成，而是 `score_labels` 对四个完整候选回复求 log probability。`prior_adjust` 只在 tune 上搜索偏置，随后固定应用到 test；`compare`、`noise_aware`、`ranking` 分别提供 query bootstrap、full/clean/merged 与偏置无关排序指标。

### E2a 冻结配置

| 类别 | E2a / N1 正式值 |
| --- | --- |
| 数据 | `e1_relevance` train/val；20,000 / 1,000 条；5,000 / 250 queries |
| 输入与目标 | `qwen` template，cutoff 384，只输出 relevance JSON |
| 量化 | bitsandbytes 4-bit NF4；LlamaFactory v0.9.5 默认 double quantization |
| LoRA | r=16，alpha=32，dropout=0.05，target=all linear modules |
| batch | E2a：micro 16 × accumulation 2；N1 实际：micro 1 × accumulation 32；两者 effective 32，单 GPU |
| 优化 | lr 1e-4，cosine，warmup 0.03，1 epoch，bf16 |
| checkpoint | 每 100 步 eval/save；按最低 `eval_loss`；训练结束加载并保存最佳模型 |
| seeds | 42 / 43 / 44（seed 42 为 E2a，43/44 为 E5 的同配置复现） |

N1 唯一研究变量是基座改为 `Qwen/Qwen2.5-14B-Instruct`。`configs/sft/n1_qwen14b_qlora.yaml` 还显式写出 E2a 当时依赖默认值的 bnb、NF4、double quantization 与 gradient checkpointing；这不改变 E2a 语义，只防止依赖升级漂移。

### 数据身份与泄漏边界

| 名称 | 来源 | 用途 | 是否参与选择 |
| --- | --- | --- | --- |
| train | 官方 ESCI train，seed 42/43 的分组抽样 | 梯度更新 | 是 |
| val | 官方 ESCI train，先抽 250 queries | `eval_loss` checkpoint 选择 | 是 |
| tune | 官方 ESCI train，1,500 queries；排除 E1 train/val 与 E2b train | 每个模型单独搜索 prior bias | 是，但只选择推理偏置 |
| test2500 | 官方 ESCI test，2,500 queries；剔除与官方 train 共享的唯一 query | 最终报告 | 否 |

`scripts/autodl/prepare_n1_data.sh` 复用原代码生成这些集合；`ecomq.data.audit_n1` 检查四个集合的 query 两两不重叠、ID 唯一、行数/标签统计，并记录原始与处理后文件 SHA256。仓库没有提交 CCDS 的 canonical hash，因此首次 AutoDL manifest 只能作为新基线；若组长能提供 CCDS 文件，应逐项比对后再训练。

### 直接复用与替换

直接复用：`prepare_esci`、`to_alpaca`、schema/prompt、`score_labels`、`prior_adjust`、`compare`、`noise_aware`、`ranking`、`metrics`、E2a 配置语义和现有单测。

替换：SLURM/PBS 的 `train.sbatch`、`label_scores.sbatch`、`run_e2/e3/e5.sh` 与 module/conda 初始化。AutoDL 使用 `scripts/autodl/` 下的单机脚本，不调用调度器。

### 风险与公平性

1. 预注册的 A100 80GB / micro 16 方案没有实际运行；RTX 5090 32GB 最终使用 micro 1 × accumulation 32。有效 batch 相同不代表数值与吞吐完全等价，因此必须保留偏差说明。任何新组合仍需 smoke，并要求 micro × accumulation = 32。
2. 不允许通过缩短 cutoff、减少数据或 epoch 解决正式实验 OOM。smoke 的 `max_samples` 与短 epoch 仅用于工程验证，输出目录与 formal 隔离。
3. bf16 必须由 `torch.cuda.is_bf16_supported()` 实测通过。
4. LlamaFactory v0.9.5 自动从输出目录最新 checkpoint 续训；完成标记存在时脚本拒绝覆盖。
5. 原 `score_labels --load-4bit` 明确使用 NF4 + bf16 compute，但没有显式启用训练时的 double quantization。N1 使用新增的 `--double-quant` 与训练严格一致；为避免只改变 N1，正式比较前必须用 `rescore_e2a_for_n1.sh` 把 E2a 三个种子按相同协议重新打分。分析脚本会检查确认标记，拒绝混用历史分数。
6. clean 口径必须使用不参与 E2a/N1 比较的委员会预测；`analyze_n1.sh` 因此要求显式提供 `COMMITTEE_PREDS`，不会用 N1 自己定义争议集。
7. AutoDL 镜像可能带与 CCDS 锁定环境不同的 PyTorch/CUDA。安装脚本不猜版本、不重装 torch，而是先检查镜像，再安装其余锁定核心依赖并保存 freeze。

## 2. GPU 与实例选择

预注册方案首选单卡 NVIDIA A100 80GB，以尽量保持 E2a 的 micro-batch。实际 N1 三种子实验运行于 RTX 5090 32GB；由于显存限制，统一使用 micro-batch 1、gradient accumulation 32，并保持 effective batch 32。该硬件与 micro-batch 偏差已写入每个 `run_manifest.json` 和 `reports/n1/README.md`，最终报告必须保留这项披露。

后续复跑应优先复用同类环境；若更换 GPU，必须确认持久化数据盘、CUDA 可见、bf16 实测通过，并记录 GPU、driver、CUDA、PyTorch、micro-batch、gradient accumulation 和峰值显存。不要在不同 seed 之间使用不同 batch 配置。

## 3. Mac → GitHub → AutoDL

N1 功能分支已推送到个人 fork `Pat7ryk/ecom-query-llm`。在组长仓库合并 N1 PR 之前，AutoDL 应从个人 fork 的同一功能分支拉取；合并后，新环境可直接从组长仓库的 `main` 拉取。模型、数据、checkpoint 始终在 `$PERSIST_ROOT`，不进 Git。

```bash
git clone \
  --branch feat/n1-autodl-qwen14b \
  --single-branch \
  https://github.com/Pat7ryk/ecom-query-llm.git
cd ecom-query-llm
git rev-parse HEAD
# 应为本指南所在的已验证 N1 commit；复跑前以 PR/功能分支最新 SHA 为准。
cp .env.autodl.example .env
# 如 AutoDL 的持久化挂载点不同，只编辑 .env
bash scripts/autodl/setup_autodl.sh
```

预期：GPU 检查报告显示目标 GPU、CUDA available=true、bf16_supported=true；依赖 freeze 写到持久化输出目录。原始 N1 结果使用 RTX 5090 32GB、driver 595.71.05、CUDA 13.2、PyTorch 2.15.0。

## 4. 数据与模型

```bash
source .venv/bin/activate
bash scripts/autodl/prepare_n1_data.sh
bash scripts/autodl/download_n1_model.sh
```

检查：

```bash
python -m ecomq.data.audit_n1 \
  --data-root "$DATA_ROOT" \
  --output "$DATA_ROOT/n1_data_manifest.recheck.json"
diff "$DATA_ROOT/n1_data_manifest.json" "$DATA_ROOT/n1_data_manifest.recheck.json"
cat "$MODEL_ROOT/Qwen2.5-14B-Instruct/model_revision.json"
```

模型脚本解析并固定 Hugging Face commit，检查 config、tokenizer 和 index 中列出的全部 safetensors shards。若团队指定模型 revision，可在首次下载前设置 `QWEN14B_REVISION=<commit>`。

## 5. 冒烟、N0 门禁与正式训练

先用正式 micro-batch 做短 smoke；它用于显存和 CLI 检查，不是实验结果：

```bash
bash scripts/autodl/train_n1.sh 42 smoke
```

N0 已由组员在 L40S 上通过，但 AutoDL 软件栈与 GPU 不同。严谨做法是在 AutoDL 用原 E2a 配置重跑 seed 42，并完成 tune/test 打分；prior Macro F1 应落在 0.541–0.554：

```bash
bash scripts/download_model.sh Qwen/Qwen2.5-7B-Instruct
bash scripts/autodl/run_n0_gate.sh
test -f "$OUTPUT_ROOT/n0/analysis/N0_GATE_PASSED"
```

该步骤需要 GPU，当前没有伪造结果。若团队书面接受 L40S N0 作为跨环境门禁，至少仍需 N1 smoke、数据 hash 与版本快照全部通过。

正式串行三种子会强制检查 seed 42 smoke 和 N0 门禁。若团队明确接受已经完成的 L40S N0，可记录审批理由后跳过 AutoDL N0；不能无记录绕过：

```bash
# 仅当团队明确批准复用 L40S N0 时设置；否则不要设置
export N0_GATE_WAIVER_REASON='团队于 YYYY-MM-DD 接受最终仓库中的 L40S N0 作为门禁'
nohup bash scripts/autodl/run_n1_all.sh \
  > "$OUTPUT_ROOT/n1/run-all.log" 2>&1 &
echo $! > "$OUTPUT_ROOT/n1/run-all.pid"
tail -f "$OUTPUT_ROOT/n1/run-all.log"
```

每个 seed 独立目录，完成后有 `COMPLETED`、`run_manifest.json`、训练日志、环境版本和 GPU 显存 CSV。失败时保留 checkpoint；修复原因后用同一命令安全恢复。已完成目录不会被覆盖。

路径默认由 `.env` 的三个 root 推导；需要挂载到其他已准备目录时，可设置 `N1_MODEL_PATH`、`N1_DATASET_DIR`、`N1_OUTPUT_DIR`（最后一个是包含所有 seed 子目录的 N1 输出根目录），无需改脚本。覆盖训练数据时必须同时设置与之匹配的 `N1_DATA_MANIFEST`；若 tune/test 也在另一位置，同时设置 `N1_TUNE_PATH`、`N1_TEST_PATH`，避免训练 manifest 与评估集合错配。

若新环境无法使用预注册的 micro-batch，先保留日志并停止。经团队确认后覆盖 batch，同时保持 effective batch 32，并写明原因。原始 RTX 5090 复跑配置为：

```bash
export N1_MICRO_BATCH=1
export N1_GRAD_ACCUM=32
export N1_BATCH_OVERRIDE_REASON='RTX 5090 32GB：micro-batch=1，gradient accumulation=32，保持 effective batch 32'
bash scripts/autodl/train_n1.sh 42 formal
```

不要自动尝试多个 batch，也不要在三个 seed 之间使用不同 batch。

## 6. 正式分析

先把 E2a 三个种子的 adapter 复制到持久化目录，并按 N1 的 double-quant 协议重新打分。不能直接复用历史 E2a scores。clean 委员会沿用 S18/S20 的独立 rich-input 三种子预测。

```bash
export E2A_ADAPTER_ROOT="$OUTPUT_ROOT/baseline/e2a/adapters"
bash scripts/autodl/rescore_e2a_for_n1.sh
export E2A_SCORE_DIR="$OUTPUT_ROOT/baseline/e2a-double-quant/scores"
export E2A_PRED_DIR="$OUTPUT_ROOT/baseline/e2a-double-quant/preds"
export COMMITTEE_PREDS="$OUTPUT_ROOT/committee/rich-s42.jsonl,$OUTPUT_ROOT/committee/rich-s43.jsonl,$OUTPUT_ROOT/committee/rich-s44.jsonl"
bash scripts/autodl/analyze_n1.sh
```

产出：`training_summary.json/csv`（三个种子的状态、最佳 checkpoint、eval loss、耗时、有效 batch、峰值显存），以及每 seed 的 prior report（包含 Accuracy、Macro F1、各类 P/R/F1、混淆矩阵）、raw/prior predictions、两组种子的 mean/range 与 query bootstrap、full/clean/merged、排序 AUC/AP。所有 prior bias 只由 tune 得出，再应用到 test。

已完成的三种子摘要、prior reports、数据/模型 provenance 和结论边界位于 `reports/n1/`。后续统一重算 E2a 与稳健性分析时，在同一目录追加小型 JSON 报告并更新 `README.md` / `RESULTS.md`；不要提交 scores、逐样本预测缓存、数据、模型、完整日志或 checkpoint。

## 7. 当前状态与剩余门槛

已完成：RTX 5090/bf16 环境检查、数据 hash 审核、模型 revision 固定、三种子正式训练、tune/test 标签打分，以及 full test 的 raw/prior 指标。三个 seed 均完成 625 steps，结果见 `reports/n1/README.md`。

返回的精简结果包没有包含 N1 smoke marker 和 N0 gate/waiver 原件；这些证据应由实验负责人继续保留。最终宣称“14B 稳定优于 7B”之前仍必须：

1. 获取 `e2a-1ep-best`、`e5-20k-s43`、`e5-20k-s44` 三个 adapter，并按相同 double-quant 协议重算。
2. 完成 full/clean/merged、query-level bootstrap 95% CI 和 ranking 分析。
3. 补齐 smoke 与 N0 gate/团队批准 waiver 的审计证据。
4. 确认三种子范围、置信区间与噪声口径均支持预注册判定标准。
