# ecom-query-llm

电商搜索 Query 理解的大模型微调项目：一个 7B 级模型，用统一 JSON 输出同时完成意图识别、属性抽取、Query 改写、相关性判断。训练路线为 SFT（LoRA / 全参数）→ DPO → GRPO，并做蒸馏与 vLLM 部署对比。

## 统一输出格式

```json
{"intent": "商品搜索", "attributes": {"品牌": "耐克", "品类": "跑步鞋"}, "rewrite": "耐克 跑步鞋", "relevance": null}
```

`relevance` 仅在输入带商品标题时判断，否则为 `null`。标签体系和校验规则见 [src/ecomq/schema.py](src/ecomq/schema.py)。

## 目录

```
configs/sft/       LLaMA-Factory 训练配置：冒烟 QLoRA、单卡 QLoRA、LoRA、全参数
configs/ds/        DeepSpeed ZeRO-2 / ZeRO-3（仅 NSCC 多卡使用）
src/ecomq/data/    synth 合成 → validate 校验去重 → split 分组划分 → to_alpaca 转格式；make_smoke 冒烟数据
src/ecomq/eval/    generate(vLLM, NSCC) / generate_hf(transformers, CCDS 单卡) → score(指标)
scripts/slurm/     CCDS（SLURM）训练与评测作业
scripts/pbs/       NSCC（PBS）训练与评测作业
scripts/           setup_env_ccds / setup_env / prepare_data / remote
data_sample/       十几条手写样本，仅用于单测和格式示例
docs/WORKFLOW.md   本地 ↔ GitHub ↔ CCDS（冒烟测试）→ NSCC（完整实验）的工作流
reports/           评测指标 JSON（小文件，进 git）
```

数据、模型、训练产出只放在远端，路径通过 `.env` 配置，不进 git。

## 快速开始（本地）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test          # 单测
make sample-check  # 用样本数据跑通 校验 → 划分 → 转格式
```

远端环境、作业提交和数据流见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

## 实验计划

| 阶段 | 内容 |
|---|---|
| 基线 | 未微调基座 few-shot |
| 方式对比 | LoRA / QLoRA / 全参数 |
| 消融 | r、alpha、目标模块、学习率、数据量、单任务 vs 多任务 |
| 后训练 | DPO、GRPO（相关性与属性抽取用标准答案作可验证奖励） |
| 扩展 | 14B 级复验；蒸馏到 1.5B–3B；vLLM 吞吐与延迟 |

## 数据说明

数据来自公开数据集与强模型合成，不含任何真实线上日志。划分按分组哈希进行，同一商品或 query 模板的样本不会跨集合。测试集需要人工校对后再用于报告。
