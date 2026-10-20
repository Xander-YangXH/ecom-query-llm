# 本地 ↔ GitHub ↔ 两个集群的工作流

两个集群分工：**CCDS 用来免费做单卡冒烟与小规模测试，通过后再到 NSCC 跑完整实验。**
原则不变：**代码进 git，数据/模型/产出只放远端，指标小文件按需回传。**

| | CCDS GPU Cluster TC2 | NSCC |
|---|---|---|
| 用途 | 单卡冒烟、QLoRA/LoRA 小规模验证 | 完整实验：多卡、全参数、DPO/GRPO、14B 复验 |
| 调度器 | SLURM（`sbatch`） | PBS（`qsub`） |
| GPU | 单卡 L40S（48GB） | 多卡（型号与数量以 NSCC 实际分配为准） |
| 访问 | 仅 NTU 校网或 NTU VPN；账号为 NTU 账号 | 以 NSCC 文档为准 |
| 限制（指南示例值，以 `mytcinfo` 为准） | 1 GPU、10 CPU、30GB 内存、同时 2 个作业、单作业约 6 小时、家目录约 100GB | 以项目配额为准 |
| 环境 | conda（`scripts/setup_env_ccds.sh`） | venv（`scripts/setup_env.sh`） |
| 作业脚本 | `scripts/slurm/*.sbatch` | `scripts/pbs/*.pbs` |
| `.env` 模板 | `.env.ccds.example` | `.env.nscc.example` |

```
本地(写代码、跑单测) --git push--> GitHub --git pull--> CCDS(冒烟/小测)
                                          └--git pull--> NSCC(完整实验)
                     reports/*.json  <-- git 或 rsync --  两个集群
```

| 内容 | 位置 | 是否进 git |
|---|---|---|
| 代码、配置、作业脚本、文档 | 仓库 | 是 |
| `data_sample/`（十几条样本，用于单测） | 仓库 | 是 |
| 全量数据、合成数据、划分结果 | 远端 `$DATA_ROOT` | 否 |
| 基座模型、LoRA 适配器、checkpoint | 远端 `$MODEL_ROOT` / `$OUTPUT_ROOT` | 否 |
| 指标 JSON（`reports/`） | 仓库 | 是（体积很小） |
| `.env`（路径、API Key） | 各机器自己维护 | 否 |

## 1. 首次：本地推到 GitHub

先在 GitHub 网页上新建一个**私有空仓库**（不要勾选自动生成 README），然后：

```bash
cd ecom-query-llm
git add -A && git commit -m "init: project skeleton"
git branch -M main
git remote add origin git@github.com:<你的用户名>/ecom-query-llm.git
git push -u origin main
```

## 2. CCDS：环境搭建与冒烟测试

CCDS 规则要点（来自用户指南）：头节点只能做登录、传文件、编辑和提交作业，**不能编译或运行计算代码**；计算一律用 `sbatch`；避免大量 SSH 连接；只能访问自己的家目录。

```bash
# 在 NTU 校网/VPN 下登录（密码由你自己输入）
ssh <你的用户名小写>@<ccds-host>      # 主机地址见集群用户指南

git clone git@github.com:<你的用户名>/ecom-query-llm.git ~/ecom-query-llm
cd ~/ecom-query-llm
cp .env.ccds.example .env          # 默认数据放 ~/ecomq，家目录配额约 100GB
bash scripts/setup_env_ccds.sh      # 创建 conda 环境 ecomq（只做安装）
```

GitHub 认证建议用只读 Deploy Key：在 CCDS 上 `ssh-keygen -t ed25519`，把公钥加到仓库 Settings → Deploy keys（不勾选写权限）。如果集群出站网络限制 GitHub，就用 `rsync` 上传代码。

冒烟测试的目标只有一个：**证明整条链路能跑通**。数据是把 13 条样本复制成 400 条，验证集和测试集与训练集重叠，所以指标没有意义。

```bash
# 1) 下载基座模型到 $MODEL_ROOT/Qwen2.5-7B-Instruct（计算节点是否有外网指南没写，先在登录节点下载）

# 2) 提交 QLoRA 冒烟训练；MAKE_SMOKE=1 会在计算节点上先生成冒烟数据（头节点不运行计算代码）
mkdir -p logs
sbatch --export=ALL,MAKE_SMOKE=1,CONFIG=configs/sft/smoke_qlora.yaml,MODEL=Qwen2.5-7B-Instruct,RUN=smoke scripts/slurm/train.sbatch

# 3) 查看队列与日志
squeue -u $USER
tail -f logs/ecomq-train_<jobid>.out

# 4) 训练成功后做一次推理加打分；QLoRA 训练的模型要加 LOAD_4BIT=1，让基座的量化方式与训练一致（见 DEVLOG P8）
sbatch --export=ALL,MODEL=Qwen2.5-7B-Instruct,RUN=smoke,LORA=1,LOAD_4BIT=1 scripts/slurm/eval.sbatch
```

冒烟通过的判断标准：

- [ ] 作业正常结束，日志里没有 OOM 或加载错误
- [ ] `$OUTPUT_ROOT/smoke` 下有 LoRA 适配器和 checkpoint
- [ ] 日志里 loss 在下降，并记下实际显存占用与每步耗时（用于估算正式训练时间）
- [ ] `reports/smoke.json` 生成，格式合法率明显高于 0
- [ ] 用同一个 `RUN` 重新提交一次，日志里确认从 checkpoint 续训（为 6 小时上限做准备）

## 3. CCDS：正式的小规模实验

通过冒烟后，在 CCDS 上可以做 QLoRA 与 bf16 LoRA 的小规模对比：

```bash
sbatch --export=ALL,CONFIG=configs/sft/qwen7b_qlora_r16.yaml,MODEL=Qwen2.5-7B-Instruct,RUN=qwen7b-qlora-r16 scripts/slurm/train.sbatch
```

- 单作业约 6 小时，到点后用**同样的命令**重新提交即可续训。
- 同时运行的作业数和资源总和都有上限，超限会一直 PENDING；排队多时先别重复提交。
- 7B 全参数、DeepSpeed、14B、vLLM 都不要放在 CCDS，留给 NSCC。
- 注意家目录配额：`save_total_limit` 已设为 2，定期用 `ncdu` 检查。

## 4. 迁移到 NSCC

CCDS 冒烟通过、配置稳定后再迁移：

1. 在 NSCC 上 clone 仓库，`cp .env.nscc.example .env`，运行 `bash scripts/setup_env.sh`。
2. 按 NSCC 文档修改 `scripts/pbs/*.pbs` 里的项目号、队列和 `select=` 资源写法。
3. 多卡 LoRA：`qsub -v CONFIG=configs/sft/qwen7b_lora_r16.yaml,MODEL=...,RUN=...,DS=configs/ds/ds_z2.json scripts/pbs/train.pbs`；全参数：`CONFIG=configs/sft/qwen7b_full.yaml` 并设 `DS=configs/ds/ds_z3.json`。
4. 评测用 `scripts/pbs/eval.pbs`（vLLM 版）。

迁移检查表：

- [ ] 数据格式、提示词模板在 CCDS 上已验证一致（`tests/test_smoke.py` 里有训练与推理提示词一致性测试）
- [ ] 在 CCDS 上记录的显存与耗时，作为 NSCC 申请资源的依据
- [ ] 基座模型版本、LLaMA-Factory 版本已用 `pip freeze` 锁定，两边一致
- [ ] 路径全部来自 `.env`，没有硬编码

## 5. 数据流

1. **合成**：`ecomq.data.synth` 需要外网调用强模型，建议在你自己的电脑上运行，产出 `synth.jsonl` 后用 `scripts/remote.sh put` 传到远端 `$DATA_ROOT/raw/`。
2. **处理**：在远端（或本地）运行 `bash scripts/prepare_data.sh`：校验去重 → 分组划分 → 转 Alpaca 格式。
3. **训练 / 评测**：见上面各节，产出写入 `$OUTPUT_ROOT`，指标 JSON 写到仓库 `reports/`。
4. **回传指标**：在远端 `git add reports && git commit && git push`，或在本地 `scripts/remote.sh get-reports`。

## 6. 需要你自己核对的地方

我没有核对两个集群的实时配置，下面这些在脚本里是占位或假设：

- CCDS：`--partition`、`--qos`、`--mem`、`--time` 来自指南的示例值；实际数值用 `mytcinfo` 或 `sacctmgr` 确认。
- CCDS：计算节点是否有外网、PyTorch wheel 的 CUDA 版本是否与节点驱动兼容（脚本里写的是 cu128，请先核对）。
- CCDS：指南说明资源仅限学术用途，请确认你的使用方式在允许范围内。
- NSCC：PBS 头部的项目号、队列、`select=` 写法，登录主机，存储路径与配额，计算节点外网策略。
- 两边：LLaMA-Factory 的参数名随版本会变（例如量化、续训行为），首次运行请在日志里确认。
