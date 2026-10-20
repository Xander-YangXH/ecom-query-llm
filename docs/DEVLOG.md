# 技术日志（DEVLOG）

按时间顺序记录每一步：**为什么做、怎么做、效果如何、遗留问题**。
写给三类读者：组员（了解进度与决策）、未来的自己（复现与排错）、面试官（讲清工程取舍）。

约定：
- 服务器主机名、用户名、密码、API Key 一律不写进仓库。文中用 `<ccds-host>`、`<user>` 代替。
- 数字只写实际跑出来的；估算会明确标注“估算”。

---

## S0 · 项目骨架（2026-10-17）

**为什么**
- 先把「数据格式 → 处理流水线 → 训练配置 → 评测」的接口定下来，后面换数据、换模型、换集群都不用改主干。

**怎么做**
- `src/ecomq/schema.py`：统一 JSON 输出（`intent` / `attributes` / `rewrite` / `relevance`），标签集合与校验规则只在这一处定义。
- 数据流水线：`validate`（规则校验 + 精确去重）→ `split`（按分组哈希划分）→ `to_alpaca`（转 LLaMA-Factory 格式并生成 `dataset_info.json`）。
- 评测：`generate` 与 `score` 解耦，指标包括格式合法率、意图准确率与宏 F1、属性实体级 F1、相关性准确率，并按难度分层。

**关键设计决策**
| 决策 | 原因 |
|---|---|
| 属性值必须在 query 或标题原文中出现 | 拦截强模型合成数据里的幻觉属性 |
| 按分组键（同一商品 / 同一 query 模板）哈希划分 | 防止近似样本跨 train/test 泄漏导致指标虚高 |
| 格式合法率单独作为指标 | 下游程序要直接解析 JSON，格式错一次就是线上故障 |
| 解析失败按全错计分 | 不让“输出不合法”被平均掉 |

**效果**
- 本地 12 项单测通过；`make sample-check` 跑通“校验 → 划分 → 转格式”。

---

## S1 · 双集群适配：CCDS 测试 → NSCC 实验（2026-10-20）

**为什么**
- 可用算力有两套：CCDS GPU Cluster TC2（免费，单卡 L40S 48GB，SLURM）和 NSCC（PBS，多卡）。先在免费单卡上把整条链路跑通，再到 NSCC 做贵的实验，能把环境和代码问题提前暴露，节省 NSCC 配额。

**怎么做**
- 新增 `scripts/slurm/train.sbatch`、`eval.sbatch`；保留 `scripts/pbs/*.pbs` 给 NSCC。
- 新增单卡配置：`configs/sft/smoke_qlora.yaml`（冒烟）、`qwen7b_qlora_r16.yaml`（正式单卡 QLoRA）。
- DeepSpeed 从基础配置中移出，改为 NSCC 多卡作业通过 `DS=...` 追加，单卡不加载。
- 续训：配置里 `overwrite_output_dir: false` + `save_total_limit: 2`，CCDS 单作业 6 小时上限到点后，用同一个 `RUN` 重新提交即从最新 checkpoint 继续。
- 新增 `make_smoke.py`：把 13 条手写样本复制成 400 条，只用于验证流程。
- 新增 `generate_hf.py`：transformers + peft 推理，不依赖 vLLM，CCDS 上安装更简单。

**修复的一个隐蔽问题：训练与推理提示词不一致**
- 现象（代码审查发现，未上机）：LLaMA-Factory 处理 Alpaca 样本时，把 `instruction` 和 `input` 用换行拼成**同一条 user 消息**；而原推理脚本把 `instruction` 放在 **system** 消息里。
- 影响：模型在推理时看到的输入分布与训练时不同，评测分数会被系统性低估，而且很难从结果上看出原因。
- 修复：`schema.build_user_prompt()` 作为唯一的推理提示词构造函数，两个推理脚本都调用它；新增单测 `test_inference_prompt_matches_training_prompt` 锁死这一点。

**效果**
- 本地 15 项单测通过，8 个 shell 脚本语法检查通过。
- 以上脚本在集群上的实际运行结果见后续条目。

---

## S2 · CCDS 访问配置与环境探测（2026-10-20）

**为什么**
- 需要让自动化脚本（以及 Claude）能非交互地登录服务器执行命令，同时**密码不经过任何第三方**。

**怎么做**
- 本机已有 ed25519 密钥；由用户本人在终端执行 `ssh-copy-id` 并输入密码，把公钥写入服务器的 `authorized_keys`。
- 之后用 `ssh -o BatchMode=yes` 验证免密登录（BatchMode 下如果需要密码会直接失败，而不是卡住等待输入）。
- 只读探测：`mytcinfo`、`df -h ~`、`module avail anaconda`、从头节点 `curl` 三个外部站点。

**效果**

| 项目 | 结果 |
|---|---|
| 免密登录 | 成功 |
| QoS | `normal`：cpu=10、gpu=1、mem=30G、同时 2 个作业、单作业 06:00:00；分区 `MGPU-TC2` |
| 家目录 | 共享文件系统（总 10T，已用 90%，剩约 1.1T）；个人配额按指南约 100GB（`quota` 命令无输出） |
| conda | 模块 `anaconda/25.5.1` |
| 头节点外网 | Hugging Face、download.pytorch.org、GitHub 均返回 HTTP 200 |
| 清理 | 用户本人删除了两个不再需要的旧项目目录，约 8.2GB |

**对后续的影响**
- SLURM 脚本中的资源（8 CPU、30G 内存、05:55:00）均在 QoS 上限之内。
- 头节点能联网 → 模型和依赖可以直接在头节点下载，不需要从本机中转。
- 计算节点能否联网仍未知，将在 GPU 验证作业中测试。

**遗留问题**
- 共享文件系统整体已用 90%，大文件（模型、checkpoint）要控制数量，及时清理。

---

## S3 · GitHub 同步与代码上传（2026-10-20）

**为什么**
- GitHub 私有仓库是代码的唯一来源，也是备份（集群明确不提供备份）。
- 服务器上的代码需要和本地一致，但不想在服务器上存放任何能写 GitHub 的凭证。

**怎么做**
- 本机通过已有的 SSH 密钥推送到私有仓库；未登录访问仓库地址返回 404，确认不是公开仓库。
- 代码上传到 CCDS 用 `scripts/remote.sh sync-code`（rsync）：
  - 排除 `.git/`、`.env`、数据、模型、产出、`logs/`、`reports/`。
  - 不使用 `--delete`，避免误删服务器上作业生成的日志和指标。
  - 服务器上的 `.env` 只在不存在时从 `.env.ccds.example` 复制，之后不会被覆盖。
- 提交前扫描仓库，确认没有写入服务器地址和用户名（仓库以后可能公开用于简历）。

**效果**
- 首次提交已推送到 `main`；服务器 `~/ecom-query-llm` 与本地代码一致。

**取舍**
- CCDS 上的代码不是 git 仓库，所以不能在服务器上 `git pull`，每次改完代码要在本机执行一次 `sync-code`。换来的好处是服务器上不需要 GitHub 凭证。NSCC 上可以再决定是否用只读 Deploy Key + `git pull`。

---

## S4 · 指定 L40S 节点（2026-10-20）

**为什么**
- 用户告知 L40S 只在 TC2N07、TC2N08 上。分区 `MGPU-TC2` 共 8 个节点，不指定的话作业可能被分到其他型号的 GPU 上，显存和速度都与预期不同，实验数据也不可比。

**怎么做**
- 只读查询 `sinfo`、`scontrol show node`：所有节点的 GRES 都是 `gpu:nvidia`，没有标注显卡型号，也没有可用的 feature 标签，因此无法用 `--gres=gpu:l40s:1` 或 `--constraint` 选型号。
- 在全部 SLURM 脚本中加入 `#SBATCH --exclude=TC2N0[1-6]`。
  - 不用 `--nodelist=TC2N07,TC2N08`：SLURM 会要求同时分配列出的所有节点，与 `--nodes=1` 冲突。
  - 用排除法，两个 L40S 节点哪个有空就用哪个，排队更短。

**效果**
- 待 GPU 验证作业确认（作业日志会打印 `nvidia-smi` 的显卡型号）。

| 节点 | GPU 数 | CPU | 内存 |
|---|---|---|---|
| TC2N01–02 | 8 | 104 | 约 340GB |
| TC2N03–08 | 4 | 128 | 约 450GB |
