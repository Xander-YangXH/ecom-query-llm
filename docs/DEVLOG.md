# 技术日志（DEVLOG）

按时间顺序记录每一步：**为什么做、怎么做、效果如何、遇到的问题与解决方式、遗留问题**。
写给三类读者：组员（了解进度与决策）、未来的自己（复现与排错）、面试官（讲清工程取舍）。

约定：
- 服务器主机名、用户名、密码、API Key 一律不写进仓库。文中用 `<ccds-host>`、`<user>` 代替。
- 数字只写实际跑出来的；估算会明确标注“估算”。
- 每个技术问题按「现象 → 原因 → 解决 → 验证」记录，并在下面的索引里登记一行。

## 问题索引

| 编号 | 步骤 | 问题 | 一句话解决 |
|---|---|---|---|
| P1 | S1 | 训练与推理的提示词结构不一致（评测会被系统性低估） | 统一用 `build_user_prompt()`，并加单测锁定 |
| P2 | S4 | 分区内 GPU 未标型号，无法按型号申请 L40S | `--exclude=TC2N0[1-6]` 只留 L40S 节点 |
| P3 | S5 | 导入 peft 报 `torchvision::nms does not exist` | torch/torchvision/torchaudio 从同一个 cu128 源安装 |
| P4 | S5 | 模型只下载了 6 个文件，缺 `config.json` 和分词器 | 去掉 `hf download` 的多模式 `--include`，下载整个仓库 |
| P5 | S5 | transformers 5.x 移除了 `overwrite_output_dir`，配置会解析失败 | 从所有配置删除该参数，依赖 LLaMA-Factory 默认续训 |
| P6 | S5 | transformers 5.x 中 `torch_dtype` 已弃用 | 推理脚本改用 `dtype` |

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
- 续训：`save_total_limit: 2` 控制 checkpoint 数量；CCDS 单作业 6 小时上限到点后，用同一个 `RUN` 重新提交即从最新 checkpoint 继续（最初用 `overwrite_output_dir: false` 实现，后因 P5 改为依赖 LLaMA-Factory 的默认行为）。
- 新增 `make_smoke.py`：把 13 条手写样本复制成 400 条，只用于验证流程。
- 新增 `generate_hf.py`：transformers + peft 推理，不依赖 vLLM，CCDS 上安装更简单。

**P1 · 训练与推理提示词不一致**
- 现象：代码审查时发现，未上机。LLaMA-Factory 处理 Alpaca 样本时，把 `instruction` 和 `input` 用换行拼成**同一条 user 消息**；而原推理脚本把 `instruction` 放在 **system** 消息里。
- 原因：写推理脚本时按常见的「system 放指令」习惯写，没有对照训练框架实际的拼接方式。
- 影响：模型在推理时看到的输入分布与训练时不同，评测分数会被系统性低估，而且很难从结果上看出原因。
- 解决：`schema.build_user_prompt()` 作为唯一的推理提示词构造函数，两个推理脚本都调用它。
- 验证：新增单测 `test_inference_prompt_matches_training_prompt`，逐条比对两边拼出的文本完全一致。

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

---

## S5 · CCDS 环境搭建、模型下载与 GPU 验证（2026-10-20）

**为什么**
- 在提交任何训练之前，先单独确认「环境 + 硬件 + 模型文件」三件事都没问题。否则训练作业失败时很难判断是代码、环境还是硬件的问题，而且每次排队都浪费时间。

**怎么做**
1. 环境：`scripts/setup_env_ccds.sh` 在头节点创建 conda 环境 `ecomq`（只做安装，不跑计算）。
   - 只用 conda-forge 频道：新版 conda 的 Anaconda 默认频道要求先接受服务条款，conda-forge 不需要。
   - `pip install --no-cache-dir`：家目录配额约 100GB，避免 pip 缓存再存一份几 GB 的 wheel。
   - 装完 `pip freeze > requirements-lock-ccds.txt`，锁定版本，NSCC 上按同样的版本装。
2. 模型：`scripts/download_model.sh Qwen/Qwen2.5-7B-Instruct`，在头节点下载到 `$MODEL_ROOT`。
3. 验证：`scripts/slurm/gpu_check.sbatch`（10 分钟、8GB 内存的小作业），检查显卡型号与驱动、PyTorch CUDA、bf16、bitsandbytes 4bit 前向、各库能否导入、计算节点外网。

**效果**

| 项目 | 结果 |
|---|---|
| 环境安装耗时 | 约 8 分钟；环境占用 8.4GB |
| 关键版本 | torch 2.11.0+cu128、transformers 5.6.0、peft 0.18.1、bitsandbytes 0.50.2、llamafactory 0.9.5、trl 0.24.0 |
| 依赖一致性 | `pip check`：No broken requirements found |
| 模型 | Qwen2.5-7B-Instruct，15GB，13 个文件齐全 |
| GPU 节点 | TC2N08，NVIDIA L40S，46068 MiB，驱动 570.148.08（支持 CUDA 12.8） |
| bf16 / 4bit | bf16 矩阵乘与 bitsandbytes NF4 前向均通过 |
| 计算节点外网 | 可访问 Hugging Face（HTTP 200） |
| 验证作业 | 第一次（44740）失败，修复 P3 后第二次（44741）通过，耗时 11 秒 |

**P3 · 导入 peft 报 `operator torchvision::nms does not exist`**
- 现象：第一次 GPU 验证作业中，PyTorch、bf16、bitsandbytes 都通过，但 `import peft` 失败，错误链最终是 `Could not import module 'PreTrainedModel'`。
- 原因：环境脚本先从 PyTorch 的 cu128 源只装了 `torch`；随后安装 LLaMA-Factory 时，pip 为满足依赖从 PyPI 装了 `torchvision`、`torchaudio`。PyPI 上的这两个包是另一种 CUDA 构建，与 `torch+cu128` 的 C++ 算子注册不兼容。transformers 导入时会触发 torchvision，于是连带 peft 失败。
- 解决：
  - 服务器上：`pip install --force-reinstall --no-deps torchvision==0.26.0 torchaudio==2.11.0 --index-url .../cu128`。
  - 脚本里：改为 `pip install torch torchvision torchaudio --index-url .../cu128` 一次装齐，后续依赖发现已满足就不会再从 PyPI 拉取。
- 验证：`pip list` 显示三者均为 `+cu128`；第二次 GPU 验证作业全部通过。
- 经验：错误信息指向 transformers/peft，根因却在 torchvision。看到 `operator ... does not exist` 这类 C++ 算子错误，先查 torch 系列包的 CUDA 构建是否一致。

**P4 · 模型只下载了 6 个文件**
- 现象：下载结束显示 15GB，但目录里没有 `config.json` 和分词器文件，日志里有警告 `Ignoring --include since filenames have been explicitly set`。
- 原因：脚本写成 `hf download repo --include "*.json" "*.safetensors" ...`。新版 `hf` CLI 只把第一个模式当作 `--include` 的值，其余被当作位置参数（文件名），于是 `--include` 整体被忽略，只按“文件名”下载了能匹配到的几个文件。
- 解决：去掉 `--include`，直接下载整个仓库（该仓库里没有多余的大文件）；重跑时已下载的分片会被跳过。
- 验证：13 个文件齐全。
- 经验：只看总大小会误判“下载完成”，要核对关键文件是否存在。

**P5 · transformers 5.x 移除了 `overwrite_output_dir`**
- 现象：上机前检查已安装的 LLaMA-Factory 源码，发现注释 `for mca training args and transformers >= 5.0`；进一步确认 transformers 5.6 的 `TrainingArguments` 中已没有这个字段。
- 原因：配置里写了 `overwrite_output_dir`，而 LLaMA-Factory 默认不允许多余参数（`allow_extra_keys=False`），作业一启动就会解析失败。
- 解决：从所有训练配置中删除该参数。续训改为依赖 LLaMA-Factory 的默认行为：只要 `output_dir` 里有 `checkpoint-*`，就自动从最新的继续（日志出现 `Resuming training from`）。
- 副作用：冒烟配置不能再“强制从头训练”，要从头重跑时换一个 `RUN` 名即可。
- 经验：大版本升级（transformers 4 → 5）后，先读一遍训练框架的参数解析代码，比提交作业等报错更省时间。

**P6 · transformers 5.x 中 `torch_dtype` 已弃用**
- 解决：`generate_hf.py` 加载模型时改用 `dtype=torch.bfloat16`。属于预防性修改，避免日志被弃用警告淹没，以及后续版本彻底移除后报错。

**遗留问题**
- 冒烟训练与评测结果见 S6。
