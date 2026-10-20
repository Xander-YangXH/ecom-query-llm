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
| P7 | S6 | 代码同步漏传 `src/ecomq/data/`，作业报 `No module named 'ecomq.data'` | 改为只同步 git 跟踪的文件，并在同步后抽查子包 |
| P8 | S6 | QLoRA 适配器配 bf16 基座评测，边界样本判断翻转 | QLoRA 模型评测时用 `LOAD_4BIT=1`，与训练时的量化方式一致 |
| P9 | S6 | `pip freeze` 把含用户名的本地路径写进了锁定文件并已推送 | 改用 `pip freeze --exclude-editable`；历史提交中的残留待决定是否清理 |
| P10 | S7 | 一次性提交整条依赖链被拒：`QOSMaxSubmitJobPerUserLimit` | 每人最多同时提交 2 个作业；改用头节点上的顺序调度脚本 `chain_e1.sh` |
| P11 | S7 | 调度脚本挂在本机 SSH 会话里，断开后可能被 SIGPIPE 终止；`pkill -f` 误杀了执行它的 SSH 会话 | 改用 `setsid nohup` 在服务器上独立运行、直接写日志；`pkill` 的匹配模式不要包含命令行里出现的文本 |
| P12 | S9 | `sbatch --export=...,PREP_ARGS="... complement=0.10,irrelevant=0.20"` 会把值里的逗号当成变量分隔符 | 含逗号的值改为通过环境变量传递，依靠 `--export=ALL` 继承 |

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

---

## S6 · 冒烟训练与评测（2026-10-20）

**为什么**
- 用最小的成本（400 条复制样本、1 个 epoch）把「数据生成 → 4bit 加载 → LoRA 训练 → 保存 → 推理 → 打分」整条链路在真实 GPU 上走一遍。冒烟的目的是验证流程，不看效果。

**怎么做**
- 训练：`sbatch --export=ALL,MAKE_SMOKE=1,CONFIG=configs/sft/smoke_qlora.yaml,MODEL=Qwen2.5-7B-Instruct,RUN=smoke scripts/slurm/train.sbatch`
  - `MAKE_SMOKE=1` 让冒烟数据在计算节点上生成（头节点不允许运行计算代码）。
- 评测：`sbatch --dependency=afterok:<训练作业号> ... scripts/slurm/eval.sbatch`
  - 用 SLURM 依赖把评测接在训练成功之后自动运行，不用人工盯着；训练失败时评测作业会被取消。

**P7 · 代码同步漏传 `src/ecomq/data/`**
- 现象：第一次冒烟训练（作业 44742）4 秒就失败，报 `No module named 'ecomq.data'`；依赖它的评测作业 44743 随之被取消。
- 原因：`remote.sh sync-code` 用了 `rsync --exclude 'data/'`，本意是排除仓库根目录的数据目录。但 rsync 的排除模式不带前导 `/` 时会匹配**任意层级**的同名目录，于是源码目录 `src/ecomq/data/` 也被排除了。本地单测能过，是因为本地文件是全的，问题只在同步这一环。
- 解决：同步方式改为 `git ls-files | rsync --files-from=-`，只传 git 跟踪的文件。数据、`.env`、模型本来就不在 git 里，不再需要任何排除规则，从根上避免误伤。同步后再到远端抽查每个子包的 `__init__.py`。
- 验证：重新同步后，`src/ecomq/data/` 下 7 个文件全部上传，抽查输出 `sync-check-ok`。
- 经验：用排除规则维护“哪些不传”容易漏、也容易误伤；用 git 的跟踪列表定义“哪些要传”更可靠。代价是新文件必须先 `git add` 才会被同步。

**效果：冒烟通过**

| 检查项 | 结果 |
|---|---|
| 训练作业 44744 | COMPLETED，作业总耗时 4 分 02 秒（含加载模型），纯训练 65 秒，节点 TC2N08（L40S） |
| 训练规模 | 400 条 × 1 epoch，有效 batch 4，共 100 步；约 1.5 步/秒、6.1 样本/秒 |
| 可训练参数 | 40,370,176 / 7,655,986,688（0.53%），LoRA r=16、作用于全部线性层 |
| loss | train_loss 0.0379；eval_loss 从 0.0086（epoch 0.2）降到 7.9e-05（epoch 1.0） |
| 产物 | LoRA 适配器、`checkpoint-100`、loss 曲线图 |
| 评测作业 44745 | COMPLETED，38 秒；`reports/smoke.json` 已生成 |
| 续训验证（作业 44748） | 日志出现 `Resuming training from .../checkpoint-100`，从第 100 步接着跑，立即结束 |

评测结果（13 条，与训练集重叠，只说明链路正确，不代表效果）：

| 指标 | bf16 加载基座 | 4bit 加载基座 |
|---|---|---|
| 格式合法率 | 1.000 | 1.000 |
| 意图准确率 | 0.923 | 1.000 |
| 意图宏 F1 | 0.733 | 1.000 |
| 属性 F1 | 1.000 | 1.000 |
| 相关性准确率 | 1.000 | 1.000 |
| 改写完全匹配 | 1.000 | 1.000 |

**P8 · QLoRA 适配器用 bf16 基座评测时，边界样本判断翻转**
- 现象：第一次评测中唯一的错误是 m-005「小米电视和华为电视哪个好」，标注为「比价」，模型输出「咨询」，其余字段全对。这条样本在训练集中重复出现了约 30 次，eval_loss 已接近 0，按理应能完全记住。
- 分析：有两种可能。
  1. 训练时基座是 4bit（NF4）量化加载的，LoRA 适配器学到的是「在量化权重上的修正」；评测时基座按 bf16 加载，权重数值与训练时不同，在模型本身就犹豫的样本上可能翻转。
  2. 标签本身有歧义：「哪个好」比较的是质量而不是价格，「比价」与「咨询」的边界不清。
- 验证：给 `eval.sbatch` 增加 `LOAD_4BIT=1` 选项，用与训练一致的 4bit 方式加载基座重新评测（作业 44749，21 秒）。m-005 输出变为「比价」，13 条全部正确。
- 结论：结果支持第 1 种解释。但只有 1 条样本发生变化，证据有限；第 2 种歧义仍然存在，正式定义标签体系时需要把「比价」改名或写清楚边界（例如「对比」覆盖价格和质量的比较）。
- 做法：QLoRA 训练的模型评测时默认加 `LOAD_4BIT=1`；如果要用 bf16 部署（例如 vLLM），需单独比较「合并后 bf16」与「4bit + 适配器」的指标差异，作为部署前的检查项。

**P9 · 锁定文件里混入了含用户名的本地路径**
- 现象：提交前的敏感信息扫描发现，`requirements-lock-ccds.txt` 有一行 `-e /home/<集群目录>/<user>/ecom-query-llm`。
- 原因：项目本身以 `pip install -e .` 安装，`pip freeze` 会把可编辑安装写成本机绝对路径。S5 提交前的扫描只检查了手写的文件，没有覆盖这个自动生成的文件。
- 解决：删除该行；环境脚本改用 `pip freeze --exclude-editable`。之后每次提交前的扫描覆盖整个工作区（包括生成文件）。
- 遗留：这一行已经存在于提交 `39c9654` 中。仓库目前是私有的，影响有限；如果以后要公开仓库，需要先改写这段历史，或者以一个干净的新仓库发布。
- 经验：自动生成的文件（锁定文件、日志、配置导出）最容易带出机器上的信息，扫描时不能只看手写的代码。

**新增：显存采样**
- 冒烟训练时没有记录显存，无法用于估算正式训练的 batch 大小。`train.sbatch` 现在会每 15 秒把显存与利用率写入 `logs/gpu_mem_<作业号>.csv`。续训验证作业只加载了模型就结束（峰值约 6GB），不代表训练时的显存，正式训练时再记录。

**下一步**
- 用真实数据（ECInstruct、ESCI 与合成数据）替换冒烟数据，先在 CCDS 上跑一组正式的单卡 QLoRA 小规模实验，记录显存、速度与指标，作为迁移到 NSCC 的依据。


---

## S7 · 真实数据与第一组正式实验 E1 的设计（2026-10-20）

**为什么**
- 冒烟只证明链路通。要得到可以写进简历、能和公开结果对比的数字，需要真实的人工标注数据和一个干净的实验设定。

**数据选择**

| 数据集 | 用途 | 理由 |
|---|---|---|
| ESCI（Apache-2.0） | E1 的训练与测试 | 唯一有大规模人工标注、又直接对应搜索业务的数据；官方 Task 2 就是四分类，有公开基线（micro-F1 0.62 / macro-F1 0.23，来自官方仓库说明） |
| ECInstruct（CC-BY-4.0，子集许可证不一） | 本次只下载和统计 | 它的属性抽取是“从商品标题/描述抽取”，不是“从 query 抽取”；它的多分类任务由 ESCI 派生，与 E1 测试集可能重叠，直接混用有泄漏风险 |
| 意图、改写 | 暂缓 | 没有公开标签，需要用强模型合成（需要 API），放到 E2 之后 |

**标签体系**
- 改为英文，并拆成两个任务：`relevance`（真实标注）与 `full`（合成标注），详见 `docs/LABELS.md`。
- relevance 采用 ESCI 原生的 4 类（exact/substitute/complement/irrelevant），而不是早期设计的 3 类，以便与官方 Task 2 直接对比。
- 意图标签「比价」改为 `compare`，覆盖价格与质量比较，解决冒烟中发现的边界歧义（P8）。

**E1 数据构造**（`prepare_esci.py`，作业内运行）
- 范围：美国英文、`large_version=1`（官方 Task 2 的数据范围）。
- 按 query 分组抽样：train 5000 个 query、val 250 个（从官方 train 划出，与 train 不重叠）、test 600 个（从官方 test 抽）；每个 query 随机取至多 4 个商品。
  - 先抽 query 再抽商品：同样的样本量覆盖更多 query，同一 query 不会跨集合。
  - 保留自然标签分布，不做类别平衡，作为基线设定；类别平衡留给后续对照实验。
- 输入只用 query + 商品标题（截断到 200 字符），控制序列长度。
- 统计写入 `$DATA_ROOT/e1_relevance/stats.json`。

**E1 实验设定**

| 项 | 设定 | 理由 |
|---|---|---|
| 基座 | Qwen2.5-7B-Instruct | 与冒烟一致，环境已验证 |
| 方法 | QLoRA（4bit NF4），r=16、alpha=32、全部线性层 | 单卡 L40S 的主力方案 |
| 训练 | 2 epoch、lr 1e-4、cosine、有效 batch 32、cutoff 384 | 分类任务输出很短，2 epoch 足够观察收敛 |
| 基线 | 同一指令下的基座 zero-shot（bf16） | 回答“微调带来多少提升” |
| 评测 | 4bit 基座 + 适配器（与训练一致）；另测 bf16 基座 + 适配器 | 同时量化 P8 在真实数据上的影响 |

**P10 · 一次性提交整条依赖链被拒**
- 现象：用 `--dependency` 一次提交“数据 → 基线 → 训练 → 两次评测”5 个作业，第 3 个起报 `QOSMaxSubmitJobPerUserLimit`。
- 原因：QoS 除了“同时运行 2 个作业”，还限制每人同时**提交**（含排队）的作业数为 2；用户指南只写了前者。
- 解决：新增 `scripts/slurm/chain_e1.sh`，在头节点上按顺序提交、等待上一个结束再提交下一个；只调用 `squeue/sacct/sbatch`，不做计算。训练超时（TIMEOUT）时自动用同一 RUN 续训，最多 2 次。
- 经验：提交前用 `sacctmgr show qos` 查全部限制字段（如 MaxSubmitJobsPU），不要只依赖指南里的示例。

**数据处理放在哪里**
- 下载在头节点（联网、非计算）；处理在作业里（集群规则禁止头节点计算）。
- 代价：集群要求每个作业都申请 GPU，这个 CPU 作业会占用一张 GPU 几分钟；因此它不限定 L40S 节点，避免占用稀缺资源。

**数据处理结果**（作业 44756，29 秒；下载在头节点用时 152 秒）
- 池子：美国英文 large_version 共 1,818,825 条标注（全部有标题）；官方 train 1,393,063 条、test 425,762 条。

| 集合 | 条数 | query 数 | exact | substitute | complement | irrelevant |
|---|---|---|---|---|---|---|
| train | 20,000 | 5,000 | 69.6% | 20.3% | 2.0% | 8.0% |
| val | 1,000 | 250 | 71.9% | 18.6% | 1.9% | 7.6% |
| test | 2,400 | 600 | 62.6% | 24.0% | 3.1% | 10.3% |

- 解读：标签极不均衡，**测试集上“全部猜 exact”就有 62.6% 准确率**，所以 E1 以宏 F1 为主指标，准确率只作参考；complement 在测试集只有 75 条，它的 F1 波动会很大。
- 抽样后的 train 分布与官方 train 池（E 69.7% / S 19.7% / I 8.6% / C 2.1%）一致，说明按 query 分组抽样没有引入明显偏差。

**ECInstruct 统计与泄漏检查**（`reports/ecinstruct_stats.json`）
- 共 263,562 行，10 个任务。
- 与 E1 测试集（600 个 query）的重叠：ECInstruct 中由 ESCI 派生的三个任务，**train 部分重叠为 0**；val/test 部分各有 17–27 个 query 重叠。
- 结论：ECInstruct 的 train 部分以后可以安全地混入训练；它的 val/test 不能当作与 E1 独立的测试集使用。


**P11 · 调度脚本的运行方式**
- 现象：最初在本机后台用 `ssh ... 'bash chain_e1.sh | tee -a log'` 启动调度。本机休眠或 SSH 断开时，服务器上的 `tee` 写已关闭的通道会收到 SIGPIPE 退出，调度脚本下一次输出时也会被 SIGPIPE 终止，后续评测就不会被提交。
- 处理过程中又踩了一个坑：用 `pkill -f "chain_e1.sh"` 停旧进程时，执行这条命令的 SSH 会话自身的命令行里也包含 `chain_e1.sh`，于是把自己也杀了，命令没有输出（训练作业不受影响）。
- 解决：
  - 调度脚本改为 `setsid nohup bash scripts/slurm/chain_e1.sh >> logs/chain_e1.log 2>&1 < /dev/null &`，与 SSH 会话完全脱离（确认父进程为 1）。
  - 新增 `TRAIN_JOB=<作业号>`：调度脚本重启时接管已提交的训练作业，避免重复训练。
  - 本机只用短连接定期检查日志，连接断开不影响服务器上的调度。
- 经验：长时间运行的东西不要依赖本机连接；`pkill -f` 前先用 `pgrep -a` 看清会匹配到哪些进程。

**E1 基线：基座 zero-shot**（作业 44757，1 分 18 秒，2,400 条，bf16）

| 指标 | 值 |
|---|---|
| 格式合法率 | 1.000 |
| 准确率 | 0.526（低于“全猜 exact”的 0.626） |
| 宏 F1 | 0.372 |
| 各类 F1 | exact 0.706、substitute 0.328、complement 0.077、irrelevant 0.377 |
| 预测分布 | exact 1079、substitute 559、irrelevant 576、complement 186（真实为 1503 / 576 / 246 / 75） |

- 解读：基座能稳定输出合法 JSON，但判断标准明显偏严，大量 exact 被判成 irrelevant 或 complement。也就是说，基座缺的是对 ESCI 标注口径的理解，而不是格式能力，这正是监督微调要解决的问题。
- 对比说明：官方基线（micro-F1 0.62 / macro-F1 0.23）是在完整测试集上算的，而这里是 600 个 query 的子集，两者只能作量级参考，不能直接比较。

---

## S8 · E1 结果：ESCI 相关性四分类，单卡 QLoRA（2026-10-20）

**一句话结论**：QLoRA 微调把宏 F1 从 0.372 提升到 0.511（+0.139，95% CI [0.098, 0.176]），准确率从 0.526 提升到 0.692，超过“全猜 exact”的 0.626；单卡 L40S 训练 1 小时 2 分钟。

**训练过程**（作业 44759，节点 TC2N08）

| 项 | 值 |
|---|---|
| 作业总耗时 | 1:03:19（纯训练 1:02:09） |
| 吞吐 | 10.7 样本/秒，0.335 步/秒（有效 batch 32，共 1,250 步） |
| 显存峰值 | 32,645 MiB / 46,068 MiB（约 71%），GPU 平均利用率 97.7% |
| train_loss | 0.058 |
| eval_loss | 0.0700（epoch 0.32）→ **0.0632（epoch 0.96，最低）** → 0.0718（epoch 1.28）→ 0.0702（epoch 2.0） |

- 第 2 个 epoch 的验证损失回升，说明在 2 万条数据上 1 个 epoch 左右已经足够，第 2 个 epoch 开始轻微过拟合。下面的评测用的是训练结束时（epoch 2）的适配器，**不是**验证损失最低的那个。

**测试集结果**（2,400 条 / 600 个 query；评测作业各约 2 分钟）

| 系统 | 格式合法 | 准确率 [95% CI] | 宏 F1 [95% CI] | exact | substitute | complement | irrelevant |
|---|---|---|---|---|---|---|---|
| 基座 zero-shot（bf16） | 1.000 | 0.526 [0.501, 0.553] | 0.372 [0.349, 0.396] | 0.706 | 0.328 | 0.077 | 0.377 |
| QLoRA，4bit 基座评测 | 1.000 | **0.692** [0.667, 0.716] | 0.511 [0.471, 0.552] | 0.808 | 0.524 | 0.309 | 0.401 |
| QLoRA，bf16 基座评测 | 1.000 | 0.677 [0.651, 0.702] | **0.514** [0.478, 0.550] | 0.795 | 0.546 | 0.290 | 0.425 |

置信区间用 `ecomq.eval.compare` 计算：按 query 重采样 1000 次。按 query 而不是按样本重采样，是因为同一 query 下几个商品的判断彼此相关，按样本重采样会低估方差。结果在 `reports/e1-compare.json`。

| 差值 | 准确率 [95% CI] | 宏 F1 [95% CI] | 结论 |
|---|---|---|---|
| QLoRA(4bit) − 基座 | +0.165 [0.138, 0.191] | +0.139 [0.098, 0.176] | 显著提升 |
| QLoRA(bf16) − 基座 | +0.151 [0.125, 0.176] | +0.142 [0.107, 0.175] | 显著提升 |
| 4bit − bf16 评测 | +0.015 [0.000, 0.027] | −0.003 [−0.022, 0.015] | 准确率处于显著边缘，宏 F1 无差异 |

**误差分析**（4bit 评测的混淆矩阵，行 = 真实，列 = 预测）

| 真实 \ 预测 | exact | substitute | complement | irrelevant | 召回 |
|---|---|---|---|---|---|
| exact（1503） | **1255** | 189 | 14 | 45 | 0.835 |
| substitute（576） | 237 | **302** | 8 | 29 | 0.524 |
| complement（75） | 27 | 14 | **19** | 15 | 0.253 |
| irrelevant（246） | 84 | 71 | 7 | **84** | 0.341 |

1. **最大的错误来源是 exact ↔ substitute 互相混淆**（189 + 237 = 426 条，占全部错误的 58%）。这两类的边界本身就模糊（“缺了一个要求”算不算替代品），也是 ESCI 标注一致性最低的地方。
2. **irrelevant 召回只有 0.34**：155 条无关商品被判成 exact 或 substitute。模型倾向于“宽松匹配”，这和基座正好相反（基座偏严）。对搜索业务来说，把无关商品放进结果比漏掉相关商品代价更高，这是下一步要优先改的。
3. **complement 召回 0.25**：训练集只有 408 条（2%），样本太少。

**P8 在真实数据上的复核**
- 冒烟时发现 QLoRA 适配器配 bf16 基座会让边界样本翻转。在 2,400 条真实数据上：两种加载方式的**宏 F1 没有差异**，但**预测分布明显不同**（4bit 预测 exact 1,603 条，bf16 只有 1,378 条，后者更多判为 substitute）。
- 结论：量化方式不一致主要改变的是决策边界（校准），而不是整体质量。如果部署时用合并后的 bf16 模型，各类别的精确率和召回会变，需要按部署形态重新评测，不能直接沿用 4bit 的数字。

**与公开结果的关系**
- 官方 Task 2 基线（micro-F1 0.62 / macro-F1 0.23）是在完整测试集上算的，E1 测试集是其中 600 个 query 的随机子集，只能作量级参考；不能写成“超过官方基线”。

**下一步（E2 候选，按预期收益排序）**
1. **训练 1 个 epoch，或选验证损失最低的 checkpoint**：验证曲线显示第 2 个 epoch 过拟合，成本最低。
2. **类别重采样**（提高 irrelevant 与 complement 的比例）：针对召回最差的两类。
3. **加大数据量**（2 万 → 5 万到 10 万）：画出“数据量–宏 F1”曲线；可以混入 ECInstruct 的 train 部分（已确认与测试集无重叠）。
4. **输入加入品牌、商品要点等字段**：可能有助于区分 exact 和 substitute。

**小修复**
- 调度日志的时间戳：原写法 `echo "$(date) ... $(wait_job)"` 中 `date` 在提交时就已求值，日志里看起来像评测作业“提交即完成”。已改为先等作业结束再取时间；调度顺序本身经 `sacct` 核实无误（三个作业依次运行）。


---

## S9 · E2 设计：训练轮数与类别重采样（2026-10-20）

**为什么**
- E1 暴露了两个问题：验证损失在 epoch≈1 最低、第 2 个 epoch 回升（过拟合）；irrelevant 召回 0.34、complement 召回 0.25（少数类学不好）。
- 用户要求同时做“1 个 epoch + 选最优 checkpoint”和“提高 irrelevant、complement 的采样比例”。

**设计：拆成两个对照实验，而不是一次改两个变量**

| 实验 | 训练数据 | 训练设定 | 和谁比、回答什么 |
|---|---|---|---|
| E1（已完成） | 自然分布 | 2 epoch，用最后的适配器 | — |
| E2a | 与 E1 完全相同 | 1 epoch，每 100 步评测，加载验证损失最低的 checkpoint | E2a vs E1：只改训练轮数与 checkpoint 选择的效果 |
| E2b | 重采样（complement 10%、irrelevant 20%） | 与 E2a 完全相同 | E2b vs E2a：只改采样比例的效果 |

如果一次同时改两个变量，即使分数提升了，也说不清是哪个改动起的作用；拆开只多花一次约 35 分钟的训练。

**重采样怎么做**（`prepare_esci.py --rebalance complement=0.10,irrelevant=0.20`）
- 只改 train，总量保持 20,000 条：从训练池中抽取**此前未使用过的真实样本**，把 complement 补到 10%、irrelevant 补到 20%；exact 与 substitute 按原有比例随机删减。
- 不复制已有样本：复制只会让模型反复看同几条 complement，更容易过拟合；训练池里有约 2.9 万条 complement、12 万条 irrelevant 可用。
- val/test 保持自然分布，并用同样的种子生成。调度脚本在训练前用 `cmp` 核对 E2b 的 val/test 与 E1 **逐字节一致**，不一致就停止，保证三组实验在同一测试集上比较。
- 预期的代价：训练分布偏离测试分布（测试集 exact 占 62.6%），所以 E2b 的准确率可能下降，宏 F1 和少数类召回应当上升。这是预期内的取舍，以宏 F1 为主指标判断。

**最优 checkpoint 怎么选**
- `load_best_model_at_end: true` + `metric_for_best_model: eval_loss`：每 100 步在验证集上算一次损失并保存，训练结束时把损失最低的那个作为最终适配器。
- 局限：选择依据是验证损失而不是宏 F1。损失对 exact 这样的多数类更敏感，不一定与宏 F1 同步；如果 E2 的结果显示两者不一致，再考虑用生成式评测在验证集上选 checkpoint。

**调度**
- `scripts/slurm/chain.sh`：通用的“数据准备 → 训练（超时自动续训）→ 评测”调度，取代只能跑 E1 的 `chain_e1.sh`。
- `scripts/slurm/run_e2.sh`：先生成 E2b 数据并做一致性核对，再依次跑 E2a、E2b；用 `setsid nohup` 在头节点独立运行。

**P12 · 参数里的逗号被 sbatch 拆开**
- 现象：代码审查时发现，提交前修复。`--export=ALL,SKIP_INSPECT=1,PREP_ARGS="--rebalance complement=0.10,irrelevant=0.20"` 中，sbatch 用逗号分隔变量，`irrelevant=0.20` 会被当成一个新的环境变量，`PREP_ARGS` 被截断，数据会按错误的参数生成。
- 解决：含逗号的值先 `export` 到环境里，`--export=ALL` 会继承当前环境；`--export=` 列表只放不含逗号的值。
- 经验：`--export` 的值里不能有逗号；复杂参数一律走环境变量或写进文件。
