# ReDAG 交接手册

这套流水线把论文 §5 的全部工作（数据 → 三个 SFT → 拒绝采样 → 交替 RL → 实验 1/2/3）排成一张任务图，由一个调度器按 GPU 自动派活：

- **一条命令**就能启动或续跑；同一条命令什么时候重复执行都安全。
- 断电、关机、迁移、进程崩溃之后，再执行同一条命令即可续跑：已完成的任务直接跳过，没跑完的任务从最近的检查点接着跑。
- 卡数是 1、2、4、8 都能自适应。

---

## 0. 一页速览

| 你需要 | 说明 |
|---|---|
| GPU | ≥ 80 GB 显存。RTX PRO 6000 96G 最合适：RL 单卡就能跑。A800、H800 80G 跑 RL 需要同机两张。48G 及以下不行 |
| 数据盘 | ≥ 300 GB（模型 120 GiB，数据约 40 GiB，运行产物约 100 GiB） |
| 代码 | GitHub `NonEuclideanSpace/ReDAG` 的 `data-pipeline` 分支（私有仓库，需要读权限） |
| 基础镜像 | Ubuntu 22.04、Python 3.12、torch 2.12.1+cu130（驱动 ≥ 580），或者直接用共享给你的镜像 |

```bash
cd /root/autodl-tmp && git clone https://github.com/NonEuclideanSpace/ReDAG.git && cd ReDAG && git checkout data-pipeline
bash scripts/launch/bootstrap.sh     # 装依赖、下载模型（约 120 GiB）、自检；无卡模式下跑最省钱
bash scripts/launch/go.sh            # 无卡模式：只跑 CPU 数据阶段；有卡模式：跑全流程
bash scripts/launch/go.sh status     # 看进度：每张卡在跑什么、进度、预计完成时间
```

> 访问 GitHub 前先执行 `source /etc/network_turbo`（AutoDL 学术加速）。HF 镜像和 ModelScope 不需要。

---

## 1. 三种交接方式

| 方式 | 适合 | 步骤 |
|---|---|---|
| **A. 同账号，直接用现有实例** | 最快，零准备 | 在控制台开机（有卡模式）→ `bash scripts/launch/go.sh`。所有进度都在数据盘上 |
| **B. 同账号，克隆实例** | 原机没卡可用 | 控制台"克隆实例"，选有空卡的主机 → 新实例开机 → `go.sh`。克隆会带上模型、数据和进度 |
| **C. 你自己的账号** | 独立预算 | 见下 |

**方式 C 的步骤：**
1. 交接人在控制台：关机 → 更多 → 保存镜像 → 镜像页"共享"，填你的**用户 ID**（在页头用户名下拉里）。镜像只包含系统盘（环境），不包含数据盘。
2. 你用这个镜像创建实例（先用无卡模式），然后 `git clone` 代码，再执行 `bash scripts/launch/bootstrap.sh`（模型从 ModelScope 断点续传下载）。
3. 数据二选一：
   - **推荐：用交接人导出的数据包**，保证和原实验用的是完全相同的数据，包括 LLM 标注出的条件图：
     - 交接人执行 `bash scripts/launch/go.sh bundle export data --split 3900`，把生成的分卷和 `*.manifest.json` 上传网盘（AutoPanel 里的"公网网盘"支持阿里云盘和百度网盘）。
     - 你下载后执行 `bash scripts/launch/bootstrap.sh --skip-models --data-bundle <manifest>`。导入时会校验 sha256，并自动跑一遍数据检查。
   - 自己重建：直接执行 `go.sh`。但 GPU 上的 LLM 标注每次的结果会略有不同，数据不会和原版完全一致。
4. 切到有卡模式，执行 `bash scripts/launch/go.sh`。

---

## 2. 日常操作（全部通过 `bash scripts/launch/go.sh …`）

| 命令 | 作用 |
|---|---|
| `go.sh` | 在后台启动或续跑；已经在跑时只显示状态 |
| `go.sh status` | 显示每张卡在跑什么、进度、ETA，各阶段完成数，失败项，以及预计完成时间（`--json` 给监控脚本用） |
| `go.sh logs JOB [-f]` | 查看某个任务的日志 |
| `go.sh stop` | 立刻停止。正在跑的任务被终止，下次从检查点续跑 |
| `go.sh drain` | 正在跑的任务做完再停，不再启动新任务。适合计划关机前用 |
| `go.sh retry` | 启动，并重试"在当前代码下失败过"的任务。修好代码后不需要这条：代码一变，失败的任务会自动重试 |
| `go.sh reload` | 只重启调度器（`git pull` 之后用），正在跑的任务不受影响，会被新调度器接管 |
| `go.sh restart JOB…` | 让正在跑的任务换上新代码：从检查点续跑，不算一次失败 |
| `go.sh preflight` | 手动重跑 GPU 预检（约 2 分钟），改动较大的代码之后用 |
| `go.sh plan --gpus 1 2 4 8` | 估算换成不同卡数需要多久 |
| `go.sh reset JOB [--dependents]` | 让某个任务（及其下游）重新跑。输出会保留，可续跑的任务会跳过已完成的部分 |
| `go.sh list` | 列出全部约 130 个任务、依赖关系和预估时长 |

**断点续跑的粒度：**

| 任务 | 粒度 |
|---|---|
| SFT | 每 20 分钟（`CKPT_MINUTES`）和每个 epoch 末各存一次检查点 |
| RL | 每一步都存检查点（含优化器、随机数状态、数据顺序），续跑结果与不中断时逐步一致 |
| 评测、拒绝采样、图标注 | 每完成一个会话（或一个状态）就落盘 |
| 画布、状态打分 | 按文件续跑 |

---

## 3. 时长与成本（按 A800 速度估算，开跑后会用实测速度修正）

| 卡数（96G，RL 单卡） | 总时长 | 计费卡时 | 比 1 张多花 |
|---|---|---|---|
| 1 | 8.2 天 | 197 | — |
| 2 | 4.1 天 | 198 | +0.5% |
| 3 | 2.8 天 | 199 | +1% |
| 4 | 2.1 天 | 202 | +2.5% |
| 6 | 1.4 天 | 209 | +6% |
| 8 | 1.2 天 | 231 | +17% |

- 全用 80G 卡时，RL 每次要两张，总量约 228 卡时。
- 结论：**1 到 4 张卡总花费几乎一样，按截止时间选卡数即可**；超过 6 张会明显浪费，因为关键路径约 29 小时，卡再多也缩不短。
- 运行中随时用 `go.sh status` 看实际的预计完成时间。

---

## 4. 多台机器一起跑（每台一张卡也可以）

每个任务只需要一张 ≥80G 的卡，任务之间只通过文件交接，所以不在同一台机器上的卡也能用：

1. 用"克隆实例"得到第二台机器。最好在数据阶段跑完之后再克隆，这样数据已经在盘上。
2. 用**同一个正则**说明哪些任务交给第二台：在主机上它是 `EXTERNAL_JOBS`（这些不在本机跑），在第二台上它是 `LOCAL_JOBS`（只跑这些）。CPU 数据阶段和预检每台机器都会各自跑。RL 必须留在 96G 卡上。例如：
   ```bash
   # 主机 local.env
   EXTERNAL_JOBS='^(sft_editor|exp1_s\d+|exp3_(zero_shot|sft)_s\d+)$'
   # 第二台 local.env
   LOCAL_JOBS='^(sft_editor|exp1_s\d+|exp3_(zero_shot|sft)_s\d+)$'
   ```
3. 在主机上配置到第二台的免密 ssh，然后在主机的 `local.env` 里写上 `SYNC_PEER`。此后每次执行 `go.sh` 都会自动启动同步（重启后也是）。同步是双向的，新文件覆盖旧文件，且先传输出、后传完成标记，所以不会出现"标记到了文件还没到"：
   ```bash
   ssh-keygen -t ed25519 -N '' -f ~/.ssh/redag_sync    # 把 ~/.ssh/redag_sync.pub 追加到第二台的 ~/.ssh/authorized_keys
   echo "SYNC_PEER=root@connect.<地区>.seetacloud.com:<端口>" >> local.env    # 第二台的 SSH 登录指令里的地址和端口
   ```
4. 两台机器都执行 `go.sh`。对方跑完的任务一同步过来，本机的下游任务就会自动开始。三台以上时，主机的 `SYNC_PEER` 写成空格分隔的多个地址，主机作为中转和每台各同步一路。
5. **谁先空谁来跑**（可选，推荐）：`SHARED_JOBS` 里的任务两台都能跑，哪台先空出卡就"认领"哪个，认领记录放在第二台上（第一台通过同一把 ssh 密钥去认领），所以同一个任务不会被跑两次。`SHARED_JOBS` 优先于 `EXTERNAL_JOBS`/`LOCAL_JOBS`。
   ```bash
   # 两台 local.env 写同一行；只在主机上加 CLAIMS_HOST（= 第二台的地址，和 SYNC_PEER 一样）
   SHARED_JOBS='^(annotate_shard[0-7]|canvases_shard[0-3]|rs_part[0-9](_[0-9])?)$'
   CLAIMS_HOST=root@connect.<地区>.seetacloud.com:<端口>
   # 长的旁支任务等关键路径过去再开始，免得占住对方马上要用的卡（只影响先后，不影响结果）
   NOT_BEFORE='^(sft_planner|eval_planner_zero_shot)$>rs_merge'
   ```
   认领了任务的机器如果没了，用 `go.sh unclaim JOB` 交还，另一台会接手。第二台连不上超过 15 分钟（`CLAIMS_FALLBACK_MIN`）时，主机不再等认领记录，最坏是某个任务两台各跑一遍，浪费卡时但结果不受影响。

   跑 RL 的机器如果只有一张卡，不要把长的旁支任务（`sft_planner`、`eval_planner_*`、`precompute_rl_states`）放进它的 `SHARED_JOBS`，而是写进它的 `EXTERNAL_JOBS`。单卡机器上调度器不会给"马上就绪"的任务预留 GPU：`rs_merge` 一完成，`sft_planner`（4 小时）就会在 `check_controller_sft` 那一分钟里抢走这张卡，控制器 SFT 和 RL 跟着推迟 4 小时。
6. 某台机器暂时无事可做（在等对方的任务）时，调度器会一直等着，不会退出。

---

## 5. 规则（违反了会被拒绝，或者导致结果不一致）

- **不要中途改影响结果的设置**，例如 `STEPS_EVAL`、`GEN_BUDGET`、分片数、模型路径。第一次运行时这些设置会记入 `runs/state/config.json`，之后设置不同就拒绝启动。确实要改时加 `--accept-config-change`，并重置受影响的任务。
- **不要同时开两个调度器**，也不要手动跑旧的 `0x_*.sh`。它们现在只是调度器的包装，会被拒绝。
- 不要删 `runs/state/`、`runs/logs/*.done`。要重跑请用 `go.sh reset`。
- 停任务请用 `go.sh stop` 或 `drain`，不要 `kill -9` 作业进程。
- 本机专用的设置写在仓库根目录的 `local.env` 里，它不进 git。可写的有：`NOTIFY_URL`、`AUTO_SHUTDOWN`、`FETCH_WORKERS`、`WORKERS`、`CPU_SLOTS`、`ANNOTATE_BATCH`、`HOLD`、`EXTERNAL_JOBS`/`LOCAL_JOBS`、`SHARED_JOBS`、`CLAIMS_HOST`、`NOT_BEFORE`、`SYNC_PEER`。

---

## 6. 出问题怎么办

| 现象 | 处理 |
|---|---|
| `status` 显示 `FAILED job` | 先 `go.sh logs job -n 80` 看原因。修好代码就会自动重试；如果是偶发问题（网络、别的进程占了显存），执行 `go.sh retry` |
| 预检 `PREFLIGHT FAILED` | 看 `runs/logs/preflight.log` 和 `runs/logs/wiring_check.log`。常见原因是驱动太旧（torch cu130 需要驱动 ≥ 580）或权重不完整（执行 `python scripts/launch/download_models.py`） |
| RL 显存不够（单卡） | 在 `local.env` 写 `RL_LAYOUT=split`（需要同机两张卡），然后 `go.sh reload`。这期间评测任务照常在跑 |
| 某张卡被其他进程占用 | `status` 会显示 `GPUs used by other processes`，调度器会自动绕开这张卡 |
| 实例关机、迁移、重启 | 开机后执行 `go.sh`，所有任务从检查点续跑 |
| 磁盘不足 | `df -h /root/autodl-tmp`，可以删 `runs/wiring`、`runs/smoke`、`/root/autodl-tmp/bundles` |
| GitHub 拉不动 | 先 `source /etc/network_turbo` |
| 图标注慢、爆显存 | 日志里每批有一行 `generate:`（每步毫秒数、答案长度）。爆显存时 batch 自动减半，连续 3 批成功后再加回去；也可以在 `local.env` 里调 `ANNOTATE_BATCH`（96G 用 40，80G 用 24）后 `go.sh reload`。解码注意力的加速可以用 `ANNOTATE_GQA_DECODE=0` 关掉 |
| 改了代码，想让正在跑的任务用上 | `git pull` 之后执行 `go.sh restart 任务名`：从检查点续跑，不算一次失败 |

---

## 7. 结果在哪

| 内容 | 位置 |
|---|---|
| 实验 1、2、3 的合并结果 | `runs/eval/exp1.json`、`exp2.json`、`exp3.json` |
| 汇总 | `runs/RESULTS.md` |
| 规划器评测（零样本 vs SFT） | `runs/eval/planner_zero_shot.json`、`planner_sft.json` |
| 训练日志 | `runs/sft/*/metrics.jsonl`、`runs/rl/**/metrics.jsonl` |
| 实验 3 的阶段检查点 | `runs/rl/main/stage_checkpoints.json` |
| 人工盲评 | `runs/eval/exp3.blind_pairs.json` 和 `exp3_images/blind_*`；答案在 `exp3.blind_key.json`，不要发给评审人 |

- 打包回传：`go.sh bundle export results`，要连 LoRA 一起带上时加 `--with-adapters`，导出的文件放在 `/root/autodl-tmp/bundles/`。

---

## 8. 通知与自动关机（写进 `local.env`）

```bash
NOTIFY_URL=https://sctapi.ftqq.com/<SendKey>.send      # Server酱。也支持 PushPlus、飞书/钉钉/企业微信机器人、Bark、ntfy
AUTO_SHUTDOWN=done                                     # 全部完成后 10 分钟关机（SHUTDOWN_GRACE_MIN 可调）
```

- `go.sh notify-test` 可以发一条测试消息。
- 会收到通知的事件：调度器启动（附预计完成时间）、任务最终失败、各里程碑完成、全部完成、流程卡住。
- `AUTO_SHUTDOWN=idle` 表示流程卡住时也关机。省钱，但关机后这张卡可能被别人租走，按需选择。

---

## 9. 交接清单（交接人逐项打勾）

- [ ] GitHub：把合作者加为 `NonEuclideanSpace/ReDAG` 的协作者
- [ ] AutoDL：方式 A/B 告知实例；方式 C 共享镜像给对方用户 ID
- [ ] 数据：`go.sh bundle export data --split 3900`，上传网盘，把链接发给对方
- [ ] 把当前 `go.sh status` 的输出和 `local.env` 发给对方，说明哪些任务已完成、哪些在跑
- [ ] 对方配置好 `NOTIFY_URL`，并用 `go.sh notify-test` 确认能收到

---

## 10. 2026-10-31 大规模实验：状态、结论与学校服务器续跑

### 10.1 本次加入的代码（`data-pipeline` ≥ `846b372`；新开关默认全关）

| 改动 | 提交 | 是否采用 |
|---|---|---|
| 受影响子图按编辑足迹重叠（≥ 10%）+ 依赖后代 + 交互邻居 + 同短语计数 + 受保护禁止 | `78b4afe` | 采用（A/V 从 0.96 降到 0.61） |
| 基线 flat_full / tree_rollback_only 整图编辑、单阶段验证（此前 51 组旧规则会话不可合并） | `44c938a` | 采用 |
| 同一目标连续失败 2 次后换目标 | `5cbdf14` | 采用 |
| `--editor_work_side` 工作分辨率渲染 | `502ef50` | 未采用：36 个小窗口 20 vs 20 成功 |
| `--protect_masks` 受保护实体掩码 | `0d7e28b` `34c04de` `2eaef97` | 未采用：v2 成功 12→8，v3 成功 14→8 |
| `--cost_budget/--generation_cost`、`--budget_per_turn` | `8660198` | 协议选项 |
| 拒绝采样 `--hard_multiplier`，每组候选写入 `<output>.groups.jsonl` | | 供成对离线训练 |
| 缓存行记录 `code`、`editor`、`variant`；合并前按此过滤 | | 采用 |
| 调度器不把显存仍被占用的卡分出去 | `17a550b` | 采用 |
| 边增强：`--closure_instructions`（编辑实体时把同框内未满足的属性/关系写进指令）、`--confirm_drops`（仅因依赖边进入检查的受保护节点下降时，先比较编辑前后的放大证据区域，像素未变或放大分数未降则不算损伤）、`--soft_gating`（只有实体存在是硬前提，其余父节点只影响顺序） | `dfd43fd` `846b372` | 评估中：`runs/eval/exp1_redag_edge.s*` |

`local.env` 可写项：`EVAL_EDITOR_ARGS`、`TRAIN_EDITOR_ARGS`、`COST_BUDGET`、`GENERATION_COST`、`BUDGET_PER_TURN`、`RS_HARD_MULTIPLIER`。`COST_BUDGET` 等尚未进入 `RESULT_KEYS`，开新实验前请加入并重置相关任务。

### 10.2 结论（同记录同种子配对；“接受数” = 每会话被接受的编辑数；评测参数 `--max_steps 4 --generation_budget 8 --verification_budget 400`）

1. **对整图基线，越复杂优势越大**（75 对，s0/s1/s10）：

   | | 对 flat_full | 对 tree_rollback_only |
   |---|---|---|
   | 全部 | 1.59 vs 1.17（好 27 / 差 9，p = 0.004） | 1.59 vs 1.25（27 / 13，p = 0.04） |
   | K = 10 | 1.56 vs 1.48（不显著） | 1.56 vs 1.60（不显著） |
   | K = 30 | 1.85 vs 1.00（14 / 2，p = 0.004） | 1.85 vs 0.96（15 / 2，p = 0.002） |
   | 轮数 ≥ 3 | 1.68 vs 1.10（20 / 4，p = 0.002） | 1.68 vs 1.18（19 / 6，p = 0.015） |
   | 依赖密集 | 1.74 vs 1.18（16 / 5，p = 0.03） | 1.74 vs 1.24（16 / 6，p = 0.05） |

   损伤（候选打坏的已满足条件）2.9 vs 8.2 / 会话；验证调用 77 vs 108。整图编辑每次打坏 12% 已满足条件，全部存活率随已满足数 m 近似按 0.88^m 下降（`pm_curve_1708.txt`）。
2. **对无边版 spatial_no_gates（50 对）不占优**：1.62 vs 1.70（11 / 13），K = 20 时 1.38 vs 1.81（1 / 7，p = 0.07）；验证调用 78.5 vs 67.8。依赖边的代价是检查范围更大、门控堵塞前沿。边增强开关针对这三点。
3. **共同裁判（Qwen3-VL-8B 对最终图逐条打分，不分模式、不门控）**：满足率四种方法都在 0.72–0.76。依赖一致满足（子条件只有在所有父条件都满足时才计）：对 flat +0.64（22 / 14，p = 0.24），对 tree +0.64（24 / 14，p = 0.14），对无边版 +0.30（13 / 7，p = 0.26）。每会话只接受 1–2 次编辑，最终满足率变化 ≤ 0.05：编辑器成功率是瓶颈。该裁判与环内验证器同一模型族，不是独立评价。
4. **RL vs SFT**（58 个未见状态）：贪心接受率 0.517 / 0.552 / 0.586（SFT / RL 控制器 / RL 双方），随机采样 0.517 / 0.483 / 0.500，不显著。
5. **exp2 等步数**（控制器阶段各 16 步）：奖励方差终端广播 0.063 ≫ 全表 0.009 > 子图信用 0.006；16 个开发状态测不出策略差异。

### 10.3 数据现状（分析素材在 M2 `/root/autodl-tmp/report_materials`；各机 `runs/eval/` 互相同步）

- `exp1.s0/s1/s10.cases.jsonl`：flat / tree / redag 各 25，完整；`exp1.s2`：flat 25 + tree 3。
- `exp1_spatial_no_gates.s0/s1`：完整；s2–s8 与 `exp1_redag_full.s2–s8` 由各机核心链依次生成。
- `exp1_redag_edge.s0–s3`：边增强版，每片跑完用共同裁判打分（`judge_edge_s*.jsonl`）。
- `exp1_turnbudget.s0`：K = 30、每轮生成预算 4，flat / spatial / redag 各 8 条记录。
- 分析：`python scripts/paired_report.py --base redag_edge/redag_full --judges <裁判 jsonl …>`（配对表、按 K / 轮数 / 依赖密度拆分、组件统计、裁判满足率与依赖一致满足率）；裁判打分：`python scripts/judge_finals.py --cases runs/eval/exp1*.cases.jsonl --model <VLM> --out <jsonl>`（可续跑）。

### 10.4 学校服务器上按优先级做（实测每会话 4–6 分钟，PRO6000 / A800）

1. **边增强版定论**：对 12 片补齐 redag_edge、redag_full、spatial_no_gates（每种 300 会话；扣除已完成部分约 700 会话 ≈ 60 GPU 小时）。命令与调度器的 exp1 任务相同，只改模式、输出名并加开关：

   ```bash
   python scripts/run_eval_benchmarks.py <调度器 exp1 任务的模型与适配器参数> \
     --max_steps 4 --generation_budget 8 --verification_budget 400 --num_denoise_steps 28 \
     --exp_id 1 --data data/final/eval_main.jsonl --num_shards 12 --shard <k> \
     --exp1_modes redag_full --output runs/eval/exp1_redag_edge.s<k>.json \
     --closure_instructions --confirm_drops --soft_gating
   ```

   无边版用 `--exp1_modes spatial_no_gates --output runs/eval/exp1_spatial_no_gates.s<k>.json`，不加开关。采用标准（事先定好）：对无边版接受数不低且依赖一致满足更高 → 作为正式方法；只在依赖密集或多轮子集更好 → 写成条件结论；否则如实写“边只改善一致性”。
2. **独立评价**：换一个不同家族的 VLM 做裁判（`scripts/judge_finals.py --model <其他家族 VLM>`），并用 50 张图的人工标注校准；再加人工盲评。
3. **R4 两框动作**（`r4_multi_window_design.md`）→ `RS_HARD_MULTIPLIER=3` 重新拒绝采样 → 控制器 SFT → rl_main。
4. **RL 测量**：`N_RL_DEV` ≥ 64，每 8 步评估一次，`STEPS_RL` 与 `STEPS_EVAL` 对齐。
5. **协议子集**：`BUDGET_PER_TURN=1` 下 K = 30 的三种方法各 100 条（约 25 GPU 小时），检验“越复杂越有优势”在等预算下是否成立。
6. exp3 五阶段消融。
