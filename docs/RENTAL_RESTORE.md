# 租用实例交付包恢复说明

这份说明用于恢复外接盘中的小模型实验。各部分完成状态、唯一推荐代码版本及未完成项以顶层 `DELIVERY_INDEX.json` 为准。用户已将范围冻结为交付现有全部成果：正式评测为 Base 141、SFT 320、Agent RL 160、Joint RL 16，共 637/1280 条；剩余 643 条未形成完成的正式评测记录，不能算作完整评测。执行恢复应使用新的 Linux x86_64 工作环境，不能覆盖正在训练的实例。

备份根目录为 `dag-pilot-rental-backup-20261027T130933Z-9769b4`。外盘上的 `remote/agentic-image-pilot-small` 是早期未完成的直接同步副本，`local/pilot-small` 也不是最终代码入口；均不能单独作为完整交付。实际实验数据来自 `remote/project-results-archives` 中有对应 manifest 的增量包。忽略 `._*`、`.part`、`.parts` 和传输日志。

早期自动保留策略裁剪过部分中间检查点的权重和训练状态。具体阶段与步数见 `DELIVERY_INDEX.json` 的 `known_missing` 及 `reports/historical-checkpoint-delivery-summary.json`。这些位置仍有 COMMITTED 元数据，但不能据此恢复权重；各阶段最终检查点和实际保留的完整检查点另有文件证据。

## 先生成可检查的恢复计划

直接使用外盘顶层 `tools/restore_rental_bundle.py`，无需先解压代码才能开始。恢复后的项目中也包含同一入口 `scripts/restore_rental_bundle.py`。本脚本生成计划时只读取逐包 manifest、文件长度及小型代码包，不扫描大模型或实验归档的哈希。执行恢复时才对选中的归档逐包解压并同步计算 SHA-256。

```bash
python3 /path/to/backup/tools/restore_rental_bundle.py \
  --backup-root /path/to/dag-pilot-rental-backup-20261027T130933Z-9769b4 \
  --plan /path/to/restore-plan.json
```

默认从顶层 `DELIVERY_INDEX.json` 选择代码包并核对记录的 SHA-256；没有顶层索引时才读取 `code-revisions/code-update-order.json`。也可用 `--code-archive` 显式指定已验证的完整代码包。不要将历史补丁包再叠到新的完整代码包上。覆盖保留历史文件，不会删除新代码包里已不存在的旧代码文件，因此报告会明确 `code_overlay_exact_file_set=false`。

恢复计划记录生成时的备份路径，应在执行恢复的机器上生成。跨机器复制的旧计划只能作审计记录；挂载位置变化后重新生成计划，不改写原实验配置来凑路径。

计划必须满足 `ready=true` 且 `errors=[]`，才允许执行。`ready` 只表示当前已选目录的归档/成员具备恢复条件；不表示最终训练已结束、所有未来文件已回传或模型环境已验证。若传输仍进行，manifest 可能早于对应完整压缩包到达，此时应等待回传并重新生成计划，不能把 `.part` 当成归档。

```bash
python3 /path/to/backup/tools/restore_rental_bundle.py \
  --plan /path/to/restore-plan.json \
  --destination /new/empty-parent/agentic-image-pilot-small \
  --execute
```

目标目录必须尚不存在。脚本不接受覆盖现有训练目录，也不自行启动队列、服务或 GPU 工作。失败时保留已恢复内容与同级的 `*.restore-report.json`；修复原因后请使用另一个新目录重做，不能把不完整目录标成恢复成功。

## 增量版本与软链接语义

- 逐包名称中的 generation 与 chunk 序号决定顺序，不使用外接盘文件修改时间排序。
- 同一路径只取最后一个稳定版本；若最后一次记录仍为不稳定读取，计划报错。实际执行按归档顺序读取选中的包，每个选中成员仅写一次，再覆盖所选完整代码包。
- 源端后来删除的文件仍保留在备份中，不执行删除。这是历史保留语义，不是最终源文件系统逐字节镜像；历史文件不能自动算作当前有效实验结果。
- 接受普通文件、目录、安全的内部软链接，以及目标仍为同一已存版本的硬链接。内部绝对链接 `/workspace/agentic-image-pilot-small/...` 会改为指向新恢复根目录内的相对链接。外部链接、越界路径、特殊文件和目标版本已变化的硬链接必须先明确处理，不能静默追随。
- 权限的普通 `rwx` 位和时间被恢复；不恢复原所有者或 setuid/setgid 等特殊权限。
- 不应直接启动归档中的旧 `deployment` 队列状态。旧 PID、退出码和 started 文件是实验记录，不是新机器上的进程状态。

## 模型、环境与检查点是不同交付项

| 外盘位置 | 对应内容 / 目标 |
|---|---|
| `public-model-snapshots/actor/<revision>` | 复制到项目 `models/actor/<revision>` |
| `public-model-snapshots/renderer/<revision>` | 复制到项目 `models/renderer/<revision>` |
| `public-model-snapshots/judge/<revision>` | 复制到项目 `models/judge/<revision>` |
| `remote/unique-models-archives` | 包内路径从 `models/actor-4b-nf4`、`models/judge-8b-nf4`、`models/tools` 开始 |
| `remote/environments/linux-environments` | `py312.tar.gz`、`tools.tar.gz`、`support.tar.gz` 和环境 manifest |
| `remote/environments/proot-runtime.tar.gz` | 已使用的 proot 5.1.0、libtalloc、版权和手册；对应 `proot-runtime.manifest.json` |
| `remote/priority-checkpoints` | SFT / Agent RL / joint RL 最终可恢复包，是否完成以对应实际收据为准 |
| `code-revisions` | 最新完整代码包及历史修复记录 |
| `reports`、`manifests` | 完成证据、评测结果、恢复和传输收据 |

本实验固定使用以下精确模型版本：

- Qwen3.5-4B：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- FLUX.2-klein-base-4B：`a3b4f4849157f664bdbc776fd7453c2783562f4d`
- Qwen3-VL-8B-Instruct：`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`

本脚本只恢复项目增量与完整代码，尚不负责解压以上模型/环境/最终恢复包。不要将项目恢复成功误写为“整套离线恢复成功”。

## 环境路径与恢复顺序

保存的环境保留原 Linux 前缀：`/usr/local/miniconda3/envs/py312` 和 `/workspace/agentic-image-pilot-small/.venv-tools`。环境 `support` 包还包含 `.runtime`、`third_party` 及较早的 `src`、`pyproject.toml`、`scripts/serve_tools.py`。若合并到同一项目树，环境支持文件必须先恢复、最新完整代码最后覆盖。本文的命令将环境解压到独立根目录，不复制旧 support 源码到项目；GPU 主路线显式使用最新项目的 `src` 和已恢复的基础模型，CPU proot 检查另有明确绑定，因此不受两次解压先后顺序影响。

`restore_rental_environments.py` 要求原前缀不存在，不支持任意目录重定位；直接在已恢复项目之后运行它会与现有 `src` 等路径冲突。推荐在新 Linux 主机上将项目恢复到尚不存在的原路径 `/workspace/agentic-image-pilot-small`，将环境独立解压到 `/workspace/env-restore-check/root`。CPU 环境检查保留 proot 路线；主 GPU 恢复使用 `--execution native`，直接执行该解包树内的 Python，并将 `PYTHONHOME` 指向同一物理 prefix，不通过 proot 执行 GPU 内核。当前 GPU 恢复检查还会验证 `project.resolve()` 等于 manifest 中的原项目路径；单纯把项目放到任意物理目录再传给该检查并不支持。若选择原生原前缀安装，需要人工规划空目录顺序并明确合并环境支持文件，本脚本不执行这种合并。

只有改顶层 YAML 不足以重定位：环境 shebang、包元数据、检查点配置、评测注册信息、历史绝对路径与软链接也会引用原位置。本流程保留原项目路径与归档内部目录结构，只由启动器显式设置主 Python 的解包物理 prefix；工具环境的 CPU 检查仍保留原虚拟前缀。历史证据原件不应为重定位而改写。

## 实际运行配置与权重

默认 `configs/pilot-small.yaml` 保留早期试验参数，不能代替已完成实验配置。使用最终恢复包中的 `committed_restore_config.json` 及对应 COMMITTED 检查点。实际选定采样器为 50 步、`eta=0.15`，不要无意回退默认配置中的旧值。

最终恢复包包含两侧适配器、优化器、调度器、RNG、训练游标及冻结参考。基础模型、任务/校准数据和完整 rollout 记录在其他交付项中，最终权重包不能单独代替整套实验数据。

`check_restored_environment_gpu.py --execution native --recovery-archive ... --recovery-manifest ... --execute` 在独立解压副本上通过 `JointTransaction.load_committed` 恢复两侧状态，保留原双卡映射，并运行真实样本检查。GPU 主路线不依赖 proot/libtalloc。它仍依赖宿主 Linux 内核、NVIDIA 驱动及设备节点、glibc/系统 ELF 库，以及 GPU JIT 使用的 C 编译器、binutils、头文件和 `ldconfig`。基础模型与当前项目代码仍是显式复用的宿主输入，并没有因这次检查而独立恢复验证。

2026-10-28 11:34:13 UTC，原租用实例上的必要恢复 retry3 已实际完成，进程退出码为 0。`deployment/env-restored-gpu-check.json` 记录 `status=verified`、`gpu_verified=true`、`checkpoint_recovery_verified=true`。Actor 与 renderer 均完成真实样本 preflight，权重、优化器、调度器及训练状态恢复一致，RNG 与下一次随机抽样一致，preflight 后恢复状态保持不变。外盘证据保存在 `reports/final-native-gpu-restore/deployment/env-restored-gpu-check.json`、同目录退出码文件，以及 `reports/final-native-gpu-restore/runs/env-restored-gpu-check-retry3/recovered-preflight/checkpoint_restore.json`。它验证的是独立解包环境和最终 joint 检查点在原实例上的恢复，不能写成“新机器全套离线部署已经验证”。

native 启动显式优先搜索解包 prefix 的 `lib`、`lib/python*/site-packages/nvidia/*/lib` 和 `lib/python*/site-packages/torch/lib`；遗漏后两项曾导致 `cublasLtCreate` 初始化失败。验证器核对实际 Python executable、prefix、包文件以及运行前后已导入模块的物理路径、设备号、inode 和文件长度，并要求它们属于环境 manifest 的归档 roots。项目 `src`、验证器自身及本次专用缓存中生成的 JIT 扩展单独记录；其它宿主 Python 模块不能混入。不能用仅设置 `PYTHONHOME` 或一次 `import torch` 代替这些来源和恢复检查。

## 新 Linux 主机的具体恢复命令

以下命令是恢复流程说明，未在新主机完整执行。使用 Bash、宿主 Python 3.12 或更新版本、GNU tar，目标文件系统应支持 Linux 权限与软链接。只修改 `RENTAL_BACKUP` 为外盘实际挂载位置；项目和环境路径保持下面的原路径。需要有写入 `/workspace` 的权限，目标项目目录与环境根目录必须尚不存在。不要在仍运行训练的租用实例上重复执行。

```bash
set -euo pipefail
export RENTAL_BACKUP='/mnt/backup/dag-pilot-rental-backup-20261027T130933Z-9769b4'
export RENTAL_PROJECT='/workspace/agentic-image-pilot-small'
export RENTAL_RESTORE='/workspace/env-restore-check'
export RENTAL_ENVROOT="$RENTAL_RESTORE/root"
test ! -e "$RENTAL_PROJECT"
test ! -L "$RENTAL_PROJECT"
test ! -e "$RENTAL_ENVROOT"
mkdir -p "$RENTAL_RESTORE"
python3 "$RENTAL_BACKUP/tools/restore_rental_bundle.py" \
  --backup-root "$RENTAL_BACKUP" --plan "$RENTAL_RESTORE/project-plan.json"
python3 "$RENTAL_BACKUP/tools/restore_rental_bundle.py" \
  --plan "$RENTAL_RESTORE/project-plan.json" --destination "$RENTAL_PROJECT" --execute
```

项目恢复会包含该计划选择的真实任务数据、训练记录、检查点及最新代码。应先确保索引已经完成封存；在仍有 `known_incomplete` 时恢复，只能得到当前已有部分。不要启动归档中的历史队列。

### 复制精确公开模型

下面按公开模型 manifest 逐个复制，并在复制时验证每个文件的原生 SHA-256 或 Git blob SHA-1，不额外重复扫描模型。失败时保留新目标中的已复制文件，不能把失败目录视为完整模型。

```bash
python3 - <<'PYCODE'
import hashlib, json, os
from pathlib import Path
source = Path(os.environ['RENTAL_BACKUP']) / 'public-model-snapshots'
target = Path(os.environ['RENTAL_PROJECT']) / 'models'
manifest = json.loads((source / 'manifest.json').read_text())
assert json.loads((source / 'status.json').read_text())['status'] == 'completed'
for role in ('actor', 'renderer', 'judge'):
    assert not (target / role).exists()
for row in manifest['files']:
    relative = Path(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    origin, destination = source / relative, target / relative
    assert origin.is_file() and not origin.is_symlink() and origin.stat().st_size == row['size']
    algorithm = row['hash_algorithm']
    assert algorithm in ('sha256', 'git_blob_sha1')
    digest = hashlib.sha256() if algorithm == 'sha256' else hashlib.sha1()
    if algorithm == 'git_blob_sha1':
        digest.update(('blob ' + str(row['size']) + '\0').encode())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with origin.open('rb') as incoming, destination.open('xb') as outgoing:
        while data := incoming.read(8 * 1024 * 1024):
            outgoing.write(data)
            digest.update(data)
    assert destination.stat().st_size == row['size'] and digest.hexdigest() == row['native_hash'], str(relative)
print('Public model files copied and verified:', len(manifest['files']))
PYCODE
```

### 解压 NF4 与工具模型

这些模型来自独立的 `unique-models-archives`，不在公开模型目录中。以下命令只接受本交付包使用的稳定普通文件/目录，按 generation/chunk 顺序恢复；每个实际解压的包先核对压缩字节 SHA-256。不会覆盖已存在的目标模型。

```bash
python3 - <<'PYCODE'
import hashlib, json, os, subprocess
from pathlib import Path
source = Path(os.environ['RENTAL_BACKUP']) / 'remote/unique-models-archives'
project = Path(os.environ['RENTAL_PROJECT'])
allowed = {'models/actor-4b-nf4', 'models/judge-8b-nf4', 'models/tools'}
for name in allowed:
    assert not (project / name).exists() and not (project / name).is_symlink()
manifests = sorted(source.glob('[0-9]*.manifest.json'))
assert manifests
for path in manifests:
    record = json.loads(path.read_text())
    assert record['group'] in allowed
    for member in record['members']:
        name = Path(member['path'])
        assert member['stable'] and member['tar_type'] in ('0', '5')
        assert not name.is_absolute() and '..' not in name.parts
        assert any(name == Path(root) or Path(root) in name.parents for root in allowed)
    archive = source / record['archive']
    assert archive.parent == source and archive.stat().st_size == record['archive_bytes']
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
    subprocess.run(['tar', '--extract', '--gzip', '--file', str(archive), '--directory', str(project),
        '--no-same-owner', '--keep-old-files', '--no-overwrite-dir'], check=True)
print('Unique model archives restored:', len(manifests))
PYCODE
```

### 恢复独立环境与已交付启动器

启动器包包含 `proot-runtime/usr/bin/proot` 与 `proot-runtime/usr/lib/x86_64-linux-gnu/libtalloc.so.2.3.3`，包括 `libtalloc.so.2` 相对链接。解压位置必须对应 `/workspace/env-restore-check/proot-runtime`；无需在线安装 proot。普通 `ldd proot` 可能报告找不到 libtalloc，下面和验证脚本都通过显式 loader/library path 使用随包提供的库。

```bash
python3 - <<'PYCODE'
import hashlib, json, os, subprocess
from pathlib import Path
source = Path(os.environ['RENTAL_BACKUP']) / 'remote/environments'
record = json.loads((source / 'proot-runtime.manifest.json').read_text())
archive = source / 'proot-runtime.tar.gz'
target = Path(os.environ['RENTAL_RESTORE']) / 'proot-runtime'
assert record['status'] == 'completed' and archive.stat().st_size == record['bytes']
assert not target.exists() and not target.is_symlink()
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
subprocess.run(['tar', '--extract', '--gzip', '--file', str(archive), '--directory', str(target.parent),
    '--no-same-owner', '--keep-old-files'], check=True)
PYCODE
export RENTAL_PROOT="$RENTAL_RESTORE/proot-runtime/usr/bin/proot"
/lib64/ld-linux-x86-64.so.2 \
  --library-path "$RENTAL_RESTORE/proot-runtime/usr/lib/x86_64-linux-gnu" \
  "$RENTAL_PROOT" --version
python3 "$RENTAL_PROJECT/scripts/check_isolated_rental_restore.py" \
  "$RENTAL_BACKUP/remote/environments/linux-environments" \
  --stage all --root "$RENTAL_ENVROOT" --proot "$RENTAL_PROOT" \
  --mib-per-second 16 --report "$RENTAL_RESTORE/environment-cpu-check.json"
mkdir -p "$RENTAL_PROJECT/delivery/linux-environments"
cp "$RENTAL_BACKUP/remote/environments/linux-environments/manifest.json" \
  "$RENTAL_PROJECT/delivery/linux-environments/manifest.json"
```

`--stage all` 在新环境根目录内一次流式解压并校验环境包，然后执行 CPU 导入、张量和 OCR 检查。它会保留同级 `extraction_completed.json` 和根目录标记，后续 GPU 验证依赖这两项，不能移动后只复制根目录。它的 CPU 检查使用归档内的旧 support 源码核对来源；正式 GPU 检查通过 `PYTHONPATH` 显式读取最新恢复项目的 `src`，不复制旧 support 源码覆盖项目。

proot 包仍依赖宿主 `/lib64/ld-linux-x86-64.so.2`、`libc.so.6`、Linux 内核与 ptrace 支持。Python 扩展另需宿主 glibc/系统 ELF 库。GPU 验证还需两个可见 NVIDIA GPU、驱动及设备节点；交付环境的主 PyTorch 为 CUDA 13.2 构建，原实例驱动记录为 610.57.04、算力 8.9。这些是运行记录，不是任意新机器兼容性承诺。环境包与 proot 包均不包含内核驱动。

### 放回最终恢复包并验证 joint 检查点

最终包的导出 manifest 中 `source` 记录了原始归档路径。下面复制到原项目的 `delivery/priority-checkpoints`，保留 manifest 原件；不能用传输回执代替 `rl_recovery_export_v1` 导出 manifest，也不能把尚未完成的 joint 包标成已恢复。

```bash
python3 - <<'PYCODE'
import hashlib, json, os
from pathlib import Path
backup, project = Path(os.environ['RENTAL_BACKUP']), Path(os.environ['RENTAL_PROJECT'])
index = json.loads((backup / 'DELIVERY_INDEX.json').read_text())
assert index['final_checkpoints'] and all(row['present'] and row['verified_from_receipt'] for row in index['final_checkpoints'])
target = project / 'delivery/priority-checkpoints'
target.mkdir(parents=True, exist_ok=True)
for row in index['final_checkpoints']:
    source = backup / row['archive']
    destination = target / source.name
    assert source.stat().st_size == row['bytes'] and not destination.exists()
    digest = hashlib.sha256()
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        while data := incoming.read(8 * 1024 * 1024):
            outgoing.write(data)
            digest.update(data)
    assert digest.hexdigest() == row['sha256']
print('Priority recovery archives copied:', len(index['final_checkpoints']))
PYCODE
python3 "$RENTAL_PROJECT/scripts/check_restored_environment_gpu.py" \
  --project "$RENTAL_PROJECT" --root "$RENTAL_ENVROOT" --execution native \
  --components actor renderer \
  --recovery-archive "$RENTAL_PROJECT/delivery/priority-checkpoints/full-joint-rl32-complete.tar.gz" \
  --recovery-manifest "$RENTAL_BACKUP/reports/full-joint-rl32-export.received.json" \
  --output runs/env-restored-gpu-check-new-host \
  --report deployment/env-restored-gpu-check-new-host.json --execute
```

确认两个 GPU 均空闲后才执行最后这条命令。它不会推进原训练阶段，而是在新输出中的 `recovered-checkpoint` 解压最终 joint 包，用正式接口恢复状态后执行有限真实样本检查。`committed_restore_config.json`、原训练检查点及 frozen references 留在恢复树中。SFT 和 Agent 的完整检查点也来自项目增量和各自优先恢复包；joint 验证通过不能替代“每一个历史检查点都独立加载验证”的声明。

### native 主 Python 与 CPU 工具环境的启动入口

GPU 恢复以此前的 `--execution native` 验证命令为准；不要再用 proot 包装该 GPU 命令。原配置的 actor、judge 为 `cuda:1`，renderer、encoders 为 `cuda:0`，验证器保留 `CUDA_VISIBLE_DEVICES=0,1`，不将双卡状态改成单卡。可见 GPU 序号通过 UUID 映射至实际设备节点 minor，不能假设第 1 号 GPU 对应 `/dev/nvidia1`；原租用实例实际节点为 `/dev/nvidia0` 和 `/dev/nvidia4`。

以下 native 函数只用于启动主环境 Python。它复用验证脚本中的动态库路径配置，隔离缓存并显式指定项目源码，但普通 CLI 调用不会自动产生完整来源审计或检查点恢复证明；这些证明仍来自上面的验证器。示例只显示帮助，不启动训练。后续运行必须选择新的输出目录并使用真实阶段配置，不能恢复历史 worker 状态。

```bash
rental_python() {
  python3 - "$@" <<'PYCODE'
import importlib.util, os, sys
from pathlib import Path
project = Path(os.environ['RENTAL_PROJECT']).resolve()
root = Path(os.environ['RENTAL_ENVROOT']).resolve()
session = Path(os.environ['RENTAL_RESTORE']).resolve() / 'native-session'
spec = importlib.util.spec_from_file_location('restored_gpu_launcher', project / 'scripts/check_restored_environment_gpu.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
environment = {
    'PYTHONPATH': str(project / 'src'), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONFAULTHANDLER': '1', 'PYTHONUNBUFFERED': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
    'CUDA_VISIBLE_DEVICES': '0,1', 'NVIDIA_VISIBLE_DEVICES': '0,1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
    'LD_LIBRARY_PATH': '/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu',
}
for key in ('HOME', 'TMPDIR', 'HF_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH'):
    path = session / 'cache' / key.lower()
    path.mkdir(parents=True, exist_ok=True)
    environment[key] = str(path)
environment = module.native_environment(root, environment)
executable = root / module.PYTHON.lstrip('/')
assert executable.is_file()
os.chdir(project)
os.execve(str(executable), [str(executable), *sys.argv[1:]], environment)
PYCODE
}
rental_python -m agentic_image.cli --help
```

主 Python 直接来自 `/workspace/env-restore-check/root/usr/local/miniconda3/envs/py312/bin/python`。宿主 `python3` 只执行上述不导入模型库的启动配置，随后由 `execve` 替换为独立环境 Python。此函数消费标准输入来装载启动配置，不能用 `rental_python -` 传入另一个 stdin 脚本；使用 `-c`、`-m` 或脚本文件。

工具环境 `.venv-tools` 保留 CPU proot 路线。下面隐藏 CUDA，只检查工具库导入，不是 GPU 工具服务验证，也未在新机器上独立完整验证。proot 是路径映射工具，绑定可写，不是安全沙箱。

```bash
mkdir -p "$RENTAL_RESTORE/tools-session/tmp" "$RENTAL_RESTORE/tools-session/home"
rental_tools_cpu() {
  local -a host_binds=()
  local host_path
  for host_path in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /lib64 /proc /sys /dev /bin /usr/bin; do
    if [ -e "$host_path" ]; then host_binds+=(-b "$host_path:$host_path"); fi
  done
  env -i HOME="$RENTAL_RESTORE/tools-session/home" TMPDIR="$RENTAL_RESTORE/tools-session/tmp" \
    PATH="$RENTAL_PROJECT/.venv-tools/bin:/usr/bin:/bin" \
    PYTHONPATH="$RENTAL_PROJECT/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    LD_LIBRARY_PATH='/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu' \
    /lib64/ld-linux-x86-64.so.2 \
    --library-path "$RENTAL_RESTORE/proot-runtime/usr/lib/x86_64-linux-gnu" "$RENTAL_PROOT" \
    -r "$RENTAL_ENVROOT" -w "$RENTAL_PROJECT" "${host_binds[@]}" \
    -b "$RENTAL_PROJECT:$RENTAL_PROJECT" \
    -b "$RENTAL_ENVROOT$RENTAL_PROJECT/.venv-tools:$RENTAL_PROJECT/.venv-tools" \
    -b "$RENTAL_ENVROOT$RENTAL_PROJECT/.runtime:$RENTAL_PROJECT/.runtime" \
    -b "$RENTAL_ENVROOT$RENTAL_PROJECT/third_party:$RENTAL_PROJECT/third_party" \
    -b "$RENTAL_RESTORE/tools-session:$RENTAL_RESTORE/tools-session" \
    "$RENTAL_PROJECT/.venv-tools/bin/python" "$@"
}
rental_tools_cpu -c 'import torch, groundingdino, sam2; print(torch.__version__); assert not torch.cuda.is_available()'
```

完整续训应走项目的恢复入口，不能用仅加载 adapter 文件替代优化器、RNG 和游标恢复。当前交付冻结不授权启动任何新的训练或可选评测。

## 完整交付的验收条件

1. 索引能找到唯一推荐代码、全部基础模型与环境、最终 SFT/Agent/joint 权重、实际配置和结果报告；每项状态清楚。
2. 最终训练检查点有真实梯度、参数变化和完整提交证据；固定评测按用户最后冻结范围交付 637 条，剩余 643 条明确列为缺少完成记录，不把完整 1280 条原计划标成已完成。不能用下载或进程记录代替训练证据。
3. 最终增量范围无未解释的读取错误、未完成文件或未稳定成员，全部必需归档已落盘并有匹配的流式校验收据。
4. 独立恢复检查实际成功加载两侧权重、优化器、调度器、RNG 和游标，并完成规定的真实 GPU 样本检查；宿主依赖与未验证范围保留在报告中。
5. 在新目录生成的恢复计划无错误，恢复入口和路径说明可用；没有实际执行的新机器完整离线部署仍标为未验证。

封存文件应包括 `manifests/transfer-receipts/project-results`、`manifests/project-results/status.json`、各轮 `manifests/project-results/passes/*.pass.json`、实际训练/评测完成报告及 GPU 恢复检查报告。任何缺项仍列入索引的未完成项。

宿主驱动与编译工具链没有作为独立可启动操作系统归档；它们在新机器上的兼容性仍需验证。原实例 retry3 成功证据和此前失败记录都应保留，不能用新报告覆盖历史失败证据。
# 2026-10-28 最后交付说明

最后 152 个归档从电脑内置盘通过系统原生复制，直接放入 `remote/project-results-archives`，保留原始文件字节，没有再次压缩或打大包。新增传输按用户要求不做内容哈希扫描；系统复制成功退出及文件长度记入 `reports/final-transfer-records.json`，旧的历史校验结果保留。内置盘原件保留。

恢复前，先在备份根目录执行 `tar -xzf manifests/project-results/final-project-metadata.tar.gz`，展开服务器已经打包好的最终目录、逐包成员清单和各次归档记录，再按本文恢复项目、基础模型、环境和检查点。最新完整代码包直接保存在 `code-revisions/pilot-small-code-20261028-final.tar.gz`。

`remote/final-archives-bundle.tar.part` 是早先已停止方案留下的未完成临时文件，不参与恢复。完成状态以顶层 `DELIVERY_INDEX.json` 及 `FINAL_HANDOVER_README.txt` 为准。
