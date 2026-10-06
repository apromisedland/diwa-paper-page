# DIWA — Decision-Influential World Abstraction for VLA-WAM Policies

根据当前论文整理的 PyTorch 参考实现，复用现有项目的 DreamVLA 集成代码，补充独立训练与推理接口、配置、严格数据检查和自动测试。

**本包包含源码与运行方法，不包含论文实验的训练权重、原始轨迹或机器人运行环境。** 论文中的实测数值保持不变；`smoke` 使用单独标记的随机张量检查代码，不产生或替代论文实验结果。

## 1. 代码入口

| 内容 | 文件 |
| --- | --- |
| 对象槽、Slot Attention、时间对齐 | `models/diwa/object_tokens.py` |
| 动作提案、影响打分、Top-K、自适应预算、稀疏世界解码 | `models/diwa/core.py` |
| 共享噪声/时间的干预监督 | `models/diwa/influence.py` |
| Twin-Q、进度估计、EMA 目标网络 | `models/diwa/critic.py` |
| 测得候选回报的 regret 几何与对比损失 | `DIWACore._regret_losses` |
| 扩散 / flow 动作头 | `models/action_model/action_model.py` |
| 基于预计算特征的训练和推理 | `diwa_cli.py`、`models/diwa/feature_policy.py` |
| 图像、语言、本体感觉的完整集成 | `models/dreamvla_model.py`、`train.py`、`eval_libero.py` |
| LIBERO 候选动作分支回报采集 | `data_process/collect_libero_diwa_supervision.py` |
| 监督数据打包 / 方法诊断指标 | `data_process/build_diwa_supervision.py`、`data_process/evaluate_diwa_metrics.py` |

论文对应关系见 [docs/PAPER_TO_CODE.md](docs/PAPER_TO_CODE.md)。本次 5 项审查问题的修复、回归测试和历史复现边界见 [docs/REVIEW_FIXES.md](docs/REVIEW_FIXES.md)。

## 2. 先运行本机示例

在解压后的 `DIWA` 目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python diwa_cli.py smoke --output runs/quickstart
```

这个命令在 CPU 上运行小型模型，执行训练、保存、重新加载和稀疏推理。它保留论文的 **16 个对象槽 × 3 个未来时刻 = 48 个候选**，推理最多解码 12 个；隐藏宽度和动作头已缩小，训练阶段也被加速，便于检查所有损失。它不是论文规模的训练或机器人评测。

输出文件：

```text
runs/quickstart/
  fixture/                  明确标注的合成测试输入
  training/last.pt           示例模型与优化器状态
  training/run.json          配置、版本、种子和数据指纹
  training/metrics.jsonl     各项损失与保留数量
  prediction.npz            动作、影响分数和选择索引
  smoke_report.json          功能检查报告
```

已有输出目录中的模型不会被无提示覆盖。重新跑示例时使用新的输出目录。

本次实际验证环境为 Python 3.12.9、PyTorch 2.11.0、NumPy 2.2.5、timm 1.0.12，macOS CPU。精确版本记录在 `requirements-cpu-tested.txt`；常规依赖范围见 `requirements.txt`。完整图像训练建议另建下文的环境。

## 3. 用你的数据训练 DIWA 模块

如果已有 VLA 编码器，可先导出真实观测的上下文与视觉特征，按 [docs/FEATURE_DATA.md](docs/FEATURE_DATA.md) 准备 episode NPZ 和清单。这个入口训练 DIWA 和动作头，**预计算的编码器特征固定不变**；端到端 DreamVLA 实验使用第 4 节。

```bash
python diwa_cli.py validate-data \
  --manifest /path/to/features/manifest.json \
  --config configs/diwa_paper.json

python diwa_cli.py train \
  --manifest /path/to/features/manifest.json \
  --config configs/diwa_paper.json \
  --output runs/diwa_seed42 \
  --device cuda --seed 42
```

当前论文的参考配置为隐藏宽度 1024、16 个注意力头、两层世界解码器、两层策略融合、动作长度 3、48 个候选、预算范围 3–12、6 个候选动作规则。三个训练种子可分别使用 `--seed 42`、`43`、`44` 和不同输出目录。

特征入口是单设备训练：batch 16、累积 4 时有效批量为 64。论文完整启动脚本默认 8 个设备，有效批量 512。二者不能当作相同实验设置。配置中的 `window_stride` 是新增特征读取器的窗口步长，不代表历史实验记录。

断点按完整 epoch 保存，并恢复优化器、学习率调度、随机状态、EMA 网络和影响监督的运行尺度：

```bash
python diwa_cli.py train \
  --manifest /path/to/features/manifest.json \
  --resume runs/diwa_seed42/last.pt \
  --output runs/diwa_seed42 --device cuda
```

恢复时使用 checkpoint 中的原配置，并检查数据指纹。当前特征 checkpoint format 为 3；format 1/2 可用于结构兼容的推理，但不能直接续训。这个入口不会把缺失候选回报替换为模型预测，也不允许用示例合成数据启动正式训练。

推理只需要已观测的 `context_tokens` 和 `visual_tokens`：

```bash
python diwa_cli.py predict \
  --checkpoint runs/diwa_seed42/last.pt \
  --input /path/to/observed_features.npz \
  --output runs/diwa_seed42/actions.npz --device cuda --seed 66
```

输出的是最新观测时刻的动作 chunk：连续坐标限制在 `[-1,1]`，夹爪坐标限制在 `[0,1]`。机器人适配器负责反归一化和夹爪协议转换。该命令只保存动作，不连接或驱动物理机器人。

## 4. 完整 DreamVLA + LIBERO 流程

完整集成保留项目原有的数据、视觉编码器和仿真接口。按 [docs/LIBERO_INSTALL.md](docs/LIBERO_INSTALL.md) 准备 Python 3.10 / CUDA / LIBERO 环境；项目原依赖保存在 `requirements-upstream.txt`，补充依赖见 `requirements-full.txt`。这套 GPU/仿真环境本次没有运行验证。

```bash
python -m pip install -r requirements-full.txt
```

按 [docs/LIBERO_RUN.md](docs/LIBERO_RUN.md) 转换官方轨迹、准备视觉特征和预训练权重。测得候选监督的入口：

```bash
python data_process/collect_libero_diwa_supervision.py \
  --libero-path /path/to/LIBERO --suite libero_10 \
  --dataset-dir /path/to/libero_10 --output /path/to/measured_libero_10

python data_process/build_diwa_supervision.py \
  --source /path/to/measured_libero_10 --output /path/to/diwa_supervision
```

训练：

```bash
SAVE_CHECKPOINT_PATH=/path/to/checkpoints \
ROOT_DIR=/path/to/converted_data_parent \
LIBERO_DATASET_NAME=libero_10_converted \
VIT_CHECKPOINT_PATH=/path/to/mae_pretrain_vit_base.pth \
PRETRAINED_CHECKPOINT=/path/to/dreamvla_pretrained.pth \
DIWA_SUPERVISION_PATH=/path/to/diwa_supervision \
NUM_GPUS=8 SEED=42 \
bash scripts/LIBERO/DIWA/train_latent_diwa.sh
```

评估：

```bash
CHECKPOINT=/path/to/diwa_checkpoint.pth \
VIT_CHECKPOINT_PATH=/path/to/mae_pretrain_vit_base.pth \
LIBERO_PATH=/path/to/LIBERO NUM_GPUS=1 DIWA_PROFILE=1 \
DIWA_PROFILE_WARMUP_STEPS=5 \
DIWA_PROFILE_OUTPUT=/path/to/diwa_profile.json \
bash scripts/LIBERO/DIWA/eval_latent_diwa.sh
```

性能统计会在每个 GPU 分别剔除预热更新，再由 rank 0 汇总并写入严格 JSON。完整 DreamVLA checkpoint 会核对全部可训练参数、DIWA 状态、张量形状、建模参数、优化与分阶段损失配置，以及未写入 checkpoint 的冻结底座参数 SHA-256 签名；同时按 rank 保存并恢复 Python、NumPy、Torch CPU/CUDA 随机状态，并采用原子替换写盘。迁移学习 checkpoint 必须覆盖所有未显式重置的非 DIWA 可训练参数。当前完整 checkpoint schema 为 4，其中包含终点感知窗口、未来标签隔离、配对采样与 mask 校准的训练语义；缺少运行元数据、随机状态、冻结底座签名或使用旧 schema 的断点默认拒绝续训，可在独立核验后显式使用 `--diwa_allow_legacy_checkpoint`。评估仍可加载结构兼容的旧模型权重。

LIBERO 的 DIWA pretrain/finetune 均按每个设备配对采样同任务、不同轨迹的窗口：batch 必须为至少 2 的偶数，每个任务至少有两条轨迹。每个 epoch 访问所有 anchor 并额外采样 partner，计数包含 partner 与补齐样本，约为原窗口数的两倍；调度使用实际批次数。原生 rewards/dones 可以正常转换，再由外部 sidecar 提供完整 DIWA 监督。预计算的未来 CoTracker 位移仅用于 EMA 目标，不进入策略输入。

新增独立 `mask` 校准项，默认权重 0.01，只训练替换 token，保持干预目标停止梯度。具体目标与权重是本次修复补充的实现选择，论文未说明这一具体项，不能据此断言历史实测训练采用了它。

特征入口的 `last.pt` 与完整 DreamVLA checkpoint 格式不同，不可直接互换。RoboTwin、RoboCasa、CALVIN 和离线数据适配器见 `scripts/MULTI_DATASET/DIWA/README.md`；其 smoke 流程用于接口检查，不等价于论文成功率评测。私有真实机器人数据分支需显式传入 `--real_dataset_adapter module.path:ClassName`，公开包不包含论文实验使用的私有数据读取器。

## 5. 验证与实现边界

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

测试覆盖稀疏解码、批内不同预算、共享噪声、干预目标梯度隔离、候选规则身份、measured-return 数据要求、跨轨迹配对、终点监督保留、未来有效位掩码、实际 DataLoader 批次数、动作维度适配、flow 精确采样步数、预训练覆盖率、checkpoint 完整性、逐 rank 随机状态、原子写盘与冻结底座签名、分布式性能汇总和断点续训。验证结果见 `verification/`。

若要重新得到论文表格，需要对应实验的真实轨迹、候选回报、训练权重和平台配置。在本次交付中完成的是代码层验证，未重新执行论文的 GPU 训练、各基准成功率评测或物理机器人试验。

## 6. 来源与本次整理

代码从已有 DIWA/DreamVLA 项目复制整理。原项目说明保留在 `UPSTREAM_README.md`，第三方来源见 `NOTICE.md`。本次修改保存在本代码包，未改写上层原工程。

主要修正：TD 目标使用下一时刻的动作提案；严格校验候选监督、规则顺序、预算、张量契约和训练配置；终点时刻的 reward/done/progress 监督不再因未来观测不足而丢失，缺少的未来输入只作重复填充并由有效位掩码排除损失；DataLoader 元数据使用真实批次数；flow 采样严格执行请求的积分步数并使用传入噪声及其设备；两个入口共用同一个干预监督实现；checkpoint schema 4 核对训练状态、结构参数、优化配置、逐 rank 随机状态和冻结底座并原子写盘；训练与评估按配置处理动作/状态维度；多 GPU 性能样本统一汇总。评估脚本显式声明自适应预算，原参数解析器的该默认值本来就为开启。
