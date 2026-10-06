# 2026-10-06 代码审查修复

本次修复保存在 `DIWA-paper/code/DIWA`，交付压缩包为 `DIWA-paper/output/DIWA_code.zip`。论文正文、作者提供的实测数字和上层原工程未改写。以下新增实现选择不代表对历史实验配置的重新确认。

## 1. 原生 LIBERO 转换与外部监督

官方轨迹已有 `rewards` / `dones` 时，转换器保留这些原生字段，不再将它们视为不完整的 DIWA 标注。只有出现 `progress` 或 `candidate_*` 字段时才要求六项嵌入式 DIWA 标注齐全。外部 measured sidecar 仍优先提供正式监督；转换器不会补造 progress 或候选回报。

测试使用临时 HDF5 完成实际转换，再由原有监督读取方法加载外部 sidecar；真正缺少 DIWA 字段的输入仍被拒绝。

## 2. CoTracker 未来标签隔离

提取器在时刻 t 保存的位移使用 t 与 t+frame_gap 的图像，属于未来监督。DreamVLA 的在线对象编码器不再接收这些位移或未来可见性；它们仅供训练时的停止梯度 EMA 目标编码器使用。推理完全沿用在线图像/SAM 编码路径。新轨迹文件记录 `temporal_role=future_supervision`、frame_gap、schema_version 和 future_valid；旧文件同样只用于目标分支。

回归测试改变未来轨迹，要求训练与推理的动作、提案、选择索引、影响分数、在线对象槽与动作特征保持完全相同。训练时允许未来预测目标及其损失变化。测试执行真实 DIWA 前向方法与核心，绕过需要下载权重的视觉/语言编码器。

## 3. 缺少候选回报的离线数据

DROID/OXE 的离线导出保留 `candidate_q_values=NaN`；没有实测进度时 progress 也明确为 NaN。共同 runner 使用 `offline_imitation` 模式，关闭 critic、候选 Q 回归、progress、regret 和 contrastive，并从 influence 目标中移除 Q/progress 响应。动作学习、proposal、future、policy influence、budget 和 mask 校准仍可训练。

报告列出 `optimized_auxiliary_losses` 与 `disabled_auxiliary_losses`。不会将缺失 Q 填为零，也不会用待训练 critic 的预测替代标签。每条轨迹分别保存语言与任务身份，避免对不同任务进行同任务交换。CALVIN/RoboCasa/RoboTwin 无法通过关闭导出标记绕过 measured Q 验证；正式 LIBERO 与特征训练始终保持严格监督。

## 4. 每个 rank 的跨轨迹配对

LIBERO pretrain/finetune 的 DIWA 分支使用 `DistributedTaskPairBatchSampler`。批量必须是至少 2 的偶数，同一任务至少有两条不同轨迹。每个 anchor 在本 rank 的批次中伴随一条同任务、不同 episode 的窗口；不同 rank 的完整批次数相等。epoch 与 seed 控制采样顺序，`DataInfo.set_epoch` 会更新新采样器。

episode 身份包含数据根路径，多个数据集中的相同数字 ID 不会被视为同一轨迹。严格核心会拒绝缺少有效配对的监督批次，避免 regret / swap 静默失效。各 rank 的配对与批次平衡通过 CPU 测试模拟验证；本次未运行真实多 GPU 训练。

每个 epoch 访问所有窗口作为 anchor，并额外采样一个 partner，末尾不足时补齐 anchor。因此 global 样本数包含 partner 与补齐样本，约为原窗口数的两倍。学习率调度与累积组使用实际 `len(loader)`；这些采样语义应记录在重新训练的实验配置中。

## 5. mask token 的独立训练目标

干预响应继续停止梯度，只训练 influence scorer。新增独立 `mask` 项：对有效监督状态的完整 world tokens 求停止梯度的全局均值 μ，最小化 `smooth_l1(mask_token, μ)`。无效填充状态不参与均值；DDP 汇总总和与计数。此项仅对 mask token 产生梯度，既不训练世界目标，也不反向改变 influence 目标。

默认权重为 0.01，通过 `loss_weights.mask` / `--diwa_loss_mask` 配置，从训练开始生效。该校准目标和权重是本次为解决“可学习 mask 实际没有梯度”而补充的实现选择，论文未给出这一具体校准项，不能声称历史实测训练采用了它。若要严格复现历史训练，应使用作者确认的 mask 学习目标和配置。回归测试同时要求 mask 更新及原有干预梯度隔离成立。

## 断点与验证范围

完整 DreamVLA checkpoint schema 升为 4；特征 checkpoint format 升为 3，并记录 method_revision。旧特征断点可用于结构兼容的推理，但不能直接续训；完整训练的旧断点默认拒绝，已有显式 legacy 入口仅供独立核验后的迁移。新版本的 epoch 续训与连续训练仍需完全一致。

运行 `python -m pytest -q tests` 与 `python diwa_cli.py smoke --output runs/review_smoke` 可检查本次修复。具体通过数量、真实执行环境、源码哈希与合成功能测试损失见 `verification/`。本次代码检查不等同于完整 GPU 训练、仿真成功率复现或物理机器人验证。
