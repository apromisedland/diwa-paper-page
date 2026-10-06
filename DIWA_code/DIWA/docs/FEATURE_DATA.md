# 预计算特征数据接口

此接口将 VLA 编码器与 DIWA 训练分开。输入特征必须来自真实观测，且时间 `t` 的特征只能编码 `<= t` 的图像、语言和本体状态，不能编码未来帧、动作标签、任务成功标签或候选回报。数据检查可以检查形状与有限数值，不能证明特征无泄漏或标签的采集来源。

## Episode 文件

每个 NPZ 对应一条完整或截断的 episode；不允许把一次 reset 后的状态拼接在同一文件中。所有数组共享观察时间索引。设 episode 长度为 `T`，隐藏宽度 `D`，动作维度 `A`，动作长度 `L`，候选数 `M`：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `context_tokens` | `[T,C,D]` | 因果上下文，可包括语言、本体状态和当前全局视觉特征 |
| `visual_tokens` | `[T,V,D]` | 当前视觉 token，供对象分槽；与 context 使用相同特征宽度 |
| `actions` | `[T,L,A]` | 示教动作 chunk，按策略坐标归一化 |
| `reward` | `[T]` | 执行该时刻记录动作后测得的奖励 |
| `done` | `[T]` | 终止标记，取 0/1；仅最后一个时刻可以为 1 |
| `progress` | `[T]` | 测得的任务进度，范围 `[0,1]` |
| `candidate_actions` | `[T,M,L,A]` | 各状态下按共同规则生成并执行的候选 chunk |
| `candidate_q_values` | `[T,M]` | 对应候选动作分支测得的回报 |
| `candidate_rule_ids` | `[M]` 字符串 | 稳定的候选规则名称，与 manifest 顺序完全一致 |
| `task_id` | 标量字符串/整数 | 任务身份；同一任务使用相同 ID |
| `episode_id` | 标量字符串/整数 | 轨迹身份；数据集内必须唯一 |

浮点数据转换为 FP32；`done` 转换为 bool。连续动作范围为 `[-1,1]`，夹爪范围为 `[0,1]`。LIBERO 原生夹爪 `-1/+1` 应在导出时转换为 `(g+1)/2`，示教和候选动作都必须转换。

默认 `L=3, A=7, M=6, D=1024`。每条轨迹至少有 `sequence_length` 个受监督观测（参考配置为 7）；窗口不会跨越 episode。终点对齐窗口如果缺少未来 lookahead，读取器只为组成固定形状张量而重复最后一个观测，并在 `observation_valid` 中将这些位置标为无效；未来预测损失会忽略它们，而终点的 reward、done 和 progress 仍保留监督。每个 batch 的锚点配同任务的另一条 episode，因此每个任务至少需要两条有效轨迹。不能按时间索引伪造 progress，也不能用 critic 输出填充缺失回报；只包含已执行动作的离线轨迹通常不足以提供全部候选监督。

候选回报必须对应其状态、动作、discount、执行长度及 shaping 规则。LIBERO 默认六个规则为示教、保持夹爪的零连续动作、`+x/-x/+y/-y` 扰动。跨平台时允许更改动作维度和规则，但同一次配对训练中的索引语义、动作归一化和回报定义须一致。

## 数据清单

将下面的结构保存为 `manifest.json`，填写自己实际使用的编码器与回报定义；示例路径和描述不代表实验记录。

```json
{
  "format_version": 1,
  "data_kind": "measured",
  "feature_encoder": "填入编码器名称、checkpoint 哈希和导出版本",
  "feature_causality": "past_and_current_only",
  "action_normalization": "填入坐标系、连续动作缩放和夹爪映射；输出须为 [-1,1] / [0,1]",
  "return_definition": "填入真实的 discount、分支执行长度、reward、progress 与 shaping 定义",
  "candidate_rules": ["demonstration", "no_motion_keep_gripper", "plus_x", "minus_x", "plus_y", "minus_y"],
  "episodes": ["task0_episode0.npz", "task0_episode1.npz"]
}
```

使用 `numpy.savez_compressed(...)` 写入上述数组；不使用 pickle 对象数组。每个 episode 的 `candidate_rule_ids` 须为与清单相同的六个字符串。未知或不可获取的标签应保持缺失并先完成采集，不能以零值替代。

NPZ 是便于交换的读取方式，会在每次加载窗口时解压；大型训练可在保持相同 batch 接口的情况下替换为分片读取器。当前特征入口不加载 SAM / CoTracker 额外输入；完整 DreamVLA 路径中，预计算的未来 CoTracker 轨迹只用于 EMA 监督目标，不进入在线策略。

## 推理输入

推理文件只需要 `context_tokens: [S,C,D]` 和 `visual_tokens: [S,V,D]`，包含到当前时刻为止的观测历史。无需未来帧或任何监督字段。输出 `actions: [1,L,A]` 属于历史最后一个观测时刻。

`selected_indices` 为展平对象–时间网格的索引：`future_offset = index // num_slots + 1`，`slot = index % num_slots`。`selected_valid_mask` 标记每行实际有效的选择；批处理时世界解码器按 batch 内最大 K 填充，不能把填充位置计入有效 token 数。

## 断点与数据身份

checkpoint 记录清单、episode 文件内容以及窗口语义版本的 SHA-256 指纹。续训会检查配置、数据和窗口构造规则是否变化。该指纹用于识别运行输入，不是数据真实性证明。随机种子和训练配置随 checkpoint 保存，自动测试验证了本机 CPU 环境下连续训练与按 epoch 续训的一致性。
