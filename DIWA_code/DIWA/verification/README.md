# 本次代码验证记录

验证日期：2026-10-06。环境为 macOS CPU；具体软件版本见 `cpu_smoke_report.json` 和 `requirements-cpu-tested.txt`。

- 自动测试：**114 passed**。包含特征训练、稀疏推理、严格监督、共享随机性、梯度隔离、阶段开关、有效目标归一化、终点监督保留、未来有效位掩码、真实 DataLoader 批次数、flow 精确积分步数、候选规则身份、独立候选分支重置、严格 checkpoint 兼容性、逐 rank 随机状态恢复、原子写盘、通用动作维度、轨迹可视化重建、标准 JSON、分布式性能汇总和续训检查。
- 新增 20 项审查回归测试：原生 LIBERO HDF5 转换与外部监督、真正不完整 DIWA 标注拒绝、未来轨迹标签隔离、离线 NaN 候选回报训练、正式模拟器监督不可关闭、每 rank 配对与批次数平衡、LIBERO pretrain/finetune 接入、mask 参数更新与目标梯度隔离、旧特征续训拒绝。
- 完成 16 个小批次、8 次优化更新的 CPU 功能示例；各项损失及影响评分器梯度均为有限值。
- 训练世界解码器展开 48 个候选；示例推理实际展开 12 个。
- mask 校准实际产生非零梯度并改变参数；具体数值记录在 `cpu_smoke_report.json`。该新增目标和 0.01 权重属于实现修复，历史实验是否采用未作断言。
- 模型重新加载后，在相同初始噪声下输出差异为 0。
- 按 epoch 中断、恢复训练的模型参数与连续训练完全一致；测试也覆盖不足一个完整累积组的末尾批次。
- 扩散和 flow 动作头均通过 CPU 推理与共享随机性检查，flow 采样实际调用次数与请求步数完全一致。
- 正式数据入口会拒绝本包的合成测试输入以及缺失、非有限、候选规则身份缺失或顺序不一致等数据。
- 完整 checkpoint schema 4 会拒绝缺失的可训练参数、未知或形状/类型不符的张量、结构/优化/损失日程不一致、旧窗口语义，以及被省略冻结底座的 SHA-256 签名不一致；恢复会按 rank 还原 Python、NumPy、Torch CPU/CUDA 随机状态，写盘采用原子替换；无完整兼容性元数据的旧 checkpoint 必须显式放行。
- LIBERO 候选采集器在每个分支前重置环境计时/终止状态，再恢复同一序列化模拟器状态；测试会在分支状态泄漏时失败。
- 评估入口要求显式 checkpoint；私有真实机器人数据分支要求显式 `module.path:ClassName` 适配器，不再引用公开包中不存在的类。
- 离线指标输出通过标准 JSON 解析；无成功样本时 `tokens_per_success` 为 `null`，标准/OOD 结果分开报告且不再生成 retention 比值。
- 性能统计的纯函数测试覆盖每个 rank 独立剔除预热样本、全局均值/P95、峰值显存和严格 JSON 报告。
- 所有 Python 文件通过语法检查及 Ruff 致命错误检查；独立 DIWA 模块、CLI、轻量工具、转换器、已修改的跨数据集适配器和测试通过 E4/E7/E9/F 检查；大型上游集成文件保留其可选依赖导入结构并通过致命错误检查；LIBERO DIWA 训练、评估脚本通过 shell 语法检查。
- 最终 ZIP 在临时目录中独立解压后，再次通过全部 114 项测试和 CPU 训练、保存、重载、稀疏推理示例；压缩包文件集合与 `source_manifest.json` 完全一致。
- 论文项目保留此前（2026-10-05）使用本地完整 TeX 工程成功编译的记录，本次未改写或重新编译论文：当时编译为 18 页 PDF，日志中没有未定义引用、未定义引文或排版警告。内置单文件编译器无法加载同目录 ICLR 样式文件，因此其失败不作为正文诊断。

图像集成回归测试执行真实 DIWA 前向方法与损失，绕过可选 CLIP/视觉编码器的构造和权重下载；分布式采样测试在 CPU 模拟多个 rank。它们不等同于实际多 GPU 图像训练。

测试过程有一条来自安装环境的 `torch.jit.interface` 弃用提示；未出现测试失败。

本目录中的损失值只来自明确标注的 `synthetic_fixture`，用于代码验证。未重新进行论文的完整 GPU 训练、仿真基准成功率评测或物理机器人试验。验证报告不对论文实测数值作重新计算或替换。

运行命令：

```bash
python -m pytest -q tests
python diwa_cli.py smoke --output runs/cpu_smoke
python diwa_cli.py predict \
  --checkpoint runs/cpu_smoke/training/last.pt \
  --input runs/cpu_smoke/fixture/observations.npz \
  --output runs/cpu_smoke/cli_prediction.npz
```

`source_manifest.json` 记录本次源码文件与所依据手稿的 SHA-256，用于识别交付版本。
