# Cobot Magic 工作总结 — 2026-09-13

## 机械臂 Ready Pose

此前的非奇异 Ready Pose 后来确认是旧 100k 训练集中的中段姿态。虽然其 TCP
坐标曾出现在数据集里，但它对应的任务阶段不同，导致 rollout 起始不稳定。

系统 Ready Pose 已改为旧训练源 `cobot_magic_il_2` 的真实任务起始状态：
episode 31、frame 0。

```text
[0.067, 0.004, -0.345, -0.110, 0.571, -0.128, 0.000]
```

- TCP 位于训练示范的任务起始分布中。
- Piper Jacobian condition number 约为 `74`，远低于运行时 `1000` 的奇异警告阈值。
- 本机源文件：`cobot_bridge/move_init_pose.py`。
- AGX 端 `/home/agilex/lerobot/cobot_bridge/run_lerobot_policy.py` 已同步为同一姿态。
- 本次仅修改配置，未发送初始化或其他机械臂运动命令。

## Debug rollout 分析

最新 debug 记录：

```text
/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/debug_rollouts/latest.raw.h5
```

该记录使用旧 65k BC checkpoint，以 15 FPS 运行 41.67 秒，包含 20 次 policy decision。

结论：

- 放置阶段先执行了示范中常见的连续下降动作。
- 用户确认第一次放置没有完全对上，最后一次下压才使金属工件嵌合。
- 关键视觉判据是中间两个定位孔内是否存在下方圆凸起：未嵌合时孔呈暗色/空孔；嵌合后可见居中的亮金属圆凸起。
- 右相机原始图像中有此线索，但 VLA 实际看到的是 `640×480 → 256×192`、再 padding 到 `256×256` 的图像。夹爪遮挡和金属反光使当前 VLA 能否稳定判断 `seated/not_seated` 尚未被证明。
- 因此，单条 rollout 不能证明最后下压一定由视觉反馈触发，也可能是轨迹先验。要做因果验证，应固定 robot state，仅替换图像为“已嵌合/未嵌合”两种画面，比较输出动作。

右相机视频导出：

```text
/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/debug_rollouts/latest_right_camera_15fps.mp4
```

编码为 H.264、640×480、15 FPS、625 帧、41.67 秒。

## 常驻旁路 Debug Recorder

Dashboard 启动一个独立、低优先级的旁路观察器，实现在
`cobot_bridge/debug_capture.py`。

- 它使用独立 bridge 连接，**绝不发送控制命令**。
- 当 `COBOT_DEBUG_CAPTURE_ENABLED` 启用时（默认启用），它随 Dashboard 启动。
- 默认以 15 FPS 采集 `front`、`right` 相机、机器人反馈和控制状态。
- Dashboard 的 motion job 会标记采集开始/结束。因此即使没有正式 IL/RL recorder，policy rollout 调试也会自动保留最近一段记录。
- 它写入 `debug_rollouts/`，不写入 `raw_datasets/`，不会被误转换进训练数据。
- 只保留最近一条成功关闭的记录：

```text
debug_rollouts/latest.raw.h5
debug_rollouts/latest.json
```

  新记录只有在正常关闭并校验后才会原子替换 `latest`；若新会话过短或无效，旧记录仍保留。
- 原始 H5 含相机 JPEG、双臂反馈、命令/动作字段、时间戳、控制 mode/source 与 phase。
- 对 policy rollout，runner 还会写 JSONL decision sidecar。debug worker 会把它合并为 H5 中的 `/policy_decisions`：其中包括 native/absolute/rollout action chunk、model state、chunk latency、RTC prefix 和基于 bridge sequence 的 raw-frame 对齐。

该旁路是滚动调试产物，不是 episode archive，也不应直接用作训练标签来源。

## 原始数据集发现的问题

原始训练源为 `cobot_magic_il_2_eef`，共 70 条 episode、44,788 帧。检查和重建过程中发现以下问题：

1. **前 4 条 episode 的首次夹取位置异常**

   原始 source episode 0–3 在“第一次夹爪闭合”时的工件/TCP 位置相对其余批次偏离约 80–135 mm。它们与后续主要工作台设置不一致，会把“初始工件在哪里”的视觉—动作对应关系拉宽。

   处理：新数据集排除了 source episode 0–3，仅保留 66 条一致样本。

2. **任务语言文本混用**

   原始 70 条中有两种 task 字符串：

   ```text
   20 条：pick and insert the object
   50 条：pick and place the object
   ```

   虽然实际操作对象/夹具设置相近，但语言输入不同会让 VLA 把同一任务拆成两个条件。 

   处理：新数据集统一为：

   ```text
   Pick up the metal workpiece, align it with the matching fixture, and place it securely.
   ```

3. **首尾静止帧未裁剪**

   旧 66 条训练集包含约 9.9% 的首尾静止帧。开头的静止段会强化“启动后继续保持”的动作先验；结束静止段会增加无信息的 hold action 样本。

   处理：新 v2 数据集从 raw H5 重建，并裁掉 3,671 帧。

4. **录制起点不含初始化过程，且起始关节姿态存在变化**

   66 条 raw 都从 Teach active、非零 joint state 开始；初始化或人工摆位若发生，均发生在 recorder 开始前。因此数据没有“归零/Ready Pose 移动”的监督，也不能从 raw 中证明每条 episode 使用同一个 joint-space 起点。

   处理：系统 Ready Pose 选取真实示范起始状态，而不是全零关节位；训练/rollout 需要保持该起点语义一致。

5. **嵌合成功没有逐帧视觉标签**

   raw episode 的 success 只表示整条任务最终成功，不说明哪一帧已由两个定位孔/圆凸起完成嵌合。末尾画面还可能因腕部相机撤离或夹爪遮挡而不可判定。

   处理：后续 YOLO/状态分类数据必须人工复核放置接触阶段，使用 `seated`、`not_seated`、`unknown/occluded` 标签，不能用 terminal success 自动标全程。

## 数据集静止帧裁剪

此前十字任务已经验证过首尾静止帧裁剪路径，位于
`cobot_bridge/convert_raw_to_lerobot.py`：

```text
--trim-stationary-holds
--trim-joint-tolerance 0.01
--trim-gripper-tolerance 0.002
--trim-settle-seconds 0
```

未裁剪的 66 条数据约有 9.9% 为起始/结束静止帧。旧训练在约 65k steps 时已暂停，确认
`065000` checkpoint 保存后结束；旧 checkpoint 均保留。

新增可复现构建脚本：

```text
scripts/build_cobot_magic_il2_trimmed_fixture_v2.sh
```

脚本会排除原始 episode 0–3、统一 task 文本、直接从 raw H5 裁剪、转换 joint state/action 为绝对 gripper-center EEF，并重算 chunk-30 anchor-relative EEF action 统计。

新数据集：

```text
joint: cobot_magic_il_2_trimmed_unified_metal_fixture_v2_drop_ep0_3
eef:   cobot_magic_il_2_trimmed_unified_metal_fixture_v2_drop_ep0_3_eef
```

已验证的输出：

```text
episodes: 66
frames:   40,417 → 36,746
removed:  3,671 frames (9.1%)
FPS:      30
task:     Pick up the metal workpiece, align it with the matching fixture, and place it securely.
```

保存的 relative-action stats 为有限数值，并使用以下契约：

```text
reference: observation.state[t]
action:    inverse(T_state_t) @ T_action[t+h], h=0..29
frame:     gripper_center
```

录制数据的重要观察：

- 这 66 条 raw 记录均没有包含“全零关节初始化”的运动过程。录制从 Teach active 和非零 joint state 开始；若录制前做过初始化，它发生在 recorder 启动之前。
- 长尾主要发生在 Teach 仍保持 active、但人工已完成动作并停住的阶段；Teach 松开后只多记录约一帧。因此只依靠“Teach release 自动结束 recorder”不能去掉尾部静止帧。

## 新 BC 训练

已从 SmolVLA base checkpoint 启动新的训练。模型架构、batch size、双相机输入、图像增强、relative EEF chunk-30 动作表示和优化器计划均与旧 65k 高成功率模型相同；唯一预期实验变量是裁剪后的数据集。

```text
run:
cobot_magic_smolvla_il2_eef_trimmed_fixture_v2_66ep_relative_eef_chunk30_100k_globalbs32_gpu0_256_photoaug

dataset:
cobot_magic_il_2_trimmed_unified_metal_fixture_v2_drop_ep0_3_eef

zellij session:
cobot_il2_trimmed_v2_train
```

共享训练脚本现在支持覆盖数据集，而不改变其原有默认历史数据集：

```text
scripts/train_cobot_magic_il2_eef_unified_66_bs32_gpu0.sh
  COBOT_DATASET_ROOT
  COBOT_DATASET_REPO_ID
```

## Dashboard / UI 修改

Dashboard 已完成以下布局和运维改进：

- RL/RLT 页面改用浏览器页面滚动，较长内容不会再把上方控制区域压缩进固定高度面板。
- Policy rollout 控制放在更高位置；桌面端 grid 约束列宽，避免 dropdown 和按钮超出视口。
- UI 会显示当前相机 preview path。当前本机 PC Dashboard 使用 PC relay MJPEG；相机数据采集与 policy observation 不依赖该 preview 路径。
- System 页面保留非奇异 Ready Pose 操作，以及 arms/cameras/live services 的设备恢复操作。

System 页的 Arm Feedback 已改为每张卡完整纵向显示：

```text
Age
J1 … J6
Gripper
```

卡片会自动换列，不再截断旧版单行 `q=[…]` 字符串。相关文件：

- `cobot_bridge/web_static/index.html`
- `cobot_bridge/web_static/app.js`
- `cobot_bridge/web_static/styles.css`

`node --check` 已通过，运行中的 Dashboard 已提供带新 cache-bust 版本的 CSS/JS。

## 尚未完成：YOLO 嵌合状态模型

尚未创建持久化的 YOLO labels、detector 或 classifier。

下一步推荐实现右相机 `seated` 状态分类器或 YOLO-pose，并用以下人工复核标签：

```text
seated
not_seated
unknown / occluded
```

不要将 episode terminal success/failure 或单次 gripper 事件直接当作通用逐帧标签：这些时刻右腕相机经常已经撤离或被夹爪遮挡。应人工复核接触阶段图像，加入真实的“第一次未嵌合/最后压入成功”正负对，并按 episode 而非随机 frame 划分 train/validation/test，避免相邻视频帧泄漏。

当前 conda 环境已有 PyTorch 与 OpenCV，但尚未安装 `ultralytics`。
