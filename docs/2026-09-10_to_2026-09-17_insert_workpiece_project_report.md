# 重大专项插工件场景阶段工作报告

**周期：2026-09-10 至 2026-09-17**  
**任务：抓取金属工件，对准目标夹具并完成插入/放置**

## 一、阶段目标与当前结论

本阶段完成了重大专项插工件场景的机器人运行环境、数据采集链路、数据清洗与转换、
SmolVLA BC 模型训练及真机验证。基础 BC 模型已经能够完成抓取、搬运、对位、放置和
撤离的大部分流程，当前真机阶段性成功率约为 **60%**。

当前主要失败模式不是第一阶段抓取，而是最后的低位精密插入：

1. 工件与夹具存在小幅位置或姿态误差，第一次放置没有完全到位；
2. 模型未继续做有效纠正，提前将任务判断为完成并撤离；
3. 针对该问题加入局部人工纠正数据后，模型已能学习到二次对位/下压动作；
4. 但新的 BC 微调模型出现明显退化：容易停留在纠正区域，撤离变慢；部分 rollout
   回到近似初始位置后又重新执行抓取动作，表现出阶段轨迹重复和局部过拟合。

因此，当前结论是：**基础任务已基本跑通，但局部纠正数据的混合方式破坏了原 BC
策略的完整任务阶段分布。下一阶段重点不是继续增加训练步数，而是重新设计局部纠正
样本的裁剪、采样权重和评估方式。**

## 二、机器人与软件环境搭建

### 2.1 真机控制链路

已完成 Cobot Magic 四臂系统的基础控制链路：

- PC 侧运行 Dashboard、数据管理、SmolVLA 推理及训练；
- AGX/机器人侧运行机械臂 bridge、CAN/设备控制和相机采集；
- 支持右 puppet 单臂 policy rollout，以及 master/puppet 同步的人工示教模式；
- 支持设备重启、Ready Pose、IL/RL 录制、Replay、Policy Server 管理和紧急停止；
- 真机低层使用连续关节控制，高层 policy 以 30 FPS action 语义运行；
- EEF 路径使用 Pink IK 将模型 EEF 目标转换为受约束的高频关节轨迹。

### 2.2 Ready Pose 与 EEF 坐标链路

原全零关节位置位于不利于 Cartesian IK 的奇异区域。系统 Ready Pose 已更换为真实任务
示范的起始姿态：

```text
[0.067, 0.004, -0.345, -0.110, 0.571, -0.128, 0.000]
```

该姿态来自原始任务数据的真实起点，既位于训练分布内，也避免全零位姿的 IK 奇异性。

EEF 数据和 rollout 统一采用 gripper-center 位姿，relative EEF action 使用当前观测状态
作为整段 chunk 的 anchor：

```text
reference = observation.state[t]
action[h] = inverse(T_state_t) @ T_target[t+h], h = 0 ... 29
gripper    = absolute command
```

训练续训加载 checkpoint 自己保存的 pre/post processor 和 normalization statistics，避免
微调过程中改变动作坐标系；相关 state/action normalization 契约已通过离线 round-trip
验证。

### 2.3 Dashboard 与调试设施

Dashboard 已完成以下能力：

- IL Collect、RL Collect、RLT、Data、Jobs & Logs、System 页面；
- 三路相机预览、设备状态、四臂 J1–J6 与 gripper 完整反馈；
- Policy rollout checkpoint、GPU、action space、平滑方式和运行速度选择；
- 页面使用浏览器自然滚动，主要控制区已调整到高频操作位置；
- AGX 本机和 PC 监看链路分离，避免预览影响 policy observation；
- Replay 增加停止能力，设备掉线后可从 System 页面恢复。

新增了低开销、只读的旁路 Debug Recorder：

```text
/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/debug_rollouts/latest.raw.h5
/media/fuxin/59b726b5-293a-4235-af97-1e18589c462e/lerobot0_4_0/debug_rollouts/latest.json
```

该 recorder 默认以 15 FPS 保存最近一次完整真机动作，记录相机、机器人 state、实际命令、
policy 原始输出、恢复后的绝对动作、chunk 边界和 observation 对齐关系，但不会发送控制
命令，也不会自动进入正式训练数据集。

## 三、BC 数据采集与数据治理

### 3.1 原始数据检查

原始金属工件任务数据共 70 条。检查后发现：

- 前 4 条示范的工件/首次夹取位置与主体数据偏离约 80–135 mm；
- task text 混有 `pick and insert the object` 与 `pick and place the object`；
- 部分 episode 含较长首尾静止段；
- 相机、state、action 和任务阶段没有逐帧“已完全嵌合”标签；
- 所有录制从实际示教起始姿态开始，不包含从全零位或 Ready Pose 移动到任务起点的过程。

最终排除异常的前 4 条，保留 66 条主场景成功示范，并统一语言为：

```text
Pick up the metal workpiece, align it with the matching fixture, and place it securely.
```

### 3.2 清洗后的主 BC 数据集

构建脚本：

```text
scripts/build_cobot_magic_il2_trimmed_fixture_v2.sh
```

数据集：

```text
cobot_magic_il_2_trimmed_unified_metal_fixture_v2_drop_ep0_3_eef
```

主要统计：

```text
episodes: 66
frames:   40,417 → 36,746
removed:  3,671 frames
fps:      30
cameras:  front + right wrist
state:    absolute gripper-center EEF + gripper
action:   anchor-relative EEF chunk + absolute gripper
chunk:    30
```

首尾静止裁剪仅移除长时间无动作的 hold，不会将相邻 episode 拼接。LeRobot 在构造 action
chunk 时会把索引限制在当前 episode 的 `[start, end-1]` 范围内；episode 末尾不足 30 步
的部分使用末帧 padding，并产生 `action_is_pad` mask。

## 四、基础 BC 模型训练与结果

基础模型使用 SmolVLA、双相机输入、256×256 padding resize、图像颜色增强和 relative EEF
chunk-30 训练。主要训练输出为：

```text
outputs/
cobot_magic_smolvla_il2_eef_trimmed_fixture_v2_66ep_
relative_eef_chunk30_100k_globalbs32_gpu0_256_photoaug/
```

目前真机阶段性结果：

- 抓取金属工件总体稳定；
- 能完成从抓取区到夹具区的长程搬运；
- 多数情况下能进入正确插入区域；
- 完整任务成功率约为 **60%**；
- 主要失败集中在最后数毫米的平移/旋转对位和下压深度；
- 失败时常出现“工件尚未完全嵌合，但策略已撤离”的情况。

右腕相机中确实存在嵌合判据：正确嵌合后，工件中部定位孔内可看到下方圆凸起；未嵌合时
孔内更暗或位置偏离。但该线索尺寸小，且受腕部遮挡、金属反光和 256×256 模型输入影响。
当前 BC 能否稳定使用该视觉线索仍未通过固定 state、只替换图像的因果实验验证。

## 五、局部纠正数据采集与专项微调

### 5.1 采集目的

为解决“第一次放置未到位后直接结束”，专门采集了低位插入失败后的人工纠正轨迹。
这些轨迹覆盖：

```text
未完全对齐/未完全嵌合
→ 人工进行局部平移、旋转和下压
→ 工件完成嵌合
→ 夹爪释放
→ 机械臂撤离
```

### 5.2 数据核对

最初采集 8 条，其中 1 条仅 15 帧、未形成有效纠正过程，已删除；当前保留 7 条有效数据：

```text
raw_datasets/cobot_magic_il-fake/
```

核对结果：

- 7 条轨迹均从局部失败/纠正姿态开始，不是从 Ready Pose 开始；
- 原始轨迹和转换后的 LeRobot episode 均未反转；
- 不存在一条 episode 的结尾与下一条 episode 开头被拼成 action chunk；
- Replay 如果在真机仍处于 Ready Pose 时直接播放局部轨迹，会先追赶局部轨迹首帧，视觉上
  像“Ready→纠正位”，这是 replay 起始状态不匹配，不属于训练监督；
- 局部轨迹中的对位、下压和观察阶段较长，完成后的撤离阶段相对较短。

### 5.3 混合数据集

第一版专项 BC 将 7 条局部轨迹各复制 8 次，与原 66 条主任务数据混合：

```text
dataset:
cobot_magic_il2_base66_localcorrection7x8_chunk10_eef

total episodes: 122 = 66 + 7×8
total frames:   56,402
local frames:   19,656（约 35%）
```

数据集名称中的 `chunk10` 是早期命名遗留；存储层 action 为逐帧绝对 EEF target，训练加载时
根据 policy config 动态构造 chunk。本次实际专项微调使用的是 chunk 30。

### 5.4 专项 BC 配置

专项微调从正确的基础模型 80k checkpoint 开始：

```text
outputs/cobot_magic_smolvla_il2_eef_trimmed_fixture_v2_66ep_
relative_eef_chunk30_100k_globalbs32_gpu0_256_photoaug/
checkpoints/080000/pretrained_model
```

配置：

```text
chunk size:             30
batch size:             32
learning rate:          1e-5 cosine → 1e-6
vision encoder:         frozen
trainable:              action expert + state projection
image augmentation:     enabled
steps:                  1000
checkpoint interval:    100
normalization:          inherit from source 80k checkpoint
```

第一轮和修复 padding mask 后的对照模型均已保存。修复后模型路径：

```text
outputs/cobot_magic_smolvla_il2_localcorrection7x8_bc_
1000steps_bs32_chunk30_from80k_trimmedfixture_localmix8_padmaskfix_lr1e-5/
```

最终 step 1000 loss 约为 `0.024`；normalization、relative EEF processor 和 chunk-30 契约
均通过离线验证。

## 六、专项微调后出现的新问题

### 6.1 真机现象

`000100` checkpoint 已经能执行此前缺失的局部纠正动作，但出现三类退化：

1. 在纠正位置停留或反复调整，不能及时进入撤离阶段；
2. 撤离/回程速度变慢，偶尔不能回到原策略的稳定终点；
3. 工件已经放好并完成撤离后，又重新向抓取区运动并尝试再次夹取。

这说明模型并非完全没有学到新数据，而是**局部纠正行为被强化过度，同时破坏了完整任务的
阶段切换和终止附近动作分布**。

### 6.2 已排除：相邻 episode 被 action chunk 拼接

对 episode 65→66、66→67、72→73、79→80 和最后一个 episode 121 的实际训练索引进行了
检查。所有 30-step action chunk 均只包含当前 episode 的 action；超出末尾的索引被 clamp
到当前 episode 最后一帧，并标记为 padding。

因此不存在：

```text
上一条轨迹末尾（撤离/Ready）
→ 下一条局部轨迹开头（纠正位置）
```

被直接构造成同一个训练 chunk 的问题。

### 6.3 发现并修复：padding mask 键名错误

SmolVLA forward 原来读取：

```python
batch.get("actions_id_pad")
```

数据集实际提供：

```python
batch["action_is_pad"]
```

该问题导致 episode 尾部 padding action 未从 loss 中排除。现已同时修复普通 BC forward 和
RLT Stage 1 forward，并通过 SmolVLA 测试：`2 passed, 3 skipped`。

但是 A/B 检查表明，**padding 修复不是当前重复动作退化的直接原因**：

- 修复前的 x8/100 checkpoint 与修复后的 x8/100 checkpoint，在相同终端 observation 上
  输出几乎一致；
- 修复只影响训练 loss mask，不改变 rollout inference 路径；
- 两版 x8 专项模型都表现出相同的重新接近趋势。

### 6.4 原始 80k 与专项 `000100` 的闭环差异

最新 debug rollout 使用修复后的 `000100`，共记录 51 次 chunk decision：

```text
decision 46–47：完成放置后撤离，动作接近保持
decision 48：开始重新向抓取区域移动
decision 49–50：继续接近，并再次准备夹取
```

使用完全相同的终端图像、固定 flow-matching 随机种子进行离线 A/B：

- 原始 80k 在被强制输入该 `decision 48` 状态时也存在重新启动趋势；
- 但沿各自预测的撤离终点继续闭环一步时，80k 仅产生约 **2.2 cm** 的轻微位置漂移；
- 专项 `000100` 产生约 **11.5 cm** 的明显重新接近动作。

因此用户观察到“80k 不会重复、专项模型会重复”是成立的。更准确的解释是：原始 80k
位于一个相对稳定的终端/保持区域，而 x8 专项微调轻微改变了撤离终点和后续动作，使闭环
状态越过阶段边界，进入了模型原有的“新一轮接近/抓取”吸引域。

### 6.5 当前最可能根因

当前混合方式把 7 条完全相同的局部轨迹复制 8 次，使局部帧占总训练帧约 35%。这些轨迹
的主体是长时间对位/下压，而撤离段较短；它们的撤离终点 EEF 又与完整任务的 terminal/
起始附近状态重叠。

在 vision encoder 冻结、只更新 action expert 和 state projection 的情况下，模型更容易学习：

```text
当前 EEF/state 所处阶段
→ 继续播放高频出现的局部轨迹或下一阶段轨迹
```

而不是稳定使用“工件已在目标位置、起始抓取区已为空”的视觉差异。终端图像从人眼看与任务
起点不同，但金属工件和夹具在右腕相机中尺寸较小、反光明显且部分受机械臂遮挡；专项训练
又没有专门构造固定 state 下“已放好/未放好”的视觉对照。因此更符合
**state/phase-conditioned trajectory replay + 局部样本过权重**，而不是数据发生跨 episode
拼接。

## 七、RLT / 在线强化学习探索

本阶段同时完成了 RLT 基础设施和多轮 Stage 1/Stage 2 试验，包括：

- Stage 1 RL token/embedding 蒸馏；
- Stage 2 actor BC warm-up；
- reference-only、residual/direct actor、不同 actor chunk 的对照；
- terminal sparse reward、RECAP value/稠密 reward 和 actor–critic 更新；
- mode 1 自主 rollout 与 mode 2 人工干预数据采集；
- persistent rollout server、actor checkpoint 选择、rollback 和异步训练状态展示。

现阶段结论是：基础 BC 尚存在低位对位和终止阶段问题时，RLT/TD3 critic 很容易利用不可靠
的局部 Q 梯度破坏原本可用的抓取能力。当前优先级应回到 BC 数据分布和专项纠正训练，先
获得稳定、不回归的基准策略，再继续在线 RL 微调。

## 八、下一阶段建议

### 8.1 立即停止使用 x8 全轨迹复制作为最终训练方案

保留当前 x8 checkpoints 作为诊断对照，但真机基准应回到原始 80k。不要继续增加 x8 模型
训练步数；当前问题是数据权重和状态分布冲突，不是训练不足。

### 8.2 重新构造局部纠正样本

建议从 7 条局部轨迹中只保留：

```text
未放准的局部状态
→ 对位/下压纠正
→ 明确嵌合
→ 释放
→ 短距离安全撤离
```

删除长时间等待、重复观察、重复下压，以及进入与完整任务 terminal/起始状态重叠区域后的
长回程。完整回程继续由原 66 条全任务示范提供监督。

### 8.3 降低专项数据权重

第一轮建议采用 `7×1` 或 `7×2`，而不是 `7×8`：

```text
66 条完整任务 + 7 条局部纠正 × 1/2
```

并保存更密集的早期 checkpoint，例如 20、40、60、100 steps，分别做完整任务真机验证。
专项数据的目标是补足一个局部动作分支，而不是占据三分之一训练分布。

### 8.4 建立防回归离线评估

至少固定三组 observation：

1. 正常任务起始画面；
2. 未完全嵌合、需要纠正的画面；
3. 已完全嵌合并撤离、应保持或结束的画面。

对每个 checkpoint 比较 30-step EEF 输出：

- 起始画面是否仍能正常抓取；
- 未嵌合画面是否产生正确纠正；
- 已完成画面是否重新进入抓取轨迹；
- EEF endpoint、chunk path length、gripper 时序和多随机种子方差。

只有同时满足“保留原任务能力 + 学会纠正 + 不重复启动”的 checkpoint 才进入真机测试。

### 8.5 验证视觉是否真正参与完成判断

使用同一 EEF state，分别输入：

```text
工件未嵌合
工件已嵌合
抓取区有工件
抓取区无工件
```

比较模型第一段动作和预测终点。如果输出差异很小，说明当前模型主要依赖 state/phase；应
增加对应视觉对照数据，或对视觉—动作层进行受控微调，而不是继续重复同一轨迹。

## 九、阶段交付物

主要代码和文档包括：

```text
cobot_bridge/                                  真机 bridge、rollout、dashboard、debug recorder
pink_ik/                                       EEF→joint IK 与轨迹执行
scripts/build_cobot_magic_il2_trimmed_fixture_v2.sh
scripts/build_cobot_magic_il2_local_correction7_bc.sh
scripts/train_cobot_magic_smolvla_local_correction7_bc.sh
docs/2026-09-13_cobot_magic_work_summary.md
docs/2026-09-10_to_2026-09-17_insert_workpiece_project_report.md
```

当前推荐基准 checkpoint：

```text
outputs/cobot_magic_smolvla_il2_eef_trimmed_fixture_v2_66ep_
relative_eef_chunk30_100k_globalbs32_gpu0_256_photoaug/
checkpoints/080000/pretrained_model
```

当前专项诊断 checkpoint（不建议直接作为最终基准）：

```text
outputs/cobot_magic_smolvla_il2_localcorrection7x8_bc_
1000steps_bs32_chunk30_from80k_trimmedfixture_localmix8_padmaskfix_lr1e-5/
checkpoints/000100/pretrained_model
```

## 十、总结

本周期已经完成从设备、控制、数据、训练到真机评估的闭环，基础 BC 模型达到约 60% 的阶段
成功率，并明确定位了剩余瓶颈：最后低位插入对位和任务完成判断。局部纠正数据证明模型可以
学习二次对位动作，但直接将少量局部轨迹复制 8 次会显著改变完整任务的阶段分布，造成卡在
纠正区、撤离退化和任务重复执行。

下一阶段应以原始 80k 为稳定基线，对局部数据做精确时序裁剪和低权重混合，并建立起始、
纠正、已完成三类固定 observation 的离线防回归评估。只有通过该评估后，才继续真机测试
和 RLT 在线强化学习。
