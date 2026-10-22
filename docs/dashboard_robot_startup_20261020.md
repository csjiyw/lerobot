# Dashboard 一键自检与启动

入口：System → 机器人自检与一键启动。

- **仅自检**：通过 SSH 检查 AGX 进程、CAN、bridge 数据、相机和预览，不启动程序。
- **一键自检并启动机器人服务**：自检 → 生成缺失组件计划 → 补启动 → 连续复检。
- 报告在当前卡片展开查看；详细日志在 Jobs & Logs；JSON 留存于 Dashboard 的 web_runs。

## 启动范围

1. 补启动缺失的 bridge，使冷启动时也能检查控制状态。
2. 必要时复用 AGX 的 configure_cobot_magic_can.sh 配置四路 CAN。
3. 四臂 ROS 驱动与控制仲裁器，三相机 ROS launch，本地 latest-JPEG 预览。
4. 检查四臂 7 维反馈、三路相机 fresh，预览版本连续增长，控制状态 SAFE、Teach 关闭、输出关闭。

复用既有 cobot_bridge / cobot_magic:arms,camera / cobot_collect:preview 会话命名，不覆盖机器人端文件。
本功能不启动云端 middleware、sampler、假底盘、policy server 或录制。这些不是 Dashboard 手动控制的必要服务。

## 安全与限制

- 不回零、不使能机械臂、不发送关节目标、不清除急停、不自动开启运动输出。
- 机械臂 launch 的 auto_enable=false；live_output=true 只保留现有仲裁能力，仍由 SAFE/output 门控。
- 录制、运动任务、Teach 或运动输出开启时，不进行补启动。
- 正常组件复用；已有进程但没数据时等待复检，不自动强杀或重复创建驱动。异常需要人工处理或明确使用原有 Restart All Devices。
- 启动失败不回滚已成功启动的服务，报告保留已启动列表；机械臂输出不会因此打开。
- 就绪表示软件数据链路就绪，不代表机械限位、碰撞空间、任务姿态或模型能力通过安全验收；实际运动仍走原来的确认流程。

## 配置

复用 COBOT_ROBOT_SSH_HOST、COBOT_BRIDGE_ENDPOINT、COBOT_ROBOT_CONTROL_WORKSPACE。
CAN 需要权限时，在 Dashboard 进程环境配置 COBOT_ROBOT_SUDO_PASSWORD_FILE 为本机 owner-only（600）凭据文件。
凭据仅经 SSH stdin 交给 sudo，不出现在报告或命令行。没有凭据则在 CAN 步骤明确失败，不弹交互密码框。

## 本次验证

2026-10-20：本机单测、前端语法检查，以及 AGX 只读自检。
AGX SSH 可达；can0–can3 未启动，目标逻辑 CAN 名未就绪，未发现 bridge/四臂/相机/预览进程。
没有运行真机补启动，也没有执行机械臂运动测试。
