# AGX 单机版数据采集操作说明

本流程用于在 AGX 本机测试原版 `sampler_rosbag` 单机采集链路。

注意：本流程不会修改四条机械臂的 CAN 映射或主从模式。底盘话题使用零值兼容心跳，只用于通过网页和中间件的数据存在性检查，不能用于训练或导航。

## 一、启动机器人和相机

如果四条机械臂和三路相机已经在运行，不要重复启动。

如果还没有启动，在 AGX 终端执行：

```bash
bash /home/agilex/lerobot/scripts/start_cobot_magic.sh
```

这个启动脚本包含：

- 四条机械臂 ROS 节点；
- 三路 RealSense 相机；
- `/cmd_vel` 零值心跳；
- `/odom` 零值心跳。

相机启动后等待 5–10 秒，再启动 sampler。

如果启动脚本发现已有 ROS 进程，不要强行重复启动。此时可以单独启动假底盘：

```bash
bash /home/agilex/lerobot/scripts/start_fake_base_heartbeat.sh
```

不要同时运行旧的：

```bash
/home/agilex/start.sh
```

旧脚本会启动真实底盘和另一套 ROS 控制链路，可能与单机测试的零值 `/cmd_vel`、`/odom` 冲突。

## 二、启动单机 sampler

新开一个 AGX 终端：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/openloong_ws/install/setup.bash
unset ROS_DOMAIN_ID
export ROS_LOCALHOST_ONLY=1

ros2 launch sampler_rosbag sampler.launch.py \
  params_file:=/home/agilex/lerobot/config/sampler_single_machine.yaml
```

看到下面的日志说明 sampler 已经启动：

```text
WebSocket server started on ws://0.0.0.0:18780
```

这个单机配置包含：

```text
/cmd_vel                       50 Hz，假零值
/odom                          50 Hz，假零值
/camera_f/color/image_raw/compressed
/camera_l/color/image_raw/compressed
/camera_r/color/image_raw/compressed
```

## 三、打开单机网页

新开另一个 AGX 终端：

```bash
cd /home/agilex/local_viewer
python3 -m http.server 8080 --bind 127.0.0.1
```

在 AGX 自带浏览器地址栏打开：

```text
http://127.0.0.1:8080
```

网页顶部应显示：

```text
ws://127.0.0.1:18780
已连接
```

这是 AGX 单机测试网页，不是项目 Dashboard 的 `8088` 页面。

## 四、网页采集步骤

按以下顺序操作：

1. 点击“预览”；
2. 确认 `camera_f`、`camera_l`、`camera_r` 都有画面；
3. 点击“开始采集”；
4. 等待 5–10 秒；
5. 确认进度中的 `/cmd_vel`、`/odom` 和三路相机计数持续增加；
6. 点击“停止”；
7. 点击“后处理”。

默认数据位置：

```text
/tmp/data_storage/record/raw_data
```

## 五、检查 MCAP

采集结束后，在终端执行：

```bash
source /opt/ros/humble/setup.bash
ros2 bag info /tmp/data_storage/record/raw_data
```

应能看到以下类型的话题：

```text
/cmd_vel                         geometry_msgs/msg/Twist
/odom                            nav_msgs/msg/Odometry
/camera_f/color/image_raw/compressed
/camera_l/color/image_raw/compressed
/camera_r/color/image_raw/compressed
```

## 六、停止服务

分别回到对应终端按 `Ctrl+C`：

- 停止网页 HTTP 服务；
- 停止 sampler；
- 停止假底盘心跳。

不要为了停止单机采集而重启或修改机械臂 CAN 配置。

## 七、已验证结果

一次约 5.75 秒的测试结果：

```text
MCAP 大小：约 36.9 MiB
消息数：约 6179
/cmd_vel：约 288 条
/odom：约 288 条
camera_f：约 173 帧
camera_l：约 173 帧
camera_r：约 153 帧
```

假底盘数据必须在后续 H5/LeRobot 转换时排除，并标记为：

```text
base_source=synthetic_zero
base_data_valid=false
```

## 八、常见问题

### 网页显示未连接

确认 sampler 仍在运行，并检查端口：

```bash
ss -ltnp | grep 18780
```

### 前视相机没有画面

先等待 5–10 秒。若仍无画面，检查相机进程是否存在：

```bash
ps -ef | grep realsense2_camera | grep -v grep
```

### `/cmd_vel` 或 `/odom` 计数为 0

确认假底盘心跳正在运行：

```bash
ps -ef | grep fake_base_heartbeat | grep -v grep
```

并确认 sampler 使用的是：

```text
/home/agilex/lerobot/config/sampler_single_machine.yaml
```
