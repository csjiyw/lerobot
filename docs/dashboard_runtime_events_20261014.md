# Dashboard 任务完成通知（SSE）

## 行为

保留手动状态刷新，不恢复整页定时轮询。首次打开仍加载一次overview；操作后的必要刷新保留；相机仍走原MJPEG链路。

新增同源`GET /api/events`长连接，复用原X-Cobot-Token认证，token不放URL。后台只有一个轻量观察线程，检测已管理进程的运行/完成/失败和初始化phase变化；模型服务状态每2秒集中检查一次。不同网页共享这套观察结果，不各自扫描目录。

录制进程结束后，读取该任务metadata.output指定的单个H5结果；完成和成功以实际文件attrs为准，不把进程返回0直接当success。网页收到通知后恢复录制按钮、更新任务状态、合并新增episode并显示结果。

模型ready/offline、rollout或replay结束也推送。通知不发机器人控制命令，不改变归一化、模型或训练。

## 重连

浏览器用fetch读取SSE分块，携带Last-Event-ID重连。服务器保留最近128条事件；客户端超过缓冲或服务器重启时发送轻量runtime_snapshot，不扫描全部数据。快照包含最近最多128条录制文件结果，恢复按钮和新增记录。心跳15秒；断线提示并每3秒重连。

手动overview与事件并发时合并已到达的新事件，避免旧overview覆盖录制已结束状态。新episode、jobs和model状态局部更新；完整目录清单仍手动刷新。

## 文件与验证

- `cobot_bridge/dashboard_events.py`：观察器、事件环形缓冲、快照。
- `cobot_bridge/web_dashboard.py`：SSE路由和单文件raw读取。
- `cobot_bridge/web_static/app.js`：流读取、重连、局部渲染。
- `tests/cobot_bridge/test_dashboard_events.py`：录制完成只读单文件一次、时间戳不制造事件、历史任务不误通知、重放/缓冲溢出/重启。

38项相关测试通过；既有RLT测试`test_rlt_tab_inventory_and_read_only_jobs`因其actor-max-decisions旧断言与已有代码不一致排除，未改RLT逻辑。JS语法和分块流模拟通过，验证auth header、cursor、录制完成状态与原始记录追加，没有overview轮询。

已重启Dashboard部署；部署前确认没有录制或rollout进程。模型服务和GPU1训练保持运行。上线SSE实测HTTP200及runtime_snapshot，policy_ready=True。尚未代用户进行真实Teach录制测试。

PC和AGX旧页面需刷新一次加载新JS；随后应看到“任务通知已连接”。通知不更新所有设备实时状态，设备状态仍以手动快照为准。
