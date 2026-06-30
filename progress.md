# Ego Daq-O — 项目进度说明 (2026-06-30)

## 已完成

### 项目基础设施
- [x] pyproject.toml — uv + Python 3.12，核心依赖 depthai / numpy / opencv-python
- [x] .gitignore — 排除 .venv、Data、depthai-core、HumanEgo、depthai_hand_tracker、test/
- [x] 目录结构 — DaO/{config,core,ui} 三层模块化
- [x] README.md — 中英文用户教程，一键启动脚本 run.bat / run.sh
- [x] PROGRESS.md & TODO.md — 开发进度 & 待办清单

### 相机 + IMU
- [x] pipeline.py — DepthAI v3：3×Camera @ 30fps + IMU + StereoDepth → RTABMapVIO
- [x] capture_worker.py — CaptureWorker（QObject，采集循环跑在 threading.Thread）
- [x] IMU 标定自动化（检查 identity extrinsics）

### 界面
- [x] main_window.py — 工具栏 + 三目视图 + IMU/VIO 面板 + 状态栏
- [x] camera_view.py — CameraPane ×3，手部骨架 overlay，30fps 渲染限流，单通道兼容
- [x] imu_panel.py — 加速度计/陀螺仪条形图 + 互补滤波姿态估计
- [x] imu_3d_widget.py — IMU 3D 姿态（QPainter 投影）
- [x] vio_3d_widget.py — VIO 3D 轨迹（自适应缩放）

### 手部追踪
- [x] hand_tracker.py — MediaPipe 0.10 Tasks API，每帧检测
- [x] 支持 mono → RGB 转换（黑白镜头兼容）
- [x] 置信度阈值 0.3（适配黑白相机成像质量）
- [x] 信号驱动 UI 更新（hands_ready），不受 timer 轮询

### 数据录制
- [x] DataRecorder — Raw 格式：mp4 + imu.csv + hands.json + vio.json
  - VideoWriter 延迟初始化（按实际帧分辨率）
  - 单通道帧自动转三通道
- [x] HumanEgoRecorder — HumanEgo 兼容格式（详见下文）
  - 输出目录: `Data/HumanEgo/mps_{ts}_vrs/preprocess/all_data/{idx:05d}/`
  - per-frame JSON: `aria_cam_rgb.json` / `aria_slam.json` / `aria_hands.json` / `training_data.json`
  - 后台 threading.Thread 异步落盘，stop() 立即返回

### 线程架构
- [x] CaptureWorker — threading.Thread（采集 + 手部检测），信号跨线程 dispatch
- [x] HumanEgoRecorder — threading.Thread daemon（磁盘 I/O）
- [x] UI 线程独立 — 仅做渲染，不受 I/O 和计算阻塞

### 代码清理
- [x] 移除 depthai_hand_tracker 旧文件（hand_tracker_v3, mediapipe_utils, blob, models）到 test/
- [x] 移除旧的 hand_overlay.py 到 test/
- [x] 精简所有 DaO/*.py 文件，移除冗余注释和未使用代码

## 已知问题

### 高优先级
- [ ] VIO 跟踪质量：静止时手部移动误判为相机运动；移动时可能丢失跟踪
- [ ] 相机内参：当前使用默认值，需要从 device calibration 获取

### 中优先级
- [ ] IMU-to-Camera 标定 — 需要 checkerboard + depthai 标定工具
- [ ] BasaltVIO 不可用 — RVC2 固件不支持，需 RVC4 或新版 depthai
- [ ] Ubuntu 适配测试

### 低优先级
- [ ] 单元测试
- [ ] on-device NN 手部检测（hand_tracker_v3）集成

## 模块依赖

```
DaO/main.py
 └── MainWindow (main_window.py)
      ├── CameraViewWidget — CameraPane ×3 (camera_view.py)
      ├── CaptureWorker (capture_worker.py)
      │   ├── pipeline.py — 3×Camera + IMU + StereoDepth + RTABMapVIO
      │   ├── hand_tracker.py — MediaPipe Tasks API
      │   ├── recorder.py — Raw 录制
      │   └── humanego_recorder.py — HumanEgo 录制
      ├── ImuPanel (imu_panel.py)
      ├── Imu3DWidget (imu_3d_widget.py)
      └── Vio3DWidget (vio_3d_widget.py)
```

## 运行方式

```bash
# Windows
run.bat

# Ubuntu / macOS
./run.sh
```
