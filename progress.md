# Ego Daq-O — 项目进度说明（2026-08-03）

应用名与当前版本统一定义在 `DaO/config/config.py`；Python 包、GUI、日志及
打包配置均从该处读取。本文件以下按发布版本保留历史修改记录。

## 运行方式

```bash
# GUI 模式
python -m DaO.main
main.exe                    # Nuitka 编译后

# 无GUI录制模式（Windows 开机启动）
start_no_gui.bat            # 放入 shell:startup 文件夹
main.exe --no-gui           # 命令行直接启动
```
Headless 启动后处于 idle。Ctrl+Q 可在“连接并录制”和“保存并断开”之间切换；
本机 `127.0.0.1:9876` 同时接受 `start`、`stop`、`quit` 命令。

## 配置文件 (config.json)

放置于项目根目录，示例见 `config.example.json`：

```json
{
  "recording": {
    "data_root": "./Data",
    "video_backend": "ffmpeg_hw",
    "hw_encoder": "h264_qsv",
    "enable_raw": true,
    "enable_humanego": false,
    "disk_keep_days": 3
  },
  "camera": { "fps": 30 },
  "imu": {
    "accel_rate_hz": 200,
    "gyro_rate_hz": 200,
    "batch_threshold": 1
  },
  "app": {
    "enable_vio": true,
    "enable_hand_tracking": true,
    "hand_tracker_backend": "mediapipe"
  }
}
```

## 手部追踪后端

| 后端 | 配置值 | 说明 |
|------|--------|------|
| MediaPipe | `"mediapipe"` | **默认**，CPU 推理，准确稳定 |
| OpenVINO | `"openvino"` | Intel GPU 加速 (~2.5x)，左右手判定偶有偏差 |
| Mercury | `"mercury"` | Monado ONNX 实验方案 |

通过 `config.json` 中的 `hand_tracker_backend` 切换。

## 数据格式

- **Raw**：`Data/Raw/{timestamp}/` → mp4 + imu.csv + hands.jsonl + vio.jsonl
- **HumanEgo**：`Data/HumanEgo/mps_{timestamp}_vrs/preprocess/all_data/` → per-frame PNG + JSON
- 转换：GUI 工具栏"转换"按钮，选择 Raw 文件夹即可转 HumanEgo

## 0.3.0 已完成

- 保留原有 DepthAI 采集、同步视频写入和逐中心帧 MediaPipe 推理架构；未引入异步视频队列或额外采集定时器。
- 修复 Headless 热键未真正启动 recorder、socket `start` 未启动相机、idle 状态无法 `quit` 及停止顺序问题。
- MediaPipe detector 在进程内复用；断开相机时只重置滤波状态，避免离线环境同步关闭遥测造成约 42 秒阻塞。
- 修复 Raw → HumanEgo 的 hands/VIO 按帧索引对齐、视频打开检查和资源释放。
- 修复 HumanEgo 空帧崩溃、手部参数顺序、HumanEgo-only 长录制丢失早期帧及退出时后台 flush 丢失。
- 修复 SLAM 欧拉角跨 ±180° 跳变和非单调时间戳速度异常。
- 修复 IMU 加速度单位重复除以 1000、使用真实采样时间间隔并保护 pitch 奇异点。
- 修复 GUI 后台转换线程直接操作 Qt 控件；恢复高质量预览缩放，并减少 VIO 重复重绘。
- 修复重复初始化日志时 handler 叠加；补全 Nuitka/包数据配置。

## 已验证

- `compileall` 通过，4 项核心数据路径回归测试通过，`git diff --check` 通过。
- Windows + 已连接 OAK：GUI 原信号录制链 12 秒得到左/中/右 `353 / 355 / 351` 帧，约 `29.25–29.58 FPS/路`，同期执行 355 次 MediaPipe 手部推理。
- Headless 通过 socket 完成 `start → stop → quit`；相机实际运行约 8 秒得到 `250 / 252 / 249` 帧，三路视频均可解码且声明 30 FPS。
- 单独稳态观测中三路均保持 `29.9–30.0 FPS/路`；状态栏的 FPS 明确表示中心相机单路帧率，不是三路总和。

## 已知问题

- VIO 跟踪质量：静止时手部移动误判为相机运动
- 相机内参使用 OAK 校准数据（fx≈567）
- OpenVINO 后端偶有左右手误判、GPU 设备稳定性
- OAK 设备 crash 后需物理拔插
- 默认视频路径仍沿用远程原版的主机 FFmpeg QSV 配置；目标主机需验证编码器和可执行文件可用性，可切换到 `opencv`
- RK3588/ARM 不在当前分支范围内，后续单独开发
