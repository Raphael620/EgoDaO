# Ego Daq-O — 项目进度说明 (2026-07-16)

## 运行方式

```bash
# GUI 模式
python -m DaO.main
main.exe                    # Nuitka 编译后

# 无GUI录制模式（Windows 开机启动）
start_no_gui.bat            # 放入 shell:startup 文件夹
main.exe --no-gui           # 命令行直接启动
```
Ctrl+Q 启停录制，Ctrl+C 退出。

## 配置文件 (config.json)

放置于项目根目录，示例见 `config.example.json`：

```json
{
  "recording": {
    "data_root": "./Data",
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

## 关键改动 (vs 0.2.1)

- 时序滤波器（异常值剔除 + 自适应平滑 + 左右手稳定性）
- 3D 深度恢复（MediaPipe world_landmarks + 相机内参）
- Aria MPS 关键点重映射
- 无GUI 录制模式 + Ctrl+Q 全局热键
- 文件日志（`logs/` 目录，保留最近 10 个文件）
- 自动清理旧数据（按日期文件夹，保留最近 N 天）
- IMU 采样率可配置（默认 200Hz, batch=1）
- 编码器容错（mp4v/XVID 自动回退，灰度图自动转 BGR）
- 录制防损坏（VideoWriter 写入失败容错，退出时 signal handler 保存数据）
- Nuitka 编译脚本（`build.bat` / `build.sh` / `build-here.bat`）

## 已知问题

- VIO 跟踪质量：静止时手部移动误判为相机运动
- 相机内参使用 OAK 校准数据（fx≈567）
- OpenVINO 后端偶有左右手误判、GPU 设备稳定性
- OAK 设备 crash 后需物理拔插
