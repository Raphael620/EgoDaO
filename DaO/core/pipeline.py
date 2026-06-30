"""DepthAI v3 pipeline: 3×Camera + IMU + StereoDepth → RTABMapVIO."""
from __future__ import annotations

import depthai as dai
from depthai import CameraBoardSocket
from DaO.config import AppConfig

_SOCKET_LABEL = {
    CameraBoardSocket.CAM_A: "center",
    CameraBoardSocket.CAM_B: "left",
    CameraBoardSocket.CAM_C: "right",
}
_SOCKET_ORDER = [CameraBoardSocket.CAM_B, CameraBoardSocket.CAM_A, CameraBoardSocket.CAM_C]


def _find_socket(connected: list, role: str):
    for s in connected:
        if _SOCKET_LABEL.get(s) == role:
            return s
    mapping = {"left": 0, "center": 1, "right": 2}
    idx = mapping.get(role, -1)
    return connected[idx] if 0 <= idx < len(connected) else None


def _ensure_imu_calibration(device: dai.Device):
    calib = device.readCalibration()
    try:
        calib.getImuToCameraExtrinsics(CameraBoardSocket.CAM_B)
    except RuntimeError:
        calib.setImuExtrinsics(CameraBoardSocket.CAM_B,
                               [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                               [0.0, 0.0, 0.0])
        device.flashCalibration(calib)


def create_pipeline(device: dai.Device, config: AppConfig):
    _ensure_imu_calibration(device)

    p = dai.Pipeline(device)
    connected = device.getConnectedCameras()
    ordered = [s for s in _SOCKET_ORDER if s in connected]
    if not ordered:
        raise RuntimeError("No cameras detected.")

    cam_nodes: dict[str, dai.node.Camera] = {}
    cam_queues: dict[str, dai.DataOutputQueue] = {}
    resolution = config.camera.resolution
    vio_res = config.vio_camera_resolution

    for role in ("left", "center", "right"):
        sock = _find_socket(ordered, role)
        if sock is None:
            continue
        cam = p.create(dai.node.Camera).build(sock, sensorFps=30)
        try:
            cam_queues[role] = cam.requestFullResolutionOutput().createOutputQueue(maxSize=4, blocking=False)
        except RuntimeError:
            cam_queues[role] = cam.requestOutput(resolution).createOutputQueue(maxSize=4, blocking=False)
        cam_nodes[role] = cam

    imu = p.create(dai.node.IMU)
    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], config.imu.accel_rate_hz)
    imu.setBatchReportThreshold(config.imu.batch_threshold)
    imu.setMaxBatchReports(config.imu.max_batch_reports)
    imu_queue = imu.out.createOutputQueue(maxSize=50, blocking=False)

    vio_queue: dai.DataOutputQueue | None = None
    left_cam = cam_nodes.get("left")
    right_cam = cam_nodes.get("right")

    if config.enable_vio and left_cam is not None and right_cam is not None:
        try:
            stereo = p.create(dai.node.StereoDepth)
            stereo.setDepthAlign(CameraBoardSocket.CAM_B)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setExtendedDisparity(False)
            stereo.enableDistortionCorrection(True)
            left_cam.requestOutput(vio_res).link(stereo.left)
            right_cam.requestOutput(vio_res).link(stereo.right)
            vio = p.create(dai.node.RTABMapVIO)
            stereo.rectifiedLeft.link(vio.rect)
            stereo.depth.link(vio.depth)
            imu.out.link(vio.imu)
            vio_queue = vio.transform.createOutputQueue(maxSize=8, blocking=False)
        except Exception as e:
            print(f"VIO setup failed: {e}")
            vio_queue = None

    return p, cam_queues, imu_queue, vio_queue
