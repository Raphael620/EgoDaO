"""USB UVC camera + RK3588 NPU hand tracking — live window."""
import cv2, numpy as np, time, os, sys

sys.path.insert(0, "/home/admin/projects/egodao")
from DaO.core.hand_tracker_rknn import create_rknn_tracker

tracker = create_rknn_tracker(K=None)

cap = cv2.VideoCapture("/dev/video3")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

COLOR = (0, 255, 0)
t0 = time.perf_counter()
fc = 0
print("RK3588 NPU Hand Tracking — press Q to quit", flush=True)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    fc += 1
    hands, _ = tracker.process(frame)

    for label, kpts in hands:
        for a, b in CONNS:
            cv2.line(frame, (int(kpts[a,0]), int(kpts[a,1])),
                     (int(kpts[b,0]), int(kpts[b,1])), COLOR, 2)
        for j in range(21):
            if j == 0:
                cv2.circle(frame, (int(kpts[j,0]), int(kpts[j,1])), 6, (0,0,255), -1)
            else:
                cv2.circle(frame, (int(kpts[j,0]), int(kpts[j,1])), 4, COLOR, -1)
        cv2.putText(frame, label, (int(kpts[0,0])+10, int(kpts[0,1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if fc % 30 == 0:
        fps = 30 / (time.perf_counter() - t0 + 1e-6)
        t0 = time.perf_counter()
        print(f"FPS: {fps:.1f}  Hands: {len(hands)}", flush=True)

    cv2.imshow("RK3588 NPU Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.close()
cap.release()
cv2.destroyAllWindows()
print("Done.")
