"""USB UVC camera + TFLite CPU hand tracking — live window."""
import cv2, numpy as np, time, os, sys

sys.path.insert(0, "/home/admin/projects/egodao")
from DaO.core.hand_tracker_tflite import create_tflite_tracker

tracker = create_tflite_tracker(K=None)

cap = cv2.VideoCapture("/dev/video3")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

t0 = time.perf_counter()
fc = 0
print("TFLite CPU Hand Tracking — press Q to quit", flush=True)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    fc += 1
    hands, _ = tracker.process(frame)

    for label, kpts in hands:
        for a, b in CONNS:
            cv2.line(frame, (int(kpts[a,0]), int(kpts[a,1])),
                     (int(kpts[b,0]), int(kpts[b,1])), (0,255,0), 2)
        for j in range(21):
            r = 6 if j == 0 else 4
            cv2.circle(frame, (int(kpts[j,0]), int(kpts[j,1])), r, (0,0,255) if j==0 else (0,255,0), -1)
        cv2.putText(frame, label, (int(kpts[0,0])+10, int(kpts[0,1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if fc % 30 == 0:
        fps = 30 / (time.perf_counter() - t0 + 1e-6)
        t0 = time.perf_counter()
        print(f"FPS: {fps:.1f}  Hands: {len(hands)}", flush=True)

    cv2.imshow("TFLite CPU Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.close()
cap.release()
cv2.destroyAllWindows()
print("Done.")
