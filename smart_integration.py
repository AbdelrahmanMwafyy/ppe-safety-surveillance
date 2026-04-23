import cv2
import numpy as np
import os
import urllib.parse
import time
import torch
from threading import Thread, Lock
from queue import Queue, Empty
from ultralytics import YOLO
from zone import ZONES

# ===============================
# CONFIG
# ===============================
script_dir = os.path.dirname(os.path.abspath(__file__))
PERSON_MODEL  = os.path.join(script_dir, "yolov8n.pt")
PPE_MODEL_PATH = r"C:\Mwafy\uni\GP\Smart-System\ppe\best (4).pt"

TAPO_IP     = "your_camera_ip"
TAPO_USER   = "your_camera_username"
TAPO_PASS   = "your_camera_password"
RTSP_URI = f"rtsp://{urllib.parse.quote(CAM_USER)}:{urllib.parse.quote(CAM_PASS)}@{CAM_IP}:554/stream1"

FRAME_W, FRAME_H = 1280, 720
INFER_W, INFER_H = 640, 360
scale_x = FRAME_W / INFER_W
scale_y = FRAME_H / INFER_H

# ===============================
# LOAD MODELS
# ===============================
person_model = YOLO(PERSON_MODEL)
ppe_model    = YOLO(PPE_MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
person_model.to(device)
ppe_model.to(device)
print(f"✔ Using {'GPU' if device == 'cuda' else 'CPU'} for inference")

ppe_class_names  = ppe_model.names
violation_classes = {'NO-Hardhat', 'NO-Safety-Vest'}
print(f"Monitoring: {violation_classes}")


# ===============================
# VIDEO STREAM
# ===============================
class VideoStream:
    """
    The thread calls cap.read() in a tight loop and ALWAYS overwrites
    self.frame with the newest result. The main thread just picks up
    whatever is there — no buffer, no drain loop, no lag.
    """
    def __init__(self, src):
        # Minimal, proven RTSP options — nothing fancy that breaks cameras
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        self.src   = src
        self.cap   = self._open()
        self.frame = None
        self.ret   = False
        self.lock  = Lock()
        self.stopped = False
        Thread(target=self._update, daemon=True).start()

    def _open(self):
        print(f"Connecting to {self.src} ...")
        cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("✔ Opened" if cap.isOpened() else "✘ Failed to open — check camera/network")
        return cap

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                print("Reconnecting in 2s...")
                self.cap.release()
                time.sleep(2)
                self.cap = self._open()
                continue

            # read() blocks until the NEXT frame arrives from the camera,
            # then immediately returns it. Because this thread loops as fast
            # as the camera sends frames, self.frame is ALWAYS the newest one.
            ret, frame = self.cap.read()

            if ret and frame is not None:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


# ===============================
# ASYNC INFERENCE WORKER
# ===============================
class InferenceWorker:
    """
    Runs in its own thread. Queue size = 1 means if inference is slow,
    stale frames are dropped automatically — no backlog ever builds up.
    """
    def __init__(self):
        self.q      = Queue(maxsize=1)
        self.lock   = Lock()
        self.persons    = []
        self.violations = []
        self.stopped = False
        Thread(target=self._run, daemon=True).start()

    def submit(self, frame_small):
        # Drop pending frame before adding new one
        try:
            self.q.get_nowait()
        except Empty:
            pass
        try:
            self.q.put_nowait(frame_small)
        except Exception:
            pass

    def _run(self):
        while not self.stopped:
            try:
                frame_small = self.q.get(timeout=0.5)
            except Empty:
                continue

            p_res = person_model(frame_small, conf=0.40, verbose=False)
            v_res = ppe_model   (frame_small, conf=0.35, verbose=False)

            new_p, new_v = [], []

            for r in p_res:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                        new_p.append((int(x1*scale_x), int(y1*scale_y),
                                      int(x2*scale_x), int(y2*scale_y)))

            for r in v_res:
                for box in r.boxes:
                    name = ppe_class_names[int(box.cls[0])]
                    if name in violation_classes:
                        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                        new_v.append((int(x1*scale_x), int(y1*scale_y),
                                      int(x2*scale_x), int(y2*scale_y), name))

            with self.lock:
                self.persons    = new_p
                self.violations = new_v

    def get(self):
        with self.lock:
            return list(self.persons), list(self.violations)

    def stop(self):
        self.stopped = True


# ===============================
# HELPERS
# ===============================
def box_center(b):
    return ((b[0]+b[2])//2, (b[1]+b[3])//2)

def in_zone(point, polygon):
    pts = np.array([tuple(map(int, p)) for p in polygon], np.int32)
    return cv2.pointPolygonTest(pts, point, False) >= 0


# ===============================
# STARTUP
# ===============================
vs     = VideoStream(RTSP_URI)
worker = InferenceWorker()

cv2.namedWindow("Integrated System: Zones + PPE", cv2.WINDOW_NORMAL)

print("Waiting for first frame (up to 15s)...")
deadline = time.time() + 15
while time.time() < deadline:
    ret, frame = vs.read()
    if ret and frame is not None:
        print("✔ Stream live — starting.")
        break
    time.sleep(0.1)
else:
    print("✘ No frame after 15s. Check camera IP/credentials.")
    vs.stop()
    cv2.destroyAllWindows()
    exit(1)


# ===============================
# MAIN LOOP
# ===============================
n = 0
INFER_EVERY = 20   # increase if CPU is too slow (e.g. 8 or 10)

while True:
    ret, frame = vs.read()
    if not ret or frame is None:
        time.sleep(0.005)
        continue

    n += 1
    if n % INFER_EVERY == 0:
        worker.submit(cv2.resize(frame, (INFER_W, INFER_H)))

    persons, violations = worker.get()
    disp = cv2.resize(frame, (FRAME_W, FRAME_H))

    # Zones
    for zname, zone in ZONES.items():
        pts   = np.array(zone["points"], np.int32)
        color = zone["color"]
        hit   = any(in_zone(box_center(p), zone["points"]) for p in persons)
        col   = (0, 0, 255) if hit else color
        cv2.polylines(disp, [pts], True, col, 2)
        cv2.putText(disp, zname, (pts[0][0]+5, pts[0][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    # Persons
    for x1,y1,x2,y2 in persons:
        cv2.rectangle(disp, (x1,y1), (x2,y2), (255,255,0), 2)

    # Violations
    for x1,y1,x2,y2,cls in violations:
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(disp, cls, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Integrated System: Zones + PPE", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
worker.stop()
cv2.destroyAllWindows()