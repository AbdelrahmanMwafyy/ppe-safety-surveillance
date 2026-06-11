"""
aruco_test.py — ArUco + Full PPE Detection Test
Tests ArUco reading alongside person/helmet/vest/fire detection.

Shows:
  - Person boxes with role labels
  - Helmet / vest violations
  - Zone overlays
  - ArUco marker ID when detected
  - Pixel size of marker (GOOD / TOO SMALL)
  - Tracking: ID persists after ArUco disappears

Press Q to quit, S to save screenshot.
"""

import cv2
import numpy as np
import threading
import queue
import time
import json
import os
import torch
from collections import deque, Counter
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TAPO_IP     = "172.20.10.4"
TAPO_USER   = "body200003%40gmail.com"
TAPO_PASS   = "Vswaa2003"
TAPO_STREAM = 1

RTSP_URL   = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream{TAPO_STREAM}"
USE_CUDA   = True
FRAME_SIZE = (1280, 720)
INFER_SIZE = (640, 360)

# ArUco
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Model paths
MODEL_DIR  = r"C:\Mwafy\uni\GP\Graduation project\AI_Master_System"
MODEL_PERSON  = os.path.join(MODEL_DIR, "best (8).pt")
MODEL_HELMET  = os.path.join(MODEL_DIR, "best (3and7).pt")
MODEL_VEST    = os.path.join(MODEL_DIR, "best (3and7and9).pt")
MODEL_FIRE    = os.path.join(MODEL_DIR, "best_fire.pt")

# Zones file
ZONES_FILE = os.path.join(MODEL_DIR, "three_zones.json")

# Person registry — ArUco ID → name/role
PERSON_REGISTRY = {
    0: {"name": "Hazem",   "role": "Worker"},
    1: {"name": "Ahmed",   "role": "Worker"},
    2: {"name": "Amr",     "role": "Engineer"},
    3: {"name": "Youssef", "role": "Visitor"},
}

CONF_PERSON = 0.45
CONF_HELMET = 0.25
CONF_VEST   = 0.25
CONF_FIRE   = 0.40

SCALE_X = FRAME_SIZE[0] / INFER_SIZE[0]
SCALE_Y = FRAME_SIZE[1] / INFER_SIZE[1]


# ─────────────────────────────────────────────
#  LOAD ZONES
# ─────────────────────────────────────────────
active_zones = []
if os.path.exists(ZONES_FILE):
    with open(ZONES_FILE) as f:
        zone_data = json.load(f)
    # Use 'moving' zones for this test
    for z in zone_data.get("moving", []):
        if z.get("points"):
            active_zones.append({
                "name":   z.get("name", "zone"),
                "points": np.array(z["points"], dtype=np.int32),
                "color":  tuple(z.get("color", [0, 255, 0])),
            })
    print(f"Loaded {len(active_zones)} zones")


# ─────────────────────────────────────────────
#  THREADED VIDEO STREAM
# ─────────────────────────────────────────────
class VideoStream:
    _ENV_SET = False

    def __init__(self, src):
        if not VideoStream._ENV_SET:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay"
            )
            VideoStream._ENV_SET = True
        self.cap   = None
        self.ret   = False
        self.frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._open(src)
        threading.Thread(target=self._reader, daemon=True).start()
        deadline = time.time() + 8
        while time.time() < deadline:
            with self._lock:
                if self.ret: return
            time.sleep(0.1)
        raise Exception("No frames received after 8s")

    def _open(self, src):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Stream connected.")

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret: time.sleep(0.5); continue
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self._stop.set()
        if self.cap: self.cap.release()


# ─────────────────────────────────────────────
#  INFERENCE WORKER
# ─────────────────────────────────────────────
class InferenceWorker:
    def __init__(self, models, device):
        self.models  = models   # dict: {name: (model, conf)}
        self.device  = device
        self._in_q   = queue.Queue(maxsize=1)
        self._out_q  = queue.Queue(maxsize=1)
        self._stop   = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        use_half = (self.device != "cpu")
        while not self._stop.is_set():
            try: frame = self._in_q.get(timeout=0.5)
            except queue.Empty: continue
            try:
                small   = cv2.resize(frame, INFER_SIZE)
                results = {}
                for name, (model, conf) in self.models.items():
                    results[name] = model(small, conf=conf, iou=0.45,
                                          device=self.device, verbose=False,
                                          half=use_half)
                if self._out_q.full():
                    try: self._out_q.get_nowait()
                    except queue.Empty: pass
                self._out_q.put((frame, results))
            except Exception as e:
                print(f"[INFERENCE ERROR] {e}")

    def submit(self, frame):
        if self._in_q.full():
            try: self._in_q.get_nowait()
            except queue.Empty: pass
        self._in_q.put(frame)

    def get_result(self):
        try: return self._out_q.get_nowait()
        except queue.Empty: return None

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────
#  ARUCO DETECTOR
# ─────────────────────────────────────────────
class ArucoDetector:
    def __init__(self):
        aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        params       = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin    = 3
        params.adaptiveThreshWinSizeMax    = 53
        params.adaptiveThreshWinSizeStep   = 4
        params.minMarkerPerimeterRate      = 0.02
        params.polygonalApproxAccuracyRate = 0.05
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def detect(self, frame):
        """Returns list of (marker_id, center_x, center_y, pixel_size)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        results = []
        if ids is not None:
            for corner, mid in zip(corners, ids.flatten()):
                pts    = corner[0].astype(int)
                cx     = int(pts[:, 0].mean())
                cy     = int(pts[:, 1].mean())
                px_size = int(np.linalg.norm(pts[0] - pts[1]))
                results.append((int(mid), cx, cy, px_size, pts))
        return results


# ─────────────────────────────────────────────
#  PERSON STATE TRACKER
# ─────────────────────────────────────────────
class PersonTracker:
    def __init__(self):
        self.states = []
        self.memory_frames  = 60
        self.vote_threshold = 3
        self.vote_window    = 10
        self.role_history   = 20

    def _scale(self, coords):
        x1, y1, x2, y2 = coords
        return (int(x1*SCALE_X), int(y1*SCALE_Y),
                int(x2*SCALE_X), int(y2*SCALE_Y))

    def _overlap(self, a, b):
        return max(a[0],b[0]) < min(a[2],b[2]) and max(a[1],b[1]) < min(a[3],b[3])

    def update(self, new_boxes, aruco_detections,
               helmet_boxes, yellow_vests, red_vests, blue_vests):

        # Match persons
        matched, used = [], set()
        for box in new_boxes:
            px, py = (box[0]+box[2])/2, (box[1]+box[3])/2
            best_i, best_d = None, 300
            for i, s in enumerate(self.states):
                if i in used: continue
                ox = (s['box'][0]+s['box'][2])/2
                oy = (s['box'][1]+s['box'][3])/2
                d  = np.hypot(px-ox, py-oy)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                used.add(best_i)
                s = self.states[best_i]
                s['box'] = tuple(int(0.4*o+0.6*n) for o,n in zip(s['box'], box))
                matched.append(s)
            else:
                matched.append({
                    'box':           box,
                    'aruco_id':      None,
                    'name':          None,
                    'role':          None,
                    'role_history':  deque(maxlen=self.role_history),
                    'helmet_memory': 0, 'vest_memory': 0,
                    'helmet_vote':   0, 'vest_vote':   0,
                })
        self.states = matched

        # Assign ArUco IDs — match marker center to nearest person
        for (mid, cx, cy, px_size, pts) in aruco_detections:
            best_i, best_d = None, 200
            for i, s in enumerate(self.states):
                bx = (s['box'][0]+s['box'][2])/2
                by = (s['box'][1]+s['box'][3])/2
                d  = np.hypot(cx-bx, cy-by)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                info = PERSON_REGISTRY.get(mid, {"name": f"ID{mid}", "role": None})
                self.states[best_i]['aruco_id'] = mid
                self.states[best_i]['name']     = info['name']
                # Override role from registry if available
                if info['role']:
                    self.states[best_i]['role'] = info['role']

        # PPE checks
        for i, s in enumerate(self.states):
            x1, y1, x2, y2 = s['box']
            h, w = y2-y1, x2-x1
            head  = (x1, y1, x2, y1+int(h*0.25))
            torso = (x1, y1+int(h*0.15), x2, y1+int(h*0.80))

            has_helmet = any(self._overlap(head, hb) for hb in helmet_boxes)
            has_yellow = any(self._overlap(torso, v)  for v  in yellow_vests)
            has_red    = any(self._overlap(torso, v)  for v  in red_vests)
            has_blue   = any(self._overlap(torso, v)  for v  in blue_vests)
            has_vest   = has_yellow or has_red or has_blue

            # Role from vest color (only if no registry role)
            if s['aruco_id'] is None:
                if has_blue:    raw_role = "Engineer"
                elif has_yellow: raw_role = "Worker"
                elif has_red:   raw_role = "Visitor"
                else:           raw_role = None
                s['role_history'].append(raw_role)
                counts   = Counter(r for r in s['role_history'] if r)
                s['role'] = counts.most_common(1)[0][0] if counts else None

            # Helmet vote
            if not has_helmet:
                s['helmet_vote'] = min(s['helmet_vote']+2, self.vote_window)
            else:
                s['helmet_vote'] = max(s['helmet_vote']-2, 0)
            if   s['helmet_vote'] >= self.vote_threshold: s['helmet_memory'] = self.memory_frames
            elif s['helmet_vote'] == 0:                   s['helmet_memory'] = 0
            elif s['helmet_memory'] > 0:                  s['helmet_memory'] -= 1

            # Vest vote
            if not has_vest:
                s['vest_vote'] = min(s['vest_vote']+2, self.vote_window)
            else:
                s['vest_vote'] = max(s['vest_vote']-2, 0)
            if   s['vest_vote'] >= self.vote_threshold: s['vest_memory'] = self.memory_frames
            elif s['vest_vote'] == 0:                   s['vest_memory'] = 0
            elif s['vest_memory'] > 0:                  s['vest_memory'] -= 1


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    # Device
    device = "cuda:0" if USE_CUDA and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    models = {
        "person": (YOLO(MODEL_PERSON), CONF_PERSON),
        "helmet": (YOLO(MODEL_HELMET), CONF_HELMET),
        "vest":   (YOLO(MODEL_VEST),   CONF_VEST),
        "fire":   (YOLO(MODEL_FIRE),   CONF_FIRE),
    }
    class_names = {k: v[0].names for k, v in models.items()}
    print("Models loaded.")

    # Connect stream
    print(f"Connecting: {RTSP_URL}")
    stream  = VideoStream(RTSP_URL)
    worker  = InferenceWorker(models, device)
    aruco   = ArucoDetector()
    tracker = PersonTracker()

    cv2.namedWindow("ArUco + PPE Detection Test")
    frame_count      = 0
    screenshot_count = 0
    display          = None
    prev_time        = time.time()

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                if display is not None:
                    cv2.imshow("ArUco + PPE Detection Test", display)
                if cv2.waitKey(100) & 0xFF == ord('q'): break
                continue

            frame = cv2.resize(frame, FRAME_SIZE)
            frame_count += 1

            # Submit every 3rd frame for YOLO
            if frame_count % 3 == 0:
                worker.submit(frame)

            # ArUco runs every frame (fast, CPU only)
            aruco_detections = aruco.detect(frame)

            # Get YOLO result if ready
            result = worker.get_result()
            if result is not None:
                inf_frame, results = result

                # Parse persons
                persons = []
                for r in results["person"]:
                    for box in r.boxes:
                        name = class_names["person"][int(box.cls[0])].lower()
                        if "person" in name or "people" in name:
                            c = box.xyxy[0].cpu().numpy().astype(int)
                            persons.append((int(c[0]*SCALE_X), int(c[1]*SCALE_Y),
                                            int(c[2]*SCALE_X), int(c[3]*SCALE_Y)))

                # Parse helmets
                helmet_boxes = []
                for r in results["helmet"]:
                    for box in r.boxes:
                        name = class_names["helmet"][int(box.cls[0])].lower()
                        if "helmet" in name or "hardhat" in name:
                            c = box.xyxy[0].cpu().numpy().astype(int)
                            helmet_boxes.append((int(c[0]*SCALE_X), int(c[1]*SCALE_Y),
                                                 int(c[2]*SCALE_X), int(c[3]*SCALE_Y)))

                # Parse vests
                yellow_vests, red_vests, blue_vests = [], [], []
                for r in results["vest"]:
                    for box in r.boxes:
                        name = class_names["vest"][int(box.cls[0])].lower()
                        c    = box.xyxy[0].cpu().numpy().astype(int)
                        coords = (int(c[0]*SCALE_X), int(c[1]*SCALE_Y),
                                  int(c[2]*SCALE_X), int(c[3]*SCALE_Y))
                        if "yellow" in name or "worker" in name: yellow_vests.append(coords)
                        elif "red"  in name or "visitor" in name: red_vests.append(coords)
                        elif "blue" in name or "engineer" in name: blue_vests.append(coords)

                # Parse fire
                fire_boxes = []
                for r in results["fire"]:
                    for box in r.boxes:
                        name = class_names["fire"][int(box.cls[0])].lower()
                        if "fire" in name or "flame" in name or "smoke" in name:
                            c = box.xyxy[0].cpu().numpy().astype(int)
                            fire_boxes.append((int(c[0]*SCALE_X), int(c[1]*SCALE_Y),
                                               int(c[2]*SCALE_X), int(c[3]*SCALE_Y)))

                # Update tracker
                tracker.update(persons, aruco_detections,
                                helmet_boxes, yellow_vests, red_vests, blue_vests)

                # ── DRAW ────────────────────────────────────────
                out = inf_frame.copy()

                # Zones
                for z in active_zones:
                    cv2.polylines(out, [z['points']], True, z['color'], 2)
                    cv2.putText(out, z['name'],
                                (int(z['points'][0][0])+5, int(z['points'][0][1])+22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, z['color'], 2)

                # Persons
                for s in tracker.states:
                    x1, y1, x2, y2 = s['box']
                    role = s.get('role')
                    name = s.get('name')
                    aid  = s.get('aruco_id')

                    if role == "Engineer":  col = (255, 140, 0)
                    elif role == "Worker":  col = (0, 220, 255)
                    elif role == "Visitor": col = (180, 180, 180)
                    else:                   col = (0, 200, 0)

                    cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)

                    # Label: name if known, else role, else Person
                    label = name if name else (role if role else "Person")
                    if aid is not None:
                        label = f"{label} [ID:{aid}]"
                    cv2.putText(out, label, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                    y_off = y1 + 22
                    if s.get('helmet_memory', 0) > 0:
                        cv2.putText(out, "No Helmet", (x1, y_off),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        y_off += 22
                    if s.get('vest_memory', 0) > 0:
                        cv2.putText(out, "No Vest", (x1, y_off),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Fire boxes
                for fb in fire_boxes:
                    cv2.rectangle(out, (fb[0],fb[1]), (fb[2],fb[3]), (0,0,255), 3)
                    cv2.putText(out, "FIRE!", (fb[0], fb[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                # ArUco detections drawn on top
                for (mid, cx, cy, px_size, pts) in aruco_detections:
                    good = px_size >= 20
                    col  = (0, 255, 0) if good else (0, 0, 255)
                    cv2.polylines(out, [pts], True, col, 3)
                    for pt in pts:
                        cv2.circle(out, tuple(pt), 5, col, -1)
                    info = PERSON_REGISTRY.get(mid, {"name": f"Unknown{mid}"})
                    cv2.putText(out, f"ArUco {mid}: {info['name']}",
                                (pts[0][0], pts[0][1]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    status = f"{px_size}px {'OK' if good else 'TOO SMALL'}"
                    cv2.putText(out, status, (cx-30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                # FPS
                now  = time.time()
                fps  = 1.0 / (now - prev_time + 1e-9)
                prev_time = now
                cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                # ArUco status top-right
                aruco_txt = f"ArUco: {len(aruco_detections)} detected"
                cv2.putText(out, aruco_txt, (FRAME_SIZE[0]-280, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0,255,0) if aruco_detections else (100,100,100), 2)

                # Legend
                legend = [
                    ("Worker  — yellow vest", (0,200,255)),
                    ("Engineer — blue vest",  (255,120,0)),
                    ("Visitor  — red vest",   (180,180,180)),
                    ("Green ArUco = detected OK", (0,255,0)),
                    ("Red ArUco = too small",     (0,0,255)),
                ]
                lx, ly = 10, FRAME_SIZE[1]-120
                for txt, c in legend:
                    cv2.rectangle(out, (lx,ly-10),(lx+12,ly+2), c, -1)
                    cv2.putText(out, txt, (lx+18,ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1)
                    ly += 18

                display = out

            cv2.imshow("ArUco + PPE Detection Test",
                       display if display is not None else frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"aruco_ppe_screenshot_{screenshot_count}.jpg"
                if display is not None:
                    cv2.imwrite(fname, display)
                    print(f"[SAVED] {fname}")
                    screenshot_count += 1

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        worker.stop()
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()