import cv2
import json
import numpy as np
import os
import sys
import threading
import queue
import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CAMERA CONFIG
# ─────────────────────────────────────────────
TAPO_IP     = "your_camera_ip"
TAPO_USER   = "your_camera_username"
TAPO_PASS   = "your_camera_password"
TAPO_STREAM = 1

RTSP_URL   = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream{TAPO_STREAM}"
USE_CUDA   = True
FRAME_SIZE = (1280, 720)
INFER_SIZE = (640, 360)

# ─────────────────────────────────────────────
#  ZONE ACCESS RULES
#  Map zone color (BGR tuple) → roles allowed in that zone.
#  Adjust colors to match whatever colors you defined in zone.py / zones.json.
# ─────────────────────────────────────────────
# Red zone   → Engineer only
# Orange zone → Engineer + Worker
# Green zone  → Engineer + Worker + Visitor
ZONE_ACCESS = {
    "red":    {"color_range": ((0, 0, 150), (80, 80, 255)),   "allowed": {"Engineer"}},
    "orange": {"color_range": ((0, 100, 150), (80, 200, 255)), "allowed": {"Engineer", "Worker"}},
    "green":  {"color_range": ((0, 100, 0), (80, 255, 80)),   "allowed": {"Engineer", "Worker", "Visitor"}},
}

def get_zone_tier(color_bgr):
    """Return tier name ('red'/'orange'/'green') based on the zone's BGR color tuple."""
    b, g, r = color_bgr
    if r > 150 and g < 100 and b < 100:
        return "red"
    if r > 150 and 80 < g < 200 and b < 80:
        return "orange"
    if g > 100 and r < 100 and b < 100:
        return "green"
    return "unknown"  # no access restrictions applied

def is_authorized(role, zone_tier):
    """Return True if role is allowed inside this zone tier."""
    if role is None or zone_tier == "unknown":
        return True   # no role detected → don't spam false violations
    allowed = ZONE_ACCESS.get(zone_tier, {}).get("allowed", set())
    return role in allowed


# ─────────────────────────────────────────────
#  LOAD ZONES
# ─────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

# zone.py and zones.json live in Smart-System/zone/ — add that to path
_zone_dir = os.path.join(script_dir, "Smart-System", "zone")
if _zone_dir not in sys.path:
    sys.path.insert(0, _zone_dir)

# Static zones from zone.py  (format: {name: {points:[...], color:(B,G,R)}})
try:
    from zone import ZONES as STATIC_ZONES
    print(f"Loaded {len(STATIC_ZONES)} static zone(s) from zone.py")
except ImportError:
    print("WARNING: zone.py not found — no static zones loaded.")
    STATIC_ZONES = {}

# Dynamic zones from zones.json — search Smart-System/zone/ first
dynamic_zones = []
for _candidate in [os.path.join(_zone_dir, "zones.json"),
                   os.path.join(script_dir, "zones.json")]:
    if os.path.exists(_candidate):
        zones_path = _candidate
        break
else:
    zones_path = None

if zones_path:
    print(f"Loading dynamic zones from: {zones_path}")
    with open(zones_path, "r") as f:
        for z in json.load(f):
            pts   = z.get("points", [])
            color = tuple(z.get("color", [0, 255, 255]))
            if pts:
                dynamic_zones.append({
                    "name":   z.get("name", "Unnamed"),
                    "points": pts,
                    "color":  color,
                })
    print(f"Loaded {len(dynamic_zones)} dynamic zone(s)")
else:
    print("WARNING: zones.json not found — no dynamic zones loaded.")


# ─────────────────────────────────────────────
#  ZONE GEOMETRY HELPERS
# ─────────────────────────────────────────────
def point_in_polygon(point, polygon):
    polygon = [tuple(map(int, p)) for p in polygon]
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def box_center_in_zone(box, polygon):
    x1, y1, x2, y2 = box
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return point_in_polygon(center, polygon), center

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, zone in STATIC_ZONES.items():
            if point_in_polygon((x, y), zone["points"]):
                print(f"[CLICK] Inside static zone: {name}")
                return
        for z in dynamic_zones:
            if point_in_polygon((x, y), z["points"]):
                print(f"[CLICK] Inside dynamic zone: {z['name']}")
                return
        print(f"[CLICK] ({x},{y}) — not inside any zone.")


# ─────────────────────────────────────────────
#  THREADED VIDEO STREAM  (TCP RTSP, auto-reconnect)
# ─────────────────────────────────────────────
class VideoStream:
    _ENV_SET = False

    def __init__(self, src, reconnect_delay=3):
        if not VideoStream._ENV_SET:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|flags;low_delay"
            )
            VideoStream._ENV_SET = True

        self.src             = src
        self.reconnect_delay = reconnect_delay
        self.cap             = None
        self.ret             = False
        self.frame           = None
        self._lock           = threading.Lock()
        self._stop           = threading.Event()

        self._open_cap()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        import time
        deadline = time.time() + 8
        while time.time() < deadline:
            with self._lock:
                if self.ret and self.frame is not None:
                    return
            time.sleep(0.1)
        raise Exception("Stream opened but no frames received after 8 s.")

    def _open_cap(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open stream: {self.src}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Stream connected (TCP).")

    def _reader(self):
        import time
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print(f"Frame read failed — reconnecting in {self.reconnect_delay} s…")
                time.sleep(self.reconnect_delay)
                try:
                    self._open_cap()
                except Exception as e:
                    print(f"Reconnect failed: {e}")
                continue
            with self._lock:
                self.ret   = ret
                self.frame = frame

    def read(self):
        with self._lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self._stop.set()
        self._thread.join()
        if self.cap:
            self.cap.release()


# ─────────────────────────────────────────────
#  INFERENCE WORKER  (both PPE models in background)
# ─────────────────────────────────────────────
class InferenceWorker:
    def __init__(self, model, model_vest, device, conf=0.10, conf_vest=0.35, iou=0.40):
        self.model      = model
        self.model_vest = model_vest
        self.device     = device
        self.conf       = conf
        self.conf_vest  = conf_vest
        self.iou        = iou
        self._in_q      = queue.Queue(maxsize=1)
        self._out_q     = queue.Queue(maxsize=1)
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                frame = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            small        = cv2.resize(frame, INFER_SIZE)
            results      = self.model(small, conf=self.conf, iou=self.iou,
                                      device=self.device, verbose=False)
            results_vest = self.model_vest(small, conf=self.conf_vest, iou=self.iou,
                                           device=self.device, verbose=False)
            if self._out_q.full():
                try: self._out_q.get_nowait()
                except queue.Empty: pass
            self._out_q.put((frame, results, results_vest))

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
        self._thread.join()


# ─────────────────────────────────────────────
#  MAIN DETECTOR
# ─────────────────────────────────────────────
class PPEZoneDetector:
    def __init__(self,
                 model_path      = r"C:\Mwafy\uni\GP\Smart-System\ppe\yolov11safetyhelmet (2)\yolov11safetyhelmet\best (3).pt",
                 model_vest_path = r"C:\Mwafy\uni\GP\Smart-System\ppe\yolov11safetyhelmet (2)\yolov11safetyhelmet\best (7).pt"):

        for p in (model_path, model_vest_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Model not found: {p}")

        print(f"Loading PPE model:  {model_path}")
        self.model            = YOLO(model_path)
        self.class_names      = self.model.names

        print(f"Loading vest model: {model_vest_path}")
        self.model_vest       = YOLO(model_vest_path)
        self.class_names_vest = self.model_vest.names

        if USE_CUDA and torch.cuda.is_available():
            self.device = "cuda:0"
            print(f"CUDA — GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("Running on CPU")

        # Person tracking state
        self.person_states  = []
        self.memory_frames  = 30
        self.vote_window    = 6
        self.vote_threshold = 2

        # Scale factors (inference → display)
        self.scale_x = FRAME_SIZE[0] / INFER_SIZE[0]
        self.scale_y = FRAME_SIZE[1] / INFER_SIZE[1]

    # ── Person tracking ─────────────────────
    def _match_persons(self, new_boxes):
        matched, used = [], set()
        for box in new_boxes:
            px = (box[0] + box[2]) / 2
            py = (box[1] + box[3]) / 2
            best_idx, best_dist = None, 200

            for i, s in enumerate(self.person_states):
                if i in used: continue
                ox = (s['box'][0] + s['box'][2]) / 2
                oy = (s['box'][1] + s['box'][3]) / 2
                d  = np.hypot(px - ox, py - oy)
                if d < best_dist:
                    best_dist, best_idx = d, i

            if best_idx is not None:
                used.add(best_idx)
                s = self.person_states[best_idx]
                s['box'] = tuple(int(0.4*o + 0.6*n) for o,n in zip(s['box'], box))
                matched.append(s)
            else:
                matched.append({
                    'box': box, 'role': None,
                    'helmet_memory': 0, 'vest_memory': 0,
                    'helmet_vote': 0,   'vest_vote': 0,
                })
        self.person_states = matched

    def _is_overlapping(self, a, b):
        return max(a[0],b[0]) < min(a[2],b[2]) and max(a[1],b[1]) < min(a[3],b[3])

    def _detect_yellow(self, frame, region):
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return False
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([20,100,100]), np.array([35,255,255]))
        return np.sum(mask > 0) / (roi.shape[0] * roi.shape[1]) > 0.15

    # ── Scale coords from inference → display ─
    def _scale(self, coords):
        x1, y1, x2, y2 = coords
        return (int(x1*self.scale_x), int(y1*self.scale_y),
                int(x2*self.scale_x), int(y2*self.scale_y))

    # ── Process one inference result ─────────
    def _process_results(self, frame, results, results_vest):
        persons, helmet_boxes, black_vest_boxes, blue_vest_boxes = [], [], [], []

        for r in results:
            for box in r.boxes:
                name   = self.class_names[int(box.cls[0])]
                coords = self._scale(tuple(box.xyxy[0].cpu().numpy().astype(int)))
                if   name == 'Person': persons.append(coords)
                elif name == 'Helmet': helmet_boxes.append(coords)

        for r in results_vest:
            for box in r.boxes:
                name   = self.class_names_vest[int(box.cls[0])]
                coords = self._scale(tuple(box.xyxy[0].cpu().numpy().astype(int)))
                if   name == 'black vest': black_vest_boxes.append(coords)
                elif name == 'blue vest':  blue_vest_boxes.append(coords)

        self._match_persons(persons)

        helmet_miss, vest_miss = set(), set()

        for i, s in enumerate(self.person_states):
            x1, y1, x2, y2 = s['box']
            h = y2 - y1
            head  = (x1, y1, x2, y1 + int(h * 0.20))
            torso = (x1, y1 + int(h * 0.22), x2, y1 + int(h * 0.65))

            helmet_ok     = any(self._is_overlapping(head, hb) for hb in helmet_boxes)
            black_ok      = any(self._is_overlapping(torso, bv) for bv in black_vest_boxes)
            blue_ok       = any(self._is_overlapping(torso, bv) for bv in blue_vest_boxes)
            yellow_ok     = self._detect_yellow(frame, torso)
            vest_ok       = black_ok or blue_ok or yellow_ok

            # Role from vest color
            if yellow_ok:     s['role'] = "Worker"
            elif blue_ok:     s['role'] = "Visitor"
            elif black_ok:    s['role'] = "Engineer"
            else:             s['role'] = None

            if not helmet_ok: helmet_miss.add(i)
            if not vest_ok:   vest_miss.add(i)

        # Voting / memory
        for i, s in enumerate(self.person_states):
            s['helmet_vote'] = min(s['helmet_vote'] + 2, self.vote_window) if i in helmet_miss \
                               else max(s['helmet_vote'] - 1, 0)
            s['helmet_memory'] = self.memory_frames if s['helmet_vote'] >= self.vote_threshold \
                                  else max(0, s['helmet_memory'] - 1)

            if i in vest_miss:
                s['vest_vote']   = min(s['vest_vote'] + 2, self.vote_window)
                if s['vest_vote'] >= self.vote_threshold:
                    s['vest_memory'] = self.memory_frames
            else:
                s['vest_vote']   = 0
                s['vest_memory'] = 0

    # ── Draw zones + persons + violations ────
    def _annotate(self, frame):
        out = frame.copy()

        all_zones = (
            list(STATIC_ZONES.items()) +
            [(z["name"], z) for z in dynamic_zones]
        )

        # ── Draw persons + PPE labels first ──
        for s in self.person_states:
            x1, y1, x2, y2 = s['box']
            role = s.get('role')

            box_color = (0, 255, 0) if role else (160, 160, 160)
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

            if role:
                cv2.putText(out, f"Role: {role}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            y_off = y1 + 22
            if s.get('helmet_memory', 0) > 0:
                cv2.putText(out, "No Helmet", (x1, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_off += 22
            if s.get('vest_memory', 0) > 0:
                cv2.putText(out, "No Vest", (x1, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_off += 22

            # Zone authorization dots + warnings
            for zone_name, zone in all_zones:
                inside, center = box_center_in_zone(s['box'], zone["points"])
                if not inside:
                    continue

                tier = get_zone_tier(zone["color"])
                if not is_authorized(role, tier):
                    # Red dot on center
                    cv2.circle(out, center, 7, (0, 0, 255), -1)

                    # Warning banner on person box
                    msg = f"UNAUTHORIZED: {role or 'Unknown'} in {zone_name}"
                    cv2.putText(out, msg, (x1, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    y_off += 22

                    print(f"[VIOLATION] {msg}")
                else:
                    # Green dot — authorized
                    cv2.circle(out, center, 5, (0, 255, 0), -1)

        # ── Draw static zones on top — same as detect.py ─────────────────
        for name, zone in STATIC_ZONES.items():
            pts   = np.array(zone["points"], np.int32)
            color = zone["color"]   # reset to zone's own color each iteration

            for s in self.person_states:
                inside, _ = box_center_in_zone(s['box'], zone["points"])
                if inside:
                    color = (0, 0, 255)
                    break

            cv2.polylines(out, [pts], True, color, 2)
            cv2.putText(out, name, (pts[0][0] + 5, pts[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ── Draw dynamic zones on top — same as detect.py ────────────────
        for z in dynamic_zones:
            pts   = np.array(z["points"], np.int32)
            color = z.get("color", (0, 255, 255))   # reset per-zone

            for s in self.person_states:
                inside, _ = box_center_in_zone(s['box'], z["points"])
                if inside:
                    color = (0, 0, 255)
                    break

            cv2.polylines(out, [pts], True, color, 2)
            cv2.putText(out, z["name"], (pts[0][0] + 5, pts[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ── Legend ───────────────────────────
        legend = [
            ("Worker   — yellow vest  (orange + green zones)", (0, 200, 255)),
            ("Engineer — blue vest    (all zones)",             (255, 120, 0)),
            ("Visitor  — black vest   (green zone only)",       (180, 180, 180)),
            ("UNAUTHORIZED",                                     (0, 0, 255)),
        ]
        lx, ly = 10, 20
        for txt, col in legend:
            cv2.rectangle(out, (lx, ly - 12), (lx + 14, ly + 2), col, -1)
            cv2.putText(out, txt, (lx + 20, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            ly += 20

        return out

    # ── Main loop ────────────────────────────
    def run(self, source=RTSP_URL):
        print(f"Connecting: {source}")
        stream  = VideoStream(source)
        worker  = InferenceWorker(self.model, self.model_vest, self.device)
        display = None

        cv2.namedWindow("PPE + Zone Detector")
        cv2.setMouseCallback("PPE + Zone Detector", mouse_callback)
        print("Running. Press Q to quit.")

        frame_count = 0
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                if display is not None:
                    cv2.imshow("PPE + Zone Detector", display)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                continue

            frame = cv2.resize(frame, FRAME_SIZE)
            frame_count += 1

            if frame_count % 2 == 0:
                worker.submit(frame)

            result = worker.get_result()
            if result is not None:
                inf_frame, results, results_vest = result
                self._process_results(inf_frame, results, results_vest)
                display = self._annotate(inf_frame)

            cv2.imshow("PPE + Zone Detector", display if display is not None else frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        worker.stop()
        stream.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    PPEZoneDetector().run()