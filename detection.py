import cv2
import json
import numpy as np
import os
import urllib.parse
from zone import ZONES
from ultralytics import YOLO
from threading import Thread

# ===============================
# CAMERA CONFIG
# ===============================


TAPO_IP     = "your_camera_ip"        # e.g. 192.168.1.100
TAPO_USER   = "your_camera_username"  # your Tapo account email
TAPO_PASS   = "your_camera_password"
TAPO_STREAM = 2                        # 1 = main stream, 2 = sub-stream
RTSP_URL = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream{TAPO_STREAM}"

USE_CUDA   = True
FRAME_SIZE = (1280,720)


FRAME_W, FRAME_H = 1280, 720        # display size
INFER_W, INFER_H = 640, 360         # YOLO inference size
SKIP_FRAMES = 1                     # skip frames for speed

# ===============================
# LOAD YOLO MODEL
# ===============================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolov8n.pt")
model = YOLO(model_path)  # YOLOv8 Nano

# ===============================
# LOAD DYNAMIC ZONES
# ===============================
dynamic_zones = []
zones_path = os.path.join(script_dir, "zones.json")
if os.path.exists(zones_path):
    with open(zones_path, "r") as f:
        data = json.load(f)
        for z in data:
            name = z.get("name", "Unnamed Zone")
            pts = z.get("points", [])
            color = tuple(z.get("color", [255, 255, 0]))  # fallback color if not in JSON
            if pts:
                dynamic_zones.append({"name": name, "points": pts, "color": color})

# ===============================
# HELPER FUNCTIONS
# ===============================
def point_in_polygon(point, polygon):
    polygon = [tuple(map(int, p)) for p in polygon]
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

def box_center_in_zone(box, polygon):
    x1, y1, x2, y2 = box
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return point_in_polygon(center, polygon), center

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = (x, y)
        print("\nClicked at:", clicked)

        for name, zone in ZONES.items():
            if point_in_polygon(clicked, zone["points"]):
                print(f"✔ Inside STATIC zone: {name}")
                return

        for z in dynamic_zones:
            if point_in_polygon(clicked, z["points"]):
                print(f"✔ Inside DYNAMIC zone: {z['name']}")
                return

        print("✖ Not inside any zone.")

# ===============================
# THREADED VIDEO CAPTURE
# ===============================
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                self.cap.release()
                self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

vs = VideoStream(RTSP_URL)
cv2.namedWindow("Zone Viewer")
cv2.setMouseCallback("Zone Viewer", mouse_event)

print("=== ZONE VIEWER READY ===")
print("Click anywhere to test zones. Press 'q' to quit.")

frame_count = 0

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = vs.read()
    if not ret:
        continue

    frame_count += 1
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    persons = []

    # Run YOLO every SKIP_FRAMES + 1 frames
    if frame_count % (SKIP_FRAMES + 1) == 0:
        frame_small = cv2.resize(frame, (INFER_W, INFER_H))
        results = model(frame_small, verbose=False)

        scale_x = FRAME_W / INFER_W
        scale_y = FRAME_H / INFER_H

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # PERSON
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    persons.append((x1, y1, x2, y2))

    # Draw persons and check zones
    for box in persons:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        inside_any = False

        for name, zone in ZONES.items():
            inside, center = box_center_in_zone(box, zone["points"])
            if inside:
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                print(f"⚠ PERSON IN STATIC ZONE: {name.upper()}")
                inside_any = True
                break

        if inside_any:
            continue

        for z in dynamic_zones:
            inside, center = box_center_in_zone(box, z["points"])
            if inside:
                cv2.circle(frame, center, 5, (0, 255, 255), -1)
                print(f"⚠ PERSON IN DYNAMIC ZONE: {z['name'].upper()}")
                break

    # ===============================
    # Draw static zones with their color
    # ===============================
    for name, zone in ZONES.items():
        pts = np.array(zone["points"], np.int32)
        color = zone["color"]  # original zone color (red, orange, green)

        for box in persons:
            inside, _ = box_center_in_zone(box, zone["points"])
            if inside:
                color = (0, 0, 255)  # red highlight if occupied
                break

        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, name, (pts[0][0]+5, pts[0][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ===============================
    # Draw dynamic zones with their color
    # ===============================
    for z in dynamic_zones:
        pts = np.array(z["points"], np.int32)
        color = z.get("color", (0, 255, 255))  # fallback if JSON missing

        for box in persons:
            inside, _ = box_center_in_zone(box, z["points"])
            if inside:
                color = (0, 0, 255)  # red highlight if occupied
                break

        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, z["name"], (pts[0][0]+5, pts[0][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Zone Viewer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
