import cv2
from ultralytics import YOLO
import numpy as np
import os
import threading
import queue
import torch

# ─────────────────────────────────────────────
#  TAPO C200 CONFIG
# ─────────────────────────────────────────────

TAPO_IP     = "your_camera_ip"        # e.g. 192.168.1.100
TAPO_USER   = "your_camera_username"  # your Tapo account email
TAPO_PASS   = "your_camera_password"
TAPO_STREAM = 2                        # 1 = main stream, 2 = sub-stream

RTSP_URL = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream{TAPO_STREAM}"

USE_CUDA   = True
FRAME_SIZE = (1280, 720)   # (width, height)


# ─────────────────────────────────────────────
#  THREADED VIDEO STREAM
# ─────────────────────────────────────────────
class VideoStream:
    _ENV_SET = False

    def __init__(self, src, reconnect_delay=3):
        if not VideoStream._ENV_SET:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "stimeout;5000000|"
                "fflags;nobuffer|"
                "flags;low_delay"
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
        raise Exception(
            "Stream opened but no frames received after 8 s.\n"
            "  - Try TAPO_STREAM = 2 (sub-stream)\n"
            "  - Verify IP and credentials")

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
#  INFERENCE WORKER
# ─────────────────────────────────────────────
class InferenceWorker:
    def __init__(self, model, model_vest, device, conf=0.10, conf_vest=0.35, iou=0.40):
        self.model      = model
        self.model_vest = model_vest
        self.device     = device
        self.conf       = conf
        self.conf_vest  = conf_vest
        self.iou        = iou

        self._in_q  = queue.Queue(maxsize=1)
        self._out_q = queue.Queue(maxsize=1)
        self._stop  = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                frame = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            results      = self.model(frame, conf=self.conf, iou=self.iou,
                                      device=self.device, verbose=False)
            results_vest = self.model_vest(frame, conf=self.conf_vest, iou=self.iou,
                                           device=self.device, verbose=False)

            if self._out_q.full():
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
            self._out_q.put((frame, results, results_vest))

    def submit(self, frame):
        if self._in_q.full():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
        self._in_q.put(frame)

    def get_result(self):
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        self._thread.join()


# ─────────────────────────────────────────────
#  PPE DETECTOR
# ─────────────────────────────────────────────
class PPEDetector:
    def __init__(self,
                 model_path      = r"model_path",
                 model_vest_path = r"model_path"):

        print(f"Loading model 1: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model 2: {model_vest_path}")
        if not os.path.exists(model_vest_path):
            raise FileNotFoundError(f"Model not found: {model_vest_path}")

        self.model            = YOLO(model_path)
        self.class_names      = self.model.names

        self.model_vest_color = YOLO(model_vest_path)
        self.class_names_vest = self.model_vest_color.names

        # ── CUDA resolution ──────────────────
        if USE_CUDA and torch.cuda.is_available():
            self.device = "cuda:0"
            print(f"CUDA enabled — GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            if USE_CUDA:
                print("WARNING: CUDA requested but not available — falling back to CPU")
            else:
                print("Running on CPU")

        print(f"Model 1 classes: {list(self.class_names.values())}")
        print(f"Model 2 classes: {list(self.class_names_vest.values())}")

        self.memory_frames  = 30
        self.person_states  = []

        # Voting config
        self.vote_window    = 6
        self.vote_threshold = 2

    # ─────────────────────────────────────────
    def match_persons(self, new_persons):
        matched_states = []
        used = set()

        for person in new_persons:
            px = (person[0] + person[2]) / 2
            py = (person[1] + person[3]) / 2
            best_idx, best_dist = None, 200

            for i, state in enumerate(self.person_states):
                if i in used:
                    continue
                ox = (state['box'][0] + state['box'][2]) / 2
                oy = (state['box'][1] + state['box'][3]) / 2
                dist = np.sqrt((px - ox) ** 2 + (py - oy) ** 2)
                if dist < best_dist:
                    best_dist, best_idx = dist, i

            if best_idx is not None:
                used.add(best_idx)
                state = self.person_states[best_idx]
                state['box'] = tuple(int(0.4 * o + 0.6 * n)
                                     for o, n in zip(state['box'], person))
                matched_states.append(state)
            else:
                matched_states.append({
                    'box':           person,
                    'helmet_memory': 0,
                    'vest_memory':   0,
                    'helmet_vote':   0,
                    'vest_vote':     0,
                    'role':          None,
                })

        self.person_states = matched_states

    # ─────────────────────────────────────────
    def is_overlapping(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        return xA < xB and yA < yB

    # ─────────────────────────────────────────
    def detect_yellow(self, frame, region):
        x1, y1, x2, y2 = region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([20, 100, 100]),
                           np.array([35, 255, 255]))
        return np.sum(mask > 0) / (roi.shape[0] * roi.shape[1]) > 0.15

    # ─────────────────────────────────────────
    def _process_results(self, frame, results, results_vest):
        """Parse detections, update person states."""
        persons, helmet_boxes         = [], []
        vest_boxes                    = []
        black_vest_boxes, blue_vest_boxes = [], []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name   = self.class_names[cls_id]
                coords = tuple(box.xyxy[0].cpu().numpy().astype(int))
                if name == 'Person':
                    persons.append(coords)
                elif name == 'Helmet':
                    helmet_boxes.append(coords)
                elif name == 'Vest':
                    vest_boxes.append(coords)

        for result in results_vest:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name   = self.class_names_vest[cls_id]
                coords = tuple(box.xyxy[0].cpu().numpy().astype(int))
                if name == 'black vest':
                    black_vest_boxes.append(coords)
                elif name == 'blue vest':
                    blue_vest_boxes.append(coords)

        self.match_persons(persons)

        helmet_missing_set = set()
        vest_missing_set   = set()

        for i, state in enumerate(self.person_states):
            x1, y1, x2, y2 = state['box']
            h = y2 - y1

            head_region  = (x1, y1,              x2, y1 + int(h * 0.20))
            torso_region = (x1, y1 + int(h * 0.22), x2, y1 + int(h * 0.65))

            helmet_found    = any(self.is_overlapping(head_region, hb) for hb in helmet_boxes)
            vest_found_black = any(self.is_overlapping(torso_region, bv) for bv in black_vest_boxes)
            vest_found_blue  = any(self.is_overlapping(torso_region, bv) for bv in blue_vest_boxes)
            yellow_found     = self.detect_yellow(frame, torso_region)
            vest_found       = vest_found_black or vest_found_blue or yellow_found

            # Role assignment
            if yellow_found:
                state['role'] = "Worker"
            elif vest_found_blue:
                state['role'] = "Visitor"
            elif vest_found_black:
                state['role'] = "Engineer"
            else:
                state['role'] = None

            if not helmet_found:
                helmet_missing_set.add(i)
            if not vest_found:
                vest_missing_set.add(i)

        # Voting / memory update
        for i, state in enumerate(self.person_states):
            if i in helmet_missing_set:
                state['helmet_vote'] = min(state['helmet_vote'] + 2, self.vote_window)
            else:
                state['helmet_vote'] = max(state['helmet_vote'] - 1, 0)

            if state['helmet_vote'] >= self.vote_threshold:
                state['helmet_memory'] = self.memory_frames
            else:
                state['helmet_memory'] = max(0, state['helmet_memory'] - 1)

            if i in vest_missing_set:
                state['vest_vote'] = min(state['vest_vote'] + 2, self.vote_window)
                if state['vest_vote'] >= self.vote_threshold:
                    state['vest_memory'] = self.memory_frames
            else:
                state['vest_vote']   = 0
                state['vest_memory'] = 0

    # ─────────────────────────────────────────
    def _annotate(self, frame):
        """Draw bounding boxes and labels onto a copy of frame."""
        annotated = frame.copy()

        for state in self.person_states:
            x1, y1, x2, y2 = state['box']
            h = y2 - y1

            if state.get('role'):
                cv2.rectangle(annotated, (x1, y1), (x2, y1 + int(h * 0.18)),
                              (0, 255, 0), 2)
                cv2.putText(annotated, f"Role: {state['role']}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if state.get('helmet_memory', 0) > 0:
                cv2.putText(annotated, "No Helmet",
                            (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if state.get('vest_memory', 0) > 0:
                cv2.putText(annotated, "No Vest",
                            (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated

    # ─────────────────────────────────────────
    def detect(self, source=RTSP_URL):
        print(f"Connecting to: {source}")
        stream  = VideoStream(source)
        worker  = InferenceWorker(self.model, self.model_vest_color, self.device)
        display = None

        print("Stream open. Press Q to quit.")

        frame_count = 0

        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                if display is not None:
                    cv2.imshow("PPE Detection + Roles", display)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                continue

            if FRAME_SIZE:
                frame = cv2.resize(frame, FRAME_SIZE)

            frame_count += 1
            # Submit every other frame to inference (matches original every-2-frames logic)
            if frame_count % 2 == 0:
                worker.submit(frame)

            result = worker.get_result()
            if result is not None:
                inf_frame, results, results_vest = result
                self._process_results(inf_frame, results, results_vest)
                display = self._annotate(inf_frame)

            cv2.imshow("PPE Detection + Roles", display if display is not None else frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        worker.stop()
        stream.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    PPEDetector().detect()