import asyncio
import cv2
import json
import numpy as np
import os
import socket
import threading
import queue
import time
import torch
from collections import deque, Counter
from ultralytics import YOLO
from onvif import ONVIFCamera
from bleak import BleakClient
from database import SafetyDB
from datetime import datetime

# ─────────────────────────────────────────────
#  CAMERA CONFIG
# ─────────────────────────────────────────────
TAPO_IP     = "172.20.10.4"
CAM_PORT    = 2020
TAPO_USER   = "Mwafyy"
TAPO_PASS   = "Vswaa2003"
TAPO_STREAM = 1
RTSP_URL    = f"rtsp://Mwafyy:{TAPO_PASS}@{TAPO_IP}:554/stream{TAPO_STREAM}"

USE_CUDA    = True
FRAME_SIZE  = (1280, 720)
INFER_SIZE  = (640, 640)

CONF_HELMET = 0.25
CONF_VEST   = 0.25      

# ─────────────────────────────────────────────
#  ESP32 CONFIG
# ─────────────────────────────────────────────
ESP_IP   = "172.20.10.6"   # ← update to serial monitor output
ESP_PORT = 8080

# ─────────────────────────────────────────────
#  TILT CONFIG
# ─────────────────────────────────────────────
TILT_RIGHT    = 0.25
TILT_LEFT     = -0.25
TILT_HOME     =  0.0
TILT_SPEED    =  0.15
DWELL_TIME    = 10.0
REACH_TOL     =  0.05
REACH_TIMEOUT = 15.0

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
_BASE_DIR   = r"C:\Mwafy\uni\GP\Smart-System\Full integration"
_ZONES_FILE = os.path.join(_BASE_DIR, "three_zones.json")

# ─────────────────────────────────────────────
#  ARUCO + PERSON REGISTRY
# ─────────────────────────────────────────────
ARUCO_DICT_ID   = cv2.aruco.DICT_4X4_50
ARUCO_KEEP_FRAMES = 10        # frames to keep a state alive after YOLO loses person (~0.3s)

PERSON_REGISTRY = {
    0: {"name": "Mohamed"},
    1: {"name": "Ahmed"},
    2: {"name": "Khaled"},
    3: {"name": "Youssef"},
}  

SCREENSHOT_DIR = r"C:\Mwafy\uni\GP\Graduation project\AI_Master_System\screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  BLE LASER CONFIG & PAYLOADS
# ─────────────────────────────────────────────
DEVICE_MAC   = "95:2E:99:7A:2B:90"
WRITE_UUID   = "0000ff02-0000-1000-8000-00805f9b34fb"
NOTIFY_UUID  = "0000ff01-0000-1000-8000-00805f9b34fb"
PACKET_GAP   = 0.02
READY_SIGNAL = bytes.fromhex("010200e4e5e6e7")

CONFIG_A = (1.755, [
    bytes.fromhex("c0c1c2c3040001ffff0000ff00faffffffff6400"),
    bytes.fromhex("80000000000000000480000000008000000000ff"),
    bytes.fromhex("ffffffffffffff80000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("c4c5c6c7"),
])
CONFIG_B = (1.0, [
    bytes.fromhex("c0c1c2c3040001ffff0000ff00fa01ffffff6400"),
    bytes.fromhex("80000000000000000480000000008000000000ff"),
    bytes.fromhex("ffffffffffffff80000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("c4c5c6c7"),
])
CONFIG_A_FIRE = (1.755, [
    bytes.fromhex("c0c1c2c3040001fffffa00ff00faffffffff6400"),
    bytes.fromhex("80000000000000000480000000008000000000ff"),
    bytes.fromhex("ffffffffffffff80000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("c4c5c6c7"),
])
CONFIG_A_WIPE = (1.755, [
    bytes.fromhex("c0c1c2c3080001ffff0000ff00faffffffff6400"),
    bytes.fromhex("80000000000000000480000000008000000000ff"),
    bytes.fromhex("ffffffffffffff80000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("c4c5c6c7"),
])
CONFIG_B_WIPE = (1.0, [
    bytes.fromhex("c0c1c2c3080001ffff0000ff00faffffffff6400"),
    bytes.fromhex("80000000000000000480000000008000000000ff"),
    bytes.fromhex("ffffffffffffff80000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("0000000000000000000000000000000000000000"),
    bytes.fromhex("c4c5c6c7"),
])

LASER_PAYLOADS = {
    "HELMET": (1.5, [
        bytes.fromhex("a0a1a2a300270f82a8002e0282a880331382a800"),
        bytes.fromhex("020282680002138269002d028267803313822100"),
        bytes.fromhex("2c028253002c2182528032218222803223825300"),
        bytes.fromhex("02028220000223820c002d02820b80323181e480"),
        bytes.fromhex("323381d080320281c4002041819b803141817200"),
        bytes.fromhex("204181678030438120002c028153002c51815280"),
        bytes.fromhex("3251812180325381520002028120000253810c00"),
        bytes.fromhex("2c0280db002c6380f3002c0280f3803363809f00"),
        bytes.fromhex("0000803b0000000029000000008d00000000f100"),
        bytes.fromhex("0000015500000001b9000000021d000000028100"),
        bytes.fromhex("0000010f55473c7e474564646464646464646406"),
        bytes.fromhex("0603050604010101010101010101000000000000"),
        bytes.fromhex("0000000000020304060102030405060708090a0b"),
        bytes.fromhex("0a0a0a090032a4a5a6a7"),
    ]),
    "NOENTRY": (1.5, [
        bytes.fromhex("a0a1a2a3003b1183208032028322002a1182d680"),
        bytes.fromhex("301182d8002c138294002f0282ab00282082b800"),
        bytes.fromhex("1d2082c200082082c2800f2082b980232082a980"),
        bytes.fromhex("30208293803620827d803220826a802520826080"),
        bytes.fromhex("0b20826400122082710023208281002c20829400"),
        bytes.fromhex("2f23821a00000081a1002c0281d4002c3181d380"),
        bytes.fromhex("323181a280323381d300020281a1000233818b80"),
        bytes.fromhex("3202818d002a4181428030418143002c43812e00"),
        bytes.fromhex("2c0280fd002c538115002c02811580335380e980"),
        bytes.fromhex("330280e9002c6180c5002b6080bc00276080b400"),
        bytes.fromhex("206080b300156080b700056080cc80026080e980"),
        bytes.fromhex("026380da80020280b5803463809f002d02808080"),
        bytes.fromhex("0373808080340280808003708062002d73802600"),
        bytes.fromhex("0000003e00000000a20000000106000000016a00"),
        bytes.fromhex("000001ce0000000232000000029600000002fa00"),
        bytes.fromhex("000001115f7778475f454a516464646464646464"),
        bytes.fromhex("64040f010604040b050101010101010101010000"),
        bytes.fromhex("0000000000000000010202040506080102030405"),
        bytes.fromhex("060708090a0a0a0b0a0a0a090032a4a5a6a7"),
    ]),
    "VEST": (1.5, [
        bytes.fromhex("a0a1a2a300230d8252002c02822c803011820500"),
        bytes.fromhex("2c1381bf002c0281f1002c2181f080322181bf80"),
        bytes.fromhex("322381f000020281be0002238178001f02818600"),
        bytes.fromhex("2d308197002e3081a300223081a2001230819700"),
        bytes.fromhex("0630817f800c308177801d30817b802c30818780"),
        bytes.fromhex("3530819980353081a5802a3081aa802133816300"),
        bytes.fromhex("2c028132002c43814a002c02814a80334380f600"),
        bytes.fromhex("00008092000000802e0000000036000000009a00"),
        bytes.fromhex("000000fe000000016200000001c6000000022a00"),
        bytes.fromhex("0000010d6147484564646464646464646403060d"),
        bytes.fromhex("0401010101010101010100000000000000000000"),
        bytes.fromhex("0102040102030405060708090a0a0a090032a4a5"),
        bytes.fromhex("a6a7"),
    ]),
    "FIRE": (1.5, [
        bytes.fromhex("a0a1a2a300280d81f9002c028224002c11822480"),
        bytes.fromhex("3113822400020281f900021381e5002c0281e500"),
        bytes.fromhex("2c2381e400120281e480332381d000110281d080"),
        bytes.fromhex("343381d080020281c4000e3081b800123381a480"),
        bytes.fromhex("0d028160800d418163000240816c000d40817c00"),
        bytes.fromhex("1440818b0013408198000d4081a000034081a480"),
        bytes.fromhex("0c4081a4801a4081a08025408198802f40818e80"),
        bytes.fromhex("3540818180364081738034408169802d40816480"),
        bytes.fromhex("2543812400000080c0000000805c000000000800"),
        bytes.fromhex("0000006c00000000d00000000134000000019800"),
        bytes.fromhex("000001fc000000010d3f142c5864646464646464"),
        bytes.fromhex("6464050405110101010101010101010000000000"),
        bytes.fromhex("00000000000002040102030405060708090a0b0a"),
        bytes.fromhex("090032a4a5a6a7"),
    ]),
    "SAFE": (1.5, [
        bytes.fromhex("f0f1f2f300000000000000000000000000006400"),
        bytes.fromhex("000200130045020013004573f4f5f6f7"),
    ]),
}


# ─────────────────────────────────────────────
#  ZONE ACCESS RULES
# ─────────────────────────────────────────────
ZONE_ACCESS = {
    "danger": {"allowed": {"Engineer"}},
    "orange": {"allowed": {"Engineer", "Worker"}},
    "safe":   {"allowed": {"Engineer", "Worker", "Visitor"}},
}
LASER_PRIORITY = {
    'FIRE':    5,
    'HELMET':  4,
    'NOENTRY': 3,
    'VEST':    2,
    'SAFE':    1,
}

def get_zone_tier(zone_name: str) -> str:
    n = zone_name.lower()
    if "danger" in n or "red" in n or "restricted" in n: return "danger"
    if "orange" in n or "warning" in n or "caution" in n: return "orange"
    return "safe"

def is_authorized(role, zone_tier: str) -> bool:
    if role is None: return False
    return role in ZONE_ACCESS.get(zone_tier, {}).get("allowed", set())

def point_in_polygon(point, polygon_pts: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon_pts, (float(point[0]), float(point[1])), False) >= 0


# ─────────────────────────────────────────────
#  BLE LASER WORKER  (persistent — no subprocess)
#  Stays in the same process. Connects once per
#  state change, sends packets, done.
#  No 2.5 s Windows BT release delay needed.
# ─────────────────────────────────────────────
class LaserBLEWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.state_queue   = queue.Queue(maxsize=1)
        self.current_state = "SAFE"
        self._stop_event   = threading.Event()

    def set_state(self, new_state: str):
        if self.state_queue.full():
            try: self.state_queue.get_nowait()
            except queue.Empty: pass
        self.state_queue.put(new_state)

    def stop(self):
        self._stop_event.set()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_loop())

    async def _send_packets(self, client, packets):
        for pkt in packets:
            await client.write_gatt_char(WRITE_UUID, pkt, response=False)
            await asyncio.sleep(PACKET_GAP)

    def _queue_has_new(self):
        """Check if a different state is waiting in queue."""
        if self.state_queue.empty():
            return False
        try:
            next_state = self.state_queue.queue[0]
            return next_state != self.current_state
        except: return False

    async def _interruptible_sleep(self, duration: float) -> bool:
        """
        Sleep for duration seconds but check every 0.1s if a new state arrived.
        Returns True if interrupted (new state waiting), False if completed normally.
        """
        elapsed = 0.0
        step    = 0.1
        while elapsed < duration:
            await asyncio.sleep(min(step, duration - elapsed))
            elapsed += step
            if self._queue_has_new():
                return True   # abort — new state is waiting
        return False

    async def _async_loop(self):
        while not self._stop_event.is_set():

            # ── Try to establish persistent connection ──────────────────
            print("[LASER] Connecting persistently...")
            try:
                async with BleakClient(DEVICE_MAC, timeout=10.0) as client:
                    print("[LASER] ✅ Persistent connection established")

                    ready_event = asyncio.Event()
                    def on_notify(sender, data):
                        if READY_SIGNAL in data:
                            ready_event.set()

                    await client.start_notify(NOTIFY_UUID, on_notify)

                    # Send init handshake once
                    token       = os.urandom(4)
                    init_packet = bytes.fromhex("e0e1e2e3") + token + bytes.fromhex("e4e5e6e7")
                    await client.write_gatt_char(WRITE_UUID, init_packet, response=False)
                    try:
                        await asyncio.wait_for(ready_event.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass

                    # ── Main loop — stay connected, send on state change ─
                    while not self._stop_event.is_set():

                        if not self.state_queue.empty():
                            new_state = self.state_queue.get()
                            # drain queue — always show latest
                            while not self.state_queue.empty():
                                try: new_state = self.state_queue.get_nowait()
                                except queue.Empty: break

                            if new_state != self.current_state:
                                self.current_state = new_state
                                print(f"[LASER] → {new_state}")

                                word_payload = LASER_PAYLOADS.get(new_state)
                                if not word_payload:
                                    await asyncio.sleep(0.1)
                                    continue

                                if new_state == "SAFE":
                                    groups = [CONFIG_A_WIPE, CONFIG_B_WIPE, word_payload]
                                elif new_state == "FIRE":
                                    groups = [CONFIG_A_FIRE, CONFIG_B, word_payload]
                                else:
                                    groups = [CONFIG_A, CONFIG_B, word_payload]

                                # Send packets — interruptible between groups
                                for delay, packets in groups:
                                    interrupted = await self._interruptible_sleep(delay)
                                    if interrupted:
                                        print(f"[LASER] ⚡ Interrupted — new state waiting")
                                        break
                                    await self._send_packets(client, packets)
                                else:
                                    print(f"[LASER] ✅ Displayed: {new_state}")

                        await asyncio.sleep(0.1)

            except Exception as e:
                print(f"[LASER] ❌ Connection lost: {e} — reconnecting in 3s...")
                await asyncio.sleep(3)
                # Loop restarts → reconnects automatically


# ─────────────────────────────────────────────
#  THREADED VIDEO STREAM
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
        deadline = time.time() + 8
        while time.time() < deadline:
            with self._lock:
                if self.ret and self.frame is not None:
                    return
            time.sleep(0.1)
        raise Exception("Stream opened but no frames received after 8 s.")

    def _open_cap(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open stream: {self.src}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Stream connected (TCP).")

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print(f"Frame read failed — reconnecting in {self.reconnect_delay} s…")
                time.sleep(self.reconnect_delay)
                try: self._open_cap()
                except Exception as e: print(f"Reconnect failed: {e}")
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
        if self.cap: self.cap.release()


# ─────────────────────────────────────────────
#  INFERENCE WORKER
# ─────────────────────────────────────────────
class InferenceWorker:
    def __init__(self, model_person, model_helmet, model_vest, model_fire, device,
                 conf_person=0.45, conf_helmet=0.15, conf_vest=0.2, conf_fire=0.99, iou=0.45):
        self.model_person = model_person
        self.model_helmet = model_helmet
        self.model_vest   = model_vest
        self.model_fire   = model_fire
        self.device       = device
        self.conf_person  = conf_person
        self.conf_helmet  = conf_helmet
        self.conf_vest    = conf_vest
        self.conf_fire    = conf_fire
        self.iou          = iou
        self._in_q  = queue.Queue(maxsize=1)
        self._out_q = queue.Queue(maxsize=1)
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        # half=True (fp16) is faster on GPU but crashes on CPU or some model configs
        use_half = (self.device != "cpu")
        while not self._stop.is_set():
            try:
                frame = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                small    = cv2.resize(frame, INFER_SIZE)
                r_person = self.model_person(small, conf=self.conf_person, iou=self.iou,
                                             device=self.device, verbose=False, half=use_half)
                r_helmet = self.model_helmet(small, conf=self.conf_helmet, iou=self.iou,
                                             device=self.device, verbose=False, half=use_half)
                r_vest   = self.model_vest(small, conf=self.conf_vest, iou=self.iou,
                                           device=self.device, verbose=False, half=use_half)
                r_fire   = self.model_fire(small, conf=self.conf_fire, iou=self.iou,
                                           device=self.device, verbose=False, half=use_half)
                if self._out_q.full():
                    try: self._out_q.get_nowait()
                    except queue.Empty: pass
                self._out_q.put((frame, r_person, r_helmet, r_vest, r_fire))
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
        self._thread.join()


# ─────────────────────────────────────────────
#  ARUCO DETECTOR
# ─────────────────────────────────────────────
class ArucoDetector:
    """Runs on every frame — CPU only, no GPU needed, very fast."""

    def __init__(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        params     = cv2.aruco.DetectorParameters()
        # Tune for better detection at distance
        params.adaptiveThreshWinSizeMin    = 3
        params.adaptiveThreshWinSizeMax    = 53
        params.adaptiveThreshWinSizeStep   = 4
        params.minMarkerPerimeterRate      = 0.02
        params.polygonalApproxAccuracyRate = 0.05
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def detect(self, frame):
        """
        Returns list of (marker_id, cx, cy, px_size, corners_pts).
        All coordinates are in frame's pixel space.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        results = []
        if ids is not None:
            for corner, mid in zip(corners, ids.flatten()):
                pts     = corner[0].astype(int)
                cx      = int(pts[:, 0].mean())
                cy      = int(pts[:, 1].mean())
                px_size = int(np.linalg.norm(pts[0] - pts[1]))
                results.append((int(mid), cx, cy, px_size, pts))
        return results


# ─────────────────────────────────────────────
#  HARDWARE MANAGER  (tilt + motor cycle)
# ─────────────────────────────────────────────
class HardwareManager(threading.Thread):
    def __init__(self, detector):
        super().__init__(daemon=True)
        self.detector = detector
        self.stopped  = False
        self._cam     = None
        self._esp     = None

    # ── ONVIF Camera ─────────────────────────
    def _cam_connect(self):
        print("[CAM] Connecting to ONVIF...")
        cam    = ONVIFCamera(TAPO_IP, CAM_PORT, TAPO_USER, TAPO_PASS)
        ptz    = cam.create_ptz_service()
        token  = cam.create_media_service().GetProfiles()[0].token
        home_y = ptz.GetStatus({"ProfileToken": token}).Position.PanTilt.y
        self._cam = {"ptz": ptz, "token": token, "home_y": home_y}
        print(f"[CAM] Connected. home_y={home_y:.3f}")

    def _cam_move(self, target_x):
        c = self._cam
        req = c["ptz"].create_type("AbsoluteMove")
        req.ProfileToken = c["token"]
        req.Position = {"PanTilt": {"x": target_x, "y": c["home_y"]}}
        req.Speed    = {"PanTilt": {"x": TILT_SPEED, "y": 0.0}}
        c["ptz"].AbsoluteMove(req)
        print(f"[CAM] Moving → x={target_x:+.2f}")

    def _cam_wait(self, target_x):
        deadline = time.time() + REACH_TIMEOUT
        while time.time() < deadline:
            try:
                pos = self._cam["ptz"].GetStatus(
                    {"ProfileToken": self._cam["token"]}).Position.PanTilt.x
                if abs(pos - target_x) < REACH_TOL:
                    print(f"[CAM] Reached x={pos:.3f} ✓")
                    return True
            except: pass
            time.sleep(0.5)
        print(f"[CAM] WARNING: Timeout reaching x={target_x:+.2f}")
        return False

    # ── ESP32 TCP ────────────────────────────
    def _esp_connect(self):
        print(f"[ESP] Connecting to {ESP_IP}:{ESP_PORT} ...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect((ESP_IP, ESP_PORT))
        sock.settimeout(0.05)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._esp = sock
        print("[ESP] Connected.")

    def _esp_reconnect(self, retries=5, delay=3):
        """Reconnect after socket failure — fixes cycle stopping after N runs."""
        for attempt in range(1, retries + 1):
            print(f"[ESP] Reconnect attempt {attempt}/{retries}...")
            try:
                if self._esp:
                    try: self._esp.close()
                    except: pass
                self._esp_connect()
                print("[ESP] Reconnected ✓")
                return True
            except Exception as e:
                print(f"[ESP] Reconnect failed: {e}")
                time.sleep(delay)
        print("[ESP] Could not reconnect — stopping.")
        self.stopped = True
        return False

    def _esp_send(self, cmd: str):
        try:
            self._esp.sendall(cmd.encode())
            print(f"[ESP] → '{cmd}'")
        except Exception as e:
            print(f"[ESP] Send error: {e}")
            self._esp_reconnect()

    def _esp_wait_limit(self, current_dir: str):
        """
        Wait for limit switch. While motor is moving:
          - Violation detected → stop motor, wait for clear, restart motor
          - Socket drops      → reconnect, restart motor (no cycle limit)
        """
        print(f"[ESP] Motor moving '{current_dir}' — waiting for limit...")
        while not self.stopped:

            # ── Violation check during movement ───────────────────────
            if self.detector.global_violation_active:
                # Confirm violation persists for 2s before stopping motor (avoids false positives)
                confirm_start = time.time()
                while time.time() - confirm_start < 1.0:
                    if not self.detector.global_violation_active:
                        break
                    time.sleep(0.1)
                else:
                    print("[HOLD] Violation confirmed — stopping motor...")
                    self._esp_send("S")
                if not self.detector.global_violation_active:
                    continue

                while self.detector.global_violation_active and not self.stopped:
                    time.sleep(0.2)
                if self.stopped:
                    return None

                # Confirm clear for 1 s
                clear_start = time.time()
                while time.time() - clear_start < 1.0:
                    if self.detector.global_violation_active:
                        clear_start = time.time()
                    time.sleep(0.1)

                print(f"[RESUME] Restarting motor → '{current_dir}'")
                self._esp_send(current_dir)

            # ── Read from ESP32 ───────────────────────────────────────
            try:
                data = self._esp.recv(64).decode(errors="ignore")
                for ch in data:
                    if ch.upper() in ("R", "L"):
                        print(f"[ESP] Limit hit: '{ch.upper()}'")
                        return ch.upper()

            except socket.timeout:
                pass  # Normal — no data yet

            except Exception as e:
                # Broken pipe / connection reset — was causing N-cycle stop
                print(f"[ESP] Socket error: {e} — reconnecting...")
                if self._esp_reconnect():
                    self._esp_send(current_dir)  # restart motor after reconnect
                else:
                    return None

        return None

    # ── Violation helpers (motor already stopped during dwell/tilt) ──
    def _wait_if_violation(self):
        """Pause sequence while violation active. Motor is already stopped here."""
        if not self.detector.global_violation_active:
            return
        print("[HOLD] Violation active — pausing tilt/dwell sequence...")
        while self.detector.global_violation_active and not self.stopped:
            time.sleep(0.2)
        if self.stopped:
            return
        clear_start = time.time()
        while time.time() - clear_start < 1.0:
            if self.detector.global_violation_active:
                clear_start = time.time()
            time.sleep(0.1)
        print("[RESUME] Violation cleared — continuing.")

    def _interruptible_dwell(self, duration):
        """Dwell for duration seconds, pausing countdown if violation active."""
        deadline = time.time() + duration
        while time.time() < deadline and not self.stopped:
            self._wait_if_violation()
            time.sleep(min(0.2, max(0, deadline - time.time())))

    # ── Main cycle loop ───────────────────────
    def run(self):
        try:
            self._cam_connect()
        except Exception as e:
            print(f"[CAM] ONVIF failed: {e} — continuing without tilt.")

        try:
            self._esp_connect()
        except Exception as e:
            print(f"[ESP] Connection failed: {e}")
            return

        print("[HW] Homing camera to center...")
        self.detector.set_active_zone("moving")
        if self._cam:
            self._cam_move(TILT_HOME)
            self._cam_wait(TILT_HOME)

        current_dir = "R"
        print(f"[HW] Starting motor → {current_dir}")
        self._esp_send(current_dir)

        cycle = 0
        while not self.stopped:
            cycle += 1
            print(f"\n{'─'*50}")
            print(f"[HW CYCLE {cycle}] Motor '{current_dir}' — waiting for limit...")

            # Step 1 — wait for limit (stops motor if violation fires during movement)
            side = self._esp_wait_limit(current_dir)
            if side is None or self.stopped:
                break

            # Step 2 — pause if violation before tilting (motor already stopped at limit)
            self._wait_if_violation()
            if self.stopped:
                break

            # Step 3 — tilt camera + swap zones
            tilt_target = TILT_RIGHT if side == "R" else TILT_LEFT
            zone_key    = "tilted_left" if side == "R" else "tilted_right"
            print(f"[HW] Tilting → {tilt_target:+.2f} | zones → {zone_key}")
            self.detector.camera_moving = False  # stationary — tracking reliable
            self.detector.set_active_zone(zone_key)
            if self._cam:
                self._cam_move(tilt_target)
                self._cam_wait(tilt_target)

            # Step 4 — dwell (pauses countdown if violation fires)
            print(f"[HW] Dwelling {DWELL_TIME:.0f}s at {side} end...")
            self._interruptible_dwell(DWELL_TIME)
            if self.stopped:
                break

            # Step 5 — pause before returning
            self._wait_if_violation()
            if self.stopped:
                break

            # Step 6 — tilt back home + swap zones
            print("[HW] Returning camera to home...")
            self.detector.set_active_zone("moving")
            if self._cam:
                self._cam_move(TILT_HOME)
                self._cam_wait(TILT_HOME)

            # Step 7 — send motor in opposite direction
            current_dir = "L" if side == "R" else "R"
            self.detector.camera_moving = True   # motor moving
            print(f"[HW] Motor → {current_dir}")
            self._esp_send(current_dir)

        # Cleanup
        self._esp_send("S")
        if self._esp:
            try: self._esp.close()
            except: pass
        print("[HW] Stopped.")

#  MAIN DETECTOR
# ─────────────────────────────────────────────
class PPEZoneDetector:
    _MODEL_DIR = r"C:\Mwafy\uni\GP\Graduation project\AI_Master_System"

    _FIRE_MODEL_PATH = r"C:\Mwafy\uni\GP\Graduation project\AI_Master_System\best_fire_detection.engine"

    def __init__(self, model_person_path=None, model_helmet_path=None,
                 model_vest_path=None, model_fire_path=None):
        base = self._MODEL_DIR
        if model_person_path is None: model_person_path = os.path.join(base, "best (8).engine")
        if model_helmet_path is None: model_helmet_path = os.path.join(base, "best (3and7).engine")
        if model_vest_path   is None: model_vest_path   = os.path.join(base, "best (3and7and9).engine")
        if model_fire_path   is None: model_fire_path   = self._FIRE_MODEL_PATH

        self.model_person       = YOLO(model_person_path)
        self.class_names_person = self.model_person.names
        self.model_helmet       = YOLO(model_helmet_path)
        self.class_names_helmet = self.model_helmet.names
        self.model_vest         = YOLO(model_vest_path)
        self.class_names_vest   = self.model_vest.names
        self.model_fire         = YOLO(model_fire_path)
        self.class_names_fire   = self.model_fire.names
        print(f"Fire classes: {list(self.class_names_fire.values())}")

        if USE_CUDA and torch.cuda.is_available():
            self.device = "cuda:0"
            print(f"CUDA — GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"

        # ── Zone sets from three_zones.json ──
        self.all_zones    = {}
        self.active_zones = []
        self._zone_lock   = threading.Lock()

        if os.path.exists(_ZONES_FILE):
            with open(_ZONES_FILE) as f:
                for key, z_list in json.load(f).items():
                    self.all_zones[key] = [
                        {"name":   z.get("name", "zone"),
                         "points": np.array(z.get("points", []), dtype=np.int32),
                         "color":  tuple(z.get("color", [0, 255, 0]))}
                        for z in z_list if z.get("points")
                    ]
            print(f"Loaded zone sets: {list(self.all_zones.keys())}")
        else:
            print(f"WARNING: {_ZONES_FILE} not found.")

        self.set_active_zone('moving')

        self.person_states    = []
        self.memory_frames    = 90
        self.vote_window      = 10
        self.vote_threshold   = 5
        self.role_history_len = 20
        self.scale_x = FRAME_SIZE[0] / INFER_SIZE[0]
        self.scale_y = FRAME_SIZE[1] / INFER_SIZE[1]

        self.global_violation_active = False
        self.current_laser_state     = "SAFE"
        self._pending_laser_state = "SAFE"
        self._pending_laser_count = 0
        # ── Merged BLE laser worker (no subprocess) ──
        self.laser = LaserBLEWorker()
        self.laser.start()

        # ── ArUco detector ──────────────────────────────────────────
        self.aruco         = ArucoDetector()
        self._latest_aruco = []      # updated every frame in run()
        self.camera_moving = True    # False during dwell (stationary)
        self._laser_state_since   = time.time()   # ← add
        self._pending_laser_state = "SAFE"        # ← add
        self._pending_laser_count = 0             # ← add
        # ── Database ────────────────────────────────────────────────
        self.db = SafetyDB()

    def set_active_zone(self, key: str):
        with self._zone_lock:
            if key in self.all_zones:
                self.active_zones = self.all_zones[key]
                print(f"[ZONES] → '{key}'")

    def _trigger_laser(self, state: str):
        """Route detection result directly to BLE worker — no subprocess."""
        self.laser.set_state(state)

    def _assign_aruco(self, aruco_detections):
        """
        Match each detected ArUco marker to a person_state.
        FIX: Uses bounding box containment (not centroid distance) so the
        marker is always assigned to the person whose box contains it.
        Falls back to nearest centroid only if no box contains the marker.
        """
        for (mid, cx, cy, px_size, pts) in aruco_detections:
            if mid not in PERSON_REGISTRY:
                continue
            info   = PERSON_REGISTRY[mid]
            best_i = None

            # Primary: find the person whose bounding box CONTAINS the marker
            MARGIN = 40   # px margin — card can slightly exceed box edge
            for i, s in enumerate(self.person_states):
                if s.get('frames_unseen', 0) > 0:
                    continue  # skip ghost states
                x1, y1, x2, y2 = s['box']
                if (x1 - MARGIN <= cx <= x2 + MARGIN and
                        y1 - MARGIN <= cy <= y2 + MARGIN):
                    best_i = i
                    break   # marker is inside this box — no ambiguity

            # Fallback: nearest active centroid within 300px
            if best_i is None:
                best_dist = 300
                for i, s in enumerate(self.person_states):
                    if s.get('frames_unseen', 0) > 0:
                        continue  # skip ghost states
                    bx = (s['box'][0] + s['box'][2]) / 2
                    by = (s['box'][1] + s['box'][3]) / 2
                    d  = np.hypot(cx - bx, cy - by)
                    if d < best_dist:
                        best_dist, best_i = d, i

            if best_i is not None:
                s = self.person_states[best_i]
                s['aruco_id']   = mid
                s['person_name'] = info['name']

                # Log attendance once per day on very first detection
                if not s.get('attendance_ok', False):
                    try:
                        self.db.log_attendance(mid, info['name'])
                    except Exception as e:
                        print(f'[DB] Attendance error: {e}')
                    s['attendance_ok'] = True

    def _save_screenshot(self, frame, tag: str) -> str:
        """Save frame as JPEG and return its path."""
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'{tag}_{ts}.jpg'
        path     = os.path.join(SCREENSHOT_DIR, filename)
        try:
            cv2.imwrite(path, frame)
        except Exception as e:
            print(f'[SCREENSHOT] Error: {e}')
        return path

    def _log_violations(self, frame):
        """
        Per-person violation start/end logging to DB.
        Called every frame after PPE memory is updated.
        One DB record per violation event (not per frame).
        """
        for s in self.person_states:
            pid   = s.get('aruco_id')
            pname = s.get('person_name')
            pzone = s.get('zone')
            role  = None if s.get('vest_memory', 0) > 0 else s.get('role')
            cam   = 'moving' if self.camera_moving else 'dwell'
            conf  = 'HIGH' if pid is not None else 'LOW'

            # Build set of violations active this frame for this person
            current = set()
            if s.get('helmet_memory', 0) > 0:
                current.add('NO_HELMET')
            if s.get('vest_memory', 0) > 0:
                current.add('NO_VEST')
            if pzone and not is_authorized(role, get_zone_tier(pzone)):
                current.add('UNAUTH_ZONE')

            active = s.setdefault('active_violations', {})

            for vtype in current - set(active):
                try:
                    # Reattach to existing open violation if state was reset
                    existing_vid = self.db.has_active_violation(pid, vtype)
                    if existing_vid:
                        active[vtype] = existing_vid
                    else:
                        path = self._save_screenshot(frame, f'{vtype}_{pname or "unk"}')
                        vid  = self.db.log_violation_start(
                            violation_type  = vtype,
                            person_id       = pid,
                            person_name     = pname,
                            zone_name       = pzone,
                            camera_position = cam,
                            screenshot_path = path,
                            confidence      = conf,
                        )
                        active[vtype] = vid
                except Exception as e:
                    print(f'[DB] Violation start error: {e}')

            # Violations that just ended this frame
            for vtype in set(active) - current:
                try:
                    self.db.log_violation_end(active.pop(vtype))
                except Exception as e:
                    print(f'[DB] Violation end error: {e}')

    # ── Person tracking ──────────────────────
    def _match_persons(self, new_boxes):
        matched, used = [], set()

        # Increment unseen counter for all existing states
        for s in self.person_states:
            s['frames_unseen'] = s.get('frames_unseen', 0) + 1

        for box in new_boxes:
            px, py = (box[0]+box[2])/2, (box[1]+box[3])/2
            best_idx, best_dist = None, 500
            for i, s in enumerate(self.person_states):
                if i in used: continue
                d = np.hypot(px-(s['box'][0]+s['box'][2])/2,
                             py-(s['box'][1]+s['box'][3])/2)
                if d < best_dist:
                    best_dist, best_idx = d, i
            if best_idx is not None:
                used.add(best_idx)
                s = self.person_states[best_idx]
                s['box'] = tuple(int(0.4*o+0.6*n) for o,n in zip(s['box'], box))
                s['frames_unseen'] = 0   # person visible — reset counter
                matched.append(s)
            else:
                # Brand new person — create fresh state with all fields
                matched.append({
                    'box':              box,
                    'role':             None,
                    'role_history':     deque(maxlen=self.role_history_len),
                    'helmet_memory':    0, 'vest_memory':  0,
                    'helmet_vote':      0, 'vest_vote':    0,
                    'frames_unseen':    0,
                    # ArUco identity fields
                    'aruco_id':         None,
                    'person_name':      None,
                    'attendance_ok':    False,
                    # Per-person active DB violation IDs {type: violation_id}
                    'active_violations': {},
                })

        # Keep states that disappeared recently (up to ARUCO_KEEP_FRAMES)
        # This preserves aruco_id and person_name across brief YOLO detection gaps
        for i, s in enumerate(self.person_states):
            if i not in used:
                if s.get('frames_unseen', 0) <= ARUCO_KEEP_FRAMES:
                    matched.append(s)   # keep alive — still within window
                else:
                    # State timed out — person left frame
                    # Close all open violations so DB active count decreases
                    for vtype, vid in list(s.get('active_violations', {}).items()):
                        try:
                            self.db.log_violation_end(vid)
                            print(f"[DB] Auto-closed {vtype} — person left frame")
                        except Exception as e:
                            print(f"[DB] Auto-close error: {e}")

        self.person_states = matched

    @staticmethod
    def _overlap_area(a, b):
        return max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))

    @staticmethod
    def _nms_persons(boxes, min_w=70, min_h=90, min_aspect=0.80, iou_thresh=0.40):
        boxes = [b for b in boxes if (b[2]-b[0])>=min_w and (b[3]-b[1])>=min_h
                 and (b[3]-b[1])/max(b[2]-b[0],1)>=min_aspect]
        if len(boxes) <= 1: return boxes
        boxes = sorted(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
        keep, suppressed = [], set()
        for i, a in enumerate(boxes):
            if i in suppressed: continue
            keep.append(a)
            area_a = (a[2]-a[0])*(a[3]-a[1])
            for j, b in enumerate(boxes[i+1:], i+1):
                if j in suppressed: continue
                inter  = PPEZoneDetector._overlap_area(a, b)
                area_b = (b[2]-b[0])*(b[3]-b[1])
                union  = area_a + area_b - inter
                if union > 0 and inter/union > iou_thresh:
                    suppressed.add(j)
        return keep

    def _scale(self, c):
        return (int(c[0]*self.scale_x), int(c[1]*self.scale_y),
                int(c[2]*self.scale_x), int(c[3]*self.scale_y))

    def _get_dominant_color(self, frame, coords):
        x1, y1, x2, y2 = coords
        h_f, w_f = frame.shape[:2]
        crop = frame[max(0,y1):min(h_f,y2), max(0,x1):min(w_f,x2)]
        if crop.size == 0: return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        r1 = cv2.countNonZero(cv2.inRange(hsv, np.array([0,100,50]),   np.array([10,255,255])))
        r2 = cv2.countNonZero(cv2.inRange(hsv, np.array([160,100,50]), np.array([180,255,255])))
        red_px    = r1 + r2
        yellow_px = cv2.countNonZero(cv2.inRange(hsv, np.array([15,80,50]),  np.array([80,255,255])))
        blue_px   = cv2.countNonZero(cv2.inRange(hsv, np.array([90,50,50]),  np.array([130,255,255])))
        total = max((min(w_f,x2)-max(0,x1))*(min(h_f,y2)-max(0,y1)), 1)
        mx = max(red_px, yellow_px, blue_px)
        if mx >= total * 0.20:
            if mx == blue_px:    return "blue"
            if mx == red_px:     return "red"
            if mx == yellow_px:  return "yellow"
        return "unknown"

    # ── Process results (priority unchanged) ──
    def _process_results(self, frame, r_person, r_helmet, r_vest, r_fire):
        persons, helmet_boxes = [], []
        yellow_vest_boxes, red_vest_boxes, blue_vest_boxes = [], [], []

        for r in r_person:
            for box in r.boxes:
                if self.class_names_person[int(box.cls[0])].lower() in ('person','people'):
                    persons.append(self._scale(tuple(box.xyxy[0].cpu().numpy().astype(int))))

        for r in r_helmet:
            for box in r.boxes:
                if ('helmet' in self.class_names_helmet[int(box.cls[0])].lower()
                        and float(box.conf[0]) >= CONF_HELMET):
                    helmet_boxes.append(self._scale(tuple(box.xyxy[0].cpu().numpy().astype(int))))

        for r in r_vest:
            for box in r.boxes:
                if ('vest' in self.class_names_vest[int(box.cls[0])].lower()
                        and float(box.conf[0]) >= CONF_VEST):
                    coords = self._scale(tuple(box.xyxy[0].cpu().numpy().astype(int)))
                    col    = self._get_dominant_color(frame, coords)
                    if col == "blue":    blue_vest_boxes.append(coords)
                    elif col == "red":   red_vest_boxes.append(coords)
                    elif col == "yellow": yellow_vest_boxes.append(coords)

        persons = self._nms_persons(persons)
        self._match_persons(persons)

        helmet_assigned, yellow_assigned, red_assigned, blue_assigned = set(), set(), set(), set()

        def _best_person(ppe_box, zones):
            ppe_area = max((ppe_box[2]-ppe_box[0])*(ppe_box[3]-ppe_box[1]), 1)
            ppe_cx   = (ppe_box[0]+ppe_box[2]) / 2
            best_i, best_score = -1, 0.0
            for i, zone in enumerate(zones):
                inter = self._overlap_area(zone, ppe_box)
                if inter == 0: continue
                zone_area = max((zone[2]-zone[0])*(zone[3]-zone[1]), 1)
                zone_cx   = (zone[0]+zone[2]) / 2
                score = ((inter/ppe_area)*(inter/zone_area) *
                         max(0.1, 1.0 - abs(ppe_cx-zone_cx)/max(zone[2]-zone[0],1)*2.0))
                if score > best_score:
                    best_score, best_i = score, i
            return best_i if best_score >= 0.02 else -1

        head_zones, torso_zones = [], []
        for s in self.person_states:
            x1, y1, x2, y2 = s['box']
            h, w = y2-y1, x2-x1
            tx1, tx2 = x1+int(w*0.20), x2-int(w*0.20)
            head_zones.append( (tx1, max(0,y1-int(h*0.10)), tx2, y1+int(h*0.25)) )
            torso_zones.append((tx1, y1+int(h*0.15), tx2, y1+int(h*0.80)))

        for s in self.person_states: s.pop('vest_cx', None)

        for hb in helmet_boxes:
            idx = _best_person(hb, head_zones)
            if idx >= 0: helmet_assigned.add(idx)

        for vb, assigned in [(yellow_vest_boxes, yellow_assigned),
                              (red_vest_boxes,    red_assigned),
                              (blue_vest_boxes,   blue_assigned)]:
            for box in vb:
                idx = _best_person(box, torso_zones)
                if idx >= 0:
                    assigned.add(idx)
                    self.person_states[idx]['vest_cx'] = (box[0]+box[2]) / 2

        helmet_miss, vest_miss = set(), set()
        for i, s in enumerate(self.person_states):
            has_helmet = i in helmet_assigned
            has_vest   = (i in yellow_assigned) or (i in red_assigned) or (i in blue_assigned)

            if i in blue_assigned:    raw_role = "Engineer"
            elif i in yellow_assigned: raw_role = "Worker"
            elif i in red_assigned:   raw_role = "Visitor"
            else:                     raw_role = None

            s['role_history'].append(raw_role)
            counts = Counter(r for r in s['role_history'] if r is not None)
            s['role'] = counts.most_common(1)[0][0] if counts else None

            if not has_helmet: helmet_miss.add(i)
            if not has_vest:   vest_miss.add(i)

        for i, s in enumerate(self.person_states):
            # Helmet memory
            s['helmet_vote'] = min(s['helmet_vote']+1, self.vote_window) if i in helmet_miss \
                   else max(s['helmet_vote']-2, 0)
            if   s['helmet_vote'] >= self.vote_threshold: s['helmet_memory'] = self.memory_frames
            elif s['helmet_vote'] == 0:                   s['helmet_memory'] = 0
            elif s['helmet_memory'] > 0:                  s['helmet_memory'] -= 1

            # Vest memory
            if i in vest_miss:
                s['vest_vote'] = min(s['vest_vote']+1, self.vote_window)
                if s['vest_vote'] >= self.vote_threshold:
                    s['role_history'].clear()
                    s['role'] = None
            else:
                s['vest_vote'] = max(s['vest_vote']-2, 0)
            if   s['vest_vote'] >= self.vote_threshold: s['vest_memory'] = self.memory_frames
            elif s['vest_vote'] == 0:                   s['vest_memory'] = 0
            elif s['vest_memory'] > 0:                  s['vest_memory'] -= 1

        # Zone lookup using current active zones
        with self._zone_lock:
            current_zones = list(self.active_zones)

        for s in self.person_states:
            x1, y1, x2, y2 = s['box']
            feet_x = int(s['vest_cx']) if 'vest_cx' in s else int((x1+x2)/2)
            feet_y = int(y2)
            s['zone'] = next((z['name'] for z in current_zones
                              if point_in_polygon((feet_x, feet_y), z['points'])), None)

        # ── ArUco: assign IDs to nearest person states ─────────────────
        self._assign_aruco(self._latest_aruco)

        # ── DB: log violation start/end per person ───────────────────────
        self._log_violations(frame)

        # ── Fire detection (HIGHEST priority) ──
        fire_detected = False
        fire_boxes    = []
        for r in r_fire:
            for box in r.boxes:
                name = self.class_names_fire[int(box.cls[0])].lower()
                if "fire" in name or "flame" in name or "smoke" in name:
                    fire_boxes.append(self._scale(tuple(box.xyxy[0].cpu().numpy().astype(int))))
                    fire_detected = True
        self.fire_boxes = fire_boxes

        # ── Laser priority: 1.FIRE 2.Helmet 3.Red zone 4.No vest 5.Visitor in orange ──
        helmet_viol = red_viol = vest_viol = orange_viol = False
        for s in self.person_states:
            if s.get('helmet_memory', 0) > 0: helmet_viol = True
            if s.get('vest_memory',   0) > 0: vest_viol   = True
            role = None if s.get('vest_memory', 0) > 0 else s.get('role')
            pz   = s.get('zone')
            if pz is not None:
                tier = get_zone_tier(pz)
                if not is_authorized(role, tier):
                    if tier == "danger":                         red_viol    = True
                    elif tier == "orange" and role == "Visitor": orange_viol = True

        if fire_detected: new_state = "FIRE"
        elif helmet_viol: new_state = "HELMET"
        elif red_viol:    new_state = "NOENTRY"
        elif vest_viol:   new_state = "VEST"
        elif orange_viol: new_state = "NOENTRY"
        else:             new_state = "SAFE"

        self.global_violation_active = (new_state != "SAFE")

        # Stabilise: require 5 consistent frames before switching laser
        if new_state == self._pending_laser_state:
            self._pending_laser_count += 1
        else:
            self._pending_laser_state = new_state
            self._pending_laser_count  = 1

        if (self._pending_laser_count >= 5 and
                new_state != self.current_laser_state):
            now          = time.time()
            held         = now - self._laser_state_since
            cur_priority = LASER_PRIORITY.get(self.current_laser_state, 1)
            new_priority = LASER_PRIORITY.get(new_state, 1)
            if new_priority > cur_priority or held >= 3.0:
                self.current_laser_state  = new_state
                self._laser_state_since   = now
                self._pending_laser_count = 0
                self._trigger_laser(new_state)

    # ── Annotate ─────────────────────────────
    def _annotate(self, frame):
        out = frame.copy()
        with self._zone_lock:
            current_zones = list(self.active_zones)

        # ── 1. Draw zones ──
        for z in current_zones:
            cv2.polylines(out, [z['points']], True, z['color'], 2)
            cv2.putText(out, z['name'],
                        (int(z['points'][0][0]) + 5, int(z['points'][0][1]) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, z['color'], 2)

        # ── 2. Draw persons — only active states (frames_unseen == 0) ──
        # Ghost states (frames_unseen > 0) are kept for ID continuity but NOT drawn
        for s in self.person_states:
            if s.get('frames_unseen', 0) > 0:
                continue   # person left frame — skip rendering
            x1, y1, x2, y2 = s['box']
            role = None if s.get('vest_memory', 0) > 0 else s.get('role')

            if role == "Engineer":  box_color = (255, 140, 0)
            elif role == "Worker":  box_color = (0, 220, 255)
            elif role == "Visitor": box_color = (180, 180, 180)
            else:                   box_color = (0, 200, 0)

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            # Show name (ArUco) if known, else role, else 'Person'
            pname = s.get('person_name')
            if pname and role:
                label = f'{pname} ({role})'
            elif pname:
                label = pname
            elif role:
                label = role
            else:
                label = 'Person'
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            y_off = y1 + 22

            # ── PPE violations ──
            if s.get('helmet_memory', 0) > 0:
                cv2.putText(out, "No Helmet", (x1, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_off += 22
            if s.get('vest_memory', 0) > 0:
                cv2.putText(out, "No Vest", (x1, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_off += 22

            # ── Feet dot + zone label ──
            feet_x = int(s['vest_cx']) if 'vest_cx' in s else int((x1 + x2) / 2)
            feet_y = int(y2)
            fp     = (feet_x, feet_y)
            pz     = s.get('zone')

            authorized = True
            if pz is not None:
                tier       = get_zone_tier(pz)
                authorized = is_authorized(role, tier)

            if pz is not None:
                if authorized:
                    cv2.circle(out, fp, 7, (0, 255, 0), -1)
                    cv2.circle(out, fp, 8, (255, 255, 255), 1)
                    cv2.putText(out, pz, (fp[0] + 9, fp[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    cv2.circle(out, fp, 9, (0, 0, 255), -1)
                    cv2.circle(out, fp, 10, (255, 255, 255), 1)
                    cv2.putText(out, f"UNAUTHORIZED in {pz}", (x1, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    y_off += 22
            else:
                cv2.circle(out, fp, 5, (150, 150, 150), -1)

        # ── 3. Legend ──
        legend = [
            ("Worker   — yellow vest", (0, 200, 255)),
            ("Engineer — blue vest",   (255, 120, 0)),
            ("Visitor  — red vest",    (180, 180, 180)),
            ("o feet inside zone  (green=OK  red=UNAUTHORIZED)", (200, 200, 200)),
        ]
        lx, ly = 10, 30
        for txt, col in legend:
            cv2.rectangle(out, (lx, ly - 12), (lx + 14, ly + 2), col, -1)
            cv2.putText(out, txt, (lx + 20, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            ly += 22

        # ── 4. ArUco overlay — draw detected markers ──
        for (mid, cx, cy, px_size, pts) in self._latest_aruco:
            info = PERSON_REGISTRY.get(mid, {'name': f'ID{mid}'})
            col  = (0, 255, 0) if px_size >= 15 else (0, 120, 255)
            cv2.polylines(out, [pts], True, col, 2)
            cv2.putText(out, f"ArUco {mid}: {info['name']}",
                        (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        # ── 5. Fire boxes ──
        for fb in getattr(self, 'fire_boxes', []):
            fx1, fy1, fx2, fy2 = fb
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
            cv2.putText(out, "FIRE!", (fx1, fy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ── 6. Laser state HUD (bottom) ──
        display_state = self.current_laser_state if self.current_laser_state else "SAFE"
        laser_col = (0, 0, 255) if display_state == "FIRE" else (0, 255, 255)
        cv2.putText(out, f"LASER STATE: {display_state}", (10, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, laser_col, 2)

        return out

    # ── Main loop ────────────────────────────
    def run(self, source=RTSP_URL):
        print(f"Connecting to stream: {source}")
        stream = VideoStream(source)
        worker = InferenceWorker(self.model_person, self.model_helmet,
                                 self.model_vest, self.model_fire, self.device)
        hw     = HardwareManager(self)
        hw.start()

        display = None
        cv2.namedWindow("PPE + Zone + Rail")
        self._trigger_laser("SAFE")
        print("System running. Press Q to quit.")

        frame_count = 0
        try:
            while True:
                ret, frame = stream.read()
                if not ret or frame is None:
                    if display is not None:
                        cv2.imshow("PPE + Zone + Rail", display)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    continue

                frame = cv2.resize(frame, FRAME_SIZE)
                frame_count += 1

                if frame_count % 3 == 0:
                    worker.submit(frame)

                # ArUco runs every frame — fast CPU-only detection
                self._latest_aruco = self.aruco.detect(frame)

                result = worker.get_result()
                if result is not None:
                    inf_frame, r_person, r_helmet, r_vest, r_fire = result
                    self._process_results(inf_frame, r_person, r_helmet, r_vest, r_fire)
                    display = self._annotate(inf_frame)

                # Always show something — raw frame until first inference result ready
                cv2.imshow("PPE + Zone + Rail", display if display is not None else frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Keyboard interrupt.")

        finally:
            print("Shutting down cleanly...")
            # Close any violations still open in DB
            for s in self.person_states:
                for vtype, vid in list(s.get('active_violations', {}).items()):
                    try: self.db.log_violation_end(vid)
                    except: pass
            hw.stopped = True
            self.laser.stop()
            worker.stop()
            stream.release()
            cv2.destroyAllWindows()
            print("Done.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    PPEZoneDetector().run()