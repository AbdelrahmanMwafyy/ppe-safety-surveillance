# AI Safety Surveillance System
### CUFE-STEP 2026 | Cairo University | Mechatronics Engineering (MEE)

A closed-loop industrial safety system that detects PPE violations, identifies individual workers, enforces zone access control, and responds through a BLE laser projector, audio alerts, and a live web dashboard — all in real time without human intervention.

---

## What It Does

The system mounts a PTZ camera on a motorized linear rail that continuously sweeps the factory floor. Four YOLOv8 models run simultaneously on a dedicated GPU, detecting persons, helmets, safety vests, and fire. When a violation is confirmed, the system:

- Projects a warning sign directly onto the floor via a BLE laser projector
- Announces the worker ID and violation type via audio
- Stops the rail motor and holds the camera on the scene
- Logs the event to a SQLite database with a screenshot
- Displays everything on a password-protected web dashboard accessible from any device on the network

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Python Workstation                  │
│                                                     │
│  VideoStream ──► InferenceWorker (GPU, TensorRT)    │
│       │                  │                          │
│  Main Thread ◄────────── │ ──► Database (SQLite)    │
│       │                  │         │                │
│  HardwareManager    ArucoDetector  │                │
│  (ESP32 + ONVIF)         │     Dashboard (Flask)    │
│       │           LaserBLEWorker                    │
└───────┼──────────────────┼──────────────────────────┘
        │                  │
    ESP32 Rail          BLE Laser
    + Tapo C220         Projector
```

**Five concurrent threads — main thread never blocks on I/O:**

| Thread | Responsibility |
|---|---|
| VideoStream | Continuous RTSP frame reading, auto-reconnect on dropout |
| InferenceWorker | All 4 YOLO models on GPU via one-slot queue |
| HardwareManager | ESP32 TCP motor control + ONVIF PTZ tilt |
| LaserBLEWorker | Persistent BLE connection, priority-based laser projection |
| Main Thread | ArUco detection, frame annotation, display, DB logging |

---

## Hardware

| Component | Spec |
|---|---|
| Camera | Tapo C220 — 2K QHD, 114° FOV, RTSP + ONVIF PTZ |
| Rail Motor | NEMA 17 stepper via A4988/DRV8825 driver |
| Controller | ESP32 38-pin — WiFi TCP server on port 8080 |
| Laser | BLE RGB projector — reverse engineered, no SDK |
| GPU | NVIDIA RTX 3070 Laptop — CUDA 12.x, TensorRT |

---

## Detection Pipeline

### Four YOLOv8 Models (TensorRT .engine — 2-5x faster than .pt)
| Model | Target | Confidence |
|---|---|---|
| Person | Human presence | 0.45 |
| Helmet | Head PPE | 0.25 |
| Vest | Body PPE | 0.25 |
| Fire | Fire / flame / smoke | 0.40 |

Inference runs every 3rd frame on 640×640 input. Results scaled back to 1280×720 display coordinates.

### Role Classification (HSV Vest Color)
| Vest Color | Role | Zone Access |
|---|---|---|
| Blue | Engineer | Danger + Orange + Safe |
| Yellow | Worker | Orange + Safe |
| Red | Visitor | Safe only |

### Zone Access Control
Three zone tiers loaded from `three_zones.json`. Three zone sets maintained simultaneously (moving / tilted-right / tilted-left) — active set swaps atomically when camera position changes. Zone membership checked at person's **foot point** (bottom-center of bounding box) for physical accuracy.

---

## ArUco Worker Identification

Each worker wears a printed ArUco ID card (DICT_4X4_50, 10cm) clipped to their chest. The system identifies individuals without face recognition — CPU-only, single pass per frame, no GPU required.

| ArUco ID | Name | Role |
|---|---|---|
| 0 | Mohamed | Engineer |
| 1 | Ahmed | Visitor |
| 2 | Khaled | Worker |
| 3 | Youssef | Worker |

**Assignment algorithm:** Primary method checks if marker center falls inside person bounding box (±40px tolerance). Fallback uses nearest active centroid within 300px. Identity persists for 10 frames after marker disappears — prevents label flickering.

---

## BLE Laser — Reverse Engineered

The laser projector had no SDK, no documentation, and no open protocol. The full communication protocol was reverse engineered by:

1. Enumerating GATT services with a BLE scanner
2. Enabling Android HCI Bluetooth snoop log on a Samsung Galaxy
3. Capturing all ATT write packets in Wireshark during normal app operation
4. Analyzing byte sequences to reconstruct the command structure

**Packet structure per projection:**
```
Handshake → CONFIG_A → CONFIG_B → WORD payload
(minimum 20ms gap between packets)
```

**Five laser states with priority cascade:**

| Priority | State | Trigger |
|---|---|---|
| 5 | FIRE | Fire/smoke detected anywhere |
| 4 | HELMET | Any person missing helmet |
| 3 | NOENTRY | Unauthorized zone entry |
| 2 | VEST | Any person missing vest |
| 1 | SAFE | No active violations |

Higher priority always switches immediately. Lower priority requires 3-second hold before downgrading. State must be consistent for 5 frames before triggering to eliminate 1-2 frame flicker.

---

## Motor Sweep Cycle

```
Home → Motor RIGHT → Right limit hit → Tilt +0.25 → Zones switch → Dwell 10s
     → Home → Motor LEFT → Left limit hit → Tilt -0.25 → Zones switch → Dwell 10s
     → Repeat indefinitely
```

Violation detected during sweep → motor stops immediately → camera holds on violation → resumes after violation clears + 1 second confirmation.

---

## Database (SQLite — safety_system.db)

Four tables designed for operational safety management:

**persons** — static worker registry (ArUco ID → name, role)

**violations** — every confirmed event: person, type, zone, start/end time, duration, screenshot path, confidence (HIGH = ArUco confirmed / LOW = centroid tracked)

**daily_summary** — pre-aggregated counts per person per violation type per day (powers trend charts without expensive GROUP BY on full history)

**attendance** — first ArUco detection per person per day via INSERT OR IGNORE — automatic sign-in, no manual process

Violations under 5 seconds are automatically deleted (false positive filter). Stale open violations from crashed sessions are closed on startup.

---

## Web Dashboard (Flask)

Password-protected, accessible from any device on the local network.

| Page | Content |
|---|---|
| Overview | Live stats, today's violations, attendance — auto-refreshes every 5s |
| Workers | All 4 workers: today/total violations, presence indicator, role badge |
| Person Profile | 30-day bar chart, violation type doughnut, full history, attendance log |
| Violations | Filterable log by type/confidence/person with screenshot links |
| Trends | Weekly bar, all-time doughnut, 30-day per-person line, hourly distribution |
| Review | LOW confidence / unknown violations — manual identity assignment |
| Reset | Two-step confirmation to clear all data (preserves worker registry) |

---

## Audio Alerts (pyttsx3 — offline TTS)

Synchronized with laser state changes. Same priority, same moment:

```
Laser → HELMET   : "ID 0, wear your helmet"
Laser → NOENTRY  : "ID 2, you are in an unauthorized zone"
Laser → VEST     : "ID 1, wear your safety vest"
Laser → FIRE     : "Fire detected. Evacuate immediately."
Laser → SAFE     : silence
```

---

## Tech Stack

```
Python 3.11        ultralytics (YOLOv8)    TensorRT (.engine)
OpenCV             PyTorch + CUDA 12.x     onvif-zeep (PTZ)
bleak (BLE)        pyttsx3 (TTS)           Flask (dashboard)
SQLite3            threading / asyncio     Chart.js
```

---

## Project Structure

```
AI_Master_System/
    master_system.py    ← main system (detection + hardware + DB)
    Dashboard.py        ← Flask web dashboard
    database.py         ← SQLite interface (shared by both)
    safety_system.db    ← auto-created on first run
    run.py              ← single launcher for full system
    three_zones.json    ← zone polygon definitions
    screenshots/        ← auto-saved violation JPEGs
```

---

## Run

```bash
# Install dependencies
pip install ultralytics opencv-python torch bleak onvif-zeep flask pyttsx3

# Start everything (camera system + dashboard)
python run.py

# Or separately:
python master_system.py
python Dashboard.py      # opens at http://localhost:5000
```

Dashboard password: `step2026`

For external device access (phone/tablet on same WiFi), add Windows Firewall rule:
```powershell
New-NetFirewallRule -DisplayName "SafetyAI Dashboard" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```

---

## Keywords

PPE Detection · ArUco Person Identification · Real-Time Safety Surveillance · Zone Access Control · BLE Laser Warning · TensorRT YOLO Inference · Multi-Threaded Architecture · ESP32 Motor Control · ONVIF PTZ · SQLite · Flask Dashboard

---

*Cairo University — Faculty of Engineering | CUFE-STEP 2026 | MEC_07*
