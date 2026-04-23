# Real-Time PPE Safety Surveillance System

Detects missing helmets and safety vests on live IP camera 
feeds with zone-based role authorization and vest color 
classification.

![Visitor No Helmet Detection](assets/demo/visitor_no_helmet.png)
![worker No Helmet Detection](assets/demo/worker_no_helmet.png)

## What It Does

- Detects PPE violations in real time on live RTSP stream
- Classifies worker roles by vest color:
  - Yellow vest = Worker
  - Blue vest = Engineer  
  - black = Visitor
- Zone-based access control:
  - Red zone: Engineer only
  - Orange zone: Engineer + Worker
  - Green zone: All roles
- Multi-model pipeline: role classification + PPE detection
- CUDA-accelerated inference on NVIDIA RTX 3070
- Auto-reconnect if camera stream drops
- Threaded VideoStream prevents frame buffer lag

## Detection Classes

Helmet · NO-Hardhat · Safety-Vest · NO-Safety-Vest · 
Person · Safety-Cone · Mask · NO-Mask · machine

## Demo

![Visitor Role - No Helmet](assets/demo/visitor_no_helmet.png)

## Tech Stack

- YOLOv8 (Ultralytics) — custom trained model
- OpenCV
- CUDA / PyTorch
- RTSP via Tapo C200 IP camera
- Python 3.10+

## Setup

1. Install dependencies: pip install -r requirements.txt
2. Edit camera config at top of detection.py:
   TAPO_IP = "your_camera_ip"
   TAPO_USER = "your_camera_username"
   TAPO_PASS = "your_camera_password"
3. Run: python detection.py

## File Structure

- detection.py — main PPE detection on RTSP stream
- zone_detection.py — zone-based role authorization
- smart_integration.py — dual model pipeline
- zone/zone.py — static zone definitions
- zone/zones.json — dynamic zone configuration

## Business Use Case

Automated PPE compliance monitoring for construction sites,
factories, and industrial facilities. Detects violations 
and identifies worker roles in real time without manual 
supervision.

Built as graduation project — Cairo University 
Mechatronics Engineering 2026.
