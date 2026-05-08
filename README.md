# 👁 DRISHTI — AI Vision Assistant Demo
### By Aditya Pratap Singh · Sunbeam School, Lahartara
### 🏆 IIC Regional Meet 2024 — First Position (ATL Schools)

---

## What It Does

DRISHTI is a real-time AI vision system for the visually impaired, featuring:

| Feature | Technology | Description |
|---|---|---|
| 🔍 **Object Detection** | YOLOv8n (YOLO nano) | Detects 80+ everyday objects in real time |
| 👤 **Face Detection** | OpenCV Haar Cascades | Counts and locates faces in frame |
| 📄 **Text Reader (OCR)** | Tesseract OCR | Reads text from signs, books, menus |
| 🔊 **Voice Output (TTS)** | pyttsx3 (offline) | Announces detections in real time |

All processing runs **on CPU only** — no dedicated GPU required.

---

## Quick Setup (5 minutes)

### Step 1 — Install Python 3.10+
Download from https://python.org if not already installed.

### Step 2 — Install Tesseract OCR (for text reading)
- **Windows**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
  - Install to default path: `C:\Program Files\Tesseract-OCR\`
- **macOS**: `brew install tesseract`
- **Linux/Ubuntu**: `sudo apt install tesseract-ocr`

### Step 3 — Install Python packages
Open a terminal in this folder and run:
```bash
pip install -r requirements.txt
```
> Note: `ultralytics` will download the YOLOv8n model (~6MB) on first run.

### Step 4 — Run the app
```bash
python app.py
```

Then open your browser at **http://localhost:5000**

---

## Usage

### Modes (also accessible with keyboard shortcuts 1–4)
- **1 — Object Detect**: Identifies objects in your camera view
- **2 — Face Detect**: Counts and highlights faces
- **3 — Text Reader**: Reads text from documents/signs (OCR)
- **4 — Combined**: All features simultaneously

### Controls
- **V key**: Toggle voice on/off
- **Confidence slider**: Adjust detection sensitivity
- **Voice Output toggle**: Enable/disable TTS narration

---

## Performance Tips (CPU-only laptops)

The app is already optimized for CPU:
- Uses YOLOv8 **nano** model (fastest variant)
- Processes every other frame to maintain smooth video
- JPEG streaming keeps latency under 50ms
- Tesseract OCR runs every 15 frames to avoid CPU overload

Expected performance on modern laptop CPU:
- Object detection: ~10–20 FPS
- Face detection: ~25–30 FPS
- OCR: ~5–10 FPS (updates every ~2 seconds)

---

## Troubleshooting

**"No camera found"** — Make sure no other app is using your webcam. Try closing video call apps.

**"OCR not available"** — Install Tesseract (Step 2) and ensure it's in your system PATH.

**Low FPS** — Lower the confidence threshold slider, or switch to Face-only mode.

**No voice output** — Some systems need additional TTS voices. On Windows, check Settings → Ease of Access → Narrator. On Linux, install: `sudo apt install espeak`

---

## Project Background

DRISHTI ("vision" in Sanskrit) aims to make AI-powered visual assistance accessible and affordable. Commercial solutions like OrCam MyEye cost ₹3.69 lakhs; DRISHTI targets ₹13,500 — over **27× cheaper**.

Built on:
- **Python** (OOP architecture for maintainability)
- **YOLO** (You Only Look Once) for real-time object detection
- **Raspberry Pi** compatible (the hardware version runs on RPi 4)
- **100+ language TTS** support planned for commercial release
