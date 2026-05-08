"""
DRISHTI - AI Vision Assistant for the Visually Impaired
Demo Application Backend
Author: Based on project by Aditya Pratap Singh, Sunbeam School Lahartara
"""

import cv2
import numpy as np
import threading
import time
import queue
import json
import io
import os
import sys
from collections import defaultdict, deque
from flask import Flask, render_template, Response, jsonify, request

# ─── Optional imports with graceful fallback ────────────────────────────────

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. Object detection disabled.")

OCR_AVAILABLE = False
TTS_AVAILABLE = False

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)

# ─── Global State ────────────────────────────────────────────────────────────

state = {
    "mode": "object",          # object | ocr | face | combined
    "latitude": None,
    "longitude": None,
    "tts_enabled": True,
    "confidence": 0.45,
    "running": True,
    "fps": 0,
    "coords": None,
    "detected": [],
    "ocr_text": "",
    "face_count": 0,
    "last_frame_time": time.time(),
    "status": "Initializing…",
    "frame_skip": 2,
    "frame_count": 0,
}

output_frame = None
frame_lock = threading.Lock()

# ─── TTS Compatibility Stub ───────────────────────────────────────────────────

def speak(text):
    # Speech feature removed; preserve interface for API compatibility.
    return

# ─── Model Loading ───────────────────────────────────────────────────────────

yolo_model = None
face_cascade = None

def load_models():
    global yolo_model, face_cascade
    state["status"] = "Loading models…"

    # YOLO
    if YOLO_AVAILABLE:
        try:
            print("[Model] Loading YOLOv8n…")
            yolo_model = YOLO("yolov8n.pt")
            yolo_model.to("cpu")
            print("[Model] YOLOv8n loaded.")
        except Exception as e:
            print(f"[Model] YOLO load error: {e}")

    # OpenCV Face Cascade
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print("[Model] Face cascade loaded.")
    except Exception as e:
        print(f"[Model] Face cascade error: {e}")

    state["status"] = "Ready"

# ─── Frame Processing ─────────────────────────────────────────────────────────

COLORS = {
    "object": (0, 200, 255),
    "face":   (0, 255, 150),
    "text":   (255, 200, 0),
}

def detect_objects(frame):
    if yolo_model is None:
        return [], frame

    h, w = frame.shape[:2]
    results = yolo_model(frame, imgsz=640, conf=state["confidence"],
                         verbose=False, device="cpu")
    detections = []
    annotated = frame.copy()

    if results and results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = COLORS["object"]
            # Draw fancy box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Draw label background
            label_text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            detections.append({"label": label, "confidence": round(conf, 2),
                                "bbox": [x1, y1, x2, y2]})

    # Announce new objects
    return detections, annotated


def detect_faces(frame):
    if face_cascade is None:
        return 0, frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    annotated = frame.copy()
    count = len(faces)

    for (x, y, w, h) in faces:
        color = COLORS["face"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        # Corner accents
        L = 20
        cv2.line(annotated, (x, y), (x + L, y), color, 3)
        cv2.line(annotated, (x, y), (x, y + L), color, 3)
        cv2.line(annotated, (x + w, y), (x + w - L, y), color, 3)
        cv2.line(annotated, (x + w, y), (x + w, y + L), color, 3)
        cv2.line(annotated, (x, y + h), (x + L, y + h), color, 3)
        cv2.line(annotated, (x, y + h), (x, y + h - L), color, 3)
        cv2.line(annotated, (x + w, y + h), (x + w - L, y + h), color, 3)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - L), color, 3)

        label = "Face"
        cv2.putText(annotated, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return count, annotated


def draw_overlay(frame, mode, detections, face_count, ocr_text, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Top bar ────────────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 48), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, overlay)

    # DRISHTI title
    cv2.putText(overlay, "DRISHTI", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)

    # Mode badge
    mode_label = {"object": "OBJECT DETECT", "ocr": "TEXT READER",
                  "face": "FACE DETECT", "combined": "COMBINED"}.get(mode, mode.upper())
    (mw, _), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    mx = w // 2 - mw // 2
    cv2.putText(overlay, mode_label, (mx, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 50), 1, cv2.LINE_AA)

    # FPS
    fps_text = f"{fps:.0f} FPS"
    (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(overlay, fps_text, (w - fw - 12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

    # Coordinates
    if state["coords"]:
        lat, lon = state["coords"]
        coord_text = f"{lat:5f}, {lon:.5f}"
        (cw, _), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(overlay, coord_text, (w - cw - 12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 255), 1, cv2.LINE_AA)

    # ── Bottom status ──────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, h - 36), (w, h), (10, 10, 20), -1)
    if mode in ("object", "combined") and detections:
        labels = list(dict.fromkeys(d["label"] for d in detections))
        summary = "Detected: " + ", ".join(labels[:6])
        cv2.putText(overlay, summary, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    elif mode == "face":
        cv2.putText(overlay, f"Faces Detected: {face_count}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1, cv2.LINE_AA)
    elif mode == "ocr" and ocr_text:
        short = ocr_text.replace("\n", " ")[:70]
        cv2.putText(overlay, short, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 50), 1, cv2.LINE_AA)

    # ── Latitude & Longitude ───────────────────────────────────────────
    # lat = state.get("latitude")
    # lon = state.get("longitude")
    # if lat is not None and lon is not None:
    #     coord_text = f"Lat: {lat:.6f}, Lon: {lon:.6f}"
    #     (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #     cv2.putText(overlay, coord_text, (w - fw - 12, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

    return overlay


# ─── Camera Thread ───────────────────────────────────────────────────────────

def camera_thread():
    global output_frame
    cap = None
    fps_window = deque(maxlen=30)

    # Try camera indices
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[Camera] Opened camera {idx}")
            break
        cap = None

    if cap is None:
        state["status"] = "No camera found"
        print("[Camera] No camera found. Using test pattern.")

    if cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while state["running"]:
        t0 = time.time()

        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
        else:
            # Test pattern
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 2)

        state["frame_count"] += 1
        mode = state["mode"]

        detections = state["detected"]
        face_count = state["face_count"]
        ocr_text   = state["ocr_text"]

        # ── Processing (every frame_skip frames) ──────────────────────────
        skip = state["frame_count"] % state["frame_skip"] == 0

        if skip:
            if mode == "object":
                detections, frame = detect_objects(frame)
                state["detected"] = detections

            elif mode == "face":
                face_count, frame = detect_faces(frame)
                state["face_count"] = face_count

            elif mode == "combined":
                detections, frame = detect_objects(frame)
                state["detected"] = detections
                face_count, frame = detect_faces(frame)
                state["face_count"] = face_count

        # ── Draw overlay ────────────────────────────────────────────────────
        fps = state["fps"]
        frame = draw_overlay(frame, mode, detections, face_count, ocr_text, fps)

        # ── FPS ─────────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        fps_window.append(elapsed)
        if fps_window:
            state["fps"] = 1.0 / (sum(fps_window) / len(fps_window))

        # ── Encode ──────────────────────────────────────────────────────────
        _, buf = cv2.imencode(".jpg", frame,
                              [cv2.IMWRITE_JPEG_QUALITY, 82])
        with frame_lock:
            output_frame = buf.tobytes()

    if cap:
        cap.release()


# ─── Flask Routes ────────────────────────────────────────────────────────────

def gen_frames():
    global output_frame
    while True:
        with frame_lock:
            frame = output_frame
        if frame is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.015)  # ~60fps max


@app.route("/")
def index():
    return render_template("index.html",
                           yolo=YOLO_AVAILABLE,
                           ocr=False,
                           tts=False)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify({
        "mode":       state["mode"],
        "fps":        round(state["fps"], 1),
        "detected":   state["detected"][:10],
        "face_count": state["face_count"],
        "ocr_text":   state["ocr_text"],
        "tts":        state["tts_enabled"],
        "status":     state["status"],
        "confidence": state["confidence"],
    })


@app.route("/set_mode", methods=["POST"])
def set_mode():
    data = request.get_json()
    mode = data.get("mode", "object")
    if mode in ("object", "ocr", "face", "combined"):
        state["mode"] = mode
        state["detected"] = []
        state["ocr_text"] = ""
        state["face_count"] = 0
    return jsonify({"ok": True, "mode": state["mode"]})


@app.route("/set_tts", methods=["POST"])
def set_tts():
    data = request.get_json()
    state["tts_enabled"] = bool(data.get("enabled", True))
    return jsonify({"ok": True, "tts": state["tts_enabled"]})


@app.route("/set_confidence", methods=["POST"])
def set_confidence():
    data = request.get_json()
    val = float(data.get("value", 0.45))
    state["confidence"] = max(0.1, min(0.95, val))
    return jsonify({"ok": True, "confidence": state["confidence"]})


@app.route("/speak_objects", methods=["POST"])
def speak_objects():
    mode = state["mode"]
    parts = []

    if state["detected"]:
        labels = list(dict.fromkeys(d["label"] for d in state["detected"]))
        parts.append(", ".join(labels))

    if mode in ("face", "combined") and state["face_count"] > 0:
        n = state["face_count"]
        parts.append(f"{n} face{'s' if n != 1 else ''}")

    if mode in ("ocr", "combined") and state["ocr_text"]:
        parts.append(state["ocr_text"][:200])

    text = ". ".join(parts) if parts else "Nothing detected right now."
    speak(text)
    return jsonify({"ok": True, "spoken": text})

@app.route("/set_coords", methods=["POST"])
def set_coords():
    data = request.get_json()
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is not None and lon is not None:
        state["coords"] = (float(lat), float(lon))
    return jsonify({"ok": True}) 


# ─── Startup ─────────────────────────────────────────────────────────────────

def startup():
    load_models()
    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()
    speak("DRISHTI is ready. Welcome.")


if __name__ == "__main__":
    startup()
    print("\n" + "═" * 50)
    print("  DRISHTI Demo → http://127.0.0.1:5000")
    print("═" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
