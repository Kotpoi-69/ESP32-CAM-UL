from flask import Flask, request, jsonify
from ultralytics import YOLO
import requests
import time
import threading
import queue
import os
import cv2
import numpy as np
import gc

app = Flask(__name__)

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "exp-7.pt"

FIREBASE_URL = "https://smart-agriculture-berbasis-ai-default-rtdb.asia-southeast1.firebasedatabase.app/cam.json"

CONF_THRESHOLD = 0.25
TARGET_CLASS = "item"

# =====================================================
# GLOBAL
# =====================================================
model = None
model_lock = threading.Lock()

last_request_time = time.time()

# lebih aman untuk render free tier
MODEL_TIMEOUT = 600

# =====================================================
# QUEUE
# =====================================================
job_queue = queue.Queue(maxsize=5)

# =====================================================
# HTTP SESSION
# =====================================================
session = requests.Session()

# =====================================================
# WORKER
# =====================================================
def worker():
    global model
    global last_request_time

    while True:
        image_bytes = job_queue.get()

        try:
            print("\n[WORKER] Processing...")

            # =========================================
            # VALIDASI IMAGE
            # =========================================
            if len(image_bytes) < 1000:
                print("[WORKER] Invalid image size")
                continue

            # =========================================
            # LOAD MODEL
            # =========================================
            with model_lock:
                if model is None:
                    print("[WORKER] Loading model...")

                    model = YOLO(MODEL_PATH)

                    print("[WORKER] Model classes:",
                          model.names)

                    print("[WORKER] Model loaded")

            # =========================================
            # BYTE TO IMAGE
            # =========================================
            npimg = np.frombuffer(
                image_bytes,
                np.uint8
            )

            img = cv2.imdecode(
                npimg,
                cv2.IMREAD_COLOR
            )

            if img is None:
                print("[WORKER] Invalid image decode")
                continue

            # =========================================
            # YOLO INFERENCE
            # =========================================
            start_time = time.time()

            with model_lock:
                results = model(
                    img,
                    conf=CONF_THRESHOLD,
                    verbose=False
                )

            inference_time = round(
                time.time() - start_time,
                2
            )

            print(f"[WORKER] Inference: {inference_time}s")

            # =========================================
            # DETECTION
            # =========================================
            max_conf = 0
            detected = False

            for result in results:

                boxes = result.boxes

                if boxes is None:
                    continue

                for box in boxes:

                    cls_id = int(box.cls[0])

                    conf = float(box.conf[0])

                    class_name = \
                        model.names[cls_id].lower()

                    if class_name == TARGET_CLASS:

                        detected = True

                        max_conf = max(
                            max_conf,
                            conf
                        )

            # =========================================
            # PAYLOAD
            # =========================================
            payload = {
                "tanaman": detected,
                "confidence": round(max_conf, 3),
                "timestamp": int(time.time())
            }

            print("[WORKER] Payload:", payload)

            # =========================================
            # FIREBASE
            # =========================================
            res_fb = session.put(
                FIREBASE_URL,
                json=payload,
                timeout=10
            )

            print("[WORKER] Firebase:",
                  res_fb.status_code)

            # =========================================
            # CLEANUP
            # =========================================
            del results
            gc.collect()

        except Exception as e:

            print("[WORKER] ERROR:", str(e))

        finally:

            job_queue.task_done()

# =====================================================
# AUTO SLEEP
# =====================================================
def auto_sleep():
    global model

    while True:

        try:
            if model is not None:

                idle_time = \
                    time.time() - last_request_time

                if idle_time > MODEL_TIMEOUT:

                    print(
                        "\n💤 No activity, unloading model..."
                    )

                    with model_lock:
                        model = None

                    gc.collect()

                    print(
                        "✅ Model unloaded (sleep mode)"
                    )

        except Exception as e:

            print("[AUTO_SLEEP] ERROR:", str(e))

        time.sleep(10)

# =====================================================
# START THREADS
# =====================================================
threading.Thread(
    target=auto_sleep,
    daemon=True
).start()

# single worker lebih aman
for i in range(1):

    threading.Thread(
        target=worker,
        daemon=True
    ).start()

# =====================================================
# ROUTE DETECT
# =====================================================
@app.route('/detect', methods=['POST'])
def detect():

    global last_request_time

    try:
        image_bytes = request.data

        # =============================================
        # VALIDASI
        # =============================================
        if not image_bytes:
            return jsonify({
                "error": "no image"
            }), 400

        # =============================================
        # UPDATE REQUEST TIME
        # =============================================
        last_request_time = time.time()

        # =============================================
        # QUEUE CHECK
        # =============================================
        if job_queue.full():

            print("[SERVER] Queue full")

            return jsonify({
                "status": "busy"
            }), 503

        # =============================================
        # ADD TO QUEUE
        # =============================================
        job_queue.put(image_bytes)

        print("[SERVER] Queued:",
              job_queue.qsize())

        return jsonify({
            "status": "queued"
        }), 200

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# =====================================================
# HEALTH CHECK
# =====================================================
@app.route('/')
def home():

    return jsonify({
        "status": "YOLO backend running"
    })

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )