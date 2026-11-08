
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

# GANTI DENGAN URL FIREBASE REALTIME DATABASE LU
# Pastikan endpoint mengarah ke /cam.json
FIREBASE_URL = "https://smart-agriculture-berbasis-ai-default-rtdb.asia-southeast1.firebasedatabase.app/cam.json"

CONF_THRESHOLD = 0.25
TARGET_CLASS = "item"


# =====================================================
# GLOBAL
# =====================================================

model = None

# Lock untuk YOLO model
model_lock = threading.Lock()

# Lock untuk urutan frame dan Firebase update
frame_lock = threading.Lock()

# Waktu request terakhir yang diterima
last_request_time = time.time()

# Auto unload setelah 600 detik tanpa request
MODEL_TIMEOUT = 600

# Nomor urut frame (hanya di memori backend)
latest_frame_id = 0


# =====================================================
# LATEST FRAME QUEUE
# =====================================================

# Hanya satu gambar yang boleh menunggu
job_queue = queue.Queue(maxsize=1)


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
    global latest_frame_id

    while True:

        # Ambil frame terbaru dari antrean
        frame_id, image_bytes = job_queue.get()

        try:

            print("\n[WORKER] Processing frame:", frame_id)

            # =========================================
            # VALIDASI IMAGE
            # =========================================

            if len(image_bytes) < 1000:

                print("[WORKER] Invalid image size")

                continue

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
            # LOAD MODEL + YOLO INFERENCE
            # =========================================

            start_time = time.time()

            with model_lock:

                # Load model jika belum tersedia
                if model is None:

                    print("[WORKER] Loading model...")

                    model = YOLO(MODEL_PATH)

                    print(
                        "[WORKER] Model classes:",
                        model.names
                    )

                    print("[WORKER] Model loaded")

                # Inference dilakukan dalam lock yang sama
                # agar auto_sleep tidak meng-unload model
                # ketika inference sedang berjalan.

                results = model(
                    img,
                    conf=CONF_THRESHOLD,
                    verbose=False
                )

                class_names = model.names

            inference_time = round(
                time.time() - start_time,
                2
            )

            print(
                f"[WORKER] Inference: {inference_time}s"
            )

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

                    class_name = str(
                        class_names[cls_id]
                    ).lower()

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

                "confidence": round(
                    max_conf,
                    3
                ),

                "timestamp": int(time.time())

            }

            print(
                "[WORKER] Payload:",
                payload
            )

            # =========================================
            # VALIDASI LATEST FRAME
            # =========================================

            with frame_lock:

                # Pastikan tidak ada frame lebih baru
                # yang sudah diterima backend.

                if frame_id != latest_frame_id:

                    print(
                        f"[WORKER] Frame {frame_id} outdated."
                        " Skip Firebase update."
                    )

                    continue

                # =====================================
                # FIREBASE UPDATE
                # =====================================

                # Lock tetap ditahan selama PUT.
                # Dengan demikian, request frame baru
                # tidak dapat masuk di tengah proses
                # update Firebase.

                res_fb = session.put(

                    FIREBASE_URL,

                    json=payload,

                    timeout=10

                )

                res_fb.raise_for_status()

                print(
                    "[WORKER] Firebase:",
                    res_fb.status_code
                )

                print(
                    "[WORKER] Firebase updated with frame:",
                    frame_id
                )

            # =========================================
            # CLEANUP
            # =========================================

            del results
            del img
            del npimg

            gc.collect()

        except Exception as e:

            print(
                "[WORKER] ERROR:",
                str(e)
            )

        finally:

            # Selalu tandai tugas selesai
            job_queue.task_done()


# =====================================================
# AUTO SLEEP
# =====================================================

def auto_sleep():

    global model
    global last_request_time

    while True:

        try:

            if model is not None:

                idle_time = (
                    time.time() - last_request_time
                )

                if idle_time > MODEL_TIMEOUT:

                    # Pastikan tidak ada gambar menunggu
                    if job_queue.empty():

                        print(
                            "\n[WORKER] No activity."
                            " Checking model unload..."
                        )

                        with model_lock:

                            # Cek ulang setelah memperoleh lock
                            idle_time = (
                                time.time() -
                                last_request_time
                            )

                            if (
                                idle_time > MODEL_TIMEOUT
                                and job_queue.empty()
                            ):

                                model = None

                                gc.collect()

                                print(
                                    "[AUTO_SLEEP] "
                                    "Model unloaded."
                                )

        except Exception as e:

            print(
                "[AUTO_SLEEP] ERROR:",
                str(e)
            )

        time.sleep(10)


# =====================================================
# START THREADS
# =====================================================

# Auto sleep thread
threading.Thread(

    target=auto_sleep,

    daemon=True

).start()


# Single YOLO worker
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
    global latest_frame_id

    try:

        # =============================================
        # RECEIVE IMAGE
        # =============================================

        image_bytes = request.data

        # =============================================
        # VALIDASI IMAGE
        # =============================================

        if not image_bytes:

            return jsonify({

                "error": "no image"

            }), 400

        if len(image_bytes) < 1000:

            return jsonify({

                "error": "invalid image size"

            }), 400

        # =============================================
        # LATEST FRAME PROCESSING
        # =============================================

        with frame_lock:

            # Increment frame ID
            latest_frame_id += 1

            frame_id = latest_frame_id

            # =========================================
            # REMOVE OLD WAITING FRAME
            # =========================================

            if job_queue.full():

                try:

                    old_frame_id, old_image = (
                        job_queue.get_nowait()
                    )

                    job_queue.task_done()

                    print(
                        "[SERVER] Discarded old frame:",
                        old_frame_id
                    )

                    del old_image

                except queue.Empty:

                    pass

            # =========================================
            # INSERT LATEST FRAME
            # =========================================

            job_queue.put_nowait(

                (frame_id, image_bytes)

            )

            # Update activity time
            last_request_time = time.time()

        print(
            "[SERVER] Latest frame accepted:",
            frame_id
        )

        print(
            "[SERVER] Queue size:",
            job_queue.qsize()
        )

        # =============================================
        # RESPONSE
        # =============================================

        return jsonify({

            "status": "queued",

            "frame_id": frame_id,

            "message": "Latest frame accepted"

        }), 200

    except Exception as e:

        print(
            "[SERVER] ERROR:",
            str(e)
        )

        return jsonify({

            "error": str(e)

        }), 500


# =====================================================
# HEALTH CHECK
# =====================================================

@app.route('/')
def home():

    return jsonify({

        "status": "YOLO backend running",

        "queue_strategy": "latest_frame",

        "queue_size": job_queue.maxsize,

        "worker_count": 1,

        "model_timeout": MODEL_TIMEOUT

    })


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False,

        threaded=True

    )
