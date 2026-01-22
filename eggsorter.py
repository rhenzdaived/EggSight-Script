# -*- coding: utf-8 -*-

import os
import time
import threading
import numpy as np
import cv2
import RPi.GPIO as GPIO
import pigpio
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
from datetime import datetime
from zoneinfo import ZoneInfo
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import Increment
from google.api_core.exceptions import NotFound, Aborted, DeadlineExceeded

FIREBASE_KEY = "/home/pi/eggsight/firebase-eggsight.json"
EGGS_COLLECTION = "eggs"
SUMMARY_COLLECTION = "daily_summary"

LOCAL_TZ = ZoneInfo("Asia/Manila") 

def init_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEY)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def _today_key():
    return datetime.now(LOCAL_TZ).date().isoformat()

def _ensure_daily_summary(db, date_key: str):
    ref = db.collection(SUMMARY_COLLECTION).document(date_key)
    try:
        snap = ref.get()
        if not snap.exists:
            ref.set(
                {
                    "date": date_key,
                    "fresh_count": 0,
                    "rotten_count": 0,
                    "total_count": 0,
                    "last_update": firestore.SERVER_TIMESTAMP,
                }
            )
    except Exception:
        # Fallback: create with merge
        ref.set(
            {
                "date": date_key,
                "fresh_count": 0,
                "rotten_count": 0,
                "total_count": 0,
                "last_update": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
    return ref

def _inc_with_readback(ref, field: str):
    try:
        ref.update({field: Increment(1), "last_update": firestore.SERVER_TIMESTAMP})
    except NotFound:
        # If somehow missing, create then increment
        ref.set({field: 0, "last_update": firestore.SERVER_TIMESTAMP}, merge=True)
        ref.update({field: Increment(1), "last_update": firestore.SERVER_TIMESTAMP})
    except (Aborted, DeadlineExceeded):
        # Rare transient issues on Pi/network; retry once quickly
        time.sleep(0.2)
        ref.update({field: Increment(1), "last_update": firestore.SERVER_TIMESTAMP})

    # Readback (debug)
    try:
        snap = ref.get()
        print(f"[FS] {field} now = {snap.get(field)}")
    except Exception:
        print("[FS] readback failed")


def log_total_on_detect(db):
    date_key = _today_key()
    print(f"[FS] using date_key={date_key} (total++)")
    ref = _ensure_daily_summary(db, date_key)
    _inc_with_readback(ref, "total_count")


def log_classification(db, status: str, conf_prob_0to1: float):
    status = status.lower().strip()

    try:
        db.collection(EGGS_COLLECTION).add(
            {
                "timestamp": firestore.SERVER_TIMESTAMP,
                "status": status,
                "confidence": round(float(conf_prob_0to1) * 100, 2),
                "date_key": _today_key(),
            }
        )
        print("[FS] egg log added")
    except Exception as e:
        print(f"[FS] egg log failed: {e}")

    #Increment per-day counter
    try:
        date_key = _today_key()
        print(f"[FS] using date_key={date_key} ({status}++)")
        ref = _ensure_daily_summary(db, date_key)
        field = "fresh_count" if status == "fresh" else "rotten_count"
        _inc_with_readback(ref, field)
    except Exception as e:
        print(f"[FS] classification increment failed: {e}")
        
def async_log_total(db):
    try:
        threading.Thread(target=log_total_on_detect, args=(db,), daemon=True).start()
    except Exception as e:
        print(f"[FS] async spawn failed: {e}")

MODEL = "/home/pi/eggsight/models/eggsight_mnv2_tf214.tflite"
LABELS_FILE = "/home/pi/eggsight/models/labels.txt"
IMG_SIZE = (224, 224)
THRESHOLD = 0.5
DELAY_BEFORE_STOP = 1.1
SETTLE_TIME = 1.5

#IR
GPIO.setmode(GPIO.BCM)
IR_GPIO = 22
GPIO.setup(IR_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#SG90
SERVO_GPIO = 23
MIN_PW, MAX_PW = 500, 2500
pi = pigpio.pi()
if not pi.connected:
    print("Error: pigpio daemon not running. Start with: sudo systemctl start pigpiod")
    raise SystemExit(1)
pi.set_mode(SERVO_GPIO, pigpio.OUTPUT)


def angle_to_pulsewidth(angle):
    return int(MIN_PW + (angle / 180.0) * (MAX_PW - MIN_PW))

# VNH2SP30
INA, INB, PWM_PIN = 17, 27, 18
GPIO.setup(INA, GPIO.OUT)
GPIO.setup(INB, GPIO.OUT)
GPIO.setup(PWM_PIN, GPIO.OUT)

def motor_forward():
    GPIO.output(INA, 1)
    GPIO.output(INB, 0)
    GPIO.output(PWM_PIN, 1)

def motor_stop():
    GPIO.output(INA, 0)
    GPIO.output(INB, 0)
    GPIO.output(PWM_PIN, 0)


if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE) as f:
        labels = [ln.strip() for ln in f if ln.strip()]
else:
    labels = ["fresh", "rotten"]

interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
binary_output = (output_details[0]["shape"][-1] == 1)


def preprocess(frame_rgb):
    img = cv2.resize(
        frame_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()


for _ in range(10):
    _ = picam2.capture_array()

try:
    db = init_firestore()
    try:
        _ensure_daily_summary(db, _today_key())
        print("Firestore connected and daily_summary warmed.")
    except Exception:
        print("Firestore connected.")
except Exception as e:
    db = None
    print(f"Firestore init failed: {e}")

print("System initializing (stabilizing sensors + motor)...")
motor_stop()
time.sleep(0.3)
motor_forward()
time.sleep(2)  # allow motor and IR to stabilize

print("System ready. Conveyor running. Press ESC to quit.")
prev_ir = GPIO.input(IR_GPIO)
last_trigger_ts = 0.0
DEBOUNCE_SECONDS = 0.25

try:
    while True:
        frame_rgb = picam2.capture_array()
        overlay, color = "Waiting for egg...", (255, 255, 0)

        ir_now = GPIO.input(IR_GPIO)

        if prev_ir == 1 and ir_now == 0: 
            now = time.time()
            if now - last_trigger_ts >= DEBOUNCE_SECONDS:
                last_trigger_ts = now
                print("[IR] Edge detected -> Egg present")

                if db is not None:
                    try:
                        async_log_total(db)
                        print("Counted TOTAL (async Firestore increment)")
                    except Exception as e:
                        print(f"total_count async spawn failed: {e}")

                print(f"Egg detected â€” delaying stop for {DELAY_BEFORE_STOP:.1f}s...")
                time.sleep(DELAY_BEFORE_STOP)

                print("Stopping motor...")
                motor_stop()
                print(f"Letting egg settle for {SETTLE_TIME:.1f}s...")
                time.sleep(SETTLE_TIME)

                frame_rgb = picam2.capture_array()

                # ---- Inference ----
                x = preprocess(frame_rgb)
                interpreter.set_tensor(input_index, x)
                interpreter.invoke()
                raw = interpreter.get_tensor(output_index)

                if binary_output:
                    p_rotten = float(raw.squeeze())
                    p_fresh = 1.0 - p_rotten
                    if p_rotten >= THRESHOLD:
                        decision, conf, color = "rotten", p_rotten, (0, 0, 255)
                    else:
                        decision, conf, color = "fresh", p_fresh, (0, 200, 0)
                    label = decision
                else:
                    probs = raw[0]
                    idx = int(np.argmax(probs))
                    label = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
                    conf = float(probs[idx])
                    decision = "fresh" if "fresh" in label.lower() else "rotten"
                    color = (0, 200, 0) if decision == "fresh" else (0, 0, 255)

                overlay = f"{label}: {conf:.1%}"
                print(f"Classification: {overlay}")

                # ---- Log classification ----
                if db is not None:
                    try:
                        log_classification(db, decision, conf)
                    except Exception as e:
                        print(f"âš ï¸ class log failed: {e}")
                else:
                    print("âš ï¸ Skipped Firestore log (db not initialized)")

                # ---- Servo sorting ----
                angle = 45 if decision == "fresh" else 130
                pi.set_servo_pulsewidth(SERVO_GPIO, angle_to_pulsewidth(angle))
                print(f"Servo -> {angle}Â° ({decision.title()} bin)")
                time.sleep(1.5)
                pi.set_servo_pulsewidth(SERVO_GPIO, 0)

                # ---- Resume conveyor ----
                print("Resuming conveyor...")
                motor_forward()

        prev_ir = ir_now

        # ---- Preview ----
        cv2.putText(
            frame_rgb,
            overlay,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("EggSight Live", frame_rgb)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

except KeyboardInterrupt:
    print("Stopping system...")

finally:
    picam2.stop()
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_GPIO, 0)
    pi.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("System shut down.")
