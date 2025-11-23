# detector.py - THREADING VERSION (Fixed for GPU conflicts)
import os
import time
import json
import csv
import cv2
from datetime import datetime
import threading
from queue import Queue


# Config
POSTURE_MODEL_PATH = "posture_detection/Sitting-Posture-Detection-YOLOv5/data/inference_models/small640.pt"
EMOTION_MODEL_PATH = "Emotion-detection/src/model.h5"
CAM_INDEX = 0
CAPTURE_DIR = "captures"
SHARED_DIR = "shared"
LATEST_IMG = os.path.join(SHARED_DIR, "latest.jpg")
CSV_PATH = os.path.join(CAPTURE_DIR, "detections.csv")
JSONL_PATH = os.path.join(CAPTURE_DIR, "detections.jsonl")
AUTO_CAPTURE_COOLDOWN = 2.0

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(SHARED_DIR, exist_ok=True)

# csv header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "image", "posture_count", "posture_labels", "emotion_count", "emotion_labels"])


def save_capture(image_bgr, posture_list, emotion_list):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_name = f"capture_{ts}.png"
    img_path = os.path.join(CAPTURE_DIR, img_name)
    cv2.imwrite(img_path, image_bgr)
    
    posture_labels = []
    for p in (posture_list or []):
        lbl = p.get("name") or p.get("label") or str(p.get("class", "posture"))
        conf = float(p.get("confidence", 0.0) or 0.0)
        posture_labels.append(f"{lbl}:{conf:.2f}")
    
    emotion_labels = []
    for e in (emotion_list or []):
        lbl = e.get("label", "Unknown")
        conf = float(e.get("confidence", 0.0) or 0.0)
        emotion_labels.append(f"{lbl}:{conf:.1f}%")
    
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, img_name, len(posture_list or []), ";".join(posture_labels), 
                        len(emotion_list or []), ";".join(emotion_labels)])
    
    rec = {"timestamp": ts, "image": img_name, "posture": posture_list or [], "emotion": emotion_list or []}
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    
    print(f"[detector] saved {img_name} posture:{len(posture_list or [])} emotion:{len(emotion_list or [])}")


class CameraThread(threading.Thread):
    """Thread-safe camera capture"""
    def __init__(self, frame_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.stop_flag = threading.Event()
    
    def run(self):
        print("[camera] Starting camera capture...")
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"[camera] CAP_DSHOW failed, trying default...")
            cap = cv2.VideoCapture(CAM_INDEX)
        
        if not cap.isOpened():
            print(f"[camera] ERROR: Cannot open camera {CAM_INDEX}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("[camera] Warming up camera...")
        time.sleep(1.5)
        
        # Discard first few frames
        for _ in range(5):
            cap.read()
            time.sleep(0.1)
        
        print("[camera] Camera ready!")
        
        frame_count = 0
        while not self.stop_flag.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[camera] Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Skip initial black frames
            if frame_count < 10 and frame.mean() < 5:
                frame_count += 1
                time.sleep(0.1)
                continue
            
            # Put frame in queue (non-blocking)
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            
            time.sleep(0.01)
        
        cap.release()
        print("[camera] Camera released")
    
    def stop(self):
        self.stop_flag.set()

class PostureThread(threading.Thread):
    """Thread for accurate posture detection using MediaPipe + ergonomic angles"""

    def __init__(self, frame_queue, result_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_flag = threading.Event()

    def run(self):
        print("[posture] Loading MediaPipe Pose...")

        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print("[posture] Failed to load MediaPipe:", e)
            return

        import math

        def angle_3pt(a, b, c):
            ax, ay = a
            bx, by = b
            cx, cy = c
            ba = (ax - bx, ay - by)
            bc = (cx - bx, cy - by)
            dot = ba[0]*bc[0] + ba[1]*bc[1]
            mag1 = math.sqrt(ba[0]**2 + ba[1]**2)
            mag2 = math.sqrt(bc[0]**2 + bc[1]**2)
            if mag1 * mag2 == 0:
                return 0
            return math.degrees(math.acos(dot / (mag1 * mag2)))

        def vdiff(a, b):
            return abs(a[1] - b[1])

        def hdiff(a, b):
            return abs(a[0] - b[0])

        # Balanced thresholds
        NECK_THRESHOLD = 135
        SHOULDER_TILT_THRESHOLD = 45
        HEAD_TILT_THRESHOLD = 40
        BACK_SLANT_THRESHOLD = 60
        HEAD_DROP_THRESHOLD = 35  # chin-to-chest detection

        print("[posture] Model ready.")

        # Main loop
        while not self.stop_flag.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            detections = []

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Key Points
                left_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                           int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))

                right_sh = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                            int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))

                left_ear = (int(lm[mp_pose.PoseLandmark.LEFT_EAR].x * w),
                            int(lm[mp_pose.PoseLandmark.LEFT_EAR].y * h))

                right_ear = (int(lm[mp_pose.PoseLandmark.RIGHT_EAR].x * w),
                             int(lm[mp_pose.PoseLandmark.RIGHT_EAR].y * h))

                left_hip = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                            int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h))

                nose_point = (int(lm[mp_pose.PoseLandmark.NOSE].x * w),
                              int(lm[mp_pose.PoseLandmark.NOSE].y * h))

                # Metrics
                neck_angle = angle_3pt(left_ear, left_sh, left_hip)
                shoulder_tilt = vdiff(left_sh, right_sh)
                head_tilt = vdiff(left_ear, right_ear)
                back_slant = hdiff(left_sh, left_hip)
                head_drop = nose_point[1] - left_ear[1]  # chin-to-chest

                # Posture Decision
                good = True

                if neck_angle < NECK_THRESHOLD:
                    good = False

                if shoulder_tilt > SHOULDER_TILT_THRESHOLD:
                    good = False

                if head_tilt > HEAD_TILT_THRESHOLD:
                    good = False

                if back_slant > BACK_SLANT_THRESHOLD:
                    good = False

                if head_drop > HEAD_DROP_THRESHOLD:
                    good = False

                label = "Correct Posture" if good else "Incorrect Posture"
                conf = 0.90 if good else 0.75

                # Bounding Box (upper body)
                xs = [left_sh[0], right_sh[0], left_hip[0]]
                ys = [left_sh[1], right_sh[1], left_hip[1], left_ear[1]]

                xmin = max(0, min(xs) - 20)
                xmax = min(w, max(xs) + 20)
                ymin = max(0, min(ys) - 20)
                ymax = min(h, max(ys) + 20)

                detections.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "name": label,
                    "confidence": conf
                })

            # Return result
            self.result_queue.put(("posture", detections, frame))

        pose.close()
        print("[posture] Stopped.")

    def stop(self):
        self.stop_flag.set()




class EmotionThread(threading.Thread):
    """Thread for emotion detection - Uses GPU (TensorFlow gets priority)"""
    def __init__(self, frame_queue, result_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_flag = threading.Event()
        self.model = None
    
    def run(self):
        print("[emotion] Loading model...")
        
        # Configure TensorFlow to use limited GPU memory
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Limit GPU memory to 2GB for TensorFlow
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
                    )
                    print("[emotion] TensorFlow GPU memory limited to 2GB")
                except RuntimeError as e:
                    print(f"[emotion] GPU config error: {e}")
        except:
            pass
        
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(EMOTION_MODEL_PATH, compile=False)
            print("[emotion] Model loaded successfully")
        except:
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
                
                def build_model():
                    m = Sequential([
                        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
                        Conv2D(64, (3,3), activation='relu'),
                        MaxPooling2D((2,2)),
                        Dropout(0.25),
                        Conv2D(128, (3,3), activation='relu'),
                        MaxPooling2D((2,2)),
                        Conv2D(128, (3,3), activation='relu'),
                        MaxPooling2D((2,2)),
                        Dropout(0.25),
                        Flatten(),
                        Dense(1024, activation='relu'),
                        Dropout(0.5),
                        Dense(7, activation='softmax')
                    ])
                    return m
                
                self.model = build_model()
                self.model.load_weights(EMOTION_MODEL_PATH)
                print("[emotion] Model built and weights loaded")
            except Exception as e:
                print(f"[emotion] Model loading failed: {e}")
                # Continue without model
                while not self.stop_flag.is_set():
                    try:
                        frame = self.frame_queue.get(timeout=1.0)
                        self.result_queue.put(("emotion", [], frame))
                    except:
                        continue
                return
        
        emotion_labels = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while not self.stop_flag.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            emotions = []
            
            for (x,y,w,h) in faces:
                try:
                    face = cv2.resize(gray[y:y+h, x:x+w], (48,48))
                    face = face.astype("float32") / 255.0
                    face = face.reshape((1,48,48,1))
                    
                    preds = self.model.predict(face, verbose=0)[0]
                    label_idx = int(preds.argmax())
                    emotions.append({
                        "box": (int(x),int(y),int(w),int(h)),
                        "label_idx": label_idx,
                        "label": emotion_labels.get(label_idx, "Unknown"),
                        "confidence": float(preds.max())*100.0
                    })
                except Exception as e:
                    print(f"[emotion] Face processing error: {e}")
            
            self.result_queue.put(("emotion", emotions, frame))
    
    def stop(self):
        self.stop_flag.set()


def main():
    print("[detector] Starting pipeline...")
    
    frame_q = Queue(maxsize=4)
    result_q = Queue(maxsize=8)
    
    # Start all threads
    camera_thread = CameraThread(frame_q)
    posture_thread = PostureThread(frame_q, result_q)
    emotion_thread = EmotionThread(frame_q, result_q)
    
    camera_thread.start()
    time.sleep(2)  # Let camera initialize first
    posture_thread.start()
    emotion_thread.start()
    
    # Renderer loop (runs in main thread)
    last_posture = None
    last_emotion = None
    last_auto_capture = 0.0
    
    EMOTION_COLORS = {
        "Angry": (0,0,255), "Disgusted": (128,0,128), "Fearful": (255,0,255),
        "Happy": (0,255,0), "Neutral": (255,255,255), "Sad": (255,255,0), "Surprised": (0,255,255)
    }
    
    print("[detector] Pipeline running. Press Ctrl+C to stop...")
    
    try:
        while True:
            try:
                kind, data, frame = result_q.get(timeout=1.0)
            except:
                continue
            
            if kind == "posture":
                last_posture = data
            elif kind == "emotion":
                last_emotion = data
            
            # Draw annotations
            out = frame.copy()
            
            # Draw posture boxes (green)
            if last_posture:
                for d in last_posture:
                    try:
                        x1, y1 = int(d["xmin"]), int(d["ymin"])
                        x2, y2 = int(d["xmax"]), int(d["ymax"])
                        label = d.get("name", "posture")
                        conf = float(d.get("confidence", 0.0))
                        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(0,y1-6)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    except:
                        pass
            
            # Draw emotion boxes
            if last_emotion:
                for e in last_emotion:
                    try:
                        x, y, w, h = e["box"]
                        label = e.get("label", "Unknown")
                        conf = e.get("confidence", 0.0)
                        color = EMOTION_COLORS.get(label, (255,0,0))
                        cv2.rectangle(out, (x,y), (x+w,y+h), color, 2)
                        cv2.putText(out, f"{label} ({conf:.1f}%)", (x, max(0,y-6)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except:
                        pass
            
            # Save latest.jpg
            try:
                cv2.imwrite(LATEST_IMG, out)
            except Exception as e:
                print(f"[detector] Error writing latest.jpg: {e}")
            
            # Auto-capture
            now = time.time()
            has_detection = (last_posture and len(last_posture) > 0) or (last_emotion and len(last_emotion) > 0)
            if has_detection and (now - last_auto_capture) > AUTO_CAPTURE_COOLDOWN:
                try:
                    save_capture(out, last_posture or [], last_emotion or [])
                    last_auto_capture = now
                except Exception as e:
                    print(f"[detector] Capture error: {e}")
    
    except KeyboardInterrupt:
        print("\n[detector] Stopping...")
    finally:
        camera_thread.stop()
        posture_thread.stop()
        emotion_thread.stop()
        print("[detector] Stopped.")


if __name__ == "__main__":
    main()
