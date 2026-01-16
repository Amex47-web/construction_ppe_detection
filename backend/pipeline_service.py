import cv2 as cv
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
from collections import Counter
import time
import os
import threading
from database.database import log_violation, register_new_worker, find_matching_worker

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, "storage", "faces")
ALERTS_DIR = os.path.join(BASE_DIR, "storage", "alerts")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)

class PPEPipeline:
    def __init__(self):
        print("[INFO] Initializing PPE Pipeline...")
        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names
        print(f"[INFO] Loaded classes: {self.class_names}")
        
        # Configuration
        self.min_face_size = 40
        self.sharpness_threshold = 80
        self.vote_limit = 5
        self.wait_for_face_limit = 25
        self.model_name = "Facenet"
        
        # State
        # State
        self.cap = None
        self.identity_manager = {}
        self.frames_count = 0
        self.source_path = None
        self.session_start_time = time.time()
        
        # Statistics State
        self.current_stats = {
            "total_workers": 0,
            "helmet_count": 0,
            "vest_count": 0,
            "mask_count": 0,
            "violations_today": 0 
        }

    def set_source(self, video_path):
        """Sets the video source and resets state if needed."""
        self.source_path = video_path
        if self.cap:
            self.cap.release()
        self.cap = cv.VideoCapture(video_path)
        
        # Reset Session State
        self.identity_manager = {} 
        self.frames_count = 0
        self.session_start_time = time.time() # Track when this video started
        self.current_stats = {
            "total_workers": 0,
            "helmet_count": 0,
            "vest_count": 0,
            "mask_count": 0,
            "violations_today": 0 
        }
        print(f"[INFO] Video source set to: {video_path} at {self.session_start_time}")
        print(f"[INFO] Video source set to: {video_path}")

    def get_stats(self):
        """Returns current statistics."""
        return self.current_stats

    def _extract_appearance_embedding(self, person_img):
        """Extracts a color histogram as a simple appearance embedding."""
        try:
            if person_img.size == 0: return None
            resized = cv.resize(person_img, (64, 128)) 
            hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
            hist = cv.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv.normalize(hist, hist)
            return hist.flatten()
        except Exception as e:
            print(f"[ERROR] Appearance extraction failed: {e}")
            return None

    def _is_clear_face(self, face_img):
        if face_img.shape[0] < self.min_face_size or face_img.shape[1] < self.min_face_size:
            return False
        gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        score = cv.Laplacian(gray, cv.CV_64F).var()
        return score > self.sharpness_threshold

    def generate_frames(self):
        """Yields MJPEG frames."""
        if not self.cap or not self.cap.isOpened():
             # If no video is selected, return a blank frame or waiting image
             blank = np.zeros((360, 640, 3), dtype=np.uint8)
             cv.putText(blank, "Waiting for video...", (50, 180), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             ret, buffer = cv.imencode('.jpg', blank)
             frame_bytes = buffer.tobytes()
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
             return

        while True:
            success, frame = self.cap.read()
            if not success:
                # Video finished, maybe loop or stop ?
                # For now, loop it
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.frames_count += 1
            if self.frames_count % 2 == 0: # Skip every other frame for performance
                continue
                
            # --- PROCESS FRAME (Logic from main.py) ---
            processed_frame = self._process_frame(frame)
            
            # Compress to JPG
            ret, buffer = cv.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def _process_frame(self, frame):
        results = self.model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        
        helmet_c = 0
        vest_c = 0
        mask_c = 0
        
        if results[0].boxes is not None:
            persons = []
            equipment = []
            
            # Separate Detections
            for box in results[0].boxes:
                cls = int(box.cls[0])
                tid = int(box.id[0]) if box.id is not None else None
                coords = box.xyxy[0].cpu().numpy().astype(int)
                
                if cls == 5: # Person
                    persons.append({"tid": tid, "box": coords})
                elif cls in [0, 1, 6, 7]: # PPE Classes (Adjust indices if needed)
                    # Check class names: ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']
                    # Using indices from original main.py logic: cls=5 is Person.
                    equipment.append({"cls": cls, "box": coords})
                    
                    # Updates stats (Simple count for now)
                    if self.class_names[cls] == 'Hardhat': helmet_c += 1
                    if self.class_names[cls] == 'Mask': mask_c += 1
                    if self.class_names[cls] == 'Safety Vest': vest_c += 1


            for p in persons:
                px1, py1, px2, py2 = p["box"]
                tid = p["tid"]
                
                # Setup Manager for Person
                if tid not in self.identity_manager:
                    self.identity_manager[tid] = {
                        "votes": [], 
                        "final_uuid": None, 
                        "frame_count": 0, 
                        "has_logged_violation": False
                    }
                
                mgr = self.identity_manager[tid]
                mgr["frame_count"] += 1
                
                # Crop Person
                person_crop = frame[max(0, py1):min(frame.shape[0], py2), 
                                    max(0, px1):min(frame.shape[1], px2)]
                                    
                # IDENTITY IDENTIFICATION logic
                if mgr["final_uuid"] is None:
                    # A. Face
                    if self._is_clear_face(person_crop):
                        try:
                            objs = DeepFace.represent(img_path=person_crop, model_name=self.model_name, 
                                                    enforce_detection=True, detector_backend='opencv')
                            if objs:
                                encoding = np.array(objs[0]["embedding"])
                                matched_uuid = find_matching_worker(encoding, threshold=0.45)
                                
                                if matched_uuid:
                                    mgr["votes"].append(matched_uuid)
                                    if len(mgr["votes"]) >= self.vote_limit:
                                        real_uuid = Counter(mgr["votes"]).most_common(1)[0][0]
                                        mgr["final_uuid"] = real_uuid
                                else:
                                    # Register New
                                    face_fn = f"face_{tid}_{int(time.time())}.jpg"
                                    face_path = os.path.join(FACES_DIR, face_fn)
                                    cv.imwrite(face_path, person_crop)
                                    new_id = register_new_worker(encoding, face_fn) # passing filename as display name for now
                                    mgr["final_uuid"] = new_id
                        except Exception:
                            pass
                    
                    # B. Appearance Fallback
                    elif mgr["frame_count"] > self.wait_for_face_limit:
                        app_emb = self._extract_appearance_embedding(person_crop)
                        if app_emb is not None:
                            snap_fn = f"appearance_{tid}_{int(time.time())}.jpg"
                            snap_path = os.path.join(FACES_DIR, snap_fn)
                            cv.imwrite(snap_path, person_crop)
                            new_id = register_new_worker(app_emb, snap_fn)
                            mgr["final_uuid"] = new_id

                # PPE ASSOCIATION
                equipped_list = []
                for e in equipment:
                    ex1, ey1, ex2, ey2 = e["box"]
                    cx, cy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
                    if px1 < cx < px2 and py1 < cy < py2:
                        equipped_list.append(self.class_names[e['cls']])
                
                required_ppe = ['Hardhat', 'Mask', 'Safety Vest']
                missing_ppe = [item for item in required_ppe if item not in equipped_list]

                # LOGGING VIOLATION
                if missing_ppe and mgr["final_uuid"] is not None:
                    if not mgr["has_logged_violation"]:
                        alert_fn = f"violation_{tid}_{int(time.time())}.jpg"
                        alert_path = os.path.join(ALERTS_DIR, alert_fn)
                        
                        try:
                            log_violation(
                                worker_uuid=str(mgr["final_uuid"]), 
                                equipped=", ".join(equipped_list), 
                                violated=", ".join(missing_ppe),
                                evidence_path=alert_path
                            )
                            cv.imwrite(alert_path, person_crop)
                            mgr["has_logged_violation"] = True
                            self.current_stats["violations_today"] += 1
                        except Exception as e:
                            print(f"Error logging violation: {e}")

                # VISUALIZATION
                if len(missing_ppe) == 3: color = (0, 0, 255)
                elif len(missing_ppe) > 0: color = (0, 255, 255)
                else: color = (0, 255, 0)
                
                cv.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                label = f"ID: {mgr['final_uuid'] or 'Scanning'}"
                cv.putText(frame, label, (px1, py1 - 10), 0, 0.6, color, 2)

        # Update Stats Snapshot
        self.current_stats["helmet_count"] = helmet_c
        self.current_stats["vest_count"] = vest_c
        self.current_stats["mask_count"] = mask_c
        
        # 'active_workers' = number of people currently detected in this frame
        # This is more stable and 'live' than accumulating unique IDs which grows indefinitely
        self.current_stats["total_workers"] = len(persons)
        
        return frame
