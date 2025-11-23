import os
import time
import threading
from collections import deque
from datetime import datetime

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from config import Config

try:
    import easyocr
except Exception:
    easyocr = None


class Detector:
    def __init__(self, mongo_collection, model_path=None):

        # Auto-select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        self.model = YOLO(model_path or Config.YOLO_MODEL_PATH)
        self.model.overrides['conf'] = Config.CONF_THRESHOLD
        self.model.overrides['iou'] = Config.IOU_THRESHOLD
        self.model.overrides['max_det'] = 500
        self.model.overrides['classes'] = 2, 5, 7 # Define specific class for detection

        self.mongo = mongo_collection
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.last_positions = {}
        self.tracking_trajectories = {}
        self.fps = Config.DEFAULT_FPS

        # OCR setup
        self.ocr = easyocr.Reader(['en']) if Config.ENABLE_PLATE_OCR and easyocr is not None else None

    def estimate_speed(self, obj_id, centroid, timestamp):
        if obj_id not in self.last_positions:
            self.last_positions[obj_id] = (centroid, timestamp)
            return 0.0
        prev_centroid, prev_ts = self.last_positions[obj_id]
        dt = max(1e-6, timestamp - prev_ts)
        dx, dy = centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1]
        dist_m = ((dx ** 2 + dy ** 2) ** 0.5) * Config.M_PER_PIXEL
        kmph = (dist_m / dt) * 3.6
        self.last_positions[obj_id] = (centroid, timestamp)
        return round(kmph, 2)

    def ocr_plate(self, frame, bbox):
        if not self.ocr:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        try:
            results = self.ocr.readtext(crop)
            if results:
                return " ".join([r[1] for r in results]).strip()
        except Exception:
            pass
        return None



    def process_video(self, video_path, output_path):


        def draw_text_with_bg(img, text, pos, font_scale=0.6, text_color=(255, 255, 255),
                              bg_color=(0, 0, 0), alpha=0.5, thickness=3):
            """Draw text with semi-transparent background."""
            overlay = img.copy()
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x, y = pos
            y = max(y, text_h + 5)
            cv2.rectangle(overlay, (x, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            return img

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            self.running = False
            return

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_FPS
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print(f"[INFO] Processing started: {video_path}")
        frame_count = 0
        last_time = time.time()

        # Tracking data
        self.tracks = {}
        id_counter = 0

        # Lane coordinates
        lanes = {
            "Lane1": ((2899, 1307), (3414, 1644)),
            "Lane2": ((1670, 1171), (2121, 1215)),
            "Lane3": ((383, 1390), (1441, 1418)),
        }

        lane_counts = {name: 0 for name in lanes}
        total_count = 0
        counted_ids = set()

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video or read error.")
                break
            frame_count += 1

            # FPS log
            now = time.time()
            if now - last_time >= 1.0:
                print(f"[INFO] FPS: {frame_count}")
                frame_count = 0
                last_time = now

            # Draw lane lines and labels
            for name, (p1, p2) in lanes.items():
                cv2.line(frame, p1, p2, (255, 255, 0), 3)
                frame = draw_text_with_bg(frame, name, (p1[0], p1[1] - 10), font_scale=0.8)

            # YOLO inference (auto GPU if available)
            try:
                results = self.model(frame, verbose=False, device=self.device)
            except Exception as e:
                print(f"[WARN] YOLO failed: {e}")
                continue

            detections = results[0]
            current_positions = []

            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = detections.names.get(cls_id, str(cls_id))

                if cls_name not in ["car", "bus", "truck", "motorbike"]:
                    continue

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                current_positions.append((cx, cy, cls_name, conf, (x1, y1, x2, y2)))

            # Simple tracking
            updated_tracks = {}
            for cx, cy, cls_name, conf, bbox in current_positions:
                min_dist = float("inf")
                matched_id = None
                for tid, data in self.tracks.items():
                    px, py = data["pos"]
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if dist < 50 and dist < min_dist:
                        matched_id = tid
                        min_dist = dist

                if matched_id is None:
                    id_counter += 1
                    matched_id = id_counter
                    updated_tracks[matched_id] = {"pos": (cx, cy), "traj": [(cx, cy)], "speed": 0.0}
                else:
                    prev = self.tracks[matched_id]
                    prev_pos = prev["pos"]
                    prev_traj = prev["traj"]
                    dx, dy = cx - prev_pos[0], cy - prev_pos[1]
                    dist_px = (dx ** 2 + dy ** 2) ** 0.5
                    speed = (dist_px * fps) * 0.036

                    updated_tracks[matched_id] = {
                        "pos": (cx, cy),
                        "traj": (prev_traj + [(cx, cy)])[-30:],
                        "speed": speed
                    }

                # Draw box + info
                x1, y1, x2, y2 = bbox
                label = f"ID:{matched_id} {cls_name} {conf:.2f} {updated_tracks[matched_id]['speed']:.1f}km/h"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                frame = draw_text_with_bg(frame, label, (x1, max(y1 - 10, 20)))

                # Draw trajectory
                pts = updated_tracks[matched_id]["traj"]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), 2)

                # Lane crossing detection
                for lname, (p1, p2) in lanes.items():
                    dist = abs((p2[1] - p1[1]) * cx - (p2[0] - p1[0]) * cy + p2[0]*p1[1] - p2[1]*p1[0]) / (
                            ((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) ** 0.5)
                    if dist < 15 and matched_id not in counted_ids:
                        lane_counts[lname] += 1
                        total_count += 1
                        counted_ids.add(matched_id)
                        print(f"[COUNT] Vehicle {matched_id} crossed {lname}")

            self.tracks = updated_tracks

            # Display total counts
            y0 = 40
            frame = draw_text_with_bg(frame, f"Total Vehicles: {total_count}", (20, y0), font_scale=1.5)
            for i, (lname, count) in enumerate(lane_counts.items()):
                frame = draw_text_with_bg(frame, f"{lname}: {count}", (20, y0 + 40 * (i + 1)), font_scale=1.0)

            out.write(frame)
            self.latest_frame = cv2.imencode('.jpg', frame)[1].tobytes()
            time.sleep(0.005)

        cap.release()
        out.release()
        self.running = False
        print("[INFO] Processing finished.")




    def start(self, video_path, output_path):
        if self.running:
            print("[WARN] Detector already running.")
            return False
        self.running = True
        self.thread = threading.Thread(target=self.process_video, args=(video_path, output_path), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        if not self.running:
            return False
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        print("[INFO] Detector stopped.")
        return True
